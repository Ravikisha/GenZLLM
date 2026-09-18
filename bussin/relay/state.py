"""Run state and distributed lease.

The relay's whole job is to let one logical training run be carried by a
sequence of short, unreliable, heterogeneous sessions without ever letting two
of them advance the same step. There is no server, so coordination rides on
compare-and-swap against a single JSON file.

On the Hugging Face Hub, `create_commit(parent_commit=...)` is rejected if the
branch has moved -- that is the CAS primitive. A local filesystem backend with
the same semantics exists so the logic is testable offline.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

STATE_FILE = "RUN_STATE.json"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


class LeaseHeld(Exception):
    """Another worker holds a live lease."""


class CASConflict(Exception):
    """The state file moved under us; re-read and retry."""


# ------------------------------------------------------------------ #
# Backends
# ------------------------------------------------------------------ #


class StateBackend(Protocol):
    def read(self) -> tuple[dict[str, Any] | None, str | None]:
        """Returns (state, revision). (None, None) if it does not exist yet."""

    def write(self, state: dict[str, Any], parent_revision: str | None) -> str:
        """Compare-and-swap. Raises CASConflict if parent_revision is stale."""


class LocalBackend:
    """Filesystem backend. Revision is the file's content hash."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / STATE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _rev(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def read(self):
        if not self.path.exists():
            return None, None
        text = self.path.read_text(encoding="utf-8")
        return json.loads(text), self._rev(text)

    def write(self, state, parent_revision):
        _, current = self.read()
        if current != parent_revision:
            raise CASConflict(f"state moved: {parent_revision} -> {current}")
        text = json.dumps(state, indent=2)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)
        return self._rev(text)


class HFBackend:
    """Hugging Face Hub backend using commit-SHA compare-and-swap."""

    def __init__(self, repo_id: str, token: str | None = None,
                 repo_type: str = "model", revision: str = "main") -> None:
        from huggingface_hub import HfApi

        self.api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision

    def read(self):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        try:
            refs = self.api.list_repo_refs(self.repo_id, repo_type=self.repo_type)
            sha = next(
                (b.target_commit for b in refs.branches if b.name == self.revision), None
            )
        except RepositoryNotFoundError:
            return None, None

        try:
            path = hf_hub_download(
                self.repo_id, STATE_FILE, repo_type=self.repo_type,
                revision=self.revision, force_download=True,
            )
        except EntryNotFoundError:
            return None, sha
        return json.loads(Path(path).read_text(encoding="utf-8")), sha

    def write(self, state, parent_revision):
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.utils import HfHubHTTPError

        op = CommitOperationAdd(
            path_in_repo=STATE_FILE,
            path_or_fileobj=json.dumps(state, indent=2).encode(),
        )
        # `lease` is None after release(), so this cannot assume a dict.
        lease = state.get("lease") or {}
        who = lease.get("worker_id", "released")
        try:
            info = self.api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                operations=[op],
                commit_message=f"relay: step {state.get('step')} by {who}",
                parent_commit=parent_revision,
            )
        except HfHubHTTPError as exc:
            # 412 Precondition Failed is the CAS rejection we rely on.
            if "412" in str(exc) or "parent" in str(exc).lower():
                raise CASConflict(str(exc)) from exc
            raise
        return info.oid


# ------------------------------------------------------------------ #
# State
# ------------------------------------------------------------------ #


@dataclass
class Lease:
    worker_id: str
    platform: str
    claimed_at: str
    expires_at: str

    def is_live(self, now: datetime | None = None) -> bool:
        exp = _parse(self.expires_at)
        return exp is not None and exp > (now or _utcnow())


@dataclass
class RunState:
    run_id: str
    step: int = 0
    stage: str = "S1"
    tokens_seen: int = 0
    latest_ckpt: str | None = None
    milestones: list[str] = field(default_factory=list)
    lease: dict[str, Any] | None = None
    lr_scale: float = 1.0
    data_cursor: dict[str, Any] = field(default_factory=lambda: {"shard": 0, "offset": 0, "epoch": 0})
    history: list[dict[str, Any]] = field(default_factory=list)
    config_hash: str | None = None
    finished: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def lease_obj(self) -> Lease | None:
        return Lease(**self.lease) if self.lease else None


class RelayState:
    """Read/claim/update/release the run state."""

    def __init__(self, backend: StateBackend, run_id: str) -> None:
        self.backend = backend
        self.run_id = run_id
        self._revision: str | None = None
        self._state: RunState | None = None

    # -------------------------------------------------------------- #

    def read(self) -> RunState:
        raw, rev = self.backend.read()
        self._revision = rev
        self._state = RunState.from_dict(raw) if raw else RunState(run_id=self.run_id)
        return self._state

    def claim(
        self,
        worker_id: str,
        platform: str,
        lease_seconds: int,
        force: bool = False,
        retries: int = 3,
    ) -> RunState:
        """Take the lease, or raise LeaseHeld.

        `lease_seconds` should be the session limit plus a margin, so a session
        that dies without releasing cannot block the next worker for long.
        """
        for attempt in range(retries):
            state = self.read()

            if state.finished:
                raise LeaseHeld(f"run {state.run_id} is already marked finished")

            existing = state.lease_obj()
            if existing and existing.is_live() and not force:
                raise LeaseHeld(
                    f"lease held by {existing.worker_id} ({existing.platform}) "
                    f"until {existing.expires_at}"
                )

            now = _utcnow()
            state.lease = asdict(
                Lease(
                    worker_id=worker_id,
                    platform=platform,
                    claimed_at=_iso(now),
                    expires_at=_iso(now + timedelta(seconds=lease_seconds)),
                )
            )
            try:
                self._revision = self.backend.write(state.to_dict(), self._revision)
                self._state = state
                return state
            except CASConflict:
                if attempt == retries - 1:
                    raise LeaseHeld("lost the compare-and-swap race for the lease")
                time.sleep(1.0 + attempt * 2)
        raise LeaseHeld("could not claim lease")

    def _commit(self, state: RunState, retries: int = 3) -> None:
        for attempt in range(retries):
            try:
                self._revision = self.backend.write(state.to_dict(), self._revision)
                self._state = state
                return
            except CASConflict:
                if attempt == retries - 1:
                    raise
                # Someone else wrote. Re-read, re-apply our fields, retry.
                fresh = self.read()
                if (fresh.lease or {}).get("worker_id") != (state.lease or {}).get("worker_id"):
                    raise LeaseHeld("lease was taken by another worker mid-run")
                fresh.step = state.step
                fresh.tokens_seen = state.tokens_seen
                fresh.latest_ckpt = state.latest_ckpt
                fresh.data_cursor = state.data_cursor
                fresh.stage = state.stage
                fresh.lease = state.lease
                state = fresh
                time.sleep(1.0 + attempt * 2)

    def heartbeat(self, lease_seconds: int) -> None:
        """Extend the lease. Called every ~10 minutes while training."""
        state = self._state or self.read()
        if not state.lease:
            return
        state.lease["expires_at"] = _iso(_utcnow() + timedelta(seconds=lease_seconds))
        self._commit(state)

    def update(
        self,
        step: int,
        tokens_seen: int,
        latest_ckpt: str | None = None,
        data_cursor: dict[str, Any] | None = None,
        stage: str | None = None,
        milestone: str | None = None,
        lease_seconds: int | None = None,
    ) -> None:
        state = self._state or self.read()
        state.step = step
        state.tokens_seen = tokens_seen
        if latest_ckpt:
            state.latest_ckpt = latest_ckpt
        if data_cursor:
            state.data_cursor = data_cursor
        if stage:
            state.stage = stage
        if milestone and milestone not in state.milestones:
            state.milestones.append(milestone)
        if lease_seconds and state.lease:
            state.lease["expires_at"] = _iso(_utcnow() + timedelta(seconds=lease_seconds))
        self._commit(state)

    def release(self, steps_done: int = 0, wall_seconds: float = 0.0,
                train_seconds: float = 0.0, finished: bool = False) -> None:
        """Drop the lease and record what this session achieved."""
        state = self._state or self.read()
        lease = state.lease or {}
        state.history.append(
            {
                "worker_id": lease.get("worker_id"),
                "platform": lease.get("platform"),
                "claimed_at": lease.get("claimed_at"),
                "released_at": _iso(_utcnow()),
                "steps": steps_done,
                "end_step": state.step,
                "wall_s": round(wall_seconds, 1),
                "train_s": round(train_seconds, 1),
                # The efficiency number worth watching: target >= 0.95 (SPEC §9.4).
                "efficiency": round(train_seconds / wall_seconds, 3) if wall_seconds else None,
            }
        )
        state.lease = None
        if finished:
            state.finished = True
        self._commit(state)

    def request_rollback(self, milestone: str, lr_scale: float = 0.5) -> None:
        """Record a divergence recovery so the next worker resumes correctly."""
        state = self._state or self.read()
        state.latest_ckpt = milestone
        state.lr_scale = lr_scale
        self._commit(state)


def make_backend(uri: str, token: str | None = None) -> StateBackend:
    """`hf://org/repo` or a local path."""
    if uri.startswith("hf://"):
        return HFBackend(uri[len("hf://"):], token=token)
    return LocalBackend(uri)
