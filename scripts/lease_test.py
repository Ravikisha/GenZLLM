"""Lease and compare-and-swap tests.

The relay's safety property is that two workers can never advance the same
step. There is no coordination server -- it rides entirely on compare-and-swap
against a single JSON file. These tests exercise the cases that actually
happen: a second worker arriving while the first is live, a worker whose
session died without releasing, and two workers racing for the same free lease.

Run:  python scripts/lease_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.relay.state import (
    CASConflict,
    LeaseHeld,
    LocalBackend,
    RelayState,
    RunState,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  {PASS if ok else FAIL}  {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bussin_lease_"))
    try:
        print("=== 1. first worker claims a fresh run ===")
        a = RelayState(LocalBackend(tmp), "run-1")
        st = a.claim("worker-A", "kaggle-gpu", lease_seconds=600)
        check("fresh claim succeeds", st.lease is not None)
        check("step starts at 0", st.step == 0)

        print("\n=== 2. second worker is refused while the lease is live ===")
        b = RelayState(LocalBackend(tmp), "run-1")
        refused = False
        try:
            b.claim("worker-B", "colab", lease_seconds=600)
        except LeaseHeld as exc:
            refused = True
            detail = str(exc)
        check("live lease blocks a second worker", refused, detail if refused else "")

        print("\n=== 3. worker A makes progress ===")
        a.update(step=500, tokens_seen=500 * 262_144, latest_ckpt="ckpt-00000500",
                 data_cursor={"shard": 2, "offset": 4096, "epoch": 0},
                 milestone="ckpt-00000500")
        st = b.read()
        check("progress is visible to other workers", st.step == 500, f"step={st.step}")
        check("milestone recorded", st.milestones == ["ckpt-00000500"])

        print("\n=== 4. worker A releases; B takes over ===")
        a.release(steps_done=500, wall_seconds=3600, train_seconds=3450)
        st = b.read()
        check("lease cleared on release", st.lease is None)
        check("history records the session", len(st.history) == 1)
        eff = st.history[0]["efficiency"]
        check("efficiency computed", eff is not None and abs(eff - 0.9583) < 1e-3, f"{eff}")

        st = b.claim("worker-B", "kaggle-tpu", lease_seconds=600)
        check("B claims the freed lease", st.lease["worker_id"] == "worker-B")
        check("B resumes at A's step", st.step == 500)
        check("B inherits the data cursor", st.data_cursor["shard"] == 2)

        print("\n=== 5. dead worker: expired lease is reclaimable ===")
        raw, rev = LocalBackend(tmp).read()
        raw["lease"]["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat().replace("+00:00", "Z")
        LocalBackend(tmp).write(raw, rev)

        c = RelayState(LocalBackend(tmp), "run-1")
        st = c.claim("worker-C", "lightning", lease_seconds=600)
        check("expired lease is reclaimed", st.lease["worker_id"] == "worker-C")

        print("\n=== 6. compare-and-swap rejects a stale write ===")
        backend = LocalBackend(tmp)
        _, rev_old = backend.read()
        state = RunState(run_id="run-1", step=999)
        backend.write(state.to_dict(), rev_old)          # someone else writes first
        conflicted = False
        try:
            backend.write(RunState(run_id="run-1", step=1000).to_dict(), rev_old)
        except CASConflict:
            conflicted = True
        check("stale parent revision is rejected", conflicted)

        print("\n=== 6b. a moving branch is not a lost lease ===")
        # The Hub CASes on the branch head, not on the state file, and the
        # ledger, the dashboard snapshot and checkpoint uploads all commit to
        # the same repo. So a worker routinely gets 412 for commits that have
        # nothing to do with its lease. release() clears the lease before
        # committing, so the ownership check compared our worker id against
        # None and called every release a lost lease -- which killed the first
        # dispatched training sessions on exit, after training had worked.
        #
        # LocalBackend CAS is per-file and never produces this, which is why
        # the suite passed while production failed. The conflict is injected.
        tmp6 = Path(tempfile.mkdtemp(prefix="bussin_moved_"))
        try:
            inner = LocalBackend(tmp6)
            fired = {"n": 0}

            class MovingBranch:
                """Rejects the first write the way a bumped branch does."""

                def read(self):
                    return inner.read()

                def write(self, state, parent_revision):
                    if fired["n"] == 0:
                        fired["n"] = 1
                        raise CASConflict("412 Precondition Failed: branch moved")
                    return inner.write(state, parent_revision)

            r = RelayState(MovingBranch(), "run-6b")
            r.read()
            r.claim("worker-a", "kaggle-gpu", 600)
            released = True
            try:
                r.release(steps_done=10, wall_seconds=1.0, train_seconds=0.0)
            except LeaseHeld:
                released = False
            after = RelayState(LocalBackend(tmp6), "run-6b").read()
            check("release survives a branch bumped by another writer",
                  released and after.lease is None,
                  f"conflicts_injected={fired['n']} lease={after.lease}")
        finally:
            shutil.rmtree(tmp6, ignore_errors=True)

        print("\n=== 7. two workers race for one free lease ===")
        tmp2 = Path(tempfile.mkdtemp(prefix="bussin_race_"))
        try:
            r1 = RelayState(LocalBackend(tmp2), "run-2")
            r2 = RelayState(LocalBackend(tmp2), "run-2")
            r1.read()
            r2.read()                       # both see the same empty state
            r1.claim("racer-1", "kaggle-gpu", 600)
            lost = False
            try:
                r2._revision = None         # simulate r2 acting on its stale read
                r2.claim("racer-2", "colab", 600, retries=1)
            except LeaseHeld:
                lost = True
            winner = RelayState(LocalBackend(tmp2), "run-2").read()
            check("exactly one worker wins the race",
                  lost and winner.lease["worker_id"] == "racer-1",
                  f"holder={winner.lease['worker_id']}")
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        print("\n=== 8. a finished run refuses new claims ===")
        d = RelayState(LocalBackend(tmp), "run-1")
        d.claim("worker-D", "local", 600, force=True)
        d.release(finished=True)
        blocked = False
        try:
            RelayState(LocalBackend(tmp), "run-1").claim("worker-E", "local", 600)
        except LeaseHeld:
            blocked = True
        check("finished run blocks further work", blocked)

        failed = [r for r in results if r[0] == FAIL]
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
        if failed:
            for _, name, _d in failed:
                print(f"  FAILED: {name}")
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
