"""The orchestrator tick.

One invocation does one thing: decide whether to start a training session
somewhere, and if so, start it. Then exit.

It is deliberately **stateless and idempotent**. Every tick re-reads the truth
from Hugging Face -- run state, lease, ledger -- and holds nothing in memory
between invocations. A missed tick, a crashed runner, or two ticks racing all
resolve themselves, because the lease is the only thing that decides who is
allowed to train, and it already survives every failure mode the relay tests
cover.

That is why "stop when quotas are exhausted and resume next week" needs no
code: when nothing has budget, each tick exits in a couple of seconds until
the reset boundary rolls over.

Run:
  python -m bussin.orchestrator.tick --config configs/400m.yaml
  python -m bussin.orchestrator.tick --dry-run     # decide, don't dispatch
  python -m bussin.orchestrator.tick --status      # print state and exit
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import ProjectConfig, project
from ..relay.state import LocalBackend, RelayState, make_backend
from .ledger import BUDGETS, Ledger, iso, rank_platforms, utcnow
from .platforms import build_adapters

LEDGER_FILE = "orchestrator/ledger.json"
DASHBOARD_FILE = "orchestrator/dashboard.json"


# ------------------------------------------------------------------ #
# Ledger storage (same repo as RUN_STATE, same CAS discipline)
# ------------------------------------------------------------------ #


def load_ledger(cfg: ProjectConfig) -> Ledger:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            cfg.ckpt_repo, LEDGER_FILE, repo_type="model",
            token=cfg.creds.hf_token, force_download=True,
        )
        return Ledger.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (EntryNotFoundError, RepositoryNotFoundError, Exception):
        return Ledger()


def save_ledger(cfg: ProjectConfig, ledger: Ledger) -> None:
    from huggingface_hub import HfApi

    ledger.updated_at = iso(utcnow())
    HfApi(token=cfg.creds.hf_token).upload_file(
        path_or_fileobj=json.dumps(ledger.to_dict(), indent=1).encode(),
        path_in_repo=LEDGER_FILE,
        repo_id=cfg.ckpt_repo,
        repo_type="model",
        commit_message="orchestrator: ledger update",
    )


def save_dashboard(cfg: ProjectConfig, payload: dict[str, Any]) -> None:
    from huggingface_hub import HfApi

    HfApi(token=cfg.creds.hf_token).upload_file(
        path_or_fileobj=json.dumps(payload, indent=1).encode(),
        path_in_repo=DASHBOARD_FILE,
        repo_id=cfg.ckpt_repo,
        repo_type="model",
        commit_message="orchestrator: dashboard snapshot",
    )


# ------------------------------------------------------------------ #
# Tick
# ------------------------------------------------------------------ #


def tick(
    config_path: str = "configs/400m.yaml",
    run_id: str | None = None,
    dry_run: bool = False,
    code_slug: str = "",
    datasets: list[str] | None = None,
    local_root: str | None = None,
    publish_dashboard: bool = True,
    force_platform: str | None = None,
) -> dict[str, Any]:
    cfg = project()
    now = utcnow()
    out: dict[str, Any] = {"at": iso(now), "action": None, "reason": ""}

    # --- 1. run state -------------------------------------------------
    backend = (LocalBackend(local_root) if local_root
               else make_backend(cfg.state_uri, token=cfg.creds.hf_token))
    run_id = run_id or Path(config_path).stem
    relay = RelayState(backend, run_id)
    state = relay.read()
    out["run"] = {
        "run_id": state.run_id, "step": state.step, "stage": state.stage,
        "tokens_seen": state.tokens_seen, "latest_ckpt": state.latest_ckpt,
        "finished": state.finished,
    }

    if state.finished:
        out["action"] = "none"
        out["reason"] = "run is marked finished"
        return _finish(cfg, out, Ledger(), publish_dashboard, dry_run)

    # --- 2. is someone already training? ------------------------------
    lease = state.lease_obj()
    if lease and lease.is_live(now):
        out["action"] = "none"
        out["reason"] = (f"lease held by {lease.worker_id} ({lease.platform}) "
                         f"until {lease.expires_at}")
        out["lease"] = state.lease
        ledger = load_ledger(cfg)
        return _finish(cfg, out, ledger, publish_dashboard, dry_run)

    # --- 3. ledger ----------------------------------------------------
    ledger = load_ledger(cfg)

    # A lease that has expired means the session it belonged to is gone;
    # close its ledger record so its hours stop accruing as "running".
    if lease and not lease.is_live(now):
        if ledger.close_dispatch(lease.worker_id, status="unknown",
                                 detail="lease expired without release"):
            out["closed_stale"] = lease.worker_id

    # --- 4. choose a platform ----------------------------------------
    ranked = rank_platforms(ledger, now)
    if force_platform:
        # Manual override, for integration tests and for pinning a run to one
        # accelerator while the other backend is still unproven.
        ranked = [force_platform] if force_platform in BUDGETS else []
    out["quota"] = {
        name: {
            "remaining_h": round(ledger.remaining_hours(name, now), 2),
            "used_h": round(ledger.used_hours(name, now), 2),
            "quota_h": BUDGETS[name].quota_hours,
        }
        for name in BUDGETS
    }

    if not ranked:
        soonest = min(
            ((name, ledger.next_reset(name, now)) for name in BUDGETS),
            key=lambda kv: kv[1],
        )
        out["action"] = "sleep"
        out["reason"] = (f"all quotas exhausted; next reset {iso(soonest[1])} "
                         f"({soonest[0]}, in "
                         f"{(soonest[1] - now).total_seconds() / 3600:.1f}h)")
        out["next_reset"] = iso(soonest[1])
        return _finish(cfg, out, ledger, publish_dashboard, dry_run)

    adapters = build_adapters(cfg, code_slug, datasets)
    out["candidates"] = ranked

    for name in ranked:
        adapter = adapters.get(name)
        if adapter is None:
            continue
        ok, why = adapter.preflight()
        if not ok:
            out.setdefault("skipped", []).append({"platform": name, "reason": why})
            continue

        worker_id = f"{name}-{now.strftime('%m%d%H%M')}"
        if dry_run:
            out["action"] = "would_dispatch"
            out["reason"] = f"{name} (dry run; nothing launched)"
            out["platform"] = name
            return _finish(cfg, out, ledger, publish_dashboard, dry_run)

        result = adapter.dispatch(config_path, worker_id)
        if result.ok:
            ledger.record_dispatch(name, worker_id, detail=result.ref)
            out["action"] = "dispatched"
            out["platform"] = name
            out["worker_id"] = worker_id
            out["reason"] = result.detail
            return _finish(cfg, out, ledger, publish_dashboard, dry_run)

        out.setdefault("failed", []).append(
            {"platform": name, "detail": result.detail})
        if not result.retryable:
            continue

    out["action"] = "none"
    out["reason"] = "quota available but no platform could be dispatched"
    return _finish(cfg, out, ledger, publish_dashboard, dry_run)


def _finish(cfg: ProjectConfig, out: dict[str, Any], ledger: Ledger,
            publish: bool, dry_run: bool) -> dict[str, Any]:
    out["ledger"] = ledger.snapshot()
    if publish and not dry_run:
        try:
            save_ledger(cfg, ledger)
        except Exception as exc:
            out["ledger_save_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        try:
            from .dashboard import build_snapshot

            save_dashboard(cfg, build_snapshot(cfg, out))
        except Exception as exc:
            out["dashboard_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


# ------------------------------------------------------------------ #


def format_report(out: dict[str, Any]) -> str:
    lines = [f"[{out['at']}]  action={out['action']}  {out['reason']}"]
    run = out.get("run", {})
    if run:
        lines.append(
            f"  run   step={run['step']:,} tokens={run['tokens_seen'] / 1e9:.3f}B "
            f"stage={run['stage']} ckpt={run['latest_ckpt']}"
        )
    for name, q in (out.get("quota") or {}).items():
        bar_len = int(20 * q["used_h"] / max(q["quota_h"], 1e-9))
        bar = "#" * min(bar_len, 20) + "." * max(0, 20 - bar_len)
        lines.append(
            f"  {name:<12} [{bar}] {q['used_h']:>5.1f}/{q['quota_h']:<5.1f}h  "
            f"free {q['remaining_h']:>5.1f}h"
        )
    for s in out.get("skipped", []):
        lines.append(f"  skip  {s['platform']:<12} {s['reason']}")
    for f in out.get("failed", []):
        lines.append(f"  FAIL  {f['platform']:<12} {f['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("bussin-orchestrator")
    ap.add_argument("--config", default="configs/400m.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="decide but do not dispatch or write")
    ap.add_argument("--status", action="store_true",
                    help="print state and quota, change nothing")
    ap.add_argument("--code-dataset", default="",
                    help="Kaggle dataset holding the source "
                         "(default <user>/bussin-code)")
    ap.add_argument("--dataset", action="append", default=[],
                    help="Kaggle dataset to attach; repeatable")
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--platform", default=None,
                    help="force a platform instead of ranking by throughput")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    out = tick(
        config_path=args.config, run_id=args.run_id,
        dry_run=args.dry_run or args.status, code_slug=args.code_dataset,
        datasets=args.dataset, local_root=args.local_root,
        publish_dashboard=not args.status, force_platform=args.platform,
    )
    print(json.dumps(out, indent=1) if args.json else format_report(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
