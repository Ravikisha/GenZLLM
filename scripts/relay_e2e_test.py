"""End-to-end relay test: two sessions, one run.

`scripts/resume_test.py` proves the *numerics* resume exactly. This proves the
*operational* path around them: a worker claims a lease, trains, checkpoints to
a store, exits; a second worker then picks the run up from the store and
continues without a discontinuity.

That is the whole project's foundation -- everything after Phase 3 is this same
loop with a bigger config -- so it is worth an explicit test rather than being
inferred from a successful training run.

Run:  python scripts/relay_e2e_test.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.relay.bootstrap import run
from bussin.relay.state import LocalBackend, RelayState

CONFIG = "configs/smoke.yaml"
ROOT = Path(".relay_e2e")
STEPS_PER_SESSION = 25


def read_state() -> dict:
    raw, _ = LocalBackend(ROOT).read()
    return raw or {}


def metrics_from(ckpt: Path) -> list[dict]:
    path = ckpt / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    checks: list[tuple[bool, str]] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        checks.append((ok, label))
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))

    print("=== session 1 ===")
    rc = run(CONFIG, max_steps=STEPS_PER_SESSION, local_root=str(ROOT))
    check(rc == 0, "session 1 exits cleanly", f"rc={rc}")

    s1 = read_state()
    check(s1.get("lease") is None, "lease released on exit")
    check(s1.get("step", 0) > 0, "progress recorded", f"step={s1.get('step')}")
    check(bool(s1.get("latest_ckpt")), "checkpoint registered", str(s1.get("latest_ckpt")))
    check(len(s1.get("history", [])) == 1, "session recorded in history")
    step_after_1 = s1.get("step", 0)
    ckpt1 = ROOT / str(s1.get("latest_ckpt"))
    check(ckpt1.exists(), "checkpoint present in the store", str(ckpt1))

    from bussin.relay.checkpoint import verify_checkpoint

    check(verify_checkpoint(ckpt1), "checkpoint passes hash verification")

    # Session 2's pruner will delete this checkpoint (keep_rolling=2), so read
    # what we need from it now rather than after the handover.
    cursor_1 = json.loads((ckpt1 / "data_cursor.json").read_text())

    print("\n=== session 2 (fresh worker, same store) ===")
    rc2 = run(CONFIG, max_steps=STEPS_PER_SESSION, local_root=str(ROOT))
    check(rc2 == 0, "session 2 exits cleanly", f"rc={rc2}")

    s2 = read_state()
    check(s2.get("step", 0) > step_after_1, "second session advanced the run",
          f"{step_after_1} -> {s2.get('step')}")
    check(len(s2.get("history", [])) == 2, "both sessions in history")
    check(s2.get("lease") is None, "lease released again")

    w1 = s2["history"][0].get("worker_id")
    w2 = s2["history"][1].get("worker_id")
    check(w1 != w2, "the two sessions were different workers", f"{w1} vs {w2}")

    print("\n=== loss continuity across the handover ===")
    final_ckpt = ROOT / str(s2.get("latest_ckpt"))
    rows = [r for r in metrics_from(final_ckpt) if "loss" in r]
    if len(rows) >= 6:
        boundary = step_after_1
        before = [r["loss"] for r in rows if r["step"] <= boundary][-3:]
        after = [r["loss"] for r in rows if r["step"] > boundary][:3]
        print(f"  last 3 before step {boundary}: {[round(x, 4) for x in before]}")
        print(f"  first 3 after           : {[round(x, 4) for x in after]}")
        if before and after:
            jump = abs(after[0] - before[-1]) / max(abs(before[-1]), 1e-9)
            # A real discontinuity (lost optimizer state, reshuffled data) shows
            # up as a large jump. Normal step-to-step noise is a few percent.
            check(jump < 0.25, "no discontinuity at the handover", f"jump {jump:.1%}")
        check(rows[-1]["step"] == s2.get("step"), "metrics span both sessions",
              f"{rows[0]['step']}..{rows[-1]['step']}")
    else:
        check(False, "enough metrics recorded to compare", f"{len(rows)} rows")

    print("\n=== data cursor advanced, not reset ===")
    c2 = json.loads((final_ckpt / "data_cursor.json").read_text())
    print(f"  session 1 cursor: {cursor_1}")
    print(f"  session 2 cursor: {c2}")
    advanced = (c2["epoch"], c2["shard"], c2["offset"]) > (
        cursor_1["epoch"], cursor_1["shard"], cursor_1["offset"]
    )
    check(advanced, "cursor moved forward across sessions")

    failed = [c for c in checks if not c[0]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    for _, label in failed:
        print(f"  FAILED: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
