"""Token accounting and curriculum order, asserted on what the loader serves.

bussin-125m-v1 recorded 6.74B tokens seen. The data loader had actually served
3.38B -- exactly half. The batch was planned for 2 devices (micro 4 x accum 32 x
2 dev) but nothing ever spread work across them, so a "2x T4" session ran on one
GPU, every step processed 128 sequences instead of 256, and `tokens_seen` and
`tok_per_s` were computed from the plan rather than from the data.

Separately, PackedDataset shuffled every shard globally, so the S1 -> S2 -> S3
curriculum -- which is purely shard order -- never ran.

Neither was visible in a loss curve. These tests check them directly:

  A  one device: tokens_seen equals what the loader served
  B  two devices (simulated; DataParallel falls back to passthrough on a CPU-only
     box): the loader must yield micro_batch * n_dp, and tokens_seen still equals
     what was served
  C  a loader/plan MISMATCH -- the old bug -- raises on the first step instead of
     silently training on half the data
  D  shard order shuffles within a stage but never across stages

Run:  python scripts/accounting_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from bussin.data.loader import (Cursor, Manifest, PackedDataset, ShardInfo,
                                TorchLoader)
from bussin.model.bussin_model import BussinForCausalLM
from bussin.model.config import BussinConfig
from bussin.relay.platform import PlatformInfo, plan_batch
from bussin.train.schedule import Curriculum, build_schedule
from bussin.train.trainer import Precision, TrainConfig, Trainer
from resume_test import SEQ_LEN, VOCAB, make_corpus

CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    CHECKS.append((ok, label))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))


def make_trainer(n_devices: int, device_type: str):
    torch.manual_seed(0)
    mcfg = BussinConfig(name="acct", vocab_size=VOCAB, n_layers=2, d_model=128,
                        n_heads=2, n_kv_heads=1, max_seq_len=SEQ_LEN, tie_embeddings=True)
    tcfg = TrainConfig(run_id="acct", total_steps=20, global_batch_tokens=SEQ_LEN * 16,
                       seq_len=SEQ_LEN, lr_max=1e-3, lr_min=1e-4, warmup_steps=2,
                       stable_frac=0.6, log_every=10_000, val_every=10_000)
    info = PlatformInfo(platform="local", device_type=device_type, n_devices=n_devices,
                        device_name="sim", supports_bf16=False, session_limit_s=9999,
                        checkpoint_reserve_s=1, worker_id="t", host="t")
    plan = plan_batch(tcfg.global_batch_tokens, SEQ_LEN, n_devices, micro_batch=4,
                      device_type=device_type)
    sched = build_schedule(tcfg.total_steps, tcfg.lr_max, tcfg.lr_min,
                           tcfg.warmup_steps, tcfg.stable_frac)
    trainer = Trainer(BussinForCausalLM(mcfg), mcfg, tcfg, plan, info, sched,
                      torch.device("cpu"), Precision(info, force="fp32"),
                      Curriculum(tcfg.total_steps))
    return trainer, plan


class CountingLoader:
    """Wraps a loader and counts the tokens it hands out."""

    def __init__(self, it):
        self.it, self.tokens = it, 0

    def __iter__(self):
        return self

    def __next__(self):
        b = next(self.it)
        self.tokens += int(b["input_ids"].numel())
        return b


def run_steps(trainer, plan, manifest_path: Path, loader_batch: int, steps: int = 3):
    ds = PackedDataset(Manifest.load(manifest_path), SEQ_LEN, cursor=Cursor(),
                       seed=1, shuffle_shards=True)
    loader = CountingLoader(iter(TorchLoader(ds, loader_batch, "cpu")))
    for _ in range(steps):
        trainer.train_step(loader)
    return loader.tokens


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bussin_acct_"))
    try:
        mp = make_corpus(tmp / "corpus", n_tokens=40_000)

        print("=== A. one device ===")
        tr, plan = make_trainer(1, "cpu")
        served = run_steps(tr, plan, mp, plan.micro_batch * getattr(tr, "n_dp", 1))
        check(getattr(tr, "n_dp", 1) == 1, "a single device runs un-wrapped", f"n_dp={getattr(tr, "n_dp", 1)}")
        check(tr.tokens_seen == served, "tokens_seen equals what the loader served",
              f"recorded {tr.tokens_seen:,} vs served {served:,}")
        check(tr.tokens_seen == 3 * plan.global_batch_tokens,
              "and equals steps x the planned global batch", f"{tr.tokens_seen:,}")

        print("\n=== B. two devices (simulated) ===")
        tr, plan = make_trainer(2, "cuda")
        served = run_steps(tr, plan, mp, plan.micro_batch * getattr(tr, "n_dp", 1))
        check(getattr(tr, "n_dp", 1) == 2, "two devices are used", f"n_dp={getattr(tr, "n_dp", 1)}")
        check(tr.tokens_seen == served == 3 * plan.global_batch_tokens,
              "with a micro_batch * n_dp loader, accounting is exact",
              f"recorded {tr.tokens_seen:,} served {served:,}")

        print("\n=== C. the old bug: a plan for 2 devices, a loader for 1 ===")
        tr, plan = make_trainer(2, "cuda")
        raised, msg = False, ""
        try:
            run_steps(tr, plan, mp, plan.micro_batch)      # loader batch NOT multiplied
        except RuntimeError as exc:
            raised, msg = True, str(exc)
        check(raised, "the mismatch raises on the first step instead of training on half the data",
              msg[:90] if msg else "no error -- silently trained on half the planned tokens")

        print("\n=== D. curriculum order ===")
        shards = ([ShardInfo(f"a{i}.bin", 100, stage="S1") for i in range(6)]
                  + [ShardInfo(f"b{i}.bin", 100, stage="S2") for i in range(4)]
                  + [ShardInfo(f"c{i}.bin", 100, stage="S3") for i in range(3)])
        m = Manifest(shards, root=tmp, seq_len=SEQ_LEN, vocab_size=VOCAB)
        rank = {"S1": 0, "S2": 1, "S3": 2}
        orders = []
        for ep in range(8):
            ds = PackedDataset(m, SEQ_LEN, seed=1337, shuffle_shards=True)
            orders.append(ds._shard_order(ep))
        mono = all(
            [rank[shards[i].stage] for i in o] == sorted(rank[shards[i].stage] for i in o)
            for o in orders)
        check(mono, "every epoch visits S1, then S2, then S3 -- never interleaved",
              " ".join(shards[i].stage for i in orders[0]))
        varies = len({tuple(o[:6]) for o in orders}) > 1
        check(varies, "shards are still shuffled within a stage between epochs")
        check(all(sorted(o) == list(range(13)) for o in orders),
              "every shard is visited exactly once per epoch")

        print("\n=== E. pruning is scoped to its own run ===")
        from bussin.relay.checkpoint import (CheckpointStore, checkpoint_name,
                                             checkpoint_prefix)
        store_dir = tmp / "store"
        for d in ("ckpt-00025757", "ckpt-runB-00000300", "ckpt-runA-00000100",
                  "ckpt-runA-00000200", "ckpt-runA-00000300"):
            (store_dir / d).mkdir(parents=True)
        removed = CheckpointStore(str(store_dir)).prune(
            ["ckpt-runA-00000300"], prefix=checkpoint_prefix("runA"))
        left = sorted(d.name for d in store_dir.iterdir())
        check(sorted(removed) == ["ckpt-runA-00000100", "ckpt-runA-00000200"],
              "prune removes only this run's superseded checkpoints", f"removed {sorted(removed)}")
        check("ckpt-00025757" in left and "ckpt-runB-00000300" in left,
              "another run's checkpoints, and the old run's final weights, survive")
        raised = False
        try:
            CheckpointStore(str(store_dir)).prune([], prefix="ckpt-")
        except ValueError:
            raised = True
        check(raised, "the shared 'ckpt-' prefix is refused outright")
        check(checkpoint_name(5, "r") == "ckpt-r-00000005", "names carry the run id")

        failed = [lab for ok, lab in CHECKS if not ok]
        print(f"\n{'PASS' if not failed else 'FAIL'}: "
              f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks")
        return 0 if not failed else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
