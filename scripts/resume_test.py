"""The resume test -- the foundation the whole project stands on.

Trains a tiny model twice:
  A) uninterrupted for 2N steps
  B) N steps, checkpoint, destroy everything, restore, N more steps

If the relay is correct, the two loss curves are identical to floating-point
noise. If optimizer moments, the LR schedule, the RNG state or the data cursor
are dropped on resume, run B diverges from run A -- subtly at first, which is
exactly why this has to be tested rather than assumed.

Run:  python scripts/resume_test.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bussin.data.loader import Cursor, Manifest, PackedDataset, ShardInfo, TorchLoader
from bussin.model.bussin_model import BussinForCausalLM
from bussin.model.config import BussinConfig
from bussin.relay.checkpoint import load_checkpoint, verify_checkpoint
from bussin.relay.platform import PlatformInfo, plan_batch
from bussin.train.schedule import Curriculum, build_schedule
from bussin.train.trainer import Precision, TrainConfig, Trainer

STEPS = 12
SEQ_LEN = 64
VOCAB = 512


def make_corpus(root: Path, n_tokens: int = 60_000, seed: int = 7) -> Path:
    """A deterministic fake corpus with document boundaries at token 0."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shards = []
    for i in range(3):
        toks = rng.integers(1, VOCAB, size=n_tokens, dtype=np.uint16)
        toks[::137] = 0  # EOS every ~137 tokens -> exercises the document mask
        path = root / f"shard_{i:05d}.bin"
        toks.tofile(path)
        shards.append(ShardInfo(path=path.name, n_tokens=int(toks.size), stage="S1"))
    m = Manifest(shards, root=root, seq_len=SEQ_LEN, vocab_size=VOCAB)
    m.save(root / "manifest.json")
    return root / "manifest.json"


def build(seed: int = 1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mcfg = BussinConfig(
        name="bussin-test", vocab_size=VOCAB, n_layers=2, d_model=128,
        n_heads=2, n_kv_heads=1, max_seq_len=SEQ_LEN, tie_embeddings=True,
    )
    tcfg = TrainConfig(
        run_id="resume-test", total_steps=STEPS * 2,
        global_batch_tokens=SEQ_LEN * 4, seq_len=SEQ_LEN,
        lr_max=1e-3, lr_min=1e-4, warmup_steps=3, stable_frac=0.6,
        log_every=10_000, val_every=10_000,
    )
    info = PlatformInfo(
        platform="local", device_type="cpu", n_devices=1, device_name="cpu",
        supports_bf16=False, session_limit_s=9999, checkpoint_reserve_s=1,
        worker_id="test", host="test",
    )
    plan = plan_batch(tcfg.global_batch_tokens, SEQ_LEN, 1, micro_batch=4,
                      device_type="cpu")
    model = BussinForCausalLM(mcfg)
    sched = build_schedule(tcfg.total_steps, tcfg.lr_max, tcfg.lr_min,
                           tcfg.warmup_steps, tcfg.stable_frac)
    trainer = Trainer(model, mcfg, tcfg, plan, info, sched, torch.device("cpu"),
                      Precision(info, force="fp32"), Curriculum(tcfg.total_steps))
    return trainer, mcfg, tcfg, plan


def loader_for(manifest_path: Path, cursor: Cursor, micro_batch: int, seed: int):
    ds = PackedDataset(Manifest.load(manifest_path), SEQ_LEN, cursor=cursor,
                       seed=seed, shuffle_shards=True)
    return ds, iter(TorchLoader(ds, micro_batch, "cpu"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bussin_resume_"))
    try:
        manifest = make_corpus(tmp / "corpus")
        print(f"corpus: {manifest.parent}")

        # ---------------- Run A: uninterrupted ----------------
        print("\n=== Run A: uninterrupted, 2N steps ===")
        trainer_a, *_ = build()
        ds_a, load_a = loader_for(manifest, Cursor(), trainer_a.plan.micro_batch, 1337)
        losses_a = []
        for _ in range(STEPS * 2):
            row = trainer_a.train_step(load_a)
            losses_a.append(row["loss"])
        print(f"  first {losses_a[0]:.6f} -> last {losses_a[-1]:.6f}")

        # ---------------- Run B: interrupted ------------------
        print(f"\n=== Run B: {STEPS} steps, checkpoint, destroy, resume ===")
        trainer_b, mcfg, tcfg, plan = build()
        ds_b, load_b = loader_for(manifest, Cursor(), plan.micro_batch, 1337)
        losses_b = []
        for _ in range(STEPS):
            losses_b.append(trainer_b.train_step(load_b)["loss"])

        ckpt_root = tmp / "ckpt"
        path = trainer_b.save(ckpt_root, ds_b.state(), val_loss=None)
        print(f"  checkpointed at step {trainer_b.step} -> {path.name}")
        print(f"  manifest verifies: {verify_checkpoint(path)}")
        cursor_saved = json.loads((path / 'data_cursor.json').read_text())
        print(f"  data cursor: {cursor_saved}")

        # Destroy everything the session held.
        del trainer_b, ds_b, load_b

        print("  --- new 'session' ---")
        trainer_c, *_ = build(seed=999)          # different init seed on purpose:
        loaded = load_checkpoint(path, trainer_c.model, trainer_c.optimizer,  # the
                                 trainer_c.schedule, trainer_c.scaler)       # checkpoint
        trainer_c.step = loaded["meta"]["step"]                              # must fully
        trainer_c.tokens_seen = loaded["meta"]["tokens_seen"]                # determine state
        ds_c, load_c = loader_for(manifest, Cursor.from_dict(loaded["data_cursor"]),
                                  trainer_c.plan.micro_batch, 1337)
        print(f"  restored step {trainer_c.step}, cursor {ds_c.state()}")

        for _ in range(STEPS):
            losses_b.append(trainer_c.train_step(load_c)["loss"])

        # ---------------- Compare ----------------
        print("\n=== Loss curve comparison ===")
        print(f"{'step':>5} {'run A':>12} {'run B':>12} {'abs diff':>12} {'rel':>9}")
        worst_rel = 0.0
        for i, (a, b) in enumerate(zip(losses_a, losses_b), start=1):
            rel = abs(a - b) / max(abs(a), 1e-9)
            worst_rel = max(worst_rel, rel)
            mark = "  <-- resume here" if i == STEPS + 1 else ""
            if i <= 3 or i in (STEPS, STEPS + 1, STEPS + 2) or i > STEPS * 2 - 2:
                print(f"{i:>5} {a:>12.6f} {b:>12.6f} {abs(a - b):>12.3e} {rel:>8.2%}{mark}")

        print(f"\nworst relative divergence across all {len(losses_a)} steps: {worst_rel:.3e}")

        post = [
            abs(a - b) / max(abs(a), 1e-9)
            for a, b in zip(losses_a[STEPS:], losses_b[STEPS:])
        ]
        print(f"worst after the resume boundary:                  {max(post):.3e}")

        tol = 1e-4
        ok = worst_rel < tol
        print(f"\n{'PASS' if ok else 'FAIL'}: curves agree within {tol:.0e}"
              if ok else
              f"\nFAIL: divergence {worst_rel:.3e} exceeds {tol:.0e} -- "
              "something is not being restored")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
