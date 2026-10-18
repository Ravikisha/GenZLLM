"""The model must learn NEXT-token prediction, through the real loader.

Why this test exists
--------------------
`bussin-125m` trained 25,757 steps (~110 GPU-hours, 6.74B tokens) on the wrong
objective. The loader emits aligned pairs (input_ids = window[:-1],
labels = window[1:]) and the model shifted a second time, so position i was
scored against token i+2: the model learned to predict the token AFTER the next.

Every other test missed it because they check that code *runs* or that numbers
*reproduce*:

  * resume_test    -- the curve before and after a resume agree. They did, for
                      the wrong objective.
  * lease_test     -- concurrency; unrelated.
  * the loss curve -- it descended smoothly. Skip-ahead prediction is learnable,
                      just harder, so a falling loss is not evidence of a
                      correct objective.

This test asserts on what the model LEARNED. The data is a deterministic cycle
(token t is always followed by (t + 1) mod V), pushed through the real shard
writer, the real PackedDataset and the real model loss. After a short fit:

  * argmax at position i must equal token i+1  (next-token)
  * and must NOT equal token i+2               (the old, wrong behaviour)

With the bug the second check passes and the first fails, which is the point.

Run:  python scripts/objective_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bussin.data.loader import Manifest, PackedDataset, ShardWriter, TorchLoader
from bussin.model.bussin_model import BussinForCausalLM
from bussin.model.config import BussinConfig

VOCAB = 64
SEQ = 32


def main() -> int:
    torch.manual_seed(0)
    root = Path(tempfile.mkdtemp(prefix="bussin_objective_"))
    try:
        # token t is always followed by (t+1) % VOCAB; start each doc at a
        # random phase so the model cannot memorise absolute positions
        rng = np.random.default_rng(0)
        w = ShardWriter(root, shard_tokens=10_000_000, prefix="cyc")
        for _ in range(400):
            start = int(rng.integers(1, VOCAB))
            w.add([(start + i) % VOCAB for i in range(SEQ * 4)])
        shards = w.close()
        Manifest(shards, root=root, seq_len=SEQ, vocab_size=VOCAB).save(root / "manifest.json")

        manifest = Manifest.load(root / "manifest.json")
        ds = PackedDataset(manifest, seq_len=SEQ, emit_document_ids=False,
                           shuffle_shards=False)
        loader = iter(TorchLoader(ds, micro_batch=16, device="cpu"))

        cfg = BussinConfig(
            name="objective-test", vocab_size=VOCAB, n_layers=2, d_model=64,
            n_heads=4, n_kv_heads=2, head_dim=16, max_seq_len=SEQ,
            tie_embeddings=True, z_loss=0.0, use_document_mask=False,
        )
        model = BussinForCausalLM(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

        model.train()
        for step in range(250):
            batch = next(loader)
            out = model(batch["input_ids"], labels=batch["labels"])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
        print(f"final train loss {out['loss'].item():.4f}")

        model.eval()
        batch = next(loader)
        x = batch["input_ids"]
        with torch.inference_mode():
            pred = model(x)["logits"].argmax(-1)
        acc_next = (pred[:, :-1] == x[:, 1:]).float().mean().item()
        acc_skip = (pred[:, :-2] == x[:, 2:]).float().mean().item()
        print(f"argmax == token i+1 (next-token): {acc_next:.1%}")
        print(f"argmax == token i+2 (skip-ahead): {acc_skip:.1%}")

        ok_next = acc_next > 0.95
        ok_not_skip = acc_skip < 0.5
        print(f"\n  {'PASS' if ok_next else 'FAIL'}  learns next-token prediction")
        print(f"  {'PASS' if ok_not_skip else 'FAIL'}  does not learn skip-ahead")
        passed = ok_next and ok_not_skip
        print(f"\n{'PASS' if passed else 'FAIL'}: objective is "
              f"{'next-token' if passed else 'WRONG'}")
        return 0 if passed else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
