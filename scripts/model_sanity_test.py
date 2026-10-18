"""Architecture invariants, asserted on the real model code.

Written after bussin-125m trained for ~110 GPU-hours on a double-shifted target
and nothing noticed: the loss fell smoothly, resumes were bit-exact, and every
test passed. Reading the code did not find it either. These tests state the
properties a correct decoder must have and check them numerically, so a
violation fails here, in seconds on a tiny model, rather than after weeks.

  1. init loss      -- an untrained model scores ~ln(V) per token
  2. causality      -- changing future tokens cannot change earlier logits
  3. doc isolation  -- with document ids, one packed document cannot see another
  4. RoPE reference -- apply_rope equals an independent complex-rotation version
  5. RoPE relative  -- attention scores depend on relative, not absolute, position
  6. GQA grouping   -- repeat_kv matches torch.repeat_interleave

(The objective itself -- next-token vs skip-ahead -- is scripts/objective_test.py.)

Run:  python scripts/model_sanity_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from bussin.model.bussin_model import (BussinForCausalLM, RotaryEmbedding,
                                       apply_rope, repeat_kv)
from bussin.model.config import BussinConfig

V, T = 512, 48
FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    if not ok:
        FAILS.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))


def tiny(**kw) -> BussinConfig:
    base = dict(name="sanity", vocab_size=V, n_layers=2, d_model=64, n_heads=4,
                n_kv_heads=2, head_dim=16, max_seq_len=T, tie_embeddings=True,
                z_loss=0.0, use_document_mask=True)
    base.update(kw)
    return BussinConfig(**base)


def main() -> int:
    torch.manual_seed(0)

    print("=== 1. init loss ~ ln(V) ===")
    m = BussinForCausalLM(tiny()).eval()
    x = torch.randint(1, V, (8, T + 1))
    with torch.inference_mode():
        # the aligned pair the loader emits: target of position i is token i+1.
        # (Passing labels=x scores each position against its OWN input token,
        # which tied embeddings make easy -- a loss below ln(V) that says
        # nothing about the model.)
        loss = m(x[:, :-1], labels=x[:, 1:])["loss"].item()
    check(abs(loss - math.log(V)) < 0.5, "untrained loss is near ln(V)",
          f"{loss:.3f} vs ln({V}) = {math.log(V):.3f}")

    print("\n=== 2. causality: the future cannot leak into the past ===")
    a = torch.randint(1, V, (2, T))
    b = a.clone()
    cut = 20
    b[:, cut:] = torch.randint(1, V, (2, T - cut))
    with torch.inference_mode():
        la, lb = m(a)["logits"], m(b)["logits"]
    drift = (la[:, :cut] - lb[:, :cut]).abs().max().item()
    check(drift < 1e-5, "logits before the change are unaffected", f"max |delta| = {drift:.2e}")
    check((la[:, cut:] - lb[:, cut:]).abs().max().item() > 1e-3,
          "logits after the change do move (the test can detect change)")

    print("\n=== 3. document isolation under packing ===")
    d = torch.zeros(2, T, dtype=torch.long)
    d[:, 24:] = 1
    c = a.clone()
    c[:, :24] = torch.randint(1, V, (2, 24))          # rewrite document 0 only
    with torch.inference_mode():
        l1 = m(a, document_ids=d)["logits"]
        l2 = m(c, document_ids=d)["logits"]
    iso = (l1[:, 24:] - l2[:, 24:]).abs().max().item()
    check(iso < 1e-5, "document 1 logits ignore document 0 content", f"max |delta| = {iso:.2e}")
    no_mask = BussinForCausalLM(tiny(use_document_mask=False)).eval()
    with torch.inference_mode():
        leak = (no_mask(a, document_ids=d)["logits"][:, 24:]
                - no_mask(c, document_ids=d)["logits"][:, 24:]).abs().max().item()
    check(leak > 1e-3, "without the mask they DO see each other (the test can detect leakage)")

    print("\n=== 4. RoPE equals an independent reference ===")
    hd = 16
    rope = RotaryEmbedding(hd, T)
    cos, sin = rope(T, "cpu", torch.float32)
    q = torch.randn(1, 3, T, hd)
    k = torch.randn(1, 3, T, hd)
    qr, kr = apply_rope(q, k, cos, sin)
    inv = 1.0 / (10_000.0 ** (torch.arange(0, hd, 2).float() / hd))
    ang = torch.outer(torch.arange(T).float(), inv)               # (T, hd/2)
    rot = torch.polar(torch.ones_like(ang), ang)                  # e^{i*theta}

    def ref(z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[..., : hd // 2], z[..., hd // 2:]
        c = torch.complex(z1, z2) * rot                           # pairs (i, i+hd/2)
        return torch.cat([c.real, c.imag], dim=-1)

    check(torch.allclose(qr, ref(q), atol=1e-5) and torch.allclose(kr, ref(k), atol=1e-5),
          "apply_rope matches the complex-rotation reference")

    print("\n=== 5. RoPE attention depends on relative position only ===")
    q0 = torch.randn(1, 1, 1, hd)
    k0 = torch.randn(1, 1, 1, hd)

    def score(i: int, j: int) -> float:
        qq = q0.expand(1, 1, T, hd).clone()
        kk = k0.expand(1, 1, T, hd).clone()
        qr_, kr_ = apply_rope(qq, kk, cos, sin)
        return (qr_[0, 0, i] * kr_[0, 0, j]).sum().item()

    s_a, s_b = score(10, 4), score(30, 24)                        # same offset (6)
    check(abs(s_a - s_b) < 1e-4, "score(i, j) == score(i+s, j+s)", f"{s_a:.5f} vs {s_b:.5f}")
    check(abs(score(10, 4) - score(10, 7)) > 1e-3, "different offsets give different scores")

    print("\n=== 6. GQA groups consecutive query heads ===")
    kv = torch.randn(2, 2, 5, 8)
    got = repeat_kv(kv, 3)
    want = torch.repeat_interleave(kv, 3, dim=1)
    check(torch.equal(got, want), "repeat_kv == repeat_interleave (query head h uses kv head h // n_rep)")

    print(f"\n{'PASS' if not FAILS else 'FAIL'}: "
          f"{'all invariants hold' if not FAILS else ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
