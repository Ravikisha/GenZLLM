"""Does the model actually USE the register control tokens?

The thesis of the project is p(text | register). If the model ignores the
register prefix, none of the downstream machinery -- RCE, BussBench's register
tasks, controllable generation -- can work, whatever the loss curve says.

This is a pure likelihood test, so it has no sampling noise and needs no
generation. Take real held-out windows, which begin with their true register
prefix, and score the SAME text under three prefixes:

  true      the document's own register
  shuffled  another document's register (chosen to differ from the true one)
  none      the prefix removed

If the model conditions on register, text is more likely under its true prefix:
NLL(true) < NLL(shuffled), paired per document. The paired difference has far
tighter error bars than comparing the group means, because document difficulty
cancels out.

`--offset` is the prediction distance the model was trained for: 1 for a correct
next-token model, 2 for bussin-125m-v1, which trained on a double-shifted target
(see scripts/objective_test.py). Scoring at the wrong offset measures nothing.

Run:
  python scripts/register_conditioning_test.py --ckpt eval_run/ckpt-25757 --offset 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from bussin.model.bussin_model import BussinForCausalLM
from bussin.model.config import BussinConfig
from bussin.relay.checkpoint import load_checkpoint

PREFIX_LEN = 7          # <|reg|> + five dimensions + <|/reg|>


def bootstrap_ci(x: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@torch.inference_mode()
def nll_per_doc(model, seqs: list[np.ndarray], offset: int, skip: int,
                batch: int = 16) -> np.ndarray:
    """Mean NLL per document over target positions >= `skip`, predicting at
    distance `offset`. Right-padded; causal attention means padding cannot
    affect earlier positions."""
    out = np.zeros(len(seqs))
    for lo in range(0, len(seqs), batch):
        chunk = seqs[lo:lo + batch]
        L = max(len(s) for s in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        for i, s in enumerate(chunk):
            x[i, :len(s)] = torch.from_numpy(s)
        lsm = model(x)["logits"].float().log_softmax(-1)
        for i, s in enumerate(chunk):
            n = len(s)
            tgt = torch.from_numpy(s[offset:]).long()                  # token j = i+offset
            lp = lsm[i, :n - offset].gather(-1, tgt[:, None])[:, 0]    # logits[i] -> token i+offset
            use = lp[max(skip - offset, 0):]
            out[lo + i] = -use.mean().item()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/125m.yaml")
    ap.add_argument("--shard", default="data/real_shards/val_S1_00000.bin")
    ap.add_argument("--offset", type=int, required=True,
                    help="prediction distance the model was trained for (1 = correct)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = BussinForCausalLM(BussinConfig.from_dict(raw["model"]))
    load_checkpoint(Path(args.ckpt), model, strict_verify=False)
    model.eval()

    a = np.fromfile(args.shard, dtype=np.uint16).astype(np.int64)
    eos = np.flatnonzero(a == 0)
    eos = eos[eos < a.size - args.window - 1]
    rng = np.random.default_rng(args.seed)
    starts = rng.choice(eos, size=min(args.n * 3, len(eos)), replace=False) + 1

    wins = []
    for s in starts:
        w = a[s:s + args.window]
        # a real document start: <|reg|> ... <|/reg|> then text
        if len(w) == args.window and w[0] != 0 and w[PREFIX_LEN - 1] != 0:
            wins.append(w)
        if len(wins) >= args.n:
            break
    n = len(wins)
    prefixes = [tuple(w[:PREFIX_LEN]) for w in wins]
    print(f"{n} held-out windows of {args.window} tokens, offset +{args.offset}, "
          f"{len(set(prefixes))} distinct register prefixes")

    # shuffled: another window's prefix, chosen to differ from the true one
    shuf = []
    for i in range(n):
        for _ in range(50):
            j = int(rng.integers(0, n))
            if prefixes[j] != prefixes[i]:
                shuf.append(np.array(prefixes[j]))
                break
        else:
            shuf.append(np.array(prefixes[i]))
    n_same = sum(1 for i in range(n) if tuple(shuf[i]) == prefixes[i])

    true_seqs = [w for w in wins]
    shuf_seqs = [np.concatenate([shuf[i], wins[i][PREFIX_LEN:]]) for i in range(n)]
    none_seqs = [wins[i][PREFIX_LEN:] for i in range(n)]

    # Score only text tokens, and identically across variants: target index >=
    # PREFIX_LEN + 1 in the prefixed sequences, >= 1 in the unprefixed one.
    skip = PREFIX_LEN + 1
    nll_true = nll_per_doc(model, true_seqs, args.offset, skip)
    nll_shuf = nll_per_doc(model, shuf_seqs, args.offset, skip)
    # The unprefixed sequence is PREFIX_LEN shorter, so the same text tokens
    # (those at prefixed index >= PREFIX_LEN + 1) sit at text index >= 1.
    nll_none = nll_per_doc(model, none_seqs, args.offset, skip - PREFIX_LEN)

    d_shuf = nll_shuf - nll_true
    d_none = nll_none - nll_true
    lo, hi = bootstrap_ci(d_shuf)
    lo2, hi2 = bootstrap_ci(d_none)

    print(f"\nmean NLL on the same text tokens (lower = more likely):")
    print(f"  true register      {nll_true.mean():.4f}")
    print(f"  shuffled register  {nll_shuf.mean():.4f}")
    print(f"  no register        {nll_none.mean():.4f}")
    print(f"\npaired differences (positive = the true register helps):")
    print(f"  shuffled - true    {d_shuf.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"true better in {(d_shuf > 0).mean():.1%} of docs")
    print(f"  none     - true    {d_none.mean():+.4f}   95% CI [{lo2:+.4f}, {hi2:+.4f}]   "
          f"true better in {(d_none > 0).mean():.1%} of docs")
    if n_same:
        print(f"\nnote: {n_same} windows had no differing prefix to swap in")

    sig = lo > 0
    print(f"\n{'conditioning detected' if sig else 'NO conditioning detected'}: "
          f"CI for shuffled-true {'excludes' if sig else 'includes'} 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
