"""Build the manifest for the realign run.

bussin-125m-v1 trained on the wrong target (see docs/SPEC.md 5.9e). The realign
run continues from its weights with the corrected loss, on the data that carries
the register signal: a sample of S2 shards (mixed) followed by every S3 shard
(the Gen-Z anneal), in that order so the final low-LR steps see the most
contemporary text.

The sample is seeded, and balanced across the five sharding partitions so no one
partition's slice of the corpus is over-represented.

Run:
  python scripts/make_realign_manifest.py --publish
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

GLOBAL_BATCH = 262_144


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/manifests/train_manifest.json")
    ap.add_argument("--out", default="data/manifests/realign_manifest.json")
    ap.add_argument("--s2-shards", type=int, default=9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    src = json.loads(Path(args.src).read_text(encoding="utf-8"))
    shards = src["shards"]
    s2 = [s for s in shards if s["stage"] == "S2"]
    s3 = [s for s in shards if s["stage"] == "S3"]

    def part(s: dict) -> int:
        m = re.search(r"_s(\d+)_", s["path"])
        return int(m.group(1)) if m else -1

    # round-robin over partitions, shuffled within each, so the sample is balanced
    rng = np.random.default_rng(args.seed)
    by_part: dict[int, list[dict]] = {}
    for s in s2:
        by_part.setdefault(part(s), []).append(s)
    for v in by_part.values():
        rng.shuffle(v)
    picked: list[dict] = []
    while len(picked) < args.s2_shards and any(by_part.values()):
        for p in sorted(by_part):
            if by_part[p] and len(picked) < args.s2_shards:
                picked.append(by_part[p].pop())
    picked.sort(key=lambda s: s["path"])
    s3.sort(key=lambda s: s["path"])

    chosen = picked + s3
    total = sum(s["n_tokens"] for s in chosen)
    steps = total // GLOBAL_BATCH
    s2_tok = sum(s["n_tokens"] for s in picked)

    out = {"seq_len": src["seq_len"], "vocab_size": src["vocab_size"],
           "n_shards": len(chosen), "total_tokens": total, "shards": chosen}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"S2: {len(picked)} shards  {s2_tok / 1e9:.3f}B   (partitions "
          f"{dict(sorted(Counter(part(s) for s in picked).items()))})")
    print(f"S3: {len(s3)} shards  {(total - s2_tok) / 1e9:.3f}B")
    print(f"total {total:,} tokens  -> {steps:,} steps at a {GLOBAL_BATCH:,}-token batch")
    print(f"S2 fraction of steps {s2_tok / total:.4f}  (decay should begin there)")
    print(f"wrote {args.out}")

    if args.publish:
        from bussin.config import project
        from huggingface_hub import HfApi

        cfg = project()
        HfApi(token=cfg.creds.hf_token).upload_file(
            path_or_fileobj=args.out,
            path_in_repo="data/shards/train/manifest_realign.json",
            repo_id=cfg.corpus_repo, repo_type="dataset",
            commit_message="shards: realign manifest",
        )
        print(f"published -> {cfg.corpus_repo}:data/shards/train/manifest_realign.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
