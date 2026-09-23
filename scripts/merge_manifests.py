"""Merge the per-partition shard manifests into the one the trainer reads.

Sharding runs as N parallel Kaggle sessions, each writing `manifest_s{i}.json`
for its own fifth of the corpus. The trainer wants a single manifest.

Shard **order is the curriculum** (SPEC 4.1, 11.1): S1 general English first,
S3 the Gen-Z anneal last, so the final gradients -- taken at the lowest
learning rate -- see the most contemporary language. So the merge cannot simply
concatenate the partitions, which would interleave S3 shards from partition 0
ahead of S1 shards from partition 1. It orders by stage first, partition
second.

Run:
  python scripts/merge_manifests.py --split train
  python scripts/merge_manifests.py --split train --publish
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.config import project

STAGE_RANK = {"S1": 0, "S2": 1, "S3": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None, help="local manifest path")
    ap.add_argument("--publish", action="store_true",
                    help="upload the merged manifest back to the corpus repo")
    args = ap.parse_args()

    cfg = project()
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=cfg.creds.hf_token)
    info = api.repo_info(cfg.corpus_repo, repo_type="dataset")
    prefix = f"data/shards/{args.split}/"
    parts = sorted(
        s.rfilename for s in info.siblings
        if s.rfilename.startswith(prefix)
        and Path(s.rfilename).name.startswith("manifest_s")
    )
    if not parts:
        raise SystemExit(f"no manifest_s*.json under {prefix}")
    print(f"merging {len(parts)} partition manifests")

    shards: list[dict] = []
    seq_len = vocab_size = None
    for rel in parts:
        p = hf_hub_download(cfg.corpus_repo, rel, repo_type="dataset",
                            token=cfg.creds.hf_token, force_download=True)
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        seq_len = seq_len or d.get("seq_len")
        vocab_size = vocab_size or d.get("vocab_size")
        if d.get("vocab_size") != vocab_size:
            raise SystemExit(f"{rel}: vocab_size {d.get('vocab_size')} != {vocab_size}; "
                             f"partitions were built with different tokenizers")
        shards.extend(d["shards"])
        print(f"  {Path(rel).name:<18} {len(d['shards']):>3} shards  "
              f"{d.get('total_tokens', 0) / 1e9:.3f}B tokens")

    # Paths inside each partition manifest are relative to the same directory,
    # so they stay valid; only the ordering changes.
    shards.sort(key=lambda s: (STAGE_RANK.get(s.get("stage", "S1"), 9),
                               Path(s["path"]).name))

    total = sum(s["n_tokens"] for s in shards)
    merged = {"seq_len": seq_len, "vocab_size": vocab_size,
              "n_shards": len(shards), "total_tokens": total, "shards": shards}

    # Not data/shards/<split>/manifest.json: the relay end-to-end test builds
    # its fixtures there, and writing the real 75-shard manifest over the
    # test's three-shard one made the suite fail on missing shards.
    out = Path(args.out or f"data/manifests/{args.split}_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    by_stage: dict[str, list[int]] = {}
    for s in shards:
        by_stage.setdefault(s.get("stage", "S1"), []).append(s["n_tokens"])
    print(f"\nmerged -> {out}")
    for stage in sorted(by_stage, key=lambda k: STAGE_RANK.get(k, 9)):
        n = by_stage[stage]
        print(f"  {stage}  {len(n):>3} shards  {sum(n) / 1e9:.3f}B tokens")
    print(f"  total {len(shards)} shards, {total:,} tokens ({total / 1e9:.3f}B)")
    print(f"  order: {' '.join(s.get('stage', '?') for s in shards[:8])} ...")

    if args.publish:
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=f"{prefix}manifest.json",
            repo_id=cfg.corpus_repo, repo_type="dataset",
            commit_message=f"shards: merged {args.split} manifest",
        )
        print(f"  published -> {cfg.corpus_repo}:{prefix}manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
