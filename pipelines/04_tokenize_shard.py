"""Tokenize the corpus into uint16 shards, ordered by curriculum stage.

Each document is prefixed with its measured register control tokens (SPEC §3.4)
before tokenization, so the model learns p(text | register) rather than a
marginal averaged over wildly different registers.

Shard ordering is the curriculum. S1 shards (general English) come first, S3
shards (Gen-Z anneal) last, and within S3 documents are ordered oldest-to-newest
so the final gradients -- taken at the lowest learning rate -- see the most
contemporary language. Slang freshness is a position in the shard sequence, not
a filter (SPEC §4.1, §11.1).

Run:
  python pipelines/04_tokenize_shard.py --corpus data/corpus \
      --tokenizer tokenizer/tokenizer.json --out data/shards
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bussin.data.loader import Manifest, ShardWriter
from bussin.data.register import Register

# Which pools feed which curriculum stage, and in what proportion.
# Mirrors configs/*.yaml. See SPEC §4.4.
STAGE_MIX = {
    "S1": {"general": 1.00},
    "S2": {"general": 0.40, "internet": 0.35, "genz": 0.20, "hinglish": 0.03,
           "lexicon": 0.02},
    "S3": {"general": 0.20, "internet": 0.25, "genz": 0.50, "hinglish": 0.03,
           "lexicon": 0.02},
}
STAGE_SHARE = {"S1": 0.667, "S2": 0.300, "S3": 0.033}


def iter_jsonl(paths: list[Path]) -> Iterator[dict]:
    for p in paths:
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def bucket_by_pool(paths: list[Path]) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {}
    for row in iter_jsonl(paths):
        pools.setdefault(row.get("pool", "internet"), []).append(row)
    return pools


def stage_documents(pools: dict[str, list[dict]], stage: str,
                    budget_docs: int) -> list[dict]:
    """Draw documents for one stage according to its mixture."""
    mix = STAGE_MIX[stage]
    out: list[dict] = []
    for pool, share in mix.items():
        available = pools.get(pool, [])
        if not available:
            continue
        want = int(budget_docs * share)
        take = available[:want]
        # Leave the remainder for later stages rather than reusing documents.
        pools[pool] = available[want:]
        out.extend(take)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--no-register", action="store_true",
                    help="omit register control tokens (ablation only)")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    if vocab_size > 65_536:
        raise SystemExit(f"vocab {vocab_size} exceeds uint16 range")
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer is missing <|endoftext|>")
    print(f"tokenizer: vocab {vocab_size:,}, eos id {eos_id}")

    src = Path(args.corpus) / args.split
    paths = sorted(src.rglob("*.jsonl.gz")) + sorted(src.rglob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no jsonl under {src}")
    print(f"reading {len(paths)} file(s) from {src}")

    print("\n=== bucketing by pool ===")
    pools = bucket_by_pool(paths)
    for pool, rows in sorted(pools.items(), key=lambda kv: -len(kv[1])):
        print(f"  {pool:<10} {len(rows):>9,} documents")
    total_docs = sum(len(v) for v in pools.values())

    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    all_shards = []
    stats: Counter = Counter()
    register_hist: Counter = Counter()

    stages = ["S1", "S2", "S3"] if args.split == "train" else ["S1"]
    for stage in stages:
        budget = int(total_docs * STAGE_SHARE.get(stage, 1.0))
        docs = stage_documents(pools, stage, budget) if args.split == "train" \
            else [d for rows in pools.values() for d in rows]
        if not docs:
            print(f"\n  {stage}: no documents available, skipping")
            continue

        # Within the anneal, oldest first so the freshest language lands last.
        if stage == "S3":
            docs.sort(key=lambda d: d.get("created_utc", 0) or 0)

        print(f"\n=== {stage}: {len(docs):,} documents ===")
        writer = ShardWriter(out_dir, args.shard_tokens,
                             prefix=f"{args.split}_{stage}", stage=stage)

        batch_texts, batch_size = [], 1000
        for i, doc in enumerate(docs):
            text = doc["text"]
            if not args.no_register and doc.get("register"):
                reg = Register(**doc["register"])
                register_hist[f"slang_{reg.slang}"] += 1
                text = reg.to_tokens() + text
            batch_texts.append(text)

            if len(batch_texts) >= batch_size or i == len(docs) - 1:
                for enc in tok.encode_batch(batch_texts):
                    ids = enc.ids + [eos_id]
                    writer.add(ids)
                    stats["tokens"] += len(ids)
                stats["docs"] += len(batch_texts)
                batch_texts = []
                if stats["docs"] % 100_000 < batch_size:
                    print(f"    {stats['docs']:,} docs, "
                          f"{stats['tokens'] / 1e6:.1f}M tokens", flush=True)

        shards = writer.close()
        all_shards.extend(shards)
        print(f"  {len(shards)} shard(s), "
              f"{sum(s.n_tokens for s in shards) / 1e6:.2f}M tokens")

    manifest = Manifest(all_shards, root=out_dir, seq_len=args.seq_len,
                        vocab_size=vocab_size)
    manifest_path = out_dir / "manifest.json"
    manifest.save(manifest_path)

    print("\n=== summary ===")
    print(f"  documents          {stats['docs']:,}")
    print(f"  tokens             {stats['tokens']:,}  ({stats['tokens'] / 1e9:.3f}B)")
    print(f"  shards             {len(all_shards)}")
    print(f"  bytes on disk      {stats['tokens'] * 2 / 1e9:.2f} GB (uint16)")
    if stats["docs"]:
        print(f"  tokens/doc         {stats['tokens'] / stats['docs']:.1f}")
    if register_hist:
        print("\n  slang register distribution:")
        peak = max(register_hist.values())
        for i in range(5):
            n = register_hist.get(f"slang_{i}", 0)
            print(f"    slang_{i}  {n:>9,}  {'#' * int(50 * n / peak)}")
    print(f"\nwrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
