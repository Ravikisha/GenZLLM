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
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bussin.data.loader import Manifest, ShardWriter
from bussin.data.lexicon import Lexicon
from bussin.data.register import Register, RegisterAnnotator

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


def count_pools(paths: list[Path]) -> tuple[Counter, "array"]:
    """First streaming pass: how many documents per pool, and each document's
    pool in stream order.

    Deliberately does NOT hold the documents. The corpus is ~52M documents and
    the general pool is long-form, so materialising them to plan the curriculum
    needed tens of gigabytes and could not run on any free CPU session. Only
    the pool code is retained -- one byte per document, ~52 MB -- which is
    enough to plan exactly, and the text is re-streamed in the second pass.
    """
    from array import array

    counts: Counter = Counter()
    codes = array("B")
    order: dict[str, int] = {}
    for row in iter_jsonl(paths):
        pool = row.get("pool", "internet")
        if pool not in order:
            order[pool] = len(order)
        counts[pool] += 1
        codes.append(order[pool])
    counts.pool_order = order        # type: ignore[attr-defined]
    return counts, codes


def plan_stages(counts: Counter, stages: list[str]
                ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, tuple[int, int]]]]:
    """Turn per-pool counts into a per-stage, per-pool quota.

    Same policy as before, expressed as numbers rather than slices: draw each
    stage's mixture, then give every unconsumed document to its natural stage
    so a corpus that is not mixed in the curriculum's proportions is used
    rather than discarded.
    """
    remaining = dict(counts)
    total_docs = sum(counts.values())
    quota: dict[str, dict[str, int]] = {st: {} for st in stages}
    report: dict[str, dict[str, tuple[int, int]]] = {st: {} for st in stages}

    for stage in stages:
        budget = int(total_docs * STAGE_SHARE.get(stage, 1.0))
        for pool, share in STAGE_MIX[stage].items():
            want = int(budget * share)
            got = min(want, remaining.get(pool, 0))
            if got:
                quota[stage][pool] = quota[stage].get(pool, 0) + got
                remaining[pool] -= got
            report[stage][pool] = (want, got)

    for pool, left in remaining.items():
        if left <= 0:
            continue
        stage = LEFTOVER_STAGE.get(pool, "S2")
        if stage not in quota:
            stage = stages[-1]
        quota[stage][pool] = quota[stage].get(pool, 0) + left
    return quota, report


# Where leftover documents go when a stage could not absorb them. Keeps the
# curriculum's intent: general English early, Gen-Z late.
LEFTOVER_STAGE = {"general": "S1", "internet": "S2", "hinglish": "S2",
                  "lexicon": "S2", "genz": "S3"}


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
    ap.add_argument("--lexicon", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--deadline-seconds", type=int, default=0,
                    help="stop cleanly, close the shards and write the "
                         "manifest before the host kills the session "
                         "(0 = no limit). A killed session exits non-zero and "
                         "publishes nothing, losing the entire run rather "
                         "than degrading it.")
    ap.add_argument("--use-stored-register", action="store_true",
                    help="trust the register written at mining time instead of "
                         "recomputing it (not recommended: the corpus outlives "
                         "the annotator that labelled it)")
    args = ap.parse_args()
    t0 = time.monotonic()

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

    print("\n=== pass 1: counting pools ===")
    counts, codes = count_pools(paths)
    for pool, n in counts.most_common():
        print(f"  {pool:<10} {n:>10,} documents")
    total_docs = sum(counts.values())
    print(f"  {total_docs:,} documents, {len(codes) / 1e6:.0f} MB of pool codes")

    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    all_shards = []
    stats: Counter = Counter()
    register_hist: Counter = Counter()
    lex = Lexicon.load(args.lexicon) if Path(args.lexicon).exists() else None
    if lex is None and not args.no_register:
        print(f"WARNING: no lexicon at {args.lexicon}; slang will score 0")
    annotator = RegisterAnnotator(lex)

    stages = ["S1", "S2", "S3"] if args.split == "train" else ["S1"]

    if args.split == "train":
        print("\n=== curriculum supply vs demand ===")
        quota, report = plan_stages(counts, stages)
        for stage in stages:
            print(f"  {stage}: {sum(quota[stage].values()):,} documents")
            for pool, (want, got) in sorted(report[stage].items()):
                if got < want:
                    print(f"      {pool:<10} short by {want - got:,} "
                          f"(wanted {want:,}, had {got:,})")
    else:
        quota = {"S1": dict(counts)}

    placed = sum(sum(q.values()) for q in quota.values())
    if placed != total_docs:
        print(f"\n  WARNING: {total_docs - placed:,} documents unplaced")

    # NOTE: the S3 "oldest first" ordering of SPEC 11.1 is NOT applied here.
    # It sorted on `created_utc`, which the miner does not carry, so every key
    # was 0 and the sort has always been a no-op. Restoring it needs a
    # timestamp threaded through mining. Recording the gap rather than
    # pretending the ordering happens.

    pool_order = counts.pool_order          # type: ignore[attr-defined]
    code_to_pool = {v: k for k, v in pool_order.items()}
    writers = {
        stage: ShardWriter(out_dir, args.shard_tokens,
                           prefix=f"{args.split}_{stage}", stage=stage)
        for stage in stages if sum(quota.get(stage, {}).values()) > 0
    }
    # Consumed as the second pass runs: a document goes to the first stage
    # that still wants its pool.
    left = {st: dict(q) for st, q in quota.items()}

    print("\n=== pass 2: tokenizing ===")
    batch: dict[str, list[str]] = {st: [] for st in writers}
    BATCH = 1000

    def flush(stage: str) -> None:
        texts = batch[stage]
        if not texts:
            return
        for enc in tok.encode_batch(texts):
            ids = enc.ids + [eos_id]
            writers[stage].add(ids)
            stats["tokens"] += len(ids)
        stats["docs"] += len(texts)
        batch[stage] = []

    for i, row in enumerate(iter_jsonl(paths)):
        pool = code_to_pool.get(codes[i], "internet") if i < len(codes) else "internet"
        stage = next((st for st in stages
                      if left.get(st, {}).get(pool, 0) > 0), None)
        if stage is None or stage not in writers:
            stats["unplaced"] += 1
            continue
        left[stage][pool] -= 1

        text = row["text"]
        if not args.no_register:
            # Recomputed by default. Mining takes ~9h and annotation runs at
            # ~30k docs/sec/core, so labels stored during a mine are older
            # than the annotator by the time shards are built -- and the
            # control tokens are what the model is conditioned on.
            if args.use_stored_register and row.get("register"):
                reg = Register(**row["register"])
            else:
                reg = annotator.annotate(text)
            register_hist[f"slang_{reg.slang}"] += 1
            text = reg.to_tokens() + text

        batch[stage].append(text)
        if len(batch[stage]) >= BATCH:
            flush(stage)
            if stats["docs"] % 500_000 < BATCH:
                print(f"    {stats['docs']:,} docs, "
                      f"{stats['tokens'] / 1e6:.1f}M tokens, "
                      f"{(time.monotonic() - t0) / 60:.0f} min", flush=True)
            if args.deadline_seconds and time.monotonic() - t0 >= args.deadline_seconds:
                # Partial shards are useful; a killed session is not.
                print(f"\n!! deadline reached after {stats['docs']:,} of "
                      f"{total_docs:,} documents -- closing shards", flush=True)
                stats["deadline_hit"] = 1
                break

    for stage in list(writers):
        flush(stage)

    for stage, writer in writers.items():
        shards = writer.close()
        all_shards.extend(shards)
        print(f"  {stage}: {len(shards)} shard(s), "
              f"{sum(sh.n_tokens for sh in shards) / 1e6:.2f}M tokens")

    manifest = Manifest(all_shards, root=out_dir, seq_len=args.seq_len,
                        vocab_size=vocab_size)
    manifest_path = out_dir / "manifest.json"
    manifest.save(manifest_path)

    print("\n=== summary ===")
    if stats.get("deadline_hit"):
        print(f"  PARTIAL: stopped at the deadline; "
              f"{total_docs - stats['docs']:,} documents not tokenized")
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
