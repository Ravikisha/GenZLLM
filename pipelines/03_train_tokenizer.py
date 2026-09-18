"""Train the Bussin byte-level BPE tokenizer, then gate it.

The tokenizer is trained on a **deliberately over-weighted** sample: 35% Gen-Z
core against its ~8% share of the real mixture. Training it on the mixture
proportions would optimise merges for prose and under-serve exactly the text
this project exists to model (SPEC §6.4).

Run:
  python pipelines/03_train_tokenizer.py --corpus data/corpus/train \
      --vocab-size 49152 --out tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.tokenizer.train_tokenizer import (
    build_forced_tokens,
    check_gate,
    count_emoji,
    evaluate_tokenizer,
    train_tokenizer,
)

# Over-weighted on purpose. See SPEC §6.4.
TOKENIZER_MIX = {"genz": 0.35, "internet": 0.25, "general": 0.30, "hinglish": 0.07}


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


def sample_corpus(corpus_dir: Path, max_bytes: int, seed: int = 13) -> tuple[list[str], dict]:
    """Draw a mixture-weighted sample, capped at `max_bytes` of text.

    Beyond ~5 GB the BPE merges have converged and further data only costs CPU.
    """
    files = sorted(corpus_dir.rglob("*.jsonl.gz")) + sorted(corpus_dir.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"no jsonl files under {corpus_dir}")

    budgets = {pool: int(max_bytes * w) for pool, w in TOKENIZER_MIX.items()}
    got: Counter = Counter()
    out: list[str] = []
    rng = random.Random(seed)

    for row in iter_jsonl(files):
        pool = row.get("pool", "internet")
        if pool not in budgets:
            pool = "internet"
        if got[pool] >= budgets.get(pool, 0):
            if all(got[p] >= budgets[p] for p in budgets):
                break
            continue
        text = row.get("text", "")
        if not text:
            continue
        out.append(text)
        got[pool] += len(text.encode("utf-8"))

    rng.shuffle(out)
    return out, dict(got)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/train")
    ap.add_argument("--lexicon", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--vocab-size", type=int, default=49152)
    ap.add_argument("--out", default="tokenizer/tokenizer.json")
    ap.add_argument("--max-sample-bytes", type=int, default=5_000_000_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    print("=== 1. sampling the tokenizer corpus (mixture-weighted) ===")
    texts, got = sample_corpus(Path(args.corpus), args.max_sample_bytes)
    total = sum(got.values()) or 1
    for pool, n in sorted(got.items(), key=lambda kv: -kv[1]):
        print(f"  {pool:<10} {n / 1e6:>9.2f} MB  ({100 * n / total:>5.1f}%  "
              f"target {100 * TOKENIZER_MIX.get(pool, 0):.0f}%)")
    print(f"  {len(texts):,} documents, {total / 1e6:.1f} MB total")

    print("\n=== 2. forced vocabulary ===")
    emoji_counts = count_emoji(texts, limit=200_000)
    forced = build_forced_tokens(args.lexicon, emoji_counts, top_emoji=1000)
    print(f"  {len(forced):,} forced tokens "
          f"(current slang + abbreviations + emotes + Hinglish + "
          f"{len(emoji_counts):,} distinct emoji seen)")

    print("\n=== 3. training BPE ===")
    tok = train_tokenizer(
        corpus=iter(texts), vocab_size=args.vocab_size, out_path=args.out,
        forced_tokens=forced, min_frequency=args.min_frequency,
    )

    print("\n=== 4. gate (SPEC §6.8) ===")
    samples: dict[str, list[str]] = {"general": [], "genz": [], "hinglish": []}
    files = sorted(Path(args.corpus).rglob("*.jsonl.gz"))
    for row in iter_jsonl(files):
        pool = row.get("pool", "internet")
        key = pool if pool in samples else ("genz" if pool == "internet" else None)
        if key and len(samples[key]) < 2000:
            samples[key].append(row["text"])
        if all(len(v) >= 2000 for v in samples.values()):
            break

    from bussin.data.lexicon import Lexicon

    lex = Lexicon.load(args.lexicon) if Path(args.lexicon).exists() else None
    forced_terms = lex.single_word_current()[:500] if lex else []

    report = evaluate_tokenizer(tok, samples, forced_terms)
    for k, v in sorted(report.items()):
        print(f"  {k:<32} {v:.4f}")

    ok, failures = check_gate(report)
    print(f"\n  gate: {'PASS' if ok else 'FAIL'}")
    for f in failures:
        print(f"    - {f}")
    if not ok:
        print("\n  Do not tokenize the full corpus until the gate passes.")
    Path(args.out).with_name("tokenizer_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
