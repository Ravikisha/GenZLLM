"""Discover current slang from the corpus, because dictionaries cannot supply it.

Measured 2026-09-17: the largest public Urban Dictionary dump ends 2023-11-09,
and 13 of 19 probed contemporary terms were absent from it (gyatt, delulu,
mogging, looksmaxxing, skibidi, sigma, aura, cooked, bussin, yap, glazing,
pookie, "crash out").

Lexicographers lag usage by years and the dumps lag the lexicographers, so a
dated dictionary is good for deciding what is *dead* and structurally useless
for deciding what is *current*. Current slang is found by contrast instead:
frequent in recent social text, absent from standard English, and either unknown
to the lexicon or rising sharply against an older baseline.

Two corpus slices are counted:
  recent   -- Bittensor Subnet 13 Reddit scrapes (continuously updated)
  baseline -- GECLM Reddit (2006 to Jan 2023), for the growth ratio

Run (sampled, local):
  python pipelines/01_discover_emerging.py --recent-rows 200000 --baseline-rows 200000
Run (full, on a Kaggle CPU session -- costs zero GPU quota):
  python pipelines/01_discover_emerging.py --recent-rows 0 --baseline-rows 2000000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.data.clean import CleanConfig, clean_text, content_words
from bussin.data.lexicon import Lexicon, discover_emerging_terms

RECENT_REPOS = [
    "tensorshield/reddit_dataset_157",
    "coldmind/reddit_dataset_94",
    "wenknow/reddit_dataset_232",
]
BASELINE_REPO = "HuggingFaceGECLM/REDDIT_comments"
BASELINE_SPLITS = ["gaming", "Showerthoughts", "relationship_advice"]

TEXT_KEYS = ("text", "body", "content", "Message", "message", "comment")


def extract_text(row: dict) -> str:
    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def count_stream(repo: str, split: str, max_rows: int, label: str) -> tuple[Counter, int]:
    """Stream a dataset and count lowercased word tokens.

    Streaming matters: these repos are 24-109 GB and we never want them on disk.
    """
    from datasets import load_dataset

    counts: Counter = Counter()
    total = 0
    kept = dropped = 0
    cfg = CleanConfig()
    print(f"  [{label}] streaming {repo} :: {split}", flush=True)
    try:
        ds = load_dataset(repo, split=split, streaming=True)
    except Exception as exc:
        print(f"    unavailable: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return counts, 0

    for i, row in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        # Cleaning must precede counting. Running discovery on raw text
        # returns `https`, `png`, `webp` and `redd` as the top 'emerging
        # slang' -- markup artefacts, not language.
        res = clean_text(extract_text(row), cfg, author=row.get("author"))
        if not res.ok:
            dropped += 1
            continue
        kept += 1
        words = content_words(res.text)
        counts.update(words)
        total += len(words)
        if i and i % 50_000 == 0:
            print(f"    {i:,} rows, {kept:,} kept, {total:,} tokens, "
                  f"{len(counts):,} types", flush=True)
    rate = kept / max(kept + dropped, 1)
    print(f"    done: {kept:,} kept / {kept + dropped:,} rows ({rate:.1%}), "
          f"{total:,} tokens, {len(counts):,} distinct", flush=True)
    return counts, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--out", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--emerging-out", default="data/lexicon/emerging.jsonl")
    ap.add_argument("--recent-rows", type=int, default=200_000, help="0 = all")
    ap.add_argument("--baseline-rows", type=int, default=200_000)
    ap.add_argument("--min-per-million", type=float, default=2.0)
    ap.add_argument("--english-vocab", type=int, default=50_000)
    args = ap.parse_args()

    print("=== 1. recent slice (2024-2026 Reddit) ===")
    recent, recent_total = Counter(), 0
    for repo in RECENT_REPOS:
        c, t = count_stream(repo, "train", args.recent_rows, "recent")
        recent.update(c)
        recent_total += t
        if recent_total and args.recent_rows and recent_total > args.recent_rows * 40:
            break
    if recent_total == 0:
        print("  no recent data reachable; cannot discover emerging terms")
        return 1

    print("\n=== 2. baseline slice (<=2023 Reddit) ===")
    baseline, baseline_total = Counter(), 0
    for split in BASELINE_SPLITS:
        c, t = count_stream(BASELINE_REPO, split, args.baseline_rows, "baseline")
        baseline.update(c)
        baseline_total += t

    print("\n=== 3. standard English vocabulary ===")
    from wordfreq import top_n_list

    english = set(top_n_list("en", args.english_vocab))
    print(f"  {len(english):,} common English words as the negative filter")

    print("\n=== 4. discovery ===")
    lex = Lexicon.load(args.lexicon)
    known = set(lex.by_term)
    emerging = discover_emerging_terms(
        recent_counts=recent,
        recent_total_tokens=recent_total,
        known_terms=known,
        english_vocab=english,
        min_per_million=args.min_per_million,
        baseline_counts=baseline or None,
        baseline_total_tokens=baseline_total or None,
    )
    print(f"  {len(emerging):,} emerging terms found")

    print("\n  top 40 by rate (per million tokens in the recent slice):")
    for e in emerging[:40]:
        old = baseline.get(e.term, 0) * (1e6 / baseline_total) if baseline_total else 0.0
        growth = f"{e.corpus_rate / old:>6.1f}x" if old > 0 else "   new"
        print(f"    {e.term:<20} {e.corpus_rate:>9.2f}/M   was {old:>8.2f}/M  {growth}")

    Path(args.emerging_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.emerging_out, "w", encoding="utf-8") as fh:
        for e in emerging:
            fh.write(e.to_json() + "\n")
    print(f"\n  wrote {args.emerging_out}")

    print("\n=== 5. re-attesting the historical lexicon ===")
    from bussin.data.lexicon import refine_status_from_corpus

    before = lex.stats()
    refine_status_from_corpus(lex.entries, recent, recent_total)
    merged = Lexicon(lex.entries + emerging)
    merged.save(args.out)
    after = merged.stats()
    print(f"  status before: {before}")
    print(f"  status after:  {after}")
    print(f"  wrote {args.out}")

    probe = ["rizz", "gyatt", "delulu", "mogging", "skibidi", "sigma", "aura",
             "cooked", "bussin", "yap", "glazing", "pookie", "mid", "opp"]
    print("\n  contemporary-term coverage after discovery:")
    have = set(merged.by_term)
    for t in probe:
        e = merged.by_term.get(t)
        mark = "PRESENT" if t in have else "MISSING"
        extra = f"  ({e.source}, {e.corpus_rate:.2f}/M, {e.status})" if e else ""
        print(f"    {t:<12} {mark}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
