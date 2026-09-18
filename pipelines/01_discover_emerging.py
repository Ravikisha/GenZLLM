"""Discover current slang from the corpus, because dictionaries cannot supply it.

Measured 2026-09-17: the largest public Urban Dictionary dump ends 2023-11-09,
and 13 of 19 probed contemporary terms were absent from it (gyatt, delulu,
mogging, looksmaxxing, skibidi, sigma, aura, cooked, bussin, yap, glazing,
pookie, "crash out").

Lexicographers lag usage by years and the dumps lag the lexicographers, so a
dated dictionary decides what is *dead* and structurally cannot decide what is
*current*. Current slang is found by contrast against the corpus instead.

## The three failures this script is shaped by

All three looked like success until the output was read.

**1. Discovery on raw text returns markup, not language.** First run's top
"emerging slang" was `https 3166/M`, `png 713/M`, `webp 489/M`, `redd 499/M`.
Cleaning must precede counting.

**2. An un-topic-matched baseline measures topic, not time.** After cleaning,
the top became `anyone (3.3x)`, `currently (5.6x)`, `wondering (6.2x)`. Nothing
about those words changed; the *subreddit mix* did.

**3. The recent slice was not Gen-Z.** The Bittensor SN13 scrapes are an
unweighted long tail -- in a 3,000-row sample the most common community was
r/ASUSROG at 23 rows, alongside r/FAFSA, r/Wealthsimple and r/ConselhosLegais
(Portuguese). Mining that for Gen-Z slang finds `wifi`, `iphone`, `salary`
and `mais`, which is a correct answer to the wrong question.

The fix for (3) is not a hand-written subreddit allowlist, which would bake in
guesses about where Gen-Z posts. Instead the **register annotator ranks the
communities**: sample rows, group by `communityName`, score each community's
mean register, and mine only the high-register ones. The same measurement that
conditions the model also selects its data.

Run (sampled, local):
  python pipelines/01_discover_emerging.py --rank-rows 60000 --recent-rows 250000
Run (full, Kaggle CPU -- zero GPU quota):
  python pipelines/01_discover_emerging.py --rank-rows 500000 --recent-rows 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.data.clean import CleanConfig, clean_text, content_words
from bussin.data.lexicon import Lexicon, discover_emerging_terms
from bussin.data.register import RegisterAnnotator

RECENT_REPOS = [
    "tensorshield/reddit_dataset_157",
    "coldmind/reddit_dataset_94",
    "wenknow/reddit_dataset_232",
]
BASELINE_REPO = "HuggingFaceGECLM/REDDIT_comments"
BASELINE_SPLITS = ["gaming", "Showerthoughts", "relationship_advice"]

TEXT_KEYS = ("text", "body", "content", "Message", "message", "comment")
COMMUNITY_KEYS = ("communityName", "label", "subreddit", "community")

MIN_COMMUNITY_ROWS = 8      # below this the mean register is noise


def extract_text(row: dict) -> str:
    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def extract_community(row: dict) -> str:
    for k in COMMUNITY_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v.lower().lstrip("r/").strip("/")
    return ""


def stream(repo: str, split: str, max_rows: int):
    from datasets import load_dataset

    try:
        ds = load_dataset(repo, split=split, streaming=True,
                          encoding="utf-8", encoding_errors="replace")
    except (TypeError, ValueError):
        ds = load_dataset(repo, split=split, streaming=True)
    it = iter(ds)
    i = 0
    while not max_rows or i < max_rows:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as exc:
            print(f"    row error: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            return
        i += 1


def rank_communities(repos: list[str], annotator: RegisterAnnotator,
                     max_rows: int, top_k: int) -> tuple[set[str], list[tuple]]:
    """Score every community by mean measured register, keep the top ones.

    This replaces a hand-written subreddit allowlist. The register annotator is
    already the project's definition of "how Gen-Z is this text", so using it to
    choose communities keeps data selection and model conditioning consistent.
    """
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    cfg = CleanConfig()
    scanned = 0

    for repo in repos:
        print(f"  ranking from {repo}", flush=True)
        for row in stream(repo, "train", max_rows // len(repos)):
            scanned += 1
            community = extract_community(row)
            if not community:
                continue
            res = clean_text(extract_text(row), cfg)
            if not res.ok or res.n_words < 5:
                continue
            reg = annotator.annotate(res.text)
            acc = sums[community]
            acc[0] += reg.slang
            acc[1] += reg.abbrev + reg.emoji + reg.elong
            acc[2] += 3 - reg.formal
            acc[3] += 1
            if scanned % 50_000 == 0:
                print(f"    {scanned:,} rows, {len(sums):,} communities", flush=True)

    scored = []
    for community, (slang, informal, casual, n) in sums.items():
        if n < MIN_COMMUNITY_ROWS:
            continue
        # Weighted so slang density dominates but a community cannot score high
        # on slang alone while writing like a press release.
        score = (2.0 * slang / n) + (1.0 * informal / n) + (1.0 * casual / n)
        scored.append((community, score, n))
    scored.sort(key=lambda x: -x[1])
    keep = {c for c, _, _ in scored[:top_k]}
    return keep, scored


def count_recent(repos: list[str], keep: set[str], max_rows: int,
                 annotator: RegisterAnnotator) -> tuple[Counter, int, Counter]:
    counts: Counter = Counter()
    per_community: Counter = Counter()
    total = 0
    cfg = CleanConfig()
    for repo in repos:
        print(f"  counting {repo}", flush=True)
        kept = seen = 0
        for row in stream(repo, "train", max_rows // len(repos) if max_rows else 0):
            seen += 1
            if keep and extract_community(row) not in keep:
                continue
            res = clean_text(extract_text(row), cfg)
            if not res.ok:
                continue
            kept += 1
            words = content_words(res.text)
            counts.update(words)
            total += len(words)
            per_community[extract_community(row)] += 1
        print(f"    {seen:,} scanned -> {kept:,} in high-register communities, "
              f"{total:,} tokens", flush=True)
    return counts, total, per_community


def count_baseline(max_rows: int) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    cfg = CleanConfig()
    for split in BASELINE_SPLITS:
        print(f"  baseline {BASELINE_REPO}::{split}", flush=True)
        for row in stream(BASELINE_REPO, split, max_rows // len(BASELINE_SPLITS)):
            res = clean_text(extract_text(row), cfg, author=row.get("author"))
            if not res.ok:
                continue
            words = content_words(res.text)
            counts.update(words)
            total += len(words)
    print(f"    {total:,} baseline tokens", flush=True)
    return counts, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--out", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--emerging-out", default="data/lexicon/emerging.jsonl")
    ap.add_argument("--communities-out", default="data/lexicon/communities.json")
    ap.add_argument("--rank-rows", type=int, default=60_000)
    ap.add_argument("--recent-rows", type=int, default=250_000, help="0 = all")
    ap.add_argument("--baseline-rows", type=int, default=120_000)
    ap.add_argument("--top-communities", type=int, default=400)
    ap.add_argument("--min-per-million", type=float, default=3.0)
    ap.add_argument("--english-vocab", type=int, default=50_000)
    ap.add_argument("--skip-ranking", action="store_true")
    args = ap.parse_args()

    lex = Lexicon.load(args.lexicon)
    annotator = RegisterAnnotator(lex)

    print("=== 1. ranking communities by measured register ===")
    keep: set[str] = set()
    if not args.skip_ranking:
        keep, scored = rank_communities(RECENT_REPOS, annotator, args.rank_rows,
                                        args.top_communities)
        Path(args.communities_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.communities_out).write_text(
            json.dumps([{"community": c, "score": round(s, 3), "n": n}
                        for c, s, n in scored], indent=1), encoding="utf-8")
        print(f"\n  {len(scored):,} communities scored, keeping top {len(keep):,}")
        print("\n  highest-register communities (these get mined):")
        for c, s, n in scored[:20]:
            print(f"    r/{c:<30} score {s:>5.2f}  ({n} rows)")
        print("\n  lowest-register communities (these get skipped):")
        for c, s, n in scored[-10:]:
            print(f"    r/{c:<30} score {s:>5.2f}  ({n} rows)")

    print("\n=== 2. counting the high-register recent slice ===")
    recent, recent_total, per_community = count_recent(
        RECENT_REPOS, keep, args.recent_rows, annotator)
    if recent_total == 0:
        print("  no data; cannot discover")
        return 1

    print("\n=== 3. baseline slice (<=2023) ===")
    baseline, baseline_total = count_baseline(args.baseline_rows)

    print("\n=== 4. standard English vocabulary ===")
    from wordfreq import top_n_list

    english = set(top_n_list("en", args.english_vocab))
    print(f"  {len(english):,} common English words as the negative filter")

    print("\n=== 5. discovery ===")
    emerging = discover_emerging_terms(
        recent_counts=recent, recent_total_tokens=recent_total,
        known_terms=set(lex.by_term), english_vocab=english,
        min_per_million=args.min_per_million,
        baseline_counts=baseline or None,
        baseline_total_tokens=baseline_total or None,
    )
    print(f"  {len(emerging):,} emerging terms found "
          f"from {recent_total:,} high-register tokens")

    print("\n  top 40 by rate:")
    for e in emerging[:40]:
        old = baseline.get(e.term, 0) * (1e6 / baseline_total) if baseline_total else 0.0
        growth = f"{e.corpus_rate / old:>6.1f}x" if old > 0 else "   new"
        print(f"    {e.term:<20} {e.corpus_rate:>9.2f}/M   was {old:>8.2f}/M  {growth}")

    Path(args.emerging_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.emerging_out, "w", encoding="utf-8") as fh:
        for e in emerging:
            fh.write(e.to_json() + "\n")

    print("\n=== 6. re-attesting the historical lexicon ===")
    from bussin.data.lexicon import refine_status_from_corpus

    before = lex.stats()
    refine_status_from_corpus(lex.entries, recent, recent_total)
    merged = Lexicon(lex.entries + emerging)
    merged.save(args.out)
    print(f"  before: {before}")
    print(f"  after:  {merged.stats()}")

    probe = ["rizz", "gyatt", "delulu", "mogging", "skibidi", "sigma", "aura",
             "cooked", "bussin", "yap", "glazing", "pookie", "mid", "opp", "npc"]
    print("\n  contemporary-term coverage:")
    hit = 0
    for t in probe:
        e = merged.by_term.get(t)
        if e:
            hit += 1
            print(f"    {t:<12} PRESENT  ({e.source}, {e.corpus_rate:.2f}/M, {e.status})")
        else:
            print(f"    {t:<12} MISSING")
    print(f"\n  coverage: {hit}/{len(probe)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
