"""Register annotator calibration (SPEC §16 Phase 1 gate (a)).

The gate: an independent rater scores documents 0-4 on slang density; the
annotator must reach Spearman rho >= 0.7 against them.

Why it matters: the annotator assigns the register vector that *every* training
document is conditioned on. If it is biased, the model learns the bias instead
of the language, and nothing downstream reveals that until the model is done.

Two modes:

  emit    sample documents, stratified across the annotator's own buckets, and
          print them with ids and the annotator's score hidden
  score   read a ratings file and report Spearman rho, per-bucket agreement and
          the worst disagreements

Stratifying by the annotator's buckets is deliberate: a random sample of this
corpus is overwhelmingly slang_0, and a correlation computed on it would look
excellent while telling you nothing about the high-slang end.

Run:
  python scripts/register_calibration.py emit  --n 50 > sheet.txt
  python scripts/register_calibration.py score --ratings ratings.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.data.lexicon import Lexicon
from bussin.data.register import RegisterAnnotator

SHEET = "data/audit/register_sheet.json"


def iter_docs(corpus: Path):
    for p in sorted(corpus.rglob("*.jsonl.gz")) + sorted(corpus.rglob("*.jsonl")):
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def spearman(a: list[float], b: list[float]) -> float:
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


# Stratify by SOURCE, never by the annotator's own output.
#
# The first calibration attempt sampled equally from each of the annotator's
# slang buckets, which is circular: when the annotator was broken its "slang_4"
# bucket was full of router posts, so the strata were random with respect to
# real slang and the test could not have worked.
#
# Source is independent evidence. fineweb-edu is edited prose, Twitch and
# Discord are chat. That gives the scale genuine range regardless of what the
# annotator currently believes.
SOURCE_STRATA = {
    "formal":  ["fineweb-edu", "smollm-corpus", "cosmopedia"],
    "prose":   ["REDDIT_comments"],
    "social":  ["reddit_dataset_157", "reddit_dataset_232", "reddit_dataset_94",
                "youtube-comment-sentiment"],
    "chat":    ["twitch_chat", "discord-messages"],
}


def _stratum(source: str) -> str | None:
    tail = (source or "").split("/")[-1]
    for name, members in SOURCE_STRATA.items():
        if tail in members:
            return name
    return None


def cmd_emit(args) -> int:
    lex = Lexicon.load(args.lexicon) if Path(args.lexicon).exists() else None
    ann = RegisterAnnotator(lex)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for doc in iter_docs(Path(args.corpus)):
        text = doc.get("text", "")
        if not (40 <= len(text) <= 600):
            continue
        stratum = _stratum(doc.get("source", ""))
        if stratum is None:
            continue
        reg = ann.annotate(text)
        buckets[stratum].append({"text": text, "pool": doc.get("pool"),
                                 "source": doc.get("source"),
                                 "stratum": stratum, "slang": reg.slang})

    rng = random.Random(args.seed)
    per = max(args.n // len(SOURCE_STRATA), 1)
    sample: list[dict] = []
    for name in SOURCE_STRATA:
        pool = buckets.get(name, [])
        rng.shuffle(pool)
        sample.extend(pool[:per])
    rng.shuffle(sample)
    for i, s in enumerate(sample):
        s["id"] = i

    out = Path(SHEET)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"# {len(sample)} documents, stratified by SOURCE (not by annotator output)")
    print("# rate each 0-4 for SLANG DENSITY only -- not emoji, not abbreviations")
    print("#   0 none   1 a trace   2 noticeable   3 heavy   4 saturated")
    print(f"# annotator scores hidden; written to {out}\n")
    for s in sample:
        print(f"[{s['id']:02d}] {' '.join(s['text'].split())[:args.width]}")
    print(f"\n# available per stratum: "
          f"{ {k: len(v) for k, v in buckets.items()} }")
    return 0


def cmd_score(args) -> int:
    sheet = json.loads(Path(SHEET).read_text(encoding="utf-8"))
    ratings = json.loads(Path(args.ratings).read_text(encoding="utf-8"))
    ratings = {int(k): int(v) for k, v in ratings.items()}

    pairs = [(s["slang"], ratings[s["id"]]) for s in sheet if s["id"] in ratings]
    if len(pairs) < 10:
        raise SystemExit(f"only {len(pairs)} rated; need at least 10")

    auto = [p[0] for p in pairs]
    human = [p[1] for p in pairs]
    rho = spearman(auto, human)
    exact = sum(1 for a, h in pairs if a == h) / len(pairs)
    within1 = sum(1 for a, h in pairs if abs(a - h) <= 1) / len(pairs)
    bias = sum(a - h for a, h in pairs) / len(pairs)

    print(f"=== register calibration: {len(pairs)} documents ===")
    print(f"  Spearman rho      {rho:.3f}   (gate: >= 0.70)")
    print(f"  exact agreement   {exact:.1%}")
    print(f"  within +/-1       {within1:.1%}")
    print(f"  mean bias         {bias:+.2f}   (annotator minus rater)")

    print("\n  confusion (rows = annotator, cols = rater):")
    grid = [[0] * 5 for _ in range(5)]
    for a, h in pairs:
        grid[a][h] += 1
    print("        " + "".join(f"{c:>5}" for c in range(5)))
    for r in range(5):
        print(f"    {r}   " + "".join(f"{grid[r][c]:>5}" for c in range(5)))

    worst = sorted(
        ((abs(s["slang"] - ratings[s["id"]]), s) for s in sheet if s["id"] in ratings),
        key=lambda kv: -kv[0])[:8]
    print("\n  worst disagreements:")
    for gap, s in worst:
        if gap == 0:
            break
        print(f"    gap {gap}  auto={s['slang']} rater={ratings[s['id']]}  "
              f"pool={s['pool']}")
        print(f"      {' '.join(s['text'].split())[:150]}")

    passed = rho >= 0.70
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  Do not commit the corpus. The annotator's scores are what every")
        print("  training document is conditioned on; a biased annotator teaches")
        print("  the bias rather than the language.")
    Path("data/audit/register_calibration.json").write_text(
        json.dumps({"n": len(pairs), "spearman": rho, "exact": exact,
                    "within1": within1, "bias": bias, "passed": passed}, indent=1),
        encoding="utf-8")
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit")
    e.add_argument("--corpus", default="data/corpus/train")
    e.add_argument("--lexicon", default="data/lexicon/lexicon.jsonl")
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--width", type=int, default=210)
    e.add_argument("--seed", type=int, default=20260918)
    e.set_defaults(func=cmd_emit)

    s = sub.add_parser("score")
    s.add_argument("--ratings", default="data/audit/ratings.json")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
