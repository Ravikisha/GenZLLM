"""PII audit over the post-pipeline corpus (SPEC §16 Phase 1 gate).

The gate: sample real documents *after* cleaning and read them. Any real name,
handle, email, phone number or address means the pipeline failed and must be
fixed before the corpus is committed to.

Two passes:

1. **Mechanical** -- run detectors over every document, independently of the
   cleaner, so a bug in the cleaner cannot hide from its own regexes. These are
   deliberately broader than `clean.py`'s (they flag things the cleaner is
   *supposed* to have already removed) and will over-flag.
2. **Review** -- print every flagged document plus a random sample of clean
   ones, so a reader can judge whether the flags are real.

The detectors here are intentionally *not* imported from `clean.py`. Auditing a
scrubber with its own patterns only proves it is self-consistent.

Run:
  python scripts/pii_audit.py --corpus data/corpus/train --sample 1000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Independent detectors. Broader than the cleaner's on purpose.
DETECTORS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+\s?(?:@|\[at\]|\(at\))\s?[\w-]+\.\w{2,}\b", re.I),
    "phone": re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{3,5}[\s.-])\d{3}[\s.-]?\d{3,4}(?!\d)"
    ),
    "url_with_user": re.compile(
        r"(?:twitter\.com|x\.com|instagram\.com|github\.com|linkedin\.com/in|"
        r"facebook\.com|t\.me|reddit\.com/u)/[A-Za-z0-9_.-]{3,}", re.I),
    "handle": re.compile(r"(?<![\w/])@[A-Za-z][A-Za-z0-9_]{3,29}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card_like": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "postcode_uk": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"),
    "zip_us": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "street_address": re.compile(
        r"\b\d{1,5}\s+[A-Z][a-z]+\s+(street|st|road|rd|avenue|ave|lane|ln|drive|"
        r"dr|boulevard|blvd|way|court|ct)\b", re.I),
    "dob": re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[012])[/-](?:19|20)\d\d\b"),
}

# "My name is X" style self-identification, which column-dropping cannot catch.
NAME_CONTEXT = re.compile(
    r"\b(?:my name(?:'s| is)|i'm called|this is|signed,|regards,|sincerely,|"
    r"contact|call me|reach me at|dm me at)\s+([A-Z][a-z]{2,})\b")

# Placeholders the cleaner inserts. Finding these is success, not a leak.
PLACEHOLDERS = re.compile(r"<\|(?:email|phone|address|number|redacted)\|>")

# Things that look like PII but are not, in this corpus.
BENIGN_ZIP_CONTEXT = re.compile(r"\b(?:19|20)\d{2}\b")   # bare years


def iter_docs(corpus: Path):
    files = sorted(corpus.rglob("*.jsonl.gz")) + sorted(corpus.rglob("*.jsonl"))
    for p in files:
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def scan(text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name, pat in DETECTORS.items():
        found = [m.group(0) for m in pat.finditer(text)]
        if name == "zip_us":
            # A bare 5-digit number is almost always a year, score or count here.
            found = [f for f in found if not BENIGN_ZIP_CONTEXT.fullmatch(f)]
        if name == "handle":
            # @user is the cleaner's own placeholder for a stripped mention.
            found = [f for f in found if f.lower() != "@user"]
        if found:
            hits[name] = found[:5]
    names = [m.group(1) for m in NAME_CONTEXT.finditer(text)]
    if names:
        hits["name_context"] = names[:5]
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/train")
    ap.add_argument("--sample", type=int, default=1000,
                    help="documents to read in the review pass")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", default="data/audit/pii_audit.json")
    ap.add_argument("--show", type=int, default=40,
                    help="max flagged documents to print")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"no corpus at {corpus}")

    print("=== pass 1: mechanical scan over every document ===")
    total = 0
    flagged: list[dict] = []
    counts: Counter = Counter()
    placeholder_docs = 0

    for doc in iter_docs(corpus):
        total += 1
        text = doc.get("text", "")
        if PLACEHOLDERS.search(text):
            placeholder_docs += 1
        hits = scan(text)
        if hits:
            for k in hits:
                counts[k] += 1
            flagged.append({"text": text, "hits": hits,
                            "pool": doc.get("pool"), "source": doc.get("source")})

    print(f"  scanned          {total:,} documents")
    print(f"  with placeholders{placeholder_docs:>8,}  "
          f"({100 * placeholder_docs / max(total, 1):.2f}%)  "
          f"<- cleaner did substitute")
    print(f"  flagged          {len(flagged):>8,}  "
          f"({100 * len(flagged) / max(total, 1):.2f}%)")
    print("\n  by detector:")
    for name, n in counts.most_common():
        print(f"    {name:<16} {n:>6,}  ({100 * n / max(total, 1):.2f}%)")

    print(f"\n=== pass 2: review ({min(len(flagged), args.show)} flagged shown) ===")
    rng = random.Random(args.seed)
    rng.shuffle(flagged)
    for i, f in enumerate(flagged[:args.show]):
        kinds = ", ".join(f"{k}={v}" for k, v in f["hits"].items())
        snippet = " ".join(f["text"].split())[:240]
        print(f"\n  [{i:02d}] pool={f['pool']} | {kinds}")
        print(f"       {snippet}")

    print(f"\n=== random clean sample ({args.sample} scanned, 12 shown) ===")
    clean_docs = []
    for doc in iter_docs(corpus):
        if not scan(doc.get("text", "")):
            clean_docs.append(doc)
        if len(clean_docs) >= args.sample:
            break
    for d in rng.sample(clean_docs, min(12, len(clean_docs))):
        print(f"  [{d.get('pool')}] {' '.join(d['text'].split())[:170]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "total": total, "flagged": len(flagged),
        "placeholder_docs": placeholder_docs,
        "by_detector": dict(counts),
        "flag_rate": len(flagged) / max(total, 1),
        "examples": flagged[:200],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")

    print("\n=== gate ===")
    print("  The detectors over-flag by design. The question is not the flag")
    print("  count but whether any flagged text contains a REAL identifier.")
    print("  Read the samples above before accepting this corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
