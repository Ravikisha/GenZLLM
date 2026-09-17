"""Build the dated slang lexicon.

Sources (row counts verified against the HF datasets-server):
  * georgiyozhegov/urbandictionary -- 152,941 definitions WITH `time` and `score`.
    The timestamps are the whole point: they are what let us separate current
    slang from dead slang mechanically instead of by hand.
  * MLBtrio/genz-slang-dataset -- 1,779 curated entries, merged as trusted seeds.

Output: data/lexicon/lexicon.jsonl  (one LexEntry per line)

Licence: Urban Dictionary content is CC-BY-SA-4.0. We use this as a *lexicon
for filtering* -- which words exist and when they appeared, which is not itself
copyrightable expression -- rather than as training text, keeping the
share-alike obligation off the model weights. See SPEC §1.12.

Run:  python pipelines/00_build_lexicon.py --out data/lexicon/lexicon.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.data.lexicon import Lexicon, build_lexicon, merge_curated

UD_REPO = "georgiyozhegov/urbandictionary"
CURATED_REPO = "MLBtrio/genz-slang-dataset"


def _sanitize(name: str) -> str:
    """Windows forbids : * ? " < > | in filenames, and several HF datasets ship
    files with timestamps in the name. Map them to a safe local name."""
    out = name
    for ch in ':*?"<>|':
        out = out.replace(ch, "_")
    return out


def download_repo_files(repo: str, cache: Path, exts=(".jsonl", ".json", ".csv", ".parquet")) -> list[Path]:
    """Fetch a dataset repo's data files directly, bypassing the `datasets`
    cache layer (which cannot write colon-bearing filenames on Windows)."""
    import urllib.request

    from huggingface_hub import HfApi, hf_hub_url

    cache.mkdir(parents=True, exist_ok=True)
    files = [
        f for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.lower().endswith(exts) and not f.startswith(".")
    ]
    out: list[Path] = []
    for remote in files:
        local = cache / _sanitize(Path(remote).name)
        if not local.exists() or local.stat().st_size == 0:
            url = hf_hub_url(repo, remote, repo_type="dataset")
            print(f"    downloading {remote} -> {local.name}", flush=True)
            urllib.request.urlretrieve(url, local)
        out.append(local)
    return out


def _read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if suffix == ".csv":
        import csv

        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    rows = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":                       # a single JSON array
            return json.load(fh)
        for line in fh:                        # JSON lines
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_hf_rows(repo: str, split: str = "train") -> list[dict]:
    print(f"  loading {repo} ...", flush=True)
    try:
        from datasets import load_dataset

        ds = load_dataset(repo, split=split)
        print(f"    {len(ds):,} rows, columns {ds.column_names}", flush=True)
        return ds
    except OSError as exc:
        # Windows filename restriction, almost always. Fall back to raw fetch.
        print(f"    datasets loader failed ({type(exc).__name__}); "
              f"falling back to direct download", flush=True)

    cache = Path("data/_raw") / repo.replace("/", "__")
    rows: list[dict] = []
    for path in download_repo_files(repo, cache):
        rows.extend(_read_rows(path))
    if not rows:
        raise RuntimeError(f"no rows recovered from {repo}")
    print(f"    {len(rows):,} rows, columns {sorted(rows[0].keys())}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--min-score", type=int, default=1)
    ap.add_argument("--ud-repo", default=UD_REPO)
    ap.add_argument("--curated-repo", default=CURATED_REPO)
    ap.add_argument("--skip-curated", action="store_true")
    args = ap.parse_args()

    print("=== 1. Urban Dictionary (dated) ===")
    ud = load_hf_rows(args.ud_repo)
    raw_n = len(ud)

    print("\n=== 2. aggregating per term ===")
    entries = build_lexicon(ud, min_score=args.min_score)
    print(f"  {raw_n:,} definitions -> {len(entries):,} unique terms "
          f"({100 * (1 - len(entries) / max(raw_n, 1)):.1f}% removed as junk, "
          f"low-score, or merged)")

    if not args.skip_curated:
        print("\n=== 3. merging curated seeds ===")
        try:
            curated = load_hf_rows(args.curated_repo)
            before = len(entries)
            entries = merge_curated(entries, curated)
            print(f"  {len(entries) - before:,} new terms added, "
                  f"{len(curated) - (len(entries) - before):,} already present")
        except Exception as exc:
            print(f"  skipped: {exc}")

    lex = Lexicon(entries)
    lex.save(args.out)

    print("\n=== 4. lexicon summary ===")
    stats = lex.stats()
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v:>8,}")

    years: Counter = Counter(e.first_seen[:4] for e in entries)
    print("\n  first-attestation histogram (the recency tail is what matters):")
    for year in sorted(years):
        if year < "2009":
            continue
        bar = "#" * max(1, int(60 * years[year] / max(years.values())))
        print(f"    {year}  {years[year]:>6,}  {bar}")

    print("\n  single-word current terms (tokenizer forcing set): "
          f"{len(lex.single_word_current()):,}")

    print("\n  sample CURRENT terms:")
    for e in [e for e in entries if e.status == "current"][:12]:
        print(f"    {e.term:<22} {e.first_seen}  score {e.median_score:>6.0f}  "
              f"w={e.recency_weight:.2f}  {e.definition[:60]}")

    print("\n  sample DEAD terms (these become the outdated-slang eval, SPEC §14.13):")
    for e in [e for e in entries if e.status == "dead"][:8]:
        print(f"    {e.term:<22} {e.first_seen}  w={e.recency_weight:.3f}")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
