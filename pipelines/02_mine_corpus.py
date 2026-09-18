"""Mine, clean, register-annotate and deduplicate the training corpus.

Streams every source -- nothing is ever downloaded whole, because the Reddit
corpora alone are 109 GB and 291 GB. Designed to run on a Kaggle CPU batch
session, which costs zero GPU quota and can be fanned out five ways.

Pipeline order matters and follows SPEC §5.14. In particular:
  * PII columns are dropped at ingest, before anything touches disk
  * dedup runs BEFORE the train/val split, so near-duplicates cannot leak
  * register annotation runs AFTER the split, so annotation statistics
    computed over training data cannot leak into validation

Output: sharded JSONL of {text, register, pool, stage} ready for tokenization.

Run (local smoke test):
  python pipelines/02_mine_corpus.py --target-tokens 5_000_000 --out data/corpus
Run (Kaggle CPU, one of five):
  python pipelines/02_mine_corpus.py --shard-index 0 --n-shards 5 \
      --target-tokens 7_000_000_000 --out /kaggle/working/corpus
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xxhash

from bussin.data.clean import CleanConfig, clean_text, content_words
from bussin.data.lexicon import Lexicon
from bussin.data.register import RegisterAnnotator

# PII columns dropped at ingest. Never written to disk in any form (SPEC §1.12).
DROP_COLUMNS = {
    "author", "author_fullname", "author_flair_text", "permalink", "id",
    "subreddit_id", "link_id", "parent_id", "name", "CommentID", "VideoID",
    "VideoTitle", "user", "username", "user_id", "channel_id", "server_id",
    "AuthorName", "AuthorChannelID", "author_name", "display_name",
}


@dataclass
class Source:
    repo: str
    split: str
    pool: str
    weight: float                 # share of the target token budget
    config: str | None = None     # HF dataset config, where the repo needs one
    text_key: str | None = None   # explicit column, incl. "a.b" for nested
    prefer_human: bool = False    # findnitai carries source=1 for human-annotated


# Weights follow the 400m mixture in SPEC §4.4, flattened across stages.
# Stage assignment happens later, at shard-ordering time.
SOURCES: list[Source] = [
    # --- Gen-Z core (the scarce pool: only ~3.3B tokens exist in total) ---
    Source("tensorshield/reddit_dataset_157", "train", "genz", 0.040),
    Source("coldmind/reddit_dataset_94", "train", "genz", 0.020),
    Source("wenknow/reddit_dataset_232", "train", "genz", 0.015),
    Source("lparkourer10/twitch_chat", "train", "genz", 0.008),
    Source("llmtraining-scraper/discord-messages", "train", "genz", 0.008),
    Source("AmaanP314/youtube-comment-sentiment", "train", "genz", 0.004,
           text_key="CommentText"),
    # --- internet broad ---
    Source("HuggingFaceGECLM/REDDIT_comments", "gaming", "internet", 0.020),
    Source("HuggingFaceGECLM/REDDIT_comments", "relationship_advice", "internet", 0.017),
    Source("HuggingFaceGECLM/REDDIT_comments", "Showerthoughts", "internet", 0.016),
    Source("HuggingFaceGECLM/REDDIT_comments", "Games", "internet", 0.011),
    Source("HuggingFaceGECLM/REDDIT_comments", "mildlyinteresting", "internet", 0.009),
    Source("HuggingFaceGECLM/REDDIT_comments", "technology", "internet", 0.008),
    Source("HuggingFaceGECLM/REDDIT_comments", "explainlikeimfive", "internet", 0.007),
    Source("HuggingFaceGECLM/REDDIT_comments", "AskHistorians", "internet", 0.005),  # formal anchor
    # --- Hinglish ---
    # Text is nested under `translation`; `source==1` marks human annotation.
    Source("findnitai/english-to-hinglish", "train", "hinglish", 0.006,
           text_key="translation.hi_ng", prefer_human=True),
    Source("Abhishekcr448/Hinglish-Everyday-Conversations-1M", "train", "hinglish",
           0.004, text_key="output"),
    # --- general English (the majority; 79% in the final mixture) ---
    # smollm-corpus has no default config; fineweb-edu-dedup is the web half
    # and cosmopedia-v2 the synthetic-textbook half (SPEC §4.4).
    Source("HuggingFaceTB/smollm-corpus", "train", "general", 0.500,
           config="fineweb-edu-dedup"),
    Source("HuggingFaceTB/smollm-corpus", "train", "general", 0.100,
           config="cosmopedia-v2"),
    Source("HuggingFaceFW/fineweb-edu", "train", "general", 0.202,
           config="sample-10BT"),
]

TEXT_KEYS = ("text", "body", "content", "Message", "message", "comment",
             "Comment", "CommentText", "comment_text", "definition", "hi_ng",
             "output", "tweet", "post", "selftext", "review")


class Deduper:
    """Exact dedup by hash, plus frequency capping for short chat messages.

    Short chat is deliberately NOT deduplicated. 'W', 'KEKW' and 'lets go'
    repeat legitimately thousands of times and are real linguistic events;
    deleting them removes the register. They are frequency-capped instead
    (SPEC §5.1 pass 3).
    """

    def __init__(self, short_cap: int = 5000, short_len: int = 10) -> None:
        self.seen: set[int] = set()
        self.short_counts: Counter = Counter()
        self.short_cap = short_cap
        self.short_len = short_len
        self.dropped_exact = 0
        self.dropped_capped = 0

    def accept(self, text: str, n_words: int) -> bool:
        if n_words <= self.short_len:
            key = text.lower().strip()
            self.short_counts[key] += 1
            if self.short_counts[key] > self.short_cap:
                self.dropped_capped += 1
                return False
            return True
        h = xxhash.xxh64(" ".join(text.lower().split()).encode("utf-8")).intdigest()
        if h in self.seen:
            self.dropped_exact += 1
            return False
        self.seen.add(h)
        return True


def stream_source(src: Source, max_rows: int | None, shard_index: int,
                  n_shards: int) -> Iterator[dict]:
    from datasets import load_dataset

    # Plain-text datasets (the Discord dump is one) are opened with the system
    # default encoding, which on Windows is cp1252 and dies on the first emoji.
    # Ask for UTF-8 explicitly, and fall back for builders that reject the kwarg.
    args = [src.repo] + ([src.config] if src.config else [])
    try:
        ds = load_dataset(*args, split=src.split, streaming=True,
                          encoding="utf-8", encoding_errors="replace")
    except (TypeError, ValueError):
        try:
            ds = load_dataset(*args, split=src.split, streaming=True)
        except Exception as exc:
            print(f"    unavailable: {type(exc).__name__}: {str(exc)[:140]}", flush=True)
            return
    except Exception as exc:
        print(f"    unavailable: {type(exc).__name__}: {str(exc)[:140]}", flush=True)
        return

    it = iter(ds)
    i = 0
    while True:
        try:
            row = next(it)
        except StopIteration:
            break
        except Exception as exc:
            # One malformed row must not kill a 12-hour ETL session.
            print(f"    row error, skipping: {type(exc).__name__}: {str(exc)[:90]}",
                  flush=True)
            break
        if max_rows and i >= max_rows:
            break
        if n_shards == 1 or i % n_shards == shard_index:
            yield row          # deterministic fan-out across parallel sessions
        i += 1


def _dig(row: dict, path: str):
    """Follow a dotted path, so 'translation.hi_ng' reaches a nested column."""
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def extract(row: dict, src: Source | None = None) -> str:
    if src and src.text_key:
        v = _dig(row, src.text_key)
        if isinstance(v, str) and v.strip():
            return v

    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # Several sets nest the payload one level down -- findnitai puts everything
    # under `translation`, which is why an earlier run read 189,102 rows and
    # kept none of them.
    for v in row.values():
        if isinstance(v, dict):
            for k in TEXT_KEYS:
                inner = v.get(k)
                if isinstance(inner, str) and inner.strip():
                    return inner
            for inner in v.values():
                if isinstance(inner, str) and len(inner) > 20:
                    return inner

    # Last resort. It must reject identifiers, or a schema whose text column is
    # not in TEXT_KEYS silently yields IDs instead: the YouTube set has its text
    # in `CommentText`, so this fallback returned `CommentID` and put 2,782
    # documents of "UgyRjrEdJIPrf68uND14AaABAg" into the corpus.
    for v in row.values():
        if not isinstance(v, str) or len(v) <= 20:
            continue
        if " " not in v.strip():
            continue            # no spaces -> an identifier, hash or URL
        if _looks_like_id(v):
            continue
        return v
    return ""


_ID_LIKE = re.compile(r"^[A-Za-z0-9_\-]{16,}$")


def _looks_like_id(v: str) -> bool:
    s = v.strip()
    if _ID_LIKE.match(s):
        return True
    # High digit+case-mixing density with no spaces is an identifier.
    alnum = [c for c in s if c.isalnum()]
    if not alnum:
        return False
    mixed = sum(1 for c in alnum if c.isdigit()) / len(alnum)
    return " " not in s and mixed > 0.15


def is_human_annotated(row: dict) -> bool:
    """findnitai flags provenance as source==1 (human) / 0 (synthetic)."""
    for v in row.values():
        if isinstance(v, dict) and "source" in v:
            try:
                return int(v["source"]) == 1
            except (TypeError, ValueError):
                return True
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--lexicon", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--target-tokens", type=int, default=5_000_000)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--val-fraction", type=float, default=0.002)
    ap.add_argument("--docs-per-file", type=int, default=100_000)
    ap.add_argument("--only-pool", default=None, help="genz|internet|general|hinglish")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    lex = Lexicon.load(args.lexicon) if Path(args.lexicon).exists() else None
    if lex is None:
        print(f"WARNING: no lexicon at {args.lexicon}; slang dimension will be 0")
    annotator = RegisterAnnotator(lex)
    deduper = Deduper()

    sources = [s for s in SOURCES if not args.only_pool or s.pool == args.only_pool]
    total_weight = sum(s.weight for s in sources)

    stats: Counter = Counter()
    pool_tokens: Counter = Counter()
    register_hist: Counter = Counter()
    val_hash_seed = 0xB0551

    train_fh = gzip.open(out / "train" / f"part-{args.shard_index:02d}.jsonl.gz",
                         "wt", encoding="utf-8")
    val_fh = gzip.open(out / "val" / f"part-{args.shard_index:02d}.jsonl.gz",
                       "wt", encoding="utf-8")

    try:
        for src in sources:
            budget = int(args.target_tokens * src.weight / total_weight)
            if budget < 1000:
                continue
            print(f"\n=== {src.repo} :: {src.split}  [{src.pool}]  "
                  f"budget {budget:,} tokens ===", flush=True)
            cfg = CleanConfig.strict() if src.pool == "general" else CleanConfig()
            got = 0
            rows = 0
            for row in stream_source(src, None, args.shard_index, args.n_shards):
                rows += 1
                if src.prefer_human and not is_human_annotated(row):
                    stats["drop:synthetic_flagged"] += 1
                    continue
                res = clean_text(extract(row, src), cfg, author=row.get("author"),
                                 pool=src.pool)
                if not res.ok:
                    stats[f"drop:{res.reason.split('(')[0]}"] += 1
                    continue
                if not deduper.accept(res.text, res.n_words):
                    stats["drop:duplicate"] += 1
                    continue

                # Split by document hash BEFORE annotation, so annotation
                # statistics cannot leak across the boundary (SPEC §5.14).
                h = xxhash.xxh64(res.text.encode("utf-8"), seed=val_hash_seed).intdigest()
                is_val = (h % 100000) < int(args.val_fraction * 100000)

                reg = annotator.annotate(res.text)
                register_hist[f"slang_{reg.slang}"] += 1
                record = {
                    "text": res.text,
                    "register": reg.to_dict(),
                    "pool": res.pool,
                    "source": src.repo,
                    "n_words": res.n_words,
                }
                (val_fh if is_val else train_fh).write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                approx_tokens = int(res.n_words * 1.35)   # words -> BPE tokens
                got += approx_tokens
                pool_tokens[res.pool] += approx_tokens
                stats["kept"] += 1

                if got >= budget:
                    break
                if stats["kept"] % 20000 == 0:
                    print(f"    {rows:,} rows -> {stats['kept']:,} kept, "
                          f"{got / 1e6:.2f}M tokens", flush=True)
            print(f"    {rows:,} rows scanned, {got / 1e6:.2f}M tokens kept", flush=True)
    finally:
        train_fh.close()
        val_fh.close()

    print("\n=== summary ===")
    print(f"  kept                {stats['kept']:,} documents")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        if k.startswith("drop:"):
            print(f"  {k:<22} {v:,}")
    print(f"  exact duplicates    {deduper.dropped_exact:,}")
    print(f"  frequency-capped    {deduper.dropped_capped:,}")

    print("\n  tokens by pool (approx):")
    total = sum(pool_tokens.values()) or 1
    for pool, n in pool_tokens.most_common():
        print(f"    {pool:<10} {n / 1e6:>10.2f}M  ({100 * n / total:>5.1f}%)")

    print("\n  slang-register distribution (this is the conditioning signal):")
    for i in range(5):
        n = register_hist[f"slang_{i}"]
        bar = "#" * int(50 * n / max(register_hist.values() or [1]))
        print(f"    slang_{i}  {n:>9,}  {bar}")

    meta = {
        "target_tokens": args.target_tokens,
        "shard_index": args.shard_index,
        "kept_documents": stats["kept"],
        "pool_tokens": dict(pool_tokens),
        "register_hist": dict(register_hist),
        "drops": {k: v for k, v in stats.items() if k.startswith("drop:")},
    }
    (out / f"mine_meta_{args.shard_index:02d}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
