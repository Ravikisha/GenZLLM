"""BUSSBENCH -- the held-out benchmark.

Construction rules that matter more than the items themselves (SPEC §14.0):

1. Items are sourced from text **postdating** the training corpus, or authored.
2. Every item is checked against the training corpus by **13-gram containment**;
   any hit is rewritten or dropped.
3. The benchmark lives in its own directory with no path into any data pipeline.
4. Item hashes are recorded in `manifest.json`, and the training dataloader
   asserts none of them appear.
5. A live slice is collected monthly from *after* the model's cutoff, so drift
   stays measurable for the model's whole life.

A benchmark that leaks into training is worse than no benchmark, because it
reports success. Hence `assert_not_contaminated`, which is meant to be run in
CI rather than trusted to discipline.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

BENCH_ROOT = Path("eval/BUSSBENCH")
NGRAM_N = 13

# The 15 tasks of SPEC §14.1. `auto` marks the ones scorable without a human.
TASKS: dict[str, dict[str, Any]] = {
    "lm":             {"n": 0,    "metric": "perplexity", "auto": True},
    "slang_understanding": {"n": 500, "metric": "accuracy", "auto": True},
    "slang_generation":    {"n": 200, "metric": "lexicon_match", "auto": False},
    "conversation":        {"n": 150, "metric": "human_1_5", "auto": False},
    "contextual_slang":    {"n": 300, "metric": "accuracy", "auto": True},
    "style_transfer":      {"n": 400, "metric": "meaning+register", "auto": False},
    "emoji":               {"n": 250, "metric": "accuracy", "auto": True},
    "comprehension":       {"n": 300, "metric": "f1", "auto": True},
    "code_switching":      {"n": 200, "metric": "accuracy", "auto": True},
    "hinglish":            {"n": 250, "metric": "accuracy", "auto": False},
    "english_retention":   {"n": 1000, "metric": "accuracy", "auto": True},
    "safety":              {"n": 300, "metric": "toxicity_rate", "auto": True},
    "hallucination":       {"n": 200, "metric": "accuracy+abstention", "auto": True},
    "outdated_slang":      {"n": 250, "metric": "accuracy", "auto": True},
    "cringe":              {"n": 350, "metric": "see_metrics_py", "auto": True},
}


@dataclass
class BenchItem:
    id: str
    task: str
    prompt: str
    answer: Any
    options: list[str] | None = None
    context: str | None = None
    note: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        blob = json.dumps(
            {"task": self.task, "prompt": self.prompt, "context": self.context},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def searchable_text(self) -> str:
        """The text that must not appear in training data."""
        return " ".join(p for p in (self.context, self.prompt) if p)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ------------------------------------------------------------------ #
# Contamination
# ------------------------------------------------------------------ #


_WORD = re.compile(r"[a-z0-9']+")


def ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class ContaminationIndex:
    """13-gram index over the training corpus.

    Stored as hashed 64-bit ints rather than tuples: a 30B-token corpus has too
    many distinct 13-grams to hold as Python objects, and a false-positive rate
    of ~0 at this scale is fine because every hit is reviewed by hand anyway.
    """

    def __init__(self, n: int = NGRAM_N) -> None:
        self.n = n
        self.hashes: set[int] = set()
        self.docs = 0

    @staticmethod
    def _h(gram: tuple[str, ...]) -> int:
        return int.from_bytes(
            hashlib.blake2b(" ".join(gram).encode("utf-8"), digest_size=8).digest(),
            "big",
        )

    def add_text(self, text: str) -> None:
        self.docs += 1
        for g in ngrams(text, self.n):
            self.hashes.add(self._h(g))

    def add_corpus(self, paths: Iterable[Path], text_key: str = "text",
                   limit: int | None = None) -> None:
        import gzip

        for p in paths:
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.add_text(json.loads(line).get(text_key, ""))
                    except json.JSONDecodeError:
                        continue
                    if limit and self.docs >= limit:
                        return

    def overlap(self, text: str) -> float:
        grams = ngrams(text, self.n)
        if not grams:
            return 0.0
        hit = sum(1 for g in grams if self._h(g) in self.hashes)
        return hit / len(grams)

    def save(self, path: str | Path) -> None:
        import struct

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            fh.write(struct.pack("<QQ", len(self.hashes), self.docs))
            for h in self.hashes:
                fh.write(struct.pack("<Q", h))

    @classmethod
    def load(cls, path: str | Path) -> "ContaminationIndex":
        import struct

        idx = cls()
        with Path(path).open("rb") as fh:
            count, docs = struct.unpack("<QQ", fh.read(16))
            idx.docs = docs
            for _ in range(count):
                idx.hashes.add(struct.unpack("<Q", fh.read(8))[0])
        return idx


def check_contamination(
    items: Sequence[BenchItem], index: ContaminationIndex, threshold: float = 0.0
) -> list[tuple[BenchItem, float]]:
    """Any item sharing a 13-gram with training data is contaminated."""
    hits = []
    for item in items:
        ov = index.overlap(item.searchable_text())
        if ov > threshold:
            hits.append((item, ov))
    return hits


def assert_not_contaminated(items: Sequence[BenchItem], index: ContaminationIndex) -> None:
    hits = check_contamination(items, index)
    if hits:
        lines = "\n".join(f"  {i.id} ({i.task}) overlap={o:.2%}" for i, o in hits[:20])
        raise AssertionError(
            f"{len(hits)} BUSSBENCH items appear in the training corpus:\n{lines}\n"
            "A contaminated benchmark reports success. Rewrite or drop these."
        )


# ------------------------------------------------------------------ #
# Storage
# ------------------------------------------------------------------ #


def save_task(items: Sequence[BenchItem], task: str, root: Path = BENCH_ROOT) -> Path:
    d = root / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{task}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.to_json() + "\n")
    return path


def load_task(task: str, root: Path = BENCH_ROOT) -> list[BenchItem]:
    path = root / "tasks" / f"{task}.jsonl"
    if not path.exists():
        return []
    return [
        BenchItem(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_all(root: Path = BENCH_ROOT) -> dict[str, list[BenchItem]]:
    return {t: load_task(t, root) for t in TASKS if (root / "tasks" / f"{t}.jsonl").exists()}


def write_manifest(root: Path = BENCH_ROOT) -> Path:
    all_items = load_all(root)
    manifest = {
        "version": 1,
        "n_tasks": len(all_items),
        "n_items": sum(len(v) for v in all_items.values()),
        "tasks": {
            t: {"n": len(items), "hashes": sorted(i.content_hash() for i in items)}
            for t, items in all_items.items()
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def blocked_hashes(root: Path = BENCH_ROOT) -> set[str]:
    """Hashes the training dataloader must refuse to serve."""
    path = root / "manifest.json"
    if not path.exists():
        return set()
    m = json.loads(path.read_text(encoding="utf-8"))
    return {h for t in m.get("tasks", {}).values() for h in t.get("hashes", [])}


# ------------------------------------------------------------------ #
# Scoring
# ------------------------------------------------------------------ #


def score_multiple_choice(
    items: Sequence[BenchItem], predict: Callable[[BenchItem], int]
) -> dict[str, float]:
    """`predict` returns the chosen option index.

    Also reports accuracy split by lexicon status, because a model that scores
    well overall but badly on `current` terms has learned a dictionary rather
    than a language.
    """
    correct = 0
    by_status: Counter = Counter()
    total_status: Counter = Counter()

    for item in items:
        got = predict(item)
        ok = got == item.answer
        correct += ok
        status = item.meta.get("status")
        if status:
            total_status[status] += 1
            by_status[status] += ok

    out = {"accuracy": correct / max(len(items), 1), "n": float(len(items))}
    for status, n in total_status.items():
        out[f"accuracy_{status}"] = by_status[status] / n
    return out


def loglikelihood_choice(
    item: BenchItem, score_fn: Callable[[str, str], float]
) -> int:
    """Pick the option with the highest length-normalised log-likelihood.

    `score_fn(prompt, continuation) -> logprob`. Length normalisation matters:
    without it the model just picks the shortest option.
    """
    if not item.options:
        raise ValueError(f"item {item.id} has no options")
    prompt = f"{item.context}\n{item.prompt}" if item.context else item.prompt
    scores = [
        score_fn(prompt, opt) / max(len(opt.split()), 1) for opt in item.options
    ]
    return max(range(len(scores)), key=lambda i: scores[i])
