"""Build the mechanically-derivable BUSSBENCH tasks from the dated lexicon.

Three of the fifteen tasks fall straight out of the lexicon, which is the payoff
for having built it with dates attached:

  slang_understanding  term shown in real usage -> pick the definition
  outdated_slang       classify a term as current / aging / dead
  slang_generation     definition -> produce the term

The rest need authored items, mined emoji contexts, or a human study, and are
left as stubs with their target counts recorded so the gaps stay visible.

Distractor quality is the whole game for the MCQ tasks. Drawing random wrong
definitions makes items a model can solve by topic alone, so distractors are
drawn from terms of *similar length and era*, and items where the correct
answer is trivially the longest option are rejected.

Run:
  python pipelines/05_build_bussbench.py --lexicon data/lexicon/lexicon.jsonl \
      --corpus data/corpus/train
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.data.lexicon import Lexicon
from bussin.eval.bussbench import (
    BENCH_ROOT,
    BenchItem,
    ContaminationIndex,
    TASKS,
    check_contamination,
    save_task,
    write_manifest,
)

MIN_DEF_WORDS = 3
MAX_DEF_WORDS = 40

# Quality gate for benchmark inclusion.
#
# `status == "current"` is NOT sufficient. Until `refine_status_from_corpus`
# has run against a recent slice, status is derived from age alone, so
# "current" means "recently coined on Urban Dictionary" rather than "actually
# used". Building the benchmark straight off it produced items about
# "celery muncher", "lana coded" and `ao` ("Area of Operations. Army-speak.").
#
# So a term must additionally be either hand-curated or strongly
# community-endorsed, and must not simply be an ordinary English word.
MIN_MEDIAN_SCORE = 150
ENGLISH_EXCLUDE_TOP_N = 30_000

# Urban Dictionary carries a lot of entries that are slurs, sexual content
# about real people, or hate speech dressed as humour. None of it belongs in a
# benchmark that gets published.
OFFENSIVE = re.compile(
    r"(?i)\b(pedophile|paedophile|rape|rapist|molest|nigg|fag|tranny|retard|"
    r"kike|spic|chink|coon|whore|slut|incest|bestiality|clitoris|penis|vagina|"
    r"genitals|masturbat|porn|nazi|hitler|holocaust|kill (yourself|himself))\b"
)


def _english_words(n: int = ENGLISH_EXCLUDE_TOP_N) -> set[str]:
    try:
        from wordfreq import top_n_list

        return set(top_n_list("en", n))
    except ImportError:
        return set()


ENGLISH = _english_words()


def usable(entry) -> bool:
    """Structurally answerable as a multiple-choice item."""
    if not entry.definition or not entry.example:
        return False
    n = len(entry.definition.split())
    if not (MIN_DEF_WORDS <= n <= MAX_DEF_WORDS):
        return False
    # The example has to actually contain the term, or the item is unanswerable.
    return entry.term.split()[0].lower() in entry.example.lower()


def is_initialism(entry) -> bool:
    """True when the definition is just the term's letters spelled out.

    `MLBtrio/genz-slang-dataset` is not the uniformly contemporary set its name
    suggests. Of 1,322 entries roughly a third are AIM/SMS-era initialisms --
    `nifoc` ("Naked in front of computer"), `aamof` ("As a matter of fact"),
    `g2g` ("Got to go"), `suyf`, `wrud`, `oic`, `l33t`, `aisb`. The other ~460
    are genuinely current (`glow up`, `rent free`, `hits different`, `periodt`,
    `understood the assignment`, `clapback`, `i oop`, `ok boomer`).

    Term shape alone does not separate them -- `rizz`, `fam`, `stan` and `w` are
    all short and consonant-heavy but entirely contemporary. What does separate
    them is that an initialism's definition is its own expansion: the initial
    letters of the definition's words reproduce the term.
    """
    term = re.sub(r"[^a-z0-9]", "", entry.term.lower())
    if not term or len(term) > 6 or " " in entry.term.strip():
        return False
    words = re.findall(r"[a-z0-9]+", (entry.definition or "").lower())
    if len(words) < 2:
        return False
    initials = "".join(w[0] for w in words[: len(term) + 1])
    # Allow a leading-digit shorthand like g2g -> "got to go".
    normalised = term.replace("2", "t").replace("4", "f").replace("8", "a")
    return initials.startswith(term[:4]) or initials.startswith(normalised[:4])


def is_safe(entry) -> bool:
    if OFFENSIVE.search(entry.definition or "") or OFFENSIVE.search(entry.term):
        return False
    return not OFFENSIVE.search(entry.example or "")


def benchmark_quality(entry, attested: bool = False) -> bool:
    """Good enough to put in front of a human rater.

    Which evidence counts for `current` depends on whether corpus re-attestation
    has run:

    * **Attested** -- `corpus_rate` is real usage frequency in a recent slice,
      and is the right signal.
    * **Not attested** -- status is age-only, and a high Urban Dictionary score
      selects the site's *most-upvoted* entries, which skew 2003-2011. Filtering
      on it produced "sul" (See you later), "bbiaf" (Be back in a few) and
      "computer whiz" as supposedly current Gen-Z slang. So `current` items are
      restricted to the hand-curated set, which is contemporary by construction.

    `aging` and `dead` are safe from Urban Dictionary either way: the dump's
    dates are authoritative for what *was* used and when.
    """
    if not is_safe(entry):
        return False
    # Plain English words are not slang, however recently someone defined them.
    if entry.term.lower() in ENGLISH and entry.source != "curated":
        return False
    if entry.term.isdigit():
        return False

    if entry.status == "current":
        if attested:
            return entry.corpus_rate > 0
        # Provisional: curated, minus the legacy initialism tail.
        return entry.source == "curated" and not is_initialism(entry)
    return entry.source == "curated" or entry.median_score >= MIN_MEDIAN_SCORE


def attestation_has_run(lex: Lexicon) -> bool:
    """True once `refine_status_from_corpus` has written real usage rates."""
    return any(e.corpus_rate > 0 for e in lex.entries)


def pick_distractors(target, pool: list, rng: random.Random, k: int = 3) -> list:
    """Distractors of similar length and era.

    Random distractors let a model answer from topic alone; length-matched ones
    force it to actually know the term.
    """
    n = len(target.definition.split())
    year = target.first_seen[:4] if target.first_seen else ""

    def closeness(e) -> tuple:
        dn = abs(len(e.definition.split()) - n)
        dy = abs(int(e.first_seen[:4]) - int(year)) if (year and e.first_seen) else 99
        return (dn, dy)

    candidates = [e for e in pool if e.term != target.term and usable(e)]
    rng.shuffle(candidates)
    candidates.sort(key=closeness)
    return candidates[: k * 4][:k] if len(candidates) >= k else []


def build_slang_understanding(lex: Lexicon, n: int, rng: random.Random,
                              attested: bool = False) -> list[BenchItem]:
    pool = [e for e in lex.entries if usable(e) and benchmark_quality(e, attested)]
    rng.shuffle(pool)
    # Weight towards current/aging: a benchmark of dead slang tests history.
    ranked = sorted(pool, key=lambda e: {"current": 0, "aging": 1, "dead": 2}[e.status])
    items: list[BenchItem] = []

    for entry in ranked:
        if len(items) >= n:
            break
        distractors = pick_distractors(entry, pool, rng)
        if len(distractors) < 3:
            continue
        options = [entry.definition] + [d.definition for d in distractors]
        # Reject items solvable by "pick the longest option".
        lengths = [len(o.split()) for o in options]
        if lengths[0] == max(lengths) and lengths.count(max(lengths)) == 1:
            continue
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        items.append(
            BenchItem(
                id=f"su_{len(items):04d}",
                task="slang_understanding",
                context=entry.example,
                prompt=f'What does "{entry.term}" mean here?',
                options=shuffled,
                answer=order.index(0),
                meta={"term": entry.term, "status": entry.status,
                      "first_seen": entry.first_seen, "source": entry.source},
            )
        )
    return items


def build_outdated_slang(lex: Lexicon, n: int, rng: random.Random,
                         attested: bool = False) -> list[BenchItem]:
    """Classify a term's lifecycle stage.

    Balanced across the three classes, otherwise a model that always answers
    'dead' scores 80% on a lexicon that is 80% dead.
    """
    by_status: dict[str, list] = {"current": [], "aging": [], "dead": []}
    for e in lex.entries:
        if e.definition and e.status in by_status and benchmark_quality(e, attested):
            by_status[e.status].append(e)
    for v in by_status.values():
        rng.shuffle(v)

    per_class = n // 3
    options = ["currently in common use", "fading but still recognised",
               "outdated -- people would notice it sounds dated"]
    status_to_idx = {"current": 0, "aging": 1, "dead": 2}

    items: list[BenchItem] = []
    for status, entries in by_status.items():
        for e in entries[:per_class]:
            items.append(
                BenchItem(
                    id=f"os_{len(items):04d}",
                    task="outdated_slang",
                    prompt=f'Is the slang term "{e.term}" current, fading, or outdated?',
                    options=list(options),
                    answer=status_to_idx[status],
                    meta={"term": e.term, "status": status,
                          "first_seen": e.first_seen,
                          "corpus_rate": e.corpus_rate, "source": e.source},
                )
            )
    rng.shuffle(items)
    for i, item in enumerate(items):
        item.id = f"os_{i:04d}"
    return items


def build_slang_generation(lex: Lexicon, n: int, rng: random.Random,
                           attested: bool = False) -> list[BenchItem]:
    pool = [e for e in lex.entries
            if e.status == "current" and usable(e) and benchmark_quality(e, attested)]
    rng.shuffle(pool)
    items = []
    for e in pool[:n]:
        items.append(
            BenchItem(
                id=f"sg_{len(items):04d}",
                task="slang_generation",
                prompt=f"What slang term means: {e.definition}",
                answer=e.term,
                note="scored by lexicon match plus human judgement of plausible alternatives",
                meta={"term": e.term, "status": e.status, "first_seen": e.first_seen},
            )
        )
    return items


STUB_NOTES = {
    "lm": "perplexity over test-general and the time-sliced test-genz split",
    "conversation": "5-turn dialogues, human 1-5 plus LLM judge",
    "contextual_slang": "polysemous terms in 2+ contexts; needs authored pairs",
    "style_transfer": "both directions; Gen-Z side must be mined, never generated",
    "emoji": "mine emoji-in-context from the corpus, then author the options",
    "comprehension": "QA over held-out social text",
    "code_switching": "mixed-register input",
    "hinglish": "comprehension and generation",
    "english_retention": "HellaSwag / ARC-e / PIQA / LAMBADA subsets, unmodified",
    "safety": "RealToxicityPrompts subset plus slur-elicitation probes",
    "hallucination": "factual QA with an explicit 'I don't know' option",
    "cringe": "the register suite; scored by bussin/eval/metrics.py",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/lexicon/lexicon.jsonl")
    ap.add_argument("--corpus", default="data/corpus/train")
    ap.add_argument("--root", default=str(BENCH_ROOT))
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--index-limit", type=int, default=200_000,
                    help="documents to index for the contamination check")
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    lex = Lexicon.load(args.lexicon)
    print(f"lexicon: {lex.stats()}")

    attested = attestation_has_run(lex)
    if attested:
        n_att = sum(1 for e in lex.entries if e.corpus_rate > 0)
        print(f"corpus re-attestation: RUN ({n_att:,} terms carry usage rates)")
    else:
        print(
            "corpus re-attestation: NOT RUN\n"
            "  Term status is age-derived, so 'current' means 'recently coined\n"
            "  on Urban Dictionary', not 'actually used'. Current-slang items are\n"
            "  therefore restricted to the hand-curated set and this build is\n"
            "  PROVISIONAL. Run pipelines/01 at full ranking scale on a Kaggle\n"
            "  CPU session, then rebuild."
        )

    print("\n=== building mechanically-derivable tasks ===")
    built: dict[str, list[BenchItem]] = {
        "slang_understanding": build_slang_understanding(
            lex, TASKS["slang_understanding"]["n"], rng, attested),
        "outdated_slang": build_outdated_slang(
            lex, TASKS["outdated_slang"]["n"], rng, attested),
        "slang_generation": build_slang_generation(
            lex, TASKS["slang_generation"]["n"], rng, attested),
    }
    for task, items in built.items():
        target = TASKS[task]["n"]
        status = Counter(i.meta.get("status") for i in items)
        print(f"  {task:<22} {len(items):>4}/{target:<4} {dict(status)}")

    print("\n=== contamination check (13-gram containment) ===")
    corpus_files = sorted(Path(args.corpus).rglob("*.jsonl.gz")) + \
        sorted(Path(args.corpus).rglob("*.jsonl"))
    if corpus_files:
        index = ContaminationIndex()
        index.add_corpus(corpus_files, limit=args.index_limit)
        print(f"  indexed {index.docs:,} documents, {len(index.hashes):,} distinct 13-grams")
        clean: dict[str, list[BenchItem]] = {}
        for task, items in built.items():
            hits = check_contamination(items, index)
            bad = {i.id for i, _ in hits}
            clean[task] = [i for i in items if i.id not in bad]
            print(f"  {task:<22} {len(hits):>3} contaminated -> dropped "
                  f"({len(clean[task])} remain)")
            for item, ov in hits[:3]:
                print(f"      {item.id} overlap {ov:.1%}: "
                      f"{(item.context or item.prompt)[:70]!r}")
        built = clean
        index.save(root / "contamination_index.bin")
    else:
        print(f"  no corpus at {args.corpus}; SKIPPED -- "
              f"rerun before trusting any result from this benchmark")

    print("\n=== writing ===")
    for task, items in built.items():
        path = save_task(items, task, root)
        print(f"  {path}  ({len(items)} items)")

    for task, note in STUB_NOTES.items():
        p = root / "tasks" / f"{task}.TODO.md"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                f"# {task}\n\nTarget: {TASKS[task]['n']} items, "
                f"metric `{TASKS[task]['metric']}`.\n\n{note}\n\n"
                f"Construction rules: SPEC §14.0. Items must postdate the training\n"
                f"corpus or be authored, and must pass the 13-gram contamination\n"
                f"check in `bussin/eval/bussbench.py` before being committed.\n",
                encoding="utf-8",
            )

    manifest = write_manifest(root)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    print(f"\n  manifest: {m['n_items']:,} items across {m['n_tasks']} tasks -> {manifest}")

    total_target = sum(t["n"] for t in TASKS.values())
    print(f"\n  built {m['n_items']:,} of {total_target:,} planned items "
          f"({100 * m['n_items'] / total_target:.0f}%); the rest need authoring "
          f"(see tasks/*.TODO.md)")

    print("\n  sample items:")
    for task, items in built.items():
        if not items:
            continue
        it = items[0]
        print(f"\n  [{task}] {it.id}")
        if it.context:
            print(f"    context: {it.context[:100]}")
        print(f"    prompt : {it.prompt}")
        if it.options:
            for j, o in enumerate(it.options):
                mark = " <-" if j == it.answer else "  "
                print(f"      {j}. {o[:80]}{mark}")
        else:
            print(f"    answer : {it.answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
