# BUSSBENCH

Held-out benchmark for `bussin-*`. **Never touched by training.**

> **Status: v0.1 PROVISIONAL.** 949 of 4,650 planned items. The three
> lexicon-derived tasks are built; the other twelve need authoring. The
> *current-slang* half of the built tasks is provisional pending corpus
> re-attestation — see "The attestation gate" below.

## Construction rules (SPEC §14.0)

1. Items are sourced from text **postdating** the training corpus, or authored.
2. Every item is checked against the training corpus by **13-gram containment**.
   Any hit is dropped.
3. This directory has no path into any data pipeline.
4. `manifest.json` records a content hash per item; the training dataloader
   asserts none of them appear.
5. A live slice is collected monthly from *after* the model's cutoff, so drift
   stays measurable for the model's whole life.

A benchmark that leaks into training is worse than no benchmark, because it
reports success. `bussin/eval/bussbench.py::assert_not_contaminated` is meant
to run in CI rather than be trusted to discipline.

## Built

| Task | Items | Target | Metric | Source |
|---|---:|---:|---|---|
| `slang_understanding` | 500 | 500 | accuracy | term in real usage → pick definition |
| `outdated_slang` | 249 | 250 | accuracy | classify current / aging / dead, balanced 83/83/83 |
| `slang_generation` | 200 | 200 | lexicon match + human | definition → produce the term |

Contamination: **0 items** overlapped a 20,501-document / 994,602-13-gram index
of the training corpus.

### Item design

Distractors are the whole game for the MCQ tasks. Random wrong definitions let
a model answer from topic alone, so distractors are drawn from terms of
**similar definition length and era**, and any item where the correct answer is
uniquely the longest option is rejected.

`outdated_slang` is balanced across its three classes. Unbalanced, a model that
always answers "dead" would score 80% against a lexicon that is 80% dead.

## The attestation gate

`current` status in the lexicon is currently derived from **age alone** — a term
first attested within four years. That is not the same as "actually used", and
building on it directly produced items about *"celery muncher"*, *"lana coded"*
and `ao` ("Area of Operations. Army-speak.").

Filtering instead on Urban Dictionary score did not fix it: the site's
most-upvoted entries skew 2003-2011, which yielded `sul` ("See you later"),
`bbiaf` ("Be back in a few") and `computer whiz` as supposedly current slang.

So until re-attestation runs, current-slang items come from the hand-curated
`MLBtrio` set only, minus its legacy tail:

> **`MLBtrio/genz-slang-dataset` is not uniformly contemporary.** Of its 1,322
> usable entries, roughly a third are AIM/SMS-era initialisms — `nifoc`,
> `aamof`, `g2g`, `suyf`, `wrud`, `oic`, `l33t`, `aisb`. The other ~460 are
> genuinely current: `glow up`, `rent free`, `hits different`, `periodt`,
> `understood the assignment`, `clapback`, `i oop`, `ok boomer`, `big yikes`.
>
> Term shape does not separate them — `rizz`, `fam`, `stan` and `w` are all
> short and consonant-heavy but entirely contemporary. What does separate them
> is that an initialism's definition is its own expansion, so
> `is_initialism()` checks whether the initial letters of the definition's
> words reproduce the term.

**To lift the gate:** run `pipelines/01_discover_emerging.py` at full ranking
scale on a Kaggle CPU session (`--rank-rows 500000 --recent-rows 0`, zero GPU
quota), which writes real `corpus_rate` values. `build_bussbench` detects this
automatically via `attestation_has_run()` and switches to usage-based evidence.

## Not yet built

Each has a `tasks/<name>.TODO.md` recording its target count and construction
note: `lm`, `conversation`, `contextual_slang`, `style_transfer`, `emoji`,
`comprehension`, `code_switching`, `hinglish`, `english_retention`, `safety`,
`hallucination`, `cringe`.

`style_transfer` carries the sharpest constraint: the Gen-Z side must be
**mined human text**, never generated. An LLM asked to produce slang emits its
own 2-3 year stale register, which is the exact failure the benchmark exists to
detect (SPEC §13.2).

## Content filtering

Urban Dictionary carries slurs, sexual content about real people, and hate
speech dressed as humour. `OFFENSIVE` in `pipelines/05_build_bussbench.py`
drops any entry whose term, definition or example matches. This is a coarse
filter and the items should still be read before publication.

## Rebuilding

```bash
python pipelines/05_build_bussbench.py \
    --lexicon data/lexicon/lexicon.jsonl \
    --corpus  data/corpus/train
```
