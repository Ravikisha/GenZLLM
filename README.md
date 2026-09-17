# Bussin

**A Gen-Z-native language model trained from scratch on free compute.**

`bussin-125m` · `bussin-400m` · `bussin-1b` — decoder-only transformers trained from random
initialisation on a mined corpus of contemporary internet English, using only free-tier
compute (Kaggle GPU + TPU, Colab, Lightning).

> **Full technical specification: [`docs/SPEC.md`](docs/SPEC.md)** — 17 sections, every claim sourced.

---

## The short version

**Why this isn't "Llama with a slang prompt":**

1. **There is no Gen-Z pretraining corpus.** Every Gen-Z-labelled dataset on Hugging Face,
   combined, is under 2 MB — and most of it is the same three files re-uploaded. So the
   register gets *mined* out of Reddit / Twitch / Discord / YouTube using a date-stamped
   slang lexicon, not downloaded.

2. **Cringe is register mismatch, not slang.** Nobody cringes at a teenager saying
   "that's so cooked" to a friend; they cringe when a bank's Twitter account says it.
   So Bussin models `p(text | register)` — every pretraining document is auto-annotated with a
   measured 5-dimensional register vector, and the model learns the conditional. Slang becomes
   a dial, and "cringe" becomes a number you can optimise
   (**Register Calibration Error**).

3. **The compute actually works out.** Kaggle gives 30 GPU-hours *and* a separate 20 TPU-hours
   per week. `bussin-400m` at 30B tokens (75.7 tokens/parameter — nearly 4x Chinchilla-optimal)
   takes about 6 weeks of that.

## Target model

| | `bussin-400m` |
|---|---|
| Parameters | **396,418,176** |
| Training tokens | **30B** (75.7 tok/param) |
| Architecture | Llama-style: RMSNorm, SwiGLU, RoPE, GQA-6, no biases, tied embeddings |
| Context | 2048 |
| Vocabulary | 49,152 (custom byte-level BPE) |
| Peak memory | ~9.2 GB — fits one 16 GB T4 |
| Time on free compute | **~6.2 weeks** |

`bussin-1b` (1,255,245,824 params, 25B tokens) is a conditional Phase 5 — it requires the
TPU backend to reach Chinchilla-optimal, and has an explicit kill criterion.

## Repo layout

```
docs/SPEC.md          the specification
configs/              one YAML per model size
bussin/model/         architecture
bussin/tokenizer/     BPE training + forced vocabulary
bussin/data/          mining, cleaning, PII, dedup, lexicon, register annotation
bussin/train/         trainer, WSD schedule, precision, FSDP, XLA
bussin/relay/         lease, checkpoint, watchdog, platform detection
bussin/eval/          BUSSBENCH, register metrics, cringe detection
pipelines/            00_build_lexicon .. 07_publish
notebooks/            kaggle_cpu_etl, kaggle_gpu_train, kaggle_tpu_train, colab, lightning
eval/BUSSBENCH/       held-out benchmark, never touched by training
```

## Status

| Phase | State |
|---|---|
| 0 — Research & spec | **done** — `docs/SPEC.md` |
| 1 — Data pipeline | next |
| 2 — Tokenizer | — |
| 3 — `bussin-125m` prototype + relay resume test | — |
| 4 — `bussin-400m` | — |
| 5 — `bussin-1b` (conditional) | — |
| 6 — Instruction tuning | — |
| 7 — BUSSBENCH | — |
| 8 — Deployment | — |

## Notes on data and licensing

- Training data is tiered Green / Amber / Red in [`docs/SPEC.md §1.12`](docs/SPEC.md).
  **Amber-tier source text is never republished** — only filter code and manifest hashes,
  so results reproduce without redistributing scraped user content.
- All authorship columns are dropped at ingest; PII scrubbing runs over message bodies,
  not just schemas, and is gated on a manual 1,000-document audit.
- Much of the lexicon has **AAVE provenance**. Model cards say so, and slang explanations
  include etymology rather than presenting terms as ownerless internet artefacts.
- Models ship under a research licence pending legal review.

## Compute policy

The relay is platform- and credential-agnostic by design. The **recommended configuration is
one account per platform** — Kaggle's Terms of Use (June 22, 2025) prohibit operating more than
one Kaggle account, and the penalty is a ban of all of them, mid-run.
One account per platform supplies ~50-75 compute-hours/week, which is enough for the plan above.
