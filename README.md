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
| 1 — Data pipeline | **built and run at small scale**; full corpus build pending |
| 2 — Tokenizer | **built, gate passing** |
| 3 — Relay + trainer | **built and verified** (see below); `bussin-125m` run pending real compute |
| 4 — `bussin-400m` | — |
| 5 — `bussin-1b` (conditional on TPU) | — |
| 6 — Instruction tuning | — |
| 7 — BUSSBENCH | metrics built (`bussin/eval/metrics.py`); benchmark items pending |
| 8 — Deployment | — |

### Verified on real Kaggle hardware

A full cycle ran on Kaggle T4 x2 and completed: trained 400 steps, checkpointed
to Hugging Face, released the lease, exited clean.

| Check | Result |
|---|---|
| Hardware detection | `2x Tesla T4 (cuda) \| bf16=False` — exactly as the spec predicted |
| Precision selection | `fp16 (scaler=on)` chosen automatically (Turing has no bf16) |
| Batch-plan invariant | `micro 4 x accum 2 x 2 dev` reproducing the global batch exactly |
| Measured throughput | **160,936 tok/s** (4.87 TFLOP/s effective on the 5M smoke model) |
| Checkpoint to HF | `ckpt-00000400` — model, optimizer, scheduler, scaler, RNG, cursor, manifest |
| Lease from Kaggle | claimed and released against the live Hub |
| Dashboard | published to a static Space, HTTP 200 |

The throughput figure is a **floor, not an estimate for `bussin-400m`**: a 5M
model at sequence length 512 is overhead-bound. SPEC §9.6's 24-36 TFLOP/s for
T4 x2 still needs measuring with the real config.

### Verified locally

| Check | Result |
|---|---|
| Parameter counts, analytic vs built | exact match at all four sizes |
| Untrained cross-entropy | 10.95 ≈ ln(49152) = 10.80 |
| Document mask | exactly block-diagonal causal; no cross-document leakage |
| **Resume determinism** (`scripts/resume_test.py`) | **0.0 relative divergence** — interrupted run reproduces uninterrupted bit-identically |
| **Lease / compare-and-swap** (`scripts/lease_test.py`) | **15/15**, including the two-worker race |
| **Two-session relay** (`scripts/relay_e2e_test.py`) | hand-off works; continuity probe 4.4% drift; session efficiency 0.99 |
| Tokenizer gate | PASS — roundtrip 1.0000, 3.37 bytes/token on Gen-Z |
| Lexicon build | 152,941 UD definitions → 55,499 terms + 1,322 curated |
| Corpus mining | 20,501 documents, register spread across all five slang buckets |

## Automated operation

An orchestrator tick decides whether to start a session somewhere and then
exits. It is stateless: every tick re-reads run state, lease and quota from
Hugging Face, so a missed tick or a crashed runner resolves itself.

```bash
python -m bussin.orchestrator.tick --config configs/400m.yaml --status   # inspect
python -m bussin.orchestrator.tick --config configs/400m.yaml            # dispatch
```

`.github/workflows/orchestrator.yml` runs it every 15 minutes on GitHub Actions
(~960 of the 2,000 free minutes/month; unlimited if the repo is public).
"Stop when quotas are exhausted, resume next week" needs no code — ticks exit
in two seconds until the reset boundary.

Dashboard: a **pre-rendered static page** pushed to a free HF Space on every
tick. HF now returns `402 Payment Required` for Gradio and Docker Spaces on
free CPU; only static Spaces are free. Rendering server-side also keeps the
checkpoint repo private, since the browser never needs a token.

## Running it

```bash
python pipelines/00_build_lexicon.py --out data/lexicon/lexicon.jsonl
python pipelines/01_discover_emerging.py --rank-rows 500000   # needs a big pass
python pipelines/02_mine_corpus.py --target-tokens 7_000_000_000 --shard-index 0 --n-shards 5
python pipelines/03_train_tokenizer.py --vocab-size 49152     # gate must pass first
python pipelines/04_tokenize_shard.py --split train
python -m bussin.relay.bootstrap configs/400m.yaml            # a relay worker
```

Tests: `python scripts/resume_test.py`, `scripts/lease_test.py`, `scripts/relay_e2e_test.py`.

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
