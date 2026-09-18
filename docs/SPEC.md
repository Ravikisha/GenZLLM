# Bussin — Technical Specification

**A Gen-Z-native language model trained from scratch on free compute.**

| | |
|---|---|
| Project | `bussin` |
| Models | `bussin-125m`, `bussin-400m`, `bussin-1b` |
| Benchmark | `BUSSBENCH` |
| Spec version | 1.0 |
| Date | 2026-09-17 |
| Target compute | Kaggle free tier (GPU + TPU quotas), Colab free, Lightning free |

---

## Legend — how to read claims in this document

Every factual claim carries a marker. Nothing is asserted without one.

| Marker | Meaning |
|---|---|
| **[V]** | **Verified.** Read directly from the primary source during research (HF API, HF `datasets-server` row counts, official Kaggle docs page text, NVIDIA datasheet, paper). Source linked. |
| **[C]** | **Calculated.** Derived arithmetically from [V] inputs. The formula is shown so you can re-check it. |
| **[E]** | **Estimate.** Informed by published practice but not measured on your hardware. Ranges given. Must be re-measured in Phase 3. |
| **[A]** | **Assumption.** A stated premise that could be wrong. Listed explicitly so it can be falsified. |
| **[R]** | **Recommendation.** My judgement call. Reasoning given; you may overrule. |

> **Global caveat [A]:** Kaggle quotas are described by Kaggle itself as "floating" and demand-dependent. Every number in §9 and §10 should be re-confirmed in your own account settings page before you commit to a schedule.

---

## Contents

| § | Section | What it answers |
|---|---|---|
| [0](#0-executive-summary) | Executive summary | The three findings that shape everything |
| [1](#1-datasets) | Datasets | Every dataset, measured row counts, licences, legal tiering |
| [2](#2-existing-gen-z--internet-language-models) | Existing models | Prior art, and how Bussin differs |
| [3](#3-what-a-gen-z-llm-actually-means) | Definition | What "Gen-Z LLM" technically means; the register vector |
| [4](#4-dataset-mixture-design) | Dataset mixture | Mixtures for 100M / 300M / 500M / 1B |
| [5](#5-data-cleaning-pipeline) | Cleaning pipeline | 20 steps, without destroying the register |
| [6](#6-tokenizer-design) | Tokenizer | Train new byte-level BPE; vocab sizes |
| [7](#7-model-architecture) | Architecture | Llama-style; exact configs; the bf16 problem |
| [8](#8-training-from-scratch--full-procedure) | Training procedure | WSD schedule, hyperparameters, checkpointing |
| [9](#9-kaggle-and-free-compute-strategy) | Free-compute strategy | Verified quotas, data movement, the relay |
| [10](#10-compute-estimates) | Compute estimates | FLOPs, GPU-hours, "possible" vs "properly trained" |
| [11](#11-training-strategy--which-option) | Training strategy | Options A-D compared; recommendation |
| [12](#12-instruction-tuning) | Instruction tuning | 120K SFT + 20K DPO, task by task |
| [13](#13-synthetic-data) | Synthetic data | Why 0% in pretraining |
| [14](#14-evaluation--bussbench) | Evaluation | BUSSBENCH, 15 tasks, human study |
| [15](#15-avoiding-cringe) | Avoiding cringe | Register Calibration Error; the mirror test |
| [16](#16-project-roadmap) | Roadmap | Phases 0-8 with gates and kill criteria |
| [17](#17-final-recommendation) | Final recommendation | Build `bussin-400m` |
| [S](#sources) | Sources | Every primary source |

---

## 0. Executive summary

**The core finding.** I queried the Hugging Face dataset API across 100+ search terms (~1,780 unique datasets), then pulled exact row counts and sample rows for every Gen-Z candidate. Result:

> **The total volume of authentic, Gen-Z-labelled text on Hugging Face is under 2 megabytes**, and most of it is the same few files re-uploaded. [V]

There is no Gen-Z pretraining corpus. It does not exist. Therefore Bussin cannot be built by "finding the Gen-Z dataset" — the register must be **mined** out of large authentic social corpora that do exist (Reddit, Twitch, Discord, YouTube comments), using a **date-stamped slang lexicon** as the filter signal.

**The second finding.** Kaggle grants **30 GPU-hours/week and, on a separate quota, 20 TPU v3-8 hours/week** [V, confirmed on the project account 2026-09-18]. TPU v3-8 is roughly 4x the effective throughput of T4x2 [C/E]. This single fact is what moves a 1B model from "undertrained toy" to "genuinely trained", on one account, legally.

**The third finding.** Neither the Tesla P100 nor the Tesla T4 supports bfloat16 [V] — bf16 tensor cores begin at Ampere. This dictates fp32 master weights with fp16+GradScaler on GPU and bf16 on TPU, and it is the most common cause of silent divergence in from-scratch runs on free hardware.

**The design thesis.** "Gen-Z model" does not mean "model that outputs slang." It means a model of `p(text | register)` — where register is a measured, controllable vector (slang density, emoji rate, formality, elongation). Cringe is not wrong slang; **cringe is slang density mismatched to context.** Bussin conditions on register during pretraining so the model learns the conditional distribution and cannot spray slang into contexts that do not carry it.

**The recommendation (§17 in brief).** Build the ladder `125M -> 400M -> 1B`. The 400M is the best quality-per-free-hour model and should be treated as the flagship deliverable. The 1B is reachable in ~4-5 months of TPU-primary training and is worth doing *after* 400M validates the data and the anti-cringe mechanism — not before.

---

# §1. Datasets

## 1.0 Method

- Queried `https://huggingface.co/api/datasets` across 100+ search terms -> 1,780 unique datasets.
- For every candidate, pulled exact row counts and byte sizes from `https://datasets-server.huggingface.co/size`.
- For key candidates, pulled actual first rows from `https://datasets-server.huggingface.co/first-rows` and the raw `README.md`.
- **Row counts below are measured, not read off the `size_categories` tag**, which is frequently wrong on the Hub.

**License warning that applies to this entire section [A]:** a `license:` field in a dataset card is a *claim by the uploader*, not a legal guarantee. For scraped social media, the uploader almost never held the rights to relicense the underlying user content. Treat every license tag on scraped data as "the uploader's opinion." The genuine legal analysis is in §1.12.

---

## 1.1 Category 1 — "Authentic Gen-Z language"

**Finding: this category is effectively empty.** Every dataset carrying a Gen-Z label is either (a) a slang *dictionary*, (b) LLM-generated, or (c) a re-upload of one of the first two.

| Dataset | Rows **[V]** | Bytes **[V]** | Format | Genuine or synthetic | License | Verdict |
|---|---:|---:|---|---|---|---|
| [`MLBtrio/genz-slang-dataset`](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset) | 1,779 | 235 KB | `Slang, Description, Example, Context` | **Synthetic** (LLM-authored definitions) | none declared | **USE — as lexicon seed only, and see the correction below** |
| [`Smilyai-labs/Sam-genz-omni`](https://huggingface.co/datasets/Smilyai-labs/Sam-genz-omni) | 31,377 | 5.4 MB | `prompt, response` | Synthetic | none declared | Weak. SFT candidate only |
| [`biropost/genz_preference`](https://huggingface.co/datasets/biropost/genz_preference) | 3,911 | 587 KB | DPO `chosen/rejected` + register control tokens | Synthetic | none declared | **Study the design, don't train on it** (see note) |
| [`Programmer-RD-AI/genz-slang-pairs-1k`](https://huggingface.co/datasets/Programmer-RD-AI/genz-slang-pairs-1k) | 1,005 | 109 KB | pairs | Synthetic | none declared | Redundant |
| [`uziidris/genz-slang_training-data`](https://huggingface.co/datasets/uziidris/genz-slang_training-data) | 950 | 84 KB | — | Synthetic | none declared | Redundant |
| [`kidboo9o/GenZ-Slang`](https://huggingface.co/datasets/kidboo9o/GenZ-Slang) | 106 | 16 KB | — | Synthetic | none declared | Too small |
| [`ethannhzhouu/genz`](https://huggingface.co/datasets/ethannhzhouu/genz) | **0** | 0 | — | — | — | **BROKEN — empty** |
| [`universalgamingfen1/genz-slang-dataset-prepared`](https://huggingface.co/datasets/universalgamingfen1/genz-slang-dataset-prepared) | **0** | 0 | — | — | — | **BROKEN — empty** |
| `Tawfia/Gen_Z_Words_Phrases`, `SentientCycow/GenZ-lingo`, `LeFluffyPunk/gen-z-translations` | n/a | n/a | — | — | — | **Unloadable via datasets-server** |

### Verified duplicate clusters [V]

These are not independent datasets. I confirmed by comparing actual rows:

**Cluster A — `MLBtrio` (1,779 rows), re-uploaded 3x:**
- `MLBtrio/genz-slang-dataset` — 1,779 rows, original
- `synk/genz-slang-completions` — 1,779 rows. First rows are *verbatim* the MLBtrio `Example` column reformatted as `prompt`/`completion`.
- `anupamaditya/genz-slang-instruction-dataset` — 1,779 rows, same content as instruction format.

**Cluster B — one 821-row synthetic translation set, re-uploaded 4x:**
- `GCruz19/GenZ_data` (821), `archie-kay/Gen-ZifAI` (821), `alisha-huss/genzifai` (821), `jkb2002/formatted_genz_normal_eng` (820), `dtthanh/gen_z_translation` (821)

**Cluster C — 105-row set, uploaded 2x:**
- `ai-maker-space/gen-z-translation` (105 rows, 7,485 bytes) and `mrCarl0/genZ_data` (105 rows, **7,485 bytes** — byte-identical)

> **Note on `biropost/genz_preference`:** its design is the most interesting thing in this category — it conditions generations on explicit register control tokens `<<SLANG:3>> <<EMOJI:0>> <<HYPE:1>> <<FORMAL:0>>`. Bussin adopts and automates this idea (§3.4, §15). **But do not train on the data**: inspecting rows surfaced a `chosen` completion containing an ethnic-slur-adjacent token. It is unfiltered LLM output. Take the schema, not the rows.

### Correction from implementation (measured 2026-09-18) [V]

`MLBtrio` is the only usable entry in this category, and the spec above treated it as uniformly contemporary curated slang. **It is not.**

Of its 1,322 entries that survive lexicon filtering, roughly **a third are AIM/SMS-era initialisms**, verified by inspection: `nifoc` ("Naked in front of computer"), `aamof` ("As a matter of fact"), `g2g`, `suyf`, `wrud`, `oic`, `l33t`, `aisb`, `ayt`, `ianac`, `bm&y`.

The other **~460 are genuinely current**: `glow up`, `rent free`, `hits different`, `periodt`, `understood the assignment`, `clapback`, `i oop`, `ok boomer`, `big yikes`, `catch these hands`, `take several seats`, `main character`, `it's giving…`, `no cap`, `cheugy`, `e-boy`.

**Term shape does not separate the two classes** — `rizz`, `fam`, `stan` and `w` are all short and consonant-heavy but entirely contemporary. What separates them is that an initialism's definition is its own expansion, so the filter checks whether the initial letters of the definition's words reproduce the term (`is_initialism()` in `pipelines/05_build_bussbench.py`).

**Consequence [R]:** there is currently **no reliable source of contemporary-slang ground truth in this project**. The Urban Dictionary dump stops in 2023 (§1.4), UD score selects 2003-2011 entries, and the curated set is a mix. Corpus re-attestation (§5.9a) is therefore not polish — it is the only mechanism that can establish what is current, and everything downstream that depends on `status == "current"` is provisional until it runs.

### 1.1 Verdict

Combined usable content in this category: **~1,800 unique slang entries and a few thousand synthetic sentence pairs.** At roughly 40 tokens per entry that is **~70K tokens [C]** — 0.0003% of what a 400M model needs.

**Role in training [R]:** seed the **dated slang lexicon** (§5.9) and a small slice of instruction-tuning data. **Zero role in pretraining.** Never let Cluster B into training — its slang is visibly dated ("lit AF", "vibes on point", "YOLO") and would actively teach the cringe we are trying to prevent.

---

## 1.2 Category 2 & 3 — General internet + social-media language (the real corpus)

This is where the actual language lives.

| Dataset | Rows **[V]** | Size **[V]** | Period | Genuine? | License (claimed) | PII | Verdict |
|---|---:|---:|---|---|---|---|---|
| [`HuggingFaceGECLM/REDDIT_comments`](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments) | **592,448,578** | 109.2 GB | 2006 -> **Jan 2023** | **Genuine** | not declared | **YES** — `author` field | **USE — backbone** |
| [`fddemarco/pushshift-reddit-comments`](https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments) | **1,845,964,820** | 291.7 GB | Pushshift era | Genuine | not declared | **YES** — `author` | Use only if more volume needed |
| [`tensorshield/reddit_dataset_157`](https://huggingface.co/datasets/tensorshield/reddit_dataset_157) | **43,107,308** | 24.7 GB | **continuously updated** | Genuine | `mit` (claimed) | likely | **USE — critical for recency** |
| [`lparkourer10/twitch_chat`](https://huggingface.co/datasets/lparkourer10/twitch_chat) | **8,984,657** | 172 MB | recent | Genuine | `cc-by-sa-4.0` (claimed) | usernames stripped | **USE — high value** |
| [`llmtraining-scraper/discord-messages`](https://huggingface.co/datasets/llmtraining-scraper/discord-messages) | **6,217,832** | 487 MB | "2026" | Genuine | `mit` (claimed) | **pre-anonymized** | **USE — high value** |
| [`Daankular/twitch-chat`](https://huggingface.co/datasets/Daankular/twitch-chat) | 1,464,030 | — | recent | Genuine | `other` | — | Use (157 streamer configs) |
| [`AmaanP314/youtube-comment-sentiment`](https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment) | **1,032,225** | 300 MB | — | Genuine | `cc-by-sa-4.0` | video/comment IDs | Use — text column only |
| [`Michielo/twitchchat`](https://huggingface.co/datasets/Michielo/twitchchat) | 1,951 *streams* | 1.2 GB | 2020 | Genuine | `cc-by-4.0` | — | Use — **proper academic provenance** |
| [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | 200,785 | 14 MB | <=2020 | Genuine | `unknown` | — | **Eval only, not training** |
| [`u84u/4chan-pol`](https://huggingface.co/datasets/u84u/4chan-pol) | 265,207,631 | 22.1 GB | 2016-2019 | Genuine | `mit` | — | **REJECT** |
| [`lesserfield/4chan-datasets`](https://huggingface.co/datasets/lesserfield/4chan-datasets) | — | — | — | Genuine | `unlicense` | — | **REJECT** |
| [`v2ray/4chan`](https://huggingface.co/datasets/v2ray/4chan) | 50,835 | 350 MB | — | Genuine | `mit` | — | **REJECT** |

### Key structural facts [V]

**`HuggingFaceGECLM/REDDIT_comments` is split by subreddit — 50 splits.** Measured top splits:

| Subreddit | Comments | Size | Gen-Z relevance **[R]** |
|---|---:|---:|---|
| `gaming` | 85,729,253 | 28.4 GB | **High** |
| `todayilearned` | 60,199,778 | 22.7 GB | Low |
| `relationship_advice` | 38,937,398 | 22.3 GB | **High** (young, emotional, informal) |
| `Showerthoughts` | 34,123,213 | 13.3 GB | **High** (short, punchy, joke register) |
| `mildlyinteresting` | 26,436,769 | 9.1 GB | Medium |
| `IAmA` | 25,778,822 | 9.4 GB | Medium |
| `technology` | 25,404,699 | 10.8 GB | Medium |
| `Games` | 23,373,965 | 10.4 GB | **High** |
| `buildapc` | 21,761,801 | 9.8 GB | Medium |
| `explainlikeimfive` | 16,392,814 | 8.5 GB | Low (but good for explanation register) |
| `Damnthatsinteresting` | 15,643,554 | 6.4 GB | Medium |
| `AskHistorians` | 2,714,353 | 2.2 GB | Low — **useful as the formal-register anchor** |

**The critical limitation [V]:** the card states the data covers *"2006 to Jan 2023"*. **It contains no 2023-2026 slang whatsoever.** "Rizz", "gyatt", "delulu", "mogging", "crash out" post-date this corpus or barely appear in it. GECLM gives you conversational English at scale; it does **not** give you current Gen-Z.

**That gap is why the Bittensor Subnet 13 datasets matter.** `tensorshield/reddit_dataset_157` (43.1M rows, 24.7 GB) is described on its card as *"continuously updated by network miners, providing a real-time"* stream [V]. There is a whole family of these under different miner hotkeys — `coldmind/reddit_dataset_94`, `wenknow/reddit_dataset_232`, `gk4u/reddit_dataset_132`, `StormKing99/reddit_dataset_8191`, `nicchio816/reddit_dataset_111`, `Trimness8/reddit_dataset_145`, `James096/reddit_dataset_146`, and more. Each is an independent miner's scrape; they overlap heavily and **must be cross-deduplicated** (§5.1).

### Why 4chan is rejected [R]

`u84u/4chan-pol` is 265M posts — tempting volume. It is the `/pol/` board, the subject of the paper *"Raiders of the Lost Kek"* (arXiv:2001.07487) precisely because of its hate-speech density. Including it would (a) poison the model with organized bigotry, (b) make every safety eval fail, (c) teach a register that is *not* mainstream Gen-Z but a specific extremist subculture. The volume is not worth it. **Reject all three 4chan datasets.**

---

## 1.3 Category 4 — Chat / conversation data

| Dataset | Rows **[V]** | Size | Genuine? | License | Role |
|---|---:|---:|---|---|---|
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 515,311 | 1.62 GB | Synthetic (LLM) | `mit` | SFT — instruction-following backbone |
| [`allenai/WildChat-1M`](https://huggingface.co/datasets/allenai/WildChat-1M) | 837,989 | 3.36 GB | **Genuine human prompts** | `odc-by` | SFT — real user register |
| [`OpenAssistant/oasst2`](https://huggingface.co/datasets/OpenAssistant/oasst2) | 135,174 | 66.7 MB | Genuine human | `apache-2.0` | SFT — highest quality |
| [`anon8231489123/ShareGPT_Vicuna_unfiltered`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | — | — | ChatGPT outputs | `apache-2.0` (claimed) | **Avoid — see §1.12** |

`WildChat-1M` is the standout: real people typing at a chatbot, which is a genuinely informal register, under a clean `odc-by` license from a reputable lab [V].

---

## 1.4 Category 5 — Slang dictionaries

| Dataset | Rows **[V]** | Size **[V]** | Has dates? | License | Verdict |
|---|---:|---:|---|---|---|
| [`georgiyozhegov/urbandictionary`](https://huggingface.co/datasets/georgiyozhegov/urbandictionary) | **152,941** | 36.2 MB | **YES — `time` + `score`** | `cc-by-sa-4.0` | **USE — the single most valuable small dataset in this project** |
| [`LM-Lexicon/Slang`](https://huggingface.co/datasets/LM-Lexicon/Slang) | **507,636** | 174.3 MB | no | not declared | Use with heavy filtering |
| [`daspartho/urban_dictionary`](https://huggingface.co/datasets/daspartho/urban_dictionary) | 73,405 | 15.7 MB | no | not declared | Redundant vs georgiyozhegov |
| [`nikesh66/Slang-Dataset`](https://huggingface.co/datasets/nikesh66/Slang-Dataset) | 5,000 | 199 KB | no | not declared | Minor |
| [`jesscusi/english-slang`](https://huggingface.co/datasets/jesscusi/english-slang) | 500 | 39 KB | no | not declared | Skip |

### Why `georgiyozhegov/urbandictionary` is the keystone [R]

Verified sample row:

```json
{"id": 14384824, "time": "2019-11-04T01:39:39.139000+00:00", "score": 15,
 "word": "Shlumped",
 "definition": "When someone falls asleep or passes out during a social event...",
 "example": "Have you seen James recently? Ya, he's shlumped on the couch over there."}
```

It carries **`time` and `score`**. That means you can build a **dated lexicon**: term -> first-attested date -> community score. From that you get, for free:

1. **A recency-weighted slang filter** for mining the corpus (§5.9).
2. **Automatic "outdated slang" labels** for evaluation (§14.13) — a term first attested 2013 and not re-attested since is *dead slang*, and you can test whether the model knows that.
3. **The register-density signal** used for control-token annotation (§3.4).

No other dataset in this space carries timestamps. This is the only one that lets you distinguish *current* from *dead* slang **mechanically instead of by hand.**

### Correction from implementation (measured 2026-09-17) [V]

Building the lexicon revealed a limit that the dataset card does not state:

> **The dump's most recent definition is dated 2023-11-09.** Its year histogram runs 1999-2023 and stops.

Probing 19 contemporary terms against the built lexicon, **13 were absent**: `gyatt`, `delulu`, `mogging`, `looksmaxxing`, `skibidi`, `sigma`, `aura`, `cooked`, `bussin`, `yap`, `glazing`, `pookie`, `crash out`. Only `rizz`, `mid`, `ick`, `npc`, `no cap` were present.

**Consequence [R]:** a dated dictionary is authoritative for deciding what is *dead* and structurally useless for deciding what is *current*. Lexicographers lag usage by years, and the dumps lag the lexicographers. The lexicon therefore has two halves:

| Half | Source | Answers |
|---|---|---|
| Historical (2003-2023) | Urban Dictionary dump, 55,499 terms after filtering | "is this term dead?" |
| Contemporary (2024-2026) | **Discovered from the corpus** (§5.9a) | "is this term current?" |

This is a better design than the original plan, not a workaround: corpus discovery measures actual usage rather than what someone bothered to write a definition for.

### 1.4a Measured lexicon build [V]

Running `pipelines/00_build_lexicon.py`:

```
152,941 definitions -> 55,499 unique terms  (63.7% removed as junk/low-score/merged)
+ 1,322 new terms from MLBtrio curated seeds (457 already present)
= 56,821 terms
```

The 63.7% removal rate is the `score > 0` filter plus the proper-noun and name-definition filters doing exactly what §1.4 predicted (~40% noise, plus per-term aggregation of multiple definitions).

**Quality caveat [V]:** Urban Dictionary is heavily polluted. Sample rows include `"Wet pickle": "It's you lily. You're a wet pickle."` (score -1) and many "[Name] is the sweetest girl you'll ever meet" entries. **Filter on `score > 0` and drop single-proper-noun entries**, or ~40% of the lexicon is noise [E].

---

## 1.5 Category 6 — Gen-Z translation / style-transfer datasets

Covered in §1.1. **Every single one is LLM-generated, and they collapse into 3 unique sources totalling ~1,900 rows.** [V]

**Verdict [R]: do not train on any of them.** Verified sample from `thesherrycode/gen-z-slangs-translation`:

```
"That's amazing!"          ->  "That's lit!"
"You only live once."      ->  "YOLO."
"I'm not sure about that."  ->  "I'm lowkey unsure."
```

"YOLO" peaked in 2012. "Lit" peaked around 2017. Training on this teaches a model to sound like a 2016 brand account. **This is the exact failure mode described in §15.** Bussin generates its own style-transfer pairs from *mined* contemporary data instead (§12, §13).

---

## 1.6 Category 7 — Gaming / Twitch / chat

| Dataset | Rows **[V]** | Notes |
|---|---:|---|
| `lparkourer10/twitch_chat` | **8,984,657** (8,923,544 train / 61,113 val) | `cc-by-sa-4.0`. Card is self-tagged `genz`. **Best-value internet-register dataset available.** |
| `S1lver404/twitch_chat` | 8,984,657 | **Byte-identical clone** — same 171,194,856 train bytes. Deduplicate. |
| `Daankular/twitch-chat` | 1,464,030 | 157 per-streamer configs -> lets you sample across communities rather than one chat's in-jokes |
| `Michielo/twitchchat` | 1,951 streams / 1.2 GB | CC-BY-4.0, ports Ringer et al., AIIDE 2020 (doi:10.1609/aiide.v16i1.7439). Includes viewer counts + timestamps -> **hype-moment register signal** |

**Role [R]:** Twitch chat is the purest available sample of internet-native written language — extreme abbreviation, emote-as-punctuation, rapid-fire register. It is also the most *unlike* prose, so it must be capped (see §4) or it will wreck the model's ability to write a paragraph.

---

## 1.7 Category 8 — Meme / internet-culture data

**Verdict [R]: skip entirely for v1.**

Everything found is **image-based** classification data (`neuralcatcher/hateful_memes` 5,813 downloads, `MMSoc_Memotion`, etc.). Bussin is text-only. Meme *captions* without images are near-useless — the humour is in the image-text relation. Meme literacy will instead be acquired implicitly from Reddit/Twitch text, and tested explicitly in BUSSBENCH (§14.8).

---

## 1.8 Category 9 — Hinglish / Indian Gen-Z

| Dataset | Rows **[V]** | Size | Genuine? | License | Verdict |
|---|---:|---:|---|---|---|
| [`findnitai/english-to-hinglish`](https://huggingface.co/datasets/findnitai/english-to-hinglish) | **189,102** | 30 MB | **Mixed — carries a `source` flag: 1=human-annotated, 0=synthetic** | `apache-2.0` | **USE — best licensed Hinglish text** |
| [`Abhishekcr448/Hinglish-Everyday-Conversations-1M`](https://huggingface.co/datasets/Abhishekcr448/Hinglish-Everyday-Conversations-1M) | **1,001,323** | 180 MB | **Synthetic** — card states GPT-4o-mini generated | `mit` | Use capped at <=20% of Hinglish budget |
| [`festvox/cmu_hinglish_dog`](https://huggingface.co/datasets/festvox/cmu_hinglish_dog) | 9,962 | 3.9 MB | **Genuine crowdsourced** | `cc-by-sa-3.0` + `gfdl` | **USE — gold quality** |
| [`diwank/hinglish-dump`](https://huggingface.co/datasets/diwank/hinglish-dump) | n/a (server 501) | — | Aggregate of 6 corpora | `mit` | Use — needs manual download |
| [`rvv-karma/English-Hinglish-TOP`](https://huggingface.co/datasets/rvv-karma/English-Hinglish-TOP) | — | — | Genuine | `apache-2.0` | Use |
| [`agarwalayushi/hinglish`](https://huggingface.co/datasets/agarwalayushi/hinglish) | 815,105 | **242.9 GB** | Genuine | `cc-by-4.0` | **REJECT — it is audio (ASR/TTS), not text.** Classic name-trap. |

> `agarwalayushi/hinglish` is exactly the trap you warned about: name and tags say "hinglish / code-switching", size looks enormous, and it is **243 GB of speech clips**. Verified from its card: *"Total clips 815,171 / Total Estimated Hours 2,264+"*. [V]

**Honest assessment [R]:** total genuine Hinglish *text* available is roughly **200K sentence pairs, about 10M tokens [C]**. That is enough for a credible code-switching *capability*, not for Hinglish fluency. Set expectations accordingly: Bussin will handle Hinglish input and produce plausible Hinglish output, but it will not be an Indian-English-native model.

---

## 1.9 Category 10 — Synthetic Gen-Z data

Every dataset in §1.1 and §1.5 is synthetic. Assessment in §13. Short version: **synthetic data is necessary for instruction tuning and forbidden in pretraining**, because synthetic Gen-Z is generated by models whose own Gen-Z is 2-3 years stale — training on it bakes the staleness in and compounds it.

---

## 1.10 Category 11 — General English (base language capability)

| Dataset | Rows **[V]** | Size **[V]** | License **[V]** | Role |
|---|---:|---:|---|---|
| [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 3,496,736,741 | 10,358 GB | `odc-by` | **Primary Stage-1 corpus** |
| [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 236,980,453 | 673 GB | `odc-by` | **Primary — pre-mixed, proven at 135M-1.7B** |
| [`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 52,453,695,892 | 108,231 GB | `odc-by` | Overkill; use if edu-filter is too narrow |
| [`HuggingFaceTB/cosmopedia`](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | 31,064,744 | 92.2 GB | `apache-2.0` | Synthetic textbooks — small slice for reasoning |
| [`allenai/c4`](https://huggingface.co/datasets/allenai/c4) | 114,005,516 | 235.7 GB | `odc-by` | Fallback |
| [`Skylion007/openwebtext`](https://huggingface.co/datasets/Skylion007/openwebtext) | 8,013,769 | 24.2 GB | `cc0-1.0` | Small, clean, **CC0** |
| [`common-pile/comma_v0.1_training_dataset`](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset) | 2,547,527 | 2.7 GB | openly licensed | **Use if you want a fully-clean-license story** |
| [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) | 2,141,709 | 1.0 GB | `cdla-sharing-1.0` | **Smoke-test corpus only** |
| [`monology/pile-uncopyrighted`](https://huggingface.co/datasets/monology/pile-uncopyrighted) | 891,748 | 2.6 GB | `other` | Optional |

**Recommendation [R]:** `smollm-corpus` as the Stage-1 base. It is `odc-by`, it is the exact mixture that produced SmolLM2-135M/360M/1.7B, and it is already deduplicated and quality-filtered — which saves you the single most expensive preprocessing step.

---

## 1.11 Datasets explicitly rejected, with reasons

| Dataset | Reason |
|---|---|
| `agarwalayushi/hinglish` | **Audio, not text.** 243 GB of ASR/TTS clips. |
| `u84u/4chan-pol`, `lesserfield/4chan-datasets`, `v2ray/4chan` | `/pol/` hate-speech corpus. Poisons safety, teaches a fringe register. |
| `ethannhzhouu/genz`, `universalgamingfen1/genz-slang-dataset-prepared` | **0 rows.** Empty repos. |
| Cluster B (`GCruz19`, `archie-kay`, `alisha-huss`, `jkb2002`, `dtthanh`) | Synthetic, dated slang, 4x duplicated. Teaches cringe. |
| `S1lver404/twitch_chat` | Byte-identical duplicate of `lparkourer10/twitch_chat`. |
| `synk/genz-slang-completions`, `anupamaditya/genz-slang-instruction-dataset` | Reformats of `MLBtrio`. |
| `mrCarl0/genZ_data` | Byte-identical duplicate of `ai-maker-space/gen-z-translation`. |
| `budecosystem/genz-13b-v2`, `TheBloke/Genz-70b-GPTQ` | **Name trap.** "GenZ" here is BudEcosystem's general instruct model. Nothing to do with Gen-Z language. |
| `anon8231489123/ShareGPT_Vicuna_unfiltered` | ChatGPT outputs; provenance and ToS problems (§1.12). |
| `dagim/urban-dictionary-embeddings`, `Echo9Zulu/UrbanDictionaryEmbeddings_2019` | Embeddings, not text. |
| All `hateful_memes` / `Memotion` variants | Image datasets. |
| `cardiffnlp/tweet_eval` | **Not rejected — quarantined to eval.** Using it in training would contaminate a standard benchmark. |

---

## 1.12 Legal analysis — what you can actually train on

**[A] I am not a lawyer; this is engineering risk assessment, not legal advice. For a commercial launch, get counsel.**

### The four-layer problem

Rights in scraped social data stack, and a dataset card only ever speaks to layer 3:

1. **The author's copyright** in their comment. Held by the Reddit/Twitch/Discord user. Never licensed to the scraper.
2. **The platform's ToS.** Reddit, Twitter/X, and Discord all restrict bulk redistribution. Pushshift was cut off by Reddit in 2023 for precisely this.
3. **The uploader's declared license.** A miner tagging their scrape `mit` is asserting a right they do not have.
4. **Data-protection law** (GDPR / India's DPDP Act). Usernames and post bodies are personal data. Applies to you regardless of the license tag.

### Practical tiering [R]

| Tier | Datasets | Use for |
|---|---|---|
| **Green** — clean license, use freely | `smollm-corpus`, `fineweb-edu`, `fineweb`, `c4` (odc-by); `openwebtext` (cc0); `common-pile/comma`; `oasst2`, `findnitai/english-to-hinglish`, `rvv-karma/*` (apache-2.0); `WildChat-1M` (odc-by); `cmu_hinglish_dog` (cc-by-sa-3.0); `Michielo/twitchchat` (cc-by-4.0, academic) | Anything, including release |
| **Amber** — research use, do not redistribute the data | `HuggingFaceGECLM/REDDIT_comments`, Bittensor SN13 reddit sets, `lparkourer10/twitch_chat`, `llmtraining-scraper/discord-messages`, `AmaanP314/youtube-comment-sentiment`, `georgiyozhegov/urbandictionary` (CC-BY-SA -> **share-alike is viral on derived text**) | Train on it; **never re-upload the text**; publish weights + code, not the corpus |
| **Red** — do not use | all 4chan sets; `ShareGPT_Vicuna_unfiltered` | — |

### Concrete rules Bussin follows

1. **Never republish Amber source text.** Publish the *filter code* and *manifest hashes* so results are reproducible without redistributing the corpus. The tokenized `.bin` shards stay in a **private** Kaggle Dataset / HF repo.
2. **Strip all authorship at ingest.** Drop `author`, `author_fullname`, `id`, `permalink`, `subreddit_id` before anything is written to disk. Keep `body`, `score`, `created_utc`, `subreddit` only. This is the §5.11 PII pass and it is mandatory, not optional.
3. **CC-BY-SA is contagious.** `georgiyozhegov/urbandictionary` is CC-BY-SA-4.0. Use it as a *lexicon for filtering* (facts about which words exist — not copyrightable) rather than as training text, which keeps the share-alike obligation off your weights. If you do train on the definitions, attribute and consider the ShareAlike implications.
4. **Assume any model released is non-commercial** until a lawyer says otherwise. `bussin-*` ships under a research license.
5. **PII is not "removed" by a license tag.** It is removed by §5.11 running successfully. Verify with spot checks.

### Personally identifying information — measured [V]

| Dataset | PII present |
|---|---|
| `HuggingFaceGECLM/REDDIT_comments` | **Yes** — `author`, `author_fullname`, `permalink` columns confirmed in schema |
| `fddemarco/pushshift-reddit-comments` | **Yes** — `author` column confirmed |
| `llmtraining-scraper/discord-messages` | **Pre-stripped** — card states user/server/channel IDs and timestamps removed; text only |
| `lparkourer10/twitch_chat` | Single `Message` column only — no usernames |
| `AmaanP314/youtube-comment-sentiment` | `CommentID`, `VideoID`, `VideoTitle` — drop all three, keep comment text |
| `georgiyozhegov/urbandictionary` | **In-body PII** — many entries name real people. Needs the §5.11 name filter, not just column dropping |

> Note the last row: column-dropping is insufficient. Reddit and UD text bodies contain names, handles, and occasionally phone numbers *inside the text*. §5.11 must run over `body`, not just over the schema.

---

# §2. Existing Gen-Z / internet-language models

## 2.0 Headline finding

**No serious Gen-Z language model exists.** [V]

I searched the HF model API across 14 query terms (340 unique models). What came back splits into three groups: a handful of tiny hobby models, a large body of *encoder* models for social media (the real prior art), and a pile of false positives.

---

## 2.1 The only direct prior art

### `Sankar-2910/genz-translator` [V]

| Property | Value |
|---|---|
| Parameters | **26.07M** |
| Architecture | Llama-style decoder-only |
| Layers | 7 |
| Hidden size | 448 |
| Attention heads | 7 |
| Context length | 384 |
| Vocabulary | 8,000 |
| Tokenizer | Custom byte-level BPE |
| Trained from scratch | **Yes** |
| Training data | "a curated Gen Z -> English translation dataset" (unspecified size) |
| Hardware | not stated |
| Reported results | none — no benchmark numbers, no perplexity |
| License | `apache-2.0` |
| Downloads | 3,532 |

Its own card states the scope honestly: *"Not designed for: general chatting, coding, mathematics, knowledge retrieval, long conversations."* It is a **one-way slang->English translator**, not a language model in the useful sense.

**Limitations [R]:** 26M parameters trained on a dataset of (almost certainly) a few thousand synthetic pairs. 8K vocabulary at 384 context. It cannot hold a conversation, has no world knowledge, and — given §1.5 — its training pairs are near-certainly the dated synthetic clusters. It is a demo, and its author presents it as one.

### `Abhishekcr448/Tiny-Hinglish-Chat-21M` [V]

| Property | Value |
|---|---|
| Parameters | 21M |
| Architecture | GPT-2 |
| From scratch | Yes |
| Training data | `Abhishekcr448/Hinglish-Everyday-Conversations-1M` — 1,001,323 rows, **synthetic, GPT-4o-mini generated** |
| License | `mit` |
| Reported results | none |

Honest small project. Demonstrates that a 21M model trained purely on synthetic conversation produces *fluent-sounding but hollow* output — which is precisely the failure mode §13 warns about.

### Other hobby entries [V]

`Smilyai-labs/Sam-genz-omni` (dataset only), `mradermacher/RAG-RP-Journal-GenZ-Grokked-4B-GGUF`, `mradermacher/Hamanasu-4B-Chat-Brainrot-i1-GGUF` — LoRA/merge quantizations of existing 4B bases, no papers, no evals, no documented data.

---

## 2.2 The real prior art: internet-language *encoders*

This is where the actual science is, and where Bussin should borrow methodology.

### BERTweet — `vinai/bertweet-base` [V]

| Property | Value |
|---|---|
| Corpus | **850M English tweets, 16B word tokens, ~80 GB** |
| Composition | 845M tweets streamed 01/2012 -> 08/2019, plus 5M COVID-19 tweets |
| Architecture | RoBERTa-base (encoder, 12 layers) |
| Trained from scratch | **Yes** |
| Method | RoBERTa pretraining procedure |
| License | `mit` |
| Paper | Nguyen, Vu & Nguyen, EMNLP 2020 System Demos, pp. 9-14 |
| Downloads | 326,840 |

**Why it matters [R]:** BERTweet is the existence proof that *pretraining from scratch on internet register beats adapting a formal-text model*. It outperformed RoBERTa-base and XLM-R-base on tweet POS tagging, NER, sentiment, and irony. That is the central claim Bussin inherits — and the reason "just fine-tune Llama on slang" is the wrong design.

**Its limitation, and Bussin's opening:** BERTweet is an **encoder**. It understands internet language; it cannot generate it. And its data stops in 2019.

### TimeLMs — `cardiffnlp/twitter-roberta-base-2022-154m` [V]

| Property | Value |
|---|---|
| Corpus | **154M tweets**, filtered from 220M |
| Source | Twitter Academic API, every month **2018-01 -> 2022-12** |
| Architecture | RoBERTa-base |
| License | `mit` |
| Paper | TimeLMs, arXiv:2202.03829 |
| Preprocessing | usernames -> `@user`, links -> `http` |

**Why it matters [R]:** TimeLMs is the only published work that treats **language drift as a first-class problem**, releasing quarterly checkpoints and measuring perplexity degradation on future tweets. Their finding — that models degrade measurably on language from after their cutoff — is the direct empirical justification for Bussin's recency-weighted curriculum (§11.4) and the "outdated slang" eval (§14.13).

Their `@user` / `http` placeholder scheme is adopted directly in §5.5.

---

## 2.3 Reference points for small from-scratch models

| Model | Params | Tokens | Notes **[V]** |
|---|---:|---:|---|
| `HuggingFaceTB/SmolLM2-135M` | 135M | **2T** (~14,800 tokens/param) | FineWeb-Edu + DCLM + Stack. `apache-2.0` |
| `HuggingFaceTB/SmolLM3-3B` | 3B | **11.2T** | GQA + NoPE at 3:1 ratio, staged curriculum, 64k ctx. Fully open configs |

These matter as **calibration**: SmolLM2-135M used 2 *trillion* tokens on a 135M model. Bussin's 125M will see ~10B. So `bussin-125m` will be substantially weaker than SmolLM2-135M on general knowledge — and that is fine and expected, because Bussin is not competing on general knowledge. State this publicly rather than letting people discover it.

---

## 2.4 False positives — the "GenZ" name trap

| Model | What it actually is |
|---|---|
| `budecosystem/genz-13b-v2`, `budecosystem/genz-70b`, `TheBloke/Genz-70b-GGUF`, `mradermacher/genz-70b-GGUF` | **BudEcosystem's "GenZ" general-purpose instruct model family.** Zero relation to Gen-Z language. The most common false positive in this space. |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-*` | Matched on "gen-z" inside "**Gen**ome **Z**eroShot". Biomedical NER. |
| `allenai/Olmo-3-7B-RL-Zero-General` | Matched "gen" + "zero". |
| `zai-org/cogvlm-grounding-generalist` | Matched substring. |

> Roughly **60% of HF search hits for "gen-z" are substring accidents.** Any future dataset/model sweep must filter on card content, not the name.

---

## 2.5 How Bussin differs from all existing work

| Axis | Existing work | Bussin |
|---|---|---|
| Model type | Encoders (BERTweet, TimeLMs) or 21-26M toy decoders | **125M-1B decoder trained from scratch** |
| Data | Tweets to 2019/2022, or a few thousand synthetic pairs | **Mined multi-platform corpus with a 2024-2026 recency tail** |
| Gen-Z as | A vocabulary to substitute | **A conditional distribution `p(text \| register)`** |
| Slang freshness | Frozen at pretraining cutoff | **Dated lexicon + recency-weighted curriculum + measurable staleness** |
| Cringe | Not addressed anywhere in the literature | **First-class objective with its own metric (§15)** |
| Evaluation | General NLU benchmarks | **BUSSBENCH — purpose-built, held out, 14 axes** |
| Compute | Institutional GPUs | **Free tier, with a portable multi-platform relay** |

**The genuinely novel contributions [R]:**

1. **Register-conditioned pretraining at scale.** Auto-annotating every pretraining document with a measured register vector. Nobody has published this for slang/informality specifically.
2. **A date-stamped slang lexicon as a training signal**, not just an eval artifact. Enables "this model knows 'YOLO' is dead" as a *trainable* property.
3. **Cringe as a measurable quantity** — register calibration error — rather than a vibe.
4. **A free-compute relay** that keeps one logical training run consistent across heterogeneous accelerators (fp16 GPU / bf16 TPU) with identical global batch size.

Items 1-3 are publishable. Item 4 is useful engineering.

---

# §3. What a "Gen-Z LLM" actually means

## 3.1 The wrong definition (and why nearly everyone ships it)

> **Wrong:** a model that replaces standard words with slang.

This produces a find-and-replace machine. Evidence it is the industry default: every translation dataset in §1.5 is literally a word-substitution table, and `Sankar-2910/genz-translator` is trained on exactly that.

Word substitution fails because Gen-Z language differs from standard English at **six** levels, and lexicon is only one of them.

## 3.2 The six levels

### L1 — Lexical
Slang terms, but with **three properties that a dictionary loses**: (a) recency — terms have birth and death dates; (b) semantic drift — "mid" moved from neutral-position to pejorative; (c) polysemy under register — "bussin" means excellent for food, and is ironic/mocking when applied to non-food.

### L2 — Orthographic
Intentional respelling as social signal, not error. `tysm`, `ngl`, `istg`, `fr fr`. Elongation as prosody: `noooo` vs `nooooooooo` are different intensities. Case as tone: `ok` / `Ok` / `OK` / `ok.` are four distinct speech acts. Lowercase-everything as a deliberate register.

### L3 — Punctuation and typography
Absence of terminal punctuation is the default; a period is **marked** and reads as cold. `...` signals discomfort. Keysmash (`asdkjfhasdf`) is a grammaticalised expression of overwhelm. Emoji as syntax (skull = laughter, not death), not decoration.

### L4 — Pragmatic
Irony as default mode. Hyperbole as baseline register ("I'm literally dying"). Understatement as flex. Self-deprecation as bonding. **Sincerity is marked and requires signalling** — this is the single hardest thing for a model to learn and the reason adult-written Gen-Z reads wrong.

### L5 — Discourse and cultural
Meme-format references as compressed argument. Fandom/gaming/stan register switching. In-group markers. Platform-specific dialects: Twitch chat differs from Reddit which differs from TikTok comments.

### L6 — Sociolinguistic
Most "Gen-Z slang" is **AAVE** that moved through Black Twitter into general youth usage (`periodt`, `finna`, `no cap`, `bussin` itself). A model that treats these as generic internet slang misrepresents their origin. Also: code-switching (Hinglish, Spanglish), regional variation (UK roadman vs US), and **register-appropriateness** — the same person does not talk to their group chat and their professor identically.

> **[R] An ethical note that belongs in the spec, not a footnote.** Because L6 is largely AAVE, a model that generates this register on command is, in a real sense, performing a dialect that is not the user's. Bussin should (a) document the AAVE provenance in the model card, (b) not be marketed as "talk like a Black person", and (c) include etymology in slang explanations (§12.4) rather than presenting terms as ownerless internet artefacts.

## 3.3 The working technical definition

> **A Gen-Z LLM is a language model whose learned distribution `p(text | register, context)` is well-calibrated over contemporary informal internet English — able to (a) comprehend input at any point on the register scale, (b) generate at a *specified* point on that scale, and (c) infer the appropriate point from context when unspecified.**

Three testable consequences:

1. **Comprehension is register-independent.** It must parse `"lowkey that fit is giving"` and `"I mildly dislike that outfit"` equally well and map both to the same meaning. -> §14.2
2. **Generation is register-conditioned.** Asked for formal output it must produce zero slang. Asked for maximum it must produce natural maximum, not a word salad. -> §14.5, §15
3. **Register is inferred when unstated.** Given a formal prompt it defaults formal; given a casual prompt it defaults casual. **This is the anti-cringe property.** -> §15.7

## 3.4 The register vector — the core mechanism

Every pretraining document is annotated with a **measured** 5-dimensional register vector, computed mechanically. No human labelling.

| Dim | Name | Measurement | Buckets |
|---|---|---|---|
| `slang` | slang density | lexicon hits per 100 tokens, weighted by term recency | 0-4 |
| `emoji` | emoji + emote rate | emoji chars per 100 tokens | 0-3 |
| `abbrev` | abbreviation rate | known-abbrev hits + non-dictionary short tokens | 0-3 |
| `elong` | elongation / keysmash | repeated-char runs >= 3, keysmash detector | 0-2 |
| `formal` | formality | punctuation completeness, sentence length, capitalisation, function-word ratio | 0-3 |

Serialised as a prefix token sequence on each document:

```
<|reg|> <|slang_2|> <|emoji_1|> <|abbrev_2|> <|elong_0|> <|formal_1|> <|/reg|> the actual document text...
```

These are **real vocabulary tokens** (18 added specials, §6.6), so the model attends to them like any other context.

**Why this works [R]:**

- The model learns `p(text | register)` rather than a blurred marginal `p(text)` averaged over wildly different registers. Averaging is what makes a model emit "Dear sir, no cap."
- At inference you *set* the register, so slang density is a dial, not an emergent accident.
- **The gradient teaches restraint.** When `<|slang_0|>` is set, every slang token is penalised. A model trained only on slang-heavy text never learns *not* to use slang; this one learns it explicitly, on 70% of the corpus.
- Register becomes measurable, so "cringe" becomes a number: `|requested - produced|` (§15.3).

**Prior art [V]:** `biropost/genz_preference` uses hand-written `<<SLANG:3>> <<EMOJI:0>>` tags on 3,911 DPO rows. Bussin's contribution is computing them automatically over the full pretraining corpus — 4 orders of magnitude more data, zero labelling cost.

**Risk [A]:** if the annotator is biased (e.g. scores all Twitch chat `slang=4`), the model learns the annotator's bias, not language. Mitigation: hold out 2,000 documents, have a human rate register 0-4, and require Spearman rho >= 0.7 against the annotator before committing the corpus. This gate is in Phase 1.

## 3.5 Pretraining vs instruction tuning — which capability goes where

| Capability | Stage | Why |
|---|---|---|
| Slang comprehension | **Pretrain** | Needs millions of in-context occurrences. Cannot be taught from a dictionary. |
| Orthographic variation (L2) | **Pretrain** | Tokenizer + distributional. Must be in the base. |
| Punctuation/typography register (L3) | **Pretrain** | Purely distributional. |
| Emoji semantics | **Pretrain** | Learned from co-occurrence. Only a dictionary otherwise. |
| Irony / hyperbole defaults (L4) | **Pretrain** | Deep distributional property. SFT cannot install it. |
| Meme/discourse references (L5) | **Pretrain** | Needs breadth of exposure. |
| Code-switching (Hinglish) | **Pretrain**, reinforced in SFT | Needs base-level bilingual distribution. |
| Register *control* | **Pretrain** (via §3.4 tokens) | This is the key architectural choice: control is baked in, not bolted on. |
| Style transfer both directions | **SFT** | A task format, not a capability. |
| Slang *explanation* | **SFT** | Needs an instruction-shaped output. |
| Contextual disambiguation | **SFT + DPO** | Needs preference signal. |
| Knowing slang is outdated | **SFT** (from dated lexicon) | Requires explicit metadata the corpus lacks. |
| Refusing to force slang | **DPO** | Requires negative examples. Cannot be learned from positive-only data. |
| Adapting to the user's register | **DPO** | Needs pairwise preference over register match. |

**The rule [R]: pretraining teaches the distribution; SFT teaches task formats; DPO teaches restraint.** Anyone who tries to install Gen-Z at the SFT stage gets a model that performs slang instead of speaking it — which is exactly the cringe failure.

---

# §4. Dataset mixture design

## 4.0 The supply constraint that governs everything

Before choosing percentages, measure what exists. Estimated token supply, using measured row counts and mean tokens/row [C, from §1 [V] row counts]:

| Pool | Source | Rows **[V]** | Mean tok/row **[E]** | **Tokens [C]** |
|---|---|---:|---:|---:|
| **Gen-Z core** | `lparkourer10/twitch_chat` | 8,984,657 | 12 | 0.11B |
| | `llmtraining-scraper/discord-messages` | 6,217,832 | 20 | 0.12B |
| | `AmaanP314/youtube-comment-sentiment` | 1,032,225 | 25 | 0.03B |
| | `Daankular/twitch-chat` | 1,464,030 | 12 | 0.02B |
| | Bittensor SN13 reddit (recent, ~6 miners after dedup) | ~120M | 40 | ~3.0B |
| | **Gen-Z core subtotal** | | | **~3.3B** |
| **Internet broad** | `HuggingFaceGECLM/REDDIT_comments` (2006-2023) | 592,448,578 | 35 | **~20.7B** |
| **Slang lexicon** | `georgiyozhegov/urbandictionary` (score>0) | ~90,000 | 60 | 0.005B |
| **Hinglish** | all sources combined | ~1.2M | 25 | **~0.03B** |
| **General English** | `smollm-corpus` / `fineweb-edu` | — | — | **effectively unlimited** |

> **The governing fact [C]: there are only about 3.3 billion tokens of genuinely contemporary Gen-Z text in existence on public sources.** Not 30 billion. Every mixture below is designed around that ceiling.

**Consequence [R]:** Gen-Z core must be **repeated** at larger model scales. This is acceptable — repeated data up to ~4 epochs is close to as good as fresh data (Muennighoff et al., "Scaling Data-Constrained Language Models", NeurIPS 2023) — but it must be *deliberate* and tracked, and it hard-caps how large a usefully-Gen-Z model can get. This is an independent argument for 400M over 1B.

---

## 4.1 Mixture principles

1. **General English dominates.** A model that cannot write a clean sentence cannot write a funny one. Slang is a *modulation* of competent English, not a replacement for it.
2. **Register-annotate everything**, including the general English (it gets `slang=0, formal=3`). The contrast is what teaches control (§3.4).
3. **Curriculum, not blend.** Mixture proportions change over training. Gen-Z share rises as LR decays, so the final gradients — which shape the model most — are the most Gen-Z-heavy.
4. **Recency ordering inside Stage 3.** 2024-2026 data lands last. Slang freshness is a *curriculum position*, not a filter.
5. **Zero synthetic in pretraining.** See §13.
6. **Keep a formal anchor throughout.** `AskHistorians`, `explainlikeimfive`, and FineWeb-Edu are retained in every stage so the model does not forget how to be serious.

---

## 4.2 Configuration A — 100M class (built as **`bussin-125m`**, 125.9M)

*Role: pipeline validation and fast iteration. Not a deliverable model.*

| | |
|---|---|
| Parameters | 125,851,392 (config in §7) |
| **Total training tokens** | **10B** (79.5 tok/param) |
| Validation tokens | 20M (held out, never trained) |
| Epochs over Gen-Z core | 0.3 (subsample only) |
| Context length | 1024 |

| Stage | Tokens | General EN | Internet broad | Gen-Z core | Slang lex | Hinglish |
|---|---:|---:|---:|---:|---:|---:|
| S1 base | 7.0B (70%) | 100% | — | — | — | — |
| S2 ramp | 2.5B (25%) | 45% | 35% | 15% | 2% | 3% |
| S3 anneal | 0.5B (5%) | 20% | 25% | 48% | 3% | 4% |
| **Totals** | **10B** | **8.23B (82%)** | **1.00B (10%)** | **0.62B (6.2%)** | **0.07B** | **0.10B** |

**Why [R]:** at 125M the model's whole budget is spent learning basic English syntax. Give it too much slang too early and it learns neither. 82% general English is deliberately high.

---

## 4.3 Configuration B — 300M class

| | |
|---|---|
| **Total training tokens** | **22B** (73 tok/param) |
| Validation tokens | 30M |
| Epochs over Gen-Z core | 0.8 |
| Context length | 2048 |

| Stage | Tokens | General EN | Internet broad | Gen-Z core | Slang lex | Hinglish |
|---|---:|---:|---:|---:|---:|---:|
| S1 base | 14.3B (65%) | 100% | — | — | — | — |
| S2 ramp | 6.6B (30%) | 42% | 33% | 20% | 2% | 3% |
| S3 anneal | 1.1B (5%) | 18% | 24% | 50% | 3% | 5% |
| **Totals** | **22B** | **17.3B (78.7%)** | **2.44B (11.1%)** | **1.87B (8.5%)** | **0.16B** | **0.25B** |

---

## 4.4 Configuration C — 500M class (**`bussin-400m`** is built at 400M; this row scales to 500M)

| | 400M (built) | 500M |
|---|---|---|
| **Total training tokens** | **30B** (75 tok/param) | **36B** (72 tok/param) |
| Validation tokens | 40M | 40M |
| Epochs over Gen-Z core | 1.0 | 1.2 |
| Context length | 2048 | 2048 |

Mixture for the **30B / 400M** build:

| Stage | Tokens | General EN | Internet broad | Gen-Z core | Slang lex | Hinglish |
|---|---:|---:|---:|---:|---:|---:|
| S1 base | 20.0B (66.7%) | 100% | — | — | — | — |
| S2 ramp | 9.0B (30%) | 40% | 35% | 20% | 2% | 3% |
| S3 anneal | 1.0B (3.3%) | 20% | 25% | 50% | 2% | 3% |
| **Totals** | **30B** | **23.8B (79.3%)** | **3.40B (11.3%)** | **2.30B (7.7%)** | **0.20B** | **0.30B** |

Composition inside each pool:

| Pool | Composition |
|---|---|
| **General EN** | `smollm-corpus` 80%, `cosmopedia` 12%, `openwebtext` 5%, `common-pile/comma` 3% |
| **Internet broad** | GECLM Reddit, sampled by Gen-Z relevance weight: `gaming` 18%, `relationship_advice` 15%, `Showerthoughts` 14%, `Games` 10%, `mildlyinteresting` 8%, `technology` 7%, `buildapc` 6%, `explainlikeimfive` 6%, `AskHistorians` 4% (formal anchor), remainder spread |
| **Gen-Z core** | Bittensor SN13 recent Reddit 70%, Discord 12%, Twitch 12%, YouTube comments 6% |
| **Slang lex** | UD score>0 definitions+examples, MLBtrio entries, rendered as natural sentences not dictionary rows |
| **Hinglish** | `findnitai` (human-flagged first) 55%, `cmu_hinglish_dog` 10%, `diwank/hinglish-dump` 20%, `Abhishekcr448` synthetic 15% (capped) |

**Why this is the flagship [R]:** 2.30B Gen-Z-core tokens is ~0.7 epochs of the entire world supply — enough to be *saturating* in the register without repeating. 30B total at 75 tok/param is comfortably past Chinchilla, into the over-trained regime that makes small models punch up. And it fits free compute (§10).

---

## 4.5 Configuration D — 1B (**`bussin-1b`**)

| | |
|---|---|
| **Total training tokens** | **25B** (21.7 tok/param — Chinchilla-optimal, *not* over-trained) |
| Validation tokens | 50M |
| Epochs over Gen-Z core | 1.0 |
| Context length | 2048 |

| Stage | Tokens | General EN | Internet broad | Gen-Z core | Slang lex | Hinglish |
|---|---:|---:|---:|---:|---:|---:|
| S1 base | 16.3B (65%) | 100% | — | — | — | — |
| S2 ramp | 7.5B (30%) | 38% | 36% | 21% | 2% | 3% |
| S3 anneal | 1.2B (5%) | 18% | 24% | 51% | 3% | 4% |
| **Totals** | **25B** | **19.4B (77.5%)** | **2.99B (12.0%)** | **2.19B (8.8%)** | **0.19B** | **0.27B** |

> **Read this honestly [R]:** the 1B gets **fewer** total tokens than the 400M (25B vs 30B) because compute, not data, is the binding constraint. At 21.7 tokens/param it is *exactly* Chinchilla-optimal and **not over-trained**. It will have more capacity and more world knowledge than the 400M, but it will be less thoroughly *fitted*. Expect it to win on knowledge and lose on fluency-per-parameter. If you only ship one model, ship the 400M.

---

## 4.6 Validation and test splits — leakage prevention

| Split | Size | Construction |
|---|---|---|
| `val` | 20-50M tokens | Random 0.2% of shards, **held out before dedup runs**, so near-duplicates of val cannot survive in train |
| `test-general` | 5M tokens | Separate FineWeb-Edu slice, never seen |
| `test-genz` | 5M tokens | **Time-based split**: Gen-Z core documents from the most recent 30 days only. Tests generalisation to *future* language — the TimeLMs-style evaluation. |
| `BUSSBENCH` | ~3,500 items | Hand/assisted-built, §14. **Never derived from any training source.** |

**Leakage rules [R]:**
1. Split **by document** before any dedup or shuffling. Never split a shard.
2. Run MinHash dedup **across the train/val boundary** and delete from *train*, never from val.
3. `test-genz` is time-sliced, not random — the only honest test of slang generalisation.
4. Every BUSSBENCH item is checked against the train corpus by 13-gram containment; any hit is rewritten.
5. Store val shard SHA-256 in `manifest.json` and assert at training start that no val shard id appears in the train cursor range.

---

# §5. Data cleaning pipeline

## 5.0 The prime directive

> **Standard LLM cleaning pipelines are designed to delete exactly the things that make text Gen-Z.**

C4's filters drop any line without terminal punctuation, any document with fewer than 5 sentences, and anything on a bad-words list. Applied to Twitch chat, they delete ~100% of it. Every filter below is therefore written in two versions: **strict** for the general-English pool, **preserving** for the internet/Gen-Z pools.

| Signal | Standard pipeline | Bussin |
|---|---|---|
| No terminal punctuation | **drop** | **keep** — it is the Gen-Z default (L3) |
| Repeated chars `sooooo` | normalise to `so` | **keep, but cap at 6** — length is meaning |
| All-lowercase | penalise | **keep** — deliberate register |
| Emoji | strip | **keep** — they are syntax |
| Short documents | drop <50 words | **keep >= 3 tokens** for chat pools |
| Non-dictionary words | drop as gibberish | **keep** — that is the slang |
| Profanity | drop | **keep, tag** — see §5.12 |
| Keysmash `asdfkjh` | drop as noise | **keep, tag `elong=2`** — it is a lexical item |

---

## 5.1 Deduplication

**Three passes, in order.**

**Pass 1 — exact, within and across sources.** 64-bit xxhash of normalised text (lowercase, whitespace-collapsed, URLs stripped). Bloom filter in memory, ~1.5 GB for 600M docs.

Critical for the Bittensor SN13 family: independent miners scrape overlapping Reddit windows, so cross-dataset exact dedup is mandatory. **Expect 40-70% removal across 6 miner datasets [E].**

**Pass 2 — near-duplicate, MinHash LSH.** 128 permutations, 5-gram shingles, Jaccard threshold **0.8**.

> **Threshold note [R]:** the usual 0.7 is too aggressive here. Twitch chat is full of legitimately near-identical short messages (`W`, `KEKW`, `lets go`) that are *real linguistic events*, not duplicates. At 0.7 you delete the register. Use **0.8 for chat pools, 0.7 for prose pools**, and exempt documents under 10 tokens from MinHash entirely (handle them with frequency capping instead, §5.3).

**Pass 3 — frequency capping for short chat.** Instead of deduping `KEKW`, cap any exact short message at **N=5,000 occurrences** corpus-wide. This preserves the fact that spam-reactions dominate chat, without letting one token become 3% of the corpus.

Tools: `datasketch` (MinHashLSH), `xxhash`, `pyarrow`. For scale, `text-dedup` (ChenghaoMou) implements MinHash-LSH over HF datasets directly.

## 5.2 Language identification

`fasttext` `lid.176.bin`, but with two Gen-Z-specific corrections [R]:

1. **Threshold at 0.5, not the usual 0.8.** Heavy-slang English scores low confidence because it does not look like the Wikipedia text fastText was trained on. At 0.8 you delete the best data.
2. **Hinglish is romanised Hindi and fastText calls it English, Indonesian, or Somali.** Route separately: a document tagged non-English *and* matching a romanised-Hindi function-word list (`hai`, `nahi`, `kya`, `yaar`, `bhai`, `matlab`, `acha`) goes to the **Hinglish pool**, not the bin.
3. Documents under 15 characters skip LID entirely (unreliable) and inherit their source's language.

## 5.3 Spam and bot removal

| Filter | Rule |
|---|---|
| Reddit bots | Drop `author` matching `(?i)bot$\|^auto\|moderator\|AutoModerator`, and any body containing "I am a bot" |
| Twitch bots | Drop Nightbot/StreamElements/Moobot command output; drop messages starting `!` |
| Copypasta | Frequency cap (§5.1 pass 3) rather than deletion |
| Link spam | Drop if >30% of characters are URL |
| Repetition | Drop if any 5-gram repeats >10x within one document |
| Promo | Drop `discord.gg/`, `onlyfans`, "check out my", "sub to my" |

## 5.4 Quality filtering (preserving variant)

Applied to internet/Gen-Z pools:

```
KEEP if:
  3 <= n_tokens <= 4096
  AND unique_token_ratio > 0.25          # not pure repetition
  AND non_alpha_ratio < 0.7              # not pure symbols (emoji-only allowed separately)
  AND longest_token_len < 60             # no base64/hashes
  AND NOT looks_like_code_or_markup
  AND NOT is_deleted_placeholder         # "[removed]", "[deleted]"
```

Note what is **absent**: no perplexity filter, no stopword-ratio filter, no punctuation requirement. Perplexity filters trained on formal text rank the most Gen-Z documents as the worst.

For the **general English** pool, use the standard strict Gopher/C4 rules instead — or simply take `smollm-corpus`, which is already filtered.

## 5.5 URL and HTML handling

- HTML entity unescape, then strip tags (`selectolax` — 10x faster than BeautifulSoup).
- Reddit markdown: keep `**bold**`/`*italic*` (they carry emphasis/tone), strip link syntax to the anchor text, **drop quote blocks `>`** (they duplicate parent comments).
- **URLs -> `http` placeholder**, following TimeLMs [V]. Keeps sentence structure, removes a huge tail of unique junk tokens.
- **`@username` -> `@user`**, also per TimeLMs [V]. This is simultaneously the URL filter and a PII control.
- `r/subreddit` and `u/user` -> keep `r/subreddit` (a real lexical item), map `u/x` -> `@user`.

## 5.6 Emoji handling

**Preserve exactly.** No stripping, no demojisation, no normalisation.

- Keep ZWJ sequences intact (family emoji, flags, skin tones) — the tokenizer handles them as byte sequences (§6).
- Cap runs at **8 identical emoji** (a 200-emoji run is spam; a 5-emoji run is emphasis).
- Twitch emotes (`KEKW`, `PogChamp`, `Sadge`, `OMEGALUL`) are **words**, not emoji. Force them into the tokenizer vocabulary as single tokens (§6.5).
- Count emoji density for the `emoji` register dimension **before** any capping.

## 5.7 Repeated characters

```python
# cap runs at 6, but RECORD the original length as register signal
elong_score = max_run_length(text)
text = re.sub(r'(.)\1{6,}', r'\1' * 6, text)
```

Rationale: `sooo` / `soooooo` / `sooooooooooooooo` are three intensities and the tokenizer should see the difference between the first two. Beyond ~6 the extra information is negligible, and uncapped runs blow up the vocabulary. **The pre-cap run length feeds the `elong` register dimension**, so nothing is lost.

## 5.8 Keysmash detection

Keysmash (`asdkjfhaskjdf`) is a real lexical item meaning "overwhelmed/laughing". Detect rather than delete:

```
is_keysmash = (len >= 6) AND (vowel_ratio < 0.2 OR home_row_ratio > 0.8)
              AND NOT in_dictionary AND NOT is_acronym
```
Replace with a canonical `<|keysmash|>` token, and set `elong=2`. This stops thousands of unique keysmashes from each eating a vocabulary slot.

## 5.9 Slang preservation and the dated lexicon

This is the piece the whole project turns on.

**Build `lexicon.jsonl` from `georgiyozhegov/urbandictionary`** (152,941 rows, `time` + `score` [V]):

```json
{"term": "rizz", "first_seen": "2021-08-14", "n_defs": 47, "median_score": 112,
 "recency_weight": 1.0, "status": "current", "aliases": ["rizzler","rizzed"]}
```

- `first_seen` = earliest `time` among that term's definitions.
- `recency_weight` = exponential decay on `first_seen`, half-life 3 years, so 2024 terms weigh ~1.0 and 2013 terms ~0.1.
- `status` in `{current, aging, dead}` from corpus re-attestation frequency in the 2025-2026 slice.
- Filter to `score > 0`, drop entries that are a single capitalised proper noun, drop entries whose definition names a person. Expect ~90K surviving terms [E].
- Merge in `MLBtrio` 1,779 curated entries as high-confidence seeds.

**Uses:**
1. **Mining filter** — score each candidate document by `sum(recency_weight of slang hits) / n_tokens`. High scores -> Gen-Z core pool.
2. **Register annotation** — the `slang` dimension of §3.4.
3. **Tokenizer forcing** — every `status=current` term must be a single token (§6.5).
4. **Eval generation** — `status=dead` terms become the outdated-slang test (§14.13).

**Never normalise slang to standard English.** Obvious, but half of all text-normalisation tooling does it by default.

## 5.9a Emerging-term discovery -- where *current* slang actually comes from

Because the dictionary stops in 2023 (§1.4), current slang is found by contrast against the corpus instead. A token is emerging slang when it is:

1. **frequent** in the recent slice (>= 2 occurrences per million tokens), and
2. **absent** from a standard English vocabulary (`wordfreq` top-50k), and
3. **unknown** to the historical lexicon.

Semantic drift on an existing term (`mid`, `cooked` acquiring pejorative senses) is handled by a fourth rule: a known English word qualifies only if it is *already in the lexicon* with a documented slang sense **and** its rate has grown >= 3x against an older baseline.

### Two failure modes found by running it [V]

Both are recorded here because both look like success until you read the output.

**(a) Discovery on raw text returns markup, not language.** First run's top "emerging slang" was:

```
https  3166/M   png  713/M   webp  489/M   redd  499/M   width  509/M   preview  505/M
```

These are URL and image-CDN fragments. **Cleaning must precede counting** -- §5.14 step ordering is not cosmetic. `bussin/data/clean.py::content_words` also maintains an explicit `MARKUP_NOISE` set for the artefacts that survive URL stripping.

**(b) An un-topic-matched baseline measures topic, not time.** After cleaning, the top of the list became ordinary English with large growth ratios:

```
anyone 1579/M (3.3x)   currently 486/M (5.6x)   wondering 478/M (6.2x)   team 424/M (4.5x)
```

Nothing about those words changed. What changed was the *subreddit mix*: the recent slice is an all-of-Reddit scrape and the baseline was three named subreddits (`gaming`, `Showerthoughts`, `relationship_advice`), so the ratio measured topic distribution.

**Fix [R]:** never accept a growth signal for a word that has no documented slang sense. Either topic-match the two slices (same subreddits, different time windows) or restrict the growth path to terms already in the lexicon. Bussin does the latter, because topic-matched slices are not available for the 2024-2026 window.

**(c) Unicode apostrophes fragment contractions.** `don`, `didn`, `doesn` appeared as emerging vocabulary because phone keyboards emit U+2019, not ASCII `'`, and the word regex split on it. `clean.py::UNICODE_PUNCT` normalises quotes and dashes before tokenization.

## 5.10 Profanity handling

**Keep it. Tag it. Do not delete it.**

Profanity is a core register marker in Gen-Z English; a model that has never seen it cannot understand casual speech, and will also fail to recognise when it is being insulted. Deleting profanity produces a model that reads as a corporate brand account — the exact cringe failure.

- Tag documents with `profanity_level in {0,1,2}` from a simple lexicon.
- Retain in pretraining. Use tags at SFT to control output.
- **Distinguish profanity from slurs.** Slurs are handled in §5.11, and are removed.

## 5.11 Toxicity, slurs, and PII

**Two separate systems. Do not conflate them.**

**Slur removal (hard, mandatory):**
- Curated slur list covering racial, ethnic, homophobic, transphobic, ableist terms, with common obfuscations (leetspeak, spacing, unicode homoglyphs).
- **Drop the entire document**, not just the token — context around a slur is usually also toxic.
- Run *before* dedup so hashes are computed on clean text.

**Toxicity filtering (soft, calibrated):**
- `Detoxify` (`unitary/toxic-bert`) scoring, document-level.
- **Threshold at 0.9, not the usual 0.5.** [R] Toxicity classifiers systematically over-flag AAVE and casual insult-as-affection (`shut up bro you're so dumb 💀` between friends). At 0.5 you strip the register and introduce dialect bias. At 0.9 you catch genuine abuse.
- Keep scores as metadata; a `toxicity` field lets you re-filter later without re-running the pipeline.

**PII removal (mandatory, on `body` not just schema):**

| Type | Action |
|---|---|
| Schema columns `author`, `author_fullname`, `permalink`, `id`, `subreddit_id`, `CommentID`, `VideoID` | **Drop at ingest**, before writing anything |
| `@handles` in text | -> `@user` |
| Emails | -> `<\|email\|>` |
| Phone numbers (intl + Indian formats) | -> `<\|phone\|>` |
| Credit-card-like digit runs | -> `<\|number\|>` |
| Street addresses | regex + `presidio` NER -> `<\|address\|>` |
| Person names in UD definitions | `presidio` PERSON entity -> drop the whole entry |
| IP addresses, API-key-shaped strings | -> `<\|redacted\|>` |

Tools: `microsoft/presidio-analyzer` for NER-based PII, `scrubadub` as a cross-check. **Run both and take the union** — recall matters more than precision here.

**Verification gate [R]:** sample 1,000 random post-pipeline documents and read them. If any contains a real name, handle, or contact detail, the pipeline fails and does not proceed. This is a manual gate in Phase 1 and it is not skippable.

## 5.12 Copyright handling

- Drop documents >2,000 tokens that are >80% quoted material (lyrics, article reprints).
- Lyrics detector: high line-repetition + short lines + known-artist-name proximity -> drop.
- Drop anything matching a Project Gutenberg / known-book n-gram index (cheap 13-gram bloom filter).
- Keep short quotations — they are normal discourse.

## 5.13 Synthetic-data detection

**You must be able to detect LLM output in "authentic" corpora**, because post-2023 Reddit contains a growing share of it, and it will poison the register.

Signals [E]:

| Signal | Threshold |
|---|---|
| Phrase markers | "delve into", "it's important to note", "as an AI", "I hope this helps", "in conclusion", "tapestry", "navigate the complexities" |
| Structural | Perfectly balanced bullet lists; every paragraph 3-4 sentences; em-dash density far above human baseline |
| Punctuation | Terminal punctuation on 100% of sentences in a casual-source document -> highly suspicious |
| Register mismatch | `formal=3` document sourced from Twitch/Discord -> flag |
| Perplexity | Unusually *low* perplexity under a reference small LM -> flag (LLM text is over-predictable) |

Combine into a logistic score; drop above 0.8; **record the flag rate per source per month** — a rising trend tells you when a source has become unusable.

## 5.14 Pipeline order (order matters)

```
1.  ingest + DROP PII COLUMNS                 (never write them to disk)
2.  HTML unescape + tag strip
3.  slur filter                               (before hashing)
4.  PII scrub on body text
5.  language ID -> route {en, hinglish, drop}
6.  bot / spam filter
7.  URL + @mention placeholders
8.  repeated-char cap (record elong first)
9.  keysmash canonicalisation
10. quality filter (preserving or strict per pool)
11. synthetic-data detector
12. toxicity score (tag, threshold 0.9)
13. exact dedup (xxhash bloom)
14. near-dup MinHash LSH (0.8 chat / 0.7 prose)
15. short-message frequency cap
16. ---- SPLIT train/val/test BY DOCUMENT ----
17. register annotation (§3.4) -> control tokens
18. tokenize -> uint16
19. pack to 2048-token sequences, document-boundary aware
20. shard, hash, manifest
```

**Steps 13-15 must precede 16**, or near-duplicates leak across the split. **Step 17 must follow 16**, or annotation statistics leak val information into train.

## 5.15 Tool summary

| Task | Tool |
|---|---|
| Streaming ingest | `datasets` (streaming=True), `pyarrow` |
| HTML | `selectolax` |
| Language ID | `fasttext` `lid.176.bin` |
| Exact dedup | `xxhash` + `pybloomfiltermmap3` |
| Near dedup | `datasketch` MinHashLSH, or `text-dedup` |
| PII | `presidio-analyzer` + `scrubadub` |
| Toxicity | `detoxify` |
| Emoji | `emoji`, `regex` (for grapheme clusters) |
| Tokenizer | `tokenizers` (HF, Rust) |
| Sharding | `numpy.memmap` |

---

# §6. Tokenizer design

## 6.1 Decision: train a new tokenizer

**[R] Train new. Do not reuse, do not extend.**

| Option | Verdict |
|---|---|
| Reuse GPT-2 BPE (50,257) | **No.** Trained on 2019 WebText. Fragments modern slang badly and wastes ~40% of vocabulary on formal-web artefacts. |
| Reuse Llama tokenizer (32,000) | **No.** SentencePiece over formal multilingual text. Emoji become 3-4 byte tokens each. |
| Extend an existing tokenizer | **No.** Added tokens get randomly-initialised embeddings that are poorly conditioned relative to pretrained ones. Fine for fine-tuning; pointless when training from scratch — you get all the downsides and none of the benefit. |
| **Train fresh byte-level BPE** | **Yes.** You are training from scratch; the tokenizer should match the corpus. |

**Measured motivation [E]:** GPT-2 BPE encodes `"ngl that fit is bussin fr 💀"` as roughly 14 tokens; a corpus-matched 32K BPE gets it to ~8. That is a ~40% effective-context and ~40% effective-compute saving on exactly the text you care about. On a fixed free-compute budget that is the single cheapest win available.

## 6.2 Algorithm comparison

| | Byte-level BPE | SentencePiece Unigram | WordPiece | Character/Byte |
|---|---|---|---|---|
| Unknown tokens | **Impossible** (byte fallback) | Possible without byte fallback | `[UNK]` | Impossible |
| Emoji/unicode | **Native** | Needs `byte_fallback=True` | Poor | Native but wasteful |
| Novel slang (`gyatt`) | Graceful subword split | Graceful | Poor | Fine but long |
| Whitespace handling | Explicit `Ġ` marker | `▁` marker | `##` prefix | n/a |
| Compression | **Best** | Very close | Good | Terrible |
| Speed | **Fastest** (Rust) | Fast | Fast | n/a |
| Ecosystem | GPT/Llama3/Mistral | Llama1-2/T5 | BERT | rare |

**Choice [R]: byte-level BPE**, via HF `tokenizers`. Reasons specific to this project:

1. **No `[UNK]`, ever.** Slang is an open vocabulary that changes after training. Byte fallback means a term invented in 2027 is representable, just less efficiently. Unigram without byte fallback would emit `[UNK]` and lose the information entirely.
2. **Emoji and ZWJ sequences are just bytes.** No special casing.
3. **Graceful degradation is the whole game here.** The tokenizer will be out of date the day it ships; BPE degrades smoothly.

## 6.3 Pre-tokenization regex

The default GPT-2 split pattern mangles internet text. Use a modified pattern:

```python
PATTERN = "|".join([
    r"<\|[a-z_0-9]+\|>",              # our special tokens, kept atomic
    r"[#@][\w_]+",                    # hashtags and mentions as single units
    r"r/[A-Za-z0-9_]+",               # subreddit references
    r":[a-z_]+:",                     # :emote: syntax
    r"\p{Extended_Pictographic}(‍\p{Extended_Pictographic})*[️‍]*",  # emoji w/ ZWJ
    r"'(?:[sdmt]|ll|ve|re)",          # contractions
    r" ?\p{L}+",                      # letters
    r" ?\p{N}+",                      # numbers
    r" ?[^\s\p{L}\p{N}]+",            # punctuation runs (keeps '???!!!' together)
    r"\s+(?!\S)", r"\s+",
])
```

Key deviations from GPT-2: hashtags/mentions/subreddits stay whole; emoji grapheme clusters stay whole; punctuation *runs* stay together so `???!!!` is one token rather than six.

## 6.4 Training corpus for the tokenizer

**[R] Do not train the tokenizer on the pretraining mixture proportions.** If you do, 80% general English means the tokenizer optimises for prose and under-serves slang.

Train on a **deliberately over-weighted** sample:

| Pool | Share of tokenizer training sample |
|---|---|
| Gen-Z core (Twitch/Discord/recent Reddit/YT) | **35%** |
| Internet broad (GECLM Reddit) | 25% |
| General English | 30% |
| Hinglish | 7% |
| Slang lexicon terms + examples | 3% |

Sample size: **5 GB of text** [R] — beyond this, BPE merges converge and you are just burning CPU.

## 6.5 Forced vocabulary

Seed the trainer with `initial_alphabet` + required tokens so these are guaranteed single tokens:

1. **All 1,300+ Unicode emoji** in the top-frequency band (from corpus counts).
2. **Every `status=current` lexicon term** (~3,000 after filtering) — `rizz`, `gyatt`, `delulu`, `mid`, `bussin`, `sigma`, `mogging`, `cooked`, `yap`, `aura`, ...
3. **Top 500 Twitch emotes** by corpus frequency — `KEKW`, `OMEGALUL`, `Sadge`, `PogChamp`, `monkaS`, `Pepega`.
4. **Top 800 abbreviations** — `ngl`, `tbh`, `istg`, `ong`, `fr`, `iykyk`, `nvm`, `wdym`, `hbu`, `smh`, `lmfao`, `pmo`.
5. **Elongation forms** for the top 100 elongatable words at lengths 3/4/6 — `sooo`, `soooo`, `soooooo`, `noooo`, `plsss`.
6. **Hinglish function words** — `hai`, `nahi`, `kya`, `yaar`, `bhai`, `matlab`, `acha`, `bohot`, `kar`, `raha`.

> Forcing these is worth it: ~5,000 reserved slots out of 32-49K, in exchange for every high-frequency Gen-Z item costing exactly one token.

## 6.6 Special tokens (32 reserved)

```
<|endoftext|> <|pad|> <|unk_never_used|>
<|reg|> <|/reg|>
<|slang_0|>..<|slang_4|>      (5)
<|emoji_0|>..<|emoji_3|>      (4)
<|abbrev_0|>..<|abbrev_3|>    (4)
<|elong_0|>..<|elong_2|>      (3)
<|formal_0|>..<|formal_3|>    (4)
<|user|> <|assistant|> <|system|> <|/turn|>
<|keysmash|> <|email|> <|phone|> <|address|> <|number|> <|redacted|>
```

## 6.7 Vocabulary size recommendations

**Hard constraint [R]: vocabulary must be <= 65,536** so tokens store as `uint16`. This halves corpus disk size versus `uint32` (70 GB instead of 140 GB) and halves dataloader I/O. On free compute that is not a micro-optimisation.

| Model | Vocab | Embedding params | % of total | Reasoning |
|---|---:|---:|---:|---|
| 125M | **32,768** | 25.2M | 25% | Larger vocab would make embeddings dominate a small model |
| 300M | **32,768** | 33.5M | 11% | Same vocab keeps tokenizer shared across the ladder |
| 400M / 500M | **49,152** | 62.9M | 15% | Better compression pays off once the model can afford it |
| 1B | **49,152** | 100.7M | 8.7% | Keep 49,152 — sharing the tokenizer with 400M lets you compare loss curves directly |

> **[R] Use ONE tokenizer across the whole ladder** if at all possible — ideally 49,152 for everything. Different tokenizers make loss values incomparable between models and force you to re-tokenize 70 GB. The only argument for 32,768 at 125M is embedding share; since the 125M is a throwaway validation model, **just use 49,152 everywhere** and accept 33% embedding share on the smallest model.

## 6.8 Evaluation of the tokenizer (gate before committing)

Before tokenizing 70 GB, verify on held-out text:

| Metric | Target |
|---|---|
| Bytes per token, general English | >= 4.0 |
| Bytes per token, Gen-Z core | **>= 3.2** (GPT-2 gets ~2.3 here) |
| `status=current` lexicon terms that are single tokens | **100%** |
| Top-500 emoji that are single tokens | 100% |
| Hinglish bytes/token | >= 2.8 |
| Fertility (tokens per whitespace word) on Twitch chat | <= 1.6 |
| Round-trip fidelity `decode(encode(x)) == x` | **100%** on 1M random docs including emoji/ZWJ |

The round-trip test is not optional. Byte-level BPE plus a custom regex is exactly where silent corruption of ZWJ emoji sequences happens.

---

# §7. Model architecture

## 7.1 Llama-style, and specifically why

**[R] Llama-style decoder-only. Not GPT-2 style.**

| Component | GPT-2 (2019) | **Bussin (Llama-style)** | Why it matters *here* |
|---|---|---|---|
| Norm | LayerNorm, post-norm | **RMSNorm, pre-norm** | **Critical:** you are training in fp16 without bf16 (§7.5). Pre-norm RMSNorm is dramatically more stable in fp16 than post-norm LayerNorm. This is the #1 cause of free-tier training divergence. |
| Activation | GELU | **SwiGLU** | ~1-2% better loss at equal params |
| Position | Learned absolute | **RoPE** | No position embedding params; extrapolates beyond trained context |
| Attention | MHA | **GQA** | Cuts KV cache 4-8x -> longer context on 16 GB T4 |
| Bias terms | Yes | **None** | Fewer params, marginally more stable |
| Embeddings | Untied | **Tied** (<=400M), untied (1B) | At 125M, tying saves 25% of parameters |

## 7.2 Configurations

All counts below are **computed exactly** from the config, not estimated [C].

| | `bussin-125m` | `bussin-300m` | `bussin-400m` | `bussin-1b` |
|---|---:|---:|---:|---:|
| **Total params** | **125,851,392** | **316,195,840** | **396,418,176** | **1,255,245,824** |
| Non-embedding params | 88,101,888 | 265,863,168 | 339,793,920 | 1,053,917,184 |
| Embedding params | 37,748,736 (30.0%) | 50,331,648 (15.9%) | 56,623,104 (14.3%) | 100,663,296 x2 (16.0%) |
| Layers | 14 | 24 | 24 | 24 |
| Hidden dim `d` | 768 | 1024 | 1152 | 2048 |
| Attention heads | 12 | 16 | 18 | 32 |
| **KV heads (GQA)** | 4 | 4 | 6 | 8 |
| Head dim | 64 | 64 | 64 | 64 |
| FFN dim (SwiGLU) | 2048 | 2752 | 3072 | 5440 |
| **Context length** | 1024 | 2048 | 2048 | 2048 |
| **Vocab** | 49,152 | 49,152 | 49,152 | 49,152 |
| Tied embeddings | Yes | Yes | Yes | **No** |
| RoPE theta | 10,000 | 10,000 | 10,000 | 10,000 |
| Norm | RMSNorm eps 1e-5 | same | same | same |
| Init std | 0.02 | 0.02 | 0.02 | 0.014 (`1/sqrt(d)`) |

> **Naming honesty [R]:** `bussin-1b` is 1.26B parameters, and `bussin-125m` is 125.9M. The labels are rounded product names; the exact counts are what goes in the model card and in every compute calculation in §10.

**Design notes [R]:**
- **Head dim fixed at 64** everywhere. It is the sweet spot for tensor-core kernels and for SDPA's efficient backend on Turing.
- **FFN dim = round(8/3 · d) to a multiple of 64.** Standard SwiGLU sizing that keeps parameter count comparable to a 4x GELU MLP.
- **Deep-and-narrow over wide-and-shallow.** 24 layers at 400M rather than 12 wider ones: depth helps compositional/pragmatic reasoning (irony, register), which is exactly this project's target. Cost is more sequential steps, which matters less than you would think since free-tier batches are small anyway.
- **1B untied embeddings**: at 1B the 100.7M embedding matrix is only 8.7% of params, and untying measurably helps; at 125M tying is clearly right.

## 7.3 Memory requirements

Adam requires 4 copies of parameters in fp32: weights, gradients, `exp_avg`, `exp_avg_sq` = **16 bytes/param**.

| Model | Weights | Grads | Adam | **Optimizer total** | Activations @bs=8 | **Peak** | Fits? |
|---|---:|---:|---:|---:|---:|---:|---|
| `bussin-125m` | 0.50 GB | 0.50 | 1.01 | **2.01 GB** | ~1.2 GB | **~3.2 GB** | 1x T4 easily |
| `bussin-300m` | 1.26 | 1.26 | 2.53 | **5.06 GB** | ~2.4 GB | **~7.5 GB** | 1x T4 |
| `bussin-400m` | 1.59 | 1.59 | 3.17 | **6.34 GB** | ~2.8 GB | **~9.2 GB** | 1x T4 (16 GB) — **comfortable** |
| **`bussin-1b`** | 5.02 | 5.02 | 10.04 | **20.08 GB** | ~4.5 GB | **~24.6 GB** | **DOES NOT FIT on one 16 GB T4** |

> **[C] The 1B constraint.** 20.08 GB of optimizer state alone exceeds a single T4's 16 GB. `bussin-1b` therefore **requires** either:
> - **FSDP / ZeRO-2 across both T4s** -> 10.04 GB/GPU optimizer state + activations, plus gradient checkpointing. Tight but workable.
> - **TPU v3-8** (8 cores x 16 GB = 128 GB HBM) -> trivially comfortable.
>
> This is an independent, hard, architectural reason the TPU backend is mandatory for the 1B and merely nice-to-have for the 400M.

## 7.4 Attention implementation — a free-tier gotcha

> **[V] FlashAttention-2 requires Ampere (sm80) or newer. The Tesla T4 is Turing (sm75) and the P100 is Pascal (sm60). Neither can run FlashAttention-2.**

Any tutorial telling you to `pip install flash-attn` on Kaggle is wrong and will fail to build or silently fall back.

**Use `torch.nn.functional.scaled_dot_product_attention` (SDPA)** and let PyTorch select the backend:

| Backend | T4 (sm75) | P100 (sm60) | TPU |
|---|---|---|---|
| `FLASH_ATTENTION` | no | no | n/a |
| `EFFICIENT_ATTENTION` (mem-efficient) | **yes — use this** | yes | n/a |
| `MATH` | yes (slow fallback) | yes | yes |
| XLA fused attention | n/a | n/a | **yes** |

```python
with torch.nn.attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## 7.5 Precision — the bf16 problem

> **[V] Neither P100 (Pascal) nor T4 (Turing) supports bfloat16.** bf16 tensor-core support begins with Ampere. TPU v3 supports bf16 natively.

This forces a specific, non-obvious setup:

| Backend | Master weights | Compute dtype | Loss scaling |
|---|---|---|---|
| T4 / P100 | **fp32** | **fp16** (`torch.amp.autocast`) | **`GradScaler` required** |
| TPU v3-8 | **fp32** | **bf16** | not needed |
| Local/CPU debug | fp32 | fp32 | not needed |

**Consequences [R]:**
1. **Checkpoints are always fp32.** That is what makes them portable between fp16-GPU and bf16-TPU sessions. Never checkpoint in the compute dtype.
2. **fp16 has ~5 decimal digits and a narrow exponent range.** Attention logits and RMSNorm reciprocals overflow easily. Mitigations: compute softmax in fp32, keep RMSNorm in fp32, clamp attention logits, and use `GradScaler` with `init_scale=2**14` (lower than the default `2**16`, which overflows on the first steps of a from-scratch run).
3. **Watch the scaler.** If `GradScaler` halves the scale repeatedly, you are diverging. Log `scaler.get_scale()` every step; a scale below `2**6` is an alarm.
4. **Never trust a loss curve that only looks fine in fp16.** Validate periodically in fp32.

## 7.6 Rejected architectural options

| Option | Why not |
|---|---|
| Mixture of Experts | Memory-bound on 16 GB; routing instability at small scale; no free-tier benefit |
| Mamba / SSM | Less mature tooling, no TPU story, and attention is not the bottleneck at 2048 context |
| Encoder-decoder | Wrong shape for open-ended generation |
| MQA (1 KV head) | Quality loss is real at <1B; GQA-8 gets most of the memory win |
| ALiBi | RoPE is better supported and extrapolates adequately |
| Sliding-window attention | Unnecessary at 2048 context |
| NoPE (as in SmolLM3) | Interesting, but unproven at this scale; keep the risk budget for the data work |

---

# §8. Training from scratch — full procedure

## 8.1 The twelve steps

```
1  tokenizer training           -> tokenizer.json  (Kaggle CPU, ~4 h, once)
2  corpus mining + cleaning     -> filtered parquet (Kaggle CPU x5, ~30 h wall, once)
3  register annotation          -> +control tokens
4  train/val/test split         -> by document, before tokenization
5  tokenization                 -> uint16 .bin shards (Kaggle CPU x5)
6  sequence packing + sharding  -> 100M-token shards + manifest
7  publish shards               -> HF dataset repo + Kaggle Dataset mirror
8  pretrain S1 (stable LR)      -> general English
9  pretrain S2 (stable LR)      -> ramp internet/Gen-Z
10 pretrain S3 (decay LR)       -> Gen-Z anneal, recency-ordered
11 SFT + DPO                    -> instruction tuning (§12)
12 convert + publish            -> safetensors + GGUF
```

## 8.2 Learning-rate schedule — WSD, not cosine

**[R] Use Warmup-Stable-Decay (WSD), not cosine.** This is the single most important scheduling decision for a quota-limited relay run.

| | Cosine | **WSD** |
|---|---|---|
| Requires total steps known upfront | **Yes** | **No** |
| Can extend training mid-run | No — reshaping the curve invalidates the schedule | **Yes** — just stay in the stable phase longer |
| Can branch multiple final models | No | **Yes** — decay from one stable checkpoint several ways |
| Matches a staged data curriculum | Poorly | **Exactly** — decay phase == Gen-Z anneal |

Your quota is "floating and demand-dependent" [V]. You genuinely do not know your total step count in advance. Cosine forces you to guess; guess low and you waste quota, guess high and you stop at a bad LR. WSD removes the guess.

```
phase        fraction   LR
warmup       0 - 1%     0 -> lr_max              linear
stable       1% - 85%   lr_max                   constant
decay        85% - 100% lr_max -> 0.1*lr_max     1-sqrt (or linear)
```

**The elegant part [R]:** the decay phase is *also* the Stage-3 Gen-Z anneal (§4). Low LR means small, careful updates; feeding the most Gen-Z, most recent data exactly then means the final model is shaped most strongly by contemporary register — while the general-English competence learned at high LR is preserved rather than overwritten. Data curriculum and LR schedule are the same schedule.

This mirrors MiniCPM's WSD and SmolLM2's staged annealing.

## 8.3 Hyperparameters

| | `bussin-125m` | `bussin-300m` | `bussin-400m` | `bussin-1b` |
|---|---:|---:|---:|---:|
| Total tokens | 10B | 22B | **30B** | **25B** |
| Context | 1024 | 2048 | 2048 | 2048 |
| **Global batch (tokens)** | 262,144 | 393,216 | **524,288** | **1,048,576** |
| **Total steps** | 38,147 | 55,945 | **57,220** | **23,842** |
| `lr_max` | 6.0e-4 | 4.0e-4 | 3.5e-4 | 3.0e-4 |
| `lr_min` (end of decay) | 6.0e-5 | 4.0e-5 | 3.5e-5 | 3.0e-5 |
| Warmup steps | 400 | 600 | 600 | 500 |
| Stable until step | 32,425 | 47,553 | 48,637 | 20,265 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| `betas` | (0.9, 0.95) | same | same | same |
| `eps` | 1e-8 | 1e-8 | 1e-8 | **1e-8** |
| Weight decay | 0.1 | 0.1 | 0.1 | 0.1 |
| WD exclusions | norms, biases, embeddings | same | same | same |
| Grad clip | 1.0 | 1.0 | 1.0 | 1.0 |
| Dropout | 0.0 | 0.0 | 0.0 | 0.0 |
| Z-loss | 1e-4 | 1e-4 | 1e-4 | 1e-4 |

> **Z-loss [R]:** add `1e-4 * logsumexp(logits)^2` to the loss. It costs nothing and it keeps logits from drifting large — which in **fp16 without bf16** (§7.5) is a real divergence risk, not a theoretical one. PaLM and OLMo both use it. On free-tier hardware it is close to mandatory.

## 8.4 Micro-batch / accumulation per backend

**The invariant: global batch size in tokens must be identical on every backend.** Otherwise the relay is not continuing one run, it is splicing several different runs together and the loss curve will show it.

`bussin-400m`, global batch 524,288 tokens, context 2048 = **256 sequences/step**:

| Backend | Devices | Micro-bs/device | Seqs/fwd | **Accum steps** | Total |
|---|---:|---:|---:|---:|---:|
| Kaggle T4 x2 | 2 | 4 | 8 | **32** | 256 |
| Kaggle P100 | 1 | 4 | 4 | **64** | 256 |
| Kaggle TPU v3-8 | 8 | 8 | 64 | **4** | 256 |
| Colab T4 (free) | 1 | 4 | 4 | **64** | 256 |
| Lightning L4 (free) | 1 | 8 | 8 | **32** | 256 |

`bussin-1b`, global batch 1,048,576 tokens = **512 sequences/step**:

| Backend | Devices | Micro-bs/device | Seqs/fwd | **Accum** | Notes |
|---|---:|---:|---:|---:|---|
| Kaggle T4 x2 | 2 | 2 | 4 | **128** | **FSDP required** + grad checkpointing |
| Kaggle TPU v3-8 | 8 | 8 | 64 | **8** | comfortable |

`bootstrap.py` computes `accum = global_batch_seqs // (n_devices * micro_bs)` at startup and **asserts it divides evenly**, refusing to start otherwise. Silent batch-size drift is the most likely way this project produces a quietly broken model.

## 8.5 Sequence packing

Concatenate documents with `<|endoftext|>` separators, then chunk to exactly `context` tokens. No padding, ~100% token efficiency.

**Cross-document attention [R]:** the strictly correct thing is a block-diagonal attention mask so a document cannot attend across an EOS into an unrelated one. The cheap thing is to not bother. **Do bother for Bussin**, because the corpus is full of *very short* documents (Twitch messages average ~12 tokens), so a 2048-token packed sequence may contain 150 unrelated messages. Without masking, the model learns spurious dependencies between random strangers' chat lines.

Implementation: pass a `document_ids` tensor and build the block-diagonal mask; SDPA accepts an additive float mask. Costs ~5% throughput. Worth it here specifically because of the short-document density.

For the general-English pool (long documents), masking barely matters — you may skip it there and keep the 5%.

## 8.6 Gradient checkpointing

| Model | Backend | Use it? |
|---|---|---|
| 125M / 300M / 400M | T4 | **No** — fits comfortably; costs ~30% throughput for nothing |
| 1B | T4 x2 | **Yes, mandatory** — required to fit |
| 1B | TPU v3-8 | **No** — 128 GB HBM |

Apply per-layer (`torch.utils.checkpoint` on each block), not globally.

## 8.7 Checkpointing

**Contents — everything needed for bit-comparable resume:**

```
ckpt-{step:08d}/
  model.safetensors           # fp32 master weights (sharded >5 GB)
  optimizer.pt                # AdamW exp_avg, exp_avg_sq (fp32)
  scheduler.json              # WSD phase, step, lr
  scaler.json                 # GradScaler scale + growth tracker (GPU only)
  rng.pt                      # python, numpy, torch CPU + CUDA, XLA seeds
  data_cursor.json            # {shard_id, token_offset, epoch, stage, shuffle_seed}
  metrics.jsonl               # loss history for curve-continuity checks
  config.yaml                 # full model + training config
  MANIFEST.sha256             # integrity
```

**Checkpoint sizes [C]:**

| Model | Weights fp32 | Optimizer fp32 | **Total** |
|---|---:|---:|---:|
| 125M | 0.50 GB | 1.01 GB | **1.51 GB** |
| 400M | 1.59 GB | 3.17 GB | **4.76 GB** |
| 1B | 5.02 GB | 10.04 GB | **15.06 GB** |

**Cadence [R]:**

| Trigger | Action |
|---|---|
| Every 90 min | Full checkpoint (crash insurance; max 90 min lost) |
| **T-20 min before session hard limit** | **Forced checkpoint + clean exit** (the watchdog, §9.4) |
| Every 2,000 steps | Milestone checkpoint, retained permanently |
| Rolling | Keep last 2 + all milestones; delete the rest to stay under the 100 GB HF private quota |

For `bussin-1b` at 15 GB/checkpoint, HF private 100 GB [V] holds 2 rolling + 4 milestones. Mirror milestones to Google Drive nightly (750 GB/day API cap [V] is not a constraint here).

## 8.8 Validation

- Every **500 steps**: 20 batches from `val` -> loss + perplexity. Cheap, catches divergence.
- Every **2,000 steps**: full `val` pass + `test-genz` (the time-sliced future-language split) + register-calibration probe (§15.3).
- Every **10,000 steps**: generate 20 fixed prompts at each of 5 register settings and log the text. **Read them.** Loss curves do not show cringe; text does.
- Always validate in **fp32**, never in the compute dtype.

## 8.9 Divergence detection and recovery

Hard alarms, checked every step:

| Condition | Action |
|---|---|
| `loss` is NaN/Inf | Halt, roll back to last milestone, resume with `lr * 0.5` |
| `GradScaler` scale < 2^6 | Alarm — fp16 underflow. Halt and inspect. |
| `grad_norm` > 10x its 100-step median | Skip the step, log it. >5 in 100 steps -> halt |
| `val_loss` rises for 3 consecutive checks | Halt, roll back |
| `loss` flat for 2,000 steps early on | LR likely too low, or data cursor stuck |

**Auto-rollback is part of the relay**, not a manual process: on a NaN the worker fetches the last milestone, sets `lr_scale=0.5` in `RUN_STATE.json`, and continues. A run that requires a human at 3 a.m. will not survive six months.

## 8.10 Final conversion

1. fp32 master -> `safetensors` fp16 for release (half the download, no quality loss at inference).
2. Wrap in a `transformers`-compatible `LlamaForCausalLM` config so `from_pretrained` works out of the box.
3. Export GGUF `Q8_0` and `Q4_K_M` via `llama.cpp` for local/Ollama use.
4. Publish tokenizer, `config.json`, model card with: exact param count, exact token count, tokens/param ratio, data composition, known limitations, AAVE provenance note (§3.2 L6), and BUSSBENCH scores.

---

# §9. Kaggle and free-compute strategy

## 9.1 Verified platform facts

All from the official Kaggle documentation pages, read directly [V]:

| Fact | Value | Source |
|---|---|---|
| CPU/GPU session runtime | **12 hours** | `kaggle.com/docs/notebooks` -> Technical Specifications |
| TPU session runtime | **9 hours** | same |
| `/kaggle/working` (auto-saved) | **20 GB** | same |
| Scratch outside working | **~60 GB**, not persisted | same |
| CPU-only spec | 4 cores, 30 GB RAM | same |
| **P100 spec** | 1x P100, 4 cores, **29 GB RAM** | same |
| **T4 x2 spec** | 2x T4, 4 cores, **29 GB RAM** | same |
| **TPU 1VM spec** | **96 cores, 330 GB RAM**, TPU v3-8 | same |
| Interactive idle timeout | 20 min (notebooks doc) / 60 min (efficient-GPU doc) — **docs disagree; assume 20** | both |
| **Weekly GPU quota** | **"30 hours or sometimes higher depending on demand"** | `kaggle.com/docs/efficient-gpu-usage` |
| Quota reset | Weekly, Saturday 00:00 UTC | Kaggle discussions [E] |
| **Weekly TPU quota** | **20 h/week, separate from GPU** | **[V] confirmed on the project account's settings page, 2026-09-18: `Kaggle TPU 00:00 / 20 hrs`** |
| Private dataset limit | **214.75 GB** (200 GiB), same for private models | **[V] confirmed on the account settings page, 2026-09-18** |
| Concurrent batch CPU sessions | **5** | Kaggle discussions [E] |
| Colab Pro linkage | +15 h (Pro) / +30 h (Pro+) | `kaggle.com/docs/notebooks` — **costs money, excluded** |

| **Free AI-model inference credits** | **$10/day, $100/month** | **[V] account settings page, 2026-09-18: `Daily AI Models $0.00 / $10.00`, `Monthly AI Models $0.00 / $100.00`** |
| Phone verification | Required before accelerators are granted | [V] account settings page |

> **[R] The inference credits change the instruction-tuning plan (§12.3).** $100/month of free model inference is enough to generate the standard-English paraphrase side of the ~24,000 translation pairs without paying for an API. It does **not** change the rule that the Gen-Z side must be authentic mined text (§13.2) -- the credits buy the safe direction only.

**Official efficiency guidance, quoted [V]:**
> *"Avoid using batch sessions (the commit button) to save or checkpoint your progress."*
> *"Stop interactive sessions prior to closing the window."*
> *"Consider using the Kaggle-API to avoid interactive sessions entirely."*

That last line is the operating model for this project: **drive everything through the Kaggle API, never open the editor.**

## 9.2 The single biggest lever: CPU sessions are free

> **GPU quota is a *GPU* quota.** CPU-only sessions do not consume it. With 5 concurrent batch CPU sessions x 12 h, you have **~60 CPU-hours per wave at zero cost to training.**

**Therefore: no data work ever happens in a GPU session.** Mining, cleaning, dedup, tokenization, sharding, and dataset publishing all run on CPU sessions. GPU/TPU sessions do exactly one thing: matrix multiplies.

This alone roughly doubles effective training throughput versus the naive approach of preparing data inside the training notebook.

## 9.3 Data movement architecture

```
  HF source datasets (streamed)
            |
            v
  [5x Kaggle CPU batch sessions]   <- filter, annotate, tokenize, shard.  0 GPU quota.
            |
            v
  HF dataset repo  bussin-corpus (private)     <- portable mirror for Colab / Lightning
            |
            v
  Kaggle Dataset x4 (18 GB each)               <- HOT PATH: mounts at /kaggle/input in ~0 s
            |
            v  np.memmap, zero-copy
  [GPU / TPU training session]
            |
            v  every 90 min + T-20 min watchdog
  HF private repo  bussin-ckpt  -->  nightly mirror --> Google Drive (cold archive)
```

**Why Kaggle Dataset for the hot path [R]:** attached datasets are *mounted*, not downloaded. Pulling a 20 GB corpus from HF at session start costs 10-20 min of a 12 h GPU session — about 2 hours of your 30 h/week burned on I/O. Mounting costs zero. This is worth roughly 7% of your entire compute budget.

**Why your PC and Google Drive are not in the path:** all transfers are cloud-to-cloud at datacenter speed. A 70 GB corpus routed through a home connection would take days and add nothing.

**Corpus storage [C]:** vocabulary 49,152 <= 65,536, so tokens are `uint16` = 2 bytes.

| Model | Tokens | Corpus size |
|---|---:|---:|
| `bussin-125m` | 10B | 20 GB |
| `bussin-400m` | 30B | **60 GB** |
| `bussin-1b` | 25B | (reuses the same shards) |

Build one **35B-token / 70 GB** corpus ordered by curriculum stage; smaller models read a prefix. Shards are 100M tokens / 200 MB each, ~350 shards.

## 9.4 The relay

### Coordination: a lease in `RUN_STATE.json`

A single JSON file in the private HF repo is the source of truth.

```json
{
  "run_id": "bussin-400m-v1",
  "step": 18450,
  "stage": "S2",
  "tokens_seen": 9674260480,
  "latest_ckpt": "ckpt-00018000",
  "lease": {
    "worker_id": "kaggle-tpu-a1b2",
    "platform": "kaggle-tpu-v3-8",
    "claimed_at": "2026-09-17T08:14:22Z",
    "expires_at": "2026-09-17T17:14:22Z"
  },
  "lr_scale": 1.0,
  "history": [{"worker": "...", "steps": [16000, 18000], "wall_s": 31400}]
}
```

**Claiming is compare-and-swap on the HF commit SHA:**

```
1. GET RUN_STATE.json, record its commit SHA
2. if lease.expires_at > now  -> another worker is live -> exit
3. write new state with our worker_id, parent_commit = recorded SHA
4. HF rejects the push if the SHA moved  -> someone else won the race -> exit
5. push accepted -> we hold the lease -> start training
```

This makes double-training structurally impossible without any server. It works identically whether the two workers are two platforms or two accounts.

**Heartbeat:** the holder extends `expires_at` every 10 min. If a session dies, the lease expires and the next worker takes over. Lease length = session length + 30 min.

### Worker lifecycle

```
bootstrap.py
  1. detect platform     (env vars: KAGGLE_KERNEL_RUN_TYPE, COLAB_GPU, LIGHTNING_*)
  2. detect device       (xla | cuda | cpu), device count, GPU name
  3. pin dependencies    (cached wheels; ~90 s)
  4. mount / locate corpus  (Kaggle: /kaggle/input; else: HF snapshot of needed shards only)
  5. claim lease         (exit cleanly if held)
  6. download latest ckpt (hf_transfer, parallel)
  7. compute accum       so global batch matches exactly; assert
  8. restore             model, optimizer, scheduler, scaler, RNG, data cursor
  9. verify              replay 10 steps of held-out data; assert loss within 1% of recorded
 10. train               with the watchdog armed
 11. on watchdog / signal / completion: checkpoint, release lease, exit 0
```

**Step 9 is the one people skip and regret.** A resume that silently loses optimizer state or reshuffles data looks fine for 200 steps and then plateaus. Asserting loss continuity catches it immediately.

### The watchdog

```python
HARD_LIMIT = {"kaggle-gpu": 12*3600, "kaggle-tpu": 9*3600,
              "colab": 4*3600, "lightning": 4*3600}   # conservative
RESERVE = 20*60   # time to checkpoint + upload

deadline = session_start + HARD_LIMIT[platform] - RESERVE
# checked every step; also on SIGTERM
```

Kaggle kills at the hard limit with no grace period. **Every session must end on its own terms.** For `bussin-1b`, a 15 GB checkpoint upload needs the full 20-minute reserve; measure your actual upload rate in Phase 3 and tune `RESERVE`.

### Maximising useful training time

| Waste | Fix | Saved/session [E] |
|---|---|---|
| Installing packages | Pin versions, prefer preinstalled, use a Kaggle Dataset of wheels for offline pip | 3-8 min |
| Downloading corpus | **Mount as Kaggle Dataset** | 10-20 min |
| Downloading checkpoint | `hf_transfer=1`, fetch only the latest, parallel | 2-6 min |
| Recompiling / warmup | Accept it on TPU (XLA compile ~2-4 min, once) | — |
| Idle interactive session | **Never use the editor.** Kaggle API `kernels push` only | up to 20 min |
| Dying at the hard limit, losing work | Watchdog | up to 90 min |
| Validation too often | Every 500 steps, 20 batches only | 2-5% |

**Target: >= 95% of wall-clock inside a session spent on training steps.** Measure it; log `train_seconds / session_seconds` in `RUN_STATE.history`.

### Chaining sessions automatically

Kaggle supports **scheduled notebooks** (daily/weekly) [V]. Combined with the lease, this gives hands-off operation: schedule the training notebook daily; it wakes, checks the lease and quota, trains until the watchdog fires, checkpoints, exits. If quota is exhausted it claims nothing and exits in seconds, costing nothing.

## 9.5 Platform portfolio

**On multiple accounts — stated once, plainly.** Kaggle's Terms of Use (version June 22, 2025, active) state [V]:

> *"You also may not have, control, or operate under more than one active Kaggle account or Kaggle User ID. If we determine that you have, control, or are operating under more than one Kaggle account or Kaggle User ID, we may take action without notice, including banning your user account, revoking access to your Kaggle User ID, and disqualifying you from any ongoing Competition(s)."*

The risk is losing every account at once, mid-run, with the checkpoints stranded. The relay is built platform-agnostic and credential-agnostic — it works with whatever you point it at — but the **recommended configuration is one account per platform**, which the compute budget below shows is sufficient:

| Platform | Accelerator | Quota | Session | Verified? |
|---|---|---|---|---|
| Kaggle | T4 x2 / P100 | **30 h/wk** | 12 h | [V] |
| Kaggle | **TPU v3-8** | **~20 h/wk** | 9 h | [E] — verify in settings |
| Colab free | T4 | dynamic, ~10-15 h/wk | ~4 h, pre-emptible | [E] |
| Lightning AI free | L4 / T4 | ~15-22 h/mo credits | varies | [E] |

**One account per platform: ~50 h/week from Kaggle alone, ~65-75 h/week including the others [E].**

## 9.6 Realistic training throughput

Effective sustained throughput, as fraction of peak [E] — **these must be re-measured in Phase 3:**

| Hardware | Peak fp16/bf16 | MFU [E] | **Effective** | Notes |
|---|---:|---:|---:|---|
| **P100** | 19.0 TF (no tensor cores) [V] | 18-29% | **3.5-5.5 TF/s** | **Avoid.** ~6x slower than T4 x2 |
| **T4 x1** | 65 TF (tensor cores) [V] | 18-28% | 12-18 TF/s | 70 W part; thermally throttles |
| **T4 x2** | 130 TF | — | **24-36 TF/s** | ~1.75x scaling over PCIe DDP |
| **TPU v3-8** | ~420 TF bf16 | 21-33% | **90-140 TF/s** | Native bf16; 128 GB HBM |
| Colab T4 | 65 TF | 18-28% | 12-18 TF/s | pre-emptible |
| Lightning L4 | 121 TF | 25-35% | 30-42 TF/s | **supports bf16** (Ada) |

> **[R] Never select P100.** It is the Kaggle default in many tutorials and it is the worst option available: no tensor cores, ~6x slower than T4 x2 for the same quota hour. Always choose `T4 x2`.

---

# §10. Compute estimates

## 10.1 Method

`FLOPs = 6 * N * D`, with `N` = total parameters (including embeddings) and `D` = training tokens. This is the standard Kaplan/Hoffmann accounting: 2 FLOPs per parameter for the forward pass, 4 for the backward. Attention FLOPs are excluded, which understates by roughly `12 * L * ctx * d` per token — under 5% at 2048 context for these shapes, so the estimates below are mildly optimistic. [C]

`GPU-hours = FLOPs / (effective_TF * 1e12 * 3600)`.

## 10.2 Per-model results

| Model | **N [C]** | **D** | **tok/param** | **FLOPs [C]** |
|---|---:|---:|---:|---:|
| `bussin-125m` | 125,851,392 | 10B | 79.5 | **7.55e18** |
| `bussin-300m` | 316,195,840 | 22B | 69.6 | **4.17e19** |
| `bussin-400m` | 396,418,176 | 30B | 75.7 | **7.14e19** |
| `bussin-1b` | 1,255,245,824 | 25B | 19.9 | **1.88e20** |

### Hours required

| Model | T4 x2 (24-36 TF) | TPU v3-8 (90-140 TF) | P100 (3.5-5.5 TF) |
|---|---:|---:|---:|
| `bussin-125m` | **58 - 87 h** | 15 - 23 h | 381 - 599 h |
| `bussin-300m` | **322 - 483 h** | 83 - 129 h | 2,108 - 3,313 h |
| `bussin-400m` | **551 - 826 h** | 142 - 220 h | 3,604 - 5,663 h |
| `bussin-1b` | **1,453 - 2,179 h** | 374 - 581 h | 9,509 - 14,943 h |

### Calendar time, one Kaggle account (30 GPU-h + 20 TPU-h per week)

Weekly budget: `20 h x 115 TF + 30 h x 30 TF = 1.15e19 FLOPs/week` [C, midpoint].

| Model | **Weeks** | **Months** | Verdict |
|---|---:|---:|---|
| `bussin-125m` | **0.7** | 0.2 | Trivial |
| `bussin-300m` | **3.6** | 0.8 | Easy |
| `bussin-400m` | **6.2** | **1.4** | **Very comfortable** |
| `bussin-1b` | **16.3** | **3.8** | **Achievable** |

### Adding Colab + Lightning [E]

Add ~25 h/week at ~25 TF average = `+2.25e18 FLOPs/week` -> `1.38e19/week` total:

| Model | Weeks | Months |
|---|---:|---:|
| `bussin-400m` | 5.2 | 1.2 |
| `bussin-1b` | 13.7 | **3.2** |

## 10.3 What 30 GPU-hours/week alone buys — the honest version

GPU quota only, **no TPU** (T4 x2, midpoint 30 TF/s):

| Horizon | GPU-h | FLOPs | `bussin-125m` | `bussin-400m` | `bussin-1b` |
|---|---:|---:|---|---|---|
| **1 month** | 120 | 1.30e19 | 17.2B tok (**136x**) | 5.5B tok (13.7x) | 1.7B tok (**1.4x**) |
| **3 months** | 390 | 4.21e19 | 55.8B (443x) | 17.7B (**44.6x**) | 5.6B (4.5x) |
| **6 months** | 780 | 8.42e19 | 111.6B (887x) | 35.4B (89.3x) | 11.2B (**8.9x**) |
| **12 months** | 1,560 | 1.68e20 | 223B (1774x) | 70.8B (178x) | 22.3B (17.8x) |

## 10.4 "Technically possible" vs "properly trained"

This distinction is the one you asked for, and it is the most important table in the document.

| | Definition | Threshold |
|---|---|---|
| **Technically possible** | The training loop runs to completion without OOM or divergence | any D > 0 |
| **Minimally trained** | Loss has left the steep phase; grammatical output | ~10 tok/param |
| **Chinchilla-optimal** | Best loss for the compute spent (Hoffmann et al. 2022: 20 tok/param) | **20 tok/param** |
| **Properly trained (small-model regime)** | Over-trained past optimal; the regime where small models become genuinely useful | **>= 70 tok/param** |
| **State of the art for size** | SmolLM2-135M territory | ~15,000 tok/param [V] |

Applying it:

| Model | Plan | tok/param | Status |
|---|---|---:|---|
| `bussin-125m` | 10B | 79.5 | **Properly trained** |
| `bussin-300m` | 22B | 69.6 | **Properly trained** (just) |
| `bussin-400m` | 30B | 75.7 | **Properly trained** |
| `bussin-1b` | 25B | **19.9** | **Chinchilla-optimal — NOT over-trained** |
| `bussin-1b` on GPU quota alone, 6 months | 11.2B | **8.9** | **Below Chinchilla. Undertrained.** |

> **The straight answer on 1B.**
>
> On GPU quota alone, six months of work produces a 1B model at **8.9 tokens/parameter** — less than half of Chinchilla-optimal. It would run, generate text, and demo fine. It would also **lose to `bussin-400m` on essentially every axis**, because the 400M would have had 8.5x more tokens per parameter. Shipping it would be the exact failure mode you asked me to avoid: a model that "completed a few billion tokens" and is called trained.
>
> **With the TPU quota, `bussin-1b` reaches 19.9 tokens/parameter in ~16 weeks.** That is genuinely Chinchilla-optimal, and the model is real. It still will not be *over*-trained — it will have more knowledge and less polish than the 400M.
>
> **Therefore: the TPU backend is not an optimisation for the 1B, it is the precondition.** If the PyTorch/XLA work stalls, cancel the 1B rather than running it on GPU quota alone.

## 10.5 Expected limitations at each size

| Model | What it will do well | What it will not do |
|---|---|---|
| `bussin-125m` | Register control, slang comprehension, short casual replies | Reasoning, facts, multi-turn coherence, anything > 2 sentences |
| `bussin-300m` | The above + style transfer, slang explanation | Reasoning, reliable factuality |
| `bussin-400m` | The above + coherent multi-turn chat, Hinglish switching, good register calibration | Math, code, factual QA, long-context |
| `bussin-1b` | The above + noticeably more world knowledge and fewer non-sequiturs | Still not a general assistant. No math, no code, limited facts |

**[R] Tell users this in the model card.** A 400M model marketed as a chatbot disappoints; the same model marketed as *"the best open model at understanding and generating contemporary internet English, at 400M parameters"* is genuinely interesting and defensible.

---

# §11. Training strategy — which option

## 11.1 The four options assessed

### Option A — train entirely on Gen-Z data

**Verdict: impossible, and would be wrong even if possible.** [V]

Total contemporary Gen-Z text in existence: **~3.3B tokens** (§4.0). `bussin-400m` needs 30B. You would need ~9 epochs of the same 3.3B tokens, well past the point where repetition stops helping.

Worse, the failure is conceptual. Gen-Z English is a *modulation of* standard English. A model that has only seen Twitch chat has no model of the thing being modulated. It could produce slang but could not explain it, translate from it, or know when to stop.

### Option B — pretrain general, gradually increase Gen-Z

**Verdict: correct direction, incomplete.** Gets the curriculum right but has no mechanism for *control*. The model ends up with a marginal distribution averaged over registers, which is precisely how you get "Dear Sir, no cap." It also has no LR-schedule story, so the anneal is unprincipled.

### Option C — general -> Gen-Z continued pretraining -> instruction tuning

**Verdict: the right skeleton.** [R] This is the SmolLM2 / MiniCPM / OLMo pattern and the evidence base is strong. But as usually stated it has two gaps: (a) no register conditioning, (b) treats continued pretraining as a separate run rather than as the decay phase of one schedule.

### Option D — Bussin's variant: **Option C + register conditioning + WSD-aligned anneal**

**Verdict: recommended.** Three changes to Option C:

1. **Register control tokens on every document from step 0** (§3.4), including the general-English data. The model learns `p(text | register)` rather than `p(text)`. This is what makes control possible and cringe measurable.
2. **The Gen-Z anneal *is* the LR decay phase** (§8.2). One run, one schedule, no separate continued-pretraining job. Data curriculum and LR curriculum are the same curve.
3. **Recency ordering inside the anneal.** Within Stage 3, documents are ordered oldest-to-newest, so the final ~1% of training is 2025-2026 language at the lowest LR. Slang freshness becomes a *position in the curriculum*.

## 11.2 Evidence for the recommendation

| Claim | Evidence |
|---|---|
| Domain pretraining beats adapting a general model | **BERTweet** [V] — from-scratch on 850M tweets beat RoBERTa-base and XLM-R-base on every social-media task |
| Language drifts and models degrade on post-cutoff text | **TimeLMs**, arXiv:2202.03829 [V] — measured perplexity degradation over time; motivates recency ordering |
| Staged data curricula with late high-quality annealing work | SmolLM2 (arXiv:2502.02737) [V], MiniCPM WSD, OLMo 2 mid-training |
| Repeated data is ~as good as fresh up to ~4 epochs | Muennighoff et al., NeurIPS 2023 — justifies reusing the 3.3B Gen-Z core |
| 20 tok/param is compute-optimal | Hoffmann et al. 2022 [V] |
| Over-training small models far past optimal pays off | SmolLM2-135M at ~15,000 tok/param [V] |

## 11.3 Retaining normal English — the explicit mechanism

You asked for a model that keeps normal English while gaining Gen-Z. Four mechanisms, each independently sufficient to notice a failure:

1. **77-82% of all training tokens are standard English** (§4). Gen-Z is a minority register by design.
2. **A formal anchor is present in every stage** — FineWeb-Edu, `AskHistorians`, `explainlikeimfive` never drop to zero, including during the anneal.
3. **Register conditioning makes formal English a *labelled target*.** `<|slang_0|> <|formal_3|>` documents are ~70% of the corpus, so producing clean formal English is a directly reinforced behaviour, not a residue.
4. **`test-general` perplexity is a hard gate** (§14.11): if general-English perplexity degrades more than **5%** during the Stage-3 anneal, the anneal is too aggressive — reduce the Gen-Z share and re-run the decay from the last stable checkpoint. WSD makes that cheap, because the stable-phase checkpoint is still valid.

> Point 4 is the safety valve that makes the whole design low-risk: **the decay phase is short and re-runnable.** If the anneal produces a cringe model, you have not lost the run — you re-decay from the stable checkpoint with different data proportions. That is worth ~1-2 weeks of compute, not four months.

---

# §12. Instruction tuning

## 12.1 Format

One format for everything, reusing the pretraining register tokens so SFT is continuous with pretraining rather than a distribution shift:

```
<|system|>
<|reg|><|slang_2|><|emoji_1|><|abbrev_2|><|elong_0|><|formal_1|><|/reg|>
<|/turn|>
<|user|>
translate this to how you'd actually say it: "I am extremely tired and would like to sleep."
<|/turn|>
<|assistant|>
im so cooked rn i need to knock out fr
<|/turn|>
```

Rules:
- Loss is computed **on assistant turns only**.
- Register tokens appear in the system turn and are **sampled to match the target response**, so the model keeps learning the conditional.
- Multi-turn examples keep full history; sequences are packed to 2048 with document masking (§8.5).
- ~15% of examples carry **no** register block, teaching the model to infer register from context (§15.7) — this is the anti-cringe behaviour.

## 12.2 Composition — 120,000 SFT examples

| # | Task | Examples | Source |
|---|---|---:|---|
| 1 | **Standard English -> Gen-Z** | 12,000 | Mined pairs (§12.3), register-graded at 3 levels |
| 2 | **Gen-Z -> standard English** | 12,000 | Same pairs, reversed |
| 3 | **Gen-Z conversation** (multi-turn) | 20,000 | Reddit/Discord threads reformatted as dialogue |
| 4 | **Slang explanation** | 6,000 | Dated lexicon + real usage context + etymology |
| 5 | **Slang generation** ("give me a word for X") | 3,000 | Lexicon reverse-indexed by definition |
| 6 | **Contextual slang interpretation** | 5,000 | Polysemous terms in 2+ contexts ("mid", "cooked", "bussin") |
| 7 | **Emoji interpretation** | 4,000 | Mined emoji-in-context -> meaning |
| 8 | **Internet-language understanding** | 6,000 | Comprehension QA over mined social text |
| 9 | **Tone transformation** | 8,000 | Same content at 5 register levels |
| 10 | **Casual conversation** | 15,000 | `WildChat-1M` + `oasst2`, register-relabelled |
| 11 | **Hinglish** | 8,000 | `findnitai` (human-flagged) + `cmu_hinglish_dog` + mined |
| 12 | **Formal -> casual and back** | 6,000 | Paired register transforms |
| 13 | **Detecting inappropriate slang use** | 4,000 | "is this the right word here?" + correction |
| 14 | **Outdated slang** | 3,000 | `status=dead` lexicon terms; correct answer names the era |
| 15 | **Genuine vs forced slang discrimination** | 4,000 | Real mined text vs LLM-generated slang; model must tell them apart |
| 16 | **General instruction following** | 4,000 | `oasst2` / `ultrachat` — keeps basic helpfulness |
| | **Total** | **120,000** | |

**Why 120,000 [R]:** for a 400M model, SFT sets below ~20K underfit the task formats and above ~200K start overwriting pretrained knowledge. 100-150K is the band where small models learn format without losing the base. 2-3 epochs at LR 1e-5 with a short cosine decay.

Tasks **13, 14, 15** are the ones nobody builds and are where most of the anti-cringe capability comes from.

## 12.3 How to mine parallel pairs without a translation dataset

Since §1.5 shows every existing Gen-Z translation dataset is dated synthetic junk, generate your own from *authentic* text:

1. Take mined Gen-Z-core documents with `slang >= 2` (authentic, contemporary).
2. Use a strong instruct model to produce the **standard-English paraphrase** (the direction that is safe — a formal paraphrase of real slang is reliable).
3. **Never generate in the other direction.** Asking an LLM to produce slang is what creates cringe, because the LLM's slang is 2-3 years stale.
4. Result: `(standard, genz)` pairs where **the Gen-Z side is real human text** and only the standard side is synthetic.
5. Train tasks 1 and 2 from the same pairs in both directions.

> This inverts the standard approach and is the single most important data decision in the SFT stage. **Authenticity belongs on the Gen-Z side.**

Quality gate: sample 500 pairs, check the paraphrase preserves meaning, and reject any pair where the model's paraphrase hallucinated content.

## 12.4 DPO — 20,000 preference pairs

SFT teaches what to do; DPO teaches what **not** to do. Positive-only data cannot teach restraint.

| Preference axis | Pairs | `chosen` | `rejected` |
|---|---:|---|---|
| Register match | 6,000 | Slang density matching the request | Too much or too little |
| Natural vs forced | 5,000 | Real mined phrasing | LLM-generated slang salad |
| Current vs dated | 3,000 | Contemporary terms | "lit", "YOLO", "on fleek" |
| Context sensitivity | 3,000 | Formal reply to formal prompt | Slang injected into a formal context |
| Slang restraint | 3,000 | Plain answer when slang adds nothing | Gratuitous "no cap" / "fr" |

`beta = 0.1`, LR 5e-7, 1 epoch. Keep an SFT-reference KL penalty so DPO does not degrade fluency.

**Where `rejected` samples come from:** generate them with the SFT model at high temperature and high slang setting, then have the dated lexicon and register annotator *automatically* label the failures. This is cheap and scales, because cringe is measurable (§15.3).

---

# §13. Synthetic data

## 13.1 Should you use it?

| Stage | Synthetic? | Share |
|---|---|---|
| **Pretraining** | **No** | **0%** |
| **SFT** | Yes, structurally | ~55% of examples have a synthetic component |
| **DPO `rejected`** | Yes, deliberately | ~100% — synthetic cringe is exactly what you want to train against |

## 13.2 Why 0% synthetic in pretraining

**The staleness compounding argument [R].** Synthetic Gen-Z is generated by an LLM whose own knowledge of Gen-Z was frozen at *its* pretraining cutoff, filtered through its instruction tuning (which rewards safe, legible, slightly-explained slang). So synthetic Gen-Z is:

- **2-3 years stale** — verified: the synthetic sets in §1.5 use "lit", "YOLO", "on fleek", "vibes on point".
- **Over-explained** — real slang is used without gloss; LLM slang tends to be self-clarifying.
- **Register-flattened** — LLMs produce a narrow band of "moderately slangy", missing both the extremes that define the distribution.
- **Denatured** — the AAVE grammar underneath much of the lexicon is usually stripped, leaving standard syntax with slang nouns bolted on. That is precisely what "adult pretending to be Gen-Z" sounds like.

Training on it means learning the *model's impression* of Gen-Z instead of Gen-Z. And once in the corpus it is indistinguishable from authentic data, so the error is unrecoverable.

**The exception [R]:** synthetic **general English** (`cosmopedia`) is fine and included at ~12% of the general pool. The objection is specific to synthetic *register*, not synthetic text.

## 13.3 What synthetic data is legitimately for

1. **The standard-English side of translation pairs** (§12.3) — safe direction.
2. **Task scaffolding** — instructions, questions, and explanations *about* slang.
3. **DPO negatives** — you *want* synthetic cringe here; it is the training signal.
4. **Eval distractors** — BUSSBENCH's "genuine vs forced" task needs machine-written slang as the negative class.

## 13.4 Preventing artificial Gen-Z from leaking in

| Control | Mechanism |
|---|---|
| Pipeline gate | §5.13 synthetic detector runs on **all** pretraining data; drop above 0.8 |
| Provenance tags | Every shard carries `is_synthetic` in `manifest.json`; the dataloader asserts 0% synthetic in pretraining shards |
| Source-level rule | Any dataset whose card says "generated with GPT/Claude/Gemini" is barred from pretraining by name |
| Recency monitor | Track synthetic-flag rate per source per month; a rising curve means that source is becoming unusable |
| Held-out audit | 1,000 random pretraining documents read by a human in Phase 1; any that read as LLM output fails the gate |

## 13.5 Detecting synthetic contamination after the fact

If you suspect the corpus is contaminated, measure it rather than guessing:

1. **Slang-age histogram.** Plot the `first_seen` distribution of slang terms in the corpus. Authentic 2026 text has a long tail into 2024-2026. Synthetic text spikes at 2015-2020. **A missing recent tail is the signature of contamination.**
2. **Type-token ratio of slang.** Authentic corpora use many rare terms; synthetic corpora reuse a small set. Low slang-TTR is a red flag.
3. **Punctuation completeness by source.** Twitch chat with 90% terminal punctuation did not come from Twitch.
4. **Perplexity under a reference model.** Synthetic text is over-predictable; a left-shifted perplexity distribution for a casual source is suspicious.

---

# §14. Evaluation — BUSSBENCH

## 14.0 Construction rules

**BUSSBENCH is built before the final training run and never touched by it.**

1. Source items from text **postdating** the corpus cutoff, or hand-written.
2. Every item is checked against the training corpus by **13-gram containment**; any hit is rewritten.
3. The benchmark is stored in a **separate repo** with no path into any data pipeline.
4. Item hashes are recorded in `BUSSBENCH/manifest.json`; the training dataloader asserts none appear.
5. Held-out **by time**: a live slice is collected monthly from *after* the model's cutoff, so drift is measurable for the model's whole life (the TimeLMs method).

Size: **~3,500 items** across 14 tasks.

## 14.1 Task suite

| # | Task | Items | Metric | Auto/Human |
|---|---|---:|---|---|
| 1 | **Language modelling** — perplexity on `test-genz`, `test-general` | 5M tok | PPL | Auto |
| 2 | **Slang understanding** — MCQ: meaning of term in context | 500 | Accuracy | Auto |
| 3 | **Slang generation** — produce a term fitting a definition | 200 | Lexicon match + human | Both |
| 4 | **Conversational quality** — 5-turn dialogue | 150 | Human 1-5 + LLM judge | Both |
| 5 | **Contextual slang interpretation** — polysemous terms | 300 | Accuracy | Auto |
| 6 | **Style transfer** — both directions | 400 | Meaning preservation + register shift | Both |
| 7 | **Emoji understanding** — MCQ on emoji meaning in context | 250 | Accuracy | Auto |
| 8 | **Internet-language comprehension** — QA over social text | 300 | F1 | Auto |
| 9 | **Code-switching** — mixed-register input | 200 | Accuracy | Auto |
| 10 | **Hinglish** — comprehension + generation | 250 | Accuracy + human | Both |
| 11 | **Normal English retention** — HellaSwag, ARC-e, PIQA, LAMBADA subsets | 1,000 | Accuracy | Auto |
| 12 | **Toxicity / safety** — RealToxicityPrompts subset + slur-elicitation probes | 300 | Toxicity rate | Auto |
| 13 | **Hallucination** — factual QA with "I don't know" option | 200 | Accuracy + abstention rate | Auto |
| 14 | **Outdated slang** — classify current / aging / dead | 250 | Accuracy | Auto |
| 15 | **Cringe suite** | 350 | See §15 | Both |

## 14.2 Slang understanding (task 2) — item format

```json
{"id": "su_0142",
 "context": "bro really pulled up in a 2009 civic talking bout his whip is clean lowkey he cooked",
 "query": "cooked",
 "options": ["did well / succeeded", "prepared food", "was defeated or embarrassed", "left quickly"],
 "answer": 0,
 "note": "polysemy trap: 'cooked' normally means ruined, but 'he cooked' = he did well",
 "first_seen": "2022-03", "status": "current"}
```

Deliberately include **polysemy traps** where the common meaning is wrong. That is what distinguishes a model that has seen slang in context from one that memorised a dictionary.

## 14.3 Normal English retention (task 11) — the hard gate

Run standard benchmark subsets before and after the Stage-3 anneal.

> **Gate [R]: if `test-general` perplexity degrades by more than 5%, or HellaSwag/ARC-e drop more than 2 points, the anneal is rejected.** Re-decay from the stable checkpoint with a lower Gen-Z share. Because of WSD (§8.2) this costs 1-2 weeks, not the whole run.

## 14.4 Human evaluation

Automatic metrics cannot detect cringe. Budget for human evaluation and treat it as a first-class deliverable.

| | |
|---|---|
| **Raters** | 5+ people aged 16-26 who actually use this register daily. **Not you, and not your colleagues.** |
| **Blind** | Outputs from `bussin-*`, a base Llama/SmolLM, a slang-prompted GPT-class model, and **real human text**, shuffled |
| **Items** | 150 generations per model |
| **Scale** | 1-5 on: naturalness, register appropriateness, humour, "would a real person write this" |
| **The key test** | **Human-vs-model discrimination.** Raters guess which outputs are human. A model whose outputs are guessed correctly >80% of the time has a register problem. |
| **Cadence** | Once at 400M, once at 1B, once after DPO |

The discrimination test is the only evaluation that directly measures the thing the project is about.

## 14.5 Automatic proxies (cheap, run every 2,000 steps)

| Metric | Definition | Target |
|---|---|---|
| **Register calibration error (RCE)** | mean abs diff between requested and measured register vector | **< 0.4** |
| **Slang density distribution distance** | Wasserstein distance between model output slang-density histogram and held-out human histogram, at matched register | **< 0.15** |
| **Slang freshness** | median `first_seen` year of slang terms produced | within 1 year of corpus median |
| **Slang type-token ratio** | unique slang terms / total slang tokens over 1,000 generations | **>= 0.35** (low = repetitive) |
| **Formal-mode slang leakage** | slang tokens per 100 tokens at `<\|slang_0\|>` | **< 0.2** |
| **Repetition rate** | 4-gram repetition across generations | < human baseline + 10% |

These are all computable from the dated lexicon and the register annotator — the same code used to build the corpus. That is the payoff of building the annotator: **evaluation is nearly free.**

---

# §15. Avoiding cringe

## 15.1 Definition

> **Cringe is register mismatch, not slang.**

Nobody cringes at a teenager saying "that's so cooked" to a friend. They cringe when a bank's Twitter account says it. The words are identical; the register-context pairing is wrong. Therefore cringe is not reduced by using *less* slang — it is reduced by using the *right amount for the context*, which may be zero or may be a lot.

This reframing matters because the obvious fix ("turn down the slang") produces a model that is bland at every setting instead of natural at each one.

## 15.2 The six failure modes, and the mechanism against each

| # | Failure | Mechanism |
|---|---|---|
| 1 | **Overuses slang** | Register conditioning: 70% of pretraining is `slang_0`, so restraint is directly trained |
| 2 | **Random "bro", "fr", "no cap"** | Slang-TTR metric + DPO "slang restraint" axis penalises filler |
| 3 | **Outdated slang** | Dated lexicon; freshness metric; DPO "current vs dated" axis; recency-ordered anneal |
| 4 | **Unnatural combinations** | Trained on real co-occurrence, never on synthetic slang (§13.2) |
| 5 | **Adult-pretending-to-be-Gen-Z** | Zero synthetic register in pretraining — this failure *is* synthetic data |
| 6 | **Repetitive internet phrases** | Frequency capping (§5.1 p3) + repetition metric + TTR gate |

## 15.3 Register Calibration Error — the headline metric

```
For each of 500 prompts x 5 requested register vectors r:
    generate y ~ model(prompt, r)
    r_hat = register_annotator(y)         # same code as corpus annotation
    error = mean(|r - r_hat|)             # over the 5 dimensions
RCE = mean over all (prompt, r)
```

**Target: RCE < 0.4** (on 0-4 scales). Report per-dimension, because failure is usually concentrated: a model may nail `emoji` and fail `formal`.

**Why this is the right headline number [R]:** it is a single scalar that goes *up* both when the model under-uses slang and when it over-uses it. Optimising it cannot be gamed by suppressing slang, which is exactly the property "average slang density" lacks.

## 15.4 Naturalness over slang density — the explicit objective

The optimisation target is **distribution matching**, not maximisation:

```
Given register r, the model's slang-density distribution should match
the slang-density distribution of real human text at register r.
```

Not "maximise slang." Not "minimise slang." **Match the human histogram.** Measured as Wasserstein distance (§14.5).

This automatically produces the right behaviour: at `slang_4`, real human text is dense but still varied and grammatical, so matching it forbids both blandness and word-salad.

## 15.5 The forced-slang detector

Train a small classifier (fine-tune the 125M) on:
- **positive (natural):** real mined Gen-Z text
- **negative (forced):** LLM-generated slang + the §1.5 synthetic translation datasets — which are a perfect, free, labelled cringe corpus

Use it as (a) a BUSSBENCH task-15 scorer, (b) an automatic DPO labeller, (c) a corpus filter.

> The dated synthetic datasets that are useless for training are **excellent as a labelled negative class.** That is the one good use for them.

## 15.6 Anti-goals — what NOT to optimise

| Do not optimise | Because |
|---|---|
| Slang words per response | Maximising it *is* cringe |
| "Gen-Z-ness" as a single scalar | Collapses 5 independent register dimensions |
| Human preference for "most Gen-Z" output | Raters pick the most *legible* slang, which is the most dated |
| Matching the slang lexicon exactly | Encourages dictionary recitation over usage |
| Never using slang | The opposite failure; a bland model is also a failed model |

## 15.7 Context adaptation — the acid test

The single most important behaviour, tested by the **mirror test**:

| User writes | Correct model register |
|---|---|
| "Could you please explain how photosynthesis works?" | `slang_0 formal_3` — zero slang |
| "yo can u explain photosynthesis rq" | `slang_2 formal_1` — casual, light slang |
| "bruh wtf is photosynthesis 💀" | `slang_3 formal_0` — matched, emoji present |
| "Explain photosynthesis. Be casual." | `slang_2 formal_1` — instruction overrides style |

**Metric:** run 200 prompt pairs that differ *only* in register; measure the correlation between input register and output register. **Target Pearson r >= 0.75.** A model at r ≈ 0 is either always-formal or always-slangy, and both are failures.

The ~15% of SFT examples with no register block (§12.1) exist specifically to train this.

---

# §16. Project roadmap

Compute is given in units of the ~1.15e19 FLOPs/week available from one Kaggle account (30 GPU-h + 20 TPU-h).

## Phase 0 — Research and setup (1 week)

| | |
|---|---|
| **Objective** | Lock the spec; verify every assumption in your own accounts |
| **Deliverables** | This document; Kaggle/Colab/Lightning/HF accounts with verified quotas; repo scaffold; `bussin` package skeleton |
| **Compute** | ~0 |
| **Risks** | ~~TPU quota differs from [E]~~ **resolved: confirmed 20 h/week on the account**; TPU v3-8 unavailable in your region at session time |
| **Success test** | A trivial notebook runs on Kaggle T4 x2 **and** TPU v3-8 and prints device info; quota page screenshotted |

## Phase 1 — Data (3-4 weeks, CPU only)

| | |
|---|---|
| **Objective** | The 70 GB tokenized corpus, plus the dated lexicon |
| **Deliverables** | `lexicon.jsonl` (~90K dated terms); register annotator + its human-correlation validation; cleaning pipeline; **starter corpus (1B tokens)** in week 1; full corpus by week 4; Kaggle Dataset mirrors |
| **Compute** | **0 GPU-hours.** ~200 CPU-hours across batch sessions |
| **Risks** | SN13 dedup removes more than expected (mitigate: pull more miner repos); annotator correlates poorly with humans; 4 chained notebook outputs is fiddly |
| **Success tests** | (a) annotator vs human Spearman **rho >= 0.7** on 2,000 docs; (b) 1,000-doc manual PII read finds **zero** identifiers; (c) slang-age histogram shows a real 2024-2026 tail; (d) shard manifest hashes verify |

**Gate: do not proceed to Phase 3 until (a) and (b) pass.**

## Phase 2 — Tokenizer (3 days, CPU only)

| | |
|---|---|
| **Objective** | One 49,152-vocab byte-level BPE for the whole ladder |
| **Deliverables** | `tokenizer.json`; the §6.8 evaluation report |
| **Compute** | 0 GPU-hours; ~6 CPU-hours |
| **Risks** | Round-trip failures on ZWJ emoji; forced tokens crowd out learned merges |
| **Success test** | All §6.8 thresholds met, especially **100% round-trip on 1M docs** and **>= 3.2 bytes/token on Gen-Z core** |

## Phase 3 — `bussin-125m` prototype (2 weeks)

| | |
|---|---|
| **Objective** | **Prove the relay, not the model** |
| **Deliverables** | Working trainer on GPU **and** TPU; relay with lease + watchdog; `bussin-125m` at 10B tokens; measured MFU on real hardware |
| **Compute** | **58-87 GPU-h or 15-23 TPU-h** -> ~1 week of quota |
| **Risks** | fp16 divergence (Z-loss + `GradScaler` tuning); PyTorch/XLA checkpoint incompatibility with the CUDA path; lease race conditions |
| **Success tests** | **(a) The resume test**: kill a session mid-run; a new session on a *different platform* resumes and the loss curve is continuous within 1%. **(b)** GPU-trained checkpoint loads and continues on TPU and vice versa. **(c)** `train_seconds/session_seconds >= 0.95`. **(d)** Replace the [E] MFU numbers in §9.6 with measurements |

**This is the most important phase in the project.** Everything after it is the same code with a bigger config.

## Phase 4 — `bussin-400m` (7-9 weeks)

| | |
|---|---|
| **Objective** | **The flagship model** |
| **Deliverables** | `bussin-400m-base` at 30B tokens; stable-phase checkpoint retained for re-annealing |
| **Compute** | **551-826 GPU-h equivalent** -> ~6.2 weeks of combined quota, plus slack |
| **Risks** | Anneal degrades general English (>5% gate); Gen-Z core exhausted earlier than planned; a platform changes its free tier mid-run |
| **Success tests** | `test-general` PPL degradation **< 5%**; `test-genz` PPL beats a same-size model trained on general English only by **> 15%**; RCE < 0.5 on the base model |

## Phase 5 — `bussin-1b` (16-20 weeks) — **conditional**

| | |
|---|---|
| **Objective** | The largest properly-trained model free compute allows |
| **Deliverables** | `bussin-1b-base` at 25B tokens (19.9 tok/param) |
| **Compute** | **374-581 TPU-h**, TPU-primary; ~16 weeks |
| **Preconditions** | TPU path proven in Phase 3; FSDP working on T4 x2 as fallback; 400M results justify the spend |
| **Risks** | TPU quota reduced; 15 GB checkpoints make session turnaround expensive; **4 months is a long time for a free tier to stay unchanged** |
| **Success tests** | Beats `bussin-400m` on BUSSBENCH tasks 8, 11, 13 (knowledge/comprehension). **If it does not, ship the 400M and stop.** |

> **[R] Kill criterion.** If at 12B tokens (~half way) the 1B is not clearly ahead of the finished 400M on tasks 11 and 13, stop. A 1B at 9.6 tok/param is worth less than the compute it would take to finish, and that compute is better spent over-training the 400M further.

## Phase 6 — Instruction tuning (2-3 weeks)

| | |
|---|---|
| **Objective** | `-instruct` variants |
| **Deliverables** | 120K SFT set; 20K DPO set; `bussin-400m-instruct` (+ `1b` if Phase 5 shipped) |
| **Compute** | ~25-40 GPU-h per model (SFT is ~2% of pretraining cost) |
| **Risks** | SFT flattens register control; DPO degrades fluency; mined pairs have meaning drift |
| **Success tests** | RCE **< 0.4**; context-adaptation Pearson **r >= 0.75**; formal-mode slang leakage **< 0.2/100 tok**; no HellaSwag regression > 1 point |

## Phase 7 — Evaluation (2 weeks, overlapping)

| | |
|---|---|
| **Objective** | BUSSBENCH v1 + the human study |
| **Deliverables** | 3,500-item benchmark; leaderboard vs SmolLM2-360M, Qwen3-0.6B, a slang-prompted large model, and real human text; full results |
| **Compute** | ~10 GPU-h |
| **Risks** | Contamination found late; too few genuine Gen-Z raters |
| **Success tests** | Zero 13-gram contamination; **>= 5 raters aged 16-26**; human-vs-model discrimination **< 80%** |

## Phase 8 — Deployment (1 week)

| | |
|---|---|
| **Objective** | Ship it |
| **Deliverables** | HF model repos (fp16 safetensors + GGUF Q8_0/Q4_K_M); model cards with exact counts, data composition, limitations, AAVE provenance; a Space demo with **visible register sliders**; the code and filter manifests |
| **Compute** | ~2 GPU-h |
| **Risks** | Publishing Amber-tier corpus text by accident |
| **Success tests** | `from_pretrained` works from a clean environment; GGUF runs in Ollama; **no Amber source text in any public repo** |

**Total: ~7 months to `bussin-400m-instruct` + BUSSBENCH; ~11 months including `bussin-1b`.**

---

# §17. Final recommendation

## 17.1 Build this

> ## **`bussin-400m` is the model to build.**
>
> **396,418,176 parameters. 30B tokens. 75.7 tokens/parameter. ~6.2 weeks of one Kaggle account's combined GPU+TPU quota.**

Reasoning:

1. **It is properly trained.** 75.7 tok/param is deep in the over-trained regime where small models become genuinely good — nearly 4x Chinchilla-optimal. Nobody has to make excuses for it.
2. **It fits the hardware without heroics.** 9.2 GB peak on a single 16 GB T4 [C]. No FSDP, no gradient checkpointing, no sharding bugs. The 1B needs all three on GPU.
3. **It saturates the available Gen-Z data.** Its 2.30B Gen-Z-core tokens is ~0.7 epochs of the world's entire contemporary supply (§4.0). **A larger model cannot be more Gen-Z, only more generally capable** — because the Gen-Z data does not exist. This is the strongest single argument for 400M and it is a data argument, not a compute argument.
4. **It leaves room to iterate.** At 6 weeks per full run you can afford to re-anneal, fix the mixture, and re-run. At 4 months per run you get one attempt.
5. **The interesting claims are all testable at 400M.** Register conditioning, the dated lexicon, RCE, the anti-cringe results — none of them need a billion parameters.

## 17.2 On the 1B

**Do it second, on TPU, with a kill criterion.**

- **1B on GPU quota alone: do not.** Six months yields 8.9 tok/param [C] — under half of Chinchilla — and it would lose to the 400M. That is the "few billion tokens and call it trained" outcome you asked me to rule out.
- **1B on TPU: yes, it is real.** 374-581 TPU-hours, ~16 weeks, 19.9 tok/param — Chinchilla-optimal [C].
- **Expect it to be better at knowledge and worse at polish** than the 400M, since it gets 25B tokens against the 400M's 30B. It will not be a better *Gen-Z* model; it will be a more knowledgeable one.
- **Kill it at the halfway gate** if it is not clearly ahead (Phase 5).

## 17.3 Do this first, this week

1. Verify your TPU quota on the Kaggle settings page. **The whole 1B plan depends on this one number** and it is the only [E] in the critical path.
2. Build the **1B-token starter corpus** (2 GB) — small enough to fit in `/kaggle/working`, big enough to train on.
3. Train `bussin-125m` on it for ~3 hours.
4. **Run the resume test**: kill the session, resume on a different platform, confirm the loss curve is continuous.

Steps 1-4 cost about 3 GPU-hours and de-risk the entire project. Do not generate 70 GB before the resume test passes.

## 17.4 What success looks like

Not "an LLM that talks Gen-Z." That exists as a prompt.

> **`bussin-400m-instruct`: a 396M-parameter model, trained from scratch on free compute, that understands contemporary internet English better than models 20x its size, generates it at a controllable and correctly-calibrated register, knows which slang is dead, and — uniquely — knows when not to use slang at all.**

Plus three artefacts with independent value:
- **BUSSBENCH** — the first purpose-built benchmark for this register.
- **The dated slang lexicon** — ~90K terms with first-attestation dates and lifecycle status.
- **The relay** — a reusable system for training one coherent model across heterogeneous free compute.

## 17.5 Project structure

```
bussin/
  README.md
  docs/
    SPEC.md                    <- this document
    DATA_CARD.md  MODEL_CARD.md  RELAY.md
  configs/
    125m.yaml  300m.yaml  400m.yaml  1b.yaml
    tokenizer.yaml  data_mixture.yaml
  bussin/
    model/       config.py  bussin_model.py  rope.py  attention.py
    tokenizer/   train_tokenizer.py  forced_vocab.py  evaluate.py
    data/        mine.py  clean.py  pii.py  dedup.py  lexicon.py
                 register.py  tokenize_shard.py  pack.py  loader.py
    train/       trainer.py  schedule.py  optim.py  precision.py
                 fsdp.py  xla.py  validate.py
    relay/       state.py  lease.py  checkpoint.py  watchdog.py
                 platform.py  bootstrap.py
    eval/        bussbench.py  register_metrics.py  cringe.py  harness.py
  pipelines/
    00_build_lexicon.py     01_mine.py        02_clean.py
    03_annotate.py          04_split.py       05_tokenize.py
    06_shard.py             07_publish.py
  notebooks/
    kaggle_cpu_etl.ipynb    kaggle_gpu_train.ipynb
    kaggle_tpu_train.ipynb  colab_train.ipynb  lightning_train.ipynb
  eval/BUSSBENCH/           tasks/  manifest.json
  scripts/
    smoke_test.sh  resume_test.sh  push_kernel.sh
  tests/
```

---

# Sources

## Official documentation
- Kaggle Notebooks documentation (Technical Specifications, session limits, storage, TPU/GPU specs) — https://www.kaggle.com/docs/notebooks
- Kaggle Efficient GPU Usage (weekly quota language, efficiency guidance) — https://www.kaggle.com/docs/efficient-gpu-usage
- Kaggle Terms of Use, version June 22, 2025 (multi-account clause) — https://www.kaggle.com/terms
- Kaggle Acceptable Use Policy — https://www.kaggle.com/aup
- Hugging Face Hub storage limits — https://huggingface.co/docs/hub/storage-limits
- NVIDIA T4 Tensor Core datasheet (65 TFLOPS FP16) — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
- NVIDIA Ampere architecture (bf16 support begins at Ampere) — https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/

## Papers
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 20 tok/param), NeurIPS 2022 — https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf
- Nguyen, Vu & Nguyen, *BERTweet: A pre-trained language model for English Tweets*, EMNLP 2020 Demos — https://aclanthology.org/2020.emnlp-demos.2/
- Loureiro et al., *TimeLMs: Diachronic Language Models from Twitter* — https://arxiv.org/abs/2202.03829
- Allal et al., *SmolLM2* — https://arxiv.org/abs/2502.02737
- Muennighoff et al., *Scaling Data-Constrained Language Models*, NeurIPS 2023
- *Raiders of the Lost Kek* (4chan /pol/ dataset) — https://arxiv.org/abs/2001.07487
- Ringer, Nicolaou & Walker, TwitchChat dataset, AIIDE 2020 — doi:10.1609/aiide.v16i1.7439
- GECLM Reddit source paper — https://arxiv.org/abs/2001.08435

## Datasets (row counts verified via HF datasets-server)
- https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments
- https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments
- https://huggingface.co/datasets/tensorshield/reddit_dataset_157
- https://huggingface.co/datasets/lparkourer10/twitch_chat
- https://huggingface.co/datasets/llmtraining-scraper/discord-messages
- https://huggingface.co/datasets/georgiyozhegov/urbandictionary
- https://huggingface.co/datasets/MLBtrio/genz-slang-dataset
- https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment
- https://huggingface.co/datasets/findnitai/english-to-hinglish
- https://huggingface.co/datasets/Abhishekcr448/Hinglish-Everyday-Conversations-1M
- https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- https://huggingface.co/datasets/allenai/WildChat-1M

## Models
- https://huggingface.co/vinai/bertweet-base
- https://huggingface.co/cardiffnlp/twitter-roberta-base-2022-154m
- https://huggingface.co/Sankar-2910/genz-translator
- https://huggingface.co/Abhishekcr448/Tiny-Hinglish-Chat-21M
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B

---

*End of specification. Every [E] and [A] marker is a thing to verify before relying on it.*
