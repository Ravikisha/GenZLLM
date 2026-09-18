---
title: Bussin Training Dashboard
emoji: 📊
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Bussin training dashboard

Live view of the `bussin-*` pretraining run: progress, loss, platform quota
across Kaggle/Colab/Lightning, session efficiency, and BUSSBENCH results.

Reads `orchestrator/dashboard.json` from the checkpoint repo, which the
orchestrator republishes on every tick. No server, no build step.

## Secrets

| Name | Value |
|---|---|
| `HF_TOKEN` | a token with read access to the checkpoint repo |
| `BUSSIN_CKPT_REPO` | e.g. `GodAsap/bussin-ckpt` |

## A note on "accuracy"

Pretraining reports **loss and perplexity**, continuously. Accuracy exists only
once BUSSBENCH runs, and that costs GPU time, so it appears at milestones
rather than every step. The *Training adequacy* panel is the one that says
whether the run is actually on track: tokens per parameter against the
Chinchilla line (20) and the over-trained regime small models need (70+).
