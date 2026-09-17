"""Generate the platform notebooks.

Kept as a generator rather than hand-edited .ipynb files because the four
notebooks differ only in a handful of lines, and hand-maintained notebooks drift
apart silently -- which for this project would mean the GPU and TPU workers
disagreeing about global batch size and quietly producing two different runs.

Run:  python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

REPO_URL = "https://github.com/CHANGEME/bussin.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(True),
    }


def notebook(cells: list[dict], accelerator: str = "none") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": accelerator,
            "kaggle": {"accelerator": accelerator, "dataSources": [],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


SETUP = f'''# --- Bussin worker setup ---------------------------------------------
# Pinned and quiet: every second here is a second of quota not spent on
# matrix multiplies. Target is >= 95% of session wall time in training steps.
import os, subprocess, sys, time
T0 = time.time()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"      # parallel checkpoint pulls
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.exists("bussin"):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}", "."], check=False)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "hf_transfer", "safetensors", "huggingface_hub", "pyyaml", "regex"],
               check=False)

sys.path.insert(0, os.getcwd())
print(f"setup took {{time.time() - T0:.1f}}s")'''

SECRETS = '''# --- Credentials ------------------------------------------------------
# Store the token in Kaggle "Add-ons -> Secrets" as HF_TOKEN. Never paste a
# token into a notebook cell: notebooks get shared, and the token grants write
# access to your checkpoint repo.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded from Kaggle secrets")
except Exception as exc:
    print(f"could not load Kaggle secret ({exc}); falling back to env HF_TOKEN")
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN is required"'''

PLATFORM = '''# --- What did we wake up on? -----------------------------------------
from bussin.relay.platform import get_platform_info, plan_batch

info = get_platform_info()
print(info)

# The bf16 question decides the whole precision path. Kaggle's T4 is Turing
# (sm75) and the P100 is Pascal (sm60); bf16 tensor cores start at Ampere, so
# on Kaggle GPUs this always prints False and training runs fp16 + GradScaler.
print(f"bf16 available: {info.supports_bf16}")
if info.device_type == "cuda" and "P100" in info.device_name:
    print("WARNING: P100 selected. It has no tensor cores and is roughly 6x "
          "slower than T4 x2 for the same quota hour. Switch to T4 x2.")'''

TRAIN = '''# --- Train ------------------------------------------------------------
# The worker claims the lease, restores the run, trains until the watchdog
# fires at (session limit - 20 min), checkpoints, releases the lease, exits.
#
# If another worker holds a live lease this exits in seconds without starting,
# so a scheduled notebook costs nothing when the run is already being carried.
from bussin.relay.bootstrap import run

exit_code = run(CONFIG)
print(f"worker exited {exit_code}")'''

VERIFY = '''# --- Verify before spending quota -------------------------------------
# A dry run does everything except train: claims and releases the lease, pulls
# the checkpoint, rebuilds the model, checks the batch plan reproduces the
# global batch exactly, and probes loss continuity. Run it first.
from bussin.relay.bootstrap import run
run(CONFIG, dry_run=True)'''


def build_gpu() -> dict:
    return notebook(
        [
            md(
                "# Bussin -- Kaggle GPU worker\n\n"
                "**Before running:**\n\n"
                "1. Settings -> Accelerator -> **GPU T4 x2** (not P100: no tensor "
                "cores, ~6x slower per quota hour)\n"
                "2. Settings -> Internet -> **On**\n"
                "3. Add-ons -> Secrets -> add `HF_TOKEN`\n"
                "4. Add Data -> your `bussin-corpus` dataset (mounts instantly at "
                "`/kaggle/input`, costs no quota; downloading the corpus instead "
                "would burn ~2h/week)\n"
                "5. Save Version -> **Save & Run All**. Do not use the interactive "
                "editor: idle sessions silently eat quota.\n"
            ),
            code(SETUP),
            code(SECRETS),
            code('CONFIG = "configs/400m.yaml"   # or 125m.yaml / 1b.yaml'),
            code(PLATFORM),
            code(VERIFY),
            code(TRAIN),
        ],
        accelerator="nvidiaTeslaT4",
    )


def build_tpu() -> dict:
    tpu_setup = SETUP + '''

# torch_xla is preinstalled on Kaggle TPU VMs. Importing it here so a missing
# runtime fails now rather than 40 minutes in.
import torch_xla.core.xla_model as xm
print("XLA device:", xm.xla_device())
print("XLA world size:", xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else 8)'''
    return notebook(
        [
            md(
                "# Bussin -- Kaggle TPU worker\n\n"
                "TPU v3-8 runs on a **quota separate from the 30 GPU hours** "
                "(~20 h/week, 9 h sessions) and is roughly 4x the effective "
                "throughput of T4 x2. It is what makes `bussin-1b` reachable at "
                "Chinchilla-optimal token counts -- verify your own quota on the "
                "Kaggle settings page before planning around it.\n\n"
                "**Before running:**\n\n"
                "1. Settings -> Accelerator -> **TPU VM v3-8**\n"
                "2. Settings -> Internet -> **On**\n"
                "3. Add-ons -> Secrets -> `HF_TOKEN`\n"
                "4. Add Data -> `bussin-corpus`\n"
                "5. **Save & Run All**\n\n"
                "TPU sessions are 9 h, not 12 h. The watchdog reads that from "
                "`SESSION_LIMITS` automatically.\n"
            ),
            code(tpu_setup),
            code(SECRETS),
            code('CONFIG = "configs/400m.yaml"'),
            code(PLATFORM),
            code(VERIFY),
            code(TRAIN),
        ],
        accelerator="TPU VM v3-8",
    )


def build_cpu_etl() -> dict:
    etl = '''# --- Data pipeline (CPU only) -----------------------------------------
# CPU sessions do not consume the GPU quota, and Kaggle allows 5 concurrent
# batch CPU sessions of 12 h each. That is ~60 CPU-hours per wave, free, and it
# is why no data work should ever happen inside a GPU session.
import subprocess, sys

STAGE = "lexicon"       # lexicon | discover | mine | tokenize | shard
SHARD_INDEX = 0          # 0..4, so five sessions split the work
N_SHARDS = 5

cmds = {
    "lexicon":  [sys.executable, "pipelines/00_build_lexicon.py",
                 "--out", "/kaggle/working/lexicon/lexicon.jsonl"],
    "discover": [sys.executable, "pipelines/01_discover_emerging.py",
                 "--lexicon", "/kaggle/input/bussin-lexicon/lexicon.jsonl",
                 "--out", "/kaggle/working/lexicon/lexicon.jsonl",
                 "--recent-rows", "0"],
    "mine":     [sys.executable, "pipelines/02_mine_corpus.py",
                 "--shard-index", str(SHARD_INDEX), "--n-shards", str(N_SHARDS),
                 "--out", "/kaggle/working/corpus"],
}
subprocess.run(cmds[STAGE], check=True)'''

    publish = '''# --- Publish -----------------------------------------------------------
# Outputs go to /kaggle/working (20 GB, auto-saved) and then become a Kaggle
# Dataset, which future GPU sessions mount instantly instead of downloading.
#
# One notebook output caps at 20 GB, so the 70 GB corpus is built by four
# sessions writing ~18 GB each and attached as four inputs.
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    repo_id="CHANGEME/bussin-corpus",
    repo_type="dataset",
    folder_path="/kaggle/working/corpus",
    path_in_repo=f"shards/part-{SHARD_INDEX:02d}",
)
print("pushed to the HF mirror; now Save Version to create the Kaggle Dataset")'''

    return notebook(
        [
            md(
                "# Bussin -- CPU ETL worker\n\n"
                "All data work happens here, **not** in a GPU session.\n\n"
                "The GPU quota is a *GPU* quota: CPU-only sessions do not touch "
                "it, and Kaggle allows 5 concurrent batch CPU sessions of 12 h "
                "each. That is roughly 60 free CPU-hours per wave for mining, "
                "cleaning, deduplication, tokenization and sharding.\n\n"
                "Set `STAGE` and `SHARD_INDEX`, then Save & Run All. Launch five "
                "copies with `SHARD_INDEX` 0..4 to fan out.\n\n"
                "**Accelerator must be None.**\n"
            ),
            code(SETUP),
            code(SECRETS),
            code(etl),
            code(publish),
        ],
        accelerator="none",
    )


def build_colab() -> dict:
    colab_setup = f'''# --- Bussin worker on Colab free --------------------------------------
import os, subprocess, sys, time
T0 = time.time()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

if not os.path.exists("bussin"):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}", "/content/bussin"],
                   check=False)
os.chdir("/content/bussin")
sys.path.insert(0, os.getcwd())

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "hf_transfer", "safetensors", "pyyaml", "regex"], check=False)

from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
print(f"setup took {{time.time() - T0:.1f}}s")'''

    colab_data = '''# --- Corpus ------------------------------------------------------------
# Colab cannot mount a Kaggle Dataset, so it pulls the shards it needs from the
# HF mirror. Only fetch what this session will actually consume: the free tier
# is pre-emptible and a 60 GB download would never finish.
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="CHANGEME/bussin-corpus", repo_type="dataset",
    allow_patterns=["manifest.json", "shards/part-00/*"],
    local_dir="/content/corpus",
)'''

    return notebook(
        [
            md(
                "# Bussin -- Colab free worker\n\n"
                "Colab free is pre-emptible and gives roughly 4 h sessions, so "
                "the watchdog uses a shorter budget here. It contributes real "
                "hours but should not be the primary carrier of a run.\n\n"
                "Add `HF_TOKEN` via the key icon in the left sidebar.\n"
            ),
            code(colab_setup),
            code('CONFIG = "configs/400m.yaml"'),
            code(colab_data),
            code(PLATFORM),
            code(TRAIN),
        ],
        accelerator="GPU",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    built = {
        "kaggle_gpu_train.ipynb": build_gpu(),
        "kaggle_tpu_train.ipynb": build_tpu(),
        "kaggle_cpu_etl.ipynb": build_cpu_etl(),
        "colab_train.ipynb": build_colab(),
    }
    for name, nb in built.items():
        path = OUT / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"  wrote {path.relative_to(ROOT)}  ({len(nb['cells'])} cells)")
    print(f"\n{len(built)} notebooks generated. Set REPO_URL and the CHANGEME "
          f"repo ids before use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
