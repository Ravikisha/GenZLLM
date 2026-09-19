"""Dashboard data.

Assembles everything worth watching into one JSON blob, published to the
checkpoint repo on every orchestrator tick. A Hugging Face Space renders it
(`spaces/app.py`); nothing needs a server.

One honest note that shapes the panels: **pretraining has no accuracy.** It has
loss and perplexity, continuously. Accuracy exists only once BUSSBENCH runs,
and that costs GPU time, so it appears at milestones rather than every step.
The dashboard therefore separates:

  live      loss, perplexity, tokens/s, grad norm, scaler health
  periodic  BUSSBENCH task accuracy, Register Calibration Error
  derived   tokens/parameter against the Chinchilla and over-trained lines,
            and ETA at observed throughput vs. the plan

The derived panel is the one that tells you whether the run is on track, which
none of the raw numbers do on their own.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from .ledger import BUDGETS, iso, utcnow

CHINCHILLA_RATIO = 20.0      # Hoffmann et al. 2022
WELL_TRAINED_RATIO = 70.0    # the small-model over-trained regime (SPEC §10.4)


def _load_metrics(cfg: ProjectConfig, ckpt: str | None,
                  max_rows: int = 4000) -> list[dict]:
    """Pull metrics.jsonl out of the latest checkpoint."""
    if not ckpt:
        return []
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            cfg.ckpt_repo, f"{ckpt}/metrics.jsonl", repo_type="model",
            token=cfg.creds.hf_token,
        )
    except Exception:
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in r:
            rows.append(r)
    if len(rows) > max_rows:  # decimate for the chart, keep the tail exact
        stride = len(rows) // max_rows + 1
        rows = rows[::stride] + rows[-200:]
    return rows


def _downsample(rows: list[dict], key: str, n: int = 600) -> list[list[float]]:
    pts = [(r["step"], r[key]) for r in rows if key in r and r.get(key) is not None]
    if len(pts) <= n:
        return [[s, round(float(v), 5)] for s, v in pts]
    stride = len(pts) // n + 1
    return [[s, round(float(v), 5)] for s, v in pts[::stride]]


def _plan_from_config(config_path: str) -> dict[str, Any]:
    try:
        import yaml

        d = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    model, train = d.get("model", {}), d.get("train", {})
    try:
        from ..model.config import BussinConfig

        n_params = BussinConfig.from_dict(model).n_params()["total"]
    except Exception:
        n_params = 0
    return {
        "name": model.get("name"),
        "n_params": n_params,
        "total_tokens": train.get("total_tokens"),
        "total_steps": train.get("total_steps"),
        "global_batch_tokens": train.get("global_batch_tokens"),
    }


ETL_STAGES = ("mine-0", "mine-1", "mine-2", "rank-fast", "tokenize")


def _data_pipeline(cfg: ProjectConfig) -> dict[str, Any]:
    """What the CPU side is doing.

    Before training starts this is the only thing happening, and none of the
    training panels can show it: the corpus is built by Kaggle CPU kernels that
    consume no GPU quota and therefore never touch the ledger. Every lookup is
    best-effort -- the dashboard must never be the reason a tick fails.
    """
    out: dict[str, Any] = {"etl": [], "corpus": None}

    try:
        cfg.creds.export_kaggle()
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        for stage in ETL_STAGES:
            slug = f"{cfg.creds.kaggle_username}/bussin-etl-{stage}"
            try:
                st = str(getattr(api.kernels_status(slug), "status", "?"))
                out["etl"].append({"stage": stage, "status": st.split(".")[-1]})
            except Exception:
                continue
    except Exception as exc:
        out["etl_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"

    try:
        from huggingface_hub import HfApi

        info = HfApi(token=cfg.creds.hf_token).repo_info(
            cfg.corpus_repo, repo_type="dataset", files_metadata=True
        )
        parts = [s for s in info.siblings if s.rfilename.endswith(".jsonl.gz")]
        out["corpus"] = {
            "repo": cfg.corpus_repo,
            "files": len(parts),
            "bytes": sum(s.size or 0 for s in parts),
            "updated": iso(info.last_modified) if info.last_modified else None,
        }
    except Exception as exc:
        out["corpus_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"

    return out


def build_snapshot(cfg: ProjectConfig, tick_out: dict[str, Any],
                   config_path: str = "configs/400m.yaml") -> dict[str, Any]:
    now = utcnow()
    run = tick_out.get("run", {}) or {}
    ledger_snap = tick_out.get("ledger", {}) or {}
    plan = _plan_from_config(config_path)
    metrics = _load_metrics(cfg, run.get("latest_ckpt"))

    tokens = run.get("tokens_seen") or 0
    n_params = plan.get("n_params") or 0
    total_tokens = plan.get("total_tokens") or 0

    # --- throughput, from what actually happened, not what was hoped ---
    recent = [r for r in metrics[-500:] if r.get("tok_per_s")]
    observed_tok_s = (
        sum(r["tok_per_s"] for r in recent) / len(recent) if recent else 0.0
    )

    # Weekly token capacity implied by remaining quota across all platforms.
    weekly_tokens = 0.0
    if n_params:
        for name, b in BUDGETS.items():
            p = (ledger_snap.get("platforms") or {}).get(name, {})
            if not p.get("automatable", True):
                continue
            flops = b.quota_hours * 3600 * b.effective_tflops * 1e12
            weekly_tokens += flops / (6 * n_params)

    remaining_tokens = max(total_tokens - tokens, 0)
    weeks_left = remaining_tokens / weekly_tokens if weekly_tokens else None

    progress = {
        "tokens_seen": tokens,
        "total_tokens": total_tokens,
        "pct_complete": round(100 * tokens / total_tokens, 2) if total_tokens else 0.0,
        "tokens_per_param": round(tokens / n_params, 2) if n_params else 0.0,
        "chinchilla_ratio": CHINCHILLA_RATIO,
        "well_trained_ratio": WELL_TRAINED_RATIO,
        "pct_of_chinchilla": (
            round(100 * (tokens / n_params) / CHINCHILLA_RATIO, 1) if n_params else 0.0
        ),
        "observed_tokens_per_s": round(observed_tok_s),
        "weekly_token_capacity": round(weekly_tokens),
        "eta_weeks": round(weeks_left, 1) if weeks_left else None,
        "eta_date": (
            iso(now + timedelta(weeks=weeks_left)) if weeks_left else None
        ),
    }

    losses = _downsample(metrics, "loss")
    last = metrics[-1] if metrics else {}
    live = {
        "step": run.get("step", 0),
        "total_steps": plan.get("total_steps"),
        "stage": run.get("stage"),
        "loss": last.get("loss"),
        "perplexity": (
            round(math.exp(min(last.get("ce_loss", last.get("loss", 20)), 20)), 2)
            if last else None
        ),
        "lr": last.get("lr"),
        "grad_norm": last.get("grad_norm"),
        "scaler_scale": last.get("scaler_scale"),
        "tok_per_s": last.get("tok_per_s"),
        "phase": last.get("phase"),
    }

    # Session efficiency: fraction of wall time actually spent training.
    # Target >= 0.95 (SPEC §9.4). A slipping value means quota is going to
    # setup, downloads or validation rather than matrix multiplies.
    history = tick_out.get("history") or []
    effs = [h["efficiency"] for h in history if h.get("efficiency")]

    return {
        "generated_at": iso(now),
        "model": plan,
        "status": {
            "action": tick_out.get("action"),
            "reason": tick_out.get("reason"),
            "lease": tick_out.get("lease"),
            "next_reset": tick_out.get("next_reset"),
            "finished": run.get("finished", False),
        },
        "live": live,
        "progress": progress,
        "quota": ledger_snap.get("platforms", {}),
        "running": ledger_snap.get("running"),
        "recent_dispatches": ledger_snap.get("recent", [])[-15:],
        "charts": {
            "loss": losses,
            "lr": _downsample(metrics, "lr"),
            "grad_norm": _downsample(metrics, "grad_norm"),
            "tok_per_s": _downsample(metrics, "tok_per_s"),
        },
        "sessions": {
            "count": len(history),
            "mean_efficiency": round(sum(effs) / len(effs), 3) if effs else None,
            "recent": history[-10:],
        },
        "data_pipeline": _data_pipeline(cfg),
        # Populated by the eval harness at milestones; absent during pretraining.
        "bussbench": tick_out.get("bussbench"),
        "notes": [
            "Pretraining reports loss and perplexity. Accuracy appears only "
            "when BUSSBENCH runs at milestones.",
            "Quota is an estimate: Kaggle does not expose it via API. "
            "Reconcile against the settings page occasionally.",
        ],
    }
