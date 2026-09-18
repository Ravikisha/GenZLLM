"""The training loop.

Everything unusual here exists because of one constraint: this run is carried
by a sequence of short sessions on hardware that disagrees about numerics.
A Kaggle T4 has no bfloat16 (Turing predates Ampere), so it trains fp32 master
weights under fp16 autocast with a GradScaler. A Kaggle TPU v3-8 has native
bf16 and needs no scaler at all. Both must produce checkpoints the other can
pick up mid-run without a discontinuity in the loss curve.

The rules that follow from that:
  * master weights and checkpoints are always fp32
  * compute dtype is decided at startup from the hardware, never from config
  * global batch size in tokens is fixed by the run, and micro-batch/accum are
    derived per platform to reproduce it exactly (see relay/platform.py)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from ..model.bussin_model import BussinForCausalLM
from ..model.config import BussinConfig
from ..relay.checkpoint import CheckpointMeta, checkpoint_name, save_checkpoint
from ..relay.platform import BatchPlan, PlatformInfo
from ..relay.watchdog import DivergenceDetector, StopReason, Watchdog
from .schedule import Curriculum, WSDSchedule


@dataclass
class TrainConfig:
    run_id: str = "bussin-400m-v1"
    total_steps: int = 57_220
    global_batch_tokens: int = 524_288
    seq_len: int = 2048

    lr_max: float = 3.5e-4
    lr_min: float = 3.5e-5
    warmup_steps: int = 600
    stable_frac: float = 0.85
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0

    # cadence
    log_every: int = 10
    val_every: int = 500
    val_batches: int = 20
    full_val_every: int = 2_000
    checkpoint_every_s: int = 90 * 60
    milestone_every: int = 2_000
    heartbeat_every_s: int = 10 * 60

    gradient_checkpointing: bool = False
    compile_model: bool = False
    seed: int = 1337

    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Precision
# ------------------------------------------------------------------ #


class Precision:
    """Picks the compute dtype the hardware can actually do.

    bfloat16 requires compute capability >= 8.0. Kaggle's T4 is 7.5 and its
    P100 is 6.0, so on Kaggle GPUs this always resolves to fp16 + GradScaler.
    Getting this wrong is the most common cause of a from-scratch run diverging
    a few thousand steps in.
    """

    def __init__(self, info: PlatformInfo, force: str | None = None) -> None:
        self.info = info
        if force:
            self.mode = force
        elif info.device_type == "xla":
            self.mode = "bf16"
        elif info.device_type == "cuda":
            self.mode = "bf16" if info.supports_bf16 else "fp16"
        else:
            self.mode = "fp32"

        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}[self.mode]
        self.needs_scaler = self.mode == "fp16"

    def autocast(self):
        if self.mode == "fp32" or self.info.device_type not in ("cuda", "cpu"):
            from contextlib import nullcontext

            return nullcontext()
        device = "cuda" if self.info.device_type == "cuda" else "cpu"
        return torch.autocast(device_type=device, dtype=self.dtype)

    def make_scaler(self):
        if not self.needs_scaler:
            return None
        # init_scale 2**14, not the 2**16 default: a from-scratch model's first
        # steps overflow at the higher scale and waste ~50 steps recovering.
        try:
            return torch.amp.GradScaler("cuda", init_scale=2**14)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(init_scale=2**14)

    def __str__(self) -> str:
        return f"{self.mode} (scaler={'on' if self.needs_scaler else 'off'})"


# ------------------------------------------------------------------ #
# Trainer
# ------------------------------------------------------------------ #


class Trainer:
    def __init__(
        self,
        model: BussinForCausalLM,
        model_cfg: BussinConfig,
        train_cfg: TrainConfig,
        plan: BatchPlan,
        info: PlatformInfo,
        schedule: WSDSchedule,
        device: torch.device,
        precision: Precision | None = None,
        curriculum: Curriculum | None = None,
    ) -> None:
        self.model = model
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.plan = plan
        self.info = info
        self.schedule = schedule
        self.device = device
        self.precision = precision or Precision(info)
        self.curriculum = curriculum or Curriculum(train_cfg.total_steps)

        self.optimizer = torch.optim.AdamW(
            model.param_groups(train_cfg.weight_decay),
            lr=train_cfg.lr_max, betas=train_cfg.betas, eps=train_cfg.eps,
        )
        self.scaler = self.precision.make_scaler()
        self.detector = DivergenceDetector()

        self.step = 0
        self.tokens_seen = 0
        self.metrics: list[dict] = []
        self.train_seconds = 0.0
        self._last_ckpt_t = time.monotonic()
        self._last_hb_t = time.monotonic()

        if train_cfg.gradient_checkpointing:
            model.enable_gradient_checkpointing()

    # -------------------------------------------------------------- #

    def _xla_step(self) -> None:
        try:
            import torch_xla.core.xla_model as xm

            xm.optimizer_step(self.optimizer, barrier=True)
        except ImportError:
            self.optimizer.step()

    def train_micro_batch(self, batch: dict[str, torch.Tensor], accum: int) -> dict[str, float]:
        with self.precision.autocast():
            out = self.model(
                batch["input_ids"],
                labels=batch["labels"],
                document_ids=batch.get("document_ids"),
            )
            loss = out["loss"] / accum

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            "loss": float(out["loss"].detach()),
            "ce_loss": float(out.get("ce_loss", out["loss"]).detach()),
            "z_loss": float(out["z_loss"].detach()) if "z_loss" in out else 0.0,
        }

    def optimizer_step(self) -> tuple[float | None, float | None, bool]:
        """Returns (grad_norm, scaler_scale, stepped)."""
        scale = None
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            scale = float(self.scaler.get_scale())

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        )

        fatal, message, skip = self.detector.check(self.step, 0.0, grad_norm, scale)
        if fatal:
            raise DivergenceError(message)

        stepped = True
        if skip:
            self.optimizer.zero_grad(set_to_none=True)
            # `scaler.update()` is mandatory even when the step is skipped.
            # `unscale_()` marks the optimizer as unscaled for this iteration,
            # and only `update()` clears that, so skipping without it makes the
            # *next* step raise "unscale_() has already been called on this
            # optimizer since the last update()". Only reachable under fp16,
            # which is why it survived every fp32 CPU test and appeared on the
            # first real T4 run.
            if self.scaler is not None:
                self.scaler.update()
            stepped = False
        elif self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        elif self.info.device_type == "xla":
            self._xla_step()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if message and not fatal:
            self._log_event("grad_spike", message)
        return grad_norm, scale, stepped

    # -------------------------------------------------------------- #

    def train_step(self, loader: Iterator[dict[str, torch.Tensor]]) -> dict[str, float]:
        t0 = time.monotonic()
        self.model.train()
        lr = self.schedule.apply(self.optimizer)

        agg = {"loss": 0.0, "ce_loss": 0.0, "z_loss": 0.0}
        for _ in range(self.plan.grad_accum):
            batch = next(loader)
            m = self.train_micro_batch(batch, self.plan.grad_accum)
            for k in agg:
                agg[k] += m[k] / self.plan.grad_accum

        if not math.isfinite(agg["loss"]):
            raise DivergenceError(f"non-finite loss {agg['loss']} at step {self.step}")

        grad_norm, scale, stepped = self.optimizer_step()

        self.step += 1
        self.schedule.step_forward()
        if stepped:
            self.tokens_seen += self.plan.global_batch_tokens
        dt = time.monotonic() - t0
        self.train_seconds += dt

        row = {
            "step": self.step, "lr": lr, "phase": self.schedule.phase,
            "stage": self.curriculum.stage_at(self.step).name,
            "tokens": self.tokens_seen, "grad_norm": grad_norm,
            "scaler_scale": scale, "step_s": round(dt, 4),
            "tok_per_s": int(self.plan.global_batch_tokens / dt) if dt > 0 else 0,
            **{k: round(v, 5) for k, v in agg.items()},
        }
        self.metrics.append(row)
        return row

    @torch.no_grad()
    def validate(self, loader: Iterator[dict[str, torch.Tensor]], n_batches: int) -> dict[str, float]:
        """Always fp32. A loss curve that only looks healthy in fp16 is not healthy."""
        self.model.eval()
        total, n = 0.0, 0
        for _ in range(n_batches):
            try:
                batch = next(loader)
            except StopIteration:
                break
            out = self.model(batch["input_ids"], labels=batch["labels"],
                             document_ids=batch.get("document_ids"))
            total += float(out["ce_loss"])
            n += 1
        self.model.train()
        if n == 0:
            return {"val_loss": float("nan"), "val_ppl": float("nan")}
        mean = total / n
        return {"val_loss": mean, "val_ppl": math.exp(min(mean, 20))}

    # -------------------------------------------------------------- #

    def should_checkpoint(self, watchdog: Watchdog) -> tuple[bool, bool]:
        """Returns (save_now, is_milestone)."""
        milestone = self.step > 0 and self.step % self.cfg.milestone_every == 0
        periodic = (time.monotonic() - self._last_ckpt_t) >= self.cfg.checkpoint_every_s
        return (milestone or periodic), milestone

    def save(self, out_root: str | Path, data_cursor: dict, val_loss: float | None = None,
             is_milestone: bool = False) -> Path:
        name = checkpoint_name(self.step)
        meta = CheckpointMeta(
            step=self.step, tokens_seen=self.tokens_seen,
            stage=self.curriculum.stage_at(self.step).name,
            run_id=self.cfg.run_id, model_name=self.model_cfg.name,
            n_params=self.model_cfg.n_params()["total"],
            loss=self.metrics[-1]["loss"] if self.metrics else None,
            val_loss=val_loss, wall_seconds=self.train_seconds,
            written_by=self.info.worker_id, platform=self.info.platform,
            device_type=self.info.device_type,
        )
        path = save_checkpoint(
            Path(out_root) / name, self.model, self.optimizer, self.schedule,
            self.scaler, meta, data_cursor,
            {"model": self.model_cfg.to_dict(), "train": _cfg_to_dict(self.cfg)},
            self.metrics,
        )
        self._last_ckpt_t = time.monotonic()
        return path

    def _log_event(self, kind: str, detail: str) -> None:
        self.metrics.append({"step": self.step, "event": kind, "detail": detail})

    # -------------------------------------------------------------- #

    def format_log(self, row: dict) -> str:
        parts = [
            f"step {row['step']:>7}/{self.cfg.total_steps}",
            f"{row['stage']}/{row['phase']:<6}",
            f"loss {row['loss']:.4f}",
            f"ppl {math.exp(min(row['ce_loss'], 20)):>8.2f}",
            f"lr {row['lr']:.2e}",
            f"gn {row['grad_norm']:.2f}" if row.get("grad_norm") is not None else "",
            f"tok/s {row['tok_per_s']:,}",
            f"seen {row['tokens'] / 1e9:.3f}B",
        ]
        if row.get("scaler_scale"):
            parts.append(f"scale {row['scaler_scale']:.0f}")
        return " | ".join(p for p in parts if p)


class DivergenceError(RuntimeError):
    """Training diverged; the relay should roll back to the last milestone."""


def _cfg_to_dict(cfg: TrainConfig) -> dict:
    d = {k: v for k, v in cfg.__dict__.items()}
    d["betas"] = list(cfg.betas)
    return d


def build_optimizer_and_schedule(
    model: BussinForCausalLM, cfg: TrainConfig, lr_scale: float = 1.0
) -> tuple[torch.optim.Optimizer, WSDSchedule]:
    from .schedule import build_schedule

    opt = torch.optim.AdamW(
        model.param_groups(cfg.weight_decay), lr=cfg.lr_max,
        betas=cfg.betas, eps=cfg.eps,
    )
    sched = build_schedule(
        total_steps=cfg.total_steps, lr_max=cfg.lr_max, lr_min=cfg.lr_min,
        warmup_steps=cfg.warmup_steps, stable_frac=cfg.stable_frac, lr_scale=lr_scale,
    )
    return opt, sched
