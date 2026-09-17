"""Warmup-Stable-Decay learning-rate schedule.

Cosine needs the total step count decided before the first step. Kaggle's quota
is, in Kaggle's own words, "30 hours or sometimes higher depending on demand" --
so the total is genuinely unknown in advance. Guess low and you waste quota;
guess high and you stop at a bad learning rate.

WSD removes the guess: stay in the stable phase as long as the quota lasts,
then decay whenever you decide to finish. It also composes with the data
curriculum, because the decay phase *is* the Gen-Z anneal (SPEC §8.2) -- low
learning rate plus the most contemporary data means the final gradients shape
register without overwriting the general-English competence learned earlier.

And because the stable-phase checkpoint stays valid, a bad anneal costs one or
two weeks to redo rather than the whole run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class WSDConfig:
    lr_max: float
    lr_min: float
    warmup_steps: int
    stable_until: int          # step at which decay begins
    total_steps: int
    decay_shape: str = "one_minus_sqrt"   # one_minus_sqrt | linear | cosine

    def validate(self) -> None:
        if not 0 <= self.warmup_steps <= self.stable_until <= self.total_steps:
            raise ValueError(
                f"require 0 <= warmup({self.warmup_steps}) <= "
                f"stable_until({self.stable_until}) <= total({self.total_steps})"
            )
        if self.lr_min > self.lr_max:
            raise ValueError("lr_min must not exceed lr_max")


class WSDSchedule:
    """Stateful so it can be checkpointed and resumed exactly."""

    def __init__(self, cfg: WSDConfig, lr_scale: float = 1.0) -> None:
        cfg.validate()
        self.cfg = cfg
        self.lr_scale = lr_scale      # set <1 by the divergence recovery path
        self.step = 0

    # -------------------------------------------------------------- #

    def lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            frac = (step + 1) / max(c.warmup_steps, 1)
            lr = c.lr_max * frac
        elif step < c.stable_until:
            lr = c.lr_max
        else:
            span = max(c.total_steps - c.stable_until, 1)
            t = min((step - c.stable_until) / span, 1.0)
            if c.decay_shape == "linear":
                factor = 1.0 - t
            elif c.decay_shape == "cosine":
                factor = 0.5 * (1.0 + math.cos(math.pi * t))
            else:  # one_minus_sqrt: holds high LR longer, then drops fast.
                factor = 1.0 - math.sqrt(t)
            lr = c.lr_min + (c.lr_max - c.lr_min) * factor
        return lr * self.lr_scale

    def phase_at(self, step: int) -> str:
        c = self.cfg
        if step < c.warmup_steps:
            return "warmup"
        if step < c.stable_until:
            return "stable"
        return "decay"

    # -------------------------------------------------------------- #

    def get_lr(self) -> float:
        return self.lr_at(self.step)

    @property
    def phase(self) -> str:
        return self.phase_at(self.step)

    def apply(self, optimizer) -> float:
        lr = self.get_lr()
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step_forward(self) -> None:
        self.step += 1

    def extend(self, new_total_steps: int, new_stable_until: int | None = None) -> None:
        """Grow the run because more quota arrived.

        Legal only while still in warmup or stable; once decay has begun the
        curve is committed and extending it would produce a discontinuity.
        """
        if self.phase == "decay":
            raise RuntimeError(
                "cannot extend a run that has entered the decay phase; "
                "re-decay from the last stable checkpoint instead"
            )
        self.cfg.total_steps = new_total_steps
        self.cfg.stable_until = (
            new_stable_until if new_stable_until is not None
            else int(new_total_steps * 0.85)
        )
        self.cfg.validate()

    def begin_decay_now(self, decay_steps: int) -> None:
        """Start the anneal from the current step -- 'finish with what we have'."""
        self.cfg.stable_until = self.step
        self.cfg.total_steps = self.step + decay_steps
        self.cfg.validate()

    # -------------------------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "lr_scale": self.lr_scale, "cfg": asdict(self.cfg)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step = int(state["step"])
        self.lr_scale = float(state.get("lr_scale", 1.0))
        self.cfg = WSDConfig(**state["cfg"])
        self.cfg.validate()


def build_schedule(
    total_steps: int,
    lr_max: float,
    lr_min: float | None = None,
    warmup_steps: int | None = None,
    stable_frac: float = 0.85,
    warmup_frac: float = 0.01,
    decay_shape: str = "one_minus_sqrt",
    lr_scale: float = 1.0,
) -> WSDSchedule:
    lr_min = lr_min if lr_min is not None else lr_max * 0.1
    warmup = warmup_steps if warmup_steps is not None else max(int(total_steps * warmup_frac), 100)
    return WSDSchedule(
        WSDConfig(
            lr_max=lr_max,
            lr_min=lr_min,
            warmup_steps=warmup,
            stable_until=int(total_steps * stable_frac),
            total_steps=total_steps,
            decay_shape=decay_shape,
        ),
        lr_scale=lr_scale,
    )


# ------------------------------------------------------------------ #
# Data curriculum, keyed to the same step axis
# ------------------------------------------------------------------ #


@dataclass
class StageSpec:
    name: str
    until_frac: float               # fraction of total steps this stage ends at
    mixture: dict[str, float]       # pool -> share, must sum to ~1


DEFAULT_CURRICULUM = [
    StageSpec("S1", 0.667, {"general": 1.0}),
    StageSpec("S2", 0.967, {"general": 0.40, "internet": 0.35, "genz": 0.20,
                            "lexicon": 0.02, "hinglish": 0.03}),
    StageSpec("S3", 1.000, {"general": 0.20, "internet": 0.25, "genz": 0.50,
                            "lexicon": 0.02, "hinglish": 0.03}),
]


class Curriculum:
    """Maps step -> data mixture, aligned to the WSD phases (SPEC §11)."""

    def __init__(self, total_steps: int, stages: list[StageSpec] | None = None) -> None:
        self.total_steps = total_steps
        self.stages = stages or DEFAULT_CURRICULUM
        for s in self.stages:
            total = sum(s.mixture.values())
            if abs(total - 1.0) > 0.02:
                raise ValueError(f"stage {s.name} mixture sums to {total:.3f}, expected 1.0")

    def stage_at(self, step: int) -> StageSpec:
        frac = step / max(self.total_steps, 1)
        for s in self.stages:
            if frac < s.until_frac:
                return s
        return self.stages[-1]

    def mixture_at(self, step: int) -> dict[str, float]:
        return self.stage_at(step).mixture

    def boundaries(self) -> dict[str, int]:
        return {s.name: int(s.until_frac * self.total_steps) for s in self.stages}
