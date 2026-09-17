"""Session watchdog.

Kaggle terminates at the hard session limit with no grace period. A session
that has not checkpointed by then loses everything since its last save. So
every session must end on its own terms: the watchdog fires at
`limit - reserve`, forcing a final checkpoint and a clean exit.

It also handles SIGTERM (Colab pre-emption, Lightning shutdown), which arrives
with only seconds of warning.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class StopReason(str, Enum):
    NONE = "none"
    DEADLINE = "deadline"          # watchdog: approaching session limit
    SIGNAL = "signal"              # SIGTERM/SIGINT: pre-emption or Ctrl-C
    STEPS_DONE = "steps_done"      # reached target step count
    DIVERGED = "diverged"          # NaN / scaler collapse
    QUOTA = "quota"                # platform quota exhausted
    ERROR = "error"


@dataclass
class WatchdogStatus:
    should_stop: bool
    reason: StopReason
    elapsed_s: float
    remaining_s: float
    detail: str = ""


class Watchdog:
    def __init__(
        self,
        session_limit_s: int,
        checkpoint_reserve_s: int,
        total_steps: int | None = None,
        on_stop: Callable[[StopReason], None] | None = None,
    ) -> None:
        self.start = time.monotonic()
        self.session_limit_s = session_limit_s
        self.checkpoint_reserve_s = checkpoint_reserve_s
        self.deadline = self.start + session_limit_s - checkpoint_reserve_s
        self.total_steps = total_steps
        self.on_stop = on_stop

        self._signalled = False
        self._reason = StopReason.NONE
        self._detail = ""
        self._install_handlers()

    def _install_handlers(self) -> None:
        def handler(signum, _frame):
            self._signalled = True
            self._reason = StopReason.SIGNAL
            self._detail = f"received signal {signum}"
            if self.on_stop:
                self.on_stop(StopReason.SIGNAL)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not in main thread, or platform lacks it

    # -------------------------------------------------------------- #

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    @property
    def remaining(self) -> float:
        return max(self.deadline - time.monotonic(), 0.0)

    def trip(self, reason: StopReason, detail: str = "") -> None:
        """Externally force a stop (divergence, quota exhaustion)."""
        self._reason = reason
        self._detail = detail
        self._signalled = True

    def check(self, step: int | None = None) -> WatchdogStatus:
        now = time.monotonic()
        if self._signalled:
            return WatchdogStatus(True, self._reason, now - self.start,
                                  max(self.deadline - now, 0.0), self._detail)
        if now >= self.deadline:
            return WatchdogStatus(
                True, StopReason.DEADLINE, now - self.start, 0.0,
                f"within {self.checkpoint_reserve_s}s of the session limit",
            )
        if self.total_steps is not None and step is not None and step >= self.total_steps:
            return WatchdogStatus(True, StopReason.STEPS_DONE, now - self.start,
                                  self.deadline - now, f"reached {self.total_steps} steps")
        return WatchdogStatus(False, StopReason.NONE, now - self.start, self.deadline - now)

    def budget_report(self, train_seconds: float) -> dict[str, float]:
        """Efficiency: fraction of wall time actually spent on training steps.

        Target >= 0.95 (SPEC §9.4). Anything lower means the session is losing
        quota to setup, downloads or validation.
        """
        elapsed = max(self.elapsed, 1e-9)
        return {
            "wall_s": round(elapsed, 1),
            "train_s": round(train_seconds, 1),
            "efficiency": round(train_seconds / elapsed, 4),
            "remaining_s": round(self.remaining, 1),
        }


class DivergenceDetector:
    """Hard alarms from SPEC §8.9, checked every step."""

    def __init__(
        self,
        grad_norm_window: int = 100,
        grad_norm_factor: float = 10.0,
        min_scaler_scale: float = 2**6,
        max_skipped: int = 5,
    ) -> None:
        self.window: list[float] = []
        self.grad_norm_window = grad_norm_window
        self.grad_norm_factor = grad_norm_factor
        self.min_scaler_scale = min_scaler_scale
        self.max_skipped = max_skipped
        self.skipped = 0
        self.skipped_recent: list[int] = []

    def check(self, step: int, loss: float, grad_norm: float | None,
              scaler_scale: float | None) -> tuple[bool, str, bool]:
        """Returns (fatal, message, skip_this_step)."""
        import math

        if not math.isfinite(loss):
            return True, f"loss is {loss} at step {step}", True

        if scaler_scale is not None and scaler_scale < self.min_scaler_scale:
            return (
                True,
                f"GradScaler scale collapsed to {scaler_scale:g} at step {step}: "
                "fp16 underflow. Halt and inspect (SPEC §8.9).",
                True,
            )

        if grad_norm is not None and math.isfinite(grad_norm):
            if len(self.window) >= 20:
                median = sorted(self.window)[len(self.window) // 2]
                if median > 0 and grad_norm > self.grad_norm_factor * median:
                    self.skipped += 1
                    self.skipped_recent.append(step)
                    self.skipped_recent = [s for s in self.skipped_recent if step - s <= 100]
                    if len(self.skipped_recent) > self.max_skipped:
                        return (
                            True,
                            f"{len(self.skipped_recent)} gradient spikes in 100 steps "
                            f"(latest {grad_norm:.1f} vs median {median:.1f})",
                            True,
                        )
                    return False, f"grad spike {grad_norm:.1f} (median {median:.1f}); skipping", True
            self.window.append(grad_norm)
            if len(self.window) > self.grad_norm_window:
                self.window.pop(0)
        elif grad_norm is not None:
            return True, f"grad norm is {grad_norm} at step {step}", True

        return False, "", False
