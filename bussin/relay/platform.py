"""Platform and device detection.

One training run has to survive being handed between Kaggle T4x2, Kaggle
TPU v3-8, Colab, Lightning and a local box. The single invariant that makes
that one run rather than several is: **global batch size in tokens must be
identical everywhere**. This module works out how to hit that number on
whatever hardware it woke up on, and refuses to start if it cannot.
"""

from __future__ import annotations

import os
import platform as _platform
import socket
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

# Conservative session budgets in seconds. Kaggle kills at the hard limit with
# no grace period, so these are deliberately below the advertised maxima.
SESSION_LIMITS: dict[str, int] = {
    "kaggle-gpu": 12 * 3600,     # verified: 12h CPU/GPU
    "kaggle-tpu": 9 * 3600,      # verified: 9h TPU
    "kaggle-cpu": 12 * 3600,
    "colab": 4 * 3600,           # free tier is pre-emptible; assume little
    "lightning": 4 * 3600,
    "local": 365 * 24 * 3600,
    "unknown": 3 * 3600,
}

# Time reserved at session end to checkpoint and upload. A 15 GB bussin-1b
# checkpoint needs all of this; measure your real upload rate in Phase 3.
CHECKPOINT_RESERVE: dict[str, int] = {
    "kaggle-gpu": 20 * 60,
    "kaggle-tpu": 20 * 60,
    "colab": 15 * 60,
    "lightning": 15 * 60,
    "local": 60,
    "unknown": 15 * 60,
}


@dataclass
class PlatformInfo:
    platform: str
    device_type: str          # cuda | xla | mps | cpu
    n_devices: int
    device_name: str
    supports_bf16: bool
    session_limit_s: int
    checkpoint_reserve_s: int
    worker_id: str
    host: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.platform} | {self.n_devices}x {self.device_name} "
            f"({self.device_type}) | bf16={self.supports_bf16} | "
            f"session<={self.session_limit_s // 3600}h | worker={self.worker_id}"
        )


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        if os.environ.get("TPU_NAME") or os.environ.get("XRT_TPU_CONFIG") \
           or Path("/usr/share/tpu").exists():
            return "kaggle-tpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "kaggle-gpu"
        except ImportError:
            pass
        return "kaggle-cpu"
    if os.environ.get("COLAB_GPU") is not None or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if any(k.startswith("LIGHTNING_") for k in os.environ):
        return "lightning"
    if os.environ.get("BUSSIN_PLATFORM"):
        return os.environ["BUSSIN_PLATFORM"]
    return "local"


def detect_device() -> tuple[str, int, str, bool]:
    """Returns (device_type, n_devices, device_name, supports_bf16)."""
    # TPU first: torch_xla is only importable where it is meant to be used.
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        import torch_xla.runtime as xr

        n = xr.world_size() if hasattr(xr, "world_size") else 8
        return "xla", n, "TPU v3-8", True  # TPU v3 has native bf16
    except ImportError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            # bf16 needs compute capability >= 8.0 (Ampere). T4 is 7.5,
            # P100 is 6.0 -- both fall back to fp16 + GradScaler.
            major, _ = torch.cuda.get_device_capability(0)
            return "cuda", n, name, major >= 8
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", 1, "Apple MPS", False
    except ImportError:
        pass

    return "cpu", 1, _platform.processor() or "cpu", False


def get_platform_info() -> PlatformInfo:
    plat = detect_platform()
    dev_type, n_dev, dev_name, bf16 = detect_device()
    return PlatformInfo(
        platform=plat,
        device_type=dev_type,
        n_devices=n_dev,
        device_name=dev_name,
        supports_bf16=bf16,
        session_limit_s=SESSION_LIMITS.get(plat, SESSION_LIMITS["unknown"]),
        checkpoint_reserve_s=CHECKPOINT_RESERVE.get(plat, CHECKPOINT_RESERVE["unknown"]),
        # The dispatcher records a worker_id in the quota ledger before the
        # session starts. If the worker then invents its own, the ledger can
        # never match the session that finished and the record accrues
        # forever -- which saturated the quota and stopped all dispatching.
        worker_id=os.environ.get("BUSSIN_WORKER_ID")
        or f"{plat}-{uuid.uuid4().hex[:6]}",
        host=socket.gethostname(),
    )


# ------------------------------------------------------------------ #
# The global-batch invariant
# ------------------------------------------------------------------ #


@dataclass
class BatchPlan:
    global_batch_tokens: int
    global_batch_seqs: int
    seq_len: int
    micro_batch: int
    grad_accum: int
    n_devices: int

    def verify(self) -> None:
        realised = self.micro_batch * self.grad_accum * self.n_devices
        if realised != self.global_batch_seqs:
            raise ValueError(
                f"batch plan does not reproduce the global batch: "
                f"micro_batch({self.micro_batch}) * accum({self.grad_accum}) * "
                f"devices({self.n_devices}) = {realised}, expected "
                f"{self.global_batch_seqs}. Training would silently diverge "
                f"from the run's schedule -- refusing to start."
            )

    def __str__(self) -> str:
        return (
            f"global {self.global_batch_tokens:,} tok = {self.global_batch_seqs} seq "
            f"x {self.seq_len} | micro {self.micro_batch} x accum {self.grad_accum} "
            f"x {self.n_devices} dev"
        )


def plan_batch(
    global_batch_tokens: int,
    seq_len: int,
    n_devices: int,
    micro_batch: int | None = None,
    device_type: str = "cuda",
    device_name: str = "",
) -> BatchPlan:
    """Work out micro-batch and accumulation to hit the exact global batch.

    `micro_batch` may be pinned by config; otherwise a conservative default is
    chosen per device class and then reduced until the accumulation divides
    evenly.
    """
    if global_batch_tokens % seq_len != 0:
        raise ValueError(
            f"global_batch_tokens ({global_batch_tokens}) must be divisible by "
            f"seq_len ({seq_len})"
        )
    global_seqs = global_batch_tokens // seq_len

    if micro_batch is None:
        micro_batch = _default_micro_batch(device_type, device_name, seq_len)

    # Shrink until (micro_batch * n_devices) divides the global sequence count.
    while micro_batch > 1 and global_seqs % (micro_batch * n_devices) != 0:
        micro_batch -= 1

    per_step = micro_batch * n_devices
    if global_seqs % per_step != 0:
        raise ValueError(
            f"cannot hit global batch {global_seqs} seqs with {n_devices} devices; "
            "adjust global_batch_tokens or device count"
        )

    plan = BatchPlan(
        global_batch_tokens=global_batch_tokens,
        global_batch_seqs=global_seqs,
        seq_len=seq_len,
        micro_batch=micro_batch,
        grad_accum=global_seqs // per_step,
        n_devices=n_devices,
    )
    plan.verify()
    return plan


def _default_micro_batch(device_type: str, device_name: str, seq_len: int) -> int:
    name = (device_name or "").lower()
    if device_type == "xla":
        base = 8
    elif device_type == "cuda":
        if "t4" in name or "p100" in name:
            base = 4
        elif "l4" in name or "a10" in name:
            base = 8
        elif "a100" in name or "h100" in name:
            base = 16
        else:
            base = 4
    else:
        base = 1
    # Halve for every doubling of sequence length past 2048.
    while seq_len > 2048 and base > 1:
        base //= 2
        seq_len //= 2
    return max(base, 1)
