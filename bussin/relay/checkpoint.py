"""Checkpoint save/load, built for cross-backend portability.

A checkpoint written on a Kaggle T4 (fp16 autocast + GradScaler) must resume
byte-for-byte on a Kaggle TPU (bf16), and vice versa. That is only possible if
**everything on disk is fp32** and the compute dtype is re-derived at load time
from the hardware. Never checkpoint in the compute dtype.

Contents (SPEC §8.7): model, optimizer, scheduler, GradScaler, RNG state for
every generator in play, and the data cursor. Omitting any one of them makes a
resume that looks fine for a few hundred steps and then silently plateaus.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def writable_scratch(name: str = ".bussin_scratch") -> Path:
    """A directory we can actually write to.

    Workers run with their cwd inside a mounted dataset, which is a
    **read-only** filesystem, so anything relative fails with
    `OSError: [Errno 30] Read-only file system`.

    Prefer Kaggle's unsaved scratch (~60 GB) over `/kaggle/working` (20 GB and
    persisted as notebook output): checkpoints go straight to the Hub, so a
    second copy inside the output quota is waste.
    """
    import tempfile

    for base in ("/kaggle/temp", "/kaggle/tmp", "/kaggle/working", "/content",
                 tempfile.gettempdir()):
        p = Path(base)
        if not p.is_dir():
            continue
        try:
            target = p / name
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return target
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix="bussin_"))


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import torch_xla.core.xla_model as xm

        state["xla_seed"] = xm.get_rng_state()
    except Exception:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(_as_tuple(state["python"]))
    if "numpy" in state:
        np.random.set_state(_as_tuple(state["numpy"]))
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])
        except Exception:
            pass  # different device count between sessions: not fatal
    if "xla_seed" in state:
        try:
            import torch_xla.core.xla_model as xm

            xm.set_rng_state(state["xla_seed"])
        except Exception:
            pass


def _as_tuple(obj):
    """torch.save round-trips tuples as lists in some versions."""
    if isinstance(obj, list):
        return tuple(_as_tuple(o) for o in obj)
    return obj


@dataclass
class CheckpointMeta:
    step: int
    tokens_seen: int
    stage: str
    run_id: str
    model_name: str
    n_params: int
    loss: float | None = None
    val_loss: float | None = None
    wall_seconds: float = 0.0
    written_by: str = ""
    platform: str = ""
    device_type: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _scaler_jsonable(v):
    """JSON-safe without destroying integer types."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if torch.is_tensor(v):
        return float(v.item()) if v.numel() == 1 else v.tolist()
    return v


def save_checkpoint(
    out_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    meta: CheckpointMeta,
    data_cursor: dict[str, Any],
    config: dict[str, Any],
    metrics: list[dict] | None = None,
) -> Path:
    out = Path(out_dir)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    # --- model: always fp32, always on CPU ---
    raw = model.module if hasattr(model, "module") else model
    sd = {k: v.detach().to("cpu", torch.float32) for k, v in raw.state_dict().items()}
    try:
        from safetensors.torch import save_file

        # Tied weights share storage; safetensors refuses aliases, so clone.
        save_file({k: v.clone().contiguous() for k, v in sd.items()},
                  str(tmp / "model.safetensors"))
    except ImportError:
        torch.save(sd, tmp / "model.pt")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    if scheduler is not None:
        (tmp / "scheduler.json").write_text(
            json.dumps(scheduler.state_dict()), encoding="utf-8"
        )
    if scaler is not None and hasattr(scaler, "state_dict"):
        # Do NOT coerce everything to float. GradScaler's `growth_interval`
        # and `_growth_tracker` are ints, and torch's C++ `_amp_update_scale_`
        # rejects a float: a resumed session died with "argument
        # 'growth_interval' must be int, not float" on its first optimizer
        # step. Booleans are checked before ints because bool is a subclass.
        (tmp / "scaler.json").write_text(
            json.dumps({k: _scaler_jsonable(v)
                        for k, v in scaler.state_dict().items()}),
            encoding="utf-8",
        )

    torch.save(collect_rng_state(), tmp / "rng.pt")
    (tmp / "data_cursor.json").write_text(json.dumps(data_cursor, indent=2), encoding="utf-8")
    (tmp / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    (tmp / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if metrics:
        with (tmp / "metrics.jsonl").open("w", encoding="utf-8") as fh:
            for row in metrics[-5000:]:
                fh.write(json.dumps(row) + "\n")

    manifest = {
        p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size}
        for p in sorted(tmp.iterdir())
        if p.is_file()
    }
    (tmp / "MANIFEST.sha256").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _atomic_promote(tmp, out)
    return out


def _atomic_promote(tmp: Path, out: Path, attempts: int = 6) -> None:
    """Move a finished checkpoint directory into place.

    A half-written checkpoint must never be loadable, so the write goes to a
    temporary directory and is promoted only once complete.

    Windows makes this awkward: directory deletion is not synchronous, and
    `os.replace` refuses a destination that still exists, so a promote issued
    immediately after `rmtree` fails with `WinError 5: Access is denied`. It
    also fails if any file in the tree still has an open handle (an indexer or
    antivirus scanner is enough).

    So: retry with backoff, and if the rename still will not go through, fall
    back to moving the old directory aside first. `MANIFEST.sha256` is written
    last inside `tmp`, so even a worst-case partial state fails
    `verify_checkpoint` rather than loading silently.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
            os.replace(tmp, out)
            return
        except OSError as exc:
            last = exc
            time.sleep(0.25 * (2**i))

    # Last resort: park the old directory under a unique name and retry once.
    if out.exists():
        parked = out.with_name(f"{out.name}.old-{int(time.time())}")
        try:
            os.replace(out, parked)
            os.replace(tmp, out)
            shutil.rmtree(parked, ignore_errors=True)
            return
        except OSError as exc:
            last = exc

    raise RuntimeError(f"could not promote checkpoint {tmp} -> {out}: {last}")


def verify_checkpoint(ckpt_dir: str | Path) -> bool:
    d = Path(ckpt_dir)
    mpath = d / "MANIFEST.sha256"
    if not mpath.exists():
        return False
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for name, info in manifest.items():
        p = d / name
        if not p.exists() or p.stat().st_size != info["bytes"] or _sha256(p) != info["sha256"]:
            return False
    return True


def load_checkpoint(
    ckpt_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str = "cpu",
    strict_verify: bool = True,
) -> dict[str, Any]:
    d = Path(ckpt_dir)
    if strict_verify and not verify_checkpoint(d):
        raise RuntimeError(f"checkpoint {d} failed hash verification")

    st = d / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        sd = load_file(str(st), device=map_location)
    else:
        sd = torch.load(d / "model.pt", map_location=map_location)

    raw = model.module if hasattr(model, "module") else model
    missing, unexpected = raw.load_state_dict(sd, strict=False)
    # Tied embeddings mean lm_head.weight legitimately may not appear.
    missing = [m for m in missing if not m.endswith("lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")

    if optimizer is not None and (d / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(d / "optimizer.pt", map_location=map_location))
        # Adam's exp_avg/exp_avg_sq load onto `map_location` (cpu by default)
        # while the parameters are already on the accelerator, and the first
        # optimizer.step() then dies with "Expected all tensors to be on the
        # same device, cuda:0 and cpu". Only reachable on a *resume* that
        # carries real optimizer state, so a fresh run and a CPU-only resume
        # test both pass while a real handover fails.
        for group in optimizer.param_groups:
            for param in group["params"]:
                st = optimizer.state.get(param)
                if not st:
                    continue
                for k, v in st.items():
                    if torch.is_tensor(v) and v.device != param.device:
                        st[k] = v.to(param.device)
    if scheduler is not None and (d / "scheduler.json").exists():
        scheduler.load_state_dict(json.loads((d / "scheduler.json").read_text(encoding="utf-8")))
    if scaler is not None and (d / "scaler.json").exists() and hasattr(scaler, "load_state_dict"):
        try:
            raw_scaler = json.loads((d / "scaler.json").read_text(encoding="utf-8"))
            # Checkpoints written before the fix above stored these as floats.
            for key in ("growth_interval", "_growth_tracker"):
                if key in raw_scaler and raw_scaler[key] is not None:
                    raw_scaler[key] = int(raw_scaler[key])
            scaler.load_state_dict(raw_scaler)
        except Exception:
            pass  # moving fp16->bf16 backend: scaler state is not transferable
    if (d / "rng.pt").exists():
        restore_rng_state(torch.load(d / "rng.pt", map_location="cpu", weights_only=False))

    return {
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
        "data_cursor": json.loads((d / "data_cursor.json").read_text(encoding="utf-8")),
        "config": json.loads((d / "config.json").read_text(encoding="utf-8")),
    }


# ------------------------------------------------------------------ #
# Remote store
# ------------------------------------------------------------------ #


class CheckpointStore:
    """Push/pull checkpoints to a Hugging Face repo (or a local directory)."""

    def __init__(self, uri: str, token: str | None = None,
                 cache_dir: str | Path | None = None):
        self.uri = uri
        self.is_hf = uri.startswith("hf://")
        self.repo_id = uri[len("hf://"):] if self.is_hf else None
        self.token = token or os.environ.get("HF_TOKEN")
        self.cache_dir = (Path(cache_dir) if cache_dir
                          else writable_scratch(".ckpt_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.is_hf:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    def push(self, local_dir: str | Path, name: str) -> None:
        local = Path(local_dir)
        if not self.is_hf:
            dest = Path(self.uri) / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(local, dest)
            return
        from huggingface_hub import HfApi

        HfApi(token=self.token).upload_folder(
            repo_id=self.repo_id, folder_path=str(local),
            path_in_repo=name, commit_message=f"ckpt {name}",
        )

    def pull(self, name: str) -> Path:
        if not self.is_hf:
            return Path(self.uri) / name
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=self.repo_id, allow_patterns=f"{name}/*",
            cache_dir=str(self.cache_dir), token=self.token,
        )
        return Path(path) / name

    def prune(self, keep: list[str]) -> list[str]:
        """Delete rolling checkpoints not in `keep`. Milestones are never pruned."""
        removed: list[str] = []
        if not self.is_hf:
            root = Path(self.uri)
            for d in root.iterdir() if root.exists() else []:
                if d.is_dir() and d.name.startswith("ckpt-") and d.name not in keep:
                    shutil.rmtree(d)
                    removed.append(d.name)
            return removed
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        names = {
            f.split("/")[0]
            for f in api.list_repo_files(self.repo_id)
            if f.startswith("ckpt-")
        }
        for name in names - set(keep):
            api.delete_folder(repo_id=self.repo_id, path_in_repo=name,
                              commit_message=f"prune {name}")
            removed.append(name)
        return removed


def checkpoint_name(step: int) -> str:
    return f"ckpt-{step:08d}"
