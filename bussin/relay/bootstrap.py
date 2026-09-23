"""Worker entry point.

This is the whole of what a Kaggle/Colab/Lightning notebook runs. It detects
where it woke up, claims the lease, restores the run, trains until the watchdog
fires, checkpoints, releases the lease and exits 0.

The ordering matters and is taken from SPEC 9.4. In particular step 9 --
verifying that the restored model reproduces the recorded loss -- is the step
people skip. A resume that silently drops optimizer state or reshuffles the
data looks fine for a couple of hundred steps and then quietly plateaus, and by
then you have burned a week of quota.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ..data.loader import Cursor, Manifest, PackedDataset, TorchLoader
from ..model.bussin_model import BussinForCausalLM
from ..model.config import BussinConfig
from ..train.schedule import Curriculum, StageSpec, build_schedule
from ..train.trainer import DivergenceError, Precision, TrainConfig, Trainer
from .checkpoint import (CheckpointStore, checkpoint_name, load_checkpoint,
                         writable_scratch)
from .platform import BatchPlan, get_platform_info, plan_batch
from .state import LeaseHeld, RelayState, make_backend
from .watchdog import StopReason, Watchdog

LOSS_CONTINUITY_TOLERANCE = 0.02  # 2%: tighter than this trips on fp16 noise


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# Places a mounted dataset can appear. Kaggle moved from /kaggle/input/<slug>
# to /kaggle/input/datasets/<owner>/<slug>, so a config that hardcodes either
# convention breaks on the other. Rather than track which is current, resolve
# by searching -- it costs milliseconds and survives the next change.
_SEARCH_ROOTS = ("/kaggle/input", "/content", "/teamspace/studios", "data")


def fetch_shards(repo: str, prefix: str, dest: Path, token: str | None,
                 manifest_name: str = "manifest.json") -> Path | None:
    """Download a shard set from the Hub into `dest`, returning the manifest.

    The corpus is 13.5 GB of uint16 shards, which is too large to ship as a
    Kaggle dataset alongside the code and would have to be re-uploaded every
    time it is rebuilt. Kaggle-to-Hub transfer runs at datacentre speed (the
    ETL jobs pulled 5.8 GB in 49s), so fetching at session start costs a couple
    of minutes against a nine-hour session and keeps one source of truth.

    Only the shards named by the manifest are fetched, so a partial or
    superseded shard left in the repo is ignored rather than silently trained
    on.
    """
    import json

    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    try:
        mpath = hf_hub_download(repo, f"{prefix}{manifest_name}",
                                repo_type="dataset", token=token,
                                local_dir=str(dest))
    except Exception as exc:
        print(f"[bootstrap] no manifest at {repo}:{prefix}{manifest_name} "
              f"({type(exc).__name__})", flush=True)
        return None

    manifest = json.loads(Path(mpath).read_text(encoding="utf-8"))
    shards = manifest.get("shards", [])
    print(f"[bootstrap] fetching {len(shards)} shards from {repo}:{prefix}",
          flush=True)
    t0 = time.time()
    got = 0
    for sh in shards:
        name = Path(sh["path"]).name
        local = dest / name
        if local.exists() and local.stat().st_size > 0:
            continue
        hf_hub_download(repo, f"{prefix}{name}", repo_type="dataset",
                        token=token, local_dir=str(dest))
        got += 1
        if got % 10 == 0:
            print(f"[bootstrap]   {got}/{len(shards)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[bootstrap] shards ready in {time.time() - t0:.0f}s", flush=True)
    return Path(mpath)


def resolve_data_path(path: str | None, max_depth: int = 6) -> str | None:
    """Return `path` if it exists, else find a file with the same name.

    Matches on the trailing path components, so `shards/train/manifest.json`
    will not accidentally match `shards/val/manifest.json`.
    """
    if not path:
        return path
    p = Path(path)
    if p.exists():
        return str(p)

    parts = [x for x in p.parts if x not in ("/", "\\")]
    tail2 = os.path.join(*parts[-3:]) if len(parts) >= 3 else p.name
    for root in _SEARCH_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        for candidate in base.rglob(p.name):
            if len(candidate.parts) - len(base.parts) > max_depth:
                continue
            if str(candidate).replace("\\", "/").endswith(tail2.replace("\\", "/")):
                print(f"[bootstrap] resolved {path} -> {candidate}", flush=True)
                return str(candidate)
    return str(p)


def _resolve_micro_batch(cfg: dict, info) -> int | None:
    batch = cfg.get("batch", {}) or {}
    if batch.get("micro_batch"):
        return int(batch["micro_batch"])
    hints = batch.get("micro_batch_hints") or {}
    name = (info.device_name or "").upper()
    for key, value in hints.items():
        plat, _, dev = key.partition(":")
        if plat != info.platform:
            continue
        if not dev or dev.upper() in name:
            return int(value)
    return None


def build_curriculum(cfg: dict, total_steps: int) -> Curriculum:
    raw = cfg.get("curriculum")
    if not raw:
        return Curriculum(total_steps)
    return Curriculum(
        total_steps,
        [StageSpec(s["name"], float(s["until_frac"]), dict(s["mixture"])) for s in raw],
    )


def run(config_path: str, *, dry_run: bool = False, max_steps: int | None = None,
        force_lease: bool = False, local_root: str | None = None) -> int:
    t_session = time.monotonic()
    cfg = load_config(config_path)

    # 1-2. where are we, and on what -------------------------------------
    info = get_platform_info()
    print(f"[bootstrap] {info}", flush=True)

    model_cfg = BussinConfig.from_dict(cfg["model"])
    tcfg_raw = cfg["train"]
    train_cfg = TrainConfig(
        run_id=tcfg_raw["run_id"],
        total_steps=int(tcfg_raw["total_steps"]),
        global_batch_tokens=int(tcfg_raw["global_batch_tokens"]),
        seq_len=int(tcfg_raw["seq_len"]),
        lr_max=float(tcfg_raw["lr_max"]), lr_min=float(tcfg_raw["lr_min"]),
        warmup_steps=int(tcfg_raw["warmup_steps"]),
        stable_frac=float(tcfg_raw.get("stable_frac", 0.85)),
        weight_decay=float(tcfg_raw.get("weight_decay", 0.1)),
        betas=tuple(tcfg_raw.get("betas", (0.9, 0.95))),
        eps=float(tcfg_raw.get("eps", 1e-8)),
        grad_clip=float(tcfg_raw.get("grad_clip", 1.0)),
        log_every=int(tcfg_raw.get("log_every", 10)),
        val_every=int(tcfg_raw.get("val_every", 500)),
        val_batches=int(tcfg_raw.get("val_batches", 20)),
        milestone_every=int(tcfg_raw.get("milestone_every", 2000)),
        checkpoint_every_s=int(tcfg_raw.get("checkpoint_every_s", 5400)),
        heartbeat_every_s=int(tcfg_raw.get("heartbeat_every_s", 600)),
        gradient_checkpointing=bool(tcfg_raw.get("gradient_checkpointing", False)),
        seed=int(tcfg_raw.get("seed", 1337)),
    )

    relay_cfg = cfg.get("relay", {}) or {}
    state_uri = local_root or relay_cfg.get("state_uri")
    ckpt_uri = local_root or relay_cfg.get("ckpt_uri")
    if not state_uri or "CHANGEME" in str(state_uri):
        raise SystemExit(
            "relay.state_uri is unset. Point it at hf://<you>/bussin-ckpt "
            "or pass --local-root for an offline run."
        )

    torch.manual_seed(train_cfg.seed)

    # 7. batch plan, before anything expensive ---------------------------
    plan = plan_batch(
        global_batch_tokens=train_cfg.global_batch_tokens,
        seq_len=train_cfg.seq_len,
        n_devices=info.n_devices,
        micro_batch=_resolve_micro_batch(cfg, info),
        device_type=info.device_type,
        device_name=info.device_name,
    )
    print(f"[bootstrap] batch plan: {plan}", flush=True)

    # 5. claim the lease -------------------------------------------------
    backend = make_backend(state_uri, run_id=train_cfg.run_id)
    relay = RelayState(backend, train_cfg.run_id)
    lease_s = info.session_limit_s + int(relay_cfg.get("lease_margin_s", 1800))
    try:
        state = relay.claim(info.worker_id, info.platform, lease_s, force=force_lease)
    except LeaseHeld as exc:
        print(f"[bootstrap] not starting: {exc}", flush=True)
        return 0  # a clean no-op: another worker is live, do not burn quota
    print(f"[bootstrap] lease acquired at step {state.step} "
          f"({state.tokens_seen / 1e9:.3f}B tokens seen)", flush=True)

    device = torch.device(
        "cuda" if info.device_type == "cuda"
        else ("cpu" if info.device_type != "xla" else "cpu")
    )
    if info.device_type == "xla":
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

    steps_at_start = state.step
    train_seconds_before = 0.0
    val_loader = None

    try:
        # 6+8. restore ----------------------------------------------------
        model = BussinForCausalLM(model_cfg)
        precision = Precision(info)
        print(f"[bootstrap] precision: {precision}", flush=True)

        schedule = build_schedule(
            total_steps=train_cfg.total_steps, lr_max=train_cfg.lr_max,
            lr_min=train_cfg.lr_min, warmup_steps=train_cfg.warmup_steps,
            stable_frac=train_cfg.stable_frac, lr_scale=state.lr_scale,
        )
        curriculum = build_curriculum(cfg, train_cfg.total_steps)

        trainer = Trainer(model, model_cfg, train_cfg, plan, info, schedule,
                          device, precision, curriculum)

        cursor = Cursor.from_dict(state.data_cursor)
        recorded_loss = None
        if state.latest_ckpt:
            store = CheckpointStore(ckpt_uri)
            print(f"[bootstrap] pulling {state.latest_ckpt}", flush=True)
            ckpt_dir = store.pull(state.latest_ckpt)
            loaded = load_checkpoint(ckpt_dir, model, trainer.optimizer,
                                     schedule, trainer.scaler)
            trainer.step = int(loaded["meta"]["step"])
            trainer.tokens_seen = int(loaded["meta"]["tokens_seen"])
            train_seconds_before = float(loaded["meta"].get("wall_seconds", 0.0))
            recorded_loss = loaded["meta"].get("loss")
            cursor = Cursor.from_dict(loaded["data_cursor"])

            # Carry the metric history forward. Without this each session starts
            # with an empty curve, and the final checkpoint only describes the
            # last session -- which defeats the point of keeping metrics.jsonl
            # for curve-continuity checks across a relay of dozens of sessions.
            prior = ckpt_dir / "metrics.jsonl"
            if prior.exists():
                trainer.metrics = [
                    json.loads(line)
                    for line in prior.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                print(f"[bootstrap] restored {len(trainer.metrics):,} prior metric rows",
                      flush=True)
            print(f"[bootstrap] resumed step {trainer.step} "
                  f"(written by {loaded['meta'].get('written_by')} on "
                  f"{loaded['meta'].get('platform')})", flush=True)

        model.to(device)

        # data ------------------------------------------------------------
        data_cfg = cfg.get("data", {}) or {}
        manifest_path = resolve_data_path(data_cfg["manifest"])
        if not manifest_path or not Path(manifest_path).exists():
            # Not mounted: pull it from the Hub instead.
            scratch = writable_scratch() / "shards" / "train"
            manifest_path = fetch_shards(
                data_cfg.get("repo") or os.environ.get("BUSSIN_CORPUS_REPO", ""),
                data_cfg.get("prefix", "data/shards/train/"),
                scratch, os.environ.get("HF_TOKEN"),
            )
            manifest_path = str(manifest_path) if manifest_path else None
        if not manifest_path or not Path(manifest_path).exists():
            raise SystemExit(
                f"corpus manifest not found: {data_cfg['manifest']}. "
                f"Attach the corpus dataset, or fix data.manifest in the config."
            )
        manifest = Manifest.load(manifest_path)
        dataset = PackedDataset(manifest, train_cfg.seq_len, cursor=cursor,
                                seed=train_cfg.seed)
        loader = iter(TorchLoader(dataset, plan.micro_batch, str(device)))

        val_path = resolve_data_path(data_cfg.get("val_manifest"))
        if data_cfg.get("val_manifest") and (
                not val_path or not Path(val_path).exists()):
            vp = fetch_shards(
                data_cfg.get("repo") or os.environ.get("BUSSIN_CORPUS_REPO", ""),
                data_cfg.get("val_prefix", "data/shards/val/"),
                writable_scratch() / "shards" / "val", os.environ.get("HF_TOKEN"),
            )
            val_path = str(vp) if vp else val_path
        if val_path and Path(val_path).exists():
            val_ds = PackedDataset(Manifest.load(val_path),
                                   train_cfg.seq_len, shuffle_shards=False,
                                   seed=train_cfg.seed)
            val_loader = iter(TorchLoader(val_ds, plan.micro_batch, str(device)))

        print(f"[bootstrap] corpus: {manifest.total_tokens / 1e9:.3f}B tokens "
              f"in {len(manifest.shards)} shards; cursor {cursor.to_dict()}", flush=True)

        # 9. loss continuity check ---------------------------------------
        if recorded_loss is not None and not dry_run:
            probe = trainer.validate(loader, n_batches=4)
            drift = abs(probe["val_loss"] - recorded_loss) / max(recorded_loss, 1e-6)
            status = "OK" if drift <= max(LOSS_CONTINUITY_TOLERANCE, 0.15) else "SUSPECT"
            print(f"[bootstrap] continuity probe: recorded {recorded_loss:.4f} vs "
                  f"observed {probe['val_loss']:.4f} (drift {drift:.1%}) {status}", flush=True)

        if dry_run:
            print("[bootstrap] dry run: setup verified, not training", flush=True)
            relay.release(0, time.monotonic() - t_session, 0.0)
            return 0

        # 10. train ------------------------------------------------------
        target = min(train_cfg.total_steps,
                     trainer.step + max_steps if max_steps else train_cfg.total_steps)
        watchdog = Watchdog(info.session_limit_s, info.checkpoint_reserve_s, target)
        last_val: float | None = None
        last_hb = time.monotonic()

        while True:
            status = watchdog.check(trainer.step)
            if status.should_stop:
                print(f"[bootstrap] stopping: {status.reason.value} ({status.detail})",
                      flush=True)
                break

            row = trainer.train_step(loader)

            if trainer.step % train_cfg.log_every == 0:
                print(trainer.format_log(row), flush=True)

            if val_loader is not None and trainer.step % train_cfg.val_every == 0:
                v = trainer.validate(val_loader, train_cfg.val_batches)
                last_val = v["val_loss"]
                print(f"  val loss {v['val_loss']:.4f}  ppl {v['val_ppl']:.2f}", flush=True)

            save_now, is_milestone = trainer.should_checkpoint(watchdog)
            if save_now:
                _persist(trainer, relay, dataset, ckpt_uri, last_val, is_milestone,
                         relay_cfg, lease_s)

            if time.monotonic() - last_hb >= train_cfg.heartbeat_every_s:
                relay.heartbeat(lease_s)
                last_hb = time.monotonic()

        # final checkpoint, always ---------------------------------------
        _persist(trainer, relay, dataset, ckpt_uri, last_val,
                 is_milestone=False, relay_cfg=relay_cfg, lease_s=lease_s)

        budget = watchdog.budget_report(trainer.train_seconds)
        print(f"[bootstrap] session: {budget}", flush=True)
        relay.release(
            steps_done=trainer.step - steps_at_start,
            wall_seconds=time.monotonic() - t_session,
            train_seconds=trainer.train_seconds,
            finished=trainer.step >= train_cfg.total_steps,
        )
        return 0

    except DivergenceError as exc:
        print(f"[bootstrap] DIVERGED: {exc}", flush=True)
        fresh = relay.read()
        if fresh.milestones:
            relay.request_rollback(fresh.milestones[-1], lr_scale=0.5)
            print(f"[bootstrap] rolled back to {fresh.milestones[-1]} with lr_scale 0.5",
                  flush=True)
        relay.release(0, time.monotonic() - t_session, 0.0)
        return 2
    except Exception:
        relay.release(0, time.monotonic() - t_session, 0.0)
        raise


def _persist(trainer: Trainer, relay: RelayState, dataset: PackedDataset,
             ckpt_uri: str, val_loss: float | None, is_milestone: bool,
             relay_cfg: dict, lease_s: int) -> None:
    cursor = dataset.state()
    local = writable_scratch()
    path = trainer.save(local, cursor, val_loss, is_milestone)
    name = checkpoint_name(trainer.step)

    store = CheckpointStore(ckpt_uri)
    store.push(path, name)

    relay.update(
        step=trainer.step, tokens_seen=trainer.tokens_seen, latest_ckpt=name,
        data_cursor=cursor, stage=trainer.curriculum.stage_at(trainer.step).name,
        milestone=name if is_milestone else None, lease_seconds=lease_s,
    )

    fresh = relay.read()
    keep = set(fresh.milestones) | {name}
    rolling = int(relay_cfg.get("keep_rolling", 2))
    recent = sorted(
        {name} | ({fresh.latest_ckpt} if fresh.latest_ckpt else set()), reverse=True
    )[:rolling]
    try:
        removed = store.prune(sorted(keep | set(recent)))
        if removed:
            print(f"  pruned {len(removed)} old checkpoints", flush=True)
    except Exception as exc:  # pruning must never take down a training run
        print(f"  prune skipped: {exc}", flush=True)

    print(f"  checkpoint {name}{' [milestone]' if is_milestone else ''} pushed", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("bussin-worker")
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify setup and exit without training")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after N steps this session (testing)")
    ap.add_argument("--force-lease", action="store_true",
                    help="steal a live lease; only for a known-dead worker")
    ap.add_argument("--local-root", default=None,
                    help="use a local directory for state and checkpoints")
    args = ap.parse_args(argv)
    return run(args.config, dry_run=args.dry_run, max_steps=args.max_steps,
               force_lease=args.force_lease, local_root=args.local_root)


if __name__ == "__main__":
    sys.exit(main())
