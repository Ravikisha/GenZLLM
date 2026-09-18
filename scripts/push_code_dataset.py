"""Ship the Bussin source to Kaggle as a private dataset.

The worker needs the code. Two ways to get it there:

  git clone   needs a public repo, or a token in the notebook, and costs
              clone time on every session start
  dataset     mounts instantly at /kaggle/input, stays private, no second
              account involved

The dataset wins, and for the same reason the corpus is a dataset: attached
inputs are *mounted*, not downloaded, so they cost no session time. Pushing a
new version is one command, and Kaggle keeps the version history.

Run:
  python scripts/push_code_dataset.py                 # create or update
  python scripts/push_code_dataset.py --dry-run       # show what ships
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.config import project

ROOT = Path(__file__).resolve().parents[1]

# Code, configs, pipelines, and the lexicon -- the ETL stages need all four,
# and the lexicon is small enough (~10 MB) to ship rather than mount separately.
INCLUDE = ["bussin", "configs", "pipelines", "scripts", "data/lexicon"]
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".bin", ".idx", ".safetensors", ".pt"}
EXCLUDE_DIRS = {"data/_raw", "data/corpus", "data/shards", "data/audit"}
# Belt and braces: these must never leave the machine.
FORBIDDEN = {".env", "kaggle.json", "access_token", ".env.local"}


def collect(root: Path) -> list[Path]:
    files: list[Path] = []
    for top in INCLUDE:
        base = root / top
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_NAMES for part in p.parts):
                continue
            if p.suffix in EXCLUDE_SUFFIXES:
                continue
            if p.name in FORBIDDEN:
                raise SystemExit(f"refusing to ship {p}: looks like a secret")
            files.append(p)
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=None, help="default <user>/bussin-code")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default="update bussin source")
    args = ap.parse_args()

    cfg = project()
    user = cfg.creds.kaggle_username
    if not user:
        raise SystemExit("KAGGLE_USERNAME not set")
    slug = args.slug or f"{user}/bussin-code"

    files = collect(ROOT)
    total = sum(f.stat().st_size for f in files)
    print(f"shipping {len(files)} files, {total / 1e6:.2f} MB -> {slug}")
    for f in files[:12]:
        print(f"   {f.relative_to(ROOT)}")
    if len(files) > 12:
        print(f"   ... and {len(files) - 12} more")

    if args.dry_run:
        print("\ndry run; nothing uploaded")
        return 0

    stage = Path(tempfile.mkdtemp(prefix="bussin_code_"))
    try:
        for f in files:
            dest = stage / f.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

        (stage / "dataset-metadata.json").write_text(
            json.dumps({"title": "bussin-code", "id": slug,
                        "licenses": [{"name": "CC0-1.0"}]}, indent=2),
            encoding="utf-8",
        )

        cfg.creds.export_kaggle()
        cfg.creds.write_kaggle_json()
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        exists = True
        try:
            api.dataset_status(slug)
        except Exception:
            exists = False

        if exists:
            print("\nexisting dataset -> pushing a new version")
            api.dataset_create_version(str(stage), version_notes=args.message,
                                       dir_mode="zip", quiet=False)
        else:
            print("\nnew dataset -> creating (private)")
            api.dataset_create_new(str(stage), public=False, dir_mode="zip",
                                   quiet=False)

        print(f"\nattach to a kernel as: {slug}")
        print(f"worker path: /kaggle/input/{slug.split('/')[-1]}")
        return 0
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
