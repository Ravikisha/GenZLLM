"""Deploy the dashboard to a free Hugging Face Space.

The Space reads `orchestrator/dashboard.json` from the checkpoint repo, which
the orchestrator rewrites on every tick. So the Space never needs redeploying
when the run advances -- only when the dashboard code itself changes.

Run:
  python scripts/deploy_space.py
  python scripts/deploy_space.py --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bussin.config import project

LOCAL = Path(__file__).resolve().parents[1] / "spaces" / "dashboard"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bussin-dashboard")
    ap.add_argument("--private", action="store_true",
                    help="private Space (still free; only you can view it)")
    args = ap.parse_args()

    cfg = project()
    if not cfg.creds.hf_token or not cfg.creds.hf_user:
        raise SystemExit("HF_TOKEN / HF_USER not set")

    from huggingface_hub import HfApi

    api = HfApi(token=cfg.creds.hf_token)
    repo_id = f"{cfg.creds.hf_user}/{args.name}"

    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static",
                    private=args.private, exist_ok=True)
    print(f"space: https://huggingface.co/spaces/{repo_id}")

    # A static Space needs only index.html, which the orchestrator rewrites on
    # every tick. Seed it now so the Space is live before the first tick.
    from bussin.orchestrator.render import render

    placeholder = {
        "generated_at": "not yet", "model": {}, "live": {}, "progress": {},
        "status": {"action": "none", "reason": "waiting for the first "
                                               "orchestrator tick"},
        "quota": {}, "sessions": {}, "charts": {},
        "notes": ["This page is rewritten by the orchestrator every 15 minutes."],
    }
    api.upload_file(path_or_fileobj=render(placeholder).encode("utf-8"),
                    path_in_repo="index.html", repo_id=repo_id,
                    repo_type="space", commit_message="seed dashboard")
    print("uploaded index.html")

    # The Space needs its own credentials to read a private checkpoint repo.
    for key, value in (("HF_TOKEN", cfg.creds.hf_token),
                       ("BUSSIN_CKPT_REPO", cfg.ckpt_repo)):
        try:
            api.add_space_secret(repo_id=repo_id, key=key, value=value)
            print(f"  secret set: {key}")
        except Exception as exc:
            print(f"  could not set {key} automatically ({type(exc).__name__}); "
                  f"add it in the Space settings")

    print("\nBuilding takes a couple of minutes. Then it auto-refreshes every "
          "120s from the checkpoint repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
