"""Credential and project configuration.

Secrets are read from the environment, and `.env` is loaded for local runs.
Nothing is hardcoded, for two reasons that both bite in this project:

* The repo may be made public -- GitHub Actions minutes are unlimited for
  public repositories, which is how the orchestrator runs for free.
* Rotating a leaked token then means editing one gitignored file, not grepping
  through source and git history.

On a runner, set the same names as GitHub Actions secrets. On Kaggle, use
Add-ons -> Secrets. The lookup order is env first, then `.env`, so a runner's
secrets always win.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"


def load_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Minimal .env loader.

    Deliberately not `python-dotenv`: its `find_dotenv()` walks the caller's
    stack frame and raises `AssertionError` when called from `python -` or an
    exec'd string, which is exactly how the orchestrator gets invoked from CI.
    """
    path = path or ENV_FILE
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


class MissingCredential(RuntimeError):
    pass


def get(name: str, default: str | None = None, required: bool = False) -> str | None:
    load_env()
    value = os.environ.get(name, default)
    if required and not value:
        raise MissingCredential(
            f"{name} is not set. Add it to {ENV_FILE} (gitignored) for local "
            f"runs, or as a GitHub Actions / Kaggle secret on a runner."
        )
    return value


def redact(secret: str | None) -> str:
    """For logs. Never print a token in full -- CI logs are often public."""
    if not secret:
        return "<unset>"
    return f"{secret[:6]}...{secret[-4:]} ({len(secret)} chars)"


@dataclass
class Credentials:
    kaggle_username: str | None = None
    kaggle_key: str | None = None
    hf_token: str | None = None
    hf_user: str | None = None

    @classmethod
    def load(cls) -> "Credentials":
        load_env()
        return cls(
            kaggle_username=os.environ.get("KAGGLE_USERNAME"),
            kaggle_key=os.environ.get("KAGGLE_KEY"),
            hf_token=os.environ.get("HF_TOKEN"),
            hf_user=os.environ.get("HF_USER"),
        )

    def export_kaggle(self) -> None:
        """Kaggle's client reads env vars or ~/.kaggle/kaggle.json."""
        if self.kaggle_username:
            os.environ["KAGGLE_USERNAME"] = self.kaggle_username
        if self.kaggle_key:
            os.environ["KAGGLE_KEY"] = self.kaggle_key

    def write_kaggle_json(self, path: Path | None = None) -> Path | None:
        """Materialise ~/.kaggle/kaggle.json, which some tooling requires."""
        if not (self.kaggle_username and self.kaggle_key):
            return None
        path = path or Path.home() / ".kaggle" / "kaggle.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"username": self.kaggle_username, "key": self.kaggle_key}),
            encoding="utf-8",
        )
        try:
            path.chmod(0o600)
        except OSError:
            pass  # Windows
        return path

    def summary(self) -> str:
        return (
            f"kaggle={self.kaggle_username or '<unset>'} "
            f"key={redact(self.kaggle_key)} | "
            f"hf={self.hf_user or '<unset>'} token={redact(self.hf_token)}"
        )


@dataclass
class ProjectConfig:
    ckpt_repo: str = "bussin-ckpt"
    corpus_repo: str = "bussin-corpus"
    lexicon_repo: str = "bussin-lexicon"
    ledger_path: str = "orchestrator/ledger.json"
    dashboard_path: str = "orchestrator/dashboard.json"
    creds: Credentials = field(default_factory=Credentials.load)

    @classmethod
    def load(cls) -> "ProjectConfig":
        load_env()
        creds = Credentials.load()
        user = creds.hf_user or ""

        def qualify(name: str) -> str:
            return name if "/" in name else (f"{user}/{name}" if user else name)

        return cls(
            ckpt_repo=qualify(os.environ.get("BUSSIN_CKPT_REPO", "bussin-ckpt")),
            corpus_repo=qualify(os.environ.get("BUSSIN_CORPUS_REPO", "bussin-corpus")),
            lexicon_repo=qualify(os.environ.get("BUSSIN_LEXICON_REPO", "bussin-lexicon")),
            creds=creds,
        )

    @property
    def state_uri(self) -> str:
        return f"hf://{self.ckpt_repo}"


@lru_cache(maxsize=1)
def project() -> ProjectConfig:
    return ProjectConfig.load()


def verify(verbose: bool = True) -> dict[str, object]:
    """Check both credentials actually work. Run before relying on them."""
    cfg = project()
    cfg.creds.export_kaggle()
    report: dict[str, object] = {"kaggle": None, "hf": None}

    # --- Hugging Face ---
    try:
        from huggingface_hub import HfApi

        me = HfApi(token=cfg.creds.hf_token).whoami()
        role = (me.get("auth") or {}).get("accessToken", {}).get("role")
        report["hf"] = {"ok": True, "user": me.get("name"), "role": role}
    except Exception as exc:
        report["hf"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # --- Kaggle ---
    try:
        cfg.creds.write_kaggle_json()
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        kernels = api.kernels_list(mine=True, page_size=5)
        report["kaggle"] = {
            "ok": True,
            "user": cfg.creds.kaggle_username,
            "own_kernels": len(kernels),
        }
    except Exception as exc:
        report["kaggle"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    if verbose:
        for service, r in report.items():
            assert isinstance(r, dict)
            mark = "OK  " if r.get("ok") else "FAIL"
            detail = ", ".join(f"{k}={v}" for k, v in r.items() if k != "ok")
            print(f"  {mark} {service:<8} {detail}")
    return report


if __name__ == "__main__":
    cfg = project()
    print(f"credentials: {cfg.creds.summary()}")
    print(f"ckpt repo  : {cfg.ckpt_repo}")
    print(f"corpus repo: {cfg.corpus_repo}")
    print("\nverifying:")
    verify()
