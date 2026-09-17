"""Model configuration for Bussin.

Parameter counts here are exact, not estimates. `BussinConfig.n_params()`
reproduces the numbers in docs/SPEC.md §7.2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Register control tokens. These are real vocabulary entries, not prompt text:
# the model attends to them like any other token. See SPEC §3.4.
REGISTER_DIMS: dict[str, int] = {
    "slang": 5,   # 0..4  slang density
    "emoji": 4,   # 0..3  emoji / emote rate
    "abbrev": 4,  # 0..3  abbreviation rate
    "elong": 3,   # 0..2  elongation / keysmash
    "formal": 4,  # 0..3  formality
}

SPECIAL_TOKENS: list[str] = (
    ["<|endoftext|>", "<|pad|>", "<|reg|>", "<|/reg|>"]
    + [f"<|{dim}_{i}|>" for dim, n in REGISTER_DIMS.items() for i in range(n)]
    + ["<|user|>", "<|assistant|>", "<|system|>", "<|/turn|>"]
    + [
        "<|keysmash|>",
        "<|email|>",
        "<|phone|>",
        "<|address|>",
        "<|number|>",
        "<|redacted|>",
    ]
)


@dataclass
class BussinConfig:
    """Llama-style decoder-only transformer config.

    Defaults are `bussin-400m`. See SPEC §7.2 for the full ladder.
    """

    name: str = "bussin-400m"

    # --- architecture ---
    vocab_size: int = 49_152
    n_layers: int = 24
    d_model: int = 1152
    n_heads: int = 18
    n_kv_heads: int = 6  # GQA; must divide n_heads
    head_dim: int = 64   # fixed at 64 across the ladder (SPEC §7.2)
    ffn_dim: int | None = None  # None -> round(8/3 * d_model) to multiple of 64
    max_seq_len: int = 2048

    # --- normalisation / position ---
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0

    # --- embeddings ---
    tie_embeddings: bool = True

    # --- init ---
    init_std: float | None = None  # None -> 0.02 for d<=1152, else 1/sqrt(d)

    # --- regularisation ---
    dropout: float = 0.0
    z_loss: float = 1e-4  # SPEC §8.3: cheap insurance against fp16 logit drift

    # --- attention behaviour ---
    use_document_mask: bool = True  # block-diagonal mask within packed sequences

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if self.ffn_dim is None:
            self.ffn_dim = self._round_to(int(round((8 / 3) * self.d_model)), 64)
        if self.init_std is None:
            self.init_std = 0.02 if self.d_model <= 1152 else (1.0 / self.d_model**0.5)
        self.validate()

    @staticmethod
    def _round_to(x: int, mult: int) -> int:
        return max(mult, int(round(x / mult)) * mult)

    def validate(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads*head_dim ({self.n_heads}*{self.head_dim}="
                f"{self.n_heads * self.head_dim}) must equal d_model ({self.d_model})"
            )
        # uint16 token storage constraint -- SPEC §6.7. Halves corpus size on disk.
        if self.vocab_size > 65_536:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) exceeds 65536; token shards are "
                "stored as uint16 (SPEC §6.7)"
            )

    # ------------------------------------------------------------------ #

    @property
    def n_kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def n_params(self) -> dict[str, int]:
        """Exact parameter count, matching SPEC §7.2."""
        d, V = self.d_model, self.vocab_size
        attn = 2 * d * d + 2 * (d * self.n_kv_heads * self.head_dim)  # Wq,Wo + Wk,Wv
        mlp = 3 * d * self.ffn_dim                                    # SwiGLU gate,up,down
        norms = 2 * d                                                 # attn_norm + ffn_norm
        per_layer = attn + mlp + norms
        non_emb = self.n_layers * per_layer
        emb = V * d
        final_norm = d
        total = non_emb + emb + final_norm + (0 if self.tie_embeddings else emb)
        return {
            "total": total,
            "non_embedding": non_emb,
            "embedding": emb * (1 if self.tie_embeddings else 2),
            "per_layer": per_layer,
        }

    def memory_bytes(self, optimizer: str = "adamw") -> dict[str, float]:
        """fp32 master weights + grads + Adam moments. SPEC §7.3."""
        n = self.n_params()["total"]
        states = {"adamw": 2, "sgd": 0, "sgd_momentum": 1}[optimizer]
        return {
            "weights_gb": n * 4 / 1e9,
            "grads_gb": n * 4 / 1e9,
            "optimizer_gb": n * 4 * states / 1e9,
            "total_gb": n * 4 * (2 + states) / 1e9,
        }

    def flops(self, n_tokens: int) -> float:
        """Training FLOPs, 6*N*D (Kaplan/Hoffmann). SPEC §10.1."""
        return 6 * self.n_params()["total"] * n_tokens

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BussinConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> BussinConfig:
        import yaml

        text = Path(path).read_text(encoding="utf-8")
        d = yaml.safe_load(text) if str(path).endswith((".yaml", ".yml")) else json.loads(text)
        return cls.from_dict(d.get("model", d))


# --------------------------------------------------------------------- #
# The ladder. SPEC §7.2.
# --------------------------------------------------------------------- #

PRESETS: dict[str, dict[str, Any]] = {
    "125m": dict(name="bussin-125m", n_layers=14, d_model=768, n_heads=12,
                 n_kv_heads=4, max_seq_len=1024, tie_embeddings=True),
    "300m": dict(name="bussin-300m", n_layers=24, d_model=1024, n_heads=16,
                 n_kv_heads=4, max_seq_len=2048, tie_embeddings=True),
    "400m": dict(name="bussin-400m", n_layers=24, d_model=1152, n_heads=18,
                 n_kv_heads=6, max_seq_len=2048, tie_embeddings=True),
    "1b": dict(name="bussin-1b", n_layers=24, d_model=2048, n_heads=32,
               n_kv_heads=8, max_seq_len=2048, tie_embeddings=False),
}


def preset(size: str) -> BussinConfig:
    if size not in PRESETS:
        raise KeyError(f"unknown preset {size!r}; have {sorted(PRESETS)}")
    return BussinConfig(**PRESETS[size])


if __name__ == "__main__":
    hdr = f"{'name':<14}{'layers':>7}{'d':>6}{'heads':>6}{'kv':>4}{'ffn':>6}"
    print(hdr + f"{'params':>15}{'non-emb':>14}{'opt GB':>9}")
    for size in PRESETS:
        c = preset(size)
        p, m = c.n_params(), c.memory_bytes()
        print(
            f"{c.name:<14}{c.n_layers:>7}{c.d_model:>6}{c.n_heads:>6}"
            f"{c.n_kv_heads:>4}{c.ffn_dim:>6}{p['total']:>15,}"
            f"{p['non_embedding']:>14,}{m['total_gb']:>9.2f}"
        )
