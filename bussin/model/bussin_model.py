"""Bussin: Llama-style decoder-only transformer.

Design constraints that shaped this file (see docs/SPEC.md §7):

* **No bfloat16 on the target GPUs.** Kaggle's P100 (Pascal) and T4 (Turing)
  predate Ampere, so training runs fp32 master weights with fp16 autocast and
  a GradScaler. Every numerically delicate op (RMSNorm reciprocal, softmax,
  final logits, loss) is therefore forced to fp32 regardless of autocast.
* **No FlashAttention-2.** It requires sm80+. We use `scaled_dot_product_attention`
  and let PyTorch pick the memory-efficient kernel, which does support sm75/sm60.
* **Packed sequences contain many short documents.** Twitch messages average
  ~12 tokens, so a 2048-token packed sequence can hold 150 unrelated messages.
  An optional block-diagonal document mask stops the model learning dependencies
  across unrelated strangers' text.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BussinConfig

# ------------------------------------------------------------------ #
# Attention backend selection
# ------------------------------------------------------------------ #

try:  # PyTorch >= 2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    def _attn_ctx():
        return sdpa_kernel(_SDPA_BACKENDS)

except ImportError:  # pragma: no cover - older torch
    from contextlib import nullcontext

    def _attn_ctx():
        return nullcontext()


# ------------------------------------------------------------------ #
# Normalisation
# ------------------------------------------------------------------ #


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, computed in fp32.

    The fp32 cast is not optional. Under fp16 autocast, ``x.pow(2).mean(-1)``
    on a 2048-wide activation overflows once activations grow past ~256, which
    is a silent NaN a few thousand steps into a from-scratch run.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)


# ------------------------------------------------------------------ #
# Rotary position embedding
# ------------------------------------------------------------------ #


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = self._build(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)   # (T, head_dim)
        return emb.cos(), emb.sin()

    def forward(self, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:  # extrapolate beyond trained context
            cos, sin = self._build(seq_len)
            self.cos_cached = cos.to(self.cos_cached.device)
            self.sin_cached = sin.to(self.sin_cached.device)
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ------------------------------------------------------------------ #
# Attention
# ------------------------------------------------------------------ #


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, D) -> (B, n_kv*n_rep, T, D) for grouped-query attention."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class Attention(nn.Module):
    def __init__(self, cfg: BussinConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_kv_groups
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # When a document mask is supplied it already encodes causality, so
        # is_causal must be False or the two would compose incorrectly.
        with _attn_ctx():
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=attn_mask is None,
            )

        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out)


# ------------------------------------------------------------------ #
# Feed-forward
# ------------------------------------------------------------------ #


class SwiGLU(nn.Module):
    def __init__(self, cfg: BussinConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ------------------------------------------------------------------ #
# Block
# ------------------------------------------------------------------ #


class Block(nn.Module):
    def __init__(self, cfg: BussinConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ------------------------------------------------------------------ #
# Masking
# ------------------------------------------------------------------ #


def build_document_mask(
    document_ids: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Causal mask restricted to within-document attention.

    ``document_ids`` is (B, T): a monotonically increasing integer per packed
    document. Returns an additive float mask of shape (B, 1, T, T).
    """
    b, t = document_ids.shape
    same_doc = document_ids[:, :, None] == document_ids[:, None, :]      # (B,T,T)
    causal = torch.ones(t, t, dtype=torch.bool, device=document_ids.device).tril()
    allowed = same_doc & causal[None]
    neg_inf = torch.finfo(dtype).min
    return torch.zeros_like(allowed, dtype=dtype).masked_fill(~allowed, neg_inf)[:, None]


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #


class BussinForCausalLM(nn.Module):
    def __init__(self, cfg: BussinConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.gradient_checkpointing = False

        self.apply(self._init_weights)
        # Scaled init on residual-output projections keeps the residual stream
        # variance ~constant with depth (GPT-2 / Llama convention).
        scale = (2 * cfg.n_layers) ** -0.5
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.wo.weight, mean=0.0, std=cfg.init_std * scale)
            torch.nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=cfg.init_std * scale)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------ #

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.embed(input_ids)
        cos, sin = self.rope(t, x.device, x.dtype)

        attn_mask = None
        if document_ids is not None and self.cfg.use_document_mask:
            attn_mask = build_document_mask(document_ids, x.dtype)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, attn_mask, use_reentrant=False
                )
            else:
                x = block(x, cos, sin, attn_mask)

        x = self.norm(x)

        # Logits and loss in fp32: under fp16 autocast a 49k-wide logit row
        # saturates well before cross-entropy is computed.
        logits = self.lm_head(x).float()

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            out["ce_loss"] = ce

            loss = ce
            if self.cfg.z_loss > 0:
                valid = shift_labels != -100
                if valid.any():
                    lse = torch.logsumexp(shift_logits[valid], dim=-1)
                    z = self.cfg.z_loss * (lse**2).mean()
                    out["z_loss"] = z
                    loss = loss + z
            out["loss"] = loss
        return out

    # ------------------------------------------------------------------ #

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def param_groups(self, weight_decay: float = 0.1) -> list[dict]:
        """Decay 2-D weights only; never norms, biases, or embeddings (SPEC §8.3)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "norm" in name or "embed" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Reference sampler. No KV cache -- fine for evaluation, not serving."""
        self.eval()
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.cfg.max_seq_len :]
            logits = self.forward(window)["logits"][:, -1]
            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - torch.softmax(sorted_logits, -1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits
                )

            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            if eos_token_id is not None and (nxt == eos_token_id).all():
                break
        return input_ids


def build_model(cfg: BussinConfig) -> BussinForCausalLM:
    return BussinForCausalLM(cfg)
