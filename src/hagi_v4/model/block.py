"""V24 transformer block: attention + Hebbian bilinear FFN, UniLM modes.

Self-contained additive module. Selects attention_mode per call
(bidir / causal / prefix / soft_causal) so a single stack supports both
efficient masked training and AR generation. The token/channel mixer is
the HebbianBilinearFFN (hebbian-mlps) instead of a dense FFT block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.hebbian_ffn import HebbianFFNConfig
from hagi_v4.model.hebbian_ffn import HebbianBilinearFFN
from hagi_v4.model.norms import RMSNorm

@dataclass
class AttentionConfig:
    """Local attention config for the V24 block (does not touch V21 config)."""

    num_heads: int = 6
    head_dim: int = 64
    rope_theta: float = 10000.0
    attn_entropy_floor: float = 0.5  # prevent attention collapse


class RotaryEmbedding(nn.Module):
    """Standard RoPE cache: cos/sin by (T, device, dtype)."""

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_cos_sin(self, T: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (T, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = (cos, sin)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _build_prefix_mask(B: int, T: int, prefix_len: torch.Tensor | int | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if prefix_len is None:
        raise ValueError("prefix_len required for attention_mode='prefix'")
    pl = torch.full((B,), prefix_len, device=device, dtype=torch.long) if isinstance(prefix_len, int) else prefix_len.to(device=device, dtype=torch.long)
    idx = torch.arange(T, device=device)
    causal_allowed = idx.view(T, 1) <= idx.view(1, T)
    mask = torch.zeros(B, 1, T, T, device=device, dtype=dtype)
    mask.masked_fill_(~causal_allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    pl_b = pl.view(B, 1, 1, 1)
    both_prefix = (idx.view(1, 1, T, 1) < pl_b) & (idx.view(1, 1, 1, T) < pl_b)
    mask.masked_fill_(both_prefix, 0.0)
    return mask


def _build_soft_causal_mask(T: int, beta: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(T, device=device)
    dist = (idx.view(1, T) - idx.view(T, 1)).clamp(min=0)
    return (-beta * dist.float().to(dtype)).unsqueeze(0).unsqueeze(0)


class Attention(nn.Module):
    """Pre-norm MHA + RoPE with bidir / causal / prefix / soft_causal modes."""

    def __init__(self, hidden_size: int, cfg: AttentionConfig, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = cfg.num_heads
        self.head_dim = hidden_size // cfg.num_heads
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta=cfg.rope_theta)
        self.attn_entropy_floor = float(cfg.attn_entropy_floor)

    def set_attn_entropy_floor(self, floor: float) -> None:
        self.attn_entropy_floor = float(floor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.attn_norm(x)
        B, T, _ = h.shape
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope.get_cos_sin(T, q.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        is_causal = False
        attn_mask = None
        if attention_mode == "bidir":
            pass
        elif attention_mode == "causal":
            is_causal = True
        elif attention_mode == "prefix":
            attn_mask = _build_prefix_mask(B, T, prefix_len, x.device, x.dtype)
        elif attention_mode == "soft_causal":
            beta = 2.0 if soft_beta is None else soft_beta
            attn_mask = _build_soft_causal_mask(T, beta, x.device, x.dtype)
        else:
            raise ValueError(f"unknown attention_mode {attention_mode!r}")

        scale = 1.0 / (self.head_dim**0.5)
        entropy_pen = None
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)

        # Attention-entropy floor penalty (anti-collapse). Computed with a
        # cheap fused/SDPA-friendly path: run the same attention once to get
        # probabilities and recombine v — no separate [B,H,T,T] materialization
        # held for the main output (the SDPA path above is fused/Flash where
        # available). The penalty is only needed for autograd at train time.
        if self.attn_entropy_floor > 0.0 and self.training:
            scores = (q @ k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask
            if is_causal:
                scores = scores + torch.triu(
                    torch.full((T, T), float("-inf"), device=q.device, dtype=scores.dtype), diagonal=1
                )
            attn_weights = torch.softmax(scores, dim=-1)
            entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1)
            entropy_pen = (self.attn_entropy_floor - entropy).clamp_min(0.0).mean()

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.out_proj(attn_out), entropy_pen


class TransformerBlock(nn.Module):
    """Pre-norm attention + HebbianBilinearFFN with residual."""

    def __init__(self, hidden_size: int, attn_cfg: AttentionConfig, ffn_cfg: HebbianFFNConfig, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.attn = Attention(hidden_size, attn_cfg, norm_eps)
        self.ffn = HebbianBilinearFFN(hidden_size, ffn_cfg, norm_eps)
        self._last_attn_entropy_penalty: torch.Tensor | None = None

    def set_attn_entropy_floor(self, floor: float) -> None:
        self.attn.set_attn_entropy_floor(floor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
    ) -> torch.Tensor:
        attn_out, pen = self.attn(x, attention_mode, prefix_len, soft_beta)
        x = x + attn_out
        self._last_attn_entropy_penalty = pen
        x = x + self.ffn(x)
        return x
