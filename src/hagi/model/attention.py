"""Ternary attention with real grouped-query attention (GQA), RoPE, KV-cache.

Pre-norm MHA with RoPE and four attention modes (bidir / causal / prefix /
soft_causal) so a single stack supports both masked training and causal AR
generation. Grouped-query attention: ``num_query_heads`` query heads share
``num_kv_heads`` key/value heads (KV is repeated across the group). This is
the V25 gap — V25 *claimed* GQA but ignored ``num_kv_heads`` and ran full MHA.

An optional per-layer :class:`~hagi.model.kv_cache.KVCache` makes autoregressive
generation O(T) instead of O(T^2): frozen positions' KV are reused rather than
recomputed. Only the 2D projections (qkv, out_proj) are ternary via
``BitLinear`` when ``use_ternary``; RoPE inv_freq and the attn_norm gain are FP.

The attention-entropy anti-collapse penalty is computed from a fresh
softmax of the scores (training only) and surfaced as a side output for the
loss aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hagi.model.kv_cache import KVCache
from hagi.model.norms import RMSNorm
from hagi.model.rope import RotaryEmbedding, apply_rope
from hagi.model.ternary import BitLinear


@dataclass
class AttentionConfig:
    """Local attention config for the ternary block."""

    num_heads: int = 8  # number of query heads
    num_kv_heads: int = 4  # GQA: number of key/value heads (<= num_heads)
    head_dim: int = 64
    rope_theta: float = 10000.0
    attn_entropy_floor: float = 0.5


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads along the head axis for GQA.

    Args:
        x: ``[B, n_kv, T, hd]``.
        n_rep: ``num_query_heads // num_kv_heads``.

    Returns:
        ``[B, n_q, T, hd]``.
    """
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


def _build_prefix_mask(b: int, t: int, prefix_len: torch.Tensor | int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if prefix_len is None:
        raise ValueError("prefix_len required for attention_mode='prefix'")
    pl = torch.full((b,), prefix_len, device=device, dtype=torch.long) if isinstance(prefix_len, int) else prefix_len.to(device=device, dtype=torch.long)
    idx = torch.arange(t, device=device)
    causal_allowed = idx.view(t, 1) <= idx.view(1, t)
    mask = torch.zeros(b, 1, t, t, device=device, dtype=dtype)
    mask.masked_fill_(~causal_allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    pl_b = pl.view(b, 1, 1, 1)
    both_prefix = (idx.view(1, 1, t, 1) < pl_b) & (idx.view(1, 1, 1, t) < pl_b)
    mask.masked_fill_(both_prefix, 0.0)
    return mask


def _build_soft_causal_mask(t: int, beta: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(t, device=device)
    dist = (idx.view(1, t) - idx.view(t, 1)).clamp(min=0)
    return (-beta * dist.float().to(dtype)).unsqueeze(0).unsqueeze(0)


class Attention(nn.Module):
    """Pre-norm GQA + RoPE with bidir / causal / prefix / soft_causal modes.

    Args:
        hidden_size: H (must equal num_heads * head_dim).
        cfg: attention config.
        norm_eps: RMSNorm epsilon.
        use_ternary: ternarize the 2D qkv / out_proj via BitLinear.
    """

    def __init__(self, hidden_size: int, cfg: AttentionConfig, norm_eps: float = 1e-6, use_ternary: bool = True) -> None:
        super().__init__()
        if hidden_size != cfg.num_heads * cfg.head_dim:
            raise ValueError(f"hidden_size={hidden_size} must equal num_heads*head_dim={cfg.num_heads}*{cfg.head_dim}")
        if cfg.num_heads % cfg.num_kv_heads != 0:
            raise ValueError(f"num_heads ({cfg.num_heads}) must be divisible by num_kv_heads ({cfg.num_kv_heads})")
        self.hidden_size = hidden_size
        self.n_heads = cfg.num_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)

        def _proj(out_features: int) -> nn.Module:
            return BitLinear(hidden_size, out_features, bias=False) if use_ternary else nn.Linear(hidden_size, out_features, bias=False)

        self.q_proj = _proj(cfg.num_heads * cfg.head_dim)
        self.kv_proj = _proj(2 * cfg.num_kv_heads * cfg.head_dim)
        self.out_proj = _proj(cfg.num_heads * cfg.head_dim)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta=cfg.rope_theta)
        self.attn_entropy_floor = float(cfg.attn_entropy_floor)
        self._kv_cache: KVCache | None = None

    def set_attn_entropy_floor(self, floor: float) -> None:
        self.attn_entropy_floor = float(floor)

    def attach_cache(self, cache: KVCache) -> None:
        """Bind a KV-cache for incremental decoding."""
        self._kv_cache = cache

    def detach_cache(self) -> None:
        self._kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mode: str = "causal",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute attention.

        Args:
            x: ``[B, T, H]``.
            attention_mode: one of bidir / causal / prefix / soft_causal.
            prefix_len: prefix length for prefix mode.
            soft_beta: soft-causal decay (default 2.0).
            positions: ``[T]`` absolute positions for RoPE. Defaults to
                ``arange(T)`` offset by the current cache length when a cache
                is attached (so a decode step applies RoPE at the right place).

        Returns:
            ``(out, entropy_penalty)`` where ``out`` is ``[B, T, H]`` and the
            penalty is ``None`` at eval / when the floor is 0.
        """
        h = self.attn_norm(x)
        b, t, _ = h.shape
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(h).view(b, t, 2, self.n_kv_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cache_len = 0
        if self._kv_cache is not None:
            cache_len = self._kv_cache.length
        if positions is None:
            positions = torch.arange(cache_len, cache_len + t, device=q.device)
        cos, sin = self.rope(positions, q.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if self._kv_cache is not None:
            self._kv_cache.update(k, v)
            k, v = self._kv_cache.get()
            # Full-sequence attention: every query attends to all cached keys.
            # The causal / mask structure is applied below over the full T_total.
            attn_mask = _build_full_mask(k.shape[2], t, attention_mode, prefix_len, soft_beta, q.device, q.dtype)
            is_causal = False  # mask already encoded
        else:
            attn_mask, is_causal = _build_mask(t, attention_mode, prefix_len, soft_beta, b, q.device, q.dtype)

        scale = 1.0 / (self.head_dim**0.5)
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        entropy_pen = None
        out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)

        if self.attn_entropy_floor > 0.0 and self.training:
            scores = (q @ k_rep.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask
            if is_causal:
                t_tot = k_rep.shape[2]
                scores = scores + torch.triu(
                    torch.full((t, t_tot), float("-inf"), device=q.device, dtype=scores.dtype), diagonal=(t_tot - t + 1)
                )
            attn_weights = torch.softmax(scores, dim=-1)
            entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1)
            entropy_pen = (self.attn_entropy_floor - entropy).clamp_min(0.0).mean()

        out = out.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        return self.out_proj(out), entropy_pen


def _build_mask(t: int, mode: str, prefix_len, soft_beta, b: int, device, dtype) -> tuple[torch.Tensor | None, bool]:
    """Build the attention mask for the non-cached path. Returns (mask, is_causal)."""
    if mode == "bidir":
        return None, False
    if mode == "causal":
        return None, True
    if mode == "prefix":
        return _build_prefix_mask(b, t, prefix_len, device, dtype), False
    if mode == "soft_causal":
        beta = 2.0 if soft_beta is None else soft_beta
        return _build_soft_causal_mask(t, beta, device, dtype), False
    raise ValueError(f"unknown attention_mode {mode!r}")


def _build_full_mask(t_total: int, t_q: int, mode: str, prefix_len, soft_beta, device, dtype) -> torch.Tensor:
    """Build an explicit additive mask over ``[t_q, t_total]`` for the cached path.

    ``t_q`` is the number of query positions (the new block); ``t_total`` is the
    full cached length. For causal generation of a single token, ``t_q == 1``
    and the single row is fully unmasked (every query attends to all keys). For
    cached prefill of a block, rows are causal over their own block.
    """
    q_idx = torch.arange(t_total - t_q, t_total, device=device)
    k_idx = torch.arange(t_total, device=device)
    # base: causal (query i attends to keys 0..i)
    allowed = k_idx.view(1, t_total) <= (q_idx.view(t_q, 1))
    mask = torch.zeros(1, t_q, t_total, device=device, dtype=dtype)
    mask.masked_fill_(~allowed.unsqueeze(0), float("-inf"))

    if mode == "bidir":
        mask = torch.zeros(1, t_q, t_total, device=device, dtype=dtype)
    elif mode == "prefix":
        if prefix_len is None:
            raise ValueError("prefix_len required for attention_mode='prefix'")
        pl = prefix_len if isinstance(prefix_len, int) else int(prefix_len[0].item())
        prefix_keys = k_idx < pl
        mask.masked_fill_(prefix_keys.view(1, 1, t_total), 0.0)
    elif mode == "soft_causal":
        beta = 2.0 if soft_beta is None else soft_beta
        dist = (q_idx.view(t_q, 1) - k_idx.view(1, t_total)).clamp(min=0)
        mask = (-beta * dist.float().to(dtype)).unsqueeze(0)
    elif mode != "causal":
        raise ValueError(f"unknown attention_mode {mode!r}")
    return mask
