"""Grouped-query attention with QK-norm, RoPE, KV-cache and windowing.

Attention is the *correlator* of the channel: it scores a query against every
key and uses the softmax of those scores as a soft demodulation weight. Three
properties matter information-theoretically, and V31 enforces all three.

**Bounded logit range (QK-norm).** The information a softmax transports from
scores to output is maximal when the distribution is neither uniform nor
saturated. With unnormalized q/k, the logit scale rides on ``||q|| ||k||``, which
grows with the projection norms — and a matrix-sign optimizer like Muon removes
the ``1/||W||`` brake that plain SGD has, so those norms drift outward
monotonically. Past a point the softmax saturates: the layer stops transporting
information *and* stops passing gradient. RMS-normalizing each head's q and k
makes the logit scale a function of ``head_dim`` alone. This is the single
highest-value change in V31: the V30 run's ce went 2.32 -> 6.6 between step 19k
and 53k with no configuration change, which is what runaway logit scale looks
like.

**Finite memory where it is cheap (sliding window).** A windowed causal layer is
a finite-state channel: O(T*W) work, memory bounded by W. Long-range dependence
is relayed by the full-attention layers, which act as global parity checks.

**Frozen state (KV-cache).** Once a position's key/value are computed they never
change, so decoding is O(T) rather than O(T^2).

Masks are built by the caller and shared across layers — with packed sequences
the mask also encodes document boundaries, and rebuilding that per layer is pure
waste.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hagi.model.ffn import BranchScale, linear
from hagi.model.kv_cache import KVCache
from hagi.model.norms import HeadNorm, RMSNorm
from hagi.model.rope import RotaryEmbedding, apply_rope


@dataclass
class AttentionConfig:
    """Per-layer attention geometry."""

    num_heads: int = 24
    num_kv_heads: int = 6
    head_dim: int = 64
    rope_theta: float = 10000.0
    qk_norm: bool = True
    sliding_window: int = 0  # 0 = full attention
    history_stride: int = 0
    per_head_qk: bool = False  # per-head QK gain (block-diagonal expert merge)



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``[B, n_kv, T, hd]`` to ``[B, n_kv*n_rep, T, hd]`` for GQA."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


def local_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Causal sliding-window SDPA with correct support for every query position.

    The previous train-time optimization truncated K/V to the *last* ``window``
    keys of the sequence. That is only valid for the final ``window`` queries;
    earlier positions received an all-``-inf`` score row and a zero attention
    output (measured: early-token branch norm 0.0). That is both a silent
    correctness bug and wasted capacity on a bandwidth-bound part.

    This path processes queries in chunks of size ``window``. Query block
    ``[i0, i1)`` attends only to keys ``[max(0, i0-window+1), i1)`` — exact
    window semantics, O(T·W) work, no full ``T×T`` materialization. On the
    Radeon 8060S ROCm math SDPA backend this measured ~7.5 ms vs ~12 ms for a
    dense window mask at ``T=1024, W=256``.

    Args:
        q, k, v: ``[B, heads, T, head_dim]`` (GQA already expanded on k/v).
        window: positive window width W.

    Returns:
        Attention output, same shape as ``q``.
    """
    if window <= 0:
        raise ValueError(f"local_window_attention requires window > 0, got {window}")
    t = q.shape[-2]
    if t <= window:
        # Every query's full causal past fits inside the window.
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    outs: list[torch.Tensor] = []
    step = window
    for i0 in range(0, t, step):
        i1 = min(t, i0 + step)
        k0 = max(0, i0 - window + 1)
        qi = q[:, :, i0:i1]
        ki = k[:, :, k0:i1]
        vi = v[:, :, k0:i1]
        if k0 == 0 and i0 == 0:
            # Prefill of the first block: standard lower-triangular causal.
            outs.append(F.scaled_dot_product_attention(qi, ki, vi, is_causal=True))
            continue
        tq = i1 - i0
        tk = i1 - k0
        q_abs = torch.arange(i0, i1, device=q.device).view(1, 1, tq, 1)
        k_abs = torch.arange(k0, i1, device=q.device).view(1, 1, 1, tk)
        allowed = (k_abs <= q_abs) & (k_abs > q_abs - window)
        band = qi.new_zeros(1, 1, tq, tk)
        band = band.masked_fill(~allowed, float("-inf"))
        outs.append(F.scaled_dot_product_attention(qi, ki, vi, attn_mask=band))
    return torch.cat(outs, dim=2)


def compressed_history_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int, stride: int
) -> torch.Tensor:
    """Local attention plus a strided, causal summary of older KV states."""
    t = q.shape[-2]
    if t <= window or stride <= 0:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    outs: list[torch.Tensor] = []
    for i0 in range(0, t, window):
        i1 = min(t, i0 + window)
        local_start = max(0, i1 - window)
        old = torch.arange(0, local_start, stride, device=q.device)
        recent = torch.arange(local_start, i1, device=q.device)
        indices = torch.cat([old, recent])
        ki = k.index_select(-2, indices)
        vi = v.index_select(-2, indices)
        q_abs = torch.arange(i0, i1, device=q.device).view(1, 1, -1, 1)
        k_abs = indices.view(1, 1, 1, -1)
        allowed = k_abs <= q_abs
        bias = q.new_zeros(1, 1, i1 - i0, indices.numel())

        bias = bias.masked_fill(~allowed, float("-inf"))
        outs.append(F.scaled_dot_product_attention(q[:, :, i0:i1], ki, vi, attn_mask=bias))
    return torch.cat(outs, dim=2)


class Attention(nn.Module):
    """Pre-norm GQA with QK-norm, RoPE, optional window and KV-cache.

    Args:
        hidden_size: H, must equal ``num_heads * head_dim``.
        cfg: attention geometry (including this layer's window).
        norm_eps: RMSNorm epsilon.
        use_ternary: quantize the 2D projections.
        residual_scale: init scale on ``out_proj`` (depth scaling).
    """

    def __init__(
        self,
        hidden_size: int,
        cfg: AttentionConfig,
        norm_eps: float = 1e-5,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
        init_orthogonal: bool = False,
        rope: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        if hidden_size != cfg.num_heads * cfg.head_dim:
            raise ValueError(
                f"hidden_size={hidden_size} must equal num_heads*head_dim="
                f"{cfg.num_heads}*{cfg.head_dim}"
            )
        if cfg.num_kv_heads < 1 or cfg.num_heads % cfg.num_kv_heads:
            raise ValueError(
                f"num_heads ({cfg.num_heads}) must be a positive multiple of "
                f"num_kv_heads ({cfg.num_kv_heads})"
            )
        self.hidden_size = hidden_size
        self.n_heads = cfg.num_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.sliding_window = int(cfg.sliding_window or 0)
        self.history_stride = int(getattr(cfg, "history_stride", 0) or 0)

        def proj(out_features: int) -> nn.Module:
            return linear(hidden_size, out_features, use_ternary, init_orthogonal)

        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        # Fused QKV projection: one [H, H + 2*K*Hd] matrix instead of separate
        # q and kv projections. On a launch-bound part (Radeon 8060S iGPU) this
        # halves the number of linear calls per layer and lets backward read the
        # weight once instead of twice — measured ~6% on the isolated matmul,
        # more when launch overhead dominates.
        n_q_out = cfg.num_heads * cfg.head_dim
        n_kv_out = 2 * cfg.num_kv_heads * cfg.head_dim
        self.qkv_proj = proj(n_q_out + n_kv_out)
        self.out_proj = proj(cfg.num_heads * cfg.head_dim)
        if init_orthogonal:
            with torch.no_grad():
                self.out_proj.weight.mul_(residual_scale)
        else:
            nn.init.normal_(self.out_proj.weight, std=residual_scale / hidden_size**0.5)

        self.q_norm = (
            HeadNorm(cfg.head_dim, eps=norm_eps, per_head=cfg.per_head_qk, n_heads=cfg.num_heads)
            if cfg.qk_norm
            else None
        )
        self.k_norm = (
            HeadNorm(cfg.head_dim, eps=norm_eps, per_head=cfg.per_head_qk, n_heads=cfg.num_kv_heads)
            if cfg.qk_norm
            else None
        )
        # Shared RoPE across layers (same head_dim/theta) — one table, one cache.
        self.rope = rope if rope is not None else RotaryEmbedding(cfg.head_dim, rope_theta=cfg.rope_theta)
        self.branch_scale = BranchScale(residual_scale)
        self._kv_cache: KVCache | None = None

    def attach_cache(self, cache: KVCache) -> None:
        """Bind a KV-cache for incremental decoding."""
        self._kv_cache = cache

    def detach_cache(self) -> None:
        self._kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention branch output for ``[B, T, H]`` (residual added by the caller).

        Args:
            x: ``[B, T, H]`` residual-stream input.
            positions: ``[T]`` absolute positions for RoPE; defaults to
                ``arange(T)`` offset by the cache length.
            mask: additive mask broadcastable to ``[B, 1, T, T_total]``. None
                means plain causal, which uses SDPA's fused causal kernel.

        Returns:
            ``[B, T, H]``.
        """
        h = self.attn_norm(x)
        b, t, _ = h.shape
        # One fused projection, then split into q / k / v heads.
        qkv = self.qkv_proj(h)
        n_q = self.n_heads * self.head_dim
        q = qkv[..., :n_q].view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        kv = qkv[..., n_q:].view(b, t, 2, self.n_kv_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QK-norm before RoPE (Gemma-2 / Chameleon convention). RoPE is a
        # rotation and preserves norms, so the order only decides which tensor
        # the learnable gain scales.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cache_len = self._kv_cache.length if self._kv_cache is not None else 0
        if positions is None:
            positions = torch.arange(cache_len, cache_len + t, device=q.device)
        cos, sin = self.rope(positions, q.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if self._kv_cache is not None:
            self._kv_cache.update(k, v)
            k, v = self._kv_cache.get()

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        window = self.sliding_window
        if mask is not None:
            # Caller-built mask already encodes window / docs / prefix.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif (
            self.training
            and self.history_stride > 0
            and self.sliding_window > 0
            and self._kv_cache is None
            and k.shape[-2] == t
        ):
            out = compressed_history_attention(q, k, v, self.sliding_window, self.history_stride)
        elif self.training and self.sliding_window > 0 and self._kv_cache is None and k.shape[-2] == t:
            # Pure window, no doc/prefix constraints: O(T·W) local SDPA.
            out = local_window_attention(q, k, v, window)
        elif t == 1:
            # Single-token decode: one query against the whole cached prefix.
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.branch_scale(
            self.out_proj(out.transpose(1, 2).reshape(b, t, self.hidden_size))
        )


def build_attention_mask(
    t_q: int,
    t_total: int,
    *,
    window: int = 0,
    doc_ids: torch.Tensor | None = None,
    prefix_len: int = 0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build the additive attention mask, or None when plain causal suffices.

    Composes three constraints:

    * **Causal** — a query attends only to positions at or before it. This is the
      objective's definition; violating it makes next-token prediction trivial at
      training time and impossible at inference.
    * **Document boundaries** — with packed sequences one window holds several
      independent documents. Attention across a boundary leaks one source into
      another, so ``doc_ids`` restricts each query to its own document.
    * **Window** — a query attends only to the last ``window`` positions.

    A multimodal prefix (``prefix_len > 0``) is exempt from all three: prefix
    tokens are mutually visible and visible to every text position, because the
    prefix is side information available in full before decoding starts.

    Args:
        t_q: number of query positions.
        t_total: number of key positions (>= t_q; the difference is cached).
        window: window size, 0 for unlimited.
        doc_ids: ``[B, t_total]`` document id per key position (queries take the
            last ``t_q``). None disables the boundary constraint.
        prefix_len: leading positions that bypass causality.
        device, dtype: output device/dtype.

    Returns:
        ``[B, 1, t_q, t_total]`` (or ``[1, 1, t_q, t_total]`` when ``doc_ids`` is
        None) additive mask with ``-inf`` at disallowed pairs, or None when the
        result is exactly plain causal.
    """
    if window <= 0 and doc_ids is None and prefix_len <= 0:
        return None

    q_pos = torch.arange(t_total - t_q, t_total, device=device).view(t_q, 1)
    k_pos = torch.arange(t_total, device=device).view(1, t_total)
    allowed = k_pos <= q_pos
    if window > 0:
        allowed = allowed & (k_pos > q_pos - window)
    allowed = allowed.view(1, 1, t_q, t_total)

    if doc_ids is not None:
        if doc_ids.shape[1] != t_total:
            raise ValueError(f"doc_ids has {doc_ids.shape[1]} positions, expected {t_total}")
        q_doc = doc_ids[:, t_total - t_q :].unsqueeze(-1)  # [B, t_q, 1]
        same_doc = (q_doc == doc_ids.unsqueeze(1)).unsqueeze(1)  # [B, 1, t_q, t_total]
        allowed = allowed & same_doc

    if prefix_len > 0:
        is_prefix_key = (k_pos < prefix_len).view(1, 1, 1, t_total)
        allowed = allowed | is_prefix_key

    mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))
