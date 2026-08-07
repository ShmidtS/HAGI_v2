"""Rotary position embedding — 1D and 2D.

RoPE encodes position as a rotation of ``(q, k)`` in ``head_dim/2`` planes, at
frequencies ``theta^(-2i/d)``. Because a rotation is orthogonal, the inner
product ``<R(p_q) q, R(p_k) k>`` depends only on ``p_q - p_k``: relative
position, achieved without adding anything to the signal. That matters for a
communication view of attention — an additive positional signal consumes
amplitude that would otherwise carry content, while a rotation consumes none.

Two variants:

* **1D** (:func:`rope_cos_sin`) — sequence position, for text and audio frames.
* **2D** (:func:`rope_cos_sin_2d`) — the head dimension is split into a row band
  and a column band, so an image patch's rotation depends on ``(row, col)`` and
  the inner product depends on ``(Δrow, Δcol)``. This replaces a fixed learned
  table, which cannot represent a resolution outside its training range.

All rotations use the half-split convention: ``x = [x1 | x2]`` rotates to
``[x1*cos - x2*sin | x2*cos + x1*sin]``, with ``cos``/``sin`` duplicated across
the halves. It is equivalent to the interleaved-pairs form up to a permutation of
channels and is the convention every current implementation uses.
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map ``[x1 | x2]`` to ``[-x2 | x1]`` over the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1D rotation table for ``positions``.

    Args:
        positions: ``[T]`` absolute positions.
        head_dim: per-head width (must be even).
        rope_theta: base frequency.
        device, dtype: output device/dtype.

    Returns:
        ``(cos, sin)``, each ``[T, head_dim]`` with the band duplicated across
        halves.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions.to(device=device, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rope_cos_sin_2d(
    rows: torch.Tensor,
    cols: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D rotation table: first half encodes row, second half encodes column.

    Args:
        rows, cols: ``[T]`` grid coordinates per token.
        head_dim: per-head width (must be divisible by 4 — each band needs an
            even width of its own).
        rope_theta: base frequency.
        device, dtype: output device/dtype.

    Returns:
        ``(cos, sin)``, each ``[T, head_dim]`` as ``row_band || col_band``.
    """
    if head_dim % 4:
        raise ValueError(f"2D-RoPE needs head_dim divisible by 4, got {head_dim}")
    half = head_dim // 2
    cos_r, sin_r = rope_cos_sin(rows, half, rope_theta, device, dtype)
    cos_c, sin_c = rope_cos_sin(cols, half, rope_theta, device, dtype)
    return torch.cat([cos_r, cos_c], dim=-1), torch.cat([sin_r, sin_c], dim=-1)


def rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply a rotation table to ``[..., T, head_dim]``.

    Handles both 1D tables (one band spanning ``head_dim``) and 2D tables (two
    independent bands of ``head_dim/2``): the halves are rotated separately, so a
    concatenated row/column table rotates each band by its own angle.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    r1 = x1 * cos[..., :half] + rotate_half(x1) * sin[..., :half]
    r2 = x2 * cos[..., half:] + rotate_half(x2) * sin[..., half:]
    return torch.cat([r1, r2], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q`` ``[B, n_q, T, hd]`` and ``k`` ``[B, n_kv, T, hd]`` in place-free form."""
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    return q * cos_b + rotate_half(q) * sin_b, k * cos_b + rotate_half(k) * sin_b


class RotaryEmbedding(nn.Module):
    """Cached 1D RoPE table.

    The table is a pure function of ``(positions, device, dtype)``, so it is
    cached. Prefill and decode ask for different position ranges, so the cache is
    keyed by the range rather than only its length.

    Args:
        head_dim: per-head width (even).
        rope_theta: base frequency.
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.rope_theta = float(rope_theta)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``[T, head_dim]`` for ``positions``.

        When ``positions`` is the default ``arange(T)`` (training prefill), the
        cache key is derived from its length alone — no ``.item()`` calls, so
        the whole forward stays inside a single ``torch.compile`` graph.
        """
        if positions.numel() == 0:
            empty = torch.empty(0, self.head_dim, device=device, dtype=dtype)
            return empty, empty
        # Derive cache key without data-dependent Python branches. In training
        # (prefill) positions is always arange(cache_len, cache_len+T); for
        # compile-friendliness we use numel and min/max tensor reads only when
        # not compiling.
        compiling = not torch.jit.is_scripting() and torch.compiler.is_compiling()
        if compiling:
            # During compilation we cannot use data-dependent keys. The table is
            # a pure function of positions, so just compute it directly.
            freqs = torch.outer(
                positions.to(device=device, dtype=torch.float32),
                self.inv_freq.to(device=device, dtype=torch.float32),
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)
        key = (int(positions[0]), int(positions[-1]), positions.shape[0], device.type, device.index, dtype)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        freqs = torch.outer(
            positions.to(device=device, dtype=torch.float32),
            self.inv_freq.to(device=device, dtype=torch.float32),
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        entry = (emb.cos().to(dtype), emb.sin().to(dtype))
        if len(self._cache) > 16:
            self._cache.clear()
        self._cache[key] = entry
        return entry
