"""Rotary position embedding (RoPE) — 1D and 2D.

Relative position encoding applied inside attention: rotates (q, k) by an angle
proportional to position so their dot product depends only on relative distance.
RoPE is the only positional mechanism in V28 — the V25 duplicated sinusoidal
absolute "pilot" PE is dropped (RoPE + the causal conv's local context suffice).

* **1D-RoPE** (``apply_rope``): text tokens (and audio frames) rotate by their
  sequence position.
* **2D-RoPE** (``apply_rope_2d``): image patches rotate by (row, col) of their
  patch grid. Replaces V27's fixed-1024-entry learned positional table (which
  was neither 2D nor scalable past 32x32 patches). The head dimension is split
  in half: one half carries the row rotation, the other the column rotation.

A position offset is supported so a KV-cached decode step can apply RoPE at the
correct absolute position without recomputing history.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Standard RoPE inv-frequency buffer.

    Args:
        head_dim: per-head dimension (must be even).
        rope_theta: base frequency.
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[torch.device, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self,
        positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (cos, sin) for the given absolute ``positions``.

        Args:
            positions: ``[T]`` long/float tensor of absolute positions. For a
                full sequence this is ``arange(T)``; for a cached decode step
                this is the single new absolute position.
            device, dtype: target device/dtype of the returned cos/sin.

        Returns:
            ``(cos, sin)`` each of shape ``[T, head_dim]``.
        """
        key = (device, dtype, int(positions.max().item()) if positions.numel() else 0, positions.shape[0])
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        t = positions.to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = (cos, sin)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and rotate: (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``q`` and ``k``.

    Args:
        q: ``[B, n_q, T, hd]``.
        k: ``[B, n_kv, T, hd]``.
        cos, sin: ``[T, hd]`` — broadcast over batch/heads.

    Returns:
        Rotated ``(q, k)`` with the same shapes.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (cos, sin) ``[T, head_dim]`` for the given 1D ``positions``."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
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
    """2D-RoPE (cos, sin) for a patch grid.

    The head dimension is split in half: the first half encodes the row
    position, the second half the column position. Each half gets its own
    frequency band (``head_dim//2`` dims), so the rotation is a genuine 2D
    relative-position encoding: the q-k dot product depends on (drow, dcol).

    Args:
        rows: ``[T]`` patch row indices (long).
        cols: ``[T]`` patch column indices (long).
        head_dim: per-head dimension (must be divisible by 4).
        rope_theta: base frequency.
        device, dtype: target device/dtype of the returned cos/sin.

    Returns:
        ``(cos, sin)`` each of shape ``[T, head_dim]``.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"2D-RoPE needs head_dim divisible by 4, got {head_dim}")
    half = head_dim // 2
    cos_row, sin_row = rope_cos_sin(rows, half, rope_theta, device, dtype)
    cos_col, sin_col = rope_cos_sin(cols, half, rope_theta, device, dtype)
    cos = torch.cat([cos_row, cos_col], dim=-1)
    sin = torch.cat([sin_row, sin_col], dim=-1)
    return cos, sin


def apply_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D-RoPE to ``q`` and ``k`` with (cos, sin) from :func:`rope_cos_sin_2d`.

    Rotation is per-half: the first ``head_dim//2`` dims use the row band, the
    rest the column band. ``rotate_half`` is applied within each half so the
    two bands rotate independently.

    Args:
        q: ``[B, n_q, T, hd]``.
        k: ``[B, n_kv, T, hd]``.
        cos, sin: ``[T, hd]`` (concatenated row||col bands).

    Returns:
        Rotated ``(q, k)``.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    half = q.shape[-1] // 2

    def _rot2d(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half], x[..., half:]
        r1 = (x1 * cos[..., :half]) + (rotate_half(x1) * sin[..., :half])
        r2 = (x2 * cos[..., half:]) + (rotate_half(x2) * sin[..., half:])
        return torch.cat([r1, r2], dim=-1)

    return _rot2d(q), _rot2d(k)
