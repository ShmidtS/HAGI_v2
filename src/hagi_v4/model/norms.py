"""RMSNorm and RoPE utilities for HAGI V4.

Ported from V3 norms.py. RoPE works identically for bidirectional attention
(position encoding is position-relative, not direction-dependent).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with optional fp32 variance computation.

    When input is bf16/fp16 on CUDA, upcast to fp32 for the variance
    computation (7 mantissa bits -> 23 bits). The elementwise multiply
    returns the original dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6, fp32_variance: bool = True):
        super().__init__()
        self.eps = eps
        self.fp32_variance = fp32_variance
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp32_variance and x.dtype in (torch.bfloat16, torch.float16) and x.is_cuda:
            orig_dtype = x.dtype
            x_f32 = x.float()
            out = F.rms_norm(x_f32, x_f32.shape[-1:], self.weight.float(), self.eps)
            return out.to(orig_dtype)
        return F.rms_norm(x, x.shape[-1:], self.weight, self.eps)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings.

    Returns:
        cos: [seq_len, head_dim // 2]
        sin: [seq_len, head_dim // 2]
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(end=seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings using rotate_half convention.

    x: [B, H, T, D] or [..., T, D]
    cos/sin: [T, D // 2]

    Uses contiguous reshape to pairs for efficient even/odd rotation.
    """
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_pairs.unbind(-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)
