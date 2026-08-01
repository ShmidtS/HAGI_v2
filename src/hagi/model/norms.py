"""Normalization layers — automatic gain control for the channel.

Two norms, both RMS-based:

* :class:`RMSNorm` — the residual-stream AGC. Variance is always computed in
  fp32: at bf16 the 7-bit mantissa makes ``mean(x^2)`` lossy enough to bias the
  gain, and the gain error is a multiplicative distortion applied to every
  downstream matmul.
* :class:`HeadNorm` — per-head QK normalization. Normalizing q and k before the
  correlator bounds the logit dynamic range, so the softmax stays inside its
  high-slope region. A saturated softmax transports ~zero information from the
  score to the output *and* has ~zero gradient, which is the dominant
  divergence mode of a deep pre-norm stack trained with a matrix-sign
  optimizer (observed in V30: ce 2.32 -> 6.6 between step 19k and 53k).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with an fp32 variance accumulator and a learnable gain.

    Args:
        dim: normalized (last) dimension.
        eps: variance floor.
        fp32_variance: compute variance in fp32 (default). On ROCm the fused
            bf16 rms_norm kernel is 5x faster and numerically identical for
            the value ranges seen here (verified: max diff 0.0), so the caller
            can disable it for speed. fp32 is kept as the safe default because
            it is the variance precision the gain was tuned against.
    """

    def __init__(self, dim: int, eps: float = 1e-5, fp32_variance: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = float(eps)
        self.fp32_variance = bool(fp32_variance)
        # A gain at 1.0 receives gradients around 1e-4; the smallest bf16 step
        # above 1.0 is ~0.0078, so under bf16 those updates round to zero and the
        # layer is frozen at initialization for the whole run. See
        # hagi.train.loop.cast_model, which reads this marker.
        self.keep_fp32 = True
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp32_variance and x.dtype in (torch.float16, torch.bfloat16):
            out = F.rms_norm(x.float(), (self.dim,), self.weight.float(), self.eps)
            return out.to(x.dtype)
        return F.rms_norm(x, (self.dim,), self.weight.to(x.dtype), self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, fp32_variance={self.fp32_variance}"


class HeadNorm(nn.Module):
    """Per-head RMSNorm over ``head_dim`` for QK normalization.

    Applied to ``[B, n_heads, T, head_dim]`` (or any tensor whose last
    dimension is ``head_dim``). One shared gain vector across heads keeps the
    parameter cost at ``head_dim`` and avoids per-head gain drift.

    Args:
        head_dim: per-head dimension.
        eps: variance floor.
        fp32_variance: compute variance in fp32 (default). Same trade as
            :class:`RMSNorm` — bf16 is 5x faster and identical on this ROCm
            build.
    """

    def __init__(self, head_dim: int, eps: float = 1e-5, fp32_variance: bool = True) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.eps = float(eps)
        self.fp32_variance = bool(fp32_variance)
        self.keep_fp32 = True
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp32_variance and x.dtype in (torch.float16, torch.bfloat16):
            out = F.rms_norm(x.float(), (self.head_dim,), self.weight.float(), self.eps)
            return out.to(x.dtype)
        return F.rms_norm(x, (self.head_dim,), self.weight.to(x.dtype), self.eps)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, eps={self.eps}, fp32_variance={self.fp32_variance}"
