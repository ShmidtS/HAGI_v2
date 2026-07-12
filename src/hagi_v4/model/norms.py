"""RMSNorm for HAGI V4.

5G analog: RMSNorm = signal normalization before modulation (AGC).
RoPE removed in V7 — 2D FFT provides position encoding via phase.
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
