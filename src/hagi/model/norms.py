"""RMSNorm.

5G analog: RMSNorm = signal normalization before modulation (AGC).
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

    Note: weight may be fp32 (for gradient precision) while input is bf16.
    F.rms_norm handles this correctly in the non-fused path; on CUDA the
    fp32_variance path already upcasts input to match.
    """

    def __init__(self, dim: int, eps: float = 1e-6, fp32_variance: bool = True):
        super().__init__()
        self.eps = eps
        self.fp32_variance = fp32_variance
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp32_variance and x.is_cuda:
            orig_dtype = x.dtype
            x_f32 = x.float()
            out = F.rms_norm(x_f32, x_f32.shape[-1:], self.weight.float(), self.eps)
            return out.to(orig_dtype)
        # Non-CUDA or fp32_variance disabled: cast weight to match input dtype
        # for fused kernel compatibility. The fp32 master is preserved for
        # gradient accumulation in the optimizer.
        weight = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        return F.rms_norm(x, x.shape[-1:], weight, self.eps)
