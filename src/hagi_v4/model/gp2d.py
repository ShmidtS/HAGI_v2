"""2D Geometric Product — cross-token Clifford convolution.

The V4 innovation: apply the geometric product between adjacent token
positions in the temporal dimension. This creates a temporal "convolution"
in the Clifford algebra, capturing cross-token geometric structure.

With window w=1, three positions are considered: t-1, t, t+1.
The hidden state [B, T, H] is reshaped to [B, T, n_heads, 8] multivectors,
and geometric_product is applied between the current and shifted positions.
Learnable temporal weights control the contribution of each offset.
A sigmoid gate on a learnable parameter controls residual blending.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi_v4.algebra.clifford import geometric_product
from hagi_v4.config import GP2DConfig
from hagi_v4.model.norms import RMSNorm


class GeometricProduct2D(nn.Module):
    """2D Geometric Product — temporal Clifford convolution.

    Reshapes hidden state into multivectors [B, T, n_heads, 8], computes
    geometric products with temporally shifted positions, weights them
    with learnable temporal weights, and blends into the residual stream
    via a sigmoid gate.
    """

    def __init__(self, cfg: GP2DConfig, hidden_size: int = 576):
        super().__init__()
        self.w = cfg.window
        self.n_heads = hidden_size // 8
        self.temporal_weights = nn.Parameter(torch.ones(2 * cfg.window + 1))
        nn.init.normal_(self.temporal_weights, mean=0.0, std=0.02)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))
        self.norm = RMSNorm(hidden_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, H = h.shape
        mv = h.reshape(B, T, self.n_heads, 8)
        accumulated = torch.zeros_like(mv)
        for i, delta in enumerate(range(-self.w, self.w + 1)):
            shifted = torch.roll(mv, shifts=delta, dims=1)
            prod = geometric_product(mv, shifted)
            accumulated = accumulated + prod * self.temporal_weights[i]
        out = accumulated.reshape(B, T, H)
        out = self.norm(self.proj(out))
        gate = torch.sigmoid(self.gate)
        return h + gate * out
