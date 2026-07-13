"""2D Geometric Product — systematic parity channel code.

V5: GP2D acts as a channel encoder. The geometric product between
adjacent token positions generates parity information — redundant data
that the decoder can use for error correction (consistency checking).

Structure: h = data + gate * GP(data[t-1], data[t], data[t+1])
where GP output = parity bits. The residual returned is the parity
contribution, tracked as parity_strength for the channel coding loss.

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
from hagi_v4.model.codec_contracts import GP2DDecodeConfig
from hagi_v4.model.norms import RMSNorm


class GeometricProduct2D(nn.Module):
    """2D Geometric Product — temporal Clifford convolution.

    Reshapes hidden state into multivectors [B, T, n_heads, 8], computes
    geometric products with temporally shifted positions, weights them
    with learnable temporal weights, and blends into the residual stream
    via a sigmoid gate.
    """

    def __init__(self, cfg: GP2DDecodeConfig, hidden_size: int = 576):
        super().__init__()
        self.cfg = cfg
        self.w = cfg.window
        self.n_heads = hidden_size // 8
        self.temporal_weights = nn.Parameter(torch.ones(2 * cfg.window + 1))
        nn.init.normal_(self.temporal_weights, mean=0.0, std=0.02)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))
        self.norm = RMSNorm(hidden_size)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H = h.shape
        mv = h.reshape(B, T, self.n_heads, 8)
        deltas = list(range(-self.w, self.w + 1))
        if len(deltas) > 1:
            shifted_list = []
            for d in deltas:
                if d == 0:
                    shifted_list.append(mv)
                elif d > 0:
                    padded = torch.zeros(B, d, self.n_heads, 8, device=h.device, dtype=h.dtype)
                    shifted_list.append(torch.cat([padded, mv[:, : max(0, T - d)]], dim=1)[:, :T])
                else:
                    ad = -d
                    padded = torch.zeros(B, ad, self.n_heads, 8, device=h.device, dtype=h.dtype)
                    shifted_list.append(torch.cat([mv[:, min(ad, T) :], padded], dim=1)[:, :T])
            shifted_stack = torch.stack(shifted_list, dim=0)
            prods = geometric_product(mv.unsqueeze(0), shifted_stack)
            accumulated = (prods * self.temporal_weights.view(-1, 1, 1, 1, 1)).sum(0)
        else:
            accumulated = geometric_product(mv, mv) * self.temporal_weights[0]
        out = accumulated.reshape(B, T, H)
        out = self.norm(self.proj(out))
        gate = torch.sigmoid(self.gate)
        residual = gate * out
        return h + residual, residual
