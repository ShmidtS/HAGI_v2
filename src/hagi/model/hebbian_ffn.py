"""HebbianBilinearFFN — trainable bilinear feed-forward block.

Derived from HazyResearch/hebbian-mlps (COLM 2026). The bilinear feature map
``phi(h) = (A0 h) . silu(A1 h)`` has an information-theoretically optimal
capacity structure for associative memory. This version makes A0, A1, W
trainable Parameters, giving a SwiGLU-shaped FFN with the bilinear capacity
geometry.

This is the SINGLE source of truth for the bilinear FFN (V25 duplicated it
across ``hebbian_ffn.py`` and ``block.py``; V27 imports this one). Only the 2D
weights (A0, A1, W) are ternary via ``BitLinear`` when ``use_ternary``; the
per-channel gate is a 1D FP parameter.

The ``(1 + tanh(gate))`` modulation (not a zero-init multiplier) keeps the FFN
branch live at init so A0/A1/W receive gradient — a pure zero-init multiplier
would zero the whole branch and starve it (the dead-gradient failure mode).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hagi.model.norms import RMSNorm
from hagi.model.ternary import BitLinear


@dataclass
class HebbianFFNConfig:
    """HebbianBilinearFFN parameters."""

    expansion: int = 4  # m = expansion * H
    dropout: float = 0.0


def _proj(in_f: int, out_f: int, use_ternary: bool) -> nn.Module:
    return BitLinear(in_f, out_f, bias=False) if use_ternary else nn.Linear(in_f, out_f, bias=False)


class HebbianBilinearFFN(nn.Module):
    """phi(h) = (A0 h) . silu(A1 h); output = W(phi) * (1 + tanh(gate)).

    Args:
        hidden_size: H.
        cfg: expansion and dropout.
        norm_eps: RMSNorm epsilon.
        use_ternary: ternarize A0/A1/W via BitLinear.
    """

    def __init__(self, hidden_size: int, cfg: HebbianFFNConfig, norm_eps: float = 1e-6, use_ternary: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        m = cfg.expansion * hidden_size
        self.m = m
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.A0 = _proj(hidden_size, m, use_ternary)
        self.A1 = _proj(hidden_size, m, use_ternary)
        self.W = _proj(m, hidden_size, use_ternary)
        if isinstance(self.W, BitLinear):
            nn.init.normal_(self.W.weight, std=1.0 / (m ** 0.5))
        else:
            assert isinstance(self.W, nn.Linear)
            nn.init.normal_(self.W.weight, std=1.0 / (m ** 0.5))
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        phi = self.A0(h) * F.silu(self.A1(h))
        phi = self.dropout(phi)
        return x + self.W(phi) * (1.0 + torch.tanh(self.gate))
