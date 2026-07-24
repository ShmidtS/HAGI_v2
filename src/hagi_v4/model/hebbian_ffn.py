"""HebbianBilinearFFN (§4.2) — trainable bilinear feed-forward block.

Self-contained additive module. Derived from HazyResearch/hebbian-mlps
(arXiv 2607.10034, COLM 2026). The paper's bilinear feature map
φ(x) = (A0 x) ⊙ (A1 x) has an information-theoretically optimal capacity
structure for associative memory. This version makes A0, A1, W trainable
Parameters, giving an SwiGLU-shaped FFN with the bilinear capacity geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hagi_v4.model.norms import RMSNorm


@dataclass
class HebbianFFNConfig:
    """HebbianBilinearFFN parameters (§4.2)."""

    expansion: int = 4  # m = expansion * H  (paper default m = 4·d)
    dropout: float = 0.0


class HebbianBilinearFFN(nn.Module):
    """φ(h) = (A0 h) ⊙ silu(A1 h); output = W(φ) · (1 + tanh(gate)).

    Args:
        hidden_size: H.
        cfg: expansion and dropout.
    """

    def __init__(self, hidden_size: int, cfg: HebbianFFNConfig, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        m = cfg.expansion * hidden_size
        self.m = m
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        # Bilinear feature map: A0, A1 ∈ R^{m×H}. Trainable.
        self.A0 = nn.Linear(hidden_size, m, bias=False)
        self.A1 = nn.Linear(hidden_size, m, bias=False)
        # Readout W ∈ R^{H×m}.
        self.W = nn.Linear(m, hidden_size, bias=False)
        # LayerScale gate as additive modulation, NOT a hard multiplier:
        # a zero-init multiplier (tanh(gate)) would zero the whole FFN branch
        # and starve A0/A1/W of gradient (V18 dead-gradient class). (1+tanh(g))
        # is 1 at g=0 (FFN live), letting the model learn per-channel suppression.
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.W.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        phi = self.A0(h) * torch.nn.functional.silu(self.A1(h))
        phi = self.dropout(phi)
        return x + self.W(phi) * (1.0 + torch.tanh(self.gate))


def construct_bilinear_warm_start(
    A0_weight: torch.Tensor,
    A1_weight: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Closed-form readout W from a factset (optional warm start).

    Mirrors hebbian-mlps `full_ridge_readout`: given fixed bilinear feature
    matrices A0, A1 and a (key, value) factset, solve the ridge-regularized
    readout W = (ΦᵀΦ/n + λI)⁻¹ (VᵀΦ/n)ᵀ. Returns W ∈ R^{H×m}.

    OFF by default — from-scratch LM training has no pre-extracted factset.
    """
    with torch.no_grad():
        features = (keys @ A0_weight.t()) * (keys @ A1_weight.t())  # [n, m]
        n = features.shape[0]
        covariance = (features.t() @ features) / n  # [m, m]
        correlation = (values.t() @ features) / n  # [H, m]
        m = covariance.shape[0]
        reg = covariance + ridge * torch.eye(m, device=covariance.device, dtype=covariance.dtype)
        W = torch.linalg.solve(reg, correlation.t()).t()  # [H, m]
    return W
