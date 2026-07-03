"""Coherence head — geometric coherence between adjacent predictions.

V4 doesn't use CAST for K-token prediction (that's autoregressive). Instead:
- Simple output: lm_head(final_norm(h)) -> [B, T, V] logits for ALL positions
- Coherence: geometric product between adjacent h positions -> regularization loss
- coherence_loss = mean(||geometric_product(h[t], h[t+1])||^2) — encourages
  smooth predictions across the temporal plane.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.algebra.clifford import BLADE_COUNT, geometric_product


class CoherenceHead(nn.Module):
    """Geometric coherence regularizer + output projection for V4.

    Computes:
    1. Logits for all positions via lm_head(final_norm(h)) -> [B, T, V]
    2. Coherence loss: mean squared norm of geometric product between
       adjacent positions h[t] and h[t+1]. This encourages smooth
       predictions across the temporal plane.
    """

    def __init__(self, hidden_size: int = 576, gate_init: float = -5.0):
        super().__init__()
        self.n_heads = hidden_size // BLADE_COUNT
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def coherence_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Geometric coherence loss between adjacent positions.

        coherence_loss = mean(||geometric_product(h[t], h[t+1])||^2)

        The bivector "area" between adjacent positions measures
        relational smoothness. Minimizing this encourages coherent
        predictions across the temporal plane.
        """
        B, T, H = h.shape
        if T < 2:
            return h.new_zeros(())
        mv = h.reshape(B, T, self.n_heads, BLADE_COUNT)
        area = geometric_product(mv[:, :-1], mv[:, 1:])
        gate = torch.sigmoid(self.gate_logit)
        return gate * (area.float() ** 2).mean()

    def forward(
        self,
        h: torch.Tensor,
        lm_head_weight: torch.Tensor,
        final_norm: nn.Module,
    ) -> torch.Tensor:
        """Compute logits for all positions.

        Returns: [B, T, V] logits for the full plane.
        """
        h_normed = final_norm(h)
        return F.linear(h_normed, lm_head_weight)
