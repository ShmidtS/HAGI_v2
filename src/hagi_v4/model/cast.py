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
from hagi_v4.config import CASTConfig


class CoherenceHead(nn.Module):
    """Geometric coherence regularizer + output projection for V4.

    Computes:
    1. Logits for all positions via lm_head(final_norm(h)) -> [B, T, V]
    2. Coherence loss: mean squared norm of geometric product between
       adjacent positions h[t] and h[t+1]. This encourages smooth
       predictions across the temporal plane.
    3. Per-grade coherence: when enabled, only applies to scalar + vector
       grades (smooth signals), skipping bivector/trivector (can vary sharply).
    """

    def __init__(
        self,
        cfg: CASTConfig,
        hidden_size: int = 576,
        lm_head: nn.Module | None = None,
        final_norm: nn.Module | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_heads = hidden_size // BLADE_COUNT
        self.gate_logit = nn.Parameter(torch.tensor(float(cfg.coherence_gate_init)))
        self._scalar_dim = cfg.scalar_dim
        self._vector_dim = cfg.vector_dim
        self._sv_dim = cfg.scalar_dim + cfg.vector_dim
        self._sv_heads = max(1, self._sv_dim // BLADE_COUNT)
        # Ensure _sv_dim is divisible by BLADE_COUNT
        self._sv_dim = self._sv_heads * BLADE_COUNT
        if lm_head is not None:
            object.__setattr__(self, "_lm_head", lm_head)
        if final_norm is not None:
            object.__setattr__(self, "_final_norm", final_norm)

    def coherence_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Geometric coherence loss between adjacent positions.

        coherence_loss = mean(||geometric_product(h[t], h[t+1])||^2)

        When per_grade_coherence is enabled, only scalar + vector grades
        are used — bivector and trivector can legitimately vary sharply.
        """
        B, T, H = h.shape
        if T < 2:
            return h.new_zeros(())
        if self.cfg.use_per_grade_coherence and self._sv_dim < H:
            sv = h[..., : self._sv_dim]
            n_heads = self._sv_heads
            mv = sv.reshape(B, T, n_heads, BLADE_COUNT)
        else:
            mv = h.reshape(B, T, self.n_heads, BLADE_COUNT)
        area = geometric_product(mv[:, :-1], mv[:, 1:])
        gate = torch.sigmoid(self.gate_logit)
        return gate * (area.float() ** 2).mean()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute logits for all positions.

        Returns: [B, T, V] logits for the full plane.
        """
        h_normed = self._final_norm(h)
        return F.linear(h_normed, self._lm_head.weight)
