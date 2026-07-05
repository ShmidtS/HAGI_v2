"""Geometric enricher — O(T) Clifford self-product enrichment for attention.

Enriches hidden state with Cl(3,0,0) geometric structure BEFORE standard
GQA/SDPA. No T x T intermediates — FlashAttention preserved.

Pipeline:
    x -> to_mv -> [B, T, n_mv, 8]           O(T)
       -> self-geometric-product (g0, g2)    O(T)
       -> learnable grade mixing              O(T)
       -> from_mv -> geo_feat                 O(T)
    output = x + sigmoid(gate) * geo_feat     O(T)

The enriched hidden state flows into QKV projections, so Q and K carry
geometric structure. Standard dot product Q . K then implicitly captures
geometric relationships through cross-terms between original and
self-product (bivector) components.

Grade weights w0..w3 are learnable per-layer:
    w0: self-product scalar   (confidence)
    w1: original vector       (entities)
    w2: self-product bivector (relations)
    w3: original trivector    (higher-order)

Init [1, 0, 0, 0] — scalar-only start. Model learns which grades matter.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi_v4.algebra.clifford import BLADE_COUNT, geometric_product_self_g02, grade_projection


class GeometricEnricher(nn.Module):
    """O(T) geometric enrichment via Cl(3,0,0) self-product + grade mixing.

    Applied before standard attention. Does NOT modify attention scores
    directly (no T x T custom scores). Instead enriches Q/K/V input with
    geometric structure from self-geometric product.
    """

    def __init__(
        self,
        hidden_size: int = 576,
        n_mv: int = 8,
        gate_init: float = -2.0,
        grade_weights_init: tuple = (1.0, 0.0, 0.0, 0.0),
    ):
        super().__init__()
        self.n_mv = n_mv
        geo_dim = n_mv * BLADE_COUNT

        self.to_mv = nn.Linear(hidden_size, geo_dim, bias=False)
        self.from_mv = nn.Linear(geo_dim, hidden_size, bias=False)

        self.grade_weights = nn.Parameter(torch.tensor(list(grade_weights_init), dtype=torch.float32))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        h_mv = self.to_mv(x).view(B, T, self.n_mv, BLADE_COUNT)

        g0, g2 = geometric_product_self_g02(h_mv)

        w = self.grade_weights.to(x.dtype)
        mixed = w[0] * g0 + w[1] * grade_projection(h_mv, 1) + w[2] * g2 + w[3] * grade_projection(h_mv, 3)

        geo_feat = self.from_mv(mixed.reshape(B, T, -1))

        return x + torch.sigmoid(self.gate) * geo_feat
