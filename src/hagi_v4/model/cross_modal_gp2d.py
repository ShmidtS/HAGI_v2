"""Cross-modal GP2D — Multiple Description Coding parity (V7).

Creates parity between modality streams:
  - Intra-modality parity (existing MultiScaleGP2D): text<->text, image<->image
  - Cross-modality parity (NEW): text<->image, text<->audio, image<->audio

If one modality is dropped (erasure), cross-modal parity allows partial
recovery from remaining modalities. This is MDC: graceful degradation
under modality loss, no catastrophic failure.

Information theory:
  - Cross-modal GP = parity between descriptions (MDC)
  - Geometric product between text and image tokens = parity bits
  - If image erased, text cross-parity partially recovers image info
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.algebra.clifford import geometric_product
from hagi_v4.config import GP2DConfig
from hagi_v4.model.multiscale_gp2d import MultiScaleGP2D
from hagi_v4.model.norms import RMSNorm

BLADE_COUNT = 8


class CrossModalGP2D(nn.Module):
    """Cross-modal geometric product for multiple description coding."""

    def __init__(
        self,
        cfg: GP2DConfig,
        hidden_size: int,
        num_modalities: int = 3,
        gate_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.hidden_size = hidden_size

        self.intra_gp2d = MultiScaleGP2D(
            cfg,
            hidden_size,
            scales=cfg.multiscale_windows,
            gate_inits=cfg.multiscale_gate_inits,
            use_interleave=cfg.use_interleave,
        )

        n_pairs = num_modalities * (num_modalities - 1) // 2
        self.cross_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_pairs)])
        self.cross_gates = nn.ParameterList([nn.Parameter(torch.tensor(gate_init)) for _ in range(n_pairs)])
        self.cross_norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        h: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, intra_residual = self.intra_gp2d(h)

        if modality_ids is None:
            return h, intra_residual

        B, T, C = h.shape
        cross_residual = h.new_zeros(B, T, C)
        pair_idx = 0
        n_heads = C // BLADE_COUNT

        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                mask_i = modality_ids == i
                mask_j = modality_ids == j

                if not (mask_i.any() and mask_j.any()):
                    pair_idx += 1
                    continue

                h_i = h[mask_i]
                h_j = h[mask_j]

                mv_i = h_i.reshape(-1, n_heads, BLADE_COUNT)
                mv_j = h_j.reshape(-1, n_heads, BLADE_COUNT)

                n_min = min(mv_i.shape[0], mv_j.shape[0])
                cross_gp = geometric_product(mv_i[:n_min], mv_j[:n_min])
                cross_gp = cross_gp.reshape(n_min, C)

                cross_out = self.cross_proj[pair_idx](cross_gp)
                cross_out = self.cross_norm(cross_out)
                gate = torch.sigmoid(self.cross_gates[pair_idx])
                cross_out = gate * cross_out

                for b in range(B):
                    mask_b = modality_ids[b] == i
                    idx_b = torch.where(mask_b)[0]
                    n_b = min(idx_b.shape[0], n_min)
                    if n_b > 0:
                        cross_residual[b, idx_b[:n_b]] += cross_out[:n_b]

                pair_idx += 1

        total_residual = intra_residual + cross_residual
        return h + cross_residual, total_residual
