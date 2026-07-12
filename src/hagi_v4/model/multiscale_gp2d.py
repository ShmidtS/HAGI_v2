"""Multi-Scale Geometric Product 2D — LDPC-like multi-degree parity.

Shannon analogy: LDPC codes use a sparse parity-check matrix with
variable check node degrees. Different check nodes verify different
subsets of the codeword, providing multi-scale error correction.

V5's GP2D uses a single scale (window=1: positions t-1, t, t+1).
This is like a single-degree parity check — only catches local errors.

V6's MultiScaleGP2D uses multiple scales (window=1, 4, 16):
  Scale 1 (window=1): adjacent parity — catches local errors (high-freq)
  Scale 2 (window=4): mid-range parity — catches burst errors (mid-freq)
  Scale 3 (window=16): long-range parity — catches structural errors (low-freq)

Interleaving between scales provides burst error protection (like
interleaving in turbo/LDPC codes): errors correlated at one scale
become decorrelated at another scale after permutation.

Each scale has its own learnable gate, initialized low so the model
learns to progressively activate longer-range parity as training proceeds.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi_v4.config import GP2DConfig
from hagi_v4.model.gp2d import GeometricProduct2D
from hagi_v4.model.norms import RMSNorm


class MultiScaleGP2D(nn.Module):
    """Multi-scale geometric product with interleaving for burst error protection.

    Combines multiple GeometricProduct2D layers at different window sizes.
    Each scale contributes parity at a different frequency band.
    Scale weights are learnable and initialized to favor local parity.
    """

    def __init__(
        self,
        cfg: GP2DConfig,
        hidden_size: int = 288,
        scales: tuple[int, ...] = (1, 4, 16),
        gate_inits: tuple[float, ...] = (-2.0, -3.0, -4.0),
        use_interleave: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.scales = scales
        self.use_interleave = use_interleave

        self.gp_layers = nn.ModuleList()
        for window, gate_init in zip(scales, gate_inits):
            scale_cfg = GP2DConfig(
                window=window,
                gate_init=gate_init,
                use_whiteness_loss=cfg.use_whiteness_loss,
                whiteness_weight=cfg.whiteness_weight,
                use_systematic_parity=cfg.use_systematic_parity,
                parity_weight=cfg.parity_weight,
            )
            self.gp_layers.append(GeometricProduct2D(scale_cfg, hidden_size))

        self.scale_gates = nn.Parameter(torch.zeros(len(scales)))
        self.fusion_norm = RMSNorm(hidden_size, eps=1e-6)
        self.fusion_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def _interleave(self, h: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Permute positions for burst error protection.

        Different scales use different permutation patterns, so burst
        errors that correlate at one scale become decorrelated at another.
        """
        if not self.use_interleave or scale_idx == 0:
            return h
        B, T, H = h.shape
        shift = self.scales[scale_idx]
        return torch.roll(h, shifts=shift, dims=1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-scale parity computation.

        Returns:
            h_out: [B, T, H] hidden state with multi-scale parity added.
            residual: [B, T, H] combined parity residual for tracking.
        """
        residuals = []

        for idx, gp in enumerate(self.gp_layers):
            h_interleaved = self._interleave(h, idx)
            h_out_i, residual_i = gp(h_interleaved)
            if self.use_interleave and idx > 0:
                residual_i = self._interleave(residual_i, -idx)
            residuals.append(residual_i)

        scale_gates = torch.sigmoid(self.scale_gates)
        combined_residual = sum(g * r for g, r in zip(scale_gates, residuals))

        fused = self.fusion_norm(self.fusion_proj(combined_residual))
        gate = torch.sigmoid(self.fusion_gate)
        final_residual = gate * fused

        return h + final_residual, final_residual
