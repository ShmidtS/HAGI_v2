"""Clifford cross-modal mixing — rotor convolution in frequency domain.

Unifies two ideas:
  1. Cross-spectrum (MIMO channel estimation analog): A_f * conj(B_f)
  2. Clifford geometric product: full multivector product capturing
     scalar correlation + bivector wedge + trivector triple correlation

Key insight: FFT of multivector-valued signal gives complex coefficients
per blade. The geometric product of A_f with conj(B_f) in frequency domain
produces a frequency-resolved cross-correlation with Clifford structure:

  - Grade 0 (scalar blade): cross-spectrum power (what CrossModalFreqMix computed)
  - Grade 1 (vectors): directional correlation between modalities
  - Grade 2 (bivectors): wedge product — rotational coupling between modalities
  - Grade 3 (trivector): triple correlation — volumetric dependence

Rotor sandwich R * X * reverse(R) rotates the cross-product in Clifford space.
This IS the MIMO channel matrix, but geometrically structured: instead of an
arbitrary complex weight, the coupling is constrained to rotations (unitary
transformations preserving norm), which is more physically meaningful.

Cl(3,0,0) has 8 blades, so hidden_size must be divisible by 8.
Each "head" is one 8-component multivector.

Information theory:
  - Geometric product = full correlation structure between modalities
  - Rotor = optimal rotation aligning modality subspaces
  - Frequency domain = frequency-resolved cross-correlation
  - Gated residual = adaptive activation (starts near zero)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.algebra.clifford import geometric_product, reverse_mv
from hagi_v4.model.norms import RMSNorm

BLADE_COUNT = 8


class CliffordCrossModal(nn.Module):
    """Clifford rotor cross-modal mixing in frequency domain.

    For each modality pair (i, j):
      1. Extract hidden states for each modality
      2. Reshape to multivectors [B, T, n_heads, 8]
      3. FFT per blade along sequence dimension
      4. Cross-geometric-product: GP(A_f, conj(B_f))
      5. Rotor sandwich: R * cross_gp * reverse(R)
      6. Bandlimit (keep low-frequency bins)
      7. IFFT back, project, gate, scatter to residual
    """

    def __init__(
        self,
        hidden_size: int,
        gate_init: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert hidden_size % BLADE_COUNT == 0, (
            f"hidden_size ({hidden_size}) must be divisible by BLADE_COUNT ({BLADE_COUNT}) for Clifford algebra"
        )
        self.hidden_size = hidden_size
        self.n_heads = hidden_size // BLADE_COUNT

        self.norm = RMSNorm(hidden_size, eps=norm_eps)

        n_pairs_max = 3
        rotors = torch.zeros(n_pairs_max, self.n_heads, BLADE_COUNT)
        rotors[:, :, 0] = 1.0
        self.rotors = nn.Parameter(rotors)

        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def _rotor_sandwich(self, x: torch.Tensor, pair_idx: int) -> torch.Tensor:
        """Apply rotor sandwich R * x * reverse(R) in Clifford algebra.

        Args:
            x: [..., 8] complex or real multivector coefficients.
            pair_idx: which rotor to use.

        Returns:
            [..., 8] rotated multivector.
        """
        R = self.rotors[pair_idx].unsqueeze(0)
        Rx = geometric_product(R.expand(x.shape[:-1] + (BLADE_COUNT,)), x)
        R_rev = reverse_mv(R).expand(x.shape[:-1] + (BLADE_COUNT,))
        return geometric_product(Rx, R_rev)

    def _cross_modal_pair(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        pair_idx: int,
    ) -> torch.Tensor:
        """Compute cross-modal Clifford product for one modality pair.

        Args:
            h_i: [B, T_i, H] hidden states of modality i.
            h_j: [B, T_j, H] hidden states of modality j.
            pair_idx: rotor index.

        Returns:
            [B, T_min, H] cross-modal residual.
        """
        B, T_i, H = h_i.shape
        T_j = h_j.shape[1]
        T_min = min(T_i, T_j)

        h_i = h_i[:, :T_min]
        h_j = h_j[:, :T_min]

        mv_i = h_i.reshape(B, T_min, self.n_heads, BLADE_COUNT)
        mv_j = h_j.reshape(B, T_min, self.n_heads, BLADE_COUNT)

        A_f = torch.fft.rfft(mv_i.float(), dim=1)
        B_f = torch.fft.rfft(mv_j.float(), dim=1)
        B_f_conj = torch.conj(B_f)

        n_freq = A_f.shape[1]
        k_freq = max(1, n_freq // 2)

        cross_gp = geometric_product(A_f[:, :k_freq], B_f_conj[:, :k_freq])

        cross_rotated = self._rotor_sandwich(cross_gp, pair_idx)

        out_f = torch.zeros_like(A_f)
        out_f[:, :k_freq] = cross_rotated

        x_out = torch.fft.irfft(out_f, n=T_min, dim=1).to(h_i.dtype)
        return x_out.reshape(B, T_min, H)

    def forward(
        self,
        h: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        num_modalities: int = 3,
    ) -> torch.Tensor:
        """Apply cross-modal Clifford mixing.

        Args:
            h: [B, T, H] concatenated modality hidden states.
            modality_ids: [B, T] modality index per position (0=text, 1=image, ...).
            num_modalities: number of modalities present.

        Returns:
            [B, T, H] hidden states with cross-modal residual added.
        """
        if modality_ids is None:
            return h

        B, T, H = h.shape
        gate = torch.sigmoid(self.gate)
        cross_residual = h.new_zeros(B, T, H)

        pair_idx = 0
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                mask_i = modality_ids == i
                mask_j = modality_ids == j
                if not (mask_i.any() and mask_j.any()):
                    pair_idx += 1
                    continue

                for b in range(B):
                    idx_i = torch.where(mask_i[b])[0]
                    idx_j = torch.where(mask_j[b])[0]
                    if len(idx_i) == 0 or len(idx_j) == 0:
                        continue

                    h_i = h[b : b + 1, idx_i]
                    h_j = h[b : b + 1, idx_j]

                    cross_ij = self._cross_modal_pair(h_i, h_j, pair_idx)
                    cross_ji = self._cross_modal_pair(h_j, h_i, pair_idx)

                    n_min_i = min(len(idx_i), cross_ij.shape[1])
                    n_min_j = min(len(idx_j), cross_ji.shape[1])
                    if n_min_i > 0:
                        cross_proj_i = self.norm(self.proj(cross_ij[0, :n_min_i]))
                        cross_residual[b, idx_i[:n_min_i]] += cross_proj_i
                    if n_min_j > 0:
                        cross_proj_j = self.norm(self.proj(cross_ji[0, :n_min_j]))
                        cross_residual[b, idx_j[:n_min_j]] += cross_proj_j

                pair_idx += 1

        return h + gate * cross_residual
