"""EXIT chart estimator — convergence criterion for iterative decoding.

EXIT (Extrinsic Information Transfer) chart tracks the mutual information
between consecutive extrinsic updates in the iterative decoder.

When MI(ext_before, ext_after) -> 0, further iterations add no new
information -> convergence achieved -> halt.

5G NR analog: EXIT chart stopping criterion for LDPC/turbo decoders.

Implementation:
  MI = 1 - H(sigmoid(|mean(ext_delta)|))
  H(p) = -p*log(p) - (1-p)*log(1-p)  (binary entropy)

  When ext_delta is small -> p ~ 0.5 -> H(p) ~ 1 -> MI ~ 0 -> converged.
  When ext_delta is large -> p ~ 0 or 1 -> H(p) ~ 0 -> MI ~ 1 -> not converged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EXITChartEstimator(nn.Module):
    """Estimate extrinsic information transfer for convergence detection.

    Args:
        threshold: MI threshold below which we declare convergence.
        min_iterations: minimum iterations before convergence check active.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        min_iterations: int = 1,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_iterations = min_iterations

    def compute_mi(self, ext_before: torch.Tensor, ext_after: torch.Tensor) -> torch.Tensor:
        """Compute approximate mutual information between extrinsic updates.

        Args:
            ext_before: extrinsic info from previous iteration [B, T, C].
            ext_after: extrinsic info from current iteration [B, T, C].

        Returns:
            mi: scalar tensor, MI estimate in [0, 1].
                0 = no new information (converged).
                1 = maximum new information (not converged).
        """
        ext_delta = (ext_after - ext_before).float()
        delta_mag = ext_delta.abs().mean(dim=-1)  # [B, T]

        p = torch.sigmoid(delta_mag)

        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1.0 - eps)

        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        mi = 1.0 - entropy.mean()

        return mi

    def should_halt(
        self,
        ext_before: torch.Tensor,
        ext_after: torch.Tensor,
        iteration: int,
    ) -> tuple[bool, torch.Tensor]:
        """Check if decoder should halt based on EXIT chart."""
        if iteration < self.min_iterations:
            mi = torch.tensor(0.0, device=ext_after.device)
            return False, mi

        mi = self.compute_mi(ext_before, ext_after)
        should_halt = mi.item() < self.threshold
        return should_halt, mi
