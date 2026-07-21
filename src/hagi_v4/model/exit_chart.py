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
        """Compute convergence proxy via norm ratio.

        Returns the ratio ||ext_after|| / ||ext_before||. When this drops below
        threshold, iterations add little magnitude -> halt. This avoids the
        cosine-similarity MI fiction and removes the .item() GPU sync.

        Args:
            ext_before: extrinsic info from previous iteration [B, T, C].
            ext_after: extrinsic info from current iteration [B, T, C].

        Returns:
            novelty: scalar tensor, ratio in [0, inf). Lower = more converged.
        """
        eb = ext_before.float().reshape(-1)
        ea = ext_after.float().reshape(-1)
        return (ea.norm() / (eb.norm() + 1e-8)).clamp(0.0, 1e6)

    def should_halt(
        self,
        ext_before: torch.Tensor,
        ext_after: torch.Tensor,
        iteration: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if decoder should halt based on norm-ratio novelty. No GPU sync."""
        if iteration < self.min_iterations:
            mi = torch.tensor(0.0, device=ext_after.device)
            return torch.tensor(False, device=ext_after.device), mi

        mi = self.compute_mi(ext_before, ext_after)
        should_halt = mi < self.threshold
        return should_halt, mi
