"""Learned per-position uncertainty for iterative decoding.

V9: the variance estimator projects to a *scalar* per position rather than a
full ``[C]`` vector. The Kalman-form update only needs ``sigma2_pred`` and
``sigma2_meas`` as scalars (they are combined additively and inverted), so a
per-dimension variance carried no information that a scalar did not. This
cuts ``LearnedUncertainty`` from ``C*C`` to ``C+1`` parameters (~200K -> 449
for ``C=448``) with no loss of capacity in the update equations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedUncertainty(nn.Module):
    """Per-position scalar learned variance estimator.

    Maps hidden state to a scalar ``sigma^2 in [0, inf)`` via
    ``softplus(Linear(h))``. The scalar is broadcast across the channel
    dimension inside the inverse-variance update.

    Args:
        hidden_size: C (hidden dimension).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.log_var = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.log_var.weight, std=0.01)
        nn.init.zeros_(self.log_var.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Estimate per-position variance.

        Args:
            h: [B, T, C] hidden state.

        Returns:
            sigma2: [B, T, C] non-negative variance estimate, broadcast
                from the scalar projection.
        """
        scalar = F.softplus(self.log_var(h).squeeze(-1))  # [B, T]
        return scalar.unsqueeze(-1).expand_as(h)


def inverse_variance_update(
    z_pred: torch.Tensor,
    innovation: torch.Tensor,
    sigma2_pred: torch.Tensor,
    sigma2_meas: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bayes-optimal update under Gaussian assumption with learned variances.

    Equivalent to Kalman gain K = P/(P+R), but P and R are learned per-position
    rather than tracked via random-walk dynamics.

    Args:
        z_pred: [B, T, C] predicted state.
        innovation: [B, T, C] measurement residual (back-projected parity).
        sigma2_pred: [B, T, C] predicted state variance (from LearnedUncertainty).
        sigma2_meas: [B, T, C] measurement variance (from parity residual).
        eps: numerical stability.

    Returns:
        z_corrected: [B, T, C] uncertainty-weighted blend.
        k: [B, T, C] Kalman-like gain in [0, 1].
    """
    k = sigma2_pred / (sigma2_pred + sigma2_meas + eps)
    z_corrected = z_pred + k * innovation
    return z_corrected, k
