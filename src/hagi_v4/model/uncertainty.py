"""Learned per-position uncertainty for iterative decoding.

Replaces the Kalman filter's assumed-variance model with a learned,
per-position, per-dimension variance estimator. The update step uses
inverse-variance weighting (Bayes-optimal for Gaussian), which is the
same mathematical structure as the Kalman gain but with learned rather
than assumed variances.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedUncertainty(nn.Module):
    """Per-position, per-dimension learned variance estimator.

    Maps hidden state to sigma^2 in [0, inf) via softplus(Linear(h)).
    Unlike the Kalman filter's global Q/R, this adapts to each position
    and each dimension independently.

    Args:
        hidden_size: C (hidden dimension).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.log_var = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.log_var.weight, std=0.01)
        nn.init.zeros_(self.log_var.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Estimate per-position variance.

        Args:
            h: [B, T, C] hidden state.

        Returns:
            sigma2: [B, T, C] non-negative variance estimate.
        """
        return F.softplus(self.log_var(h))


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
