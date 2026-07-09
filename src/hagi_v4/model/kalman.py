"""Kalman filter for iterative decoding — optimal state estimation.

5G analog: channel estimation + equalization in iterative receivers.
The Kalman filter optimally blends prediction (FreqBlock) with
measurement (GP2D parity) based on tracked uncertainty.

Position in turbo loop:
  1. Predict: z_pred = FreqBlock(z_prev)
  2. Kalman predict: P_pred = P_prev + Q
  3. Measure: z_meas = GP2D(z_pred)
  4. Kalman update: K = P_pred / (P_pred + R)
     z = z_pred + K * (z_meas - z_pred)
     P = (1 - K) * P_pred

Diagonal covariance (O(C) per iteration, negligible overhead):
  P: [C] per-dimension variance
  Q: [C] process noise (learnable)
  R: [C] measurement noise (learnable)

Kalman gain adapts: high uncertainty → trust measurement (parity).
Low uncertainty → trust prediction (reasoning). This is optimal
Bayesian estimation, replacing fixed alpha extrinsic exchange.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    """Diagonal-covariance Kalman filter for hidden state estimation.

    Tracks per-dimension uncertainty P across turbo iterations.
    Blends prediction (Component A) with measurement (Component B)
    using optimal Kalman gain.

    Args:
        dim: hidden dimension C
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        # Process noise (uncertainty growth per iteration)
        self.log_q = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.log_q, mean=-4.0, std=0.5)

        # Measurement noise (GP2D parity reliability)
        self.log_r = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.log_r, mean=-2.0, std=0.5)

    def predict(self, p_prev: torch.Tensor) -> torch.Tensor:
        """Prediction step: uncertainty grows by process noise Q.

        P_pred = P_prev + Q
        """
        q = torch.exp(self.log_q).to(p_prev.dtype)
        return p_prev + q

    def update(
        self,
        z_pred: torch.Tensor,
        z_meas: torch.Tensor,
        p_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measurement update: optimal Bayesian blend.

        K = P_pred / (P_pred + R)
        z = z_pred + K * (z_meas - z_pred)
        P = (1 - K) * P_pred

        Args:
            z_pred: [B, T, C] predicted state (from FreqBlock)
            z_meas: [B, T, C] measured state (from GP2D)
            p_pred: [C] predicted diagonal covariance

        Returns:
            z_corrected: [B, T, C] optimally estimated state
            p_corrected: [C] updated diagonal covariance
        """
        r = torch.exp(self.log_r).to(p_pred.dtype)

        # Kalman gain (diagonal, broadcast over B,T)
        k = p_pred / (p_pred + r + 1e-8)

        # Innovation
        innovation = z_meas - z_pred

        # Optimal blend
        z_corrected = z_pred + k.unsqueeze(0).unsqueeze(0) * innovation

        # Updated covariance
        p_corrected = (1 - k) * p_pred

        return z_corrected, p_corrected
