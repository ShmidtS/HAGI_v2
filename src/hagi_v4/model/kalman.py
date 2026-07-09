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

    def predict(self, p_prev: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        if q is None:
            q = torch.exp(self.log_q).to(p_prev.dtype)
        if p_prev.dim() == 3:
            return p_prev + q.unsqueeze(0).unsqueeze(0)
        return p_prev + q

    def update(
        self,
        z_pred: torch.Tensor,
        z_meas: torch.Tensor,
        p_pred: torch.Tensor,
        r_scale: float = 1.0,
        r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if r is None:
            r = torch.exp(self.log_r).to(p_pred.dtype) * r_scale
        else:
            r = r * r_scale
        if p_pred.dim() == 3:
            r = r.unsqueeze(0).unsqueeze(0)
        k = p_pred / (p_pred + r + 1e-8)
        innovation = z_meas - z_pred
        z_corrected = z_pred + k * innovation
        p_corrected = (1 - k) * p_pred
        return z_corrected, p_corrected
