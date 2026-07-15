"""Kalman filter for iterative decoding — optimal state estimation.

5G analog: channel estimation + equalization in iterative receivers.
The Kalman filter optimally blends prediction (FreqBlock) with
measurement (GP2D parity) based on tracked uncertainty.

Q (process noise) and R (measurement noise) are learnable parameters
initialized to zero — the model discovers optimal noise levels through
training. Kalman gain K = P/(P+R) naturally bounds the measurement
contribution in [0,1] without any clipping or parametrization.

Position in turbo loop:
  1. Predict: z_pred = FreqBlock(z_prev)
  2. Kalman predict: P_pred = P_prev + Q
  3. Measure: z_meas = GP2D(z_pred)
  4. Kalman update: K = P_pred / (P_pred + R)
     z = z_pred + K * (z_meas - z_pred)
     P = (1 - K) * P_pred

Diagonal covariance (O(C) per iteration, negligible overhead):
  P: [C] per-dimension variance
  Q: [C] process noise (learnable, zero-init)
  R: [C] measurement noise (learnable, zero-init)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    """Diagonal-covariance Kalman filter for hidden state estimation.

    Tracks per-dimension uncertainty P across turbo iterations.
    Blends prediction (Component A) with measurement (Component B)
    using optimal Kalman gain. Q and R are learnable, zero-initialized —
    the model learns optimal noise levels naturally.

    Args:
        dim: hidden dimension C
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.q_logit = nn.Parameter(torch.full((dim,), -2.0))
        self.r_logit = nn.Parameter(torch.full((dim,), -2.0))

    def _q(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.sigmoid(self.q_logit).to(dtype)

    def _r(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.sigmoid(self.r_logit).to(dtype)

    def predict(self, p_prev: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        if q is None:
            q = self._q(p_prev.dtype)
        if p_prev.dim() == 3:
            result = p_prev + q.unsqueeze(0).unsqueeze(0)
        else:
            result = p_prev + q
        if not torch.isfinite(result).all():
            raise FloatingPointError("kalman stage=predict nonfinite state")
        return result

    def update(
        self,
        z_pred: torch.Tensor,
        innovation: torch.Tensor,
        p_pred: torch.Tensor,
        r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if r is None:
            r = self._r(p_pred.dtype)
        if p_pred.dim() == 3:
            r = r.unsqueeze(0).unsqueeze(0)
        k = (p_pred.float() / (p_pred.float() + r.float())).to(z_pred.dtype)
        z_corrected = z_pred + k * innovation
        p_corrected = (1 - k) * p_pred
        if not torch.isfinite(z_corrected).all() or not torch.isfinite(p_corrected).all():
            raise FloatingPointError("kalman stage=update nonfinite state")
        return z_corrected, p_corrected
