"""Channel Quality Indicator (CQI) — adaptive transmit/receive parameters.

5G NR analog: UE measures CQI and feeds back to base station.
Base station adapts modulation (QPSK->256QAM), coding rate, MIMO mode.

In HAGI, CQI is computed from the hidden state and controls:
  - Frequency gate width (adaptive modulation: wide gates = high-order QAM)
  - Kalman process noise Q (adaptive coding: high Q = conservative decoding)
  - MSA DFE taps (adaptive equalizer: more taps for bad channel)
  - Mask ratio (adaptive erasure: mask less when channel is bad)

CQI = sigmoid(Linear(h)) — per-position channel quality in [0, 1].
High CQI = strong signal, clean channel -> aggressive parameters.
Low CQI = weak signal, noisy channel -> conservative parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CQIEstimator(nn.Module):
    """Estimate per-position channel quality from hidden state.

    Cheap: one Linear(H, 1) + sigmoid. O(H) params.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.proj.weight, std=1.0 / (hidden_size**0.5))
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute CQI from hidden state.

        Args:
            h: [B, T, H] hidden state (before or after bottleneck)

        Returns:
            cqi: [B, T] in [0, 1]. 1.0 = good channel, 0.0 = bad channel.
        """
        return torch.sigmoid(self.proj(h).squeeze(-1))
