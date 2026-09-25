"""Opt-in finite-option decision head for causal HAGI states."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.model.adaptive import AdaptiveComponent


class DecisionHead(AdaptiveComponent):
    """Project one final causal hidden state to fixed decision options.

    The head is observational: it never feeds its logits back into the LM
    path. Its weight is an FP32 master and is cast to the activation dtype
    for the matmul, matching the training-side precision policy used by
    HAGI's protected scalar gains. The zero initialization consumes no RNG,
    so enabling the head does not perturb the paired baseline's later
    ``out_norm``/LM-head initialization.
    """

    def __init__(self, hidden_size: int, num_options: int) -> None:
        super().__init__()
        if hidden_size < 1 or num_options < 2:
            raise ValueError("decision head requires hidden_size >= 1 and num_options >= 2")
        self.hidden_size = int(hidden_size)
        self.num_options = int(num_options)
        self.weight = nn.Parameter(torch.zeros(self.num_options, self.hidden_size))
        self.keep_fp32 = True

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map ``[B, H]`` to ``[B, num_options]`` without mutating hidden."""
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"decision hidden must be [B, {self.hidden_size}], got {tuple(hidden.shape)}"
            )
        return F.linear(hidden, self.weight.to(hidden.dtype))

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_options={self.num_options}"
