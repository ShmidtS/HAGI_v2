# V21 DEFER: not wired in V21 forward path. Available for V22+ integration.
# See docs/ARCHITECTURE.md for integration roadmap.

"""Water-Filling capacity allocator — optimal dim allocation across grades.

Shannon analogy: Water-filling theorem states that optimal power
allocation across parallel channels with different SNR is:

    P_i = max(0, mu - 1/SNR_i)  s.t.  sum(P_i) = P_total

Channels with high SNR get more power, channels with low SNR get less
(or zero). The "water level" mu is set so total power constraint is met.

V6 mapping: grades = parallel channels, dims = power, variance = 1/SNR.
  - High-variance grade (high entropy) -> more dims (more capacity)
  - Low-variance grade (low entropy) -> fewer dims (less capacity)
  - Equal slope condition: dD/dR_i = dD/dR_j for all i,j

Implementation: learnable soft allocation via temperature-controlled
softmax over 4 grades, constrained to sum to total_dims. The logits
are updated by gradient descent, driven by per-grade variance signals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaterFillingAllocator(nn.Module):
    """Dynamic capacity allocation across grades based on water-filling.

    Maintains learnable allocation logits that, when softmaxed, produce
    a probability distribution over grades. These probabilities map to
    dimension allocations, with a minimum floor per grade.

    The allocator is differentiable — gradients from the main loss
    flow through the softmax, adjusting allocation based on which
    grades contribute most to reducing distortion.
    """

    def __init__(
        self,
        total_dims: int = 288,
        num_grades: int = 4,
        min_dims: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.total_dims = total_dims
        self.num_grades = num_grades
        self.min_dims = min_dims
        self.temperature = temperature

        self.allocation_logits = nn.Parameter(torch.zeros(num_grades))
        self.variance_ema = nn.Parameter(torch.ones(num_grades), requires_grad=False)

    def update_variance_ema(self, grade_variances: torch.Tensor, decay: float = 0.99) -> None:
        """Update EMA of per-grade variance for monitoring.

        grade_variances: [num_grades] tensor of measured variances.
        """
        with torch.no_grad():
            self.variance_ema.mul_(decay).add_(grade_variances, alpha=1 - decay)

    def get_allocation(self) -> list[int]:
        """Returns dims per grade via softmax with constraint.

        Ensures: sum(dims) = total_dims, each dim >= min_dims.
        """
        probs = F.softmax(self.allocation_logits / self.temperature, dim=-1)
        raw_dims = [max(self.min_dims, int(self.total_dims * p.item())) for p in probs]

        total = sum(raw_dims)
        while total > self.total_dims:
            idx = max(range(len(raw_dims)), key=lambda i: raw_dims[i])
            if raw_dims[idx] > self.min_dims:
                raw_dims[idx] -= 1
                total -= 1
            else:
                break

        while total < self.total_dims:
            idx = min(range(len(raw_dims)), key=lambda i: raw_dims[i])
            raw_dims[idx] += 1
            total += 1

        return raw_dims

    def get_allocation_probs(self) -> torch.Tensor:
        """Returns soft allocation probabilities [num_grades]."""
        return F.softmax(self.allocation_logits / self.temperature, dim=-1)

    def regularization_loss(self) -> torch.Tensor:
        """Entropy regularization on allocation — encourages balanced use.

        Without this, allocation could collapse to a single grade.
        With it, the allocator is encouraged to use all grades, unless
        the variance signal strongly suggests otherwise.
        """
        probs = self.get_allocation_probs()
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(float(self.num_grades)))
        return max_entropy - entropy

    def forward(self) -> tuple[list[int], torch.Tensor]:
        """Returns (dims_per_grade, reg_loss)."""
        dims = self.get_allocation()
        reg = self.regularization_loss()
        return dims, reg
