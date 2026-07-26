"""Water-filling capacity allocator — optimal per-expert intermediate width.

Rehabilitated from the V6-era ``water_filling.py`` (deleted in the V8->V25
collapse) and re-placed at the MoE layer. The original idea was the genuine
Shannon water-filling theorem:

    P_i = max(0, mu - 1/SNR_i)   s.t.   sum(P_i) = P_total

Power (here: intermediate width / capacity) follows SNR: high-SNR channels get
more capacity, low-SNR channels get less (or zero). The "water level" mu is set
so the total-power constraint holds.

This allocator maps the theorem to MoE *expert capacity*: each expert's
intermediate width is a soft allocation driven by a running estimate of its
per-token residual variance (1/variance ~ SNR). Heavier-load experts grow
wider. Allocation is differentiable through a temperature-controlled softmax
with a min-floor per expert, and an entropy regularizer prevents collapse to a
single expert (the dual of the water-filling sum-power constraint).

OPT-IN. Default MoE uses homogeneous expert widths; this allocator is enabled
only for the large multimodal config where architecture-level water-filling
matters. It carries NO in-path state on the main LM logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaterFillingAllocator(nn.Module):
    """Learnable per-expert capacity allocation = architecture-level water-filling.

    Maintains learnable allocation logits that softmax to a probability
    distribution over experts; the per-expert intermediate width is
    ``max(min_width, round(total_width * p_i))``. A variance EMA (detached,
    updated by the caller from the SNR signal) steers the logits via a
    ``_snr_logit`` addend so high-SNR experts attract capacity. An entropy
    regularizer encourages balanced use (capacity maximization across the
    parallel expert channels).

    Args:
        total_width: sum of expert intermediate widths (= num_experts * base).
        num_experts: number of routed experts.
        min_width: minimum per-expert width floor (no channel starves).
        temperature: softmax temperature (lower -> sharper allocation).
        snr_weight: weight of the SNR logit addend driving capacity toward
            high-SNR experts.
    """

    def __init__(
        self,
        total_width: int,
        num_experts: int,
        min_width: int = 64,
        temperature: float = 1.0,
        snr_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if total_width < num_experts * min_width:
            raise ValueError(
                f"total_width ({total_width}) must be >= num_experts*min_width "
                f"({num_experts * min_width})"
            )
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.total_width = int(total_width)
        self.num_experts = int(num_experts)
        self.min_width = int(min_width)
        self.temperature = float(temperature)
        self.snr_weight = float(snr_weight)
        self.allocation_logits = nn.Parameter(torch.zeros(num_experts))
        # Per-expert SNR proxy EMA (1/variance). Detached signal, not trained.
        self.register_buffer("snr_ema", torch.ones(num_experts))

    @torch.no_grad()
    def update_snr_ema(self, per_expert_residual_var: torch.Tensor, decay: float = 0.99) -> None:
        """Update the per-expert SNR EMA from residual variance.

        SNR_i ~ 1 / variance_i. High variance (noisy channel) -> low SNR -> less
        capacity. The EMA is detached; it steers allocation without gradient.

        Args:
            per_expert_residual_var: ``[num_experts]`` measured per-expert
                residual variance (the caller computes this from dispatched
                tokens). Detached.
            decay: EMA decay in [0, 1).
        """
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        var = per_expert_residual_var.detach().to(self.snr_ema).clamp_min(1e-6)
        snr = 1.0 / var
        self.snr_ema.mul_(decay).add_(snr, alpha=1.0 - decay)

    def allocation_probs(self) -> torch.Tensor:
        """Soft allocation probabilities over experts ``[num_experts]``.

        The SNR EMA steers the softmax: higher-SNR experts get larger logits
        (more capacity). Entropy-regularized in :meth:`regularization_loss`.
        """
        snr_logit = self.snr_weight * (self.snr_ema.log() - self.snr_ema.log().mean())
        return F.softmax((self.allocation_logits + snr_logit) / self.temperature, dim=-1)

    def get_widths(self) -> list[int]:
        """Discrete per-expert widths summing to ``total_width``, each ``>= min_width``.

        Greedy redistribution: floor at ``min_width``, distribute the residual
        budget proportional to probabilities, then correct any rounding drift.
        """
        probs = self.allocation_probs().detach()
        floors = torch.full((self.num_experts,), self.min_width, dtype=torch.float)
        budget = float(self.total_width - self.min_width * self.num_experts)
        raw = floors + probs * budget
        widths = [int(round(w.item())) for w in raw]
        # Correct rounding drift toward the largest-deficit expert.
        while sum(widths) > self.total_width:
            i = max(range(self.num_experts), key=lambda k: widths[k] - self.min_width)
            if widths[i] > self.min_width:
                widths[i] -= 1
            else:
                break
        while sum(widths) < self.total_width:
            i = max(range(self.num_experts), key=lambda k: probs[k].item())
            widths[i] += 1
        return widths

    def regularization_loss(self) -> torch.Tensor:
        """Entropy gap: ``log(num_experts) - H(p)``. Encourages balanced use.

        Zero (free) at the uniform allocation; positive when capacity collapses
        to few experts. This is the dual of the water-filling sum-power
        constraint (maximize H over the simplex = spread capacity across channels).
        """
        probs = self.allocation_probs()
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=probs.device, dtype=probs.dtype))
        return max_entropy - entropy
