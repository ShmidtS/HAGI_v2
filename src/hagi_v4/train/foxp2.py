"""FOXP2 controller — synaptic plasticity regulator (training-only).

Biology:
  FOXP2 is a transcription factor that regulates synaptic plasticity
  in language-learning circuits. It doesn't compute signals — it
  controls HOW FAST synapses learn. Selective: some pathways learn
  faster, others slower.


ML mapping:
  FOXP2 = per-parameter-group learning rate modulator.
  Input: gradient statistics (norm, variance) + training progress.
  Output: per-group scaling gate in [0, 1].
  Applied: AFTER backward(), BEFORE optimizer.step().
  Training-only: not in model forward, not in inference.

5G analog:
  Adaptive coding rate — base station adjusts coding per-subchannel
  based on feedback (CQI), not on signal content. FOXP2 adjusts
  learning rate per-layer based on gradient feedback.

Update rule with FOXP2:
  Standard:  w_i <- w_i - lr * dL/dw_i
  FOXP2:     w_i <- w_i - lr * g_i * dL/dw_i
  where g_i = sigmoid(FOXP2(grad_stats_i, progress))

  This is y = f(w^T x + b) where:
    x = [grad_norm, grad_var, progress] (per-layer statistics)
    w, b = FOXP2 parameters (learnable)
    f = sigmoid (gate in [0, 1])
    y = g_i (per-layer plasticity gate)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FOXP2Controller(nn.Module):
    """Per-layer plasticity gate based on gradient statistics.

    Generates a scaling factor in [0, 1] for each parameter group,
    controlling how much of the gradient flows through to the update.
    Mr. Fox decides which advice (gradients) reaches the student (weights).

    Args:
        num_groups: number of parameter groups (e.g. layers)
        hidden: hidden dimension of controller network
    """

    def __init__(self, num_groups: int, hidden: int = 32) -> None:
        super().__init__()
        self.num_groups = num_groups

        self.proj = nn.Linear(3, hidden)
        self.out = nn.Linear(hidden, num_groups)

        self.gate_bias = nn.Parameter(torch.zeros(num_groups))

        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        grad_stats: torch.Tensor,
        progress: float,
    ) -> torch.Tensor:
        """Generate per-group plasticity gates.

        Args:
            grad_stats: [num_groups, 2] tensor (grad_norm, grad_var per group).
            progress: training progress in [0, 1].

        Returns:
            gates: [num_groups] tensor in [0, 1] via sigmoid.
        """
        p = torch.full(
            (self.num_groups, 1),
            progress,
            device=grad_stats.device,
            dtype=grad_stats.dtype,
        )
        x = torch.cat([grad_stats, p], dim=-1)
        h = F.silu(self.proj(x))
        logits = self.out(h) + self.gate_bias
        return torch.sigmoid(logits)

    def compute_grad_stats(
        self,
        param_groups: list[list[torch.nn.Parameter]],
    ) -> torch.Tensor:
        """Collect gradient statistics per parameter group.

        Returns:
            [num_groups, 2] tensor: (log_grad_norm, log_grad_var)
        """
        stats = []
        for group in param_groups:
            grads = [p.grad for p in group if p.grad is not None]
            if not grads:
                stats.append(torch.tensor([0.0, 0.0]))
                continue
            norms = torch.stack([g.norm() for g in grads])
            stats.append(torch.stack([norms.mean().log(), norms.var().log()]))
        return torch.stack(stats)


def apply_foxp2(
    controller: FOXP2Controller,
    param_groups: list[list[torch.nn.Parameter]],
    progress: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Apply FOXP2 plasticity gates to gradients.

    Call AFTER backward(), BEFORE optimizer.step().

    Args:
        controller: FOXP2Controller instance.
        param_groups: list of parameter lists (per group/layer).
        progress: training progress in [0, 1].
        device: device for computation.
        dtype: dtype for computation.

    Returns:
        gates: [num_groups] tensor (for logging).
    """
    with torch.no_grad():
        grad_stats = controller.compute_grad_stats(param_groups).to(
            device=device, dtype=dtype
        )
        gates = controller(grad_stats, progress)

        for idx, group in enumerate(param_groups):
            scale = float(gates[idx].item())
            for p in group:
                if p.grad is not None:
                    p.grad.mul_(scale)

    return gates
