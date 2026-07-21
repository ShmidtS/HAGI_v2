# V21 DEFER: not wired in V21 forward path. Available for V22+ integration.
# See docs/ARCHITECTURE.md for integration roadmap.

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
    """Per-layer plasticity gate based on gradient norm.

    Generates a scaling factor in [0, 1] for each parameter group,
    controlling how much of the gradient flows through to the update.
    Mr. Fox decides which advice (gradients) reaches the student (weights).

    Design (per biological analog, Notty):
      - Input: gradient norm ONLY (1 feature). More features degrade it —
        FOXP2 is a selection parameter, not a channel-state estimator.
      - Training-only: not in model forward, not in inference.
      - Not context-dependent on the input signal — depends on gradient
        statistics, never on hidden state / CQI.
      - Task signal: alignment surrogate. Gate is rewarded where the current
        grad-norm is STABLE (close to its EMA) — that indicates a consistent
        signal, not noise. Gate is suppressed where the norm is an outlier
        (likely a noisy/spurious update). This replaces the L2 collapse
        failure (all gates -> 0) and the entropy collapse (all gates -> 0.5).

    Args:
        num_groups: number of parameter groups (e.g. layers)
        hidden: hidden dimension of controller network
        ema_decay: EMA decay for the log-norm trend estimate
    """

    def __init__(self, num_groups: int, hidden: int = 32, ema_decay: float = 0.99) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.ema_decay = ema_decay

        self.proj = nn.Linear(1, hidden)
        self.out = nn.Linear(hidden, 1)

        self.gate_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

        self.register_buffer("ema_log_norm", torch.zeros(num_groups), persistent=False)

    def forward(
        self,
        grad_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Generate per-group plasticity gates.

        Args:
            grad_norms: [num_groups] tensor (log-scaled grad norm per group).

        Returns:
            gates: [num_groups] tensor in [0, 1] via sigmoid.
        """
        x = grad_norms.unsqueeze(-1)
        h = F.silu(self.proj(x))
        logits = self.out(h).squeeze(-1) + self.gate_bias
        return torch.sigmoid(logits)

    @torch.no_grad()
    def update_ema(self, log_norms: torch.Tensor) -> None:
        """Update EMA of log-grad-norm per group. Call after the meta-step."""
        self.ema_log_norm.mul_(self.ema_decay).add_(log_norms * (1.0 - self.ema_decay))

    def compute_grad_stats(
        self,
        param_groups: list[list[torch.nn.Parameter]],
    ) -> torch.Tensor:
        """Collect log gradient norm per parameter group.

        Returns:
            [num_groups] tensor of log(grad_norm).
        """
        stats = []
        for group in param_groups:
            grads = [p.grad for p in group if p.grad is not None]
            if not grads:
                stats.append(
                    torch.tensor(0.0, device=param_groups[0][0].device if param_groups and param_groups[0] else "cpu")
                )
                continue
            norms = torch.stack([g.norm() for g in grads])
            stats.append(norms.mean().log())
        return torch.stack(stats)


def foxp2_alignment_loss(gates: torch.Tensor, log_norms: torch.Tensor, ema_log_norm: torch.Tensor) -> torch.Tensor:
    """Alignment surrogate: reward gates where grad-norm is stable (signal).

    alignment_i = exp(-|log_norm_i - ema_log_norm_i|)  in [0,1]
      ~1: current norm matches the trend → consistent signal → keep gate high
      ~0: current norm is an outlier → likely noise → suppress gate
    Loss = -mean(gates * alignment)  (minimize → raise gates on stable groups).

    This is the task signal Mr. Fox learns from: which groups carry a stable
    gradient signal vs which are noisy, WITHOUT a second forward pass.
    """
    alignment = torch.exp(-(log_norms - ema_log_norm).abs())
    return -(gates * alignment.detach()).mean()


def foxp2_entropy_loss(gates: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Entropy regularization: encourage DIVERSE, non-collapsed gates.

    Maximizing entropy = minimizing -H. Returns loss to be MINIMIZED:
    L = -mean( -p*log p - (1-p)*log(1-p) ),  p = gates.
    Uniform p=0.5 → max entropy → loss 0. Collapsed p≈0 or p≈1 → low entropy → high loss.
    This counteracts the L2-collapse failure mode.
    """
    p = gates.clamp(eps, 1.0 - eps)
    binary_entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return -binary_entropy.mean()


def apply_foxp2(
    controller: FOXP2Controller,
    param_groups: list[list[torch.nn.Parameter]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply FOXP2 plasticity gates to gradients.

    Call AFTER backward(), BEFORE optimizer.step().

    Args:
        controller: FOXP2Controller instance.
        param_groups: list of parameter lists (per group/layer).
        device: device for computation.
        dtype: dtype for computation.

    Returns:
        (gates, log_norms): gates [num_groups] with grad_fn (for meta-loss),
        log_norms [num_groups] detached (for alignment loss + EMA update).
    """
    with torch.no_grad():
        log_norms = controller.compute_grad_stats(param_groups).to(device=device, dtype=dtype)

    gates = controller(log_norms)

    with torch.no_grad():
        for idx, group in enumerate(param_groups):
            scale = float(gates[idx].item())
            for p in group:
                if p.grad is not None:
                    p.grad.mul_(scale)

    return gates, log_norms
