"""Loss aggregation for the ternary RD-channel LM.

Objective = cross-entropy + auxiliary regularizers:
  * rate       = KL[q(z|h)||N(0,I)]  (the genuine information-bottleneck rate)
  * distortion = normalized RD reconstruction (beta-annealed over warmup)
  * perception = RDP residual-autocorrelation axis (beta-annealed over warmup)
  * attn_entropy = anti-collapse penalty (active from step 0)
  * ternary_bias / moe_lb = opt-in (default inert)

The CE is the next-token loss supplied by the training loop (causal). The RD
distortion/perception are beta-annealed over warmup so the LM signal via the
main path shapes the representation first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hagi_v4.config import Config
from hagi_v4.model.outputs import ModelOutput


def selected_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over already gathered prediction rows."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits/targets must have shapes [N,V] and [N]")
    if logits.shape[0] == 0:
        raise ValueError("selected rows must be non-empty")
    return F.cross_entropy(logits, targets)


def suffix_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    is_suffix_prediction: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy restricted to gathered rows from suffix tasks."""
    if not is_suffix_prediction.any():
        return logits.new_full((), float("nan"))
    return F.cross_entropy(logits[is_suffix_prediction], targets[is_suffix_prediction])


class LossAggregator:
    """Computes total loss = CE + active auxiliary regularizers."""

    def __init__(self, cfg: Config):
        t = cfg.train
        self.w_rate = t.w_rate
        self.w_distortion = t.w_distortion
        self.w_perception = t.w_perception
        self.w_ternary_bias = t.w_ternary_bias
        self.w_moe_load_balance = t.w_moe_load_balance
        self.w_attn_entropy = t.w_attn_entropy
        self.warmup_steps = max(1, t.warmup_steps)

    def __call__(self, model_output: ModelOutput, step: int = 0) -> torch.Tensor:
        if model_output.ce_loss is None:
            raise ValueError("model output must include ce_loss")
        total = model_output.ce_loss
        aux = model_output.aux

        # IB rate (the only genuine rate notion).
        if aux.rate is not None:
            total = total + self.w_rate * aux.rate

        # Distortion warmup (VAE beta-annealing): distortion is computed over the
        # un-normalized h_ctx, so at init it scales with ||h_ctx||^2 and would
        # dominate CE. Ramp its weight 0->full over warmup.
        if aux.distortion is not None and self.w_distortion > 0.0:
            frac = min(1.0, step / self.warmup_steps)
            total = total + (self.w_distortion * frac) * aux.distortion

        # Perception (RDP third axis) beta-anneals alongside distortion.
        if aux.perception is not None and self.w_perception > 0.0:
            perc_frac = min(1.0, step / self.warmup_steps)
            total = total + (self.w_perception * perc_frac) * aux.perception

        # Attention entropy regularization (anti-collapse, active from step 0).
        if aux.attn_entropy is not None and self.w_attn_entropy > 0.0:
            total = total + self.w_attn_entropy * aux.attn_entropy

        # Opt-in ternary-bias lattice-alignment regularizer.
        if aux.ternary_bias is not None and self.w_ternary_bias > 0.0:
            total = total + self.w_ternary_bias * aux.ternary_bias

        # Opt-in MoE load-balance (Switch-Transformer CV^2).
        if aux.moe_lb is not None and self.w_moe_load_balance > 0.0:
            total = total + self.w_moe_load_balance * aux.moe_lb

        return total
