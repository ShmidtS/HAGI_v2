"""Loss aggregation for the codec-channel LM.

Objective = cross-entropy + auxiliary regularizers:
  * rate       = KL[q(z|h)||N(0,I)]  (information-bottleneck rate)
  * distortion = normalized RD reconstruction (beta-annealed over warmup)
  * vicreg     = grounded infomax joint-embedding stabilizer (multimodal)
  * infonce    = cross-modal MI lower bound (multimodal)
  * moe_lb     = Switch CV^2 expert load-balance (MoE)
  * attn_entropy = anti-collapse penalty (active from step 0)

The CE is the next-token loss supplied by the training loop (causal). RD
distortion is beta-annealed over warmup so the LM signal shapes the
representation first (distortion scales with ||h_ctx||^2 at init and would
dominate CE otherwise).
"""

from __future__ import annotations

import torch

from hagi.config import Config
from hagi.model.outputs import ModelOutput


def selected_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over already gathered prediction rows."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits/targets must have shapes [N,V] and [N]")
    if logits.shape[0] == 0:
        raise ValueError("selected rows must be non-empty")
    return torch.nn.functional.cross_entropy(logits, targets)


class LossAggregator:
    """Computes total loss = CE + active auxiliary regularizers."""

    def __init__(self, cfg: Config):
        t = cfg.train
        self.w_rate = t.w_rate
        self.w_distortion = t.w_distortion
        self.w_vicreg = t.w_vicreg
        self.w_infonce = t.w_infonce
        self.w_moe_lb = t.w_moe_lb
        self.w_attn_entropy = t.w_attn_entropy
        self.warmup_steps = max(1, t.warmup_steps)

    def __call__(self, model_output: ModelOutput, step: int = 0) -> torch.Tensor:
        if model_output.ce_loss is None:
            raise ValueError("model output must include ce_loss")
        total = model_output.ce_loss
        aux = model_output.aux

        # IB rate (the genuine rate notion).
        if aux.rate is not None:
            total = total + self.w_rate * aux.rate

        # Distortion warmup (VAE beta-annealing): computed over the un-normalized
        # h_ctx, so at init it scales with ||h_ctx||^2 and would dominate CE.
        if aux.distortion is not None and self.w_distortion > 0.0:
            frac = min(1.0, step / self.warmup_steps)
            total = total + (self.w_distortion * frac) * aux.distortion

        # Grounded infomax (multimodal).
        if aux.vicreg is not None and self.w_vicreg > 0.0:
            total = total + self.w_vicreg * aux.vicreg
        if aux.infonce is not None and self.w_infonce > 0.0:
            total = total + self.w_infonce * aux.infonce

        # MoE load-balance (Switch CV^2).
        if aux.moe_lb is not None and self.w_moe_lb > 0.0:
            total = total + self.w_moe_lb * aux.moe_lb

        # Attention entropy regularization (anti-collapse, active from step 0).
        if aux.attn_entropy is not None and self.w_attn_entropy > 0.0:
            total = total + self.w_attn_entropy * aux.attn_entropy

        return total
