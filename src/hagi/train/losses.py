"""Loss aggregation for the codec-channel LM.

Objective = cross-entropy + auxiliary regularizers:
  * rate       = KL[q(z|h)||N(0,I)]  (information-bottleneck rate)
  * distortion = normalized RD reconstruction (beta-annealed over warmup,
                 FREEZED once the EXIT-chart novelty converges)
  * vicreg     = grounded infomax joint-embedding stabilizer (multimodal)
  * infonce    = cross-modal MI lower bound (multimodal)
  * moe_lb     = Switch CV^2 expert load-balance (MoE)
  * route_entropy = routing-entropy CAPACITY MAXIMIZATION (water-filling dual;
                 subtracted: higher routing entropy = more capacity used)
  * water_filling = per-expert capacity allocator entropy-gap regularizer (MoE;
                 added: minimize log(E)-H(p_alloc) -> spread capacity)
  * refinement = off-path HEP predictive-refinement loss (opt-in)
  * attn_entropy = anti-collapse penalty (active from step 0)

The CE is the next-token loss supplied by the training loop (causal). RD
distortion is beta-annealed over warmup so the LM signal shapes the
representation first; the EXIT-chart halt freezes beta when the representation
converges (no new extrinsic information).
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
    """Computes total loss = CE + active auxiliary regularizers.

    Args:
        cfg: top-level config (reads w_* weights and warmup_steps).
        exit_halt: optional :class:`~hagi.model.exit_chart.EXITChartHalt`. When
            the distortion beta-anneal's EXIT novelty has converged, ``beta``
            is frozen at its current value (the representation has stopped
            improving; further annealing adds nothing).
    """

    def __init__(self, cfg: Config, exit_halt=None):
        t = cfg.train
        self.w_rate = t.w_rate
        self.w_distortion = t.w_distortion
        self.w_vicreg = t.w_vicreg
        self.w_infonce = t.w_infonce
        self.w_moe_lb = t.w_moe_lb
        self.w_route_entropy = t.w_route_entropy
        self.w_water_filling = t.w_water_filling
        self.w_refine = t.w_refine
        self.w_attn_entropy = t.w_attn_entropy
        self.warmup_steps = max(1, t.warmup_steps)
        self._frozen_beta: float | None = None
        self._exit_halt = exit_halt

    def update_exit_novelty(self, aux_novelty: float | None) -> None:
        """Feed the per-step refinement novelty into the EXIT halt.

        Once halted, the distortion beta is frozen (no further annealing).
        """
        if self._exit_halt is None or aux_novelty is None:
            return
        if self._frozen_beta is None:
            self._exit_halt.observe(float(aux_novelty))

    def __call__(self, model_output: ModelOutput, step: int = 0) -> torch.Tensor:
        if model_output.ce_loss is None:
            raise ValueError("model output must include ce_loss")
        total = model_output.ce_loss
        aux = model_output.aux

        # IB rate (the genuine rate notion).
        if aux.rate is not None:
            total = total + self.w_rate * aux.rate

        # Distortion warmup (VAE beta-annealing), FREEZED on EXIT convergence.
        if aux.distortion is not None and self.w_distortion > 0.0:
            if self._frozen_beta is None:
                frac = min(1.0, step / self.warmup_steps)
                # If the EXIT halt just fired, freeze beta at the current frac.
                if self._exit_halt is not None and self._exit_halt.halted:
                    self._frozen_beta = frac
            beta = self._frozen_beta if self._frozen_beta is not None else min(1.0, step / self.warmup_steps)
            total = total + (self.w_distortion * beta) * aux.distortion

        # Grounded infomax (multimodal).
        if aux.vicreg is not None and self.w_vicreg > 0.0:
            total = total + self.w_vicreg * aux.vicreg
        if aux.infonce is not None and self.w_infonce > 0.0:
            total = total + self.w_infonce * aux.infonce

        # MoE load-balance (Switch CV^2).
        if aux.moe_lb is not None and self.w_moe_lb > 0.0:
            total = total + self.w_moe_lb * aux.moe_lb

        # Routing-entropy capacity maximization: SUBTRACT (maximize H(p_route)).
        # Water-filling dual: spread capacity across the parallel expert channels.
        if aux.route_entropy is not None and self.w_route_entropy > 0.0:
            total = total - self.w_route_entropy * aux.route_entropy

        # Water-filling allocator entropy-gap regularizer: ADD (minimize the gap
        # log(E)-H(p_alloc) -> keep capacity spread across expert channels).
        if aux.water_filling is not None and self.w_water_filling > 0.0:
            total = total + self.w_water_filling * aux.water_filling

        # Off-path HEP refinement (opt-in).
        if aux.refinement is not None and self.w_refine > 0.0:
            total = total + self.w_refine * aux.refinement

        # Attention entropy regularization (anti-collapse, active from step 0).
        if aux.attn_entropy is not None and self.w_attn_entropy > 0.0:
            total = total + self.w_attn_entropy * aux.attn_entropy

        return total

    @property
    def exit_halted(self) -> bool:
        return self._frozen_beta is not None
