"""Multi-objective CodecLoss for HAGI V5.

V5 loss is aligned with communication theory:
  Loss = CE (fidelity) + IB (compression) + Parity (redundancy)
       + ExtrinsicInfo (decoding) + Efficiency (convergence)
       + aux losses (MoE/GDR/MSA load balance, coherence, whiteness)

LossAggregator replaces inline loss aggregation formerly in hagi_v4.py.
Stateless loss helpers (compute_whiteness_loss, compute_grade_spec_loss)
live in model/outputs.py to avoid model→train circular dependency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.outputs import AuxLosses, ModelOutput


def masked_cross_entropy_chunked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Cross-entropy on masked positions, or all positions if mask is None.

    Uses chunking when no mask is provided to limit peak memory.
    """
    if mask is not None and mask.any():
        return F.cross_entropy(logits[mask], targets[mask])
    if mask is not None and not mask.any():
        return logits.new_zeros(())
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    total_ce = flat_logits.new_zeros(())
    for i in range(0, flat_logits.size(0), chunk_size):
        end = min(i + chunk_size, flat_logits.size(0))
        ce_chunk = F.cross_entropy(flat_logits[i:end], flat_targets[i:end], reduction="sum")
        total_ce = total_ce + ce_chunk
    return total_ce / flat_targets.size(0)


class LossAggregator:
    """Computes total loss from ModelOutput + targets + mask.

    V5: multi-objective CodecLoss aligned with communication theory.
    """

    def __init__(self, cfg: HAGIv4Config):
        self.w_moe_aux = cfg.train.w_moe_aux
        self.w_gdr_router = cfg.train.w_gdr_router
        self.w_coherence = cfg.train.w_coherence
        self.w_whiteness = cfg.train.w_whiteness
        self.w_grade_spec = cfg.train.w_grade_specialization
        self.w_parity = cfg.train.w_parity
        self.w_extrinsic_info = cfg.train.w_extrinsic_info
        self.w_efficiency = cfg.train.w_efficiency
        self.w_kl_variational = getattr(cfg.train, "w_kl_variational", 0.01)
        self.w_msa_lb = 0.01

    def __call__(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        w_coherence_override: float | None = None,
    ) -> torch.Tensor:
        if model_output.ce_loss is not None:
            ce_loss = model_output.ce_loss
        else:
            ce_loss = masked_cross_entropy_chunked(model_output.logits, targets, mask)

        w_coh = w_coherence_override if w_coherence_override is not None else self.w_coherence

        aux: AuxLosses = model_output.aux
        total = ce_loss

        if aux.moe_lb is not None:
            total = total + self.w_moe_aux * aux.moe_lb
        if aux.gdr_router is not None:
            total = total + self.w_gdr_router * aux.gdr_router
        if aux.msa_lb is not None:
            total = total + self.w_msa_lb * aux.msa_lb
        if aux.deep_supervision is not None:
            total = total + aux.deep_supervision
        if aux.coherence is not None:
            total = total + w_coh * aux.coherence
        if aux.whiteness is not None:
            total = total + self.w_whiteness * aux.whiteness
        if aux.grade_spec is not None:
            total = total + self.w_grade_spec * aux.grade_spec
        if aux.parity is not None:
            total = total - self.w_parity * aux.parity
        if aux.extrinsic_info is not None:
            total = total - self.w_extrinsic_info * aux.extrinsic_info
        if aux.efficiency is not None:
            total = total + self.w_efficiency * aux.efficiency
        if aux.ib is not None:
            total = total + self.w_kl_variational * aux.ib

        return total
