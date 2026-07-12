"""Multi-objective CodecLoss for HAGI V7.1.

Loss = CE (fidelity) + Parity (redundancy reward) + Whiteness (decorrelation)
     + ExtrinsicInfo (decoding reward) + Efficiency (convergence cost)
     + RateDistortion (information loss) + MSA load balance
     + Contrastive (modality alignment, multimodal only).
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
    """Computes total loss from ModelOutput + targets + mask."""

    def __init__(self, cfg: HAGIv4Config):
        self.w_whiteness = cfg.train.w_whiteness
        self.w_parity = cfg.train.w_parity
        self.w_extrinsic_info = cfg.train.w_extrinsic_info
        self.w_efficiency = cfg.train.w_efficiency
        self.w_msa_lb = getattr(cfg.train, "w_msa_lb", 0.01)
        self.w_rate_distortion = cfg.train.w_rate_distortion
        self.w_contrastive = cfg.train.w_contrastive if hasattr(cfg.train, "w_contrastive") else 0.0

    def __call__(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if model_output.ce_loss is not None:
            ce_loss = model_output.ce_loss
        else:
            ce_loss = masked_cross_entropy_chunked(model_output.logits, targets, mask)

        aux: AuxLosses = model_output.aux
        total = ce_loss

        if aux.msa_lb is not None:
            total = total + self.w_msa_lb * aux.msa_lb
        if aux.whiteness is not None:
            total = total + self.w_whiteness * aux.whiteness
        if aux.parity is not None:
            total = total - self.w_parity * aux.parity.clamp(max=1.0)
        if aux.extrinsic_info is not None:
            total = total - self.w_extrinsic_info * aux.extrinsic_info
        if aux.efficiency is not None:
            total = total + self.w_efficiency * aux.efficiency
        if aux.rate_distortion is not None:
            total = total + self.w_rate_distortion * aux.rate_distortion
        if aux.contrastive is not None:
            total = total + self.w_contrastive * aux.contrastive

        return total
