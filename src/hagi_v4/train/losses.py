"""V8 CodecLoss — 3-level loss hierarchy for from-scratch training.

Level 1 (Fidelity): CE — always active
Level 2 (Code Quality): Parity reward, Rate distortion — after warmup
Level 3 (Convergence): Extrinsic info, Whiteness — after 2×warmup

V8 simplification: 4 aux losses instead of V7's 7.
Removed: efficiency (redundant), msa_lb (impl detail), contrastive (multimodal-only).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.codec_contracts import TrainLossConfig
from hagi_v4.model.outputs import AuxLosses, ModelOutput


def _validate_selected_rows(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("logits and targets must have shapes [N,V] and [N]")
    if logits.shape[0] == 0:
        raise ValueError("selected rows must be non-empty")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same number of rows")


def selected_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over already gathered prediction rows."""
    _validate_selected_rows(logits, targets)
    return F.cross_entropy(logits, targets)


def suffix_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    is_suffix_prediction: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy restricted to gathered rows from suffix tasks."""
    _validate_selected_rows(logits, targets)
    if is_suffix_prediction.ndim != 1 or is_suffix_prediction.shape[0] != logits.shape[0]:
        raise ValueError("is_suffix_prediction must have shape [N]")
    if is_suffix_prediction.dtype != torch.bool:
        raise ValueError("is_suffix_prediction must be boolean")
    if not is_suffix_prediction.any():
        return logits.new_full((), float("nan"))
    return F.cross_entropy(logits[is_suffix_prediction], targets[is_suffix_prediction])


def mean_top2_probability_mass(logits: torch.Tensor) -> torch.Tensor:
    """Mean posterior mass assigned to each row's two likeliest tokens."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have non-empty shape [N,V]")
    probabilities = logits.float().softmax(dim=-1)
    return probabilities.topk(min(2, logits.shape[-1]), dim=-1).values.sum(dim=-1).mean()


def mean_posterior_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean categorical posterior entropy in nats over gathered rows."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have non-empty shape [N,V]")
    log_probabilities = logits.float().log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=-1).mean()


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
    """Computes total loss with 3-level hierarchy.

    Phase 0 (step < warmup): CE only — model learns basic representations.
    Phase 1 (warmup ≤ step < 2×warmup): CE + L2 (code quality).
    Phase 2 (step ≥ 2×warmup): CE + L2 + L3 (full codec optimization).

    This staged activation prevents loss explosion when training from scratch.
    """

    def __init__(self, cfg: HAGIv4Config | TrainLossConfig):
        contract = TrainLossConfig.from_hagi_config(cfg) if isinstance(cfg, HAGIv4Config) else cfg
        self.w_whiteness = contract.whiteness_weight
        self.w_parity = contract.parity_weight
        self.w_correction_alignment = contract.correction_alignment_weight
        self.w_rate_distortion = contract.rate_distortion_weight
        self.w_contrastive = contract.contrastive_weight
        self.warmup_steps = cfg.train.warmup_steps if isinstance(cfg, HAGIv4Config) else 5000

    def _loss_level(self, step: int) -> int:
        """Return active loss level: 1, 2, or 3."""
        w = self.warmup_steps
        if step < w:
            return 1
        if step < 2 * w:
            return 2
        return 3

    def __call__(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        step: int = 0,
    ) -> torch.Tensor:
        if model_output.ce_loss is not None:
            ce_loss = model_output.ce_loss
        else:
            ce_loss = masked_cross_entropy_chunked(model_output.logits, targets, mask)

        aux: AuxLosses = model_output.aux
        total = ce_loss

        level = self._loss_level(step)
        if level == 1:
            return total

        if aux.rate_distortion is not None:
            total = total + self.w_rate_distortion * aux.rate_distortion
        if aux.parity is not None:
            total = total + self.w_parity * torch.log1p(aux.parity.float())
        if aux.contrastive is not None:
            total = total + self.w_contrastive * aux.contrastive

        if level == 2:
            return total

        if aux.whiteness is not None:
            total = total + self.w_whiteness * aux.whiteness
        if aux.correction_alignment is not None:
            total = total + self.w_correction_alignment * aux.correction_alignment
        if aux.parity_recovery is not None:
            total = total + self.w_parity * 0.01 * aux.parity_recovery

        return total
