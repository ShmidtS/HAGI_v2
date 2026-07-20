"""V9 CodecLoss — 3-level loss hierarchy for from-scratch training.

Level 1 (Fidelity): CE — always active
Level 2 (Code Quality): Parity reward, Rate distortion — after warmup
Level 3 (Convergence): Extrinsic info, Whiteness — after 2×warmup

V9 change vs V8: ``correction_alignment`` weight reduced from 0.01 to 0.001.
The V8 log showed ``correction_alignment ~1.75`` dominating the CE signal
(``~6.0``) once scaled (0.0175 vs 6.0 is small, but the metric still grew
instead of shrinking — the optimizer reduced it, inflating ``loss`` as a
Goodhart target while ``masked_ce`` stagnated). The smaller weight keeps the
objective aligned with the actual fidelity measure (CE) while still
encouraging the decoder to recover erased systematic symbols.
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
        self.warmup_steps = cfg.train.warmup_steps if isinstance(cfg, HAGIv4Config) else 5000
        # V12: parity-code diversity weight. Active from step 0 (Level 1)
        # because collapse can occur very early — the V12 step-1500 log
        # showed ``par`` already at 0.01 by step 1500, meaning the code
        # had degenerated before the Level 2 codec losses kicked in.
        if isinstance(cfg, HAGIv4Config):
            self.w_parity_diversity = getattr(cfg.train, "w_parity_diversity", 0.05)
        else:
            self.w_parity_diversity = 0.05

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

        # V12/V18: parity diversity is the FIRST codec-side loss to activate
        # (Level 1). It operates purely on the parity-check matrix H and does
        # not depend on the decoder's forward pass, so it is safe to apply
        # before the decoder has learned anything useful. V17 had w=0.0 —
        # that was a bug that let the LDPC graph collapse.
        if aux.parity_diversity is not None and self.w_parity_diversity > 0.0:
            total = total + self.w_parity_diversity * aux.parity_diversity

        level = self._loss_level(step)
        if level == 1:
            return total

        if aux.rate_distortion is not None:
            total = total + self.w_rate_distortion * aux.rate_distortion
        if aux.parity is not None:
            total = total + self.w_parity * torch.log1p(aux.parity.float())
        # V18: parity_recovery explicitly supervised (erasure-tolerance loss).
        # Active from Level 2 so the LDPC code is forced to carry real
        # information about erased systematic positions.
        if aux.parity_recovery is not None:
            total = total + self.w_parity * 0.5 * aux.parity_recovery

        if level == 2:
            return total

        # V18: w_correction_alignment defaults to 0 — V8 showed it Goodhart-ised.
        if self.w_correction_alignment > 0.0 and aux.correction_alignment is not None:
            total = total + self.w_correction_alignment * aux.correction_alignment
        if self.w_whiteness > 0.0 and aux.whiteness is not None:
            total = total + self.w_whiteness * aux.whiteness
        return total
