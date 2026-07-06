"""Adaptive erasure channel for V5 codec training.

V5: mask ratio adapts to model confidence (capacity matching).
When the model is confident (high prediction confidence), the mask
ratio increases to push harder. When uncertain, it decreases.

Random token masking (BERT/LLaDA-style) with progressive or adaptive
mask ratio: 15% early -> 30% late training, or confidence-driven.
"""

from __future__ import annotations

import torch


def create_random_mask(
    input_ids: torch.Tensor,
    mask_ratio: float = 0.3,
    mask_token_id: int = 49153,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random token masking. Returns (masked_ids, mask).

    Args:
        input_ids: [B, T] token IDs.
        mask_ratio: fraction of positions to mask.
        mask_token_id: token ID used for masked positions.

    Returns:
        masked_ids: [B, T] with masked positions replaced by mask_token_id.
        mask: [B, T] bool tensor, True where masked.
    """
    B, T = input_ids.shape
    n_mask = max(1, int(T * mask_ratio))
    scores = torch.rand(B, T, device=input_ids.device)
    _, indices = scores.topk(n_mask, dim=-1)
    mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
    mask.scatter_(1, indices, True)
    masked_ids = input_ids.clone()
    masked_ids[mask] = mask_token_id
    return masked_ids, mask


def progressive_mask_ratio(
    step: int,
    max_steps: int,
    start_ratio: float = 0.15,
    end_ratio: float = 0.30,
) -> float:
    """Progressive masking: start with start_ratio, increase to end_ratio.

    Linear ramp over max_steps. After max_steps, stays at end_ratio.
    """
    if step >= max_steps:
        return end_ratio
    progress = step / max(max_steps, 1)
    return start_ratio + (end_ratio - start_ratio) * progress


def adaptive_mask_ratio(
    avg_confidence: float,
    current_ratio: float,
    adaptation_rate: float = 0.01,
    min_ratio: float = 0.05,
    max_ratio: float = 0.50,
) -> float:
    """Capacity matching: adjust mask ratio based on model confidence.

    p = 1 - confidence (Shannon erasure channel capacity).
    Smoothed via EMA to avoid oscillation.

    Args:
        avg_confidence: mean prediction confidence (0-1).
        current_ratio: current mask ratio (EMA state).
        adaptation_rate: EMA smoothing factor.
        min_ratio: minimum mask ratio.
        max_ratio: maximum mask ratio.

    Returns:
        Updated mask ratio.
    """
    target = 1.0 - avg_confidence
    target = max(min_ratio, min(max_ratio, target))
    return (1.0 - adaptation_rate) * current_ratio + adaptation_rate * target
