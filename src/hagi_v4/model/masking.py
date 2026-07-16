"""Semantic prediction and physical corruption policies."""

from __future__ import annotations

import torch


def create_physical_corruption_mask(
    input_ids: torch.Tensor,
    corruption_ratio: float | torch.Tensor = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample physical channel corruption without modifying token IDs."""
    B, T = input_ids.shape
    ratios = torch.as_tensor(corruption_ratio, device=input_ids.device).expand(B)
    if not torch.isfinite(ratios).all() or ((ratios < 0) | (ratios > 1)).any():
        raise ValueError("corruption_ratio must be in [0, 1]")
    mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
    if not ratios.any():
        return mask
    counts = (ratios * T).to(torch.long)
    scores = torch.rand(B, T, device=input_ids.device, generator=generator)
    for row, count in enumerate(counts.tolist()):
        if count:
            mask[row, scores[row].topk(count).indices] = True
    return mask


def create_semantic_corruption(
    valid_target_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    random_probability: float = 0.5,
    random_ratio: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random or contiguous full-suffix semantic prediction tasks."""
    if valid_target_mask.ndim != 2 or valid_target_mask.dtype != torch.bool:
        raise ValueError("valid_target_mask must be a BoolTensor[B,T]")
    if not 0.0 <= random_probability <= 1.0:
        raise ValueError("random_probability must be in [0, 1]")
    if not 0.0 <= random_ratio <= 1.0:
        raise ValueError("random_ratio must be in [0, 1]")
    if not valid_target_mask.any(dim=1).all():
        raise ValueError("every row must contain at least one valid target")
    if ((~valid_target_mask[:, :-1]) & valid_target_mask[:, 1:]).any():
        raise ValueError("valid_target_mask must be a contiguous True prefix per row")

    B, T = valid_target_mask.shape
    device = valid_target_mask.device
    is_suffix = torch.rand(B, generator=generator, device=device) >= random_probability
    random_scores = torch.rand(B, T, generator=generator, device=device)
    start_scores = torch.rand(B, generator=generator, device=device)
    semantic_unknown_mask = (random_scores < random_ratio) & valid_target_mask

    valid_counts = valid_target_mask.sum(dim=1)
    random_rows = ~is_suffix
    missing_random = random_rows & ~semantic_unknown_mask.any(dim=1)
    fallback_indices = (start_scores * valid_counts).to(torch.long)
    if missing_random.any():
        rows = missing_random.nonzero(as_tuple=False).flatten()
        semantic_unknown_mask[rows, fallback_indices[rows]] = True

    suffix_starts = (start_scores * valid_counts).to(torch.long)
    positions = torch.arange(T, device=device).unsqueeze(0)
    suffix_mask = (positions >= suffix_starts.unsqueeze(1)) & valid_target_mask
    semantic_unknown_mask = torch.where(is_suffix.unsqueeze(1), suffix_mask, semantic_unknown_mask)
    prediction_mask = semantic_unknown_mask.clone()
    return semantic_unknown_mask, prediction_mask, is_suffix


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
