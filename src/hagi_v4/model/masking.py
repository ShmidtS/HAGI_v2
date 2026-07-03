"""Masking utilities for V4 plane prediction training.

Supports three masking patterns:
- Random token masking (60% of batches) — BERT/LLaDA-style
- Span masking (15% of batches) — contiguous spans of span_length
- Suffix masking (25% of batches) — matches inference distribution

Progressive mask ratio: 15% early -> 30% late training.
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
    mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
    for b in range(B):
        idx = torch.randperm(T, device=input_ids.device)[:n_mask]
        mask[b, idx] = True
    masked_ids = input_ids.clone()
    masked_ids[mask] = mask_token_id
    return masked_ids, mask


def create_span_mask(
    input_ids: torch.Tensor,
    mask_ratio: float = 0.3,
    span_length: int = 3,
    mask_token_id: int = 49153,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Span masking: mask contiguous spans of span_length tokens.

    Returns (masked_ids, mask).
    """
    B, T = input_ids.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
    n_spans = max(1, int(T * mask_ratio / span_length))
    for b in range(B):
        for _ in range(n_spans):
            start = torch.randint(0, max(1, T - span_length + 1), (1,)).item()
            end = min(start + span_length, T)
            mask[b, start:end] = True
    masked_ids = input_ids.clone()
    masked_ids[mask] = mask_token_id
    return masked_ids, mask


def create_suffix_mask(
    input_ids: torch.Tensor,
    mask_ratio: float = 0.3,
    mask_token_id: int = 49153,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Suffix masking: mask contiguous suffix (left context only).

    Matches inference distribution where prompt is visible and
    generation region is masked.

    Returns (masked_ids, mask).
    """
    B, T = input_ids.shape
    n_mask = max(1, int(T * mask_ratio))
    mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
    mask[:, T - n_mask :] = True
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
