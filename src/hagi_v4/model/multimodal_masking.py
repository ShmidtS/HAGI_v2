# V21 DEFER: not wired in V21 forward path. Available for V22+ integration.
# See docs/ARCHITECTURE.md for integration roadmap.

"""Multimodal masking — multi-level erasure + modality dropout (V7).

Per-modality mask ratios (UEP): text protected more than image/audio.
Modality dropout: complete modality erasure (fading channel simulation).
Training with modality dropout = robustness to modality erasure at inference.

Information theory:
  - Per-modality mask ratios = UEP (Unequal Error Protection)
  - Text: p=0.15 (high capacity, important content, max protection)
  - Image: p=0.30 (lower capacity, visual detail, medium protection)
  - Modality dropout = fading channel (complete signal loss)
"""

from __future__ import annotations

import torch


def create_multimodal_mask(
    modality_ids: torch.Tensor,
    modality_mask_ratios: tuple = (0.15, 0.30, 0.25),
    modality_dropout_prob: float = 0.10,
    use_span_masking: bool = True,
    span_length: int = 3,
) -> tuple[torch.Tensor, list]:
    """Multi-level erasure + modality dropout.

    Args:
        modality_ids: [B, T] -- 0=text, 1=image, 2=audio
        modality_mask_ratios: per-modality erasure rates (UEP)
        modality_dropout_prob: probability of dropping entire modality (fading)
        use_span_masking: contiguous block masking
        span_length: span size for span masking

    Returns:
        mask: [B, T] bool -- True where masked
        dropped: list of sets with dropped modality IDs per batch element
    """
    B, T = modality_ids.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=modality_ids.device)
    dropped: list[set[int]] = []

    for b in range(B):
        dropped_b: set[int] = set()

        for mod_id in range(len(modality_mask_ratios)):
            if torch.rand(1).item() < modality_dropout_prob:
                mod_mask = modality_ids[b] == mod_id
                mask[b, mod_mask] = True
                dropped_b.add(mod_id)

        for mod_id, ratio in enumerate(modality_mask_ratios):
            if mod_id in dropped_b:
                continue
            mod_mask = modality_ids[b] == mod_id
            mod_positions = torch.where(mod_mask)[0]
            n_mod = mod_positions.shape[0]
            if n_mod == 0:
                continue

            if use_span_masking:
                n_spans = max(1, int(n_mod * ratio / span_length))
                for _ in range(n_spans):
                    start_idx = mod_positions[torch.randint(0, n_mod, (1,)).item()]
                    for s in range(span_length):
                        pos = start_idx + s
                        if pos < T and modality_ids[b, pos] == mod_id:
                            mask[b, pos] = True
            else:
                n_mask = max(1, int(n_mod * ratio))
                perm = torch.randperm(n_mod, device=modality_ids.device)[:n_mask]
                mask[b, mod_positions[perm]] = True

        dropped.append(dropped_b)

    return mask, dropped
