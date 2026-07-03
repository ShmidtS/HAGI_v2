"""Loss functions for HAGI V4: masked CE + aux losses.

V4 uses masked cross-entropy (predict masked positions, not next-token).
Auxiliary losses: MoE load-balance, GDR router, MSA load-balance,
coherence regularization, deep supervision (from refinement loop).

Section 7.2 of ARCHITECTURE_V4.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Cross-entropy only on masked positions.

    Args:
        logits: [B, T, V] logits for all positions.
        targets: [B, T] target token IDs (same-position, not shifted).
        mask: [B, T] bool tensor, True where masked. If None, CE on all.

    Returns:
        Scalar CE loss.
    """
    if mask is None:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    if not mask.any():
        return logits.new_zeros(())
    masked_logits = logits[mask]
    masked_targets = targets[mask]
    return F.cross_entropy(masked_logits, masked_targets)


def unmask_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 0.3,
) -> torch.Tensor:
    """CE on unmasked positions — signal density boost.

    Forces the model to also "explain" visible tokens, increasing
    training signal from ~30% to ~100% at reduced weight.
    """
    unmask = ~mask
    if not unmask.any():
        return logits.new_zeros(())
    return weight * F.cross_entropy(
        logits[unmask],
        targets[unmask],
    )


def compute_total_loss(
    ce_loss: torch.Tensor,
    moe_aux: torch.Tensor,
    gdr_router: torch.Tensor,
    msa_lb: torch.Tensor,
    deep_supervision: torch.Tensor,
    coherence: torch.Tensor,
    w_moe_aux: float = 0.01,
    w_gdr_router: float = 0.01,
    w_msa_lb: float = 0.01,
    w_coherence: float = 0.01,
) -> torch.Tensor:
    """Weighted sum of all loss terms."""
    return (
        ce_loss
        + w_moe_aux * moe_aux
        + w_gdr_router * gdr_router
        + w_msa_lb * msa_lb
        + deep_supervision
        + w_coherence * coherence
    )
