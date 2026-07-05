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


def information_bottleneck_loss(
    h: torch.Tensor,
    targets: torch.Tensor,
    lm_head_weight: torch.Tensor,
    beta: float = 1.0,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Information Bottleneck regularizer: I(X;Z) - beta * I(Y;Z).

    I(X;Z) proxy: hidden state variance (complexity — how much input info is stored).
    I(Y;Z) proxy: negative cross-entropy (predictive information — how much target info is captured).

    Minimizing this loss drives the model toward the IB bound:
    reduce complexity while maintaining predictive information.
    """
    complexity = h.float().var(dim=(0, 1)).sum()

    flat_h = h.reshape(-1, h.size(-1))
    flat_t = targets.reshape(-1)
    total_ce = h.new_zeros(())
    for i in range(0, flat_h.size(0), chunk_size):
        end = min(i + chunk_size, flat_h.size(0))
        logits_c = F.linear(flat_h[i:end], lm_head_weight)
        total_ce = total_ce + F.cross_entropy(logits_c, flat_t[i:end], reduction="sum")
    ce = total_ce / flat_t.size(0)
    del flat_h, flat_t

    predictive_info = -ce
    return complexity - beta * predictive_info


def gp2d_whiteness_loss(residual: torch.Tensor) -> torch.Tensor:
    """Penalize lag-1 autocorrelation of GP2D residual along temporal axis.

    Optimal predictive coding produces white (uncorrelated) residuals.
    Nonzero autocorrelation means remaining structure is not exploited.
    """
    if residual.size(1) < 2:
        return residual.new_zeros(())
    r_t = residual[:, :-1].reshape(-1, residual.size(-1))
    r_t1 = residual[:, 1:].reshape(-1, residual.size(-1))
    cos_sim = F.cosine_similarity(r_t.float(), r_t1.float(), dim=-1)
    return cos_sim.abs().mean()
