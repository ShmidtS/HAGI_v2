"""Typed outputs for HAGI V8 — simplified aux losses (4 instead of 7).

V8 loss hierarchy (3 levels):
  Level 1 (Fidelity): CE — always active
  Level 2 (Code Quality): Parity reward, Rate distortion — after warmup
  Level 3 (Convergence): Extrinsic info, Whiteness — after 2×warmup

Removed vs V7: efficiency and collapsing extrinsic norm minimization,
msa_lb (implementation detail), contrastive (multimodal-only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AuxLosses:
    """Auxiliary loss terms produced by the model forward pass (V8: 6 terms)."""

    whiteness: torch.Tensor | None = None
    parity: torch.Tensor | None = None
    correction_alignment: torch.Tensor | None = None
    rate_distortion: torch.Tensor | None = None
    contrastive: torch.Tensor | None = None
    parity_recovery: torch.Tensor | None = None
    # V12: parity-code diversity. Penalises the channel code when the
    # parity subspace becomes collinear with the systematic subspace
    # (observed collapse: ``par`` metric dropped 0.13 -> 0.01, indicating
    # the decoder trivially satisfies parity without adding information).
    # The regularizer is the mutual coherence μ(H) of the parity-check
    # matrix H: high μ means rows are nearly parallel → low-rank code.
    parity_diversity: torch.Tensor | None = None
    # V22: attention entropy penalty (prevents attention collapse)
    attn_entropy: torch.Tensor | None = None
    # V24: genuine rate-distortion terms from the information bottleneck (§3.1).
    # rate = KL[q(z|h)||N(0,I)] (the honest "code rate"); perception = residual
    # autocorrelation (RDP axis). Only set on the V24 path.
    rate: torch.Tensor | None = None
    distortion: torch.Tensor | None = None
    perception: torch.Tensor | None = None
    # V25: ternary-bias regularizer (§4). A lattice-alignment penalty on the
    # ternary FP masters so the per-output-channel absmean scale tracks a clean
    # {-1,0,+1} point. Only set on the V25 path (None on V24/V23). 0-weight by
    # default (ternary is loss-free at this scale); opt-in via w_ternary_bias.
    ternary_bias: torch.Tensor | None = None
    # V25: MoE load-balance loss (Switch-Transformer coefficient of variation).
    # Only set when model.v25.moe_enabled (MoE is dropped by default — §8 YAGNI).
    moe_lb: torch.Tensor | None = None


@dataclass
class ModelOutput:
    """Unified output from model forward pass (training and inference)."""

    logits: torch.Tensor | None
    hidden: torch.Tensor
    aux: AuxLosses
    ce_loss: torch.Tensor | None = None
    iterations_used: torch.Tensor | None = None
    prediction_indices: torch.Tensor | None = None


@dataclass
class RefinementSideInfo:
    """Side info from iterative refinement (Turbo/HRM decoder).

    V21: carries extrinsic information, deep supervision loss, and
    convergence diagnostics from the iterative decoding loop.
    """

    deep_supervision_loss: torch.Tensor | None = None
    gdr_gate_probs: torch.Tensor | None = None
    gdr_router_loss: torch.Tensor | None = None
    gp2d_residual: torch.Tensor | None = None
    moe_router_probs: list[torch.Tensor] | None = None
    moe_lb: torch.Tensor | None = None
    msa_lb: torch.Tensor | None = None
    iterations_used: torch.Tensor | None = None
    extrinsic_norms: list[float] | None = None
    parity_strength: torch.Tensor | None = None


def compute_whiteness_loss(residual: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Lag-1 autocorrelation penalty — decorrelate parity residuals.

    Low autocorrelation = errors are random (good for LDPC).
    High autocorrelation = burst errors (bad — overwhelm local checks).
    """
    if residual.size(1) < 2:
        return residual.new_zeros(())
    r_t = residual[:, :-1].reshape(-1, residual.size(-1))
    r_t1 = residual[:, 1:].reshape(-1, residual.size(-1))
    cos_sim = F.cosine_similarity(r_t.float(), r_t1.float(), dim=-1)
    if valid_mask is not None:
        adjacent = valid_mask[:, 1:] & valid_mask[:, :-1]
        if adjacent.any():
            return cos_sim[adjacent.reshape(-1)].abs().mean()
        return residual.new_zeros(())
    return cos_sim.abs().mean()


def compute_parity_diversity_loss(parity_weights_masked: torch.Tensor) -> torch.Tensor:
    """V12: anti-collapse regularizer for the LDPC parity-check matrix.

    Information-theoretic rationale: a channel code's minimum distance
    d_min is bounded below by the row-space geometry of H. When rows of
    H become collinear (μ → 1), the code degenerates: every codeword
    satisfies every check trivially, parity residual collapses to zero
    regardless of input, and the decoder adds no information.

    The regularizer is the mean off-diagonal absolute cosine similarity
    of the rows of H (mutual coherence μ(H)). Minimising μ(H) keeps the
    checks geometrically spread, preserving d_min and the decoder's
    ability to localise errors.

    Args:
        parity_weights_masked: the masked parity-check matrix H of shape
            ``[M, C]`` (``SparseParityEncoder.masked_weights``).

    Returns:
        Scalar tensor in ``[0, 1]``. Zero means rows are orthogonal
        (ideal); one means all rows are parallel (full collapse).
    """
    if parity_weights_masked.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {tuple(parity_weights_masked.shape)}")
    M, _ = parity_weights_masked.shape
    if M < 2:
        return parity_weights_masked.new_zeros(())
    h = parity_weights_masked.float()
    norms = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    h_unit = h / norms
    # Gram matrix of unit rows; diagonal is 1, off-diagonal is cos(θ).
    gram = h_unit @ h_unit.t()
    eye = torch.eye(M, device=gram.device, dtype=gram.dtype)
    off_diag = (gram - eye).abs()
    # Mean of upper-triangular entries (symmetric matrix → take half).
    triu_mask = torch.triu(torch.ones_like(gram, dtype=torch.bool), diagonal=1)
    if not triu_mask.any():
        return parity_weights_masked.new_zeros(())
    return off_diag[triu_mask].mean()
