"""Typed outputs and stateless loss helpers for HAGI V4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TrainOutput:
    """Output from training forward pass."""

    loss: torch.Tensor
    moe_aux_loss: torch.Tensor
    gdr_router_loss: torch.Tensor
    coherence_loss: torch.Tensor
    deep_supervision_loss: torch.Tensor | None
    hidden: torch.Tensor
    mask: torch.Tensor


@dataclass
class InferenceOutput:
    """Output from inference forward pass."""

    logits: torch.Tensor
    hidden: torch.Tensor
    iterations_used: torch.Tensor


@dataclass
class AuxLosses:
    """Auxiliary loss terms produced by the model forward pass."""

    coherence: torch.Tensor | None = None
    whiteness: torch.Tensor | None = None
    grade_spec: torch.Tensor | None = None
    ib: torch.Tensor | None = None
    deep_supervision: torch.Tensor | None = None
    moe_lb: torch.Tensor | None = None
    msa_lb: torch.Tensor | None = None
    gdr_router: torch.Tensor | None = None
    parity: torch.Tensor | None = None
    extrinsic_info: torch.Tensor | None = None
    efficiency: torch.Tensor | None = None


@dataclass
class ModelOutput:
    """Unified output from model forward pass (training and inference)."""

    logits: torch.Tensor | None
    hidden: torch.Tensor
    aux: AuxLosses
    ce_loss: torch.Tensor | None = None
    iterations_used: torch.Tensor | None = None


@dataclass
class RefinementSideInfo:
    """Side information collected during the refinement loop."""

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


def compute_whiteness_loss(residual: torch.Tensor) -> torch.Tensor:
    if residual is None or residual.size(1) < 2:
        return residual.new_zeros(()) if residual is not None else torch.tensor(0.0)
    r_t = residual[:, :-1].reshape(-1, residual.size(-1))
    r_t1 = residual[:, 1:].reshape(-1, residual.size(-1))
    cos_sim = F.cosine_similarity(r_t.float(), r_t1.float(), dim=-1)
    return cos_sim.abs().mean()


def compute_grade_spec_loss(
    gate_probs: torch.Tensor,
    router_probs: torch.Tensor,
    num_experts: int = 4,
) -> torch.Tensor:
    if gate_probs is None or router_probs is None:
        return torch.tensor(0.0)
    probs = router_probs[:, :num_experts]
    gate_flat = gate_probs.reshape(-1, gate_probs.size(-1)) if gate_probs.dim() > 2 else gate_probs
    if gate_flat.shape[0] != probs.shape[0]:
        n = min(gate_flat.shape[0], probs.shape[0])
        gate_flat = gate_flat[:n]
        probs = probs[:n]
    target = gate_flat[:, :num_experts]
    return F.cross_entropy(probs, target.argmax(dim=-1))
