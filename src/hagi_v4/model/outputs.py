"""Typed outputs and stateless loss helpers for HAGI V7."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AuxLosses:
    """Auxiliary loss terms produced by the model forward pass."""

    whiteness: torch.Tensor | None = None
    msa_lb: torch.Tensor | None = None
    parity: torch.Tensor | None = None
    extrinsic_info: torch.Tensor | None = None
    efficiency: torch.Tensor | None = None
    rate_distortion: torch.Tensor | None = None
    contrastive: torch.Tensor | None = None


@dataclass
class ModelOutput:
    """Unified output from model forward pass (training and inference)."""

    logits: torch.Tensor | None
    hidden: torch.Tensor
    aux: AuxLosses
    ce_loss: torch.Tensor | None = None
    iterations_used: torch.Tensor | None = None


def compute_whiteness_loss(residual: torch.Tensor) -> torch.Tensor:
    if residual is None or residual.size(1) < 2:
        return residual.new_zeros(()) if residual is not None else torch.tensor(0.0)
    r_t = residual[:, :-1].reshape(-1, residual.size(-1))
    r_t1 = residual[:, 1:].reshape(-1, residual.size(-1))
    cos_sim = F.cosine_similarity(r_t.float(), r_t1.float(), dim=-1)
    return cos_sim.abs().mean()
