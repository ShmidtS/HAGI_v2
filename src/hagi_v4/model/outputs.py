"""Typed outputs for HAGI V4 training and inference.

V4 uses masked CE (not next-token), so TrainOutput carries mask info.
InferenceOutput tracks per-token iterations for adaptive halting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
