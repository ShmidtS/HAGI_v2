"""Typed model outputs — auxiliary loss terms for the ternary RD-channel LM."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AuxLosses:
    """Auxiliary loss terms produced by the model forward pass.

    All terms are ``None`` when inactive. The genuine RD terms (rate/distortion/
    perception) come from the auxiliary information bottleneck; the attention
    entropy penalty prevents attention collapse; ternary_bias and moe_lb are
    opt-in regularizers.
    """

    rate: torch.Tensor | None = None
    distortion: torch.Tensor | None = None
    perception: torch.Tensor | None = None
    attn_entropy: torch.Tensor | None = None
    ternary_bias: torch.Tensor | None = None
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
