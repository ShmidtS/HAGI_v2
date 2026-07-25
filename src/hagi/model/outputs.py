"""Typed model outputs — auxiliary loss terms for the codec-channel LM.

Every auxiliary is computed off the main LM path. The genuine rate/distortion
terms come from the information bottleneck; vicreg/infonce ground the
multimodal joint embedding; moe_lb balances expert load; attn_entropy
prevents attention collapse. A term is ``None`` when its subsystem is off.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AuxLosses:
    """Auxiliary loss terms produced by the model forward pass.

    All terms are ``None`` when their subsystem is inactive. The aggregator
    weights them by the matching ``w_*`` config value.
    """

    # Information bottleneck (always on).
    rate: torch.Tensor | None = None
    distortion: torch.Tensor | None = None
    # Grounded infomax (multimodal only).
    vicreg: torch.Tensor | None = None
    infonce: torch.Tensor | None = None
    # Mixture of experts (MoE only).
    moe_lb: torch.Tensor | None = None
    # Attention anti-collapse (training only).
    attn_entropy: torch.Tensor | None = None


@dataclass
class ModelOutput:
    """Unified output from model forward pass (training and inference)."""

    logits: torch.Tensor | None
    hidden: torch.Tensor
    aux: AuxLosses
    ce_loss: torch.Tensor | None = None
    prediction_indices: torch.Tensor | None = None
