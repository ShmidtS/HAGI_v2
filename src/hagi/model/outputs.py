"""Typed output of the language and optional grounding objectives."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ModelOutput:
    """Result of a forward pass.

    Attributes:
        loss: total scalar objective, or None when no targets were supplied.
        ce: training receiver cost in nats/token. With sampled softmax this is
            a local conditional-NCE partition, not the full coding cost.
        exact_ce: optional exact full-alphabet CE diagnostic on sampled rows.
        z_loss: mean squared log-partition of the LM head (unweighted).
        grounding: cross-modal InfoNCE plus anti-collapse terms (multimodal only).
        hidden: final normalized hidden states, ``[B, T, H]``.
        logits: only populated for generation/diagnostics — training never
            materializes an ``[N, V]`` tensor.
        n_tokens: number of scored positions, for correct averaging across
            gradient-accumulation microbatches of unequal size.
        diagnostics: scalar observables such as QK and receiver gains.
    """

    loss: torch.Tensor | None = None
    ce: torch.Tensor | None = None
    exact_ce: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None
    grounding: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    n_tokens: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)
