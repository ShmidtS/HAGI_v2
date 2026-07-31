"""Transformer block: attention branch + mixer branch, both pre-norm.

Two residual branches per layer. The residual stream is the channel state; each
branch adds an increment to it. Pre-norm (normalize the branch *input*, leave the
stream untouched) is what makes a deep stack trainable — the stream stays an
unmodified sum of increments, so the gradient reaches layer 0 without passing
through L normalizations.

Depth scaling: every branch's output projection is initialized with
``1/sqrt(2L)``. With 2L branches each contributing variance ``s^2``, the stream's
variance at the output is ``1 + 2L*s^2``, so ``s = 1/sqrt(2L)`` keeps it at O(1)
independent of depth. Skipping this makes the first phase of training a search
for smaller output weights instead of a search for structure.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi.model.attention import Attention, AttentionConfig
from hagi.model.ffn import FeedForward
from hagi.model.moe import MoE


class Block(nn.Module):
    """One layer: ``x + attn(x)`` then ``x + mixer(x)``.

    Args:
        hidden_size: H.
        attn_cfg: attention geometry for this layer (window included).
        mixer: the channel mixer — :class:`FeedForward` or :class:`MoE`.
        norm_eps: RMSNorm epsilon.
        use_ternary: quantize the 2D weights.
        residual_scale: init scale for both branches' output projections.
    """

    def __init__(
        self,
        hidden_size: int,
        attn_cfg: AttentionConfig,
        mixer: nn.Module,
        norm_eps: float = 1e-5,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.attn = Attention(hidden_size, attn_cfg, norm_eps, use_ternary, residual_scale)
        self.mixer = mixer

    @property
    def is_moe(self) -> bool:
        return isinstance(self.mixer, MoE)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(x, positions, mask)
        return x + self.mixer(x)


def build_mixer(
    hidden_size: int,
    intermediate_size: int,
    moe_cfg,
    use_moe: bool,
    norm_eps: float,
    use_ternary: bool,
    residual_scale: float,
) -> nn.Module:
    """Construct the layer's mixer: MoE when selected, dense SwiGLU otherwise."""
    if use_moe:
        return MoE(hidden_size, intermediate_size, moe_cfg, norm_eps, use_ternary, residual_scale)
    return FeedForward(hidden_size, intermediate_size, norm_eps, use_ternary, residual_scale)
