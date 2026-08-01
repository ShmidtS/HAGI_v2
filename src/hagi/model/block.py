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
from hagi.model.spectral import SpectralRecurrence


class Block(nn.Module):
    """One layer: ``x + attn(x)`` then ``x + spectral(x)`` (if selected) then ``x + mixer(x)``.

    The spectral branch is optional and adds its increment between attention and
    the mixer. It is a *parallel* path, not a replacement: attention handles
    content-addressable retrieval, the spectral branch handles frequency-local
    structure, and the mixer the per-position nonlinear map.

    Args:
        hidden_size: H.
        attn_cfg: attention geometry for this layer (window included).
        mixer: the channel mixer — :class:`FeedForward` or :class:`MoE`.
        norm_eps: RMSNorm epsilon.
        use_ternary: quantize the 2D weights.
        residual_scale: init scale for both branches' output projections.
        spectral_cfg: spectral configuration, or None for no spectral branch.
        use_spectral: whether this layer carries the spectral branch.
    """

    def __init__(
        self,
        hidden_size: int,
        attn_cfg: AttentionConfig,
        mixer: nn.Module,
        norm_eps: float = 1e-5,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
        spectral_cfg=None,
        use_spectral: bool = False,
        init_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.attn = Attention(hidden_size, attn_cfg, norm_eps, use_ternary, residual_scale, init_orthogonal)
        self.mixer = mixer
        self.spectral = None
        if use_spectral and spectral_cfg is not None:
            from hagi.config import SpectralConfig

            if not isinstance(spectral_cfg, SpectralConfig):
                raise TypeError("spectral_cfg must be a SpectralConfig when use_spectral=True")
            self.spectral = SpectralRecurrence(hidden_size, spectral_cfg, norm_eps, use_ternary, residual_scale, init_orthogonal)

    @property
    def is_moe(self) -> bool:
        return isinstance(self.mixer, MoE)

    @property
    def has_spectral(self) -> bool:
        return self.spectral is not None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_spectral_state: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(x, positions, mask)
        if self.spectral is not None:
            x = x + self.spectral(x, use_state=use_spectral_state)
        return x + self.mixer(x)


def build_mixer(
    hidden_size: int,
    intermediate_size: int,
    moe_cfg,
    use_moe: bool,
    norm_eps: float,
    use_ternary: bool,
    residual_scale: float,
    init_orthogonal: bool = False,
) -> nn.Module:
    """Construct the layer's mixer: MoE when selected, dense SwiGLU otherwise."""
    if use_moe:
        return MoE(hidden_size, intermediate_size, moe_cfg, norm_eps, use_ternary, residual_scale, init_orthogonal)
    return FeedForward(hidden_size, intermediate_size, norm_eps, use_ternary, residual_scale, init_orthogonal)
