"""Transformer block: attention branch + dense mixer branch, both pre-norm."""

from __future__ import annotations

import torch
from torch import nn

from hagi.model.attention import Attention, AttentionConfig


class Block(nn.Module):
    """One channel layer: ``x + attention(x)`` then ``x + mixer(x)``."""

    def __init__(
        self,
        hidden_size: int,
        attn_cfg: AttentionConfig,
        mixer: nn.Module,
        norm_eps: float = 1e-5,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
        init_orthogonal: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        self.attn = Attention(
            hidden_size,
            attn_cfg,
            norm_eps,
            use_ternary,
            residual_scale,
            init_orthogonal,
            rope=rope,
        )
        self.mixer = mixer

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(x, positions, mask)
        return x + self.mixer(x)
