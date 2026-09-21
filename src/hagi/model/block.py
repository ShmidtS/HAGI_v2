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
        # Opt-in residual adapters (pyramid / TTT-LoRA). Only set by HAGI when
        # cfg.model.adapters.enabled is True; stays None otherwise so the
        # default forward is reproduced bit-for-bit.
        self.adapters: nn.Module | None = None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Capture the exact residual stream that the base mixer receives.
        # The base attention and mixer keep their original ordering; adapters
        # run only after both updates and receive the same mixer input.
        x = x + self.attn(x, positions, mask)
        mixer_input = x
        x = x + self.mixer(x)
        adapter = self.adapters
        if adapter is not None:
            x = x + adapter(mixer_input, x)
        return x
