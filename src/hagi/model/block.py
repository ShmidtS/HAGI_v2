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
        clamp_ratio: float = 2.0,
        attn_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # backbone="recurrent": вместо GQA-ветки сюда передаётся готовый
        # Recurrent (drop-in: forward(x, positions, mask) -> [B,T,H]). Слот
        # остаётся с именем ``attn`` — все точки модели (depth_grow, adapters,
        # allocate_cache) адресуют миксер через него.
        self.attn: nn.Module = attn_module if attn_module is not None else Attention(
            hidden_size,
            attn_cfg,
            norm_eps,
            use_ternary,
            residual_scale,
            init_orthogonal,
            rope=rope,
            clamp_ratio=clamp_ratio,
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
