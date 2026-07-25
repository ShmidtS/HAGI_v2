"""TransformerBlock — pre-norm GQA attention + (Hebbian bilinear FFN | MoE).

A single KV-cacheable block. Selects its mixer per block: every
``moe_every``-th block uses :class:`MoESwiGLU` when MoE is enabled, else
:class:`HebbianBilinearFFN`. Attention mode is selected per call so one stack
supports masked training and causal AR generation.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi.model.attention import Attention, AttentionConfig
from hagi.model.hebbian_ffn import HebbianBilinearFFN, HebbianFFNConfig
from hagi.model.moe import MoESwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm attention + FFN/MoE mixer with residual.

    Args:
        hidden_size: H.
        attn_cfg: attention config.
        ffn_cfg: Hebbian FFN config (expansion).
        norm_eps: RMSNorm epsilon.
        use_ternary: ternarize 2D weights via BitLinear.
        mixer: the block's channel mixer (FFN or MoE).
    """

    def __init__(
        self,
        hidden_size: int,
        attn_cfg: AttentionConfig,
        ffn_cfg: HebbianFFNConfig,
        norm_eps: float = 1e-6,
        use_ternary: bool = True,
        mixer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.attn = Attention(hidden_size, attn_cfg, norm_eps, use_ternary=use_ternary)
        self.ffn = mixer if mixer is not None else HebbianBilinearFFN(hidden_size, ffn_cfg, norm_eps, use_ternary=use_ternary)
        self._last_attn_entropy_penalty: torch.Tensor | None = None

    def set_attn_entropy_floor(self, floor: float) -> None:
        self.attn.set_attn_entropy_floor(floor)

    @property
    def is_moe(self) -> bool:
        return isinstance(self.ffn, MoESwiGLU)

    @property
    def moe(self) -> MoESwiGLU | None:
        return self.ffn if self.is_moe else None

    def forward(
        self,
        x: torch.Tensor,
        attention_mode: str = "causal",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, pen = self.attn(x, attention_mode, prefix_len, soft_beta, positions)
        x = x + attn_out
        self._last_attn_entropy_penalty = pen
        x = self.ffn(x)
        return x
