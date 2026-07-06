"""Standard transformer block: RMSNorm -> GQA -> RMSNorm -> MoE.

V4: GQA is bidirectional (no causal mask) for plane prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi_v4.config import ModelConfig
from hagi_v4.model.attention import GroupedQueryAttention
from hagi_v4.model.moe import MoESwiGLU
from hagi_v4.model.norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        m = cfg
        self.attn_norm = RMSNorm(m.hidden_size, m.norm_eps, fp32_variance=m.attention.fp32_rmsnorm)
        self.attn = GroupedQueryAttention(
            hidden_size=m.hidden_size,
            num_q_heads=m.attention.num_query_heads,
            num_kv_heads=m.attention.num_kv_heads,
            head_dim=m.attention.head_dim,
            rope_theta=m.attention.rope_theta,
            bidirectional=m.attention.bidirectional,
            fp16_attention=m.attention.fp16_attention,
        )
        self.ffn_norm = RMSNorm(m.hidden_size, m.norm_eps, fp32_variance=m.attention.fp32_rmsnorm)
        self.moe = MoESwiGLU(cfg=m.moe, hidden_size=m.hidden_size)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        moe_out, aux, router_probs = self.moe(self.ffn_norm(x))
        return x + moe_out, aux, router_probs
