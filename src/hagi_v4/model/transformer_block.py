"""Standard transformer block: RMSNorm -> GQA -> RMSNorm -> MoE.

V4: GQA is bidirectional (no causal mask) for plane prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi_v4.config import ModelConfig, MoEConfig
from hagi_v4.model.attention import GroupedQueryAttention
from hagi_v4.model.moe import MoESwiGLU
from hagi_v4.model.norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden_size: int | None = None):
        super().__init__()
        m = cfg
        H = hidden_size if hidden_size is not None else m.hidden_size
        nh = m.attention.num_query_heads
        hd = m.attention.head_dim
        if H < nh * hd:
            nh = max(1, H // hd)
        self.attn_norm = RMSNorm(H, m.norm_eps, fp32_variance=m.attention.fp32_rmsnorm)
        self.attn = GroupedQueryAttention(
            hidden_size=H,
            num_q_heads=nh,
            num_kv_heads=max(1, nh // 2),
            head_dim=hd,
            rope_theta=m.attention.rope_theta,
            bidirectional=m.attention.bidirectional,
            fp16_attention=m.attention.fp16_attention,
        )
        i_size = max(64, m.moe.intermediate_size * H // m.hidden_size)
        moe_cfg = MoEConfig(
            num_experts=m.moe.num_experts,
            top_k=m.moe.top_k,
            intermediate_size=i_size,
            use_mod_skip=m.moe.use_mod_skip,
            alpha=m.moe.alpha,
            use_grade_specialization=m.moe.use_grade_specialization,
            grade_specialization_weight=m.moe.grade_specialization_weight,
            n_shared_bases=m.moe.n_shared_bases,
        )
        self.ffn_norm = RMSNorm(H, m.norm_eps, fp32_variance=m.attention.fp32_rmsnorm)
        self.moe = MoESwiGLU(cfg=moe_cfg, hidden_size=H)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        moe_out, aux, router_probs = self.moe(self.ffn_norm(x))
        return x + moe_out, aux, router_probs
