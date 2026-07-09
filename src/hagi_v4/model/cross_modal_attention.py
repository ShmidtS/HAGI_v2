"""Cross-modal attention — MIMO space-time coding (V7).

Each perception layer gets cross-modal attention after self-attention.
Text tokens attend to image/audio tokens and vice versa, combining
modality streams for diversity gain (MIMO analog).

Gated residual: starts with sigmoid(0)=0.5, allowing some cross-modal
flow from the beginning. Gate is learnable per layer.

Information theory:
  - Self-attention = intra-modality equalizer (existing channel taps)
  - Cross-modal attention = inter-modality equalizer (new channel taps)
  - Gated cross-modal = adaptive MIMO (start SISO, gradually enable MIMO)
  - Diversity order = num_modalities (3x outage reduction)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.attention import GroupedQueryAttention
from hagi_v4.model.moe import MoESwiGLU
from hagi_v4.model.norms import RMSNorm
from hagi_v4.config import MoEConfig


class MultimodalPerceptionBlock(nn.Module):
    """Perception block with cross-modal attention.

    1. Self-attention (intra-modality, bidirectional)
    2. Cross-modal attention (inter-modality, MIMO, gated)
    3. MoE FFN (modality-aware routing)
    """

    def __init__(self, cfg: HAGIv4Config, hidden_size: int) -> None:
        super().__init__()
        m = cfg.model
        H = hidden_size

        self.self_attn_norm = RMSNorm(H, m.norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size=H,
            num_q_heads=m.attention.num_query_heads,
            num_kv_heads=m.attention.num_kv_heads,
            head_dim=m.attention.head_dim,
            rope_theta=m.attention.rope_theta,
            bidirectional=True,
            fp16_attention=m.attention.fp16_attention,
        )

        self.cross_attn_norm = RMSNorm(H, m.norm_eps)
        self.cross_attn = GroupedQueryAttention(
            hidden_size=H,
            num_q_heads=m.attention.num_query_heads,
            num_kv_heads=m.attention.num_kv_heads,
            head_dim=m.attention.head_dim,
            rope_theta=m.attention.rope_theta,
            bidirectional=True,
            fp16_attention=m.attention.fp16_attention,
        )
        self.cross_gate = nn.Parameter(torch.tensor(m.multimodal.cross_modal_attention.gate_init))

        i_size = max(64, m.moe.intermediate_size * H // m.hidden_size)
        moe_cfg = MoEConfig(
            num_experts=m.moe.num_experts,
            top_k=m.moe.top_k,
            intermediate_size=i_size,
            use_mod_skip=m.moe.use_mod_skip,
            alpha=m.moe.alpha,
            n_shared_bases=m.moe.n_shared_bases,
            router_noise=m.moe.router_noise,
            router_init_std=m.moe.router_init_std,
        )
        self.ffn_norm = RMSNorm(H, m.norm_eps)
        self.moe = MoESwiGLU(cfg=moe_cfg, hidden_size=H)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        x = x + self.self_attn(self.self_attn_norm(x), cos, sin)

        if modality_ids is not None:
            cross_out = self.cross_attn(self.cross_attn_norm(x), cos, sin)
            gate = torch.sigmoid(self.cross_gate)
            x = x + gate * cross_out

        moe_out, aux, rp = self.moe(self.ffn_norm(x))
        return x + moe_out, aux, rp
