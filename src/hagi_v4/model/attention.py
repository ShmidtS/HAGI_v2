"""Grouped Query Attention with RoPE, fp16 attention, and bidirectional support.

V4 key difference: bidirectional attention (no causal mask) for plane
prediction. The causal mask is still available via the bidirectional=False
flag for potential autoregressive fallback.

Fused QKV projection, RoPE position encoding, and optional fp16 cast
for SDPA softmax (8x better resolution than bf16, zero speed cost on Ampere).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi_v4.model.norms import apply_rope


class GroupedQueryAttention(nn.Module):
    """GQA with fused QKV projection and bidirectional support.

    Args:
        hidden_size: Input/output dimension.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of KV heads (must divide num_q_heads).
        head_dim: Dimension per head.
        rope_theta: Base frequency for RoPE.
        bidirectional: If True, no causal mask (plane prediction).
        fp16_attention: Cast bf16 Q,K,V to fp16 for SDPA softmax.
    """

    def __init__(
        self,
        hidden_size: int = 576,
        num_q_heads: int = 8,
        num_kv_heads: int = 4,
        head_dim: int = 72,
        rope_theta: float = 10000.0,
        bidirectional: bool = True,
        fp16_attention: bool = True,
    ):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "query heads must be divisible by kv heads"
        self.n_q = num_q_heads
        self.n_kv = num_kv_heads
        self.head_dim = head_dim
        self.repeat = num_q_heads // num_kv_heads
        self.bidirectional = bidirectional
        self.fp16_attention = fp16_attention

        qkv_out = (num_q_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_size, qkv_out, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        orig_dtype = x.dtype

        qkv = self.qkv_proj(x)
        q_size = self.n_q * self.head_dim
        kv_size = self.n_kv * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(B, T, self.n_q, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        use_fp16 = self.fp16_attention and orig_dtype == torch.bfloat16 and q.is_cuda
        if use_fp16:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        if self.repeat > 1:
            k = (
                k.unsqueeze(2)
                .expand(B, self.n_kv, self.repeat, T, self.head_dim)
                .reshape(B, self.n_q, T, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(B, self.n_kv, self.repeat, T, self.head_dim)
                .reshape(B, self.n_q, T, self.head_dim)
            )

        is_causal = not self.bidirectional
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        if use_fp16:
            out = out.to(orig_dtype)

        out = out.transpose(1, 2).reshape(B, T, self.n_q * self.head_dim)
        return self.o_proj(out)
