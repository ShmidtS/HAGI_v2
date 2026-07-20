"""Bidirectional attention blocks for HAGI V16 Source Encode/Decode.

FreqBlock is demoted to the channel equalizer (LDPCDecoder.reasoning).
Source stacks use content-addressable MHA + SwiGLU + RoPE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


def _swiglu_intermediate(hidden_size: int, ffn_mult: float = 4.0) -> int:
    """SwiGLU intermediate size: ~ (2/3)*mult*H, rounded up to 64."""
    raw = int(hidden_size * ffn_mult * 2 / 3)
    return max(64, ((raw + 63) // 64) * 64)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q,k shaped [B, n_heads, T, head_dim]. cos/sin: [T, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    """Standard RoPE cache: cos/sin by (T, device, dtype)."""

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_cos_sin(self, T: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (T, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = (cos, sin)
        return cos, sin


class SwiGLU(nn.Module):
    """SwiGLU FFN: Linear(H, 2*inter) -> silu*chunk -> Linear(inter, H)."""

    def __init__(self, hidden_size: int, ffn_mult: float = 4.0) -> None:
        super().__init__()
        inter = _swiglu_intermediate(hidden_size, ffn_mult)
        self.w_gate = nn.Linear(hidden_size, 2 * inter, bias=False)
        self.w_out = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * up)


class AttentionBlock(nn.Module):
    """Pre-norm bidirectional MHA + RoPE + SwiGLU. Content-addressable source mixer."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        ffn_mult: float = 4.0,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by n_heads={n_heads}")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.dropout = dropout

        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta=rope_theta)

        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn = SwiGLU(hidden_size, ffn_mult=ffn_mult)
        nn.init.normal_(self.ffn.w_out.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        B, T, _ = h.shape
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope.get_cos_sin(T, q.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        drop_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        x = x + self.out_proj(attn)
        x = x + self.ffn(self.ffn_norm(x))
        return x
