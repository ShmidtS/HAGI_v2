"""Attention blocks for HAGI V20 Source Encode/Decode.

V20: support bidirectional, causal, and prefix-LM attention modes.
- bidir: full attention (BERT-style, for masked recovery)
- causal: lower-triangular mask (GPT-style, for AR generation)
- prefix: bidir on prefix tokens, causal on suffix tokens (GLM-style)

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
    """Pre-norm MHA + RoPE + SwiGLU with selectable attention mode.

    V20 introduces the ``attention_mode`` argument to support three regimes
    in a single stack:

    - ``"bidir"``: full attention (BERT-style, for masked recovery training)
    - ``"causal"``: lower-triangular mask (GPT-style, for AR generation)
    - ``"prefix"``: bidir on prefix tokens, causal on suffix tokens
      (GLM-style). Requires ``prefix_len`` (int or [B] tensor) so the block
      knows where bidir ends and causal begins.

    The mode is selected per forward call, not at construction time, so the
    same block can be used for bidirectional perception during masked-CE
    training and causal generation during AR inference.
    """

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

        # V22: attention entropy floor for distribution stabilization.
        # Prevents attention collapse to a single position (which causes
        # cyclic loss oscillation and entropy → 0).
        self.attn_entropy_floor: float = 0.0  # 0 = disabled

    def set_attn_entropy_floor(self, floor: float) -> None:
        """V22: set minimum attention entropy. Prevents collapse."""
        self.attn_entropy_floor = float(floor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run attention + FFN with the requested attention mode.

        Args:
            x: [B, T, H] hidden states.
            attention_mode: one of "bidir", "causal", "prefix", "soft_causal".
            prefix_len: required when attention_mode == "prefix". Scalar int
                or [B] tensor giving the number of prefix tokens per sample
                (those positions get full bidirectional attention; positions
                >= prefix_len get causal attention over all preceding tokens
                including the prefix).
            soft_beta: controls soft_causal sharpness. None defaults to 2.0.
                0 = full bidir, large = strict causal.
            return_attn_weights: when True, returns (output, attn_weights).
        """
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

        is_causal = False
        attn_mask: torch.Tensor | None = None

        if attention_mode == "bidir":
            is_causal = False
        elif attention_mode == "causal":
            is_causal = True
        elif attention_mode == "prefix":
            # Build a per-sample mask: prefix positions attend to all prefix
            # positions bidirectionally; suffix positions attend causally to
            # everything up to and including themselves. Implementation: start
            # from a causal mask (lower-triangular), then open up the prefix
            # block to full attention.
            # attn_mask shape: [B, 1, T, T] additive mask (-inf = blocked).
            if prefix_len is None:
                raise ValueError("prefix_len is required when attention_mode='prefix'")
            if isinstance(prefix_len, int):
                pl = torch.full((B,), prefix_len, device=x.device, dtype=torch.long)
            else:
                pl = prefix_len.to(device=x.device, dtype=torch.long)
            idx = torch.arange(T, device=x.device)
            q_idx = idx.view(T, 1)
            k_idx = idx.view(1, T)
            causal_allowed = k_idx <= q_idx  # [T, T]
            mask = torch.zeros(B, 1, T, T, device=x.device, dtype=x.dtype)
            mask.masked_fill_(~causal_allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
            # Open prefix block: for each sample, positions < pl[b] can attend
            # to all positions < pl[b] bidirectionally.
            pl_b = pl.view(B, 1, 1, 1)
            q_in_prefix = idx.view(1, 1, T, 1) < pl_b
            k_in_prefix = idx.view(1, 1, 1, T) < pl_b
            both_prefix = q_in_prefix & k_in_prefix  # [B, 1, T, T]
            mask = mask.masked_fill(both_prefix, 0.0)
            attn_mask = mask
            is_causal = False
        elif attention_mode == "soft_causal":
            # V22: soft causal blending — smooth transition between bidir and causal.
            # Future positions get a penalty proportional to distance, not -inf.
            # soft_beta controls sharpness: 0=full bidir, large=strict causal.
            if soft_beta is None:
                soft_beta = 2.0
            idx = torch.arange(T, device=x.device)
            dist = (idx.view(1, T) - idx.view(T, 1)).clamp(min=0)  # [T, T], 0 for past/self
            penalty = -soft_beta * dist.float().to(x.dtype)
            attn_mask = penalty.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)
            is_causal = False
        else:
            raise ValueError(
                f"attention_mode must be one of 'bidir', 'causal', 'prefix', 'soft_causal'; got {attention_mode!r}"
            )

        if self.attn_entropy_floor > 0.0 or return_attn_weights:
            # Manual attention computation (needed for entropy regularization)
            scale = 1.0 / (self.head_dim**0.5)
            if attn_mask is not None:
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_mask
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if is_causal:
                causal_mask = torch.triu(
                    torch.full((T, T), float("-inf"), device=q.device, dtype=scores.dtype), diagonal=1
                )
                scores = scores + causal_mask
            attn_weights = torch.softmax(scores, dim=-1)  # [B, n_heads, T, T]

            # V22: attention entropy regularization
            if self.attn_entropy_floor > 0.0 and self.training:
                log_attn = torch.log(attn_weights + 1e-8)
                entropy = -(attn_weights * log_attn).sum(dim=-1)  # [B, n_heads, T]
                entropy_deficit = (self.attn_entropy_floor - entropy).clamp_min(0.0)
                self._last_attn_entropy_penalty = entropy_deficit.mean()
            else:
                self._last_attn_entropy_penalty = None

            attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
            if return_attn_weights:
                x = x + self.out_proj(attn_out)
                x = x + self.ffn(self.ffn_norm(x))
                return x, attn_weights
            x = x + self.out_proj(attn_out)
            x = x + self.ffn(self.ffn_norm(x))
            return x

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=drop_p,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        x = x + self.out_proj(attn)
        x = x + self.ffn(self.ffn_norm(x))
        return x
