"""V25 ternary TransformerBlock (ARCHITECTURE_V25 §2 STAGE2/STAGE5, §5 ternary).

The ternary version of the channel body block. Only the 2D hidden mixing
matrices are ternarized via ``BitLinear`` (BitNet b1.58, per-output-channel
absmean scale, identity STE -- see ``ternary.py``): attention ``qkv`` /
``out_proj`` and the Hebbian bilinear FFN ``A0`` / ``A1`` / ``W``. Everything
that must stay FP is untouched: ``RMSNorm`` gains, ``RoPE`` inv_freq,
``attn_norm``, the entropy-floor math, the FFN ``gate`` (1D Parameter). At
``use_ternary=False`` the block is a faithful dense port (the FP masters flow
to Muon either way; ``is_muon_param`` already selects 2D weights).

This block is a drop-in for ``HAGIv4._stack_forward``: the
``TransformerBlock.forward`` signature and single-tensor return MUST NOT
change (grad-checkpointing depends on it), and the attention-entropy-floor
penalty is cached on ``block._last_attn_entropy_penalty`` so the stack sums
it into ``aux.attn_entropy``.

Reuses (NOT net-new): ``AttentionConfig`` / ``HebbianFFNConfig``,
``RotaryEmbedding``, ``apply_rope``, the prefix / soft-causal mask builders,
and ``RMSNorm``. Net-new: the ternary ``Attention``, ``HebbianBilinearFFN``,
and ``TransformerBlock`` here, plus ``ternary.BitLinear``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.block import AttentionConfig
from hagi_v4.model.block import HebbianFFNConfig
from hagi_v4.model.block import RotaryEmbedding
from hagi_v4.model.block import apply_rope
from hagi_v4.model.block import _build_prefix_mask
from hagi_v4.model.block import _build_soft_causal_mask
from hagi_v4.model.norms import RMSNorm
from hagi_v4.model.ternary import BitLinear


def _proj(hidden_size_in: int, hidden_size_out: int, bias: bool, use_ternary: bool) -> nn.Module:
    """Pick a 2D hidden-weight projection: BitLinear (ternary) or nn.Linear (FP).

    Only 2D hidden mixing weights are ternarized. The same call site is used
    for qkv / out_proj / A0 / A1 / W -- all 2D, all eligible for Muon.
    """
    if use_ternary:
        return BitLinear(hidden_size_in, hidden_size_out, bias=bias)
    return nn.Linear(hidden_size_in, hidden_size_out, bias=bias)


class Attention(nn.Module):
    """Pre-norm MHA + RoPE with bidir / causal / prefix / soft_causal modes.

    Args:
        hidden_size: H.
        cfg: ``AttentionConfig`` (num_heads, head_dim, rope_theta,
            attn_entropy_floor).
        norm_eps: RMSNorm epsilon.
        use_ternary: when True, ``qkv`` / ``out_proj`` are ``BitLinear`` (2D
            ternary masters). When False, plain ``nn.Linear``. The RMSNorm
            gain, RoPE inv_freq, attn_norm, and entropy-floor math stay FP
            regardless.
    """

    def __init__(self, hidden_size: int, cfg: AttentionConfig, norm_eps: float = 1e-6, use_ternary: bool = True) -> None:
        super().__init__()
        if hidden_size % cfg.num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={cfg.num_heads}")
        self.hidden_size = hidden_size
        self.n_heads = cfg.num_heads
        self.head_dim = hidden_size // cfg.num_heads
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        # 2D hidden weights -> ternary when use_ternary. qkv: H -> 3H, out: H -> H.
        self.qkv = _proj(hidden_size, 3 * hidden_size, bias=False, use_ternary=use_ternary)
        self.out_proj = _proj(hidden_size, hidden_size, bias=False, use_ternary=use_ternary)
        # Match the V24 dense init on out_proj (small, so the residual branch
        # starts near-identity). BitLinear already inits N(0, 0.02); re-applying
        # is consistent and a no-op in effect there.
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta=cfg.rope_theta)
        self.attn_entropy_floor = float(cfg.attn_entropy_floor)

    def set_attn_entropy_floor(self, floor: float) -> None:
        self.attn_entropy_floor = float(floor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.attn_norm(x)
        B, T, _ = h.shape
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope.get_cos_sin(T, q.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        is_causal = False
        attn_mask = None
        if attention_mode == "bidir":
            pass
        elif attention_mode == "causal":
            is_causal = True
        elif attention_mode == "prefix":
            attn_mask = _build_prefix_mask(B, T, prefix_len, x.device, x.dtype)
        elif attention_mode == "soft_causal":
            beta = 2.0 if soft_beta is None else soft_beta
            attn_mask = _build_soft_causal_mask(T, beta, x.device, x.dtype)
        else:
            raise ValueError(f"unknown attention_mode {attention_mode!r}")

        scale = 1.0 / (self.head_dim**0.5)
        entropy_pen = None
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)

        # Attention-entropy floor penalty (anti-collapse), kept fully FP.
        # Only materialized when active and training (the SDPA path above is
        # fused/Flash). The penalty is returned so the block can cache it for
        # the stack to sum into aux.attn_entropy.
        if self.attn_entropy_floor > 0.0 and self.training:
            scores = (q @ k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask
            if is_causal:
                scores = scores + torch.triu(
                    torch.full((T, T), float("-inf"), device=q.device, dtype=scores.dtype), diagonal=1
                )
            attn_weights = torch.softmax(scores, dim=-1)
            entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1)
            entropy_pen = (self.attn_entropy_floor - entropy).clamp_min(0.0).mean()

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.out_proj(attn_out), entropy_pen


class HebbianBilinearFFN(nn.Module):
    """Ternary Hebbian bilinear FFN: phi(h)=(A0 h) (.) silu(A1 h); out=W(phi)*(1+tanh(gate)).

    A0, A1 are R^{m x H} (m = expansion*H) and W is R^{H x m} -- all 2D, all
    eligible for BitLinear and Muon. The ``gate`` is a 1D Parameter and stays
    FP (LayerScale): ``(1+tanh(gate))`` is 1 at gate=0 so the FFN branch is
    live from step 0 (the V18 dead-gradient fix -- a zero-init multiplier
    would starve A0/A1/W of gradient).

    Args:
        hidden_size: H.
        cfg: ``HebbianFFNConfig`` (expansion, dropout).
        norm_eps: RMSNorm epsilon.
        use_ternary: when True, ``A0`` / ``A1`` / ``W`` are ``BitLinear``.
    """

    def __init__(self, hidden_size: int, cfg: HebbianFFNConfig, norm_eps: float = 1e-6, use_ternary: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        m = cfg.expansion * hidden_size
        self.m = m
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        # Bilinear feature map A0, A1 in R^{m x H}; readout W in R^{H x m}.
        # All 2D -> ternary masters when use_ternary.
        self.A0 = _proj(hidden_size, m, bias=False, use_ternary=use_ternary)
        self.A1 = _proj(hidden_size, m, bias=False, use_ternary=use_ternary)
        self.W = _proj(m, hidden_size, bias=False, use_ternary=use_ternary)
        # LayerScale gate as additive modulation (1D Parameter, stays FP).
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.W.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        phi = self.A0(h) * F.silu(self.A1(h))
        phi = self.dropout(phi)
        return x + self.W(phi) * (1.0 + torch.tanh(self.gate))


class TransformerBlock(nn.Module):
    """Pre-norm ternary attention + HebbianBilinearFFN with residual.

    Drop-in for ``HAGIv4._stack_forward``: ``forward`` returns a single
    tensor (grad-checkpointing wraps it) and caches the attention-entropy-floor
    penalty on ``self._last_attn_entropy_penalty`` for the stack to sum.

    Args:
        hidden_size: H.
        attn_cfg: ``AttentionConfig``.
        ffn_cfg: ``HebbianFFNConfig``.
        norm_eps: RMSNorm epsilon (forwarded to the sub-blocks).
        use_ternary: forwarded to ``Attention`` and ``HebbianBilinearFFN``
            (swaps the 2D mixing weights to BitLinear). Default True.
    """

    def __init__(
        self,
        hidden_size: int,
        attn_cfg: AttentionConfig,
        ffn_cfg: HebbianFFNConfig,
        norm_eps: float = 1e-6,
        use_ternary: bool = True,
    ) -> None:
        super().__init__()
        self.attn = Attention(hidden_size, attn_cfg, norm_eps, use_ternary=use_ternary)
        self.ffn = HebbianBilinearFFN(hidden_size, ffn_cfg, norm_eps, use_ternary=use_ternary)
        self._last_attn_entropy_penalty: torch.Tensor | None = None

    def set_attn_entropy_floor(self, floor: float) -> None:
        self.attn.set_attn_entropy_floor(floor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
    ) -> torch.Tensor:
        attn_out, pen = self.attn(x, attention_mode, prefix_len, soft_beta)
        x = x + attn_out
        self._last_attn_entropy_penalty = pen
        x = x + self.ffn(x)
        return x
