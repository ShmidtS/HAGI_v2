"""HAGI V21 Source Encoder — memoryless source coding stage.

Owns: token embedding (ConvEmbedding or nn.Embedding), semantic erasure
indicator, bottleneck gate, rate matcher (H<->C), perception stack
(AttentionBlock x N_perc), pilot position encoding caches.

V21 refactor: extracted verbatim from the monolithic HAGIv4 class. No
behavioural changes; only the ownership boundary moved.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.attention_block import AttentionBlock
from hagi_v4.model.codec._utils import _block_call
from hagi_v4.model.codec_contracts import (
    CodecShapeConfig,
    SourceEncodeResult,
)
from hagi_v4.model.conv_embedding import ConvEmbedding
from hagi_v4.model.cqi import CQIEstimator

if TYPE_CHECKING:
    from hagi_v4.inference.spectral_cache import SpectralCache


class SourceEncoder(nn.Module):
    """Source encoder: discrete tokens -> continuous systematic latent z.

    Pipeline (verbatim from HAGIv4 V18):
      ConvEmbed (or nn.Embedding) with semantic erasure
        -> sinusoidal pilot PE
        -> perception stack (AttentionBlock x N_perc, checkpointed in train)
        -> rate_down (FactoredLinear H->C) with LayerScale + SiLU bottleneck
        -> z_sys in R^C
    """

    def __init__(self, cfg: HAGIv4Config) -> None:
        super().__init__()
        m = cfg.model
        self.codec_shape = CodecShapeConfig.from_hagi_config(cfg)
        H = m.hidden_size
        C = m.core_hidden_size
        self._H = H
        self._C = C

        # ---- Source encoder ----
        em = m.embeddings
        if em.use_conv_embedding:
            self.embed = ConvEmbedding(
                vocab_size=m.vocab_size,
                hidden_size=H,
                factor_rank=em.factor_rank,
                kernel_size=em.kernel_size,
                norm_eps=m.norm_eps,
                init=em.init,
            )
        else:
            self.embed = nn.Embedding(m.vocab_size, H)
            nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(H))

        # Learned erasure indicator (semantic). Replaces the compressed source
        # code on masked positions BEFORE pulse-shaping / cache writes /
        # frequency mixing — channel-correct placement.
        self.semantic_unknown_embed = nn.Parameter(torch.empty(H))
        nn.init.normal_(self.semantic_unknown_embed, mean=0.0, std=1.0 / math.sqrt(H))

        # V19: Learnable per-channel bottleneck gate through non-linearity.
        # V18 bug: bottleneck_scale=ones was sandwiched between two Linear
        # layers (rate_down -> *scale -> rate_up) with NO non-linearity, so
        # it was reparametrisation-equivalent to scaling rate_up columns ->
        # gradient identically zero (verified: std=0.0 across 1000 steps).
        # Fix: insert SiLU non-linearity + bounded tanh gate (zero-init like
        # CaiT LayerScale). tanh makes scale identifiable, SiLU breaks the
        # linear-linear invariance, zero-init keeps cold-start stable.
        self.bottleneck_gate = nn.Parameter(torch.zeros(C))

        self.core_mask_embed = nn.Parameter(torch.empty(C))
        nn.init.uniform_(self.core_mask_embed, -1.0 / math.sqrt(C), 1.0 / math.sqrt(C))

        # V14 Learned Rate Matcher (Linear H<->C). V18: NO norm on the
        # systematic latent; only the LayerScale above.
        lrm_rank = max(1, min(C, H // 2))
        self._build_rate_matcher(H, C, lrm_rank)

        # Sinusoidal pilot position encoding (zero params, distinguishes
        # masked positions by location).
        self._pilot_pos_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._pilot_pos_max = 4
        self._pilot_idx_cache: dict[int, torch.Tensor] = {}
        self._pilot_mask_cache: dict[int, torch.Tensor] = {}
        self._pilot_cache_max = 4

        # Source stacks: AttentionBlock (V15 win). head_dim forced to 64 so
        # H is divisible (config head_dim 72 does not divide cleanly).
        head_dim_src = 64
        n_heads_src = max(1, H // head_dim_src)
        if H % n_heads_src != 0:
            raise ValueError(f"hidden_size={H} must be divisible by n_heads={n_heads_src} (head_dim={head_dim_src})")
        rope_theta = float(getattr(m.attention, "rope_theta", 10000.0))
        self.perception = nn.ModuleList(
            AttentionBlock(
                H,
                n_heads=n_heads_src,
                ffn_mult=4.0,
                norm_eps=m.norm_eps,
                rope_theta=rope_theta,
            )
            for _ in range(m.perception_layers)
        )

        # V21: CQI estimator — per-position channel quality indicator.
        # Disabled by default; enable via config (model.cqi).
        self.cqi_estimator: CQIEstimator | None = None
        if getattr(m, "cqi", None) is not None:
            self.cqi_estimator = CQIEstimator(hidden_size=H)

        self.moe_layer = None
        moe_cfg = getattr(m, "moe", None)
        if moe_cfg is not None and getattr(moe_cfg, "enabled", False) and getattr(moe_cfg, "num_experts", 0) > 1:
            from hagi_v4.model.moe import MoESwiGLU

            self.moe_layer = MoESwiGLU(moe_cfg, hidden_size=H)

    def _build_rate_matcher(self, H: int, C: int, rank: int) -> None:
        """Construct the rate_down / rate_up pair.

        V18: ``rate_up.expand`` is initialised with std=1/sqrt(C) (NOT zero)
        so the channel path carries gradient from step 0. This resolves the
        cold-start deadlock identified in the architect review: with zero
        init, the only learnable parameter on the source-decode path at
        step 0 is ``rate_up.expand.weight`` itself, starving the source
        encoder of gradient signal for the first ~100-200 steps.
        """
        from hagi_v4.model.freq_layer import FactoredLinear

        self.rate_down = FactoredLinear(H, C, rank, bias=False)
        self.rate_up = FactoredLinear(C, H, rank, bias=False)
        # V18 critical init: expand must be nonzero for gradient flow.
        nn.init.normal_(self.rate_up.expand.weight, mean=0.0, std=1.0 / math.sqrt(C))

    def _stack_forward(
        self,
        h: torch.Tensor,
        blocks: nn.ModuleList,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run blocks sequentially with the requested attention mode.

        V22: returns (output, attn_entropy_penalty) for distribution
        stabilization. attention_mode propagates through perception/expression
        stacks. During training, the loop selects "bidir" (masked recovery),
        "soft_causal" (smooth blending), or "causal". During AR inference,
        "causal".
        """
        total_entropy_pen = None
        for blk in blocks:
            if self.training:
                h = checkpoint(
                    _block_call,
                    blk,
                    h,
                    attention_mode,
                    prefix_len,
                    soft_beta,
                    use_reentrant=False,
                )
            else:
                h = blk(h, attention_mode=attention_mode, prefix_len=prefix_len, soft_beta=soft_beta)
            pen = getattr(blk, "_last_attn_entropy_penalty", None)
            if pen is not None:
                total_entropy_pen = pen if total_entropy_pen is None else total_entropy_pen + pen
        return h, total_entropy_pen

    def _get_pilot_idx(self, T: int, device: torch.device) -> torch.Tensor:
        spacing = self.codec_shape.pilot_spacing
        if T not in self._pilot_idx_cache:
            if len(self._pilot_idx_cache) >= self._pilot_cache_max:
                oldest = next(iter(self._pilot_idx_cache))
                del self._pilot_idx_cache[oldest]
            self._pilot_idx_cache[T] = torch.arange(0, T, spacing, device=device)
        return self._pilot_idx_cache[T]

    def _get_pilot_mask(self, T: int, device: torch.device) -> torch.Tensor:
        spacing = self.codec_shape.pilot_spacing
        if T not in self._pilot_mask_cache:
            if len(self._pilot_mask_cache) >= self._pilot_cache_max:
                oldest = next(iter(self._pilot_mask_cache))
                del self._pilot_mask_cache[oldest]
            pm = torch.ones(T, dtype=torch.bool, device=device)
            pm[::spacing] = False
            self._pilot_mask_cache[T] = pm
        return self._pilot_mask_cache[T]

    def _get_pilot_position_encoding(self, T: int, H: int, device: torch.device) -> torch.Tensor:
        key = (T, H)
        if key not in self._pilot_pos_cache:
            if len(self._pilot_pos_cache) >= self._pilot_pos_max:
                oldest = next(iter(self._pilot_pos_cache))
                del self._pilot_pos_cache[oldest]
            position = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, H, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / max(H, 1))
            )
            pe = torch.zeros(T, H, device=device, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self._pilot_pos_cache[key] = pe
        return self._pilot_pos_cache[key]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        semantic_unknown_mask: torch.Tensor | None,
        cache: SpectralCache | None,
        pre_encoded_h: torch.Tensor | None = None,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
    ) -> SourceEncodeResult:
        if pre_encoded_h is not None:
            h = pre_encoded_h
            cached_len = 0
            h, enc_entropy_pen = self._stack_forward(
                h, self.perception, attention_mode=attention_mode, prefix_len=prefix_len, soft_beta=soft_beta
            )
            self._last_attn_entropy_penalty = enc_entropy_pen
        else:
            B, T = input_ids.shape
            embed_dev = self.embed.weight.device
            ids_dev = input_ids.device
            unknown = self.semantic_unknown_embed.to(device=embed_dev, dtype=self.embed.token_expand.weight.dtype)
            unknown_pos = unknown if semantic_unknown_mask is not None else None
            if embed_dev != ids_dev:
                ids_on_dev = input_ids.to(embed_dev)
                mask_on_dev = semantic_unknown_mask.to(embed_dev) if semantic_unknown_mask is not None else None
                h = self.embed.forward_with_erasure(ids_on_dev, unknown_pos, mask_on_dev).to(ids_dev)
            else:
                h = self.embed.forward_with_erasure(input_ids, unknown_pos, semantic_unknown_mask)

            cached_len = 0
            if cache is not None and cache.context_len > 0:
                cached_h = cache.get_context(0)
                if cached_h is not None and cached_h.shape[0] == h.shape[0] and cached_h.shape[2] == h.shape[2]:
                    h = torch.cat([cached_h.to(h.dtype), h], dim=1)
                    cached_len = cached_h.shape[1]
            if cache is not None:
                cache.update_context(0, h, new_tokens=T)

            pilot_pe = self._get_pilot_position_encoding(h.shape[1], self._H, h.device)
            pilot_scale = 1.0 / max(self._H, 1) ** 0.5
            h = h + pilot_scale * pilot_pe.to(h.dtype).unsqueeze(0)

            h, enc_entropy_pen = self._stack_forward(
                h, self.perception, attention_mode=attention_mode, prefix_len=prefix_len, soft_beta=soft_beta
            )
            self._last_attn_entropy_penalty = enc_entropy_pen
            if cached_len > 0:
                h = h[:, cached_len:]

        if self.moe_layer is not None:
            h, moe_aux, moe_rp = self.moe_layer(h)
            self._last_moe_aux = moe_aux
        else:
            self._last_moe_aux = None

        # V19: LayerScale bottleneck with non-linearity (fixes V18 frozen
        # gradient). gate=0 at init -> tanh(0)=0, SiLU(rate_down(z))=0 ->
        # identity-ish cold start via residual to mean(z).  As gate grows,
        # bottleneck learns per-channel suppression.
        # V21: CQI — per-position channel quality from hidden state.
        cqi = None
        if self.cqi_estimator is not None:
            cqi = self.cqi_estimator(h)
        z_linear = self.rate_down(h)
        z_gated = torch.nn.functional.silu(z_linear) * torch.tanh(self.bottleneck_gate)
        z = z_linear + z_gated  # residual: at gate=0, z = z_linear (identity)
        return SourceEncodeResult(systematic=z, mask=semantic_unknown_mask, cqi=cqi, pre_bottleneck=None)
