"""HAGI V21 Source Decoder — rate-up + expression stack + LM head.

Owns: expression stack (AttentionBlock x N_expr), final_norm, lm_compress,
lm_expand (factored, weight-tied with token_compress). The rate_up layer
is owned by SourceEncoder but applied here (passed by reference).

V21 refactor: extracted verbatim from the monolithic HAGIv4 class. No
behavioural changes; only the ownership boundary moved.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.attention_block import AttentionBlock
from hagi_v4.model.codec._utils import _block_call
from hagi_v4.model.codec_contracts import DecodeResult
from hagi_v4.model.lorentz import LorentzSphereNorm, lorentz_log_origin
from hagi_v4.model.norms import RMSNorm


class SourceDecoder(nn.Module):
    """Source decoder: recovered latent z -> hidden h -> LM logits.

    Pipeline (verbatim from HAGIv4 V18):
      decoded z (post-channel)
        -> rate_up (FactoredLinear C->H, shared with SourceEncoder)
        -> expression stack (AttentionBlock x N_expr, causal)
        -> final_norm
        -> lm_compress (H->r) -> lm_expand (r->V) [factored, weight-tied]

    V20: source decoder always runs in causal mode — it is producing
    the rightward (post-channel) reconstruction that feeds the LM head,
    and we want each position's reconstruction to depend only on
    positions up to and including itself (no leakage of future latent
    into the LM head's input). This preserves train/infer consistency
    when the LM head is queried causally at inference time.
    """

    def __init__(
        self,
        cfg: HAGIv4Config,
        rate_up: nn.Module | None = None,
        token_compress_weight: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        m = cfg.model
        H = m.hidden_size
        em = m.embeddings
        self._H = H

        # rate_up is owned by SourceEncoder but applied here. Passed by
        # reference to preserve weight sharing.
        if rate_up is not None:
            self.rate_up = rate_up

        # ---- Source decoder (mirror of perception, NOT alias) ----
        head_dim_src = 64
        n_heads_src = max(1, H // head_dim_src)
        if H % n_heads_src != 0:
            raise ValueError(f"hidden_size={H} must be divisible by n_heads={n_heads_src} (head_dim={head_dim_src})")
        rope_theta = float(getattr(m.attention, "rope_theta", 10000.0))
        n_expr = max(1, int(getattr(m, "expression_layers", 4) or 4))
        self.expression = nn.ModuleList(
            AttentionBlock(
                H,
                n_heads=n_heads_src,
                ffn_mult=4.0,
                norm_eps=m.norm_eps,
                rope_theta=rope_theta,
            )
            for _ in range(n_expr)
        )

        self.final_norm = RMSNorm(H, eps=m.norm_eps)

        # Source decoder head: factored, weight-tied with token_compress (V12 win).
        self.lm_compress = nn.Linear(H, em.factor_rank, bias=False)
        self.lm_expand = nn.Linear(em.factor_rank, m.vocab_size, bias=False)
        nn.init.normal_(self.lm_compress.weight, mean=0.0, std=1.0 / math.sqrt(H))
        nn.init.normal_(self.lm_expand.weight, mean=0.0, std=1.0 / math.sqrt(em.factor_rank))
        self.tie_source_codebook: bool = em.use_conv_embedding
        if self.tie_source_codebook and token_compress_weight is not None:
            with torch.no_grad():
                self.lm_expand.weight = token_compress_weight

        self.lorentz_norm: LorentzSphereNorm | None = None
        if getattr(m, "lorentz_enabled", False):
            self.lorentz_norm = LorentzSphereNorm(dim=H)

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
        stabilization. attention_mode propagates through expression stacks.
        During training, the loop selects "bidir" (masked recovery),
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

    def _chunked_ce(self, h: torch.Tensor, targets: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        B, T, H = h.shape
        h_flat = h.reshape(B * T, H)
        t_flat = targets.reshape(B * T)
        compress_dev = self.lm_compress.weight.device
        z = self.lm_compress(h_flat.to(compress_dev))
        total_loss = z.new_zeros(())
        n = z.shape[0]
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            logits_chunk = self.lm_expand(z[i:end])
            total_loss = total_loss + F.cross_entropy(logits_chunk, t_flat[i:end].to(compress_dev), reduction="sum")
        return total_loss / max(n, 1)

    def forward(
        self, decoded: DecodeResult, attention_mode: str = "causal", soft_beta: float | None = None
    ) -> torch.Tensor:
        # V18: strict SCS — NO pre_bottleneck bypass. The source decoder
        # receives only the post-channel latent.
        # V20: source decoder always runs in causal mode — it is producing
        # the rightward (post-channel) reconstruction that feeds the LM head,
        # and we want each position's reconstruction to depend only on
        # positions up to and including itself (no leakage of future latent
        # into the LM head's input). This preserves train/infer consistency
        # when the LM head is queried causally at inference time.
        # V22: caller can override attention_mode for soft_causal blending.
        z = decoded.latent
        h = self.rate_up(z)
        if self.lorentz_norm is not None:
            h = self.lorentz_norm(h)
            h = lorentz_log_origin(h)
        h, dec_entropy_pen = self._stack_forward(h, self.expression, attention_mode=attention_mode, soft_beta=soft_beta)
        self._last_attn_entropy_penalty = dec_entropy_pen
        return h
