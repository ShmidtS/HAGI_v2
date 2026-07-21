"""HAGI V18 — Strict Shannon Source-Channel Separation codec LM.

Pipeline:
  Source Encode (ConvEmbed + sinusoidal PE + AttentionBlock × N_perc + LayerScale + rate_down)
    -> z_sys in R^C
  Channel Encode (SparseParity + BlockInterleaver) -> codeword
  Physical Channel (AWGN + erasure on systematic only)
  Channel Decode (pure LDPC BP: syndrome -> H^T back-projection -> gated correction)
  Source Decode (rate_up + AttentionBlock × N_expr + final_norm + factored tied lm_head)

V18 changes vs V17 (see docs/ARCHITECTURE.md):
  - No source_skip_scale bypass (strict SCS)
  - No FreqBlock in decoder (pure LDPC BP)
  - No bottleneck_norm (LayerScale only)
  - No Lorentz sphere, HARQ buffer, EXIT halt, mutation branch, multimodal
  - rate_up.expand init N(0, 1/sqrt(C)) for cold-start gradient flow
  - corr_gate_w/b init neutral (0.1, 0.0)
  - Channel always open (no channel_open schedule)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.attention_block import AttentionBlock
from hagi_v4.model.codec_contracts import (
    ChannelEncodeResult,
    CodecShapeConfig,
    DecodeResult,
    DecodeState,
    InferenceShapeConfig,
    SemanticMaskBatch,
    SourceEncodeResult,
    TurboDecodeConfig,
)
from hagi_v4.model.conv_embedding import ConvEmbedding
from hagi_v4.model.interleaver import BlockInterleaver
from hagi_v4.model.norms import RMSNorm
from hagi_v4.model.outputs import AuxLosses, ModelOutput, compute_parity_diversity_loss, compute_whiteness_loss
from hagi_v4.model.sparse_parity import SparseParityChecker, SparseParityEncoder

if TYPE_CHECKING:
    from hagi_v4.inference.spectral_cache import SpectralCache


class LDPCDecoder(nn.Module):
    """Pure LDPC belief-propagation decoder with Kalman validation gate (V19).

    Each iteration:
      syndrome     = parity_recv - H @ z_pred        [B, T, M]
      d2           = ||syndrome||^2 / sigma^2         (Mahalanobis)
      gate_valid   = sigmoid((d2 - chi2_crit) / tau)  (validation gate)
      correction   = H^T @ syndrome * residual_scale
      gate_mag     = sigmoid(w * |syndrome|_mean + b) (magnitude gate)
      z_pred       = z_pred + gate_mag * gate_valid * correction

    Positions whose syndrome falls inside the strobe (d2 < chi2_crit) are
    statistically indistinguishable from AWGN and do NOT receive a correction
    update — value of computation (Horvitz/Russell/Wefald): expected info
    gain below compute cost. This is the Kalman/JPDA measurement validation
    gate applied at every BP iteration.

    If <2% of positions are active for two consecutive iterations at
    inference time, BP halts early (global convergence).

    No Kalman filter state, no HARQ buffer, no FreqBlock reasoning, no
    mutation branch. The Tanner-graph H is shared with the encoder (fixed
    sparse mask + frozen parity_base + learnable per-check edge_log_scale).
    """

    def __init__(
        self,
        hidden_size: int,
        n_parity_checks: int,
        edges_per_check: int,
        norm_eps: float = 1e-6,
        shared_parity_weights: nn.Parameter | None = None,
        shared_sparse_mask: torch.Tensor | None = None,
        shared_edge_log_scale: nn.Parameter | None = None,
        shared_parity_base: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_parity_checks = n_parity_checks
        self.edges_per_check = edges_per_check

        self.parity_checker = SparseParityChecker(
            n_vars=hidden_size,
            n_checks=n_parity_checks,
            edges_per_check=edges_per_check,
            seed=42,
            norm_eps=norm_eps,
            shared_weights=shared_parity_weights,
            shared_mask=shared_sparse_mask,
            shared_edge_log_scale=shared_edge_log_scale,
            shared_parity_base=shared_parity_base,
        )

        # V19: corr_gate init raised for faster learning. V18 init (0.1, 0.0)
        # barely moved in 1000 steps (final sigmoid~0.55, no selectivity).
        # Raise w to 0.3 so syndrome magnitude translates more sharply to
        # gate output, keep b=0.0 (neutral). Sigmoid(0.3*0.5 + 0) = 0.55 still
        # at init but the gradient signal is 3x stronger.
        self.corr_gate_w = nn.Parameter(torch.tensor([0.3]))
        self.corr_gate_b = nn.Parameter(torch.tensor([0.0]))

        # V19: AWGN sigma is fed in by the training loop / inference caller.
        # None means "unknown" — forward() falls back to conservative 0.1.
        self._current_awgn_sigma: float | None = None

    def set_awgn_sigma(self, sigma: float | None) -> None:
        """Inform the decoder of the current AWGN sigma so the validation
        gate can compute the correct Mahalanobis distance. Called by the
        training loop and the inference path."""
        self._current_awgn_sigma = float(sigma) if sigma is not None else None

    def forward(
        self,
        z_sys: torch.Tensor,
        parity_received: torch.Tensor,
        training: bool,
        state: DecodeState,
        mask: torch.Tensor | None = None,
        refinement_iterations: int | None = None,
        n_iters: int = 3,
    ) -> DecodeResult:
        """Run at most ``n_iters`` BP iterations with Kalman validation gating.

        Per-position Mahalanobis-style syndrome gate (validation gate /
        measurement gating from Kalman filtering / JPDA). Before each
        expensive update — H^T back-projection, gradient flow — the syndrome
        is tested against the noise covariance. Positions whose squared
        Mahalanobis distance d2 = s^T R^{-1} s falls below the chi2 threshold
        are considered "inside the strobe" — the residual is statistically
        indistinguishable from pure noise — and do NOT participate in the
        update. This is value-of-computation (Horvitz/Russell/Wefald): the
        expected information gain of running BP on those positions is below
        the compute cost, so BP is skipped for them.

        Two operating modes:
          * training: soft gate via sigmoid (differentiable, gradient flows)
          * inference: hard gate (positions with d2<threshold are frozen for
            the iteration, skipping H^T back-projection entirely)

        Additionally, if across the WHOLE batch no position exceeds the
        threshold for two consecutive iterations, BP halts early (global
        convergence criterion).
        """
        iteration_limit = refinement_iterations if refinement_iterations is not None else n_iters
        if type(iteration_limit) is not int or iteration_limit < 1:
            raise ValueError(f"refinement_iterations must be a positive int, got {iteration_limit!r}")

        z_pred = z_sys
        total_parity = z_sys.new_zeros(())
        last_residual = torch.zeros_like(z_sys[..., :1].expand_as(z_sys))

        h_matrix = self.parity_checker.masked_weights  # [M, C]
        M = h_matrix.shape[0]  # number of parity checks = degrees of freedom for chi2

        # V19 Kalman validation gate. AWGN covariance R = sigma^2 * I, so
        # Mahalanobis distance reduces to scaled L2: d2 = ||s||^2 / sigma^2.
        # chi2 critical value at p=0.95, df=M (large-M approximation:
        # chi2_{0.95, M} ~= M + sqrt(2M)*1.645). This is the strobe radius.
        chi2_crit = float(M) + (2.0 * float(M)) ** 0.5 * 1.645
        # Use the noise sigma from the decoder's training-time schedule. At
        # inference we don't know it, so fall back to a conservative 0.1
        # (the midpoint of the V19 schedule 0.05-0.15).
        sigma_sq = (self._current_awgn_sigma**2) if self._current_awgn_sigma is not None else 0.01

        # Per-position activation mask, accumulates "did we update this
        # position in at least one iteration". Drives iterations_used output.
        B, T = z_sys.shape[0], z_sys.shape[1]
        ever_active = torch.zeros((B, T), dtype=torch.bool, device=z_sys.device)
        iters_used_per_pos = torch.zeros((B, T), dtype=torch.long, device=z_sys.device)

        prev_global_active_frac = 1.0  # track convergence for early halt

        for iteration in range(iteration_limit):
            residual, _parity_computed = self.parity_checker(z_pred, parity_received)
            total_parity = total_parity + residual.pow(2).mean().to(total_parity.dtype)
            last_residual = residual

            # --- V19 Kalman validation gate ---------------------------------
            # d2[b, t] = sum_m residual[b,t,m]^2 / sigma^2
            d2 = residual.pow(2).sum(dim=-1) / max(sigma_sq, 1e-8)  # [B, T]
            global_active_frac = (d2 > chi2_crit).float().mean().item()

            if training:
                # Soft gate: differentiable sigmoid centered at the threshold.
                # Temperature tau controls sharpness; tau=chi2_crit/6 puts the
                # 0.5 crossing at d2=chi2_crit with reasonable slope.
                tau = max(chi2_crit / 6.0, 1.0)
                gate_valid = torch.sigmoid((d2 - chi2_crit) / tau)  # [B, T]
            else:
                # Hard gate: strobe — positions inside the gate are frozen.
                gate_valid = (d2 > chi2_crit).to(z_pred.dtype)  # [B, T]

            # Track which positions ever got an update (for VOC accounting).
            active_this_iter = d2 > chi2_crit
            ever_active = ever_active | active_this_iter
            iters_used_per_pos += active_this_iter.long()

            # --- Apply gated correction -------------------------------------
            # Back-project syndrome from parity space M to systematic space C
            # via H^T (transpose of the parity-check matrix).
            correction = torch.einsum("mc,btm->btc", h_matrix, residual)
            # Normalise by the systematic magnitude for stable gradients.
            z_scale = z_pred.float().pow(2).mean(dim=-1, keepdim=True).to(z_pred.dtype) + 1e-6
            correction = correction * (1.0 / torch.sqrt(z_scale)).clamp_max(4.0)

            # corr_gate is the magnitude gate (how strongly syndrome magnitude
            # translates to correction strength). Multiplied by the validation
            # gate (whether to correct at all).
            gate_mag = torch.sigmoid(
                residual.abs().float().mean(dim=-1, keepdim=True) * self.corr_gate_w + self.corr_gate_b
            ).to(z_pred.dtype)

            update = gate_mag * gate_valid.unsqueeze(-1) * correction
            if mask is not None:
                update = update * mask.unsqueeze(-1).to(update.dtype)
            z_pred = z_pred + update

            # --- Early halt: global convergence ------------------------------
            # If <2% of positions are active for two consecutive iterations,
            # BP has converged — no point continuing. This is rational
            # metareasoning: stop when VOC(continue BP) < 0.
            if not training and iteration >= 1 and global_active_frac < 0.02 and prev_global_active_frac < 0.02:
                iteration_limit = iteration + 1
                break
            prev_global_active_frac = global_active_frac

        side_info = {
            "parity_strength": total_parity / max(iteration_limit, 1),
            "iterations_used": iters_used_per_pos.clamp(min=1 if not training else 0, max=iteration_limit),
            "parity_residual": last_residual,
            # V19 gating diagnostics
            "ever_active_frac": ever_active.float().mean(),
            "mean_iters_used": iters_used_per_pos.float().mean() / max(iteration_limit, 1),
        }
        state.iteration = iteration_limit
        return DecodeResult(latent=z_pred, state=state, side_info=side_info)


def _block_call(
    block: nn.Module,
    x: torch.Tensor,
    attention_mode: str,
    prefix_len: torch.Tensor | int | None,
) -> torch.Tensor:
    """Helper for checkpoint-friendly invocation of AttentionBlock.

    torch.utils.checkpoint needs a callable with positional tensor args
    for reliable grad accumulation. This wraps the block call so the
    attention_mode/prefix_len kwargs are passed through.
    """
    return block(x, attention_mode=attention_mode, prefix_len=prefix_len)


class HAGIv4(nn.Module):
    """HAGI V18 — Strict Source-Channel Separation codec language model."""

    def __init__(self, cfg: HAGIv4Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.codec_shape = CodecShapeConfig.from_hagi_config(cfg)
        self.turbo_config = TurboDecodeConfig.from_hagi_config(cfg)
        self.inference_config = InferenceShapeConfig.from_hagi_config(cfg)
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

        # ---- Channel encoder ----
        self.parity_encoder = SparseParityEncoder(
            n_vars=C,
            n_checks=self.codec_shape.n_parity_checks,
            edges_per_check=self.codec_shape.edges_per_check,
            seed=42,
            norm_eps=m.norm_eps,
        )

        self.interleaver = BlockInterleaver(
            block_len=m.attention.max_seq_len,
            mode=self.codec_shape.interleaver_mode,
        )

        # ---- Channel decoder ----
        self.decoder = LDPCDecoder(
            hidden_size=C,
            n_parity_checks=self.codec_shape.n_parity_checks,
            edges_per_check=self.codec_shape.edges_per_check,
            norm_eps=m.norm_eps,
            shared_parity_weights=self.parity_encoder.edge_log_scale,
            shared_sparse_mask=self.parity_encoder.sparse_mask,
            shared_edge_log_scale=self.parity_encoder.edge_log_scale,
            shared_parity_base=self.parity_encoder.parity_base,
        )

        # ---- Source decoder (mirror of perception, NOT alias) ----
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
        if self.tie_source_codebook:
            with torch.no_grad():
                self.lm_expand.weight = self.embed.token_compress.weight

        # V13: distill_align allocated lazily when distillation is enabled.
        if cfg.train.distill_enabled:
            teacher_hidden = cfg.train.distill_teacher_hidden_size
            self.distill_align = nn.Identity() if teacher_hidden == H else nn.Linear(teacher_hidden, H, bias=False)
        else:
            self.distill_align = None

        self._init_weights()

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

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """Effective [V, H] output projection for compatibility callers."""
        return self.lm_expand.weight @ self.lm_compress.weight

    def _init_weights(self) -> None:
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "mut_" in name or mod is self.lm_compress:
                    continue
                if self.tie_source_codebook and mod is self.lm_expand:
                    continue
                if name.endswith("out_proj") or name.endswith("ffn.w_out") or mod is self.rate_up.expand:
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                    continue
                std = 1.0 / math.sqrt(max(1, mod.weight.shape[1]))
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                if mod is self.embed or mod is getattr(self.embed, "token_compress", None):
                    continue
                std = 1.0 / math.sqrt(max(1, mod.weight.shape[1]))
                nn.init.normal_(mod.weight, mean=0.0, std=std)

    def train(self, mode: bool = True) -> HAGIv4:
        return super().train(mode)

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

    def _stack_forward(
        self,
        h: torch.Tensor,
        blocks: nn.ModuleList,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Run blocks sequentially with the requested attention mode.

        V20: attention_mode propagates through perception/expression stacks.
        During training, the loop selects "bidir" (masked recovery) or
        "prefix" (causal suffix prediction). During AR inference, "causal".
        """
        for blk in blocks:
            if self.training:
                # checkpoint doesn't accept extra positional args reliably
                # with use_reentrant=False in older torch versions, so we
                # fall back to a lambda that closes over the kwargs.
                h = checkpoint(
                    _block_call,
                    blk,
                    h,
                    attention_mode,
                    prefix_len,
                    use_reentrant=False,
                )
            else:
                h = blk(h, attention_mode=attention_mode, prefix_len=prefix_len)
        return h

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

    def _source_encode(
        self,
        input_ids: torch.Tensor | None,
        semantic_unknown_mask: torch.Tensor | None,
        cache: SpectralCache | None,
        pre_encoded_h: torch.Tensor | None = None,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
    ) -> SourceEncodeResult:
        if pre_encoded_h is not None:
            h = pre_encoded_h
            cached_len = 0
            h = self._stack_forward(h, self.perception, attention_mode=attention_mode, prefix_len=prefix_len)
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

            h = self._stack_forward(h, self.perception, attention_mode=attention_mode, prefix_len=prefix_len)
            if cached_len > 0:
                h = h[:, cached_len:]

        # V19: LayerScale bottleneck with non-linearity (fixes V18 frozen
        # gradient). gate=0 at init -> tanh(0)=0, SiLU(rate_down(z))=0 ->
        # identity-ish cold start via residual to mean(z).  As gate grows,
        # bottleneck learns per-channel suppression.
        z_linear = self.rate_down(h)
        z_gated = torch.nn.functional.silu(z_linear) * torch.tanh(self.bottleneck_gate)
        z = z_linear + z_gated  # residual: at gate=0, z = z_linear (identity)
        return SourceEncodeResult(systematic=z, mask=semantic_unknown_mask, cqi=None, pre_bottleneck=None)

    def _channel_encode(self, encoded: SourceEncodeResult) -> ChannelEncodeResult:
        systematic = encoded.systematic
        parity = self.parity_encoder(systematic)
        codeword = torch.cat([systematic, parity], dim=-1)
        codeword = self.interleaver.interleave(codeword)
        return ChannelEncodeResult(codeword=codeword, systematic=systematic, parity=parity)

    def _apply_erasure(
        self,
        encoded: ChannelEncodeResult,
        erasure_mask: torch.Tensor | None,
        awgn_sigma: float = 0.0,
    ) -> ChannelEncodeResult:
        if erasure_mask is None and (not self.training or awgn_sigma <= 0.0):
            return encoded
        codeword = self.interleaver.deinterleave(encoded.codeword).clone()
        systematic = codeword[..., : self._C]
        if self.training and awgn_sigma > 0.0:
            systematic.add_(awgn_sigma * torch.randn_like(systematic))
        if erasure_mask is not None:
            systematic[erasure_mask] = self.core_mask_embed.to(systematic.dtype)
        return ChannelEncodeResult(
            codeword=self.interleaver.interleave(codeword),
            systematic=encoded.systematic,
            parity=encoded.parity,
            interleaver_perm=encoded.interleaver_perm,
            erasure_mask=erasure_mask,
        )

    def _channel_decode(
        self,
        ch_encoded: ChannelEncodeResult,
        encoded: SourceEncodeResult,
        state: DecodeState,
        training: bool,
        refinement_iterations: int | None = None,
    ) -> DecodeResult:
        codeword = self.interleaver.deinterleave(ch_encoded.codeword)
        C = self._C
        z_sys = codeword[..., :C]
        parity = codeword[..., C:]
        decode_mask = ch_encoded.erasure_mask
        if encoded.mask is not None:
            decode_mask = encoded.mask if decode_mask is None else (decode_mask | encoded.mask)
        train_bp = getattr(self.cfg.train, "bp_iterations", None)
        n_iters = int(train_bp) if train_bp is not None else 3
        result = self.decoder(
            z_sys=z_sys,
            parity_received=parity,
            training=training,
            state=state,
            mask=decode_mask,
            refinement_iterations=refinement_iterations,
            n_iters=n_iters,
        )
        if ch_encoded.erasure_mask is not None and ch_encoded.erasure_mask.any():
            predicted = result.latent - z_sys
            target = ch_encoded.systematic - z_sys
            selected_predicted = predicted[ch_encoded.erasure_mask].float()
            selected_target = target[ch_encoded.erasure_mask].float()
            target_scale = selected_target.pow(2).mean().clamp_min(1e-6)
            normalized_mse = (selected_predicted - selected_target).pow(2).mean() / target_scale
            pred_norm = selected_predicted.norm(dim=-1)
            has_signal = (pred_norm > 1e-6).any()
            if has_signal:
                pred_safe = selected_predicted / pred_norm.clamp_min(1e-6).unsqueeze(-1)
                target_safe = selected_target / selected_target.norm(dim=-1).clamp_min(1e-6).unsqueeze(-1)
                cosine = (pred_safe * target_safe).sum(dim=-1).mean()
            else:
                cosine = torch.zeros((), device=z_sys.device, dtype=z_sys.dtype)
            result.side_info["correction_alignment"] = normalized_mse + 1.0 - cosine
        else:
            result.side_info["correction_alignment"] = z_sys.new_zeros(())
        return result

    def _source_decode(self, decoded: DecodeResult, encoded: SourceEncodeResult) -> torch.Tensor:
        # V18: strict SCS — NO pre_bottleneck bypass. The source decoder
        # receives only the post-channel latent.
        # V20: source decoder always runs in causal mode — it is producing
        # the rightward (post-channel) reconstruction that feeds the LM head,
        # and we want each position's reconstruction to depend only on
        # positions up to and including itself (no leakage of future latent
        # into the LM head's input). This preserves train/infer consistency
        # when the LM head is queried causally at inference time.
        z = decoded.latent
        h = self.rate_up(z)
        return self._stack_forward(h, self.expression, attention_mode="causal")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        *,
        semantic_unknown_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
        valid_target_mask: torch.Tensor,
        physical_corruption_mask: torch.Tensor,
        step: int = 0,
        cached_p: torch.Tensor | None = None,
        cache: SpectralCache | None = None,
        awgn_sigma: float = 0.0,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
        refinement_iterations: int | None = None,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
    ) -> ModelOutput:
        """V20: attention_mode selects how perception layers attend.

        - "bidir":  full bidirectional (default; masked-LM training)
        - "prefix": GLM-style prefix-LM (prefix_len bidir, rest causal)
        - "causal": pure GPT-style (AR inference)

        The source decoder (expression stack) always runs in causal mode
        to match the LM head's left-to-right query pattern at inference.
        """
        if input_ids is None:
            raise ValueError("input_ids is required")
        if refinement_iterations is None and self.training:
            train_bp = getattr(self.cfg.train, "bp_iterations", None)
            if train_bp is None:
                train_bp = 3
        masks = SemanticMaskBatch(
            semantic_unknown_mask,
            prediction_mask,
            valid_target_mask,
            physical_corruption_mask,
        )
        masks.validate(input_ids, None)
        state = cache.to_decode_state() if cache is not None else DecodeState(kalman_p=cached_p)
        if cache is not None and state.kalman_p is None:
            state.kalman_p = cached_p

        encoded = self._source_encode(
            input_ids,
            semantic_unknown_mask,
            cache,
            attention_mode=attention_mode,
            prefix_len=prefix_len,
        )
        chEncoded = self._apply_erasure(self._channel_encode(encoded), physical_corruption_mask, awgn_sigma)
        # V19: inform the LDPC decoder of the current AWGN sigma so its
        # Kalman validation gate uses the correct Mahalanobis threshold.
        self.decoder.set_awgn_sigma(awgn_sigma if awgn_sigma > 0.0 else None)
        decoded = self._channel_decode(chEncoded, encoded, state, self.training, refinement_iterations)
        h = self._source_decode(decoded, encoded)

        if cache is not None:
            cache.update_decode_state(decoded.state)

        side_info = decoded.side_info
        rd_loss = (encoded.systematic.float() - decoded.latent.float()).pow(2).mean().to(h.dtype)

        h_normed = self.final_norm(h)
        selected = prediction_mask & valid_target_mask
        prediction_indices = selected.flatten().nonzero(as_tuple=False).squeeze(-1)
        prediction_hidden = h_normed[:, : input_ids.shape[1]]
        selected_hidden = prediction_hidden.flatten(0, 1).index_select(0, prediction_indices.to(h_normed.device))
        logits = self.lm_expand(self.lm_compress(selected_hidden.to(self.lm_compress.weight.device))).to(
            h_normed.device
        )
        if targets is not None:
            if prediction_indices.numel() == 0:
                raise ValueError("prediction_mask must select at least one target during training")
            selected_targets = targets.flatten().index_select(0, prediction_indices.to(targets.device))
            ce = F.cross_entropy(logits, selected_targets.to(logits.device))
        else:
            ce = None

        aux = AuxLosses()
        if targets is not None:
            if side_info.get("parity_strength") is not None:
                aux.parity = side_info["parity_strength"]
            if self.codec_shape.use_whiteness_loss and side_info.get("parity_residual") is not None:
                pilot_mask = self._get_pilot_mask(
                    side_info["parity_residual"].shape[1],
                    side_info["parity_residual"].device,
                )
                valid = pilot_mask.unsqueeze(0).expand(side_info["parity_residual"].shape[0], -1)
                valid = valid & ~physical_corruption_mask.to(valid.device)
                aux.whiteness = compute_whiteness_loss(side_info["parity_residual"], valid)
            aux.correction_alignment = side_info["correction_alignment"].to(h.dtype)
            aux.rate_distortion = rd_loss

            if getattr(self.cfg.train, "w_parity_diversity", 0.0) > 0.0:
                aux.parity_diversity = compute_parity_diversity_loss(self.parity_encoder.masked_weights)

            if self.training and physical_corruption_mask is not None and physical_corruption_mask.any():
                z_clean = encoded.systematic
                z_erased = chEncoded.codeword[..., : self._C]
                recovery_error = (z_clean.float() - z_erased.float()).pow(2)
                erased_error = recovery_error[physical_corruption_mask.to(recovery_error.device)]
                if erased_error.numel() > 0:
                    aux.parity_recovery = erased_error.mean().to(h.dtype)

        return ModelOutput(
            logits=logits,
            hidden=h,
            aux=aux,
            ce_loss=ce,
            iterations_used=side_info.get("iterations_used"),
            prediction_indices=prediction_indices,
        )
