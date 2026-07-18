"""HAGI V4 — Source-Channel Separation codec language model.

Pipeline:
  Source Encode (Embed + FreqBlock + IB Bottleneck) -> systematic
  Channel Encode (SparseParity + Interleave) -> codeword
  Channel Decode (iterative BP: VN update + CN check + Kalman + HARQ) -> decoded
  Source Decode (RateDematch + FreqBlock + LM Head) -> logits
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.clifford_cross_modal import CliffordCrossModal
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
from hagi_v4.model.contrastive import ContrastiveAlignment
from hagi_v4.model.cqi import CQIEstimator
from hagi_v4.model.exit_chart import EXITChartEstimator
from hagi_v4.model.freq_layer import FactoredSwiGLU, FreqBlock
from hagi_v4.model.interleaver import BlockInterleaver
from hagi_v4.model.lorentz import LorentzSphereNorm
from hagi_v4.model.msa import HARQBuffer
from hagi_v4.model.multimodal_input import MultimodalInput
from hagi_v4.model.norms import RMSNorm
from hagi_v4.model.outputs import AuxLosses, ModelOutput, compute_whiteness_loss
from hagi_v4.model.sparse_parity import SparseParityChecker, SparseParityEncoder
from hagi_v4.model.uncertainty import LearnedUncertainty, inverse_variance_update

if TYPE_CHECKING:
    from hagi_v4.inference.spectral_cache import SpectralCache


class LDPCDecoder(nn.Module):
    """Iterative LDPC-style belief propagation decoder (extrinsic-only)."""

    def __init__(
        self,
        cfg: TurboDecodeConfig,
        hidden_size: int,
        n_parity_checks: int,
        edges_per_check: int,
        exit_threshold: float,
        shared_parity_weights: nn.Parameter | None = None,
        shared_sparse_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_iters = cfg.num_iterations
        self.min_iters = cfg.min_iterations
        self.use_convergence_halt = cfg.use_convergence_halt

        n_modes_t = cfg.freq_n_modes_t
        n_modes_h = cfg.freq_n_modes_h
        ffn_int = hidden_size * 2
        rank = cfg.freq_complex_rank
        head_dim = cfg.attention_head_dim
        n_heads = max(1, hidden_size // head_dim)
        proj_rank = max(1, hidden_size // 4)
        ffn_rank = max(1, hidden_size // 4)

        init_std = 1.0 / math.sqrt(max(1, n_heads * head_dim))
        shared_w = (
            nn.Parameter(torch.zeros(n_heads, head_dim, rank)),
            nn.Parameter(torch.zeros(n_heads, head_dim, rank)),
            nn.Parameter(torch.zeros(n_heads, rank, head_dim)),
            nn.Parameter(torch.zeros(n_heads, rank, head_dim)),
        )
        for p in shared_w:
            nn.init.normal_(p, std=init_std)
        shared_phase = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
        nn.init.normal_(shared_phase, std=init_std)
        self.shared_w = nn.ParameterList(shared_w)
        self.shared_phase = shared_phase

        # Derivative branch shared weights (Phase 2 Task 6): reuse dT/dH
        # weights across all reasoning layers (same memory-saving pattern as
        # the main branch). Packed into 12-tuple convention: 4 main + 4 dT + 4 dH.
        shared_w_dT = (
            nn.Parameter(torch.zeros(n_heads, head_dim, rank)),
            nn.Parameter(torch.zeros(n_heads, head_dim, rank)),
            nn.Parameter(torch.zeros(n_heads, rank, head_dim)),
            nn.Parameter(torch.zeros(n_heads, rank, head_dim)),
        )
        shared_w_dH = (
            nn.Parameter(torch.zeros(n_heads, head_dim, rank)),
            nn.Parameter(torch.zeros(n_heads, head_dim, rank)),
            nn.Parameter(torch.zeros(n_heads, rank, head_dim)),
            nn.Parameter(torch.zeros(n_heads, rank, head_dim)),
        )
        for p in (*shared_w_dT, *shared_w_dH):
            nn.init.normal_(p, std=init_std)
        self.shared_w_dT = nn.ParameterList(shared_w_dT)
        self.shared_w_dH = nn.ParameterList(shared_w_dH)

        shared_phase_dT = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
        shared_phase_dH = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
        nn.init.normal_(shared_phase_dT, std=init_std)
        nn.init.normal_(shared_phase_dH, std=init_std)
        self.shared_phase_dT = shared_phase_dT
        self.shared_phase_dH = shared_phase_dH

        shared_w_full = tuple(shared_w) + tuple(shared_w_dT) + tuple(shared_w_dH)

        shared_ffn = FactoredSwiGLU(hidden_size, ffn_int, ffn_rank)
        self.shared_ffn = shared_ffn

        self.reasoning = nn.ModuleList(
            FreqBlock(
                hidden_size,
                n_heads=n_heads,
                head_dim=head_dim,
                n_modes_t=n_modes_t,
                n_modes_h=n_modes_h,
                ffn_intermediate=ffn_int,
                T_max=cfg.attention_max_seq_len,
                rank=rank,
                proj_rank=proj_rank,
                ffn_rank=ffn_rank,
                shared_weights=shared_w_full,
                shared_phase=shared_phase,
                shared_phase_dT=shared_phase_dT,
                shared_phase_dH=shared_phase_dH,
                shared_ffn=shared_ffn,
                norm_eps=cfg.norm_eps,
                use_derivative=True,
                share_branch_weights=False,
            )
            for _ in range(cfg.reasoning_layers)
        )

        self.parity_checker = SparseParityChecker(
            n_vars=hidden_size,
            n_checks=n_parity_checks,
            edges_per_check=edges_per_check,
            seed=42,
            norm_eps=cfg.norm_eps,
            shared_weights=shared_parity_weights,
            shared_mask=shared_sparse_mask,
        )

        self.harq = HARQBuffer(cfg.msa, hidden_size)
        self.uncertainty = LearnedUncertainty(hidden_size)
        self.exit_estimator = EXITChartEstimator(
            threshold=exit_threshold,
            min_iterations=self.min_iters,
        )

        self._layers_per_iter = max(2, cfg.reasoning_layers // 2)

        mut_rank = max(1, hidden_size // 8)
        self.mut_down_w = nn.Parameter(torch.empty(mut_rank, hidden_size))
        self.mut_down_b = nn.Parameter(torch.empty(mut_rank))
        self.mut_up_w = nn.Parameter(torch.empty(hidden_size, mut_rank))

        self.corr_gate_w = nn.Parameter(torch.zeros(1))
        self.corr_gate_b = nn.Parameter(torch.tensor([-3.0]))

        self.ext_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z_sys: torch.Tensor,
        parity_received: torch.Tensor | None,
        training: bool,
        state: DecodeState,
        cqi_mean: torch.Tensor | float = 0.5,
        mask: torch.Tensor | None = None,
        refinement_iterations: int | None = None,
    ) -> DecodeResult:
        B, T, C = z_sys.shape
        iteration_limit = self.n_iters if refinement_iterations is None else refinement_iterations
        if type(iteration_limit) is not int or not 1 <= iteration_limit <= self.n_iters:
            raise ValueError(f"refinement_iterations must be an integer in [1, {self.n_iters}]")

        total_parity = z_sys.new_zeros(())
        extrinsic_norms: list = []
        converged_at = iteration_limit
        iterations_used = torch.full((B, T), iteration_limit, dtype=torch.long, device=z_sys.device)
        parity_residual = torch.zeros_like(z_sys)

        self.harq.clear()
        if state.cache_active and state.harq_feedback is not None:
            self.harq.restore_feedback(state.harq_feedback)

        w_shared = None
        phase_shared = None
        w_shared_dT = None
        w_shared_dH = None
        phase_shared_dT = None
        phase_shared_dH = None
        if self.shared_w is not None:
            w_re = self.shared_w[0].float() @ self.shared_w[2].float()
            w_im = self.shared_w[1].float() @ self.shared_w[3].float()
            w_shared = torch.complex(w_re, w_im)
        if hasattr(self, "shared_w_dT"):
            w_re_dT = self.shared_w_dT[0].float() @ self.shared_w_dT[2].float()
            w_im_dT = self.shared_w_dT[1].float() @ self.shared_w_dT[3].float()
            w_shared_dT = torch.complex(w_re_dT, w_im_dT)
        if hasattr(self, "shared_w_dH"):
            w_re_dH = self.shared_w_dH[0].float() @ self.shared_w_dH[2].float()
            w_im_dH = self.shared_w_dH[1].float() @ self.shared_w_dH[3].float()
            w_shared_dH = torch.complex(w_re_dH, w_im_dH)
        if self.shared_phase is not None:
            Kt = min(self.shared_phase.shape[1], T)
            Kh = min(self.shared_phase.shape[2], self.reasoning[0].freq.head_dim)
            phase_shared = torch.exp(1j * self.shared_phase[:, :Kt, :Kh].float())
        if hasattr(self, "shared_phase_dT"):
            Kt = min(self.shared_phase_dT.shape[1], T)
            Kh = min(self.shared_phase_dT.shape[2], self.reasoning[0].freq.head_dim)
            phase_shared_dT = torch.exp(1j * self.shared_phase_dT[:, :Kt, :Kh].float())
        if hasattr(self, "shared_phase_dH"):
            Kt = min(self.shared_phase_dH.shape[1], T)
            Kh = min(self.shared_phase_dH.shape[2], self.reasoning[0].freq.head_dim)
            phase_shared_dH = torch.exp(1j * self.shared_phase_dH[:, :Kt, :Kh].float())

        ext = torch.zeros_like(z_sys)

        for iteration in range(iteration_limit):
            ext_before = ext

            if iteration > 0:
                stored_ext = self.harq.read(ext, top_k=self.harq.cfg.top_k)
                # Per-position uncertainty from previous iteration's parity
                # residual magnitude. LearnedUncertainty has no persistent
                # state, so this serves as the HARQ weighting signal.
                uncertainty_scalar = parity_residual.float().pow(2).mean(dim=-1).to(z_sys.dtype)
                ext = self.harq.combine(ext, stored_ext, uncertainty_scalar)

            z_work = z_sys + ext

            layers_per_iter = self._layers_per_iter
            start = (iteration * layers_per_iter) % len(self.reasoning)
            for i in range(layers_per_iter):
                idx = (start + i) % len(self.reasoning)
                blk = self.reasoning[idx]
                if training:
                    freq_out = blk.freq(
                        z_work,
                        cached_w=w_shared,
                        cached_phase=phase_shared,
                        cached_w_dT=w_shared_dT,
                        cached_w_dH=w_shared_dH,
                        cached_phase_dT=phase_shared_dT,
                        cached_phase_dH=phase_shared_dH,
                    )
                    z_work = z_work + blk.freq_scale * freq_out
                    z_work = z_work + blk.ffn_scale * checkpoint(blk.ffn, blk.ffn_norm(z_work), use_reentrant=False)
                else:
                    z_work = blk(
                        z_work,
                        cached_w=w_shared,
                        cached_phase=phase_shared,
                        cached_w_dT=w_shared_dT,
                        cached_w_dH=w_shared_dH,
                        cached_phase_dT=phase_shared_dT,
                        cached_phase_dH=phase_shared_dH,
                    )

            z_pred = z_work

            residual, parity_computed = self.parity_checker(z_pred, parity_received)
            total_parity = total_parity + residual.pow(2).mean().to(total_parity.dtype)
            parity_residual = residual

            # Innovation: back-project residual from parity space M to systematic
            # space C via the transpose of the parity-check matrix H^T.
            # Normalized relative to z_pred magnitude for stable gradients.
            h_matrix = self.parity_checker.masked_weights  # [M, C]
            innovation = torch.einsum("mc,btm->btc", h_matrix, residual)
            z_scale = z_pred.float().pow(2).mean(dim=-1, keepdim=True).to(z_pred.dtype) + 1e-6
            innovation = innovation * (1.0 / torch.sqrt(z_scale)).clamp_max(4.0)

            # Learned per-position uncertainty (replaces Kalman predict/update).
            sigma2_pred = self.uncertainty(z_work)
            sigma2_meas = residual.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6)
            sigma2_meas = sigma2_meas.expand_as(sigma2_pred).to(z_work.dtype)
            z_corrected, k_gain = inverse_variance_update(z_work, innovation, sigma2_pred, sigma2_meas)

            correction = z_corrected - z_pred
            correction_gate = torch.sigmoid(
                residual.abs().float().mean(dim=-1, keepdim=True) * self.corr_gate_w + self.corr_gate_b
            ).to(z_pred.dtype)
            z_corrected = z_pred + correction_gate * correction

            mut_gate = torch.sigmoid(
                residual.abs().float().mean(dim=-1, keepdim=True) * self.corr_gate_w + self.corr_gate_b
            ).to(z_corrected.dtype)
            z_corrected = z_corrected + mut_gate * F.linear(
                F.silu(F.linear(z_corrected, self.mut_down_w, self.mut_down_b)), self.mut_up_w
            )

            ext_new = z_corrected - z_pred

            if mask is not None:
                ext_new = ext_new * mask.unsqueeze(-1)

            # Extrinsic-only: replace, not accumulate (turbo principle).
            # Each iteration produces ONLY new information; accumulation
            # causes divergence because old beliefs are re-broadcast.
            ext = ext_new
            ext_rms = ext.float().pow(2).mean(dim=-1, keepdim=True).to(ext.dtype) + 1e-6
            ext = ext * (1.0 / torch.sqrt(ext_rms)).clamp_max(4.0)

            ext_delta = ext - ext_before
            ext_norm = ext_delta.float().norm(dim=-1).mean()
            extrinsic_norms.append(ext_norm)

            self.harq.write(ext_new)

            if self.use_convergence_halt and iteration >= self.min_iters:
                _, mi = self.exit_estimator.should_halt(ext_before, ext, iteration)
                if bool(mi.item() < self.exit_estimator.threshold):
                    converged_at = iteration + 1
                    iterations_used = torch.full_like(iterations_used, iteration + 1)
                    break

        side_info = {
            "parity_strength": total_parity / max(converged_at, 1),
            "extrinsic_norms": extrinsic_norms,
            "iterations_used": iterations_used,
            "parity_residual": parity_residual,
        }

        state.harq_feedback = self.harq.serialize_feedback()
        state.iteration = converged_at

        decoded = z_sys + torch.tanh(self.ext_gate) * ext
        return DecodeResult(latent=decoded, state=state, side_info=side_info)


class HAGIv4(nn.Module):
    """HAGI V4 — Source-Channel Separation codec language model."""

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

        self.embed = nn.Embedding(m.vocab_size, H)
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(H))

        self.semantic_unknown_embed = nn.Parameter(torch.empty(H))
        nn.init.normal_(self.semantic_unknown_embed, mean=0.0, std=1.0 / math.sqrt(H))

        self.bottleneck_norm = RMSNorm(C, eps=m.norm_eps)
        self.use_lorentz = m.freq_coding.use_lorentz
        self.lorentz_sphere = LorentzSphereNorm(C, mode=m.freq_coding.lorentz_mode) if self.use_lorentz else None
        self.core_mask_embed = nn.Parameter(torch.zeros(C))
        nn.init.uniform_(self.core_mask_embed, -1.0 / math.sqrt(C), 1.0 / math.sqrt(C))

        self.cqi = CQIEstimator(H)
        self.cqi_bw_logit = nn.Parameter(torch.tensor([2.0]))
        self.cqi_mag_logit = nn.Parameter(torch.zeros(1))
        self.pilot_eq_strength = nn.Parameter(torch.zeros(1))

        n_bins_down = C // 2 + 1
        rolloff_start = int(n_bins_down * 3 / 4)
        t = torch.arange(max(1, n_bins_down - rolloff_start), dtype=torch.float32)
        t = t / max(n_bins_down - rolloff_start, 1)
        pass_logit = math.log(0.99 / 0.01)
        stop_logit = math.log(0.5 / 0.5)
        rc = torch.full((n_bins_down,), pass_logit)
        rc[rolloff_start:] = stop_logit + (pass_logit - stop_logit) * 0.5 * (1 + torch.cos(t * math.pi))
        self.bottleneck_gate = nn.Parameter(rc)

        n_bins_up = H // 2 + 1
        c_bins = C // 2 + 1
        rc_up = torch.full((n_bins_up,), stop_logit)
        rc_up[:c_bins] = pass_logit
        rolloff_up = max(1, int(c_bins / 4))
        n_rolloff = max(1, c_bins - rolloff_up)
        t_up = torch.arange(n_rolloff, dtype=torch.float32) / max(n_rolloff, 1)
        rc_up[rolloff_up:c_bins] = stop_logit + (pass_logit - stop_logit) * 0.5 * (1 + torch.cos(t_up * math.pi))
        self.bottleneck_up_gate = nn.Parameter(rc_up)

        self._pilot_idx_cache: dict[int, torch.Tensor] = {}
        self._pilot_mask_cache: dict[int, torch.Tensor] = {}

        self.use_freq_coding = m.freq_coding.enabled
        n_modes_t = m.freq_coding.n_modes_t
        n_modes_h = m.freq_coding.n_modes_h
        ffn_int_h = H * 2
        if self.use_freq_coding:
            n_heads_h = H // m.attention.head_dim
            self.perception = nn.ModuleList(
                FreqBlock(
                    H,
                    n_heads=n_heads_h,
                    head_dim=m.attention.head_dim,
                    n_modes_t=n_modes_t,
                    n_modes_h=n_modes_h,
                    ffn_intermediate=ffn_int_h,
                    T_max=m.attention.max_seq_len,
                    ffn_rank=H // 4,
                    norm_eps=m.norm_eps,
                    use_derivative=True,
                )
                for _ in range(m.perception_layers)
            )
        else:
            raise ValueError("freq_coding.enabled must be True")

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

        self.decoder = LDPCDecoder(
            self.turbo_config,
            C,
            n_parity_checks=self.codec_shape.n_parity_checks,
            edges_per_check=self.codec_shape.edges_per_check,
            exit_threshold=self.codec_shape.exit_threshold,
            shared_parity_weights=self.parity_encoder.parity_weights,
            shared_sparse_mask=self.parity_encoder.sparse_mask,
        )

        self.expression = self.perception

        self.final_norm = RMSNorm(H, eps=m.norm_eps)
        self.lm_head = nn.Linear(H, m.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        teacher_hidden = cfg.train.distill_teacher_hidden_size
        self.distill_align = nn.Identity() if teacher_hidden == H else nn.Linear(teacher_hidden, H, bias=False)

        self.multimodal_enabled = m.multimodal.enabled
        if self.multimodal_enabled:
            self.multimodal_input = MultimodalInput(cfg)
            self.multimodal_input.text_embed.weight = self.embed.weight
            self.cross_modal = CliffordCrossModal(
                H,
                gate_init=m.multimodal.cross_freq_gate_init,
                norm_eps=m.norm_eps,
            )
            self.contrastive = ContrastiveAlignment(
                H,
                temperature=m.multimodal.contrastive_temperature,
            )

        self._init_weights()
        self._init_turbo_weights()

    def _init_weights(self) -> None:
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "mut_" in name or mod is self.lm_head:
                    continue
                std = 1.0 / math.sqrt(max(1, mod.weight.shape[1]))
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                if mod is self.embed:
                    continue
                if self.multimodal_enabled and mod is self.multimodal_input.text_embed:
                    continue
                std = 1.0 / math.sqrt(max(1, mod.weight.shape[1]))
                nn.init.normal_(mod.weight, mean=0.0, std=std)

    def _init_turbo_weights(self) -> None:
        std_down = 1.0 / math.sqrt(max(1, self.decoder.mut_down_w.shape[1]))
        nn.init.normal_(self.decoder.mut_down_w, std=std_down)
        nn.init.zeros_(self.decoder.mut_down_b)
        nn.init.zeros_(self.decoder.mut_up_w)

    def train(self, mode: bool = True) -> HAGIv4:
        result = super().train(mode)
        if mode:
            for blk in self.perception:
                blk.reset_cache()
            for blk in self.decoder.reasoning:
                blk.reset_cache()
        return result

    def _chunked_ce(self, h: torch.Tensor, targets: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        B, T, H = h.shape
        h_flat = h.reshape(B * T, H)
        t_flat = targets.reshape(B * T)
        lm_dev = self.lm_head.weight.device
        total_loss = h_flat.new_zeros(())
        n = h_flat.shape[0]
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            logits_chunk = F.linear(h_flat[i:end].to(lm_dev), self.lm_head.weight)
            total_loss = total_loss + F.cross_entropy(logits_chunk, t_flat[i:end].to(lm_dev), reduction="sum")
        return total_loss / max(n, 1)

    def _freq_blocks_forward(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.perception:
            if self.training:
                # Match FreqBlock.forward semantics: apply LayerScale on both
                # freq and ffn branches. Use checkpoint only on ffn (dominant
                # activation memory).
                freq_out = blk.freq(h)
                h = h + blk.freq_scale * freq_out
                h = h + blk.ffn_scale * checkpoint(blk.ffn, blk.ffn_norm(h), use_reentrant=False)
            else:
                h = blk(h)
        return h

    def _get_pilot_idx(self, T: int, device: torch.device) -> torch.Tensor:
        spacing = self.codec_shape.pilot_spacing
        if T not in self._pilot_idx_cache:
            self._pilot_idx_cache[T] = torch.arange(0, T, spacing, device=device)
        return self._pilot_idx_cache[T]

    def _get_pilot_mask(self, T: int, device: torch.device) -> torch.Tensor:
        spacing = self.codec_shape.pilot_spacing
        if T not in self._pilot_mask_cache:
            pm = torch.ones(T, dtype=torch.bool, device=device)
            pm[::spacing] = False
            self._pilot_mask_cache[T] = pm
        return self._pilot_mask_cache[T]

    def _source_encode(
        self,
        input_ids: torch.Tensor | None,
        semantic_unknown_mask: torch.Tensor | None,
        cache: SpectralCache | None,
        awgn_sigma: float = 0.0,
        pre_encoded_h: torch.Tensor | None = None,
    ) -> SourceEncodeResult:
        if pre_encoded_h is not None:
            h = pre_encoded_h
            B, T, _ = h.shape
            cached_len = 0
            h = self._freq_blocks_forward(h)
        else:
            B, T = input_ids.shape
            embed_dev = self.embed.weight.device
            ids_dev = input_ids.device
            if embed_dev != ids_dev:
                h = self.embed(input_ids.to(embed_dev)).to(ids_dev)
            else:
                h = self.embed(input_ids)
            if semantic_unknown_mask is not None:
                unknown = self.semantic_unknown_embed.to(device=h.device, dtype=h.dtype)
                h = torch.where(semantic_unknown_mask.to(h.device).unsqueeze(-1), unknown, h)

            cached_len = 0
            if cache is not None and cache.context_len > 0:
                cached_h = cache.get_context(0)
                if cached_h is not None and cached_h.shape[0] == B and cached_h.shape[2] == h.shape[2]:
                    h = torch.cat([cached_h.to(h.dtype), h], dim=1)
                    cached_len = cached_h.shape[1]
            if cache is not None:
                cache.update_context(0, h, new_tokens=T)

            h = self._freq_blocks_forward(h)
            if cached_len > 0:
                h = h[:, cached_len:]
                B, T, _ = h.shape

        cqi = self.cqi(h)

        h_f = torch.fft.rfft(h.float(), dim=-1)
        n_bins = self._C // 2 + 1
        base_gate = torch.sigmoid(self.bottleneck_gate)
        cqi_expanded = cqi.unsqueeze(-1)
        bw_base = torch.sigmoid(self.cqi_bw_logit)
        bw_scale = bw_base + (1.0 - bw_base) * cqi_expanded
        bin_idx = torch.arange(n_bins, device=h.device, dtype=h.dtype)
        cutoff = n_bins * bw_scale
        sharpness = 1.0 + bw_base.abs() * n_bins
        dyn_mask = torch.sigmoid((cutoff - bin_idx) * sharpness)
        mag_scale = torch.sigmoid(self.cqi_mag_logit) + (1.0 - torch.sigmoid(self.cqi_mag_logit)) * cqi_expanded
        gate = base_gate * dyn_mask * mag_scale
        z = torch.fft.irfft(h_f[:, :, :n_bins] * gate, n=self._C, dim=-1).to(h.dtype)
        z = self.bottleneck_norm(z)
        if self.lorentz_sphere is not None:
            # Project to Lorentz hyperboloid (Minkowski sphere) and back to
            # tangent space: hyperbolic normalization that respects the
            # negatively-curved geometry of semantic space. The systematic
            # codeword stays in R^C (channel encoder expects Euclidean input).
            z = self.lorentz_sphere.inverse(self.lorentz_sphere(z))

        return SourceEncodeResult(systematic=z, mask=semantic_unknown_mask, cqi=cqi, pre_bottleneck=h)

    def _channel_encode(self, encoded: SourceEncodeResult) -> ChannelEncodeResult:
        systematic = encoded.systematic
        parity = self.parity_encoder(systematic)
        codeword = torch.cat([systematic, parity], dim=-1)
        # Interleaver spreads burst errors into random errors for the
        # iterative decoder. Applied unconditionally: during training the
        # erasure mask may contain bursts (span masking), and during
        # inference real-world corruption is often bursty.
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
        cqi_mean = encoded.cqi.mean()
        result = self.decoder(
            z_sys=z_sys,
            parity_received=parity,
            training=training,
            state=state,
            cqi_mean=cqi_mean,
            mask=ch_encoded.erasure_mask,
            refinement_iterations=refinement_iterations,
        )
        if ch_encoded.erasure_mask is not None and ch_encoded.erasure_mask.any():
            predicted = result.latent - z_sys
            target = ch_encoded.systematic - z_sys
            selected_predicted = predicted[ch_encoded.erasure_mask].float()
            selected_target = target[ch_encoded.erasure_mask].float()
            target_scale = selected_target.pow(2).mean().clamp_min(1e-6)
            normalized_mse = (selected_predicted - selected_target).pow(2).mean() / target_scale
            # Use MSE-only alignment when predicted is near-zero (init phase).
            # Cosine similarity explodes when ||predicted|| -> 0.
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
        z = decoded.latent
        B, T, _ = z.shape
        z_f = torch.fft.rfft(z.float(), dim=-1)
        z_pad = torch.zeros(B, T, self._H // 2 + 1, dtype=z_f.dtype, device=z.device)
        c_bins = self._C // 2 + 1
        mag_sigmoid = torch.sigmoid(self.cqi_mag_logit)
        gate = torch.sigmoid(self.bottleneck_up_gate[:c_bins]) * (
            mag_sigmoid + (1.0 - mag_sigmoid) * encoded.cqi
        ).unsqueeze(-1)
        z_pad[:, :, :c_bins] = z_f[:, :, :c_bins] * gate
        return self._freq_blocks_forward(torch.fft.irfft(z_pad, n=self._H, dim=-1).to(z.dtype))

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
    ) -> ModelOutput:
        if input_ids is None:
            raise ValueError("input_ids is required")
        masks = SemanticMaskBatch(
            semantic_unknown_mask,
            prediction_mask,
            valid_target_mask,
            physical_corruption_mask,
        )
        use_multimodal = self.multimodal_enabled and (images is not None or spectrograms is not None)
        mm_h: torch.Tensor | None = None
        modality_ids: torch.Tensor | None = None
        if use_multimodal:
            mm_h, modality_ids, _ = self.multimodal_input(
                input_ids=input_ids,
                images=images,
                spectrograms=spectrograms,
            )
        masks.validate(input_ids, mm_h.shape[:2] if mm_h is not None else None)
        state = cache.to_decode_state() if cache is not None else DecodeState(kalman_p=cached_p)
        if cache is not None and state.kalman_p is None:
            state.kalman_p = cached_p

        if use_multimodal:
            assert mm_h is not None
            text_len = input_ids.shape[1]
            unknown = self.semantic_unknown_embed.to(device=mm_h.device, dtype=mm_h.dtype)
            mm_h[:, :text_len] = torch.where(
                semantic_unknown_mask.to(mm_h.device).unsqueeze(-1),
                unknown,
                mm_h[:, :text_len],
            )
            mm_h = self.cross_modal(mm_h, modality_ids, self.multimodal_input.num_modalities)
            encoded = self._source_encode(None, semantic_unknown_mask, cache, awgn_sigma, pre_encoded_h=mm_h)
        else:
            encoded = self._source_encode(input_ids, semantic_unknown_mask, cache, awgn_sigma)

        chEncoded = self._apply_erasure(self._channel_encode(encoded), physical_corruption_mask, awgn_sigma)
        decoded = self._channel_decode(chEncoded, encoded, state, self.training, refinement_iterations)
        h = self._source_decode(decoded, encoded)

        if cache is not None:
            cache.update_decode_state(decoded.state)

        side_info = decoded.side_info
        rd_loss = (encoded.systematic.float() - decoded.latent.float()).pow(2).mean().to(h.dtype)

        h_normed = self.final_norm(h)
        lm_dev = self.lm_head.weight.device
        selected = prediction_mask & valid_target_mask
        prediction_indices = selected.flatten().nonzero(as_tuple=False).squeeze(-1)
        prediction_hidden = h_normed[:, : input_ids.shape[1]] if use_multimodal else h_normed
        selected_hidden = prediction_hidden.flatten(0, 1).index_select(0, prediction_indices.to(h_normed.device))
        logits = F.linear(selected_hidden.to(lm_dev), self.lm_head.weight).to(h_normed.device)
        if targets is not None:
            if prediction_indices.numel() == 0:
                raise ValueError("prediction_mask must select at least one target during training")
            selected_targets = targets.flatten().index_select(0, prediction_indices.to(targets.device))
            ce = F.cross_entropy(logits.to(lm_dev), selected_targets.to(lm_dev))
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

            # Recovery guarantee: parity must enable reconstruction under erasure.
            # Randomly erase systematic positions, then measure how well the
            # decoder recovers using parity. This trains the code to be
            # genuinely erasure-tolerant (MDS-like property).
            if self.training and physical_corruption_mask is not None and physical_corruption_mask.any():
                z_clean = encoded.systematic
                z_erased = chEncoded.codeword[..., : self._C]
                recovery_error = (z_clean.float() - z_erased.float()).pow(2)
                erased_error = recovery_error[physical_corruption_mask.to(recovery_error.device)]
                if erased_error.numel() > 0:
                    aux.parity_recovery = erased_error.mean().to(h.dtype)

        if use_multimodal and modality_ids is not None:
            aux.contrastive = self.contrastive(h, modality_ids)

        return ModelOutput(
            logits=logits,
            hidden=h,
            aux=aux,
            ce_loss=ce,
            iterations_used=side_info.get("iterations_used"),
            prediction_indices=prediction_indices,
        )
