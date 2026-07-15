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
from hagi_v4.model.codec_contracts import (
    ChannelEncodeResult,
    CodecShapeConfig,
    DecodeResult,
    DecodeState,
    InferenceShapeConfig,
    SourceEncodeResult,
    TurboDecodeConfig,
)
from hagi_v4.model.cqi import CQIEstimator
from hagi_v4.model.exit_chart import EXITChartEstimator
from hagi_v4.model.clifford_cross_modal import CliffordCrossModal
from hagi_v4.model.contrastive import ContrastiveAlignment
from hagi_v4.model.freq_layer import FreqBlock, FactoredSwiGLU
from hagi_v4.model.interleaver import BlockInterleaver
from hagi_v4.model.kalman import KalmanFilter
from hagi_v4.model.multimodal_input import MultimodalInput
from hagi_v4.model.msa import HARQBuffer
from hagi_v4.model.norms import RMSNorm
from hagi_v4.model.outputs import AuxLosses, ModelOutput, compute_whiteness_loss
from hagi_v4.model.sparse_parity import SparseParityChecker, SparseParityEncoder

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
                shared_weights=shared_w,
                shared_phase=shared_phase,
                shared_ffn=shared_ffn,
                norm_eps=cfg.norm_eps,
            )
            for _ in range(cfg.reasoning_layers)
        )

        self.parity_checker = SparseParityChecker(
            n_vars=hidden_size,
            n_checks=n_parity_checks,
            edges_per_check=edges_per_check,
            seed=42,
            norm_eps=cfg.norm_eps,
        )

        self.harq = HARQBuffer(cfg.msa, hidden_size)
        self.kalman = KalmanFilter(hidden_size)
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
        self.corr_gate_b = nn.Parameter(torch.zeros(1))

        self.ext_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z_sys: torch.Tensor,
        parity_received: torch.Tensor | None,
        training: bool,
        state: DecodeState,
        cqi_mean: torch.Tensor | float = 0.5,
        mask: torch.Tensor | None = None,
    ) -> DecodeResult:
        B, T, C = z_sys.shape

        total_parity = z_sys.new_zeros(())
        extrinsic_norms: list = []
        converged_at = self.n_iters
        iterations_used = torch.full((B, T), self.n_iters, dtype=torch.long, device=z_sys.device)
        parity_residual = torch.zeros_like(z_sys)

        p = state.kalman_p if state.kalman_p is not None else torch.ones(C, device=z_sys.device, dtype=z_sys.dtype)

        cqi_val = float(cqi_mean) if isinstance(cqi_mean, (int, float)) else cqi_mean.item()
        q_scale = 1.0 + (1.0 - cqi_val)

        self.harq.clear()
        if state.cache_active and state.harq_feedback is not None:
            self.harq.restore_feedback(state.harq_feedback)

        w_shared = None
        phase_shared = None
        if self.shared_w is not None:
            w_re = self.shared_w[0].float() @ self.shared_w[2].float()
            w_im = self.shared_w[1].float() @ self.shared_w[3].float()
            w_shared = torch.complex(w_re, w_im)
        if self.shared_phase is not None:
            Kt = min(self.shared_phase.shape[1], T)
            Kh = min(self.shared_phase.shape[2], self.reasoning[0].freq.head_dim)
            phase_shared = torch.exp(1j * self.shared_phase[:, :Kt, :Kh].float())

        q_cached = self.kalman._q(z_sys.dtype)
        r_cached = self.kalman._r(z_sys.dtype)

        ext = torch.zeros_like(z_sys)

        for iteration in range(self.n_iters):
            ext_before = ext

            if iteration > 0:
                stored_ext = self.harq.read(z_sys + ext, top_k=self.harq.cfg.top_k)
                uncertainty = p.unsqueeze(0).expand(B, T, -1).mean(dim=-1)
                ext = self.harq.combine(ext, stored_ext, uncertainty)

            z_work = z_sys + ext

            layers_per_iter = self._layers_per_iter
            start = (iteration * layers_per_iter) % len(self.reasoning)
            for i in range(layers_per_iter):
                idx = (start + i) % len(self.reasoning)
                blk = self.reasoning[idx]
                if training:
                    z_work = z_work + blk.freq(z_work, cached_w=w_shared, cached_phase=phase_shared)
                    z_work = z_work + checkpoint(blk.ffn, blk.ffn_norm(z_work), use_reentrant=False)
                else:
                    z_work = blk(z_work, cached_w=w_shared, cached_phase=phase_shared)

            z_pred = z_work

            p_pred = self.kalman.predict(p, q=q_cached)
            p_pred = p_pred * q_scale

            residual, parity_computed = self.parity_checker(z_pred, parity_received)
            total_parity = total_parity + residual.pow(2).mean().to(total_parity.dtype)
            parity_residual = residual

            innovation = residual.mean(dim=-1, keepdim=True).expand_as(z_pred)
            z_corrected, p = self.kalman.update(z_pred, innovation, p_pred, r=r_cached)

            correction = z_corrected - z_pred
            correction_gate = torch.sigmoid(
                residual.abs().float().mean(dim=-1, keepdim=True) * self.corr_gate_w + self.corr_gate_b
            ).to(z_pred.dtype)
            z_corrected = z_pred + correction_gate * correction

            z_corrected = z_corrected + F.linear(
                F.silu(F.linear(z_corrected, self.mut_down_w, self.mut_down_b)), self.mut_up_w
            )

            ext_new = z_corrected - z_pred

            if mask is not None:
                ext_new = ext_new * ~mask.unsqueeze(-1)

            ext = ext + ext_new
            ext = ext * torch.rsqrt(ext.float().pow(2).mean(dim=-1, keepdim=True).to(ext.dtype) + 1e-6)

            ext_delta = ext - ext_before
            ext_norm = ext_delta.float().norm(dim=-1).mean()
            extrinsic_norms.append(ext_norm)

            self.harq.write(ext_new)

            if self.use_convergence_halt and iteration >= self.min_iters:
                _, mi = self.exit_estimator.should_halt(ext_before, ext, iteration)
                if mi.item() < self.exit_estimator.threshold:
                    converged_at = iteration + 1
                    iterations_used = torch.full_like(iterations_used, iteration + 1)
                    break

        side_info = {
            "parity_strength": total_parity / max(converged_at, 1),
            "extrinsic_norms": extrinsic_norms,
            "iterations_used": iterations_used,
            "parity_residual": parity_residual,
        }

        state.kalman_p = p
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

        bound = 1.0 / math.sqrt(H)
        self.mask_embed = nn.Parameter(torch.empty(H))
        nn.init.uniform_(self.mask_embed, -bound, bound)

        self.bottleneck_norm = RMSNorm(C, eps=m.norm_eps)
        self.core_mask_embed = nn.Parameter(torch.zeros(C))
        nn.init.uniform_(self.core_mask_embed, -1.0 / math.sqrt(C), 1.0 / math.sqrt(C))

        self.cqi = CQIEstimator(H)
        self.cqi_bw_logit = nn.Parameter(torch.tensor([-3.0]))
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
        )

        self.expression = self.perception

        self.final_norm = RMSNorm(H, eps=m.norm_eps)
        self.lm_head = nn.Linear(H, m.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

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
        std_up = 1.0 / math.sqrt(max(1, self.decoder.mut_up_w.shape[1]))
        nn.init.normal_(self.decoder.mut_up_w, std=std_up)

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
                h = h + blk.freq(h)
                h = h + checkpoint(blk.ffn, blk.ffn_norm(h), use_reentrant=False)
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
        mask: torch.Tensor | None,
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

            if mask is not None:
                pilot_mask = self._get_pilot_mask(T, h.device)
                mask = mask & pilot_mask.unsqueeze(0)
                h = torch.where(mask.unsqueeze(-1), self.mask_embed.expand(B, T, -1), h)

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

        if self.training and awgn_sigma > 0.0:
            h = h + awgn_sigma * torch.randn_like(h)

        if mask is not None and T >= self.codec_shape.pilot_spacing:
            pilot_idx = self._get_pilot_idx(T, h.device)
            h_pilot_ref = h[:, pilot_idx].mean(dim=1, keepdim=True)
            h = h + self.pilot_eq_strength * (h_pilot_ref - h.mean(dim=1, keepdim=True))

        cqi = self.cqi(h)

        h_f = torch.fft.rfft(h.float(), dim=-1)
        n_bins = self._C // 2 + 1
        base_gate = torch.sigmoid(self.bottleneck_gate)
        cqi_expanded = cqi.unsqueeze(-1)
        bw_scale = 1.0 - torch.sigmoid(self.cqi_bw_logit) * cqi_expanded
        bin_idx = torch.arange(n_bins, device=h.device, dtype=h.dtype)
        cutoff = n_bins * bw_scale
        sharpness = 1.0 + torch.sigmoid(self.cqi_bw_logit).abs() * n_bins
        dyn_mask = torch.sigmoid((cutoff - bin_idx) * sharpness)
        mag_scale = torch.sigmoid(self.cqi_mag_logit) + (1.0 - torch.sigmoid(self.cqi_mag_logit)) * cqi_expanded
        gate = base_gate * dyn_mask * mag_scale
        z = torch.fft.irfft(h_f[:, :, :n_bins] * gate, n=self._C, dim=-1).to(h.dtype)
        z = self.bottleneck_norm(z)

        if mask is not None:
            z = torch.where(mask.unsqueeze(-1), self.core_mask_embed.expand(B, T, -1), z)

        return SourceEncodeResult(systematic=z, mask=mask, cqi=cqi, pre_bottleneck=h)

    def _channel_encode(self, encoded: SourceEncodeResult) -> ChannelEncodeResult:
        systematic = encoded.systematic
        parity = self.parity_encoder(systematic)
        codeword = torch.cat([systematic, parity], dim=-1)
        codeword = self.interleaver.interleave(codeword)
        return ChannelEncodeResult(codeword=codeword, systematic=systematic, parity=parity)

    def _channel_decode(
        self,
        ch_encoded: ChannelEncodeResult,
        encoded: SourceEncodeResult,
        state: DecodeState,
        training: bool,
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
            mask=None,
        )
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
        mask: torch.Tensor | None = None,
        step: int = 0,
        cached_p: torch.Tensor | None = None,
        cache: SpectralCache | None = None,
        awgn_sigma: float = 0.0,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
    ) -> ModelOutput:
        state = cache.to_decode_state() if cache is not None else DecodeState(kalman_p=cached_p)
        if cache is not None and state.kalman_p is None:
            state.kalman_p = cached_p

        use_multimodal = self.multimodal_enabled and (images is not None or spectrograms is not None)

        modality_ids: torch.Tensor | None = None
        if use_multimodal:
            mm_h, modality_ids, _ = self.multimodal_input(input_ids=input_ids, images=images, spectrograms=spectrograms)
            mm_h = self.cross_modal(mm_h, modality_ids, self.multimodal_input.num_modalities)
            encoded = self._source_encode(None, None, cache, awgn_sigma, pre_encoded_h=mm_h)
        else:
            encoded = self._source_encode(input_ids, mask, cache, awgn_sigma)

        chEncoded = self._channel_encode(encoded)
        decoded = self._channel_decode(chEncoded, encoded, state, self.training)
        h = self._source_decode(decoded, encoded)

        if cache is not None:
            cache.update_decode_state(decoded.state)

        side_info = decoded.side_info
        rd_loss = (encoded.pre_bottleneck.float() - h.float()).pow(2).mean().to(h.dtype)

        h_normed = self.final_norm(h)
        lm_dev = self.lm_head.weight.device
        if targets is not None and encoded.mask is not None:
            h_masked = h_normed[encoded.mask]
            if h_masked.shape[0] > 0:
                ce = F.cross_entropy(
                    F.linear(h_masked.to(lm_dev), self.lm_head.weight),
                    targets[encoded.mask].to(lm_dev),
                )
            else:
                ce = self._chunked_ce(h_normed, targets)
            logits = None
        elif targets is not None:
            ce = self._chunked_ce(h_normed, targets)
            logits = None
        else:
            logits = F.linear(h_normed.to(lm_dev), self.lm_head.weight).to(h_normed.device)
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
                if encoded.mask is not None:
                    valid = valid & ~encoded.mask
                aux.whiteness = compute_whiteness_loss(side_info["parity_residual"], valid)
            if side_info.get("extrinsic_norms") and len(side_info["extrinsic_norms"]) > 1:
                ext_sum = sum(side_info["extrinsic_norms"])
                aux.extrinsic_info = (ext_sum / len(side_info["extrinsic_norms"])).to(h.dtype)
            aux.rate_distortion = rd_loss

        if use_multimodal and modality_ids is not None:
            aux.contrastive = self.contrastive(h, modality_ids)

        return ModelOutput(
            logits=logits,
            hidden=h,
            aux=aux,
            ce_loss=ce,
            iterations_used=side_info.get("iterations_used"),
        )
