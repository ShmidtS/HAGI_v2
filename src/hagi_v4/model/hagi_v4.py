"""HAGI V7.1 — 5G NR-style pipeline with multimodal + spectral cache.

5G NR physical layer mapping:
  Embed + Mask     = Transport block + CRC (source coding + erasure)
  FreqBlock x N    = OFDM modulation (FFT/IFFT + equalization + QAM)
  FFT bottleneck   = Rate matching (frequency-domain puncturing, H->C)
  Turbo loop x K   = LDPC iterative decoding
    FreqBlock      = Component A (local OFDM equalization)
    GP2D           = Component B (parity check)
    Kalman         = Optimal channel estimation (per-position uncertainty)
    MSA            = DFE + HARQ buffer (channel memory)
  FFT zero-pad     = Rate dematching (C->H)
  LM head          = Demodulation -> bits

Multimodal (5G MIMO analog):
  MultimodalInput   = Separate source encoders per modality (CDMA)
  CrossModalFreqMix = MIMO cross-spectrum channel estimation
  CrossModalGP2D    = Multiple Description Coding parity
  CrossModalMSA     = Wyner-Ziv side information
  ContrastiveAlign  = InfoNCE modality alignment (Slepian-Wolf)

Spectral cache:
  SpectralCache = OFDM cyclic prefix analog for block-parallel inference
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
    CodecShapeConfig,
    DecodeResult,
    DecodeState,
    InferenceShapeConfig,
    RateMatchResult,
    SourceEncodeResult,
    TurboDecodeConfig,
)
from hagi_v4.model.cqi import CQIEstimator
from hagi_v4.model.freq_layer import FreqBlock
from hagi_v4.model.gp2d import GeometricProduct2D
from hagi_v4.model.kalman import KalmanFilter
from hagi_v4.model.msa import MSAModule
from hagi_v4.model.multiscale_gp2d import MultiScaleGP2D
from hagi_v4.model.norms import RMSNorm
from hagi_v4.model.outputs import AuxLosses, ModelOutput, compute_whiteness_loss

if TYPE_CHECKING:
    from hagi_v4.inference.spectral_cache import SpectralCache


class TurboLoop(nn.Module):
    """Turbo decoding loop — 5G LDPC + channel memory + Kalman.

    Channel memory techniques:
      MSA read (iter>0) = Decision Feedback Equalizer (DFE)
      MSA write          = HARQ buffer
      Kalman filter      = Optimal channel estimation
      GP2D               = LDPC parity check
      Convergence halt   = EXIT chart stopping criterion
    """

    def __init__(self, cfg: TurboDecodeConfig, hidden_size: int) -> None:
        super().__init__()
        self.n_iters = cfg.num_iterations
        self.min_iters = cfg.min_iterations
        self.convergence_threshold = cfg.convergence_threshold
        self.use_convergence_halt = cfg.use_convergence_halt
        self.tanh_scale = cfg.tanh_scale

        n_modes_t = cfg.freq_n_modes_t
        n_modes_h = cfg.freq_n_modes_h
        ffn_int = hidden_size * 2
        rank = cfg.freq_complex_rank
        head_dim = cfg.attention_head_dim
        n_heads = max(1, hidden_size // head_dim)
        proj_rank = hidden_size // 4
        ffn_rank = max(32, hidden_size // 4)

        shared_w = (
            nn.Parameter(torch.zeros(n_heads, head_dim, rank)),
            nn.Parameter(torch.zeros(n_heads, head_dim, rank)),
            nn.Parameter(torch.zeros(n_heads, rank, head_dim)),
            nn.Parameter(torch.zeros(n_heads, rank, head_dim)),
        )
        for p in shared_w:
            nn.init.normal_(p, std=0.02)
        shared_phase = nn.Parameter(torch.zeros(n_heads, n_modes_t, n_modes_h))
        nn.init.normal_(shared_phase, std=0.1)
        self.shared_w = nn.ParameterList(shared_w)
        self.shared_phase = shared_phase

        from hagi_v4.model.freq_layer import FactoredSwiGLU

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

        if cfg.gp2d.use_multiscale:
            self.gp2d = MultiScaleGP2D(
                cfg.gp2d,
                hidden_size,
                scales=cfg.gp2d.multiscale_windows,
                gate_inits=cfg.gp2d.multiscale_gate_inits,
                use_interleave=cfg.gp2d.use_interleave,
            )
        else:
            self.gp2d = GeometricProduct2D(cfg.gp2d, hidden_size)

        self.msa = MSAModule(cfg.msa, hidden_size)

        self.kalman = KalmanFilter(hidden_size)
        self._layers_per_iter = max(2, cfg.reasoning_layers // 2)

        mut_rank = 32
        self.mut_down_w = nn.Parameter(torch.empty(mut_rank, hidden_size))
        self.mut_down_b = nn.Parameter(torch.empty(mut_rank))
        self.mut_up_w = nn.Parameter(torch.empty(hidden_size, mut_rank))
        self.mut_rank = mut_rank

        self.corr_gate_w = nn.Parameter(torch.tensor(5.0))
        self.corr_gate_b = nn.Parameter(torch.tensor(-1.0))
        self.write_gate_w = nn.Parameter(torch.tensor(5.0))
        self.sic_w = nn.Parameter(torch.tensor(10.0))
        self.sic_b = nn.Parameter(torch.tensor(-10.0))
        self.harq_gate = nn.Parameter(torch.tensor(-5.0))

    def forward(
        self,
        z: torch.Tensor,
        training: bool,
        state: DecodeState,
        cqi_mean: torch.Tensor | float = 0.5,
        mask: torch.Tensor | None = None,
    ) -> DecodeResult:
        B, T, C = z.shape

        total_parity = z.new_zeros(())
        total_msa_lb = z.new_zeros(())
        extrinsic_norms: list = []
        converged_at = self.n_iters
        iterations_used = torch.full((B, T), self.n_iters, dtype=torch.long, device=z.device)
        gp2d_residual = torch.zeros_like(z)
        prev_residual = None

        p = state.kalman_p if state.kalman_p is not None else torch.ones(C, device=z.device, dtype=z.dtype)

        base_k = self.msa.cfg.top_k
        cqi_val = float(cqi_mean) if isinstance(cqi_mean, (int, float)) else cqi_mean.item()
        adaptive_k = base_k + int((1.0 - cqi_val) * 4)

        q_scale = 1.0 + (1.0 - cqi_val)

        self.msa.clear()
        if state.cache_active and state.msa_feedback is not None:
            self.msa.restore_feedback(state.msa_feedback)

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
        q_cached = self.kalman._q(z.dtype)
        r_cached = self.kalman._r(z.dtype)

        for iteration in range(self.n_iters):
            h_prior = z

            if iteration > 0:
                msa_out, lb = self.msa.read(z, top_k=adaptive_k)
                if prev_residual is not None:
                    uncertainty = prev_residual.abs().float().mean(dim=-1)
                    harq_alpha = (torch.sigmoid(self.harq_gate.float()) * torch.sigmoid(uncertainty * 10.0)).to(z.dtype)
                    z = z + harq_alpha.unsqueeze(-1) * msa_out
                else:
                    z = z + msa_out
                total_msa_lb = total_msa_lb + lb

            layers_per_iter = self._layers_per_iter
            start = (iteration * layers_per_iter) % len(self.reasoning)
            for i in range(layers_per_iter):
                idx = (start + i) % len(self.reasoning)
                blk = self.reasoning[idx]
                if training:
                    z = z + blk.freq(z, cached_w=w_shared, cached_phase=phase_shared)
                    z = z + checkpoint(blk.ffn, blk.ffn_norm(z), use_reentrant=False)
                else:
                    z = blk(z, cached_w=w_shared, cached_phase=phase_shared)

            z_pred = z

            p_pred = self.kalman.predict(p, q=q_cached)
            p_pred = p_pred * q_scale

            z_meas, gp2d_residual = self.gp2d(z_pred)
            total_parity = total_parity + gp2d_residual.pow(2).mean().to(total_parity.dtype)

            innovation = z_meas - z_pred
            r_decay = 0.5**iteration
            z_kalman, p = self.kalman.update(z_pred, innovation, p_pred, r_scale=r_decay, r=r_cached)

            correction = z_kalman - z_pred
            correction_gate = torch.sigmoid(gp2d_residual.abs().float() * self.corr_gate_w + self.corr_gate_b).to(
                z.dtype
            )
            z = z_pred + correction_gate * correction

            z = z + F.linear(F.silu(F.linear(z, self.mut_down_w, self.mut_down_b)), self.mut_up_w)

            z = z / (1.0 + z.float().square().mean(dim=-1, keepdim=True) / self.tanh_scale**2).to(z.dtype).sqrt()

            if training and iteration < self.n_iters - 1:
                confidence = 1.0 / (1.0 + gp2d_residual.abs().float().mean(dim=-1))
                sic_gate = torch.sigmoid(confidence * self.sic_w.float() + self.sic_b.float()).to(z.dtype)
                z = z * (1 - sic_gate.unsqueeze(-1)) + z.detach() * sic_gate.unsqueeze(-1)

            ext_norm = innovation.float().norm(dim=-1).mean()
            extrinsic_norms.append(ext_norm)

            innovation_harq = z - h_prior
            inn_mag = innovation_harq.float().norm(dim=-1, keepdim=True)
            if mask is not None:
                valid = ~mask
                inn_mean = (inn_mag * valid.unsqueeze(-1).float()).sum() / valid.float().sum()
            else:
                inn_mean = inn_mag.mean()
            write_gate = torch.sigmoid((inn_mag - inn_mean) * self.write_gate_w).to(innovation_harq.dtype)
            self.msa.write(innovation_harq * write_gate)

            prev_residual = gp2d_residual

            combined_ext = ext_norm
            if self.use_convergence_halt and iteration >= self.min_iters:
                if combined_ext < self.convergence_threshold:
                    converged_at = iteration + 1
                    iterations_used = torch.full_like(iterations_used, iteration + 1)
                    break

        side_info = {
            "parity_strength": total_parity / max(converged_at, 1),
            "extrinsic_norms": extrinsic_norms,
            "iterations_used": iterations_used,
            "msa_lb": total_msa_lb / max(converged_at, 1),
            "gp2d_residual": gp2d_residual,
        }

        state.kalman_p = p
        state.msa_feedback = self.msa.serialize_feedback()
        state.iteration = converged_at
        return DecodeResult(latent=z, state=state, side_info=side_info)


class HAGIv4(nn.Module):
    """HAGI V7.1 — 5G NR-style codec language model with multimodal."""

    def __init__(self, cfg: HAGIv4Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.codec_shape = CodecShapeConfig.from_hagi_config(cfg)
        self.turbo_config = TurboDecodeConfig.from_hagi_config(cfg)
        self.inference_config = InferenceShapeConfig.from_hagi_config(cfg)
        H = m.hidden_size
        C = m.core_hidden_size
        self.use_multimodal = m.multimodal.enabled

        if self.use_multimodal:
            from hagi_v4.model.multimodal_input import MultimodalInput

            self.mm_input = MultimodalInput(cfg)
            self.embed = self.mm_input.text_embed
        else:
            self.embed = nn.Embedding(m.vocab_size, H)

        self.mask_embed = nn.Parameter(torch.zeros(H))

        self.bottleneck_norm = RMSNorm(C, eps=m.norm_eps)
        self.core_mask_embed = nn.Parameter(torch.zeros(C))
        self._H = H
        self._C = C

        self.cqi = CQIEstimator(H)

        self.pilot_eq_strength = nn.Parameter(torch.tensor(0.1))

        n_bins_down = C // 2 + 1
        rolloff_start = int(n_bins_down * 0.75)
        t = torch.arange(n_bins_down - rolloff_start, dtype=torch.float32)
        t = t / max(n_bins_down - rolloff_start, 1)
        rc = torch.ones(n_bins_down)
        rc[rolloff_start:] = 0.5 * (1 + torch.cos(t * math.pi))
        self.bottleneck_gate = nn.Parameter(rc)

        n_bins_up = H // 2 + 1
        c_bins = C // 2 + 1
        rc_up = torch.zeros(n_bins_up)
        rc_up[:c_bins] = 1.0
        rolloff_up = max(1, int(c_bins * 0.25))
        n_rolloff = c_bins - rolloff_up
        t_up = torch.arange(n_rolloff, dtype=torch.float32) / max(n_rolloff, 1)
        rc_up[rolloff_up:c_bins] = 0.5 * (1 + torch.cos(t_up * math.pi))
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
            raise ValueError("V7.1 requires freq_coding.enabled=True (attention fallback removed)")

        self.turbo = TurboLoop(self.turbo_config, C)

        self.expression = self.perception

        self.final_norm = RMSNorm(H, eps=m.norm_eps)
        self.lm_head = nn.Linear(H, m.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        if self.use_multimodal:
            from hagi_v4.model.contrastive import ContrastiveAlignment
            from hagi_v4.model.cross_modal_attention import CrossModalFreqMix
            from hagi_v4.model.cross_modal_gp2d import CrossModalGP2D
            from hagi_v4.model.cross_modal_msa import CrossModalMSA

            self.cross_freq = CrossModalFreqMix(
                H,
                n_heads=n_heads_h,
                head_dim=m.attention.head_dim,
                gate_init=m.multimodal.cross_freq_gate_init,
                norm_eps=m.norm_eps,
            )
            self.cross_gp2d = CrossModalGP2D(
                self.turbo_config.gp2d,
                C,
                num_modalities=m.multimodal.num_modalities,
                gate_init=m.multimodal.cross_gp2d_gate_init,
            )
            self.cross_msa = CrossModalMSA(self.turbo_config.msa, C, num_modalities=m.multimodal.num_modalities)
            self.contrastive = ContrastiveAlignment(C, temperature=m.multimodal.contrastive_temperature)

        self._init_weights()
        self._init_mask_embeds()
        nn.init.normal_(self.turbo.mut_down_w, std=0.02)
        nn.init.zeros_(self.turbo.mut_down_b)
        nn.init.zeros_(self.turbo.mut_up_w)

        if cfg.train.freeze_embeddings:
            self.embed.weight.requires_grad_(False)

    def train(self, mode: bool = True) -> HAGIv4:
        result = super().train(mode)
        if mode:
            for blk in self.perception:
                blk.reset_cache()
            for blk in self.turbo.reasoning:
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

    def _init_weights(self) -> None:
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "mut_" in name or mod is self.lm_head:
                    continue
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                if mod is self.embed:
                    continue
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)

    def _init_mask_embeds(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.mask_embed, std=0.02)
            nn.init.normal_(self.core_mask_embed, std=0.02)

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
        images: torch.Tensor | None,
        spectrograms: torch.Tensor | None,
        modality_ids: torch.Tensor | None,
        cache: SpectralCache | None,
        awgn_sigma: float = 0.0,
    ) -> SourceEncodeResult:
        if self.use_multimodal and (images is not None or spectrograms is not None):
            h, modality_ids, _ = self.mm_input(input_ids, images, spectrograms)
            B, T, _ = h.shape
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
        if self.use_multimodal and modality_ids is not None:
            h = self.cross_freq(h, modality_ids, num_modalities=self.codec_shape.num_modalities)
        if mask is not None and T >= self.codec_shape.pilot_spacing:
            pilot_idx = self._get_pilot_idx(T, h.device)
            h_pilot_ref = h[:, pilot_idx].mean(dim=1, keepdim=True)
            h = h + self.pilot_eq_strength * (h_pilot_ref - h.mean(dim=1, keepdim=True))
        return SourceEncodeResult(h, mask, modality_ids, self.cqi(h), h)

    def _rate_match(self, encoded: SourceEncodeResult) -> RateMatchResult:
        h = encoded.source
        B, T, _ = h.shape
        h_f = torch.fft.rfft(h.float(), dim=-1)
        n_bins = self._C // 2 + 1
        base_gate = torch.sigmoid(self.bottleneck_gate)
        cqi = encoded.cqi.unsqueeze(-1)
        bw_scale = 1.0 - 0.15 * cqi
        bin_idx = torch.arange(n_bins, device=h.device, dtype=h.dtype)
        cutoff = n_bins * bw_scale
        dyn_mask = torch.sigmoid((cutoff - bin_idx) * 6.0)
        gate = base_gate * dyn_mask * (0.5 + 0.5 * cqi)
        z = torch.fft.irfft(h_f[:, :, :n_bins] * gate, n=self._C, dim=-1).to(h.dtype)
        z = self.bottleneck_norm(z)
        if encoded.mask is not None:
            z = torch.where(encoded.mask.unsqueeze(-1), self.core_mask_embed.expand(B, T, -1), z)
        return RateMatchResult(z, encoded)

    def _turbo_decode(self, matched: RateMatchResult, state: DecodeState, training: bool) -> DecodeResult:
        result = self.turbo(
            matched.latent,
            training,
            state,
            matched.source.cqi.mean(),
            matched.source.mask,
        )
        z = result.latent
        modality_ids = matched.source.modality_ids
        if self.use_multimodal and modality_ids is not None:
            z, _ = self.cross_gp2d(z, modality_ids)
            self.cross_msa.clear()
            self.cross_msa.write(z, modality_ids)
            side_info_q = z + self.cross_msa.read_cross(
                z, modality_ids, target_modality=0, top_k=self.codec_shape.msa_top_k
            )
            z = z + 0.1 * (side_info_q - z)
            result.latent = z
        return result

    def _source_decode(self, decoded: DecodeResult, encoded: SourceEncodeResult) -> torch.Tensor:
        z = decoded.latent
        B, T, _ = z.shape
        z_f = torch.fft.rfft(z.float(), dim=-1)
        z_pad = torch.zeros(B, T, self._H // 2 + 1, dtype=z_f.dtype, device=z.device)
        c_bins = self._C // 2 + 1
        gate = torch.sigmoid(self.bottleneck_up_gate[:c_bins]) * (0.5 + 0.5 * encoded.cqi).unsqueeze(-1)
        z_pad[:, :, :c_bins] = z_f[:, :, :c_bins] * gate
        return self._freq_blocks_forward(torch.fft.irfft(z_pad, n=self._H, dim=-1).to(z.dtype))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        step: int = 0,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        cached_p: torch.Tensor | None = None,
        cache: SpectralCache | None = None,
        awgn_sigma: float = 0.0,
    ) -> ModelOutput:
        state = cache.to_decode_state() if cache is not None else DecodeState(kalman_p=cached_p)
        if cache is not None and state.kalman_p is None:
            state.kalman_p = cached_p
        encoded = self._source_encode(input_ids, mask, images, spectrograms, modality_ids, cache, awgn_sigma)
        matched = self._rate_match(encoded)
        decoded = self._turbo_decode(matched, state, self.training)
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
            if side_info["parity_strength"] is not None:
                aux.parity = side_info["parity_strength"]
            if self.codec_shape.use_whiteness_loss and side_info.get("gp2d_residual") is not None:
                pilot_mask = self._get_pilot_mask(
                    side_info["gp2d_residual"].shape[1],
                    side_info["gp2d_residual"].device,
                )
                valid = pilot_mask.unsqueeze(0).expand(side_info["gp2d_residual"].shape[0], -1)
                if encoded.mask is not None:
                    valid = valid & ~encoded.mask
                aux.whiteness = compute_whiteness_loss(side_info["gp2d_residual"], valid)
            if side_info.get("extrinsic_norms") and len(side_info["extrinsic_norms"]) > 1:
                ext_sum = sum(side_info["extrinsic_norms"])
                aux.extrinsic_info = (ext_sum / len(side_info["extrinsic_norms"])).to(h.dtype)
            if side_info.get("iterations_used") is not None:
                aux.efficiency = side_info["iterations_used"].float().mean()
            if side_info.get("msa_lb") is not None:
                aux.msa_lb = side_info["msa_lb"]
            aux.rate_distortion = rd_loss

            if self.use_multimodal and encoded.modality_ids is not None and hasattr(self, "contrastive"):
                aux.contrastive = self.contrastive(h, encoded.modality_ids)

        return ModelOutput(
            logits=logits,
            hidden=h,
            aux=aux,
            ce_loss=ce,
            iterations_used=side_info.get("iterations_used"),
        )
