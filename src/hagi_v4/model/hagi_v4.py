"""HAGI V7 — 5G NR-style pipeline.

5G NR physical layer mapping:
  Embed + Mask     = Transport block + CRC (source coding + erasure)
  FreqBlock × N    = OFDM modulation (FFT/IFFT + equalization + QAM)
  FFT bottleneck   = Rate matching (frequency-domain puncturing, H→C)
  Turbo loop × K   = LDPC iterative decoding
    FreqBlock      = Component A (local OFDM equalization)
    GP2D           = Component B (parity check)
    Kalman         = Optimal channel estimation (per-position uncertainty)
    MSA            = DFE + HARQ buffer (channel memory)
  FFT zero-pad     = Rate dematching (C→H)
  LM head          = Demodulation → bits

Retained from V5 (with 5G analog):
  MSA (DFE + HARQ buffer), pilot-aided equalization, CQI adaptive coding
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.gp2d import GeometricProduct2D
from hagi_v4.model.multiscale_gp2d import MultiScaleGP2D
from hagi_v4.model.msa import MSAModule
from hagi_v4.model.norms import RMSNorm, build_rope_cache
from hagi_v4.model.outputs import AuxLosses, ModelOutput, compute_whiteness_loss
from hagi_v4.model.transformer_block import TransformerBlock
from hagi_v4.model.freq_layer import FreqBlock
from hagi_v4.model.kalman import KalmanFilter
from hagi_v4.model.cqi import CQIEstimator


class TurboLoop(nn.Module):
    """Simplified turbo decoding loop — 5G LDPC + channel memory + Kalman.

    Channel memory techniques:
      MSA read (iter>0) = Decision Feedback Equalizer (DFE) — past decisions cancel ISI
      MSA write          = HARQ buffer — store parity-checked states for combining
      Kalman filter      = Optimal channel estimation (blend prediction + measurement)
      GP2D               = LDPC parity check
      Convergence halt   = EXIT chart stopping criterion

    No z_H/z_L, no GDR, no deep supervision, no coherence.
    """

    def __init__(self, cfg: HAGIv4Config, hidden_size: int) -> None:
        super().__init__()
        m = cfg.model
        self.n_iters = m.refinement.num_iterations
        self.min_iters = m.refinement.min_iterations
        self.convergence_threshold = m.refinement.convergence_threshold
        self.use_convergence_halt = m.refinement.use_convergence_halt

        n_modes_t = m.freq_coding.n_modes_t
        n_modes_h = m.freq_coding.n_modes_h
        ffn_int = hidden_size * 2
        rank = 16
        head_dim = m.attention.head_dim
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
                T_max=m.attention.max_seq_len,
                rank=rank,
                proj_rank=proj_rank,
                ffn_rank=ffn_rank,
                shared_weights=shared_w,
                shared_phase=shared_phase,
                shared_ffn=shared_ffn,
                norm_eps=m.norm_eps,
            )
            for _ in range(m.reasoning_layers)
        )

        if m.gp2d.use_multiscale:
            self.gp2d = MultiScaleGP2D(
                m.gp2d,
                hidden_size,
                scales=m.gp2d.multiscale_windows,
                gate_inits=m.gp2d.multiscale_gate_inits,
                use_interleave=m.gp2d.use_interleave,
            )
        else:
            self.gp2d = GeometricProduct2D(m.gp2d, hidden_size)

        self.msa = MSAModule(m.msa, hidden_size)

        self.kalman = KalmanFilter(hidden_size)

    def forward(
        self,
        z: torch.Tensor,
        training: bool,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cqi_mean: float = 0.5,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, T, C = z.shape

        total_parity = z.new_zeros(())
        total_msa_lb = z.new_zeros(())
        extrinsic_norms: list = []
        converged_at = self.n_iters
        iterations_used = torch.full((B, T), self.n_iters, dtype=torch.long, device=z.device)
        gp2d_residual = torch.zeros_like(z)

        # Per-position Kalman state: diagonal covariance P [B,T,C] in float32
        p = torch.ones(C, device=z.device, dtype=z.dtype)

        # Adaptive DFE taps: bad channel -> more feedback taps (5G adaptive equalizer)
        base_k = self.msa.cfg.top_k
        adaptive_k = base_k + int((1.0 - cqi_mean) * 4)

        # Adaptive process noise: bad channel -> higher Q (5G adaptive coding)
        q_scale = 1.0 + (1.0 - cqi_mean)

        self.msa.clear()

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
        tanh_scale = 10.0
        q_cached = torch.exp(self.kalman.log_q).to(z.dtype)
        r_cached = torch.exp(self.kalman.log_r).to(z.dtype)

        for iteration in range(self.n_iters):
            h_prior = z

            if iteration > 0:
                msa_out, lb = self.msa.read(z, top_k=adaptive_k)
                z = z + msa_out
                total_msa_lb = total_msa_lb + lb

            layers_per_iter = max(2, len(self.reasoning) // 2)
            start = (iteration * layers_per_iter) % len(self.reasoning)
            for i in range(layers_per_iter):
                idx = (start + i) % len(self.reasoning)
                blk = self.reasoning[idx]
                if training:
                    z = z + blk.freq(z, cos, sin, cached_w=w_shared, cached_phase=phase_shared)
                    z = z + checkpoint(blk.ffn, blk.ffn_norm(z), use_reentrant=False)
                else:
                    z, _, _ = blk(z, cos, sin, cached_w=w_shared, cached_phase=phase_shared)

            z_pred = z

            p_pred = self.kalman.predict(p, q=q_cached)
            p_pred = p_pred * q_scale

            z_meas, gp2d_residual = self.gp2d(z_pred)
            total_parity = total_parity + gp2d_residual.pow(2).mean().to(total_parity.dtype)

            r_decay = 0.5**iteration
            z_kalman, p = self.kalman.update(z_pred, z_meas, p_pred, r_scale=r_decay, r=r_cached)

            correction = z_kalman - z_pred
            correction_gate = torch.sigmoid(gp2d_residual.abs().float() * 5 - 1).to(z.dtype)
            z = z_pred + correction_gate * correction

            z = tanh_scale * torch.tanh(z / tanh_scale)

            ext_norm = (z_meas - z_pred).float().norm(dim=-1).mean().detach()
            extrinsic_norms.append(ext_norm)

            innovation = z - h_prior
            inn_mag = innovation.float().norm(dim=-1, keepdim=True)
            write_gate = torch.sigmoid((inn_mag - inn_mag.mean()) * 5).to(innovation.dtype)
            self.msa.write(innovation * write_gate)

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

        return z, side_info


class HAGIv4(nn.Module):
    """HAGI V7 — 5G NR-style codec language model."""

    def __init__(self, cfg: HAGIv4Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        H = m.hidden_size
        C = m.core_hidden_size

        self.embed = nn.Embedding(m.vocab_size, H)
        self.mask_embed = nn.Parameter(torch.zeros(H))

        # Rate matching: raised-cosine FFT truncation (5G pulse shaping)
        # Hard truncation = brick-wall filter = Gibbs ringing (echo)
        # Raised-cosine = smooth rolloff = no echo (5G standard)
        self.bottleneck_norm = RMSNorm(C, eps=m.norm_eps)
        self.core_mask_embed = nn.Parameter(torch.zeros(C))
        self._H = H
        self._C = C

        # CQI estimator (5G adaptive modulation/coding)
        self.cqi = CQIEstimator(H)

        # Learnable pilot equalization strength (5G channel estimation weight)
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

        # OFDM modulation: FreqBlocks at full bandwidth (perception merged)
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
            self.perception = nn.ModuleList(TransformerBlock(m, hidden_size=H) for _ in range(m.perception_layers))

        # Turbo decoding loop (LDPC iterative decode)
        self.turbo = TurboLoop(cfg, C)

        # Expression: tied with perception (shared OFDM tx/rx hardware)
        if self.use_freq_coding:
            self.expression = self.perception
        else:
            self.expression = nn.ModuleList(TransformerBlock(m, hidden_size=H) for _ in range(m.expression_layers))

        self.final_norm = RMSNorm(H, eps=m.norm_eps)
        self.lm_head = nn.Linear(H, m.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()
        self._init_mask_embeds()

    def _chunked_ce(self, h: torch.Tensor, targets: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        """Chunked cross-entropy — avoids materializing full [B,T,V] logits.

        5G analog: block-wise demodulation — process subcarriers in chunks
        to fit in receiver buffer. Full logits [B,T,V] = 805MB at B=8,T=512.
        Chunked: only [chunk, V] in memory at a time.
        """
        B, T, H = h.shape
        h_flat = h.reshape(B * T, H)
        t_flat = targets.reshape(B * T)
        total_loss = h_flat.new_zeros(())
        n = h_flat.shape[0]
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            logits_chunk = F.linear(h_flat[i:end], self.lm_head.weight)
            total_loss = total_loss + F.cross_entropy(logits_chunk, t_flat[i:end], reduction="sum")
        return total_loss / max(n, 1)

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)

    def _init_mask_embeds(self) -> None:
        with torch.no_grad():
            self.mask_embed.data.copy_(self.embed.weight.mean(dim=0))
            me_f = torch.fft.rfft(self.mask_embed.data.float())
            gate = torch.sigmoid(self.bottleneck_gate)
            me_f_c = me_f[: self._C // 2 + 1] * gate.float()
            self.core_mask_embed.data.copy_(torch.fft.irfft(me_f_c, n=self._C).to(self.mask_embed.dtype))

    def _freq_blocks_forward(self, h: torch.Tensor, cos, sin) -> torch.Tensor:
        for blk in self.perception:
            if self.training:
                h = h + blk.freq(h, cos, sin)
                h = h + checkpoint(blk.ffn, blk.ffn_norm(h), use_reentrant=False)
            else:
                h, _, _ = blk(h, cos, sin)
        return h

    def _get_pilot_idx(self, T: int, device: torch.device) -> torch.Tensor:
        if T not in self._pilot_idx_cache:
            self._pilot_idx_cache[T] = torch.arange(0, T, 8, device=device)
        return self._pilot_idx_cache[T]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        step: int = 0,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
    ) -> ModelOutput:
        B, T = input_ids.shape

        # 1. Source coding: token -> embedding
        h = self.embed(input_ids)
        if mask is not None:
            # Pilot symbols (5G): every 8th position is never masked
            # These serve as channel estimation references for MSA/Kalman
            pilot_mask = torch.ones(T, dtype=torch.bool, device=input_ids.device)
            pilot_mask[::8] = False  # every 8th position is pilot
            mask = mask & pilot_mask.unsqueeze(0)
            h = torch.where(mask.unsqueeze(-1), self.mask_embed.expand(B, T, -1), h)

        # 2. OFDM pre-equalization (perception)
        cos, sin = (None, None) if self.use_freq_coding else self._rope(T, h.device, h.dtype)
        h = self._freq_blocks_forward(h, cos, sin)

        if mask is not None and T >= 8:
            pilot_idx = self._get_pilot_idx(T, h.device)
            h_pilot_ref = h[:, pilot_idx].mean(dim=1, keepdim=True)
            h_mean = h.mean(dim=1, keepdim=True)
            h = h + self.pilot_eq_strength * (h_pilot_ref - h_mean)

        # CQI: channel quality indicator (5G adaptive modulation/coding)
        cqi = self.cqi(h)  # [B, T] in [0, 1]
        cqi_mean_t = cqi.mean()
        cqi_mean = cqi_mean_t.item()

        # Save pre-bottleneck for rate-distortion loss
        h_pre_bottleneck = h

        # 3. Rate matching: raised-cosine FFT truncation H -> C
        # Per-frequency CQI gating (5G per-subcarrier AMC): low-rank projection
        h_f = torch.fft.rfft(h.float(), dim=-1)
        gate = torch.sigmoid(self.bottleneck_gate) * (0.5 + 0.5 * cqi_mean_t)
        h_f_c = h_f[:, :, : self._C // 2 + 1] * gate.float()
        z = torch.fft.irfft(h_f_c, n=self._C, dim=-1).to(h.dtype)
        z = self.bottleneck_norm(z)
        if mask is not None:
            z = torch.where(mask.unsqueeze(-1), self.core_mask_embed.expand(B, T, -1), z)

        # 4. LDPC iterative decoding (turbo loop, adaptive parameters)
        z, side_info = self.turbo(
            z,
            training=self.training,
            cos=cos,
            sin=sin,
            cqi_mean=cqi_mean,
            mask=mask,
        )

        # 5. Rate dematching: raised-cosine zero-padding C -> H
        # CQI-adaptive (symmetric with Stage 3): high CQI -> wider reconstruction
        z_f = torch.fft.rfft(z.float(), dim=-1)
        z_pad = torch.zeros(B, T, self._H // 2 + 1, dtype=z_f.dtype, device=z.device)
        c_bins = self._C // 2 + 1
        gate_up = torch.sigmoid(self.bottleneck_up_gate) * (0.5 + 0.5 * cqi_mean_t)
        z_pad[:, :, :c_bins] = z_f * gate_up[:c_bins].float()
        h = torch.fft.irfft(z_pad, n=self._H, dim=-1).to(z.dtype)

        # 6. Demodulation refinement (expression = 5G demapper)
        h = self._freq_blocks_forward(h, cos, sin)

        rd_loss = (h_pre_bottleneck.float() - h.float()).pow(2).mean().to(h.dtype)

        # 7. Final demod -> bits (chunked to avoid materializing full logits)
        h_normed = self.final_norm(h)
        mask_valid = mask is not None and mask.any().item() if mask is not None else False
        if targets is not None and mask_valid:
            h_masked = h_normed[mask]
            t_masked = targets[mask]
            ce = F.cross_entropy(F.linear(h_masked, self.lm_head.weight), t_masked)
            logits = None
        elif targets is not None:
            ce = self._chunked_ce(h_normed, targets)
            logits = None
        else:
            logits = F.linear(h_normed, self.lm_head.weight)
            ce = None

        # 7. Auxiliary losses (minimal — only parity + whiteness)
        aux = AuxLosses()
        if targets is not None:
            if side_info["parity_strength"] is not None:
                aux.parity = side_info["parity_strength"]
            if self.cfg.model.gp2d.use_whiteness_loss and side_info.get("gp2d_residual") is not None:
                aux.whiteness = compute_whiteness_loss(side_info["gp2d_residual"])
            if side_info.get("extrinsic_norms") and len(side_info["extrinsic_norms"]) > 1:
                aux.extrinsic_info = torch.tensor(
                    sum(side_info["extrinsic_norms"]) / len(side_info["extrinsic_norms"]),
                    device=h.device,
                    dtype=h.dtype,
                )
            if side_info.get("iterations_used") is not None:
                aux.efficiency = side_info["iterations_used"].float().mean()
            aux.rate_distortion = rd_loss

        return ModelOutput(
            logits=logits,
            hidden=h,
            aux=aux,
            ce_loss=ce,
            iterations_used=side_info.get("iterations_used"),
        )

    def _rope(self, T: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.cfg.model.attention
        return build_rope_cache(T, a.head_dim, a.rope_theta, device, dtype)
