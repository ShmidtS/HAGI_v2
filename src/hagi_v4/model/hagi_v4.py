"""HAGI V7 — 5G NR-style pipeline (simplified).

5G NR physical layer mapping:
  Embed + Mask     = Transport block + CRC (source coding + erasure)
  FreqBlock × N    = OFDM modulation (FFT/IFFT + equalization + QAM)
  Linear bottleneck = Rate matching (deterministic puncturing, H→C)
  Turbo loop × K   = LDPC iterative decoding
    FreqBlock      = Component A (local OFDM equalization)
    GP2D           = Component B (parity check)
    Extrinsic      = Soft-information exchange
  Linear up        = Rate dematching (C→H)
  LM head          = Demodulation → bits

Removed (no 5G analog):
  GDR, MSA, z_H/z_L state machine, water_filling,
  coherence head, deep supervision, VIB, perception/expression split
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.gp2d import GeometricProduct2D
from hagi_v4.model.multiscale_gp2d import MultiScaleGP2D
from hagi_v4.model.msa import MSAModule
from hagi_v4.model.norms import RMSNorm, build_rope_cache
from hagi_v4.model.outputs import AuxLosses, ModelOutput, compute_whiteness_loss
from hagi_v4.model.transformer_block import TransformerBlock
from hagi_v4.model.freq_layer import FreqBlock
from hagi_v4.model.kalman import KalmanFilter


class TurboLoop(nn.Module):
    """Simplified turbo decoding loop — 5G LDPC + channel memory + Kalman.

    Channel memory techniques:
      MSA read (iter>0) = Decision Feedback Equalizer (DFE) — past decisions cancel ISI
      MSA write          = HARQ buffer — store parity-checked states for combining
      Kalman filter      = Optimal channel estimation (blend prediction + measurement)
      GP2D               = LDPC parity check
      Convergence halt   = EXIT chart stopping criterion

    No z_H/z_L, no GDR, no deep supervision, no coherence, no water_filling.
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
        targets: torch.Tensor | None,
        lm_head_weight: torch.Tensor,
        mask: torch.Tensor | None,
        training: bool,
        step: int,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, T, C = z.shape

        total_parity = z.new_zeros(())
        total_msa_lb = z.new_zeros(())
        extrinsic_norms: list[float] = []
        converged_at = self.n_iters
        iterations_used = torch.full((B, T), self.n_iters, dtype=torch.long, device=z.device)
        gp2d_residual = torch.zeros_like(z)

        # Kalman state: diagonal covariance P per dimension
        p = torch.ones(C, device=z.device, dtype=z.dtype)

        self.msa.clear()

        for iteration in range(self.n_iters):
            h_prior = z

            # DFE: read channel memory from previous iterations (cancel ISI)
            if iteration > 0:
                msa_out, lb = self.msa.read(z, top_k=self.msa.cfg.top_k)
                z = z + msa_out
                total_msa_lb = total_msa_lb + lb

            # Component A: reasoning blocks (prediction = OFDM equalization)
            for blk in self.reasoning:
                z, _, _ = blk(z, cos, sin)

            z_pred = z

            # Kalman prediction step: uncertainty grows
            p_pred = self.kalman.predict(p)

            # Component B: GP2D parity check (measurement)
            z_meas, gp2d_residual = self.gp2d(z_pred)
            total_parity = total_parity + gp2d_residual.pow(2).mean().to(total_parity.dtype)

            # Kalman update: iteration-dependent R (5G iterative channel estimation)
            # R decreases with iteration: trust measurement more as decoding converges
            r_decay = 0.5**iteration
            z, p = self.kalman.update(z_pred, z_meas, p_pred, r_scale=r_decay)

            # Innovation norm for convergence check
            ext_norm = (z_meas - z_pred).float().norm(dim=-1).mean().item()
            extrinsic_norms.append(ext_norm)

            # HARQ Type II: store INNOVATION (incremental redundancy), not full state
            innovation = z - h_prior
            self.msa.write(innovation.detach() if not training else innovation)

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
            "moe_lb": z.new_zeros(()),
            "msa_lb": total_msa_lb / max(converged_at, 1),
            "gdr_router_loss": z.new_zeros(()),
            "gdr_gate_probs": None,
            "moe_router_probs": None,
            "gp2d_residual": gp2d_residual,
            "deep_supervision_loss": None,
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
        for blk in self.perception:
            h, _, _ = blk(h, cos, sin)

        # 3. Rate matching: raised-cosine FFT truncation H -> C
        h_f = torch.fft.rfft(h.float(), dim=-1)
        gate = torch.sigmoid(self.bottleneck_gate)
        h_f_c = h_f[:, :, : self._C // 2 + 1] * gate.float()
        z = torch.fft.irfft(h_f_c, n=self._C, dim=-1).to(h.dtype)
        z = self.bottleneck_norm(z)
        if mask is not None:
            z = torch.where(mask.unsqueeze(-1), self.core_mask_embed.expand(B, T, -1), z)

        # 4. LDPC iterative decoding (turbo loop)
        z, side_info = self.turbo(
            z,
            targets,
            self.lm_head.weight,
            mask,
            training=self.training,
            step=step,
            cos=cos,
            sin=sin,
        )

        # 5. Rate dematching: raised-cosine zero-padding C -> H
        z_f = torch.fft.rfft(z.float(), dim=-1)
        z_pad = torch.zeros(B, T, self._H // 2 + 1, dtype=z_f.dtype, device=z.device)
        c_bins = self._C // 2 + 1
        gate_up = torch.sigmoid(self.bottleneck_up_gate)
        z_pad[:, :, :c_bins] = z_f * gate_up[:c_bins].float()
        h = torch.fft.irfft(z_pad, n=self._H, dim=-1).to(z.dtype)

        # 6. Demodulation refinement (expression = 5G demapper)
        for blk in self.expression:
            h, _, _ = blk(h, cos, sin)

        # 7. Final demod -> bits
        h_normed = self.final_norm(h)
        mask_valid = mask is not None and mask.any().item() if mask is not None else False
        if targets is not None and mask_valid:
            h_masked = h_normed[mask]
            t_masked = targets[mask]
            logits_masked = F.linear(h_masked, self.lm_head.weight)
            ce = F.cross_entropy(logits_masked, t_masked)
            logits = None
        elif targets is not None:
            logits = F.linear(h_normed, self.lm_head.weight)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
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
                aux.extrinsic_info = h.new_tensor(sum(side_info["extrinsic_norms"]) / len(side_info["extrinsic_norms"]))
            if side_info.get("iterations_used") is not None:
                aux.efficiency = side_info["iterations_used"].float().mean()

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
