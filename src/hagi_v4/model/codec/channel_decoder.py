"""HAGI V23 Channel Decoder — LDPC belief-propagation with Kalman gate.

Owns: LDPCDecoder (pure BP with Mahalanobis validation gate), ChannelDecoder
wrapper that de-interleaves and runs BP + correction-alignment diagnostics.

V21 refactor: extracted verbatim from the monolithic HAGIv4 class. No
behavioural changes; only the ownership boundary moved.

V23: integrates KalmanFilter (Bayes-optimal blending), HARQBuffer (extrinsic-
only soft combining across BP iterations), and HRM dual-component state
transitions (z_H spatial/coarse + z_L per-token/fine). All modules are
optional — when None, V22 behavior is preserved exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.codec_contracts import (
    ChannelEncodeResult,
    CodecShapeConfig,
    DecodeResult,
    DecodeState,
    MSADecodeConfig,
    SourceEncodeResult,
)
from hagi_v4.model.exit_chart import EXITChartEstimator
from hagi_v4.model.hrm import HTransition, LTransition
from hagi_v4.model.kalman import KalmanFilter
from hagi_v4.model.msa import HARQBuffer
from hagi_v4.model.sparse_parity import SparseParityChecker


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

    V23: optional KalmanFilter (Bayes-optimal blend of prediction and
    measurement), HARQBuffer (extrinsic-only soft combining across BP
    iterations), and HRM dual-component state transitions (z_H spatial +
    z_L per-token). When all V23 modules are None, behavior is identical
    to V22. The Tanner-graph H is shared with the encoder (fixed sparse
    mask + frozen parity_base + learnable per-check edge_log_scale).
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
        # V23 new params:
        kalman_filter: KalmanFilter | None = None,
        harq_buffer: HARQBuffer | None = None,
        l_transition: LTransition | None = None,
        h_transition: HTransition | None = None,
        hrm_h_init: nn.Linear | None = None,
        hrm_l_init: nn.Linear | None = None,
        hrm_z_h_to_hidden: nn.Linear | None = None,
        hrm_z_l_to_hidden: nn.Linear | None = None,
        hrm_z_h_init: nn.Parameter | None = None,
        hrm_z_l_init: nn.Parameter | None = None,
        hrm_stride: int = 4,
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

        # V21: EXIT chart convergence estimator. None = use legacy 2%-halt.
        self.exit_chart: EXITChartEstimator | None = None

        # V21: LearnedUncertainty — per-position variance for Bayes-optimal update.
        # None = use fixed sigma_sq (legacy V19 behavior).
        self.uncertainty_estimator: nn.Module | None = None
        # V21: always-soft-gate (no hard gate at inference).
        self.always_soft_gate: bool = False

        # V23: KalmanFilter for Bayes-optimal blending of prediction and measurement.
        self.kalman_filter = kalman_filter

        # V23: HARQBuffer for extrinsic-only soft combining across BP iterations.
        self.harq_buffer = harq_buffer

        # V23: HRM dual-component state transitions (z_H spatial + z_L per-token).
        self.l_transition = l_transition
        self.h_transition = h_transition
        self.hrm_h_init = hrm_h_init
        self.hrm_l_init = hrm_l_init
        self.hrm_z_h_to_hidden = hrm_z_h_to_hidden
        self.hrm_z_l_to_hidden = hrm_z_l_to_hidden
        self.hrm_z_h_init = hrm_z_h_init
        self.hrm_z_l_init = hrm_z_l_init
        self.hrm_stride = hrm_stride
        self._hrm_enabled = (
            l_transition is not None
            and h_transition is not None
            and hrm_h_init is not None
            and hrm_l_init is not None
            and hrm_z_h_to_hidden is not None
            and hrm_z_l_to_hidden is not None
            and hrm_z_h_init is not None
            and hrm_z_l_init is not None
        )

    def set_exit_chart(self, estimator: EXITChartEstimator | None) -> None:
        """Enable or disable V21 EXIT chart convergence halting."""
        self.exit_chart = estimator

    def set_uncertainty_estimator(self, estimator: nn.Module | None) -> None:
        """Enable V21 learned per-position uncertainty."""
        self.uncertainty_estimator = estimator

    def set_always_soft_gate(self, enabled: bool) -> None:
        """V21: always use soft gate (no hard gate at inference)."""
        self.always_soft_gate = enabled

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
        cqi: torch.Tensor | None = None,
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
        prev_update: torch.Tensor | None = None  # V21: for EXIT chart

        # V23: Kalman filter covariance state — [B, T, C], init to moderate
        # uncertainty (0.5). None when kalman_filter is disabled (V22 path).
        if self.kalman_filter is not None:
            P = torch.full((B, T, self.hidden_size), 0.001, device=z_sys.device, dtype=z_sys.dtype)
        else:
            P = None

        # V23: HRM dual-component state. z_H is spatial/coarse (strided),
        # z_L is per-token/fine (full resolution). Disabled when any HRM
        # component is None or sequence too short for striding.
        hrm_active = False
        if self._hrm_enabled:
            S = self.hrm_stride
            if T >= S:
                T_c = T // S
                h_coarse = z_sys[:, : T_c * S].view(B, T_c, S, self.hidden_size).mean(dim=2)
                z_L = self.hrm_l_init(z_sys) + self.hrm_z_l_init
                z_H = self.hrm_h_init(h_coarse) + self.hrm_z_h_init
                hrm_active = True
            else:
                S = 0
                T_c = 0
                z_L = None
                z_H = None
        else:
            S = 0
            T_c = 0
            z_L = None
            z_H = None

        # V23: diagnostics accumulators (detached, for monitoring only).
        kalman_gain_sum = z_sys.new_zeros(())
        harq_stored_ext_norm_sum = z_sys.new_zeros(())
        hrm_h_bias_norm_sum = z_sys.new_zeros(())

        for iteration in range(iteration_limit):
            # V23: save state before any modification (for extrinsic delta).
            z_prior = z_pred

            # V23: HARQ read — combine stored extrinsic from prior iterations.
            # Iterations > 0 only (first iteration has no stored info).
            if self.harq_buffer is not None and iteration > 0:
                stored_ext = self.harq_buffer.read(z_pred, top_k=6)
                if self.uncertainty_estimator is not None:
                    uncertainty = self.uncertainty_estimator(z_pred).mean(dim=-1)  # [B, T]
                else:
                    uncertainty = torch.zeros(B, T, device=z_pred.device, dtype=z_pred.dtype)
                z_pred = self.harq_buffer.combine(z_pred, stored_ext, uncertainty)
                harq_stored_ext_norm_sum = harq_stored_ext_norm_sum + stored_ext.float().norm().detach()

            # V23: Kalman predict — P_pred = P + Q (process noise inflation).
            if self.kalman_filter is not None:
                P_pred = self.kalman_filter.predict(P)

            residual, _parity_computed = self.parity_checker(z_pred, parity_received)
            total_parity = total_parity + residual.pow(2).mean().to(total_parity.dtype)
            last_residual = residual

            # --- V19/V21 validation gate ------------------------------------
            # V21: learned per-position sigma when uncertainty_estimator is set.
            if self.uncertainty_estimator is not None:
                sigma2_pos = self.uncertainty_estimator(z_pred)  # [B, T, C]
                # Use mean variance across C for the Mahalanobis gate
                sigma2_scalar = sigma2_pos.mean(dim=-1)  # [B, T]
                d2 = residual.pow(2).sum(dim=-1) / sigma2_scalar.clamp_min(1e-8)
            else:
                d2 = residual.pow(2).sum(dim=-1) / max(sigma_sq, 1e-8)  # [B, T]
            global_active_frac = (d2 > chi2_crit).float().mean().item()

            if training or self.always_soft_gate:
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
            # V21: CQI modulates gate magnitude (not validation gate).
            if cqi is not None:
                # High CQI = good channel = trust correction more.
                # cqi in [0,1], scale gate_mag by (0.5 + cqi) to keep it non-zero.
                cqi_factor = (0.5 + 0.5 * cqi).unsqueeze(-1).to(gate_mag.dtype)
                gate_mag = gate_mag * cqi_factor

            innovation = gate_mag * gate_valid.unsqueeze(-1) * correction
            if mask is not None:
                innovation = innovation * mask.unsqueeze(-1).to(innovation.dtype)

            # V23: Kalman update — Bayes-optimal blend of prediction and
            # measurement. K = P_pred / (P_pred + R) bounds the measurement
            # contribution in [0, 1]. When kalman_filter is None, falls back
            # to the V22 direct-add path.
            if self.kalman_filter is not None:
                with torch.no_grad():
                    r = self.kalman_filter._r(P_pred.dtype)
                    if P_pred.dim() == 3:
                        r = r.unsqueeze(0).unsqueeze(0)
                    k_diag = P_pred.float() / (P_pred.float() + r.float())
                    kalman_gain_sum = kalman_gain_sum + k_diag.mean()
                z_pred, P = self.kalman_filter.update(z_pred, innovation, P_pred)
            else:
                z_pred = z_pred + innovation

            # V23: HRM dual-component state transition. z_L is updated from
            # the current z_pred (per-token), then coarsened to update z_H
            # (spatial). The upsampled z_H + z_L produce a bias added to z_pred.
            if hrm_active:
                z_L = self.l_transition(z_L, z_pred)
                l_dim = z_L.shape[-1]
                z_L_coarse = z_L[:, : T_c * S].view(B, T_c, S, l_dim).mean(dim=2)
                z_H = self.h_transition(z_H, z_L_coarse)
                z_H_up = z_H.repeat_interleave(S, dim=1)
                if z_H_up.shape[1] < T:
                    pad = z_H[:, -1:].repeat(1, T - z_H_up.shape[1], 1)
                    z_H_up = torch.cat([z_H_up, pad], dim=1)
                z_H_up = z_H_up[:, :T]
                h_bias = self.hrm_z_h_to_hidden(z_H_up) + self.hrm_z_l_to_hidden(z_L)
                z_pred = z_pred + h_bias
                hrm_h_bias_norm_sum = hrm_h_bias_norm_sum + h_bias.float().norm().detach()

            # V23: HARQ write — store the extrinsic delta (what changed this
            # iteration), NOT the full state. This is the LDPC BP principle:
            # each iteration adds NEW information, not re-broadcasted beliefs.
            if self.harq_buffer is not None:
                ext_delta = z_pred - z_prior
                self.harq_buffer.write(ext_delta)

            # --- Early halt: convergence detection ---------------------------
            if self.exit_chart is not None and iteration >= 1 and prev_update is not None:
                # V21: EXIT chart norm-ratio convergence (tensor, no GPU sync)
                should_halt, _mi = self.exit_chart.should_halt(prev_update, innovation, iteration)
                if should_halt.item() and not training:
                    iteration_limit = iteration + 1
                    break
            elif not training and iteration >= 1 and global_active_frac < 0.02 and prev_global_active_frac < 0.02:
                # Legacy 2%-active halt (V19)
                iteration_limit = iteration + 1
                break
            prev_global_active_frac = global_active_frac
            prev_update = innovation

        side_info = {
            "parity_strength": total_parity / max(iteration_limit, 1),
            "iterations_used": iters_used_per_pos.clamp(min=1 if not training else 0, max=iteration_limit),
            "parity_residual": last_residual,
            # V19 gating diagnostics
            "ever_active_frac": ever_active.float().mean(),
            "mean_iters_used": iters_used_per_pos.float().mean() / max(iteration_limit, 1),
            # V23 module diagnostics
            "kalman_gain_mean": kalman_gain_sum / max(iteration_limit, 1),
            "harq_stored_ext_norm": harq_stored_ext_norm_sum / max(iteration_limit, 1),
            "hrm_h_bias_norm": hrm_h_bias_norm_sum / max(iteration_limit, 1),
        }
        state.iteration = iteration_limit
        return DecodeResult(latent=z_pred, state=state, side_info=side_info)


class ChannelDecoder(nn.Module):
    """Channel decoder: de-interleave -> LDPC BP -> correction diagnostics.

    Wraps an LDPCDecoder instance and owns the interleaver reference needed
    to invert the channel encoder's permutation before running BP. The
    shared parity params (edge_log_scale, sparse_mask, parity_base) come
    from the ChannelEncoder's parity_encoder — passed in to preserve the
    shared Tanner-graph structure.

    Pipeline (verbatim from HAGIv4 V18):
      interleaved codeword
        -> de-interleave
        -> split [z_sys, parity]
        -> LDPCDecoder (BP with Kalman validation gate)
        -> correction_alignment diagnostic
    """

    def __init__(
        self,
        cfg: HAGIv4Config,
        codec_shape: CodecShapeConfig,
        interleaver,
        shared_parity_weights: nn.Parameter | None,
        shared_sparse_mask: torch.Tensor | None,
        shared_edge_log_scale: nn.Parameter | None,
        shared_parity_base: torch.Tensor | None,
    ) -> None:
        super().__init__()
        m = cfg.model
        self._C = m.core_hidden_size
        self.interleaver = interleaver

        # V23: Kalman filter — Bayes-optimal blending of prediction and measurement.
        self._kalman: KalmanFilter | None = None
        kalman_cfg = getattr(cfg.model, "kalman", None)
        if kalman_cfg is not None:
            self._kalman = KalmanFilter(dim=m.core_hidden_size)

        # V23: HARQ buffer — extrinsic-only soft combining across BP iterations.
        self._harq: HARQBuffer | None = None
        msa_cfg = getattr(cfg.model, "msa", None)
        if msa_cfg is not None and getattr(msa_cfg, "max_slots", 0) > 0:
            msa_decode_cfg = MSADecodeConfig(
                max_slots=msa_cfg.max_slots,
                slot_chunk_size=msa_cfg.slot_chunk_size,
                top_k=msa_cfg.top_k,
                routing_key_dim=msa_cfg.routing_key_dim,
                n_kv_heads=msa_cfg.n_kv_heads,
                head_dim=msa_cfg.head_dim,
                mla_compress_dim=msa_cfg.mla_compress_dim,
                mla_up_dim=msa_cfg.mla_up_dim,
            )
            self._harq = HARQBuffer(msa_decode_cfg, hidden_size=m.core_hidden_size)

        # V23: HRM dual-component — z_H spatial/coarse + z_L per-token/fine.
        # All components are constructed here and passed to LDPCDecoder; the
        # decoder owns them via its own attribute registration.
        self._hrm_enabled = False
        _l_trans: LTransition | None = None
        _h_trans: HTransition | None = None
        _h_init: nn.Linear | None = None
        _l_init: nn.Linear | None = None
        _z_h_to_hidden: nn.Linear | None = None
        _z_l_to_hidden: nn.Linear | None = None
        _z_h_init: nn.Parameter | None = None
        _z_l_init: nn.Parameter | None = None
        _hrm_stride = 4
        hrm_cfg = getattr(cfg.model, "hrm", None)
        if hrm_cfg is not None:
            _l_trans = LTransition(hrm_cfg.l_state_dim, m.core_hidden_size)
            _h_trans = HTransition(hrm_cfg.h_state_dim, hrm_cfg.l_state_dim)
            _h_init = nn.Linear(m.core_hidden_size, hrm_cfg.h_state_dim, bias=False)
            _l_init = nn.Linear(m.core_hidden_size, hrm_cfg.l_state_dim, bias=False)
            _z_h_to_hidden = nn.Linear(hrm_cfg.h_state_dim, m.core_hidden_size, bias=False)
            _z_l_to_hidden = nn.Linear(hrm_cfg.l_state_dim, m.core_hidden_size, bias=False)
            nn.init.zeros_(_z_h_to_hidden.weight)
            nn.init.zeros_(_z_l_to_hidden.weight)
            _z_h_init = nn.Parameter(torch.zeros(hrm_cfg.h_state_dim))
            _z_l_init = nn.Parameter(torch.zeros(hrm_cfg.l_state_dim))
            _hrm_stride = hrm_cfg.h_stride
            self._hrm_enabled = True

        self.decoder = LDPCDecoder(
            hidden_size=m.core_hidden_size,
            n_parity_checks=codec_shape.n_parity_checks,
            edges_per_check=codec_shape.edges_per_check,
            norm_eps=m.norm_eps,
            shared_parity_weights=shared_parity_weights,
            shared_sparse_mask=shared_sparse_mask,
            shared_edge_log_scale=shared_edge_log_scale,
            shared_parity_base=shared_parity_base,
            kalman_filter=self._kalman,
            harq_buffer=self._harq,
            l_transition=_l_trans,
            h_transition=_h_trans,
            hrm_h_init=_h_init,
            hrm_l_init=_l_init,
            hrm_z_h_to_hidden=_z_h_to_hidden,
            hrm_z_l_to_hidden=_z_l_to_hidden,
            hrm_z_h_init=_z_h_init,
            hrm_z_l_init=_z_l_init,
            hrm_stride=_hrm_stride,
        )

        # V21: EXIT chart estimator — None = legacy 2%-halt
        self._exit_chart: EXITChartEstimator | None = None
        exit_cfg = getattr(cfg.model, "exit_chart", None)
        if exit_cfg is not None:
            self._exit_chart = EXITChartEstimator(
                threshold=exit_cfg.threshold,
                min_iterations=exit_cfg.min_iterations,
            )
            self.decoder.set_exit_chart(self._exit_chart)

        # V21: LearnedUncertainty — per-position variance estimator
        self._uncertainty_estimator: nn.Module | None = None
        unc_cfg = getattr(cfg.model, "uncertainty", None)
        if unc_cfg is not None:
            from hagi_v4.model.uncertainty import LearnedUncertainty

            self._uncertainty_estimator = LearnedUncertainty(
                hidden_size=m.core_hidden_size,
            )
            self.decoder.set_uncertainty_estimator(self._uncertainty_estimator)

        # V21: always-soft-gate (no hard gate at inference)
        # Enabled when uncertainty estimator is active (they work together)
        if self._uncertainty_estimator is not None:
            self.decoder.set_always_soft_gate(True)

        # V23: FreqCoding2D as a frequency-domain MIMO equalizer on the
        # systematic portion of the received codeword. Applied AFTER
        # de-interleaving but BEFORE the LDPC BP loop — pre-conditions the
        # received signal so BP starts from a channel-compensated estimate.
        # V15 lesson: FreqCoding2D is a weak language computer in the Source
        # path; here it serves its information-theoretically correct role as
        # a MIMO channel equalizer (complex weight = frequency-selective
        # channel inversion). Config-gated: when freq_coding.enabled is
        # False, behavior is identical to V22.
        self._freq_equalizer: nn.Module | None = None
        freq_cfg = getattr(m, "freq_coding", None)
        if freq_cfg is not None and getattr(freq_cfg, "enabled", False):
            from hagi_v4.model.freq_layer import FreqCoding2D

            # head_dim forced to 64 so C divides cleanly (matches the Source
            # path convention in source_encoder.py). When n_heads*head_dim !=
            # C, FreqCoding2D falls back to FactoredLinear projections
            # (CDMA spreading analog) — no divisibility hard-failure.
            head_dim_eq = 64
            n_heads_eq = max(1, m.core_hidden_size // head_dim_eq)
            self._freq_equalizer = FreqCoding2D(
                hidden_size=m.core_hidden_size,
                n_heads=n_heads_eq,
                head_dim=head_dim_eq,
                n_modes_t=freq_cfg.n_modes_t,
                n_modes_h=freq_cfg.n_modes_h,
                T_max=m.attention.max_seq_len,
                rank=freq_cfg.complex_rank,
                norm_eps=m.norm_eps,
                use_derivative=freq_cfg.use_derivative,
                share_branch_weights=freq_cfg.share_branch_weights,
            )

    def set_awgn_sigma(self, sigma: float | None) -> None:
        self.decoder.set_awgn_sigma(sigma)

    def forward(
        self,
        ch_encoded: ChannelEncodeResult,
        encoded: SourceEncodeResult,
        state: DecodeState,
        training: bool,
        bp_iterations: int | None = None,
        refinement_iterations: int | None = None,
    ) -> DecodeResult:
        # V23: clear HARQ buffer between forward passes (fresh decode context).
        if self._harq is not None:
            self._harq.clear()
        codeword = self.interleaver.deinterleave(ch_encoded.codeword)
        C = self._C
        if self._freq_equalizer is not None:
            z_sys_pre = codeword[..., :C]
            z_sys_eq = self._freq_equalizer(z_sys_pre)
            codeword = torch.cat([z_sys_eq, codeword[..., C:]], dim=-1)
        z_sys = codeword[..., :C]
        parity = codeword[..., C:]
        decode_mask = ch_encoded.erasure_mask
        if encoded.mask is not None:
            decode_mask = encoded.mask if decode_mask is None else (decode_mask | encoded.mask)
        n_iters = int(bp_iterations) if bp_iterations is not None else 3
        result = self.decoder(
            z_sys=z_sys,
            parity_received=parity,
            training=training,
            state=state,
            mask=decode_mask,
            refinement_iterations=refinement_iterations,
            n_iters=n_iters,
            cqi=encoded.cqi,  # V21: pass CQI to decoder
        )
        result.side_info["freq_equalizer_active"] = (
            z_sys.new_ones(()) if self._freq_equalizer is not None else z_sys.new_zeros(())
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
