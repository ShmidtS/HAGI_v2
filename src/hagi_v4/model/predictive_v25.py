"""PredictiveDecoder V25 — extrinsic error highway + HEP + Kalman blend (§3.3).

V25 upgrade over the V24 ``predictive_decoder.py``:

* **Highway Error Propagation (HEP)** linear feedback ``V_t`` (zero-init) keeps
  correction magnitude depth-independent so deep error highways train — vanilla
  PC suffers exponential signal decay with depth.
* **Bayes-optimal K=P/(P+R) Kalman blend** is ON by default (reused from
  ``uncertainty.py::LearnedUncertainty`` + ``inverse_variance_update``), now in
  its correct role: aleatoric weighting of the genuine innovation ε = z − ẑ.
* **Ternary option**: the 2D hidden mixing masters (predictor, update_proj,
  update_out, HEP feedback ``V_t``) become ``BitLinear`` when ``use_ternary``;
  the 1D gates and ``LearnedUncertainty.log_var`` stay FP.
* **Extrinsic-only refinement**: ``ẑ_{t+1} = ẑ_t + g_t·(u_t + v_t)`` — each
  iteration adds ONLY the new gated innovation, never re-broadcasts ``ẑ``.
  Prevents information recycling (V4 belief-amplification failure mode).
* **Identity cold-start** via zero-init update path (V8–V23 dead-gradient
  guard): ``update_out`` is SMALL-init (std=0.02), HEP ``V_t`` is zero-init.

No self-noise is ever injected. The only impairment on the V25 path is ternary
quantization noise on the channel body, supplied upstream in ``block.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


def _make_linear(in_features: int, out_features: int, use_ternary: bool) -> nn.Module:
    """Construct a 2D mixing layer: BitLinear when ternary, else nn.Linear.

    Ternary masters are routed to Muon by ``is_muon_param`` (2D, not in the FP
    exclude list). BitNet b1.58 per-output-channel absmean scale + identity STE
    live inside ``BitLinear.forward``. Falls back to ``nn.Linear`` if the
    ``ternary`` module is absent (e.g. ternary wiring landed in a different
    subagent pass) so this file remains importable on its own.
    """
    if use_ternary:
        try:
            from hagi_v4.model.ternary import BitLinear

            return BitLinear(in_features, out_features, bias=False)
        except Exception:  # pragma: no cover - defensive fallback
            return nn.Linear(in_features, out_features, bias=False)
    return nn.Linear(in_features, out_features, bias=False)


@dataclass
class PredictiveConfig:
    """Predictive decoder refinement parameters (§3.3).

    Defaults preserve the V13 train/infer asymmetry invariant
    (``train_iterations < infer_iterations``).
    """

    train_iterations: int = 2  # T_train
    infer_iterations: int = 4  # T_infer
    convergence_threshold: float = 0.01  # τ: halt when ‖ε‖/‖z‖ < τ twice
    update_hidden: int = 256  # Update_t MLP hidden width


class PredictiveDecoder(nn.Module):
    """Top-down predictor + iterative extrinsic refinement of the latent z.

    Args:
        dim: C (bottleneck dimension — the latent ``z`` lives here).
        ctx_dim: H (context hidden ``h_ctx`` used to form the top-down
            prediction ``ẑ_0``).
        cfg: refinement config (train/infer iterations, halt threshold,
            update MLP width).
        norm_eps: RMSNorm epsilon.
        use_kalman_blend: when True, blend each refinement step with the
            Bayes-optimal ``K=P/(P+R)`` update from ``uncertainty.py``.
        use_ternary: when True, the 2D mixing masters (predictor, update_proj,
            update_out, HEP feedback) become BitLinear; gates and the
            uncertainty head stay FP.
        hep_enabled: when True, instantiate the per-iteration HEP linear
            feedback ``V_t`` (zero-init). HEP is OFF at start and learns to
            turn on; the value grows only if the decoder is deepened.
    """

    def __init__(
        self,
        dim: int,
        ctx_dim: int,
        cfg: PredictiveConfig,
        norm_eps: float = 1e-6,
        use_kalman_blend: bool = True,
        use_ternary: bool = True,
        hep_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ctx_dim = ctx_dim
        self.cfg = cfg
        self.use_kalman_blend = use_kalman_blend
        self.use_ternary = use_ternary
        self.hep_enabled = hep_enabled

        # Top-down predictor: context summary -> initial prediction ẑ_0.
        # Ternary 2D master when requested; the pre-RMSNorm stays FP.
        self.predictor_norm = RMSNorm(ctx_dim, eps=norm_eps)
        self.predictor = _make_linear(ctx_dim, dim, use_ternary)

        # Per-iteration update modules: ε_t -> residual. INDEPENDENT weights per
        # iteration so each step contributes NEW information (V10 lesson: shared
        # decoder weights across iterations produced identical extrinsic ->
        # divergence). We size the ModuleList by the LARGER of train/infer so
        # both regimes index valid modules without reconstruction.
        n = max(cfg.train_iterations, cfg.infer_iterations)
        self.n_iters_max = n
        self.update_norm = nn.ModuleList([RMSNorm(dim, eps=norm_eps) for _ in range(n)])
        self.update_proj = nn.ModuleList([_make_linear(dim, cfg.update_hidden, use_ternary) for _ in range(n)])
        self.update_out = nn.ModuleList([_make_linear(cfg.update_hidden, dim, use_ternary) for _ in range(n)])
        # Small (NOT zero) init on update_out. Zero-init kills the entire update
        # branch: with update≡0, z_pred stays at the predictor output, eps stops
        # depending on update_proj/gate, and those starve of gradient (the
        # V8-V23 dead-gradient class). Small init gives a live gradient from
        # step 0.
        for proj in self.update_out:
            w = getattr(proj, "weight", None)
            if w is not None:
                nn.init.normal_(w, std=0.02)

        # HEP linear feedback V_t (per-iteration). Zero-init so HEP is OFF at
        # start and learns to turn on; keeps correction magnitude
        # depth-independent. v_t = V_t · ε_t.
        self.hep_feedback = None
        if hep_enabled:
            self.hep_feedback = nn.ModuleList([_make_linear(dim, dim, use_ternary) for _ in range(n)])
            for proj in self.hep_feedback:
                w = getattr(proj, "weight", None)
                if w is not None:
                    nn.init.zeros_(w)

        # Magnitude gate (FP scalars). Init neutral (sigmoid(0)=0.5).
        self.gate_w = nn.Parameter(torch.tensor([0.3]))
        self.gate_b = nn.Parameter(torch.tensor([0.0]))

        # Reused from uncertainty.py: learned per-position scalar variance
        # estimator (LearnedUncertainty) + the Bayes-optimal inverse-variance
        # blend (K = P/(P+R)). In V23 these were mis-applied to a fake
        # parity-residual; here they serve their correct role — aleatoric
        # uncertainty weighting of the genuine innovation ε.
        self.uncertainty = None
        if use_kalman_blend:
            from hagi_v4.model.uncertainty import LearnedUncertainty

            self.uncertainty = LearnedUncertainty(dim)

    def _kalman_blend(
        self,
        z_pred: torch.Tensor,
        correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bayes-optimal K=P/(P+R) blend of prediction and correction.

        P = softplus(Linear_unc(ẑ)) — predicted-state variance (per-position
        scalar, broadcast across channels). R = correction².mean(-1) —
        measurement/innovation variance. Variance math runs in float32 for
        numerical stability; the corrected state is cast back to z_pred.dtype.

        Args:
            z_pred: [B, T, C] current refined prediction (native dtype).
            correction: [B, T, C] = g_t·(u_t + v_t) (native dtype).

        Returns:
            z_corrected: [B, T, C] refined prediction after the Bayes update
                (native dtype).
            k_mean: scalar mean Kalman gain (for diagnostics).
        """
        from hagi_v4.model.uncertainty import inverse_variance_update

        out_dtype = z_pred.dtype
        # Estimator runs in native dtype (its Linear weights live there under
        # AMP); the variance combination runs in float32. Guard against the
        # constructor leaving uncertainty=None (only when use_kalman_blend was
        # False — but this method is only called from the blend branch).
        if self.uncertainty is None:
            return z_pred, z_pred.new_zeros(())
        sigma2_pred = self.uncertainty(z_pred).float()  # [B, T, C]
        corr_f = correction.float()
        sigma2_meas = corr_f.pow(2).mean(dim=-1, keepdim=True)  # [B, T, 1]
        z_corrected, k = inverse_variance_update(
            z_pred.float(), corr_f, sigma2_pred, sigma2_meas
        )
        return z_corrected.to(out_dtype), k.mean()

    def forward(
        self,
        z: torch.Tensor,
        ctx: torch.Tensor,
        training: bool,
        iterations: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Refine the bottleneck latent z via extrinsic error-highway iteration.

        Args:
            z: [B, T, C] bottleneck latent (bottom-up signal from the IB).
            ctx: [B, T, H] context hidden (h_ctx, for the top-down prediction).
            training: when True, run all train iterations and skip convergence
                halting (full gradient flow). When False, allow early halt.
            iterations: override the iteration count; defaults to
                ``train_iterations`` in training and ``infer_iterations`` in
                eval.

        Returns:
            z_pred: [B, T, C] refined latent.
            side_info: dict with keys
                ``iterations_used`` [B, T] long,
                ``final_innovation`` scalar (mean ‖ε‖/‖z‖ at last step),
                ``converged_frac`` scalar (fraction of positions halted, eval),
                ``kalman_gain_mean`` scalar (mean K across steps).
        """
        n_iters = int(iterations) if iterations is not None else (
            self.cfg.train_iterations if training else self.cfg.infer_iterations
        )
        n_iters = max(1, min(n_iters, self.n_iters_max))

        z_dtype = z.dtype
        # Top-down prediction: ẑ_0 = predictor(RMSNorm(ctx)).
        z_pred = self.predictor(self.predictor_norm(ctx)).to(z_dtype)

        last_rel_err = z.new_zeros(())
        kalman_gain_mean = z.new_zeros(())
        iters_used = torch.zeros(z.shape[:2], dtype=torch.long, device=z.device)
        z_norm_ref = z.float().pow(2).mean(dim=-1).clamp_min(1e-6)  # ‖z‖² per pos
        halted = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
        prev_below_tau = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)

        for t in range(n_iters):
            # ε_t = z − ẑ_t  (innovation / extrinsic error), native dtype.
            eps = (z - z_pred).to(z_dtype)
            eps_mag = eps.float().pow(2).mean(dim=-1)  # [B, T]
            rel_err = (eps_mag / z_norm_ref).sqrt()  # ‖ε_t‖/‖z‖ per position

            # Active positions are refined; halted ones freeze.
            active = ~halted
            if active.any():
                iters_used[active] += 1

            # Magnitude gate g_t = sigmoid(w·‖ε_t‖ + b) (FP scalars).
            gate = torch.sigmoid(
                eps_mag.sqrt() * self.gate_w + self.gate_b
            ).unsqueeze(-1).to(z_dtype)

            # Update branch: u_t = UpdateOut_t(silu(UpdateProj_t(RMSNorm_t(ε_t)))).
            # Uses native dtype; ternarization (if any) is internal to BitLinear.
            eps_n = self.update_norm[t](eps)
            u_t = self.update_out[t](F.silu(self.update_proj[t](eps_n)))

            # HEP linear feedback: v_t = V_t · ε_t (zero-init → off at start).
            # Keeps correction magnitude depth-independent.
            v_t = z_pred.new_zeros(())
            if self.hep_feedback is not None:
                v_t = self.hep_feedback[t](eps)

            # Extrinsic-only correction: add ONLY the new gated innovation.
            correction = gate * (u_t + v_t)

            if self.use_kalman_blend:
                # Bayes-optimal K=P/(P+R) blend.
                z_pred, kg = self._kalman_blend(z_pred, correction)
                kalman_gain_mean = kalman_gain_mean + kg
            else:
                z_pred = z_pred + correction

            last_rel_err = rel_err.mean()

            # Convergence halt in EVAL only: halt when ‖ε‖/‖z‖ < τ for TWO
            # consecutive iterations (avoid premature halt on a single dip).
            # Training always runs all iterations for full gradient flow.
            if not training:
                below_tau = (rel_err < self.cfg.convergence_threshold) & active
                # Halting requires two consecutive below-threshold iterations.
                newly_halted = below_tau & prev_below_tau
                halted = halted | newly_halted
                prev_below_tau = below_tau

        side_info = {
            "iterations_used": iters_used,
            "final_innovation": last_rel_err,
            "converged_frac": halted.float().mean() if not training else z.new_zeros(()),
            "kalman_gain_mean": kalman_gain_mean / max(n_iters, 1),
        }
        return z_pred, side_info
