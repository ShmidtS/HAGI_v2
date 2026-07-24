"""PredictiveDecoder — extrinsic error highway + HEP + Kalman blend.

OFF the main LM path by default (inserting it deadlocked from-scratch training).
Activated only for ablation/research (body.predictive.enabled=True).

  * **Highway Error Propagation (HEP)** linear feedback ``V_t`` (zero-init)
    keeps correction magnitude depth-independent so deep error highways train.
  * **Bayes-optimal K=P/(P+R) Kalman blend** (reused from ``uncertainty.py``)
    in its correct role: aleatoric weighting of the genuine innovation
    ε = z − z_hat.
  * **Ternary option**: the 2D hidden mixing masters become ``BitLinear`` when
    ``use_ternary``; the 1D gates and ``LearnedUncertainty.log_var`` stay FP.
  * **Extrinsic-only refinement**: ``z_hat_{t+1} = z_hat_t + g_t·(u_t + v_t)``
    — each iteration adds ONLY the new gated innovation, never re-broadcasts
    ``z_hat`` (prevents information recycling).
  * **Identity cold-start** via small-init update path: ``update_out`` is
    small-init (std=0.02), HEP ``V_t`` is zero-init.

No self-noise is ever injected. The only impairment on the path is ternary
quantization noise on the channel body, supplied upstream in ``block.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.model.norms import RMSNorm


def _make_linear(in_features: int, out_features: int, use_ternary: bool) -> nn.Module:
    """2D mixing layer: BitLinear when ternary, else nn.Linear."""
    if use_ternary:
        from hagi_v4.model.ternary import BitLinear

        return BitLinear(in_features, out_features, bias=False)
    return nn.Linear(in_features, out_features, bias=False)


@dataclass
class PredictiveConfig:
    """Predictive decoder refinement parameters.

    Defaults preserve the train/infer asymmetry invariant
    (``train_iterations < infer_iterations``).
    """

    train_iterations: int = 2
    infer_iterations: int = 4
    convergence_threshold: float = 0.01
    update_hidden: int = 256


class PredictiveDecoder(nn.Module):
    """Top-down predictor + iterative extrinsic refinement of the latent z.

    Args:
        dim: C (bottleneck dimension — the latent ``z`` lives here).
        ctx_dim: H (context hidden ``h_ctx`` used to form the top-down
            prediction ``z_hat_0``).
        cfg: refinement config (train/infer iterations, halt threshold,
            update MLP width).
        norm_eps: RMSNorm epsilon.
        use_kalman_blend: when True, blend each refinement step with the
            Bayes-optimal ``K=P/(P+R)`` update from ``uncertainty.py``.
        use_ternary: when True, the 2D mixing masters become BitLinear;
            gates and the uncertainty head stay FP.
        hep_enabled: when True, instantiate the per-iteration HEP linear
            feedback ``V_t`` (zero-init).
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

        self.predictor_norm = RMSNorm(ctx_dim, eps=norm_eps)
        self.predictor = _make_linear(ctx_dim, dim, use_ternary)

        n = max(cfg.train_iterations, cfg.infer_iterations)
        self.n_iters_max = n
        self.update_norm = nn.ModuleList([RMSNorm(dim, eps=norm_eps) for _ in range(n)])
        self.update_proj = nn.ModuleList([_make_linear(dim, cfg.update_hidden, use_ternary) for _ in range(n)])
        self.update_out = nn.ModuleList([_make_linear(cfg.update_hidden, dim, use_ternary) for _ in range(n)])
        for proj in self.update_out:
            w = getattr(proj, "weight", None)
            if w is not None:
                nn.init.normal_(w, std=0.02)

        self.hep_feedback = None
        if hep_enabled:
            self.hep_feedback = nn.ModuleList([_make_linear(dim, dim, use_ternary) for _ in range(n)])
            for proj in self.hep_feedback:
                w = getattr(proj, "weight", None)
                if w is not None:
                    nn.init.zeros_(w)

        self.gate_w = nn.Parameter(torch.tensor([0.3]))
        self.gate_b = nn.Parameter(torch.tensor([0.0]))

        self.uncertainty = None
        if use_kalman_blend:
            from hagi_v4.model.uncertainty import LearnedUncertainty

            self.uncertainty = LearnedUncertainty(dim)

    def _kalman_blend(
        self,
        z_pred: torch.Tensor,
        correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bayes-optimal K=P/(P+R) blend of prediction and correction."""
        from hagi_v4.model.uncertainty import inverse_variance_update

        out_dtype = z_pred.dtype
        if self.uncertainty is None:
            return z_pred, z_pred.new_zeros(())
        sigma2_pred = self.uncertainty(z_pred).float()
        corr_f = correction.float()
        sigma2_meas = corr_f.pow(2).mean(dim=-1, keepdim=True)
        z_corrected, k = inverse_variance_update(z_pred.float(), corr_f, sigma2_pred, sigma2_meas)
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
            z: [B, T, C] bottleneck latent.
            ctx: [B, T, H] context hidden (h_ctx, for the top-down prediction).
            training: when True, run all train iterations and skip convergence
                halting (full gradient flow). When False, allow early halt.
            iterations: override the iteration count.

        Returns:
            z_pred: [B, T, C] refined latent.
            side_info: dict with 'iterations_used', 'final_innovation',
                'converged_frac', 'kalman_gain_mean'.
        """
        n_iters = int(iterations) if iterations is not None else (
            self.cfg.train_iterations if training else self.cfg.infer_iterations
        )
        n_iters = max(1, min(n_iters, self.n_iters_max))

        z_dtype = z.dtype
        z_pred = self.predictor(self.predictor_norm(ctx)).to(z_dtype)

        last_rel_err = z.new_zeros(())
        kalman_gain_mean = z.new_zeros(())
        iters_used = torch.zeros(z.shape[:2], dtype=torch.long, device=z.device)
        z_norm_ref = z.float().pow(2).mean(dim=-1).clamp_min(1e-6)
        halted = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
        prev_below_tau = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)

        for t in range(n_iters):
            eps = (z - z_pred).to(z_dtype)
            eps_mag = eps.float().pow(2).mean(dim=-1)
            rel_err = (eps_mag / z_norm_ref).sqrt()

            active = ~halted
            if active.any():
                iters_used[active] += 1

            gate = torch.sigmoid(eps_mag.sqrt() * self.gate_w + self.gate_b).unsqueeze(-1).to(z_dtype)

            eps_n = self.update_norm[t](eps)
            u_t = self.update_out[t](F.silu(self.update_proj[t](eps_n)))

            v_t = z_pred.new_zeros(())
            if self.hep_feedback is not None:
                v_t = self.hep_feedback[t](eps)

            correction = gate * (u_t + v_t)

            if self.use_kalman_blend:
                z_pred, kg = self._kalman_blend(z_pred, correction)
                kalman_gain_mean = kalman_gain_mean + kg
            else:
                z_pred = z_pred + correction

            last_rel_err = rel_err.mean()

            if not training:
                below_tau = (rel_err < self.cfg.convergence_threshold) & active
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
