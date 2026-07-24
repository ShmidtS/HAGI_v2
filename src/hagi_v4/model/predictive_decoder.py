"""PredictiveDecoder — extrinsic error-highway refinement (§3.2).

Self-contained additive module. Replaces an LDPC-BP-style loop with a
predictive-coding refinement that propagates genuine innovation ε = z − ẑ
(a residual against the bottleneck signal). No self-noise is ever injected.
Pattern: error highway (arXiv 2606.22744).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hagi_v4.model.norms import RMSNorm


@dataclass
class PredictiveConfig:
    """Predictive decoder refinement parameters (§3.2)."""

    train_iterations: int = 2  # T_train  (kept < infer for V13 asymmetry win)
    infer_iterations: int = 4  # T_infer
    convergence_threshold: float = 0.01  # τ: halt when ‖ε‖/‖z‖ < τ twice
    update_hidden: int = 256  # Update_t MLP hidden width


class PredictiveDecoder(nn.Module):
    """Top-down predictor + iterative extrinsic refinement of the latent z.

    Args:
        dim: C (bottleneck dimension).
        ctx_dim: H (context hidden used to form the top-down prediction).
        cfg: refinement config (train/infer iterations, halt threshold).
    """

    def __init__(
        self,
        dim: int,
        ctx_dim: int,
        cfg: PredictiveConfig,
        norm_eps: float = 1e-6,
        use_kalman_blend: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cfg = cfg

        # Top-down predictor: context summary -> initial prediction ẑ_0.
        self.predictor = nn.Sequential(
            RMSNorm(ctx_dim, eps=norm_eps),
            nn.Linear(ctx_dim, dim, bias=False),
        )

        # Per-iteration update modules: ε_t -> residual. Independent weights so
        # each iteration contributes NEW information (V10 lesson: shared decoder
        # weights across iterations produced identical extrinsic -> divergence).
        n = cfg.infer_iterations
        self.update_norm = nn.ModuleList([RMSNorm(dim, eps=norm_eps) for _ in range(n)])
        self.update_proj = nn.ModuleList([nn.Linear(dim, cfg.update_hidden, bias=False) for _ in range(n)])
        self.update_out = nn.ModuleList([nn.Linear(cfg.update_hidden, dim, bias=False) for _ in range(n)])
        # Small (NOT zero) init on update_out. Zero-init kills the entire update
        # branch: with update≡0, z_pred stays at the predictor output, eps stops
        # depending on update_proj/gate, and those starve of gradient (the V8-V23
        # dead-gradient class). Small init gives a live gradient from step 0.
        for proj in self.update_out:
            nn.init.normal_(proj.weight, std=0.02)

        # Magnitude gate (replaces V19 corr_gate). Init neutral (sigmoid(0)=0.5).
        self.gate_w = nn.Parameter(torch.tensor([0.3]))
        self.gate_b = nn.Parameter(torch.tensor([0.0]))

        # Reused from V23's uncertainty.py: a learned per-position scalar
        # variance estimator (LearnedUncertainty) + the Bayes-optimal
        # inverse-variance blend (K = P/(P+R)). In V23 these were mis-applied to
        # a fake-parity-residual; here they serve their correct role —
        # aleatoric uncertainty weighting of the genuine innovation ε.
        # Optional: when None, the magnitude gate alone is used.
        self.use_kalman_blend = use_kalman_blend
        self.uncertainty = None
        if use_kalman_blend:
            from hagi_v4.model.uncertainty import LearnedUncertainty

            self.uncertainty = LearnedUncertainty(dim)

    def _kalman_blend(self, z_pred: torch.Tensor, innovation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K = P/(P+R) blend of prediction and innovation (reused from V23).

        The uncertainty Linear weights live in z_pred.dtype (bf16 under AMP),
        so call the estimator in that dtype; the variance math runs in float32
        for numerical stability, and the corrected state is cast back to
        z_pred.dtype to keep the iteration's dtypes consistent.
        """
        from hagi_v4.model.uncertainty import inverse_variance_update

        out_dtype = z_pred.dtype
        sigma2_pred = self.uncertainty(z_pred).float()  # estimator in native dtype
        innov_f = innovation.float()
        sigma2_meas = innov_f.pow(2).mean(dim=-1, keepdim=True)
        z_corrected, k = inverse_variance_update(z_pred.float(), innov_f, sigma2_pred, sigma2_meas)
        return z_corrected.to(out_dtype), k.mean()

    def forward(
        self,
        z: torch.Tensor,
        ctx: torch.Tensor,
        training: bool,
        iterations: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        n_iters = int(iterations) if iterations is not None else (
            self.cfg.train_iterations if training else self.cfg.infer_iterations
        )
        n_iters = max(1, n_iters)

        z_dtype = z.dtype
        z_pred = self.predictor(ctx).to(z_dtype)
        last_eps_norm = z.new_zeros(())
        kalman_gain_mean = z.new_zeros(())
        iters_used = torch.zeros(z.shape[:2], dtype=torch.long, device=z.device)
        z_norm_ref = z.float().pow(2).mean(dim=-1).clamp_min(1e-6)
        halted = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)

        for t in range(n_iters):
            eps = (z - z_pred).to(z_dtype)  # innovation (extrinsic error), keep native dtype
            eps_mag = eps.float().pow(2).mean(dim=-1)  # [B, T]
            rel_err = (eps_mag / z_norm_ref).sqrt()  # ‖ε‖/‖z‖
            active = ~halted
            iters_used[active] += 1

            gate = torch.sigmoid(eps_mag.sqrt() * self.gate_w + self.gate_b).unsqueeze(-1).to(z_dtype)
            update = self.update_out[t](
                torch.nn.functional.silu(self.update_proj[t](self.update_norm[t](eps)))
            )
            if self.use_kalman_blend:
                # Bayes-optimal K=P/(P+R) blend (reused V23 uncertainty.py).
                z_pred, kg = self._kalman_blend(z_pred, gate * update)
                kalman_gain_mean = kalman_gain_mean + kg
            else:
                z_pred = z_pred + gate * update  # extrinsic-only: adds NEW residual

            last_eps_norm = rel_err.mean()
            converged = rel_err < self.cfg.convergence_threshold
            if not training:
                halted = halted | converged

        side_info = {
            "iterations_used": iters_used,
            "final_innovation": last_eps_norm,
            "converged_frac": halted.float().mean() if not training else z.new_zeros(()),
            "kalman_gain_mean": kalman_gain_mean / max(n_iters, 1),
        }
        return z_pred, side_info
