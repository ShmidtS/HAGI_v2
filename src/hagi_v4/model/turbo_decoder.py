"""Turbo Decoder — dual-component iterative belief propagation.

Shannon analogy: Turbo codes use two parallel concatenated component
decoders that exchange extrinsic information iteratively. Each component
decoder processes the received signal with a different parity structure,
and they exchange ONLY extrinsic information (not full posterior) to
avoid information recycling.

V6 mapping:
  Component A: attention-based reasoning (local parity checks)
    7 reasoning layers -> extrinsic_A = h_out - h_prior
  Component B: GDR + GP2D refinement (long-range parity checks)
    grade update + geometric product -> extrinsic_B = h_out - h_prior
  Exchange: h = h_prior + alpha_A * extrinsic_A + alpha_B * extrinsic_B
  Convergence: ||extrinsic_A|| + ||extrinsic_B|| < epsilon -> stop

Drop-in replacement for RefinementCore — same forward() signature.
Creates its own EntropyScheduler and DeepSupervisor internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hagi_v4.config import HRMConfig, RefinementConfig
from hagi_v4.model.hrm import DeepSupervisor, EntropyScheduler
from hagi_v4.model.outputs import RefinementSideInfo


class TurboDecoder(nn.Module):
    """Dual-component Turbo decoder with extrinsic information exchange.

    Component A (attention): local parity via bidirectional attention.
    Component B (GDR+GP2D): long-range parity via grade update + geometric product.
    Extrinsic from each component feeds the other's next iteration.
    """

    def __init__(
        self,
        cfg: HRMConfig,
        refinement_cfg: RefinementConfig,
        hidden_size: int = 288,
    ):
        super().__init__()
        self.cfg = cfg
        self.refinement_cfg = refinement_cfg
        self.hidden_size = hidden_size
        self.h_dim = cfg.h_state_dim
        self.l_dim = cfg.l_state_dim
        self.stride = cfg.h_stride
        self.min_iterations = refinement_cfg.min_iterations
        self.max_iterations = refinement_cfg.num_iterations
        self.convergence_threshold = refinement_cfg.convergence_threshold

        self.alpha_a = nn.Parameter(torch.tensor(float(refinement_cfg.extrinsic_alpha)))
        self.alpha_b = nn.Parameter(torch.tensor(float(refinement_cfg.extrinsic_alpha)))

        self.h_init = nn.Linear(hidden_size, self.h_dim, bias=False)
        self.l_init = nn.Linear(hidden_size, self.l_dim, bias=False)
        self.z_h_init = nn.Parameter(torch.zeros(self.h_dim))
        self.z_l_init = nn.Parameter(torch.zeros(self.l_dim))
        self.z_h_to_hidden = nn.Linear(self.h_dim, hidden_size, bias=False)
        self.z_l_to_hidden = nn.Linear(self.l_dim, hidden_size, bias=False)

        self.entropy_scheduler = EntropyScheduler(refinement_cfg)
        self.deep_supervisor = DeepSupervisor(refinement_cfg)
        self._max_steps: int = 150000

    def set_max_steps(self, max_steps: int) -> None:
        self._max_steps = max_steps

    def _upsample_z_h(self, z_H: torch.Tensor, T: int, S: int) -> torch.Tensor:
        z_H_up = z_H.repeat_interleave(S, dim=1)
        if z_H_up.shape[1] < T:
            pad = z_H[:, -1:].repeat(1, T - z_H_up.shape[1], 1)
            z_H_up = torch.cat([z_H_up, pad], dim=1)
        return z_H_up[:, :T]

    def forward(
        self,
        h: torch.Tensor,
        reasoning_blocks: nn.ModuleList,
        gdr: nn.Module,
        gp2d: nn.Module,
        msa: nn.Module,
        cos: torch.Tensor,
        sin: torch.Tensor,
        targets: torch.Tensor | None,
        lm_head_weight: torch.Tensor,
        training: bool,
        mask: torch.Tensor | None,
        step: int = 0,
        extrinsic_alpha: float = 1.0,
        convergence_threshold: float = 0.01,
        use_convergence_halt: bool = True,
    ) -> tuple[torch.Tensor, RefinementSideInfo]:
        B, T, H = h.shape
        device = h.device
        S = self.stride
        T_c = T // S

        n_iters = self.entropy_scheduler.compute_n_iters(h)

        h_coarse = h[:, : T_c * S].view(B, T_c, S, H).mean(dim=2)
        z_H = self.h_init(h_coarse) + self.z_h_init
        z_L = self.l_init(h) + self.z_l_init

        total_moe_aux = h.new_zeros(())
        total_gdr_router = h.new_zeros(())
        total_msa_lb = h.new_zeros(())
        total_deep_supervision = h.new_zeros(())
        total_parity = h.new_zeros(())

        halted = torch.zeros(B, T, dtype=torch.bool, device=device)
        iterations_used = torch.full((B, T), n_iters, dtype=torch.long, device=device)

        deep_weight_sum = 0.0
        extrinsic_norms: list[float] = []
        converged_at = n_iters

        last_gate_probs: torch.Tensor | None = None
        last_gp2d_residual: torch.Tensor | None = None
        last_router_probs_list: list[torch.Tensor] | None = None

        self.deep_supervisor.init_ema(n_iters)

        alpha_a = torch.sigmoid(self.alpha_a)
        alpha_b = torch.sigmoid(self.alpha_b)

        for iteration in range(n_iters):
            h_prior = h

            if iteration > 0:
                msa_out, lb = msa.read(h, top_k=6)
                h = h + msa_out
                total_msa_lb = total_msa_lb + lb

            z_H_up = self._upsample_z_h(z_H, T, S)
            h_bias = self.z_h_to_hidden(z_H_up) + self.z_l_to_hidden(z_L)
            h = h + h_bias

            # === Component A: Attention-based (local parity) ===
            if training:

                def _run_a(h_inner):
                    moe_aux = h_inner.new_zeros(())
                    rp_list = []
                    for blk in reasoning_blocks:
                        h_inner, aux, rp = blk(h_inner, cos, sin)
                        moe_aux = moe_aux + aux
                        rp_list.append(rp)
                    rp_stacked = torch.stack(rp_list, dim=0) if rp_list else h_inner.new_zeros((0,))
                    return h_inner, moe_aux, rp_stacked

                h_a, moe_aux, rp_stacked = torch_checkpoint(_run_a, h, use_reentrant=False)
            else:
                moe_aux = h.new_zeros(())
                rp_list = []
                for blk in reasoning_blocks:
                    h_a, aux, rp = blk(h, cos, sin)
                    moe_aux = moe_aux + aux
                    rp_list.append(rp)
                rp_stacked = torch.stack(rp_list, dim=0) if rp_list else h.new_zeros((0,))

            total_moe_aux = total_moe_aux + moe_aux

            extrinsic_a = h_a - h_prior
            h = h_prior + alpha_a * extrinsic_a
            ext_a_norm = extrinsic_a.float().norm(dim=-1).mean().item()

            # === Component B: GDR + GP2D (long-range parity) ===
            h_b_prior = h
            h, router_loss, gate_probs = gdr(h)
            if router_loss is not None:
                total_gdr_router = total_gdr_router + router_loss

            if training:

                def _gp2d_run(h_inner):
                    return gp2d(h_inner)

                h, gp2d_residual = torch_checkpoint(_gp2d_run, h, use_reentrant=False)
            else:
                h, gp2d_residual = gp2d(h)

            total_parity = total_parity + gp2d_residual.pow(2).mean().to(total_parity.dtype)
            msa.write(h)

            extrinsic_b = h - h_b_prior
            h = h_b_prior + alpha_b * extrinsic_b
            ext_b_norm = extrinsic_b.float().norm(dim=-1).mean().item()
            extrinsic_norms.append(ext_a_norm + ext_b_norm)

            # Deep supervision
            ds_loss, ds_weight = self.deep_supervisor.compute(
                h, h_prior, targets, mask, lm_head_weight, iteration, training
            )
            if ds_loss is not None:
                total_deep_supervision = total_deep_supervision + ds_loss
                deep_weight_sum += ds_weight

            # HRM state transitions
            from hagi_v4.model.hrm import HTransition, LTransition

            if not hasattr(self, "_l_transition"):
                self._l_transition = LTransition(self.l_dim, self.hidden_size)
                self._h_transition = HTransition(self.h_dim, self.l_dim)
                self._l_transition = self._l_transition.to(device)
                self._h_transition = self._h_transition.to(device)

            z_L = self._l_transition(z_L, h)
            z_L_coarse = z_L[:, : T_c * S].view(B, T_c, S, self.l_dim).mean(dim=2)
            z_H = self._h_transition(z_H, z_L_coarse)

            last_gate_probs = gate_probs
            last_gp2d_residual = gp2d_residual
            if rp_stacked.numel() > 0:
                last_router_probs_list = list(rp_stacked.unbind(0))

            # Convergence check (Turbo: combined extrinsic)
            combined_ext = ext_a_norm + ext_b_norm
            if use_convergence_halt and iteration >= self.min_iterations:
                if combined_ext < convergence_threshold:
                    converged_at = iteration + 1
                    not_halted = ~halted
                    iterations_used = torch.where(
                        not_halted & (iterations_used == n_iters),
                        torch.full_like(iterations_used, iteration + 1),
                        iterations_used,
                    )
                    break

        actual_iters = max(converged_at, 1)
        n = max(n_iters, 1)

        side_info = RefinementSideInfo(
            deep_supervision_loss=self.deep_supervisor.finalize(total_deep_supervision, deep_weight_sum),
            gdr_gate_probs=last_gate_probs,
            gdr_router_loss=total_gdr_router / n,
            gp2d_residual=last_gp2d_residual,
            moe_router_probs=last_router_probs_list,
            moe_lb=total_moe_aux / n,
            msa_lb=total_msa_lb / n,
            iterations_used=iterations_used,
            extrinsic_norms=extrinsic_norms,
            parity_strength=total_parity / actual_iters,
        )
        return h, side_info
