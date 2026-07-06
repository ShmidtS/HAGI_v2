"""HRM — Spatial Refinement Controller for V4 plane prediction.

Key differences from V1/V3:
- z_H: [B, T//stride, h_dim] — spatial plane (not single vector)
- z_L: [B, T, l_dim] — per-token refinement state
- Iterations replace l_cycles
- NO h.detach() — gradient checkpointing per iteration instead
- Deep supervision at each iteration (NOT detached — provides gradient)
- Adaptive halting: scalar grade confidence → halt decision
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hagi_v4.config import HRMConfig, RefinementConfig
from hagi_v4.model.norms import RMSNorm
from hagi_v4.model.outputs import RefinementSideInfo


class LTransition(nn.Module):
    def __init__(self, l_dim: int, hidden_size: int):
        super().__init__()
        in_dim = l_dim + hidden_size
        self.norm = RMSNorm(in_dim)
        self.proj = nn.Linear(in_dim, 2 * l_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, z_L_prev: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_L_prev, h], dim=-1)
        x = self.norm(x)
        up, gate = self.proj(x).chunk(2, dim=-1)
        return z_L_prev + torch.sigmoid(gate) * self.act(up)


class HTransition(nn.Module):
    def __init__(self, h_dim: int, l_dim: int):
        super().__init__()
        in_dim = h_dim + l_dim
        self.proj = nn.Linear(in_dim, 2 * h_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, z_H_prev: torch.Tensor, z_L_coarse: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_H_prev, z_L_coarse], dim=-1)
        up, gate = self.proj(x).chunk(2, dim=-1)
        return z_H_prev + torch.sigmoid(gate) * self.act(up)


class AdaptiveHalting(nn.Module):
    def __init__(
        self,
        threshold: float = 0.9,
        min_iterations: int = 1,
        threshold_start: float = 0.05,
        threshold_end: float = 0.001,
    ):
        super().__init__()
        self.threshold = threshold
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.min_iterations = min_iterations
        self.halt_proj = nn.Linear(64, 1, bias=True)

    def forward(
        self,
        h_new: torch.Tensor,
        h_prev: torch.Tensor,
        iteration: int,
        already_halted: torch.Tensor,
        current_threshold: float | None = None,
    ) -> torch.Tensor:
        B, T, _ = h_new.shape
        if iteration < self.min_iterations:
            return torch.zeros(B, T, dtype=torch.bool, device=h_new.device)
        delta = (h_new - h_prev).float()
        delta_norm = delta.norm(dim=-1)
        h_norm = h_prev.float().norm(dim=-1)
        relative_delta = torch.where(h_norm > 0, delta_norm / h_norm, torch.zeros_like(delta_norm))
        min_satisfied = iteration >= self.min_iterations
        thresh = current_threshold if current_threshold is not None else 0.01
        return min_satisfied & (relative_delta < thresh) & (~already_halted)


class EntropyScheduler:
    """Determines the number of refinement iterations based on entropy proxy."""

    def __init__(self, cfg: RefinementConfig):
        self.n_iterations = cfg.num_iterations
        self.use_entropy_adaptive = cfg.use_entropy_adaptive_refinement
        self.entropy_low_threshold = cfg.entropy_low_threshold
        self.entropy_high_threshold = cfg.entropy_high_threshold
        self.entropy_low_iterations = cfg.entropy_low_iterations
        self.entropy_high_iterations = cfg.entropy_high_iterations

    def compute_n_iters(self, h: torch.Tensor) -> int:
        if not self.use_entropy_adaptive:
            return self.n_iterations
        with torch.no_grad():
            entropy_proxy = h.float().var(dim=1).mean().item()
        if entropy_proxy < self.entropy_low_threshold:
            return self.entropy_low_iterations
        if entropy_proxy > self.entropy_high_threshold:
            return self.entropy_high_iterations
        return self.n_iterations


class DeepSupervisor:
    """Deep supervision CE + EMA-weighted accumulation."""

    def __init__(self, cfg: RefinementConfig):
        self.use_deep_supervision = cfg.use_deep_supervision
        self.deep_supervision_decay = cfg.deep_supervision_decay
        self.deep_supervision_weight = cfg.deep_supervision_weight
        self.use_adaptive_ds_weight = cfg.use_adaptive_ds_weight
        self.ds_ema_decay = cfg.ds_ema_decay
        self._ds_ema: list[float] = []

    def init_ema(self, n_iters: int) -> None:
        if self.use_adaptive_ds_weight and len(self._ds_ema) < n_iters:
            self._ds_ema = [0.0] * n_iters

    def compute(
        self,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        targets: torch.Tensor | None,
        mask: torch.Tensor | None,
        lm_head_weight: torch.Tensor,
        final_norm: nn.Module,
        iteration: int,
        training: bool,
    ) -> tuple[torch.Tensor | None, float]:
        if not (training and self.use_deep_supervision and targets is not None and mask is not None and mask.any()):
            return None, 0.0

        h_masked = h[mask]
        t_masked = targets[mask]
        logits_masked = F.linear(final_norm(h_masked), lm_head_weight)
        ce = F.cross_entropy(logits_masked, t_masked)

        if self.use_adaptive_ds_weight:
            delta_norm = (h - h_prev).float().norm(dim=-1).mean().item()
            self._ds_ema[iteration] = self.ds_ema_decay * self._ds_ema[iteration] + (1 - self.ds_ema_decay) * delta_norm
            weight = self._ds_ema[iteration]
        else:
            weight = self.deep_supervision_decay**iteration

        del logits_masked, h_masked, t_masked
        return weight * ce, weight

    def finalize(self, total: torch.Tensor, weight_sum: float) -> torch.Tensor:
        if weight_sum > 0:
            return total * self.deep_supervision_weight / max(weight_sum, 1.0)
        return total


class RefinementCore(nn.Module):
    def __init__(
        self,
        cfg: HRMConfig,
        refinement_cfg: RefinementConfig,
        hidden_size: int = 576,
    ):
        super().__init__()
        self.cfg = cfg
        self.refinement_cfg = refinement_cfg
        self.hidden_size = hidden_size
        self.h_dim = cfg.h_state_dim
        self.l_dim = cfg.l_state_dim
        self.stride = cfg.h_stride
        self.halt_threshold_start = refinement_cfg.halt_threshold_start
        self.halt_threshold_end = refinement_cfg.halt_threshold_end
        self._max_steps: int = 150000

        self.h_init = nn.Linear(hidden_size, self.h_dim, bias=False)
        self.l_init = nn.Linear(hidden_size, self.l_dim, bias=False)
        self.z_h_init = nn.Parameter(torch.zeros(self.h_dim))
        self.z_l_init = nn.Parameter(torch.zeros(self.l_dim))

        self.z_h_to_hidden = nn.Linear(self.h_dim, hidden_size, bias=False)
        self.z_l_to_hidden = nn.Linear(self.l_dim, hidden_size, bias=False)

        self.l_transition = LTransition(self.l_dim, hidden_size)
        self.h_transition = HTransition(self.h_dim, self.l_dim)

        self.adaptive_halt: AdaptiveHalting | None = None
        if refinement_cfg.use_adaptive_halt:
            self.adaptive_halt = AdaptiveHalting(
                threshold=refinement_cfg.halt_threshold,
                min_iterations=refinement_cfg.min_iterations,
                threshold_start=refinement_cfg.halt_threshold_start,
                threshold_end=refinement_cfg.halt_threshold_end,
            )

        self.entropy_scheduler = EntropyScheduler(refinement_cfg)
        self.deep_supervisor = DeepSupervisor(refinement_cfg)

    def set_max_steps(self, max_steps: int) -> None:
        self._max_steps = max_steps

    def _adaptive_halt_threshold(self, step: int) -> float:
        progress = min(1.0, step / max(self._max_steps, 1))
        return self.halt_threshold_start + (self.halt_threshold_end - self.halt_threshold_start) * progress

    def _run_reasoning_blocks(
        self,
        h: torch.Tensor,
        reasoning_blocks: nn.ModuleList,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        moe_aux_acc = h.new_zeros(())
        router_probs_list = []
        for blk in reasoning_blocks:
            h, aux, rp = blk(h, cos, sin)
            moe_aux_acc = moe_aux_acc + aux
            router_probs_list.append(rp)
        if router_probs_list:
            router_probs_stacked = torch.stack(router_probs_list, dim=0)
        else:
            router_probs_stacked = h.new_zeros((0,))
        return h, moe_aux_acc, router_probs_stacked

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
        final_norm: nn.Module,
        training: bool,
        mask: torch.Tensor | None,
        step: int = 0,
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

        h_prev = h.clone()
        halted = torch.zeros(B, T, dtype=torch.bool, device=device)
        iterations_used = torch.full((B, T), n_iters, dtype=torch.long, device=device)

        deep_weight_sum = 0.0
        halt_threshold = self._adaptive_halt_threshold(step)

        self.deep_supervisor.init_ema(n_iters)

        last_gate_probs: torch.Tensor | None = None
        last_gp2d_residual: torch.Tensor | None = None
        last_router_probs_list: list[torch.Tensor] | None = None

        for iteration in range(n_iters):
            if iteration > 0:
                msa_out, lb = msa.read(h, top_k=6)
                h = h + msa_out
                total_msa_lb = total_msa_lb + lb

            z_H_up = self._upsample_z_h(z_H, T, S)
            h_bias = self.z_h_to_hidden(z_H_up) + self.z_l_to_hidden(z_L)
            h = h + h_bias

            if training:

                def _run(h_inner):
                    return self._run_reasoning_blocks(h_inner, reasoning_blocks, cos, sin)

                h, moe_aux, rp_stacked = torch_checkpoint(_run, h, use_reentrant=False)
            else:
                h, moe_aux, rp_stacked = self._run_reasoning_blocks(h, reasoning_blocks, cos, sin)
            total_moe_aux = total_moe_aux + moe_aux

            h, router_loss, gate_probs = gdr(h)
            if router_loss is not None:
                total_gdr_router = total_gdr_router + router_loss

            if training:

                def _gp2d_run(h_inner):
                    return gp2d(h_inner)

                h, gp2d_residual = torch_checkpoint(_gp2d_run, h, use_reentrant=False)
            else:
                h, gp2d_residual = gp2d(h)

            msa.write(h)

            if self.adaptive_halt is not None:
                new_halts = self.adaptive_halt(h, h_prev, iteration, halted, current_threshold=halt_threshold)
                h = torch.where(new_halts.unsqueeze(-1), h_prev, h)
                halted = halted | new_halts
                iterations_used = torch.where(
                    new_halts & (iterations_used == n_iters),
                    torch.full_like(iterations_used, iteration + 1),
                    iterations_used,
                )

            h_prev = h

            ds_loss, ds_weight = self.deep_supervisor.compute(
                h, h_prev, targets, mask, lm_head_weight, final_norm, iteration, training
            )
            if ds_loss is not None:
                total_deep_supervision = total_deep_supervision + ds_loss
                deep_weight_sum += ds_weight

            z_L = self.l_transition(z_L, h)
            z_L_coarse = z_L[:, : T_c * S].view(B, T_c, S, self.l_dim).mean(dim=2)
            z_H = self.h_transition(z_H, z_L_coarse)

            last_gate_probs = gate_probs
            last_gp2d_residual = gp2d_residual
            if rp_stacked.numel() > 0:
                last_router_probs_list = list(rp_stacked.unbind(0))

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
        )
        return h, side_info
