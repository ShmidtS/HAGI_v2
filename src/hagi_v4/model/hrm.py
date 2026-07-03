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


class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, H = h.shape
        T_c = T // self.stride
        h_trim = h[:, : T_c * self.stride]
        h_coarse = h_trim.view(B, T_c, self.stride, H).mean(dim=2)
        return self.proj(h_coarse)


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
    def __init__(self, threshold: float = 0.9, min_iterations: int = 1):
        super().__init__()
        self.threshold = threshold
        self.min_iterations = min_iterations
        self.halt_proj = nn.Linear(64, 1, bias=True)

    def forward(
        self,
        h_new: torch.Tensor,
        h_prev: torch.Tensor,
        iteration: int,
        already_halted: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = h_new.shape
        if iteration < self.min_iterations:
            return torch.zeros(B, T, dtype=torch.bool, device=h_new.device)
        delta = (h_new - h_prev).float()
        delta_norm = delta.norm(dim=-1)  # [B, T]
        h_norm = h_prev.float().norm(dim=-1)  # [B, T]
        relative_delta = torch.where(h_norm > 0, delta_norm / h_norm, torch.zeros_like(delta_norm))  # [B, T]
        min_satisfied = iteration >= self.min_iterations
        return min_satisfied & (relative_delta < 0.01) & (~already_halted)


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
        self.n_iterations = refinement_cfg.num_iterations
        self.min_iterations = refinement_cfg.min_iterations
        self.use_deep_supervision = refinement_cfg.use_deep_supervision
        self.deep_supervision_decay = refinement_cfg.deep_supervision_decay
        self.deep_supervision_weight = refinement_cfg.deep_supervision_weight

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
            )

    def _run_reasoning_blocks(
        self,
        h: torch.Tensor,
        reasoning_blocks: nn.ModuleList,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moe_aux_acc = torch.tensor(0.0, device=h.device, dtype=torch.float32)
        for blk in reasoning_blocks:
            h, aux = blk(h, cos, sin)
            moe_aux_acc = moe_aux_acc + aux
        return h, moe_aux_acc

    def _deep_supervision_ce(
        self,
        h: torch.Tensor,
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h_masked = h[mask]
        t_masked = targets[mask]
        logits = F.linear(h_masked, lm_head_weight)
        return F.cross_entropy(logits, t_masked)

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
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        B, T, H = h.shape
        device = h.device
        S = self.stride
        T_c = T // S

        h_coarse = h[:, : T_c * S].view(B, T_c, S, H).mean(dim=2)
        z_H = self.h_init(h_coarse) + self.z_h_init
        z_L = self.l_init(h) + self.z_l_init

        total_moe_aux = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_gdr_router = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_msa_lb = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_deep_supervision = torch.tensor(0.0, device=device, dtype=torch.float32)

        h_prev = h.clone()
        halted = torch.zeros(B, T, dtype=torch.bool, device=device)
        iterations_used = torch.full((B, T), self.n_iterations, dtype=torch.long, device=device)

        deep_weight_sum = 0.0

        for iteration in range(self.n_iterations):
            if iteration > 0:
                msa_out, lb = msa.read(h, top_k=6)
                h = h + msa_out
                total_msa_lb = total_msa_lb + lb

        z_H_up = z_H.repeat_interleave(S, dim=1)
        if z_H_up.shape[1] < T:
            pad = z_H[:, -1:].repeat(1, T - z_H_up.shape[1], 1)
            z_H_up = torch.cat([z_H_up, pad], dim=1)
        z_H_up = z_H_up[:, :T]
        h_bias = self.z_h_to_hidden(z_H_up) + self.z_l_to_hidden(z_L)
        h = h + h_bias

        if training:

            def _run(h_inner):
                return self._run_reasoning_blocks(h_inner, reasoning_blocks, cos, sin)

            h, moe_aux = torch_checkpoint(_run, h, use_reentrant=False)
        else:
            h, moe_aux = self._run_reasoning_blocks(h, reasoning_blocks, cos, sin)
        total_moe_aux = total_moe_aux + moe_aux

        h, router_loss = gdr(h)
        if router_loss is not None:
            total_gdr_router = total_gdr_router + router_loss

        h = gp2d(h)

        msa.write(h)

        if self.adaptive_halt is not None:
            new_halts = self.adaptive_halt(h, h_prev, iteration, halted)
            h = torch.where(new_halts.unsqueeze(-1), h_prev, h)
            halted = halted | new_halts
            iterations_used = torch.where(
                new_halts & (iterations_used == self.n_iterations),
                torch.full_like(iterations_used, iteration + 1),
                iterations_used,
            )

        h_prev = h

        if training and self.use_deep_supervision and targets is not None and mask is not None and mask.any():
            h_masked = h[mask]
            t_masked = targets[mask]
            logits_masked = F.linear(final_norm(h_masked), lm_head_weight)
            ce = F.cross_entropy(logits_masked, t_masked)
            weight = self.deep_supervision_decay**iteration
            total_deep_supervision = total_deep_supervision + weight * ce
            deep_weight_sum += weight
            del logits_masked, h_masked, t_masked

        z_L = self.l_transition(z_L, h)
        z_L_coarse = z_L[:, : T_c * S].view(B, T_c, S, self.l_dim).mean(dim=2)
        z_H = self.h_transition(z_H, z_L_coarse)

        n = max(self.n_iterations, 1)
        losses = {
            "moe_aux": total_moe_aux / n,
            "gdr_router": total_gdr_router / n,
            "msa_lb": total_msa_lb / n,
            "deep_supervision": total_deep_supervision * self.deep_supervision_weight / max(deep_weight_sum, 1.0)
            if deep_weight_sum > 0
            else total_deep_supervision,
        }
        return h, losses, iterations_used
