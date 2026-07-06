"""Mixture of Experts (MoE) with MoD skip — Switch Transformer style.

4 experts, top-1 routing. Optional MoD skip slot for trivial tokens
(residual identity). Load-balance aux loss (Shazeer/Switch) over real
experts only. Fused gate+up weight (gu_weight) for efficiency.

Uses mask-based dispatch (not sort/unique_consecutive) for vectorized
expert routing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi_v4.config import MoEConfig


class SwiGLUExpert(nn.Module):
    """Fused gate+up SwiGLU expert.

    gu_weight: [2*intermediate, hidden] — fused gate and up projections.
    down: nn.Linear [hidden, intermediate] (weight shape).
    Named gu_weight to avoid "gate" token in Muon residual-scale exclude list.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gu_weight = nn.Parameter(torch.cat([gate.weight, up.weight], dim=0).contiguous())
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = F.linear(x, self.gu_weight)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class MoESwiGLU(nn.Module):
    """MoE with top-k routing and optional MoD skip slot."""

    def __init__(self, cfg: MoEConfig, hidden_size: int = 576):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.hidden_size = hidden_size
        self.intermediate_size = cfg.intermediate_size
        self.alpha = cfg.alpha
        self.use_mod_skip = cfg.use_mod_skip
        self.use_grade_specialization = cfg.use_grade_specialization
        router_out = self.num_experts + (1 if self.use_mod_skip else 0)
        self.skip_idx = self.num_experts if self.use_mod_skip else -1

        self.router = nn.Linear(hidden_size, router_out, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        self.experts = nn.ModuleList(SwiGLUExpert(hidden_size, self.intermediate_size) for _ in range(self.num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        router_logits = self.router(flat)
        if self.training:
            noise = torch.randn_like(router_logits) * 0.01
            router_logits = router_logits + noise.detach()

        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.top_k > 1:
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat)

        for k_idx in range(self.top_k):
            expert_idx = top_k_indices[:, k_idx]
            probs = top_k_probs[:, k_idx]

            if self.use_mod_skip:
                skip_mask = expert_idx == self.skip_idx
                if skip_mask.any():
                    skip_out = flat[skip_mask] * probs[skip_mask].unsqueeze(-1)
                    if self.top_k == 1:
                        output[skip_mask] = skip_out
                    else:
                        indices = torch.where(skip_mask)[0]
                        idx_exp = indices.unsqueeze(-1).expand(-1, D)
                        output.scatter_add_(0, idx_exp, skip_out)

            for e in range(self.num_experts):
                mask = expert_idx == e
                if not mask.any():
                    continue
                tokens = flat[mask]
                expert_out = self.experts[e](tokens)
                if expert_out.dtype != output.dtype:
                    expert_out = expert_out.to(output.dtype)
                weighted = expert_out * probs[mask].unsqueeze(-1)
                if self.top_k == 1:
                    output[mask] = weighted
                else:
                    indices = torch.where(mask)[0]
                    idx_exp = indices.unsqueeze(-1).expand(-1, D)
                    output.scatter_add_(0, idx_exp, weighted)

        output = output.reshape(B, T, D)

        if self.training:
            real_probs = router_probs[:, : self.num_experts]
            router_prob_per_expert = real_probs.mean(dim=0)
            top_k_mask = torch.zeros(
                B * T,
                self.num_experts,
                device=x.device,
                dtype=router_probs.dtype,
            )
            if self.use_mod_skip:
                nonskip_sel = top_k_indices != self.skip_idx
                safe_idx = torch.where(
                    top_k_indices >= self.num_experts, torch.zeros_like(top_k_indices), top_k_indices
                )
                top_k_mask.scatter_(1, safe_idx, 1.0)
                top_k_mask = top_k_mask * nonskip_sel.float()
            else:
                top_k_mask.scatter_(1, top_k_indices, 1.0)
            fraction_per_expert = top_k_mask.mean(dim=0)
            aux_loss = self.alpha * self.num_experts * (fraction_per_expert * router_prob_per_expert).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return output, aux_loss, router_probs.detach()
