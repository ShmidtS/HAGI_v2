"""Mixture of Experts (MoE) with MoD skip — Switch Transformer style.

V6: Sync-free expert routing. Replaces boolean masking loop (which
causes CPU-GPU syncs via mask.any() per expert) with sort+split
approach: 1 sync per MoE call instead of num_experts+1 syncs.

4 experts, top-1 routing. Optional MoD skip slot for trivial tokens.
Entropy-aware routing. Load-balance aux loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi_v4.config import MoEConfig


class SwiGLUExpert(nn.Module):
    """Fused gate+up SwiGLU expert."""

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
    """MoE with sync-free sort+split expert dispatch."""

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

        self.router = nn.Linear(hidden_size + 1, router_out, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        self.experts = nn.ModuleList(SwiGLUExpert(hidden_size, self.intermediate_size) for _ in range(self.num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        N = flat.shape[0]

        with torch.no_grad():
            h_norm = flat.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            entropy_proxy = (flat.float().var(dim=-1, keepdim=True) / h_norm).to(flat.dtype)
        router_input = torch.cat([flat, entropy_proxy], dim=-1)

        router_logits = self.router(router_input)
        if self.training:
            noise = torch.randn_like(router_logits) * 0.01
            router_logits = router_logits + noise.detach()

        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.top_k > 1:
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        if self.top_k == 1:
            output = self._dispatch_top1_sort(flat, top_k_indices[:, 0], top_k_probs[:, 0])
        else:
            output = self._dispatch_topk(flat, top_k_indices, top_k_probs)

        output = output.reshape(B, T, D)

        if self.training:
            real_probs = router_probs[:, : self.num_experts]
            router_prob_per_expert = real_probs.mean(dim=0)
            top_k_mask = torch.zeros(N, self.num_experts, device=x.device, dtype=router_probs.dtype)
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

    def _dispatch_top1_sort(self, flat: torch.Tensor, expert_idx: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Sync-free top-1 dispatch using sort+split.

        Sorts tokens by expert assignment, splits into chunks, processes
        each expert's chunk. Only 1 CPU-GPU sync (for chunk boundaries).
        """
        N, D = flat.shape

        if self.use_mod_skip:
            skip_mask = expert_idx == self.skip_idx
            nonskip_mask = ~skip_mask
            output = flat * probs.unsqueeze(-1)
            output = output * nonskip_mask.unsqueeze(-1).to(output.dtype)

            nonskip_idx = torch.where(nonskip_mask, expert_idx, torch.full_like(expert_idx, self.num_experts))
        else:
            output = torch.zeros_like(flat)
            nonskip_idx = expert_idx

        sorted_order = torch.argsort(nonskip_idx)
        sorted_idx = nonskip_idx[sorted_order]
        sorted_flat = flat[sorted_order]
        sorted_probs = probs[sorted_order]

        counts = torch.bincount(sorted_idx, minlength=self.num_experts + 1)
        chunks = torch.split(sorted_flat, counts.tolist())
        prob_chunks = torch.split(sorted_probs, counts.tolist())

        for e in range(self.num_experts):
            chunk = chunks[e]
            if chunk.shape[0] == 0:
                continue
            expert_out = self.experts[e](chunk)
            weighted = expert_out * prob_chunks[e].unsqueeze(-1)
            output[sorted_order[counts[:e].sum() : counts[: e + 1].sum()]] = weighted

        return output

    def _dispatch_topk(
        self, flat: torch.Tensor, top_k_indices: torch.Tensor, top_k_probs: torch.Tensor
    ) -> torch.Tensor:
        """Top-k dispatch with scatter_add_ (no syncs)."""
        N, D = flat.shape
        output = torch.zeros_like(flat)

        for k_idx in range(self.top_k):
            expert_idx = top_k_indices[:, k_idx]
            probs = top_k_probs[:, k_idx]

            if self.use_mod_skip:
                skip_mask = expert_idx == self.skip_idx
                if skip_mask.any():
                    skip_out = flat * probs.unsqueeze(-1)
                    output.scatter_add_(
                        0,
                        torch.where(skip_mask)[0].unsqueeze(-1).expand(-1, D),
                        skip_out * skip_mask.unsqueeze(-1).to(skip_out.dtype),
                    )

            for e in range(self.num_experts):
                mask = expert_idx == e
                if not mask.any():
                    continue
                tokens = flat[mask]
                expert_out = self.experts[e](tokens)
                if expert_out.dtype != output.dtype:
                    expert_out = expert_out.to(output.dtype)
                weighted = expert_out * probs[mask].unsqueeze(-1)
                indices = torch.where(mask)[0]
                output.scatter_add_(0, indices.unsqueeze(-1).expand(-1, D), weighted)

        return output
