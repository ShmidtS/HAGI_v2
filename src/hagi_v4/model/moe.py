# V21 DEFER: not wired in V21 forward path. Available for V22+ integration.
# See docs/ARCHITECTURE.md for integration roadmap.

"""Mixture of Experts (MoE) with MoD skip — Switch Transformer style.

V5: Entropy-aware routing. Router input includes per-position entropy
as an uncertainty signal. Low-entropy positions route to simple experts
(few bits needed), high-entropy positions route to complex experts
(variable-rate capacity allocation).

4 experts, top-1 routing. Optional MoD skip slot for trivial tokens
(residual identity). Load-balance aux loss (Shazeer/Switch) over real
experts only. Fused gate+up weight (gu_weight) for efficiency.
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
    """MoE with top-k routing, optional MoD skip, and shared basis decomposition.

    When n_shared_bases > 0: gate+up weights are shared across experts via
    N shared bases + per-expert mixing coefficients (PAW-style).
    Down weights remain per-expert (SiLU nonlinearity prevents sharing).
    This reduces MoE matmuls by (E-N)/E and params by ~33% at N=2.

    When n_shared_bases == 0: standard per-expert weights (backward compat).
    """

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
        self.n_shared_bases = cfg.n_shared_bases
        router_out = self.num_experts + (1 if self.use_mod_skip else 0)
        self.skip_idx = self.num_experts if self.use_mod_skip else -1

        self.router = nn.Linear(hidden_size + 1, router_out, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=cfg.router_init_std)

        if self.n_shared_bases > 0:
            N = self.n_shared_bases
            Itm = self.intermediate_size
            Hsz = hidden_size
            self.gu_bases = nn.Parameter(torch.randn(N, 2 * Itm, Hsz) * 0.02)
            self.gu_alpha = nn.Parameter(torch.randn(self.num_experts, N) * 0.1)
            self.down_weights = nn.Parameter(torch.randn(self.num_experts, Hsz, Itm) * 0.02)
        else:
            self.experts = nn.ModuleList(
                SwiGLUExpert(hidden_size, self.intermediate_size) for _ in range(self.num_experts)
            )

    def _shared_basis_forward(
        self, flat: torch.Tensor, expert_idx: torch.Tensor, probs_k: torch.Tensor, D: int
    ) -> torch.Tensor:
        N = self.n_shared_bases
        Itm = self.intermediate_size
        output = torch.zeros_like(flat)

        if self.use_mod_skip:
            skip_mask = (expert_idx == self.skip_idx).unsqueeze(-1).to(flat.dtype)
            output = output + flat * probs_k.unsqueeze(-1) * skip_mask

        basis_gu = F.linear(flat, self.gu_bases.reshape(N * 2 * Itm, self.hidden_size))
        basis_gu = basis_gu.view(-1, N, 2 * Itm)

        combined_gu = torch.einsum("en,...ni->...ei", self.gu_alpha, basis_gu)

        for e in range(self.num_experts):
            gate, up = combined_gu[:, e].chunk(2, dim=-1)
            act = F.silu(gate) * up
            expert_out = F.linear(act, self.down_weights[e])
            if expert_out.dtype != output.dtype:
                expert_out = expert_out.to(output.dtype)
            emask = (expert_idx == e).unsqueeze(-1).to(output.dtype)
            output = output + expert_out * probs_k.unsqueeze(-1) * emask

        return output

    def _per_expert_forward(
        self, flat: torch.Tensor, expert_idx: torch.Tensor, probs_k: torch.Tensor, D: int
    ) -> torch.Tensor:
        output = torch.zeros_like(flat)

        if self.use_mod_skip:
            skip_mask = (expert_idx == self.skip_idx).unsqueeze(-1).to(flat.dtype)
            output = output + flat * probs_k.unsqueeze(-1) * skip_mask

        for e in range(self.num_experts):
            expert_out = self.experts[e](flat)
            if expert_out.dtype != output.dtype:
                expert_out = expert_out.to(output.dtype)
            emask = (expert_idx == e).unsqueeze(-1).to(output.dtype)
            output = output + expert_out * probs_k.unsqueeze(-1) * emask

        return output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        with torch.no_grad():
            h_norm = flat.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            entropy_proxy = (flat.float().var(dim=-1, keepdim=True) / h_norm).to(flat.dtype)
        router_input = torch.cat([flat, entropy_proxy], dim=-1)

        router_logits = self.router(router_input)
        if self.training:
            noise = torch.randn_like(router_logits) * self.cfg.router_noise
            router_logits = router_logits + noise.detach()

        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.top_k > 1:
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat)

        if self.top_k == 1:
            expert_idx = top_k_indices[:, 0]
            probs_k = top_k_probs[:, 0]

            if self.n_shared_bases > 0:
                output = self._shared_basis_forward(flat, expert_idx, probs_k, D)
            else:
                output = self._per_expert_forward(flat, expert_idx, probs_k, D)
        else:
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[:, k_idx]
                probs = top_k_probs[:, k_idx]

                if self.use_mod_skip:
                    skip_mask = expert_idx == self.skip_idx
                    skip_out = flat[skip_mask] * probs[skip_mask].unsqueeze(-1)
                    indices = torch.where(skip_mask)[0]
                    idx_exp = indices.unsqueeze(-1).expand(-1, D)
                    output = output + torch.zeros_like(flat).scatter_add_(0, idx_exp, skip_out)

                for e in range(self.num_experts):
                    mask = expert_idx == e
                    tokens = flat[mask]
                    if self.n_shared_bases > 0:
                        N = self.n_shared_bases
                        Itm = self.intermediate_size
                        bg = F.linear(tokens, self.gu_bases.reshape(N * 2 * Itm, self.hidden_size))
                        bg = bg.view(-1, N, 2 * Itm)
                        cg = torch.einsum("en,...ni->...ei", self.gu_alpha, bg)
                        gate, up = cg[:, e].chunk(2, dim=-1)
                        expert_out = F.linear(F.silu(gate) * up, self.down_weights[e])
                    else:
                        expert_out = self.experts[e](tokens)
                    if expert_out.dtype != output.dtype:
                        expert_out = expert_out.to(output.dtype)
                    weighted = expert_out * probs[mask].unsqueeze(-1)
                    indices = torch.where(mask)[0]
                    idx_exp = indices.unsqueeze(-1).expand(-1, D)
                    output = output + torch.zeros_like(flat).scatter_add_(0, idx_exp, weighted)

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
