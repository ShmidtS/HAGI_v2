"""Mixture of Experts (MoE) — entropy-aware routing = water-filling capacity.

Rehabilitated from the V23 ``moe.py`` (deleted as YAGNI in V25). MoE is the
scalability lever: it grows parameter count without growing per-token compute.
Information-theoretically it is water-filling — allocate capacity (power/dims)
to the parallel channels (positions) with the highest entropy/SNR.

Design:
  * ``n_shared`` shared experts (DeepSeek-Mo) ALWAYS fire and carry the common
    capacity, so routed experts specialize on the residual.
  * Top-k routing over the ``num_experts`` routed experts via a FP router that
    takes the hidden state + a per-position entropy scalar (cheap statistic) —
    high-entropy positions may route to heavier capacity = variable-rate coding.
  * Load-balance auxiliary loss = Switch-Transformer ``E * sum(f_i * P_i)``
    (CV^2 of expert usage).
  * Expert bodies are ternary ``BitLinear``; the router and the entropy gate
    are FP ``nn.Linear`` -> AdamW (routing is a source-side decision).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.config import MoEConfig
from hagi.model.norms import RMSNorm
from hagi.model.ternary import BitLinear


def _proj(in_f: int, out_f: int, use_ternary: bool) -> nn.Module:
    return BitLinear(in_f, out_f, bias=False) if use_ternary else nn.Linear(in_f, out_f, bias=False)


class SwiGLUExpert(nn.Module):
    """SwiGLU expert: down(silu(gate x) * (up x)). Body ternary when use_ternary."""

    def __init__(self, hidden_size: int, intermediate_size: int, use_ternary: bool) -> None:
        super().__init__()
        self.gate = _proj(hidden_size, intermediate_size, use_ternary)
        self.up = _proj(hidden_size, intermediate_size, use_ternary)
        self.down = _proj(intermediate_size, hidden_size, use_ternary)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoESwiGLU(nn.Module):
    """Top-k routed MoE + shared experts + entropy-aware routing.

    Args:
        hidden_size: H.
        intermediate_size: per-expert FFN width.
        cfg: MoE config (num_experts, top_k, n_shared, entropy_gate_weight).
        norm_eps: RMSNorm epsilon.
        use_ternary: ternarize expert bodies via BitLinear.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, cfg: MoEConfig, norm_eps: float = 1e-6, use_ternary: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.n_shared = cfg.n_shared
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.shared_experts = nn.ModuleList(
            SwiGLUExpert(hidden_size, intermediate_size, use_ternary) for _ in range(cfg.n_shared)
        )
        self.experts = nn.ModuleList(
            SwiGLUExpert(hidden_size, intermediate_size, use_ternary) for _ in range(cfg.num_experts)
        )
        # Router: hidden (+ entropy scalar) -> routing logits. FP -> AdamW.
        self.router = nn.Linear(hidden_size + 1, cfg.num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=cfg.router_init_std)
        self._entropy_gate_weight = float(cfg.entropy_gate_weight)
        self._last_load_balance: torch.Tensor | None = None

    @property
    def last_load_balance(self) -> torch.Tensor | None:
        return self._last_load_balance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Routed + shared expert output, residual-added to ``x``.

        Args:
            x: ``[B, T, H]``.

        Returns:
            ``[B, T, H]``.
        """
        b, t, h = x.shape
        h_n = self.norm(x)
        flat = h_n.reshape(b * t, h)

        # Entropy scalar per position: proxy local uncertainty from per-dim
        # variance of the (detached) hidden. Cheap, differentiable through
        # the router input only (the scalar itself is detached to avoid the
        # router gaming it).
        ent = flat.detach().pow(2).mean(dim=-1, keepdim=True).sqrt() * self._entropy_gate_weight
        router_in = torch.cat([flat, ent], dim=-1)
        logits = self.router(router_in)
        probs = F.softmax(logits, dim=-1)  # [N, E]

        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        n = flat.shape[0]
        out = flat.new_zeros(n, h)
        for k in range(self.top_k):
            idx_k = topk_idx[:, k]  # [N]
            w_k = topk_vals[:, k].unsqueeze(-1)  # [N,1]
            # Gather expert outputs via masked scatter.
            for e in range(self.num_experts):
                mask = idx_k == e
                if not mask.any():
                    continue
                out[mask] = out[mask] + self.experts[e](flat[mask]) * w_k[mask]

        # Shared experts always fire on the full batch.
        for se in self.shared_experts:
            out = out + se(flat)

        # Load-balance aux (Switch CV^2). Tracked on detached probs so it does
        # not leak into the shared-expert gradient path spuriously; the router
        # still receives the gradient through `probs` in the main forward above.
        if self.training:
            with torch.no_grad():
                f = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).float() / max(n, 1)
            p_mean = probs.mean(dim=0)
            self._last_load_balance = (self.num_experts * (f * p_mean).sum())

        out = out.view(b, t, h)
        return x + out
