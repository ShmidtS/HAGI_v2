"""Water-filling Mixture of Experts (MoE) — capacity allocation by SNR.

V28 rewrite of the V27 ``MoESwiGLU``. The V27 router fed ``flat.pow(2).mean(-1)
.sqrt()`` (the RMS MAGNITUDE of h) into the router and called it an "entropy
gate" / "water-filling". That was information-theoretically dishonest: magnitude
is not entropy, not SNR, and the V27 forward was a python ``for k: for e:``
masked-dispatch loop.

This module implements genuine water-filling capacity allocation:

  * **SNR proxy gate** (the real water-filling signal). Capacity (routed-expert
    power) should follow the per-position SNR of the residual channel. The shared
    expert carries the baseline capacity; the routed experts specialize on the
    residual. A position whose residual after the shared expert is large has low
    SNR (much unexplained variance ~ noise); a small residual means high SNR.
    Water-filling ``P_i = max(0, mu - 1/SNR_i)`` allocates power to high-SNR
    channels, so the gate signal is ``s = 1/(||residual|| + eps)`` (detached).
  * **Entropy-regularized routing** = capacity maximization over the simplex:
    the routing loss adds a *routing-entropy maximization* bonus on top of the
    Switch CV^2 load balance, so all expert channels stay in use (no channel is
    starved -> capacity is maximized across the parallel expert channels).
  * **Batched expert dispatch** (no python masking loop): tokens are grouped by
    chosen expert via sort + segment offsets, experts run as batched matmuls.
    Scales with experts without per-token python overhead.

Information-theoretically it is water-filling: allocate capacity (power/dims)
to the parallel channels (positions) with the highest SNR.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hagi.config import MoEConfig
from hagi.model.norms import RMSNorm
from hagi.model.ternary import BitLinear
from hagi.model.water_filling import WaterFillingAllocator


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


class WaterFillingMoE(nn.Module):
    """Top-k routed MoE + shared experts + SNR-gated water-filling routing.

    Args:
        hidden_size: H.
        intermediate_size: per-expert FFN width.
        cfg: MoE config (num_experts, top_k, n_shared, snr_gate_weight,
            route_entropy_weight).
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
        # Router: hidden (+ SNR scalar) -> routing logits. FP -> AdamW.
        self.router = nn.Linear(hidden_size + 1, cfg.num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=cfg.router_init_std)
        self._snr_gate_weight = float(cfg.snr_gate_weight)
        self._last_load_balance: torch.Tensor | None = None
        self._last_routing_entropy: torch.Tensor | None = None
        self._last_water_filling_loss: torch.Tensor | None = None
        self._deferred_residual: torch.Tensor | None = None
        self._deferred_probs: torch.Tensor | None = None
        # Water-filling capacity allocator: per-expert capacity gate (log-bias on
        # the routing logits) steered by a per-expert SNR EMA on the shared-expert
        # residual. allocation_logits is an FP Parameter -> AdamW (not Muon).
        self.allocator = WaterFillingAllocator(
            total_width=self.num_experts * intermediate_size,
            num_experts=self.num_experts,
            min_width=min(64, intermediate_size),
        )

    @property
    def last_load_balance(self) -> torch.Tensor | None:
        return self._last_load_balance

    @property
    def last_routing_entropy(self) -> torch.Tensor | None:
        return self._last_routing_entropy

    @property
    def last_water_filling_loss(self) -> torch.Tensor | None:
        return self._last_water_filling_loss

    @torch.no_grad()
    def _update_allocator(self, residual: torch.Tensor, probs: torch.Tensor) -> None:
        """Steer the allocator's SNR EMA from the routed shared-expert residual.

        Per-expert residual variance (soft-weighted by the pre-allocation routing
        probs) feeds the allocator's ``1/variance`` SNR EMA. The EMA is detached;
        only :meth:`WaterFillingAllocator.allocation_probs` carries gradient
        through ``allocation_logits``.
        """
        token_var = residual.detach().float().pow(2).mean(dim=-1)  # [N]
        weights = probs.detach()  # [N, E]
        w_sum = weights.sum(dim=0).clamp_min(1e-6)  # [E]
        per_expert_var = (weights * token_var.unsqueeze(-1)).sum(dim=0) / w_sum  # [E]
        self.allocator.update_snr_ema(per_expert_var.to(self.allocator.snr_ema))

    def _snr_gate(self, residual: torch.Tensor) -> torch.Tensor:
        """Water-filling SNR proxy: ``s = 1 / (||residual||_2 + eps)`` per token.

        The residual is what the shared experts leave unexplained. High residual
        = noisy channel = low SNR -> less routed capacity; low residual = high
        SNR -> more capacity. Detached so the router cannot game the gate.
        """
        r = residual.detach().float().pow(2).mean(dim=-1).clamp_min(1e-6).sqrt()
        return (1.0 / r).to(residual.dtype) * self._snr_gate_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Routed (water-filling) + shared expert output, residual-added to ``x``.

        Args:
            x: ``[B, T, H]``.

        Returns:
            ``[B, T, H]``.
        """
        b, t, h = x.shape
        h_n = self.norm(x)
        flat = h_n.reshape(b * t, h)

        # Shared experts always fire (carry the common capacity).
        shared_out = flat.new_zeros(flat.shape)
        for se in self.shared_experts:
            shared_out = shared_out + se(flat)

        # SNR gate = water-filling signal on the shared-expert RESIDUAL.
        residual = flat - shared_out
        snr = self._snr_gate(residual)
        router_in = torch.cat([flat, snr.unsqueeze(-1)], dim=-1)
        logits = self.router(router_in)
        probs = F.softmax(logits, dim=-1)  # [N, E]

        # Water-filling capacity allocation: the allocator's per-expert capacity
        # gate is a log-bias on the routing logits. Uniform at init (constant
        # shift -> softmax-invariant), so the cold start is unchanged; as the SNR
        # EMA diverges the bias steers routing toward high-capacity (high-SNR)
        # experts. The EMA update is DEFERRED (stashed here, committed after
        # backward via commit_ema_update()) to keep the forward deterministic
        # under ckpt.checkpoint(use_reentrant=False): save and recompute must
        # read the SAME snr_ema, otherwise routing diverges and bincount shape
        # mismatches crash with CheckpointError.
        if self.training:
            self._deferred_residual = residual.detach()
            self._deferred_probs = probs.detach()
        alloc_probs = self.allocator.allocation_probs()  # [E]
        probs = F.softmax(logits + torch.log(alloc_probs + 1e-8).unsqueeze(0), dim=-1)

        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Batched dispatch: route shared-residual tokens to their chosen experts.
        routed_out = self._batched_dispatch(flat, residual, topk_idx, topk_vals)
        out = shared_out + routed_out

        # Auxiliaries (training only): Switch CV^2 load balance + routing entropy
        # + water-filling allocator entropy-gap regularizer.
        if self.training:
            n = flat.shape[0]
            with torch.no_grad():
                f = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).float() / max(n, 1)
            p_mean = probs.mean(dim=0)
            self._last_load_balance = self.num_experts * (f * p_mean).sum()
            # Routing entropy H(p): maximized -> capacity spread across channels.
            routed_prob_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            self._last_routing_entropy = routed_prob_entropy
            self._last_water_filling_loss = self.allocator.regularization_loss()

        out = out.view(b, t, h)
        return x + out

    @torch.no_grad()
    def commit_ema_update(self) -> None:
        """Apply the deferred SNR EMA update stashed during the last forward.

        Called from the training loop ONCE per step, AFTER backward (outside
        any ckpt.checkpoint scope). This keeps forward deterministic under
        use_reentrant=False: save and recompute read the same snr_ema, so
        routing log-bias -> probs -> topk_idx -> bincount counts are identical
        across both passes.
        """
        if self._deferred_residual is None or self._deferred_probs is None:
            return
        self._update_allocator(self._deferred_residual, self._deferred_probs)
        self._deferred_residual = None
        self._deferred_probs = None

    def _batched_dispatch(self, flat: torch.Tensor, residual: torch.Tensor, topk_idx: torch.Tensor, topk_vals: torch.Tensor) -> torch.Tensor:
        """Group tokens by chosen expert via sort + segment offsets; batched matmul.

        Always uses the batched path (sort + index_select + index_add_) — no
        boolean masking, so no nonzero/index_put launches. The exact naive
        masked-dispatch loop (:meth:`_naive_dispatch`) is kept as a reference
        for equivalence testing but is no longer on the forward path: it was the
        source of the nonzero/index_put launch storm the profiler attributed
        13.7% CPU / 2.25% CUDA to on a launch-bound GPU. A single ``.tolist()``
        pulls the per-expert offsets/counts in one host sync instead of
        ``2*num_experts`` per-expert ``.item()`` syncs.

        Args:
            flat: ``[N, H]`` normed hidden.
            residual: ``[N, H]`` shared-expert residual (the routed-expert input).
            topk_idx: ``[N, K]`` chosen expert indices.
            topk_vals: ``[N, K]`` normalized routing weights.

        Returns:
            ``[N, H]`` summed routed-expert output.
        """
        n = flat.shape[0]
        out = flat.new_zeros(n, self.hidden_size)
        if n == 0:
            return out
        # Batched path (sort + segment + index_select/index_add_): no boolean
        # masking, so no nonzero/index_put launches. The naive masked-dispatch
        # loop (below) was the source of the nonzero/index_put launch storm
        # the profiler attributed 13.7% CPU / 2.25% CUDA to on a launch-bound
        # GPU (ProfilerStep* 37.9% self CUDA = GPU starving under CPU dispatch).
        # A single .tolist() pulls the per-expert offsets/counts in one host
        # sync instead of 2*num_experts per-expert .item() syncs.
        flat_k = topk_idx.shape[1]
        # Flatten (token, slot) -> expert id, track source token.
        experts_flat = topk_idx.reshape(-1)  # [N*K]
        weights_flat = topk_vals.reshape(-1)  # [N*K]
        token_src = torch.arange(n, device=flat.device).repeat_interleave(flat_k)  # [N*K]
        # Sort by expert id so each expert's tokens are contiguous.
        order = torch.argsort(experts_flat, stable=True)
        experts_sorted = experts_flat[order]
        tokens_sorted = token_src[order]
        weights_sorted = weights_flat[order]
        # Segment boundaries per expert.
        counts = torch.bincount(experts_sorted, minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0) - counts
        offsets_l = offsets.tolist()
        counts_l = counts.tolist()
        for e in range(self.num_experts):
            start = offsets_l[e]
            cnt = counts_l[e]
            if cnt == 0:
                continue
            sel_tokens = tokens_sorted[start : start + cnt]
            sel_weights = weights_sorted[start : start + cnt].unsqueeze(-1)  # [cnt,1]
            sel_res = residual.index_select(0, sel_tokens)
            expert_out = self.experts[e](sel_res) * sel_weights
            out.index_add_(0, sel_tokens, expert_out.to(out.dtype))
        return out

    def _naive_dispatch(self, residual: torch.Tensor, topk_idx: torch.Tensor, topk_vals: torch.Tensor) -> torch.Tensor:
        """Naive masked-dispatch (the V27 pattern) — exact, for small N / tests."""
        n = residual.shape[0]
        out = residual.new_zeros(n, self.hidden_size)
        for k in range(self.top_k):
            idx_k = topk_idx[:, k]  # [N]
            w_k = topk_vals[:, k].unsqueeze(-1)  # [N,1]
            for e in range(self.num_experts):
                mask = idx_k == e
                if not mask.any():
                    continue
                out[mask] = out[mask] + self.experts[e](residual[mask]) * w_k[mask]
        return out


# Back-compat alias: existing imports / smoke tests use ``MoESwiGLU``.
MoESwiGLU = WaterFillingMoE
