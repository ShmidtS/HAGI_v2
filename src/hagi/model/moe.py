"""Mixture of experts — variable-rate coding with bias-controlled balance.

MoE is the mechanism that decouples capacity from per-token compute: total
parameters grow with ``num_experts`` while the FLOPs per token grow only with
``top_k``. Read as a code, each expert is a sub-codebook and the router performs
variable-rate assignment — tokens are sent to the experts that code them best.

Balance is the whole problem. If routing collapses onto a few experts the
effective capacity drops to those experts and the remaining parameters are dead
weight; the routing distribution's entropy is exactly the usable fraction of the
parallel channels.

V31 balances with a **bias controller** instead of an auxiliary loss:

    logits_e   = router(x)_e
    selection  = top_k(logits_e + b_e)
    weights    = softmax over the selected logits, WITHOUT b
    b_e       <- b_e + gamma * sign(mean_load - load_e)         (no gradient)

Two properties make this work where the V28 auxiliary loss did not:

* The bias affects *selection* only, never the combining weights. The gradient
  the experts receive is the pure language-modelling gradient — nothing competes
  with it, and the balance mechanism cannot distort the function being learned.
* It is a feedback controller with a fixed step, so it corrects an imbalance at a
  guaranteed rate regardless of how strong the opposing LM gradient is. The V28
  CV^2 penalty was a *gradient*, so a stronger LM gradient simply overruled it:
  measured ``moe_lb = 8.2`` against the 8.0 ceiling for E=4/top_k=2 (complete
  collapse) for 50k steps while the penalty was active with weight 0.1.

Also removed from V28: the "water-filling" SNR gate. It multiplied the routing
logits by ``1 + 1/||residual||``, which sharpens the softmax (a temperature
reduction, not a power allocation), and the companion allocator added a log-bias
toward experts whose residual was already smallest. Both push the same direction
— toward whichever expert is already winning. Water-filling assigns *more* power
to high-SNR channels only under a total-power constraint that makes the
assignment zero-sum; there was no such constraint here, so the result was
positive feedback with nothing opposing it.

The dispatch is a sort-by-expert plus segmented matmul: no boolean masking, one
host sync for the segment offsets, and cost proportional to assigned tokens.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi.config import MoEConfig
from hagi.model.ffn import SwiGLU
from hagi.model.norms import RMSNorm


class MoE(nn.Module):
    """Top-k routed experts + always-on shared experts.

    Args:
        hidden_size: H.
        intermediate_size: per-expert mixer width.
        cfg: MoE configuration.
        norm_eps: RMSNorm epsilon.
        use_ternary: quantize expert bodies.
        residual_scale: init scale on every expert's down-projection.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        cfg: MoEConfig,
        norm_eps: float = 1e-5,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
        init_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        if cfg.num_experts < 2:
            raise ValueError("MoE needs at least 2 experts")
        if not 1 <= cfg.top_k <= cfg.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.n_shared = cfg.n_shared
        self.hidden_size = hidden_size
        self.bias_update_rate = float(cfg.bias_update_rate)

        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.experts = nn.ModuleList(
            SwiGLU(hidden_size, intermediate_size, use_ternary, residual_scale, init_orthogonal)
            for _ in range(cfg.num_experts)
        )
        self.shared = nn.ModuleList(
            SwiGLU(hidden_size, intermediate_size, use_ternary, residual_scale, init_orthogonal)
            for _ in range(cfg.n_shared)
        )

        # Router stays floating point and rides AdamW: it is a decision function,
        # not a channel weight, and ternarizing a 2->E decision boundary quantizes
        # the decision itself. It is also kept in fp32 under bf16 training — top-k
        # over E logits is a comparison of nearby numbers, and bf16's 7-bit
        # mantissa makes the ordering of two close experts arbitrary.
        self.router = nn.Linear(hidden_size, cfg.num_experts, bias=False)
        std = cfg.router_init_std if cfg.router_init_std > 0 else hidden_size**-0.5
        nn.init.normal_(self.router.weight, mean=0.0, std=std)
        self.router.keep_fp32 = True

        # Controller state. A buffer, not a parameter: it is updated by a rule,
        # never by autograd, and it must survive checkpointing.
        self.register_buffer("expert_bias", torch.zeros(cfg.num_experts))
        self.register_buffer("load_ema", torch.full((cfg.num_experts,), 1.0 / cfg.num_experts))
        self._pending_load: torch.Tensor | None = None
        self.last_router_z_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixer branch output for ``[B, T, H]`` (residual added by the block)."""
        b, t, h = x.shape
        flat = self.norm(x).reshape(b * t, h)

        logits = self.router(flat.float())

        # Selection uses the bias; combining weights do not. Softmax over only the
        # selected logits keeps the weights a proper distribution over the experts
        # that actually run, so the branch output scale is independent of top_k.
        _, idx = (logits + self.expert_bias).topk(self.top_k, dim=-1)
        sel_logits = logits.gather(-1, idx)
        weights = sel_logits.softmax(dim=-1).to(flat.dtype)

        out = self._dispatch(flat, idx, weights)
        for expert in self.shared:
            out = out + expert(flat)

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(idx.reshape(-1), minlength=self.num_experts).float()
                self._pending_load = counts / counts.sum().clamp_min(1.0)
            # z-loss on the router: the log-partition function does not change the
            # routing decision, but an unbounded one makes top-k selection depend
            # on differences between large numbers. Pinning it keeps the decision
            # numerically meaningful.
            self.last_router_z_loss = logits.logsumexp(dim=-1).pow(2).mean()
        else:
            self.last_router_z_loss = None

        return out.view(b, t, h)

    @torch.no_grad()
    def commit_bias_update(self) -> None:
        """Apply one controller step from the last forward's measured load.

        Called once per optimizer step, after backward. Deferring it keeps the
        forward pure, which is required under activation checkpointing: the
        recomputed forward must select the same experts as the original, or the
        recomputed graph does not match the saved one.
        """
        if self._pending_load is None:
            return
        load = self._pending_load.to(self.load_ema)
        self.load_ema.mul_(0.99).add_(load, alpha=0.01)
        if self.bias_update_rate > 0:
            target = 1.0 / self.num_experts
            self.expert_bias.add_(self.bias_update_rate * torch.sign(target - load))
            # Anchor the gauge: only bias *differences* affect top-k, so removing
            # the mean prevents a slow common-mode drift into a range where the
            # bias dominates the logits.
            self.expert_bias.sub_(self.expert_bias.mean())
        self._pending_load = None

    def load_stats(self) -> dict[str, float]:
        """Balance diagnostics: normalized routing entropy and worst-case load.

        ``entropy_ratio`` is ``H(load) / log(E)`` — the usable fraction of the
        expert channels. 1.0 is perfect balance; ``1/E``-ish means collapse.
        """
        load = self.load_ema.float()
        p = load / load.sum().clamp_min(1e-9)
        entropy = -(p * p.clamp_min(1e-9).log()).sum()
        max_entropy = torch.log(torch.tensor(float(self.num_experts)))
        return {
            "entropy_ratio": float(entropy / max_entropy),
            "max_load": float(p.max()),
            "min_load": float(p.min()),
            "bias_span": float(self.expert_bias.max() - self.expert_bias.min()),
        }

    def _dispatch(self, flat: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Sort tokens by expert, run each expert on its contiguous segment.

        Args:
            flat: ``[N, H]`` normalized hidden states.
            idx: ``[N, K]`` selected expert indices.
            weights: ``[N, K]`` combining weights.

        Returns:
            ``[N, H]`` weighted sum of expert outputs.
        """
        n = flat.shape[0]
        out = flat.new_zeros(n, self.hidden_size)
        if n == 0:
            return out

        experts_flat = idx.reshape(-1)
        weights_flat = weights.reshape(-1)
        token_src = torch.arange(n, device=flat.device).repeat_interleave(idx.shape[1])

        order = torch.argsort(experts_flat, stable=True)
        tokens_sorted = token_src[order]
        weights_sorted = weights_flat[order].unsqueeze(-1)
        counts = torch.bincount(experts_flat[order], minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0) - counts

        # One host sync for all segment boundaries rather than 2E per-expert syncs.
        offsets_list = offsets.tolist()
        counts_list = counts.tolist()
        for e in range(self.num_experts):
            count = counts_list[e]
            if count == 0:
                continue
            start = offsets_list[e]
            sel = tokens_sorted[start : start + count]
            expert_out = self.experts[e](flat.index_select(0, sel))
            out.index_add_(0, sel, (expert_out * weights_sorted[start : start + count]).to(out.dtype))
        return out
