"""Muon + AdamW hybrid optimizer — TYPE-based routing.

Muon (Newton-Schulz orthogonalization + scale-aware weight decay) for the 2D
ternary channel weights (every ``BitLinear.weight``); AdamW for everything
else (embeddings, 1D gates, FP linears, norms, routers, IB/multimodal source
codebooks). V27 routes by module TYPE, not by name-substring matching — this
captures exactly the ternary body (including MoE expert bodies) with zero
string matching and no manual exclusion list to maintain (the V25 fragility:
``_MUON_EXCLUDE`` had to be extended for every new module).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from hagi.config import Config
from hagi.model.ternary import BitLinear


def newton_schulz5(G: torch.Tensor, steps: int = 5, coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)) -> torch.Tensor:
    """Newton-Schulz quintic iteration approximating G(G^T G)^{-1/2}.

    Orthogonalizes a 2D matrix so its singular values become 1.
    """
    a, b, c = coeffs
    x = G.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x.to(G.dtype)


class Muon(Optimizer):
    """Momentum SGD with per-step Newton-Schulz orthogonalization.

    Scale-aware weight decay: ``wd_eff = wd * min(max(1, sqrt(fan_out/fan_in)), wd_cap)``.
    Bounds ``||W||`` — the root fix for the train_v1_1 step-~10000 divergence.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        wd_cap: float = 2.0,
        momentum_offload: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                weight_decay=weight_decay, ns_coeffs=ns_coeffs, wd_cap=wd_cap,
                momentum_offload=momentum_offload,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group.get("weight_decay", 0.0)
            ns_coeffs = group["ns_coeffs"]
            wd_cap = group["wd_cap"]
            offload = group["momentum_offload"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                fan_ratio = min(max(1.0, p.size(0) / p.size(1)) ** 0.5, wd_cap)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd * fan_ratio)
                if offload:
                    # Offload path: momentum on CPU (frees ~16 GiB VRAM for the
                    # 9.37B run). On Strix Halo APU the GPU->CPU "transfer" is a
                    # cache-coherence flush, not a PCIe DMA. NS5 stays on GPU:
                    # its matmuls are 100-1000x slower on CPU. See V28 OOM notes.
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g, device="cpu")
                    buf = state["momentum_buffer"]
                    # Normalize device on resume: an older checkpoint may have
                    # saved momentum on GPU (pre-offload). Move it to CPU once.
                    if buf.device.type != "cpu":
                        buf = buf.to("cpu")
                        state["momentum_buffer"] = buf
                    g_cpu = g.detach().to("cpu")
                    buf.mul_(momentum).add_(g_cpu)
                    update_cpu = g_cpu.add(buf, alpha=momentum) if nesterov else buf
                    update = newton_schulz5(update_cpu.to(p.device), ns_steps, ns_coeffs)
                else:
                    # On-device path: keep momentum on the same device as p. No
                    # per-param host sync — critical on launch-bound small models
                    # where copy_/to 22% CPU stalls the dispatch pipeline and
                    # starves the GPU (ProfilerStep* ~37% self CUDA = idle).
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    if buf.device != p.device or buf.dtype != g.dtype:
                        buf = buf.to(device=p.device, dtype=g.dtype)
                        state["momentum_buffer"] = buf
                    buf.mul_(momentum).add_(g)
                    update = g.add(buf, alpha=momentum) if nesterov else buf
                    update = newton_schulz5(update, ns_steps, ns_coeffs)
                p.add_(update.reshape(p.shape), alpha=-lr * fan_ratio)


class CombinedOptimizer:
    """Steps Muon and AdamW together with a unified interface.

    Deliberately does not inherit from torch.optim.Optimizer because the two
    sub-optimizers manage disjoint parameter groups with different update rules.
    """

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW):
        self.muon = muon
        self.adamw = adamw
        self.param_groups = muon.param_groups + adamw.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> float | None:
        self.muon.step()
        self.adamw.step()
        return None

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def _is_bitlinear_weight(module: nn.Module, name: str) -> bool:
    """True if ``module`` is a BitLinear and ``name`` is its 'weight' attribute.

    Type-based routing: only the 2D ternary channel masters ride Muon. This is
    robust to module naming (no substring matching) and automatically covers
    MoE expert bodies, the attention projections, and the FFN bilinear weights.
    """
    return isinstance(module, BitLinear) and name == "weight"


def _muon_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect every BitLinear.weight parameter (the ternary channel body)."""
    params: list[nn.Parameter] = []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if _is_bitlinear_weight(module, param_name) and param.requires_grad:
                params.append(param)
    return params


def _can_attempt_fused_adamw(params: list[nn.Parameter]) -> bool:
    return bool(params) and all(
        param.device == params[0].device and param.dtype == params[0].dtype and param.is_floating_point()
        for param in params
    )


def supports_fused_adamw(params: list[nn.Parameter]) -> bool:
    return _can_attempt_fused_adamw(params) and params[0].device.type == "cuda"


def _is_expected_fused_adamw_error(error: Exception) -> bool:
    message = str(error).lower()
    if "unexpected keyword argument 'fused'" in message or 'unexpected keyword argument "fused"' in message:
        return True
    return any(
        pattern in message
        for pattern in (
            "fused adamw is not supported for this device", "fused adamw is not supported for this dtype",
            "fused adamw is not supported on this device", "fused adamw is not supported on this dtype",
            "fused is not supported for this device", "fused is not supported for this dtype",
            "fused is not supported on this device", "fused is not supported on this dtype",
        )
    )


def build_adamw(
    param_groups: list[dict],
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build AdamW, probing fused support without touching model parameters."""
    params = [param for group in param_groups for param in group["params"]]
    kwargs = {"lr": lr, "betas": (beta1, beta2), "eps": eps}
    if not supports_fused_adamw(params):
        return torch.optim.AdamW(param_groups, **kwargs, fused=False)
    probe_param = nn.Parameter(torch.zeros(1, device=params[0].device, dtype=params[0].dtype))
    probe_param.grad = torch.ones_like(probe_param)
    try:
        probe = torch.optim.AdamW([{"params": [probe_param], "weight_decay": 0.0}], **kwargs, fused=True)
        probe.step()
    except (RuntimeError, TypeError, NotImplementedError) as error:
        if not _is_expected_fused_adamw_error(error):
            raise
        return torch.optim.AdamW(param_groups, **kwargs, fused=False)
    return torch.optim.AdamW(param_groups, **kwargs, fused=True)


def build_optimizer(model: nn.Module, cfg: Config) -> CombinedOptimizer:
    """Build Muon (2D ternary channel weights) + AdamW (everything else).

    Routing is by type: BitLinear.weight -> Muon; all other trainable params ->
    AdamW. Every trainable parameter appears in exactly one group (verified).
    """
    tc = cfg.train
    muon_params = _muon_params(model)
    muon_ids = {id(p) for p in muon_params}
    rest = [(n, p) for n, p in model.named_parameters() if p.requires_grad and id(p) not in muon_ids]
    decay = [p for n, p in rest if p.ndim >= 2 and "norm" not in n.lower()]
    no_decay = [p for n, p in rest if not (p.ndim >= 2 and "norm" not in n.lower())]
    optimized_ids = [id(p) for p in muon_params + decay + no_decay]
    if len(optimized_ids) != len(set(optimized_ids)) or set(optimized_ids) != {id(p) for _, p in model.named_parameters() if p.requires_grad}:
        raise RuntimeError("every trainable parameter must appear in exactly one optimizer group")

    muon = Muon(
        muon_params,
        lr=tc.muon.lr, momentum=tc.muon.momentum, nesterov=tc.muon.nesterov,
        ns_steps=tc.muon.ns_steps, weight_decay=tc.muon.weight_decay,
        ns_coeffs=tc.muon.ns_coeffs, wd_cap=tc.muon.wd_cap,
        momentum_offload=tc.muon.momentum_offload,
    )
    muon.param_groups[0]["_muon"] = True
    adamw = build_adamw(
        [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tc.learning_rate, beta1=tc.adam.beta1, beta2=tc.adam.beta2, eps=tc.adam.eps,
    )
    return CombinedOptimizer(muon, adamw)
