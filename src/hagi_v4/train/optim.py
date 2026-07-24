"""Muon + AdamW hybrid optimizer for HAGI V4.

Same architecture as V1/V3: Muon for 2D weights (Newton-Schulz
orthogonalization + scale-aware weight decay), AdamW for embeddings,
1D params, norms, gates, routers.

Section 7.5 of ARCHITECTURE_V4.md.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from hagi_v4.config import HAGIv4Config


def newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz quintic iteration approximating G(G^T G)^{-1/2}.

    Orthogonalizes a 2D matrix so its singular values become 1.
    Coefficients (a, b, c) = (3.4445, -4.7750, 2.0315).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
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

    Scale-aware weight decay: wd_eff = wd * min(max(1, sqrt(fan_out/fan_in)), 2.0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd * min(max(1.0, p.size(0) / p.size(1)) ** 0.5, 2.0))
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = newton_schulz5(update, ns_steps)
                p.add_(update.reshape(p.shape), alpha=-lr * min(max(1.0, p.size(0) / p.size(1)) ** 0.5, 2.0))


class CombinedOptimizer:
    """Steps Muon and AdamW together with a unified interface.

    Deliberately does not inherit from torch.optim.Optimizer because the two
    sub-optimizers manage disjoint parameter groups with different update rules.
    Implements step, zero_grad, state_dict, and load_state_dict for compatibility
    with checkpoint save/load and training loops.
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
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


_MUON_EXCLUDE = frozenset(
    {
        "embed",
        "lm_compress",
        "lm_expand",
        "token_compress",
        "token_expand",
        "norm",
        "router",
        "gate",
        # Rate-critical / source-codebook FP32 masters flow to AdamW. The
        # ternary 2D body masters (qkv/out_proj/A0/A1/W/rate_up/predictor/
        # update_proj/update_out/hep_feedback) are NOT here — they are 2D and
        # ride Muon. Only the bottleneck KL/decoder linears, the multimodal
        # source encoders, and the learned-uncertainty head stay FP32.
        "to_mu",
        "to_logvar",
        "decompress",
        "image_embed",
        "audio_embed",
        "shared_down",
        "shared_up",
        "log_var",
    }
)


def is_muon_param(name: str, param: nn.Parameter) -> bool:
    """True if param should use Muon (2D weight, not in exclude list).

    Uses exact word matching on dot/underscore-separated segments to avoid
    false positives (e.g. 'gate' matching 'aggregate', 'norm' matching 'transform').
    """
    if param.ndim != 2:
        return False
    for seg in name.lower().split("."):
        if seg in _MUON_EXCLUDE:
            return False
        for part in seg.split("_"):
            if part in _MUON_EXCLUDE:
                return False
    return True


def _can_attempt_fused_adamw(params: list[nn.Parameter]) -> bool:
    """Return whether a fused AdamW probe is meaningful for this parameter group."""
    return bool(params) and all(
        param.device == params[0].device and param.dtype == params[0].dtype and param.is_floating_point()
        for param in params
    )


def supports_fused_adamw(params: list[nn.Parameter]) -> bool:
    """Return whether parameters meet the legacy fused CUDA preconditions."""
    return _can_attempt_fused_adamw(params) and params[0].device.type == "cuda"


def _is_expected_fused_adamw_error(error: Exception) -> bool:
    message = str(error).lower()
    if "unexpected keyword argument 'fused'" in message or 'unexpected keyword argument "fused"' in message:
        return True
    return any(
        pattern in message
        for pattern in (
            "fused adamw is not supported for this device",
            "fused adamw is not supported for this dtype",
            "fused adamw is not supported on this device",
            "fused adamw is not supported on this dtype",
            "fused is not supported for this device",
            "fused is not supported for this dtype",
            "fused is not supported on this device",
            "fused is not supported on this dtype",
        )
    )


def build_adamw(param_groups: list[dict], *, lr: float) -> torch.optim.AdamW:
    """Build AdamW, probing fused support without touching model parameters."""
    params = [param for group in param_groups for param in group["params"]]
    kwargs = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
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


def build_optimizer(model: nn.Module, cfg: HAGIv4Config) -> CombinedOptimizer:
    """Build Muon (2D hidden weights) + AdamW (everything else)."""
    tc = cfg.train
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    muon_params = [p for n, p in named if is_muon_param(n, p)]
    rest = [(n, p) for n, p in named if not is_muon_param(n, p)]
    decay = [p for n, p in rest if p.ndim >= 2 and "norm" not in n.lower()]
    no_decay = [p for n, p in rest if not (p.ndim >= 2 and "norm" not in n.lower())]
    optimized_ids = [id(p) for p in muon_params + decay + no_decay]
    if len(optimized_ids) != len(set(optimized_ids)) or set(optimized_ids) != {id(p) for _, p in named}:
        raise RuntimeError("every trainable parameter must appear in exactly one optimizer group")

    muon = Muon(
        muon_params,
        lr=tc.muon_lr,
        momentum=tc.muon_momentum,
        weight_decay=tc.muon_weight_decay,
    )
    muon.param_groups[0]["_muon"] = True
    adamw = build_adamw(
        [
            {"params": decay, "weight_decay": tc.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tc.learning_rate,
    )
    return CombinedOptimizer(muon, adamw)
