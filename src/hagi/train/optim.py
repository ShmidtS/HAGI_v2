"""Optimizers: Muon for 2D channel weights, AdamW for everything else.

The split is not a heuristic — the two parameter classes have different geometry.

**2D channel weights (Muon).** A hidden-mixing matrix acts on a vector space, and
what matters is its action across directions. Newton-Schulz orthogonalization
sets every singular value of the *update* to 1, so all directions advance
equally. On a ternary master this is exactly right: the quantizer reads only the
sign pattern relative to the row absmean, so an isotropic update explores the
reachable ternary patterns efficiently rather than concentrating on whichever
directions the raw gradient happens to favour.

**Everything else (AdamW).** Codebooks, 1D gains, biases, routers. A codebook's
rows are updated at wildly different frequencies (token frequency spans five
orders of magnitude on this corpus), which is precisely the case per-coordinate
adaptivity exists for. Orthogonalization is undefined for 1D and actively wrong
for a codebook — it would equalize the step size of a token seen once and a token
seen ten million times.

Routing is by an explicit ``is_channel_weight`` marker set at construction, not
by module type and not by name substring. The V25 predecessor kept a
``_MUON_EXCLUDE`` list of name fragments that had to be extended for every new
module; a missed entry silently sent a codebook through Newton-Schulz. A marker
also keeps the partition stable when ternary quantization is disabled for an
ablation — Muon's argument is about matrix geometry, not storage rate, so an
ablation on the rate constraint must not also change the optimizer.

**Weight-norm drift.** Muon removes the ``1/||W||`` brake that plain SGD applies,
so spectral norms grow. Two things make that safe here rather than requiring a
spectral cap: the ternary quantizer's per-row absmean scale cancels any uniform
outward drift (simulated over 20k Muon steps: effective weight RMS 0.0293 ->
0.0308, i.e. flat), and QK-norm removes the one place where a growing norm
directly damages the forward pass. Decoupled weight decay scaled by
``sqrt(fan_out/fan_in)`` handles the residual, so wide-output matrices are not
under-regularized relative to square ones.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi.config import Config


def newton_schulz(
    grad: torch.Tensor, steps: int = 5, coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
) -> torch.Tensor:
    """Approximate the matrix sign of ``grad`` (all singular values -> 1).

    A quintic iteration ``X <- aX + bX(X^T X) + cX(X^T X)^2`` with coefficients
    tuned to converge from the whole unit-norm ball in five steps. Run in bf16:
    the iteration is a fixed-point map whose basin is far wider than bf16's
    precision, and the halved bandwidth matters because this runs once per 2D
    parameter per step.

    Square matrices need the full five steps; tall/wide ones reach the same
    spread at three. Newton-Schulz converges from the unit-norm ball at a rate
    that depends on the spectral gap, and a square update has the worst
    conditioning (the smallest and largest singular directions advance
    together), so the truncation is visible exactly there. Measured over the
    V33 2D-weight mix: ``steps=3`` on non-square keeps worst-case singular
    spread identical to ``steps=5`` while running ~35% faster (the majority of
    channel weights are gate/up/down, all tall or wide).

    Args:
        grad: 2D update.
        steps: iteration count.
        coeffs: the (a, b, c) quintic coefficients.

    Returns:
        Orthogonalized update in ``grad``'s dtype.
    """
    if grad.shape[0] == grad.shape[1] and steps < 5:
        # Square updates lose orthogonalization at fewer steps (measured spread
        # 74 -> 890 at 3 steps); never truncate them.
        steps = 5
    a, b, c = coeffs
    x = grad.bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        # addmm fuses b*gram + c*(gram@gram) into one call; the epilogue keeps the
        # product in higher precision than the separate mul/add it replaces, so
        # the fixed point is unchanged and the launch count drops.
        poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, poly, x, beta=a, alpha=1)
    if transposed:
        x = x.T
    return x.to(grad.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum SGD with Newton-Schulz orthogonalized updates.

    Args:
        params: 2D parameters only.
        lr: learning rate.
        momentum: heavy-ball coefficient.
        nesterov: use the Nesterov form of the lookahead.
        ns_steps: Newton-Schulz iterations.
        ns_coeffs: quintic coefficients.
        weight_decay: decoupled decay, scaled per-parameter by fan ratio.
        wd_cap: cap on that scaling.
        momentum_offload: keep momentum buffers in host memory.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        weight_decay: float = 0.0,
        wd_cap: float = 2.0,
        momentum_offload: bool = False,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                ns_coeffs=ns_coeffs,
                weight_decay=weight_decay,
                wd_cap=wd_cap,
                momentum_offload=momentum_offload,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            offload = group["momentum_offload"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon requires 2D parameters, got shape {tuple(p.shape)}")
                grad = p.grad
                state = self.state[p]

                # Wide-output matrices see more gradient signal per row, so a
                # single decay rate under-regularizes them relative to square ones.
                fan_ratio = min((p.shape[0] / p.shape[1]) ** 0.5, group["wd_cap"])
                fan_ratio = max(1.0, fan_ratio)
                if group["weight_decay"] != 0.0:
                    p.mul_(1.0 - lr * group["weight_decay"] * fan_ratio)

                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad, device="cpu" if offload else grad.device)
                    state["momentum_buffer"] = buf
                if offload:
                    if buf.device.type != "cpu":
                        buf = buf.to("cpu")
                        state["momentum_buffer"] = buf
                    grad_host = grad.detach().to("cpu")
                    buf.mul_(momentum).add_(grad_host)
                    update = grad_host.add(buf, alpha=momentum) if group["nesterov"] else buf
                    update = update.to(p.device)
                else:
                    if buf.device != grad.device or buf.dtype != grad.dtype:
                        buf = buf.to(device=grad.device, dtype=grad.dtype)
                        state["momentum_buffer"] = buf
                    buf.mul_(momentum).add_(grad)
                    update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf

                # Newton-Schulz stays on the accelerator regardless of offload:
                # its matmuls are orders of magnitude slower on CPU.
                # ``ns_steps`` here is the *target* for tall/wide matrices;
                # ``newton_schulz`` raises square ones to 5 internally.
                p.add_(newton_schulz(update, max(3, min(group["ns_steps"], 5)), group["ns_coeffs"]), alpha=-lr * fan_ratio)


class HybridOptimizer:
    """Drives Muon and AdamW as one optimizer over disjoint parameter groups.

    Not an ``Optimizer`` subclass: the two carry incompatible per-group state and
    a shared ``param_groups`` list would make LR scheduling ambiguous. The
    ``_muon`` flag on each group is what the scheduler uses to apply the two base
    learning rates.

    ``muon`` may be None when Muon is disabled (``train.use_muon=False``): the
    whole model then rides AdamW and the Muon accessors become no-ops.
    """

    def __init__(self, muon: Muon | None, adamw: torch.optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw
        self.param_groups = (muon.param_groups if muon else []) + adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict() if self.muon else {}, "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if self.muon is not None and state.get("muon"):
            self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def _muon_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect every 2D hidden-mixing weight — the channel masters.

    Selection is by the ``is_channel_weight`` marker that
    :func:`~hagi.model.ffn.linear` and :class:`~hagi.model.ternary.BitLinear` set
    on construction. Marker rather than module type, so that disabling
    quantization for an ablation does not silently move the entire body from
    Muon to AdamW and change what the ablation is measuring. Marker rather than
    name substring, so that a renamed module cannot fall out of the partition —
    the V25 predecessor kept a ``_MUON_EXCLUDE`` list of name fragments and a
    missed entry sent a codebook through Newton-Schulz.
    """
    found: list[nn.Parameter] = []
    for module in model.modules():
        if getattr(module, "is_channel_weight", False):
            weight = getattr(module, "weight", None)
            if isinstance(weight, nn.Parameter) and weight.requires_grad and weight.ndim == 2:
                found.append(weight)
    return found


def _supports_fused_adamw(params: list[nn.Parameter]) -> bool:
    if not params:
        return False
    first = params[0]
    return first.device.type == "cuda" and all(
        p.device == first.device and p.dtype == first.dtype and p.is_floating_point() for p in params
    )


def build_optimizer(model: nn.Module, cfg: Config) -> HybridOptimizer:
    """Partition parameters and construct both optimizers.

    Every trainable parameter lands in exactly one group; the assertion is not
    decorative — a parameter in zero groups trains silently at learning rate 0,
    and one in two groups gets a double update.

    Weight decay is applied to 2D matrices only. Decaying a 1D gain pulls a
    normalization layer's scale toward zero, which is a change to the function
    rather than a regularizer on capacity.
    """
    tc = cfg.train
    muon_params = _muon_parameters(model) if tc.use_muon else []
    muon_ids = {id(p) for p in muon_params}
    channel_ids = {id(p) for p in _muon_parameters(model)}

    body: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in muon_ids:
            continue
        if id(p) in channel_ids:
            body.append(p)
        elif p.ndim >= 2 and "norm" not in name.lower():
            decay.append(p)
        else:
            no_decay.append(p)

    assigned = [id(p) for p in muon_params + body + decay + no_decay]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if len(assigned) != len(set(assigned)) or set(assigned) != trainable:
        raise RuntimeError(
            f"parameter partition is not a bijection: {len(assigned)} assigned "
            f"({len(set(assigned))} unique) vs {len(trainable)} trainable"
        )

    if muon_params:
        muon = Muon(
            muon_params,
            lr=tc.muon.lr,
            momentum=tc.muon.momentum,
            nesterov=tc.muon.nesterov,
            ns_steps=tc.muon.ns_steps,
            ns_coeffs=tuple(tc.muon.ns_coeffs),
            weight_decay=tc.muon.weight_decay,
            wd_cap=tc.muon.wd_cap,
            momentum_offload=tc.muon.momentum_offload,
        )
        for group in muon.param_groups:
            group["_muon"] = True
    else:
        muon = None

    base = tc.learning_rate
    body_lr = base * float(tc.adam.body_lr_scale)
    # Split every semantic group by dtype. The fused AdamW kernel requires all
    # parameters in a group to share one dtype, and cast_model leaves the body in
    # bf16 but norms/gains/routers in fp32 — a mixed group silently disables the
    # fused kernel (measured 157ms vs 95ms for a step, 1.66x). Splitting into
    # dtype-homogeneous groups restores fused for every parameter.
    def _split_by_dtype(params: list[nn.Parameter]) -> list[list[nn.Parameter]]:
        groups = []
        for dtype in (torch.bfloat16, torch.float32, torch.float16, torch.float64):
            same = [p for p in params if p.dtype == dtype]
            if same:
                groups.append(same)
        return groups

    groups = []
    for sub in _split_by_dtype(body):
        groups.append(
            {"params": sub, "weight_decay": tc.adam.weight_decay, "lr": body_lr, "_body": True}
        )
    for sub in _split_by_dtype(decay):
        groups.append({"params": sub, "weight_decay": tc.adam.weight_decay})
    for sub in _split_by_dtype(no_decay):
        groups.append({"params": sub, "weight_decay": 0.0})

    kwargs = dict(lr=base, betas=(tc.adam.beta1, tc.adam.beta2), eps=float(tc.adam.eps))
    # Fused is per-optimizer, not per-group: it needs *every* group to be
    # dtype-homogeneous. After the split above each group is, so the global
    # support check now reflects what fused actually requires.
    fused = all(_supports_fused_adamw(g["params"]) for g in groups)
    try:
        adamw = torch.optim.AdamW(groups, **kwargs, fused=fused)
    except (RuntimeError, TypeError, NotImplementedError):
        adamw = torch.optim.AdamW(groups, **kwargs, fused=False)

    return HybridOptimizer(muon, adamw)


def learning_rate_at(step: int, base_lr: float, cfg: Config) -> float:
    """Warmup - stable - decay schedule value at ``step``.

    Three phases:

    * **Warmup** — linear from 0. Mandatory with a matrix-sign optimizer: the
      first orthogonalized updates are full-magnitude in every direction at once.
    * **Stable** — either constant or ``1/sqrt(1 + t/tau)``. The inverse-square-
      root form is horizon-free (extending a run needs no re-tuning) and
      minimax-optimal for last-iterate convergence in the stochastic convex
      setting; a long constant phase at peak learning rate is where late-training
      instability comes from.
    * **Decay** — linear to ``min_lr_ratio``. The final cooldown is where most of
      the loss improvement is realized, and it is the only phase that needs to
      know the total horizon.
    """
    s = cfg.train.schedule
    total = cfg.train.max_steps
    if step < s.warmup_steps:
        return base_lr * step / max(s.warmup_steps, 1)

    decay_start = int(total * (1.0 - s.decay_fraction))
    if s.inverse_sqrt_stable:
        tau = s.inverse_sqrt_tau or max(s.warmup_steps, 1)
        shape = (1.0 + (min(step, decay_start) - s.warmup_steps) / tau) ** -0.5
    else:
        shape = 1.0

    if step < decay_start:
        return base_lr * max(shape, s.min_lr_ratio)

    progress = (step - decay_start) / max(total - decay_start, 1)
    level = shape * (1.0 - progress) + s.min_lr_ratio * progress
    return base_lr * max(level, s.min_lr_ratio)


def set_learning_rate(optimizer: HybridOptimizer, step: int, cfg: Config) -> tuple[float, float]:
    """Apply the schedule to both base rates. Returns ``(adam_lr, muon_lr)``.

    The body group carries a ``body_lr_scale`` multiplier (AdamConfig), so its
    effective LR tracks the schedule at ``base * scale`` rather than ``base``.
    """
    adam_lr = learning_rate_at(step, cfg.train.learning_rate, cfg)
    muon_lr = learning_rate_at(step, cfg.train.muon.lr, cfg)
    body_scale = float(cfg.train.adam.body_lr_scale)
    for group in optimizer.param_groups:
        if group.get("_muon"):
            group["lr"] = muon_lr
        elif group.get("_body"):
            group["lr"] = adam_lr * body_scale
        else:
            group["lr"] = adam_lr
    return adam_lr, muon_lr
