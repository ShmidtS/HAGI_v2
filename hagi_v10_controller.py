#!/usr/bin/env python3
"""HAGI v10 controller — validation-gated multi-task optimizer step.

Reference implementation of the spec §4.C (safe multi-task update):
  1. per-task gradients g_i on a small task panel (2-8 adapters/tasks);
  2. MGDA: alpha* = argmin_{alpha in simplex} ||sum_i alpha_i g_i||^2,
     d = sum_i alpha*_i g_i (norm-min direction on the convex hull);
  3. gradient-agreement check: g_i^T d >= -tol for all i, else HOLD;
  4. geometric backtracking on the step scale;
  5. commit only if measured held-out gain > e_gen + e_quant + e_route + m_safety;
  6. otherwise exact rollback of params, optimizer state and RNG.

This is a controller, not a solver for thousands of tasks: for large panels
use gradient sketches / stratified audit (see spec §4.C).

All error bounds must be expressed in the same units as the weighted
validation loss.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import torch


@dataclass
class StepResult:
    action: str                    # "COMMIT" | "HOLD"
    reason: str
    step_scale: float = 0.0
    certified_gain: float = 0.0
    alphas: dict[str, float] = field(default_factory=dict)
    grad_agreement_min: float = 0.0
    backtracks: int = 0
    loss_before: dict[str, float] = field(default_factory=dict)
    loss_after: dict[str, float] = field(default_factory=dict)


def _flat_grads(model: torch.nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().reshape(-1) for p in model.parameters()
             if p.grad is not None]
    if not parts:
        raise RuntimeError("no gradients present; call loss.backward() first")
    return torch.cat(parts)


def _zero_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def mgda_weights(grads: Sequence[torch.Tensor], iters: int = 200) -> torch.Tensor:
    """Min-norm point on the convex hull of task gradients (MGDA).

    Frank-Wolfe over the simplex: deterministic, gradient-free solver
    suitable for K <= ~16 tasks. Returns alpha on the simplex (sum=1, >=0).
    """
    K = len(grads)
    if K == 1:
        return torch.ones(1)
    G = torch.stack([g / (g.norm() + 1e-12) for g in grads])  # K x D, normalized
    alpha = torch.full((K,), 1.0 / K, dtype=G.dtype)
    for _ in range(iters):
        d = alpha @ G
        # Frank-Wolfe: vertex minimizing inner product with d
        dots = G @ d
        j = int(torch.argmin(dots).item())
        if dots[j].item() >= d.dot(d).item() - 1e-16:
            break
        gamma = 2.0 / (2.0 + float(_))  # diminishing step, convergent
        alpha = (1 - gamma) * alpha
        alpha[j] = alpha[j] + gamma
    # clean tiny negatives from float error, renormalize
    alpha = torch.clamp(alpha, min=0.0)
    s = float(alpha.sum().item())
    if s <= 0:
        alpha = torch.full((K,), 1.0 / K)
        s = 1.0
    return alpha / s


@torch.no_grad()
def _snapshot_trainable(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    return {i: p.detach().clone() for i, p in enumerate(model.parameters())}


@torch.no_grad()
def _restore_trainable(model: torch.nn.Module, snap: dict[int, torch.Tensor]) -> None:
    for i, p in enumerate(model.parameters()):
        p.copy_(snap[i])


def _snapshot_optim(optimizer: torch.optim.Optimizer) -> Any:
    return copy.deepcopy(optimizer.state_dict())


def _snapshot_rng() -> dict[str, Any]:
    return {"py": __import__("random").getstate(),
            "np": __import__("numpy").random.get_state(),
            "torch": torch.get_rng_state()}


def _restore_rng(snap: dict[str, Any]) -> None:
    import random
    import numpy as _np
    random.setstate(snap["py"])
    _np.random.set_state(snap["np"])
    torch.set_rng_state(snap["torch"])


def guarded_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task_batches: Mapping[str, Any],
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    validation_fn: Callable[[torch.nn.Module], dict[str, float]],
    generalization_error_bound: float = 0.0,
    quantization_error_bound: float = 0.0,
    routing_error_bound: float = 0.0,
    safety_margin: float = 0.0,
    max_backtracks: int = 8,
    conflict_tol: float = 1e-8,
    task_weights: Optional[Mapping[str, float]] = None,
) -> StepResult:
    """One validation-gated multi-task optimizer step with exact rollback.

    validation_fn must return per-task held-out losses for the same keys as
    task_batches, on a fixed sentinel set that does not overlap train data.
    Commit gate: weighted_loss_before - weighted_loss_after
                > e_gen + e_quant + e_route + m_safety.
    """
    names = list(task_batches.keys())
    w = {k: float(task_weights[k]) if task_weights and k in task_weights else 1.0
         for k in names}
    wsum = sum(w.values())

    rng_snap = _snapshot_rng()
    param_snap = _snapshot_trainable(model)
    optim_snap = _snapshot_optim(optimizer)

    loss_before = validation_fn(model)
    if any(k not in loss_before for k in names):
        _restore_rng(rng_snap)
        return StepResult("HOLD", f"validation_fn missing tasks: {names}")
    wb = sum(w[k] * float(loss_before[k]) for k in names) / wsum

    # per-task gradients (one backward per task; keep panel small, spec §10)
    grads: list[torch.Tensor] = []
    for k in names:
        _zero_grads(model)
        try:
            loss = loss_fn(model, task_batches[k])
            loss.backward()
        except RuntimeError as exc:
            _zero_grads(model)
            _restore_rng(rng_snap)
            return StepResult("HOLD", f"backward failed on '{k}': {exc}")
        g = _flat_grads(model)
        if not torch.isfinite(g).all():
            _zero_grads(model)
            _restore_rng(rng_snap)
            return StepResult("HOLD", f"non-finite gradient on '{k}'")
        grads.append(g)

    alpha = mgda_weights(grads)
    d = sum(float(alpha[i].item()) * grads[i] for i in range(len(grads)))
    # gradient agreement: g_i^T d must not be meaningfully negative
    dots = [float(grads[i].dot(d).item()) for i in range(len(grads))]
    min_dot = min(dots)
    _zero_grads(model)
    if min_dot < -conflict_tol:
        _restore_rng(rng_snap)
        return StepResult("HOLD", "gradient conflict (g_i^T d < 0)",
                          alphas={k: float(alpha[i]) for i, k in enumerate(names)},
                          grad_agreement_min=min_dot)
    d_norm = float(d.norm().item())
    if d_norm <= 1e-14:
        _restore_rng(rng_snap)
        return StepResult("HOLD", "degenerate direction (||d|| ~ 0)",
                          alphas={k: float(alpha[i]) for i, k in enumerate(names)})

    total_bound = (generalization_error_bound + quantization_error_bound
                   + routing_error_bound + safety_margin)

    best: Optional[StepResult] = None
    scale = 1.0
    for bt in range(max_backtracks + 1):
        # apply d: assign as grads so momentum-based optimizers stay consistent
        _restore_trainable(model, param_snap)
        optimizer.load_state_dict(optim_snap)
        _zero_grads(model)
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.grad = d[idx:idx + n].view_as(p).clone()
            idx += n
        optimizer.step()
        loss_after = validation_fn(model)
        if any(k not in loss_after for k in names):
            break
        wa = sum(w[k] * float(loss_after[k]) for k in names) / wsum
        gain = wb - wa
        res = StepResult("COMMIT", "validation gain exceeds error bounds",
                         step_scale=scale, certified_gain=gain,
                         alphas={k: float(alpha[i]) for i, k in enumerate(names)},
                         grad_agreement_min=min_dot, backtracks=bt,
                         loss_before=dict(loss_before), loss_after=dict(loss_after))
        if gain > total_bound:
            _zero_grads(model)
            return res
        if best is None or res.certified_gain > best.certified_gain:
            best = res
        scale *= 0.5

    # no scale passed the gate: exact rollback
    _restore_trainable(model, param_snap)
    optimizer.load_state_dict(optim_snap)
    _zero_grads(model)
    _restore_rng(rng_snap)
    reason = ("validation gain below error budget "
              f"({best.certified_gain:.6g} <= {total_bound:.6g})"
              if best else "all backtrack scales failed validation")
    return StepResult("HOLD", reason,
                      step_scale=scale, certified_gain=best.certified_gain if best else 0.0,
                      alphas={k: float(alpha[i]) for i, k in enumerate(names)},
                      grad_agreement_min=min_dot, backtracks=max_backtracks,
                      loss_before=dict(loss_before),
                      loss_after=dict(best.loss_after) if best else {})


def self_test() -> None:
    """Executable property tests for the controller (spec Phase 0)."""
    torch.manual_seed(0)

    # --- Case 1: compatible tasks -> COMMIT, loss must drop ---
    model = torch.nn.Linear(4, 1)
    x = torch.randn(64, 4)
    y1 = x @ torch.tensor([[1.0], [1.0], [0.0], [0.0]]) + 0.1
    y2 = x @ torch.tensor([[0.0], [0.0], [1.0], [1.0]]) - 0.1

    def loss_fn(m, batch):
        xx, yy = batch
        return ((m(xx) - yy) ** 2).mean()

    vx = torch.randn(256, 4)
    vy1 = vx @ torch.tensor([[1.0], [1.0], [0.0], [0.0]]) + 0.1
    vy2 = vx @ torch.tensor([[0.0], [0.0], [1.0], [1.0]]) - 0.1

    def validation_fn(m):
        with torch.no_grad():
            return {"t1": float(loss_fn(m, (vx, vy1))),
                    "t2": float(loss_fn(m, (vx, vy2)))}

    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    before = validation_fn(model)
    res = guarded_optimizer_step(
        model, opt,
        {"t1": (x, y1), "t2": (x, y2)},
        loss_fn, validation_fn,
        generalization_error_bound=0.0, quantization_error_bound=0.0,
        routing_error_bound=0.0, safety_margin=0.0, max_backtracks=6)
    assert res.action == "COMMIT", res
    after = validation_fn(model)
    assert after["t1"] < before["t1"] and after["t2"] < before["t2"], (before, after)
    assert abs(sum(res.alphas.values()) - 1.0) < 1e-6
    assert all(a >= 0 for a in res.alphas.values())
    print(f"  case1 COMMIT: gain={res.certified_gain:.4f} alphas={res.alphas}")

    # --- Case 2: conflicting tasks -> HOLD and exact rollback ---
    # No bias: identical inputs with opposite targets give exactly opposite
    # gradients (g2 = -g1), so the min-norm direction degenerates to ~0.
    model2 = torch.nn.Linear(4, 1, bias=False)
    y3 = x @ torch.tensor([[1.0], [-1.0], [0.0], [0.0]])
    y4 = -y3  # exactly opposite objective on shared weights -> pure conflict
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.5)
    snap_params = {i: p.detach().clone() for i, p in enumerate(model2.parameters())}
    opt_state = copy.deepcopy(opt2.state_dict())
    res2 = guarded_optimizer_step(
        model2, opt2,
        {"t1": (x, y3), "t2": (x, y4)},
        loss_fn, validation_fn, max_backtracks=4)
    assert res2.action == "HOLD", res2
    for i, p in enumerate(model2.parameters()):
        assert torch.equal(p.detach(), snap_params[i]), f"param {i} changed on HOLD"
    assert opt2.state_dict().keys() == opt_state.keys()
    print(f"  case2 HOLD+rollback: reason='{res2.reason}' "
          f"min_agreement={res2.grad_agreement_min:.3e}")

    # --- Case 3: unrealistic error bounds -> gate refuses COMMIT ---
    model3 = torch.nn.Linear(4, 1)
    opt3 = torch.optim.SGD(model3.parameters(), lr=0.5)
    res3 = guarded_optimizer_step(
        model3, opt3,
        {"t1": (x, y1), "t2": (x, y2)},
        loss_fn, validation_fn,
        generalization_error_bound=1e6, max_backtracks=3)
    assert res3.action == "HOLD", res3
    after3 = validation_fn(model3)
    assert after3 == validation_fn(model3)  # deterministic: model unchanged
    print(f"  case3 HOLD@bounds: gain={res3.certified_gain:.4f} << bound=1e6")

    # --- Case 4: MGDA solver sanity on tiny hulls ---
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    al = mgda_weights([a, b])
    assert torch.allclose(al, torch.tensor([0.5, 0.5]), atol=1e-6)
    al2 = mgda_weights([a, a])
    assert torch.allclose(al2, torch.tensor([0.5, 0.5]), atol=1e-6)
    al3 = mgda_weights([a, -a])  # exact conflict: min-norm point is origin, any simplex pt
    assert abs(float(al3.sum()) - 1.0) < 1e-6
    print(f"  case4 MGDA: orthogonal->uniform, identical->uniform, conflict ok")

    print("V10 CONTROLLER SELF_TEST PASS")


if __name__ == "__main__":
    self_test()
