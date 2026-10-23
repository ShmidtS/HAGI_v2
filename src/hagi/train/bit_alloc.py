"""Bit allocation across tensors/layers (Budget/BitAlloc.lean).

Ports the marginal-exchange theory: the error of layer i at bit depth
f_i is c_i / 2^f_i; moving ONE bit j -> k changes total error by the
exact exchange price; equilibrium is a factor-2 balance; the inner
optimum is water-filling. Small pure module — allocation decisions for
quantization/distillation budgets.
"""
from __future__ import annotations

import math


def layer_error(c: float, bits: float) -> float:
    """layerError: error of a layer with difficulty c at bit depth b
    is c / 2^b (the dyadic quantization model of BitAlloc.lean)."""
    if bits < 0:
        raise ValueError("bits must be nonnegative")
    if c < 0.0:
        raise ValueError("c must be nonnegative")
    return c / 2.0 ** bits


def transfer_exact(
    c_j: float, f_j: float, c_k: float, f_k: float
) -> float:
    """Exact price of moving one bit j -> k [``transfer_exact``]:
    delta_total = (c_k / 2^{f_k}) - (c_j / 2^{f_j}) / 2 — the gain
    doubled speed of the dyadic grid. Negative = beneficial.
    """
    return layer_error(c_k, f_k) - layer_error(c_j, f_j - 1)


def imbalance_yields_gain(c_j: float, f_j: float, c_k: float, f_k: float) -> bool:
    """Exchange j -> k is beneficial iff 2*err_k < err_j
    [``imbalance_yields_gain`` — the factor-2 rule]."""
    return 2.0 * layer_error(c_k, f_k) < layer_error(c_j, f_j)


def stable_factor_two(errors: list[float]) -> bool:
    """Equilibrium certificate [``stable_factor_two``]: allocation is
    factor-2 balanced when max(err) <= 2 * min(err) over layers —
    no single-bit exchange can help."""
    if not errors:
        raise ValueError("empty error list")
    return max(errors) <= 2.0 * min(errors)


def water_filling_marginals(c: list[float], B: float) -> list[float]:
    """Inner optimum [``water_filling_inner_optimum``]: equalize the
    marginals c_i / 2^{f_i} = lambda over active layers under the bit
    budget sum(f) <= B; layers with c_i <= lambda get 0 bits (the
    clamped corner). Returns the bit allocation. Continuous relaxation
    (integer rounding is the caller's policy), comme in KVWater/RankBudget.
    """
    if not c or any(ci < 0 for ci in c):
        raise ValueError("c must be nonempty nonnegative")
    if B < 0:
        raise ValueError("budget must be nonnegative")
    n = len(c)
    if max(c) == 0.0:
        return [0.0] * n
    lo, hi = 0.0, max(c)
    for _ in range(200):
        mid = max(hi * 0.5, math.sqrt(max(lo, 1e-300) * hi))
        s = sum(max(0.0, math.log2(ci / mid)) for ci in c if ci > 0)
        if s > B:
            lo = mid
        else:
            hi = mid
    lam = hi
    return [max(0.0, math.log2(ci / lam)) if ci > 0 else 0.0 for ci in c]
