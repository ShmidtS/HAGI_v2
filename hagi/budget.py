"""Bit-budget allocation (C428 weightedExpAllocation + C472 FisherQuant).

Closed-form optimal allocation of an exponential-family budget:
maximize Σ c_i·exp(−κ_i·r_i) under Σ w_i·r_i = B.
With Fisher weights (κ_i = curvature c_j from Gershgorin column mass),
coordinates with larger curvature receive MORE bits (fisher_alloc_mono).
"""
from __future__ import annotations

import math
from typing import Sequence


def weighted_exp_budget(kappa: Sequence[float],
                        w: Sequence[float]) -> float:
    return sum(wi / ki for wi, ki in zip(w, kappa))


def weighted_exp_log_coeff(c: Sequence[float], kappa: Sequence[float],
                           w: Sequence[float]) -> float:
    return sum((wi / ki) * math.log(ci * ki / wi)
               for ci, ki, wi in zip(c, kappa, w))


def weighted_exp_allocation(c: Sequence[float], kappa: Sequence[float],
                            w: Sequence[float], B: float) -> list[float]:
    """Exact closed form (Budget/WeightedExpAlloc.lean, C428):
    alloc_i = (log(c_i·κ_i/w_i) − (logCoeff − B)/budget) / κ_i."""
    if not (len(c) == len(kappa) == len(w)) or not c:
        raise ValueError("c/kappa/w must be same non-empty length")
    if any(k <= 0 for k in kappa) or any(wi <= 0 for wi in w) \
            or any(ci <= 0 for ci in c):
        raise ValueError("all c, kappa, w must be positive")
    budget = weighted_exp_budget(kappa, w)
    coeff = weighted_exp_log_coeff(c, kappa, w)
    shift = (coeff - B) / budget
    return [(math.log(ci * ki / wi) - shift) / ki
            for ci, ki, wi in zip(c, kappa, w)]


def fisher_tolerances(col_curv: Sequence[float], B: float,
                      w: Sequence[float] | None = None) -> list[float]:
    """C472 bridge: per-coordinate quantization tolerance ε_j from the
    Gershgorin column curvature c_j (hess_le_colCurv). Flat coordinates
    (small c_j) tolerate more error — tolerance is MONOTONE decreasing
    in curvature (fisher_tolerance_mono)."""
    n = len(col_curv)
    ww = w if w is not None else [1.0] * n
    if any(cj <= 0 for cj in col_curv):
        raise ValueError("column curvature must be positive")
    # objective weights c_i = 1 (uniform loss mass), κ_i = c_j
    return weighted_exp_allocation([1.0] * n, list(col_curv), ww, B)
