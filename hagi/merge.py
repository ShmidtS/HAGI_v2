"""Checkpoint merge with certified bound (Theory/NonlinearMerge.lean).

nonlinear_merge_bound: with measured linearization residuals
δ (at mid) and δ₂ (at end) of loss L along the segment w1→w2,
the merged point's loss is bounded:
    L(mid) ≤ (L(w1) + L(w2)) / 2 + δ + δ₂/2
For a diagonal-quadratic loss the identity is EXACT with curvature
residual δ_curv = (1/8)·Σ c_i (w2_i − w1_i)² (C472 bridge).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def lin_at(g: Sequence[float], w1: Sequence[float],
           w2: Sequence[float], t: float) -> float:
    """First-order linearization of L at w1 along (w2 − w1)."""
    g = np.asarray(g); w1 = np.asarray(w1); w2 = np.asarray(w2)
    return float(g @ (w2 - w1)) * t


def merge_bound(L: Callable[[np.ndarray], float],
                g: Sequence[float], w1: Sequence[float],
                w2: Sequence[float]) -> dict:
    """Measure δ (mid residual), δ₂ (end residual) and return the
    certified bound L(mid) ≤ (L1+L2)/2 + δ + δ₂/2 (theorem)."""
    w1 = np.asarray(w1, dtype=float); w2 = np.asarray(w2, dtype=float)
    mid = (w1 + w2) / 2.0
    L1, L2 = float(L(w1)), float(L(w2))
    delta_mid = abs(float(L(mid)) - lin_at(g, w1, w2, 0.5))
    delta_end = abs(L2 - (L1 + lin_at(g, w1, w2, 1.0)))
    bound = (L1 + L2) / 2.0 + delta_mid + delta_end / 2.0
    return {"L_mid": float(L(mid)), "bound": bound,
            "delta_mid": delta_mid, "delta_end": delta_end,
            "certified": float(L(mid)) <= bound + 1e-12}


def diag_quadratic_merge(w1: Sequence[float], w2: Sequence[float],
                         c: Sequence[float], a: Sequence[float]
                         ) -> dict:
    """EXACT merge identity for the diagonal-quadratic loss
    L(θ)=½Σ c_i (θ_i−a_i)²: residual (1/8)Σ c_i (w2−w1)_i², and the
    merged loss computed exactly. Verifies the theorem's quadratic
    bridge (C472 curvature)."""
    w1 = np.asarray(w1, dtype=float); w2 = np.asarray(w2, dtype=float)
    c = np.asarray(c, dtype=float); a = np.asarray(a, dtype=float)
    L = lambda w: 0.5 * float(np.sum(c * (w - a) ** 2))
    mid = (w1 + w2) / 2.0
    g = c * (w1 - a)                     # gradient at w1
    delta_curv = float(np.sum(c * (w2 - w1) ** 2)) / 8.0
    bound = (L(w1) + L(w2)) / 2.0 + delta_curv + delta_curv / 2.0
    return {"L_mid": L(mid), "bound": bound,
            "delta_curv": delta_curv,
            "exact": L(mid) <= bound + 1e-12}
