"""Generalization telemetry (Generalization/ModeState.lean).

Qgen five-probe scalarization, the Lyapunov object
hagiPotential = energy + lam·protectedRisk + nu·max(0, Qtarget − Qgen),
the mode-drop rejection gate, and Pareto checkpoint selection.

All probe values are measured by the caller (h_emp_G discipline:
this module owns containers and certified logic only, never
measurements). The Pareto ⟹ ∃weights direction needs convexity and
is not claimed — only the proved direction (strictly positive
weights ⇒ scalarized argmin is Pareto-optimal).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_PROBES = 5


@dataclass(frozen=True)
class GenMetrics:
    """Five measured generalization probes (h_emp_G, lower = better
    energy semantics live in `energy`; probes are higher = better)."""

    q1: float
    q2: float
    q3: float
    q4: float
    q5: float

    def vec(self) -> np.ndarray:
        return np.array([self.q1, self.q2, self.q3, self.q4, self.q5])


def qgen(metrics: GenMetrics, weights: np.ndarray) -> float:
    """Qgen = Σ wᵢ·qᵢ, w ≥ 0 (nonnegative scalarization)."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (N_PROBES,) or np.any(w < 0):
        raise ValueError("weights must be a nonnegative length-5 vector")
    return float(np.dot(w, metrics.vec()))


def hagi_potential(
    energy: float,
    protected_risk: float,
    metrics: GenMetrics,
    weights: np.ndarray,
    *,
    lam: float,
    nu: float,
    q_target: float,
) -> float:
    """Φ_HAGI = energy + lam·protectedRisk + nu·max(0, Qtarget − Qgen)."""
    if lam < 0 or nu < 0:
        raise ValueError("lam, nu must be nonnegative")
    q = qgen(metrics, weights)
    return energy + lam * protected_risk + nu * max(0.0, q_target - q)


def generalization_safe_step(
    fit: float,
    *,
    lam: float,
    budget: float,
    nu: float,
    eps_q: float,
) -> bool:
    """`generalization_safe_step`: the step is certified when the fit
    covers lam·budget and nu·ε_Q — Φ decreases by
    fit − lam·budget − nu·ε_Q > 0."""
    return fit - lam * budget - nu * eps_q > 0.0


def mode_drop_gate(
    metrics_before: GenMetrics,
    metrics_after: GenMetrics,
    weights: np.ndarray,
    *,
    tol: float,
) -> tuple[bool, float]:
    """`modeDrop_rejected` contrapositive as a runtime gate.

    Returns (accept, q_drop). A step that improves training fit but
    drops Qgen by more than the certified probe tolerance is REJECTED:
    a certified tolerance strictly below the drop contradicts
    "energy improves while Qgen drops" (ModeDrop)."""
    if tol < 0:
        raise ValueError("tol must be nonnegative")
    q_drop = qgen(metrics_before, weights) - qgen(metrics_after, weights)
    return (q_drop <= tol, q_drop)


def _dominates(a: tuple[GenMetrics, float], b: tuple[GenMetrics, float]) -> bool:
    """Strict domination in the joint (energy ↓, probes ↑) space."""
    ma, ea = a
    mb, eb = b
    va, vb = ma.vec(), mb.vec()
    ge = bool(np.all(va >= vb)) and ea <= eb
    gt = bool(np.any(va > vb)) or ea < eb
    return ge and gt


def pareto_select(
    candidates: list[tuple[GenMetrics, float]],  # (probes, energy)
    weights: np.ndarray,
) -> int:
    """`scalarized_argmin_pareto` / `selector_skips_dominated`.

    Selects the checkpoint minimizing Φ_HAGI (energy + nu-flame is
    folded into the caller's energy/probes; here: scalarize probes
    and energy with strictly positive weights). Never returns a
    strictly dominated checkpoint (proved direction only)."""
    if not candidates:
        raise ValueError("no candidates")
    w = np.asarray(weights, dtype=float)
    if np.any(w <= 0):
        raise ValueError("Pareto guarantee needs strictly positive weights")
    # objective: energy + Qgen shortfall handled by caller via energy;
    # here scalarized score = energy − Qgen (both smaller-is-better).
    scores = [energy - qgen(m, w) for m, energy in candidates]
    best = int(np.argmin(scores))
    best_pair = (candidates[best][0], candidates[best][1])
    for c in candidates:
        if c is candidates[best]:
            continue
        if _dominates((c[0], c[1]), best_pair):
            raise AssertionError(
                "scalarized argmin returned a dominated checkpoint"
            )
    return best
