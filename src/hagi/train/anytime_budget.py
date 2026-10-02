"""Anytime-valid confidence budgets: one delta for an infinite horizon.

``Hagi/.../R93``. Ville's inequality for non-negative supermartingales
says that for a supermartingale ``M_t`` and any level ``alpha``,

    Pr[exists t <= T : M_t >= alpha] <= alpha / E[M_0]

so spending ``delta_t = alpha_t / E[M_0]`` at step ``t`` bounds the
probability that ANY certificate up to T ever fails. The bound holds
UNIFORMLY in T -- that is the "anytime" part, and it is the reason the
budget is not ``delta / T`` per step.

The geometric schedule ``delta_t = delta_0 * rho^t`` is admissible when

    delta_0 / (1 - rho) <= delta,

i.e. the geometric series sums to the total budget. With that choice

    Pr[forall t <= T, every certificate is valid] >= 1 - delta

for ALL T simultaneously, from ONE global delta -- not ``delta`` per
step, which over T steps would need ``T * delta`` and becomes vacuous.

**Why this matters for the controller.** ``safeQP``/``stochastic_safeqp``
in this project draw ``delta`` per check. Over a long run that budget
compounds: at ``delta = 0.05`` per step, 400 steps would nominally
require 20. The anytime schedule instead gives early steps most of the
budget (small ``t`` -> large ``delta_0 * rho^t`` when ``rho < 1``) and
later steps little, while still summing to ``delta`` overall.

The practical consequence: the controller can keep certifying every step
forever on one budget, and the early, high-uncertainty steps get the
statistical power that is actually worth spending there.
"""

from __future__ import annotations

import math


def admissible_schedule(
    total_delta: float, decay: float, horizon: int | None = None
) -> float:
    """Largest ``delta_0`` with ``delta_0/(1-decay) <= total_delta``.

    Args:
        total_delta: the global ``delta`` for the whole horizon.
        decay: ``rho`` in ``(0, 1)``; smaller decays faster.
        horizon: unused for the bound (which is uniform in T), accepted
            so callers can pass one and the formula stays readable.

    Returns:
        ``delta_0``, the per-step budget at ``t = 0``.

    Raises:
        ValueError: if ``decay`` is not in ``(0, 1)`` or ``delta`` is
            not positive.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError("decay (rho) must lie in (0, 1)")
    if total_delta <= 0.0:
        raise ValueError("total_delta must be positive")
    return total_delta * (1.0 - decay)


def step_budget(delta_0: float, decay: float, step: int) -> float:
    """``delta_t = delta_0 * rho^t`` -- the budget available at ``step``.

    Args:
        delta_0: the value from :func:`admissible_schedule`.
        decay: ``rho`` in ``(0, 1)``.
        step: the non-negative step index.

    Returns:
        The confidence level to spend at this step.

    Raises:
        ValueError: on a negative step.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    return delta_0 * decay**step


def total_spend(delta_0: float, decay: float, steps: int) -> float:
    """``sum_{t<steps} delta_0 * rho^t`` -- the budget actually consumed.

    The closed form is ``delta_0 * (1 - rho^steps)/(1 - rho)``, which
    stays at or below the global budget for every horizon. Useful as a
    runtime assertion: spending more than the budget means the schedule
    is not admissible and the anytime guarantee does not hold.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return 0.0
    if abs(decay - 1.0) < 1e-15:
        return delta_0 * steps
    return delta_0 * (1.0 - decay**steps) / (1.0 - decay)


def budget_exhausts_uniformly(total_delta: float, per_step_delta: float,
                              steps: int) -> bool:
    """Does a naive ``per_step`` budget run out over ``steps`` steps?

    The comparison the anytime schedule replaces. Returns True when
    ``steps * per_step_delta > total_delta`` -- the regime where
    spending ``per_step_delta`` each step is not admissible and the
    naive rule gives up.
    """
    return steps * per_step_delta > total_delta


def anytime_alpha(supermartingale_start: float, delta_t: float) -> float:
    """Ville's threshold ``alpha_t = delta_t * E[M_0]``.

    ``Pr[exists t: M_t >= alpha_t] <= delta_t``. Exposed so a caller can
    check its own statistic against the threshold rather than
    re-deriving the constant.
    """
    if supermartingale_start <= 0.0:
        raise ValueError("supermartingale_start must be positive")
    return delta_t * supermartingale_start


def anytime_probability_bound(supermartingale_start: float, threshold: float) -> float:
    """``P[exists t: M_t >= threshold] <= threshold / E[M_0]`` (Ville).

    Args:
        supermartingale_start: ``E[M_0]``, positive.
        threshold: the level ``alpha``.

    Returns:
        The failure probability bound, which may exceed 1 (then the
        bound is vacuous and the caller should tighten the schedule).
    """
    if supermartingale_start <= 0.0:
        raise ValueError("supermartingale_start must be positive")
    return threshold / supermartingale_start
