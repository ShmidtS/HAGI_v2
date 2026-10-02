"""Stochastic SafeQP: minibatch gradients with an explicit safety margin.

``Hagi/Step/StochasticSafeQP.lean`` (R92). The exact SafeQP controller
conditions the direction ``d`` on the EXACT per-domain gradients
``g_i``. Requiring a full pass per domain per step is what makes the
controller expensive: K extra backwards for K domains.

R92 replaces ``g_i`` by minibatch estimates
``ĝ_i = g_i + (1/m) * sum_j xi_{i,j}`` and pays for the substitution
with an EXPLICIT inflation of the safety budget:

    eps_noise = sigma * ||d|| * sqrt(2 * log(|K| / delta) / m)

with ``sigma`` the per-sample noise bound, ``||d||`` the direction
magnitude, ``m`` the minibatch size, ``K`` the protected domains (the
union bound over domains) and ``delta`` the confidence level.

The guarantees (all proved, no hidden constants):

``minibatch_inner_concentration``
    ``Pr[forall i, <g_i,d> - eps_noise <= <ĝ_i,d>] >= 1 - delta``
    -- each per-domain one-sided Hoeffding tail is exactly ``delta/|K|``
    by the choice of ``eps_noise``.

``minibatch_inner_concentration_upper``
    The mirrored upper tail, so the transfer is TWO-SIDED in the
    deployed direction.

``stochastic_safeQP_feasibility``
    If ``d`` satisfies the stochastic constraints
    ``<ĝ_i,d> >= -eps_i`` (which the QP enforces by construction), then
    with probability at least ``1-delta`` the TRUE gradients satisfy

        <g_i,d> >= -(eps_i + eps_noise)

The ``m`` inside ``eps_noise ~ 1/sqrt(m)`` is the variance reduction
that makes minibatching viable at all: quadrupling the minibatch
halves the required safety inflation.

**Why this is worth deploying.** The margin is not a penalty paid for
noise -- it is the price of NOT doing K full backward passes. With
``eps_noise`` computed rather than tuned, a controller can size its
minibatch to the accuracy it actually needs.
"""

from __future__ import annotations

import math


def anytime_delta(total_delta: float, decay: float, step: int) -> float:
    """``delta_t`` from the R93 geometric schedule, for one step.

    Wiring ``anytime_budget`` into the SafeQP margin: pass this instead
    of a fixed ``delta`` and the union bound stays valid for the whole
    run on ONE budget. A fixed per-step delta of 0.05 needs ``T * 0.05``
    over T steps -- vacuous by step 400.

    Args:
        total_delta: the global budget for the entire horizon.
        decay: ``rho`` in ``(0, 1)``.
        step: the non-negative step index.

    Returns:
        The confidence level to spend at this step.
    """
    from hagi.train.anytime_budget import admissible_schedule, step_budget

    return step_budget(admissible_schedule(total_delta, decay), decay, step)


def noise_epsilon(
    noise_bound: float,
    d_norm: float,
    minibatch: int,
    n_domains: int,
    delta: float,
) -> float:
    """``eps_noise = sigma * ||d|| * sqrt(2 * log(K/delta) / m)``.

    ``minibatch_inner_concentration``'s explicit constant. The union
    bound over ``K`` domains enters through ``log(K/delta)``, so each
    per-domain tail is ``delta/K``.

    Args:
        noise_bound: ``sigma``, the per-sample noise bound (``||xi||``).
        d_norm: ``||d||``, the magnitude of the step direction.
        minibatch: ``m``, the samples per domain per step.
        n_domains: ``K``, the number of protected domains.
        delta: the confidence level in ``(0, 1)``.

    Returns:
        The safety inflation required in place of exact gradients.

    Raises:
        ValueError: on non-positive ``minibatch``/``n_domains``, or a
            ``delta`` outside ``(0, 1)``.
    """
    if minibatch <= 0:
        raise ValueError("noise_epsilon needs a positive minibatch size")
    if n_domains <= 0:
        raise ValueError("noise_epsilon needs at least one protected domain")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return noise_bound * d_norm * math.sqrt(
        2.0 * math.log(n_domains / delta) / minibatch
    )


def feasible_margin(
    epsilon_i: float, noise_epsilon_value: float
) -> float:
    """``<g_i,d> >= -(eps_i + eps_noise)`` -- the certified true-gradient margin.

    The stochastic QP enforces ``<ĝ_i,d> >= -eps_i``; by
    ``stochastic_safeQP_feasibility`` the true gradients then satisfy
    this inflated margin with probability at least ``1-delta``.

    Args:
        epsilon_i: the per-domain stochastic budget.
        noise_epsilon_value: the inflation from :func:`noise_epsilon`.

    Returns:
        The margin the TRUE gradients are certified to respect.
    """
    return -(epsilon_i + noise_epsilon_value)


def hoeffding_tail_probability(
    threshold: float, minibatch: int, radius: float
) -> float:
    """``Pr[sum_j (-N_j) >= m*threshold] <= exp(-m*threshold^2/(2*R^2))``.

    ``minibatch_inner_tail`` with ``R = sigma * ||d||``. Returned as the
    FAILURE probability bound (the complement of the guarantee), so
    smaller is better and it is directly comparable to a delta budget.

    Raises:
        ValueError: on a non-positive radius or minibatch.
    """
    if radius <= 0.0:
        raise ValueError("hoeffding_tail_probability needs a positive radius")
    if minibatch <= 0:
        raise ValueError("hoeffding_tail_probability needs a positive minibatch")
    return math.exp(-minibatch * threshold * threshold / (2.0 * radius * radius))


def minibatch_for_margin(
    noise_bound: float,
    d_norm: float,
    n_domains: int,
    delta: float,
    target_epsilon: float,
    max_minibatch: int = 1_000_000,
) -> int:
    """Smallest ``m`` with ``eps_noise <= target`` -- a closed form.

    Inverting ``eps_noise <= target`` gives

        m >= 2 * (sigma*||d||)^2 * log(K/delta) / target^2

    so the minibatch size that meets a required accuracy is READ OFF
    rather than searched. This is the practical step: it turns "how big
    a minibatch do I need" into arithmetic, which is what makes the
    stochastic controller plannable.

    Args:
        noise_bound: ``sigma``.
        d_norm: ``||d||``.
        n_domains: ``K``.
        delta: the confidence level.
        target_epsilon: the largest acceptable safety inflation.
        max_minibatch: a ceiling on the returned size.

    Returns:
        ``(minibatch, achievable)``. ``achievable`` is False when the
        required size exceeds ``max_minibatch`` -- the target is then
        unreachable by minibatching at all, and the caller must either
        enlarge ``max_minibatch`` or fall back to exact gradients.
        Silently returning a clamped size would certify a margin the
        data cannot support.

    Raises:
        ValueError: on non-positive inputs or a non-positive target.
    """
    if noise_bound <= 0.0 or d_norm <= 0.0:
        raise ValueError("minibatch_for_margin needs positive sigma and ||d||")
    if target_epsilon <= 0.0:
        raise ValueError("minibatch_for_margin needs a positive target")
    if n_domains <= 0:
        raise ValueError("minibatch_for_margin needs at least one domain")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    needed = (
        2.0
        * (noise_bound * d_norm) ** 2
        * math.log(n_domains / delta)
        / target_epsilon**2
    )
    if needed > max_minibatch:
        return max_minibatch, False
    return max(1, int(math.ceil(needed))), True


def certified_feasible(
    inner_hat: float, epsilon_i: float, noise_bound: float, d_norm: float,
    minibatch: int, n_domains: int, delta: float,
) -> bool:
    """Check the stochastic constraint AND report the true-gradient safety.

    ``<ĝ_i,d> >= -eps_i`` is the constraint the QP enforces; whether the
    TRUE ``<g_i,d> >= -(eps_i + eps_noise)`` follows is a probabilistic
    statement, so this returns whether the constraint holds and, through
    :func:`feasible_margin`, what margin it certifies.

    Args:
        inner_hat: the measured ``<ĝ_i,d>`` from the minibatch.
        epsilon_i: the per-domain stochastic budget.
        noise_bound: ``sigma``.
        d_norm: ``||d||``.
        minibatch: ``m``.
        n_domains: ``K``.
        delta: the confidence level.

    Returns:
        True iff the stochastic constraint holds at this delta.
    """
    return inner_hat >= feasible_margin(epsilon_i, noise_epsilon(
        noise_bound, d_norm, minibatch, n_domains, delta
    )) + epsilon_i
