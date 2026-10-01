"""The growth meta-law: certified takeoff and the epsilon countdown.

``Hagi/Dynamics/FastGrowth.lean`` and ``Hagi/Dynamics/CapabilityGain.lean``.

``capability_multiplicative``
    ``C (t+1) >= C t * (1 + alpha)`` implies ``C T >= C 0 * (1+alpha)^T``
    -- a constant rate every cycle.

``capability_takeoff_counted``
    ``C (t+1) >= C t * (1+alpha)^(s t)`` with ``s t`` the SUCCESS
    indicator (1 = certified success, 0 = no harm) implies

        C T >= C 0 * (1+alpha)^(sum_t s_t)

    exponential in the SUCCESS COUNT, not in the cycle count. The
    practical consequence: cycles are a budget, successes are the
    progress. A run with 90% failed cycles still compounds, but only
    on the 10% -- so the rate to optimize is the success rate, not
    the step rate. Failures are NEUTRAL (``s t = 0`` multiplies by 1):
    safety and growth in one law.

``capability_takeoff_floor``
    A measured success count ``N >= sum s_t`` gives ``C T >= C 0 (1+a)^N``
    -- the bridge to the (still open) probability layer.

``capability_gain_transfer``
    ``Rext1 = E1 + g1``, ``Rext2 = E2 + g2``, certified ``E2 <= E1 - G``
    and non-worsening actual gap ``g2 <= g1`` give ``Rext2 <= Rext1 - G``
    -- the external cost drops by the certified gain plus whatever the
    gap gives up, so a step that improves the objective can never be
    cancelled by a gap that drifts the other way.

``pb_gap_bound_mono``
    The PAC-Bayes certified gap is monotone in the KL budget: keeping
    ``KL_(t+1) <= KL_t`` keeps the bound from growing.
"""

from __future__ import annotations

import math


def capability_floor(c0: float, alpha: float, successes: int) -> float:
    """``C_0 * (1+alpha)^N`` -- the certified capability after N successes.

    Args:
        c0: the starting capability ``C 0``.
        alpha: the relative per-success gain, strictly positive.
        successes: the counted successes ``sum_t s_t``.

    Returns:
        The certified lower bound on capability.

    Raises:
        ValueError: when ``alpha <= 0`` or ``c0 < 0``.
    """
    if alpha <= 0.0:
        raise ValueError("capability_floor needs a positive alpha")
    if c0 < 0.0:
        raise ValueError("capability_floor needs a non-negative c0")
    return c0 * (1.0 + alpha) ** successes


def gain_transfer(e1: float, g1: float, e2: float, g2: float, gamma: float) -> float:
    """``Rext2 <= Rext1 - Gamma`` (``capability_gain_transfer``).

    Args:
        e1: objective before the step; ``e2`` after.
        g1: the gap term before; ``g2`` after (``g2 <= g1`` required).
        gamma: the certified gain ``E2 <= E1 - Gamma``.

    Returns:
        The certified upper bound on the external cost after the step.
    """
    rext1 = e1 + g1
    rext2 = e2 + g2
    return rext1 - gamma


def gap_bound(kl: float, base: float) -> float:
    """``KL + base`` -- the PAC-Bayes gap bound (``pb_gap_bound_mono``)."""
    return kl + base


def required_successes(target_ratio: float, alpha: float) -> int:
    """``N >= ln(target/C0) / ln(1+alpha)`` -- cycles needed for a target.

    The inverse of :func:`capability_floor`. This is the planning
    number: given a goal and a per-success rate, how many certified
    successes the run has to actually land. Used to decide whether a
    configuration is worth starting at all.

    Raises:
        ValueError: when the target is unreachable.
    """
    if target_ratio <= 0.0:
        return 0
    if alpha <= 0.0:
        raise ValueError("required_successes needs a positive alpha")
    return max(0, math.ceil(math.log(target_ratio) / math.log1p(alpha)))


def takeoff_horizon(success_rate: float, alpha: float, target_ratio: float) -> int:
    """``T >= N / p`` -- cycles needed when successes arrive at rate p.

    The bridge the Lean module names as still open on the probability
    side: ``N ~ pT`` is a concentration statement, not a determinstic
    one. This function takes p as MEASURED and returns the horizon; it
    does not prove the concentration, it only reports what a measured
    rate implies.

    Args:
        success_rate: the measured ``p`` in ``(0, 1]``.
        alpha: the per-success gain.
        target_ratio: the capability multiple to reach.

    Raises:
        ValueError: on a non-positive rate.
    """
    if not 0.0 < success_rate <= 1.0:
        raise ValueError("takeoff_horizon needs a success rate in (0, 1]")
    needed = required_successes(target_ratio, alpha)
    return math.ceil(needed / success_rate)


def risk_floor(r0: float, beta: float, successes: int) -> float:
    """``R_0 * (1-beta)^N`` -- exponential risk decay in successes.

    The dual of :func:`capability_floor`: each certified success
    multiplies the external risk by at most ``1-beta``, failures are
    neutral. The epsilon-floor is therefore reached after finitely many
    successes rather than in finite time -- progress is not guaranteed
    by waiting, only by succeeding.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("risk_floor needs beta in (0, 1)")
    return r0 * (1.0 - beta) ** successes