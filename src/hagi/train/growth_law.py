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


# --- R95: the probabilistic chain success -> concentration -> growth ----


def success_count_lower(mean_rate: float, rounds: int, delta: float) -> float:
    """``p0*T - sqrt(2 T log(1/delta))`` -- Hoeffding's lower bound.

    ``success_count_lower``: if the success indicators have mean at
    least ``p0``, then

        Pr[sum_t S_t >= p0*T - sqrt(2 T log(1/delta))] >= 1 - delta.

    The deviation ``sqrt(2 T log(1/delta))`` is ``O(sqrt(T))``, so the
    RELATIVE error on the success count decays as ``1/sqrt(T)``: a long
    horizon is worth running exactly while that shrinkage matters.

    This is the honest place to spend ``delta``. In this project
    ``R93`` supplies it from an anytime schedule, so the bound holds
    for every horizon on one budget rather than ``delta`` per run.

    Args:
        mean_rate: ``p0``, the measured mean success rate in ``[0, 1]``.
        rounds: ``T``.
        delta: the confidence level in ``(0, 1)``.

    Returns:
        The guaranteed lower bound on the success count. May be
        negative for small ``T`` -- the bound is then vacuous, and the
        caller should read that as "not enough rounds to certify".

    Raises:
        ValueError: on an out-of-range rate, negative ``T``, or a bad
            ``delta``.
    """
    if not 0.0 <= mean_rate <= 1.0:
        raise ValueError("mean_rate must lie in [0, 1]")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if rounds == 0:
        return 0.0
    return mean_rate * rounds - math.sqrt(2.0 * rounds * math.log(1.0 / delta))


def deviation(rounds: int, delta: float) -> float:
    """``sqrt(2 T log(1/delta))`` -- the Hoeffding deviation on its own."""
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return math.sqrt(2.0 * rounds * math.log(1.0 / delta))


def takeoff_time_form(
    c0: float,
    mean_rate: float,
    gain_per_success: float,
    friction: float,
    rounds: int,
    delta: float,
) -> dict:
    """``Pr[C_T >= C0 exp(a(p0 T - D) - sum eps_t)] >= 1 - delta`` (R95).

    ``log_growth`` + ``takeoff_time_form``: if a success multiplies the
    log-capability by ``a`` (up to per-round friction ``eps_t``, summed
    here as a total ``friction``), then with probability at least
    ``1 - delta``

        C_T >= C0 * exp(a * (p0*T - D) - friction),  D = sqrt(2 T log(1/delta))

    The exponent is the payoff and the whole point: it grows LINEARLY
    in ``T`` at rate ``a * p0`` while the deviation grows as
    ``sqrt(T)``. So the ranking rule the review draws out follows
    directly -- with ``a * p0`` as the objective, raising the success
    RATE beats raising the per-success gain whenever the product is
    larger, and long horizons pay off exactly because ``a*p0*T``
    outruns ``a*sqrt(T)``.

    Args:
        c0: the starting capability.
        mean_rate: ``p0``, the measured success rate.
        gain_per_success: ``a > 0``.
        friction: the summed per-round loss ``sum eps_t``.
        rounds: ``T``.
        delta: the confidence level.

    Returns:
        ``{"successes_lower", "exponent", "capability_lower", "delta"}``.

    Raises:
        ValueError: on non-positive ``c0``/``a``, negative friction, or
            a bad rate/delta.
    """
    if c0 <= 0.0:
        raise ValueError("takeoff_time_form needs a positive c0")
    if gain_per_success <= 0.0:
        raise ValueError("takeoff_time_form needs a positive gain a")
    if friction < 0.0:
        raise ValueError("friction must be non-negative")
    successes = success_count_lower(mean_rate, rounds, delta)
    exponent = gain_per_success * successes - friction
    # exp(exponent) overflows a float past ~709, which a long horizon
    # reaches easily (exponent 111 at T=1000, 1110 at T=10000). The
    # exponent is the quantity to compare, so it is returned first and
    # the capability saturates at +inf rather than raising -- a caller
    # asking "is the target reached" reads ``log_capability_lower``.
    log_lower = math.log(c0) + exponent
    try:
        capability = c0 * math.exp(exponent)
    except OverflowError:
        capability = float("inf")
    return {
        "successes_lower": successes,
        "exponent": exponent,
        "log_capability_lower": log_lower,
        "capability_lower": capability,
        "delta": delta,
    }


def additive_as_multiplicative(capability: float, gain: float) -> float:
    """``additive_as_multiplicative``: an additive gain IS a multiplier.

    ``R96``. The controller measures an ADDITIVE improvement
    ``growGain`` per cycle, while ``capability_takeoff_counted`` speaks
    in MULTIPLIERS ``(1+alpha)``. The bridge is exact and requires no
    assumption:

        C + g  ==  C * (1 + g/C)

    so the multiplier for a measured additive gain is ``1 + g/C`` and
    nothing is lost in translation. This is what makes the takeoff law a
    statement about the REAL cycle fields (``growGain`` and
    ``capability``) rather than about abstract numbers supplied by hand.

    Args:
        capability: ``C > 0``.
        gain: ``g``, any real value (a negative gain is a legal, harmful
            cycle -- its multiplier is below 1).

    Returns:
        The multiplicative factor ``1 + g/C``.

    Raises:
        ValueError: on non-positive ``capability``.
    """
    if capability <= 0.0:
        raise ValueError("additive_as_multiplicative needs a positive capability")
    return 1.0 + gain / capability


def alpha_from_gain(capability: float, gain: float) -> float:
    """``alpha = g/C`` -- the per-success rate that feeds takeoff.

    The argument of :func:`additive_as_multiplicative`, isolated so a
    caller can read the growth rate directly off the measured state.

    Raises:
        ValueError: on non-positive ``capability``.
    """
    if capability <= 0.0:
        raise ValueError("alpha_from_gain needs a positive capability")
    return gain / capability


def horizon_needed(
    c0: float, target: float, mean_rate: float, gain_per_success: float,
    delta: float, cap: int = 100_000,
) -> int:
    """Smallest ``T`` whose R95 exponent already reaches ``log(target/c0)``.

    The planning inverse of :func:`takeoff_time_form`. Because the
    exponent is ``a*p0*T - a*sqrt(2 T log(1/delta)) - friction``, this is
    a monotone search over a closed form -- no simulation.

    Returns:
        The horizon, or ``-1`` when the rate cannot reach the target.

    Raises:
        ValueError: on non-positive inputs or ``target <= c0``.
    """
    if c0 <= 0.0 or target <= c0:
        raise ValueError("horizon_needed needs c0 > 0 and target > c0")
    if not 0.0 < mean_rate <= 1.0:
        raise ValueError("mean_rate must lie in (0, 1]")
    if gain_per_success <= 0.0:
        raise ValueError("horizon_needed needs a positive gain a")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    need = math.log(target / c0)
    lo, hi = 1, cap
    if takeoff_time_form(c0, mean_rate, gain_per_success, 0.0, hi, delta)[
        "exponent"
    ] < need:
        return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if takeoff_time_form(c0, mean_rate, gain_per_success, 0.0, mid, delta)[
            "exponent"
        ] >= need:
            hi = mid
        else:
            lo = mid + 1
    return lo