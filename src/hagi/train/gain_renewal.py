"""Gain renewal: sustained growth iff the frontier scales (R104).

``Hagi/Growth/GainRenewal.lean``. R102 proved that a FIXED gain buys only
a bounded transient. This module closes that bridge, and it does so in
the direction that matters for the growth loop: the certificate for
unbounded growth is now a MEASURED condition on telemetry, not a slogan.

``gain_renewal_recurrence``
    If the gain is a linear harvest of usable disagreement
    (``γ·D ≤ G ≤ γ·D`` -- the harvest must be EXACT, see below) and the
    diversity is replenished (``D_{t+1} ≥ ρ·D_t + inj − ξ``, ``inj > ξ``),
    then ``G_{t+1} ≥ ρ·G_t + γ(inj − ξ)``.

    Exactness matters and is not decoration. A one-sided bound
    ``G ≥ γ·D`` is enough to make the recurrence derivable only if the
    gain is also not LARGER than the harvest; otherwise a gain from some
    other source could carry the growth while looking like disagreement.
    :func:`harvest_is_the_only_source` is that invariant, and it doubles
    as a diagnostic: if it fails, something is producing gain that the
    telemetry does not see.

``gain_renewal_floor_pos``
    The stationary gain floor ``G_min = γ(inj − ξ)/(1 − ρ) > 0``. Under
    ``inj > ξ`` the gains never die, at any horizon. At ``ρ = 1`` the
    gains grow LINEARLY instead.

``sustained_takeoff_window_lift``
    R102's window is LIFTED: under the renewal semantics, each
    generation's gate ``α·C_t ≤ G_t`` is paid by that generation's own
    produced gain, so ``C_T ≥ C₀(1+α)^T`` for every T. No window, no
    bound.

``renewal_feeds_takeoff``
    The gate is supplied by the FRONTIER-SCALING premise
    ``α·C_t ≤ γ·D_t``: usable disagreement scales with capability.
    That premise is EMPIRICAL and named as such here. PPT's stationary
    good-mass (R98) is the candidate source; connecting it to ``D_t`` is
    the open measurement, not a theorem.

``bounded_frontier_no_sustained_growth``
    The converse, and the reason the iff is honest: with an exact
    harvest and a BOUNDED frontier ``D_t ≤ D̄``, capability is capped
    forever at ``γ·D̄/α``. So sustained growth is not merely unproven
    without frontier scaling -- it is IMPOSSIBLE. The growth gate has a
    definite stopping signal: if the telemetry shows ``D_t`` flat, more
    merges will not help and the frontier must be widened instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RenewalParams:
    """The four constants of the renewal law.

    Attributes:
        gamma: the harvest rate, ``γ > 0`` -- gain per unit of usable
            disagreement.
        rho: the diversity retention, ``ρ ∈ (0, 1]``.
        injection: ``inj``, the fresh independent data arriving per step.
        decay: ``ξ``, the decay the same distribution imposes.
    """

    gamma: float
    rho: float
    injection: float
    decay: float

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must lie in (0, 1]")
        if self.injection < 0.0 or self.decay < 0.0:
            raise ValueError("injection and decay must be non-negative")

    @property
    def net_injection(self) -> float:
        """``inj − ξ`` -- positive is the liveness condition of §4."""
        return self.injection - self.decay

    @property
    def injection_exceeds_decay(self) -> bool:
        """``inj > ξ``: the data axis is alive."""
        return self.net_injection > 0.0

    @property
    def gain_floor(self) -> float:
        """``G_min = γ(inj − ξ)/(1 − ρ)``, the stationary gain.

        ``gain_renewal_floor_pos``: strictly positive whenever
        ``inj > ξ``, at any horizon. ``inf`` at ``ρ = 1``, where the
        gains grow linearly instead of settling.
        """
        if self.rho >= 1.0:
            return math.inf
        return self.gamma * self.net_injection / (1.0 - self.rho)

    def renew(self, gain: float) -> float:
        """``G_{t+1} ≥ ρ·G_t + γ(inj − ξ)`` -- one renewal step.

        Args:
            gain: the current gain ``G_t``.

        Returns:
            The guaranteed next gain.

        Raises:
            ValueError: on a negative gain.
        """
        if gain < 0.0:
            raise ValueError("gain must be non-negative")
        return self.rho * gain + self.gamma * self.net_injection


def renewal_floor(gamma: float, rho: float, injection: float,
                  decay: float) -> float:
    """``G_min = γ(inj − ξ)/(1 − ρ)`` -- the stationary gain floor."""
    return RenewalParams(gamma, rho, injection, decay).gain_floor


def gain_is_renewed(params: RenewalParams, gain: float,
                    steps: int = 1) -> float:
    """The gain after ``steps`` renewals.

    The recurrence solved in closed form, so "will my growth channel
    still be producing after 100 cycles?" is arithmetic.

    Args:
        params: the renewal constants.
        gain: ``G_0``.
        steps: how many generations to project forward.

    Returns:
        The guaranteed gain after ``steps``.

    Raises:
        ValueError: on a negative gain or negative steps.
    """
    if gain < 0.0:
        raise ValueError("gain must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    floor = params.gain_floor
    if math.isinf(floor):
        # rho == 1: linear growth, no stationary floor.
        return gain + steps * params.gamma * params.net_injection
    return floor + (gain - floor) * (params.rho ** steps)


def harvest_is_the_only_source(measured_gain: float, diversity: float,
                               gamma: float) -> bool:
    """``G ≤ γ·D`` -- the invariant, and a diagnostic.

    ``gain_renewal_recurrence`` needs the harvest to be EXACT: gain
    equals ``γ·D``, not merely at least it. Otherwise growth could be
    carried by a source the disagreement telemetry does not see, and the
    frontier-scaling condition of ``renewal_feeds_takeoff`` would be
    measured against a gain it does not explain.

    So this is not only a premise check. Run it on real telemetry: a
    ``False`` here means the gain has a source the frontier does not
    account for, which is a finding in its own right.

    Args:
        measured_gain: the gain actually observed.
        diversity: the usable disagreement measured.
        gamma: the harvest rate.

    Returns:
        True iff the gain is within what the harvest can explain.
    """
    return measured_gain <= gamma * diversity


# --- the lifted window ---------------------------------------------------


def sustained_capability(alpha: float, c0: float, cycles: int) -> float:
    """``C₀(1+α)^T`` -- the certified capability under renewal.

    ``sustained_takeoff_window_lift`` / ``renewal_feeds_takeoff``: with
    each generation's gate paid by its own produced gain there is NO
    window -- the multiplicative law applies at every cycle.

    Args:
        alpha: the gate parameter.
        c0: the initial capability.
        cycles: the horizon.

    Returns:
        The certified capability at that horizon.

    Raises:
        ValueError: on non-positive alpha or c0, or negative cycles.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if c0 <= 0.0:
        raise ValueError("c0 must be positive")
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    return c0 * (1.0 + alpha) ** cycles


def gate_is_supplied(alpha: float, capability: float, diversity: float,
                     gamma: float) -> bool:
    """``α·C_t ≤ γ·D_t`` -- the frontier-scaling premise, as a check.

    ``h_emp_frontier_scaling``. This is the ONE empirical premise R104
    leaves open, and it is stated so a caller can evaluate it on
    telemetry rather than assume it: the stationary frontier level must
    cover what the gate demands, at every generation.

    Args:
        alpha: the gate parameter.
        capability: the current capability field.
        diversity: the current usable disagreement.
        gamma: the harvest rate.

    Returns:
        True iff the frontier supplies the gate at this generation.
    """
    return alpha * capability <= gamma * diversity


def sustained_growth_possible(alpha: float, gamma: float,
                              d_bar: float) -> float:
    """``γ·D̄/α`` -- the ceiling a bounded frontier imposes, forever.

    ``bounded_frontier_no_sustained_growth``: with an exact harvest and
    ``D_t ≤ D̄``, capability can never exceed this. So a growth
    controller that observes a flat frontier knows its ceiling exactly,
    and knows that more merges will not raise it.

    Args:
        alpha: the gate parameter.
        gamma: the harvest rate.
        d_bar: the frontier bound.

    Returns:
        The capability ceiling.

    Raises:
        ValueError: on non-positive alpha or gamma, or a negative bound.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if d_bar < 0.0:
        raise ValueError("d_bar must be non-negative")
    return gamma * d_bar / alpha


def frontier_must_scale(alpha: float, capability: float,
                        gamma: float) -> float:
    """``(α/γ)·C_t`` -- the frontier level the gate will demand.

    The companion to :func:`gate_is_supplied`: not "is the premise
    satisfied" but "what would it take". Since capability grows
    geometrically under takeoff, so must the frontier -- which is why
    the frontier is what a growth controller has to widen.

    Args:
        alpha: the gate parameter.
        capability: the current capability.
        gamma: the harvest rate.

    Returns:
        The required disagreement level.

    Raises:
        ValueError: on non-positive alpha or gamma.
    """
    if alpha <= 0.0 or gamma <= 0.0:
        raise ValueError("alpha and gamma must be positive")
    return alpha * capability / gamma