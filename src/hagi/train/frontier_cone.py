"""R107: the frontier cone, and growth derived rather than assumed.

``Hagi/Growth/FrontierScaling.lean``. R104 left ONE premise open:
``h_emp_frontier_scaling``, ``alpha·C_t <= gamma·D_t``, that usable
disagreement scales with capability. R107 closes it by making the cone
INVARIANT under a production dynamic:

    D_{t+1} >= rho·D_t + beta·C_t − xi_t

``frontier_cone_inductive``
    The cone ``D_t >= (alpha/gamma)·C_t`` is preserved by one step,
    provided

        beta >= (alpha/gamma)·((1+alpha) − rho) + xi_t/C_t

    plus a cap the audit's sketch left IMPLICIT: ``C_{t+1} <= (1+alpha)·C_t``.
    Without the cap a capability jump can outrun the frontier and the
    cone breaks -- the formalisation found a missing hypothesis in the
    sketch rather than proving the sketch as drawn.

``frontier_cone_invariant``
    Induction over all t, carrying capability positivity alongside so the
    circular dependency between the cone and ``C_t > 0`` is broken
    rather than assumed.

``sustained_takeoff_from_production``
    cone => gate => ``C_T >= C_0(1+alpha)^T``, with NO empirical premise
    in the hypotheses. This is the causal loop the audit called the
    project's main gap: a growing model PRODUCES new disagreement at
    rate ``beta``, and that production is what sustains growth.

``frontier_asymptotic``
    Under proportional friction ``xi_t <= xibar·C_t`` the threshold
    becomes a CONSTANT ``(alpha/gamma)((1+alpha) − rho) + xibar <= beta``.
    Four measured constants decide the regime.

Two consequences the module states as numbers rather than prose:

- ``cone_threshold`` is the operational criterion. Above it growth is
  unbounded; below it R104's converse caps capability forever. Measured
  from (alpha, gamma, rho, xibar) -- not a slogan.
- ``frontier_ratio`` is the early warning: one number per generation,
  ``D_t/C_t`` against ``k = alpha/gamma``. Approaching the boundary says
  the frontier is exhausted, long before capability plateaus.

The port also records the additive-dynamics finding: under exact
additive production the gain settles at ``G = alpha·C``, so the system
runs at EXACTLY the certified rate, not faster. The overlay of the
certificate does not permit overshoot.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass


def cone_threshold(alpha: float, gamma: float, rho: float,
                   xibar: float) -> float:
    """``(alpha/gamma)((1+alpha) − rho) + xibar`` -- the asymptotic bar.

    ``frontier_asymptotic``: under proportional friction the per-step
    threshold becomes a constant, so the regime is decided by four
    measured constants rather than a varying condition.

    Args:
        alpha: the gate parameter.
        gamma: the harvest rate.
        rho: the frontier retention.
        xibar: the proportional friction coefficient.

    Returns:
        The value ``beta`` must reach.

    Raises:
        ValueError: on non-positive alpha or gamma.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    return (alpha / gamma) * ((1.0 + alpha) - rho) + xibar


def cone_holds(alpha: float, gamma: float, capability: float,
               frontier: float) -> bool:
    """``D_t >= (alpha/gamma)·C_t`` -- the cone, as a check.

    R104's open premise. A caller evaluating this on telemetry learns
    whether the premise currently holds rather than assuming it.

    Raises:
        ValueError: on non-positive alpha or gamma.
    """
    if alpha <= 0.0 or gamma <= 0.0:
        raise ValueError("alpha and gamma must be positive")
    return frontier >= (alpha / gamma) * capability


def frontier_ratio(alpha: float, gamma: float, capability: float,
                   frontier: float) -> float:
    """``(D_t/C_t) / (alpha/gamma)`` -- distance to the cone boundary.

    ``>= 1`` inside the cone, approaching ``0`` as the frontier is
    exhausted. This is the early-warning number: it falls long before
    capability plateaus, so a controller can switch to widening the
    frontier (new domain, synthetic data, discovery) while there is
    still capability left to protect.

    Returns:
        The ratio; ``inf`` when capability is zero.

    Raises:
        ValueError: on non-positive alpha or gamma, or negative inputs.
    """
    if alpha <= 0.0 or gamma <= 0.0:
        raise ValueError("alpha and gamma must be positive")
    if capability < 0.0 or frontier < 0.0:
        raise ValueError("capability and frontier must be non-negative")
    if capability == 0.0:
        return math.inf
    return (frontier / capability) / (alpha / gamma)


def cone_step(
    alpha: float, gamma: float, rho: float, beta: float,
    capability: float, frontier: float, friction: float,
    capability_next: float,
) -> bool:
    """``frontier_cone_inductive``: is the cone preserved by one step?

    Args:
        alpha: the gate parameter.
        gamma: the harvest rate.
        rho: the frontier retention.
        beta: the production rate -- how fast disagreement is PRODUCED.
        capability: ``C_t``.
        frontier: ``D_t``.
        friction: ``xi_t``, the per-step decay of the frontier.
        capability_next: ``C_{t+1}``, needed for the CAP check.

    Returns:
        True iff the cone holds at ``t+1`` given it holds at ``t`` and the
        threshold and cap are met.

    Raises:
        ValueError: on non-positive alpha or gamma.
    """
    if alpha <= 0.0 or gamma <= 0.0:
        raise ValueError("alpha and gamma must be positive")
    if capability <= 0.0:
        raise ValueError("the cone argument needs positive capability")
    if not cone_holds(alpha, gamma, capability, frontier):
        return False
    # the cap the audit's sketch left implicit
    if capability_next > (1.0 + alpha) * capability + 1e-12:
        return False
    threshold = cone_threshold(alpha, gamma, rho, friction / capability)
    if beta < threshold - 1e-12:
        return False
    # the dynamic itself
    return cone_holds(alpha, gamma, capability_next,
                      rho * frontier + beta * capability - friction)


def regime(cone_is_held: bool, beta: float, threshold: float) -> str:
    """``"sustained"`` or ``"capped"`` -- the operational verdict.

    Above the threshold the cone holds and growth is unbounded; below it
    R104's converse caps capability at ``gamma·Dbar/alpha`` forever. This
    is the decision a growth controller has to make, as a string rather
    than as an inference the caller draws itself.
    """
    if not cone_is_held:
        return "capped"
    return "sustained" if beta >= threshold else "capped"


def certified_capability(alpha: float, c0: float, cycles: int) -> float:
    """``C_0(1+alpha)^T`` -- the certificate under sustained production.

    Valid once the cone is established; with no empirical premise in the
    hypotheses this is a consequence of the production dynamic rather
    than an assumption about disagreement.

    Saturates at ``inf`` past the float range rather than raising: a
    long horizon reaches ``(1.1)^10000`` around step 709 of exponent and
    overflows a float, and "how large is the certificate at T=10^4" is a
    question whose honest answer is "past the largest float", not an
    exception. A caller comparing horizons should compare small ones,
    where the number is exact.

    Raises:
        ValueError: on non-positive alpha or c0, or a negative horizon.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if c0 <= 0.0:
        raise ValueError("c0 must be positive")
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    exponent = math.log(c0) + cycles * math.log1p(alpha)
    if exponent > math.log(sys.float_info.max):
        return math.inf
    return math.exp(exponent)


def settled_gain(alpha: float) -> float:
    """The gain the system settles at, per unit capability: ``alpha``.

    Under exact additive production the gain is squeezed to ``G = alpha·C``
    -- the system runs at EXACTLY the certified rate, not faster. The
    overlay of the certificate does not permit overshoot, which is worth
    knowing: a mechanism that appears to exceed the certificate is
    exceeding it in measurement error, not in fact.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return alpha


@dataclass(frozen=True)
class ConeState:
    """One generation's cone diagnostics.

    Attributes:
        capability: ``C_t``.
        frontier: ``D_t``.
        ratio: ``(D_t/C_t)/(alpha/gamma)``; ``>= 1`` is inside the cone.
        threshold: the ``beta`` that must be reached this step.
    """

    capability: float
    frontier: float
    ratio: float
    threshold: float

    @property
    def inside_cone(self) -> bool:
        return self.ratio >= 1.0

    @property
    def verdict(self) -> str:
        return "sustained" if self.inside_cone else "capped"


def diagnose(alpha: float, gamma: float, rho: float, beta: float,
             capability: float, frontier: float, friction: float
             ) -> ConeState:
    """The full cone diagnostic for one generation.

    Reports the distance to the boundary and the production rate needed
    to hold it -- so a controller can tell "the frontier is thin" from
    "production is too slow", which need different responses.

    Raises:
        ValueError: on non-positive alpha, gamma or capability.
    """
    if capability <= 0.0:
        raise ValueError("the cone diagnostic needs positive capability")
    return ConeState(
        capability=capability,
        frontier=frontier,
        ratio=frontier_ratio(alpha, gamma, capability, frontier),
        threshold=cone_threshold(alpha, gamma, rho, friction / capability),
    )