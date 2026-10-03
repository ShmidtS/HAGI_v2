"""The self-development loop: selection, certification, renewal.

``Hagi/Growth/SelfDevelopment.lean``, ``Hagi/Growth/StateBinding.lean``,
``Hagi/Growth/RecursiveSelfDevelopment.lean``,
``Hagi/Probability/AdaptiveSuccess.lean`` (R118-R122).

R118 ``self_development_find`` / ``certified_gain_select``
    If the safe candidate set holds an action with positive certified
    gain, the argmax controller selects one that is positive too; with
    measurement concentration ``|Ghat a - g a| <= eps`` (R114) and a
    candidate at ``3*eps`` measured, the selected action's TRUE gain is
    >= ``g a0 - 2*eps > 0``. The 3*eps rule turns "best measured" into
    "actually positive": below it the controller cannot tell.
    :func:`self_development_select` and :func:`selection_threshold`
    implement the pair.

R119 ``StateBinding``
    The takeoff sequences are fields of the actual growth state
    (``C_t = capability(S_t)``, ``D_t = dataField(S_t)``), not shadow
    bookkeeping. :class:`GrowthState` binds the measured quantities the
    project already produces (eval AVG, frontier D_t, cycle costs).

R120 ``DirectionalParetoController``
    Action selection by a single scalar (gain/cost) fixes one point of
    view. The directed controller scores a candidate by the largest
    balanced level ``tau`` with ``u(a, i) >= omega_i * tau`` on every
    coordinate (capability, wall-clock, held-out floor, risk).
    :func:`dir_tau` and :func:`pareto_select` port it.

R121 ``AdaptiveSuccess``
    Successes are not i.i.d.: each cycle's outcome depends on what the
    system already learned. With per-cycle fresh randomness the
    conditional floors ``E[S_t | F_t] >= p0`` still concentrate:
    ``P[sum S >= n*p0 - sqrt(n*log(1/delta)/2)] >= 1 - delta``.
    :func:`adaptive_success_floor` gives the certified sum floor.

R122 ``opportunity_renewal`` / ``closed_loop_takeoff``
    The self-development arrow: a successful cycle must RENEW the
    opportunity set, ``D' >= D + delta_O`` with
    ``delta_O <= beta*C - xi - (1-rho)*D`` -- otherwise the loop is
    repeated fine-tuning. :func:`opportunity_delta` computes the net
    renewal; :func:`multiplicative_growth` composes the per-cycle law
    ``C_{t+1} >= C_t * exp(alpha*S_t - eps_t)``; :func:`closed_loop_floor`
    is the R121-composed wall-clock capability floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# R118: selection under certified measurement noise
# ----------------------------------------------------------------------

def selection_threshold(eps: float) -> float:
    """The R118 3*eps bar an action's measured gain must clear.

    ``certified_gain_select``: with ``|Ghat - g| <= eps`` per action and
    a candidate at ``g(a0) >= 3*eps``, the argmax-by-measurement has
    TRUE gain >= ``g(a0) - 2*eps > eps > 0``. Below the bar the
    controller cannot certify positivity: the honest verdict is
    UNDECIDED, not the argmax.
    """
    return 3.0 * eps


def self_development_select(
    measured: dict[str, float],
    eps: float,
    safe: set[str] | None = None,
) -> tuple[str | None, float]:
    """``self_development_find`` + ``certified_gain_select`` (R118).

    Args:
        measured: measured certified gains per action (Ghat).
        eps: the R114 measurement radius (``|Ghat - g| <= eps``).
        safe: the safe candidate names; defaults to all keys.

    Returns:
        ``(name, lower_bound)`` of the argmax-by-measurement safe
        candidate when it clears :func:`selection_threshold`, else
        ``(None, 0.0)`` -- no action certifies a TRUE positive gain and
        the honest choice is to not spend compute.
    """
    pool = {k: v for k, v in measured.items() if safe is None or k in safe}
    if not pool:
        return None, 0.0
    best = max(pool, key=lambda k: pool[k])
    bound = pool[best] - 2.0 * eps
    if bound <= 0.0:
        return None, 0.0
    return best, bound


def search_cost_bound(cost_per_measurement: float, n_candidates: int) -> float:
    """``search_cost_bound`` (R118): the measurement budget is linear.

    Evaluating the whole candidate set costs ``|A| * c_meas`` -- finite
    and known before the search starts.
    """
    return cost_per_measurement * n_candidates


# ----------------------------------------------------------------------
# R120: directed Pareto selection
# ----------------------------------------------------------------------

def dir_tau(u: dict[str, dict[str, float]], omega: dict[str, float],
            a: str) -> float:
    """The directed Pareto level of action ``a`` (R120 ``dirTau``).

    ``tau(a) = max { tau : forall i, u(a, i) >= omega_i * tau }`` --
    the largest balanced improvement level across the ``omega``-weighted
    coordinates. Positive iff ``a`` improves EVERY coordinate.
    """
    levels = []
    for i, w in omega.items():
        if w <= 0.0 or i not in u[a]:
            continue
        levels.append(u[a][i] / w)
    return min(levels) if levels else 0.0


def pareto_select(u: dict[str, dict[str, float]], omega: dict[str, float],
                  eps: dict[str, float] | None = None) -> tuple[str | None, float]:
    """``dir_controller_find`` / ``noisy_pareto_select`` (R120).

    Selects the argmax-``tau`` action. With per-coordinate measurement
    radii ``eps_i`` the TRUE ``tau`` of the choice is certified at
    ``tau_hat - 2 * max_i(eps_i / omega_i)``; below zero the selection
    does not certify a directed improvement.
    """
    taus = {a: dir_tau(u, omega, a) for a in u}
    if not taus:
        return None, 0.0
    best = max(taus, key=lambda k: taus[k])
    bound = taus[best]
    if eps is not None:
        penalty = 2.0 * max(eps[i] / omega[i] for i in omega
                            if i in eps and omega[i] > 0.0)
        bound = taus[best] - penalty
    if bound <= 0.0:
        return None, 0.0
    return best, bound


# ----------------------------------------------------------------------
# R119: the growth state the takeoff sequences bind to
# ----------------------------------------------------------------------

@dataclass
class GrowthState:
    """The measured fields the R119 StateBinding theorems quantify over.

    ``C_t`` is the capability field (eval AVG, lower is CE-better:
    stored negated so growth is up), ``D_t`` the usable frontier (the
    measured Jensen gap of the expert pool), ``S_t`` the realized
    success level of the last cycle in [0, 1].
    """

    capability: float
    frontier: float
    success: float = 0.0
    eps_t: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.success <= 1.0:
            raise ValueError("success must lie in [0, 1]")
        if self.capability < 0.0:
            raise ValueError("capability must be non-negative")

    def capability_after_cycle(self, alpha: float) -> float:
        """The per-cycle law ``C' >= C * exp(alpha*S - eps)`` (R122)."""
        return self.capability * math.exp(alpha * self.success - self.eps_t)


# ----------------------------------------------------------------------
# R121: adaptive (non-i.i.d.) success concentration
# ----------------------------------------------------------------------

def adaptive_success_floor(n: int, p0: float, delta: float) -> float:
    """``adaptive_success_concentrated`` (R121).

    With per-cycle conditional success floors ``E[S_t | F_t] >= p0``
    and fresh randomness per cycle, with probability ``>= 1 - delta``:

        sum_t S_t >= n*p0 - sqrt(n * log(1/delta) / 2)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return n * p0 - math.sqrt(n * math.log(1.0 / delta) / 2.0)


# ----------------------------------------------------------------------
# R122: opportunity renewal and the closed loop
# ----------------------------------------------------------------------

def opportunity_delta(d: float, rho: float, beta: float, c: float,
                      xi: float) -> float:
    """The net renewal ``delta_O`` (R122 ``opportunity_renewal``).

    ``delta_O = beta*C - xi - (1 - rho)*D``: the frontier production
    minus the decay tax on the existing frontier. Positive iff the
    cycle RENEWS the opportunity set (``D' >= D + delta_O``); at zero
    or below the loop is repeated fine-tuning, not recursive growth.
    """
    net = beta * c - xi - (1.0 - rho) * d
    return net


def multiplicative_growth(c0: float, successes: list[float],
                          leaks: list[float], alpha: float) -> float:
    """``multiplicative_growth`` (R122): ``C_T >= C_0 * exp(alpha*sum S - sum eps)``."""
    if len(successes) != len(leaks):
        raise ValueError("successes and leaks must align")
    exp_term = alpha * sum(successes) - sum(leaks)
    return c0 * math.exp(exp_term)


def closed_loop_floor(c0: float, n: int, p0: float, delta: float,
                      alpha: float, leak_total: float) -> float:
    """``closed_loop_takeoff`` (R122): the certified wall-clock floor.

    Composes R121's adaptive success concentration with the
    multiplicative law: with probability ``>= 1 - delta``,
    ``C_n >= C_0 * exp(alpha * adaptive_success_floor(n, p0, delta) - sum eps_t)``.
    """
    s_floor = adaptive_success_floor(n, p0, delta)
    return c0 * math.exp(alpha * s_floor - leak_total)
