"""Selector thermostat (Growth/Thermostat.lean R387–R388,
Probability/GibbsConcentration R386).

The thermodynamic closing loop: temperature is a *measured
controller variable*, not a hyperparameter. Cooling schedule
T_t = T₀·α^t; exploration stops BY LAW when the Gibbs mass of
competitors e^{−δ/T_t} drops below θ (``gibbs_concentration``
R386: minimizer weight ≥ 1 − (n−1)e^{−δ/T}; R387: along the
annealing graph the weight → 1).

``thermostat_stop`` gives the closed-form horizon:

    N = ⌈ ln(δ/(T₀·ln(1/θ))) / ln α ⌉

so for all t ≥ N: e^{−δ/(T₀·α^t)} ≤ θ — stopping-by-law
instead of the stop_condition heuristic; the §1 selector,
liveness two-axis and saturation fold into one physical
circuit.

Measured premises (h_emp_): the energy gap δ between the
incumbent and the candidate ensemble is measured per round
(runtime proxy: certified A/B delta); T₀, α, θ are explicit
controller knobs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    EXPLORE = "explore"   # T high: ensemble wide, Jensen gap accrues
    EXPLOIT = "exploit"   # T low: cooldown/consolidation dominates


@dataclass(frozen=True)
class ThermostatConfig:
    """Explicit controller knobs (no hidden defaults)."""

    t0: float    # initial temperature T₀ > 0
    alpha: float  # cooling rate α ∈ (0, 1)
    theta: float  # exploration stop threshold θ ∈ (0, 1)


def validate(cfg: ThermostatConfig) -> None:
    if not cfg.t0 > 0:
        raise ValueError("t0 must be positive")
    if not 0.0 < cfg.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < cfg.theta < 1.0:
        raise ValueError("theta must be in (0, 1)")


def temperature(cfg: ThermostatConfig, t: int) -> float:
    """T_t = T₀·α^t."""
    if t < 0:
        raise ValueError("t must be nonnegative")
    return cfg.t0 * (cfg.alpha ** t)


def stop_horizon(cfg: ThermostatConfig, delta: float) -> int:
    """N = ⌈ln(δ/(T₀·ln(1/θ)))/ln α⌉ — exploration is lawfully
    exhausted for all t ≥ N (R388 ``thermostat_stop``).

    ``delta`` is the measured (h_emp_) energy gap between the
    incumbent and the competing candidates.
    """
    validate(cfg)
    if not delta > 0:
        raise ValueError("delta (energy gap) must be positive")
    c = delta / (cfg.t0 * math.log(1.0 / cfg.theta))
    # ln α < 0: ceil of a negative-denominator quotient
    return math.ceil(math.log(c) / math.log(cfg.alpha))


def competitor_mass(cfg: ThermostatConfig, delta: float, t: int) -> float:
    """Gibbs competitor mass e^{−δ/T_t} (R386: minimizer weight
    ≥ 1 − (n−1)·this). Crossing θ ends exploration by law.
    T_t → 0⁺ (float underflow at extreme t) is the annealing limit
    R387: the mass IS 0.0, not a division error."""
    t_t = temperature(cfg, t)
    if t_t <= 0.0:
        return 0.0
    return math.exp(-delta / t_t)


def phase(
    cfg: ThermostatConfig, delta: float, t: int
) -> tuple[Phase, float]:
    """Explore while e^{−δ/T_t} > θ; exploit from the horizon on.

    Returns (phase, competitor_mass) so callers log the physical
    stopping cause, not a heuristic one.
    """
    mass = competitor_mass(cfg, delta, t)
    return (Phase.EXPLOIT if mass <= cfg.theta else Phase.EXPLORE, mass)


def admission_temperature_scale(cfg: ThermostatConfig, t: int) -> float:
    """Temperature-scaled admission threshold multiplier: at high T
    the controller tolerates speculative entries, as T → 0 the
    admission gate tightens to pure certified gain
    (novelty ∧ gain, R273). Scale = T_t/T₀ ∈ (0, 1]."""
    return temperature(cfg, t) / cfg.t0


# --- R407/R408/R412 port (2026-10-14): phase point + concentration ---
# C:\primes\Hagi\Growth\Thermostat.lean: thermostat_phase_point,
# phase_concentration, consolidation_certificate.

def phase_point(cfg: ThermostatConfig, delta: float, j_gap: float) -> int:
    """Unified switching horizon N = max(stop horizon, merge-gate
    horizon) — `Thermostat.thermostat_phase_point` (R407): for all
    t ≥ N BOTH exploration has stopped (e^{−δ/T_t} ≤ θ) and any
    positive Jensen gap J passes the merge gate (J ≥ θ·T_t).

    ``j_gap``: the measured Jensen gap J > 0 of the candidate merge."""
    validate(cfg)
    if not delta > 0:
        raise ValueError("delta (energy gap) must be positive")
    if not j_gap > 0:
        raise ValueError("j_gap (Jensen gap) must be positive")
    n_stop = stop_horizon(cfg, delta)
    # merge gate: J >= theta * T_t  <=>  t >= ln(J/(theta*T0))/ln(alpha)
    # (ln alpha < 0; horizon is ceil of that quotient)
    n_gate = math.ceil(
        math.log(j_gap / (cfg.theta * cfg.t0)) / math.log(cfg.alpha)
    )
    return max(n_stop, n_gate)


def gibbs_share(energies, v_star: int, t: int, t0: float | None = None,
                alpha: float | None = None) -> float:
    """Gibbs weight of the minimizer at temperature T_t — premise of
    `Thermostat.phase_concentration` (R408): share ≥ 1 − C·θ after
    the phase point (C = number of non-minimizer states)."""
    tt = (t0 * (alpha ** t)) if (t0 and alpha) else t
    if tt <= 0.0:
        return 1.0
    m = min(energies)
    z = sum(math.exp(-(e - m) / tt) for e in energies)
    return math.exp(-(energies[v_star] - m) / tt) / z


def consolidation_certificate(cfg: ThermostatConfig, delta: float,
                              j_gap: float, energies,
                              v_star: int, t: int) -> dict:
    """3-in-1 controller capstone — `Thermostat.
    consolidation_certificate` (R412): past the phase point the
    physical mode "consolidation" holds and persists:
    (1) exploration stopped: e^{−δ/T_t} ≤ θ;
    (2) merge gate open: θ·T_t ≤ J;
    (3) Gibbs concentration: share(v*) ≥ 1 − C·θ (C = n−1)."""
    n = phase_point(cfg, delta, j_gap)
    stopped = competitor_mass(cfg, delta, t) <= cfg.theta
    gate_open = cfg.theta * temperature(cfg, t) <= j_gap
    share = gibbs_share(energies, v_star, temperature(cfg, t))
    c_count = len(energies) - 1
    concentrated = share >= 1.0 - c_count * cfg.theta
    return {
        "phase_point_N": n,
        "t": t,
        "past_phase_point": t >= n,
        "exploration_stopped": stopped,
        "merge_gate_open": gate_open,
        "share_v_star": share,
        "concentrated": concentrated,
        "consolidation": stopped and gate_open and concentrated and t >= n,
    }


# --- R510–R512 port (2026-10-16): GibbsFluctuations variance-stop -------
# C:\primes\Hagi\Probability\GibbsFluctuations.lean:
# gibbs_energy_variance_bound / thermostat_var_stop /
# thermostat_threshold / gibbs_var_steps_bound.
#
# The closing half of the thermostat circuit: R386–R388 stop EXPLORATION
# by law (minimizer Gibbs mass -> 1); R510–R512 stop VARIANCE by law:
# Var_{p_T}(E) <= Delta^2 * (n-1) * e^{-delta/T}, where delta is the
# measured gap from the minimizer to every non-minimizer state and
# Delta = max E - E(vstar) is the full energy range. The temperature at
# which the residual variance drops under eps is analytic, not tuned.


def variance_bound(delta: float, delta_range: float, n: int, t: float) -> float:
    """Var_{p_T}(E) <= Delta^2 * (n-1) * e^{-delta/T} [R498/R510].

    ``delta``: the measured (h_emp_) energy gap from the minimizer to
    every other state (min non-star gap, premise hgap); ``delta_range``:
    Delta = max_v E(v) - E(vstar); ``n``: number of states; ``t``:
    temperature. Float underflow at extreme cooling returns the
    annealing limit 0.0.
    """
    if delta <= 0.0:
        raise ValueError("delta (gap) must be positive")
    if delta_range <= 0.0:
        raise ValueError("delta_range must be positive")
    if n < 2:
        raise ValueError("need >= 2 states")
    if t <= 0.0:
        return 0.0
    return (delta_range * delta_range) * (n - 1) * math.exp(-delta / t)


def variance_stop_temperature(
    delta: float, delta_range: float, n: int, eps: float
) -> float:
    """T <= delta / ln(Delta^2 (n-1) / eps) makes Var <= eps
    [R510 ``thermostat_threshold``].

    The exact temperature below which the Gibbs ensemble's energy
    variance is certified under eps — an ANALYTIC cooldown target:
    cooling below it buys nothing (variance already bounded), staying
    above it leaves thermal spread uncertified.
    """
    if delta <= 0.0:
        raise ValueError("delta (gap) must be positive")
    if delta_range <= 0.0:
        raise ValueError("delta_range must be positive")
    if n < 2:
        raise ValueError("need >= 2 states")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    c = (delta_range * delta_range) * (n - 1)
    if c <= eps:
        # bound is already <= c <= eps at any T (trivial branch of the
        # theorem) — any positive temperature is safe
        return math.inf
    return delta / math.log(c / eps)


def variance_stop_steps(
    delta: float, delta_range: float, n: int, eps: float
) -> int:
    """k + 1 >= ln(Delta^2 (n-1) / eps) suffices on the schedule
    T_k = delta / (k+1) [R512 ``gibbs_var_steps_bound``].

    The number of cooldown steps after which the variance certificate
    holds — the horizon of the variance axis, in the same style as
    ``stop_horizon`` for the exploration axis.
    """
    t_stop = variance_stop_temperature(delta, delta_range, n, eps)
    if math.isinf(t_stop):
        return 0
    # T_k = delta/(k+1) <= T_stop  <=>  k + 1 >= delta / T_stop
    return max(0, math.ceil(delta / t_stop) - 1)


def variance_certificate(energies, t: float, eps: float) -> dict:
    """Measured variance-stop certificate at temperature ``t``.

    Computes the actual Gibbs variance of ``energies`` at T = t and the
    R510–R512 certificates: the analytic bound and the stop temperature
    for the target eps. Premises measured from the energy vector:
    delta = min adjacent gap, delta_range = full range.
    """
    if len(energies) < 2:
        raise ValueError("need >= 2 states")
    if t <= 0.0:
        raise ValueError("t must be positive")
    e = sorted(energies)
    delta = min(b - a for a, b in zip(e, e[1:]))
    full = e[-1] - e[0]
    if full <= 0.0:
        raise ValueError("degenerate energy vector")
    z = sum(math.exp(-(x - e[0]) / t) for x in e)
    p = [math.exp(-(x - e[0]) / t) / z for x in e]
    mu = sum(pi * x for pi, x in zip(p, e))
    var = sum(pi * (x - mu) ** 2 for pi, x in zip(p, e))
    bound = variance_bound(delta, full, len(e), t)
    return {
        "t": t,
        "var": var,
        "var_bound": bound,
        "bound_holds": var <= bound,
        "var_below_eps": var <= eps,
        "stop_temperature": variance_stop_temperature(
            delta, full, len(e), eps),
        "stop_steps": variance_stop_steps(delta, full, len(e), eps),
    }


# --- Gibbs + Nernst composite (R549, Unified/NernstFannes.lean) -------------


def gibbs_nernst_bound(n: int, delta: float, t: float) -> float:
    """S(p_T) <= (n-1) e^{-delta/T} ln n + (1 + (n-1) e^{-delta/T}) ln 2
    [R549 ``gibbs_nernst``].

    Third law with an explicit exponential rate: for a Gibbs
    distribution with energy gap delta around the minimizer v*, the
    Shannon entropy is bounded by t-hat = (n-1) e^{-delta/T} via
    gibbs_concentration + nernst_fannes + binary entropy <= ln 2.
    As T -> 0+ the bound decays exponentially to 0 — cooling is
    certified to drive the selector distribution to a point.

    Premises (the theorem's): energies on n >= 2 states with a gap
    delta > 0: E[v*] + delta <= E[v] for every v != v* (measured,
    h_emp_); T > 0.
    """
    if n < 2:
        raise ValueError("n must be >= 2 states")
    if delta <= 0.0:
        raise ValueError("delta (energy gap) must be positive")
    if t <= 0.0:
        raise ValueError("t must be positive")
    t_hat = (n - 1) * math.exp(-delta / t)
    return t_hat * math.log(n) + (1.0 + t_hat) * math.log(2.0)
