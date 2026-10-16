"""Ports of the primes wave R510–R530 (2026-10-16).

- R510–R512 thermostat variance-stop (GibbsFluctuations.lean);
- R523–R525 anytime-safe threshold (CertifiedEstimator.lean);
- R522/R527/R528 condensation certificates (ThermoPrune /
  ThermoCondensation / AnytimeCondensation.lean);
- R529 risk-budget termination (Unified/TopLevel.lean);
- R530 gamma-deficit (Growth/GammaDeficit.lean).
"""

import math

import pytest

from hagi.model import formal
from hagi.train import certified_controller as cc
from hagi.train import thermostat as th
from hagi.train.stage_telescope import (
    StageCertificate,
    risk_budget_termination,
    termination_bound,
)


# --- R525: anytime_safe_threshold ----------------------------------------


def test_anytime_threshold_wider_than_single_round():
    # uniform over T rounds is strictly wider than one round (alpha)
    u = cc.anytime_safe_threshold(sigma2=1.0, T=100, alpha=0.05)
    assert u == pytest.approx(math.sqrt(2.0 * math.log(100 / 0.05)))
    assert u > math.sqrt(2.0 * 1.0 * math.log(1 / 0.05))


def test_anytime_accept_gate_clears_band():
    u = cc.anytime_safe_threshold(0.01, 50, 0.05)
    ok, band = cc.anytime_accept_gate(band + 1e-6, 0.01, 50, 0.05) \
        if (band := u) else (None, None)
    assert ok and band == pytest.approx(u)
    ok2, _ = cc.anytime_accept_gate(u * 0.5, 0.01, 50, 0.05)
    assert not ok2


def test_anytime_threshold_rejects_bad_args():
    with pytest.raises(ValueError):
        cc.anytime_safe_threshold(0.0, 10, 0.05)
    with pytest.raises(ValueError):
        cc.anytime_safe_threshold(1.0, 0, 0.05)
    with pytest.raises(ValueError):
        cc.anytime_safe_threshold(1.0, 10, 1.0)


# --- R528: anytime_condensation_cost --------------------------------------


def test_anytime_condensation_matches_band_square():
    sigma2, T, alpha = 0.5, 100, 0.05
    cost = formal.anytime_condensation_cost(sigma2, T, alpha)
    band = cc.anytime_safe_threshold(sigma2, T, alpha)
    assert cost == pytest.approx(band * band)


# --- R522/R527: condensation budget round-trip ----------------------------


def test_condensation_round_trip():
    assert formal.condensation_budget_cost(0.0) == 0.0  # trivial rebuild
    for m in (0.4, 1.0, 2.0):
        eps = formal.condensation_budget_cost(m)
        assert formal.condensation_allowed_deviation(eps) == pytest.approx(m)


def test_condensation_budget_quadratic():
    # M^2/4 scaling: doubling the deviation quadruples the cost
    assert formal.condensation_budget_cost(2.0) == pytest.approx(
        4.0 * formal.condensation_budget_cost(1.0)
    )


# --- R510–R512: thermostat variance-stop -----------------------------------


def test_variance_bound_holds_numerically():
    energies = [0.0, 1.0, 2.5, 2.7]
    for t in (0.05, 0.1, 0.5, 2.0):
        cert = th.variance_certificate(energies, t=t, eps=0.01)
        assert cert["bound_holds"], f"bound violated at T={t}"


def test_variance_stop_temperature_is_certified():
    energies = [0.0, 1.0, 2.5]
    e = sorted(energies)
    delta = min(b - a for a, b in zip(e, e[1:]))
    full = e[-1] - e[0]
    t_stop = th.variance_stop_temperature(delta, full, 3, 0.01)
    # at T = t_stop the analytic bound itself is <= eps (theorem)
    assert th.variance_bound(delta, full, 3, t_stop) <= 0.01 * (1 + 1e-9)


def test_variance_stop_steps_reach_certificate():
    energies = [0.0, 1.0, 2.5]
    e = sorted(energies)
    delta = min(b - a for a, b in zip(e, e[1:]))
    full = e[-1] - e[0]
    k = th.variance_stop_steps(delta, full, 3, 0.01)
    t_k = delta / (k + 1)
    assert th.variance_bound(delta, full, 3, t_k) <= 0.01 * (1 + 1e-9)


def test_variance_stop_trivial_branch():
    # c <= eps: any temperature is safe -> inf sentinel, 0 steps
    assert math.isinf(
        th.variance_stop_temperature(1.0, 0.01, 2, 1.0))
    assert th.variance_stop_steps(1.0, 0.01, 2, 1.0) == 0


# --- R529: risk_budget_termination ----------------------------------------


def _chain() -> list[StageCertificate]:
    return [
        StageCertificate(1, 4.7, 4.6, 0.10, 0.01, 100),
        StageCertificate(2, 4.6, 4.5, 0.10, 0.01, 100),
        StageCertificate(3, 4.5, 4.4, 0.10, 0.01, 100),
    ]


def test_risk_budget_horizon_formula():
    # T <= (Phi0 - Emin + R)/eps = (4.7 - 4.0 + 0.05)/0.1 = 7.5
    assert risk_budget_termination(_chain(), 4.0, 0.1, 0.05) == \
        pytest.approx(7.5)


def test_risk_budget_stronger_than_uniform_r():
    # R529 puts risk in the numerator linearly; compare with the
    # uniform-r form termination_bound((eps - r) denominator)
    chain = _chain()
    t_risk = risk_budget_termination(chain, 4.0, 0.1, 0.05)
    t_unif = termination_bound(chain, 4.0, 0.1, 0.1 / 3)
    # uniform-r needs eps > r; with total risk 0.03 spread over 3 gens
    # the per-gen r = 0.01; horizon (4.7-4.0)/(0.1-0.01) ~ 7.78 > 7.5
    assert t_risk < t_unif


def test_risk_budget_rejects_underperformance():
    with pytest.raises(ValueError, match="decrease"):
        risk_budget_termination(_chain(), 4.0, 0.2, 0.05)
    with pytest.raises(ValueError, match="risk"):
        risk_budget_termination(_chain(), 4.0, 0.1, 0.02)


# --- R530: gamma-deficit ----------------------------------------------------


def test_gamma_min_for_target():
    assert cc.gamma_min_for_target(0.5, 2.0, 1.0) == pytest.approx(1.0)


def test_gamma_deficit_verdict():
    deficit, req = cc.gamma_deficit_no_takeoff(0.1, 0.5, 2.0, 1.0)
    assert deficit and req == pytest.approx(1.0)
    deficit2, _ = cc.gamma_deficit_no_takeoff(1.0, 0.5, 2.0, 1.0)
    assert not deficit2


def test_gamma_ceiling_below_target_blocks_takeoff():
    # gamma * Dbar / alpha < C*  <=>  deficit (contrapositive pair)
    ceiling = cc.gamma_ceiling(0.1, 1.0, 0.5)
    assert ceiling == pytest.approx(0.2)
    assert ceiling < 2.0  # target unreachable: no takeoff by law


def test_gamma_args_validated():
    for bad in (
        lambda: cc.gamma_min_for_target(0.0, 1.0, 1.0),
        lambda: cc.gamma_min_for_target(0.5, 0.0, 1.0),
        lambda: cc.gamma_min_for_target(0.5, 1.0, 0.0),
        lambda: cc.gamma_ceiling(-1.0, 1.0, 0.5),
    ):
        with pytest.raises(ValueError):
            bad()
