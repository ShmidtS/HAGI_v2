"""Tests: selector thermostat (Growth/Thermostat.lean R387–R388)."""
import math

import pytest

from hagi.train.thermostat import (
    Phase,
    ThermostatConfig,
    admission_temperature_scale,
    competitor_mass,
    phase,
    stop_horizon,
    temperature,
    validate,
)


def _cfg():
    return ThermostatConfig(t0=1.0, alpha=0.7, theta=0.05)


def test_validate_rejects_bad_knobs():
    for bad in [
        ThermostatConfig(0.0, 0.7, 0.05),
        ThermostatConfig(1.0, 1.0, 0.05),
        ThermostatConfig(1.0, 0.7, 0.0),
        ThermostatConfig(1.0, 0.7, 1.0),
    ]:
        with pytest.raises(ValueError):
            validate(bad)


def test_temperature_schedule():
    cfg = _cfg()
    assert temperature(cfg, 0) == pytest.approx(1.0)
    assert temperature(cfg, 3) == pytest.approx(0.7**3)
    with pytest.raises(ValueError):
        temperature(cfg, -1)


def test_stop_horizon_certificate():
    # R388: for all t >= N, e^{-delta/(T0*alpha^t)} <= theta
    cfg = _cfg()
    delta = 0.5
    n = stop_horizon(cfg, delta)
    assert n >= 0
    for t in range(n, n + 5):
        assert competitor_mass(cfg, delta, t) <= cfg.theta + 1e-12
    # and N is tight: just before N exploration may still be alive
    if n > 0:
        assert competitor_mass(cfg, delta, n - 1) > cfg.theta


def test_phase_transitions_once():
    cfg = _cfg()
    delta = 0.5
    n = stop_horizon(cfg, delta)
    states = [phase(cfg, delta, t)[0] for t in range(n + 3)]
    # monotone: EXPLORE...EXPLORE then EXPLOIT forever (cooling)
    first_exploit = states.index(Phase.EXPLOIT)
    assert all(s is Phase.EXPLOIT for s in states[first_exploit:])
    assert first_exploit <= n


def test_horizon_grows_as_gap_shrinks():
    # small energy gap -> competitors stay hot longer -> later stop
    cfg = _cfg()
    assert stop_horizon(cfg, 0.1) >= stop_horizon(cfg, 1.0)


def test_admission_scale_decays_to_zero():
    cfg = _cfg()
    s0 = admission_temperature_scale(cfg, 0)
    assert s0 == pytest.approx(1.0)
    assert admission_temperature_scale(cfg, 50) < 1e-7


def test_horizon_matches_direct_solve():
    # closed form vs brute force
    cfg = ThermostatConfig(t0=2.0, alpha=0.9, theta=0.01)
    delta = 0.3
    n = stop_horizon(cfg, delta)
    brute = next(
        t
        for t in range(10000)
        if math.exp(-delta / temperature(cfg, t)) <= cfg.theta
    )
    assert n == brute
