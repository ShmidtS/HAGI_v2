"""Tests for the probabilistic growth chain (R95) and the state bridge (R96)."""

from __future__ import annotations

import math
import random

import pytest

from hagi.train.growth_law import (
    additive_as_multiplicative,
    alpha_from_gain,
    deviation,
    horizon_needed,
    success_count_lower,
    takeoff_time_form,
)


# --- R95: success -> concentration -> growth ---------------------------


def test_success_count_lower_is_the_hoeffding_bound():
    """``p0*T - sqrt(2 T log(1/delta))``."""
    assert success_count_lower(0.3, 100, 0.05) == pytest.approx(
        30.0 - math.sqrt(200.0 * math.log(20.0))
    )
    assert deviation(100, 0.05) == pytest.approx(math.sqrt(200.0 * math.log(20.0)))


def test_relative_error_decays_as_one_over_sqrt_T():
    """Why a long horizon pays: the RELATIVE error shrinks like 1/sqrt(T)."""
    rate, delta = 0.3, 0.05
    errors = [
        deviation(t, delta) / (rate * t)
        for t in (10, 100, 1000, 10000)
    ]
    assert errors == sorted(errors, reverse=True)
    # Each tenfold increase in T divides the relative error by sqrt(10).
    for i in range(1, len(errors)):
        assert errors[i - 1] / errors[i] == pytest.approx(math.sqrt(10.0), rel=1e-6)


def test_exponent_grows_linearly_while_deviation_grows_like_sqrt_T():
    """``a*(p0 T - D)`` is dominated by the linear term as T grows.

    The exponent is exactly ``a*(p0*T - sqrt(2 T log(1/delta)))``, so
    10x T multiplies the linear term by 10 but the deviation only by
    sqrt(10). The ratio therefore approaches 10 from below rather than
    equalling it: at T=100 the deviation is still comparable to p0*T
    (24.5 against 30), and only by T=1000 does the linear term dominate
    cleanly.
    """
    def exponent(t: int) -> float:
        return takeoff_time_form(1.0, 0.3, 0.5, 0.0, t, 0.05)["exponent"]

    exps = [exponent(t) for t in (100, 1000, 10000)]
    assert exps == sorted(exps)
    # The linear scaling bound: exponent <= a*p0*T, and the deviation
    # is a strictly smaller subtraction from it.
    for t, e in zip((100, 1000, 10000), exps):
        assert e < 0.5 * 0.3 * t
        assert e == pytest.approx(0.5 * (0.3 * t - deviation(t, 0.05)))
    # The deviation's SHARE of the linear term shrinks like 1/sqrt(T).
    rel = [(0.5 * 0.3 * t - e) / (0.5 * 0.3 * t) for t, e in zip((100, 1000, 10000), exps)]
    assert rel == sorted(rel, reverse=True)
    for i in range(1, len(rel)):
        assert rel[i - 1] / rel[i] == pytest.approx(math.sqrt(10.0), rel=1e-6)


def test_takeoff_saturates_instead_of_overflowing():
    """exp(exponent) overflows past ~709; the API must not raise."""
    r = takeoff_time_form(1.0, 0.3, 0.5, 0.0, 10_000, 0.05)
    assert r["capability_lower"] == float("inf")
    assert math.isfinite(r["log_capability_lower"])
    assert r["exponent"] > 1000.0


def test_friction_reduces_the_exponent_exactly():
    a = takeoff_time_form(2.0, 0.4, 0.7, 0.0, 500, 0.05)["exponent"]
    b = takeoff_time_form(2.0, 0.4, 0.7, 3.0, 500, 0.05)["exponent"]
    assert a - b == pytest.approx(3.0)


def test_raising_the_success_rate_beats_raising_the_gain():
    """The review's ranking rule, read straight off the law.

    With ``a*p0`` as the objective, tripling the success rate shortens
    the horizon more than tripling the per-success gain does.
    """
    slow_rate = horizon_needed(1.0, 10.0, 0.3, 0.5, 0.05)
    fast_rate = horizon_needed(1.0, 10.0, 0.9, 0.5, 0.05)
    slow_gain = horizon_needed(1.0, 10.0, 0.3, 1.5, 0.05)
    assert fast_rate < slow_gain <= slow_rate
    assert slow_rate == 95


def test_horizon_needed_is_exact_and_minimal():
    c0, target, rate, a, delta = 1.0, 10.0, 0.5, 0.6, 0.05
    t = horizon_needed(c0, target, rate, a, delta)
    assert takeoff_time_form(c0, rate, a, 0.0, t, delta)["exponent"] >= math.log(target)
    if t > 1:
        assert takeoff_time_form(c0, rate, a, 0.0, t - 1, delta)["exponent"] < math.log(target)


def test_unreachable_target_is_reported():
    assert horizon_needed(1.0, 1e6, 0.3, 1e-6, 0.05) == -1


def test_short_horizons_give_a_vacuous_bound():
    """At small T the deviation exceeds the whole success count."""
    assert success_count_lower(0.3, 10, 0.05) < 0.0


# --- R96: the bridge from a measured additive gain to a multiplier ----


def test_additive_as_multiplicative_is_exact():
    """``C * (1 + g/C) == C + g``, with no assumption."""
    random.seed(30)
    for _ in range(20000):
        c = random.uniform(0.01, 1e6)
        g = random.uniform(-0.5 * c, 2.0 * c)
        assert c * additive_as_multiplicative(c, g) == pytest.approx(c + g, rel=1e-9)


def test_a_negative_gain_is_a_multiplier_below_one():
    assert additive_as_multiplicative(100.0, -5.0) == pytest.approx(0.95)
    assert alpha_from_gain(100.0, -5.0) == pytest.approx(-0.05)


def test_alpha_from_gain_matches_the_bridge():
    c, g = 250.0, 12.5
    assert additive_as_multiplicative(c, g) == pytest.approx(1.0 + alpha_from_gain(c, g))


def test_bridge_feeds_takeoff_with_a_measured_rate():
    """The measured gain becomes the takeoff rate with no free parameter."""
    c, g = 100.0, 10.0
    alpha = alpha_from_gain(c, g)
    r = takeoff_time_form(c, 1.0, 1.0, 0.0, 100, 0.05)
    # With p0=1, a=1 and no friction the count is just T minus deviation.
    assert r["exponent"] == pytest.approx(
        success_count_lower(1.0, 100, 0.05)
    )
    assert alpha == pytest.approx(0.1)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_bridge_rejects_nonpositive_capability(bad: float):
    with pytest.raises(ValueError):
        additive_as_multiplicative(bad, 1.0)
    with pytest.raises(ValueError):
        alpha_from_gain(bad, 1.0)


def test_success_count_rejects_bad_inputs():
    with pytest.raises(ValueError):
        success_count_lower(1.5, 10, 0.05)
    with pytest.raises(ValueError):
        success_count_lower(0.3, -1, 0.05)
    with pytest.raises(ValueError):
        success_count_lower(0.3, 10, 1.0)


def test_takeoff_rejects_bad_inputs():
    with pytest.raises(ValueError):
        takeoff_time_form(0.0, 0.3, 0.5, 0.0, 100, 0.05)
    with pytest.raises(ValueError):
        takeoff_time_form(1.0, 0.3, 0.0, 0.0, 100, 0.05)
    with pytest.raises(ValueError):
        takeoff_time_form(1.0, 0.3, 0.5, -1.0, 100, 0.05)
