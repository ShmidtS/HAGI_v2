"""Tests for the frontier cone (R107).

R104 left one premise open -- ``h_emp_frontier_scaling`` -- and R107
closes it by making the cone INVARIANT under a production dynamic. The
properties worth pinning:

  - the threshold is a computable number from four measured constants,
    so the regime is decidable rather than asserted;
  - the CAP the audit's sketch left implicit is load-bearing: a
    capability jump that outruns the frontier breaks the cone, and the
    formalisation found that rather than proving the sketch as drawn;
  - the cone can FAIL, and the tests exercise that. A cone test that
    only checks the happy path cannot catch the project's own regime.
"""

from __future__ import annotations

import math

import pytest

from hagi.train.frontier_cone import (
    ConeState,
    certified_capability,
    cone_holds,
    cone_step,
    cone_threshold,
    diagnose,
    frontier_ratio,
    regime,
    settled_gain,
)


# --- the threshold ------------------------------------------------------


def test_the_threshold_is_computable_from_four_constants():
    """An operational criterion, not a slogan."""
    assert cone_threshold(0.1, 0.002, 0.9, 0.0) == pytest.approx(10.0)
    assert cone_threshold(0.1, 0.1, 0.9, 0.0) == pytest.approx(0.2)


def test_friction_raises_the_threshold():
    assert cone_threshold(0.1, 0.1, 0.9, 1.0) > \
        cone_threshold(0.1, 0.1, 0.9, 0.0)


def test_a_faster_harvest_lowers_the_threshold():
    """gamma in the denominator: harvesting more needs less production."""
    assert cone_threshold(0.1, 0.1, 0.9, 0.0) > \
        cone_threshold(0.1, 1.0, 0.9, 0.0)


def test_retention_lowers_the_threshold():
    assert cone_threshold(0.1, 0.1, 0.5, 0.0) > \
        cone_threshold(0.1, 0.1, 0.9, 0.0)


def test_a_non_positive_alpha_or_gamma_is_refused():
    with pytest.raises(ValueError):
        cone_threshold(0.0, 1.0, 0.9, 0.0)
    with pytest.raises(ValueError):
        cone_threshold(0.1, 0.0, 0.9, 0.0)


# --- the cone itself ----------------------------------------------------


def test_the_cone_is_the_alpha_over_gamma_line():
    k = 0.1 / 0.002
    assert cone_holds(0.1, 0.002, 3.3, 18.25) is False   # 5.5 < 50
    assert cone_holds(0.1, 0.002, 0.36, 18.25) is True    # 50.7 >= 50


def test_the_projects_own_measurements_are_outside_the_cone():
    """The regime this project is actually in, pinned as a fact.

    D = 18.25 nats, C = 3.3, gamma = 0.002 -> D/C = 5.53 against a
    boundary of alpha/gamma = 50. So R107's cone does NOT hold, and
    R104's converse caps capability. Recording it here means a future
    improvement has something to compare against.
    """
    assert cone_holds(0.1, 0.002, 3.3, 18.25) is False
    assert frontier_ratio(0.1, 0.002, 3.3, 18.25) == pytest.approx(
        0.1106, abs=1e-3
    )


def test_the_ratio_is_one_on_the_boundary():
    alpha, gamma = 0.1, 0.002
    k = alpha / gamma
    assert frontier_ratio(alpha, gamma, 3.3, k * 3.3) == pytest.approx(1.0)


def test_a_larger_frontier_raises_the_ratio():
    assert frontier_ratio(0.1, 0.1, 1.0, 10.0) > \
        frontier_ratio(0.1, 0.1, 1.0, 1.0)


def test_a_larger_capability_lowers_the_ratio():
    assert frontier_ratio(0.1, 0.1, 2.0, 1.0) < \
        frontier_ratio(0.1, 0.1, 1.0, 1.0)


def test_zero_capability_is_infinite_ratio():
    assert frontier_ratio(0.1, 0.1, 0.0, 1.0) == math.inf


def test_negative_inputs_are_refused():
    with pytest.raises(ValueError):
        frontier_ratio(0.1, 0.1, -1.0, 1.0)
    with pytest.raises(ValueError):
        frontier_ratio(0.1, 0.1, 1.0, -1.0)


# --- the step is an INVARIANT -------------------------------------------


def test_one_step_preserves_the_cone_when_the_threshold_is_met():
    alpha, gamma, rho = 0.1, 1.0, 0.5
    c, d = 1.0, 10.0                 # D = 10 >= (0.1/1.0)*1 = 0.1
    beta = cone_threshold(alpha, gamma, rho, 0.0) + 1.0
    c_next = (1.0 + alpha) * c
    assert cone_step(alpha, gamma, rho, beta, c, d, 0.0, c_next) is True


def test_the_cap_is_load_bearing():
    """The audit's sketch left this implicit; the formalisation found it.

    A capability jump beyond ``(1+alpha)·C_t`` breaks the cone no matter
    how fast the frontier is produced.
    """
    alpha, gamma, rho = 0.1, 1.0, 0.5
    c, d = 1.0, 10.0
    beta = 1e6                       # produce as much as you like
    too_big = (1.0 + alpha) * c * 1.5
    assert cone_step(alpha, gamma, rho, beta, c, d, 0.0, too_big) is False


def test_insufficient_production_breaks_the_cone():
    alpha, gamma, rho = 0.1, 1.0, 0.5
    c, d = 1.0, 10.0
    weak = cone_threshold(alpha, gamma, rho, 0.0) - 1.0
    c_next = (1.0 + alpha) * c
    assert cone_step(alpha, gamma, rho, weak, c, d, 0.0, c_next) is False


def test_a_cone_that_does_not_hold_cannot_be_stepped():
    """There is no point checking the step from outside the cone."""
    assert cone_step(0.1, 1.0, 0.5, 1e6, 1.0, 0.01, 0.0, 1.1) is False


def test_friction_can_break_the_cone():
    alpha, gamma, rho = 0.1, 1.0, 0.5
    c, d = 1.0, 10.0
    beta = cone_threshold(alpha, gamma, rho, 0.0)
    c_next = (1.0 + alpha) * c
    # enough friction that the frontier cannot survive the step
    assert cone_step(alpha, gamma, rho, beta, c, d, 1e6, c_next) is False


# --- the regime verdict -------------------------------------------------


def test_the_verdict_follows_the_cone():
    assert regime(True, 10.0, 10.0) == "sustained"
    assert regime(True, 1.0, 10.0) == "capped"
    assert regime(False, 1e6, 0.0) == "capped"


def test_the_diagnostic_reports_both_numbers():
    """A controller must be able to tell 'thin frontier' from 'slow
    production' -- they need different responses."""
    st = diagnose(alpha=0.1, gamma=0.002, rho=0.9, beta=0.0,
                  capability=3.3, frontier=18.25, friction=0.0)
    assert isinstance(st, ConeState)
    assert st.ratio < 1.0
    assert st.threshold == pytest.approx(10.0)
    assert st.verdict == "capped"


def test_a_healthy_state_reports_sustained():
    st = diagnose(alpha=0.1, gamma=1.0, rho=0.5, beta=1.0,
                  capability=1.0, frontier=10.0, friction=0.0)
    assert st.inside_cone is True
    assert st.verdict == "sustained"


def test_the_diagnostic_needs_positive_capability():
    with pytest.raises(ValueError):
        diagnose(0.1, 0.1, 0.5, 1.0, 0.0, 10.0, 0.0)


# --- the certificate and the settled gain -------------------------------


def test_the_capability_certificate_has_no_empirical_premise():
    """``sustained_takeoff_from_production``: derived, not assumed."""
    for t in (0, 1, 10, 100):
        assert certified_capability(0.1, 1.0, t) == pytest.approx(1.1 ** t)


def test_the_certificate_is_unbounded():
    """The contrast with R102's bounded transient, which is the point."""
    assert certified_capability(0.1, 1.0, 10_000) > math.e * math.exp(0.1)


def test_the_settled_gain_is_exactly_the_gate_rate():
    """Under exact additive production the system runs AT the certified
    rate -- the certificate's overlay does not permit overshoot."""
    assert settled_gain(0.1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        settled_gain(0.0)


def test_certificate_parameters_are_checked():
    with pytest.raises(ValueError):
        certified_capability(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        certified_capability(0.1, 0.0, 5)
    with pytest.raises(ValueError):
        certified_capability(0.1, 1.0, -1)