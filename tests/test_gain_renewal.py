"""Tests for the gain-renewal law (R104).

R104 is the theorem that closes R102's bridge. R102 showed a fixed gain
buys only a bounded transient; R104 shows that PRODUCED gains lift the
window entirely, and -- via the converse -- that a bounded frontier makes
sustained growth impossible rather than merely unproven.

These tests pin the iff: sustained growth requires the frontier to scale
with capability, and that is checkable on telemetry.
"""

from __future__ import annotations

import math

import pytest

from hagi.train.gain_renewal import (
    RenewalParams,
    frontier_must_scale,
    gain_is_renewed,
    gate_is_supplied,
    harvest_is_the_only_source,
    renewal_floor,
    sustained_capability,
    sustained_growth_possible,
)


def params(gamma=1.0, rho=0.9, injection=0.1, decay=0.05):
    return RenewalParams(gamma, rho, injection, decay)


# --- the recurrence -----------------------------------------------------


def test_one_renewal_step_matches_the_recurrence():
    p = params(gamma=1.0, rho=0.9, injection=0.1, decay=0.05)
    # G_{t+1} >= rho*G_t + gamma*(inj - xi)
    assert p.renew(1.0) == pytest.approx(0.9 * 1.0 + 1.0 * 0.05)


def test_more_injection_gives_a_bigger_next_gain():
    low = params(injection=0.06, decay=0.05).renew(1.0)
    high = params(injection=0.50, decay=0.05).renew(1.0)
    assert high > low


def test_injection_below_decay_shrinks_the_gain():
    """The liveness condition of §4, in gain terms."""
    dying = params(rho=0.9, injection=0.01, decay=0.5).renew(1.0)
    assert dying < 0.9            # less than rho*G, so it decays


def test_a_negative_net_injection_is_reported_as_such():
    p = params(injection=0.01, decay=0.5)
    assert p.net_injection < 0
    assert p.injection_exceeds_decay is False


# --- the floor ----------------------------------------------------------


def test_the_gain_floor_is_positive_when_injection_beats_decay():
    p = params(gamma=1.0, rho=0.9, injection=0.1, decay=0.05)
    assert p.injection_exceeds_decay
    assert p.gain_floor == pytest.approx(1.0 * 0.05 / 0.1)


def test_the_floor_is_zero_when_injection_equals_decay():
    p = params(rho=0.9, injection=0.05, decay=0.05)
    assert p.gain_floor == pytest.approx(0.0)


def test_a_faster_decay_raises_the_floor():
    slow = renewal_floor(1.0, 0.5, 0.2, 0.1)
    fast = renewal_floor(1.0, 0.9, 0.2, 0.1)
    assert fast > slow


def test_rho_one_gives_linear_growth_instead_of_a_floor():
    p = params(gamma=1.0, rho=1.0, injection=0.2, decay=0.1)
    assert p.gain_floor == math.inf
    assert gain_is_renewed(p, 0.0, 5) == pytest.approx(5 * 1.0 * 0.1)


def test_the_gain_never_dies_under_a_positive_floor():
    """``gain_renewal_floor_pos``: gains do not die, at any horizon."""
    p = params(gamma=1.0, rho=0.9, injection=0.1, decay=0.05)
    g = 0.0
    for _ in range(1000):
        g = p.renew(g)
    assert g > 0.0
    # ... and it settles at the floor from below.
    assert g == pytest.approx(p.gain_floor, rel=1e-6)


def test_the_projection_converges_to_the_floor():
    p = params(gamma=1.0, rho=0.9, injection=0.1, decay=0.05)
    assert gain_is_renewed(p, 0.0, 500) == pytest.approx(p.gain_floor, rel=1e-6)


def test_projection_agrees_with_iterating():
    """The closed form and the recurrence must be the same function."""
    p = params(gamma=2.0, rho=0.8, injection=0.3, decay=0.1)
    g = 0.7
    for _ in range(12):
        g = p.renew(g)
    assert gain_is_renewed(p, 0.7, 12) == pytest.approx(g, rel=1e-9)


def test_zero_steps_leaves_the_gain_alone():
    p = params()
    assert gain_is_renewed(p, 1.25, 0) == pytest.approx(1.25)


# --- the harvest invariant ----------------------------------------------


def test_an_exact_harvest_passes_the_invariant():
    assert harvest_is_the_only_source(measured_gain=0.9, diversity=1.0,
                                      gamma=1.0) is True


def test_a_gain_larger_than_the_harvest_fails_the_invariant():
    """The diagnostic: gain the frontier does not explain."""
    assert harvest_is_the_only_source(measured_gain=1.5, diversity=1.0,
                                      gamma=1.0) is False


def test_a_smaller_harvest_rate_makes_the_invariant_harder():
    assert harvest_is_the_only_source(0.9, 1.0, gamma=1.0) is True
    assert harvest_is_the_only_source(0.9, 1.0, gamma=0.5) is False


# --- the lifted window --------------------------------------------------


def test_the_capability_is_certified_at_every_horizon():
    """``sustained_takeoff_window_lift``: no window, no bound."""
    for t in (0, 1, 5, 20, 100):
        assert sustained_capability(0.1, 1.0, t) == pytest.approx(1.1 ** t)


def test_the_lifted_capability_is_unbounded_where_r102_bounded_it():
    """The concrete difference the two rounds make.

    R102: a fixed gain's factor is capped at ``e * e^alpha`` forever.
    R104: under renewal the capability keeps multiplying by ``1+alpha``,
    so at T=1000 it is astronomically past anything R102 could certify.
    """
    alpha, c0 = 0.1, 1.0
    r102_bound = math.e * math.exp(alpha)
    assert sustained_capability(alpha, c0, 1000) > r102_bound


def test_a_larger_alpha_certifies_faster():
    assert sustained_capability(0.2, 1.0, 10) > \
        sustained_capability(0.1, 1.0, 10)


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        sustained_capability(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        sustained_capability(0.1, 0.0, 5)
    with pytest.raises(ValueError):
        sustained_capability(0.1, 1.0, -1)


# --- frontier scaling: the one open premise ----------------------------


def test_the_frontier_supplies_the_gate_when_it_scales():
    """``h_emp_frontier_scaling``, as a check rather than an assumption."""
    assert gate_is_supplied(alpha=0.1, capability=5.0, diversity=1.0,
                            gamma=1.0) is True


def test_a_stationary_frontier_stops_supplying_the_gate():
    """Exactly the bounded-frontier regime the converse rules out."""
    assert gate_is_supplied(alpha=0.1, capability=50.0, diversity=1.0,
                            gamma=1.0) is False


def test_a_growing_capability_eventually_outgrows_a_fixed_frontier():
    """Why the premise is not free, and why it must scale."""
    d_bar, gamma, alpha = 1.0, 1.0, 0.1
    for c in (1.0, 5.0, 9.0):
        assert gate_is_supplied(alpha, c, d_bar, gamma) is True
    assert gate_is_supplied(alpha, 10.1, d_bar, gamma) is False


def test_the_required_frontier_is_what_would_satisfy_the_gate():
    """The companion: not 'is it satisfied' but 'what would it take'."""
    c, alpha, gamma = 7.0, 0.1, 1.0
    need = frontier_must_scale(alpha, c, gamma)
    assert gate_is_supplied(alpha, c, need, gamma) is True
    # just below the requirement and the gate is not met
    assert gate_is_supplied(alpha, c, need * 0.99, gamma) is False


def test_the_requirement_grows_with_capability():
    """So the frontier must grow geometrically too."""
    small = frontier_must_scale(0.1, 1.0, 1.0)
    large = frontier_must_scale(0.1, 100.0, 1.0)
    assert large == pytest.approx(100.0 * small)


def test_a_bigger_gamma_asks_less_of_the_frontier():
    assert frontier_must_scale(0.1, 10.0, 1.0) > \
        frontier_must_scale(0.1, 10.0, 5.0)


# --- the converse -------------------------------------------------------


def test_a_bounded_frontier_caps_capability_exactly():
    """``bounded_frontier_no_sustained_growth``: the ceiling is exact."""
    assert sustained_growth_possible(alpha=0.1, gamma=1.0, d_bar=2.0) == \
        pytest.approx(20.0)


def test_the_ceiling_does_not_depend_on_the_horizon():
    """``C_t <= gamma*Dbar/alpha`` for EVERY t -- bounded forever."""
    ceiling = sustained_growth_possible(0.1, 1.0, 2.0)
    for t in (1, 10, 1000, 10**6):
        assert ceiling == pytest.approx(20.0)
        assert ceiling < 21.0


def test_widening_the_frontier_raises_the_ceiling():
    narrow = sustained_growth_possible(0.1, 1.0, 1.0)
    wide = sustained_growth_possible(0.1, 1.0, 10.0)
    assert wide == pytest.approx(10.0 * narrow)


def test_a_bigger_alpha_lowers_the_ceiling():
    assert sustained_growth_possible(0.2, 1.0, 1.0) < \
        sustained_growth_possible(0.1, 1.0, 1.0)


def test_invalid_ceiling_parameters_are_refused():
    with pytest.raises(ValueError):
        sustained_growth_possible(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        sustained_growth_possible(0.1, 0.0, 1.0)
    with pytest.raises(ValueError):
        sustained_growth_possible(0.1, 1.0, -1.0)


def test_the_ceiling_and_the_requirement_meet_exactly():
    """The iff, as one number: the frontier needed to sustain growth at
    capability ``C`` is precisely ``C / ceiling_per_unit``."""
    alpha, gamma, d_bar = 0.1, 2.0, 1.5
    ceiling = sustained_growth_possible(alpha, gamma, d_bar)
    # the gate is exactly on the boundary at capability == ceiling
    assert gate_is_supplied(alpha, ceiling, d_bar, gamma) is True
    assert gate_is_supplied(alpha, ceiling * 1.01, d_bar, gamma) is False