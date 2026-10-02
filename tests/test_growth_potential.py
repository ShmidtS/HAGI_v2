"""Tests for the generation potential (R91).

R91 composes four stage lemmas into one bound on ``Φ = energy +
protectedRisk``. Two properties matter and are pinned here:

  - the bound is a SUM OF NAMED TERMS, so the stages are tradeable
    against each other rather than opaque;
  - it is CONDITIONAL on empirical premises, and says so. A theorem that
    composes premises it does not prove must not let a caller read the
    conclusion as unconditional.
"""

from __future__ import annotations

import pytest

from hagi.train.growth_potential import (
    CycleBound,
    GrowthState,
    compression_cost,
    cycle_bound,
    energy_only_is_not_a_potential,
    generations_to_epsilon,
)


def state(**kw) -> GrowthState:
    base = dict(energy=3.0, protected_risk=0.5, gap=8.0, quant_err=0.1)
    base.update(kw)
    return GrowthState(**base)


# --- the potential ------------------------------------------------------


def test_the_potential_is_energy_plus_protected_risk():
    s = state(energy=3.0, protected_risk=0.5)
    assert s.potential == pytest.approx(3.5)


def test_energy_alone_is_not_a_potential():
    """The reason the terms are summed rather than tracked separately."""
    assert energy_only_is_not_a_potential() is False


def test_risk_alone_is_not_either():
    """Symmetric statement, on the other term."""
    low_energy = state(energy=1.0, protected_risk=9.0)
    high_energy = state(energy=1.2, protected_risk=0.1)
    # energy went UP but the potential went DOWN: neither term alone
    # would have flagged this as progress.
    assert high_energy.energy > low_energy.energy
    assert high_energy.potential < low_energy.potential


# --- the composed bound -------------------------------------------------


def test_the_bound_is_the_minus_gains_plus_costs():
    s = state()
    b = cycle_bound(s, grow_gain=0.2, joint_gain=0.05)
    assert b.delta == pytest.approx(-b.total_gain + b.total_cost)
    assert b.total_gain == pytest.approx(0.2 + 8.0 + 0.05)


def test_the_merge_gain_is_the_gap():
    """The merge stage's certified gain IS the ensemble disagreement."""
    s = state(gap=8.0)
    b = cycle_bound(s, grow_gain=0.0, joint_gain=0.0)
    assert b.merge_gain == pytest.approx(8.0)


def test_a_large_gap_dominates_the_bound():
    """Which is the point: the frontier is the generation's main gain.

    Compared at equal COST, so the only thing varying is the gap. With
    zero costs any non-negative gap descends, which is trivially true;
    the informative comparison holds the risks fixed.
    """
    risk = dict(grow_risk=5.0, merge_risk=5.0)
    small = cycle_bound(state(gap=0.1, **risk), 0.0, 0.0)
    large = cycle_bound(state(gap=18.0, **risk), 0.0, 0.0)
    assert small.descends is False
    assert large.descends is True
    assert large.delta < small.delta


def test_a_zero_gap_with_no_cost_is_neutral():
    """delta = 0 counts as descending: no progress, no regression."""
    b = cycle_bound(state(gap=0.0), 0.0, 0.0)
    assert b.delta == pytest.approx(0.0)
    assert b.descends is True


def test_each_named_term_shows_up_in_the_total():
    s = state(grow_risk=0.1, merge_risk=0.2, joint_risk=0.3,
              compress_risk=0.05, quant_err=0.4)
    b = cycle_bound(s, 0.0, 0.0, kappa=2.0, grid=0.5)
    # kappa*s/2 = 0.5, plus the four risks
    assert b.total_cost == pytest.approx(0.1 + 0.2 + 0.3 + 0.5 + 0.05)


def test_descends_is_exactly_delta_at_most_zero():
    for gain, risk in ((1.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)):
        s = state(gap=0.0, grow_risk=risk)
        b = cycle_bound(s, gain, 0.0)
        assert b.descends is (b.delta <= 0.0)


# --- honesty about the premises -----------------------------------------


def test_a_bound_without_premises_is_marked_conditional():
    """R91 proves the composition, not the stage premises."""
    b = cycle_bound(state(), 0.2, 0.05)
    assert b.is_conditional is False     # nothing claimed to be measured
    b2 = cycle_bound(state(), 0.2, 0.05,
                     premises={"h_emp_grow", "h_emp_merge"})
    assert b2.is_conditional is True
    assert "h_emp_grow" in b2.premises


def test_the_premises_are_recorded_verbatim():
    p = {"h_emp_grow", "h_emp_merge", "h_emp_smooth", "h_emp_lip"}
    b = cycle_bound(state(), 1.0, 0.1, premises=p)
    assert set(b.premises) == p


# --- the telescope ------------------------------------------------------


def test_the_horizon_needs_a_descent_at_least_epsilon():
    b = cycle_bound(state(gap=10.0), 0.5, 0.1)
    assert generations_to_epsilon(b, potential0=10.0, epsilon=0.25) == \
        pytest.approx(40.0)


def test_an_epsilon_larger_than_the_descent_is_refused():
    """Otherwise the telescope would promise more than it certifies."""
    b = cycle_bound(state(gap=0.5), 0.01, 0.0)
    with pytest.raises(ValueError, match="certifies a descent"):
        generations_to_epsilon(b, potential0=10.0, epsilon=1.0)


def test_an_increasing_bound_cannot_reach_a_floor():
    b = cycle_bound(state(gap=0.0), 0.0, 0.0)
    with pytest.raises(ValueError):
        generations_to_epsilon(b, potential0=10.0, epsilon=0.1)


# --- guards -------------------------------------------------------------


def test_a_negative_gap_is_refused():
    with pytest.raises(ValueError, match="gap"):
        state(gap=-1.0)


def test_an_out_of_range_quant_error_is_refused():
    with pytest.raises(ValueError):
        state(quant_err=0.6)
    with pytest.raises(ValueError):
        state(quant_err=-0.1)


def test_negative_risks_are_refused():
    for field in ("grow_risk", "merge_risk", "joint_risk", "compress_risk"):
        with pytest.raises(ValueError):
            state(**{field: -1.0})


def test_negative_gains_are_refused():
    s = state()
    with pytest.raises(ValueError):
        cycle_bound(s, grow_gain=-1.0, joint_gain=0.0)
    with pytest.raises(ValueError):
        cycle_bound(s, grow_gain=0.0, joint_gain=-1.0)


def test_compression_cost_is_the_bounded_form():
    """``κs/2 + risk``. The honest sqrt(n) form lives in ternary_exact;
    this one assumes the caller folded saturation into quant_err."""
    assert compression_cost(kappa=2.0, grid=0.5, quant_err=0.1) == \
        pytest.approx(0.5)
    assert compression_cost(2.0, 0.5, 0.1, compress_risk=0.25) == \
        pytest.approx(0.75)


def test_negative_compression_terms_are_refused():
    with pytest.raises(ValueError):
        compression_cost(-1.0, 0.5, 0.1)
    with pytest.raises(ValueError):
        compression_cost(1.0, -0.5, 0.1)