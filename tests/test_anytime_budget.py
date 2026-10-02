"""Tests for the anytime confidence budget (R93, Ville's inequality)."""

from __future__ import annotations

import math

import pytest

from hagi.train.anytime_budget import (
    admissible_schedule,
    anytime_alpha,
    anytime_probability_bound,
    budget_exhausts_uniformly,
    step_budget,
    total_spend,
)


@pytest.mark.parametrize("decay", [0.5, 0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("horizon", [1, 10, 100, 1000, 10_000, 1_000_000])
def test_budget_is_uniform_in_the_horizon(decay: float, horizon: int):
    """THE property: the total spend never exceeds delta, at any T.

    This is what "anytime-valid" means. The naive rule -- delta per
    step -- would need T*delta and becomes vacuous; the geometric
    schedule sums to at most delta for every horizon simultaneously.
    """
    total = 0.05
    d0 = admissible_schedule(total, decay)
    assert total_spend(d0, decay, horizon) <= total * (1 + 1e-12)


def test_naive_budget_exhausts_where_anytime_does_not():
    """The concrete saving: 400 steps at delta=0.05 each is 20x the budget."""
    assert budget_exhausts_uniformly(0.05, 0.05, 400)
    d0 = admissible_schedule(0.05, 0.9)
    assert total_spend(d0, 0.9, 400) <= 0.05


def test_admissible_schedule_saturates_the_budget():
    """delta_0/(1-rho) == delta exactly at the limit rho -> 1."""
    for decay in (0.5, 0.9, 0.99):
        d0 = admissible_schedule(0.05, decay)
        # The infinite sum is the budget.
        assert d0 / (1.0 - decay) == pytest.approx(0.05, rel=1e-12)


def test_closed_form_matches_the_literal_sum():
    d0 = admissible_schedule(0.05, 0.85)
    literal = sum(step_budget(d0, 0.85, t) for t in range(500))
    assert total_spend(d0, 0.85, 500) == pytest.approx(literal, rel=1e-12)


def test_early_steps_get_more_power():
    """rho < 1 means the budget front-loads onto the uncertain early steps."""
    d0 = admissible_schedule(0.05, 0.9)
    early = step_budget(d0, 0.9, 0)
    late = step_budget(d0, 0.9, 100)
    assert early > late > 0.0
    assert step_budget(d0, 0.9, 0) == pytest.approx(d0)


def test_step_budget_rejects_negative_step():
    with pytest.raises(ValueError):
        step_budget(0.01, 0.9, -1)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_admissible_schedule_rejects_bad_decay(bad: float):
    with pytest.raises(ValueError):
        admissible_schedule(0.05, bad)


def test_admissible_schedule_rejects_nonpositive_delta():
    with pytest.raises(ValueError):
        admissible_schedule(0.0, 0.9)


def test_ville_bound_is_the_threshold_ratio():
    """``P[exists t: M_t >= alpha] <= alpha / E[M_0]``."""
    assert anytime_probability_bound(1.0, 0.25) == pytest.approx(0.25)
    assert anytime_probability_bound(2.0, 1.0) == pytest.approx(0.5)
    # A threshold above E[M_0] makes the bound vacuous -- the caller must
    # tighten the schedule; the function reports it rather than hiding it.
    assert anytime_probability_bound(1.0, 4.0) == pytest.approx(4.0)


def test_anytime_alpha_matches_the_bound():
    """``alpha_t = delta_t * E[M_0]`` inverts to ``delta_t``."""
    for m0 in (0.5, 1.0, 7.5):
        for d in (1e-6, 0.01, 0.05):
            alpha = anytime_alpha(m0, d)
            assert anytime_probability_bound(m0, alpha) == pytest.approx(d, rel=1e-12)


def test_ville_rejects_nonpositive_start():
    with pytest.raises(ValueError):
        anytime_probability_bound(0.0, 0.5)
    with pytest.raises(ValueError):
        anytime_alpha(0.0, 0.5)


def test_zero_steps_spends_nothing():
    d0 = admissible_schedule(0.05, 0.9)
    assert total_spend(d0, 0.9, 0) == 0.0
    with pytest.raises(ValueError):
        total_spend(d0, 0.9, -1)


def test_decay_one_is_handled_explicitly():
    """rho == 1 would be 0/0 in the closed form; it is linear there."""
    d0 = admissible_schedule(0.05, 1.0 - 1e-16)
    assert math.isfinite(total_spend(d0, 1.0 - 1e-16, 10))
