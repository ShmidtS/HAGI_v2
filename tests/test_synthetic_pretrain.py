"""Tests for the synthetic-pretraining gate (R99)."""

from __future__ import annotations

import math

import pytest

from hagi.train.synthetic_pretrain import (
    SyntheticInvestment,
    capability_per_compute,
    required_step_savings,
    select_task,
    synthetic_phase_pays,
    time_to_capability,
)


# --- synth_investment_dominates ----------------------------------------


def test_gate_is_a_single_comparison():
    """``dT * c_step > C_synth`` -- nothing tuned, nothing else."""
    assert synthetic_phase_pays(SyntheticInvestment(500, 1.0, 400))
    assert not synthetic_phase_pays(SyntheticInvestment(300, 1.0, 400))


def test_tie_is_excluded_so_the_simpler_schedule_wins():
    """Strict ``>``: on a tie the two are equal per compute, so skip."""
    assert not synthetic_phase_pays(SyntheticInvestment(400, 1.0, 400))


def test_break_even_is_reported_as_a_step_count():
    inv = SyntheticInvestment(0, 1.0, 400.0)
    assert required_step_savings(inv) == pytest.approx(400.0)
    # Exactly break-even does not pay; one step more does.
    assert not synthetic_phase_pays(SyntheticInvestment(400, 1.0, 400.0))
    assert synthetic_phase_pays(SyntheticInvestment(401, 1.0, 400.0))


def test_zero_step_cost_makes_the_break_even_infinite():
    assert required_step_savings(SyntheticInvestment(0, 0.0, 400.0)) == math.inf


def test_net_gain_and_saved_compute_agree_with_the_gate():
    inv = SyntheticInvestment(500, 2.0, 900.0)
    assert inv.saved_compute == pytest.approx(1000.0)
    assert inv.net_gain == pytest.approx(100.0)
    assert synthetic_phase_pays(inv)


def test_capability_per_compute_is_the_optimized_quantity():
    assert capability_per_compute(10.0, 2.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        capability_per_compute(0.0, 1.0)
    with pytest.raises(ValueError):
        capability_per_compute(1.0, 0.0)


def test_investment_rejects_nonsense():
    with pytest.raises(ValueError):
        SyntheticInvestment(-1, 1.0, 1.0)
    with pytest.raises(ValueError):
        SyntheticInvestment(1, -1.0, 1.0)
    with pytest.raises(ValueError):
        SyntheticInvestment(1, 1.0, -1.0)


# --- TimeToCapability ---------------------------------------------------


def test_time_to_capability_is_the_first_crossing():
    assert time_to_capability([1, 2, 3, 5, 6], 5.0) == 3
    assert time_to_capability([5, 6], 5.0) == 0
    assert time_to_capability([1, 2, 3], 5.0) == -1


def test_never_reaching_is_distinct_from_reaching_late():
    """"-1" means the schedule fails; more time is not the fix."""
    assert time_to_capability([1, 2, 3], 9.0) == -1
    assert time_to_capability([1, 2, 9], 9.0) == 2


def test_monotonicity_is_verified_not_assumed():
    """The theorem needs a monotone curve; a non-monotone one is a bug
    in the data, not something to silently tolerate."""
    with pytest.raises(ValueError):
        time_to_capability([1, 5, 3], 2.0)
    with pytest.raises(ValueError):
        time_to_capability([1, 2], -1.0)


def test_a_monotone_curve_is_accepted_at_full_length():
    curve = [float(i) for i in range(100)]
    assert time_to_capability(curve, 50.0) == 50


# --- task_selection_marginal --------------------------------------------


def test_task_selection_takes_the_best_marginal_ratio():
    # 'a' has the smallest gain but by far the cheapest, so g/c wins.
    assert select_task({"a": 3.0, "b": 10.0, "c": 5.0},
                       {"a": 1.0, "b": 10.0, "c": 5.0}) == ("a", 3.0)


def test_task_selection_is_the_finite_argmax():
    gains = {"x": 4.0, "y": 8.0, "z": 6.0}
    costs = {"x": 8.0, "y": 4.0, "z": 3.0}
    task, ratio = select_task(gains, costs)
    assert task == max(gains, key=lambda k: gains[k] / costs[k])
    assert ratio == pytest.approx(2.0)


def test_task_selection_rejects_malformed_candidates():
    with pytest.raises(ValueError):
        select_task({}, {})
    with pytest.raises(ValueError):
        select_task({"a": 1.0}, {})          # no cost
    with pytest.raises(ValueError):
        select_task({"a": -1.0}, {"a": 1.0})
    with pytest.raises(ValueError):
        select_task({"a": 1.0}, {"a": 0.0})