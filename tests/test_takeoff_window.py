"""Tests for the certified takeoff window and honest compression cost (R102).

R102 is the most important round for the growth loop, and it is a
NEGATIVE result stated as a theorem: with a FIXED gain the certified
takeoff is a bounded transient. These tests pin that bound so "unbounded
exponential growth" cannot be read back into the code, and pin the
sqrt(n) that R102 FIX 1 restored to the compression cost.
"""

from __future__ import annotations

import math

import pytest

from hagi.train.takeoff_window import (
    TakeoffWindow,
    compression_cost,
    cycle_budget,
    generations_to_floor,
    positive_cost_required,
    success_ceiling_reached,
    suggested_form_is_not_implied,
    sustained_takeoff_needs,
    sustained_takeoff_possible,
    takeoff_window,
    window_is_empty,
)


# --- the window is ABSOLUTE ---------------------------------------------


def test_the_success_ceiling_does_not_depend_on_the_horizon():
    """The theorem's core: the bound is uniform in T.

    This is what "anytime" buys and what makes it a stopping rule rather
    than a forecast.
    """
    w = takeoff_window(0.05, 1.0, 1.0)
    assert takeoff_window(0.05, 1.0, 1.0, horizon=1).max_successes == \
        pytest.approx(w.max_successes)
    assert takeoff_window(0.05, 1.0, 1.0, horizon=10**6).max_successes == \
        pytest.approx(w.max_successes)


def test_a_worked_example_reproduces_the_theorems_numbers():
    """alpha=0.05, C0=1, G=1: ceiling 20, factor e."""
    w = takeoff_window(0.05, 1.0, 1.0)
    assert w.max_successes == pytest.approx(20.0)
    assert w.certified_factor_bound == pytest.approx(math.e, rel=1e-12)


def test_the_factor_is_bounded_by_e_times_e_alpha():
    """``exp(1 + α − αC₀/G) ≤ e·e^α`` -- the theorem's stated bound."""
    for alpha, c0, gain in ((0.05, 1.0, 1.0), (0.2, 3.0, 4.0), (0.01, 2.0, 2.0)):
        w = takeoff_window(alpha, c0, gain)
        assert w.certified_factor_bound <= math.e * math.exp(alpha) + 1e-9


def test_the_factor_is_finite_for_every_admissible_window():
    for alpha in (0.01, 0.05, 0.1, 0.5, 0.9):
        w = takeoff_window(alpha, 1.0, 10.0)
        assert w.is_bounded_transient
        assert math.isfinite(w.certified_factor_bound)


def test_a_larger_gain_opens_a_wider_window():
    """The mechanism's parameter that actually matters."""
    narrow = takeoff_window(0.05, 1.0, 1.0)
    wide = takeoff_window(0.05, 1.0, 2.0)
    assert wide.max_successes > narrow.max_successes
    assert wide.certified_factor_bound > narrow.certified_factor_bound


def test_a_smaller_alpha_opens_a_wider_window():
    tight = takeoff_window(0.1, 1.0, 1.0)
    loose = takeoff_window(0.05, 1.0, 1.0)
    assert loose.max_successes > tight.max_successes


def test_a_failed_gate_at_zero_means_an_empty_window():
    """``α·C₀ > G``: no cycle can EVER succeed, so the count is zero."""
    w = takeoff_window(0.5, 10.0, 1.0)
    assert not w.gate_at_zero_holds
    assert window_is_empty(w)
    assert w.max_successes < 0.0


def test_a_window_that_opens_at_zero_is_not_empty():
    w = takeoff_window(0.05, 1.0, 1.0)
    assert w.gate_at_zero_holds
    assert not window_is_empty(w)


# --- the stopping rule ---------------------------------------------------


def test_the_ceiling_is_a_stopping_rule_for_a_mechanism():
    """Once spent, a fixed-gain channel is finished -- not 'maybe'."""
    w = takeoff_window(0.05, 1.0, 1.0)
    assert success_ceiling_reached(5.0, w) is False
    assert success_ceiling_reached(w.max_successes, w) is True
    assert success_ceiling_reached(w.max_successes + 5.0, w) is True


def test_the_ceiling_is_zero_for_an_impossible_window():
    w = takeoff_window(0.5, 10.0, 1.0)
    assert success_ceiling_reached(0.0, w) is True


# --- the suggested form is NOT provable ----------------------------------


def test_the_suggested_form_is_flagged_as_unproved():
    """The audit proposed ``e^{G/C₀}``; the theorem proves something else.

    Where the suggested form would be TIGHTER it is claiming more than
    was proved, and this reports it as a number rather than leaving the
    discrepancy in a comment where a later reader might "fix" the code to
    match the optimistic form.
    """
    small_ratio = takeoff_window(0.2, 1.0, 0.5)   # G/C0 = 0.5
    assert suggested_form_is_not_implied(small_ratio) is True


def test_the_suggested_form_is_not_always_tighter():
    big_ratio = takeoff_window(0.2, 1.0, 20.0)   # G/C0 = 20
    assert suggested_form_is_not_implied(big_ratio) is False


def test_the_proved_form_is_never_looser_than_its_stated_bound():
    w = takeoff_window(0.2, 1.0, 0.5)
    assert w.certified_factor_bound <= math.e * math.exp(0.2) + 1e-9


# --- sustained takeoff needs a growing gain ------------------------------


def test_sustained_takeoff_becomes_impossible_past_one_over_alpha():
    """The forced part of the open bridge.

    A success at cycle t needs ``G ≥ α(C₀ + tG)``, which has a solution
    only while ``α(T−1) < 1``. Past ``1/α`` cycles NO constant gain can
    satisfy the gate -- so the gain-growth bridge is not an optional
    improvement, it is forced by the arithmetic.
    """
    w = takeoff_window(0.1, 3.32, 0.3)
    assert sustained_takeoff_possible(w, 10) is True     # alpha*(10-1)=0.9
    assert sustained_takeoff_possible(w, 11) is False    # alpha*10 = 1.0
    assert sustained_takeoff_needs(w, 11) == math.inf
    assert sustained_takeoff_needs(w, 20) == math.inf


def test_the_required_gain_grows_with_the_horizon():
    """The longer the horizon, the more the gain must grow to keep up.

    It rises to infinity exactly as the horizon approaches 1/alpha, which
    is the same forced bridge seen from the other side.
    """
    w = takeoff_window(0.1, 1.0, 1.0)
    needs = [sustained_takeoff_needs(w, c) for c in range(1, 11)]
    assert needs == sorted(needs)
    assert all(math.isfinite(n) for n in needs)
    assert needs[-1] > needs[0]


def test_the_requirement_at_one_cycle_is_just_the_gate():
    w = takeoff_window(0.1, 2.0, 5.0)
    assert sustained_takeoff_needs(w, 1) == pytest.approx(0.1 * 2.0)


def test_a_bigger_capability_needs_a_bigger_gain():
    small = takeoff_window(0.1, 1.0, 5.0)
    large = takeoff_window(0.1, 10.0, 5.0)
    assert sustained_takeoff_needs(large, 5) > sustained_takeoff_needs(small, 5)


# --- FIX 1: the honest compression cost ----------------------------------


def test_the_compression_cost_carries_sqrt_n():
    """The undercount R102 FIX 1 removed.

    Measured on one coordinate the cost is ``κ·s/2`` regardless of the
    matrix size, which understates a generation's compress stage by
    ``√n``: 24x at n=576, 34x at n=1152 (this project's width).
    """
    for n in (576, 1152, 3456):
        assert compression_cost(n, 0.1, 1.0) == pytest.approx(
            math.sqrt(n) * 0.05
        )
        assert compression_cost(n, 0.1, 1.0) / 0.05 == pytest.approx(
            math.sqrt(n)
        )


def test_the_compression_cost_scales_with_the_grid_and_kappa():
    assert compression_cost(100, 0.2, 1.0) == pytest.approx(
        2.0 * compression_cost(100, 0.1, 1.0)
    )
    assert compression_cost(100, 0.1, 3.0) == pytest.approx(
        3.0 * compression_cost(100, 0.1, 1.0)
    )


def test_a_zero_grid_or_kappa_costs_nothing():
    assert compression_cost(1152, 0.0, 1.0) == 0.0
    assert compression_cost(1152, 0.1, 0.0) == 0.0


def test_a_measured_residual_gives_the_exact_cost():
    assert compression_cost(1152, 0.1, 2.0, residual=3.0) == pytest.approx(6.0)


def test_a_measured_residual_is_charged_not_bounded():
    """With the measurement in hand the bound is not needed."""
    small = compression_cost(1152, 0.1, 1.0, residual=0.001)
    large = compression_cost(1152, 0.1, 1.0, residual=5.0)
    assert small < large
    assert small != pytest.approx(large)


# --- the per-generation budget ------------------------------------------


def test_a_positive_budget_is_descent():
    """Every term favourable: the gain plus the step's own descent.

    eta*||d*||^2/2 = 0.2 against a compression charge of 0.12 and a noise
    charge of 0.01, so the generation still descends.
    """
    # The theorem is E4 <= Emean - budget, so a POSITIVE budget is a
    # descent: energy falls by that amount.
    assert cycle_budget(gain=1.0, eta=0.1, dnorm_sq=4.0, noise=0.1,
                        n_coords=576, grid=0.01, kappa=1.0) > 0.0


def test_compression_can_swallow_the_gain():
    """The honest cost is exactly why a badly sized compress stage hurts.

    At grid=0.001 the compress stage costs 0.0024 and the generation
    descends (budget 0.0976); at grid=0.05 on 3456 coordinates it costs
    1.47 and the generation COSTS energy (budget -1.37) despite the same
    merge gain. Under the old sqrt(1) accounting both charges looked like
    0.025 and neither stage would ever have looked dangerous.
    """
    cheap = cycle_budget(0.1, 0.0, 0.0, 0.0, 576, 0.001, 1.0)
    pricey = cycle_budget(0.1, 0.0, 0.0, 0.0, 3456, 0.05, 1.0)
    assert cheap > 0.0 > pricey


def test_noise_and_compression_both_reduce_the_budget():
    base = cycle_budget(1.0, 0.1, 1.0, 0.0, 576, 0.001, 1.0)
    noisy = cycle_budget(1.0, 0.1, 1.0, 0.5, 576, 0.001, 1.0)
    assert noisy < base


def test_a_positive_budget_is_the_descent_the_theorem_claims():
    """Sign convention, pinned: E4 <= Emean - budget, so > 0 descends."""
    budget = cycle_budget(0.5, 0.2, 1.0, 0.01, 576, 0.01, 1.0)
    assert budget > 0.0


def test_a_negative_term_is_refused():
    with pytest.raises(ValueError):
        cycle_budget(-1.0, 0.1, 1.0, 0.0, 576, 0.001, 1.0)
    with pytest.raises(ValueError):
        cycle_budget(1.0, 0.1, -1.0, 0.0, 576, 0.001, 1.0)


@pytest.mark.parametrize("bad", [(0, 0.1, 1.0), (10, -0.1, 1.0), (10, 0.1, -1.0)])
def test_invalid_compression_arguments_are_refused(bad):
    with pytest.raises(ValueError):
        compression_cost(*bad)


# --- guards --------------------------------------------------------------


@pytest.mark.parametrize("args", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
                                  (0.1, 0.0, 1.0), (0.1, -1.0, 1.0),
                                  (0.1, 1.0, 0.0), (0.1, 1.0, -1.0)])
def test_a_window_needs_positive_parameters(args):
    with pytest.raises(ValueError):
        takeoff_window(*args)


def test_a_non_positive_cycle_count_is_refused():
    w = takeoff_window(0.1, 1.0, 1.0)
    with pytest.raises(ValueError):
        sustained_takeoff_needs(w, 0)
    with pytest.raises(ValueError):
        sustained_takeoff_needs(w, -1)


def test_the_horizon_form_is_a_restatement_and_still_works():
    """R102 notes it adds no new mathematics; the port keeps it anyway."""
    assert generations_to_floor(10.0, 2.0, 0.5) == pytest.approx(16.0)
    with pytest.raises(ValueError):
        generations_to_floor(10.0, 2.0, 0.0)
    with pytest.raises(ValueError):
        generations_to_floor(2.0, 10.0, 1.0)


# --- FIX 3: task selection needs positive costs --------------------------


def test_a_zero_or_negative_cost_fails_the_premise():
    """R102 FIX 3: the argmax g/c is only defined for positive costs."""
    assert positive_cost_required({"a": 1.0, "b": 2.0}) is True
    assert positive_cost_required({"a": 0.0}) is False
    assert positive_cost_required({"a": -1.0}) is False
    assert positive_cost_required({}) is False