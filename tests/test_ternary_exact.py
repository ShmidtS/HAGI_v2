"""Tests for the honest ternary saturation bound (R103).

The compression bound this project used carried a premise it could not
honour: every coordinate within half a step of the grid. R103 replaces
it with an identity that holds for the real saturating quantiser.

The tests below are mostly about the premise being GONE rather than
assumed: a weight tensor with heavy saturation must produce a cost that
reflects it, and the old form must be shown to undercount.
"""

from __future__ import annotations

import math

import pytest
import torch

from hagi.train.ternary_exact import (
    in_range,
    no_saturation_recover,
    qtern_error,
    recalibrate_pays,
    sat_tail,
    ternary_split,
)


def weights(*vals, n=0, spread=1.0, seed=0):
    if n:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(n, generator=g, dtype=torch.float64) * spread
    return torch.tensor(vals, dtype=torch.float64)


# --- the two regimes ---------------------------------------------------


def test_an_in_range_coordinate_keeps_the_half_step_guarantee():
    """``qTern_error_in``: ``|x - qTern| <= s/2``."""
    s = 0.1
    for x in (0.0, 0.05, -0.05, 0.1, -0.1, 0.14, -0.14, 0.15, -0.15):
        assert qtern_error(s, x) <= s / 2 + 1e-12


def test_a_saturated_coordinate_obeys_the_exact_saturation_law():
    """``qTern_error_out``: the error is exactly ``|x| - s``."""
    s = 0.1
    for x in (0.16, 0.5, 5.0, -0.16, -0.5, -5.0):
        assert qtern_error(s, x) == pytest.approx(abs(x) - s)


def test_the_saturation_law_is_where_the_old_hypothesis_broke():
    """The premise ``|w - q| <= s/2`` fails here and only here."""
    s = 0.1
    x = 5.0
    assert qtern_error(s, x) > s / 2
    # ... and the exact law says how far past it we are.
    assert qtern_error(s, x) == pytest.approx(x - s)


def test_the_error_is_symmetric_in_sign():
    s = 0.1
    for x in (0.07, 0.3, 2.0):
        assert qtern_error(s, x) == pytest.approx(qtern_error(s, -x))


def test_in_range_is_the_boundary_at_three_halves():
    assert in_range(0.1, 0.15) is True
    assert in_range(0.1, 0.1500001) is False


def test_a_non_positive_scale_is_refused():
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError):
            qtern_error(bad, 1.0)
        with pytest.raises(ValueError):
            in_range(bad, 1.0)


# --- the split is an IDENTITY ------------------------------------------


def test_the_split_is_exact_on_a_saturating_tensor():
    """``ternary_split_bound``: total = in-range + satTail, exactly."""
    w = weights(n=500, spread=1.0, seed=1)
    for s in (0.05, 0.2, 0.5):
        assert ternary_split(w, s).splits_exactly


def test_the_split_is_exact_with_no_saturation_at_all():
    w = torch.full((100,), 0.05, dtype=torch.float64)
    sp = ternary_split(w, 0.1)
    assert sp.tail == pytest.approx(0.0)
    assert sp.splits_exactly


def test_saturation_carries_most_of_the_error_at_a_small_scale():
    """The regime the old bound silently ignored."""
    w = weights(n=500, spread=1.0, seed=2)
    sp = ternary_split(w, 0.01)
    assert sp.tail > sp.in_range
    assert sp.splits_exactly


def test_the_tail_grows_as_the_grid_shrinks():
    """Two effects both push the same way, which is why it is monotone.

    Narrowing ``s`` puts MORE coordinates out of range, and it also
    lengthens each overshoot (``|w| − s`` grows as ``s`` shrinks). So
    the tail increases strictly as the grid narrows -- it is not a
    trade-off between the two effects, both amplify.

    Checked in the order the list is written: ``s`` decreasing, tail
    increasing.
    """
    w = weights(n=200, spread=1.0, seed=3)
    tails = [sat_tail(w, s) for s in (0.5, 0.2, 0.05, 0.01)]
    assert tails == sorted(tails)
    assert tails[-1] > tails[0]


def test_the_tail_is_zero_when_nothing_saturates():
    w = torch.full((50,), 0.05, dtype=torch.float64)
    assert sat_tail(w, 0.1) == pytest.approx(0.0)


def test_the_tail_ignores_the_sign_of_a_coordinate():
    w = torch.tensor([5.0, -5.0, 0.05], dtype=torch.float64)
    assert sat_tail(w, 0.1) == pytest.approx(2 * (5.0 - 0.1) ** 2)


# --- the bridge ---------------------------------------------------------


def test_the_old_premise_is_recovered_exactly_when_nothing_saturates():
    """``no_saturation_recover``: the R70 hypothesis is DERIVED here."""
    w = torch.full((20,), 0.05, dtype=torch.float64)
    assert no_saturation_recover(w, 0.1) is True
    # ... and the split degenerates to the old form.
    assert ternary_split(w, 0.1).tail == pytest.approx(0.0)


def test_a_saturating_tensor_does_not_claim_the_old_premise():
    w = torch.tensor([0.05, 5.0], dtype=torch.float64)
    assert no_saturation_recover(w, 0.1) is False


# --- the operational decision -------------------------------------------


def test_widening_the_grid_is_cheaper_than_living_with_the_tail():
    """The comparison the theorem exists to enable."""
    w = weights(n=400, spread=1.0, seed=4)
    ok, now, new = recalibrate_pays(w, 0.02, 1.0, 0.5)
    assert ok is True
    assert new < now


def test_the_decision_is_a_comparison_not_a_constant():
    """At a wide enough grid the current one is already tail-free."""
    w = torch.full((10,), 0.05, dtype=torch.float64)
    ok, now, new = recalibrate_pays(w, 0.1, 1.0, 0.2)
    assert ok is False          # both are zero; nothing to gain
    assert now == pytest.approx(0.0)
    assert new == pytest.approx(0.0)


def test_a_wider_grid_never_increases_the_tail():
    w = weights(n=300, spread=1.0, seed=5)
    tails = [sat_tail(w, s) for s in (0.05, 0.1, 0.2, 0.4, 0.8)]
    assert tails == sorted(tails, reverse=True)


def test_invalid_scales_are_refused():
    w = torch.ones(4, dtype=torch.float64)
    with pytest.raises(ValueError):
        recalibrate_pays(w, 0.0, 1.0, 0.1)
    with pytest.raises(ValueError):
        recalibrate_pays(w, 0.1, 1.0, -1.0)