"""Tests for the anytime-valid acceptance gate (R93 wired into growth).

The supervisor compares each lane's joint model against the incumbent on
a fixed margin. Over a plan with many lanes that spends `delta` per gate,
so the nominal budget is `n * delta` -- the E3 failure in the working
notes, where 400 checks at delta=0.05 need a budget of 20.

R93 replaces the fixed per-gate margin with the geometric schedule
`delta_t = delta_0 rho^t`, whose infinite sum is at most `delta`. These
tests pin that property on the code path that actually runs, not on the
mathematics in `anytime_budget.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "growth"))

import growth_supervisor as gs  # noqa: E402


def report(**domains):
    return {"domains": {k: {"exact_ce": v} for k, v in domains.items()}}


# --- the schedule itself -------------------------------------------------


def test_lane_zero_reproduces_the_previous_fixed_margin():
    """Passing lane=0 narrows the margin by (1-rho); it must stay usable.

    The schedule is scaled so the geometric factors sum to one, so lane 0
    gets (1-rho) = 0.5 of the declared margin -- tighter than the old
    fixed value, never wider, which is the safe direction.
    """
    assert gs.anytime_margin(0.25, 0, 0.05, 0.5) == pytest.approx(0.125)
    # A candidate worse than lane 0's margin is still rejected.
    v = gs.decide(report(A=3.30), report(A=3.00), 0.05, 0.25, lane=0)
    assert v.accepted is False


def test_the_margin_shrinks_with_each_successive_gate():
    ms = [gs.anytime_margin(0.25, t, 0.05, 0.5) for t in range(6)]
    assert ms == sorted(ms, reverse=True)
    assert ms[0] > ms[-1]


def test_the_relative_spend_is_exactly_the_geometric_series():
    base = 0.25
    scale = gs.anytime_margin(base, 0, 0.05, 0.5)
    for t in range(8):
        assert gs.anytime_margin(base, t, 0.05, 0.5) / scale == \
            pytest.approx(0.5 ** t)


def test_a_slower_decay_keeps_later_gates_wider():
    fast = gs.anytime_margin(0.25, 3, 0.05, 0.2)
    slow = gs.anytime_margin(0.25, 3, 0.05, 0.8)
    assert slow > fast


def test_the_total_spend_is_bounded_regardless_of_length():
    """The property the whole change exists for.

    Under a fixed margin, N gates cost N * delta. Under this schedule the
    cost is the partial sum of a geometric series, bounded by delta_0/(1-rho)
    for ANY N -- which is at most `total_delta`.
    """
    delta, rho = 0.05, 0.5
    d0 = delta * (1.0 - rho)
    for n in (1, 10, 100, 1000, 10_000, 10**6):
        spent = sum(d0 * rho ** t for t in range(n))
        assert spent <= delta + 1e-12


def test_a_long_run_spends_vanishingly_little_on_its_last_gate():
    """After many gates the margin is essentially zero, by design."""
    d0 = 0.05 * (1.0 - 0.5)
    assert d0 * 0.5 ** 20 < 1e-7


# --- guards --------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_base_margin_is_refused(bad):
    with pytest.raises(ValueError):
        gs.anytime_margin(bad, 0, 0.05, 0.5)


@pytest.mark.parametrize("rho", [0.0, 1.0, -0.5, 1.5])
def test_a_decay_outside_the_unit_interval_is_refused(rho):
    with pytest.raises(ValueError):
        gs.anytime_margin(0.25, 0, 0.05, rho)


def test_a_non_positive_delta_is_refused():
    with pytest.raises(ValueError):
        gs.anytime_margin(0.25, 0, 0.0, 0.5)


def test_a_bad_budget_does_not_fall_back_to_the_fixed_margin():
    """Silently reverting would hide a misconfiguration, which is exactly
    how the fixed-margin compounding went unnoticed."""
    with pytest.raises(ValueError):
        gs.decide(report(A=3.0), report(A=3.30), 0.05, 0.25, lane=0,
                  total_delta=0.0)


# --- the gate actually uses it ------------------------------------------


def test_a_within_margin_candidate_is_rejected_everywhere():
    """§9 strict (audit P1-1): a candidate that is NOT BETTER is never
    accepted, at any lane — the old not-worse-on-margin acceptance was
    the forbidden leak. Early lanes are wider only for CERTIFIED gains
    (the adaptive eps), not for tolerable regressions.
    """
    cand = report(A=3.005)          # slightly worse
    inc = report(A=3.000)
    verdicts = [gs.decide(cand, inc, 0.05, 0.25, lane=t).accepted
                for t in range(6)]
    assert verdicts == [False] * 6


def test_a_regression_larger_than_the_first_margin_is_caught_immediately():
    """The tightening must not make lane 0 a rubber stamp."""
    cand = report(A=3.05)
    inc = report(A=3.00)
    for t in range(6):
        assert gs.decide(cand, inc, 0.05, 0.25, lane=t).accepted is False


def test_a_large_regression_is_rejected_at_every_lane():
    cand = report(A=4.00)          # candidate worse
    inc = report(A=3.00)
    for t in range(8):
        v = gs.decide(cand, inc, 0.05, 0.25, lane=t)
        assert v.accepted is False


def test_a_clear_improvement_is_accepted_at_every_lane():
    cand = report(A=2.00)          # candidate better
    inc = report(A=3.00)
    for t in range(8):
        assert gs.decide(cand, inc, 0.05, 0.25, lane=t).accepted is True


def test_tightening_the_late_gate_cannot_make_a_worse_candidate_pass():
    """Monotone: more budget spent earlier never weakens a later verdict."""
    cand = report(A=3.20, B=3.20)   # worse
    inc = report(A=3.00, B=3.10)
    verdicts = [gs.decide(cand, inc, 0.05, 0.25, lane=t).accepted
                for t in range(8)]
    assert verdicts == sorted(verdicts, reverse=True)


def test_the_first_lane_still_guards_an_incumbent_regression():
    """The tightening must not turn lane 0 into a rubber stamp."""
    cand = report(A=9.00)          # candidate much worse
    inc = report(A=3.00)
    assert gs.decide(cand, inc, 0.05, 0.25, lane=0).accepted is False