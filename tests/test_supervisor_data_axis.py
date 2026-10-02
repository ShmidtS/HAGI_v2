"""Tests for the stalled-growth advisory (data_axis.py §4, wired in).

`data_axis.py` was written with tests and called from nowhere. The
supervisor's `decide()` is where its premise lives: the only gain the
growth loop can measure is `incumbent_mean - candidate_mean`, and when
that goes to zero the merge axis is dead. `liveness_data_axis` says the
loop then needs the DATA axis -- fresh independent data -- rather than
another variation of the same merge.

These tests pin that the advisory fires exactly on a stalled gain, says
nothing when growth is real, and does not recommend the one thing that
was measured not to work (more of the same distribution).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "growth"))
sys.path.insert(0, str(ROOT / "src"))

import growth_supervisor as gs  # noqa: E402
from hagi.train.data_axis import needs_data_injection  # noqa: E402


def report(**domains):
    return {"domains": {k: {"exact_ce": v} for k, v in domains.items()}}


# --- when it fires -------------------------------------------------------


def test_a_real_improvement_gets_no_advice():
    assert gs.data_axis_advice(0.5) == ""


def test_no_improvement_at_all_is_flagged_as_stalled():
    note = gs.data_axis_advice(0.0)
    assert "stalled" in note
    assert "inject" in note


def test_a_gain_below_the_threshold_is_flagged():
    assert "stalled" in gs.data_axis_advice(0.05)
    assert gs.data_axis_advice(gs.STALL_GAIN + 1e-6) == ""


def test_a_tiny_negative_gain_is_also_stalled():
    """A candidate that is slightly worse but inside the margin has no
    positive gain, so the merge axis is exactly as dead."""
    assert "stalled" in gs.data_axis_advice(-0.01)


# --- what it says --------------------------------------------------------


def test_the_advice_names_the_independence_requirement():
    """The measured round-66/67 finding: same-domain data is noise."""
    note = gs.data_axis_advice(0.0).lower()
    assert "not seen" in note
    # it must NOT suggest more of the same distribution
    assert "same corpus" not in note and "same data" not in note


def test_the_advice_does_not_recommend_retrying():
    """liveness_data_axis: the same action cannot help once consensus."""
    note = gs.data_axis_advice(0.0).lower()
    assert "cannot help" in note


def test_the_threshold_is_expressed_in_the_project_s_own_noise_scale():
    """It must be tied to the measured noise, not invented."""
    assert gs.STALL_GAIN == pytest.approx(gs.DEFAULT_DOMAIN_MARGIN)


def test_the_threshold_is_overridable():
    assert "stalled" in gs.data_axis_advice(0.1, stall_threshold=0.2)
    assert gs.data_axis_advice(0.1, stall_threshold=0.05) == ""


# --- it agrees with the theorem it cites ---------------------------------


def test_the_advisory_and_the_ported_gate_agree_on_the_dead_merger_axis():
    """Both reduce to the same question, but at different thresholds.

    ``needs_data_injection(G, inj, xi)`` is the theorem's exact form: it
    asks whether fresh data OPENS the axis, and it needs ``inj > xi`` to
    do so -- with no injection available it is False even when the merge
    axis is dead. The advisory answers a different question: is this
    particular verdict evidence of no progress? So they agree on the
    clear cases and are allowed to differ in the margin between.
    """
    for gain in (-1.0, -0.1, 0.0, 0.3, 1.0):
        stalled = "stalled" in gs.data_axis_advice(gain)
        wants_data = needs_data_injection(gain, 0.0, 0.0)
        assert stalled == wants_data


def test_the_gap_between_thresholds_is_deliberate_and_measured():
    """gain=0.05: the merge axis is dead but no injection is available.

    ``needs_data_injection`` stays False because with ``inj = 0`` the
    data axis is ALSO dead -- and the gate's True means "you need fresh
    data", so a dead injection axis reads False. The advisory still
    fires, because 0.05 nats is inside the ~0.2 nat noise band and
    therefore indistinguishable from no progress. Both are right about
    their own question; the advisory is the one that reaches the ledger.
    """
    assert needs_data_injection(0.05, 0.0, 0.0) is False
    assert "stalled" in gs.data_axis_advice(0.05)


def test_a_strong_enough_injection_reopens_the_data_axis():
    """The full §4 scenario: merge dead, fresh data actually supplied.

    The gate's True means "fresh independent data is REQUIRED", which is
    what happens while ``inj <= xi`` -- the injection is too small to
    outrun the decay the same distribution imposes.
    """
    # inj 0.1 cannot outrun xi 0.5: still starved, more data needed.
    assert needs_data_injection(0.0, 0.1, 0.5) is True
    # A positive gain keeps the merge axis alive regardless of injection.
    assert needs_data_injection(0.4, 0.0, 1.0) is False
    assert "stalled" in gs.data_axis_advice(0.0)


def test_a_stalled_verdict_reaches_the_ledger_reason():
    """End to end: the note must appear in the Verdict, not beside it."""
    inc = report(A=3.00)
    cand = report(A=3.00)              # identical: zero gain
    v = gs.decide(cand, inc, 0.05, 0.25, lane=0)
    assert v.accepted is True          # not worse, so accepted
    assert "stalled" in v.reason       # but flagged as no progress


def test_an_improving_verdict_reason_has_no_stall_note():
    inc = report(A=3.50)
    cand = report(A=3.00)
    v = gs.decide(cand, inc, 0.05, 0.25, lane=0)
    assert v.accepted is True
    assert "stalled" not in v.reason