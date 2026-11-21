#!/usr/bin/env python3
"""Transfer gate tests (Phase 4): the 4 COMMIT conditions + rollback."""

from hagi.measurement import CapabilityMeter, TaskResult


def _held_out_good():
    return [
        TaskResult("t1", True, True, False, 17, 4),
        TaskResult("t2", True, True, False, 17, 4),
        TaskResult("t3", True, True, False, 17, 4),
    ]


def test_commit_when_all_conditions_hold():
    m = CapabilityMeter()
    v = m.evaluate(
        held_out_results=_held_out_good(),
        matched_control_results=[TaskResult("c1", False, False, False, 99, 99)],
        pre_solve_count=0, pre_program_count=0,
        pre_avg_bits=48.0, post_avg_bits=17.0,
        pre_avg_probes=48.0, post_avg_probes=4.0,
    )
    assert v.action == "COMMIT", v.reasons


def test_rollback_on_regression():
    m = CapabilityMeter()
    v = m.evaluate(
        held_out_results=[TaskResult("t1", True, True, False, 17, 4)],
        matched_control_results=[],
        pre_solve_count=5, pre_program_count=2,
        pre_avg_bits=10.0, post_avg_bits=17.0,
        pre_avg_probes=4.0, post_avg_probes=4.0,
    )
    assert v.action == "ROLLBACK"
    assert any("REGRESSION" in r for r in v.reasons)


def test_rollback_on_false_structural_controls():
    m = CapabilityMeter(alpha_fdr=0.0)
    v = m.evaluate(
        held_out_results=_held_out_good(),
        matched_control_results=[TaskResult("c1", False, True, False, 17, 4)],
        pre_solve_count=0, pre_program_count=0,
        pre_avg_bits=48.0, post_avg_bits=17.0,
        pre_avg_probes=48.0, post_avg_probes=4.0,
    )
    assert v.action == "ROLLBACK"
    assert any("FALSE_STRUCTURAL" in r for r in v.reasons)


def test_rollback_on_no_capacity_gain():
    m = CapabilityMeter()
    # solved == pre_solve_count, all PROGRAM (no structural growth)
    v = m.evaluate(
        held_out_results=[TaskResult("t1", True, False, True, 10, 3)],
        matched_control_results=[],
        pre_solve_count=1, pre_program_count=1,
        pre_avg_bits=48.0, post_avg_bits=10.0,
        pre_avg_probes=48.0, post_avg_probes=3.0,
    )
    assert v.action == "ROLLBACK"
    assert any("NO_CAPACITY_GAIN" in r for r in v.reasons)


def test_rollback_on_no_cost_win():
    m = CapabilityMeter()
    v = m.evaluate(
        held_out_results=_held_out_good(),
        matched_control_results=[],
        pre_solve_count=0, pre_program_count=0,
        pre_avg_bits=10.0, post_avg_bits=17.0,
        pre_avg_probes=1.0, post_avg_probes=4.0,
    )
    assert v.action == "ROLLBACK"
    assert any("NO_COST_WIN" in r for r in v.reasons)
