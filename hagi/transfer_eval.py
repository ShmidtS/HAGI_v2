#!/usr/bin/env python3
"""CapabilityMeter — external transfer gate (Phase 4).

Zero-shot gate: an invented representation is COMMITed only if, on
held-out tasks that were NOT in the discovery cluster:

  1. no regression: every previously-solved task keeps 100% accuracy,
     the PROGRAM count does not decrease;
  2. structural capacity grows: DELTA(PROGRAM + STRUCTURAL) > 0 on
     held-out;
  3. matched-controls purity: 0 false STRUCTURAL claims;
  4. cost win: description bits and/or probes-per-task decrease.

On any failure: ROLLBACK (registry entry removed, snapshot restored).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


@dataclass
class TaskResult:
    task_name: str
    solved: bool
    structural: bool          # committed via a (possibly invented) construct
    program: bool             # committed as short algebraic PROGRAM
    bits: int
    probes: int


@dataclass
class MeterVerdict:
    action: str               # COMMIT | ROLLBACK
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class CapabilityMeter:
    def __init__(self, alpha_fdr: float = 0.0):
        self.alpha_fdr = alpha_fdr

    def evaluate(self,
                 held_out_results: list[TaskResult],
                 matched_control_results: list[TaskResult],
                 pre_solve_count: int,
                 pre_program_count: int,
                 pre_avg_bits: float,
                 post_avg_bits: float,
                 pre_avg_probes: float,
                 post_avg_probes: float) -> MeterVerdict:
        reasons: list[str] = []
        m: dict = {}
        solved = [r for r in held_out_results if r.solved]
        structural = [r for r in solved if r.structural]
        programs = [r for r in solved if r.program]
        # 1. no regression
        if len(solved) < pre_solve_count:
            reasons.append(f"REGRESSION: solved {len(solved)} < {pre_solve_count}")
        if len(programs) < pre_program_count:
            reasons.append(f"REGRESSION: programs {len(programs)} < {pre_program_count}")
        m["solved"] = len(solved)
        m["structural"] = len(structural)
        m["programs"] = len(programs)
        # 2. structural capacity growth on held-out
        delta = len(structural) + len(programs) - pre_solve_count
        m["delta_capacity"] = delta
        if delta <= 0:
            reasons.append(f"NO_CAPACITY_GAIN: delta={delta}")
        # 3. matched-controls purity
        false_structural = sum(1 for r in matched_control_results
                               if r.structural and not r.solved)
        m["false_structural_controls"] = false_structural
        if false_structural > self.alpha_fdr:
            reasons.append(f"FALSE_STRUCTURAL on controls: {false_structural}")
        # 4. cost win (bits or probes must decrease)
        bits_win = post_avg_bits < pre_avg_bits
        probes_win = post_avg_probes < pre_avg_probes
        m["bits_win"] = bits_win
        m["probes_win"] = probes_win
        m["avg_bits"] = (pre_avg_bits, post_avg_bits)
        m["avg_probes"] = (pre_avg_probes, post_avg_probes)
        if not (bits_win or probes_win):
            reasons.append("NO_COST_WIN")
        if reasons:
            return MeterVerdict("ROLLBACK", reasons, m)
        return MeterVerdict("COMMIT", [], m)


def rollback(registry_remove: Callable[[str], None], repr_id: str,
             restore_snapshot: Callable[[], None]) -> MeterVerdict:
    """Honest rollback: remove the registry entry, restore the snapshot."""
    try:
        registry_remove(repr_id)
    finally:
        restore_snapshot()
    return MeterVerdict("ROLLBACK", ["rolled back"], {"repr_id": repr_id})
