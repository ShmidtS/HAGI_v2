"""V10-3: CapabilityEvaluator — measure whether the system actually grew.

Two-part MDL: TotalMDL = L(library) + sum(bits(expert | library)) +
penalty(unsolved). A construct counts as a real capability gain only
if: DeltaTotalMDL < 0 AND held-out transfer solve-rate improved AND
zero noise acceptances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CapabilitySnapshot:
    version: int
    library_meta_bits: int
    experts_bits: int
    unsolved_penalty_bits: int
    total_mdl_bits: int
    heldout_solve_rate: float
    mean_samples_to_solve: float
    noise_false_accepts: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def total_mdl(library_meta_bits: int,
              expert_bits: list[int],
              unsolved: int,
              penalty_per_unsolved: int = 64) -> int:
    return (int(library_meta_bits)
            + sum(int(b) for b in expert_bits)
            + int(unsolved) * int(penalty_per_unsolved))


def capability_delta(before: CapabilitySnapshot,
                     after: CapabilitySnapshot) -> dict:
    d_mdl = after.total_mdl_bits - before.total_mdl_bits
    d_rate = after.heldout_solve_rate - before.heldout_solve_rate
    d_samples = (after.mean_samples_to_solve
                 - before.mean_samples_to_solve)
    return {
        "delta_total_mdl": d_mdl,          # want < 0
        "delta_heldout_solve_rate": d_rate,  # want > 0
        "delta_mean_samples": d_samples,     # want <= 0 (efficiency)
        "noise_false_accepts": after.noise_false_accepts,  # want == 0
        "REAL_GAIN": (d_mdl < 0 and d_rate > 0
                      and after.noise_false_accepts == 0),
    }


def heldout_suite(tasks: list[tuple[str, Callable, range, int]],
                  solver: Callable[[str, range, int], Optional[tuple[int, int]]],
                  ) -> tuple[int, float, float]:
    """Run held-out transfer panel. solver(name, domain, target) returns
    (bits_cost | None-if-HOLD, samples_used). Returns (solved, rate,
    mean_samples_to_solve over solved only)."""
    solved = 0
    samples = []
    for name, fn, dom, target in tasks:
        res = solver(name, dom, target)
        if res is not None and res[0] is not None:
            solved += 1
            samples.append(res[1])
    rate = solved / max(1, len(tasks))
    mean_s = sum(samples) / len(samples) if samples else 0.0
    return solved, rate, mean_s


def noise_panel(noise_tables: list[dict[int, int]],
                inventor: Callable[[dict[int, int]], object],
                ) -> int:
    """Feed randomized tables; count accepted STRUCTURAL constructs.
    Honest system must return 0."""
    accepts = 0
    for tbl in noise_tables:
        if inventor(tbl) is not None:
            accepts += 1
    return accepts
