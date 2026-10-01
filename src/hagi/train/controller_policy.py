"""Action selection by certified gain-per-cost (ratio_dominance).

``Hagi/Dynamics/ControllerPolicy.lean``:

``ratio_dominance``
    For the best-ratio action ``ibest`` and any ``j``,
    ``Gam j * (B / K j) <= Gam ibest * (B / K ibest)`` -- spending the
    whole budget B on ``ibest`` yields at least as much certified gain
    as spending it on ``j``.

``budget_allocation_dominance``
    Under a common budget, ``sum_a Gam(idx a) * alloc a <=
    (Gam ibest / K ibest) * B`` for ANY split allocation ``alloc``
    satisfying ``sum_a alloc a * K (idx a) <= B``. So concentrating
    the budget on ``argmax Gam/K`` is not a heuristic preference: it
    is certified to dominate every split.

This module turns the theorems into the selection rule. The certified
gains come from the sibling theorems, each attached to the action it
justifies:

======================  ==================================  ==========
action                  certified gain ``Gam``             source
======================  ==================================  ==========
``merge``               ``twoGap``                          GapLaw
``joint``               ``eta*||d*||^2 / 2``                SafeQP
``internalize``         KL-gap of the insight              R79
``prune``               ``saving - delta^2 / 4``            R72
``grow``                ``G_new - cost``                    liveness
======================  ==================================  ==========

``ALGORITHMS.md`` §1 also records the prohibition: selecting on a
candidate's STANDALONE CE is refuted by ``selection_hurts``. Only
certified gains may enter ``Gam``; :func:`select_action` therefore
takes gains as its argument and never inspects standalone CE.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The action space of ALGORITHMS.md §1.
ACTIONS = (
    "merge",
    "joint",
    "internalize",
    "prune",
    "grow_leaf",
    "distill",
    "ttt_refresh",
)


@dataclass(frozen=True)
class Candidate:
    """One action with its certified gain and its measured cost.

    Attributes:
        name: the action identifier.
        gain: ``Gam`` -- the CERTIFIED gain, never a standalone CE.
        cost: ``K`` -- the measured wall-clock/FLOPs cost, must be > 0.
    """

    name: str
    gain: float
    cost: float

    def __post_init__(self) -> None:
        if self.cost <= 0.0:
            raise ValueError(f"action {self.name}: cost must be positive")
        if not math.isfinite(self.gain) or not math.isfinite(self.cost):
            raise ValueError(f"action {self.name}: non-finite gain or cost")

    @property
    def ratio(self) -> float:
        """``Gam / K`` -- the quantity ``ratio_dominance`` maximizes."""
        return self.gain / self.cost


def select_action(
    candidates: list[Candidate], budget: float | None = None
) -> Candidate | None:
    """``argmax_i Gam_i / K_i`` (``ratio_dominance``).

    ``budget_allocation_dominance`` says concentrating the whole budget
    on the argmax certifies at least as much gain as any split, so the
    rule needs no search over allocations: evaluate the ratios, take
    the best. ``budget`` is used only to report the certified gain of
    that choice -- it does not change the selection.

    Args:
        candidates: the feasible actions with certified gains.
        budget: total budget B, used for the reported certified gain.

    Returns:
        The best-ratio candidate, or None when none is available or
        all ratios are non-positive (no action certifies progress, and
        the honest answer is to do nothing rather than spend compute).
    """
    usable = [c for c in candidates if c.ratio > 0.0]
    if not usable:
        return None
    best = max(usable, key=lambda c: c.ratio)
    if budget is not None:
        certified = best.gain * (budget / best.cost)
        logger.info(
            "controller: %s (ratio=%.6g), certified gain >= %.6g at budget %.6g",
            best.name, best.ratio, certified, budget,
        )
    return best


def certified_gain(candidates: list[Candidate], budget: float) -> float:
    """``(Gam_ibest / K_ibest) * B`` -- the floor every allocation meets.

    This is the right-hand side of ``budget_allocation_dominance``:
    what ANY policy is certified to achieve at this budget. A split
    allocation scoring at or below it has no certified advantage.
    """
    usable = [c for c in candidates if c.ratio > 0.0]
    if not usable:
        return 0.0
    return max(c.ratio for c in usable) * budget
