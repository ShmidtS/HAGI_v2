"""When a synthetic pre-training phase is worth running, as arithmetic.

``Hagi/Pretraining/...`` (R99). The question "should the next generation
spend compute on a synthetic warm-up before the main pre-train, or go
straight to the main pre-train?" is usually answered by taste. Three
theorems turn it into a check:

``TimeToCapability`` + correctness
    ``T`` is the first step at which capability reaches a threshold
    ``tau``, and is monotone in the schedule -- so a schedule that
    reaches ``tau`` earlier is strictly better, not "probably better".

``synth_investment_dominates``
    With ONE empirical premise -- ``h_emp_gate``: the synthetic phase
    saves ``dT`` steps at a cost of ``c_step`` each, and the synthetic
    phase itself costs ``C_synth`` --

        dT * c_step > C_synth   ==>   capability-per-compute of
        (synth -> main) is STRICTLY higher than (main alone).

    So the gate is a single comparison of measured quantities. If it
    fails, the synthetic phase is not "maybe not worth it": it is
    strictly worse per unit compute, and the honest answer is to skip
    it. Nothing here is a heuristic and nothing is tuned.

``task_selection_marginal``
    Among candidate synthetic tasks, pick ``argmax_j g_j / c_j`` -- the
    marginal gain per unit cost. Finite argmax, so it is a scan, not a
    search over schedules.

The retrieval mechanism remains an empirical premise
(``h_emp_retrieval_transfer``): that a synthetic corpus actually
transfers to the target distribution is MEASURED, not proved, and this
module says so rather than hiding it. What is proved is the accounting
around it -- given the transfer, when the investment pays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticInvestment:
    """The measured quantities behind ``h_emp_gate``.

    Attributes:
        steps_saved: ``dT`` -- steps the synthetic phase removes from
            the main pre-train.
        step_cost: ``c_step`` -- compute cost of one main-training
            step, in whatever unit ``synthetic_cost`` uses.
        synthetic_cost: ``C_synth`` -- total cost of the synthetic
            phase, in the same unit.
    """

    steps_saved: float
    step_cost: float
    synthetic_cost: float

    def __post_init__(self) -> None:
        if self.steps_saved < 0:
            raise ValueError("steps_saved must be non-negative")
        if self.step_cost < 0 or self.synthetic_cost < 0:
            raise ValueError("costs must be non-negative")

    @property
    def saved_compute(self) -> float:
        """``dT * c_step`` -- the compute the synthetic phase returns."""
        return self.steps_saved * self.step_cost

    @property
    def net_gain(self) -> float:
        """``dT*c_step - C_synth`` -- positive exactly when it pays."""
        return self.saved_compute - self.synthetic_cost


def synthetic_phase_pays(inv: SyntheticInvestment) -> bool:
    """``h_emp_gate``: ``dT * c_step > C_synth``.

    ``synth_investment_dominates``: strictly greater, so the boundary
    case (equal) is NOT taken -- on a tie the two schedules deliver the
    same capability per compute and the simpler one (no synthetic
    phase) should win.

    Args:
        inv: the measured investment.

    Returns:
        True iff the synthetic phase is strictly better per compute.
    """
    return inv.saved_compute > inv.synthetic_cost


def capability_per_compute(
    capability: float, cost: float
) -> float:
    """``C / cost`` -- the quantity the gate optimizes.

    Raises:
        ValueError: on non-positive capability or non-positive cost.
    """
    if capability <= 0.0:
        raise ValueError("capability_per_compute needs a positive capability")
    if cost <= 0.0:
        raise ValueError("capability_per_compute needs a positive cost")
    return capability / cost


def time_to_capability(
    capability_curve: list[float], threshold: float
) -> int:
    """``TimeToCapability``: the first step whose capability reaches ``tau``.

    ``Nat.find`` over a monotone capability curve. Returns ``-1`` when
    the threshold is never reached, which is a different answer from a
    late step and must not be conflated with it: "never" means the
    schedule fails, and the fix is a different schedule, not more time.

    Args:
        capability_curve: ``[C_0, C_1, ...]`` per step.
        threshold: ``tau``.

    Returns:
        The first index reaching ``tau``, or ``-1``.

    Raises:
        ValueError: on a negative threshold or a curve that ever
            decreases (which would break the monotonicity premise).
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    for i in range(1, len(capability_curve)):
        if capability_curve[i] < capability_curve[i - 1]:
            raise ValueError(
                f"capability curve decreases at step {i}: "
                f"{capability_curve[i - 1]} -> {capability_curve[i]}; "
                "TimeToCapability assumes monotonicity"
            )
    for i, c in enumerate(capability_curve):
        if c >= threshold:
            return i
    return -1


def select_task(gains: dict[str, float], costs: dict[str, float]) -> tuple[str, float]:
    """``task_selection_marginal``: ``argmax_j g_j / c_j``.

    The finite argmax the theorem prescribes: among candidate synthetic
    tasks, take the one with the highest marginal gain per unit cost.

    Args:
        gains: ``{task: gain}``, non-negative.
        costs: ``{task: cost}``, strictly positive.

    Returns:
        ``(task, ratio)`` for the best task.

    Raises:
        ValueError: on an empty set, negative gains, non-positive
            costs, or a task missing from either mapping.
    """
    if not gains:
        raise ValueError("select_task needs at least one candidate")
    best: tuple[str, float] | None = None
    for task, gain in gains.items():
        if task not in costs:
            raise ValueError(f"task {task!r} has no cost")
        if gain < 0.0:
            raise ValueError(f"task {task!r} has a negative gain")
        cost = costs[task]
        if cost <= 0.0:
            raise ValueError(f"task {task!r} needs a positive cost")
        ratio = gain / cost
        if best is None or ratio > best[1]:
            best = (task, ratio)
    assert best is not None  # non-empty gains
    return best


def required_step_savings(inv: SyntheticInvestment) -> float:
    """The ``dT`` at which the gate flips: ``C_synth / c_step``.

    The break-even point, so a caller can ask "how many steps must the
    synthetic phase actually save?" instead of guessing. Returns
    ``inf`` when a step costs nothing, meaning any saving suffices.

    Raises:
        ValueError: on a negative step cost.
    """
    if inv.step_cost < 0:
        raise ValueError("step_cost must be non-negative")
    if inv.step_cost == 0.0:
        return math.inf
    return inv.synthetic_cost / inv.step_cost