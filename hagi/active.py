"""Active query selection for finite deterministic hypothesis committees.

For a deterministic oracle, the mutual information between a candidate
hypothesis H and its predicted label Y at x equals the entropy of Y under
the current posterior over H. Selecting the point with maximum predictive
entropy therefore maximizes one-step expected information gain.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence, TypeVar

H = TypeVar("H")
X = TypeVar("X")
Y = TypeVar("Y", bound=Hashable)


@dataclass(frozen=True)
class QueryChoice:
    point: object
    information_gain_bits: float
    outcome_probabilities: tuple[tuple[Hashable, float], ...]


def select_query_by_committee(
    candidates: Sequence[H],
    pool: Sequence[X],
    predict: Callable[[H, X], Y],
    weights: Sequence[float] | None = None,
) -> QueryChoice | None:
    """Choose the pool point with greatest predictive-label entropy.

    ``weights`` are non-negative posterior/prior masses for candidates. The
    function is deterministic: ties preserve the supplied pool order. It
    returns ``None`` when fewer than two hypotheses remain or no pool point
    separates them. A point is never queried here; this is pure selection.
    """
    if len(candidates) < 2 or not pool:
        return None
    if weights is None:
        masses = [1.0] * len(candidates)
    else:
        if len(weights) != len(candidates):
            raise ValueError("weights must match the candidate count")
        masses = [float(value) for value in weights]
        if any(not math.isfinite(value) or value < 0 for value in masses):
            raise ValueError("weights must be finite and non-negative")
    total_mass = sum(masses)
    if total_mass <= 0:
        raise ValueError("at least one candidate weight must be positive")
    masses = [value / total_mass for value in masses]

    best: QueryChoice | None = None
    for point in pool:
        outcome_mass: dict[Hashable, float] = defaultdict(float)
        for candidate, mass in zip(candidates, masses):
            if mass <= 0:
                continue
            outcome = predict(candidate, point)
            outcome_mass[outcome] += mass
        if len(outcome_mass) < 2:
            continue
        entropy = -sum(prob * math.log2(prob) for prob in outcome_mass.values() if prob > 0)
        if best is None or entropy > best.information_gain_bits + 1e-12:
            best = QueryChoice(
                point=point,
                information_gain_bits=entropy,
                outcome_probabilities=tuple(sorted(outcome_mass.items(), key=lambda item: repr(item[0]))),
            )
    return best
