"""Typed, cached oracle access for active and counterexample-guided learning.

The cache is local to one task-learning transaction. Repeated reads of the
same input tuple return the first label without calling the teacher again.
This module deliberately does not choose queries: acquisition policies live
in separate modules and must call ``query`` explicitly after selecting a
point.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable


class OracleQueryBudgetExceeded(RuntimeError):
    """Raised before a teacher call would exceed the unique-query budget."""


@dataclass(frozen=True)
class OracleStats:
    unique_queries: int
    requests: int
    cache_hits: int
    unique_by_purpose: tuple[tuple[str, int], ...]
    cache_hits_by_purpose: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "unique_queries": self.unique_queries,
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "unique_by_purpose": dict(self.unique_by_purpose),
            "cache_hits_by_purpose": dict(self.cache_hits_by_purpose),
        }


class CachedOracle:
    """A per-task memoizing wrapper around a deterministic teacher callable.

    Args:
        teacher: Callable that accepts one positional value per input axis.
        arity: Optional expected number of input coordinates.
        max_unique_queries: Optional hard cap on distinct teacher calls.

    The wrapper stores integer labels because the current HAGI symbolic task
    interface is integer-valued. It is intentionally ignorant of task
    semantics, train/validation splits and active acquisition policy.
    """

    def __init__(
        self,
        teacher: Callable[..., int],
        *,
        arity: int | None = None,
        max_unique_queries: int | None = None,
    ) -> None:
        if not callable(teacher):
            raise TypeError("teacher must be callable")
        if arity is not None and arity < 1:
            raise ValueError("arity must be positive")
        if max_unique_queries is not None and max_unique_queries < 0:
            raise ValueError("max_unique_queries must be non-negative")
        self._teacher = teacher
        self.arity = arity
        self.max_unique_queries = max_unique_queries
        self._labels: dict[tuple[int, ...], int] = {}
        self._requests = 0
        self._cache_hits = 0
        self._unique_by_purpose: Counter[str] = Counter()
        self._cache_hits_by_purpose: Counter[str] = Counter()

    def query(self, inputs: Iterable[int], *, purpose: str = "unspecified") -> int:
        """Return a cached label or call the teacher once for this tuple."""
        key = tuple(int(value) for value in inputs)
        if self.arity is not None and len(key) != self.arity:
            raise ValueError(f"oracle input arity {len(key)} != expected {self.arity}")
        if not purpose or not isinstance(purpose, str):
            raise ValueError("purpose must be a non-empty string")

        self._requests += 1
        if key in self._labels:
            self._cache_hits += 1
            self._cache_hits_by_purpose[purpose] += 1
            return self._labels[key]

        if (
            self.max_unique_queries is not None
            and len(self._labels) >= self.max_unique_queries
        ):
            raise OracleQueryBudgetExceeded(
                f"unique oracle query budget exhausted ({self.max_unique_queries})"
            )

        label = int(self._teacher(*key))
        self._labels[key] = label
        self._unique_by_purpose[purpose] += 1
        return label

    @property
    def unique_queries(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> dict[tuple[int, ...], int]:
        """Return a defensive snapshot for fingerprints and audit reports."""
        return dict(self._labels)

    def stats(self) -> OracleStats:
        return OracleStats(
            unique_queries=len(self._labels),
            requests=self._requests,
            cache_hits=self._cache_hits,
            unique_by_purpose=tuple(sorted(self._unique_by_purpose.items())),
            cache_hits_by_purpose=tuple(sorted(self._cache_hits_by_purpose.items())),
        )
