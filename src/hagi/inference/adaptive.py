"""Model-agnostic adaptive inference orchestration primitives.

This module intentionally does not inspect or mask dense model weights. It routes
among explicitly registered execution paths and keeps online learning opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence


class Route(str, Enum):
    SMALL = "small_model"
    SPECIALIST = "specialist"
    MAIN = "main_model"
    MAIN_LORA = "main_model_with_lora"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    domain: str
    confidence: float
    risk: float = 1.0
    adapter_ids: tuple[str, ...] = ()
    reason: str = "router"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        if not self.domain.strip():
            raise ValueError("domain must be non-empty")


@dataclass(frozen=True)
class RouteCandidate:
    route: Route
    domain: str
    estimated_latency_ms: float
    max_risk: float = 1.0
    adapter_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.estimated_latency_ms < 0:
            raise ValueError("estimated latency must be non-negative")
        if not 0.0 <= self.max_risk <= 1.0:
            raise ValueError("max_risk must be in [0, 1]")


@dataclass(frozen=True)
class InferenceResult:
    output: Any
    route: Route
    accepted: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceTrace:
    decision: RouteDecision
    selected_route: Route
    fallback_used: bool
    elapsed_ms: float
    verifier_accepted: bool
    learner_enqueued: bool


class Router(Protocol):
    def predict(self, request: Any) -> RouteDecision: ...


class Verifier(Protocol):
    def verify(self, request: Any, result: InferenceResult) -> bool: ...


class Learner(Protocol):
    def enqueue(self, request: Any, result: InferenceResult, decision: RouteDecision) -> bool: ...


class ParameterMap:
    """Registry of explicitly available routes; no arbitrary tensor slicing."""

    def __init__(self, candidates: Sequence[RouteCandidate]):
        self._candidates = tuple(candidates)
        if not self._candidates:
            raise ValueError("at least one route candidate is required")

    def get_candidates(self, domain: str) -> tuple[RouteCandidate, ...]:
        exact = tuple(c for c in self._candidates if c.domain == domain)
        generic = tuple(c for c in self._candidates if c.domain == "*")
        return exact + generic


class BudgetPolicy:
    """Select lowest estimated latency among candidates satisfying risk/budget."""

    def select(
        self,
        candidates: Sequence[RouteCandidate],
        decision: RouteDecision,
        latency_budget_ms: float,
    ) -> RouteCandidate:
        feasible = [
            c for c in candidates
            if c.estimated_latency_ms <= latency_budget_ms
            and decision.risk <= c.max_risk
        ]
        if not feasible:
            # Fail safe: choose the least-risk candidate, then let executor enforce
            # its own hard budget/timeout. Never silently claim budget compliance.
            feasible = list(candidates)
        if not feasible:
            raise RuntimeError(f"no execution route available for domain={decision.domain!r}")
        return min(feasible, key=lambda c: (c.estimated_latency_ms, c.max_risk))


class AdaptiveInferenceController:
    """Synchronous orchestration; model-specific execution is injected as callables."""

    def __init__(
        self,
        router: Router,
        parameter_map: ParameterMap,
        executor: Callable[[Any, RouteCandidate], InferenceResult],
        verifier: Verifier,
        *,
        policy: BudgetPolicy | None = None,
        learner: Learner | None = None,
        fallback: Callable[[Any, InferenceResult], InferenceResult] | None = None,
        shadow: bool = False,
    ) -> None:
        self.router = router
        self.parameter_map = parameter_map
        self.executor = executor
        self.verifier = verifier
        self.policy = policy or BudgetPolicy()
        self.learner = learner
        self.fallback = fallback
        self.shadow = shadow

    def generate(self, request: Any, latency_budget_ms: float = float("inf")) -> tuple[InferenceResult, InferenceTrace]:
        started = perf_counter()
        decision = self.router.predict(request)
        candidates = self.parameter_map.get_candidates(decision.domain)
        if not candidates:
            raise RuntimeError(f"no route candidates for domain={decision.domain!r}")
        selected = self.policy.select(candidates, decision, latency_budget_ms)

        # Shadow mode measures routing while preserving baseline route selection:
        # the baseline is represented by a candidate with route=MAIN.
        if self.shadow:
            selected = next((c for c in candidates if c.route == Route.MAIN), selected)

        result = self.executor(request, selected)
        accepted = self.verifier.verify(request, result)
        fallback_used = False
        if not accepted:
            if self.fallback is None:
                result = InferenceResult(result.output, result.route, accepted=False,
                                         metadata={**dict(result.metadata), "verification_failed": True})
            else:
                result = self.fallback(request, result)
                fallback_used = True
                accepted = self.verifier.verify(request, result)

        enqueued = False
        if self.learner is not None and accepted:
            # Learner is responsible for requiring trusted supervision; verifier
            # acceptance alone must not imply permission to self-train.
            enqueued = bool(self.learner.enqueue(request, result, decision))

        trace = InferenceTrace(
            decision=decision,
            selected_route=selected.route,
            fallback_used=fallback_used,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            verifier_accepted=accepted,
            learner_enqueued=enqueued,
        )
        return result, trace
