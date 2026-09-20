"""Safe online-learning queue for adapters/TTT backends.

The queue is intentionally backend-agnostic. A trainer callback can consume
trusted records and update a LoRA/TTT adapter outside the inference critical path.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hagi.inference.adaptive import InferenceResult, RouteDecision


@dataclass(frozen=True)
class LearningExample:
    timestamp: float
    request: Any
    output: Any
    domain: str
    route: str
    adapter_ids: tuple[str, ...]
    trusted: bool


class ReplayBufferLearner:
    """Versioned replay queue with strict trusted-supervision gating."""

    def __init__(
        self,
        *,
        capacity: int = 512,
        persist_path: str | Path | None = None,
        trainer: Callable[[list[LearningExample]], bool] | None = None,
        trusted_only: bool = True,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.persist_path = Path(persist_path) if persist_path else None
        self.trainer = trainer
        self.trusted_only = trusted_only
        self._items: deque[LearningExample] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.version = 0

    def enqueue(self, request: Any, result: InferenceResult, decision: RouteDecision) -> bool:
        trusted = bool(result.metadata.get("trusted_supervision", False))
        if self.trusted_only and not trusted:
            return False
        item = LearningExample(
            timestamp=time.time(),
            request=request,
            output=result.output,
            domain=decision.domain,
            route=decision.route.value,
            adapter_ids=tuple(decision.adapter_ids),
            trusted=trusted,
        )
        with self._lock:
            self._items.append(item)
            self._persist_locked()
        return True

    def snapshot(self) -> list[LearningExample]:
        with self._lock:
            return list(self._items)

    def train_once(self, batch_size: int = 8) -> bool:
        if self.trainer is None:
            return False
        batch = self.snapshot()[-max(1, batch_size):]
        if not batch:
            return False
        ok = bool(self.trainer(batch))
        if ok:
            self.version += 1
        return ok

    def _persist_locked(self) -> None:
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [e.__dict__ for e in self._items]
        self.persist_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
