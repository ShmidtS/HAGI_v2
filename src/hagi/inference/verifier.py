"""Cheap post-generation verification hooks for adaptive inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hagi.inference.adaptive import InferenceResult


@dataclass(frozen=True)
class VerificationReport:
    accepted: bool
    trusted_supervision: bool = False
    reason: str = ""
    score: float = 1.0


class GenerationVerifier:
    """Low-cost structural verifier.

    This deliberately does not pretend to know semantic correctness. Projects
    can compose it with an exact solver, unit-test runner, or stronger judge.
    """

    def __init__(
        self,
        *,
        require_nonempty: bool = True,
        max_output_tokens: int | None = None,
        trust_predicate: Callable[[Any], bool] | None = None,
    ):
        self.require_nonempty = require_nonempty
        self.max_output_tokens = max_output_tokens
        self.trust_predicate = trust_predicate

    def report(self, result: InferenceResult) -> VerificationReport:
        output = result.output
        if self.require_nonempty and (output is None or not str(output).strip()):
            return VerificationReport(False, False, "empty_output", 0.0)
        if self.max_output_tokens is not None:
            tokens = output if isinstance(output, (list, tuple)) else str(output).split()
            if len(tokens) > self.max_output_tokens:
                return VerificationReport(False, False, "output_limit", 0.0)
        trusted = bool(self.trust_predicate(output)) if self.trust_predicate else False
        return VerificationReport(True, trusted, "structural_ok", 1.0)

    def verify(self, request: Any, result: InferenceResult) -> bool:
        return self.report(result).accepted
