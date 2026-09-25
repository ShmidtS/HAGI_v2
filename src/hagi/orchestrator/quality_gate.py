"""Fail-closed verifier for paired growth quality reports.

no reference: the repository already owns the transactional growth state and
exact-CE evaluator; this module only adds a small schema/verdict layer.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypedDict

from hagi.orchestrator.state import canonical_json_bytes, sha256_bytes


class QualityReport(TypedDict):
    schema: str
    seeds: tuple[int, int, int]
    holdout_sha256: str
    growth_ce: tuple[float, float, float]
    scratch_ce: tuple[float, float, float]
    training_tokens_growth: int
    training_tokens_scratch: int


def quality_digest(report: QualityReport) -> str:
    return sha256_bytes(canonical_json_bytes(report))


def validate_quality_report(
    report: QualityReport,
    *,
    expected_holdout_sha256: str,
    min_gain: float = 0.0,
    max_token_ratio: float = 1.25,
) -> dict[str, object]:
    """Validate paired evidence and return a machine-readable verdict."""
    if report.get("schema") != "hagi_paired_quality_v1":
        raise ValueError("invalid paired quality schema")
    if report.get("seeds") != (1234, 2243, 3252):
        raise ValueError("quality seeds are not preregistered")
    if report.get("holdout_sha256") != expected_holdout_sha256:
        raise ValueError("quality holdout binding mismatch")
    for field in ("growth_ce", "scratch_ce"):
        values = report.get(field)
        if not isinstance(values, tuple) or len(values) != 3:
            raise ValueError("paired quality metrics must contain three seeds")
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise ValueError("paired quality metrics must be finite floats")
    growth_tokens = report.get("training_tokens_growth")
    scratch_tokens = report.get("training_tokens_scratch")
    if type(growth_tokens) is not int or type(scratch_tokens) is not int:
        raise ValueError("training token budgets must be integers")
    if growth_tokens <= 0 or scratch_tokens <= 0:
        raise ValueError("training token budgets must be positive")
    if growth_tokens > scratch_tokens * max_token_ratio:
        raise ValueError("growth training budget exceeds declared limit")
    deltas = tuple(scratch - growth for scratch, growth in zip(
        report["scratch_ce"], report["growth_ce"], strict=True
    ))
    mean_gain = sum(deltas) / len(deltas)
    accepted = all(delta >= min_gain for delta in deltas) and mean_gain >= min_gain
    return {
        "schema": "hagi_paired_quality_verdict_v1",
        "quality_supported": accepted,
        "production_promotion": accepted,
        "mean_gain": mean_gain,
        "paired_deltas": deltas,
        "holdout_sha256": expected_holdout_sha256,
        "report_sha256": quality_digest(report),
        "training_token_ratio": growth_tokens / scratch_tokens,
    }


__all__ = ["QualityReport", "quality_digest", "validate_quality_report"]
