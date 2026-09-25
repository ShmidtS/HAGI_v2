"""Tests for the fail-closed paired quality gate."""
from __future__ import annotations

import math

import pytest

from hagi.orchestrator.quality_gate import (
    QualityReport,
    quality_digest,
    validate_quality_report,
)


def _report(**changes: object) -> QualityReport:
    report: QualityReport = {
        "schema": "hagi_paired_quality_v1",
        "seeds": (1234, 2243, 3252),
        "holdout_sha256": "a" * 64,
        "growth_ce": (1.0, 1.1, 0.9),
        # Third seed: 1.1 - 0.9 = 0.2, matching the stated pairing. It read
        # 1.0 before, which made the third delta 0.1 and the mean 0.1667.
        "scratch_ce": (1.2, 1.3, 1.1),
        "training_tokens_growth": 100,
        "training_tokens_scratch": 100,
    }
    report.update(changes)  # type: ignore[typeddict-item]
    return report


def test_accepts_all_seed_improvements() -> None:
    verdict = validate_quality_report(_report(), expected_holdout_sha256="a" * 64)
    assert verdict["quality_supported"] is True
    assert verdict["production_promotion"] is True
    # scratch_ce - growth_ce is (0.2, 0.2, 0.2) across the three seeds.
    assert verdict["mean_gain"] == pytest.approx(0.2)
    assert verdict["paired_deltas"] == pytest.approx((0.2, 0.2, 0.2))
    assert verdict["training_token_ratio"] == 1.0
    assert len(verdict["report_sha256"]) == 64


def test_rejects_one_seed_regression() -> None:
    verdict = validate_quality_report(
        _report(growth_ce=(1.0, 1.1, 1.21)),
        expected_holdout_sha256="a" * 64,
    )
    assert verdict["quality_supported"] is False
    assert verdict["production_promotion"] is False


def test_rejects_tampered_holdout_and_cost() -> None:
    with pytest.raises(ValueError, match="holdout"):
        validate_quality_report(_report(), expected_holdout_sha256="b" * 64)
    with pytest.raises(ValueError, match="budget"):
        validate_quality_report(
            _report(training_tokens_growth=126),
            expected_holdout_sha256="a" * 64,
        )


def test_rejects_nonfinite_metrics() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_quality_report(
            _report(growth_ce=(1.0, math.nan, 0.9)),
            expected_holdout_sha256="a" * 64,
        )


def test_digest_is_deterministic() -> None:
    assert quality_digest(_report()) == quality_digest(_report())
