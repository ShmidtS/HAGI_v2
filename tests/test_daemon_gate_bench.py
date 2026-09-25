"""Tests for the bounded daemon gate benchmark harness.

The harness is the measurement lane for the fast-path: it must keep gating
exactly when a feature fires, and it must keep failing closed when it does
not. A harness that passes vacuously is worse than no harness.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "daemon_gate_bench.py"
_SPEC = importlib.util.spec_from_file_location("daemon_gate_bench", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def test_scripted_trace_marks_every_third_turn_gated() -> None:
    assert bench._CLEAN_EVERY == 3
    assert bench._expected_gated(9) == 3
    assert bench._expected_gated(1) == 0
    # A turn is flagged exactly when the count says so.
    for turn in range(1, 10):
        result = bench._scripted_result(turn)
        assert result["confidence"] == pytest.approx(
            0.05 if turn % bench._CLEAN_EVERY == 0 else 0.95
        )


def test_measure_matches_scripted_gated_calls(tmp_path: Path) -> None:
    report = bench.run(tmp_path / "run", 9)
    assert report["returncode"] == 0
    assert report["turns"] == 9
    assert report["gated_critique_calls"] == 3
    assert report["gated_retry_calls"] == 3
    assert report["expected_gated_calls"] == 3
    assert report["gating_matches_expectation"] is True
    assert report["scripted_generations"] == 9
    assert report["accepted"] == 9
    assert report["rejected"] == 0
    assert report["orchestration_ms_per_turn"] > 0


def test_cli_exits_two_when_gating_disagrees(tmp_path: Path, monkeypatch) -> None:
    """Fail closed: a gate that stops gating must exit 2, not report a pass."""
    monkeypatch.setattr(bench, "_expected_gated", lambda turns: turns)
    rc = bench.main(["--output", str(tmp_path / "run"), "--turns", "6"])
    assert rc == 2


def test_cli_rejects_out_of_range_turns(tmp_path: Path) -> None:
    assert bench.main(["--output", str(tmp_path / "run"), "--turns", "0"]) == 2
    assert bench.main(["--output", str(tmp_path / "run"), "--turns", "999"]) == 2


def test_report_is_json_serializable(tmp_path: Path) -> None:
    report = bench.run(tmp_path / "run", 3)
    encoded = json.dumps(report, sort_keys=True)
    assert json.loads(encoded)["schema"] == "hagi_daemon_gate_bench_v1"
