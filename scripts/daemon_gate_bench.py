#!/usr/bin/env python3
"""Bounded daemon gate: measure fast-path call cost and acceptance on a script.

The daemon fast-path skips the ~160-token self-critique when no feature
fires. That saving was previously proven only by a call counter inside a
unit test; end-to-end cost and acceptance over several turns were never
measured. This harness runs a deterministic scripted generation trace through
``bonsai_evolution_daemon.main`` and reports measured evidence:

  - critique/retry LLM calls actually issued (the gated cost)
  - accepted / rejected / retry-rejected counts
  - wall-clock per turn (harness cost, not model latency -- the model is
    scripted, so this is the orchestration cost the gate adds or removes)

Acceptance is fail-closed: the report exits 2 unless the trace's expected
number of gated calls matches the observed number. A gate that silently stops
gating is a failure, not a pass.

Usage:
    python scripts/daemon_gate_bench.py --output reports/daemon_gate
    python scripts/daemon_gate_bench.py --output reports/daemon_gate --turns 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for candidate in (_SRC, _ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import bonsai_evolution_daemon as daemon  # noqa: E402

# Deterministic clean turns (no feature fires) and flagged turns (one fires).
# The trace repeats so per-turn orchestration cost is measured, not guessed.
_CLEAN = {
    "content": "clear confident answer",
    "reasoning": "",
    "tool_calls": None,
    "finish_reason": "stop",
    "entropy": 0.1,
    "confidence": 0.95,
    "rep": 0.0,
    "ntok": 4,
}
_FLAGGED = {
    "content": "hesitant answer",
    "reasoning": "",
    "tool_calls": None,
    "finish_reason": "stop",
    "entropy": 0.1,
    "confidence": 0.05,
    "rep": 0.0,
    "ntok": 2,
}
_CLEAN_EVERY = 3


def _scripted_result(turn: int) -> dict:
    """Return the scripted candidate for a 1-based turn index."""
    if turn % _CLEAN_EVERY == 0:
        return dict(_FLAGGED)
    return dict(_CLEAN)


def _expected_gated(turns: int) -> int:
    return sum(
        1 for turn in range(1, turns + 1) if turn % _CLEAN_EVERY == 0
    )


def run(output: Path, turns: int) -> dict[str, object]:
    """Run the scripted trace and return the measured report."""
    output.mkdir(parents=True, exist_ok=True)
    state_dir = output / "state"

    previous = (
        daemon.STATE_DIR,
        daemon.STATE_FILE,
        daemon.LOG_FILE,
        daemon.MAX_TURNS,
        daemon.HORIZON_TOKENS,
    )
    daemon.STATE_DIR = str(state_dir)
    state_path = state_dir / "state.json"
    daemon.STATE_FILE = str(state_path)
    daemon.LOG_FILE = str(output / "daemon.log")
    daemon.MAX_TURNS = turns
    daemon.HORIZON_TOKENS = 0
    daemon._RUN = {"stop": False, "timeout": False, "start": 0.0}

    calls = {"critique": 0, "retry": 0, "gen": 0}

    def fake_gen(_messages, with_tools=True):
        calls["gen"] += 1
        return _scripted_result(calls["gen"])

    def fake_critique(_messages, _content):
        calls["critique"] += 1
        return "critique"

    def fake_branch(_messages, prev):
        calls["retry"] += 1
        return prev, {"ent": 0.1, "conf": 0.05, "rep": 0.0, "ntok": 2}

    originals = (daemon.gen_one, daemon.critique, daemon.branch_retry)
    daemon.gen_one = fake_gen
    daemon.critique = fake_critique
    daemon.branch_retry = fake_branch
    try:
        started = time.perf_counter()
        rc = daemon.main()
        elapsed = time.perf_counter() - started
    finally:
        daemon.gen_one, daemon.critique, daemon.branch_retry = originals
        (
            daemon.STATE_DIR,
            daemon.STATE_FILE,
            daemon.LOG_FILE,
            daemon.MAX_TURNS,
            daemon.HORIZON_TOKENS,
        ) = previous

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("daemon state is unreadable after the run") from exc

    accepted = int(state.get("total_accepted", 0))
    rejected = int(state.get("total_rejected", 0))
    retry_rejected = len(state.get("rejected_branches", []))
    observed_gated = calls["critique"]
    expected_gated = _expected_gated(turns)
    return {
        "schema": "hagi_daemon_gate_bench_v1",
        "returncode": rc,
        "turns": turns,
        "scripted_generations": calls["gen"],
        "gated_critique_calls": observed_gated,
        "gated_retry_calls": calls["retry"],
        "expected_gated_calls": expected_gated,
        "gating_matches_expectation": observed_gated == expected_gated,
        "accepted": accepted,
        "rejected": rejected,
        "retry_rejected": retry_rejected,
        "elapsed_s": round(elapsed, 3),
        "orchestration_ms_per_turn": round(1000.0 * elapsed / turns, 3),
        "state_path": str(state_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=15)
    args = parser.parse_args(argv)
    if not 1 <= args.turns <= 120:
        print("daemon gate bench failed: --turns must be in [1, 120]", file=sys.stderr)
        return 2
    try:
        report = run(args.output, args.turns)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"daemon gate bench failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    if not report["gating_matches_expectation"]:
        print(
            "daemon gate bench failed: observed gated calls do not match the "
            "scripted expectation",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
