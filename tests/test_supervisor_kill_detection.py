"""Tests for the killed-vs-crashed distinction in the supervisor.

A training child can die two ways, and the retry loop must not report
them the same way:

  - it crashes, leaving a traceback in the log -- a training problem;
  - it is killed, leaving the log simply truncated -- almost always
    another process holding the GPU.

The second was observed twice on gen-4: two supervisors launched on the
same lane, and each killed the other's child while the log just stopped.
Both exits were non-zero with no traceback, so nothing said "look for a
duplicate supervisor" and the next occurrence would have to be
debugged from scratch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "growth"))

import growth_supervisor as gs  # noqa: E402


# --- the classifier -----------------------------------------------------


@pytest.mark.parametrize("tail", [
    "2026-10-02 16:12:53 | step 120 | ce=3.7315 | bpt=5.383",
    "",
    "saved checkpoint",
    "some ordinary training output without any marker",
])
def test_a_silent_stop_is_not_a_crash(tail):
    """The case that was misread: the log just ends."""
    assert gs.looks_like_exception(tail) is False


@pytest.mark.parametrize("tail", [
    "Traceback (most recent call last):\n  File ...",
    "RuntimeError: Expected all tensors to be on the same device",
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "HIP error: unspecified launch failure",
    "ValueError: keep must lie in (0, 384]",
    "AssertionError",
])
def test_a_traceback_is_a_crash(tail):
    assert gs.looks_like_exception(tail) is True


def test_the_two_cases_are_actually_distinguishable():
    """The property the diagnostic exists for."""
    crashed = "Traceback (most recent call last):\nRuntimeError: boom"
    killed = "2026-10-02 16:12:53 | step 120 | ce=3.7315"
    assert gs.looks_like_exception(crashed) != gs.looks_like_exception(killed)


# --- log_tail -----------------------------------------------------------


def test_log_tail_returns_the_last_lines(tmp_path):
    p = tmp_path / "t.log"
    p.write_text("a\nb\nc\nd\n", encoding="utf-8")
    assert gs.log_tail(p, lines=2) == "c\nd"


def test_log_tail_on_a_missing_file_is_empty_not_an_exception(tmp_path):
    """A vanished log must not turn a retry into a crash report."""
    assert gs.log_tail(tmp_path / "nope.log") == ""


def test_log_tail_of_an_empty_file_is_empty(tmp_path):
    p = tmp_path / "e.log"
    p.write_text("", encoding="utf-8")
    assert gs.log_tail(p) == ""


def test_log_tail_handles_a_single_line(tmp_path):
    p = tmp_path / "one.log"
    p.write_text("only\n", encoding="utf-8")
    assert gs.log_tail(p, lines=3) == "only"


def test_log_tail_of_a_real_training_log_looks_like_a_stop(tmp_path):
    """Reproduces the observed gen-4 case end to end."""
    p = tmp_path / "gen4.log"
    p.write_text(
        "2026-10-02 16:11:51 | step 100 | ce=3.7584 | bpt=5.422\n"
        "2026-10-02 16:12:24 | step 110 | ce=3.3094 | bpt=4.774\n"
        "2026-10-02 16:13:21 | step 130 | ce=3.5008 | bpt=5.051\n",
        encoding="utf-8",
    )
    tail = gs.log_tail(p, lines=3)
    assert gs.looks_like_exception(tail) is False