"""Tests for the supervisor's divergence guard.

The gen-4 autonomous cycle trained sib1 and sib2 to convergence and then
sib3 diverged -- CE 4.07 -> 46.18 across steps 1230..1370 -- while
exiting 0. Exit status alone cannot see that, and accepting the
checkpoint would have fed a diverged expert into the merge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "growth"))

from growth_supervisor import converged  # noqa: E402


def _log(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_accepts_a_converged_run(tmp_path: Path):
    p = _log(tmp_path, "ok.log", [
        "2026-01-01 | step 0 | ce=4.2917 | bpt=5.0",
        "2026-01-01 | step 10 | ce=3.6892 | bpt=5.0",
        "2026-01-01 | step 20 | ce=3.2000 | bpt=5.0",
    ])
    ok, why = converged(p)
    assert ok, why


def test_refuses_a_diverged_run(tmp_path: Path):
    """The exact gen-4 sib3 shape: healthy, then a monotonic blow-up."""
    p = _log(tmp_path, "diverged.log", [
        "2026-01-01 | step 1230 | ce=4.0653 | bpt=5.0",
        "2026-01-01 | step 1240 | ce=5.6122 | bpt=5.0",
        "2026-01-01 | step 1250 | ce=12.1867 | bpt=5.0",
        "2026-01-01 | step 1370 | ce=46.1820 | bpt=66.6",
    ])
    ok, why = converged(p)
    assert not ok
    assert "46.18" in why


def test_a_late_blowup_is_caught_not_just_a_high_start(tmp_path: Path):
    """Only the FINAL ce gates; an early high value is normal."""
    p = _log(tmp_path, "late.log", [
        "2026-01-01 | step 0 | ce=10.3973 | bpt=15.0",
        "2026-01-01 | step 10 | ce=9.1 | bpt=14.0",
        "2026-01-01 | step 20 | ce=3.9 | bpt=5.0",
    ])
    assert converged(p)[0]


def test_refuses_nan_ce(tmp_path: Path):
    p = _log(tmp_path, "nan.log", [
        "2026-01-01 | step 10 | ce=3.2 | bpt=5.0",
        "2026-01-01 | step 20 | ce=nan | bpt=5.0",
    ])
    assert not converged(p)[0]


def test_missing_or_empty_log_is_not_a_failure(tmp_path: Path):
    """A short or freshly resumed run may log nothing; that is not divergence."""
    assert converged(tmp_path / "absent.log")[0]
    assert converged(_log(tmp_path, "empty.log", []))[0]
    assert converged(_log(tmp_path, "header.log", ["hagi 4.2.0 | device cuda"]))[0]


def test_ceiling_is_configurable(tmp_path: Path):
    p = _log(tmp_path, "mid.log", [
        "2026-01-01 | step 10 | ce=7.0 | bpt=9.0",
    ])
    assert not converged(p, ceiling=6.0)[0]
    assert converged(p, ceiling=8.0)[0]


def test_ignores_non_ce_numbers(tmp_path: Path):
    """Only the ce= field is read; bpt/ppl must not be mistaken for it."""
    p = _log(tmp_path, "noise.log", [
        "params: total 99.4M | body 23.9M",
        "2026-01-01 | step 0 | ce=3.9 | bpt=5.6 | ppl=41.2",
        "weight rate: 16.000 bits/weight -> body 0.048 GB",
    ])
    assert converged(p)[0]


def test_exhausted_attempts_still_judge_the_checkpoint(tmp_path: Path, monkeypatch):
    """Regression: the exhausted-attempts path must not return blindly.

    gen-4's sib3 diverged and still reached the merge, because when all
    attempts are spent the function returned ``latest_checkpoint(out)``
    with no convergence check. A destroyed expert must never be handed
    on just because the retry budget ran out.
    """
    import growth_supervisor as gs

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "train:\n  checkpoint_dir: checkpoints/does_not_matter\n", encoding="utf-8"
    )
    ckpt = tmp_path / "step-0001600.pt"
    ckpt.write_bytes(b"x")
    out = tmp_path / "ckpts"
    out.mkdir()
    (out / "step-0001600.pt").write_bytes(b"x")

    log = _log(tmp_path, "diverged_attempts.log", [
        "2026-01-01 | step 1400 | ce=60.4560 | bpt=90.0",
        "2026-01-01 | step 1590 | ce=53.9990 | bpt=70.0",
    ])

    monkeypatch.setattr(gs, "ROOT", tmp_path)
    monkeypatch.setattr(gs, "latest_checkpoint", lambda d: ckpt)
    monkeypatch.setattr(gs, "run", lambda *a, **k: 1)  # every attempt fails

    assert gs.train_phase(cfg, log, "cuda", attempts=2, max_ce=6.0) is None


def test_exhausted_attempts_return_a_converged_checkpoint(tmp_path: Path, monkeypatch):
    """The same path still yields the checkpoint when the run is healthy."""
    import growth_supervisor as gs

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "train:\n  checkpoint_dir: checkpoints/does_not_matter\n", encoding="utf-8"
    )
    ckpt = tmp_path / "step-0001600.pt"
    ckpt.write_bytes(b"x")
    log = _log(tmp_path, "ok_attempts.log", [
        "2026-01-01 | step 1590 | ce=3.2995 | bpt=4.7",
    ])

    monkeypatch.setattr(gs, "ROOT", tmp_path)
    monkeypatch.setattr(gs, "latest_checkpoint", lambda d: ckpt)
    monkeypatch.setattr(gs, "run", lambda *a, **k: 1)

    assert gs.train_phase(cfg, log, "cuda", attempts=2, max_ce=6.0) == ckpt
