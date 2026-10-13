"""Tests for the certified merge selector (ALGORITHMS.md §1, §9).

``hagi.orchestrator.merge_select`` keeps the cheap standalone-CE screen
ONLY as a pre-filter; the DECISION routes through the §9 certified gate
(reused from ``certified_controller.certified_ab``) exactly as §1's
``selection_hurts`` demands: the screen winner is committed only on a
certified gap ``CE_incumbent - CE_winner > 2*eps`` at the Hoeffding
sample size; UNDECIDED falls back to the documented not-worse choice
(keep the incumbent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hagi.orchestrator.merge_select import (  # noqa: E402
    enumerate_candidates,
    screen_steps_for,
    select_screened,
)


def _row(label: str, ce: float) -> dict:
    return {"label": label, "screen_exact_ce": ce}


# --- §1: candidate enumeration (pre-filter surface, unchanged) ------------


def test_enumerate_candidates_covers_all_scales():
    cands = enumerate_candidates(["a.pt", "b.pt"])
    assert [c.mixer_init_scale for c in cands] == [0.0, 0.05]
    assert all(c.experts == ("a.pt", "b.pt") for c in cands)


def test_screen_steps_floor_and_fraction():
    assert screen_steps_for(1000) == 200       # floor
    assert screen_steps_for(9000) == 540       # 6%


# --- §9: certified selection over the screened candidates -----------------


def test_certified_win_selects_screen_winner():
    # incumbent 3.20, winner 3.00: gap 0.20 > 2*eps=0.10 at n=1000 >= 738
    scored = [_row("all24_ms0.0", 3.20), _row("all24_ms0.05", 3.00)]
    winner, meta = select_screened(
        scored, incumbent_label="all24_ms0.0", eps=0.05, delta=0.05, eval_n=1000,
    )
    assert winner["label"] == "all24_ms0.05"
    assert meta["decision"] == "CERTIFIED_WIN"
    assert meta["verdict"] == "ACCEPT"


def test_certified_rejection_keeps_incumbent():
    # external incumbent CE 3.00; the screen winner (3.40) is certified
    # WORSE -> REJECT -> keep the incumbent (not-worse by certification)
    scored = [_row("all24_ms0.0", 3.50), _row("all24_ms0.05", 3.40)]
    winner, meta = select_screened(
        scored, incumbent_ce=3.00, incumbent_n=1000,
        eps=0.05, delta=0.05, eval_n=1000,
    )
    assert winner["label"] == "incumbent"
    assert meta["decision"] == "CERTIFIED_REJECT_KEEP_INCUMBENT"
    assert meta["verdict"] == "REJECT"


def test_undecided_falls_back_to_not_worse_incumbent():
    # gap 0.05 < 2*eps=0.10 at n=1000: inside the certification band ->
    # UNDECIDED -> documented not-worse fallback keeps the incumbent
    scored = [_row("all24_ms0.0", 3.20), _row("all24_ms0.05", 3.15)]
    winner, meta = select_screened(
        scored, incumbent_label="all24_ms0.0", eps=0.05, delta=0.05, eval_n=1000,
    )
    assert winner["label"] == "all24_ms0.0"
    assert meta["decision"] == "UNDECIDED_NOT_WORSE_FALLBACK"
    assert meta["verdict"] == "UNDECIDED"


def test_undecided_for_sample_size_also_keeps_incumbent():
    # clear gap 0.30 but n=10 << n_required: UNDECIDED, not a win
    scored = [_row("all24_ms0.0", 3.50), _row("all24_ms0.05", 3.20)]
    winner, meta = select_screened(
        scored, incumbent_label="all24_ms0.0", eps=0.05, delta=0.05, eval_n=10,
    )
    assert winner["label"] == "all24_ms0.0"
    assert meta["decision"] == "UNDECIDED_NOT_WORSE_FALLBACK"


def test_no_incumbent_row_returns_prefilter_argmin():
    # K=1 / first lane: nothing to certify against -> plain pre-filter
    scored = [_row("all24_ms0.0", 3.20), _row("all24_ms0.05", 3.10)]
    winner, meta = select_screened(scored, eps=0.05, delta=0.05)
    assert winner["label"] == "all24_ms0.05"
    assert meta["decision"] == "PREFILTER_ARGMIN"


def test_incumbent_itself_is_argmin_passes_through():
    scored = [_row("all24_ms0.0", 3.00), _row("all24_ms0.05", 3.30)]
    winner, meta = select_screened(
        scored, incumbent_label="all24_ms0.0", eps=0.05, delta=0.05,
    )
    assert winner["label"] == "all24_ms0.0"
    assert meta["decision"] == "PREFILTER_ARGMIN"


def test_no_scored_candidate_returns_none():
    winner, meta = select_screened(
        [{"label": "x", "screen_exact_ce": None}], incumbent_label="x",
    )
    assert winner is None
    assert meta["decision"] == "NO_SCORED_CANDIDATE"
