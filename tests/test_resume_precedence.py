"""Tests for --resume taking precedence over train.init_from.

A config that sets `train.init_from` (every sibling config does, to
inherit the parent's shared prior) used to SILENTLY swallow `--resume`:
the initialization branch ran first and the resume branch was never
reached. The supervisor -- which resumes from whatever checkpoint is on
disk -- therefore discarded every trained step and restarted from the
parent, exiting 0 each time, so the autonomous cycle could not advance
past the first expert.

Observed on gen-4: a checkpoint at step 1300 on disk, the supervisor
correctly passing `--resume`, and the run logging
`initialized weights from ... (fresh optimizer, step 0)`.

These tests check the ORDER, since that is the whole bug: a resume must
continue, an initialization must start.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train.py"

CFG = "configs/dbridge_gen4_sib_math.yaml"
CKPT = "checkpoints/dbridge_gen4_sib_math/step-0001300.pt"


def _config_sets_init_from() -> bool:
    import yaml

    y = yaml.safe_load((ROOT / CFG).read_text(encoding="utf-8"))
    return bool(y.get("train", {}).get("init_from"))


def _run_dry(args: list[str]) -> str:
    # The child writes em-dashes (logging) and a cp1251 host locale (the
    # default on a Windows box without PYTHONIOENCODING/utf-8 mode) would
    # encode them outside UTF-8, crashing the text-mode reader thread and
    # leaving out.stderr=None. Pin the child to UTF-8 both ways.
    import os

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = subprocess.run(
        [sys.executable, str(TRAIN), "--config", CFG, *args,
         "--device", "cpu", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600, env=env,
    )
    return out.stdout + out.stderr


pytestmark = pytest.mark.skipif(
    not (ROOT / CKPT).exists(),
    reason=f"needs {CKPT} on disk",
)


@pytest.mark.skipif(not _config_sets_init_from(), reason="config sets init_from")
def test_resume_wins_over_a_configured_init_from():
    """The regression this file exists for.

    Before the fix the log said "initialized ... step 0" even with
    --resume passed, because the init_from branch was tested first and
    the resume branch was an `elif` that never ran.
    """
    out = _run_dry(["--resume", CKPT])
    assert "resumed" in out, out[-1500:]
    assert "fresh optimizer, step 0" not in out, (
        "init_from overrode --resume: the run restarted from the parent "
        "and every trained step was discarded"
    )


@pytest.mark.skipif(not _config_sets_init_from(), reason="config sets init_from")
def test_the_resumed_step_is_the_checkpoint_s_step():
    """Not merely 'resumed' -- resumed AT 1300, not at 0."""
    out = _run_dry(["--resume", CKPT])
    m = re.search(r"resumed \S+ at step (\d+)", out)
    assert m, out[-1500:]
    assert int(m.group(1)) == 1300


def test_initialization_still_works_when_no_resume_is_given():
    """The other branch must be untouched: init_from is the recursive-
    growth primitive and config-only runs depend on it."""
    out = _run_dry([])
    assert "initialized weights from" in out, out[-1500:]
    assert "fresh optimizer, step 0" in out


def test_an_explicit_init_from_still_starts_a_fresh_run():
    """--init-from is for starting over on purpose; it must not become a
    resume by accident now that resume has priority over the config."""
    out = _run_dry(["--init-from", CKPT])
    assert "initialized weights from" in out
    assert "fresh optimizer, step 0" in out