"""C479: unified six-phase cycle — glue semantics (commit/hold honesty)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hagi.full_pipeline import six_phase_cycle

OBJ = lambda w: float(0.5 * np.sum((w - 3.0) ** 2))
GRAD = lambda w: (w - 3.0)


def test_train_commits_certified():
    r = six_phase_cycle(OBJ, GRAD, np.array([10.0, -4.0, 7.0, 1.0]),
                        steps_train=5)
    tr = {t["phase"]: t for t in r["trace"]}
    assert tr["TRAIN"]["commit"] and tr["TRAIN"]["drop"] > 0


def test_ef21_reject_is_hold():
    # already-converged weights: quantized step cannot improve -> HOLD
    w0 = np.array([3.0, 3.0, 3.0, 3.0])
    r = six_phase_cycle(OBJ, GRAD, w0, steps_train=1)
    tr = {t["phase"]: t for t in r["trace"]}
    assert tr["EF21"]["commit"] is False   # reject-hold semantics


def test_merge_gate_meas_required():
    # merge candidate that does not beat the bound -> honest no-op
    r = six_phase_cycle(OBJ, GRAD, np.array([10.0, -4.0, 7.0, 1.0]),
                        steps_train=2, merge_cand=np.array([50.0, 50.0,
                                                            50.0, 50.0]))
    tr = {t["phase"]: t for t in r["trace"]}
    assert tr["MERGE"]["commit"] is False


def test_stop_blocks_after_stall():
    r = six_phase_cycle(OBJ, GRAD, np.array([3.0, 3.0, 3.0, 3.0]),
                        steps_train=1, stop_patience=2)
    tr = {t["phase"]: t for t in r["trace"]}
    assert tr["STOP"]["commit"] is False   # Ville stop blocks commits
