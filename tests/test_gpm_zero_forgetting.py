"""C485: GPM zero-forgetting (Theory/ContinualBench.lean).

Contract: after consolidate_subspace(X_A) with full span, training
task B leaves loss(A) EXACTLY invariant (gpm_zero_forgetting):
Delta_W lives in null(U^T) => X_A @ Delta_W = 0.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hagi.core import ContinualGPMMemory


def _scenario(seed: int):
    gpm = ContinualGPMMemory(d_in=16, d_out=1, seed=seed)
    XA = np.random.default_rng(seed).normal(size=(24, 16))
    YA = (XA @ gpm.W).round()
    gpm.train_task_ef21_muon(XA, YA, steps=15)
    gpm.consolidate_subspace(XA, energy_thresh=1.0)
    before = float(np.mean((XA @ gpm.W - YA) ** 2))
    XB = np.random.default_rng(seed + 100).normal(size=(24, 16))
    YB = (XB @ np.random.default_rng(seed + 200).normal(size=(16, 1))).round()
    gpm.train_task_ef21_muon(XB, YB, steps=15)
    after = float(np.mean((XA @ gpm.W - YA) ** 2))
    return before, after


def test_zero_forgetting_exact_multi_seed():
    for seed in (1, 2, 7):
        before, after = _scenario(seed)
        assert after <= before + 1e-9, (
            f"seed {seed}: loss(A) grew {before} -> {after} "
            f"(+{(after-before)/max(before,1e-12)*100:.1f}%)")


def test_full_span_consolidation():
    gpm = ContinualGPMMemory(d_in=16, d_out=1, seed=3)
    X = np.random.default_rng(4).normal(size=(24, 16))
    gpm.consolidate_subspace(X, energy_thresh=1.0)
    # full span: rank equals d_in (no silent trailing-direction drop)
    assert gpm.U_basis.shape[1] == 16
    # orthonormality of the protected basis
    gram = gpm.U_basis.T @ gpm.U_basis
    assert np.allclose(gram, np.eye(16), atol=1e-7)
