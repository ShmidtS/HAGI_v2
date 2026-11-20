"""C477: Hadamard incoherence gate (Theory/HadamardIncoherence.lean)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from hagi.hadamard import sylvester, coh_sq, rotate_gate, quantize_rows_with_gate


def test_sylvester_parseval():
    H = sylvester(3)
    assert H.shape == (8, 8) and set(np.unique(H)) <= {-1, 1}
    # H symmetric
    assert np.array_equal(H, H.T)
    # integer Parseval: ||Hw||^2 = 2^k ||w||^2
    w = np.random.default_rng(5).normal(size=8)
    hw = H @ w
    assert abs((hw * hw).sum() / (w * w).sum() - 8.0) < 1e-9


def test_cohsq_scale_invariance():
    w = np.array([3.0, -1.0, 0.5, 2.0])
    assert abs(coh_sq(w) - coh_sq(7.5 * w)) < 1e-12


def test_gate_honest_on_constant():
    # rotate_hurts_constant: gate must REFUSE
    r = rotate_gate(np.full(8, 2.0))
    assert r["rotate"] is False
    assert r["cohSq_after"] > r["cohSq_before"]


def test_gate_accepts_sparse():
    r = rotate_gate(np.array([9.0, 0, 0, 0, 0, 0, 0, 0], dtype=float))
    assert r["rotate"] is True
    assert r["cohSq_after"] < r["cohSq_before"]


def test_quantize_rows_gate_roundtrip_shapes():
    rng = np.random.default_rng(7)
    W = rng.normal(size=(4, 8))
    Q, flags = quantize_rows_with_gate(W)
    assert Q.shape == W.shape
    assert len(flags) == 4 and all(isinstance(f, bool) for f in flags)
    # levels ternary in each row's OWN basis (s from the pre-quantized
    # rotated row; per-row scheme is not idempotent — wave 160)
    H = sylvester(3).astype(float)
    for i in range(4):
        v = W[i]
        if rotate_gate(v)["rotate"]:
            v = H @ v
        s = np.abs(v).mean() or 1.0
        assert np.abs(np.round(Q[i] / s)).max() <= 1.0 + 1e-9
