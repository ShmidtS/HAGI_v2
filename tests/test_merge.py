"""NonlinearMerge: certified checkpoint-merge bound."""
import numpy as np
from hagi.merge import merge_bound, diag_quadratic_merge


def test_merge_bound_holds_on_quadratic():
    rng = np.random.default_rng(5)
    w1 = rng.normal(size=16); w2 = rng.normal(size=16)
    c = rng.uniform(0.5, 2.0, 16); a = rng.normal(size=16)
    L = lambda w: 0.5 * float(np.sum(c * (w - a) ** 2))
    g = c * (w1 - a)
    r = merge_bound(L, g, w1, w2)
    assert r["certified"]


def test_diag_quadratic_exact_bridge():
    rng = np.random.default_rng(6)
    w1 = rng.normal(size=10); w2 = rng.normal(size=10)
    c = rng.uniform(0.2, 3.0, 10); a = rng.normal(size=10)
    r = diag_quadratic_merge(w1, w2, c, a)
    assert r["exact"]
    # curvature residual is exactly (1/8)Σc(w2−w1)²
    expect = float(np.sum(c * (w2 - w1) ** 2)) / 8.0
    assert abs(r["delta_curv"] - expect) < 1e-12


def test_merge_flat_is_free():
    # identical points: bound tight, residual 0
    w = np.ones(4) * 2.0
    r = diag_quadratic_merge(w, w, np.ones(4), np.zeros(4))
    assert r["delta_curv"] == 0.0 and r["exact"]
