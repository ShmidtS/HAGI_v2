"""WAVE 247: merge_bound (MergeQuant.lean), hold taxonomy, OUT_PERIOD."""
import numpy as np
import pytest

from hagi.diagnosis import build_residual_profile, classify_hold
from hagi.invention import basis_out_period
from hagi.trunk_ops import merge_bound


def _hold(y):
    xs = list(range(len(y)))
    p = build_residual_profile([[v] for v in xs], [int(v) for v in y])
    return classify_hold("NO_VALIDATED_PROGRAM", p).value, p


# ---- 1. merge_bound == nonlinear_merge_bound (linAt = L1 + g.(w2-w1)t) ----
def test_merge_bound_matches_lean_linearisation():
    L = lambda w: 10.0 + float(np.sum(np.asarray(w) ** 2))
    g = lambda w: 2 * np.asarray(w, float)
    r = merge_bound(L, g([0., 0.]), [0., 0.], [2., 0.])
    assert r["delta_mid"] == pytest.approx(1.0)   # was 11.0 (L1 dropped)
    assert r["delta_end"] == pytest.approx(4.0)
    assert r["bound"] == pytest.approx(15.0)      # was 25.0
    assert r["L_mid"] == pytest.approx(11.0) and r["certified"]


def test_merge_bound_is_valid_on_a_nonconvex_loss():
    rng = np.random.default_rng(0)
    L = lambda w: float(np.sum(np.tanh(np.asarray(w) * 3.0)))
    for _ in range(200):                          # never claims a false bound
        w1, w2 = rng.normal(size=6), rng.normal(size=6)
        eps = 1e-6
        g = np.array([(L(np.where(np.arange(6) == j, w1 + eps, w1))
                       - L(np.where(np.arange(6) == j, w1 - eps, w1))) / (2 * eps)
                      for j in range(6)])                # per-coordinate
        r = merge_bound(L, g, w1, w2)
        assert r["L_mid"] <= r["bound"] + 1e-9


# ---- 2. hold taxonomy ----
def test_narrow_alphabet_noise_is_random_not_a_representation_gap():
    r = np.random.default_rng(0)
    for k in (4, 8, 16):
        label, p = _hold(r.integers(0, k, 2000))
        assert label == "RANDOM_INCOMPRESSIBLE", k
        assert p.period_hit < 0.3 and p.shift_chance == pytest.approx(3 / k)


def test_periodic_recurrence_laws_are_not_called_noise():
    def fibmod(m, n=4000):
        f, a, b = [], 0, 1
        for _ in range(n):
            f.append(a); a, b = b, (a + b) % m
        return f
    for m, period in ((13, 28), (7, 16), (16, 24)):
        label, p = _hold(fibmod(m))
        assert label == "REPRESENTATION_GAP" and p.period == period


def test_period_probe_reports_unknown_on_sparse_tables():
    r = np.random.default_rng(0)
    label, p = _hold(np.where(r.random(2000) < 0.02, r.integers(1, 256, 2000), 0))
    assert p.period_hit == 0.0 and label == "REPRESENTATION_GAP"


def test_period_probe_ignores_the_dominant_value():
    r = np.random.default_rng(0)
    y = np.where(r.random(2000) < 0.05, 7, 0)      # 95% one value, no law
    _, p = _hold(y)
    assert p.period_hit < 0.95                    # was 0.95 (zero == zero)
    assert p.period_cover < 0.2                   # scored on the 5% minority


# ---- 3. basis_out_period: exhaustive reference on small instances ----
def _reference(pairs):
    for p in range(2, 257):
        for o in range(p):
            if all((x + o) % p == y for x, y in pairs):
                return {"kind": "OUT_PERIOD", "p": p, "o": o}
    return None


@pytest.mark.parametrize("seed", range(12))
def test_basis_out_period_equals_exhaustive_scan(seed):
    r = np.random.default_rng(seed)
    for _ in range(150):
        k = int(r.integers(3, 9))
        mode = r.integers(0, 6)
        xs = r.integers(0, 256, k)
        ys = {0: xs % 2, 1: r.integers(250, 256, k), 2: xs,
              3: r.integers(0, 256, k),
              4: (xs + int(r.integers(0, 26))) % int(r.choice([3, 13, 26])),
              5: (xs * 7 + 3) % 256}[mode]
        pairs = list(zip(map(int, xs), map(int, ys)))
        assert basis_out_period(pairs) == _reference(pairs), pairs


def test_basis_out_period_is_fast():
    import time
    pairs = [(x, (x + 5) % 13) for x in range(256)]
    t = time.perf_counter(); out = basis_out_period(pairs)
    assert out == {"kind": "OUT_PERIOD", "p": 13, "o": 5}
    assert time.perf_counter() - t < 0.01
