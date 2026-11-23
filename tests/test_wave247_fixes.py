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


# ---- 3b. invent_multi: period search (WAVE 247) ----
K = 3


def _stream(period, fn, n=3000, seed=0):
    import random
    r = random.Random(seed)
    data = bytearray(r.randbytes(n + K))
    for i in range(K, len(data)):
        if (i - K) % period == 0:
            data[i] = fn(data[i - 3], data[i - 2], data[i - 1]) & 255
    return [(tuple(data[i - K:i]), data[i]) for i in range(K, len(data))]


@pytest.mark.parametrize("period", [3, 4, 5, 6, 7, 8])
def test_invent_multi_finds_laws_on_every_period_up_to_8(period):
    # 6000 bytes: every stratum then has >= 750 pairs. With 3000 the
    # period-8 stratum (375 pairs) is too small for a near-injective
    # law — it is still FOUND, with support 0.744 and raw accuracy
    # 0.971 (the PARTIAL_LAW contract reports exactly that).
    import random
    from hagi.invention_multi import invent_multi
    pairs = _stream(period, lambda a, b, c: a + 2 * b - c + 7, n=6000)
    r = invent_multi(pairs, random.Random(0))
    assert r["status"] == "PARTIAL_LAW"
    bp, P, s = r["blueprints"][0], r["n_shifts"], r["shift"]
    assert P == period
    idx = [i for i in range(len(pairs)) if i % P == s]
    assert sum(bp.apply(pairs[i][0]) == pairs[i][1] for i in idx) / len(idx) > 0.99


def test_invent_multi_best_law_wins_over_the_first_that_passes():
    # A weaker fit on a MULTIPLE of the period passes every gate at an
    # earlier feature; the search must return the true period.
    import random
    from hagi.invention_multi import invent_multi
    pairs = _stream(4, lambda a, b, c: a + 2 * b - c + 7)
    r = invent_multi(pairs, random.Random(0))
    assert r["n_shifts"] == 4 and r["feature"] == "aff(1, 2, -1)+0"


def test_invent_multi_never_invents_a_law_for_noise():
    import random
    from hagi.invention_multi import invent_multi
    for seed in range(8):
        r0 = random.Random(100 + seed)
        pairs = [(tuple(r0.randbytes(K)), r0.randbytes(1)[0])
                 for _ in range(3000)]
        assert invent_multi(pairs, random.Random(seed))["status"] == "NO_LAW"


def test_vectorised_features_are_bit_exact_with_the_callables():
    from hagi.invention_multi import _kind, _vfeat, enumerate_features
    r = np.random.default_rng(0)
    for k in (2, 3, 4):
        X = r.integers(0, 256, (2000, k), dtype=np.uint8)
        for name, param, fn in enumerate_features(k):
            got = _vfeat(_kind(name), param, X)
            ref = np.fromiter((fn(tuple(int(v) for v in row)) for row in X),
                              dtype=np.int64, count=len(X))
            assert not (got != ref).any(), name


# ---- 4. compile_factored_gf2 / compile_optimal (WAVE 247) ----
def _anf(n, fn):
    outs = []
    for o in range(len(fn((0,) * n))):
        T = set()
        for s in range(1 << n):
            c = 0
            for t in range(1 << n):
                if t & s == t:
                    c ^= fn(tuple((t >> i) & 1 for i in range(n)))[o]
            if c:
                T.add(s)
        outs.append(frozenset(T))
    return tuple(outs)


def _adder2(x):
    s = x[0] + 2 * x[1] + x[2] + 2 * x[3]
    return ((s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1)


def _mul2(x):
    p = (x[0] + 2 * x[1]) * (x[2] + 2 * x[3])
    return ((p >> 3) & 1, (p >> 2) & 1, (p >> 1) & 1, p & 1)


def _exact(net, n, fn):
    import hagi.skill_engine as S
    return S.symbolic_verify_general(net, _anf(n, fn)) and all(
        S.eval_net(net, tuple((q >> i) & 1 for i in range(n)))
        == fn(tuple((q >> i) & 1 for i in range(n))) for q in range(1 << n))


def test_compile_optimal_shrinks_the_2x2_multiplier():
    import hagi.skill_engine as S
    net = S.ProofCarryingANFLearner.compile_optimal(4, _anf(4, _mul2))
    m = net["metrics"]
    assert len(net["gates"]) <= 7 and m["depth"] <= 3 and m["program_bits"] <= 100
    assert _exact(net, 4, _mul2)


def test_compile_optimal_keeps_the_adder_and_is_never_worse_than_before():
    import hagi.skill_engine as S
    net = S.ProofCarryingANFLearner.compile_optimal(4, _anf(4, _adder2))
    assert len(net["gates"]) == 8 and _exact(net, 4, _adder2)
    r = np.random.default_rng(0)
    for n in (3, 4, 5):
        for _ in range(8):                       # random functions
            outs = []
            for _ in range(int(r.integers(1, 4))):
                tt = r.integers(0, 2, 1 << n)
                outs.append(frozenset(s for s in range(1 << n)
                                      if sum(tt[t] for t in range(1 << n) if t & s == t) % 2))
            outs = tuple(outs)
            opt = S.ProofCarryingANFLearner.compile_optimal(n, outs)
            base = S.compile_factored_gf2(n, outs)          # flag off
            assert S.symbolic_verify_general(opt, outs)
            key = lambda nt: (len(nt["gates"]), nt["metrics"]["depth"], nt["metrics"]["program_bits"])
            assert key(opt) <= min(key(base), key(S.compact_netlist(S.ProofCarryingANFLearner.compile(n, outs))))


# ---- 5. Polar-Express certificate (WAVE 248, PolarExpressCert.lean) ----
def test_polar_express_singular_spectrum_lies_in_the_lean_box():
    from hagi.trunk_ops import polar_express
    # The function acts on singular values only; feed it as a small
    # diagonal matrix (U = V = I, spectrum = diagonal) in batches.
    # (A first version built one 20001x20001 diagonal — ~8e12 flops.)
    for lo, box in ((0.1, (0.9075, 1.0212)), (1e-3, (9 / 512, 1049 / 1024))):
        s = np.linspace(lo, 1.0, 20001)
        for i in range(0, len(s), 64):
            out = np.linalg.svd(polar_express(np.diag(s[i:i + 64])), compute_uv=False)
            assert out.min() >= box[0] - 1e-9 and out.max() <= box[1] + 1e-9, (lo, out.min(), out.max())


def test_polar_express_handles_wide_rank_deficient_and_zero():
    from hagi.trunk_ops import polar_express
    r = np.random.default_rng(0)
    G = r.normal(size=(8, 2)) @ r.normal(size=(2, 64))          # wide, rank 2
    out = polar_express(G)
    s = np.linalg.svd(out, compute_uv=False)
    assert out.shape == G.shape and s[0] <= 1049 / 1024 + 1e-9 and s[2:].max() < 1e-9
    assert np.array_equal(polar_express(np.zeros((3, 3))), np.zeros((3, 3)))
