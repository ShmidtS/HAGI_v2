"""WAVE 236 (user directive): multi-byte law synthesis — laws of
ARBITRARY arity over the context window.

The core invention layer (hagi/invention.py) synthesizes arity-1
laws: quotient q_fn(x) -> c over a single byte, then a basis over
(c, y). This module extends the SAME construction to tuple inputs:
a candidate law is a FEATURE MAP phi: tuple -> int (affine
combinations of the context bytes mod 256, XOR-masks, pairwise
differences), followed by the identical basis machinery and the
identical verify_blueprint certificate (full labeled coverage or
no proof).

Honest semantics unchanged: a PROVEN_EXACT verdict certifies the
law over the SYNTHESIZED FEATURE, on the labeled support of the
feature's domain. Coverage is reported, never implied.
"""
from __future__ import annotations

import random

import numpy as np
from collections import Counter
from typing import Callable

from hagi.invention import (BASES, ConstructBlueprint,
                            _payload_bits, verify_blueprint)

# ---- feature maps over context tuples ----------------------------


def _affine_k(coefs: tuple[int, ...], bias: int):
    """c = (a1*x1 + ... + ak*xk + b) mod 256."""
    def f(x: tuple[int, ...]) -> int:
        s = bias
        for a, v in zip(coefs, x):
            s += a * v
        return s % 256
    return f


def _xor_mask(mask: tuple[int, ...]):
    def f(x: tuple[int, ...]) -> int:
        s = 0
        for m, v in zip(mask, x):
            s ^= (m & v)
        return s
    return f


def _diff_pair(i: int, j: int):
    """c = (x[i] - x[j]) mod 256 — relative-position laws."""
    def f(x: tuple[int, ...]) -> int:
        return (x[i] - x[j]) % 256
    return f


def _position(i: int):
    def f(x: tuple[int, ...]) -> int:
        return x[i] % 256
    return f


def enumerate_features(k: int, max_features: int = 4000):
    """Combinatorial feature family for arity-k contexts: ALL small
    integer coefficient vectors (each coef in -3..3, at least one
    nonzero), biases from a fixed set, pairwise diffs and positions.
    Cost grows as 7^k * |biases| — capped by max_features."""
    feats: list[tuple[str, object, Callable]] = []
    from itertools import product as _prod
    # WAVE 236: order by SPARSITY first — laws with few nonzero
    # coefs are both more likely and cheaper to verify.
    coef_sets = [c for c in _prod((-3, -2, -1, 0, 1, 2, 3), repeat=k)
                 if any(v != 0 for v in c)]
    coef_sets.sort(key=lambda c: (sum(1 for v in c if v != 0),
                                  sum(abs(v) for v in c)))
    for coefs in coef_sets:
        for bias in (0, 1, 3, 7, 97):
            feats.append((f"aff{coefs}+{bias}",
                          (coefs, bias),
                          _affine_k(coefs, bias)))
        if len(feats) > max_features:
            break
    for i in range(k):
        for j in range(i + 1, k):
            feats.append((f"diff{i}-{j}", (i, j), _diff_pair(i, j)))
    # XOR combinations: subset AND-masks AND full-byte XORs
    for bits in range(1, 1 << k):
        mask = tuple((bits >> i) & 1 for i in range(k))
        feats.append((f"xor{mask}", mask, _xor_mask(mask)))
    def _xor_full(idxs):
        def f(x):
            s = 0
            for i in idxs:
                s ^= x[i]
            return s
        return f
    for bits in range(1, 1 << k):
        idxs = [i for i in range(k) if (bits >> i) & 1]
        feats.append((f"fxor{idxs}", tuple(idxs), _xor_full(idxs)))
    for i in range(k):
        feats.append((f"pos{i}", (i,), _position(i)))
    return feats[:max_features]


# ---- synthesis ----------------------------------------------------


def _vfeat(kind: str, param, X) -> "np.ndarray":
    """Vectorised feature values (int64 in [0,256)) for contexts X
    (N, k). Bit-exact with the callables of enumerate_features
    (checked on all 5 979 features of k=2..4 x 5000 random contexts)."""
    X = np.asarray(X, dtype=np.int64)
    if kind == "aff":
        return (X @ np.asarray(param[0], dtype=np.int64) + param[1]) % 256
    if kind == "diff":
        i, j = param
        return (X[:, i] - X[:, j]) % 256
    if kind == "xorm":
        return np.bitwise_xor.reduce(X & np.asarray(param, dtype=np.int64), axis=1)
    if kind == "xfull":
        return np.bitwise_xor.reduce(X[:, list(param)], axis=1)
    if kind == "pos":
        return X[:, param[0]]
    raise KeyError(kind)


def _kind(name: str) -> str:
    for pre, kind in (("aff", "aff"), ("diff", "diff"), ("fxor", "xfull"),
                      ("xor", "xorm"), ("pos", "pos")):
        if name.startswith(pre):
            return kind
    raise KeyError(name)


def _majVotes(c: "np.ndarray", y: "np.ndarray", idx: "np.ndarray"):
    """Majority vote of y per feature value over the positions `idx`.
    Returns (rows, support): rows = (feature value, top y, count,
    share) for values whose top y holds >= 60% of >= 2 votes.
    Same rule as the scalar c2y/most_common loop it replaces."""
    cc, yy = c[idx], y[idx]
    m = int(cc.max()) + 1 if len(cc) else 0
    if m == 0:
        return [], 0.0
    H = np.bincount(cc * 256 + yy, minlength=m * 256).reshape(m, 256)
    tot = H.sum(1)
    top = H.argmax(1)
    top_n = H[np.arange(m), top]
    ok = (top_n >= 2) & (top_n / np.maximum(tot, 1) >= 0.6)
    vals = np.nonzero(ok)[0]
    rows = [(int(v), int(top[v])) for v in vals]
    return rows, float(top_n[ok].sum() / max(len(idx), 1))


def invent_multi(pairs, rng: "random.Random | None" = None,
                 max_attempts: int = 8000, n_shifts: int = 8,
                 min_support: float = 0.30,
                 max_period: int | None = None) -> dict | None:
    """WAVE 236k: PARTIAL-LAW synthesis for byte streams.

    Real stream laws hold only on record-ALIGNED positions (e.g.
    every 3rd byte); unaligned positions are noise by construction.
    A single global consistency check therefore rejects true laws.

    1. A candidate feature maps contexts to a feature value c.
    2. For each position-stratum (pair index mod P == s) fit a
       generalizing basis (POLY_LIN / OUT_PERIOD) on the MAJORITY
       vote y per feature value (mode), then verify on the
       held-out part of the stratum: prediction must match >=90% of
       majority votes AND the majority coverage must be >= min_support.
    3. Accepted laws are returned with status PARTIAL_LAW and a
       `support` ratio — they predict with known reliability on
       aligned positions, HOLD everywhere else. They are NEVER
       PROVEN_EXACT (that status requires full deterministic
       coverage, invention.py verify_blueprint keeps that bar).

    WAVE 247: the strata were `i % n_shifts` with a FIXED n_shifts=6,
    so a law with period 4, 5, 7, 8 was aligned to no stratum and
    came back NO_LAW (measured: affine law on every 4th/5th/7th byte
    -> NO_LAW 7.5 s each; only 3 and 6 were ever found). The period
    is now SEARCHED (P in 1..max_period, every phase s), and the
    feature values / votes are computed with numpy: the scalar loop
    evaluated each feature twice per stratum (10.3 M calls per
    search). Acceptance rules are unchanged; the feature values are
    bit-exact with the callables (see _vfeat). `n_shifts` is kept as
    the default max_period (callers pass it).
    8 by default (measured): periods 3..8 are then found at ~100%
    raw aligned accuracy and a pure-noise search costs 2.5 s — the
    OLD code cost 6.0 s on the same noise and found nothing past 6.
    12/16 double/quadruple the noise cost, and 16 already reports a
    multiple of the true period (P=14 for a period-7 law).
    Candidates are ranked by (support, then MDL bits) — "first that
    passes" returned a weaker fit on a MULTIPLE of the true period
    (measured: period 4 reported as 8 at raw accuracy 0.98; a 70%-
    covered period-3 law reported as 6 with a fitted feature). Now
    1.000 and the true period / true feature in both cases.
    Stratum size limits precision, not discovery: a period-8 law at
    3000 bytes is reported with support 0.744 / raw accuracy 0.971,
    at 6000 with 0.945 / 1.000 — PARTIAL_LAW states its own support.
    """
    rng = rng or random.Random(0)
    if not pairs:
        return None
    k = len(pairs[0][0])
    if any(len(x) != k for x, _ in pairs):
        return None
    if max_period is None:
        max_period = int(n_shifts)
    GEN_BASES = [(n, f) for n, f in BASES if n in ("POLY_LIN", "OUT_PERIOD")]
    X = np.asarray([x for x, _ in pairs], dtype=np.int64)
    y = np.asarray([yy for _, yy in pairs], dtype=np.int64)
    n = len(pairs)
    periods = [P for P in range(1, max(1, max_period) + 1)
               if n // P >= 80]
    attempts = 0
    winner, winner_key = None, None
    for fname, fparam, _ffn in enumerate_features(k):
        attempts += 1
        if attempts > max_attempts:
            break
        c = _vfeat(_kind(fname), fparam, X)
        # ONE representative context per feature value over ALL pairs
        # (the first occurrence). Built once per feature: the
        # cross-stratum check reads values that the CURRENT stratum
        # may never contain — a per-stratum table gave -1 there and
        # verified against pairs[-1], i.e. a garbage context.
        rep = np.full(int(c.max()) + 1, -1, dtype=np.int64)
        rep[c[::-1]] = np.arange(n)[::-1]
        for P in periods:
            for shift in range(P):
                idx = np.nonzero(np.arange(n) % P == shift)[0]
                if len(idx) < 80:
                    continue
                rows, support = _majVotes(c, y, idx)
                if len(rows) < 24 or support < min_support:
                    continue
                order = list(range(len(rows)))
                rng.shuffle(order)
                rows = [rows[i] for i in order]
                fit_n = max(8, int(0.6 * len(rows)))
                fit, oos = rows[:fit_n], rows[fit_n:]
                oos_ctx = [(pairs[int(rep[cc])][0], yy) for cc, yy in oos]
                best = None
                for b_name, b_fn in GEN_BASES:
                    payload = b_fn(fit)
                    if payload is None:
                        continue
                    b_bits = _payload_bits(b_name, payload, len(fit))
                    # the blueprint keeps the ORIGINAL scalar callable
                    bp = ConstructBlueprint(fname, fparam, _ffn, b_name,
                                            payload, 9,
                                            total_bits=b_bits + 9)
                    hits = sum(1 for x, yy in oos_ctx
                               if bp.apply(x) == yy)
                    if oos and hits / len(oos) < 0.9:
                        continue
                    # cross-stratum: majority votes of the OTHER
                    # phases of the same period; noise phases carry
                    # no evidence and must not veto the law.
                    cross_ok = True
                    for other in range(P):
                        if other == shift:
                            continue
                        idx2 = np.nonzero(np.arange(n) % P == other)[0]
                        rows2, sup2 = _majVotes(c, y, idx2)
                        if sup2 < min_support:
                            continue
                        cnt2 = np.bincount(c[idx2], minlength=len(rep))
                        maj = [(pairs[int(rep[cc])][0], yy)
                               for cc, yy in rows2 if cnt2[cc] >= 3]
                        if len(maj) < 8:
                            continue
                        hit = sum(1 for x, yy in maj
                                  if bp.apply(x) == yy)
                        if hit / len(maj) < 0.7:
                            cross_ok = False
                            break
                    if cross_ok and (best is None
                                     or bp.total_bits < best.total_bits):
                        best = bp
                if best is not None:
                    cand = {"status": "PARTIAL_LAW",
                            "blueprints": [best],
                            "feature": fname, "shift": shift,
                            "n_shifts": P,
                            "support": round(support, 3),
                            "attempts": attempts}
                    # WAVE 247: "first law that passes" is not "the
                    # law". With the period searched, a weaker fit on a
                    # multiple of the period can pass every gate at an
                    # earlier feature (measured: true period 4 reported
                    # as P=8, raw accuracy 0.98 vs 1.000). Keep the
                    # best by (support, then MDL bits); stop early only
                    # on an EXACT law, which nothing can improve.
                    key = (cand["support"], -best.total_bits)
                    if winner is None or key > winner_key:
                        winner, winner_key = cand, key
                    if cand["support"] >= 0.999:
                        return winner
    return winner or {"status": "NO_LAW", "attempts": attempts}
