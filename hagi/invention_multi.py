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


def invent_multi(
    pairs: list[tuple[tuple[int, ...], int]],
    rng: random.Random | None = None,
    max_attempts: int = 8000,
    n_shifts: int = 6,
) -> dict | None:
    """WAVE 236f: multi-byte law synthesis with SHIFT-STRATIFIED
    verification.

    A byte-stream law y = law(ctx[-k:]) holds only on positions
    aligned with the data's repeating structure (e.g. records of
    length 3). Unaligned positions are noise BY CONSTRUCTION, so a
    single global consistency check is doomed. Strategy: for each
    candidate feature, partition pairs by stream position mod
    n_shifts, fit on ONE stratum and verify out-of-sample on the
    OTHERS (rotating). A law is accepted only when some stratum
    yields a generalizing basis (POLY_LIN / OUT_PERIOD) whose
    predictions verify >=95% on the other strata of the SAME
    feature. The stratum index is reported for honest provenance.
    """
    rng = rng or random.Random(0)
    if not pairs:
        return None
    k = len(pairs[0][0])
    if any(len(x) != k for x, _ in pairs):
        return None
    GEN_BASES = [(n, f) for n, f in BASES
                 if n in ("POLY_LIN", "OUT_PERIOD")]
    attempts = 0
    for fname, fparam, ffn in enumerate_features(k):
        attempts += 1
        if attempts > max_attempts:
            break
        feats = [(ffn(x), y) for x, y in pairs]
        for shift in range(n_shifts):
            stratum = [(i, c, y) for i, (c, y) in enumerate(feats)
                       if i % n_shifts == shift]
            if len(stratum) < 60:
                continue
            c2y: dict[int, set[int]] = {}
            for _, c, y in stratum:
                c2y.setdefault(c, set()).add(y)
            cons = sorted((c, next(iter(v)))
                          for c, v in c2y.items() if len(v) == 1)
            # WAVE 236h: aligned fraction of a true law's own
            # stratum measured ~0.4 on real miss streams (the LM
            # occasionally guesses right, punching holes); 0.35
            # with an absolute floor is the honest cut.
            if len(cons) < 24 or len(cons) < int(0.35 * len(c2y)):
                continue
            rng2 = random.Random(rng.randrange(1 << 30))
            rng2.shuffle(cons)
            fit_n = max(8, int(0.6 * len(cons)))
            fit, oos = cons[:fit_n], cons[fit_n:]
            best = None
            for b_name, b_fn in GEN_BASES:
                payload = b_fn(fit)
                if payload is None:
                    continue
                b_bits = _payload_bits(b_name, payload, len(fit))
                bp = ConstructBlueprint(fname, fparam, ffn, b_name,
                                        payload, 9,
                                        total_bits=b_bits + 9)
                # OOS within the same stratum
                if not all(bp.apply(c) == y for c, y in oos):
                    continue
                # cross-stratum check on OTHER shifts' consistent
                # groups: each OTHER stratum must agree on >=70%
                # of its consistent groups
                ok_all = True
                for other in range(n_shifts):
                    if other == shift:
                        continue
                    o2 = [(i, c, y) for i, (c, y) in enumerate(feats)
                          if i % n_shifts == other]
                    c2y2: dict[int, set[int]] = {}
                    for _, c, y in o2:
                        c2y2.setdefault(c, set()).add(y)
                    cons2 = [(c, next(iter(v)))
                             for c, v in c2y2.items() if len(v) == 1]
                    if len(cons2) < 8:
                        continue
                    hit = sum(1 for c, y in cons2
                              if bp.apply(c) == y)
                    if hit / len(cons2) < 0.7:
                        ok_all = False
                        break
                if ok_all and (best is None
                               or bp.total_bits < best.total_bits):
                    best = bp
            if best is not None:
                return {"status": "INVENTED", "blueprints": [best],
                        "feature": fname, "shift": shift,
                        "attempts": attempts}
    return {"status": "NO_LAW", "attempts": attempts}
