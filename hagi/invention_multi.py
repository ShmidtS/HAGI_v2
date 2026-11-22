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


def invent_multi(
    pairs: list[tuple[tuple[int, ...], int]],
    rng: random.Random | None = None,
    max_attempts: int = 8000,
    n_shifts: int = 6,
    min_support: float = 0.30,
) -> dict | None:
    """WAVE 236k: PARTIAL-LAW synthesis for byte streams.

    Real stream laws hold only on record-ALIGNED positions (e.g.
    every 3rd byte); unaligned positions are noise by construction.
    A single global consistency check therefore rejects true laws.
    New semantics:

    1. A candidate feature maps contexts to a feature value c.
    2. For each position-stratum (pair index mod n_shifts) fit a
       generalizing basis (POLY_LIN / OUT_PERIOD) on the MAJORITY
       vote y per feature value (mode), then verify on the
       held-out strata: prediction must match >=90% of majority
       votes AND the majority coverage must be >= min_support.
    3. Accepted laws are returned with status PARTIAL_LAW and a
       `support` ratio — they predict with known reliability on
       aligned positions, HOLD everywhere else. They are NEVER
       PROVEN_EXACT (that status requires full deterministic
       coverage, invention.py verify_blueprint keeps that bar).
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
            if len(stratum) < 80:
                continue
            # majority vote per feature value in this stratum
            c2y: dict[int, Counter] = {}
            for _, c, y in stratum:
                c2y.setdefault(c, Counter())[y] += 1
            fit_pairs = []
            support_hits = 0
            # keep one representative CONTEXT per feature value so
            # bp.apply(context) runs on real tuples, not ints
            c2ctx: dict[int, tuple] = {}
            for _, c, y in stratum:
                c2ctx.setdefault(c, None)
            ctx_of: dict[int, tuple] = {}
            for idx_pair, (x, y) in enumerate(pairs):
                if idx_pair % n_shifts != shift:
                    continue
                c = ffn(x)
                if c not in ctx_of:
                    ctx_of[c] = x
            feat_fit = []     # (c, y) для basis (инт-пары)
            ctx_fit = []      # (ctx, y) для apply-верификации
            support_hits = 0
            for c, cnt in c2y.items():
                top_y, top_n = cnt.most_common(1)[0]
                if top_n >= 2 and top_n / sum(cnt.values()) >= 0.6:
                    feat_fit.append((c, top_y))
                    ctx_fit.append((ctx_of[c], top_y))
                    support_hits += top_n
            support = support_hits / max(len(stratum), 1)
            if len(feat_fit) < 24 or support < min_support:
                continue
            rng2 = random.Random(rng.randrange(1 << 30))
            rng2.shuffle(feat_fit)
            fit_n = max(8, int(0.6 * len(feat_fit)))
            fit, oos = feat_fit[:fit_n], feat_fit[fit_n:]
            oos_ctx = [(ctx_of[c], y) for c, y in oos]
            best = None
            for b_name, b_fn in GEN_BASES:
                payload = b_fn(fit)
                if payload is None:
                    continue
                b_bits = _payload_bits(b_name, payload, len(fit))
                bp = ConstructBlueprint(fname, fparam, ffn, b_name,
                                        payload, 9,
                                        total_bits=b_bits + 9)
                # OOS: apply на контекстах (bp сам вычислит фичу)
                hits = sum(1 for x, y in oos_ctx
                           if bp.apply(x) == y)
                if oos and hits / len(oos) < 0.9:
                    continue
                # cross-stratum: majority votes of other strata
                # WAVE 236k: cross-stratum only against STRUCTURAL
                # strata (support >= min_support); noise phases carry
                # no evidence and must not veto the law.
                cross_ok = True
                for other in range(n_shifts):
                    if other == shift:
                        continue
                    if _stratum_support(feats, other,
                                        n_shifts) < min_support:
                        continue
                    o2 = [(i, c, y) for i, (c, y) in enumerate(feats)
                          if i % n_shifts == other]
                    c2y2: dict[int, Counter] = {}
                    for _, c, y in o2:
                        c2y2.setdefault(c, Counter())[y] += 1
                    ctx2: dict[int, tuple] = {}
                    for idx_pair, (x, y) in enumerate(pairs):
                        if idx_pair % n_shifts != other:
                            continue
                        c2 = ffn(x)
                        if c2 not in ctx2:
                            ctx2[c2] = x
                    maj = [(ctx2[c], cnt.most_common(1)[0][0])
                           for c, cnt in c2y2.items()
                           if cnt.most_common(1)[0][1]
                           / sum(cnt.values()) >= 0.6
                           and sum(cnt.values()) >= 3 and c in ctx2]
                    if len(maj) < 8:
                        continue
                    hit = sum(1 for x, y in maj if bp.apply(x) == y)
                    if hit / len(maj) < 0.7:
                        cross_ok = False
                        break
                if cross_ok and (best is None
                                 or bp.total_bits < best.total_bits):
                    best = bp
            if best is not None:
                return {"status": "PARTIAL_LAW",
                        "blueprints": [best],
                        "feature": fname, "shift": shift,
                        "support": round(support, 3),
                        "attempts": attempts}
    return {"status": "NO_LAW", "attempts": attempts}

def _stratum_support(feats, shift, n_shifts) -> float:
    """Majority-vote support of one stratum (share of its mass on
    >=60%-dominant y values)."""
    stratum = [(c, y) for i, (c, y) in enumerate(feats)
               if i % n_shifts == shift]
    if not stratum:
        return 0.0
    c2y: dict[int, Counter] = {}
    for c, y in stratum:
        c2y.setdefault(c, Counter())[y] += 1
    hits = 0
    for cnt in c2y.values():
        top_y, top_n = cnt.most_common(1)[0]
        if top_n >= 2 and top_n / sum(cnt.values()) >= 0.6:
            hits += top_n
    return hits / len(stratum)
