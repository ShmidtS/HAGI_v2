"""Certified softmax sandwich (primes C489/C495/C502).

Every formula here is the direct Python image of a proven Lean
theorem (Hagi.Theory.RationalSoftmax / SandwichMonotone /
BandTightening):

  lo  = (1 + sv/nm) ** nm          — lower exp bound (bernoulliLower_mono)
  hi  = (1 - sv/nm) ** (-nm)       — upper exp bound (expUpper_base_mono)
  sv  = z - max(z)                 — max-shifted logits, in [-R, 0]

Monotone in nm (C502): growing the denominator tightens both
walls, so an eval-time loop doubling nm eventually separates any
truly separable top-1 and certifies the argmax (cert_pred_adaptive).

Range guard: sv must satisfy -sv < nm (else the upper base 1-sv/nm
leaves (0,1]); the function honestly returns None = HOLD.

Use Fraction inputs for exact rational arithmetic (bench parity);
floats work for advisory checks.
"""
from __future__ import annotations

from fractions import Fraction

DEFAULT_NM = 4
EVAL_NM_MAX = 1024


def sandwich_terms(z, nm: int = DEFAULT_NM):
    """lo/hi sandwich bounds on max-shifted logits.

    Returns (lo, hi) lists, or None = HOLD when the range guard
    -min(s) >= nm fails (the certificate does not exist at nm).
    """
    from fractions import Fraction as F
    M = max(z)
    s = [v - M for v in z]
    if -min(s) >= nm:
        return None
    lo = [(F(1) + sv / nm) ** nm for sv in s]
    hi = [(F(1) - sv / nm) ** (-nm) for sv in s]
    return lo, hi


def sandwich_mid(z, nm: int = DEFAULT_NM):
    """Midpoint surrogate (sandwichMid): accurate to half band width
    (sandwich_mid_acc). None = HOLD."""
    t = sandwich_terms(z, nm)
    if t is None:
        return None
    lo, hi = t
    slo, shi = sum(lo, type(lo[0])(0)), sum(hi, type(hi[0])(0))
    return [(hi_v / shi + lo_v / slo) / 2
            for lo_v, hi_v in zip(lo, hi)]


def certified_argmax(z, nm: int = DEFAULT_NM, nm_max: int = EVAL_NM_MAX):
    """C502 cert_pred_adaptive: the top-1 by rational midpoint,
    certified, with nm doubling until the band separates the top-2
    (bandWidth_mono guarantees refinement only shrinks the band).

    Returns (class, nm_used) or (None, None) = HOLD.
    """
    nm = nm
    while nm <= nm_max:
        mid = sandwich_mid(z, nm)
        if mid is None:
            # WAVE 203 FIX (P0, research report): the range guard
            # (-min(s) >= nm) failing at the CURRENT nm does NOT mean
            # the certificate is impossible — C502 BandTightening
            # guarantees larger nm tightens the band AND relaxes the
            # guard (guard is -min(s) < nm). Double and retry instead
            # of HOLDing separable logits like [10, 0, 0].
            nm *= 2
            continue
        # WAVE 204 FIX (P1.5, audit bundle v2): correct probability
        # bounds. lo_i/sum(lo) was NOT a valid lower bound — the
        # denominator must use the OPPOSITE extremes of the rest:
        #   L_i = l_i / (l_i + sum_{j!=i} u_j)
        #   U_i = u_i / (u_i + sum_{j!=i} l_j)
        # A tie (e.g. [3,3,0]) can no longer be falsely certified.
        t = sandwich_terms(z, nm)
        if t is None:
            nm *= 2
            continue
        lo, hi = t
        n = len(z)
        order = sorted(range(n), key=lambda i: mid[i], reverse=True)
        L = [lo[i] / (lo[i] + sum(hi[j] for j in range(n) if j != i))
             for i in range(n)]
        U = [hi[i] / (hi[i] + sum(lo[j] for j in range(n) if j != i))
             for i in range(n)]
        a = order[0]
        if all(L[a] > U[j] for j in range(n) if j != a):
            return a, nm
        nm *= 2
    return None, None


def band_drift_check(z, nm: int = DEFAULT_NM) -> dict:
    """WAVE 213 (C568 band describes measurement): drift detector
    between the certified rational band and the actual softmax.
    bandDescribes holds for genuine certified bands — this measures
    whether the band still DESCRIBES the measurement (if the true
    softmax point escapes the [L_i, U_i] interval, the band is stale:
    wrong logits domain or float precision drift)."""
    from fractions import Fraction as F
    import math
    t = sandwich_terms(z, nm)
    if t is None:
        if nm * 2 <= EVAL_NM_MAX:
            return band_drift_check(z, nm * 2)
        return {"ok": False, "reason": "range guard failed at nm_max"}
    lo, hi = t
    n = len(z)
    L = [lo[i] / (lo[i] + sum(hi[j] for j in range(n) if j != i))
         for i in range(n)]
    U = [hi[i] / (hi[i] + sum(lo[j] for j in range(n) if j != i))
         for i in range(n)]
    mx = max(z)
    probs = [math.exp(float(v - mx)) for v in z]
    s = sum(probs)
    probs = [p / s for p in probs]
    escapes = [i for i in range(n)
               if not (float(L[i]) <= probs[i] <= float(U[i]))]
    # WAVE 213b: range-guard / escapes at current nm — double like
    # certified_argmax (BandTightening: larger nm tightens walls).
    if escapes and nm * 2 <= EVAL_NM_MAX:
        return band_drift_check(z, nm * 2)
    return {"ok": not escapes, "escapes": escapes,
            "nm": nm, "widths": [float(U[i] - L[i]) for i in range(n)]}
