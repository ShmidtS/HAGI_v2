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
            return None, None
        order = sorted(range(len(z)), key=lambda i: mid[i], reverse=True)
        a, b = order[0], order[1]
        # certified separation: mid a > mid b + band width (C495
        # certified_pred via disjoint intervals)
        t = sandwich_terms(z, nm)
        lo, hi = t
        slo, shi = sum(lo, type(lo[0])(0)), sum(hi, type(hi[0])(0))
        width_a = hi[a] / shi - lo[a] / slo
        width_b = hi[b] / shi - lo[b] / slo
        half = max(width_a, width_b) / 2
        if mid[a] - mid[b] > 2 * half:
            return a, nm
        nm *= 2
    return None, None
