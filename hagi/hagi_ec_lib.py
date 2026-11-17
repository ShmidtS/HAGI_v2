#!/usr/bin/env python3
"""EC-processor library — the WINNING genomes exposed as callable code.

Built by Architecture Genesis / evolutionary search (hagi_ec_evolution).
Each profile gets its optimized scalar-multiply as a flat function
(Nadya-compatible: takes/returns ints) plus a tuple-level API.

Certified: every function passes group-law invariants (see
hagi_ec_evolution.certified_correct) on the task curve
y^2 = x^3 + 497x + 1768 mod 9739.
"""
from __future__ import annotations

P = 9739
A = 497
B = 1768
G = (1804, 5368)
N = 9735          # group order of G


def ec_add(Pt, Qt):
    """Affine point addition (winner: throughput/balanced genome)."""
    if Pt is None:
        return Qt
    if Qt is None:
        return Pt
    x1, y1 = Pt
    x2, y2 = Qt
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if Pt == Qt:
        lam = (3 * x1 * x1 + A) * pow(2 * y1, -1, P) % P
    else:
        lam = (y2 - y1) * pow((x2 - x1) % P, -1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return (x3, (lam * (x1 - x3) - y1) % P)


def ec_mul(k: int, Pt=G):
    """Scalar multiply — throughput winner (double_add, precompute 1)."""
    R = None
    A_ = Pt
    while k:
        if k & 1:
            R = ec_add(R, A_)
        A_ = ec_add(A_, A_)
        k >>= 1
    return R


def ec_mul_secure(k: int, Pt=G):
    """Montgomery ladder — security winner (constant-time structure)."""
    R0, R1 = None, Pt
    for bit in bin(k)[2:]:
        if bit == "1":
            R0 = ec_add(R0, R1)
            R1 = ec_add(R1, R1)
        else:
            R1 = ec_add(R0, R1)
            R0 = ec_add(R0, R0)
    return R0


# --- Nadya-compatible flat functions (int in, int out) ---

def EC_X(k: int) -> int:
    """x-coordinate of k*G (0 if identity)."""
    r = ec_mul(k)
    return r[0] if r else 0


def EC_Y(k: int) -> int:
    """y-coordinate of k*G (0 if identity)."""
    r = ec_mul(k)
    return r[1] if r else 0


def EC_ON_CURVE(x: int, y: int) -> int:
    """1 iff (x,y) lies on the task curve."""
    return 1 if (y * y - x * x * x - A * x - B) % P == 0 else 0


def EC_ORDER_OF(k: int) -> int:
    """Smallest m with (k*m mod N)*G = identity — order of kG."""
    Pt = ec_mul(k)
    if Pt is None:
        return 1
    m = 1
    R = Pt
    while R is not None:
        R = ec_add(R, Pt)
        m += 1
        if m > N:
            raise RuntimeError("group order exceeded — corruption")
    return m
