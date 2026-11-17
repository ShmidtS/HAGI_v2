#!/usr/bin/env python3
"""ANF (Zhegalkin polynomial) — python port of primes AnfTransform.lean.

Möbius transform over GF(2): ANF coefficients are parities of the
function on subcubes. anf_inversion: applying the transform twice
recovers the truth table (theory-exact).
"""
from __future__ import annotations

from typing import Callable
import itertools


def truth_table(f: Callable[[tuple], bool], n: int) -> list[int]:
    return [int(f(x)) for x in itertools.product([0, 1], repeat=n)]


def anf_transform(tt: list[int]) -> list[int]:
    """Möbius (Zhegalkin) transform: ANF coefficients from truth table."""
    a = list(tt)
    n_vars = int(len(tt) ** 0.5) if int(len(tt) ** 0.5) ** 2 == len(tt) else None
    m = len(a)
    step = 1
    while step < m:
        for i in range(m):
            if i & step:
                a[i] ^= a[i & ~step]
        step <<= 1
    return a


def anf_eval(coeffs: list[int], x: tuple) -> int:
    """Evaluate ANF polynomial at input x (theory-exact):
    f(x) = XOR over coefficient indices S where S's bits are subset
    of x's bits of coeffs[S]."""
    n_vars = len(x)
    out = 0
    for s in range(len(coeffs)):
        if coeffs[s] and all((s >> i) & 1 <= x[i] for i in range(n_vars)):
            out ^= 1
    return out


def anf_compose(coeffs_f, coeffs_g, compose: Callable[[int, int], int]):
    """Compose two ANF polynomials pointwise (theory: anf_add for XOR,
    anf_inversion for NOT; general composition is pointwise here)."""
    return [compose(cf, cg) for cf, cg in zip(coeffs_f, coeffs_g)]


def verify_all() -> tuple[bool, int]:
    ok, n = True, 0
    for n_vars in (1, 2, 3):
        m = 2 ** n_vars
        for bits in range(m):
            tt = [(bits >> i) & 1 for i in range(m)]
            coeffs = anf_transform(tt)
            # anf_inversion: transform twice recovers truth table
            back = anf_transform(coeffs)
            ok &= (back == tt)
            # evaluation matches the original function on all inputs
            for x in itertools.product([0, 1], repeat=n_vars):
                idx = sum(bit << i for i, bit in enumerate(x))
                ok &= (anf_eval(coeffs, x) == tt[idx])
                n += 1
            n += 1
    return ok, n


def dispatch(candidates: list[tuple[str, list[int], Callable]]) -> tuple[str, int] | None:
    """AnfDispatch port: apply candidate rules; return None on
    conflicting answers (abstain), else (name, answer)."""
    answers = []
    for name, coeffs, x in candidates:
        answers.append((name, anf_eval(coeffs, x)))
    vals = {v for _, v in answers}
    if len(vals) > 1:
        return None  # ambiguous — refuse
    return answers[0]
