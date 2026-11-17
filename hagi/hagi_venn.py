#!/usr/bin/env python3
"""Venn head — python port of primes Hagi/Model/Logic/VennHead.lean.

Logic atoms as ReLU thresholds (theory-exact):
  vAtom(sA, sB, A, B) = relu(sA*A + sB*B - 1)
  vAtoms(A, B) = [vAtom(1,1), vAtom(1,-1), vAtom(-1,1), vAtom(-1,-1)]
  vennHead(mask, A, B) = sum_k mask_k * vAtoms(A,B)_k

mask: 4-bit truth table in atomIdx order (++),(+-),(-+),(--).
Verified: all 16 masks x 4 inputs exactly match vennBool
(vennHead_exact, 64/64).
"""
from __future__ import annotations

from typing import Callable


def relu(x: float) -> float:
    return max(0.0, x)


def v_atom(s_a: float, s_b: float, a: float, b: float) -> float:
    return relu(s_a * a + s_b * b - 1)


def v_atoms(a: float, b: float) -> tuple[float, ...]:
    return (v_atom(1, 1, a, b), v_atom(1, -1, a, b),
            v_atom(-1, 1, a, b), v_atom(-1, -1, a, b))


def atom_idx(a: bool, b: bool) -> int:
    return 0 if (a and b) else (1 if (a and not b)
                                else (2 if (not a and b) else 3))


def venn_head(mask, a: float, b: float) -> float:
    vs = v_atoms(a, b)
    return sum(mi * v for mi, v in zip(mask, vs))


def mask_from_fn(f: Callable[[bool, bool], bool]) -> tuple[int, ...]:
    return tuple(int(f(a, b)) for a, b in
                 [(True, True), (True, False), (False, True), (False, False)])


def named_masks() -> dict[str, tuple[int, ...]]:
    ops = {
        "FALSE": lambda a, b: False, "AND": lambda a, b: a and b,
        "A_GT_B": lambda a, b: a and not b, "A": lambda a, b: a,
        "B_GT_A": lambda a, b: not a and b, "B": lambda a, b: b,
        "XOR": lambda a, b: a != b, "OR": lambda a, b: a or b,
        "NOR": lambda a, b: not (a or b), "XNOR": lambda a, b: a == b,
        "NOT_B": lambda a, b: not b, "B_LE_A": lambda a, b: b <= a,
        "NOT_A": lambda a, b: not a, "A_LE_B": lambda a, b: a <= b,
        "NAND": lambda a, b: not (a and b), "TRUE": lambda a, b: True,
    }
    return {k: mask_from_fn(f) for k, f in ops.items()}


def verify_all() -> tuple[bool, int]:
    ok, n = True, 0
    for m in named_masks().values():
        for av, bv, a, b in [(1, 1, True, True), (1, -1, True, False),
                             (-1, 1, False, True), (-1, -1, False, False)]:
            idx = atom_idx(a, b)
            got = 1 if venn_head(m, av, bv) > 0.5 else 0
            n += 1
            ok &= (got == m[idx])
    return ok, n
