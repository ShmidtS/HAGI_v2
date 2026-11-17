"""Logic basis (CDGC stage A) — Venn head + ANF/Zhegalkin.

Venn: 16 two-argument boolean functions as 4-atom ReLU masks
(theory-exact vs primes VennHead.lean, 64/64).
ANF: Moebius GF(2) transform, theory-exact evaluation, dispatch
with abstain-on-conflict (AnfDispatch).
"""


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


def venn_verify_all() -> tuple[bool, int]:
    ok, n = True, 0
    for m in named_masks().values():
        for av, bv, a, b in [(1, 1, True, True), (1, -1, True, False),
                             (-1, 1, False, True), (-1, -1, False, False)]:
            idx = atom_idx(a, b)
            got = 1 if venn_head(m, av, bv) > 0.5 else 0
            n += 1
            ok &= (got == m[idx])
    return ok, n


# ======================================================================
# ANF / Zhegalkin (GF(2))
# ======================================================================

#!/usr/bin/env python3
"""ANF (Zhegalkin polynomial) — python port of primes AnfTransform.lean.

Möbius transform over GF(2): ANF coefficients are parities of the
function on subcubes. anf_inversion: applying the transform twice
recovers the truth table (theory-exact).
"""

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


def anf_verify_all() -> tuple[bool, int]:
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
