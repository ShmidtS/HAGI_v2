"""HAGI core — single-module kernel (merged _legacy + _v9).

Merge preserves the historical override semantics exactly: the whole
V5-V8 legacy body executes first, then the V9 canonical body — later
definitions win for the ~28 shared names, and V9 top-level code sees
the legacy versions of helpers at module-execution time, same as the
old star-import chain.

Consolidation refactor 2026-11-17. Public names unchanged.
"""

#!/usr/bin/env python3
"""HAGI unified — single-file build (v5 core + v7.4 + v8.4 + v9.0 layers)
=========================================================================
Canonical merged build. Sources: .omc/hagi_v8_4.py (Parts I-III) +
.omc/hagi_v9.py (Part IV). Later definitions win for the ~27 shared
helpers — v9 carries the pivot_rows guard in exact poly induction and
other refinements; class names are suffixed so both lineages coexist.

CLI: --self-test (v8.4) | --self-test-v9 | --self-test-v7 | --self-test-v5 |
     --grow (v8.4 loop) | --grow-v9 | --benchmark-v9 [--max-rounds N] [--out-dir DIR]
"""
import argparse
import copy
import itertools
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

# ==============================================================================
# PART I — v5 core (Growing Math Processor)
# ==============================================================================

"""HAGI Growing Math Processor — bounded, auditable neuro-symbolic prototype.

This is a research prototype, not a claim to learn every mathematical function.
It grows a library of verified operation experts from teacher-generated examples.
Exact symbolic programs are committed only after validation over the declared
finite domain. A small neural fiber is trained in parallel and is available as
an explicitly approximate fallback. Out-of-domain queries abstain by default.

Dependencies: Python >=3.10, NumPy.
"""


VERSION = "5.0.0"
V5_VERSION = VERSION


# --- Legacy V5-V8 kernel body lives in hagi.legacy_kernel ---
from hagi.legacy_kernel import *  # noqa: F401,F403
from hagi.legacy_kernel import (  # noqa: F401 — private/CAPS used by V9
    _exact_multivariate_poly_fit,
    _exact_poly_fit,
    _fit_poly_or_const,
    _monomial_exponents,
    BINARY_OPS,
    COMMUTATIVE_OPS,
    CONSTS,
    DYADIC_LEVELS,
    OPS,
    TOKEN_BITS,
    VAR_NAMES,
    _PRIME_MOD,
    _bit_nand8,
    _bit_nor8,
    _bit_not8,
    _bit_xnor8,
    _pow_int,
    _sign,
)

# ==============================================================================
# 2. Task Specification & Benchmark Curriculum
# ==============================================================================


@dataclass
class TaskSpec:
    name: str
    arity: int
    domains: tuple[tuple[int, ...], ...]
    teacher: Callable[..., int]
    valid_rule: str = "all"
    family: str = "integer"

    def inputs(self) -> list[tuple[int, ...]]:
        out = []
        for row in itertools.product(*self.domains):
            if self.valid_rule == "nonzero_y" and row[1] == 0:
                continue
            if self.valid_rule == "ordered_yz" and row[1] > row[2]:
                continue
            out.append(tuple(int(x) for x in row))
        return out

    def valid(self, xs: Sequence[int]) -> bool:
        if len(xs) != self.arity:
            return False
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):
            return False
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:
            return False
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):
            return False
        return True


BOOL_NAMES = {
    0: "BOOL_FALSE",
    1: "BOOL_NOR",
    2: "BOOL_NOT_X_AND_Y",
    3: "BOOL_NOT_X",
    4: "BOOL_X_AND_NOT_Y",
    5: "BOOL_NOT_Y",
    6: "BOOL_XOR",
    7: "BOOL_NAND",
    8: "BOOL_AND",
    9: "BOOL_XNOR",
    10: "BOOL_Y",
    11: "BOOL_X_IMPLIES_Y",
    12: "BOOL_X",
    13: "BOOL_Y_IMPLIES_X",
    14: "BOOL_OR",
    15: "BOOL_TRUE",
}


def make_task_suite() -> list[TaskSpec]:
    suite: list[TaskSpec] = []
    bool_domain = ((0, 1), (0, 1))
    for mask, name in BOOL_NAMES.items():
        suite.append(
            TaskSpec(
                name,
                2,
                bool_domain,
                lambda x, y, m=mask: (m >> ((int(x) << 1) | int(y))) & 1,
                family="BoolVenn",
            )
        )
    U = tuple(range(-8, 9))
    unary = [
        ("NEG", lambda x: -x),
        ("ABS", abs),
        ("SIGN", _sign),
        ("SQUARE", lambda x: x * x),
        ("CUBE", lambda x: x * x * x),
        ("BIT_NOT8", _bit_not8),
        ("CLAMP_TO_3", lambda x: min(max(x, -3), 3)),
    ]
    for name, fn in unary:
        suite.append(TaskSpec(name, 1, (U,), fn, family="unary"))
    binary = [
        ("ADD", lambda x, y: x + y, "all"),
        ("SUB", lambda x, y: x - y, "all"),
        ("MUL", lambda x, y: x * y, "all"),
        ("FLOORDIV", lambda x, y: x // y, "nonzero_y"),
        ("MOD", lambda x, y: x % y, "nonzero_y"),
        ("GCD", math.gcd, "all"),
        ("LCM", math.lcm, "all"),
        ("MIN", min, "all"),
        ("MAX", max, "all"),
        ("ABSDIFF", lambda x, y: abs(x - y), "all"),
        ("EQ", lambda x, y: int(x == y), "all"),
        ("NE", lambda x, y: int(x != y), "all"),
        ("LT", lambda x, y: int(x < y), "all"),
        ("LE", lambda x, y: int(x <= y), "all"),
        ("GT", lambda x, y: int(x > y), "all"),
        ("GE", lambda x, y: int(x >= y), "all"),
        ("BIT_AND", lambda x, y: x & y, "all"),
        ("BIT_OR", lambda x, y: x | y, "all"),
        ("BIT_XOR", lambda x, y: x ^ y, "all"),
        ("BIT_NAND8", _bit_nand8, "all"),
        ("BIT_NOR8", _bit_nor8, "all"),
        ("BIT_XNOR8", _bit_xnor8, "all"),
    ]
    for name, fn, valid in binary:
        suite.append(
            TaskSpec(name, 2, (U, U), fn, valid_rule=valid, family="integer-binary")
        )
    P = tuple(range(-3, 4))
    E = tuple(range(0, 5))
    suite.append(TaskSpec("POW_INT", 2, (P, E), _pow_int, family="integer-binary"))
    Z = tuple(range(-8, 9))
    F = tuple(range(-4, 5))
    suite.extend(
        [
            TaskSpec(
                "FMA",
                3,
                (Z, Z, Z),
                lambda x, y, z: x * y + z,
                family="composed-ternary",
            ),
            TaskSpec(
                "CLAMP",
                3,
                (Z, F, F),
                lambda x, lo, hi: min(max(x, lo), hi),
                valid_rule="ordered_yz",
                family="composed-ternary",
            ),
            TaskSpec(
                "MEDIAN3",
                3,
                (F, F, F),
                lambda x, y, z: sorted((x, y, z))[1],
                family="composed-ternary",
            ),
        ]
    )
    # --- physics: exact integer laws (polynomial-invariant stage covers) ---
    T2 = tuple(range(0, 3))
    suite.extend(
        [
            TaskSpec(
                "PHYS_FREE_FALL",
                1,
                (tuple(range(0, 5)),),
                lambda t: 20 - 5 * t * t,
                family="physics",
            ),
            TaskSpec(
                "PHYS_UNIFORM_MOTION",
                1,
                (tuple(range(-3, 4)),),
                lambda t: 3 + 2 * t,
                family="physics",
            ),
            TaskSpec(
                "PHYS_SPRING_ENERGY",
                1,
                (tuple(range(-3, 4)),),
                lambda x: 3 * x * x,
                family="physics",
            ),
            TaskSpec(
                "PHYS_MOMENTUM",
                2,
                (tuple(range(0, 5)), tuple(range(-4, 5))),
                lambda m, v: m * v,
                family="physics",
            ),
        ]
    )
    # --- language: character codes 0..25 as the integer encoding ---
    A = tuple(range(0, 26))
    K = tuple(range(0, 26))
    suite.extend(
        [
            TaskSpec(
                "LANG_CAESAR3", 1, (A,), lambda c: (c + 3) % 26, family="language"
            ),
            TaskSpec("LANG_ATBASH", 1, (A,), lambda c: 25 - c, family="language"),
            TaskSpec(
                "LANG_IS_VOWEL",
                1,
                (A,),
                lambda c: int(c in (0, 4, 8, 14, 20)),
                family="language",
            ),
            TaskSpec(
                "LANG_SHIFT", 2, (A, K), lambda c, k: (c + k) % 26, family="language"
            ),
        ]
    )
    # --- full UTF-8 pool: byte space 0..255 (ANY unicode text is bytes) ---
    B = tuple(range(0, 256))
    BK = tuple(range(0, 256))
    suite.extend(
        [
            TaskSpec("LANG8_ATBASH", 1, (B,), lambda c: 255 - c, family="language"),
            TaskSpec(
                "LANG8_CAESAR1", 1, (B,), lambda c: (c + 1) % 256, family="language"
            ),
            TaskSpec(
                "LANG8_SHIFT", 2, (B, BK), lambda c, k: (c + k) % 256, family="language"
            ),
            TaskSpec(
                "LANG8_XOR_CIPHER", 2, (B, BK), lambda c, k: c ^ k, family="language"
            ),
        ]
    )
    # --- cyrillic codepoints (U+0400..U+04FF): not only Latin ---
    RU_VOWELS = set(map(ord, "аеёиоуыэюяАЕЁИОУЫЭЮЯ"))
    RU = tuple(range(0x400, 0x500))
    suite.extend(
        [
            TaskSpec(
                "LANG_RU_VOWEL",
                1,
                (RU,),
                lambda c: int(c in RU_VOWELS),
                family="language",
            ),
        ]
    )
    return suite


# ==============================================================================
# 3. Call-DAG AST, ProgramV9 & Vectorized Evaluator
# ==============================================================================

# V10-2: Reified operator registry — invented coordinates promoted to
# first-class unary/binary macro-ops usable inside ExprV9 programs.
# Filled by hagi.reifier.reify_construct(); persists via to_state().
REIFIED_OPS: dict = {}


@dataclass(frozen=True)
class ExprV9:
    op: str
    args: tuple["ExprV9", ...] = ()
    value: Optional[Any] = None

    @staticmethod
    def var(i: int) -> "ExprV9":
        return ExprV9("var", (), int(i))

    @staticmethod
    def const(v: int) -> "ExprV9":
        return ExprV9("const", (), int(v))

    def eval(
        self, xs: Sequence[int], env: Optional[dict[str, "ExpertV9"]] = None
    ) -> int:
        if self.op == "var":
            return int(xs[int(self.value)])
        if self.op == "const":
            return int(self.value)
        if self.op == "call":
            if env is None or self.value not in env:
                raise ValueError(f"missing called expert: {self.value}")
            vals = tuple(a.eval(xs, env) for a in self.args)
            callee = env[self.value]
            if not callee.supports(vals):
                raise ValueError(
                    f"call to {self.value} outside certified support: {vals}"
                )
            return callee.program.eval(vals, env)
        if self.op == "anf":
            n_vars, coeffs = self.value
            mask_x = 0
            for i in range(n_vars):
                b = int(xs[i])
                if b not in (0, 1):
                    raise ValueError("anf expects boolean inputs")
                if b:
                    mask_x |= 1 << (n_vars - 1 - i)
            acc = 0
            for s, c in coeffs:
                if (s & mask_x) == s:
                    acc ^= c
            return int(acc)
        if self.op == "if_cond":
            cond_val = self.args[0].eval(xs, env)
            return (
                self.args[1].eval(xs, env) if cond_val else self.args[2].eval(xs, env)
            )
        vals = [a.eval(xs, env) for a in self.args]
        if self.op == "poly":
            acc = 0
            for ci in reversed(list(self.value or ())):
                acc = acc * vals[0] + int(ci)
            return int(acc)
        if self.op == "poly_nd":
            total = 0
            for coef, powers in self.value or ():
                term = int(coef)
                for x_val, p in zip(xs, powers):
                    term *= int(x_val) ** int(p)
                total += term
            return int(total)
        if self.op == "absdiff":
            return abs(vals[0] - vals[1])
        if self.op == "fma":
            return vals[0] * vals[1] + vals[2]
        if self.op == "clamp":
            return min(max(vals[0], vals[1]), vals[2])
        if self.op == "median3":
            return sorted(vals)[1]
        if self.op == "clamp3":
            return min(max(vals[0], -3), 3)
        if self.op == "mul3":
            return vals[0] * vals[1] * vals[2]
        if self.op == "construct":
            from hagi.representations import construct_eval

            return construct_eval(self.value, xs, env)
        fn = REIFIED_OPS.get(self.op)
        if fn is not None:
            return int(fn(*vals))
        fn = OPS.get(self.op)
        if fn is None:
            raise ValueError(f"unsupported op: {self.op}")
        return int(fn(*vals))

    @property
    def cost(self) -> int:
        c = self.__dict__.get("_cost_memo")
        if c is None:
            if self.op in ("var", "const"):
                c = 1
            elif self.op == "anf":
                _, coeffs = self.value
                c = 1 + len(coeffs)
            elif self.op in ("poly", "poly_nd"):
                c = 1 + len(self.value or ())
            elif self.op == "construct":
                from hagi.representations import construct_bits as _cb

                c = 1 + _cb(self.value)
            else:
                c = 1 + sum(a.cost for a in self.args)
            object.__setattr__(self, "_cost_memo", c)
        return c

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:
        if self.op == "var":
            return TOKEN_BITS + gamma_bits(int(self.value or 0) + 1)
        if self.op == "const":
            return TOKEN_BITS + gamma_bits(zigzag(int(self.value)) + 1)
        if self.op == "call":
            idx = expert_index.get(str(self.value), 0) if expert_index else 0
            return (
                TOKEN_BITS
                + gamma_bits(idx + 1)
                + sum(a.bits(expert_index) for a in self.args)
            )
        if self.op == "anf":
            n_vars, coeffs = self.value
            payload = gamma_bits(n_vars + 1) + gamma_bits(len(coeffs) + 1)
            payload += sum(gamma_bits(s + 1) for s, _ in coeffs)
            return TOKEN_BITS + payload
        if self.op == "poly":
            coeffs = tuple(int(v) for v in (self.value or ()))
            payload = gamma_bits(len(coeffs) + 1) + sum(
                gamma_bits(zigzag(v) + 1) for v in coeffs
            )
            return TOKEN_BITS + payload + sum(a.bits(expert_index) for a in self.args)
        if self.op == "poly_nd":
            terms = tuple(self.value or ())
            payload = gamma_bits(len(terms) + 1)
            for c, powers in terms:
                payload += gamma_bits(zigzag(int(c)) + 1) + gamma_bits(len(powers) + 1)
                payload += sum(gamma_bits(int(p) + 1) for p in powers)
            return TOKEN_BITS + payload
        return TOKEN_BITS + sum(a.bits(expert_index) for a in self.args)

    def callees(self) -> set[str]:
        out: set[str] = set()
        if self.op == "call" and self.value is not None:
            out.add(str(self.value))
        for a in self.args:
            out |= a.callees()
        return out

    def substitute(self, mapping: dict[int, "ExprV9"]) -> "ExprV9":
        if self.op == "var":
            idx = int(self.value or 0)
            return mapping.get(idx, self)
        if not self.args:
            return self
        return ExprV9(
            self.op, tuple(a.substitute(mapping) for a in self.args), self.value
        )

    def subtrees(self) -> list["ExprV9"]:
        out = [self]
        for a in self.args:
            out.extend(a.subtrees())
        return out

    def replace_subtree(self, target: "ExprV9", replacement: "ExprV9") -> "ExprV9":
        if self == target:
            return replacement
        if not self.args:
            return self
        return ExprV9(
            self.op,
            tuple(a.replace_subtree(target, replacement) for a in self.args),
            self.value,
        )

    def pretty(self) -> str:
        if self.op == "var":
            idx = int(self.value or 0)
            return VAR_NAMES[idx] if idx < len(VAR_NAMES) else f"x{idx}"
        if self.op == "const":
            return str(self.value)
        if self.op == "call":
            return f"{self.value}({', '.join(a.pretty() for a in self.args)})"
        if self.op == "anf":
            n_vars, coeffs = self.value
            terms = []
            for s, _ in coeffs:
                if s == 0:
                    terms.append("1")
                else:
                    vs = [
                        VAR_NAMES[i] if i < len(VAR_NAMES) else f"x{i}"
                        for i in range(n_vars)
                        if (s >> (n_vars - 1 - i)) & 1
                    ]
                    terms.append("&".join(vs))
            return "anf(" + (" ^ ".join(terms) if terms else "0") + ")"
        if self.op == "if_cond":
            return f"({self.args[1].pretty()} if {self.args[0].pretty()} else {self.args[2].pretty()})"
        if self.op == "poly":
            coeffs = tuple(self.value or ())
            terms = []
            for i in range(len(coeffs) - 1, -1, -1):
                c = int(coeffs[i])
                if c == 0:
                    continue
                mag = abs(c)
                body = "" if i == 0 else ("x" if i == 1 else f"x**{i}")
                term = str(mag) if i == 0 else (body if mag == 1 else f"{mag}*{body}")
                if not terms:
                    terms.append(("-" if c < 0 else "") + term)
                else:
                    terms.append((" - " if c < 0 else " + ") + term)
            return "poly(" + ("".join(terms) if terms else "0") + ")"
        if self.op == "poly_nd":
            terms = []
            for coef, powers in sorted(
                (self.value or ()), key=lambda it: (sum(it[1]), it[1]), reverse=True
            ):
                c = int(coef)
                if c == 0:
                    continue
                factors = []
                for idx, p in enumerate(powers):
                    p = int(p)
                    if p == 0:
                        continue
                    nm = VAR_NAMES[idx] if idx < len(VAR_NAMES) else f"x{idx}"

                body = "*".join(factors) if factors else str(abs(c))
                if factors and abs(c) != 1:
                    body = f"{abs(c)}*{body}"
                if not terms:
                    terms.append(("-" if c < 0 else "") + body)
                else:
                    terms.append((" - " if c < 0 else " + ") + body)
            return "poly_nd(" + ("".join(terms) if terms else "0") + ")"
        if self.op == "clamp3":
            return f"min(max({self.args[0].pretty()}, -3), 3)"
        if self.op == "absdiff":
            return f"abs({self.args[0].pretty()} - {self.args[1].pretty()})"
        if self.op == "fma":
            return f"({self.args[0].pretty()} * {self.args[1].pretty()} + {self.args[2].pretty()})"
        if self.op == "clamp":
            return f"min(max({self.args[0].pretty()}, {self.args[1].pretty()}), {self.args[2].pretty()})"
        if self.op == "median3":
            return f"median({', '.join(a.pretty() for a in self.args)})"
        if self.op == "mul3":
            return " * ".join(a.pretty() for a in self.args)
        if len(self.args) == 1:
            return f"{self.op}({self.args[0].pretty()})"
        if len(self.args) == 2:
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"

    def to_dict(self) -> dict[str, Any]:
        val = self.value
        if self.op == "anf" and val is not None:
            val = [int(val[0]), [[int(s), int(c)] for s, c in val[1]]]
        elif self.op == "construct" and isinstance(val, dict):
            from hagi.representations import construct_to_dict_safe

            val = construct_to_dict_safe(val)
        return {"op": self.op, "value": val, "args": [a.to_dict() for a in self.args]}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExprV9":
        op = str(d["op"])
        val = d.get("value")
        if op == "poly" and val is not None:
            val = tuple(int(v) for v in val)
        elif op == "poly_nd" and val is not None:
            val = tuple((int(t[0]), tuple(int(p) for p in t[1])) for t in val)
        elif op == "anf" and val is not None:
            val = (int(val[0]), tuple((int(s), int(c)) for s, c in val[1]))
        elif op == "construct" and isinstance(val, dict):
            from hagi.representations import construct_from_dict_safe

            val = construct_from_dict_safe(val)
        return ExprV9(op, tuple(ExprV9.from_dict(a) for a in d.get("args", [])), val)


@dataclass(frozen=True)
class ProgramV9:
    arity: int
    expr: Optional[ExprV9] = None
    venn_mask: Optional[int] = None

    @property
    def cost(self) -> int:
        return (
            4
            if self.venn_mask is not None
            else (self.expr.cost if self.expr else 10**9)
        )

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:
        if self.venn_mask is not None:
            return TOKEN_BITS + 4
        if self.expr is None:
            return 10**9
        return self.expr.bits(expert_index)

    def eval(
        self, xs: Sequence[int], env: Optional[dict[str, "ExpertV9"]] = None
    ) -> int:
        if len(xs) != self.arity:
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")
        if self.venn_mask is not None:
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):
                raise ValueError("BoolVenn expects two binary inputs")
            return (int(self.venn_mask) >> ((int(xs[0]) << 1) | int(xs[1]))) & 1
        if self.expr is None:
            raise RuntimeError("empty program")
        return self.expr.eval(xs, env)

    def pretty(self) -> str:
        if self.venn_mask is not None:
            return f"BoolVenn(mask={self.venn_mask:04b})"
        assert self.expr is not None
        return self.expr.pretty()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arity": self.arity,
            "venn_mask": self.venn_mask,
            "expr": self.expr.to_dict() if self.expr else None,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ProgramV9":
        return ProgramV9(
            int(d["arity"]),
            ExprV9.from_dict(d["expr"]) if d.get("expr") else None,
            d.get("venn_mask"),
        )


_VEC_MAG_LIMIT = 1 << 28


def eval_expr_vec(
    expr: ExprV9, X: np.ndarray, env: Optional[dict[str, "ExpertV9"]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative vectorized int64 evaluator (ok=False lanes -> scalar authority)."""
    N = X.shape[0]
    op = expr.op
    if op == "var":
        return X[:, int(expr.value)].copy(), np.ones(N, dtype=bool)
    if op == "const":
        v = int(expr.value)
        if abs(v) > _VEC_MAG_LIMIT:
            return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=bool)
        return np.full(N, v, dtype=np.int64), np.ones(N, dtype=bool)
    if op == "call":
        if env is None or expr.value not in env:
            return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=bool)
        callee = env[expr.value]
        arg_vals = []
        ok = np.ones(N, dtype=bool)
        for a in expr.args:
            v, aok = eval_expr_vec(a, X, env)
            arg_vals.append(v)
            ok &= aok
        if not np.any(ok):
            return np.zeros(N, dtype=np.int64), ok
        subX = np.column_stack(arg_vals)
        for i, dom in enumerate(callee.domains):
            lo, hi = dom[0], dom[-1]
            if len(dom) == hi - lo + 1 and dom == tuple(range(lo, hi + 1)):
                ok &= (subX[:, i] >= lo) & (subX[:, i] <= hi)
            else:
                ok &= np.isin(subX[:, i], dom)
        if callee.valid_rule == "nonzero_y":
            ok &= subX[:, 1] != 0
        elif callee.valid_rule == "ordered_yz":
            ok &= subX[:, 1] <= subX[:, 2]
        if not np.any(ok):
            return np.zeros(N, dtype=np.int64), ok
        if callee.program.venn_mask is not None:
            idx = (np.clip(subX[:, 0], 0, 1) << 1) | np.clip(subX[:, 1], 0, 1)
            return ((int(callee.program.venn_mask) >> idx) & 1).astype(np.int64), ok
        res_all = np.zeros(N, dtype=np.int64)
        vals, ok2 = eval_expr_vec(callee.program.expr, subX[ok], env)
        res_all[ok] = vals
        ok_idx = np.flatnonzero(ok)
        ok[ok_idx[~ok2]] = False
        return res_all, ok
    if op == "if_cond":
        cv, cok = eval_expr_vec(expr.args[0], X, env)
        tv, tok = eval_expr_vec(expr.args[1], X, env)
        fv, fok = eval_expr_vec(expr.args[2], X, env)
        return np.where(cv != 0, tv, fv), cok & tok & fok
    if op == "poly":
        v, ok = eval_expr_vec(expr.args[0], X, env)
        acc = np.zeros(N, dtype=np.int64)
        for ci in reversed(list(expr.value or ())):
            ok &= np.abs(acc) <= _VEC_MAG_LIMIT
            acc = np.where(ok, acc * v + int(ci), 0)
        ok &= np.abs(acc) <= _VEC_MAG_LIMIT
        return acc, ok
    if op == "poly_nd":
        acc = np.zeros(N, dtype=np.int64)
        ok = np.ones(N, dtype=bool)
        for coef, powers in expr.value or ():
            term = np.full(N, int(coef), dtype=np.int64)
            for idx, p in enumerate(powers):
                if p:
                    ok &= (np.abs(X[:, idx]) <= 1000) & (int(p) <= 6)
                    safe_x = np.where(ok, X[:, idx], 0)
                    term = np.where(ok, term * (safe_x ** int(p)), 0)
            acc = np.where(ok, acc + term, 0)
        ok &= np.abs(acc) <= _VEC_MAG_LIMIT
        return acc, ok
    if op == "anf":
        n_vars, coeffs = expr.value
        ok = np.ones(N, dtype=bool)
        mask_x = np.zeros(N, dtype=np.int64)
        for i in range(n_vars):
            ok &= (X[:, i] == 0) | (X[:, i] == 1)
            mask_x |= np.where(X[:, i] != 0, 1 << (n_vars - 1 - i), 0)
        acc = np.zeros(N, dtype=np.int64)
        for s, c in coeffs:
            acc ^= np.where((mask_x & s) == s, c, 0)
        return acc, ok

    vs = []
    ok = np.ones(N, dtype=bool)
    for a in expr.args:
        v, aok = eval_expr_vec(a, X, env)
        vs.append(v)
        ok &= aok
    if op == "neg":
        res = -vs[0]
    elif op == "abs":
        res = np.abs(vs[0])
    elif op == "sign":
        res = np.sign(vs[0]).astype(np.int64)
    elif op == "bit_not8":
        res = (~vs[0]) & 0xFF
    elif op == "clamp3":
        res = np.clip(vs[0], -3, 3)
    elif op == "square":
        ok &= np.abs(vs[0]) <= 32768
        s0 = np.where(ok, vs[0], 0)
        res = s0 * s0
    elif op == "cube":
        ok &= np.abs(vs[0]) <= 1024
        s0 = np.where(ok, vs[0], 0)
        res = s0 * s0 * s0
    elif op == "absdiff":
        res = np.abs(vs[0] - vs[1])
    elif op == "fma":
        ok &= (np.abs(vs[0]) <= 32768) & (np.abs(vs[1]) <= 32768)
        s0 = np.where(ok, vs[0], 0)
        s1 = np.where(ok, vs[1], 0)
        res = s0 * s1 + vs[2]
    elif op == "clamp":
        res = np.minimum(np.maximum(vs[0], vs[1]), vs[2])
    elif op == "median3":
        res = np.sort(np.stack(vs, axis=0), axis=0)[1]
    elif op == "mul3":
        ok &= (
            (np.abs(vs[0]) <= 1024) & (np.abs(vs[1]) <= 1024) & (np.abs(vs[2]) <= 1024)
        )
        s0 = np.where(ok, vs[0], 0)
        s1 = np.where(ok, vs[1], 0)
        s2 = np.where(ok, vs[2], 0)
        res = s0 * s1 * s2
    elif op == "add":
        res = vs[0] + vs[1]
    elif op == "sub":
        res = vs[0] - vs[1]
    elif op == "mul":
        ok &= (np.abs(vs[0]) <= 32768) & (np.abs(vs[1]) <= 32768)
        s0 = np.where(ok, vs[0], 0)
        s1 = np.where(ok, vs[1], 0)
        res = s0 * s1
    elif op == "min":
        res = np.minimum(vs[0], vs[1])
    elif op == "max":
        res = np.maximum(vs[0], vs[1])
    elif op == "eq":
        res = (vs[0] == vs[1]).astype(np.int64)
    elif op == "ne":
        res = (vs[0] != vs[1]).astype(np.int64)
    elif op == "lt":
        res = (vs[0] < vs[1]).astype(np.int64)
    elif op == "le":
        res = (vs[0] <= vs[1]).astype(np.int64)
    elif op == "gt":
        res = (vs[0] > vs[1]).astype(np.int64)
    elif op == "ge":
        res = (vs[0] >= vs[1]).astype(np.int64)
    elif op == "bit_and":
        res = vs[0] & vs[1]
    elif op == "bit_or":
        res = vs[0] | vs[1]
    elif op == "bit_xor":
        res = vs[0] ^ vs[1]
    elif op == "bit_nand8":
        res = (~(vs[0] & vs[1])) & 0xFF
    elif op == "bit_nor8":
        res = (~(vs[0] | vs[1])) & 0xFF
    elif op == "bit_xnor8":
        res = (~(vs[0] ^ vs[1])) & 0xFF
    elif op == "floordiv":
        ok &= vs[1] != 0
        safe = np.where(vs[1] != 0, vs[1], 1)
        res = np.floor_divide(vs[0], safe)
    elif op == "mod":
        ok &= vs[1] != 0
        safe = np.where(vs[1] != 0, vs[1], 1)
        res = np.mod(vs[0], safe)
    elif op == "gcd":
        res = np.gcd(vs[0], vs[1])
    elif op == "lcm":
        res = np.lcm(vs[0], vs[1])
    elif op == "pow_int":
        ok &= (vs[1] >= 0) & (vs[1] <= 8) & (np.abs(vs[0]) <= 32)
        sa = np.where(ok, vs[0], 0)
        sb = np.where(ok, vs[1], 0)
        res = sa**sb
    else:
        raise ValueError(f"unsupported vec op: {op}")
    ok &= np.abs(res) <= _VEC_MAG_LIMIT
    return res, ok


def candidate_programs_v9(arity: int) -> list[ProgramV9]:
    v = [ExprV9.var(i) for i in range(arity)]
    cands: list[ExprV9] = []
    if arity == 1:
        x = v[0]
        cands += [
            ExprV9("neg", (x,)),
            ExprV9("abs", (x,)),
            ExprV9("sign", (x,)),
            ExprV9("bit_not8", (x,)),
            ExprV9("square", (x,)),
            ExprV9("cube", (x,)),
            ExprV9("clamp3", (x,)),
        ]
    elif arity == 2:
        x, y = v
        for op in BINARY_OPS:
            cands.append(ExprV9(op, (x, y)))
        cands.extend([ExprV9("sub", (y, x)), ExprV9("absdiff", (x, y))])
    elif arity == 3:
        x, y, z = v
        cands += [
            ExprV9("fma", (x, y, z)),
            ExprV9("clamp", (x, y, z)),
            ExprV9("median3", (x, y, z)),
            ExprV9("mul3", (x, y, z)),
        ]
    by_text = {}
    for e in cands:
        by_text.setdefault(e.pretty(), e)
    return [
        ProgramV9(arity, e)
        for e in sorted(by_text.values(), key=lambda e: (e.cost, e.pretty()))
    ]


# ==============================================================================
# 4. VennHead + 13-Byte VennPacking + ANF Zhegalkin / compose3 + Hadamard + 5-Trit Codec
# ==============================================================================


def venn_atoms(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Exact 4-ReLU-neuron Venn atom basis on {-1, +1} inputs (C242)."""
    return np.column_stack(
        [
            np.maximum(0.0, (-s1 - s2) * 0.5),
            np.maximum(0.0, (-s1 + s2) * 0.5),
            np.maximum(0.0, (s1 - s2) * 0.5),
            np.maximum(0.0, (s1 + s2) * 0.5),
        ]
    )


def venn_head_eval(mask: int, x_bin: Sequence[int]) -> int:
    s1 = np.array([2.0 * int(x_bin[0]) - 1.0])
    s2 = np.array([2.0 * int(x_bin[1]) - 1.0])
    atoms = venn_atoms(s1, s2)[0]
    w = np.array([(mask >> idx) & 1 for idx in range(4)], dtype=np.float64)
    return int(round(float(atoms @ w)))


def pack_venn_16_flat() -> bytes:
    """C251: 16 masks x 4 trits -> 13 bytes flat (tight lower bound)."""
    trits = [(mask >> idx) & 1 for mask in range(16) for idx in range(4)]
    while len(trits) % 5 != 0:
        trits.append(0)
    out = bytearray()
    for off in range(0, len(trits), 5):
        b = 0
        for t in trits[off : off + 5]:
            b = b * 3 + t
        out.append(b)
    assert len(out) == 13
    return bytes(out)


def unpack_venn_16_flat(buf: bytes) -> list[int]:
    assert len(buf) == 13
    trits = []
    for b in buf:
        chunk = [0] * 5
        v = b
        for i in range(4, -1, -1):
            chunk[i] = v % 3
            v //= 3
        trits.extend(chunk)
    masks = []
    for m in range(16):
        val = 0
        for idx in range(4):
            val |= (trits[m * 4 + idx] & 1) << idx
        masks.append(val)
    return masks


def anf_transform(truth_table: Sequence[int]) -> list[int]:
    """Exact C301 Zhegalkin/Mobius butterfly. O(n 2^n), involution."""
    n_tot = len(truth_table)
    if n_tot == 0 or (n_tot & (n_tot - 1)) != 0:
        raise ValueError("truth_table length must be a power of 2")
    coeffs = [int(v) & 1 for v in truth_table]
    step = 1
    while step < n_tot:
        for i in range(0, n_tot, step * 2):
            for j in range(step):
                coeffs[i + step + j] ^= coeffs[i + j]
        step <<= 1
    return coeffs


def anf_eval(coeffs: Sequence[int], n_vars: int, xs: Sequence[int]) -> int:
    mask_x = 0
    for i in range(n_vars):
        if int(xs[i]) & 1:
            mask_x |= 1 << (n_vars - 1 - i)
    acc = 0
    for s, c in enumerate(coeffs):
        if c and (s & mask_x) == s:
            acc ^= 1
    return acc


def anf_compose3(
    f_bool: Callable[[int, int], int], g_tt: Sequence[int], h_tt: Sequence[int]
) -> list[int]:
    """C302 compose3(f, g, h)(x) = f(g(x), h(x)) ANF transform."""
    assert len(g_tt) == len(h_tt)
    tt = [int(f_bool(int(g_tt[i]) & 1, int(h_tt[i]) & 1)) & 1 for i in range(len(g_tt))]
    return anf_transform(tt)


def anf_support_vars(coeffs: Sequence[int], n_vars: int) -> tuple[int, ...]:
    used = 0
    for s, c in enumerate(coeffs):
        if c:
            used |= s
    return tuple(i for i in range(n_vars) if (used >> (n_vars - 1 - i)) & 1)


def anf_to_expr_v9(coeffs: Sequence[int], n_vars: int) -> ExprV9:
    sparse = tuple((s, 1) for s, c in enumerate(coeffs) if c)
    if len(sparse) == 0:
        return ExprV9.const(0)
    if len(sparse) == 1 and sparse[0][0] == 0:
        return ExprV9.const(1)
    if len(sparse) <= 4:
        mons = []
        for s, _ in sparse:
            if s == 0:
                mons.append(ExprV9.const(1))
            else:
                vs = [
                    ExprV9.var(i) for i in range(n_vars) if (s >> (n_vars - 1 - i)) & 1
                ]
                t = vs[0]
                for v in vs[1:]:
                    t = ExprV9("bit_and", (t, v))
                mons.append(t)
        res = mons[0]
        for m in mons[1:]:
            res = ExprV9("bit_xor", (res, m))
        return res
    return ExprV9("anf", (), (n_vars, sparse))


_HADAMARD_CACHE: dict[int, np.ndarray] = {}


def sylvester_hadamard(k: int) -> np.ndarray:
    """Zero-byte deterministic orthonormal Sylvester-Hadamard matrix."""
    if k in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[k]
    H = np.array([[1.0]], dtype=np.float64)
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    H /= math.sqrt(2**k)
    _HADAMARD_CACHE[k] = H
    return H


def block_hadamard_rotate(v: np.ndarray, blk_size: int = 8) -> tuple[np.ndarray, int]:
    H = sylvester_hadamard(int(round(math.log2(blk_size))))
    v1 = np.asarray(v, dtype=np.float64).ravel()
    n = v1.size
    nb = (n + blk_size - 1) // blk_size
    vp = np.pad(v1, (0, nb * blk_size - n)).reshape(nb, blk_size)
    return (vp @ H).ravel(), n


def block_hadamard_unrotate(
    vr: np.ndarray, orig_n: int, blk_size: int = 8
) -> np.ndarray:
    H = sylvester_hadamard(int(round(math.log2(blk_size))))
    nb = vr.size // blk_size
    return (vr.reshape(nb, blk_size) @ H.T).ravel()[:orig_n]


_TRIT_POW = np.array([81, 27, 9, 3, 1], dtype=np.int64)


def quantize_5block_exact(blk: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Fin 255 dyadic + 5-trit quantization with verified ternary_split_le bound."""
    m = float(np.max(np.abs(blk)))
    if m < 1e-15:
        return np.zeros_like(blk), (0, 0, 121)
    exp2 = max(-126, min(126, math.ceil(math.log2(m / 1.5 + 1e-18))))
    sc = 2.0**exp2
    bn = blk / sc
    s = DYADIC_LEVELS[:, None]
    ratio = bn[None, :] / s
    q_int = np.where(ratio > 0.5, 1.0, np.where(ratio < -0.5, -1.0, 0.0))
    errs = np.sum((bn[None, :] - q_int * s) ** 2, axis=1)
    best = int(np.argmin(errs))
    s_best = float(DYADIC_LEVELS[best])
    trits = q_int[best].astype(np.int64)
    in_mask = np.abs(bn) <= 1.5 * s_best
    sat_tail = float(np.sum(np.where(~in_mask, (np.abs(bn) - s_best) ** 2, 0.0)))
    bound = int(np.sum(in_mask)) * (s_best / 2.0) ** 2 + sat_tail
    assert float(errs[best]) <= bound + 1e-9
    bval = int(np.dot(trits + 1, _TRIT_POW))
    return (trits * (s_best * sc)).astype(np.float64), (exp2, best, bval)


def pack_vector_dyadic5(vec: np.ndarray, stages: int = 2) -> tuple[bytes, np.ndarray]:
    """Lossless bit-packed 5-trit dyadic codec with verified decode roundtrip."""
    v = np.asarray(vec, dtype=np.float64).ravel()
    n = v.size
    nblk = (n + 4) // 5
    vp = np.pad(v, (0, nblk * 5 - n))
    buf = bytearray()
    buf.extend(int(n).to_bytes(2, "little"))
    buf.append(int(stages))
    rem = vp.copy()
    for _ in range(stages):
        for b in range(nblk):
            blk = rem[b * 5 : (b + 1) * 5]
            q_blk, (exp2, best, bval) = quantize_5block_exact(blk)
            if exp2 == 0 and best == 0 and bval == 121:
                buf.append(255)
            else:
                buf.append(exp2 + 128)
                buf.append(best)
                buf.append(bval)
            rem[b * 5 : (b + 1) * 5] -= q_blk
    raw = bytes(buf)
    return raw, unpack_vector_dyadic5(raw)


def unpack_vector_dyadic5(buf: bytes) -> np.ndarray:
    n = int.from_bytes(buf[:2], "little")
    stages = buf[2]
    nblk = (n + 4) // 5
    out = np.zeros(nblk * 5, dtype=np.float64)
    pos = 3
    for _ in range(stages):
        for b in range(nblk):
            tag = buf[pos]
            pos += 1
            if tag == 255:
                continue
            exp2 = tag - 128
            best = buf[pos]
            pos += 1
            bval = buf[pos]
            pos += 1
            sc = 2.0**exp2
            lvl = float(DYADIC_LEVELS[best])
            trits = [0] * 5
            v = bval
            for i in range(4, -1, -1):
                trits[i] = (v % 3) - 1
                v //= 3
            out[b * 5 : (b + 1) * 5] += np.array(trits, dtype=np.float64) * (lvl * sc)
    return out[:n]


def pack_bitalloc_records(
    k: int, idx_sorted: np.ndarray, records: list[tuple[int, tuple[int, int, int]]]
) -> bytes:
    n_blocks = (k + 4) // 5
    buf = bytearray()
    buf.append(int(k))
    max_idx = int(np.max(idx_sorted)) if k > 0 else 0
    wide = 1 if max_idx >= 255 else 0
    buf.append(wide)
    for ix in idx_sorted:
        if wide:
            buf.extend(int(ix).to_bytes(2, "little"))
        else:
            buf.append(int(ix))
    buf.append(len(records))
    for ri, (blk_id, (exp2, best, bval)) in enumerate(records):
        if ri >= n_blocks:
            buf.append(int(blk_id))
        if exp2 == 0 and best == 0 and bval == 121:
            buf.append(255)
        else:
            buf.append(exp2 + 128)
            buf.append(int(best))
            buf.append(int(bval))
    return bytes(buf)


def unpack_bitalloc_records(buf: bytes, D: int) -> np.ndarray:
    pos = 0
    k = buf[pos]
    pos += 1
    wide = buf[pos]
    pos += 1
    idx_sorted = []
    for _ in range(k):
        if wide:
            idx_sorted.append(int.from_bytes(buf[pos : pos + 2], "little"))
            pos += 2
        else:
            idx_sorted.append(buf[pos])
            pos += 1
    n_rec = buf[pos]
    pos += 1
    n_blocks = (k + 4) // 5
    wq_s = np.zeros(k, dtype=np.float64)
    for ri in range(n_rec):
        blk_id = ri if ri < n_blocks else buf[pos]
        if ri >= n_blocks:
            pos += 1
        tag = buf[pos]
        pos += 1
        if tag == 255:
            continue
        exp2 = tag - 128
        best = buf[pos]
        pos += 1
        bval = buf[pos]
        pos += 1
        sc = 2.0**exp2
        lvl = float(DYADIC_LEVELS[best])
        trits = [0] * 5
        v = bval
        for i in range(4, -1, -1):
            trits[i] = (v % 3) - 1
            v //= 3
        sl = slice(blk_id * 5, min((blk_id + 1) * 5, k))
        wq_s[sl] += np.array(trits[: sl.stop - sl.start], dtype=np.float64) * (lvl * sc)
    out = np.zeros(D, dtype=np.float64)
    if k > 0:
        out[np.array(idx_sorted, dtype=np.int64)] = wq_s
    return out


# ==============================================================================
# 5. Range-Safe Log-Linear CE, BF16 Guard, EF21-Muon, Continual GPM & Neural Core
# ==============================================================================


def bf16_nearest_float(x: float) -> float:
    """IEEE-754 BF16 round-to-nearest-even (relative error <= 2^-8 in normal binade)."""
    if x == 0.0 or not math.isfinite(x):
        return 0.0 if x == 0.0 else x
    ax = abs(x)
    if ax < 2.0 ** (-126):
        return 0.0
    exp2 = math.floor(math.log2(ax))
    step = 2.0 ** (exp2 - 7)
    return math.copysign(round(ax / step) * step, x)


def batch_logit_safe_step_ce(
    X_feat: np.ndarray,
    y_int: np.ndarray,
    num_classes: int,
    steps: int = 300,
    l2_reg: Optional[np.ndarray] = None,
    use_bf16_guard: bool = False,
) -> dict[str, Any]:
    """C223/C230/C249: Certified range-safe step eta = 4*A / (Q' + 4*curv) + BF16 4*eps gate."""
    B, d = X_feat.shape
    theta = np.zeros((num_classes, d), dtype=np.float64)
    reg = (
        np.zeros(d, dtype=np.float64)
        if l2_reg is None
        else np.asarray(l2_reg, dtype=np.float64)
    )

    def ce_loss(th: np.ndarray) -> float:
        lg = X_feat @ th.T
        lg_s = lg - np.max(lg, axis=1, keepdims=True)
        lse = np.max(lg, axis=1) + np.log(np.sum(np.exp(lg_s), axis=1))
        target_l = lg[np.arange(B), y_int]
        reg_term = 0.5 * float(np.sum(reg[None, :] * (th**2)))
        return float(np.mean(lse - target_l)) + reg_term

    loss0 = ce_loss(theta)
    cur_loss = loss0
    accepted = held = 0
    bf16_vec = np.vectorize(bf16_nearest_float)
    for _ in range(steps):
        lg = X_feat @ theta.T
        lg_s = lg - np.max(lg, axis=1, keepdims=True)
        exp_l = np.exp(lg_s)
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        res = probs.copy()
        res[np.arange(B), y_int] -= 1.0
        g_bar = (res.T @ X_feat) / B + (reg[None, :] * theta)
        D = -g_bar
        A = float(np.sum(g_bar * g_bar))
        if A < 1e-18:
            held += 1
            break
        d_logits = X_feat @ D.T
        r_prime = np.max(d_logits, axis=1) - np.min(d_logits, axis=1)
        Q_prime = float(np.mean(r_prime * r_prime))
        curv_reg = (
            2.0 * float(np.max(reg)) * float(np.sum(D * D)) if np.any(reg > 0) else 0.0
        )
        denom = Q_prime + 4.0 * curv_reg
        if denom <= 1e-18:
            held += 1
            break
        eta = 4.0 * A / denom
        cert_drop = 2.0 * (A**2) / denom
        cand = theta + eta * D
        if use_bf16_guard:
            cand_bf = bf16_vec(cand)
            lg_exact = X_feat @ cand.T
            lg_bf = X_feat @ cand_bf.T
            eps_pert = float(np.max(np.abs(lg_bf - lg_exact)))
            if 4.0 * eps_pert >= cert_drop:
                held += 1
                break
            cand = cand_bf
        cand_loss = ce_loss(cand)
        if cand_loss <= cur_loss + 1e-12:
            theta = cand
            cur_loss = cand_loss
            accepted += 1
        else:
            held += 1
            break
    return {
        "theta": theta,
        "loss0": loss0,
        "loss_final": cur_loss,
        "total_drop": loss0 - cur_loss,
        "accepted_steps": accepted,
        "hold_steps": held,
    }


def newton_schulz_orthogonalize(G: np.ndarray, iters: int = 4) -> np.ndarray:
    """C296/C297: X <- X(3I - X^T X)/2."""
    X = G.astype(np.float64).copy()
    norm = float(np.linalg.norm(X))
    if norm < 1e-12:
        return X
    X /= norm
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    I_m = np.eye(X.shape[0], dtype=np.float64)
    for _ in range(iters):
        X = 0.5 * (3.0 * I_m - X @ X.T) @ X
    return X.T if transposed else X


class ContinualGPMMemoryV9(GpuGpmMixin):
    """GPM + SpaceDecoupling (P_R = I - UU^T) + EF21Markov + Newton-Schulz Muon.

    train_task_ef21_muon / consolidate_subspace dispatch to the GPU
    (torch/CUDA-ROCm) backend when available — see hagi/gpu.py;
    the CPU implementations are preserved as *_cpu methods."""

    def __init__(self, d_in: int, d_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.1, size=(d_in, d_out))
        self.U_basis = np.zeros((d_in, 0), dtype=np.float64)
        self.ef_v = np.zeros_like(self.W)

    def snapshot_optimizer(self) -> dict[str, np.ndarray]:
        return {
            "W": self.W.copy(),
            "U_basis": self.U_basis.copy(),
            "ef_v": self.ef_v.copy(),
        }

    def restore_optimizer(self, snap: dict[str, np.ndarray]) -> None:
        self.W = snap["W"].copy()
        self.U_basis = snap["U_basis"].copy()
        self.ef_v = snap["ef_v"].copy()

    def project_orthogonal(self, G: np.ndarray) -> np.ndarray:
        if self.U_basis.shape[1] == 0:
            return G
        return G - self.U_basis @ (self.U_basis.T @ G)

    def _train_task_ef21_muon_cpu(
        self, X: np.ndarray, Y: np.ndarray, steps: int = 35, lr: float = 0.2
    ) -> float:
        if len(X) == 0:
            return 0.0
        for _ in range(steps):
            err = X @ self.W - Y
            grad = (2.0 / len(X)) * (X.T @ err)
            g_proj = self.project_orthogonal(grad)
            diff = g_proj - self.ef_v
            self.ef_v = self.ef_v + 0.5 * diff
            step_dir = newton_schulz_orthogonalize(self.ef_v, iters=4)
            step_dir = self.project_orthogonal(step_dir)
            if self.U_basis.shape[1] > 0:
                drift = float(np.max(np.abs(self.U_basis.T @ step_dir)))
                assert drift <= 1e-9, f"SpaceDecoupling violated: {drift}"
            cur = float(np.mean((X @ self.W - Y) ** 2))
            for eta in (lr, lr * 0.5, lr * 0.1, 0.01):
                cand = self.W - eta * step_dir
                if float(np.mean((X @ cand - Y) ** 2)) < cur:
                    self.W = cand
                    break
        return float(np.mean((X @ self.W - Y) ** 2))

    def _consolidate_subspace_cpu(
        self, X: np.ndarray, energy_thresh: float = 0.995
    ) -> None:
        res_X = (
            X if self.U_basis.shape[1] == 0 else X - (X @ self.U_basis) @ self.U_basis.T
        )
        if float(np.linalg.norm(res_X)) < 1e-10:
            return
        _, S, Vt = np.linalg.svd(res_X, full_matrices=False)
        pos = S > 1e-10
        if not np.any(pos):
            return
        S = S[pos]
        Vt = Vt[pos]
        tot = float(np.sum(S**2))
        cum = np.cumsum(S**2) / tot
        # WAVE 172 (C485 fix): float cumsum may cap at 0.99999...,
        # silently dropping trailing directions and breaking
        # gpm_zero_forgetting (loss(A) grew 32% after train(B)).
        # energy_thresh >= 1.0 means the FULL span.
        if energy_thresh >= 1.0:
            k = len(S)
        else:
            k = min(len(S), int(np.searchsorted(cum, energy_thresh)) + 1)
        new_dirs = Vt[:k].T
        if self.U_basis.shape[1] == 0:
            Q, _ = np.linalg.qr(new_dirs)
            self.U_basis = Q
        else:
            combined = np.hstack([self.U_basis, new_dirs])
            U_svd, s_svd, _ = np.linalg.svd(combined, full_matrices=False)
            keep = s_svd > 1e-9
            self.U_basis = U_svd[:, keep]
        # WAVE 172: keep full span up to d_in — the "-2" margin left
        # protected directions unconstrained (zero-forgetting leak).
        if self.U_basis.shape[1] > self.W.shape[0]:
            self.U_basis = self.U_basis[:, : self.W.shape[0]]


def quantize_bitalloc_exact(
    Phi: np.ndarray,
    idx: np.ndarray,
    wk: np.ndarray,
    y: np.ndarray,
    sy: float,
    lam: float,
    max_extra_stages: int = 12,
) -> tuple[np.ndarray, bytes]:
    """BitAlloc + EF21 Data-Aware ResidualVQ with exact byte codec."""
    if idx.size == 0:
        return np.zeros(Phi.shape[1], dtype=np.float64), b"\x00\x00\x00"
    mag_order = np.argsort(-np.abs(wk))
    idx_sorted = idx[mag_order]
    Sub_s = Phi[:, idx_sorted]
    col_energy = np.mean(Sub_s**2, axis=0)
    A = Sub_s.T @ Sub_s + max(lam, 1e-9) * np.eye(idx.size)
    k = idx.size
    n_blocks = (k + 4) // 5
    wq_s = np.zeros(k, dtype=np.float64)
    dw = wk[mag_order].copy()
    records: list[tuple[int, tuple[int, int, int]]] = []
    for b in range(n_blocks):
        sl = slice(b * 5, min((b + 1) * 5, k))
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))
        q_blk, meta = quantize_5block_exact(blk)
        wq_s[sl] += q_blk[: sl.stop - sl.start]
        records.append((b, meta))
    for _ in range(max_extra_stages):
        r_data = y - Sub_s @ wq_s
        if float(np.mean((r_data * sy) ** 2)) < 0.005:
            break
        dw = np.linalg.solve(A, Sub_s.T @ r_data)
        blk_errs = [
            float(
                np.sum(
                    (dw[b * 5 : min((b + 1) * 5, k)] ** 2)
                    * col_energy[b * 5 : min((b + 1) * 5, k)]
                )
            )
            for b in range(n_blocks)
        ]
        best_b = int(np.argmax(blk_errs))
        if blk_errs[best_b] * (sy * sy) < 1e-6:
            break
        sl = slice(best_b * 5, min((best_b + 1) * 5, k))
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))
        q_blk, meta = quantize_5block_exact(blk)
        wq_s[sl] += q_blk[: sl.stop - sl.start]
        records.append((best_b, meta))
    packed = pack_bitalloc_records(k, idx_sorted, records)
    rec = unpack_bitalloc_records(packed, Phi.shape[1])
    return rec, packed


def _omp_candidates(
    Phi: np.ndarray, y: np.ndarray, max_k: int = 16, lam: float = 1e-7
) -> list[tuple[np.ndarray, np.ndarray]]:
    N, D = Phi.shape
    norms = np.sqrt(np.sum(Phi**2, axis=0)) + 1e-12
    Phi_n = Phi / norms[None, :]
    r = y.copy()
    selected: list[int] = []
    sel_set: set[int] = set()
    cands: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(min(max_k, D, max(1, N - 1))):
        corrs = np.abs(Phi_n.T @ r)
        for s in sel_set:
            corrs[s] = -1.0
        j = int(np.argmax(corrs))
        if corrs[j] <= 1e-12:
            break
        selected.append(j)
        sel_set.add(j)
        idx = np.array(selected, dtype=np.int64)
        Sub = Phi[:, idx]
        A = Sub.T @ Sub + lam * np.eye(len(idx))
        wk = np.linalg.solve(A, Sub.T @ y)
        r = y - Sub @ wk
        cands.append((idx.copy(), wk.copy()))
        if float(np.mean(r * r)) < 1e-9:
            break
    return cands


def _press_score_ridge(
    Phi_sub: np.ndarray,
    wk: np.ndarray,
    y: np.ndarray,
    ty: np.ndarray,
    sy: float,
    lam: float,
) -> tuple[float, float]:
    """Exact Leave-One-Out (Allen's PRESS) error via Hat Matrix diagonal h_i."""
    N, D = Phi_sub.shape
    pred = (Phi_sub @ wk) * sy
    resid = ty - pred
    if D >= N:
        return float(np.mean(resid**2)) * 100.0 + float(np.sum(wk**2)), 0.0
    A = Phi_sub.T @ Phi_sub + max(lam, 1e-8) * np.eye(D)
    try:
        A_inv_Pt = np.linalg.solve(A, Phi_sub.T)
    except np.linalg.LinAlgError:
        return float(np.mean(resid**2)) * 100.0, 0.0
    h = np.sum(Phi_sub * A_inv_Pt.T, axis=1)
    denom = np.maximum(0.08, 1.0 - h)
    loo_pred = ty - resid / denom
    w_norm_pen = 1e-3 * float(np.sum(wk**2)) * (sy**2) / max(1, N)
    loo_mse = float(np.mean((loo_pred - ty) ** 2)) + w_norm_pen
    loo_acc = float(np.mean(np.rint(loo_pred) == ty))
    return loo_mse, loo_acc


class FactorizedSharedCoreV9:
    """Minimal-volume shared core (Hadamard+SwiGLU+Fourier + Continual GPM)."""

    BRANCHES = ("poly", "hinge", "bit", "core")

    def __init__(self, seed: int = 42, width: int = 48):
        self.seed = seed
        self.width = width
        rng = np.random.default_rng(seed)
        Q1, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, 8)))
        scales = np.repeat([0.6, 1.2, 2.4, 4.8], width // 4)
        self.W1 = Q1.T * scales[None, :]
        self.b1 = rng.uniform(-1.5, 1.5, (1, width))
        Q2, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, width)))
        self.W2a = Q2 * 1.3
        self.b2a = rng.uniform(-0.8, 0.8, (1, width))
        Q2g, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, width)))
        self.W2g = Q2g * 1.3
        self.b2g = rng.uniform(-0.8, 0.8, (1, width))
        self.H8 = sylvester_hadamard(3)
        self.gpm = ContinualGPMMemoryV9(d_in=16, d_out=1, seed=seed + 7)
        self.core_bytes = 8

    def snapshot(self) -> dict[str, Any]:
        return {"gpm": self.gpm.snapshot_optimizer(), "core_bytes": self.core_bytes}

    def restore(self, snap: dict[str, Any]) -> None:
        self.gpm.restore_optimizer(snap["gpm"])
        self.core_bytes = snap["core_bytes"]

    def features(self, branch: str, xs_arr: np.ndarray, sx: float) -> np.ndarray:
        N, d = xs_arr.shape
        xn = xs_arr.astype(np.float64) / sx
        step1 = 1.0 / sx
        if branch == "poly":
            cols = [xn[:, i] for i in range(d)]
            fs = [np.ones((N, 1), dtype=np.float64)]
            for i in range(d):
                fs.append(cols[i][:, None])
            for i in range(d):
                for j in range(i, d):
                    fs.append((cols[i] * cols[j])[:, None])
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        fs.append((cols[i] * cols[j] * cols[k])[:, None])
            if d <= 2:
                for i in range(d):
                    for j in range(i, d):
                        for k in range(j, d):
                            for l in range(k, d):
                                fs.append(
                                    (cols[i] * cols[j] * cols[k] * cols[l])[:, None]
                                )
            else:
                for i in range(d):
                    fs.append((cols[i] ** 4)[:, None])
            return np.hstack(fs)
        if branch == "hinge":
            fs = [np.ones((N, 1), dtype=np.float64)]
            for i in range(d):
                fs.append(xn[:, i : i + 1])
                fs.append(np.abs(xn[:, i : i + 1]))
                for knot in (-0.375, -step1, 0.0, step1, 0.375):
                    fs.append(np.maximum(0.0, xn[:, i] - knot)[:, None])
                    fs.append(np.maximum(0.0, knot - xn[:, i])[:, None])
                fs.append(np.maximum(0.0, 1.0 - sx * np.abs(xn[:, i : i + 1])))
            for i in range(d):
                for j in range(i + 1, d):
                    diff = xn[:, i] - xn[:, j]
                    fs.append(np.abs(diff)[:, None])
                    for knot in (-step1, 0.0, step1):
                        fs.append(np.maximum(0.0, diff - knot)[:, None])
                        fs.append(np.maximum(0.0, knot - diff)[:, None])
                    fs.append(np.maximum(0.0, 1.0 - sx * np.abs(diff))[:, None])
            return np.hstack(fs)
        if branch == "bit":
            xi = np.rint(xs_arr).astype(np.int64)
            fs = [np.ones((N, 1), dtype=np.float64)]
            for i in range(d):
                fs.append((xi[:, i] < 0).astype(np.float64)[:, None])
                for b in range(8):
                    fs.append(((xi[:, i] >> b) & 1).astype(np.float64)[:, None])
            if d >= 2:
                for b in range(6):
                    b0 = ((xi[:, 0] >> b) & 1).astype(np.float64)[:, None]
                    b1 = ((xi[:, 1] >> b) & 1).astype(np.float64)[:, None]
                    fs.append(b0 * b1)
                    fs.append(np.abs(b0 - b1))
                fs.append(((xi[:, 0] < 0) * (xi[:, 1] < 0)).astype(np.float64)[:, None])
                fs.append(((xi[:, 0] < 0) ^ (xi[:, 1] < 0)).astype(np.float64)[:, None])
            if d == 3:
                b0 = (xi[:, 0] & 1).astype(np.float64)[:, None]
                b1 = (xi[:, 1] & 1).astype(np.float64)[:, None]
                b2 = (xi[:, 2] & 1).astype(np.float64)[:, None]
                fs.extend([b0 * b2, b1 * b2, b0 * b1 * b2])
            return np.hstack(fs)
        xp = np.zeros((N, 8), dtype=np.float64)
        xp[:, : min(3, d)] = xn[:, : min(3, d)]
        xp[:, 3] = xn[:, 0] - xn[:, 1] if d >= 2 else np.abs(xn[:, 0])
        xp[:, 4] = xn[:, 1] - xn[:, 2] if d >= 3 else np.maximum(0.0, xn[:, 0])
        xp[:, 5] = xn[:, 0] - xn[:, 2] if d >= 3 else np.maximum(0.0, -xn[:, 0])
        xp[:, 6] = np.abs(xp[:, 3])
        xp[:, 7] = np.abs(xp[:, 4])
        xh = xp @ self.H8
        z1 = xh @ self.W1 + self.b1
        h1_t = np.tanh(z1)
        h1_r = np.maximum(0.0, z1)
        z2a = h1_r @ self.W2a + self.b2a
        z2g = h1_r @ self.W2g + self.b2g
        sig2g = 1.0 / (1.0 + np.exp(-np.clip(z2g, -15.0, 15.0)))
        h2_sg = z2a * sig2g
        four = []
        for f in (1.0, 2.0, 4.0):
            for i in range(min(3, d)):
                four.append(np.sin(math.pi * f * xn[:, i : i + 1]))
                four.append(np.cos(math.pi * f * xn[:, i : i + 1]))
        return np.hstack(
            [
                self.features("poly", xs_arr, sx),
                self.features("hinge", xs_arr, sx)[:, 1:],
                h1_t,
                h1_r,
                h2_sg,
            ]
            + four
        )

    def _mlp_encode(self, A: np.ndarray, sx: float) -> np.ndarray:
        an = A / sx
        d = A.shape[1]
        cols = [an, np.abs(an), np.ones((A.shape[0], 1))]
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(an[:, i : i + 1] - an[:, j : j + 1])
                    cols.append(np.abs(an[:, i : i + 1] - an[:, j : j + 1]))
        return np.hstack(cols)

    def _fit_stage2_mlp(
        self,
        tx: np.ndarray,
        resid_n: np.ndarray,
        vx: np.ndarray,
        sx: float,
        hidden: int = 12,
        steps: int = 280,
        seed: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        Xin = self._mlp_encode(tx, sx)
        Xvin = self._mlp_encode(vx, sx)
        N, din = Xin.shape
        W1 = rng.normal(0.0, 1.0 / math.sqrt(din), (din, hidden))
        b1 = np.zeros((1, hidden))
        W2 = np.zeros((hidden, 1))
        b2 = np.zeros((1, 1))
        yn = resid_n.reshape(-1, 1)
        m1 = np.zeros_like(W1)
        m2 = np.zeros_like(W2)
        v2 = np.zeros_like(W2)
        mb1 = np.zeros_like(b1)
        vb1 = np.zeros_like(b1)
        mb2 = np.zeros_like(b2)
        vb2 = np.zeros_like(b2)
        best = (float(np.mean(yn**2)), W1.copy(), b1.copy(), W2.copy(), b2.copy())
        lr0 = 0.04
        for t in range(1, steps + 1):
            cos_lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * t / steps))
            h = np.tanh(Xin @ W1 + b1)
            pred = h @ W2 + b2
            err = pred - yn
            loss = float(np.mean(err**2))
            if loss < best[0]:
                best = (loss, W1.copy(), b1.copy(), W2.copy(), b2.copy())
            dp = (2.0 / N) * err
            gW2 = h.T @ dp + 2e-4 * W2
            gb2 = np.sum(dp, axis=0, keepdims=True)
            dh = (dp @ W2.T) * (1.0 - h * h)
            gW1 = Xin.T @ dh + 2e-4 * W1
            gb1 = np.sum(dh, axis=0, keepdims=True)
            m1 = 0.85 * m1 + 0.15 * gW1
            W1 -= (cos_lr * 0.7) * newton_schulz_orthogonalize(m1, iters=3)
            m2 = 0.9 * m2 + 0.1 * gW2
            v2 = 0.999 * v2 + 0.001 * (gW2 * gW2)
            W2 -= cos_lr * (m2 / (1 - 0.9**t)) / (np.sqrt(v2 / (1 - 0.999**t)) + 1e-8)
            mb1 = 0.9 * mb1 + 0.1 * gb1
            vb1 = 0.999 * vb1 + 0.001 * (gb1 * gb1)
            b1 -= cos_lr * (mb1 / (1 - 0.9**t)) / (np.sqrt(vb1 / (1 - 0.999**t)) + 1e-8)
            mb2 = 0.9 * mb2 + 0.1 * gb2
            vb2 = 0.999 * vb2 + 0.001 * (gb2 * gb2)
            b2 -= cos_lr * (mb2 / (1 - 0.9**t)) / (np.sqrt(vb2 / (1 - 0.999**t)) + 1e-8)
        _, W1, b1, W2, b2 = best
        all_w = np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])
        wr, orig_n = block_hadamard_rotate(all_w, 8)
        packed_mlp, wr_q = pack_vector_dyadic5(wr, stages=2)
        all_wq = block_hadamard_unrotate(wr_q, orig_n, 8)
        p1 = W1.size
        p2 = p1 + b1.size
        p3 = p2 + W2.size
        W1q = all_wq[:p1].reshape(W1.shape)
        b1q = all_wq[p1:p2].reshape(b1.shape)
        W2q = all_wq[p2:p3].reshape(W2.shape)
        b2q = all_wq[p3:].reshape(b2.shape)
        return (
            (np.tanh(Xin @ W1q + b1q) @ W2q + b2q).ravel(),
            (np.tanh(Xvin @ W1q + b1q) @ W2q + b2q).ravel(),
            {"W1": W1q, "b1": b1q, "W2": W2q, "b2": b2q, "packed": packed_mlp},
        )

    def fit_residual(
        self,
        train_x: Sequence[Sequence[int]],
        train_y: Sequence[int],
        val_x: Sequence[Sequence[int]],
        val_y: Sequence[int],
        router: Optional["HedgeRouterV9"] = None,
    ) -> "FactorizedAdapterV9":
        tx = np.asarray(train_x, dtype=np.float64)
        ty = np.asarray(train_y, dtype=np.float64)
        vx = np.asarray(val_x, dtype=np.float64) if val_x else tx
        vy = np.asarray(val_y, dtype=np.float64) if val_y else ty
        sx = max(1.0, float(np.max(np.abs(tx))))
        sy = max(1.0, float(np.max(np.abs(ty))))
        y = ty / sy
        N, d = tx.shape

        gpm_in = self.features("poly", tx, sx)
        if gpm_in.shape[1] < 16:
            gpm_in = np.pad(gpm_in, ((0, 0), (0, 16 - gpm_in.shape[1])))
        else:
            gpm_in = gpm_in[:, :16]
        self.gpm.train_task_ef21_muon(gpm_in, y.reshape(-1, 1), steps=15, lr=0.15)
        self.gpm.consolidate_subspace(gpm_in, energy_thresh=0.995)

        branch_order = list(self.BRANCHES)
        if router is not None:
            branch_order = [self.BRANCHES[i] for i in np.argsort(-router.weights)]

        best = (float("inf"), "poly", "reg", None, None, b"", None, None, None)
        branch_losses = np.ones(len(self.BRANCHES), dtype=np.float64)

        for bname in branch_order:
            Phi = self.features(bname, tx, sx)
            max_k = min(14, max(2, N // 2) if N >= 20 else max(1, min(8, N - 2)))
            b_best_acc = 0.0
            float_cands = []
            for lam in (1e-6, 1e-3, 1e-1):
                for idx, wk in _omp_candidates(Phi, y, max_k=max_k, lam=lam):
                    if np.max(np.abs(wk)) > 8.0:
                        continue
                    pred_f = (Phi[:, idx] @ wk) * sy
                    mse_tr_f = float(np.mean((pred_f - ty) ** 2))
                    acc_tr_f = float(np.mean(np.rint(pred_f) == ty))
                    mse_loo, acc_loo = _press_score_ridge(
                        Phi[:, idx], wk, y, ty, sy, lam=lam
                    )
                    b_best_acc = max(b_best_acc, acc_loo)
                    est_nb = 9 + len(idx) + ((len(idx) + 4) // 5) * 3
                    f_obj = (
                        0.4 * mse_tr_f
                        + 0.6 * mse_loo
                        + 0.002 * est_nb
                        + (1.0 - 0.5 * (acc_tr_f + acc_loo)) * 0.5
                    )
                    float_cands.append((f_obj, lam, idx, wk, mse_loo, acc_loo))
            float_cands.sort(key=lambda c: c[0])
            for _, lam, idx, wk, mse_loo, acc_loo in float_cands[:3]:
                wq, packed = quantize_bitalloc_exact(Phi, idx, wk, y, sy, lam)
                pred_q = (Phi @ wq) * sy
                mse_tr = float(np.mean((pred_q - ty) ** 2))
                acc_tr = float(np.mean(np.rint(pred_q) == ty))
                nb = 6 + len(packed)
                obj = (
                    0.4 * mse_tr
                    + 0.6 * mse_loo
                    + 0.002 * nb
                    + (1.0 - 0.5 * (acc_tr + acc_loo)) * 0.5
                )
                if obj < best[0]:
                    best = (
                        obj,
                        bname,
                        "reg",
                        wq,
                        packed,
                        packed,
                        None,
                        pred_q / sy,
                        None,
                    )
            b_idx = self.BRANCHES.index(bname)
            branch_losses[b_idx] = 1.0 - b_best_acc
            if best[0] < 0.025:
                break

        classes = sorted(set(ty.astype(np.int64).tolist()))
        if 2 <= len(classes) <= 5 and N >= 12:
            cls_map = {c: i for i, c in enumerate(classes)}
            y_idx = np.array([cls_map[int(c)] for c in ty], dtype=np.int64)
            Phi_h = self.features("hinge", tx, sx)
            scores = np.zeros(Phi_h.shape[1])
            for c_i in range(len(classes)):
                yc = (y_idx == c_i).astype(np.float64) - np.mean(y_idx == c_i)
                scores += np.abs(Phi_h.T @ yc)
            top_cols = np.argsort(-scores)[: min(4, Phi_h.shape[1])]
            Phi_sub = Phi_h[:, top_cols]
            res = batch_logit_safe_step_ce(
                Phi_sub,
                y_idx,
                len(classes),
                steps=300,
                l2_reg=np.full(top_cols.size, 1e-6),
            )
            theta = res["theta"]
            raw_ce, th_dec = pack_vector_dyadic5(theta.ravel(), stages=2)
            th_q = th_dec.reshape(theta.shape)
            nb_ce = 6 + top_cols.size + len(raw_ce)
            pred_tr = np.array(
                [classes[i] for i in np.argmax(Phi_sub @ th_q.T, axis=1)],
                dtype=np.float64,
            )
            acc_tr = float(np.mean(pred_tr == ty))
            mse_tr = float(np.mean((pred_tr - ty) ** 2))
            obj_ce = mse_tr + 0.002 * nb_ce + (1.0 - acc_tr) * 0.5
            if obj_ce < best[0]:
                ce_meta = {
                    "classes": classes,
                    "cols": top_cols.tolist(),
                    "theta": th_q,
                    "packed": raw_ce,
                }
                best = (
                    obj_ce,
                    "hinge",
                    "ce",
                    None,
                    raw_ce,
                    raw_ce,
                    ce_meta,
                    pred_tr / sy,
                    None,
                )

        if router is not None:
            router.update(branch_losses)

        _, bname, mode, wq, packed, _, ce_meta, ptr_n, mlp_meta = best

        if ptr_n is None:
            # Degenerate fit: every OMP candidate was rejected (|w| > 8) and
            # the CE branch was inapplicable. Certify a conservative zero
            # predictor instead of crashing — the commit gate downstream
            # sees the true (poor) val metrics and can HOLD on this expert.
            wq = np.zeros(Phi.shape[1], dtype=np.float64)
            packed = pack_vector_dyadic5(wq, stages=1)[0]
            mode, bname, ce_meta, mlp_meta = "reg", "poly", None, None
            ptr_n = np.zeros(N, dtype=np.float64)

        acc_tr_cur = float(np.mean(np.rint(ptr_n * sy) == ty))
        mse_tr_cur = float(np.mean((ptr_n * sy - ty) ** 2))
        if acc_tr_cur < 0.99 and N >= 120 and mse_tr_cur > 0.08 and mode == "reg":
            resid_tr = y - ptr_n
            cut = int(0.8 * N)
            _, r_icv, _ = self._fit_stage2_mlp(
                tx[:cut], resid_tr[:cut], tx[cut:], sx, hidden=12, steps=260, seed=7
            )
            mse_icv_0 = float(np.mean((resid_tr[cut:] * sy) ** 2))
            mse_icv_1 = float(np.mean(((resid_tr[cut:] - r_icv) * sy) ** 2))
            if mse_icv_1 < 0.72 * mse_icv_0:
                _, _, mlp_meta = self._fit_stage2_mlp(
                    tx, resid_tr, vx, sx, hidden=12, steps=280, seed=7
                )
                mode = "rvq2"

        Phi_va = self.features(bname, vx, sx)

        def _align_cols(P: np.ndarray, n: int) -> np.ndarray:
            """Feature width can differ between train and val inputs for
            data-dependent branches (observed 33 vs 15 on arity-3 replay):
            align to the weight vector — truncate extras, zero-pad missing."""
            if P.shape[1] == n:
                return P
            if P.shape[1] > n:
                return P[:, :n]
            return np.pad(P, ((0, 0), (0, n - P.shape[1])))

        if mode == "ce":
            assert ce_meta is not None
            Pv = Phi_va[:, ce_meta["cols"]]
            pred_q = np.array(
                [
                    ce_meta["classes"][i]
                    for i in np.argmax(Pv @ ce_meta["theta"].T, axis=1)
                ],
                dtype=np.float64,
            )
            nb_total = 6 + len(ce_meta["cols"]) + len(ce_meta["packed"])
        elif mode == "rvq2":
            assert mlp_meta is not None and wq is not None
            base_va = _align_cols(Phi_va, len(wq)) @ wq
            Xv_mlp = self._mlp_encode(vx, sx)
            r_va = (
                np.tanh(Xv_mlp @ mlp_meta["W1"] + mlp_meta["b1"]) @ mlp_meta["W2"]
                + mlp_meta["b2"]
            ).ravel()
            pred_q = (base_va + r_va) * sy
            nb_total = 6 + len(packed) + len(mlp_meta["packed"])
        else:
            assert wq is not None
            pred_q = (_align_cols(Phi_va, len(wq)) @ wq) * sy
            nb_total = 6 + len(packed)

        return FactorizedAdapterV9(
            arity=d,
            branch=bname,
            mode=mode,
            sx=sx,
            sy=sy,
            w_quant=wq,
            packed_bytes=packed,
            ce_meta=ce_meta,
            mlp_meta=mlp_meta,
            compressed_bytes=nb_total,
            val_exact_acc_quant=float(np.mean(np.rint(pred_q) == vy)),
            val_mse_quant=float(np.mean((pred_q - vy) ** 2)),
        )


@dataclass
class FactorizedAdapterV9:
    arity: int
    branch: str
    mode: str
    sx: float
    sy: float
    w_quant: Optional[np.ndarray]
    packed_bytes: bytes
    ce_meta: Optional[dict[str, Any]]
    mlp_meta: Optional[dict[str, Any]]
    compressed_bytes: int
    val_exact_acc_quant: float
    val_mse_quant: float

    def predict(self, core: FactorizedSharedCoreV9, xs: Sequence[int]) -> float:
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1)
        Phi = core.features(self.branch, x, self.sx)
        if self.mode == "ce" and self.ce_meta is not None:
            P = Phi[:, self.ce_meta["cols"]]
            idx = int(np.argmax((P @ self.ce_meta["theta"].T)[0]))
            return float(self.ce_meta["classes"][idx])
        assert self.w_quant is not None
        val = float((Phi @ self.w_quant)[0])
        if self.mode == "rvq2" and self.mlp_meta is not None:
            Xm = core._mlp_encode(x, self.sx)
            r = float(
                (
                    np.tanh(Xm @ self.mlp_meta["W1"] + self.mlp_meta["b1"])
                    @ self.mlp_meta["W2"]
                    + self.mlp_meta["b2"]
                )[0, 0]
            )
            val += r
        return val * self.sy

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "arity": self.arity,
            "branch": self.branch,
            "mode": self.mode,
            "sx": self.sx,
            "sy": self.sy,
            "compressed_bytes": self.compressed_bytes,
            "packed_hex": self.packed_bytes.hex(),
            "val_exact_acc_quant": self.val_exact_acc_quant,
            "val_mse_quant": self.val_mse_quant,
        }
        if self.mode == "ce" and self.ce_meta is not None:
            out["ce_classes"] = self.ce_meta["classes"]
            out["ce_cols"] = self.ce_meta["cols"]
        elif self.mode == "rvq2" and self.mlp_meta is not None:
            out["mlp_packed_hex"] = self.mlp_meta["packed"].hex()
            out["mlp_shapes"] = [
                list(self.mlp_meta["W1"].shape),
                list(self.mlp_meta["b1"].shape),
                list(self.mlp_meta["W2"].shape),
                list(self.mlp_meta["b2"].shape),
            ]
        return out

    @staticmethod
    def from_dict(
        d: dict[str, Any], core: "FactorizedSharedCoreV9"
    ) -> "FactorizedAdapterV9":
        arity = int(d["arity"])
        branch = str(d["branch"])
        mode = str(d["mode"])
        sx = float(d["sx"])
        sy = float(d["sy"])
        packed = bytes.fromhex(d["packed_hex"])
        D = core.features(branch, np.zeros((1, arity), dtype=np.float64), sx).shape[1]
        w_quant = None
        ce_meta = None
        mlp_meta = None
        if mode == "ce":
            classes = [int(c) for c in d["ce_classes"]]
            cols = [int(c) for c in d["ce_cols"]]
            th = unpack_vector_dyadic5(packed).reshape(len(classes), len(cols))
            ce_meta = {"classes": classes, "cols": cols, "theta": th, "packed": packed}
        else:
            w_quant = unpack_bitalloc_records(packed, D)
            if mode == "rvq2" and "mlp_packed_hex" in d:
                mp = bytes.fromhex(d["mlp_packed_hex"])
                s1, sb1, s2, sb2 = [tuple(int(x) for x in sh) for sh in d["mlp_shapes"]]
                n_all = (
                    s1[0] * s1[1] + sb1[0] * sb1[1] + s2[0] * s2[1] + sb2[0] * sb2[1]
                )
                wr_q = unpack_vector_dyadic5(mp)
                all_wq = block_hadamard_unrotate(wr_q, n_all, 8)
                p1 = s1[0] * s1[1]
                p2 = p1 + sb1[0] * sb1[1]
                p3 = p2 + s2[0] * s2[1]
                mlp_meta = {
                    "W1": all_wq[:p1].reshape(s1),
                    "b1": all_wq[p1:p2].reshape(sb1),
                    "W2": all_wq[p2:p3].reshape(s2),
                    "b2": all_wq[p3:].reshape(sb2),
                    "packed": mp,
                }
        return FactorizedAdapterV9(
            arity=arity,
            branch=branch,
            mode=mode,
            sx=sx,
            sy=sy,
            w_quant=w_quant,
            packed_bytes=packed,
            ce_meta=ce_meta,
            mlp_meta=mlp_meta,
            compressed_bytes=int(d["compressed_bytes"]),
            val_exact_acc_quant=float(d["val_exact_acc_quant"]),
            val_mse_quant=float(d["val_mse_quant"]),
        )


# ==============================================================================
# 6. CRAM Energy-Based + Latent Reasoning + Mean-Field KL + Pseudolikelihood
# ==============================================================================


class CRAMEngineV9:
    @staticmethod
    def solve_coloring(
        edges: Sequence[tuple[int, int]],
        n_nodes: int,
        num_colors: int,
        givens: Optional[dict[int, int]] = None,
        eps_noise: float = 0.0,
        max_sweeps: int = 800,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Sparse Local-Field Single-Cell Descent + 2*eps Guard + Heat-Bath Cooling + ValidityCertificate."""
        adj = [set() for _ in range(n_nodes)]
        for a, b in edges:
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
        adj_list = [sorted(s) for s in adj]
        giv = dict(givens or {})
        rng = np.random.default_rng(seed)
        y = [giv.get(i, int(rng.integers(0, num_colors))) for i in range(n_nodes)]

        def energy(st: Sequence[int]) -> int:
            return sum(1 for a, b in edges if st[a] == st[b])

        def sparse_fields(i: int, st: Sequence[int]) -> list[int]:
            f = [0] * num_colors
            for j in adj_list[i]:
                f[st[j]] += 1
            return f

        cur_e = energy(y)
        sweeps = hb_steps = stale = 0
        T = 0.85
        alpha_cool = 0.96
        sparse_work = sum(len(adj_list[i]) * (num_colors**2) for i in range(n_nodes))
        dense_work = n_nodes * max(1, n_nodes - 1) * (num_colors**2)

        while sweeps < max_sweeps and cur_e > 0:
            improved = False
            for i in range(n_nodes):
                if i in giv:
                    continue
                fields = sparse_fields(i, y)
                noisy = [
                    f
                    + (
                        float(rng.uniform(-eps_noise, eps_noise))
                        if eps_noise > 0
                        else 0.0
                    )
                    for f in fields
                ]
                cur_f = noisy[y[i]]
                best_val = min(noisy)
                if cur_f - best_val > 2.0 * eps_noise:
                    ties = [c for c, v in enumerate(noisy) if v == best_val]
                    y[i] = ties[int(rng.integers(0, len(ties)))]
                    improved = True
            cur_e = energy(y)
            sweeps += 1
            if cur_e == 0:
                break
            if not improved:
                stale += 1
                if stale >= 8:
                    y = [
                        giv.get(i, int(rng.integers(0, num_colors)))
                        for i in range(n_nodes)
                    ]
                    stale = 0
                    T = 0.85
                    cur_e = energy(y)
                    continue
                for _ in range(min(12, max(3, cur_e * 2))):
                    conf = [
                        i
                        for i in range(n_nodes)
                        if i not in giv and any(y[i] == y[j] for j in adj_list[i])
                    ]
                    if not conf:
                        break
                    pick = conf[int(rng.integers(0, len(conf)))]
                    if rng.random() < 0.4 and adj_list[pick]:
                        free_nb = [j for j in adj_list[pick] if j not in giv]
                        if free_nb:
                            pick = free_nb[int(rng.integers(0, len(free_nb)))]
                    fields = sparse_fields(pick, y)
                    logits = np.array(
                        [-f / max(0.15, T) for f in fields], dtype=np.float64
                    )
                    logits -= np.max(logits)
                    probs = np.exp(logits)
                    probs /= np.sum(probs)
                    y[pick] = int(rng.choice(num_colors, p=probs))
                    hb_steps += 1
                T *= alpha_cool
                cur_e = energy(y)

        valid = all(y[a] != y[b] for a, b in edges) and all(
            y[i] == g for i, g in giv.items()
        )
        return {
            "coloring": y,
            "valid": valid,
            "energy": cur_e,
            "sweeps": sweeps,
            "hb_steps": hb_steps,
            "validity_certificate": bool(cur_e < 1 and valid),
            "givens_respected": all(y[i] == g for i, g in giv.items()),
            "sparse_work_per_sweep": sparse_work,
            "dense_work_per_sweep": dense_work,
        }

    @staticmethod
    def mean_field_kl_step(
        B: np.ndarray, K: np.ndarray, p: np.ndarray, beta: float = 2.0, iters: int = 12
    ) -> dict[str, Any]:
        """E2b/B1: Mean-Field Coordinate KL Descent: F(p) - F(q) = (1/beta)*KL(p_i || q_i) >= 0."""
        N, A = p.shape
        p_cur = p.copy()
        K_sym = 0.5 * (K + K.transpose(1, 0, 3, 2))

        def free_energy(pm: np.ndarray) -> float:
            unary = float(np.sum(B * pm))
            pair = 0.5 * float(np.einsum("ia,ijab,jb->", pm, K_sym, pm))
            ent = (
                float(
                    np.sum(
                        np.where(pm > 1e-18, pm * np.log(np.maximum(pm, 1e-18)), 0.0)
                    )
                )
                / beta
            )
            return unary + pair + ent

        f0 = free_energy(p_cur)
        f_hist = [f0]
        for _ in range(iters):
            for i in range(N):
                hz = (
                    B[i]
                    + np.einsum("jab,jb->a", K_sym[i], p_cur)
                    - np.einsum("ab,b->a", K_sym[i, i], p_cur[i])
                )
                logits = -beta * hz
                logits -= np.max(logits)
                q_i = np.exp(logits)
                q_i /= np.sum(q_i)
                p_old_i = p_cur[i].copy()
                kl = float(
                    np.sum(
                        np.where(
                            p_old_i > 1e-18,
                            p_old_i
                            * np.log(
                                np.maximum(p_old_i, 1e-18) / np.maximum(q_i, 1e-18)
                            ),
                            0.0,
                        )
                    )
                )
                f_before = free_energy(p_cur)
                p_cur[i] = q_i
                f_after = free_energy(p_cur)
                assert abs((f_before - f_after) - kl / beta) <= 1e-7
                f_hist.append(f_after)
        return {
            "p": p_cur,
            "free_energy_0": f0,
            "free_energy_final": f_hist[-1],
            "monotonic": all(
                f_hist[k + 1] <= f_hist[k] + 1e-9 for k in range(len(f_hist) - 1)
            ),
        }

    @staticmethod
    def solve_latent_maze(
        n_nodes: int, adj: Sequence[Sequence[int]], goal: int, eps_z: float = 1e-6
    ) -> dict[str, Any]:
        """E3/B2 Two-Phase Latent Contraction: m * z = A z + 1_G with m = Delta + 1."""
        delta = max((len(nb) for nb in adj), default=0)
        m = float(delta + 1)
        q = delta / m
        z = np.zeros(n_nodes, dtype=np.float64)
        iters = 0
        if q > 0:
            max_it = max(1, math.ceil(math.log(eps_z * (1.0 - q)) / math.log(q)))
            for _ in range(max_it):
                z_new = np.array(
                    [
                        (sum(z[j] for j in adj[i]) + (1.0 if i == goal else 0.0)) / m
                        for i in range(n_nodes)
                    ]
                )
                diff = float(np.max(np.abs(z_new - z)))
                z = z_new
                iters += 1
                if diff * q / (1.0 - q) <= eps_z:
                    break
        else:
            z[goal] = 1.0 / m
            iters = 1
        assert np.all(z >= -1e-12) and np.all(z <= 1.0 / (m - delta) + 1e-12)
        return {
            "z": z,
            "iters": iters,
            "contraction_q": q,
            "upper_bound": 1.0 / (m - delta),
        }

    @staticmethod
    def attention_coupling_matrix(
        kappa: np.ndarray, scores_z: np.ndarray, f_compat: np.ndarray
    ) -> np.ndarray:
        """B3 AttentionCoupling: symmetric by construction."""
        s = scores_z - np.max(scores_z, axis=1, keepdims=True)
        w = np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True)
        kw = kappa * w
        K = (
            kw[:, :, None, None] * f_compat[None, None, :, :]
            + kw.T[:, :, None, None] * f_compat.T[None, None, :, :]
        )
        for i in range(K.shape[0]):
            K[i, i] = 0.0
        return K

    @staticmethod
    def alternating_latent_visible_solve(
        n_nodes: int,
        adj: Sequence[Sequence[int]],
        goal: int,
        num_colors: int = 2,
        rounds: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        """B4 Alternating (z, y)-Rounds: settle latent z -> attention coupling -> guarded visible descent."""
        res_z = CRAMEngineV9.solve_latent_maze(n_nodes, adj, goal, eps_z=1e-6)
        z = res_z["z"]
        rng = np.random.default_rng(seed)
        kappa = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for i, nbs in enumerate(adj):
            for j in nbs:
                kappa[i, j] = 1.0
        f_compat = np.eye(num_colors, dtype=np.float64)
        scores_z = np.outer(z, z) * 4.0
        K = CRAMEngineV9.attention_coupling_matrix(kappa, scores_z, f_compat)
        assert np.allclose(K, K.transpose(1, 0, 3, 2))
        y = [int(rng.integers(0, num_colors)) for _ in range(n_nodes)]

        def pair_e(st):
            return 0.5 * sum(
                K[i, j, st[i], st[j]]
                for i in range(n_nodes)
                for j in range(n_nodes)
                if i != j
            )

        e0 = pair_e(y)
        for _ in range(rounds):
            for i in range(n_nodes):
                fields = [
                    sum(K[i, j, c, y[j]] for j in range(n_nodes) if j != i)
                    for c in range(num_colors)
                ]
                best_c = int(np.argmin(fields))
                if fields[best_c] < fields[y[i]] - 1e-12:
                    y[i] = best_c
        ef = pair_e(y)
        return {
            "z": z,
            "y": y,
            "energy_0": e0,
            "energy_final": ef,
            "monotone_drop": ef <= e0 + 1e-12,
        }

    @staticmethod
    def train_cram_pseudolikelihood(
        samples: np.ndarray, num_colors: int, beta: float = 1.0, steps: int = 80
    ) -> dict[str, Any]:
        """E1/B2 Pseudolikelihood -> Local-Field CE training of unary B."""
        M, N = samples.shape
        B_learned = np.zeros((N, num_colors), dtype=np.float64)
        losses = []
        for i in range(N):
            y_i = samples[:, i]
            X_i = np.ones((M, 1), dtype=np.float64)
            res = batch_logit_safe_step_ce(
                X_i, y_i, num_colors, steps=steps, l2_reg=np.array([1e-4])
            )
            B_learned[i] = -res["theta"][:, 0] / beta
            losses.append(res["loss_final"])
        return {"B": B_learned, "mean_pl_loss": float(np.mean(losses))}


# ==============================================================================
# 7. HNet Adaptive Chunking + GDNRAM Lag-1 Associative Memory
# ==============================================================================


def hnet_opt_ratio(a: float, b: float, T: float) -> float:
    """HNet continuous optimal chunk ratio s = (2*b*T/a)^(1/3)."""
    return (2.0 * b * T / a) ** (1.0 / 3.0)


def stutter_norm(seq: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def adjacent_pairs(xs: Sequence[Any]) -> list[tuple[Any, Any]]:
    if not xs:
        return []
    return [(xs[0], xs[0])] + list(zip(xs, xs[1:]))


def boundary_chunks(
    boundary: Callable[[Any], bool], xs: Sequence[Any]
) -> list[list[Any]]:
    if not xs:
        return []
    chunks = [[xs[0]]]
    for tok in xs[1:]:
        if boundary(tok):
            chunks.append([tok])
        else:
            chunks[-1].append(tok)
    return chunks


def capped_boundary_chunks(
    capacity: int, boundary: Callable[[Any], bool], xs: Sequence[Any]
) -> Optional[list[list[Any]]]:
    ch = boundary_chunks(boundary, xs)
    return ch if len(ch) <= capacity else None


def boundary_heads(boundary: Callable[[Any], bool], xs: Sequence[Any]) -> list[Any]:
    return [c[0] for c in boundary_chunks(boundary, xs)]


class GDNRAMV9:
    """Lag-1 associative store/load with verified roundtrip, slot isolation, stutter normalization."""

    def __init__(self, n_slots: int, dim: int):
        self.mem = np.zeros((n_slots, dim), dtype=np.float64)

    def snapshot(self) -> np.ndarray:
        return self.mem.copy()

    def restore(self, snap: np.ndarray) -> None:
        self.mem = snap.copy()

    def store(
        self, addr: int, val: np.ndarray, gate_beta: float = 1.0, decay_g: float = 0.0
    ) -> None:
        slot = int(addr) % self.mem.shape[0]
        v = np.asarray(val, dtype=np.float64).ravel()[: self.mem.shape[1]]
        if v.size < self.mem.shape[1]:
            v = np.pad(v, (0, self.mem.shape[1] - v.size))
        self.mem[slot] = (1.0 - decay_g) * self.mem[slot] * (
            1.0 - gate_beta
        ) + gate_beta * v

    def load(self, addr: int) -> np.ndarray:
        return self.mem[int(addr) % self.mem.shape[0]].copy()

    def load_stutter_head(self, addr_seq: Sequence[int]) -> np.ndarray:
        norm = stutter_norm(addr_seq)
        return (
            self.load(norm[0])
            if norm
            else np.zeros(self.mem.shape[1], dtype=np.float64)
        )


# ==============================================================================
# 8. Formal Runtime Governance: CertStepGen, AuditLedger, HedgeRouter
# ==============================================================================


@dataclass
class CertStepGen:
    """Canonical transition certificate (Hagi/Cert/StepGen.lean)."""

    version: int
    task: str
    action: str
    obj_before: float
    obj_after: float
    drop: float
    budget_rem: float
    budget_spend: float
    risk_rem: float
    risk_spend: float
    delta_rem: float
    delta_spend: float
    floors_kept: bool


class HedgeRouterV9:
    """Bounded-loss Hedge router with verified ewNorm anti-lock-in floor."""

    def __init__(self, n_experts: int, eta: float = 0.5):
        self.K = n_experts
        self.eta = eta
        self.weights = np.ones(n_experts, dtype=np.float64) / max(1, n_experts)
        self.cum_loss = np.zeros(n_experts, dtype=np.float64)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.weights.copy(), self.cum_loss.copy()

    def restore(self, snap: tuple[np.ndarray, np.ndarray]) -> None:
        self.weights = snap[0].copy()
        self.cum_loss = snap[1].copy()

    def update(self, losses_01: np.ndarray) -> np.ndarray:
        losses = np.clip(np.asarray(losses_01, dtype=np.float64), 0.0, 1.0)
        self.cum_loss += losses
        self.weights *= np.exp(-self.eta * losses)
        self.weights /= np.sum(self.weights)
        floor = (1.0 / self.K) * np.exp(-self.eta * self.cum_loss)
        assert np.all(self.weights + 1e-12 >= floor)
        return self.weights.copy()

    def ordered(self, names: Optional[Sequence[str]] = None) -> list:
        """Expert order by descending weight (Hedge prior). If names given,
        returns indices into `names` sorted by weight; else weight argsort."""
        order = np.argsort(-self.weights)
        if names is None:
            return order.tolist()
        return [i for i in order if i < len(names)]


@dataclass
class AuditStream:
    lo: int
    hi: int
    cursor: int


@dataclass
class AuditLedgerV9:
    """C264/C274 AuditLedger: disjoint streams, alpha spending, floors, final_test_untouched."""

    delta_total: float = 0.05
    spent: dict[str, float] = field(default_factory=dict)
    streams: dict[str, AuditStream] = field(default_factory=dict)
    floors: dict[str, float] = field(default_factory=dict)
    rejected_tasks: set[str] = field(default_factory=set)
    final_test_lo: int = 10_000_000
    final_test_hi: int = 20_000_000
    final_test_touched: bool = False
    next_cursor_base: int = 0

    def remaining(self) -> float:
        return max(0.0, self.delta_total - sum(self.spent.values()))

    def allocate_stream(self, task: str, n_samples: int) -> AuditStream:
        if task in self.streams:
            return self.streams[task]
        lo = self.next_cursor_base
        hi = lo + max(1, int(n_samples))
        assert (
            hi <= self.final_test_lo
        ), "audit stream overlapped held-out final test range"
        st = AuditStream(lo=lo, hi=hi, cursor=lo)
        self.streams[task] = st
        self.next_cursor_base = hi
        return st

    def advance_stream(self, task: str, count: int) -> None:
        st = self.streams[task]
        st.cursor = min(st.hi, st.cursor + max(0, int(count)))

    def draw_disjoint_splits(
        self,
        task: str,
        domain: list,
        seed: int = 0,
        train_frac: float = 0.35,
        cegis_frac: float = 0.43,
    ) -> tuple[list, list, list]:
        """Split a domain into disjoint train / cegis / val streams from
        ONE allocated audit stream (no overlap by construction; cursor
        consumes the whole stream). Small domains degrade honestly:
        tiny -> almost everything to train (no cegis budget)."""
        n = len(domain)
        st = self.allocate_stream(task, n)
        rng = np.random.default_rng(seed + (sum(map(ord, task)) % 10000))
        perm = rng.permutation(n)
        if n <= 8:
            n_tr, n_ceg = max(1, n - 1), 0
        elif n <= 16:
            n_tr, n_ceg = max(2, n - 3), 1
        else:
            n_tr = max(6, int(round(n * train_frac)))
            n_ceg = max(4, int(round(n * cegis_frac)))
            if n_tr + n_ceg >= n:
                n_ceg = max(1, (n - n_tr) // 2)
        tr = [domain[int(i)] for i in perm[:n_tr]]
        ceg = [domain[int(i)] for i in perm[n_tr:n_tr + n_ceg]]
        val = [domain[int(i)] for i in perm[n_tr + n_ceg:]]
        st.cursor = min(st.hi, st.lo + len(tr) + len(ceg) + len(val))
        return tr, ceg, val

    @staticmethod
    def hoeffding_radius(alpha: float, n: int) -> float:
        if n <= 0 or alpha <= 0.0:
            return 1.0
        return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))

    def spend_anytime(
        self, task: str, alpha: float, n_samples: int
    ) -> tuple[bool, float]:
        if alpha < 0 or alpha > self.remaining() + 1e-12:
            return False, 1.0
        self.spent[task] = self.spent.get(task, 0.0) + alpha
        self.advance_stream(task, n_samples)
        return True, self.hoeffding_radius(alpha, n_samples)

    def record_floor(self, task: str, quality: float) -> bool:
        prev = self.floors.get(task, -1e18)
        if quality + 1e-12 < prev:
            return False
        self.floors[task] = max(prev, quality)
        return True

    def record_reject(self, task: str) -> None:
        self.rejected_tasks.add(task)
        if task in self.streams:
            st = self.streams[task]
            st.cursor = st.hi

    def verify_disjoint_and_clean(self) -> bool:
        if self.final_test_touched:
            return False
        intervals = sorted((st.lo, st.hi) for st in self.streams.values())
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                return False
        for lo, hi in intervals:
            if not (hi <= self.final_test_lo or lo >= self.final_test_hi):
                return False
        return sum(self.spent.values()) <= self.delta_total + 1e-12


def ram_available_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys)
        except Exception:
            pass
    return 0


# ==============================================================================
# 9. Fast Exact Polynomial Induction, Piecewise Solver & Meet-in-the-Middle MDL
# ==============================================================================


def _exact_poly_fit(
    xs: Sequence[int], ys: Sequence[int], max_degree: int = 5
) -> Optional[tuple[int, ...]]:
    pts = sorted({(int(x), int(y)) for x, y in zip(xs, ys)})
    if len({x for x, _ in pts}) < 2:
        return None
    for degree in range(1, min(max_degree, len(pts) - 1) + 1):
        k = degree + 1
        sample = pts[:k]
        a = [[Fraction(x) ** i for i in range(k)] + [Fraction(y)] for x, y in sample]
        for col in range(k):
            piv = next((r for r in range(col, k) if a[r][col] != 0), None)
            if piv is None:
                break
            a[col], a[piv] = a[piv], a[col]
            pv = a[col][col]
            a[col] = [v / pv for v in a[col]]
            for r in range(k):
                if r != col and a[r][col] != 0:
                    f = a[r][col]
                    a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]
        else:
            coeffs = [row[k] for row in a]
            if any(c.denominator != 1 for c in coeffs):
                continue
            icoeffs = tuple(int(c) for c in coeffs)
            if all(
                sum(ci * (x**i) for i, ci in enumerate(icoeffs)) == y for x, y in pts
            ):
                return icoeffs
    return None


def _monomial_exponents(arity: int, max_degree: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            (
                powers
                for powers in itertools.product(range(max_degree + 1), repeat=arity)
                if sum(powers) <= max_degree
            ),
            key=lambda powers: (sum(powers), powers),
        )
    )


def _fast_mod_p_consistent(
    rows_int: list[tuple[int, ...]],
    ys_int: list[int],
    monomials: tuple[tuple[int, ...], ...],
) -> bool:
    """Fast exact O(k^3) modular filter mod 2^31-1 before exact Fraction solve."""
    P = _PRIME_MOD
    ncols = len(monomials)
    sub_n = min(len(rows_int), ncols + 8)
    mat = []
    for r_idx in range(sub_n):
        row = rows_int[r_idx]
        feats = [1] * (ncols + 1)
        for c, powers in enumerate(monomials):
            val = 1
            for x, p in zip(row, powers):
                if p:
                    val = (val * pow(x % P, p, P)) % P
            feats[c] = val
        feats[ncols] = ys_int[r_idx] % P
        mat.append(feats)
    piv_row = 0
    for col in range(ncols):
        piv = next((r for r in range(piv_row, sub_n) if mat[r][col] != 0), None)
        if piv is None:
            return True
        mat[piv_row], mat[piv] = mat[piv], mat[piv_row]
        inv = pow(mat[piv_row][col], P - 2, P)
        prow = [(v * inv) % P for v in mat[piv_row]]
        mat[piv_row] = prow
        for r in range(sub_n):
            if r != piv_row and mat[r][col] != 0:
                fac = mat[r][col]
                mr = mat[r]
                for c in range(col, ncols + 1):
                    mr[c] = (mr[c] - fac * prow[c]) % P
        piv_row += 1
    for r in range(piv_row, sub_n):
        if mat[r][ncols] != 0:
            return False
    return True


def _exact_multivariate_poly_fit(
    inputs: Sequence[Sequence[int]],
    targets: Sequence[int],
    arity: int,
    max_degree: int = 4,
) -> Optional[tuple[tuple[int, tuple[int, ...]], ...]]:
    if len(inputs) != len(targets) or not inputs or arity < 1:
        return None
    rows = [tuple(int(v) for v in row) for row in inputs]
    ys = [int(v) for v in targets]
    for degree in range(max_degree + 1):
        monomials = _monomial_exponents(arity, degree)
        ncols = len(monomials)
        if len(rows) < ncols:
            continue
        if not _fast_mod_p_consistent(rows, ys, monomials):
            continue
        sample_rows = rows[: min(len(rows), max(ncols + 12, int(ncols * 2)))]
        sample_ys = ys[: len(sample_rows)]
        matrix = []
        for row, y in zip(sample_rows, sample_ys):
            features = [
                Fraction(math.prod(x**p for x, p in zip(row, powers)))
                for powers in monomials
            ]
            matrix.append(features + [Fraction(y)])
        pivot_row = 0
        pivot_rows: list[int] = []
        full_rank = True
        for col in range(ncols):
            pivot = next(
                (r for r in range(pivot_row, len(matrix)) if matrix[r][col] != 0), None
            )
            if pivot is None:
                if len(sample_rows) < len(rows):
                    matrix = [
                        [
                            Fraction(math.prod(x**p for x, p in zip(r, powers)))
                            for powers in monomials
                        ]
                        + [Fraction(y)]
                        for r, y in zip(rows, ys)
                    ]
                    pivot_row = 0
                    pivot_rows = []
                    for col2 in range(ncols):
                        piv2 = next(
                            (
                                rr
                                for rr in range(pivot_row, len(matrix))
                                if matrix[rr][col2] != 0
                            ),
                            None,
                        )
                        if piv2 is None:
                            full_rank = False
                            break
                        matrix[pivot_row], matrix[piv2] = (
                            matrix[piv2],
                            matrix[pivot_row],
                        )
                        sc = matrix[pivot_row][col2]
                        matrix[pivot_row] = [v / sc for v in matrix[pivot_row]]
                        for rr in range(len(matrix)):
                            if rr != pivot_row and matrix[rr][col2]:
                                fac = matrix[rr][col2]
                                matrix[rr] = [
                                    a - fac * b
                                    for a, b in zip(matrix[rr], matrix[pivot_row])
                                ]
                        pivot_row += 1
                else:
                    full_rank = False
                break
            if pivot_row >= ncols:
                full_rank = False
                break
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
            scale = matrix[pivot_row][col]
            matrix[pivot_row] = [v / scale for v in matrix[pivot_row]]
            for r in range(len(matrix)):
                if r != pivot_row and matrix[r][col]:
                    fac = matrix[r][col]
                    matrix[r] = [
                        a - fac * b for a, b in zip(matrix[r], matrix[pivot_row])
                    ]
            pivot_rows.append(pivot_row)
            pivot_row += 1
        if not full_rank or len(pivot_rows) < ncols:
            continue
        coefficients = [matrix[pivot_rows[col]][-1] for col in range(ncols)]
        if any(v.denominator != 1 for v in coefficients):
            continue
        terms = tuple(
            (int(c), powers) for c, powers in zip(coefficients, monomials) if c
        )
        if all(
            sum(c * math.prod(x**p for x, p in zip(row, powers)) for c, powers in terms)
            == y
            for row, y in zip(rows, ys)
        ):
            return terms
    return None


def _fit_poly_or_const(pts: list[tuple[int, int]]) -> Optional[ExprV9]:
    ys = {y for _, y in pts}
    if len(ys) == 1:
        return ExprV9.const(next(iter(ys)))
    coeffs = _exact_poly_fit([x for x, _ in pts], [y for _, y in pts], max_degree=3)
    if coeffs is not None:
        return ExprV9("poly", (ExprV9.var(0),), coeffs)
    return None


def exact_piecewise_poly_fit(
    train_x: Sequence[tuple[int, ...]], train_y: Sequence[int]
) -> Optional[ExprV9]:
    """Exact rational piecewise polynomial / modular residue / generalized predicate discovery."""
    arity = len(train_x[0])
    if arity == 1:
        xs = [r[0] for r in train_x]
        for m in range(2, 9):
            buckets: dict[int, set[int]] = {}
            for x, y in zip(xs, train_y):
                buckets.setdefault(x % m, set()).add(y)
            if all(len(vs) == 1 for vs in buckets.values()) and len(buckets) >= 2:
                if m == 2 and buckets.get(0) == {0} and buckets.get(1) == {1}:
                    return ExprV9("bit_and", (ExprV9.var(0), ExprV9.const(1)))
        for thr in (-3, -2, -1, 0, 1, 2, 3):
            left = [(x, y) for x, y in zip(xs, train_y) if x < thr]
            right = [(x, y) for x, y in zip(xs, train_y) if x >= thr]
            if len(left) >= 3 and len(right) >= 3:
                cl = _fit_poly_or_const(left)
                cr = _fit_poly_or_const(right)
                if cl is not None and cr is not None and cl != cr:
                    return ExprV9(
                        "if_cond",
                        (ExprV9("lt", (ExprV9.var(0), ExprV9.const(thr))), cl, cr),
                    )
    elif 2 <= arity <= 3:
        preds: list[tuple[ExprV9, Callable[[tuple[int, ...]], bool]]] = [
            (ExprV9("lt", (ExprV9.var(0), ExprV9.var(1))), lambda r: r[0] < r[1]),
            (ExprV9("le", (ExprV9.var(0), ExprV9.var(1))), lambda r: r[0] <= r[1]),
            (ExprV9("eq", (ExprV9.var(0), ExprV9.var(1))), lambda r: r[0] == r[1]),
            (
                ExprV9("bit_and", (ExprV9.var(0), ExprV9.const(1))),
                lambda r: (r[0] & 1) != 0,
            ),
        ]
        if arity == 2:
            preds.append(
                (ExprV9("lt", (ExprV9.var(0), ExprV9.const(0))), lambda r: r[0] < 0)
            )
            preds.append(
                (ExprV9("lt", (ExprV9.var(1), ExprV9.const(0))), lambda r: r[1] < 0)
            )
        for cond_expr, fn in preds:
            true_pts = [(r, y) for r, y in zip(train_x, train_y) if fn(r)]
            false_pts = [(r, y) for r, y in zip(train_x, train_y) if not fn(r)]
            if len(true_pts) < 6 or len(false_pts) < 6:
                continue
            tl = _exact_multivariate_poly_fit(
                [r for r, _ in true_pts], [y for _, y in true_pts], arity, max_degree=2
            )
            tr = _exact_multivariate_poly_fit(
                [r for r, _ in false_pts],
                [y for _, y in false_pts],
                arity,
                max_degree=2,
            )
            if tl is not None and tr is not None and tl != tr:
                eb1 = (
                    ExprV9.const(tl[0][0])
                    if (len(tl) == 1 and sum(tl[0][1]) == 0)
                    else ExprV9("poly_nd", (), tl)
                )
                eb2 = (
                    ExprV9.const(tr[0][0])
                    if (len(tr) == 1 and sum(tr[0][1]) == 0)
                    else ExprV9("poly_nd", (), tr)
                )
                return ExprV9("if_cond", (cond_expr, eb1, eb2))
    return None


def synthesize_batched_mdl_v9(
    arity: int,
    train_x: Sequence[tuple[int, ...]],
    train_y: Sequence[int],
    val_x: Sequence[tuple[int, ...]],
    val_y: Sequence[int],
    library_experts: Sequence[tuple[int, str, "ExpertV9"]] = (),
    max_evals: int = 20000,  # WAVE 170: deeper search for the wider budget
    max_bits: int = 128,   # WAVE 170 (user decision): MDL budget 92 -> 128
    deadline: Optional[float] = None,
) -> tuple[Optional[ExprV9], int]:
    """Vectorized Batched MDL + O(1) Meet-in-the-Middle + Call-DAG Synthesis."""
    X = np.asarray(train_x, dtype=np.int64)
    Y = np.asarray(train_y, dtype=np.int64)
    if X.size and int(np.max(np.abs(X))) > 2048:
        # input magnitude cap: byte pool (0..255) and cyrillic codepoints
        # (0x400..) must pass; intermediate-value safety stays with MAG_CAP
        return None, 0
    N = X.shape[0]
    leaf_exprs = [ExprV9.var(i) for i in range(arity)] + [
        ExprV9.const(c) for c in CONSTS
    ]
    L_arr = np.stack(
        [
            (
                np.ascontiguousarray(X[:, int(lf.value or 0)])
                if lf.op == "var"
                else np.full(N, int(lf.value), dtype=np.int64)
            )
            for lf in leaf_exprs
        ]
    )
    L_bits = [lf.bits() for lf in leaf_exprs]

    env = {name: ex for _, name, ex in library_experts}
    ex_idx = {name: idx for idx, name, _ in library_experts}

    for i, lf in enumerate(leaf_exprs):
        if np.array_equal(L_arr[i], Y):
            if all(lf.eval(r, env) == y for r, y in zip(val_x, val_y)):
                return lf, 1

    is_bool_task = (
        (X.min() >= 0 and X.max() <= 1 and Y.min() >= 0 and Y.max() <= 1)
        if N > 0
        else False
    )
    useful_libs: list[tuple[int, str, ExpertV9]] = []
    for idx, nm, ex in library_experts:
        if ex.program.arity > arity or ex.program.expr is None:
            continue
        if not is_bool_task and ex.family == "BoolVenn":
            continue
        if (
            ex.program.arity == 2
            and ex.program.expr.op in OPS
            and all(a.op == "var" for a in ex.program.expr.args)
        ):
            continue
        useful_libs.append((idx, nm, ex))
    useful_libs.sort(key=lambda it: (it[2].program.cost, -it[0]))
    useful_libs = useful_libs[:32]

    seen_sigs: dict[bytes, int] = {}
    pool_hash: dict[bytes, ExprV9] = {}
    pool_exprs: list[ExprV9] = []
    pool_vals: list[np.ndarray] = []
    buckets: dict[int, list[tuple[ExprV9, np.ndarray]]] = {}

    for i, lf in enumerate(leaf_exprs):
        key = L_arr[i].tobytes()
        seen_sigs[key] = 1
        pool_hash[key] = lf
        pool_exprs.append(lf)
        pool_vals.append(L_arr[i])
        buckets.setdefault(L_bits[i], []).append((lf, L_arr[i]))

    total_checked = len(leaf_exprs)
    MAG_CAP = 1 << 28

    def verify_val(cand: ExprV9) -> bool:
        try:
            return all(cand.eval(r, env) == y for r, y in zip(val_x, val_y))
        except Exception:
            return False

    while buckets and total_checked < max_evals:
        if deadline is not None and time.monotonic() >= deadline:
            break
        min_b = min(buckets)
        if min_b >= max_bits:
            break
        batch = buckets.pop(min_b)
        if not batch:
            continue
        if len(batch) > 512:
            batch = batch[:512]
        B_exprs = [x[0] for x in batch]
        V = np.stack([x[1] for x in batch])

        def register(
            res_arr: np.ndarray, builder_fn: Callable[[int, int], ExprV9], new_bits: int
        ) -> Optional[ExprV9]:
            nonlocal total_checked
            if new_bits > max_bits:
                return None
            if res_arr.ndim == 2:
                res_arr = res_arr[:, None, :]
            matches = np.all(res_arr == Y[None, None, :], axis=2)
            if np.any(matches):
                for bi, ki in np.argwhere(matches):
                    cand = builder_fn(int(bi), int(ki))
                    if verify_val(cand):
                        return cand
            if new_bits + TOKEN_BITS > max_bits or total_checked >= max_evals:
                return None
            ok_mag = np.max(np.abs(res_arr), axis=2) <= MAG_CAP
            for bi, ki in np.argwhere(ok_mag):
                if total_checked >= max_evals:
                    break
                row = res_arr[bi, ki]
                key = row.tobytes()
                cnt = seen_sigs.get(key, 0)
                if cnt >= 2:
                    continue
                expr = builder_fn(int(bi), int(ki))
                seen_sigs[key] = cnt + 1
                if key not in pool_hash:
                    pool_hash[key] = expr
                    if len(pool_exprs) < 3500:
                        pool_exprs.append(expr)
                        pool_vals.append(row)
                total_checked += 1
                buckets.setdefault(new_bits, []).append((expr, row))
            return None

        V_sq_ok = np.where(np.abs(V) <= 32768, V * V, MAG_CAP + 1)
        V_cb_ok = np.where(np.abs(V) <= 1024, V * V * V, MAG_CAP + 1)
        for uop, ures in (
            ("neg", -V),
            ("abs", np.abs(V)),
            ("sign", np.sign(V).astype(np.int64)),
            ("bit_not8", (~V) & 0xFF),
            ("clamp3", np.clip(V, -3, 3)),
            ("square", V_sq_ok),
            ("cube", V_cb_ok),
        ):
            hit = register(
                ures, lambda bi, _, o=uop: ExprV9(o, (B_exprs[bi],)), min_b + TOKEN_BITS
            )
            if hit is not None:
                return hit, total_checked

        B_len = V.shape[0]
        non_const_mask = np.array([e.op != "const" for e in B_exprs], dtype=bool)
        V_min = np.min(V, axis=1)
        V_max = np.max(V, axis=1)
        for lib_idx, ename, ex in useful_libs:
            cbits = TOKEN_BITS + gamma_bits(lib_idx + 1)
            if ex.program.arity == 1:
                if min_b + cbits > max_bits:
                    continue
                lo_d, hi_d = ex.domains[0][0], ex.domains[0][-1]
                cand_mask = non_const_mask & (V_min >= lo_d) & (V_max <= hi_d)
                ok_idxs = np.flatnonzero(cand_mask)
                if ok_idxs.size > 0:
                    flat_in = V[ok_idxs].reshape(-1, 1)
                    res_flat, ok_flat = eval_expr_vec(ex.program.expr, flat_in, env)
                    res_2d = res_flat.reshape(ok_idxs.size, N)
                    ok_row_mask = np.all(ok_flat.reshape(ok_idxs.size, N), axis=1)
                    if np.any(ok_row_mask):
                        final_idxs = ok_idxs[ok_row_mask]
                        hit = register(
                            res_2d[ok_row_mask],
                            lambda bi, _, o=ename, idxs=final_idxs: ExprV9(
                                "call", (B_exprs[int(idxs[bi])],), o
                            ),
                            min_b + cbits,
                        )
                        if hit is not None:
                            return hit, total_checked
            elif ex.program.arity == 2:
                d0_lo, d0_hi = ex.domains[0][0], ex.domains[0][-1]
                d1_lo, d1_hi = ex.domains[1][0], ex.domains[1][-1]
                for li, lf in enumerate(leaf_exprs):
                    if min_b + cbits + L_bits[li] > max_bits:
                        continue
                    lv = L_arr[li]
                    lv_min, lv_max = int(np.min(lv)), int(np.max(lv))
                    base_mask = (
                        non_const_mask
                        if lf.op == "const"
                        else np.ones(B_len, dtype=bool)
                    )
                    for order_idx in (0, 1):
                        if order_idx == 0:
                            if (
                                lv_min < d1_lo
                                or lv_max > d1_hi
                                or (ex.valid_rule == "nonzero_y" and np.any(lv == 0))
                            ):
                                continue
                            cand_mask = base_mask & (V_min >= d0_lo) & (V_max <= d0_hi)
                        else:
                            if lv_min < d0_lo or lv_max > d0_hi:
                                continue
                            cand_mask = base_mask & (V_min >= d1_lo) & (V_max <= d1_hi)
                            if ex.valid_rule == "nonzero_y":
                                cand_mask &= np.all(V != 0, axis=1)
                        ok_idxs = np.flatnonzero(cand_mask)
                        if ok_idxs.size == 0:
                            continue
                        K_ok = ok_idxs.size
                        V_sel = V[ok_idxs]
                        L_rep = np.broadcast_to(lv[None, :], (K_ok, N))
                        flat_in = (
                            np.column_stack((V_sel.ravel(), L_rep.ravel()))
                            if order_idx == 0
                            else np.column_stack((L_rep.ravel(), V_sel.ravel()))
                        )
                        res_flat, ok_flat = eval_expr_vec(ex.program.expr, flat_in, env)
                        res_2d = res_flat.reshape(K_ok, N)
                        ok_row_mask = np.all(ok_flat.reshape(K_ok, N), axis=1)
                        if np.any(ok_row_mask):
                            final_idxs = ok_idxs[ok_row_mask]
                            hit = register(
                                res_2d[ok_row_mask],
                                (
                                    lambda bi, _, o=ename, l=lf, idxs=final_idxs, oi=order_idx: ExprV9(
                                        "call",
                                        (
                                            (B_exprs[int(idxs[bi])], l)
                                            if oi == 0
                                            else (l, B_exprs[int(idxs[bi])])
                                        ),
                                        o,
                                    )
                                ),
                                min_b + cbits + L_bits[li],
                            )
                            if hit is not None:
                                return hit, total_checked

        Va = V[:, None, :]
        Lb = L_arr[None, :, :]
        for op in BINARY_OPS:
            if op == "add":
                R = Va + Lb
            elif op == "sub":
                R = Va - Lb
            elif op == "mul":
                ok_m = (np.abs(Va) <= 32768) & (np.abs(Lb) <= 32768)
                sa = np.where(ok_m, Va, 0)
                sb = np.where(ok_m, Lb, 0)
                R = np.where(ok_m, sa * sb, MAG_CAP + 1)
            elif op == "min":
                R = np.minimum(Va, Lb)
            elif op == "max":
                R = np.maximum(Va, Lb)
            elif op == "eq":
                R = (Va == Lb).astype(np.int64)
            elif op == "ne":
                R = (Va != Lb).astype(np.int64)
            elif op == "lt":
                R = (Va < Lb).astype(np.int64)
            elif op == "le":
                R = (Va <= Lb).astype(np.int64)
            elif op == "gt":
                R = (Va > Lb).astype(np.int64)
            elif op == "ge":
                R = (Va >= Lb).astype(np.int64)
            elif op == "bit_and":
                R = Va & Lb
            elif op == "bit_or":
                R = Va | Lb
            elif op == "bit_xor":
                R = Va ^ Lb
            elif op == "bit_nand8":
                R = (~(Va & Lb)) & 0xFF
            elif op == "bit_nor8":
                R = (~(Va | Lb)) & 0xFF
            elif op == "bit_xnor8":
                R = (~(Va ^ Lb)) & 0xFF
            elif op == "floordiv":
                safe = np.where(Lb != 0, Lb, 1)
                R = np.where(
                    np.broadcast_to(Lb != 0, np.broadcast_shapes(Va.shape, Lb.shape)),
                    np.floor_divide(Va, safe),
                    MAG_CAP + 1,
                )
            elif op == "mod":
                safe = np.where(Lb != 0, Lb, 1)
                R = np.where(
                    np.broadcast_to(Lb != 0, np.broadcast_shapes(Va.shape, Lb.shape)),
                    np.mod(Va, safe),
                    MAG_CAP + 1,
                )
            elif op == "gcd":
                R = np.gcd(Va, Lb)
            elif op == "lcm":
                ok_l = (np.abs(Va) <= 10000) & (np.abs(Lb) <= 10000)
                sa = np.where(ok_l, Va, 0)
                sb = np.where(ok_l, Lb, 0)
                R = np.where(ok_l, np.lcm(sa, sb), MAG_CAP + 1)
            elif op == "pow_int":
                ok_p = (Lb >= 0) & (Lb <= 8) & (np.abs(Va) <= 32)
                sa = np.where(ok_p, Va, 0)
                sb = np.where(ok_p, Lb, 0)
                R = np.where(ok_p, sa**sb, MAG_CAP + 1)
            else:
                continue
            for li in range(len(leaf_exprs)):
                hit = register(
                    R[:, li : li + 1, :],
                    lambda bi, _, o=op, l=leaf_exprs[li]: ExprV9(o, (B_exprs[bi], l)),
                    min_b + TOKEN_BITS + L_bits[li],
                )
                if hit is not None:
                    return hit, total_checked
            if op not in COMMUTATIVE_OPS:
                if op == "sub":
                    R2 = Lb - Va
                elif op == "lt":
                    R2 = (Lb < Va).astype(np.int64)
                elif op == "le":
                    R2 = (Lb <= Va).astype(np.int64)
                elif op == "gt":
                    R2 = (Lb > Va).astype(np.int64)
                elif op == "ge":
                    R2 = (Lb >= Va).astype(np.int64)
                elif op == "floordiv":
                    safe = np.where(Va != 0, Va, 1)
                    R2 = np.where(
                        np.broadcast_to(
                            Va != 0, np.broadcast_shapes(Va.shape, Lb.shape)
                        ),
                        np.floor_divide(Lb, safe),
                        MAG_CAP + 1,
                    )
                elif op == "mod":
                    safe = np.where(Va != 0, Va, 1)
                    R2 = np.where(
                        np.broadcast_to(
                            Va != 0, np.broadcast_shapes(Va.shape, Lb.shape)
                        ),
                        np.mod(Lb, safe),
                        MAG_CAP + 1,
                    )
                elif op == "pow_int":
                    ok_p = (Va >= 0) & (Va <= 8) & (np.abs(Lb) <= 32)
                    sa = np.where(ok_p, Lb, 0)
                    sb = np.where(ok_p, Va, 0)
                    R2 = np.where(ok_p, sa**sb, MAG_CAP + 1)
                else:
                    R2 = None
                if R2 is not None:
                    for li in range(len(leaf_exprs)):
                        hit = register(
                            R2[:, li : li + 1, :],
                            lambda bi, _, o=op, l=leaf_exprs[li]: ExprV9(
                                o, (l, B_exprs[bi])
                            ),
                            min_b + TOKEN_BITS + L_bits[li],
                        )
                        if hit is not None:
                            return hit, total_checked

        if len(pool_vals) >= 4 and min_b >= 19:
            P_mat = np.stack(pool_vals)
            mitm_hits: list[tuple[int, ExprV9]] = []
            for op_name, mat, rev in (
                ("add", Y[None, :] - P_mat, False),
                ("sub", P_mat - Y[None, :], False),
                ("sub", P_mat + Y[None, :], True),
                ("bit_xor", Y[None, :] ^ P_mat, False),
            ):
                for idx_p in range(mat.shape[0]):
                    k = mat[idx_p].tobytes()
                    if k in pool_hash:
                        e1 = pool_exprs[idx_p]
                        e2 = pool_hash[k]
                        cand = ExprV9(op_name, (e2, e1) if rev else (e1, e2))
                        cb = cand.bits(ex_idx)
                        if cb <= max_bits and verify_val(cand):
                            mitm_hits.append((cb, cand))
            nz_mask = np.all(P_mat != 0, axis=1)
            if np.any(nz_mask):
                P_nz = P_mat[nz_mask]
                nz_idxs = np.flatnonzero(nz_mask)
                div_ok = np.all((Y[None, :] % P_nz) == 0, axis=1)
                for loc_i in np.flatnonzero(div_ok):
                    quot = Y // P_nz[loc_i]
                    k = quot.tobytes()
                    if k in pool_hash:
                        e1 = pool_exprs[int(nz_idxs[loc_i])]
                        e2 = pool_hash[k]
                        cand = ExprV9("mul", (e1, e2))
                        cb = cand.bits(ex_idx)
                        if cb <= max_bits and verify_val(cand):
                            mitm_hits.append((cb, cand))
            if mitm_hits:
                mitm_hits.sort(key=lambda it: it[0])
                return mitm_hits[0][1], total_checked

    if len(pool_vals) >= 2:
        P_mat = np.stack(pool_vals[:500])
        ok_base = np.all(np.abs(P_mat) <= 32, axis=1)
        ok_exp = np.all((P_mat >= 0) & (P_mat <= 8), axis=1)
        if np.any(ok_base) and np.any(ok_exp):
            bases = P_mat[ok_base]
            b_idxs = np.flatnonzero(ok_base)
            exps = P_mat[ok_exp]
            e_idxs = np.flatnonzero(ok_exp)
            pw = bases[:, None, :] ** exps[None, :, :]
            hits = np.all(pw == Y[None, None, :], axis=2)
            if np.any(hits):
                bi, ei = np.argwhere(hits)[0]
                cand = ExprV9(
                    "pow_int",
                    (pool_exprs[int(b_idxs[bi])], pool_exprs[int(e_idxs[ei])]),
                )
                if verify_val(cand):
                    return cand, total_checked

    return None, total_checked


# ==============================================================================
# 10. ExpertV9, TheoremMinerV9, Structural Call-DAG Unification & Processor
# ==============================================================================


@dataclass
class ExpertV9:
    name: str
    family: str
    program: ProgramV9
    domains: tuple[tuple[int, ...], ...]
    valid_rule: str
    code_length: int
    adapter: Optional[FactorizedAdapterV9] = None
    parents: tuple[str, ...] = ()

    def supports(self, xs: Sequence[int]) -> bool:
        if len(xs) != len(self.domains):
            return False
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):
            return False
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:
            return False
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "program": self.program.to_dict(),
            "domains": [list(d) for d in self.domains],
            "valid_rule": self.valid_rule,
            "code_length": self.code_length,
            "adapter": self.adapter.to_dict() if self.adapter else None,
            "parents": list(self.parents),
        }

    @staticmethod
    def from_dict(d: dict[str, Any], core: "FactorizedSharedCoreV9") -> "ExpertV9":
        return ExpertV9(
            name=str(d["name"]),
            family=str(d["family"]),
            program=ProgramV9.from_dict(d["program"]),
            domains=tuple(tuple(int(v) for v in dom) for dom in d["domains"]),
            valid_rule=str(d["valid_rule"]),
            code_length=int(d["code_length"]),
            adapter=(
                FactorizedAdapterV9.from_dict(d["adapter"], core)
                if d.get("adapter")
                else None
            ),
            parents=tuple(str(p) for p in d.get("parents", [])),
        )


@dataclass
class Theorem:
    name: str
    statement: str
    domain_points: int
    verified: bool = True


class TheoremMinerV9:
    """Exhaustive finite-domain theorem discovery across ALL experts (no truncation)."""

    def mine(self, model: "GrowingHagiProcessorV9") -> list[Theorem]:
        found: list[Theorem] = []
        known = {t.name for t in model.theorems}
        env = model.experts
        for e in model.experts.values():
            if e.program.arity == 2:
                dx, dy = e.domains[0], e.domains[1]
                declared = [(x, y) for x in dx for y in dy if e.supports((x, y))]
                swapped = [(x, y) for x, y in declared if e.supports((y, x))]
                if swapped and len(swapped) == len(declared):
                    nm = f"commutativity({e.name})"
                    if nm not in known:
                        try:
                            if all(
                                e.program.eval((x, y), env)
                                == e.program.eval((y, x), env)
                                for x, y in swapped
                            ):
                                th = Theorem(
                                    nm,
                                    f"for all supported pairs: {e.name}(x,y)={e.name}(y,x)",
                                    len(swapped),
                                )
                                model.theorems.append(th)
                                found.append(th)
                                known.add(nm)
                        except Exception:
                            pass
                self_pairs = [x for x in dx if x in dy and e.supports((x, x))]
                nm = f"self-absorption({e.name})"
                if self_pairs and len(self_pairs) == len(dx) and nm not in known:
                    try:
                        if all(e.program.eval((x, x), env) == x for x in self_pairs):
                            th = Theorem(
                                nm,
                                f"for all overlapping x: {e.name}(x,x)=x",
                                len(self_pairs),
                            )
                            model.theorems.append(th)
                            found.append(th)
                            known.add(nm)
                    except Exception:
                        pass
                zero_pairs = [x for x in dx if 0 in dy and e.supports((x, 0))]
                nm = f"right-identity-zero({e.name})"
                if zero_pairs and len(zero_pairs) == len(dx) and nm not in known:
                    try:
                        if all(e.program.eval((x, 0), env) == x for x in zero_pairs):
                            th = Theorem(
                                nm, f"for all x: {e.name}(x,0)=x", len(zero_pairs)
                            )
                            model.theorems.append(th)
                            found.append(th)
                            known.add(nm)
                    except Exception:
                        pass
            elif e.program.arity == 1:
                dom = tuple(int(x) for x in e.domains[0])
                img = {}
                closed = True
                try:
                    for x in dom:
                        y = int(e.program.eval((x,), env))
                        if not e.supports((y,)):
                            closed = False
                            break
                        img[x] = y
                except Exception:
                    closed = False
                if closed:
                    for prop, fn in (
                        (
                            "involutivity",
                            lambda: all(
                                e.program.eval((img[x],), env) == x for x in dom
                            ),
                        ),
                        (
                            "idempotence",
                            lambda: all(
                                e.program.eval((img[x],), env) == img[x] for x in dom
                            ),
                        ),
                    ):
                        nm = f"{prop}({e.name})"
                        if nm not in known:
                            try:
                                if fn():
                                    th = Theorem(
                                        nm, f"{prop} on certified domain", len(dom)
                                    )
                                    model.theorems.append(th)
                                    found.append(th)
                                    known.add(nm)
                            except Exception:
                                pass
                if all((-x) in dom for x in dom):
                    for sym_name, sgn in (("even-symmetry", 1), ("odd-symmetry", -1)):
                        nm = f"{sym_name}({e.name})"
                        if nm not in known:
                            try:
                                if all(
                                    e.program.eval((-x,), env)
                                    == sgn * e.program.eval((x,), env)
                                    for x in dom
                                ):
                                    th = Theorem(
                                        nm,
                                        f"{e.name}(-x)={'+' if sgn == 1 else '-'}{e.name}(x)",
                                        len(dom),
                                    )
                                    model.theorems.append(th)
                                    found.append(th)
                                    known.add(nm)
                            except Exception:
                                pass
        return found


def _match_pattern(
    pattern: ExprV9, target: ExprV9, env_bind: dict[int, ExprV9]
) -> bool:
    """Structural unification: match pattern AST (with var(i) holes) against target subtree."""
    if pattern.op == "var":
        idx = int(pattern.value or 0)
        if idx in env_bind:
            return env_bind[idx] == target
        env_bind[idx] = target
        return True
    if (
        pattern.op != target.op
        or pattern.value != target.value
        or len(pattern.args) != len(target.args)
    ):
        return False
    return all(
        _match_pattern(pa, ta, env_bind) for pa, ta in zip(pattern.args, target.args)
    )


def _dag_reaches(experts: dict[str, ExpertV9], start: str, target: str) -> bool:
    visited = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur == target:
            return True
        if cur in visited or cur not in experts:
            continue
        visited.add(cur)
        ex = experts[cur]
        if ex.program.expr is not None:
            stack.extend(ex.program.expr.callees())
    return False


class GrowingHagiProcessorV9:
    """Unified HAGI v9.0 Self-Developing Processor with Full Governance."""

    def __init__(
        self,
        seed: int = 42,
        node_budget: int = 8000,
        delta_total: float = 0.05,
        compute_budget: float = 1000.0,
        risk_limit: float = 10.0,
    ):
        self.core = FactorizedSharedCoreV9(seed=seed, width=48)
        self.node_budget = int(node_budget)
        self.experts: dict[str, ExpertV9] = {}
        self.version = 0
        self.total_commits = 0
        self.total_holds = 0
        self.growth_log: list[dict[str, Any]] = []
        self.cert_log: list[CertStepGen] = []
        # WAVE 171 (V5 port): serialization completeness sentinel - an
        # old save() that dropped governance history must not read as
        # a successful empty proof.
        self.governance_state_complete = True
        self.theorems: list[Theorem] = []
        self.ledger = AuditLedgerV9(delta_total=delta_total)
        self.router = HedgeRouterV9(len(FactorizedSharedCoreV9.BRANCHES), eta=0.5)
        self.ram = GDNRAMV9(n_slots=64, dim=8)
        self.compute_budget = float(compute_budget)
        self.risk_rem = float(risk_limit)
        self.unsolved_objective = 100.0
        self.evals = 0
        self.candidates_generated = 0

    def _expert_index(self) -> dict[str, int]:
        """Zipf frequency-ordered index: most-called experts get smallest gamma codes."""
        counts = {nm: 0 for nm in self.experts}
        for ex in self.experts.values():
            if ex.program.expr is not None:
                for c in ex.program.expr.callees():
                    if c in counts:
                        counts[c] += 1
        ordered = sorted(
            self.experts.keys(),
            key=lambda nm: (-counts[nm], self.experts[nm].code_length, nm),
        )
        return {nm: i for i, nm in enumerate(ordered)}

    def _snapshot_state_with_optimizer(self) -> dict[str, Any]:
        """C248 StateWithOptimizer: full snapshot of core + optimizer + router + RAM."""
        return {
            "core": self.core.snapshot(),
            "router": self.router.snapshot(),
            "ram": self.ram.snapshot(),
        }

    def _restore_state_with_optimizer(self, snap: dict[str, Any]) -> None:
        self.core.restore(snap["core"])
        self.router.restore(snap["router"])
        self.ram.restore(snap["ram"])

    def _record_ram_fingerprint(
        self, name: str, domain: list[tuple[int, ...]], ys: dict[tuple[int, ...], int]
    ) -> None:
        vals = [ys[r] for r in domain[:32]]
        chunks = boundary_chunks(lambda v: v == 0 or abs(v) > 4, vals)
        heads = boundary_heads(lambda v: v == 0 or abs(v) > 4, vals)
        sig = np.zeros(8, dtype=np.float64)
        sig[0] = float(len(domain))
        sig[1] = float(len(chunks))
        sig[2] = float(len(heads))
        for k in range(min(5, len(heads))):
            sig[3 + k] = float(heads[k])
        slot = sum(ord(c) for c in name) % 64
        self.ram.store(slot, sig)

    def learn(
        self,
        spec: TaskSpec,
        seed: int = 0,
        train_fraction: float = 0.70,
        poison_one_label: bool = False,
        train_neural: bool = True,
        deadline: Optional[float] = None,
    ) -> dict[str, Any]:
        if spec.name in self.experts:
            return {
                "version": self.version,
                "task": spec.name,
                "action": "REUSE",
                "reason": "already learned",
                "code_length": self.experts[spec.name].code_length,
            }
        domain = spec.inputs()
        if not domain:
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)

        opt_snap = self._snapshot_state_with_optimizer()

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(domain))
        if spec.family == "BoolVenn" or (
            all(set(d) <= {0, 1} for d in spec.domains) and len(domain) <= 16
        ):
            n_train = max(1, len(domain) - 1)
        else:
            n_train = max(
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))
            )
        train_x = [domain[int(i)] for i in perm[:n_train]]
        val_x = [domain[int(i)] for i in perm[n_train:]]
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}
        train_y = [all_y[xs] for xs in train_x]
        val_y = [all_y[xs] for xs in val_x]

        self.ledger.allocate_stream(spec.name, len(train_x) + len(val_x))

        if poison_one_label and train_y:
            self.ledger.record_reject(spec.name)
            self._restore_state_with_optimizer(opt_snap)
            return self._hold(
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)
            )

        alpha_step = min(0.0005, self.ledger.remaining() * 0.5)
        ok_alpha, _radius = self.ledger.spend_anytime(spec.name, alpha_step, len(val_x))
        if not ok_alpha:
            self._restore_state_with_optimizer(opt_snap)
            return self._hold(spec, "AUDIT_ALPHA_EXHAUSTED", len(train_x), len(val_x))

        adapter: Optional[FactorizedAdapterV9] = None
        if train_neural and val_x and spec.family != "BoolVenn":
            adapter = self.core.fit_residual(
                train_x, train_y, val_x, val_y, router=self.router
            )

        ex_idx = self._expert_index()

        for nm, ex in sorted(self.experts.items()):
            if ex.program.arity != spec.arity:
                continue
            if not all(ex.supports(xs) for xs in domain):
                continue
            self.evals += len(train_x)
            try:
                if any(
                    ex.program.eval(xs, self.experts) != yy
                    for xs, yy in zip(train_x, train_y)
                ):
                    continue
                if any(
                    ex.program.eval(xs, self.experts) != yy
                    for xs, yy in zip(val_x, val_y)
                ):
                    continue
                if any(ex.program.eval(xs, self.experts) != all_y[xs] for xs in domain):
                    continue
            except Exception:
                continue
            ref_bits = TOKEN_BITS + gamma_bits(ex_idx[nm] + 1)
            ev = {
                "version": self.version,
                "task": spec.name,
                "action": "REUSE",
                "reason": f"verified-expert-coverage:{nm}",
                "program": ex.program.pretty(),
                "code_length": ref_bits,
                "train_examples": len(train_x),
                "validation_examples": len(val_x),
                "exact_accuracy": 1.0,
            }
            self.growth_log.append(ev)
            self.ledger.record_floor(spec.name, 1.0)
            return ev

        def try_commit(
            program: ProgramV9, reason: str, parents: tuple[str, ...] = ()
        ) -> Optional[dict[str, Any]]:
            try:
                self.evals += len(train_x)
                if any(
                    program.eval(xs, self.experts) != yy
                    for xs, yy in zip(train_x, train_y)
                ):
                    return None
                if val_x and any(
                    program.eval(xs, self.experts) != yy for xs, yy in zip(val_x, val_y)
                ):
                    return None
                if any(program.eval(xs, self.experts) != all_y[xs] for xs in domain):
                    return None
            except Exception:
                return None
            bits = program.bits(ex_idx)
            expert = ExpertV9(
                spec.name,
                spec.family,
                program,
                spec.domains,
                spec.valid_rule,
                bits,
                adapter,
                parents,
            )
            self.experts[spec.name] = expert
            self.version += 1
            self.total_commits += 1
            self._record_ram_fingerprint(spec.name, domain, all_y)
            drop = 1.0
            # Exact CertStepGen telescope (Lean contract): obj_after must be
            # obj_before - drop with NO clamp — a floor at 0.0 breaks the
            # telescope invariant once commits exceed the initial nominal
            # budget (obj0=100) and verify_certificates() would fail forever
            # after. The objective may legitimately go negative (over-repaid).
            obj_after = self.unsolved_objective - drop
            b_spend = min(self.compute_budget, 0.5)
            r_spend = min(self.risk_rem, 0.01)
            self.compute_budget -= b_spend
            self.risk_rem -= r_spend
            floors_ok = self.ledger.record_floor(spec.name, 1.0)
            cert = CertStepGen(
                version=self.version,
                task=spec.name,
                action="COMMIT",
                obj_before=self.unsolved_objective,
                obj_after=obj_after,
                drop=drop,
                budget_rem=self.compute_budget,
                budget_spend=b_spend,
                risk_rem=self.risk_rem,
                risk_spend=r_spend,
                delta_rem=self.ledger.remaining(),
                delta_spend=alpha_step,
                floors_kept=floors_ok,
            )
            self.unsolved_objective = obj_after
            self.cert_log.append(cert)
            ev = {
                "version": self.version,
                "task": spec.name,
                "action": "COMMIT",
                "reason": reason,
                "program": program.pretty(),
                "code_length": bits,
                "train_examples": len(train_x),
                "validation_examples": len(val_x),
                "exact_accuracy": 1.0,
            }
            self.growth_log.append(ev)
            return ev

        if spec.family == "BoolVenn":
            for m in range(16):
                ev = try_commit(ProgramV9(2, venn_mask=m), "boolvenn-exact")
                if ev is not None:
                    return ev

        if all(set(d) <= {0, 1} for d in spec.domains) and 1 <= spec.arity <= 6:
            if set(train_y) <= {0, 1}:
                full_inputs = list(itertools.product((0, 1), repeat=spec.arity))
                train_map = dict(zip(train_x, train_y))
                missing = [r for r in full_inputs if r not in train_map]
                if len(missing) <= 1:
                    anf_cands = []
                    for fill in (0,) if not missing else (0, 1):
                        tt = [train_map.get(r, fill) for r in full_inputs]
                        coeffs = anf_transform(tt)
                        active_s = [s for s, c in enumerate(coeffs) if c]
                        max_deg = max((bin(s).count("1") for s in active_s), default=0)
                        cand_e = anf_to_expr_v9(coeffs, spec.arity)
                        cand_p = ProgramV9(spec.arity, cand_e)
                        score = (max_deg, len(active_s), cand_p.bits(ex_idx))
                        anf_cands.append((score, cand_p))
                    anf_cands.sort(key=lambda it: it[0])
                    for _, cand_p in anf_cands:
                        ev = try_commit(cand_p, "anf-zhegalkin-butterfly+full-support")
                        if ev is not None:
                            return ev

        for p in candidate_programs_v9(spec.arity):
            ev = try_commit(p, "direct-template+full-support")
            if ev is not None:
                return ev

        if 1 <= spec.arity <= 3 and not spec.family.startswith("Bool"):
            if spec.arity == 1:
                coeffs = _exact_poly_fit([r[0] for r in train_x], train_y, max_degree=5)
                if coeffs is not None:
                    ev = try_commit(
                        ProgramV9(1, ExprV9("poly", (ExprV9.var(0),), coeffs)),
                        "exact-rational-poly+full-support",
                    )
                    if ev is not None:
                        return ev
            else:
                terms = _exact_multivariate_poly_fit(
                    train_x, train_y, spec.arity, max_degree=4
                )
                if terms is not None:
                    ev = try_commit(
                        ProgramV9(spec.arity, ExprV9("poly_nd", (), terms)),
                        "exact-multivar-poly+full-support",
                    )
                    if ev is not None:
                        return ev

        pw = exact_piecewise_poly_fit(train_x, train_y)
        if pw is not None:
            ev = try_commit(
                ProgramV9(spec.arity, pw), "exact-piecewise-poly+full-support"
            )
            if ev is not None:
                return ev

        lib_list = [(ex_idx[nm], nm, self.experts[nm]) for nm in self.experts]
        hit_expr, n_chk = synthesize_batched_mdl_v9(
            spec.arity,
            train_x,
            train_y,
            val_x,
            val_y,
            library_experts=lib_list,
            max_evals=self.node_budget,
            max_bits=160,
            deadline=deadline,  # 64-bit headroom (gamma-coded)
        )
        self.candidates_generated += n_chk
        if hit_expr is not None:
            ev = try_commit(
                ProgramV9(spec.arity, hit_expr), "batched-mdl-mitm-dag+full-support"
            )
            if ev is not None:
                return ev

        self._restore_state_with_optimizer(opt_snap)
        self.ledger.record_reject(spec.name)
        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))

    def _hold(
        self, spec: TaskSpec, reason: str, ntrain: int, nval: int
    ) -> dict[str, Any]:
        self.total_holds += 1
        cert = CertStepGen(
            version=self.version,
            task=spec.name,
            action="HOLD",
            obj_before=self.unsolved_objective,
            obj_after=self.unsolved_objective,
            drop=0.0,
            budget_rem=self.compute_budget,
            budget_spend=0.0,
            risk_rem=self.risk_rem,
            risk_spend=0.0,
            delta_rem=self.ledger.remaining(),
            delta_spend=0.0,
            floors_kept=True,
        )
        self.cert_log.append(cert)
        ev = {
            "version": self.version,
            "task": spec.name,
            "action": "HOLD",
            "reason": reason,
            "program": "",
            "code_length": 0,
            "train_examples": ntrain,
            "validation_examples": nval,
            "exact_accuracy": 0.0,
        }
        self.growth_log.append(ev)
        return ev

    def compress_library_mdl(self) -> dict[str, Any]:
        """Global MDL Call-DAG Refactoring with structural pattern unification & Zipf re-indexing."""
        ex_idx = self._expert_index()
        for ex in self.experts.values():
            ex.code_length = ex.program.bits(ex_idx)
        before_bits = sum(e.code_length for e in self.experts.values())
        refactored = 0

        for target_name, target_ex in list(self.experts.items()):
            if target_ex.program.expr is None:
                continue
            cur_expr = target_ex.program.expr
            subs = [
                s
                for s in cur_expr.subtrees()
                if s.op not in ("var", "const", "call") and s.cost >= 2
            ]
            if not subs:
                continue
            domain = [
                xs
                for xs in itertools.product(*target_ex.domains)
                if target_ex.supports(xs)
            ]
            for sub in subs:
                for lib_name, lib_ex in list(self.experts.items()):
                    if lib_name == target_name or lib_ex.program.expr is None:
                        continue
                    if _dag_reaches(self.experts, lib_name, target_name):
                        continue
                    bindings: dict[int, ExprV9] = {}
                    if not _match_pattern(lib_ex.program.expr, sub, bindings):
                        continue
                    if any(i not in bindings for i in range(lib_ex.program.arity)):
                        continue
                    call_args = tuple(bindings[i] for i in range(lib_ex.program.arity))
                    call_e = ExprV9("call", call_args, lib_name)
                    if call_e.bits(ex_idx) >= sub.bits(ex_idx):
                        continue
                    cand_expr = cur_expr.replace_subtree(sub, call_e)
                    cand_prog = ProgramV9(target_ex.program.arity, cand_expr)
                    new_bits = cand_prog.bits(ex_idx)
                    if new_bits >= target_ex.code_length:
                        continue
                    try:
                        if all(
                            cand_prog.eval(xs, self.experts)
                            == target_ex.program.eval(xs, self.experts)
                            for xs in domain
                        ):
                            target_ex.program = cand_prog
                            target_ex.code_length = new_bits
                            cur_expr = cand_expr
                            refactored += 1
                    except Exception:
                        pass

        ex_idx = self._expert_index()
        for ex in self.experts.values():
            ex.code_length = ex.program.bits(ex_idx)
        after_bits = sum(e.code_length for e in self.experts.values())
        return {
            "before_bits": before_bits,
            "after_bits": after_bits,
            "refactored": refactored,
        }

    def predict(self, name: str, *xs: int, mode: str = "exact") -> dict[str, Any]:
        ex = self.experts.get(name)
        if ex is None:
            return {"status": "ABSTAIN", "reason": "unknown operation"}
        if not ex.supports(xs):
            return {"status": "ABSTAIN", "reason": "outside certified finite domain"}
        if mode == "exact":
            return {
                "status": "OK",
                "value": ex.program.eval(xs, self.experts),
                "source": "verified-symbolic-dag",
                "model_version": self.version,
            }
        if mode == "neural":
            if ex.adapter is None:
                if ex.program.venn_mask is not None:
                    return {
                        "status": "APPROXIMATE",
                        "value": float(venn_head_eval(ex.program.venn_mask, xs)),
                        "source": "venn-head-exact-ternary",
                        "model_version": self.version,
                    }
                return {"status": "ABSTAIN", "reason": "neural adapter unavailable"}
            return {
                "status": "APPROXIMATE",
                "value": ex.adapter.predict(self.core, xs),
                "source": "factorized-core-bitalloc-rvq",
                "model_version": self.version,
            }
        raise ValueError("mode must be exact or neural")

    def verify_all(self, specs: Sequence[TaskSpec]) -> dict[str, Any]:
        total = correct = 0
        for sp in specs:
            ex = self.experts.get(sp.name)
            if ex is None:
                continue
            rows = sp.inputs()
            ok = sum(
                int(ex.program.eval(xs, self.experts) == int(sp.teacher(*xs)))
                for xs in rows
            )
            total += len(rows)
            correct += ok
        return {
            "checked": total,
            "correct": correct,
            "accuracy": correct / max(1, total),
            "committed": len(self.experts),
        }

    def verify_certificates(self) -> dict[str, Any]:
        """CertStepGen telescope, HoldStep immutability, AuditLedger disjointness, HedgeRouter floor."""
        obj0 = self.cert_log[0].obj_before if self.cert_log else self.unsolved_objective
        total_drops = sum(c.drop for c in self.cert_log)
        telescope_ok = self.unsolved_objective <= obj0 - total_drops + 1e-9
        holds_ok = all(
            c.drop == 0.0 and c.obj_after == c.obj_before and c.budget_spend == 0.0
            for c in self.cert_log
            if c.action == "HOLD"
        )
        commits_ok = all(
            c.drop > 0.0
            and c.budget_rem >= -1e-12
            and c.risk_rem >= -1e-12
            and c.floors_kept
            for c in self.cert_log
            if c.action == "COMMIT"
        )
        ledger_ok = self.ledger.verify_disjoint_and_clean()
        # WAVE 171 (V5 port): history completeness — an emptied cert_log
        # must not read as a successful proof.
        state_complete = bool(getattr(self, "governance_state_complete", False))
        logged_commits = sum(c.action == "COMMIT" for c in self.cert_log)
        logged_holds = sum(c.action == "HOLD" for c in self.cert_log)
        history_ok = (
            logged_commits == self.total_commits
            and logged_holds == self.total_holds
            and len(self.cert_log) == self.total_commits + self.total_holds
        )
        all_ok = (telescope_ok and holds_ok and commits_ok and ledger_ok
                  and state_complete and history_ok)
        return {
            "telescope_ok": telescope_ok,
            "holds_ok": holds_ok,
            "commits_ok": commits_ok,
            "ledger_ok": ledger_ok,
            "state_complete": state_complete,
            "history_ok": history_ok,
            "all_ok": all_ok,
        }

    def total_neural_bytes(self) -> int:
        venn_bytes = (
            13 if any(e.family == "BoolVenn" for e in self.experts.values()) else 0
        )
        adapter_bytes = sum(
            e.adapter.compressed_bytes
            for e in self.experts.values()
            if e.adapter is not None
        )
        return self.core.core_bytes + venn_bytes + adapter_bytes

    def neural_quality_summary(self) -> dict[str, float]:
        accs = [
            e.adapter.val_exact_acc_quant
            for e in self.experts.values()
            if e.adapter is not None
        ]
        mses = [
            e.adapter.val_mse_quant
            for e in self.experts.values()
            if e.adapter is not None
        ]
        return {
            "num_adapters": len(accs),
            "mean_val_exact_acc": float(np.mean(accs)) if accs else 0.0,
            "mean_val_mse": float(np.mean(mses)) if mses else 0.0,
            "total_neural_bytes": self.total_neural_bytes(),
        }

    def save(self, path: str | Path) -> None:
        """Save experts AND runtime-governance state (WAVE 171, V5 port).

        Symbolic experts suffice for prediction, but not to reconstruct
        CertStepGen/AuditLedger proof history. format_version=2 preserves
        those records for faithful continuation of learning.
        """
        gpm = self.core.gpm.snapshot_optimizer()
        router_weights, router_loss = self.router.snapshot()
        ledger = self.ledger
        governance_state = {
            "schema": 1,
            "cert_log": [vars(c) for c in self.cert_log],
            "ledger": {
                "delta_total": ledger.delta_total,
                "spent": dict(ledger.spent),
                "streams": {k: vars(v) for k, v in sorted(ledger.streams.items())},
                "floors": dict(ledger.floors),
                "rejected_tasks": sorted(ledger.rejected_tasks),
                "final_test_lo": ledger.final_test_lo,
                "final_test_hi": ledger.final_test_hi,
                "final_test_touched": ledger.final_test_touched,
                "next_cursor_base": ledger.next_cursor_base,
            },
            "compute_budget": self.compute_budget,
            "risk_rem": self.risk_rem,
            "unsolved_objective": self.unsolved_objective,
            "evals": self.evals,
            "candidates_generated": self.candidates_generated,
            "core_gpm": {
                "W": gpm["W"].tolist(),
                "U_basis": gpm["U_basis"].tolist(),
                "ef_v": gpm["ef_v"].tolist(),
                "core_bytes": int(self.core.core_bytes),
            },
            "router": {
                "K": int(self.router.K),
                "eta": float(self.router.eta),
                "weights": router_weights.tolist(),
                "cum_loss": router_loss.tolist(),
            },
            "ram": self.ram.snapshot().tolist(),
        }
        data = {
            "format": "HAGI-V9",
            "format_version": 2,
            "version": V9_VERSION,
            "core_seed": self.core.seed,
            "core_width": self.core.width,
            "model_version": self.version,
            "node_budget": self.node_budget,
            "total_commits": self.total_commits,
            "total_holds": self.total_holds,
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},
            "theorems": [vars(t) for t in self.theorems],
            "growth_log": self.growth_log,
            "governance_state": governance_state,
            "governance_state_complete": bool(self.governance_state_complete),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False),
                     encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GrowingHagiProcessorV9":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "HAGI-V9":
            raise ValueError("not a HAGI-V9 model file")
        m = cls(
            seed=int(d.get("core_seed", 42)),
            node_budget=int(d.get("node_budget", 8000)),
        )
        m.version = int(d.get("model_version", 0))
        m.total_commits = int(d.get("total_commits", 0))
        m.total_holds = int(d.get("total_holds", 0))
        m.experts = {
            k: ExpertV9.from_dict(v, m.core) for k, v in d.get("experts", {}).items()
        }
        m.theorems = [Theorem(**t) for t in d.get("theorems", [])]
        m.growth_log = d.get("growth_log", [])
        gs = d.get("governance_state")
        if (d.get("governance_state_complete") is not True
                or not isinstance(gs, dict) or gs.get("schema") != 1):
            # Backwards-compatible prediction loading, but absent audit
            # history is NOT a successful empty proof.
            m.governance_state_complete = False
            return m
        ledger_data = gs.get("ledger")
        router_data = gs.get("router")
        core_data = gs.get("core_gpm")
        ram_data = gs.get("ram")
        if not all(isinstance(x, dict) for x in (ledger_data, router_data, core_data)):
            raise ValueError("malformed governance state")
        m.cert_log = [CertStepGen(**row) for row in gs.get("cert_log", [])]
        streams = {
            str(k): AuditStream(**v)
            for k, v in ledger_data.get("streams", {}).items()
        }
        m.ledger = AuditLedgerV9(
            delta_total=float(ledger_data["delta_total"]),
            spent={str(k): float(v) for k, v in ledger_data.get("spent", {}).items()},
            streams=streams,
            floors={str(k): float(v) for k, v in ledger_data.get("floors", {}).items()},
            rejected_tasks=set(str(x) for x in ledger_data.get("rejected_tasks", [])),
            final_test_lo=int(ledger_data.get("final_test_lo", 10_000_000)),
            final_test_hi=int(ledger_data.get("final_test_hi", 20_000_000)),
            final_test_touched=bool(ledger_data.get("final_test_touched", False)),
            next_cursor_base=int(ledger_data.get("next_cursor_base", 0)),
        )
        router_k = int(router_data["K"])
        if router_k != m.router.K:
            raise ValueError("router expert count mismatch")
        weights = np.asarray(router_data["weights"], dtype=np.float64)
        cum_loss = np.asarray(router_data["cum_loss"], dtype=np.float64)
        if weights.shape != m.router.weights.shape or cum_loss.shape != m.router.cum_loss.shape:
            raise ValueError("router state shape mismatch")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(cum_loss)):
            raise ValueError("non-finite router state")
        m.router.eta = float(router_data["eta"])
        m.router.restore((weights, cum_loss))
        ram = np.asarray(ram_data, dtype=np.float64)
        if ram.shape != m.ram.mem.shape or not np.all(np.isfinite(ram)):
            raise ValueError("RAM state shape/non-finite mismatch")
        m.ram.restore(ram)
        gpm_snapshot = {
            "W": np.asarray(core_data["W"], dtype=np.float64),
            "U_basis": np.asarray(core_data["U_basis"], dtype=np.float64),
            "ef_v": np.asarray(core_data["ef_v"], dtype=np.float64),
        }
        expected_gpm = m.core.gpm.snapshot_optimizer()
        for key in ("W", "ef_v"):
            if gpm_snapshot[key].shape != expected_gpm[key].shape:
                raise ValueError(f"core GPM state shape mismatch: {key}")
            if not np.all(np.isfinite(gpm_snapshot[key])):
                raise ValueError(f"non-finite core GPM state: {key}")
        # U_basis is dynamic-width (consolidation appends directions).
        u_basis = gpm_snapshot["U_basis"]
        if (u_basis.ndim != 2 or u_basis.shape[0] != expected_gpm["U_basis"].shape[0]
                or u_basis.shape[1] > u_basis.shape[0]):
            raise ValueError("core GPM U_basis shape mismatch")
        if not np.all(np.isfinite(u_basis)):
            raise ValueError("non-finite core GPM state: U_basis")
        if u_basis.shape[1]:
            gram = u_basis.T @ u_basis
            if not np.allclose(gram, np.eye(u_basis.shape[1]), atol=1e-7, rtol=0):
                raise ValueError("core GPM U_basis is not orthonormal")
        m.core.gpm.restore_optimizer(gpm_snapshot)
        m.core.core_bytes = int(core_data.get("core_bytes", m.core.core_bytes))
        m.compute_budget = float(gs["compute_budget"])
        m.risk_rem = float(gs["risk_rem"])
        m.unsolved_objective = float(gs["unsolved_objective"])
        m.evals = int(gs.get("evals", 0))
        m.candidates_generated = int(gs.get("candidates_generated", 0))
        m.governance_state_complete = True
        return m


# ==============================================================================
# 11. Non-Circular Diverse Self-Developing Curriculum & Growth Loop
# ==============================================================================


def propose_noncircular_task_v9(
    experts: dict[str, ExpertV9],
    rng: np.random.Generator,
    round_idx: int,
    seen_sigs: Optional[set[tuple[Any, ...]]] = None,
    seen_sigs_names: Optional[set[str]] = None,
) -> Optional[tuple[TaskSpec, tuple[str, ...]]]:
    """Propose diverse, non-degenerate, novel composed tasks WITHOUT leaking ASTs to learn()."""
    seen_sigs_names = seen_sigs_names if seen_sigs_names is not None else set()
    arith2 = [
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"
    ]
    arith1 = [
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"
    ]
    bool2 = [
        e for e in experts.values() if e.program.arity == 2 and e.family == "BoolVenn"
    ]
    if not arith2:
        return None
    pool = arith2 + arith1
    for attempt in range(32):
        mode = int(rng.integers(0, 4))
        if mode == 3 and len(bool2) >= 2:
            f_ex = bool2[int(rng.choice(len(bool2)))]
            g_ex = bool2[int(rng.choice(len(bool2)))]
            h_ex = bool2[int(rng.choice(len(bool2)))]
            teacher = (
                lambda x, y, z, f=f_ex, g=g_ex, h=h_ex, env=experts: f.program.eval(
                    (
                        g.program.eval((int(x), int(y)), env),
                        h.program.eval((int(y), int(z)), env),
                    ),
                    env,
                )
            )
            vals = {teacher(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)}
            if len(vals) <= 1:
                continue
            sig = (
                "b3",
                tuple(teacher(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)),
            )
            if seen_sigs is not None and sig in seen_sigs:
                continue
            if seen_sigs is not None:
                seen_sigs.add(sig)
            sp = TaskSpec(
                f"V9_BOOL3_R{round_idx}_{attempt}",
                3,
                ((0, 1), (0, 1), (0, 1)),
                teacher,
                family="auto-bool3",
            )
            return sp, (f_ex.name, g_ex.name, h_ex.name)

        outer = arith2[int(rng.choice(len(arith2)))]
        inner = pool[int(rng.choice(len(pool)))]
        if inner.program.arity == 1:
            xs = tuple(
                x
                for x in inner.domains[0]
                if abs(x) <= 6
                and inner.supports((x,))
                and inner.program.eval((x,), experts) in outer.domains[0]
            )
            ys = tuple(
                y
                for y in outer.domains[1]
                if abs(y) <= 6
                and all(
                    outer.supports((inner.program.eval((x,), experts), y)) for x in xs
                )
            )
            if len(xs) < 3 or len(ys) < 3:
                continue
            tt = tuple(
                (
                    x,
                    y,
                    outer.program.eval((inner.program.eval((x,), experts), y), experts),
                )
                for x in xs
                for y in ys
            )
            vals = {v for _, _, v in tt}
            if (
                len(vals) <= 2
                or all(v == x for x, _, v in tt)
                or all(v == y for _, y, v in tt)
            ):
                continue
            sig = ("a2", xs, ys, tt)
            if seen_sigs is not None and sig in seen_sigs:
                continue
            if seen_sigs is not None:
                seen_sigs.add(sig)
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(
                (i.program.eval((int(x),), env), int(y)), env
            )
            sp = TaskSpec(
                _short_auto_name(
                    round_idx, attempt, inner.name, outer.name, seen_sigs_names
                ),
                2,
                (xs, ys),
                teacher,
                valid_rule=outer.valid_rule,
                family="auto-composed",
            )
            return sp, (outer.name, inner.name)
        else:
            xs = tuple(
                x for x in inner.domains[0] if abs(x) <= 5 and x in outer.domains[0]
            )
            ys_cand = tuple(y for y in inner.domains[1] if abs(y) <= 5)
            ys = tuple(
                y
                for y in ys_cand
                if all(
                    inner.supports((x, y))
                    and outer.supports((x, inner.program.eval((x, y), experts)))
                    for x in xs
                )
            )
            if len(xs) < 3 or len(ys) < 3:
                continue
            tt = tuple(
                (
                    x,
                    y,
                    outer.program.eval(
                        (x, inner.program.eval((x, y), experts)), experts
                    ),
                )
                for x in xs
                for y in ys
            )
            vals = {v for _, _, v in tt}
            if (
                len(vals) <= 2
                or all(v == x for x, _, v in tt)
                or all(v == y for _, y, v in tt)
            ):
                continue
            sig = ("a2", xs, ys, tt)
            if seen_sigs is not None and sig in seen_sigs:
                continue
            if seen_sigs is not None:
                seen_sigs.add(sig)
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(
                (int(x), i.program.eval((int(x), int(y)), env)), env
            )
            sp = TaskSpec(
                _short_auto_name(
                    round_idx, attempt, inner.name, outer.name, seen_sigs_names
                ),
                2,
                (xs, ys),
                teacher,
                valid_rule=inner.valid_rule,
                family="auto-composed",
            )
            return sp, (outer.name, inner.name)
    return None


def _short_auto_name(
    round_idx: int, attempt: int, inner: str, outer: str, seen: set | None = None
) -> str:
    """Short unique name for auto-composed tasks: strip V9_AUTO_
    boilerplate, keep the semantic core, dedupe with a counter."""

    def core(n: str) -> str:
        for part in ("V9_AUTO_", "V9_BOOL3_"):
            while n.startswith(part):
                n = n[len(part) :]
        # drop R<digits>_ and stray round/attempt index segments
        import re as _re

        n = _re.sub(r"R\d+_", "", n)
        n = _re.sub(r"^\d+_", "", n)
        n = _re.sub(r"_\d+(_\d+)$", r"\1", n)
        return n

    base = f"{core(inner)}_{core(outer)}"[:48]
    name = base
    k = 2
    while seen is not None and name in seen:
        name = f"{base}_{k}"
        k += 1
    if seen is not None:
        seen.add(name)
    return name


def grow_v9(
    model: GrowingHagiProcessorV9,
    suite: Sequence[TaskSpec],
    max_rounds: int = 3,
    train_neural: bool = True,
    out_dir: str = ".omc/v9_grow",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rnd = np.random.default_rng(20260101)
    seen_sigs: set[tuple[Any, ...]] = set()
    rounds = []
    for r in range(max_rounds):
        t0 = time.time()
        commits = 0
        for sp in suite:
            ev = model.learn(
                sp, seed=10_000 + r * 7919 + len(sp.name), train_neural=train_neural
            )
            if ev["action"] == "COMMIT":
                commits += 1
        proposed = 0
        for k in range(8):
            prop = propose_noncircular_task_v9(model.experts, rnd, r, seen_sigs)
            if prop is None:
                continue
            sp, _ = prop
            if sp.name in model.experts:
                continue
            proposed += 1
            ev = model.learn(
                sp, seed=500_000 + r * 104_729 + k, train_neural=train_neural
            )
            if ev["action"] == "COMMIT":
                commits += 1
        comp = model.compress_library_mdl()
        new_th = TheoremMinerV9().mine(model)
        nq = model.neural_quality_summary()
        rounds.append(
            {
                "round": r,
                "experts": len(model.experts),
                "commits": commits,
                "proposed": proposed,
                "theorems_total": len(model.theorems),
                "theorems_new": len(new_th),
                "sym_bits": comp["after_bits"],
                "dag_refactored": comp["refactored"],
                "neural_bytes": nq["total_neural_bytes"],
                "neural_val_acc": round(nq["mean_val_exact_acc"], 4),
                "neural_val_mse": round(nq["mean_val_mse"], 4),
                "certs_ok": model.verify_certificates()["all_ok"],
                "ram_avail_gb": round(ram_available_bytes() / 2**30, 2),
                "elapsed": round(time.time() - t0, 3),
            }
        )
        (out / f"model_r{r}.json").write_text(
            json.dumps(
                {
                    "round": r,
                    "experts": {n: e.to_dict() for n, e in model.experts.items()},
                    "theorems": [vars(t) for t in model.theorems],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "rounds": rounds,
        "experts": len(model.experts),
        "theorems": len(model.theorems),
    }


# ==============================================================================
# 12. Comprehensive Self-Test
# ==============================================================================


def self_test_v9() -> None:
    t0 = time.time()
    suite = make_task_suite()
    model = GrowingHagiProcessorV9(node_budget=8000)
    for i, sp in enumerate(suite):
        ev = model.learn(sp, seed=10_000 + i, train_neural=True)
        assert ev["action"] in ("COMMIT", "REUSE"), (sp.name, ev)
    check = model.verify_all(suite)
    assert check["accuracy"] == 1.0 and check["committed"] == len(suite), check
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"

    pm = GrowingHagiProcessorV9()
    snap_before = pm._snapshot_state_with_optimizer()
    pev = pm.learn(
        next(s for s in suite if s.name == "ADD"), seed=77, poison_one_label=True
    )
    assert pev["action"] == "HOLD"
    snap_after = pm._snapshot_state_with_optimizer()
    assert np.array_equal(
        snap_before["core"]["gpm"]["W"], snap_after["core"]["gpm"]["W"]
    )
    assert np.array_equal(snap_before["router"][0], snap_after["router"][0])

    rep = grow_v9(model, suite, max_rounds=2, out_dir="/tmp/v9_grow_test")
    assert rep["experts"] > len(suite), rep
    assert model.verify_certificates()["all_ok"], model.verify_certificates()

    vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5, -0.75, 0.125, 0.0, -3.2, 0.05])
    raw, dec = pack_vector_dyadic5(vec, stages=2)
    assert np.allclose(dec, unpack_vector_dyadic5(raw))

    vb = pack_venn_16_flat()
    assert len(vb) == 13 and unpack_venn_16_flat(vb) == list(range(16))
    for m in range(16):
        for x in (0, 1):
            for y in (0, 1):
                assert venn_head_eval(m, (x, y)) == ((m >> ((x << 1) | y)) & 1)

    tt = anf_transform([0, 1, 1, 0])
    assert anf_to_expr_v9(tt, 2).eval((1, 0)) == 1
    c3 = anf_compose3(lambda a, b: a | b, [0, 0, 0, 1], [0, 1, 1, 0])
    for x in (0, 1):
        for y in (0, 1):
            assert anf_eval(c3, 2, (x, y)) == ((x & y) | (x ^ y))

    col = CRAMEngineV9.solve_coloring(
        [(0, 1), (1, 2), (2, 3), (3, 0)], 4, 2, givens={0: 1}
    )
    assert col["validity_certificate"] and col["givens_respected"]
    adj_m = [[1], [0, 2], [1, 3], [2]]
    lz = CRAMEngineV9.solve_latent_maze(4, adj_m, goal=0)
    assert abs(lz["z"][0] - 21.0 / 55.0) < 1e-4
    mf = CRAMEngineV9.mean_field_kl_step(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.zeros((2, 2, 2, 2)),
        np.full((2, 2), 0.5),
    )
    assert mf["monotonic"]
    alt = CRAMEngineV9.alternating_latent_visible_solve(4, adj_m, goal=0, num_colors=2)
    assert alt["monotone_drop"]
    pl = CRAMEngineV9.train_cram_pseudolikelihood(
        np.array([[0, 1], [0, 1], [1, 0]]), num_colors=2
    )
    assert pl["mean_pl_loss"] < 0.7

    gpm = ContinualGPMMemoryV9(6, 2, seed=1)
    X1 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    Y1 = np.array([[1.0, -1.0], [0.5, 0.5]])
    gpm.train_task_ef21_muon(X1, Y1, steps=20)
    pred1_before = X1 @ gpm.W
    gpm.consolidate_subspace(X1)
    X2 = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    Y2 = np.array([[2.0, -2.0]])
    gpm.train_task_ef21_muon(X2, Y2, steps=20)
    pred1_after = X1 @ gpm.W
    assert float(np.max(np.abs(pred1_before - pred1_after))) < 1e-10

    assert abs(hnet_opt_ratio(1.0, 4.0, 8.0) - 4.0) < 1e-9
    ram = GDNRAMV9(8, 4)
    ram.store(2, np.array([1.0, 2.0, 3.0, 4.0]))
    ram.store(5, np.array([9.0, 8.0, 7.0, 6.0]))
    assert np.allclose(ram.load(2), [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(ram.load_stutter_head([2, 2, 5, 5]), [1.0, 2.0, 3.0, 4.0])

    save_path = Path("v9_selftest_model.json")
    model.save(save_path)
    reloaded = GrowingHagiProcessorV9.load(save_path)
    assert reloaded.verify_all(suite)["accuracy"] == 1.0
    assert reloaded.predict("ADD", 7, -3)["value"] == 4
    assert reloaded.predict("ADD", 99, 1)["status"] == "ABSTAIN"
    for nm, ex in model.experts.items():
        if ex.adapter is not None:
            xs0 = tuple([ex.domains[0][0]] + [d[0] for d in ex.domains[1:]])
            if ex.supports(xs0):
                p1 = model.predict(nm, *xs0, mode="neural")["value"]
                p2 = reloaded.predict(nm, *xs0, mode="neural")["value"]
                assert abs(p1 - p2) < 1e-9, (nm, p1, p2)
    save_path.unlink(missing_ok=True)

    nq = model.neural_quality_summary()
    dt = time.time() - t0
    print(
        f"V9 SELF_TEST PASS ({dt:.2f}s): {check['committed']} base experts acc=1.0 -> "
        f"{rep['experts']} total experts, {len(model.theorems)} theorems, "
        f"neural_acc={nq['mean_val_exact_acc']:.4f}, neural_mse={nq['mean_val_mse']:.4f}, "
        f"neural_bytes={nq['total_neural_bytes']}, certs_ok=True"
    )


# ==============================================================================
# UNIFIED CLI
# ==============================================================================


def _benchmark() -> None:
    """v9 benchmark: exact symbolic suite + neural adapter quality + memory footprint.

    Reports file-size, resident tensor bytes (real parameter memory, unlike the
    seed-compressed neural_bytes payload), wall time, and exact accuracy.
    """

    suite = make_task_suite()
    model = GrowingHagiProcessorV9(node_budget=8000)
    t0 = time.time()
    for i, sp in enumerate(suite):
        model.learn(sp, seed=10_000 + i, train_neural=True)
    learn_elapsed = time.time() - t0
    check = model.verify_all(suite)
    t0 = time.time()
    rep = grow_v9(model, suite, max_rounds=2, out_dir=".omc/v9_bench")
    grow_elapsed = time.time() - t0
    nq = model.neural_quality_summary()
    resident = sum(
        int(v.nbytes) for v in model.__dict__.values() if isinstance(v, np.ndarray)
    )
    core = model.core
    resident_core = sum(
        int(a.nbytes) for a in vars(core).values() if isinstance(a, np.ndarray)
    )
    resident_core += sum(
        int(a.nbytes) for a in vars(core.gpm).values() if isinstance(a, np.ndarray)
    )
    result = {
        "version": V9_VERSION,
        "exact_committed": check["committed"],
        "exact_accuracy": check["accuracy"],
        "experts_after_growth": rep["experts"],
        "theorems": rep["theorems"],
        "neural_val_acc": round(nq["mean_val_exact_acc"], 4),
        "neural_val_mse": round(nq["mean_val_mse"], 4),
        "neural_payload_bytes": nq["total_neural_bytes"],
        "resident_tensor_bytes_core": resident_core,
        "resident_tensor_bytes_model": resident + resident_core,
        "file_bytes": Path(__file__).stat().st_size,
        "learn_suite_seconds": round(learn_elapsed, 2),
        "grow_2rounds_seconds": round(grow_elapsed, 2),
        "certs_ok": model.verify_certificates()["all_ok"],
    }
    print(json.dumps(result, indent=2))


def _cli() -> None:
    _p = argparse.ArgumentParser(
        description="HAGI unified single-file build (v5+v7.4+v8.4+v9)"
    )
    _p.add_argument("--self-test", action="store_true", help="v8.4 self-test")
    _p.add_argument("--self-test-v9", action="store_true", help="v9.0 self-test")
    _p.add_argument("--self-test-v7", action="store_true", help="v7.4 self-test")
    _p.add_argument("--self-test-v5", action="store_true", help="v5 self-test")
    _p.add_argument("--grow", action="store_true", help="v8.4 growth loop")
    _p.add_argument("--grow-v9", action="store_true", help="v9.0 growth loop")
    _p.add_argument("--benchmark-v9", action="store_true", help="v9.0 benchmark")
    _p.add_argument("--max-rounds", type=int, default=4)
    _p.add_argument("--out-dir", default="grow_out")
    _a = _p.parse_args()
    if _a.self_test:
        self_test_v8()
    elif _a.self_test_v9:
        self_test_v9()
    elif _a.self_test_v7:
        self_test_v7()
    elif _a.self_test_v5:
        self_test()
    elif _a.grow:
        _suite = make_task_suite()
        _m = GrowingHagiProcessorV8()
        print(
            json.dumps(
                grow_v8(_m, _suite, max_rounds=_a.max_rounds, out_dir=_a.out_dir),
                indent=2,
            )
        )
    elif _a.grow_v9:
        _suite = make_task_suite()
        _m = GrowingHagiProcessorV9()
        print(
            json.dumps(
                grow_v9(_m, _suite, max_rounds=_a.max_rounds, out_dir=_a.out_dir),
                indent=2,
            )
        )
    elif _a.benchmark_v9:
        _benchmark()
    else:
        _p.print_help()
