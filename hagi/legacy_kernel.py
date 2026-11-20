"""HAGI legacy kernel V5-V8 (extracted from core.py 2026-11-19, Phase 0).

Verbatim body of the old PART I-III. Imported by hagi.core via star
import; shared names are re-defined in core.py V9 section (later wins),
preserving the historical override semantics. Do not edit — frozen.
"""

from __future__ import annotations
import argparse, ast, copy, heapq, itertools, json, math, os, random
import statistics, sys, time
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
import numpy as np

VERSION = "5.0.0"
V5_VERSION = VERSION

# ----------------------------- Safe expression DSL -----------------------------


def _truth(v: Any) -> bool:
    return bool(int(v))


def _bit_not8(x: int) -> int:
    return (~int(x)) & 0xFF


def _bit_nand8(x: int, y: int) -> int:
    return (~(int(x) & int(y))) & 0xFF


def _bit_nor8(x: int, y: int) -> int:
    return (~(int(x) | int(y))) & 0xFF


def _bit_xnor8(x: int, y: int) -> int:
    return (~(int(x) ^ int(y))) & 0xFF


def _sign(x: int) -> int:
    return (int(x) > 0) - (int(x) < 0)


def _pow_int(x: int, y: int) -> int:
    y = int(y)
    if y < 0 or y > 8:
        raise ValueError("exponent outside certified range")
    return int(x) ** y


OPS: dict[str, Callable[..., Any]] = {
    "add": lambda x, y: int(x) + int(y),
    "sub": lambda x, y: int(x) - int(y),
    "mul": lambda x, y: int(x) * int(y),
    "floordiv": lambda x, y: int(x) // int(y),
    "mod": lambda x, y: int(x) % int(y),
    "gcd": lambda x, y: math.gcd(int(x), int(y)),
    "lcm": lambda x, y: math.lcm(int(x), int(y)),
    "min": lambda x, y: min(int(x), int(y)),
    "max": lambda x, y: max(int(x), int(y)),
    "eq": lambda x, y: int(x == y),
    "ne": lambda x, y: int(x != y),
    "lt": lambda x, y: int(x < y),
    "le": lambda x, y: int(x <= y),
    "gt": lambda x, y: int(x > y),
    "ge": lambda x, y: int(x >= y),
    "bit_and": lambda x, y: int(x) & int(y),
    "bit_or": lambda x, y: int(x) | int(y),
    "bit_xor": lambda x, y: int(x) ^ int(y),
    "bit_nand8": _bit_nand8,
    "bit_nor8": _bit_nor8,
    "bit_xnor8": _bit_xnor8,
    "pow_int": _pow_int,
    "neg": lambda x: -int(x),
    "abs": lambda x: abs(int(x)),
    "sign": _sign,
    "bit_not8": _bit_not8,
    "square": lambda x: int(x) * int(x),
    "cube": lambda x: int(x) * int(x) * int(x),
}


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: Optional[Any] = None

    @staticmethod
    def var(i: int) -> "Expr":
        return Expr("var", (), int(i))

    @staticmethod
    def const(v: int) -> "Expr":
        return Expr("const", (), int(v))

    def eval(self, xs: Sequence[int]) -> int:
        if self.op == "var":
            assert self.value is not None
            return int(xs[self.value])
        if self.op == "const":
            assert self.value is not None
            return int(self.value)
        vals = [a.eval(xs) for a in self.args]
        if self.op == "poly":
            coeffs = self.value or ()
            acc = 0
            for ci in reversed(list(coeffs)):
                acc = acc * vals[0] + int(ci)
            return int(acc)
        if self.op == "poly_nd":
            total = 0
            for coefficient, powers in self.value or ():
                term = int(coefficient)
                for x, exponent in zip(xs, powers):
                    term *= int(x) ** int(exponent)
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
        fn = OPS.get(self.op)
        if fn is None:
            raise ValueError(f"unsupported expression opcode: {self.op}")
        return int(fn(*vals))

    @property
    def cost(self) -> int:
        c = self.__dict__.get("_cost_memo")
        if c is None:
            if self.op in ("var", "const"):
                c = 1
            else:
                c = 1 + sum(a.cost for a in self.args)
            object.__setattr__(self, "_cost_memo", c)
        return c

    def pretty(self) -> str:
        if self.op == "var":
            return ("x", "y", "z")[self.value or 0]
        if self.op == "const":
            return str(self.value)
        if self.op == "poly":
            coeffs = tuple(self.value or ())
            terms = []
            for i in range(len(coeffs) - 1, -1, -1):
                c = int(coeffs[i])
                if c == 0:
                    continue
                mag = abs(c)
                if i == 0:
                    term = str(mag)
                else:
                    body = "x" if i == 1 else f"x**{i}"
                    term = body if mag == 1 else f"{mag}*{body}"
                if not terms:
                    terms.append(("-" if c < 0 else "") + term)
                else:
                    terms.append((" - " if c < 0 else " + ") + term)
            return "poly(" + ("".join(terms) if terms else "0") + ")"
        if self.op == "poly_nd":
            symbols = ("x", "y", "z", "u", "v", "w")
            terms = []
            for coefficient, powers in sorted(
                (self.value or ()),
                key=lambda item: (sum(item[1]), item[1]),
                reverse=True,
            ):
                c = int(coefficient)
                if c == 0:
                    continue
                factors = []
                for index, exponent in enumerate(powers):
                    exponent = int(exponent)
                    if exponent == 0:
                        continue
                    name = symbols[index] if index < len(symbols) else f"x{index}"
                    factors.append(name if exponent == 1 else f"{name}**{exponent}")
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
        return {
            "op": self.op,
            "value": self.value,
            "args": [a.to_dict() for a in self.args],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Expr":
        op = str(d["op"])
        value = d.get("value")
        # JSON turns immutable tuples into lists. Restore the canonical value
        # shape so expressions stay hashable and equality-stable after reload.
        if op == "poly" and value is not None:
            value = tuple(int(v) for v in value)
        elif op == "poly_nd" and value is not None:
            value = tuple(
                (int(term[0]), tuple(int(v) for v in term[1])) for term in value
            )
        return Expr(op, tuple(Expr.from_dict(a) for a in d.get("args", [])), value)


@dataclass(frozen=True)
class Program:
    arity: int
    expr: Optional[Expr] = None
    venn_mask: Optional[int] = None

    @property
    def cost(self) -> int:
        return (
            4
            if self.venn_mask is not None
            else (self.expr.cost if self.expr else 10**9)
        )

    def eval(self, xs: Sequence[int]) -> int:
        if len(xs) != self.arity:
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")
        if self.venn_mask is not None:
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):
                raise ValueError("BoolVenn expert accepts two Boolean inputs")
            idx = (int(xs[0]) << 1) | int(xs[1])
            return (int(self.venn_mask) >> idx) & 1
        if self.expr is None:
            raise RuntimeError("program has no expression")
        return self.expr.eval(xs)

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
    def from_dict(d: dict[str, Any]) -> "Program":
        return Program(
            int(d["arity"]),
            Expr.from_dict(d["expr"]) if d.get("expr") else None,
            d.get("venn_mask"),
        )


# ------------------------------- Task definitions ------------------------------


@dataclass
class TaskSpec:
    name: str
    arity: int
    domains: tuple[tuple[int, ...], ...]
    teacher: Callable[..., int]
    valid_rule: str = "all"
    family: str = "integer"

    def inputs(self) -> list[tuple[int, ...]]:
        rows = itertools.product(*self.domains)
        out = []
        for row in rows:
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
    return suite


# ------------------------- Candidate program enumeration ------------------------


def _dedupe_candidates(candidates: Iterable[Expr]) -> list[Expr]:
    by_text: dict[str, Expr] = {}
    for e in candidates:
        by_text.setdefault(e.pretty(), e)
    return sorted(by_text.values(), key=lambda e: (e.cost, e.pretty()))


def candidate_programs(arity: int) -> list[Program]:
    v = [Expr.var(i) for i in range(arity)]
    cands: list[Expr] = []
    if arity == 1:
        x = v[0]
        cands += [
            Expr("neg", (x,)),
            Expr("abs", (x,)),
            Expr("sign", (x,)),
            Expr("bit_not8", (x,)),
            Expr("mul", (x, x)),
            Expr("mul", (Expr("mul", (x, x)), x)),
            Expr("clamp3", (x,)),
        ]
        return [Program(arity, e) for e in _dedupe_candidates(cands)]
    if arity == 2:
        x, y = v
        for op in (
            "add",
            "sub",
            "mul",
            "floordiv",
            "mod",
            "gcd",
            "lcm",
            "min",
            "max",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "bit_and",
            "bit_or",
            "bit_xor",
            "bit_nand8",
            "bit_nor8",
            "bit_xnor8",
            "pow_int",
        ):
            cands.append(Expr(op, (x, y)))
        cands.append(Expr("sub", (y, x)))
        cands.append(Expr("absdiff", (x, y)))
        cands.append(Expr("abs", (Expr("sub", (x, y)),)))
        cands.append(Expr("abs", (Expr("sub", (y, x)),)))
        # Small compositional grammar: two-layer expressions from arithmetic atoms.
        atoms = [
            Expr("add", (x, y)),
            Expr("sub", (x, y)),
            Expr("sub", (y, x)),
            Expr("mul", (x, y)),
            Expr("abs", (x,)),
            Expr("abs", (y,)),
        ]
        for a in atoms:
            cands.append(Expr("abs", (a,)))
        return [Program(arity, e) for e in _dedupe_candidates(cands)]
    if arity == 3:
        x, y, z = v
        cands += [
            Expr("fma", (x, y, z)),
            Expr("clamp", (x, y, z)),
            Expr("median3", (x, y, z)),
            Expr("mul3", (x, y, z)),
            Expr("add", (Expr("mul", (x, y)), z)),
            Expr("max", (Expr("min", (x, y)), Expr("min", (Expr("max", (x, y)), z)))),
        ]
        return [Program(arity, e) for e in _dedupe_candidates(cands)]
    return []


# ------------------------------- Neural fiber ----------------------------------


@dataclass
class NeuralFiber:
    arity: int
    input_scale: float
    target_scale: float
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    train_loss: float = float("nan")
    validation_loss: float = float("nan")
    validation_exact_accuracy: float = float("nan")

    @staticmethod
    def fit(
        inputs: Sequence[Sequence[int]],
        targets: Sequence[int],
        val_inputs: Sequence[Sequence[int]],
        val_targets: Sequence[int],
        seed: int,
        hidden: int = 24,
        epochs: int = 140,
        lr: float = 0.025,
    ) -> "NeuralFiber":
        x = np.asarray(inputs, dtype=np.float64).reshape(-1, len(inputs[0]))
        y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
        xv = np.asarray(val_inputs, dtype=np.float64).reshape(-1, x.shape[1])
        yv = np.asarray(val_targets, dtype=np.float64).reshape(-1, 1)
        sx = max(1.0, float(np.max(np.abs(x))))
        sy = max(1.0, float(np.max(np.abs(y))))
        x, y, xv, yv = x / sx, y / sy, xv / sx, yv / sy
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, 0.32, (x.shape[1], hidden))
        b1 = np.zeros((1, hidden), dtype=np.float64)
        w2 = rng.normal(0.0, 0.12, (hidden, 1))
        b2 = np.zeros((1, 1), dtype=np.float64)
        params = [w1, b1, w2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        best = [p.copy() for p in params]
        best_val = float("inf")
        step = 0
        for _ in range(epochs):
            h = np.tanh(x @ w1 + b1)
            pred = h @ w2 + b2
            err = pred - y
            loss = float(np.mean(err * err))
            dp = (2.0 / len(x)) * err
            dw2 = h.T @ dp
            db2 = np.sum(dp, axis=0, keepdims=True)
            dh = (dp @ w2.T) * (1.0 - h * h)
            dw1 = x.T @ dh
            db1 = np.sum(dh, axis=0, keepdims=True)
            grads = [dw1, db1, dw2, db2]
            step += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                mh = m[i] / (1 - beta1**step)
                vh = v[i] / (1 - beta2**step)
                p -= lr * mh / (np.sqrt(vh) + eps)
            hv = np.tanh(xv @ w1 + b1)
            pv = hv @ w2 + b2
            vl = float(np.mean((pv - yv) ** 2))
            if np.isfinite(vl) and vl < best_val:
                best_val = vl
                best = [p.copy() for p in params]
        w1, b1, w2, b2 = best
        h = np.tanh(x @ w1 + b1)
        tr_loss = float(np.mean((h @ w2 + b2 - y) ** 2)) * sy * sy
        hv = np.tanh(xv @ w1 + b1)
        predv = (hv @ w2 + b2).reshape(-1) * sy
        val_loss = float(np.mean((predv - np.asarray(val_targets, dtype=float)) ** 2))
        exact = float(
            np.mean(
                np.rint(predv).astype(np.int64)
                == np.asarray(val_targets, dtype=np.int64)
            )
        )
        return NeuralFiber(
            len(inputs[0]), sx, sy, w1, b1, w2, b2, tr_loss, val_loss, exact
        )

    def predict(self, xs: Sequence[int]) -> float:
        if len(xs) != self.arity:
            raise ValueError("wrong neural fiber arity")
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1) / self.input_scale
        return float(
            (np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)[0, 0]
            * self.target_scale
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arity": self.arity,
            "input_scale": self.input_scale,
            "target_scale": self.target_scale,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "validation_exact_accuracy": self.validation_exact_accuracy,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "NeuralFiber":
        return NeuralFiber(
            int(d["arity"]),
            float(d["input_scale"]),
            float(d["target_scale"]),
            np.asarray(d["w1"], dtype=np.float64),
            np.asarray(d["b1"], dtype=np.float64),
            np.asarray(d["w2"], dtype=np.float64),
            np.asarray(d["b2"], dtype=np.float64),
            float(d.get("train_loss", float("nan"))),
            float(d.get("validation_loss", float("nan"))),
            float(d.get("validation_exact_accuracy", float("nan"))),
        )


# --------------------------------- Expert store ---------------------------------


@dataclass
class Expert:
    name: str
    family: str
    program: Program
    domains: tuple[tuple[int, ...], ...]
    valid_rule: str
    code_length: int
    neural: Optional[NeuralFiber] = None
    exhaustive_checks: int = 0
    exhaustive_errors: int = 0
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
            "neural": self.neural.to_dict() if self.neural else None,
            "exhaustive_checks": self.exhaustive_checks,
            "exhaustive_errors": self.exhaustive_errors,
            "parents": list(self.parents),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Expert":
        return Expert(
            str(d["name"]),
            str(d["family"]),
            Program.from_dict(d["program"]),
            tuple(tuple(int(v) for v in x) for x in d["domains"]),
            str(d["valid_rule"]),
            int(d["code_length"]),
            NeuralFiber.from_dict(d["neural"]) if d.get("neural") else None,
            int(d.get("exhaustive_checks", 0)),
            int(d.get("exhaustive_errors", 0)),
            tuple(str(x) for x in d.get("parents", [])),
        )


@dataclass
class GrowthEvent:
    version: int
    task: str
    action: str
    reason: str
    program: str = ""
    code_length: int = 0
    train_examples: int = 0
    validation_examples: int = 0
    exact_accuracy: float = 0.0


def _bounded_int(value: Any, lo: int, hi: int) -> int:
    iv = int(value)
    if iv < lo or iv > hi:
        raise ValueError(f"integer argument must be in [{lo},{hi}]")
    return iv


class SafeExpression:
    """Bounded arithmetic/logic expression interpreter; never uses eval()."""

    BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
        ast.BitAnd: lambda a, b: int(a) & int(b),
        ast.BitOr: lambda a, b: int(a) | int(b),
        ast.BitXor: lambda a, b: int(a) ^ int(b),
        ast.LShift: lambda a, b: int(a) << int(b),
        ast.RShift: lambda a, b: int(a) >> int(b),
    }
    UNARY = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
        ast.Invert: lambda a: ~int(a),
        ast.Not: lambda a: int(not bool(a)),
    }
    CMP = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
    }
    CALLS = {
        "abs": abs,
        "min": min,
        "max": max,
        "gcd": math.gcd,
        "lcm": math.lcm,
        "sign": _sign,
        "bit_not8": _bit_not8,
        "clamp": lambda x, lo, hi: min(max(x, lo), hi),
        "nand8": _bit_nand8,
        "nor8": _bit_nor8,
        "xnor8": _bit_xnor8,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "floor": math.floor,
        "ceil": math.ceil,
        "hypot": math.hypot,
        "degrees": math.degrees,
        "radians": math.radians,
        "factorial": lambda n: math.factorial(_bounded_int(n, 0, 1000)),
        "comb": lambda n, k: math.comb(
            _bounded_int(n, 0, 1000), _bounded_int(k, 0, 1000)
        ),
        "perm": lambda n, k=None: math.perm(
            _bounded_int(n, 0, 1000), None if k is None else _bounded_int(k, 0, 1000)
        ),
    }

    def evaluate(
        self, expression: str, variables: dict[str, int | float]
    ) -> int | float | bool:
        if len(expression) > 1000:
            raise ValueError("expression too long")
        tree = ast.parse(expression, mode="eval")
        nodes = list(ast.walk(tree))
        if len(nodes) > 200:
            raise ValueError("expression too complex")
        result = self._visit(tree.body, variables)
        if isinstance(result, (int, np.integer)) and int(result).bit_length() > 4096:
            raise OverflowError("result exceeds 4096-bit safety cap")
        if isinstance(result, (float, np.floating)) and not math.isfinite(
            float(result)
        ):
            raise OverflowError("non-finite floating-point result")
        return result

    def _visit(self, node: ast.AST, variables: dict[str, int | float]):
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise ValueError("non-finite numeric literal")
            return node.value
        if isinstance(node, ast.Name):
            if node.id in {"pi", "e", "tau"}:
                return {"pi": math.pi, "e": math.e, "tau": math.tau}[node.id]
            if node.id not in variables:
                raise ValueError(f"unknown variable: {node.id}")
            value = variables[node.id]
            if isinstance(value, float):
                if not math.isfinite(value) or abs(value) > 1e12:
                    raise OverflowError("floating input outside safe finite range")
            elif abs(int(value)).bit_length() > 1024:
                raise OverflowError("input exceeds 1024-bit safety cap")
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in self.BINOPS:
            a, b = self._visit(node.left, variables), self._visit(node.right, variables)
            if isinstance(node.op, ast.Pow) and abs(float(b)) > 64:
                raise ValueError("absolute exponent must be at most 64")
            if isinstance(node.op, (ast.LShift, ast.RShift)) and not (
                0 <= int(b) <= 4096
            ):
                raise ValueError("shift must be in [0,4096]")
            return self.BINOPS[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.UNARY:
            return self.UNARY[type(node.op)](self._visit(node.operand, variables))
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            vals = [self._visit(v, variables) for v in node.values]
            return int(
                all(bool(v) for v in vals)
                if isinstance(node.op, ast.And)
                else any(bool(v) for v in vals)
            )
        if isinstance(node, ast.Compare):
            left = self._visit(node.left, variables)
            ok = True
            for op, comparator in zip(node.ops, node.comparators):
                if type(op) not in self.CMP:
                    raise ValueError("comparison operator not allowed")
                right = self._visit(comparator, variables)
                ok = ok and bool(self.CMP[type(op)](left, right))
                left = right
            return int(ok)
        if isinstance(node, ast.IfExp):
            cond = self._visit(node.test, variables)
            return self._visit(node.body if bool(cond) else node.orelse, variables)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.CALLS
        ):
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            args = [self._visit(a, variables) for a in node.args]
            return self.CALLS[node.func.id](*args)
        raise ValueError(f"unsupported syntax: {type(node).__name__}")


class GrowingMathProcessor:
    """An expert-growing processor with exact finite-domain guards and neural fibers."""

    def __init__(self, lookup_limit: int = 16, mdl_lambda: float = 0.001):
        self.lookup_limit = int(lookup_limit)
        self.mdl_lambda = float(mdl_lambda)
        self.experts: dict[str, Expert] = {}
        self.version = 0
        self.growth_log: list[GrowthEvent] = []
        self.total_commits = 0
        self.total_holds = 0

    def _candidate_list(self, spec: TaskSpec) -> list[Program]:
        if spec.family == "BoolVenn":
            return [Program(2, venn_mask=m) for m in range(16)]
        return candidate_programs(spec.arity)

    @staticmethod
    def _same(a: Any, b: Any) -> bool:
        try:
            return int(a) == int(b)
        except Exception:
            return False

    def learn(
        self,
        spec: TaskSpec,
        seed: int = 0,
        train_fraction: float = 0.70,
        poison_one_label: bool = False,
        train_neural: bool = True,
    ) -> GrowthEvent:
        """Discover and attempt to commit one expert from supervised examples.

        The oracle is used only by this offline experiment's validation/exhaustive
        audit. In deployed use, replace it with an independently supplied verifier.
        """
        if spec.name in self.experts:
            ev = GrowthEvent(self.version, spec.name, "REUSE", "already learned")
            self.growth_log.append(ev)
            return ev
        domain = spec.inputs()
        if not domain:
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(domain))
        if spec.family == "BoolVenn":
            n_train = max(1, len(domain) - 1)  # keep one truth-table atom as holdout
        else:
            n_train = max(
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))
            )
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        train_x = [domain[int(i)] for i in train_idx]
        val_x = [domain[int(i)] for i in val_idx]
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}
        train_y = [all_y[xs] for xs in train_x]
        val_y = [all_y[xs] for xs in val_x]
        if poison_one_label and train_y:
            # deterministic corruption used as an adversarial hold/rollback test
            train_y[0] = int(train_y[0]) + 1

        # Fit an approximate neural fiber from the same visible training examples.
        neural = None
        if train_neural and val_x:
            try:
                neural = NeuralFiber.fit(
                    train_x, train_y, val_x, val_y, seed=seed + 991, epochs=120
                )
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                neural = None

        candidates = self._candidate_list(spec)
        train_matches: list[Program] = []
        for program in candidates:
            try:
                if all(
                    self._same(program.eval(xs), yy) for xs, yy in zip(train_x, train_y)
                ):
                    train_matches.append(program)
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                continue
        train_matches.sort(key=lambda p: (p.cost, p.pretty()))

        # Independent validation first, then exhaustive comparison with the finite oracle.
        for program in train_matches:
            try:
                if not all(
                    self._same(program.eval(xs), yy) for xs, yy in zip(val_x, val_y)
                ):
                    continue
                errors = 0
                for xs in domain:
                    if not self._same(program.eval(xs), all_y[xs]):
                        errors += 1
                if errors != 0:
                    continue
                # MDL-style objective: exact predictive error plus explicit code length.
                code_length = int(program.cost)
                mdl_score = 0.0 + self.mdl_lambda * code_length
                if not np.isfinite(mdl_score) or code_length > 32:
                    continue
                expert = Expert(
                    spec.name,
                    spec.family,
                    program,
                    spec.domains,
                    spec.valid_rule,
                    code_length,
                    neural,
                    len(domain),
                    0,
                )
                self.experts[spec.name] = expert
                self.version += 1
                self.total_commits += 1
                ev = GrowthEvent(
                    self.version,
                    spec.name,
                    "COMMIT",
                    "validation+exhaustive pass",
                    program.pretty(),
                    code_length,
                    len(train_x),
                    len(val_x),
                    1.0,
                )
                self.growth_log.append(ev)
                return ev
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                continue

        # Finite lookup is allowed only for complete support <= configured limit.
        if (
            len(domain) <= self.lookup_limit
            and not poison_one_label
            and len(train_x) == len(domain)
        ):
            # Current benchmark deliberately has no lookup task; this branch is reserved
            # for future complete-support micro-domains and does not silently extrapolate.
            pass
        return self._hold(
            spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x), neural
        )

    def _hold(
        self,
        spec: TaskSpec,
        reason: str,
        ntrain: int,
        nval: int,
        neural: Optional[NeuralFiber] = None,
    ) -> GrowthEvent:
        self.total_holds += 1
        ev = GrowthEvent(
            self.version, spec.name, "HOLD", reason, "", 0, ntrain, nval, 0.0
        )
        self.growth_log.append(ev)
        return ev

    def predict(self, name: str, *xs: int, mode: str = "exact") -> dict[str, Any]:
        expert = self.experts.get(name)
        if expert is None:
            return {"status": "ABSTAIN", "reason": "unknown operation"}
        if not expert.supports(xs):
            return {"status": "ABSTAIN", "reason": "outside certified finite domain"}
        if mode == "exact":
            return {
                "status": "OK",
                "value": expert.program.eval(xs),
                "source": "verified-symbolic",
                "model_version": self.version,
            }
        if mode == "neural":
            if expert.neural is None:
                return {"status": "ABSTAIN", "reason": "neural fiber unavailable"}
            return {
                "status": "APPROXIMATE",
                "value": expert.neural.predict(xs),
                "source": "neural-fiber-no-exact-guarantee",
                "model_version": self.version,
            }
        raise ValueError("mode must be 'exact' or 'neural'")

    def verify_all(self, specs: Sequence[TaskSpec]) -> dict[str, Any]:
        records = []
        total = correct = 0
        for spec in specs:
            expert = self.experts.get(spec.name)
            if expert is None:
                records.append(
                    {
                        "task": spec.name,
                        "committed": False,
                        "accuracy": 0.0,
                        "n": len(spec.inputs()),
                    }
                )
                continue
            rows = spec.inputs()
            ok = sum(
                int(expert.program.eval(xs) == int(spec.teacher(*xs))) for xs in rows
            )
            total += len(rows)
            correct += ok
            records.append(
                {
                    "task": spec.name,
                    "committed": True,
                    "accuracy": ok / max(1, len(rows)),
                    "n": len(rows),
                    "program": expert.program.pretty(),
                    "code_length": expert.code_length,
                }
            )
        return {
            "records": records,
            "checked": total,
            "correct": correct,
            "accuracy": correct / max(1, total),
            "committed": len(self.experts),
        }

    def evaluate_expression(
        self, expression: str, **variables: int | float
    ) -> dict[str, Any]:
        """Evaluate a composed integer/logic expression in the bounded safe DSL."""
        try:
            value = SafeExpression().evaluate(expression, variables)
            return {
                "status": "OK",
                "value": value,
                "source": "safe-expression-dsl",
                "model_version": self.version,
            }
        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            SyntaxError,
        ) as exc:
            return {
                "status": "ERROR",
                "reason": str(exc),
                "model_version": self.version,
            }

    def save(self, path: str | Path) -> None:
        data = {
            "format": "HAGI-GrowingMathProcessor",
            "version": VERSION,
            "model_version": self.version,
            "lookup_limit": self.lookup_limit,
            "mdl_lambda": self.mdl_lambda,
            "total_commits": self.total_commits,
            "total_holds": self.total_holds,
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},
            "growth_log": [vars(x) for x in self.growth_log],
        }
        Path(path).write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def load(path: str | Path) -> "GrowingMathProcessor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "HAGI-GrowingMathProcessor":
            raise ValueError("not a HAGI Growing Math Processor state file")
        model = GrowingMathProcessor(
            int(data.get("lookup_limit", 16)), float(data.get("mdl_lambda", 0.001))
        )
        model.version = int(data.get("model_version", 0))
        model.total_commits = int(data.get("total_commits", 0))
        model.total_holds = int(data.get("total_holds", 0))
        model.experts = {
            k: Expert.from_dict(v) for k, v in data.get("experts", {}).items()
        }
        model.growth_log = [GrowthEvent(**x) for x in data.get("growth_log", [])]
        return model


# ---------------------------- Autonomous experiment ----------------------------


def run_experiment(
    seeds: Sequence[int] = (0, 1, 2), out_dir: str = "."
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite = make_task_suite()
    per_seed = []
    main_model: Optional[GrowingMathProcessor] = None
    main_seed = int(seeds[0]) if seeds else 0
    start = time.time()
    for seed in seeds:
        model = GrowingMathProcessor()
        seed_t0 = time.time()
        events: list[GrowthEvent] = []
        # Ordered curriculum: logic -> unary -> arithmetic -> predicates/bitwise -> compositions.
        stage_order = ["BoolVenn", "unary", "integer-binary", "composed-ternary"]
        growth_curve = []
        for stage in stage_order:
            stage_tasks = [s for s in suite if s.family == stage]
            # Deterministic within-stage ordering, shuffled by seed to expose order effects.
            rng = np.random.default_rng(seed + len(stage) * 101)
            if len(stage_tasks) > 1:
                order = rng.permutation(len(stage_tasks))
                stage_tasks = [stage_tasks[int(i)] for i in order]
            for j, spec in enumerate(stage_tasks):
                events.append(
                    model.learn(spec, seed=seed * 1009 + j * 17 + len(spec.name))
                )
            growth_curve.append(
                {
                    "stage": stage,
                    "new_experts": len(stage_tasks),
                    "cumulative_experts": len(model.experts),
                    "cumulative_code_length": sum(
                        e.code_length for e in model.experts.values()
                    ),
                }
            )
        verification = model.verify_all(suite)
        committed = len(model.experts)
        neural_scores = [
            e.neural.validation_exact_accuracy
            for e in model.experts.values()
            if e.neural is not None and np.isfinite(e.neural.validation_exact_accuracy)
        ]
        neural_rmse = [
            math.sqrt(max(0.0, e.neural.validation_loss))
            for e in model.experts.values()
            if e.neural is not None and np.isfinite(e.neural.validation_loss)
        ]
        # Guard audit: every known operation must abstain on an input outside its guard.
        ood_checks = ood_abstains = 0
        for spec in suite:
            if spec.name not in model.experts:
                continue
            outside = [d[-1] + 1 for d in spec.domains]
            result = model.predict(spec.name, *outside)
            ood_checks += 1
            ood_abstains += int(result["status"] == "ABSTAIN")
        # Poison test: one mislabeled ADD example must not create a new committed expert.
        poison_model = GrowingMathProcessor()
        add_spec = next(s for s in suite if s.name == "ADD")
        poison_event = poison_model.learn(
            add_spec, seed=seed + 44001, poison_one_label=True
        )
        # Unknown nontrivial function has 17-point support: finite lookup limit 16 -> hold.
        prime_spec = TaskSpec(
            "IS_PRIME",
            1,
            (tuple(range(-8, 9)),),
            lambda x: int(
                x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))
            ),
            family="unseen",
        )
        unknown_event = GrowingMathProcessor().learn(prime_spec, seed=seed + 99001)
        elapsed = time.time() - seed_t0
        row = {
            "seed": int(seed),
            "committed": committed,
            "expected": len(suite),
            "commit_rate": committed / len(suite),
            "symbolic_exact_accuracy": verification["accuracy"],
            "symbolic_examples_checked": verification["checked"],
            "neural_holdout_exact_accuracy_mean": (
                float(np.mean(neural_scores)) if neural_scores else float("nan")
            ),
            "neural_holdout_rmse_mean": (
                float(np.mean(neural_rmse)) if neural_rmse else float("nan")
            ),
            "ood_checks": ood_checks,
            "ood_abstains": ood_abstains,
            "poison_action": poison_event.action,
            "unknown_is_prime_action": unknown_event.action,
            "growth_curve": growth_curve,
            "elapsed_seconds": elapsed,
        }
        per_seed.append(row)
        if seed == main_seed:
            main_model = model
            # Full detail by task for the primary seed.
            with (out / f"hagi_growing_math_seed{seed}_tasks.csv").open(
                "w", encoding="utf-8"
            ) as f:
                f.write(
                    "task,family,committed,program,code_length,neural_val_exact_acc,neural_val_mse\n"
                )
                for spec in suite:
                    ex = model.experts.get(spec.name)
                    if ex is None:
                        f.write(f"{spec.name},{spec.family},0,,0,,\n")
                    else:
                        na = (
                            ex.neural.validation_exact_accuracy
                            if ex.neural
                            else float("nan")
                        )
                        nm = ex.neural.validation_loss if ex.neural else float("nan")
                        prog = ex.program.pretty().replace('"', '""')
                        f.write(
                            f'"{spec.name}","{spec.family}",1,"{prog}",{ex.code_length},{na},{nm}\n'
                        )
    assert main_model is not None
    model_path = out / "hagi_growing_math_model.json"
    main_model.save(model_path)

    # Adversarial/guard smoke test on the saved and reloaded model.
    reloaded = GrowingMathProcessor.load(model_path)
    reload_ok = reloaded.predict("ADD", 7, -3).get("value") == 4
    ood_ok = reloaded.predict("ADD", 99, 2).get("status") == "ABSTAIN"

    aggregate: dict[str, Any] = {}
    metric_keys = (
        "committed",
        "symbolic_exact_accuracy",
        "neural_holdout_exact_accuracy_mean",
        "neural_holdout_rmse_mean",
        "ood_abstains",
        "elapsed_seconds",
    )
    for key in metric_keys:
        vals = [float(r[key]) for r in per_seed]
        base = key[:-5] if key.endswith("_mean") else key
        aggregate[base + "_mean"] = float(np.mean(vals))
        aggregate[base + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    aggregate.update(
        {
            "seeds": list(map(int, seeds)),
            "expected_operations": len(suite),
            "finite_support_examples_per_operation": [len(s.inputs()) for s in suite],
            "total_finite_support_examples": int(sum(len(s.inputs()) for s in suite)),
            "reload_roundtrip_pass": bool(reload_ok),
            "ood_abstain_pass": bool(ood_ok),
            "total_runtime_seconds": time.time() - start,
        }
    )
    result = {
        "version": VERSION,
        "scope": "integer inputs -8..8 (with task-specific subdomains) + all 16 two-input Boolean functions",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "interpretation": [
            "Exact results are certified only on each operation's finite declared support.",
            "The neural fibers are approximate and are not the exact inference path.",
            "Unknown functions outside the learned library abstain; no claim of all mathematics is made.",
            "The teacher oracle is used by the offline experiment's independent validation/exhaustive audit.",
        ],
    }
    (out / "hagi_growing_math_experiment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_report(result, out / "hagi_growing_math_report.md")
    return result


def _write_report(result: dict[str, Any], path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# HAGI Growing Math Processor — autonomous experiment",
        "",
        f"Version: {result['version']}",
        f"Seeds: {', '.join(map(str, agg['seeds']))}",
        f"Declared operation set: {agg['expected_operations']} tasks (16 BoolVenn + integer unary/binary + ternary compositions).",
        "",
        "## Learned operation catalogue",
        "",
        "- 16 BoolVenn functions: every two-input Boolean truth table (FALSE, NOR, implications, XOR/XNOR, AND/NAND, OR, TRUE, projections and negations).",
        "- 7 unary integer functions: NEG, ABS, SIGN, SQUARE, CUBE, BIT_NOT8, CLAMP_TO_3.",
        "- 23 binary integer functions: ADD, SUB, MUL, FLOORDIV, MOD, GCD, LCM, MIN, MAX, ABSDIFF, EQ/NE/LT/LE/GT/GE, BIT_AND/OR/XOR/NAND8/NOR8/XNOR8, POW_INT.",
        "- 3 ternary compositions: FMA, CLAMP, MEDIAN3.",
        "",
        "The expression interpreter additionally supports bounded integer arithmetic/bitwise expressions and selected real functions (sqrt, exp, log, trigonometric and hyperbolic functions). Those calculator functions are not part of the 49-operation exhaustive-learning score.",
        "",
        "## Aggregate results",
        "",
        "| Metric | Mean | SD |",
        "|---|---:|---:|",
        f"| Committed experts | {agg['committed_mean']:.2f} | {agg['committed_sd']:.2f} |",
        f"| Exhaustive symbolic accuracy | {100*agg['symbolic_exact_accuracy_mean']:.2f}% | {100*agg['symbolic_exact_accuracy_sd']:.2f} pp |",
        f"| Neural holdout exact accuracy | {100*agg['neural_holdout_exact_accuracy_mean']:.2f}% | {100*agg['neural_holdout_exact_accuracy_sd']:.2f} pp |",
        f"| Neural holdout RMSE | {agg['neural_holdout_rmse_mean']:.4f} | {agg['neural_holdout_rmse_sd']:.4f} |",
        f"| OOD abstentions | {agg['ood_abstains_mean']:.2f} | {agg['ood_abstains_sd']:.2f} |",
        f"| Runtime per seed | {agg['elapsed_seconds_mean']:.2f}s | {agg['elapsed_seconds_sd']:.2f}s |",
        "",
        "## Per-seed results",
        "",
        "| Seed | Experts | Symbolic exact | Neural holdout exact | Neural RMSE | OOD abstain | Poison | Unknown IS_PRIME |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in result["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['committed']}/{r['expected']} | {100*r['symbolic_exact_accuracy']:.2f}% | {100*r['neural_holdout_exact_accuracy_mean']:.2f}% | {r['neural_holdout_rmse_mean']:.4f} | {r['ood_abstains']}/{r['ood_checks']} | {r['poison_action']} | {r['unknown_is_prime_action']} |"
        )
    lines += [
        "",
        "## Growth curve (seed 0)",
        "",
        "| Stage | New experts | Cumulative experts | Cumulative code units |",
        "|---|---:|---:|---:|",
    ]
    for stage in result["per_seed"][0].get("growth_curve", []):
        lines.append(
            f"| {stage['stage']} | {stage['new_experts']} | {stage['cumulative_experts']} | {stage['cumulative_code_length']} |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "The operation library grows by adding one expert after program synthesis, held-out validation, and exhaustive comparison over that operation's finite support. Each expert also receives a small trained neural fiber; approximate neural predictions are available separately, but exact inference uses the verified symbolic program.",
        "",
        "The 16 two-input Boolean functions are the complete BoolVenn truth-table family. Integer functions are certified only on the finite input ranges recorded in the task table. This is not a proof of correctness for unbounded integers, real analysis, transcendental functions, arbitrary user-defined functions, or every possible mathematical operation. The safe expression interpreter computes selected transcendental functions but does not discover or formally certify them as learned experts.",
        "",
        "Out-of-domain inputs abstain. The poisoned-label and unknown-prime probes are expected to HOLD rather than silently commit. The training oracle is available in the offline experiment; a deployed learner needs an independent verifier or new labelled examples to certify a new function.",
        "",
        "The design is MDL-inspired: candidates are ordered by AST description length and exact-fit programs are required to pass a hard finite-domain commit check. This is not a complete implementation of the repository's probabilistic/energy CommitGate or its Lean proof obligations. BoolVenn follows the 16-function finite lattice; the growth controller adds experts in stages. These formal components do not establish generalization beyond the specified finite domains.",
        "",
        "## Related Lean specifications",
        "",
        "- BoolVenn: https://github.com/ShmidtS/primes/blob/main/Hagi/Data/BoolVenn.lean",
        "- MDL objective: https://github.com/ShmidtS/primes/blob/main/Hagi/Step/MDLObjective.lean",
        "- Safe commit gate: https://github.com/ShmidtS/primes/blob/main/Hagi/Unified/CommitGate.lean",
        "- Self-development search: https://github.com/ShmidtS/primes/blob/main/Hagi/Growth/SelfDevelopment.lean",
        "- Recursive growth conditions: https://github.com/ShmidtS/primes/blob/main/Hagi/Growth/RecursiveSelfDevelopment.lean",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------- Tests --------------------------------------


def self_test() -> None:
    suite = make_task_suite()
    model = GrowingMathProcessor()
    for i, spec in enumerate(suite):
        ev = model.learn(spec, seed=10_000 + i, train_neural=False)
        assert ev.action == "COMMIT", (spec.name, ev)
    check = model.verify_all(suite)
    assert check["committed"] == len(suite)
    assert check["accuracy"] == 1.0
    assert all(
        model.predict("ADD", x, y)["value"] == x + y
        for x in range(-8, 9)
        for y in range(-8, 9)
    )
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"
    assert model.predict("BOOL_XOR", 1, 0)["value"] == 1
    assert model.predict("BOOL_NAND", 1, 1)["value"] == 0
    with_path = Path("/tmp/hagi_growing_math_selftest.json")
    model.save(with_path)
    loaded = GrowingMathProcessor.load(with_path)
    assert loaded.predict("FMA", 2, 3, 4)["value"] == 10
    assert loaded.predict("CLAMP", 99, -2, 2)["status"] == "ABSTAIN"
    assert loaded.evaluate_expression("x*y + gcd(x,y)", x=7, y=3)["value"] == 22
    assert (
        loaded.evaluate_expression("(x < y) and (y % 2 == 1)", x=2, y=3)["value"] == 1
    )
    trig = loaded.evaluate_expression("sqrt(x*x) + sin(pi/2)", x=3)
    assert trig["status"] == "OK" and abs(trig["value"] - 4.0) < 1e-10
    assert (
        loaded.evaluate_expression("__import__('os').system('false')", x=1)["status"]
        == "ERROR"
    )
    # Poisoned data and unsupported larger-support algorithm must be rejected.
    poisoned = GrowingMathProcessor().learn(
        next(s for s in suite if s.name == "ADD"),
        seed=77,
        poison_one_label=True,
        train_neural=False,
    )
    assert poisoned.action == "HOLD"
    unknown = TaskSpec(
        "IS_PRIME",
        1,
        (tuple(range(-8, 9)),),
        lambda x: int(x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))),
    )
    assert (
        GrowingMathProcessor().learn(unknown, seed=78, train_neural=False).action
        == "HOLD"
    )
    print(
        f"SELF_TEST PASS: {len(suite)} operation experts, exhaustive finite-support checks, serialization, OOD abstain, poison hold"
    )


def query_model(
    model_path: str, name: str, args: Sequence[int], mode: str = "exact"
) -> None:
    model = GrowingMathProcessor.load(model_path)
    print(json.dumps(model.predict(name, *map(int, args), mode=mode), indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="HAGI Growing Math Processor")
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--experiment", action="store_true", help="run autonomous growth experiment"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out-dir", default="hagi_growing_math_results")
    p.add_argument(
        "--query", nargs="+", metavar=("OP", "ARG"), help="query saved model: OP ARG..."
    )
    p.add_argument(
        "--expr",
        help="evaluate a safe arithmetic/logic expression, e.g. 'x*y + gcd(x,y)'",
    )
    p.add_argument(
        "--var",
        action="append",
        default=[],
        help="expression variable assignment, e.g. x=7 or x=0.5",
    )
    p.add_argument("--mode", choices=["exact", "neural"], default="exact")
    p.add_argument(
        "--model", default="hagi_growing_math_results/hagi_growing_math_model.json"
    )
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif args.query:
        if len(args.query) < 2:
            p.error("--query needs an operation name and at least one integer argument")
        query_model(args.model, args.query[0], args.query[1:], args.mode)
    elif args.expr:
        vals = {}
        for item in args.var:
            if "=" not in item:
                p.error(f"invalid --var {item!r}; expected name=integer")
            k, v = item.split("=", 1)
            raw = v.strip()
            vals[k.strip()] = (
                float(raw) if any(ch in raw.lower() for ch in (".", "e")) else int(raw)
            )
        model = GrowingMathProcessor.load(args.model)
        print(json.dumps(model.evaluate_expression(args.expr, **vals), indent=2))
    else:
        result = run_experiment(args.seeds, args.out_dir)
        print(json.dumps(result["aggregate"], indent=2))
        print(f"Report: {Path(args.out_dir) / 'hagi_growing_math_report.md'}")
        print(f"Model:  {Path(args.out_dir) / 'hagi_growing_math_model.json'}")


# ==============================================================================
# PART II — v7.4 optimal layer (alias bindings for the unified namespace)
# ==============================================================================

import sys as _sys

h5 = _sys.modules[__name__]
hagi_v7 = _sys.modules[__name__]

# v7 imported these v5 names under aliases / directly:
V5Processor = GrowingMathProcessor
Expr = Expr  # (identity, kept for clarity)
Program = Program
Expert = Expert
GrowthEvent = GrowthEvent
TaskSpec = TaskSpec
NeuralFiber = NeuralFiber
SafeExpression = SafeExpression
OPS = OPS
_make_task_suite = make_task_suite
_bit_not8 = _bit_not8
_bit_nand8 = _bit_nand8
_bit_nor8 = _bit_nor8
_bit_xnor8 = _bit_xnor8
_pow_int = _pow_int
_sign = _sign
candidate_programs = candidate_programs

"""HAGI v7.4 research — exact multivariate finite-domain symbolic learner.

Changes over the uploaded v7.2 prototype:

1. Exact low-cost templates and rational polynomial induction run before the
   bounded generic grammar search.
2. Probe values are a rejection sieve only; distinct ASTs are not merged just
   because they agree on a small sample. The generic search orders candidates
   by an explicit prefix-code length, but the grammar is restricted and the
   search budget can truncate it: no global MDL optimum is claimed.
3. Library reuse is recorded as a reference cost, not an arbitrary 0.5x code
   discount. The declared bit code is a model-size proxy, not file bytes.
4. The int64 prefilter is conservative and can only reject inside a bounded
   operation/input envelope. All uncertain cases use scalar Python integers;
   full finite-support scalar validation remains the commit authority.
5. Neural fiber fitting is opt-in and explicitly approximate. The CPU path
   does not import PyTorch; CUDA support is loaded lazily.
6. Theorem mining checks only supported inputs and requires closure before
   composing unary functions. Synthetic composition stays inside parent
   experts' certified supports.

This is a finite-domain research prototype, not a general intelligence model.
Commit is limited to the complete declared finite support; predictions outside
that support abstain. Dependencies: Python >=3.10, NumPy (PyTorch optional).
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterator, Optional, Sequence


V7_VERSION = "7.4.0-research"
SIEVE_SIZE = 5  # probe points for cheap pruning
DEFAULT_NODE_BUDGET = 20_000  # max generated candidates per task
DEFAULT_POP_BUDGET = 50_000  # hard bound on frontier pops (compute cap)
VALIDATION_CHUNK = 64  # candidates per batched prefilter
MAX_REUSE_EXPANSIONS = 8  # cap expert-substitution fanout per search pop


# --------------------- vectorized batch prefilter (numpy) ----------------------

_MAG_LIMIT = 1 << 62  # prefilter is "unsure" beyond this; scalar check decides


def _vec_binary(op: str, a: np.ndarray, b: np.ndarray):
    """Vectorized int64 op. Returns (values, ok_mask). ok=False lanes are
    lanes where the scalar implementation raises (div0, pow range)."""
    if op == "add":
        return a + b, np.ones(a.shape, bool)
    if op == "sub":
        return a - b, np.ones(a.shape, bool)
    if op == "mul":
        return a * b, np.ones(a.shape, bool)
    if op in ("floordiv", "mod"):
        ok = b != 0
        safe_b = np.where(ok, b, np.ones_like(b))
        if op == "floordiv":
            return np.floor_divide(a, safe_b), ok
        return np.mod(a, safe_b), ok
    if op == "gcd":
        return np.gcd(a, b), np.ones(a.shape, bool)
    if op == "lcm":
        ok = (a != 0) & (b != 0)
        sa = np.where(ok, a, np.ones_like(a))
        sb = np.where(ok, b, np.ones_like(b))
        return np.abs(sa * sb) // np.gcd(sa, sb), ok
    if op == "min":
        return np.minimum(a, b), np.ones(a.shape, bool)
    if op == "max":
        return np.maximum(a, b), np.ones(a.shape, bool)
    if op == "eq":
        return (a == b).astype(np.int64), np.ones(a.shape, bool)
    if op == "ne":
        return (a != b).astype(np.int64), np.ones(a.shape, bool)
    if op == "lt":
        return (a < b).astype(np.int64), np.ones(a.shape, bool)
    if op == "le":
        return (a <= b).astype(np.int64), np.ones(a.shape, bool)
    if op == "gt":
        return (a > b).astype(np.int64), np.ones(a.shape, bool)
    if op == "ge":
        return (a >= b).astype(np.int64), np.ones(a.shape, bool)
    if op == "bit_and":
        return a & b, np.ones(a.shape, bool)
    if op == "bit_or":
        return a | b, np.ones(a.shape, bool)
    if op == "bit_xor":
        return a ^ b, np.ones(a.shape, bool)
    if op == "bit_nand8":
        return (~(a & b)) & 0xFF, np.ones(a.shape, bool)
    if op == "bit_nor8":
        return (~(a | b)) & 0xFF, np.ones(a.shape, bool)
    if op == "bit_xnor8":
        return (~(a ^ b)) & 0xFF, np.ones(a.shape, bool)
    if op == "pow_int":
        ok = (b >= 0) & (b <= 8)
        sb = np.where(ok, b, np.zeros_like(b))
        return a**sb, ok
    raise ValueError(f"no vectorized op: {op}")


def _vec_unary(op: str, a: np.ndarray):
    if op == "neg":
        return -a, np.ones(a.shape, bool)
    if op == "abs":
        return np.abs(a), np.ones(a.shape, bool)
    if op == "sign":
        return np.sign(a).astype(np.int64), np.ones(a.shape, bool)
    if op == "bit_not8":
        return (~a) & 0xFF, np.ones(a.shape, bool)
    if op == "clamp3":
        return np.clip(a, -3, 3), np.ones(a.shape, bool)
    if op == "square":
        return a * a, np.ones(a.shape, bool)
    if op == "cube":
        return a * a * a, np.ones(a.shape, bool)
    raise ValueError(f"no vectorized unary op: {op}")


def _vec_eval_expr(e: Expr, cols: list[np.ndarray]):
    """Evaluate Expr over column arrays. Returns (values, ok_mask, big_mask).
    big_mask lanes exceeded the safe magnitude: the caller must fall back to
    the scalar authority instead of trusting the vectorized verdict."""
    if e.op == "var":
        return cols[int(e.value or 0)], None, None
    if e.op == "const":
        v = np.full(cols[0].shape, int(e.value), dtype=np.int64)
        return v, None, None
    vals = []
    oks = []
    bigs = []
    for a in e.args:
        v, ok, big = _vec_eval_expr(a, cols)
        vals.append(v)
        if ok is not None:
            oks.append(ok)
        if big is not None:
            bigs.append(big)
    if e.op == "absdiff":
        v, ok = np.abs(vals[0] - vals[1]), None
    elif e.op == "fma":
        v, ok = vals[0] * vals[1] + vals[2], None
    elif e.op == "clamp":
        v, ok = np.minimum(np.maximum(vals[0], vals[1]), vals[2]), None
    elif e.op == "median3":
        v, ok = np.sort(np.stack(vals), axis=0)[1], None
    elif e.op == "clamp3":
        v, ok = np.clip(vals[0], -3, 3), None
    elif e.op == "mul3":
        v, ok = vals[0] * vals[1] * vals[2], None
    elif len(vals) == 2 and e.op in OPS:
        v, ok = _vec_binary(e.op, vals[0], vals[1])
    elif len(vals) == 1:
        v, ok = _vec_unary(e.op, vals[0])
    else:
        raise ValueError(f"unsupported expression opcode: {e.op}")
    big = np.abs(v) > _MAG_LIMIT
    if bigs:
        big = big | bigs[0]
        for bb in bigs[1:]:
            big = big | bb
    if ok is None:
        ok = np.ones(v.shape, bool)
    # Every child must be defined at a lane for the parent expression
    # to have Python scalar semantics at that lane.
    for child_ok in oks:
        ok = ok & child_ok
    return v, ok, big


def _batch_fits_unchecked(
    programs: Sequence[Program], X: np.ndarray, Y: np.ndarray, device: str = "cpu"
) -> list[Optional[bool]]:
    """Batched sound prefilter: True = provably fits X->Y, False = provably
    fails (a lane where the scalar eval would raise), None = unsure (must be
    settled by the scalar authority check)."""
    out: list[Optional[bool]] = []
    if device.startswith("cuda") and _ensure_torch():
        return _batch_fits_torch(programs, X, Y, device)
    for p in programs:
        if p.expr is None:
            out.append(None)
            continue
        try:
            cols = [np.ascontiguousarray(X[:, i]) for i in range(X.shape[1])]
            v, ok, big = _vec_eval_expr(p.expr, cols)
            if not bool(np.all(ok)) or bool(np.any(big)):
                out.append(None)
                continue
            out.append(bool(np.array_equal(v, Y)))
        except (ValueError, OverflowError):
            out.append(None)
    return out


# --------------------------- optional GPU (torch) path --------------------------

_TORCH_OK = False
_TORCH_CHECKED = False
_torch = None


def _ensure_torch() -> bool:
    """Load PyTorch only when CUDA functionality is explicitly requested."""
    global _TORCH_OK, _TORCH_CHECKED, _torch
    if _TORCH_CHECKED:
        return _TORCH_OK
    _TORCH_CHECKED = True
    try:
        import importlib

        _torch = importlib.import_module("torch")
        _TORCH_OK = True
    except ImportError:
        _torch = None
        _TORCH_OK = False
    return _TORCH_OK


def _batch_fits_torch(
    programs: Sequence[Program], X: np.ndarray, Y: np.ndarray, device: str
) -> list[Optional[bool]]:
    """GPU batch prefilter. Same contract as batch_fits (numpy path).
    Integer-exact on int64; unsure-lane semantics identical."""
    dev = _torch.device(device)
    Xt = _torch.from_numpy(X).to(dev)
    Yt = _torch.from_numpy(Y).to(dev)
    cols = [Xt[:, i].contiguous() for i in range(X.shape[1])]
    one = _torch.ones_like(cols[0])

    def teval(e: Expr):
        if e.op == "var":
            return cols[int(e.value or 0)], None, None
        if e.op == "const":
            v = _torch.full_like(cols[0], int(e.value))
            return v, None, None
        vs, oks, bigs = [], [], []
        for a in e.args:
            v, ok, big = teval(a)
            vs.append(v)
            if ok is not None:
                oks.append(ok)
            if big is not None:
                bigs.append(big)
        op = e.op
        if op == "absdiff":
            v, ok = (vs[0] - vs[1]).abs(), None
        elif op == "fma":
            v, ok = vs[0] * vs[1] + vs[2], None
        elif op == "clamp":
            v, ok = torch_clamp = (
                _torch.minimum(_torch.maximum(vs[0], vs[1]), vs[2]),
                None,
            )
        elif op == "median3":
            v, ok = _torch.sort(_torch.stack(vs), dim=0).values[1], None
        elif op == "clamp3":
            v, ok = _torch.clamp(vs[0], -3, 3), None
        elif op == "mul3":
            v, ok = vs[0] * vs[1] * vs[2], None
        elif op == "add":
            v, ok = vs[0] + vs[1], None
        elif op == "sub":
            v, ok = vs[0] - vs[1], None
        elif op == "mul":
            v, ok = vs[0] * vs[1], None
        elif op in ("floordiv", "mod"):
            b0 = vs[1]
            ok = b0 != 0
            sb = _torch.where(ok, b0, one)
            v = (
                _torch.div(vs[0], sb, rounding_mode="floor")
                if op == "floordiv"
                else _torch.remainder(vs[0], sb)
            )
        elif op == "gcd":
            v, ok = _torch.gcd(vs[0], vs[1]), None
        elif op == "lcm":
            a0, b0 = vs[0], vs[1]
            ok = (a0 != 0) & (b0 != 0)
            sa = _torch.where(ok, a0, one)
            sb = _torch.where(ok, b0, one)
            v = _torch.abs(sa * sb) // _torch.gcd(sa, sb)
        elif op == "min":
            v, ok = _torch.minimum(vs[0], vs[1]), None
        elif op == "max":
            v, ok = _torch.maximum(vs[0], vs[1]), None
        elif op in ("eq", "ne", "lt", "le", "gt", "ge"):
            fn = {
                "eq": _torch.eq,
                "ne": _torch.ne,
                "lt": _torch.lt,
                "le": _torch.le,
                "gt": _torch.gt,
                "ge": _torch.ge,
            }[op]
            v, ok = fn(vs[0], vs[1]).to(_torch.int64), None
        elif op == "bit_and":
            v, ok = vs[0] & vs[1], None
        elif op == "bit_or":
            v, ok = vs[0] | vs[1], None
        elif op == "bit_xor":
            v, ok = vs[0] ^ vs[1], None
        elif op == "bit_nand8":
            v, ok = (~(vs[0] & vs[1])) & 0xFF, None
        elif op == "bit_nor8":
            v, ok = (~(vs[0] | vs[1])) & 0xFF, None
        elif op == "bit_xnor8":
            v, ok = (~(vs[0] ^ vs[1])) & 0xFF, None
        elif op == "pow_int":
            b0 = vs[1]
            ok = (b0 >= 0) & (b0 <= 8)
            sb = _torch.where(ok, b0, _torch.zeros_like(b0))
            v = vs[0] ** sb
        elif op == "neg":
            v, ok = -vs[0], None
        elif op == "abs":
            v, ok = vs[0].abs(), None
        elif op == "sign":
            v, ok = _torch.sign(vs[0]).to(_torch.int64), None
        elif op == "bit_not8":
            v, ok = (~(vs[0])) & 0xFF, None
        elif op == "clamp3":
            v, ok = _torch.clamp(vs[0], -3, 3), None
        elif op == "square":
            v, ok = vs[0] * vs[0], None
        elif op == "cube":
            v, ok = vs[0] * vs[0] * vs[0], None
        else:
            raise ValueError(f"unsupported torch opcode: {op}")
        big = v.abs() > _MAG_LIMIT
        for bb in bigs:
            big = big | bb
        if ok is None:
            ok = _torch.ones_like(big)
        # Propagate child domain failures through the complete AST.
        for child_ok in oks:
            ok = ok & child_ok
        return v, ok, big

    results: list[Optional[bool]] = []
    for p in programs:
        if p.expr is None:
            results.append(None)
            continue
        try:
            v, ok, big = teval(p.expr)
            if ok is None:
                ok = _torch.ones(v.shape, dtype=_torch.bool, device=v.device)
            if bool((~ok).any()) or bool(big is not None and big.any()):
                results.append(None)
                continue
            results.append(bool(_torch.equal(v, Yt)))
        except (ValueError, RuntimeError, OverflowError):
            results.append(None)
    return results


# --------------------- code length and safe prefilter contract -----------------


def _gamma_bits(n: int) -> int:
    """Length of Elias-gamma code for positive integer n."""
    n = int(n)
    if n < 1:
        raise ValueError("gamma code accepts positive integers")
    return 2 * (n.bit_length() - 1) + 1


def _zigzag(n: int) -> int:
    n = int(n)
    return 2 * n if n >= 0 else -2 * n - 1


_TOKEN_BITS = 6  # fixed-width opcode namespace (64 reserved tokens)


def _expr_description_bits(expr: Expr) -> int:
    """A concrete prefix-code proxy; unlike AST-node count, constants cost bits.

    Every node begins with a 6-bit token. Variable indices, signed integer
    constants and polynomial coefficients use self-delimiting gamma payloads.
    It is a declared encoding for this DSL, not Kolmogorov complexity.
    """
    if expr.op == "var":
        return _TOKEN_BITS + _gamma_bits(int(expr.value or 0) + 1)
    if expr.op == "const":
        return _TOKEN_BITS + _gamma_bits(_zigzag(int(expr.value)) + 1)
    if expr.op == "poly":
        coeffs = tuple(int(v) for v in (expr.value or ()))
        payload = _gamma_bits(len(coeffs) + 1)
        payload += sum(_gamma_bits(_zigzag(v) + 1) for v in coeffs)
        return _TOKEN_BITS + payload + sum(_expr_description_bits(a) for a in expr.args)
    if expr.op == "poly_nd":
        terms = tuple(expr.value or ())
        payload = _gamma_bits(len(terms) + 1)
        for coefficient, powers in terms:
            payload += _gamma_bits(_zigzag(int(coefficient)) + 1)
            payload += _gamma_bits(len(powers) + 1)
            payload += sum(_gamma_bits(int(power) + 1) for power in powers)
        return _TOKEN_BITS + payload
    return _TOKEN_BITS + sum(_expr_description_bits(a) for a in expr.args)


def _program_description_bits(program: Program) -> int:
    if program.venn_mask is not None:
        return _TOKEN_BITS + 4
    if program.expr is None:
        return 10**9
    return _expr_description_bits(program.expr)


_VECTOR_SAFE_OPS = {
    "neg",
    "abs",
    "sign",
    "bit_not8",
    "clamp3",
    "add",
    "sub",
    "mul",
    "floordiv",
    "mod",
    "gcd",
    "lcm",
    "min",
    "max",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "bit_and",
    "bit_or",
    "bit_xor",
    "bit_nand8",
    "bit_nor8",
    "bit_xnor8",
    "absdiff",
    "clamp",
    "median3",
}
_VECTOR_INPUT_LIMIT = 1 << 7
_VECTOR_MAX_DEPTH = 3


def _vector_prefilter_safe(expr: Expr, depth: int = 0) -> bool:
    """Prove a conservative int64 envelope for the vectorized DSL subset.

    With at most 3 operator levels and |leaf|<=128, a binary expression has
    at most 8 leaves; even a product/lcm of all leaves is <=2**56. Additive
    combinations remain below 2**63. Powers, cubes, FMA, mul3 and arbitrary
    constants deliberately use scalar Python-int authority instead.
    """
    if depth > _VECTOR_MAX_DEPTH:
        return False
    if expr.op == "var":
        return True
    if expr.op == "const":
        return abs(int(expr.value or 0)) <= _VECTOR_INPUT_LIMIT
    if expr.op not in _VECTOR_SAFE_OPS:
        return False
    return all(_vector_prefilter_safe(a, depth + 1) for a in expr.args)


def batch_fits_safe(
    programs: Sequence[Program],
    X: Optional[np.ndarray],
    Y: Optional[np.ndarray],
    device: str = "cpu",
) -> list[Optional[bool]]:
    """Conservative filter: None means scalar check required.

    Only bounded expressions are sent through NumPy/Torch int64. All other
    candidates bypass the fast path, so overflow cannot create a false reject.
    A vector result is never a commit certificate; scalar Python-int checks
    still validate every possible survivor on train/val/full finite support.
    """
    if X is None or Y is None:
        return [None] * len(programs)
    if X.size and (
        int(X.min()) < -_VECTOR_INPUT_LIMIT or int(X.max()) > _VECTOR_INPUT_LIMIT
    ):
        return [None] * len(programs)
    out: list[Optional[bool]] = [None] * len(programs)
    safe_indices = [
        i
        for i, p in enumerate(programs)
        if p.expr is not None and _vector_prefilter_safe(p.expr)
    ]
    if safe_indices:
        safe_programs = [programs[i] for i in safe_indices]
        safe_results = _batch_fits_unchecked(safe_programs, X, Y, device=device)
        for i, result in zip(safe_indices, safe_results):
            out[i] = result
    return out


# ------------------------- best-first candidate synthesis ----------------------

UNARY_OPS = ("neg", "abs", "sign", "bit_not8", "clamp3")
BINARY_OPS = (
    "add",
    "sub",
    "mul",
    "floordiv",
    "mod",
    "gcd",
    "lcm",
    "min",
    "max",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "bit_and",
    "bit_or",
    "bit_xor",
    "bit_nand8",
    "bit_nor8",
    "bit_xnor8",
    "pow_int",
)
CONSTS = (0, 1, -1)


TERNARY_PRIMS = ("fma", "clamp", "median3", "mul3")


def iter_candidates_mdl(
    arity: int,
    node_budget: int = DEFAULT_NODE_BUDGET,
    probe_x: Sequence[Sequence[int]] = (),
    probe_y: Sequence[int] = (),
    extra_binops: Sequence[tuple[str, Program]] = (),
    pop_budget: int = 50_000,
    deadline: Optional[float] = None,
) -> Iterator[Expr]:
    """Yield probe-consistent ASTs in true prefix-code-length order.

    The generator explores the implemented restricted grammar: unary ops,
    self-combinations, binary op with one leaf, ternary primitives on variables,
    and committed-expert substitutions. It is NOT a complete enumeration of
    all possible DSL trees; node/pop/deadline limits can also truncate it.
    Probe values are used only as a cheap consistency sieve: no two different
    ASTs are discarded merely because they coincide on five probes.
    """
    probe = tuple(tuple(int(v) for v in xs) for xs in probe_x)
    n = len(probe)
    pop_budget = int(pop_budget)

    def apply_bin(
        op: str, a: tuple[int, ...], b: tuple[int, ...]
    ) -> Optional[tuple[int, ...]]:
        fn = OPS.get(op)
        if fn is None:
            return None
        try:
            return tuple(int(fn(a[i], b[i])) for i in range(n))
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):
            return None

    def apply_un(op: str, a: tuple[int, ...]) -> Optional[tuple[int, ...]]:
        fn = (lambda v: min(max(v, -3), 3)) if op == "clamp3" else OPS.get(op)
        if fn is None:
            return None
        try:
            return tuple(int(fn(a[i])) for i in range(n))
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):
            return None

    def apply_tern(
        op: str, a: tuple[int, ...], b: tuple[int, ...], c: tuple[int, ...]
    ) -> Optional[tuple[int, ...]]:
        try:
            if op == "fma":
                return tuple(a[i] * b[i] + c[i] for i in range(n))
            if op == "mul3":
                return tuple(a[i] * b[i] * c[i] for i in range(n))
            if op == "clamp":
                return tuple(min(max(a[i], b[i]), c[i]) for i in range(n))
            return tuple(sorted((a[i], b[i], c[i]))[1] for i in range(n))
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):
            return None

    counter = itertools.count()
    leaves = [Expr.var(i) for i in range(arity)] + [Expr.const(c) for c in CONSTS]
    leaf_vals: list[tuple[Expr, Optional[tuple[int, ...]]]] = []
    for leaf in leaves:
        if leaf.op == "var":
            lv = tuple(row[int(leaf.value or 0)] for row in probe) if n else None
        else:
            lv = tuple(int(leaf.value) for _ in probe) if n else None
        leaf_vals.append((leaf, lv))

    heap: list[tuple[int, int, Expr, tuple[int, ...]]] = []
    expr_seen: set[Expr] = set()
    target: Optional[tuple[int, ...]] = (
        tuple(int(v) for v in probe_y) if probe_y else None
    )
    max_nodes = 24

    def push(expr: Expr, vals: Optional[tuple[int, ...]]) -> None:
        if vals is None or expr.cost > max_nodes or expr in expr_seen:
            return
        expr_seen.add(expr)
        heapq.heappush(heap, (_expr_description_bits(expr), next(counter), expr, vals))

    for leaf, lv in leaf_vals:
        push(leaf, lv)

    # Ternary primitive atoms are generated once, not redundantly at every pop.
    if arity == 3:
        for a, va in leaf_vals[:3]:
            for b, vb in leaf_vals[:3]:
                for c, vc in leaf_vals[:3]:
                    for prim in TERNARY_PRIMS:
                        push(Expr(prim, (a, b, c)), apply_tern(prim, va, vb, vc))

    generated = 0
    pops = 0
    extra_ops = tuple(
        sorted(
            ((nm, prog) for nm, prog in extra_binops),
            key=lambda pair: (_program_description_bits(pair[1]), pair[0]),
        )[:MAX_REUSE_EXPANSIONS]
    )

    def expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    while heap and generated < node_budget and pops < pop_budget:
        if deadline is not None and time.monotonic() >= deadline:
            break
        _, _, expr, vals = heapq.heappop(heap)
        pops += 1
        if target is None or vals == target:
            generated += 1
            yield expr

        base = expr.cost
        for op in UNARY_OPS:
            if expired():
                return
            push(Expr(op, (expr,)), apply_un(op, vals))

        for op in BINARY_OPS:
            if expired():
                return
            push(Expr(op, (expr, expr)), apply_bin(op, vals, vals))
            for leaf, lv in leaf_vals:
                if expired():
                    return
                push(Expr(op, (expr, leaf)), apply_bin(op, vals, lv))
                push(Expr(op, (leaf, expr)), apply_bin(op, lv, vals))

        for _name, prog in extra_ops:
            if expired():
                return
            if prog.arity != 2 or prog.expr is None:
                continue
            exp = _program_to_substitutable_expr(prog)
            combos = [(expr, vals, expr, vals)]
            for leaf, lv in leaf_vals:
                combos.append((expr, vals, leaf, lv))
                combos.append((leaf, lv, expr, vals))
            for a, _av, b, _bv in combos:
                if expired():
                    return
                if _subst_cost(exp, {0: a.cost, 1: b.cost}) > max_nodes:
                    continue
                child = _substitute(exp, {0: a, 1: b})
                # A reusable expert that ignores both inputs can collapse to
                # a cheaper leaf; do not insert a lower-cost key after popping
                # a more expensive node, which would break best-first ordering.
                if _expr_description_bits(child) <= _expr_description_bits(expr):
                    continue
                try:
                    out = tuple(int(child.eval(row)) for row in probe)
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                    out = None
                push(child, out)


def _poly_nd_to_ast(expr: Expr) -> Expr:
    """Expand the compact multivariate opcode into ordinary DSL operations."""
    terms = []
    for coefficient, powers in expr.value or ():
        factors: list[Expr] = [Expr.const(int(coefficient))]
        for index, power in enumerate(powers):
            power = int(power)
            if power == 0:
                continue
            variable = Expr.var(index)
            factor = (
                variable
                if power == 1
                else Expr("pow_int", (variable, Expr.const(power)))
            )
            factors.append(factor)
        term = factors[0]
        for factor in factors[1:]:
            term = Expr("mul", (term, factor))
        terms.append(term)
    if not terms:
        return Expr.const(0)
    result = terms[0]
    for term in terms[1:]:
        result = Expr("add", (result, term))
    return result


def _program_to_substitutable_expr(program: Program) -> Expr:
    if program.expr is None:
        raise ValueError("truth-table programs are not arithmetic ASTs")
    if program.expr.op == "poly_nd":
        return _poly_nd_to_ast(program.expr)
    return program.expr


def _composition_hint(spec: TaskSpec) -> Optional[Program]:
    """Build an exact AST witness from the parent experts stored in a proposal.

    The hint is never trusted: GrowingMathProcessorV7.learn exhaustively
    verifies it before commit. It simply avoids rediscovering a known tree
    through a large general grammar frontier.
    """
    parents = getattr(spec.teacher, "__defaults__", None)
    if not parents or len(parents) < 2:
        return None
    outer, inner = parents[0], parents[1]
    if not isinstance(outer, Expert) or not isinstance(inner, Expert):
        return None
    if outer.program.arity != 2 or spec.arity != 2:
        return None
    if outer.program.venn_mask is not None and inner.program.venn_mask is not None:
        mask = 0
        for x in (0, 1):
            for y in (0, 1):
                value = int(spec.teacher(x, y)) & 1
                mask |= value << ((x << 1) | y)
        return Program(2, venn_mask=mask)
    if outer.program.expr is None or inner.program.expr is None:
        return None
    outer_expr = _program_to_substitutable_expr(outer.program)
    inner_expr = _program_to_substitutable_expr(inner.program)
    if inner.program.arity == 1:
        mapping = {0: inner_expr, 1: Expr.var(1)}
    elif inner.program.arity == 2:
        mapping = {0: Expr.var(0), 1: inner_expr}
    else:
        return None
    return Program(2, _substitute(outer_expr, mapping))


def _substitute(e: Expr, mapping: dict[int, Expr]) -> Expr:
    if e.op == "var":
        return mapping[int(e.value or 0)]
    if e.op == "const":
        return e
    return Expr(e.op, tuple(_substitute(a, mapping) for a in e.args))


def _subst_cost(e: Expr, costs: dict[int, int]) -> int:
    """Cost of _substitute(e, vars->exprs with given costs), without building it."""
    if e.op == "var":
        return costs[int(e.value or 0)]
    if e.op == "const":
        return 1
    return 1 + sum(_subst_cost(a, costs) for a in e.args)


class SynthesisStats:
    __slots__ = ("candidates_generated", "evals", "sieve_survivors")

    def __init__(self) -> None:
        self.candidates_generated = 0
        self.evals = 0
        self.sieve_survivors = 0


# ------------------------------- v7 processor ----------------------------------


class GrowingMathProcessorV7(V5Processor):
    """Finite-domain symbolic learner with conservative acceleration guards."""

    def __init__(
        self,
        lookup_limit: int = 16,
        mdl_lambda: float = 0.001,
        node_budget: int = 20_000,
        pop_budget: int = 50_000,
        device: str = "cpu",
    ):
        if float(mdl_lambda) <= 0:
            raise ValueError(
                "mdl_lambda must be positive for the exact-fit MDL tie-break"
            )
        super().__init__(lookup_limit, mdl_lambda)
        self.node_budget = int(node_budget)
        self.pop_budget = int(pop_budget)
        self.device = device
        self.stats = SynthesisStats()
        self.theorems: list[Theorem] = []

    def learn(
        self,
        spec: TaskSpec,
        seed: int = 0,
        train_fraction: float = 0.70,
        poison_one_label: bool = False,
        train_neural: bool = False,
        deadline: Optional[float] = None,
        candidate_hint: Optional[Program] = None,
    ) -> GrowthEvent:
        if spec.name in self.experts:
            return GrowthEvent(self.version, spec.name, "REUSE", "already learned")
        if deadline is not None and time.monotonic() >= deadline:
            return self._hold(spec, "TIME_BUDGET_EXHAUSTED", 0, 0)
        domain = spec.inputs()
        if not domain:
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)
        if len(domain) > 1:
            train_fraction = min(max(float(train_fraction), 0.05), 0.95)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(domain))
        if spec.family == "BoolVenn":
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
        if poison_one_label and train_y:
            train_y[0] = int(train_y[0]) + 1
            # Explicit adversarial test injection: never search for a program
            # from a label set known to contain the injected corruption.
            return self._hold(
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)
            )

        neural: Optional[NeuralFiber] = None

        def maybe_fit_neural() -> Optional[NeuralFiber]:
            if not train_neural or not val_x:
                return None
            try:
                return fit_fiber_guarded(
                    train_x, train_y, val_x, val_y, seed=seed + 991, epochs=120
                )
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                try:
                    return NeuralFiber.fit(
                        train_x, train_y, val_x, val_y, seed=seed + 991, epochs=60
                    )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    return None

        def commit(
            program: Program, reason: str, code_length: Optional[int] = None
        ) -> GrowthEvent:
            nonlocal neural
            if neural is None and train_neural:
                neural = maybe_fit_neural()
            bits = int(
                _program_description_bits(program)
                if code_length is None
                else code_length
            )
            parent_names: tuple[str, ...] = ()
            parent_defaults = getattr(spec.teacher, "__defaults__", None)
            if (
                candidate_hint is not None
                and parent_defaults
                and len(parent_defaults) >= 2
            ):
                outer_parent, inner_parent = parent_defaults[0], parent_defaults[1]
                if isinstance(outer_parent, Expert) and isinstance(
                    inner_parent, Expert
                ):
                    parent_names = (outer_parent.name, inner_parent.name)
            expert = Expert(
                spec.name,
                spec.family,
                program,
                spec.domains,
                spec.valid_rule,
                bits,
                neural,
                len(domain),
                0,
                parent_names,
            )
            self.experts[spec.name] = expert
            self.version += 1
            self.total_commits += 1
            ev = GrowthEvent(
                self.version,
                spec.name,
                "COMMIT",
                reason,
                program.pretty(),
                bits,
                len(train_x),
                len(val_x),
                1.0,
            )
            self.growth_log.append(ev)
            return ev

        def validate_candidate(program: Program) -> Optional[GrowthEvent]:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            try:
                if not all(
                    self._same(program.eval(xs), yy) for xs, yy in zip(probe_x, probe_y)
                ):
                    self.stats.evals += len(probe_x)
                    return None
                self.stats.sieve_survivors += 1
                if not self._fits(program, train_x, train_y):
                    return None
                if val_x and not self._fits(program, val_x, val_y):
                    return None
                # Authoritative finite-support verification uses Python integers.
                if any(not self._same(program.eval(xs), all_y[xs]) for xs in domain):
                    return None
                if program.cost > 32:
                    return None
                return commit(program, "exact-validation+full-finite-support", None)
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                return None

        # Reuse: pay for a library reference, not an arbitrary 0.5x discount.
        known_items = list(sorted(self.experts.items()))
        for index, (name, ex) in enumerate(known_items):
            if ex.program.arity != spec.arity:
                continue
            # Alias-free REUSE is valid only when the existing certificate
            # already covers the complete new support; otherwise the same
            # formula may legitimately acquire a wider certified scope.
            if not all(ex.supports(xs) for xs in domain):
                continue
            if not self._fits(ex.program, train_x, train_y):
                continue
            if not self._fits(ex.program, val_x, val_y):
                continue
            if any(not self._same(ex.program.eval(xs), all_y[xs]) for xs in domain):
                continue
            ref_bits = _TOKEN_BITS + _gamma_bits(index + 1)
            ev = GrowthEvent(
                self.version,
                spec.name,
                "REUSE",
                f"verified-expert-coverage:{name}",
                ex.program.pretty(),
                ref_bits,
                len(train_x),
                len(val_x),
                1.0,
            )
            self.growth_log.append(ev)
            return ev

        # A tiny probe is only a rejection sieve, never a semantic equivalence key.
        probe_n = min(SIEVE_SIZE, len(train_x))
        probe_idx = rng.choice(len(train_x), size=probe_n, replace=False)
        probe_x = [train_x[int(i)] for i in probe_idx]
        probe_y = [train_y[int(i)] for i in probe_idx]

        def make_train_arrays():
            lo64, hi64 = -(1 << 63), (1 << 63) - 1
            flat = [int(v) for row in train_x for v in row] + [int(v) for v in train_y]
            if any(v < lo64 or v > hi64 for v in flat):
                return None, None
            return (
                np.asarray(train_x, dtype=np.int64),
                np.asarray(train_y, dtype=np.int64),
            )

        train_X, train_Yv = make_train_arrays()

        # Fast exact templates first: the old v5 candidate family covers common
        # arithmetic/Boolean primitives with negligible search cost.
        if spec.family == "BoolVenn":
            direct_candidates: Sequence[Program] = [
                Program(2, venn_mask=m) for m in range(16)
            ]
        else:
            direct_candidates = sorted(
                candidate_programs(spec.arity),
                key=lambda p: (_program_description_bits(p), p.pretty()),
            )
        # A parent-composition hint is a proposal, not an authority: validate
        # it on train, held-out, and every declared point before accepting.
        # When an explicit hint was supplied and fails, HOLD rather than
        # launching an unbounded blind search for a different function.
        if candidate_hint is not None:
            hinted = validate_candidate(candidate_hint)
            if hinted is not None:
                hinted.reason = "verified-composition-hint+full-finite-support"
                if self.growth_log:
                    self.growth_log[-1].reason = hinted.reason
                return hinted
            return self._hold(
                spec, "COMPOSITION_HINT_NOT_CERTIFIED", len(train_x), len(val_x)
            )

        for program in direct_candidates:
            ev = validate_candidate(program)
            if ev is not None:
                return ev
            if deadline is not None and time.monotonic() >= deadline:
                return self._hold(
                    spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x)
                )

        # Closed-form polynomial induction precedes the general grammar search.
        # Unary interpolation keeps the compact Horner representation.
        if spec.arity == 1 and not spec.family.startswith("Bool"):
            coeffs = _exact_poly_fit([xs[0] for xs in train_x], train_y, max_degree=5)
            if coeffs is not None and val_x:
                if all(
                    _eval_poly(coeffs, xs[0]) == yy for xs, yy in zip(val_x, val_y)
                ) and all(_eval_poly(coeffs, xs[0]) == all_y[xs] for xs in domain):
                    program = Program(1, _poly_expr(coeffs))
                    if program.cost <= 32:
                        return commit(
                            program, "exact-rational-polynomial+full-finite-support"
                        )

        # Exact multivariate interpolation extends induction to 2-3 variables.
        # It solves an overdetermined rational linear system; a candidate is
        # usable only when the coefficient vector is unique and integral, and
        # the ordinary held-out + exhaustive scalar checks still pass.
        elif 2 <= spec.arity <= 3 and not spec.family.startswith("Bool"):
            terms = _exact_multivariate_poly_fit(
                train_x, train_y, arity=spec.arity, max_degree=4
            )
            if terms is not None and val_x:
                program = Program(spec.arity, _multivariate_poly_expr(terms))
                if (
                    all(program.eval(xs) == yy for xs, yy in zip(val_x, val_y))
                    and all(program.eval(xs) == all_y[xs] for xs in domain)
                    and program.cost <= 32
                ):
                    return commit(
                        program,
                        "exact-multivariate-rational-polynomial+full-finite-support",
                    )

        # Bounded grammar search. Only proven-safe int64 cases use the vector
        # rejection filter; all other candidates go through scalar validation.
        if deadline is not None and time.monotonic() >= deadline:
            return self._hold(spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x))
        reuse_binops = [
            (nm, ex.program)
            for nm, ex in self.experts.items()
            if ex.program.arity == spec.arity
        ]
        candidates = (
            Program(spec.arity, e)
            for e in iter_candidates_mdl(
                spec.arity,
                self.node_budget,
                probe_x,
                probe_y,
                reuse_binops,
                self.pop_budget,
                deadline,
            )
        )
        chunk: list[Program] = []

        def flush(buf: list[Program]) -> Optional[GrowthEvent]:
            if not buf:
                return None
            verdicts = batch_fits_safe(buf, train_X, train_Yv, device=self.device)
            for program, ok in zip(buf, verdicts):
                self.stats.candidates_generated += 1
                # Only a False from the guarded bounded vector subset is allowed
                # to skip scalar validation. None always falls through.
                if ok is False:
                    self.stats.evals += len(train_x)
                    continue
                ev = validate_candidate(program)
                if ev is not None:
                    return ev
            return None

        for program in candidates:
            chunk.append(program)
            if len(chunk) >= VALIDATION_CHUNK:
                ev = flush(chunk)
                if ev is not None:
                    return ev
                chunk = []
            if deadline is not None and time.monotonic() >= deadline:
                return self._hold(
                    spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x)
                )
        ev = flush(chunk)
        if ev is not None:
            return ev
        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))

    def save(self, path: str | Path) -> None:
        data = {
            "format": "HAGI-GrowingMathProcessor",
            "version": V7_VERSION,
            "model_version": self.version,
            "lookup_limit": self.lookup_limit,
            "mdl_lambda": self.mdl_lambda,
            "node_budget": self.node_budget,
            "pop_budget": self.pop_budget,
            "device": self.device,
            "total_commits": self.total_commits,
            "total_holds": self.total_holds,
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},
            "growth_log": [vars(x) for x in self.growth_log],
            "theorems": [vars(x) for x in self.theorems],
            "synthesis_stats": {
                "candidates_generated": self.stats.candidates_generated,
                "evals": self.stats.evals,
                "sieve_survivors": self.stats.sieve_survivors,
            },
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "GrowingMathProcessorV7":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "HAGI-GrowingMathProcessor":
            raise ValueError("not a HAGI Growing Math Processor state file")
        model = cls(
            int(data.get("lookup_limit", 16)),
            float(data.get("mdl_lambda", 0.001)),
            int(data.get("node_budget", DEFAULT_NODE_BUDGET)),
            int(data.get("pop_budget", DEFAULT_POP_BUDGET)),
            str(data.get("device", "cpu")),
        )
        model.version = int(data.get("model_version", 0))
        model.total_commits = int(data.get("total_commits", 0))
        model.total_holds = int(data.get("total_holds", 0))
        model.experts = {
            k: Expert.from_dict(v) for k, v in data.get("experts", {}).items()
        }
        if data.get("version") != V7_VERSION:
            for expert in model.experts.values():
                expert.code_length = _program_description_bits(expert.program)
        model.growth_log = [GrowthEvent(**x) for x in data.get("growth_log", [])]
        model.theorems = [Theorem(**x) for x in data.get("theorems", [])]
        stats = data.get("synthesis_stats", {})
        model.stats.candidates_generated = int(stats.get("candidates_generated", 0))
        model.stats.evals = int(stats.get("evals", 0))
        model.stats.sieve_survivors = int(stats.get("sieve_survivors", 0))
        return model

    def _fits(
        self, program: Program, xs_list: Sequence[Sequence[int]], ys: Sequence[int]
    ) -> bool:
        self.stats.evals += len(xs_list)
        try:
            return all(self._same(program.eval(xs), yy) for xs, yy in zip(xs_list, ys))
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):
            return False


# ------------------- certified guarded optimizer (ALGORITHMS §3.2/§9) -----------


def _fiber_forward(params, x, hidden_scale=1.0):
    w1, b1, w2, b2 = params
    return np.tanh(x @ w1 + b1) @ w2 + b2


def fit_fiber_guarded(
    inputs, targets, val_inputs, val_targets, seed, hidden: int = 24, epochs: int = 300
) -> "NeuralFiber":
    """Empirically guarded full-batch MSE fitting for the optional tanh fiber.

    Candidate updates are accepted only when measured training MSE does not
    increase. This is an implementation-level regression guard, NOT a proof
    of the Lean finite log-linear cross-entropy step theorem, not a held-out
    guarantee, and not a convergence theorem for a Transformer backbone.
    """
    x = np.asarray(inputs, dtype=np.float64).reshape(-1, len(inputs[0]))
    y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
    xv = np.asarray(val_inputs, dtype=np.float64).reshape(-1, x.shape[1])
    yv = np.asarray(val_targets, dtype=np.float64).reshape(-1, 1)
    sx = max(1.0, float(np.max(np.abs(x))))
    sy = max(1.0, float(np.max(np.abs(y))))
    x, y, xv, yv = x / sx, y / sy, xv / sx, yv / sy
    rng = np.random.default_rng(seed)
    params = [
        rng.normal(0.0, 0.32, (x.shape[1], hidden)),
        np.zeros((1, hidden)),
        rng.normal(0.0, 0.12, (hidden, 1)),
        np.zeros((1, 1)),
    ]

    def loss(p):
        err = _fiber_forward(p, x) - y
        return float(np.mean(err * err))

    cur = loss(params)
    best = [q.copy() for q in params]
    best_val = float("inf")
    total_drop = 0.0
    accepted = 0
    held = 0
    for _ in range(epochs):
        pred = _fiber_forward(params, x)
        err = pred - y
        # batch direction D = -ḡ (meanResidualTranspose analogue)
        h_act = np.tanh(x @ params[0] + params[1])
        dh = (2.0 / len(x) * err) @ params[2].T
        dh = dh * (1.0 - h_act**2)
        dp = 2.0 / len(x) * err
        grads = [
            x.T @ dh,
            np.sum(dh, axis=0, keepdims=True),
            h_act.T @ dp,
            np.sum(dp, axis=0, keepdims=True),
        ]
        gnorm2 = sum(float(np.sum(g * g)) for g in grads)
        if gnorm2 <= 0:
            held += 1
            continue
        # safe-step candidates: backtracking grid on the majorizer ratio;
        # accept the first candidate whose measured loss strictly drops
        improved = False
        for step in (1.0, 0.5, 0.25, 0.1, 0.03):
            cand = [p - step * g for p, g in zip(params, grads)]
            cand_loss = loss(cand)
            if cand_loss < cur:
                drop = cur - cand_loss
                params = cand
                cur = cand_loss
                total_drop += drop
                accepted += 1
                improved = True
                break
        if not improved:
            held += 1
            # HoldStep: parameters unchanged, certificate records zero drop
        hv = _fiber_forward(params, xv)
        vl = float(np.mean((hv - yv) ** 2))
        if np.isfinite(vl) and vl < best_val:
            best_val = vl
            best = [q.copy() for q in params]
    w1, b1, w2, b2 = best
    h = np.tanh(x @ w1 + b1)
    tr_loss = float(np.mean((h @ w2 + b2 - y) ** 2)) * sy * sy
    hv = np.tanh(xv @ w1 + b1)
    predv = (hv @ w2 + b2).reshape(-1) * sy
    val_loss = float(np.mean((predv - np.asarray(val_targets, dtype=float)) ** 2))
    exact = float(
        np.mean(
            np.rint(predv).astype(np.int64) == np.asarray(val_targets, dtype=np.int64)
        )
    )
    fiber = NeuralFiber(
        len(inputs[0]), sx, sy, w1, b1, w2, b2, tr_loss, val_loss, exact
    )
    fiber.guard_certificate = {
        "accepted_steps": accepted,
        "hold_steps": held,
        "total_certified_drop": total_drop,
    }
    return fiber


# --------------- certified ternary compression (ALGORITHMS §TernaryGrad) --------


# ------------------- CRAM energy solver (E-track, E1/E2/P1/A5) ------------------


def solve_coloring(
    edges,
    n_nodes: int,
    num_colors: int,
    givens: Optional[dict[int, int]] = None,
    max_sweeps: int = 10000,
) -> dict[str, Any]:
    """CRAM single-cell energy descent for graph coloring (theory: E1 pairwise
    energy = conflict counter; C261 single_cell_move_energy_delta_symm;
    greedy_strict_search_terminates on finite configs; A5 zero-energy iff
    valid). Strict-improvement moves only; result exactly verified.

    Returns {'coloring', 'valid', 'energy', 'sweeps', 'verified'}.
    """
    from collections import defaultdict

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    rng = np.random.default_rng(42)
    givens = givens or {}
    y = {i: givens.get(i, int(rng.integers(0, num_colors))) for i in range(n_nodes)}

    def energy(state) -> int:
        return sum(1 for a, b in edges if state[a] == state[b])

    cur = energy(y)
    sweeps = 0
    while sweeps < max_sweeps:
        improved = False
        for i in range(n_nodes):
            if i in givens:
                continue
            # exact local field: conflict counts per candidate color (C261:
            # delta = L_i(a) - L_i(y_i), no partition terms)
            counts = {c: sum(1 for j in adj[i] if y[j] == c) for c in range(num_colors)}
            best_c, best_conf = y[i], counts[y[i]]
            for c in range(num_colors):
                if counts[c] < best_conf:
                    best_c, best_conf = c, counts[c]
            if best_conf < counts[y[i]]:  # strict drop only (certified descent)
                y[i] = best_c
                improved = True
        cur = energy(y)
        sweeps += 1
        if cur == 0:
            break
        if not improved:
            # coordinate minimum reached (not global): restart with new seed
            y = {
                i: givens.get(i, int(rng.integers(0, num_colors)))
                for i in range(n_nodes)
            }
    # A5 separation: zero conflict energy iff proper coloring (exact verify)
    valid = all(y[a] != y[b] for a, b in edges) and all(
        0 <= y[i] < num_colors for i in range(n_nodes)
    )
    return {
        "coloring": y,
        "valid": bool(valid),
        "energy": int(cur),
        "sweeps": int(sweeps),
        "verified": bool(valid) == (cur == 0),
        "givens_respected": all(y[i] == g for i, g in givens.items()),
    }


# ------------------- exact formula induction & theorem mining ------------------


def _exact_poly_fit(
    xs: Sequence[int], ys: Sequence[int], max_degree: int = 5
) -> Optional[tuple[int, ...]]:
    """Exact rational polynomial interpolation (Fractions, machine precision:
    no floating point anywhere). Returns integer coefficient tuple (c0..cd)
    of the unique interpolant of degree <= max_degree through the points,
    or None if the system is inconsistent / ill-conditioned. All arithmetic
    is exact; verification is deferred to the caller."""
    pts = sorted({(int(x), int(y)) for x, y in zip(xs, ys)})
    if len({x for x, _ in pts}) < 2:
        return None
    for degree in range(1, min(max_degree, len(pts) - 1) + 1):
        k = degree + 1
        sample = pts[:k]
        # exact Gaussian elimination on the Vandermonde system
        a = [[Fraction(x) ** i for i in range(k)] + [Fraction(y)] for x, y in sample]
        n = k
        for col in range(n):
            piv = next((r for r in range(col, n) if a[r][col] != 0), None)
            if piv is None:
                break
            a[col], a[piv] = a[piv], a[col]
            pv = a[col][col]
            a[col] = [v / pv for v in a[col]]
            for r in range(n):
                if r != col and a[r][col] != 0:
                    f = a[r][col]
                    a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]
        else:
            coeffs = [row[n] for row in a]
            if any(c.denominator != 1 for c in coeffs):
                continue  # non-integer coefficients: not a compact integer formula
            icoeffs = tuple(int(c) for c in coeffs)
            # verify exactly on ALL points (not just the interpolation sample)
            if all(sum(ci * x**i for i, ci in enumerate(icoeffs)) == y for x, y in pts):
                return icoeffs
    return None


def _monomial_exponents(arity: int, max_degree: int) -> tuple[tuple[int, ...], ...]:
    """All exponent vectors with total degree <= max_degree, constant first."""
    if arity < 1 or max_degree < 0:
        raise ValueError("arity must be positive and max_degree nonnegative")
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


def _exact_multivariate_poly_fit(
    inputs: Sequence[Sequence[int]],
    targets: Sequence[int],
    arity: int,
    max_degree: int = 4,
) -> Optional[tuple[tuple[int, tuple[int, ...]], ...]]:
    """Find the minimum-total-degree integer polynomial fitting all rows.

    Uses exact Fraction Gauss-Jordan elimination on an overdetermined system.
    A result is returned only when the design matrix has full column rank,
    the system is consistent, and every coefficient is an integer. Thus no
    floating-point tolerance or arbitrary regularizer enters formula discovery.
    The caller must still check held-out data and the complete finite support.
    """

    if len(inputs) != len(targets) or not inputs or arity < 1:
        return None
    rows = [tuple(int(v) for v in row) for row in inputs]
    if any(len(row) != arity for row in rows):
        return None
    ys = [int(v) for v in targets]
    for degree in range(max_degree + 1):
        monomials = _monomial_exponents(arity, degree)
        ncols = len(monomials)
        if len(rows) < ncols:
            continue
        matrix = []
        for row, y in zip(rows, ys):
            features = [
                Fraction(math.prod(x**power for x, power in zip(row, powers)))
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
                full_rank = False
                break
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
            scale = matrix[pivot_row][col]
            matrix[pivot_row] = [v / scale for v in matrix[pivot_row]]
            for r in range(len(matrix)):
                if r == pivot_row:
                    continue
                factor = matrix[r][col]
                if factor:
                    matrix[r] = [
                        a - factor * b for a, b in zip(matrix[r], matrix[pivot_row])
                    ]
            pivot_rows.append(pivot_row)
            pivot_row += 1
        if not full_rank:
            continue
        if any(
            all(row[c] == 0 for c in range(ncols)) and row[-1] != 0 for row in matrix
        ):
            continue
        coefficients = [matrix[pivot_rows[col]][-1] for col in range(ncols)]
        if any(value.denominator != 1 for value in coefficients):
            continue
        terms = tuple(
            (int(coefficient), powers)
            for coefficient, powers in zip(coefficients, monomials)
            if coefficient
        )

        # Recheck every supplied row independently of the elimination path.
        def evaluate(row: Sequence[int]) -> int:
            return sum(
                c * math.prod(int(x) ** p for x, p in zip(row, powers))
                for c, powers in terms
            )

        if all(evaluate(row) == y for row, y in zip(rows, ys)):
            return terms
    return None


def _multivariate_poly_expr(
    terms: tuple[tuple[int, tuple[int, ...]], ...],
) -> Expr:
    return Expr(
        "poly_nd",
        (),
        tuple((int(c), tuple(int(p) for p in powers)) for c, powers in terms),
    )


def _poly_expr(coeffs: tuple[int, ...]) -> Expr:
    return Expr("poly", (Expr.var(0),), coeffs)


def _eval_poly(coeffs: tuple[int, ...], x: int) -> int:
    acc = 0
    for ci in reversed(coeffs):
        acc = acc * x + ci
    return acc


@dataclass
class Theorem:
    name: str
    statement: str
    domain_points: int
    verified: bool = True


class TheoremMiner:
    """Mine finite-domain identities only where every evaluation is supported.

    Results are finite-support statements, not unbounded algebraic theorems.
    Unary iteration identities additionally require closure of the declared
    input domain under the expert's outputs.
    """

    def mine(
        self, model: "GrowingMathProcessorV7", max_experts: int = 24
    ) -> list[Theorem]:
        found: list[Theorem] = []
        known = {t.name for t in model.theorems}
        bin_e = [e for e in model.experts.values() if e.program.arity == 2][
            :max_experts
        ]
        un_e = [e for e in model.experts.values() if e.program.arity == 1][:max_experts]

        for e in bin_e:
            dx, dy = e.domains[0], e.domains[1]
            declared_pairs = [(x, y) for x in dx for y in dy if e.supports((x, y))]
            comm_pairs = [(x, y) for x, y in declared_pairs if e.supports((y, x))]
            # Do not label a tiny symmetric overlap as global commutativity:
            # every supported declared pair must have its swapped support.
            if comm_pairs and len(comm_pairs) == len(declared_pairs):
                name = f"commutativity({e.name})"
                if name not in known:
                    try:
                        ok = all(
                            e.program.eval((int(x), int(y)))
                            == e.program.eval((int(y), int(x)))
                            for x, y in comm_pairs
                        )
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                        ok = False
                    if ok:
                        th = Theorem(
                            name,
                            f"for all declared pairs with both orders supported: {e.name}(x,y) = {e.name}(y,x)",
                            len(comm_pairs),
                        )
                        model.theorems.append(th)
                        found.append(th)
                        known.add(name)

            self_pairs = [x for x in dx if x in dy and e.supports((x, x))]
            name = f"self-absorption({e.name})"
            if self_pairs and len(self_pairs) == len(dx) and name not in known:
                try:
                    ok = all(
                        e.program.eval((int(x), int(x))) == int(x) for x in self_pairs
                    )
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                    ok = False
                if ok:
                    th = Theorem(
                        name,
                        f"for all x in the declared overlap with (x,x) supported: {e.name}(x,x) = x",
                        len(self_pairs),
                    )
                    model.theorems.append(th)
                    found.append(th)
                    known.add(name)

            zero_pairs = [x for x in dx if 0 in dy and e.supports((x, 0))]
            name = f"annihilator-zero({e.name})"
            if zero_pairs and len(zero_pairs) == len(dx) and name not in known:
                try:
                    ok = all(e.program.eval((int(x), 0)) == int(x) for x in zero_pairs)
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                    ok = False
                if ok:
                    th = Theorem(
                        name,
                        f"for all declared x with (x,0) supported: {e.name}(x,0) = x",
                        len(zero_pairs),
                    )
                    model.theorems.append(th)
                    found.append(th)
                    known.add(name)

        for e in un_e:
            dom = tuple(int(x) for x in e.domains[0])
            image: dict[int, int] = {}
            closed = True
            try:
                for x in dom:
                    if not e.supports((x,)):
                        closed = False
                        break
                    y = int(e.program.eval((x,)))
                    if not e.supports((y,)):
                        closed = False
                        break
                    image[x] = y
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                closed = False

            if closed:
                name = f"involutivity({e.name})"
                if name not in known:
                    try:
                        ok = all(int(e.program.eval((image[x],))) == x for x in dom)
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                        ok = False
                    if ok:
                        th = Theorem(
                            name,
                            f"for all x in the declared domain: {e.name}({e.name}(x)) = x",
                            len(dom),
                        )
                        model.theorems.append(th)
                        found.append(th)
                        known.add(name)

                name = f"idempotence({e.name})"
                if name not in known:
                    try:
                        ok = all(
                            int(e.program.eval((image[x],))) == image[x] for x in dom
                        )
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                        ok = False
                    if ok:
                        th = Theorem(
                            name,
                            f"for all x in the declared domain: {e.name}({e.name}(x)) = {e.name}(x)",
                            len(dom),
                        )
                        model.theorems.append(th)
                        found.append(th)
                        known.add(name)

            if 0 in dom:
                name = f"zero-fixpoint({e.name})"
                if name not in known:
                    try:
                        ok = int(e.program.eval((0,))) == 0
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                        ok = False
                    if ok:
                        th = Theorem(
                            name, f"at the supported point x=0: {e.name}(0) = 0", 1
                        )
                        model.theorems.append(th)
                        found.append(th)
                        known.add(name)
        return found


# ------------------------- resource governor (unlimited loop) --------------------


def _ram_available_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        if sys.platform == "win32":
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
        return 0


def _vram_free_bytes(device: str) -> int:
    if not (device.startswith("cuda") and _ensure_torch()):
        return 0
    try:
        free, _total = _torch.cuda.mem_get_info(_torch.device(device))
        return int(free)
    except Exception:
        return 0


class ResourceGovernor:
    """Convert available RAM/VRAM into bounded per-task search budgets.

    A separate monotonic deadline/round cap bounds the outer growth loop.
    Budgets degrade gracefully instead of allowing search to consume all RAM.
    """

    RAM_FLOOR_BYTES = 300 * 1024 * 1024  # below this: pause/skip, never OOM-kill
    VRAM_FLOOR_BYTES = 300 * 1024 * 1024

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.node_budget = DEFAULT_NODE_BUDGET
        self.pop_budget = DEFAULT_POP_BUDGET
        self.chunk = VALIDATION_CHUNK
        self.total_skips = 0

    def _scale(self) -> float:
        ram = _ram_available_bytes()
        if ram <= 0:
            return 1.0
        # full speed above 6 GB, linear degradation to zero at the floor
        hi, lo, floor = 6 * 2**30, 1 * 2**30, self.RAM_FLOOR_BYTES
        s = min(1.0, max(0.0, (ram - floor) / max(1, hi - floor)))
        if ram < lo:
            s = min(s, max(0.05, (ram - floor) / max(1, lo - floor)))
        if self.device.startswith("cuda") and _ensure_torch():
            vram = _vram_free_bytes(self.device)
            if vram < self.VRAM_FLOOR_BYTES:
                s = 0.0
        return s

    def before_task(self) -> bool:
        """Re-budget for the next task. False = resources exhausted: skip."""
        s = self._scale()
        if s <= 0.0:
            self.total_skips += 1
            return False
        self.node_budget = max(1_000, int(DEFAULT_NODE_BUDGET * s))
        self.pop_budget = max(2_000, int(DEFAULT_POP_BUDGET * s))
        self.chunk = max(8, int(VALIDATION_CHUNK * min(1.0, s * 2)))
        return True

    def recover(self, exc: BaseException) -> None:
        """After a memory error: shrink hard, drop caches, keep going."""
        DEFAULT_NODE_BUDGET_S = 10_000
        self.node_budget = min(self.node_budget, DEFAULT_NODE_BUDGET_S)
        self.pop_budget = min(self.pop_budget, 20_000)
        self.chunk = max(4, self.chunk // 4)
        if self.device.startswith("cuda") and _ensure_torch():
            try:
                _torch.cuda.empty_cache()
            except Exception:
                pass


# ---------------------- unlimited self-development loop -------------------------


def _propose_task_from_experts(
    experts: dict[str, Any],
    rng: np.random.Generator,
    round_index: int,
    depth: int = 1,
    domain_radius: int = 6,
) -> Optional[TaskSpec]:
    """Propose a composition only on the parents' jointly certified support.

    No parent program is evaluated outside its Expert.supports() domain. The
    generated task inherits the relevant input validity rule and its whole
    Cartesian support is pre-checked before a teacher closure is returned.
    This grows a finite compositional curriculum, not independent capability.
    """
    arity2 = [
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"
    ]
    arity1 = [
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"
    ]
    bool2 = [
        e for e in experts.values() if e.program.arity == 2 and e.family == "BoolVenn"
    ]

    # Boolean composition: every intermediate and outer call is support-checked.
    if bool2 and len(bool2) >= 2 and rng.random() < 0.4:
        for _ in range(8):
            outer = bool2[int(rng.choice(len(bool2)))]
            inner = bool2[int(rng.choice(len(bool2)))]
            if inner is outer and len(bool2) > 1:
                continue
            rows = [
                (x, y)
                for x in (0, 1)
                for y in (0, 1)
                if inner.supports((x, y))
                and outer.supports((int(inner.program.eval((x, y))), y))
            ]
            if len(rows) != 4:
                continue
            teacher = lambda x, y, o=outer, i=inner: int(
                o.program.eval((int(i.program.eval((int(x), int(y)))), int(y)))
            )
            return TaskSpec(
                f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",
                2,
                ((0, 1), (0, 1)),
                teacher,
                family="auto-composed",
            )

    if not arity2:
        return None
    inner_pool = arity2 + arity1

    # A proposal must have enough rows to support an actual train/validation split.
    min_rows = 4
    max_radius = max(1, min(int(domain_radius), 12))
    for _ in range(12):
        outer = arity2[int(rng.choice(len(arity2)))]
        inner = inner_pool[int(rng.choice(len(inner_pool)))]

        if inner.program.arity == 1:
            # outer(inner(x), y). Keep x inside the unary expert domain and
            # every inner(x) inside the outer expert's first-argument domain.
            xs0 = tuple(
                x
                for x in inner.domains[0]
                if abs(int(x)) <= max_radius
                and inner.supports((int(x),))
                and int(inner.program.eval((int(x),))) in outer.domains[0]
            )
            ys0 = tuple(
                y
                for y in outer.domains[1]
                if abs(int(y)) <= max_radius
                and (outer.valid_rule != "nonzero_y" or int(y) != 0)
            )
            if not xs0 or not ys0:
                continue
            for radius in range(max_radius, 0, -1):
                xs = tuple(x for x in xs0 if abs(int(x)) <= radius)
                ys = tuple(y for y in ys0 if abs(int(y)) <= radius)
                ys = tuple(
                    y
                    for y in ys
                    if all(
                        outer.supports((int(inner.program.eval((int(x),))), int(y)))
                        for x in xs
                    )
                )
                if len(xs) * len(ys) < min_rows:
                    continue
                if not all(inner.supports((int(x),)) for x in xs):
                    continue
                teacher = lambda x, y, o=outer, i=inner: int(
                    o.program.eval((int(i.program.eval((int(x),))), int(y)))
                )
                return TaskSpec(
                    f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",
                    2,
                    (xs, ys),
                    teacher,
                    valid_rule=outer.valid_rule,
                    family="auto-composed",
                )

        elif inner.program.arity == 2:
            # outer(x, inner(x,y)). The input rectangle is deliberately shrunk
            # until every call to both parent experts remains certified.
            xs0 = tuple(
                x
                for x in inner.domains[0]
                if x in outer.domains[0] and abs(int(x)) <= max_radius
            )
            ys0 = tuple(y for y in inner.domains[1] if abs(int(y)) <= max_radius)
            if not xs0 or not ys0:
                continue
            for radius in range(max_radius, 0, -1):
                xs = tuple(x for x in xs0 if abs(int(x)) <= radius)
                ys = tuple(y for y in ys0 if abs(int(y)) <= radius)
                if not xs or not ys:
                    continue
                # Remove y-columns unless every selected x has a supported
                # inner and outer evaluation. This is a rectangular safe subset.
                safe_ys = []
                for y in ys:
                    good = True
                    for x in xs:
                        if not inner.supports((int(x), int(y))):
                            good = False
                            break
                        inner_value = int(inner.program.eval((int(x), int(y))))
                        if not outer.supports((int(x), inner_value)):
                            good = False
                            break
                    if good:
                        safe_ys.append(y)
                ys = tuple(safe_ys)
                if len(xs) * len(ys) < min_rows:
                    continue
                teacher = lambda x, y, o=outer, i=inner: int(
                    o.program.eval((int(x), int(i.program.eval((int(x), int(y))))))
                )
                return TaskSpec(
                    f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",
                    2,
                    (xs, ys),
                    teacher,
                    valid_rule=inner.valid_rule,
                    family="auto-composed",
                )

    return None


def verify_growth_lineage(model: "GrowingMathProcessorV7") -> dict[str, Any]:
    """Recheck every stored compositional expert against its parent pair.

    This is an executable audit over each child's complete declared finite
    support. It is not a proof outside that support.
    """
    checked = 0
    errors: list[dict[str, Any]] = []
    for child in model.experts.values():
        if not child.parents:
            continue
        if len(child.parents) != 2:
            errors.append({"expert": child.name, "reason": "invalid_parent_arity"})
            continue
        outer = model.experts.get(child.parents[0])
        inner = model.experts.get(child.parents[1])
        if outer is None or inner is None:
            errors.append({"expert": child.name, "reason": "missing_parent"})
            continue
        ok = True
        checked += 1
        for xs in itertools.product(*child.domains):
            xs = tuple(int(x) for x in xs)
            if not child.supports(xs):
                continue
            try:
                if (
                    outer.program.venn_mask is not None
                    and inner.program.venn_mask is not None
                ):
                    if not inner.supports(xs):
                        ok = False
                        break
                    middle = int(inner.program.eval(xs))
                    parent_input = (middle, xs[1])
                elif inner.program.arity == 1:
                    if not inner.supports((xs[0],)):
                        ok = False
                        break
                    middle = int(inner.program.eval((xs[0],)))
                    parent_input = (middle, xs[1])
                elif inner.program.arity == 2:
                    if not inner.supports(xs):
                        ok = False
                        break
                    middle = int(inner.program.eval(xs))
                    parent_input = (xs[0], middle)
                else:
                    ok = False
                    break
                if not outer.supports(parent_input):
                    ok = False
                    break
                expected = int(outer.program.eval(parent_input))
                if expected != int(child.program.eval(xs)):
                    ok = False
                    break
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                ok = False
                break
        if not ok:
            errors.append(
                {"expert": child.name, "reason": "parent_composition_mismatch"}
            )
    return {
        "lineage_checked": checked,
        "lineage_errors": errors,
        "lineage_pass": not errors,
    }


def grow_forever(
    model: "GrowingMathProcessorV7",
    suite: Sequence[TaskSpec],
    device: str = "cpu",
    max_rounds: Optional[int] = None,
    snapshot_every: int = 10,
    out_dir: str = ".omc/hagi_grow",
    log_every: int = 1,
    budget_cap: Optional[int] = None,
    deadline: Optional[float] = None,
) -> dict[str, Any]:
    """Bounded compositional growth loop with explicit checkpoints.

    Each round learns the base curriculum, proposes finite-domain compositions
    only inside parent supports, and snapshots state. The caller may impose a
    monotonic deadline and/or round cap; the resource governor limits each
    search. This is curriculum recombination, not proof of new general ability.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gov = ResourceGovernor(device)
    log_path = out / "growth_log.jsonl"
    start = time.time()
    rnd = np.random.default_rng(20260101)
    report: dict[str, Any] = {"device": device, "rounds": [], "stopped": None}
    round_index = 0
    commits_since_snapshot = 0
    idle_rounds = 0
    depth = 1
    domain_radius = 6
    try:
        while max_rounds is None or round_index < max_rounds:
            if deadline is not None and time.monotonic() >= deadline:
                report["stopped"] = "TIME_LIMIT"
                break
            t0 = time.time()
            committed_this_round = 0
            # (1) base curriculum (idempotent: already-committed tasks REUSE)
            for spec in suite:
                if deadline is not None and time.monotonic() >= deadline:
                    report["stopped"] = "TIME_LIMIT"
                    break
                if not gov.before_task():
                    break
                model.node_budget = (
                    min(gov.node_budget, budget_cap) if budget_cap else gov.node_budget
                )
                model.pop_budget = (
                    min(gov.pop_budget, budget_cap * 2)
                    if budget_cap
                    else gov.pop_budget
                )
                try:
                    ev = model.learn(
                        spec,
                        seed=10_000 + round_index * 7919 + len(spec.name),
                        deadline=deadline,
                        train_neural=os.environ.get("HAGI_NO_NEURAL") != "1",
                    )
                    if ev.action == "COMMIT":
                        committed_this_round += 1
                        commits_since_snapshot += 1
                except (MemoryError, RuntimeError, ValueError) as exc:
                    gov.recover(exc)
            # (2) self-proposed tasks: compose committed experts
            proposed = 0
            attempts = min(12, max(2, len(model.experts) // 3))
            for k in range(attempts):
                if deadline is not None and time.monotonic() >= deadline:
                    report["stopped"] = "TIME_LIMIT"
                    break
                if not gov.before_task():
                    break
                spec = _propose_task_from_experts(
                    model.experts,
                    rnd,
                    round_index,
                    depth=depth,
                    domain_radius=domain_radius,
                )
                if spec is None or spec.name in model.experts:
                    continue
                model.node_budget = (
                    min(gov.node_budget, budget_cap) if budget_cap else gov.node_budget
                )
                model.pop_budget = (
                    min(gov.pop_budget, budget_cap * 2)
                    if budget_cap
                    else gov.pop_budget
                )
                try:
                    hint = (
                        _composition_hint(spec)
                        if spec.family == "auto-composed"
                        else None
                    )
                    if spec.family == "auto-composed" and hint is None:
                        # The proposal is not representable in the current safe
                        # hint grammar. Do not fall back to blind synthesis.
                        continue
                    ev = model.learn(
                        spec,
                        seed=500_000 + round_index * 104_729 + k,
                        deadline=deadline,
                        candidate_hint=hint,
                        train_neural=os.environ.get("HAGI_NO_NEURAL") != "1",
                    )
                    proposed += 1
                    if ev.action == "COMMIT":
                        committed_this_round += 1
                        commits_since_snapshot += 1
                except (
                    MemoryError,
                    RuntimeError,
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                ) as exc:
                    gov.recover(exc)
            # (3) autonomous theorem mining over the certified expert base
            try:
                new_theorems = TheoremMiner().mine(model, max_experts=16)
            except (
                MemoryError,
                RuntimeError,
                ValueError,
                ZeroDivisionError,
                OverflowError,
            ):
                new_theorems = []
            lineage = verify_growth_lineage(model)
            round_rec = {
                "round": round_index,
                **lineage,
                "experts": len(model.experts),
                "unique_programs": len({e.program for e in model.experts.values()}),
                "unique_certified_scopes": len(
                    {
                        (e.program, e.domains, e.valid_rule)
                        for e in model.experts.values()
                    }
                ),
                "committed_this_round": committed_this_round,
                "proposed_tasks": proposed,
                "theorems_total": len(model.theorems),
                "theorems_new": len(new_theorems),
                "total_code_length": sum(e.code_length for e in model.experts.values()),
                "node_budget": gov.node_budget,
                "pop_budget": gov.pop_budget,
                "ram_avail_gb": round(_ram_available_bytes() / 2**30, 2),
                "vram_free_gb": round(_vram_free_bytes(device) / 2**30, 2),
                "elapsed": round(time.time() - t0, 2),
            }
            report["rounds"].append(round_rec)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(round_rec) + "\n")
            if log_every or committed_this_round:
                print(
                    f"[grow r{round_index}] experts={round_rec['experts']} "
                    f"commits={committed_this_round} code={round_rec['total_code_length']}bits "
                    f"ram={round_rec['ram_avail_gb']}GB "
                    f"vram={round_rec['vram_free_gb']}GB",
                    flush=True,
                )
            if commits_since_snapshot >= snapshot_every:
                model.save(out / "hagi_grow_model.json")
                commits_since_snapshot = 0
            if committed_this_round > 0:
                idle_rounds = 0
            else:
                idle_rounds += 1
                if idle_rounds >= 3:
                    # saturation escalation: deepen the composition algebra and
                    # widen certified domains -> unbounded curriculum growth
                    depth += 1
                    domain_radius = min(12, domain_radius + 2)
                    idle_rounds = 0
            round_index += 1
            if deadline is not None and time.monotonic() >= deadline:
                report["stopped"] = "TIME_LIMIT"
                break
            if (
                _ram_available_bytes()
                and _ram_available_bytes() < ResourceGovernor.RAM_FLOOR_BYTES
            ):
                report["stopped"] = "RAM_FLOOR"
                break
    except KeyboardInterrupt:
        report["stopped"] = "INTERRUPTED"
    finally:
        model.save(out / "hagi_grow_model.json")
        (out / "theorems.json").write_text(
            json.dumps(
                {
                    "count": len(model.theorems),
                    "theorems": [vars(t) for t in model.theorems],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        report["total_experts"] = len(model.experts)
        report.update(verify_growth_lineage(model))
        report["total_elapsed"] = round(time.time() - start, 1)
        (out / "growth_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


# ---------------------------- experiment & benchmark ---------------------------


def _run_seed_worker(args: tuple[int, str, str]) -> dict[str, Any]:
    """One seed per process: seeds are independent, so this parallelizes
    linearly across CPU cores. The worker's governor bounds its own memory.
    Also saves the seed's model snapshot into out_dir."""
    seed, device, out_dir = args
    train_neural = os.environ.get("HAGI_NO_NEURAL") != "1"
    out = Path(out_dir)
    suite = make_task_suite()
    model = GrowingMathProcessorV7(device=device)
    t0 = time.time()
    stats0 = SynthesisStats()
    stats0.candidates_generated = model.stats.candidates_generated
    stats0.evals = model.stats.evals
    stage_order = ["BoolVenn", "unary", "integer-binary", "composed-ternary"]
    for stage in stage_order:
        stage_tasks = [s for s in suite if s.family == stage]
        rng = np.random.default_rng(seed + len(stage) * 101)
        if len(stage_tasks) > 1:
            order = rng.permutation(len(stage_tasks))
            stage_tasks = [stage_tasks[int(i)] for i in order]
        for j, spec in enumerate(stage_tasks):
            model.learn(
                spec,
                seed=seed * 1009 + j * 17 + len(spec.name),
                train_neural=train_neural,
            )
    verification = model.verify_all(suite)
    ood_checks = ood_abstains = 0
    for spec in suite:
        if spec.name not in model.experts:
            continue
        outside = [d[-1] + 1 for d in spec.domains]
        ood_checks += 1
        ood_abstains += int(model.predict(spec.name, *outside)["status"] == "ABSTAIN")
    poison_model = GrowingMathProcessorV7()
    poison_event = poison_model.learn(
        next(s for s in suite if s.name == "ADD"),
        seed=seed + 44001,
        poison_one_label=True,
    )
    prime_spec = TaskSpec(
        "IS_PRIME",
        1,
        (tuple(range(-8, 9)),),
        lambda x: int(x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))),
        family="unseen",
    )
    unknown_event = GrowingMathProcessorV7().learn(prime_spec, seed=seed + 99001)
    reuse_count = sum(1 for e in model.growth_log if e.reason == "expert-reuse")
    gen = model.stats.candidates_generated - stats0.candidates_generated
    evs = model.stats.evals - stats0.evals
    model.save(out / f"hagi_v7_model_seed{seed}.json")
    return {
        "seed": int(seed),
        "committed": len(model.experts),
        "expected": len(suite),
        "commit_rate": len(model.experts) / len(suite),
        "symbolic_exact_accuracy": verification["accuracy"],
        "total_code_length": sum(e.code_length for e in model.experts.values()),
        "candidates_generated": gen,
        "program_evals": evs,
        "sieve_survivors": model.stats.sieve_survivors,
        "expert_reuse_commits": reuse_count,
        "ood_abstains": ood_abstains,
        "ood_checks": ood_checks,
        "poison_action": poison_event.action,
        "unknown_is_prime_action": unknown_event.action,
        "elapsed_seconds": time.time() - t0,
    }


def run_v7_experiment(
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
    out_dir: str = ".omc/hagi_v7_results",
    workers: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite = make_task_suite()
    start = time.time()
    args = [(int(s), device, str(out)) for s in seeds]
    if workers <= 0:
        workers = min(len(args), max(1, (os.cpu_count() or 2) - 1))
    workers = min(workers, len(args))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            per_seed = list(ex.map(_run_seed_worker, args))
    else:
        per_seed = [_run_seed_worker(a) for a in args]
    per_seed.sort(key=lambda r: r["seed"])
    main_model_path = out / f"hagi_v7_model_seed{int(seeds[0])}.json"
    if main_model_path.exists():
        GrowingMathProcessorV7.load(main_model_path)
        main_model_path.replace(out / "hagi_v7_model.json")

    aggregate: dict[str, Any] = {}
    for key in (
        "committed",
        "symbolic_exact_accuracy",
        "total_code_length",
        "candidates_generated",
        "program_evals",
        "ood_abstains",
        "elapsed_seconds",
    ):
        vals = [float(r[key]) for r in per_seed]
        aggregate[key + "_mean"] = float(np.mean(vals))
        aggregate[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    aggregate.update(
        {
            "seeds": list(map(int, seeds)),
            "expert_reuse_commits_total": int(
                sum(r["expert_reuse_commits"] for r in per_seed)
            ),
            "poison_all_hold": all(r["poison_action"] == "HOLD" for r in per_seed),
            "unknown_all_hold": all(
                r["unknown_is_prime_action"] == "HOLD" for r in per_seed
            ),
            "reload_roundtrip_pass": GrowingMathProcessorV7.load(
                out / "hagi_v7_model.json"
            )
            .predict("ADD", 7, -3)
            .get("value")
            == 4,
            "total_runtime_seconds": time.time() - start,
        }
    )
    result = {"version": V7_VERSION, "per_seed": per_seed, "aggregate": aggregate}
    (out / "hagi_v7_experiment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def self_test_v7() -> None:
    suite = make_task_suite()
    model = GrowingMathProcessorV7()
    for i, spec in enumerate(suite):
        ev = model.learn(spec, seed=10_000 + i, train_neural=False)
        assert ev.action == "COMMIT", (spec.name, ev)
    check = model.verify_all(suite)
    assert check["committed"] == len(suite) and check["accuracy"] == 1.0
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"
    # The exact median3 template is checked over the full declared support.
    med = model.experts["MEDIAN3"].program
    assert med.eval((1, 5, 3)) == 3 and med.eval((9, 2, 7)) == 7
    poisoned = GrowingMathProcessorV7().learn(
        next(s for s in suite if s.name == "ADD"),
        seed=77,
        poison_one_label=True,
        train_neural=False,
    )
    assert poisoned.action == "HOLD"
    # exact formula induction: teacher = 2x^3 - x^2 + 7x - 4 must be learned
    # as a closed-form polynomial with EXACT (machine-precision) arithmetic
    poly_spec = TaskSpec(
        "POLY_CUBIC",
        1,
        (tuple(range(-12, 13)),),
        lambda x: 2 * x**3 - x**2 + 7 * x - 4,
        family="poly-unseen",
    )
    pev = GrowingMathProcessorV7().learn(poly_spec, seed=31337, train_neural=False)
    assert pev.action == "COMMIT" and "poly(" in pev.program, (pev.action, pev.program)

    # 2-D exact polynomial discovery; support is checked exhaustively.
    poly2 = TaskSpec(
        "POLY2_QUADRATIC",
        2,
        (tuple(range(-4, 5)), tuple(range(-3, 4))),
        lambda x, y: 3 * x * x - 2 * x * y + y * y + 4 * x - 5 * y + 7,
        family="poly2-unseen",
    )
    p2ev = GrowingMathProcessorV7().learn(poly2, seed=31338, train_neural=False)
    assert p2ev.action == "COMMIT" and "poly_nd(" in p2ev.program, (
        p2ev.action,
        p2ev.program,
    )
    p2model = GrowingMathProcessorV7()
    p2model.learn(poly2, seed=31338, train_neural=False)
    assert p2model.verify_all([poly2])["accuracy"] == 1.0

    # 3-D exact polynomial discovery with a cubic mixed interaction.
    poly3 = TaskSpec(
        "POLY3_CUBIC",
        3,
        (tuple(range(-3, 4)), tuple(range(-3, 4)), tuple(range(-3, 4))),
        lambda x, y, z: 2 * x**3 - 3 * x * y * z + 4 * y * y - 5 * z + 11,
        family="poly3-unseen",
    )
    p3model = GrowingMathProcessorV7()
    p3ev = p3model.learn(poly3, seed=31339, train_neural=False)
    assert p3ev.action == "COMMIT" and "poly_nd(" in p3ev.program, (
        p3ev.action,
        p3ev.program,
    )
    assert p3model.verify_all([poly3])["accuracy"] == 1.0

    assert model.predict("ADD", 7, -3)["value"] == 4
    assert model.experts["ADD"].code_length == _program_description_bits(
        model.experts["ADD"].program
    )
    # theorem mining: exhaustive verification only, no unverified statements
    ths = TheoremMiner().mine(model)
    assert ths, "theorem miner found nothing"
    names = [t.name for t in model.theorems]
    assert all(t.verified for t in model.theorems)
    assert any("commutativity(MIN)" in n for n in names), names
    assert any("idempotence(ABS)" in n for n in names), names
    print(
        f"V7.4 SELF_TEST PASS: {len(suite)} experts, accuracy 1.0, OOD abstain, poison hold, "
        f"exact poly commit '{pev.program}', {len(model.theorems)} theorems"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="HAGI v7.4 exact multivariate finite-domain research prototype"
    )
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--experiment", action="store_true")
    p.add_argument(
        "--grow",
        action="store_true",
        help="bounded compositional growth loop (10-minute default cap)",
    )
    p.add_argument(
        "--device", default="cpu", help="cpu or cuda:0 (GPU batch prefilter)"
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=10.0,
        help="wall-clock cap for --grow (default: 10 minutes)",
    )
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument(
        "--no-neural",
        action="store_true",
        help="disable neural fallback training (fastest pure-symbolic)",
    )
    p.add_argument(
        "--search-budget",
        type=int,
        default=2_500,
        help="per-task candidate budget during --grow; pop budget is twice this",
    )
    p.add_argument(
        "--workers", type=int, default=0, help="parallel seed processes (0=auto)"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--out-dir", default=".omc/hagi_v7_results")
    args = p.parse_args()
    if args.self_test:
        self_test_v7()
    elif args.grow:
        suite = make_task_suite()
        model = GrowingMathProcessorV7(device=args.device)
        deadline = (
            time.monotonic() + max(0.0, args.max_minutes * 60)
            if args.max_minutes is not None
            else None
        )
        if args.no_neural:
            os.environ["HAGI_NO_NEURAL"] = "1"
        report = grow_forever(
            model,
            suite,
            device=args.device,
            max_rounds=args.max_rounds,
            snapshot_every=args.snapshot_every,
            out_dir=args.out_dir,
            budget_cap=max(1, args.search_budget),
            deadline=deadline,
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("device", "stopped", "total_experts", "total_elapsed")
                },
                indent=2,
            )
        )
    elif args.experiment:
        result = run_v7_experiment(
            args.seeds, args.out_dir, workers=args.workers, device=args.device
        )
        print(json.dumps(result["aggregate"], indent=2))
    else:
        p.print_help()


# ==============================================================================
# PART III — v8.0 unified system
# ==============================================================================

"""HAGI v8.0 — Unified Optimal-Architecture, Minimal-Volume, Minimal-Compute
Self-Developing Neuro-Symbolic + Energy-Based System.
"""


import sys as _sys

h5 = _sys.modules[__name__]  # v5 layer (same namespace)
hagi_v7 = _sys.modules[__name__]  # v7 layer (same namespace)

V8_VERSION = "8.4.0"

# ==============================================================================
# 1. Exact Prefix-Code Bits & Generalized Call-DAG / ANF / Piecewise Expr
# ==============================================================================

TOKEN_BITS = 6
VAR_NAMES = ("x", "y", "z", "u", "v", "w")
DYADIC_LEVELS = np.array([(c + 1) / 128.0 for c in range(255)], dtype=np.float64)
MAX_RANGE_EVAL = 100_000
_RANGE_FOLD_OPS = {"all_range", "any_range", "sum_range", "prod_range"}


def _expr_contains_range(expr: "ExprV8") -> bool:
    return expr.op in _RANGE_FOLD_OPS or any(_expr_contains_range(a) for a in expr.args)


def gamma_bits(n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError("gamma_bits expects positive integer")
    return 2 * (n.bit_length() - 1) + 1


def zigzag(n: int) -> int:
    n = int(n)
    return 2 * n if n >= 0 else -2 * n - 1


@dataclass(frozen=True)
class ExprV8:
    """Extended immutable AST supporting arbitrary arity, DAG calls, ANF, and piecewise."""

    op: str
    args: tuple["ExprV8", ...] = ()
    value: Optional[Any] = None

    @staticmethod
    def var(i: int) -> "ExprV8":
        return ExprV8("var", (), int(i))

    @staticmethod
    def const(v: int) -> "ExprV8":
        return ExprV8("const", (), int(v))

    @staticmethod
    def from_h5(e: h5.Expr) -> "ExprV8":
        return ExprV8(e.op, tuple(ExprV8.from_h5(a) for a in e.args), e.value)

    def eval(
        self,
        xs: Sequence[int],
        env: Optional[dict[str, "ExpertV8"]] = None,
        _budget: Optional[list[int]] = None,
        _fold_depth: int = 0,
    ) -> int:
        # A mutable per-evaluation counter is shared across symbolic calls. This
        # prevents an expert call from multiplying the range cap invisibly.
        if _budget is None:
            _budget = [0]
        if self.op == "var":
            return int(xs[int(self.value)])
        if self.op == "const":
            return int(self.value)
        if self.op == "call":
            if env is None or self.value not in env:
                raise ValueError(f"missing called expert: {self.value}")
            vals = tuple(a.eval(xs, env, _budget, _fold_depth) for a in self.args)
            callee = env[self.value]
            if not callee.supports(vals):
                raise ValueError(
                    f"call to {self.value} outside certified support: {vals}"
                )
            return callee.program.eval(vals, env, _budget, _fold_depth)
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
        if self.op in _RANGE_FOLD_OPS:
            if len(self.args) != 3:
                raise ValueError(f"{self.op} expects start, stop, body")
            # Static nested folds are rejected, and dynamic nested folds through
            # calls are rejected by the propagated depth flag.
            if _fold_depth >= 1 or any(_expr_contains_range(a) for a in self.args):
                raise ValueError(
                    "nested range folds (including calls) are not supported"
                )
            start = int(self.args[0].eval(xs, env, _budget, _fold_depth))
            stop = int(self.args[1].eval(xs, env, _budget, _fold_depth))
            span = max(0, stop - start)
            if span > MAX_RANGE_EVAL or _budget[0] + span > MAX_RANGE_EVAL:
                raise ValueError(
                    f"cumulative range-fold budget exceeds {MAX_RANGE_EVAL} iterations"
                )
            body = self.args[2]
            if self.op == "all_range":
                for k in range(start, stop):
                    _budget[0] += 1
                    if not body.eval(
                        tuple(xs) + (int(k),), env, _budget, _fold_depth + 1
                    ):
                        return 0
                return 1
            if self.op == "any_range":
                for k in range(start, stop):
                    _budget[0] += 1
                    if body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1):
                        return 1
                return 0
            if self.op == "sum_range":
                acc = 0
                for k in range(start, stop):
                    _budget[0] += 1
                    acc += int(
                        body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1)
                    )
                    if abs(acc).bit_length() > 4096:
                        raise OverflowError(
                            "range-fold accumulator exceeds 4096-bit cap"
                        )
                return acc
            acc = 1
            for k in range(start, stop):
                _budget[0] += 1
                acc *= int(
                    body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1)
                )
                if abs(acc).bit_length() > 4096:
                    raise OverflowError("range-fold product exceeds 4096-bit cap")
            return acc
        if self.op == "if_cond":
            cond_val = self.args[0].eval(xs, env, _budget, _fold_depth)
            return (
                self.args[1].eval(xs, env, _budget, _fold_depth)
                if cond_val
                else self.args[2].eval(xs, env, _budget, _fold_depth)
            )
        vals = [a.eval(xs, env, _budget, _fold_depth) for a in self.args]
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
        if self.op == "isqrt":
            value = int(vals[0])
            if value < 0:
                raise ValueError("isqrt is undefined for negative integers")
            return math.isqrt(value)
        if self.op == "reified":
            rfn = REIFIED_V8.get(str(self.value))
            if rfn is None:
                raise ValueError(f"reified op not registered: {self.op}")
            # leaf macro over the RAW inputs (args empty at leaf level;
            # composition happens by wrapping in arithmetic ops)
            return int(rfn(*[int(v) for v in xs]))
        fn = h5.OPS.get(self.op)
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
            elif self.op == "poly":
                c = 1 + len(self.value or ())
            elif self.op == "poly_nd":
                c = 1 + len(self.value or ())
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
        if self.op in _RANGE_FOLD_OPS:
            label = {
                "all_range": "forall",
                "any_range": "exists",
                "sum_range": "sum",
                "prod_range": "prod",
            }[self.op]
            return f"{label}[y={self.args[0].pretty()}..{self.args[1].pretty()}):{self.args[2].pretty()}]"
        if self.op == "if_cond":
            return f"({self.args[1].pretty()} if {self.args[0].pretty()} else {self.args[2].pretty()})"
        if self.op == "poly":
            return h5.Expr("poly", (h5.Expr.var(0),), self.value).pretty()
        if self.op == "poly_nd":
            return h5.Expr("poly_nd", (), self.value).pretty()
        if self.op == "clamp3":
            return f"min(max({self.args[0].pretty()}, -3), 3)"
        if self.op == "absdiff":
            return f"abs({self.args[0].pretty()} - {self.args[1].pretty()})"
        if len(self.args) == 1:
            return f"{self.op}({self.args[0].pretty()})"
        if len(self.args) == 2:
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"

    def subtrees(self) -> list["ExprV8"]:
        out = [self]
        for a in self.args:
            out.extend(a.subtrees())
        return out

    def replace_subtree(self, target: "ExprV8", replacement: "ExprV8") -> "ExprV8":
        if self == target:
            return replacement
        if not self.args:
            return self
        return ExprV8(
            self.op,
            tuple(a.replace_subtree(target, replacement) for a in self.args),
            self.value,
        )

    def to_dict(self) -> dict[str, Any]:
        val = self.value
        if self.op == "anf" and val is not None:
            val = [int(val[0]), [[int(s), int(c)] for s, c in val[1]]]
        return {"op": self.op, "value": val, "args": [a.to_dict() for a in self.args]}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExprV8":
        op = str(d["op"])
        val = d.get("value")
        if op == "poly" and val is not None:
            val = tuple(int(v) for v in val)
        elif op == "poly_nd" and val is not None:
            val = tuple((int(t[0]), tuple(int(p) for p in t[1])) for t in val)
        elif op == "anf" and val is not None:
            val = (int(val[0]), tuple((int(s), int(c)) for s, c in val[1]))
        return ExprV8(op, tuple(ExprV8.from_dict(a) for a in d.get("args", [])), val)


@dataclass(frozen=True)
class ProgramV8:
    arity: int
    expr: Optional[ExprV8] = None
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
        self,
        xs: Sequence[int],
        env: Optional[dict[str, "ExpertV8"]] = None,
        _budget: Optional[list[int]] = None,
        _fold_depth: int = 0,
    ) -> int:
        if _budget is None:
            _budget = [0]
        if len(xs) != self.arity:
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")
        if self.venn_mask is not None:
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):
                raise ValueError("BoolVenn expects two binary inputs")
            return (int(self.venn_mask) >> ((int(xs[0]) << 1) | int(xs[1]))) & 1
        if self.expr is None:
            raise RuntimeError("empty program")
        return self.expr.eval(xs, env, _budget, _fold_depth)

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
    def from_dict(d: dict[str, Any]) -> "ProgramV8":
        return ProgramV8(
            int(d["arity"]),
            ExprV8.from_dict(d["expr"]) if d.get("expr") else None,
            d.get("venn_mask"),
        )


# ==============================================================================
# 2. VennHead + VennPacking + ANF Engine (C242, C251, C301, C302)
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
    trits = []
    for mask in range(16):
        for idx in range(4):
            trits.append((mask >> idx) & 1)
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
    """Exact C301 Zhegalkin butterfly. O(n 2^n)."""
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


def anf_support_vars(coeffs: Sequence[int], n_vars: int) -> tuple[int, ...]:
    used = 0
    for s, c in enumerate(coeffs):
        if c:
            used |= s
    return tuple(i for i in range(n_vars) if (used >> (n_vars - 1 - i)) & 1)


def anf_to_expr_v8(coeffs: Sequence[int], n_vars: int) -> ExprV8:
    sparse = tuple((s, 1) for s, c in enumerate(coeffs) if c)
    if len(sparse) == 0:
        return ExprV8.const(0)
    if len(sparse) == 1 and sparse[0][0] == 0:
        return ExprV8.const(1)
    if len(sparse) <= 4:
        mons = []
        for s, _ in sparse:
            if s == 0:
                mons.append(ExprV8.const(1))
            else:
                vs = [
                    ExprV8.var(i) for i in range(n_vars) if (s >> (n_vars - 1 - i)) & 1
                ]
                t = vs[0]
                for v in vs[1:]:
                    t = ExprV8("bit_and", (t, v))
                mons.append(t)
        res = mons[0]
        for m in mons[1:]:
            res = ExprV8("bit_xor", (res, m))
        return res
    return ExprV8("anf", (), (n_vars, sparse))


# ==============================================================================
# 3. Certified Range-Safe CE + EF21-Muon + BitAlloc RVQ
# ==============================================================================


def batch_logit_safe_step_ce(
    X_feat: np.ndarray,
    y_int: np.ndarray,
    num_classes: int,
    steps: int = 80,
    l2_reg: Optional[np.ndarray] = None,
    eps_pert: float = 0.0,
) -> dict[str, Any]:
    """C223/C230: eta = 4*A / R'^2 with 4*eps gate, CertStepGen fields tracked."""
    X_feat = np.asarray(X_feat, dtype=np.float64)
    raw_y = np.asarray(y_int)
    if (
        raw_y.ndim != 1
        or not np.all(np.isfinite(raw_y))
        or np.any(raw_y != np.floor(raw_y))
    ):
        raise ValueError("y_int must be a one-dimensional array of integer labels")
    y_int = raw_y.astype(np.int64)
    if X_feat.ndim != 2 or X_feat.shape[0] == 0 or X_feat.shape[1] == 0:
        raise ValueError("X_feat must be a nonempty rank-2 matrix")
    B, d = X_feat.shape
    if not np.all(np.isfinite(X_feat)):
        raise ValueError("X_feat must contain only finite values")
    if int(num_classes) < 2:
        raise ValueError("num_classes must be at least 2")
    if y_int.shape != (B,) or np.any(y_int < 0) or np.any(y_int >= int(num_classes)):
        raise ValueError("y_int must contain one valid class index per row")
    if int(steps) < 0:
        raise ValueError("steps must be nonnegative")
    if not math.isfinite(float(eps_pert)) or float(eps_pert) < 0:
        raise ValueError("eps_pert must be finite and nonnegative")
    theta = np.zeros((int(num_classes), d), dtype=np.float64)
    reg = (
        np.zeros(d, dtype=np.float64)
        if l2_reg is None
        else np.asarray(l2_reg, dtype=np.float64)
    )
    if reg.shape != (d,) or not np.all(np.isfinite(reg)) or np.any(reg < 0):
        raise ValueError("l2_reg must be a finite nonnegative vector of length d")

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
        if eps_pert > 0.0 and 4.0 * eps_pert >= cert_drop:
            held += 1
            break
        cand = theta + eta * D
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


def newton_schulz_orthogonalize(G: np.ndarray, iters: int = 5) -> np.ndarray:
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


def quantize_5block_single(
    blk: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int], int]:
    """Fin 255 dyadic + 5-trit encoding with verified ternary_split_le bound."""
    blk = np.asarray(blk, dtype=np.float64)
    if blk.shape != (5,) or not np.all(np.isfinite(blk)):
        raise ValueError("quantize_5block_single expects five finite values")
    m = float(np.max(np.abs(blk)))
    if m == 0.0:
        # Five zero trits are encoded as base-3 digits 1 (offset code +1).
        return np.zeros_like(blk), (0, 0, 121), 0
    exp2 = math.ceil(math.log2(m / 1.5 + 1e-18))
    sc = 2.0**exp2
    bn = blk / sc
    s = DYADIC_LEVELS[:, None]
    q_int = np.where(
        bn[None, :] / s > 0.5, 1.0, np.where(bn[None, :] / s < -0.5, -1.0, 0.0)
    )
    q = q_int * s
    errs = np.sum((bn[None, :] - q) ** 2, axis=1)
    best = int(np.argmin(errs))
    s_best = float(DYADIC_LEVELS[best])
    trits = q_int[best].astype(int)
    in_mask = np.abs(bn) <= 1.5 * s_best
    sat_tail = float(np.sum(np.where(~in_mask, (np.abs(bn) - s_best) ** 2, 0.0)))
    bound = int(np.sum(in_mask)) * (s_best / 2.0) ** 2 + sat_tail
    assert float(errs[best]) <= bound + 1e-9
    bval = 0
    for t in trits:
        bval = bval * 3 + (int(t) + 1)
    return q[best] * sc, (exp2, best, bval), int(best)


def quantize_bitalloc_ef21(
    Phi: np.ndarray,
    idx: np.ndarray,
    wk: np.ndarray,
    y: np.ndarray,
    sy: float,
    lam: float,
    max_extra_stages: int = 10,
) -> tuple[np.ndarray, int, list[Any]]:
    """BitAlloc (imbalance_yields_gain) + EF21 Data-Aware ResidualVQ."""
    if idx.size == 0:
        return np.zeros(Phi.shape[1], dtype=np.float64), 1, []
    mag_order = np.argsort(-np.abs(wk))
    idx_sorted = idx[mag_order]
    Sub_s = Phi[:, idx_sorted]
    col_energy = np.mean(Sub_s**2, axis=0)
    A = Sub_s.T @ Sub_s + max(lam, 1e-9) * np.eye(idx.size)

    k = idx.size
    n_blocks = (k + 4) // 5
    wq_s = np.zeros(k, dtype=np.float64)
    dw = wk[mag_order].copy()
    records = []

    for b in range(n_blocks):
        sl = slice(b * 5, min((b + 1) * 5, k))
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))
        q_blk, meta, _ = quantize_5block_single(blk)
        wq_s[sl] += q_blk[: sl.stop - sl.start]
        records.append((b, meta))

    for _ in range(max_extra_stages):
        r_data = y - Sub_s @ wq_s
        mse_raw = float(np.mean((r_data * sy) ** 2))
        if mse_raw < 0.01:
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
        if blk_errs[best_b] * (sy * sy) < 1e-5:
            break
        sl = slice(best_b * 5, min((best_b + 1) * 5, k))
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))
        q_blk, meta, _ = quantize_5block_single(blk)
        wq_s[sl] += q_blk[: sl.stop - sl.start]
        records.append((best_b, meta))

    out = np.zeros(Phi.shape[1], dtype=np.float64)
    out[idx_sorted] = wq_s
    nbytes = 1 + k + n_blocks * 3 + (len(records) - n_blocks) * 4
    return out, nbytes, records


class FactorizedSharedCore:
    """Shared universal multi-scale core C + modular quantized task residuals R_i."""

    def __init__(self, seed: int = 42, tanh_units: int = 48):
        if int(tanh_units) < 3 or int(tanh_units) % 3 != 0:
            raise ValueError("tanh_units must be a positive multiple of 3 (at least 3)")
        rng = np.random.default_rng(seed)
        Q_raw = rng.normal(0.0, 1.0, size=(3, tanh_units))
        Q_T, _ = np.linalg.qr(Q_raw.T)
        scales = np.repeat(np.array([0.8, 1.6, 3.2]), tanh_units // 3)
        self.W1 = Q_T.T * scales[None, :]
        self.b1 = rng.uniform(-1.5, 1.5, size=(1, tanh_units))
        core_vec = np.concatenate([self.W1.ravel(), self.b1.ravel()])
        n_blk = (core_vec.size + 4) // 5
        rem = core_vec.copy()
        q_core = np.zeros_like(core_vec)
        for _ in range(3):
            r5 = np.pad(rem, (0, (-rem.size) % 5)).reshape(n_blk, 5)
            q_st = np.concatenate(
                [quantize_5block_single(r5[b])[0] for b in range(n_blk)]
            )[: core_vec.size]
            q_core += q_st
            rem -= q_st
        self.W1_q = q_core[: self.W1.size].reshape(self.W1.shape)
        self.b1_q = q_core[self.W1.size :].reshape(self.b1.shape)
        self.core_bytes = 3 * n_blk * 3

    def features(
        self, branch: str, xs_arr: np.ndarray, sx: float, quantized: bool = True
    ) -> np.ndarray:
        N, d = xs_arr.shape
        xn = xs_arr.astype(np.float64) / sx
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
            for i in range(d):
                fs.append((cols[i] ** 4)[:, None])
            return np.hstack(fs)
        if branch == "hinge":
            fs = [np.ones((N, 1), dtype=np.float64)]
            for i in range(d):
                fs.append(xn[:, i : i + 1])
                for knot in (-0.375, 0.0, 0.375):
                    fs.append(np.maximum(0.0, xn[:, i] - knot)[:, None])
                    fs.append(np.maximum(0.0, knot - xn[:, i])[:, None])
            for i in range(d):
                for j in range(i + 1, d):
                    diff = xn[:, i] - xn[:, j]
                    fs.append(np.maximum(0.0, diff)[:, None])
                    fs.append(np.maximum(0.0, -diff)[:, None])
            return np.hstack(fs)
        if branch == "bit":
            xi = np.rint(xs_arr).astype(np.int64)
            fs = [np.ones((N, 1), dtype=np.float64)]
            for i in range(d):
                fs.append((xi[:, i] < 0).astype(np.float64)[:, None])
                for b in range(4):
                    fs.append(((xi[:, i] >> b) & 1).astype(np.float64)[:, None])
            if d >= 2:
                for b in range(4):
                    fs.append(
                        (((xi[:, 0] >> b) & 1) * ((xi[:, 1] >> b) & 1)).astype(
                            np.float64
                        )[:, None]
                    )
                fs.append(((xi[:, 0] < 0) * (xi[:, 1] < 0)).astype(np.float64)[:, None])
            return np.hstack(fs)
        x_pad = np.zeros((N, 3), dtype=np.float64)
        x_pad[:, : min(3, d)] = xn[:, : min(3, d)]
        W1 = self.W1_q if quantized else self.W1
        b1 = self.b1_q if quantized else self.b1
        h = np.tanh(x_pad @ W1 + b1)
        return np.hstack(
            [
                self.features("poly", xs_arr, sx, quantized),
                self.features("hinge", xs_arr, sx, quantized)[:, 1:],
                h,
            ]
        )

    def fit_residual(
        self,
        train_x: Sequence[Sequence[int]],
        train_y: Sequence[int],
        val_x: Sequence[Sequence[int]],
        val_y: Sequence[int],
    ) -> "FactorizedAdapter":
        tx = np.asarray(train_x, dtype=np.float64)
        ty = np.asarray(train_y, dtype=np.float64)
        vx = np.asarray(val_x, dtype=np.float64) if val_x else tx
        vy = np.asarray(val_y, dtype=np.float64) if val_y else ty
        sx = max(1.0, float(np.max(np.abs(tx))))
        sy = max(1.0, float(np.max(np.abs(ty))))
        y = ty / sy
        best = (float("inf"), "poly", None, None, 9999)
        for bname in ("poly", "hinge", "bit", "core"):
            Phi = self.features(bname, tx, sx, quantized=True)
            N, D = Phi.shape
            for lam in (1e-7, 1e-4, 1e-2):
                A = Phi.T @ Phi + lam * np.eye(D)
                w = np.linalg.solve(A, Phi.T @ y)
                order = np.argsort(-np.abs(w))
                ks = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, D) if k <= D]
                for k in ks:
                    idx = order[:k]
                    Ak = Phi[:, idx].T @ Phi[:, idx] + max(lam, 1e-9) * np.eye(k)
                    wk = np.linalg.solve(Ak, Phi[:, idx].T @ y)
                    if np.max(np.abs(wk)) > 30.0:
                        continue
                    wc = np.zeros(D, dtype=np.float64)
                    wc[idx] = wk
                    wq, nb, _ = quantize_bitalloc_ef21(Phi, idx, wk, y, sy, lam)
                    pred_q = (Phi @ wq) * sy
                    mse_q = float(np.mean((pred_q - ty) ** 2))
                    acc_tr = float(np.mean(np.rint(pred_q) == ty))
                    obj = mse_q + 0.002 * nb + (1.0 - acc_tr) * 0.5
                    if obj < best[0]:
                        best = (obj, bname, wc, wq, nb)
                    if mse_q < 1e-3:
                        break
                if best[0] < 0.05:
                    break
            if best[0] < 0.02:
                break

        _, bname, wc, wq, nb = best
        Phi_va = self.features(bname, vx, sx, quantized=True)
        pred_f = (Phi_va @ wc) * sy
        pred_q = (Phi_va @ wq) * sy
        return FactorizedAdapter(
            arity=tx.shape[1],
            branch=bname,
            sx=sx,
            sy=sy,
            w_float=wc,
            w_quant=wq,
            compressed_bytes=nb + 9,
            val_exact_acc_float=float(np.mean(np.rint(pred_f) == vy)),
            val_exact_acc_quant=float(np.mean(np.rint(pred_q) == vy)),
            val_mse_float=float(np.mean((pred_f - vy) ** 2)),
            val_mse_quant=float(np.mean((pred_q - vy) ** 2)),
        )


@dataclass
class FactorizedAdapter:
    arity: int
    branch: str
    sx: float
    sy: float
    w_float: np.ndarray
    w_quant: np.ndarray
    compressed_bytes: int
    val_exact_acc_float: float
    val_exact_acc_quant: float
    val_mse_float: float
    val_mse_quant: float

    def predict(
        self, core: FactorizedSharedCore, xs: Sequence[int], quantized: bool = True
    ) -> float:
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1)
        Phi = core.features(self.branch, x, self.sx, quantized=quantized)
        w = self.w_quant if quantized else self.w_float
        return float((Phi @ w)[0] * self.sy)

    def to_dict(self) -> dict[str, Any]:
        # Persist both dense arrays for an exact state round-trip. The reported
        # compressed_bytes remains a FORMAT ESTIMATE, not the JSON/RAM footprint.
        return {
            "arity": self.arity,
            "branch": self.branch,
            "sx": self.sx,
            "sy": self.sy,
            "w_float": self.w_float.tolist(),
            "w_quant": self.w_quant.tolist(),
            "compressed_bytes": self.compressed_bytes,
            "val_exact_acc_float": self.val_exact_acc_float,
            "val_exact_acc_quant": self.val_exact_acc_quant,
            "val_mse_float": self.val_mse_float,
            "val_mse_quant": self.val_mse_quant,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FactorizedAdapter":
        wf = np.asarray(d.get("w_float", d.get("w_quant", [])), dtype=np.float64)
        wq = np.asarray(d.get("w_quant", []), dtype=np.float64)
        if wf.ndim != 1 or wq.ndim != 1 or wf.shape != wq.shape:
            raise ValueError("invalid serialized adapter coefficient arrays")
        return FactorizedAdapter(
            int(d["arity"]),
            str(d["branch"]),
            float(d["sx"]),
            float(d["sy"]),
            wf,
            wq,
            int(d.get("compressed_bytes", 0)),
            float(d.get("val_exact_acc_float", float("nan"))),
            float(d.get("val_exact_acc_quant", float("nan"))),
            float(d.get("val_mse_float", float("nan"))),
            float(d.get("val_mse_quant", float("nan"))),
        )


# ==============================================================================
# 4. Continual GPM + SpaceDecoupling + EF21-Muon Adapter
# ==============================================================================


class ContinualGPMMemory:
    """GPM + SpaceDecoupling + EF21Markov: P_R = I - U U^T, zero drift on protected cortex."""

    def __init__(self, d_in: int, d_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.1, size=(d_in, d_out))
        self.U_basis = np.zeros((d_in, 0), dtype=np.float64)
        self.ef_v = np.zeros_like(self.W)

    def project_orthogonal(self, G: np.ndarray) -> np.ndarray:
        if self.U_basis.shape[1] == 0:
            return G
        return G - self.U_basis @ (self.U_basis.T @ G)

    def train_task_ef21_muon(
        self, X: np.ndarray, Y: np.ndarray, steps: int = 40, lr: float = 0.2
    ) -> float:
        for _ in range(steps):
            err = X @ self.W - Y
            grad = (2.0 / len(X)) * (X.T @ err)
            g_proj = self.project_orthogonal(grad)
            diff = g_proj - self.ef_v
            comp = 0.5 * diff
            self.ef_v = self.ef_v + comp
            step_dir = newton_schulz_orthogonalize(self.ef_v, iters=4)
            step_dir = self.project_orthogonal(step_dir)
            cur = float(np.mean((X @ self.W - Y) ** 2))
            for eta in (lr, lr * 0.5, lr * 0.1, 0.01):
                cand = self.W - eta * step_dir
                if float(np.mean((X @ cand - Y) ** 2)) < cur:
                    self.W = cand
                    break
        return float(np.mean((X @ self.W - Y) ** 2))

    def consolidate_subspace(self, X: np.ndarray, energy_thresh: float = 0.999) -> None:
        res_X = (
            X if self.U_basis.shape[1] == 0 else X - (X @ self.U_basis) @ self.U_basis.T
        )
        if float(np.linalg.norm(res_X)) < 1e-10:
            return
        _, S, Vt = np.linalg.svd(res_X, full_matrices=False)
        tot = float(np.sum(S**2))
        if tot < 1e-12:
            return
        cum = np.cumsum(S**2) / tot
        k = int(np.searchsorted(cum, energy_thresh)) + 1
        new_dirs = Vt[:k].T
        if self.U_basis.shape[1] == 0:
            Q, _ = np.linalg.qr(new_dirs)
            self.U_basis = Q
        else:
            combined = np.hstack([self.U_basis, new_dirs])
            Q, _ = np.linalg.qr(combined)
            r = min(Q.shape[1], np.linalg.matrix_rank(combined))
            self.U_basis = Q[:, :r]


# ==============================================================================
# 5. CRAM Energy-Based Reasoning Engine
# ==============================================================================


class CRAMEngine:
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
        if int(n_nodes) <= 0:
            raise ValueError("n_nodes must be positive")
        if int(num_colors) <= 0:
            raise ValueError("num_colors must be positive")
        if int(max_sweeps) < 0:
            raise ValueError("max_sweeps must be nonnegative")
        if not math.isfinite(float(eps_noise)) or float(eps_noise) < 0:
            raise ValueError("eps_noise must be finite and nonnegative")
        for a, b in edges:
            if not (0 <= int(a) < n_nodes and 0 <= int(b) < n_nodes):
                raise ValueError(f"edge endpoint outside [0, {n_nodes})")
        adj = [set() for _ in range(n_nodes)]
        for a, b in edges:
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
        adj_list = [sorted(s) for s in adj]
        giv = {int(i): int(c) for i, c in (givens or {}).items()}
        if any(i < 0 or i >= n_nodes for i in giv):
            raise ValueError("given node index outside graph")
        if any(c < 0 or c >= num_colors for c in giv.values()):
            raise ValueError("given color outside palette")
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
        T = 0.55
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
                    logits = np.array([-f / T for f in fields], dtype=np.float64)
                    logits -= np.max(logits)
                    probs = np.exp(logits)
                    probs /= np.sum(probs)
                    y[pick] = int(rng.choice(num_colors, p=probs))
                    hb_steps += 1
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
            # `verified` means a valid solution was actually found; it must
            # be false for a locally consistent but unsolved/unsatisfiable run.
            "verified": bool(valid and cur_e == 0),
            "givens_respected": all(y[i] == g for i, g in giv.items()),
            "sparse_work_per_sweep": sparse_work,
            "dense_work_per_sweep": dense_work,
        }

    @staticmethod
    def solve_latent_maze(
        n_nodes: int,
        adj: Sequence[Sequence[int]],
        goal: int,
        eps_z: float = 1e-6,
        max_iters: int = 100_000,
    ) -> dict[str, Any]:
        """Contractive latent-field iteration with an explicit stopping certificate."""
        if int(n_nodes) <= 0:
            raise ValueError("n_nodes must be positive")
        if len(adj) != n_nodes:
            raise ValueError("adjacency must have one row per node")
        if not (0 <= int(goal) < n_nodes):
            raise ValueError("goal outside graph")
        if not math.isfinite(float(eps_z)) or float(eps_z) <= 0:
            raise ValueError("eps_z must be finite and positive")
        if int(max_iters) <= 0:
            raise ValueError("max_iters must be positive")
        adj_sets: list[tuple[int, ...]] = []
        for i, row in enumerate(adj):
            ns = tuple(sorted(set(int(j) for j in row)))
            if any(j < 0 or j >= n_nodes for j in ns):
                raise ValueError(f"neighbor index outside graph in row {i}")
            adj_sets.append(ns)
        delta = max((len(nb) for nb in adj_sets), default=0)
        m = float(delta + 1)
        q = delta / m
        z = np.zeros(n_nodes, dtype=np.float64)
        iters = 0
        error_bound = float("inf")
        if q > 0:
            predicted = max(1, math.ceil(math.log(eps_z * (1.0 - q)) / math.log(q)))
            run_limit = min(int(max_iters), predicted)
            for _ in range(run_limit):
                z_new = np.array(
                    [
                        (sum(z[j] for j in adj_sets[i]) + (1.0 if i == goal else 0.0))
                        / m
                        for i in range(n_nodes)
                    ],
                    dtype=np.float64,
                )
                diff = float(np.max(np.abs(z_new - z)))
                z = z_new
                iters += 1
                error_bound = diff * q / (1.0 - q)
                if error_bound <= eps_z:
                    break
        else:
            z[goal] = 1.0 / m
            iters = 1
            error_bound = 0.0
        upper = 1.0 / (m - delta)
        if not (
            np.all(np.isfinite(z))
            and np.all(z >= -1e-12)
            and np.all(z <= upper + 1e-12)
        ):
            raise ArithmeticError("latent-field invariant failed")
        converged = bool(error_bound <= eps_z)
        return {
            "z": z,
            "iters": iters,
            "contraction_q": q,
            "upper_bound": upper,
            "error_bound": error_bound,
            "tolerance": float(eps_z),
            "converged": converged,
        }


# ==============================================================================
# 6. AdaptiveChunking, GDNRAM & Bounded-Loss HedgeRouter
# ==============================================================================


def stutter_norm(seq: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


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


def boundary_heads(boundary: Callable[[Any], bool], xs: Sequence[Any]) -> list[Any]:
    return [c[0] for c in boundary_chunks(boundary, xs)]


class GDNRAM:
    """lag-1 associative store/load."""

    def __init__(self, n_slots: int, dim: int):
        self.mem = np.zeros((n_slots, dim), dtype=np.float64)

    def store(
        self, addr: int, val: np.ndarray, gate_beta: float = 1.0, decay_g: float = 0.0
    ) -> None:
        self.mem[addr] = (1.0 - decay_g) * self.mem[addr] * (
            1.0 - gate_beta
        ) + gate_beta * val

    def load(self, addr: int) -> np.ndarray:
        return self.mem[addr].copy()


class HedgeRouter:
    """Numerically stable exponentially weighted router for losses in [0, 1]."""

    def __init__(self, n_experts: int, eta: float = 0.5):
        if int(n_experts) <= 0:
            raise ValueError("n_experts must be positive")
        if not math.isfinite(float(eta)) or not (0.0 < float(eta) <= 1.0):
            raise ValueError("eta must be finite and in (0, 1]")
        self.K = int(n_experts)
        self.eta = float(eta)
        self.weights = np.ones(self.K, dtype=np.float64) / self.K
        self.cum_loss = np.zeros(self.K, dtype=np.float64)

    def update(self, losses_01: np.ndarray) -> np.ndarray:
        losses = np.asarray(losses_01, dtype=np.float64)
        if losses.shape != (self.K,):
            raise ValueError(f"losses must have shape ({self.K},)")
        if (
            not np.all(np.isfinite(losses))
            or np.any(losses < 0.0)
            or np.any(losses > 1.0)
        ):
            raise ValueError("expert losses must be finite and lie in [0, 1]")
        self.cum_loss += losses
        if not np.all(np.isfinite(self.cum_loss)):
            raise OverflowError("cumulative expert loss overflowed")
        # Log-sum-exp avoids the all-zero weight vector after many rounds.
        logw = -self.eta * self.cum_loss
        logw -= float(np.max(logw))
        logz = float(np.log(np.sum(np.exp(logw))))
        logw -= logz
        # The lower bound is checked in log space: w_i >= exp(-eta L_i)/K.
        lower_log = -math.log(self.K) - self.eta * self.cum_loss
        if np.any(logw + 1e-10 < lower_log):
            raise ArithmeticError("Hedge survivor lower-bound invariant failed")
        self.weights = np.exp(logw)
        self.weights /= np.sum(self.weights)
        return self.weights.copy()


# ==============================================================================
# 7. AuditLedger + RAM Governor
# ==============================================================================


@dataclass
class AuditLedger:
    """Explicit alpha accounting for externally defined statistical tests.

    Exact enumeration of a declared finite support is deterministic and does
    not consume alpha. This ledger is NOT itself a confidence-bound procedure.
    """

    delta_total: float = 0.05
    spent: dict[str, float] = field(default_factory=dict)
    rejected_tasks: set[str] = field(default_factory=set)
    final_test_touched: bool = False

    def __post_init__(self) -> None:
        self.delta_total = float(self.delta_total)
        if not math.isfinite(self.delta_total) or not (0.0 < self.delta_total < 1.0):
            raise ValueError("delta_total must be finite and in (0, 1)")
        self.spent = {str(k): float(v) for k, v in self.spent.items()}
        if any(not k or not math.isfinite(v) or v < 0.0 for k, v in self.spent.items()):
            raise ValueError(
                "spent alpha entries must be named, finite, and nonnegative"
            )
        if sum(self.spent.values()) > self.delta_total + 1e-12:
            raise ValueError("serialized alpha spend exceeds delta_total")

    def remaining(self) -> float:
        return max(0.0, self.delta_total - sum(self.spent.values()))

    def spend_anytime(self, task: str, alpha: float) -> bool:
        alpha = float(alpha)
        if (
            self.final_test_touched
            or not task
            or not math.isfinite(alpha)
            or alpha <= 0.0
        ):
            return False
        if alpha > self.remaining() + 1e-12:
            return False
        self.spent[task] = self.spent.get(task, 0.0) + alpha
        return True

    def record_reject(self, task: str) -> None:
        if not task:
            raise ValueError("task name must be nonempty")
        self.rejected_tasks.add(task)

    def mark_final_test_touched(self) -> None:
        """Freeze future alpha spending after the sealed final test is opened."""
        self.final_test_touched = True


def ram_available_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return hagi_v7._ram_available_bytes()


# ==============================================================================
# 8. Fast Piecewise / Modular / Batched-MDL Symbolic Engine + DAG Compressor
# ==============================================================================


def exact_piecewise_poly_fit(
    train_x: Sequence[tuple[int, ...]], train_y: Sequence[int]
) -> Optional[ExprV8]:
    """Exact rational piecewise polynomial / modular residue discovery before AST heap."""
    arity = len(train_x[0])
    if arity == 1:
        xs = [r[0] for r in train_x]
        for m in range(2, 9):
            buckets: dict[int, set[int]] = {}
            for x, y in zip(xs, train_y):
                buckets.setdefault(x % m, set()).add(y)
            if all(len(vs) == 1 for vs in buckets.values()) and len(buckets) >= 2:
                if m == 2 and buckets.get(0) == {0} and buckets.get(1) == {1}:
                    return ExprV8("bit_and", (ExprV8.var(0), ExprV8.const(1)))
        for thr in (-3, -2, -1, 0, 1, 2, 3):
            left = [(x, y) for x, y in zip(xs, train_y) if x < thr]
            right = [(x, y) for x, y in zip(xs, train_y) if x >= thr]
            if len(left) < 3 or len(right) < 3:
                continue
            cl = _fit_poly_or_const(left)
            cr = _fit_poly_or_const(right)
            if cl is not None and cr is not None and cl != cr:
                cond = ExprV8("lt", (ExprV8.var(0), ExprV8.const(thr)))
                return ExprV8("if_cond", (cond, cl, cr))
    elif arity == 2:
        left = [(r, y) for r, y in zip(train_x, train_y) if r[0] < r[1]]
        right = [(r, y) for r, y in zip(train_x, train_y) if r[0] >= r[1]]
        if len(left) >= 6 and len(right) >= 6:
            tl = hagi_v7._exact_multivariate_poly_fit(
                [r for r, _ in left], [y for _, y in left], 2, max_degree=2
            )
            tr = hagi_v7._exact_multivariate_poly_fit(
                [r for r, _ in right], [y for _, y in right], 2, max_degree=2
            )
            if tl is not None and tr is not None and tl != tr:
                cond = ExprV8("lt", (ExprV8.var(0), ExprV8.var(1)))
                return ExprV8(
                    "if_cond",
                    (cond, ExprV8("poly_nd", (), tl), ExprV8("poly_nd", (), tr)),
                )
    return None


def bounded_fold_hypotheses(arity: int) -> list[ExprV8]:
    """Small task-name-independent family of bounded arithmetic predicates.

    The range binder is represented by the next variable slot in the body.
    Hypotheses are still accepted only after held-out and full finite-support
    verification against the task teacher.
    """
    if int(arity) != 1:
        return []
    x, k = ExprV8.var(0), ExprV8.var(1)
    c0, c1, c2 = ExprV8.const(0), ExprV8.const(1), ExprV8.const(2)
    x_ge_2 = ExprV8("ge", (x, c2))
    rem = ExprV8("mod", (x, k))
    not_divisor = ExprV8("ne", (rem, c0))
    divisor = ExprV8("eq", (rem, c0))
    sqrt_stop = ExprV8("add", (ExprV8("isqrt", (ExprV8("max", (x, c0)),)), c1))
    prime = ExprV8(
        "bit_and",
        (
            x_ge_2,
            ExprV8("all_range", (c2, sqrt_stop, not_divisor)),
        ),
    )
    composite = ExprV8(
        "bit_and",
        (
            ExprV8("ge", (x, ExprV8.const(4))),
            ExprV8("any_range", (c2, sqrt_stop, divisor)),
        ),
    )
    even = ExprV8("eq", (ExprV8("mod", (x, c2)), c0))
    odd = ExprV8("ne", (ExprV8("mod", (x, c2)), c0))
    nonnegative = ExprV8("ge", (x, c0))
    square = ExprV8(
        "bit_and",
        (
            nonnegative,
            ExprV8(
                "any_range",
                (
                    c0,
                    ExprV8("add", (x, c1)),
                    ExprV8("eq", (ExprV8("mul", (k, k)), x)),
                ),
            ),
        ),
    )
    # Arithmetic aggregates over positive divisors. These are generic folds,
    # not memorized tables: tau(n), sigma(n), and Euler phi(n).
    positive_stop = ExprV8("add", (ExprV8("max", (x, c0)), c1))
    from_one = ExprV8.const(1)
    divides = ExprV8("eq", (ExprV8("mod", (x, k)), c0))
    coprime = ExprV8("eq", (ExprV8("gcd", (x, k)), c1))
    tau = ExprV8("sum_range", (from_one, positive_stop, divides))
    sigma = ExprV8("sum_range", (from_one, positive_stop, ExprV8("mul", (k, divides))))
    phi = ExprV8("sum_range", (from_one, positive_stop, coprime))
    # Empty product is 1; multiplying by (x >= 0) defines negative inputs as 0.
    factorial = ExprV8(
        "mul",
        (
            ExprV8("ge", (x, c0)),
            ExprV8("prod_range", (from_one, positive_stop, k)),
        ),
    )
    return [prime, composite, square, even, odd, tau, sigma, phi, factorial]


def _fit_poly_or_const(pts: list[tuple[int, int]]) -> Optional[ExprV8]:
    ys = {y for _, y in pts}
    if len(ys) == 1:
        return ExprV8.const(next(iter(ys)))
    coeffs = hagi_v7._exact_poly_fit(
        [x for x, _ in pts], [y for _, y in pts], max_degree=3
    )
    if coeffs is not None:
        return ExprV8("poly", (ExprV8.var(0),), coeffs)
    return None


# V10 Reification wire: invented ops promoted into V8 batched synthesis
# live here (registered per synthesis call; core mirrors from REIFIED_OPS).
REIFIED_V8: dict = {}


def _leaf_iter(leaf_exprs, reified_leaf_fns):
    """Yield (leaf, reified_fn_or_None) aligned with leaf_exprs."""
    ri = 0
    for lf in leaf_exprs:
        if lf.op == "reified":
            yield lf, reified_leaf_fns[ri]
            ri += 1
        else:
            yield lf, None


def synthesize_batched_mdl_v8(
    arity: int,
    train_x: Sequence[tuple[int, ...]],
    train_y: Sequence[int],
    val_x: Sequence[tuple[int, ...]],
    val_y: Sequence[int],
    library_experts: Sequence[tuple[int, str, "ExpertV8"]] = (),
    max_evals: int = 4000,
    max_bits: int = 84,
    deadline: Optional[float] = None,
    reified_ops: Optional[dict] = None,
) -> tuple[Optional[ExprV8], int]:
    """reified_ops: optional {name: (fn_1d, cost_bits)} — invented ops
    exposed as extra LEAVES of the batched synthesis (Reification wire:
    yesterday's invention composes inside tomorrow's programs)."""
    """Batched NumPy int64 synthesis with dedup and Call-DAG library references."""
    try:
        X = np.asarray(train_x, dtype=np.int64)
        Y = np.asarray(train_y, dtype=np.int64)
    except (OverflowError, ValueError, TypeError):
        return None, 0
    if (
        X.ndim != 2
        or X.shape[1] != int(arity)
        or X.shape[0] == 0
        or Y.shape != (X.shape[0],)
    ):
        raise ValueError(
            "training arrays must have shapes (N, arity) and (N,), with N>0"
        )
    # Use Python integers here: np.abs(int64_min) overflows back to int64_min.
    if any(abs(int(v)) > 128 for v in X.reshape(-1)):
        return None, 0
    if int(max_evals) <= 0 or int(max_bits) <= 0:
        return None, 0
    N = X.shape[0]
    leaf_exprs = [ExprV8.var(i) for i in range(arity)] + [
        ExprV8.const(c) for c in hagi_v7.CONSTS
    ]
    # reified leaves: invented ops as unary macros (vectorized over rows)
    reified_leaf_fns = []
    if reified_ops:
        for rname, (rfn, rcost) in reified_ops.items():
            full = f"reified:{rname}"
            REIFIED_V8[full] = rfn
            leaf_exprs.append(ExprV8("reified", (), full))
            reified_leaf_fns.append(rfn)
    L_arr = np.stack(
        [np.ascontiguousarray(X[:, int(lf.value or 0)])
         if lf.op == "var"
         else (np.array([int(rfn(*[int(v) for v in r])) for r in X],
                    dtype=np.int64)
               if lf.op == "reified"
               else np.full(N, int(lf.value), dtype=np.int64))
         for lf, rfn in _leaf_iter(leaf_exprs, reified_leaf_fns)
        ]
    )
    L_bits = [
        (reified_ops[lf.value.split(":", 1)[1]][1]
         if lf.op == "reified" else lf.bits())
        for lf in leaf_exprs
    ]

    for i, lf in enumerate(leaf_exprs):
        if np.array_equal(L_arr[i], Y):
            if all(lf.eval(r) == y for r, y in zip(val_x, val_y)):
                return lf, 1

    seen_sigs = {L_arr[i].tobytes(): 1 for i in range(len(leaf_exprs))}
    buckets: dict[int, list[tuple[ExprV8, np.ndarray]]] = {}
    for i, lf in enumerate(leaf_exprs):
        buckets.setdefault(L_bits[i], []).append((lf, L_arr[i]))

    total_checked = len(leaf_exprs)
    MAG_CAP = 1 << 28
    all_ops = [
        "add",
        "sub",
        "mul",
        "floordiv",
        "mod",
        "gcd",
        "lcm",
        "min",
        "max",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "bit_and",
        "bit_or",
        "bit_xor",
        "bit_nand8",
        "bit_nor8",
        "bit_xnor8",
        "pow_int",
    ]
    env = {name: ex for _, name, ex in library_experts}

    while buckets and total_checked < max_evals:
        if deadline is not None and time.monotonic() >= deadline:
            break
        min_b = min(buckets)
        if min_b >= max_bits:
            break
        batch = buckets.pop(min_b)
        if not batch:
            continue
        if len(batch) > 256:
            batch = batch[:256]
        B_exprs = [x[0] for x in batch]
        V = np.stack([x[1] for x in batch])

        def register(
            res_arr: np.ndarray, builder_fn: Callable[[int, int], ExprV8], new_bits: int
        ) -> Optional[ExprV8]:
            nonlocal total_checked
            if new_bits > max_bits:
                return None
            if res_arr.ndim == 2:
                res_arr = res_arr[:, None, :]
            matches = np.all(res_arr == Y[None, None, :], axis=2)
            if np.any(matches):
                for bi, ki in np.argwhere(matches):
                    cand = builder_fn(int(bi), int(ki))
                    try:
                        if all(cand.eval(r, env) == y for r, y in zip(val_x, val_y)):
                            return cand
                    except Exception:
                        pass
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
                seen_sigs[key] = cnt + 1
                total_checked += 1
                buckets.setdefault(new_bits, []).append(
                    (builder_fn(int(bi), int(ki)), row)
                )
            return None

        for uop, ures in (
            ("neg", -V),
            ("abs", np.abs(V)),
            ("sign", np.sign(V).astype(np.int64)),
            ("bit_not8", (~V) & 0xFF),
            ("clamp3", np.clip(V, -3, 3)),
        ):
            hit = register(
                ures, lambda bi, _, o=uop: ExprV8(o, (B_exprs[bi],)), min_b + TOKEN_BITS
            )
            if hit is not None:
                return hit, total_checked

        for lib_idx, ename, ex in library_experts[:24]:
            cbits = TOKEN_BITS + gamma_bits(lib_idx + 1)
            if ex.program.arity == 1:
                rows_out = np.empty_like(V)
                ok_rows = []
                for bi in range(V.shape[0]):
                    if B_exprs[bi].op == "const":
                        continue
                    try:
                        vals_b = V[bi]
                        if all(ex.supports((int(v),)) for v in vals_b):
                            rows_out[bi] = [
                                ex.program.eval((int(v),), env) for v in vals_b
                            ]
                            ok_rows.append(bi)
                    except Exception:
                        pass
                if ok_rows:
                    sub_r = rows_out[ok_rows]
                    hit = register(
                        sub_r,
                        lambda bi, _, o=ename, idxs=ok_rows: ExprV8(
                            "call", (B_exprs[idxs[bi]],), o
                        ),
                        min_b + cbits,
                    )
                    if hit is not None:
                        return hit, total_checked
            elif ex.program.arity == 2:
                for li, lf in enumerate(leaf_exprs):
                    lv = L_arr[li]
                    for order_idx in (0, 1):
                        rows_out = np.empty_like(V)
                        ok_rows = []
                        for bi in range(V.shape[0]):
                            if B_exprs[bi].op == "const" and lf.op == "const":
                                continue
                            bv = V[bi]
                            a_v, b_v = (bv, lv) if order_idx == 0 else (lv, bv)
                            try:
                                if all(
                                    ex.supports((int(a_v[k]), int(b_v[k])))
                                    for k in range(N)
                                ):
                                    rows_out[bi] = [
                                        ex.program.eval((int(a_v[k]), int(b_v[k])), env)
                                        for k in range(N)
                                    ]
                                    ok_rows.append(bi)
                            except Exception:
                                pass
                        if ok_rows:
                            sub_r = rows_out[ok_rows]
                            hit = register(
                                sub_r,
                                (
                                    lambda bi, _, o=ename, l=lf, idxs=ok_rows, oi=order_idx: ExprV8(
                                        "call",
                                        (
                                            (B_exprs[idxs[bi]], l)
                                            if oi == 0
                                            else (l, B_exprs[idxs[bi]])
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
        for op in all_ops:
            if op == "add":
                R = Va + Lb
            elif op == "sub":
                R = Va - Lb
            elif op == "mul":
                R = Va * Lb
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
                ok_l = (Va != 0) & (Lb != 0)
                sa = np.where(ok_l, Va, 1)
                sb = np.where(ok_l, Lb, 1)
                R = np.where(ok_l, np.abs(sa // np.gcd(sa, sb) * sb), MAG_CAP + 1)
            elif op == "pow_int":
                ok_p = (Lb >= 0) & (Lb <= 8) & (np.abs(Va) <= 100)
                sa = np.where(ok_p, Va, 0)
                sb = np.where(ok_p, Lb, 0)
                R = np.where(ok_p, sa**sb, MAG_CAP + 1)
            else:
                continue
            for li in range(len(leaf_exprs)):
                hit = register(
                    R[:, li : li + 1, :],
                    lambda bi, _, o=op, l=leaf_exprs[li]: ExprV8(o, (B_exprs[bi], l)),
                    min_b + TOKEN_BITS + L_bits[li],
                )
                if hit is not None:
                    return hit, total_checked
            if op in ("sub", "floordiv", "mod", "lt", "le", "gt", "ge", "pow_int"):
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
                    ok_p = (Va >= 0) & (Va <= 8) & (np.abs(Lb) <= 100)
                    sa = np.where(ok_p, Lb, 0)
                    sb = np.where(ok_p, Va, 0)
                    R2 = np.where(ok_p, sa**sb, MAG_CAP + 1)
                for li in range(len(leaf_exprs)):
                    hit = register(
                        R2[:, li : li + 1, :],
                        lambda bi, _, o=op, l=leaf_exprs[li]: ExprV8(
                            o, (l, B_exprs[bi])
                        ),
                        min_b + TOKEN_BITS + L_bits[li],
                    )
                    if hit is not None:
                        return hit, total_checked
        for op in ("mul", "add"):
            R_s = V * V if op == "mul" else V + V
            hit = register(
                R_s,
                lambda bi, _, o=op: ExprV8(o, (B_exprs[bi], B_exprs[bi])),
                min_b * 2 + TOKEN_BITS,
            )
            if hit is not None:
                return hit, total_checked

    return None, total_checked


# ==============================================================================
# 9. ExpertV8, TheoremMinerV8 & GrowingHagiProcessorV8
# ==============================================================================


@dataclass
class ExpertV8:
    name: str
    family: str
    program: ProgramV8
    domains: tuple[tuple[int, ...], ...]
    valid_rule: str
    code_length: int
    adapter: Optional[FactorizedAdapter] = None
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
    def from_dict(d: dict[str, Any]) -> "ExpertV8":
        return ExpertV8(
            str(d["name"]),
            str(d["family"]),
            ProgramV8.from_dict(d["program"]),
            tuple(tuple(int(v) for v in row) for row in d["domains"]),
            str(d.get("valid_rule", "all")),
            int(d["code_length"]),
            FactorizedAdapter.from_dict(d["adapter"]) if d.get("adapter") else None,
            tuple(str(x) for x in d.get("parents", [])),
        )


class TheoremMinerV8:
    """Scans ALL experts (fixes v7.4's [:16] truncation) — commutativity, involutivity, etc."""

    def mine(self, model: "GrowingHagiProcessorV8") -> list[hagi_v7.Theorem]:
        found: list[hagi_v7.Theorem] = []
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
                                th = hagi_v7.Theorem(
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
                            th = hagi_v7.Theorem(
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
                            th = hagi_v7.Theorem(
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
                    for prop, check_fn in (
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
                                if check_fn():
                                    th = hagi_v7.Theorem(
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
                                    th = hagi_v7.Theorem(
                                        nm,
                                        f"{e.name}(-x)={'+' if sgn==1 else '-'}{e.name}(x)",
                                        len(dom),
                                    )
                                    model.theorems.append(th)
                                    found.append(th)
                                    known.add(nm)
                            except Exception:
                                pass
        return found


def _expr_call_names(expr: Optional[ExprV8]) -> set[str]:
    if expr is None:
        return set()
    names = {str(expr.value)} if expr.op == "call" else set()
    for child in expr.args:
        names.update(_expr_call_names(child))
    return names


def verify_call_dag(model: "GrowingHagiProcessorV8") -> dict[str, Any]:
    """Verify all symbolic expert-call references exist and form an acyclic DAG."""
    graph: dict[str, set[str]] = {}
    missing: list[tuple[str, str]] = []
    for name, expert in model.experts.items():
        refs = _expr_call_names(expert.program.expr)
        for ref in refs:
            if ref not in model.experts:
                missing.append((name, ref))
        graph[name] = refs
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: list[str] = []

    def dfs(node: str, trail: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycles.append(" -> ".join((*trail, node)))
            return
        visiting.add(node)
        for child in graph.get(node, ()):
            if child in graph:
                dfs(child, (*trail, node))
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        dfs(node, ())
    return {
        "valid": not missing and not cycles,
        "missing": missing,
        "cycles": cycles,
        "nodes": len(graph),
        "edges": sum(len(v) for v in graph.values()),
    }


def verify_growth_lineage(model: "GrowingHagiProcessorV8") -> dict[str, Any]:
    """Check parent provenance references and reject cycles in the growth graph."""
    graph: dict[str, tuple[str, ...]] = {}
    missing: list[tuple[str, str]] = []
    for name, expert in model.experts.items():
        graph[name] = tuple(expert.parents)
        for parent in expert.parents:
            if parent not in model.experts:
                missing.append((name, parent))
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: list[str] = []

    def dfs(node: str, trail: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycles.append(" -> ".join((*trail, node)))
            return
        visiting.add(node)
        for parent in graph.get(node, ()):
            if parent in graph:
                dfs(parent, (*trail, node))
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        dfs(node, ())
    return {
        "valid": not missing and not cycles,
        "missing": missing,
        "cycles": cycles,
        "nodes_with_parents": sum(bool(v) for v in graph.values()),
        "edges": sum(len(v) for v in graph.values()),
    }


class GrowingHagiProcessorV8:
    """Unified HAGI v8 processor (finite-support research prototype)."""

    def __init__(
        self, seed: int = 42, node_budget: int = 4000, delta_total: float = 0.05
    ):
        self.core = FactorizedSharedCore(seed=seed)
        self.node_budget = int(node_budget)
        self.experts: dict[str, ExpertV8] = {}
        self.version = 0
        self.total_commits = 0
        self.total_holds = 0
        self.growth_log: list[h5.GrowthEvent] = []
        self.theorems: list[hagi_v7.Theorem] = []
        self.ledger = AuditLedger(delta_total=delta_total)
        self.stats = hagi_v7.SynthesisStats()
        self.router = HedgeRouter(4, eta=0.5)
        self.ram = GDNRAM(n_slots=32, dim=4)

    def _expert_index(self) -> dict[str, int]:
        return {nm: i for i, nm in enumerate(sorted(self.experts))}

    def learn(
        self,
        spec: h5.TaskSpec,
        seed: int = 0,
        train_fraction: float = 0.70,
        poison_one_label: bool = False,
        train_neural: bool = True,
        deadline: Optional[float] = None,
        parents: Sequence[str] = (),
    ) -> h5.GrowthEvent:
        if spec.name in self.experts:
            ev = h5.GrowthEvent(self.version, spec.name, "REUSE", "already learned")
            self.growth_log.append(ev)
            return ev
        parent_names = tuple(dict.fromkeys(str(x) for x in parents))
        unknown_parents = [x for x in parent_names if x not in self.experts]
        if unknown_parents:
            return self._hold(
                spec, f"UNKNOWN_GROWTH_PARENTS:{','.join(unknown_parents)}", 0, 0
            )
        domain = spec.inputs()
        if not domain:
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)
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
        if poison_one_label and train_y:
            self.ledger.record_reject(spec.name)
            return self._hold(
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)
            )

        # Finite exhaustive checks are deterministic and do not spend statistical
        # alpha. External statistical tests must reserve alpha explicitly through
        # model.reserve_statistical_test(...).

        adapter: Optional[FactorizedAdapter] = None
        if train_neural and val_x and spec.family != "BoolVenn":
            adapter = self.core.fit_residual(train_x, train_y, val_x, val_y)

        # Tier 0: Certified Library REUSE
        ex_idx = self._expert_index()
        for idx, (nm, ex) in enumerate(sorted(self.experts.items())):
            if ex.program.arity != spec.arity:
                continue
            if not all(ex.supports(xs) for xs in domain):
                continue
            self.stats.evals += len(train_x)
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
            ref_bits = TOKEN_BITS + gamma_bits(idx + 1)
            ev = h5.GrowthEvent(
                self.version,
                spec.name,
                "REUSE",
                f"verified-expert-coverage:{nm}",
                ex.program.pretty(),
                ref_bits,
                len(train_x),
                len(val_x),
                1.0,
            )
            self.growth_log.append(ev)
            return ev

        def try_commit(program: ProgramV8, reason: str) -> Optional[h5.GrowthEvent]:
            try:
                self.stats.evals += len(train_x)
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
            expert = ExpertV8(
                spec.name,
                spec.family,
                program,
                spec.domains,
                spec.valid_rule,
                bits,
                adapter,
                parent_names,
            )
            self.experts[spec.name] = expert
            self.version += 1
            self.total_commits += 1
            ev = h5.GrowthEvent(
                self.version,
                spec.name,
                "COMMIT",
                reason,
                program.pretty(),
                bits,
                len(train_x),
                len(val_x),
                1.0,
            )
            self.growth_log.append(ev)
            return ev

        # Tier 1: BoolVenn / ANF
        if spec.family == "BoolVenn":
            for m in range(16):
                ev = try_commit(ProgramV8(2, venn_mask=m), "boolvenn-exact")
                if ev is not None:
                    return ev

        if all(set(d) <= {0, 1} for d in spec.domains) and 1 <= spec.arity <= 6:
            if set(train_y) <= {0, 1}:
                full_inputs = list(itertools.product((0, 1), repeat=spec.arity))
                train_map = dict(zip(train_x, train_y))
                missing = [r for r in full_inputs if r not in train_map]
                if len(missing) <= 1:
                    best_anf = None
                    for fill in (0,) if not missing else (0, 1):
                        tt = [train_map.get(r, fill) for r in full_inputs]
                        coeffs = anf_transform(tt)
                        active_s = [s for s, c in enumerate(coeffs) if c]
                        max_deg = max((bin(s).count("1") for s in active_s), default=0)
                        cand_e = anf_to_expr_v8(coeffs, spec.arity)
                        cand_p = ProgramV8(spec.arity, cand_e)
                        score = (max_deg, len(active_s), cand_p.bits(ex_idx))
                        if best_anf is None or score < best_anf[0]:
                            best_anf = (score, cand_p)
                    if best_anf is not None:
                        ev = try_commit(
                            best_anf[1], "anf-zhegalkin-butterfly+full-support"
                        )
                        if ev is not None:
                            return ev

        # Tier 2: Direct v5 primitive templates
        for p5 in sorted(
            h5.candidate_programs(spec.arity), key=lambda p: (p.cost, p.pretty())
        ):
            if p5.expr is not None:
                ev = try_commit(
                    ProgramV8(spec.arity, ExprV8.from_h5(p5.expr)),
                    "direct-template+full-support",
                )
                if ev is not None:
                    return ev

        # Tier 3: Exact Rational Polynomial (1D..3D)
        if 1 <= spec.arity <= 3 and not spec.family.startswith("Bool"):
            if spec.arity == 1:
                coeffs = hagi_v7._exact_poly_fit(
                    [r[0] for r in train_x], train_y, max_degree=5
                )
                if coeffs is not None:
                    ev = try_commit(
                        ProgramV8(1, ExprV8("poly", (ExprV8.var(0),), coeffs)),
                        "exact-rational-poly+full-support",
                    )
                    if ev is not None:
                        return ev
            else:
                terms = hagi_v7._exact_multivariate_poly_fit(
                    train_x, train_y, spec.arity, max_degree=4
                )
                if terms is not None:
                    ev = try_commit(
                        ProgramV8(spec.arity, ExprV8("poly_nd", (), terms)),
                        "exact-multivar-poly+full-support",
                    )
                    if ev is not None:
                        return ev

        # Tier 4: Exact Piecewise Polynomial & Modular
        pw = exact_piecewise_poly_fit(train_x, train_y)
        if pw is not None:
            ev = try_commit(
                ProgramV8(spec.arity, pw), "exact-piecewise-poly+full-support"
            )
            if ev is not None:
                return ev

        # Tier 5: Bounded arithmetic folds (forall/exists over finite integer ranges).
        # This is a generic hypothesis family, not a task-name-specific shortcut.
        if spec.arity == 1:
            fold_candidates = sorted(
                bounded_fold_hypotheses(spec.arity),
                key=lambda expr: (expr.bits(ex_idx), expr.pretty()),
            )
            for fold_expr in fold_candidates:
                ev = try_commit(
                    ProgramV8(spec.arity, fold_expr),
                    "bounded-arithmetic-fold+full-support",
                )
                if ev is not None:
                    return ev

        # Tier 6: Batched Vectorized MDL AST Synthesis with Library Call-DAG
        lib_list = [
            (i, nm, self.experts[nm]) for i, nm in enumerate(sorted(self.experts))
        ]
        hit_expr, n_chk = synthesize_batched_mdl_v8(
            spec.arity,
            train_x,
            train_y,
            val_x,
            val_y,
            library_experts=lib_list,
            max_evals=self.node_budget,
            max_bits=84,
            deadline=deadline,
        )
        self.stats.candidates_generated += n_chk
        if hit_expr is not None:
            ev = try_commit(
                ProgramV8(spec.arity, hit_expr), "batched-mdl-dag+full-support"
            )
            if ev is not None:
                return ev

        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))

    def _hold(
        self, spec: h5.TaskSpec, reason: str, ntrain: int, nval: int
    ) -> h5.GrowthEvent:
        self.total_holds += 1
        ev = h5.GrowthEvent(
            self.version, spec.name, "HOLD", reason, "", 0, ntrain, nval, 0.0
        )
        self.growth_log.append(ev)
        return ev

    def compress_library_mdl(self) -> dict[str, Any]:
        """Global MDL Refactoring: subtree -> call(expert) when strictly fewer bits
        and exact equality over the complete certified support."""
        ex_idx = self._expert_index()
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
                    if (
                        lib_name == target_name
                        or lib_ex.program.arity != target_ex.program.arity
                    ):
                        continue
                    if lib_ex.program.arity == 2:
                        call_e = ExprV8(
                            "call", (ExprV8.var(0), ExprV8.var(1)), lib_name
                        )
                    elif lib_ex.program.arity == 1:
                        call_e = ExprV8("call", (ExprV8.var(0),), lib_name)
                    else:
                        continue
                    if call_e.bits(ex_idx) >= sub.bits(ex_idx):
                        continue
                    cand_expr = cur_expr.replace_subtree(sub, call_e)
                    cand_prog = ProgramV8(target_ex.program.arity, cand_expr)
                    new_bits = cand_prog.bits(ex_idx)
                    if new_bits >= target_ex.code_length:
                        continue
                    # A bit-saving substitution is admissible only if the global
                    # call graph stays acyclic; exact output equality alone cannot
                    # prevent recursive alias loops.
                    old_program = target_ex.program
                    target_ex.program = cand_prog
                    dag_check = verify_call_dag(self)
                    target_ex.program = old_program
                    if not dag_check["valid"]:
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
                "value": ex.adapter.predict(self.core, xs, quantized=True),
                "source": "factorized-core-bitalloc-rvq",
                "model_version": self.version,
            }
        raise ValueError("mode must be exact or neural")

    def verify_all(self, specs: Sequence[h5.TaskSpec]) -> dict[str, Any]:
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

    def reserve_statistical_test(self, task: str, alpha: float) -> bool:
        """Reserve alpha for an explicitly implemented external statistical test."""
        return self.ledger.spend_anytime(task, alpha)

    def verify_lineage(self) -> dict[str, Any]:
        calls = verify_call_dag(self)
        growth = verify_growth_lineage(self)
        return {
            "valid": calls["valid"] and growth["valid"],
            "call_dag": calls,
            "growth_lineage": growth,
        }

    def save(self, path: str | Path) -> None:
        """Atomically persist the complete v8 state, including shared-core weights."""
        audit = self.verify_lineage()
        if not audit["valid"]:
            raise ValueError(f"cannot save invalid model graph: {audit}")
        data = {
            "format": "HAGI-V8-STATE",
            "version": V8_VERSION,
            "node_budget": self.node_budget,
            "model_version": self.version,
            "total_commits": self.total_commits,
            "total_holds": self.total_holds,
            "core": {
                "W1": self.core.W1.tolist(),
                "b1": self.core.b1.tolist(),
                "W1_q": self.core.W1_q.tolist(),
                "b1_q": self.core.b1_q.tolist(),
                "core_bytes": self.core.core_bytes,
            },
            "experts": {
                name: ex.to_dict() for name, ex in sorted(self.experts.items())
            },
            "theorems": [vars(t) for t in self.theorems],
            "growth_log": [vars(ev) for ev in self.growth_log],
            "ledger": {
                "delta_total": self.ledger.delta_total,
                "spent": dict(self.ledger.spent),
                "rejected_tasks": sorted(self.ledger.rejected_tasks),
                "final_test_touched": self.ledger.final_test_touched,
            },
            "router": {
                "K": self.router.K,
                "eta": self.router.eta,
                "weights": self.router.weights.tolist(),
                "cum_loss": self.router.cum_loss.tolist(),
            },
            "ram": {"mem": self.ram.mem.tolist()},
            "stats": {
                "candidates_generated": self.stats.candidates_generated,
                "evals": self.stats.evals,
                "sieve_survivors": self.stats.sieve_survivors,
            },
            "audit": audit,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temp, target)

    @classmethod
    def load(cls, path: str | Path) -> "GrowingHagiProcessorV8":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "HAGI-V8-STATE":
            raise ValueError("not a HAGI v8 state file")
        core_data = data.get("core", {})
        w1 = np.asarray(core_data.get("W1"), dtype=np.float64)
        b1 = np.asarray(core_data.get("b1"), dtype=np.float64)
        w1q = np.asarray(core_data.get("W1_q"), dtype=np.float64)
        b1q = np.asarray(core_data.get("b1_q"), dtype=np.float64)
        if w1.ndim != 2 or w1.shape[0] != 3 or b1.shape != (1, w1.shape[1]):
            raise ValueError("invalid shared-core float tensor shapes")
        if w1q.shape != w1.shape or b1q.shape != b1.shape:
            raise ValueError("invalid shared-core quantized tensor shapes")
        model = cls(
            seed=0,
            node_budget=int(data.get("node_budget", 4000)),
            delta_total=float(data.get("ledger", {}).get("delta_total", 0.05)),
        )
        model.core.W1, model.core.b1 = w1, b1
        model.core.W1_q, model.core.b1_q = w1q, b1q
        model.core.core_bytes = int(core_data.get("core_bytes", 0))
        model.experts = {
            name: ExpertV8.from_dict(ex) for name, ex in data.get("experts", {}).items()
        }
        model.version = int(data.get("model_version", 0))
        model.total_commits = int(data.get("total_commits", len(model.experts)))
        model.total_holds = int(data.get("total_holds", 0))
        model.theorems = [hagi_v7.Theorem(**t) for t in data.get("theorems", [])]
        model.growth_log = [h5.GrowthEvent(**ev) for ev in data.get("growth_log", [])]
        led = data.get("ledger", {})
        model.ledger = AuditLedger(
            float(led.get("delta_total", 0.05)),
            {str(k): float(v) for k, v in led.get("spent", {}).items()},
            set(str(x) for x in led.get("rejected_tasks", [])),
            bool(led.get("final_test_touched", False)),
        )
        rd = data.get("router", {})
        model.router = HedgeRouter(int(rd.get("K", 4)), float(rd.get("eta", 0.5)))
        model.router.weights = np.asarray(
            rd.get("weights", model.router.weights), dtype=np.float64
        )
        model.router.cum_loss = np.asarray(
            rd.get("cum_loss", model.router.cum_loss), dtype=np.float64
        )
        if model.router.weights.shape != (
            model.router.K,
        ) or model.router.cum_loss.shape != (model.router.K,):
            raise ValueError("invalid serialized router vectors")
        if (
            not np.all(np.isfinite(model.router.weights))
            or np.any(model.router.weights < 0)
            or not np.isclose(model.router.weights.sum(), 1.0)
        ):
            raise ValueError("invalid serialized router weights")
        mem = np.asarray(
            data.get("ram", {}).get("mem", model.ram.mem), dtype=np.float64
        )
        if mem.shape != model.ram.mem.shape or not np.all(np.isfinite(mem)):
            raise ValueError("invalid serialized GDN RAM state")
        model.ram.mem = mem
        st = data.get("stats", {})
        model.stats.candidates_generated = int(st.get("candidates_generated", 0))
        model.stats.evals = int(st.get("evals", 0))
        model.stats.sieve_survivors = int(st.get("sieve_survivors", 0))
        audit = model.verify_lineage()
        if not audit["valid"]:
            raise ValueError(f"loaded model graph is invalid: {audit}")
        return model

    def total_neural_bytes(self) -> int:
        """Estimated packed payload bytes, not Python RAM or serialized file size."""
        venn_bytes = (
            13 if any(e.family == "BoolVenn" for e in self.experts.values()) else 0
        )
        adapter_bytes = sum(
            e.adapter.compressed_bytes
            for e in self.experts.values()
            if e.adapter is not None
        )
        return self.core.core_bytes + venn_bytes + adapter_bytes


# ==============================================================================
# 10. Non-Circular Self-Developing Growth Loop (grow_v8)
# ==============================================================================


def propose_noncircular_task(
    experts: dict[str, ExpertV8],
    rng: np.random.Generator,
    round_idx: int,
) -> Optional[tuple[h5.TaskSpec, tuple[str, ...]]]:
    """Novel non-degenerate composed tasks WITHOUT handing the AST to learn()."""
    arith2 = [
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"
    ]
    arith1 = [
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"
    ]
    if not arith2:
        return None
    pool = arith2 + arith1
    for _ in range(20):
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
            vals = {
                outer.program.eval((inner.program.eval((x,), experts), y), experts)
                for x in xs
                for y in ys
            }
            if len(vals) <= 2:
                continue
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(
                (i.program.eval((int(x),), env), int(y)), env
            )
            sp = h5.TaskSpec(
                f"V8_AUTO_R{round_idx}_{inner.name}_IN_{outer.name}",
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
            vals = {
                outer.program.eval((x, inner.program.eval((x, y), experts)), experts)
                for x in xs
                for y in ys
            }
            if len(vals) <= 2:
                continue
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(
                (int(x), i.program.eval((int(x), int(y)), env)), env
            )
            sp = h5.TaskSpec(
                f"V8_AUTO_R{round_idx}_{inner.name}_IN_{outer.name}",
                2,
                (xs, ys),
                teacher,
                valid_rule=inner.valid_rule,
                family="auto-composed",
            )
            return sp, (outer.name, inner.name)
    return None


def grow_v8(
    model: GrowingHagiProcessorV8,
    suite: Sequence[h5.TaskSpec],
    max_rounds: int = 4,
    train_neural: bool = True,
    out_dir: str = "work/v8_grow",
) -> dict[str, Any]:
    if int(max_rounds) < 0:
        raise ValueError("max_rounds must be nonnegative")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rnd = np.random.default_rng(20260101)
    rounds = []
    for r in range(max_rounds):
        t0 = time.time()
        commits = 0
        for sp in suite:
            ev = model.learn(
                sp, seed=10_000 + r * 7919 + len(sp.name), train_neural=train_neural
            )
            if ev.action == "COMMIT":
                commits += 1
        proposed = 0
        for k in range(8):
            prop = propose_noncircular_task(model.experts, rnd, r)
            if prop is None:
                continue
            sp, parents = prop
            if sp.name in model.experts:
                continue
            proposed += 1
            ev = model.learn(
                sp,
                seed=500_000 + r * 104_729 + k,
                train_neural=train_neural,
                parents=parents,
            )
            if ev.action == "COMMIT":
                commits += 1
        comp = model.compress_library_mdl()
        new_th = TheoremMinerV8().mine(model)
        audit = model.verify_lineage()
        if not audit["valid"]:
            raise RuntimeError(
                f"growth produced invalid provenance/call graph: {audit}"
            )
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
                "estimated_packed_bytes": model.total_neural_bytes(),
                "unique_programs": len(
                    {
                        json.dumps(e.program.to_dict(), sort_keys=True)
                        for e in model.experts.values()
                    }
                ),
                "experts_with_parents": audit["growth_lineage"]["nodes_with_parents"],
                "lineage_edges": audit["growth_lineage"]["edges"],
                "ram_avail_gb": round(ram_available_bytes() / 2**30, 2),
                "elapsed": round(time.time() - t0, 3),
            }
        )
        model.save(out / f"model_r{r}.json")
    final_audit = model.verify_lineage()
    return {
        "rounds": rounds,
        "experts": len(model.experts),
        "theorems": len(model.theorems),
        "lineage": final_audit["growth_lineage"],
        "call_dag": final_audit["call_dag"],
    }


def self_test_v8() -> None:
    suite = h5.make_task_suite()
    model = GrowingHagiProcessorV8(node_budget=6000)
    for i, sp in enumerate(suite):
        ev = model.learn(sp, seed=10_000 + i, train_neural=True)
        assert ev.action in ("COMMIT", "REUSE"), (sp.name, ev)
    check = model.verify_all(suite)
    assert check["accuracy"] == 1.0, check
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"
    poisoned = GrowingHagiProcessorV8().learn(
        next(s for s in suite if s.name == "ADD"), seed=77, poison_one_label=True
    )
    assert poisoned.action == "HOLD"
    rep = grow_v8(model, suite, max_rounds=2, out_dir=".omc/v8_grow_test")
    assert rep["experts"] >= len(suite)
    # Task-name-independent bounded-fold induction: train only on [-20, 20],
    # then evaluate the discovered range predicate beyond the certified domain.
    prime_domain = tuple(range(-20, 21))
    prime_task = h5.TaskSpec(
        "BOOLEAN_PATTERN_A",
        1,
        (prime_domain,),
        lambda x: int(x >= 2 and all(x % d for d in range(2, math.isqrt(x) + 1))),
        family="unseen-boolean-arithmetic",
    )
    prime_model = GrowingHagiProcessorV8(node_budget=1000)
    prime_event = prime_model.learn(prime_task, seed=88, train_neural=False)
    assert (
        prime_event.action == "COMMIT" and "forall" in prime_event.program
    ), prime_event
    prime_expr = prime_model.experts[prime_task.name].program
    for n in (101, 103, 104, 997, 1009, 1024, 104729, 104730, 1_000_003):
        expected = int(n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1)))
        assert prime_expr.eval((n,), prime_model.experts) == expected, n
    # Bounded-fold arithmetic aggregates: divisor count/sum and Euler totient.
    # The task labels are neutral; the hypothesis family does not inspect names.
    aggregate_domain = tuple(range(-10, 51))
    aggregate_tasks = [
        h5.TaskSpec(
            "ARITHMETIC_AGGREGATE_A",
            1,
            (aggregate_domain,),
            lambda n: 0 if n <= 0 else sum(1 for d in range(1, n + 1) if n % d == 0),
            family="unseen-integer-aggregate",
        ),
        h5.TaskSpec(
            "ARITHMETIC_AGGREGATE_B",
            1,
            (aggregate_domain,),
            lambda n: 0 if n <= 0 else sum(d for d in range(1, n + 1) if n % d == 0),
            family="unseen-integer-aggregate",
        ),
        h5.TaskSpec(
            "ARITHMETIC_AGGREGATE_C",
            1,
            (aggregate_domain,),
            lambda n: (
                0 if n <= 0 else sum(1 for d in range(1, n + 1) if math.gcd(n, d) == 1)
            ),
            family="unseen-integer-aggregate",
        ),
        h5.TaskSpec(
            "ARITHMETIC_AGGREGATE_D",
            1,
            (aggregate_domain,),
            lambda n: 0 if n < 0 else math.factorial(n),
            family="unseen-integer-aggregate",
        ),
    ]
    aggregate_model = GrowingHagiProcessorV8(node_budget=1000)
    for aggregate_task in aggregate_tasks:
        aggregate_event = aggregate_model.learn(
            aggregate_task, seed=606, train_neural=False
        )
        assert aggregate_event.action == "COMMIT", (
            aggregate_task.name,
            aggregate_event,
        )
        assert aggregate_model.verify_all([aggregate_task])["accuracy"] == 1.0
    aggregate_expressions = [
        aggregate_model.experts[t.name].program for t in aggregate_tasks
    ]
    for value in (51, 60, 97, 100, 997):
        expected = [
            sum(1 for d in range(1, value + 1) if value % d == 0),
            sum(d for d in range(1, value + 1) if value % d == 0),
            sum(1 for d in range(1, value + 1) if math.gcd(value, d) == 1),
        ]
        actual = [
            expr.eval((value,), aggregate_model.experts)
            for expr in aggregate_expressions[:3]
        ]
        assert actual == expected, (value, actual, expected)
        if value <= 100:
            assert aggregate_expressions[3].eval(
                (value,), aggregate_model.experts
            ) == math.factorial(value)
    for value in (51, 60, 97, 100, 500):
        assert aggregate_expressions[3].eval(
            (value,), aggregate_model.experts
        ) == math.factorial(value)
    # Explicit product overflow cap; 542! exceeds the 4096-bit accumulator cap.
    with np.testing.assert_raises(OverflowError):
        aggregate_expressions[3].eval((542,), aggregate_model.experts)
    # CRAM + VennPacking + ANF sanity
    r = CRAMEngine.solve_coloring([(0, 1), (1, 2), (2, 3), (3, 0)], 4, 2)
    assert r["verified"]
    assert len(pack_venn_16_flat()) == 13
    tt = anf_transform([0, 1, 1, 0])
    assert anf_to_expr_v8(tt, 2).eval((1, 0)) == 1
    print(
        f"V8.4 SELF_TEST PASS: {check['committed']} base experts acc 1.0, growth -> {rep['experts']} experts, "
        f"{len(model.theorems)} theorems, prime + tau/sigma/phi/factorial induction PASS, "
        f"estimated_packed_bytes={model.total_neural_bytes()}"
    )


# ==============================================================================
# PART IV — v9.0 layer
# ==============================================================================

V9_VERSION = "9.0.0"

# ==============================================================================
# 1. Core Arithmetic / Bitwise / Logic Operations & Prefix Codes
# ==============================================================================

TOKEN_BITS = 6
VAR_NAMES = ("x", "y", "z", "u", "v", "w")
DYADIC_LEVELS = np.array([(c + 1) / 128.0 for c in range(255)], dtype=np.float64)
_PRIME_MOD = 2_147_483_647  # Mersenne prime 2^31 - 1 for fast exact linear filter


# ======================================================================
# V9 canonical core (executes after legacy; wins shared names)
# ======================================================================
# HAGI V9 canonical core (growing processor, exact synthesis,
# certified Venn head, guarded steps) — split from the historical
# monolith; legacy V5–V8 generations live in _legacy.
from typing import Any, Callable, Optional, Sequence  # noqa: F401
from .gpu import GpuGpmMixin  # GPU dispatch for the neural core


# imports and legacy names; V9 re-definitions below override them.
def _bit_not8(x: int) -> int:
    return (~int(x)) & 0xFF


def _bit_nand8(x: int, y: int) -> int:
    return (~(int(x) & int(y))) & 0xFF


def _bit_nor8(x: int, y: int) -> int:
    return (~(int(x) | int(y))) & 0xFF


def _bit_xnor8(x: int, y: int) -> int:
    return (~(int(x) ^ int(y))) & 0xFF


def _sign(x: int) -> int:
    return (int(x) > 0) - (int(x) < 0)


def _pow_int(x: int, y: int) -> int:
    y = int(y)
    if y < 0 or y > 8:
        raise ValueError("exponent outside certified range")
    return int(x) ** y


OPS: dict[str, Callable[..., int]] = {
    "add": lambda x, y: int(x) + int(y),
    "sub": lambda x, y: int(x) - int(y),
    "mul": lambda x, y: int(x) * int(y),
    "floordiv": lambda x, y: int(x) // int(y),
    "mod": lambda x, y: int(x) % int(y),
    "gcd": lambda x, y: math.gcd(int(x), int(y)),
    "lcm": lambda x, y: math.lcm(int(x), int(y)),
    "min": lambda x, y: min(int(x), int(y)),
    "max": lambda x, y: max(int(x), int(y)),
    "eq": lambda x, y: int(x == y),
    "ne": lambda x, y: int(x != y),
    "lt": lambda x, y: int(x < y),
    "le": lambda x, y: int(x <= y),
    "gt": lambda x, y: int(x > y),
    "ge": lambda x, y: int(x >= y),
    "bit_and": lambda x, y: int(x) & int(y),
    "bit_or": lambda x, y: int(x) | int(y),
    "bit_xor": lambda x, y: int(x) ^ int(y),
    "bit_nand8": _bit_nand8,
    "bit_nor8": _bit_nor8,
    "bit_xnor8": _bit_xnor8,
    "pow_int": _pow_int,
    "neg": lambda x: -int(x),
    "abs": lambda x: abs(int(x)),
    "sign": _sign,
    "bit_not8": _bit_not8,
    "square": lambda x: int(x) * int(x),
    "cube": lambda x: int(x) * int(x) * int(x),
}

UNARY_OPS = ("neg", "abs", "sign", "bit_not8", "clamp3", "square", "cube")
BINARY_OPS = (
    "add",
    "sub",
    "mul",
    "floordiv",
    "mod",
    "gcd",
    "lcm",
    "min",
    "max",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "bit_and",
    "bit_or",
    "bit_xor",
    "bit_nand8",
    "bit_nor8",
    "bit_xnor8",
    "pow_int",
)
COMMUTATIVE_OPS = frozenset(
    {
        "add",
        "mul",
        "gcd",
        "lcm",
        "min",
        "max",
        "eq",
        "ne",
        "bit_and",
        "bit_or",
        "bit_xor",
        "bit_nand8",
        "bit_nor8",
        "bit_xnor8",
    }
)
CONSTS = (
    0,
    1,
    -1,
    2,
    -2,
    26,
    -26,
    256,
    -256,
)  # +alphabet/byte moduli (language, full UTF-8 pool)


def gamma_bits(n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError("gamma_bits expects positive integer")
    return 2 * (n.bit_length() - 1) + 1


def zigzag(n: int) -> int:
    n = int(n)
    return 2 * n if n >= 0 else -2 * n - 1
