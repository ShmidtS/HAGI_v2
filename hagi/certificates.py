"""Portable, self-contained finite-domain certificates for HAGI-V9.

This module intentionally imports only the Python standard library. It does
not import HAGI's learner, TaskSpec, ExprV9, or teacher functions. The verifier
reconstructs the declared finite input domain, evaluates a restricted integer
AST, and compares it to the labels embedded in the certificate artifact.

A valid artifact proves agreement with its embedded finite label table. It does
NOT authenticate who supplied those labels, prove an unbounded theorem, or
establish correctness outside the recorded domain.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAT_V1 = "HAGI-PORTABLE-CERT-V1"
FORMAT = "HAGI-PORTABLE-CERT-V2"
FORMAT_VERSION = 2
AST_SCHEMA = "hagi.expr.v1"
VERIFIER_PROFILE = "hagi.portable-verifier.v2"
TOKEN_BITS = 6
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_SIGNATURE_BYTES = 128 * 1024
MAX_DOMAIN_ROWS = 250_000
MAX_TOTAL_DOMAIN_ROWS = 250_000
MAX_TASKS = 10_000
MAX_EXPERTS = 10_000
MAX_AST_NODES = 20_000
MAX_TOTAL_AST_NODES = 250_000
MAX_AST_DEPTH = 128
MAX_AST_PAYLOAD_ITEMS = 100_000
MAX_EVAL_WORK_PER_TASK = 20_000_000
# Backward-compatible name used by older local scripts.
MAX_EVAL_NODES_PER_TASK = MAX_EVAL_WORK_PER_TASK
MAX_INTEGER_BITS = 4096

_UNARY_OPS = {"neg", "abs", "sign", "bit_not8", "square", "cube", "clamp3"}
_BINARY_OPS = {
    "add", "sub", "mul", "floordiv", "mod", "gcd", "lcm", "min", "max",
    "eq", "ne", "lt", "le", "gt", "ge", "bit_and", "bit_or", "bit_xor",
    "bit_nand8", "bit_nor8", "bit_xnor8", "pow_int", "absdiff",
}
_TERNARY_OPS = {"fma", "clamp", "median3", "mul3"}
_AST_OP_ARITY = {
    "var": 0, "const": 0, "if_cond": 3, "call": None,
    "in_set": 1, "in_ranges": 1, "anf": 0, "poly": 1, "poly_nd": 0,
    **{op: 1 for op in _UNARY_OPS},
    **{op: 2 for op in _BINARY_OPS},
    **{op: 3 for op in _TERNARY_OPS},
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


_AST_SCHEMA_DESCRIPTOR = {
    "schema": AST_SCHEMA,
    "integer_model": "signed integer; every evaluated node result is limited to 4096 bits",
    "operators": {
        "var": {"arity": 0, "semantics": "read zero-based input variable"},
        "const": {"arity": 0, "semantics": "integer literal"},
        "if_cond": {"arity": 3, "semantics": "lazy; evaluate condition then selected branch"},
        "call": {"arity": "callee-arity", "semantics": "acyclic call to expert; inputs must lie in callee support"},
        "in_set": {"arity": 1, "semantics": "integer membership in strictly increasing finite set"},
        "in_ranges": {"arity": 1, "semantics": "inclusive membership in sorted disjoint non-adjacent intervals"},
        "anf": {"arity": 0, "semantics": "xor coefficients whose subset mask is contained in input bitmask"},
        "poly": {"arity": 1, "semantics": "integer polynomial evaluated by Horner's method; coefficients low-to-high"},
        "poly_nd": {"arity": 0, "semantics": "sum of integer coefficient times each input raised to its nonnegative power"},
        "neg": {"arity": 1, "semantics": "-x"},
        "abs": {"arity": 1, "semantics": "absolute value"},
        "sign": {"arity": 1, "semantics": "-1, 0, or 1"},
        "bit_not8": {"arity": 1, "semantics": "(~x) & 255"},
        "square": {"arity": 1, "semantics": "x*x"},
        "cube": {"arity": 1, "semantics": "x*x*x"},
        "clamp3": {"arity": 1, "semantics": "min(max(x,-3),3)"},
        "add": {"arity": 2, "semantics": "a+b"},
        "sub": {"arity": 2, "semantics": "a-b"},
        "mul": {"arity": 2, "semantics": "a*b"},
        "floordiv": {"arity": 2, "semantics": "Python integer floor division"},
        "mod": {"arity": 2, "semantics": "Python integer modulo"},
        "gcd": {"arity": 2, "semantics": "nonnegative greatest common divisor"},
        "lcm": {"arity": 2, "semantics": "nonnegative least common multiple"},
        "min": {"arity": 2, "semantics": "minimum"},
        "max": {"arity": 2, "semantics": "maximum"},
        "eq": {"arity": 2, "semantics": "1 iff a equals b, else 0"},
        "ne": {"arity": 2, "semantics": "1 iff a differs from b, else 0"},
        "lt": {"arity": 2, "semantics": "1 iff a is less than b, else 0"},
        "le": {"arity": 2, "semantics": "1 iff a is at most b, else 0"},
        "gt": {"arity": 2, "semantics": "1 iff a is greater than b, else 0"},
        "ge": {"arity": 2, "semantics": "1 iff a is at least b, else 0"},
        "bit_and": {"arity": 2, "semantics": "integer bitwise and"},
        "bit_or": {"arity": 2, "semantics": "integer bitwise or"},
        "bit_xor": {"arity": 2, "semantics": "integer bitwise xor"},
        "bit_nand8": {"arity": 2, "semantics": "(~(a&b)) & 255"},
        "bit_nor8": {"arity": 2, "semantics": "(~(a|b)) & 255"},
        "bit_xnor8": {"arity": 2, "semantics": "(~(a^b)) & 255"},
        "pow_int": {"arity": 2, "semantics": "a**b; exponent must be in [0,8]"},
        "absdiff": {"arity": 2, "semantics": "abs(a-b)"},
        "fma": {"arity": 3, "semantics": "a*b+c"},
        "clamp": {"arity": 3, "semantics": "min(max(a,b),c)"},
        "median3": {"arity": 3, "semantics": "median of three integers"},
        "mul3": {"arity": 3, "semantics": "a*b*c"},
    },
}
AST_SCHEMA_SHA256 = _sha256(_AST_SCHEMA_DESCRIPTOR)


def _int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer")
    if abs(value).bit_length() > MAX_INTEGER_BITS:
        raise ValueError(f"{where} exceeds integer bit limit")
    return value


def _gamma_bits(n: int) -> int:
    if n < 1:
        raise ValueError("gamma code requires positive integer")
    return 2 * (n.bit_length() - 1) + 1


def _zigzag(n: int) -> int:
    return 2 * n if n >= 0 else -2 * n - 1


def _valid_input(xs: Sequence[int], valid_rule: str) -> bool:
    if valid_rule == "all":
        return True
    if valid_rule == "nonzero_y":
        return len(xs) >= 2 and xs[1] != 0
    if valid_rule == "ordered_yz":
        return len(xs) >= 3 and xs[1] <= xs[2]
    raise ValueError(f"unsupported valid_rule: {valid_rule}")


def _validated_domain_dimensions(
    domains: Any, valid_rule: str, *, row_limit: int = MAX_DOMAIN_ROWS,
) -> list[tuple[int, ...]]:
    if not isinstance(domains, list) or not domains or len(domains) > 8:
        raise ValueError("domains must be a non-empty list of at most 8 dimensions")
    if valid_rule not in {"all", "nonzero_y", "ordered_yz"}:
        raise ValueError(f"unsupported valid_rule: {valid_rule}")
    parsed: list[tuple[int, ...]] = []
    size = 1
    for i, dim in enumerate(domains):
        if not isinstance(dim, list) or not dim:
            raise ValueError(f"domain dimension {i} must be a non-empty list")
        vals = tuple(_int(v, f"domains[{i}][]") for v in dim)
        if len(set(vals)) != len(vals):
            raise ValueError(f"domain dimension {i} contains duplicate values")
        parsed.append(vals)
        size *= len(vals)
        if size > row_limit:
            raise ValueError("declared Cartesian domain exceeds verifier row limit")
    return parsed


def _domain_rows(domains: Any, valid_rule: str) -> list[tuple[int, ...]]:
    parsed = _validated_domain_dimensions(domains, valid_rule)
    return [tuple(int(v) for v in row) for row in itertools.product(*parsed)
            if _valid_input(row, valid_rule)]


def _program_from_library(library: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    item = library.get(name)
    if not isinstance(item, dict):
        raise ValueError(f"missing program library entry: {name}")
    return item


def _supports(item: Mapping[str, Any], xs: Sequence[int]) -> bool:
    domains = item.get("domains")
    if not isinstance(domains, list) or len(domains) != len(xs):
        return False
    if any(int(xs[i]) not in set(domains[i]) for i in range(len(xs))):
        return False
    return _valid_input(xs, str(item.get("valid_rule", "all")))


def _eval_program(
    program: Mapping[str, Any], xs: Sequence[int], library: Mapping[str, Any],
    stack: tuple[str, ...] = (), counter: list[int] | None = None,
) -> int:
    arity = _int(program.get("arity"), "program.arity")
    if len(xs) != arity:
        raise ValueError("program arity mismatch")
    mask = program.get("venn_mask")
    if mask is not None:
        mask = _int(mask, "program.venn_mask")
        if arity != 2 or not 0 <= mask <= 15 or program.get("expr") is not None:
            raise ValueError("invalid BoolVenn program")
        if int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):
            raise ValueError("BoolVenn input is not binary")
        return (mask >> ((int(xs[0]) << 1) | int(xs[1]))) & 1
    expr = program.get("expr")
    if not isinstance(expr, dict):
        raise ValueError("program has neither BoolVenn mask nor expression")
    return _eval_expr(expr, xs, library, stack, counter, 0)


def _eval_expr_inner(
    expr: Mapping[str, Any], xs: Sequence[int], library: Mapping[str, Any],
    stack: tuple[str, ...], counter: list[int] | None, depth: int,
) -> int:
    if depth > MAX_AST_DEPTH:
        raise ValueError("AST nesting limit exceeded")
    if counter is not None:
        counter[0] += 1
        if counter[0] > MAX_EVAL_WORK_PER_TASK:
            raise ValueError("evaluation work budget exceeded")
    op = expr.get("op")
    if not isinstance(op, str):
        raise ValueError("AST op must be a string")
    args = expr.get("args", [])
    if not isinstance(args, list):
        raise ValueError("AST args must be a list")
    value = expr.get("value")
    if counter is not None:
        payload_cost = 0
        if op == "in_set" and isinstance(value, list):
            payload_cost = len(value)
        elif op == "in_ranges" and isinstance(value, list):
            payload_cost = len(value)
        elif op == "anf" and isinstance(value, list) and len(value) == 2 and isinstance(value[1], list):
            payload_cost = len(value[1])
        elif op == "poly" and isinstance(value, list):
            payload_cost = len(value)
        elif op == "poly_nd" and isinstance(value, list):
            payload_cost = len(value) * (len(xs) + 1)
        counter[0] += payload_cost
        if counter[0] > MAX_EVAL_WORK_PER_TASK:
            raise ValueError("evaluation payload work budget exceeded")
    if op == "var":
        idx = _int(value, "var.value")
        if args or idx < 0 or idx >= len(xs):
            raise ValueError("invalid variable reference")
        return int(xs[idx])
    if op == "const":
        if args:
            raise ValueError("constant cannot have arguments")
        return _int(value, "const.value")
    if op == "if_cond":
        if len(args) != 3:
            raise ValueError("if_cond expects three arguments")
        # Lazy branch evaluation is required to preserve expression semantics.
        cond = _eval_expr(args[0], xs, library, stack, counter, depth + 1)
        branch = args[1] if cond else args[2]
        return _eval_expr(branch, xs, library, stack, counter, depth + 1)
    vals = [_eval_expr(a, xs, library, stack, counter, depth + 1)
            for a in args]
    unary = _UNARY_OPS
    binary = _BINARY_OPS
    if op in unary and len(vals) != 1:
        raise ValueError(f"{op} expects one argument")
    if op in binary and len(vals) != 2:
        raise ValueError(f"{op} expects two arguments")
    if op == "call":
        if not isinstance(value, str) or not value:
            raise ValueError("call requires a named expert")
        if value in stack:
            raise ValueError("recursive call cycle in certificate")
        callee = _program_from_library(library, value)
        call_xs = tuple(vals)
        if not _supports(callee, call_xs):
            raise ValueError(f"call outside support of {value}")
        return _eval_program(callee["program"], call_xs, library,
                             stack + (value,), counter)
    if op == "in_set":
        if len(vals) != 1 or not isinstance(value, list):
            raise ValueError("invalid in_set expression")
        points = tuple(_int(v, "in_set.value[]") for v in value)
        if not points or tuple(sorted(set(points))) != points:
            raise ValueError("in_set values must be strictly increasing")
        return int(vals[0] in points)
    if op == "in_ranges":
        if len(vals) != 1 or not isinstance(value, list) or not value:
            raise ValueError("invalid in_ranges expression")
        ranges = [tuple(_int(z, "in_ranges bound") for z in pair)
                  for pair in value]
        if any(len(pair) != 2 or pair[0] > pair[1] for pair in ranges):
            raise ValueError("invalid in_ranges bounds")
        if any(a[1] + 1 >= b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError("in_ranges must be sorted, disjoint and non-adjacent")
        return int(any(lo <= vals[0] <= hi for lo, hi in ranges))
    if op == "anf":
        if args or not isinstance(value, list) or len(value) != 2:
            raise ValueError("invalid anf payload")
        n_vars = _int(value[0], "anf.n_vars")
        coeffs = value[1]
        if n_vars != len(xs) or not isinstance(coeffs, list):
            raise ValueError("anf arity/payload mismatch")
        mask_x = 0
        for i, x in enumerate(xs):
            if x not in (0, 1):
                raise ValueError("anf expects binary inputs")
            if x:
                mask_x |= 1 << (n_vars - 1 - i)
        acc = 0
        for pair in coeffs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid anf coefficient")
            subset = _int(pair[0], "anf subset")
            coeff = _int(pair[1], "anf coefficient")
            if subset < 0 or subset >= (1 << n_vars):
                raise ValueError("anf subset out of range")
            if (subset & mask_x) == subset:
                acc ^= coeff
        return acc
    if op == "poly":
        if len(args) != 1 or not isinstance(value, list) or not value:
            raise ValueError("invalid poly expression")
        acc = 0
        for coef in reversed(value):
            acc = acc * vals[0] + _int(coef, "poly coefficient")
        return acc
    if op == "poly_nd":
        if args or not isinstance(value, list):
            raise ValueError("invalid poly_nd expression")
        total = 0
        for term in value:
            if not isinstance(term, list) or len(term) != 2:
                raise ValueError("invalid poly_nd term")
            coef = _int(term[0], "poly_nd coefficient")
            powers = term[1]
            if not isinstance(powers, list) or len(powers) != len(xs):
                raise ValueError("poly_nd power arity mismatch")
            product = coef
            for x, power in zip(xs, powers):
                p = _int(power, "poly_nd power")
                if p < 0 or p > 100:
                    raise ValueError("poly_nd power outside verifier limit")
                product *= int(x) ** p
            total += product
        return total
    if op in unary or op in binary:
        # The evaluator is explicit and contains no dynamic dispatch/eval.
        if op == "neg": return -vals[0]
        if op == "abs": return abs(vals[0])
        if op == "sign": return (vals[0] > 0) - (vals[0] < 0)
        if op == "bit_not8": return (~vals[0]) & 0xFF
        if op == "square": return vals[0] * vals[0]
        if op == "cube": return vals[0] * vals[0] * vals[0]
        if op == "clamp3": return min(max(vals[0], -3), 3)
        a, b = vals
        if op == "add": return a + b
        if op == "sub": return a - b
        if op == "mul": return a * b
        if op == "floordiv": return a // b
        if op == "mod": return a % b
        if op == "gcd": return math.gcd(a, b)
        if op == "lcm": return math.lcm(a, b)
        if op == "min": return min(a, b)
        if op == "max": return max(a, b)
        if op == "eq": return int(a == b)
        if op == "ne": return int(a != b)
        if op == "lt": return int(a < b)
        if op == "le": return int(a <= b)
        if op == "gt": return int(a > b)
        if op == "ge": return int(a >= b)
        if op == "bit_and": return a & b
        if op == "bit_or": return a | b
        if op == "bit_xor": return a ^ b
        if op == "bit_nand8": return (~(a & b)) & 0xFF
        if op == "bit_nor8": return (~(a | b)) & 0xFF
        if op == "bit_xnor8": return (~(a ^ b)) & 0xFF
        if op == "pow_int":
            if b < 0 or b > 8:
                raise ValueError("pow_int exponent outside [0,8]")
            return a ** b
        if op == "absdiff": return abs(a - b)
    if op in {"fma", "clamp", "median3", "mul3"}:
        if len(vals) != 3:
            raise ValueError(f"{op} expects three arguments")
        a, b, c = vals
        if op == "fma": return a * b + c
        if op == "clamp": return min(max(a, b), c)
        if op == "median3": return sorted(vals)[1]
        if op == "mul3": return a * b * c
    raise ValueError(f"unsupported or malformed AST op: {op}")


def _eval_expr(
    expr: Mapping[str, Any], xs: Sequence[int], library: Mapping[str, Any],
    stack: tuple[str, ...], counter: list[int] | None, depth: int,
) -> int:
    value = _eval_expr_inner(expr, xs, library, stack, counter, depth)
    if abs(value).bit_length() > MAX_INTEGER_BITS:
        raise ValueError("expression result exceeds integer bit limit")
    return int(value)


def _expr_bits(expr: Mapping[str, Any], expert_indices: Mapping[str, int]) -> int:
    op = expr["op"]
    value = expr.get("value")
    args = expr.get("args", [])
    if op == "var": return TOKEN_BITS + _gamma_bits(_int(value, "var") + 1)
    if op == "const": return TOKEN_BITS + _gamma_bits(_zigzag(_int(value, "const")) + 1)
    if op == "call":
        if value not in expert_indices: raise ValueError(f"unknown call index: {value}")
        return TOKEN_BITS + _gamma_bits(int(expert_indices[value]) + 1) + sum(_expr_bits(a, expert_indices) for a in args)
    if op == "anf":
        n_vars, coeffs = value
        payload = _gamma_bits(int(n_vars) + 1) + _gamma_bits(len(coeffs) + 1)
        payload += sum(_gamma_bits(int(pair[0]) + 1) for pair in coeffs)
        return TOKEN_BITS + payload
    if op == "poly":
        payload = _gamma_bits(len(value) + 1) + sum(_gamma_bits(_zigzag(int(v)) + 1) for v in value)
        return TOKEN_BITS + payload + sum(_expr_bits(a, expert_indices) for a in args)
    if op == "poly_nd":
        payload = _gamma_bits(len(value) + 1)
        for coef, powers in value:
            payload += _gamma_bits(_zigzag(int(coef)) + 1) + _gamma_bits(len(powers) + 1)
            payload += sum(_gamma_bits(int(p) + 1) for p in powers)
        return TOKEN_BITS + payload
    if op == "in_set":
        pts = [int(v) for v in value]
        payload = _gamma_bits(len(pts) + 1) + _gamma_bits(_zigzag(pts[0]) + 1)
        payload += sum(_gamma_bits(b - a) for a, b in zip(pts, pts[1:]))
        return TOKEN_BITS + payload + sum(_expr_bits(a, expert_indices) for a in args)
    if op == "in_ranges":
        ranges = value
        payload = _gamma_bits(len(ranges) + 1) + _gamma_bits(_zigzag(int(ranges[0][0])) + 1)
        payload += _gamma_bits(int(ranges[0][1]) - int(ranges[0][0]) + 1)
        for prev, curr in zip(ranges, ranges[1:]):
            payload += _gamma_bits(int(curr[0]) - int(prev[1]))
            payload += _gamma_bits(int(curr[1]) - int(curr[0]) + 1)
        return TOKEN_BITS + payload + sum(_expr_bits(a, expert_indices) for a in args)
    return TOKEN_BITS + sum(_expr_bits(a, expert_indices) for a in args)


def _program_bits(program: Mapping[str, Any], expert_indices: Mapping[str, int]) -> int:
    if program.get("venn_mask") is not None:
        return TOKEN_BITS + 4
    expr = program.get("expr")
    if not isinstance(expr, dict): raise ValueError("missing program expression")
    return _expr_bits(expr, expert_indices)


def _require_keys(
    obj: Mapping[str, Any], required: set[str], optional: set[str], where: str,
) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be an object")
    keys = set(obj)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise ValueError(f"{where} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{where} has unknown keys: {sorted(extra)}")


def _validate_expr_shape(
    expr: Any, *, arity: int, library: Mapping[str, Any], where: str,
    depth: int = 0, node_counter: list[int] | None = None,
    aggregate_counter: list[int] | None = None,
) -> set[str]:
    if depth > MAX_AST_DEPTH:
        raise ValueError(f"{where}: AST nesting limit exceeded")
    if not isinstance(expr, dict):
        raise ValueError(f"{where}: expression must be an object")
    _require_keys(expr, {"op", "args", "value"}, set(), where)
    if node_counter is None:
        node_counter = [0]
    node_counter[0] += 1
    if node_counter[0] > MAX_AST_NODES:
        raise ValueError(f"{where}: AST node limit exceeded")
    if aggregate_counter is not None:
        aggregate_counter[0] += 1
        if aggregate_counter[0] > MAX_TOTAL_AST_NODES:
            raise ValueError("total expert AST node limit exceeded")
    op = expr["op"]
    args = expr["args"]
    value = expr["value"]
    if not isinstance(op, str) or op not in _AST_OP_ARITY:
        raise ValueError(f"{where}: unsupported AST operator {op!r}")
    if not isinstance(args, list):
        raise ValueError(f"{where}: args must be an array")
    required_arity = _AST_OP_ARITY[op]
    if required_arity is not None and len(args) != required_arity:
        raise ValueError(f"{where}: {op} expects {required_arity} args")
    if op == "var":
        idx = _int(value, f"{where}.value")
        if idx < 0 or idx >= arity:
            raise ValueError(f"{where}: variable index outside program arity")
    elif op == "const":
        _int(value, f"{where}.value")
    elif op == "call":
        if not isinstance(value, str) or not value:
            raise ValueError(f"{where}: call requires expert name")
        callee = _program_from_library(library, value)
        callee_arity = _int(callee["program"]["arity"], f"{where}.callee_arity")
        if len(args) != callee_arity:
            raise ValueError(f"{where}: call arity does not match {value}")
    elif op == "in_set":
        if not isinstance(value, list) or not value or len(value) > MAX_AST_PAYLOAD_ITEMS:
            raise ValueError(f"{where}: invalid/oversized in_set payload")
        points = [_int(v, f"{where}.value[]") for v in value]
        if points != sorted(set(points)):
            raise ValueError(f"{where}: in_set must be strictly increasing")
    elif op == "in_ranges":
        if not isinstance(value, list) or not value or len(value) > MAX_AST_PAYLOAD_ITEMS:
            raise ValueError(f"{where}: invalid/oversized in_ranges payload")
        ranges: list[tuple[int, int]] = []
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"{where}: malformed in_ranges pair")
            lo = _int(pair[0], f"{where}.range.lo")
            hi = _int(pair[1], f"{where}.range.hi")
            if lo > hi:
                raise ValueError(f"{where}: reversed in_ranges interval")
            ranges.append((lo, hi))
        if any(a[1] + 1 >= b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError(f"{where}: in_ranges must be sorted, disjoint, non-adjacent")
    elif op == "anf":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{where}: malformed anf payload")
        n_vars = _int(value[0], f"{where}.anf.n_vars")
        coeffs = value[1]
        if n_vars != arity or not isinstance(coeffs, list) or len(coeffs) > MAX_AST_PAYLOAD_ITEMS:
            raise ValueError(f"{where}: invalid anf dimensions")
        for pair in coeffs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"{where}: malformed anf coefficient")
            subset = _int(pair[0], f"{where}.anf.subset")
            _int(pair[1], f"{where}.anf.coefficient")
            if subset < 0 or subset >= (1 << n_vars):
                raise ValueError(f"{where}: anf subset outside arity")
    elif op == "poly":
        if not isinstance(value, list) or not value or len(value) > MAX_AST_PAYLOAD_ITEMS:
            raise ValueError(f"{where}: invalid/oversized poly payload")
        for coefficient in value:
            _int(coefficient, f"{where}.poly.coefficient")
    elif op == "poly_nd":
        if not isinstance(value, list) or len(value) > MAX_AST_PAYLOAD_ITEMS:
            raise ValueError(f"{where}: invalid/oversized poly_nd payload")
        for term in value:
            if not isinstance(term, list) or len(term) != 2:
                raise ValueError(f"{where}: malformed poly_nd term")
            _int(term[0], f"{where}.poly_nd.coefficient")
            powers = term[1]
            if not isinstance(powers, list) or len(powers) != arity:
                raise ValueError(f"{where}: poly_nd power arity mismatch")
            for power in powers:
                p = _int(power, f"{where}.poly_nd.power")
                if p < 0 or p > 100:
                    raise ValueError(f"{where}: poly_nd power outside [0,100]")
    else:
        if value is not None:
            raise ValueError(f"{where}: operator {op} must not have a payload")
    calls: set[str] = {value} if op == "call" else set()
    for index, child in enumerate(args):
        calls.update(_validate_expr_shape(
            child, arity=arity, library=library,
            where=f"{where}.args[{index}]", depth=depth + 1,
            node_counter=node_counter, aggregate_counter=aggregate_counter,
        ))
    return calls


def _validate_library(library: Any) -> tuple[dict[str, int], dict[str, set[str]]]:
    if not isinstance(library, dict) or len(library) > MAX_EXPERTS:
        raise ValueError("expert_library must be an object within expert limit")
    indices: dict[str, int] = {}
    graph: dict[str, set[str]] = {}
    aggregate_ast_nodes = [0]
    for name, item in library.items():
        if not isinstance(name, str) or not name:
            raise ValueError("expert names must be non-empty strings")
        _require_keys(
            item,
            {"family", "program", "domains", "valid_rule", "code_length", "expert_index"},
            set(), f"expert_library[{name}]",
        )
        if not isinstance(item["family"], str) or not item["family"]:
            raise ValueError(f"{name}: family must be a non-empty string")
        valid_rule = item["valid_rule"]
        if not isinstance(valid_rule, str):
            raise ValueError(f"{name}: valid_rule must be a string")
        # Validate support metadata without materializing the Cartesian domain.
        dimensions = _validated_domain_dimensions(item["domains"], valid_rule)
        if not isinstance(item["program"], dict):
            raise ValueError(f"{name}: program must be an object")
        program = item["program"]
        _require_keys(program, {"arity", "expr", "venn_mask"}, set(), f"{name}.program")
        arity = _int(program["arity"], f"{name}.program.arity")
        if arity != len(dimensions):
            raise ValueError(f"{name}: program arity/domain mismatch")
        if program["venn_mask"] is not None:
            mask = _int(program["venn_mask"], f"{name}.venn_mask")
            if arity != 2 or not 0 <= mask <= 15 or program["expr"] is not None:
                raise ValueError(f"{name}: invalid BoolVenn program")
            dependencies: set[str] = set()
        else:
            if program["expr"] is None:
                raise ValueError(f"{name}: missing expression")
            dependencies = _validate_expr_shape(
                program["expr"], arity=arity, library=library,
                where=f"{name}.expr", aggregate_counter=aggregate_ast_nodes,
            )
        idx = _int(item["expert_index"], f"{name}.expert_index")
        if idx < 0:
            raise ValueError(f"{name}: expert_index must be non-negative")
        indices[name] = idx
        graph[name] = dependencies
        code_length = _int(item["code_length"], f"{name}.code_length")
        if code_length < 0:
            raise ValueError(f"{name}: code_length must be non-negative")
    if len(set(indices.values())) != len(indices):
        raise ValueError("expert_index values must be unique")
    if set(indices.values()) != set(range(len(indices))):
        raise ValueError("expert_index values must be contiguous from zero")
    for name, item in library.items():
        for dep in graph[name]:
            if dep not in library:
                raise ValueError(f"{name}: call references unknown expert {dep}")
        expected_bits = _program_bits(item["program"], indices)
        if expected_bits != _int(item["code_length"], f"{name}.code_length"):
            raise ValueError(f"{name}: code_length does not match AST")
    # Expert-to-expert calls must be acyclic; otherwise the finite verifier's
    # result could depend on traversal depth rather than a closed program.
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(f"recursive expert call cycle includes {name}")
        if name in visited:
            return
        visiting.add(name)
        for dep in graph[name]:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
    for name in graph:
        visit(name)
    return indices, graph


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_artifact_source(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact exceeds size limit")
        artifact = json.loads(
            p.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_object_keys,
        )
    else:
        artifact = json.loads(
            _canonical_bytes(dict(source)).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    if not isinstance(artifact, dict):
        raise ValueError("portable certificate root must be an object")
    return artifact


def _verify_artifact_hash(artifact: Mapping[str, Any]) -> str:
    claimed_digest = artifact.get("artifact_sha256")
    if not isinstance(claimed_digest, str) or len(claimed_digest) != 64:
        raise ValueError("artifact_sha256 must be a 64-character hex digest")
    try:
        int(claimed_digest, 16)
    except ValueError as exc:
        raise ValueError("artifact_sha256 is not hexadecimal") from exc
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256", None)
    actual = _sha256(unsigned)
    if claimed_digest != actual:
        raise ValueError("artifact SHA-256 mismatch")
    return claimed_digest


def _load_json_object(source: str | Path | Mapping[str, Any], *, limit: int, where: str) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > limit:
            raise ValueError(f"{where} exceeds size limit")
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_object_keys,
        )
    else:
        value = json.loads(
            _canonical_bytes(dict(source)).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _rsa_key_id(n_hex: str, e: int) -> str:
    return _sha256({
        "algorithm": "RSA_PKCS1_V1_5_SHA256",
        "e": e,
        "n_hex": n_hex.lower(),
    })


def _signature_message(artifact_digest: str) -> bytes:
    return b"HAGI-PORTABLE-CERT-SIGNATURE-V1\0" + bytes.fromhex(artifact_digest)


def verify_portable_signature(
    artifact_source: str | Path | Mapping[str, Any],
    signature_source: str | Path | Mapping[str, Any],
    trusted_public_key: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a detached RSA/SHA-256 signature using only the Python stdlib.

    The caller-supplied public key is the trust anchor. This function cannot
    determine whether that key genuinely belongs to the claimed publisher.
    """
    try:
        artifact = _parse_artifact_source(artifact_source)
        artifact_result = verify_portable_certificate(artifact)
        if not artifact_result.get("valid"):
            raise ValueError(f"certificate invalid: {artifact_result.get('error')}")
        artifact_digest = _verify_artifact_hash(artifact)

        signature = _load_json_object(
            signature_source, limit=MAX_SIGNATURE_BYTES, where="signature file",
        )
        _require_keys(
            signature,
            {"format", "algorithm", "key_id", "artifact_sha256", "signature_b64"},
            set(), "signature",
        )
        if signature["format"] != "HAGI-PORTABLE-SIG-V1":
            raise ValueError("unsupported detached signature format")
        if signature["algorithm"] != "RSA_PKCS1_V1_5_SHA256":
            raise ValueError("unsupported signature algorithm")
        if signature["artifact_sha256"] != artifact_digest:
            raise ValueError("signature is for a different artifact digest")
        if not isinstance(signature["signature_b64"], str):
            raise ValueError("signature_b64 must be a string")
        try:
            sig_bytes = base64.b64decode(signature["signature_b64"], validate=True)
        except Exception as exc:
            raise ValueError("signature_b64 is malformed") from exc

        public_key = _load_json_object(
            trusted_public_key, limit=MAX_SIGNATURE_BYTES, where="trusted public key",
        )
        _require_keys(
            public_key,
            {"format", "algorithm", "key_id", "n_hex", "e"},
            set(), "public key",
        )
        if public_key["format"] != "HAGI-PORTABLE-PUBKEY-V1":
            raise ValueError("unsupported public key format")
        if public_key["algorithm"] != "RSA_PKCS1_V1_5_SHA256":
            raise ValueError("unsupported public key algorithm")
        n_hex = public_key["n_hex"]
        if not isinstance(n_hex, str) or not n_hex or len(n_hex) > 2048:
            raise ValueError("public modulus must be a bounded hexadecimal string")
        try:
            n = int(n_hex, 16)
        except ValueError as exc:
            raise ValueError("public modulus is not hexadecimal") from exc
        e = _int(public_key["e"], "public exponent")
        if n.bit_length() < 2048 or n.bit_length() > 8192:
            raise ValueError("RSA modulus must be between 2048 and 8192 bits")
        if e < 3 or e >= n or e % 2 == 0:
            raise ValueError("invalid RSA public exponent")
        key_id = _rsa_key_id(n_hex, e)
        if public_key["key_id"] != key_id or signature["key_id"] != key_id:
            raise ValueError("signature/public-key key_id mismatch")

        k = (n.bit_length() + 7) // 8
        if len(sig_bytes) != k:
            raise ValueError("RSA signature length does not match modulus")
        sig_int = int.from_bytes(sig_bytes, "big")
        if sig_int >= n:
            raise ValueError("RSA signature representative out of range")
        encoded = pow(sig_int, e, n).to_bytes(k, "big")
        digest_info_prefix = bytes.fromhex("3031300d060960864801650304020105000420")
        digest_info = digest_info_prefix + hashlib.sha256(_signature_message(artifact_digest)).digest()
        if len(encoded) < len(digest_info) + 11:
            raise ValueError("RSA encoded message too short")
        padding_len = k - len(digest_info) - 3
        expected = b"\x00\x01" + (b"\xff" * padding_len) + b"\x00" + digest_info
        if padding_len < 8 or not hmac.compare_digest(encoded, expected):
            raise ValueError("detached RSA signature verification failed")

        return {
            "valid": True,
            "algorithm": "RSA_PKCS1_V1_5_SHA256",
            "key_id": key_id,
            "artifact_sha256": artifact_digest,
            "trust_basis": "caller-supplied public key; publisher identity requires out-of-band trust",
        }
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}


def export_portable_certificates(model: Any, specs: Sequence[Any], path: str | Path) -> dict[str, Any]:
    """Export task labels and verified symbolic programs to a portable JSON file.

    The caller supplies TaskSpec-like objects only during export. The resulting
    verifier does not require the teacher functions or HAGI learner package.
    """
    library: dict[str, Any] = {}
    expert_indices = model._expert_index()
    for name, expert in sorted(model.experts.items()):
        library[name] = {
            "family": expert.family,
            "program": expert.program.to_dict(),
            "domains": [list(d) for d in expert.domains],
            "valid_rule": expert.valid_rule,
            "code_length": int(expert.code_length),
            "expert_index": int(expert_indices.get(name, 0)),
        }
    growth = {str(row.get("task")): row for row in model.growth_log}
    tasks = []
    for spec in specs:
        rows = [tuple(int(v) for v in xs) for xs in spec.inputs()]
        if len(rows) > MAX_DOMAIN_ROWS:
            raise ValueError(f"task {spec.name} exceeds portable certificate row limit")
        outputs = [_int(int(spec.teacher(*xs)), f"teacher output for {spec.name}") for xs in rows]
        event = growth.get(spec.name, {})
        action = str(event.get("action", "HOLD"))
        source_name = None
        if spec.name in library:
            source_name = spec.name
        elif action == "REUSE":
            reason = str(event.get("reason", ""))
            prefix = "verified-expert-coverage:"
            if reason.startswith(prefix):
                source_name = reason[len(prefix):]
            elif "already learned" in reason:
                source_name = spec.name if spec.name in library else None
        if source_name is not None and source_name not in library:
            raise ValueError(f"task {spec.name} refers to missing expert {source_name}")
        program_bits = (
            _program_bits(library[source_name]["program"], expert_indices)
            if source_name is not None else None
        )
        inputs_json = [list(xs) for xs in rows]
        labels_digest = _sha256({"inputs": inputs_json, "outputs": outputs})
        tasks.append({
            "name": str(spec.name),
            "family": str(spec.family),
            "arity": int(spec.arity),
            "domains": [list(map(int, d)) for d in spec.domains],
            "valid_rule": str(spec.valid_rule),
            "domain_size": len(rows),
            "expected_outputs": outputs,
            "labels_sha256": labels_digest,
            "action": action,
            "program_source": source_name,
            "program_bits": program_bits,
            "reported_code_length": event.get("code_length"),
            "reason": event.get("reason"),
        })
    artifact = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "ast_schema": AST_SCHEMA,
        "ast_schema_sha256": AST_SCHEMA_SHA256,
        "verifier_profile": VERIFIER_PROFILE,
        "hagi_version": "9.0.0",
        "scope_notice": (
            "PROVES PROGRAM AGREEMENT WITH EMBEDDED LABELS ON THE DECLARED FINITE DOMAIN ONLY; "
            "LABEL PROVENANCE IS NOT AUTHENTICATED; NO CLAIM IS MADE OUTSIDE THE DOMAIN."
        ),
        "model_version": int(model.version),
        "total_commits": int(model.total_commits),
        "task_count": len(tasks),
        "expert_library": library,
        "tasks": tasks,
    }
    artifact["artifact_sha256"] = _sha256(artifact)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False,
                            allow_nan=False), encoding="utf-8")
    verified = verify_portable_certificate(p)
    if not verified["valid"]:
        raise ValueError(f"exported portable certificate failed self-verification: {verified}")
    return {"path": str(p), "bytes": p.stat().st_size, **verified}


def verify_portable_certificate(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Verify V1 (legacy) or V2 portable certificates using stdlib only.

    V2 binds an explicit AST schema digest and verifier profile. V1 remains
    readable under the frozen hagi.expr.v1 interpretation for migration.
    """
    try:
        artifact = _parse_artifact_source(source)
        artifact_format = artifact.get("format")
        if artifact_format == FORMAT_V1:
            expected_keys = {
                "format", "format_version", "hagi_version", "scope_notice",
                "model_version", "total_commits", "task_count", "expert_library",
                "tasks", "artifact_sha256",
            }
            _require_keys(artifact, expected_keys, set(), "V1 artifact")
            version = _int(artifact["format_version"], "format_version")
            if version != 1:
                raise ValueError("unsupported V1 format version")
            ast_schema = AST_SCHEMA
            verifier_profile = "hagi.portable-verifier.v1-legacy"
        elif artifact_format == FORMAT:
            expected_keys = {
                "format", "format_version", "ast_schema", "ast_schema_sha256",
                "verifier_profile", "hagi_version", "scope_notice", "model_version",
                "total_commits", "task_count", "expert_library", "tasks",
                "artifact_sha256",
            }
            _require_keys(artifact, expected_keys, set(), "V2 artifact")
            version = _int(artifact["format_version"], "format_version")
            if version != FORMAT_VERSION:
                raise ValueError("unsupported V2 format version")
            ast_schema = artifact["ast_schema"]
            if ast_schema != AST_SCHEMA:
                raise ValueError(f"unsupported AST schema: {ast_schema!r}")
            if artifact["ast_schema_sha256"] != AST_SCHEMA_SHA256:
                raise ValueError("AST schema digest mismatch")
            verifier_profile = artifact["verifier_profile"]
            if verifier_profile != VERIFIER_PROFILE:
                raise ValueError(f"unsupported verifier profile: {verifier_profile!r}")
        else:
            raise ValueError("wrong portable certificate format")

        claimed_digest = _verify_artifact_hash(artifact)
        if not isinstance(artifact["hagi_version"], str) or not artifact["hagi_version"]:
            raise ValueError("hagi_version must be a non-empty string")
        if not isinstance(artifact["scope_notice"], str) or not artifact["scope_notice"]:
            raise ValueError("scope_notice must be a non-empty string")
        model_version = _int(artifact["model_version"], "model_version")
        total_commits = _int(artifact["total_commits"], "total_commits")
        task_count = _int(artifact["task_count"], "task_count")
        if min(model_version, total_commits, task_count) < 0:
            raise ValueError("model_version, total_commits and task_count must be non-negative")
        if task_count > MAX_TASKS:
            raise ValueError("task_count exceeds verifier task limit")

        library = artifact["expert_library"]
        tasks = artifact["tasks"]
        if not isinstance(tasks, list) or len(tasks) != task_count:
            raise ValueError("task_count mismatch")
        expert_indices, _call_graph = _validate_library(library)

        seen_names: set[str] = set()
        proof_rows = 0
        total_domain_rows = 0
        proved = 0
        holds = 0
        reused = 0
        per_task = []
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("task entry must be an object")
            _require_keys(
                task,
                {
                    "name", "family", "arity", "domains", "valid_rule",
                    "domain_size", "expected_outputs", "labels_sha256", "action",
                    "program_source", "program_bits", "reported_code_length", "reason",
                },
                set(), "task",
            )
            name = task["name"]
            if not isinstance(name, str) or not name or name in seen_names:
                raise ValueError("empty or duplicate task name")
            seen_names.add(name)
            if not isinstance(task["family"], str) or not task["family"]:
                raise ValueError(f"{name}: family must be a non-empty string")
            arity = _int(task["arity"], f"{name}.arity")
            domains = task["domains"]
            if not isinstance(domains, list) or len(domains) != arity:
                raise ValueError(f"{name}: domain arity mismatch")
            valid_rule = task["valid_rule"]
            if not isinstance(valid_rule, str):
                raise ValueError(f"{name}: valid_rule must be a string")
            rows = _domain_rows(domains, valid_rule)
            domain_size = _int(task["domain_size"], f"{name}.domain_size")
            if len(rows) != domain_size:
                raise ValueError(f"{name}: domain_size mismatch")
            total_domain_rows += len(rows)
            if total_domain_rows > MAX_TOTAL_DOMAIN_ROWS:
                raise ValueError("total declared domain rows exceed verifier limit")

            outputs = task["expected_outputs"]
            if not isinstance(outputs, list) or len(outputs) != len(rows):
                raise ValueError(f"{name}: incomplete expected_outputs")
            outputs = [_int(y, f"{name}.expected_outputs[]") for y in outputs]
            label_digest = _sha256({"inputs": [list(r) for r in rows], "outputs": outputs})
            if label_digest != task["labels_sha256"]:
                raise ValueError(f"{name}: labels SHA-256 mismatch")

            action = task["action"]
            if not isinstance(action, str):
                raise ValueError(f"{name}: action must be a string")
            source_name = task["program_source"]
            program_bits_value = task["program_bits"]
            reported_bits = task["reported_code_length"]
            if program_bits_value is not None:
                _int(program_bits_value, f"{name}.program_bits")
            if reported_bits is not None:
                _int(reported_bits, f"{name}.reported_code_length")
            if task["reason"] is not None and not isinstance(task["reason"], str):
                raise ValueError(f"{name}: reason must be a string or null")

            if action in {"COMMIT", "REUSE"}:
                if not isinstance(source_name, str) or not source_name:
                    raise ValueError(f"{name}: committed/reused task lacks program source")
                if action == "COMMIT" and source_name != name:
                    raise ValueError(f"{name}: COMMIT must point to its own expert")
                if source_name not in library:
                    raise ValueError(f"{name}: source expert {source_name} is missing")
                source = _program_from_library(library, source_name)
                for xs in rows:
                    if not _supports(source, xs):
                        raise ValueError(f"{name}: source expert {source_name} does not support {xs}")
                program = source["program"]
                total_eval_work = 0
                for i, (xs, expected) in enumerate(zip(rows, outputs)):
                    row_work = [0]
                    # Include the root expert name in the call stack so a
                    # direct self-call is rejected just like indirect cycles.
                    got = _eval_program(program, xs, library, (source_name,), row_work)
                    total_eval_work += row_work[0]
                    if total_eval_work > MAX_EVAL_WORK_PER_TASK:
                        raise ValueError(f"{name}: evaluation work limit exceeded")
                    if got != expected:
                        raise ValueError(
                            f"{name}: counterexample at row {i}, input={xs}, "
                            f"expected={expected}, got={got}"
                        )
                recomputed_bits = _program_bits(program, expert_indices)
                if recomputed_bits != _int(program_bits_value, f"{name}.program_bits"):
                    raise ValueError(f"{name}: program bit-length mismatch")
                proof_rows += len(rows)
                proved += 1
                if action == "REUSE":
                    reused += 1
                per_task.append({
                    "task": name, "status": "PROVEN_EXACT_FINITE",
                    "checked": len(rows), "program_source": source_name,
                    "program_bits": recomputed_bits,
                })
            elif action == "HOLD":
                if source_name is not None or program_bits_value is not None:
                    raise ValueError(f"{name}: HOLD must not contain a program source or bit length")
                holds += 1
                per_task.append({"task": name, "status": "HOLD_NO_PROGRAM", "checked": 0})
            else:
                raise ValueError(f"{name}: unsupported action {action}")

        return {
            "valid": True,
            "format": artifact_format,
            "format_version": version,
            "ast_schema": ast_schema,
            "verifier_profile": verifier_profile,
            "artifact_sha256": claimed_digest,
            "tasks": len(tasks),
            "proven_exact_finite": proved,
            "reused": reused,
            "held": holds,
            "domain_rows_checked": proof_rows,
            "results": per_task,
            "scope": "finite-domain agreement with embedded labels; provenance not authenticated",
        }
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}


def upgrade_portable_certificate_v1_to_v2(
    source: str | Path | Mapping[str, Any], destination: str | Path | None = None,
) -> dict[str, Any]:
    """Migrate a V1 artifact to V2 without changing labels or programs.

    The migration makes AST/verifier semantics explicit. It does not authenticate
    label provenance; the V2 digest remains an unkeyed integrity checksum.
    """
    artifact = _parse_artifact_source(source)
    checked = verify_portable_certificate(artifact)
    if not checked.get("valid"):
        raise ValueError(f"cannot migrate invalid portable certificate: {checked.get('error')}")
    if artifact.get("format") == FORMAT:
        upgraded = artifact
    elif artifact.get("format") == FORMAT_V1:
        upgraded = dict(artifact)
        upgraded.pop("artifact_sha256", None)
        upgraded["format"] = FORMAT
        upgraded["format_version"] = FORMAT_VERSION
        upgraded["ast_schema"] = AST_SCHEMA
        upgraded["ast_schema_sha256"] = AST_SCHEMA_SHA256
        upgraded["verifier_profile"] = VERIFIER_PROFILE
        upgraded["artifact_sha256"] = _sha256(upgraded)
    else:
        raise ValueError("unsupported source certificate version")
    result = verify_portable_certificate(upgraded)
    if not result.get("valid"):
        raise ValueError(f"upgraded portable certificate failed verification: {result.get('error')}")
    if destination is not None:
        p = Path(destination)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            upgraded, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
        ), encoding="utf-8")
        result["path"] = str(p)
        result["bytes"] = p.stat().st_size
    return {"artifact": upgraded, **result}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a HAGI portable finite-domain certificate")
    parser.add_argument("certificate", help="path to portable certificate JSON")
    parser.add_argument("--upgrade-to-v2", metavar="OUTPUT", help="migrate a valid V1 artifact to V2")
    parser.add_argument("--signature", help="detached RSA/SHA-256 signature JSON")
    parser.add_argument("--trusted-public-key", help="trusted public key JSON for signature verification")
    args = parser.parse_args(argv)
    if args.upgrade_to_v2:
        try:
            migrated = upgrade_portable_certificate_v1_to_v2(args.certificate, args.upgrade_to_v2)
            result = {k: v for k, v in migrated.items() if k != "artifact"}
        except Exception as exc:
            print(json.dumps({"valid": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
            return 2
    else:
        result = verify_portable_certificate(args.certificate)
    if bool(args.signature) != bool(args.trusted_public_key):
        result = {"valid": False, "error": "--signature and --trusted-public-key must be supplied together"}
    elif args.signature and args.trusted_public_key:
        signature_result = verify_portable_signature(
            args.certificate, args.signature, args.trusted_public_key,
        )
        result = {**result, "signature_verification": signature_result}
        result["valid"] = bool(result.get("valid")) and bool(signature_result.get("valid"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("valid") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
