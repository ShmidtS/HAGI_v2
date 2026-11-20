"""V10-2: Reification — promote invented constructs to first-class ops.

When a ConstructBlueprint is accepted (transfer gate COMMIT), its
quotient function q(x) and composition can be exported as a REIFIED
macro-operator: ExprV9("reified:<name>", args) evaluates via
REIFIED_OPS registry (core). This lets BatchedMDL and downstream
synthesis COMPOSE the invented concept instead of treating it as an
opaque construct leaf.
"""
from __future__ import annotations

from typing import Any, Callable

from hagi.core import REIFIED_OPS
from hagi.invention import ConstructBlueprint

# meta cost of a reified op reference inside a program (bits)
REIFIED_OP_COST_BITS = 12


def reify_construct(bp: ConstructBlueprint, name: str,
                    fn: Callable[..., int],
                    arity: int = 1,
                    domain: Any = None) -> str:
    """Register `fn` under op-name `reified:<name>`; return the op name.

    fn(*vals) -> int must be a pure certified function (verified by the
    transfer gate before this call). domain = certified support set
    (optional; enforcement happens at artifact level, the op itself is
    a pure arithmetic map).
    """
    op_name = f"reified:{name}"
    if op_name in REIFIED_OPS:
        raise ValueError(f"reified op already exists: {op_name}")

    def _wrapped(*vals: int) -> int:
        if len(vals) != arity:
            raise ValueError(f"{op_name} expects {arity} args, got {len(vals)}")
        return int(fn(*vals))

    _wrapped.__name__ = op_name
    _wrapped.reified_meta = {
        "op": op_name, "arity": arity, "meta_bits": REIFIED_OP_COST_BITS,
        "origin": bp.q_name if hasattr(bp, "q_name") else "invented",
        "domain": sorted(domain) if domain is not None else None,
    }
    REIFIED_OPS[op_name] = _wrapped
    return op_name


def snapshot_reified() -> dict:
    """Serializable state for rewrite.py persistence."""
    out = {}
    for op, fn in REIFIED_OPS.items():
        meta = getattr(fn, "reified_meta", {})
        out[op] = dict(meta)
    return out


def restore_reified(fns: dict[str, Callable[..., int]]) -> None:
    """Restore from persisted pure functions (rewrite.py --resume)."""
    for op, fn in fns.items():
        if not op.startswith("reified:"):
            raise ValueError(f"bad reified op name: {op}")
        REIFIED_OPS[op] = fn


def clear_reified() -> None:
    REIFIED_OPS.clear()
