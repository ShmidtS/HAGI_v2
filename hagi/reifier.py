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
    import inspect
    try:
        _candidate = inspect.getsource(fn).strip()
        # only top-level `def` persists: a lambda's source is embedded
        # in a larger statement and is not self-contained (honest skip)
        _source = _candidate if _candidate.startswith("def ") else None
    except (OSError, TypeError):
        _source = None  # dynamic/built-in fn: not persistable (honest)
    _wrapped.reified_meta = {
        "op": op_name, "arity": arity, "meta_bits": REIFIED_OP_COST_BITS,
        "origin": bp.q_name if hasattr(bp, "q_name") else "invented",
        "domain": sorted(domain) if domain is not None else None,
        "source": _source,
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


def clear_reified() -> None:
    REIFIED_OPS.clear()


def save_reified_ops_state(state_dir) -> int:
    """Persist reified ops as source + meta (invented_reprs.json).
    Returns the number saved. Source-based: survives cold restart."""
    import json
    from pathlib import Path
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for op, fn in REIFIED_OPS.items():
        meta = getattr(fn, "reified_meta", {})
        code = meta.get("source")
        if code:
            out[op] = {"code": code, "meta": {k: v for k, v in meta.items()
                                              if k != "source"}}
    (d / "invented_reprs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(out)


def restore_reified_ops_state(state_dir) -> int:
    """Restore reified ops from invented_reprs.json (exec in a clean
    namespace; the __reified_meta__ attribute is re-attached)."""
    import json
    from pathlib import Path
    f = Path(state_dir) / "invented_reprs.json"
    if not f.exists():
        return 0
    data = json.loads(f.read_text(encoding="utf-8"))
    n = 0
    import textwrap
    for op, rec in data.items():
        if op in REIFIED_OPS:
            continue  # already live (idempotent resume)
        code = textwrap.dedent(rec["code"])
        ns: dict = {"__name__": "reified_restore"}
        exec(compile(code, f"<reified:{op}>", "exec"), ns)
        fns = [v for v in ns.values() if callable(v)
               and getattr(v, "__module__", None) == "reified_restore"]
        if not fns:
            continue
        pure = fns[0]
        meta = rec.get("meta", {})
        arity = int(meta.get("arity", 1))

        def _wrapped(*vals: int, _p=pure, _a=arity, _op=op):
            if len(vals) != _a:
                raise ValueError(f"{_op} expects {_a} args, got {len(vals)}")
            return int(_p(*vals))

        _wrapped.__name__ = op
        _wrapped.reified_meta = dict(meta)
        REIFIED_OPS[op] = _wrapped
        n += 1
    return n
