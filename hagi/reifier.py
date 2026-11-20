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


def _normalize_domain(domain: Any, arity: int):
    """WAVE 179 (from local-refactor fork): certified support as a
    frozenset of tuples, arity-checked."""
    if domain is None:
        return None
    support = set()
    for point in domain:
        key = (tuple(int(v) for v in point)
               if isinstance(point, (tuple, list)) else (int(point),))
        if len(key) != arity:
            raise ValueError(
                f"domain point {point!r} has arity {len(key)}, expected {arity}")
        support.add(key)
    return frozenset(support)


def reify_construct(bp: ConstructBlueprint, name: str,
                    fn: Callable[..., int],
                    arity: int = 1,
                    domain: Any = None) -> str:
    """Register `fn` under op-name `reified:<name>`; return the op name.

    fn(*vals) -> int must be a pure certified function (verified by the
    transfer gate before this call). domain = certified support set:
    WAVE 179 — enforced at runtime (calls outside certified support
    raise, instead of silently returning a number outside the
    certificate's validity).
    """
    op_name = f"reified:{name}"
    if op_name in REIFIED_OPS:
        raise ValueError(f"reified op already exists: {op_name}")

    support = _normalize_domain(domain, arity)

    def _wrapped(*vals: int) -> int:
        if len(vals) != arity:
            raise ValueError(f"{op_name} expects {arity} args, got {len(vals)}")
        key = tuple(int(v) for v in vals)
        if support is not None and key not in support:
            raise ValueError(
                f"{op_name} outside certified support: {key}")
        return int(fn(*key))

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


def _state_sig_path(state_dir):
    from pathlib import Path
    return Path(state_dir) / "invented_reprs.sig"


def _state_hmac(payload_bytes: bytes, state_dir) -> str:
    """Machine-bound HMAC integrity tag: detects corruption/silent
    tampering of invented_reprs.json between save and resume (RSA
    identity signatures live in certificates.py)."""
    import hashlib
    import os
    from pathlib import Path
    key_file = Path(state_dir) / ".reified_key"
    if key_file.exists():
        key = key_file.read_bytes()
    else:
        key = os.urandom(32)
        key_file.write_bytes(key)
    return hashlib.sha256(key + payload_bytes).hexdigest()


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
    payload = json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")
    (d / "invented_reprs.json").write_bytes(payload)
    _state_sig_path(d).write_text(_state_hmac(payload, d))
    return len(out)


def restore_reified_ops_state(state_dir, require_tag: bool = True) -> int:
    """Restore reified ops from invented_reprs.json.

    WAVE 175: verifies the HMAC integrity tag before executing ANY
    saved source (the old path exec'ed raw JSON content unverified).
    Missing/mismatched tag fails closed by default (require_tag).
    """
    import json
    from pathlib import Path
    f = Path(state_dir) / "invented_reprs.json"
    if not f.exists():
        return 0
    payload = f.read_bytes()
    tag = _state_sig_path(state_dir)
    if tag.exists():
        if tag.read_text().strip() != _state_hmac(payload, state_dir):
            raise ValueError("reified ops integrity check FAILED — "
                             "invented_reprs.json modified after save; "
                             "refusing to exec")
    elif require_tag:
        raise ValueError("reified ops store has no integrity tag; "
                         "refusing to exec unsigned state")
    data = json.loads(payload.decode("utf-8"))
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
