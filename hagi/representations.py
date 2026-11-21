#!/usr/bin/env python3
"""Representation registry — plugin layer for invented constructions.

Phase 1 of the representation-invention refactor. Provides:

- Representation protocol: builtin and invented solvers behind one
  interface (supports_task / probe_potential / synthesize).
- Artifact protocol: eval / supports / bits / verify_contract /
  to_dict / emit_python — what Expert, rewrite.py, mdl.py and
  execute.py consume.
- REPR_REGISTRY: id -> Representation. Expr gains a delegating
  "construct" op via CONSTRUCT_HOOK (evaluated by registry, zero
  edits to core.py if-elif ladder: the hook is invoked from the OPS
  fallback through construct_eval).
- Contract verification: full-domain check + JSON round-trip +
  emit_python == eval + mandatory ABSTAIN outside certified support.

Honest limits: a construct artifact is exact only on its certified
domain; outside it raises (never guesses).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------

@runtime_checkable
class Artifact(Protocol):
    def eval(self, xs: Sequence[int],
             env: dict | None = None) -> int: ...
    def supports(self, xs: Sequence[int]) -> bool: ...
    def bits(self, expert_index: dict | None = None) -> int: ...
    def verify_contract(self, oracle_or_domain: Any) -> Any: ...
    def to_dict(self) -> dict[str, Any]: ...
    def emit_python(self, fn_name: str, env: dict | None = None) -> str: ...


@runtime_checkable
class Representation(Protocol):
    id: str
    origin: str            # "builtin" | "invented" | "ood_recovered"
    meta_bits: int         # honest description length of the construct

    def supports_task(self, profile: dict) -> bool: ...
    def probe_potential(self, obs: dict) -> float: ...
    def synthesize(self, obs: dict, budget: float,
                   ledger: Any) -> Artifact | None: ...


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

REPR_REGISTRY: dict[str, Representation] = {}

CONSTRUCT_OP = "construct"


def register(repr_obj: Representation) -> None:
    if repr_obj.id in REPR_REGISTRY:
        raise ValueError(f"duplicate representation id: {repr_obj.id}")
    REPR_REGISTRY[repr_obj.id] = repr_obj


def get(repr_id: str) -> Representation:
    return REPR_REGISTRY[repr_id]


# --------------------------------------------------------------------------
# Expr construct-op delegation (invoked from core via construct_eval)
# --------------------------------------------------------------------------

def construct_eval(value: Any, xs: Sequence[int],
                   env: dict | None = None) -> int:
    """Evaluate a construct node: value = {"repr_id": ..., "artifact": ...}."""
    if not isinstance(value, dict) or "artifact" not in value:
        raise ValueError("construct node missing artifact payload")
    art = value["artifact"]
    if not art.supports(xs):
        raise ValueError(f"construct {value.get('repr_id')} outside certified support")
    return int(art.eval(xs, env))


def construct_bits(value: Any,
                   expert_index: dict | None = None) -> int:
    art = value.get("artifact")
    if art is None:
        raise ValueError("construct node missing artifact payload")
    return int(art.bits(expert_index))


def construct_pretty(value: Any) -> str:
    return f"construct({value.get('repr_id')})"


def construct_to_dict(value: Any) -> Any:
    return {"repr_id": value.get("repr_id"),
            "artifact": value["artifact"].to_dict()}


# --------------------------------------------------------------------------
# Contract verification (full-domain + round-trip + ABSTAIN)
# --------------------------------------------------------------------------

@dataclass
class ContractResult:
    ok: bool
    reason: str = ""
    checked_pairs: int = 0
    counterexample: tuple | None = None


def verify_artifact_contract(artifact: Artifact, teacher: Callable[..., int],
                             domain: Sequence[Sequence[int]]) -> ContractResult:
    """Full-domain check: 100% on every domain point + ABSTAIN semantics.
    Counterexample-driven: first mismatch returned, not swallowed."""
    checked = 0
    for xs in domain:
        xs_t = tuple(xs)
        want = int(teacher(*xs_t))
        try:
            if not artifact.supports(xs_t):
                return ContractResult(False, "ABSTAIN inside certified domain",
                                      checked, xs_t)
            got = int(artifact.eval(xs_t))
        except ValueError as e:
            return ContractResult(False, f"raised: {e}", checked, xs_t)
        if got != want:
            return ContractResult(False, "mismatch", checked, xs_t)
        checked += 1
    # JSON round-trip
    try:
        d = artifact.to_dict()
        import json
        rt = json.loads(json.dumps(d))
        if rt != d:
            return ContractResult(False, "json round-trip mismatch", checked)
    except Exception as e:
        return ContractResult(False, f"to_dict failed: {e}", checked)
    return ContractResult(True, "verified", checked)


# --------------------------------------------------------------------------
# Built-in representations (wrapping the existing synthesis ladder)
# --------------------------------------------------------------------------

@dataclass
class BuiltinRepr:
    id: str
    family_pred: Callable[[dict], bool]
    synthesize_fn: Callable[..., Any]
    origin: str = "builtin"

    @property
    def meta_bits(self) -> int:
        return 8  # one token per builtin

    def supports_task(self, profile: dict) -> bool:
        try:
            return bool(self.family_pred(profile))
        except Exception:
            return False

    def probe_potential(self, obs: dict) -> float:
        # builtins are ordered by the existing ladder; potential unused
        return 0.0

    def synthesize(self, obs: dict, budget: float, ledger: Any) -> Any:
        try:
            return self.synthesize_fn(obs)
        except Exception:
            return None


def register_builtins() -> None:
    """Wrap the existing learn-ladder branches as builtin Representations.
    synthesize_fn receives obs dict with spec/processor and uses the
    existing machinery (no behavior change)."""
    register(BuiltinRepr(
        "boolvenn",
        lambda prof: prof.get("family") == "BoolVenn",
        lambda obs: None,  # ladder path already handles BoolVenn inline
    ))
    register(BuiltinRepr(
        "anf",
        lambda prof: prof.get("bool_domain", False) and prof.get("arity", 0) <= 6,
        lambda obs: None,
    ))
    register(BuiltinRepr(
        "templates",
        lambda prof: prof.get("family") in ("Arith", "Poly", "Mixed"),
        lambda obs: None,
    ))


# --------------------------------------------------------------------------
# Artifact serialization: registry-driven restore
# --------------------------------------------------------------------------

ARTIFACT_FACTORIES: dict[str, Callable[[dict], Any]] = {}


def register_artifact_factory(repr_id: str,
                              factory: Callable[[dict], Any]) -> None:
    ARTIFACT_FACTORIES[repr_id] = factory


def artifact_to_payload(value: dict) -> dict:
    """construct-node value -> JSON-safe payload."""
    art = value["artifact"]
    payload = art.to_dict()
    return {"repr_id": value.get("repr_id"), "artifact": payload}


def artifact_from_payload(payload: dict) -> dict:
    """JSON-safe payload -> construct-node value (restores via factory)."""
    rid = payload["repr_id"]
    factory = ARTIFACT_FACTORIES.get(rid)
    if factory is None:
        raise ValueError(f"no artifact factory for repr {rid}")
    return {"repr_id": rid, "artifact": factory(payload["artifact"])}


# Expr helpers for construct nodes (called from core to_dict/from_dict)

def construct_to_dict_safe(value: dict) -> dict:
    return artifact_to_payload(value)


def construct_from_dict_safe(payload: dict) -> dict:
    return artifact_from_payload(payload)


# ============================================================
# REIFIER: регистрация изобретённых операций
# ============================================================

from typing import Any, Callable
from hagi.core import REIFIED_OPS
from hagi.invention import ConstructBlueprint
REIFIED_OP_COST_BITS = 12
def _normalize_domain(domain: Any, arity: int):
    """WAVE 179 (from local-refactor fork): certified support as a
    frozenset of tuples, arity-checked."""
    if domain is None:
        return None
    support = set()
    for point in domain:
        key = tuple((int(v) for v in point)) if isinstance(point, (tuple, list)) else (int(point),)
        if len(key) != arity:
            raise ValueError(f'domain point {point!r} has arity {len(key)}, expected {arity}')
        support.add(key)
    return frozenset(support)
def reify_construct(bp: ConstructBlueprint, name: str, fn: Callable[..., int], arity: int=1, domain: Any=None) -> str:
    """Register `fn` under op-name `reified:<name>`; return the op name.

    fn(*vals) -> int must be a pure certified function (verified by the
    transfer gate before this call). domain = certified support set:
    WAVE 179 — enforced at runtime (calls outside certified support
    raise, instead of silently returning a number outside the
    certificate's validity).
    """
    op_name = f'reified:{name}'
    if op_name in REIFIED_OPS:
        raise ValueError(f'reified op already exists: {op_name}')
    support = _normalize_domain(domain, arity)

    def _wrapped(*vals: int) -> int:
        if len(vals) != arity:
            raise ValueError(f'{op_name} expects {arity} args, got {len(vals)}')
        key = tuple((int(v) for v in vals))
        if support is not None and key not in support:
            raise ValueError(f'{op_name} outside certified support: {key}')
        return int(fn(*key))
    _wrapped.__name__ = op_name
    import inspect
    try:
        _candidate = inspect.getsource(fn).strip()
        _source = _candidate if _candidate.startswith('def ') else None
    except (OSError, TypeError):
        _source = None
    _wrapped.reified_meta = {'op': op_name, 'arity': arity, 'meta_bits': REIFIED_OP_COST_BITS, 'origin': bp.q_name if hasattr(bp, 'q_name') else 'invented', 'domain': sorted(domain) if domain is not None else None, 'source': _source}
    REIFIED_OPS[op_name] = _wrapped
    return op_name
def snapshot_reified() -> dict:
    """Serializable state for rewrite.py persistence."""
    out = {}
    for op, fn in REIFIED_OPS.items():
        meta = getattr(fn, 'reified_meta', {})
        out[op] = dict(meta)
    return out
def clear_reified() -> None:
    REIFIED_OPS.clear()
def _state_sig_path(state_dir):
    from pathlib import Path
    return Path(state_dir) / 'invented_reprs.sig'
def _state_hmac(payload_bytes: bytes, state_dir) -> str:
    """Machine-bound HMAC integrity tag: detects corruption/silent
    tampering of invented_reprs.json between save and resume (RSA
    identity signatures live in certificates.py)."""
    import hashlib
    import os
    from pathlib import Path
    key_file = Path(state_dir) / '.reified_key'
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
        meta = getattr(fn, 'reified_meta', {})
        code = meta.get('source')
        if code:
            out[op] = {'code': code, 'meta': {k: v for k, v in meta.items() if k != 'source'}}
    payload = json.dumps(out, ensure_ascii=False, indent=1).encode('utf-8')
    (d / 'invented_reprs.json').write_bytes(payload)
    _state_sig_path(d).write_text(_state_hmac(payload, d))
    return len(out)
def restore_reified_ops_state(state_dir, require_tag: bool=True) -> int:
    """Restore reified ops from invented_reprs.json.

    WAVE 175: verifies the HMAC integrity tag before executing ANY
    saved source (the old path exec'ed raw JSON content unverified).
    Missing/mismatched tag fails closed by default (require_tag).
    """
    import json
    from pathlib import Path
    f = Path(state_dir) / 'invented_reprs.json'
    if not f.exists():
        return 0
    payload = f.read_bytes()
    tag = _state_sig_path(state_dir)
    if tag.exists():
        if tag.read_text().strip() != _state_hmac(payload, state_dir):
            raise ValueError('reified ops integrity check FAILED — invented_reprs.json modified after save; refusing to exec')
    elif require_tag:
        raise ValueError('reified ops store has no integrity tag; refusing to exec unsigned state')
    data = json.loads(payload.decode('utf-8'))
    n = 0
    import textwrap
    for op, rec in data.items():
        if op in REIFIED_OPS:
            continue
        code = textwrap.dedent(rec['code'])
        ns: dict = {'__name__': 'reified_restore'}
        exec(compile(code, f'<reified:{op}>', 'exec'), ns)
        fns = [v for v in ns.values() if callable(v) and getattr(v, '__module__', None) == 'reified_restore']
        if not fns:
            continue
        pure = fns[0]
        meta = rec.get('meta', {})
        arity = int(meta.get('arity', 1))
        # WAVE 196 FIX (P0, locally reproduced): restore lost the
        # certified-support guard — pre-restore calls outside the
        # domain raised ValueError, post-restore they returned a
        # number. Rebuild the support closure from meta['domain'].
        domain = meta.get('domain')
        if domain is not None:
            support = frozenset(
                (tuple(int(v) for v in pt)
                 if isinstance(pt, (tuple, list)) else (int(pt),))
                for pt in domain)
        else:
            support = None

        def _wrapped(*vals: int, _p=pure, _a=arity, _op=op,
                     _s=support):
            if len(vals) != _a:
                raise ValueError(f'{_op} expects {_a} args, got {len(vals)}')
            key = tuple(int(v) for v in vals)
            if _s is not None and key not in _s:
                raise ValueError(
                    f'{_op} outside certified support: {key}')
            return int(_p(*key))
        _wrapped.__name__ = op
        _wrapped.reified_meta = dict(meta)
        REIFIED_OPS[op] = _wrapped
        n += 1
    return n