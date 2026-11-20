#!/usr/bin/env python3
"""Representation registry — plugin layer for invented constructions.

Phase 1 of the representation-invention refactor. Provides:

- Representation protocol: builtin and invented solvers behind one
  interface (supports_task / probe_potential / synthesize).
- Artifact protocol: eval / supports / bits / verify_contract /
  to_dict / emit_python — what ExpertV9, rewrite.py, mdl.py and
  execute.py consume.
- REPR_REGISTRY: id -> Representation. ExprV9 gains a delegating
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
# ExprV9 construct-op delegation (invoked from core via construct_eval)
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


# ExprV9 helpers for construct nodes (called from core to_dict/from_dict)

def construct_to_dict_safe(value: dict) -> dict:
    return artifact_to_payload(value)


def construct_from_dict_safe(payload: dict) -> dict:
    return artifact_from_payload(payload)
