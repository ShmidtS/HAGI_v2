#!/usr/bin/env python3
"""Representation registry tests (Phase 1 of invention refactor)."""

from hagi.core import Expr
from hagi.representations import (
    ContractResult, register_artifact_factory, verify_artifact_contract,
)


class SqArtifact:
    def __init__(self):
        pass

    def eval(self, xs, env=None):
        return int(xs[0]) ** 2

    def supports(self, xs):
        return 0 <= xs[0] <= 10

    def bits(self, ei=None):
        return 24

    def to_dict(self):
        return {"kind": "sq"}

    def emit_python(self, fn, env=None):
        return f"def {fn}(x): return x*x\n"


def test_construct_node_eval_and_abstain():
    register_artifact_factory("sq", lambda payload: SqArtifact())
    e = Expr("construct", (), {"repr_id": "sq", "artifact": SqArtifact()})
    assert e.eval([3]) == 9
    assert e.bits() > 0
    try:
        e.eval([99])
        raise AssertionError("expected ABSTAIN outside support")
    except ValueError:
        pass


def test_construct_roundtrip():
    register_artifact_factory("sq", lambda payload: SqArtifact())
    e = Expr("construct", (), {"repr_id": "sq", "artifact": SqArtifact()})
    e2 = Expr.from_dict(e.to_dict())
    assert all(e.eval([x]) == e2.eval([x]) for x in range(11))


def test_contract_full_domain_and_counterexample():
    art = SqArtifact()
    r = verify_artifact_contract(art, lambda x: x * x,
                                 [(i,) for i in range(11)])
    assert r.ok and r.checked_pairs == 11
    # domain includes an out-of-support point -> counterexample found
    r2 = verify_artifact_contract(art, lambda x: x * x,
                                  [(i,) for i in range(12)])
    assert not r2.ok and r2.counterexample is not None


def test_contract_catches_mismatch():
    art = SqArtifact()
    r = verify_artifact_contract(art, lambda x: x * x + 1,
                                 [(i,) for i in range(11)])
    assert not r.ok and r.reason == "mismatch"
