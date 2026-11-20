"""V10-2: reification — invented concepts become composable ops."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hagi.core import ExprV9, REIFIED_OPS
from hagi.invention import ConstructBlueprint
from hagi.reifier import (reify_construct, snapshot_reified, clear_reified)


def _bp(q="invented"):
    return ConstructBlueprint(q_name=q, q_param=0, q_fn=lambda x: x,
                              basis="FULL", payload={"kind": "FULL"},
                              meta_bits=9, depth=1)


def test_reified_op_composes_inside_program():
    clear_reified()
    op = reify_construct(_bp(), "bitweight", lambda x: bin(int(x)).count("1"))
    # popcount(x XOR y) — composition of invented unary with builtin binary
    e = ExprV9(op, (ExprV9("bit_xor", (ExprV9.var(0), ExprV9.var(1))),))
    assert e.eval([13, 6]) == bin(13 ^ 6).count("1") == 3
    # meta cost is bounded (two-part MDL accounts for it)
    assert e.cost > 0


def test_reified_snapshot_and_arity_guard():
    clear_reified()
    op = reify_construct(_bp(), "bw2", lambda x: bin(int(x)).count("1"))
    snap = snapshot_reified()
    assert op in snap and snap[op]["arity"] == 1
    fn = ExprV9.eval  # arity guard
    e = ExprV9(op, (ExprV9.var(0), ExprV9.var(1)))
    try:
        e.eval([1, 2])
        raise AssertionError("expected arity ValueError")
    except ValueError:
        pass
    clear_reified()


def test_duplicate_name_rejected():
    clear_reified()
    reify_construct(_bp(), "dup", lambda x: x)
    try:
        reify_construct(_bp(), "dup", lambda x: x)
        raise AssertionError("expected duplicate rejection")
    except ValueError:
        pass
    clear_reified()
