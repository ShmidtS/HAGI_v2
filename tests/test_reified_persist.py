"""Reification persistence: invented ops survive cold restart."""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hagi.core import ExprV9, REIFIED_OPS
from hagi.invention import ConstructBlueprint
from hagi.reifier import (reify_construct, clear_reified,
                          save_reified_ops_state, restore_reified_ops_state)


def _bp():
    return ConstructBlueprint(q_name='bw', q_param=0, q_fn=lambda x: x,
                              basis='FULL', payload={'kind': 'FULL'},
                              meta_bits=9)


def popcount(x):
    """test top-level pure function (persistable source)"""
    return bin(int(x)).count('1')


def test_roundtrip():
    clear_reified()
    op = reify_construct(_bp(), 'popcount', popcount, domain=range(256))
    d = Path(tempfile.mkdtemp())
    assert save_reified_ops_state(d) == 1
    clear_reified()
    assert restore_reified_ops_state(d) == 1
    e = ExprV9(op, (ExprV9.var(0),))
    assert e.eval([13]) == 3
    # arity guard survives
    try:
        ExprV9(op, (ExprV9.var(0), ExprV9.var(1))).eval([1, 2])
        raise AssertionError("arity guard lost")
    except ValueError:
        pass
    clear_reified()


def test_lambda_honest_skip_in_file():
    # even in a file, lambda source is embedded in a larger statement
    # -> honest skip (no silent garbage persistence)
    clear_reified()
    op = reify_construct(_bp(), 'dyn', lambda x: x * 2)
    d = Path(tempfile.mkdtemp())
    assert save_reified_ops_state(d) == 0
    clear_reified()
clear_reified()
