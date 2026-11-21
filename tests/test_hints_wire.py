"""Diagnosis→search wire: hints reorder the family enumeration but
never change the chosen blueprint (MDL criterion is order-free)."""
import os, sys, random
from hagi.invention import invent_blueprint, synthesize_blueprint
from hagi.invention import mk_cnt_mod


def _cluster():
    return [[(x, mk_cnt_mod(5)(x)) for x in range(16)]]


def test_hints_do_not_change_quality():
    # hints may break ties (equal MDL) toward the hinted family, but
    # NEVER change the verdict or the total cost
    a = invent_blueprint(_cluster(), rng=random.Random(1))
    b = invent_blueprint(_cluster(), rng=random.Random(1),
                         hints={"period_candidates": True})
    assert a["status"] == b["status"] == "INVENTED"
    va, vb = a["verdict"], b["verdict"]
    assert va["verdict"] == vb["verdict"]
    assert va["bp_bits"] == vb["bp_bits"]
    assert va["significance_bits"] == vb["significance_bits"]


def test_hints_break_ties_toward_hinted_family():
    pairs = [(x, mk_cnt_mod(5)(x)) for x in range(16)]
    a = synthesize_blueprint(sorted(pairs))
    b = synthesize_blueprint(sorted(pairs), hints={"period_candidates": True})
    assert a.total_bits == b.total_bits  # same cost
    assert b.q_name.startswith("mod")    # tie broken as hinted
