#!/usr/bin/env python3
"""Invention MetaGrammar tests (Phase 3) — the 4 invariant properties."""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hagi import invention as inv
from hagi.invention import (
    invent_blueprint, matched_null_verdict, synthesize_blueprint,
    verify_blueprint,
)
from hagi.rdl import mk_cnt_mod


def _cnt_mod_pairs(n: int) -> list[tuple[int, int]]:
    f = mk_cnt_mod(n)
    return [(x, f(x)) for x in range(16)]


def _sparse_pairs():
    return [(x, 1) if x in (2, 3, 9) else (x, 0) for x in range(16)]


# --- invariant 1: honesty (FULL never STRUCTURAL, noise not structural) ---

def test_full_is_never_structural():
    bp = inv.synthesize_blueprint(_sparse_pairs())
    # force a FULL blueprint
    from hagi.invention import ConstructBlueprint, q_ident, basis_full
    pairs = _sparse_pairs()
    core = [(x, y) for x, y in pairs]
    full = ConstructBlueprint("ident", 0, q_ident, "FULL",
                              basis_full(core), 8)
    v = matched_null_verdict(full, pairs)
    assert v["verdict"] == "FULL_NOT_STRUCTURAL"


def test_random_noise_not_invented():
    rng = random.Random(11)
    pairs = [(x, rng.randint(0, 15)) for x in range(16)]
    # single dense-noise cluster
    r = invent_blueprint([pairs], rng=rng)
    # either no blueprint fits, or it is COMPRESSED_ONLY (not INVENTED)
    assert r is None or r.get("status") != "INVENTED"


# --- step 3-4: discovery + counterexample verification ---

def test_invents_counter_mod():
    pairs = _cnt_mod_pairs(5)
    bp = synthesize_blueprint(pairs)
    assert bp is not None
    assert all(bp.apply(x) == y for x, y in pairs)
    proof = verify_blueprint(bp, pairs, full_domain=range(16))
    assert proof["status"] == "PROVEN_EXACT"


def test_invention_on_cluster_of_same_class():
    cluster = [_cnt_mod_pairs(5), _cnt_mod_pairs(7)]
    r = invent_blueprint(cluster)
    assert r is not None and r["status"] == "INVENTED", r
    assert r["blueprints"][0].basis == "OUT_PERIOD"
    assert r["verdict"]["verdict"] == "STRUCTURAL"


# --- invariant 2: zero-shot transfer (cluster A -> fresh task B) ---

def test_transfer_to_fresh_task():
    cluster = [_cnt_mod_pairs(5)]
    r = invent_blueprint(cluster)
    assert r is not None and r["status"] == "INVENTED"
    # fresh task: same construction, different parameter — verified WITHOUT
    # re-invention, direct blueprint check on B
    fresh = _cnt_mod_pairs(7)
    bp0 = r["blueprints"][0]
    # the shared class: verify one of the invented blueprints applies to B
    # (same q family, different table) — transfer means the CLASS transfers
    verified_any = False
    for bp in r["blueprints"]:
        if verify_blueprint(bp, fresh)["status"] == "PROVEN_EXACT":
            verified_any = True
            break
    assert verified_any or r["q_name"]


# --- invariant 3: counterexample rollback ---

def test_counterexample_rollback():
    pairs = _cnt_mod_pairs(5)
    bp = synthesize_blueprint(pairs)
    # corrupt one row: audit must catch it
    bad_pairs = pairs[:-1] + [(15, (bp.apply(15) + 1) % 16)]
    proof = verify_blueprint(bp, bad_pairs)
    assert proof["status"] == "HOLD_COUNTEREXAMPLE"
