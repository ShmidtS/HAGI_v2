"""Data-born quotients: analytical mask derivation from XOR-difference
structure (user insight: construct born from the data, not the menu)."""
import os, sys, random
from hagi.invention import (derive_mask_quotient, invent_blueprint,
                            QUOTIENTS)


def test_derives_nonadjacent_mask():
    # f depends only on bits 0 and 2 (M = 5)
    pairs = [(x, bin(x & 5).count('1') % 2) for x in range(32)]
    masks = derive_mask_quotient(pairs)
    assert any(m[0] == 'mask5' for m in masks)


def test_noise_yields_nothing():
    rng = random.Random(3)
    noise = [(x, rng.randint(0, 1)) for x in range(32)]
    assert derive_mask_quotient(noise) == []


def test_mask_grammar_grows_and_solves():
    before = len(QUOTIENTS)
    pairs = [(x, bin(x & 5).count('1') % 2) for x in range(64)]
    r = invent_blueprint([pairs], rng=random.Random(2))
    assert len(QUOTIENTS) > before            # grammar grew at runtime
    assert r["status"] == "INVENTED"
    bp = r["blueprints"][0]
    assert bp.q_name == "mask5"
    assert bp.total_bits > 0                  # MDL cost recorded
    # exact on the whole visible range
    assert all(bp.apply(x) == bin(x & 5).count('1') % 2 for x in range(256))


def test_dedup_no_double_register():
    pairs = [(x, bin(x & 5).count('1') % 2) for x in range(64)]
    invent_blueprint([pairs], rng=random.Random(2))
    n = len(QUOTIENTS)
    invent_blueprint([pairs], rng=random.Random(2))
    assert len(QUOTIENTS) == n                # dedup by name
