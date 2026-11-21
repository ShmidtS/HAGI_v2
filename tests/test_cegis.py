"""V10-1: Active CEGIS invariant tests."""
import random
import sys, os
from hagi.invention import cegis_invent
from hagi.rdl import mk_cnt_mod


def test_cegis_finds_periodic_law_without_all_labels():
    dom = range(128)
    labels = {x: mk_cnt_mod(5)(x) for x in dom}
    r = cegis_invent([3, 17, 44, 90, 101, 112, 120],
                     lambda x: labels[x], list(range(128)), dom,
                     rng=random.Random(7))
    assert r.status == "PROVEN_EXACT"
    # certification labels the full domain ONCE at the end (search
    # never saw full all_y — only seed + probe stream during search)
    assert r.queries <= len(dom) + len(dom) + 10
    assert r.blueprint is not None
    # exact on whole domain
    assert all(r.blueprint.apply(x) == labels[x] for x in dom)


def test_cegis_rejects_random_noise():
    dom = range(128)
    rng = random.Random(11)
    labels = {x: rng.randint(0, 1) for x in dom}
    r = cegis_invent([3, 17, 44, 90, 101, 112, 120],
                     lambda x: labels[x], list(range(128)), dom,
                     rng=random.Random(7))
    assert r.status != "PROVEN_EXACT"


def test_cegis_budget_is_honest():
    dom = range(128)
    labels = {x: mk_cnt_mod(5)(x) for x in dom}
    r = cegis_invent([3, 17, 44, 90, 101, 112, 120],
                     lambda x: labels[x], list(range(128)), dom,
                     query_budget=20, rng=random.Random(7))
    # budget too small for mod-5 -> honest HOLD, never silent guess
    assert r.status == "HOLD_CEGIS_BUDGET"
    assert r.queries <= 20
