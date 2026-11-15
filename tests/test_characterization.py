"""Characterization tests — pin current hagi behavior before/after refactor.

Golden numbers recorded on 2026-11-15 from the pre-refactor hagi.py
(deterministic: fixed seeds, pure-python paths).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hagi  # noqa: E402

GOLD = {
    "suite_len": 49,
    "suite_experts": 49,
    "suite_accuracy": 1.0,
    "suite_committed": 49,
    "venn_head_eval": 0,
    "rot": [12.727922, -1.414214, -2.828427, 0.0,
            -5.656854, 0.0, 0.0, 0.0],
}


@pytest.fixture(scope="module")
def trained_v9():
    suite = hagi.make_task_suite()
    p = hagi.GrowingHagiProcessorV9(node_budget=2000)
    for sp in suite:
        p.learn(sp, seed=10_000)
    return p, suite


def test_suite_shape():
    suite = hagi.make_task_suite()
    assert len(suite) == GOLD["suite_len"]


def test_v9_grow_and_verify(trained_v9):
    p, suite = trained_v9
    assert len(p.experts) == GOLD["suite_experts"]
    chk = p.verify_all(suite)
    assert chk["accuracy"] == GOLD["suite_accuracy"]
    assert chk["committed"] == GOLD["suite_committed"]


def test_determinism_second_run():
    suite = hagi.make_task_suite()
    p2 = hagi.GrowingHagiProcessorV9(node_budget=2000)
    for sp in suite:
        p2.learn(sp, seed=10_000)
    assert len(p2.experts) == GOLD["suite_experts"]
    assert p2.verify_all(suite)["accuracy"] == 1.0


def test_exact_piecewise():
    r = hagi.exact_piecewise_poly_fit(
        [(1,), (2,), (3,), (4,), (5,), (6,)], [1, 0, 1, 0, 1, 0])
    assert r is not None  # parity rule found


def test_venn_and_hadamard():
    assert hagi.venn_head_eval(0b1010, [1, 0, 1, 1]) == GOLD["venn_head_eval"]
    rot, n = hagi.block_hadamard_rotate(np.array([1.0, 2, 3, 4, 5, 6, 7, 8]))
    assert np.asarray(rot).round(6).tolist() == GOLD["rot"]


def test_selfloop_round1_golden():
    """Round-1 of the self-development loop must reproduce its certified
    numbers exactly (waste cut by flooring predicted HOLDs). Guards the
    parallel/merge machinery end-to-end (batch semantics, sequential
    workers for determinism in tests)."""
    from hagi_selfloop import ParallelLearner, run_stream
    from hagi_selfmodel import TracedProcessorV9
    inc = TracedProcessorV9(node_budget=2000)
    run_stream(inc, [(10, 501_000)], suite=None, cache=None,
               learner=ParallelLearner(inc, 1), batch_size=8)
    wasted = sum(t["cost"] for t in inc.trace if not t["success"])
    assert len(inc.trace) == 59
    assert wasted == 6000
    assert sum(t["success"] for t in inc.trace) == 56
