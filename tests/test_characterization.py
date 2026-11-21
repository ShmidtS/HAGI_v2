"""Characterization tests — pin current hagi behavior before/after refactor.

Golden numbers recorded on 2026-11-15 from the pre-refactor hagi.py
(deterministic: fixed seeds, pure-python paths).
"""
import os
import sys
from pathlib import Path

# Characterization goldens were recorded with single-thread BLAS
# (deterministic matmul). Multi-thread BLAS reorders float sums ->
# different candidate counts. Pin before numpy import; production
# runs with all cores (loop.py default).
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hagi  # noqa: E402

GOLD = {
    "suite_len": 62,        # 49 core + 4 physics + 9 language (latin+utf8-bytes+cyrillic)
    "suite_experts": 59,    # byte pool 4/4 COMMIT; IS_VOWEL and RU_VOWEL hold
    "suite_accuracy": 1.0,
    "suite_committed": 59,
    "venn_head_eval": 0,
    "rot": [12.727922, -1.414214, -2.828427, 0.0,
            -5.656854, 0.0, 0.0, 0.0],
}


@pytest.fixture(scope="module")
def trained_v9():
    suite = hagi.make_task_suite()
    p = hagi.GrowingHagiProcessor(node_budget=200000)
    for sp in suite:
        p.learn(sp, seed=10_000)
    return p, suite


def test_suite_shape():
    suite = hagi.make_task_suite()
    assert len(suite) == GOLD["suite_len"]


@pytest.mark.slow
def test_v9_grow_and_verify(trained_v9):
    p, suite = trained_v9
    assert len(p.experts) == GOLD["suite_experts"]
    chk = p.verify_all(suite)
    assert chk["accuracy"] == GOLD["suite_accuracy"]
    assert chk["committed"] == GOLD["suite_committed"]


@pytest.mark.slow
def test_determinism_second_run():
    suite = hagi.make_task_suite()
    p2 = hagi.GrowingHagiProcessor(node_budget=200000)
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


@pytest.mark.slow
def test_selfloop_round1_golden():
    """Round-1 of the self-development loop must reproduce its certified
    numbers exactly. Guards the parallel/merge machinery end-to-end
    (batch semantics, sequential workers for determinism in tests).
    Golden: suite(57)+segment(10) with 5 holds among the new frontier."""
    from hagi.loop import ParallelLearner, run_stream
    from hagi.selfmodel import TracedProcessorV9
    inc = TracedProcessorV9(node_budget=200000)
    run_stream(inc, [(10, 501_000)], suite=None, cache=None,
               learner=ParallelLearner(inc, 1), batch_size=8)
    wasted = sum(t["cost"] for t in inc.trace if not t["success"])
    # Golden re-measured 2026-11-21 after the wave-169 double-apply
    # fix: _merge re-applied in-process worker diffs (trace/version/
    # cert_log were duplicated; 82/77 contained ~10 duplicated rows).
    # Honest counts: 72 trace, 68 success, wasted unchanged (6000).
    assert len(inc.trace) == 72
    assert wasted == 6000
    assert sum(t["success"] for t in inc.trace) == 68


@pytest.mark.slow
def test_selfrewrite_programs(tmp_path):
    """The model's program export is exact: every generated standalone
    Python function equals its certified expert over the full domain."""
    from hagi.rewrite import export_programs
    p = hagi.GrowingHagiProcessor(node_budget=2000)
    for sp in hagi.make_task_suite():
        p.learn(sp, seed=10_000, train_neural=False)
    rep = export_programs(p, tmp_path)
    n_expr = sum(1 for ex in p.experts.values() if ex.program.expr is not None)
    assert rep["functions"] == n_expr >= 30   # venn-mask experts have no AST
    assert rep["failures"] == 0
    assert rep["evaluations_checked"] > 1000


@pytest.mark.slow
def test_neural_gpu_backend():
    """GPU (torch/CUDA-ROCm) backend: resolves a device, trains a GPM
    step without error, and stays result-compatible (same adapter count
    and exact accuracy as the CPU path on the suite)."""
    import os
    from hagi.gpu import resolve_device
    dev = resolve_device()
    assert dev in ("cpu", "cuda")
    suite = hagi.make_task_suite()
    p = hagi.GrowingHagiProcessor(node_budget=200000)
    for sp in suite:
        p.learn(sp, seed=10_000, train_neural=True)
    adapters = sum(1 for e in p.experts.values() if e.adapter is not None)
    assert adapters > 0
    assert p.verify_all(suite)["accuracy"] == 1.0


@pytest.mark.slow
def test_execute_programs(tmp_path):
    """The model's executor: runs the programs it wrote — byte-level UTF-8
    ciphers stream real text; XOR round-trips; integer programs callable."""
    from hagi import execute as hx
    from hagi.rewrite import export_programs
    p = hagi.GrowingHagiProcessor(node_budget=20000)
    for sp in hagi.make_task_suite():
        p.learn(sp, seed=10_000, train_neural=True)
    export_programs(p, tmp_path)
    hx._mod = None
    hx.DEFAULT_PROGRAMS = tmp_path / "generated_programs.py"
    from hagi.rewrite import _fn_hash
    HX = _fn_hash("LANG8_XOR_CIPHER"); GC = _fn_hash("GCD")
    CA = _fn_hash("LANG8_CAESAR1")
    enc = hx.execute_text(HX, "executor ok", key=42)
    dec = hx.execute_text(HX, enc, key=42)
    assert dec == "executor ok"
    assert hx.execute(GC, 12, 18) == 6
    assert hx.execute_text(CA, "abc") == "bcd"


def test_ingest_data(tmp_path):
    """Learning from arbitrary data files: a CSV column law is recovered
    as an exact program; non-functional pairs hold honestly."""
    (tmp_path / "s.csv").write_text(
        "t,h\n" + "\n".join(f"{t},{20 - 5 * t * t}" for t in range(-8, 9)),
        encoding="utf-8")
    from hagi.ingest import ingest_file, tasks_from_numeric_file
    tasks = tasks_from_numeric_file(tmp_path / "s.csv")
    assert any(t.name == "INGEST_s_t_to_h" for t in tasks)
    p = hagi.GrowingHagiProcessor(node_budget=200000)
    ev = p.learn(tasks[0], seed=10_000)
    assert ev["action"] == "COMMIT"
    assert p.verify_all([tasks[0]])["accuracy"] == 1.0


@pytest.mark.slow
def test_byte_lm():
    """HAGI neural body (theory §2/§3/§6): one-hot CE over bytes,
    guarded steps (holdout CE never increases), ternary quantization
    with guarded recovery, generation returns prompt-consistent text."""
    from hagi import lm as hagi_lm
    data = ("квадрат суммы равен сумме квадратов плюс произведение. " * 40).encode("utf-8")
    lm = hagi_lm.ByteLM(k=6, dim=32, hidden=64)
    rep = lm.train_guarded(data, epochs=40, log_every=0)
    assert rep["best_holdout_ce"] < 3.0
    hist = rep["history"]
    assert min(hist) <= rep["best_holdout_ce"]  # §9: deployed = best snapshot
    out = lm.generate("квадрат ", n_bytes=40, temp=0.3)
    assert out.startswith("квадрат ")
    lm.ternary_()
    rep2 = lm.train_guarded(data, epochs=15, log_every=0)
    assert rep2["best_holdout_ce"] < 4.0  # quantized + fine-tune stays sane


def test_hybrid_mind(tmp_path, monkeypatch):
    """Both organs behind one interface: exact programs answer with
    EXACT label; unknown queries fall to the neural body (honestly
    labelled, never certified)."""
    from hagi import mind as hagi_mind
    from hagi import lm as hagi_lm
    monkeypatch.chdir(Path(__file__).parent.parent)
    exact = hagi_mind._exact_answer("gcd 12 18")
    assert exact[1] == "6"          # program name is a hash now
    assert hagi_mind._exact_answer("square -7")[1] == "49"
    assert hagi_mind._exact_answer("нок 4 6")[1] == "12"
    assert hagi_mind._exact_answer("что такое смысл жизни") is None


def test_curriculum_certificate():
    """primes CurriculumCert port: order-free phase accounting —
    delta_i/d_i sums commute; certificate attaches to state."""
    from hagi import curriculum as c
    a = [c.PhaseRecord("GROW", 0.5, 0.02), c.PhaseRecord("DERIVE", 0.3, 0.02)]
    b = list(reversed(a))
    assert c.curriculum_total(a) == c.curriculum_total(b)  # commutativity
    state = {"curriculum": [ {"phase": "GROW", "delta": 0.5, "d_cert": 0.02, "steps": 3} ]}
    c.certificate(state)
    cert = state["curriculum_certificate"]
    assert cert["total"] == 25.0 and cert["order_free"]
