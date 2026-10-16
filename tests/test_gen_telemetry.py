"""Tests: Generalization telemetry port (Generalization/ModeState.lean)."""
import numpy as np
import pytest

from hagi.train.gen_telemetry import (
    GenMetrics,
    generalization_safe_step,
    hagi_potential,
    mode_drop_gate,
    pareto_select,
    qgen,
)

W = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


def test_qgen_monotone_componentwise():
    lo = GenMetrics(0.1, 0.2, 0.3, 0.4, 0.5)
    hi = GenMetrics(0.2, 0.3, 0.3, 0.5, 0.5)
    assert qgen(hi, W) >= qgen(lo, W)


def test_qgen_rejects_negative_weights():
    m = GenMetrics(1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        qgen(m, np.array([1, 1, 1, 1, -1]))


def test_potential_q_flame():
    m = GenMetrics(1, 1, 1, 1, 1)  # Qgen = 1.0 under W
    base = hagi_potential(2.0, 0.5, m, W, lam=1.0, nu=0.0, q_target=1.0)
    flamed = hagi_potential(2.0, 0.5, m, W, lam=1.0, nu=2.0, q_target=2.0)
    assert base == pytest.approx(2.5)
    assert flamed == pytest.approx(2.5 + 2.0 * 1.0)  # nu·(Qtarget−Qgen)


def test_generalization_safe_step():
    assert generalization_safe_step(1.0, lam=1.0, budget=0.2, nu=1.0, eps_q=0.1)
    assert not generalization_safe_step(
        0.3, lam=1.0, budget=0.2, nu=1.0, eps_q=0.2
    )


def test_mode_drop_gate():
    before = GenMetrics(0.9, 0.9, 0.9, 0.9, 0.9)
    dropped = GenMetrics(0.3, 0.3, 0.3, 0.3, 0.3)
    small = GenMetrics(0.85, 0.9, 0.9, 0.9, 0.9)
    accept, drop = mode_drop_gate(before, dropped, W, tol=0.1)
    assert not accept and drop > 0.1  # REJECT: Qgen fell beyond tol
    accept, drop = mode_drop_gate(before, small, W, tol=0.1)
    assert accept and drop == pytest.approx(0.01)


def test_pareto_select_skips_dominated():
    dom = GenMetrics(1.0, 1.0, 1.0, 1.0, 1.0)
    dominated = GenMetrics(0.5, 0.5, 0.5, 0.5, 0.5)  # strictly dominated
    incomparable = GenMetrics(1.5, 1.5, 0.0, 0.0, 0.0)
    cands = [(dominated, 1.0), (dom, 1.0), (incomparable, 1.0)]
    idx = pareto_select(cands, W)
    assert idx != 0  # never the strictly dominated checkpoint
    assert cands[idx][0] is dom  # uniform energy -> highest Qgen wins


def test_pareto_select_energy_tradeoff():
    weak = GenMetrics(0.4, 0.4, 0.4, 0.4, 0.4)
    strong = GenMetrics(0.9, 0.9, 0.9, 0.9, 0.9)
    idx = pareto_select([(weak, 1.0), (strong, 1.6)], W)
    # scores: 1.0−0.4 = 0.6 vs 1.6−0.9 = 0.7 -> weak wins on energy
    assert idx == 0
