"""Tests: LiveDelta memory cycle port (R271–R273)."""
import numpy as np
import pytest

from hagi.train.memory_cycle import (
    AdmissionMeasurements,
    AdmissionThresholds,
    Decision,
    LiveDeltaMemory,
    consolidation_frob_budget,
    memory_cycle_quality_bound,
    novelty,
)


def _thresholds():
    return AdmissionThresholds(tau=0.5, lam_m=1.0, lam_r=1.0, eps_cert=0.01)


def _meas(gain=1.0, cost=0.1, risk=0.1):
    return AdmissionMeasurements(
        gain_hat=gain, delta_c_state=cost, delta_r_protected=risk
    )


RNG = np.random.default_rng(0)


def test_novelty_empty_and_identical():
    k = RNG.normal(size=8)
    assert novelty(k, []) == 1.0
    assert novelty(k, [k]) == pytest.approx(0.0, abs=1e-6)


def test_admit_conjunction():
    t = _thresholds()
    stored = np.array([0.0, 1.0] + [0.0] * 6)  # e2
    near = np.array([0.05, 0.999] + [0.0] * 6)  # ~collinear with e2
    far = np.array([1.0, 0.0] + [0.0] * 6)  # orthogonal e1
    from hagi.train.memory_cycle import admit as _admit

    # novelty without gain -> reject
    d, why = _admit(far, [stored], _meas(gain=0.0), t)
    assert d is Decision.REJECT and "gain" in why
    # gain without novelty -> reject
    d, why = _admit(near, [stored], _meas(gain=10.0), t)
    assert d is Decision.REJECT and "novelty" in why
    # both -> admit
    d, _ = _admit(far, [stored], _meas(gain=1.0), t)
    assert d is Decision.ADMIT


def test_noise_far_away_rejected():
    t = _thresholds()
    stored = [np.array([1.0] + [0.0] * 7)]
    noise = np.array([0.0, 1.0] + [0.0] * 6)  # far but useless
    from hagi.train.memory_cycle import admit as _admit

    d, why = _admit(noise, stored, _meas(gain=0.0), t)
    assert d is Decision.REJECT  # R273: novelty != usefulness


def test_memory_grows_only_on_admit():
    mem = LiveDeltaMemory(np.zeros((4, 4)))
    a1 = np.eye(4)[0]; b1 = np.ones(4)
    a2 = np.eye(4)[1]; b2 = np.ones(4)
    assert mem.try_admit(a1, b1, _meas(), _thresholds())[0] is Decision.ADMIT
    n = len(mem.blocks)
    # a2 is novel vs stored key a1? cosine(a1,a2)=0 -> novelty 1 > tau
    d, _ = mem.try_admit(a2, b2, _meas(gain=0.0), _thresholds())
    assert d is Decision.REJECT and len(mem.blocks) == n


def test_eff_weights_identity_and_sum():
    mem = LiveDeltaMemory(np.ones((3, 3)))
    a = np.array([1.0, 2.0, 0.0]); b = np.array([0.0, 1.0, 1.0])
    mem.blocks = [(a, b)]
    assert np.allclose(mem.eff_weights(), np.ones((3, 3)) + np.outer(a, b))


def test_consolidate_step0_identity():
    mem = LiveDeltaMemory(np.ones((3, 3)))
    a = np.array([1.0, 0.0, 0.0]); b = np.array([1.0, 2.0, 0.0])
    mem.blocks = [(a, b)]
    before = mem.eff_weights()
    dev = mem.consolidate([(a, b)])  # news == olds
    assert dev == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(mem.eff_weights(), before)


def test_consolidation_budget_cauchy():
    olds = [(RNG.normal(size=4), RNG.normal(size=4)) for _ in range(3)]
    news = [(RNG.normal(size=4), RNG.normal(size=4)) for _ in range(3)]
    k = len(olds)
    eps_sum = consolidation_frob_budget(olds, news)
    old_eff = sum((np.outer(a, b) for a, b in olds), start=np.zeros((4, 4)))
    new_eff = sum((np.outer(a, b) for a, b in news), start=np.zeros((4, 4)))
    actual = float(np.sum((old_eff - new_eff) ** 2))
    assert actual <= k * eps_sum + 1e-9  # frob(eff−eff') ≤ k·Σᵢ frob(bᵢ−bᵢ')


def test_consolidate_cap_enforced():
    mem = LiveDeltaMemory(np.zeros((4, 4)))
    a = np.array([1.0, 0.0, 0.0, 0.0]); b = np.array([1.0, 0, 0, 0.0])
    mem.blocks = [(a, b)]
    with pytest.raises(ValueError):
        mem.consolidate([(np.zeros(4), np.zeros(4))], max_eps_sum=0.01)


def test_quality_bound_numeric():
    # L·√(k·Σε): exact check against a constructed case
    k, eps_sum, L = 4, 0.25, 3.0
    assert memory_cycle_quality_bound(k, eps_sum, L) == pytest.approx(
        3.0 * np.sqrt(1.0)
    )
    with pytest.raises(ValueError):
        memory_cycle_quality_bound(1, -1.0, 1.0)
