"""Tests for the growth, batch and insight laws.

Each test is the executable form of a Lean statement; the theorem is
named so a failure points at the broken proposition.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from hagi.train.batch_law import (
    optimal_batch,
    speedup_vs_batch,
    step_time,
    time_lower_bound,
)
from hagi.train.growth_law import (
    capability_floor,
    gain_transfer,
    required_successes,
    risk_floor,
    takeoff_horizon,
)
from hagi.train.insight_currency import (
    cycle_bracket,
    drift_null_ok,
    kl_insight_gain,
)


# --- Audit/Exactness.lean: the AM-GM batch law -------------------------


def test_amgm_equality_and_time_floor() -> None:
    """``B^2 = Bn*t0/c`` makes ``T(B*)`` equal the AM-GM floor exactly."""
    random.seed(10)
    for _ in range(2000):
        bn = random.uniform(1.0, 1e6)
        t0 = random.uniform(1e-4, 10.0)
        c = random.uniform(1e-9, 1e-3)
        b = optimal_batch(bn, t0, c)
        assert c * b + bn * t0 / b == pytest.approx(
            2.0 * math.sqrt(c * bn * t0), rel=1e-6
        )
        assert step_time(b, bn, t0, c) == pytest.approx(
            time_lower_bound(bn, t0, c), rel=1e-6
        )


def test_amgm_uniqueness() -> None:
    """No other batch size beats ``B*``."""
    random.seed(11)
    for _ in range(1000):
        bn = random.uniform(1.0, 1e6)
        t0 = random.uniform(1e-4, 10.0)
        c = random.uniform(1e-9, 1e-3)
        b = optimal_batch(bn, t0, c)
        best = step_time(b, bn, t0, c)
        for factor in (0.3, 0.7, 1.5, 3.0):
            assert step_time(b * factor, bn, t0, c) >= best - 1e-12


def test_optimal_batch_grows_with_overhead() -> None:
    """``B* ~ sqrt(t0)``: worse per-step overhead demands a LARGER batch."""
    bn, c = 32768.0, 1e-6
    small = optimal_batch(bn, 0.01, c)
    large = optimal_batch(bn, 0.04, c)
    assert large == pytest.approx(2.0 * small, rel=1e-9)


def test_speedup_ratio_is_a_real_speedup() -> None:
    """``T(batch)/T(B*)`` is 1 at the optimum and grows as batch shrinks."""
    args = (32768.0, 0.05, 1e-6)
    assert speedup_vs_batch(optimal_batch(*args), *args) == pytest.approx(1.0, rel=1e-9)
    tiny = speedup_vs_batch(1.0, *args)
    larger = speedup_vs_batch(100.0, *args)
    assert 1.0 <= larger < tiny


def test_optimal_batch_rejects_nonpositive_inputs() -> None:
    with pytest.raises(ValueError):
        optimal_batch(0.0, 0.1, 1e-6)
    with pytest.raises(ValueError):
        optimal_batch(100.0, 0.0, 1e-6)


# --- Dynamics/FastGrowth.lean: the growth meta-law ----------------------


def test_capability_takeoff_counted() -> None:
    """``C_T >= C_0 (1+alpha)^(sum s_t)`` -- replay the per-cycle law."""
    random.seed(12)
    for _ in range(2000):
        alpha = random.uniform(0.001, 1.0)
        c0 = random.uniform(0.1, 100.0)
        successes = [random.randint(0, 1) for _ in range(12)]
        c = c0
        for s in successes:
            c *= (1.0 + alpha) ** s
        assert c >= capability_floor(c0, alpha, sum(successes)) - 1e-9


def test_failures_are_neutral() -> None:
    """``s = 0`` multiplies by 1 -- a failed cycle costs nothing."""
    assert capability_floor(2.0, 0.1, 0) == pytest.approx(2.0)
    assert capability_floor(2.0, 0.1, 5) == pytest.approx(2.0 * 1.1**5)


def test_capability_gain_transfer() -> None:
    """``Rext2 <= Rext1 - Gamma`` when the gap does not worsen."""
    random.seed(13)
    for _ in range(5000):
        e1 = random.uniform(0.0, 10.0)
        g1 = random.uniform(0.0, 5.0)
        gamma = random.uniform(0.0, 3.0)
        e2 = e1 - gamma
        g2 = random.uniform(0.0, g1)  # hgap: g2 <= g1
        assert gain_transfer(e1, g1, e2, g2, gamma) >= e2 + g2 - 1e-12


def test_required_successes_is_the_exact_inverse() -> None:
    """``N = ceil(ln(target)/ln(1+alpha))`` reaches the target."""
    random.seed(14)
    for _ in range(2000):
        alpha = random.uniform(0.01, 0.5)
        target = random.uniform(1.1, 10.0)
        n = required_successes(target, alpha)
        assert capability_floor(1.0, alpha, n) >= target - 1e-9
        if n > 0:
            assert capability_floor(1.0, alpha, n - 1) < target


def test_takeoff_horizon_scales_inversely_with_the_rate() -> None:
    """Halving the success rate doubles the horizon, up to ceiling rounding."""
    fast = takeoff_horizon(0.5, 0.1, 10.0)
    slow = takeoff_horizon(0.25, 0.1, 10.0)
    assert slow == 2 * fast or slow == 2 * fast - 1
    with pytest.raises(ValueError):
        takeoff_horizon(0.0, 0.1, 10.0)


def test_risk_floor_decays_geometrically() -> None:
    assert risk_floor(1.0, 0.2, 10) == pytest.approx(0.8**10)
    assert risk_floor(1.0, 0.2, 20) == pytest.approx(0.8**20)
    with pytest.raises(ValueError):
        risk_floor(1.0, 1.5, 10)


# --- Autonomy/Insight.lean: the insight currency ------------------------


def test_ce_gap_kl_identity() -> None:
    """``CE(q,p') - CE(q,p) == KL(q,p') - KL(q,p)`` -- the gain has one value."""

    def ce(q: torch.Tensor, p: torch.Tensor) -> float:
        return float(-(q * p.clamp_min(1e-30).log()).sum())

    def kl(q: torch.Tensor, p: torch.Tensor) -> float:
        return float(
            (q * (q.clamp_min(1e-30).log() - p.clamp_min(1e-30).log())).sum()
        )

    random.seed(15)
    for _ in range(2000):
        v = random.randint(2, 40)
        q = torch.softmax(torch.randn(v) * 2, dim=0)
        pb = torch.softmax(torch.randn(v) * 2, dim=0)
        pa = torch.softmax(torch.randn(v) * 2, dim=0)
        assert ce(q, pa) - ce(q, pb) == pytest.approx(
            kl(q, pa) - kl(q, pb), abs=1e-4
        )
        assert kl_insight_gain(q, pb, pa) == pytest.approx(
            kl(q, pb) - kl(q, pa), abs=1e-4
        )


def test_tldr_drift_null() -> None:
    """``B x = 0`` forces ``(A B) x = 0`` -- EXACTLY, not an epsilon."""
    torch.manual_seed(16)
    m, n, r = 8, 16, 4
    b = torch.randn(m, n, dtype=torch.float64)
    a = torch.randn(r, m, dtype=torch.float64)
    assert drift_null_ok(b, torch.zeros(n, dtype=torch.float64), a)
    # A genuine kernel vector of B also gives zero composite drift.
    _, _, vh = torch.linalg.svd(b)
    assert drift_null_ok(b, vh[-1], a)
    # A generic input does not.
    assert not drift_null_ok(b, torch.randn(n, dtype=torch.float64), a)


def test_cycle_bracket_sign_decides() -> None:
    """The sign alone decides keep / reject (``experience_cycle_bound``)."""
    assert cycle_bracket(0.5, 0.3, 1.0, 0.1, 0.2, 0.1) > 0.0
    assert cycle_bracket(0.0, 0.0, 1.0, 0.1, 0.9, 0.5) < 0.0
    # The joint-step term is exactly eta*||d*||^2/2.
    assert cycle_bracket(0.0, 0.0, 4.0, 0.5, 0.0, 0.0) == pytest.approx(1.0)
