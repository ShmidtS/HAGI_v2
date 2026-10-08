"""Tests for the growth law.

Each test is the executable form of a Lean statement; the theorem is
named so a failure points at the broken proposition.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from hagi.train.growth_law import (
    capability_floor,
    gain_transfer,
    required_successes,
    risk_floor,
    takeoff_horizon,
)



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


