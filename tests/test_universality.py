"""T6: Universality + LongHorizonSafety — тесты.

risk_i(θ_T) ≤ min_k risk_i(leaf_k) + Σ_t ε_t: телескоп MasterHAGI,
универсальность через лучший лист, суммируемость бюджета.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.universality import (  # noqa: E402
    risk_horizon_bound,
    safeqp_budget_sum,
    sum_vs_scale,
    universality_floor,
)


class TestRiskHorizonBound:
    def test_empty_horizon_identity(self):
        assert risk_horizon_bound(3.0, []) == pytest.approx(3.0)

    def test_telescope(self):
        assert risk_horizon_bound(1.0, [0.1, 0.2, 0.3]) == pytest.approx(1.6)

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError):
            risk_horizon_bound(1.0, [-0.1])


class TestUniversalityFloor:
    def test_best_leaf_picked(self):
        risks = {"a": 2.1, "b": 1.7, "c": 2.9}
        val, name = universality_floor(risks)
        assert val == pytest.approx(1.7)
        assert name == "b"

    def test_single_leaf(self):
        assert universality_floor({"only": 0.5}) == (0.5, "only")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            universality_floor({})


class TestSafeQPBudgetSum:
    def test_zero_horizon(self):
        assert safeqp_budget_sum(0.1, 0.5, 0) == pytest.approx(0.0)

    def test_rho_one_linear(self):
        assert safeqp_budget_sum(0.1, 1.0, 10) == pytest.approx(1.0)

    def test_geometric(self):
        # Σ_{t<3} 0.1·0.5^t = 0.1 + 0.05 + 0.025
        assert safeqp_budget_sum(0.1, 0.5, 3) == pytest.approx(0.175)

    def test_summable_converges(self):
        # T→∞: ε₀/(1−ρ) = 0.2
        big = safeqp_budget_sum(0.1, 0.5, 1000)
        assert big == pytest.approx(0.2, abs=1e-9)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            safeqp_budget_sum(-0.1, 0.5, 3)


class TestSumVsScale:
    def test_summable_finite(self):
        assert sum_vs_scale(0.1, 0.5) == pytest.approx(0.2)

    def test_rho_one_infinite(self):
        assert sum_vs_scale(0.1, 1.0) == math.inf

    def test_rho_above_one_infinite(self):
        assert sum_vs_scale(0.1, 1.2) == math.inf

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            sum_vs_scale(-0.1, 0.5)


class TestT6Form:
    def test_full_chain(self):
        # risk_i(θ_T) ≤ min_k risk_i(leaf_k) + Σ ε_t на числах
        leaves = {"gen5": 2.2, "gen6": 1.9}
        floor, _ = universality_floor(leaves)
        budgets = [0.1 * 0.5**t for t in range(5)]
        bound = risk_horizon_bound(floor, budgets)
        # θ_T стартует от ансамбля и обязан оставаться ниже floor+Σε
        assert bound == pytest.approx(1.9 + 0.19375)

    def test_summable_budget_beats_linear(self):
        # суммируемый ε₀ρ^t дешевле постоянного ε₀
        geo = safeqp_budget_sum(0.1, 0.5, 50)
        lin = 0.1 * 50
        assert geo < lin
