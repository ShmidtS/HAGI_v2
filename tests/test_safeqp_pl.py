"""T4: SafeQP-PL rate — тесты (план §6, 2606.29521).

Каждый тест пришпиливает звено вывода к измеряемой величине:
kappa_t = ‖d*‖²/‖g‖², rate = 1 − μ·κ/L, прогресс ≥ κ·‖g‖²/(2L),
√(K−1)-закон.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.safeqp_pl import (  # noqa: E402
    conflict_scaling_bound,
    kappa_projection,
    pl_progress,
    pl_rate,
    pl_residual,
    safeqp_pl_step,
)


class TestKappaProjection:
    def test_free_step_kappa_one(self):
        g = torch.tensor([1.0, 2.0])
        assert kappa_projection(g.clone(), g) == pytest.approx(1.0)

    def test_full_cutoff_kappa_zero(self):
        g = torch.tensor([1.0, 1.0])
        d = torch.zeros(2)
        assert kappa_projection(d, g) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            kappa_projection(torch.ones(3), torch.ones(4))

    def test_zero_gradient_raises(self):
        with pytest.raises(ValueError):
            kappa_projection(torch.ones(2), torch.zeros(2))


class TestPLRate:
    def test_free_step_classical_rate(self):
        # kappa=1: классический PL-rate 1 − mu/L
        assert pl_rate(1.0, 0.5, 1.0) == pytest.approx(0.5)

    def test_kappa_zero_stalls(self):
        # kappa=0: фактор = 1, сжатия нет
        assert pl_rate(0.0, 0.5, 1.0) == pytest.approx(1.0)

    def test_mu_zero_gives_one(self):
        assert pl_rate(0.5, 0.0, 1.0) == pytest.approx(1.0)

    def test_higher_kappa_faster(self):
        assert pl_rate(0.5, 0.5, 1.0) < pl_rate(0.1, 0.5, 1.0)

    def test_kappa_above_one_raises(self):
        with pytest.raises(ValueError):
            pl_rate(1.5, 0.5, 1.0)


class TestPLProgress:
    def test_progress_formula(self):
        assert pl_progress(4.0, 1.0) == pytest.approx(2.0)

    def test_zero_gradient_zero_progress(self):
        assert pl_progress(0.0, 1.0) == pytest.approx(0.0)

    def test_negative_g_sq_raises(self):
        with pytest.raises(ValueError):
            pl_progress(-1.0, 1.0)

    def test_zero_lipschitz_raises(self):
        with pytest.raises(ValueError):
            pl_progress(4.0, 0.0)


class TestPLResidual:
    def test_pl_satisfied_residual_negative(self):
        # ‖g‖² >= 2*mu*(E−E*): PL выполнен, residual <= 0
        assert pl_residual(10.0, 8.0, 4.0, 1.0) <= 0.0

    def test_pl_violated_residual_positive(self):
        # ‖g‖² = 4 < 2μ(E−E*) = 2·1·8: PL нарушен, residual > 0
        assert pl_residual(10.0, 2.0, 4.0, 1.0) > 0.0

    def test_mu_zero_raises(self):
        with pytest.raises(ValueError):
            pl_residual(1.0, 0.0, 1.0, 0.0)


def _quadratic(seed: int, d: int):
    torch.manual_seed(seed)
    m = torch.randn(d, d, dtype=torch.float64)
    gram = m @ m.T + 0.5 * torch.eye(d, dtype=torch.float64)
    w0 = torch.randn(d, dtype=torch.float64)
    b = torch.randn(d, dtype=torch.float64)
    return gram, w0, b


class TestSafeQPPLStep:
    def test_free_quadratic_contraction_measured(self):
        # сильно выпуклая квадратика без ограничений: измеренный
        # residual_next <= factor * residual_prev (теорема на числах)
        gram, w0, b = _quadratic(0, 6)
        r = safeqp_pl_step(w0, gram, b)
        assert r["residual"] <= r["factor"] * r["residual_prev"] + 1e-9

    def test_box_constraint_progress_bound(self):
        # бокс: проекция срезает часть градиента, kappa <= 1, но шаг
        # всё равно даёт прогресс >= kappa*|g|^2/(2L)
        gram, w0, b = _quadratic(1, 4)
        w0 = w0.abs() * 0.001 + 0.001  # близко к границе 0
        lower = torch.zeros(4, dtype=torch.float64)
        r = safeqp_pl_step(w0, gram, b, lower=lower)
        assert r["kappa"] <= 1.0
        assert r["progress"] >= r["progress_bound"] - 1e-9

    def test_singular_gram_raises(self):
        gram = torch.zeros(2, 2, dtype=torch.float64)
        with pytest.raises(ValueError):
            safeqp_pl_step(
                torch.ones(2, dtype=torch.float64),
                gram,
                torch.zeros(2, dtype=torch.float64),
            )


class TestConflictScaling:
    def test_single_domain_full_kappa(self):
        assert conflict_scaling_bound(1) == pytest.approx(1.0)

    def test_sqrt_law(self):
        # K=2: 1/1 = 1; K=5: 1/2; K=10: 1/3
        assert conflict_scaling_bound(2) == pytest.approx(1.0)
        assert conflict_scaling_bound(5) == pytest.approx(0.5)
        assert conflict_scaling_bound(10) == pytest.approx(1.0 / 3.0)

    def test_monotone_decreasing(self):
        vals = [conflict_scaling_bound(k) for k in range(2, 20)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            conflict_scaling_bound(0)
