"""R131: SafeQP ↔ GPM мост — тесты (план §4 Phase A).

Ядро: точная ортогональность шага к row-space активаций старой задачи
⟹ первый порядок забывания = 0. Плюс SPACE-LoRA усиление (нулевой
residual response при любом output-факторе), Davis-Kahan легитимация
ε-бюджета и мост к T2 (ker Σ ≡ GPM-подпространство).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.merge_price import ker_violation, sigma_gram  # noqa: E402
from hagi.train.safeqp_gpm import (  # noqa: E402
    activation_subspace,
    davis_kahan_bound,
    empirical_eps_budget,
    first_order_forgetting,
    gpm_step,
    space_lora_factor,
)

D, RANK = 32, 8


def _old_task(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Активации старой задачи ранга RANK + residuals."""
    torch.manual_seed(seed)
    basis = torch.randn(D, RANK, dtype=torch.float64)
    coefs = torch.randn(200, RANK, dtype=torch.float64)
    x = coefs @ basis.T  # [n, D], rank RANK
    residuals = torch.randn(200, dtype=torch.float64)
    return x, residuals


class TestActivationSubspace:
    def test_basis_orthonormal(self):
        x, _ = _old_task()
        u = activation_subspace(x)
        gram = u.T @ u
        assert torch.allclose(gram, torch.eye(RANK, dtype=gram.dtype), atol=1e-10)

    def test_rank_detected_from_singular_values(self):
        x, _ = _old_task()
        u = activation_subspace(x)
        assert u.shape == (D, RANK)

    def test_full_rank_gives_full_basis(self):
        torch.manual_seed(1)
        x = torch.randn(100, D, dtype=torch.float64)
        assert activation_subspace(x).shape == (D, D)

    def test_activations_live_in_own_rowspace(self):
        # X·U = X: строки активаций лежат в своём row-space
        x, _ = _old_task()
        u = activation_subspace(x)
        assert torch.allclose(x @ u @ u.T, x, atol=1e-8)


class TestGPMStep:
    def test_first_order_forgetting_exactly_zero(self):
        # ЯДРО ТЕОРЕМЫ: d ⊥ row-space(X_old) ⟹ ⟨∇L_old, d⟩ = 0 точно
        x, r = _old_task(2)
        torch.manual_seed(3)
        d = torch.randn(D, dtype=torch.float64)
        u = activation_subspace(x)
        d_gpm = gpm_step(d, u)
        assert abs(first_order_forgetting(d_gpm, x, r)) < 1e-10

    def test_generic_step_has_nonzero_forgetting(self):
        # без проекции первый порядок забывания generically ненулевой
        x, r = _old_task(4)
        torch.manual_seed(5)
        d = torch.randn(D, dtype=torch.float64)
        assert abs(first_order_forgetting(d, x, r)) > 1e-6

    def test_projection_removes_exactly_subspace_component(self):
        # ‖d − d_gpm‖ = ‖Uᵀ d‖: удалена ровно компонента в подпространстве
        x, _ = _old_task(6)
        torch.manual_seed(7)
        d = torch.randn(D, dtype=torch.float64)
        u = activation_subspace(x)
        d_gpm = gpm_step(d, u)
        assert torch.allclose((d - d_gpm).norm(), (u.T @ d).norm(), atol=1e-10)

    def test_orthogonal_step_unchanged(self):
        x, _ = _old_task(8)
        u = activation_subspace(x)
        torch.manual_seed(9)
        d_free = gpm_step(torch.randn(D, dtype=torch.float64), u)
        assert torch.allclose(gpm_step(d_free, u), d_free, atol=1e-12)

    def test_shape_mismatch_raises(self):
        x, _ = _old_task()
        u = activation_subspace(x)
        with pytest.raises(ValueError):
            gpm_step(torch.randn(D + 1, dtype=torch.float64), u)


class TestSpaceLoraFactor:
    def test_residual_response_zero_for_any_output_factor(self):
        # SPACE-LoRA: X_old·Aᵀ = 0 ⟹ X_old·ΔWᵀ = 0 при ЛЮБОМ B
        x, _ = _old_task(10)
        u = activation_subspace(x)
        torch.manual_seed(11)
        a = torch.randn(4, D, dtype=torch.float64)
        a_proj = space_lora_factor(a, u)
        b = torch.randn(200, 4, dtype=torch.float64)  # любой output-фактор
        dw = b @ a_proj  # ΔW как B·Aᵀ в [n, D] интерпретации
        assert float(dw.norm()) > 0.0  # обновление нетривиально
        resp = x @ a_proj.T  # residual response по входу
        assert float(resp.abs().max()) < 1e-10

    def test_factor_norm_preserved_outside_subspace(self):
        # проецированный фактор сохраняет ортогональную компоненту
        x, _ = _old_task(12)
        u = activation_subspace(x)
        torch.manual_seed(13)
        a = torch.randn(4, D, dtype=torch.float64)
        a_proj = space_lora_factor(a, u)
        # весь фактор теперь в ортогональном дополнении
        assert torch.allclose(a_proj @ u, torch.zeros(4, RANK, dtype=torch.float64), atol=1e-10)


class TestDavisKahan:
    def test_zero_gap_infinite_bound(self):
        assert davis_kahan_bound(0.0, 0.1) == float("inf")

    def test_bound_capped_at_one(self):
        assert davis_kahan_bound(0.01, 10.0) == 1.0

    def test_monotone_in_perturbation(self):
        assert davis_kahan_bound(1.0, 0.5) < davis_kahan_bound(1.0, 0.9)

    def test_exact_gram_zero_error(self):
        assert davis_kahan_bound(1.0, 0.0) == 0.0

    def test_negative_perturbation_raises(self):
        with pytest.raises(ValueError):
            davis_kahan_bound(1.0, -0.1)


class TestEpsBudget:
    def test_exact_basis_zero_budget(self):
        assert empirical_eps_budget(1.0, 1.0, 0.0) == 0.0

    def test_budget_scales_with_norms(self):
        small = empirical_eps_budget(0.1, 1.0, 0.5)
        large = empirical_eps_budget(2.0, 1.0, 0.5)
        assert small < large

    def test_sin_theta_capped(self):
        assert empirical_eps_budget(1.0, 1.0, 5.0) == pytest.approx(1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            empirical_eps_budget(1.0, 1.0, -0.1)


class TestBridgeToT2:
    def test_gpm_step_lies_in_kernel_of_sigma(self):
        # Мост к T2 (2607.09202): ker Σ_t ≡ GPM-подпространство.
        # Шаг в ортогональном дополнении row-space ⟹ нулевая цена
        # слияния: ker_violation(d_gpm, Σ) = 0.
        x, _ = _old_task(14)
        sigma = sigma_gram(x)
        torch.manual_seed(15)
        d = torch.randn(D, dtype=torch.float64)
        u = activation_subspace(x)
        d_gpm = gpm_step(d, u)
        assert ker_violation(d_gpm, sigma) < 1e-10

    def test_generic_step_violates_kernel(self):
        # generic шаг имеет компоненту вне ker Σ — цена слияния > 0
        x, _ = _old_task(16)
        sigma = sigma_gram(x)
        torch.manual_seed(17)
        d = torch.randn(D, dtype=torch.float64)
        assert ker_violation(d, sigma) > 0.1
