"""R131: SafeQP ↔ GPM мост — no-forgetting через геометрию активаций.

План §4 Phase A (T-программа §6 п.2): связать SafeQP-проекцию шага с
GPM-подпространством активаций старых задач.

**Ядро теоремы (GPM 2103.09762 + OWM null-space):** если шаг d точно
ортогонален подпространству активаций старой задачи (span строк
матрицы активаций X_old), то в квадратичной модели лосса первый
порядок изменения лосса старой задачи нулевой:

    ΔL_old = ⟨∇L_old, d⟩ = ⟨X_oldᵀ(X_old w_old − y_old)/n, d⟩

∇L_old лежит в span(X_oldᵀ·) = row-space(X_old); d ⊥ row-space(X_old)
⟹ ⟨∇L_old, d⟩ = 0 точно. Это порт GPM в Lean-термины SafeQP: множество
C = {d : ⟨g_i, d⟩ ≥ −ε_i} при ε_i = 0 содержит точное ортогональное
дополнение, когда g_i ∈ row-space(X_i) — активационная геометрия
легитимирует нулевой бюджет.

**SPACE-LoRA усиление (2609.34453):** проекция INPUT-фактора LoRA на
ортогональное дополнение старых активаций даёт тождественно нулевой
residual response при ЛЮБОМ output-факторе — сильнее условия
«ортогонален шаг»: целый класс обновлений безопасен по построению.

**Davis-Kahan предостережение (2607.05872):** без spectral gap базис
подпространства принципиально шумный — оценка ортогональности по
эмпирическому базису имеет ошибку O(‖ΔΣ‖/gap). Поэтому ε-бюджет
SafeQP НЕ слабость, а легитимная форма: точная проекция на
эмпирический базис ≠ точная проекция на истинное подпространство.

**Мост к T2 (2607.09202):** ker Σ_t ≡ GPM-подпространство:
Σ-ортогонализация девиации (условие нулевой цены слияния) — та же
активационная ортогональность. merge_price/ker_violation — численная
сторона той же геометрии.

Runtime-формы:

* :func:`activation_subspace` — базис row-space активаций (SVD);
* :func:`gpm_step` — проекция шага на ортогональное дополнение
  (точный GPM-режим SafeQP);
* :func:`first_order_forgetting` — ⟨∇L_old, d⟩: 0 точно при GPM-шаге;
* :func:`space_lora_factor` — input-фактор в ортогональном
  дополнении: residual response = 0 при любом output-факторе;
* :func:`davis_kahan_bound` — ошибка подпространства по spectral gap;
* :func:`empirical_eps_budget` — ε_i из шума базиса (Davis-Kahan ⟹
  почему ε-бюджет обязателен на практике).
"""
from __future__ import annotations

import torch

__all__ = [
    "activation_subspace",
    "gpm_step",
    "first_order_forgetting",
    "space_lora_factor",
    "davis_kahan_bound",
    "empirical_eps_budget",
]


def activation_subspace(
    activations: torch.Tensor, rank: int | None = None
) -> torch.Tensor:
    """Ортонормальный базис row-space активаций [n, d] → [d, r].

    GPM-память задачи: span(Xᵀ·) — подпространство, в котором живёт
    градиент квадратичного лосса по последнему слою. r = rank или
    полный численный ранг (σ > tol).
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be [n, d], got {tuple(activations.shape)}")
    x = activations.to(torch.float64)
    # row-space активаций = column-space Xᵀ: левые сингулярные векторы
    # Xᵀ = right сингулярные векторы X. Численный ранг — по сингулярным
    # числам (строки vh всегда ортонормальны, даже при нулевых σ)
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    tol = 1e-10 * max(1.0, float(s.max()))
    live = int((s > tol).sum())
    r = min(rank, live) if rank is not None else live
    return vh[:r].T.contiguous()  # [d, r] ортонормальные колонки


def gpm_step(step: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Шаг в ортогональном дополнении GPM-подпространства.

    d' = (I − U Uᵀ) d: удаляет компоненту шага в row-space старых
    активаций. Первый порядок изменения старого лосса — ровно 0.
    """
    if basis.ndim != 2 or step.ndim != 1:
        raise ValueError("basis [d, r], step [d] expected")
    if basis.shape[0] != step.shape[0]:
        raise ValueError("basis and step dimension mismatch")
    d = step.to(torch.float64)
    u = basis.to(torch.float64)
    return d - u @ (u.T @ d)


def first_order_forgetting(
    step: torch.Tensor,
    activations: torch.Tensor,
    residuals: torch.Tensor,
) -> float:
    """⟨∇L_old, d⟩ — первый порядок изменения лосса старой задачи.

    Квадратичная модель: L(w) = ‖Xw − y‖²/(2n), ∇ = Xᵀ(Xw−y)/n.
    GPM-шаг даёт 0 точно (∇ ∈ span(Xᵀ·), d ⊥ ему).
    """
    if activations.shape[0] != residuals.shape[0]:
        raise ValueError("activations and residuals row count mismatch")
    x = activations.to(torch.float64)
    r = residuals.to(torch.float64).reshape(-1)
    d = step.to(torch.float64)
    grad = x.T @ r / x.shape[0]
    return float(grad @ d)


def space_lora_factor(
    input_factor: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    """SPACE-LoRA: input-фактор A в ортогональном дополнении старых
    активаций — ΔW = B·Aᵀ имеет нулевой residual response на старых
    активациях при ЛЮБОМ output-факторе B: X_old·Aᵀ = 0 ⟹ X_old·ΔW =
    X_old·Aᵀ·Bᵀ = 0.
    """
    if input_factor.ndim != 2:
        raise ValueError("input_factor [r, d] expected")
    a = input_factor.to(torch.float64)
    u = basis.to(torch.float64)
    return a - (a @ u) @ u.T


def davis_kahan_bound(
    sigma_gap: float, perturbation_norm: float
) -> float:
    """Ошибка подпространства: sin θ ≤ ‖ΔΣ‖ / gap (Davis-Kahan).

    Если spectral gap между «живыми» и «мёртвыми» направлениями мал,
    эмпирический базис подпространства далёк от истинного — точная
    проекция на НЕГО не гарантирует ортогональность ИСТИННОМУ
    подпространству. Формула легитимирует ε-бюджет SafeQP.
    """
    if sigma_gap <= 0.0:
        return float("inf")
    if perturbation_norm < 0.0:
        raise ValueError("perturbation_norm must be >= 0")
    return min(1.0, perturbation_norm / sigma_gap)


def empirical_eps_budget(
    step_norm: float, grad_norm: float, sin_theta: float
) -> float:
    """ε_i = ‖d‖·‖g_i‖·sin θ — бюджет, покрывающий ошибку базиса.

    Реальный «нулевой» бюджет на эмпирическом базисе — не 0, а
    произведение норм на угол между истинным и эмпирическим
    подпространствами (остаточная интерференция первого порядка).
    """
    if step_norm < 0.0 or grad_norm < 0.0 or sin_theta < 0.0:
        raise ValueError("norms and sin_theta must be >= 0")
    return step_norm * grad_norm * min(sin_theta, 1.0)
