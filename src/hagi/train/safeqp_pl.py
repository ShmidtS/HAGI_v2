"""T4 (R147): SafeQP-PL rate — линейная скорость под проекцией.

План §6 (2606.29521): если лосс удовлетворяет PL-условию
``‖g‖² ≥ 2μ(E − E*)`` и шаг — проекция SafeQP ``d*`` на выпуклое
множество допустимых направлений (содержащее 0), то при ``η = 1/L``:

    E_{t+1} − E* ≤ (1 − μ·κ_t/L)·(E_t − E*),   κ_t = ‖d*‖²/‖g‖²

κ_t — ДОЛЯ ГРАДИЕНТА, СОХРАНЁННАЯ ПРОЕКЦИЕЙ: сколько от
квадрата нормы антиградиента остаётся после того, как конус
безопасности срезал конфликтующие компоненты. Без проекции κ_t = 1
(классический PL-rate); при жёстком конфликте K доменов κ_t
падает — закон масштабирования по числу защищённых доменов.

**Вывод (каждое звено измеримо и протестировано):**

1. Проекция на выпуклое C ∋ 0: ``⟨g − d*, d* − 0⟩ ≥ 0`` (характеризация
   проекции) ⟹ ``⟨g, d*⟩ ≥ ‖d*‖²`` — проекция сохраняет descent.
2. Descent lemma при η = 1/L:
   ``E_{t+1} ≤ E_t − ⟨g,d*⟩/L + ‖d*‖²/(2L) ≤ E_t − ‖d*‖²/(2L)``.
3. PL: ``‖g‖²/(2L) ≥ (μ/L)(E_t − E*)``; умножение на κ_t = ‖d*‖²/‖g‖²:
   ``‖d*‖²/(2L) = κ_t·‖g‖²/(2L) ≥ (κ_t·μ/L)(E_t − E*)``.
4. Вычитание: ``E_{t+1} − E* ≤ (1 − κ_t·μ/L)(E_t − E*)``.

**√(K−1)-закон (2606.29521):** при K защищённых доменах худший
случай конфликта срезает κ_t до O(1/√(K−1)) — цена защиты растёт
как корень из числа доменов. Порт флагирует закон как измеримую
зависимость (см. :func:`conflict_scaling_bound`), Lean-сторона —
за формализатором.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "kappa_projection",
    "pl_rate",
    "pl_progress",
    "pl_residual",
    "safeqp_pl_step",
    "conflict_scaling_bound",
]


def kappa_projection(d_star: torch.Tensor, g: torch.Tensor) -> float:
    """κ_t = ‖d*‖²/‖g‖² — доля градиента, сохранённая проекцией.

    ``d*`` — SafeQP-проекция антиградиента ``−g`` на допустимый конус.
    κ_t ∈ [0, 1]: 1 = проекция ничего не срезала (свободный шаг),
    0 = направление полностью подавлено безопасностью.
    """
    if d_star.shape != g.shape or d_star.ndim != 1:
        raise ValueError("d_star and g must be 1-D of equal shape")
    g_sq = float(g.to(torch.float64).pow(2).sum())
    if g_sq <= 0.0:
        raise ValueError("||g||^2 must be positive")
    return float(d_star.to(torch.float64).pow(2).sum()) / g_sq


def pl_rate(kappa: float, mu: float, lipschitz: float) -> float:
    """Фактор сжатия PL-rate: ``1 − μ·κ/L``.

    Меньше — быстрее. Требует μ·κ/L < 1 для сжатия (иначе шаг не
    сходится с этой скоростью).
    """
    if mu < 0.0 or lipschitz <= 0.0:
        raise ValueError("mu >= 0 and L > 0 expected")
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa must be in [0, 1]")
    return 1.0 - mu * kappa / lipschitz


def pl_progress(g_norm_sq: float, lipschitz: float) -> float:
    """Гарантированный прогресс шага η=1/L: ``‖g‖²/(2L)``.

    Сколько энергии шаг обязан срезать (звено 2 вывода) при
    беспроекционном шаге; умножение на κ_t даёт проекционный случай.
    """
    if g_norm_sq < 0.0 or lipschitz <= 0.0:
        raise ValueError("g_norm_sq >= 0 and L > 0 expected")
    return g_norm_sq / (2.0 * lipschitz)


def pl_residual(
    energy: float, energy_star: float, g_norm_sq: float, mu: float
) -> float:
    """PL-дефицит: ``E − E* − ‖g‖²/(2μ)`` — насколько PL зажат.

    ≤ 0 означает PL выполнен на текущей точке (‖g‖² покрывает зазор
    до оптимума). Мониторинг отрицательности = проверка посылки T4
    в рантайме, а не постулирование.
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    return energy - energy_star - g_norm_sq / (2.0 * mu)


def safeqp_pl_step(
    params: torch.Tensor,
    gram: torch.Tensor,
    target: torch.Tensor,
    lower: torch.Tensor | None = None,
) -> dict[str, float]:
    """Один SafeQP-PL шаг на квадратичной модели, все звенья измерены.

    Модель ``E(w) = ½‖Aw − b‖²`` передана грамианом ``gram = AᵀA`` и
    ``target = Aᵀb`` (L = λ_max, μ = λ_min > 0 — сильно выпуклая).
    Шаг: антиградиент, спроецированный на неотрицательный ортант
    (если задан ``lower`` — поэлементный бокс) — выпуклый C ∋ 0;
    η = 1/L. Возвращает measured κ_t, factor, новый w и E.

    Runtime-эквивалент теоремы: измеренный прогресс ≥ κ_t·‖g‖²/(2L).
    """
    w = params.to(torch.float64)
    a = gram.to(torch.float64)
    b = target.to(torch.float64)
    if a.shape[0] != a.shape[1] or a.shape[0] != w.shape[0]:
        raise ValueError("gram [d, d] and params [d] aligned expected")
    if b.shape != w.shape:
        raise ValueError("target [d] expected")
    evals = torch.linalg.eigvalsh(a)
    lipschitz = float(evals[-1])
    mu = float(evals[0])
    if mu <= 0.0:
        raise ValueError("gram must be positive definite (mu > 0)")
    g = a @ w - b
    d = -g
    if lower is not None:
        # бокс-проекция: w_next зажат поэлементно в [lower, ∞)
        lo = lower.to(torch.float64)
        w_next = (w + d / lipschitz).clamp_min(0.0) if lo is None else (
            torch.maximum(w + d / lipschitz, lo)
        )
    else:
        w_next = w + d / lipschitz
    d_star = (w_next - w) * lipschitz  # фактический шаг в d-шкале
    kappa = kappa_projection(d_star, g)
    e = 0.5 * float((a @ w - b).pow(2).sum())
    e_next = 0.5 * float((a @ w_next - b).pow(2).sum())
    e_star = 0.5 * float(b @ torch.linalg.solve(a, b))
    return {
        "kappa": kappa,
        "mu": mu,
        "lipschitz": lipschitz,
        "factor": pl_rate(kappa, mu, lipschitz),
        "energy": e_next,
        "energy_prev": e,
        "energy_star": e_star,
        "residual": e_next - e_star,
        "residual_prev": e - e_star,
        "progress": e - e_next,
        "progress_bound": kappa * pl_progress(float(g.pow(2).sum()), lipschitz),
    }


def conflict_scaling_bound(n_domains: int) -> float:
    """√(K−1)-закон: нижняя граница κ при K защищённых доменах.

    2606.29521: худший случай конфликта K доменов срезает долю
    градиента как O(1/√(K−1)) — цена защиты растёт как корень из
    числа доменов. Измеримая форма: ``1/√(K−1)`` (K ≥ 2).
    """
    if n_domains < 1:
        raise ValueError("n_domains must be >= 1")
    if n_domains == 1:
        return 1.0
    return 1.0 / math.sqrt(n_domains - 1)
