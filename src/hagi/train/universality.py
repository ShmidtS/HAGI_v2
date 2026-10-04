"""T6: Universality + LongHorizonSafety — риск на горизонте T.

Форма плана: ``risk_i(θ_T) ≤ min_k risk_i(leaf_k) + Σ_t ε_t`` —
универсальность (не хуже лучшего листа) + долгогоризонтная
безопасность (накопленный бюджет Σε_t). MasterHAGI (R111) даёт
рисковую компоненту: ``risk_T − risk_0 ≤ Σ epsReg_t``; эта T6
соединяет её с шагом-0 (θ_0 = ансамбль, R146/T1) и минимизацией
по листьям. Runtime-порт: каждая компонента измерима.

Компоненты:

* :func:`risk_horizon_bound` — телескоп MasterHAGI risk_bound;
* :func:`universality_floor` — min_k risk_i(leaf_k): стартовый
  уровень θ_0 = ансамбль (T1 step-0: merged CE = mean leaf CE ≤
  max, а Jensen-gap из GapLaw держит ансамбль ниже mean);
* :func:`safeqp_budget_sum` — Σ_t ε_t: суммируемость требует
  ε_t = ε₀ρ^t (план §7.3: конус R107/R124 — конечный горизонт);
* :func:`sum_vs_scale` — Σ ρ^t = ε₀/(1−ρ): суммируемый бюджет
  конечен при ρ < 1 — время решает, не бесконечный рост.
"""
from __future__ import annotations

import math

__all__ = [
    "risk_horizon_bound",
    "universality_floor",
    "safeqp_budget_sum",
    "sum_vs_scale",
]


def risk_horizon_bound(
    risk_0: float, eps_reg: list[float]
) -> float:
    """``risk_T ≤ risk_0 + Σ_t eps_t`` — телескоп MasterHAGI.

    Каждая итерация тратит не более ``eps_reg[t]`` защищённого
    риска (SafeQP per-domain budget); телескопирование по T.
    """
    if any(e < 0.0 for e in eps_reg):
        raise ValueError("per-cycle budgets must be non-negative")
    return risk_0 + sum(eps_reg)


def universality_floor(
    leaf_risks: dict[str, float],
) -> tuple[float, str]:
    """``min_k risk_i(leaf_k)`` — лучший лист на домене i.

    θ_0 (ансамбль/merged) имеет риск ≤ mean leaf risk (Jensen-gap:
    смесь разногласящих экспертов сглаживает, R101 GapLaw), но
    НЕ автоматически ≤ min. T6-цель: за T шагов опуститься до
    min_k + Σε_t — универсальность как «не хуже лучшего листа,
    плюс накопленный бюджет».
    """
    if not leaf_risks:
        raise ValueError("leaf_risks must be non-empty")
    best = min(leaf_risks, key=leaf_risks.get)
    return leaf_risks[best], best


def safeqp_budget_sum(eps0: float, rho: float, horizon: int) -> float:
    """``Σ_{t<T} ε₀ρ^t`` — геометрическая сумма бюджета.

    План §7.3: конус R107/R124 помечен конечным горизонтом T*;
    при ρ < 1 бюджет суммируем: Σ_{t<∞} = ε₀/(1−ρ) — рост не
    взрывает долг, каждый шаг дешевле предыдущего.
    """
    if eps0 < 0.0 or rho < 0.0 or horizon < 0:
        raise ValueError("eps0, rho >= 0 and horizon >= 0 expected")
    if rho == 1.0:
        return eps0 * horizon
    return eps0 * (1.0 - rho ** horizon) / (1.0 - rho)


def sum_vs_scale(eps0: float, rho: float) -> float:
    """``ε₀/(1−ρ)`` — предел бюджета при T→∞ (суммируемость).

    ρ < 1: конечен; ρ ≥ 1: бесконечен — «универсальность» плана
    §7.3 НЕ завышается: без суммируемости горизонт T* обязателен.
    """
    if rho < 0.0 or eps0 < 0.0:
        raise ValueError("eps0, rho >= 0 expected")
    if rho >= 1.0:
        return math.inf
    return eps0 / (1.0 - rho)
