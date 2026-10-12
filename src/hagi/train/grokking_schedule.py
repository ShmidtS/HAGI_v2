"""Grokging-расписание R366: weight decay выведен, а не настроен.

Теорема (primes R366, ``Hagi/Foundations/Recurrence``,
``grokking_threshold_time``; спектральная теория grokking,
arXiv:2609.26679): остаточная динамика

    r(t+1) = r(t) − ηλ·r(t)

достигает порога ε к моменту

    T = ln(r₀/ε)/(η·λ),

при условии фазовой безопасности ηλ ≤ 1 (ηλ > 1 — расходимость).

Правило планировщика (ALGORITHMS.md §7b): для целевого времени
генерализации T* выбирать η·λ ≥ ln(r₀/ε)/T*; при фиксированном η
это ЗАДАЁТ weight decay λ — гиперпараметр выведен, а не подобран.
"""

from __future__ import annotations

import math


def grokking_threshold_time(r0: float, eps: float, eta_lambda: float) -> float:
    """R366: T = ln(r0/eps)/(eta*lambda).

    ``eta_lambda`` — произведение η·λ (шаг × decay за шаг).
    Посылки: r0 > eps > 0, eta_lambda > 0. Время, за которое
    остаток r(t) = r0·(1 − ηλ)^t падает до ε.
    """
    if r0 <= 0.0:
        raise ValueError(f"r0 must be > 0, got {r0}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if eps >= r0:
        raise ValueError(f"eps must be < r0, got eps={eps}, r0={r0}")
    if eta_lambda <= 0.0:
        raise ValueError(f"eta_lambda must be > 0, got {eta_lambda}")
    return math.log(r0 / eps) / eta_lambda


def grokking_lambda(r0: float, eps: float, eta: float, T_star: float) -> float:
    """R366 (обратная форма): λ = ln(r0/eps)/(η·T*).

    Phase-safety: η·λ ≤ 1 (условие теоремы; ηλ > 1 — расходимость).
    Нарушение означает, что T* недостижим при данном η без
    расходимости — поднять T* или η.
    """
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0, got {eta}")
    if T_star <= 0.0:
        raise ValueError(f"T_star must be > 0, got {T_star}")
    lam = math.log(r0 / eps) / (eta * T_star)
    if eta * lam > 1.0:
        raise ValueError(
            f"phase divergence: eta*lambda = {eta * lam:.4f} > 1 "
            f"(theorem condition eta*lambda <= 1). Required rate "
            f"ln(r0/eps)/T* = {eta * lam:.4f}/eta exceeds the stable "
            f"phase at eta = {eta}. Raise T_star or eta."
        )
    return lam


def grokking_weight_decay(
    r0: float, eps: float, eta: float, T_star: float,
    fan_ratio: float = 1.0,
) -> float:
    """R366 → Muon decoupled decay: wd = λ_теоремы / fan_ratio.

    Мост в рантайм: Muon применяет decay как
    ``p.mul_(1 − lr·weight_decay·fan_ratio)`` (optim.py, HybridOptimizer
    Muon-ветка) — эффективная скорость затухания за шаг есть
    η·wd·fan_ratio. Теорема требует η·λ, значит
    wd = λ_theorem / fan_ratio.
    """
    if fan_ratio <= 0.0:
        raise ValueError(f"fan_ratio must be > 0, got {fan_ratio}")
    return grokking_lambda(r0, eps, eta, T_star) / fan_ratio
