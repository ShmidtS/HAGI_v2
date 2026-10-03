"""R125/R126: Saturation + state-closed рост — полный жизненный цикл.

Порт ``Hagi/Growth/Saturation.lean`` и ``StateClosedRenewal.lean``.
R124 дал экспоненциальный takeoff как НИЖНЮЮ оценку неограниченной
величины; реальные метрики ограничены. R125 замыкает: ёмкость C*
и прогрессивно-сужающееся (PL) окно роста, ДВЕ односторонние
пошаговые формы (одним измерением: gain_t = C_{t+1} - C_t):

* PL-верх ``pl_up``:   C_{t+1} <= C_t + sigma*(C* - C_t)
  (шаг не больше доли sigma остатка — нет перелёта);
* PL-низ ``pl_lo``:    C_t + sigma*(C* - C_t) <= C_{t+1}
  (равносильно сжатию зазора C* - C_{t+1} <= (1-sigma)(C* - C_t)).

Теоремы: ``never_overshoot`` (C_t <= C* всегда), ``pl_gap_geometric``
(зазор сжимается геометрически), ``saturation_limit`` (сходимость
к C*), ``takeoff_with_saturation`` (двусторонняя полоса
C0*(1+gamma*k)^T <= C_T <= C* - (1-sigma)^T*(C* - C0)).

R126 замыкает контур на ПОЛЯХ состояния (без shadow-последовательностей
C G D : Nat -> Real): capability/dataField измеряются на каждом цикле,
``state_closed_band`` — полный контракт: зажигание -> экспонента ->
насыщение -> честный стоп (переключение оси: инъекция данных или
расширение C* новой архитектурой). ξ-компенсация ТОЧНАЯ: трение
входит в полю зажигания аддитивно xi/C (R124-набросок, теперь
формально на состоянии).
"""

from __future__ import annotations

import math
from enum import Enum

__all__ = [
    "LifeCycleVerdict",
    "pl_step_gain",
    "never_overshoot",
    "pl_gap_geometric",
    "pl_gap_geometric_lo",
    "saturation_horizon",
    "growth_band",
    "state_closed_band",
    "lifecycle_verdict",
]


class LifeCycleVerdict(Enum):
    """Вердикт жизненного цикла по измеряемым полям (§13)."""

    IGNITE = "ignite"
    GROW = "grow"
    SATURATE = "saturate"
    SWITCH_AXIS = "switch_axis"


def pl_step_gain(C: float, Cstar: float, sigma: float) -> float:
    """PL-шаг: sigma*(C* - C_t) — доля остатка до ёмкости.

    PL-окно шага: gain_t ∈ [sigma*gap, sigma*gap] в точной форме
    означает gain == sigma*gap; runtime-измерение сравнивается
    с этим значением с допуском.
    """
    if sigma < 0.0 or sigma > 1.0:
        raise ValueError("sigma must be in [0, 1]")
    return sigma * (Cstar - C)


def never_overshoot(
    traj_C: list[float], Cstar: float, sigma: float, tol: float = 1e-9
) -> bool:
    """``never_overshoot`` (R125): при PL-верхней форме capability
    никогда не превосходит ёмкость — проверяется на измеренной
    траектории (C_0 <= C* и каждый шаг <= sigma-доли остатка).
    """
    if traj_C[0] > Cstar + tol:
        return False
    for t in range(len(traj_C) - 1):
        gap = Cstar - traj_C[t]
        if traj_C[t + 1] > traj_C[t] + sigma * gap + tol:
            return False
    return True


def pl_gap_geometric(traj_C: list[float], Cstar: float, sigma: float,
                     tol: float = 1e-9) -> bool:
    """``pl_gap_geometric`` (R125): нижняя PL-форма даёт сжатие
    зазора C* - C_{t+1} <= (1-sigma)*(C* - C_t) — геометрическое
    приближение к ёмкости. Проверяется на измеренной траектории.
    """
    for t in range(len(traj_C) - 1):
        gap = Cstar - traj_C[t]
        if traj_C[t] + sigma * gap - tol > traj_C[t + 1]:
            return False
    return True


def pl_gap_geometric_lo(traj_C: list[float], Cstar: float, sigma: float,
                        tol: float = 1e-9) -> bool:
    """``pl_gap_geometric_lo`` (R125): верхняя PL-форма даёт
    (1-sigma)^t*(C* - C_0) <= C* - C_t — рост не быстрее логистики.
    """
    if traj_C[0] > Cstar + tol:
        return False
    for t in range(len(traj_C) - 1):
        gap = Cstar - traj_C[t]
        if traj_C[t + 1] > traj_C[t] + sigma * gap + tol:
            return False
    return True


def saturation_horizon(gap0: float, sigma: float, epsilon: float) -> int:
    """``saturation_limit`` (R125): конструктивный горизонт
    ε-близости к C*: T с (C*-C_0)/(sigma*T) < epsilon.

    Телескоп σ(1-σ)^i = (1-σ)^i - (1-σ)^{i+1} в Lean даёт
    T*σ*(1-σ)^T <= 1; runtime-форма упрощает до ceil(gap0/(σ·ε)),
    что совпадает по порядку и всегда консервативнее.
    """
    if sigma <= 0.0 or epsilon <= 0.0:
        raise ValueError("sigma and epsilon must be > 0")
    return math.ceil(gap0 / (sigma * epsilon))


def growth_band(C0: float, gamma: float, k: float, Cstar: float,
                sigma: float, T: int) -> tuple[float, float]:
    """``takeoff_with_saturation`` (R125): двусторонняя полоса

        C_0*(1+gamma*k)^T <= C_T <= C* - (1-sigma)^T*(C* - C_0).

    Нижняя граница — ratio-взлёт (гарантия роста), верхняя —
    PL-окно (физический предел). Экспонента взлёта сопоставлена
    с ограниченной метрикой.
    """
    lo = C0 * (1.0 + gamma * k) ** T
    hi = Cstar - (1.0 - sigma) ** T * (Cstar - C0)
    return lo, hi


def state_closed_band(
    capability: list[float],
    data_field: list[float],
    gamma: float,
    rho: float,
    beta: float,
    k: float,
    Cstar: float,
    sigma: float,
    xi: float = 0.0,
    tol: float = 1e-9,
) -> tuple[float, float, bool]:
    """``state_closed_band`` (R126): полный state-closed контракт
    на измеренных полях состояния.

    Проверяет пошаговые посылки на траектории (шаг capability,
    динамика frontier с ξ-компенсацией, конус r >= k, PL-окно)
    и возвращает (lower, upper, all_steps_verified). Проверка
    ПЕРЕД вычислением полосы: если какой-то шаг провалился,
    сертификат полосы не заявляется (all_steps_verified=False),
    хотя сами границы вычисляются для информации.
    """
    from hagi.train.ratio_takeoff import cone_holds

    n = len(capability)
    ok = True
    for t in range(n - 1):
        C, D = capability[t], data_field[t]
        Cn, Dn = capability[t + 1], data_field[t + 1]
        # шаг capability: C' = C + gamma*D (допуск на измерение)
        if abs(Cn - (C + gamma * D)) > tol * max(1.0, abs(Cn)):
            ok = False
        # динамика frontier с ξ: D' >= rho*D + beta*C - xi
        if Dn < rho * D + beta * C - xi - tol:
            ok = False
        # PL-окно
        gap = Cstar - C
        if Cn > C + sigma * gap + tol or Cn < C + sigma * gap - tol:
            ok = False
        if not cone_holds(Cn, Dn, k):
            ok = False
    lo, hi = growth_band(capability[0], gamma, k, Cstar, sigma, n - 1)
    return lo, hi, ok


def lifecycle_verdict(
    beta_hat: float,
    rho_hat: float,
    gamma: float,
    k: float,
    C: float,
    Cstar: float | None = None,
    sigma: float | None = None,
    eps_c: float = 0.05,
    xi: float = 0.0,
) -> LifeCycleVerdict:
    """Полный вердикт жизненного цикла (ALGORITHMS.md §13).

    IGNITE: конус ещё не горит, но полю зажигания достигнут —
    продолжать цикл, сертификат (1+gamma*k)-мультипликатора.

    GROW: полю выполнен И capability далеко от ёмкости
    (C < C* - eps_c при известных C*, sigma) — экспоненциальная
    фаза.

    SATURATE: ε-близость к C* (gap < eps_c) — стоп роста по этой
    оси, вердикт переключения.

    SWITCH_AXIS: полю зажигания не достигнут — инъекция данных
    или расширение C* новой архитектурой, а не цикл.
    """
    from hagi.train.ratio_takeoff import (
        decay_ok,
        ignition_threshold,
    )

    threshold = ignition_threshold(gamma, rho_hat, k)
    if C > 0.0:
        threshold += xi / C
    if beta_hat < threshold or not decay_ok(rho_hat, gamma, k):
        return LifeCycleVerdict.SWITCH_AXIS
    if Cstar is not None and Cstar - C < eps_c:
        return LifeCycleVerdict.SATURATE
    if Cstar is None:
        return LifeCycleVerdict.IGNITE
    return LifeCycleVerdict.GROW
