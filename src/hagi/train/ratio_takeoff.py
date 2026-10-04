"""R123/R124: RatioTakeoff — гейт зажигания роста вместо хрупкого h_C_cap.

Порт ``Hagi/Growth/RatioTakeoff.lean`` и ``FrontierScaling.lean``
(takeoff_edge_forced). Старый конус-метод с верхней границей
C' <= (1+alpha)*C зажимает систему ровно на границу конуса
(G = alpha*C, D = (alpha/gamma)*C тождественно) — сертификат
хрупкий, не использовать. Правильная постановка — динамика
ОТНОШЕНИЯ r = D/C при

    C' = C + gamma*D,   D' >= rho*D + beta*C - xi.

Конус {r >= k} инвариантен при двух односторонних условиях:

* порог зажигания: beta >= gamma*k^2 + (1-rho)*k + xi/C;
* скорость декея: rho >= gamma*k.

Ключевая алгебра (cone_ratio_step): при точной динамике

    D' - k*C' = (rho - gamma*k)*(D - k*C)
                + (beta - gamma*k^2 - (1-rho)*k)*C - xi,

поэтому из конуса в t, порога зажигания и rho >= gamma*k следует
конус в t+1 — одностороннее неравенство, ничего не форсирующее.

При инвариантном конусе ratio_takeoff даёт односторонний рост
C_T >= C0*(1+gamma*k)^T БЕЗ верхней границы; при beta = 0 и
rho < 1 frontier геометрически затухает (frontier_decay_no_growth):
D_T <= rho^T*D0. Это формализует бифуркацию «зажигание/коллапс»
с измеряемым порогом; после провала цикла конус перепроверяется
пошагово по измерениям — не мгновенный отказ (ALGORITHMS.md §10).

**Конечный горизонт (§7.3 заморозки, честная форма):** конус
инвариантен при соблюдении условий на КАЖДОМ шаге — бессрочных
гарантий нет. Явный горизонт: зажигание поддерживается, пока ξ_t
(«компенсация» свежими данными) не исчерпает источник — при
суммируемом ξ-профиле T* ≈ ln(C*/C₀)/ln(1+γk) шагов до насыщения
ёмкости C* (R125 Saturation). После T* утверждать рост нельзя:
условие зажигания включает ξ_t/C, а C растёт — порог становится
недостижимым. Утверждение «(1+γk)^T-рост» действительно до T*,
не при всех T.
"""

from __future__ import annotations

from enum import Enum

# Допуск для float-сравнений на границе конуса (Lean-арифметика
# точна; runtime-измерения — нет).
_TOL = 1e-9

__all__ = [
    "IgnitionVerdict",
    "IgnitionParams",
    "ignition_threshold",
    "decay_ok",
    "ignition_margin",
    "cone_holds",
    "cone_step_from_dynamics",
    "cone_ratio_invariant",
    "ratio_takeoff_floor",
    "frontier_decay_bound",
    "bifurcation_verdict",
    "step_ignition_slack",
]


class IgnitionVerdict(Enum):
    """Трёхчастный вердикт бифуркации R124 (ALGORITHMS.md §10)."""

    GROW = "grow"
    DECAY = "decay"
    INJECT = "inject"


def ignition_threshold(gamma: float, rho: float, k: float) -> float:
    """Порог зажигания beta_ign = gamma*k^2 + (1-rho)*k.

    Минимальная скорость производства frontier, при которой конус
    r >= k инвариантен (xi = 0 форма; трение добавляется отдельно
    через step_ignition_slack / bifurcation_verdict).
    """
    return gamma * k * k + (1.0 - rho) * k


def decay_ok(rho: float, gamma: float, k: float) -> bool:
    """Условие скорости декея: rho >= gamma*k.

    Задержанный frontier не сгорает быстрее, чем растёт конус.
    """
    return rho >= gamma * k


def ignition_margin(beta: float, gamma: float, rho: float, k: float) -> float:
    """Запас зажигания beta - beta_ign; >= 0 ⟺ зажигание."""
    return beta - ignition_threshold(gamma, rho, k)


def cone_holds(C: float, D: float, k: float) -> bool:
    """Прямая проверка конуса на ИЗМЕРЕННОМ состоянии: D >= k*C.

    Это runtime-форма определения ConeRatio — конус перепроверяется
    по телеметрии на каждом шаге, а не доказывается раз и навсегда.
    """
    return D - k * C >= -_TOL * max(1.0, abs(D), abs(k * C))


def cone_step_from_dynamics(
    C: float,
    D: float,
    gamma: float,
    rho: float,
    beta: float,
    k: float,
    xi: float = 0.0,
) -> bool:
    """Шаг конуса из подогнанной динамики (cone_ratio_step).

    Предсказывает следующий状态 по D' = rho*D + beta*C - xi,
    C' = C + gamma*D и проверяет конус в нём. Это точная runtime-
    форма Lean-теоремы: если измеренный (C, D) лежит в конусе,
    выполнены порог зажигания (с xi/C-компенсацией) и
    rho >= gamma*k, шаг сохраняет конус.
    """
    C_next = C + gamma * D
    D_next = rho * D + beta * C - xi
    return cone_holds(C_next, D_next, k)


def cone_ratio_invariant(
    traj_C: list[float],
    traj_D: list[float],
    k: float,
) -> bool:
    """Инвариант конуса по всей измеренной траектории
    (cone_ratio_invariant): D_t >= k*C_t для всех t.
    """
    return all(cone_holds(c, d, k) for c, d in zip(traj_C, traj_D))


def ratio_takeoff_floor(C0: float, gamma: float, k: float, T: int) -> float:
    """Односторонний сертификат роста C_T >= C0*(1+gamma*k)^T
    (ratio_takeoff). Верхней границы НЕТ: система может расти
    быстрее сертификата.
    """
    if T < 0:
        raise ValueError("T must be >= 0")
    return C0 * (1.0 + gamma * k) ** T


def frontier_decay_bound(D0: float, rho: float, T: int) -> float:
    """Затухающая ветвь (frontier_decay_no_growth): при beta = 0
    и 0 <= rho < 1 frontier D_T <= rho^T*D0 — без производства
    нет роста; НЕ циклить, ждать инъекцию данных.
    """
    if rho < 0.0:
        raise ValueError("rho must be >= 0")
    return rho**T * D0


def bifurcation_verdict(
    beta: float,
    rho: float,
    gamma: float,
    k: float,
    C: float | None = None,
    xi: float = 0.0,
) -> IgnitionVerdict:
    """Вердикт бифуркации для следующего цикла (ALGORITHMS.md §10).

    GROW: порог зажигания достигнут (с xi/C-компенсацией при
    известных C и xi) И rho >= gamma*k — продолжать рост,
    сертификат C_T >= C0*(1+gamma*k)^T.

    DECAY: производства нет (beta = 0) и rho < 1 — frontier
    затухает геометрически; НЕ циклить, ждать инъекцию данных.

    INJECT: beta между 0 и порогом — производство есть, но не
    покрывает расширение конуса; нарастить свежесть/разнообразие
    данных (ось D), а не гонять цикл.
    """
    threshold = ignition_threshold(gamma, rho, k)
    if C is not None and C > 0.0:
        threshold += xi / C
    if beta <= 0.0:
        return IgnitionVerdict.DECAY if rho < 1.0 else IgnitionVerdict.INJECT
    if beta >= threshold and decay_ok(rho, gamma, k):
        return IgnitionVerdict.GROW
    return IgnitionVerdict.INJECT


def step_ignition_slack(
    C: float,
    D: float,
    gamma: float,
    rho: float,
    beta: float,
    k: float,
    xi: float = 0.0,
) -> float:
    """Запас конкретного измеренного шага перед порогом зажигания.

    slack = (beta - beta_ign - xi/C)*C: сколько производства остаётся
    после компенсации расширения конуса и трения. Отрицательный
    slack — ранний сигнал «инъекция, а не цикл» (стоп-условие
    контура R122: (1-rho)*D + delta_O > beta*C - xi).
    """
    if C <= 0.0:
        raise ValueError("C must be > 0")
    return beta * C - ignition_threshold(gamma, rho, k) * C - xi
