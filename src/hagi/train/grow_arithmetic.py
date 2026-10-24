"""Арифметика решений роста — порты волн полной перечитки Lean (2026-10-24).

Каждая функция — прямое следствие машинно-верифицированной теоремы
(C:\\primes\\Hagi), с измеримыми посылками в сигнатуре (честный отказ
при нарушении). Потребитель — генератор поколений (выбор рычага
gen-N+1) и merge-конвейер.

  * certified_gain / grow_epsilon_stop   (Growth/Grow.lean):
    добавление листа standalone-CE c в пул N листов со средним M
    улучшает границу среднего CE ровно на (M−c)/(N+1); если у ЛУЧШЕГО
    кандидата выигрыш < ε — сертифицированный СТОП добавления листьев
    (у всех остальных хуже).
  * merge_skip_certificate / two_gap_ceiling
    (Ensemble/Hoeffding.lean twoGap_bounded, merge_skip_certificate):
    диаметр несогласия логитов M ⟹ twoGap ≤ ½log cosh M ≤ M²/4;
    M²/4 < ε ⟹ SKIP merge-цикла (одна скалярная посылка вместо
    полного прогона).
  * disagreement_gain_floor (Ensemble/Hoeffding.lean
    domain_disagreement_floor): одна пара листьев с |d_u−d_v| ≥ δ ⟹
    twoGap ≥ ½·log(1 + p_u·p_v·(cosh δ − 1)) > 0 — гарантированный пол
    выигрыша слияния.
  * disagreement_noise_floor (Growth/NoiseDisentangle.lean
    disagreement_convergence_floor): усреднённое несогласие имеет
    неотделимый пол σ²/N — измеренный twoGap НИЖЕ пола это шум,
    сливать нечего; N (размер ансамбля) — рычаг денойзинга.
  * seed_only_dispersion (Growth/SeedOnly.lean seed_only_disp_bound /
    seed_only_grow_stop): поколения с идентичным init (только
    пере-сеянные адаптеры) имеют дисперсию ≤ Λ²·s·Σa^i —
    геометрически затухающую; цель недостижима seed-only инъекцией,
    нужна свежесть данных/доменов (диагноз замедления лестницы).
  * critical_batch / batch_noise_variance (Budget/BatchCap.lean
    sfo_global_min; Probability/KLSBridge.lean): критический батч
    b* = 9Y²/(4(ε−Z)²) из измеренного шумового члена Y — ВЫВЕДЕННЫЙ
    кап batch-роста вместо хардкода.
"""

from __future__ import annotations

import math


def certified_gain(n_leaves: int, mean_ce: float, candidate_ce: float) -> float:
    """(M−c)/(N+1): точное улучшение границы среднего CE пула.

    Посылки теоремы certifiedGain: пул из N листьев со средним M,
    кандидат со standalone CE c. Выигрыш положителен ⟺ c < M
    (кандидат лучше среднего; selection_hurts запрещает отбирать по
    standalone CE — эта арифметика о ГРАНИЦЕ, не об отборе).
    """
    if n_leaves < 0:
        raise ValueError(f"n_leaves must be >= 0, got {n_leaves}")
    return (mean_ce - candidate_ce) / (n_leaves + 1)


def grow_epsilon_stop(n_leaves: int, mean_ce: float, best_candidate_ce: float, eps: float) -> bool:
    """Сертифицированный СТОП добавления листьев (grow_epsilon_stop).

    Если выигрыш ЛУЧШЕГО кандидата < ε, то у любого кандидата c ≥ c*
    выигрыш тоже < ε (gain монотонно убывает по c) — продолжать
    перебор кандидатов бессмысленно: certified verdict.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    return certified_gain(n_leaves, mean_ce, best_candidate_ce) < eps


def two_gap_ceiling(diameter: float) -> float:
    """twoGap ≤ ½·log cosh M ≤ M²/4 — потолок выигрыша слияния по
    диаметру несогласия логитов M = max|d_u − d_v| (по токену)."""
    if diameter < 0:
        raise ValueError(f"diameter must be >= 0, got {diameter}")
    return 0.5 * math.log(math.cosh(diameter))


def merge_skip_certificate(diameter: float, eps: float, transport_efficiency: float = 1.0) -> bool:
    """SKIP merge-цикла, если η·twoGap ≤ η·M²/4 < ε.

    transport_efficiency η ∈ [0,1] — доля зазора, реально переносимая
    в объединённую модель (измеренный коэффициент конвейера).
    """
    if not 0.0 <= transport_efficiency <= 1.0:
        raise ValueError(f"transport_efficiency must be in [0,1], got {transport_efficiency}")
    return transport_efficiency * min(two_gap_ceiling(diameter), diameter * diameter / 4.0) < eps


def disagreement_gain_floor(p_u: float, p_v: float, delta: float) -> float:
    """Пол выигрыша слияния от одной несогласной пары листьев.

    twoGap ≥ ½·log(1 + p_u·p_v·(cosh δ − 1)); положителен при
    p_u,p_v > 0, δ > 0 — sibling-per-domain гарантированно бьёт
    same-mix siblings (domain_disagreement_floor).
    """
    if not (0.0 < p_u <= 1.0 and 0.0 < p_v <= 1.0):
        raise ValueError(f"weights must be in (0,1], got {p_u}, {p_v}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    return 0.5 * math.log(1.0 + p_u * p_v * (math.cosh(delta) - 1.0))


def disagreement_noise_floor(sigma_sq: float, n_experts: int) -> float:
    """Неотделимый пол σ²/N усреднённого несогласия.

    Измеренный twoGap ниже σ²/N — это шум, не сигнал: цикл слияния
    исчерпан (round_exhausted), рычаг — рост N.
    """
    if sigma_sq < 0:
        raise ValueError(f"sigma_sq must be >= 0, got {sigma_sq}")
    if n_experts < 1:
        raise ValueError(f"n_experts must be >= 1, got {n_experts}")
    return sigma_sq / n_experts


def seed_only_dispersion(disp0: float, rate: float, steps: int, carry: float = 1.0) -> float:
    """Дисперсия seed-only поколений: dispL t ≤ Λ²·s·Σ_{i<t} a^i.

    rate a < 1 ⟹ затухает к cΛ²s/(1−a); цель (gapInf < ε) недостижима
    без внешней инъекции свежести (seed_only_grow_stop) — смена
    data-offsets / новых доменов обязательна, а не опциональна.
    """
    if rate < 0 or steps < 0:
        raise ValueError(f"rate and steps must be >= 0, got {rate}, {steps}")
    if rate == 1.0:
        return disp0 * carry * steps
    return disp0 * carry * (1.0 - rate**steps) / (1.0 - rate)


def batch_noise_variance(half_diff_sq: float, microbatch_tokens: int) -> float:
    """σ̂² градиентного шума из сплит-батч пробы, в пересчёте на токен.

    half_diff_sq = ‖g1 − g2‖²/2 по половинкам одного батча (обе
    несмещены на μ ⟹ E‖g1−g2‖²/2 = Var(ĝ_half)); microbatch_tokens
    — токенов в половинке. Возврат: σ² на токен (инвариант
    LR↔batch: шумовой бюджет η²σ²/m).
    """
    if half_diff_sq < 0 or microbatch_tokens < 1:
        raise ValueError("probe must be >= 0 and tokens >= 1")
    return half_diff_sq / microbatch_tokens


def critical_batch(noise_term: float, target_eps: float, bias_term: float = 0.0) -> float:
    """b* = 9Y²/(4(ε−Z)²) — критический батч (sfo_global_min).

    Y — измеренный шумовой член (из batch_noise_variance),
    ε — целевая ошибка, Z — член смещения (bias). Кап batch-роста
    ВЫВОДИТСЯ, не подбирается: b_eff(B) = min(√(AB/C), b*).
    """
    denom = target_eps - bias_term
    if noise_term < 0 or denom <= 0:
        raise ValueError(
            f"noise_term must be >= 0 and eps > bias (got Y={noise_term}, eps-bias={denom})"
        )
    return 9.0 * noise_term**2 / (4.0 * denom**2)
