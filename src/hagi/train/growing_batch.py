"""Growing-batch расписание R371: стохастический хвост гаснет сам.

Теорема (primes R371, ``Hagi/Foundations/Recurrence``,
``growing_batch_stabilized``): рекурсия

    E(t+1) ≤ ρ·E(t) + η²σ²/(b₀·δ^t),   ρ = 1 − ηL < 1, δ ≥ 1, ρδ > 1

при геометрически растущем batch b(t) = b₀·δ^t даёт

    E(T) ≤ ρ^T·E₀ + c·ρ^(T−1)/(1 − (ρδ)^(−1)),

т.е. шумовой (стохастический) член гаснет на СКОРОСТИ КОНТРАКЦИИ ρ,
а не остаётся полкой η²σ²/b₀ фиксированного batch (plan §2
growing-batch family, WSqD-наследие R254 выведено, а не подобрано).

Runtime-порт: grad-accumulation растёт геометрически

    accum(t) = round(base · δ^t),  capped (память),

эквивалент LR↔batch (ALGORITHMS.md §2, lr_batch_equivalence):
эффективный batch растёт при фиксированном η — последовательных
обновлений на ту же кривую меньше, шумовой бюджет шага сжимается.
"""

from __future__ import annotations

import math


def growing_accum(base: int, delta: float, step: int, cap: int | None = None) -> int:
    """R371: accum(t) = max(1, round(base · δ^t)) с капом по памяти.

    ``base`` — стартовый grad_accum_steps (b₀ в единицах microbatch);
    ``delta`` — геометрический фактор роста batch (δ ≥ 1); ``step`` —
    номер optimizer-шага; ``cap`` — верхняя граница (память/латентность
    Витерби занятости GPU); None → без капа (осторожно: δ^t растёт
    быстро, обычно cap обязателен).
    """
    if base < 1:
        raise ValueError(f"base must be >= 1, got {base}")
    if delta < 1.0:
        raise ValueError(f"delta must be >= 1 (theorem), got {delta}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if cap is not None and cap < base:
        raise ValueError(f"cap ({cap}) must be >= base ({base})")
    acc = max(1, round(base * (delta ** step)))
    if cap is not None:
        acc = min(acc, cap)
    return acc


def tail_bound(rho: float, E0: float, c: float, delta: float, T: int) -> float:
    """R371: E(T) ≤ ρ^T·E₀ + c·ρ^(T−1)/(1 − (ρδ)^(−1)).

    Посылки теоремы: 0 < ρ < 1 (контракция), δ ≥ 1, ρ·δ > 1
    (растущий batch обгоняет контракцию — иначе хвост не гаснет),
    E₀ ≥ 0 — стартовый уровень ошибки, c ≥ 0 — масштаб шумовой
    инъекции (η²σ²/b₀ в единицах шага).
    """
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho must be in (0,1), got {rho}")
    if delta < 1.0:
        raise ValueError(f"delta must be >= 1 (theorem), got {delta}")
    if rho * delta <= 1.0:
        raise ValueError(
            f"theorem condition rho*delta > 1 violated: "
            f"{rho:.4f}*{delta:.4f} = {rho * delta:.4f} <= 1. "
            "The stochastic tail does NOT extinguish — raise delta "
            "(grow the batch faster) or reduce eta."
        )
    if E0 < 0.0 or c < 0.0:
        raise ValueError("E0, c must be >= 0")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}")
    inv = 1.0 / (rho * delta)
    return rho ** T * E0 + c * rho ** (T - 1) / (1.0 - inv) if T >= 1 else E0


def delta_from_rho(rho: float, safety: float = 1.0) -> float:
    """Минимальный δ, гасящий хвост на скорости контракции.

    Теорема требует ρ·δ > 1 ⟹ δ > 1/ρ. При safety = 1 берётся строгая
    граница снизу (минимальный рост batch); safety > 1 гасит хвост
    быстрее контракции (запас). ρ = 1 − ηL выводится из η-капа 1/L
    (ALGORITHMS.md §2: η = 1/L всегда) и фактически занятой доли η.
    """
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho must be in (0,1), got {rho}")
    if safety < 1.0:
        raise ValueError(f"safety must be >= 1, got {safety}")
    return safety / rho
