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


# --- WSqD noise budget (Probability/WSqD.lean R, NoiseTemperature.lean) ----


def batch_variance(sigma2: float, batch: int) -> float:
    """Reservoir noise temperature of a batch estimator [NoiseTemperature
    ``batch_variance_eq``]: Var(batchMean) = popVar / B. The gradient
    noise enters the step at scale sigma2 / B — the batch axis is a
    noise-cooling axis ("1/B temperature").

    ``batch`` must be >= 1; sigma2 >= 0 is the population gradient
    variance (measured, h_emb_).
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if sigma2 < 0.0:
        raise ValueError("sigma2 must be nonnegative")
    return sigma2 / batch


def stable_phase_noise(sigma2: float, batch0: float, n_stable: int) -> float:
    """Stable-phase noise cost [WSqD ``stable_phase_noise``]:
    n_w steps at constant batch B_0 cost exactly sigma^2 * n_w / B_0.
    """
    if batch0 <= 0.0:
        raise ValueError("batch0 must be positive")
    if n_stable < 0:
        raise ValueError("n_stable must be nonnegative")
    return sigma2 * n_stable / batch0


def cooldown_tail_constant(sigma2: float, batch0: float, c: float) -> float:
    """Cooldown tail is CONSTANT [WSqD ``cooldown_tail_constant`` ←
    ``anneal_by_batch``]: a decay phase of ANY length with geometric
    batch growth B_{t+1} = c*B_t (c > 1) contributes at most
    sigma^2 * c / (B_0 (c-1)) — an unboundedly long cooldown is nearly
    free in noise.
    """
    if batch0 <= 0.0:
        raise ValueError("batch0 must be positive")
    if c <= 1.0:
        raise ValueError("growth factor c must be > 1")
    if sigma2 < 0.0:
        raise ValueError("sigma2 must be nonnegative")
    return sigma2 * c / (batch0 * (c - 1.0))


def wsqd_noise_budget(
    sigma2: float, batch0: float, n_stable: int, c: float
) -> float:
    """Composed WSqD noise budget [WSqD ``wsqd_noise_budget``]:
    stable phase (n_w steps at B_0) + decay phase (any length, growth c)
    obeys the ADDITIVE budget
        sigma^2 * n_w / B_0 + sigma^2 * c / (B_0 (c-1)).
    The stable phase does the work; the cooldown polishes at a fixed
    total price however long it runs — the scheduler licence behind
    ``growing_accum``.
    """
    return stable_phase_noise(sigma2, batch0, n_stable) + cooldown_tail_constant(
        sigma2, batch0, c
    )


def flatness_moment_bound(tail_constant: float) -> float:
    """Stationary-flatness kernel [StationaryFlatness
    ``flatness_moment_bound``]: a flatness functional with tail
    P[X >= k] <= C/k^2 has E[X] <= 2C — "flat minima generalize"
    becomes a measurable moment from the tail constant C (measured
    from a k-sweep of the loss Hessian proxy).
    """
    if tail_constant < 0.0:
        raise ValueError("tail constant C must be nonnegative")
    return 2.0 * tail_constant
