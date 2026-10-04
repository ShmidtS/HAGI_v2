"""T2 (переписанный R129): MergePrice Σ — цена слияния в метрике задач.

Рецензия 2026-10-04 вердикт 1: старый R129 тавтологичен (mergeDecomp —
определение, «2.8e-16» проверяет float-сложение). Содержательная теорема
— квадратичная цена слияния (2607.09202 Interference and Retention):

Th.1  ΔL_i = ½·ΔᵀΣ_iΔ, где Δ = θ̄ − θ_i = −dev_i (усреднение смещает
      каждый эксперт на его отклонение от среднего), Σ_i — Гессиан
      (сегментно усреднённый) задачи i. Потеря на задаче i — энергия
      интерференции в метрике Σ_i, НЕ линейная норма ‖dev‖.

Th.4  ΔL_i = 0 ⟺ dev_i ∈ ker Σ_i — Σ-ортогонализация необходимо И
      достаточна (generically necessary). Это поднимает merge-гейт с
      эмпирической меры конфликта до точного критерия.

Th.5  distortion floor D ≥ ¼·σ_u·δ² на shared-направлении с
      расхождением δ (никакое слияние не лучше пола).

Runtime-формы (этот модуль):

* :func:`merge_price` — ½·devᵀΣ·dev по теореме Th.1;
* :func:`sigma_gram` — эмпирическая Σ из активаций (XᵀX/n): для
  квадратичной модели лосса Гессиан по весам последнего слоя равен
  Gram-матрице входов (segment-averaged surrogate);
* :func:`ker_violation` — доля энергии dev в строчном пространстве Σ
  (Th.4: цена нулевая ⟺ эта доля нулевая);
* :func:`merge_gate` — сливать только если twoGap > Σᵢ wᵢ·priceᵢ +
  κ√n·s/2 (цена сжатия GapLaw): прирост от разногласия должен
  ПОКРЫВАТЬΣ-цену, а не только шум;
* :func:`distortion_floor` — Th.5: D ≥ ¼σ_uδ².

Единицы: всё в натах (CE) — метрического смешения нет (вердикт 2
рецензии об η без метрики Σ адресован здесь явно).
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "merge_price",
    "sigma_gram",
    "ker_violation",
    "merge_gate",
    "distortion_floor",
]


def merge_price(dev: torch.Tensor, sigma: torch.Tensor) -> float:
    """Th.1: ΔL_i = ½·devᵀΣ_i·dev — цена усреднения для задачи i.

    ``dev`` — отклонение эксперта от среднего (θ_i − θ̄; знак не важен,
    форма квадратичная), ``sigma`` — [d, d] Гессиан-суррогат задачи.
    Возвращает скаляр в натах квадратичной модели.
    """
    if dev.ndim != 1:
        raise ValueError(f"dev must be 1-D, got {tuple(dev.shape)}")
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"sigma must be square, got {tuple(sigma.shape)}")
    if sigma.shape[0] != dev.shape[0]:
        raise ValueError(
            f"sigma {tuple(sigma.shape)} incompatible with dev {tuple(dev.shape)}"
        )
    d = dev.to(torch.float64)
    s = sigma.to(torch.float64)
    return 0.5 * float(d @ s @ d)


def sigma_gram(activations: torch.Tensor) -> torch.Tensor:
    """Эмпирическая Σ из активаций: XᵀX/n (segment-averaged surrogate).

    Для квадратичной модели лосса ``L(w) = ½‖Xw − y‖²/n`` Гессиан по
    весам последнего слоя — ровно XᵀX/n; для CE-модели это стандартный
    суррогат (Fisher/GGN), чего достаточно для ГЕЙТА (сравнение с
    twoGap в тех же единицах).
    """
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be [n, d], got {tuple(activations.shape)}"
        )
    x = activations.to(torch.float64)
    n = x.shape[0]
    return (x.T @ x) / n


def ker_violation(dev: torch.Tensor, sigma: torch.Tensor) -> float:
    """Th.4: доля энергии dev ВНЕ ker Σ (в строчном пространстве Σ).

    Цена нулевая ⟺ dev ∈ ker Σ ⟺ эта доля = 0. Вычисляется через
    собственную декомпозицию Σ (симметричной): проекция dev на
    собственные направления с |λ| > tol, доля их энергии.
    """
    lam, vecs = torch.linalg.eigh(sigma.to(torch.float64))
    tol = 1e-12 * float(lam.abs().max().clamp_min(1e-30))
    live = lam.abs() > tol
    proj = vecs[:, live].T @ dev.to(torch.float64)
    dev_n2 = float(dev.to(torch.float64) @ dev.to(torch.float64))
    if dev_n2 == 0.0:
        return 0.0
    return float(proj @ proj) / dev_n2


def merge_gate(
    two_gap: float,
    prices: list[float],
    weights: list[float] | None = None,
    kappa: float = 0.05,
    s: float = 1.0,
    n: int = 200,
) -> bool:
    """Гейт слияния: twoGap > Σᵢ wᵢ·priceᵢ + κ√n·s/2.

    Правая часть — Σ-цена усреднения (Th.1, взвешенная по доменам) ПЛЮС
    цена тернарного сжатия GapLaw (κ√n·s/2). Прирост разногласия должен
    покрывать обе; иначе слияние дороже своего выигрыша.
    """
    if two_gap <= 0.0:
        raise ValueError("two_gap must be > 0 (no disagreement, no merge)")
    if kappa < 0.0 or s < 0.0 or n <= 0:
        raise ValueError("kappa, s >= 0 and n > 0 required")
    w = weights if weights is not None else [1.0 / len(prices)] * len(prices)
    if len(w) != len(prices):
        raise ValueError("weights and prices length mismatch")
    total_price = sum(wi * pi for wi, pi in zip(w, prices))
    compression_cost = kappa * math.sqrt(n) * s / 2.0
    return two_gap > total_price + compression_cost


def distortion_floor(sigma_u: float, delta: float) -> float:
    """Th.5: D ≥ ¼·σ_u·δ² — пол искажения на shared-направлении.

    σ_u — сингулярное значение shared-направления (чувствительность),
    δ — расхождение экспертов вдоль него. Никакая схема слияния не
    заходит ниже этого пола на этом направлении.
    """
    if sigma_u < 0.0:
        raise ValueError("sigma_u must be >= 0")
    return 0.25 * sigma_u * delta * delta
