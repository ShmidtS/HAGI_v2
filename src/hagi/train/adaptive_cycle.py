"""Adaptive cycle controllers (Optimization/{DirectionPacking,
SubspaceGrowth,InexactContraction,SphericalDamping,AdaptiveCycle}.lean).

Порт «трёх геометрий» адаптивной HAGI:

* **Упаковка направлений** (Welch): M эксперт-направлений с
  когерентностью μ вмещаются в ранг d только при
  ``M/(1+(M−1)·μ²) ≤ d`` (``welch_min_dimension``,
  DirectionPacking.lean:273). Контрапозиция ``welch_capacity_full``
  (DirectionPacking.lean:299) — триггер роста ранга: если
  (M+1)/(1+M·μ²) > d, новое направление невозможно без роста d или μ.
* **Рост подпространства**: новое измерение легитимно только при
  достаточной невязке проекции ``projResidual u v = v − Σ⟪uᵢ,v⟫uᵢ``
  (SubspaceGrowth.lean); рост конечен — не более ⌊C₀/δ⌋ расширений
  (``growth_finite`` / ``adaptive_cycle_expansions``: N·δ ≤ C₀).
* **Внутренний пересчёт**: неточное сжатие x(k+1) ≤ q·x(k)+ε имеет
  пол ε/(1−q) и явный горизонт останова K: q^K < (τ − ε/(1−q))/B
  (``inexact_contraction_halt``, InexactContraction.lean).
* **Угловое демпфирование**: окно устойчивого шага 0 < η < 2/(1−ζ)
  для действительной моды ζ < 1 (``angular_damping_window``,
  SphericalDamping.lean:55); хордовый остаток останова
  ρ = 1 − ⟨u,v⟩ = ½‖u−v‖² (``chordal_residual_eq``, :38).
"""
from __future__ import annotations

import math

import torch

# --- Welch packing (DirectionPacking.lean) -------------------------------


def welch_min_dimension(m: int, mu: float, d: int) -> bool:
    """M/(1 + (M−1)·μ²) ≤ d — вмещаются ли M направлений с
    когерентностью μ в ранг d (``welch_min_dimension``,
    DirectionPacking.lean:273).
    """
    if m < 1:
        raise ValueError("M must be >= 1")
    if mu < 0.0:
        raise ValueError("mu must be nonnegative")
    if d < 0:
        raise ValueError("d must be nonnegative")
    return m / (1.0 + (m - 1) * mu * mu) <= d


def welch_required_dimension(m: int, mu: float) -> int:
    """Минимальный ранг d = ⌈M/(1 + (M−1)·μ²)⌉ (следствие
    ``welch_min_dimension``, DirectionPacking.lean:273).
    """
    if m < 1:
        raise ValueError("M must be >= 1")
    if mu < 0.0:
        raise ValueError("mu must be nonnegative")
    return math.ceil(m / (1.0 + (m - 1) * mu * mu))


def welch_capacity_full(m: int, mu: float, d: int) -> bool:
    """Упаковка исчерпана: НЕ (M+1)/(1 + M·μ²) ≤ d — контрапозиция
    ``welch_capacity_full`` (DirectionPacking.lean:299). True —
    триггер роста ранга вместо нового листа.
    """
    return not welch_min_dimension(m + 1, mu, d)


# --- Subspace growth (SubspaceGrowth.lean) --------------------------------


def proj_residual(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Невязка представления v ортонормированным базисом (строки u,
    после QR-ортонормализации): r = v − Σᵢ⟪uᵢ,v⟫·uᵢ
    (``projResidual``, SubspaceGrowth.lean).
    """
    if u.dim() != 2:
        raise ValueError("u must be a 2-D tensor of basis rows")
    if v.dim() != 1 or v.shape[0] != u.shape[1]:
        raise ValueError("v must be a 1-D vector matching u columns")
    if u.shape[0] == 0:
        return v.clone()
    q, _ = torch.linalg.qr(u.mT)  # columns of q = orthonormalized rows
    coeffs = q.mT @ v
    return v - q @ coeffs


def proj_residual_ratio(u: torch.Tensor, v: torch.Tensor) -> float:
    """‖r‖/‖v‖ — невязка в долях нормы запроса (``projResidual_le_norm``
    даёт ratio ≤ 1). Новое направление легитимно iff ratio ≥ δ.
    """
    nv = float(v.norm())
    if nv == 0.0:
        raise ValueError("v must be nonzero")
    return float(proj_residual(u, v).norm()) / nv


def growth_finite(c0: float, delta: float) -> int:
    """Максимальное число расширений N = ⌊C₀/δ⌋ (``growth_finite`` /
    ``adaptive_cycle_expansions``: N·δ ≤ C₀, SubspaceGrowth.lean /
    AdaptiveCycle.lean).
    """
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if c0 < 0.0:
        raise ValueError("c0 must be nonnegative")
    return math.floor(c0 / delta)


# --- Inexact contraction (InexactContraction.lean) -----------------------


def contraction_floor(q: float, eps: float) -> float:
    """Предельный пол ошибки ε/(1−q) (``inexact_contraction_floor``,
    InexactContraction.lean).
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    return eps / (1.0 - q)


def inexact_contraction_halt_k(
    q: float, eps: float, tau: float, b: float
) -> int | None:
    """Минимальный K с q^K < (τ − ε/(1−q))/B; None если пол
    ε/(1−q) ≥ τ (останов невозможен) — ``inexact_contraction_halt``,
    InexactContraction.lean.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    if not b > 0.0:
        raise ValueError("B must be positive")
    floor = contraction_floor(q, eps)
    margin = tau - floor
    if margin <= 0.0:
        return None
    # q^K < margin/B  <=>  K > ln(B/margin)/ln q  (ln q < 0)
    bound = margin / b
    if bound >= 1.0:
        return 0
    k = math.floor(math.log(bound) / math.log(q))
    # smallest strict integer: adjust for float equality
    while q**k >= bound:
        k += 1
    while k > 0 and q ** (k - 1) < bound:
        k -= 1
    return k


# --- Spherical damping (SphericalDamping.lean) ---------------------------


def angular_damping_window(zeta: float) -> tuple[float, float]:
    """Окно устойчивого шага (0, 2/(1−ζ)) для действительной моды
    ζ < 1 (``angular_damping_window``, SphericalDamping.lean:55).
    Возвращает (0.0, 2/(1−ζ)); политика выбора η — снаружи.
    """
    if not zeta < 1.0:
        raise ValueError("zeta must be < 1")
    return (0.0, 2.0 / (1.0 - zeta))


def chordal_residual(u: torch.Tensor, v: torch.Tensor) -> float:
    """ρ = 1 − ⟨u,v⟩ для единичных векторов (= ½‖u−v‖²) — критерий
    останова итерации (``chordal_residual_eq``, SphericalDamping.lean:38).
    """
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    return float(1.0 - torch.dot(u.flatten(), v.flatten()))
