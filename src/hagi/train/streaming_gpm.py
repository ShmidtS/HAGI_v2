"""Шаг 2 плана формализатора: streaming-PCA U_cov — GPM-базис без
хранения активаций.

R131 (safeqp_gpm) требует базис row-space активаций старой задачи;
activation_subspace делает полный SVD по X [n, d] — на живом joint-стейдже
хранить n строк активаций нельзя (H=10368: один батч — миллионы строк).
Frequent Directions (Liberty 2013; Ghashami-DeWitt 2016) держит скетч
B [ell, d], ell << n, с ДЕТЕРМИНИРОВАННОЙ границей ковариации (без
концентрации, без условий на данные):

    SimpleFD (G-D'16 Alg 1, halving):  ‖XᵀX − BᵀB‖₂ ≤ 2‖X‖²_F / ell
    FD (G-D'16 Alg 2, вычитание):      PSD-невязка + tail-форма (Thm 3)

Реализован Alg 2 (вычитание σ_min² из каждого сингулярного значения):
halving геометрически убивает сигнал на длинных потоках (~1000
переполнений → денормали), вычитание вычитает только слабейшее
направление за переполнение — сигнал выживает, невязка PSD.

Цепочка к R131 (теорема та же, базис потоковый):

1. ε_cov = 2‖X‖²_F/ell — консервативный envelope (Thm 1).
2. Weyl: |λ_i(XᵀX) − λ_i(BᵀB)| ≤ ε_cov — спектр скетча близок к
   спектру ковариации активаций.
3. Davis-Kahan (safeqp_gpm.davis_kahan_bound): sin θ между базисом
   скетча и истинным подпространством ≤ ε_cov/gap.
4. empirical_eps_budget: ε_i = ‖d‖·‖g_i‖·sin θ — SafeQP-бюджет на
   потоковый базис. Первый порядок forgetting на stream-базисе не 0,
   а ≤ ε_i — та же легитимация ε-бюджета, что у R131 против шума
   эмпирического базиса.

Runtime-формы:

* :class:`FrequentDirections` — потоковый скетч (update / basis /
  sketch / cov_error_bound / reset);
* :func:`streaming_activation_basis` — streaming-замена
  activation_subspace: тот же контракт [d, r], кормится в gpm_step /
  space_lora_factor без изменений.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["FrequentDirections", "streaming_activation_basis"]


class FrequentDirections:
    """FD-скетч row-space потока активаций (фиксированный ell × d).

    update() поглощает чанки [n, d]; при заполнении ell строк — SVD,
    вычитание σ_min² из каждого сингулярного значения, повёрнутые
    строки diag(√(s²−σ_min²))·Vᵀ (слабейшая строка → 0, слот свободы).
    basis() возвращает ортонормальный базис top-r направлений —
    контракт activation_subspace.
    """

    def __init__(self, dim: int, sketch_rows: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if sketch_rows < 2:
            raise ValueError(f"sketch_rows must be >= 2, got {sketch_rows}")
        self.dim = dim
        self.sketch_rows = sketch_rows
        self._b = torch.zeros(sketch_rows, dim, dtype=torch.float64)
        self._used = 0
        self._fro2 = 0.0

    def update(self, x: torch.Tensor) -> None:
        """Поглотить чанк активаций [*, d] (любые ведущие формы)."""
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        rows = x.detach().reshape(-1, self.dim).to(torch.float64)
        if rows.shape[0] == 0:
            return
        self._fro2 += float((rows * rows).sum())
        s = 0
        while s < rows.shape[0]:
            take = min(self.sketch_rows - self._used, rows.shape[0] - s)
            self._b[self._used:self._used + take] = rows[s:s + take]
            self._used += take
            s += take
            if self._used == self.sketch_rows:
                self._compress()

    def _compress(self) -> None:
        # скетч полон: все ell строк ненулевые, σ_min > 0; после
        # вычитания слабейшая строка (или больше, при связях) → 0
        _, s, vh = torch.linalg.svd(self._b, full_matrices=False)
        s2 = (s.square() - s.min().square()).clamp_min(0.0)
        self._b = s2.sqrt().unsqueeze(1) * vh
        # гарантируем ≥1 свободный слот — иначе update зациклится
        self._used = min(int((s2 > 0.0).sum()), self.sketch_rows - 1)

    def sketch(self) -> torch.Tensor:
        """Текущий скетч B [ell, d] (строки за _used — нули)."""
        return self._b

    def cov_error_bound(self) -> float:
        """ε_cov = 2‖X‖²_F/ell — консервативная граница ‖XᵀX−BᵀB‖₂."""
        return 2.0 * self._fro2 / self.sketch_rows

    def basis(self, rank: int | None = None) -> torch.Tensor:
        """Ортонормальный базис [d, r] top-r направлений скетча.

        r = rank или полный численный ранг (σ > tol относительно
        σ_max — толеранс масштабо-инвариантен к вычитаниям). Пустой
        скетч → [d, 0].
        """
        if self._used == 0:
            return torch.zeros(self.dim, 0, dtype=torch.float64)
        _, s, vh = torch.linalg.svd(self._b[: self._used], full_matrices=False)
        tol = 1e-10 * max(1.0, float(s.max()))
        live = int((s > tol).sum())
        r = min(rank, live) if rank is not None else live
        return vh[:r].T.contiguous()

    def reset(self) -> None:
        """Сброс скетча (новая задача / новый домен)."""
        self._b = torch.zeros_like(self._b)
        self._used = 0
        self._fro2 = 0.0


def streaming_activation_basis(
    chunks: Iterable[torch.Tensor],
    dim: int,
    sketch_rows: int,
    rank: int | None = None,
) -> torch.Tensor:
    """Streaming-замена activation_subspace: базис [d, r] из потока.

    ``chunks`` — итератор тензоров [*, d] (активации по батчам,
    память O(sketch_rows · dim)). Контракт выхода совпадает с
    activation_subspace: колонки ортонормальны, кормится в gpm_step /
    space_lora_factor без изменений.
    """
    fd = FrequentDirections(dim, sketch_rows)
    for chunk in chunks:
        fd.update(chunk)
    return fd.basis(rank)
