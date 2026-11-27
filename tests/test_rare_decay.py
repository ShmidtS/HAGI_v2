"""Тест связанного затухания весов (rare decay).

Теория: primes Hagi/Step/RareDecay.lean (Pinto, Galanti, Balestriero,
«The Fair Language Model Paradox», arXiv:2410.11985):
  - dense_decay_fixed_point: затухание на КАЖДОМ шаге даёт равновесие w* = f·η·g/λ —
    масштаб редкого признака (частота f) подавляется пропорционально f;
  - coupled_decay_fixed_point: затухание только на шагах обновления
    (HAGI_v2 HybridMuonAdamW: p.mul_(where(act, 1 − lr·wd, 1))) даёт w* = η·g/λ —
    равновесный масштаб НЕ зависит от частоты;
  - dense_over_coupled: отношение равновесий равно f.

Сторож: если «упростить» затухание до плотного, этот тест падает — редкие строки
таблицы теряют масштаб относительно частых."""

import torch as t  # тензоры

from hagi.models.optimal_lm import HybridMuonAdamW  # оптимизатор


def _table_step(opt, table, row_grads: dict[int, float], steps: int) -> None:
    """Сделать steps шагов оптимизатора для одной таблицы; row_grads — градиент строки (одинаковый каждый шаг)."""
    for _ in range(steps):  # по шагам
        table.grad = t.zeros_like(table)  # сброс градиента
        for r, gv in row_grads.items():  # по активным строкам
            table.grad[r] = gv  # задать градиент строки
        opt.step()  # шаг оптимизатора


def test_coupled_decay_keeps_rare_row_scale():
    t.manual_seed(0)  # зерно
    table = t.nn.Parameter(t.zeros(8, 2))  # таблица 8 строк × 2
    vec = t.nn.Parameter(t.zeros(2))  # вектор (не участвует)
    opt = HybridMuonAdamW([], [table, vec], lr_muon=0.0, lr_adam=0.05, momentum=0.0, wd=0.5)  # сильное затухание
    opt.names = ["table", "vec"]  # имена в порядке mats + vecs (без них ключи GPM не находятся)
    rare, common = 3, 5  # редкая и частая строки
    f = 0.2  # редкая строка обновляется в 1 из 5 шагов
    for k in range(600):  # длинная серия
        table.grad = t.zeros_like(table); vec.grad = t.zeros_like(vec)  # сброс
        table.grad[common] = 1.0  # частая — каждый шаг
        if k % 5 == 0:  # редкая — каждый пятый
            table.grad[rare] = 1.0  # тот же градиент
        opt.step()  # шаг
    ratio = float(table[rare].abs().sum() / table[common].abs().sum())  # отношение масштабов
    # Связанное затухание: отношение ~1 (не зависит от f). Плотное дало бы ~f = 0.2.
    assert ratio > 0.6, f"редкая строка потеряла масштаб: отношение {ratio:.3f} (ожидалось ~1, плотное затухание дало бы ~{f})"
