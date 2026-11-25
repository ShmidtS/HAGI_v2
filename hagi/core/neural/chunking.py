"""Адаптивная сегментация последовательностей, растущая адресная память и маршрутизатор экспертов.

Содержит три независимых механизма:
  * сегментация по границам (stutter_norm, boundary_chunks, boundary_heads):
    подряд идущие одинаковые элементы схлопываются; новый сегмент начинается
    с элемента, для которого boundary(элемент) истинно; heads — первые элементы
    сегментов;
  * GDNRAM — адресная память с затуханием и воротами: запись в слот addr
    mem[addr] ← (1 − decay_g)·mem[addr]·(1 − gate_beta) + gate_beta·val;
  * HedgeRouter — маршрутизатор «эксперты с весами» для потерь из [0, 1]:
    L_i — накопленная потеря эксперта i, веса
    w_i = exp(−eta·L_i) / Σ_k exp(−eta·L_k).
    Для численной устойчивости вычисление идёт в логарифмах: из логитов вычитается
    максимум и логарифм суммы экспонент. Гарантируется нижняя граница
    w_i >= exp(−eta·L_i) / K, проверяемая в логарифмической шкале.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов
import math  # модули стандартной библиотеки: аргументы командной строки, абстрактное дерево, копирование, куча, переборы, JSON, математика, ОС, случайность
from typing import Any, Callable, Sequence  # типы аннотаций: произвольный, вызываемый, перечислимый, необязательный, последовательность
import numpy as np  # численные массивы
from hagi.core.synthesis.prefilter import (  # импорт из модуля предварительной фильтрации
    Any,  # тип «любой»
    Sequence  # тип «последовательность»
)  # конец импорта

# ==============================================================================
# Адаптивная сегментация, адресная память и маршрутизатор с ограниченной потерей
# ==============================================================================


def boundary_chunks(  # разбить последовательность на сегменты
    boundary: Callable[[Any], bool], xs: Sequence[Any]  # boundary — признак начала сегмента; xs — элементы
) -> list[list[Any]]:  # аннотация возврата
    """Разбить xs на сегменты: элемент с boundary(элемент) = True начинает новый сегмент.

        Возвращает список сегментов (каждый — список элементов); пустой вход
        даёт пустой список. Первый элемент всегда начинает первый сегмент."""
    if not xs:  # пустая последовательность
        return []  # нет сегментов
    chunks = [[xs[0]]]  # первый сегмент из первого элемента
    for tok in xs[1:]:  # по остальным элементам
        if boundary(tok):  # элемент начинает новый сегмент
            chunks.append([tok])  # новый сегмент из одного элемента
        else:  # иначе
            chunks[-1].append(tok)  # добавить в текущий (последний) сегмент
    return chunks  # список сегментов


class GDNRAM:
    """Адресная память с запаздыванием на один шаг: запись с воротами и затуханием, чтение без изменения.

        Память — матрица mem формы (n_slots, dim). При записи в слот addr
        значения val: mem[addr] ← (1 − decay_g)·mem[addr]·(1 − gate_beta) +
        gate_beta·val. Чтение возвращает копию слота."""

    def __init__(self, n_slots: int, dim: int):  # n_slots — число слотов; dim — размерность слота
        """Создать память из n_slots слотов размерности dim, заполненную нулями."""
        self.mem = np.zeros((n_slots, dim), dtype=np.float64)  # матрица нулей формы (n_slots, dim)

    def store(  # записать значение в слот
        self, addr: int, val: np.ndarray, gate_beta: float = 1.0, decay_g: float = 0.0  # addr — номер слота; val — значение; gate_beta — пропуск ворот; decay_g — затухание
    ) -> None:  # ничего не возвращает
        """Записать val в слот addr: mem ← (1 − decay_g)·mem·(1 − gate_beta) + gate_beta·val.

            gate_beta — доля нового значения, decay_g — доля дополнительного
            затухания старого; при gate_beta = 1, decay_g = 0 слок перезаписывается."""
        self.mem[addr] = (1.0 - decay_g) * self.mem[addr] * (  # старое значение, умноженное на (1 − decay_g)
            1.0 - gate_beta  # и на (1 − gate_beta)
        ) + gate_beta * val  # плюс gate_beta·val

    def load(self, addr: int) -> np.ndarray:  # прочитать слот addr
        """Вернуть копию слота addr (изменение результата не меняет память)."""
        return self.mem[addr].copy()  # копия строки addr


class HedgeRouter:
    """Устойчивый в численном отношении экспоненциально взвешенный маршрутизатор для потерь из [0, 1].

        K экспертов, темп обучения eta ∈ (0, 1]. Для накопленных потерь L_i веса
        w_i = exp(−eta·L_i) / Σ_k exp(−eta·L_k). После каждого обновления
        L_i ← L_i + loss_i; возвращает копию весов."""

    def __init__(self, n_experts: int, eta: float = 0.5):  # n_experts — число экспертов; eta — темп обучения
        """Создать маршрутизатор на n_experts экспертов с равными весами 1 / K и нулевыми потерями."""
        if int(n_experts) <= 0:  # число экспертов не положительно
            raise ValueError("n_experts must be positive")  # ошибка аргумента
        if not math.isfinite(float(eta)) or not (0.0 < float(eta) <= 1.0):  # eta не конечна или вне (0, 1]
            raise ValueError("eta must be finite and in (0, 1]")  # ошибка аргумента
        self.K = int(n_experts)  # число экспертов K
        self.eta = float(eta)  # темп обучения eta
        self.weights = np.ones(self.K, dtype=np.float64) / self.K  # веса: w_i = 1 / K
        self.cum_loss = np.zeros(self.K, dtype=np.float64)  # накопленные потери L_i = 0

    def update(self, losses_01: np.ndarray) -> np.ndarray:  # обновить веса по потерям из [0, 1]
        """Прибавить потери к накопленным и пересчитать веса.

            losses_01 — потери экспертов за шаг, каждая в [0, 1], форма (K,).
            L_i ← L_i + loss_i; w_i = exp(−eta·L_i) / Σ_k exp(−eta·L_k).
            Возвращает копию нового вектора весов."""
        losses = np.asarray(losses_01, dtype=np.float64)  # потери вещественным массивом
        if losses.shape != (self.K,):  # неверная форма
            raise ValueError(f"losses must have shape ({self.K},)")  # ошибка формы
        if (  # проверка значений потерь
            not np.all(np.isfinite(losses))  # есть не конечная потеря
            or np.any(losses < 0.0)  # есть потеря меньше 0
            or np.any(losses > 1.0)  # есть потеря больше 1
        ):  # конец условия
            raise ValueError("expert losses must be finite and lie in [0, 1]")  # ошибка значений
        self.cum_loss += losses  # накопленная потеря: L_i ← L_i + loss_i
        if not np.all(np.isfinite(self.cum_loss)):  # накопленная потеря перестала быть конечной
            raise OverflowError("cumulative expert loss overflowed")  # переполнение
        # Логарифм суммы экспонент не даёт вектору весов обратиться в нуль
        # после многих раундов.
        logw = -self.eta * self.cum_loss  # логарифмы весей: log w_i = −eta·L_i (до нормировки)
        logw -= float(np.max(logw))  # вычесть максимум для устойчивости: log w_i − max_j log w_j
        logz = float(np.log(np.sum(np.exp(logw))))  # логарифм суммы: log Z = log Σ_k exp(log w_k)
        logw -= logz  # нормировка в логарифмах: log w_i ← log w_i − log Z
        # Нижняя граница проверяется в логарифмической шкале: w_i >= exp(−eta·L_i) / K.
        lower_log = -math.log(self.K) - self.eta * self.cum_loss  # log нижней границы: −log K − eta·L_i
        if np.any(logw + 1e-10 < lower_log):  # какой-то вес ниже границы (с допуск 1e-10)
            raise ArithmeticError("Hedge survivor lower-bound invariant failed")  # инвариант нарушен
        self.weights = np.exp(logw)  # веса: w_i = exp(log w_i)
        self.weights /= np.sum(self.weights)  # окончательная нормировка: w_i ← w_i / Σ w_k
        return self.weights.copy()  # копия весов

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'boundary_chunks', 'GDNRAM', 'HedgeRouter',  # имя интерфейса
]  # конец перечня
