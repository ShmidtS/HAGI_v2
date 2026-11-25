"""Адаптивная сегментация и ассоциативная память с лагом 1 — модуль нейросетевой группы пакета hagi.core.neural.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

from typing import Any, Callable, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части

# ==============================================================================
# 7. Адаптивная сегментация HNet и ассоциативная память с лагом 1 (GDNRAM)
# ==============================================================================


def hnet_opt_ratio(a: float, b: float, T: float) -> float:  # оптимальная доля сегмента HNet
    """Оптимальное отношение длины сегмента: s = (2·b·T/a)^(1/3).

        a — цена за сегмент, b — цена за токен внутри сегмента, T — горизонт."""
    return (2.0 * b * T / a) ** (1.0 / 3.0)  # кубический корень: s = (2·b·T/a)^(1/3)


def stutter_norm(seq: Sequence[Any]) -> list[Any]:  # убрать подряд идущие повторы
    """Схлопнуть подряд идущие одинаковые элементы: из [a, a, b] → [a, b].
    
        seq: любая последовательность. Возвращает новый список без соседних дубликатов."""
    out: list[Any] = []  # результат
    for item in seq:  # по элементам
        if not out or out[-1] != item:  # пусто или отличается от последнего
            out.append(item)  # добавить
    return out  # вернуть


def boundary_chunks(  # разбить последовательность на сегменты по границе
    boundary: Callable[[Any], bool], xs: Sequence[Any]  # предикат границы и элементы
) -> list[list[Any]]:  # возвращает список сегментов
    """Разбить xs на сегменты: элемент, для которого boundary(x) истинно, начинает новый сегмент.
    
        boundary: предикат границы; xs: элементы. Первый элемент всегда начинает первый сегмент.
        Возвращает список сегментов (пустой список для пустых xs)."""
    if not xs:  # пустая последовательность
        return []  # пустой список
    chunks = [[xs[0]]]  # первый сегмент из первого элемента
    for tok in xs[1:]:  # по остальным элементам
        if boundary(tok):  # токен — граница
            chunks.append([tok])  # начать новый сегмент
        else:  # иначе
            chunks[-1].append(tok)  # добавить в текущий сегмент
    return chunks  # вернуть сегменты


def boundary_heads(boundary: Callable[[Any], bool], xs: Sequence[Any]) -> list[Any]:  # первые элементы сегментов
    """Вернуть первые элементы сегментов: [c[0] для каждого сегмента boundary_chunks]."""
    return [c[0] for c in boundary_chunks(boundary, xs)]  # первые элементы сегментов


class GDNRAM:  # ассоциативная память с лагом 1
    """Ассоциативная память с лагом 1: запись и чтение векторов по адресу-слоту.
    
        Хранит матрицу mem (n_slots, dim). Адрес берётся по модулю числа слотов; значения
        длиннее dim обрезаются, короче — дополняются нулями. Запись с вентилем gate_beta и
        затуханием decay_g: mem[slot] = (1 − decay_g)·mem[slot]·(1 − gate_beta) + gate_beta·v.
        Чтение по последовательности адресов («заикание») схлопывает повторы и берёт первый адрес."""

    def __init__(self, n_slots: int, dim: int):  # создать память: n_slots слотов, dim-мерные векторы
        self.mem = np.zeros((n_slots, dim), dtype=np.float64)  # нулевая матрица памяти (n_slots, dim)

    def snapshot(self) -> np.ndarray:  # снимок памяти
        return self.mem.copy()  # копия матрицы

    def restore(self, snap: np.ndarray) -> None:  # восстановить из снимка
        self.mem = snap.copy()  # заменить память копией снимка

    def store(  # записать вектор в слот
        self, addr: int, val: np.ndarray, gate_beta: float = 1.0, decay_g: float = 0.0  # адрес, значение, вентиль, затухание
    ) -> None:  # ничего не возвращает
        slot = int(addr) % self.mem.shape[0]  # номер слота: addr mod n_slots
        v = np.asarray(val, dtype=np.float64).ravel()[: self.mem.shape[1]]  # вектор-строка, обрезанный до dim
        if v.size < self.mem.shape[1]:  # короче dim
            v = np.pad(v, (0, self.mem.shape[1] - v.size))  # дополнить нулями до dim
        self.mem[slot] = (1.0 - decay_g) * self.mem[slot] * (  # запись: mem[slot] = (1 − decay_g)·mem[slot]·
            1.0 - gate_beta  # умножить на (1 − gate_beta)
        ) + gate_beta * v  # плюс gate_beta·v

    def load(self, addr: int) -> np.ndarray:  # прочитать слот
        return self.mem[int(addr) % self.mem.shape[0]].copy()  # копия вектора слота addr mod n_slots

    def load_stutter_head(self, addr_seq: Sequence[int]) -> np.ndarray:  # прочитать первый адрес после схлопывания повторов
        norm = stutter_norm(addr_seq)  # адреса без соседних повторов
        return (  # вернуть
            self.load(norm[0])  # значение первого адреса
            if norm  # если адреса есть
            else np.zeros(self.mem.shape[1], dtype=np.float64)  # иначе нулевой вектор
        )

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'hnet_opt_ratio', 'stutter_norm', 'boundary_chunks', 'boundary_heads', 'GDNRAM',  # имя интерфейса
]  # конец перечня
