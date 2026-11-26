"""Описание задач и учебный набор задач для обучения экспертов.

Задача TaskSpec = имя, число аргументов arity, области допустимых значений
каждого аргумента domains, учитель teacher (функция правильного ответа) и
правило допустимости входа valid_rule:
  "all" — любой вход из областей;
  "nonzero_y" — второй аргумент не ноль (для деления и остатка);
  "ordered_yz" — второй аргумент не больше третьего (для удержания в отрезке).

Полный набор входов — декартово произведение областей
D = D_1 × … × D_arity без недопустимых строк; вход x допустим, если
len(x) = arity, x_i ∈ D_i и выполнено правило.

Учебный набор make_task_suite:
  • 16 булевых функций двух аргументов; номер строки таблицы —
    idx = 2·x + y, ответ = (mask >> idx) & 1;
  • унарные операции на отрезке −8..8: NEG = −x, ABS = |x|, SIGN,
    SQUARE = x², CUBE = x³, BIT_NOT8, CLAMP_TO_3 = min(max(x, −3), 3);
  • 22 бинарные операции на том же отрезке (арифметика, сравнения, битовые);
  • POW_INT с основанием −3..3 и показателем 0..4;
  • три тернарные задачи: FMA = x·y + z, CLAMP = min(max(x, lo), hi),
    MEDIAN3 — средний из трёх.

Экспортирует: TaskSpec, BOOL_NAMES, make_task_suite.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import itertools, math  # модули стандартной библиотеки: аргументы, синтаксическое дерево, копирование, куча, комбинаторы, JSON, математика, ОС, случайность
from dataclasses import dataclass  # класс данных TaskSpec
from typing import Callable, Sequence  # аннотации типов
from hagi.core.memory.aliases import _bit_nand8, _bit_nor8, _bit_not8, _bit_xnor8, _pow_int, _sign  # имена из «aliases» (модуль, где они определены)

# ------------------------------- Описание задач ------------------------------


@dataclass  # класс данных: автоматические __init__, __eq__ и другие
class TaskSpec:
    """Спецификация задачи: операция, её области значений и учитель.

        name — имя операции; arity — число аргументов; domains — кортеж
        областей допустимых значений (по одной на аргумент); teacher —
        функция правильного ответа; valid_rule — правило допустимости
        входа ("all", "nonzero_y", "ordered_yz"); family — имя семейства,
        по которому обучение выбирает способ поиска программы."""
    name: str  # имя операции
    arity: int  # число аргументов
    domains: tuple[tuple[int, ...], ...]  # допустимые значения каждого аргумента
    teacher: Callable[..., int]  # функция правильного ответа
    valid_rule: str = "all"  # правило допустимости входа, по умолчанию любое
    family: str = "integer"  # семейство задачи, по умолчанию целочисленное

    def inputs(self) -> list[tuple[int, ...]]:
        """Полный список допустимых входов: декартово произведение областей
        D_1 × … × D_arity минус строки, нарушающие valid_rule.

            nonzero_y отбрасывает строки с row[1] = 0; ordered_yz — строки
            с row[1] > row[2]. Каждый элемент приводится к int."""
        rows = itertools.product(*self.domains)  # все наборы (d_1, …, d_arity), d_i ∈ D_i
        out = []  # результат — список допустимых входов
        for row in rows:  # по всем наборам
            if self.valid_rule == "nonzero_y" and row[1] == 0:  # правило «второй не ноль» и второй равен нулю
                continue  # набор недопустим — пропустить
            if self.valid_rule == "ordered_yz" and row[1] > row[2]:  # правило «второй ≤ третий» и второй больше третьего
                continue  # набор недопустим — пропустить
            out.append(tuple(int(x) for x in row))  # добавить набор, каждый элемент приведён к int
        return out  # все допустимые входы

    def valid(self, xs: Sequence[int]) -> bool:
        """Допустим ли вход xs: число аргументов равно arity, каждый xs[i]
        лежит в domains[i] и выполнено правило valid_rule."""
        if len(xs) != self.arity:  # число аргументов не совпадает
            return False  # недопустим
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):  # найдётся i, для которого xs[i] ∉ D_i
            return False  # вне области — недопустим
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:  # правило «второй не ноль» нарушено
            return False  # недопустим
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):  # правило «второй ≤ третий» нарушено
            return False  # недопустим
        return True  # все условия выполнены


BOOL_NAMES = {  # имена 16 булевых функций двух аргументов по номеру маски (маска = 4-битная таблица истинности)
    0: "BOOL_FALSE",  # маска 0000: всегда 0
    1: "BOOL_NOR",  # маска 0001: ИЛИ-НЕ
    2: "BOOL_NOT_X_AND_Y",  # маска 0010: (не x) и y
    3: "BOOL_NOT_X",  # маска 0011: не x
    4: "BOOL_X_AND_NOT_Y",  # маска 0100: x и (не y)
    5: "BOOL_NOT_Y",  # маска 0101: не y
    6: "BOOL_XOR",  # маска 0110: исключающее ИЛИ
    7: "BOOL_NAND",  # маска 0111: И-НЕ
    8: "BOOL_AND",  # маска 1000: И
    9: "BOOL_XNOR",  # маска 1001: исключающее-ИЛИ-НЕ
    10: "BOOL_Y",  # маска 1010: y
    11: "BOOL_X_IMPLIES_Y",  # маска 1011: x влечёт y
    12: "BOOL_X",  # маска 1100: x
    13: "BOOL_Y_IMPLIES_X",  # маска 1101: y влечёт x
    14: "BOOL_OR",  # маска 1110: ИЛИ
    15: "BOOL_TRUE",  # маска 1111: всегда 1
}


def make_task_suite() -> list[TaskSpec]:
    """Собрать учебный набор задач: список TaskSpec.

        Порядок: 16 булевых функций, 7 унарных, 22 бинарные, POW_INT,
        затем три тернарные задачи FMA, CLAMP, MEDIAN3."""
    suite: list[TaskSpec] = []  # накопитель задач
    bool_domain = ((0, 1), (0, 1))  # оба аргумента булевы: {0, 1}
    for mask, name in BOOL_NAMES.items():  # по маске и имени каждой булевой функции
        suite.append(  # добавить задачу
            TaskSpec(  # спецификация булевой функции
                name,  # имя операции
                2,  # два аргумента
                bool_domain,  # оба аргумента из {0, 1}
                lambda x, y, m=mask: (m >> ((int(x) << 1) | int(y))) & 1,  # ответ = (m >> (2·x + y)) & 1; m закреплена значением по умолчанию
                family="BoolVenn",  # семейство — таблица истинности
            )  # конец TaskSpec
        )  # конец append

    U = tuple(range(-8, 9))  # целочисленная область U = −8..8
    unary = [  # унарные операции: (имя, функция)
        ("NEG", lambda x: -x),  # смена знака: NEG(x) = −x
        ("ABS", abs),  # модуль: ABS(x) = |x|
        ("SIGN", _sign),  # знак: [x > 0] − [x < 0]
        ("SQUARE", lambda x: x * x),  # квадрат: SQUARE(x) = x²
        ("CUBE", lambda x: x * x * x),  # куб: CUBE(x) = x³
        ("BIT_NOT8", _bit_not8),  # побитовое отрицание в 8 битах: (~x) & 255
        ("CLAMP_TO_3", lambda x: min(max(x, -3), 3)),  # удержать в отрезке: min(max(x, −3), 3)
    ]  # конец списка унарных операций
    for name, fn in unary:  # по имени и функции каждой унарной операции
        suite.append(TaskSpec(name, 1, (U,), fn, family="unary"))  # задача с одним аргументом из области U

    binary = [  # бинарные операции: (имя, функция, правило допустимости)
        ("ADD", lambda x, y: x + y, "all"),  # сумма x + y
        ("SUB", lambda x, y: x - y, "all"),  # разность x − y
        ("MUL", lambda x, y: x * y, "all"),  # произведение x·y
        ("FLOORDIV", lambda x, y: x // y, "nonzero_y"),  # деление нацело ⌊x / y⌋, y ≠ 0
        ("MOD", lambda x, y: x % y, "nonzero_y"),  # остаток x mod y, y ≠ 0
        ("GCD", math.gcd, "all"),  # наибольший общий делитель
        ("LCM", math.lcm, "all"),  # наименьшее общее кратное
        ("MIN", min, "all"),  # минимум min(x, y)
        ("MAX", max, "all"),  # максимум max(x, y)
        ("ABSDIFF", lambda x, y: abs(x - y), "all"),  # модуль разности |x − y|
        ("EQ", lambda x, y: int(x == y), "all"),  # признак равенства: [x = y]
        ("NE", lambda x, y: int(x != y), "all"),  # признак неравенства: [x ≠ y]
        ("LT", lambda x, y: int(x < y), "all"),  # признак: [x < y]
        ("LE", lambda x, y: int(x <= y), "all"),  # признак: [x ≤ y]
        ("GT", lambda x, y: int(x > y), "all"),  # признак: [x > y]
        ("GE", lambda x, y: int(x >= y), "all"),  # признак: [x ≥ y]
        ("BIT_AND", lambda x, y: x & y, "all"),  # побитовое И
        ("BIT_OR", lambda x, y: x | y, "all"),  # побитовое ИЛИ
        ("BIT_XOR", lambda x, y: x ^ y, "all"),  # побитовое исключающее ИЛИ
        ("BIT_NAND8", _bit_nand8, "all"),  # И-НЕ в 8 битах
        ("BIT_NOR8", _bit_nor8, "all"),  # ИЛИ-НЕ в 8 битах
        ("BIT_XNOR8", _bit_xnor8, "all"),  # исключающее-ИЛИ-НЕ в 8 битах
    ]  # конец списка бинарных операций
    for name, fn, valid in binary:  # по каждой тройке (имя, функция, правило)
        suite.append(  # добавить задачу
            TaskSpec(name, 2, (U, U), fn, valid_rule=valid, family="integer-binary")  # два аргумента из U, заданное правило, семейство целочисленных бинарных
        )  # конец append
    P = tuple(range(-3, 4))  # основания степени: −3..3
    E = tuple(range(0, 5))  # показатели степени: 0..4
    suite.append(TaskSpec("POW_INT", 2, (P, E), _pow_int, family="integer-binary"))  # задача возведения в степень x^y

    Z = tuple(range(-8, 9))  # область −8..8
    F = tuple(range(-4, 5))  # область −4..5 (записана как range(−4, 5))
    suite.extend(  # добавить три тернарные задачи
        [  # список задач
            TaskSpec(  # умножение со сложением
                "FMA",  # имя операции
                3,  # три аргумента
                (Z, Z, Z),  # все три из −8..8
                lambda x, y, z: x * y + z,  # умножение со сложением: FMA(x, y, z) = x·y + z
                family="composed-ternary",  # семейство составных тернарных
            ),  # конец TaskSpec
            TaskSpec(  # удержание в отрезке
                "CLAMP",  # имя операции
                3,  # три аргумента
                (Z, F, F),  # значение из −8..8, границы из −4..4
                lambda x, lo, hi: min(max(x, lo), hi),  # удержать в отрезке: CLAMP = min(max(x, lo), hi)
                valid_rule="ordered_yz",  # допустимы только lo ≤ hi
                family="composed-ternary",  # семейство составных тернарных
            ),  # конец TaskSpec
            TaskSpec(  # медиана трёх
                "MEDIAN3",  # имя операции
                3,  # три аргумента
                (F, F, F),  # все три из −4..4
                lambda x, y, z: sorted((x, y, z))[1],  # средний элемент отсортированной тройки
                family="composed-ternary",  # семейство составных тернарных
            ),  # конец TaskSpec
        ]  # конец списка
    )  # конец extend
    return suite  # весь учебный набор

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'TaskSpec', 'BOOL_NAMES', 'make_task_suite',  # имя интерфейса
]  # конец перечня
