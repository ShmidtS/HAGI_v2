"""Ядро HAGI в одном модуле (слито наследие версии 5–8 и каноническое тело версии 9).

Порядок исполнения сохраняет прежнюю семантику перекрытия имён: сначала
выполняется всё наследное тело, затем каноническое — для общих имён
побеждает более позднее определение, а код верхнего уровня версии 9 во
время исполнения модуля видит наследные версии вспомогательных функций,
точно так же, как прежде при цепочке star-импортов.
"""
"""
"""

#!/usr/bin/env python3
"""HAGI — единая сборка: ядро версии 5 + слои 7.4, 8.4, 9.0.

Для общих вспомогательных функций побеждает более позднее определение:
тело версии 9 содержит проверку опорных строк при точном построении
многочлена и другие уточнения; имена классов несут суффиксы, поэтому
обе линии определений сосуществуют.

Строка запуска: --self-test-core | --self-test-v7 | --self-test-v5 |
     --grow (наследный цикл роста) | --grow-deep | --benchmark-core [--max-rounds N] [--out-dir DIR]
"""
import argparse  # разбор аргументов командной строки
import copy  # глубокое копирование объектов
import itertools  # декартово произведение областей определения
import json  # чтение и запись JSON-состояния
import math  # математические функции (gcd, lcm, sqrt и т. д.)
import random  # генератор случайных чисел
import statistics  # среднее, медиана, дисперсия
import sys  # стандартный вывод и аргументы запуска
import time  # замер времени
from collections import Counter  # словарь-счётчик Frequency
from dataclasses import dataclass, field  # декоратор dataclass и поле со значением по умолчанию
from fractions import Fraction  # точные рациональные дроби
from functools import lru_cache  # запоминание результатов функции по аргументам
from pathlib import Path  # работа с путями файлов
from typing import Any, Callable, Iterable, Optional, Sequence  # аннотации типов

import numpy as np  # численные массивы и линейная алгебра

# ==============================================================================
# ЧАСТЬ I — ядро версии 5 (растущий математический процессор)
# ==============================================================================

"""HAGI — растущий математический процессор: ограниченный нейро-символьный прототип.

Это исследовательский прототип, а не утверждение, что любая математическая
функция будет выучена. По примерам от учителя он наращивает библиотеку
проверенных экспертов — операций над числами. Точная символьная программа
заносится в библиотеку только после проверки на объявленном конечном
множестве входов. Параллельно обучается небольшая нейронная «ворсинка» —
явно приближённый запасной путь. На входах вне области определения запрос
по умолчанию остаётся без ответа.

Зависимости: Python >=3.10, NumPy."""


VERSION = "5.0.0"  # номер версии ядра
V5_VERSION = VERSION  # тот же номер для наследного кода версии 5


# --- наследное тело версий 5–8 вынесено в модуль hagi.legacy_kernel ---
from hagi.legacy_kernel import *  # noqa: F401,F403  # импорт всех публичных имён наследного модуля
from hagi.legacy_kernel import (  # noqa: F401  # явный импорт приватных и констант, нужных телу модуля
    _exact_multivariate_poly_fit,  # точная многомерная полиномиальная подгонка
    _exact_poly_fit,  # точная полиномиальная подгонка по одному аргументу
    _fit_poly_or_const,  # подгонка многочлена или константы по точкам
    _monomial_exponents,  # все векторы показателей степени при заданной арности
    BINARY_OPS,  # кортеж имён бинарных операций
    COMMUTATIVE_OPS,  # множество коммутативных операций
    CONSTS,  # кортеж допустимых констант
    DYADIC_LEVELS,  # уровни (c+1)/128 для c = 0..254
    OPS,  # словарь базовых операций по имени
    TOKEN_BITS,  # число бит в токене
    VAR_NAMES,  # имена переменных
    _PRIME_MOD,  # простой модуль 2^31 − 1 для точной линейной фильтрации
    _bit_nand8,  # побитовое И-НЕ по младшим 8 битам
    _bit_nor8,  # побитовое ИЛИ-НЕ по младшим 8 битам
    _bit_not8,  # побитовое НЕ по младшим 8 битам
    _bit_xnor8,  # побитовое исключающее ИЛИ-НЕ по младшим 8 битам
    _pow_int,  # целая степень с показателем 0..8
    _sign,  # знак числа: −1, 0 или 1
)

# ==============================================================================
# 2. Описание задач и набор задач для проверки
# ==============================================================================


@dataclass  # поля класса автоматически становятся аргументами конструктора
class TaskSpec:
    """Описание учебной задачи: имя, арность, области аргументов и функция учителя.

    name — имя задачи; arity — число аргументов; domains — кортеж из arity
    допустимых значений по каждому аргументу; teacher — точная функция
    учителя f(x_1..x_arity); valid_rule — правило отбора допустимых входов
    ("all" — все, "nonzero_y" — второй аргумент не равен 0,
    "ordered_yz" — второй аргумент не больше третьего);
    family — семейство задач."""
    name: str  # имя задачи
    arity: int  # число аргументов функции
    domains: tuple[tuple[int, ...], ...]  # допустимые значения каждого аргумента
    teacher: Callable[..., int]  # функция учителя: по аргументам даёт верный ответ
    valid_rule: str = "all"  # правило допустимости входа; по умолчанию — любой вход
    family: str = "integer"  # семейство задач; по умолчанию «integer»

    def inputs(self) -> list[tuple[int, ...]]:
        """Все допустимые входы задачи: декартово произведение областей без строк, нарушающих valid_rule."""
        out = []  # накопитель допустимых входов
        for row in itertools.product(*self.domains):  # перебор всех строк декартова произведения D_1 × … × D_arity
            if self.valid_rule == "nonzero_y" and row[1] == 0:  # правило nonzero_y: пропустить строку, если x_2 = 0
                continue  # перейти к следующей строке
            if self.valid_rule == "ordered_yz" and row[1] > row[2]:  # правило ordered_yz: пропустить строку, если x_2 > x_3
                continue  # перейти к следующей строке
            out.append(tuple(int(x) for x in row))  # добавить строку, каждый элемент приведён к int
        return out  # список всех допустимых входов

    def valid(self, xs: Sequence[int]) -> bool:
        """Проверить, что xs — допустимый вход задачи: True или False.

        Условия: длина равна arity, каждый x_i лежит в domains[i], и выполнено
        valid_rule (nonzero_y: x_2 ≠ 0; ordered_yz: x_2 ≤ x_3)."""
        if len(xs) != self.arity:  # длина входа должна равняться арности
            return False  # иначе вход недопустим
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):  # проверка: существует ли i, для которого int(x_i) ∉ domains[i]
            return False  # есть аргумент вне своей области — вход недопустим
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:  # правило nonzero_y нарушено, если x_2 = 0
            return False  # иначе вход недопустим
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):  # правило ordered_yz нарушено, если x_2 > x_3
            return False  # иначе вход недопустим
        return True  # все условия выполнены — вход допустим


BOOL_NAMES = {  # имена всех 16 булевых функций двух аргументов по номеру таблицы истинности m
    0: "BOOL_FALSE",  # m = 0: f = 0
    1: "BOOL_NOR",  # m = 1: f = НЕ(x ИЛИ y)
    2: "BOOL_NOT_X_AND_Y",  # m = 2: f = (НЕ x) И y
    3: "BOOL_NOT_X",  # m = 3: f = НЕ x
    4: "BOOL_X_AND_NOT_Y",  # m = 4: f = x И (НЕ y)
    5: "BOOL_NOT_Y",  # m = 5: f = НЕ y
    6: "BOOL_XOR",  # m = 6: f = x XOR y
    7: "BOOL_NAND",  # m = 7: f = НЕ(x И y)
    8: "BOOL_AND",  # m = 8: f = x И y
    9: "BOOL_XNOR",  # m = 9: f = НЕ(x XOR y)
    10: "BOOL_Y",  # m = 10: f = y
    11: "BOOL_X_IMPLIES_Y",  # m = 11: f = x → y, то есть (НЕ x) ИЛИ y
    12: "BOOL_X",  # m = 12: f = x
    13: "BOOL_Y_IMPLIES_X",  # m = 13: f = y → x, то есть x ИЛИ (НЕ y)
    14: "BOOL_OR",  # m = 14: f = x ИЛИ y
    15: "BOOL_TRUE",  # m = 15: f = 1
}


def make_task_suite() -> list[TaskSpec]:
    """Собрать полный набор учебных задач: булевы, унарные, бинарные, трёхместные, физика и язык.

    Каждая задача — TaskSpec с точной функцией учителя. Возвращает список
    объектов TaskSpec."""
    suite: list[TaskSpec] = []  # пустой список задач
    bool_domain = ((0, 1), (0, 1))  # оба аргумента булевых задач принимают значения 0 и 1
    for mask, name in BOOL_NAMES.items():  # по парам (номер таблицы истинности m, имя функции)
        suite.append(  # добавить задачу
            TaskSpec(  # описание булевой задачи
                name,  # имя из таблицы
                2,  # арность 2
                bool_domain,  # области {0, 1} × {0, 1}
                lambda x, y, m=mask: (m >> ((int(x) << 1) | int(y))) & 1,  # учитель: f(x, y) = (m >> (2·x + y)) & 1 — бит m с номером 2·x + y
                family="BoolVenn",  # семейство — диаграммы Венна для булевых функций
            )
        )  # добавление TaskSpec в список
    U = tuple(range(-8, 9))  # U = {−8, …, 8} — область значений аргументов
    unary = [  # список (имя, функция) унарных задач
        ("NEG", lambda x: -x),  # NEG: f(x) = −x
        ("ABS", abs),  # ABS: f(x) = |x|
        ("SIGN", _sign),  # SIGN: знак x
        ("SQUARE", lambda x: x * x),  # SQUARE: f(x) = x²
        ("CUBE", lambda x: x * x * x),  # CUBE: f(x) = x³
        ("BIT_NOT8", _bit_not8),  # BIT_NOT8: побитовое НЕ по младшим 8 битам
        ("CLAMP_TO_3", lambda x: min(max(x, -3), 3)),  # CLAMP_TO_3: f(x) = min(max(x, −3), 3)
    ]
    for name, fn in unary:  # по каждой унарной задаче
        suite.append(TaskSpec(name, 1, (U,), fn, family="unary"))  # задача арности 1 с областью U
    binary = [  # список (имя, функция, правило допустимости) бинарных задач
        ("ADD", lambda x, y: x + y, "all"),  # ADD: x + y
        ("SUB", lambda x, y: x - y, "all"),  # SUB: x − y
        ("MUL", lambda x, y: x * y, "all"),  # MUL: x·y
        ("FLOORDIV", lambda x, y: x // y, "nonzero_y"),  # FLOORDIV: ⌊x / y⌋, y ≠ 0
        ("MOD", lambda x, y: x % y, "nonzero_y"),  # MOD: остаток x mod y, y ≠ 0
        ("GCD", math.gcd, "all"),  # GCD: наибольший общий делитель
        ("LCM", math.lcm, "all"),  # LCM: наименьшее общее кратное
        ("MIN", min, "all"),  # MIN: min(x, y)
        ("MAX", max, "all"),  # MAX: max(x, y)
        ("ABSDIFF", lambda x, y: abs(x - y), "all"),  # ABSDIFF: |x − y|
        ("EQ", lambda x, y: int(x == y), "all"),  # EQ: 1, если x = y, иначе 0
        ("NE", lambda x, y: int(x != y), "all"),  # NE: 1, если x ≠ y, иначе 0
        ("LT", lambda x, y: int(x < y), "all"),  # LT: 1, если x < y, иначе 0
        ("LE", lambda x, y: int(x <= y), "all"),  # LE: 1, если x ≤ y, иначе 0
        ("GT", lambda x, y: int(x > y), "all"),  # GT: 1, если x > y, иначе 0
        ("GE", lambda x, y: int(x >= y), "all"),  # GE: 1, если x ≥ y, иначе 0
        ("BIT_AND", lambda x, y: x & y, "all"),  # BIT_AND: x & y
        ("BIT_OR", lambda x, y: x | y, "all"),  # BIT_OR: x | y
        ("BIT_XOR", lambda x, y: x ^ y, "all"),  # BIT_XOR: x ^ y
        ("BIT_NAND8", _bit_nand8, "all"),  # BIT_NAND8: И-НЕ по младшим 8 битам
        ("BIT_NOR8", _bit_nor8, "all"),  # BIT_NOR8: ИЛИ-НЕ по младшим 8 битам
        ("BIT_XNOR8", _bit_xnor8, "all"),  # BIT_XNOR8: исключающее ИЛИ-НЕ по младшим 8 битам
    ]
    for name, fn, valid in binary:  # по каждой бинарной задаче
        suite.append(  # добавление TaskSpec в список
            TaskSpec(name, 2, (U, U), fn, valid_rule=valid, family="integer-binary")  # задача арности 2 с областью U × U и своим правилом допустимости
        )
    P = tuple(range(-3, 4))  # P = {−3, …, 3} — основание степени
    E = tuple(range(0, 5))  # E = {0, …, 4} — показатель степени
    suite.append(TaskSpec("POW_INT", 2, (P, E), _pow_int, family="integer-binary"))  # POW_INT: f(x, y) = x^y на области P × E
    Z = tuple(range(-8, 9))  # Z = {−8, …, 8} — область значений
    F = tuple(range(-4, 5))  # F = {−4, …, 4} — область значений
    suite.extend(  # добавить три трёхместные задачи
        [  # начало списка задач
            TaskSpec(  # задача FMA
                "FMA",  # имя FMA
                3,  # арность 3
                (Z, Z, Z),  # область Z × Z × Z
                lambda x, y, z: x * y + z,  # учитель: f(x, y, z) = x·y + z
                family="composed-ternary",  # семейство составных трёхместных задач
            ),
            TaskSpec(  # задача CLAMP
                "CLAMP",  # имя CLAMP
                3,  # арность 3
                (Z, F, F),  # область Z × F × F: значение и границы
                lambda x, lo, hi: min(max(x, lo), hi),  # учитель: f(x, lo, hi) = min(max(x, lo), hi)
                valid_rule="ordered_yz",  # границы упорядочены: lo ≤ hi
                family="composed-ternary",  # семейство составных трёхместных задач
            ),
            TaskSpec(  # задача MEDIAN3
                "MEDIAN3",  # имя MEDIAN3
                3,  # арность 3
                (F, F, F),  # область F × F × F
                lambda x, y, z: sorted((x, y, z))[1],  # медиана — средний элемент отсортированного кортежа (x, y, z)
                family="composed-ternary",  # семейство составных трёхместных задач
            ),
        ]
    )  # добавить список к набору задач
    # --- физика: точные целочисленные законы ---
    T2 = tuple(range(0, 3))  # T2 = {0, 1, 2} (область времени; здесь не используется)
    suite.extend(  # добавить четыре физические задачи
        [  # начало списка задач
            TaskSpec(  # задача свободного падения
                "PHYS_FREE_FALL",  # имя PHYS_FREE_FALL
                1,  # арность 1
                (tuple(range(0, 5)),),  # время t ∈ {0, …, 4}
                lambda t: 20 - 5 * t * t,  # высота h(t) = 20 − 5·t²
                family="physics",  # семейство физических задач
            ),
            TaskSpec(  # задача равномерного движения
                "PHYS_UNIFORM_MOTION",  # имя PHYS_UNIFORM_MOTION
                1,  # арность 1
                (tuple(range(-3, 4)),),  # время t ∈ {−3, …, 3}
                lambda t: 3 + 2 * t,  # координата s(t) = 3 + 2·t
                family="physics",  # семейство физических задач
            ),
            TaskSpec(  # задача энергии пружины
                "PHYS_SPRING_ENERGY",  # имя PHYS_SPRING_ENERGY
                1,  # арность 1
                (tuple(range(-3, 4)),),  # смещение x ∈ {−3, …, 3}
                lambda x: 3 * x * x,  # энергия E(x) = 3·x²
                family="physics",  # семейство физических задач
            ),
            TaskSpec(  # задача импульса
                "PHYS_MOMENTUM",  # имя PHYS_MOMENTUM
                2,  # арность 2
                (tuple(range(0, 5)), tuple(range(-4, 5))),  # масса m ∈ {0, …, 4}, скорость v ∈ {−4, …, 4}
                lambda m, v: m * v,  # импульс p = m·v
                family="physics",  # семейство физических задач
            ),
        ]
    )  # добавить список к набору задач
    # --- язык: буквы латинского алфавита закодированы числами 0..25 ---
    A = tuple(range(0, 26))  # A = {0, …, 25} — код буквы
    K = tuple(range(0, 26))  # K = {0, …, 25} — ключ сдвига
    suite.extend(  # добавить четыре языковые задачи
        [  # начало списка задач
            TaskSpec(  # задача шифра Цезаря со сдвигом 3
                "LANG_CAESAR3", 1, (A,), lambda c: (c + 3) % 26, family="language"  # арность 1; f(c) = (c + 3) mod 26
            ),  # семейство языковых задач
            TaskSpec("LANG_ATBASH", 1, (A,), lambda c: 25 - c, family="language"),  # LANG_ATBASH: f(c) = 25 − c (буква заменяется на симметричную)
            TaskSpec(  # задача «гласная ли буква»
                "LANG_IS_VOWEL",  # имя LANG_IS_VOWEL
                1,  # арность 1
                (A,),  # код буквы c ∈ {0, …, 25}
                lambda c: int(c in (0, 4, 8, 14, 20)),  # 1, если c — код гласной a, e, i, o, u (0, 4, 8, 14, 20), иначе 0
                family="language",  # семейство языковых задач
            ),
            TaskSpec(  # задача сдвига по ключу
                "LANG_SHIFT", 2, (A, K), lambda c, k: (c + k) % 26, family="language"  # арность 2; f(c, k) = (c + k) mod 26
            ),
        ]  # добавить список к набору задач
    )
    # --- весь пул байтов 0..255: любой текст есть последовательность байтов ---
    B = tuple(range(0, 256))  # B = {0, …, 255} — код байта
    BK = tuple(range(0, 256))  # BK = {0, …, 255} — ключ
    suite.extend(  # добавить четыре байтовые языковые задачи
        [  # начало списка задач
            TaskSpec("LANG8_ATBASH", 1, (B,), lambda c: 255 - c, family="language"),  # LANG8_ATBASH: f(c) = 255 − c
            TaskSpec(  # задача циклического сдвига байта на 1
                "LANG8_CAESAR1", 1, (B,), lambda c: (c + 1) % 256, family="language"  # арность 1; f(c) = (c + 1) mod 256
            ),
            TaskSpec(  # задача сдвига байта по ключу
                "LANG8_SHIFT", 2, (B, BK), lambda c, k: (c + k) % 256, family="language"  # арность 2; f(c, k) = (c + k) mod 256
            ),
            TaskSpec(  # задача шифра исключающим ИЛИ
                "LANG8_XOR_CIPHER", 2, (B, BK), lambda c, k: c ^ k, family="language"  # арность 2; f(c, k) = c XOR k
            ),
        ]
    )  # добавить список к набору задач
    # --- кириллические кодовые точки U+0400..U+04FF ---
    RU_VOWELS = set(map(ord, "аеёиоуыэюяАЕЁИОУЫЭЮЯ"))  # коды всех кириллических гласных (строчных и прописных)
    RU = tuple(range(0x400, 0x500))  # RU = {0x400, …, 0x4FF} — область кириллических кодовых точек
    suite.extend(  # добавить одну задачу
        [  # начало списка задач
            TaskSpec(  # задача «кириллическая гласная»
                "LANG_RU_VOWEL",  # имя LANG_RU_VOWEL
                1,  # арность 1
                (RU,),  # код c ∈ RU
                lambda c: int(c in RU_VOWELS),  # 1, если c — код гласной буквы, иначе 0
                family="language",  # семейство языковых задач
            ),
        ]
    )  # добавить список к набору задач
    return suite  # готовый набор задач


# ==============================================================================
# 3. Дерево выражения, программа и векторизованный вычислитель
# ==============================================================================

# Реестр операторов, построенных методом переификации: изобретённые координаты
# становятся полноправными унарными и бинарными макрооперациями внутри
# выражений Expr. Заполняется в hagi.representations функцией reify_construct();
REIFIED_OPS: dict = {}  # реестр макрооператоров: имя -> функция


@dataclass(frozen=True)  # неизменяемый класс: экземпляры нельзя менять после создания
class Expr:
    """Узел дерева выражения (направленный ациклический граф вызовов): операция и её аргументы.

    op — имя операции; args — кортеж подвыражений-аргументов; value —
    дополнительное значение: индекс переменной, константа, имя вызываемого
    эксперта, коэффициенты многочлена, множество или список диапазонов."""
    op: str  # имя операции
    args: tuple["Expr", ...] = ()  # подвыражения-аргументы; по умолчанию пусто
    value: Optional[Any] = None  # дополнительное значение узла; по умолчанию нет

    @staticmethod  # метод без доступа к экземпляру
    def var(i: int) -> "Expr":
        """Построить узел переменной: i-й аргумент входа."""
        return Expr("var", (), int(i))  # узел var с индексом i

    @staticmethod  # метод без доступа к экземпляру
    def const(v: int) -> "Expr":
        """Построить узел константы со значением v."""
        return Expr("const", (), int(v))  # узел const со значением v

    def eval(
        self, xs: Sequence[int], env: Optional[dict[str, "Expert"]] = None
    ) -> int:  # возвращает целое значение
        """Вычислить выражение на входе xs: значение int.

        xs — входные аргументы; env — словарь имени эксперта -> объект
        эксперта для операции call (необязателен). Операции:
        var: xs[value]; const: value; call: программа вызываемого эксперта
        на значениях аргументов; anf: XOR коэффициентов c мономов s ⊆ mask_x,
        где mask_x = Σ b_i·2^(n−1−i); if_cond: вторая ветвь, если условие
        истинно, иначе третья; poly: схема Горнера acc = acc·x + c_i по
        коэффициентам от старшего к младшему; poly_nd: Σ coef·Π x_j^p_j;
        далее базовые операции и операторы из REIFIED_OPS и OPS."""
        if self.op == "var":  # операция — переменная
            return int(xs[int(self.value)])  # взять аргумент xs[value]
        if self.op == "const":  # операция — константа
            return int(self.value)  # вернуть значение value
        if self.op == "call":  # операция — вызов другого эксперта
            if env is None or self.value not in env:  # нет среды или вызываемый эксперт в ней не зарегистрирован
                raise ValueError(f"missing called expert: {self.value}")  # ошибка: эксперт отсутствует
            vals = tuple(a.eval(xs, env) for a in self.args)  # значения аргументов вызова (рекурсивно)
            callee = env[self.value]  # вызываемый эксперт по имени
            if not callee.supports(vals):  # вход вне проверенной области применимости эксперта
                raise ValueError(  # ошибка: вызов вне сертифицированной области
                    f"call to {self.value} outside certified support: {vals}"  # текст ошибки с именем и аргументами
                )
            return callee.program.eval(vals, env)  # результат программы вызываемого эксперта на значениях аргументов
        if self.op == "anf":  # операция — алгебраическая нормальная форма (АНФ)
            n_vars, coeffs = self.value  # число переменных n и список пар (моном s, коэффициент c)
            mask_x = 0  # битовая маска входа, изначально 0
            for i in range(n_vars):  # по номерам переменных i = 0..n−1
                b = int(xs[i])  # значение i-й переменной
                if b not in (0, 1):  # вход должен быть булевым
                    raise ValueError("anf expects boolean inputs")  # ошибка: АНФ принимает только булевы входы
                if b:  # если b_i = 1
                    mask_x |= 1 << (n_vars - 1 - i)  # установить бит n−1−i: mask_x |= 2^(n−1−i)
            acc = 0  # аккумулятор результата, изначально 0
            for s, c in coeffs:  # по парам (моном s, коэффициент c)
                if (s & mask_x) == s:  # моном входит в вход: s AND mask_x = s
                    acc ^= c  # аккумулятор XOR коэффициент
            return int(acc)  # значение АНФ
        if self.op == "if_cond":  # операция — условное выражение
            cond_val = self.args[0].eval(xs, env)  # значение условия (первый аргумент)
            return (  # вернуть вторую ветвь, если условие истинно, иначе третью
                self.args[1].eval(xs, env) if cond_val else self.args[2].eval(xs, env)  # ветвь «то» args[1] при cond_val ≠ 0, иначе ветвь «иначе» args[2]
            )
        vals = [a.eval(xs, env) for a in self.args]  # значения всех аргументов (рекурсивно)
        if self.op == "poly":  # операция — многочлен одного аргумента
            acc = 0  # аккумулятор схемы Горнера, изначально 0
            for ci in reversed(list(self.value or ())):  # по коэффициентам от старшего к младшему (пустой список, если value нет)
                acc = acc * vals[0] + int(ci)  # шаг Горнера: acc = acc·x + c_i
            return int(acc)  # значение многочлена
        if self.op == "poly_nd":  # операция — многочлен многих переменных
            total = 0  # сумма членов, изначально 0
            for coef, powers in self.value or ():  # по парам (коэффициент, показатели степени)
                term = int(coef)  # член начинается с коэффициента
                for x_val, p in zip(xs, powers):  # по парам (переменная x_j, показатель p_j)
                    term *= int(x_val) ** int(p)  # умножить член: term = term·x_j^p_j
                total += term  # прибавить член: total = total + term
            return int(total)  # сумма членов total
        if self.op == "absdiff":  # операция — абсолютная разность
            return abs(vals[0] - vals[1])  # вернуть |a_0 − a_1|
        if self.op == "fma":  # операция — умножение со сложением
            return vals[0] * vals[1] + vals[2]  # вернуть a_0·a_1 + a_2
        if self.op == "clamp":  # операция — ограничение диапазоном
            return min(max(vals[0], vals[1]), vals[2])  # вернуть min(max(a_0, a_1), a_2)
        if self.op == "median3":  # операция — медиана трёх
            return sorted(vals)[1]  # средний элемент отсортированных аргументов
        if self.op == "clamp3":  # операция — ограничение отрезком [−3, 3]
            return min(max(vals[0], -3), 3)  # вернуть min(max(a_0, −3), 3)
        if self.op == "mul3":  # операция — произведение трёх
            return vals[0] * vals[1] * vals[2]  # вернуть a_0·a_1·a_2
        if self.op == "in_set":  # операция — принадлежность конечному множеству
            # значение a_0 должно входить в строго возрастающее конечное множество;
            # разреженное запоминание ответов разрешено только для задач
            # языкового семейства (проверяется при занесении в библиотеку),
            # но не как общий способ находить законы.
            return int(vals[0] in set(self.value or ()))  # 1, если a_0 ∈ value, иначе 0
        if self.op == "in_ranges":  # операция — принадлежность диапазонам
            # включающая принадлежность отсортированным непересекающимся
            # несмежным отрезкам — парам (lo, hi).
            v = vals[0]  # проверяемое значение v = a_0
            return int(any(lo <= v <= hi for lo, hi in (self.value or ())))  # 1, если найдётся отрезок с lo ≤ v ≤ hi, иначе 0
        if self.op == "construct":  # операция — конструкция из реестра
            from hagi.representations import construct_eval  # вычислитель конструкций импортируется здесь, чтобы избежать циклического импорта

            return construct_eval(self.value, xs, env)  # результат конструкции на входе xs в среде env
        fn = REIFIED_OPS.get(self.op)  # функция макрооператора из реестра, если есть
        if fn is not None:  # оператор найден в реестре
            return int(fn(*vals))  # применить его к значениям аргументов
        fn = OPS.get(self.op)  # иначе — базовая операция из OPS
        if fn is None:  # операция неизвестна
            raise ValueError(f"unsupported op: {self.op}")  # ошибка: операция не поддерживается
        return int(fn(*vals))  # применить базовую операцию к значениям аргументов

    @property  # сделать метод свойством: вызывается без скобок
    def cost(self) -> int:
        """Стоимость выражения cost: число узлов дерева (1 + сумма стоимостей аргументов).

        anf: 1 + число мономов; poly и poly_nd: 1 + число коэффициентов
        (членов); construct: 1 + число бит конструкции. Результат запоминается
        в атрибуте _cost_memo, поэтому повторный вызов не пересчитывает."""
        c = self.__dict__.get("_cost_memo")  # взять запомненную стоимость, если она уже вычислялась
        if c is None:  # стоимость ещё не вычислена
            if self.op in ("var", "const"):  # переменная или константа
                c = 1  # лист дерева имеет стоимость 1
            elif self.op == "anf":  # алгебраическая нормальная форма
                _, coeffs = self.value  # извлечь список пар (моном, коэффициент); число переменных не нужно
                c = 1 + len(coeffs)  # стоимость = 1 + число мономов
            elif self.op in ("poly", "poly_nd"):  # многочлен одного или многих аргументов
                c = 1 + len(self.value or ())  # стоимость = 1 + число коэффициентов (0, если value пусто)
            elif self.op == "construct":  # конструкция из реестра
                from hagi.representations import construct_bits as _cb  # импортировать здесь оценку числа бит конструкции, чтобы избежать циклического импорта

                c = 1 + _cb(self.value)  # стоимость = 1 + битовая длина конструкции
            else:  # любая другая операция
                c = 1 + sum(a.cost for a in self.args)  # стоимость = 1 + Σ cost(a_i) по аргументам
            object.__setattr__(self, "_cost_memo", c)  # запомнить стоимость, обойдя запрет изменения замороженного объекта
        return c  # вернуть стоимость выражения

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:
        """Число бит описания выражения: код токена плюс полезная нагрузка и биты аргументов.

        expert_index — словарь «имя эксперта -> номер», нужный операции call.
        Везде TOKEN_BITS — цена кода операции; gamma_bits(n) = 2·(⌊log2 n⌋) + 1
        — код Элиаса-гамма числа n ≥ 1; zigzag(v) = 2v при v ≥ 0 и −2v − 1 при
        v < 0 — перевод целого в неотрицательное число."""
        if self.op == "var":  # переменная
            return TOKEN_BITS + gamma_bits(int(self.value or 0) + 1)  # var: TOKEN_BITS + gamma_bits(индекс + 1)
        if self.op == "const":  # константа
            return TOKEN_BITS + gamma_bits(zigzag(int(self.value)) + 1)  # const: TOKEN_BITS + gamma_bits(zigzag(значение) + 1)
        if self.op == "call":  # вызов другого эксперта
            idx = expert_index.get(str(self.value), 0) if expert_index else 0  # номер вызываемого эксперта (0, если словарь не задан или имени нет)
            return (  # итог: call: TOKEN_BITS
                TOKEN_BITS  # плюс код токена операции
                + gamma_bits(idx + 1)  # плюс gamma_bits(номер эксперта + 1)
                + sum(a.bits(expert_index) for a in self.args)  # плюс сумма бит аргументов
            )
        # та же формула дельта-кодирования множества и диапазонов, что и в
        # hagi.certificates._expr_bits: без неё стоимость запоминания ответов
        # считается заниженной.
        if self.op == "in_set":  # принадлежность конечному множеству
            pts = [int(v) for v in (self.value or ())]  # список точек множества как int
            if not pts:  # множество пусто
                return TOKEN_BITS  # пустое множество стоит только кода токена
            pay = (gamma_bits(len(pts) + 1)  # нагрузка = gamma_bits(число точек + 1)
                   + gamma_bits(zigzag(pts[0]) + 1))  # плюс gamma_bits(zigzag(первая точка) + 1)
            pay += sum(gamma_bits(b - a) for a, b in zip(pts, pts[1:]))  # плюс Σ gamma_bits(p_{k+1} − p_k) по соседним точкам — дельта-кодирование
            return TOKEN_BITS + pay + sum(a.bits(expert_index)  # вернуть TOKEN_BITS + нагрузка + биты аргументов
                                          for a in self.args)  # сумма бит аргументов
        if self.op == "in_ranges":  # принадлежность диапазонам
            rs = [(int(lo), int(hi)) for lo, hi in (self.value or ())]  # список пар (lo, hi) как int
            if not rs:  # список пуст
                return TOKEN_BITS  # нет диапазонов — только код токена
            pay = (gamma_bits(len(rs) + 1)  # нагрузка = gamma_bits(число диапазонов + 1)
                   + gamma_bits(zigzag(rs[0][0]) + 1)  # плюс gamma_bits(zigzag(lo первого диапазона) + 1)
                   + gamma_bits(rs[0][1] - rs[0][0] + 1))  # плюс gamma_bits(длина первого диапазона hi − lo + 1)
            for pr, c in zip(rs, rs[1:]):  # по парам (предыдущий pr, текущий c) диапазонов
                pay += (gamma_bits(c[0] - pr[1])  # прибавить gamma_bits(промежуток c.lo − pr.hi)
                        + gamma_bits(c[1] - c[0] + 1))  # плюс gamma_bits(длина c.hi − c.lo + 1)
            return TOKEN_BITS + pay + sum(a.bits(expert_index)  # вернуть TOKEN_BITS + нагрузка + биты аргументов
                                          for a in self.args)  # сумма бит аргументов
        if self.op == "anf":  # алгебраическая нормальная форма
            n_vars, coeffs = self.value  # число переменных и список пар (моном s, коэффициент c)
            payload = gamma_bits(n_vars + 1) + gamma_bits(len(coeffs) + 1)  # нагрузка = gamma_bits(n + 1) + gamma_bits(число мономов + 1)
            payload += sum(gamma_bits(s + 1) for s, _ in coeffs)  # плюс Σ gamma_bits(s + 1) по мономам
            return TOKEN_BITS + payload  # вернуть TOKEN_BITS + нагрузка
        if self.op == "poly":  # многочлен одного аргумента
            coeffs = tuple(int(v) for v in (self.value or ()))  # кортеж коэффициентов как int
            payload = gamma_bits(len(coeffs) + 1) + sum(  # нагрузка = gamma_bits(число коэффициентов + 1) + сумма
                gamma_bits(zigzag(v) + 1) for v in coeffs  # Σ gamma_bits(zigzag(v) + 1) по коэффициентам v
            )
            return TOKEN_BITS + payload + sum(a.bits(expert_index) for a in self.args)  # вернуть TOKEN_BITS + нагрузка + сумма бит аргументов
        if self.op == "poly_nd":  # многочлен многих переменных
            terms = tuple(self.value or ())  # кортеж членов (коэффициент, показатели степени)
            payload = gamma_bits(len(terms) + 1)  # нагрузка = gamma_bits(число членов + 1)
            for c, powers in terms:  # по членам (коэффициент c, показатели powers)
                payload += gamma_bits(zigzag(int(c)) + 1) + gamma_bits(len(powers) + 1)  # прибавить gamma_bits(zigzag(c) + 1) + gamma_bits(число показателей + 1)
                payload += sum(gamma_bits(int(p) + 1) for p in powers)  # прибавить Σ gamma_bits(p + 1) по показателям p
            return TOKEN_BITS + payload  # вернуть TOKEN_BITS + нагрузка
        return TOKEN_BITS + sum(a.bits(expert_index) for a in self.args)  # прочие операции: TOKEN_BITS + сумма бит аргументов

    def callees(self) -> set[str]:
        """Множество имён экспертов, вызываемых в этом выражении и его подвыражениях."""
        out: set[str] = set()  # накопитель имён
        if self.op == "call" and self.value is not None:  # узел — вызов и имя задано
            out.add(str(self.value))  # добавить имя вызываемого эксперта
        for a in self.args:  # по аргументам
            out |= a.callees()  # объединить с именами аргумента
        return out  # вернуть множество имён

    def substitute(self, mapping: dict[int, "Expr"]) -> "Expr":
        """Подставить в выражение новые подвыражения вместо переменных: новое выражение Expr.

        mapping — словарь «индекс переменной -> выражение»; переменные, которых
        нет в словаре, и узлы без аргументов остаются как есть."""
        if self.op == "var":  # переменная
            idx = int(self.value or 0)  # индекс переменной
            return mapping.get(idx, self)  # вернуть замену из словаря или сам узел, если замены нет
        if not self.args:  # узел без аргументов
            return self  # не меняется
        return Expr(  # собрать новый узел
            self.op, tuple(a.substitute(mapping) for a in self.args), self.value  # с теми же операцией и значением, но с рекурсивно заменёнными аргументами
        )

    def subtrees(self) -> list["Expr"]:
        """Список всех поддеревьев выражения, включая само выражение (корень первым)."""
        out = [self]  # начать с самого узла
        for a in self.args:  # по аргументам
            out.extend(a.subtrees())  # добавить все их поддеревья
        return out  # список поддеревьев

    def replace_subtree(self, target: "Expr", replacement: "Expr") -> "Expr":
        """Заменить в дереве поддерево target на replacement: новое выражение Expr.

        Сравнение — по равенству значений (узлы заморожены и сравнимы как
        кортежи полей)."""
        if self == target:  # текущий узел совпадает с искомым
            return replacement  # вернуть замену
        if not self.args:  # узел без аргументов
            return self  # остаётся как есть
        return Expr(  # собрать новый узел
            self.op,  # с той же операцией
            tuple(a.replace_subtree(target, replacement) for a in self.args),  # с рекурсивно обработанными аргументами
            self.value,  # и тем же значением
        )

    def pretty(self) -> str:
        """Читаемая запись выражения строкой: имена переменных, знаки операций, скобки."""
        if self.op == "var":  # переменная
            idx = int(self.value or 0)  # индекс переменной
            return VAR_NAMES[idx] if idx < len(VAR_NAMES) else f"x{idx}"  # имя из VAR_NAMES, если индекс в пределах списка, иначе x<индекс>
        if self.op == "const":  # константа
            return str(self.value)  # запись числа
        if self.op == "call":  # вызов эксперта
            return f"{self.value}({', '.join(a.pretty() for a in self.args)})"  # «имя(аргумент через запятую)»
        if self.op == "anf":  # алгебраическая нормальная форма
            n_vars, coeffs = self.value  # число переменных и список мономов с коэффициентами
            terms = []  # список текстов мономов
            for s, _ in coeffs:  # по мономам s (коэффициент не входит в текст)
                if s == 0:  # пустой моном
                    terms.append("1")  # записывается как константа 1
                else:  # иначе
                    vs = [  # список имён переменных монома
                        VAR_NAMES[i] if i < len(VAR_NAMES) else f"x{i}"  # имя переменной i, если i в пределах VAR_NAMES, иначе x<i>
                        for i in range(n_vars)  # по i = 0..n−1
                        if (s >> (n_vars - 1 - i)) & 1  # брать только переменные, чей бит (s >> (n−1−i)) & 1 установлен
                    ]  # закрытие списка имён
                    terms.append("&".join(vs))  # моном — имена переменных через «&»
            return "anf(" + (" ^ ".join(terms) if terms else "0") + ")"  # итог: «anf(мономы через ^)», а при пустом списке «anf(0)»
        if self.op == "if_cond":  # условное выражение
            return f"({self.args[1].pretty()} if {self.args[0].pretty()} else {self.args[2].pretty()})"  # «(то если условие иначе иначе)»
        if self.op == "poly":  # многочлен одного аргумента
            coeffs = tuple(self.value or ())  # кортеж коэффициентов от младшего к старшему
            terms = []  # список текстов членов
            for i in range(len(coeffs) - 1, -1, -1):  # по степеням i от старшей к младшей
                c = int(coeffs[i])  # коэффициент при x^i
                if c == 0:  # нулевой коэффициент
                    continue  # не показывать
                mag = abs(c)  # модуль коэффициента
                body = "" if i == 0 else ("x" if i == 1 else f"x**{i}")  # множитель x^i: пусто при i = 0, x при i = 1, иначе x**i
                term = str(mag) if i == 0 else (body if mag == 1 else f"{mag}*{body}")  # текст члена: модуль при i = 0, иначе множитель или «модуль*множитель»
                if not terms:  # первый член
                    terms.append(("-" if c < 0 else "") + term)  # знак минус перед ним ставится, только если коэффициент отрицателен
                else:  # последующие члены
                    terms.append((" - " if c < 0 else " + ") + term)  # перед ними « - » или « + » по знаку коэффициента
            return "poly(" + ("".join(terms) if terms else "0") + ")"  # итог: «poly(члены)», а при пустом списке «poly(0)»
        if self.op == "poly_nd":  # многочлен многих переменных
            terms = []  # список текстов членов
            for coef, powers in sorted(  # по членам, отсортированным по убыванию полной степени и лексикографически
                (self.value or ()), key=lambda it: (sum(it[1]), it[1]), reverse=True  # ключ сортировки: (сумма показателей, вектор показателей)
            ):
                c = int(coef)  # коэффициент члена
                if c == 0:  # нулевой коэффициент
                    continue  # не показывать
                factors = []  # список множителей x_j^p_j
                for idx, p in enumerate(powers):  # по номерам переменных idx и показателям p
                    p = int(p)  # показатель как int
                    if p == 0:  # показатель 0
                        continue  # переменная не входит в член
                    nm = VAR_NAMES[idx] if idx < len(VAR_NAMES) else f"x{idx}"  # имя переменной
                    factors.append(f"{nm}^{p}" if p != 1 else nm)  # множитель «имя^p» при p ≠ 1, иначе просто имя
                body = "*".join(factors) if factors else str(abs(c))  # тело члена: произведение множителей, а без множителей — модуль коэффициента
                if factors and abs(c) != 1:  # есть множители и модуль коэффициента не равен 1
                    body = f"{abs(c)}*{body}"  # дописать модуль коэффициента перед произведением
                if not terms:  # первый член
                    terms.append(("-" if c < 0 else "") + body)  # минус перед ним, только если коэффициент отрицателен
                else:  # последующие члены
                    terms.append((" - " if c < 0 else " + ") + body)  # перед ними « - » или « + » по знаку коэффициента
            return "poly_nd(" + ("".join(terms) if terms else "0") + ")"  # итог: «poly_nd(члены)», а при пустом списке «poly_nd(0)»
        if self.op == "clamp3":  # ограничение отрезком [−3, 3]
            return f"min(max({self.args[0].pretty()}, -3), 3)"  # «min(max(аргумент, -3), 3)»
        if self.op == "absdiff":  # абсолютная разность
            return f"abs({self.args[0].pretty()} - {self.args[1].pretty()})"  # «abs(первый − второй)»
        if self.op == "fma":  # умножение со сложением
            return f"({self.args[0].pretty()} * {self.args[1].pretty()} + {self.args[2].pretty()})"  # «(первый * второй + третий)»
        if self.op == "clamp":  # ограничение диапазоном
            return f"min(max({self.args[0].pretty()}, {self.args[1].pretty()}), {self.args[2].pretty()})"  # «min(max(значение, нижняя граница), верхняя граница)»
        if self.op == "median3":  # медиана трёх
            return f"median({', '.join(a.pretty() for a in self.args)})"  # «median(аргументы через запятую)»
        if self.op == "mul3":  # произведение трёх
            return " * ".join(a.pretty() for a in self.args)  # аргументы через « * »
        if self.op == "in_set":  # принадлежность конечному множеству
            return f"in_set({sorted(set(self.value or ()))})"  # «in_set(отсортированное множество)»
        if self.op == "in_ranges":  # принадлежность диапазонам
            return f"in_ranges({self.value})"  # «in_ranges(список диапазонов)»
        if len(self.args) == 1:  # один аргумент
            return f"{self.op}({self.args[0].pretty()})"  # «операция(аргумент)»
        if len(self.args) == 2:  # два аргумента
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"  # «(первый операция второй)» — инфиксная запись
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"  # иначе «операция(все аргументы через запятую)»

    def to_dict(self) -> dict[str, Any]:
        """Сериализация выражения в словарь для записи в JSON."""
        val = self.value  # значение узла
        if self.op == "anf" and val is not None:  # АЛН со значением
            val = [int(val[0]), [[int(s), int(c)] for s, c in val[1]]]  # value = [n, [[s, c], …]] — списки вместо кортежей
        elif self.op == "construct" and isinstance(val, dict):  # конструкция, заданная словарём
            from hagi.representations import construct_to_dict_safe  # импортируется здесь, чтобы избежать циклического импорта

            val = construct_to_dict_safe(val)  # преобразовать словарь конструкции в безопасный для JSON вид
        return {"op": self.op, "value": val, "args": [a.to_dict() for a in self.args]}  # словарь «имя операции, значение, список аргументов»

    @staticmethod  # метод без доступа к экземпляру — восстановление
    def from_dict(d: dict[str, Any]) -> "Expr":  # имя операции
        """Восстановить выражение Expr из словаря, полученного to_dict."""
        op = str(d["op"])  # значение узла или None
        val = d.get("value")  # многочлен одного аргумента
        if op == "poly" and val is not None:  # коэффициенты — кортеж int
            val = tuple(int(v) for v in val)  # многочлен многих переменных
        elif op == "poly_nd" and val is not None:  # value = ((коэффициент, (показатели)), …) — кортежи int
            val = tuple((int(t[0]), tuple(int(p) for p in t[1])) for t in val)  # АЛН
        elif op == "anf" and val is not None:  # value = (n, ((моном, коэффициент), …)) — кортежи int
            val = (int(val[0]), tuple((int(s), int(c)) for s, c in val[1]))  # конструкция, заданная словарём
        elif op == "construct" and isinstance(val, dict):  # импортируется здесь, чтобы избежать циклического импорта
            from hagi.representations import construct_from_dict_safe  # импортируется здесь, чтобы избежать циклического импорта
  # восстановить значение конструкции из словаря
            val = construct_from_dict_safe(val)  # новый узел с рекурсивно восстановленными аргументами
        return Expr(op, tuple(Expr.from_dict(a) for a in d.get("args", [])), val)


@dataclass(frozen=True)  # неизменяемый класс
class Program:
    """Программа эксперта: либо выражение Expr, либо булева диаграмма Венна по маске.

    arity — число аргументов; expr — дерево выражения (для обычных программ);
    venn_mask — 4-битная таблица истинности булевой функции двух аргументов
    f(x, y) = (venn_mask >> (2·x + y)) & 1."""
    arity: int  # число аргументов
    expr: Optional[Expr] = None  # дерево выражения; по умолчанию нет
    venn_mask: Optional[int] = None  # маска булевой функции; по умолчанию нет

    @property  # сделать метод свойством
    def cost(self) -> int:
        """Стоимость программы cost: 4 для диаграммы Венна, иначе стоимость выражения."""
        return (  # вернуть
            4  # вернуть стоимость: 4,
            if self.venn_mask is not None  # если задана маска Венна,
            else (self.expr.cost if self.expr else 10**9)  # иначе cost выражения, а без выражения — 10^9 (бесконечно дорого)
        )

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:
        """Число бит описания программы: TOKEN_BITS + 4 для Венна, иначе биты выражения."""
        if self.venn_mask is not None:  # задана маска Венна
            return TOKEN_BITS + 4  # TOKEN_BITS + 4 бита маски
        if self.expr is None:  # выражения нет
            return 10**9  # 10^9 — бесконечно большое описание
        return self.expr.bits(expert_index)  # биты выражения

    def eval(
        self, xs: Sequence[int], env: Optional[dict[str, "Expert"]] = None
    ) -> int:  # возвращает значение int
        """Вычислить программу на входе xs: значение int или ошибка.

        Проверяет число аргументов; для Венна — два булевых входа; иначе
        вычисляет выражение. env — среда вызываемых экспертов."""
        if len(xs) != self.arity:  # число аргументов должно равняться арности
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")  # ошибка о неверном числе входов
        if self.venn_mask is not None:  # программа — диаграмма Венна
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):  # Венн требует арность 2 и значения x, y ∈ {0, 1}
                raise ValueError("BoolVenn expects two binary inputs")  # ошибка: ожидались два булевых входа
            return (int(self.venn_mask) >> ((int(xs[0]) << 1) | int(xs[1]))) & 1  # f(x, y) = (маска >> (2·x + y)) & 1
        if self.expr is None:  # выражения нет
            raise RuntimeError("empty program")  # ошибка: пустая программа
        return self.expr.eval(xs, env)  # значение выражения на входе

    def pretty(self) -> str:
        """Читаемая запись программы строкой."""
        if self.venn_mask is not None:  # диаграмма Венна
            return f"BoolVenn(mask={self.venn_mask:04b})"  # «BoolVenn(mask=4 бита маски)»
        assert self.expr is not None  # далее выражение обязано существовать
        return self.expr.pretty()  # читаемая запись выражения

    def to_dict(self) -> dict[str, Any]:
        """Сериализация программы в словарь для JSON: арность, маска, выражение."""
        return {  # словарь
            "arity": self.arity,  # арность
            "venn_mask": self.venn_mask,  # маска Венна или None
            "expr": self.expr.to_dict() if self.expr else None,  # словарь выражения или None
        }

    @staticmethod  # метод без доступа к экземпляру
    def from_dict(d: dict[str, Any]) -> "Program":
        """Восстановить программу Program из словаря, полученного to_dict."""
        return Program(  # новая программа
            int(d["arity"]),  # с арностью
            Expr.from_dict(d["expr"]) if d.get("expr") else None,  # выражением из словаря (или None, если его нет)
            d.get("venn_mask"),  # и маской Венна (или None)
        )


_VEC_MAG_LIMIT = 1 << 28  # предел модуля числа для векторизованных вычислений: 2^28, чтобы произведение не переполнило int64


def eval_expr_vec(
    expr: Expr, X: np.ndarray, env: Optional[dict[str, "Expert"]] = None
) -> tuple[np.ndarray, np.ndarray]:  # возвращает кортеж (значения, флаги)
    """Векторизованный вычислитель выражения на матрице входов X: (значения, флаги ok).

    X — (N, число переменных) int64; env — среда вызываемых экспертов.
    Возвращает значения (N,) int64 и ok (N,) bool. Вычисление осторожное:
    где результат мог быть неточным (переполнение, вход вне области
    вызываемого эксперта), значение обнуляется, а ok ложен — такие строки
    досчитывает скалярный вычислитель."""
    N = X.shape[0]  # N — число входных строк
    op = expr.op  # имя операции
    if op == "var":  # переменная
        return X[:, int(expr.value)].copy(), np.ones(N, dtype=bool)  # столбец X[:, индекс] и все строки верны
    if op == "const":  # константа
        v = int(expr.value)  # значение константы
        if abs(v) > _VEC_MAG_LIMIT:  # модуль больше предела
            return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=bool)  # нули и все строки неверны
        return np.full(N, v, dtype=np.int64), np.ones(N, dtype=bool)  # вектор из N копий значения и все строки верны
    if op == "call":  # вызов эксперта
        if env is None or expr.value not in env:  # среды нет или эксперта в ней нет
            return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=bool)  # вернуть стоимость: нули и все строки неверны
        callee = env[expr.value]  # вызываемый эксперт
        arg_vals = []  # значения аргументов по столбцам
        ok = np.ones(N, dtype=bool)  # все строки пока верны
        for a in expr.args:  # по аргументам
            v, aok = eval_expr_vec(a, X, env)  # значения и флаги аргумента
            arg_vals.append(v)  # добавить столбец значений
            ok &= aok  # ok = ok AND флаги аргумента
        if not np.any(ok):  # верных строк нет
            return np.zeros(N, dtype=np.int64), ok  # нули и все флаги ложны
        subX = np.column_stack(arg_vals)  # матрица аргументов (N, число аргументов) — столбцы значений
        for i, dom in enumerate(callee.domains):  # по аргументам вызываемого эксперта и их областям
            lo, hi = dom[0], dom[-1]  # lo = первый элемент области, hi = последний
            if len(dom) == hi - lo + 1 and dom == tuple(range(lo, hi + 1)):  # область — непрерывный отрезок lo..hi
                ok &= (subX[:, i] >= lo) & (subX[:, i] <= hi)  # итог: ok &= lo ≤ x_i ≤ hi
            else:  # иначе
                ok &= np.isin(subX[:, i], dom)  # ok &= x_i входит в область (точная проверка принадлежности)
        if callee.valid_rule == "nonzero_y":  # правило nonzero_y у вызываемого эксперта
            ok &= subX[:, 1] != 0  # итог: ok &= x_2 ≠ 0
        elif callee.valid_rule == "ordered_yz":  # правило ordered_yz
            ok &= subX[:, 1] <= subX[:, 2]  # итог: ok &= x_2 ≤ x_3
        if not np.any(ok):  # верных строк не осталось
            return np.zeros(N, dtype=np.int64), ok  # нули и флаги ok
        if callee.program.venn_mask is not None:  # вызываемая программа — диаграмма Венна
            idx = (np.clip(subX[:, 0], 0, 1) << 1) | np.clip(subX[:, 1], 0, 1)  # итог: idx = 2·clip(x_1, 0, 1) + clip(x_2, 0, 1)
            return ((int(callee.program.venn_mask) >> idx) & 1).astype(np.int64), ok  # значение = (маска >> idx) & 1
        res_all = np.zeros(N, dtype=np.int64)  # результат по всем строкам, изначально нули
        vals, ok2 = eval_expr_vec(callee.program.expr, subX[ok], env)  # вычислить выражение вызываемого эксперта на верных строках
        res_all[ok] = vals  # записать значения в верные строки
        ok_idx = np.flatnonzero(ok)  # номера верных строк
        ok[ok_idx[~ok2]] = False  # сбросить ok там, где вызов вернул флаг лжи
        return res_all, ok  # значения и флаги
    if op == "if_cond":  # условное выражение
        cv, cok = eval_expr_vec(expr.args[0], X, env)  # значения и флаги условия
        tv, tok = eval_expr_vec(expr.args[1], X, env)  # значения и флаги ветви «то»
        fv, fok = eval_expr_vec(expr.args[2], X, env)  # значения и флаги ветви «иначе»
        return np.where(cv != 0, tv, fv), cok & tok & fok  # условие ≠ 0 → ветвь «то», иначе «иначе»; ok = ok_условия AND ok_то AND ok_иначе
    if op == "poly":  # многочлен одного аргумента
        v, ok = eval_expr_vec(expr.args[0], X, env)  # значения и флаги аргумента x
        acc = np.zeros(N, dtype=np.int64)  # аккумулятор схемы Горнера, изначально нули
        for ci in reversed(list(expr.value or ())):  # по коэффициентам от старшего к младшему
            ok &= np.abs(acc) <= _VEC_MAG_LIMIT  # строки с |acc| > 2^28 признать неверными
            acc = np.where(ok, acc * v + int(ci), 0)  # acc = acc·x + c_i в верных строках, иначе 0
        ok &= np.abs(acc) <= _VEC_MAG_LIMIT  # окончательная проверка |acc| ≤ 2^28
        return acc, ok  # значения и флаги
    if op == "poly_nd":  # многочлен многих переменных
        acc = np.zeros(N, dtype=np.int64)  # сумма членов, изначально нули
        ok = np.ones(N, dtype=bool)  # все строки пока верны
        for coef, powers in expr.value or ():  # по членам (коэффициент, показатели степени)
            term = np.full(N, int(coef), dtype=np.int64)  # член — вектор из N копий коэффициента
            for idx, p in enumerate(powers):  # по номерам переменных idx и показателям p
                if p:  # показатель не ноль
                    ok &= (np.abs(X[:, idx]) <= 1000) & (int(p) <= 6)  # ok &= |x_idx| ≤ 1000 и p ≤ 6
                    safe_x = np.where(ok, X[:, idx], 0)  # x_idx в верных строках, иначе 0
                    term = np.where(ok, term * (safe_x ** int(p)), 0)  # term = term·x_idx^p в верных строках, иначе 0
            acc = np.where(ok, acc + term, 0)  # acc = acc + term в верных строках, иначе 0
        ok &= np.abs(acc) <= _VEC_MAG_LIMIT  # итог: ok &= |acc| ≤ 2^28
        return acc, ok  # значения и флаги
    if op == "anf":  # алгебраическая нормальная форма
        n_vars, coeffs = expr.value  # число переменных и список мономов с коэффициентами
        ok = np.ones(N, dtype=bool)  # все строки пока верны
        mask_x = np.zeros(N, dtype=np.int64)  # битовая маска входа, изначально нули
        for i in range(n_vars):  # по переменным i = 0..n−1
            ok &= (X[:, i] == 0) | (X[:, i] == 1)  # итог: ok &= x_i ∈ {0, 1}
            mask_x |= np.where(X[:, i] != 0, 1 << (n_vars - 1 - i), 0)  # маска |= 2^(n−1−i) в строках, где x_i ≠ 0
        acc = np.zeros(N, dtype=np.int64)  # аккумулятор результата, изначально нули
        for s, c in coeffs:  # по мономам s и коэффициентам c
            acc ^= np.where((mask_x & s) == s, c, 0)  # acc = acc XOR c в строках, где s AND маска = s
        return acc, ok  # значения и флаги

    vs = []  # список столбцов значений аргументов
    ok = np.ones(N, dtype=bool)  # все строки пока верны
    for a in expr.args:  # по аргументам
        v, aok = eval_expr_vec(a, X, env)  # значения и флаги аргумента
        vs.append(v)  # добавить столбец
        ok &= aok  # ok = ok AND флаги аргумента
    if op == "neg":  # итог: neg
        res = -vs[0]  # итог: res = −x
    elif op == "abs":  # итог: abs
        res = np.abs(vs[0])  # итог: res = |x|
    elif op == "sign":  # итог: sign
        res = np.sign(vs[0]).astype(np.int64)  # res = знак x как int64
    elif op == "bit_not8":  # итог: bit_not8
        res = (~vs[0]) & 0xFF  # res = НЕ x AND 0xFF — инверсия младших 8 бит
    elif op == "clamp3":  # итог: clamp3
        res = np.clip(vs[0], -3, 3)  # итог: res = clip(x, −3, 3)
    elif op == "square":  # итог: square
        ok &= np.abs(vs[0]) <= 32768  # итог: ok &= |x| ≤ 2^15
        s0 = np.where(ok, vs[0], 0)  # x в верных строках, иначе 0
        res = s0 * s0  # итог: res = x²
    elif op == "cube":  # итог: cube
        ok &= np.abs(vs[0]) <= 1024  # итог: ok &= |x| ≤ 2^10
        s0 = np.where(ok, vs[0], 0)  # x в верных строках, иначе 0
        res = s0 * s0 * s0  # итог: res = x³
    elif op == "absdiff":  # итог: absdiff
        res = np.abs(vs[0] - vs[1])  # итог: res = |x_0 − x_1|
    elif op == "fma":  # итог: fma
        ok &= (np.abs(vs[0]) <= 32768) & (np.abs(vs[1]) <= 32768)  # ok &= |x_0| ≤ 2^15 и |x_1| ≤ 2^15
        s0 = np.where(ok, vs[0], 0)  # x_0 в верных строках, иначе 0
        s1 = np.where(ok, vs[1], 0)  # x_1 в верных строках, иначе 0
        res = s0 * s1 + vs[2]  # итог: res = x_0·x_1 + x_2
    elif op == "clamp":  # итог: clamp
        res = np.minimum(np.maximum(vs[0], vs[1]), vs[2])  # итог: res = min(max(x_0, x_1), x_2)
    elif op == "median3":  # итог: median3
        res = np.sort(np.stack(vs, axis=0), axis=0)[1]  # res = средний элемент построчной сортировки трёх аргументов
    elif op == "in_set":  # итог: in_set
        table = np.zeros(1 << 12, dtype=bool)  # таблица принадлежности на 4096 значений, изначально ложь
        for v in (expr.value or ()):  # по элементам множества
            if 0 <= int(v) < table.size:  # элемент попадает в таблицу: 0 ≤ v < 4096
                table[int(v)] = True  # пометить значение v как входящее
        res = table[np.clip(vs[0], 0, table.size - 1)]  # res = таблица[clip(x, 0, 4095)]
    elif op == "in_ranges":  # итог: in_ranges
        res = np.zeros_like(vs[0], dtype=bool)  # накопитель принадлежности, изначально ложь
        for lo, hi in (expr.value or ()):  # по диапазонам (lo, hi)
            res |= (vs[0] >= int(lo)) & (vs[0] <= int(hi))  # итог: res |= lo ≤ x ≤ hi
        res = res.astype(vs[0].dtype)  # привести к типу аргумента
    elif op == "mul3":  # итог: mul3
        ok &= (  # итог: ok &= (
            (np.abs(vs[0]) <= 1024) & (np.abs(vs[1]) <= 1024) & (np.abs(vs[2]) <= 1024)  # |x_0| ≤ 2^10 и |x_1| ≤ 2^10 и |x_2| ≤ 2^10
        )
        s0 = np.where(ok, vs[0], 0)  # x_0 в верных строках, иначе 0
        s1 = np.where(ok, vs[1], 0)  # x_1 в верных строках, иначе 0
        s2 = np.where(ok, vs[2], 0)  # x_2 в верных строках, иначе 0
        res = s0 * s1 * s2  # mul3: итог res = x_0·x_1·x_2
    elif op == "add":  # итог: add
        res = vs[0] + vs[1]  # итог: res = x_0 + x_1
    elif op == "sub":  # итог: sub
        res = vs[0] - vs[1]  # итог: res = x_0 − x_1
    elif op == "mul":  # итог: mul
        ok &= (np.abs(vs[0]) <= 32768) & (np.abs(vs[1]) <= 32768)  # проверка: ok &= |x_0| ≤ 2^15 и |x_1| ≤ 2^15
        s0 = np.where(ok, vs[0], 0)  # x_0 в верных строках, иначе 0
        s1 = np.where(ok, vs[1], 0)  # x_1 в верных строках, иначе 0
        res = s0 * s1  # итог: res = x_0·x_1
    elif op == "min":  # итог: min
        res = np.minimum(vs[0], vs[1])  # итог: res = min(x_0, x_1)
    elif op == "max":  # итог: max
        res = np.maximum(vs[0], vs[1])  # итог: res = max(x_0, x_1)
    elif op == "eq":  # итог: eq
        res = (vs[0] == vs[1]).astype(np.int64)  # res = 1, если x_0 = x_1, иначе 0
    elif op == "ne":  # итог: ne
        res = (vs[0] != vs[1]).astype(np.int64)  # res = 1, если x_0 ≠ x_1, иначе 0
    elif op == "lt":  # итог: lt
        res = (vs[0] < vs[1]).astype(np.int64)  # res = 1, если x_0 < x_1, иначе 0
    elif op == "le":  # итог: le
        res = (vs[0] <= vs[1]).astype(np.int64)  # res = 1, если x_0 ≤ x_1, иначе 0
    elif op == "gt":  # итог: gt
        res = (vs[0] > vs[1]).astype(np.int64)  # res = 1, если x_0 > x_1, иначе 0
    elif op == "ge":  # итог: ge
        res = (vs[0] >= vs[1]).astype(np.int64)  # res = 1, если x_0 ≥ x_1, иначе 0
    elif op == "bit_and":  # итог: bit_and
        res = vs[0] & vs[1]  # итог: res = x_0 AND x_1
    elif op == "bit_or":  # итог: bit_or
        res = vs[0] | vs[1]  # итог: res = x_0 OR x_1
    elif op == "bit_xor":  # итог: bit_xor
        res = vs[0] ^ vs[1]  # итог: res = x_0 XOR x_1
    elif op == "bit_nand8":  # итог: bit_nand8
        res = (~(vs[0] & vs[1])) & 0xFF  # res = НЕ(x_0 AND x_1) по младшим 8 битам
    elif op == "bit_nor8":  # итог: bit_nor8
        res = (~(vs[0] | vs[1])) & 0xFF  # res = НЕ(x_0 OR x_1) по младшим 8 битам
    elif op == "bit_xnor8":  # итог: bit_xnor8
        res = (~(vs[0] ^ vs[1])) & 0xFF  # res = НЕ(x_0 XOR x_1) по младшим 8 битам
    elif op == "floordiv":  # итог: floordiv
        ok &= vs[1] != 0  # делитель 0 делает строку неверной: ok &= x_1 ≠ 0
        safe = np.where(vs[1] != 0, vs[1], 1)  # безопасный делитель: x_1, а где он равен 0 — 1
        res = np.floor_divide(vs[0], safe)  # res = ⌊x_0 / делитель⌋
    elif op == "mod":  # итог: mod
        ok &= vs[1] != 0  # итог: ok &= x_1 ≠ 0
        safe = np.where(vs[1] != 0, vs[1], 1)  # безопасный делитель: x_1, а где он равен 0 — 1
        res = np.mod(vs[0], safe)  # res = x_0 mod делитель
    elif op == "gcd":  # итог: gcd
        res = np.gcd(vs[0], vs[1])  # res = НОД(x_0, x_1)
    elif op == "lcm":  # итог: lcm
        res = np.lcm(vs[0], vs[1])  # res = НОК(x_0, x_1)
    elif op == "pow_int":  # итог: pow_int
        ok &= (vs[1] >= 0) & (vs[1] <= 8) & (np.abs(vs[0]) <= 32)  # ok &= 0 ≤ x_1 ≤ 8 и |x_0| ≤ 32
        sa = np.where(ok, vs[0], 0)  # основание в верных строках, иначе 0
        sb = np.where(ok, vs[1], 0)  # показатель в верных строках, иначе 0
        res = sa**sb  # итог: res = x_0^x_1
    else:  # неизвестная операция
        raise ValueError(f"unsupported vec op: {op}")  # ошибка: операция не поддерживается векторизованно
    ok &= np.abs(res) <= _VEC_MAG_LIMIT  # ok &= |res| ≤ 2^28 — результат не выходит за предел
    return res, ok  # вернуть значения и флаги


def candidate_programs(arity: int) -> list[Program]:
    """Начальные программы-кандидаты заданной арности: по одному простому выражению на каждую базовую операцию.

    Кандидаты — выражения только из переменных и одной операции. Дубликаты
    по тексту убираются. Возвращает список Program, отсортированный по
    (стоимость, читаемая запись)."""
    v = [Expr.var(i) for i in range(arity)]  # узлы переменных x_0, …, x_{arity−1}
    cands: list[Expr] = []  # список выражений-кандидатов
    if arity == 1:  # арность 1
        x = v[0]  # аргумент x
        cands += [  # добавить семь унарных выражений
            Expr("neg", (x,)),  # итог: neg(x)
            Expr("abs", (x,)),  # итог: abs(x)
            Expr("sign", (x,)),  # итог: sign(x)
            Expr("bit_not8", (x,)),  # итог: bit_not8(x)
            Expr("square", (x,)),  # итог: square(x)
            Expr("cube", (x,)),  # итог: cube(x)
            Expr("clamp3", (x,)),  # итог: clamp3(x)
        ]
    elif arity == 2:  # арность 2
        x, y = v  # аргументы x, y
        for op in BINARY_OPS:  # по всем бинарным операциям
            cands.append(Expr(op, (x, y)))  # кандидат op(x, y)
        cands.extend([Expr("sub", (y, x)), Expr("absdiff", (x, y))])  # плюс обратное вычитание y − x и |x − y|
    elif arity == 3:  # арность 3
        x, y, z = v  # аргументы x, y, z
        cands += [  # добавить четыре трёхместных выражения
            Expr("fma", (x, y, z)),  # итог: fma(x, y, z)
            Expr("clamp", (x, y, z)),  # итог: clamp(x, y, z)
            Expr("median3", (x, y, z)),  # итог: median3(x, y, z)
            Expr("mul3", (x, y, z)),  # итог: mul3(x, y, z)
        ]
    by_text = {}  # словарь «текст выражения -> выражение» для удаления дубликатов
    for e in cands:  # по кандидатам
        by_text.setdefault(e.pretty(), e)  # запомнить первого с таким текстом
    return [  # вернуть список программ
        Program(arity, e)  # программа заданной арности с выражением e
        for e in sorted(by_text.values(), key=lambda e: (e.cost, e.pretty()))  # в порядке (стоимость, текст)
    ]


#============================== разделитель разделов ==============================
# 4. Голова Венна, упаковка 16 масок в 13 байт, полином Жегалкина, Адамар, 5-тритовый кодек
#============================== разделитель разделов ==============================


def venn_atoms(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Базис атомов Венна из четырёх ReLU-нейронов на входах {−1, +1}: матрица (N, 4).

    s1, s2 — входы со значениями −1 или +1. Ровно один атом равен 1, остальные
    0: a_0 = max(0, (−s1 − s2)/2), a_1 = max(0, (−s1 + s2)/2),
    a_2 = max(0, (s1 − s2)/2), a_3 = max(0, (s1 + s2)/2)."""
    return np.column_stack(  # соединить четыре столбца-атома в матрицу
        [  # начало списка столбцов-атомов
            np.maximum(0.0, (-s1 - s2) * 0.5),  # атом 0: max(0, (−s1 − s2)/2) — активен при s1 = s2 = −1
            np.maximum(0.0, (-s1 + s2) * 0.5),  # атом 1: max(0, (−s1 + s2)/2) — активен при s1 = −1, s2 = +1
            np.maximum(0.0, (s1 - s2) * 0.5),  # атом 2: max(0, (s1 − s2)/2) — активен при s1 = +1, s2 = −1
            np.maximum(0.0, (s1 + s2) * 0.5),  # атом 3: max(0, (s1 + s2)/2) — активен при s1 = s2 = +1
        ]
    )


def venn_head_eval(mask: int, x_bin: Sequence[int]) -> int:
    """Вычислить булеву функцию маски Венна на двух булевых входах как нейрон «атомы · веса».

    Входы 0/1 переводятся в −1/+1: s = 2·x − 1; вес w_idx = бит маски idx;
    результат = round(Σ_idx a_idx·w_idx)."""
    s1 = np.array([2.0 * int(x_bin[0]) - 1.0])  # s1 = 2·x_1 − 1 как массив из одного элемента
    s2 = np.array([2.0 * int(x_bin[1]) - 1.0])  # итог: s2 = 2·x_2 − 1
    atoms = venn_atoms(s1, s2)[0]  # атомы Венна для этой пары входов
    w = np.array([(mask >> idx) & 1 for idx in range(4)], dtype=np.float64)  # веса: w_idx = (маска >> idx) & 1 для idx = 0..3
    return int(round(float(atoms @ w)))  # вернуть round(атомы · веса)


def pack_venn_16_flat() -> bytes:
    """Упаковать 16 четырёхбитных масок в 13 байт без потерь.

    Все 64 бита масок собираются в поток по 5 штук (последняя пятёрка
    дополняется нулями до 65 бит); байт = Σ_{i=0..4} t_i·3^(4−i) — число в
    интервале 0..242; 13 байт = 65 бит — точная нижняя граница."""
    """Упаковать 16 масок по 4 трита в 13 байт без потерь (точная нижняя граница)."""
    trits = [(mask >> idx) & 1 for mask in range(16) for idx in range(4)]  # поток битов: для каждой маски 0..15 её биты idx = 0..3
    while len(trits) % 5 != 0:  # пока длина не кратна 5
        trits.append(0)  # добавить нулевой бит
    out = bytearray()  # накопитель байтов
    for off in range(0, len(trits), 5):  # по пятёркам бит
        b = 0  # байт, изначально 0
        for t in trits[off : off + 5]:  # по пяти битам t пятёрки
            b = b * 3 + t  # b = b·3 + t — накопление числа по основанию 3
        out.append(b)  # добавить байт
    assert len(out) == 13  # проверка: ровно 13 байт
    return bytes(out)  # вернуть упакованные байты


def unpack_venn_16_flat(buf: bytes) -> list[int]:
    """Распаковать 13 байт обратно в список из 16 масок (действие, обратное pack_venn_16_flat)."""
    assert len(buf) == 13  # проверка длины
    trits = []  # накопитель битов
    for b in buf:  # по байтам
        chunk = [0] * 5  # пять тритов байта, изначально нули
        v = b  # остаток от деления
        for i in range(4, -1, -1):  # от младшего трита к старшему
            chunk[i] = v % 3  # трит i = v mod 3
            v //= 3  # итог: v = ⌊v / 3⌋
        trits.extend(chunk)  # добавить триты в поток
    masks = []  # накопитель масок
    for m in range(16):  # по номерам масок
        val = 0  # значение маски, изначально 0
        for idx in range(4):  # по битам idx = 0..3
            val |= (trits[m * 4 + idx] & 1) << idx  # маска |= (трит & 1) << idx
        masks.append(val)  # добавить маску
    return masks  # вернуть 16 масок


def anf_transform(truth_table: Sequence[int]) -> list[int]:
    """Преобразование Жегалкина (Мёбиуса) таблицы истинности в коэффициенты АНФ: «бабочка».

    n = длина таблицы — степень 2. Коэффициент при мономе s равен XOR значений
    таблицы на всех индексах t ⊆ s (поразрядно). Сложность O(n·log n);
    преобразование — инволюция: повторное применение возвращает исходную
    таблицу."""
    """Преобразование Жегалкина (Мёбиуса) таблицы истинности: «бабочка», O(n·log n), инволюция."""
    n_tot = len(truth_table)  # длина таблицы
    if n_tot == 0 or (n_tot & (n_tot - 1)) != 0:  # пустая таблица или длина не степень 2 (n AND (n − 1) ≠ 0)
        raise ValueError("truth_table length must be a power of 2")  # ошибка: длина должна быть степенью 2
    coeffs = [int(v) & 1 for v in truth_table]  # рабочая копия коэффициентов, приведённая к битам 0/1
    step = 1  # размер бабочки, изначально 1
    while step < n_tot:  # пока шаг меньше длины
        for i in range(0, n_tot, step * 2):  # по блокам длины 2·step
            for j in range(step):  # по позициям внутри половины блока
                coeffs[i + step + j] ^= coeffs[i + j]  # верхняя половина XOR нижняя: c[i + step + j] ^= c[i + j]
        step <<= 1  # итог: step = 2·step
    return coeffs  # вернуть коэффициенты АНФ


def anf_eval(coeffs: Sequence[int], n_vars: int, xs: Sequence[int]) -> int:
    """Вычислить значение АНФ на входе xs по списку коэффициентов (1 на месте коэффициента 1).

    mask_x = Σ x_i·2^(n−1−i); результат = XOR 1 по всем s, у которых
    коэффициент c_s = 1 и моном s ⊆ mask_x (s AND mask_x = s)."""
    mask_x = 0  # битовая маска входа, изначально 0
    for i in range(n_vars):  # по переменным
        if int(xs[i]) & 1:  # младший бит x_i равен 1
            mask_x |= 1 << (n_vars - 1 - i)  # маска |= 2^(n−1−i)
    acc = 0  # аккумулятор, изначально 0
    for s, c in enumerate(coeffs):  # по мономам s и коэффициентам c
        if c and (s & mask_x) == s:  # коэффициент равен 1 и моном входит в вход
            acc ^= 1  # итог: acc ^= 1
    return acc  # вернуть значение АНФ


def anf_compose3(
    f_bool: Callable[[int, int], int], g_tt: Sequence[int], h_tt: Sequence[int]  # типы аргументов: внешняя функция и две таблицы истинности
) -> list[int]:  # возвращает список коэффициентов АНФ
    """АНФ суперпозиции h(x) = f(g(x), h(x)) по таблицам истинности g и h.

    g_tt, h_tt — таблицы истинности внутренних функций на одних и тех же
    входах; f_bool — внешняя булева функция двух аргументов. Строится таблица
    t_i = f(g_i, h_i) и преобразуется Жегалкина."""
    """АНФ суперпозиции (f ∘ (g, h))(x) = f(g(x), h(x)) по таблицам истинности g и h."""
    assert len(g_tt) == len(h_tt)  # таблицы g и h обязаны быть одинаковой длины
    tt = [int(f_bool(int(g_tt[i]) & 1, int(h_tt[i]) & 1)) & 1 for i in range(len(g_tt))]  # таблица t_i = f(g_i & 1, h_i & 1) & 1
    return anf_transform(tt)  # вернуть её коэффициенты АНФ


def anf_support_vars(coeffs: Sequence[int], n_vars: int) -> tuple[int, ...]:
    """Номера переменных, реально входящих в АНФ: кортеж индексов в порядке x_0, x_1, …"""
    used = 0  # объединение битов всех мономов с единичным коэффициентом, изначально 0
    for s, c in enumerate(coeffs):  # по мономам s
        if c:  # коэффициент равен 1
            used |= s  # итог: used |= s
    return tuple(i for i in range(n_vars) if (used >> (n_vars - 1 - i)) & 1)  # вернуть i, у которых бит (used >> (n−1−i)) & 1 установлен


def anf_to_expr(coeffs: Sequence[int], n_vars: int) -> Expr:
    """Выразить АНФ деревом выражения Expr.

    Не более 4 мономов — явная сумма XOR произведений (bit_and) переменных;
    пустая АНФ — константа 0; единственный пустой моном — константа 1;
    иначе узел anf."""
    sparse = tuple((s, 1) for s, c in enumerate(coeffs) if c)  # разреженный список пар (моном s, коэффициент 1) для c_s = 1
    if len(sparse) == 0:  # нет мономов
        return Expr.const(0)  # константа 0
    if len(sparse) == 1 and sparse[0][0] == 0:  # единственный моном — пустой (константа)
        return Expr.const(1)  # константа 1
    if len(sparse) <= 4:  # не более 4 мономов — развернуть в операции
        mons = []  # список мономов-выражений
        for s, _ in sparse:  # по мономам
            if s == 0:  # пустой моном
                mons.append(Expr.const(1))  # записывается как константа 1
            else:  # иначе
                vs = [  # переменные, чьи биты входят в моном
                    Expr.var(i) for i in range(n_vars) if (s >> (n_vars - 1 - i)) & 1  # узел x_i для i, у которых (s >> (n−1−i)) & 1 = 1
                ]
                t = vs[0]  # произведение начинается с первой переменной
                for v in vs[1:]:  # по остальным переменным монома
                    t = Expr("bit_and", (t, v))  # t = t AND v — конъюнкция слева направо
                mons.append(t)  # добавить моном
        res = mons[0]  # сумма начинается с первого монома
        for m in mons[1:]:  # по остальным мономам
            res = Expr("bit_xor", (res, m))  # итог: res = res XOR m
        return res  # вернуть выражение
    return Expr("anf", (), (n_vars, sparse))  # иначе узел anf с числом переменных и разреженным списком мономов


_HADAMARD_CACHE: dict[int, np.ndarray] = {}  # кэш матриц Адамара по показателю k


def sylvester_hadamard(k: int) -> np.ndarray:
    """Ортонормированная матрица Адамара Сильвестра порядка 2^k, построенная детерминированно.

    H_0 = [1]; H_{k+1} = [[H_k, H_k], [H_k, −H_k]]; итог делится на √(2^k),
    так что H·H^T = E."""
    if k in _HADAMARD_CACHE:  # матрица уже в кэше
        return _HADAMARD_CACHE[k]  # взять из кэша
    H = np.array([[1.0]], dtype=np.float64)  # итог: H_0 = [1]
    for _ in range(k):  # k раз удвоить порядок
        H = np.block([[H, H], [H, -H]])  # итог: H = [[H, H], [H, −H]]
    H /= math.sqrt(2**k)  # нормировка: H = H / √(2^k)
    _HADAMARD_CACHE[k] = H  # запомнить в кэше
    return H  # вернуть матрицу


def block_hadamard_rotate(v: np.ndarray, blk_size: int = 8) -> tuple[np.ndarray, int]:
    """Поворот вектора блоками умножением на матрицу Адамара: (повёрнутый вектор, исходная длина).

    Вектор дополняется нулями до кратности blk_size, делится на блоки и
    каждый блок умножается на H справа: v' = v·H. Разворачивается обратно
    в одну строку."""
    H = sylvester_hadamard(int(round(math.log2(blk_size))))  # матрица Адамара порядка blk_size (blk_size = 2^k)
    v1 = np.asarray(v, dtype=np.float64).ravel()  # вектор float64 в одну строку
    n = v1.size  # исходная длина n
    nb = (n + blk_size - 1) // blk_size  # число блоков nb = ⌈n / blk_size⌉
    vp = np.pad(v1, (0, nb * blk_size - n)).reshape(nb, blk_size)  # дополнить нулями до nb·blk_size и разрезать на блоки (nb, blk_size)
    return (vp @ H).ravel(), n  # вернуть (блоки · H) в одну строку и исходную длину n


def block_hadamard_unrotate(
    vr: np.ndarray, orig_n: int, blk_size: int = 8  # повёрнутый вектор, исходная длина и размер блока
) -> np.ndarray:  # возвращает вектор исходной длины
    """Обратный блочный поворот Адамара: (повёрнутый вектор, длина) -> вектор.

    Каждый блок умножается на H^T (обратная ортогональная матрица); лишние
    элементы отбрасываются до исходной длины orig_n."""
    H = sylvester_hadamard(int(round(math.log2(blk_size))))  # матрица Адамара порядка blk_size
    nb = vr.size // blk_size  # число блоков nb = длина / blk_size
    return (vr.reshape(nb, blk_size) @ H.T).ravel()[:orig_n]  # вернуть (блоки · H^T) в строку, обрезанную до orig_n


_TRIT_POW = np.array([81, 27, 9, 3, 1], dtype=np.int64)  # веса разрядов пяти тритов по основанию 3: [3^4, 3^3, 3^2, 3, 1]


def quantize_5block_exact(blk: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Точное квантование блока из 5 чисел: (квантованный блок, (exp2, best, bval)).

    Шаг = 2^exp2 · s, где exp2 = clamp(⌈log2(m/1.5)⌉, −126, 126), m = max|blk|,
    а уровень s перебирается по DYADIC_LEVELS. Для каждого s: трит
    q_i = 1, если b_i/s > 0.5; −1, если < −0.5; иначе 0. Выбирается s с
    минимальной ошибкой Σ(b_i − q_i·s)². bval = Σ (q_i + 1)·3^(4−i) — код
    тритов; (0, 0, 121) обозначает нулевой блок."""
    m = float(np.max(np.abs(blk)))  # m = max|blk| — наибольший модуль в блоке
    if m < 1e-15:  # блок практически нулевой
        return np.zeros_like(blk), (0, 0, 121)  # нулевой блок: нули и код (0, 0, 121)
    exp2 = max(-126, min(126, math.ceil(math.log2(m / 1.5 + 1e-18))))  # итог: exp2 = clamp(⌈log2(m/1.5)⌉, −126, 126)
    sc = 2.0**exp2  # масштаб sc = 2^exp2
    bn = blk / sc  # нормированный блок bn = blk / sc
    s = DYADIC_LEVELS[:, None]  # столбец уровней s (255, 1)
    ratio = bn[None, :] / s  # итог: ratio[i, j] = bn_j / s_i
    q_int = np.where(ratio > 0.5, 1.0, np.where(ratio < -0.5, -1.0, 0.0))  # q = 1, если ratio > 0.5; −1, если ratio < −0.5; иначе 0
    errs = np.sum((bn[None, :] - q_int * s) ** 2, axis=1)  # errs_i = Σ_j (bn_j − q_ij·s_i)² — ошибка уровня s_i
    best = int(np.argmin(errs))  # номер лучшего уровня (минимальная ошибка)
    s_best = float(DYADIC_LEVELS[best])  # лучший уровень s_best
    trits = q_int[best].astype(np.int64)  # триты лучшего уровня как int64
    in_mask = np.abs(bn) <= 1.5 * s_best  # маска элементов с |bn| ≤ 1.5·s_best — внутри диапазона
    sat_tail = float(np.sum(np.where(~in_mask, (np.abs(bn) - s_best) ** 2, 0.0)))  # хвост насыщения: Σ (|bn| − s_best)² вне диапазона
    bound = int(np.sum(in_mask)) * (s_best / 2.0) ** 2 + sat_tail  # верхняя граница ошибки: n_внутри·(s_best/2)² + хвост
    assert float(errs[best]) <= bound + 1e-9  # проверка: ошибка не больше границы
    bval = int(np.dot(trits + 1, _TRIT_POW))  # bval = Σ_i (трит_i + 1)·3^(4−i)
    return (trits * (s_best * sc)).astype(np.float64), (exp2, best, bval)  # вернуть триты·s_best·sc и код (exp2, best, bval)


def pack_vector_dyadic5(vec: np.ndarray, stages: int = 2) -> tuple[bytes, np.ndarray]:
    """Упаковать вектор float64 в байты 5-тритовым диадическим кодеком без потерь: (байты, раскодированное).

    Заголовок: длина n (2 байта, младший вперёд) и число стадий stages.
    Остаток кодируется стадиями: каждая стадия квантует блоки текущего
    остатка по 5 элементов и вычитает квантованное. Блок 3 байта:
    (exp2 + 128, best, bval) или один байт 255 для нулевого блока.
    Второй элемент — результат unpack_vector_dyadic5 для проверки."""
    v = np.asarray(vec, dtype=np.float64).ravel()  # вектор float64 в одну строку
    n = v.size  # длина n
    nblk = (n + 4) // 5  # число блоков nblk = ⌈n / 5⌉
    vp = np.pad(v, (0, nblk * 5 - n))  # дополнить нулями до nblk·5
    buf = bytearray()  # накопитель байтов
    buf.extend(int(n).to_bytes(2, "little"))  # записать n двумя байтами, младший вперёд
    buf.append(int(stages))  # записать число стадий
    rem = vp.copy()  # остаток — копия вектора
    for _ in range(stages):  # по стадиям
        for b in range(nblk):  # по блокам
            blk = rem[b * 5 : (b + 1) * 5]  # текущий блок остатка из 5 элементов
            q_blk, (exp2, best, bval) = quantize_5block_exact(blk)  # квантовать блок
            if exp2 == 0 and best == 0 and bval == 121:  # нулевой блок (exp2 = 0, best = 0, bval = 121)
                buf.append(255)  # записать один байт 255
            else:  # иначе
                buf.append(exp2 + 128)  # байт показателя exp2 + 128 (сдвиг в неотрицательные)
                buf.append(best)  # байт номера уровня
                buf.append(bval)  # байт кода тритов
            rem[b * 5 : (b + 1) * 5] -= q_blk  # остаток блока уменьшить на квантованное: rem −= q_blk
    raw = bytes(buf)  # готовые байты
    return raw, unpack_vector_dyadic5(raw)  # вернуть байты и их раскодировку


def unpack_vector_dyadic5(buf: bytes) -> np.ndarray:
    """Раскодировать вектор, упакованный pack_vector_dyadic5: массив float64 длины n."""
    n = int.from_bytes(buf[:2], "little")  # длина n из первых двух байтов
    stages = buf[2]  # число стадий из третьего байта
    nblk = (n + 4) // 5  # число блоков nblk = ⌈n / 5⌉
    out = np.zeros(nblk * 5, dtype=np.float64)  # результат, изначально нули длины nblk·5
    pos = 3  # позиция чтения — сразу после заголовка
    for _ in range(stages):  # по стадиям
        for b in range(nblk):  # по блокам
            tag = buf[pos]  # байт-метка блока
            pos += 1  # сдвинуть позицию
            if tag == 255:  # метка 255 — нулевой блок
                continue  # пропустить блок
            exp2 = tag - 128  # exp2 = метка − 128
            best = buf[pos]  # номер уровня
            pos += 1  # сдвинуть позицию
            bval = buf[pos]  # код тритов
            pos += 1  # сдвинуть позицию
            sc = 2.0**exp2  # масштаб sc = 2^exp2
            lvl = float(DYADIC_LEVELS[best])  # уровень lvl = DYADIC_LEVELS[best]
            trits = [0] * 5  # пять тритов, изначально нули
            v = bval  # остаток кода
            for i in range(4, -1, -1):  # от младшего трита к старшему
                trits[i] = (v % 3) - 1  # трит i = (v mod 3) − 1
                v //= 3  # итог: v = ⌊v / 3⌋
            out[b * 5 : (b + 1) * 5] += np.array(trits, dtype=np.float64) * (lvl * sc)  # прибавить к блоку триты·lvl·sc
    return out[:n]  # вернуть первые n элементов


def pack_bitalloc_records(
    k: int, idx_sorted: np.ndarray, records: list[tuple[int, tuple[int, int, int]]]  # типы аргументов: k, индексы и список записей
) -> bytes:  # возвращает упакованные байты
    """Упаковать записи о распределении бит: (k, отсортированные индексы, записи) -> байты.

    k — число ненулевых координат; idx_sorted — их отсортированные индексы;
    records — (номер блока, (exp2, best, bval)). Формат: k; признак wide
    (1, если максимальный индекс ≥ 255 — индексы по 2 байта); индексы;
    число записей; для записи с номером ≥ числа блоков — номер блока;
    затем 255 для нулевого блока или (exp2 + 128, best, bval)."""
    n_blocks = (k + 4) // 5  # число блоков по 5 координат: ⌈k / 5⌉
    buf = bytearray()  # накопитель байтов
    buf.append(int(k))  # записать k
    max_idx = int(np.max(idx_sorted)) if k > 0 else 0  # наибольший индекс (0, если k = 0)
    wide = 1 if max_idx >= 255 else 0  # wide = 1, если наибольший индекс ≥ 255, иначе 0
    buf.append(wide)  # записать признак wide
    for ix in idx_sorted:  # по индексам в порядке возрастания
        if wide:  # широкий формат
            buf.extend(int(ix).to_bytes(2, "little"))  # записать индекс двумя байтами, младший вперёд
        else:  # иначе
            buf.append(int(ix))  # записать индекс одним байтом
    buf.append(len(records))  # записать число записей
    for ri, (blk_id, (exp2, best, bval)) in enumerate(records):  # по записям с номерами ri
        if ri >= n_blocks:  # номер записи больше числа блоков — номер блока не подразумевается
            buf.append(int(blk_id))  # записать номер блока
        if exp2 == 0 and best == 0 and bval == 121:  # нулевой блок
            buf.append(255)  # записать 255
        else:  # иначе
            buf.append(exp2 + 128)  # записать exp2 + 128
            buf.append(int(best))  # записать номер уровня
            buf.append(int(bval))  # записать код тритов
    return bytes(buf)  # вернуть байты


def unpack_bitalloc_records(buf: bytes, D: int) -> np.ndarray:
    """Раскодировать записи о распределении бит в вектор длины D (обратно pack_bitalloc_records)."""
    pos = 0  # позиция чтения
    k = buf[pos]  # k — число ненулевых координат
    pos += 1  # сдвинуть позицию
    wide = buf[pos]  # признак широкого формата
    pos += 1  # сдвинуть позицию
    idx_sorted = []  # список индексов
    for _ in range(k):  # читать k индексов
        if wide:  # широкий формат
            idx_sorted.append(int.from_bytes(buf[pos : pos + 2], "little"))  # индекс из двух байтов, младший вперёд
            pos += 2  # позиция + 2
        else:  # иначе
            idx_sorted.append(buf[pos])  # индекс из одного байта
            pos += 1  # позиция + 1
    n_rec = buf[pos]  # число записей
    pos += 1  # сдвинуть позицию
    n_blocks = (k + 4) // 5  # число блоков ⌈k / 5⌉
    wq_s = np.zeros(k, dtype=np.float64)  # сжатые квантованные значения по k координатам, изначально нули
    for ri in range(n_rec):  # по записям
        blk_id = ri if ri < n_blocks else buf[pos]  # номер блока: ri, если ri < числа блоков, иначе читается из байта
        if ri >= n_blocks:  # номер блока был записан явно
            pos += 1  # позиция + 1
        tag = buf[pos]  # метка блока
        pos += 1  # позиция + 1
        if tag == 255:  # метка 255 — нулевой блок
            continue  # пропустить
        exp2 = tag - 128  # exp2 = метка − 128
        best = buf[pos]  # номер уровня
        pos += 1  # позиция + 1
        bval = buf[pos]  # код тритов
        pos += 1  # позиция + 1
        sc = 2.0**exp2  # масштаб sc = 2^exp2
        lvl = float(DYADIC_LEVELS[best])  # уровень lvl = DYADIC_LEVELS[best]
        trits = [0] * 5  # пять тритов, изначально нули
        v = bval  # остаток кода
        for i in range(4, -1, -1):  # от младшего трита к старшему
            trits[i] = (v % 3) - 1  # трит i = (v mod 3) − 1
            v //= 3  # итог: v = ⌊v / 3⌋
        sl = slice(blk_id * 5, min((blk_id + 1) * 5, k))  # срез координат блока: [5·blk_id, min(5·(blk_id + 1), k))
        wq_s[sl] += np.array(trits[: sl.stop - sl.start], dtype=np.float64) * (lvl * sc)  # прибавить к срезу триты (обрезанные по длине среза)·lvl·sc
    out = np.zeros(D, dtype=np.float64)  # выходной вектор длины D, изначально нули
    if k > 0:  # есть ненулевые координаты
        out[np.array(idx_sorted, dtype=np.int64)] = wq_s  # развернуть сжатые значения по их индексам: out[idx_i] = wq_s_i
    return out  # вернуть вектор


#============================== разделитель разделов ==============================
# 5. Логарифмически-линейная перекрёстная энтропия с защитой диапазона, BF16, EF21-Muon, GPM и нейронное ядро
#============================== разделитель разделов ==============================


def bf16_nearest_float(x: float) -> float:
    """Округлить число до ближайшего в формате BF16 (16 бит: 8 показательных, 7 значащих).

    Нормальные числа: шаг = 2^(⌊log2|x|⌋ − 7); результат =
    sign(x)·round(|x| / шаг)·шаг (округление к ближайшему чётному).
    Модуль меньше 2^−126 даёт 0; ноль, NaN и бесконечности возвращаются как есть."""
    if x == 0.0 or not math.isfinite(x):  # ноль, NaN или бесконечность — округлять нечего
        return 0.0 if x == 0.0 else x  # ноль остаётся нулём, остальное возвращается как есть
    ax = abs(x)  # модуль числа
    if ax < 2.0 ** (-126):  # модуль меньше наименьшей нормальной степени 2^−126
        return 0.0  # округляется до нуля
    exp2 = math.floor(math.log2(ax))  # exp2 = ⌊log2|x|⌋ — двоичная экспонента
    step = 2.0 ** (exp2 - 7)  # шаг сетки: 2^(exp2 − 7)
    return math.copysign(round(ax / step) * step, x)  # sign(x)·round(|x| / шаг)·шаг


def batch_logit_safe_step_ce(
    X_feat: np.ndarray,  # признаки (B, d)
    y_int: np.ndarray,  # номера правильных классов (B,)
    num_classes: int,  # число классов
    steps: int = 300,  # максимум шагов
    l2_reg: Optional[np.ndarray] = None,  # коэффициенты L2-регуляризации или None
    use_bf16_guard: bool = False,  # проверять округление BF16
) -> dict[str, Any]:  # возвращает словарь с весами и статистикой шагов
    """Градиентный шаг логистической регрессии с сертифицированной длиной шага.

    X_feat — (B, d) признаки; y_int — (B,) номера классов; num_classes —
    число классов; steps — максимум шагов; l2_reg — (d,) коэффициенты
    L2-регуляризации или None; use_bf16_guard — проверка того, что
    округление весов до BF16 не съедает гарантированное убывание.

    Потери: CE = mean_i (lse_i − logit правильного класса) +
    ½·Σ reg_j·θ_j². Шаг — вдоль антиградиента D = −ḡ, где
    ḡ = (p − y)^T X / B + reg·θ. Длина шага η = 4A / (Q' + 4·curv),
    A = ‖ḡ‖², Q' = mean_i (range_i(X·D^T))², curv — вклад регуляризации;
    гарантированное убывание 2A² / (Q' + 4·curv)."""
    B, d = X_feat.shape  # B — число примеров, d — число признаков
    theta = np.zeros((num_classes, d), dtype=np.float64)  # веса θ (num_classes, d), изначально нули
    reg = (  # регуляризация:
        np.zeros(d, dtype=np.float64)  # нули, если l2_reg не задан,
        if l2_reg is None  # если регуляризация не задана
        else np.asarray(l2_reg, dtype=np.float64)  # иначе заданный массив как float64
    )

    """Потери перекрёстной энтропии с регуляризацией на весах th: число."""
    def ce_loss(th: np.ndarray) -> float:  # локальная функция потерь от весов th
        lg = X_feat @ th.T  # логиты lg = X·θ^T, форма (B, num_classes)
        lg_s = lg - np.max(lg, axis=1, keepdims=True)  # логиты без максимума по строке — для устойчивости exp
        lse = np.max(lg, axis=1) + np.log(np.sum(np.exp(lg_s), axis=1))  # lse = max_j lg_ij + log Σ_j exp(lg_s_ij) — логсуммаэксп по строке
        target_l = lg[np.arange(B), y_int]  # логит правильного класса target_i = lg_i,y_i
        reg_term = 0.5 * float(np.sum(reg[None, :] * (th**2)))  # штраф ½·Σ_j reg_j·Σ_c θ_cj²
        return float(np.mean(lse - target_l)) + reg_term  # CE = mean_i (lse_i − target_i) + штраф

    loss0 = ce_loss(theta)  # начальные потери
    cur_loss = loss0  # текущие потери
    accepted = held = 0  # счётчики принятых и удержанных шагов
    bf16_vec = np.vectorize(bf16_nearest_float)  # поточечное округление до BF16
    for _ in range(steps):  # до steps шагов
        lg = X_feat @ theta.T  # логиты текущих весов
        lg_s = lg - np.max(lg, axis=1, keepdims=True)  # логиты без максимума по строке
        exp_l = np.exp(lg_s)  # exp логитов
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)  # итог: softmax: p_ij = e_ij / Σ_c e_ic
        res = probs.copy()  # копия вероятностей — заготовка невязки
        res[np.arange(B), y_int] -= 1.0  # res = p − onehot(y): вычесть 1 в правильном классе
        g_bar = (res.T @ X_feat) / B + (reg[None, :] * theta)  # ḡ = res^T·X / B + reg·θ — средний градиент с регуляризацией
        D = -g_bar  # направление спуска D = −ḡ
        A = float(np.sum(g_bar * g_bar))  # A = Σ ḡ² — квадрат нормы градиента
        if A < 1e-18:  # градиент практически нулевой
            held += 1  # шаг удержан
            break  # стоп
        d_logits = X_feat @ D.T  # изменения логитов X·D^T
        r_prime = np.max(d_logits, axis=1) - np.min(d_logits, axis=1)  # размах по строке r'_i = max_j − min_j
        Q_prime = float(np.mean(r_prime * r_prime))  # итог: Q' = mean r'^2
        curv_reg = (  # кривизна регуляризации:
            2.0 * float(np.max(reg)) * float(np.sum(D * D)) if np.any(reg > 0) else 0.0  # 2·max(reg)·Σ D², если регуляризация есть, иначе 0
        )
        denom = Q_prime + 4.0 * curv_reg  # знаменатель Q' + 4·curv
        if denom <= 1e-18:  # знаменатель практически нулевой
            held += 1  # шаг удержан
            break  # стоп
        eta = 4.0 * A / denom  # длина шага η = 4A / знаменатель
        cert_drop = 2.0 * (A**2) / denom  # гарантированное убывание 2A² / знаменатель
        cand = theta + eta * D  # кандидат θ + η·D
        if use_bf16_guard:  # проверка округления BF16
            cand_bf = bf16_vec(cand)  # кандидат, округлённый до BF16
            lg_exact = X_feat @ cand.T  # точные логиты кандидата
            lg_bf = X_feat @ cand_bf.T  # логиты округлённого кандидата
            eps_pert = float(np.max(np.abs(lg_bf - lg_exact)))  # наибольшее возмущение логитов ε = max |lg_bf − lg_exact|
            if 4.0 * eps_pert >= cert_drop:  # 4ε не меньше гарантированного убывания — шаг не оправдан
                held += 1  # шаг удержан
                break  # стоп
            cand = cand_bf  # использовать округлённый кандидат
        cand_loss = ce_loss(cand)  # потери кандидата
        if cand_loss <= cur_loss + 1e-12:  # кандидат не хуже текущего (допуск 1e-12)
            theta = cand  # принять веса
            cur_loss = cand_loss  # обновить текущие потери
            accepted += 1  # принятых шагов + 1
        else:  # иначе
            held += 1  # шаг удержан
            break  # стоп
    return {  # вернуть словарь результатов
        "theta": theta,  # итоговые веса
        "loss0": loss0,  # начальные потери
        "loss_final": cur_loss,  # итоговые потери
        "total_drop": loss0 - cur_loss,  # суммарное убывание loss0 − текущие
        "accepted_steps": accepted,  # число принятых шагов
        "hold_steps": held,  # число удержанных шагов
    }


def newton_schulz_orthogonalize(G: np.ndarray, iters: int = 4) -> np.ndarray:
    """Ортогонализация Ньютона–Шульца: за iters итераций X ← X·(3E − X^T·X)/2 приближает матрицу к ортогональной.

    Матрица предварительно делится на норму Фробениуса и при необходимости
    транспонируется (строк не больше, чем столбцов); в конце вид восстанавливается."""
    X = G.astype(np.float64).copy()  # рабочая копия в float64
    norm = float(np.linalg.norm(X))  # норма Фробениуса
    if norm < 1e-12:  # норма практически нулевая
        return X  # вернуть как есть
    X /= norm  # нормировать: X = X / ‖X‖
    transposed = False  # признак транспонирования
    if X.shape[0] > X.shape[1]:  # строк больше, чем столбцов
        X = X.T  # транспонировать
        transposed = True  # запомнить
    I_m = np.eye(X.shape[0], dtype=np.float64)  # единичная матрица размера число строк
    for _ in range(iters):  # iters раз
        X = 0.5 * (3.0 * I_m - X @ X.T) @ X  # итог: X = ½·(3E − X·X^T)·X
    return X.T if transposed else X  # вернуть X^T, если транспонировали, иначе X


class ContinualGPMMemory(GpuGpmMixin):
    """Память последовательного обучения с защитой от забывания.

        Хранит веса W, базис U защищённого подпространства и вектор v фильтра
        EF21. Обучение задачи и консолидация подпространства на GPU выполняет
        hagi.gpu, если он доступен; процессорные реализации сохранены как
        методы *_cpu."""

    def __init__(self, d_in: int, d_out: int, seed: int = 0):  # d_in — входная размерность, d_out — выходная, seed — зерно генератора
        rng = np.random.default_rng(seed)  # генератор случайных чисел
        self.W = rng.normal(0.0, 0.1, size=(d_in, d_out))  # W ~ N(0, 0.1), форма (d_in, d_out)
        self.U_basis = np.zeros((d_in, 0), dtype=np.float64)  # базис защищённого подпространства U, изначально пустой (d_in, 0)
        self.ef_v = np.zeros_like(self.W)  # вектор EF21 v, изначально нулевой, той же формы, что W

    def snapshot_optimizer(self) -> dict[str, np.ndarray]:
        """Снимок состояния обучения: копии W, U_basis и ef_v."""
        return {  # словарь снимка
            "W": self.W.copy(),  # копия весов
            "U_basis": self.U_basis.copy(),  # копия базиса
            "ef_v": self.ef_v.copy(),  # копия вектора EF21
        }

    def restore_optimizer(self, snap: dict[str, np.ndarray]) -> None:
        """Восстановить W, U_basis и ef_v из снимка snap."""
        self.W = snap["W"].copy()  # веса из снимка
        self.U_basis = snap["U_basis"].copy()  # базис из снимка
        self.ef_v = snap["ef_v"].copy()  # вектор EF21 из снимка

    def project_orthogonal(self, G: np.ndarray) -> np.ndarray:
        """Спроецировать градиент G на подпространство, ортогональное базису U: (E − U·U^T)·G."""
        if self.U_basis.shape[1] == 0:  # базис пуст
            return G  # проекция не меняет G
        return G - self.U_basis @ (self.U_basis.T @ G)  # итог: G − U·(U^T·G)

    def _train_task_ef21_muon_cpu(
        self, X: np.ndarray, Y: np.ndarray, steps: int = 35, lr: float = 0.2  # self, примеры X, цели Y, число шагов и начальная длина шага
    ) -> float:  # возвращает средний квадрат ошибки
        """Обучить веса W на парах (X, Y) на процессоре; вернуть средний квадрат ошибки.

        err = X·W − Y; градиент grad = (2/N)·X^T·err проецируется ортогонально
        базису U; вектор EF21 v ← v + 0.5·(g_proj − v); направление —
        ортогонализация Ньютона–Шульца вектора v, снова спроецированная.
        Шаг W − η·step_dir принимается с первым η из (lr, lr/2, lr/10, 0.01),
        уменьшающим ошибку mean((X·W − Y)²)."""
        if len(X) == 0:  # нет примеров
            return 0.0  # ошибка 0
        for _ in range(steps):  # steps шагов
            err = X @ self.W - Y  # ошибка err = X·W − Y
            grad = (2.0 / len(X)) * (X.T @ err)  # градиент grad = (2/N)·X^T·err
            g_proj = self.project_orthogonal(grad)  # проекция градиента ортогонально базису
            diff = g_proj - self.ef_v  # разность diff = g_proj − v
            self.ef_v = self.ef_v + 0.5 * diff  # итог: v = v + 0.5·diff
            step_dir = newton_schulz_orthogonalize(self.ef_v, iters=4)  # направление шага — ортогонализация v
            step_dir = self.project_orthogonal(step_dir)  # снова спроецировать ортогонально базису
            if self.U_basis.shape[1] > 0:  # базис непуст
                drift = float(np.max(np.abs(self.U_basis.T @ step_dir)))  # максимальное смещение шага в защищённом подпространстве
                assert drift <= 1e-9, f"SpaceDecoupling violated: {drift}"  # проверка: смещение не больше 1e-9
            cur = float(np.mean((X @ self.W - Y) ** 2))  # текущая средняя квадратичная ошибка cur
            for eta in (lr, lr * 0.5, lr * 0.1, 0.01):  # по убывающим длинам шага η
                cand = self.W - eta * step_dir  # кандидат W − η·step_dir
                if float(np.mean((X @ cand - Y) ** 2)) < cur:  # ошибка кандидата меньше текущей
                    self.W = cand  # принять кандидата
                    break  # прекратить перебор η
        return float(np.mean((X @ self.W - Y) ** 2))  # вернуть средний квадрат ошибки

    def _consolidate_subspace_cpu(
        self, X: np.ndarray, energy_thresh: float = 0.995  # self, входы X и порог доли энергии
    ) -> None:  # ничего не возвращает
        """Пополнить базис U направлениями входов X, ортогональными уже защищённым.

        res_X — часть X вне span(U). Сингулярные направления берутся в порядке
        убывания сингулярных чисел, пока доля накопленной энергии
        Σ_{i≤k} S_i² / Σ S_i² не достигнет energy_thresh (при пороге ≥ 1.0 —
        все направления). Новые направления объединяются с U через SVD."""
        res_X = (  # остаток входов:
            X if self.U_basis.shape[1] == 0 else X - (X @ self.U_basis) @ self.U_basis.T  # X, если базис пуст, иначе X − (X·U)·U^T
        )
        if float(np.linalg.norm(res_X)) < 1e-10:  # остаток практически нулевой
            return  # нечего добавлять
        _, S, Vt = np.linalg.svd(res_X, full_matrices=False)  # сингулярное разложение остатка: A = U·diag(S)·Vt
        pos = S > 1e-10  # маска сингулярных чисел больше 1e-10
        if not np.any(pos):  # нет значимых сингулярных чисел
            return  # выйти
        S = S[pos]  # оставить только значимые сингулярные числа
        Vt = Vt[pos]  # и соответствующие правые сингулярные векторы
        tot = float(np.sum(S**2))  # полная энергия Σ S²
        cum = np.cumsum(S**2) / tot  # накопленная доля энергии cum_k = Σ_{i≤k} S_i² / полная
        # Накопленная сумма в float может остановиться на 0.99999…, и тогда
        # последние направления молча теряются — защищённое подпространство
        # становится неполным, и ошибка ранее обученной задачи растёт.
        # Порог energy_thresh ≥ 1.0 означает всё подпространство целиком.
        if energy_thresh >= 1.0:  # порог ≥ 1.0
            k = len(S)  # взять все направления: k = len(S)
        else:  # иначе
            k = min(len(S), int(np.searchsorted(cum, energy_thresh)) + 1)  # k = min(len(S), первый индекс cum ≥ порог + 1)
        new_dirs = Vt[:k].T  # новые направления — первые k правых сингулярных векторов как столбцы
        if self.U_basis.shape[1] == 0:  # базис пуст
            Q, _ = np.linalg.qr(new_dirs)  # ортогонализация новых направлений QR-разложением
            self.U_basis = Q  # базис = Q
        else:  # иначе
            combined = np.hstack([self.U_basis, new_dirs])  # объединить старый базис с новыми направлениями
            U_svd, s_svd, _ = np.linalg.svd(combined, full_matrices=False)  # SVD объединённой матрицы
            keep = s_svd > 1e-9  # оставить только значимые (s > 1e-9) направления
            self.U_basis = U_svd[:, keep]  # базис = соответствующие левые сингулярные векторы
        # Базис ограничен входной размерностью d_in: запас «−2» оставлял бы
        # защищённые направления без ограничения и допускал утечку забывания.
        if self.U_basis.shape[1] > self.W.shape[0]:  # столбцов больше, чем строк у W
            self.U_basis = self.U_basis[:, : self.W.shape[0]]  # обрезать базис до первых d_in столбцов


def quantize_bitalloc_exact(
    Phi: np.ndarray,  # матрица признаков (N, D)
    idx: np.ndarray,  # номера выбранных признаков
    wk: np.ndarray,  # их веса
    y: np.ndarray,  # цель
    sy: float,  # масштаб цели
    lam: float,  # L2-параметр
    max_extra_stages: int = 12,  # максимум дополнительных стадий
) -> tuple[np.ndarray, bytes]:  # возвращает квантованный вектор в размерности D и упакованные байты
    """Квантование вектора весов по выбранным признакам с точным байтовым кодеком: (вектор, байты).

    Phi — (N, D) признаки; idx — выбранные номера признаков; wk — их веса;
    y — цель; sy — масштаб цели; lam — L2-параметр. Веса сортируются по
    убыванию модуля, режутся на блоки по 5 и кодируются квантованием
    quantize_5block_exact. Затем до max_extra_stages раз: остаток
    r = y − Sub·w_q решает задачу dw = A^−1·Sub^T·r, A = Sub^T·Sub +
    max(lam, 1e-9)·E; выбирается блок с наибольшей ошибкой
    Σ dw_i²·energy_i и его поправка добавляется. Останавливается при
    mean((r·sy)²) < 0.005 или малой ошибке блока."""
    if idx.size == 0:  # признаки не выбраны
        return np.zeros(Phi.shape[1], dtype=np.float64), b"\x00\x00\x00"  # нулевой вектор и заголовок из трёх нулевых байтов
    mag_order = np.argsort(-np.abs(wk))  # порядок убывания модуля весов
    idx_sorted = idx[mag_order]  # номера признаков в этом порядке
    Sub_s = Phi[:, idx_sorted]  # подматрица признаков Sub_s = Phi[:, idx_sorted]
    col_energy = np.mean(Sub_s**2, axis=0)  # средняя энергия столбца energy_j = mean_i Sub_s_ij²
    A = Sub_s.T @ Sub_s + max(lam, 1e-9) * np.eye(idx.size)  # итог: A = Sub_s^T·Sub_s + max(lam, 1e-9)·E
    k = idx.size  # k — число выбранных признаков
    n_blocks = (k + 4) // 5  # число блоков ⌈k / 5⌉
    wq_s = np.zeros(k, dtype=np.float64)  # квантованные веса по k признакам, изначально нули
    dw = wk[mag_order].copy()  # рабочая копия весов dw в порядке убывания модуля
    records: list[tuple[int, tuple[int, int, int]]] = []  # список записей (номер блока, (exp2, best, bval))
    for b in range(n_blocks):  # по блокам
        sl = slice(b * 5, min((b + 1) * 5, k))  # срез блока: [5b, min(5(b + 1), k))
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))  # блок, дополненный нулями до 5 элементов
        q_blk, meta = quantize_5block_exact(blk)  # квантовать блок
        wq_s[sl] += q_blk[: sl.stop - sl.start]  # прибавить квантованные веса (без дополнения)
        records.append((b, meta))  # запомнить запись блока
    for _ in range(max_extra_stages):  # до max_extra_stages дополнительных стадий
        r_data = y - Sub_s @ wq_s  # остаток данных r = y − Sub_s·w_q
        if float(np.mean((r_data * sy) ** 2)) < 0.005:  # масштабная средняя мощность остатка < 0.005 — стоп
            break  # выйти из цикла стадий
        dw = np.linalg.solve(A, Sub_s.T @ r_data)  # поправка dw = A^−1·(Sub_s^T·r)
        blk_errs = [  # ошибки блоков:
            float(  # ошибка блока b как число
                np.sum(  # сумма по блоку
                    (dw[b * 5 : min((b + 1) * 5, k)] ** 2)  # квадратов поправок dw_i²
                    * col_energy[b * 5 : min((b + 1) * 5, k)]  # умноженных на энергию столбца energy_i
                )
            )
            for b in range(n_blocks)  # по блокам
        ]
        best_b = int(np.argmax(blk_errs))  # номер блока с наибольшей ошибкой
        if blk_errs[best_b] * (sy * sy) < 1e-6:  # ошибка блока в масштабе цели < 1e-6 — стоп
            break  # выйти из цикла стадий
        sl = slice(best_b * 5, min((best_b + 1) * 5, k))  # срез лучшего блока
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))  # блок, дополненный нулями до 5 элементов
        q_blk, meta = quantize_5block_exact(blk)  # квантовать блок
        wq_s[sl] += q_blk[: sl.stop - sl.start]  # прибавить поправку к квантованным весам
        records.append((best_b, meta))  # запомнить запись блока
    packed = pack_bitalloc_records(k, idx_sorted, records)  # упаковать k, номера признаков и записи
    rec = unpack_bitalloc_records(packed, Phi.shape[1])  # раскодировать обратно в размерность Phi для проверки
    return rec, packed  # вернуть раскодированный вектор и упакованные байты


def _omp_candidates(
    Phi: np.ndarray, y: np.ndarray, max_k: int = 16, lam: float = 1e-7  # Phi — (N, D), максимум признаков, L2-параметр
) -> list[tuple[np.ndarray, np.ndarray]]:  # возвращает список пар (номера признаков, веса)
    """Кандидаты жадного выбора признаков (пошаговое ортогональное согласование): список (idx, wk).

    На каждом шаге берётся нормированный столбец Phi_n с наибольшей
    |корреляцией| с остатком r; веса wk = A^−1·Sub^T·y, A = Sub^T·Sub +
    lam·E; остаток r = y − Sub·wk. Не более max_k шагов; стоп при
    корреляции ≤ 1e-12 или mean(r²) < 1e-9."""
    N, D = Phi.shape  # N, D — размеры матрицы признаков
    norms = np.sqrt(np.sum(Phi**2, axis=0)) + 1e-12  # нормы столбцов + 1e-12 для защиты от нуля
    Phi_n = Phi / norms[None, :]  # нормированные признаки Phi_n = Phi / нормы
    r = y.copy()  # остаток r = y
    selected: list[int] = []  # выбранные номера признаков по порядку
    sel_set: set[int] = set()  # множество выбранных
    cands: list[tuple[np.ndarray, np.ndarray]] = []  # список кандидатов
    for _ in range(min(max_k, D, max(1, N - 1))):  # не более min(max_k, D, max(1, N − 1)) шагов
        corrs = np.abs(Phi_n.T @ r)  # модули корреляций |Phi_n^T·r|
        for s in sel_set:  # по уже выбранным
            corrs[s] = -1.0  # исключить: корреляция −1
        j = int(np.argmax(corrs))  # номер столбца с наибольшей корреляцией
        if corrs[j] <= 1e-12:  # корреляция практически нулевая
            break  # стоп
        selected.append(j)  # добавить признак в порядок
        sel_set.add(j)  # в множество
        idx = np.array(selected, dtype=np.int64)  # массив выбранных номеров
        Sub = Phi[:, idx]  # подматрица выбранных признаков
        A = Sub.T @ Sub + lam * np.eye(len(idx))  # итог: A = Sub^T·Sub + lam·E
        wk = np.linalg.solve(A, Sub.T @ y)  # итог: wk = A^−1·(Sub^T·y)
        r = y - Sub @ wk  # новый остаток r = y − Sub·wk
        cands.append((idx.copy(), wk.copy()))  # запомнить кандидата (копии idx и wk)
        if float(np.mean(r * r)) < 1e-9:  # средний квадрат остатка < 1e-9
            break  # стоп
    return cands  # вернуть список кандидатов


def _press_score_ridge(
    Phi_sub: np.ndarray,  # подматрица выбранных признаков
    wk: np.ndarray,  # веса ридж-регрессии
    y: np.ndarray,  # целевая переменная в исходном масштабе
    ty: np.ndarray,  # целые цели
    sy: float,  # масштаб цели
    lam: float,  # L2-параметр
) -> tuple[float, float]:  # возвращает (ошибка оставь один, точность)
    """Оценка модели по «оставь один» (PRESS) через диагональ матрицы проекции: (ошибка, точность).

    wk — ридж-веса (A = Phi_sub^T·Phi_sub + max(lam, 1e-8)·E);
    ty — целые цели; sy — масштаб цели. h_i = diag(Phi_sub·A^−1·Phi_sub^T);
    прогноз без i-го примера loo_i = ty_i − resid_i / max(0.08, 1 − h_i),
    resid = ty − Phi_sub·wk·sy. Ошибка = mean((loo − ty)²) + 1e-3·Σwk²·sy² / max(1, N);
    точность = доля round(loo) = ty."""
    N, D = Phi_sub.shape  # N, D — размеры подматрицы признаков
    pred = (Phi_sub @ wk) * sy  # прогноз (Phi_sub·wk)·sy
    resid = ty - pred  # невязка resid = ty − прогноз
    if D >= N:  # признаков не меньше, чем примеров — оценка оставь один невозможна
        return float(np.mean(resid**2)) * 100.0 + float(np.sum(wk**2)), 0.0  # штраф 100·mean(resid²) + Σwk², точность 0
    A = Phi_sub.T @ Phi_sub + max(lam, 1e-8) * np.eye(D)  # итог: A = Phi_sub^T·Phi_sub + max(lam, 1e-8)·E
    try:  # решение может не существовать
        A_inv_Pt = np.linalg.solve(A, Phi_sub.T)  # итог: A^−1·Phi_sub^T
    except np.linalg.LinAlgError:  # вырожденная матрица
        return float(np.mean(resid**2)) * 100.0, 0.0  # штраф 100·mean(resid²), точность 0
    h = np.sum(Phi_sub * A_inv_Pt.T, axis=1)  # итог: h_i = diag(Phi_sub·A^−1·Phi_sub^T) = Σ_j Phi_ij·(A^−1·Phi^T)_ji
    denom = np.maximum(0.08, 1.0 - h)  # знаменатель max(0.08, 1 − h_i)
    loo_pred = ty - resid / denom  # loo = ty − resid / знаменатель
    w_norm_pen = 1e-3 * float(np.sum(wk**2)) * (sy**2) / max(1, N)  # штраф за веса 1e-3·Σwk²·sy² / max(1, N)
    loo_mse = float(np.mean((loo_pred - ty) ** 2)) + w_norm_pen  # средний квадрат ошибки loo + штраф
    loo_acc = float(np.mean(np.rint(loo_pred) == ty))  # доля точных округлённых прогнозов
    return loo_mse, loo_acc  # вернуть ошибку и точность


class FactorizedSharedCore:
    """Разделяемое ядро малого объёма: случайные признаки (Адамар + SwiGLU) и память GPM.

    Ветви: poly — мономы; hinge — ломаные признаки; bit — биты и их
    произведения; core — нелинейные признаки нейронного слоя. Веса слоёв
    строятся из ортогональных матриц, поэтому ядро занимает 8 байт — зерно
    генератора и ширина."""

    BRANCHES = ("poly", "hinge", "bit", "core")  # имена ветвей признаков: полиномиальная, ломаная, битовая, ядро

    def __init__(self, seed: int = 42, width: int = 48):  # seed — зерно, width — ширина слоя
        self.seed = seed  # запомнить зерно
        self.width = width  # запомнить ширину
        rng = np.random.default_rng(seed)  # генератор случайных чисел
        Q1, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, 8)))  # Q1 — ортогональная матрица (width, 8) из QR-разложения случайной
        scales = np.repeat([0.6, 1.2, 2.4, 4.8], width // 4)  # масштабы 0.6, 1.2, 2.4, 4.8, повторённые width/4 раз
        self.W1 = Q1.T * scales[None, :]  # W1 = Q1^T·diag(масштабы), форма (8, width)
        self.b1 = rng.uniform(-1.5, 1.5, (1, width))  # b1 — смещение, равномерное на [−1.5, 1.5], форма (1, width)
        Q2, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, width)))  # Q2 — ортогональная матрица (width, width)
        self.W2a = Q2 * 1.3  # W2a = Q2·1.3 — вес ветви значения SwiGLU
        self.b2a = rng.uniform(-0.8, 0.8, (1, width))  # b2a — её смещение, равномерное на [−0.8, 0.8]
        Q2g, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, width)))  # Q2g — вторая ортогональная матрица (width, width)
        self.W2g = Q2g * 1.3  # W2g = Q2g·1.3 — вес ветви gating SwiGLU
        self.b2g = rng.uniform(-0.8, 0.8, (1, width))  # b2g — её смещение, равномерное на [−0.8, 0.8]
        self.H8 = sylvester_hadamard(3)  # H8 — нормированная матрица Адамара порядка 8
        self.gpm = ContinualGPMMemory(d_in=16, d_out=1, seed=seed + 7)  # память GPM на 16 входов и 1 выход
        self.core_bytes = 8  # объём ядра — 8 байт

    def snapshot(self) -> dict[str, Any]:
        """Снимок состояния ядра: снимок памяти GPM и объём ядра."""
        return {"gpm": self.gpm.snapshot_optimizer(), "core_bytes": self.core_bytes}  # словарь снимка

    def restore(self, snap: dict[str, Any]) -> None:
        """Восстановить ядро из снимка snap."""
        self.gpm.restore_optimizer(snap["gpm"])  # восстановить память GPM
        self.core_bytes = snap["core_bytes"]  # восстановить объём ядра

    def features(self, branch: str, xs_arr: np.ndarray, sx: float) -> np.ndarray:
        """Матрица признаков ветви branch для входов xs_arr при масштабе sx.

        branch — одна из BRANCHES. Возвращает (N, число признаков) float64."""
        N, d = xs_arr.shape  # N — число входов, d — число аргументов
        xn = xs_arr.astype(np.float64) / sx  # нормированные входы xn = xs / sx
        step1 = 1.0 / sx  # шаг одного целого в нормированной шкале 1 / sx
        if branch == "poly":  # ветвь poly — мономы
            cols = [xn[:, i] for i in range(d)]  # столбцы-переменные
            fs = [np.ones((N, 1), dtype=np.float64)]  # начать с столбца единиц
            for i in range(d):  # по переменным
                fs.append(cols[i][:, None])  # добавить линейный член x_i
            for i in range(d):  # по парам i ≤ j
                for j in range(i, d):  # по j от i до d−1
                    fs.append((cols[i] * cols[j])[:, None])  # добавить квадратичный член x_i·x_j
            for i in range(d):  # по тройкам i ≤ j ≤ k
                for j in range(i, d):  # по j
                    for k in range(j, d):  # по k
                        fs.append((cols[i] * cols[j] * cols[k])[:, None])  # добавить кубический член x_i·x_j·x_k
            if d <= 2:  # арность не больше 2 — добавить и четвёртые степени
                for i in range(d):  # по i
                    for j in range(i, d):  # по j
                        for k in range(j, d):  # по k
                            for l in range(k, d):  # по l
                                fs.append(  # добавить член
                                    (cols[i] * cols[j] * cols[k] * cols[l])[:, None]  # итог: x_i·x_j·x_k·x_l
                                )
            else:  # иначе
                for i in range(d):  # по переменным
                    fs.append((cols[i] ** 4)[:, None])  # добавить только x_i^4
            return np.hstack(fs)  # соединить столбцы в матрицу
        if branch == "hinge":  # ветвь hinge — ломаные признаки
            fs = [np.ones((N, 1), dtype=np.float64)]  # начать с столбца единиц
            for i in range(d):  # по переменным
                fs.append(xn[:, i : i + 1])  # добавить x_i
                fs.append(np.abs(xn[:, i : i + 1]))  # добавить |x_i|
                for knot in (-0.375, -step1, 0.0, step1, 0.375):  # по узлам излома −0.375, −шаг, 0, шаг, 0.375
                    fs.append(np.maximum(0.0, xn[:, i] - knot)[:, None])  # добавить max(0, x_i − узел)
                    fs.append(np.maximum(0.0, knot - xn[:, i])[:, None])  # добавить max(0, узел − x_i)
                fs.append(np.maximum(0.0, 1.0 - sx * np.abs(xn[:, i : i + 1])))  # добавить max(0, 1 − sx·|x_i|) — индикатор близости к целому
            for i in range(d):  # по парам разных переменных i < j
                for j in range(i + 1, d):  # по j > i
                    diff = xn[:, i] - xn[:, j]  # итог: diff = x_i − x_j
                    fs.append(np.abs(diff)[:, None])  # добавить |diff|
                    for knot in (-step1, 0.0, step1):  # по узлам −шаг, 0, шаг
                        fs.append(np.maximum(0.0, diff - knot)[:, None])  # добавить max(0, diff − узел)
                        fs.append(np.maximum(0.0, knot - diff)[:, None])  # добавить max(0, узел − diff)
                    fs.append(np.maximum(0.0, 1.0 - sx * np.abs(diff))[:, None])  # добавить max(0, 1 − sx·|diff|)
            return np.hstack(fs)  # соединить столбцы в матрицу
        if branch == "bit":  # ветвь bit — битовые признаки
            xi = np.rint(xs_arr).astype(np.int64)  # входы, округлённые до целых
            fs = [np.ones((N, 1), dtype=np.float64)]  # начать с столбца единиц
            for i in range(d):  # по переменным
                fs.append((xi[:, i] < 0).astype(np.float64)[:, None])  # добавить признак x_i < 0
                for b in range(8):  # по битам b = 0..7
                    fs.append(((xi[:, i] >> b) & 1).astype(np.float64)[:, None])  # добавить бит b числа x_i: (x_i >> b) & 1
            if d >= 2:  # арность не меньше 2 — парные битовые признаки
                for b in range(6):  # по младшим 6 битам
                    b0 = ((xi[:, 0] >> b) & 1).astype(np.float64)[:, None]  # бит b первого аргумента
                    b1 = ((xi[:, 1] >> b) & 1).astype(np.float64)[:, None]  # бит b второго аргумента
                    fs.append(b0 * b1)  # добавить И: b0·b1
                    fs.append(np.abs(b0 - b1))  # добавить XOR: |b0 − b1|
                fs.append(((xi[:, 0] < 0) * (xi[:, 1] < 0)).astype(np.float64)[:, None])  # добавить «оба отрицательные»
                fs.append(((xi[:, 0] < 0) ^ (xi[:, 1] < 0)).astype(np.float64)[:, None])  # добавить «в точности одно отрицательное»
            if d == 3:  # арность 3 — произведения младших бит
                b0 = (xi[:, 0] & 1).astype(np.float64)[:, None]  # младший бит x_0
                b1 = (xi[:, 1] & 1).astype(np.float64)[:, None]  # младший бит x_1
                b2 = (xi[:, 2] & 1).astype(np.float64)[:, None]  # младший бит x_2
                fs.extend([b0 * b2, b1 * b2, b0 * b1 * b2])  # добавить b0·b2, b1·b2, b0·b1·b2
            return np.hstack(fs)  # соединить столбцы в матрицу
        xp = np.zeros((N, 8), dtype=np.float64)  # ветвь core — восемь смесей входов xp, изначально нули
        xp[:, : min(3, d)] = xn[:, : min(3, d)]  # первые min(3, d) компонент — сами переменные
        xp[:, 3] = xn[:, 0] - xn[:, 1] if d >= 2 else np.abs(xn[:, 0])  # xp_3 = x_0 − x_1 при d ≥ 2, иначе |x_0|
        xp[:, 4] = xn[:, 1] - xn[:, 2] if d >= 3 else np.maximum(0.0, xn[:, 0])  # xp_4 = x_1 − x_2 при d ≥ 3, иначе max(0, x_0)
        xp[:, 5] = xn[:, 0] - xn[:, 2] if d >= 3 else np.maximum(0.0, -xn[:, 0])  # xp_5 = x_0 − x_2 при d ≥ 3, иначе max(0, −x_0)
        xp[:, 6] = np.abs(xp[:, 3])  # итог: xp_6 = |xp_3|
        xp[:, 7] = np.abs(xp[:, 4])  # итог: xp_7 = |xp_4|
        xh = xp @ self.H8  # поворот Адамара: xh = xp·H8
        z1 = xh @ self.W1 + self.b1  # z1 = xh·W1 + b1, форма (N, width)
        h1_t = np.tanh(z1)  # ветвь tanh: tanh(z1)
        h1_r = np.maximum(0.0, z1)  # ветвь ReLU: max(0, z1)
        z2a = h1_r @ self.W2a + self.b2a  # линейный слой значения: z2a = h1_r·W2a + b2a, форма (N, width)
        z2g = h1_r @ self.W2g + self.b2g  # линейный слой вентиля: z2g = h1_r·W2g + b2g
        sig2g = 1.0 / (1.0 + np.exp(-np.clip(z2g, -15.0, 15.0)))  # сигмоида с обрезкой аргумента: sig2g = 1 / (1 + exp(−clip(z2g, −15, 15)))
        h2_sg = z2a * sig2g  # вентилируемый слой: h2_sg = z2a·sig2g
        four = []  # пустой список гармонических признаков
        for f in (1.0, 2.0, 4.0):  # по частотам f = 1, 2, 4
            for i in range(min(3, d)):  # по первым min(3, d) переменным
                four.append(np.sin(math.pi * f * xn[:, i : i + 1]))  # добавить sin(π·f·x_i)
                four.append(np.cos(math.pi * f * xn[:, i : i + 1]))  # добавить cos(π·f·x_i)
        return np.hstack(  # вернуть конкатенацию всех блоков признаков по столбцам
            [  # начало списка блоков
                self.features("poly", xs_arr, sx),  # полиномиальные признаки
                self.features("hinge", xs_arr, sx)[:, 1:],  # ломаные признаки без столбца единиц
                h1_t,  # tanh-ветвь скрытого слоя
                h1_r,  # ReLU-ветвь скрытого слоя
                h2_sg,  # вентилируемый второй слой
            ]  # конец списка блоков
            + four  # плюс гармонические признаки
        )

    def _mlp_encode(self, A: np.ndarray, sx: float) -> np.ndarray:
        """Матрица входов корректора-перцептрона по строкам A.

            A: (N, d) целочисленные аргументы, sx: масштаб входа.
            Столбцы: a_i = A_i / sx для каждой переменной, |a_i|, столбец единиц,
            а при d ≥ 2 для всех пар i < j — разности a_i − a_j и |a_i − a_j|.
            Возвращает (N, 2d + 1 + d(d−1)) float64."""
        an = A / sx  # нормировка: an = A / sx
        d = A.shape[1]  # d — число переменных (арность)
        cols = [an, np.abs(an), np.ones((A.shape[0], 1))]  # начать с a_i, |a_i| и столбца единиц
        if d >= 2:  # парные признаки нужны только при d ≥ 2
            for i in range(d):  # по переменным i
                for j in range(i + 1, d):  # по j > i
                    cols.append(an[:, i : i + 1] - an[:, j : j + 1])  # добавить разность a_i − a_j
                    cols.append(np.abs(an[:, i : i + 1] - an[:, j : j + 1]))  # добавить |a_i − a_j|
        return np.hstack(cols)  # соединить столбцы в матрицу (N, число признаков)

    def _fit_stage2_mlp(
        self,  # экземпляр
        tx: np.ndarray,  # (N, d) обучающие аргументы
        resid_n: np.ndarray,  # (N,) остаток в нормированных единицах y / sy
        vx: np.ndarray,  # (M, d) проверочные аргументы
        sx: float,  # масштаб входа
        hidden: int = 12,  # ширина скрытого слоя по умолчанию 12
        steps: int = 280,  # число шагов обучения по умолчанию 280
        seed: int = 7,  # зерно генератора по умолчанию 7
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:  # возвращает (прогноз на train, прогноз на val, словарь весов)
        """Обучить двухслойный перцептрон, предсказывающий остаток основной модели.

            Сеть: h = tanh(X·W1 + b1), pred = h·W2 + b2, потеря MSE = mean((pred − r)^2).
            tx: (N, d) обучающие аргументы, resid_n: (N,) остаток в нормированных
            единицах y / sy, vx: (M, d) проверочные аргументы, sx: масштаб входа,
            hidden: ширина скрытого слоя, steps: число шагов, seed: зерно генератора.
            W1 обновляется методом MUON (момент, ортогонализация Ньютона–Шульца,
            шаг 0.7·lr), W2, b1, b2 — Adam. Шаг lr затухает по косинусу:
            lr_t = lr0·0.5·(1 + cos(π·t/steps)), lr0 = 0.04. Хранится лучший по MSE
            набор весов. В конце все веса слитно поворачиваются Адамаром, квантуются
            dyadic5 и разворачиваются обратно. Возвращает (pred_train, pred_val,
            словарь квантованных весов и упакованных байтов)."""
        rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
        Xin = self._mlp_encode(tx, sx)  # матрица входов обучающей выборки
        Xvin = self._mlp_encode(vx, sx)  # матрица входов проверочной выборки
        N, din = Xin.shape  # N — число примеров, din — число входных признаков
        W1 = rng.normal(0.0, 1.0 / math.sqrt(din), (din, hidden))  # W1 ~ N(0, 1/√din), форма (din, hidden)
        b1 = np.zeros((1, hidden))  # смещение скрытого слоя — нули
        W2 = np.zeros((hidden, 1))  # веса выхода — нули (первый прогноз равен 0)
        b2 = np.zeros((1, 1))  # смещение выхода — нули
        yn = resid_n.reshape(-1, 1)  # целевой остаток в столбец (N, 1)
        m1 = np.zeros_like(W1)  # момент первого порядка для W1 (нулевая матрица той же формы)
        m2 = np.zeros_like(W2)  # момент первого порядка для W2
        v2 = np.zeros_like(W2)  # момент второго порядка для W2
        mb1 = np.zeros_like(b1)  # момент первого порядка для b1
        vb1 = np.zeros_like(b1)  # момент второго порядка для b1
        mb2 = np.zeros_like(b2)  # момент первого порядка для b2
        vb2 = np.zeros_like(b2)  # момент второго порядка для b2
        best = (float(np.mean(yn**2)), W1.copy(), b1.copy(), W2.copy(), b2.copy())  # старт: MSE при нулевом прогнозе = mean(yn^2) и нулевые веса
        lr0 = 0.04  # начальная скорость обучения lr0
        for t in range(1, steps + 1):  # по шагам t = 1..steps
            cos_lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * t / steps))  # косинусное затухание: cos_lr = lr0·0.5·(1 + cos(π·t/steps))
            h = np.tanh(Xin @ W1 + b1)  # скрытый слой: h = tanh(Xin·W1 + b1)
            pred = h @ W2 + b2  # прогноз: pred = h·W2 + b2
            err = pred - yn  # ошибка: err = pred − yn
            loss = float(np.mean(err**2))  # потеря: loss = mean(err^2)
            if loss < best[0]:  # потеря меньше рекордной
                best = (loss, W1.copy(), b1.copy(), W2.copy(), b2.copy())  # запомнить рекорд и копии весов
            dp = (2.0 / N) * err  # градиент по pred: dp = (2/N)·err
            gW2 = h.T @ dp + 2e-4 * W2  # градиент W2: hᵀ·dp + 2e-4·W2 (L2-регуляризация)
            gb2 = np.sum(dp, axis=0, keepdims=True)  # градиент b2: Σ_n dp_n
            dh = (dp @ W2.T) * (1.0 - h * h)  # градиент по h: dp·W2ᵀ, умножённый на производную tanh 1 − h^2
            gW1 = Xin.T @ dh + 2e-4 * W1  # градиент W1: Xinᵀ·dh + 2e-4·W1
            gb1 = np.sum(dh, axis=0, keepdims=True)  # градиент b1: Σ_n dh_n
            m1 = 0.85 * m1 + 0.15 * gW1  # момент MUON: m1 = 0.85·m1 + 0.15·gW1
            W1 -= (cos_lr * 0.7) * newton_schulz_orthogonalize(m1, iters=3)  # W1 -= 0.7·cos_lr·orth(m1), orth — ортогонализация Ньютона–Шульца (3 итерации)
            m2 = 0.9 * m2 + 0.1 * gW2  # метод Adam: m2 = 0.9·m2 + 0.1·gW2
            v2 = 0.999 * v2 + 0.001 * (gW2 * gW2)  # метод Adam: v2 = 0.999·v2 + 0.001·gW2^2
            W2 -= cos_lr * (m2 / (1 - 0.9**t)) / (np.sqrt(v2 / (1 - 0.999**t)) + 1e-8)  # метод Adam: W2 -= cos_lr·(m2/(1−0.9^t)) / (√(v2/(1−0.999^t)) + 1e-8)
            mb1 = 0.9 * mb1 + 0.1 * gb1  # метод Adam: mb1 = 0.9·mb1 + 0.1·gb1
            vb1 = 0.999 * vb1 + 0.001 * (gb1 * gb1)  # метод Adam: vb1 = 0.999·vb1 + 0.001·gb1^2
            b1 -= cos_lr * (mb1 / (1 - 0.9**t)) / (np.sqrt(vb1 / (1 - 0.999**t)) + 1e-8)  # метод Adam: b1 -= cos_lr·(mb1/(1−0.9^t)) / (√(vb1/(1−0.999^t)) + 1e-8)
            mb2 = 0.9 * mb2 + 0.1 * gb2  # метод Adam: mb2 = 0.9·mb2 + 0.1·gb2
            vb2 = 0.999 * vb2 + 0.001 * (gb2 * gb2)  # метод Adam: vb2 = 0.999·vb2 + 0.001·gb2^2
            b2 -= cos_lr * (mb2 / (1 - 0.9**t)) / (np.sqrt(vb2 / (1 - 0.999**t)) + 1e-8)  # метод Adam: b2 -= cos_lr·(mb2/(1−0.9^t)) / (√(vb2/(1−0.999^t)) + 1e-8)
        _, W1, b1, W2, b2 = best  # взять лучший набор весов
        all_w = np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])  # все веса одним вектором
        wr, orig_n = block_hadamard_rotate(all_w, 8)  # повёрнутый вектор wr (блоки по 8) и исходная длина orig_n
        packed_mlp, wr_q = pack_vector_dyadic5(wr, stages=2)  # квантование dyadic5 (2 стадии): байты packed_mlp и квантованный вектор wr_q
        all_wq = block_hadamard_unrotate(wr_q, orig_n, 8)  # обратный поворот и обрезка до orig_n: квантованные веса
        p1 = W1.size  # граница W1: p1 = число элементов W1
        p2 = p1 + b1.size  # граница b1: p2 = p1 + число элементов b1
        p3 = p2 + W2.size  # граница W2: p3 = p2 + число элементов W2
        W1q = all_wq[:p1].reshape(W1.shape)  # W1q — первые p1 элементов в форме W1
        b1q = all_wq[p1:p2].reshape(b1.shape)  # b1q — срез p1..p2−1 в форме b1
        W2q = all_wq[p2:p3].reshape(W2.shape)  # W2q — срез p2..p3−1 в форме W2
        b2q = all_wq[p3:].reshape(b2.shape)  # b2q — остаток с p3 в форме b2
        return (  # вернуть прогнозы и веса
            (np.tanh(Xin @ W1q + b1q) @ W2q + b2q).ravel(),  # прогноз на train: tanh(Xin·W1q + b1q)·W2q + b2q, развёрнутый в вектор
            (np.tanh(Xvin @ W1q + b1q) @ W2q + b2q).ravel(),  # прогноз на val: tanh(Xvin·W1q + b1q)·W2q + b2q, развёрнутый в вектор
            {"W1": W1q, "b1": b1q, "W2": W2q, "b2": b2q, "packed": packed_mlp},  # словарь квантованных весов и упакованных байтов
        )

    def fit_residual(
        self,  # экземпляр
        train_x: Sequence[Sequence[int]],  # обучающие аргументы (строки целых)
        train_y: Sequence[int],  # обучающие значения
        val_x: Sequence[Sequence[int]],  # проверочные аргументы
        val_y: Sequence[int],  # проверочные значения
        router: Optional["HedgeRouter"] = None,  # маршрутизатор ветвей (задаёт порядок перебора и получает потери)
    ) -> "FactorizedAdapter":  # возвращает обученный адаптер
        """Обучить компактный адаптер, предсказывающий y по целочисленным аргументам.
        
                train_x/train_y: обучающие пары, val_x/val_y: проверочные пары (пустые — берутся
                обучающие), router: маршрутизатор ветвей. Масштабы: sx = max(1, max|train_x|),
                sy = max(1, max|train_y|), y = train_y / sy.
        
                Перебираются ветви признаков Phi. Для трёх λ ∈ {1e-6, 1e-3, 1e-1} отбор OMP
                даёт не более max_k весов wk; кандидат с max|wk| > 8 отбрасывается. Цель:
                f = 0.4·mse_train + 0.6·mse_loo + 0.002·nb + 0.5·(1 − 0.5·(acc_train + acc_loo)),
                где mse_loo/acc_loo — оценка leave-one-out по PRESS, nb = 9 + k + 3·⌈k/5⌉,
                k — число признаков. У трёх лучших кандидатов веса квантуются (nb = 6 + длина packed).
        
                При 2..5 классах и N ≥ 12 дополнительно обучается классификатор с перекрёстной
                энтропией на не более 4 ломаных признаках. Если регрессия неточна, на её остатке
                обучается корректор-перцептрон (режим rvq2, если он снижает проверочную MSE
                более чем на 28%). Возвращает FactorizedAdapter с метриками на val."""
        tx = np.asarray(train_x, dtype=np.float64)  # обучающие аргументы в массив float64
        ty = np.asarray(train_y, dtype=np.float64)  # обучающие значения в массив float64
        vx = np.asarray(val_x, dtype=np.float64) if val_x else tx  # проверочные аргументы; если пусто — обучающие
        vy = np.asarray(val_y, dtype=np.float64) if val_y else ty  # проверочные значения; если пусто — обучающие
        sx = max(1.0, float(np.max(np.abs(tx))))  # масштаб входа: sx = max(1, max|tx|)
        sy = max(1.0, float(np.max(np.abs(ty))))  # масштаб выхода: sy = max(1, max|ty|)
        y = ty / sy  # нормированная цель: y = ty / sy
        N, d = tx.shape  # N — число примеров, d — арность

        gpm_in = self.features("poly", tx, sx)  # полиномиальные признаки обучающей выборки
        if gpm_in.shape[1] < 16:  # столбцов меньше 16
            gpm_in = np.pad(gpm_in, ((0, 0), (0, 16 - gpm_in.shape[1])))  # дополнить нулями справа до 16 столбцов
        else:  # столбцов не меньше 16
            gpm_in = gpm_in[:, :16]  # взять первые 16 столбцов
        self.gpm.train_task_ef21_muon(gpm_in, y.reshape(-1, 1), steps=15, lr=0.15)  # обучить гейт-модель gpm: 15 шагов, скорость 0.15
        self.gpm.consolidate_subspace(gpm_in, energy_thresh=0.995)  # закрепить подпространство, объясняющее 99.5% энергии

        branch_order = list(self.BRANCHES)  # порядок перебора ветвей — исходный список
        if router is not None:  # маршрутизатор задан
            branch_order = [self.BRANCHES[i] for i in np.argsort(-router.weights)]  # ветви по убыванию веса: argsort(−weights)

        best = (float("inf"), "poly", "reg", None, None, b"", None, None, None)  # рекорд: (цель, ветвь, режим, веса, байты, байты, метаданные CE, прогноз train/sy, метаданные MLP)
        branch_losses = np.ones(len(self.BRANCHES), dtype=np.float64)  # потери ветвей, изначально 1

        for bname in branch_order:  # по ветвям в выбранном порядке
            Phi = self.features(bname, tx, sx)  # матрица признаков Phi (N, число признаков)
            max_k = min(14, max(2, N // 2) if N >= 20 else max(1, min(8, N - 2)))  # предел числа признаков: min(14, max(2, N/2)) при N ≥ 20, иначе min(8, max(1, N − 2))
            b_best_acc = 0.0  # лучшая LOO-точность ветви, изначально 0
            float_cands = []  # список кандидатов с вещественными весами
            for lam in (1e-6, 1e-3, 1e-1):  # по коэффициентам регуляризации λ
                for idx, wk in _omp_candidates(Phi, y, max_k=max_k, lam=lam):  # по парам (набор признаков idx, веса wk) от OMP
                    if np.max(np.abs(wk)) > 8.0:  # наибольший |вес| больше 8
                        continue  # отклонить кандидата
                    pred_f = (Phi[:, idx] @ wk) * sy  # прогноз: pred_f = (Phi_idx·wk)·sy
                    mse_tr_f = float(np.mean((pred_f - ty) ** 2))  # MSE на train: mean((pred_f − ty)^2)
                    acc_tr_f = float(np.mean(np.rint(pred_f) == ty))  # точность на train: доля round(pred_f) = ty
                    mse_loo, acc_loo = _press_score_ridge(  # оценка leave-one-out по PRESS: MSE и точность
                        Phi[:, idx], wk, y, ty, sy, lam=lam  # по выбранным признакам, весам и λ
                    )
                    b_best_acc = max(b_best_acc, acc_loo)  # обновить лучшую LOO-точность ветви
                    est_nb = 9 + len(idx) + ((len(idx) + 4) // 5) * 3  # оценка байтов: 9 + k + 3·⌈k/5⌉, k = len(idx)
                    f_obj = (  # цель кандидата
                        0.4 * mse_tr_f  # 0.4·MSE на train
                        + 0.6 * mse_loo  # + 0.6·MSE leave-one-out (оценка по одному)
                        + 0.002 * est_nb  # + 0.002·оценка байтов
                        + (1.0 - 0.5 * (acc_tr_f + acc_loo)) * 0.5  # + 0.5·(1 − 0.5·(точность train + точность leave-one-out))
                    )
                    float_cands.append((f_obj, lam, idx, wk, mse_loo, acc_loo))  # сохранить кандидата
            float_cands.sort(key=lambda c: c[0])  # отсортировать кандидатов по возрастанию цели
            for _, lam, idx, wk, mse_loo, acc_loo in float_cands[:3]:  # по трём лучшим кандидатам
                wq, packed = quantize_bitalloc_exact(Phi, idx, wk, y, sy, lam)  # квантование с распределением бит: веса wq и упакованные байты
                pred_q = (Phi @ wq) * sy  # прогноз квантованной модели: pred_q = (Phi·wq)·sy
                mse_tr = float(np.mean((pred_q - ty) ** 2))  # MSE на train: mean((pred_q − ty)^2)
                acc_tr = float(np.mean(np.rint(pred_q) == ty))  # точность на train: доля round(pred_q) = ty
                nb = 6 + len(packed)  # число байтов: nb = 6 + длина packed
                obj = (  # цель квантованного кандидата
                    0.4 * mse_tr  # 0.4·MSE на train
                    + 0.6 * mse_loo  # + 0.6·MSE leave-one-out (оценка по одному) (вещественных весов)
                    + 0.002 * nb  # + 0.002·число байтов
                    + (1.0 - 0.5 * (acc_tr + acc_loo)) * 0.5  # + 0.5·(1 − 0.5·(точность train + точность leave-one-out))
                )
                if obj < best[0]:  # цель меньше рекордной
                    best = (  # новый рекорд
                        obj,  # цель
                        bname,  # ветвь признаков
                        "reg",  # режим — регрессия
                        wq,  # квантованные веса
                        packed,  # упакованные байты
                        packed,  # упакованные байты (второй слот)
                        None,  # метаданных CE нет
                        pred_q / sy,  # прогноз train в нормированных единицах
                        None,  # метаданных MLP нет
                    )
            b_idx = self.BRANCHES.index(bname)  # номер ветви в списке BRANCHES
            branch_losses[b_idx] = 1.0 - b_best_acc  # потеря ветви = 1 − лучшая LOO-точность
            if best[0] < 0.025:  # цель уже меньше 0.025
                break  # достаточно — прекратить перебор

        classes = sorted(set(ty.astype(np.int64).tolist()))  # отсортированные классы — различные целые значения ty
        if 2 <= len(classes) <= 5 and N >= 12:  # от 2 до 5 классов и не меньше 12 примеров
            cls_map = {c: i for i, c in enumerate(classes)}  # соответствие «значение класса → индекс»
            y_idx = np.array([cls_map[int(c)] for c in ty], dtype=np.int64)  # индексы классов для каждого примера
            Phi_h = self.features("hinge", tx, sx)  # ломаные признаки
            scores = np.zeros(Phi_h.shape[1])  # оценки столбцов, изначально нули
            for c_i in range(len(classes)):  # по классам
                yc = (y_idx == c_i).astype(np.float64) - np.mean(y_idx == c_i)  # центрированный индикатор класса: 1[c = c_i] − доля класса c_i
                scores += np.abs(Phi_h.T @ yc)  # scores += |Phi_hᵀ·yc| — связь признака с индикатором
            top_cols = np.argsort(-scores)[: min(4, Phi_h.shape[1])]  # до 4 столбцов с наибольшей оценкой
            Phi_sub = Phi_h[:, top_cols]  # матрица отобранных признаков
            res = batch_logit_safe_step_ce(  # один безопасный шаг перекрёстной энтропии
                Phi_sub,  # по отобранным признакам
                y_idx,  # с индексами классов
                len(classes),  # число классов
                steps=300,  # 300 шагов
                l2_reg=np.full(top_cols.size, 1e-6),  # L2-регуляризация 1e-6 на каждый признак
            )
            theta = res["theta"]  # матрица весов классов (число классов, число признаков)
            raw_ce, th_dec = pack_vector_dyadic5(theta.ravel(), stages=2)  # квантование dyadic5: байты raw_ce и квантованный вектор th_dec
            th_q = th_dec.reshape(theta.shape)  # квантованные веса в форме theta
            nb_ce = 6 + top_cols.size + len(raw_ce)  # число байтов: 6 + число признаков + длина raw_ce
            pred_tr = np.array(  # прогноз на train
                [classes[i] for i in np.argmax(Phi_sub @ th_q.T, axis=1)],  # класс с максимальной логитой: classes[argmax(Phi_sub·th_qᵀ)]
                dtype=np.float64,  # в float64
            )
            acc_tr = float(np.mean(pred_tr == ty))  # точность на train: доля совпадений с ty
            mse_tr = float(np.mean((pred_tr - ty) ** 2))  # MSE на train: mean((pred_tr − ty)^2)
            obj_ce = mse_tr + 0.002 * nb_ce + (1.0 - acc_tr) * 0.5  # цель CE: mse_tr + 0.002·nb_ce + 0.5·(1 − acc_tr)
            if obj_ce < best[0]:  # цель меньше рекордной
                ce_meta = {  # метаданные классификатора
                    "classes": classes,  # список классов
                    "cols": top_cols.tolist(),  # номера столбцов признаков
                    "theta": th_q,  # квантованные веса
                    "packed": raw_ce,  # упакованные байты
                }
                best = (  # новый рекорд
                    obj_ce,  # цель
                    "hinge",  # ветвь признаков
                    "ce",  # режим — классификация
                    None,  # весов регрессии нет
                    raw_ce,  # упакованные байты
                    raw_ce,  # упакованные байты (второй слот)
                    ce_meta,  # метаданные CE
                    pred_tr / sy,  # прогноз train в нормированных единицах
                    None,  # метаданных MLP нет
                )

        if router is not None:  # маршрутизатор задан
            router.update(branch_losses)  # передать ему потери ветвей

        _, bname, mode, wq, packed, _, ce_meta, ptr_n, mlp_meta = best  # распаковать рекорд

        if ptr_n is None:  # ни один кандидат не принят — вырожденная подгонка
            # Все кандидаты OMP отклонены (|w| > 8), ветвь классификации неприменима.
            # Вместо сбоя сертифицируется консервативный нулевой прогноз: дальнейший
            # допуск коммита увидит истинные (плохие) метрики val и поставит HOLD на этом эксперте.
            wq = np.zeros(Phi.shape[1], dtype=np.float64)  # нулевые веса (прогноз 0)
            packed = pack_vector_dyadic5(wq, stages=1)[0]  # упакованные байты нулевых весов (1 стадия)
            mode, bname, ce_meta, mlp_meta = "reg", "poly", None, None  # режим reg, ветвь poly, без метаданных CE и MLP
            ptr_n = np.zeros(N, dtype=np.float64)  # нулевой прогноз train в нормированных единицах

        acc_tr_cur = float(np.mean(np.rint(ptr_n * sy) == ty))  # точность на train: доля round(ptr_n·sy) = ty
        mse_tr_cur = float(np.mean((ptr_n * sy - ty) ** 2))  # MSE на train: mean((ptr_n·sy − ty)^2)
        if acc_tr_cur < 0.99 and N >= 120 and mse_tr_cur > 0.08 and mode == "reg":  # точность < 0.99, N ≥ 120, MSE > 0.08, режим reg — пробовать корректор
            resid_tr = y - ptr_n  # остаток в нормированных единицах: resid_tr = y − ptr_n
            cut = int(0.8 * N)  # граница нарезки: cut = ⌊0.8·N⌋
            _, r_icv, _ = self._fit_stage2_mlp(  # внутренняя проверка: MLP на первых 80%, прогноз остатка на последних 20%
                tx[:cut], resid_tr[:cut], tx[cut:], sx, hidden=12, steps=260, seed=7  # train — срез 0..cut−1, проверка — срез cut..N−1
            )
            mse_icv_0 = float(np.mean((resid_tr[cut:] * sy) ** 2))  # MSE остатка без корректора на хвосте: mean((resid·sy)^2)
            mse_icv_1 = float(np.mean(((resid_tr[cut:] - r_icv) * sy) ** 2))  # MSE остатка после корректора: mean(((resid − r_icv)·sy)^2)
            if mse_icv_1 < 0.72 * mse_icv_0:  # корректор снижает MSE более чем на 28%
                _, _, mlp_meta = self._fit_stage2_mlp(  # обучить корректор на всей выборке
                    tx, resid_tr, vx, sx, hidden=12, steps=280, seed=7  # train — все данные, val — vx
                )
                mode = "rvq2"  # режим — регрессия плюс корректор

        Phi_va = self.features(bname, vx, sx)  # матрица признаков проверочной выборки

        def _align_cols(P: np.ndarray, n: int) -> np.ndarray:  # привести число столбцов P к n
            """Число столбцов признаков на train и val может различаться для ветвей, зависящих от данных;
            поэтому матрица приводится к длине вектора весов: лишние столбцы отбрасываются,
            недостающие дополняются нулями. Возвращает P из n столбцов."""
            if P.shape[1] == n:  # число столбцов совпадает
                return P  # вернуть без изменений
            if P.shape[1] > n:  # столбцов больше, чем нужно
                return P[:, :n]  # взять первые n столбцов
            return np.pad(P, ((0, 0), (0, n - P.shape[1])))  # дополнить нулями справа до n столбцов

        if mode == "ce":  # режим классификации
            assert ce_meta is not None  # метаданные CE обязательны
            Pv = Phi_va[:, ce_meta["cols"]]  # отобранные столбцы проверочной матрицы
            pred_q = np.array(  # прогноз на val
                [  # по каждому примеру
                    ce_meta["classes"][i]  # класс с индексом максимальной логиты
                    for i in np.argmax(Pv @ ce_meta["theta"].T, axis=1)  # i = argmax(Pv·thetaᵀ)
                ],
                dtype=np.float64,  # в float64
            )
            nb_total = 6 + len(ce_meta["cols"]) + len(ce_meta["packed"])  # всего байтов: 6 + число признаков + длина packed
        elif mode == "rvq2":  # режим регрессии с корректором
            assert mlp_meta is not None and wq is not None  # метаданные MLP и веса обязательны
            base_va = _align_cols(Phi_va, len(wq)) @ wq  # базовый прогноз: Phi_va (выровненная)·wq
            Xv_mlp = self._mlp_encode(vx, sx)  # входы корректора на val
            r_va = (  # поправка корректора на val
                np.tanh(Xv_mlp @ mlp_meta["W1"] + mlp_meta["b1"]) @ mlp_meta["W2"]  # tanh(X·W1 + b1)·W2
                + mlp_meta["b2"]  # + b2
            ).ravel()  # развёрнуто в вектор
            pred_q = (base_va + r_va) * sy  # прогноз на val: (base + поправка)·sy
            nb_total = 6 + len(packed) + len(mlp_meta["packed"])  # всего байтов: 6 + байты регрессии + байты корректора
        else:  # режим чистой регрессии
            assert wq is not None  # веса обязательны
            pred_q = (_align_cols(Phi_va, len(wq)) @ wq) * sy  # прогноз на val: (Phi_va·wq)·sy
            nb_total = 6 + len(packed)  # всего байтов: 6 + длина packed

        return FactorizedAdapter(  # собрать адаптер
            arity=d,  # арность
            branch=bname,  # ветвь признаков
            mode=mode,  # режим (reg, ce или rvq2)
            sx=sx,  # масштаб входа
            sy=sy,  # масштаб выхода
            w_quant=wq,  # квантованные веса регрессии
            packed_bytes=packed,  # упакованные байты
            ce_meta=ce_meta,  # метаданные классификатора
            mlp_meta=mlp_meta,  # метаданные корректора
            compressed_bytes=nb_total,  # оценка размера в байтах
            val_exact_acc_quant=float(np.mean(np.rint(pred_q) == vy)),  # точность на val: доля round(pred_q) = vy
            val_mse_quant=float(np.mean((pred_q - vy) ** 2)),  # MSE на val: mean((pred_q − vy)^2)
        )


@dataclass  # поля класса автоматически становятся аргументами конструктора
class FactorizedAdapter:  # обученный компактный адаптер
    """Обученный адаптер: компактная модель y = f(x) по целочисленным аргументам.
    
        Поля: arity — арность; branch — ветвь признаков (poly, hinge, bit, core); mode — режим
        (reg — линейная регрессия по признакам, ce — классификатор, rvq2 — регрессия плюс
        корректор-перцептрон); sx, sy — масштабы входа и выхода; w_quant — квантованные веса;
        packed_bytes — упакованные байты; ce_meta, mlp_meta — метаданные классификатора и
        корректора; compressed_bytes — оценка размера; val_exact_acc_quant, val_mse_quant —
        точность и MSE на val. Прогноз: reg → (Phi(x)·w)·sy; ce → класс с максимальной логитой;
        rvq2 → ((Phi(x)·w) + mlp(x))·sy."""
    arity: int  # арность — число аргументов
    branch: str  # ветвь признаков
    mode: str  # режим: reg, ce или rvq2
    sx: float  # масштаб входа
    sy: float  # масштаб выхода
    w_quant: Optional[np.ndarray]  # квантованные веса регрессии
    packed_bytes: bytes  # упакованные байты
    ce_meta: Optional[dict[str, Any]]  # метаданные классификатора
    mlp_meta: Optional[dict[str, Any]]  # метаданные корректора
    compressed_bytes: int  # оценка размера в байтах
    val_exact_acc_quant: float  # точность на val
    val_mse_quant: float  # MSE на val

    def predict(self, core: FactorizedSharedCore, xs: Sequence[int]) -> float:  # предсказать значение по одному набору аргументов
        """Предсказать значение для одного набора аргументов xs.
        
                core: ядро с построителем признаков. Возвращает число по формулам, приведённым в описании класса."""
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1)  # аргументы в строку-матрицу (1, d)
        Phi = core.features(self.branch, x, self.sx)  # признаки Phi (1, число признаков)
        if self.mode == "ce" and self.ce_meta is not None:  # режим классификации
            P = Phi[:, self.ce_meta["cols"]]  # отобранные столбцы P
            idx = int(np.argmax((P @ self.ce_meta["theta"].T)[0]))  # индекс класса с максимальной логитой: argmax(P·thetaᵀ)
            return float(self.ce_meta["classes"][idx])  # вернуть значение класса
        assert self.w_quant is not None  # веса обязательны
        val = float((Phi @ self.w_quant)[0])  # базовый прогноз в нормированных единицах: val = Phi·w
        if self.mode == "rvq2" and self.mlp_meta is not None:  # режим регрессии с корректором
            Xm = core._mlp_encode(x, self.sx)  # входы корректора
            r = float(  # поправка корректора
                (  # открытие выражения
                    np.tanh(Xm @ self.mlp_meta["W1"] + self.mlp_meta["b1"])  # tanh(X·W1 + b1)
                    @ self.mlp_meta["W2"]  # · W2
                    + self.mlp_meta["b2"]  # + b2
                )[0, 0]  # взять единственный элемент
            )
            val += r  # val = val + поправка
        return val * self.sy  # вернуть в исходном масштабе: val·sy

    def to_dict(self) -> dict[str, Any]:  # сериализация в словарь
        """Словарь для сериализации: все поля, packed — в hex; для ce — классы и столбцы признаков,
                для rvq2 — упакованные байты и формы весов корректора."""
        out: dict[str, Any] = {  # словарь полей
            "arity": self.arity,  # арность
            "branch": self.branch,  # ветвь признаков
            "mode": self.mode,  # режим
            "sx": self.sx,  # масштаб входа
            "sy": self.sy,  # масштаб выхода
            "compressed_bytes": self.compressed_bytes,  # оценка размера
            "packed_hex": self.packed_bytes.hex(),  # упакованные байты в hex
            "val_exact_acc_quant": self.val_exact_acc_quant,  # точность на val
            "val_mse_quant": self.val_mse_quant,  # MSE на val
        }
        if self.mode == "ce" and self.ce_meta is not None:  # режим классификации — добавить классы и столбцы
            out["ce_classes"] = self.ce_meta["classes"]  # список классов
            out["ce_cols"] = self.ce_meta["cols"]  # номера столбцов
        elif self.mode == "rvq2" and self.mlp_meta is not None:  # режим с корректором — добавить его байты и формы
            out["mlp_packed_hex"] = self.mlp_meta["packed"].hex()  # упакованные байты корректора в hex
            out["mlp_shapes"] = [  # формы весов корректора
                list(self.mlp_meta["W1"].shape),  # форма W1
                list(self.mlp_meta["b1"].shape),  # форма b1
                list(self.mlp_meta["W2"].shape),  # форма W2
                list(self.mlp_meta["b2"].shape),  # форма b2
            ]
        return out  # вернуть словарь

    @staticmethod  # статический метод — не получает self
    def from_dict(  # восстановить адаптер из словаря
        d: dict[str, Any], core: "FactorizedSharedCore"  # d — словарь, core — ядро с построителем признаков
    ) -> "FactorizedAdapter":  # возвращает адаптер
        """Восстановить адаптер из словаря d (обратная операция к to_dict).
        
                d: словарь из to_dict, core: ядро (нужен для ширины матрицы признаков D).
                Веса регрессии распаковываются из packed записей распределения бит по D признаков;
                веса CE — из dyadic5-упаковки в форму (число классов, число столбцов);
                веса корректора — dyadic5-упаковка, обратный поворот Адамара и разрезка по формам."""
        arity = int(d["arity"])  # арность
        branch = str(d["branch"])  # ветвь признаков
        mode = str(d["mode"])  # режим
        sx = float(d["sx"])  # масштаб входа
        sy = float(d["sy"])  # масштаб выхода
        packed = bytes.fromhex(d["packed_hex"])  # упакованные байты из hex
        D = core.features(branch, np.zeros((1, arity), dtype=np.float64), sx).shape[1]  # ширина матрицы признаков D — по нулевому входу
        w_quant = None  # веса регрессии — пока нет
        ce_meta = None  # метаданные CE — пока нет
        mlp_meta = None  # метаданные корректора — пока нет
        if mode == "ce":  # режим классификации
            classes = [int(c) for c in d["ce_classes"]]  # классы
            cols = [int(c) for c in d["ce_cols"]]  # номера столбцов
            th = unpack_vector_dyadic5(packed).reshape(len(classes), len(cols))  # распаковать веса theta в форму (число классов, число столбцов)
            ce_meta = {"classes": classes, "cols": cols, "theta": th, "packed": packed}  # метаданные классификатора
        else:  # режим регрессии
            w_quant = unpack_bitalloc_records(packed, D)  # распаковать D весов из записей распределения бит
            if mode == "rvq2" and "mlp_packed_hex" in d:  # режим с корректором и его байты есть
                mp = bytes.fromhex(d["mlp_packed_hex"])  # упакованные байты корректора
                s1, sb1, s2, sb2 = [tuple(int(x) for x in sh) for sh in d["mlp_shapes"]]  # формы W1, b1, W2, b2
                n_all = (  # общее число весов
                    s1[0] * s1[1] + sb1[0] * sb1[1] + s2[0] * s2[1] + sb2[0] * sb2[1]  # n_all = |W1| + |b1| + |W2| + |b2| — число весов
                )
                wr_q = unpack_vector_dyadic5(mp)  # распакованный повёрнутый вектор весов
                all_wq = block_hadamard_unrotate(wr_q, n_all, 8)  # обратный поворот Адамара (блоки 8) и обрезка до n_all
                p1 = s1[0] * s1[1]  # граница W1: p1 = |W1|
                p2 = p1 + sb1[0] * sb1[1]  # граница b1: p2 = p1 + |b1|
                p3 = p2 + s2[0] * s2[1]  # граница W2: p3 = p2 + |W2|
                mlp_meta = {  # метаданные корректора
                    "W1": all_wq[:p1].reshape(s1),  # W1 — первые p1 элементов
                    "b1": all_wq[p1:p2].reshape(sb1),  # b1 — срез p1..p2−1
                    "W2": all_wq[p2:p3].reshape(s2),  # W2 — срез p2..p3−1
                    "b2": all_wq[p3:].reshape(sb2),  # b2 — остаток
                    "packed": mp,  # упакованные байты
                }
        return FactorizedAdapter(  # собрать адаптер
            arity=arity,  # арность
            branch=branch,  # ветвь
            mode=mode,  # режим
            sx=sx,  # масштаб входа
            sy=sy,  # масштаб выхода
            w_quant=w_quant,  # веса
            packed_bytes=packed,  # байты
            ce_meta=ce_meta,  # метаданные CE
            mlp_meta=mlp_meta,  # метаданные корректора
            compressed_bytes=int(d["compressed_bytes"]),  # размер
            val_exact_acc_quant=float(d["val_exact_acc_quant"]),  # точность на val
            val_mse_quant=float(d["val_mse_quant"]),  # MSE на val
        )


class CRAMEngine:  # движок CRAM: энергия, среднее поле, скрытые состояния, псевдо-правдоподобие

    @staticmethod  # статический метод
    def solve_coloring(  # раскраска графа локальным спуском
        edges: Sequence[tuple[int, int]],  # рёбра
        n_nodes: int,  # число вершин
        num_colors: int,  # число цветов
        givens: Optional[dict[int, int]] = None,  # фиксированные цвета вершин
        eps_noise: float = 0.0,  # амплитуда шума поля
        max_sweeps: int = 800,  # предел проходов
        seed: int = 42,  # зерно генератора
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Решить задачу раскраски графа локальным спуском с шумом и теплованной баней.
        
                edges: рёбра (a, b); n_nodes: число вершин; num_colors: число цветов; givens:
                фиксированные цвета вершин; eps_noise: амплитуда шума поля; max_sweeps: предел
                проходов; seed: зерно. Энергия E = число рёбер с одинаковыми цветами концов.
                Поле вершины i: f[c] = число соседей цвета c. Вершина перекрашивается, если
                f[y_i] − min_c f[c] > 2·eps_noise (шум не мог создать улучшение), цвет выбирается
                случайно среди лучших. Без улучшений 8 проходов подряд — случайный перезапуск;
                иначе — heat-bath шаги: P(c) ∝ exp(−f[c]/max(0.15, T)), затем T *= 0.96.
                Возвращает словарь с раскраской, энергией, счётчиками работы и сертификатом
                корректности: energy < 1, все рёбра правильные и условия givens выполнены."""
        adj = [set() for _ in range(n_nodes)]  # список множеств соседей
        for a, b in edges:  # по рёбрам
            if a != b:  # без петель
                adj[a].add(b)  # b — сосед a
                adj[b].add(a)  # a — сосед b
        adj_list = [sorted(s) for s in adj]  # отсортированные списки соседей
        giv = dict(givens or {})  # копия фиксированных цветов (пусто, если None)
        rng = np.random.default_rng(seed)  # генератор с зерном
        y = [giv.get(i, int(rng.integers(0, num_colors))) for i in range(n_nodes)]  # начальная раскраска: фиксированные или случайный цвет

        def energy(st: Sequence[int]) -> int:  # энергия раскраски st
            """Энергия E = число рёбер, у которых концы одного цвета."""
            return sum(1 for a, b in edges if st[a] == st[b])  # энергия E = Σ_{(a,b)} 1[st[a] = st[b]]

        def sparse_fields(i: int, st: Sequence[int]) -> list[int]:  # локальное поле вершины i
            """Поле f[c] = число соседей i цвета c в раскраске st."""
            f = [0] * num_colors  # нули по цветам
            for j in adj_list[i]:  # по соседям
                f[st[j]] += 1  # f[цвет соседа] + 1
            return f  # вернуть поле

        cur_e = energy(y)  # текущая энергия
        sweeps = hb_steps = stale = 0  # счётчики: проходы, шаги тепловой бани, застоявшиеся проходы
        T = 0.85  # начальная температура
        alpha_cool = 0.96  # коэффициент охлаждения
        sparse_work = sum(len(adj_list[i]) * (num_colors**2) for i in range(n_nodes))  # работа разреженного поля за проход: Σ_i deg(i)·C^2
        dense_work = n_nodes * max(1, n_nodes - 1) * (num_colors**2)  # работа плотного поля: N·max(1, N−1)·C^2

        while sweeps < max_sweeps and cur_e > 0:  # пока есть проходы и энергия > 0
            improved = False  # улучшений в проходе пока нет
            for i in range(n_nodes):  # по вершинам
                if i in giv:  # фиксированная вершина
                    continue  # пропустить
                fields = sparse_fields(i, y)  # поле вершины
                noisy = [  # зашумлённое поле
                    f  # значение поля
                    + (  # плюс
                        float(rng.uniform(-eps_noise, eps_noise))  # равномерный шум в [−eps, eps]
                        if eps_noise > 0  # если шум включён
                        else 0.0  # иначе 0
                    )
                    for f in fields  # по цветам
                ]
                cur_f = noisy[y[i]]  # поле текущего цвета
                best_val = min(noisy)  # минимальное поле
                if cur_f - best_val > 2.0 * eps_noise:  # улучшение больше удвоенного шума — оно настоящее
                    ties = [c for c, v in enumerate(noisy) if v == best_val]  # цвета с минимальным полем
                    y[i] = ties[int(rng.integers(0, len(ties)))]  # случайный из лучших
                    improved = True  # было улучшение
            cur_e = energy(y)  # пересчитать энергию
            sweeps += 1  # проходов + 1
            if cur_e == 0:  # энергия 0 — раскраска правильная
                break  # выход
            if not improved:  # улучшений не было
                stale += 1  # застой + 1
                if stale >= 8:  # застой ≥ 8 — перезапуск
                    y = [  # новая случайная раскраска
                        giv.get(i, int(rng.integers(0, num_colors)))  # фиксированный или случайный цвет
                        for i in range(n_nodes)  # по вершинам
                    ]
                    stale = 0  # сброс застоя
                    T = 0.85  # сброс температуры
                    cur_e = energy(y)  # пересчитать энергию
                    continue  # к следующему проходу
                for _ in range(min(12, max(3, cur_e * 2))):  # тепловая баня: min(12, max(3, 2E)) шагов
                    conf = [  # конфликтные вершины
                        i  # вершина
                        for i in range(n_nodes)  # по вершинам
                        if i not in giv and any(y[i] == y[j] for j in adj_list[i])  # не фиксирована и есть сосед того же цвета
                    ]
                    if not conf:  # конфликтов нет
                        break  # выход
                    pick = conf[int(rng.integers(0, len(conf)))]  # случайная конфликтная вершина
                    if rng.random() < 0.4 and adj_list[pick]:  # с вероятностью 0,4 перейти к соседу
                        free_nb = [j for j in adj_list[pick] if j not in giv]  # соседи без фиксированного цвета
                        if free_nb:  # такие есть
                            pick = free_nb[int(rng.integers(0, len(free_nb)))]  # случайный сосед
                    fields = sparse_fields(pick, y)  # поле выбранной вершины
                    logits = np.array(  # логиты
                        [-f / max(0.15, T) for f in fields], dtype=np.float64  # логит цвета: logit_c = −f[c] / max(0.15, T)
                    )
                    logits -= np.max(logits)  # вычесть максимум для устойчивости
                    probs = np.exp(logits)  # экспоненты
                    probs /= np.sum(probs)  # нормировка: P(c) = exp(l_c) / Σ exp(l)
                    y[pick] = int(rng.choice(num_colors, p=probs))  # выбор цвета по распределению
                    hb_steps += 1  # шагов бани + 1
                T *= alpha_cool  # охлаждение: T = T·0.96
                cur_e = energy(y)  # пересчитать энергию

        valid = all(y[a] != y[b] for a, b in edges) and all(  # правильность: все рёбра разноцветные и
            y[i] == g for i, g in giv.items()  # все фиксированные цвета соблюдены
        )
        return {  # вернуть результат
            "coloring": y,  # раскраска
            "valid": valid,  # признак правильности
            "energy": cur_e,  # энергия
            "sweeps": sweeps,  # число проходов
            "hb_steps": hb_steps,  # число шагов бани
            "validity_certificate": bool(cur_e < 1 and valid),  # сертификат: энергия < 1 и раскраска правильная
            "givens_respected": all(y[i] == g for i, g in giv.items()),  # фиксированные цвета соблюдены
            "sparse_work_per_sweep": sparse_work,  # работа разреженного поля
            "dense_work_per_sweep": dense_work,  # работа плотного поля
        }

    @staticmethod  # статический метод
    def mean_field_kl_step(  # шаг координатного среднего поля
        B: np.ndarray, K: np.ndarray, p: np.ndarray, beta: float = 2.0, iters: int = 12  # B, K — поля; p — распределение; beta — обратная температура; iters — проходы
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Один шаг координатного среднего поля (mean-field) для KL-вариационной задачи.
        
                B: (N, A) унарные поля; K: (N, N, A, A) парные; p: (N, A) текущее распределение;
                beta: обратная температура; iters: число проходов. Свободная энергия
                F(p) = Σ B·p + 0.5·Σ_{i<j} p_i K_sym p_j + (1/beta)·Σ p·log p. Для каждой вершины
                новое q_i ∝ exp(−beta·h_i), h_i = B_i + Σ_j K_sym[i] p_j − K_sym[i,i] p_i.
                Гарантия спуска: F(p) − F(q) = (1/beta)·KL(p_i‖q_i) ≥ 0 (проверяется утверждением).
                Возвращает p, свободные энергии и признак монотонности."""
        N, A = p.shape  # N — вершины, A — состояния
        p_cur = p.copy()  # текущее распределение — копия
        K_sym = 0.5 * (K + K.transpose(1, 0, 3, 2))  # симметризация: K_sym[i,j,a,b] = (K[i,j,a,b] + K[j,i,b,a]) / 2

        def free_energy(pm: np.ndarray) -> float:  # свободная энергия распределения pm
            """F(pm) = Σ B·pm + 0.5·Σ pm_i K_sym pm_j + (1/beta)·Σ pm·log pm."""
            unary = float(np.sum(B * pm))  # унарный член: Σ_{i,a} B[i,a]·pm[i,a]
            pair = 0.5 * float(np.einsum("ia,ijab,jb->", pm, K_sym, pm))  # парный член: 0.5·Σ pm_ia·K_sym[i,j,a,b]·pm_jb
            ent = (  # энтропийный член
                float(  # сумма
                    np.sum(  # сумма
                        np.where(pm > 1e-18, pm * np.log(np.maximum(pm, 1e-18)), 0.0)  # pm·log(pm) там, где pm > 1e-18, иначе 0
                    )
                )
                / beta  # делённая на beta
            )
            return unary + pair + ent  # F = унарный + парный + энтропийный

        f0 = free_energy(p_cur)  # начальная свободная энергия
        f_hist = [f0]  # история значений
        for _ in range(iters):  # по проходам
            for i in range(N):  # по вершинам
                hz = (  # локальное поле вершины
                    B[i]  # унарное поле B_i
                    + np.einsum("jab,jb->a", K_sym[i], p_cur)  # + Σ_{j,b} K_sym[i,j,a,b]·p_jb
                    - np.einsum("ab,b->a", K_sym[i, i], p_cur[i])  # − Σ_b K_sym[i,i,a,b]·p_ib (без самосвязи)
                )
                logits = -beta * hz  # логиты: −beta·h
                logits -= np.max(logits)  # вычесть максимум
                q_i = np.exp(logits)  # экспоненты
                q_i /= np.sum(q_i)  # нормировка: q_i = exp(−beta·h) / Σ exp(−beta·h)
                p_old_i = p_cur[i].copy()  # старое распределение вершины
                kl = float(  # KL-расхождение старого и нового
                    np.sum(  # сумма
                        np.where(  # где
                            p_old_i > 1e-18,  # где старое значение p_old > 1e-18
                            p_old_i  # старое значение p_old
                            * np.log(  # умножить на логарифм
                                np.maximum(p_old_i, 1e-18) / np.maximum(q_i, 1e-18)  # отношение p_old / q
                            ),
                            0.0,  # иначе 0
                        )
                    )
                )
                f_before = free_energy(p_cur)  # F до замены
                p_cur[i] = q_i  # заменить распределение вершины
                f_after = free_energy(p_cur)  # F после
                assert abs((f_before - f_after) - kl / beta) <= 1e-7  # проверка: F(p) − F(q) = KL/beta с точностью 1e-7
                f_hist.append(f_after)  # добавить в историю
        return {  # вернуть
            "p": p_cur,  # распределение
            "free_energy_0": f0,  # начальная F
            "free_energy_final": f_hist[-1],  # конечная F
            "monotonic": all(  # монотонность
                f_hist[k + 1] <= f_hist[k] + 1e-9 for k in range(len(f_hist) - 1)  # F_{k+1} ≤ F_k + 1e-9 для всех k
            ),
        }

    @staticmethod  # статический метод
    def solve_latent_maze(  # решение скрытого лабиринта итерацией сжатия
        n_nodes: int, adj: Sequence[Sequence[int]], goal: int, eps_z: float = 1e-6  # вершины, соседи, цель, точность
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Решить «скрытый лабиринт»: стационарную точку m·z = A·z + 1_G, где m = Δ + 1.
        
                adj: списки соседей; goal: вершина-цель; eps_z: точность. A — матрица смежности,
                Δ — максимальная степень. Итерации z_new[i] = (Σ_{j∈adj[i]} z[j] + 1[i = goal]) / m;
                это сжатие с коэффициентом q = Δ/m, поэтому остаток ≤ eps_z за
                ⌈log(eps_z·(1 − q)) / log(q)⌉ шагов (оценка: diff·q/(1 − q) ≤ eps_z).
                Границы: 0 ≤ z ≤ 1/(m − Δ). Возвращает z, число итераций, q и верхнюю границу."""
        delta = max((len(nb) for nb in adj), default=0)  # максимальная степень Δ
        m = float(delta + 1)  # множитель m = Δ + 1
        q = delta / m  # коэффициент сжатия q = Δ / m
        z = np.zeros(n_nodes, dtype=np.float64)  # z — нулевой вектор
        iters = 0  # число итераций
        if q > 0:  # есть рёбра
            max_it = max(1, math.ceil(math.log(eps_z * (1.0 - q)) / math.log(q)))  # предел итераций: max(1, ⌈log(eps·(1−q)) / log(q)⌉)
            for _ in range(max_it):  # по итерациям
                z_new = np.array(  # новое z
                    [  # начало списка значений
                        (sum(z[j] for j in adj[i]) + (1.0 if i == goal else 0.0)) / m  # новое значение z_i = (Σ_{j∈adj(i)} z_j + 1[i = goal]) / m
                        for i in range(n_nodes)  # по вершинам
                    ]
                )
                diff = float(np.max(np.abs(z_new - z)))  # изменение: max|z_new − z|
                z = z_new  # заменить z
                iters += 1  # итераций + 1
                if diff * q / (1.0 - q) <= eps_z:  # оценка остатка diff·q/(1−q) ≤ eps_z
                    break  # достаточно
        else:  # рёбер нет
            z[goal] = 1.0 / m  # только цель: z_goal = 1/m
            iters = 1  # одна итерация
        assert np.all(z >= -1e-12) and np.all(z <= 1.0 / (m - delta) + 1e-12)  # проверка границ: 0 ≤ z ≤ 1/(m − Δ)
        return {  # вернуть
            "z": z,  # вектор z
            "iters": iters,  # число итераций
            "contraction_q": q,  # коэффициент сжатия
            "upper_bound": 1.0 / (m - delta),  # верхняя граница 1/(m − Δ)
        }

    @staticmethod  # статический метод
    def attention_coupling_matrix(  # симметричная матрица парной связи
        kappa: np.ndarray, scores_z: np.ndarray, f_compat: np.ndarray  # смежность, оценки скрытых состояний, совместимость
    ) -> np.ndarray:  # возвращает K (N, N, A, A)
        """Построить симметричную матрицу парной связи из внимания и совместимости цветов.
        
                kappa: (N, N) смежность; scores_z: (N, N) оценки скрытых состояний;
                f_compat: (A, A) совместимость цветов. Внимание w = softmax по строкам scores_z;
                kw = kappa·w. K[i,j,a,b] = kw[i,j]·f[a,b] + kw[j,i]·f[b,a] — симметрично по построению;
                диагональ K[i,i] = 0."""
        s = scores_z - np.max(scores_z, axis=1, keepdims=True)  # вычесть максимум по строке
        w = np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True)  # внимание: softmax по строке
        kw = kappa * w  # произведение kw = kappa·w
        K = (  # матрица связи
            kw[:, :, None, None] * f_compat[None, None, :, :]  # вклад ребра: kw[i,j]·f[a,b]
            + kw.T[:, :, None, None] * f_compat.T[None, None, :, :]  # + kw[j,i]·f[b,a] — симметризация
        )
        for i in range(K.shape[0]):  # по вершинам
            K[i, i] = 0.0  # диагональ в 0
        return K  # вернуть K

    @staticmethod  # статический метод
    def alternating_latent_visible_solve(  # чередующееся решение (z, y)
        n_nodes: int,  # вершины
        adj: Sequence[Sequence[int]],  # соседи
        goal: int,  # цель
        num_colors: int = 2,  # число цветов
        rounds: int = 3,  # раунды
        seed: int = 42,  # зерно
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Чередующееся решение (z, y): скрытые z → связь внимания → спуск по цветам.
        
                Сначала z из solve_latent_maze; затем K из attention_coupling_matrix с
                scores_z = 4·z⊗z и единичной совместимостью. Энергия E = 0.5·Σ_{i≠j} K[i,j,y_i,y_j].
                В каждом раунде вершина берёт цвет c с минимальным полем Σ_j K[i,j,c,y_j],
                если он меньше текущего. Возвращает z, раскраску, энергии до/после и признак
                неубывания энергии."""
        res_z = CRAMEngine.solve_latent_maze(n_nodes, adj, goal, eps_z=1e-6)  # скрытые состояния z
        z = res_z["z"]  # вектор z
        rng = np.random.default_rng(seed)  # генератор
        kappa = np.zeros((n_nodes, n_nodes), dtype=np.float64)  # нулевая смежность
        for i, nbs in enumerate(adj):  # по вершинам и соседям
            for j in nbs:  # по соседям
                kappa[i, j] = 1.0  # отметить ребро
        f_compat = np.eye(num_colors, dtype=np.float64)  # единичная совместимость: штраф только за одинаковые цвета
        scores_z = np.outer(z, z) * 4.0  # оценки: 4·z_i·z_j
        K = CRAMEngine.attention_coupling_matrix(kappa, scores_z, f_compat)  # матрица связи
        assert np.allclose(K, K.transpose(1, 0, 3, 2))  # проверка симметрии
        y = [int(rng.integers(0, num_colors)) for _ in range(n_nodes)]  # случайная начальная раскраска

        def pair_e(st):  # энергия раскраски st
            """E(st) = 0.5·Σ_{i≠j} K[i,j,st_i,st_j]."""
            return 0.5 * sum(  # половина суммы
                K[i, j, st[i], st[j]]  # элемент связи K[i,j,st_i,st_j]
                for i in range(n_nodes)  # по i
                for j in range(n_nodes)  # по j
                if i != j  # без диагонали
            )

        e0 = pair_e(y)  # начальная энергия
        for _ in range(rounds):  # по раундам
            for i in range(n_nodes):  # по вершинам
                fields = [  # поля по цветам
                    sum(K[i, j, c, y[j]] for j in range(n_nodes) if j != i)  # поле цвета: f_c = Σ_{j≠i} K[i,j,c,y_j]
                    for c in range(num_colors)  # по цветам
                ]
                best_c = int(np.argmin(fields))  # цвет с минимальным полем
                if fields[best_c] < fields[y[i]] - 1e-12:  # строго лучше текущего
                    y[i] = best_c  # перекрасить
        ef = pair_e(y)  # конечная энергия
        return {  # вернуть
            "z": z,  # z
            "y": y,  # раскраска
            "energy_0": e0,  # энергия до
            "energy_final": ef,  # энергия после
            "monotone_drop": ef <= e0 + 1e-12,  # энергия не возросла
        }

    @staticmethod  # статический метод
    def train_cram_pseudolikelihood(  # обучение унарных полей по псевдо-правдоподобию
        samples: np.ndarray, num_colors: int, beta: float = 1.0, steps: int = 80  # выборки, цвета, обратная температура, шаги
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Обучить унарные поля B по псевдо-правдоподобию: перекрёстная энтропия локального поля.
        
                samples: (M, N) выборки цветов; num_colors: число цветов; beta: обратная
                температура; steps: число шагов. Для каждой вершины i логиты единичного
                признака обучаются предсказывать samples[:, i]; B[i] = −theta[:, 0] / beta.
                Возвращает B и среднюю потерю."""
        M, N = samples.shape  # M — выборки, N — вершины
        B_learned = np.zeros((N, num_colors), dtype=np.float64)  # обучаемые поля B — нули
        losses = []  # потери по вершинам
        for i in range(N):  # по вершинам
            y_i = samples[:, i]  # цвета вершины i по выборкам
            X_i = np.ones((M, 1), dtype=np.float64)  # единственный признак — столбец единиц
            res = batch_logit_safe_step_ce(  # обучить логиты
                X_i, y_i, num_colors, steps=steps, l2_reg=np.array([1e-4])  # по одному признаку, L2 1e-4
            )
            B_learned[i] = -res["theta"][:, 0] / beta  # B_i = −theta_i / beta
            losses.append(res["loss_final"])  # запомнить потерю
        return {"B": B_learned, "mean_pl_loss": float(np.mean(losses))}  # вернуть B и среднюю потерю


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


# ==============================================================================
# 8. Формальное управление временем выполнения: CertStepGen, AuditLedger, HedgeRouter
# ==============================================================================


@dataclass  # поля класса автоматически становятся аргументами конструктора
class CertStepGen:  # канонический сертификат перехода
    """Канонический сертификат одного шага: что сделано, как изменилась цель и что истрачено.
    
        drop = obj_before − obj_after ≥ 0; budget_* и risk_* — остаток и расход бюджета и риска;
        delta_* — остаток и расход статистической значимости; floors_kept — все нижние границы соблюдены."""

    version: int  # версия схемы
    task: str  # имя задачи
    action: str  # действие
    obj_before: float  # цель до
    obj_after: float  # цель после
    drop: float  # снижение цели
    budget_rem: float  # остаток бюджета
    budget_spend: float  # расход бюджета
    risk_rem: float  # остаток риска
    risk_spend: float  # расход риска
    delta_rem: float  # остаток значимости
    delta_spend: float  # расход значимости
    floors_kept: bool  # границы соблюдены


class HedgeRouter:  # маршрутизатор Хеджа с ограниченными потерями
    """Маршрутизатор Хеджа (Hedge): эксперты с весами, убывающими по потерям.
    
        Начальные веса 1/K. После потерь l ∈ [0, 1]: w_i *= exp(−eta·l_i), затем нормировка.
        Нижняя граница против «запирания»: w_i ≥ (1/K)·exp(−eta·cum_loss_i) — проверяется.
        K — число экспертов, eta — скорость обучения, cum_loss — накопленные потери."""

    def __init__(self, n_experts: int, eta: float = 0.5):  # создать маршрутизатор: n_experts экспертов, скорость eta
        self.K = n_experts  # число экспертов
        self.eta = eta  # скорость обучения
        self.weights = np.ones(n_experts, dtype=np.float64) / max(1, n_experts)  # веса: w_i = 1 / max(1, K)
        self.cum_loss = np.zeros(n_experts, dtype=np.float64)  # накопленные потери — нули

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:  # снимок состояния
        return self.weights.copy(), self.cum_loss.copy()  # копии весов и потерь

    def restore(self, snap: tuple[np.ndarray, np.ndarray]) -> None:  # восстановить из снимка
        self.weights = snap[0].copy()  # веса из снимка
        self.cum_loss = snap[1].copy()  # потери из снимка

    def update(self, losses_01: np.ndarray) -> np.ndarray:  # обновить веса по потерям
        losses = np.clip(np.asarray(losses_01, dtype=np.float64), 0.0, 1.0)  # потери, обрезанные в [0, 1]
        self.cum_loss += losses  # накопить потери
        self.weights *= np.exp(-self.eta * losses)  # веса: w_i *= exp(−eta·l_i)
        self.weights /= np.sum(self.weights)  # нормировка: w_i = w_i / Σ w
        floor = (1.0 / self.K) * np.exp(-self.eta * self.cum_loss)  # нижняя граница: floor_i = (1/K)·exp(−eta·cum_loss_i)
        assert np.all(self.weights + 1e-12 >= floor)  # проверка: w_i + 1e-12 ≥ floor_i
        return self.weights.copy()  # вернуть копию весов

    def ordered(self, names: Optional[Sequence[str]] = None) -> list:  # порядок экспертов по убыванию веса
        """Порядок экспертов по убыванию веса (априорный порядок Хеджа).
        
                names: если задан — возвращаются индексы из names (отсортированные по весу),
                иначе — argsort весов. """
        order = np.argsort(-self.weights)  # порядок: argsort(−weights)
        if names is None:  # имена не заданы
            return order.tolist()  # список индексов
        return [i for i in order if i < len(names)]  # индексы, не выходящие за длину names


@dataclass  # поля класса автоматически становятся аргументами конструктора
class AuditStream:  # диапазон потока аудита: [lo, hi) и курсор
    lo: int  # начало диапазона
    hi: int  # конец диапазона (не включительно)
    cursor: int  # текущая позиция


@dataclass  # поля класса автоматически становятся аргументами конструктора
class AuditLedger:  # журнал аудита: непересекающиеся потоки, расход значимости, нижние границы
    """Журнал аудита статистических проверок.
    
        delta_total — общий бюджет значимости (по умолчанию 0.05); spent — истрачено по задачам;
        streams — потоки выборок [lo, hi) с курсором; floors — нижние границы качества по задачам;
        rejected_tasks — отклонённые задачи; [final_test_lo, final_test_hi) — зарезервированный
        диапазон финального теста, который нельзя трогать. Остаток значимости:
        remaining = max(0, delta_total − Σ spent)."""

    delta_total: float = 0.05  # общий бюджет значимости
    spent: dict[str, float] = field(default_factory=dict)  # истрачено по задачам
    streams: dict[str, AuditStream] = field(default_factory=dict)  # потоки выборок
    floors: dict[str, float] = field(default_factory=dict)  # нижние границы качества
    rejected_tasks: set[str] = field(default_factory=set)  # отклонённые задачи
    final_test_lo: int = 10_000_000  # начало финального теста
    final_test_hi: int = 20_000_000  # конец финального теста
    final_test_touched: bool = False  # финальный тест не тронут
    next_cursor_base: int = 0  # начало следующего потока

    def remaining(self) -> float:  # остаток бюджета значимости
        return max(0.0, self.delta_total - sum(self.spent.values()))  # остаток: remaining = max(0, delta_total − Σ spent)

    def allocate_stream(self, task: str, n_samples: int) -> AuditStream:  # выделить поток выборок задаче
        if task in self.streams:  # поток уже есть
            return self.streams[task]  # вернуть существующий
        lo = self.next_cursor_base  # начало: следующий свободный индекс
        hi = lo + max(1, int(n_samples))  # конец: lo + max(1, n_samples)
        assert (  # проверка
            hi <= self.final_test_lo  # поток не заходит в диапазон финального теста
        ), "audit stream overlapped held-out final test range"
        st = AuditStream(lo=lo, hi=hi, cursor=lo)  # создать поток с курсором в начале
        self.streams[task] = st  # запомнить поток
        self.next_cursor_base = hi  # следующий поток начнётся с hi
        return st  # вернуть поток

    def advance_stream(self, task: str, count: int) -> None:  # продвинуть курсор потока
        st = self.streams[task]  # поток задачи
        st.cursor = min(st.hi, st.cursor + max(0, int(count)))  # курсор: cursor = min(hi, cursor + max(0, count))

    def draw_disjoint_splits(  # разбить область на непересекающиеся train / cegis / val
        self,  # экземпляр
        task: str,  # имя задачи
        domain: list,  # список элементов
        seed: int = 0,  # зерно (смещается суммой кодов символов задачи)
        train_frac: float = 0.35,  # доля train
        cegis_frac: float = 0.43,  # доля cegis
    ) -> tuple[list, list, list]:  # возвращает (train, cegis, val)
        """Разбить domain на непересекающиеся train / cegis / val из ОДНОГО потока аудита.
        
                Перестановка определяется зерном seed + (Σ ord(символы task) mod 10000). Для малых
                областей доли деградируют честно: n ≤ 8 — train n − 1, cegis 0; n ≤ 16 — train n − 3,
                cegis 1; иначе train = max(6, round(n·train_frac)), cegis = max(4, round(n·cegis_frac)),
                а если сумма ≥ n — cegis = max(1, (n − n_tr) // 2). Курсор проходит весь поток."""
        n = len(domain)  # размер области
        st = self.allocate_stream(task, n)  # выделить поток на n элементов
        rng = np.random.default_rng(seed + (sum(map(ord, task)) % 10000))  # генератор с зерном seed + Σ ord(task) mod 10000
        perm = rng.permutation(n)  # случайная перестановка индексов
        if n <= 8:  # очень малая область
            n_tr, n_ceg = max(1, n - 1), 0  # обучение: train = n − 1, cegis = 0
        elif n <= 16:  # малая область
            n_tr, n_ceg = max(2, n - 3), 1  # обучение: train = n − 3, cegis = 1
        else:  # обычная область
            n_tr = max(6, int(round(n * train_frac)))  # обучение: train = max(6, round(n·train_frac))
            n_ceg = max(4, int(round(n * cegis_frac)))  # доводка: cegis = max(4, round(n·cegis_frac))
            if n_tr + n_ceg >= n:  # сумма не оставляет val — уменьшить cegis
                n_ceg = max(1, (n - n_tr) // 2)  # доводка: cegis = max(1, (n − n_tr) // 2)
        tr = [domain[int(i)] for i in perm[:n_tr]]  # train — первые n_tr перестановки
        ceg = [domain[int(i)] for i in perm[n_tr:n_tr + n_ceg]]  # cegis — следующие n_ceg
        val = [domain[int(i)] for i in perm[n_tr + n_ceg:]]  # val — остаток
        st.cursor = min(st.hi, st.lo + len(tr) + len(ceg) + len(val))  # курсор — за всеми использованными
        return tr, ceg, val  # вернуть три части

    @staticmethod  # статический метод
    def hoeffding_radius(alpha: float, n: int) -> float:  # радиус доверительного интервала Хёфдинга
        if n <= 0 or alpha <= 0.0:  # нет выборки или нет значимости
            return 1.0  # максимальный радиус
        return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))  # радиус: r = √(ln(2/alpha) / (2n))

    def spend_anytime(  # потратить значимость и вернуть радиус
        self, task: str, alpha: float, n_samples: int  # задача, значимость, число выборок
    ) -> tuple[bool, float]:  # возвращает (успех, радиус)
        if alpha < 0 or alpha > self.remaining() + 1e-12:  # некорректная или превышающая остаток значимость
            return False, 1.0  # отказ: радиус 1
        self.spent[task] = self.spent.get(task, 0.0) + alpha  # учесть расход: spent[task] += alpha
        self.advance_stream(task, n_samples)  # продвинуть курсор
        return True, self.hoeffding_radius(alpha, n_samples)  # успех и радиус Хёфдинга

    def record_floor(self, task: str, quality: float) -> bool:  # зафиксировать нижнюю границу качества
        prev = self.floors.get(task, -1e18)  # прежняя граница (если нет — −∞)
        if quality + 1e-12 < prev:  # качество ниже границы
            return False  # граница нарушена
        self.floors[task] = max(prev, quality)  # поднять границу: max(prev, quality)
        return True  # граница соблюдена

    def record_reject(self, task: str) -> None:  # зафиксировать отклонение задачи
        self.rejected_tasks.add(task)  # добавить в отклонённые
        if task in self.streams:  # если поток был
            st = self.streams[task]  # поток задачи
            st.cursor = st.hi  # исчерпать поток

    def verify_disjoint_and_clean(self) -> bool:  # проверить непересечение потоков и чистоту финального теста
        if self.final_test_touched:  # финальный тест тронут
            return False  # нарушение
        intervals = sorted((st.lo, st.hi) for st in self.streams.values())  # интервалы потоков по возрастанию
        for i in range(len(intervals) - 1):  # по соседним парам
            if intervals[i][1] > intervals[i + 1][0]:  # конец текущего дальше начала следующего — пересечение
                return False  # нарушение
        for lo, hi in intervals:  # по интервалам
            if not (hi <= self.final_test_lo or lo >= self.final_test_hi):  # интервал пересекает финальный тест
                return False  # нарушение
        return sum(self.spent.values()) <= self.delta_total + 1e-12  # итог: расход не превышает бюджет


def ram_available_bytes() -> int:  # доступная оперативная память в байтах
    try:  # попытка 1: /proc/meminfo
        with open("/proc/meminfo", "r", encoding="utf-8") as f:  # открыть файл сведений о памяти
            for line in f:  # по строкам
                if line.startswith("MemAvailable:"):  # строка доступной памяти
                    return int(line.split()[1]) * 1024  # значение в кБ → байты: × 1024
    except OSError:  # файл недоступен
        pass  # следующий способ
    try:  # попытка 2: psutil
        import psutil  # импорт библиотеки psutil

        return int(psutil.virtual_memory().available)  # доступная память
    except ImportError:  # библиотеки нет
        pass  # следующий способ
    if sys.platform == "win32":  # Windows
        try:  # попытка 3: WinAPI
            import ctypes  # импорт ctypes

            class MEMORYSTATUSEX(ctypes.Structure):  # структура состояния памяти Windows
                _fields_ = [  # поля структуры:
                    ("dwLength", ctypes.c_ulong),  # размер структуры
                    ("dwMemoryLoad", ctypes.c_ulong),  # загрузка памяти, %
                    ("ullTotalPhys", ctypes.c_ulonglong),  # вся физическая память
                    ("ullAvailPhys", ctypes.c_ulonglong),  # доступная физическая память
                    ("ullTotalPageFile", ctypes.c_ulonglong),  # весь файл подкачки
                    ("ullAvailPageFile", ctypes.c_ulonglong),  # доступный файл подкачки
                    ("ullTotalVirtual", ctypes.c_ulonglong),  # всё виртуальное пространство
                    ("ullAvailVirtual", ctypes.c_ulonglong),  # доступное виртуальное пространство
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),  # доступное расширенное виртуальное пространство
                ]

            st = MEMORYSTATUSEX()  # создать структуру
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)  # заполнить размер
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # запросить состояние памяти
            return int(st.ullAvailPhys)  # доступная физическая память в байтах
        except Exception:  # ошибка WinAPI
            pass  # далее — вернуть 0
    return 0  # не удалось определить — 0


# ==============================================================================
# 9. Быстрая точная полиномиальная индукция, кусочный решатель и MDL «встреча посередине»
# ==============================================================================


def _exact_poly_fit(  # точная полиномиальная подгонка по одному аргументу
    xs: Sequence[int], ys: Sequence[int], max_degree: int = 5  # аргументы, значения, максимальная степень
) -> Optional[tuple[int, ...]]:  # возвращает коэффициенты (c_0..c_d) или None
    """Найти многочлен p(x) = Σ c_i·x^i с целыми коэффициентами, точно проходящий через все точки.
    
        xs, ys: целые точки; max_degree: предел степени. Для степени d берётся система из k = d + 1
        точек и решается методом Гаусса в точных дробях (Fraction); если все коэффициенты целые и
        многочлен проходит через ВСЕ точки — возвращается кортеж коэффициентов. Иначе — None."""
    pts = sorted({(int(x), int(y)) for x, y in zip(xs, ys)})  # отсортированное множество точек (без дубликатов)
    if len({x for x, _ in pts}) < 2:  # менее двух различных x — многочлен не определён
        return None  # нет решения
    for degree in range(1, min(max_degree, len(pts) - 1) + 1):  # по степеням от 1 до min(max_degree, n − 1)
        k = degree + 1  # число неизвестных k = d + 1
        sample = pts[:k]  # первые k точек
        a = [[Fraction(x) ** i for i in range(k)] + [Fraction(y)] for x, y in sample]  # расширенная матрица: строка [x^0, x^1, …, x^(k−1), y]
        for col in range(k):  # по столбцам (метод Гаусса)
            piv = next((r for r in range(col, k) if a[r][col] != 0), None)  # ведущая строка: первая с ненулевым элементом в столбце
            if piv is None:  # ведущего элемента нет — система вырождена
                break  # прервать
            a[col], a[piv] = a[piv], a[col]  # поменять строки
            pv = a[col][col]  # ведущий элемент
            a[col] = [v / pv for v in a[col]]  # нормировать ведущую строку
            for r in range(k):  # по строкам
                if r != col and a[r][col] != 0:  # другая строка с ненулевым элементом в столбце
                    f = a[r][col]  # множитель
                    a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]  # вычесть: строка_r − f·ведущая
        else:  # цикл по столбцам завершён без break — система решена
            coeffs = [row[k] for row in a]  # коэффициенты — последний столбец
            if any(c.denominator != 1 for c in coeffs):  # есть дробный коэффициент
                continue  # следующая степень
            icoeffs = tuple(int(c) for c in coeffs)  # целые коэффициенты
            if all(  # проверка на всех точках
                sum(ci * (x**i) for i, ci in enumerate(icoeffs)) == y for x, y in pts  # проверка: p(x) = Σ c_i·x^i = y
            ):
                return icoeffs  # найден точный многочлен
    return None  # не найдено


def _monomial_exponents(arity: int, max_degree: int) -> tuple[tuple[int, ...], ...]:  # показатели всех мономов степени ≤ max_degree
    """Перечислить векторы показателей (p_1, …, p_arity) с Σ p_i ≤ max_degree.
    
        Возвращает кортеж, отсортированный по (сумма показателей, лексикографически)."""
    return tuple(  # вернуть кортеж
        sorted(  # отсортированный
            (  # открытие генератора
                powers  # вектор показателей
                for powers in itertools.product(range(max_degree + 1), repeat=arity)  # все комбинации показателей 0..max_degree
                if sum(powers) <= max_degree  # сумма не больше степени
            ),
            key=lambda powers: (sum(powers), powers),  # ключ: (сумма, сам вектор)
        )
    )


def _fast_mod_p_consistent(  # быстрый модульный фильтр совместности системы
    rows_int: list[tuple[int, ...]],  # строки входов
    ys_int: list[int],  # значения
    monomials: tuple[tuple[int, ...], ...],  # мономы
) -> bool:  # True — система совместна по модулю P
    """Быстрый точный фильтр O(k^3): совместность системы по модулю P = 2^31 − 1.
    
        Строится матрица значений мономов mod P; метод Гаусса по модулю P. Если система
        несовместна по модулю — она несовместна и над дробями, и точное решение не нужно.
        Берётся не более ncols + 8 строк. Возвращает True, если противоречия не найдено."""
    P = _PRIME_MOD  # модуль P
    ncols = len(monomials)  # число мономов
    sub_n = min(len(rows_int), ncols + 8)  # число строк: не более ncols + 8
    mat = []  # матрица системы
    for r_idx in range(sub_n):  # по строкам
        row = rows_int[r_idx]  # входы строки
        feats = [1] * (ncols + 1)  # строка из единиц (ncols + 1 элемент)
        for c, powers in enumerate(monomials):  # по мономам
            val = 1  # значение монома
            for x, p in zip(row, powers):  # по (переменная, показатель)
                if p:  # показатель ненулевой
                    val = (val * pow(x % P, p, P)) % P  # значение: val = val·(x^p mod P) mod P
            feats[c] = val  # столбец монома
        feats[ncols] = ys_int[r_idx] % P  # последний столбец — y mod P
        mat.append(feats)  # добавить строку
    piv_row = 0  # строка ведущего элемента
    for col in range(ncols):  # по столбцам
        piv = next((r for r in range(piv_row, sub_n) if mat[r][col] != 0), None)  # ведущая строка
        if piv is None:  # нет ведущего — столбец свободный, фильтр не решает
            return True  # считать совместной (пропустить фильтр)
        mat[piv_row], mat[piv] = mat[piv], mat[piv_row]  # поменять строки
        inv = pow(mat[piv_row][col], P - 2, P)  # обратный элемент по малой теореме Ферма: a^(P−2) mod P
        prow = [(v * inv) % P for v in mat[piv_row]]  # нормировать ведущую строку
        mat[piv_row] = prow  # записать
        for r in range(sub_n):  # по строкам
            if r != piv_row and mat[r][col] != 0:  # другая строка с ненулевым элементом
                fac = mat[r][col]  # множитель
                mr = mat[r]  # строка r
                for c in range(col, ncols + 1):  # по столбцам от col
                    mr[c] = (mr[c] - fac * prow[c]) % P  # вычесть: mr[c] = (mr[c] − fac·prow[c]) mod P
        piv_row += 1  # следующая ведущая строка
    for r in range(piv_row, sub_n):  # по нижним строкам
        if mat[r][ncols] != 0:  # ненулевой свободный член при нулевых коэффициентах — противоречие
            return False  # система несовместна
    return True  # противоречий нет


def _exact_multivariate_poly_fit(  # точная многомерная полиномиальная подгонка
    inputs: Sequence[Sequence[int]],  # строки входов
    targets: Sequence[int],  # значения
    arity: int,  # арность
    max_degree: int = 4,  # максимальная степень
) -> Optional[tuple[tuple[int, tuple[int, ...]], ...]]:  # возвращает кортеж (коэффициент, показатели) или None
    """Найти многочлен от arity переменных с целыми коэффициентами, точно проходящий через все точки.
    
        Для степени d перебираются мономы с Σ p ≤ d; сначала быстрый модульный фильтр, затем
        точное решение методом Гаусса в дробях на подвыборке (при неполном ранге — на всех строках).
        Возвращаются только ненулевые члены (c, p): p(x) = Σ c·Π x_i^{p_i}."""
    if len(inputs) != len(targets) or not inputs or arity < 1:  # несовпадение длин, пусто или арность < 1
        return None  # нет решения
    rows = [tuple(int(v) for v in row) for row in inputs]  # строки целых
    ys = [int(v) for v in targets]  # значения целые
    for degree in range(max_degree + 1):  # по степеням 0..max_degree
        monomials = _monomial_exponents(arity, degree)  # мономы этой степени
        ncols = len(monomials)  # число мономов
        if len(rows) < ncols:  # строк меньше, чем неизвестных
            continue  # следующая степень
        if not _fast_mod_p_consistent(rows, ys, monomials):  # модульный фильтр отбраковал
            continue  # следующая степень
        sample_rows = rows[: min(len(rows), max(ncols + 12, int(ncols * 2)))]  # подвыборка: max(ncols + 12, 2·ncols) строк
        sample_ys = ys[: len(sample_rows)]  # значения подвыборки
        matrix = []  # расширенная матрица
        for row, y in zip(sample_rows, sample_ys):  # по строкам подвыборки
            features = [  # значения мономов
                Fraction(math.prod(x**p for x, p in zip(row, powers)))  # моном Π x_i^{p_i} в дроби
                for powers in monomials  # по мономам
            ]
            matrix.append(features + [Fraction(y)])  # строка [мономы…, y]
        pivot_row = 0  # текущая ведущая строка
        pivot_rows: list[int] = []  # номера ведущих строк
        full_rank = True  # признак полного ранга
        for col in range(ncols):  # по столбцам
            pivot = next(  # искать ведущую строку
                (r for r in range(pivot_row, len(matrix)) if matrix[r][col] != 0), None  # первая строка с ненулевым элементом
            )
            if pivot is None:  # ведущего нет
                if len(sample_rows) < len(rows):  # использовалась подвыборка — повторить на всех строках
                    matrix = [  # перестроить матрицу
                        [  # начало списка значений  # строка мономов
                            Fraction(math.prod(x**p for x, p in zip(r, powers)))  # моном
                            for powers in monomials  # по мономам
                        ]
                        + [Fraction(y)]  # плюс y
                        for r, y in zip(rows, ys)  # по всем строкам
                    ]
                    pivot_row = 0  # сброс ведущей строки
                    pivot_rows = []  # сброс списка
                    for col2 in range(ncols):  # метод Гаусса по столбцам
                        piv2 = next(  # ведущая строка
                            (  # открытие генератора
                                rr  # элемент генератора
                                for rr in range(pivot_row, len(matrix))  # по строкам от ведущей
                                if matrix[rr][col2] != 0  # с ненулевым элементом в столбце
                            ),
                            None,  # если не найдено — None
                        )
                        if piv2 is None:  # ранг неполный
                            full_rank = False  # признак снят
                            break  # прервать
                        matrix[pivot_row], matrix[piv2] = (  # поменять строки
                            matrix[piv2],  # вторая строка обмена
                            matrix[pivot_row],  # первая строка обмена
                        )
                        sc = matrix[pivot_row][col2]  # ведущий элемент
                        matrix[pivot_row] = [v / sc for v in matrix[pivot_row]]  # нормировать
                        for rr in range(len(matrix)):  # по строкам
                            if rr != pivot_row and matrix[rr][col2]:  # другая строка с ненулевым элементом
                                fac = matrix[rr][col2]  # множитель
                                matrix[rr] = [  # вычесть
                                    a - fac * b  # вычесть
                                    for a, b in zip(matrix[rr], matrix[pivot_row])  # по парам (элемент строки, элемент ведущей)
                                ]
                        pivot_row += 1  # следующая ведущая строка
                else:  # все строки уже использованы
                    full_rank = False  # ранг неполный
                break  # прервать
            if pivot_row >= ncols:  # ведущих строк не меньше, чем столбцов
                full_rank = False  # лишние — ранг неполный
                break  # прервать
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]  # поменять строки
            scale = matrix[pivot_row][col]  # ведущий элемент
            matrix[pivot_row] = [v / scale for v in matrix[pivot_row]]  # нормировать
            for r in range(len(matrix)):  # по строкам
                if r != pivot_row and matrix[r][col]:  # другая строка с ненулевым элементом
                    fac = matrix[r][col]  # множитель
                    matrix[r] = [  # вычесть
                        a - fac * b for a, b in zip(matrix[r], matrix[pivot_row])  # строка_r − fac·ведущая
                    ]
            pivot_rows.append(pivot_row)  # запомнить ведущую строку
            pivot_row += 1  # следующая
        if not full_rank or len(pivot_rows) < ncols:  # ранг неполный
            continue  # следующая степень
        coefficients = [matrix[pivot_rows[col]][-1] for col in range(ncols)]  # коэффициенты — последний столбец ведущих строк
        if any(v.denominator != 1 for v in coefficients):  # есть дробные
            continue  # следующая степень
        terms = tuple(  # ненулевые члены
            (int(c), powers) for c, powers in zip(coefficients, monomials) if c  # (целый коэффициент, показатели) для c ≠ 0
        )
        if all(  # проверка на всех точках
            sum(c * math.prod(x**p for x, p in zip(row, powers)) for c, powers in terms)  # проверка: p(row) = Σ c·Π x_i^{p_i}
            == y  # равно y
            for row, y in zip(rows, ys)  # по всем строкам
        ):
            return terms  # найден точный многочлен
    return None  # не найдено


def _fit_poly_or_const(pts: list[tuple[int, int]]) -> Optional[Expr]:  # константа или многочлен степени ≤ 3 по точкам
    """Если все y одинаковы — константа; иначе точный многочлен степени ≤ 3; иначе None.
    
        pts: точки (x, y). Возвращает Expr: const(y) или poly(x_0, коэффициенты)."""
    ys = {y for _, y in pts}  # множество значений
    if len(ys) == 1:  # одно значение
        return Expr.const(next(iter(ys)))  # константа
    coeffs = _exact_poly_fit([x for x, _ in pts], [y for _, y in pts], max_degree=3)  # многочлен степени ≤ 3
    if coeffs is not None:  # найден
        return Expr("poly", (Expr.var(0),), coeffs)  # полином по x_0
    return None  # не найдено


def exact_piecewise_poly_fit(  # точная кусочно-полиномиальная / модульная / предикатная подгонка
    train_x: Sequence[tuple[int, ...]], train_y: Sequence[int]  # обучающие строки и значения
) -> Optional[Expr]:  # возвращает выражение или None
    """Точный поиск простой структуры в данных.
    
        Арность 1: проверяется зависимость по вычету x mod m (m = 2..8) — если каждому вычету
        соответствует одно значение, при m = 2 и значениях {0, 1} возвращается x & 1;
        иначе порог t ∈ {−3..3}: слева и справа не менее 3 точек, обе части подгоняются
        многочленом/константой, и если они различны — if(x < t, p_лев, p_прав).
        Арность 2–3: перебираются предикаты (<, ≤, =, x_0 & 1, а при 2 — знак), каждая ветвь
        подгоняется многочленом степени ≤ 2; при различных ветвях — if(предикат, p_ист, p_лож)."""
    arity = len(train_x[0])  # арность
    if arity == 1:  # один аргумент
        xs = [r[0] for r in train_x]  # значения x
        for m in range(2, 9):  # по модулям m = 2..8
            buckets: dict[int, set[int]] = {}  # вычет → множество значений
            for x, y in zip(xs, train_y):  # по точкам
                buckets.setdefault(x % m, set()).add(y)  # добавить y в корзину вычета x mod m
            if all(len(vs) == 1 for vs in buckets.values()) and len(buckets) >= 2:  # в каждой корзине одно значение и корзин ≥ 2
                if m == 2 and buckets.get(0) == {0} and buckets.get(1) == {1}:  # m = 2, чётные → 0, нечётные → 1
                    return Expr("bit_and", (Expr.var(0), Expr.const(1)))  # младший бит: x & 1
        for thr in (-3, -2, -1, 0, 1, 2, 3):  # по порогам
            left = [(x, y) for x, y in zip(xs, train_y) if x < thr]  # точки слева от порога
            right = [(x, y) for x, y in zip(xs, train_y) if x >= thr]  # точки справа
            if len(left) >= 3 and len(right) >= 3:  # в обеих частях не менее 3 точек
                cl = _fit_poly_or_const(left)  # подгонка слева
                cr = _fit_poly_or_const(right)  # подгонка справа
                if cl is not None and cr is not None and cl != cr:  # обе найдены и различны
                    return Expr(  # условное выражение
                        "if_cond",
                        (Expr("lt", (Expr.var(0), Expr.const(thr))), cl, cr),  # if(x < thr, p_лев, p_прав)
                    )
    elif 2 <= arity <= 3:  # арность 2 или 3
        preds: list[tuple[Expr, Callable[[tuple[int, ...]], bool]]] = [  # кандидаты-предикаты (выражение, функция)
            (Expr("lt", (Expr.var(0), Expr.var(1))), lambda r: r[0] < r[1]),  # x_0 < x_1
            (Expr("le", (Expr.var(0), Expr.var(1))), lambda r: r[0] <= r[1]),  # x_0 ≤ x_1
            (Expr("eq", (Expr.var(0), Expr.var(1))), lambda r: r[0] == r[1]),  # x_0 = x_1
            (  # открытие кортежа
                Expr("bit_and", (Expr.var(0), Expr.const(1))),  # x_0 & 1
                lambda r: (r[0] & 1) != 0,  # нечётность x_0
            ),
        ]
        if arity == 2:  # арность 2 — добавить знаковые предикаты
            preds.append(  # добавить
                (Expr("lt", (Expr.var(0), Expr.const(0))), lambda r: r[0] < 0)  # x_0 < 0
            )
            preds.append(  # добавить
                (Expr("lt", (Expr.var(1), Expr.const(0))), lambda r: r[1] < 0)  # x_1 < 0
            )
        for cond_expr, fn in preds:  # по предикатам
            true_pts = [(r, y) for r, y in zip(train_x, train_y) if fn(r)]  # точки, где предикат истинен
            false_pts = [(r, y) for r, y in zip(train_x, train_y) if not fn(r)]  # точки, где предикат ложен
            if len(true_pts) < 6 or len(false_pts) < 6:  # в ветви меньше 6 точек
                continue  # следующий предикат
            tl = _exact_multivariate_poly_fit(  # подгонка истинной ветви
                [r for r, _ in true_pts], [y for _, y in true_pts], arity, max_degree=2  # степень ≤ 2
            )
            tr = _exact_multivariate_poly_fit(  # подгонка ложной ветви
                [r for r, _ in false_pts],  # строки
                [y for _, y in false_pts],  # значения
                arity,  # арность
                max_degree=2,  # степень ≤ 2
            )
            if tl is not None and tr is not None and tl != tr:  # обе найдены и различны
                eb1 = (  # выражение истинной ветви
                    Expr.const(tl[0][0])  # константа
                    if (len(tl) == 1 and sum(tl[0][1]) == 0)  # если единственный член — константа
                    else Expr("poly_nd", (), tl)  # иначе многочлен
                )
                eb2 = (  # выражение ложной ветви
                    Expr.const(tr[0][0])  # константа
                    if (len(tr) == 1 and sum(tr[0][1]) == 0)  # если единственный член — константа
                    else Expr("poly_nd", (), tr)  # иначе многочлен
                )
                return Expr("if_cond", (cond_expr, eb1, eb2))  # if(предикат, p_ист, p_лож)
    return None  # не найдено


# Точные дешёвые шаблоны: применяются до дорогостоящего перебора.
_SKETCH_CAP = 4096  # предел размера эскиза для больших областей
_MODULI = (16, 26, 64, 128, 256, 10, 12, 60, 100, 7, 3, 5, 8, 9, 11,  # кандидаты-модули (первая строка)
           13, 17, 19, 23, 32, 24, 36, 48, 251, 255, 257, 1000, 1024,  # кандидаты-модули (продолжение)
           2, 4, 6, 14, 15, 18, 20, 21, 22, 25, 27, 28, 30, 31)  # кандидаты-модули (конец)
_CURRENT_FAMILY = [""]  # текущее семейство задач (задаётся извне)
_FULL_CAND = [None]  # кандидат на всю область (задаётся извне)


def _mod_affine_expr(X, Y, arity: int):  # точная модульно-аффинная подгонка y = (Σ a_i·x_i + b) mod m
    """Точная подгонка y = (Σ a_i·x_i + b) mod m — O(N) на кандидата (a, b, m).
    
        Покрывает сдвиги и обратные алфавиты без большого перебора. Для арности 1
        a перебирается по 0..m−1, b = (y_0 − a·x_0) mod m; для арности 2 — a_1, a_2 ∈ {−2..2}.
        m выбирается из {max Y + 1, +2, +3} ∪ модулей > max Y, m ≤ 4096. Затем —
        кусочно-аффинная форма y = ((x − lo + c) mod m) + lo на непрерывных отрезках
        входов длины ≥ 4 с постоянным (y − x) mod 26. Возвращает Expr или None."""
    if len(Y) < 4:  # менее 4 примеров
        return None  # нет
    ymax = int(Y.max())  # максимальное значение
    ms = sorted({ymax + d for d in (1, 2, 3)}  # кандидаты m: max Y + 1, +2, +3
                | {m for m in _MODULI if m > ymax})  # и модули из списка, большие max Y; отсортированы
    X0 = X[0]  # первый пример
    y0 = int(Y[0])  # первое значение
    for m in ms:  # по модулям
        if m > 4096:  # модуль слишком велик
            continue  # следующий
        if arity == 1:  # один аргумент
            x = X[:, 0]  # столбец x
            for a in range(m):  # по наклонам a
                b = (y0 - a * int(X0[0])) % m  # сдвиг из первого примера: b = (y_0 − a·x_0) mod m
                if np.array_equal((a * x + b) % m, Y):  # совпадает на всех примерах
                    return Expr("mod", (  # mod(выражение, m)
                        Expr("add", (  # add(
                            Expr("mul", (Expr.var(0), Expr.const(a))),  # mul(x, a)
                            Expr.const(b))),  # константа b
                        Expr.const(m)))  # mod-константа m
        elif arity == 2:  # два аргумента
            x1, x2 = X[:, 0], X[:, 1]  # столбцы x_1, x_2
            for a1 in (-2, -1, 0, 1, 2):  # по a_1
                for a2 in (-2, -1, 0, 1, 2):  # по a_2
                    b = (y0 - a1 * int(X0[0]) - a2 * int(X0[1])) % m  # сдвиг: b = (y_0 − a_1·x_10 − a_2·x_20) mod m
                    if np.array_equal((a1 * x1 + a2 * x2 + b) % m, Y):  # совпадает на всех примерах
                        e = Expr("add", (  # сумма
                            Expr("mul", (Expr.var(0), Expr.const(a1))),  # mul(x_1, a_1)
                            Expr("mul", (Expr.var(1), Expr.const(a2)))))  # mul(x_2, a_2)
                        return Expr("mod", (  # mod(выражение, m)
                            Expr("add", (e, Expr.const(int(b)))),  # add(e, b)
                            Expr.const(m)))  # mod-константа m
    # Кусочно-аффинная форма — алфавиты со смещением: y = (x − lo + c) mod m + lo
    # внутри отрезка [lo, hi); вне отрезка вход отбрасывается проверяющим.
    if arity == 1:  # один аргумент
        order = np.argsort(X[:, 0])  # порядок по возрастанию x
        xs = X[order, 0]  # x по возрастанию
        ys = Y[order]  # соответствующие y
        # непрерывные отрезки входов с постоянным смещением
        lo_i = 0  # начало отрезка
        while lo_i < len(xs):  # пока есть входы
            hi_i = lo_i  # конец отрезка
            offs = (ys - xs) % 26  # смещения (y − x) mod 26
            base = offs[lo_i]  # смещение в начале отрезка
            while (hi_i + 1 < len(xs)  # пока следующий x на 1 больше и смещение то же
                   and int(xs[hi_i + 1]) == int(xs[hi_i]) + 1  # условие продолжения: x следующий = x текущий + 1
                   and offs[hi_i + 1] == base):  # и смещение то же
                hi_i += 1  # расширить отрезок
            if hi_i - lo_i + 1 >= 4:  # длина отрезка ≥ 4
                seg_x = xs[lo_i:hi_i + 1]  # x отрезка
                seg_y = ys[lo_i:hi_i + 1]  # y отрезка
                # форма y = ((x − lo + c) mod m) + lo; перебор c = 1..m−1 для m ∈ {26, 256}
                lo = int(seg_x[0])  # начало отрезка
                for mm in (26, 256):  # по модулям
                    for c in range(1, mm):  # по сдвигам c
                        cand_y = ((seg_x - lo + c) % mm) + lo  # кандидат: ((x − lo + c) mod m) + lo
                        if np.array_equal(cand_y, seg_y):  # совпадает на отрезке
                            # выражение ((x − lo + c) mod m) + lo собирается из доступных операций
                            shifted = Expr("add", (Expr.var(0),  # x + (c − lo)
                                                   Expr.const(c - lo)))  # константа c − lo
                            inner = Expr("mod", (shifted,  # mod(сдвиг, m)
                                                 Expr.const(mm)))  # модуль m
                            inner = Expr("add", (inner,  # + lo
                                                 Expr.const(lo)))  # константа lo
                            in_rng = Expr("in_ranges",  # условие x ∈ [x_нач, x_кон]
                                          (Expr.var(0),),  # аргумент x
                                          ((int(seg_x[0]),  # начало отрезка
                                            int(seg_x[-1])),))  # конец отрезка
                            return Expr("if_cond", (in_rng, inner,  # if(в отрезке, выражение, x)
                                                    Expr.var(0)))  # иначе — x
            lo_i = hi_i + 1  # следующий отрезок
    return None  # не найдено


def _sparse_membership(X, Y):  # разреженная принадлежность (in_set / in_ranges)
    """Разреженная принадлежность: in_set или in_ranges для бинарного 0/1 выхода.
    
        Допускается только для языкового семейства с унарным выходом {0, 1};
        корректность на всей области проверяется вызывающим. S — значения x, где y = 1;
        candidate in_set(S), а если S содержит серии подряд идущих чисел — ещё и
        in_ranges(серии). Возвращает кандидат с минимальным числом бит или None."""
    x = X[:, 0]  # столбец x
    S = tuple(sorted(int(v) for v in x[Y == 1]))  # S — отсортированные x, где y = 1
    if not S or len(S) == len(Y) or max(S) >= 4096 or min(S) < 0:  # пусто, всё множество или вне диапазона [0, 4096)
        return None  # не найдено
    cands = [Expr("in_set", (Expr.var(0),), S)]  # кандидат: x ∈ S
    runs = []  # серии подряд идущих чисел
    for v in S:  # по элементам S
        if runs and v == runs[-1][1] + 1:  # продолжает последнюю серию
            runs[-1] = (runs[-1][0], v)  # расширить серию
        else:  # новая серия
            runs.append((v, v))  # серия [v, v]
    if len(runs) < len(S):  # есть серии длины > 1
        cands.append(Expr("in_ranges", (Expr.var(0),), tuple(runs)))  # кандидат: x ∈ объединения серий
    return min(cands, key=lambda e: e.bits())  # кандидат с минимальным числом бит


def synthesize_batched_mdl(  # синтез выражения методом минимальной длины описания (MDL)
    arity: int,  # арность
    train_x: Sequence[tuple[int, ...]],  # обучающие строки
    train_y: Sequence[int],  # обучающие значения
    val_x: Sequence[tuple[int, ...]],  # проверочные строки
    val_y: Sequence[int],  # проверочные значения
    library_experts: Sequence[tuple[int, str, "Expert"]] = (),  # библиотека экспертов (индекс, имя, эксперт)
    max_evals: int = 20000,  # предел проверенных выражений
    max_bits: int = 128,  # бюджет бит для стоимости выражения
    deadline: Optional[float] = None,  # предельное время (monotonic) или None
) -> tuple[Optional[Expr], int]:  # возвращает (выражение или None, число проверенных)
    """Синтез выражения с минимальной длиной описания: перебор по возрастанию стоимости.
    
        Строится пул выражений со значениями на train (векторизованно, пакетами). На каждом шаге
        берётся самая дешёвая корзина, к её выражениям применяются унарные и бинарные операции
        и вызовы экспертов из библиотеки; результаты, равные Y на train, проверяются на val.
        Стоимость = сумма бит токенов; порог max_bits. Дубликаты по вектору значений отбрасываются
        (не более 2 представителей). «Встреча посередине»: Y − p, p − Y, p + Y, p ⊕ Y, Y / p ищутся
        в пуле. До перебора проверяются дешёвые точные шаблоны (модульно-аффинный, принадлежность).
        Для N > 4096 — поиск на эскизе с проверкой на всём train. Возвращает (expr, проверено)."""
    X = np.asarray(train_x, dtype=np.int64)  # матрица входов (N, arity)
    Y = np.asarray(train_y, dtype=np.int64)  # вектор значений (N,)
    if X.size and int(np.max(np.abs(X))) > 2048:  # вход по модулю больше 2048
        # предел величины входа: байтовый пул (0..255) и кириллические кодовые точки должны проходить;
        # безопасность промежуточных значений обеспечивает MAG_CAP
        return None, 0  # отказ: ничего не проверено

    # Точные дешёвые шаблоны применяются ДО дорогостоящего перебора.
    # Строгость не ослабляется: каждый кандидат проходит полную проверку
    # вызывающего на train, val и всей области.
    env = {nm: ex for _, nm, ex in library_experts}  # среда: имя эксперта → эксперт
    # Кандидат на всю область (задаётся извне) имеет приоритет —
    # он всё равно проверяется при коммите на всей области.
    fc = _FULL_CAND[0]  # кандидат на всю область
    if fc is not None and arity == 1:  # кандидат есть и арность 1
        try:  # попытка вычислить
            if all(fc.eval(r, env) == y for r, y in zip(val_x, val_y)):  # совпадает на val
                return fc, 1  # вернуть кандидата (1 проверка)
        except Exception:  # ошибка вычисления
            pass  # продолжить
    if X.size and arity in (1, 2):  # есть данные и арность 1 или 2
        # модульно-аффинный шаблон y = (Σ a_i·x_i + b) mod m — O(N) на (a, b, m)
        cand = _mod_affine_expr(X, Y, arity)  # кандидат
        if cand is not None:  # найден
            try:  # попытка
                if all(cand.eval(r, env) == y  # совпадает на val
                       for r, y in zip(val_x, val_y)):  # по val
                    return cand, 1  # вернуть
            except Exception:  # ошибка
                pass  # продолжить
        # разреженная принадлежность (семейство задаётся извне; только бинарный унарный выход)
        # семейство — языковое, выходы только 0/1
        if arity == 1 and _CURRENT_FAMILY[0] == "language"                 and set(np.unique(Y)) <= {0, 1}:
            cand = _sparse_membership(X, Y)  # кандидат
            if cand is not None:  # найден
                try:  # попытка
                    if all(cand.eval(r, env) == y  # совпадает на val
                           for r, y in zip(val_x, val_y)):  # по val
                        return cand, 1  # вернуть
                except Exception:  # ошибка
                    pass  # продолжить

    # эскизный MDL для огромных областей: поиск на стратифицированной подвыборке
    # (≤ 4096 строк), затем проверка кандидата на ВСЕМ train (проверка val —
    # при коммите). Память O(P·CAP) вместо O(P·N); без ограничения по времени.
# исключает нехватку памяти.
    N = X.shape[0]  # число примеров
    if N > _SKETCH_CAP and deadline is None:  # слишком много примеров и нет дедлайна
        for cap in (_SKETCH_CAP, 2 * _SKETCH_CAP):  # по размерам эскиза 4096 и 8192
            idx = np.unique((np.arange(cap) * N // cap).astype(np.int64))  # стратифицированные индексы: i·N // cap
            sx = [train_x[int(i)] for i in idx]  # эскиз входов
            sy = [int(train_y[int(i)]) for i in idx]  # эскиз значений
            e_sk, _n = synthesize_batched_mdl(  # рекурсивный поиск на эскизе
                arity, sx, sy, val_x, val_y, library_experts,  # те же параметры
                max_evals, max_bits, deadline)  # параметры поиска
            if e_sk is None:  # не найдено
                continue  # следующий размер
            try:  # проверка на всём train
                if all(e_sk.eval(r, env) == y  # совпадает
                       for r, y in zip(train_x, train_y)):  # по train
                    return e_sk, _n  # вернуть
            except Exception:  # ошибка
                continue  # следующий размер
        return None, 0  # эскиз не удался — честный отказ без полного перебора O(P·N)
    N = X.shape[0]  # число примеров
    leaf_exprs = [Expr.var(i) for i in range(arity)] + [  # листья: переменные x_0..x_{arity−1} и константы
        Expr.const(c) for c in CONSTS  # константы из CONSTS
    ]
    L_arr = np.stack(  # значения листьев (число листьев, N)
        [  # открытие списка
            (  # открытие выражения
                np.ascontiguousarray(X[:, int(lf.value or 0)])  # переменная: столбец X
                if lf.op == "var"  # если лист — переменная
                else np.full(N, int(lf.value), dtype=np.int64)  # константа: столбец из значения
            )
            for lf in leaf_exprs  # по листьям
        ]
    )
    L_bits = [lf.bits() for lf in leaf_exprs]  # стоимость листьев в битах

    env = {name: ex for _, name, ex in library_experts}  # среда (переопределяется)
    ex_idx = {name: idx for idx, name, _ in library_experts}  # имя эксперта → индекс (для стоимости вызова)

    for i, lf in enumerate(leaf_exprs):  # по листьям
        if np.array_equal(L_arr[i], Y):  # лист уже равен Y
            if all(lf.eval(r, env) == y for r, y in zip(val_x, val_y)):  # и на val
                return lf, 1  # вернуть лист

    is_bool_task = (  # признак булевой задачи
        (X.min() >= 0 and X.max() <= 1 and Y.min() >= 0 and Y.max() <= 1)  # все входы и выходы в {0, 1}
        if N > 0  # если есть примеры
        else False  # иначе нет
    )
    useful_libs: list[tuple[int, str, Expert]] = []  # полезные эксперты библиотеки
    for idx, nm, ex in library_experts:  # по экспертам
        if ex.program.arity > arity or ex.program.expr is None:  # арность больше или выражения нет
            continue  # пропустить
        if not is_bool_task and ex.family == "BoolVenn":  # небулева задача — булевы эксперты не нужны
            continue  # пропустить
        if (  # проверка: эксперт — простая бинарная операция над переменными
            ex.program.arity == 2  # арность 2
            and ex.program.expr.op in OPS  # операция из OPS
            and all(a.op == "var" for a in ex.program.expr.args)  # все аргументы — переменные
        ):
            continue  # пропустить (дешевле построить напрямую)
        useful_libs.append((idx, nm, ex))  # оставить
    useful_libs.sort(key=lambda it: (it[2].program.cost, -it[0]))  # сортировка по стоимости, затем по убыванию индекса
    useful_libs = useful_libs[:32]  # не более 32

    seen_sigs: dict[bytes, int] = {}  # сколько раз встречался вектор значений
    pool_hash: dict[bytes, Expr] = {}  # вектор значений → выражение
    pool_exprs: list[Expr] = []  # пул выражений
    pool_vals: list[np.ndarray] = []  # пул векторов значений
    buckets: dict[int, list[tuple[Expr, np.ndarray]]] = {}  # корзины: стоимость → (выражение, значения)

    for i, lf in enumerate(leaf_exprs):  # по листьям
        key = L_arr[i].tobytes()  # ключ — байты значений
        seen_sigs[key] = 1  # встречен 1 раз
        pool_hash[key] = lf  # в пуле
        pool_exprs.append(lf)  # добавить выражение
        pool_vals.append(L_arr[i])  # добавить значения
        buckets.setdefault(L_bits[i], []).append((lf, L_arr[i]))  # в корзину своей стоимости

    total_checked = len(leaf_exprs)  # проверено — столько же, сколько листьев
    MAG_CAP = 1 << 28  # предел величины промежуточных значений 2^28

    def verify_val(cand: Expr) -> bool:  # проверить кандидата на val
        try:  # попытка
            return all(cand.eval(r, env) == y for r, y in zip(val_x, val_y))  # совпадает на val
        except Exception:  # ошибка
            return False  # не проходит

    while buckets and total_checked < max_evals:  # пока есть корзины и лимит не исчерпан
        if deadline is not None and time.monotonic() >= deadline:  # дедлайн истёк
            break  # выход
        min_b = min(buckets)  # минимальная стоимость
        if min_b >= max_bits:  # достигнут предел бит
            break  # выход
        batch = buckets.pop(min_b)  # взять корзину
        if not batch:  # пустая
            continue  # следующая
        if len(batch) > 512:  # пакет больше 512
            batch = batch[:512]  # обрезать
        B_exprs = [x[0] for x in batch]  # выражения пакета
        V = np.stack([x[1] for x in batch])  # значения пакета (B, N)

        def register(  # зарегистрировать результаты: найти совпадения с Y и добавить в пул
            res_arr: np.ndarray, builder_fn: Callable[[int, int], Expr], new_bits: int  # результаты, построитель выражения по (индекс пакета, индекс листа), новая стоимость
        ) -> Optional[Expr]:  # возвращает найденное выражение или None
            nonlocal total_checked  # изменение счётчика внешней функции
            if new_bits > max_bits:  # стоимость выше предела
                return None  # не регистрировать
            if res_arr.ndim == 2:  # двумерный результат
                res_arr = res_arr[:, None, :]  # добавить ось листа: (B, 1, N)
            matches = np.all(res_arr == Y[None, None, :], axis=2)  # совпадения с Y по всей строке: (B, L)
            if np.any(matches):  # есть совпадения
                for bi, ki in np.argwhere(matches):  # по парам (пакет, лист)
                    cand = builder_fn(int(bi), int(ki))  # построить выражение
                    if verify_val(cand):  # проходит val
                        return cand  # найдено
            if new_bits + TOKEN_BITS > max_bits or total_checked >= max_evals:  # следующий уровень уже не влезает или лимит исчерпан
                return None  # не добавлять
            ok_mag = np.max(np.abs(res_arr), axis=2) <= MAG_CAP  # строки, где все значения в пределах MAG_CAP: (B, L)
            for bi, ki in np.argwhere(ok_mag):  # по допустимым парам
                if total_checked >= max_evals:  # лимит
                    break  # выход
                row = res_arr[bi, ki]  # вектор значений
                key = row.tobytes()  # ключ
                cnt = seen_sigs.get(key, 0)  # сколько раз встречался
                if cnt >= 2:  # уже два представителя
                    continue  # пропустить
                expr = builder_fn(int(bi), int(ki))  # построить выражение
                seen_sigs[key] = cnt + 1  # счётчик + 1
                if key not in pool_hash:  # в пуле ещё нет
                    pool_hash[key] = expr  # добавить в хэш
                    if len(pool_exprs) < 3500:  # пул не переполнен
                        pool_exprs.append(expr)  # добавить выражение
                        pool_vals.append(row)  # добавить значения
                total_checked += 1  # проверено + 1
                buckets.setdefault(new_bits, []).append((expr, row))  # в корзину новой стоимости
            return None  # не найдено

        V_sq_ok = np.where(np.abs(V) <= 32768, V * V, MAG_CAP + 1)  # квадрат, если |V| ≤ 32768, иначе метка переполнения
        V_cb_ok = np.where(np.abs(V) <= 1024, V * V * V, MAG_CAP + 1)  # куб, если |V| ≤ 1024, иначе метка переполнения
        for uop, ures in (  # по унарным операциям
            ("neg", -V),  # −V
            ("abs", np.abs(V)),  # |V|
            ("sign", np.sign(V).astype(np.int64)),  # знак V
            ("bit_not8", (~V) & 0xFF),  # побитное НЕ в 8 битах
            ("clamp3", np.clip(V, -3, 3)),  # обрезка в [−3, 3]
            ("square", V_sq_ok),  # квадрат
            ("cube", V_cb_ok),  # куб
        ):
            hit = register(  # зарегистрировать
                ures, lambda bi, _, o=uop: Expr(o, (B_exprs[bi],)), min_b + TOKEN_BITS  # выражение op(B_exprs[bi]), стоимость + бит токена
            )
            if hit is not None:  # найдено
                return hit, total_checked  # вернуть

        B_len = V.shape[0]  # размер пакета
        non_const_mask = np.array([e.op != "const" for e in B_exprs], dtype=bool)  # маска неконстант
        V_min = np.min(V, axis=1)  # минимум по строкам
        V_max = np.max(V, axis=1)  # максимум по строкам
        for lib_idx, ename, ex in useful_libs:  # по полезным экспертам
            cbits = TOKEN_BITS + gamma_bits(lib_idx + 1)  # стоимость вызова: токен + гамма-код индекса
            if ex.program.arity == 1:  # унарный эксперт
                if min_b + cbits > max_bits:  # не влезает
                    continue  # следующий
                lo_d, hi_d = ex.domains[0][0], ex.domains[0][-1]  # границы области определения аргумента
                cand_mask = non_const_mask & (V_min >= lo_d) & (V_max <= hi_d)  # кандидаты: неконстанта и все значения в области
                ok_idxs = np.flatnonzero(cand_mask)  # их индексы
                if ok_idxs.size > 0:  # есть
                    flat_in = V[ok_idxs].reshape(-1, 1)  # входы столбцом
                    res_flat, ok_flat = eval_expr_vec(ex.program.expr, flat_in, env)  # векторизованное вычисление и флаг корректности
                    res_2d = res_flat.reshape(ok_idxs.size, N)  # в форму (K, N)
                    ok_row_mask = np.all(ok_flat.reshape(ok_idxs.size, N), axis=1)  # строки без ошибок
                    if np.any(ok_row_mask):  # есть
                        final_idxs = ok_idxs[ok_row_mask]  # их индексы
                        hit = register(  # зарегистрировать
                            res_2d[ok_row_mask],  # результаты безошибочных строк
                            lambda bi, _, o=ename, idxs=final_idxs: Expr(  # построитель вызова
                                "call", (B_exprs[int(idxs[bi])],), o  # call(B_exprs[idx], имя)
                            ),
                            min_b + cbits,  # стоимость: min_b + стоимость вызова
                        )
                        if hit is not None:  # найдено
                            return hit, total_checked  # вернуть
            elif ex.program.arity == 2:  # бинарный эксперт
                d0_lo, d0_hi = ex.domains[0][0], ex.domains[0][-1]  # область первого аргумента
                d1_lo, d1_hi = ex.domains[1][0], ex.domains[1][-1]  # область второго аргумента
                for li, lf in enumerate(leaf_exprs):  # по листьям (второй аргумент)
                    if min_b + cbits + L_bits[li] > max_bits:  # не влезает
                        continue  # следующий
                    lv = L_arr[li]  # значения листа
                    lv_min, lv_max = int(np.min(lv)), int(np.max(lv))  # его минимум и максимум
                    base_mask = (  # базовая маска
                        non_const_mask  # для константы-листа — только неконстанты
                        if lf.op == "const"
                        else np.ones(B_len, dtype=bool)  # иначе все
                    )
                    for order_idx in (0, 1):  # по порядкам аргументов (B, лист) и (лист, B)
                        if order_idx == 0:  # порядок (B, лист)
                            if (  # проверка области листа
                                lv_min < d1_lo  # лист ниже области
                                or lv_max > d1_hi  # или выше области
                                or (ex.valid_rule == "nonzero_y" and np.any(lv == 0))  # правило «y ≠ 0» и лист содержит 0
                            ):
                                continue  # следующий порядок
                            cand_mask = base_mask & (V_min >= d0_lo) & (V_max <= d0_hi)  # кандидаты по первому аргументу
                        else:  # порядок (лист, B)
                            if lv_min < d0_lo or lv_max > d0_hi:  # лист вне первой области
                                continue  # следующий
                            cand_mask = base_mask & (V_min >= d1_lo) & (V_max <= d1_hi)  # кандидаты по второму аргументу
                            if ex.valid_rule == "nonzero_y":  # правило «y ≠ 0»
                                cand_mask &= np.all(V != 0, axis=1)  # исключить строки с нулями
                        ok_idxs = np.flatnonzero(cand_mask)  # индексы
                        if ok_idxs.size == 0:  # нет
                            continue  # следующий  # следующий порядок аргументов
                        K_ok = ok_idxs.size  # число кандидатов  # K_ok — число выражений пакета, попавших в область первого аргумента: K_ok = |ok_idxs|
                        V_sel = V[ok_idxs]  # их значения  # строки пакета-кандидата: V_sel[k, :] = V[ok_idxs[k], :], форма (K_ok, N)
                        L_rep = np.broadcast_to(lv[None, :], (K_ok, N))  # лист, повторённый K строк  # значения листа, повторённые K_ok раз: L_rep[k, :] = lv, форма (K_ok, N)
                        flat_in = (  # входы  # входы вызова эксперта — пары, развёрнутые построчно в (K_ok·N, 2): строка k·N + n
                            np.column_stack((V_sel.ravel(), L_rep.ravel()))  # (B, лист)  # порядок (выражение пакета, лист): столбцы (V_sel, L_rep)
                            if order_idx == 0  # порядок (B, лист)  # если порядок аргументов 0 — выражение пакета первым
                            else np.column_stack((L_rep.ravel(), V_sel.ravel()))  # (лист, B)  # иначе порядок (лист, выражение пакета): столбцы (L_rep, V_sel)
                        )
                        res_flat, ok_flat = eval_expr_vec(ex.program.expr, flat_in, env)  # вычисление  # векторизованное вычисление выражения эксперта на flat_in: значения res_flat и флаги корректности ok_flat, длины K_ok·N
                        res_2d = res_flat.reshape(K_ok, N)  # в форму (K, N)  # результат в форму (K_ok, N): res_2d[k, n] = res_flat[k·N + n]
                        ok_row_mask = np.all(ok_flat.reshape(K_ok, N), axis=1)  # строки без ошибок  # маска строк без ошибок: ok_row_mask[k] = все ok_flat[k·N + n] истинны
                        if np.any(ok_row_mask):  # есть  # есть строки, вычисленные без ошибок
                            final_idxs = ok_idxs[ok_row_mask]  # индексы  # final_idxs = ok_idxs[ok_row_mask] — индексы выражений пакета для этих строк
                            hit = register(  # зарегистрировать  # зарегистрировать результаты вызова и, при совпадении с Y, вернуть выражение
                                res_2d[ok_row_mask],  # результаты безошибочных строк  # результаты безошибочных строк: res_2d[ok_row_mask], форма (K', N)
                                (  # открытие выражения  # построитель выражения по индексу bi строки результата
                                    lambda bi, _, o=ename, l=lf, idxs=final_idxs, oi=order_idx: Expr(  # построитель вызова  # лямбда с зафиксированными параметрами: имя эксперта o, лист l, индексы idxs, порядок oi
                                        "call",  # вызов эксперта по имени o
                                        (  # открытие выражения  # порядок 0: аргументы (выражение пакета с индексом idxs[bi], лист l)
                                            (B_exprs[int(idxs[bi])], l)  # call(B, лист)  # если порядок аргументов 0
                                            if oi == 0  # порядок 0  # иначе порядок 1: аргументы (лист l, выражение пакета с индексом idxs[bi])
                                            else (l, B_exprs[int(idxs[bi])])  # call(лист, B)
                                        ),
                                        o,  # имя операции  # имя вызываемого эксперта
                                    )
                                ),
                                min_b + cbits + L_bits[li],  # стоимость: min_b + вызов + лист  # стоимость результата: min_b + стоимость вызова cbits + стоимость листа L_bits[li]
                            )
                            if hit is not None:  # найдено  # найдено совпадение с Y
                                return hit, total_checked  # вернуть  # вернуть выражение и число проверенных

        Va = V[:, None, :]  # значения пакета (B, 1, N)  # значения пакета с осью листа: Va[b, 0, n] = V[b, n], форма (B, 1, N)
        Lb = L_arr[None, :, :]  # значения листьев (1, L, N)  # значения листьев с осью пакета: Lb[0, l, n] = L_arr[l, n], форма (1, L, N)
        for op in BINARY_OPS:  # по бинарным операциям  # по всем бинарным операциям
            if op == "add":  # сложение  # операция сложения
                R = Va + Lb  # результат: R = V + L  # R[b, l, n] = Va[b, n] + L_arr[l, n] (по правилам вещания, форма (B, L, N))
            elif op == "sub":  # вычитание  # операция вычитания
                R = Va - Lb  # результат: R = V − L  # R[b, l, n] = Va[b, n] − L_arr[l, n]
            elif op == "mul":  # умножение  # операция умножения
                ok_m = (np.abs(Va) <= 32768) & (np.abs(Lb) <= 32768)  # оба операнда ≤ 32768 по модулю  # допустимые пары: |Va| ≤ 32768 и |Lb| ≤ 32768 (произведение не выходит за MAG_CAP)
                sa = np.where(ok_m, Va, 0)  # V с обнулением недопустимых  # первый сомножитель: Va, если пара допустима, иначе 0
                sb = np.where(ok_m, Lb, 0)  # L с обнулением недопустимых  # второй сомножитель: Lb, если пара допустима, иначе 0
                R = np.where(ok_m, sa * sb, MAG_CAP + 1)  # произведение или метка переполнения  # R = sa·sb для допустимых пар, иначе метка переполнения MAG_CAP + 1
            elif op == "min":  # минимум  # операция минимума
                R = np.minimum(Va, Lb)  # результат: R = min(V, L)  # R[b, l, n] = min(Va[b, n], Lb[0, l, n])
            elif op == "max":  # максимум  # операция максимума
                R = np.maximum(Va, Lb)  # результат: R = max(V, L)  # R[b, l, n] = max(Va[b, n], Lb[0, l, n])
            elif op == "eq":  # равенство  # операция равенства
                R = (Va == Lb).astype(np.int64)  # результат: R = 1[V = L]  # R = 1, если Va = Lb, иначе 0
            elif op == "ne":  # неравенство  # операция неравенства
                R = (Va != Lb).astype(np.int64)  # результат: R = 1[V ≠ L]  # R = 1, если Va ≠ Lb, иначе 0
            elif op == "lt":  # меньше  # операция «меньше»
                R = (Va < Lb).astype(np.int64)  # результат: R = 1[V < L]  # R = 1, если Va < Lb, иначе 0
            elif op == "le":  # меньше или равно  # операция «меньше или равно»
                R = (Va <= Lb).astype(np.int64)  # результат: R = 1[V ≤ L]  # R = 1, если Va ≤ Lb, иначе 0
            elif op == "gt":  # больше  # операция «больше»
                R = (Va > Lb).astype(np.int64)  # результат: R = 1[V > L]  # R = 1, если Va > Lb, иначе 0
            elif op == "ge":  # больше или равно  # операция «больше или равно»
                R = (Va >= Lb).astype(np.int64)  # результат: R = 1[V ≥ L]  # R = 1, если Va ≥ Lb, иначе 0
            elif op == "bit_and":  # побитное И  # побитовое И
                R = Va & Lb  # результат: R = V & L  # R = Va & Lb
            elif op == "bit_or":  # побитное ИЛИ  # побитовое ИЛИ
                R = Va | Lb  # результат: R = V | L  # R = Va | Lb
            elif op == "bit_xor":  # побитное исключающее ИЛИ  # побитовое исключающее ИЛИ
                R = Va ^ Lb  # результат: R = V ⊕ L  # R = Va ⊕ Lb
            elif op == "bit_nand8":  # И-НЕ в 8 битах  # 8-битное И-НЕ
                R = (~(Va & Lb)) & 0xFF  # результат: R = ¬(V & L) & 0xFF  # R = ¬(Va & Lb) & 0xFF (младшие 8 бит)
            elif op == "bit_nor8":  # ИЛИ-НЕ в 8 битах  # 8-битное ИЛИ-НЕ
                R = (~(Va | Lb)) & 0xFF  # результат: R = ¬(V | L) & 0xFF  # R = ¬(Va | Lb) & 0xFF
            elif op == "bit_xnor8":  # исключающее ИЛИ-НЕ в 8 битах  # 8-битное исключающее ИЛИ-НЕ
                R = (~(Va ^ Lb)) & 0xFF  # результат: R = ¬(V ⊕ L) & 0xFF  # R = ¬(Va ⊕ Lb) & 0xFF
            elif op == "floordiv":  # целочисленное деление  # целочисленное деление
                safe = np.where(Lb != 0, Lb, 1)  # делитель с заменой 0 на 1  # делитель без нуля: Lb, если Lb ≠ 0, иначе 1
                R = np.where(  # деление, где делитель ≠ 0, иначе метка  # R = Va ÷ Lb (округление вниз), если Lb ≠ 0, иначе MAG_CAP + 1
                    np.broadcast_to(Lb != 0, np.broadcast_shapes(Va.shape, Lb.shape)),  # условие L ≠ 0  # условие, развёрнутое на форму (B, L, N): Lb ≠ 0
                    np.floor_divide(Va, safe),  # деление: V // L  # частное с округлением вниз: ⌊Va / safe⌋
                    MAG_CAP + 1,  # иначе метка переполнения  # при делителе 0 результат — метка переполнения MAG_CAP + 1
                )
            elif op == "mod":  # остаток  # остаток от деления
                safe = np.where(Lb != 0, Lb, 1)  # делитель с заменой 0 на 1  # делитель без нуля: Lb, если Lb ≠ 0, иначе 1
                R = np.where(  # остаток, где делитель ≠ 0, иначе метка  # R = Va mod Lb, если Lb ≠ 0, иначе MAG_CAP + 1
                    np.broadcast_to(Lb != 0, np.broadcast_shapes(Va.shape, Lb.shape)),  # условие L ≠ 0  # условие, развёрнутое на форму (B, L, N): Lb ≠ 0
                    np.mod(Va, safe),  # остаток: V mod L  # остаток Va mod safe (знак делимого)
                    MAG_CAP + 1,  # иначе метка переполнения  # при делителе 0 результат — метка переполнения MAG_CAP + 1
                )
            elif op == "gcd":  # НОД
                R = np.gcd(Va, Lb)  # R = НОД(V, L)  # R = НОД(Va, Lb)
            elif op == "lcm":  # НОК  # наименьшее общее кратное
                ok_l = (np.abs(Va) <= 10000) & (np.abs(Lb) <= 10000)  # оба операнда ≤ 10000  # допустимые пары: |Va| ≤ 10000 и |Lb| ≤ 10000 (кратное не выходит за MAG_CAP)
                sa = np.where(ok_l, Va, 0)  # V с обнулением  # первый аргумент: Va, если пара допустима, иначе 0
                sb = np.where(ok_l, Lb, 0)  # L с обнулением  # второй аргумент: Lb, если пара допустима, иначе 0
                R = np.where(ok_l, np.lcm(sa, sb), MAG_CAP + 1)  # НОК или метка  # R = НОК(sa, sb) для допустимых пар, иначе MAG_CAP + 1
            elif op == "pow_int":  # целая степень  # возведение в целую степень
                ok_p = (Lb >= 0) & (Lb <= 8) & (np.abs(Va) <= 32)  # показатель 0..8 и основание ≤ 32  # допустимые пары: 0 ≤ Lb ≤ 8 (показатель) и |Va| ≤ 32 (основание)
                sa = np.where(ok_p, Va, 0)  # основание с обнулением недопустимого  # основание: Va, если пара допустима, иначе 0
                sb = np.where(ok_p, Lb, 0)  # показатель с обнулением недопустимого  # показатель: Lb, если пара допустима, иначе 0
                R = np.where(ok_p, sa**sb, MAG_CAP + 1)  # V^L или метка  # R = sa^sb для допустимых пар, иначе MAG_CAP + 1
            else:  # неизвестная операция  # неизвестная операция
                continue  # пропустить  # пропустить
            for li in range(len(leaf_exprs)):  # по листьям  # по индексам листьев
                hit = register(  # зарегистрировать op(V, L)  # зарегистрировать результат среза по листу li: R[:, li:li+1, :] (форма (B, 1, N))
                    R[:, li : li + 1, :],  # срез по листу li  # срез R[:, li : li + 1, :] — только столбец листа li
                    lambda bi, _, o=op, l=leaf_exprs[li]: Expr(o, (B_exprs[bi], l)),  # построитель op(B, лист)  # построитель: op(выражение пакета bi, лист l)
                    min_b + TOKEN_BITS + L_bits[li],  # стоимость: min_b + токен + лист  # стоимость: min_b + TOKEN_BITS + стоимость листа L_bits[li]
                )
                if hit is not None:  # найдено  # найдено
                    return hit, total_checked  # вернуть  # вернуть выражение и число проверенных
            if op not in COMMUTATIVE_OPS:  # некоммутативная — проверить обратный порядок op(L, V)  # для некоммутативной операции проделать то же с обратным порядком аргументов
                if op == "sub":  # вычитание  # вычитание
                    R2 = Lb - Va  # результат: R = L − V  # R2 = Lb − Va
                elif op == "lt":  # «меньше»
                    R2 = (Lb < Va).astype(np.int64)  # результат: R = 1[L < V]  # R2 = 1, если Lb < Va, иначе 0
                elif op == "le":  # «меньше или равно»
                    R2 = (Lb <= Va).astype(np.int64)  # результат: R = 1[L ≤ V]  # R2 = 1, если Lb ≤ Va, иначе 0
                elif op == "gt":  # «больше»
                    R2 = (Lb > Va).astype(np.int64)  # результат: R = 1[L > V]  # R2 = 1, если Lb > Va, иначе 0
                elif op == "ge":  # «больше или равно»
                    R2 = (Lb >= Va).astype(np.int64)  # результат: R = 1[L ≥ V]  # R2 = 1, если Lb ≥ Va, иначе 0
                elif op == "floordiv":  # целочисленное деление
                    safe = np.where(Va != 0, Va, 1)  # делитель с заменой 0 на 1  # делитель без нуля: Va, если Va ≠ 0, иначе 1
                    R2 = np.where(  # результат, где делитель ≠ 0, иначе метка  # R2 = Lb ÷ Va, если Va ≠ 0, иначе MAG_CAP + 1
                        np.broadcast_to(  # распространить условие  # условие, развёрнутое на форму (B, L, N)
                            Va != 0, np.broadcast_shapes(Va.shape, Lb.shape)  # условие: V ≠ 0  # условие: Va ≠ 0
                        ),
                        np.floor_divide(Lb, safe),  # деление: L // V  # частное с округлением вниз: ⌊Lb / safe⌋
                        MAG_CAP + 1,  # иначе метка переполнения  # при делителе 0 — метка переполнения MAG_CAP + 1
                    )
                elif op == "mod":  # остаток от деления
                    safe = np.where(Va != 0, Va, 1)  # делитель с заменой 0 на 1  # остаток от деления
                    R2 = np.where(  # результат, где делитель ≠ 0, иначе метка  # R2 = Lb mod Va, если Va ≠ 0, иначе MAG_CAP + 1
                        np.broadcast_to(  # распространить условие  # условие, развёрнутое на форму (B, L, N)
                            Va != 0, np.broadcast_shapes(Va.shape, Lb.shape)  # условие: V ≠ 0  # условие: Va ≠ 0
                        ),
                        np.mod(Lb, safe),  # остаток: L mod V  # остаток Lb mod safe
                        MAG_CAP + 1,  # иначе метка переполнения  # при делителе 0 — метка переполнения MAG_CAP + 1
                    )
                elif op == "pow_int":  # возведение в целую степень
                    ok_p = (Va >= 0) & (Va <= 8) & (np.abs(Lb) <= 32)  # показатель 0..8 и основание ≤ 32  # допустимые пары: 0 ≤ Va ≤ 8 (показатель) и |Lb| ≤ 32 (основание)
                    sa = np.where(ok_p, Lb, 0)  # основание с обнулением  # основание: Lb, если пара допустима, иначе 0
                    sb = np.where(ok_p, Va, 0)  # показатель с обнулением  # показатель: Va, если пара допустима, иначе 0
                    R2 = np.where(ok_p, sa**sb, MAG_CAP + 1)  # L^V или метка  # R2 = sa^sb для допустимых пар, иначе MAG_CAP + 1
                else:  # иначе  # для прочих операций обратный вариант не строится
                    R2 = None  # нет обратной формы  # обратного результата нет
                if R2 is not None:  # обратная форма есть  # если обратный вариант есть
                    for li in range(len(leaf_exprs)):  # по листьям  # по индексам листьев
                        hit = register(  # зарегистрировать op(L, V)  # зарегистрировать обратный результат среза по листу li
                            R2[:, li : li + 1, :],  # срез по листу li  # срез R2[:, li : li + 1, :] — только столбец листа li
                            lambda bi, _, o=op, l=leaf_exprs[li]: Expr(  # построитель  # построитель: op(лист l, выражение пакета bi)
                                o, (l, B_exprs[bi])  # op(лист, B)  # аргументы (лист l, выражение пакета B_exprs[bi])
                            ),
                            min_b + TOKEN_BITS + L_bits[li],  # стоимость: min_b + токен + лист  # стоимость: min_b + TOKEN_BITS + стоимость листа L_bits[li]
                        )
                        if hit is not None:  # найдено  # найдено
                            return hit, total_checked  # вернуть  # вернуть выражение и число проверенных

        if len(pool_vals) >= 4 and min_b >= 19:  # «встреча посередине»: пул ≥ 4 и стоимость ≥ 19 бит  # «встреча посередине»: в пуле не меньше 4 векторов и стоимость min_b ≥ 19 бит
            P_mat = np.stack(pool_vals)  # матрица значений пула (P, N)  # матрица значений пула P_mat: (P, N)
            mitm_hits: list[tuple[int, Expr]] = []  # найденные кандидаты  # найденные кандидаты «встречи»: (стоимость в битах, выражение)
            for op_name, mat, rev in (  # по формулам: (операция, искомый вектор, обратный порядок аргументов)  # по вариантам: имя операции, матрица mat (P, N) и признак обратного порядка аргументов rev
                ("add", Y[None, :] - P_mat, False),  # p + ? = Y → ? = Y − p  # Y − p: если p = Y − q, то q = p + Y; ищем q в пуле
                ("sub", P_mat - Y[None, :], False),  # p − ? = Y → ? = p − Y  # p − Y: если p = q − Y, то q = p + Y
                ("sub", P_mat + Y[None, :], True),  # ? − p = Y → ? = p + Y  # p + Y: если p = q + Y, то q = p − Y (порядок аргументов обратный)
                ("bit_xor", Y[None, :] ^ P_mat, False),  # p ⊕ ? = Y → ? = Y ⊕ p  # Y ⊕ p: если p = Y ⊕ q, то q = Y ⊕ p
            ):  # закрытие кортежа вариантов
                for idx_p in range(mat.shape[0]):  # по выражениям пула  # по векторам пула
                    k = mat[idx_p].tobytes()  # ключ искомого вектора  # ключ — байты строки матрицы mat[idx_p]
                    if k in pool_hash:  # есть в пуле  # ключ есть в пуле — второй операнд найден
                        e1 = pool_exprs[idx_p]  # первое выражение  # e1 — выражение пула с индексом idx_p
                        e2 = pool_hash[k]  # второе выражение  # e2 — выражение пула с вектором значений mat[idx_p]
                        cand = Expr(op_name, (e2, e1) if rev else (e1, e2))  # кандидат с нужным порядком  # кандидат op_name(e1, e2); при rev — op_name(e2, e1)
                        cb = cand.bits(ex_idx)  # его стоимость  # стоимость кандидата в битах (с учётом гамма-кодов индексов экспертов ex_idx)
                        if cb <= max_bits and verify_val(cand):  # влезает и проходит val  # укладывается в бюджет и проходит val
                            mitm_hits.append((cb, cand))  # добавить  # добавить кандидата в список попаданий
            nz_mask = np.all(P_mat != 0, axis=1)  # строки пула без нулей  # маска векторов пула без нулей: nz_mask[p] = все P_mat[p, n] ≠ 0
            if np.any(nz_mask):  # есть  # есть векторы без нулей
                P_nz = P_mat[nz_mask]  # их значения  # их значения P_nz: (P', N)
                nz_idxs = np.flatnonzero(nz_mask)  # их индексы  # их индексы в пуле
                div_ok = np.all((Y[None, :] % P_nz) == 0, axis=1)  # строки, на которые Y делится нацело  # признак делимости: div_ok[p] = Y mod P_nz[p, n] = 0 для всех n
                for loc_i in np.flatnonzero(div_ok):  # по ним  # по делящимся векторам
                    quot = Y // P_nz[loc_i]  # частное Y / p  # частное: quot = Y ÷ P_nz[loc_i] (деление нацело)
                    k = quot.tobytes()  # ключ  # ключ — байты частного
                    if k in pool_hash:  # частное есть в пуле  # частное есть в пуле — второй множитель найден
                        e1 = pool_exprs[int(nz_idxs[loc_i])]  # делитель  # e1 — делитель, выражение пула с индексом nz_idxs[loc_i]
                        e2 = pool_hash[k]  # частное  # e2 — выражение пула с вектором частного
                        cand = Expr("mul", (e1, e2))  # кандидат p · (Y / p)  # кандидат pow_int(основание из пула, показатель из пула)
                        cb = cand.bits(ex_idx)  # стоимость  # стоимость кандидата в битах
                        if cb <= max_bits and verify_val(cand):  # влезает и проходит  # укладывается в бюджет и проходит val
                            mitm_hits.append((cb, cand))  # добавить  # добавить кандидата
            if mitm_hits:  # есть кандидаты  # есть попадания
                mitm_hits.sort(key=lambda it: it[0])  # по стоимости  # отсортировать по возрастанию стоимости
                return mitm_hits[0][1], total_checked  # самый дешёвый  # вернуть самый дешёвый и число проверенных

    if len(pool_vals) >= 2:  # поиск степени из пула: не менее 2 выражений  # последняя попытка — степень: в пуле не меньше 2 векторов
        P_mat = np.stack(pool_vals[:500])  # первые 500 значений пула  # матрица первых 500 векторов пула: (P, N)
        ok_base = np.all(np.abs(P_mat) <= 32, axis=1)  # кандидаты в основание: |p| ≤ 32  # пригодны как основания: все |P_mat[p, n]| ≤ 32
        ok_exp = np.all((P_mat >= 0) & (P_mat <= 8), axis=1)  # кандидаты в показатель: 0 ≤ p ≤ 8  # пригодны как показатели: все 0 ≤ P_mat[p, n] ≤ 8
        if np.any(ok_base) and np.any(ok_exp):  # есть и те, и другие  # есть и основания, и показатели
            bases = P_mat[ok_base]  # основания  # значения оснований
            b_idxs = np.flatnonzero(ok_base)  # их индексы  # их индексы в пуле
            exps = P_mat[ok_exp]  # показатели  # значения показателей
            e_idxs = np.flatnonzero(ok_exp)  # их индексы  # их индексы в пуле
            pw = bases[:, None, :] ** exps[None, :, :]  # все степени основание^показатель  # все степени: pw[i, j, n] = bases[i, n] ^ exps[j, n]
            hits = np.all(pw == Y[None, None, :], axis=2)  # совпадения с Y  # попадания: hits[i, j] = pw[i, j, :] = Y
            if np.any(hits):  # есть  # есть попадание
                bi, ei = np.argwhere(hits)[0]  # первая пара  # первая пара (индекс основания, индекс показателя)
                cand = Expr(  # кандидат  # кандидат — выражение степени
                    "pow_int",  # операция возведения в целую степень
                    (pool_exprs[int(b_idxs[bi])], pool_exprs[int(e_idxs[ei])]),  # pow_int(основание, показатель)  # аргументы (основание из пула, показатель из пула)
                )  # закрытие аргументов
                if verify_val(cand):  # проходит val  # проходит val
                    return cand, total_checked  # вернуть  # вернуть выражение и число проверенных

    return None, total_checked  # не найдено  # ничего не найдено: None и число проверенных


# ==============================================================================
# 10. Эксперт, майнер теорем, структурная унификация графа вызовов и процессор
# ==============================================================================


@dataclass  # декоратор: конструктор, сравнение и представление генерируются по аннотациям полей
class Expert:
    """Эксперт: именованная программа с конечными областями аргументов.
    
            Поля: name — имя; family — семейство; program — программа (формула);
            domains — кортеж допустимых значений для каждого аргумента; valid_rule —
            дополнительное правило корректности входа ("nonzero_y" — второй аргумент не 0,
            "ordered_yz" — второй не больше третьего, иначе без правила); code_length —
            длина кода; adapter — адаптер общих весов или None; parents — имена родителей."""
    name: str  # имя эксперта
    family: str  # семейство эксперта
    program: Program  # программа — вычисляемая формула
    domains: tuple[tuple[int, ...], ...]  # области аргументов: кортеж кортежей, domains[i] — допустимые значения i-го аргумента
    valid_rule: str  # правило корректности входа
    code_length: int  # длина кода программы
    adapter: Optional[FactorizedAdapter] = None  # адаптер общих весов (по умолчанию нет)
    parents: tuple[str, ...] = ()  # имена родителей (по умолчанию пусто)

    def supports(self, xs: Sequence[int]) -> bool:
        """Проверить, допустим ли вход xs: True, если каждый аргумент в своей области и правило valid_rule выполнено.
        
                    xs — кортеж целых аргументов."""
        if len(xs) != len(self.domains):  # число аргументов не равно числу областей — вход не допустим
            return False  # вернуть «не поддерживается»
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):  # если хотя бы один аргумент вне своей области: x_i ∉ domains[i]
            return False  # вернуть «не поддерживается»
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:  # правило «второй аргумент не 0»: xs[1] = 0
            return False  # вернуть «не поддерживается»
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):  # правило «второй аргумент не больше третьего»: xs[1] > xs[2]
            return False  # вернуть «не поддерживается»
        return True  # вернуть «поддерживается»

    def to_dict(self) -> dict[str, Any]:
        """Словарь с полями эксперта для сериализации: имени, семейства, программы, областей, правила, длины кода, адаптера (словарь или None) и родителей.
        
                    Области и родители записываются списками."""
        return {  # словарь с полями эксперта
            "name": self.name,  # имя
            "family": self.family,  # семейство
            "program": self.program.to_dict(),  # программа как словарь
            "domains": [list(d) for d in self.domains],  # области — списки значений: [list(d) для каждого d]
            "valid_rule": self.valid_rule,  # правило корректности
            "code_length": self.code_length,  # длина кода
            "adapter": self.adapter.to_dict() if self.adapter else None,  # адаптер как словарь, если он есть, иначе None
            "parents": list(self.parents),  # родители — список
        }

    @staticmethod  # статический метод — создаётся без экземпляра
    def from_dict(d: dict[str, Any], core: "FactorizedSharedCore") -> "Expert":
        """Восстановить эксперта из словаря d (поля — как в to_dict).
        
                    d — словарь с ключами name, family, program, domains, valid_rule,
                    code_length и необязательными adapter, parents; core — общий блок весов,
                    нужный адаптеру для восстановления. Возвращает Expert."""
        return Expert(  # конструктор эксперта
            name=str(d["name"]),  # имя
            family=str(d["family"]),  # семейство
            program=Program.from_dict(d["program"]),  # программа из словаря
            domains=tuple(tuple(int(v) for v in dom) for dom in d["domains"]),  # области: каждый список d превращается в кортеж целых
            valid_rule=str(d["valid_rule"]),  # правило корректности
            code_length=int(d["code_length"]),  # длина кода
            adapter=(  # адаптер
                FactorizedAdapter.from_dict(d["adapter"], core)  # восстановить адаптер по словарю, если он был сохранён
                if d.get("adapter")  # если адаптер сохранён
                else None  # иначе None
            ),  # закрытие аргумента adapter
            parents=tuple(str(p) for p in d.get("parents", [])),  # родители — кортеж строк (по умолчанию пусто)
        )  # закрытие вызова конструктора


@dataclass  # декоратор: конструктор и представление генерируются по аннотациям полей
class Theorem:
    """Теорема о свойствах эксперта на конечной области: имя, текст утверждения, число проверенных точек и признак проверки.
    
            verified по умолчанию True."""
    name: str  # имя теоремы
    statement: str  # текст утверждения
    domain_points: int  # число проверенных точек области
    verified: bool = True  # признак проверки (по умолчанию истина)


class TheoremMiner:
    """Майнер теорем: полный перебор конечных областей всех экспертов модели (без усечения).
    
            Для каждого эксперта проверяются свойства: коммутативность, поглощение
            собой (e(x, x) = x), правый ноль (e(x, 0) = x) — для бинарных; замкнутость,
            инволютивность, идемпотентность, чётная и нечётная симметрия — для унарных.
            Свойство объявляется теоремой, только если оно выполнено на ВСЕХ допустимых точках."""
    """Найти теоремы модели: проверить свойства каждого эксперта на всей его конечной области.
    
                model — процессор с экспертами model.experts и списком уже известных
                теорем model.theorems. Возвращает список новых объектов Theorem; найденные
                теоремы добавляются в model.theorems."""

    def mine(self, model: "GrowingHagiProcessor") -> list[Theorem]:
        found: list[Theorem] = []  # список новых теорем
        known = {t.name for t in model.theorems}  # имена уже известных теорем
        env = model.experts  # среда вычисления — эксперты модели
        for e in model.experts.values():  # по экспертам модели
            if e.program.arity == 2:  # бинарный эксперт
                dx, dy = e.domains[0], e.domains[1]  # области первого и второго аргументов
                declared = [(x, y) for x in dx for y in dy if e.supports((x, y))]  # все допустимые пары: (x, y), x ∈ dx, y ∈ dy, e.supports((x, y)) = True
                swapped = [(x, y) for x, y in declared if e.supports((y, x))]  # пары, для которых допустим и обратный порядок (y, x)
                if swapped and len(swapped) == len(declared):  # коммутативность имеет смысл, только если допустимы все пары: |swapped| = |declared|
                    nm = f"commutativity({e.name})"  # имя теоремы коммутативности
                    if nm not in known:  # теоремы с таким именем ещё нет
                        try:  # попытка проверки (ошибка вычисления — теорема не объявляется)
                            if all(  # если для всех пар e(x, y) = e(y, x)
                                e.program.eval((x, y), env)  # вычислить e(x, y) в среде экспертов
                                == e.program.eval((y, x), env)  # равно e(y, x)
                                for x, y in swapped  # по допустимым парам
                            ):
                                th = Theorem(  # новая теорема
                                    nm,  # имя
                                    f"for all supported pairs: {e.name}(x,y)={e.name}(y,x)",
                                    len(swapped),  # число проверенных пар
                                )
                                model.theorems.append(th)  # добавить в теоремы модели
                                found.append(th)  # добавить в найденные
                                known.add(nm)  # запомнить имя
                        except Exception:  # ошибка вычисления
                            pass  # пропустить свойство
                self_pairs = [x for x in dx if x in dy and e.supports((x, x))]  # пары (x, x), где x есть в обеих областях и e.supports((x, x)) = True
                nm = f"self-absorption({e.name})"  # имя теоремы поглощения собой
                if self_pairs and len(self_pairs) == len(dx) and nm not in known:  # если x пробегает всю первую область и теоремы ещё нет
                    try:  # попытка проверки
                        if all(e.program.eval((x, x), env) == x for x in self_pairs):  # если для всех x: e(x, x) = x
                            th = Theorem(  # новая теорема
                                nm,  # имя
                                f"for all overlapping x: {e.name}(x,x)=x",
                                len(self_pairs),  # число проверенных точек
                            )
                            model.theorems.append(th)  # добавить в теоремы модели
                            found.append(th)  # добавить в найденные
                            known.add(nm)  # запомнить имя
                    except Exception:  # ошибка вычисления
                        pass  # пропустить свойство
                zero_pairs = [x for x in dx if 0 in dy and e.supports((x, 0))]  # значения x из первой области, для которых допустима пара (x, 0): 0 ∈ dy и e.supports((x, 0))
                nm = f"right-identity-zero({e.name})"  # имя теоремы о правом нуле
                if zero_pairs and len(zero_pairs) == len(dx) and nm not in known:  # если x пробегает всю первую область и теоремы ещё нет
                    try:  # попытка проверки
                        if all(e.program.eval((x, 0), env) == x for x in zero_pairs):  # если для всех x: e(x, 0) = x
                            th = Theorem(  # новая теорема
                                nm, f"for all x: {e.name}(x,0)=x", len(zero_pairs)
                            )
                            model.theorems.append(th)  # добавить в теоремы модели
                            found.append(th)  # добавить в найденные
                            known.add(nm)  # запомнить имя
                    except Exception:  # ошибка вычисления
                        pass  # пропустить свойство
            elif e.program.arity == 1:  # унарный эксперт
                dom = tuple(int(x) for x in e.domains[0])  # область определения — кортеж целых
                img = {}  # образ img: x → y
                closed = True  # признак замкнутости (образ лежит в области)
                try:  # попытка вычислить образ всей области
                    for x in dom:  # по точкам области
                        y = int(e.program.eval((x,), env))  # вычислить y = e(x) как целое
                        if not e.supports((y,)):  # y вне области — не замкнуто
                            closed = False  # прекратить перебор
                            break  # прекратить перебор области
                        img[x] = y  # запомнить y = e(x)
                except Exception:  # ошибка вычисления — не замкнуто
                    closed = False  # считать не замкнутым
                if closed:  # образ вычислен; проверять свойства e∘e и симметрию
                    for prop, fn in (  # по свойствам: имя и функция проверки без аргументов
                        (  # открытие кортежа
                            "involutivity",  # инволютивность
                            lambda: all(  # проверка: для всех x: e(e(x)) = x
                                e.program.eval((img[x],), env) == x for x in dom  # вычислить e(img[x]); равно x
                            ),  # равно x
                        ),
                        (  # идемпотентность
                            "idempotence",  # идемпотентность
                            lambda: all(  # проверка: для всех x: e(e(x)) = e(x)
                                e.program.eval((img[x],), env) == img[x] for x in dom  # вычислить e(img[x]); равно img[x]
                            ),
                        ),  # закрытие
                    ):  # закрытие кортежа свойств
                        nm = f"{prop}({e.name})"  # имя теоремы
                        if nm not in known:  # теоремы с таким именем ещё нет
                            try:  # попытка проверки
                                if fn():  # свойство выполнено на всей области
                                    th = Theorem(  # новая теорема
                                        nm, f"{prop} on certified domain", len(dom)  # текст утверждения
                                    )  # число проверенных точек
                                    model.theorems.append(th)  # добавить в теоремы модели
                                    found.append(th)  # добавить в найденные
                                    known.add(nm)  # запомнить имя
                            except Exception:  # ошибка вычисления
                                pass  # пропустить свойство
                if all((-x) in dom for x in dom):  # область симметрична: для каждого x число −x тоже в области
                    for sym_name, sgn in (("even-symmetry", 1), ("odd-symmetry", -1)):  # по видам симметрии: (название, знак) — чётная со знаком 1, нечётная со знаком −1
                        nm = f"{sym_name}({e.name})"  # имя теоремы симметрии
                        if nm not in known:  # теоремы с таким именем ещё нет
                            try:  # попытка проверки
                                if all(  # если для всех x: e(−x) = sgn·e(x)
                                    e.program.eval((-x,), env)  # вычислить e(−x); равно sgn·e(x)
                                    == sgn * e.program.eval((x,), env)  # равно sgn·e(x)
                                    for x in dom  # по точкам области
                                ):
                                    th = Theorem(  # новая теорема
                                        nm,  # имя
                                        f"{e.name}(-x)={'+' if sgn == 1 else '-'}{e.name}(x)",
                                        len(dom),  # число проверенных точек
                                    )
                                    model.theorems.append(th)  # добавить в теоремы модели
                                    found.append(th)  # добавить в найденные
                                    known.add(nm)  # запомнить имя
                            except Exception:  # ошибка вычисления
                                pass  # пропустить свойство
        return found  # вернуть все найденные теоремы


def _match_pattern(
    pattern: Expr, target: Expr, env_bind: dict[int, Expr]  # аргументы: образец pattern, цель target и словарь связываний env_bind
) -> bool:  # возвращает True, если деревья сопоставились
    """Структурная унификация: сопоставить дерево pattern с поддеревом target.
    
            pattern — выражение-образец, его узлы var(i) — «дыры» с номером i; target —
            проверяемое выражение; env_bind — накопленные связывания номер дыры → выражение
            (изменяется на месте). Возвращает True, если деревья совпадают: операция и
            значение узлов равны, аргументы сочетаются попарно, дыра связывается с любым
            поддеревом, повторная дыра требует равенства с прежним связыванием."""
    """Structural unification: match pattern AST (with var(i) holes) against target subtree."""
    if pattern.op == "var":  # узел-образец — дыра var(i)
        idx = int(pattern.value or 0)  # номер дыры idx (по умолчанию 0)
        if idx in env_bind:  # дыра уже связана
            return env_bind[idx] == target  # True, если прежнее связывание равно target
        env_bind[idx] = target  # связать дыру idx с target
        return True  # сопоставление удалось
    if (  # если узлы разные
        pattern.op != target.op  # разные операции
        or pattern.value != target.value  # или разные значения
        or len(pattern.args) != len(target.args)  # или разное число аргументов
    ):
        return False  # деревья не совпадают
    return all(  # все аргументы сочетаются попарно: _match_pattern(pattern.args[i], target.args[i]) для всех i
        _match_pattern(pa, ta, env_bind) for pa, ta in zip(pattern.args, target.args)  # генератор: сопоставить попарно аргументы pa из pattern.args и ta из target.args с общим env_bind
    )


def _dag_reaches(experts: dict[str, Expert], start: str, target: str) -> bool:
    """Проверить по графу вызовов, достижима ли задача target из задачи start: идёт ли вызов (прямо или через других экспертов) от start к target.
    
            experts — все эксперты по именам; start — имя начального эксперта; target — имя искомого.
            Обход в глубину по ребрам «эксперт → вызываемые им эксперты». Возвращает True, если target достигнут."""
    visited = set()  # множество посещённых имён
    stack = [start]  # стек обхода, начиная со start
    while stack:  # пока стек не пуст
        cur = stack.pop()  # снять имя с вершины стека
        if cur == target:  # достигли цели
            return True  # вернуть «достижима»
        if cur in visited or cur not in experts:  # уже посещено или такого эксперта нет
            continue  # пропустить
        visited.add(cur)  # пометить как посещённое
        ex = experts[cur]  # эксперт с этим именем
        if ex.program.expr is not None:  # у эксперта есть программа
            stack.extend(ex.program.expr.callees())  # положить в стек всех, кого он вызывает: список callees() его выражения
    return False  # стек исчерпан — цель не достигнута


class GrowingHagiProcessor:
    """Саморазвивающийся процессор с управлением: общий блок весов, библиотека экспертов, аудит значимости, маршрутизатор и память.
    
            Хранит экспертов по именам задач (self.experts), теоремы о них, журнал роста
            (growth_log) и сертификаты шагов (cert_log). Новая задача либо решается уже
            готовым экспертом, либо для неё синтезируется и проверяется программа;
            непроверенные изменения не принимаются (удержание). Цель — доля нерешённых
            задач: unsolved_objective уменьшается на 1 за каждый принятый эксперта,
            расходуются бюджет вычислений compute_budget и лимит риска risk_rem."""
    """Unified HAGI v9.0 Self-Developing Processor with Full Governance."""

    def __init__(  # конструктор (без docstring): каждая строка параметров пояснена комментарием
        self,  # сам экземпляр
        seed: int = 42,  # seed — зерно случайности общего блока весов
        node_budget: int = 8000,  # node_budget — предел числа проверенных выражений при синтезе
        delta_total: float = 0.05,  # delta_total — общий бюджет статистической значимости аудита
        compute_budget: float = 1000.0,  # compute_budget — бюджет вычислений
        risk_limit: float = 10.0,  # risk_limit — лимит риска
    ):
        self.core = FactorizedSharedCore(seed=seed, width=48)  # общий блок весов: 48 ветвей, зерно seed
        self.node_budget = int(node_budget)  # предел проверенных выражений как целое
        self.experts: dict[str, Expert] = {}  # библиотека экспертов: имя задачи → эксперт (изначально пуста)
        self.version = 0  # версия схемы (номер принятого изменения)
        self.total_commits = 0  # счётчик принятых изменений
        self.total_holds = 0  # счётчик удержаний
        self.growth_log: list[dict[str, Any]] = []  # журнал роста: список событий обучения
        self.cert_log: list[CertStepGen] = []  # журнал сертификатов шагов
        # признак полноты сохранённого состояния управления: старое сохранение, потерявшее
        # журнал управления, не должно читаться как успешное пустое доказательство
        # (поэтому после загрузки неполного состояния проверка сертификатов обязана провалиться)
        self.governance_state_complete = True  # управляющее состояние полностью сохранено (истина, пока не доказано обратное)
        self.theorems: list[Theorem] = []  # список теорем о свойствах экспертов
        self.ledger = AuditLedger(delta_total=delta_total)  # журнал аудита значимости с общим бюджетом delta_total
        self.router = HedgeRouter(len(FactorizedSharedCore.BRANCHES), eta=0.5)  # маршрутизатор ветвей с темпом обучения eta = 0.5
        self.ram = GDNRAM(n_slots=64, dim=8)  # ассоциативная память на 64 слота размерности 8
        self.compute_budget = float(compute_budget)  # бюджет вычислений
        self.risk_rem = float(risk_limit)  # остаток риска
        self.unsolved_objective = 100.0  # цель: доля нерешённых задач (изначально 100)
        self.evals = 0  # число вычислений программ
        self.candidates_generated = 0  # число сгенерированных кандидатов

    def _expert_index(self) -> dict[str, int]:
        """Построить индекс экспертов для гамма-кодирования: чем чаще эксперта вызывают, тем меньше его код.
        
                Число вызовов — сколько раз имя эксперта встречается среди вызываемых
                в программах всех экспертов. Сортировка по убыванию числа вызовов, затем по
                длине кода, затем по имени. Возвращает словарь имя → номер позиции."""
        """Zipf frequency-ordered index: most-called experts get smallest gamma codes."""
        counts = {nm: 0 for nm in self.experts}  # счётчики вызовов, изначально нулевые
        for ex in self.experts.values():  # по всем экспертам
            if ex.program.expr is not None:  # у эксперта есть программа
                for c in ex.program.expr.callees():  # по именам, которые он вызывает
                    if c in counts:  # вызываемый — известный эксперт
                        counts[c] += 1  # счётчик вызовов + 1
        ordered = sorted(  # отсортированные имена
            self.experts.keys(),  # ключи — имена экспертов
            key=lambda nm: (-counts[nm], self.experts[nm].code_length, nm),  # ключ сортировки: (−число вызовов, длина кода, имя)
        )  # закрытие вызова sorted
        return {nm: i for i, nm in enumerate(ordered)}  # имя → номер в отсортированном порядке

    def _snapshot_state_with_optimizer(self) -> dict[str, Any]:  # возвращает словарь снимка
        """Полный снимок изменяемого состояния: общего блока весов, маршрутизатора и памяти.
        
                Возвращает словарь с ключами "core", "router", "ram" (значения — их снимки)."""
        """Полный снимок состояния: общий блок весов, маршрутизатор и память."""
        return {  # словарь снимка
            "core": self.core.snapshot(),  # снимок общего блока весов
            "router": self.router.snapshot(),  # снимок маршрутизатора
            "ram": self.ram.snapshot(),  # снимок памяти
        }

    def _restore_state_with_optimizer(self, snap: dict[str, Any]) -> None:  # ничего не возвращает
        """Восстановить состояние из снимка, сделанного _snapshot_state_with_optimizer.
        
                snap — словарь с ключами "core", "router", "ram"."""
        self.core.restore(snap["core"])  # восстановить общий блок весов
        self.router.restore(snap["router"])  # восстановить маршрутизатор
        self.ram.restore(snap["ram"])  # восстановить память

    def _record_ram_fingerprint(  # Записать в память отпечаток таблицы значений задачи: вектор из 8 признаков по первым 32 точкам области.
        self, name: str, domain: list[tuple[int, ...]], ys: dict[tuple[int, ...], int]  # вторая строка параметров
    ) -> None:  # ничего не возвращает
        vals = [ys[r] for r in domain[:32]]  # значения первых 32 точек: vals = [ys[r] для r из domain[:32]]
        chunks = boundary_chunks(lambda v: v == 0 or abs(v) > 4, vals)  # разбиение значений на сегменты: элемент с v = 0 или |v| > 4 начинает новый сегмент
        heads = boundary_heads(lambda v: v == 0 or abs(v) > 4, vals)  # первые элементы сегментов
        sig = np.zeros(8, dtype=np.float64)  # нулевой вектор признаков из 8 компонент
        sig[0] = float(len(domain))  # sig[0] — размер области
        sig[1] = float(len(chunks))  # sig[1] — число сегментов
        sig[2] = float(len(heads))  # sig[2] — число голов
        for k in range(min(5, len(heads))):  # по первым min(5, число голов) головам
            sig[3 + k] = float(heads[k])  # sig[3 + k] = k-я голова
        slot = sum(ord(c) for c in name) % 64  # слот памяти: остаток от суммы кодов символов имени при делении на 64
        self.ram.store(slot, sig)  # записать вектор признаков sig в слот slot

    def learn(  # Выучить задачу spec: проверить готовые эксперты, затем синтезировать программу и принять её только при точном совпадении на train, val и всей области.
        self,  # сам экземпляр
        spec: TaskSpec,  # spec — задача: имя, арность, области, учитель
        seed: int = 0,  # seed — зерно разбиения на train и val
        train_fraction: float = 0.70,  # доля train в области
        poison_one_label: bool = False,  # признак «отравленной» метки: задача немедленно удерживается
        train_neural: bool = True,  # обучать ли нейросетевой адаптер
        deadline: Optional[float] = None,  # предельное время (monotonic) или None
    ) -> dict[str, Any]:  # возвращает словарь-событие
        if spec.name in self.experts:  # эксперт с именем задачи уже есть — переиспользовать
            return {  # словарь события
                "version": self.version,  # версия схемы
                "task": spec.name,  # имя задачи
                "action": "REUSE",  # действие «переиспользование»
                "reason": "already learned",  # причина «уже выучено»
                "code_length": self.experts[spec.name].code_length,  # длина кода готового эксперта
            }
        domain = spec.inputs()  # область — все допустимые входы задачи
        if not domain:  # область пуста
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)  # удержать задачу: пустая допустимая область
        # запомнить семейство задачи для языкового критерия синтеза (атрибут family или пустая строка)
        _CURRENT_FAMILY[0] = getattr(spec, "family", "")
        # кандидат на всю область: значения учителя по ВСЕЙ объявленной области;
        # каждый кандидат всё равно проходит полную проверку на train, val и всей области
        # при принятии
        _FULL_CAND[0] = None  # кандидата на всю область пока нет
        if _CURRENT_FAMILY[0] == "language" and spec.arity == 1:  # языковое семейство с одним аргументом
            try:  # попытка построить кандидата
                yy = np.array([int(spec.teacher(*r)) for r in domain],  # значения учителя yy = [int(spec.teacher(*r))] по всей области
                              dtype=np.int64)  # тип значений — целые 64-битные
                Xd = np.asarray(domain, dtype=np.int64)  # матрица входов Xd: (|domain|, 1)
                if set(np.unique(yy)) <= {0, 1}:  # значения только 0 и 1
                    _FULL_CAND[0] = _sparse_membership(Xd, yy)  # разреженная принадлежность как кандидат на всю область
                else:  # иначе
                    # модульно-аффинный или кусочный кандидат на всей области
                    # (для алфавитного типа выходов)
                    _FULL_CAND[0] = _mod_affine_expr(Xd, yy, 1)  # модульно-аффинный кандидат по арности 1
            except Exception:  # ошибка вычисления
                _FULL_CAND[0] = None  # кандидата нет

        opt_snap = self._snapshot_state_with_optimizer()  # снимок состояния до обучения

        rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
        perm = rng.permutation(len(domain))  # случайная перестановка номеров точек области
        if spec.family == "BoolVenn" or (  # булева задача или (все области из 0 и 1 и не более 16 точек)
            all(set(d) <= {0, 1} for d in spec.domains) and len(domain) <= 16  # условие: каждая область d ⊆ {0, 1} и |domain| ≤ 16
        ):  # закрытие условия
            n_train = max(1, len(domain) - 1)  # на train всё, кроме одной точки: n_train = max(1, |domain| − 1)
        else:  # иначе
            n_train = max(  # n_train = max(1, min(|domain| − 1, …)) — не больше |domain| − 1
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))  # округлённая доля: int(round(|domain|·train_fraction))
            )  # закрытие min
        train_x = [domain[int(i)] for i in perm[:n_train]]  # входы train — первые n_train точек перестановки
        val_x = [domain[int(i)] for i in perm[n_train:]]  # входы val — остальные точки перестановки
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}  # значения учителя на всей области: all_y[xs] = int(spec.teacher(*xs))
        train_y = [all_y[xs] for xs in train_x]  # значения train
        val_y = [all_y[xs] for xs in val_x]  # значения val

        self.ledger.allocate_stream(spec.name, len(train_x) + len(val_x))  # выделить задаче поток аудита на |train| + |val| точек

        if poison_one_label and train_y:  # отравленная метка и train непуст
            self.ledger.record_reject(spec.name)  # зафиксировать отклонение задачи в аудите
            self._restore_state_with_optimizer(opt_snap)  # откатить состояние к снимку
            return self._hold(  # удержать задачу
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)  # причина «внедрение отравленной метки», размеры train и val
            )  # закрытие вызова

        alpha_step = min(0.0005, self.ledger.remaining() * 0.5)  # шаг значимости: alpha_step = min(0.0005, 0.5·остаток бюджета значимости)
        ok_alpha, _radius = self.ledger.spend_anytime(spec.name, alpha_step, len(val_x))  # потратить alpha_step за |val| точек; ok_alpha — успех, _radius — радиус (не используется)
        if not ok_alpha:  # бюджет значимости исчерпан
            self._restore_state_with_optimizer(opt_snap)  # откатить состояние
            return self._hold(spec, "AUDIT_ALPHA_EXHAUSTED", len(train_x), len(val_x))  # удержать задачу: бюджет значимости исчерпан

        adapter: Optional[FactorizedAdapter] = None  # нейросетевой адаптер (по умолчанию нет)
        if train_neural and val_x and spec.family != "BoolVenn":  # нужно обучение нейросети, val непуст и задача не «BoolVenn»
            adapter = self.core.fit_residual(  # обучить адаптер остатка общего блока весов
                train_x, train_y, val_x, val_y, router=self.router  # на train и val с маршрутизатором
            )

        ex_idx = self._expert_index()  # индекс экспертов для стоимости в битах

        for nm, ex in sorted(self.experts.items()):  # по готовым экспертам в порядке имён
            if ex.program.arity != spec.arity:  # другая арность
                continue  # пропустить
            if not all(ex.supports(xs) for xs in domain):  # эксперт поддерживает не всю область задачи
                continue  # пропустить
            self.evals += len(train_x)  # число вычислений + |train|
            try:  # попытка проверить
                if any(  # если есть ошибка на train
                    ex.program.eval(xs, self.experts) != yy  # значение эксперта не равно значению учителя
                    for xs, yy in zip(train_x, train_y)  # по парам (вход, значение) train
                ):  # закрытие any
                    continue  # эксперт не подходит
                if any(  # если есть ошибка на val
                    ex.program.eval(xs, self.experts) != yy  # значение не равно yy
                    for xs, yy in zip(val_x, val_y)  # по парам val
                ):  # закрытие any
                    continue  # не подходит
                if any(ex.program.eval(xs, self.experts) != all_y[xs] for xs in domain):  # если есть ошибка хотя бы на одной точке всей области
                    continue  # не подходит
            except Exception:  # ошибка вычисления
                continue  # не подходит
            ref_bits = TOKEN_BITS + gamma_bits(ex_idx[nm] + 1)  # стоимость вызова: TOKEN_BITS + гамма-код номера (ex_idx[nm] + 1)
            ev = {  # словарь события
                "version": self.version,  # версия
                "task": spec.name,  # имя задачи
                "action": "REUSE",  # действие «переиспользование»
                "reason": f"verified-expert-coverage:{nm}",  # причина: покрытие проверенным экспертом nm
                "program": ex.program.pretty(),  # текст программы в читаемом виде
                "code_length": ref_bits,  # длина кода
                "train_examples": len(train_x),  # число точек train
                "validation_examples": len(val_x),  # число точек val
                "exact_accuracy": 1.0,  # точность 1.0
            }
            self.growth_log.append(ev)  # записать событие в журнал роста
            self.ledger.record_floor(spec.name, 1.0)  # зафиксировать нижнюю границу качества 1.0
            return ev  # вернуть событие

        def try_commit(  # Принять программу program как решение задачи, если она точна на train, val и всей области.
            program: Program, reason: str, parents: tuple[str, ...] = ()  # параметры
        ) -> Optional[dict[str, Any]]:  # возвращает событие или None
            try:  # попытка проверки
                self.evals += len(train_x)  # число вычислений + |train|
                if any(  # если есть ошибка на train
                    program.eval(xs, self.experts) != yy  # значение программы не равно yy
                    for xs, yy in zip(train_x, train_y)  # по парам train
                ):  # закрытие any
                    return None  # отклонить кандидата
                if val_x and any(  # val непуст и есть ошибка на val
                    program.eval(xs, self.experts) != yy for xs, yy in zip(val_x, val_y)  # значение не равно yy
                ):  # закрытие any
                    return None  # отклонить
                if any(program.eval(xs, self.experts) != all_y[xs] for xs in domain):  # есть ошибка хотя бы на одной точке всей области
                    return None  # отклонить
            except Exception:  # ошибка вычисления
                return None  # отклонить
            bits = program.bits(ex_idx)  # длина кода программы в битах (с учётом индекса экспертов)
            expert = Expert(  # новый эксперт
                spec.name,  # имя
                spec.family,  # семейство
                program,  # программа
                spec.domains,  # области аргументов
                spec.valid_rule,  # правило корректности
                bits,  # длина кода
                adapter,  # адаптер общих весов
                parents,  # родители
            )
            self.experts[spec.name] = expert  # записать эксперта в библиотеку
            self.version += 1  # версия + 1
            self.total_commits += 1  # принятых изменений + 1
            self._record_ram_fingerprint(spec.name, domain, all_y)  # записать отпечаток таблицы значений в память
            drop = 1.0  # снижение цели за принятого эксперта: drop = 1.0
            # точное уменьшение цели в сертификате: obj_after обязан равняться obj_before − drop
            # БЕЗ отсечки в нуле — нижняя граница 0.0 ломает это соотношение, как только число
            # принятых изменений превосходит начальный номинальный бюджет (obj0 = 100), и тогда
            # проверка сертификатов проваливалась бы навсегда. Цель может законно стать
            # отрицательной (задачи «переплачены»).
            obj_after = self.unsolved_objective - drop  # цель после шага: obj_after = unsolved_objective − drop
            b_spend = min(self.compute_budget, 0.5)  # расход бюджета вычислений: b_spend = min(остаток compute_budget, 0.5)
            r_spend = min(self.risk_rem, 0.01)  # расход риска: r_spend = min(остаток risk_rem, 0.01)
            self.compute_budget -= b_spend  # остаток бюджета вычислений − b_spend
            self.risk_rem -= r_spend  # остаток риска − r_spend
            floors_ok = self.ledger.record_floor(spec.name, 1.0)  # зафиксировать нижнюю границу качества 1.0; floors_ok — все границы соблюдены
            cert = CertStepGen(  # сертификат шага
                version=self.version,  # версия
                task=spec.name,  # имя задачи
                action="COMMIT",  # действие «принятие»
                obj_before=self.unsolved_objective,  # цель до
                obj_after=obj_after,  # цель после
                drop=drop,  # снижение
                budget_rem=self.compute_budget,  # остаток бюджета вычислений
                budget_spend=b_spend,  # расход бюджета вычислений
                risk_rem=self.risk_rem,  # остаток риска
                risk_spend=r_spend,  # расход риска
                delta_rem=self.ledger.remaining(),  # остаток значимости
                delta_spend=alpha_step,  # расход значимости
                floors_kept=floors_ok,  # нижние границы соблюдены
            )
            self.unsolved_objective = obj_after  # обновить цель
            self.cert_log.append(cert)  # записать сертификат в журнал
            ev = {  # словарь события
                "version": self.version,  # версия
                "task": spec.name,  # имя задачи
                "action": "COMMIT",  # действие «принятие»
                "reason": reason,  # причина
                "program": program.pretty(),  # текст программы
                "code_length": bits,  # длина кода
                "train_examples": len(train_x),  # число точек train
                "validation_examples": len(val_x),  # число точек val
                "exact_accuracy": 1.0,  # точность 1.0
            }
            self.growth_log.append(ev)  # записать событие в журнал роста
            return ev  # вернуть событие

        if spec.family == "BoolVenn":  # булевы задачи: перебор всех 16 булевых функций двух аргументов
            for m in range(16):  # по маске m = 0..15
                ev = try_commit(Program(2, venn_mask=m), "boolvenn-exact")  # программа с функцией Венна по маске m
                if ev is not None:  # принято
                    return ev  # вернуть событие

        if all(set(d) <= {0, 1} for d in spec.domains) and 1 <= spec.arity <= 6:  # все области из 0 и 1 и арность от 1 до 6
            if set(train_y) <= {0, 1}:  # значения train только 0 и 1
                full_inputs = list(itertools.product((0, 1), repeat=spec.arity))  # все 2^arity наборов из 0 и 1
                train_map = dict(zip(train_x, train_y))  # словарь вход → значение на train
                missing = [r for r in full_inputs if r not in train_map]  # наборы, отсутствующие в train
                if len(missing) <= 1:  # не хватает не более одного набора
                    anf_cands = []  # кандидаты (оценка, программа)
                    for fill in (0,) if not missing else (0, 1):  # заполнение недостающего набора: только 0, если ничего не хватает, иначе 0 и 1
                        tt = [train_map.get(r, fill) for r in full_inputs]  # таблица истинности: tt = train_map[r], для недостающего — fill
                        coeffs = anf_transform(tt)  # коэффициенты АНФ (полинома Жегалкина)
                        active_s = [s for s, c in enumerate(coeffs) if c]  # множества с ненулевым коэффициентом
                        max_deg = max((bin(s).count("1") for s in active_s), default=0)  # степень многочлена — максимум числа единиц в двоичной записи множества
                        cand_e = anf_to_expr(coeffs, spec.arity)  # выражение по коэффициентам АНФ
                        cand_p = Program(spec.arity, cand_e)  # программа-кандидат
                        score = (max_deg, len(active_s), cand_p.bits(ex_idx))  # оценка: (степень, число слагаемых, длина кода в битах) — меньше лучше
                        anf_cands.append((score, cand_p))  # добавить кандидата
                    anf_cands.sort(key=lambda it: it[0])  # отсортировать по оценке
                    for _, cand_p in anf_cands:  # по кандидатам в порядке оценки
                        ev = try_commit(cand_p, "anf-zhegalkin-butterfly+full-support")  # принять полином Жегалкина
                        if ev is not None:  # принято
                            return ev  # вернуть событие

        for p in candidate_programs(spec.arity):  # по программам-шаблонам для арности
            ev = try_commit(p, "direct-template+full-support")  # принять шаблон
            if ev is not None:  # принято
                return ev  # вернуть событие

        if 1 <= spec.arity <= 3 and not spec.family.startswith("Bool"):  # арность от 1 до 3 и небулево семейство
            if spec.arity == 1:  # один аргумент
                coeffs = _exact_poly_fit([r[0] for r in train_x], train_y, max_degree=5)  # точный рациональный многочлен степени не выше 5: y = Σ c_i·x^i
                if coeffs is not None:  # многочлен найден
                    ev = try_commit(  # принять
                        Program(1, Expr("poly", (Expr.var(0),), coeffs)),  # программа poly(x_0, коэффициенты)
                        "exact-rational-poly+full-support",  # причина
                    )  # закрытие вызова
                    if ev is not None:  # принято
                        return ev  # вернуть событие
            else:  # иначе — много аргументов
                terms = _exact_multivariate_poly_fit(  # точный многочлен от многих переменных степени не выше 4
                    train_x, train_y, spec.arity, max_degree=4  # арность и предел степени
                )  # закрытие вызова
                if terms is not None:  # найден
                    ev = try_commit(  # принять
                        Program(spec.arity, Expr("poly_nd", (), terms)),  # программа poly_nd с перечнем термов
                        "exact-multivar-poly+full-support",  # причина
                    )  # закрытие вызова
                    if ev is not None:  # принято
                        return ev  # вернуть событие

        pw = exact_piecewise_poly_fit(train_x, train_y)  # точная кусочно-полиномиальная подгонка
        if pw is not None:  # найдена
            ev = try_commit(  # принять
                Program(spec.arity, pw), "exact-piecewise-poly+full-support"  # программа с найденным выражением
            )  # закрытие вызова
            if ev is not None:  # принято
                return ev  # вернуть событие

        lib_list = [(ex_idx[nm], nm, self.experts[nm]) for nm in self.experts]  # библиотека для синтеза: (номер индекса, имя, эксперт)
        hit_expr, n_chk = synthesize_batched_mdl(  # перебор методом минимальной длины описания
            spec.arity,  # арность
            train_x,  # входы train
            train_y,  # значения train
            val_x,  # входы val
            val_y,  # значения val
            library_experts=lib_list,  # эксперты библиотеки
            max_evals=self.node_budget,  # предел проверенных выражений — бюджет узлов
            max_bits=160,  # бюджет бит на выражение 160
            deadline=deadline,  # предельное время
        )
        self.candidates_generated += n_chk  # счётчик сгенерированных кандидатов + число проверенных
        if hit_expr is not None:  # выражение найдено
            ev = try_commit(  # принять
                Program(spec.arity, hit_expr), "batched-mdl-mitm-dag+full-support"  # программа из найденного выражения
            )  # закрытие вызова
            if ev is not None:  # принято
                return ev  # вернуть событие

        self._restore_state_with_optimizer(opt_snap)  # ничего не принято: откатить состояние к снимку
        self.ledger.record_reject(spec.name)  # зафиксировать отклонение в аудите
        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))  # удержать задачу: нет проверенной программы

    def _hold(  # Удержать задачу: записать сертификат и событие без изменения цели и бюджетов.
        self, spec: TaskSpec, reason: str, ntrain: int, nval: int  # параметры
    ) -> dict[str, Any]:  # возвращает словарь-событие
        self.total_holds += 1  # счётчик удержаний + 1
        cert = CertStepGen(  # сертификат шага
            version=self.version,  # версия
            task=spec.name,  # имя задачи
            action="HOLD",  # действие «удержание»
            obj_before=self.unsolved_objective,  # цель до
            obj_after=self.unsolved_objective,  # цель после — та же
            drop=0.0,  # снижение 0
            budget_rem=self.compute_budget,  # остаток бюджета вычислений
            budget_spend=0.0,  # расход 0
            risk_rem=self.risk_rem,  # остаток риска
            risk_spend=0.0,  # расход 0
            delta_rem=self.ledger.remaining(),  # остаток значимости
            delta_spend=0.0,  # расход 0
            floors_kept=True,  # нижние границы соблюдены
        )
        self.cert_log.append(cert)  # записать сертификат в журнал
        ev = {  # словарь события
            "version": self.version,  # версия
            "task": spec.name,  # имя задачи
            "action": "HOLD",  # действие «удержание»
            "reason": reason,  # причина
            "program": "",  # программы нет
            "code_length": 0,  # длина кода 0
            "train_examples": ntrain,  # число точек train
            "validation_examples": nval,  # число точек val
            "exact_accuracy": 0.0,  # точность 0.0
        }
        self.growth_log.append(ev)  # записать событие в журнал роста
        return ev  # вернуть событие

    def compress_library_mdl(self) -> dict[str, Any]:
        """Global MDL Call-DAG Refactoring with structural pattern unification & Zipf re-indexing."""
        ex_idx = self._expert_index()  # индекс имён экспертов для кодирования программ битами
        for ex in self.experts.values():  # по всем экспертам
            ex.code_length = ex.program.bits(ex_idx)  # длина кода эксперта = число бит его программы: bits = Σ длина(оператор)
        before_bits = sum(e.code_length for e in self.experts.values())  # суммарная длина до сжатия: Σ_e code_length(e)
        refactored = 0  # число переформированных экспертов

        for target_name, target_ex in list(self.experts.items()):  # по экспертам библиотеки (копия списка, чтобы менять словарь во время обхода)
            if target_ex.program.expr is None:  # у цели нет программы — нечего сжимать
                continue  # у эксперта нет программы
            cur_expr = target_ex.program.expr  # текущее выражение цели
            subs = [  # поддеревья-кандидаты на замену вызовом
                s  # элемент списка
                for s in cur_expr.subtrees()  # по всем поддеревьям cur_expr
                if s.op not in ("var", "const", "call") and s.cost >= 2  # не лист и не вызов, стоимость не меньше 2
            ]  # закрытие списка
            if not subs:  # заменять нечего
                continue  # следующий эксперт
            domain = [  # область задачи — все допустимые входы
                xs  # набор аргументов
                for xs in itertools.product(*target_ex.domains)  # декартово произведение областей эксперта
                if target_ex.supports(xs)  # только входы, которые эксперт поддерживает
            ]  # закрытие списка
            for sub in subs:  # по поддеревьям-кандидатам
                for lib_name, lib_ex in list(self.experts.items()):  # по экспертам библиотеки (копия списка)
                    if lib_name == target_name or lib_ex.program.expr is None:  # сам эксперт или эксперт без программы
                        continue  # пропустить
                    if _dag_reaches(self.experts, lib_name, target_name):  # замена создала бы цикл: lib_ex вызывает (прямо или косвенно) целевого эксперта
                        continue  # пропустить
                    bindings: dict[int, Expr] = {}  # связывания дыр образца: номер → поддерево
                    if not _match_pattern(lib_ex.program.expr, sub, bindings):  # программа библиотечного эксперта не сопоставляется с поддеревом
                        continue  # пропустить
                    if any(i not in bindings for i in range(lib_ex.program.arity)):  # какая-то дыра не связалась
                        continue  # пропустить
                    call_args = tuple(bindings[i] for i in range(lib_ex.program.arity))  # аргументы вызова — поддеревья, связанные с дырами 0..arity−1
                    call_e = Expr("call", call_args, lib_name)  # вызов lib_name с найденными аргументами
                    if call_e.bits(ex_idx) >= sub.bits(ex_idx):  # вызов не дешевле поддерева
                        continue  # пропустить
                    cand_expr = cur_expr.replace_subtree(sub, call_e)  # заменить поддерево sub вызовом call_e
                    cand_prog = Program(target_ex.program.arity, cand_expr)  # программа-кандидат той же арности
                    new_bits = cand_prog.bits(ex_idx)  # её длина в битах
                    if new_bits >= target_ex.code_length:  # не короче прежней
                        continue  # пропустить
                    try:  # попытка проверки
                        if all(  # если на всей области
                            cand_prog.eval(xs, self.experts)  # значение программы-кандидата
                            == target_ex.program.eval(xs, self.experts)  # совпадает со значением прежней программы
                            for xs in domain  # по всем входам области
                        ):  # закрытие all
                            target_ex.program = cand_prog  # принять новую программу
                            target_ex.code_length = new_bits  # запомнить новую длину кода
                            cur_expr = cand_expr  # текущее выражение — новое
                            refactored += 1  # счётчик преобразований + 1
                    except Exception:  # ошибка вычисления
                        pass  # оставить как было

        ex_idx = self._expert_index()  # пересчитать индекс после преобразований
        for ex in self.experts.values():  # по всем экспертам
            ex.code_length = ex.program.bits(ex_idx)  # пересчитать длину кода по новому индексу
        after_bits = sum(e.code_length for e in self.experts.values())  # суммарная длина после
        return {  # итог словарём
            "before_bits": before_bits,  # бит до
            "after_bits": after_bits,  # бит после
            "refactored": refactored,  # число преобразований
        }

    def predict(self, name: str, *xs: int, mode: str = "exact") -> dict[str, Any]:
        """Предсказать значение задачи name на входе xs.
        
                mode "exact" — точное значение программы эксперта; mode "neural" — приближение
                нейросетевым адаптером (или точной тернарной головой функции Венна, если адаптера
                нет). Если эксперта нет или вход вне его области — «ABSTAIN» (отказ). Возвращает
                словарь со статусом status, значением value, источником source и версией модели."""
        ex = self.experts.get(name)  # эксперт по имени
        if ex is None:  # эксперта нет
            return {"status": "ABSTAIN", "reason": "unknown operation"}  # отказ: неизвестная операция
        if not ex.supports(xs):  # вход вне области эксперта
            return {"status": "ABSTAIN", "reason": "outside certified finite domain"}  # отказ: вне проверенной конечной области
        if mode == "exact":  # точный режим
            return {  # словарь ответа
                "status": "OK",  # статус «ок»
                "value": ex.program.eval(xs, self.experts),  # значение программы
                "source": "verified-symbolic-dag",  # источник — проверенный символьный граф вызовов
                "model_version": self.version,  # версия модели
            }
        if mode == "neural":  # нейросетевой режим
            if ex.adapter is None:  # адаптера нет
                if ex.program.venn_mask is not None:  # но у программы есть маска функции Венна
                    return {  # словарь ответа
                        "status": "APPROXIMATE",  # статус «приближение»
                        "value": float(venn_head_eval(ex.program.venn_mask, xs)),  # точное значение тернарной головы функции Венна как число с плавающей точкой
                        "source": "venn-head-exact-ternary",  # источник — точная тернарная голова
                        "model_version": self.version,  # версия модели
                    }
                return {"status": "ABSTAIN", "reason": "neural adapter unavailable"}  # отказ: нейросетевой адаптер недоступен
            return {  # словарь ответа
                "status": "APPROXIMATE",  # статус «приближение»
                "value": ex.adapter.predict(self.core, xs),  # предсказание адаптера на общем блоке весов
                "source": "factorized-core-bitalloc-rvq",  # источник — общий блок с распределением бит
                "model_version": self.version,  # версия модели
            }
        raise ValueError("mode must be exact or neural")  # ошибка: режим обязан быть exact или neural

    def verify_all(self, specs: Sequence[TaskSpec]) -> dict[str, Any]:
        """Проверить все задачи из specs по их экспертам и вернуть итоги.
        
                Покрытой считается задача, для которой есть принятый эксперт или записанное
                в журнале роста переиспользование чужого проверенного эксперта. Для каждой
                покрытой задачи считается число точек, где программа совпадает с учителем;
                точность accuracy = correct / max(1, total). Непокрытые задачи перечислены
                в missing_tasks."""
        total = correct = 0  # всего проверено и совпадений
        # карта переиспользования: задача, решённая чужим проверенным экспертом, не имеет
        # собственной записи в self.experts; без этой карты число покрытых задач
        # занижалось бы, а вместе с ним и проверка нижних границ качества
        # Значения расходились на одну задачу.
        coverage: dict[str, str] = {}  # задача → имя эксперта, который её покрыл
        for ev in self.growth_log:  # по событиям журнала роста
            if ev.get("action") == "REUSE" and "verified-expert-coverage:" in ev.get("reason", ""):  # событие «переиспользование» с причиной «покрытие проверенным экспертом»
                coverage[ev["task"]] = ev["reason"].split(":", 1)[1]  # имя покрывшего эксперта — часть причины после первого двоеточия
        # непокрытая задача обязана снижать точность, а не пропускаться: иначе набор
        # задач {A, B} при экспертах {A, посторонний} давал бы точность 1.0.
        # Покрытая задача — принятый эксперт или проверенное переиспользование
        # Покрытая задача — принятый эксперт или проверенное переиспользование.
        missing: list[str] = []  # список непокрытых задач
        for sp in specs:  # по задачам
            ex = self.experts.get(sp.name) or self.experts.get(coverage.get(sp.name, ""))  # эксперт задачи или эксперт, покрывший её через переиспользование
            if ex is None:  # эксперта нет
                missing.append(sp.name)  # задача непокрыта
                continue  # следующая задача
            rows = sp.inputs()  # все входы задачи
            ok = sum(  # число совпадений с учителем
                int(ex.program.eval(xs, self.experts) == int(sp.teacher(*xs)))  # 1, если программа дала ответ учителя, иначе 0
                for xs in rows  # по входам
            )
            total += len(rows)  # всего + число входов
            correct += ok  # совпадений + ok
        return {  # итог словарём
            "checked": total,  # проверено точек
            "correct": correct,  # совпадений
            "accuracy": correct / max(1, total),  # точность correct / max(1, total)
            "committed": len(self.experts),  # число принятых экспертов
            "expected_tasks": len(specs),  # число задач в наборе
            "covered_tasks": len(specs) - len(missing),  # число покрытых задач
            "missing_tasks": missing,  # непокрытые задачи
            "suite_complete": not missing,  # набор покрыт целиком, если непокрытых нет
        }

    def verify_certificates(self) -> dict[str, Any]:
        """4328	Проверить сертификаты и журнал аудита: согласованность цели, неизменность состояния при удержаниях, корректность принятий, непересечение потоков аудита и полнота истории.
        
                Возвращает словарь признаков: telescope_ok — цель не превышает
                obj0 − Σ drop (сдвиг 1e−9); holds_ok — каждое удержание ничего не меняло;
                commits_ok — каждое принятие снизило цель и не ушло в минус по бюджетам;
                ledger_ok — потоки аудита непересекающиеся и чистые; state_complete — состояние
                управления загружено полностью; history_ok — число записей равно счётчикам;
                all_ok — все признаки вместе."""
        obj0 = self.cert_log[0].obj_before if self.cert_log else self.unsolved_objective  # начальная цель obj0 — цель до первого шага; если журнала нет, текущая цель
        total_drops = sum(c.drop for c in self.cert_log)  # сумма всех снижений Σ drop
        telescope_ok = self.unsolved_objective <= obj0 - total_drops + 1e-9  # цель не превосходит obj0 − Σ drop (допуск 1e−9)
        holds_ok = all(  # все удержания корректны
            c.drop == 0.0 and c.obj_after == c.obj_before and c.budget_spend == 0.0  # снижение 0, цель не изменилась и расход бюджета 0
            for c in self.cert_log  # по шагам журнала
            if c.action == "HOLD"  # только удержания
        )
        commits_ok = all(  # все принятия корректны
            c.drop > 0.0  # снижение больше 0
            and c.budget_rem >= -1e-12  # остаток бюджета вычислений не отрицателен (допуск 1e−12)
            and c.risk_rem >= -1e-12  # остаток риска не отрицателен
            and c.floors_kept  # нижние границы соблюдены
            for c in self.cert_log  # по шагам журнала
            if c.action == "COMMIT"  # только принятия
        )
        ledger_ok = self.ledger.verify_disjoint_and_clean()  # журнал аудита: потоки непересекающиеся и чистые
        # полнота истории: очищенный журнал сертификатов не должен читаться
        # как успешное доказательство
        state_complete = bool(getattr(self, "governance_state_complete", False))  # признак полноты состояния (по умолчанию False)
        logged_commits = sum(c.action == "COMMIT" for c in self.cert_log)  # число записей о принятиях
        logged_holds = sum(c.action == "HOLD" for c in self.cert_log)  # число записей об удержаниях
        history_ok = (  # история согласована
            logged_commits == self.total_commits  # принятий в журнале столько же, сколько счётчик
            and logged_holds == self.total_holds  # удержаний столько же, сколько счётчик
            and len(self.cert_log) == self.total_commits + self.total_holds  # и длина журнала равна сумме счётчиков
        )
        all_ok = (telescope_ok and holds_ok and commits_ok and ledger_ok  # все признаки вместе
                  and state_complete and history_ok)  # (продолжение условия)
        return {  # итог словарём
            "telescope_ok": telescope_ok,
            "holds_ok": holds_ok,
            "commits_ok": commits_ok,
            "ledger_ok": ledger_ok,
            "state_complete": state_complete,
            "history_ok": history_ok,
            "all_ok": all_ok,
        }

    def total_neural_bytes(self) -> int:
        """Суммарное число байт нейросетевой части: общий блок весов + голова функции Венна + адаптеры.
        
                Голова функции Венна занимает 13 байт, если есть хоть один эксперт семейства «BoolVenn».
                Возвращает целое число байт."""
        venn_bytes = (  # байты головы функции Венна
            13 if any(e.family == "BoolVenn" for e in self.experts.values()) else 0  # 13, если есть эксперт «BoolVenn», иначе 0
        )  # закрытие скобки
        adapter_bytes = sum(  # байты всех адаптеров
            e.adapter.compressed_bytes  # размер сжатого адаптера
            for e in self.experts.values()  # по всем экспертам
            if e.adapter is not None  # у которых есть адаптер
        )
        return self.core.core_bytes + venn_bytes + adapter_bytes  # общий блок + голова + адаптеры

    def neural_quality_summary(self) -> dict[str, float]:
        """Сводка качества нейросетевых адаптеров: число адаптеров, средняя точность и средняя ошибка на val, суммарный размер.
        
                Средние считаются только по экспертам с адаптером; если адаптеров нет — 0.0.
                mean_val_mse — средний квадрат ошибки; total_neural_bytes — байты из total_neural_bytes()."""
        accs = [  # точности на val
            e.adapter.val_exact_acc_quant  # точность адаптера в сжатом (квантованном) виде
            for e in self.experts.values()  # по экспертам
            if e.adapter is not None  # с адаптером
        ]
        mses = [  # квадраты ошибки на val
            e.adapter.val_mse_quant  # средний квадрат ошибки адаптера в сжатом виде
            for e in self.experts.values()  # по экспертам
            if e.adapter is not None  # с адаптером
        ]
        return {  # итог словарём
            "num_adapters": len(accs),  # число адаптеров
            "mean_val_exact_acc": float(np.mean(accs)) if accs else 0.0,  # средняя точность: mean(accs)
            "mean_val_mse": float(np.mean(mses)) if mses else 0.0,  # средний квадрат ошибки: mean(mses)
            "total_neural_bytes": self.total_neural_bytes(),  # суммарное число байт нейросетевой части
        }

    def save(self, path: str | Path) -> None:
        """Сохранить модель в JSON-файл path: экспертов и состояние управления.
        
                Символических экспертов хватает для предсказания, но не для восстановления
                истории сертификатов и журнала аудита, поэтому format_version = 2 сохраняет
                и её: сертификаты, аудит, бюджеты, цель, веса общего блока, маршрутизатор
                и память. Файл создаётся с нужными каталогами; NaN запрещены."""
        """Сохранить модель в JSON-файл path: экспертов и состояние управления.
        
                Символических экспертов хватает для предсказания, но не для восстановления
                истории сертификатов и журнала аудита, поэтому format_version = 2 сохраняет
                и её: сертификаты, аудит, бюджеты, цель, веса общего блока, маршрутизатор
                и память. Файл создаётся с нужными каталогами; NaN запрещены."""
        gpm = self.core.gpm.snapshot_optimizer()  # снимок оптимизатора общего блока весов
        router_weights, router_loss = self.router.snapshot()  # веса маршрутизатора и его накопленные потери
        ledger = self.ledger  # журнал аудита
        governance_state = {  # состояние управления
            "schema": 1,  # версия схемы состояния
            "cert_log": [vars(c) for c in self.cert_log],  # сертификаты как словари полей
            "ledger": {  # журнал аудита
                "delta_total": ledger.delta_total,  # общий бюджет значимости
                "spent": dict(ledger.spent),  # расход по задачам
                "streams": {k: vars(v) for k, v in sorted(ledger.streams.items())},  # потоки как словари полей (по именам)
                "floors": dict(ledger.floors),  # нижние границы качества по задачам
                "rejected_tasks": sorted(ledger.rejected_tasks),  # список отклонённых задач
                "final_test_lo": ledger.final_test_lo,  # нижняя граница итоговой проверки
                "final_test_hi": ledger.final_test_hi,  # верхняя граница итоговой проверки
                "final_test_touched": ledger.final_test_touched,  # признак использования итоговой проверки
                "next_cursor_base": ledger.next_cursor_base,  # начало счётчика выборок
            },
            "compute_budget": self.compute_budget,  # бюджет вычислений
            "risk_rem": self.risk_rem,  # остаток риска
            "unsolved_objective": self.unsolved_objective,  # цель
            "evals": self.evals,  # число вычислений
            "candidates_generated": self.candidates_generated,  # число сгенерированных кандидатов
            "core_gpm": {  # общий блок весов
                "W": gpm["W"].tolist(),  # матрица W
                "U_basis": gpm["U_basis"].tolist(),  # базис U_basis
                "ef_v": gpm["ef_v"].tolist(),  # вектор ef_v
                "core_bytes": int(self.core.core_bytes),  # размер общего блока в байтах
            },
            "router": {  # маршрутизатор
                "K": int(self.router.K),  # число ветвей K
                "eta": float(self.router.eta),  # темп обучения eta
                "weights": router_weights.tolist(),  # веса ветвей
                "cum_loss": router_loss.tolist(),  # накопленные потери
            },
            "ram": self.ram.snapshot().tolist(),  # снимок памяти
        }
        data = {  # данные файла
            "format": "HAGI-V9",  # формат
            "format_version": 2,  # версия формата 2
            "version": V9_VERSION,  # версия схемы
            "core_seed": self.core.seed,  # зерно общего блока
            "core_width": self.core.width,  # ширина общего блока
            "model_version": self.version,  # версия модели
            "node_budget": self.node_budget,  # бюджет узлов
            "total_commits": self.total_commits,  # принято изменений
            "total_holds": self.total_holds,  # удержаний
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},  # эксперты как словари (по именам)
            "theorems": [vars(t) for t in self.theorems],  # теоремы как словари полей
            "growth_log": self.growth_log,  # журнал роста
            "governance_state": governance_state,  # состояние управления
            "governance_state_complete": bool(self.governance_state_complete),  # признак полноты состояния управления
        }
        p = Path(path)  # путь к файлу
        p.parent.mkdir(parents=True, exist_ok=True)  # создать родительский каталог, если его нет
        p.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False),  # записать JSON с отступом 2, сортировкой ключей и без NaN
                     encoding="utf-8")  # кодировка UTF-8

    @classmethod  # метод класса — создаёт экземпляр
    def load(cls, path: str | Path) -> "GrowingHagiProcessor":
        """Загрузить модель из JSON-файла path, сохранённого save().n
                Формат обязан быть «HAGI-V9». Восстанавливаются эксперты, теоремы, журнал роста
                и — только если состояние управления сохранено полностью и его схема та же —
                сертификаты, журнал аудита, маршрутизатор, память, веса общего блока, бюджеты
                и цель (иначе признак полноты сбрасывается в False). Формы и конечность
                чисел проверяются, при расхождении — ошибка ValueError. Возвращает процессор."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))  # прочитать и разобрать JSON
        if d.get("format") != "HAGI-V9":  # формат не тот
            raise ValueError("not a HAGI-V9 model file")  # ошибка: это не файл модели HAGI-V9
        m = cls(  # новый процессор
            seed=int(d.get("core_seed", 42)),  # зерно по умолчанию 42
            node_budget=int(d.get("node_budget", 8000)),  # бюджет узлов по умолчанию 8000
        )
        m.version = int(d.get("model_version", 0))  # версия модели
        m.total_commits = int(d.get("total_commits", 0))  # принято изменений
        m.total_holds = int(d.get("total_holds", 0))  # удержаний
        m.experts = {  # эксперты по именам
            k: Expert.from_dict(v, m.core) for k, v in d.get("experts", {}).items()  # каждый — из словаря, адаптеры привязываются к общему блоку m.core
        }  # закрытие словаря
        m.theorems = [Theorem(**t) for t in d.get("theorems", [])]  # теоремы из словарей полей
        m.growth_log = d.get("growth_log", [])  # журнал роста
        gs = d.get("governance_state")  # состояние управления (может отсутствовать)
        if (d.get("governance_state_complete") is not True  # если признак полноты не True
                or not isinstance(gs, dict) or gs.get("schema") != 1):  # или состояние не словарь, или версия схемы не 1
            # загрузка для предсказания совместима со старыми файлами, но отсутствие
            # истории аудита — не успешное пустое доказательство
            m.governance_state_complete = False  # сбросить признак полноты
            return m  # вернуть процессор без состояния управления
        ledger_data = gs.get("ledger")  # данные журнала аудита
        router_data = gs.get("router")  # данные маршрутизатора
        core_data = gs.get("core_gpm")  # данные общего блока
        ram_data = gs.get("ram")  # данные памяти
        if not all(isinstance(x, dict) for x in (ledger_data, router_data, core_data)):  # все три обязаны быть словарями
            raise ValueError("malformed governance state")  # ошибка: состояние управления повреждено
        m.cert_log = [CertStepGen(**row) for row in gs.get("cert_log", [])]  # сертификаты из словарей полей
        streams = {  # потоки аудита
            str(k): AuditStream(**v)  # по записям потоков
            for k, v in ledger_data.get("streams", {}).items()
        }  # закрытие словаря
        m.ledger = AuditLedger(  # журнал аудита
            delta_total=float(ledger_data["delta_total"]),  # общий бюджет значимости
            spent={str(k): float(v) for k, v in ledger_data.get("spent", {}).items()},  # расход по задачам
            streams=streams,  # потоки
            floors={str(k): float(v) for k, v in ledger_data.get("floors", {}).items()},  # нижние границы качества
            rejected_tasks=set(str(x) for x in ledger_data.get("rejected_tasks", [])),  # отклонённые задачи как множество строк
            final_test_lo=int(ledger_data.get("final_test_lo", 10_000_000)),  # нижняя граница итоговой проверки (по умолчанию 10^7)
            final_test_hi=int(ledger_data.get("final_test_hi", 20_000_000)),  # верхняя граница итоговой проверки (по умолчанию 2·10^7)
            final_test_touched=bool(ledger_data.get("final_test_touched", False)),  # признак использования итоговой проверки
            next_cursor_base=int(ledger_data.get("next_cursor_base", 0)),  # начало счётчика выборок
        )
        router_k = int(router_data["K"])  # число ветвей маршрутизатора
        if router_k != m.router.K:  # несовпадение с числом ветвей нового процессора
            raise ValueError("router expert count mismatch")  # ошибка: несовпадение числа экспертов маршрутизатора
        weights = np.asarray(router_data["weights"], dtype=np.float64)  # веса ветвей
        cum_loss = np.asarray(router_data["cum_loss"], dtype=np.float64)  # накопленные потери
        if weights.shape != m.router.weights.shape or cum_loss.shape != m.router.cum_loss.shape:  # несовпадение форм
            raise ValueError("router state shape mismatch")  # ошибка: несовпадение формы состояния маршрутизатора
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(cum_loss)):  # есть не конечные числа
            raise ValueError("non-finite router state")  # ошибка: не конечное состояние маршрутизатора
        m.router.eta = float(router_data["eta"])  # темп обучения
        m.router.restore((weights, cum_loss))  # восстановить (веса, потери)
        ram = np.asarray(ram_data, dtype=np.float64)  # матрица памяти
        if ram.shape != m.ram.mem.shape or not np.all(np.isfinite(ram)):  # форма не та или есть не конечные числа
            raise ValueError("RAM state shape/non-finite mismatch")  # ошибка: форма памяти не та или не конечные значения
        m.ram.restore(ram)  # восстановить память
        gpm_snapshot = {  # снимок оптимизатора общего блока
            "W": np.asarray(core_data["W"], dtype=np.float64),  # матрица W
            "U_basis": np.asarray(core_data["U_basis"], dtype=np.float64),  # базис U_basis
            "ef_v": np.asarray(core_data["ef_v"], dtype=np.float64),  # вектор ef_v
        }
        expected_gpm = m.core.gpm.snapshot_optimizer()  # ожидаемый снимок для сверки форм
        for key in ("W", "ef_v"):  # по W и ef_v
            if gpm_snapshot[key].shape != expected_gpm[key].shape:  # форма не та
                raise ValueError(f"core GPM state shape mismatch: {key}")  # ошибка: несовпадение формы состояния оптимизатора
            if not np.all(np.isfinite(gpm_snapshot[key])):  # есть не конечные числа
                raise ValueError(f"non-finite core GPM state: {key}")  # ошибка: не конечное состояние оптимизатора
        # ширина базиса U_basis переменная: закрепление добавляет направления
        u_basis = gpm_snapshot["U_basis"]  # базис
        if (u_basis.ndim != 2 or u_basis.shape[0] != expected_gpm["U_basis"].shape[0]  # если не матрица, или число строк не то
                or u_basis.shape[1] > u_basis.shape[0]):  # или число столбцов больше числа строк
            raise ValueError("core GPM U_basis shape mismatch")  # ошибка: форма базиса не та
        if not np.all(np.isfinite(u_basis)):  # есть не конечные числа
            raise ValueError("non-finite core GPM state: U_basis")  # ошибка: не конечный базис
        if u_basis.shape[1]:  # базис не пуст
            gram = u_basis.T @ u_basis  # матрица Грама U_basisᵀ·U_basis
            if not np.allclose(gram, np.eye(u_basis.shape[1]), atol=1e-7, rtol=0):  # отличается от единичной (допуск 1e−7)
                raise ValueError("core GPM U_basis is not orthonormal")  # ошибка: базис не ортонормирован
        m.core.gpm.restore_optimizer(gpm_snapshot)  # восстановить оптимизатор общего блока
        m.core.core_bytes = int(core_data.get("core_bytes", m.core.core_bytes))  # размер общего блока в байтах
        m.compute_budget = float(gs["compute_budget"])  # бюджет вычислений
        m.risk_rem = float(gs["risk_rem"])  # остаток риска
        m.unsolved_objective = float(gs["unsolved_objective"])  # цель
        m.evals = int(gs.get("evals", 0))  # число вычислений
        m.candidates_generated = int(gs.get("candidates_generated", 0))  # число сгенерированных кандидатов
        m.governance_state_complete = True  # состояние управления загружено полностью
        return m  # вернуть процессор


# ==============================================================================
# 11. Нециркулярный разнообразный саморазвивающийся учебный план и цикл роста
# ==============================================================================


def propose_noncircular_task(  # Предложить новую составную задачу из уже выученных экспертов — разнообразную, невырожденную и ранее не встречавшуюся.
    experts: dict[str, Expert],  # библиотека экспертов
    rng: np.random.Generator,  # генератор случайных чисел
    round_idx: int,  # номер раунда
    seen_sigs: Optional[set[tuple[Any, ...]]] = None,  # сигнатуры уже предложенных задач
    seen_sigs_names: Optional[set[str]] = None,  # имена уже предложенных задач
) -> Optional[tuple[TaskSpec, tuple[str, ...]]]:  # возвращает (задача, имена родителей) или None
    """4567	Предложить новую составную задачу из уже выученных экспертов — разнообразную, невырожденную и ранее не встречавшуюся.
    
            Задача передаётся в learn() только как функция-учитель и области, без готового дерева выражения.
    
            experts — библиотека; rng — генератор случайных чисел; round_idx — номер раунда
            (входит в имя задачи); seen_sigs — сигнатуры уже предложенных задач (пополняется);
            seen_sigs_names — имена уже предложенных задач. Режимы: 0–2 — композиция
            «внешний ∘ внутренний», 3 — булева функция трёх аргументов f(g(x, y), h(y, z)).
            Возвращает (TaskSpec, имена экспертов-родителей) или None, если экспертов нет."""
    seen_sigs_names = seen_sigs_names if seen_sigs_names is not None else set()  # пустое множество, если имена не переданы
    arith2 = [  # бинарные небулевы эксперты
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"  # условие: арность 2 и семейство не «BoolVenn»
    ]
    arith1 = [  # унарные небулевы эксперты
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"  # условие: арность 1 и семейство не «BoolVenn»
    ]
    bool2 = [  # бинарные булевы эксперты
        e for e in experts.values() if e.program.arity == 2 and e.family == "BoolVenn"  # условие: арность 2 и семейство «BoolVenn»
    ]
    if not arith2:  # бинарных небулевых экспертов нет — задачу не построить
        return None  # вернуть None
    pool = arith2 + arith1  # пул внутренних экспертов: бинарные и унарные небулевы
    for attempt in range(32):  # до 32 попыток
        mode = int(rng.integers(0, 4))  # режим — случайное целое 0..3
        if mode == 3 and len(bool2) >= 2:  # режим 3 и есть хотя бы два булевых бинарных эксперта
            f_ex = bool2[int(rng.choice(len(bool2)))]  # случайный булевый эксперт f
            g_ex = bool2[int(rng.choice(len(bool2)))]  # случайный булевый эксперт g
            h_ex = bool2[int(rng.choice(len(bool2)))]  # случайный булевый эксперт h
            teacher = (  # учитель: f(g(x, y), h(y, z))
                lambda x, y, z, f=f_ex, g=g_ex, h=h_ex, env=experts: f.program.eval(  # лямбда с зафиксированными f, g, h и средой
                    (  # пара аргументов
                        g.program.eval((int(x), int(y)), env),  # вычислить g(x, y)
                        h.program.eval((int(y), int(z)), env),  # вычислить h(y, z)
                    ),
                    env,  # среда вычисления
                )
            )  # закрытие лямбды
            vals = {teacher(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)}  # множество значений учителя на всех 8 наборах x, y, z ∈ {0, 1}
            if len(vals) <= 1:  # вырожденная функция (одно значение)
                continue  # другая попытка
            sig = (  # сигнатура задачи
                "b3",  # вид «булева от трёх»
                tuple(teacher(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)),  # таблица значений на 8 наборах x, y, z ∈ {0, 1}
            )  # закрытие кортежа
            if seen_sigs is not None and sig in seen_sigs:  # такая сигнатура уже была
                continue  # другая попытка
            if seen_sigs is not None:  # множество сигнатур передано
                seen_sigs.add(sig)  # запомнить сигнатуру
            sp = TaskSpec(  # описание задачи
                f"V9_BOOL3_R{round_idx}_{attempt}",  # имя: булева от трёх, раунд, попытка
                3,  # арность 3
                ((0, 1), (0, 1), (0, 1)),  # три области {0, 1}
                teacher,  # учитель
                family="auto-bool3",  # семейство «auto-bool3»
            )
            return sp, (f_ex.name, g_ex.name, h_ex.name)  # вернуть задачу и имена f, g, h

        outer = arith2[int(rng.choice(len(arith2)))]  # внешний (бинарный) эксперт
        inner = pool[int(rng.choice(len(pool)))]  # внутренний эксперт из пула
        if inner.program.arity == 1:  # внутренний эксперт унарный
            xs = tuple(  # аргументы внешнего
                x  # первый аргумент примера
                for x in inner.domains[0]  # по значениям первой области внутреннего эксперта
                if abs(x) <= 6  # только небольшие: |x| ≤ 6
                and inner.supports((x,))  # внутренний эксперт поддерживает вход (x,)
                and inner.program.eval((x,), experts) in outer.domains[0]  # и его значение inner(x) лежит в первой области внешнего эксперта
            )
            ys = tuple(  # значения второго аргумента y
                y  # второй аргумент примера
                for y in outer.domains[1]  # по второй области внешнего эксперта
                if abs(y) <= 6  # только небольшие: |y| ≤ 6
                and all(  # и для всех x из xs
                    outer.supports((inner.program.eval((x,), experts), y)) for x in xs  # внешний эксперт поддерживает пару (inner(x), y)
                )
            )  # закрытие ys
            if len(xs) < 3 or len(ys) < 3:  # слишком узкая область (меньше 3 значений по любому аргументу)
                continue  # другая попытка
            tt = tuple(  # таблица значений tt: тройки (x, y, v)
                (  # открытие тройки
                    x,  # первый аргумент x
                    y,  # второй аргумент y
                    outer.program.eval((inner.program.eval((x,), experts), y), experts),  # значение v = outer(inner(x), y)
                )
                for x in xs  # по x из xs
                for y in ys  # по y из ys
            )
            vals = {v for _, _, v in tt}  # множество различных значений
            if (  # если функция вырождена
                len(vals) <= 2  # не более двух различных значений
                or all(v == x for x, _, v in tt)  # или всюду v = x
                or all(v == y for _, y, v in tt)  # или всюду v = y
            ):
                continue  # другая попытка
            sig = ("a2", xs, ys, tt)  # сигнатура: вид «a2», области и таблица значений
            if seen_sigs is not None and sig in seen_sigs:  # такая сигнатура уже была
                continue  # другая попытка
            if seen_sigs is not None:  # если передано множество уже виденных сигнатур
                seen_sigs.add(sig)  # запомнить сигнатуру
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(  # учитель: outer(inner(x), y)
                (i.program.eval((int(x),), env), int(y)), env  # пара (inner(x), y) как целые
            )
            sp = TaskSpec(  # описание задачи
                _short_auto_name(  # уникальное короткое имя
                    round_idx, attempt, inner.name, outer.name, seen_sigs_names  # из раунда, попытки и имён внутреннего и внешнего экспертов
                ),
                2,  # арность 2
                (xs, ys),  # области (xs, ys)
                teacher,  # учитель
                valid_rule=outer.valid_rule,  # правило корректности внешнего эксперта
                family="auto-composed",  # семейство «автоматическая композиция»
            )
            return sp, (outer.name, inner.name)  # вернуть задачу и имена внешнего и внутреннего экспертов
        else:  # иначе внутренний эксперт бинарный: задача outer(x, inner(x, y))
            xs = tuple(  # значения первого аргумента x
                x for x in inner.domains[0] if abs(x) <= 5 and x in outer.domains[0]  # из первой области внутреннего эксперта: |x| ≤ 5 и x в первой области внешнего
            )
            ys_cand = tuple(y for y in inner.domains[1] if abs(y) <= 5)  # кандидаты во второй аргумент: |y| ≤ 5
            ys = tuple(  # значения второго аргумента y
                y  # второй аргумент примера
                for y in ys_cand  # по кандидатам
                if all(  # если для всех x из xs
                    inner.supports((x, y))  # внутренний эксперт поддерживает (x, y)
                    and outer.supports((x, inner.program.eval((x, y), experts)))  # и внешний поддерживает (x, inner(x, y))
                    for x in xs  # по всем x из области xs
                )  # закрытие all
            )
            if len(xs) < 3 or len(ys) < 3:  # слишком узкая область
                continue  # другая попытка
            tt = tuple(  # таблица значений tt: тройки (x, y, v)
                (  # открытие тройки
                    x,  # первый аргумент x
                    y,  # второй аргумент y
                    outer.program.eval(  # вычисление v = outer(x, inner(x, y))
                        (x, inner.program.eval((x, y), experts)), experts  # пара (x, inner(x, y)) и эксперты — вычисление внешнего эксперта
                    ),  # закрытие вычисления inner(x, y)
                )
                for x in xs  # по x из xs
                for y in ys  # по y из ys
            )
            vals = {v for _, _, v in tt}  # множество различных значений
            if (  # если функция вырождена
                len(vals) <= 2  # не более двух различных значений
                or all(v == x for x, _, v in tt)  # или всюду v = x
                or all(v == y for _, y, v in tt)  # или всюду v = y
            ):
                continue  # другая попытка
            sig = ("a2", xs, ys, tt)  # сигнатура: вид «a2», области и таблица значений
            if seen_sigs is not None and sig in seen_sigs:  # такая сигнатура уже была
                continue  # пропустить эту попытку
            if seen_sigs is not None:  # запомнить сигнатуру
                seen_sigs.add(sig)  # запомнить сигнатуру
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(  # учитель: outer(x, inner(x, y))
                (int(x), i.program.eval((int(x), int(y)), env)), env  # пара (x, inner(x, y)) как целые
            )
            sp = TaskSpec(  # описание задачи
                _short_auto_name(  # уникальное короткое имя
                    round_idx, attempt, inner.name, outer.name, seen_sigs_names  # аргументы имени: раунд, попытка, имена внутреннего и внешнего экспертов, занятые имена
                ),  # арность 2
                2,  # арность 2
                (xs, ys),  # области (xs, ys)
                teacher,  # учитель
                valid_rule=inner.valid_rule,  # правило корректности внутреннего эксперта
                family="auto-composed",  # семейство «автоматическая композиция»
            )
            return sp, (outer.name, inner.name)  # вернуть задачу и имена внешнего и внутреннего экспертов
    return None  # все попытки исчерпаны — новой задачи нет


def _short_auto_name(  # Построить короткое уникальное имя автоматически составленной задачи.
    round_idx: int, attempt: int, inner: str, outer: str, seen: set | None = None  # имена внутреннего и внешнего экспертов
) -> str:  # множество занятых имён или None; возвращает имя
    """Вырезать из имени служебные приставки: убрать все приставки «V9_AUTO_» и «V9_BOOL3_»,
                сегменты раунда «R<число>_», ведущий «<число>_» и хвостовые «_<число>(_<число>)»
                (остаётся первая из двух хвостовых цифр). Возвращает содержательную часть имени."""

    def core(n: str) -> str:
        """Ядро имени: имя без служебных приставок и номеров.
        
                    n — исходное имя. Возвращает строку."""
        for part in ("V9_AUTO_", "V9_BOOL3_"):  # по служебным приставкам
            while n.startswith(part):  # пока имя начинается с приставки
                n = n[len(part) :]  # отрезать приставку: n = n[len(part):]
        # убрать сегмент раунда «R<число>_» и лишние номера раунда и попытки
        import re as _re  # импорт модуля регулярных выражений под локальным именем

        n = _re.sub(r"R\d+_", "", n)  # убрать «R<число>_» в любом месте
        n = _re.sub(r"^\d+_", "", n)  # убрать ведущий «<число>_»
        n = _re.sub(r"_\d+(_\d+)$", r"\1", n)  # хвост «_<число>_<число>» заменить на «_<первое число>»
        return n  # вернуть содержательную часть

    base = f"{core(inner)}_{core(outer)}"[:48]  # база имени «ядро_внутреннего_ядро_внешнего», не длиннее 48 символов
    name = base  # имя-кандидат
    k = 2  # счётчик суффиксов
    while seen is not None and name in seen:  # пока имя занято
        name = f"{base}_{k}"  # имя с суффиксом _k
        k += 1  # k = k + 1 — следующий суффикс
    if seen is not None:  # если множество передано
        seen.add(name)  # занять имя
    return name  # вернуть имя


def grow_deep(  # Цикл роста модели по раундам: выучить набор задач, предложить и выучить составные задачи, сжать библиотеку, найти теоремы.
    model: GrowingHagiProcessor,  # процессор
    suite: Sequence[TaskSpec],  # набор задач
    max_rounds: int = 3,  # число раундов
    train_neural: bool = True,  # обучать ли нейросетевые адаптеры
    out_dir: str = ".omc/v9_grow",  # каталог снимков
) -> dict[str, Any]:  # возвращает словарь с итогами роста
    out = Path(out_dir)  # каталог вывода
    out.mkdir(parents=True, exist_ok=True)  # создать каталог вместе с родительскими
    rnd = np.random.default_rng(20260101)  # генератор случайных чисел с фиксированным зерном
    seen_sigs: set[tuple[Any, ...]] = set()  # сигнатуры уже предложенных задач
    rounds = []  # итоги раундов
    for r in range(max_rounds):  # по раундам
        t0 = time.time()  # начало раунда
        commits = 0  # число принятых изменений в раунде
        for sp in suite:  # по задачам набора
            ev = model.learn(  # выучить задачу
                sp, seed=10_000 + r * 7919 + len(sp.name), train_neural=train_neural  # зерно зависит от раунда и длины имени
            )
            if ev["action"] == "COMMIT":  # принято
                commits += 1  # принятий + 1
        proposed = 0  # число предложенных задач
        for k in range(8):  # до 8 предложений
            prop = propose_noncircular_task(model.experts, rnd, r, seen_sigs)  # предложить составную задачу
            if prop is None:  # предложить не удалось
                continue  # следующее предложение
            sp, _ = prop  # задача (родители не нужны)
            if sp.name in model.experts:  # уже выучена
                continue  # эта задача уже выучена — пропустить
            proposed += 1  # предложено + 1
            ev = model.learn(  # выучить предложенную задачу
                sp, seed=500_000 + r * 104_729 + k, train_neural=train_neural  # зерно зависит от раунда и номера предложения
            )
            if ev["action"] == "COMMIT":  # принято
                commits += 1  # принятий + 1
        comp = model.compress_library_mdl()  # сжать библиотеку методом минимальной длины описания
        new_th = TheoremMiner().mine(model)  # найти новые теоремы
        nq = model.neural_quality_summary()  # сводка нейросетевого качества
        rounds.append(  # добавить итог раунда
            {  # открытие словаря итога раунда
                "round": r,  # номер раунда
                "experts": len(model.experts),  # число экспертов
                "commits": commits,  # принятий в раунде
                "proposed": proposed,  # предложено задач
                "theorems_total": len(model.theorems),  # всего теорем
                "theorems_new": len(new_th),  # новых теорем
                "sym_bits": comp["after_bits"],  # биты библиотеки после сжатия
                "dag_refactored": comp["refactored"],  # число преобразований графа вызовов
                "neural_bytes": nq["total_neural_bytes"],  # байты нейросетевой части
                "neural_val_acc": round(nq["mean_val_exact_acc"], 4),  # средняя точность адаптеров на val, округлённая до 4 знаков
                "neural_val_mse": round(nq["mean_val_mse"], 4),  # средний квадрат ошибки на val, округлённый до 4 знаков
                "certs_ok": model.verify_certificates()["all_ok"],  # сертификаты корректны
                "ram_avail_gb": round(ram_available_bytes() / 2**30, 2),  # доступная оперативная память в ГиБ: байты / 2^30, округлённо до 2 знаков
                "elapsed": round(time.time() - t0, 3),  # длительность раунда в секундах
            }
        )  # закрытие словаря итога
        (out / f"model_r{r}.json").write_text(  # записать снимок модели раунда
            json.dumps(  # превращение словаря в текст JSON
                {  # словарь снимка
                    "round": r,  # номер раунда
                    "experts": {n: e.to_dict() for n, e in model.experts.items()},  # эксперты как словари
                    "theorems": [vars(t) for t in model.theorems],  # теоремы как словари полей
                },
                indent=2,  # отступ 2
            ),  # закрытие json.dumps
            encoding="utf-8",  # кодировка UTF-8
        )
    return {  # итог словарём
        "rounds": rounds,  # итоги раундов
        "experts": len(model.experts),  # число экспертов
        "theorems": len(model.theorems),  # число теорем
    }


# ==============================================================================
# 12. Полная самопроверка
# ==============================================================================


def self_test_core() -> None:
    """Полная самопроверка ядра: выучить набор задач, проверить предсказания и сохранение, отказ по отравленной метке, рост, упаковку, теоремы полиномов Жегалкина, движок CRAM, память и ассоциативную память.
            При любом расхождении возбуждает AssertionError; в конце печатает строку с итогами и числом секунд."""
    t0 = time.time()  # начало отсчёта времени
    suite = make_task_suite()  # набор задач
    model = GrowingHagiProcessor(node_budget=8000)  # процессор с бюджетом 8000 узлов
    for i, sp in enumerate(suite):  # по задачам набора
        ev = model.learn(sp, seed=10_000 + i, train_neural=True)  # выучить задачу с адаптером
        assert ev["action"] in ("COMMIT", "REUSE"), (sp.name, ev)  # каждая задача либо принята, либо переиспользована
    check = model.verify_all(suite)  # проверка всего набора
    assert check["accuracy"] == 1.0 and check["committed"] == len(suite), check  # точность 1.0 и число экспертов равно числу задач
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"  # ADD(100, 3) вне области — отказ

    pm = GrowingHagiProcessor()  # отдельный процессор для проверки отката
    snap_before = pm._snapshot_state_with_optimizer()  # снимок состояния до обучения
    pev = pm.learn(  # выучить ADD с «отравленной» меткой
        next(s for s in suite if s.name == "ADD"), seed=77, poison_one_label=True  # задача ADD и зерно 77
    )
    assert pev["action"] == "HOLD"  # задача удержана
    snap_after = pm._snapshot_state_with_optimizer()  # снимок состояния после удержания
    assert np.array_equal(  # веса общего блока не изменились
        snap_before["core"]["gpm"]["W"], snap_after["core"]["gpm"]["W"]  # сравнение матриц W до и после
    )
    assert np.array_equal(snap_before["router"][0], snap_after["router"][0])  # веса маршрутизатора не изменились

    rep = grow_deep(model, suite, max_rounds=2, out_dir="/tmp/grow_test")  # два раунда роста
    assert rep["experts"] > len(suite), rep  # экспертов стало больше, чем задач в наборе
    assert model.verify_certificates()["all_ok"], model.verify_certificates()  # сертификаты корректны

    vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5, -0.75, 0.125, 0.0, -3.2, 0.05])  # пробный вектор из 11 компонент
    raw, dec = pack_vector_dyadic5(vec, stages=2)  # упаковка в 2 ступени: сырые байты и декодированный вектор
    assert np.allclose(dec, unpack_vector_dyadic5(raw))  # декодирование сырых байтов возвращает декодированный вектор

    vb = pack_venn_16_flat()  # упакованная таблица 16 функций Венна
    assert len(vb) == 13 and unpack_venn_16_flat(vb) == list(range(16))  # её размер 13 байт и она содержит числа 0..15
    for m in range(16):  # по маскам m = 0..15
        for x in (0, 1):  # по x ∈ {0, 1}
            for y in (0, 1):  # по y ∈ {0, 1}
                assert venn_head_eval(m, (x, y)) == ((m >> ((x << 1) | y)) & 1)  # голова функции Венна даёт бит m с номером 2·x + y

    tt = anf_transform([0, 1, 1, 0])  # коэффициенты АНФ таблицы [0, 1, 1, 0] (исключающее ИЛИ)
    assert anf_to_expr(tt, 2).eval((1, 0)) == 1  # выражение даёт 1 на (1, 0)
    c3 = anf_compose3(lambda a, b: a | b, [0, 0, 0, 1], [0, 1, 1, 0])  # композиция: (a | b) с таблицами [0, 0, 0, 1] и [0, 1, 1, 0]
    for x in (0, 1):  # по x
        for y in (0, 1):  # по y
            assert anf_eval(c3, 2, (x, y)) == ((x & y) | (x ^ y))  # результат равен (x & y) | (x ^ y)

    col = CRAMEngine.solve_coloring(  # раскраска цикла из 4 вершин в 2 цвета с заданным цветом вершины 0
        [(0, 1), (1, 2), (2, 3), (3, 0)], 4, 2, givens={0: 1}  # рёбра цикла 0–1–2–3–0, 4 вершины, 2 цвета, цвет вершины 0 задан = 1
    )  # закрытие вызова
    assert col["validity_certificate"] and col["givens_respected"]  # раскраска корректна и учитывает заданные цвета
    adj_m = [[1], [0, 2], [1, 3], [2]]  # список смежности цепочки из 4 вершин
    lz = CRAMEngine.solve_latent_maze(4, adj_m, goal=0)  # решение скрытого лабиринта с целью в вершине 0
    assert abs(lz["z"][0] - 21.0 / 55.0) < 1e-4  # первая компонента решения равна 21/55 с точностью 1e−4
    mf = CRAMEngine.mean_field_kl_step(  # шаг среднего поля
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # единичные свидетельства 2×2
        np.zeros((2, 2, 2, 2)),  # нулевые парные поля
        np.full((2, 2), 0.5),  # средние значения всех 0.5
    )
    assert mf["monotonic"]  # шаг монотонный
    alt = CRAMEngine.alternating_latent_visible_solve(4, adj_m, goal=0, num_colors=2)  # чередующееся решение скрытого лабиринта в 2 цвета
    assert alt["monotone_drop"]  # энергия монотонно убывает
    pl = CRAMEngine.train_cram_pseudolikelihood(  # обучение по псевдо-правдоподобию на трёх примерах в 2 цвета
        np.array([[0, 1], [0, 1], [1, 0]]), num_colors=2  # три примера-строки в 2 цвета
    )  # закрытие вызова
    assert pl["mean_pl_loss"] < 0.7  # средняя потеря меньше 0.7

    gpm = ContinualGPMMemory(6, 2, seed=1)  # память на 6 входов и 2 выхода
    X1 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])  # входы первой задачи: два орта
    Y1 = np.array([[1.0, -1.0], [0.5, 0.5]])  # цели первой задачи
    gpm.train_task_ef21_muon(X1, Y1, steps=20)  # обучить первую задачу за 20 шагов
    pred1_before = X1 @ gpm.W  # предсказание на первой задаче до закрепления: X1·W
    gpm.consolidate_subspace(X1)  # закрепить подпространство первой задачи
    X2 = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])  # вход второй задачи
    Y2 = np.array([[2.0, -2.0]])  # цель второй задачи
    gpm.train_task_ef21_muon(X2, Y2, steps=20)  # обучить вторую задачу за 20 шагов
    pred1_after = X1 @ gpm.W  # предсказание на первой задаче после обучения второй
    assert float(np.max(np.abs(pred1_before - pred1_after))) < 1e-10  # предсказание не изменилось (допуск 1e−10) — забывания нет

    assert abs(hnet_opt_ratio(1.0, 4.0, 8.0) - 4.0) < 1e-9  # оптимальная доля: (2·4·8/1)^(1/3) = 4 с точностью 1e−9
    ram = GDNRAM(8, 4)  # память на 8 слотов размерности 4
    ram.store(2, np.array([1.0, 2.0, 3.0, 4.0]))  # записать в слот 2
    ram.store(5, np.array([9.0, 8.0, 7.0, 6.0]))  # записать в слот 5
    assert np.allclose(ram.load(2), [1.0, 2.0, 3.0, 4.0])  # чтение слота 2 возвращает записанное
    assert np.allclose(ram.load_stutter_head([2, 2, 5, 5]), [1.0, 2.0, 3.0, 4.0])  # «запинающаяся» голова [2, 2, 5, 5] читает первый адрес после схлопывания повторов — слот 2

    save_path = Path("v9_selftest_model.json")  # путь к файлу модели
    model.save(save_path)  # сохранить модель
    reloaded = GrowingHagiProcessor.load(save_path)  # загрузить модель
    assert reloaded.verify_all(suite)["accuracy"] == 1.0  # точность загруженной модели 1.0
    assert reloaded.predict("ADD", 7, -3)["value"] == 4  # ADD(7, −3) = 4
    assert reloaded.predict("ADD", 99, 1)["status"] == "ABSTAIN"  # ADD(99, 1) вне области — отказ
    for nm, ex in model.experts.items():  # по экспертам
        if ex.adapter is not None:  # у которых есть адаптер
            xs0 = tuple([ex.domains[0][0]] + [d[0] for d in ex.domains[1:]])  # первый допустимый вход: первые значения каждой области
            if ex.supports(xs0):  # эксперт поддерживает этот вход
                p1 = model.predict(nm, *xs0, mode="neural")["value"]  # нейросетевое предсказание до сохранения
                p2 = reloaded.predict(nm, *xs0, mode="neural")["value"]  # нейросетевое предсказание после загрузки
                assert abs(p1 - p2) < 1e-9, (nm, p1, p2)  # предсказания совпадают (допуск 1e−9)
    save_path.unlink(missing_ok=True)  # удалить файл, если он существует

    nq = model.neural_quality_summary()  # сводка нейросетевого качества
    dt = time.time() - t0  # затраченное время
    print(  # печать итога
        f"V9 SELF_TEST PASS ({dt:.2f}s): {check['committed']} base experts acc=1.0 -> "  # первая часть строки: время, число экспертов, точность
        f"{rep['experts']} total experts, {len(model.theorems)} theorems, "  # всего экспертов и число теорем
        f"neural_acc={nq['mean_val_exact_acc']:.4f}, neural_mse={nq['mean_val_mse']:.4f}, "  # средняя точность и средний квадрат ошибки адаптеров
        f"neural_bytes={nq['total_neural_bytes']}, certs_ok=True"  # байты нейросетевой части и признак корректности сертификатов
    )  # закрытие print


# ==============================================================================
# Единый интерфейс командной строки
# ==============================================================================


def _benchmark() -> None:
    """Замер ядра: выучить набор задач, выполнить рост, измерить время, точность и занимаемую память.
    
            Считает время обучения набора и двух раундов роста, точность на наборе, число
            экспертов и теорем после роста, метрики адаптеров, число байт тензоров в памяти
            (общий блок и вся модель), размер этого файла и корректность сертификатов.
            Печатает итог как JSON с отступом 2."""
    """Замер ядра: выучить набор задач, выполнить рост, измерить время, точность и занимаемую память.
    
            Считает время обучения набора и двух раундов роста, точность на наборе, число
            экспертов и теорем после роста, метрики адаптеров, число байт тензоров в памяти
            (общий блок и вся модель), размер этого файла и корректность сертификатов.
            Печатает итог как JSON с отступом 2."""

    suite = make_task_suite()  # набор задач
    model = GrowingHagiProcessor(node_budget=8000)  # процессор с бюджетом 8000 узлов
    t0 = time.time()  # начало отсчёта
    for i, sp in enumerate(suite):  # по задачам набора
        model.learn(sp, seed=10_000 + i, train_neural=True)  # выучить задачу с адаптером
    learn_elapsed = time.time() - t0  # время обучения набора
    check = model.verify_all(suite)  # проверка всего набора
    t0 = time.time()  # начало отсчёта
    rep = grow_deep(model, suite, max_rounds=2, out_dir=".omc/bench")  # два раунда роста
    grow_elapsed = time.time() - t0  # время роста
    nq = model.neural_quality_summary()  # сводка нейросетевого качества
    resident = sum(  # байты массивов NumPy среди атрибутов модели
        int(v.nbytes) for v in model.__dict__.values() if isinstance(v, np.ndarray)  # размер массива в байтах
    )
    core = model.core  # общий блок весов
    resident_core = sum(  # байты массивов среди атрибутов общего блока
        int(a.nbytes) for a in vars(core).values() if isinstance(a, np.ndarray)  # размер массива в байтах
    )
    resident_core += sum(  # прибавить массивы оптимизатора общего блока
        int(a.nbytes) for a in vars(core.gpm).values() if isinstance(a, np.ndarray)  # размер массива в байтах
    )
    result = {  # итог словарём
        "version": V9_VERSION,  # версия схемы
        "exact_committed": check["committed"],  # число принятых экспертов
        "exact_accuracy": check["accuracy"],  # точность на наборе
        "experts_after_growth": rep["experts"],  # число экспертов после роста
        "theorems": rep["theorems"],  # число теорем
        "neural_val_acc": round(nq["mean_val_exact_acc"], 4),  # средняя точность адаптеров на val, округлённая до 4 знаков
        "neural_val_mse": round(nq["mean_val_mse"], 4),  # средний квадрат ошибки на val, округлённый до 4 знаков
        "neural_payload_bytes": nq["total_neural_bytes"],  # байты упакованной нейросетевой части
        "resident_tensor_bytes_core": resident_core,  # байты тензоров общего блока
        "resident_tensor_bytes_model": resident + resident_core,  # байты тензоров всей модели
        "file_bytes": Path(__file__).stat().st_size,  # размер этого файла в байтах
        "learn_suite_seconds": round(learn_elapsed, 2),  # время обучения набора в секундах
        "grow_2rounds_seconds": round(grow_elapsed, 2),  # время двух раундов роста в секундах
        "certs_ok": model.verify_certificates()["all_ok"],  # сертификаты корректны
    }
    print(json.dumps(result, indent=2))  # напечатать итог как JSON с отступом 2


def _cli() -> None:
    """Разобрать аргументы командной строки и запустить выбранное действие.
    
            Флаги --self-test, --self-test-core, --self-test-v7, --self-test-v5 — самопроверки
            разных версий ядра; --grow, --grow-deep — циклы роста; --benchmark-core — замер;
            --max-rounds — число раундов (по умолчанию 4); --out-dir — каталог вывода.
            Без флага печатается справка."""
    _p = argparse.ArgumentParser(  # парсер аргументов
        description="HAGI unified single-file build (v5+v7.4+v8.4+v9)"  # описание программы
    )
    _p.add_argument("--self-test", action="store_true", help="v8.4 self-test")  # флаг самопроверки версии 8.4
    _p.add_argument("--self-test-core", action="store_true", help="canonical kernel self-test")  # флаг самопроверки канонического ядра
    _p.add_argument("--self-test-v7", action="store_true", help="v7.4 self-test")  # флаг самопроверки версии 7.4
    _p.add_argument("--self-test-v5", action="store_true", help="v5 self-test")  # флаг самопроверки версии 5
    _p.add_argument("--grow", action="store_true", help="v8.4 growth loop")  # флаг цикла роста версии 8.4
    _p.add_argument("--grow-deep", action="store_true", help="canonical growth loop")  # флаг канонического цикла роста
    _p.add_argument("--benchmark-core", action="store_true", help="canonical benchmark")  # флаг замера канонического ядра
    _p.add_argument("--max-rounds", type=int, default=4)  # максимальное число раундов (по умолчанию 4)
    _p.add_argument("--out-dir", default="grow_out")  # каталог вывода (по умолчанию «grow_out»)
    _a = _p.parse_args()  # разобранные аргументы
    if _a.self_test:  # если указана самопроверка версии 8.4
        self_test_v8()  # запустить её
    elif _a.self_test_core:  # иначе если указана самопроверка ядра
        self_test_core()  # запустить её
    elif _a.self_test_v7:  # иначе самопроверка версии 7.4
        self_test_v7()  # запустить её
    elif _a.self_test_v5:  # иначе самопроверка версии 5
        self_test()  # запустить её
    elif _a.grow:  # иначе цикл роста версии 8.4
        _suite = make_task_suite()  # набор задач
        _m = GrowingHagiProcessorV8()  # процессор версии 8.4
        print(  # напечатать итог роста
            json.dumps(  # превращение словаря в текст JSON
                grow_v8(_m, _suite, max_rounds=_a.max_rounds, out_dir=_a.out_dir),  # итог grow_v8 с числом раундов и каталогом
                indent=2,  # отступ 2
            )
        )  # закрытие json.dumps
    elif _a.grow_deep:  # иначе канонический цикл роста
        _suite = make_task_suite()  # набор задач
        _m = GrowingHagiProcessor()  # канонический процессор
        print(  # напечатать итог роста
            json.dumps(  # превращение словаря в текст JSON
                grow_deep(_m, _suite, max_rounds=_a.max_rounds, out_dir=_a.out_dir),  # итог grow_deep с числом раундов и каталогом
                indent=2,  # отступ 2
            )
        )
    elif _a.benchmark_core:  # иначе замер ядра
        _benchmark()  # запустить замер
    else:  # иначе флагов нет
        _p.print_help()  # напечатать справку
