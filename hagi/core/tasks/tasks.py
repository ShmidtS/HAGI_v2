"""Описание задач и учебный набор — часть символьного ядра HAGI.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import itertools  # импорт стандартной библиотеки, нужный этой части
import math  # импорт стандартной библиотеки, нужный этой части
from dataclasses import dataclass  # импорт стандартной библиотеки, нужный этой части
from typing import Callable, Sequence  # импорт стандартной библиотеки, нужный этой части
from hagi.core.memory.aliases import _bit_nand8, _bit_nor8, _bit_not8, _bit_xnor8, _pow_int, _sign  # имена из «aliases» (прежний общий импорт из пакета наследия)

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
    0: "BOOL_FALSE",  # тождественный ноль: m = 0: f = 0
    1: "BOOL_NOR",  # m = 1: f = НЕ(x ИЛИ y)
    2: "BOOL_NOT_X_AND_Y",  # m = 2: f = (НЕ x) И y
    3: "BOOL_NOT_X",  # m = 3: f = НЕ x
    4: "BOOL_X_AND_NOT_Y",  # m = 4: f = x И (НЕ y)
    5: "BOOL_NOT_Y",  # m = 5: f = НЕ y
    6: "BOOL_XOR",  # исключающее или: m = 6: f = x XOR y
    7: "BOOL_NAND",  # m = 7: f = НЕ(x И y)
    8: "BOOL_AND",  # m = 8: f = x И y
    9: "BOOL_XNOR",  # m = 9: f = НЕ(x XOR y)
    10: "BOOL_Y",  # второй аргумент: m = 10: f = y
    11: "BOOL_X_IMPLIES_Y",  # m = 11: f = x → y, то есть (НЕ x) ИЛИ y
    12: "BOOL_X",  # первый аргумент: m = 12: f = x
    13: "BOOL_Y_IMPLIES_X",  # m = 13: f = y → x, то есть x ИЛИ (НЕ y)
    14: "BOOL_OR",  # m = 14: f = x ИЛИ y
    15: "BOOL_TRUE",  # тождественная единица: m = 15: f = 1
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
        ("NEG", lambda x: -x),  # отрицание: NEG: f(x) = −x
        ("ABS", abs),  # модуль: ABS: f(x) = |x|
        ("SIGN", _sign),  # SIGN: знак x
        ("SQUARE", lambda x: x * x),  # квадрат: SQUARE: f(x) = x²
        ("CUBE", lambda x: x * x * x),  # куб: CUBE: f(x) = x³
        ("BIT_NOT8", _bit_not8),  # BIT_NOT8: побитовое НЕ по младшим 8 битам
        ("CLAMP_TO_3", lambda x: min(max(x, -3), 3)),  # усечение в отрезок [−3, 3]: CLAMP_TO_3: f(x) = min(max(x, −3), 3)
    ]
    for name, fn in unary:  # по каждой унарной задаче
        suite.append(TaskSpec(name, 1, (U,), fn, family="unary"))  # задача арности 1 с областью U
    binary = [  # список (имя, функция, правило допустимости) бинарных задач
        ("ADD", lambda x, y: x + y, "all"),  # сложение: ADD: x + y
        ("SUB", lambda x, y: x - y, "all"),  # вычитание: SUB: x − y
        ("MUL", lambda x, y: x * y, "all"),  # умножение: MUL: x·y
        ("FLOORDIV", lambda x, y: x // y, "nonzero_y"),  # деление нацело: FLOORDIV: ⌊x / y⌋, y ≠ 0
        ("MOD", lambda x, y: x % y, "nonzero_y"),  # MOD: остаток x mod y, y ≠ 0
        ("GCD", math.gcd, "all"),  # GCD: наибольший общий делитель
        ("LCM", math.lcm, "all"),  # LCM: наименьшее общее кратное
        ("MIN", min, "all"),  # минимум: MIN: min(x, y)
        ("MAX", max, "all"),  # максимум: MAX: max(x, y)
        ("ABSDIFF", lambda x, y: abs(x - y), "all"),  # модуль разности: ABSDIFF: |x − y|
        ("EQ", lambda x, y: int(x == y), "all"),  # EQ: 1, если x = y, иначе 0
        ("NE", lambda x, y: int(x != y), "all"),  # NE: 1, если x ≠ y, иначе 0
        ("LT", lambda x, y: int(x < y), "all"),  # LT: 1, если x < y, иначе 0
        ("LE", lambda x, y: int(x <= y), "all"),  # LE: 1, если x ≤ y, иначе 0
        ("GT", lambda x, y: int(x > y), "all"),  # GT: 1, если x > y, иначе 0
        ("GE", lambda x, y: int(x >= y), "all"),  # GE: 1, если x ≥ y, иначе 0
        ("BIT_AND", lambda x, y: x & y, "all"),  # побитовое «и»: BIT_AND: x & y
        ("BIT_OR", lambda x, y: x | y, "all"),  # побитовое «или»: BIT_OR: x | y
        ("BIT_XOR", lambda x, y: x ^ y, "all"),  # побитовое «исключающее или»: BIT_XOR: x ^ y
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
            TaskSpec("LANG8_ATBASH", 1, (B,), lambda c: 255 - c, family="language"),  # задача: LANG8_ATBASH: f(c) = 255 − c
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

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'TaskSpec', 'BOOL_NAMES', 'make_task_suite',  # имя интерфейса
]  # конец перечня
