"""Перечисление программ-кандидатов по числу аргументов.

Для каждой арности строится список выражений Expr из переменных v[i] и
операций языка выражений. Повторы убираются по текстовому виду: остаётся
первое вхождение каждого текста, затем список сортируется по
(стоимость cost, текст pretty). Результат — список Program с данным числом
аргументов.

    arity 1: −x, |x|, sign x, побитовое НЕ в 8 битах, x², x³, срезка в [−3, 3]
    arity 2: 22 двоичные операции x op y, y − x, |x − y|, ||x − y||, ||y − x||,
             |a| для шести атомов a из {x+y, x−y, y−x, x·y, |x|, |y|}
    arity 3: x·y+z, clamp, медиана трёх, x·y·z, x·y+z, max(min(x,y), min(max(x,y), z))
    иная арность: пустой список
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
from typing import Iterable  # типы для аннотаций
from hagi.core.language.expr_dsl import (  # из языка выражений импортировать:
    Expr,  # узел выражения (операция и аргументы)
    Program  # символьная программа с оценкой стоимости
)  # конец импорта из модуля выражений

# ------------------------- Перечисление программ-кандидатов ------------------------  # раздел перечисления


def _dedupe_candidates(candidates: Iterable[Expr]) -> list[Expr]:  # убрать повторы
    """Убрать повторы по текстовому виду и отсортировать по (стоимость, текст).

        Из одинаковых по написанию выражений остаётся первое; результат
        отсортирован по ключу (cost, pretty)."""
    by_text: dict[str, Expr] = {}  # соответствие текст -> выражение
    for e in candidates:  # по кандидатам
        by_text.setdefault(e.pretty(), e)  # записать выражение, если такого текста ещё не было
    return sorted(by_text.values(), key=lambda e: (e.cost, e.pretty()))  # отсортированный по (cost, текст) список


def candidate_programs(arity: int) -> list[Program]:  # кандидаты для арности arity
    """Список программ-кандидатов для функции с arity аргументами.

        Для arity 1, 2, 3 — готовые наборы выражений (перечислены в docstring
        модуля); для прочей арности — пустой список."""
    v = [Expr.var(i) for i in range(arity)]  # переменные v[0..arity−1]
    cands: list[Expr] = []  # накопитель выражений
    if arity == 1:  # один аргумент
        x = v[0]  # x — единственная переменная
        cands += [  # добавить выражения
            Expr("neg", (x,)),  # смена знака: −x
            Expr("abs", (x,)),  # модуль: |x|
            Expr("sign", (x,)),  # знак: sign x
            Expr("bit_not8", (x,)),  # побитовое НЕ в 8 битах
            Expr("mul", (x, x)),  # квадрат: x·x = x²
            Expr("mul", (Expr("mul", (x, x)), x)),  # куб: (x·x)·x = x³
            Expr("clamp3", (x,)),  # срезка x в отрезок [−3, 3]
        ]  # конец списка
        return [Program(arity, e) for e in _dedupe_candidates(cands)]  # программы из неповторяющихся выражений
    if arity == 2:  # два аргумента
        x, y = v  # x и y — переменные
        for op in (  # по кодам двоичных операций
            "add",  # сумма: x + y
            "sub",  # разность: x − y
            "mul",  # произведение: x·y
            "floordiv",  # частное: ⌊x / y⌋
            "mod",  # остаток: x mod y
            "gcd",  # НОД(x, y)
            "lcm",  # НОК(x, y)
            "min",  # минимум: min(x, y)
            "max",  # максимум: max(x, y)
            "eq",  # равенство: x = y
            "ne",  # неравенство: x ≠ y
            "lt",  # меньше: x < y
            "le",  # не больше: x ≤ y
            "gt",  # больше: x > y
            "ge",  # не меньше: x ≥ y
            "bit_and",  # побитовое И
            "bit_or",  # побитовое ИЛИ
            "bit_xor",  # исключающее ИЛИ
            "bit_nand8",  # И-НЕ в 8 битах
            "bit_nor8",  # ИЛИ-НЕ в 8 битах
            "bit_xnor8",  # равенство битов в 8 битах
            "pow_int",  # x в целой степени y
        ):  # конец кортежа операций
            cands.append(Expr(op, (x, y)))  # добавить выражение x op y
        cands.append(Expr("sub", (y, x)))  # обратная разность: y − x
        cands.append(Expr("absdiff", (x, y)))  # расстояние: |x − y|
        cands.append(Expr("abs", (Expr("sub", (x, y)),)))  # модуль разности: |x − y|
        cands.append(Expr("abs", (Expr("sub", (y, x)),)))  # модуль разности: |y − x|
        # малая композиционная грамматика: двухслойные выражения из арифметических атомов
        atoms = [  # атомы — базовые выражения
            Expr("add", (x, y)),  # сумма: x + y
            Expr("sub", (x, y)),  # разность: x − y
            Expr("sub", (y, x)),  # обратная разность: y − x
            Expr("mul", (x, y)),  # произведение: x·y
            Expr("abs", (x,)),  # модуль: |x|
            Expr("abs", (y,)),  # модуль: |y|
        ]  # конец списка атомов
        for a in atoms:  # по атомам
            cands.append(Expr("abs", (a,)))  # модуль атома: |a|
        return [Program(arity, e) for e in _dedupe_candidates(cands)]  # программы из неповторяющихся выражений
    if arity == 3:  # три аргумента
        x, y, z = v  # x, y, z — переменные
        cands += [  # добавить выражения
            Expr("fma", (x, y, z)),  # произведение-сложение: x·y + z
            Expr("clamp", (x, y, z)),  # срезка: min(max(x, y), z)
            Expr("median3", (x, y, z)),  # медиана x, y, z
            Expr("mul3", (x, y, z)),  # произведение трёх: x·y·z
            Expr("add", (Expr("mul", (x, y)), z)),  # сумма произведения: x·y + z
            Expr("max", (Expr("min", (x, y)), Expr("min", (Expr("max", (x, y)), z)))),  # максимум из минимумов: max(min(x, y), min(max(x, y), z))
        ]  # конец списка
        return [Program(arity, e) for e in _dedupe_candidates(cands)]  # программы из неповторяющихся выражений
    return []  # иная арность — кандидатов нет

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'candidate_programs',  # имя интерфейса
]  # конец перечня
