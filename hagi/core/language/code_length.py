"""Длина кода выражений и программ, а также безопасный контракты пакета.

Каждый узел выражения стоит 6 бит (код операции), числа кодируются гамма-кодом
Элиаса, знак — зигзагом. L(узел) = 6 + полезная нагрузка + Σ L(аргументов).
Гамма-код числа m ≥ 1: k = floor(log2 m) нулей, единица, k младших бит; длина 2k + 1.
Зигзаг: z(n) = 2n при n ≥ 0, иначе −2n − 1.

Безопасность векторной проверки: не более 3 уровней операций и |лист| ≤ 128
дают значение, не выходящее за int64, поэтому переполнение не вызывает ложный
отказ. Возвращает batch_fits_safe: список True/False/None (None — нужна
скалярная проверка).
"""
from __future__ import annotations  # аннотации типов не вычисляются при импорте
from typing import Optional, Sequence  # аннотации типов
import numpy as np  # числовые массивы
from hagi.core.language.expr_dsl import (  # имена из модуля языка выражений
    Expr,  # узел выражения
    Program  # программа: выражение и маска
)  # конец импорта
from hagi.core.synthesis.prefilter import _batch_fits_unchecked  # имена из «prefilter» (модуль, где они определены)

# ---- длина кода и контракт безопасного фильтра ----


def _gamma_bits(n: int) -> int:
    """Длина гамма-кода Элиаса числа n: 2·(⌊log2 n⌋) + 1 бит.

        n: целое ≥ 1. Число старших нулей k = n.bit_length() − 1 = ⌊log2 n⌋,
        затем единица и k младших бит числа; всего 2k + 1 бит."""
    n = int(n)  # привести n к целому
    if n < 1:  # гамма-код определён только для положительных чисел
        raise ValueError("gamma code accepts positive integers")  # ошибка при n < 1
    return 2 * (n.bit_length() - 1) + 1  # длина кода: 2·⌊log2 n⌋ + 1


def _zigzag(n: int) -> int:
    """Зигзаг: любое целое n → неотрицательное число.

        z(n) = 2n при n ≥ 0, z(n) = −2n − 1 при n < 0; соответствие взаимно однозначно."""
    n = int(n)  # привести n к целому
    return 2 * n if n >= 0 else -2 * n - 1  # z(n) = 2n, если n ≥ 0, иначе −2n − 1


_TOKEN_BITS = 6  # ширина кода операции: 6 бит (64 зарезервированных кода)


def _expr_description_bits(expr: Expr) -> int:
    """Длина описания выражения expr в битах: L(expr).

        Каждый узел начинается с 6-битного кода операции. Индекс переменной,
        целая константа и коэффициенты многочлена кодируются гамма-кодом от
        зигзага плюс 1, поэтому константы тоже стоят биты (в отличие от подсчёта
        числа узлов). Это объявленное кодирование для данного языка выражений,
        а не сложность по Колмогорову."""
    if expr.op == "var":  # узел-переменная
        return _TOKEN_BITS + _gamma_bits(int(expr.value or 0) + 1)  # 6 + γ(индекс + 1)
    if expr.op == "const":  # узел-константа
        return _TOKEN_BITS + _gamma_bits(_zigzag(int(expr.value)) + 1)  # 6 + γ(z(значение) + 1)
    if expr.op == "poly":  # многочлен одной переменной
        coeffs = tuple(int(v) for v in (expr.value or ()))  # кортеж целых коэффициентов
        payload = _gamma_bits(len(coeffs) + 1)  # γ(число коэффициентов + 1)
        payload += sum(_gamma_bits(_zigzag(v) + 1) for v in coeffs)  # сумма: += Σ_v γ(z(v) + 1)
        return _TOKEN_BITS + payload + sum(_expr_description_bits(a) for a in expr.args)  # 6 + нагрузка + Σ L(аргумент)
    if expr.op == "poly_nd":  # многочлен многих переменных
        terms = tuple(expr.value or ())  # кортеж пар (коэффициент, степени)
        payload = _gamma_bits(len(terms) + 1)  # γ(число слагаемых + 1)
        for coefficient, powers in terms:  # по слагаемым
            payload += _gamma_bits(_zigzag(int(coefficient)) + 1)  # += γ(z(коэффициент) + 1)
            payload += _gamma_bits(len(powers) + 1)  # += γ(число переменных + 1)
            payload += sum(_gamma_bits(int(power) + 1) for power in powers)  # += Σ γ(степень + 1)
        return _TOKEN_BITS + payload  # 6 + нагрузка
    return _TOKEN_BITS + sum(_expr_description_bits(a) for a in expr.args)  # прочий узел: 6 + Σ L(аргумент)


def _program_description_bits(program: Program) -> int:
    """Длина описания программы: 10 бит при заданной маске, иначе L(выражения), 10^9 без выражения."""
    if program.venn_mask is not None:  # маска наложена — выражение не учитывается
        return _TOKEN_BITS + 4  # длина 6 + 4 = 10 бит
    if program.expr is None:  # выражения нет
        return 10**9  # очень большое значение-штраф
    return _expr_description_bits(program.expr)  # L(программы) = L(expr)


_VECTOR_SAFE_OPS = {  # операции, для которых доказуема оценка в int64
    "neg",  # смена знака −x
    "abs",  # модуль |x|
    "sign",  # знак: −1, 0 или 1
    "bit_not8",  # побитовое отрицание в 8 битах
    "clamp3",  # усечение третьим аргументом
    "add",  # сумма a + b
    "sub",  # разность a − b
    "mul",  # произведение a·b
    "floordiv",  # деление нацело a // b
    "mod",  # остаток a mod b
    "gcd",  # НОД(a, b)
    "lcm",  # НОК(a, b)
    "min",  # минимум min(a, b)
    "max",  # максимум max(a, b)
    "eq",  # равно: a = b
    "ne",  # не равно: a ≠ b
    "lt",  # меньше: a < b
    "le",  # не больше: a ≤ b
    "gt",  # больше: a > b
    "ge",  # не меньше: a ≥ b
    "bit_and",  # побитовое И
    "bit_or",  # побитовое ИЛИ
    "bit_xor",  # побитовое исключающее ИЛИ
    "bit_nand8",  # И-НЕ в 8 битах
    "bit_nor8",  # ИЛИ-НЕ в 8 битах
    "bit_xnor8",  # равнозначность битов в 8 битах
    "absdiff",  # разность по модулю |a − b|
    "clamp",  # усечение диапазоном
    "median3",  # медиана трёх чисел
}
_VECTOR_INPUT_LIMIT = 1 << 7  # предел модуля листа: 2^7 = 128
_VECTOR_MAX_DEPTH = 3  # наибольшая допустимая глубина операций


def _vector_prefilter_safe(expr: Expr, depth: int = 0) -> bool:
    """Доказать, что значение expr остаётся в пределах int64: True — доказано.

        При глубине ≤ 3 и |лист| ≤ 128 двоичное выражение содержит не более
        8 листьев; даже произведение или НОК всех листьев ≤ 128^8 = 2^56, а суммы
        таких произведений меньше 2^63. Степени, кубы, умножение со сложением,
        произведение трёх и произвольные константы проверяются только скалярно."""
    if depth > _VECTOR_MAX_DEPTH:  # глубина превышает 3
        return False  # не доказано — небезопасно
    if expr.op == "var":  # переменная: её значения ограничены проверкой данных в вызывающей функции
        return True  # безопасно
    if expr.op == "const":  # константа
        return abs(int(expr.value or 0)) <= _VECTOR_INPUT_LIMIT  # |константа| ≤ 128
    if expr.op not in _VECTOR_SAFE_OPS:  # операция вне безопасного набора
        return False  # небезопасно
    return all(_vector_prefilter_safe(a, depth + 1) for a in expr.args)  # все аргументы безопасны при глубине depth + 1


def batch_fits_safe(
    programs: Sequence[Program],  # последовательность программ
    X: Optional[np.ndarray],  # входы или None
    Y: Optional[np.ndarray],  # выходы или None
    device: str = "cpu",  # устройство счёта
) -> list[Optional[bool]]:  # возврат: список True/False/None
    """Консервативный фильтр: None означает, что нужна скалярная проверка.

        programs: список программ; X, Y: входы и выходы (None — данных нет);
        device: устройство счёта. В int64 отправляются только ограниченные
        выражения, остальные получают None и идут мимо быстрого пути, поэтому
        переполнение не может дать ложный отказ. Векторный результат — не
        сертификат: каждый выживший проверяется скалярно на целых Python."""
    if X is None or Y is None:  # данных нет
        return [None] * len(programs)  # всем программам — None
    if X.size and (  # если X не пуст и
        int(X.min()) < -_VECTOR_INPUT_LIMIT or int(X.max()) > _VECTOR_INPUT_LIMIT  # min X < −128 или max X > 128
    ):  # конец условия
        return [None] * len(programs)  # всем программам — None
    out: list[Optional[bool]] = [None] * len(programs)  # результат, изначально все None
    safe_indices = [  # номера программ с выражением, безопасным для int64
        i  # номер программы
        for i, p in enumerate(programs)  # по парам (номер, программа)
        if p.expr is not None and _vector_prefilter_safe(p.expr)  # если выражение есть и оно безопасно
    ]
    if safe_indices:  # список не пуст
        safe_programs = [programs[i] for i in safe_indices]  # отобрать их
        safe_results = _batch_fits_unchecked(safe_programs, X, Y, device=device)  # векторная проверка пакета
        for i, result in zip(safe_indices, safe_results):  # по парам (номер, результат)
            out[i] = result  # записать результат на своё место
    return out  # список True/False/None длины len(programs)

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'batch_fits_safe',  # имя интерфейса
]  # конец перечня
