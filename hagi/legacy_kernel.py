"""Замороженное наследие ядра вычислителя выражений (слои V5, V7, V8 и V9).

    Код перенесён сюда без изменений из старого монолитного ядра и больше не редактируется. Модуль импортируется звёздочкой в hagi.core; совпадающие имена там определяются заново позже, и победает позднейшее определение — так сохраняется исторический порядок переопределений.

    Ключевые идеи:
        * выражение — дерево операций над целыми входами; стоимость = 1 + сумма стоимостей аргументов;
        * поиск программы для задачи — перебор кандидатов от коротких описаний к длинным (принцип минимальной длины описания: число бит = Σ gamma_bits(n) по узлам);
        * эксперт — программа, найденная для задачи и допускающая подстановку в другие программы.

    Содержание по порядку:
        * безопасный язык выражений: таблица операций OPS, класс Expr, класс Program, описания задач TaskSpec;
        * перечисление кандидатов, нейронный слой NeuralFiber, хранилище экспертов, класс SafeExpression и растущий процессор V5;
        * векторный предварительный фильтр (numpy и torch), синтез кандидатов по длине описания, процессор V7;
        * ограниченный оптимизатор слоя, решение задачи раскраски, точная подгонка многочленов и добыча теорем;
        * регулятор ресурсов и бесконечный цикл саморазвития, эксперимент и точка входа V7;
        * выражения V8, коды длин, векторный фильтр и синтезатор V8, процессор V8 и его проверка;
        * заново определённые базовые операции, наборы операций и констант, функции кодирования длин.

    Экспортирует все определённые здесь имена, включая Processors разных версий, Expr, Program, Expert, TaskSpec, OPS, gamma_bits и zigzag."""

from __future__ import annotations  # отложенное вычисление аннотаций типов (модуль __future__)
import argparse, ast, copy, heapq, itertools, json, math, os, random  # импорт стандартных библиотек: разбор аргументов, синтаксические деревья, копирование, куча, декартово произведение, JSON, математика, окружение, случайность
import statistics, sys, time  # статистика, системный модуль, время
from collections import Counter  # Counter — счётчик числа появлений элементов
from dataclasses import dataclass, field  # dataclass и field — классы данных
from fractions import Fraction  # Fraction — точные рациональные дроби
from functools import lru_cache  # lru_cache — запоминание результатов вызовов функций
from pathlib import Path  # Path — путь к файлу
from typing import Any, Callable, Iterable, Optional, Sequence  # типы для аннотаций
import numpy as np  # numpy — численные массивы

VERSION = "5.0.0"  # VERSION — версия базового слоя
V5_VERSION = VERSION  # V5_VERSION — то же значение версии слоя V5


# ------------------- Безопасный язык выражений -------------------

def _truth(v: Any) -> bool:
    """Истинность значения v как логическое значение.

        »    v: число или значение, приводимое к целому. Возврат: bool(int(v)) — истина, если int(v) ≠ 0."""
    return bool(int(v))  # истина, когда int(v) ≠ 0


def _bit_not8(x: int) -> int:
    """Побитовое «не» в одном байте.

        »    x: целое. Возврат: (~x) & 0xFF — инверсия младших 8 бит, старшие обнулены."""
    return (~int(x)) & 0xFF  # не(x) по модулю 256: инверсия младших 8 бит


def _bit_nand8(x: int, y: int) -> int:
    """Побитовый «не-и» в одном байте.

        »    x, y: целые. Возврат: ~(x & y) & 0xFF — инверсия побитового «и», старшие 8 бит обнулены."""
    return (~(int(x) & int(y))) & 0xFF  # не-и по модулю 256: (~(x & y)) & 0xFF


def _bit_nor8(x: int, y: int) -> int:
    """Побитовый «не-или» в одном байте.

        »    x, y: целые. Возврат: ~(x | y) & 0xFF."""
    return (~(int(x) | int(y))) & 0xFF  # не-или по модулю 256: (~(x | y)) & 0xFF


def _bit_xnor8(x: int, y: int) -> int:
    """Побитовая эквивалентность (не-исключающее или) в одном байте.

        »    x, y: целые. Возврат: ~(x ^ y) & 0xFF."""
    return (~(int(x) ^ int(y))) & 0xFF  # исключающее-или с инверсией: (~(x ^ y)) & 0xFF


def _sign(x: int) -> int:
    """Знак числа.

        »    x: целое. Возврат: −1, 0 или +1; формула: (x > 0) − (x < 0)."""
    return (int(x) > 0) - (int(x) < 0)  # знак: (x > 0) − (x < 0), то есть −1, 0 или 1


def _pow_int(x: int, y: int) -> int:
    """Целая степень с проверенным диапазоном показателя.

                x: основание, y: показатель. Показатель обязан лежать в 0..8, иначе ValueError.
        »    Возврат: x ** y."""
    y = int(y)  # показатель y приводится к целому
    if y < 0 or y > 8:  # истина, если y < 0 или y > 8
        raise ValueError("exponent outside certified range")  # ошибка: показатель вне проверенного диапазона
    return int(x) ** y  # степень: x ** y


OPS: dict[str, Callable[..., Any]] = {  # таблица операций по имени: имя -> функция от целых аргументов; значения dict-литерала идут ниже
    "add": lambda x, y: int(x) + int(y),  # add: x + y
    "sub": lambda x, y: int(x) - int(y),  # sub: x − y
    "mul": lambda x, y: int(x) * int(y),  # mul: x·y
    "floordiv": lambda x, y: int(x) // int(y),  # floordiv: x // y (деление с округлением вниз)
    "mod": lambda x, y: int(x) % int(y),  # mod: x mod y (остаток)
    "gcd": lambda x, y: math.gcd(int(x), int(y)),  # gcd: наибольший общий делитель
    "lcm": lambda x, y: math.lcm(int(x), int(y)),  # lcm: наименьшее общее кратное
    "min": lambda x, y: min(int(x), int(y)),  # min: меньшее из x, y
    "max": lambda x, y: max(int(x), int(y)),  # max: большее из x, y
    "eq": lambda x, y: int(x == y),  # eq: 1, если x = y, иначе 0
    "ne": lambda x, y: int(x != y),  # ne: 1, если x ≠ y, иначе 0
    "lt": lambda x, y: int(x < y),  # lt: 1, если x < y, иначе 0
    "le": lambda x, y: int(x <= y),  # le: 1, если x ≤ y, иначе 0
    "gt": lambda x, y: int(x > y),  # gt: 1, если x > y, иначе 0
    "ge": lambda x, y: int(x >= y),  # ge: 1, если x ≥ y, иначе 0
    "bit_and": lambda x, y: int(x) & int(y),  # bit_and: x & y
    "bit_or": lambda x, y: int(x) | int(y),  # bit_or: x | y
    "bit_xor": lambda x, y: int(x) ^ int(y),  # bit_xor: x ^ y
    "bit_nand8": _bit_nand8,  # bit_nand8: ~(x & y) & 0xFF
    "bit_nor8": _bit_nor8,  # bit_nor8: ~(x | y) & 0xFF
    "bit_xnor8": _bit_xnor8,  # bit_xnor8: ~(x ^ y) & 0xFF
    "pow_int": _pow_int,  # pow_int: x ** y, показатель 0..8
    "neg": lambda x: -int(x),  # neg: −x
    "abs": lambda x: abs(int(x)),  # abs: |x|
    "sign": _sign,  # sign: (x > 0) − (x < 0)
    "bit_not8": _bit_not8,  # bit_not8: ~x & 0xFF
    "square": lambda x: int(x) * int(x),  # square: x²
    "cube": lambda x: int(x) * int(x) * int(x),  # cube: x³
}


@dataclass(frozen=True)  # неизменяемый класс данных: после создания поля менять нельзя
class Expr:
    """Дерево выражения: операция op над кортежем аргументов args со вспомогательным значением value.

                op: имя операции ("var", "const", "poly", "poly_nd" или имя из OPS);
                args: подвыражения-аргументы; value: индекс переменной, константа либо коэффициенты многочлена.
        »    eval(xs) вычисляет значение на входах xs; cost — число узлов; pretty — текстовый вид; to_dict/from_dict — сохранение и восстановление."""
    op: str  # имя операции-узла
    args: tuple["Expr", ...] = ()  # аргументы-подвыражения, по умолчанию пусто
    value: Optional[Any] = None  # вспомогательное значение узла, по умолчанию нет

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def var(i: int) -> "Expr":
        """Выражение-переменная: обращение к входу с индексом i.

            »    i: индекс входа (0 — первый аргумент). Возврат: Expr("var", value = int(i))."""
        return Expr("var", (), int(i))  # узел var с индексом int(i)

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def const(v: int) -> "Expr":
        """Выражение-константа.

            »    v: значение. Возврат: Expr("const", value = int(v))."""
        return Expr("const", (), int(v))  # узел const со значением int(v)

    def eval(self, xs: Sequence[int]) -> int:
        """Вычислить выражение на входах xs.

                    xs: последовательность целых входов. Возврат: целое значение.
                    var: xs[value]; const: value; узлы poly, poly_nd, absdiff, fma, clamp, median3,
            »    clamp3, mul3 вычисляются напрямую; остальные — функция OPS[op], применённая к значениям аргументов."""
        if self.op == "var":  # ветка узла-переменной
            assert self.value is not None  # гарантия: индекс переменной задан
            return int(xs[self.value])  # значение входа xs[value]
        if self.op == "const":  # ветка узла-константы
            assert self.value is not None  # гарантия: константа задана
            return int(self.value)  # значение константы
        vals = [a.eval(xs) for a in self.args]  # значения аргументов vals = [a.eval(xs) для каждого аргумента a]
        if self.op == "poly":  # ветка многочлена одной переменной
            coeffs = self.value or ()  # коэффициенты от младшего к старшему; () если нет
            acc = 0  # накопитель схемы Горнера acc = 0
            for ci in reversed(list(coeffs)):  # обход коэффициентов от старшего к младшему
                acc = acc * vals[0] + int(ci)  # шаг Горнера: acc = acc·x + c_i, где x = vals[0]
            return int(acc)  # значение многочлена Σ c_i·x^i
        if self.op == "poly_nd":  # ветка многочлена многих переменных
            total = 0  # сумма членов total = 0
            for coefficient, powers in self.value or ():  # по парам (коэффициент, показатели степеней)
                term = int(coefficient)  # член term = коэффициент
                for x, exponent in zip(xs, powers):  # по парам (вход x, показатель степени)
                    term *= int(x) ** int(exponent)  # член умножается на x^exponent
                total += term  # сумма увеличивается на term
            return int(total)  # сумма членов Σ c_j · Π x_i^{e_ji}
        if self.op == "absdiff":  # ветка модуля разности
            return abs(vals[0] - vals[1])  # модуль разности: |vals[0] − vals[1]|
        if self.op == "fma":  # ветка умножения-сложения
            return vals[0] * vals[1] + vals[2]  # умножение-сложение: vals[0]·vals[1] + vals[2]
        if self.op == "clamp":  # ветка ограничения диапазоном
            return min(max(vals[0], vals[1]), vals[2])  # min(max(v0, v1), v2): не ниже v1 и не выше v2
        if self.op == "median3":  # ветка медианы трёх
            return sorted(vals)[1]  # средний по величине из трёх значений
        if self.op == "clamp3":  # ветка ограничения отрезком [−3, 3]
            return min(max(vals[0], -3), 3)  # ограничение отрезком: min(max(vals[0], −3), 3)
        if self.op == "mul3":  # ветка произведения трёх
            return vals[0] * vals[1] * vals[2]  # произведение трёх: vals[0]·vals[1]·vals[2]
        fn = OPS.get(self.op)  # функция операции из таблицы OPS
        if fn is None:  # истина, если операции нет в таблице
            raise ValueError(f"unsupported expression opcode: {self.op}")  # ошибка: неподдерживаемая операция
        return int(fn(*vals))  # результат операции над значениями аргументов

    @property  # свойство: вычисляется при обращении как атрибут
    def cost(self) -> int:
        """Стоимость выражения: число узлов дерева.

            »    cost = 1 для var и const, иначе 1 + Σ cost(аргумента). Результат запоминается в поле _cost_memo."""
        c = self.__dict__.get("_cost_memo")  # взять запомненное значение, если уже считалось
        if c is None:  # истина, если ещё не считалось
            if self.op in ("var", "const"):  # лист — переменная или константа
                c = 1  # стоимость листа 1
            else:  # иначе — внутренний узел
                c = 1 + sum(a.cost for a in self.args)  # стоимость 1 + сумма стоимостей аргументов
            object.__setattr__(self, "_cost_memo", c)  # запомнить стоимость (обход запрета замороженного класса)
        return c  # вернуть стоимость

    def pretty(self) -> str:
        """Читаемая запись выражения в текстовом виде.

                    Возврат: строка, например "x**2 + 3" или "poly_nd(x*y - 2*z)".
            »    Многочлены печатаются от старшей степени к младшей с опусканием нулевых коэффициентов."""
        if self.op == "var":  # ветка переменной
            return ("x", "y", "z")[self.value or 0]  # имя по индексу: x, y или z
        if self.op == "const":  # ветка константы
            return str(self.value)  # запись числа
        if self.op == "poly":  # ветка многочлена одной переменной
            coeffs = tuple(self.value or ())  # кортеж коэффициентов c_0..c_n
            terms = []  # список кусков текста
            for i in range(len(coeffs) - 1, -1, -1):  # степень i от n down to 0
                c = int(coeffs[i])  # коэффициент c_i
                if c == 0:  # нулевой коэффициент пропускается
                    continue  # переход к следующему i
                mag = abs(c)  # модуль коэффициента
                if i == 0:  # свободный член (i = 0)
                    term = str(mag)  # текст свободного члена
                else:  # иначе — степень i ≥ 1
                    body = "x" if i == 1 else f"x**{i}"  # "x" при i = 1, иначе "x**i"
                    term = body if mag == 1 else f"{mag}*{body}"  # "x**i", а при |c| ≠ 1 — "|c|*x**i"
                if not terms:  # первый член
                    terms.append(("-" if c < 0 else "") + term)  # минус перед первым отрицательным членом
                else:  # последующие члены
                    terms.append((" - " if c < 0 else " + ") + term)  # " + " либо " − " перед членом
            return "poly(" + ("".join(terms) if terms else "0") + ")"  # "poly(" + члены + ")", "0" если членов нет
        if self.op == "poly_nd":  # ветка многочлена многих переменных
            symbols = ("x", "y", "z", "u", "v", "w")  # имена переменных по индексу
            terms = []  # список кусков текста
            for coefficient, powers in sorted(  # члены от большей суммарной степени к меньшей
                (self.value or ()),  # сами пары (коэффициент, показатели)
                key=lambda item: (sum(item[1]), item[1]),  # ключ сортировки: (сумма показателей, кортеж показателей)
                reverse=True,  # по убыванию ключа
            ):  # конец сортировки
                c = int(coefficient)  # коэффициент члена
                if c == 0:  # нулевой коэффициент пропускается
                    continue  # переход к следующему члену
                factors = []  # множители члена
                for index, exponent in enumerate(powers):  # по парам (индекс переменной, показатель)
                    exponent = int(exponent)  # показатель степени как целое
                    if exponent == 0:  # показатель 0 — переменная не входит
                        continue  # пропустить её
                    name = symbols[index] if index < len(symbols) else f"x{index}"  # имя переменной; для индексов свыше списка — "x{индекс}"
                    factors.append(name if exponent == 1 else f"{name}**{exponent}")  # "имя", если степень 1, иначе "имя**степень"
                body = "*".join(factors) if factors else str(abs(c))  # тело члена: произведение множителей; без множителей — |c|
                if factors and abs(c) != 1:  # истина, если есть множители и |c| ≠ 1
                    body = f"{abs(c)}*{body}"  # тело = "|c|*" + тело
                if not terms:  # первый член
                    terms.append(("-" if c < 0 else "") + body)  # минус перед первым отрицательным членом
                else:  # последующие члены
                    terms.append((" - " if c < 0 else " + ") + body)  # " + " либо " − " перед членом
            return "poly_nd(" + ("".join(terms) if terms else "0") + ")"  # "poly_nd(" + члены + ")", "0" если членов нет
        if self.op == "clamp3":  # ветка ограничения отрезком [−3, 3]
            return f"min(max({self.args[0].pretty()}, -3), 3)"  # текст "min(max(аргумент, -3), 3)"
        if self.op == "absdiff":  # ветка модуля разности
            return f"abs({self.args[0].pretty()} - {self.args[1].pretty()})"  # текст "abs(арг0 − арг1)"
        if self.op == "fma":  # ветка умножения-сложения
            return f"({self.args[0].pretty()} * {self.args[1].pretty()} + {self.args[2].pretty()})"  # текст "(арг0 * арг1 + арг2)"
        if self.op == "clamp":  # ветка ограничения диапазоном
            return f"min(max({self.args[0].pretty()}, {self.args[1].pretty()}), {self.args[2].pretty()})"  # текст "min(max(арг0, арг1), арг2)"
        if self.op == "median3":  # ветка медианы трёх
            return f"median({', '.join(a.pretty() for a in self.args)})"  # текст "median(арг0, арг1, арг2)"
        if self.op == "mul3":  # ветка произведения трёх
            return " * ".join(a.pretty() for a in self.args)  # аргументы, соединённые " * "
        if len(self.args) == 1:  # один аргумент
            return f"{self.op}({self.args[0].pretty()})"  # текст "операция(аргумент)"
        if len(self.args) == 2:  # два аргумента
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"  # текст "(арг0 операция арг1)"
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"  # прочее: "операция(арг0, арг1, ...)"

    def to_dict(self) -> dict[str, Any]:
        """Сохранить выражение в словарь для записи в JSON.

            »    Возврат: {"op": имя, "value": значение, "args": [словари аргументов]}."""
        return {  # начало словаря
            "op": self.op,  # имя операции
            "value": self.value,  # вспомогательное значение
            "args": [a.to_dict() for a in self.args],  # аргументы, сохранённые тем же способом
        }

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def from_dict(d: dict[str, Any]) -> "Expr":
        """Восстановить выражение из словаря.

                    d: словарь вида to_dict(). После записи в JSON кортежи превращаются в списки;
                    для poly и poly_nd они приводятся обратно к кортежам, чтобы выражение оставалось хешируемым и равенство не менялось.
            »    Возврат: Expr."""
        op = str(d["op"])  # имя операции
        value = d.get("value")  # вспомогательное значение или None
        if op == "poly" and value is not None:  # многочлен одной переменной
            value = tuple(int(v) for v in value)  # value = кортеж целых коэффициентов
        # JSON превращает неизменяемые кортежи в списки. Восстанавливаем исходную форму значения,
        # чтобы после перечитывания выражение оставалось хешируемым и равенство не менялось.
        elif op == "poly_nd" and value is not None:  # многочлен многих переменных
            value = tuple(  # value = кортеж пар (коэффициент, кортеж показателей)
                (int(term[0]), tuple(int(v) for v in term[1])) for term in value  # пара: (int(коэффициент), кортеж int показателей)
            )  # конец генератора
        return Expr(op, tuple(Expr.from_dict(a) for a in d.get("args", [])), value)  # собрать узел: аргументы восстановлены рекурсивно


@dataclass(frozen=True)  # неизменяемый класс данных
class Program:
    """Программа-эксперт: выражение либо булева таблица истинности.

                arity: число входов; expr: выражение Expr; venn_mask: 4-битная маска таблицы истинности двух булевых входов.
        »    Если venn_mask задана, значение = бит маски с номером (x << 1) | y; иначе значение выражения expr."""
    arity: int  # число входов
    expr: Optional[Expr] = None  # выражение программы, по умолчанию нет
    venn_mask: Optional[int] = None  # 4-битная маска таблицы истинности, по умолчанию нет

    @property  # свойство: вычисляется при обращении как атрибут
    def cost(self) -> int:
        """Стоимость программы в узлах.

            »    4 для булевой маски; иначе стоимость выражения; 10^9, если выражения нет."""
        return (  # начало условного выражения
            4  # стоимость булевой маски — 4
            if self.venn_mask is not None  # если маска задана
            else (self.expr.cost if self.expr else 10**9)  # иначе cost(expr), а без выражения 10^9
        )

    def eval(self, xs: Sequence[int]) -> int:
        """Вычислить программу на входах xs.

                    xs: последовательность целых длины arity. Число входов обязано совпадать с arity.
                    Булева маска требует ровно двух входов из {0, 1}: значение = бит маски с номером 2·x + y.
            »    Иначе значение = expr.eval(xs); если выражения нет — RuntimeError."""
        if len(xs) != self.arity:  # проверка числа входов: len(xs) = arity
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")  # ошибка: число входов не совпадает
        if self.venn_mask is not None:  # ветка булевой маски
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):  # маска требует ровно двух входов, каждый 0 или 1
                raise ValueError("BoolVenn expert accepts two Boolean inputs")  # ошибка: булев эксперт принимает два булевых входа
            idx = (int(xs[0]) << 1) | int(xs[1])  # номер бита idx = 2·x + y
            return (int(self.venn_mask) >> idx) & 1  # значение = (маска >> idx) & 1
        if self.expr is None:  # ветка отсутствия выражения
            raise RuntimeError("program has no expression")  # ошибка: у программы нет выражения
        return self.expr.eval(xs)  # значение выражения

    def pretty(self) -> str:
        """Читаемая запись программы.

            »    Возврат: "BoolVenn(mask=....)" — 4 бита маски, либо текст выражения."""
        if self.venn_mask is not None:  # ветка булевой маски
            return f"BoolVenn(mask={self.venn_mask:04b})"  # текст с маской в 4-битной двоичной записи
        assert self.expr is not None  # гарантия: выражение задано
        return self.expr.pretty()  # текст выражения

    def to_dict(self) -> dict[str, Any]:
        """Сохранить программу в словарь для записи в JSON.

            »    Возврат: {"arity": ..., "venn_mask": ..., "expr": словарь выражения или None}."""
        return {  # начало словаря
            "arity": self.arity,  # число входов
            "venn_mask": self.venn_mask,  # маска таблицы истинности
            "expr": self.expr.to_dict() if self.expr else None,  # словарь выражения, если оно есть
        }

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def from_dict(d: dict[str, Any]) -> "Program":
        """Восстановить программу из словаря.

            »    d: словарь вида to_dict(). Возврат: Program."""
        return Program(  # собрать программу
            int(d["arity"]),  # число входов
            Expr.from_dict(d["expr"]) if d.get("expr") else None,  # выражение, если в словаре оно есть
            d.get("venn_mask"),  # маска таблицы истинности
        )



# ------------------------------- Описания задач ------------------------------

@dataclass  # изменяемый класс данных
class TaskSpec:
    """Описание задачи: имя, входы, обучающая функция и правило отсева входов.

                name: имя задачи; arity: число входов; domains: кортеж допустимых значений каждого входа;
                teacher: функция-учитель, дающая правильный ответ; valid_rule: "all" — все входы допустимы,
                "nonzero_y" — второй вход не равен 0, "ordered_yz" — второй вход не больше третьего;
        »    family: семейство задач. inputs() даёт все допустимые входы, valid(xs) проверяет один вход."""
    name: str  # имя задачи
    arity: int  # число входов
    domains: tuple[tuple[int, ...], ...]  # допустимые значения каждого входа
    teacher: Callable[..., int]  # функция-учитель: правильный ответ
    valid_rule: str = "all"  # правило отсева входов, по умолчанию все допустимы
    family: str = "integer"  # семейство задач

    def inputs(self) -> list[tuple[int, ...]]:
        """Все допустимые входы задачи.

            »    Возврат: список кортежей — декартово произведение областей без строк, отброшенных правилом valid_rule."""
        rows = itertools.product(*self.domains)  # декартово произведение областей
        out = []  # накопитель результата
        for row in rows:  # по строкам произведения
            if self.valid_rule == "nonzero_y" and row[1] == 0:  # правило nonzero_y и второй вход равен 0
                continue  # строка отброшена
            if self.valid_rule == "ordered_yz" and row[1] > row[2]:  # правило ordered_yz и второй вход больше третьего
                continue  # строка отброшена
            out.append(tuple(int(x) for x in row))  # добавить строку целых значений
        return out  # вернуть список входов

    def valid(self, xs: Sequence[int]) -> bool:
        """Проверить, допустим ли вход xs для задачи.

            »    xs: последовательность целых. Возврат: истина, если длина равна arity, каждое значение лежит в своей области и выполнено правило valid_rule."""
        if len(xs) != self.arity:  # длина обязана совпадать с arity
            return False  # ложь при несовпадении длины
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):  # истина, если хотя бы одно значение вне своей области
            return False  # ложь, если есть значение вне области
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:  # правило nonzero_y и xs[1] = 0
            return False  # ложь
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):  # правило ordered_yz и xs[1] > xs[2]
            return False  # ложь
        return True  # иначе вход допустим


BOOL_NAMES = {  # таблица истинности 16 булевых функций двух переменных: маска -> имя; в маске бит 2·x + y равен значению функции
    0: "BOOL_FALSE",  # маска 0: всегда 0
    1: "BOOL_NOR",  # маска 1: nor — не(x или y)
    2: "BOOL_NOT_X_AND_Y",  # маска 2: не x и y
    3: "BOOL_NOT_X",  # маска 3: не x
    4: "BOOL_X_AND_NOT_Y",  # маска 4: x и не y
    5: "BOOL_NOT_Y",  # маска 5: не y
    6: "BOOL_XOR",  # маска 6: xor — исключающее или
    7: "BOOL_NAND",  # маска 7: nand — не(x и y)
    8: "BOOL_AND",  # маска 8: and — x и y
    9: "BOOL_XNOR",  # маска 9: xnor — эквивалентность
    10: "BOOL_Y",  # маска 10: y
    11: "BOOL_X_IMPLIES_Y",  # маска 11: x влечёт y
    12: "BOOL_X",  # маска 12: x
    13: "BOOL_Y_IMPLIES_X",  # маска 13: y влечёт x
    14: "BOOL_OR",  # маска 14: or — x или y
    15: "BOOL_TRUE",  # маска 15: всегда 1
}


def make_task_suite() -> list[TaskSpec]:
    """Построить полный набор задач.

        »    Возврат: список TaskSpec — 16 булевых, 7 унарных, 22 бинарные, степень, fma, clamp и далее составные задачи."""
    suite: list[TaskSpec] = []  # накопитель задач
    bool_domain = ((0, 1), (0, 1))  # область двух булевых входов {0, 1}
    for mask, name in BOOL_NAMES.items():  # по маскам и именам булевых функций
        suite.append(  # добавить задачу
            TaskSpec(  # начало описания задачи
                name,  # имя функции
                2,  # два входа
                bool_domain,  # области {0, 1} × {0, 1}
                lambda x, y, m=mask: (m >> ((int(x) << 1) | int(y))) & 1,  # ответ = бит маски с номером 2·x + y; m фиксирован для данной итерации
                family="BoolVenn",  # семейство BoolVenn
            )
        )

    U = tuple(range(-8, 9))  # U — область унарных и бинарных задач: целые от −8 до 8
    unary = [  # унарные функции: имя и функция
        ("NEG", lambda x: -x),  # NEG: −x
        ("ABS", abs),  # ABS: |x|
        ("SIGN", _sign),  # SIGN: знак
        ("SQUARE", lambda x: x * x),  # SQUARE: x²
        ("CUBE", lambda x: x * x * x),  # CUBE: x³
        ("BIT_NOT8", _bit_not8),  # BIT_NOT8: ~x & 0xFF
        ("CLAMP_TO_3", lambda x: min(max(x, -3), 3)),  # CLAMP_TO_3: min(max(x, −3), 3)
    ]  # конец списка унарных функций
    for name, fn in unary:  # по унарным парам (имя, функция)
        suite.append(TaskSpec(name, 1, (U,), fn, family="unary"))  # задача с одним входом из U

    binary = [  # бинарные функции: имя, функция, правило отсева
        ("ADD", lambda x, y: x + y, "all"),  # ADD: x + y
        ("SUB", lambda x, y: x - y, "all"),  # SUB: x − y
        ("MUL", lambda x, y: x * y, "all"),  # MUL: x·y
        ("FLOORDIV", lambda x, y: x // y, "nonzero_y"),  # FLOORDIV: x // y, y ≠ 0
        ("MOD", lambda x, y: x % y, "nonzero_y"),  # MOD: x mod y, y ≠ 0
        ("GCD", math.gcd, "all"),  # GCD: наибольший общий делитель
        ("LCM", math.lcm, "all"),  # LCM: наименьшее общее кратное
        ("MIN", min, "all"),  # MIN: меньшее
        ("MAX", max, "all"),  # MAX: большее из x, y
        ("ABSDIFF", lambda x, y: abs(x - y), "all"),  # ABSDIFF: |x − y|
        ("EQ", lambda x, y: int(x == y), "all"),  # EQ: 1, если x = y
        ("NE", lambda x, y: int(x != y), "all"),  # NE: 1, если x ≠ y
        ("LT", lambda x, y: int(x < y), "all"),  # LT: 1, if x < y
        ("LE", lambda x, y: int(x <= y), "all"),  # LE: 1, если x ≤ y
        ("GT", lambda x, y: int(x > y), "all"),  # GT: 1, если x > y
        ("GE", lambda x, y: int(x >= y), "all"),  # GE: 1, если x ≥ y
        ("BIT_AND", lambda x, y: x & y, "all"),  # BIT_AND: x & y
        ("BIT_OR", lambda x, y: x | y, "all"),  # BIT_OR: x | y
        ("BIT_XOR", lambda x, y: x ^ y, "all"),  # BIT_XOR: x ^ y
        ("BIT_NAND8", _bit_nand8, "all"),  # BIT_NAND8: ~(x & y) & 0xFF
        ("BIT_NOR8", _bit_nor8, "all"),  # BIT_NOR8: ~(x | y) & 0xFF
        ("BIT_XNOR8", _bit_xnor8, "all"),  # BIT_XNOR8: ~(x ^ y) & 0xFF
    ]  # конец списка бинарных функций
    for name, fn, valid in binary:  # по тройкам (имя, функция, правило)
        suite.append(  # добавить задачу
            TaskSpec(name, 2, (U, U), fn, valid_rule=valid, family="integer-binary")  # два входа из U, правило отсева и семейство integer-binary
        )
    P = tuple(range(-3, 4))  # P — область основания степени: целые от −3 до 3
    E = tuple(range(0, 5))  # E — область показателя степени: целые от 0 до 4
    suite.append(TaskSpec("POW_INT", 2, (P, E), _pow_int, family="integer-binary"))  # задача POW_INT с областями P и E

    Z = tuple(range(-8, 9))  # Z — область: целые от −8 до 8
    F = tuple(range(-4, 5))  # F — область границ clamp: целые от −4 до 4
    suite.extend(  # добавить список составных задач
        [  # начало списка
            TaskSpec(  # задача
                "FMA",  # имя FMA
                3,  # три входа
                (Z, Z, Z),  # три области Z
                lambda x, y, z: x * y + z,  # ответ x·y + z
                family="composed-ternary",  # семейство составных тернарных задач
            ),
            TaskSpec(  # начало описания задачи
                "CLAMP",  # имя CLAMP
                3,  # три входа
                (Z, F, F),  # x из Z, границы lo и hi из F
                lambda x, lo, hi: min(max(x, lo), hi),  # ответ min(max(x, lo), hi)
                valid_rule="ordered_yz",  # правило: lo ≤ hi
                family="composed-ternary",  # семейство составных тернарных задач
            ),
            TaskSpec(  # начало описания задачи
                "MEDIAN3",  # имя MEDIAN3
                3,  # три входа
                (F, F, F),  # три области F
                lambda x, y, z: sorted((x, y, z))[1],  # ответ — среднее по величине из x, y, z: отсортировать и взять элемент с индексом 1
                family="composed-ternary",  # семейство составных тернарных задач
            ),
        ]  # конец списка
    )
    return suite  # вернуть набор задач



# ------------------------- Перечисление программ-кандидатов ------------------------

def _dedupe_candidates(candidates: Iterable[Expr]) -> list[Expr]:
    """Убрать повторяющиеся выражения и отсортировать по стоимости.

                candidates: перебор выражений. Два выражения считаются одинаковыми, если совпадает их текстовый вид pretty().
        »    Возврат: список уникальных выражений, отсортированный по (cost, pretty())."""
    by_text: dict[str, Expr] = {}  # словарь «текст выражения -> выражение»
    for e in candidates:  # по всем выражениям-кандидатам
        by_text.setdefault(e.pretty(), e)  # записать выражение, если такой текст ещё не встречался
    return sorted(by_text.values(), key=lambda e: (e.cost, e.pretty()))  # уникальные значения, отсортированные по (стоимость, текст)


def candidate_programs(arity: int) -> list[Program]:
    """Построить программы-кандидаты для заданного числа входов.

                arity: число входов. Для 1, 2 и 3 входов перечисляются выражения из фиксированных списков
                (унарные операции, бинарные операции над (x, y), тернарные примитивы), дубликаты убираются.
        »    Возврат: список Program; для прочих arity — пустой список."""
    v = [Expr.var(i) for i in range(arity)]  # переменные-листы: x_i = Expr.var(i), i = 0..arity−1
    cands: list[Expr] = []  # накопитель выражений-кандидатов
    if arity == 1:  # ветка одного входа
        x = v[0]  # x — единственный вход
        cands += [  # начать список унарных кандидатов
            Expr("neg", (x,)),  # −x
            Expr("abs", (x,)),  # |x|
            Expr("sign", (x,)),  # знак x
            Expr("bit_not8", (x,)),  # побитовое «не» в байте
            Expr("mul", (x, x)),  # x²
            Expr("mul", (Expr("mul", (x, x)), x)),  # x³
            Expr("clamp3", (x,)),  # ограничение отрезком [−3, 3]
        ]  # конец списка
        return [Program(arity, e) for e in _dedupe_candidates(cands)]  # программы из уникальных кандидатов
    if arity == 2:  # ветка двух входов
        x, y = v  # x, y — входы
        for op in (  # по именам бинарных операций
            "add",  # сложение
            "sub",  # вычитание
            "mul",  # умножение
            "floordiv",  # деление с округлением вниз
            "mod",  # остаток
            "gcd",  # наибольший общий делитель
            "lcm",  # наименьшее общее кратное
            "min",  # минимум
            "max",  # максимум
            "eq",  # равно
            "ne",  # не равно
            "lt",  # меньше
            "le",  # меньше или равно
            "gt",  # больше
            "ge",  # больше или равно
            "bit_and",  # побитовое «и»
            "bit_or",  # побитовое «или»
            "bit_xor",  # исключающее «или»
            "bit_nand8",  # не-и в байте
            "bit_nor8",  # не-или в байте
            "bit_xnor8",  # эквивалентность в байте
            "pow_int",  # целая степень
        ):  # конец кортежа операций
            cands.append(Expr(op, (x, y)))  # кандидат op(x, y)
        cands.append(Expr("sub", (y, x)))  # кандидат y − x (порядок аргументов изменён)
        cands.append(Expr("absdiff", (x, y)))  # кандидат |x − y|
        cands.append(Expr("abs", (Expr("sub", (x, y)),)))  # кандидат |x − y| через abs(sub)
        cands.append(Expr("abs", (Expr("sub", (y, x)),)))  # кандидат |y − x| через abs(sub)
        atoms = [  # атомы — базовые выражения двух входов
        # Небольшая составная грамматика: двухслойные выражения из арифметических атомов.
            Expr("add", (x, y)),  # x + y
            Expr("sub", (x, y)),  # x − y
            Expr("sub", (y, x)),  # y − x
            Expr("mul", (x, y)),  # x·y
            Expr("abs", (x,)),  # |x|
            Expr("abs", (y,)),  # |y|
        ]  # конец списка атомов
        for a in atoms:  # по атомам
            cands.append(Expr("abs", (a,)))  # кандидат |атом|
        return [Program(arity, e) for e in _dedupe_candidates(cands)]  # программы из уникальных кандидатов
    if arity == 3:  # ветка трёх входов
        x, y, z = v  # x, y, z — входы
        cands += [  # начать список тернарных кандидатов
            Expr("fma", (x, y, z)),  # x·y + z
            Expr("clamp", (x, y, z)),  # min(max(x, y), z)
            Expr("median3", (x, y, z)),  # медиана трёх
            Expr("mul3", (x, y, z)),  # x·y·z
            Expr("add", (Expr("mul", (x, y)), z)),  # (x·y) + z
            Expr("max", (Expr("min", (x, y)), Expr("min", (Expr("max", (x, y)), z)))),  # max(min(x, y), min(max(x, y), z))
        ]  # конец списка
        return [Program(arity, e) for e in _dedupe_candidates(cands)]  # программы из уникальных кандидатов
    return []  # прочее число входов — кандидатов нет



# ------------------------------- Нейронный слой ----------------------------------

@dataclass  # изменяемый класс данных
class NeuralFiber:
    """Нейронная сеть из одного скрытого слоя, приближающая целочисленную функцию.

                Схема: h = tanh(x·w1 + b1); pred = h·w2 + b2. Число входов arity, скрытый слой hidden.
                input_scale и target_scale — нормирующие множители: обучение идёт на x/sx и y/sy.
        »    train_loss, validation_loss — средний квадрат ошибки; validation_exact_accuracy — доля точных попаданий после округления."""
    arity: int  # число входов
    input_scale: float  # множитель нормировки входов sx
    target_scale: float  # множитель нормировки ответов sy
    w1: np.ndarray  # веса первого слоя w1
    b1: np.ndarray  # смещения первого слоя b1
    w2: np.ndarray  # веса второго слоя w2
    b2: np.ndarray  # смещение второго слоя b2
    train_loss: float = float("nan")  # ошибка на обучающих примерах, по умолчанию не число
    validation_loss: float = float("nan")  # ошибка на проверочных примерах, по умолчанию не число
    validation_exact_accuracy: float = float("nan")  # доля точных попаданий на проверке, по умолчанию не число

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def fit(
        inputs: Sequence[Sequence[int]],  # inputs — обучающие входы
        targets: Sequence[int],  # targets — обучающие ответы
        val_inputs: Sequence[Sequence[int]],  # val_inputs — проверочные входы
        val_targets: Sequence[int],  # val_targets — проверочные ответы
        seed: int,  # зерно случайности
        hidden: int = 24,  # ширина скрытого слоя, по умолчанию 24
        epochs: int = 140,  # число эпох, по умолчанию 140
        lr: float = 0.025,  # скорость обучения, по умолчанию 0.025
    ) -> "NeuralFiber":
        """Обучить нейронный слой методом Адамса и вернуть его.

                    inputs, targets: обучающие примеры и ответы; val_inputs, val_targets: проверочные.
                    seed: зерно генератора; hidden: ширина скрытого слоя; epochs: число проходов; lr: скорость обучения.
                    Входы нормируются: x/sx, y/sy, где sx = max(1, max|x|), sy = max(1, max|y|).
                    Прямой ход: h = tanh(x·w1 + b1), pred = h·w2 + b2; ошибка err = pred − y; средний квадрат loss = mean(err²).
                    Обратный ход: dp = (2/N)·err; dw2 = hᵀ·dp; dh = (dp·w2ᵀ)·(1 − h²); dw1 = xᵀ·dh.
                    Обновление Адамса: m = β1·m + (1−β1)·g; v = β2·v + (1−β2)·g²; p −= lr·(m/(1−β1^t))/(√(v/(1−β2^t)) + eps).
            »    Сохраняются веса с наименьшей проверочной ошибкой. Возврат: NeuralFiber."""
        x = np.asarray(inputs, dtype=np.float64).reshape(-1, len(inputs[0]))  # x — матрица входов (N, число входов) в float64
        y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)  # y — столбец ответов (N, 1)
        xv = np.asarray(val_inputs, dtype=np.float64).reshape(-1, x.shape[1])  # xv — проверочные входы той же ширины
        yv = np.asarray(val_targets, dtype=np.float64).reshape(-1, 1)  # yv — проверочные ответы, столбец
        sx = max(1.0, float(np.max(np.abs(x))))  # масштаб входов sx = max(1, наибольший |x|)
        sy = max(1.0, float(np.max(np.abs(y))))  # масштаб ответов sy = max(1, наибольший |y|)
        x, y, xv, yv = x / sx, y / sy, xv / sx, yv / sy  # нормировка: x/sx, y/sy, xv/sx, yv/sy
        rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
        w1 = rng.normal(0.0, 0.32, (x.shape[1], hidden))  # w1 — нормальные веса (число входов, hidden) с дисперсией 0.32
        b1 = np.zeros((1, hidden), dtype=np.float64)  # b1 — нулевые смещения (1, hidden)
        w2 = rng.normal(0.0, 0.12, (hidden, 1))  # w2 — нормальные веса (hidden, 1) с дисперсией 0.12
        b2 = np.zeros((1, 1), dtype=np.float64)  # b2 — нулевое смещение (1, 1)
        params = [w1, b1, w2, b2]  # список параметров [w1, b1, w2, b2]
        m = [np.zeros_like(p) for p in params]  # m — первые моменты Адамса, нули
        v = [np.zeros_like(p) for p in params]  # v — вторые моменты Адамса, нули
        beta1, beta2, eps = 0.9, 0.999, 1e-8  # коэффициенты β1 = 0.9, β2 = 0.999, eps = 10⁻⁸
        best = [p.copy() for p in params]  # лучшие веса — копии исходных
        best_val = float("inf")  # лучшая проверочная ошибка — бесконечность
        step = 0  # номер шага t = 0
        for _ in range(epochs):  # по эпохам
            h = np.tanh(x @ w1 + b1)  # скрытый слой h = tanh(x·w1 + b1)
            pred = h @ w2 + b2  # прогноз pred = h·w2 + b2
            err = pred - y  # ошибка err = pred − y
            loss = float(np.mean(err * err))  # средний квадрат ошибки loss = mean(err²)
            dp = (2.0 / len(x)) * err  # градиент по pred: dp = (2/N)·err
            dw2 = h.T @ dp  # градиент dw2 = hᵀ·dp
            db2 = np.sum(dp, axis=0, keepdims=True)  # градиент db2 = сумма dp по строкам
            dh = (dp @ w2.T) * (1.0 - h * h)  # градиент по h: dh = (dp·w2ᵀ)·(1 − h²) (производная tanh)
            dw1 = x.T @ dh  # градиент dw1 = xᵀ·dh
            db1 = np.sum(dh, axis=0, keepdims=True)  # градиент db1 = сумма dh по строкам
            grads = [dw1, db1, dw2, db2]  # список градиентов в порядке параметров
            step += 1  # шаг t = t + 1
            for i, (p, g) in enumerate(zip(params, grads)):  # по парам (параметр, градиент)
                m[i] = beta1 * m[i] + (1 - beta1) * g  # первый момент m = β1·m + (1−β1)·g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)  # второй момент v = β2·v + (1−β2)·g²
                mh = m[i] / (1 - beta1**step)  # исправленный первый момент mh = m/(1 − β1^t)
                vh = v[i] / (1 - beta2**step)  # исправленный второй момент vh = v/(1 − β2^t)
                p -= lr * mh / (np.sqrt(vh) + eps)  # шаг параметра: p −= lr·mh/(√vh + eps)
            hv = np.tanh(xv @ w1 + b1)  # скрытый слой проверочных входов hv = tanh(xv·w1 + b1)
            pv = hv @ w2 + b2  # проверочный прогноз pv = hv·w2 + b2
            vl = float(np.mean((pv - yv) ** 2))  # проверочная ошибка vl = mean((pv − yv)²)
            if np.isfinite(vl) and vl < best_val:  # ошибка конечна и меньше рекордной
                best_val = vl  # новый рекорд
                best = [p.copy() for p in params]  # запомнить лучшие веса
        w1, b1, w2, b2 = best  # взять лучшие веса
        h = np.tanh(x @ w1 + b1)  # скрытый слой обучающих входов
        tr_loss = float(np.mean((h @ w2 + b2 - y) ** 2)) * sy * sy  # ошибка обучения в исходном масштабе: mean((h·w2 + b2 − y)²)·sy²
        hv = np.tanh(xv @ w1 + b1)  # скрытый слой проверочных входов
        predv = (hv @ w2 + b2).reshape(-1) * sy  # проверочный прогноз в исходном масштабе: predv·sy
        val_loss = float(np.mean((predv - np.asarray(val_targets, dtype=float)) ** 2))  # проверочная ошибка val_loss = mean((predv − val_targets)²)
        exact = float(  # доля точных попаданий (начало выражения)
            np.mean(  # начало mean
                np.rint(predv).astype(np.int64)  # округлённый прогноз как целое
                == np.asarray(val_targets, dtype=np.int64)  # сравнение с целым ответом; доля совпадений
            )
        )  # конец mean
        return NeuralFiber(  # создать слой с найденными параметрами
            len(inputs[0]), sx, sy, w1, b1, w2, b2, tr_loss, val_loss, exact  # arity, sx, sy, w1, b1, w2, b2, ошибка обучения, ошибка проверки, точность
        )

    def predict(self, xs: Sequence[int]) -> float:
        """Прогноз слоя для одного входа.

            »    xs: вход длины arity. Возврат: (tanh((x/sx)·w1 + b1)·w2 + b2)·sy — число в исходном масштабе."""
        if len(xs) != self.arity:  # проверка числа входов
            raise ValueError("wrong neural fiber arity")  # ошибка: неверное число входов
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1) / self.input_scale  # x — строка (1, arity) из xs, делённая на input_scale
        return float(  # возврат числа (начало выражения)
            (np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)[0, 0]  # прогноз сети tanh(x·w1 + b1)·w2 + b2
            * self.target_scale  # умножение на target_scale
        )

    def to_dict(self) -> dict[str, Any]:
        """Сохранить слой в словарь для записи в JSON.

            »    Возврат: словарь с arity, масштабами, весами (как вложенные списки) и метриками качества."""
        return {  # начало словаря
            "arity": self.arity,  # число входов
            "input_scale": self.input_scale,  # множитель входов
            "target_scale": self.target_scale,  # множитель ответов
            "w1": self.w1.tolist(),  # w1 как список
            "b1": self.b1.tolist(),  # b1 как список
            "w2": self.w2.tolist(),  # w2 как список
            "b2": self.b2.tolist(),  # b2 как список
            "train_loss": self.train_loss,  # ошибка обучения
            "validation_loss": self.validation_loss,  # ошибка проверки
            "validation_exact_accuracy": self.validation_exact_accuracy,  # доля точных попаданий
        }

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def from_dict(d: dict[str, Any]) -> "NeuralFiber":
        """Восстановить слой из словаря.

            »    d: словарь вида to_dict(). Возврат: NeuralFiber."""
        return NeuralFiber(  # создать слой
            int(d["arity"]),  # число входов
            float(d["input_scale"]),  # множитель входов
            float(d["target_scale"]),  # множитель ответов
            np.asarray(d["w1"], dtype=np.float64),  # w1
            np.asarray(d["b1"], dtype=np.float64),  # b1
            np.asarray(d["w2"], dtype=np.float64),  # w2
            np.asarray(d["b2"], dtype=np.float64),  # b2
            float(d.get("train_loss", float("nan"))),  # ошибка обучения (nan, если ключа нет)
            float(d.get("validation_loss", float("nan"))),  # ошибка проверки (nan, если ключа нет)
            float(d.get("validation_exact_accuracy", float("nan"))),  # точность (nan, если ключа нет)
        )



# --------------------------------- Хранилище экспертов ---------------------------------

@dataclass  # изменяемый класс данных
class Expert:
    """Эксперт: найденная программа для задачи и область её применимости.

                name: имя; family: семейство задач; program: программа; domains: допустимые значения каждого входа;
                valid_rule: правило отсева входов; code_length: длина описания в битах; neural: необязательная нейросеть;
        »    exhaustive_checks, exhaustive_errors: число проверок и ошибок при полной проверке; parents: имена экспертов-родителей."""
    name: str  # имя эксперта
    family: str  # семейство задач
    program: Program  # программа эксперта
    domains: tuple[tuple[int, ...], ...]  # допустимые значения входов
    valid_rule: str  # правило отсева входов
    code_length: int  # длина описания в битах
    neural: Optional[NeuralFiber] = None  # нейросеть-приближение, по умолчанию нет
    exhaustive_checks: int = 0  # число выполненных проверок
    exhaustive_errors: int = 0  # число найденных ошибок
    parents: tuple[str, ...] = ()  # имена экспертов-родителей, по умолчанию пусто

    def supports(self, xs: Sequence[int]) -> bool:
        """Проверить, применим ли эксперт к входу xs.

                    xs: вход. Возврат: истина, если длина совпадает с числом областей, все значения в своих областях
            »    и выполнено правило valid_rule ("nonzero_y": xs[1] ≠ 0; "ordered_yz": xs[1] ≤ xs[2])."""
        if len(xs) != len(self.domains):  # длина обязана совпадать с числом областей
            return False  # ложь при несовпадении длины
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):  # истина, если хотя бы одно значение вне своей области
            return False  # ложь
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:  # правило nonzero_y и xs[1] = 0
            return False  # ложь
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):  # правило ordered_yz и xs[1] > xs[2]
            return False  # ложь
        return True  # иначе применим

    def to_dict(self) -> dict[str, Any]:
        """Сохранить эксперта в словарь для записи в JSON.

            »    Возврат: словарь со всеми полями; program — через to_dict(), neural — словарь или None."""
        return {  # начало словаря
            "name": self.name,  # имя
            "family": self.family,  # семейство
            "program": self.program.to_dict(),  # программа
            "domains": [list(d) for d in self.domains],  # области как списки списков
            "valid_rule": self.valid_rule,  # правило отсева
            "code_length": self.code_length,  # длина описания
            "neural": self.neural.to_dict() if self.neural else None,  # нейросеть, если есть
            "exhaustive_checks": self.exhaustive_checks,  # число проверок
            "exhaustive_errors": self.exhaustive_errors,  # число ошибок
            "parents": list(self.parents),  # родители как список
        }

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def from_dict(d: dict[str, Any]) -> "Expert":
        """Восстановить эксперта из словаря.

            »    d: словарь вида to_dict(). Возврат: Expert."""
        return Expert(  # создать эксперта
            str(d["name"]),  # имя
            str(d["family"]),  # семейство
            Program.from_dict(d["program"]),  # программа
            tuple(tuple(int(v) for v in x) for x in d["domains"]),  # области — кортежи целых
            str(d["valid_rule"]),  # правило отсева
            int(d["code_length"]),  # длина описания
            NeuralFiber.from_dict(d["neural"]) if d.get("neural") else None,  # нейросеть, если она сохранена
            int(d.get("exhaustive_checks", 0)),  # число проверок (0, если ключа нет)
            int(d.get("exhaustive_errors", 0)),  # число ошибок (0, если ключа нет)
            tuple(str(x) for x in d.get("parents", [])),  # родители — кортеж строк
        )


@dataclass  # изменяемый класс данных
class GrowthEvent:
    """Запись о событии роста: что и почему добавлено.

                version: номер версии; task: задача; action: действие; reason: причина; program: текст программы;
        »    code_length: длина описания в битах; train_examples, validation_examples: число примеров; exact_accuracy: точность."""
    version: int  # номер версии
    task: str  # задача
    action: str  # действие
    reason: str  # причина
    program: str = ""  # текст программы, по умолчанию пусто
    code_length: int = 0  # длина описания, по умолчанию 0
    train_examples: int = 0  # число обучающих примеров
    validation_examples: int = 0  # число проверочных примеров
    exact_accuracy: float = 0.0  # точность, по умолчанию 0


def _bounded_int(value: Any, lo: int, hi: int) -> int:
    """Целое значение в границах [lo, hi].

        »    value: приводимое к целому; lo, hi — границы. Вне границ — ValueError. Возврат: int(value)."""
    iv = int(value)  # привести к целому
    if iv < lo or iv > hi:  # истина, если вне границ
        raise ValueError(f"integer argument must be in [{lo},{hi}]")  # ошибка: аргумент вне [lo, hi]
    return iv  # вернуть целое


class SafeExpression:  # класс без декоратора dataclass
    """Ограниченный интерпретатор арифметико-логических выражений по синтаксическому дереву Python.

                Функция eval() не используется: дерево обходится вручную, число узлов и размер результата ограничены.
        »    Таблицы BINOPS, UNARY, CMP, CALLS задают допустимые операции и вызовы. evaluate(выражение, переменные) — вход."""

    BINOPS = {  # бинарные операции: класс узла -> функция
        ast.Add: lambda a, b: a + b,  # сложение
        ast.Sub: lambda a, b: a - b,  # вычитание
        ast.Mult: lambda a, b: a * b,  # умножение
        ast.Div: lambda a, b: a / b,  # деление
        ast.FloorDiv: lambda a, b: a // b,  # деление с округлением вниз
        ast.Mod: lambda a, b: a % b,  # остаток
        ast.Pow: lambda a, b: a**b,  # степень
        ast.BitAnd: lambda a, b: int(a) & int(b),  # побитовое «и»
        ast.BitOr: lambda a, b: int(a) | int(b),  # побитовое «или»
        ast.BitXor: lambda a, b: int(a) ^ int(b),  # исключающее «или»
        ast.LShift: lambda a, b: int(a) << int(b),  # сдвиг влево
        ast.RShift: lambda a, b: int(a) >> int(b),  # сдвиг вправо
    }  # конец таблицы
    UNARY = {  # унарные операции: класс узла -> функция
        ast.UAdd: lambda a: +a,  # унарный плюс
        ast.USub: lambda a: -a,  # унарный минус
        ast.Invert: lambda a: ~int(a),  # побитовая инверсия
        ast.Not: lambda a: int(not bool(a)),  # логическое «не» как 0 или 1
    }  # конец таблицы
    CMP = {  # сравнения: класс узла -> функция
        ast.Eq: lambda a, b: a == b,  # равно
        ast.NotEq: lambda a, b: a != b,  # не равно
        ast.Lt: lambda a, b: a < b,  # меньше
        ast.LtE: lambda a, b: a <= b,  # меньше или равно
        ast.Gt: lambda a, b: a > b,  # больше
        ast.GtE: lambda a, b: a >= b,  # больше или равно
    }  # конец таблицы
    CALLS = {  # допустимые вызовы функций: имя -> функция
        "abs": abs,  # модуль
        "min": min,  # минимум
        "max": max,  # максимум
        "gcd": math.gcd,  # наибольший общий делитель
        "lcm": math.lcm,  # наименьшее общее кратное
        "sign": _sign,  # знак
        "bit_not8": _bit_not8,  # побитовое «не» в байте
        "clamp": lambda x, lo, hi: min(max(x, lo), hi),  # ограничение диапазоном
        "nand8": _bit_nand8,  # не-и в байте
        "nor8": _bit_nor8,  # не-или в байте
        "xnor8": _bit_xnor8,  # эквивалентность в байте
        "sqrt": math.sqrt,  # квадратный корень
        "exp": math.exp,  # экспонента
        "log": math.log,  # натуральный логарифм
        "log2": math.log2,  # логарифм по основанию 2
        "log10": math.log10,  # логарифм по основанию 10
        "sin": math.sin,  # синус
        "cos": math.cos,  # косинус
        "tan": math.tan,  # тангенс
        "asin": math.asin,  # арксинус
        "acos": math.acos,  # арккосинус
        "atan": math.atan,  # арктангенс
        "atan2": math.atan2,  # арктангенс двух аргументов
        "sinh": math.sinh,  # гиперболический синус
        "cosh": math.cosh,  # гиперболический косинус
        "tanh": math.tanh,  # гиперболический тангенс
        "floor": math.floor,  # округление вниз
        "ceil": math.ceil,  # округление вверх
        "hypot": math.hypot,  # длина гипотенузы √(a² + b²)
        "degrees": math.degrees,  # радианы в градусы
        "radians": math.radians,  # градусы в радианы
        "factorial": lambda n: math.factorial(_bounded_int(n, 0, 1000)),  # факториал n, n в границах 0..1000
        "comb": lambda n, k: math.comb(  # число сочетаний C(n, k) (начало выражения)
            _bounded_int(n, 0, 1000), _bounded_int(k, 0, 1000)  # n и k в границах 0..1000
        ),  # конец выражения
        "perm": lambda n, k=None: math.perm(  # число размещений P(n, k) (начало выражения); k по умолчанию нет — тогда n!
            _bounded_int(n, 0, 1000), None if k is None else _bounded_int(k, 0, 1000)  # n в границах 0..1000; k в границах 0..1000 или None
        ),  # конец выражения
    }  # конец таблицы вызовов

    def evaluate(  # вычислить выражение (начало сигнатуры)
        self, expression: str, variables: dict[str, int | float]  # expression — текст выражения, variables — значения переменных
    ) -> int | float | bool:  # возврат: число или истина/ложь
        """Вычислить значение выражения с подставленными переменными.

                    expression: текст выражения Python; variables: словарь «имя -> число».
                    Ограничения: длина текста ≤ 1000, узлов ≤ 200, целый результат ≤ 4096 бит, дробный результат конечен.
            »    Возврат: число или логическое значение."""
        if len(expression) > 1000:  # длина текста не больше 1000
            raise ValueError("expression too long")  # ошибка: выражение слишком длинное
        tree = ast.parse(expression, mode="eval")  # разобрать выражение как выражение (не операторы)
        nodes = list(ast.walk(tree))  # все узлы дерева
        if len(nodes) > 200:  # узлов не больше 200
            raise ValueError("expression too complex")  # ошибка: выражение слишком сложное
        result = self._visit(tree.body, variables)  # вычислить корень дерева
        if isinstance(result, (int, np.integer)) and int(result).bit_length() > 4096:  # целый результат не длиннее 4096 бит
            raise OverflowError("result exceeds 4096-bit safety cap")  # ошибка переполнения
        if isinstance(result, (float, np.floating)) and not math.isfinite(  # дробный результат обязан быть конечным (начало условия)
            float(result)  # приведение к float
        ):  # конец условия
            raise OverflowError("non-finite floating-point result")  # ошибка: не конечный дробный результат
        return result  # вернуть результат

    def _visit(self, node: ast.AST, variables: dict[str, int | float]):
        """Вычислить узел дерева с подстановкой переменных.

                    node: узел синтаксического дерева; variables: значения переменных. Возврат: число или логическое значение.
            »    Константы, имена (pi, e, tau), бинарные и унарные операции, «и»/«или» вычисляются напрямую."""
        if isinstance(node, ast.Constant) and isinstance(  # узел-константа числа или логического значения (начало условия)
            node.value, (int, float, bool)  # конец условия типа значения
        ):  # конец условия
            if isinstance(node.value, float) and not math.isfinite(node.value):  # дробная константа обязана быть конечной
                raise ValueError("non-finite numeric literal")  # ошибка: не конечный числовой литерал
            return node.value  # вернуть константу
        if isinstance(node, ast.Name):  # узел-имя
            if node.id in {"pi", "e", "tau"}:  # имя из набора констант pi, e, tau
                return {"pi": math.pi, "e": math.e, "tau": math.tau}[node.id]  # значение константы
            if node.id not in variables:  # имя не задано в variables
                raise ValueError(f"unknown variable: {node.id}")  # ошибка: неизвестная переменная
            value = variables[node.id]  # взять значение переменной
            if isinstance(value, float):  # дробное значение
                if not math.isfinite(value) or abs(value) > 1e12:  # оно обязано быть конечным и не превосходить 10¹² по модулю
                    raise OverflowError("floating input outside safe finite range")  # ошибка переполнения
            elif abs(int(value)).bit_length() > 1024:  # целое значение длиннее 1024 бит
                raise OverflowError("input exceeds 1024-bit safety cap")  # ошибка переполнения
            return value  # вернуть значение
        if isinstance(node, ast.BinOp) and type(node.op) in self.BINOPS:  # бинарная операция из таблицы BINOPS
            a, b = self._visit(node.left, variables), self._visit(node.right, variables)  # a, b — значения левой и правой частей
            if isinstance(node.op, ast.Pow) and abs(float(b)) > 64:  # степень: |b| ≤ 64
                raise ValueError("absolute exponent must be at most 64")  # ошибка: модуль показателя больше 64
            if isinstance(node.op, (ast.LShift, ast.RShift)) and not (  # сдвиг: b в границах 0..4096 (начало условия)
                0 <= int(b) <= 4096  # сдвиг b в границах от 0 до 4096
            ):  # конец условия
                raise ValueError("shift must be in [0,4096]")  # ошибка: сдвиг вне [0, 4096]
            return self.BINOPS[type(node.op)](a, b)  # применить операцию к a, b
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.UNARY:  # унарная операция из таблицы UNARY
            return self.UNARY[type(node.op)](self._visit(node.operand, variables))  # применить операцию к значению операнда
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):  # логическая операция «и» или «или»
            vals = [self._visit(v, variables) for v in node.values]  # значения всех операндов
            return int(  # вернуть 0 или 1 (начало выражения)
                all(bool(v) for v in vals)  # истина, если все значения истинны
                if isinstance(node.op, ast.And)  # условие: операция «и»
                else any(bool(v) for v in vals)  # иначе «или»: хотя бы одно значение истинно
            )
        if isinstance(node, ast.Compare):  # узел-сравнение
            left = self._visit(node.left, variables)  # левое значение
            ok = True  # цепочка сравнений пока истинна
            for op, comparator in zip(node.ops, node.comparators):  # по парам (операция, правый операнд)
                if type(op) not in self.CMP:  # операция сравнения не из таблицы CMP
                    raise ValueError("comparison operator not allowed")  # ошибка: операция сравнения не разрешена
                right = self._visit(comparator, variables)  # правое значение
                ok = ok and bool(self.CMP[type(op)](left, right))  # ok = ok и (left операция right)
                left = right  # следующий операнд сравнивается с текущим правым
            return int(ok)  # вернуть 1 или 0
        if isinstance(node, ast.IfExp):  # узел-условное выражение «a if cond else b»
            cond = self._visit(node.test, variables)  # значение условия
            return self._visit(node.body if bool(cond) else node.orelse, variables)  # вычислить ветку body, если cond истинно, иначе else
        if (  # узел-вызов разрешённой функции (начало условия)
            isinstance(node, ast.Call)  # это вызов
            and isinstance(node.func, ast.Name)  # вызывается имя
            and node.func.id in self.CALLS  # имя есть в таблице CALLS
        ):  # конец условия
            if node.keywords:  # есть именованные аргументы
                raise ValueError("keyword arguments are not supported")  # ошибка: именованные аргументы не поддерживаются
            args = [self._visit(a, variables) for a in node.args]  # значения позиционных аргументов
            return self.CALLS[node.func.id](*args)  # вызвать функцию с аргументами
        raise ValueError(f"unsupported syntax: {type(node).__name__}")  # прочий синтаксис не поддерживается


class GrowingMathProcessor:  # класс без декоратора dataclass
    """Процессор, растущий за счёт экспертов.

                Хранит словарь экспертов «имя задачи -> Expert» с точными проверками на конечных областях
                и необязательными нейронными слоями. learn(spec) ищет программу для задачи; predict(name, xs)
        »    даёт точный ответ или отказ; evaluate_expression считает выражение ограниченным интерпретатором."""

    def __init__(self, lookup_limit: int = 16, mdl_lambda: float = 0.001):
        """Создать пустой процессор.

                    lookup_limit: предел размера области для таблицы ответов; mdl_lambda: вес длины кода в оценке.
            »    Инициализирует experts, version, growth_log, total_commits, total_holds."""
        self.lookup_limit = int(lookup_limit)  # предел таблицы ответов как целое
        self.mdl_lambda = float(mdl_lambda)  # вес длины кода как дробное
        self.experts: dict[str, Expert] = {}  # словарь экспертов, изначально пуст
        self.version = 0  # номер версии модели 0
        self.growth_log: list[GrowthEvent] = []  # журнал событий роста, пуст
        self.total_commits = 0  # счётчик закреплений 0
        self.total_holds = 0  # счётчик отказов 0

    def _candidate_list(self, spec: TaskSpec) -> list[Program]:
        """Список программ-кандидатов для задачи.

                    spec: описание задачи. Для семейства BoolVenn — все 16 масок; иначе candidate_programs(arity).
            »    Возврат: список Program."""
        if spec.family == "BoolVenn":  # семейство булевых функций
            return [Program(2, venn_mask=m) for m in range(16)]  # 16 программ с масками 0..15
        return candidate_programs(spec.arity)  # кандидаты по числу входов

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def _same(a: Any, b: Any) -> bool:
        """Сравнить a и b как целые; при невозможности — ложь.

            »    Возврат: int(a) == int(b), а при любой ошибке приведения — False."""
        try:  # попытаться сравнить
            return int(a) == int(b)  # равенство целых значений
        except Exception:  # любая ошибка приведения
            return False  # ложь

    def learn(  # метод learn: найти и закрепить эксперта
        self,  # описание задачи
        spec: TaskSpec,  # зерно случайности, по умолчанию 0
        seed: int = 0,  # доля обучающих примеров, по умолчанию 0.70
        train_fraction: float = 0.70,  # портить ли одну метку, по умолчанию нет
        poison_one_label: bool = False,  # обучать ли нейросеть, по умолчанию да
        train_neural: bool = True,  # конец сигнатуры
    ) -> GrowthEvent:  # возврат — событие роста
        """Найти и попытаться закрепить одного эксперта по обучающим примерам.

                    spec: задача; seed: зерно; train_fraction: доля обучающих примеров; poison_one_label: испортить одну метку;
                    train_neural: обучать ли нейросеть. Возврат: GrowthEvent с действием REUSE, HOLD или COMMIT.
                    Ход: (1) при уже известной задаче — REUSE; (2) случайное разбиение области на обучение и проверку;
                    n_train = max(1, round(train_fraction·N)) (для BoolVenn N − 1); (3) нейросеть на обучающих примерах;
                    (4) кандидаты, верные на всём обучении; (5) проверка на holdout и на всей области;
            »    (6) оценка mdl = λ·code_length, code_length = cost ≤ 32; (7) закрепление."""
        """Найти и попытаться закрепить одного эксперта по обучающим примерам.

                    Учитель (функция-оракул) используется только для проверки и полного перебора в этом офлайновом эксперименте;
            »    во внедрённой системе его заменяет независимо поставляемый проверяющий."""
        if spec.name in self.experts:  # задача уже изучена
            ev = GrowthEvent(self.version, spec.name, "REUSE", "already learned")  # событие REUSE «уже изучено»
            self.growth_log.append(ev)  # добавить событие в журнал
            return ev  # вернуть событие
        domain = spec.inputs()  # domain — все допустимые входы задачи
        if not domain:  # пустая область
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)  # отказ «пустая область»
        rng = np.random.default_rng(seed)  # генератор с зерном seed
        perm = rng.permutation(len(domain))  # случайная перестановка номеров входов
        if spec.family == "BoolVenn":  # семейство булевых функций
            n_train = max(1, len(domain) - 1)  # n_train = N − 1: один атом таблицы остаётся на проверку
        else:  # иначе
            n_train = max(  # n_train = max(1, min(N − 1, round(N·train_fraction))) (начало выражения)
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))  # число обучающих: минимум из N − 1 и round(N·train_fraction)
            )  # конец max
        train_idx, val_idx = perm[:n_train], perm[n_train:]  # первые n_train номеров — обучение, остальные — проверка
        train_x = [domain[int(i)] for i in train_idx]  # обучающие входы
        val_x = [domain[int(i)] for i in val_idx]  # проверочные входы
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}  # ответы учителя на всей области: all_y[xs] = teacher(xs)
        train_y = [all_y[xs] for xs in train_x]  # обучающие ответы
        val_y = [all_y[xs] for xs in val_x]  # проверочные ответы
        if poison_one_label and train_y:  # портить метку и есть обучающие примеры
            train_y[0] = int(train_y[0]) + 1  # первая обучающая метка увеличена на 1
            # детерминированное повреждение: состязательная проверка отказа и отката

        neural = None  # нейросети пока нет
        # Приближающая нейросеть на тех же видимых обучающих примерах.
        if train_neural and val_x:  # обучать нейросеть и есть проверочные примеры
            try:  # попытка обучения
                neural = NeuralFiber.fit(  # обучить слой на обучающих примерах
                    train_x, train_y, val_x, val_y, seed=seed + 991, epochs=120  # зерно seed + 991, 120 эпох
                )  # конец вызова
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):  # численные ошибки обучения
                neural = None  # нейросеть не создана

        candidates = self._candidate_list(spec)  # кандидаты для задачи
        train_matches: list[Program] = []  # кандидаты, верные на всём обучении
        for program in candidates:  # по кандидатам
            try:  # попытка вычисления
                if all(  # истина, если кандидат верен на всех обучающих примерах (начало all)
                    self._same(program.eval(xs), yy) for xs, yy in zip(train_x, train_y)  # program(xs) = y для каждой пары (xs, y)
                ):  # конец all
                    train_matches.append(program)  # добавить кандидата
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибки вычисления — кандидат пропускается
                continue  # следующий кандидат
        train_matches.sort(key=lambda p: (p.cost, p.pretty()))  # сортировка по (стоимость, текст)

        for program in train_matches:  # по кандидатам в порядке возраста
        # Сначала независимая проверка, затем полное сравнение с конечным оракулом.
            try:  # попытка
                if not all(  # неверно хотя бы на одном проверочном примере (начало all)
                    self._same(program.eval(xs), yy) for xs, yy in zip(val_x, val_y)  # program(xs) = y для проверочных пар
                ):  # конец all
                    continue  # кандидат отбрасывается
                errors = 0  # счётчик ошибок на всей области
                for xs in domain:  # по всем входам области
                    if not self._same(program.eval(xs), all_y[xs]):  # ошибка на входе xs
                        errors += 1  # ошибок стало на одну больше
                if errors != 0:  # есть ошибки
                    continue  # кандидат отбрасывается
                code_length = int(program.cost)  # длина кода = стоимость программы
                # Оценка в духе минимальной длины описания: точная предсказательная ошибка (здесь 0) плюс явная длина кода.
                mdl_score = 0.0 + self.mdl_lambda * code_length  # оценка mdl_score = 0 + λ·code_length
                if not np.isfinite(mdl_score) or code_length > 32:  # оценка не конечна или код длиннее 32
                    continue  # кандидат отбрасывается
                expert = Expert(  # создать эксперта
                    spec.name,  # имя задачи
                    spec.family,  # семейство
                    program,  # программа
                    spec.domains,  # области входов
                    spec.valid_rule,  # правило отсева
                    code_length,  # длина кода
                    neural,  # нейросеть
                    len(domain),  # число проверок = размер области
                    0,  # ошибок 0
                )  # конец описания
                self.experts[spec.name] = expert  # записать эксперта в словарь
                self.version += 1  # версия + 1
                self.total_commits += 1  # счётчик закреплений + 1
                ev = GrowthEvent(  # событие COMMIT
                    self.version,  # номер версии
                    spec.name,  # имя задачи
                    "COMMIT",  # действие COMMIT
                    "validation+exhaustive pass",  # причина: проверка и полный перебор пройдены
                    program.pretty(),  # текст программы
                    code_length,  # длина кода
                    len(train_x),  # число обучающих примеров
                    len(val_x),  # число проверочных примеров
                    1.0,  # точность 1.0
                )  # конец описания события
                self.growth_log.append(ev)  # добавить событие в журнал
                return ev  # вернуть событие
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления — следующий кандидат
                continue  # продолжить

        if (  # область не больше lookup_limit (начало условия)
        # Таблица ответов разрешена только при полной опоре не больше заданного предела.
            len(domain) <= self.lookup_limit  # размер области не больше lookup_limit
            and not poison_one_label  # метки не испорчены
            and len(train_x) == len(domain)  # обучающие примеры — вся область
        ):  # конец условия
            pass  # ничего не делать (заготовка)
        return self._hold(  # отказ «нет проверенной программы»
            # В текущем наборе задач нет задач на таблицу ответов; ветка оставлена для будущих
            # полных микрообластей и не экстраполирует молча.
            spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x), neural  # задача, причина, число обучающих и проверочных примеров, нейросеть
        )  # конец вызова

    def _hold(  # метод _hold: зарегистрировать отказ
        self,  # задача
        spec: TaskSpec,  # причина отказа
        reason: str,  # число обучающих примеров
        ntrain: int,  # число проверочных примеров
        nval: int,  # нейросеть, по умолчанию нет
        neural: Optional[NeuralFiber] = None,  # конец сигнатуры
    ) -> GrowthEvent:  # возврат — событие роста
        """Зарегистрировать отказ (HOLD) по задаче.

                    spec: задача; reason: причина; ntrain, nval: число обучающих и проверочных примеров; neural: нейросеть, если обучалась.
            »    Счётчик total_holds увеличивается на 1. Возврат: GrowthEvent."""
        self.total_holds += 1  # счётчик отказов + 1
        ev = GrowthEvent(  # событие HOLD
            self.version, spec.name, "HOLD", reason, "", 0, ntrain, nval, 0.0  # версия, задача, действие HOLD, причина, без программы и с нулевой длиной кода
        )  # конец описания события
        self.growth_log.append(ev)  # добавить в журнал
        return ev  # вернуть событие

    def predict(self, name: str, *xs: int, mode: str = "exact") -> dict[str, Any]:
        """Ответ эксперта по имени операции.

                    name: имя операции; xs: вход; mode: "exact" — точная программа, "neural" — нейросеть.
                    Возврат: словарь со status: ABSTAIN (неизвестная операция или вне области), OK (точное значение)
            »    или APPROXIMATE (нейросеть без гарантии точности)."""
        expert = self.experts.get(name)  # взять эксперта по имени
        if expert is None:  # эксперта нет
            return {"status": "ABSTAIN", "reason": "unknown operation"}  # отказ: неизвестная операция
        if not expert.supports(xs):  # вход вне области эксперта
            return {"status": "ABSTAIN", "reason": "outside certified finite domain"}  # отказ: вне проверенной конечной области
        if mode == "exact":  # точный режим
            return {  # начало словаря ответа
                "status": "OK",  # статус OK
                "value": expert.program.eval(xs),  # значение программы
                "source": "verified-symbolic",  # источник — проверенная символьная программа
                "model_version": self.version,  # версия модели
            }  # конец словаря
        if mode == "neural":  # нейронный режим
            if expert.neural is None:  # нейросети нет
                return {"status": "ABSTAIN", "reason": "neural fiber unavailable"}  # отказ: нейрослой недоступен
            return {  # начало словаря ответа
                "status": "APPROXIMATE",  # статус APPROXIMATE
                "value": expert.neural.predict(xs),  # прогноз нейросети
                "source": "neural-fiber-no-exact-guarantee",  # источник — нейрослой без гарантии точности
                "model_version": self.version,  # версия модели
            }  # конец словаря
        raise ValueError("mode must be 'exact' or 'neural'")  # ошибка: режим обязан быть exact или neural

    def verify_all(self, specs: Sequence[TaskSpec]) -> dict[str, Any]:
        """Проверить всех экспертов по всем задачам specs.

                    Для каждой задачи: если эксперт закреплён — число верных ответов на всех входах и точность ok/max(1, n).
            »    Возврат: словарь records, checked, correct, accuracy = correct/max(1, checked), committed."""
        records = []  # записи по задачам
        total = correct = 0  # всего входов и верных ответов
        for spec in specs:  # по задачам
            expert = self.experts.get(spec.name)  # эксперт задачи
            if expert is None:  # эксперт не закреплён
                records.append(  # добавить запись (начало словаря)
                    {  # начало словаря записи
                        "task": spec.name,  # имя задачи
                        "committed": False,  # не закреплён
                        "accuracy": 0.0,  # точность 0
                        "n": len(spec.inputs()),  # число входов задачи
                    }  # конец словаря
                )  # конец вызова append
                continue  # следующая задача
            rows = spec.inputs()  # входы задачи
            ok = sum(  # число верных ответов ok (начало sum)
                int(expert.program.eval(xs) == int(spec.teacher(*xs))) for xs in rows  # 1, если program(xs) = teacher(xs), суммируется по входам
            )  # конец sum
            total += len(rows)  # total = total + число входов
            correct += ok  # верных ответов стало на ok больше
            records.append(  # добавить запись (начало словаря)
                {  # начало словаря записи
                    "task": spec.name,  # имя задачи
                    "committed": True,  # закреплён
                    "accuracy": ok / max(1, len(rows)),  # точность ok / max(1, n)
                    "n": len(rows),  # число входов
                    "program": expert.program.pretty(),  # текст программы
                    "code_length": expert.code_length,  # длина кода
                }  # конец словаря
            )  # конец вызова append
        return {  # возврат — словарь итогов
            "records": records,  # записи по задачам
            "checked": total,  # всего проверено входов
            "correct": correct,  # верных ответов
            "accuracy": correct / max(1, total),  # точность correct / max(1, total)
            "committed": len(self.experts),  # число закреплённых экспертов
        }  # конец словаря

    def evaluate_expression(
        self, expression: str, **variables: int | float  # self, expression — текст выражения, variables — значения переменных по имени
    ) -> dict[str, Any]:  # возврат — словарь с результатом или ошибкой
        """Вычислить составное целое или логическое выражение ограниченным интерпретатором.

                    expression: текст выражения; variables: значения переменных по имени.
                Возврат: словарь со status OK и значением либо status ERROR и текстом причины."""
        try:  # попытка вычисления
            value = SafeExpression().evaluate(expression, variables)  # значение выражения через ограниченный интерпретатор
            return {  # возврат — словарь ответа
                "status": "OK",  # статус OK
                "value": value,  # значение выражения
                "source": "safe-expression-dsl",  # источник — ограниченный язык выражений
                "model_version": self.version,  # версия модели
            }
        except (  # перехват ошибок (начало кортежа типов)
            ValueError,  # неверное значение
            TypeError,  # неверный тип
            ZeroDivisionError,  # деление на ноль
            OverflowError,  # арифметическое переполнение
            SyntaxError,  # синтаксическая ошибка
        ) as exc:  # конец кортежа; exc — исключение
            return {  # возврат — словарь ошибки
                "status": "ERROR",  # статус ERROR
                "reason": str(exc),  # текст причины
                "model_version": self.version,  # версия модели
            }

    def save(self, path: str | Path) -> None:  # сохранить модель в файл JSON
        data = {  # начало словаря состояния
            "format": "HAGI-GrowingMathProcessor",  # формат файла
            "version": VERSION,  # версия схемы
            "model_version": self.version,  # версия модели
            "lookup_limit": self.lookup_limit,  # предел таблицы ответов
            "mdl_lambda": self.mdl_lambda,  # вес длины кода
            "total_commits": self.total_commits,  # число закреплений
            "total_holds": self.total_holds,  # число отказов
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},  # эксперты по именам в отсортированном порядке
            "growth_log": [vars(x) for x in self.growth_log],  # журнал событий — словари полей
        }  # конец словаря
        Path(path).write_text(  # записать файл
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"  # JSON с отступом 2 и сортировкой ключей
        )

    @staticmethod  # статический метод — вызывается через класс, без экземпляра
    def load(path: str | Path) -> "GrowingMathProcessor":
        """Загрузить модель из файла JSON.

                path: путь к файлу, записанному save(). Проверяет поле format. Возврат: GrowingMathProcessor."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # прочитать и разобрать JSON
        if data.get("format") != "HAGI-GrowingMathProcessor":  # формат не совпадает
            raise ValueError("not a HAGI Growing Math Processor state file")  # ошибка: это не файл состояния процессора
        model = GrowingMathProcessor(  # создать модель
            int(data.get("lookup_limit", 16)), float(data.get("mdl_lambda", 0.001))  # предел таблицы (по умолчанию 16) и вес длины кода (по умолчанию 0.001)
        )
        model.version = int(data.get("model_version", 0))  # версия модели
        model.total_commits = int(data.get("total_commits", 0))  # число закреплений
        model.total_holds = int(data.get("total_holds", 0))  # число отказов
        model.experts = {  # восстановить экспертов
            k: Expert.from_dict(v) for k, v in data.get("experts", {}).items()  # каждый эксперт через from_dict
        }
        model.growth_log = [GrowthEvent(**x) for x in data.get("growth_log", [])]  # журнал событий из словарей полей
        return model  # вернуть модель



# ---------------------------- Автономный эксперимент ----------------------------

def run_experiment(  # run_experiment: провести эксперимент по росту на нескольких зёрнах
    seeds: Sequence[int] = (0, 1, 2), out_dir: str = "."  # seeds — зёрна случайности, out_dir — каталог результатов
) -> dict[str, Any]:  # возврат — словарь итогов
    out = Path(out_dir)  # out — каталог результатов
    out.mkdir(parents=True, exist_ok=True)  # создать каталог при необходимости
    suite = make_task_suite()  # набор задач
    per_seed = []  # строки итогов по зёрнам
    main_model: Optional[GrowingMathProcessor] = None  # модель основного зерна, пока нет
    main_seed = int(seeds[0]) if seeds else 0  # основное зерно — первое из списка (0, если список пуст)
    start = time.time()  # время начала
    for seed in seeds:  # по зернам
        model = GrowingMathProcessor()  # новая модель для зерна
        seed_t0 = time.time()  # время начала зерна
        events: list[GrowthEvent] = []  # список событий роста
        stage_order = ["BoolVenn", "unary", "integer-binary", "composed-ternary"]  # порядок стадий по семействам
        # Упорядоченная программа обучения: логика -> унарные -> арифметика -> предикаты и биты -> композиции.
        growth_curve = []  # кривая роста по стадиям
        for stage in stage_order:  # по стадиям
            stage_tasks = [s for s in suite if s.family == stage]  # задачи данной стадии
            rng = np.random.default_rng(seed + len(stage) * 101)  # генератор с зернем seed + длина_имени_стадии·101
            # Порядок внутри стадии детерминирован, но перемешивается зерном, чтобы проявить влияние порядка.
            if len(stage_tasks) > 1:  # если задач больше одной
                order = rng.permutation(len(stage_tasks))  # случайный порядок номеров задач
                stage_tasks = [stage_tasks[int(i)] for i in order]  # переставить задачи
            for j, spec in enumerate(stage_tasks):  # по задачам стадии с номером j
                events.append(  # добавить событие обучения (начало вызова)
                    model.learn(spec, seed=seed * 1009 + j * 17 + len(spec.name))  # зерно задачи seed·1009 + j·17 + длина имени
                )  # конец вызова append
            growth_curve.append(  # добавить точку кривой роста (начало словаря)
                {  # начало словаря
                    "stage": stage,  # имя стадии
                    "new_experts": len(stage_tasks),  # число задач стадии
                    "cumulative_experts": len(model.experts),  # накопленное число экспертов
                    "cumulative_code_length": sum(  # накопленная длина кода экспертов (начало sum)
                        e.code_length for e in model.experts.values()  # сумма code_length по экспертам
                    ),  # конец sum
                }  # конец словаря
            )  # конец вызова append
        verification = model.verify_all(suite)  # проверить всех экспертов по всем задачам
        committed = len(model.experts)  # число закреплённых экспертов
        neural_scores = [  # точности нейросетей на проверке (начало списка)
            e.neural.validation_exact_accuracy  # validation_exact_accuracy эксперта
            for e in model.experts.values()  # по экспертам
            if e.neural is not None and np.isfinite(e.neural.validation_exact_accuracy)  # нейросеть есть и точность конечна
        ]  # конец списка
        neural_rmse = [  # корни из средних квадратов ошибок нейросетей (начало списка)
            math.sqrt(max(0.0, e.neural.validation_loss))  # корень из квадрата ошибки: √max(0, validation_loss)
            for e in model.experts.values()  # по экспертам
            if e.neural is not None and np.isfinite(e.neural.validation_loss)  # нейросеть есть и ошибка конечна
        ]  # конец списка
        ood_checks = ood_abstains = 0  # счётчики проверок и отказов — нули
        # Проверка ограждений: каждая известная операция обязана отказать на входе вне своей области.
        for spec in suite:  # по всем задачам
            if spec.name not in model.experts:  # задача не изучена
                continue  # пропустить
            outside = [d[-1] + 1 for d in spec.domains]  # вход вне области: каждое значение на 1 больше наибольшего допустимого
            result = model.predict(spec.name, *outside)  # запрос модели
            ood_checks += 1  # проверок + 1
            ood_abstains += int(result["status"] == "ABSTAIN")  # отказов + 1, если статус ABSTAIN
        poison_model = GrowingMathProcessor()  # отдельная модель для теста
        # Тест с испорченной меткой: одна неверная метка ADD не должна дать нового закреплённого эксперта.
        add_spec = next(s for s in suite if s.name == "ADD")  # задача ADD
        poison_event = poison_model.learn(  # обучение с испорченной меткой (начало вызова)
            add_spec, seed=seed + 44001, poison_one_label=True  # зерно seed + 44001, портить одну метку
        )  # конец вызова
        prime_spec = TaskSpec(  # задача «простое ли число» (начало описания)
        # Неизвестная нетривиальная функция имеет область из 17 точек: предел таблицы 16 меньше — должен быть отказ.
            "IS_PRIME",  # имя IS_PRIME
            1,  # один вход
            (tuple(range(-8, 9)),),  # область — целые от −8 до 8
            lambda x: int(  # ответ: 1, если x ≥ 2 и x не делится ни на одно d от 2 до ⌊√x⌋ (начало выражения)
                x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))  # x ≥ 2 и все остатки x mod d ненулевые для d = 2..⌊√x⌋
            ),  # конец выражения
            family="unseen",  # семейство «невиданное»
        )  # конец описания
        unknown_event = GrowingMathProcessor().learn(prime_spec, seed=seed + 99001)  # обучение на неизвестной задаче
        elapsed = time.time() - seed_t0  # время работы зерна
        row = {  # строка итогов зерна (начало словаря)
            "seed": int(seed),  # зерно
            "committed": committed,  # число закреплённых экспертов
            "expected": len(suite),  # ожидаемое число — размер набора задач
            "commit_rate": committed / len(suite),  # доля закреплений committed / len(suite)
            "symbolic_exact_accuracy": verification["accuracy"],  # точная точность символьных программ
            "symbolic_examples_checked": verification["checked"],  # число проверенных примеров
            "neural_holdout_exact_accuracy_mean": (  # средняя точность нейросетей на проверке (начало выражения)
                float(np.mean(neural_scores)) if neural_scores else float("nan")  # среднее по списку или nan, если список пуст
            ),  # конец выражения
            "neural_holdout_rmse_mean": (  # средний корень из квадрата ошибки нейросетей (начало выражения)
                float(np.mean(neural_rmse)) if neural_rmse else float("nan")  # среднее по списку или nan, если список пуст
            ),  # конец выражения
            "ood_checks": ood_checks,  # число проверок вне области
            "ood_abstains": ood_abstains,  # число отказов вне области
            "poison_action": poison_event.action,  # действие при испорченной метке
            "unknown_is_prime_action": unknown_event.action,  # действие на неизвестной задаче
            "growth_curve": growth_curve,  # кривая роста
            "elapsed_seconds": elapsed,  # время работы зерна в секундах
        }  # конец словаря
        per_seed.append(row)  # добавить строку к итогам
        if seed == main_seed:  # зерно основное
            main_model = model  # запомнить основную модель
            with (out / f"hagi_growing_math_seed{seed}_tasks.csv").open(  # открыть файл таблицы задач (начало with)
            # Полная таблица по задачам для основного зерна.
                "w", encoding="utf-8"  # режим записи, utf-8
            ) as f:  # конец заголовка with
                f.write(  # записать заголовок таблицы (начало вызова)
                    "task,family,committed,program,code_length,neural_val_exact_acc,neural_val_mse\n"  # строка заголовка столбцов
                )  # конец вызова
                for spec in suite:  # по задачам
                    ex = model.experts.get(spec.name)  # эксперт задачи
                    if ex is None:  # эксперта нет
                        f.write(f"{spec.name},{spec.family},0,,0,,\n")  # строка: не закреплён, пустые поля
                    else:  # иначе
                        na = (  # точность нейросети (начало выражения)
                            ex.neural.validation_exact_accuracy  # точность нейросети на проверке
                            if ex.neural  # если нейросеть есть
                            else float("nan")  # иначе nan
                        )  # конец выражения
                        nm = ex.neural.validation_loss if ex.neural else float("nan")  # ошибка нейросети или nan
                        prog = ex.program.pretty().replace('"', '""')  # текст программы с удвоенными кавычками
                        f.write(  # записать строку таблицы (начало вызова)
                            f'"{spec.name}","{spec.family}",1,"{prog}",{ex.code_length},{na},{nm}\n'  # имя, семейство, 1, программа, длина кода, точность, ошибка
                        )  # конец вызова
    assert main_model is not None  # гарантия: основная модель найдена
    model_path = out / "hagi_growing_math_model.json"  # путь к файлу модели
    main_model.save(model_path)  # сохранить основную модель

    reloaded = GrowingMathProcessor.load(model_path)  # загрузить модель
    # Состязательная проверка ограждений на сохранённой и заново загруженной модели.
    reload_ok = reloaded.predict("ADD", 7, -3).get("value") == 4  # ADD(7, −3) = 4
    ood_ok = reloaded.predict("ADD", 99, 2).get("status") == "ABSTAIN"  # ADD(99, 2) вне области — отказ

    aggregate: dict[str, Any] = {}  # агрегаты по зёрнам
    metric_keys = (  # метрики для усреднения (начало кортежа)
        "committed",  # число закреплённых
        "symbolic_exact_accuracy",  # точная точность
        "neural_holdout_exact_accuracy_mean",  # средняя точность нейросетей
        "neural_holdout_rmse_mean",  # средний корень из квадрата ошибки
        "ood_abstains",  # число отказов вне области
        "elapsed_seconds",  # время работы
    )  # конец кортежа
    for key in metric_keys:  # по метрикам
        vals = [float(r[key]) for r in per_seed]  # значения метрики по зёрнам
        base = key[:-5] if key.endswith("_mean") else key  # базовое имя: без суффикса «_mean», если он есть
        aggregate[base + "_mean"] = float(np.mean(vals))  # среднее по зёрнам
        aggregate[base + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0  # несмещённое стандартное отклонение (0 при одном значении)
    aggregate.update(  # дополнить агрегаты (начало словаря)
        {  # начало словаря
            "seeds": list(map(int, seeds)),  # зёрна
            "expected_operations": len(suite),  # ожидаемое число операций
            "finite_support_examples_per_operation": [len(s.inputs()) for s in suite],  # число входов у каждой операции
            "total_finite_support_examples": int(sum(len(s.inputs()) for s in suite)),  # суммарное число входов по всем операциям
            "reload_roundtrip_pass": bool(reload_ok),  # запись-чтение модели прошло
            "ood_abstain_pass": bool(ood_ok),  # отказ вне области сработал
            "total_runtime_seconds": time.time() - start,  # общее время работы
        }  # конец словаря
    )  # конец вызова update
    result = {  # итоговый словарь (начало)
        "version": VERSION,  # версия
        "scope": "integer inputs -8..8 (with task-specific subdomains) + all 16 two-input Boolean functions",  # область применения: целые входы от −8 до 8 и 16 булевых функций
        "per_seed": per_seed,  # итоги по зёрнам
        "aggregate": aggregate,  # агрегаты
        "interpretation": [  # интерпретация (список строк)
            "Exact results are certified only on each operation's finite declared support.",  # точные результаты заверены только на конечной области каждой операции
            "The neural fibers are approximate and are not the exact inference path.",  # нейросети приближённые и не являются точным путём вывода
            "Unknown functions outside the learned library abstain; no claim of all mathematics is made.",  # неизвестные функции дают отказ; утверждение обо всей математике не делается
            "The teacher oracle is used by the offline experiment's independent validation/exhaustive audit.",  # учитель используется офлайновым экспериментом для независимой проверки
        ],  # конец списка
    }  # конец словаря
    (out / "hagi_growing_math_experiment.json").write_text(  # записать файл итогов эксперимента
        json.dumps(result, indent=2), encoding="utf-8"  # JSON с отступом 2
    )
    _write_report(result, out / "hagi_growing_math_report.md")  # записать текстовый отчёт
    return result  # вернуть итоги


def _write_report(result: dict[str, Any], path: Path) -> None:
    """Провести автономный эксперимент по росту процессора на нескольких зёрнах.

                seeds: зёрна случайности; out_dir: каталог для модели, таблицы задач, итогов и отчёта.
                Для каждого зерна: новая модель учится по стадиям (логика, унарные, бинарные, композиции),
                проверяется на всех задачах, на отказах вне области, на испорченной метке и на неизвестной задаче.
            Возврат: словарь с version, scope, per_seed, aggregate и interpretation."""
    """Записать текстовый отчёт об эксперименте в файл path.

                result: словарь, возвращённый run_experiment(). Строит список строк Markdown: заголовок, каталог операций,
            агрегаты (среднее и отклонение), строки по зёрнам, кривая роста, толкование и ограничения — и записывает их в path."""
    agg = result["aggregate"]  # агрегаты результата
    lines = [  # строки отчёта (начало списка)
        "# HAGI Growing Math Processor — autonomous experiment",  # заголовок
        "",  # пустая строка
        f"Version: {result['version']}",  # версия
        f"Seeds: {', '.join(map(str, agg['seeds']))}",  # зёрна
        f"Declared operation set: {agg['expected_operations']} tasks (16 BoolVenn + integer unary/binary + ternary compositions).",  # объявленный набор операций
        "",  # пустая строка
        "## Learned operation catalogue",  # раздел «изученный каталог операций»
        "",  # пустая строка
        "- 16 BoolVenn functions: every two-input Boolean truth table (FALSE, NOR, implications, XOR/XNOR, AND/NAND, OR, TRUE, projections and negations).",  # пункт про 16 булевых функций
        "- 7 unary integer functions: NEG, ABS, SIGN, SQUARE, CUBE, BIT_NOT8, CLAMP_TO_3.",  # пункт про 7 унарных функций
        "- 23 binary integer functions: ADD, SUB, MUL, FLOORDIV, MOD, GCD, LCM, MIN, MAX, ABSDIFF, EQ/NE/LT/LE/GT/GE, BIT_AND/OR/XOR/NAND8/NOR8/XNOR8, POW_INT.",  # пункт про 23 бинарные функции
        "- 3 ternary compositions: FMA, CLAMP, MEDIAN3.",  # пункт про 3 тернарные композиции
        "",  # пустая строка
        "The expression interpreter additionally supports bounded integer arithmetic/bitwise expressions and selected real functions (sqrt, exp, log, trigonometric and hyperbolic functions). Those calculator functions are not part of the 49-operation exhaustive-learning score.",  # пункт про интерпретатор выражений
        "",  # пустая строка
        "## Aggregate results",  # раздел «агрегированные результаты»
        "",  # пустая строка
        "| Metric | Mean | SD |",  # заголовок таблицы метрик
        "|---|---:|---:|",  # разделитель таблицы
        f"| Committed experts | {agg['committed_mean']:.2f} | {agg['committed_sd']:.2f} |",  # строка: закреплённые эксперты
        f"| Exhaustive symbolic accuracy | {100*agg['symbolic_exact_accuracy_mean']:.2f}% | {100*agg['symbolic_exact_accuracy_sd']:.2f} pp |",  # строка: точная символьная точность
        f"| Neural holdout exact accuracy | {100*agg['neural_holdout_exact_accuracy_mean']:.2f}% | {100*agg['neural_holdout_exact_accuracy_sd']:.2f} pp |",  # строка: точность нейросетей
        f"| Neural holdout RMSE | {agg['neural_holdout_rmse_mean']:.4f} | {agg['neural_holdout_rmse_sd']:.4f} |",  # строка: корень из квадрата ошибки нейросетей
        f"| OOD abstentions | {agg['ood_abstains_mean']:.2f} | {agg['ood_abstains_sd']:.2f} |",  # строка: отказы вне области
        f"| Runtime per seed | {agg['elapsed_seconds_mean']:.2f}s | {agg['elapsed_seconds_sd']:.2f}s |",  # строка: время на зерно
        "",  # пустая строка
        "## Per-seed results",  # раздел «результаты по зёрнам»
        "",  # пустая строка
        "| Seed | Experts | Symbolic exact | Neural holdout exact | Neural RMSE | OOD abstain | Poison | Unknown IS_PRIME |",  # заголовок таблицы по зёрнам
        "|---:|---:|---:|---:|---:|---:|---|---|",  # разделитель таблицы
    ]  # конец списка
    for r in result["per_seed"]:  # по строкам результатов зёрен
        lines.append(  # добавить строку таблицы (начало вызова)
            f"| {r['seed']} | {r['committed']}/{r['expected']} | {100*r['symbolic_exact_accuracy']:.2f}% | {100*r['neural_holdout_exact_accuracy_mean']:.2f}% | {r['neural_holdout_rmse_mean']:.4f} | {r['ood_abstains']}/{r['ood_checks']} | {r['poison_action']} | {r['unknown_is_prime_action']} |"  # зерно, эксперты, точности, ошибка, отказы, отравление, неизвестная задача
        )  # конец вызова
    lines += [  # дополнить список (начало)
        "",  # пустая строка
        "## Growth curve (seed 0)",  # раздел «кривая роста (зерно 0)»
        "",  # пустая строка
        "| Stage | New experts | Cumulative experts | Cumulative code units |",  # заголовок таблицы кривой
        "|---|---:|---:|---:|",  # разделитель таблицы
    ]  # конец списка
    for stage in result["per_seed"][0].get("growth_curve", []):  # по стадиям кривой роста первого зерна
        lines.append(  # добавить строку таблицы (начало вызова)
            f"| {stage['stage']} | {stage['new_experts']} | {stage['cumulative_experts']} | {stage['cumulative_code_length']} |"  # стадия, новые эксперты, накопленные эксперты, накопленная длина кода
        )  # конец вызова
    lines += [  # дополнить список (начало)
        "",  # пустая строка
        "## Interpretation and limits",  # раздел «толкование и ограничения»
        "",  # пустая строка
        "The operation library grows by adding one expert after program synthesis, held-out validation, and exhaustive comparison over that operation's finite support. Each expert also receives a small trained neural fiber; approximate neural predictions are available separately, but exact inference uses the verified symbolic program.",  # абзац: библиотека растёт по одному эксперту после синтеза, проверки и полного перебора
        "",  # пустая строка
        "The 16 two-input Boolean functions are the complete BoolVenn truth-table family. Integer functions are certified only on the finite input ranges recorded in the task table. This is not a proof of correctness for unbounded integers, real analysis, transcendental functions, arbitrary user-defined functions, or every possible mathematical operation. The safe expression interpreter computes selected transcendental functions but does not discover or formally certify them as learned experts.",  # абзац: 16 булевых функций полны, целые заверены на конечных областях
        "",  # пустая строка
        "Out-of-domain inputs abstain. The poisoned-label and unknown-prime probes are expected to HOLD rather than silently commit. The training oracle is available in the offline experiment; a deployed learner needs an independent verifier or new labelled examples to certify a new function.",  # абзац: вне области отказ; отравление и неизвестная задача дают отказ
        "",  # пустая строка
        "The design is MDL-inspired: candidates are ordered by AST description length and exact-fit programs are required to pass a hard finite-domain commit check. This is not a complete implementation of the repository's probabilistic/energy CommitGate or its Lean proof obligations. BoolVenn follows the 16-function finite lattice; the growth controller adds experts in stages. These formal components do not establish generalization beyond the specified finite domains.",  # абзац: дизайн навеян минимальной длиной описания
        "",  # пустая строка
        "## Related Lean specifications",  # раздел со ссылками на родственные формальные спецификации
        "",  # пустая строка
        "- BoolVenn: https://github.com/ShmidtS/primes/blob/main/Hagi/Data/BoolVenn.lean",  # ссылка на спецификацию булевых функций
        "- MDL objective: https://github.com/ShmidtS/primes/blob/main/Hagi/Step/MDLObjective.lean",  # ссылка на спецификацию оценки длины описания
        "- Safe commit gate: https://github.com/ShmidtS/primes/blob/main/Hagi/Unified/CommitGate.lean",  # ссылка на спецификацию стыка закрепления
        "- Self-development search: https://github.com/ShmidtS/primes/blob/main/Hagi/Growth/SelfDevelopment.lean",  # ссылка на спецификацию поиска саморазвития
        "- Recursive growth conditions: https://github.com/ShmidtS/primes/blob/main/Hagi/Growth/RecursiveSelfDevelopment.lean",  # ссылка на спецификацию условий рекурсивного роста
    ]  # конец списка
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # записать строки в файл с переводами строк



# ----------------------------------- Тесты --------------------------------------

def self_test() -> None:
    """Самопроверка процессора: обучение, проверка, сериализация и отказы.

                Обучает по всем задачам без нейросетей, требует закрепления каждой и точности 1.0,
                проверяет ответы, отказ вне области, запись-чтение модели, интерпретатор выражений,
            отказ на импорте модуля, на испорченной метке и на неизвестной задаче. В конце печатает SELF_TEST PASS."""
    suite = make_task_suite()  # набор задач
    model = GrowingMathProcessor()  # модель
    for i, spec in enumerate(suite):  # по задачам с номером i
        ev = model.learn(spec, seed=10_000 + i, train_neural=False)  # обучить без нейросети с зерном 10 000 + i
        assert ev.action == "COMMIT", (spec.name, ev)  # каждая задача закреплена
    check = model.verify_all(suite)  # проверить всех экспертов
    assert check["committed"] == len(suite)  # закреплено столько же, сколько задач
    assert check["accuracy"] == 1.0  # точность 1.0
    assert all(  # ADD верна на всей области (начало all)
        model.predict("ADD", x, y)["value"] == x + y  # ADD(x, y) = x + y
        for x in range(-8, 9)  # x от −8 до 8
        for y in range(-8, 9)  # y от −8 до 8
    )  # конец all
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"  # ADD(100, 3) — отказ
    assert model.predict("BOOL_XOR", 1, 0)["value"] == 1  # XOR(1, 0) = 1
    assert model.predict("BOOL_NAND", 1, 1)["value"] == 0  # NAND(1, 1) = 0
    with_path = Path("/tmp/hagi_growing_math_selftest.json")  # путь файла модели
    model.save(with_path)  # сохранить модель
    loaded = GrowingMathProcessor.load(with_path)  # загрузить модель
    assert loaded.predict("FMA", 2, 3, 4)["value"] == 10  # FMA(2, 3, 4) = 10
    assert loaded.predict("CLAMP", 99, -2, 2)["status"] == "ABSTAIN"  # CLAMP(99, −2, 2) — отказ
    assert loaded.evaluate_expression("x*y + gcd(x,y)", x=7, y=3)["value"] == 22  # x·y + gcd(x, y) при x = 7, y = 3 равно 22
    assert (  # выражение с сравнением и логикой (начало assert)
        loaded.evaluate_expression("(x < y) and (y % 2 == 1)", x=2, y=3)["value"] == 1  # (x < y) и (y mod 2 = 1) при x = 2, y = 3 равно 1
    )  # конец assert
    trig = loaded.evaluate_expression("sqrt(x*x) + sin(pi/2)", x=3)  # √(x²) + sin(π/2) при x = 3
    assert trig["status"] == "OK" and abs(trig["value"] - 4.0) < 1e-10  # статус OK и значение 4 с точностью 10⁻¹⁰
    assert (  # запрет импорта модулей в выражении (начало assert)
        loaded.evaluate_expression("__import__('os').system('false')", x=1)["status"]  # выражение с __import__ — статус ERROR
        == "ERROR"  # сравнение со строкой ERROR
    )  # конец assert
    poisoned = GrowingMathProcessor().learn(  # обучение с испорченной меткой (начало вызова)
    # Испорченные данные и неподдерживаемый алгоритм с большей областью обязаны быть отклонены.
        next(s for s in suite if s.name == "ADD"),  # задача ADD
        seed=77,  # зерно 77
        poison_one_label=True,  # портить одну метку
        train_neural=False,  # без нейросети
    )  # конец вызова
    assert poisoned.action == "HOLD"  # действие — отказ
    unknown = TaskSpec(  # неизвестная задача «простое ли число» (начало описания)
        "IS_PRIME",  # имя
        1,  # один вход
        (tuple(range(-8, 9)),),  # область от −8 до 8
        lambda x: int(x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))),  # ответ: x ≥ 2 и нет делителей до ⌊√x⌋
    )  # конец описания
    assert (  # обучение на неизвестной задаче (начало assert)
        GrowingMathProcessor().learn(unknown, seed=78, train_neural=False).action  # зерно 78, без нейросети; действие
        == "HOLD"  # равно HOLD
    )  # конец assert
    print(  # печать итога (начало вызова)
        f"SELF_TEST PASS: {len(suite)} operation experts, exhaustive finite-support checks, serialization, OOD abstain, poison hold"  # строка SELF_TEST PASS с перечнем проверенного
    )  # конец вызова


def query_model(  # query_model: запросить операцию у сохранённой модели
    model_path: str, name: str, args: Sequence[int], mode: str = "exact"  # model_path, name, args, mode
) -> None:  # без возврата
    model = GrowingMathProcessor.load(model_path)  # загрузить модель
    print(json.dumps(model.predict(name, *map(int, args), mode=mode), indent=2))  # напечатать JSON ответа с аргументами, приведёнными к целым


def main() -> None:
    """Загрузить модель и напечатать ответ операции в JSON.

            model_path: файл модели; name: имя операции; args: аргументы; mode: exact или neural. Ничего не возвращает."""
    """Точка входа командной строки процессора.

                Разбирает аргументы: --self-test, --experiment, --seeds, --out-dir, --query, --expr, --var, --mode, --model.
            Выполняет самопроверку, запрос к модели, вычисление выражения или полный эксперимент."""
    p = argparse.ArgumentParser(description="HAGI Growing Math Processor")  # разборщик аргументов
    p.add_argument("--self-test", action="store_true")  # флаг --self-test
    p.add_argument(  # флаг --experiment (начало описания аргумента)
        "--experiment", action="store_true", help="run autonomous growth experiment"  # запустить автономный эксперимент
    )  # конец описания аргумента
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])  # зёрна, по умолчанию 0 1 2
    p.add_argument("--out-dir", default="hagi_growing_math_results")  # каталог результатов
    p.add_argument(  # --query: операция и аргументы (начало описания)
        "--query", nargs="+", metavar=("OP", "ARG"), help="query saved model: OP ARG..."  # имя операции и её аргументы
    )  # конец описания аргумента
    p.add_argument(  # --expr (начало описания аргумента)
        "--expr",  # вычислить безопасное выражение, например x*y + gcd(x,y)
        help="evaluate a safe arithmetic/logic expression, e.g. 'x*y + gcd(x,y)'",  # конец описания аргумента
    )  # --var (начало описания аргумента)
    p.add_argument(  # режим добавления значения
        "--var",  # по умолчанию пустой список
        action="append",  # присваивание переменной вида x=7 или x=0.5
        default=[],  # по умолчанию пустой список значений
        help="expression variable assignment, e.g. x=7 or x=0.5",
    )  # режим: exact или neural, по умолчанию exact
    p.add_argument("--mode", choices=["exact", "neural"], default="exact")
    p.add_argument(  # --model (начало описания аргумента)
        "--model", default="hagi_growing_math_results/hagi_growing_math_model.json"  # путь к файлу модели по умолчанию
    )  # конец описания аргумента
    args = p.parse_args()  # разобрать аргументы
    if args.self_test:  # флаг самопроверки
        self_test()  # запустить самопроверку
    elif args.query:  # иначе, если задан запрос
        if len(args.query) < 2:  # меньше двух элементов — ошибка
            p.error("--query needs an operation name and at least one integer argument")  # сообщение об ошибке
        query_model(args.model, args.query[0], args.query[1:], args.mode)  # запрос: имя, остальные аргументы, режим
    elif args.expr:  # иначе, если задано выражение
        vals = {}  # значения переменных
        for item in args.var:  # по элементам --var
            if "=" not in item:  # нет знака равенства — ошибка
                p.error(f"invalid --var {item!r}; expected name=integer")  # сообщение об ошибке
            k, v = item.split("=", 1)  # разбить на имя и значение по первому «=»
            raw = v.strip()  # значение без пробелов
            vals[k.strip()] = (  # присвоить переменной (начало выражения)
                float(raw) if any(ch in raw.lower() for ch in (".", "e")) else int(raw)  # дробное, если есть точка или e, иначе целое
            )  # конец выражения
        model = GrowingMathProcessor.load(args.model)  # загрузить модель
        print(json.dumps(model.evaluate_expression(args.expr, **vals), indent=2))  # напечатать JSON результата выражения
    else:  # иначе — полный эксперимент
        result = run_experiment(args.seeds, args.out_dir)  # провести эксперимент с заданными зёрнами и каталогом
        print(json.dumps(result["aggregate"], indent=2))  # напечатать агрегаты
        print(f"Report: {Path(args.out_dir) / 'hagi_growing_math_report.md'}")  # напечатать путь к текстовому отчёту
        print(f"Model:  {Path(args.out_dir) / 'hagi_growing_math_model.json'}")  # напечатать путь к модели



import sys as _sys  # псевдоним модуля sys под именем _sys

# ==============================================================================
# ЧАСТЬ II — оптимальный слой v7.4 (привязки имён для общего пространства имён)
# ==============================================================================
h5 = _sys.modules[__name__]  # h5 — ссылка на этот же модуль
hagi_v7 = _sys.modules[__name__]  # hagi_v7 — ссылка на этот же модуль

V5Processor = GrowingMathProcessor  # V5Processor — процессор V5
# Слой V7 импортировал эти имена слоя V5 под псевдонимами или напрямую:
Expr = Expr  # то же имя Expr (тождественная привязка для ясности)
Program = Program  # то же имя Program
Expert = Expert  # то же имя Expert
GrowthEvent = GrowthEvent  # то же имя GrowthEvent
TaskSpec = TaskSpec  # то же имя TaskSpec
NeuralFiber = NeuralFiber  # то же имя NeuralFiber
SafeExpression = SafeExpression  # то же имя SafeExpression
OPS = OPS  # то же имя OPS
_make_task_suite = make_task_suite  # _make_task_suite — набор задач
_bit_not8 = _bit_not8  # то же имя _bit_not8
_bit_nand8 = _bit_nand8  # то же имя _bit_nand8
_bit_nor8 = _bit_nor8  # то же имя _bit_nor8
_bit_xnor8 = _bit_xnor8  # то же имя _bit_xnor8
_pow_int = _pow_int  # то же имя _pow_int
_sign = _sign  # то же имя _sign
candidate_programs = candidate_programs  # то же имя candidate_programs

"""Слой V7.4 (исследовательский): точный многомерный символьный обучатель на конечных областях.

            Отличия от прототипа v7.2:
            1. Точные дешёвые шаблоны и индукция рациональных многочленов выполняются до ограниченного
               поиска по общей грамматике.
            2. Пробные значения служат только ситом отсева; разные деревья не сливаются лишь потому,
               что совпадают на малой выборке. Общий поиск упорядочивает кандидатов по явной длине
               префиксного кода, но грамматика ограничена и бюджет поиска может оборвать его:
               глобальный оптимум длины описания не утверждается.
            3. Повторное использование библиотеки учитывается как справочная стоимость, а не
               произвольная скидка на длину кода. Объявленный битовый код — мера размера модели, не байты файла.
            4. Фильтр на int64 консервативен: он может лишь отклонить внутри ограниченной оболочки
               операций и входов. Все сомнительные случаи считаются скалярными целыми Python;
               полная скалярная проверка конечной области остаётся окончательной инстанцией закрепления.
            5. Обучение нейрослоя включается явно и заведомо приближённо. Путь на CPU не импортирует
               PyTorch; поддержка CUDA загружается лениво.
            6. Добыча теорем проверяет только допустимые входы и требует замкнутости перед
               композицией унарных функций. Синтетическая композиция остаётся внутри заверенных
               областей экспертов-родителей.

            Это исследовательский прототип для конечных областей, а не общая модель интеллекта.
            Закрепление ограничено полной объявленной конечной областью; вне её прогноз — отказ.
        Зависимости: Python ≥ 3.10, NumPy (PyTorch необязателен)."""

from concurrent.futures import ProcessPoolExecutor  # пул процессов для параллельной проверки
from typing import Any, Iterator, Optional, Sequence  # типы для аннотаций


V7_VERSION = "7.4.0-research"  # версия слоя V7
SIEVE_SIZE = 5  # размер сита: число пробных точек для дешёвого отсева
DEFAULT_NODE_BUDGET = 20_000  # бюджет узлов: не более 20 000 кандидатов на задачу
DEFAULT_POP_BUDGET = 50_000  # бюджет выборок из фронтира: не более 50 000 (предел вычислений)
VALIDATION_CHUNK = 64  # размер пачки кандидатов для векторного фильтра
MAX_REUSE_EXPANSIONS = 8  # не более 8 подстановок экспертов на одну выборку из фронтира



# --------------------- векторный пакетный предварительный фильтр (numpy) ----------------------
_MAG_LIMIT = 1 << 62  # предел модуля 2^62: за ним фильтр «не знает», решает скалярная проверка


def _vec_binary(op: str, a: np.ndarray, b: np.ndarray):
    """Векторная бинарная операция над массивами int64.

                op: имя операции; a, b: массивы одинаковой формы. Возврат: (значения, маска ok).
            ok = False там, где скалярная реализация бросает ошибку (деление на 0, степень вне диапазона)."""
    if op == "add":  # сложение
        return a + b, np.ones(a.shape, bool)  # a + b, все дорожки допустимы
    if op == "sub":  # вычитание
        return a - b, np.ones(a.shape, bool)  # разность a − b, все дорожки допустимы
    if op == "mul":  # умножение
        return a * b, np.ones(a.shape, bool)  # произведение a·b, все дорожки допустимы
    if op in ("floordiv", "mod"):  # деление вниз или остаток
        ok = b != 0  # маска допустимости: b ≠ 0
        safe_b = np.where(ok, b, np.ones_like(b))  # заменить нулевой делитель на 1, чтобы не делить на 0
        if op == "floordiv":  # деление вниз
            return np.floor_divide(a, safe_b), ok  # a // safe_b и маска ok
        return np.mod(a, safe_b), ok  # a mod safe_b и маска ok
    if op == "gcd":  # наибольший общий делитель
        return np.gcd(a, b), np.ones(a.shape, bool)  # наибольший общий делитель gcd(a, b)
    if op == "lcm":  # наименьшее общее кратное
        ok = (a != 0) & (b != 0)  # ok: оба не равны 0
        sa = np.where(ok, a, np.ones_like(a))  # нулевые a заменить на 1
        sb = np.where(ok, b, np.ones_like(b))  # нулевые b заменить на 1
        return np.abs(sa * sb) // np.gcd(sa, sb), ok  # lcm = |a·b| / gcd(a, b) и маска ok
    if op == "min":  # минимум
        return np.minimum(a, b), np.ones(a.shape, bool)  # минимум min(a, b)
    if op == "max":  # максимум
        return np.maximum(a, b), np.ones(a.shape, bool)  # максимум max(a, b)
    if op == "eq":  # равно
        return (a == b).astype(np.int64), np.ones(a.shape, bool)  # (a = b) как int64
    if op == "ne":  # не равно
        return (a != b).astype(np.int64), np.ones(a.shape, bool)  # (a ≠ b) как int64
    if op == "lt":  # меньше
        return (a < b).astype(np.int64), np.ones(a.shape, bool)  # (a < b) как int64
    if op == "le":  # меньше или равно
        return (a <= b).astype(np.int64), np.ones(a.shape, bool)  # (a ≤ b) как int64
    if op == "gt":  # больше
        return (a > b).astype(np.int64), np.ones(a.shape, bool)  # (a > b) как int64
    if op == "ge":  # больше или равно
        return (a >= b).astype(np.int64), np.ones(a.shape, bool)  # (a ≥ b) как int64
    if op == "bit_and":  # побитовое «и»
        return a & b, np.ones(a.shape, bool)  # побитовое «и»: a & b
    if op == "bit_or":  # побитовое «или»
        return a | b, np.ones(a.shape, bool)  # побитовое «или»: a | b
    if op == "bit_xor":  # исключающее «или»
        return a ^ b, np.ones(a.shape, bool)  # исключающее «или»: a ^ b
    if op == "bit_nand8":  # не-и в байте
        return (~(a & b)) & 0xFF, np.ones(a.shape, bool)  # не-и в байте: ~(a & b) & 0xFF
    if op == "bit_nor8":  # не-или в байте
        return (~(a | b)) & 0xFF, np.ones(a.shape, bool)  # не-или в байте: ~(a | b) & 0xFF
    if op == "bit_xnor8":  # эквивалентность в байте
        return (~(a ^ b)) & 0xFF, np.ones(a.shape, bool)  # эквивалентность в байте: ~(a ^ b) & 0xFF
    if op == "pow_int":  # целая степень
        ok = (b >= 0) & (b <= 8)  # маска допустимости: 0 ≤ b ≤ 8
        sb = np.where(ok, b, np.zeros_like(b))  # недопустимый показатель заменить нулём
        return a**sb, ok  # a ** sb и маска ok
    raise ValueError(f"no vectorized op: {op}")  # ошибка: нет векторной операции


def _vec_unary(op: str, a: np.ndarray):
    """Векторная унарная операция над массивом int64.

            op: имя операции; a: массив. Возврат: (значения, маска ok — всегда все истинны)."""
    if op == "neg":  # отрицание
        return -a, np.ones(a.shape, bool)  # отрицание: −a
    if op == "abs":  # модуль
        return np.abs(a), np.ones(a.shape, bool)  # модуль: |a|
    if op == "sign":  # знак
        return np.sign(a).astype(np.int64), np.ones(a.shape, bool)  # sign(a) как int64
    if op == "bit_not8":  # побитовое «не» в байте
        return (~a) & 0xFF, np.ones(a.shape, bool)  # инверсия младших 8 бит: ~a & 0xFF
    if op == "clamp3":  # ограничение отрезком [−3, 3]
        return np.clip(a, -3, 3), np.ones(a.shape, bool)  # ограничение отрезком: clip(a, −3, 3)
    if op == "square":  # квадрат
        return a * a, np.ones(a.shape, bool)  # квадрат: a·a
    if op == "cube":  # куб
        return a * a * a, np.ones(a.shape, bool)  # куб: a·a·a
    raise ValueError(f"no vectorized unary op: {op}")  # ошибка: нет векторной унарной операции


def _vec_eval_expr(e: Expr, cols: list[np.ndarray]):
    """Вычислить выражение над столбцами-массивами.

                e: выражение; cols: столбец значений каждой переменной. Возврат: (значения, маска ok, маска big).
                ok = False там, где операция недопустима; big = True там, где модуль превысил безопасный предел:
            тогда вызывающий обязан вернуться к скалярной проверке, а не верить векторному вердикту."""
    if e.op == "var":  # узел-переменная
        return cols[int(e.value or 0)], None, None  # столбец с индексом value, без масок
    if e.op == "const":  # узел-константа
        v = np.full(cols[0].shape, int(e.value), dtype=np.int64)  # массив, заполненный константой, типа int64
        return v, None, None  # вернуть массив, без масок
    vals = []  # значения аргументов
    oks = []  # маски допустимости аргументов
    bigs = []  # маски превышения предела аргументов
    for a in e.args:  # по аргументам
        v, ok, big = _vec_eval_expr(a, cols)  # рекурсивно вычислить аргумент
        vals.append(v)  # добавить значение
        if ok is not None:  # если маска есть — добавить её
            oks.append(ok)  # добавить маску допустимости аргумента
        if big is not None:  # если есть маска превышения предела
            bigs.append(big)  # добавить её
    if e.op == "absdiff":  # модуль разности
        v, ok = np.abs(vals[0] - vals[1]), None  # v = |v0 − v1|, без маски недопустимости
    elif e.op == "fma":  # умножение-сложение
        v, ok = vals[0] * vals[1] + vals[2], None  # v = v0·v1 + v2, без маски недопустимости
    elif e.op == "clamp":  # ограничение диапазоном
        v, ok = np.minimum(np.maximum(vals[0], vals[1]), vals[2]), None  # v = min(max(v0, v1), v2), без маски
    elif e.op == "median3":  # медиана трёх
        v, ok = np.sort(np.stack(vals), axis=0)[1], None  # отсортировать столбец из трёх значений и взять среднее (индекс 1)
    elif e.op == "clamp3":  # ограничение отрезком [−3, 3]
        v, ok = np.clip(vals[0], -3, 3), None  # v = clip(v0, −3, 3), без маски
    elif e.op == "mul3":  # произведение трёх
        v, ok = vals[0] * vals[1] * vals[2], None  # v = v0·v1·v2, без маски
    elif len(vals) == 2 and e.op in OPS:  # два аргумента и операция из таблицы OPS
        v, ok = _vec_binary(e.op, vals[0], vals[1])  # векторная бинарная операция
    elif len(vals) == 1:  # один аргумент
        v, ok = _vec_unary(e.op, vals[0])  # векторная унарная операция
    else:  # прочее
        raise ValueError(f"unsupported expression opcode: {e.op}")  # ошибка: неподдерживаемая операция
    big = np.abs(v) > _MAG_LIMIT  # маска превышения предела: |v| > 2^62
    if bigs:  # если есть маски аргументов
        big = big | bigs[0]  # объединить с первой
        for bb in bigs[1:]:  # по остальным маскам аргументов
            big = big | bb  # объединить маску big с маской аргумента bb
    if ok is None:  # если маски допустимости нет
        ok = np.ones(v.shape, bool)  # все дорожки допустимы
    for child_ok in oks:  # по маскам аргументов
        ok = ok & child_ok  # ok = ok и маска аргумента
    # Каждый аргумент обязан быть определён на дорожке, чтобы выражение имело
    # на ней ту же семантику, что скалярный Python.
    return v, ok, big  # вернуть значения, маску допустимости, маску превышения


def _batch_fits_unchecked(
    programs: Sequence[Program], X: np.ndarray, Y: np.ndarray, device: str = "cpu"
) -> list[Optional[bool]]:  # возврат — список «подходит / не подходит / неясно» по программам
    """Пакетный надёжный предварительный фильтр соответствия программ примерам X -> Y.

                programs: программы; X: матрица входов (N, arity); Y: столбец ответов; device: "cpu" или "cuda...".
                Возврат: список по программам: True — доказанно подходит, False — доказанно не подходит
                (есть дорожка, где скалярное вычисление бросает ошибку), None — неясно (решает скалярная проверка).
            На CUDA при доступном torch используется _batch_fits_torch."""
    """Batched sound prefilter: True = provably fits X->Y, False = provably
    fails (a lane where the scalar eval would raise), None = unsure (must be
    settled by the scalar authority check)."""
    out: list[Optional[bool]] = []  # результат по программам
    if device.startswith("cuda") and _ensure_torch():  # устройство cuda и torch доступен
        return _batch_fits_torch(programs, X, Y, device)  # векторный фильтр на torch
    for p in programs:  # по программам
        if p.expr is None:  # у программы нет выражения (булева маска)
            out.append(None)  # неясно
            continue  # следующая программа
        try:  # попытка векторной оценки
            cols = [np.ascontiguousarray(X[:, i]) for i in range(X.shape[1])]  # столбцы входов — по одному на переменную
            v, ok, big = _vec_eval_expr(p.expr, cols)  # векторная оценка выражения
            if not bool(np.all(ok)) or bool(np.any(big)):  # есть недопустимая дорожка или превышение предела
                out.append(None)  # неясно
                continue  # следующая программа
            out.append(bool(np.array_equal(v, Y)))  # истина, если значения совпадают с Y на всех дорожках
        except (ValueError, OverflowError):  # ошибка значения или переполнение
            out.append(None)  # неясно
    return out  # вернуть результаты



# --------------------------- необязательный путь на GPU (torch) --------------------------
_TORCH_OK = False  # torch доступен — пока не проверено, ложь
_TORCH_CHECKED = False  # наличие torch уже проверено — ложь
_torch = None  # сам модуль torch, пока нет


def _ensure_torch() -> bool:
    """Загрузить PyTorch только когда явно запрошена работа на CUDA.

            Возврат: истина, если torch импортировался. Результат запоминается в глобальных переменных."""
    global _TORCH_OK, _TORCH_CHECKED, _torch  # глобальные переменные модуля
    if _TORCH_CHECKED:  # наличие torch уже проверено
        return _TORCH_OK  # вернуть запомненный результат
    _TORCH_CHECKED = True  # отметить, что проверка выполнена
    try:  # попытка импорта
        import importlib  # импорт модуля importlib

        _torch = importlib.import_module("torch")  # импортировать torch
        _TORCH_OK = True  # torch доступен
    except ImportError:  # torch не установлен
        _torch = None  # модуля нет
        _TORCH_OK = False  # torch недоступен
    return _TORCH_OK  # вернуть доступность torch


def _batch_fits_torch(
    programs: Sequence[Program], X: np.ndarray, Y: np.ndarray, device: str  # programs, X, Y — программы, входы, ответы; device — устройство
) -> list[Optional[bool]]:  # возврат — список «подходит / не подходит / неясно» по программам
    """Пакетный фильтр на GPU. Тот же договор, что у numpy-пути фильтра.

            Точен для int64; семантика дорожек «неясно» совпадает с numpy-путем."""
    dev = _torch.device(device)  # устройство torch
    Xt = _torch.from_numpy(X).to(dev)  # X как тензор на устройстве
    Yt = _torch.from_numpy(Y).to(dev)  # Y как тензор на устройстве
    cols = [Xt[:, i].contiguous() for i in range(X.shape[1])]  # столбцы входов
    one = _torch.ones_like(cols[0])  # тензор из единиц той же формы

    def teval(e: Expr):  # вложенная функция teval: вычислить выражение тензорно (возврат — значения, маска ok, маска big)
        if e.op == "var":  # узел-переменная
            return cols[int(e.value or 0)], None, None  # столбец с индексом value, без масок
        if e.op == "const":  # узел-константа
            v = _torch.full_like(cols[0], int(e.value))  # тензор, заполненный константой
            return v, None, None  # вернуть, без масок
        vs, oks, bigs = [], [], []  # значения, маски допустимости, маски превышения аргументов
        for a in e.args:  # по аргументам
            v, ok, big = teval(a)  # рекурсивно вычислить аргумент
            vs.append(v)  # добавить значение
            if ok is not None:  # если маска ok есть
                oks.append(ok)  # добавить её
            if big is not None:  # если маска big есть
                bigs.append(big)  # добавить её
        op = e.op  # имя операции
        if op == "absdiff":  # модуль разности
            v, ok = (vs[0] - vs[1]).abs(), None  # модуль разности |v0 − v1|, без маски
        elif op == "fma":  # умножение-сложение
            v, ok = vs[0] * vs[1] + vs[2], None  # произведение-сложение v0·v1 + v2, без маски
        elif op == "clamp":  # ограничение диапазоном
            v, ok = torch_clamp = (  # v = min(max(v0, v1), v2) (начало кортежа; результат дублируется в torch_clamp)
                _torch.minimum(_torch.maximum(vs[0], vs[1]), vs[2]),  # минимум из максимума и v2
                None,  # маски нет
            )  # конец кортежа
        elif op == "median3":  # медиана трёх
            v, ok = _torch.sort(_torch.stack(vs), dim=0).values[1], None  # средний элемент отсортированного столбца
        elif op == "clamp3":  # ограничение отрезком [−3, 3]
            v, ok = _torch.clamp(vs[0], -3, 3), None  # ограничение clamp(v0, −3, 3), без маски
        elif op == "mul3":  # произведение трёх
            v, ok = vs[0] * vs[1] * vs[2], None  # произведение v0·v1·v2, без маски
        elif op == "add":  # сложение
            v, ok = vs[0] + vs[1], None  # сумма v0 + v1, без маски
        elif op == "sub":  # вычитание
            v, ok = vs[0] - vs[1], None  # разность v0 − v1, без маски
        elif op == "mul":  # умножение
            v, ok = vs[0] * vs[1], None  # произведение v0·v1, без маски
        elif op in ("floordiv", "mod"):  # деление вниз или остаток
            b0 = vs[1]  # делитель b0
            ok = b0 != 0  # маска: b0 ≠ 0
            sb = _torch.where(ok, b0, one)  # заменить нулевой делитель на 1
            v = (  # вычислить (начало условия)
                _torch.div(vs[0], sb, rounding_mode="floor")  # деление с округлением вниз
                if op == "floordiv"  # если floordiv
                else _torch.remainder(vs[0], sb)  # иначе остаток
            )  # конец условия
        elif op == "gcd":  # наибольший общий делитель
            v, ok = _torch.gcd(vs[0], vs[1]), None  # наибольший общий делитель gcd(v0, v1), без маски
        elif op == "lcm":  # наименьшее общее кратное
            a0, b0 = vs[0], vs[1]  # a0, b0 — операнды
            ok = (a0 != 0) & (b0 != 0)  # маска: оба не нули
            sa = _torch.where(ok, a0, one)  # нулевые a0 заменить на 1
            sb = _torch.where(ok, b0, one)  # нулевые b0 заменить на 1
            v = _torch.abs(sa * sb) // _torch.gcd(sa, sb)  # наименьшее общее кратное |a0·b0| // gcd(a0, b0)
        elif op == "min":  # минимум
            v, ok = _torch.minimum(vs[0], vs[1]), None  # минимум minimum(v0, v1), без маски
        elif op == "max":  # максимум
            v, ok = _torch.maximum(vs[0], vs[1]), None  # максимум maximum(v0, v1), без маски
        elif op in ("eq", "ne", "lt", "le", "gt", "ge"):  # сравнения
            fn = {  # таблица «имя -> функция torch» (начало словаря)
                "eq": _torch.eq,  # равно
                "ne": _torch.ne,  # не равно
                "lt": _torch.lt,  # меньше
                "le": _torch.le,  # меньше или равно
                "gt": _torch.gt,  # больше
                "ge": _torch.ge,  # больше или равно
            }[op]  # выбрать функцию по имени
            v, ok = fn(vs[0], vs[1]).to(_torch.int64), None  # применить и привести к int64
        elif op == "bit_and":  # побитовое «и»
            v, ok = vs[0] & vs[1], None  # побитовое «и» v0 & v1, без маски
        elif op == "bit_or":  # побитовое «или»
            v, ok = vs[0] | vs[1], None  # побитовое «или» v0 | v1, без маски
        elif op == "bit_xor":  # исключающее «или»
            v, ok = vs[0] ^ vs[1], None  # исключающее «или» v0 ^ v1, без маски
        elif op == "bit_nand8":  # не-и в байте
            v, ok = (~(vs[0] & vs[1])) & 0xFF, None  # не-и в байте ~(v0 & v1) & 0xFF, без маски
        elif op == "bit_nor8":  # не-или в байте
            v, ok = (~(vs[0] | vs[1])) & 0xFF, None  # не-или в байте ~(v0 | v1) & 0xFF, без маски
        elif op == "bit_xnor8":  # эквивалентность в байте
            v, ok = (~(vs[0] ^ vs[1])) & 0xFF, None  # эквивалентность ~(v0 ^ v1) & 0xFF, без маски
        elif op == "pow_int":  # целая степень
            b0 = vs[1]  # показатель b0
            ok = (b0 >= 0) & (b0 <= 8)  # маска: 0 ≤ b0 ≤ 8
            sb = _torch.where(ok, b0, _torch.zeros_like(b0))  # недопустимый показатель заменить нулём
            v = vs[0] ** sb  # степень v0 ** sb
        elif op == "neg":  # отрицание
            v, ok = -vs[0], None  # отрицание −v0, без маски
        elif op == "abs":  # модуль
            v, ok = vs[0].abs(), None  # модуль |v0|, без маски
        elif op == "sign":  # знак
            v, ok = _torch.sign(vs[0]).to(_torch.int64), None  # sign(v0) как int64
        elif op == "bit_not8":  # побитовое «не» в байте
            v, ok = (~(vs[0])) & 0xFF, None  # инверсия байта ~v0 & 0xFF, без маски
        elif op == "clamp3":  # ограничение отрезком
            v, ok = _torch.clamp(vs[0], -3, 3), None  # ограничение clamp(v0, −3, 3), без маски
        elif op == "square":  # квадрат
            v, ok = vs[0] * vs[0], None  # квадрат v0·v0, без маски
        elif op == "cube":  # куб
            v, ok = vs[0] * vs[0] * vs[0], None  # куб v0·v0·v0, без маски
        else:  # прочее
            raise ValueError(f"unsupported torch opcode: {op}")  # ошибка: нет такой операции torch
        big = v.abs() > _MAG_LIMIT  # маска превышения: |v| > 2^62
        for bb in bigs:  # по маскам аргументов
            big = big | bb  # объединить маску big с маской аргумента bb
        if ok is None:  # если маски допустимости нет
            ok = _torch.ones_like(big)  # все дорожки допустимы
        for child_ok in oks:  # по маскам аргументов
        # Провести сбои областей аргументов через всё дерево выражения.
            ok = ok & child_ok  # ok = ok и маска аргумента
        return v, ok, big  # вернуть значения, маску ok, маску big

    results: list[Optional[bool]] = []  # результаты по программам
    for p in programs:  # по программам
        if p.expr is None:  # нет выражения
            results.append(None)  # неясно
            continue  # следующая программа
        try:  # попытка тензорной оценки
            v, ok, big = teval(p.expr)  # значения, маска ok, маска big
            if ok is None:  # если маска ok не задана
                ok = _torch.ones(v.shape, dtype=_torch.bool, device=v.device)  # все дорожки допустимы
            if bool((~ok).any()) or bool(big is not None and big.any()):  # есть недопустимая дорожка или превышение предела
                results.append(None)  # неясно
                continue  # следующая программа
            results.append(bool(_torch.equal(v, Yt)))  # истина, если значения равны Yt
        except (ValueError, RuntimeError, OverflowError):  # ошибка torch или переполнение
            results.append(None)  # неясно
    return results  # вернуть результаты



# --------------------- длина кода и договор безопасного фильтра -----------------

def _gamma_bits(n: int) -> int:
    """Длина гамма-кода Элиаса для положительного целого n.

            n: положительное целое. Формула: 2·(bit_length(n) − 1) + 1. Для n < 1 — ValueError."""
    n = int(n)  # привести к целому
    if n < 1:  # n < 1 — ошибка
        raise ValueError("gamma code accepts positive integers")  # сообщение об ошибке
    return 2 * (n.bit_length() - 1) + 1  # длина кода = 2·(число бит − 1) + 1


def _zigzag(n: int) -> int:
    """Зигзаг-кодирование целого: неотрицательные на чётные, отрицательные на нечётные.

            n: целое. Возврат: 2n при n ≥ 0, иначе −2n − 1."""
    n = int(n)  # привести к целому
    return 2 * n if n >= 0 else -2 * n - 1  # 2n, если n ≥ 0, иначе −2n − 1


_TOKEN_BITS = 6  # ширина токена операции: 6 бит (64 зарезервированных токена)


def _expr_description_bits(expr: Expr) -> int:
    """Длина описания выражения в битах по конкретному префиксному коду.

                В отличие от подсчёта узлов дерева, константы тоже стоят бит. Каждый узел начинается с 6-битного токена;
                индексы переменных, целые константы со знаком и коэффициенты многочлена кодируются самоделимитирующимися
                гамма-пayload-ами: gamma(x) = 2·(bit_length(x) − 1) + 1, где x = zigzag(значение) + 1 > 0.
            Это объявленное кодирование для данного языка выражений, а не колмогоровская сложность. Возврат: биты."""
    if expr.op == "var":  # узел-переменная
        return _TOKEN_BITS + _gamma_bits(int(expr.value or 0) + 1)  # 6 бит + gamma(индекс + 1)
    if expr.op == "const":  # узел-константа
        return _TOKEN_BITS + _gamma_bits(_zigzag(int(expr.value)) + 1)  # 6 бит + gamma(zigzag(значение) + 1)
    if expr.op == "poly":  # многочлен одной переменной
        coeffs = tuple(int(v) for v in (expr.value or ()))  # коэффициенты как кортеж целых
        payload = _gamma_bits(len(coeffs) + 1)  # payload = gamma(число коэффициентов + 1)
        payload += sum(_gamma_bits(_zigzag(v) + 1) for v in coeffs)  # payload += Σ gamma(zigzag(c) + 1) по коэффициентам
        return _TOKEN_BITS + payload + sum(_expr_description_bits(a) for a in expr.args)  # 6 бит + payload + сумма бит аргументов
    if expr.op == "poly_nd":  # многочлен многих переменных
        terms = tuple(expr.value or ())  # члены (коэффициент, показатели)
        payload = _gamma_bits(len(terms) + 1)  # payload = gamma(число членов + 1)
        for coefficient, powers in terms:  # по членам
            payload += _gamma_bits(_zigzag(int(coefficient)) + 1)  # payload += gamma(zigzag(коэффициент) + 1)
            payload += _gamma_bits(len(powers) + 1)  # payload += gamma(число показателей + 1)
            payload += sum(_gamma_bits(int(power) + 1) for power in powers)  # payload += Σ gamma(показатель + 1)
        return _TOKEN_BITS + payload  # 6 бит + payload
    return _TOKEN_BITS + sum(_expr_description_bits(a) for a in expr.args)  # прочий узел: 6 бит + сумма бит аргументов


def _program_description_bits(program: Program) -> int:
    """Длина описания программы в битах.

            Булева маска: 6 бит токена + 4 бита маски; нет выражения — 10^9; иначе длина описания выражения."""
    if program.venn_mask is not None:  # булева маска
        return _TOKEN_BITS + 4  # 10 бит
    if program.expr is None:  # нет выражения
        return 10**9  # 10^9 — заведомо неприменимая длина
    return _expr_description_bits(program.expr)  # длина описания выражения


_VECTOR_SAFE_OPS = {  # операции, безопасные для векторного фильтра int64
    "neg",  # отрицание
    "abs",  # модуль
    "sign",  # знак
    "bit_not8",  # инверсия байта
    "clamp3",  # ограничение отрезком [−3, 3]
    "add",  # сложение
    "sub",  # вычитание
    "mul",  # умножение
    "floordiv",  # деление вниз
    "mod",  # остаток
    "gcd",  # наибольший общий делитель
    "lcm",  # наименьшее общее кратное
    "min",  # минимум
    "max",  # максимум
    "eq",  # равно
    "ne",  # не равно
    "lt",  # меньше
    "le",  # меньше или равно
    "gt",  # больше
    "ge",  # больше или равно
    "bit_and",  # побитовое «и»
    "bit_or",  # побитовое «или»
    "bit_xor",  # исключающее «или»
    "bit_nand8",  # не-и в байте
    "bit_nor8",  # не-или в байте
    "bit_xnor8",  # эквивалентность в байте
    "absdiff",  # модуль разности
    "clamp",  # ограничение диапазоном
    "median3",  # медиана трёх
}
_VECTOR_INPUT_LIMIT = 1 << 7  # предел модуля листа: 2^7 = 128
_VECTOR_MAX_DEPTH = 3  # наибольшая глубина вложенности операций: 3


def _vector_prefilter_safe(expr: Expr, depth: int = 0) -> bool:
    """Доказать консервативную оболочку int64 для векторного подмножества языка выражений.

                expr: выражение; depth: текущая глубина. При не более 3 уровнях операций и |лист| ≤ 128 бинарное выражение
                имеет не более 8 листьев; даже произведение или наименьшее общее кратное всех листьев ≤ 2^56,
                аддитивные комбинации остаются ниже 2^63. Степени, кубы, fma, mul3 и произвольные константы
            намеренно считаются скалярными целыми Python. Возврат: истина, если выражение в оболочке."""
    if depth > _VECTOR_MAX_DEPTH:  # глубина больше 3 — небезопасно
        return False  # ложь
    if expr.op == "var":  # переменная
        return True  # безопасна
    if expr.op == "const":  # константа
        return abs(int(expr.value or 0)) <= _VECTOR_INPUT_LIMIT  # безопасна, если |значение| ≤ 128
    if expr.op not in _VECTOR_SAFE_OPS:  # операция не из безопасного набора
        return False  # ложь
    return all(_vector_prefilter_safe(a, depth + 1) for a in expr.args)  # все аргументы безопасны на глубине + 1


def batch_fits_safe(
    programs: Sequence[Program],  # programs — программы-кандидаты
    X: Optional[np.ndarray],  # X — входы или None
    Y: Optional[np.ndarray],  # Y — ответы или None
    device: str = "cpu",
) -> list[Optional[bool]]:  # возврат — список «подходит / не подходит / неясно» по программам
    """Консервативный фильтр: None означает «нужна скалярная проверка».

                programs: программы; X, Y: входы и ответы или None; device: устройство. Возврат: список True/False/None.
                Через int64 в NumPy/Torch идут только ограниченные выражения; прочие кандидаты минуют быстрый путь,
                поэтому переполнение не может дать ложный отказ. Векторный результат — никогда не свидетель закрепления:
            выжившие всё равно проверяются скалярно на обучении, проверке и всей конечной области."""
    """Conservative filter: None means scalar check required.

    Only bounded expressions are sent through NumPy/Torch int64. All other
    candidates bypass the fast path, so overflow cannot create a false reject.
    A vector result is never a commit certificate; scalar Python-int checks
    still validate every possible survivor on train/val/full finite support.
    """
    if X is None or Y is None:  # нет входов или ответов
        return [None] * len(programs)  # всем программам — неясно
    if X.size and (  # есть примеры и вход вне пределов (начало условия)
        int(X.min()) < -_VECTOR_INPUT_LIMIT or int(X.max()) > _VECTOR_INPUT_LIMIT  # min(X) < −128 или max(X) > 128
    ):  # конец условия
        return [None] * len(programs)  # всем программам — неясно
    out: list[Optional[bool]] = [None] * len(programs)  # результат, изначально «неясно» для всех
    safe_indices = [  # номера программ с выражением в безопасной оболочке (начало списка)
        i  # номер i
        for i, p in enumerate(programs)  # по парам (номер, программа)
        if p.expr is not None and _vector_prefilter_safe(p.expr)  # выражение есть и фильтр безопасен
    ]  # конец списка
    if safe_indices:  # если безопасные есть
        safe_programs = [programs[i] for i in safe_indices]  # безопасные программы
        safe_results = _batch_fits_unchecked(safe_programs, X, Y, device=device)  # векторная оценка без дополнительной проверки
        for i, result in zip(safe_indices, safe_results):  # по парам (номер, результат)
            out[i] = result  # записать результат на своё место
    return out  # вернуть результаты



# ------------------------- синтез кандидатов «сначала лучший» ----------------------
UNARY_OPS = ("neg", "abs", "sign", "bit_not8", "clamp3")  # унарные операции синтеза
BINARY_OPS = (  # бинарные операции синтеза (начало кортежа)
    "add",  # сложение
    "sub",  # вычитание
    "mul",  # умножение
    "floordiv",  # деление вниз
    "mod",  # остаток
    "gcd",  # наибольший общий делитель
    "lcm",  # наименьшее общее кратное
    "min",  # минимум
    "max",  # максимум
    "eq",  # равно
    "ne",  # не равно
    "lt",  # меньше
    "le",  # меньше или равно
    "gt",  # больше
    "ge",  # больше или равно
    "bit_and",  # побитовое «и»
    "bit_or",  # побитовое «или»
    "bit_xor",  # исключающее «или»
    "bit_nand8",  # не-и в байте
    "bit_nor8",  # не-или в байте
    "bit_xnor8",  # эквивалентность в байте
    "pow_int",  # целая степень
)
CONSTS = (0, 1, -1)  # константы синтеза: 0, 1, −1


TERNARY_PRIMS = ("fma", "clamp", "median3", "mul3")  # тернарные примитивы


def iter_candidates_mdl(  # iter_candidates_mdl: перебор выражений-кандидатов в порядке длины описания
    arity: int,  # arity — число входов
    node_budget: int = DEFAULT_NODE_BUDGET,  # node_budget — предел числа выданных кандидатов
    probe_x: Sequence[Sequence[int]] = (),  # probe_x — пробные входы
    probe_y: Sequence[int] = (),  # probe_y — пробные ответы
    extra_binops: Sequence[tuple[str, Program]] = (),  # extra_binops — дополнительные пары (имя, программа) для подстановки экспертов
    pop_budget: int = 50_000,  # pop_budget — предел выборок из кучи
    deadline: Optional[float] = None,  # deadline — предельный момент времени или None
) -> Iterator[Expr]:  # возврат — генератор выражений
    """Выдавать выражения, согласные с пробами, в порядке возраста длины префиксного кода.

                Генератор обходит реализованную ограниченную грамматику: унарные операции, применение операции
                к двум копиям узла, бинарная операция с листом, тернарные примитивы над переменными
                и подстановки закреплённых экспертов. Это НЕ полный перебор всех деревьев языка;
                пределы node/pop/deadline тоже могут оборвать его. Пробные значения служат только дешёвым
                ситом согласованности: разные деревья не отбрасываются лишь за совпадение на пяти пробах.

                Алгоритм: куча с ключом (число бит описания, порядковый номер). Из кучи достаётся кратчайший
                кандидат; если его пробные значения равны probe_y (или probe_y не задано), он выдаётся;
            затем расширяются его потомки и кладутся в кучу. Значения на пробах считаются вместе с деревом."""
    probe = tuple(tuple(int(v) for v in xs) for xs in probe_x)  # probe — пробные входы как кортежи целых
    n = len(probe)  # n — число проб
    pop_budget = int(pop_budget)  # pop_budget как целое

    def apply_bin(  # вложенная функция apply_bin: бинарная операция над столбцами пробных значений a и b
        op: str, a: tuple[int, ...], b: tuple[int, ...]  # op — имя операции; a, b — наборы значений по пробам; возврат — набор результатов или None
    ) -> Optional[tuple[int, ...]]:  # конец сигнатуры
        fn = OPS.get(op)  # функция операции из OPS
        if fn is None:  # операции нет
            return None  # неопределено
        try:  # попытка
            return tuple(int(fn(a[i], b[i])) for i in range(n))  # (fn(a_i, b_i)) по всем пробам i
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления на пробе
            return None  # неопределено

    def apply_un(op: str, a: tuple[int, ...]) -> Optional[tuple[int, ...]]:  # вложенная функция apply_un: унарная операция над столбцом пробных значений a
        fn = (lambda v: min(max(v, -3), 3)) if op == "clamp3" else OPS.get(op)  # clamp3 — ограничение отрезком, иначе функция из OPS
        if fn is None:  # операции нет
            return None  # неопределено
        try:  # попытка
            return tuple(int(fn(a[i])) for i in range(n))  # (fn(a_i)) по всем пробам
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
            return None  # неопределено

    def apply_tern(  # вложенная функция apply_tern: тернарный примитив над столбцами a, b, c
        op: str, a: tuple[int, ...], b: tuple[int, ...], c: tuple[int, ...]  # op — примитив; a, b, c — наборы значений; возврат — результаты или None
    ) -> Optional[tuple[int, ...]]:  # конец сигнатуры
        try:  # попытка
            if op == "fma":  # fma: a·b + c
                return tuple(a[i] * b[i] + c[i] for i in range(n))  # (a_i·b_i + c_i) по пробам
            if op == "mul3":  # mul3: произведение трёх
                return tuple(a[i] * b[i] * c[i] for i in range(n))  # (a_i·b_i·c_i) по пробам
            if op == "clamp":  # clamp
                return tuple(min(max(a[i], b[i]), c[i]) for i in range(n))  # min(max(a_i, b_i), c_i) по пробам
            return tuple(sorted((a[i], b[i], c[i]))[1] for i in range(n))  # иначе median3: среднее из трёх по пробам
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
            return None  # неопределено

    counter = itertools.count()  # счётчик порядка поступления для развязки равных ключей
    leaves = [Expr.var(i) for i in range(arity)] + [Expr.const(c) for c in CONSTS]  # листья: переменные x_0..x_{arity−1} и константы из CONSTS
    leaf_vals: list[tuple[Expr, Optional[tuple[int, ...]]]] = []  # пары (лист, его значения на пробах)
    for leaf in leaves:  # по листьям
        if leaf.op == "var":  # переменная
            lv = tuple(row[int(leaf.value or 0)] for row in probe) if n else None  # значения переменной на пробах (нет, если проб нет)
        else:  # константа
            lv = tuple(int(leaf.value) for _ in probe) if n else None  # константа, повторённая на каждой пробе (нет, если проб нет)
        leaf_vals.append((leaf, lv))  # добавить пару

    heap: list[tuple[int, int, Expr, tuple[int, ...]]] = []  # куча: (число бит, порядковый номер, выражение, значения на пробах)
    expr_seen: set[Expr] = set()  # множество уже добавленных выражений
    target: Optional[tuple[int, ...]] = (  # целевые значения на пробах (начало выражения)
        tuple(int(v) for v in probe_y) if probe_y else None  # probe_y как кортеж целых, если пробы заданы
    )  # конец выражения; иначе None
    max_nodes = 24  # наибольшая допустимая стоимость выражения — 24 узла

    def push(expr: Expr, vals: Optional[tuple[int, ...]]) -> None:  # вложенная функция push: положить выражение в кучу, если оно допустимо
        if vals is None or expr.cost > max_nodes or expr in expr_seen:  # нет значений, стоимость больше 24 или уже добавлено — выход
            return  # выход
        expr_seen.add(expr)  # запомнить выражение
        heapq.heappush(heap, (_expr_description_bits(expr), next(counter), expr, vals))  # положить (биты описания, номер, выражение, значения) в кучу

    for leaf, lv in leaf_vals:  # по листьям с их значениями
        push(leaf, lv)  # положить лист в кучу

    if arity == 3:  # три входа
    # Атомы тернарных примитивов порождаются один раз, а не заново при каждой выборке.
        for a, va in leaf_vals[:3]:  # по первым трём листьям-переменным a
            for b, vb in leaf_vals[:3]:  # по b
                for c, vc in leaf_vals[:3]:  # по c
                    for prim in TERNARY_PRIMS:  # по тернарным примитивам
                        push(Expr(prim, (a, b, c)), apply_tern(prim, va, vb, vc))  # положить prim(a, b, c) с его пробными значениями

    generated = 0  # число выданных кандидатов
    pops = 0  # число выборок из кучи
    extra_ops = tuple(  # дополнительные операции-эксперты (начало выражения)
        sorted(  # отсортировать
            ((nm, prog) for nm, prog in extra_binops),  # пары (имя, программа)
            key=lambda pair: (_program_description_bits(pair[1]), pair[0]),  # ключ сортировки: (биты описания программы, имя)
        )[:MAX_REUSE_EXPANSIONS]  # не более MAX_REUSE_EXPANSIONS = 8 штук
    )  # конец выражения

    def expired() -> bool:  # вложенная функция expired: истёк ли срок
        return deadline is not None and time.monotonic() >= deadline  # deadline задан и монотонное время не меньше него

    while heap and generated < node_budget and pops < pop_budget:  # пока куча не пуста и не исчерпаны бюджеты узлов и выборок
        if deadline is not None and time.monotonic() >= deadline:  # срок истёк
            break  # выход из цикла
        _, _, expr, vals = heapq.heappop(heap)  # достать из кучи выражение с наименьшим числом бит
        pops += 1  # выборок + 1
        if target is None or vals == target:  # цели нет или пробные значения совпали
            generated += 1  # выданных + 1
            yield expr  # выдать выражение

        base = expr.cost  # стоимость базового узла (не используется далее)
        for op in UNARY_OPS:  # по унарным операциям
            if expired():  # срок истёк
                return  # закончить генератор
            push(Expr(op, (expr,)), apply_un(op, vals))  # потомок op(expr) с его пробными значениями

        for op in BINARY_OPS:  # по бинарным операциям
            if expired():  # срок истёк
                return  # закончить генератор
            push(Expr(op, (expr, expr)), apply_bin(op, vals, vals))  # потомок op(expr, expr)
            for leaf, lv in leaf_vals:  # по листьям
                if expired():  # срок истёк
                    return  # закончить генератор
                push(Expr(op, (expr, leaf)), apply_bin(op, vals, lv))  # потомок op(expr, лист)
                push(Expr(op, (leaf, expr)), apply_bin(op, lv, vals))  # потомок op(лист, expr)

        for _name, prog in extra_ops:  # по дополнительным экспертам (имя, программа)
            if expired():  # срок истёк
                return  # закончить генератор
            if prog.arity != 2 or prog.expr is None:  # эксперт обязан быть бинарным с выражением
                continue  # пропустить
            exp = _program_to_substitutable_expr(prog)  # выражение эксперта с переменными для подстановки
            combos = [(expr, vals, expr, vals)]  # комбинации (левый, правый):(expr, expr) и далее с листьями
            for leaf, lv in leaf_vals:  # по листьям
                combos.append((expr, vals, leaf, lv))  # комбинация (expr, лист)
                combos.append((leaf, lv, expr, vals))  # комбинация (лист, expr)
            for a, _av, b, _bv in combos:  # по комбинациям (левое подвыражение a, правое b)
                if expired():  # срок истёк
                    return  # закончить генератор
                if _subst_cost(exp, {0: a.cost, 1: b.cost}) > max_nodes:  # стоимость после подстановки больше 24
                    continue  # пропустить
                child = _substitute(exp, {0: a, 1: b})  # подставить a и b на места переменных 0 и 1
                if _expr_description_bits(child) <= _expr_description_bits(expr):  # битов описания не больше, чем у родителя
                    continue  # пропустить
                try:  # попытка
                # Эксперт, не зависящий ни от одного входа, может свернуться в более дешёвый лист;
                # не вставлять ключ с меньшей стоимостью после выборки более дорогого узла —
                # это нарушило бы порядок «сначала лучший».
                    out = tuple(int(child.eval(row)) for row in probe)  # значения потомка на пробах
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                    out = None  # значения не определены
                push(child, out)  # положить потомка в кучу


def _poly_nd_to_ast(expr: Expr) -> Expr:
    """Развернуть компактный многочлен многих переменных в обычные операции языка.

                expr: выражение poly_nd. Каждый член c·Π x_i^{p_i} собирается из констант, переменных, pow_int и mul;
            члены соединяются add. Возврат: Expr; для пустого многочлена — константа 0."""
    """Expand the compact multivariate opcode into ordinary DSL operations."""
    terms = []  # список членов-деревьев
    for coefficient, powers in expr.value or ():  # по парам (коэффициент, показатели)
        factors: list[Expr] = [Expr.const(int(coefficient))]  # множители члена: константа-коэффициент
        for index, power in enumerate(powers):  # по парам (индекс переменной, степень)
            power = int(power)  # степень как целое
            if power == 0:  # степень 0 — переменная не входит
                continue  # пропустить
            variable = Expr.var(index)  # переменная x_index
            factor = (  # множитель (начало условия)
                variable  # сама переменная
                if power == 1  # если степень 1
                else Expr("pow_int", (variable, Expr.const(power)))  # иначе pow_int(переменная, степень)
            )  # конец условия
            factors.append(factor)  # добавить множитель
        term = factors[0]  # член — первый множитель
        for factor in factors[1:]:  # по остальным множителям
            term = Expr("mul", (term, factor))  # член = mul(член, множитель)
        terms.append(term)  # добавить член
    if not terms:  # членов нет
        return Expr.const(0)  # вернуть константу 0
    result = terms[0]  # результат — первый член
    for term in terms[1:]:  # по остальным членам
        result = Expr("add", (result, term))  # результат = add(результат, член)
    return result  # вернуть результат


def _program_to_substitutable_expr(program: Program) -> Expr:
    """Выражение программы, пригодное для подстановки вместо переменных.

                program: программа. Булевы таблицы истинности не являются деревьями — ValueError.
            poly_nd разворачивается в операции. Возврат: Expr."""
    if program.expr is None:  # выражения нет
        raise ValueError("truth-table programs are not arithmetic ASTs")  # ошибка: программы-таблицы истинности не арифметические деревья
    if program.expr.op == "poly_nd":  # компактный многочлен
        return _poly_nd_to_ast(program.expr)  # развернуть его
    return program.expr  # иначе вернуть выражение как есть


def _composition_hint(spec: TaskSpec) -> Optional[Program]:
    """Построить точное дерево-свидетеля по экспертам-родителям, хранящимся в предложении задачи.

                Указание никогда не принимается на веру: learn процесса V7 проверяет его полным перебором перед
                закреплением. Оно лишь избавляет от повторного поиска уже известного дерева через большой
                общий фронт грамматики. Родители берутся из стандартных значений функции-учителя (два первых).

                Если оба родителя — булевы маски, маска результата строится по четырём входам: bit(2·x + y) = teacher(x, y).
                Если внутренний эксперт унарный, он подставляется на место x; если бинарный — на место y.
            Возврат: Program или None, если построить нельзя."""
    parents = getattr(spec.teacher, "__defaults__", None)  # стандартные значения аргументов функции-учителя
    if not parents or len(parents) < 2:  # родителей меньше двух
        return None  # указания нет
    outer, inner = parents[0], parents[1]  # внешний и внутренний родители
    if not isinstance(outer, Expert) or not isinstance(inner, Expert):  # оба обязаны быть экспертами
        return None  # иначе нет
    if outer.program.arity != 2 or spec.arity != 2:  # внешний эксперт и задача обязаны быть бинарными
        return None  # иначе нет
    if outer.program.venn_mask is not None and inner.program.venn_mask is not None:  # оба родителя — булевы маски
        mask = 0  # маска результата
        for x in (0, 1):  # x из {0, 1}
            for y in (0, 1):  # y из {0, 1}
                value = int(spec.teacher(x, y)) & 1  # значение учителя, взятое по модулю 2
                mask |= value << ((x << 1) | y)  # маска: бит с номером 2·x + y принимает значение учителя
        return Program(2, venn_mask=mask)  # вернуть булеву программу с маской
    if outer.program.expr is None or inner.program.expr is None:  # хотя бы у одного родителя нет выражения
        return None  # нет
    outer_expr = _program_to_substitutable_expr(outer.program)  # выражение внешнего эксперта
    inner_expr = _program_to_substitutable_expr(inner.program)  # выражение внутреннего эксперта
    if inner.program.arity == 1:  # внутренний эксперт унарный
        mapping = {0: inner_expr, 1: Expr.var(1)}  # подстановка: x -> внутренний, y -> переменная y
    elif inner.program.arity == 2:  # внутренний эксперт бинарный
        mapping = {0: Expr.var(0), 1: inner_expr}  # подстановка: x -> переменная x, y -> внутренний
    else:  # прочее
        return None  # нет
    return Program(2, _substitute(outer_expr, mapping))  # вернуть программу outer с подстановкой


def _substitute(e: Expr, mapping: dict[int, Expr]) -> Expr:
    """Подставить выражения вместо переменных дерева.

            e: выражение; mapping: «индекс переменной -> выражение». Возврат: новое дерево Expr; константы не меняются."""
    if e.op == "var":  # переменная
        return mapping[int(e.value or 0)]  # заменить по словарю
    if e.op == "const":  # константа
        return e  # вернуть как есть
    return Expr(e.op, tuple(_substitute(a, mapping) for a in e.args))  # тот же узел с рекурсивно подставленными аргументами


def _subst_cost(e: Expr, costs: dict[int, int]) -> int:
    """Стоимость подстановки _substitute(e, переменные -> выражения с данными стоимостями) без построения дерева.

            e: выражение; costs: «индекс переменной -> стоимость». Формула: var -> costs[i]; const -> 1; иначе 1 + Σ по аргументам."""
    if e.op == "var":  # переменная
        return costs[int(e.value or 0)]  # стоимость подставляемого выражения
    if e.op == "const":  # константа
        return 1  # стоимость 1
    return 1 + sum(_subst_cost(a, costs) for a in e.args)  # 1 + сумма стоимостей аргументов


class SynthesisStats:  # класс без декоратора dataclass
    __slots__ = ("candidates_generated", "evals", "sieve_survivors")  # имена полей без словаря атрибутов

    def __init__(self) -> None:  # создать счётчики нулями
        self.candidates_generated = 0  # порождено 0
        self.evals = 0  # вычислений 0
        self.sieve_survivors = 0  # прошедших сито 0



# ------------------------------- процессор v7 ----------------------------------

class GrowingMathProcessorV7(V5Processor):
    """Счётчики синтеза кандидатов.

                candidates_generated — число порождённых кандидатов; evals — число вычислений на пробах;
            sieve_survivors — число прошедших сито. Поля ограничены __slots__."""
    """Символьный обучатель на конечных областях с консервативными ускоряющими ограждениями.

                Наследует GrowingMathProcessor (слой V5). Отличия: кандидаты перебираются от коротких описаний к длинным
                (iter_candidates_mdl), пробные значения отсекают заведомо неверных, векторный фильтр batch_fits_safe
            только отсекает, а окончательное слово за скалярной проверкой всей конечной области."""
    """Finite-domain symbolic learner with conservative acceleration guards."""

    def __init__(  # создать процессор V7 (начало сигнатуры)
        self,  # self — экземпляр
        lookup_limit: int = 16,  # предел таблицы ответов
        mdl_lambda: float = 0.001,  # вес длины кода
        node_budget: int = 20_000,  # бюджет узлов
        pop_budget: int = 50_000,  # бюджет выборок
        device: str = "cpu",  # устройство
    ):  # конец сигнатуры
        if float(mdl_lambda) <= 0:  # вес длины кода обязан быть положительным
            raise ValueError(  # ошибка (начало вызова)
                "mdl_lambda must be positive for the exact-fit MDL tie-break"  # текст: mdl_lambda положителен для развязки по длине описания
            )  # конец вызова
        super().__init__(lookup_limit, mdl_lambda)  # инициализировать базовый процессор
        self.node_budget = int(node_budget)  # бюджет узлов как целое
        self.pop_budget = int(pop_budget)  # бюджет выборок как целое
        self.device = device  # устройство
        self.stats = SynthesisStats()  # счётчики синтеза
        self.theorems: list[Theorem] = []  # список теорем, пока пуст

    def learn(  # метод learn слоя V7
        self,  # self — экземпляр
        spec: TaskSpec,  # задача
        seed: int = 0,  # зерно
        train_fraction: float = 0.70,  # доля обучающих примеров
        poison_one_label: bool = False,  # портить ли одну метку
        train_neural: bool = False,  # обучать ли нейросеть, по умолчанию нет
        deadline: Optional[float] = None,  # предельное время
        candidate_hint: Optional[Program] = None,  # подсказка-кандидат
    ) -> GrowthEvent:  # возврат — событие роста
        """Найти и закрепить эксперта для задачи слоя V7.

                    spec: задача; seed: зерно; train_fraction: доля обучения; poison_one_label: портить метку;
                    train_neural: обучать нейросеть; deadline: предельное время; candidate_hint: подсказка-кандидат.
                    Возврат: GrowthEvent (REUSE, HOLD или COMMIT). Ход: разбиение области на обучение и проверку;
                    при испорченной метке — HOLD без поиска; попытки: готовый эксперт из библиотеки, точные шаблоны,
                    подсказка-кандидат, перебор iter_candidates_mdl с ситом проб и векторным фильтром;
                каждый выживший проверяется на обучении, проверке и всей области (validate_candidate), затем commit."""
        if spec.name in self.experts:  # задача уже изучена — событие REUSE
            return GrowthEvent(self.version, spec.name, "REUSE", "already learned")  # вернуть событие REUSE «уже изучено»
        if deadline is not None and time.monotonic() >= deadline:  # срок истёк
            return self._hold(spec, "TIME_BUDGET_EXHAUSTED", 0, 0)  # отказ «бюджет времени исчерпан»
        domain = spec.inputs()  # все допустимые входы
        if not domain:  # область пуста
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)  # отказ «пустая область»
        if len(domain) > 1:  # больше одного входа: долю обучения ограничить отрезком [0.05, 0.95]
            train_fraction = min(max(float(train_fraction), 0.05), 0.95)  # доля обучения ограничена отрезком [0.05, 0.95]
        rng = np.random.default_rng(seed)  # генератор с зерном
        perm = rng.permutation(len(domain))  # случайная перестановка номеров входов
        if spec.family == "BoolVenn":  # булевы функции
            n_train = max(1, len(domain) - 1)  # число обучающих: на один меньше области
        else:  # иначе
            n_train = max(  # n_train = max(1, min(N − 1, round(N·train_fraction))) (начало выражения)
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))  # минимум из N − 1 и округлённой доли
            )  # конец max
        train_x = [domain[int(i)] for i in perm[:n_train]]  # обучающие входы — первые n_train из перестановки
        val_x = [domain[int(i)] for i in perm[n_train:]]  # проверочные входы — остальные
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}  # ответы учителя на всей области
        train_y = [all_y[xs] for xs in train_x]  # обучающие ответы
        val_y = [all_y[xs] for xs in val_x]  # проверочные ответы
        if poison_one_label and train_y:  # портить метку и есть обучающие примеры
            train_y[0] = int(train_y[0]) + 1  # первая метка увеличена на 1
            return self._hold(  # отказ «испорченная метка» (начало вызова)
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)  # задача, причина, размеры выборок
            # Явная состязательная инъекция: никогда не искать программу по набору меток,
            # в который заведомо внесено повреждение.
            )  # конец вызова

        neural: Optional[NeuralFiber] = None  # нейросеть, пока нет

        def maybe_fit_neural() -> Optional[NeuralFiber]:  # вложенная функция maybe_fit_neural: обучить нейросеть, если разрешено
            if not train_neural or not val_x:  # обучение не разрешено или нет проверочных примеров
                return None  # вернуть None
            try:  # попытка ограниченного обучения
                return fit_fiber_guarded(  # ограниченная подгонка слоя (начало вызова)
                    train_x, train_y, val_x, val_y, seed=seed + 991, epochs=120  # зерно seed + 991, 120 эпох
                )  # конец вызова
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):  # численная ошибка
                try:  # попытка обычного обучения
                    return NeuralFiber.fit(  # подгонка слоя (начало вызова)
                        train_x, train_y, val_x, val_y, seed=seed + 991, epochs=60  # зерно seed + 991, 60 эпох
                    )  # конец вызова
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):  # снова численная ошибка
                    return None  # вернуть None

        def commit(  # вложенная функция commit: закрепить программу за задачей
            program: Program, reason: str, code_length: Optional[int] = None  # program — программа; reason — причина; code_length — длина кода или None
        ) -> GrowthEvent:  # конец сигнатуры
            nonlocal neural  # neural — переменная объёмлющей функции
            if neural is None and train_neural:  # нейросети нет, но обучение разрешено
                neural = maybe_fit_neural()  # обучить нейросеть
            bits = int(  # длина кода в битах (начало выражения)
                _program_description_bits(program)  # биты описания программы
                if code_length is None  # если code_length не задана
                else code_length  # иначе заданная длина
            )  # конец выражения
            parent_names: tuple[str, ...] = ()  # имена родителей, пока пусто
            parent_defaults = getattr(spec.teacher, "__defaults__", None)  # стандартные значения функции-учителя
            if (  # есть подсказка и минимум два родителя (начало условия)
                candidate_hint is not None  # подсказка задана
                and parent_defaults  # родители есть
                and len(parent_defaults) >= 2  # их не меньше двух
            ):  # конец условия
                outer_parent, inner_parent = parent_defaults[0], parent_defaults[1]  # внешний и внутренний родители
                if isinstance(outer_parent, Expert) and isinstance(  # оба — эксперты (начало условия)
                    inner_parent, Expert  # внутренний родитель
                ):  # конец условия
                    parent_names = (outer_parent.name, inner_parent.name)  # запомнить их имена
            expert = Expert(  # создать эксперта
                spec.name,  # имя задачи
                spec.family,  # семейство
                program,  # программа
                spec.domains,  # области входов
                spec.valid_rule,  # правило отсева
                bits,  # длина кода в битах
                neural,  # нейросеть
                len(domain),  # число проверок — размер области
                0,  # ошибок 0
                parent_names,  # имена родителей
            )  # конец описания
            self.experts[spec.name] = expert  # записать эксперта
            self.version += 1  # версия + 1
            self.total_commits += 1  # закреплений + 1
            ev = GrowthEvent(  # событие COMMIT
                self.version,  # версия
                spec.name,  # задача
                "COMMIT",  # действие COMMIT
                reason,  # причина
                program.pretty(),  # текст программы
                bits,  # длина кода
                len(train_x),  # число обучающих примеров
                len(val_x),  # число проверочных примеров
                1.0,  # точность 1.0
            )  # конец описания события
            self.growth_log.append(ev)  # добавить в журнал
            return ev  # вернуть событие

        def validate_candidate(program: Program) -> Optional[GrowthEvent]:  # вложенная функция validate_candidate: проверить кандидата и закрепить
            if deadline is not None and time.monotonic() >= deadline:  # срок истёк
                return None  # вернуть None
            try:  # попытка
                if not all(  # кандидат неверен хотя бы на одной пробе (начало all)
                    self._same(program.eval(xs), yy) for xs, yy in zip(probe_x, probe_y)  # program(xs) = y по пробам
                ):  # конец all
                    self.stats.evals += len(probe_x)  # учтены вычисления на пробах
                    return None  # кандидат отброшен
                self.stats.sieve_survivors += 1  # прошедших сито + 1
                if not self._fits(program, train_x, train_y):  # неверен на обучении
                    return None  # отброшен
                if val_x and not self._fits(program, val_x, val_y):  # есть проверочные примеры и неверен на них
                    return None  # отброшен
                if any(not self._same(program.eval(xs), all_y[xs]) for xs in domain):  # есть ошибка хотя бы на одном входе области
                # Окончательная проверка конечной области идёт на целых Python.
                    return None  # отброшен
                if program.cost > 32:  # стоимость больше 32 узлов
                    return None  # отброшен
                return commit(program, "exact-validation+full-finite-support", None)  # закрепить с причиной «точная проверка и полная конечная область»
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                return None  # отброшен

        known_items = list(sorted(self.experts.items()))  # известные эксперты по именам в порядке
        # Повторное использование: платим за ссылку на библиотеку, а не произвольную скидку 0.5x.
        for index, (name, ex) in enumerate(known_items):  # по парам (номер, имя, эксперт)
            if ex.program.arity != spec.arity:  # число входов не совпадает
                continue  # пропустить
            # Повторное использование без псевдонима допустимо, только если имеющееся заверение
            # уже покрывает всю новую область; иначе та же формула может законно получить
            # более широкую заверенную область действия.
            if not all(ex.supports(xs) for xs in domain):  # эксперт применим не ко всем входам области
                continue  # пропустить
            if not self._fits(ex.program, train_x, train_y):  # эксперт неверен на обучающих примерах
                continue  # пропустить
            if not self._fits(ex.program, val_x, val_y):
                continue
            if any(not self._same(ex.program.eval(xs), all_y[xs]) for xs in domain):
                continue
            ref_bits = _TOKEN_BITS + _gamma_bits(index + 1)
            ev = GrowthEvent(
                self.version,
                spec.name,
                "REUSE",
                f"verified-expert-coverage:{name}",
                ex.program.pretty(),
                ref_bits,
                len(train_x),
                len(val_x),
                1.0,
            )
            self.growth_log.append(ev)
            return ev

        # A tiny probe is only a rejection sieve, never a semantic equivalence key.
        probe_n = min(SIEVE_SIZE, len(train_x))
        probe_idx = rng.choice(len(train_x), size=probe_n, replace=False)
        probe_x = [train_x[int(i)] for i in probe_idx]  # входы пробы: train_x[i] по номерам i из probe_idx
        probe_y = [train_y[int(i)] for i in probe_idx]  # выходы пробы: train_y[i] по тем же номерам

        def make_train_arrays():  # целочисленные массивы int64 обучающих входов и выходов
            lo64, hi64 = -(1 << 63), (1 << 63) - 1  # границы int64: lo64 = −2^63, hi64 = 2^63 − 1
            flat = [int(v) for row in train_x for v in row] + [int(v) for v in train_y]  # все значения входов и выходов одним списком
            if any(v < lo64 or v > hi64 for v in flat):  # есть значение вне диапазона int64
                return None, None  # массивы не строятся
            return (  # пара массивов
                np.asarray(train_x, dtype=np.int64),  # X — матрица входов (N, arity), int64
                np.asarray(train_y, dtype=np.int64),  # Y — вектор выходов (N,), int64
            )

        train_X, train_Yv = make_train_arrays()  # массивы для векторного фильтра (None, None — если вне int64)

        # Сначала быстрые точные шаблоны: старое семейство кандидатов покрывает обычные
        # арифметические и логические примитивы с ничтожной ценой поиска.
        if spec.family == "BoolVenn":  # семейство — логические функции двух аргументов
            direct_candidates: Sequence[Program] = [  # прямые кандидаты — все 16 масок Венна
                Program(2, venn_mask=m) for m in range(16)  # программа с m битами таблицы истинности, m = 0..15
            ]
        else:  # прочие семейства
            direct_candidates = sorted(  # кандидаты старой грамматики, отсортированные
                candidate_programs(spec.arity),  # все программы на spec.arity входов
                key=lambda p: (_program_description_bits(p), p.pretty()),  # ключ: (число бит описания, текст программы) — дешёвые первыми
            )
        # Подсказка композиции от родителя — предложение, а не истина: она принимается
        # только после проверки на обучении, на отложенных примерах и на каждой
        # объявленной точке. Если подсказка задана явно и не прошла — удержание, а не
        # неограниченный слепой поиск другой функции.
        if candidate_hint is not None:  # задана подсказка — программа, предложенная родителем
            hinted = validate_candidate(candidate_hint)  # проверить подсказку как обычного кандидата
            if hinted is not None:  # подсказка прошла все проверки
                hinted.reason = "verified-composition-hint+full-finite-support"  # причина: проверенная подсказка композиции и полная конечная область
                if self.growth_log:  # журнал роста не пуст
                    self.growth_log[-1].reason = hinted.reason  # ту же причину записать в последнее событие журнала
                return hinted  # вернуть событие закрепления
            return self._hold(  # иначе — удержание (отказ от закрепления)
                spec, "COMPOSITION_HINT_NOT_CERTIFIED", len(train_x), len(val_x)  # причина: подсказка композиции не заверена
            )

        for program in direct_candidates:  # по прямым кандидатам
            ev = validate_candidate(program)  # проверить кандидата
            if ev is not None:  # кандидат прошёл проверки
                return ev  # закрепить и вернуть событие
            if deadline is not None and time.monotonic() >= deadline:  # задан срок и текущее время его достигло
                return self._hold(  # удержание
                    spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x)  # причина: бюджет времени исчерпан
                )

        # Замкнутая индукция многочлена идёт до общего поиска по грамматике.
        # Унарная интерполяция сохраняет компактную запись по схеме Горнера.
        if spec.arity == 1 and not spec.family.startswith("Bool"):  # один вход и не логическое семейство
            coeffs = _exact_poly_fit([xs[0] for xs in train_x], train_y, max_degree=5)  # точная рациональная интерполяция p(x) = Σ c_i·x^i степени ≤ 5
            if coeffs is not None and val_x:  # многочлен найден и есть отложенные примеры
                if all(  # верно на всех отложенных и всех входах области
                    _eval_poly(coeffs, xs[0]) == yy for xs, yy in zip(val_x, val_y)  # p(x) = y на каждой отложенной паре
                ) and all(_eval_poly(coeffs, xs[0]) == all_y[xs] for xs in domain):  # p(x) = y на каждом входе x области
                    program = Program(1, _poly_expr(coeffs))  # программа — выражение многочлена по схеме Горнера
                    if program.cost <= 32:  # стоимость не выше предела 32
                        return commit(  # закрепить
                            program, "exact-rational-polynomial+full-finite-support"  # причина: точный рациональный многочлен и полная конечная область
                        )

        # Точная многомерная интерполяция распространяет индукцию на 2–3 переменных.
        # Решается переопределённая рациональная линейная система; кандидат годен,
        # только когда вектор коэффициентов единственен и целочислен, а обычные проверки
        # на отложенных примерах и полным перебором области тоже пройдены.
        elif 2 <= spec.arity <= 3 and not spec.family.startswith("Bool"):  # 2–3 входа и не логическое семейство
            terms = _exact_multivariate_poly_fit(  # точная многомерная интерполяция p(x_1..x_n) = Σ c_m·Π x_i^{m_i}
                train_x, train_y, arity=spec.arity, max_degree=4  # по обучающим примерам, степень ≤ 4
            )
            if terms is not None and val_x:  # многочлен найден и есть отложенные примеры
                program = Program(spec.arity, _multivariate_poly_expr(terms))  # программа — выражение многочлена
                if (  # все четыре условия выполнены
                    all(program.eval(xs) == yy for xs, yy in zip(val_x, val_y))  # верно на всех отложенных примерах
                    and all(program.eval(xs) == all_y[xs] for xs in domain)  # верно на всех входах области
                    and program.cost <= 32  # стоимость не выше 32
                ):
                    return commit(  # закрепить
                        program,  # программа многочлена
                        "exact-multivariate-rational-polynomial+full-finite-support",  # причина: точный многомерный рациональный многочлен и полная область
                    )

        # Ограниченный поиск по грамматике. Векторный фильтр отсева применяется только
        # к доказанно безопасным случаям int64; остальные идут через скалярную проверку.
        if deadline is not None and time.monotonic() >= deadline:  # задан срок и текущее время его достигло
            return self._hold(spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x))  # удержание: бюджет времени исчерпан
        reuse_binops = [  # бинарные операции уже закреплённых программ той же арности
            (nm, ex.program)  # пара (имя эксперта, его программа)
            for nm, ex in self.experts.items()  # по всем экспертам
            if ex.program.arity == spec.arity  # с тем же числом входов
        ]
        candidates = (  # генератор кандидатов по возрастанию стоимости описания
            Program(spec.arity, e)  # программа из выражения e
            for e in iter_candidates_mdl(  # выражения перебора грамматики
                spec.arity,  # число входов
                self.node_budget,  # предел числа узлов
                probe_x,  # входы пробы для сита
                probe_y,  # выходы пробы
                reuse_binops,  # известные программы для повторного использования
                self.pop_budget,  # предел популяции
                deadline,  # срок
            )
        )
        chunk: list[Program] = []  # накопитель кандидатов для векторного фильтра

        def flush(buf: list[Program]) -> Optional[GrowthEvent]:  # пропустить пачку buf через векторный фильтр и скалярную проверку
            if not buf:  # пачка пуста
                return None  # ничего не найдено
            verdicts = batch_fits_safe(buf, train_X, train_Yv, device=self.device)  # вердикты по каждому кандидату: True / False / None (недоказуемо)
            for program, ok in zip(buf, verdicts):  # по парам (кандидат, вердикт)
                self.stats.candidates_generated += 1  # счётчик порождённых кандидатов + 1
                # Пропустить скалярную проверку можно только по False из защищённого
                # векторного подмножества. None всегда идёт дальше.
                if ok is False:  # векторный фильтр доказал ошибку на обучении
                    self.stats.evals += len(train_x)  # засчитать N вычислений
                    continue  # пропустить кандидата без скалярной проверки
                ev = validate_candidate(program)  # скалярная проверка (True и None оба попадают сюда)
                if ev is not None:  # кандидат закреплён
                    return ev  # вернуть событие
            return None  # в пачке годного кандидата нет

        for program in candidates:  # по всем кандидатам перебора
            chunk.append(program)  # добавить кандидата в пачку
            if len(chunk) >= VALIDATION_CHUNK:  # пачка достигла размера VALIDATION_CHUNK = 64
                ev = flush(chunk)  # обработать пачку
                if ev is not None:  # найден годный кандидат
                    return ev  # вернуть событие
                chunk = []  # очистить пачку
            if deadline is not None and time.monotonic() >= deadline:  # задан срок и текущее время его достигло
                return self._hold(  # удержание
                    spec, "TIME_BUDGET_EXHAUSTED", len(train_x), len(val_x)  # причина: бюджет времени исчерпан
                )
        ev = flush(chunk)  # обработать последнюю неполную пачку
        if ev is not None:  # найден годный кандидат
            return ev  # вернуть событие
        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))  # удержание: годной программы не найдено

    def save(self, path: str | Path) -> None:  # сохранить состояние процессора в файл path (JSON, через временный файл)
        data = {  # словарь состояния
            "format": "HAGI-GrowingMathProcessor",  # имя формата файла
            "version": V7_VERSION,  # версия формата
            "model_version": self.version,  # номер версии модели (число закреплений)
            "lookup_limit": self.lookup_limit,  # предел размера таблицы
            "mdl_lambda": self.mdl_lambda,  # вес стоимости описания λ
            "node_budget": self.node_budget,  # предел числа узлов
            "pop_budget": self.pop_budget,  # предел популяции
            "device": self.device,  # устройство вычислений
            "total_commits": self.total_commits,  # всего закреплений
            "total_holds": self.total_holds,  # всего удержаний
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},  # эксперты по именам в порядке имён, каждый — словарём
            "growth_log": [vars(x) for x in self.growth_log],  # журнал роста — список словарей атрибутов событий
            "theorems": [vars(x) for x in self.theorems],  # теоремы — список словарей атрибутов
            "synthesis_stats": {  # статистика синтеза
                "candidates_generated": self.stats.candidates_generated,  # порождено кандидатов
                "evals": self.stats.evals,  # число вычислений
                "sieve_survivors": self.stats.sieve_survivors,  # прошедших сито
            },
        }
        target = Path(path)  # целевой путь
        target.parent.mkdir(parents=True, exist_ok=True)  # создать каталог, если его нет
        tmp = target.with_name(target.name + ".tmp")  # временный файл рядом с целевым: имя + «.tmp»
        with tmp.open("w", encoding="utf-8") as handle:  # открыть временный файл на запись
            json.dump(data, handle, indent=2, sort_keys=True)  # записать JSON с отступом 2 и сортировкой ключей
            handle.flush()  # сбросить буферы Python
            os.fsync(handle.fileno())  # сбросить на диск
        tmp.replace(target)  # атомарно заменить целевой файл временным

    @classmethod  # метод класса: конструктор из файла
    def load(cls, path: str | Path) -> "GrowingMathProcessorV7":  # загрузить состояние из JSON-файла path
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # прочитать файл и разобрать JSON
        if data.get("format") != "HAGI-GrowingMathProcessor":  # формат не тот
            raise ValueError("not a HAGI Growing Math Processor state file")  # ошибка: это не файл состояния
        model = cls(  # создать процессор с параметрами из файла
            int(data.get("lookup_limit", 16)),  # предел таблицы (по умолчанию 16)
            float(data.get("mdl_lambda", 0.001)),  # вес стоимости описания (по умолчанию 0.001)
            int(data.get("node_budget", DEFAULT_NODE_BUDGET)),  # предел узлов (по умолчанию DEFAULT_NODE_BUDGET)
            int(data.get("pop_budget", DEFAULT_POP_BUDGET)),  # предел популяции (по умолчанию DEFAULT_POP_BUDGET)
            str(data.get("device", "cpu")),  # устройство (по умолчанию cpu)
        )
        model.version = int(data.get("model_version", 0))  # версия модели (по умолчанию 0)
        model.total_commits = int(data.get("total_commits", 0))  # всего закреплений (по умолчанию 0)
        model.total_holds = int(data.get("total_holds", 0))  # всего удержаний (по умолчанию 0)
        model.experts = {  # эксперты из файла
            k: Expert.from_dict(v) for k, v in data.get("experts", {}).items()  # k — имя, v — словарь; словарь по умолчанию пуст
        }
        if data.get("version") != V7_VERSION:  # версия формата файла отличается от текущей
            for expert in model.experts.values():  # по всем экспертам
                expert.code_length = _program_description_bits(expert.program)  # пересчитать длину кода по текущему описанию
        model.growth_log = [GrowthEvent(**x) for x in data.get("growth_log", [])]  # журнал роста из списка словарей
        model.theorems = [Theorem(**x) for x in data.get("theorems", [])]  # теоремы из списка словарей
        stats = data.get("synthesis_stats", {})  # статистика синтеза (по умолчанию пустой словарь)
        model.stats.candidates_generated = int(stats.get("candidates_generated", 0))  # порождено кандидатов
        model.stats.evals = int(stats.get("evals", 0))  # число вычислений
        model.stats.sieve_survivors = int(stats.get("sieve_survivors", 0))  # прошедших сито
        return model  # восстановленный процессор

    def _fits(  # верно ли program на всех парах (xs, yy): число вычислений засчитывается в stats
        self, program: Program, xs_list: Sequence[Sequence[int]], ys: Sequence[int]  # program — программа, xs_list — входы, ys — ожидаемые выходы
    ) -> bool:  # вернуть True, если программа верна на всех парах
        self.stats.evals += len(xs_list)  # засчитать вычисления: evals += |xs_list|
        try:  # вычисление может упасть
            return all(self._same(program.eval(xs), yy) for xs, yy in zip(xs_list, ys))  # True, если program(xs) == yy на всех парах
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
            return False  # программа не годится


# ------------------- обучаемый защищённый оптимизатор -------------------------------------------


def _fiber_forward(params, x, hidden_scale=1.0):  # прямой проход волокна: y = tanh(x·w1 + b1)·w2 + b2
    w1, b1, w2, b2 = params  # распаковать параметры: w1 (d, h), b1 (1, h), w2 (h, 1), b2 (1, 1)
    return np.tanh(x @ w1 + b1) @ w2 + b2  # h = tanh(x·w1 + b1); выход y = h·w2 + b2


def fit_fiber_guarded(  # обучить волокно: спуск с приёмом только шагов, уменьшающих MSE
    inputs, targets, val_inputs, val_targets, seed, hidden: int = 24, epochs: int = 300  # входы и выходы, отложенные входы и выходы, зерно, ширина h = 24, число эпох = 300
) -> "NeuralFiber":  # вернуть обученное волокно
    """Подбор необязательного волокна tanh полным пакетом с защитой по MSE.

        Обновление-кандидат принимается, только когда измеренная обучающая MSE
        не возрастает. Это рабочая защита от регрессии на уровне реализации,
        а не доказательство шага, не гарантия на отложенных данных и не теорема
        сходимости. Параметры: inputs, targets — обучение, val_inputs, val_targets —
        отложенные данные, seed — зерно, hidden — ширина h, epochs — число эпох.
        Градиент MSE L = mean((f(x) − y)²), f = tanh(x·w1 + b1)·w2 + b2:
        dL/dw1 = xᵀ·d, dL/db1 = Σ d, d = (2/N·e·w2ᵀ) ⊙ (1 − h²),
        dL/dw2 = hᵀ·p, dL/db2 = Σ p, p = (2/N)·e. Возвращает NeuralFiber с
        удостоверением guard_certificate."""
    x = np.asarray(inputs, dtype=np.float64).reshape(-1, len(inputs[0]))  # X — матрица входов (N, d)
    y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)  # Y — столбец выходов (N, 1)
    xv = np.asarray(val_inputs, dtype=np.float64).reshape(-1, x.shape[1])  # X_val — отложенные входы (N_val, d)
    yv = np.asarray(val_targets, dtype=np.float64).reshape(-1, 1)  # Y_val — отложенные выходы (N_val, 1)
    sx = max(1.0, float(np.max(np.abs(x))))  # масштаб входов s_x = max(1, max|x|)
    sy = max(1.0, float(np.max(np.abs(y))))  # масштаб выходов s_y = max(1, max|y|)
    x, y, xv, yv = x / sx, y / sy, xv / sx, yv / sy  # нормировка: x' = x / s_x, y' = y / s_y (отложенные — теми же масштабами)
    rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
    params = [  # параметры [w1, b1, w2, b2]
        rng.normal(0.0, 0.32, (x.shape[1], hidden)),  # w1 ~ N(0, 0.32), форма (d, h)
        np.zeros((1, hidden)),  # b1 = 0, форма (1, h)
        rng.normal(0.0, 0.12, (hidden, 1)),  # w2 ~ N(0, 0.12), форма (h, 1)
        np.zeros((1, 1)),  # b2 = 0, форма (1, 1)
    ]

    def loss(p):  # среднеквадратичная ошибка MSE на обучении для параметров p
        err = _fiber_forward(p, x) - y  # остаток e = f_p(x) − y
        return float(np.mean(err * err))  # средний квадрат: MSE = mean(e²)

    cur = loss(params)  # текущие потери
    best = [q.copy() for q in params]  # лучшие параметры — копии текущих
    best_val = float("inf")  # лучшие отложенные потери +∞
    total_drop = 0.0  # суммарное сертифицированное убывание потерь
    accepted = 0  # число принятых шагов
    held = 0  # число удержанных шагов
    for _ in range(epochs):  # по эпохам
        pred = _fiber_forward(params, x)  # предсказание f(x) на обучении
        err = pred - y  # остаток e = f(x) − y
        # направление спуска D = −ḡ (аналог транспонированной средней невязки)
        h_act = np.tanh(x @ params[0] + params[1])  # скрытый слой h = tanh(x·w1 + b1)
        dh = (2.0 / len(x) * err) @ params[2].T  # градиент dL/dh = (2/N)·e·w2ᵀ
        dh = dh * (1.0 - h_act**2)  # умножить на производную тангенса: dL/dz = dL/dh ⊙ (1 − h²)
        dp = 2.0 / len(x) * err  # градиент dL/dp = (2/N)·e
        grads = [  # градиенты [w1, b1, w2, b2]
            x.T @ dh,  # градиент dL/dw1 = xᵀ·(dL/dz)
            np.sum(dh, axis=0, keepdims=True),  # dL/db1 = Σ_n dL/dz_n (строка 1×h)
            h_act.T @ dp,  # градиент dL/dw2 = hᵀ·(dL/dp)
            np.sum(dp, axis=0, keepdims=True),  # градиент dL/db2 = Σ_n dL/dp_n
        ]
        gnorm2 = sum(float(np.sum(g * g)) for g in grads)  # квадрат нормы градиента ‖g‖² = Σ по всем параметрам g²
        if gnorm2 <= 0:  # градиент нулевой
            held += 1  # шаг удержан
            continue  # перейти к следующей эпохе
        # кандидаты безопасного шага: сетка с возвратом по дроблению шага;
        # принять первый, чьи измеренные потери строго меньше текущих
        improved = False  # улучшение на этой эпохе пока не найдено
        for step in (1.0, 0.5, 0.25, 0.1, 0.03):  # шаги спуска от 1 до 0.03 (половинное дробление)
            cand = [p - step * g for p, g in zip(params, grads)]  # кандидат p' = p − step·g
            cand_loss = loss(cand)  # потери кандидата
            if cand_loss < cur:  # потери строго уменьшились
                drop = cur - cand_loss  # убывание Δ = L_old − L_new
                params = cand  # принять кандидата
                cur = cand_loss  # обновить текущие потери
                total_drop += drop  # суммарное убывание + Δ
                accepted += 1  # принятых шагов + 1
                improved = True  # улучшение достигнуто
                break  # остальные шаги не пробуются
        if not improved:  # ни один шаг не уменьшил потери
            held += 1  # шаг удержан
        # удержанный шаг: параметры не меняются, в удостоверении убывание ноль
        hv = _fiber_forward(params, xv)  # предсказание на отложенных
        vl = float(np.mean((hv - yv) ** 2))  # отложенные потери MSE_val = mean((f(x_val) − y_val)²)
        if np.isfinite(vl) and vl < best_val:  # потери конечны и лучше рекорда
            best_val = vl  # новый рекорд
            best = [q.copy() for q in params]  # запомнить копии параметров
    w1, b1, w2, b2 = best  # лучшие параметры
    h = np.tanh(x @ w1 + b1)  # h = tanh(x·w1 + b1) на обучении
    tr_loss = float(np.mean((h @ w2 + b2 - y) ** 2)) * sy * sy  # обучающие потери в исходном масштабе: MSE·s_y²
    hv = np.tanh(xv @ w1 + b1)  # h на отложенных
    predv = (hv @ w2 + b2).reshape(-1) * sy  # предсказание в исходном масштабе: ŷ = f(x_val)·s_y, вектор
    val_loss = float(np.mean((predv - np.asarray(val_targets, dtype=float)) ** 2))  # отложенные потери в исходном масштабе
    exact = float(  # точность округлённых предсказаний
        np.mean(  # доля совпадений
            np.rint(predv).astype(np.int64) == np.asarray(val_targets, dtype=np.int64)  # доля: round(ŷ_n) = y_n
        )
    )
    fiber = NeuralFiber(  # собрать волокно
        len(inputs[0]), sx, sy, w1, b1, w2, b2, tr_loss, val_loss, exact  # d, масштабы, параметры, потери, точность
    )
    fiber.guard_certificate = {  # удостоверение защитника
        "accepted_steps": accepted,  # принято шагов
        "hold_steps": held,  # удержано шагов
        "total_certified_drop": total_drop,  # суммарное убывание потерь
    }
    return fiber  # обученное волокно


# --------------- сертифицированное тернарное сжатие ---------------------------------------------


# ------------------- решатель энергии CRAM ------------------------------------------------------


def solve_coloring(  # раскраска графа спуском по числу конфликтов (энергия)
    edges,  # список рёбер (a, b)
    n_nodes: int,  # число вершин 0..n_nodes−1
    num_colors: int,  # число цветов 0..num_colors−1
    givens: Optional[dict[int, int]] = None,  # фиксированные цвета вершин (по умолчанию нет)
    max_sweeps: int = 10000,  # предел проходов по вершинам
) -> dict[str, Any]:  # вернуть словарь с раскраской и итогами
    """Спуск по энергии одной ячейки для раскраски графа: энергия равна числу
    конфликтных рёбер; ход — смена цвета одной вершины; принимаются только ходы
    со строгим убыванием энергии. На конечном множестве состояний строгий спуск
    конечен; результат проверяется точно: энергия 0 равносильна правильной
    раскраске. Параметры: edges — рёбра, n_nodes — число вершин, num_colors —
    число цветов, givens — фиксированные цвета, max_sweeps — предел проходов.
    Возвращает {'coloring', 'valid', 'energy', 'sweeps', 'verified',
    'givens_respected'}."""
    from collections import defaultdict  # словарь с заводом множества

    adj = defaultdict(set)  # списки смежности: adj[v] — соседи вершины v
    for a, b in edges:  # по рёбрам
        adj[a].add(b)  # b — сосед a
        adj[b].add(a)  # a — сосед b (граф неориентированный)
    rng = np.random.default_rng(42)  # генератор с фиксированным зерном 42
    givens = givens or {}  # пустой словарь вместо None
    y = {i: givens.get(i, int(rng.integers(0, num_colors))) for i in range(n_nodes)}  # начальная раскраска: цвет i равен заданному либо случайному из 0..num_colors−1

    def energy(state) -> int:  # энергия состояния: число рёбер с одинаковыми цветами концов
        return sum(1 for a, b in edges if state[a] == state[b])  # энергия E = Σ_{(a,b)} [y_a = y_b]

    cur = energy(y)  # текущая энергия
    sweeps = 0  # число сделанных проходов
    while sweeps < max_sweeps:  # пока не исчерпан предел проходов
        improved = False  # в этом проходе улучшений пока нет
        for i in range(n_nodes):  # по вершинам
            if i in givens:  # цвет вершины зафиксирован
                continue  # не менять
            # точное локальное поле: число конфликтов по цветам кандидата;
            # Δ = L_i(a) − L_i(y_i), слагаемых разбиения нет
            counts = {c: sum(1 for j in adj[i] if y[j] == c) for c in range(num_colors)}  # counts[c] = число соседей i цвета c = L_i(c)
            best_c, best_conf = y[i], counts[y[i]]  # лучший цвет — текущий, число его конфликтов L_i(y_i)
            for c in range(num_colors):  # по цветам
                if counts[c] < best_conf:  # у цвета c меньше конфликтов, чем у лучшего
                    best_c, best_conf = c, counts[c]  # он становится лучшим
            if best_conf < counts[y[i]]:  # строгое убывание: Δ = L_i(best_c) − L_i(y_i) < 0
                y[i] = best_c  # перекрасить вершину i
                improved = True  # проход с улучшением
        cur = energy(y)  # энергия после прохода
        sweeps += 1  # проходов + 1
        if cur == 0:  # конфликтов нет
            break  # стоп
        if not improved:  # локальный (не глобальный) минимум достигнут
            # достигнут координатный минимум (не глобальный): перезапуск с новым зерном
            y = {  # новая случайная раскраска
                i: givens.get(i, int(rng.integers(0, num_colors)))  # фиксированный цвет либо случайный из 0..num_colors−1
                for i in range(n_nodes)  # по всем вершинам i = 0..n_nodes−1
            }
    # нулевая энергия конфликтов равносильна правильной раскраске (точная сверка)
    valid = all(y[a] != y[b] for a, b in edges) and all(  # правильная раскраска: на каждом ребре цвета различны и
        0 <= y[i] < num_colors for i in range(n_nodes)  # каждый цвет в диапазоне 0..num_colors−1
    )
    return {  # словарь результата
        "coloring": y,  # раскраска: вершина → цвет
        "valid": bool(valid),  # правильность
        "energy": int(cur),  # конечная энергия
        "sweeps": int(sweeps),  # число проходов
        "verified": bool(valid) == (cur == 0),  # сверка: (раскраска правильна) ⇔ (энергия 0)
        "givens_respected": all(y[i] == g for i, g in givens.items()),  # фиксированные цвета сохранены
    }


# ------------------- точная индукция формул и поиск теорем ------------------------------------


def _exact_poly_fit(  # точная рациональная интерполяция многочлена p(x) = Σ c_i·x^i по точкам
    xs: Sequence[int], ys: Sequence[int], max_degree: int = 5  # абсциссы, ординаты, наибольшая степень 5
) -> Optional[tuple[int, ...]]:  # вернуть коэффициенты (c0..cd) либо None
    """Точная рациональная интерполяция многочлена (дроби, без плавающей точки).

        Возвращает кортеж целых коэффициентов (c0..cd) единственного интерполянта
        p(x) = Σ c_i·x^i степени ≤ max_degree через точки либо None, если система
        несовместна или не имеет единственного решения. Степень d подбирается от 1
        до max_degree; система d + 1 уравнений решается точным исключением Гаусса
        над Fraction. Вся арифметика точная; проверка на остальных точках —
        дело вызывающего."""
    pts = sorted({(int(x), int(y)) for x, y in zip(xs, ys)})  # различные точки (x, y) в порядке возрастания
    if len({x for x, _ in pts}) < 2:  # различных абсцисс меньше двух
        return None  # многочлен не определяется
    for degree in range(1, min(max_degree, len(pts) - 1) + 1):  # степени 1..min(max_degree, |pts| − 1)
        k = degree + 1  # число неизвестных коэффициентов k = d + 1
        sample = pts[:k]  # первые k точек
        # точное исключение Гаусса над рациональными числами для системы Вандермонда
        a = [[Fraction(x) ** i for i in range(k)] + [Fraction(y)] for x, y in sample]  # матрица Вандермонда: строка [1, x, .., x^d | y] из рациональных дробей
        n = k  # размер системы n = k
        for col in range(n):  # прямой ход исключения Гаусса по столбцам
            piv = next((r for r in range(col, n) if a[r][col] != 0), None)  # первая строка r ≥ col с ненулевым элементом в столбце col
            if piv is None:  # ведущего элемента нет — система вырождена
                break  # прервать (for … else не сработает)
            a[col], a[piv] = a[piv], a[col]  # поменять строки местами
            pv = a[col][col]  # ведущий элемент pv
            a[col] = [v / pv for v in a[col]]  # нормировать строку: a[col] /= pv
            for r in range(n):  # по всем строкам
                if r != col and a[r][col] != 0:  # другая строка с ненулевым элементом в столбце col
                    f = a[r][col]  # множитель f
                    a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]  # новая строка a[r] = a[r] − f·a[col]
        else:  # иначе — степень от 2 до min(max_degree, |pts| − 1)
            coeffs = [row[n] for row in a]  # коэффициенты c_0..c_d — последний столбец приведённой матрицы
            if any(c.denominator != 1 for c in coeffs):  # есть нецелый коэффициент (знаменатель ≠ 1)
                continue  # нецелые коэффициенты — не компактная целочисленная формула
            icoeffs = tuple(int(c) for c in coeffs)  # целый кортеж коэффициентов c_i
            # точная проверка на ВСЕХ точках, а не только на использованных для интерполяции
            if all(sum(ci * x**i for i, ci in enumerate(icoeffs)) == y for x, y in pts):  # p(x) = Σ_i c_i·x^i = y на ВСЕХ точках
                return icoeffs  # вернуть коэффициенты
    return None  # ни одна степень не подошла


def _monomial_exponents(arity: int, max_degree: int) -> tuple[tuple[int, ...], ...]:  # все векторы показателей (m_1..m_n) с суммой ≤ max_degree, сначала младшие
    """Все векторы показателей (m_1..m_arity) с полной степенью Σ m_i ≤ max_degree,
        первыми идут младшие (сортировка по сумме, затем по самому вектору)."""
    if arity < 1 or max_degree < 0:  # арность < 1 или отрицательная степень
        raise ValueError("arity must be positive and max_degree nonnegative")  # ошибка аргументов
    return tuple(  # вернуть кортеж
        sorted(  # отсортированный
            (  # генератор
                powers  # вектор показателей
                for powers in itertools.product(range(max_degree + 1), repeat=arity)  # по всем m_i ∈ 0..max_degree
                if sum(powers) <= max_degree  # с суммой Σ m_i ≤ max_degree
            ),
            key=lambda powers: (sum(powers), powers),  # ключ: (полная степень, лексикографически сам вектор)
        )
    )


def _exact_multivariate_poly_fit(  # точный целочисленный многочлен p(x) = Σ c_m·Π x_i^{m_i} минимальной степени по всем строкам
    inputs: Sequence[Sequence[int]],  # строки входов
    targets: Sequence[int],  # ожидаемые выходы
    arity: int,  # число переменных n
    max_degree: int = 4,  # наибольшая полная степень 4
) -> Optional[tuple[tuple[int, tuple[int, ...]], ...]]:  # вернуть кортеж слагаемых ((c, m), ...) либо None
    """Найти целочисленный многочлен минимальной полной степени, проходящий через все строки.

            Решается переопределённая система A·c = y точным методом Гаусса — Жордана
            над дробями Fraction: строка A — значения мономов Π x_i^{m_i} входа.
            Степень d перебирается от 0 вверх; результат возвращается, только если
            матрица имеет полный ранг по столбцам (решение единственно), система
            совместна и все коэффициенты целые. Плавающие допуски и регуляризация
            в поиск формулы не входят. Проверку на отложенных данных и на всей
            конечной области делает вызывающий."""

    if len(inputs) != len(targets) or not inputs or arity < 1:  # длины не совпадают, нет данных или арность < 1
        return None  # многочлен не ищется
    rows = [tuple(int(v) for v in row) for row in inputs]  # строки входов — целые кортежи
    if any(len(row) != arity for row in rows):  # есть строка длины ≠ arity
        return None  # отказ
    ys = [int(v) for v in targets]  # выходы — целые
    for degree in range(max_degree + 1):  # по полным степеням 0..max_degree (первая подходящая — минимальная)
        monomials = _monomial_exponents(arity, degree)  # векторы показателей степени ≤ degree
        ncols = len(monomials)  # число неизвестных = число мономов
        if len(rows) < ncols:  # уравнений меньше, чем неизвестных
            continue  # система недоопределена — следующая степень
        matrix = []  # расширенная матрица системы
        for row, y in zip(rows, ys):  # по строкам данных
            features = [  # значения мономов строки
                Fraction(math.prod(x**power for x, power in zip(row, powers)))  # m-й моном = Π_i x_i^{m_i}
                for powers in monomials  # по векторам показателей
            ]
            matrix.append(features + [Fraction(y)])  # строка [значения мономов | y]

        pivot_row = 0  # номер текущей ведущей строки
        pivot_rows: list[int] = []  # номера строк, выбранных ведущими по столбцам
        full_rank = True  # предположение: столбцы линейно независимы
        for col in range(ncols):  # исключение Гаусса — Жордана по столбцам
            pivot = next(  # ведущая строка
                (r for r in range(pivot_row, len(matrix)) if matrix[r][col] != 0), None  # первая строка r ≥ pivot_row с ненулевым элементом; иначе None
            )
            if pivot is None:  # в столбце нет ненулевого элемента
                full_rank = False  # столбцы зависимы
                break  # прекратить исключение
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]  # поднять ведущую строку на место pivot_row
            scale = matrix[pivot_row][col]  # ведущий элемент
            matrix[pivot_row] = [v / scale for v in matrix[pivot_row]]  # нормировать ведущую строку: ведущий элемент станет 1
            for r in range(len(matrix)):  # по всем строкам
                if r == pivot_row:  # сама ведущая строка
                    continue  # не менять
                factor = matrix[r][col]  # множитель a[r][col]
                if factor:  # он ненулевой
                    matrix[r] = [  # вычесть приведённую ведущую строку
                        a - factor * b for a, b in zip(matrix[r], matrix[pivot_row])  # новая строка: a[r] = a[r] − factor·a[pivot_row]
                    ]
            pivot_rows.append(pivot_row)  # запомнить ведущую строку столбца col
            pivot_row += 1  # перейти к следующей строке
        if not full_rank:  # решение не единственно
            continue  # следующая степень
        if any(  # есть противоречивая строка
            all(row[c] == 0 for c in range(ncols)) and row[-1] != 0 for row in matrix  # все коэффициенты 0, а свободный член ≠ 0
        ):
            continue  # система несовместна
        coefficients = [matrix[pivot_rows[col]][-1] for col in range(ncols)]  # решение: свободный член ведущей строки каждого столбца
        if any(value.denominator != 1 for value in coefficients):  # есть дробный коэффициент
            continue  # не целочисленный многочлен
        terms = tuple(  # кортеж (коэффициент, вектор показателей) ненулевых мономов
            (int(coefficient), powers)  # пара (c, m)
            for coefficient, powers in zip(coefficients, monomials)  # по найденным коэффициентам и мономам
            if coefficient  # только ненулевые
        )

        # перепроверка каждой поданной строки независимо от пути исключения
        def evaluate(row: Sequence[int]) -> int:  # значение многочлена p(row) = Σ c·Π x_i^{m_i}
            return sum(  # сумма по мономам
                c * math.prod(int(x) ** p for x, p in zip(row, powers))  # слагаемое c·Π_i x_i^{m_i}
                for c, powers in terms  # по слагаемым
            )

        if all(evaluate(row) == y for row, y in zip(rows, ys)):  # перепроверка: p(row) = y на каждой строке
            return terms  # вернуть слагаемые
    return None  # ни одна степень не дала целочисленного решения


def _multivariate_poly_expr(  # выражение «poly_nd» из слагаемых многочлена
    terms: tuple[tuple[int, tuple[int, ...]], ...],  # слагаемые (c, m)
) -> Expr:  # вернуть выражение
    return Expr(  # узел выражения
        "poly_nd",  # имя операции
        (),  # без дочерних выражений
        tuple((int(c), tuple(int(p) for p in powers)) for c, powers in terms),  # payload: ((c, (m_1..m_n)), ...) целыми
    )


def _poly_expr(coeffs: tuple[int, ...]) -> Expr:  # выражение «poly» от переменной 0 с коэффициентами coeffs
    return Expr("poly", (Expr.var(0),), coeffs)  # узел poly(Var0; c_0..c_d)


def _eval_poly(coeffs: tuple[int, ...], x: int) -> int:  # значение p(x) по схеме Горнера
    acc = 0  # аккумулятор 0
    for ci in reversed(coeffs):  # от старшего коэффициента к младшему
        acc = acc * x + ci  # Горнер: acc = acc·x + c_i
    return acc  # значение p(x)


@dataclass  # класс данных
class Theorem:  # найденное конечное тождество
    name: str  # имя
    statement: str  # текст утверждения
    domain_points: int  # число проверенных точек
    verified: bool = True  # проверено (по умолчанию да)


class TheoremMiner:  # поиск конечных тождеств среди экспертов
    """Поиск тождеств на конечной области — только там, где поддержано каждое вычисление.

            Результат — утверждения о конечной области, а не безграничные алгебраические
            теоремы. Для унарных тождеств итерации e(e(x)) дополнительно требуется,
            объявленная область была замкнута относительно выходов эксперта."""

    def mine(  # найти тождества
        self, model: "GrowingMathProcessorV7", max_experts: int = 24  # модель и предел числа экспертов каждой арности
    ) -> list[Theorem]:  # вернуть список найденных теорем
        found: list[Theorem] = []  # найденные в этом запуске теоремы
        known = {t.name for t in model.theorems}  # имена уже известных теорем
        bin_e = [e for e in model.experts.values() if e.program.arity == 2][  # бинарные эксперты, срез
            :max_experts  # первые max_experts штук
        ]
        un_e = [e for e in model.experts.values() if e.program.arity == 1][:max_experts]  # унарные эксперты, не более max_experts

        for e in bin_e:  # по бинарным экспертам
            dx, dy = e.domains[0], e.domains[1]  # объявленные области первого и второго аргументов
            declared_pairs = [(x, y) for x in dx for y in dy if e.supports((x, y))]  # все поддержанные пары декартова произведения dx × dy
            comm_pairs = [(x, y) for x, y in declared_pairs if e.supports((y, x))]  # пары, у которых поддержана и переставленная (y, x)
            # Небольшое симметричное пересечение нельзя называть глобальной коммутативностью:
            # у каждой поддержанной объявленной пары должна быть поддержана и переставленная.
            if comm_pairs and len(comm_pairs) == len(declared_pairs):  # коммутативные пары есть и совпадают со всеми объявленными
                name = f"commutativity({e.name})"  # имя теоремы
                if name not in known:  # ещё не доказана
                    try:  # вычисление может упасть
                        ok = all(  # верно на всех парах
                            e.program.eval((int(x), int(y)))  # значение e(x, y)
                            == e.program.eval((int(y), int(x)))  # равно e(y, x)
                            for x, y in comm_pairs  # по всем парам
                        )
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                        ok = False  # теорема не подтверждена
                    if ok:  # тождество выполнено
                        th = Theorem(  # новая теорема
                            name,  # имя
                            f"for all declared pairs with both orders supported: {e.name}(x,y) = {e.name}(y,x)",  # текст утверждения
                            len(comm_pairs),  # число проверенных пар
                        )  # конец аргументов
                        model.theorems.append(th)  # добавить в модель
                        found.append(th)  # и в найденные
                        known.add(name)  # запомнить имя

            self_pairs = [x for x in dx if x in dy and e.supports((x, x))]  # x из dx, лежащие в dy, с поддержанной парой (x, x)
            name = f"self-absorption({e.name})"  # имя теоремы
            if self_pairs and len(self_pairs) == len(dx) and name not in known:  # диагональных пар не меньше, чем элементов dx, и теорема новая
                try:  # вычисление может упасть
                    ok = all(  # верно на всех диагональных парах
                        e.program.eval((int(x), int(x))) == int(x) for x in self_pairs  # верно e(x, x) = x на всех таких x
                    )
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                    ok = False  # теорема не подтверждена
                if ok:  # тождество выполнено
                    th = Theorem(  # новая теорема
                        name,  # имя
                        f"for all x in the declared overlap with (x,x) supported: {e.name}(x,x) = x",  # текст утверждения
                        len(self_pairs),  # число проверенных точек
                    )  # конец аргументов
                    model.theorems.append(th)  # добавить в модель
                    found.append(th)  # и в найденные
                    known.add(name)  # запомнить имя

            zero_pairs = [x for x in dx if 0 in dy and e.supports((x, 0))]  # x из dx с поддержанной парой (x, 0) (0 должно быть в dy)
            name = f"annihilator-zero({e.name})"  # имя теоремы
            if zero_pairs and len(zero_pairs) == len(dx) and name not in known:  # таких x не меньше, чем в dx, и теорема новая
                try:  # вычисление может упасть
                    ok = all(e.program.eval((int(x), 0)) == int(x) for x in zero_pairs)  # e(x, 0) = x на всех таких x
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                    ok = False  # теорема не подтверждена
                if ok:  # тождество выполнено
                    th = Theorem(  # новая теорема
                        name,  # имя
                        f"for all declared x with (x,0) supported: {e.name}(x,0) = x",  # текст утверждения
                        len(zero_pairs),  # число проверенных точек
                    )  # конец аргументов
                    model.theorems.append(th)  # добавить в модель
                    found.append(th)  # и в найденные
                    known.add(name)  # запомнить имя

        for e in un_e:  # по унарным экспертам
            dom = tuple(int(x) for x in e.domains[0])  # объявленная область — целые
            image: dict[int, int] = {}  # образ: x → e(x)
            closed = True  # предположение: область замкнута относительно e
            try:  # вычисление может упасть
                for x in dom:  # по всем x области
                    if not e.supports((x,)):  # вход не поддержан
                        closed = False  # замыкания нет
                        break  # стоп
                    y = int(e.program.eval((x,)))  # образ y = e(x)
                    if not e.supports((y,)):  # выход не входит в поддержанную область
                        closed = False  # замыкания нет
                        break  # стоп
                    image[x] = y  # запомнить образ
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                closed = False  # замыкания нет

            if closed:  # область замкнута: e(e(x)) определено
                name = f"involutivity({e.name})"  # имя теоремы
                if name not in known:  # новая
                    try:  # вычисление может упасть
                        ok = all(int(e.program.eval((image[x],))) == x for x in dom)  # e(e(x)) = x на всей области
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                        ok = False  # не подтверждено
                    if ok:  # тождество выполнено
                        th = Theorem(  # новая теорема
                            name,  # имя
                            f"for all x in the declared domain: {e.name}({e.name}(x)) = x",  # текст утверждения
                            len(dom),  # число проверенных точек
                        )  # конец аргументов
                        model.theorems.append(th)  # добавить в модель
                        found.append(th)  # и в найденные
                        known.add(name)  # запомнить имя

                name = f"idempotence({e.name})"  # имя теоремы
                if name not in known:  # новая
                    try:  # вычисление может упасть
                        ok = all(  # верно на всей области
                            int(e.program.eval((image[x],))) == image[x] for x in dom  # верно e(e(x)) = e(x) на всей области
                        )
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                        ok = False  # не подтверждено
                    if ok:  # тождество выполнено
                        th = Theorem(  # новая теорема
                            name,  # имя
                            f"for all x in the declared domain: {e.name}({e.name}(x)) = {e.name}(x)",  # текст утверждения
                            len(dom),  # число проверенных точек
                        )  # конец аргументов
                        model.theorems.append(th)  # добавить в модель
                        found.append(th)  # и в найденные
                        known.add(name)  # запомнить имя

            if 0 in dom:  # ноль входит в область
                name = f"zero-fixpoint({e.name})"  # имя теоремы
                if name not in known:  # новая
                    try:  # вычисление может упасть
                        ok = int(e.program.eval((0,))) == 0  # верно e(0) = 0
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                        ok = False  # не подтверждено
                    if ok:  # тождество выполнено
                        th = Theorem(  # новая теорема
                            name, f"at the supported point x=0: {e.name}(0) = 0", 1  # имя, текст утверждения, одна точка
                        )
                        model.theorems.append(th)  # добавить в модель
                        found.append(th)  # и в найденные
                        known.add(name)  # запомнить имя
        return found  # найденные теоремы


# ------------------------- регулятор ресурсов (неограниченный цикл) --------------------------


def _ram_available_bytes() -> int:  # доступная оперативная память в байтах (0 — неизвестно)
    try:  # psutil может отсутствовать
        import psutil  # библиотека сведений о системе

        return int(psutil.virtual_memory().available)  # доступные байты
    except ImportError:  # psutil нет
        if sys.platform == "win32":  # ОС Windows
            import ctypes  # вызовы системных библиотек

            class MEMORYSTATUSEX(ctypes.Structure):  # структура сведений о памяти Windows
                _fields_ = [  # поля структуры
                    ("dwLength", ctypes.c_ulong),  # размер структуры
                    ("dwMemoryLoad", ctypes.c_ulong),  # загрузка памяти, %
                    ("ullTotalPhys", ctypes.c_ulonglong),  # вся физическая память
                    ("ullAvailPhys", ctypes.c_ulonglong),  # доступная физическая память
                    ("ullTotalPageFile", ctypes.c_ulonglong),  # предел файла подкачки
                    ("ullAvailPageFile", ctypes.c_ulonglong),  # доступный файл подкачки
                    ("ullTotalVirtual", ctypes.c_ulonglong),  # предел виртуальной памяти процесса
                    ("ullAvailVirtual", ctypes.c_ulonglong),  # доступная виртуальная память процесса
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),  # доступная расширенная виртуальная память
                ]

            st = MEMORYSTATUSEX()  # структура-приёмник
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)  # поле dwLength = размер структуры (требование системы)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # заполнить st системным вызовом
            return int(st.ullAvailPhys)  # доступная физическая память
        return 0  # другая платформа без psutil: неизвестно


def _vram_free_bytes(device: str) -> int:  # свободная видеопамять устройства в байтах (0 — неизвестно)
    if not (device.startswith("cuda") and _ensure_torch()):  # не cuda или torch недоступен
        return 0  # неизвестно
    try:  # вызов может упасть
        free, _total = _torch.cuda.mem_get_info(_torch.device(device))  # свободная и вся память устройства
        return int(free)  # свободные байты
    except Exception:  # любая ошибка
        return 0  # неизвестно


class ResourceGovernor:  # регулятор ресурсов: переводит свободную оперативную и видеопамять в ограниченные бюджеты поиска
    """Переводит свободную оперативную и видеопамять в ограниченные бюджеты поиска
            на задачу: node_budget, pop_budget, chunk.

            Внешний цикл роста ограничен отдельно монотонным сроком и пределом раундов.
            При нехватке памяти бюджеты плавно уменьшаются, а не позволяют поиск
            занять всю память."""

    RAM_FLOOR_BYTES = 300 * 1024 * 1024  # нижний порог оперативной памяти 300 МиБ: ниже — пропуск задачи
    VRAM_FLOOR_BYTES = 300 * 1024 * 1024  # нижний порог видеопамяти 300 МиБ

    def __init__(self, device: str = "cpu"):  # создать регулятор для устройства device
        self.device = device  # устройство
        self.node_budget = DEFAULT_NODE_BUDGET  # бюджет узлов по умолчанию
        self.pop_budget = DEFAULT_POP_BUDGET  # бюджет популяции по умолчанию
        self.chunk = VALIDATION_CHUNK  # размер пачки по умолчанию
        self.total_skips = 0  # всего пропущенных задач

    def _scale(self) -> float:  # масштаб s ∈ [0, 1] по свободной памяти
        ram = _ram_available_bytes()  # доступная оперативная память
        if ram <= 0:  # неизвестна
            return 1.0  # без ограничения
        # полная скорость выше 6 ГиБ, линейное убывание до нуля на нижнем пороге
        hi, lo, floor = 6 * 2**30, 1 * 2**30, self.RAM_FLOOR_BYTES  # hi = 6 ГиБ, lo = 1 ГиБ, floor = нижний порог
        s = min(1.0, max(0.0, (ram - floor) / max(1, hi - floor)))  # масштаб s = clip((ram − floor)/(hi − floor), 0, 1)
        if ram < lo:  # памяти меньше 1 ГиБ
            s = min(s, max(0.05, (ram - floor) / max(1, lo - floor)))  # жёстче: s = min(s, max(0.05, (ram − floor)/(lo − floor)))
        if self.device.startswith("cuda") and _ensure_torch():  # cuda и torch доступен
            vram = _vram_free_bytes(self.device)  # свободная видеопамять
            if vram < self.VRAM_FLOOR_BYTES:  # ниже порога
                s = 0.0  # полный стоп
        return s  # масштаб

    def before_task(self) -> bool:  # пересчитать бюджеты перед задачей; False — ресурсы исчерпаны, задачу пропустить
        """Пересчитать бюджеты перед следующей задачей. False — ресурсы исчерпаны, задачу пропустить."""
        s = self._scale()  # масштаб
        if s <= 0.0:  # ресурсов нет
            self.total_skips += 1  # пропусков + 1
            return False  # пропустить
        self.node_budget = max(1_000, int(DEFAULT_NODE_BUDGET * s))  # бюджет узлов = max(1000, ⌊DEFAULT·s⌋)
        self.pop_budget = max(2_000, int(DEFAULT_POP_BUDGET * s))  # бюджет популяции = max(2000, ⌊DEFAULT·s⌋)
        self.chunk = max(8, int(VALIDATION_CHUNK * min(1.0, s * 2)))  # пачка = max(8, ⌊VALIDATION_CHUNK·min(1, 2s)⌋)
        return True  # работать

    def recover(self, exc: BaseException) -> None:  # после ошибки памяти: резко урезать бюджеты, сбросить кэш, продолжать
        """После ошибки памяти: резко урезать бюджеты, сбросить кэш видеопамяти, продолжать работу."""
        DEFAULT_NODE_BUDGET_S = 10_000  # аварийный предел узлов
        self.node_budget = min(self.node_budget, DEFAULT_NODE_BUDGET_S)  # узлов не более 10 000
        self.pop_budget = min(self.pop_budget, 20_000)  # популяция не более 20 000
        self.chunk = max(4, self.chunk // 4)  # пачка вчетверо меньше, но не менее 4
        if self.device.startswith("cuda") and _ensure_torch():  # cuda и torch доступен
            try:  # вызов может упасть
                _torch.cuda.empty_cache()  # освободить кэш видеопамяти
            except Exception:  # любая ошибка
                pass  # игнорировать


# ---------------------- неограниченный цикл саморазвития -------------------------------------


def _propose_task_from_experts(  # предложить задачу-композицию на общей заверенной области родителей
    experts: dict[str, Any],  # эксперты по именям
    rng: np.random.Generator,  # генератор случайных чисел
    round_index: int,  # номер раунда
    depth: int = 1,  # глубина композиции 1
    domain_radius: int = 6,  # радиус области 6
) -> Optional[TaskSpec]:  # вернуть задачу TaskSpec либо None
    """Предложить композицию только на совместно заверенной области родителей.

            Ни одна родительская программа не вычисляется вне области Expert.supports().
            Порождённая задача наследует соответствующее правило допустимости входа,
            и вся её декартова область проверяется заранее, прежде чем вернуть
            замыкание учителя. Так растёт конечная композиционная программа задач,
            а не самостоятельная способность. Параметры: experts — эксперты по именам,
            rng — генератор случайных чисел, round_index — номер раунда, depth — глубина
            композиции, domain_radius — радиус области. Возвращает TaskSpec или None."""
    arity2 = [  # эксперты с двумя входами не из семейства BoolVenn
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"  # условие отбора
    ]
    arity1 = [  # эксперты с одним входом не из BoolVenn
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"  # условие отбора
    ]
    bool2 = [  # логические эксперты с двумя входами
        e for e in experts.values() if e.program.arity == 2 and e.family == "BoolVenn"  # условие отбора
    ]

    # логическая композиция: каждый промежуточный и внешний вызов проверяется на поддержку
    if bool2 and len(bool2) >= 2 and rng.random() < 0.4:  # логических не меньше двух и случайное число < 0.4
        for _ in range(8):  # до 8 попыток
            outer = bool2[int(rng.choice(len(bool2)))]  # внешняя функция — случайная
            inner = bool2[int(rng.choice(len(bool2)))]  # внутренняя функция — случайная
            if inner is outer and len(bool2) > 1:  # совпали, а выбор есть
                continue  # другая попытка
            rows = [  # допустимые пары входов
                (x, y)  # пара (x, y)
                for x in (0, 1)  # по x из множества {0, 1}
                for y in (0, 1)  # по y из множества {0, 1}
                if inner.supports((x, y))  # внутренняя функция поддерживает (x, y)
                and outer.supports((int(inner.program.eval((x, y))), y))  # внешняя поддерживает (inner(x, y), y)
            ]
            if len(rows) != 4:  # допустимых пар не все четыре
                continue  # другая попытка
            teacher = lambda x, y, o=outer, i=inner: int(  # учитель: outer(inner(x, y), y)
                o.program.eval((int(i.program.eval((int(x), int(y)))), int(y)))  # результат o(i(x, y), y)
            )
            return TaskSpec(  # задача композиции
                f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",  # имя задачи: номер раунда, внутренний и внешний эксперты
                2,  # два входа
                ((0, 1), (0, 1)),  # области обоих входов {0, 1}
                teacher,  # замыкание учителя
                family="auto-composed",  # семейство — автоматическая композиция
            )

    if not arity2:  # бинарных экспертов нет
        return None  # предложить нечего
    inner_pool = arity2 + arity1  # внутренняя функция может быть любой: бинарной или унарной

    # Для предложения нужно достаточно строк для разбиения на обучение и отложенную проверку.
    min_rows = 4  # минимум строк данных для разбиения на обучение и проверку
    max_radius = max(1, min(int(domain_radius), 12))  # максимальный радиус области: не больше 12 и не меньше 1
    for _ in range(12):  # до 12 попыток
        outer = arity2[int(rng.choice(len(arity2)))]  # внешняя функция — случайный бинарный эксперт
        inner = inner_pool[int(rng.choice(len(inner_pool)))]  # внутренняя — случайной арности

        if inner.program.arity == 1:  # внутренняя функция унарная: outer(inner(x), y)
            # outer(inner(x), y): x лежит в области унарного эксперта, а
            # каждое inner(x) — в области первого аргумента внешнего эксперта.
            xs0 = tuple(  # допустимые x
                x  # значение x
                for x in inner.domains[0]  # из области внутреннего эксперта
                if abs(int(x)) <= max_radius  # модуль |x| ≤ max_radius
                and inner.supports((int(x),))  # вход поддержан внутренним экспертом
                and int(inner.program.eval((int(x),))) in outer.domains[0]  # inner(x) лежит в области первого аргумента внешнего
            )
            ys0 = tuple(  # допустимые y
                y  # значение y
                for y in outer.domains[1]  # из области второго аргумента внешнего эксперта
                if abs(int(y)) <= max_radius  # модуль |y| ≤ max_radius
                and (outer.valid_rule != "nonzero_y" or int(y) != 0)  # если второе слагаемое требует y ≠ 0, то y ≠ 0
            )
            if not xs0 or not ys0:  # пусто
                continue  # другая попытка
            for radius in range(max_radius, 0, -1):  # радиус от max_radius вниз до 1
                xs = tuple(x for x in xs0 if abs(int(x)) <= radius)  # срез xs = {x из xs0 : |x| ≤ radius}
                ys = tuple(y for y in ys0 if abs(int(y)) <= radius)  # срез ys = {y из ys0 : |y| ≤ radius}
                ys = tuple(  # оставить только безопасные y
                    y  # значение y
                    for y in ys  # по ys
                    if all(  # для всех x из xs
                        outer.supports((int(inner.program.eval((int(x),))), int(y)))  # внешний поддерживает (inner(x), y)
                        for x in xs  # по x
                    )
                )
                if len(xs) * len(ys) < min_rows:  # строк |xs|·|ys| меньше четырёх
                    continue  # меньший радиус
                if not all(inner.supports((int(x),)) for x in xs):  # не каждый x поддержан внутренним экспертом
                    continue  # меньший радиус
                teacher = lambda x, y, o=outer, i=inner: int(  # учитель: outer(inner(x), y)
                    o.program.eval((int(i.program.eval((int(x),))), int(y)))  # результат o(i(x), y)
                )
                return TaskSpec(  # задача композиции
                    f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",  # имя задачи: номер раунда, внутренний и внешний эксперты
                    2,  # два входа
                    (xs, ys),  # области: xs по первому входу, ys по второму
                    teacher,  # замыкание учителя
                    valid_rule=outer.valid_rule,  # правило допустимости — как у внешнего эксперта
                    family="auto-composed",  # семейство — автоматическая композиция
                )

        elif inner.program.arity == 2:  # внутренняя функция бинарная: outer(x, inner(x, y))
            # outer(x, inner(x,y)): прямоугольник входов намеренно сужается,
            # пока каждый вызов обоих родителей остаётся заверенным.
            xs0 = tuple(  # допустимые x
                x  # значение x
                for x in inner.domains[0]  # из области первого аргумента внутреннего эксперта
                if x in outer.domains[0] and abs(int(x)) <= max_radius  # x в области первого аргумента внешнего и |x| ≤ max_radius
            )
            ys0 = tuple(y for y in inner.domains[1] if abs(int(y)) <= max_radius)  # кандидаты ys0 = {y из области второго аргумента внутреннего : |y| ≤ max_radius}
            if not xs0 or not ys0:  # пусто
                continue  # другая попытка
            for radius in range(max_radius, 0, -1):  # радиус от max_radius вниз до 1
                xs = tuple(x for x in xs0 if abs(int(x)) <= radius)  # срез xs = {x из xs0 : |x| ≤ radius}
                ys = tuple(y for y in ys0 if abs(int(y)) <= radius)  # срез ys = {y из ys0 : |y| ≤ radius}
                if not xs or not ys:  # пусто
                    continue  # меньший радиус
                # Убрать столбцы y, где не для каждого выбранного x поддержаны вычисления
                # внутреннего и внешнего экспертов. Получается прямоугольное безопасное подмножество.
                safe_ys = []  # безопасные значения y
                for y in ys:  # по y
                    good = True  # пока столбец y безопасен
                    for x in xs:  # по всем x
                        if not inner.supports((int(x), int(y))):  # внутренний не поддерживает (x, y)
                            good = False  # столбец небезопасен
                            break  # стоп
                        inner_value = int(inner.program.eval((int(x), int(y))))  # промежуточное значение inner_value = inner(x, y)
                        if not outer.supports((int(x), inner_value)):  # внешний не поддерживает (x, inner(x, y))
                            good = False  # столбец небезопасен
                            break  # стоп
                    if good:  # все проверки пройдены
                        safe_ys.append(y)  # оставить y
                ys = tuple(safe_ys)  # ys — только безопасные столбцы
                if len(xs) * len(ys) < min_rows:  # строк |xs|·|ys| меньше четырёх
                    continue  # меньший радиус
                teacher = lambda x, y, o=outer, i=inner: int(  # учитель: outer(x, inner(x, y))
                    o.program.eval((int(x), int(i.program.eval((int(x), int(y))))))  # результат o(x, i(x, y))
                )
                return TaskSpec(  # задача композиции
                    f"AUTO_R{round_index}_{inner.name}_IN_{outer.name}",  # имя задачи: номер раунда, внутренний и внешний эксперты
                    2,  # два входа
                    (xs, ys),  # области: xs по первому входу, ys по второму
                    teacher,  # замыкание учителя
                    valid_rule=inner.valid_rule,  # правило допустимости — как у внутреннего эксперта
                    family="auto-composed",  # семейство — автоматическая композиция
                )

    return None  # композицию предложить не удалось


def verify_growth_lineage(model: "GrowingMathProcessorV7") -> dict[str, Any]:  # сверить каждого эксперта-композицию с парой его родителей
    """Перепроверить каждый сохранённый эксперт-композицию по паре его родителей.

            Это исполняемая сверка на всей объявленной конечной области каждого потомка:
            child(xs) должен равняться outer от входа, собранного из inner и xs. Вне этой
            области это не доказательство. Возвращает {'lineage_checked', 'lineage_errors',
            'lineage_pass'}."""
    checked = 0  # число проверенных экспертов-композиций
    errors: list[dict[str, Any]] = []  # список ошибок
    for child in model.experts.values():  # по всем экспертам
        if not child.parents:  # нет родителей
            continue  # не композиция
        if len(child.parents) != 2:  # родителей не два
            errors.append({"expert": child.name, "reason": "invalid_parent_arity"})  # ошибка: неверное число родителей
            continue  # взять другую пару
        outer = model.experts.get(child.parents[0])  # внешний родитель
        inner = model.experts.get(child.parents[1])  # внутренний родитель
        if outer is None or inner is None:  # родитель отсутствует
            errors.append({"expert": child.name, "reason": "missing_parent"})  # ошибка: родитель потерян
            continue  # взять другую пару
        ok = True  # предположение: композиция верна
        checked += 1  # проверено ещё один
        for xs in itertools.product(*child.domains):  # по всем входам декартова произведения областей
            xs = tuple(int(x) for x in xs)  # вход — целыми
            if not child.supports(xs):  # вход не поддержан
                continue  # взять другую пару
            try:  # вычисление может упасть
                if (  # оба родителя — логические функции с маской Венна
                    outer.program.venn_mask is not None  # у внешнего есть маска Венна
                    and inner.program.venn_mask is not None  # и у внутреннего есть маска Венна
                ):
                    if not inner.supports(xs):  # внутренний не поддерживает вход
                        ok = False  # несовпадение
                        break  # стоп
                    middle = int(inner.program.eval(xs))  # промежуточное значение middle = inner(x, y)
                    parent_input = (middle, xs[1])  # вход внешнего: (middle, y)
                elif inner.program.arity == 1:  # внутренний унарный
                    if not inner.supports((xs[0],)):  # не поддерживает первый аргумент
                        ok = False  # несовпадение
                        break  # стоп
                    middle = int(inner.program.eval((xs[0],)))  # промежуточное значение middle = inner(x)
                    parent_input = (middle, xs[1])  # вход внешнего: (inner(x), y)
                elif inner.program.arity == 2:  # внутренний бинарный
                    if not inner.supports(xs):  # не поддерживает вход
                        ok = False  # несовпадение
                        break  # стоп
                    middle = int(inner.program.eval(xs))  # промежуточное значение middle = inner(x, y)
                    parent_input = (xs[0], middle)  # вход внешнего: (x, inner(x, y))
                else:  # прочая арность
                    ok = False  # неподдерживаемая арность
                    break  # стоп
                if not outer.supports(parent_input):  # внешний не поддерживает свой вход
                    ok = False  # несовпадение
                    break  # стоп
                expected = int(outer.program.eval(parent_input))  # ожидаемый результат outer(вход внешнего)
                if expected != int(child.program.eval(xs)):  # не совпадает с результатом потомка
                    ok = False  # несовпадение
                    break  # стоп
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                ok = False  # ошибка вычисления
                break  # стоп
        if not ok:  # нашли расхождение
            errors.append(  # записать ошибку
                {"expert": child.name, "reason": "parent_composition_mismatch"}  # ошибка: композиция родителей не совпадает
            )
    return {  # итог
        "lineage_checked": checked,  # сколько проверено
        "lineage_errors": errors,  # список ошибок
        "lineage_pass": not errors,  # успешно, если ошибок нет
    }


def grow_forever(  # цикл композиционного роста: обучение, композиции, снимки состояния
    model: "GrowingMathProcessorV7",  # растущая модель
    suite: Sequence[TaskSpec],  # базовый набор задач
    device: str = "cpu",  # устройство
    max_rounds: Optional[int] = None,  # предел раундов (None — без предела)
    snapshot_every: int = 10,  # снимок состояния после каждых 10 закреплений
    out_dir: str = ".omc/hagi_grow",  # каталог вывода
    log_every: int = 1,  # печатать строку журнала каждый раунд (1) или только при закреплениях (0)
    budget_cap: Optional[int] = None,  # потолок бюджета узлов
    deadline: Optional[float] = None,  # монотонный срок окончания
) -> dict[str, Any]:  # вернуть словарь итогов
    """Ограниченный цикл композиционного роста с явными снимками состояния.

            Каждый раунд: учит базовую программу задач, предлагает конечные композиции
            только внутри областей поддержки родителей и снимает состояние. Вызывающий
            может задать монотонный срок и предел раундов; регулятор ресурсов ограничивает
            каждый поиск. Это рекомбинация программы задач, а не доказательство новой
            общей способности. Параметры: model — модель, suite — базовый набор задач,
            device — устройство, max_rounds — предел раундов, snapshot_every — снимок
            после стольких закреплений, out_dir — каталог вывода, log_every — печатать
            журнал каждый раунд, budget_cap — потолок бюджета узлов, deadline — срок.
            Возвращает словарь с записями раундов и причиной остановки."""
    out = Path(out_dir)  # каталог вывода
    out.mkdir(parents=True, exist_ok=True)  # создать его при отсутствии
    gov = ResourceGovernor(device)  # регулятор ресурсов
    log_path = out / "growth_log.jsonl"  # путь журнала раундов
    start = time.time()  # время начала
    rnd = np.random.default_rng(20260101)  # генератор случайных чисел с фиксированным зерном
    report: dict[str, Any] = {"device": device, "rounds": [], "stopped": None}  # итоговый словарь: устройство, записи раундов, причина остановки
    round_index = 0  # номер раунда
    commits_since_snapshot = 0  # закреплений с последнего снимка
    idle_rounds = 0  # раундов подряд без закреплений
    depth = 1  # глубина композиции
    domain_radius = 6  # радиус области
    try:  # тело цикла охраняется от прерывания
        while max_rounds is None or round_index < max_rounds:  # пока не достигнут предел раундов
            if deadline is not None and time.monotonic() >= deadline:  # срок настал
                report["stopped"] = "TIME_LIMIT"  # причина остановки — лимит времени
                break  # стоп: срок настал
            t0 = time.time()  # время начала раунда
            committed_this_round = 0  # закреплений в этом раунде
            # (1) базовый набор задач (повторно: уже закреплённые задачи используются заново)
            for spec in suite:  # по задачам базового набора
                if deadline is not None and time.monotonic() >= deadline:  # срок настал
                    report["stopped"] = "TIME_LIMIT"  # причина остановки — лимит времени
                    break  # стоп: ресурсы исчерпаны
                if not gov.before_task():  # ресурсы исчерпаны
                    break  # стоп: срок настал
                model.node_budget = (  # бюджет узлов модели
                    min(gov.node_budget, budget_cap) if budget_cap else gov.node_budget  # min(бюджет регулятора, потолок) либо бюджет регулятора
                )
                model.pop_budget = (  # бюджет популяции модели
                    min(gov.pop_budget, budget_cap * 2)  # min(бюджет регулятора, 2·потолок)
                    if budget_cap  # если задан потолок бюджета
                    else gov.pop_budget  # иначе — бюджет регулятора
                )
                try:  # обучение может упасть
                    ev = model.learn(  # обучиться задаче
                        spec,  # задача
                        seed=10_000 + round_index * 7919 + len(spec.name),  # зерно = 10000 + раунд·7919 + длина имени
                        deadline=deadline,  # срок
                        train_neural=os.environ.get("HAGI_NO_NEURAL") != "1",  # обучать нейроволокно, если HAGI_NO_NEURAL ≠ 1
                    )
                    if ev.action == "COMMIT":  # задача закреплена
                        committed_this_round += 1  # закреплений в раунде + 1
                        commits_since_snapshot += 1  # с последнего снимка + 1
                except (MemoryError, RuntimeError, ValueError) as exc:  # ошибка памяти или выполнения
                    gov.recover(exc)  # урезать бюджеты и продолжать
            # (2) самостоятельно предложенные задачи: композиции закреплённых экспертов
            proposed = 0  # число предложенных задач
            attempts = min(12, max(2, len(model.experts) // 3))  # попыток = min(12, max(2, ⌊число экспертов / 3⌋))
            for k in range(attempts):  # по попыткам
                if deadline is not None and time.monotonic() >= deadline:  # срок настал
                    report["stopped"] = "TIME_LIMIT"  # причина остановки — лимит времени
                    break  # стоп: срок настал
                if not gov.before_task():  # ресурсы исчерпаны
                    break  # стоп: ресурсы исчерпаны
                spec = _propose_task_from_experts(  # предложить композицию
                    model.experts,  # эксперты
                    rnd,  # генератор
                    round_index,  # номер раунда
                    depth=depth,  # глубина
                    domain_radius=domain_radius,  # радиус области
                )
                if spec is None or spec.name in model.experts:  # ничего не предложено либо такая задача уже есть
                    continue  # пропустить: ничего не предложено или задача уже есть
                model.node_budget = (  # бюджет узлов модели
                    min(gov.node_budget, budget_cap) if budget_cap else gov.node_budget  # min(бюджет регулятора, потолок) либо бюджет регулятора
                )
                model.pop_budget = (  # бюджет популяции модели
                    min(gov.pop_budget, budget_cap * 2)  # min(бюджет регулятора, 2·потолок)
                    if budget_cap  # если задан потолок бюджета
                    else gov.pop_budget  # иначе — бюджет регулятора
                )
                try:  # обучение может упасть
                    hint = (  # подсказка композиции
                        _composition_hint(spec)  # выражение по задаче
                        if spec.family == "auto-composed"  # только для автоматических композиций
                        else None  # иначе подсказки нет
                    )
                    if spec.family == "auto-composed" and hint is None:  # композиция не выражается в безопасной грамматике подсказок
                    # Предложение не выражается в текущей безопасной грамматике
                    # подсказок. Слепой синтез вместо него не запускается.
                        continue  # не подбирать вслепую
                    ev = model.learn(  # обучиться задаче
                        spec,  # задача
                        seed=500_000 + round_index * 104_729 + k,  # зерно = 500000 + раунд·104729 + k
                        deadline=deadline,  # срок
                        candidate_hint=hint,  # подсказка
                        train_neural=os.environ.get("HAGI_NO_NEURAL") != "1",  # обучать нейроволокно, если HAGI_NO_NEURAL ≠ 1
                    )
                    proposed += 1  # предложенных + 1
                    if ev.action == "COMMIT":  # задача закреплена
                        committed_this_round += 1  # закреплений в раунде + 1
                        commits_since_snapshot += 1  # с последнего снимка + 1
                except (  # ловим ошибки памяти и выполнения
                    MemoryError,  # ошибка памяти
                    RuntimeError,  # ошибка выполнения
                    ValueError,  # ошибка значения
                    ZeroDivisionError,  # деление на ноль
                    OverflowError,  # переполнение
                ) as exc:  # исключение exc
                    gov.recover(exc)  # урезать бюджеты и продолжать
            # (3) самостоятельный поиск теорем по заверенной базе экспертов
            try:  # поиск теорем может упасть
                new_theorems = TheoremMiner().mine(model, max_experts=16)  # найти новые тождества (не более 16 экспертов каждой арности)
            except (  # ловим ошибки памяти и выполнения
                MemoryError,  # ошибка памяти
                RuntimeError,  # ошибка выполнения
                ValueError,  # ошибка значения
                ZeroDivisionError,  # деление на ноль
                OverflowError,  # переполнение
            ):
                new_theorems = []  # новых теорем нет
            lineage = verify_growth_lineage(model)  # сверить родословную композиций
            round_rec = {  # запись о раунде
                "round": round_index,  # номер
                **lineage,  # поля сверки родословной
                "experts": len(model.experts),  # число экспертов
                "unique_programs": len({e.program for e in model.experts.values()}),  # число различных программ
                "unique_certified_scopes": len(  # число различных заверенных областей
                    {  # множество ключей
                        (e.program, e.domains, e.valid_rule)  # ключ: (программа, области, правило допустимости)
                        for e in model.experts.values()  # по всем экспертам
                    }
                ),
                "committed_this_round": committed_this_round,  # закреплений в раунде
                "proposed_tasks": proposed,  # предложенных задач
                "theorems_total": len(model.theorems),  # всего теорем
                "theorems_new": len(new_theorems),  # новых теорем
                "total_code_length": sum(e.code_length for e in model.experts.values()),  # суммарная длина кода экспертов, бит
                "node_budget": gov.node_budget,  # бюджет узлов
                "pop_budget": gov.pop_budget,  # бюджет популяции
                "ram_avail_gb": round(_ram_available_bytes() / 2**30, 2),  # доступная оперативная память, ГиБ (2 знака)
                "vram_free_gb": round(_vram_free_bytes(device) / 2**30, 2),  # свободная видеопамять, ГиБ (2 знака)
                "elapsed": round(time.time() - t0, 2),  # длительность раунда, с (2 знака)
            }
            report["rounds"].append(round_rec)  # добавить запись
            with open(log_path, "a", encoding="utf-8") as f:  # открыть журнал на дозапись
                f.write(json.dumps(round_rec) + "\n")  # записать строку JSON с переводом строки
            if log_every or committed_this_round:  # печатать каждый раунд либо были закрепления
                print(  # напечатать строку журнала
                    f"[grow r{round_index}] experts={round_rec['experts']} "  # раунд и число экспертов
                    f"commits={committed_this_round} code={round_rec['total_code_length']}bits "  # закреплений и длина кода в битах
                    f"ram={round_rec['ram_avail_gb']}GB "  # доступная память, ГиБ
                    f"vram={round_rec['vram_free_gb']}GB",  # свободная видеопамять, ГиБ
                    flush=True,  # сбросить вывод сразу
                )
            if commits_since_snapshot >= snapshot_every:  # накоплено достаточно закреплений для снимка
                model.save(out / "hagi_grow_model.json")  # сохранить снимок модели
                commits_since_snapshot = 0  # сбросить счётчик
            if committed_this_round > 0:  # были закрепления
                idle_rounds = 0  # простой прерван
            else:  # закреплений не было
                idle_rounds += 1  # простой + 1
                if idle_rounds >= 3:  # три раунда подряд пусто
                    # насыщение: углубить алгебру композиций и расширить
                    # заверенные области — неограниченный рост программы задач
                    depth += 1  # углубить композицию
                    domain_radius = min(12, domain_radius + 2)  # расширить область на 2, не более 12
                    idle_rounds = 0  # сбросить счётчик простоя
            round_index += 1  # следующий раунд
            if deadline is not None and time.monotonic() >= deadline:  # срок настал
                report["stopped"] = "TIME_LIMIT"  # причина остановки — лимит времени
                break  # стоп: срок настал
            if (  # память
                _ram_available_bytes()  # доступная оперативная память известна (≠ 0)
                and _ram_available_bytes() < ResourceGovernor.RAM_FLOOR_BYTES  # и меньше нижнего порога
            ):
                report["stopped"] = "RAM_FLOOR"  # причина остановки — нижний порог памяти
                break  # стоп: памяти ниже порога
    except KeyboardInterrupt:  # прервано с клавиатуры
        report["stopped"] = "INTERRUPTED"  # причина остановки — прерывание
    finally:  # выполняется всегда
        model.save(out / "hagi_grow_model.json")  # сохранить модель
        (out / "theorems.json").write_text(  # записать теоремы
            json.dumps(  # сериализовать
                {  # словарь
                    "count": len(model.theorems),  # число теорем
                    "theorems": [vars(t) for t in model.theorems],  # список словарей атрибутов
                },  # конец словаря
                indent=2,  # с отступом 2
            ),  # конец аргументов dumps
            encoding="utf-8",  # в кодировке utf-8
        )  # конец вызова write_text
        report["total_experts"] = len(model.experts)  # всего экспертов
        report.update(verify_growth_lineage(model))  # добавить итоги сверки родословной
        report["total_elapsed"] = round(time.time() - start, 1)  # общее время, с (1 знак)
        (out / "growth_report.json").write_text(  # записать итоговый словарь
            json.dumps(report, indent=2), encoding="utf-8"  # JSON с отступом 2 в кодировке utf-8
        )  # конец вызова write_text
    return report  # вернуть итоги


# ---------------------------- эксперимент и замеры -------------------------------------------


def _run_seed_worker(args: tuple[int, str, str]) -> dict[str, Any]:  # один процесс — одно зерно: обучение модели и её снимок
    """Один процесс — одно зерно: зерна независимы, поэтому работа распределяется
            по ядрам процессора почти линейно. Регулятор ресурсов процесса ограничивает
            его собственную память. Снимок модели зерна сохраняется в out_dir.
            Возвращает словарь итогов."""
    seed, device, out_dir = args  # распаковать аргументы: зерно, устройство, каталог вывода
    train_neural = os.environ.get("HAGI_NO_NEURAL") != "1"  # обучать нейроволокно, если HAGI_NO_NEURAL ≠ 1
    out = Path(out_dir)  # каталог вывода
    suite = make_task_suite()  # базовый набор задач
    model = GrowingMathProcessorV7(device=device)  # свежая модель
    t0 = time.time()  # время начала
    stats0 = SynthesisStats()  # начальная статистика
    stats0.candidates_generated = model.stats.candidates_generated  # базовые показания счётчика порождённых кандидатов
    stats0.evals = model.stats.evals  # базовые показания счётчика вычислений
    stage_order = ["BoolVenn", "unary", "integer-binary", "composed-ternary"]  # порядок этапов по семействам
    for stage in stage_order:  # по этапам
        stage_tasks = [s for s in suite if s.family == stage]  # задачи этапа
        rng = np.random.default_rng(seed + len(stage) * 101)  # генератор с зерном seed + длина_имени_этапа·101
        if len(stage_tasks) > 1:  # задач больше одной
            order = rng.permutation(len(stage_tasks))  # случайный порядок номеров
            stage_tasks = [stage_tasks[int(i)] for i in order]  # задачи в этом порядке
        for j, spec in enumerate(stage_tasks):  # по задачам этапа с номером j
            model.learn(  # обучиться задаче
                spec,  # задача
                seed=seed * 1009 + j * 17 + len(spec.name),  # зерно = seed·1009 + j·17 + длина имени
                train_neural=train_neural,  # обучать ли нейроволокно
            )
    verification = model.verify_all(suite)  # проверить все задачи набора
    ood_checks = ood_abstains = 0  # счётчики проверок вне области и отказов
    for spec in suite:  # по задачам набора
        if spec.name not in model.experts:  # задача не закреплена
            continue  # пропустить: задача не закреплена
        outside = [d[-1] + 1 for d in spec.domains]  # вход вне области: каждый аргумент на 1 больше максимума своей области
        ood_checks += 1  # проверок вне области + 1
        ood_abstains += int(model.predict(spec.name, *outside)["status"] == "ABSTAIN")  # +1, если модель отказалась отвечать
    poison_model = GrowingMathProcessorV7()  # модель для опыта с испорченной меткой
    poison_event = poison_model.learn(  # обучить задаче ADD с одной неверной меткой
        next(s for s in suite if s.name == "ADD"),  # задача ADD из набора
        seed=seed + 44001,  # зерно
        poison_one_label=True,  # испортить одну метку
    )
    prime_spec = TaskSpec(  # незнакомая задача: простота числа
        "IS_PRIME",  # имя задачи
        1,  # один вход
        (tuple(range(-8, 9)),),  # область x ∈ −8..8
        lambda x: int(x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))),  # учитель: x ≥ 2 и x не делится ни на одно d из 2..⌊√x⌋
        family="unseen",  # семейство «незнакомая»
    )
    unknown_event = GrowingMathProcessorV7().learn(prime_spec, seed=seed + 99001)  # попытка выучить незнакомую задачу
    reuse_count = sum(1 for e in model.growth_log if e.reason == "expert-reuse")  # число закреплений повторным использованием
    gen = model.stats.candidates_generated - stats0.candidates_generated  # порождено кандидатов за опыт
    evs = model.stats.evals - stats0.evals  # вычислений за опыт
    model.save(out / f"hagi_v7_model_seed{seed}.json")  # сохранить модель зерна
    return {  # словарь итогов зерна
        "seed": int(seed),  # зерно
        "committed": len(model.experts),  # число закреплённых экспертов
        "expected": len(suite),  # ожидаемое число задач
        "commit_rate": len(model.experts) / len(suite),  # доля закреплений = закреплено / ожидается
        "symbolic_exact_accuracy": verification["accuracy"],  # точность на наборе
        "total_code_length": sum(e.code_length for e in model.experts.values()),  # суммарная длина кода, бит
        "candidates_generated": gen,  # порождено кандидатов
        "program_evals": evs,  # вычислений программ
        "sieve_survivors": model.stats.sieve_survivors,  # прошедших сито
        "expert_reuse_commits": reuse_count,  # закреплений повторным использованием
        "ood_abstains": ood_abstains,  # отказов вне области
        "ood_checks": ood_checks,  # проверок вне области
        "poison_action": poison_event.action,  # действие при испорченной метке
        "unknown_is_prime_action": unknown_event.action,  # действие для незнакомой задачи
        "elapsed_seconds": time.time() - t0,  # длительность, с
    }


def run_v7_experiment(  # прогнать опыт по нескольким зернам и усреднить
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),  # зерна 0..7
    out_dir: str = ".omc/hagi_v7_results",  # каталог вывода
    workers: int = 0,  # число процессов (0 — автоматически)
    device: str = "cpu",  # устройство cpu
) -> dict[str, Any]:  # вернуть словарь с результатами по зёрнам и агрегатами
    out = Path(out_dir)  # каталог вывода
    out.mkdir(parents=True, exist_ok=True)  # создать при отсутствии
    suite = make_task_suite()  # базовый набор задач
    start = time.time()  # время начала
    args = [(int(s), device, str(out)) for s in seeds]  # аргументы процессов: (зерно, устройство, каталог)
    if workers <= 0:  # автоматический выбор
        workers = min(len(args), max(1, (os.cpu_count() or 2) - 1))  # min(число зёрен, max(1, число ядер − 1))
    workers = min(workers, len(args))  # процессов не больше, чем зёрен
    if workers > 1:  # параллельно
        with ProcessPoolExecutor(max_workers=workers) as ex:  # пул процессов
            per_seed = list(ex.map(_run_seed_worker, args))  # результаты по зёрнам
    else:  # последовательно
        per_seed = [_run_seed_worker(a) for a in args]  # по одному зерну за раз
    per_seed.sort(key=lambda r: r["seed"])  # отсортировать по зерну
    main_model_path = out / f"hagi_v7_model_seed{int(seeds[0])}.json"  # путь модели первого зерна
    if main_model_path.exists():  # файл есть
        GrowingMathProcessorV7.load(main_model_path)  # убедиться, что он загружается
        main_model_path.replace(out / "hagi_v7_model.json")  # переименовать в основную модель

    aggregate: dict[str, Any] = {}  # агрегаты по зёрнам
    for key in (  # по усредняемым показателям
        "committed",  # число закреплённых
        "symbolic_exact_accuracy",  # точность
        "total_code_length",  # длина кода
        "candidates_generated",  # порождено кандидатов
        "program_evals",  # вычисления программ
        "ood_abstains",  # отказы вне области
        "elapsed_seconds",  # длительность
    ):
        vals = [float(r[key]) for r in per_seed]  # значения показателя по зёрнам
        aggregate[key + "_mean"] = float(np.mean(vals))  # среднее = Σ vals / n
        aggregate[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0  # несмещённое отклонение sqrt(Σ(v − mean)² / (n − 1)); 0 при одном значении
    aggregate.update(  # добавить прочие агрегаты
        {  # открытие словаря аргументов
            "seeds": list(map(int, seeds)),  # список зёрен
            "expert_reuse_commits_total": int(  # всего повторных закреплений
                sum(r["expert_reuse_commits"] for r in per_seed)  # сумма по зёрнам
            ),
            "poison_all_hold": all(r["poison_action"] == "HOLD" for r in per_seed),  # испорченная метка везде даёт удержание
            "unknown_all_hold": all(  # незнакомая задача везде даёт удержание
                r["unknown_is_prime_action"] == "HOLD" for r in per_seed  # по зёрнам
            ),
            "reload_roundtrip_pass": GrowingMathProcessorV7.load(  # сохранение-загрузка: ADD(7, −3) = 4
                out / "hagi_v7_model.json"  # из основной модели
            )
            .predict("ADD", 7, -3)  # предсказать ADD(7, −3)
            .get("value")  # взять значение
            == 4,  # равно 4
            "total_runtime_seconds": time.time() - start,  # общее время, с
        }
    )
    result = {"version": V7_VERSION, "per_seed": per_seed, "aggregate": aggregate}  # итоговый словарь: версия, результаты по зёрнам, агрегаты
    (out / "hagi_v7_experiment.json").write_text(  # записать итоги
        json.dumps(result, indent=2), encoding="utf-8"  # JSON с отступом 2
    )
    return result  # вернуть итоги


def self_test_v7() -> None:  # самопроверка: обучение набора, отказы вне области, точные многочлены, теоремы
    suite = make_task_suite()  # базовый набор
    model = GrowingMathProcessorV7()  # модель
    for i, spec in enumerate(suite):  # по задачам
        ev = model.learn(spec, seed=10_000 + i, train_neural=False)  # обучить без нейроволокна
        assert ev.action == "COMMIT", (spec.name, ev)  # каждая задача обязана закрепиться
    check = model.verify_all(suite)  # проверить набор
    assert check["committed"] == len(suite) and check["accuracy"] == 1.0  # закреплены все и точность 1
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"  # вне области — отказ
    # точный шаблон медианы трёх проверяется на всей объявленной области
    med = model.experts["MEDIAN3"].program  # программа медианы трёх
    assert med.eval((1, 5, 3)) == 3 and med.eval((9, 2, 7)) == 7  # медиана (1,5,3) = 3; (9,2,7) = 7
    poisoned = GrowingMathProcessorV7().learn(  # опыт с испорченной меткой
        next(s for s in suite if s.name == "ADD"),  # задача ADD из набора
        seed=77,  # зерно 77
        poison_one_label=True,  # испортить одну метку
        train_neural=False,  # без нейроволокна
    )
    assert poisoned.action == "HOLD"  # ожидается удержание
    # точная индукция формулы: учитель 2x³ − x² + 7x − 4 должен быть выучен
    # как замкнутый многочлен с точной арифметикой
    poly_spec = TaskSpec(  # задача — кубический многочлен
        "POLY_CUBIC",  # имя
        1,  # один вход
        (tuple(range(-12, 13)),),  # область x ∈ −12..12
        lambda x: 2 * x**3 - x**2 + 7 * x - 4,  # учитель p(x) = 2x³ − x² + 7x − 4
        family="poly-unseen",  # семейство
    )
    pev = GrowingMathProcessorV7().learn(poly_spec, seed=31337, train_neural=False)  # выучить многочлен
    assert pev.action == "COMMIT" and "poly(" in pev.program, (pev.action, pev.program)  # закреплён как многочлен

    # точное открытие многочлена двух переменных; область проверяется полным перебором
    poly2 = TaskSpec(  # задача — многочлен двух переменных
        "POLY2_QUADRATIC",  # имя
        2,  # два входа
        (tuple(range(-4, 5)), tuple(range(-3, 4))),  # области x ∈ −4..4, y ∈ −3..3
        lambda x, y: 3 * x * x - 2 * x * y + y * y + 4 * x - 5 * y + 7,  # учитель 3x² − 2xy + y² + 4x − 5y + 7
        family="poly2-unseen",  # семейство
    )
    p2ev = GrowingMathProcessorV7().learn(poly2, seed=31338, train_neural=False)  # выучить
    assert p2ev.action == "COMMIT" and "poly_nd(" in p2ev.program, (  # закреплён как многомерный многочлен
        p2ev.action,  # действие
        p2ev.program,  # текст программы
    )
    p2model = GrowingMathProcessorV7()  # отдельная модель
    p2model.learn(poly2, seed=31338, train_neural=False)  # выучить
    assert p2model.verify_all([poly2])["accuracy"] == 1.0  # точность 1 на всей области

    # точное открытие многочлена трёх переменных с кубическим смешанным членом
    poly3 = TaskSpec(  # задача — многочлен трёх переменных
        "POLY3_CUBIC",  # имя
        3,  # три входа
        (tuple(range(-3, 4)), tuple(range(-3, 4)), tuple(range(-3, 4))),  # каждая область −3..3
        lambda x, y, z: 2 * x**3 - 3 * x * y * z + 4 * y * y - 5 * z + 11,  # учитель 2x³ − 3xyz + 4y² − 5z + 11
        family="poly3-unseen",  # семейство
    )
    p3model = GrowingMathProcessorV7()  # отдельная модель
    p3ev = p3model.learn(poly3, seed=31339, train_neural=False)  # выучить
    assert p3ev.action == "COMMIT" and "poly_nd(" in p3ev.program, (  # закреплён как многомерный многочлен
        p3ev.action,  # действие
        p3ev.program,  # текст программы
    )
    assert p3model.verify_all([poly3])["accuracy"] == 1.0  # точность 1 на всей области

    assert model.predict("ADD", 7, -3)["value"] == 4  # проверка: предсказание ADD(7, −3) равно 4
    assert model.experts["ADD"].code_length == _program_description_bits(  # длина кода ADD равна числу бит описания
        model.experts["ADD"].program  # программа эксперта ADD
    )
    # поиск теорем: только исчерпывающая проверка, непроверенных утверждений нет
    ths = TheoremMiner().mine(model)  # найти теоремы
    assert ths, "theorem miner found nothing"  # что-то найдено
    names = [t.name for t in model.theorems]  # имена теорем
    assert all(t.verified for t in model.theorems)  # все проверены
    assert any("commutativity(MIN)" in n for n in names), names  # есть коммутативность MIN
    assert any("idempotence(ABS)" in n for n in names), names  # есть идемпотентность ABS
    print(  # напечатать итог
        f"V7.4 SELF_TEST PASS: {len(suite)} experts, accuracy 1.0, OOD abstain, poison hold, "  # первая часть строки итога
        f"exact poly commit '{pev.program}', {len(model.theorems)} theorems"  # вторая часть строки итога
    )


def main() -> None:  # разбор аргументов командной строки и запуск выбранного режима
    p = argparse.ArgumentParser(  # разборщик аргументов
        description="HAGI v7.4 exact multivariate finite-domain research prototype"  # описание программы
    )
    p.add_argument("--self-test", action="store_true")  # флаг самопроверки
    p.add_argument("--experiment", action="store_true")  # флаг опыта по зернам
    p.add_argument(  # флаг цикла роста
        "--grow",  # имя
        action="store_true",  # без значения
        help="bounded compositional growth loop (10-minute default cap)",  # справка
    )  # конец аргумента
    p.add_argument(  # аргумент устройства
        "--device", default="cpu", help="cpu or cuda:0 (GPU batch prefilter)"  # по умолчанию cpu
    )  # конец аргумента
    p.add_argument(  # аргумент лимита времени
        "--max-minutes",  # имя
        type=float,  # вещественное
        default=10.0,  # по умолчанию 10 минут
        help="wall-clock cap for --grow (default: 10 minutes)",  # справка
    )  # конец аргумента
    p.add_argument("--max-rounds", type=int, default=None)  # предел раундов
    p.add_argument("--snapshot-every", type=int, default=10)  # снимок после каждых N закреплений
    p.add_argument(  # флаг отключения нейроволокна
        "--no-neural",  # имя
        action="store_true",  # без значения
        help="disable neural fallback training (fastest pure-symbolic)",  # справка
    )  # конец аргумента
    p.add_argument(  # аргумент бюджета поиска
        "--search-budget",  # имя
        type=int,  # целое
        default=2_500,  # по умолчанию 2500
        help="per-task candidate budget during --grow; pop budget is twice this",  # справка
    )  # конец аргумента
    p.add_argument(  # аргумент числа процессов
        "--workers", type=int, default=0, help="parallel seed processes (0=auto)"  # по умолчанию автоматически
    )  # конец аргумента
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])  # список зёрен
    p.add_argument("--out-dir", default=".omc/hagi_v7_results")  # каталог вывода
    args = p.parse_args()  # разобрать аргументы
    if args.self_test:  # самопроверка
        self_test_v7()  # выполнить
    elif args.grow:  # цикл роста
        suite = make_task_suite()  # базовый набор
        model = GrowingMathProcessorV7(device=args.device)  # модель на устройстве
        deadline = (  # срок
            time.monotonic() + max(0.0, args.max_minutes * 60)  # сейчас + max(0, минуты·60)
            if args.max_minutes is not None  # если лимит задан
            else None  # иначе без срока
        )
        if args.no_neural:  # нейроволокно отключено
            os.environ["HAGI_NO_NEURAL"] = "1"  # переменная окружения
        report = grow_forever(  # запустить цикл роста
            model,  # модель
            suite,  # набор задач
            device=args.device,  # устройство
            max_rounds=args.max_rounds,  # предел раундов
            snapshot_every=args.snapshot_every,  # период снимков
            out_dir=args.out_dir,  # каталог вывода
            budget_cap=max(1, args.search_budget),  # потолок бюджета не меньше 1
            deadline=deadline,  # срок
        )
        print(  # напечатать итог
            json.dumps(  # сериализовать
                {  # открытие словаря
                    k: report[k]  # ключ k
                    for k in ("device", "stopped", "total_experts", "total_elapsed")  # по четырём ключам
                },
                indent=2,  # с отступом 2
            )
        )
    elif args.experiment:  # опыт по зернам
        result = run_v7_experiment(  # запустить
            args.seeds, args.out_dir, workers=args.workers, device=args.device  # зёрна, каталог, процессы, устройство
        )
        print(json.dumps(result["aggregate"], indent=2))  # напечатать агрегаты
    else:  # режим не задан
        p.print_help()  # напечатать справку


# ==============================================================================
# ЧАСТЬ III — единая система v8.0
# ==============================================================================

"""HAGI v8.0 — единая саморазвивающаяся нейросимвольная система с энергетическим
        подходом: оптимальная архитектура, минимальный объём описания, минимальный
        объём вычислений."""


import sys as _sys  # модуль sys под именем _sys

h5 = _sys.modules[__name__]  # слой V5 — тот же модуль (пространство имён общее)
hagi_v7 = _sys.modules[__name__]  # слой V7 — тот же модуль

V8_VERSION = "8.4.0"  # версия формата V8

# ==============================================================================
# 1. точные биты префиксного кода, обобщённый граф вызовов, АНФ и кусочные выражения
# ==============================================================================

TOKEN_BITS = 6  # бит на токен префиксного кода
VAR_NAMES = ("x", "y", "z", "u", "v", "w")  # имена переменных
DYADIC_LEVELS = np.array([(c + 1) / 128.0 for c in range(255)], dtype=np.float64)  # двоичные уровни (c + 1)/128, c = 0..254
MAX_RANGE_EVAL = 100_000  # предел суммарного числа итераций свёрток диапазона
_RANGE_FOLD_OPS = {"all_range", "any_range", "sum_range", "prod_range"}  # операции свёртки диапазона


def _expr_contains_range(expr: "ExprV8") -> bool:  # содержит ли выражение свёртку диапазона
    return expr.op in _RANGE_FOLD_OPS or any(_expr_contains_range(a) for a in expr.args)  # сам узел — свёртка или есть свёртка в каком-либо аргументе


def gamma_bits(n: int) -> int:  # длина гамма-кода Элиаса числа n
    n = int(n)  # целое
    if n < 1:  # не положительное
        raise ValueError("gamma_bits expects positive integer")  # ошибка
    return 2 * (n.bit_length() - 1) + 1  # длина кода γ(n) = 2·⌊log2 n⌋ + 1


def zigzag(n: int) -> int:  # зигзаг-кодирование целого: 0, −1, 1, −2, 2 … → 0, 1, 2, 3, 4 …
    n = int(n)  # целое
    return 2 * n if n >= 0 else -2 * n - 1  # если n ≥ 0: код 2n; иначе код −2n − 1


@dataclass(frozen=True)  # неизменяемый класс данных
class ExprV8:  # расширенное неизменяемое синтаксическое дерево выражения
    """Расширенное неизменяемое синтаксическое дерево выражения: любая арность,
            вызовы экспертов (граф вызовов), алгебраическая нормальная форма (АНФ) и
            кусочные выражения. Поля: op — операция, args — кортеж аргументов,
            value — номер переменной, константа, имя вызываемого эксперта или параметры."""

    op: str  # имя операции
    args: tuple["ExprV8", ...] = ()  # аргументы-подвыражения
    value: Optional[Any] = None  # значение (номер переменной, константа, имя вызываемого эксперта, параметры)

    @staticmethod  # статический метод
    def var(i: int) -> "ExprV8":  # узел переменной № i
        return ExprV8("var", (), int(i))  # узел переменной var(i)

    @staticmethod  # статический метод
    def const(v: int) -> "ExprV8":  # узел константы
        return ExprV8("const", (), int(v))  # узел константы const(v)

    @staticmethod  # статический метод
    def from_h5(e: h5.Expr) -> "ExprV8":  # преобразовать выражение V5 в ExprV8
        return ExprV8(e.op, tuple(ExprV8.from_h5(a) for a in e.args), e.value)  # та же операция, рекурсивно преобразованные аргументы, то же значение

    def eval(  # вычислить выражение на входах xs
        self,  # само выражение
        xs: Sequence[int],  # значения входных переменных
        env: Optional[dict[str, "ExpertV8"]] = None,  # среда вызовов: имя эксперта → эксперт
        _budget: Optional[list[int]] = None,  # общий счётчик итераций (список из одного элемента)
        _fold_depth: int = 0,  # глубина вложенности свёрток
    ) -> int:  # вернуть целое значение
        # Изменяемый счётчик общий для всех символьных вызовов одного вычисления:
        # вызов эксперта не может незаметно умножить предел диапазона.
        if _budget is None:  # счётчик не передан
            _budget = [0]  # новый счётчик 0
        if self.op == "var":  # переменная
            return int(xs[int(self.value)])  # значение входа № value
        if self.op == "const":  # константа
            return int(self.value)  # сама константа
        if self.op == "call":  # вызов эксперта
            if env is None or self.value not in env:  # среды нет или эксперта в ней нет
                raise ValueError(f"missing called expert: {self.value}")  # ошибка
            vals = tuple(a.eval(xs, env, _budget, _fold_depth) for a in self.args)  # значения аргументов вызова
            callee = env[self.value]  # вызываемый эксперт
            if not callee.supports(vals):  # аргументы вне заверенной области
                raise ValueError(  # ошибка
                    f"call to {self.value} outside certified support: {vals}"  # текст
                )
            return callee.program.eval(vals, env, _budget, _fold_depth)  # результат программы вызываемого на аргументах
        if self.op == "anf":  # алгебраическая нормальная форма
            n_vars, coeffs = self.value  # число переменных и список (маска S, коэффициент c)
            mask_x = 0  # битовая маска входа
            for i in range(n_vars):  # по переменным
                b = int(xs[i])  # значение бита
                if b not in (0, 1):  # не булево
                    raise ValueError("anf expects boolean inputs")  # ошибка
                if b:  # бит 1
                    mask_x |= 1 << (n_vars - 1 - i)  # установить бит n_vars − 1 − i
            acc = 0  # аккумулятор
            for s, c in coeffs:  # по мономам
                if (s & mask_x) == s:  # все биты монома S входят во вход: S ⊆ x
                    acc ^= c  # XOR коэффициента: f = ⊕_{S ⊆ x} c_S
            return int(acc)  # значение АНФ
        if self.op in _RANGE_FOLD_OPS:  # свёртка диапазона
            if len(self.args) != 3:  # не три аргумента
                raise ValueError(f"{self.op} expects start, stop, body")  # ошибка
            # Статически вложенные свёртки отклоняются, динамически вложенные через
            # вызовы отклоняются передаваемым флагом глубины.
            if _fold_depth >= 1 or any(_expr_contains_range(a) for a in self.args):  # уже внутри свёртки или есть вложенная свёртка
                raise ValueError(  # ошибка
                    "nested range folds (including calls) are not supported"  # текст
                )
            start = int(self.args[0].eval(xs, env, _budget, _fold_depth))  # начало диапазона
            stop = int(self.args[1].eval(xs, env, _budget, _fold_depth))  # конец диапазона (не включая)
            span = max(0, stop - start)  # длина диапазона max(0, stop − start)
            if span > MAX_RANGE_EVAL or _budget[0] + span > MAX_RANGE_EVAL:  # диапазон или суммарный бюджет превышает предел
                raise ValueError(  # ошибка
                    f"cumulative range-fold budget exceeds {MAX_RANGE_EVAL} iterations"  # текст
                )
            body = self.args[2]  # тело свёртки
            if self.op == "all_range":  # «для всех k»
                for k in range(start, stop):  # по k от start до stop − 1
                    _budget[0] += 1  # счётчик итераций + 1
                    if not body.eval(  # тело ложно
                        tuple(xs) + (int(k),), env, _budget, _fold_depth + 1  # на входах xs + k
                    ):
                        return 0  # результат 0
                return 1  # все истинны — 1
            if self.op == "any_range":  # «существует k»
                for k in range(start, stop):  # по k
                    _budget[0] += 1  # счётчик + 1
                    if body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1):  # тело истинно
                        return 1  # результат 1
                return 0  # ничего не истинно — 0
            if self.op == "sum_range":  # сумма по k
                acc = 0  # аккумулятор
                for k in range(start, stop):  # по k
                    _budget[0] += 1  # счётчик итераций + 1
                    acc += int(  # прибавить значение тела
                        body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1)  # значение тела body(xs, k)
                    )
                    if abs(acc).bit_length() > 4096:  # сумма стала длиннее 4096 бит
                        raise OverflowError(  # переполнение
                            "range-fold accumulator exceeds 4096-bit cap"  # текст ошибки
                        )
                return acc  # сумма Σ_k body(xs, k)
            acc = 1  # произведение начинается с 1
            for k in range(start, stop):  # по k
                _budget[0] += 1  # счётчик + 1
                acc *= int(  # умножить на значение тела
                    body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1)  # значение тела body(xs, k)
                )
                if abs(acc).bit_length() > 4096:  # произведение длиннее 4096 бит
                    raise OverflowError("range-fold product exceeds 4096-bit cap")  # переполнение
            return acc  # произведение Π_k body(xs, k)
        if self.op == "if_cond":  # условное выражение
            cond_val = self.args[0].eval(xs, env, _budget, _fold_depth)  # значение условия
            return (  # вернуть
                self.args[1].eval(xs, env, _budget, _fold_depth)  # ветвь «то», если условие истинно
                if cond_val  # при cond_val ≠ 0
                else self.args[2].eval(xs, env, _budget, _fold_depth)  # иначе ветвь «иначе»
            )
        vals = [a.eval(xs, env, _budget, _fold_depth) for a in self.args]  # значения всех аргументов
        if self.op == "poly":  # многочлен одной переменной
            acc = 0  # аккумулятор
            for ci in reversed(list(self.value or ())):  # от старшего коэффициента к младшему
                acc = acc * vals[0] + int(ci)  # схема Горнера: acc = acc·x + c_i
            return int(acc)  # значение p(x)
        if self.op == "poly_nd":  # многочлен нескольких переменных
            total = 0  # сумма
            for coef, powers in self.value or ():  # по слагаемым (коэффициент, показатели)
                term = int(coef)  # слагаемое начинается с коэффициента
                for x_val, p in zip(xs, powers):  # по парам (переменная, степень)
                    term *= int(x_val) ** int(p)  # слагаемое term = c·Π x_i^{p_i}
                total += term  # сумма total = Σ слагаемых
            return int(total)  # значение многочлена
        if self.op == "absdiff":  # модуль разности
            return abs(vals[0] - vals[1])  # модуль разности |a − b|
        if self.op == "fma":  # умножение-сложение
            return vals[0] * vals[1] + vals[2]  # произведение плюс слагаемое a·b + c
        if self.op == "clamp":  # ограничение снизу и сверху
            return min(max(vals[0], vals[1]), vals[2])  # ограничение min(max(x, lo), hi)
        if self.op == "median3":  # медиана трёх
            return sorted(vals)[1]  # средний элемент отсортированного списка
        if self.op == "clamp3":  # ограничение отрезком [−3, 3]
            return min(max(vals[0], -3), 3)  # ограничение min(max(x, −3), 3)
        if self.op == "mul3":  # произведение трёх
            return vals[0] * vals[1] * vals[2]  # произведение a·b·c
        if self.op == "isqrt":  # целочисленный корень
            value = int(vals[0])  # целое значение
            if value < 0:  # отрицательное
                raise ValueError("isqrt is undefined for negative integers")  # ошибка
            return math.isqrt(value)  # целый корень ⌊√value⌋
        if self.op == "reified":  # именованная операция
            rfn = REIFIED_V8.get(str(self.value))  # её функция по имени
            if rfn is None:  # не зарегистрирована
                raise ValueError(f"reified op not registered: {self.op}")  # ошибка
        # листовой макрос над ИСХОДНЫМИ входами (на листе аргументов нет;
        # композиция получается обёртыванием в арифметические операции)
            return int(rfn(*[int(v) for v in xs]))  # применить к целым исходным входам
        fn = h5.OPS.get(self.op)  # операция V5 по имени
        if fn is None:  # неизвестна
            raise ValueError(f"unsupported op: {self.op}")  # ошибка
        return int(fn(*vals))  # применить к значениям аргументов

    @property  # свойство
    def cost(self) -> int:  # стоимость выражения в узлах (с запоминанием)
        c = self.__dict__.get("_cost_memo")  # запомненное значение
        if c is None:  # ещё не вычислено
            if self.op in ("var", "const"):  # лист
                c = 1  # стоимость 1
            elif self.op == "anf":  # АНФ
                _, coeffs = self.value  # список мономов
                c = 1 + len(coeffs)  # 1 + число мономов
            elif self.op == "poly":  # многочлен
                c = 1 + len(self.value or ())  # 1 + число коэффициентов
            elif self.op == "poly_nd":  # многомерный многочлен
                c = 1 + len(self.value or ())  # 1 + число слагаемых
            else:  # прочая операция
                c = 1 + sum(a.cost for a in self.args)  # 1 + сумма стоимостей аргументов
            object.__setattr__(self, "_cost_memo", c)  # запомнить (объект неизменяемый)
        return c  # стоимость

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:  # длина префиксного кода выражения, бит
        if self.op == "var":  # переменная
            return TOKEN_BITS + gamma_bits(int(self.value or 0) + 1)  # токен + γ(номер + 1)
        if self.op == "const":  # константа
            return TOKEN_BITS + gamma_bits(zigzag(int(self.value)) + 1)  # токен + γ(zigzag(c) + 1)
        if self.op == "call":  # вызов
            idx = expert_index.get(str(self.value), 0) if expert_index else 0  # номер вызываемого эксперта (0, если нет)
            return (  # вернуть
                TOKEN_BITS  # токен
                + gamma_bits(idx + 1)  # + γ(номер + 1)
                + sum(a.bits(expert_index) for a in self.args)  # + сумма длин аргументов
            )
        if self.op == "anf":  # АНФ
            n_vars, coeffs = self.value  # число переменных и мономы
            payload = gamma_bits(n_vars + 1) + gamma_bits(len(coeffs) + 1)  # γ(n + 1) + γ(число мономов + 1)
            payload += sum(gamma_bits(s + 1) for s, _ in coeffs)  # + Σ γ(маска + 1)
            return TOKEN_BITS + payload  # токен + нагрузка
        if self.op == "poly":  # многочлен
            coeffs = tuple(int(v) for v in (self.value or ()))  # коэффициенты
            payload = gamma_bits(len(coeffs) + 1) + sum(  # γ(число + 1) + Σ
                gamma_bits(zigzag(v) + 1) for v in coeffs  # код γ(zigzag(c) + 1) по коэффициентам
            )
            return TOKEN_BITS + payload + sum(a.bits(expert_index) for a in self.args)  # токен + нагрузка + аргументы
        if self.op == "poly_nd":  # многомерный многочлен
            terms = tuple(self.value or ())  # слагаемые
            payload = gamma_bits(len(terms) + 1)  # γ(число слагаемых + 1)
            for c, powers in terms:  # по слагаемым
                payload += gamma_bits(zigzag(int(c)) + 1) + gamma_bits(len(powers) + 1)  # + γ(zigzag(c) + 1) + γ(число переменных + 1)
                payload += sum(gamma_bits(int(p) + 1) for p in powers)  # + Σ γ(степень + 1)
            return TOKEN_BITS + payload  # токен + нагрузка
        return TOKEN_BITS + sum(a.bits(expert_index) for a in self.args)  # токен + сумма по аргументам

    def pretty(self) -> str:  # текстовое представление выражения
        if self.op == "var":  # переменная
            idx = int(self.value or 0)  # номер
            return VAR_NAMES[idx] if idx < len(VAR_NAMES) else f"x{idx}"  # имя из списка либо x{номер}
        if self.op == "const":  # константа
            return str(self.value)  # её текст
        if self.op == "call":  # вызов
            return f"{self.value}({', '.join(a.pretty() for a in self.args)})"  # имя(аргументы через запятую)
        if self.op == "anf":  # АНФ
            n_vars, coeffs = self.value  # число переменных и мономы
            terms = []  # тексты мономов
            for s, _ in coeffs:  # по маскам мономов
                if s == 0:  # пустой моном
                    terms.append("1")  # константа 1
                else:  # непустой
                    vs = [  # имена входящих переменных
                        VAR_NAMES[i] if i < len(VAR_NAMES) else f"x{i}"  # имя переменной i
                        for i in range(n_vars)  # по переменным
                        if (s >> (n_vars - 1 - i)) & 1  # бит i маски установлен
                    ]
                    terms.append("&".join(vs))  # моном — переменные через &
            return "anf(" + (" ^ ".join(terms) if terms else "0") + ")"  # anf(мономы через ^) или anf(0)
        if self.op in _RANGE_FOLD_OPS:  # свёртка диапазона
            label = {  # имя операции для текста
                "all_range": "forall",  # метка «для всех»
                "any_range": "exists",  # метка «существует»
                "sum_range": "sum",  # метка «сумма»
                "prod_range": "prod",  # метка «произведение»
            }[self.op]  # по операции
            return f"{label}[y={self.args[0].pretty()}..{self.args[1].pretty()}):{self.args[2].pretty()}]"  # имя[y=начало..конец):тело
        if self.op == "if_cond":  # условное выражение
            return f"({self.args[1].pretty()} if {self.args[0].pretty()} else {self.args[2].pretty()})"  # (то если условие иначе иначе)
        if self.op == "poly":  # многочлен
            return h5.Expr("poly", (h5.Expr.var(0),), self.value).pretty()  # текст через выражение V5
        if self.op == "poly_nd":  # многомерный многочлен
            return h5.Expr("poly_nd", (), self.value).pretty()  # текст через выражение V5
        if self.op == "clamp3":  # ограничение [−3, 3]
            return f"min(max({self.args[0].pretty()}, -3), 3)"  # текст min(max(x, −3), 3)
        if self.op == "absdiff":  # модуль разности
            return f"abs({self.args[0].pretty()} - {self.args[1].pretty()})"  # текст abs(a − b)
        if len(self.args) == 1:  # один аргумент
            return f"{self.op}({self.args[0].pretty()})"  # операция(аргумент)
        if len(self.args) == 2:  # два аргумента
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"  # (левый операция правый)
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"  # операция(аргументы через запятую)

    def subtrees(self) -> list["ExprV8"]:  # все поддеревья, начиная с самого выражения
        out = [self]  # начать с себя
        for a in self.args:  # по аргументам
            out.extend(a.subtrees())  # добавить их поддеревья
        return out  # список

    def replace_subtree(self, target: "ExprV8", replacement: "ExprV8") -> "ExprV8":  # заменить поддерево target на replacement
        if self == target:  # само выражение — цель
            return replacement  # вернуть замену
        if not self.args:  # лист, не равный цели
            return self  # без изменений
        return ExprV8(  # тот же узел с заменами внутри аргументов
            self.op,  # операция
            tuple(a.replace_subtree(target, replacement) for a in self.args),  # рекурсивно обработанные аргументы
            self.value,  # значение
        )

    def to_dict(self) -> dict[str, Any]:  # словарь для сериализации
        val = self.value  # значение
        if self.op == "anf" and val is not None:  # АНФ: кортежи превращаются в списки
            val = [int(val[0]), [[int(s), int(c)] for s, c in val[1]]]  # [n, [[маска, коэффициент], ...]]
        return {"op": self.op, "value": val, "args": [a.to_dict() for a in self.args]}  # словарь

    @staticmethod  # статический метод
    def from_dict(d: dict[str, Any]) -> "ExprV8":  # восстановить выражение из словаря
        op = str(d["op"])  # операция
        val = d.get("value")  # значение
        if op == "poly" and val is not None:  # многочлен
            val = tuple(int(v) for v in val)  # кортеж целых коэффициентов
        elif op == "poly_nd" and val is not None:  # многомерный многочлен
            val = tuple((int(t[0]), tuple(int(p) for p in t[1])) for t in val)  # кортеж (c, (степени))
        elif op == "anf" and val is not None:  # АНФ
            val = (int(val[0]), tuple((int(s), int(c)) for s, c in val[1]))  # (n, ((маска, коэффициент), ...))
        return ExprV8(op, tuple(ExprV8.from_dict(a) for a in d.get("args", [])), val)  # выражение


@dataclass(frozen=True)  # неизменяемый класс данных
class ProgramV8:  # программа: выражение либо маска Венна
    arity: int  # число входов
    expr: Optional[ExprV8] = None  # выражение
    venn_mask: Optional[int] = None  # 4-битная маска таблицы истинности

    @property  # свойство
    def cost(self) -> int:  # стоимость программы
        return (  # вернуть
            4  # 4 для маски Венна
            if self.venn_mask is not None  # если маска задана
            else (self.expr.cost if self.expr else 10**9)  # иначе стоимость выражения, а без выражения 10^9
        )

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:  # длина кода программы, бит
        if self.venn_mask is not None:  # маска Венна
            return TOKEN_BITS + 4  # токен + 4 бита маски
        if self.expr is None:  # выражения нет
            return 10**9  # бесконечно большая длина
        return self.expr.bits(expert_index)  # длина кода выражения

    def eval(  # вычислить программу на входах xs
        self,  # сама программа
        xs: Sequence[int],  # входы
        env: Optional[dict[str, "ExpertV8"]] = None,  # среда вызовов
        _budget: Optional[list[int]] = None,  # общий счётчик итераций
        _fold_depth: int = 0,  # глубина свёрток
    ) -> int:  # вернуть целое значение
        if _budget is None:  # счётчик не передан
            _budget = [0]  # новый
        if len(xs) != self.arity:  # число входов не совпадает
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")  # ошибка
        if self.venn_mask is not None:  # маска Венна
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):  # не два булевых входа
                raise ValueError("BoolVenn expects two binary inputs")  # ошибка
            return (int(self.venn_mask) >> ((int(xs[0]) << 1) | int(xs[1]))) & 1  # бит маски № (2·x_1 + x_0)
        if self.expr is None:  # выражения нет
            raise RuntimeError("empty program")  # ошибка
        return self.expr.eval(xs, env, _budget, _fold_depth)  # вычислить выражение

    def pretty(self) -> str:  # текст программы
        if self.venn_mask is not None:  # маска Венна
            return f"BoolVenn(mask={self.venn_mask:04b})"  # текст с маской в 4 битах
        assert self.expr is not None  # выражение обязательно есть
        return self.expr.pretty()  # текст выражения

    def to_dict(self) -> dict[str, Any]:  # словарь для сериализации
        return {  # вернуть словарь
            "arity": self.arity,  # число входов
            "venn_mask": self.venn_mask,  # маска
            "expr": self.expr.to_dict() if self.expr else None,  # словарь выражения либо None
        }

    @staticmethod  # статический метод
    def from_dict(d: dict[str, Any]) -> "ProgramV8":  # восстановить программу из словаря
        return ProgramV8(  # программа
            int(d["arity"]),  # число входов
            ExprV8.from_dict(d["expr"]) if d.get("expr") else None,  # выражение либо None
            d.get("venn_mask"),  # маска
        )


# ==============================================================================
# 2. голова Венна, упаковка Венна и двигатель АНФ
# ==============================================================================


def venn_atoms(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:  # базис из четырёх атомов Венна на входах {−1, +1}
    """Точный базис из четырёх нейронов ReLU для атомов Венна на входах {−1, +1}:
            каждый столбец — ReLU(±s1 ± s2)/2, ровно один атом равен 1 для каждой пары."""
    return np.column_stack(  # столбцы — атомы
        [  # список столбцов
            np.maximum(0.0, (-s1 - s2) * 0.5),  # ReLU((−s1 − s2)/2) — атом (−1, −1)
            np.maximum(0.0, (-s1 + s2) * 0.5),  # ReLU((−s1 + s2)/2) — атом (−1, +1)
            np.maximum(0.0, (s1 - s2) * 0.5),  # ReLU((s1 − s2)/2) — атом (+1, −1)
            np.maximum(0.0, (s1 + s2) * 0.5),  # ReLU((s1 + s2)/2) — атом (+1, +1)
        ]
    )


def venn_head_eval(mask: int, x_bin: Sequence[int]) -> int:  # значение головы Венна: маска на паре битов
    s1 = np.array([2.0 * int(x_bin[0]) - 1.0])  # s1 = 2·x_1 − 1 (бит → ±1)
    s2 = np.array([2.0 * int(x_bin[1]) - 1.0])  # второй сигнал s2 = 2·x_2 − 1
    atoms = venn_atoms(s1, s2)[0]  # атомы — строка из 4 значений
    w = np.array([(mask >> idx) & 1 for idx in range(4)], dtype=np.float64)  # веса — биты маски
    return int(round(float(atoms @ w)))  # round(Σ_i атом_i·w_i)


def pack_venn_16_flat() -> bytes:  # упаковать 16 масок × 4 бита в 13 байт
    """Упаковать 16 масок по 4 трита в 13 байт плоским кодом (плотная нижняя граница:
            64 трита по 5 на байт в основании 3 дают 13 байт)."""
    trits = []  # триты (здесь — биты) упаковки
    for mask in range(16):  # по маскам
        for idx in range(4):  # по битам
            trits.append((mask >> idx) & 1)  # бит idx маски
    while len(trits) % 5 != 0:  # пока длина не кратна 5
        trits.append(0)  # дополнить нулём
    out = bytearray()  # выходные байты
    for off in range(0, len(trits), 5):  # по группам из 5
        b = 0  # байт-накопитель
        for t in trits[off : off + 5]:  # по тритам группы
            b = b * 3 + t  # b = 3·b + t (основание 3: значение ≤ 80 < 256)
        out.append(b)  # добавить байт
    assert len(out) == 13  # ровно 13 байт
    return bytes(out)  # упакованные байты


def unpack_venn_16_flat(buf: bytes) -> list[int]:  # распаковать 13 байт в 16 масок
    assert len(buf) == 13  # ожидается 13 байт
    trits = []  # триты
    for b in buf:  # по байтам
        chunk = [0] * 5  # группа из 5 тритов
        v = b  # остаток
        for i in range(4, -1, -1):  # от младшего трита к старшему
            chunk[i] = v % 3  # трит i = v mod 3
            v //= 3  # остаток v = ⌊v / 3⌋
        trits.extend(chunk)  # добавить группу
    masks = []  # маски
    for m in range(16):  # по маскам
        val = 0  # значение маски
        for idx in range(4):  # по битам
            val |= (trits[m * 4 + idx] & 1) << idx  # установить бит idx из трита 4·m + idx
        masks.append(val)  # добавить маску
    return masks  # 16 масок


def anf_transform(truth_table: Sequence[int]) -> list[int]:  # преобразование Жегалкина: таблица истинности → коэффициенты АНФ, O(n·2^n)
    """Точное преобразование Жегалкина («бабочка»): таблица истинности длины 2^n →
            коэффициенты АНФ; сложность O(n·2^n)."""
    n_tot = len(truth_table)  # длина таблицы
    if n_tot == 0 or (n_tot & (n_tot - 1)) != 0:  # не степень двойки
        raise ValueError("truth_table length must be a power of 2")  # ошибка
    coeffs = [int(v) & 1 for v in truth_table]  # копия таблицы по модулю 2
    step = 1  # шаг бабочки 1
    while step < n_tot:  # пока шаг меньше длины
        for i in range(0, n_tot, step * 2):  # по блокам длины 2·step
            for j in range(step):  # по первой половине блока
                coeffs[i + step + j] ^= coeffs[i + j]  # вторая половина XOR первая: c[i+step+j] ⊕= c[i+j]
        step <<= 1  # шаг ·2
    return coeffs  # коэффициенты АНФ


def anf_support_vars(coeffs: Sequence[int], n_vars: int) -> tuple[int, ...]:  # номера переменных, входящих хоть в один моном
    used = 0  # маска использованных переменных
    for s, c in enumerate(coeffs):  # по коэффициентам (номер = маска монома)
        if c:  # коэффициент ненулевой
            used |= s  # объединить маски
    return tuple(i for i in range(n_vars) if (used >> (n_vars - 1 - i)) & 1)  # переменные с установленным битом


def anf_to_expr_v8(coeffs: Sequence[int], n_vars: int) -> ExprV8:  # коэффициенты АНФ → выражение ExprV8
    sparse = tuple((s, 1) for s, c in enumerate(coeffs) if c)  # ненулевые мономы (маска, 1)
    if len(sparse) == 0:  # мономов нет
        return ExprV8.const(0)  # константа 0
    if len(sparse) == 1 and sparse[0][0] == 0:  # только константный моном
        return ExprV8.const(1)  # константа 1
    if len(sparse) <= 4:  # не более четырёх мономов — явное дерево
        mons = []  # мономы-выражения
        for s, _ in sparse:  # по мономам
            if s == 0:  # константный
                mons.append(ExprV8.const(1))  # моном — константа 1
            else:  # произведение переменных
                vs = [  # переменные монома
                    ExprV8.var(i) for i in range(n_vars) if (s >> (n_vars - 1 - i)) & 1  # переменная i, если бит маски установлен
                ]
                t = vs[0]  # начать с первой
                for v in vs[1:]:  # по остальным
                    t = ExprV8("bit_and", (t, v))  # t = t И v
                mons.append(t)  # добавить моном
        res = mons[0]  # начать с первого монома
        for m in mons[1:]:  # по остальным
            res = ExprV8("bit_xor", (res, m))  # результат res = res XOR m
        return res  # выражение
    return ExprV8("anf", (), (n_vars, sparse))  # иначе — компактный узел АНФ


# ==============================================================================
# 3. сертифицированный безопасный по диапазону шаг CE, EF21-Muon и битовое выделение RVQ
# ==============================================================================


def batch_logit_safe_step_ce(  # сертифицированный шаг логистической регрессии по перекрёстной энтропии
    X_feat: np.ndarray,  # матрица признаков (B, d)
    y_int: np.ndarray,  # метки классов (B,)
    num_classes: int,  # число классов
    steps: int = 80,  # число шагов 80
    l2_reg: Optional[np.ndarray] = None,  # L2-регуляризация по признакам (нет по умолчанию)
    eps_pert: float = 0.0,  # допуск возмущения 0
) -> dict[str, Any]:  # вернуть словарь итогов шага
    """Сертифицированный шаг: η = 4·A / R'² с воротами 4·eps; отслеживаются поля
            шага CertStepGen. Аргументы: X_feat (B, d), y_int (B,), num_classes, steps,
            l2_reg, eps_pert. Возвращает словарь итогов шага."""
    X_feat = np.asarray(X_feat, dtype=np.float64)  # признаки — float64
    raw_y = np.asarray(y_int)  # метки как массив
    if (  # проверка меток
        raw_y.ndim != 1  # не одномерные
        or not np.all(np.isfinite(raw_y))  # или есть нечисловые
        or np.any(raw_y != np.floor(raw_y))  # или нецелые
    ):
        raise ValueError("y_int must be a one-dimensional array of integer labels")  # ошибка
    y_int = raw_y.astype(np.int64)  # метки — int64
    if X_feat.ndim != 2 or X_feat.shape[0] == 0 or X_feat.shape[1] == 0:  # не матрица или пуста
        raise ValueError("X_feat must be a nonempty rank-2 matrix")  # ошибка
    B, d = X_feat.shape  # B — число строк, d — число признаков
    if not np.all(np.isfinite(X_feat)):  # есть нечисловые признаки
        raise ValueError("X_feat must contain only finite values")  # ошибка
    if int(num_classes) < 2:  # классов меньше двух
        raise ValueError("num_classes must be at least 2")  # ошибка
    if y_int.shape != (B,) or np.any(y_int < 0) or np.any(y_int >= int(num_classes)):  # размер меток неверен или метки вне 0..K−1
        raise ValueError("y_int must contain one valid class index per row")  # ошибка
    if int(steps) < 0:  # шагов меньше нуля
        raise ValueError("steps must be nonnegative")  # ошибка
    if not math.isfinite(float(eps_pert)) or float(eps_pert) < 0:  # допуск не конечен или отрицателен
        raise ValueError("eps_pert must be finite and nonnegative")  # ошибка
    theta = np.zeros((int(num_classes), d), dtype=np.float64)  # веса θ — нулевая матрица (K, d)
    reg = (  # регуляризация
        np.zeros(d, dtype=np.float64)  # нулевой вектор d
        if l2_reg is None  # если не задана
        else np.asarray(l2_reg, dtype=np.float64)  # иначе — заданный вектор
    )
    if reg.shape != (d,) or not np.all(np.isfinite(reg)) or np.any(reg < 0):  # не длины d, есть нечисловые или отрицательные
        raise ValueError("l2_reg must be a finite nonnegative vector of length d")  # ошибка

    def ce_loss(th: np.ndarray) -> float:  # перекрёстная энтропия для весов th
        lg = X_feat @ th.T  # логиты z = X·θᵀ
        lg_s = lg - np.max(lg, axis=1, keepdims=True)  # логиты минус максимум по строке: z − max_k z_k
        lse = np.max(lg, axis=1) + np.log(np.sum(np.exp(lg_s), axis=1))  # logsumexp: максимум по классам плюс логарифм суммы exp(z − max z)
        target_l = lg[np.arange(B), y_int]  # логит правильного класса z_{y_n}
        reg_term = 0.5 * float(np.sum(reg[None, :] * (th**2)))  # регуляризация 0.5·Σ reg_j·θ_j²
        return float(np.mean(lse - target_l)) + reg_term  # CE = mean_n(lse_n − z_{y_n}) + регуляризация

    loss0 = ce_loss(theta)  # начальные потери
    cur_loss = loss0  # текущие потери
    accepted = held = 0  # счётчики принятых и удержанных шагов
    for _ in range(steps):  # по шагам
        lg = X_feat @ theta.T  # логиты z = X·θᵀ
        lg_s = lg - np.max(lg, axis=1, keepdims=True)  # z − max z по строке
        exp_l = np.exp(lg_s)  # экспоненты exp(z − max z)
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)  # софтмакс p_k = exp(z_k − max z) / Σ_j exp(z_j − max z)
        res = probs.copy()  # остаток — копия вероятностей
        res[np.arange(B), y_int] -= 1.0  # на месте правильного класса p_{y_n} − 1
        g_bar = (res.T @ X_feat) / B + (reg[None, :] * theta)  # средний градиент ḡ = (1/B)·resᵀ·X + reg ⊙ θ
        D = -g_bar  # направление спуска D = −ḡ
        A = float(np.sum(g_bar * g_bar))  # квадрат нормы A = ‖ḡ‖²
        if A < 1e-18:  # градиент почти нулевой
            held += 1  # шаг удержан
            break  # стоп
        d_logits = X_feat @ D.T  # изменение логитов Δz = X·Dᵀ
        r_prime = np.max(d_logits, axis=1) - np.min(d_logits, axis=1)  # размах по классам r'_n = max_k Δz_nk − min_k Δz_nk
        Q_prime = float(np.mean(r_prime * r_prime))  # средний квадрат размаха Q' = mean_n(r'_n²)
        curv_reg = (  # кривизна регуляризации
            2.0 * float(np.max(reg)) * float(np.sum(D * D)) if np.any(reg > 0) else 0.0  # 2·max(reg)·‖D‖², если регуляризация есть, иначе 0
        )
        denom = Q_prime + 4.0 * curv_reg  # знаменатель Q' + 4·кривизна
        if denom <= 1e-18:  # знаменатель почти нулевой
            held += 1  # шаг удержан
            break  # стоп
        eta = 4.0 * A / denom  # размер шага η = 4A / (Q' + 4·кривизна)
        cert_drop = 2.0 * (A**2) / denom  # гарантированное убывание 2A² / знаменатель
        if eps_pert > 0.0 and 4.0 * eps_pert >= cert_drop:  # возмущение съедает всю гарантию
            held += 1  # шаг удержан
            break  # стоп
        cand = theta + eta * D  # кандидат θ' = θ + η·D
        cand_loss = ce_loss(cand)  # потери кандидата
        if cand_loss <= cur_loss + 1e-12:  # потери не выросли
            theta = cand  # принять
            cur_loss = cand_loss  # обновить потери
            accepted += 1  # принятых + 1
        else:  # потери выросли
            held += 1  # шаг удержан
            break  # стоп
    return {  # словарь итогов
        "theta": theta,  # итоговые веса
        "loss0": loss0,  # начальные потери
        "loss_final": cur_loss,  # конечные потери
        "total_drop": loss0 - cur_loss,  # полное убывание
        "accepted_steps": accepted,  # принято шагов
        "hold_steps": held,  # удержано шагов
    }


def newton_schulz_orthogonalize(G: np.ndarray, iters: int = 5) -> np.ndarray:  # ортогонализация Ньютона — Шульца: итерации X ← X(3I − XᵀX)/2
    """Ортогонализация Ньютона — Шульца: X ← X(3I − XᵀX)/2, iters итераций;
    нормирует вход, при высоком входе работает с транспонированным."""
    X = G.astype(np.float64).copy()  # копия в float64
    norm = float(np.linalg.norm(X))  # норма Фробениуса ‖X‖
    if norm < 1e-12:  # почти нулевая
        return X  # без изменений
    X /= norm  # нормировать X = X / ‖X‖
    transposed = False  # признак транспонирования
    if X.shape[0] > X.shape[1]:  # строк больше, чем столбцов
        X = X.T  # транспонировать
        transposed = True  # запомнить
    I_m = np.eye(X.shape[0], dtype=np.float64)  # единичная матрица размера m×m
    for _ in range(iters):  # по итерациям
        X = 0.5 * (3.0 * I_m - X @ X.T) @ X  # итерация X ← (3I − X·Xᵀ)·X / 2
    return X.T if transposed else X  # вернуть, вернув транспонирование при необходимости


def quantize_5block_single(  # квантовать блок из 5 значений в двоичные уровни с 5-тритовым кодом
    blk: np.ndarray,  # блок из 5 значений
) -> tuple[np.ndarray, tuple[int, int, int], int]:  # вернуть (квантованный блок, (экспонента, уровень, код), номер уровня)
    """Квантование блока из 5 значений: 255 двоичных уровней (c + 1)/128 и код из 5
    тритов; выбор уровня минимизирует ошибку, проверяется её верхняя граница."""
    blk = np.asarray(blk, dtype=np.float64)  # массив float64
    if blk.shape != (5,) or not np.all(np.isfinite(blk)):  # не 5 значений или есть нечисловые
        raise ValueError("quantize_5block_single expects five finite values")  # ошибка
    m = float(np.max(np.abs(blk)))  # максимум модуля m = max |b_i|
    if m == 0.0:  # все нули
        # Пять нулевых тритов кодируются цифрами основания 3, равными 1 (смещённый код +1).
        return np.zeros_like(blk), (0, 0, 121), 0  # нулевой блок, код 121 = 1 + 3 + 9 + 27 + 81 (пять смещённых нулей)
    exp2 = math.ceil(math.log2(m / 1.5 + 1e-18))  # экспонента e = ⌈log2(m / 1.5)⌉
    sc = 2.0**exp2  # масштаб sc = 2^e
    bn = blk / sc  # нормированный блок b = blk / sc
    s = DYADIC_LEVELS[:, None]  # уровни s — столбец
    q_int = np.where(  # триты по каждому уровню
        bn[None, :] / s > 0.5, 1.0, np.where(bn[None, :] / s < -0.5, -1.0, 0.0)  # t = 1, если b/s > 0.5; −1, если b/s < −0.5; иначе 0
    )
    q = q_int * s  # квантованные значения q = t·s
    errs = np.sum((bn[None, :] - q) ** 2, axis=1)  # ошибка по каждому уровню E_s = Σ (b_i − q_i)²
    best = int(np.argmin(errs))  # лучший уровень — с минимальной ошибкой
    s_best = float(DYADIC_LEVELS[best])  # его шаг s*
    trits = q_int[best].astype(int)  # триты лучшего уровня
    in_mask = np.abs(bn) <= 1.5 * s_best  # не насыщенные координаты: |b_i| ≤ 1.5·s*
    sat_tail = float(np.sum(np.where(~in_mask, (np.abs(bn) - s_best) ** 2, 0.0)))  # хвост насыщения: Σ_{насыщенные} (|b_i| − s*)²
    bound = int(np.sum(in_mask)) * (s_best / 2.0) ** 2 + sat_tail  # верхняя граница ошибки: n_внутри·(s*/2)² + хвост
    assert float(errs[best]) <= bound + 1e-9  # ошибка не выше границы
    bval = 0  # упакованный код
    for t in trits:  # по тритам
        bval = bval * 3 + (int(t) + 1)  # код = Σ (t_i + 1)·3^(4−i), цифры 0..2
    return q[best] * sc, (exp2, best, bval), int(best)  # блок в исходном масштабе, метаданные, номер уровня


def quantize_bitalloc_ef21(  # остаточное векторное квантование с поблочным выделением бит
    Phi: np.ndarray,  # матрица признаков (N, D)
    idx: np.ndarray,  # номера выбранных признаков
    wk: np.ndarray,  # их веса
    y: np.ndarray,  # нормированные целевые значения
    sy: float,  # масштаб выхода
    lam: float,  # параметр регуляризации
    max_extra_stages: int = 10,  # максимум дополнительных стадий 10
) -> tuple[np.ndarray, int, list[Any]]:  # вернуть (квантованные веса, число байт, записи стадий)
    """Поблочное выделение бит и остаточное векторное квантование, учитывающее данные:
    первая стадия квантует все блоки, далее добавляются стадии для блока с наибольшим
    вкладом остатка. Возвращает (веса, число байт, записи стадий)."""
    if idx.size == 0:  # признаков нет
        return np.zeros(Phi.shape[1], dtype=np.float64), 1, []  # нулевые веса, 1 байт, без записей
    mag_order = np.argsort(-np.abs(wk))  # порядок номеров по убыванию |w|
    idx_sorted = idx[mag_order]  # номера в этом порядке
    Sub_s = Phi[:, idx_sorted]  # подматрица выбранных столбцов
    col_energy = np.mean(Sub_s**2, axis=0)  # энергия столбца: средний квадрат
    A = Sub_s.T @ Sub_s + max(lam, 1e-9) * np.eye(idx.size)  # матрица A = SᵀS + max(λ, 1e-9)·I

    k = idx.size  # число выбранных признаков
    n_blocks = (k + 4) // 5  # число блоков по 5: ⌈k / 5⌉
    wq_s = np.zeros(k, dtype=np.float64)  # квантованная сумма стадий
    dw = wk[mag_order].copy()  # веса в порядке убывания модуля
    records = []  # записи (номер блока, метаданные)

    for b in range(n_blocks):  # по блокам — первая стадия
        sl = slice(b * 5, min((b + 1) * 5, k))  # срез 5·b .. min(5·b + 5, k)
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))  # блок, дополненный нулями до 5
        q_blk, meta, _ = quantize_5block_single(blk)  # квантовать блок
        wq_s[sl] += q_blk[: sl.stop - sl.start]  # прибавить квантованные значения
        records.append((b, meta))  # записать стадию

    for _ in range(max_extra_stages):  # дополнительные стадии
        r_data = y - Sub_s @ wq_s  # остаток данных r = y − S·w_q
        mse_raw = float(np.mean((r_data * sy) ** 2))  # его MSE в исходном масштабе
        if mse_raw < 0.01:  # остаток мал
            break  # хватит
        dw = np.linalg.solve(A, Sub_s.T @ r_data)  # поправка Δw = A⁻¹·Sᵀ·r
        blk_errs = [  # важность каждого блока
            float(  # число
                np.sum(  # сумма
                    (dw[b * 5 : min((b + 1) * 5, k)] ** 2)  # квадраты поправок блока
                    * col_energy[b * 5 : min((b + 1) * 5, k)]  # умноженные на энергию столбцов
                )
            )
            for b in range(n_blocks)  # по блокам
        ]
        best_b = int(np.argmax(blk_errs))  # самый важный блок
        if blk_errs[best_b] * (sy * sy) < 1e-5:  # его вклад мал в исходном масштабе
            break  # хватит
        sl = slice(best_b * 5, min((best_b + 1) * 5, k))  # срез блока
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))  # блок до 5 значений
        q_blk, meta, _ = quantize_5block_single(blk)  # квантовать
        wq_s[sl] += q_blk[: sl.stop - sl.start]  # прибавить
        records.append((best_b, meta))  # записать стадию

    out = np.zeros(Phi.shape[1], dtype=np.float64)  # полные веса — нули
    out[idx_sorted] = wq_s  # на места выбранных признаков — квантованная сумма
    nbytes = 1 + k + n_blocks * 3 + (len(records) - n_blocks) * 4  # байты: 1 + k + 3·блоков + 4·доп. стадий
    return out, nbytes, records  # веса, размер, записи


class FactorizedSharedCore:  # общее универсальное многошкальное ядро и модульные квантованные остатки задач
    """Общее универсальное многошкальное ядро C и модульные квантованные остатки
            задач R_i. Поля: W1, b1 — тангенсный слой трёх масштабов (0.8, 1.6, 3.2);
            W1_q, b1_q — его квантованная копия; core_bytes — размер квантованного ядра."""

    def __init__(self, seed: int = 42, tanh_units: int = 48):  # создать ядро с зерном 42 и 48 тангенсными юнитами
        if int(tanh_units) < 3 or int(tanh_units) % 3 != 0:  # меньше 3 или не кратно 3
            raise ValueError("tanh_units must be a positive multiple of 3 (at least 3)")  # ошибка
        rng = np.random.default_rng(seed)  # генератор
        Q_raw = rng.normal(0.0, 1.0, size=(3, tanh_units))  # гауссовы 3×h
        Q_T, _ = np.linalg.qr(Q_raw.T)  # QR-разложение: ортонормированные столбцы
        scales = np.repeat(np.array([0.8, 1.6, 3.2]), tanh_units // 3)  # три масштаба, каждый h/3 раз
        self.W1 = Q_T.T * scales[None, :]  # W1 = Q·scales — форма (3, h)
        self.b1 = rng.uniform(-1.5, 1.5, size=(1, tanh_units))  # смещение b1 равномерно на [−1.5, 1.5]
        core_vec = np.concatenate([self.W1.ravel(), self.b1.ravel()])  # ядро одним вектором
        n_blk = (core_vec.size + 4) // 5  # число блоков по 5
        rem = core_vec.copy()  # остаток
        q_core = np.zeros_like(core_vec)  # квантованное ядро — нули
        for _ in range(3):  # три стадии остаточного квантования
            r5 = np.pad(rem, (0, (-rem.size) % 5)).reshape(n_blk, 5)  # остаток блоками по 5
            q_st = np.concatenate(  # стадия — конкатенация квантованных блоков
                [quantize_5block_single(r5[b])[0] for b in range(n_blk)]  # квантованный блок b
            )[: core_vec.size]  # обрезать дополнение
            q_core += q_st  # добавить стадию
            rem -= q_st  # остаток минус стадия
        self.W1_q = q_core[: self.W1.size].reshape(self.W1.shape)  # квантованная W1
        self.b1_q = q_core[self.W1.size :].reshape(self.b1.shape)  # квантованная b1
        self.core_bytes = 3 * n_blk * 3  # байты ядра: 3 стадии · блоков · 3 байта

    def features(  # признаки ветви branch для входов
        self, branch: str, xs_arr: np.ndarray, sx: float, quantized: bool = True  # ветвь, входы (N, d), масштаб входа, использовать ли квантованное ядро
    ) -> np.ndarray:  # вернуть матрицу признаков
        N, d = xs_arr.shape  # N строк, d входов
        xn = xs_arr.astype(np.float64) / sx  # нормированные входы x / s_x
        if branch == "poly":  # полиномиальная ветвь
            cols = [xn[:, i] for i in range(d)]  # столбцы входов
            fs = [np.ones((N, 1), dtype=np.float64)]  # начать с столбца единиц
            for i in range(d):  # линейные члены
                fs.append(cols[i][:, None])  # линейный член x_i
            for i in range(d):  # квадраты и произведения
                for j in range(i, d):  # по j от i до d − 1
                    fs.append((cols[i] * cols[j])[:, None])  # произведение x_i·x_j
            for i in range(d):  # кубы
                for j in range(i, d):  # по j от i до d − 1
                    for k in range(j, d):  # по k от j до d − 1
                        fs.append((cols[i] * cols[j] * cols[k])[:, None])  # произведение x_i·x_j·x_k
            for i in range(d):  # четвёртые степени
                fs.append((cols[i] ** 4)[:, None])  # четвёртая степень x_i⁴
            return np.hstack(fs)  # соединить столбцы
        if branch == "hinge":  # ветвь шарнирных функций
            fs = [np.ones((N, 1), dtype=np.float64)]  # столбец единиц
            for i in range(d):  # по входам
                fs.append(xn[:, i : i + 1])  # линейный член x_i
                for knot in (-0.375, 0.0, 0.375):  # три узла
                    fs.append(np.maximum(0.0, xn[:, i] - knot)[:, None])  # ReLU(x_i − узел)
                    fs.append(np.maximum(0.0, knot - xn[:, i])[:, None])  # ReLU(узел − x_i)
            for i in range(d):  # попарные разности
                for j in range(i + 1, d):  # по j от i + 1 до d − 1
                    diff = xn[:, i] - xn[:, j]  # разность x_i − x_j
                    fs.append(np.maximum(0.0, diff)[:, None])  # ReLU(разность)
                    fs.append(np.maximum(0.0, -diff)[:, None])  # ReLU(−разность)
            return np.hstack(fs)  # соединить
        if branch == "bit":  # битовая ветвь
            xi = np.rint(xs_arr).astype(np.int64)  # округлённые целые входы
            fs = [np.ones((N, 1), dtype=np.float64)]  # единицы
            for i in range(d):  # по входам
                fs.append((xi[:, i] < 0).astype(np.float64)[:, None])  # знак: 1, если x_i < 0
                for b in range(4):  # младшие 4 бита
                    fs.append(((xi[:, i] >> b) & 1).astype(np.float64)[:, None])  # бит b числа x_i
            if d >= 2:  # для двух входов — совместные биты
                for b in range(4):  # по битам
                    fs.append(  # добавить
                        (((xi[:, 0] >> b) & 1) * ((xi[:, 1] >> b) & 1)).astype(  # И битов: бит b первого × бит b второго
                            np.float64  # как float64
                        )[:, None]  # столбцом
                    )
                fs.append(((xi[:, 0] < 0) * (xi[:, 1] < 0)).astype(np.float64)[:, None])  # оба отрицательны
            return np.hstack(fs)  # соединить
        x_pad = np.zeros((N, 3), dtype=np.float64)  # вход, дополненный нулями до 3
        x_pad[:, : min(3, d)] = xn[:, : min(3, d)]  # первые min(3, d) входов
        W1 = self.W1_q if quantized else self.W1  # квантованная либо точная W1
        b1 = self.b1_q if quantized else self.b1  # квантованная либо точная b1
        h = np.tanh(x_pad @ W1 + b1)  # скрытый слой h = tanh(x·W1 + b1)
        return np.hstack(  # соединить
            [  # открытие списка столбцов
                self.features("poly", xs_arr, sx, quantized),  # полиномиальные признаки
                self.features("hinge", xs_arr, sx, quantized)[:, 1:],  # шарнирные без столбца единиц
                h,  # скрытый слой
            ]
        )

    def fit_residual(  # подобрать остаток (адаптер) под задачу
        self,  # само ядро
        train_x: Sequence[Sequence[int]],  # обучающие входы
        train_y: Sequence[int],  # обучающие выходы
        val_x: Sequence[Sequence[int]],  # отложенные входы
        val_y: Sequence[int],  # отложенные выходы
    ) -> "FactorizedAdapter":  # вернуть адаптер
        tx = np.asarray(train_x, dtype=np.float64)  # обучающие входы
        ty = np.asarray(train_y, dtype=np.float64)  # обучающие выходы
        vx = np.asarray(val_x, dtype=np.float64) if val_x else tx  # отложенные входы (если нет — обучающие)
        vy = np.asarray(val_y, dtype=np.float64) if val_y else ty  # отложенные выходы
        sx = max(1.0, float(np.max(np.abs(tx))))  # масштаб входа s_x = max(1, max|x|)
        sy = max(1.0, float(np.max(np.abs(ty))))  # масштаб выхода s_y
        y = ty / sy  # нормированный выход
        best = (float("inf"), "poly", None, None, 9999)  # лучший вариант: (цель, ветвь, точные веса, квантованные, байты)
        for bname in ("poly", "hinge", "bit", "core"):  # по ветвям
            Phi = self.features(bname, tx, sx, quantized=True)  # признаки
            N, D = Phi.shape  # размеры
            for lam in (1e-7, 1e-4, 1e-2):  # по регуляризациям
                A = Phi.T @ Phi + lam * np.eye(D)  # матрица A = ΦᵀΦ + λ·I
                w = np.linalg.solve(A, Phi.T @ y)  # гребнёвая оценка w = A⁻¹·Φᵀ·y
                order = np.argsort(-np.abs(w))  # порядок признаков по убыванию |w|
                ks = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, D) if k <= D]  # числа отобранных признаков
                for k in ks:  # по ним
                    idx = order[:k]  # первые k признаков
                    Ak = Phi[:, idx].T @ Phi[:, idx] + max(lam, 1e-9) * np.eye(k)  # матрица A_k = Φ_kᵀΦ_k + max(λ, 1e-9)·I
                    wk = np.linalg.solve(Ak, Phi[:, idx].T @ y)  # веса подмножества
                    if np.max(np.abs(wk)) > 30.0:  # слишком большие веса
                        continue  # пропустить
                    wc = np.zeros(D, dtype=np.float64)  # полные точные веса
                    wc[idx] = wk  # на места отобранных
                    wq, nb, _ = quantize_bitalloc_ef21(Phi, idx, wk, y, sy, lam)  # квантовать
                    pred_q = (Phi @ wq) * sy  # квантованные предсказания в исходном масштабе
                    mse_q = float(np.mean((pred_q - ty) ** 2))  # их MSE
                    acc_tr = float(np.mean(np.rint(pred_q) == ty))  # точность округлённых предсказаний
                    obj = mse_q + 0.002 * nb + (1.0 - acc_tr) * 0.5  # цель: MSE + 0.002·байты + 0.5·(1 − точность)
                    if obj < best[0]:  # лучше рекорда
                        best = (obj, bname, wc, wq, nb)  # новый рекорд
                    if mse_q < 1e-3:  # MSE достаточно мала
                        break  # хватит
                if best[0] < 0.05:  # цель достаточно хороша
                    break  # не перебирать λ
            if best[0] < 0.02:  # цель очень хороша
                break  # не перебирать ветви

        _, bname, wc, wq, nb = best  # лучший вариант
        Phi_va = self.features(bname, vx, sx, quantized=True)  # признаки отложенных
        pred_f = (Phi_va @ wc) * sy  # точные предсказания
        pred_q = (Phi_va @ wq) * sy  # квантованные предсказания
        return FactorizedAdapter(  # адаптер
            arity=tx.shape[1],  # число входов
            branch=bname,  # ветвь
            sx=sx,  # масштаб входа
            sy=sy,  # масштаб выхода
            w_float=wc,  # точные веса
            w_quant=wq,  # квантованные веса
            compressed_bytes=nb + 9,  # байты + 9 служебных
            val_exact_acc_float=float(np.mean(np.rint(pred_f) == vy)),  # отложенная точность точных
            val_exact_acc_quant=float(np.mean(np.rint(pred_q) == vy)),  # отложенная точность квантованных
            val_mse_float=float(np.mean((pred_f - vy) ** 2)),  # отложенная MSE точных
            val_mse_quant=float(np.mean((pred_q - vy) ** 2)),  # отложенная MSE квантованных
        )


@dataclass  # класс данных
class FactorizedAdapter:  # адаптер задачи: ветвь признаков и два набора весов
    arity: int  # число входов
    branch: str  # ветвь признаков
    sx: float  # масштаб входа
    sy: float  # масштаб выхода
    w_float: np.ndarray  # точные веса
    w_quant: np.ndarray  # квантованные веса
    compressed_bytes: int  # оценка размера сжатых весов
    val_exact_acc_float: float  # отложенная точность точных
    val_exact_acc_quant: float  # отложенная точность квантованных
    val_mse_float: float  # отложенная MSE точных
    val_mse_quant: float  # отложенная MSE квантованных

    def predict(  # предсказать выход для входа xs
        self, core: FactorizedSharedCore, xs: Sequence[int], quantized: bool = True  # ядро, вход, квантованные ли веса
    ) -> float:  # вернуть вещественное предсказание
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1)  # вход строкой (1, d)
        Phi = core.features(self.branch, x, self.sx, quantized=quantized)  # признаки
        w = self.w_quant if quantized else self.w_float  # веса
        return float((Phi @ w)[0] * self.sy)  # предсказание ŷ = (Φ·w)·s_y

    def to_dict(self) -> dict[str, Any]:  # словарь для сериализации
        # Сохраняются оба плотных массива для точного восстановления состояния.
        # compressed_bytes остаётся оценкой формата, а не размером JSON или памяти.
        return {  # вернуть словарь
            "arity": self.arity,  # число входов
            "branch": self.branch,  # ветвь
            "sx": self.sx,  # масштаб входа
            "sy": self.sy,  # масштаб выхода
            "w_float": self.w_float.tolist(),  # точные веса списком
            "w_quant": self.w_quant.tolist(),  # квантованные веса списком
            "compressed_bytes": self.compressed_bytes,  # размер
            "val_exact_acc_float": self.val_exact_acc_float,  # точность точных
            "val_exact_acc_quant": self.val_exact_acc_quant,  # точность квантованных
            "val_mse_float": self.val_mse_float,  # MSE точных
            "val_mse_quant": self.val_mse_quant,  # MSE квантованных
        }

    @staticmethod  # статический метод
    def from_dict(d: dict[str, Any]) -> "FactorizedAdapter":  # восстановить адаптер из словаря
        wf = np.asarray(d.get("w_float", d.get("w_quant", [])), dtype=np.float64)  # точные веса (если нет — квантованные)
        wq = np.asarray(d.get("w_quant", []), dtype=np.float64)  # квантованные веса
        if wf.ndim != 1 or wq.ndim != 1 or wf.shape != wq.shape:  # не векторы или разной длины
            raise ValueError("invalid serialized adapter coefficient arrays")  # ошибка
        return FactorizedAdapter(  # адаптер
            int(d["arity"]),  # число входов
            str(d["branch"]),  # ветвь
            float(d["sx"]),  # масштаб входа
            float(d["sy"]),  # масштаб выхода
            wf,  # точные веса
            wq,  # квантованные веса
            int(d.get("compressed_bytes", 0)),  # размер (0 по умолчанию)
            float(d.get("val_exact_acc_float", float("nan"))),  # точность точных (NaN по умолчанию)
            float(d.get("val_exact_acc_quant", float("nan"))),  # точность квантованных
            float(d.get("val_mse_float", float("nan"))),  # MSE точных
            float(d.get("val_mse_quant", float("nan"))),  # MSE квантованных
        )


# ==============================================================================
# 4. непрерывная память GPM, разделение пространств и адаптер EF21-Muon
# ==============================================================================


class ContinualGPMMemory:  # непрерывная память: обучение с защитой подпространства и адаптер EF21-Muon; P_R = I − U·Uᵀ, нулевой дрейф на защищённом слое
    """Память GPM с разделением пространств и оценкой EF21Markov: проектор
            P_R = I − U·Uᵀ обнуляет дрейф весов на защищённом подпространстве U."""

    def __init__(self, d_in: int, d_out: int, seed: int = 0):  # создать память с весами d_in × d_out
        rng = np.random.default_rng(seed)  # генератор
        self.W = rng.normal(0.0, 0.1, size=(d_in, d_out))  # веса W — гауссовы с дисперсией 0.1
        self.U_basis = np.zeros((d_in, 0), dtype=np.float64)  # базис защищённого подпространства — пустой
        self.ef_v = np.zeros_like(self.W)  # состояние марковской оценки EF21 — нули

    def project_orthogonal(self, G: np.ndarray) -> np.ndarray:  # спроецировать G на дополнение защищённого подпространства
        if self.U_basis.shape[1] == 0:  # базиса нет
            return G  # без изменений
        return G - self.U_basis @ (self.U_basis.T @ G)  # проекция (I − U·Uᵀ)·G

    def train_task_ef21_muon(  # обучить задачу методом EF21-Muon; вернуть итоговую MSE
        self, X: np.ndarray, Y: np.ndarray, steps: int = 40, lr: float = 0.2  # входы, выходы, 40 шагов, скорость 0.2
    ) -> float:  # вернуть MSE
        for _ in range(steps):  # по шагам
            err = X @ self.W - Y  # ошибка e = X·W − Y
            grad = (2.0 / len(X)) * (X.T @ err)  # градиент ∇ = (2/N)·Xᵀ·e
            g_proj = self.project_orthogonal(grad)  # проекция градиента
            diff = g_proj - self.ef_v  # разность v̄ = P·∇ − v
            comp = 0.5 * diff  # поправка 0.5·v̄
            self.ef_v = self.ef_v + comp  # оценка v ← v + 0.5·(P·∇ − v)
            step_dir = newton_schulz_orthogonalize(self.ef_v, iters=4)  # ортогонализовать направление
            step_dir = self.project_orthogonal(step_dir)  # и спроецировать
            cur = float(np.mean((X @ self.W - Y) ** 2))  # текущая MSE
            for eta in (lr, lr * 0.5, lr * 0.1, 0.01):  # шаги от lr до 0.01
                cand = self.W - eta * step_dir  # кандидат W' = W − η·направление
                if float(np.mean((X @ cand - Y) ** 2)) < cur:  # MSE уменьшилась
                    self.W = cand  # принять
                    break  # стоп перебора
        return float(np.mean((X @ self.W - Y) ** 2))  # итоговая MSE

    def consolidate_subspace(self, X: np.ndarray, energy_thresh: float = 0.999) -> None:  # закрепить подпространство задачи с долей энергии 0.999
        res_X = (
            X if self.U_basis.shape[1] == 0 else X - (X @ self.U_basis) @ self.U_basis.T
        )
        if float(np.linalg.norm(res_X)) < 1e-10:
            return
        _, S, Vt = np.linalg.svd(res_X, full_matrices=False)
        tot = float(np.sum(S**2))
        if tot < 1e-12:
            return
        cum = np.cumsum(S**2) / tot
        k = int(np.searchsorted(cum, energy_thresh)) + 1
        new_dirs = Vt[:k].T
        if self.U_basis.shape[1] == 0:
            Q, _ = np.linalg.qr(new_dirs)
            self.U_basis = Q
        else:
            combined = np.hstack([self.U_basis, new_dirs])
            Q, _ = np.linalg.qr(combined)
            r = min(Q.shape[1], np.linalg.matrix_rank(combined))
            self.U_basis = Q[:, :r]


# ==============================================================================
# 5. CRAM Energy-Based Reasoning Engine
# ==============================================================================


class CRAMEngine:
    @staticmethod
    def solve_coloring(
        edges: Sequence[tuple[int, int]],
        n_nodes: int,
        num_colors: int,
        givens: Optional[dict[int, int]] = None,
        eps_noise: float = 0.0,
        max_sweeps: int = 800,
        seed: int = 42,
    ) -> dict[str, Any]:
        if int(n_nodes) <= 0:
            raise ValueError("n_nodes must be positive")
        if int(num_colors) <= 0:
            raise ValueError("num_colors must be positive")
        if int(max_sweeps) < 0:
            raise ValueError("max_sweeps must be nonnegative")
        if not math.isfinite(float(eps_noise)) or float(eps_noise) < 0:
            raise ValueError("eps_noise must be finite and nonnegative")
        for a, b in edges:
            if not (0 <= int(a) < n_nodes and 0 <= int(b) < n_nodes):
                raise ValueError(f"edge endpoint outside [0, {n_nodes})")
        adj = [set() for _ in range(n_nodes)]
        for a, b in edges:
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
        adj_list = [sorted(s) for s in adj]
        giv = {int(i): int(c) for i, c in (givens or {}).items()}
        if any(i < 0 or i >= n_nodes for i in giv):
            raise ValueError("given node index outside graph")
        if any(c < 0 or c >= num_colors for c in giv.values()):
            raise ValueError("given color outside palette")
        rng = np.random.default_rng(seed)
        y = [giv.get(i, int(rng.integers(0, num_colors))) for i in range(n_nodes)]

        def energy(st: Sequence[int]) -> int:
            return sum(1 for a, b in edges if st[a] == st[b])

        def sparse_fields(i: int, st: Sequence[int]) -> list[int]:
            f = [0] * num_colors
            for j in adj_list[i]:
                f[st[j]] += 1
            return f

        cur_e = energy(y)
        sweeps = hb_steps = stale = 0
        T = 0.55
        sparse_work = sum(len(adj_list[i]) * (num_colors**2) for i in range(n_nodes))
        dense_work = n_nodes * max(1, n_nodes - 1) * (num_colors**2)

        while sweeps < max_sweeps and cur_e > 0:
            improved = False
            for i in range(n_nodes):
                if i in giv:
                    continue
                fields = sparse_fields(i, y)
                noisy = [
                    f
                    + (
                        float(rng.uniform(-eps_noise, eps_noise))
                        if eps_noise > 0
                        else 0.0
                    )
                    for f in fields
                ]
                cur_f = noisy[y[i]]
                best_val = min(noisy)
                if cur_f - best_val > 2.0 * eps_noise:
                    ties = [c for c, v in enumerate(noisy) if v == best_val]
                    y[i] = ties[int(rng.integers(0, len(ties)))]
                    improved = True
            cur_e = energy(y)
            sweeps += 1
            if cur_e == 0:
                break
            if not improved:
                stale += 1
                if stale >= 8:
                    y = [
                        giv.get(i, int(rng.integers(0, num_colors)))
                        for i in range(n_nodes)
                    ]
                    stale = 0
                    cur_e = energy(y)
                    continue
                for _ in range(min(12, max(3, cur_e * 2))):
                    conf = [
                        i
                        for i in range(n_nodes)
                        if i not in giv and any(y[i] == y[j] for j in adj_list[i])
                    ]
                    if not conf:
                        break
                    pick = conf[int(rng.integers(0, len(conf)))]
                    if rng.random() < 0.4 and adj_list[pick]:
                        free_nb = [j for j in adj_list[pick] if j not in giv]
                        if free_nb:
                            pick = free_nb[int(rng.integers(0, len(free_nb)))]
                    fields = sparse_fields(pick, y)
                    logits = np.array([-f / T for f in fields], dtype=np.float64)
                    logits -= np.max(logits)
                    probs = np.exp(logits)
                    probs /= np.sum(probs)
                    y[pick] = int(rng.choice(num_colors, p=probs))
                    hb_steps += 1
                cur_e = energy(y)

        valid = all(y[a] != y[b] for a, b in edges) and all(
            y[i] == g for i, g in giv.items()
        )
        return {
            "coloring": y,
            "valid": valid,
            "energy": cur_e,
            "sweeps": sweeps,
            "hb_steps": hb_steps,
            # `verified` means a valid solution was actually found; it must
            # be false for a locally consistent but unsolved/unsatisfiable run.
            "verified": bool(valid and cur_e == 0),
            "givens_respected": all(y[i] == g for i, g in giv.items()),
            "sparse_work_per_sweep": sparse_work,
            "dense_work_per_sweep": dense_work,
        }

    @staticmethod
    def solve_latent_maze(
        n_nodes: int,
        adj: Sequence[Sequence[int]],
        goal: int,
        eps_z: float = 1e-6,
        max_iters: int = 100_000,
    ) -> dict[str, Any]:
        """Contractive latent-field iteration with an explicit stopping certificate."""
        if int(n_nodes) <= 0:
            raise ValueError("n_nodes must be positive")
        if len(adj) != n_nodes:
            raise ValueError("adjacency must have one row per node")
        if not (0 <= int(goal) < n_nodes):
            raise ValueError("goal outside graph")
        if not math.isfinite(float(eps_z)) or float(eps_z) <= 0:
            raise ValueError("eps_z must be finite and positive")
        if int(max_iters) <= 0:
            raise ValueError("max_iters must be positive")
        adj_sets: list[tuple[int, ...]] = []
        for i, row in enumerate(adj):
            ns = tuple(sorted(set(int(j) for j in row)))
            if any(j < 0 or j >= n_nodes for j in ns):
                raise ValueError(f"neighbor index outside graph in row {i}")
            adj_sets.append(ns)
        delta = max((len(nb) for nb in adj_sets), default=0)
        m = float(delta + 1)
        q = delta / m
        z = np.zeros(n_nodes, dtype=np.float64)
        iters = 0
        error_bound = float("inf")
        if q > 0:
            predicted = max(1, math.ceil(math.log(eps_z * (1.0 - q)) / math.log(q)))
            run_limit = min(int(max_iters), predicted)
            for _ in range(run_limit):
                z_new = np.array(
                    [
                        (sum(z[j] for j in adj_sets[i]) + (1.0 if i == goal else 0.0))
                        / m
                        for i in range(n_nodes)
                    ],
                    dtype=np.float64,
                )
                diff = float(np.max(np.abs(z_new - z)))
                z = z_new
                iters += 1
                error_bound = diff * q / (1.0 - q)
                if error_bound <= eps_z:
                    break
        else:
            z[goal] = 1.0 / m
            iters = 1
            error_bound = 0.0
        upper = 1.0 / (m - delta)
        if not (
            np.all(np.isfinite(z))
            and np.all(z >= -1e-12)
            and np.all(z <= upper + 1e-12)
        ):
            raise ArithmeticError("latent-field invariant failed")
        converged = bool(error_bound <= eps_z)
        return {
            "z": z,
            "iters": iters,
            "contraction_q": q,
            "upper_bound": upper,
            "error_bound": error_bound,
            "tolerance": float(eps_z),
            "converged": converged,
        }


# ==============================================================================
# 6. AdaptiveChunking, GDNRAM & Bounded-Loss HedgeRouter
# ==============================================================================


def stutter_norm(seq: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def boundary_chunks(
    boundary: Callable[[Any], bool], xs: Sequence[Any]
) -> list[list[Any]]:
    if not xs:
        return []
    chunks = [[xs[0]]]
    for tok in xs[1:]:
        if boundary(tok):
            chunks.append([tok])
        else:
            chunks[-1].append(tok)
    return chunks


def boundary_heads(boundary: Callable[[Any], bool], xs: Sequence[Any]) -> list[Any]:
    return [c[0] for c in boundary_chunks(boundary, xs)]


class GDNRAM:
    """lag-1 associative store/load."""

    def __init__(self, n_slots: int, dim: int):
        self.mem = np.zeros((n_slots, dim), dtype=np.float64)

    def store(
        self, addr: int, val: np.ndarray, gate_beta: float = 1.0, decay_g: float = 0.0
    ) -> None:
        self.mem[addr] = (1.0 - decay_g) * self.mem[addr] * (
            1.0 - gate_beta
        ) + gate_beta * val

    def load(self, addr: int) -> np.ndarray:
        return self.mem[addr].copy()


class HedgeRouter:
    """Numerically stable exponentially weighted router for losses in [0, 1]."""

    def __init__(self, n_experts: int, eta: float = 0.5):
        if int(n_experts) <= 0:
            raise ValueError("n_experts must be positive")
        if not math.isfinite(float(eta)) or not (0.0 < float(eta) <= 1.0):
            raise ValueError("eta must be finite and in (0, 1]")
        self.K = int(n_experts)
        self.eta = float(eta)
        self.weights = np.ones(self.K, dtype=np.float64) / self.K
        self.cum_loss = np.zeros(self.K, dtype=np.float64)

    def update(self, losses_01: np.ndarray) -> np.ndarray:
        losses = np.asarray(losses_01, dtype=np.float64)
        if losses.shape != (self.K,):
            raise ValueError(f"losses must have shape ({self.K},)")
        if (
            not np.all(np.isfinite(losses))
            or np.any(losses < 0.0)
            or np.any(losses > 1.0)
        ):
            raise ValueError("expert losses must be finite and lie in [0, 1]")
        self.cum_loss += losses
        if not np.all(np.isfinite(self.cum_loss)):
            raise OverflowError("cumulative expert loss overflowed")
        # Log-sum-exp avoids the all-zero weight vector after many rounds.
        logw = -self.eta * self.cum_loss
        logw -= float(np.max(logw))
        logz = float(np.log(np.sum(np.exp(logw))))
        logw -= logz
        # The lower bound is checked in log space: w_i >= exp(-eta L_i)/K.
        lower_log = -math.log(self.K) - self.eta * self.cum_loss
        if np.any(logw + 1e-10 < lower_log):
            raise ArithmeticError("Hedge survivor lower-bound invariant failed")
        self.weights = np.exp(logw)
        self.weights /= np.sum(self.weights)
        return self.weights.copy()


# ==============================================================================
# 7. AuditLedger + RAM Governor
# ==============================================================================


@dataclass
class AuditLedger:
    """Explicit alpha accounting for externally defined statistical tests.

    Exact enumeration of a declared finite support is deterministic and does
    not consume alpha. This ledger is NOT itself a confidence-bound procedure.
    """

    delta_total: float = 0.05
    spent: dict[str, float] = field(default_factory=dict)
    rejected_tasks: set[str] = field(default_factory=set)
    final_test_touched: bool = False

    def __post_init__(self) -> None:
        self.delta_total = float(self.delta_total)
        if not math.isfinite(self.delta_total) or not (0.0 < self.delta_total < 1.0):
            raise ValueError("delta_total must be finite and in (0, 1)")
        self.spent = {str(k): float(v) for k, v in self.spent.items()}
        if any(not k or not math.isfinite(v) or v < 0.0 for k, v in self.spent.items()):
            raise ValueError(
                "spent alpha entries must be named, finite, and nonnegative"
            )
        if sum(self.spent.values()) > self.delta_total + 1e-12:
            raise ValueError("serialized alpha spend exceeds delta_total")

    def remaining(self) -> float:
        return max(0.0, self.delta_total - sum(self.spent.values()))

    def spend_anytime(self, task: str, alpha: float) -> bool:
        alpha = float(alpha)
        if (
            self.final_test_touched
            or not task
            or not math.isfinite(alpha)
            or alpha <= 0.0
        ):
            return False
        if alpha > self.remaining() + 1e-12:
            return False
        self.spent[task] = self.spent.get(task, 0.0) + alpha
        return True

    def record_reject(self, task: str) -> None:
        if not task:
            raise ValueError("task name must be nonempty")
        self.rejected_tasks.add(task)

    def mark_final_test_touched(self) -> None:
        """Freeze future alpha spending after the sealed final test is opened."""
        self.final_test_touched = True


def ram_available_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return hagi_v7._ram_available_bytes()


# ==============================================================================
# 8. Fast Piecewise / Modular / Batched-MDL Symbolic Engine + DAG Compressor
# ==============================================================================


def exact_piecewise_poly_fit(
    train_x: Sequence[tuple[int, ...]], train_y: Sequence[int]
) -> Optional[ExprV8]:
    """Exact rational piecewise polynomial / modular residue discovery before AST heap."""
    arity = len(train_x[0])
    if arity == 1:
        xs = [r[0] for r in train_x]
        for m in range(2, 9):
            buckets: dict[int, set[int]] = {}
            for x, y in zip(xs, train_y):
                buckets.setdefault(x % m, set()).add(y)
            if all(len(vs) == 1 for vs in buckets.values()) and len(buckets) >= 2:
                if m == 2 and buckets.get(0) == {0} and buckets.get(1) == {1}:
                    return ExprV8("bit_and", (ExprV8.var(0), ExprV8.const(1)))
        for thr in (-3, -2, -1, 0, 1, 2, 3):
            left = [(x, y) for x, y in zip(xs, train_y) if x < thr]
            right = [(x, y) for x, y in zip(xs, train_y) if x >= thr]
            if len(left) < 3 or len(right) < 3:
                continue
            cl = _fit_poly_or_const(left)
            cr = _fit_poly_or_const(right)
            if cl is not None and cr is not None and cl != cr:
                cond = ExprV8("lt", (ExprV8.var(0), ExprV8.const(thr)))
                return ExprV8("if_cond", (cond, cl, cr))
    elif arity == 2:
        left = [(r, y) for r, y in zip(train_x, train_y) if r[0] < r[1]]
        right = [(r, y) for r, y in zip(train_x, train_y) if r[0] >= r[1]]
        if len(left) >= 6 and len(right) >= 6:
            tl = hagi_v7._exact_multivariate_poly_fit(
                [r for r, _ in left], [y for _, y in left], 2, max_degree=2
            )
            tr = hagi_v7._exact_multivariate_poly_fit(
                [r for r, _ in right], [y for _, y in right], 2, max_degree=2
            )
            if tl is not None and tr is not None and tl != tr:
                cond = ExprV8("lt", (ExprV8.var(0), ExprV8.var(1)))
                return ExprV8(
                    "if_cond",
                    (cond, ExprV8("poly_nd", (), tl), ExprV8("poly_nd", (), tr)),
                )
    return None


def bounded_fold_hypotheses(arity: int) -> list[ExprV8]:
    """Small task-name-independent family of bounded arithmetic predicates.

    The range binder is represented by the next variable slot in the body.
    Hypotheses are still accepted only after held-out and full finite-support
    verification against the task teacher.
    """
    if int(arity) != 1:
        return []
    x, k = ExprV8.var(0), ExprV8.var(1)
    c0, c1, c2 = ExprV8.const(0), ExprV8.const(1), ExprV8.const(2)
    x_ge_2 = ExprV8("ge", (x, c2))
    rem = ExprV8("mod", (x, k))
    not_divisor = ExprV8("ne", (rem, c0))
    divisor = ExprV8("eq", (rem, c0))
    sqrt_stop = ExprV8("add", (ExprV8("isqrt", (ExprV8("max", (x, c0)),)), c1))
    prime = ExprV8(
        "bit_and",
        (
            x_ge_2,
            ExprV8("all_range", (c2, sqrt_stop, not_divisor)),
        ),
    )
    composite = ExprV8(
        "bit_and",
        (
            ExprV8("ge", (x, ExprV8.const(4))),
            ExprV8("any_range", (c2, sqrt_stop, divisor)),
        ),
    )
    even = ExprV8("eq", (ExprV8("mod", (x, c2)), c0))
    odd = ExprV8("ne", (ExprV8("mod", (x, c2)), c0))
    nonnegative = ExprV8("ge", (x, c0))
    square = ExprV8(
        "bit_and",
        (
            nonnegative,
            ExprV8(
                "any_range",
                (
                    c0,
                    ExprV8("add", (x, c1)),
                    ExprV8("eq", (ExprV8("mul", (k, k)), x)),
                ),
            ),
        ),
    )
    # Arithmetic aggregates over positive divisors. These are generic folds,
    # not memorized tables: tau(n), sigma(n), and Euler phi(n).
    positive_stop = ExprV8("add", (ExprV8("max", (x, c0)), c1))
    from_one = ExprV8.const(1)
    divides = ExprV8("eq", (ExprV8("mod", (x, k)), c0))
    coprime = ExprV8("eq", (ExprV8("gcd", (x, k)), c1))
    tau = ExprV8("sum_range", (from_one, positive_stop, divides))
    sigma = ExprV8("sum_range", (from_one, positive_stop, ExprV8("mul", (k, divides))))
    phi = ExprV8("sum_range", (from_one, positive_stop, coprime))
    # Empty product is 1; multiplying by (x >= 0) defines negative inputs as 0.
    factorial = ExprV8(
        "mul",
        (
            ExprV8("ge", (x, c0)),
            ExprV8("prod_range", (from_one, positive_stop, k)),
        ),
    )
    return [prime, composite, square, even, odd, tau, sigma, phi, factorial]


def _fit_poly_or_const(pts: list[tuple[int, int]]) -> Optional[ExprV8]:
    ys = {y for _, y in pts}
    if len(ys) == 1:
        return ExprV8.const(next(iter(ys)))
    coeffs = hagi_v7._exact_poly_fit(
        [x for x, _ in pts], [y for _, y in pts], max_degree=3
    )
    if coeffs is not None:
        return ExprV8("poly", (ExprV8.var(0),), coeffs)
    return None


# V10 Reification wire: invented ops promoted into V8 batched synthesis
# live here (registered per synthesis call; core mirrors from REIFIED_OPS).
REIFIED_V8: dict = {}


def _leaf_iter(leaf_exprs, reified_leaf_fns):
    """Yield (leaf, reified_fn_or_None) aligned with leaf_exprs."""
    ri = 0
    for lf in leaf_exprs:
        if lf.op == "reified":
            yield lf, reified_leaf_fns[ri]
            ri += 1
        else:
            yield lf, None


def synthesize_batched_mdl_v8(
    arity: int,
    train_x: Sequence[tuple[int, ...]],
    train_y: Sequence[int],
    val_x: Sequence[tuple[int, ...]],
    val_y: Sequence[int],
    library_experts: Sequence[tuple[int, str, "ExpertV8"]] = (),
    max_evals: int = 4000,
    max_bits: int = 84,
    deadline: Optional[float] = None,
    reified_ops: Optional[dict] = None,
) -> tuple[Optional[ExprV8], int]:
    """reified_ops: optional {name: (fn_1d, cost_bits)} — invented ops
    exposed as extra LEAVES of the batched synthesis (Reification wire:
    yesterday's invention composes inside tomorrow's programs)."""
    """Batched NumPy int64 synthesis with dedup and Call-DAG library references."""
    try:
        X = np.asarray(train_x, dtype=np.int64)
        Y = np.asarray(train_y, dtype=np.int64)
    except (OverflowError, ValueError, TypeError):
        return None, 0
    if (
        X.ndim != 2
        or X.shape[1] != int(arity)
        or X.shape[0] == 0
        or Y.shape != (X.shape[0],)
    ):
        raise ValueError(
            "training arrays must have shapes (N, arity) and (N,), with N>0"
        )
    # Use Python integers here: np.abs(int64_min) overflows back to int64_min.
    if any(abs(int(v)) > 128 for v in X.reshape(-1)):
        return None, 0
    if int(max_evals) <= 0 or int(max_bits) <= 0:
        return None, 0
    N = X.shape[0]
    leaf_exprs = [ExprV8.var(i) for i in range(arity)] + [
        ExprV8.const(c) for c in hagi_v7.CONSTS
    ]
    # reified leaves: invented ops as unary macros (vectorized over rows)
    reified_leaf_fns = []
    if reified_ops:
        for rname, (rfn, rcost) in reified_ops.items():
            full = f"reified:{rname}"
            REIFIED_V8[full] = rfn
            leaf_exprs.append(ExprV8("reified", (), full))
            reified_leaf_fns.append(rfn)
    L_arr = np.stack(
        [np.ascontiguousarray(X[:, int(lf.value or 0)])
         if lf.op == "var"
         else (np.array([int(rfn(*[int(v) for v in r])) for r in X],
                    dtype=np.int64)
               if lf.op == "reified"
               else np.full(N, int(lf.value), dtype=np.int64))
         for lf, rfn in _leaf_iter(leaf_exprs, reified_leaf_fns)
        ]
    )
    L_bits = [
        (reified_ops[lf.value.split(":", 1)[1]][1]
         if lf.op == "reified" else lf.bits())
        for lf in leaf_exprs
    ]

    for i, lf in enumerate(leaf_exprs):
        if np.array_equal(L_arr[i], Y):
            if all(lf.eval(r) == y for r, y in zip(val_x, val_y)):
                return lf, 1

    seen_sigs = {L_arr[i].tobytes(): 1 for i in range(len(leaf_exprs))}
    buckets: dict[int, list[tuple[ExprV8, np.ndarray]]] = {}
    for i, lf in enumerate(leaf_exprs):
        buckets.setdefault(L_bits[i], []).append((lf, L_arr[i]))

    total_checked = len(leaf_exprs)
    MAG_CAP = 1 << 28
    all_ops = [
        "add",
        "sub",
        "mul",
        "floordiv",
        "mod",
        "gcd",
        "lcm",
        "min",
        "max",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "bit_and",
        "bit_or",
        "bit_xor",
        "bit_nand8",
        "bit_nor8",
        "bit_xnor8",
        "pow_int",
    ]
    env = {name: ex for _, name, ex in library_experts}

    while buckets and total_checked < max_evals:
        if deadline is not None and time.monotonic() >= deadline:
            break
        min_b = min(buckets)
        if min_b >= max_bits:
            break
        batch = buckets.pop(min_b)
        if not batch:
            continue
        if len(batch) > 256:
            batch = batch[:256]
        B_exprs = [x[0] for x in batch]
        V = np.stack([x[1] for x in batch])

        def register(
            res_arr: np.ndarray, builder_fn: Callable[[int, int], ExprV8], new_bits: int
        ) -> Optional[ExprV8]:
            nonlocal total_checked
            if new_bits > max_bits:
                return None
            if res_arr.ndim == 2:
                res_arr = res_arr[:, None, :]
            matches = np.all(res_arr == Y[None, None, :], axis=2)
            if np.any(matches):
                for bi, ki in np.argwhere(matches):
                    cand = builder_fn(int(bi), int(ki))
                    try:
                        if all(cand.eval(r, env) == y for r, y in zip(val_x, val_y)):
                            return cand
                    except Exception:
                        pass
            if new_bits + TOKEN_BITS > max_bits or total_checked >= max_evals:
                return None
            ok_mag = np.max(np.abs(res_arr), axis=2) <= MAG_CAP
            for bi, ki in np.argwhere(ok_mag):
                if total_checked >= max_evals:
                    break
                row = res_arr[bi, ki]
                key = row.tobytes()
                cnt = seen_sigs.get(key, 0)
                if cnt >= 2:
                    continue
                seen_sigs[key] = cnt + 1
                total_checked += 1
                buckets.setdefault(new_bits, []).append(
                    (builder_fn(int(bi), int(ki)), row)
                )
            return None

        for uop, ures in (
            ("neg", -V),
            ("abs", np.abs(V)),
            ("sign", np.sign(V).astype(np.int64)),
            ("bit_not8", (~V) & 0xFF),
            ("clamp3", np.clip(V, -3, 3)),
        ):
            hit = register(
                ures, lambda bi, _, o=uop: ExprV8(o, (B_exprs[bi],)), min_b + TOKEN_BITS
            )
            if hit is not None:
                return hit, total_checked

        for lib_idx, ename, ex in library_experts[:24]:
            cbits = TOKEN_BITS + gamma_bits(lib_idx + 1)
            if ex.program.arity == 1:
                rows_out = np.empty_like(V)
                ok_rows = []
                for bi in range(V.shape[0]):
                    if B_exprs[bi].op == "const":
                        continue
                    try:
                        vals_b = V[bi]
                        if all(ex.supports((int(v),)) for v in vals_b):
                            rows_out[bi] = [
                                ex.program.eval((int(v),), env) for v in vals_b
                            ]
                            ok_rows.append(bi)
                    except Exception:
                        pass
                if ok_rows:
                    sub_r = rows_out[ok_rows]
                    hit = register(
                        sub_r,
                        lambda bi, _, o=ename, idxs=ok_rows: ExprV8(
                            "call", (B_exprs[idxs[bi]],), o
                        ),
                        min_b + cbits,
                    )
                    if hit is not None:
                        return hit, total_checked
            elif ex.program.arity == 2:
                for li, lf in enumerate(leaf_exprs):
                    lv = L_arr[li]
                    for order_idx in (0, 1):
                        rows_out = np.empty_like(V)
                        ok_rows = []
                        for bi in range(V.shape[0]):
                            if B_exprs[bi].op == "const" and lf.op == "const":
                                continue
                            bv = V[bi]
                            a_v, b_v = (bv, lv) if order_idx == 0 else (lv, bv)
                            try:
                                if all(
                                    ex.supports((int(a_v[k]), int(b_v[k])))
                                    for k in range(N)
                                ):
                                    rows_out[bi] = [
                                        ex.program.eval((int(a_v[k]), int(b_v[k])), env)
                                        for k in range(N)
                                    ]
                                    ok_rows.append(bi)
                            except Exception:
                                pass
                        if ok_rows:
                            sub_r = rows_out[ok_rows]
                            hit = register(
                                sub_r,
                                (
                                    lambda bi, _, o=ename, l=lf, idxs=ok_rows, oi=order_idx: ExprV8(
                                        "call",
                                        (
                                            (B_exprs[idxs[bi]], l)
                                            if oi == 0
                                            else (l, B_exprs[idxs[bi]])
                                        ),
                                        o,
                                    )
                                ),
                                min_b + cbits + L_bits[li],
                            )
                            if hit is not None:
                                return hit, total_checked

        Va = V[:, None, :]
        Lb = L_arr[None, :, :]
        for op in all_ops:
            if op == "add":
                R = Va + Lb
            elif op == "sub":
                R = Va - Lb
            elif op == "mul":
                R = Va * Lb
            elif op == "min":
                R = np.minimum(Va, Lb)
            elif op == "max":
                R = np.maximum(Va, Lb)
            elif op == "eq":
                R = (Va == Lb).astype(np.int64)
            elif op == "ne":
                R = (Va != Lb).astype(np.int64)
            elif op == "lt":
                R = (Va < Lb).astype(np.int64)
            elif op == "le":
                R = (Va <= Lb).astype(np.int64)
            elif op == "gt":
                R = (Va > Lb).astype(np.int64)
            elif op == "ge":
                R = (Va >= Lb).astype(np.int64)
            elif op == "bit_and":
                R = Va & Lb
            elif op == "bit_or":
                R = Va | Lb
            elif op == "bit_xor":
                R = Va ^ Lb
            elif op == "bit_nand8":
                R = (~(Va & Lb)) & 0xFF
            elif op == "bit_nor8":
                R = (~(Va | Lb)) & 0xFF
            elif op == "bit_xnor8":
                R = (~(Va ^ Lb)) & 0xFF
            elif op == "floordiv":
                safe = np.where(Lb != 0, Lb, 1)
                R = np.where(
                    np.broadcast_to(Lb != 0, np.broadcast_shapes(Va.shape, Lb.shape)),
                    np.floor_divide(Va, safe),
                    MAG_CAP + 1,
                )
            elif op == "mod":
                safe = np.where(Lb != 0, Lb, 1)
                R = np.where(
                    np.broadcast_to(Lb != 0, np.broadcast_shapes(Va.shape, Lb.shape)),
                    np.mod(Va, safe),
                    MAG_CAP + 1,
                )
            elif op == "gcd":
                R = np.gcd(Va, Lb)
            elif op == "lcm":
                ok_l = (Va != 0) & (Lb != 0)
                sa = np.where(ok_l, Va, 1)
                sb = np.where(ok_l, Lb, 1)
                R = np.where(ok_l, np.abs(sa // np.gcd(sa, sb) * sb), MAG_CAP + 1)
            elif op == "pow_int":
                ok_p = (Lb >= 0) & (Lb <= 8) & (np.abs(Va) <= 100)
                sa = np.where(ok_p, Va, 0)
                sb = np.where(ok_p, Lb, 0)
                R = np.where(ok_p, sa**sb, MAG_CAP + 1)
            else:
                continue
            for li in range(len(leaf_exprs)):
                hit = register(
                    R[:, li : li + 1, :],
                    lambda bi, _, o=op, l=leaf_exprs[li]: ExprV8(o, (B_exprs[bi], l)),
                    min_b + TOKEN_BITS + L_bits[li],
                )
                if hit is not None:
                    return hit, total_checked
            if op in ("sub", "floordiv", "mod", "lt", "le", "gt", "ge", "pow_int"):
                if op == "sub":
                    R2 = Lb - Va
                elif op == "lt":
                    R2 = (Lb < Va).astype(np.int64)
                elif op == "le":
                    R2 = (Lb <= Va).astype(np.int64)
                elif op == "gt":
                    R2 = (Lb > Va).astype(np.int64)
                elif op == "ge":
                    R2 = (Lb >= Va).astype(np.int64)
                elif op == "floordiv":
                    safe = np.where(Va != 0, Va, 1)
                    R2 = np.where(
                        np.broadcast_to(
                            Va != 0, np.broadcast_shapes(Va.shape, Lb.shape)
                        ),
                        np.floor_divide(Lb, safe),
                        MAG_CAP + 1,
                    )
                elif op == "mod":
                    safe = np.where(Va != 0, Va, 1)
                    R2 = np.where(
                        np.broadcast_to(
                            Va != 0, np.broadcast_shapes(Va.shape, Lb.shape)
                        ),
                        np.mod(Lb, safe),
                        MAG_CAP + 1,
                    )
                elif op == "pow_int":
                    ok_p = (Va >= 0) & (Va <= 8) & (np.abs(Lb) <= 100)
                    sa = np.where(ok_p, Lb, 0)
                    sb = np.where(ok_p, Va, 0)
                    R2 = np.where(ok_p, sa**sb, MAG_CAP + 1)
                for li in range(len(leaf_exprs)):
                    hit = register(
                        R2[:, li : li + 1, :],
                        lambda bi, _, o=op, l=leaf_exprs[li]: ExprV8(
                            o, (l, B_exprs[bi])
                        ),
                        min_b + TOKEN_BITS + L_bits[li],
                    )
                    if hit is not None:
                        return hit, total_checked
        for op in ("mul", "add"):
            R_s = V * V if op == "mul" else V + V
            hit = register(
                R_s,
                lambda bi, _, o=op: ExprV8(o, (B_exprs[bi], B_exprs[bi])),
                min_b * 2 + TOKEN_BITS,
            )
            if hit is not None:
                return hit, total_checked

    return None, total_checked


# ==============================================================================
# 9. ExpertV8, TheoremMinerV8 & GrowingHagiProcessorV8
# ==============================================================================


@dataclass
class ExpertV8:
    name: str
    family: str
    program: ProgramV8
    domains: tuple[tuple[int, ...], ...]
    valid_rule: str
    code_length: int
    adapter: Optional[FactorizedAdapter] = None
    parents: tuple[str, ...] = ()

    def supports(self, xs: Sequence[int]) -> bool:
        if len(xs) != len(self.domains):
            return False
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):
            return False
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:
            return False
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "program": self.program.to_dict(),
            "domains": [list(d) for d in self.domains],
            "valid_rule": self.valid_rule,
            "code_length": self.code_length,
            "adapter": self.adapter.to_dict() if self.adapter else None,
            "parents": list(self.parents),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExpertV8":
        return ExpertV8(
            str(d["name"]),
            str(d["family"]),
            ProgramV8.from_dict(d["program"]),
            tuple(tuple(int(v) for v in row) for row in d["domains"]),
            str(d.get("valid_rule", "all")),
            int(d["code_length"]),
            FactorizedAdapter.from_dict(d["adapter"]) if d.get("adapter") else None,
            tuple(str(x) for x in d.get("parents", [])),
        )


class TheoremMinerV8:
    """Scans ALL experts (fixes v7.4's [:16] truncation) — commutativity, involutivity, etc."""

    def mine(self, model: "GrowingHagiProcessorV8") -> list[hagi_v7.Theorem]:
        found: list[hagi_v7.Theorem] = []
        known = {t.name for t in model.theorems}
        env = model.experts

        for e in model.experts.values():
            if e.program.arity == 2:
                dx, dy = e.domains[0], e.domains[1]
                declared = [(x, y) for x in dx for y in dy if e.supports((x, y))]
                swapped = [(x, y) for x, y in declared if e.supports((y, x))]
                if swapped and len(swapped) == len(declared):
                    nm = f"commutativity({e.name})"
                    if nm not in known:
                        try:
                            if all(
                                e.program.eval((x, y), env)
                                == e.program.eval((y, x), env)
                                for x, y in swapped
                            ):
                                th = hagi_v7.Theorem(
                                    nm,
                                    f"for all supported pairs: {e.name}(x,y)={e.name}(y,x)",
                                    len(swapped),
                                )
                                model.theorems.append(th)
                                found.append(th)
                                known.add(nm)
                        except Exception:
                            pass
                self_pairs = [x for x in dx if x in dy and e.supports((x, x))]
                nm = f"self-absorption({e.name})"
                if self_pairs and len(self_pairs) == len(dx) and nm not in known:
                    try:
                        if all(e.program.eval((x, x), env) == x for x in self_pairs):
                            th = hagi_v7.Theorem(
                                nm,
                                f"for all overlapping x: {e.name}(x,x)=x",
                                len(self_pairs),
                            )
                            model.theorems.append(th)
                            found.append(th)
                            known.add(nm)
                    except Exception:
                        pass
                zero_pairs = [x for x in dx if 0 in dy and e.supports((x, 0))]
                nm = f"right-identity-zero({e.name})"
                if zero_pairs and len(zero_pairs) == len(dx) and nm not in known:
                    try:
                        if all(e.program.eval((x, 0), env) == x for x in zero_pairs):
                            th = hagi_v7.Theorem(
                                nm, f"for all x: {e.name}(x,0)=x", len(zero_pairs)
                            )
                            model.theorems.append(th)
                            found.append(th)
                            known.add(nm)
                    except Exception:
                        pass
            elif e.program.arity == 1:
                dom = tuple(int(x) for x in e.domains[0])
                img = {}
                closed = True
                try:
                    for x in dom:
                        y = int(e.program.eval((x,), env))
                        if not e.supports((y,)):
                            closed = False
                            break
                        img[x] = y
                except Exception:
                    closed = False
                if closed:
                    for prop, check_fn in (
                        (
                            "involutivity",
                            lambda: all(
                                e.program.eval((img[x],), env) == x for x in dom
                            ),
                        ),
                        (
                            "idempotence",
                            lambda: all(
                                e.program.eval((img[x],), env) == img[x] for x in dom
                            ),
                        ),
                    ):
                        nm = f"{prop}({e.name})"
                        if nm not in known:
                            try:
                                if check_fn():
                                    th = hagi_v7.Theorem(
                                        nm, f"{prop} on certified domain", len(dom)
                                    )
                                    model.theorems.append(th)
                                    found.append(th)
                                    known.add(nm)
                            except Exception:
                                pass
                if all((-x) in dom for x in dom):
                    for sym_name, sgn in (("even-symmetry", 1), ("odd-symmetry", -1)):
                        nm = f"{sym_name}({e.name})"
                        if nm not in known:
                            try:
                                if all(
                                    e.program.eval((-x,), env)
                                    == sgn * e.program.eval((x,), env)
                                    for x in dom
                                ):
                                    th = hagi_v7.Theorem(
                                        nm,
                                        f"{e.name}(-x)={'+' if sgn==1 else '-'}{e.name}(x)",
                                        len(dom),
                                    )
                                    model.theorems.append(th)
                                    found.append(th)
                                    known.add(nm)
                            except Exception:
                                pass
        return found


def _expr_call_names(expr: Optional[ExprV8]) -> set[str]:
    if expr is None:
        return set()
    names = {str(expr.value)} if expr.op == "call" else set()
    for child in expr.args:
        names.update(_expr_call_names(child))
    return names


def verify_call_dag(model: "GrowingHagiProcessorV8") -> dict[str, Any]:
    """Verify all symbolic expert-call references exist and form an acyclic DAG."""
    graph: dict[str, set[str]] = {}
    missing: list[tuple[str, str]] = []
    for name, expert in model.experts.items():
        refs = _expr_call_names(expert.program.expr)
        for ref in refs:
            if ref not in model.experts:
                missing.append((name, ref))
        graph[name] = refs
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: list[str] = []

    def dfs(node: str, trail: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycles.append(" -> ".join((*trail, node)))
            return
        visiting.add(node)
        for child in graph.get(node, ()):
            if child in graph:
                dfs(child, (*trail, node))
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        dfs(node, ())
    return {
        "valid": not missing and not cycles,
        "missing": missing,
        "cycles": cycles,
        "nodes": len(graph),
        "edges": sum(len(v) for v in graph.values()),
    }


def verify_growth_lineage(model: "GrowingHagiProcessorV8") -> dict[str, Any]:
    """Check parent provenance references and reject cycles in the growth graph."""
    graph: dict[str, tuple[str, ...]] = {}
    missing: list[tuple[str, str]] = []
    for name, expert in model.experts.items():
        graph[name] = tuple(expert.parents)
        for parent in expert.parents:
            if parent not in model.experts:
                missing.append((name, parent))
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: list[str] = []

    def dfs(node: str, trail: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycles.append(" -> ".join((*trail, node)))
            return
        visiting.add(node)
        for parent in graph.get(node, ()):
            if parent in graph:
                dfs(parent, (*trail, node))
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        dfs(node, ())
    return {
        "valid": not missing and not cycles,
        "missing": missing,
        "cycles": cycles,
        "nodes_with_parents": sum(bool(v) for v in graph.values()),
        "edges": sum(len(v) for v in graph.values()),
    }


class GrowingHagiProcessorV8:
    """Unified HAGI v8 processor (finite-support research prototype)."""

    def __init__(
        self, seed: int = 42, node_budget: int = 4000, delta_total: float = 0.05
    ):
        self.core = FactorizedSharedCore(seed=seed)
        self.node_budget = int(node_budget)
        self.experts: dict[str, ExpertV8] = {}
        self.version = 0
        self.total_commits = 0
        self.total_holds = 0
        self.growth_log: list[h5.GrowthEvent] = []
        self.theorems: list[hagi_v7.Theorem] = []
        self.ledger = AuditLedger(delta_total=delta_total)
        self.stats = hagi_v7.SynthesisStats()
        self.router = HedgeRouter(4, eta=0.5)
        self.ram = GDNRAM(n_slots=32, dim=4)

    def _expert_index(self) -> dict[str, int]:
        return {nm: i for i, nm in enumerate(sorted(self.experts))}

    def learn(
        self,
        spec: h5.TaskSpec,
        seed: int = 0,
        train_fraction: float = 0.70,
        poison_one_label: bool = False,
        train_neural: bool = True,
        deadline: Optional[float] = None,
        parents: Sequence[str] = (),
    ) -> h5.GrowthEvent:
        if spec.name in self.experts:
            ev = h5.GrowthEvent(self.version, spec.name, "REUSE", "already learned")
            self.growth_log.append(ev)
            return ev
        parent_names = tuple(dict.fromkeys(str(x) for x in parents))
        unknown_parents = [x for x in parent_names if x not in self.experts]
        if unknown_parents:
            return self._hold(
                spec, f"UNKNOWN_GROWTH_PARENTS:{','.join(unknown_parents)}", 0, 0
            )
        domain = spec.inputs()
        if not domain:
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(domain))
        if spec.family == "BoolVenn" or (
            all(set(d) <= {0, 1} for d in spec.domains) and len(domain) <= 16
        ):
            n_train = max(1, len(domain) - 1)
        else:
            n_train = max(
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))
            )
        train_x = [domain[int(i)] for i in perm[:n_train]]
        val_x = [domain[int(i)] for i in perm[n_train:]]
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}
        train_y = [all_y[xs] for xs in train_x]
        val_y = [all_y[xs] for xs in val_x]
        if poison_one_label and train_y:
            self.ledger.record_reject(spec.name)
            return self._hold(
                spec, "POISONED_LABEL_INJECTION", len(train_x), len(val_x)
            )

        # Finite exhaustive checks are deterministic and do not spend statistical
        # alpha. External statistical tests must reserve alpha explicitly through
        # model.reserve_statistical_test(...).

        adapter: Optional[FactorizedAdapter] = None
        if train_neural and val_x and spec.family != "BoolVenn":
            adapter = self.core.fit_residual(train_x, train_y, val_x, val_y)

        # Tier 0: Certified Library REUSE
        ex_idx = self._expert_index()
        for idx, (nm, ex) in enumerate(sorted(self.experts.items())):
            if ex.program.arity != spec.arity:
                continue
            if not all(ex.supports(xs) for xs in domain):
                continue
            self.stats.evals += len(train_x)
            try:
                if any(
                    ex.program.eval(xs, self.experts) != yy
                    for xs, yy in zip(train_x, train_y)
                ):
                    continue
                if any(
                    ex.program.eval(xs, self.experts) != yy
                    for xs, yy in zip(val_x, val_y)
                ):
                    continue
                if any(ex.program.eval(xs, self.experts) != all_y[xs] for xs in domain):
                    continue
            except Exception:
                continue
            ref_bits = TOKEN_BITS + gamma_bits(idx + 1)
            ev = h5.GrowthEvent(
                self.version,
                spec.name,
                "REUSE",
                f"verified-expert-coverage:{nm}",
                ex.program.pretty(),
                ref_bits,
                len(train_x),
                len(val_x),
                1.0,
            )
            self.growth_log.append(ev)
            return ev

        def try_commit(program: ProgramV8, reason: str) -> Optional[h5.GrowthEvent]:
            try:
                self.stats.evals += len(train_x)
                if any(
                    program.eval(xs, self.experts) != yy
                    for xs, yy in zip(train_x, train_y)
                ):
                    return None
                if val_x and any(
                    program.eval(xs, self.experts) != yy for xs, yy in zip(val_x, val_y)
                ):
                    return None
                if any(program.eval(xs, self.experts) != all_y[xs] for xs in domain):
                    return None
            except Exception:
                return None
            bits = program.bits(ex_idx)
            expert = ExpertV8(
                spec.name,
                spec.family,
                program,
                spec.domains,
                spec.valid_rule,
                bits,
                adapter,
                parent_names,
            )
            self.experts[spec.name] = expert
            self.version += 1
            self.total_commits += 1
            ev = h5.GrowthEvent(
                self.version,
                spec.name,
                "COMMIT",
                reason,
                program.pretty(),
                bits,
                len(train_x),
                len(val_x),
                1.0,
            )
            self.growth_log.append(ev)
            return ev

        # Tier 1: BoolVenn / ANF
        if spec.family == "BoolVenn":
            for m in range(16):
                ev = try_commit(ProgramV8(2, venn_mask=m), "boolvenn-exact")
                if ev is not None:
                    return ev

        if all(set(d) <= {0, 1} for d in spec.domains) and 1 <= spec.arity <= 6:
            if set(train_y) <= {0, 1}:
                full_inputs = list(itertools.product((0, 1), repeat=spec.arity))
                train_map = dict(zip(train_x, train_y))
                missing = [r for r in full_inputs if r not in train_map]
                if len(missing) <= 1:
                    best_anf = None
                    for fill in (0,) if not missing else (0, 1):
                        tt = [train_map.get(r, fill) for r in full_inputs]
                        coeffs = anf_transform(tt)
                        active_s = [s for s, c in enumerate(coeffs) if c]
                        max_deg = max((bin(s).count("1") for s in active_s), default=0)
                        cand_e = anf_to_expr_v8(coeffs, spec.arity)
                        cand_p = ProgramV8(spec.arity, cand_e)
                        score = (max_deg, len(active_s), cand_p.bits(ex_idx))
                        if best_anf is None or score < best_anf[0]:
                            best_anf = (score, cand_p)
                    if best_anf is not None:
                        ev = try_commit(
                            best_anf[1], "anf-zhegalkin-butterfly+full-support"
                        )
                        if ev is not None:
                            return ev

        # Tier 2: Direct v5 primitive templates
        for p5 in sorted(
            h5.candidate_programs(spec.arity), key=lambda p: (p.cost, p.pretty())
        ):
            if p5.expr is not None:
                ev = try_commit(
                    ProgramV8(spec.arity, ExprV8.from_h5(p5.expr)),
                    "direct-template+full-support",
                )
                if ev is not None:
                    return ev

        # Tier 3: Exact Rational Polynomial (1D..3D)
        if 1 <= spec.arity <= 3 and not spec.family.startswith("Bool"):
            if spec.arity == 1:
                coeffs = hagi_v7._exact_poly_fit(
                    [r[0] for r in train_x], train_y, max_degree=5
                )
                if coeffs is not None:
                    ev = try_commit(
                        ProgramV8(1, ExprV8("poly", (ExprV8.var(0),), coeffs)),
                        "exact-rational-poly+full-support",
                    )
                    if ev is not None:
                        return ev
            else:
                terms = hagi_v7._exact_multivariate_poly_fit(
                    train_x, train_y, spec.arity, max_degree=4
                )
                if terms is not None:
                    ev = try_commit(
                        ProgramV8(spec.arity, ExprV8("poly_nd", (), terms)),
                        "exact-multivar-poly+full-support",
                    )
                    if ev is not None:
                        return ev

        # Tier 4: Exact Piecewise Polynomial & Modular
        pw = exact_piecewise_poly_fit(train_x, train_y)
        if pw is not None:
            ev = try_commit(
                ProgramV8(spec.arity, pw), "exact-piecewise-poly+full-support"
            )
            if ev is not None:
                return ev

        # Tier 5: Bounded arithmetic folds (forall/exists over finite integer ranges).
        # This is a generic hypothesis family, not a task-name-specific shortcut.
        if spec.arity == 1:
            fold_candidates = sorted(
                bounded_fold_hypotheses(spec.arity),
                key=lambda expr: (expr.bits(ex_idx), expr.pretty()),
            )
            for fold_expr in fold_candidates:
                ev = try_commit(
                    ProgramV8(spec.arity, fold_expr),
                    "bounded-arithmetic-fold+full-support",
                )
                if ev is not None:
                    return ev

        # Tier 6: Batched Vectorized MDL AST Synthesis with Library Call-DAG
        lib_list = [
            (i, nm, self.experts[nm]) for i, nm in enumerate(sorted(self.experts))
        ]
        hit_expr, n_chk = synthesize_batched_mdl_v8(
            spec.arity,
            train_x,
            train_y,
            val_x,
            val_y,
            library_experts=lib_list,
            max_evals=self.node_budget,
            max_bits=84,
            deadline=deadline,
        )
        self.stats.candidates_generated += n_chk
        if hit_expr is not None:
            ev = try_commit(
                ProgramV8(spec.arity, hit_expr), "batched-mdl-dag+full-support"
            )
            if ev is not None:
                return ev

        return self._hold(spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x))

    def _hold(
        self, spec: h5.TaskSpec, reason: str, ntrain: int, nval: int
    ) -> h5.GrowthEvent:
        self.total_holds += 1
        ev = h5.GrowthEvent(
            self.version, spec.name, "HOLD", reason, "", 0, ntrain, nval, 0.0
        )
        self.growth_log.append(ev)
        return ev

    def compress_library_mdl(self) -> dict[str, Any]:
        """Global MDL Refactoring: subtree -> call(expert) when strictly fewer bits
        and exact equality over the complete certified support."""
        ex_idx = self._expert_index()
        before_bits = sum(e.code_length for e in self.experts.values())
        refactored = 0
        for target_name, target_ex in list(self.experts.items()):
            if target_ex.program.expr is None:
                continue
            cur_expr = target_ex.program.expr
            subs = [
                s
                for s in cur_expr.subtrees()
                if s.op not in ("var", "const", "call") and s.cost >= 2
            ]
            if not subs:
                continue
            domain = [
                xs
                for xs in itertools.product(*target_ex.domains)
                if target_ex.supports(xs)
            ]
            for sub in subs:
                for lib_name, lib_ex in list(self.experts.items()):
                    if (
                        lib_name == target_name
                        or lib_ex.program.arity != target_ex.program.arity
                    ):
                        continue
                    if lib_ex.program.arity == 2:
                        call_e = ExprV8(
                            "call", (ExprV8.var(0), ExprV8.var(1)), lib_name
                        )
                    elif lib_ex.program.arity == 1:
                        call_e = ExprV8("call", (ExprV8.var(0),), lib_name)
                    else:
                        continue
                    if call_e.bits(ex_idx) >= sub.bits(ex_idx):
                        continue
                    cand_expr = cur_expr.replace_subtree(sub, call_e)
                    cand_prog = ProgramV8(target_ex.program.arity, cand_expr)
                    new_bits = cand_prog.bits(ex_idx)
                    if new_bits >= target_ex.code_length:
                        continue
                    # A bit-saving substitution is admissible only if the global
                    # call graph stays acyclic; exact output equality alone cannot
                    # prevent recursive alias loops.
                    old_program = target_ex.program
                    target_ex.program = cand_prog
                    dag_check = verify_call_dag(self)
                    target_ex.program = old_program
                    if not dag_check["valid"]:
                        continue
                    try:
                        if all(
                            cand_prog.eval(xs, self.experts)
                            == target_ex.program.eval(xs, self.experts)
                            for xs in domain
                        ):
                            target_ex.program = cand_prog
                            target_ex.code_length = new_bits
                            cur_expr = cand_expr
                            refactored += 1
                    except Exception:
                        pass
        after_bits = sum(e.code_length for e in self.experts.values())
        return {
            "before_bits": before_bits,
            "after_bits": after_bits,
            "refactored": refactored,
        }

    def predict(self, name: str, *xs: int, mode: str = "exact") -> dict[str, Any]:
        ex = self.experts.get(name)
        if ex is None:
            return {"status": "ABSTAIN", "reason": "unknown operation"}
        if not ex.supports(xs):
            return {"status": "ABSTAIN", "reason": "outside certified finite domain"}
        if mode == "exact":
            return {
                "status": "OK",
                "value": ex.program.eval(xs, self.experts),
                "source": "verified-symbolic-dag",
                "model_version": self.version,
            }
        if mode == "neural":
            if ex.adapter is None:
                if ex.program.venn_mask is not None:
                    return {
                        "status": "APPROXIMATE",
                        "value": float(venn_head_eval(ex.program.venn_mask, xs)),
                        "source": "venn-head-exact-ternary",
                        "model_version": self.version,
                    }
                return {"status": "ABSTAIN", "reason": "neural adapter unavailable"}
            return {
                "status": "APPROXIMATE",
                "value": ex.adapter.predict(self.core, xs, quantized=True),
                "source": "factorized-core-bitalloc-rvq",
                "model_version": self.version,
            }
        raise ValueError("mode must be exact or neural")

    def verify_all(self, specs: Sequence[h5.TaskSpec]) -> dict[str, Any]:
        total = correct = 0
        for sp in specs:
            ex = self.experts.get(sp.name)
            if ex is None:
                continue
            rows = sp.inputs()
            ok = sum(
                int(ex.program.eval(xs, self.experts) == int(sp.teacher(*xs)))
                for xs in rows
            )
            total += len(rows)
            correct += ok
        return {
            "checked": total,
            "correct": correct,
            "accuracy": correct / max(1, total),
            "committed": len(self.experts),
        }

    def reserve_statistical_test(self, task: str, alpha: float) -> bool:
        """Reserve alpha for an explicitly implemented external statistical test."""
        return self.ledger.spend_anytime(task, alpha)

    def verify_lineage(self) -> dict[str, Any]:
        calls = verify_call_dag(self)
        growth = verify_growth_lineage(self)
        return {
            "valid": calls["valid"] and growth["valid"],
            "call_dag": calls,
            "growth_lineage": growth,
        }

    def save(self, path: str | Path) -> None:
        """Atomically persist the complete v8 state, including shared-core weights."""
        audit = self.verify_lineage()
        if not audit["valid"]:
            raise ValueError(f"cannot save invalid model graph: {audit}")
        data = {
            "format": "HAGI-V8-STATE",
            "version": V8_VERSION,
            "node_budget": self.node_budget,
            "model_version": self.version,
            "total_commits": self.total_commits,
            "total_holds": self.total_holds,
            "core": {
                "W1": self.core.W1.tolist(),
                "b1": self.core.b1.tolist(),
                "W1_q": self.core.W1_q.tolist(),
                "b1_q": self.core.b1_q.tolist(),
                "core_bytes": self.core.core_bytes,
            },
            "experts": {
                name: ex.to_dict() for name, ex in sorted(self.experts.items())
            },
            "theorems": [vars(t) for t in self.theorems],
            "growth_log": [vars(ev) for ev in self.growth_log],
            "ledger": {
                "delta_total": self.ledger.delta_total,
                "spent": dict(self.ledger.spent),
                "rejected_tasks": sorted(self.ledger.rejected_tasks),
                "final_test_touched": self.ledger.final_test_touched,
            },
            "router": {
                "K": self.router.K,
                "eta": self.router.eta,
                "weights": self.router.weights.tolist(),
                "cum_loss": self.router.cum_loss.tolist(),
            },
            "ram": {"mem": self.ram.mem.tolist()},
            "stats": {
                "candidates_generated": self.stats.candidates_generated,
                "evals": self.stats.evals,
                "sieve_survivors": self.stats.sieve_survivors,
            },
            "audit": audit,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temp, target)

    @classmethod
    def load(cls, path: str | Path) -> "GrowingHagiProcessorV8":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "HAGI-V8-STATE":
            raise ValueError("not a HAGI v8 state file")
        core_data = data.get("core", {})
        w1 = np.asarray(core_data.get("W1"), dtype=np.float64)
        b1 = np.asarray(core_data.get("b1"), dtype=np.float64)
        w1q = np.asarray(core_data.get("W1_q"), dtype=np.float64)
        b1q = np.asarray(core_data.get("b1_q"), dtype=np.float64)
        if w1.ndim != 2 or w1.shape[0] != 3 or b1.shape != (1, w1.shape[1]):
            raise ValueError("invalid shared-core float tensor shapes")
        if w1q.shape != w1.shape or b1q.shape != b1.shape:
            raise ValueError("invalid shared-core quantized tensor shapes")
        model = cls(
            seed=0,
            node_budget=int(data.get("node_budget", 4000)),
            delta_total=float(data.get("ledger", {}).get("delta_total", 0.05)),
        )
        model.core.W1, model.core.b1 = w1, b1
        model.core.W1_q, model.core.b1_q = w1q, b1q
        model.core.core_bytes = int(core_data.get("core_bytes", 0))
        model.experts = {
            name: ExpertV8.from_dict(ex) for name, ex in data.get("experts", {}).items()
        }
        model.version = int(data.get("model_version", 0))
        model.total_commits = int(data.get("total_commits", len(model.experts)))
        model.total_holds = int(data.get("total_holds", 0))
        model.theorems = [hagi_v7.Theorem(**t) for t in data.get("theorems", [])]
        model.growth_log = [h5.GrowthEvent(**ev) for ev in data.get("growth_log", [])]
        led = data.get("ledger", {})
        model.ledger = AuditLedger(
            float(led.get("delta_total", 0.05)),
            {str(k): float(v) for k, v in led.get("spent", {}).items()},
            set(str(x) for x in led.get("rejected_tasks", [])),
            bool(led.get("final_test_touched", False)),
        )
        rd = data.get("router", {})
        model.router = HedgeRouter(int(rd.get("K", 4)), float(rd.get("eta", 0.5)))
        model.router.weights = np.asarray(
            rd.get("weights", model.router.weights), dtype=np.float64
        )
        model.router.cum_loss = np.asarray(
            rd.get("cum_loss", model.router.cum_loss), dtype=np.float64
        )
        if model.router.weights.shape != (
            model.router.K,
        ) or model.router.cum_loss.shape != (model.router.K,):
            raise ValueError("invalid serialized router vectors")
        if (
            not np.all(np.isfinite(model.router.weights))
            or np.any(model.router.weights < 0)
            or not np.isclose(model.router.weights.sum(), 1.0)
        ):
            raise ValueError("invalid serialized router weights")
        mem = np.asarray(
            data.get("ram", {}).get("mem", model.ram.mem), dtype=np.float64
        )
        if mem.shape != model.ram.mem.shape or not np.all(np.isfinite(mem)):
            raise ValueError("invalid serialized GDN RAM state")
        model.ram.mem = mem
        st = data.get("stats", {})
        model.stats.candidates_generated = int(st.get("candidates_generated", 0))
        model.stats.evals = int(st.get("evals", 0))
        model.stats.sieve_survivors = int(st.get("sieve_survivors", 0))
        audit = model.verify_lineage()
        if not audit["valid"]:
            raise ValueError(f"loaded model graph is invalid: {audit}")
        return model

    def total_neural_bytes(self) -> int:
        """Estimated packed payload bytes, not Python RAM or serialized file size."""
        venn_bytes = (
            13 if any(e.family == "BoolVenn" for e in self.experts.values()) else 0
        )
        adapter_bytes = sum(
            e.adapter.compressed_bytes
            for e in self.experts.values()
            if e.adapter is not None
        )
        return self.core.core_bytes + venn_bytes + adapter_bytes


# ==============================================================================
# 10. Non-Circular Self-Developing Growth Loop (grow_v8)
# ==============================================================================


def propose_noncircular_task(
    experts: dict[str, ExpertV8],
    rng: np.random.Generator,
    round_idx: int,
) -> Optional[tuple[h5.TaskSpec, tuple[str, ...]]]:
    """Novel non-degenerate composed tasks WITHOUT handing the AST to learn()."""
    arith2 = [
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"
    ]
    arith1 = [
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"
    ]
    if not arith2:
        return None
    pool = arith2 + arith1
    for _ in range(20):
        outer = arith2[int(rng.choice(len(arith2)))]
        inner = pool[int(rng.choice(len(pool)))]
        if inner.program.arity == 1:
            xs = tuple(
                x
                for x in inner.domains[0]
                if abs(x) <= 6
                and inner.supports((x,))
                and inner.program.eval((x,), experts) in outer.domains[0]
            )
            ys = tuple(
                y
                for y in outer.domains[1]
                if abs(y) <= 6
                and all(
                    outer.supports((inner.program.eval((x,), experts), y)) for x in xs
                )
            )
            if len(xs) < 3 or len(ys) < 3:
                continue
            vals = {
                outer.program.eval((inner.program.eval((x,), experts), y), experts)
                for x in xs
                for y in ys
            }
            if len(vals) <= 2:
                continue
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(
                (i.program.eval((int(x),), env), int(y)), env
            )
            sp = h5.TaskSpec(
                f"V8_AUTO_R{round_idx}_{inner.name}_IN_{outer.name}",
                2,
                (xs, ys),
                teacher,
                valid_rule=outer.valid_rule,
                family="auto-composed",
            )
            return sp, (outer.name, inner.name)
        else:
            xs = tuple(
                x for x in inner.domains[0] if abs(x) <= 5 and x in outer.domains[0]
            )
            ys_cand = tuple(y for y in inner.domains[1] if abs(y) <= 5)
            ys = tuple(
                y
                for y in ys_cand
                if all(
                    inner.supports((x, y))
                    and outer.supports((x, inner.program.eval((x, y), experts)))
                    for x in xs
                )
            )
            if len(xs) < 3 or len(ys) < 3:
                continue
            vals = {
                outer.program.eval((x, inner.program.eval((x, y), experts)), experts)
                for x in xs
                for y in ys
            }
            if len(vals) <= 2:
                continue
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(
                (int(x), i.program.eval((int(x), int(y)), env)), env
            )
            sp = h5.TaskSpec(
                f"V8_AUTO_R{round_idx}_{inner.name}_IN_{outer.name}",
                2,
                (xs, ys),
                teacher,
                valid_rule=inner.valid_rule,
                family="auto-composed",
            )
            return sp, (outer.name, inner.name)
    return None


def grow_v8(
    model: GrowingHagiProcessorV8,
    suite: Sequence[h5.TaskSpec],
    max_rounds: int = 4,
    train_neural: bool = True,
    out_dir: str = "work/v8_grow",
) -> dict[str, Any]:
    if int(max_rounds) < 0:
        raise ValueError("max_rounds must be nonnegative")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rnd = np.random.default_rng(20260101)
    rounds = []
    for r in range(max_rounds):
        t0 = time.time()
        commits = 0
        for sp in suite:
            ev = model.learn(
                sp, seed=10_000 + r * 7919 + len(sp.name), train_neural=train_neural
            )
            if ev.action == "COMMIT":
                commits += 1
        proposed = 0
        for k in range(8):
            prop = propose_noncircular_task(model.experts, rnd, r)
            if prop is None:
                continue
            sp, parents = prop
            if sp.name in model.experts:
                continue
            proposed += 1
            ev = model.learn(
                sp,
                seed=500_000 + r * 104_729 + k,
                train_neural=train_neural,
                parents=parents,
            )
            if ev.action == "COMMIT":
                commits += 1
        comp = model.compress_library_mdl()
        new_th = TheoremMinerV8().mine(model)
        audit = model.verify_lineage()
        if not audit["valid"]:
            raise RuntimeError(
                f"growth produced invalid provenance/call graph: {audit}"
            )
        rounds.append(
            {
                "round": r,
                "experts": len(model.experts),
                "commits": commits,
                "proposed": proposed,
                "theorems_total": len(model.theorems),
                "theorems_new": len(new_th),
                "sym_bits": comp["after_bits"],
                "dag_refactored": comp["refactored"],
                "estimated_packed_bytes": model.total_neural_bytes(),
                "unique_programs": len(
                    {
                        json.dumps(e.program.to_dict(), sort_keys=True)
                        for e in model.experts.values()
                    }
                ),
                "experts_with_parents": audit["growth_lineage"]["nodes_with_parents"],
                "lineage_edges": audit["growth_lineage"]["edges"],
                "ram_avail_gb": round(ram_available_bytes() / 2**30, 2),
                "elapsed": round(time.time() - t0, 3),
            }
        )
        model.save(out / f"model_r{r}.json")
    final_audit = model.verify_lineage()
    return {
        "rounds": rounds,
        "experts": len(model.experts),
        "theorems": len(model.theorems),
        "lineage": final_audit["growth_lineage"],
        "call_dag": final_audit["call_dag"],
    }


def self_test_v8() -> None:
    suite = h5.make_task_suite()
    model = GrowingHagiProcessorV8(node_budget=6000)
    for i, sp in enumerate(suite):
        ev = model.learn(sp, seed=10_000 + i, train_neural=True)
        assert ev.action in ("COMMIT", "REUSE"), (sp.name, ev)
    check = model.verify_all(suite)
    assert check["accuracy"] == 1.0, check
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"
    poisoned = GrowingHagiProcessorV8().learn(
        next(s for s in suite if s.name == "ADD"), seed=77, poison_one_label=True
    )
    assert poisoned.action == "HOLD"
    rep = grow_v8(model, suite, max_rounds=2, out_dir=".omc/v8_grow_test")
    assert rep["experts"] >= len(suite)
    # Task-name-independent bounded-fold induction: train only on [-20, 20],
    # then evaluate the discovered range predicate beyond the certified domain.
    prime_domain = tuple(range(-20, 21))
    prime_task = h5.TaskSpec(
        "BOOLEAN_PATTERN_A",
        1,
        (prime_domain,),
        lambda x: int(x >= 2 and all(x % d for d in range(2, math.isqrt(x) + 1))),
        family="unseen-boolean-arithmetic",
    )
    prime_model = GrowingHagiProcessorV8(node_budget=1000)
    prime_event = prime_model.learn(prime_task, seed=88, train_neural=False)
    assert (
        prime_event.action == "COMMIT" and "forall" in prime_event.program
    ), prime_event
    prime_expr = prime_model.experts[prime_task.name].program
    for n in (101, 103, 104, 997, 1009, 1024, 104729, 104730, 1_000_003):
        expected = int(n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1)))
        assert prime_expr.eval((n,), prime_model.experts) == expected, n
    # Bounded-fold arithmetic aggregates: divisor count/sum and Euler totient.
    # The task labels are neutral; the hypothesis family does not inspect names.
    aggregate_domain = tuple(range(-10, 51))
    aggregate_tasks = [
        h5.TaskSpec(
            "ARITHMETIC_AGGREGATE_A",
            1,
            (aggregate_domain,),
            lambda n: 0 if n <= 0 else sum(1 for d in range(1, n + 1) if n % d == 0),
            family="unseen-integer-aggregate",
        ),
        h5.TaskSpec(
            "ARITHMETIC_AGGREGATE_B",
            1,
            (aggregate_domain,),
            lambda n: 0 if n <= 0 else sum(d for d in range(1, n + 1) if n % d == 0),
            family="unseen-integer-aggregate",
        ),
        h5.TaskSpec(
            "ARITHMETIC_AGGREGATE_C",
            1,
            (aggregate_domain,),
            lambda n: (
                0 if n <= 0 else sum(1 for d in range(1, n + 1) if math.gcd(n, d) == 1)
            ),
            family="unseen-integer-aggregate",
        ),
        h5.TaskSpec(
            "ARITHMETIC_AGGREGATE_D",
            1,
            (aggregate_domain,),
            lambda n: 0 if n < 0 else math.factorial(n),
            family="unseen-integer-aggregate",
        ),
    ]
    aggregate_model = GrowingHagiProcessorV8(node_budget=1000)
    for aggregate_task in aggregate_tasks:
        aggregate_event = aggregate_model.learn(
            aggregate_task, seed=606, train_neural=False
        )
        assert aggregate_event.action == "COMMIT", (
            aggregate_task.name,
            aggregate_event,
        )
        assert aggregate_model.verify_all([aggregate_task])["accuracy"] == 1.0
    aggregate_expressions = [
        aggregate_model.experts[t.name].program for t in aggregate_tasks
    ]
    for value in (51, 60, 97, 100, 997):
        expected = [
            sum(1 for d in range(1, value + 1) if value % d == 0),
            sum(d for d in range(1, value + 1) if value % d == 0),
            sum(1 for d in range(1, value + 1) if math.gcd(value, d) == 1),
        ]
        actual = [
            expr.eval((value,), aggregate_model.experts)
            for expr in aggregate_expressions[:3]
        ]
        assert actual == expected, (value, actual, expected)
        if value <= 100:
            assert aggregate_expressions[3].eval(
                (value,), aggregate_model.experts
            ) == math.factorial(value)
    for value in (51, 60, 97, 100, 500):
        assert aggregate_expressions[3].eval(
            (value,), aggregate_model.experts
        ) == math.factorial(value)
    # Explicit product overflow cap; 542! exceeds the 4096-bit accumulator cap.
    with np.testing.assert_raises(OverflowError):
        aggregate_expressions[3].eval((542,), aggregate_model.experts)
    # CRAM + VennPacking + ANF sanity
    r = CRAMEngine.solve_coloring([(0, 1), (1, 2), (2, 3), (3, 0)], 4, 2)
    assert r["verified"]
    assert len(pack_venn_16_flat()) == 13
    tt = anf_transform([0, 1, 1, 0])
    assert anf_to_expr_v8(tt, 2).eval((1, 0)) == 1
    print(
        f"V8.4 SELF_TEST PASS: {check['committed']} base experts acc 1.0, growth -> {rep['experts']} experts, "
        f"{len(model.theorems)} theorems, prime + tau/sigma/phi/factorial induction PASS, "
        f"estimated_packed_bytes={model.total_neural_bytes()}"
    )


# ==============================================================================
# PART IV — v9.0 layer
# ==============================================================================

V9_VERSION = "9.0.0"

# ==============================================================================
# 1. Core Arithmetic / Bitwise / Logic Operations & Prefix Codes
# ==============================================================================

TOKEN_BITS = 6
VAR_NAMES = ("x", "y", "z", "u", "v", "w")
DYADIC_LEVELS = np.array([(c + 1) / 128.0 for c in range(255)], dtype=np.float64)
_PRIME_MOD = 2_147_483_647  # Mersenne prime 2^31 - 1 for fast exact linear filter


# ======================================================================
# V9 canonical core (executes after legacy; wins shared names)
# ======================================================================
# HAGI V9 canonical core (growing processor, exact synthesis,
# certified Venn head, guarded steps) — split from the historical
# monolith; legacy V5–V8 generations live in _legacy.
from typing import Any, Callable, Optional, Sequence  # noqa: F401
from .gpu import GpuGpmMixin  # GPU dispatch for the neural core


# imports and legacy names; V9 re-definitions below override them.
def _bit_not8(x: int) -> int:
    return (~int(x)) & 0xFF


def _bit_nand8(x: int, y: int) -> int:
    return (~(int(x) & int(y))) & 0xFF


def _bit_nor8(x: int, y: int) -> int:
    return (~(int(x) | int(y))) & 0xFF


def _bit_xnor8(x: int, y: int) -> int:
    return (~(int(x) ^ int(y))) & 0xFF


def _sign(x: int) -> int:
    return (int(x) > 0) - (int(x) < 0)


def _pow_int(x: int, y: int) -> int:
    y = int(y)
    if y < 0 or y > 8:
        raise ValueError("exponent outside certified range")
    return int(x) ** y


OPS: dict[str, Callable[..., int]] = {
    "add": lambda x, y: int(x) + int(y),
    "sub": lambda x, y: int(x) - int(y),
    "mul": lambda x, y: int(x) * int(y),
    "floordiv": lambda x, y: int(x) // int(y),
    "mod": lambda x, y: int(x) % int(y),
    "gcd": lambda x, y: math.gcd(int(x), int(y)),
    "lcm": lambda x, y: math.lcm(int(x), int(y)),
    "min": lambda x, y: min(int(x), int(y)),
    "max": lambda x, y: max(int(x), int(y)),
    "eq": lambda x, y: int(x == y),
    "ne": lambda x, y: int(x != y),
    "lt": lambda x, y: int(x < y),
    "le": lambda x, y: int(x <= y),
    "gt": lambda x, y: int(x > y),
    "ge": lambda x, y: int(x >= y),
    "bit_and": lambda x, y: int(x) & int(y),
    "bit_or": lambda x, y: int(x) | int(y),
    "bit_xor": lambda x, y: int(x) ^ int(y),
    "bit_nand8": _bit_nand8,
    "bit_nor8": _bit_nor8,
    "bit_xnor8": _bit_xnor8,
    "pow_int": _pow_int,
    "neg": lambda x: -int(x),
    "abs": lambda x: abs(int(x)),
    "sign": _sign,
    "bit_not8": _bit_not8,
    "square": lambda x: int(x) * int(x),
    "cube": lambda x: int(x) * int(x) * int(x),
}

UNARY_OPS = ("neg", "abs", "sign", "bit_not8", "clamp3", "square", "cube")
BINARY_OPS = (
    "add",
    "sub",
    "mul",
    "floordiv",
    "mod",
    "gcd",
    "lcm",
    "min",
    "max",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "bit_and",
    "bit_or",
    "bit_xor",
    "bit_nand8",
    "bit_nor8",
    "bit_xnor8",
    "pow_int",
)
COMMUTATIVE_OPS = frozenset(
    {
        "add",
        "mul",
        "gcd",
        "lcm",
        "min",
        "max",
        "eq",
        "ne",
        "bit_and",
        "bit_or",
        "bit_xor",
        "bit_nand8",
        "bit_nor8",
        "bit_xnor8",
    }
)
CONSTS = (
    0,
    1,
    -1,
    2,
    -2,
    26,
    -26,
    256,
    -256,
)  # +alphabet/byte moduli (language, full UTF-8 pool)


def gamma_bits(n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError("gamma_bits expects positive integer")
    return 2 * (n.bit_length() - 1) + 1


def zigzag(n: int) -> int:
    n = int(n)
    return 2 * n if n >= 0 else -2 * n - 1
