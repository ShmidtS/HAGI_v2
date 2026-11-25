"""Безопасный язык выражений: битовые операции, Expr, Program.

Идея: вместо исполнения произвольного текста программа хранится как дерево
Expr (операция-имя, дочерние выражения, значение). Вычисление — обход дерева
с обычными целыми Python, без exec/eval.

Примитивы над 8-битными числами: NOT x = (~x) & 255, NAND = ~(x & y) & 255,
NOR = ~(x | y) & 255, XNOR = ~(x ^ y) & 255. Знак: sign(x) = [x > 0] − [x < 0].
Степень x^y допускается только при 0 ≤ y ≤ 8.

Операция poly — многочлен одной переменной по схеме Горнера:
p(x) = c_d·x^d + … + c_0, acc ← acc·x + c_i, идя от c_d к c_0.
Операция poly_nd — многочлен нескольких переменных:
p(x) = Σ_t a_t · Π_k x_k^(e_{t,k}).

Стоимость узла: cost(лист) = 1, cost(узел) = 1 + Σ cost(дочерних).
Program — либо выражение, либо таблица истинности двух булевых аргументов:
результат = (mask >> (2·x + y)) & 1, её стоимость 4.

Экспортирует: _truth, битовые и прочие примитивы, словарь OPS, Expr, Program.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов
import math  # модули стандартной библиотеки: аргументы, синтаксическое дерево, копирование, куча, комбинаторы, JSON, математика, ОС, случайность
from dataclasses import dataclass  # классы данных
from typing import Any, Callable, Optional, Sequence  # аннотации типов

# ----------------------------- Безопасный язык выражений -----------------------------


def _truth(v: Any) -> bool:
    """Истинность значения как у целого числа: bool(int(v))."""
    return bool(int(v))  # привести к целому и взять истинность: True, если int(v) ≠ 0


def _bit_not8(x: int) -> int:
    """Побитовое отрицание в 8 битах: (~x) & 255."""
    return (~int(x)) & 0xFF  # инвертировать все биты и оставить младшие 8: (~x) mod 256


def _bit_nand8(x: int, y: int) -> int:
    """Побитовое И-НЕ в 8 битах: ~(x & y) & 255."""
    return (~(int(x) & int(y))) & 0xFF  # И, затем инверсия и обрезка до 8 бит


def _bit_nor8(x: int, y: int) -> int:
    """Побитовое ИЛИ-НЕ в 8 битах: ~(x | y) & 255."""
    return (~(int(x) | int(y))) & 0xFF  # ИЛИ, затем инверсия и обрезка до 8 бит


def _bit_xnor8(x: int, y: int) -> int:
    """Побитовое исключающее-ИЛИ-НЕ в 8 битах: ~(x ^ y) & 255."""
    return (~(int(x) ^ int(y))) & 0xFF  # исключающее ИЛИ, затем инверсия и обрезка до 8 бит


def _sign(x: int) -> int:
    """Знак числа: 1, если x > 0; −1, если x < 0; 0 при x = 0."""
    return (int(x) > 0) - (int(x) < 0)  # знак: sign(x) = [x > 0] − [x < 0]


def _pow_int(x: int, y: int) -> int:
    """Целая степень x^y; показатель y обязан лежать в отрезке 0..8.

        За отрезком возбуждается ValueError: так степень не даёт
        ни отрицательных значений, ни астрономического роста."""
    y = int(y)  # показатель степени — целое
    if y < 0 or y > 8:  # показатель вне отрезка 0..8
        raise ValueError("exponent outside certified range")  # ошибка: показатель вне допустим отрезка
    return int(x) ** y  # степень x^y


OPS: dict[str, Callable[..., Any]] = {  # таблица операций: имя -> функция от целых аргументов
    "add": lambda x, y: int(x) + int(y),  # сложение x + y
    "sub": lambda x, y: int(x) - int(y),  # вычитание x − y
    "mul": lambda x, y: int(x) * int(y),  # умножение x·y
    "floordiv": lambda x, y: int(x) // int(y),  # деление нацело ⌊x / y⌋
    "mod": lambda x, y: int(x) % int(y),  # остаток x mod y
    "gcd": lambda x, y: math.gcd(int(x), int(y)),  # наибольший общий делитель gcd(x, y)
    "lcm": lambda x, y: math.lcm(int(x), int(y)),  # наименьшее общее кратное lcm(x, y)
    "min": lambda x, y: min(int(x), int(y)),  # минимум min(x, y)
    "max": lambda x, y: max(int(x), int(y)),  # максимум max(x, y)
    "eq": lambda x, y: int(x == y),  # результат сравнения: единица, если x = y, иначе ноль
    "ne": lambda x, y: int(x != y),  # результат сравнения: единица, если x ≠ y, иначе ноль
    "lt": lambda x, y: int(x < y),  # результат сравнения: единица, если x < y, иначе ноль
    "le": lambda x, y: int(x <= y),  # результат сравнения: единица, если x ≤ y, иначе ноль
    "gt": lambda x, y: int(x > y),  # результат сравнения: единица, если x > y, иначе ноль
    "ge": lambda x, y: int(x >= y),  # результат сравнения: единица, если x ≥ y, иначе ноль
    "bit_and": lambda x, y: int(x) & int(y),  # побитовое И
    "bit_or": lambda x, y: int(x) | int(y),  # побитовое ИЛИ
    "bit_xor": lambda x, y: int(x) ^ int(y),  # побитовое исключающее ИЛИ
    "bit_nand8": _bit_nand8,  # И-НЕ в 8 битах
    "bit_nor8": _bit_nor8,  # ИЛИ-НЕ в 8 битах
    "bit_xnor8": _bit_xnor8,  # исключающее-ИЛИ-НЕ в 8 битах
    "pow_int": _pow_int,  # целая степень с показателем 0..8
    "neg": lambda x: -int(x),  # смена знака −x
    "abs": lambda x: abs(int(x)),  # модуль |x|
    "sign": _sign,  # знак числа
    "bit_not8": _bit_not8,  # побитовое отрицание в 8 битах
    "square": lambda x: int(x) * int(x),  # квадрат x²
    "cube": lambda x: int(x) * int(x) * int(x),  # куб x³
}


@dataclass(frozen=True)  # неизменяемый класс данных: хэшируемый, годится как ключ словаря
class Expr:
    """Узел дерева выражения.

        op — имя операции; args — дочерние выражения (пусто для листьев);
        value — для "var" номер входа i (значение xs[i]), для "const" —
        сама константа, для "poly" — кортеж коэффициентов (c_0..c_d),
        для "poly_nd" — кортеж пар (коэффициент, показатели степеней)."""
    op: str  # имя операции
    args: tuple["Expr", ...] = ()  # дочерние выражения, по умолчанию пусто
    value: Optional[Any] = None  # значение узла (номер входа, константа, коэффициенты)

    @staticmethod  # метод класса без доступа к экземпляру
    def var(i: int) -> "Expr":  # конструктор входа с номером i
        return Expr("var", (), int(i))  # лист "var" со значением i: при вычислении вернуть xs[i]

    @staticmethod  # метод класса без доступа к экземпляру
    def const(v: int) -> "Expr":  # конструктор константы v
        return Expr("const", (), int(v))  # лист "const" со значением v

    def eval(self, xs: Sequence[int]) -> int:
        """Вычислить выражение на входах xs и вернуть целое число.

            "var" → xs[value]; "const" → value. Сначала рекурсивно
            вычисляются все дочерние узлы vals, затем применяется операция:
            poly по Горнеру acc ← acc·vals[0] + c_i (от c_d к c_0);
            poly_nd = Σ a_t·Π x_k^e; absdiff = |a − b|; fma = a·b + c;
            clamp = min(max(a, lo), hi); median3 — средний из трёх;
            clamp3 = min(max(a, −3), 3); mul3 = a·b·c; прочие — таблица OPS."""
        if self.op == "var":  # узел — вход
            assert self.value is not None  # у входа обязан быть номер
            return int(xs[self.value])  # значение входа xs[value]
        if self.op == "const":  # узел — константа
            assert self.value is not None  # у константы обязано быть значение
            return int(self.value)  # сама константа
        vals = [a.eval(xs) for a in self.args]  # vals_j = вычисление j-го дочернего узла на xs
        if self.op == "poly":  # многочлен одной переменной
            coeffs = self.value or ()  # коэффициенты (c_0..c_d); если value нет — пусто
            acc = 0  # накопитель схемы Горнера
            for ci in reversed(list(coeffs)):  # от старшего c_d к младшему c_0
                acc = acc * vals[0] + int(ci)  # шаг Горнера: acc ← acc·x + c_i
            return int(acc)  # p(x) после d + 1 шагов
        if self.op == "poly_nd":  # многочлен нескольких переменных
            total = 0  # сумма членов
            for coefficient, powers in self.value or ():  # по членам (a_t, e_t)
                term = int(coefficient)  # член начинается с коэффициента a_t
                for x, exponent in zip(xs, powers):  # по переменным x_k и показателям e_{t,k}
                    term *= int(x) ** int(exponent)  # умножить член: term ← term·x_k^e_{t,k}
                total += term  # прибавить член: total ← total + term
            return int(total)  # сумма членов: Σ_t a_t·Π_k x_k^e_{t,k}
        if self.op == "absdiff":  # модуль разности
            return abs(vals[0] - vals[1])  # модуль разности: |a − b|
        if self.op == "fma":  # умножение со сложением
            return vals[0] * vals[1] + vals[2]  # умножение со сложением: a·b + c
        if self.op == "clamp":  # удержание в отрезке
            return min(max(vals[0], vals[1]), vals[2])  # удержать в отрезке: min(max(a, lo), hi)
        if self.op == "median3":  # медиана трёх
            return sorted(vals)[1]  # средний элемент отсортированного списка
        if self.op == "clamp3":  # удержание в отрезке −3..3
            return min(max(vals[0], -3), 3)  # удержать в отрезке: min(max(a, −3), 3)
        if self.op == "mul3":  # произведение трёх
            return vals[0] * vals[1] * vals[2]  # произведение трёх: a·b·c
        fn = OPS.get(self.op)  # функция операции из таблицы или None
        if fn is None:  # такой операции в таблице нет
            raise ValueError(f"unsupported expression opcode: {self.op}")  # ошибка: неизвестная операция
        return int(fn(*vals))  # применить операцию к вычисленным аргументам

    @property  # доступ как к атрибуту без скобок
    def cost(self) -> int:
        """Стоимость выражения: cost(лист) = 1, cost(узел) = 1 + Σ cost(дочерних).

            Класс заморожен, поэтому кэш пишется в обход запрета через
            object.__setattr__; хранится в словаре __dict__ под "_cost_memo"."""
        c = self.__dict__.get("_cost_memo")  # взять кэшированную стоимость или None
        if c is None:  # стоимости ещё нет
            if self.op in ("var", "const"):  # лист (вход или константа)
                c = 1  # стоимость листа 1
            else:  # внутренний узел
                c = 1 + sum(a.cost for a in self.args)  # 1 + Σ cost(дочерних)
            object.__setattr__(self, "_cost_memo", c)  # сохранить стоимость в кэше
        return c  # стоимость выражения

    def pretty(self) -> str:
        """Человекочитаемая запись выражения строкой.

            Входы именуются x, y, z; poly печатается как сумма членов
            c_i·x^i от старшего к младшему; poly_nd — как сумма одночленов
            Π x_k^e, упорядоченных по убыванию полной степени."""
        if self.op == "var":  # вход
            return ("x", "y", "z")[self.value or 0]  # имя входа по номеру: 0 → x, 1 → y, 2 → z
        if self.op == "const":  # константа
            return str(self.value)  # запись числа
        if self.op == "poly":  # многочлен одной переменной
            coeffs = tuple(self.value or ())  # коэффициенты (c_0..c_d)
            terms = []  # список кусков текста
            for i in range(len(coeffs) - 1, -1, -1):  # степени i от d до 0
                c = int(coeffs[i])  # коэффициент c_i
                if c == 0:  # нулевой коэффициент не печатается
                    continue  # к следующей степени
                mag = abs(c)  # модуль коэффициента: |c_i|
                if i == 0:  # свободный член
                    term = str(mag)  # просто число
                else:  # степень i ≥ 1
                    body = "x" if i == 1 else f"x**{i}"  # "x" или "x**i"
                    term = body if mag == 1 else f"{mag}*{body}"  # без множителя 1, иначе "|c_i|·x^i"
                if not terms:  # первый член
                    terms.append(("-" if c < 0 else "") + term)  # минус перед первым членом, если c_i < 0
                else:  # последующий член
                    terms.append((" - " if c < 0 else " + ") + term)  # разделитель " − " или " + "
            return "poly(" + ("".join(terms) if terms else "0") + ")"  # "poly(" + сумма членов или "0" + ")"
        if self.op == "poly_nd":  # многочлен нескольких переменных
            symbols = ("x", "y", "z", "u", "v", "w")  # имена переменных
            terms = []  # список кусков текста
            for coefficient, powers in sorted(  # члены, отсортированные по ключу ниже
                (self.value or ()),  # кортеж пар (a_t, e_t)
                key=lambda item: (sum(item[1]), item[1]),  # ключ: (полная степень Σ e, вектор показателей)
                reverse=True,  # по убыванию ключа
            ):  # конец sorted
                c = int(coefficient)  # коэффициент a_t
                if c == 0:  # нулевой коэффициент
                    continue  # пропустить член
                factors = []  # множители-переменные члена
                for index, exponent in enumerate(powers):  # по номеру k и показателю e_k
                    exponent = int(exponent)  # показатель — целое
                    if exponent == 0:  # переменная в степени 0
                        continue  # не входит в член
                    name = symbols[index] if index < len(symbols) else f"x{index}"  # имя переменной x, y, … или x<номер>
                    factors.append(name if exponent == 1 else f"{name}**{exponent}")  # "x" или "x**e"
                body = "*".join(factors) if factors else str(abs(c))  # произведение имён или |a_t|, если переменных нет
                if factors and abs(c) != 1:  # есть переменные и коэффициент не ±1
                    body = f"{abs(c)}*{body}"  # добавить множитель |a_t|·
                if not terms:  # первый член
                    terms.append(("-" if c < 0 else "") + body)  # минус, если a_t < 0
                else:  # последующий член
                    terms.append((" - " if c < 0 else " + ") + body)  # разделитель " − " или " + "
            return "poly_nd(" + ("".join(terms) if terms else "0") + ")"  # "poly_nd(" + сумма или "0" + ")"
        if self.op == "clamp3":  # удержание в −3..3
            return f"min(max({self.args[0].pretty()}, -3), 3)"  # min(max(аргумент, −3), 3)
        if self.op == "absdiff":  # модуль разности
            return f"abs({self.args[0].pretty()} - {self.args[1].pretty()})"  # abs(первый − второй)
        if self.op == "fma":  # умножение со сложением
            return f"({self.args[0].pretty()} * {self.args[1].pretty()} + {self.args[2].pretty()})"  # (первый * второй + третий)
        if self.op == "clamp":  # удержание в отрезке
            return f"min(max({self.args[0].pretty()}, {self.args[1].pretty()}), {self.args[2].pretty()})"  # min(max(значение, низ), верх)
        if self.op == "median3":  # медиана трёх
            return f"median({', '.join(a.pretty() for a in self.args)})"  # median(первый, второй, третий)
        if self.op == "mul3":  # произведение трёх
            return " * ".join(a.pretty() for a in self.args)  # аргументы через " * "
        if len(self.args) == 1:  # один аргумент
            return f"{self.op}({self.args[0].pretty()})"  # "оп(аргумент)"
        if len(self.args) == 2:  # два аргумента
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"  # "(первый оп второй)"
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"  # "оп(аргументы через запятую)"

    def to_dict(self) -> dict[str, Any]:
        """Представить узел словарём для записи в JSON: операция, значение, дети."""
        return {  # собрать словарь узла
            "op": self.op,  # имя операции
            "value": self.value,  # значение узла
            "args": [a.to_dict() for a in self.args],  # словари дочерних узлов
        }

    @staticmethod  # метод класса без доступа к экземпляру
    def from_dict(d: dict[str, Any]) -> "Expr":
        """Восстановить выражение из словаря, созданного to_dict.

            JSON превращает кортежи в списки, поэтому коэффициенты poly и
            poly_nd приводятся обратно к кортежам: иначе выражения перестали
            бы быть хэшируемыми и устойчиво сравниваться после загрузки."""
        op = str(d["op"])  # имя операции строкой
        value = d.get("value")  # значение узла или None
        # JSON превращает кортежи в списки; восстанавливаем канонический вид
        # значения, чтобы выражения остались хэшируемыми и равенство не менялось.
        if op == "poly" and value is not None:  # многочлен одной переменной
            value = tuple(int(v) for v in value)  # (c_0, …, c_d) — кортеж целых
        elif op == "poly_nd" and value is not None:  # многочлен нескольких переменных
            value = tuple(  # кортеж пар
                (int(term[0]), tuple(int(v) for v in term[1])) for term in value  # каждая пара: (a_t, (e_{t,0}, …))
            )  # конец кортежа
        return Expr(op, tuple(Expr.from_dict(a) for a in d.get("args", [])), value)  # узел с рекурсивно восстановленными детьми


@dataclass(frozen=True)  # неизменяемый класс данных
class Program:
    """Программа: выражение Expr либо таблица истинности venn_mask.

        arity — число входов. Таблица истинности применима к двум
        булевым входам: результат = (mask >> (2·x + y)) & 1."""
    arity: int  # число входов
    expr: Optional[Expr] = None  # выражение или None
    venn_mask: Optional[int] = None  # 4-битная маска истинности или None

    @property  # доступ как к атрибуту без скобок
    def cost(self) -> int:
        """Стоимость программы: 4 для таблицы истинности, cost(expr) для выражения."""
        return (  # вернуть
            4  # 4, если задана маска
            if self.venn_mask is not None  # таблица истинности задана
            else (self.expr.cost if self.expr else 10**9)  # cost(expr); 10^9, если выражения нет
        )

    def eval(self, xs: Sequence[int]) -> int:
        """Вычислить программу на входах xs.

            Число входов обязано равняться arity. Для маски входы — два
            булевых значения, результат (mask >> (2·x + y)) & 1; иначе
            результат выражения expr.eval(xs)."""
        if len(xs) != self.arity:  # число входов не совпадает
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")  # ошибка о неверном числе входов
        if self.venn_mask is not None:  # программа — таблица истинности
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):  # не два входа или не булевы значения
                raise ValueError("BoolVenn expert accepts two Boolean inputs")  # ошибка: нужны два булевых входа
            idx = (int(xs[0]) << 1) | int(xs[1])  # номер строки таблицы idx = 2·x + y
            return (int(self.venn_mask) >> idx) & 1  # бит маски номер idx
        if self.expr is None:  # выражения нет
            raise RuntimeError("program has no expression")  # ошибка: нечего вычислять
        return self.expr.eval(xs)  # результат выражения

    def pretty(self) -> str:
        """Человекочитаемая запись программы: маска в двоичном виде или запись выражения."""
        if self.venn_mask is not None:  # программа — таблица истинности
            return f"BoolVenn(mask={self.venn_mask:04b})"  # "BoolVenn(mask=маска в 4 битах)"
        assert self.expr is not None  # раз маски нет, выражение обязано быть
        return self.expr.pretty()  # запись выражения

    def to_dict(self) -> dict[str, Any]:
        """Представить программу словарём для записи в JSON: число входов, маска, выражение."""
        return {  # собрать словарь
            "arity": self.arity,  # число входов
            "venn_mask": self.venn_mask,  # маска или None
            "expr": self.expr.to_dict() if self.expr else None,  # словарь выражения или None
        }

    @staticmethod  # метод класса без доступа к экземпляру
    def from_dict(d: dict[str, Any]) -> "Program":
        """Восстановить программу из словаря to_dict: число входов, выражение, маска."""
        return Program(  # создать программу
            int(d["arity"]),  # число входов
            Expr.from_dict(d["expr"]) if d.get("expr") else None,  # выражение, если оно было сохранено
            d.get("venn_mask"),  # маска или None
        )

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'OPS', 'Expr', 'Program',  # имя интерфейса
]  # конец перечня
