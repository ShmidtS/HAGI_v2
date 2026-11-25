"""Синтез кандидатов в порядке возрастания длины кода («лучшие первыми»).

Кандидаты — выражения ограниченной грамматики: унарные и бинарные операции,
константы, тернарные примитивы и подстановки в уже принятые программы. Ключ
очереди с приоритетом — длина описания L(expr) в битах; при равенстве ключей
порядок задаётся счётчиком вставки. Из кучи извлекается выражение с наименьшим
L, значит кандидаты выдаются в истинном порядке длины префиксного кода.

Пробники (probe_x, probe_y) — дешёвый фильтр: выдаются только выражения,
значения которых на probe_x равны probe_y. Значение на пробниках: v_i = f(x_i).

Подстановка: _substitute(e, m) заменяет переменную с индексом k на m[k];
стоимость после подстановки C(e) = costs[k] для переменной, 1 для константы,
1 + Σ C(аргумента) иначе.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов
import heapq, itertools  # модули стандартной библиотеки: аргументы, синтаксическое дерево, копирование, куча, итераторы, JSON, математика, ОС, случайность
import time  # статистика, параметры интерпретатора, время
from typing import Optional, Sequence  # аннотации типов
from hagi.core.language.expr_dsl import Expr, Program  # имена из «expr_dsl» (модуль, где они определены)
from hagi.core.synthesis.prefilter import OPS  # имена из модуля «prefilter» (владелец определения)
from hagi.core.tasks.stored_tasks import (  # имена из модуля задач
    TaskSpec  # описание задачи
)  # конец импорта
from hagi.core.processor.experts import (  # имена из модуля экспертов
    Expert  # эксперт: программа с арностью
)  # конец импорта
from hagi.core.synthesis.prefilter import DEFAULT_NODE_BUDGET, MAX_REUSE_EXPANSIONS, OPS  # имена из «prefilter» (модуль, где они определены)
from hagi.core.processor.experts import Expert  # имена из «experts» (модуль, где они определены)
from hagi.core.language.expr_dsl import Expr, Program  # имена из «expr_dsl» (модуль, где они определены)
from typing import Iterator, Optional, Sequence  # имена из «stdlib:typing» (модуль, где они определены)
from hagi.core.tasks.stored_tasks import TaskSpec  # имена из «stored_tasks» (модуль, где они определены)
from hagi.core.language.code_length import (  # функции длины кода
    _expr_description_bits,  # длина описания выражения L(expr)
    _program_description_bits  # длина описания программы
)  # конец импорта

# ---- синтез кандидатов «лучшими первыми» ----

UNARY_OPS = ("neg", "abs", "sign", "bit_not8", "clamp3")  # унарные операции: −x, |x|, знак, не в 8 битах, усечение в [−3, 3]
BINARY_OPS = (  # бинарные операции
    "add",  # сумма
    "sub",  # разность
    "mul",  # произведение
    "floordiv",  # деление нацело
    "mod",  # остаток
    "gcd",  # НОД
    "lcm",  # НОК
    "min",  # минимум
    "max",  # максимум
    "eq",  # равно
    "ne",  # не равно
    "lt",  # меньше
    "le",  # не больше
    "gt",  # больше
    "ge",  # не меньше
    "bit_and",  # побитовое И
    "bit_or",  # побитовое ИЛИ
    "bit_xor",  # исключающее ИЛИ
    "bit_nand8",  # И-НЕ в 8 битах
    "bit_nor8",  # ИЛИ-НЕ в 8 битах
    "bit_xnor8",  # равнозначность в 8 битах
    "pow_int",  # целая степень
)
CONSTS = (0, 1, -1)  # константы-листья


TERNARY_PRIMS = ("fma", "clamp", "median3", "mul3")  # тернарные примитивы


def iter_candidates_mdl(
    arity: int,  # число входных переменных
    node_budget: int = DEFAULT_NODE_BUDGET,  # сколько кандидатов выдать максимум
    probe_x: Sequence[Sequence[int]] = (),  # строки входов пробников
    probe_y: Sequence[int] = (),  # ожидаемые выходы пробников
    extra_binops: Sequence[tuple[str, Program]] = (),  # дополнительные бинарные программы (имя, программа)
    pop_budget: int = 50_000,  # максимум извлечений из кучи
    deadline: Optional[float] = None,  # предельный момент времени (time.monotonic)
) -> Iterator[Expr]:  # возврат: итератор выражений
    """Выдавать согласованные с пробниками выражения в порядке возрастания L(expr).

        Перебирается только реализованная ограниченная грамматика: унарные
        операции, применение бинарной операции к двум копиям узла, бинарная
        операция с листом, тернарные примитивы от переменных и подстановки в
        принятые программы. Это не полное перечисление всех деревьев языка;
        перебор может быть обрезан бюджетами узлов, извлечений и временем.
        Пробники — только дешёвый фильтр согласованности: два разных дерева
        не отбрасываются лишь за совпадение на пробниках."""
    probe = tuple(tuple(int(v) for v in xs) for xs in probe_x)  # пробники как кортеж целых строк
    n = len(probe)  # число пробников
    pop_budget = int(pop_budget)  # бюджет извлечений как целое

    def apply_bin(  # применить бинарную операцию op к значениям a и b построчно
        op: str, a: tuple[int, ...], b: tuple[int, ...]  # операция и два вектора значений
    ) -> Optional[tuple[int, ...]]:  # возврат: кортеж результатов или None
        fn = OPS.get(op)  # функция операции или None
        if fn is None:  # операции нет
            return None  # не вычисляется
        try:  # вычислять с перехватом ошибок
            return tuple(int(fn(a[i], b[i])) for i in range(n))  # результаты: r_i = f(a_i, b_i)
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # деление на ноль, переполнение, выход за границы
            return None  # значение не определено

    def apply_un(op: str, a: tuple[int, ...]) -> Optional[tuple[int, ...]]:  # унарная операция op над значениями a
        fn = (lambda v: min(max(v, -3), 3)) if op == "clamp3" else OPS.get(op)  # clamp3: v → min(max(v, −3), 3), иначе функция операции
        if fn is None:  # операции нет
            return None  # не вычисляется
        try:  # вычислять с перехватом ошибок
            return tuple(int(fn(a[i])) for i in range(n))  # результаты: r_i = f(a_i)
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
            return None  # значение не определено

    def apply_tern(  # тернарный примитив op над a, b, c
        op: str, a: tuple[int, ...], b: tuple[int, ...], c: tuple[int, ...]  # операция и три вектора значений
    ) -> Optional[tuple[int, ...]]:  # возврат: кортеж результатов или None
        try:  # вычислять с перехватом ошибок
            if op == "fma":  # умножение-сложение
                return tuple(a[i] * b[i] + c[i] for i in range(n))  # результаты: r_i = a_i·b_i + c_i
            if op == "mul3":  # произведение трёх
                return tuple(a[i] * b[i] * c[i] for i in range(n))  # результаты: r_i = a_i·b_i·c_i
            if op == "clamp":  # усечение: a не выше c и не ниже b
                return tuple(min(max(a[i], b[i]), c[i]) for i in range(n))  # результаты: r_i = min(max(a_i, b_i), c_i)
            return tuple(sorted((a[i], b[i], c[i]))[1] for i in range(n))  # иначе медиана: средний из отсортированных
        except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
            return None  # значение не определено

    counter = itertools.count()  # счётчик вставок 0, 1, 2, … (тай-брейк в куче)
    leaves = [Expr.var(i) for i in range(arity)] + [Expr.const(c) for c in CONSTS]  # листья: переменные 0..arity−1 и константы
    leaf_vals: list[tuple[Expr, Optional[tuple[int, ...]]]] = []  # значения листьев на пробниках
    for leaf in leaves:  # по листьям
        if leaf.op == "var":  # переменная
            lv = tuple(row[int(leaf.value or 0)] for row in probe) if n else None  # lv_j = x_j[индекс]; без пробников None
        else:  # константа
            lv = tuple(int(leaf.value) for _ in probe) if n else None  # lv_j = константа; без пробников None
        leaf_vals.append((leaf, lv))  # добавить пару (лист, значения)

    heap: list[tuple[int, int, Expr, tuple[int, ...]]] = []  # куча: (L, счётчик, выражение, значения)
    expr_seen: set[Expr] = set()  # уже добавленные выражения
    target: Optional[tuple[int, ...]] = (  # целевой вектор выходов
        tuple(int(v) for v in probe_y) if probe_y else None  # target_j = probe_y[j]; без пробников None
    )
    max_nodes = 24  # наибольшая стоимость выражения (число узлов)

    def push(expr: Expr, vals: Optional[tuple[int, ...]]) -> None:  # добавить выражение в кучу
        if vals is None or expr.cost > max_nodes or expr in expr_seen:  # нет значений, слишком дорого или уже есть
            return  # не добавлять
        expr_seen.add(expr)  # запомнить выражение
        heapq.heappush(heap, (_expr_description_bits(expr), next(counter), expr, vals))  # в кучу с ключом (L(expr), номер)

    for leaf, lv in leaf_vals:  # по листьям
        push(leaf, lv)  # добавить лист в кучу

    # Тернарные примитивы добавляются один раз, а не повторно при каждом извлечении.
    if arity == 3:  # только для трёх переменных
        for a, va in leaf_vals[:3]:  # первый аргумент — переменная 0..2
            for b, vb in leaf_vals[:3]:  # второй аргумент
                for c, vc in leaf_vals[:3]:  # третий аргумент
                    for prim in TERNARY_PRIMS:  # по тернарным примитивам
                        push(Expr(prim, (a, b, c)), apply_tern(prim, va, vb, vc))  # добавить prim(a, b, c) с его значениями

    generated = 0  # число выданных кандидатов
    pops = 0  # число извлечений из кучи
    extra_ops = tuple(  # дополнительные программы, отсортированные по L
        sorted(  # отсортировать
            ((nm, prog) for nm, prog in extra_binops),  # пары (имя, программа)
            key=lambda pair: (_program_description_bits(pair[1]), pair[0]),  # сортировка по (L(программы), имя)
        )[:MAX_REUSE_EXPANSIONS]  # оставить первые MAX_REUSE_EXPANSIONS
    )

    def expired() -> bool:  # истёк ли крайний срок
        return deadline is not None and time.monotonic() >= deadline  # True, если t ≥ deadline

    while heap and generated < node_budget and pops < pop_budget:  # пока куча не пуста и бюджеты не исчерпаны
        if deadline is not None and time.monotonic() >= deadline:  # время вышло
            break  # закончить перебор
        _, _, expr, vals = heapq.heappop(heap)  # извлечь выражение с наименьшим L и его значения
        pops += 1  # извлечений: +1
        if target is None or vals == target:  # цели нет или значения совпали с probe_y
            generated += 1  # выданных: +1
            yield expr  # выдать кандидата

        base = expr.cost  # стоимость текущего выражения в узлах
        for op in UNARY_OPS:  # расширение унарными операциями
            if expired():  # время вышло
                return  # закончить
            push(Expr(op, (expr,)), apply_un(op, vals))  # добавить op(expr)

        for op in BINARY_OPS:  # расширение бинарными операциями
            if expired():  # время вышло
                return  # закончить
            push(Expr(op, (expr, expr)), apply_bin(op, vals, vals))  # добавить op(expr, expr)
            for leaf, lv in leaf_vals:  # по листьям
                if expired():  # время вышло
                    return  # закончить
                push(Expr(op, (expr, leaf)), apply_bin(op, vals, lv))  # добавить op(expr, лист)
                push(Expr(op, (leaf, expr)), apply_bin(op, lv, vals))  # добавить op(лист, expr)

        for _name, prog in extra_ops:  # по дополнительным программам
            if expired():  # время вышло
                return  # закончить
            if prog.arity != 2 or prog.expr is None:  # нужна бинарная арифметическая программа
                continue  # пропустить
            exp = _program_to_substitutable_expr(prog)  # её выражение для подстановки
            combos = [(expr, vals, expr, vals)]  # варианты подстановки: оба входа — текущий узел
            for leaf, lv in leaf_vals:  # по листьям
                combos.append((expr, vals, leaf, lv))  # (узел, лист)
                combos.append((leaf, lv, expr, vals))  # (лист, узел)
            for a, _av, b, _bv in combos:  # по вариантам (a, b)
                if expired():  # время вышло
                    return  # закончить
                if _subst_cost(exp, {0: a.cost, 1: b.cost}) > max_nodes:  # стоимость после подстановки больше 24
                    continue  # пропустить
                child = _substitute(exp, {0: a, 1: b})  # подставить a и b вместо переменных 0 и 1
                # Повторно используемая программа, не зависящая от входов, может свернуться в более
                # дешёвый лист; вставлять более дешёвый узел после извлечения дорогого нельзя — это
                # нарушило бы порядок «лучшие первыми».
                if _expr_description_bits(child) <= _expr_description_bits(expr):  # L(ребёнка) ≤ L(текущего)
                    continue  # пропустить
                try:  # вычислить значения ребёнка на пробниках
                    out = tuple(int(child.eval(row)) for row in probe)  # out_j = ребёнок(x_j)
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                    out = None  # значения не определены
                push(child, out)  # добавить ребёнка в кучу


def _poly_nd_to_ast(expr: Expr) -> Expr:  # раскрыть компактный многочлен многих переменных
    """Развернуть компактную запись многочлена многих переменных в обычные операции языка.

        expr.value — список пар (коэффициент, степени). Слагаемое равно
        коэффициент · Π_k x_k^p_k, где p_k — степень переменной k (при p_k = 0
        множитель опускается, при p_k = 1 — само x_k, иначе pow_int(x_k, p_k)).
        Слагаемые соединяются операцией add."""
    terms = []  # список слагаемых
    for coefficient, powers in expr.value or ():  # по парам (коэффициент, степени)
        factors: list[Expr] = [Expr.const(int(coefficient))]  # множители слагаемого, первый — коэффициент
        for index, power in enumerate(powers):  # по переменным и их степеням
            power = int(power)  # степень как целое
            if power == 0:  # степень 0 — множитель x^0 = 1
                continue  # пропустить
            variable = Expr.var(index)  # переменная x_index
            factor = (  # множитель x^power
                variable  # при power = 1 — сама переменная
                if power == 1  # если степень 1
                else Expr("pow_int", (variable, Expr.const(power)))  # иначе pow_int(x, power)
            )
            factors.append(factor)  # добавить множитель
        term = factors[0]  # произведение начинается с первого множителя
        for factor in factors[1:]:  # по остальным множителям
            term = Expr("mul", (term, factor))  # term = term · множитель
        terms.append(term)  # добавить слагаемое
    if not terms:  # слагаемых нет
        return Expr.const(0)  # многочлен тождественно равен 0
    result = terms[0]  # сумма начинается с первого слагаемого
    for term in terms[1:]:  # по остальным слагаемым
        result = Expr("add", (result, term))  # result = result + слагаемое
    return result  # выражение-сумма


def _program_to_substitutable_expr(program: Program) -> Expr:  # возврат: выражение, годное для подстановки
    """Вернуть выражение программы, в которое можно подставлять аргументы."""
    if program.expr is None:  # программа задана таблицей истинности
        raise ValueError("truth-table programs are not arithmetic ASTs")  # ошибка: не арифметическое дерево
    if program.expr.op == "poly_nd":  # компактный многочлен многих переменных
        return _poly_nd_to_ast(program.expr)  # развернуть в обычные операции
    return program.expr  # иначе вернуть выражение как есть


def _composition_hint(spec: TaskSpec) -> Optional[Program]:
    """Построить программу-подсказку: композиция внешнего и внутреннего экспертов.

        spec: задача, у которой в значениях по умолчанию функции teacher хранятся
        два родительских эксперта. Возврат: программа арности 2 или None, если
        композиция неприменима. Для табличных программ маска вычисляется по
        четырём точкам: mask |= f(x, y) << ((x << 1) | y). Для внешней программы
        арности 2 внутренняя подставляется на место первого (если она арности 1)
        или второго (если арности 2) аргумента. Подсказка не считается верной: она
        лишь избавляет от повторного поиска известного дерева."""
    parents = getattr(spec.teacher, "__defaults__", None)  # родители — значения по умолчанию аргументов teacher
    if not parents or len(parents) < 2:  # родителей нет или меньше двух
        return None  # подсказки нет
    outer, inner = parents[0], parents[1]  # внешний и внутренний эксперты
    if not isinstance(outer, Expert) or not isinstance(inner, Expert):  # оба должны быть экспертами
        return None  # иначе нет подсказки
    if outer.program.arity != 2 or spec.arity != 2:  # нужна внешняя программа арности 2 и задача арности 2
        return None  # иначе нет подсказки
    if outer.program.venn_mask is not None and inner.program.venn_mask is not None:  # обе программы табличные
        mask = 0  # маска истинности, изначально 0
        for x in (0, 1):  # по значениям x
            for y in (0, 1):  # по значениям y
                value = int(spec.teacher(x, y)) & 1  # значение: value = f(x, y) mod 2
                mask |= value << ((x << 1) | y)  # бит номер 2x + y полагается равным value
        return Program(2, venn_mask=mask)  # табличная программа арности 2 с маской mask
    if outer.program.expr is None or inner.program.expr is None:  # хотя бы одна программа табличная
        return None  # арифметической композиции нет
    outer_expr = _program_to_substitutable_expr(outer.program)  # выражение внешней программы
    inner_expr = _program_to_substitutable_expr(inner.program)  # выражение внутренней программы
    if inner.program.arity == 1:  # внутренняя программа с одним входом
        mapping = {0: inner_expr, 1: Expr.var(1)}  # вход 0 ← inner, вход 1 ← x_1
    elif inner.program.arity == 2:  # внутренняя программа с двумя входами
        mapping = {0: Expr.var(0), 1: inner_expr}  # вход 0 ← x_0, вход 1 ← inner
    else:  # другая арность
        return None  # композиция не определена
    return Program(2, _substitute(outer_expr, mapping))  # программа outer с подстановкой mapping


def _substitute(e: Expr, mapping: dict[int, Expr]) -> Expr:  # подстановка mapping в выражение e
    """Вернуть копию e, где переменная с индексом k заменена на mapping[k]."""
    if e.op == "var":  # переменная
        return mapping[int(e.value or 0)]  # заменить на выражение для её индекса
    if e.op == "const":  # константа
        return e  # неизменна
    return Expr(e.op, tuple(_substitute(a, mapping) for a in e.args))  # та же операция над подставленными аргументами


def _subst_cost(e: Expr, costs: dict[int, int]) -> int:
    """Стоимость подстановки в e, когда переменная k стоит costs[k]; само дерево не строится.

        C(переменная k) = costs[k]; C(константа) = 1; C(узел) = 1 + Σ C(аргументов)."""
    if e.op == "var":  # переменная
        return costs[int(e.value or 0)]  # cost переменной
    if e.op == "const":  # константа
        return 1  # один узел
    return 1 + sum(_subst_cost(a, costs) for a in e.args)  # 1 + Σ C(аргументов)


class SynthesisStats:  # счётчики синтеза
    __slots__ = ("candidates_generated", "evals", "sieve_survivors")  # фиксированные атрибуты без словаря

    def __init__(self) -> None:  # создать счётчики
        self.candidates_generated = 0  # порождено кандидатов — 0
        self.evals = 0  # вычислений — 0
        self.sieve_survivors = 0  # прошедших фильтр — 0

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'UNARY_OPS', 'BINARY_OPS', 'CONSTS', 'TERNARY_PRIMS', 'iter_candidates_mdl', 'SynthesisStats',  # имя интерфейса
]  # конец перечня
