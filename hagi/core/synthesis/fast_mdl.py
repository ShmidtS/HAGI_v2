"""Быстрый кусочный, модульный и пакетный МДЛ-синтез и сжатие графа вызовов.

Файл ищет короткую символьную программу, которая на обучающих примерах даёт
ровно заданные ответы. Кандидаты оцениваются по длине описания в битах
(МДЛ — минимальная длина описания): перебор идёт от дешёвых к дорогим.

Основные части:
  * точная подгонка кусочного полинома: область определения делится порогом,
    в каждой части подбирается многочлен степени не выше 3 либо константа,
    условия соединяются ветвлением «если x < порог, иначе»;
  * семейство ограниченных арифметических предикатов от одного аргумента
    (чётность, простота, квадрат, делители, факториал) для задач, чьё имя
    не используется;
  * пакетный синтез: все кандидаты одной стоимости хранятся как строки
    матрицы значений и расширяются операцией op(кандидат, лист) сразу для
    всех обучающих строк; стоимость = стоимость кандидата + биты операции +
    биты листа;
  * «пере-овеществлённые» операции: функции, изобретённые ранее, добавляются
    как дополнительные листья через словарь REIFIED_OPERATIONS.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import time  # статистика, параметры интерпретатора и monotonic-часы
from typing import TYPE_CHECKING, Callable, Optional, Sequence  # аннотации типов
import sys as _sys  # доступ к загруженным модулям: алиас подпакета
import numpy as np  # численные массивы
from hagi.core.synthesis.prefilter import (  # открыть импорт из модуля предварительной фильтрации
    Optional,  # тип «значение или None» (имя пере-экспортируется модулем prefilter)
    Sequence,  # тип последовательности (имя пере-экспортируется модулем prefilter)
)  # конец импорта из prefilter
from hagi.core.language.call_dag import CallDagExpr  # имена из «call_dag» (модуль, где они определены)
from hagi.core.memory.aliases import TOKEN_BITS, gamma_bits  # имена из «aliases» (модуль, где они определены)
from hagi.core.memory.aliases import CONSTS  # имена из модуля «aliases» (прямой импорт вместо обращения через псевдоним пакета)
from hagi.core.synthesis.induction import _exact_multivariate_poly_fit, _exact_poly_fit  # имена из модуля «induction» (прямой импорт вместо обращения через псевдоним пакета)
if TYPE_CHECKING:  # только для проверки типов аннотаций (анализатор видит имя, импорта во время выполнения нет — нет цикла)
    from hagi.core.processor.stored_growth import StoredExpert  # класс эксперта из части «growth»: имя нужно лишь в строковых аннотациях

# ==============================================================================  # разделитель
# 8. Точный кусочный синтез, ограничённые предикаты, пакетный перебор  # заголовок раздела
# ==============================================================================  # разделитель


def exact_piecewise_poly_fit(
    train_x: Sequence[tuple[int, ...]], train_y: Sequence[int]  # аргументы и ответы обучающих строк
    ) -> Optional[CallDagExpr]:  # возврат: выражение либо None
    """Точная рациональная кусочно-полиномиальная и модулярная подгонка до перебора выражений.

        train_x: строки аргументов, train_y: ответы. Возврат — выражение CallDagExpr
        либо None, если простого закона не нашлось.
        Арность 1: (а) остаток по модулю m, m = 2..8: если каждому остатку x mod m
        отвечает ровно одно значение y и остатков не меньше двух, а при m = 2
        пары (0→0, 1→1), результат bit_and(x, 1); (б) порог thr из
        {−3..3}: части x < thr и x ≥ thr, в каждой не меньше 3 точек,
        полином либо константа; если части различны — if_cond(x < thr, левая, правая).
        Арность 2: части x0 < x1 и x0 ≥ x1, в каждой не меньше 6 точек, многочлен
        двух переменных степени ≤ 2; if_cond(x0 < x1, левый, правый)."""
    arity = len(train_x[0])  # арность = длина первой строки аргументов
    if arity == 1:  # один аргумент
        xs = [r[0] for r in train_x]  # список xs = x_1..x_N
        for m in range(2, 9):  # модуль m = 2..8
            buckets: dict[int, set[int]] = {}  # buckets[r] — множество значений y с x mod m = r
            for x, y in zip(xs, train_y):  # по парам (x, y)
                buckets.setdefault(x % m, set()).add(y)  # добавить y в множество остатка r = x mod m
            if all(len(vs) == 1 for vs in buckets.values()) and len(buckets) >= 2:  # в каждом остатке ровно одно значение y и остатков ≥ 2
                if m == 2 and buckets.get(0) == {0} and buckets.get(1) == {1}:  # m = 2 и отображение 0→0, 1→1
                    return CallDagExpr("bit_and", (CallDagExpr.var(0), CallDagExpr.const(1)))  # результат y = x & 1
        for thr in (-3, -2, -1, 0, 1, 2, 3):  # порог thr из семи значений
            left = [(x, y) for x, y in zip(xs, train_y) if x < thr]  # левая часть: точки с x < thr
            right = [(x, y) for x, y in zip(xs, train_y) if x >= thr]  # правая часть: точки с x ≥ thr
            if len(left) < 3 or len(right) < 3:  # в какой-то части меньше 3 точек
                continue  # порог не годится — следующий
            cl = _fit_poly_or_const(left)  # cl — полином или константа левой части
            cr = _fit_poly_or_const(right)  # cr — полином или константа правой части
            if cl is not None and cr is not None and cl != cr:  # обе части подогнались и различны
                cond = CallDagExpr("lt", (CallDagExpr.var(0), CallDagExpr.const(thr)))  # условие cond = (x < thr)
                return CallDagExpr("if_cond", (cond, cl, cr))  # результат: cl, если cond, иначе cr
    elif arity == 2:  # два аргумента
        left = [(r, y) for r, y in zip(train_x, train_y) if r[0] < r[1]]  # левая часть: x0 < x1
        right = [(r, y) for r, y in zip(train_x, train_y) if r[0] >= r[1]]  # правая часть: x0 ≥ x1
        if len(left) >= 6 and len(right) >= 6:  # в каждой части не меньше 6 точек
            tl = _exact_multivariate_poly_fit(  # tl — многочлен 2 переменных степени ≤ 2 левой части
                [r for r, _ in left], [y for _, y in left], 2, max_degree=2  # аргументы и ответы левой части, 2 переменные, степень ≤ 2
            )  # конец вызова
            tr = _exact_multivariate_poly_fit(  # tr — многочлен правой части
                [r for r, _ in right], [y for _, y in right], 2, max_degree=2  # аргументы и ответы правой части, 2 переменные, степень ≤ 2
            )  # конец вызова
            if tl is not None and tr is not None and tl != tr:  # обе части подогнались и различны
                cond = CallDagExpr("lt", (CallDagExpr.var(0), CallDagExpr.var(1)))  # условие cond = (x0 < x1)
                return CallDagExpr(  # вернуть ветвление
                    "if_cond",  # операция «если»
                    (cond, CallDagExpr("poly_nd", (), tl), CallDagExpr("poly_nd", (), tr)),  # cond, tl, если истинно, иначе tr
                )  # конец конструктора
    return None  # закон не найден


def bounded_fold_hypotheses(arity: int) -> list[CallDagExpr]:
    """Семейство ограниченных арифметических предикатов, не зависящих от имени задачи.

        arity: число аргументов. Возврат — список выражений (только при arity = 1,
        иначе пустой список). Аргумент — переменная x; переменная k — связанная
        переменная диапазона внутри тела свёртки. Гипотеза принимается только после
        проверки на отложенной и на полной конечной выборке.
        Выражения: простота x ≥ 2 ∧ ∀k∈[2, ⌊√x⌋+1): x mod k ≠ 0;
        составность x ≥ 4 ∧ ∃k∈[2, ⌊√x⌋+1): x mod k = 0; квадрат
        x ≥ 0 ∧ ∃k∈[0, x+1): k·k = x; чётность x mod 2 = 0; нечётность x mod 2 ≠ 0;
        τ(x) = Σ_{k=1..max(x,0)} [k | x]; σ(x) = Σ_{k=1..max(x,0)} k·[k | x];
        φ(x) = Σ_{k=1..max(x,0)} [gcd(x,k) = 1]; x! = [x ≥ 0]·Π_{k=1..max(x,0)} k."""
    if int(arity) != 1:  # арность не равна 1
        return []  # гипотез нет
    x, k = CallDagExpr.var(0), CallDagExpr.var(1)  # x — аргумент 0, k — связанная переменная 1
    c0, c1, c2 = CallDagExpr.const(0), CallDagExpr.const(1), CallDagExpr.const(2)  # константы 0, 1, 2
    x_ge_2 = CallDagExpr("ge", (x, c2))  # условие x_ge_2 = (x ≥ 2)
    rem = CallDagExpr("mod", (x, k))  # остаток rem = x mod k
    not_divisor = CallDagExpr("ne", (rem, c0))  # не делитель: rem ≠ 0
    divisor = CallDagExpr("eq", (rem, c0))  # делитель: rem = 0
    sqrt_stop = CallDagExpr("add", (CallDagExpr("isqrt", (CallDagExpr("max", (x, c0)),)), c1))  # верхняя граница sqrt_stop = ⌊√max(x,0)⌋ + 1
    prime = CallDagExpr(  # простота: (x ≥ 2) ∧ (∀k∈[2, sqrt_stop): x mod k ≠ 0)
        "bit_and",  # логическое «и»
        (  # кортеж аргументов
            x_ge_2,  # условие x ≥ 2
            CallDagExpr("all_range", (c2, sqrt_stop, not_divisor)),  # для всех k из [2, sqrt_stop) выполнено not_divisor
        ),  # конец кортежа
    )  # конец конструктора
    composite = CallDagExpr(  # составность: (x ≥ 4) ∧ (∃k∈[2, sqrt_stop): x mod k = 0)
        "bit_and",  # логическое «и»
        (  # кортеж аргументов
            CallDagExpr("ge", (x, CallDagExpr.const(4))),  # условие x ≥ 4
            CallDagExpr("any_range", (c2, sqrt_stop, divisor)),  # существует k из [2, sqrt_stop) с divisor
        ),  # конец кортежа
    )  # конец конструктора
    even = CallDagExpr("eq", (CallDagExpr("mod", (x, c2)), c0))  # чётность: x mod 2 = 0
    odd = CallDagExpr("ne", (CallDagExpr("mod", (x, c2)), c0))  # нечётность: x mod 2 ≠ 0
    nonnegative = CallDagExpr("ge", (x, c0))  # неотрицательность: x ≥ 0
    square = CallDagExpr(  # квадрат: (x ≥ 0) ∧ (∃k∈[0, x+1): k·k = x)
        "bit_and",  # логическое «и»
        (  # кортеж аргументов
            nonnegative,  # условие x ≥ 0
            CallDagExpr(  # квантор существования
                "any_range",  # существует k из диапазона
                (  # кортеж аргументов
                    c0,  # нижняя граница 0
                    CallDagExpr("add", (x, c1)),  # верхняя граница x + 1
                    CallDagExpr("eq", (CallDagExpr("mul", (k, k)), x)),  # тело: k·k = x
                ),  # конец кортежа
            ),  # конец квантора
        ),  # конец кортежа
    )  # конец конструктора
    # Арифметические свёртки по положительным делителям: это общие формулы, а не
    # заранее выученные таблицы: τ(n), σ(n) и φ Эйлера.
    positive_stop = CallDagExpr("add", (CallDagExpr("max", (x, c0)), c1))  # граница positive_stop = max(x, 0) + 1
    from_one = CallDagExpr.const(1)  # нижняя граница диапазона 1
    divides = CallDagExpr("eq", (CallDagExpr("mod", (x, k)), c0))  # делимость: x mod k = 0
    coprime = CallDagExpr("eq", (CallDagExpr("gcd", (x, k)), c1))  # взаимная простота: gcd(x, k) = 1
    tau = CallDagExpr("sum_range", (from_one, positive_stop, divides))  # tau = Σ_{k=1..max(x,0)} [k | x] — число делителей
    sigma = CallDagExpr("sum_range", (from_one, positive_stop, CallDagExpr("mul", (k, divides))))  # sigma = Σ_{k=1..max(x,0)} k·[k | x] — сумма делителей
    phi = CallDagExpr("sum_range", (from_one, positive_stop, coprime))  # φ Эйлера = Σ_{k=1..max(x,0)} [gcd(x,k) = 1]
    # Пустое произведение равно 1; умножение на (x ≥ 0) задаёт для отрицательных x ноль.
    factorial = CallDagExpr(  # факториал = [x ≥ 0]·Π_{k=1..max(x,0)} k
        "mul",  # умножение
        (  # кортеж аргументов
            CallDagExpr("ge", (x, c0)),  # множитель [x ≥ 0]
            CallDagExpr("prod_range", (from_one, positive_stop, k)),  # произведение Π_{k=1..max(x,0)} k
        ),  # конец кортежа
    )  # конец конструктора
    return [prime, composite, square, even, odd, tau, sigma, phi, factorial]  # список из девяти гипотез


def _fit_poly_or_const(pts: list[tuple[int, int]]) -> Optional[CallDagExpr]:
    """Подогнать по точкам константу либо полином степени ≤ 3 от x.

        pts: пары (x, y). Если все y равны — константа y; иначе точный полином
        по всем точкам; если он не существует — None."""
    ys = {y for _, y in pts}  # множество различных значений y
    if len(ys) == 1:  # значение одно
        return CallDagExpr.const(next(iter(ys)))  # константа, равная этому y
    coeffs = _exact_poly_fit(  # coeffs — точные коэффициенты полинома либо None
        [x for x, _ in pts], [y for _, y in pts], max_degree=3  # абсциссы, ординаты, степень ≤ 3
    )  # конец вызова
    if coeffs is not None:  # полином найден
        return CallDagExpr("poly", (CallDagExpr.var(0),), coeffs)  # полином от переменной x с коэффициентами coeffs
    return None  # не найден


# Здесь живут изобретённые операции, подключаемые к пакетному синтезу:
# регистрируются при каждом вызове синтеза (словарь REIFIED_OPERATIONS).
REIFIED_OPERATIONS: dict = {}  # словарь «reified:<имя>» → функция от одного аргумента


def _leaf_iter(leaf_exprs, reified_leaf_fns):
    """Выдавать пары (лист, функция_или_None), выровненные по списку leaf_exprs.

        Лист с операцией "reified" получает очередную функцию из reified_leaf_fns,
        прочий лист — None."""
    ri = 0  # индекс следующей пере-овеществлённой функции
    for lf in leaf_exprs:  # по всем листьям
        if lf.op == "reified":  # лист — изобретённая операция
            yield lf, reified_leaf_fns[ri]  # выдать лист и его функцию
            ri += 1  # индекс функции + 1
        else:  # обычный лист (переменная или константа)
            yield lf, None  # выдать лист без функции


def synthesize_batched_mdl_stored(
    arity: int,  # число аргументов
    train_x: Sequence[tuple[int, ...]],  # аргументы обучения
    train_y: Sequence[int],  # ответы обучения
    val_x: Sequence[tuple[int, ...]],  # аргументы проверки
    val_y: Sequence[int],  # ответы проверки
    library_experts: Sequence[tuple[int, str, "StoredExpert"]] = (),  # эксперты библиотеки
    max_evals: int = 4000,  # предел числа проверенных кандидатов
    max_bits: int = 84,  # предел стоимости в битах
    deadline: Optional[float] = None,  # предельное время либо None
    reified_ops: Optional[dict] = None,  # изобретённые операции либо None
    ) -> tuple[Optional[CallDagExpr], int]:  # возврат: программа (или None) и счётчик
    """Пакетный МДЛ-синтез на целых int64 с отбрасыванием повторов и ссылками на библиотеку.

        reified_ops: необязательный словарь {имя: (функция_1d, цена_в_битах)} —
        изобретённые операции добавляются как дополнительные листья, так что
        вчерашнее изобретение становится строительным блоком сегодняшней программы.
        arity: число аргументов; train_x/train_y — обучающие строки и ответы;
        val_x/val_y — отложенная проверка найденной программы; library_experts —
        до 24 элементов (номер, имя, эксперт) библиотеки; max_evals — предел числа
        зарегистрированных кандидатов; max_bits — предел стоимости в битах;
        deadline — предельное время monotonic-часов или None.
        Возврат: (программа, число проверенных кандидатов) либо (None, то же число).
        Стоимость кандидата = Σ(биты операции + биты листа); расширение
        op(кандидат b, лист l) стоит b + TOKEN_BITS + биты(l); вызов эксперта
        библиотеки — b + TOKEN_BITS + gamma_bits(номер + 1); удвоение
        op(b, b) — 2b + TOKEN_BITS. Кандидат принимается, если совпадает с Y на всех
        обучающих строках и проходит отложенную проверку."""
    """Пакетный синтез на массивах NumPy int64 с отбрасыванием повторов и ссылками на граф вызовов."""
    try:  # попытка преобразовать вход в массивы
        X = np.asarray(train_x, dtype=np.int64)  # X — матрица аргументов (N, arity)
        Y = np.asarray(train_y, dtype=np.int64)  # Y — вектор ответов (N,)
    except (OverflowError, ValueError, TypeError):  # переполнение int64 или неверная форма/тип
        return None, 0  # ничего не найдено, проверено 0
    if (  # проверка формы входа
        X.ndim != 2  # X не двумерная
        or X.shape[1] != int(arity)  # число столбцов не равно арности
        or X.shape[0] == 0  # нет ни одной строки
        or Y.shape != (X.shape[0],)  # длина Y не равна N
    ):  # конец условия
        raise ValueError(  # ошибка неверной формы
            "training arrays must have shapes (N, arity) and (N,), with N>0"  # текст ошибки
        )  # конец вызова
    # Здесь нужны python-целые: np.abs(int64_min) переполняется обратно в int64_min.
    if any(abs(int(v)) > 128 for v in X.reshape(-1)):  # есть аргумент с |v| > 128
        return None, 0  # вне допустимого диапазона — отказ
    if int(max_evals) <= 0 or int(max_bits) <= 0:  # нулевой или отрицательный предел
        return None, 0  # отказ
    N = X.shape[0]  # N — число обучающих строк
    leaf_exprs = [CallDagExpr.var(i) for i in range(arity)] + [  # листья: переменные var(0..arity−1) плюс
        CallDagExpr.const(c) for c in CONSTS  # константы из общего набора
    ]  # конец списка листьев
    # Пере-овеществлённые листья: изобретённые операции как унарные макросы (по всем строкам).
    reified_leaf_fns = []  # функции изобретённых листьев в порядке их добавления
    if reified_ops:  # словарь изобретённых операций задан
        for rname, (rfn, rcost) in reified_ops.items():  # по парам имя → (функция, цена)
            full = f"reified:{rname}"  # полное имя «reified:имя»
            REIFIED_OPERATIONS[full] = rfn  # зарегистрировать функцию под полным именем
            leaf_exprs.append(CallDagExpr("reified", (), full))  # добавить лист операции
            reified_leaf_fns.append(rfn)  # запомнить функцию листа
    L_arr = np.stack(  # L_arr — матрица (число листьев, N) значений листьев
        [np.ascontiguousarray(X[:, int(lf.value or 0)])  # для переменной — столбец аргумента с номером value
         if lf.op == "var"  # условие «лист — переменная»
         else (np.array([int(rfn(*[int(v) for v in r])) for r in X],  # иначе для изобретённого листа: rfn(строка) по всем строкам
                    dtype=np.int64)  # тип элементов int64
               if lf.op == "reified"  # условие «лист — изобретённая операция»
               else np.full(N, int(lf.value), dtype=np.int64))  # иначе константа: N копий значения value
         for lf, rfn in _leaf_iter(leaf_exprs, reified_leaf_fns)  # по парам (лист, функция) списка листьев
        ]  # конец списка столбцов
    )  # конец stack
    L_bits = [  # L_bits — цена каждого листа в битах
        (reified_ops[lf.value.split(":", 1)[1]][1]  # для изобретённого листа — цена из словаря по имени после «reified:»
         if lf.op == "reified" else lf.bits())  # иначе число битов листа
        for lf in leaf_exprs  # по всем листьям
    ]  # конец списка

    for i, lf in enumerate(leaf_exprs):  # лист сам по себе может быть ответом
        if np.array_equal(L_arr[i], Y):  # значения листа совпадают с Y на всех строках
            if all(lf.eval(r) == y for r, y in zip(val_x, val_y)):  # и на отложенной выборке
                return lf, 1  # вернуть лист, проверен 1 кандидат

    seen_sigs = {L_arr[i].tobytes(): 1 for i in range(len(leaf_exprs))}  # сигнатуры (байты строки значений) → число повторов, у листьев 1
    buckets: dict[int, list[tuple[CallDagExpr, np.ndarray]]] = {}  # buckets[биты] — кандидаты данной стоимости (выражение, значения)
    for i, lf in enumerate(leaf_exprs):  # по листьям
        buckets.setdefault(L_bits[i], []).append((lf, L_arr[i]))  # положить лист в корзину его стоимости

    total_checked = len(leaf_exprs)  # счётчик проверенных кандидатов = число листьев
    MAG_CAP = 1 << 28  # предел модуля значения 2^28
    all_ops = [  # список бинарных операций расширения
        "add",  # сумма a + b
        "sub",  # разность a − b
        "mul",  # произведение a·b
        "floordiv",  # частное ⌊a / b⌋
        "mod",  # остаток a mod b
        "gcd",  # наибольший общий делитель gcd(a, b)
        "lcm",  # наименьшее общее кратное lcm(a, b)
        "min",  # минимум min(a, b)
        "max",  # максимум max(a, b)
        "eq",  # равно: [a = b]
        "ne",  # не равно: [a ≠ b]
        "lt",  # меньше: [a < b]
        "le",  # не больше: [a ≤ b]
        "gt",  # больше: [a > b]
        "ge",  # не меньше: [a ≥ b]
        "bit_and",  # «и»: a ∧ b
        "bit_or",  # «или»: a ∨ b
        "bit_xor",  # исключающее «или»: a ⊕ b
        "bit_nand8",  # ¬(a ∧ b) по младшим 8 битам
        "bit_nor8",  # ¬(a ∨ b) по младшим 8 битам
        "bit_xnor8",  # ¬(a ⊕ b) по младшим 8 битам
        "pow_int",  # степень a^b
    ]  # конец списка
    env = {name: ex for _, name, ex in library_experts}  # env — окружение имя эксперта → эксперт

    while buckets and total_checked < max_evals:  # пока есть корзины и не исчерпан лимит проверок
        if deadline is not None and time.monotonic() >= deadline:  # задан срок и он истёк
            break  # стоп
        min_b = min(buckets)  # минимальная стоимость среди корзин
        if min_b >= max_bits:  # самый дешёвый кандидат уже дороже предела
            break  # стоп
        batch = buckets.pop(min_b)  # взять и удалить корзину стоимости min_b
        if not batch:  # корзина пуста
            continue  # следующая итерация
        if len(batch) > 256:  # кандидатов больше 256
            batch = batch[:256]  # оставить первые 256
        B_exprs = [x[0] for x in batch]  # выражения кандидатов
        V = np.stack([x[1] for x in batch])  # V — матрица значений (размер пачки, N)

        def register(  # зарегистрировать результаты расширения
            res_arr: np.ndarray, builder_fn: Callable[[int, int], CallDagExpr], new_bits: int  # значения, конструктор, стоимость
        ) -> Optional[CallDagExpr]:  # возврат: программа либо None
            """Проверить res_arr на совпадение с Y; совпавшее вернуть, прочее положить в корзину.

                res_arr: значения формы (B, N) или (B, K, N); builder_fn(bi, ki) строит
                выражение кандидата; new_bits — его стоимость."""
            nonlocal total_checked  # изменение счётчика внешней функции
            if new_bits > max_bits:  # стоимость выше предела
                return None  # ничего
            if res_arr.ndim == 2:  # форма (B, N)
                res_arr = res_arr[:, None, :]  # добавить ось: (B, 1, N)
            matches = np.all(res_arr == Y[None, None, :], axis=2)  # matches[b, k] = (значения по всем N строкам равны Y)
            if np.any(matches):  # есть точные совпадения
                for bi, ki in np.argwhere(matches):  # по парам индексов совпадений
                    cand = builder_fn(int(bi), int(ki))  # построить выражение кандидата
                    try:  # проверка на отложенной выборке
                        if all(cand.eval(r, env) == y for r, y in zip(val_x, val_y)):  # все ответы верны
                            return cand  # вернуть программу
                    except Exception:  # ошибка вычисления кандидата
                        pass  # пропустить
            if new_bits + TOKEN_BITS > max_bits or total_checked >= max_evals:  # расширение уже не уложится или лимит исчерпан
                return None  # ничего
            ok_mag = np.max(np.abs(res_arr), axis=2) <= MAG_CAP  # ok_mag[b, k] = max_n |значение| ≤ 2^28
            for bi, ki in np.argwhere(ok_mag):  # по кандидатам с допустимым модулем
                if total_checked >= max_evals:  # лимит проверок достигнут
                    break  # выход из цикла
                row = res_arr[bi, ki]  # строка значений кандидата
                key = row.tobytes()  # сигнатура — байты строки
                cnt = seen_sigs.get(key, 0)  # сколько раз сигнатура уже встречалась
                if cnt >= 2:  # уже два раза
                    continue  # третий раз не храним
                seen_sigs[key] = cnt + 1  # счётчик сигнатуры + 1
                total_checked += 1  # проверенных кандидатов + 1
                buckets.setdefault(new_bits, []).append(  # положить в корзину стоимости new_bits
                    (builder_fn(int(bi), int(ki)), row)  # пару (выражение, значения)
                )  # конец вызова
            return None  # точного совпадения нет

        for uop, ures in (  # унарные операции: имя и значения (B, N)
            ("neg", -V),  # отрицание −v
            ("abs", np.abs(V)),  # модуль |v|
            ("sign", np.sign(V).astype(np.int64)),  # sign = знак v: −1, 0, 1
            ("bit_not8", (~V) & 0xFF),  # bit_not8 = ¬v по младшим 8 битам
            ("clamp3", np.clip(V, -3, 3)),  # ограничение max(−3, min(3, v))
        ):  # конец кортежа
            hit = register(  # зарегистрировать результаты унарной операции
                ures, lambda bi, _, o=uop: CallDagExpr(o, (B_exprs[bi],)), min_b + TOKEN_BITS  # значения, конструктор o(b), стоимость min_b + биты операции
            )  # конец вызова
            if hit is not None:  # найдена программа
                return hit, total_checked  # вернуть её и счётчик

        for lib_idx, ename, ex in library_experts[:24]:  # по первым 24 экспертам библиотеки
            cbits = TOKEN_BITS + gamma_bits(lib_idx + 1)  # cbits = биты операции + gamma_bits(номер + 1)
            if ex.program.arity == 1:  # эксперт с одним аргументом
                rows_out = np.empty_like(V)  # заготовка результатов (B, N)
                ok_rows = []  # индексы строк, где эксперт применим
                for bi in range(V.shape[0]):  # по кандидатам пачки
                    if B_exprs[bi].op == "const":  # кандидат — константа
                        continue  # применить к константе нельзя
                    try:  # вычисление может упасть
                        vals_b = V[bi]  # значения кандидата по всем строкам
                        if all(ex.supports((int(v),)) for v in vals_b):  # эксперт поддерживает все значения
                            rows_out[bi] = [  # записать результаты
                                ex.program.eval((int(v),), env) for v in vals_b  # эксперт(v) по всем строкам
                            ]  # конец списка
                            ok_rows.append(bi)  # строка bi вычислена
                    except Exception:  # ошибка вычисления
                        pass  # пропустить
                if ok_rows:  # есть применимые строки
                    sub_r = rows_out[ok_rows]  # срез вычисленных строк
                    hit = register(  # зарегистрировать вызовы эксперта
                        sub_r,  # значения
                        lambda bi, _, o=ename, idxs=ok_rows: CallDagExpr(  # конструктор
                            "call", (B_exprs[idxs[bi]],), o  # call(кандидат idxs[bi], имя эксперта)
                        ),  # конец конструктора
                        min_b + cbits,  # стоимость min_b + cbits
                    )  # конец вызова
                    if hit is not None:  # найдена программа
                        return hit, total_checked  # вернуть её и счётчик
            elif ex.program.arity == 2:  # эксперт с двумя аргументами
                for li, lf in enumerate(leaf_exprs):  # по листьям
                    lv = L_arr[li]  # значения листа по всем строкам
                    for order_idx in (0, 1):  # порядок аргументов: (кандидат, лист) либо (лист, кандидат)
                        rows_out = np.empty_like(V)  # заготовка результатов
                        ok_rows = []  # индексы применимых строк
                        for bi in range(V.shape[0]):  # по кандидатам пачки
                            if B_exprs[bi].op == "const" and lf.op == "const":  # оба аргумента — константы
                                continue  # пропуск
                            bv = V[bi]  # значения кандидата
                            a_v, b_v = (bv, lv) if order_idx == 0 else (lv, bv)  # a_v, b_v — аргументы в нужном порядке
                            try:  # вычисление может упасть
                                if all(  # эксперт поддерживает все пары
                                    ex.supports((int(a_v[k]), int(b_v[k])))  # проверка пары (a_v[k], b_v[k])
                                    for k in range(N)  # по всем строкам
                                ):  # конец условия
                                    rows_out[bi] = [  # записать результаты
                                        ex.program.eval((int(a_v[k]), int(b_v[k])), env)  # эксперт(a_v[k], b_v[k])
                                        for k in range(N)  # по всем строкам
                                    ]  # конец списка
                                    ok_rows.append(bi)  # строка bi вычислена
                            except Exception:  # ошибка вычисления
                                pass  # пропустить
                        if ok_rows:  # есть применимые строки
                            sub_r = rows_out[ok_rows]  # срез вычисленных строк
                            hit = register(  # зарегистрировать вызовы эксперта
                                sub_r,  # значения
                                (  # конструктор в скобках
                                    lambda bi, _, o=ename, l=lf, idxs=ok_rows, oi=order_idx: CallDagExpr(  # bi — индекс в подмножестве
                                        "call",  # операция вызова
                                        (  # аргументы
                                            (B_exprs[idxs[bi]], l)  # порядок 0: (кандидат, лист)
                                            if oi == 0  # если порядок 0
                                            else (l, B_exprs[idxs[bi]])  # иначе: (лист, кандидат)
                                        ),  # конец аргументов
                                        o,  # имя эксперта
                                    )  # конец конструктора
                                ),  # конец конструктора в скобках
                                min_b + cbits + L_bits[li],  # стоимость min_b + cbits + биты листа
                            )  # конец вызова
                            if hit is not None:  # найдена программа
                                return hit, total_checked  # вернуть её и счётчик

        Va = V[:, None, :]  # Va — значения кандидатов в форме (B, 1, N)
        Lb = L_arr[None, :, :]  # Lb — значения листьев в форме (1, L, N)
        for op in all_ops:  # по бинарным операциям
            if op == "add":  # сложение
                R = Va + Lb  # сумма R[b, l, n] = v_bn + l_ln
            elif op == "sub":  # вычитание
                R = Va - Lb  # разность v − l
            elif op == "mul":  # умножение
                R = Va * Lb  # произведение v·l
            elif op == "min":  # минимум
                R = np.minimum(Va, Lb)  # минимум min(v, l)
            elif op == "max":  # максимум
                R = np.maximum(Va, Lb)  # максимум max(v, l)
            elif op == "eq":  # равенство
                R = (Va == Lb).astype(np.int64)  # равно: R = [v = l]
            elif op == "ne":  # неравенство
                R = (Va != Lb).astype(np.int64)  # не равно: R = [v ≠ l]
            elif op == "lt":  # меньше
                R = (Va < Lb).astype(np.int64)  # меньше: R = [v < l]
            elif op == "le":  # не больше
                R = (Va <= Lb).astype(np.int64)  # не больше: R = [v ≤ l]
            elif op == "gt":  # больше
                R = (Va > Lb).astype(np.int64)  # больше: R = [v > l]
            elif op == "ge":  # не меньше
                R = (Va >= Lb).astype(np.int64)  # не меньше: R = [v ≥ l]
            elif op == "bit_and":  # побитовое «и»
                R = Va & Lb  # «и» v ∧ l
            elif op == "bit_or":  # побитовое «или»
                R = Va | Lb  # «или» v ∨ l
            elif op == "bit_xor":  # побитовое исключающее «или»
                R = Va ^ Lb  # исключающее «или» v ⊕ l
            elif op == "bit_nand8":  # штрих Шеффера
                R = (~(Va & Lb)) & 0xFF  # штрих Шеффера ¬(v ∧ l) & 255
            elif op == "bit_nor8":  # стрелка Пирса
                R = (~(Va | Lb)) & 0xFF  # стрелка Пирса ¬(v ∨ l) & 255
            elif op == "bit_xnor8":  # равнозначность
                R = (~(Va ^ Lb)) & 0xFF  # равнозначность ¬(v ⊕ l) & 255
            elif op == "floordiv":  # деление нацело
                safe = np.where(Lb != 0, Lb, 1)  # safe = l, если l ≠ 0, иначе 1 (защита от деления на ноль)
                R = np.where(  # R = ⌊v / l⌋, если l ≠ 0, иначе 2^28 + 1 (запрещённое значение)
                    np.broadcast_to(Lb != 0, np.broadcast_shapes(Va.shape, Lb.shape)),  # маска l ≠ 0, растянутая до формы (B, L, N)
                    np.floor_divide(Va, safe),  # частное ⌊v / safe⌋
                    MAG_CAP + 1,  # значение-запрет
                )  # конец where
            elif op == "mod":  # остаток
                safe = np.where(Lb != 0, Lb, 1)  # safe = l, если l ≠ 0, иначе 1
                R = np.where(  # R = v mod l, если l ≠ 0, иначе 2^28 + 1
                    np.broadcast_to(Lb != 0, np.broadcast_shapes(Va.shape, Lb.shape)),  # маска l ≠ 0
                    np.mod(Va, safe),  # остаток v mod safe
                    MAG_CAP + 1,  # значение-запрет
                )  # конец where
            elif op == "gcd":  # наибольший общий делитель
                R = np.gcd(Va, Lb)  # наибольший общий делитель gcd(v, l)
            elif op == "lcm":  # наименьшее общее кратное
                ok_l = (Va != 0) & (Lb != 0)  # допустимо: (v ≠ 0) ∧ (l ≠ 0)
                sa = np.where(ok_l, Va, 1)  # sa = v, если ok_l, иначе 1
                sb = np.where(ok_l, Lb, 1)  # sb = l, если ok_l, иначе 1
                R = np.where(ok_l, np.abs(sa // np.gcd(sa, sb) * sb), MAG_CAP + 1)  # R = |v/gcd(v,l)·l|, если ok_l, иначе 2^28 + 1
            elif op == "pow_int":  # целая степень
                ok_p = (Lb >= 0) & (Lb <= 8) & (np.abs(Va) <= 100)  # допустимо: (0 ≤ l ≤ 8) ∧ (|v| ≤ 100)
                sa = np.where(ok_p, Va, 0)  # sa = v, если ok_p, иначе 0
                sb = np.where(ok_p, Lb, 0)  # sb = l, если ok_p, иначе 0
                R = np.where(ok_p, sa**sb, MAG_CAP + 1)  # R = v^l, если ok_p, иначе 2^28 + 1
            else:  # операция не распознана
                continue  # пропуск
            for li in range(len(leaf_exprs)):  # по листьям
                hit = register(  # зарегистрировать op(кандидат, лист)
                    R[:, li : li + 1, :],  # срез R[:, li:li+1, :] — только лист li
                    lambda bi, _, o=op, l=leaf_exprs[li]: CallDagExpr(o, (B_exprs[bi], l)),  # конструктор op(b, l)
                    min_b + TOKEN_BITS + L_bits[li],  # стоимость min_b + биты операции + биты листа
                )  # конец вызова
                if hit is not None:  # найдена программа
                    return hit, total_checked  # вернуть её и счётчик
            if op in ("sub", "floordiv", "mod", "lt", "le", "gt", "ge", "pow_int"):  # некоммутативные операции
                if op == "sub":  # вычитание
                    R2 = Lb - Va  # обратная разность l − v
                elif op == "lt":  # меньше
                    R2 = (Lb < Va).astype(np.int64)  # меньше: R2 = [l < v]
                elif op == "le":  # не больше
                    R2 = (Lb <= Va).astype(np.int64)  # не больше: R2 = [l ≤ v]
                elif op == "gt":  # больше
                    R2 = (Lb > Va).astype(np.int64)  # больше: R2 = [l > v]
                elif op == "ge":  # не меньше
                    R2 = (Lb >= Va).astype(np.int64)  # не меньше: R2 = [l ≥ v]
                elif op == "floordiv":  # деление нацело
                    safe = np.where(Va != 0, Va, 1)  # safe = v, если v ≠ 0, иначе 1
                    R2 = np.where(  # R2 = ⌊l / v⌋, если v ≠ 0, иначе 2^28 + 1
                        np.broadcast_to(  # маска v ≠ 0
                            Va != 0, np.broadcast_shapes(Va.shape, Lb.shape)  # растянутая до формы (B, L, N)
                        ),  # конец broadcast_to
                        np.floor_divide(Lb, safe),  # частное ⌊l / safe⌋
                        MAG_CAP + 1,  # значение-запрет
                    )  # конец where
                elif op == "mod":  # остаток
                    safe = np.where(Va != 0, Va, 1)  # safe = v, если v ≠ 0, иначе 1
                    R2 = np.where(  # R2 = l mod v, если v ≠ 0, иначе 2^28 + 1
                        np.broadcast_to(  # маска v ≠ 0
                            Va != 0, np.broadcast_shapes(Va.shape, Lb.shape)  # растянутая до формы (B, L, N)
                        ),  # конец broadcast_to
                        np.mod(Lb, safe),  # остаток l mod safe
                        MAG_CAP + 1,  # значение-запрет
                    )  # конец where
                elif op == "pow_int":  # целая степень
                    ok_p = (Va >= 0) & (Va <= 8) & (np.abs(Lb) <= 100)  # допустимо: (0 ≤ v ≤ 8) ∧ (|l| ≤ 100)
                    sa = np.where(ok_p, Lb, 0)  # sa = l, если ok_p, иначе 0
                    sb = np.where(ok_p, Va, 0)  # sb = v, если ok_p, иначе 0
                    R2 = np.where(ok_p, sa**sb, MAG_CAP + 1)  # R2 = l^v, если ok_p, иначе 2^28 + 1
                for li in range(len(leaf_exprs)):  # по листьям
                    hit = register(  # зарегистрировать op(лист, кандидат)
                        R2[:, li : li + 1, :],  # срез R2[:, li:li+1, :]
                        lambda bi, _, o=op, l=leaf_exprs[li]: CallDagExpr(  # конструктор
                            o, (l, B_exprs[bi])  # операция op(l, b)
                        ),  # конец конструктора
                        min_b + TOKEN_BITS + L_bits[li],  # стоимость min_b + биты операции + биты листа
                    )  # конец вызова
                    if hit is not None:  # найдена программа
                        return hit, total_checked  # вернуть её и счётчик
        for op in ("mul", "add"):  # коммутативные операции с одинаковыми аргументами
            R_s = V * V if op == "mul" else V + V  # R_s = v·v при умножении, иначе v + v
            hit = register(  # зарегистрировать op(b, b)
                R_s,  # значения
                lambda bi, _, o=op: CallDagExpr(o, (B_exprs[bi], B_exprs[bi])),  # конструктор op(b, b)
                min_b * 2 + TOKEN_BITS,  # стоимость 2·min_b + биты операции
            )  # конец вызова
            if hit is not None:  # найдена программа
                return hit, total_checked  # вернуть её и счётчик

    return None, total_checked  # программа не найдена — None и число проверенных

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'exact_piecewise_poly_fit', 'bounded_fold_hypotheses', 'REIFIED_OPERATIONS', 'synthesize_batched_mdl_stored',  # имя интерфейса
]  # конец перечня
