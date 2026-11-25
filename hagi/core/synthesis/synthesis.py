"""Точная полиномиальная индукция, кусочный решатель, МДЛ-синтез — часть символьного ядра HAGI.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import itertools  # импорт стандартной библиотеки, нужный этой части
import math  # импорт стандартной библиотеки, нужный этой части
import time  # импорт стандартной библиотеки, нужный этой части
from fractions import Fraction  # импорт стандартной библиотеки, нужный этой части
from typing import Callable, Optional, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.core.memory.aliases import BINARY_OPS, COMMUTATIVE_OPS, CONSTS, TOKEN_BITS, _PRIME_MOD  # имена из «aliases» (прежний общий импорт из пакета наследия)
from hagi.core.synthesis.prefilter import OPS  # имена из «prefilter» (прежний общий импорт из пакета наследия)
from hagi.core.synthesis.prefilter import Expert  # имена из модуля «prefilter» (как до слияния)
from hagi.core.memory.aliases import gamma_bits  # имена из «aliases» (прежний общий импорт из пакета наследия)
from hagi.core.language.expressions import Expr  # имена из модуля ядра «expressions» (как до слияния)
from hagi.core.language.expressions import eval_expr_vec  # имена из «expressions» (модуль, где они определены)

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
                        "if_cond",  # имя узла — условный выбор
                        (Expr("lt", (Expr.var(0), Expr.const(thr))), cl, cr),  # if(x < thr, p_лев, p_прав)
                    )
    elif 2 <= arity <= 3:  # арность 2 или 3
        preds: list[tuple[Expr, Callable[[tuple[int, ...]], bool]]] = [  # кандидаты-предикаты (выражение, функция)
            (Expr("lt", (Expr.var(0), Expr.var(1))), lambda r: r[0] < r[1]),  # предикат «меньше»: x_0 < x_1
            (Expr("le", (Expr.var(0), Expr.var(1))), lambda r: r[0] <= r[1]),  # предикат «меньше или равно»: x_0 ≤ x_1
            (Expr("eq", (Expr.var(0), Expr.var(1))), lambda r: r[0] == r[1]),  # предикат равенства: x_0 = x_1
            (  # открытие кортежа
                Expr("bit_and", (Expr.var(0), Expr.const(1))),  # выделение младшего бита: x_0 & 1
                lambda r: (r[0] & 1) != 0,  # нечётность x_0
            ),
        ]
        if arity == 2:  # арность 2 — добавить знаковые предикаты
            preds.append(  # добавить
                (Expr("lt", (Expr.var(0), Expr.const(0))), lambda r: r[0] < 0)  # отрицательность первого: x_0 < 0
            )
            preds.append(  # добавить
                (Expr("lt", (Expr.var(1), Expr.const(0))), lambda r: r[1] < 0)  # отрицательность второго: x_1 < 0
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
                        Expr("add", (  # узел сложения: add(
                            Expr("mul", (Expr.var(0), Expr.const(a))),  # умножение на коэффициент: mul(x, a)
                            Expr.const(b))),  # константа b
                        Expr.const(m)))  # mod-константа m
        elif arity == 2:  # два аргумента
            x1, x2 = X[:, 0], X[:, 1]  # столбцы x_1, x_2
            for a1 in (-2, -1, 0, 1, 2):  # по a_1
                for a2 in (-2, -1, 0, 1, 2):  # по a_2
                    b = (y0 - a1 * int(X0[0]) - a2 * int(X0[1])) % m  # сдвиг: b = (y_0 − a_1·x_10 − a_2·x_20) mod m
                    if np.array_equal((a1 * x1 + a2 * x2 + b) % m, Y):  # совпадает на всех примерах
                        e = Expr("add", (  # сумма
                            Expr("mul", (Expr.var(0), Expr.const(a1))),  # умножение на коэффициент: mul(x_1, a_1)
                            Expr("mul", (Expr.var(1), Expr.const(a2)))))  # умножение на коэффициент: mul(x_2, a_2)
                        return Expr("mod", (  # mod(выражение, m)
                            Expr("add", (e, Expr.const(int(b)))),  # прибавление константы: add(e, b)
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
                            shifted = Expr("add", (Expr.var(0),  # сдвиг входа: x + (c − lo)
                                                   Expr.const(c - lo)))  # константа c − lo
                            inner = Expr("mod", (shifted,  # mod(сдвиг, m)
                                                 Expr.const(mm)))  # модуль m
                            inner = Expr("add", (inner,  # прибавление нижней границы: + lo
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
        if arity == 1 and _CURRENT_FAMILY[0] == "language"                 and set(np.unique(Y)) <= {0, 1}:  # условие: унарная задача языкового семейства, выходы только 0 и 1
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
        V = np.stack([x[1] for x in batch]).astype(np.int64, copy=False)  # значения пакета (B, N), всегда int64: если все программы пакета булевы, стек получал тип bool, и операции «−» и sign над ним падают

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
            ("neg", -V),  # кандидат: −V
            ("abs", np.abs(V)),  # кандидат: |V|
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
                        if lf.op == "const"  # условие: лист — константа
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

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'exact_piecewise_poly_fit', 'synthesize_batched_mdl',  # имя интерфейса
]  # конец перечня
