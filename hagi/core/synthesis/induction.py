"""Решатель раскраски графа, точная индукция формул и майнер теорем.
Раскраска: энергия = число рёбер с одинаковым цветом концов; спуск по энергии с точной проверкой результата.
Индукция: интерполяция целочисленного многочлена над рациональными дробями; майнер перебирает тождества на всей поддерживаемой области эксперта."""


from __future__ import annotations  # отложенное вычисление аннотаций типов
import itertools, math  # модули стандартной библиотеки: аргументы, дерево, копия, куча, произведения, JSON, математика, ОС, случайность
from dataclasses import dataclass  # декоратор класса данных и поле по умолчанию
from fractions import Fraction  # точная рациональная дробь
from typing import TYPE_CHECKING, Any, Optional, Sequence  # типы аннотаций
import numpy as np  # численные массивы numpy
from hagi.core.language.expr_dsl import (  # импорт из модуля выражений expr_dsl
    Expr  # Expr — дерево выражения программы
)  # конец импорта
from typing import Any, Optional, Sequence  # имена из «stdlib:typing» (модуль, где они определены)
from hagi.core.language.expr_dsl import Expr  # имена из «expr_dsl» (модуль, где они определены)
if TYPE_CHECKING:  # только для проверки типов аннотаций (анализатор видит имя, импорта во время выполнения нет — нет цикла)
    from hagi.core.processor.benchmark import OptimalMathProcessor  # процессор из части «benchmark»: имя нужно лишь в строковых аннотациях

# ------------------- решатель раскраски графа спуском по энергии ------------------


def solve_coloring(  # найти раскраску графа: цвета концов каждого ребра различны
    edges,  # edges — рёбра (a, b)
    n_nodes: int,  # n_nodes — число вершин с номерами 0..n_nodes−1
    num_colors: int,  # num_colors — число цветов
    givens: Optional[dict[int, int]] = None,  # givens — фиксированные цвета вершин или None
    max_sweeps: int = 10000,  # max_sweeps — предел числа проходов
) -> dict[str, Any]:  # возвращает словарь с раскраской и её проверкой
    """Спуск по энергии для раскраски графа: энергия E = число рёбер (a, b) с равными цветами концов.

    Вершина перекрашивается только при строгом уменьшении числа её конфликтов, поэтому
    на конечном множестве состояний спуск останавливается. Найденный минимум локальный:
    без улучшений берётся новая случайная начальная раскраска. Результат проверяется точно.
    Возвращает {'coloring', 'valid', 'energy', 'sweeps', 'verified', 'givens_respected'}.
    """
    from collections import defaultdict  # импорт словаря со значением по умолчанию

    adj = defaultdict(set)  # adj[v] — множество соседей вершины v
    for a, b in edges:  # по рёбрам (a, b)
        adj[a].add(b)  # b — сосед a
        adj[b].add(a)  # a — сосед b
    rng = np.random.default_rng(42)  # генератор случайных чисел с зерном 42
    givens = givens or {}  # пустой словарь, если givens не задан
    y = {i: givens.get(i, int(rng.integers(0, num_colors))) for i in range(n_nodes)}  # начальная раскраска: y_i = givens[i], иначе случайный цвет из 0..num_colors−1

    def energy(state) -> int:  # энергия состояния
        return sum(1 for a, b in edges if state[a] == state[b])  # E = Σ_{(a,b)} [state[a] == state[b]] — число конфликтных рёбер

    cur = energy(y)  # текущая энергия
    sweeps = 0  # счётчик проходов
    while sweeps < max_sweeps:  # пока проходов меньше предела
        improved = False  # флаг: было ли улучшение в проходе
        for i in range(n_nodes):  # по всем вершинам
            if i in givens:  # фиксированные вершины не меняются
                continue  # следующая вершина
            # точное локальное поле: counts[c] = число соседей вершины i цвета c;
            # при смене цвета на c энергия меняется на counts[c] − counts[y_i]
            counts = {c: sum(1 for j in adj[i] if y[j] == c) for c in range(num_colors)}  # counts[c] = Σ_{j ∈ adj[i]} [y_j == c] — число соседей цвета c
            best_c, best_conf = y[i], counts[y[i]]  # лучший цвет и число его конфликтов — текущие
            for c in range(num_colors):  # по цветам-кандидатам
                if counts[c] < best_conf:  # меньше конфликтов, чем у текущего лучшего
                    best_c, best_conf = c, counts[c]  # запомнить best_c = c, best_conf = counts[c]
            if best_conf < counts[y[i]]:  # строгое уменьшение конфликтов — шаг спуска
                y[i] = best_c  # перекрасить вершину i
                improved = True  # улучшение было
        cur = energy(y)  # пересчитать энергию
        sweeps += 1  # проходов: sweeps + 1
        if cur == 0:  # энергия нулевая — раскраска верна
            break  # выход из цикла
        if not improved:  # выход из цикла
            # координатный минимум (не глобальный): новая случайная начальная раскраска
            y = {  # новая раскраска y
                i: givens.get(i, int(rng.integers(0, num_colors)))  # y_i = givens[i], иначе случайный цвет
                for i in range(n_nodes)  # по вершинам
            }
    # верная раскраска ⇔ энергия равна нулю: точная проверка
    valid = all(y[a] != y[b] for a, b in edges) and all(  # valid: y_a ≠ y_b для всех рёбер и ...
        0 <= y[i] < num_colors for i in range(n_nodes)  # ... 0 ≤ y_i < num_colors для всех вершин
    )
    return {  # словарь результата
        "coloring": y,  # найденная раскраска
        "valid": bool(valid),  # верность раскраски
        "energy": int(cur),  # энергия
        "sweeps": int(sweeps),  # число проходов
        "verified": bool(valid) == (cur == 0),  # согласованность: valid ⇔ (энергия == 0)
        "givens_respected": all(y[i] == g for i, g in givens.items()),  # фиксированные цвета сохранены: y_i == g для всех givens
    }


# ------------------- точная индукция формул и майнер теорем ------------------


def _exact_poly_fit(  # точная интерполяция многочлена с целыми коэффициентами
    xs: Sequence[int], ys: Sequence[int], max_degree: int = 5  # xs — аргументы, ys — значения, max_degree — предел степени
) -> Optional[tuple[int, ...]]:  # возвращает коэффициенты (c0..cd) или None
    """Точная рациональная интерполяция многочлена: возвращаются целые коэффициенты
    (c0..cd) единственного интерполянта степени ≤ max_degree через точки, либо None.
    Вся арифметика — над Fraction, плавающей точки нет. Система k×k Вандермонда
    решается исключением Гаусса; полученный многочлен проверяется на всех точках.
    """
    pts = sorted({(int(x), int(y)) for x, y in zip(xs, ys)})  # точки (x, y) как целые, без повторов, по возрастанию
    if len({x for x, _ in pts}) < 2:  # менее двух различных x — интерполяция невозможна
        return None  # нет решения — менее двух различных x
    for degree in range(1, min(max_degree, len(pts) - 1) + 1):  # степень d = 1..min(max_degree, n − 1)
        k = degree + 1  # k = d + 1 — число неизвестных и число точек выборки
        sample = pts[:k]  # первые k точек
        # точное исключение Гаусса в системе Вандермонда
        a = [[Fraction(x) ** i for i in range(k)] + [Fraction(y)] for x, y in sample]  # расширенная матрица k×(k+1): строка [x^0, .., x^{k−1} | y]
        n = k  # n = k — размер системы
        for col in range(n):  # по столбцам
            piv = next((r for r in range(col, n) if a[r][col] != 0), None)  # ведущая строка: первая строка r ≥ col с ненулевым элементом
            if piv is None:  # ведущего нет — матрица вырождена
                break  # выход из цикла (ветка else не выполнится)
            a[col], a[piv] = a[piv], a[col]  # поменять строки col и piv
            pv = a[col][col]  # ведущий элемент pv = a[col][col]
            a[col] = [v / pv for v in a[col]]  # нормировать строку: a[col] = [v / pv]
            for r in range(n):  # по всем строкам
                if r != col and a[r][col] != 0:  # прочие строки с ненулевым элементом столбца
                    f = a[r][col]  # множитель f = a[r][col]
                    a[r] = [vr - f * vc for vr, vc in zip(a[r], a[col])]  # a[r] = [vr − f·vc] — вычесть f·(строка col)
        else:  # else цикла: исключение доведено до конца
            coeffs = [row[n] for row in a]  # решения — последний столбец
            if any(c.denominator != 1 for c in coeffs):  # дробный коэффициент — целой формулы нет
                continue  # нецелые коэффициенты: компактной целой формулы нет
            icoeffs = tuple(int(c) for c in coeffs)  # целые коэффициенты
            # проверка на ВСЕХ точках: Σ_i c_i·x^i == y
            if all(sum(ci * x**i for i, ci in enumerate(icoeffs)) == y for x, y in pts):  # если p(x) = Σ c_i·x^i == y на всех точках
                return icoeffs  # вернуть коэффициенты (c0..cd)
    return None  # не найдено


def _monomial_exponents(arity: int, max_degree: int) -> tuple[tuple[int, ...], ...]:  # все наборы степеней суммарной степени ≤ max_degree
    """Все векторы степеней длины arity с суммой ≤ max_degree, сначала младшие."""
    if arity < 1 or max_degree < 0:  # arity ≥ 1 и max_degree ≥ 0, иначе ошибка
        raise ValueError("arity must be positive and max_degree nonnegative")  # недопустимые параметры
    return tuple(  # вернуть кортеж
        sorted(  # сортировка
            (  # генератор наборов
                powers  # powers — набор степеней
                for powers in itertools.product(range(max_degree + 1), repeat=arity)  # все наборы из 0..max_degree длины arity
                if sum(powers) <= max_degree  # с суммой степеней ≤ max_degree
            ),
            key=lambda powers: (sum(powers), powers),  # ключ сортировки: (сумма степеней, сам набор)
        )
    )


def _exact_multivariate_poly_fit(  # найти целочисленный многочлен от arity переменных по всем строкам
    inputs: Sequence[Sequence[int]],  # inputs — строки аргументов
    targets: Sequence[int],  # targets — целевые значения
    arity: int,  # arity — число переменных
    max_degree: int = 4,  # max_degree — предел суммарной степени
) -> Optional[tuple[tuple[int, tuple[int, ...]], ...]]:  # возвращает кортеж (коэффициент, степени) одночленов или None
    """Найти целочисленный многочлен минимальной суммарной степени, проходящий через все строки.

    Степени перебираются от 0 до max_degree; для каждой строится переопределённая система
    и решается точным методом Гаусса — Жордана над Fraction. Результат возвращается только
    при полном столбцовом ранге, совместности системы и целых коэффициентах — без
    плавающей точки и регуляризации. Проверка на отложенных данных и на всей конечной
    области остаётся вызывающему.
    """

    if len(inputs) != len(targets) or not inputs or arity < 1:  # длины не совпадают, нет строк или arity < 1 — None
        return None  # нет решения
    rows = [tuple(int(v) for v in row) for row in inputs]  # rows — целые кортежи аргументов
    if any(len(row) != arity for row in rows):  # строка не длины arity — None
        return None  # нет решения
    ys = [int(v) for v in targets]  # ys — целые цели
    for degree in range(max_degree + 1):  # по суммарной степени degree = 0..max_degree
        monomials = _monomial_exponents(arity, degree)  # мономы этой степени
        ncols = len(monomials)  # ncols — число мономов
        if len(rows) < ncols:  # строк меньше мономов — степень пропускается
            continue  # следующая степень
        matrix = []  # расширенная матрица
        for row, y in zip(rows, ys):  # по строкам и целям
            features = [  # признаки строки: значение каждого монома
                Fraction(math.prod(x**power for x, power in zip(row, powers)))  # Π_j x_j^{p_j} как дробь
                for powers in monomials  # по наборам степеней
            ]  # конец списка признаков
            matrix.append(features + [Fraction(y)])  # добавить строку [признаки | y]

        pivot_row = 0  # номер текущей ведущей строки
        pivot_rows: list[int] = []  # номера ведущих строк по столбцам
        full_rank = True  # флаг полного ранга
        for col in range(ncols):  # по столбцам
            pivot = next(  # ведущая строка: первая с r ≥ pivot_row и ненулевым элементом
                (r for r in range(pivot_row, len(matrix)) if matrix[r][col] != 0), None  # условие: matrix[r][col] != 0
            )
            if pivot is None:  # ведущего нет
                full_rank = False  # ранг неполный
                break  # выход из цикла
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]  # поменять строки местами
            scale = matrix[pivot_row][col]  # ведущий элемент scale
            matrix[pivot_row] = [v / scale for v in matrix[pivot_row]]  # нормировать ведущую строку: v / scale
            for r in range(len(matrix)):  # по всем строкам
                if r == pivot_row:  # ведущую строку не трогаем
                    continue  # следующая строка
                factor = matrix[r][col]  # множитель factor = matrix[r][col]
                if factor:  # только для ненулевого множителя
                    matrix[r] = [  # вычесть ведущую строку: matrix[r] = [a − factor·b]
                        a - factor * b for a, b in zip(matrix[r], matrix[pivot_row])  # a − factor·b по парам элементов
                    ]  # конец списка
            pivot_rows.append(pivot_row)  # запомнить ведущую строку столбца
            pivot_row += 1  # следующая ведущая строка: pivot_row + 1
        if not full_rank:  # ранг неполный
            continue  # следующая степень
        if any(  # несовместность: строка [0..0 | b] с b ≠ 0
            all(row[c] == 0 for c in range(ncols)) and row[-1] != 0 for row in matrix  # все коэффициенты нулевые, а правая часть ненулевая
        ):  # конец условия
            continue  # система несовместна — следующая степень
        coefficients = [matrix[pivot_rows[col]][-1] for col in range(ncols)]  # коэффициенты — правые части ведущих строк
        if any(value.denominator != 1 for value in coefficients):  # есть нецелый коэффициент
            continue  # следующая степень
        terms = tuple(  # одночлены с ненулевым коэффициентом
            (int(coefficient), powers)  # пара (целый коэффициент, степени)
            for coefficient, powers in zip(coefficients, monomials)  # по парам (коэффициент, степени)
            if coefficient  # только ненулевые коэффициенты
        )  # конец кортежа

        # независимая проверка каждой строки вне пути исключения
        def evaluate(row: Sequence[int]) -> int:  # значение многочлена на строке аргументов
            return sum(  # сумма по одночленам
                c * math.prod(int(x) ** p for x, p in zip(row, powers))  # c·Π_j x_j^{p_j} — вклад одночлена
                for c, powers in terms  # по одночленам (c, powers)
            )  # конец суммы

        if all(evaluate(row) == y for row, y in zip(rows, ys)):  # все строки дают верные значения
            return terms  # вернуть одночлены
    return None  # не найдено


def _multivariate_poly_expr(  # обёртка одночленов в выражение poly_nd
    terms: tuple[tuple[int, tuple[int, ...]], ...],  # terms — одночлены (коэффициент, степени)
) -> Expr:  # возвращает Expr
    return Expr(  # конструктор выражения
        "poly_nd",  # имя операции poly_nd
        (),  # вложенных подвыражений нет
        tuple((int(c), tuple(int(p) for p in powers)) for c, powers in terms),  # атрибуты: кортеж (целый c, целые степени)
    )  # конец конструктора


def _poly_expr(coeffs: tuple[int, ...]) -> Expr:  # обёртка коэффициентов в многочлен от переменной 0
    return Expr("poly", (Expr.var(0),), coeffs)  # Expr('poly', [переменная 0], c0..cd)


def _eval_poly(coeffs: tuple[int, ...], x: int) -> int:  # значение многочлена в точке x по схеме Горнера
    acc = 0  # аккумулятор acc = 0
    for ci in reversed(coeffs):  # от старшего коэффициента к младшему
        acc = acc * x + ci  # acc = acc·x + c_i — шаг Горнера
    return acc  # значение p(x) = Σ c_i·x^i


@dataclass  # класс данных с авто-конструктором
class Theorem:  # теорема, проверенная на конечной области
    name: str  # name — имя теоремы
    statement: str  # statement — текст утверждения
    domain_points: int  # domain_points — число проверенных точек
    verified: bool = True  # verified — проверена (по умолчанию да)


class TheoremMiner:
    """Майнер теорем: находит тождества, верные на всей поддерживаемой области эксперта.

    Каждое равенство проверяется перебором только в поддерживаемых точках; результат —
    утверждение о конечной области. Для унарных тождеств итерации дополнительно
    требуется замкнутость объявленной области относительно выходов эксперта.
    """

    def mine(  # найти новые тождества для экспертов модели
        self, model: "OptimalMathProcessor", max_experts: int = 24  # model — процессор с экспертами, max_experts — предел числа экспертов каждой арности
    ) -> list[Theorem]:  # возвращает список новых теорем
        found: list[Theorem] = []  # найденные теоремы
        known = {t.name for t in model.theorems}  # известные имена теорем
        bin_e = [e for e in model.experts.values() if e.program.arity == 2][  # бинарные эксперты (арность 2), первые max_experts
            :max_experts  # срез [:max_experts]
        ]
        un_e = [e for e in model.experts.values() if e.program.arity == 1][:max_experts]  # унарные эксперты (арность 1), первые max_experts

        for e in bin_e:  # по бинарным экспертам
            dx, dy = e.domains[0], e.domains[1]  # dx, dy — объявленные области первого и второго аргумента
            declared_pairs = [(x, y) for x in dx for y in dy if e.supports((x, y))]  # объявленные поддерживаемые пары (x, y)
            comm_pairs = [(x, y) for x, y in declared_pairs if e.supports((y, x))]  # из них пары с поддерживаемым обратным порядком (y, x)
            # коммутативность не приписывается малому симметричному пересечению:
            # у каждой объявленной пары должна быть поддержана переставленная
            if comm_pairs and len(comm_pairs) == len(declared_pairs):  # коммутативность проверяется, только если переставленная пара поддержана у всех объявленных пар
                name = f"commutativity({e.name})"  # имя теоремы
                if name not in known:  # теорема ещё не известна
                    try:  # ошибки вычисления считаются неподтверждением
                        ok = all(  # ok: f(x,y) == f(y,x) для всех пар
                            e.program.eval((int(x), int(y)))  # вычислить f(x, y)
                            == e.program.eval((int(y), int(x)))  # сравнение с f(y, x)
                            for x, y in comm_pairs  # по парам
                        )  # конец all
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления — тождество не подтверждено
                        ok = False  # ok = False — тождество не подтверждено
                    if ok:  # тождество верно
                        th = Theorem(  # новая теорема
                            name,  # имя
                            f"for all declared pairs with both orders supported: {e.name}(x,y) = {e.name}(y,x)",  # утверждение: f(x,y) = f(y,x) на всех поддержанных парах
                            len(comm_pairs),  # число проверенных точек
                        )  # конец Theorem
                        model.theorems.append(th)  # добавить в теоремы модели
                        found.append(th)  # добавить в найденные
                        known.add(name)  # запомнить имя

            self_pairs = [x for x in dx if x in dy and e.supports((x, x))]  # x из dx, лежащие в dy, с поддерживаемой парой (x, x)
            name = f"self-absorption({e.name})"  # имя теоремы
            if self_pairs and len(self_pairs) == len(dx) and name not in known:  # все x из dx дают пару (x,x) и теорема новая
                try:  # ошибки вычисления считаются неподтверждением
                    ok = all(  # ok: f(x,x) == x для всех таких x
                        e.program.eval((int(x), int(x))) == int(x) for x in self_pairs  # f(x, x) сравнивается с x; по x
                    )  # конец all
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                    ok = False  # ok = False — тождество не подтверждено
                if ok:  # тождество верно
                    th = Theorem(  # новая теорема
                        name,  # имя
                        f"for all x in the declared overlap with (x,x) supported: {e.name}(x,x) = x",  # утверждение: f(x,x) = x
                        len(self_pairs),  # число проверенных точек
                    )  # конец Theorem
                    model.theorems.append(th)  # добавить в теоремы модели
                    found.append(th)  # добавить в найденные
                    known.add(name)  # запомнить имя

            zero_pairs = [x for x in dx if 0 in dy and e.supports((x, 0))]  # x из dx, для которых поддержана пара (x, 0)
            name = f"annihilator-zero({e.name})"  # имя теоремы
            if zero_pairs and len(zero_pairs) == len(dx) and name not in known:  # все x из dx дают пару (x,0) и теорема новая
                try:  # ошибки вычисления считаются неподтверждением
                    ok = all(e.program.eval((int(x), 0)) == int(x) for x in zero_pairs)  # ok: f(x,0) == x для всех таких x
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                    ok = False  # ok = False — тождество не подтверждено
                if ok:  # тождество верно
                    th = Theorem(  # новая теорема
                        name,  # имя
                        f"for all declared x with (x,0) supported: {e.name}(x,0) = x",  # утверждение: f(x,0) = x
                        len(zero_pairs),  # число проверенных точек
                    )  # конец Theorem
                    model.theorems.append(th)  # добавить в теоремы модели
                    found.append(th)  # добавить в найденные
                    known.add(name)  # запомнить имя
            # по унарным экспертам
        for e in un_e:  # по унарным экспертам
            dom = tuple(int(x) for x in e.domains[0])  # dom — целочисленная область аргумента
            image: dict[int, int] = {}  # образ: image[x] = f(x)
            closed = True  # флаг замкнутости области относительно f
            try:  # ошибки вычисления означают незамкнутость
                for x in dom:  # по x из dom
                    if not e.supports((x,)):  # (x,) не поддерживается — не замкнуто
                        closed = False  # closed = False — область не замкнута
                        break  # выход из цикла
                    y = int(e.program.eval((x,)))  # y = f(x) — значение эксперта
                    if not e.supports((y,)):  # (y,) не поддерживается — не замкнуто
                        closed = False  # closed = False — область не замкнута
                        break  # выход из цикла
                    image[x] = y  # образ: image[x] = y
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления — область не замкнута
                closed = False  # closed = False — область не замкнута
            # область замкнута
            if closed:  # если область замкнута
                name = f"involutivity({e.name})"  # имя теоремы
                if name not in known:  # теорема ещё не известна
                    try:  # ошибки вычисления считаются неподтверждением
                        ok = all(int(e.program.eval((image[x],))) == x for x in dom)  # ok: f(f(x)) == x для всех x
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления — тождество не подтверждено
                        ok = False  # ok = False — тождество не подтверждено
                    if ok:  # тождество верно
                        th = Theorem(  # новая теорема
                            name,  # имя
                            f"for all x in the declared domain: {e.name}({e.name}(x)) = x",  # утверждение: f(f(x)) = x
                            len(dom),  # число проверенных точек
                        )  # конец Theorem
                        model.theorems.append(th)  # добавить в теоремы модели
                        found.append(th)  # добавить в найденные
                        known.add(name)  # запомнить имя
                # имя теоремы
                name = f"idempotence({e.name})"  # имя теоремы
                if name not in known:  # теорема ещё не известна
                    try:  # ошибки вычисления считаются неподтверждением
                        ok = all(  # ok: f(f(x)) == f(x) для всех x
                            int(e.program.eval((image[x],))) == image[x] for x in dom  # f(f(x)) == f(x) для всех x из dom
                        )  # конец all
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления — тождество не подтверждено
                        ok = False  # ok = False — тождество не подтверждено
                    if ok:  # тождество верно
                        th = Theorem(  # новая теорема
                            name,  # имя
                            f"for all x in the declared domain: {e.name}({e.name}(x)) = {e.name}(x)",  # утверждение: f(f(x)) = f(x)
                            len(dom),  # число проверенных точек
                        )  # конец Theorem
                        model.theorems.append(th)  # добавить в теоремы модели
                        found.append(th)  # добавить в найденные
                        known.add(name)  # запомнить имя
            # если 0 лежит в dom
            if 0 in dom:  # 0 лежит в dom
                name = f"zero-fixpoint({e.name})"  # имя теоремы
                if name not in known:  # теорема ещё не известна
                    try:  # ошибки вычисления считаются неподтверждением
                        ok = int(e.program.eval((0,))) == 0  # ok: f(0) == 0 — ноль неподвижен
                    except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления — тождество не подтверждено
                        ok = False  # ok = False — тождество не подтверждено
                    if ok:  # тождество верно
                        th = Theorem(  # новая теорема
                            name, f"at the supported point x=0: {e.name}(0) = 0", 1  # имя, утверждение: f(0) = 0, одна точка
                        )  # конец Theorem
                        model.theorems.append(th)  # добавить в теоремы модели
                        found.append(th)  # добавить в найденные
                        known.add(name)  # запомнить имя
        return found  # вернуть найденные теоремы

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'solve_coloring', 'Theorem', 'TheoremMiner',  # имя интерфейса
]  # конец перечня
