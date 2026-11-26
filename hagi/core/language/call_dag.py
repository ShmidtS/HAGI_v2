"""Префиксный код, обобщённый граф вызовов, АНФ и кусочные выражения.

CallDagExpr — неизменяемое дерево выражения произвольной арности. Длина его кода
bits(e): каждый узел начинается с 6-битного кода операции, числа записываются
гамма-кодом Элиаса (γ(m) = 2·⌊log2 m⌋ + 1 бит, m ≥ 1), знак — зигзагом
(z(n) = 2n при n ≥ 0, иначе −2n − 1). Прочий узел: bits = 6 + Σ bits(аргументов).

Вычисление eval(x):
  call   — вызвать эксперта: f(args), только внутри подтверждённой области;
  anf    — f = ⊕_{(s, c): s ⊆ m} c, где m — битовая маска входов, x_i — бит номер
           n − 1 − i;
  свёртки по диапазону — тело вычисляется при y = start, start+1, …, stop−1,
           span = max(0, stop − start), общий счётчик итераций ≤ MAX_RANGE_EVAL;
  if_cond — arg1, если arg0 ≠ 0, иначе arg2;
  poly   — схема Горнера: acc = acc·x + c_i по коэффициентам с конца;
  poly_nd — Σ_j coef_j · Π_i x_i^p_ij.

CallDagProgram — арность плюс либо выражение, либо 4-битная маска таблицы истинности
двух двоичных входов: результат = (mask >> (2·x0 + x1)) & 1.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов
import math  # модули стандартной библиотеки: аргументы, синтаксическое дерево, копирование, куча, итераторы, JSON, математика, ОС, случайность
from dataclasses import dataclass  # декоратор неизменяемой записи данных и описатель поля
from typing import TYPE_CHECKING, Any, Optional, Sequence  # аннотации типов
import numpy as np  # числовые массивы
from hagi.core.language.expr_dsl import Expr  # имена из модуля «expr_dsl» (прямой импорт вместо обращения через псевдоним пакета)
from hagi.core.memory.aliases import OPS  # имена из модуля «aliases» (прямой импорт вместо обращения через псевдоним пакета)
if TYPE_CHECKING:  # только для проверки типов аннотаций (анализатор видит имя, импорта во время выполнения нет — нет цикла)
    from hagi.core.processor.stored_growth import StoredExpert  # класс эксперта из части «growth»: имя нужно лишь в строковых аннотациях

# ==============================================================================
# 1. Точная длина префиксного кода, обобщённый граф вызовов, АНФ, кусочные выражения
# ==============================================================================

TOKEN_BITS = 6  # ширина кода операции: 6 бит
VAR_NAMES = ("x", "y", "z", "u", "v", "w")  # имена переменных для печати
DYADIC_LEVELS = np.array([(c + 1) / 128.0 for c in range(255)], dtype=np.float64)  # уровни (c + 1)/128, c = 0..254
MAX_RANGE_EVAL = 100_000  # предел общего числа итераций свёрток за одно вычисление
_RANGE_FOLD_OPS = {"all_range", "any_range", "sum_range", "prod_range"}  # операции свёртки по диапазону


def _expr_contains_range(expr: "CallDagExpr") -> bool:
    """True, если в дереве expr есть хоть одна свёртка по диапазону."""
    return expr.op in _RANGE_FOLD_OPS or any(_expr_contains_range(a) for a in expr.args)  # корень — свёртка или она есть в каком-то аргументе


def gamma_bits(n: int) -> int:  # длина гамма-кода числа n
    """Длина гамма-кода Элиаса: 2·⌊log2 n⌋ + 1 бит для целого n ≥ 1."""
    n = int(n)  # привести к целому
    if n < 1:  # число не положительно
        raise ValueError("gamma_bits expects positive integer")  # ошибка
    return 2 * (n.bit_length() - 1) + 1  # длина кода 2·⌊log2 n⌋ + 1


def zigzag(n: int) -> int:  # зигзаг-код числа n
    """Зигзаг: z(n) = 2n при n ≥ 0, z(n) = −2n − 1 при n < 0."""
    n = int(n)  # привести к целому
    return 2 * n if n >= 0 else -2 * n - 1  # зигзаг z(n)


@dataclass(frozen=True)  # неизменяемая запись данных с автоматическими методами
class CallDagExpr:
    """Неизменяемое дерево выражения произвольной арности: вызовы экспертов, АНФ, кусочные выражения.

        op: имя операции; args: кортеж аргументов; value: индекс переменной,
        константа, имя вызываемого эксперта, коэффициенты многочлена или пара
        (число переменных, коэффициенты АНФ)."""
    op: str  # имя операции
    args: tuple["CallDagExpr", ...] = ()  # аргументы, по умолчанию пусто
    value: Optional[Any] = None  # полезная нагрузка узла, по умолчанию None

    @staticmethod  # статический метод без экземпляра
    def var(i: int) -> "CallDagExpr":  # создать узел-переменную с индексом i
        return CallDagExpr("var", (), int(i))  # CallDagExpr("var", без аргументов, value = i)

    @staticmethod  # статический метод
    def const(v: int) -> "CallDagExpr":  # создать узел-константу со значением v
        return CallDagExpr("const", (), int(v))  # CallDagExpr("const", без аргументов, value = v)

    @staticmethod  # статический метод
    def from_h5(e: Expr) -> "CallDagExpr":  # преобразовать выражение предыдущей версии
        return CallDagExpr(e.op, tuple(CallDagExpr.from_h5(a) for a in e.args), e.value)  # та же операция и значение, аргументы преобразуются рекурсивно

    def eval(  # вычислить значение выражения
        self,  # сам вычисляемый объект
        xs: Sequence[int],  # значения входных переменных
        env: Optional[dict[str, "StoredExpert"]] = None,  # доступные для вызова эксперты по имени
        _budget: Optional[list[int]] = None,  # общий счётчик итераций свёрток (список из одного числа)
        _fold_depth: int = 0,  # текущая вложенность свёрток
    ) -> int:  # возврат: целое значение
        """Вычислить выражение на входах xs.

            Переменная: xs[value]; константа: value. Вызов: аргументы вычисляются,
            проверяется supports, затем вычисляется program вызываемого эксперта с
            тем же счётчиком итераций. АНФ: f = ⊕ c по мономам (s, c) с s ⊆ m,
            где m = Σ_i x_i·2^(n − 1 − i). Свёртка: тело вычисляется при
            y = start..stop − 1; all_range — 1, если все значения истинны;
            any_range — 1, если хоть одно истинно; sum_range — Σ значений;
            prod_range — Π значений. Накопитель свёртки не шире 4096 бит."""
        # Изменяемый счётчик общий для всех символьных вызовов одного вычисления, чтобы
        # вызов эксперта не мог незаметно умножить предел числа итераций.
        if _budget is None:  # счётчик не передан
            _budget = [0]  # создать счётчик с нулём
        if self.op == "var":  # переменная
            return int(xs[int(self.value)])  # значение переменной xs[value]
        if self.op == "const":  # константа
            return int(self.value)  # значение value
        if self.op == "call":  # вызов эксперта
            if env is None or self.value not in env:  # среды нет или эксперт не найден
                raise ValueError(f"missing called expert: {self.value}")  # ошибка
            vals = tuple(a.eval(xs, env, _budget, _fold_depth) for a in self.args)  # значения аргументов
            callee = env[self.value]  # вызываемый эксперт
            if not callee.supports(vals):  # аргументы вне подтверждённой области
                raise ValueError(  # ошибка
                    f"call to {self.value} outside certified support: {vals}"  # текст ошибки
                )
            return callee.program.eval(vals, env, _budget, _fold_depth)  # program(vals) с тем же счётчиком
        if self.op == "anf":  # алгебраическая нормальная форма
            n_vars, coeffs = self.value  # число переменных и список мономов (маска, коэффициент)
            mask_x = 0  # маска входов, изначально 0
            for i in range(n_vars):  # по переменным
                b = int(xs[i])  # значение x_i
                if b not in (0, 1):  # АНФ определена только на битах
                    raise ValueError("anf expects boolean inputs")  # ошибка
                if b:  # бит равен единице: x_i = 1
                    mask_x |= 1 << (n_vars - 1 - i)  # установить бит номер n − 1 − i
            acc = 0  # накопитель, изначально 0
            for s, c in coeffs:  # по мономам
                if (s & mask_x) == s:  # маска монома s ⊆ m
                    acc ^= c  # исключающее или acc = acc ⊕ c
            return int(acc)  # значение f
        if self.op in _RANGE_FOLD_OPS:  # свёртка по диапазону
            if len(self.args) != 3:  # нужны ровно start, stop и тело
                raise ValueError(f"{self.op} expects start, stop, body")  # ошибка
            # Статически вложенные свёртки отвергаются здесь, вложенные через вызовы —
            # передаваемым флагом глубины.
            if _fold_depth >= 1 or any(_expr_contains_range(a) for a in self.args):  # уже внутри свёртки или свёртка в аргументах
                raise ValueError(  # ошибка
                    "nested range folds (including calls) are not supported"  # текст ошибки
                )
            start = int(self.args[0].eval(xs, env, _budget, _fold_depth))  # начало диапазона
            stop = int(self.args[1].eval(xs, env, _budget, _fold_depth))  # конец диапазона (не включается)
            span = max(0, stop - start)  # число итераций span = max(0, stop − start)
            if span > MAX_RANGE_EVAL or _budget[0] + span > MAX_RANGE_EVAL:  # предел итераций превышен
                raise ValueError(  # ошибка
                    f"cumulative range-fold budget exceeds {MAX_RANGE_EVAL} iterations"  # текст ошибки
                )
            body = self.args[2]  # тело свёртки
            if self.op == "all_range":  # «для всех»
                for k in range(start, stop):  # по k = start..stop − 1
                    _budget[0] += 1  # счётчик итераций: +1
                    if not body.eval(  # тело ложно при y = k
                        tuple(xs) + (int(k),), env, _budget, _fold_depth + 1  # тело(xs, y = k) на глубине +1
                    ):
                        return 0  # найден ложный элемент — результат 0
                return 1  # все истинны — результат 1
            if self.op == "any_range":  # «существует»
                for k in range(start, stop):  # по k = start..stop − 1
                    _budget[0] += 1  # счётчик итераций: +1
                    if body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1):  # тело(xs, y = k) истинно
                        return 1  # найден истинный элемент — результат 1
                return 0  # все ложны — результат 0
            if self.op == "sum_range":  # сумма
                acc = 0  # накопитель, изначально 0
                for k in range(start, stop):  # по k = start..stop − 1
                    _budget[0] += 1  # счётчик итераций: +1
                    acc += int(  # acc += тело(xs, y = k)
                        body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1)  # значение тела на глубине +1
                    )
                    if abs(acc).bit_length() > 4096:  # накопитель шире 4096 бит
                        raise OverflowError(  # ошибка переполнения
                            "range-fold accumulator exceeds 4096-bit cap"  # текст ошибки
                        )
                return acc  # сумма значений
            acc = 1  # prod_range: накопитель, изначально 1
            for k in range(start, stop):  # по k = start..stop − 1
                _budget[0] += 1  # счётчик итераций: +1
                acc *= int(  # acc *= тело(xs, y = k)
                    body.eval(tuple(xs) + (int(k),), env, _budget, _fold_depth + 1)  # значение тела на глубине +1
                )
                if abs(acc).bit_length() > 4096:  # накопитель шире 4096 бит
                    raise OverflowError("range-fold product exceeds 4096-bit cap")  # ошибка переполнения
            return acc  # произведение значений
        if self.op == "if_cond":  # условное выражение
            cond_val = self.args[0].eval(xs, env, _budget, _fold_depth)  # значение условия
            return (  # ветка по условию
                self.args[1].eval(xs, env, _budget, _fold_depth)  # arg1, если cond_val ≠ 0
                if cond_val  # условие
                else self.args[2].eval(xs, env, _budget, _fold_depth)  # иначе arg2
            )
        vals = [a.eval(xs, env, _budget, _fold_depth) for a in self.args]  # значения всех аргументов
        if self.op == "poly":  # многочлен одной переменной
            acc = 0  # накопитель, изначально 0
            for ci in reversed(list(self.value or ())):  # коэффициенты с последнего
                acc = acc * vals[0] + int(ci)  # схема Горнера: acc = acc·x + c_i
            return int(acc)  # значение многочлена
        if self.op == "poly_nd":  # многочлен многих переменных
            total = 0  # сумма, изначально 0
            for coef, powers in self.value or ():  # по слагаемым
                term = int(coef)  # слагаемое начинается с коэффициента
                for x_val, p in zip(xs, powers):  # по парам (x_i, степень p_i)
                    term *= int(x_val) ** int(p)  # множитель term *= x_i^p_i
                total += term  # total += слагаемое
            return int(total)  # сумма слагаемых Σ_j coef_j·Π_i x_i^p_ij
        if self.op == "absdiff":  # разность по модулю
            return abs(vals[0] - vals[1])  # разность по модулю |v0 − v1|
        if self.op == "fma":  # умножение-сложение
            return vals[0] * vals[1] + vals[2]  # произведение плюс слагаемое v0·v1 + v2
        if self.op == "clamp":  # усечение
            return min(max(vals[0], vals[1]), vals[2])  # усечение min(max(v0, v1), v2)
        if self.op == "median3":  # медиана трёх
            return sorted(vals)[1]  # средний элемент отсортированного списка
        if self.op == "clamp3":  # усечение в [−3, 3]
            return min(max(vals[0], -3), 3)  # усечение в отрезок min(max(v0, −3), 3)
        if self.op == "mul3":  # произведение трёх
            return vals[0] * vals[1] * vals[2]  # произведение трёх v0·v1·v2
        if self.op == "isqrt":  # целочисленный квадратный корень
            value = int(vals[0])  # подкоренное число
            if value < 0:  # отрицательное
                raise ValueError("isqrt is undefined for negative integers")  # ошибка
            return math.isqrt(value)  # целый корень ⌊√value⌋
        if self.op == "reified":  # зарегистрированная функция над сырыми входами
            from hagi.core.synthesis.fast_mdl import REIFIED_OPERATIONS  # словарь зарегистрированных функций (импорт здесь: fast_mdl импортирует этот модуль — импорт наверху дал бы цикл)
            rfn = REIFIED_OPERATIONS.get(str(self.value))  # функция по имени или None
            if rfn is None:  # не зарегистрирована
                raise ValueError(f"reified op not registered: {self.op}")  # ошибка
            # Лист-макрос над СЫРЫМИ входами (args пуста на уровне листа;
            # композиция получается обёрткой в арифметические операции).
            return int(rfn(*[int(v) for v in xs]))  # значение функции rfn(x_0, …, x_{n−1})
        fn = OPS.get(self.op)  # прочая операция из словаря предыдущей версии
        if fn is None:  # операция не найдена
            raise ValueError(f"unsupported op: {self.op}")  # ошибка
        return int(fn(*vals))  # значение функции fn(v_0, …, v_{k−1})

    @property  # свойство-атрибут
    def cost(self) -> int:  # стоимость выражения в узлах (с запоминанием)
        """Число узлов: лист = 1; anf, poly, poly_nd = 1 + число коэффициентов (слагаемых); иначе 1 + Σ cost(аргументов)."""
        c = self.__dict__.get("_cost_memo")  # запомненное значение или None
        if c is None:  # ещё не вычислено
            if self.op in ("var", "const"):  # лист
                c = 1  # стоимость 1
            elif self.op == "anf":  # АНФ
                _, coeffs = self.value  # список мономов
                c = 1 + len(coeffs)  # 1 + число мономов
            elif self.op == "poly":  # многочлен одной переменной
                c = 1 + len(self.value or ())  # 1 + число коэффициентов
            elif self.op == "poly_nd":  # многочлен многих переменных
                c = 1 + len(self.value or ())  # 1 + число слагаемых
            else:  # прочий узел
                c = 1 + sum(a.cost for a in self.args)  # 1 + Σ cost(аргументов)
            object.__setattr__(self, "_cost_memo", c)  # запомнить (обход запрета на изменение)
        return c  # стоимость

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:  # длина кода в битах
        """Длина префиксного кода: 6 бит на узел + гамма-коды полезной нагрузки + Σ bits(аргументов).

            var: 6 + γ(индекс + 1); const: 6 + γ(z(value) + 1);
            call: 6 + γ(номер эксперта + 1) + Σ bits(args);
            anf: 6 + γ(n_vars + 1) + γ(число мономов + 1) + Σ γ(маска + 1);
            poly: 6 + γ(число коэффициентов + 1) + Σ γ(z(c) + 1) + Σ bits(args);
            poly_nd: 6 + γ(число слагаемых + 1) + Σ [γ(z(coef) + 1) + γ(число степеней + 1) + Σ γ(p + 1)].
            expert_index: имя эксперта → номер (по умолчанию 0)."""
        if self.op == "var":  # переменная
            return TOKEN_BITS + gamma_bits(int(self.value or 0) + 1)  # 6 + γ(индекс + 1)
        if self.op == "const":  # константа
            return TOKEN_BITS + gamma_bits(zigzag(int(self.value)) + 1)  # константа 6 + γ(z(value) + 1)
        if self.op == "call":  # вызов
            idx = expert_index.get(str(self.value), 0) if expert_index else 0  # номер эксперта (0, если нет)
            return (  # длина кода вызова
                TOKEN_BITS  # 6 бит
                + gamma_bits(idx + 1)  # + γ(номер + 1)
                + sum(a.bits(expert_index) for a in self.args)  # + Σ bits(аргументов)
            )
        if self.op == "anf":  # АНФ
            n_vars, coeffs = self.value  # число переменных и мономы
            payload = gamma_bits(n_vars + 1) + gamma_bits(len(coeffs) + 1)  # γ(n_vars + 1) + γ(число мономов + 1)
            payload += sum(gamma_bits(s + 1) for s, _ in coeffs)  # += Σ γ(маска + 1)
            return TOKEN_BITS + payload  # 6 + нагрузка
        if self.op == "poly":  # многочлен одной переменной
            coeffs = tuple(int(v) for v in (self.value or ()))  # кортеж коэффициентов
            payload = gamma_bits(len(coeffs) + 1) + sum(  # γ(число коэффициентов + 1) + Σ γ(z(c) + 1)
                gamma_bits(zigzag(v) + 1) for v in coeffs  # по коэффициентам
            )
            return TOKEN_BITS + payload + sum(a.bits(expert_index) for a in self.args)  # 6 + нагрузка + Σ bits(аргументов)
        if self.op == "poly_nd":  # многочлен многих переменных
            terms = tuple(self.value or ())  # кортеж слагаемых
            payload = gamma_bits(len(terms) + 1)  # γ(число слагаемых + 1)
            for c, powers in terms:  # по слагаемым
                payload += gamma_bits(zigzag(int(c)) + 1) + gamma_bits(len(powers) + 1)  # += γ(z(coef) + 1) + γ(число степеней + 1)
                payload += sum(gamma_bits(int(p) + 1) for p in powers)  # сумма по степеням += Σ γ(p + 1)
            return TOKEN_BITS + payload  # 6 + нагрузка
        return TOKEN_BITS + sum(a.bits(expert_index) for a in self.args)  # прочий узел: 6 + Σ bits(аргументов)

    def pretty(self) -> str:  # текстовая запись выражения
        """Вернуть выражение в виде строки; переменные названы x, y, z, u, v, w, дальше xN."""
        if self.op == "var":  # переменная
            idx = int(self.value or 0)  # индекс переменной
            return VAR_NAMES[idx] if idx < len(VAR_NAMES) else f"x{idx}"  # имя из списка или xN
        if self.op == "const":  # константа
            return str(self.value)  # значение как строка
        if self.op == "call":  # вызов
            return f"{self.value}({', '.join(a.pretty() for a in self.args)})"  # имя(аргументы через запятую)
        if self.op == "anf":  # АНФ
            n_vars, coeffs = self.value  # число переменных и мономы
            terms = []  # тексты мономов
            for s, _ in coeffs:  # по мономам
                if s == 0:  # пустой моном
                    terms.append("1")  # константа 1
                else:  # непустой моном
                    vs = [  # переменные, входящие в моном
                        VAR_NAMES[i] if i < len(VAR_NAMES) else f"x{i}"  # имя переменной i
                        for i in range(n_vars)  # по переменным
                        if (s >> (n_vars - 1 - i)) & 1  # у которых бит маски s номер n − 1 − i равен 1
                    ]
                    terms.append("&".join(vs))  # моном — переменные через &
            return "anf(" + (" ^ ".join(terms) if terms else "0") + ")"  # anf(мономы через ^) или anf(0)
        if self.op in _RANGE_FOLD_OPS:  # свёртка по диапазону
            label = {  # имя операции на печати
                "all_range": "forall",  # для всех
                "any_range": "exists",  # существует
                "sum_range": "sum",  # сумма
                "prod_range": "prod",  # произведение
            }[self.op]  # взять имя по операции
            return f"{label}[y={self.args[0].pretty()}..{self.args[1].pretty()}):{self.args[2].pretty()}]"  # метка[y = start..stop):тело]
        if self.op == "if_cond":  # условное выражение
            return f"({self.args[1].pretty()} if {self.args[0].pretty()} else {self.args[2].pretty()})"  # условная запись (then if cond else else)
        if self.op == "poly":  # многочлен одной переменной
            return Expr("poly", (Expr.var(0),), self.value).pretty()  # печать через многочлен предыдущей версии от x
        if self.op == "poly_nd":  # многочлен многих переменных
            return Expr("poly_nd", (), self.value).pretty()  # печать через предыдущую версию
        if self.op == "clamp3":  # усечение в [−3, 3]
            return f"min(max({self.args[0].pretty()}, -3), 3)"  # min(max(аргумент, −3), 3)
        if self.op == "absdiff":  # разность по модулю
            return f"abs({self.args[0].pretty()} - {self.args[1].pretty()})"  # разность по модулю abs(a − b)
        if len(self.args) == 1:  # один аргумент
            return f"{self.op}({self.args[0].pretty()})"  # op(аргумент)
        if len(self.args) == 2:  # два аргумента
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"  # инфиксная запись (a op b)
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"  # op(все аргументы через запятую)

    def subtrees(self) -> list["CallDagExpr"]:  # все поддеревья
        """Список всех поддеревьев, начиная с корня."""
        out = [self]  # начать с корня
        for a in self.args:  # по аргументам
            out.extend(a.subtrees())  # добавить их поддеревья
        return out  # список поддеревьев

    def replace_subtree(self, target: "CallDagExpr", replacement: "CallDagExpr") -> "CallDagExpr":  # замена поддерева
        """Вернуть копию, где каждое вхождение target заменено на replacement."""
        if self == target:  # текущий узел — искомый
            return replacement  # заменить
        if not self.args:  # лист, не равный target
            return self  # без изменений
        return CallDagExpr(  # тот же узел с заменёнными аргументами
            self.op,  # операция
            tuple(a.replace_subtree(target, replacement) for a in self.args),  # аргументы после замены
            self.value,  # значение
        )

    def to_dict(self) -> dict[str, Any]:  # сериализация в словарь
        """Словарь с ключами op, value, args (JSON-совместимый)."""
        val = self.value  # значение узла
        if self.op == "anf" and val is not None:  # АНФ: списки вместо кортежей
            val = [int(val[0]), [[int(s), int(c)] for s, c in val[1]]]  # сериализация [n_vars, [[s, c], …]]
        return {"op": self.op, "value": val, "args": [a.to_dict() for a in self.args]}  # словарь узла

    @staticmethod  # статический метод
    def from_dict(d: dict[str, Any]) -> "CallDagExpr":  # восстановить выражение из словаря
        """Обратно к to_dict: восстановить выражение по словарю."""
        op = str(d["op"])  # имя операции
        val = d.get("value")  # значение или None
        if op == "poly" and val is not None:  # многочлен одной переменной
            val = tuple(int(v) for v in val)  # коэффициенты в кортеж целых
        elif op == "poly_nd" and val is not None:  # многочлен многих переменных
            val = tuple((int(t[0]), tuple(int(p) for p in t[1])) for t in val)  # кортеж пар (коэффициент, кортеж степеней)
        elif op == "anf" and val is not None:  # АНФ
            val = (int(val[0]), tuple((int(s), int(c)) for s, c in val[1]))  # (n_vars, кортеж (маска, коэффициент))
        return CallDagExpr(op, tuple(CallDagExpr.from_dict(a) for a in d.get("args", [])), val)  # узел с восстановленными аргументами


@dataclass(frozen=True)  # неизменяемая запись данных
class CallDagProgram:
    """Программа: арность и либо выражение, либо маска таблицы истинности двух битов.

        Результат табличной программы: (venn_mask >> (2·x0 + x1)) & 1.
        cost: 4 при маске, cost(expr) при выражении, 10^9 без выражения.
        bits: 6 + 4 при маске, bits(expr) при выражении, 10^9 без выражения."""
    arity: int  # число входов
    expr: Optional[CallDagExpr] = None  # выражение, по умолчанию None
    venn_mask: Optional[int] = None  # 4-битная маска, по умолчанию None

    @property  # свойство-атрибут
    def cost(self) -> int:  # стоимость программы в узлах
        return (  # стоимость
            4  # 4, если задана маска
            if self.venn_mask is not None  # маска есть
            else (self.expr.cost if self.expr else 10**9)  # иначе cost(expr) или 10^9
        )

    def bits(self, expert_index: Optional[dict[str, int]] = None) -> int:  # длина кода программы
        """Длина кода: 10 бит при маске, bits(expr) при выражении, 10^9 без выражения."""
        if self.venn_mask is not None:  # табличная программа
            return TOKEN_BITS + 4  # табличная программа 6 + 4 = 10
        if self.expr is None:  # выражения нет
            return 10**9  # штраф
        return self.expr.bits(expert_index)  # длина выражения bits(expr)

    def eval(  # вычислить программу
        self,  # сам вычисляемый объект
        xs: Sequence[int],  # входы
        env: Optional[dict[str, "StoredExpert"]] = None,  # вызываемые эксперты
        _budget: Optional[list[int]] = None,  # счётчик итераций свёрток
        _fold_depth: int = 0,  # глубина вложенности свёрток
    ) -> int:  # возврат: целое значение
        """Вычислить программу на входах xs: число входов должно равняться arity."""
        if _budget is None:  # счётчик не передан
            _budget = [0]  # создать счётчик с нулём
        if len(xs) != self.arity:  # число входов не совпадает
            raise ValueError(f"expected {self.arity} inputs, got {len(xs)}")  # ошибка
        if self.venn_mask is not None:  # табличная программа
            if self.arity != 2 or int(xs[0]) not in (0, 1) or int(xs[1]) not in (0, 1):  # нужны два двоичных входа
                raise ValueError("BoolVenn expects two binary inputs")  # ошибка
            return (int(self.venn_mask) >> ((int(xs[0]) << 1) | int(xs[1]))) & 1  # табличный результат (mask >> (2·x0 + x1)) & 1
        if self.expr is None:  # выражения нет
            raise RuntimeError("empty program")  # ошибка пустой программы
        return self.expr.eval(xs, env, _budget, _fold_depth)  # вычислить выражение

    def pretty(self) -> str:  # текстовая запись программы
        """Строка вида BoolVenn(mask=…) для маски, иначе текст выражения."""
        if self.venn_mask is not None:  # табличная программа
            return f"BoolVenn(mask={self.venn_mask:04b})"  # маска четырьмя двоичными цифрами
        assert self.expr is not None  # выражение обязано быть
        return self.expr.pretty()  # текст выражения

    def to_dict(self) -> dict[str, Any]:  # сериализация в словарь
        """Словарь с ключами arity, venn_mask, expr."""
        return {  # словарь
            "arity": self.arity,  # арность
            "venn_mask": self.venn_mask,  # маска
            "expr": self.expr.to_dict() if self.expr else None,  # словарь выражения или None
        }

    @staticmethod  # статический метод
    def from_dict(d: dict[str, Any]) -> "CallDagProgram":  # восстановить программу из словаря
        """Обратно к to_dict."""
        return CallDagProgram(  # программа
            int(d["arity"]),  # арность
            CallDagExpr.from_dict(d["expr"]) if d.get("expr") else None,  # выражение или None
            d.get("venn_mask"),  # маска
        )

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'TOKEN_BITS', 'VAR_NAMES', 'DYADIC_LEVELS', 'MAX_RANGE_EVAL', 'gamma_bits', 'zigzag',  # имя интерфейса
    'CallDagExpr', 'CallDagProgram',  # имя интерфейса
]  # конец перечня
