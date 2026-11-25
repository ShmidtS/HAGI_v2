"""Хранилище экспертов и растущий процессор GrowingMathProcessor.

Эксперт — это запись (класс Expert): имя и семейство операции, символьная
программа Program, набор допустимых кортежей аргументов (конечные области
определения), правило допустимости входа, длина кода программы и, возможно,
обученная приближённая нейронная модель NeuralFiber.

Процессор GrowingMathProcessor находит эксперта по надзорляемым примерам
задачи TaskSpec. Для задачи берётся её конечная область D (все допустимые
входы), случайно разбивается на обучающую часть (доля train_fraction, минимум
1) и проверочную часть. Кандидаты — короткие программы; программа проходит,
если она верна на обучающих примерах, затем на проверочных, затем на всей D.
Победитель — первый в порядке (cost, текст программы) из прошедших; оценка
длины mdl_score = lambda * code_length, программы длиннее 32 отбрасываются.
Если ни одна не прошла — событие HOLD (отказ), эксперт не создаётся.

SafeExpression — ограниченный интерпретатор арифметико-логических выражений
по синтаксическому дереву Python (без eval): только числа, переменные,
двоичные и унарные операции, сравнения, условное выражение и вызовы из
белого списка CALLS, с пределами на размер результата и степень.

Состояние сохраняется в JSON и загружается обратно (save / load).
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import ast, json, math  # импорт стандартной библиотеки для разбора выражений, работы с данными, графами и случайностью
from dataclasses import dataclass  # декоратор класса-записи и фабрика значений по умолчанию
from pathlib import Path  # пути к файлам
from typing import Any, Optional, Sequence  # типы для аннотаций
import numpy as np  # численные массивы и генератор случайных чисел
from hagi.core.language.expr_dsl import Program  # имена из «expr_dsl» (модуль, где они определены)
from hagi.core.memory.aliases import _bit_nand8, _bit_nor8, _bit_not8, _bit_xnor8, _sign  # имена из «aliases» (модуль, где они определены)
from hagi.core.tasks.stored_tasks import (  # из описаний задач importировать:
    TaskSpec  # описание задачи: имя, семейство, область, учитель
)  # конец импорта описаний задач
from hagi.core.tasks.candidates import (  # из перечислителя программ importировать:
    candidate_programs  # список программ-кандидатов по числу аргументов
)  # конец импорта кандидатов
from hagi.core.neural.fiber import (  # из нейронного волокна importировать:
    NeuralFiber  # приближённая нейронная модель на основе волокна
)  # конец импорта волокна
from hagi.base.config import STATE_FORMAT  # единая метка формата состояния (без номера версии)

# --------------------------------- Expert store ---------------------------------  # раздел: хранилище экспертов


@dataclass  # класс-запись: инициализатор и представление генерируются автоматически
class Expert:
    """Один подтверждённый эксперт: символьная программа для операции.

        name — имя операции (совпадает с именем задачи); family — семейство;
        program — символьная программа; domains — кортеж допустимых значений
        по каждому аргументу; valid_rule — имя правила допустимости входа
        ("nonzero_y": второй аргумент не ноль; "ordered_yz": второй не больше
        третьего); code_length — стоимость программы; neural — приближённая
        нейронная модель или None; exhaustive_checks и exhaustive_errors —
        сколько входов проверено полным перебором и сколько ошибок найдено;
        parents — имена экспертов-родителей."""

    name: str  # имя операции
    family: str  # семейство операции
    program: Program  # символьная программа, вычисляющая ответ
    domains: tuple[tuple[int, ...], ...]  # domains[i] — допустимые значения i-го аргумента
    valid_rule: str  # правило допустимости входа
    code_length: int  # длина кода (стоимость) программы
    neural: Optional[NeuralFiber] = None  # нейронная модель; по умолчанию нет
    exhaustive_checks: int = 0  # число проверенных полным перебором входов, изначально 0
    exhaustive_errors: int = 0  # число ошибок при полном переборе, изначально 0
    parents: tuple[str, ...] = ()  # имена родителей, по умолчанию пустой кортеж

    def supports(self, xs: Sequence[int]) -> bool:
        """Входит ли кортеж аргументов xs в область эксперта.

            True, если len(xs) = числу аргументов, каждый xs[i] лежит в
            domains[i] и выполнено правило valid_rule; иначе False."""
        if len(xs) != len(self.domains):  # число аргументов не совпадает
            return False  # вход вне области
        if any(int(x) not in self.domains[i] for i, x in enumerate(xs)):  # хотя бы один x_i ∉ domains[i]
            return False  # вход вне области
        if self.valid_rule == "nonzero_y" and int(xs[1]) == 0:  # правило «y не ноль» и y = 0
            return False  # вход недопустим
        if self.valid_rule == "ordered_yz" and int(xs[1]) > int(xs[2]):  # правило «y ≤ z» и y > z
            return False  # вход недопустим
        return True  # все условия выполнены

    def to_dict(self) -> dict[str, Any]:
        """Словарь для JSON-сериализации: те же поля; кортежи становятся списками."""
        return {  # вернуть словарь полей
            "name": self.name,  # имя операции
            "family": self.family,  # семейство
            "program": self.program.to_dict(),  # программа как словарь
            "domains": [list(d) for d in self.domains],  # каждая область значений — список
            "valid_rule": self.valid_rule,  # правило допустимости
            "code_length": self.code_length,  # длина кода
            "neural": self.neural.to_dict() if self.neural else None,  # нейронная модель как словарь или None
            "exhaustive_checks": self.exhaustive_checks,  # число проверок
            "exhaustive_errors": self.exhaustive_errors,  # число ошибок
            "parents": list(self.parents),  # родители списком
        }  # конец словаря

    @staticmethod  # метод без доступа к экземпляру
    def from_dict(d: dict[str, Any]) -> "Expert":
        """Восстановить эксперта из словаря, созданного to_dict.

            Каждое поле приводится к нужному типу; отсутствующие поля
            получают те же значения по умолчанию, что и в классе."""
        return Expert(  # собрать эксперта из полей словаря d
            str(d["name"]),  # имя как строка
            str(d["family"]),  # семейство как строка
            Program.from_dict(d["program"]),  # программа из словаря
            tuple(tuple(int(v) for v in x) for x in d["domains"]),  # области: список списков -> кортеж кортежей целых
            str(d["valid_rule"]),  # правило допустимости
            int(d["code_length"]),  # длина кода
            NeuralFiber.from_dict(d["neural"]) if d.get("neural") else None,  # нейронная модель, если она была сохранена
            int(d.get("exhaustive_checks", 0)),  # число проверок (0, если нет)
            int(d.get("exhaustive_errors", 0)),  # число ошибок (0, если нет)
            tuple(str(x) for x in d.get("parents", [])),  # родители как кортеж строк
        )  # конец вызова


@dataclass  # класс-запись
class GrowthEvent:
    """Запись журнала роста: что процессор сделал с задачей и почему.

        version — номер версии модели после действия; task — имя задачи;
        action — «COMMIT» (эксперт принят), «HOLD» (отказ) или «REUSE»
        (эксперт уже был); reason — причина; program — текст программы;
        code_length — её длина; train_examples и validation_examples —
        размеры обучающей и проверочной выборок; exact_accuracy — точность
        на всей области (1.0 у принятого эксперта)."""

    version: int  # номер версии модели
    task: str  # имя задачи
    action: str  # выполненное действие
    reason: str  # причина действия
    program: str = ""  # текст программы, по умолчанию пустой
    code_length: int = 0  # длина кода, по умолчанию 0
    train_examples: int = 0  # число обучающих примеров, по умолчанию 0
    validation_examples: int = 0  # число проверочных примеров, по умолчанию 0
    exact_accuracy: float = 0.0  # точная точность, по умолчанию 0.0


def _bounded_int(value: Any, lo: int, hi: int) -> int:
    """Целое значение value, если lo ≤ value ≤ hi; иначе ошибка ValueError."""
    iv = int(value)  # привести к целому: iv = int(value)
    if iv < lo or iv > hi:  # iv вне отрезка [lo, hi]
        raise ValueError(f"integer argument must be in [{lo},{hi}]")  # ошибка: аргумент вне допустимого отрезка
    return iv  # вернуть iv


class SafeExpression:
    """Ограниченный интерпретатор арифметико-логических выражений (без eval).

        Выражение разбирается в дерево ast в режиме eval. Допустимы: числа
        (только конечные), имена переменных и константы pi, e, tau, двоичные
        операции BINOPS, унарные UNARY, and/or, цепочки сравнений CMP,
        условное выражение «a if c else b» и вызовы функций из CALLS только
        по позиционным аргументам. Пределы: длина текста ≤ 1000, узлов ≤ 200,
        |степень| ≤ 64, сдвиг 0..4096, целый результат ≤ 4096 бит, дробный
        вход конечный и |x| ≤ 1e12, целый вход ≤ 1024 бит."""

    BINOPS = {  # двоичная операция по типу узла: a op b
        ast.Add: lambda a, b: a + b,  # сумма a и b
        ast.Sub: lambda a, b: a - b,  # разность a − b
        ast.Mult: lambda a, b: a * b,  # произведение a·b
        ast.Div: lambda a, b: a / b,  # дробное частное a / b
        ast.FloorDiv: lambda a, b: a // b,  # неполное частное ⌊a / b⌋
        ast.Mod: lambda a, b: a % b,  # остаток a mod b
        ast.Pow: lambda a, b: a**b,  # a в степени b
        ast.BitAnd: lambda a, b: int(a) & int(b),  # побитовое И: int(a) & int(b)
        ast.BitOr: lambda a, b: int(a) | int(b),  # побитовое ИЛИ: int(a) | int(b)
        ast.BitXor: lambda a, b: int(a) ^ int(b),  # исключающее ИЛИ: int(a) ^ int(b)
        ast.LShift: lambda a, b: int(a) << int(b),  # сдвиг влево: int(a) << int(b)
        ast.RShift: lambda a, b: int(a) >> int(b),  # сдвиг вправо: int(a) >> int(b)
    }  # конец таблицы двоичных операций
    UNARY = {  # унарная операция по типу узла: op a
        ast.UAdd: lambda a: +a,  # унарный плюс: +a
        ast.USub: lambda a: -a,  # унарный минус: −a
        ast.Invert: lambda a: ~int(a),  # побитое НЕ: ~int(a)
        ast.Not: lambda a: int(not bool(a)),  # логическое НЕ как 0/1: int(not bool(a))
    }  # конец таблицы унарных операций
    CMP = {  # сравнение по типу узла: a op b
        ast.Eq: lambda a, b: a == b,  # сравнение: a = b
        ast.NotEq: lambda a, b: a != b,  # сравнение: a ≠ b
        ast.Lt: lambda a, b: a < b,  # сравнение: a < b
        ast.LtE: lambda a, b: a <= b,  # сравнение: a ≤ b
        ast.Gt: lambda a, b: a > b,  # сравнение: a > b
        ast.GtE: lambda a, b: a >= b,  # сравнение: a ≥ b
    }  # конец таблицы сравнений
    CALLS = {  # разрешённые функции по имени
        "abs": abs,  # модуль |x|
        "min": min,  # минимум
        "max": max,  # максимум
        "gcd": math.gcd,  # наибольший общий делитель
        "lcm": math.lcm,  # наименьшее общее кратное
        "sign": _sign,  # знак: −1, 0, 1
        "bit_not8": _bit_not8,  # побитовое НЕ в 8 битах
        "clamp": lambda x, lo, hi: min(max(x, lo), hi),  # срезка: min(max(x, lo), hi)
        "nand8": _bit_nand8,  # И-НЕ в 8 битах
        "nor8": _bit_nor8,  # ИЛИ-НЕ в 8 битах
        "xnor8": _bit_xnor8,  # исключающее ИЛИ-НЕ в 8 битах
        "sqrt": math.sqrt,  # квадратный корень √x
        "exp": math.exp,  # экспонента e^x
        "log": math.log,  # натуральный логарифм ln x
        "log2": math.log2,  # двоичный логарифм log2 x
        "log10": math.log10,  # десятичный логарифм lg x
        "sin": math.sin,  # синус
        "cos": math.cos,  # косинус
        "tan": math.tan,  # тангенс
        "asin": math.asin,  # арксинус
        "acos": math.acos,  # арккосинус
        "atan": math.atan,  # арктангенс
        "atan2": math.atan2,  # угол точки (y, x)
        "sinh": math.sinh,  # гиперболический синус
        "cosh": math.cosh,  # гиперболический косинус
        "tanh": math.tanh,  # гиперболический тангенс
        "floor": math.floor,  # округление вниз ⌊x⌋
        "ceil": math.ceil,  # округление вверх ⌈x⌉
        "hypot": math.hypot,  # длина вектора √(x² + y²)
        "degrees": math.degrees,  # радианы -> градусы
        "radians": math.radians,  # градусы -> радианы
        "factorial": lambda n: math.factorial(_bounded_int(n, 0, 1000)),  # n! для n из [0, 1000]
        "comb": lambda n, k: math.comb(  # число сочетаний C(n, k)
            _bounded_int(n, 0, 1000), _bounded_int(k, 0, 1000)  # n и k из отрезка [0, 1000]
        ),  # конец сочетаний
        "perm": lambda n, k=None: math.perm(  # число размещений P(n, k); без k — n!
            _bounded_int(n, 0, 1000), None if k is None else _bounded_int(k, 0, 1000)  # n из [0, 1000]; k None или из [0, 1000]
        ),  # конец размещений
    }  # конец белого списка функций

    def evaluate(  # вычислить выражение по переменным
        self, expression: str, variables: dict[str, int | float]  # выражение и словарь имя -> значение
    ) -> int | float | bool:  # возвращает число или 0/1
        """Вычислить выражение expression с переменными variables; результат — число или 0/1."""
        if len(expression) > 1000:  # текст длиннее 1000 символов
            raise ValueError("expression too long")  # ошибка: выражение слишком длинное
        tree = ast.parse(expression, mode="eval")  # разобрать выражение в дерево
        nodes = list(ast.walk(tree))  # все узлы дерева списком
        if len(nodes) > 200:  # узлов больше 200
            raise ValueError("expression too complex")  # ошибка: выражение слишком сложное
        result = self._visit(tree.body, variables)  # рекурсивно вычислить корень дерева
        if isinstance(result, (int, np.integer)) and int(result).bit_length() > 4096:  # целый результат длиннее 4096 бит
            raise OverflowError("result exceeds 4096-bit safety cap")  # ошибка переполнения
        if isinstance(result, (float, np.floating)) and not math.isfinite(  # дробный результат не конечен
            float(result)  # проверка finiteness числа float(result)
        ):  # конец условия
            raise OverflowError("non-finite floating-point result")  # ошибка: NaN или бесконечность
        return result  # вернуть результат

    def _visit(self, node: ast.AST, variables: dict[str, int | float]):  # вычислить один узел дерева
        """Рекурсивное вычисление узла node: возвращает его значение."""
        if isinstance(node, ast.Constant) and isinstance(  # узел — числовая константа
            node.value, (int, float, bool)  # значение — int, float или bool
        ):  # конец условия
            if isinstance(node.value, float) and not math.isfinite(node.value):  # дробная константа не конечна
                raise ValueError("non-finite numeric literal")  # ошибка: недопустимый литерал
            return node.value  # значение константы
        if isinstance(node, ast.Name):  # узел — имя
            if node.id in {"pi", "e", "tau"}:  # имя — одна из констант
                return {"pi": math.pi, "e": math.e, "tau": math.tau}[node.id]  # значение константы по имени
            if node.id not in variables:  # имени нет среди переменных
                raise ValueError(f"unknown variable: {node.id}")  # ошибка: неизвестная переменная
            value = variables[node.id]  # значение переменной
            if isinstance(value, float):  # значение дробное
                if not math.isfinite(value) or abs(value) > 1e12:  # не конечно или |value| > 1e12
                    raise OverflowError("floating input outside safe finite range")  # ошибка: вне безопасного диапазона
            elif abs(int(value)).bit_length() > 1024:  # иначе целое длиннее 1024 бит
                raise OverflowError("input exceeds 1024-bit safety cap")  # ошибка: вне безопасного диапазона
            return value  # вернуть значение переменной
        if isinstance(node, ast.BinOp) and type(node.op) in self.BINOPS:  # двоичная операция из таблицы
            a, b = self._visit(node.left, variables), self._visit(node.right, variables)  # a и b — значения левого и правого операндов
            if isinstance(node.op, ast.Pow) and abs(float(b)) > 64:  # степень: |b| > 64
                raise ValueError("absolute exponent must be at most 64")  # ошибка: степень слишком велика
            if isinstance(node.op, (ast.LShift, ast.RShift)) and not (  # сдвиг с недопустимым b
                0 <= int(b) <= 4096  # допустимо только 0 ≤ b ≤ 4096
            ):  # конец условия
                raise ValueError("shift must be in [0,4096]")  # ошибка: сдвиг вне отрезка
            return self.BINOPS[type(node.op)](a, b)  # применить операцию к a и b
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.UNARY:  # унарная операция из таблицы
            return self.UNARY[type(node.op)](self._visit(node.operand, variables))  # применить операцию к значению операнда
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):  # логическая цепочка and / or
            vals = [self._visit(v, variables) for v in node.values]  # значения всех операндов списком
            return int(  # результат как 0/1
                all(bool(v) for v in vals)  # для and: все истинны
                if isinstance(node.op, ast.And)  # если операция and
                else any(bool(v) for v in vals)  # иначе (or): хотя бы один истинен
            )  # конец int(...)
        if isinstance(node, ast.Compare):  # цепочка сравнений
            left = self._visit(node.left, variables)  # левое значение первого сравнения
            ok = True  # результат цепочки, пока True
            for op, comparator in zip(node.ops, node.comparators):  # по парам (операция, правый операнд)
                if type(op) not in self.CMP:  # операция не из таблицы
                    raise ValueError("comparison operator not allowed")  # ошибка: сравнение не разрешено
                right = self._visit(comparator, variables)  # значение правого операнда
                ok = ok and bool(self.CMP[type(op)](left, right))  # цепочка: ok := ok ∧ (left op right)
                left = right  # следующий оператор сравнивается с текущущим правым
            return int(ok)  # 1, если все сравнения истинны, иначе 0
        if isinstance(node, ast.IfExp):  # условное выражение
            cond = self._visit(node.test, variables)  # значение условия
            return self._visit(node.body if bool(cond) else node.orelse, variables)  # ветка body, если cond истинно, иначе orelse
        if (  # узел — вызов разрешённой функции по имени
            isinstance(node, ast.Call)  # это вызов
            and isinstance(node.func, ast.Name)  # вызываемое — простое имя
            and node.func.id in self.CALLS  # имя есть в белом списке
        ):  # конец условия
            if node.keywords:  # есть именованные аргументы
                raise ValueError("keyword arguments are not supported")  # ошибка: именованные аргументы не поддерживаются
            args = [self._visit(a, variables) for a in node.args]  # значения позиционных аргументов
            return self.CALLS[node.func.id](*args)  # вызвать функцию с этими аргументами
        raise ValueError(f"unsupported syntax: {type(node).__name__}")  # иной узел — ошибка: синтаксис не поддерживается


class GrowingMathProcessor:
    """Растущий процессор: подбирает и хранит экспертов по задачам.

        learn — найти и принять одного эксперта для задачи (COMMIT) или
        отказаться (HOLD); predict — ответ эксперта точно или нейронной
        моделью, с отказом вне области; verify_all — точность всех
        экспертов на полных областях; evaluate_expression — безопасное
        вычисление выражения; save / load — JSON-состояние."""

    def __init__(self, lookup_limit: int = 16, mdl_lambda: float = 0.001):  # создать процессор
        self.lookup_limit = int(lookup_limit)  # предел размера области для таблицы ответов
        self.mdl_lambda = float(mdl_lambda)  # вес длины кода lambda в оценке
        self.experts: dict[str, Expert] = {}  # эксперты: имя -> Expert, пока пусто
        self.version = 0  # номер версии модели
        self.growth_log: list[GrowthEvent] = []  # журнал событий роста
        self.total_commits = 0  # всего принятых экспертов
        self.total_holds = 0  # всего отказов

    def _candidate_list(self, spec: TaskSpec) -> list[Program]:
        """Список программ-кандидатов для задачи.

            Для семейства BoolVenn — 16 программ от двух аргументов с маской
            m = 0..15 (каждый бит маски — строка таблицы истинности); иначе
            общий перечень candidate_programs по числу аргументов."""
        if spec.family == "BoolVenn":  # задача — булева диаграмма Венна
            return [Program(2, venn_mask=m) for m in range(16)]  # 16 программ с масками m = 0..15
        return candidate_programs(spec.arity)  # кандидаты по арности задачи

    @staticmethod  # метод без доступа к экземпляру
    def _same(a: Any, b: Any) -> bool:  # сравнить a и b как целые
        """True, если int(a) = int(b); при невозможности преобразования — False."""
        try:  # преобразование может вызвать ошибку
            return int(a) == int(b)  # сравнение целых значений
        except Exception:  # ошибка преобразования или вычисления
            return False  # считать значения различными

    def learn(  # обучиться одной задаче
        self,  # сам процессор
        spec: TaskSpec,  # описание задачи
        seed: int = 0,  # зерно генератора случайных чисел
        train_fraction: float = 0.70,  # доля обучающей выборки
        poison_one_label: bool = False,  # испортить одну метку (проверка отказа)
        train_neural: bool = True,  # обучать ли нейронную модель
    ) -> GrowthEvent:  # возвращает событие журнала
        """Найти и принять одного эксперта по надзорляемым примерам задачи.

            Возвращает COMMIT с новым экспертом, REUSE, если эксперт уже
            есть, или HOLD с причиной. Значения учителя spec.teacher
            используются здесь только для проверки кандидатов: при работе
            вместо них нужен независимый источник верных ответов."""
        if spec.name in self.experts:  # эксперт для такой задачи уже есть
            ev = GrowthEvent(self.version, spec.name, "REUSE", "already learned")  # событие «повторное использование»
            self.growth_log.append(ev)  # записать в журнал
            return ev  # вернуть событие
        domain = spec.inputs()  # конечная область D — все допустимые входы
        if not domain:  # область пуста
            return self._hold(spec, "EMPTY_VALID_DOMAIN", 0, 0)  # отказ: пустая область
        rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
        perm = rng.permutation(len(domain))  # случайная перестановка индексов 0..|D|−1
        if spec.family == "BoolVenn":  # булева задача
            n_train = max(1, len(domain) - 1)  # обучающих |D| − 1 (не менее 1): один вход остаётся на проверку
        else:  # прочие семейства
            n_train = max(  # обучающих не менее 1
                1, min(len(domain) - 1, int(round(len(domain) * train_fraction)))  # и не более |D| − 1: round(|D| · train_fraction)
            )  # конец max
        train_idx, val_idx = perm[:n_train], perm[n_train:]  # первые n_train индексов — обучение, остальные — проверка
        train_x = [domain[int(i)] for i in train_idx]  # обучающие входы
        val_x = [domain[int(i)] for i in val_idx]  # проверочные входы
        all_y = {xs: int(spec.teacher(*xs)) for xs in domain}  # верные ответы учителя на всей области
        train_y = [all_y[xs] for xs in train_x]  # метки обучающих примеров
        val_y = [all_y[xs] for xs in val_x]  # метки проверочных примеров
        if poison_one_label and train_y:  # включена порча метки и есть обучающие примеры
            # детерминированная порча: проверка того, что процессор откажется или откатит решение
            train_y[0] = int(train_y[0]) + 1  # первая метка увеличена на 1

        # приближённая нейронная модель обучается на тех же видимых примерах
        neural = None  # модели пока нет
        if train_neural and val_x:  # нужно обучать и есть проверочные примеры
            try:  # обучение может не сойтись
                neural = NeuralFiber.fit(  # обучить волокно
                    train_x, train_y, val_x, val_y, seed=seed + 991, epochs=120  # на train, проверка на val, зерно seed + 991, 120 эпох
                )  # конец fit
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):  # численный сбой обучения
                neural = None  # модели нет

        candidates = self._candidate_list(spec)  # программы-кандидаты
        train_matches: list[Program] = []  # прошедшие обучающую выборку
        for program in candidates:  # по всем кандидатам
            try:  # программа может упасть при вычислении
                if all(  # верна на всех обучающих примерах
                    self._same(program.eval(xs), yy) for xs, yy in zip(train_x, train_y)  # program(xs_i) = y_i для каждой пары
                ):  # конец all
                    train_matches.append(program)  # добавить в прошедшие
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                continue  # кандидат отбрасывается
        train_matches.sort(key=lambda p: (p.cost, p.pretty()))  # отсортировать по (стоимость, текст программы)

        # сначала независимая проверка, затем полный перебор области со сравнением с учителем
        for program in train_matches:  # по прошедшим в порядке возрастания стоимости
            try:  # программа может упасть при вычислении
                if not all(  # не верна на всех проверочных примерах
                    self._same(program.eval(xs), yy) for xs, yy in zip(val_x, val_y)  # program(xs_j) = y_j для каждой пары
                ):  # конец all
                    continue  # следующий кандидат
                errors = 0  # число ошибок на всей области
                for xs in domain:  # по всем входам области
                    if not self._same(program.eval(xs), all_y[xs]):  # ответ неверен
                        errors += 1  # счётчик ошибок: errors + 1
                if errors != 0:  # есть ошибки
                    continue  # следующий кандидат
                # оценка минимальной длины описания: нулевая ошибка плюс явная длина кода
                code_length = int(program.cost)  # длина кода = стоимость программы
                mdl_score = 0.0 + self.mdl_lambda * code_length  # оценка: mdl_score = lambda · code_length
                if not np.isfinite(mdl_score) or code_length > 32:  # оценка не конечна или код длиннее 32
                    continue  # следующий кандидат
                expert = Expert(  # создать эксперта
                    spec.name,  # имя операции
                    spec.family,  # семейство
                    program,  # программа
                    spec.domains,  # области значений аргументов
                    spec.valid_rule,  # правило допустимости
                    code_length,  # длина кода
                    neural,  # нейронная модель
                    len(domain),  # проверено |D| входов
                    0,  # ошибок 0
                )  # конец Expert(...)
                self.experts[spec.name] = expert  # сохранить эксперта
                self.version += 1  # версия модели + 1
                self.total_commits += 1  # счётчик принятых + 1
                ev = GrowthEvent(  # событие «принят»
                    self.version,  # версия
                    spec.name,  # задача
                    "COMMIT",  # действие
                    "validation+exhaustive pass",  # причина: прошёл проверку и полный перебор
                    program.pretty(),  # текст программы
                    code_length,  # длина кода
                    len(train_x),  # число обучающих примеров
                    len(val_x),  # число проверочных примеров
                    1.0,  # точность 1.0
                )  # конец события
                self.growth_log.append(ev)  # записать в журнал
                return ev  # вернуть событие
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):  # ошибка вычисления
                continue  # следующий кандидат

        # таблица ответов допустима только при полной области не крупнее заданного предела
        if (  # все три условия выполнены
            len(domain) <= self.lookup_limit  # размер области: |D| ≤ lookup_limit
            and not poison_one_label  # метки не испорчены
            and len(train_x) == len(domain)  # обучающая выборка — вся область
        ):  # конец условия
            # таблица ответов не создаётся намеренно: ветка оставлена для будущих
            # полных областей малого размера и не экстраполирует за их пределы
            pass  # ничего не делать
        return self._hold(  # отказ: годной программы не найдено
            spec, "NO_VALIDATED_PROGRAM", len(train_x), len(val_x), neural  # задача, причина, размеры выборок, модель
        )  # конец отказа

    def _hold(  # зарегистрировать отказ
        self,  # сам процессор
        spec: TaskSpec,  # задача
        reason: str,  # причина отказа
        ntrain: int,  # число обучающих примеров
        nval: int,  # число проверочных примеров
        neural: Optional[NeuralFiber] = None,  # нейронная модель (не сохраняется)
    ) -> GrowthEvent:  # возвращает событие журнала
        """Записать в журнал отказ HOLD по причине reason и вернуть событие."""
        self.total_holds += 1  # счётчик отказов + 1
        ev = GrowthEvent(  # событие «отказ»
            self.version, spec.name, "HOLD", reason, "", 0, ntrain, nval, 0.0  # версия, задача, действие, причина, пустая программа, длина 0, размеры выборок, точность 0.0
        )  # конец события
        self.growth_log.append(ev)  # записать в журнал
        return ev  # вернуть событие

    def predict(self, name: str, *xs: int, mode: str = "exact") -> dict[str, Any]:  # ответ эксперта name на входе xs
        """Ответ эксперта name на входе xs; mode «exact» — программой, «neural» — моделью.

            Вне области или без эксперта возвращается статус ABSTAIN (отказ);
            иначе OK с точным значением или APPROXIMATE с приближённым."""
        expert = self.experts.get(name)  # эксперт по имени или None
        if expert is None:  # эксперта нет
            return {"status": "ABSTAIN", "reason": "unknown operation"}  # отказ: неизвестная операция
        if not expert.supports(xs):  # вход вне подтверждённой области
            return {"status": "ABSTAIN", "reason": "outside certified finite domain"}  # отказ: вне области
        if mode == "exact":  # точный режим
            return {  # точный ответ
                "status": "OK",  # статус
                "value": expert.program.eval(xs),  # значение program(xs)
                "source": "verified-symbolic",  # источник: проверенная символьная программа
                "model_version": self.version,  # версия модели
            }  # конец словаря
        if mode == "neural":  # нейронный режим
            if expert.neural is None:  # модели нет
                return {"status": "ABSTAIN", "reason": "neural fiber unavailable"}  # отказ: модель недоступна
            return {  # приближённый ответ
                "status": "APPROXIMATE",  # статус
                "value": expert.neural.predict(xs),  # значение нейронной модели на xs
                "source": "neural-fiber-no-exact-guarantee",  # источник без гарантии точности
                "model_version": self.version,  # версия модели
            }  # конец словаря
        raise ValueError("mode must be 'exact' or 'neural'")  # иной режим — ошибка

    def verify_all(self, specs: Sequence[TaskSpec]) -> dict[str, Any]:  # проверить всех экспертов
        """Проверить экспертов по задачам specs: запись на задачу и общая точность.

            Для задачи с экспертом accuracy = (число входов, где
            program(xs) = teacher(xs)) / max(1, |D|); общая точность
            correct / max(1, total)."""
        records = []  # записи по задачам
        total = correct = 0  # всего входов и верных ответов
        for spec in specs:  # по задачам
            expert = self.experts.get(spec.name)  # эксперт по имени или None
            if expert is None:  # эксперта нет
                records.append(  # запись без эксперта
                    {  # словарь записи
                        "task": spec.name,  # имя задачи
                        "committed": False,  # эксперт не принят
                        "accuracy": 0.0,  # точность 0.0
                        "n": len(spec.inputs()),  # размер области |D|
                    }  # конец словаря
                )  # конец append
                continue  # следующая задача
            rows = spec.inputs()  # область D
            ok = sum(  # число верных ответов ok
                int(expert.program.eval(xs) == int(spec.teacher(*xs))) for xs in rows  # сумма индикаторов Σ 1{program(xs) = teacher(xs)}
            )  # конец sum
            total += len(rows)  # всего входов: total + |D|
            correct += ok  # верных ответов: correct + ok
            records.append(  # запись с экспертом
                {  # словарь записи
                    "task": spec.name,  # имя задачи
                    "committed": True,  # эксперт принят
                    "accuracy": ok / max(1, len(rows)),  # точность ok / max(1, |D|)
                    "n": len(rows),  # размер области
                    "program": expert.program.pretty(),  # текст программы
                    "code_length": expert.code_length,  # длина кода
                }  # конец словаря
            )  # конец append
        return {  # итоговый словарь
            "records": records,  # записи по задачам
            "checked": total,  # всего входов
            "correct": correct,  # верных ответов
            "accuracy": correct / max(1, total),  # общая точность correct / max(1, total)
            "committed": len(self.experts),  # число экспертов
        }  # конец словаря

    def evaluate_expression(  # вычислить выражение
        self, expression: str, **variables: int | float  # выражение и именованные переменные
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Безопасно вычислить выражение expression с переменными variables.

            Возвращает OK со значением или ERROR с текстом причины."""
        try:  # вычисление может вызвать ошибку
            value = SafeExpression().evaluate(expression, variables)  # значение выражения через интерпретатор
            return {  # успешный результат
                "status": "OK",  # статус
                "value": value,  # значение
                "source": "safe-expression-dsl",  # источник: безопасный язык выражений
                "model_version": self.version,  # версия модели
            }  # конец словаря
        except (  # перехватываемые ошибки
            ValueError,  # недопустимое выражение
            TypeError,  # неверный тип
            ZeroDivisionError,  # деление на ноль
            OverflowError,  # переполнение
            SyntaxError,  # синтаксическая ошибка
        ) as exc:  # exc — объект исключения
            return {  # результат с ошибкой
                "status": "ERROR",  # статус
                "reason": str(exc),  # текст ошибки
                "model_version": self.version,  # версия модели
            }  # конец словаря

    def save(self, path: str | Path) -> None:  # сохранить состояние в файл path
        """Сохранить состояние процессора в JSON-файл path (сортировка ключей)."""
        data = {  # словарь состояния
            "format": "HAGI-GrowingMathProcessor",  # метка формата файла
            "version": STATE_FORMAT,  # версия формата
            "model_version": self.version,  # версия модели
            "lookup_limit": self.lookup_limit,  # предел таблицы ответов
            "mdl_lambda": self.mdl_lambda,  # вес длины кода
            "total_commits": self.total_commits,  # всего принято
            "total_holds": self.total_holds,  # всего отказов
            "experts": {k: v.to_dict() for k, v in sorted(self.experts.items())},  # эксперты по именам в алфавитном порядке, каждый словарём
            "growth_log": [vars(x) for x in self.growth_log],  # журнал как список словарей полей
        }  # конец словаря
        Path(path).write_text(  # записать текст в файл
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"  # JSON с отступом 2 и сортировкой ключей, кодировка UTF-8
        )  # конец write_text

    @staticmethod  # метод без доступа к экземпляру
    def load(path: str | Path) -> "GrowingMathProcessor":  # загрузить процессор из файла path
        """Загрузить процессор из JSON-файла path, созданного save."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # прочитать файл и разобрать JSON
        if data.get("format") != "HAGI-GrowingMathProcessor":  # чужой формат файла
            raise ValueError("not a HAGI Growing Math Processor state file")  # ошибка: это не файл состояния процессора
        model = GrowingMathProcessor(  # процессор с параметрами из файла
            int(data.get("lookup_limit", 16)), float(data.get("mdl_lambda", 0.001))  # предел (по умолчанию 16) и lambda (по умолчанию 0.001)
        )  # конец конструктора
        model.version = int(data.get("model_version", 0))  # версия модели (по умолчанию 0)
        model.total_commits = int(data.get("total_commits", 0))  # всего принято (по умолчанию 0)
        model.total_holds = int(data.get("total_holds", 0))  # всего отказов (по умолчанию 0)
        model.experts = {  # восстановить экспертов
            k: Expert.from_dict(v) for k, v in data.get("experts", {}).items()  # имя -> Expert.from_dict(словарь)
        }  # конец словаря
        model.growth_log = [GrowthEvent(**x) for x in data.get("growth_log", [])]  # журнал: каждый словарь -> GrowthEvent
        return model  # вернуть процессор

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'Expert', 'GrowthEvent', 'SafeExpression', 'GrowingMathProcessor',  # имя интерфейса
]  # конец перечня
