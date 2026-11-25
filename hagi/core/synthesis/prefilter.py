"""Векторизованный предварительный фильтр программ и путь на GPU.

Фильтр решает для пакета программ, какая из них согласуется с примерами
(X -> Y), не запуская скалярный интерпретатор для каждой программы на каждом
примере. Все примеры (строки матрицы X) вычисляются одновременно как столбцы
массива int64.

Результат по каждой программе — одно из трёх:
  True  — доказуемо согласуется (все строки дают Y);
  False — доказуемо не согласуется (скалярное вычисление вызвало бы ошибку
          в какой-либо строке);
  None  — неопределённость (возможно переполнение int64 или операция вне
          безопасной области) — решение остаётся за скалярной проверкой.

Правило надёжности: ok = «определены все строки» ∧ «определены все дочерние
узлы»; big = (|v| > 2^62) ∨ big любого дочернего узла. При not ok или big
вердикт None. Вердикт True — только если v равно Y по всем строкам.

Путь на GPU (PyTorch) делает то же самое на тензорах int64; PyTorch
загружается лениво, только при запросе устройства cuda.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
from typing import Any, Optional, Sequence  # типы для аннотаций
import numpy as np  # численные массивы и векторные операции
from hagi.core.tasks.stored_tasks import make_task_suite  # из описаний задач импортировать:
from hagi.core.processor.experts import GrowingMathProcessor  # из хранилища экспертов импортировать:

# ==============================================================================  # разделитель раздела
# ЧАСТЬ II — слой оптимального поиска: привязка имён в едином пространстве имён  # заголовок раздела
# ==============================================================================  # разделитель раздела

import sys as _sys  # модуль sys под именем _sys (доступ к загруженным модулям)


# имена процесса наследие привязываются напрямую или под прежними псевдонимами:
LegacyProcessor = GrowingMathProcessor  # LegacyProcessor — растущий процессор
from hagi.core.language.expr_dsl import OPS  # таблица операций языка (перепродаётся дальше как атрибут этого модуля)
from hagi.core.language.expr_dsl import Expr, Program  # имена из модуля «expr_dsl» (перепродаются дальше как атрибуты этого модуля)
from hagi.core.processor.experts import Expert, GrowthEvent, SafeExpression  # имена из модуля «experts» (перепродаются дальше как атрибуты этого модуля)
from hagi.core.tasks.stored_tasks import TaskSpec  # имена из модуля «stored_tasks» (перепродаются дальше как атрибуты этого модуля)
from hagi.core.neural.fiber import NeuralFiber  # имена из модуля «fiber» (перепродаются дальше как атрибуты этого модуля)
from hagi.core.memory.aliases import _bit_not8, _bit_nand8, _bit_nor8, _bit_xnor8, _pow_int, _sign  # имена из модуля «aliases» (перепродаются дальше как атрибуты этого модуля)
from hagi.core.tasks.candidates import candidate_programs  # имена из модуля «candidates» (перепродаются дальше как атрибуты этого модуля)
Expr = Expr  # Expr — тот же класс узла выражения (связывание для наглядности)
Program = Program  # Program — тот же класс программы
Expert = Expert  # Expert — тот же класс эксперта
GrowthEvent = GrowthEvent  # GrowthEvent — тот же класс события журнала
TaskSpec = TaskSpec  # TaskSpec — тот же класс описания задачи
NeuralFiber = NeuralFiber  # NeuralFiber — тот же класс нейронного волокна
SafeExpression = SafeExpression  # SafeExpression — тот же класс интерпретатора
OPS = OPS  # OPS — та же таблица операций
_make_task_suite = make_task_suite  # _make_task_suite — фабрика набора задач
_bit_not8 = _bit_not8  # _bit_not8 — побитовое НЕ в 8 битах
_bit_nand8 = _bit_nand8  # _bit_nand8 — И-НЕ в 8 битах
_bit_nor8 = _bit_nor8  # _bit_nor8 — ИЛИ-НЕ в 8 битах
_bit_xnor8 = _bit_xnor8  # _bit_xnor8 — исключающее ИЛИ с отрицанием в 8 битах
_pow_int = _pow_int  # _pow_int — целая степень
_sign = _sign  # _sign — знак числа
candidate_programs = candidate_programs  # candidate_programs — перечень кандидатов

"""Символьный обучитель точных многомерных функций на конечных областях.

Отличия от более раннего прототипа той же линии:

1. Точные дешёвые шаблоны и индукция рациональных многочленов выполняются
   до ограниченного поиска по общей грамматике.
2. Значения на пробных точках служат только ситом для отбраковки; разные
   синтаксические деревья не сливаются лишь потому, что совпали на малой
   выборке. Общий поиск упорядочивает кандидатов по явной длине префиксного
   кода, но грамматика ограничена, а бюджет поиска может оборвать его:
   глобальный минимум длины описания не гарантируется.
3. Повторное использование библиотеки учитывается как ссылочная стоимость,
   а не как произвольная скидка на длину кода. Заявленный битовый код —
   мера размера модели, а не размер файла в байтах.
4. Предварительный фильтр на int64 консервативен: он может отвергнуть
   только внутри ограниченной оболочки операций и входов. Все сомнительные
   случаи вычисляются скалярными целыми числами Python; полная скалярная
   проверка конечной области остаётся главным условием приёма.
5. Обучение нейронного волокна включается явно и является приближённым.
   Процессорный путь не импортирует PyTorch; поддержка CUDA загружается
   лениво.
6. Поиск теорем проверяет только допустимые входы и требует замкнутости
   перед композицией унарных функций. Синтетическая композиция остаётся
   внутри подтверждённых областей экспертов-родителей.

Это исследовательский прототип для конечных областей, а не модель общего
назначения. Приём ограничен полной заявленной конечной областью; вне её
ответ не выдаётся. Зависимости: Python >=3.10, NumPy (PyTorch по желанию).
"""

from concurrent.futures import ProcessPoolExecutor  # пул процессов для параллельных вычислений
from typing import Any, Iterator, Optional, Sequence  # типы для аннотаций


SIEVE_SIZE = 5  # число пробных точек для быстрой отбраковки
DEFAULT_NODE_BUDGET = 20_000  # максимум порождаемых кандидатов на задачу
DEFAULT_POP_BUDGET = 50_000  # жёсткий предел числа извлечений из очереди (предел вычислений)
VALIDATION_CHUNK = 64  # число кандидатов в одном пакете предварительной фильтрации
MAX_REUSE_EXPANSIONS = 8  # предел расширения подстановок экспертов на одно извлечение из очереди


# --------------------- векторизованная пакетная фильтрация (numpy) ----------------------  # раздел предварительного фильтра

_MAG_LIMIT = 1 << 62  # предел модуля 2^62; сверх него фильтр «не уверен», решает скалярная проверка


def _vec_binary(op: str, a: np.ndarray, b: np.ndarray):  # векторизованная двоичная операция
    """Векторизованная двоичная операция op над массивами int64.

        Возвращает (values, ok_mask): values — результат a op b; ok_mask —
        маска строк, где скалярная реализация не вызывает ошибку. ok = False
        у строк с делителем 0 (floordiv, mod), с нулевым аргументом у lcm
        или со степенью вне 0..8. В «плохих» строках значение считается на
        подставке (b = 1 или b = 0) и дальше не используется."""
    if op == "add":  # сложение
        return a + b, np.ones(a.shape, bool)  # сумма a + b; все строки определены
    if op == "sub":  # вычитание
        return a - b, np.ones(a.shape, bool)  # разность a − b; все строки определены
    if op == "mul":  # умножение
        return a * b, np.ones(a.shape, bool)  # произведение a·b; все строки определены
    if op in ("floordiv", "mod"):  # деление нацело или остаток
        ok = b != 0  # маска ok: b ≠ 0
        safe_b = np.where(ok, b, np.ones_like(b))  # safe_b равен b, где ok, иначе 1 (защита от делителя 0)
        if op == "floordiv":  # деление нацело
            return np.floor_divide(a, safe_b), ok  # частное ⌊a / b⌋ и маска ok
        return np.mod(a, safe_b), ok  # остаток a mod b и маска ok
    if op == "gcd":  # наибольший общий делитель
        return np.gcd(a, b), np.ones(a.shape, bool)  # НОД(a, b); все строки определены
    if op == "lcm":  # наименьшее общее кратное
        ok = (a != 0) & (b != 0)  # маска ok: a ≠ 0 и b ≠ 0
        sa = np.where(ok, a, np.ones_like(a))  # sa равен a, где ok, иначе 1
        sb = np.where(ok, b, np.ones_like(b))  # sb равен b, где ok, иначе 1
        return np.abs(sa * sb) // np.gcd(sa, sb), ok  # НОК = |a·b| / НОД(a, b) и маска ok
    if op == "min":  # минимум
        return np.minimum(a, b), np.ones(a.shape, bool)  # min(a, b); все строки определены
    if op == "max":  # максимум
        return np.maximum(a, b), np.ones(a.shape, bool)  # max(a, b); все строки определены
    if op == "eq":  # равенство
        return (a == b).astype(np.int64), np.ones(a.shape, bool)  # 1, если a = b, иначе 0; все строки определены
    if op == "ne":  # неравенство
        return (a != b).astype(np.int64), np.ones(a.shape, bool)  # 1, если a ≠ b, иначе 0; все строки определены
    if op == "lt":  # меньше
        return (a < b).astype(np.int64), np.ones(a.shape, bool)  # 1, если a < b, иначе 0; все строки определены
    if op == "le":  # меньше или равно
        return (a <= b).astype(np.int64), np.ones(a.shape, bool)  # 1, если a ≤ b, иначе 0; все строки определены
    if op == "gt":  # больше
        return (a > b).astype(np.int64), np.ones(a.shape, bool)  # 1, если a > b, иначе 0; все строки определены
    if op == "ge":  # больше или равно
        return (a >= b).astype(np.int64), np.ones(a.shape, bool)  # 1, если a ≥ b, иначе 0; все строки определены
    if op == "bit_and":  # побитовое И
        return a & b, np.ones(a.shape, bool)  # a & b; все строки определены
    if op == "bit_or":  # побитовое ИЛИ
        return a | b, np.ones(a.shape, bool)  # a | b; все строки определены
    if op == "bit_xor":  # исключающее ИЛИ
        return a ^ b, np.ones(a.shape, bool)  # a ^ b; все строки определены
    if op == "bit_nand8":  # И-НЕ в 8 битах
        return (~(a & b)) & 0xFF, np.ones(a.shape, bool)  # ¬(a & b) с маской младших 8 бит; все строки определены
    if op == "bit_nor8":  # ИЛИ-НЕ в 8 битах
        return (~(a | b)) & 0xFF, np.ones(a.shape, bool)  # ¬(a | b) с маской младших 8 бит; все строки определены
    if op == "bit_xnor8":  # равенство битов в 8 битах
        return (~(a ^ b)) & 0xFF, np.ones(a.shape, bool)  # ¬(a ^ b) с маской младших 8 бит; все строки определены
    if op == "pow_int":  # целая степень
        ok = (b >= 0) & (b <= 8)  # маска ok: 0 ≤ b ≤ 8
        sb = np.where(ok, b, np.zeros_like(b))  # sb равен b, где ok, иначе 0 (заглушка)
        return a**sb, ok  # a в степени b и маска ok
    raise ValueError(f"no vectorized op: {op}")  # неизвестная операция — ошибка


def _vec_unary(op: str, a: np.ndarray):  # векторизованная унарная операция
    """Векторизованная унарная операция op над массивом int64.

        Возвращает (values, ok_mask); все перечисленные операции определены
        в каждой строке, поэтому маска всегда из True."""
    if op == "neg":  # смена знака
        return -a, np.ones(a.shape, bool)  # −a; все строки определены
    if op == "abs":  # модуль
        return np.abs(a), np.ones(a.shape, bool)  # |a|; все строки определены
    if op == "sign":  # знак
        return np.sign(a).astype(np.int64), np.ones(a.shape, bool)  # sign(a) в int64; все строки определены
    if op == "bit_not8":  # побитовое НЕ в 8 битах
        return (~a) & 0xFF, np.ones(a.shape, bool)  # ¬a с маской младших 8 бит; все строки определены
    if op == "clamp3":  # срезка в отрезок [−3, 3]
        return np.clip(a, -3, 3), np.ones(a.shape, bool)  # min(max(a, −3), 3); все строки определены
    if op == "square":  # квадрат
        return a * a, np.ones(a.shape, bool)  # a²; все строки определены
    if op == "cube":  # куб
        return a * a * a, np.ones(a.shape, bool)  # a³; все строки определены
    raise ValueError(f"no vectorized unary op: {op}")  # неизвестная операция — ошибка


def _vec_eval_expr(e: Expr, cols: list[np.ndarray]):  # вычислить выражение по столбцам
    """Вычислить выражение e по столбцам примеров cols.

        Возвращает (values, ok_mask, big_mask). Для листьев (var, const)
        маски None. ok_mask = ок текущей операции ∧ ок всех дочерних узлов;
        big_mask = (|v| > 2^62) ∨ big всех дочерних узлов. Строка с big = True
        превысила безопасный модуль: вызывающий код обязан перепроверить её
        скалярно, а не доверять векторному вердикту."""
    if e.op == "var":  # лист — переменная
        return cols[int(e.value or 0)], None, None  # столбец с номером value (или 0), масок нет
    if e.op == "const":  # лист — константа
        v = np.full(cols[0].shape, int(e.value), dtype=np.int64)  # массив формы столбца, заполненный константой value
        return v, None, None  # константный столбец, масок нет
    vals = []  # значения дочерних узлов
    oks = []  # маски ok дочерних узлов (только не-None)
    bigs = []  # маски big дочерних узлов (только не-None)
    for a in e.args:  # по дочерним узлам
        v, ok, big = _vec_eval_expr(a, cols)  # рекурсивно вычислить дочерний узел
        vals.append(v)  # запомнить значение
        if ok is not None:  # у дочернего узла есть маска ok
            oks.append(ok)  # добавить её
        if big is not None:  # у дочернего узла есть маска big
            bigs.append(big)  # добавить её
    if e.op == "absdiff":  # операция |a − b|
        v, ok = np.abs(vals[0] - vals[1]), None  # значение |a − b|; маски нет
    elif e.op == "fma":  # произведение-сложение a·b + c
        v, ok = vals[0] * vals[1] + vals[2], None  # значение a·b + c; маски нет
    elif e.op == "clamp":  # срезка
        v, ok = np.minimum(np.maximum(vals[0], vals[1]), vals[2]), None  # значение min(max(a, lo), hi); маски нет
    elif e.op == "median3":  # медиана трёх
        v, ok = np.sort(np.stack(vals), axis=0)[1], None  # значение — средний из трёх отсортированных по строке; маски нет
    elif e.op == "clamp3":  # срезка в [−3, 3]
        v, ok = np.clip(vals[0], -3, 3), None  # значение min(max(a, −3), 3); маски нет
    elif e.op == "mul3":  # произведение трёх
        v, ok = vals[0] * vals[1] * vals[2], None  # значение a·b·c; маски нет
    elif len(vals) == 2 and e.op in OPS:  # два аргумента и операция из таблицы
        v, ok = _vec_binary(e.op, vals[0], vals[1])  # двоичная операция и её маска ok
    elif len(vals) == 1:  # один аргумент
        v, ok = _vec_unary(e.op, vals[0])  # унарная операция и её маска ok
    else:  # иная арность или операция
        raise ValueError(f"unsupported expression opcode: {e.op}")  # ошибка: код операции не поддерживается
    big = np.abs(v) > _MAG_LIMIT  # маска big: |v| > 2^62
    if bigs:  # есть маски big дочерних узлов
        big = big | bigs[0]  # объединить с первой
        for bb in bigs[1:]:  # по остальным
            big = big | bb  # объединить с каждой
    if ok is None:  # маски ok нет
        ok = np.ones(v.shape, bool)  # все строки определены
    # родитель имеет скалярную семантику в строке, только если в ней
    # определён каждый дочерний узел
    for child_ok in oks:  # по маскам ok дочерних узлов
        ok = ok & child_ok  # ok := ok ∧ ок дочернего узла
    return v, ok, big  # значение и обе маски


def _batch_fits_unchecked(  # пакетная проверка без гарантии по каждой программе
    programs: Sequence[Program], X: np.ndarray, Y: np.ndarray, device: str = "cpu"  # программы, матрица входов, вектор ответов, устройство
) -> list[Optional[bool]]:  # по программе: True / False / None
    """Пакетный надёжный фильтр для списка программ на примерах X -> Y.

        True — доказуемо согласуется с X -> Y; False — доказуемо нет (есть
        строка, где скалярное вычисление вызвало бы ошибку); None — неясно,
        решает скалярная проверка. На устройстве cuda работа передаётся
        тензорному пути, если PyTorch доступен."""
    out: list[Optional[bool]] = []  # результаты по программам
    if device.startswith("cuda") and _ensure_torch():  # устройство cuda и PyTorch загружен
        return _batch_fits_torch(programs, X, Y, device)  # тензорный путь
    for p in programs:  # по программам
        if p.expr is None:  # у программы нет выражения
            out.append(None)  # вердикт неясен
            continue  # следующая программа
        try:  # вычисление может вызвать ошибку
            cols = [np.ascontiguousarray(X[:, i]) for i in range(X.shape[1])]  # столбцы входов как непрерывные массивы
            v, ok, big = _vec_eval_expr(p.expr, cols)  # векторное значение и маски
            if not bool(np.all(ok)) or bool(np.any(big)):  # есть неопределённые или слишком большие строки
                out.append(None)  # вердикт неясен
                continue  # следующая программа
            out.append(bool(np.array_equal(v, Y)))  # True, если v = Y во всех строках, иначе False
        except (ValueError, OverflowError):  # ошибка вычисления
            out.append(None)  # вердикт неясен
    return out  # список вердиктов


# --------------------------- необязательный путь на GPU (torch) --------------------------  # раздел GPU-пути

_TORCH_OK = False  # PyTorch успешно загружен — пока нет
_TORCH_CHECKED = False  # попытка загрузки была — пока нет
_torch = None  # модуль PyTorch — пока не загружен


def _ensure_torch() -> bool:  # загрузить PyTorch при необходимости
    """Загрузить PyTorch, только когда явно нужны функции CUDA; True, если доступен."""
    global _TORCH_OK, _TORCH_CHECKED, _torch  # изменить глобальные переменные модуля
    if _TORCH_CHECKED:  # попытка уже была
        return _TORCH_OK  # вернуть прежний результат
    _TORCH_CHECKED = True  # отметить попытку
    try:  # импорт может не удаться
        import importlib  # динамический импорт модулей

        _torch = importlib.import_module("torch")  # загрузить модуль torch
        _TORCH_OK = True  # успех
    except ImportError:  # PyTorch не установлен
        _torch = None  # модуля нет
        _TORCH_OK = False  # неудача
    return _TORCH_OK  # результат загрузки


def _batch_fits_torch(  # пакетная проверка на тензорах
    programs: Sequence[Program], X: np.ndarray, Y: np.ndarray, device: str  # программы, входы, ответы, устройство
) -> list[Optional[bool]]:  # по программе: True / False / None
    """Пакетный фильтр на GPU с тем же договором, что и numpy-путь.

        Точно в целых int64; правила неопределённых строк те же."""
    dev = _torch.device(device)  # устройство вычислений
    Xt = _torch.from_numpy(X).to(dev)  # входы как тензор на устройстве
    Yt = _torch.from_numpy(Y).to(dev)  # ответы как тензор на устройстве
    cols = [Xt[:, i].contiguous() for i in range(X.shape[1])]  # столбцы входов как непрерывные тензоры
    one = _torch.ones_like(cols[0])  # тензор из единиц формы столбца (заглушка)

    def teval(e: Expr):  # рекурсивное вычисление узла на тензорах
        """Вычислить узел e; возвращает (значение, маска ok или None, маска big или None)."""
        if e.op == "var":  # лист — переменная
            return cols[int(e.value or 0)], None, None  # столбец с номером value (или 0), масок нет
        if e.op == "const":  # лист — константа
            v = _torch.full_like(cols[0], int(e.value))  # тензор формы столбца, заполненный константой value
            return v, None, None  # константный столбец, масок нет
        vs, oks, bigs = [], [], []  # значения, маски ok и маски big дочерних узлов
        for a in e.args:  # по дочерним узлам
            v, ok, big = teval(a)  # рекурсивно вычислить дочерний узел
            vs.append(v)  # запомнить значение
            if ok is not None:  # есть маска ok
                oks.append(ok)  # добавить её
            if big is not None:  # есть маска big
                bigs.append(big)  # добавить её
        op = e.op  # код операции
        if op == "absdiff":  # операция |a − b|
            v, ok = (vs[0] - vs[1]).abs(), None  # значение |a − b|; маски нет
        elif op == "fma":  # произведение-сложение a·b + c
            v, ok = vs[0] * vs[1] + vs[2], None  # значение a·b + c; маски нет
        elif op == "clamp":  # срезка
            v, ok = torch_clamp = (  # значение min(max(a, lo), hi) — кортеж (значение, None)
                _torch.minimum(_torch.maximum(vs[0], vs[1]), vs[2]),  # срезка min(max(a, lo), hi)
                None,  # маски нет
            )  # конец кортежа
        elif op == "median3":  # медиана трёх
            v, ok = _torch.sort(_torch.stack(vs), dim=0).values[1], None  # средний из трёх отсортированных по строке; маски нет
        elif op == "clamp3":  # срезка в [−3, 3]
            v, ok = _torch.clamp(vs[0], -3, 3), None  # min(max(a, −3), 3); маски нет
        elif op == "mul3":  # произведение трёх
            v, ok = vs[0] * vs[1] * vs[2], None  # a·b·c; маски нет
        elif op == "add":  # сложение
            v, ok = vs[0] + vs[1], None  # a + b; маски нет
        elif op == "sub":  # вычитание
            v, ok = vs[0] - vs[1], None  # a − b; маски нет
        elif op == "mul":  # умножение
            v, ok = vs[0] * vs[1], None  # a·b; маски нет
        elif op in ("floordiv", "mod"):  # деление нацело или остаток
            b0 = vs[1]  # делитель b
            ok = b0 != 0  # маска ok: b ≠ 0
            sb = _torch.where(ok, b0, one)  # sb равен b, где ok, иначе 1
            v = (  # значение — одно из двух
                _torch.div(vs[0], sb, rounding_mode="floor")  # ⌊a / b⌋, если floordiv
                if op == "floordiv"  # условие выбора
                else _torch.remainder(vs[0], sb)  # иначе a mod b
            )  # конец условного выражения
        elif op == "gcd":  # наибольший общий делитель
            v, ok = _torch.gcd(vs[0], vs[1]), None  # НОД(a, b); маски нет
        elif op == "lcm":  # наименьшее общее кратное
            a0, b0 = vs[0], vs[1]  # операнды a и b
            ok = (a0 != 0) & (b0 != 0)  # маска ok: a ≠ 0 и b ≠ 0
            sa = _torch.where(ok, a0, one)  # sa равен a, где ok, иначе 1
            sb = _torch.where(ok, b0, one)  # sb равен b, где ok, иначе 1
            v = _torch.abs(sa * sb) // _torch.gcd(sa, sb)  # НОК = |a·b| / НОД(a, b)
        elif op == "min":  # минимум
            v, ok = _torch.minimum(vs[0], vs[1]), None  # min(a, b); маски нет
        elif op == "max":  # максимум
            v, ok = _torch.maximum(vs[0], vs[1]), None  # max(a, b); маски нет
        elif op in ("eq", "ne", "lt", "le", "gt", "ge"):  # сравнение
            fn = {  # таблица функций сравнения по коду операции
                "eq": _torch.eq,  # равенство: a = b
                "ne": _torch.ne,  # неравенство: a ≠ b
                "lt": _torch.lt,  # меньше: a < b
                "le": _torch.le,  # не больше: a ≤ b
                "gt": _torch.gt,  # больше: a > b
                "ge": _torch.ge,  # не меньше: a ≥ b
            }[op]  # выбрать функцию по коду op
            v, ok = fn(vs[0], vs[1]).to(_torch.int64), None  # 0/1 как int64; маски нет
        elif op == "bit_and":  # побитовое И
            v, ok = vs[0] & vs[1], None  # a & b; маски нет
        elif op == "bit_or":  # побитовое ИЛИ
            v, ok = vs[0] | vs[1], None  # a | b; маски нет
        elif op == "bit_xor":  # исключающее ИЛИ
            v, ok = vs[0] ^ vs[1], None  # a ^ b; маски нет
        elif op == "bit_nand8":  # И-НЕ в 8 битах
            v, ok = (~(vs[0] & vs[1])) & 0xFF, None  # ¬(a & b) с маской младших 8 бит; маски нет
        elif op == "bit_nor8":  # ИЛИ-НЕ в 8 битах
            v, ok = (~(vs[0] | vs[1])) & 0xFF, None  # ¬(a | b) с маской младших 8 бит; маски нет
        elif op == "bit_xnor8":  # равенство битов в 8 битах
            v, ok = (~(vs[0] ^ vs[1])) & 0xFF, None  # ¬(a ^ b) с маской младших 8 бит; маски нет
        elif op == "pow_int":  # целая степень
            b0 = vs[1]  # показатель b
            ok = (b0 >= 0) & (b0 <= 8)  # маска ok: 0 ≤ b ≤ 8
            sb = _torch.where(ok, b0, _torch.zeros_like(b0))  # sb равен b, где ok, иначе 0
            v = vs[0] ** sb  # a в степени b
        elif op == "neg":  # смена знака
            v, ok = -vs[0], None  # −a; маски нет
        elif op == "abs":  # модуль
            v, ok = vs[0].abs(), None  # |a|; маски нет
        elif op == "sign":  # знак
            v, ok = _torch.sign(vs[0]).to(_torch.int64), None  # sign(a) в int64; маски нет
        elif op == "bit_not8":  # побитовое НЕ в 8 битах
            v, ok = (~(vs[0])) & 0xFF, None  # ¬a с маской младших 8 бит; маски нет
        elif op == "clamp3":  # срезка в [−3, 3]
            v, ok = _torch.clamp(vs[0], -3, 3), None  # min(max(a, −3), 3); маски нет
        elif op == "square":  # квадрат
            v, ok = vs[0] * vs[0], None  # a²; маски нет
        elif op == "cube":  # куб
            v, ok = vs[0] * vs[0] * vs[0], None  # a³; маски нет
        else:  # неизвестная операция
            raise ValueError(f"unsupported torch opcode: {op}")  # ошибка: код операции не поддерживается
        big = v.abs() > _MAG_LIMIT  # маска big: |v| > 2^62
        for bb in bigs:  # по маскам big дочерних узлов
            big = big | bb  # объединить с каждой
        if ok is None:  # маски ok нет
            ok = _torch.ones_like(big)  # все строки определены
        # ошибки области определения дочерних узлов распространяются по всему дереву
        for child_ok in oks:  # по маскам ok дочерних узлов
            ok = ok & child_ok  # ok := ok ∧ ок дочернего узла
        return v, ok, big  # значение и обе маски

    results: list[Optional[bool]] = []  # результаты по программам
    for p in programs:  # по программам
        if p.expr is None:  # у программы нет выражения
            results.append(None)  # вердикт неясен
            continue  # следующая программа
        try:  # вычисление может вызвать ошибку
            v, ok, big = teval(p.expr)  # значение и маски выражения
            if ok is None:  # маски ok нет
                ok = _torch.ones(v.shape, dtype=_torch.bool, device=v.device)  # все строки определены
            if bool((~ok).any()) or bool(big is not None and big.any()):  # есть неопределённые или слишком большие строки
                results.append(None)  # вердикт неясен
                continue  # следующая программа
            results.append(bool(_torch.equal(v, Yt)))  # True, если v = Y во всех строках, иначе False
        except (ValueError, RuntimeError, OverflowError):  # ошибка вычисления
            results.append(None)  # вердикт неясен
    return results  # список вердиктов

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'LegacyProcessor', 'SIEVE_SIZE', 'DEFAULT_NODE_BUDGET', 'DEFAULT_POP_BUDGET', 'VALIDATION_CHUNK',  # имя интерфейса
    'MAX_REUSE_EXPANSIONS',  # имя интерфейса
]  # конец перечня
