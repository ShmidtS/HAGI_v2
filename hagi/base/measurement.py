"""Набор средств измерения: решает, принимать или откатывать изменение модели.

Ключевые идеи и формулы:
  - CapabilityMeter сравнивает результаты на отложенных задачах до и после
    изменения: регрессия числа решённых задач и программ; прирост ёмкости
      delta = (#структурных + #программ) − (решённых до изменения);
    выигрыш по цене = средние биты или среднее число проб стали меньше.
    Есть причины против -> 'ROLLBACK', иначе 'COMMIT'.
  - rollback удаляет представление из реестра и восстанавливает снимок.
  - weighted_exp_allocation: точное замкнутое решение задачи распределения
    бюджета B:  budget = Σ_i w_i/κ_i,  logCoeff = Σ_i (w_i/κ_i)·ln(c_i·κ_i/w_i),
      shift = (logCoeff − B)/budget,  alloc_i = (ln(c_i·κ_i/w_i) − shift)/κ_i.
  - fisher_tolerances: допуск ошибки по координате j при кривизне c_j равен
    alloc_j для c = (1,…,1), κ = col_curv, w = веса: чем больше кривизна,
    тем меньше допуск.
  - CachedOracle: обёртка над «учителем» с кэшем по кортежу входа и пределом
    числа разных запросов.
  - oe_first_representatives: оставляет первый program для каждого вектора
    ответов на пробах key = (evaluate(program, x))_{x in probes}.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов


# ============================================================
# СЧИТЧИК ПЕРЕНОСА (принять/откатить изменение)
# ============================================================

from dataclasses import dataclass, field  # классы-записи и значения по умолчанию
from typing import Callable  # тип «вызываемый объект»
@dataclass  # класс-запись: сравнение и печать по полям
class TaskResult:  # результат одной задачи
    """Результат одной задачи после изменения модели.

        task_name — имя; solved — задача решена; structural — решена структурным
        способом; program — решена программой; bits — затрачено битов; probes —
        число проб (обращений к учителю)."""
    task_name: str  # имя задачи
    solved: bool  # решена ли задача
    structural: bool  # решена ли структурным способом
    program: bool  # решена ли программой
    bits: int  # затрачено битов
    probes: int  # число проб
@dataclass  # класс-запись
class MeterVerdict:  # вердикт счётчика
    """Вердикт: action — 'COMMIT' (принять) или 'ROLLBACK' (откатить); reasons —
    список причин против; metrics — словарь измеренных величин."""
    action: str  # 'COMMIT' или 'ROLLBACK'
    reasons: list[str] = field(default_factory=list)  # причины; по умолчанию пустой список
    metrics: dict = field(default_factory=dict)  # измеренные величины; по умолчанию пустой словарь
class CapabilityMeter:  # счётчик способности: сравнение «до» и «после»

    """Счётчик способности: сравнивает итоги на отложенных задачах после
    изменения с числами до изменения."""

    def __init__(self, alpha_fdr: float=0.0):  # alpha_fdr — допустимое число ложных «структурных» на контроле
        self.alpha_fdr = alpha_fdr  # запомнить порог ложных срабатываний

    def evaluate(self, held_out_results: list[TaskResult], matched_control_results: list[TaskResult], pre_solve_count: int, pre_program_count: int, pre_avg_bits: float, post_avg_bits: float, pre_avg_probes: float, post_avg_probes: float) -> MeterVerdict:
        """Оценить изменение и вернуть вердикт.

            held_out_results — результаты на отложенных задачах после изменения;
            matched_control_results — результаты на контрольных (шумовых) задачах;
            pre_solve_count, pre_program_count — решённые задачи и программы до
            изменения; pre/post_avg_bits, pre/post_avg_probes — средние биты и
            пробы до и после.
            solved = {r : r.solved}; structural = {r ∈ solved : r.structural};
            programs = {r ∈ solved : r.program};
            delta = |structural| + |programs| − pre_solve_count.
            Вердикт 'ROLLBACK', если есть хоть одна из причин: решённых или
            программ стало меньше; delta ≤ 0; ложных структурных на контроле
            больше alpha_fdr; ни биты, ни пробы не уменьшились."""
        reasons: list[str] = []  # причины откатить изменение
        m: dict = {}  # измеренные величины
        solved = [r for r in held_out_results if r.solved]  # решённые задачи
        structural = [r for r in solved if r.structural]  # решённые структурным способом
        programs = [r for r in solved if r.program]  # решённые программой
        if len(solved) < pre_solve_count:  # решённых стало меньше, чем до изменения
            reasons.append(f'REGRESSION: solved {len(solved)} < {pre_solve_count}')  # причина: регрессия числа решённых
        if len(programs) < pre_program_count:  # программ стало меньше, чем до изменения
            reasons.append(f'REGRESSION: programs {len(programs)} < {pre_program_count}')  # причина: регрессия числа программ
        m['solved'] = len(solved)  # число решённых
        m['structural'] = len(structural)  # число структурных решений
        m['programs'] = len(programs)  # число программ
        delta = len(structural) + len(programs) - pre_solve_count  # прирост ёмкости: |structural| + |programs| − pre_solve_count
        m['delta_capacity'] = delta  # записать прирост
        if delta <= 0:  # ёмкость не выросла
            reasons.append(f'NO_CAPACITY_GAIN: delta={delta}')  # причина: нет прироста ёмкости
        false_structural = sum((1 for r in matched_control_results if r.structural))  # число контрольных задач, ошибочно помеченных структурными
        m['false_structural_controls'] = false_structural  # записать его
        if false_structural > self.alpha_fdr:  # ложных структурных больше порога
            reasons.append(f'FALSE_STRUCTURAL on controls: {false_structural}')  # причина: ложные структурные на контроле
        bits_win = post_avg_bits < pre_avg_bits  # биты уменьшились: post < pre
        probes_win = post_avg_probes < pre_avg_probes  # пробы уменьшились: post < pre
        m['bits_win'] = bits_win  # записать флаг выигрыша по битам
        m['probes_win'] = probes_win  # записать флаг выигрыша по пробам
        m['avg_bits'] = (pre_avg_bits, post_avg_bits)  # пара средних битов (до, после)
        m['avg_probes'] = (pre_avg_probes, post_avg_probes)  # пара средних проб (до, после)
        if not (bits_win or probes_win):  # ни один показатель цены не улучшился
            reasons.append('NO_COST_WIN')  # причина: нет выигрыша по цене
        if reasons:  # есть хоть одна причина против
            return MeterVerdict('ROLLBACK', reasons, m)  # вердикт: откатить
        return MeterVerdict('COMMIT', [], m)  # иначе: принять, причин против нет
def rollback(registry_remove: Callable[[str], None], repr_id: str, restore_snapshot: Callable[[], None]) -> MeterVerdict:
    """Честный откат: удалить запись repr_id из реестра и восстановить снимок.

        registry_remove(id) удаляет запись; restore_snapshot() возвращает модель
        к состоянию до изменения и выполняется всегда, даже если удаление
        вызвало ошибку."""
    try:  # снятие восстанавливается в любом случае
        registry_remove(repr_id)  # удалить запись repr_id из реестра
    finally:  # блок выполняется и при ошибке выше
        restore_snapshot()  # восстановить сохранённый снимок модели
    return MeterVerdict('ROLLBACK', ['rolled back'], {'repr_id': repr_id})  # вердикт «откатить» с одной причиной и именем представления

# ============================================================
# СНИМОК СПОСОБНОСТИ
# ============================================================

from dataclasses import dataclass  # класс-запись (повторный импорт безвреден)
@dataclass  # класс-запись
class CapabilitySnapshot:  # снимок способности модели в одной версии
    """Снимок способности модели в одной версии: версия, биты описания, доля
    решённых отложенных задач, среднее число образцов до решения и число ложных
    приёмов на шуме."""
    version: int  # номер версии модели
    library_meta_bits: int  # биты описания библиотеки
    experts_bits: int  # биты экспертов
    unsolved_penalty_bits: int  # штраф в битах за нерешённые задачи
    total_mdl_bits: int  # полная длина описания в битах
    heldout_solve_rate: float  # доля решённых отложенных задач
    mean_samples_to_solve: float  # среднее число образцов до решения
    noise_false_accepts: int  # число ложных приёмов на шуме

    def as_dict(self) -> dict:  # снимок в виде словаря
        """Вернуть копию словаря атрибутов снимка."""
        return self.__dict__.copy()  # поверхностная копия словаря атрибутов

# ============================================================
# ЛЮБОЕ ВРЕМЯ: АЛЬФА-БЮДЖЕТ
# ============================================================

import math  # математические функции
from typing import Sequence  # тип «последовательность»
def weighted_exp_budget(kappa: Sequence[float], w: Sequence[float]) -> float:
    """Суммарный бюджет: Σ_i w_i/κ_i."""
    return sum((wi / ki for wi, ki in zip(w, kappa)))  # сумма budget = Σ_i w_i/κ_i
def weighted_exp_log_coeff(c: Sequence[float], kappa: Sequence[float], w: Sequence[float]) -> float:
    """Логарифмический коэффициент: Σ_i (w_i/κ_i)·ln(c_i·κ_i/w_i)."""
    return sum((wi / ki * math.log(ci * ki / wi) for ci, ki, wi in zip(c, kappa, w)))  # коэффициент logCoeff = Σ_i (w_i/κ_i)·ln(c_i·κ_i/w_i)
def weighted_exp_allocation(c: Sequence[float], kappa: Sequence[float], w: Sequence[float], B: float) -> list[float]:
    """Точное замкнутое решение: alloc_i = (ln(c_i·κ_i/w_i) − shift)/κ_i,
    где shift = (logCoeff − B)/budget, budget = Σ w_i/κ_i,
    logCoeff = Σ (w_i/κ_i)·ln(c_i·κ_i/w_i). Выполняется Σ_i κ_i·alloc_i = B."""
    if not len(c) == len(kappa) == len(w) or not c:  # длины различаются или последовательности пусты
        raise ValueError('c/kappa/w must be same non-empty length')  # ошибка: длины должны совпадать и быть ненулевыми
    if any((k <= 0 for k in kappa)) or any((wi <= 0 for wi in w)) or any((ci <= 0 for ci in c)):  # есть неположительное κ_i, w_i или c_i
        raise ValueError('all c, kappa, w must be positive')  # ошибка: все значения должны быть положительными
    budget = weighted_exp_budget(kappa, w)  # сумма budget = Σ_i w_i/κ_i
    coeff = weighted_exp_log_coeff(c, kappa, w)  # коэффициент logCoeff = Σ_i (w_i/κ_i)·ln(c_i·κ_i/w_i)
    shift = (coeff - B) / budget  # сдвиг shift = (logCoeff − B)/budget
    return [(math.log(ci * ki / wi) - shift) / ki for ci, ki, wi in zip(c, kappa, w)]  # доли alloc_i = (ln(c_i·κ_i/w_i) − shift)/κ_i
def fisher_tolerances(col_curv: Sequence[float], B: float, w: Sequence[float] | None=None) -> list[float]:
    """Допуски квантования по координатам при бюджете B.

        col_curv — кривизна c_j по столбцу матрицы второго порядка (ошибка e_j
        стоит примерно c_j·e_j²/2 битов); w — веса координат или None (тогда все 1).
        Результат равен weighted_exp_allocation(c = (1,…,1), κ = col_curv, w, B):
        допуск монотонно убывает с кривизной — плоские координаты (малое c_j)
        допускают большую ошибку."""
    n = len(col_curv)  # число координат
    ww = w if w is not None else [1.0] * n  # веса: данные или единичные
    if any((cj <= 0 for cj in col_curv)):  # есть неположительная кривизна
        raise ValueError('column curvature must be positive')  # ошибка: кривизна должна быть положительной
    return weighted_exp_allocation([1.0] * n, list(col_curv), ww, B)  # допуски = распределение бюджета при c = 1, κ = col_curv

# ============================================================
# КЭШИРУЮЩИЙ УЧИТЕЛЬ
# ============================================================

from collections import Counter  # счётчик по ключам
from typing import Callable, Iterable  # типы «вызываемый объект» и «итерируемый»
class OracleQueryBudgetExceeded(RuntimeError):  # ошибка времени выполнения
    """Возникает, когда новый вызов учителя превысил бы бюджет разных запросов."""
@dataclass(frozen=True)  # неизменяемая класс-запись
class OracleStats:  # статистика обращений к учителю
    """Статистика: число разных запросов, общее число обращений, число
    попаданий в кэш и те же числа, разбитые по назначению purpose."""
    unique_queries: int  # число разных кортежей входа
    requests: int  # общее число обращений
    cache_hits: int  # число попаданий в кэш
    unique_by_purpose: tuple[tuple[str, int], ...]  # (назначение, число разных запросов)
    cache_hits_by_purpose: tuple[tuple[str, int], ...]  # (назначение, число попаданий в кэш)

    def to_dict(self) -> dict[str, object]:  # статистика в виде словаря
        """Вернуть статистику словарём; разбиения по назначению — тоже словари."""
        return {'unique_queries': self.unique_queries, 'requests': self.requests, 'cache_hits': self.cache_hits, 'unique_by_purpose': dict(self.unique_by_purpose), 'cache_hits_by_purpose': dict(self.cache_hits_by_purpose)}  # словарь: счётчики как есть, пары (назначение, число) превращены в словари
    """Кэширующая обёртка над детерминированным учителем в рамках одной задачи.

        Аргументы:
          teacher — функция, принимающая по одному значению на каждую ось входа;
          arity — ожидаемое число координат входа или None;
          max_unique_queries — жёсткий предел числа разных вызовов или None.

        Обёртка хранит целые метки, потому что символьные задачи принимают и
        возвращают целые числа. Она ничего не знает о смысле задачи, о разбиении
        на обучение и проверку и о стратегии выбора точек."""
class CachedOracle:  # кэширующая обёртка над учителем

    def __init__(self, teacher: Callable[..., int], *, arity: int | None=None, max_unique_queries: int | None=None) -> None:  # учитель, арность и предел запросов — только по имени
        if not callable(teacher):  # учитель не вызывается
            raise TypeError('teacher must be callable')  # ошибка типа
        if arity is not None and arity < 1:  # арность задана и меньше 1
            raise ValueError('arity must be positive')  # ошибка значения
        if max_unique_queries is not None and max_unique_queries < 0:  # предел задан и отрицателен
            raise ValueError('max_unique_queries must be non-negative')  # ошибка значения
        self._teacher = teacher  # сохранить функцию учителя
        self.arity = arity  # ожидаемое число координат
        self.max_unique_queries = max_unique_queries  # предел разных запросов
        self._labels: dict[tuple[int, ...], int] = {}  # кэш: кортеж входа -> метка
        self._requests = 0  # счётчик обращений
        self._cache_hits = 0  # счётчик попаданий в кэш
        self._unique_by_purpose: Counter[str] = Counter()  # разных запросов по назначениям
        self._cache_hits_by_purpose: Counter[str] = Counter()  # попаданий в кэш по назначениям

    def query(self, inputs: Iterable[int], *, purpose: str='unspecified') -> int:  # вход и назначение запроса
        """Вернуть метку из кэша или один раз вызвать учителя для этого кортежа."""
        key = tuple((int(value) for value in inputs))  # ключ кэша: кортеж целых входов
        if self.arity is not None and len(key) != self.arity:  # число координат не равно ожидаемому
            raise ValueError(f'oracle input arity {len(key)} != expected {self.arity}')  # ошибка: неверная арность
        if not purpose or not isinstance(purpose, str):  # назначение пустое или не строка
            raise ValueError('purpose must be a non-empty string')  # ошибка значения
        self._requests += 1  # обращений: +1
        if key in self._labels:  # кортеж уже есть в кэше
            self._cache_hits += 1  # попаданий в кэш: +1
            self._cache_hits_by_purpose[purpose] += 1  # попаданий с этим назначением: +1
            return self._labels[key]  # вернуть метку из кэша
        if self.max_unique_queries is not None and len(self._labels) >= self.max_unique_queries:  # предел задан и исчерпан: разных запросов ≥ предела
            raise OracleQueryBudgetExceeded(f'unique oracle query budget exhausted ({self.max_unique_queries})')  # ошибка: бюджет разных запросов исчерпан
        label = int(self._teacher(*key))  # новый вызов: метка = int(teacher(координаты ключа))
        self._labels[key] = label  # записать метку в кэш
        self._unique_by_purpose[purpose] += 1  # разных запросов с этим назначением: +1
        return label  # вернуть метку

    @property  # доступ к методу как к атрибуту
    def unique_queries(self) -> int:  # число разных запросов
        return len(self._labels)  # число кортежей в кэше
    @property  # доступ к методу как к атрибуту
    def labels(self) -> dict[tuple[int, ...], int]:  # снимок кэша
        """Вернуть защищённую копию кэша: для снятия отпечатков и проверок."""
        return dict(self._labels)  # копия словаря кэша

    def stats(self) -> OracleStats:  # статистика обращений
        return OracleStats(unique_queries=len(self._labels), requests=self._requests, cache_hits=self._cache_hits, unique_by_purpose=tuple(sorted(self._unique_by_purpose.items())), cache_hits_by_purpose=tuple(sorted(self._cache_hits_by_purpose.items())))  # снимок счётчиков; пары по назначениям отсортированы

# ============================================================
# ЗАПРОС ПО КОМИТЕТУ
# ============================================================

from typing import Callable, Hashable, Sequence, TypeVar  # типы и переменная типа
H = TypeVar('H')  # переменная типа для гипотез
X = TypeVar('X')  # переменная типа для точек
Y = TypeVar('Y', bound=Hashable)  # переменная типа для хешируемых меток
@dataclass(frozen=True)  # неизменяемая класс-запись
class QueryChoice:  # выбранная точка запроса
    """Выбранная точка запроса: point — сама точка; information_gain_bits —
    ожидаемый прирост информации в битах; outcome_probabilities — вероятности
    ответов (метка, вероятность)."""
    point: object  # точка, которую спрашивают
    information_gain_bits: float  # прирост информации в битах
    outcome_probabilities: tuple[tuple[Hashable, float], ...]  # список пар (метка ответа, вероятность)

def oe_first_representatives(programs, probes, evaluate):  # программы, точки-пробы, функция вычисления
    """Оставить ПЕРВУЮ программу для каждого вектора ответов на пробах.

        Наблюдательная эквивалентность: key(program) = (evaluate(program, x))
        по всем x из probes; программы с одинаковым key неотличимы на пробах,
        поэтому из каждой группы достаточно одной (первой). Это отсечение не
        теряет решений: эквивалентная программа даёт те же ответы."""
    seen = set()  # уже встречавшиеся векторы ответов
    out = []  # выбранные представители
    for program in programs:  # по программам в исходном порядке
        key = tuple(evaluate(program, x) for x in probes)  # ключ key = (evaluate(program, x))_{x ∈ probes}
        if key not in seen:  # такой вектор ответов ещё не встречался
            seen.add(key)  # запомнить вектор
            out.append(program)  # оставить программу представителем
    return out  # список представителей
