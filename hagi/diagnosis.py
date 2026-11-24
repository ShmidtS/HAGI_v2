#!/usr/bin/env python3
"""Диагностика пробелов: классификация причин, по которым задача не выучена.

    Здесь результат «нет проверенной программы» превращается в структурированную запись
    GapRecord: причина задержки HoldReason и дешёвый профиль остатка ResidualProfile.

    Причины (HoldReason):
      BUDGET_EXHAUSTED — не хватило бюджета узлов;
      NONDETERMINISTIC_OR_PARTIAL — учитель недетерминирован или определён не везде;
      RANDOM_INCOMPRESSIBLE — таблица похожа на случайный шум;
      REPRESENTATION_GAP — у модели нет подходящего представления.

    Профиль ResidualProfile считается по обучающим парам (x, y), где x — первый аргумент
    (или сумма аргументов), y — ответ. Числа профиля:
      k_density = |{x : y(x) ≠ 0}| / n — доля ненулевых ответов;
      n_runs — число отрезков подряд идущих ненулевых x (разрыв там, где x_{i+1} ≠ x_i + 1);
      autocorr_shift = P(y(x) = y(x−1) или |y(x) − y(x−1)| = 1) по x, у которых x−1 есть в таблице;
      autocorr_xor  = max_m P(y(x) = y(x ⊕ m)), m = 1..15;
      autocorr_reflect = P(y(x) = y(C − x)), C = max x;
      residual_frac_builtin — доля необъяснённого после лучшей встроенной подгонки;
      output_switch_rate = P(y ≠ y') по соседним парам x, x+1;
      shift_chance = 3 / K — случайный уровень сдвиговой пробы (равенство или ±1) при K разных ответах;
      period_hit = max_P P(y(x) = y(x + P)) по информативным парам (хотя бы один конец не равен
      самому частому ответу), P = 2..64; period_cover — доля информативных пар.

    GapBuffer накапливает записи и группирует по грубой сигнатуре профиля; кластер готов, когда
    в нём не меньше k_min задач с причиной REPRESENTATION_GAP — новое
    представление заводится не ради одной случайной задачи.

    Экспортирует: HoldReason, ResidualProfile, GapRecord, GapCluster, GapBuffer,
    build_residual_profile, classify_hold."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import math  # логарифм по основанию 2
from collections import Counter  # подсчёт повторений значений
from dataclasses import dataclass, field  # классы-записи
from enum import Enum  # перечисление причин
from typing import Callable, Iterable, Sequence  # аннотации типов


class HoldReason(Enum):
    """Причина, по которой задача осталась без проверенной программы."""
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"  # исчерпан бюджет (узлов или альфа-параметра)
    REPRESENTATION_GAP = "REPRESENTATION_GAP"  # нет подходящего представления
    NONDETERMINISTIC_OR_PARTIAL = "NONDETERMINISTIC_OR_PARTIAL"  # учитель недетерминирован или частичный
    RANDOM_INCOMPRESSIBLE = "RANDOM_INCOMPRESSIBLE"  # таблица похожа на случайный шум


@dataclass  # поля класса автоматически становятся аргументами конструктора
class ResidualProfile:
    """Дешёвые (миллисекунды) пробы по обучающим парам (train_x, train_y)."""
    n: int  # число обучающих примеров
    k_density: float                 # доля точек с ненулевым ответом: |{y ≠ 0}| / n
    uniq_outputs: int  # число различных ответов
    sparsity_gaps_entropy: float     # энтропия (в битах) распределения расстояний между соседними ненулевыми точками
    n_runs: int                      # число отрезков подряд идущих ненулевых точек
    autocorr_shift: float            # доля совпадений при сдвиге: P(f(x) = f(x−1))
    autocorr_reflect: float          # P(f(x) = f(C−x)) по x, у которых C−x есть в таблице
    autocorr_xor: float              # лучшая доля совпадений по xor: max_m P(f(x) = f(x ⊕ m)), m = 1..15
    residual_frac_builtin: float     # доля примеров, не объяснённых лучшей встроенной подгонкой
    output_switch_rate: float = 0.0  # P(y_t ≠ y_{t−1}) по соседним x; шумовая проба: у случайной
                                     # таблицы переключений около (K−1)/K, у законов они редки
    mn_classes: int | None = None  # число классов эквивалентности префиксов (для последовательностных задач); None — не считалось
    # сдвиговая проба засчитывает «равно или отличается на ±1» (для счётчиков), поэтому на узком
    # алфавите из K значений она срабатывает случайно примерно в 3/K случаев; случайный уровень
    # хранится явно, чтобы порог сравнения был относительным
    shift_chance: float = 0.0  # случайный уровень сдвиговой пробы: 3/K
    # максимум по периодам P = 2..64 доли P(f(x) = f(x + P)): настоящий периодический закон даёт 1.0,
    # случайная таблица держится около 1/K
    period_hit: float = 0.0  # лучшая доля совпадений на дистанции P
    period: int = 0  # период P, давший period_hit; 0 — период не искался
    period_cover: float = 0.0   # доля пар, на которых оценён период (информативные пары)

    def signature(self) -> tuple:
        """Грубая сигнатура профиля для группировки задач в кластер.

            Каждое число округляется до 2 знаков: bucket(v) = round(v, 2)."""
        bucket = lambda v: round(v, 2)  # округление числа до сотых
        return (bucket(self.k_density), bucket(self.autocorr_shift),  # (k_density, autocorr_shift) с округлением
                bucket(self.autocorr_xor), bucket(self.residual_frac_builtin),  # (autocorr_xor, residual_frac_builtin) с округлением
                self.mn_classes is not None)  # True, если число классов считалось


@dataclass  # поля класса автоматически становятся аргументами конструктора
class GapRecord:
    """Запись о задаче, оставшейся без программы: имя, причина и профиль остатка."""
    task_name: str  # имя задачи
    reason: HoldReason  # причина задержки
    profile: ResidualProfile  # профиль остатка задачи


@dataclass  # поля класса автоматически становятся аргументами конструктора
class GapCluster:
    """Кластер записей с одинаковой сигнатурой профиля."""
    signature: tuple  # общая сигнатура профилей
    records: list[GapRecord] = field(default_factory=list)  # записи кластера; по умолчанию пустой список

    def ready(self, k_min: int) -> bool:  # кластер готов, когда записей не меньше k_min
        return len(self.records) >= k_min  # кластер готов: число записей ≥ k_min


class GapBuffer:  # накопитель записей о пробелах с группировкой по сигнатуре
    def __init__(self, k_min: int = 3):  # создать накопитель; k_min — сколько задач нужно для готовности кластера
        self.k_min = k_min  # минимальный размер готового кластера
        self.records: list[GapRecord] = []  # все добавленные записи

    def add(self, rec: GapRecord) -> None:  # добавить запись
        self.records.append(rec)  # дописать в конец списка

    def clusters(self) -> list[GapCluster]:  # кластеры записей с причиной REPRESENTATION_GAP, от большего к меньшему
        by_sig: dict[tuple, GapCluster] = {}  # сигнатура -> кластер
        for rec in self.records:  # по всем записям
            if rec.reason != HoldReason.REPRESENTATION_GAP:  # причина не «пробел представления»
                continue  # в кластеры не входит
            sig = rec.profile.signature()  # сигнатура профиля записи
            by_sig.setdefault(sig, GapCluster(sig)).records.append(rec)  # завести кластер для сигнатуры, если новый, и добавить запись
        return sorted(by_sig.values(), key=lambda c: -len(c.records))  # кластеры по убыванию числа записей

    def ready_clusters(self) -> list[GapCluster]:  # только готовые кластеры
        return [c for c in self.clusters() if c.ready(self.k_min)]  # те, у которых записей не меньше k_min


# ------------------------------------------------------------------------------
# Реализации проб
# ------------------------------------------------------------------------------

def _entropy(values: Sequence[int]) -> float:  # энтропия Шеннона распределения значений, биты
    if not values:  # значений нет
        return 0.0  # энтропия пустого набора — 0
    cnt = Counter(values)  # число повторений каждого значения: c_v
    total = sum(cnt.values())  # всего значений N
    return -sum((c / total) * math.log2(c / total) for c in cnt.values())  # энтропия H = −Σ_v (c_v/N)·log2(c_v/N)


def build_residual_profile(  # посчитать профиль остатка по обучающим парам
    train_x: Sequence[Sequence[int]],  # аргументы: список кортежей
    train_y: Sequence[int],  # ответы
    domain: Iterable[Sequence[int]] | None = None,  # область определения (в расчёте не используется)
    builtin_fit: Callable[[Sequence, Sequence], float] | None = None,  # встроенная подгонка: доля необъяснённого; None — считать 1.0
) -> ResidualProfile:  # готовый профиль
    """Профиль остатка задачи по обучающим парам: разреженность, симметрии, период.

        x_i = train_x[i][0], если аргумент один, иначе сумма аргументов. Все доли считаются
        по таблице пар (x, y), отсортированной по x."""
    n = len(train_x)  # число примеров n
    flat_x = [x[0] if len(x) == 1 else int(sum(x)) for x in train_x]  # x_i = x_i[0] при одном аргументе, иначе Σ_j x_i[j]
    uniq_out = len(set(train_y))  # число различных ответов K
    # разреженность: отрезки ненулевых ответов по сортированным x
    pairs = sorted(zip(flat_x, train_y))  # таблица пар (x, y), сортированная по x
    active = [x for x, y in pairs if y != 0]  # точки с ненулевым ответом
    gaps = [b - a for a, b in zip(active, active[1:])] if len(active) > 1 else []  # расстояния g_i = a_{i+1} − a_i между соседними ненулевыми точками
    gap_ent = _entropy(gaps)  # энтропия распределения расстояний
    runs = (1 if active else 0)  # если ненулевые точки есть — минимум один отрезок
    runs += sum(1 for a, b in zip(active, active[1:]) if b != a + 1)  # +1 за каждый разрыв: a_{i+1} ≠ a_i + 1
    # автокорреляции
    m_by_x = dict(pairs)  # таблица x -> y
    shift_hit = tot = 0  # число попаданий и число проверенных x для сдвиговой пробы
    for x, y in pairs:  # по таблице
        if (x - 1) in m_by_x:  # точка x−1 тоже есть в таблице
            tot += 1  # проверенных: +1
            # равенство значений ИЛИ постоянная разность (счётчики):
            prev = m_by_x[x - 1]  # ответ в предыдущей точке: y(x−1)
            shift_hit += (prev == y) or ((y - prev) in (1, -1))  # +1, если y(x−1) = y или y − y(x−1) = ±1
    autoc_shift = shift_hit / tot if tot else 0.0  # autocorr_shift = shift_hit / tot (0, если проверять нечего)
    xor_best = 0.0  # лучшая доля совпадений по-xor
    for m in range(1, min(16, 1 << 8)):  # перебор масок m = 1..15
        hit = tot2 = 0  # попадания и проверки для этого m
        for x, y in pairs:  # по таблице
            xm = x ^ m  # точка с наложенной маской: x ⊕ m
            if xm in m_by_x:  # точка x ⊕ m есть в таблице
                tot2 += 1  # проверенных: +1
                hit += (m_by_x[xm] == y)  # +1, если y(x ⊕ m) = y(x)
        if tot2:  # были проверки
            xor_best = max(xor_best, hit / tot2)  # максимум доли hit/tot2 по m
    refl_hit = refl_tot = 0  # попадания и проверки для пробы отражения
    C = (max(flat_x) if flat_x else 0)  # центр отражения C = max x
    for x, y in pairs:  # по таблице
        xr = C - x  # отражённая точка x' = C − x
        if xr in m_by_x:  # она есть в таблице
            refl_tot += 1  # проверенных: +1
            refl_hit += (m_by_x[xr] == y)  # +1, если y(C − x) = y(x)
    autoc_refl = refl_hit / refl_tot if refl_tot else 0.0  # доля отражённых совпадений: refl_hit / refl_tot
    residual = builtin_fit(train_x, train_y) if builtin_fit else 1.0  # доля необъяснённого встроенной подгонкой; без неё — 1.0
    # доля переключений ответа между соседними x: у плотной случайной таблицы она
    # около (K−1)/K, у настоящих законов переключения редки
    adj = [(b[1], pairs[i + 1][1])  # пары ответов (y_i, y_{i+1}) для соседних точек
           for i, b in enumerate(pairs[:-1])  # по всем точкам, кроме последней
           if pairs[i + 1][0] == b[0] + 1]  # только если x_{i+1} = x_i + 1
    switch_rate = (sum(1 for a, b in adj if a != b) / len(adj)) if adj else 0.0  # доля переключений: |{y_i ≠ y_{i+1}}| / |adj|
    # период ищется только на плотной целой сетке (в разреженной таблице выровненных
    # пар слишком мало — попадание остаётся 0, то есть «неизвестно»). Период засчитывается
    # только по информативным парам, где хотя бы один конец не равен самому частому
    # ответу: совпадение двух самых частых значений ничего не доказывает (на таблице с
    # 2.6% ненулевых доля таких пар была 0.95 при любом P). Исключать само значение
    # «ноль» нельзя — в двоичном алфавите 0 законный символ. Доля информативных пар
    # сообщается отдельно, чтобы вызывающий код видел, какая часть таблицы покрыта периодом
    period_hit, period, period_cover = 0.0, 0, 0.0  # пока периода нет
    if len(pairs) >= 128 and all(b[0] == a[0] + 1 for a, b in zip(pairs, pairs[1:])):  # не меньше 128 точек и x идут подряд без пропусков
        ys = [y for _, y in pairs]  # ответы по порядку x
        top = Counter(ys).most_common(1)[0][0]  # самый частый ответ top
        for P in range(2, min(64, len(ys) // 2) + 1):  # перебор периодов P = 2..min(64, n/2)
            inf = [i for i in range(len(ys) - P) if not (ys[i] == top and ys[i + P] == top)]  # информативные индексы: не (y_i = top и y_{i+P} = top)
            if len(inf) < 32:                   # доказательств слишком мало
                continue  # период P не оценивается
            hit = sum(1 for i in inf if ys[i] == ys[i + P]) / len(inf)  # доля совпадений: hit(P) = |{i ∈ inf : y_i = y_{i+P}}| / |inf|
            if hit > period_hit:  # лучший период
                period_hit, period = hit, P  # запомнить долю и период
                period_cover = len(inf) / (len(ys) - P)  # доля покрытия: |inf| / (n − P)
    return ResidualProfile(  # собрать профиль
        n=n,  # число примеров
        k_density=len(active) / n if n else 0.0,  # доля ненулевых: k_density = |active| / n
        uniq_outputs=uniq_out,  # число различных ответов
        sparsity_gaps_entropy=gap_ent,  # энтропия расстояний
        n_runs=runs,  # число отрезков
        autocorr_shift=autoc_shift,  # доля сдвиговых совпадений
        autocorr_reflect=autoc_refl,  # доля совпадений при отражении
        autocorr_xor=xor_best,  # лучшая доля совпадений по ⊕
        residual_frac_builtin=min(1.0, max(0.0, residual)),  # доля необъяснённого, обрезанная в [0, 1]
        output_switch_rate=round(switch_rate, 4),  # доля переключений с 4 знаками
        shift_chance=round(3.0 / max(1, uniq_out), 4),  # shift_chance = 3 / max(1, K) с 4 знаками
        period_hit=round(period_hit, 4),  # лучшая доля периодических совпадений с 4 знаками
        period=period,  # найденный период
        period_cover=round(period_cover, 4),  # покрытие периода с 4 знаками
    )


def classify_hold(reason_str: str, profile: ResidualProfile) -> HoldReason:  # причина задержки по тексту причины и профилю
    """Причина задержки: по исходному тексту причины и профилю остатка.

        Порядок: бюджет → недетерминированность → шум → пробел представления."""
    r = (reason_str or "").upper()  # текст причины заглавными буквами (пустая строка, если None)
    if "BUDGET" in r or "NODE" in r or "ALPHA" in r:  # в тексте есть слово бюджета, узлов или альфы
        return HoldReason.BUDGET_EXHAUSTED  # причина — исчерпанный бюджет
    if "NONDET" in r or "PARTIAL" in r or "CLOCK" in r:  # недетерминированность, частичность или часы
        return HoldReason.NONDETERMINISTIC_OR_PARTIAL  # причина — недетерминированный или частичный учитель
    # Проверка на шум. Случайная плотная таблица: нет симметрий, встроенная подгонка не
    # объясняет её, и ответы почти все разные (закон повторяет значения, шум из широкого
    # алфавита — нет). Проба соседних переключений различителем не является — законы
    # с малым периодом переключаются сами по себе.
    uniq_ratio = profile.uniq_outputs / max(1, profile.n)  # uniq_ratio = K / max(1, n) (в решении ниже не используется)
    # Проверка на шум (по тем же признакам): случайная плотная таблица над алфавитом из K
    # значений содержит почти все K значений (uniq/K ≈ 1), компактный закон занимает узкий
    # кусок алфавита. Проба соседних переключений различителем не является — законы с
    # малым периодом переключаются естественно (x mod 7 переключается в 6/7 случаев).
    # Порог сдвиговой пробы отсчитывается от её случайного уровня (3/K при K разных
    # ответах), но не ниже 0.3 для широких алфавитов; периодическая таблица шумом не считается.
    shift_noise = max(0.3, 1.5 * profile.shift_chance)  # порог шума: shift_noise = max(0.3, 1.5·shift_chance)
    no_structure = (profile.autocorr_shift < shift_noise  # нет структуры, если сдвиг ниже случайного уровня
                    and profile.autocorr_xor < 0.3  # и xor-совпадений меньше 0.3
                    and profile.autocorr_reflect < 0.3  # и отражённых совпадений меньше 0.3
                    and profile.period_hit < 0.95)  # и период не выражен (< 0.95)
    if (profile.residual_frac_builtin >= 0.9  # встроенная подгонка объясняет меньше 10%
            and no_structure  # и структуры нет
            and profile.uniq_outputs > 2  # и разных ответов больше 2
            and profile.output_switch_rate >= 0.7):  # и переключения в 70% случаев и чаще
        # доля переключений поддерживает гипотезу шума; итоговая проверка покрытия
        # алфавита по max_y делается вызывающим кодом — здесь используется только
        # отношение uniq_outputs к n для плотного случая
        return HoldReason.RANDOM_INCOMPRESSIBLE  # случайный шум
    if (profile.residual_frac_builtin >= 0.9 and no_structure):  # встроенная подгонка бессильна и структуры нет
        # ни одна встроенная подгонка не объясняет, симметрий нет: либо пробел представления, либо шум.
        # У плотного случайного шума uniq_outputs велико и отрезков около n/2;
        # у структурных пробелов обычно есть какая-то симметрия или разреженность.
        if (profile.k_density > 0.6 and profile.n_runs > profile.n * 0.4  # плотность > 0.6 и отрезков больше 0.4·n
                and profile.sparsity_gaps_entropy > 2.5  # и энтропия расстояний > 2.5
                and profile.uniq_outputs > 2):  # и разных ответов больше 2
            return HoldReason.RANDOM_INCOMPRESSIBLE  # плотный хаос без структуры — шум
        return HoldReason.REPRESENTATION_GAP  # иначе — пробел представления
    return HoldReason.REPRESENTATION_GAP  # не шум и не бюджет — пробел представления
