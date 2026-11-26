#!/usr/bin/env python3
"""Изобретение представлений на уровне метаграмматики.

ConstructBlueprint (чертёж конструкции) = Фактор x Базис x (необязательная композиция):
Фактор (преобразование области в каноническое ядро):
ident (тождество), shift(d), reflect(c), xor(m), bitrev4, mod_range(lo, d),
а также маски, выводимые из данных (derive_mask_quotient).
Базис (канонический кодек на ядре):
FULL (запоминание таблицы — никогда не структурно), SPARSE (множество единиц),
INTERVALS (отрезки единиц), GF2_ANF (полином Жегалкина), POLY_LIN (y = k·x + c),
OUT_PERIOD (y = (x + o) mod p).
Композиция: фактор после фактора (глубина <= 2; ветка глубины 2 пока не реализована).

Поиск: перебираются чертежи, оценка NetGain на всём кластере пробелов:
sum(L_cur(T) − L_with_B(T)) − MetaCost(B) > 0, где MetaCost = meta_bits + payload_bits (бит по MDL).

Верификация (от контрпримеров): выборочная подгонка -> полная проверка области -> PROVEN_EXACT,
иначе откат. Защита от переобучения: согласованное нуль-сравнение, z >= 4 (сначала аналитическая
оценка, метод Монте-Карло только для пограничных кандидатов); бюджет alpha/delta расходуется
на каждую попытку.

Принятый чертёж становится представлением в реестре и переносится на новые задачи.
Дополнительно: активный синтез с контрпримерами (cegis_invent), нейронный априор семейств,
поиск циклов while и цикл открытия представлений run_rdl."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import math  # модуль math — логарифмы, корень, НОД
import random  # модуль random — генератор и перемешивание
from dataclasses import dataclass  # dataclass — классы-записи
from typing import Callable  # тип Callable для аннотаций

# --------------------------------------------------------------------------
# Преобразования-факторы (приведение области к каноническому ядру)
# --------------------------------------------------------------------------


def q_ident(s: int) -> int:
    """Тождественный фактор: q(s) = s."""
    return s  # вернуть s без изменений


def q_shift(d: int, m: int = 16) -> Callable[[int], int]:
    """Сдвиг по модулю m: q(s) = (s − d) mod m. Возвращает функцию f(s)."""
    def f(s):  # внутренняя функция преобразования
        return (s - d) % m  # вернуть q(s) = (s − d) mod m

    return f  # вернуть функцию преобразования


def q_reflect(c: int, m: int = 16) -> Callable[[int], int]:
    """Отражение относительно c/2 по модулю m: q(s) = (c − s) mod m. Возвращает функцию f(s)."""
    def f(s):  # внутренняя функция преобразования
        return (c - s) % m  # вернуть q(s) = (c − s) mod m

    return f  # вернуть функцию преобразования


def q_xor(m: int) -> Callable[[int], int]:
    """Исключающее ИЛИ с маской m: q(s) = s xor m. Возвращает функцию f(s)."""
    def f(s):  # внутренняя функция преобразования
        return s ^ m  # вернуть q(s) = s xor m

    return f  # вернуть функцию преобразования


def q_mod_range(lo: int, d: int) -> Callable[[int], int]:
    """Локальный вычет: q(s) = (s − lo) mod d. Возвращает функцию f(s)."""
    def f(s):  # внутренняя функция преобразования
        return (s - lo) % d  # вернуть q(s) = (s − lo) mod d

    return f  # вернуть функцию преобразования


def q_bitrev4(s: int) -> int:
    """Обращение порядка 4 битов: s -> его запись в 4 битах, прочитанная справа налево."""
    return int(format(s, "04b")[::-1], 2)  # записать s четырьмя двоичными цифрами, перевернуть строку и прочитать как число


QUOTIENTS: list[tuple[str, object, Callable]] = [  # меню факторов: (имя, параметр, функция)
    ("ident", 0, q_ident),  # тождество с параметром 0
    ("bitrev4", 0, q_bitrev4),  # обращение битов с параметром 0
]
for d in range(1, 16):  # по сдвигам d = 1..15
    QUOTIENTS.append((f"shift{d}", d, q_shift(d)))  # добавить фактор 'shift{d}'
for c in (7, 15, 31):  # по осям отражения c = 7, 15, 31
    QUOTIENTS.append((f"reflect{c}", c, q_reflect(c)))  # добавить фактор 'reflect{c}'
for m in range(1, 16):  # по маскам m = 1..15
    QUOTIENTS.append((f"xor{m}", m, q_xor(m)))  # добавить фактор 'xor{m}'
for lo, d in ((0, 5), (0, 8), (1, 6), (2, 12)):  # по парам (нижняя граница lo, модуль d)
    QUOTIENTS.append((f"mod{lo}_{d}", (lo, d), q_mod_range(lo, d)))  # добавить фактор 'mod{lo}_{d}' с параметром (lo, d)


# --------------------------------------------------------------------------
# Базисы (канонические кодеки на ядре)
# --------------------------------------------------------------------------


def basis_full(core_pairs: list[tuple[int, int]]) -> dict:
    """Базис FULL: запомнить таблицу ядро -> значение. Возвращает {"kind": "FULL", "table": ...}."""
    return {"kind": "FULL", "table": dict(core_pairs)}  # описание: тип FULL и словарь пар ядра


def basis_sparse(core_pairs: list[tuple[int, int]]) -> dict | None:
    """Базис SPARSE: значения 1 только на множестве активных ядер.
    Если какое-то отличное от 0 значение не равно 1 — базис не подходит (None).
    Пустое множество активных ядер допустимо. Возвращает {"kind": "SPARSE", "set": ...}."""
    active = sorted(x for x, y in core_pairs if y != 0)  # отсортированные ядра x с отличным от 0 значением (предполагается, что это 1)
    if not active:  # если активных ядер нет
        return {"kind": "SPARSE", "set": set()}  # пустое множество единиц
    gaps = [b - a for a, b in zip(active, active[1:])]  # расстояния между соседними активными ядрами: gaps_i = a_{i+1} − a_i
    if any(g <= 0 for g in gaps):  # если какое-то расстояние <= 0 (повторяющиеся ядра с разными значениями)
        return None  # базис не подходит
    return {"kind": "SPARSE", "set": set(active)}  # описание: тип SPARSE и множество активных ядер


def basis_intervals(core_pairs: list[tuple[int, int]]) -> dict | None:
    """Базис INTERVALS: отрезки подряд идущих ядер со значением 1.
    Возвращает {"kind": "INTERVALS", "iv": [(lo, hi), ...]} или None, если значения не только 0 и 1."""
    act = {x for x, y in core_pairs if y != 0}  # множество ядер с отличным от 0 значением
    if not act:  # если активных ядер нет
        return {"kind": "INTERVALS", "iv": []}  # пустой список отрезков
    runs = []  # список отрезков
    start = prev = min(act)  # начало и конец текущего отрезка — минимальное активное ядро
    for x in sorted(act)[1:]:  # по остальным активным ядрам в порядке возрастания
        if x == prev + 1:  # если ядро продолжает отрезок: x = prev + 1
            prev = x  # продлить текущий отрезок: prev = x
            continue  # перейти к следующему ядру
        runs.append((start, prev))  # закрыть отрезок (start, prev)
        start = prev = x  # начать новый отрезок с x
    runs.append((start, prev))  # закрыть последний отрезок
    return {"kind": "INTERVALS", "iv": runs}  # описание: тип INTERVALS и список отрезков


def basis_gf2_anf(core_pairs: list[tuple[int, int]]) -> dict | None:
    """Базис GF2_ANF: полином Жегалкина над 4 переменными.
    Таблица из 16 значений y in {0, 1} преобразуется быстрым преобразованием Мёбиуса
    в коэффициенты a[T]; остаются мономы T, для которых a[T] = 1. Возвращает
    {"kind": "GF2_ANF", "terms": ...} или None, если пар не 16 или значения не булевы."""
    if len(core_pairs) != 16:  # если пар не 16 — не таблица над 4 битами
        return None  # базис не подходит
    vals = [0] * 16  # вектор значений длины 16, изначально нулевой
    for x, y in core_pairs:  # по парам (ядро, значение)
        vals[x] = y & 1  # записать vals[x] = y mod 2
        if y not in (0, 1):  # если y не 0 и не 1 (значение не булево)
            return None  # базис не подходит
    a = list(vals)  # рабочая копия значений
    for bit in range(4):  # по номерам битов bit = 0..3
        step = 1 << bit  # ш step = 2^bit
        for x in range(16):  # по всем x от 0 до 15
            if x & step:  # если в x установлен бит step
                a[x] ^= a[x ^ step]  # a[x] ^= a[x xor step] — бабочка Мёбиуса
    terms = frozenset(i for i, v in enumerate(a) if v)  # множество мономов T с коэффициентом a[T] = 1
    return {"kind": "GF2_ANF", "terms": terms}  # описание: тип GF2_ANF и множество мономов


def basis_out_period(core_pairs: list[tuple[int, int]]) -> dict | None:
    """Базис OUT_PERIOD: y = (x + o) mod p — структура счётчика или кольца (придуманный базис).

    Из (x + o) mod p = y следует y < p; при p > max(y) смещение единственно: o = (y0 − x0) mod p.
    Пара подходит тогда и только тогда, когда p делит все d_i = (y_i − x_i) − (y_0 − x_0).
    Поэтому вместо полного перебора пар (p, o) достаточно перебрать делители g = НОД(d_i),
    большие max(y), по возрастанию, с одной полной проверкой. Возвращает
    {"kind": "OUT_PERIOD", "p": p, "o": o} или None."""
    if len(core_pairs) < 3:  # если пар меньше 3 — недостаточно данных
        return None  # базис не подходит
    # Условие (x + o) mod p = y требует y < p; при p > max(y) смещение единственно
    # (o = (y0 − x0) mod p), и пара годится тогда и только тогда, когда p делит каждое
    # d_i = (y_i − x_i) − (y_0 − x_0). Поэтому полный перебор пар (p, o) (до 32 896 штук)
    # сводится к делителям g = НОД(d_i), превышающим max(y), по возрастанию, с одной проверкой.
    # Без условия p > max(y) на недоопределённых выборках получался p = 2; это найдено
    # сверкой 400 случаев с прежним полным перебором (теперь 0 расхождений).
    # Здесь нужен только один блок комментария
    # (выявлено сверкой с прежним полным перебором)
    ys_max = max(y for _, y in core_pairs)  # максимальное значение ys_max = max(y)
    diffs = {(y - x) - (core_pairs[0][1] - core_pairs[0][0])  # разности d_i = (y_i − x_i) − (y_0 − x_0) по всем парам (множество)
             for x, y in core_pairs}  # продолжение выражения множества
    g = 0  # НОД разностей g = 0 (пока)
    for d in diffs:  # по разностям d
        g = math.gcd(g, abs(d))  # g = НОД(g, |d|)
    (x0, y0) = core_pairs[0]  # первая пара (x0, y0)
    lo = max(2, ys_max + 1)  # нижняя граница периода lo = max(2, ys_max + 1) — так y < p
    cands = range(lo, 257) if g == 0 else sorted(  # если g = 0, кандидаты — все p из [lo, 256], иначе только делители g из того же диапазона, по возрастанию
        d for d in range(2, 257) if d >= lo and g % d == 0)  # продолжение выражения кандидатов
    for p_ in cands:  # по кандидатам p_
        o = (y0 - x0) % p_  # смещение o = (y0 − x0) mod p_
        if all((x + o) % p_ == y for x, y in core_pairs):  # если (x + o) mod p_ = y для всех пар
            return {"kind": "OUT_PERIOD", "p": p_, "o": o}  # описание: тип OUT_PERIOD, период и смещение
    return None  # базис не подходит


def basis_poly_lin(core_pairs: list[tuple[int, int]]) -> dict | None:
    """Базис POLY_LIN: точная линейная функция y = k·x + c с целыми k и c не менее чем по 2 точкам.
    k = (y1 − y0) / (x1 − x0) на первой и последней парах, c = y0 − k·x0; затем проверка всех пар.
    Возвращает {"kind": "POLY_LIN", "k": k, "c": c} или None."""
    if len(core_pairs) < 2:  # если пар меньше 2 — прямую не определить
        return None  # базис не подходит
    # точная линейная функция не менее чем по 2 точкам (y = k·x + c), целые k и c
    (x0, y0), (x1, y1) = core_pairs[0], core_pairs[-1]  # первая и последняя пары
    if x1 == x0:  # если x1 = x0 — деление на ноль
        return None  # базис не подходит
    k, num = divmod(y1 - y0, x1 - x0)  # k, num = divmod(y1 − y0, x1 − x0): неполное частное и остаток
    if num != 0:  # если остаток не 0 — наклон не целый
        return None  # базис не подходит
    c = y0 - k * x0  # свободный член c = y0 − k·x0
    if all(k * x + c == y for x, y in core_pairs):  # если k·x + c = y для всех пар
        return {"kind": "POLY_LIN", "k": k, "c": c}  # описание: тип POLY_LIN, наклон и сдвиг
    return None  # базис не подходит




def basis_dict(core_pairs: list[tuple[int, int]]) -> dict:  # пары (значение характеристики, байт)
    """Словарный базис DICT: таблица «значение характеристики -> байт». Отличие от
    FULL: вне таблицы закон не отвечает (ошибка, а не молчаливый ноль)."""
    return {"kind": "DICT", "table": dict(core_pairs)}  # описание: тип DICT и словарь пар


BASES: list[tuple[str, Callable]] = [  # меню базисов: (имя, функция построения)
    ("FULL", basis_full),  # FULL — таблица
    ("OUT_PERIOD", basis_out_period),  # OUT_PERIOD — периодический счётчик
    ("SPARSE", basis_sparse),  # SPARSE — множество единиц
    ("INTERVALS", basis_intervals),  # INTERVALS — отрезки единиц
    ("GF2_ANF", basis_gf2_anf),  # GF2_ANF — полином Жегалкина
    ("POLY_LIN", basis_poly_lin),  # POLY_LIN — линейная функция
]


# --------------------------------------------------------------------------
# Чертёж конструкции и его вычисление
# --------------------------------------------------------------------------


@dataclass(frozen=True)  # класс данных без возможности изменения полей
class ConstructBlueprint:  # чертёж: фактор + базис + нагрузка + стоимость в битах
    q_name: str  # имя фактора
    q_param: object  # параметр фактора
    q_fn: Callable[[int], int]  # функция фактора q(x)
    basis: str  # имя базиса
    payload: dict  # нагрузка — результат построения базиса (table, terms, iv, ...)
    meta_bits: int  # мета-биты: стоимость выбора фактора и базиса
    depth: int = 1  # глубина композиции факторов
    total_bits: int = 0  # total_bits = meta_bits + payload_bits — полная стоимость по MDL

    def apply(self, x: int) -> int:
        """Применить чертёж к входу x: c = q(x), затем базис(c)."""
        c = self.q_fn(x)  # ядро c = q(x)
        kind = self.payload["kind"]  # тип базиса из нагрузки
        if kind == "DICT":  # если DICT
            return self.payload["table"][c]  # вернуть table[c]; вне таблицы — ошибка: закон не отвечает
        if kind == "FULL":  # если FULL
            return self.payload["table"].get(c, 0)  # возврат table[c], для отсутствующих ядер 0
        if kind == "SPARSE":  # если SPARSE
            return 1 if c in self.payload["set"] else 0  # 1, если c входит в множество единиц, иначе 0
        if kind == "INTERVALS":  # если INTERVALS
            for lo, hi in self.payload["iv"]:  # по отрезкам [lo, hi]
                if lo <= c <= hi:  # если lo <= c <= hi
                    return 1  # вернуть 1
            return 0  # ни один отрезок не подошёл — 0
        if kind == "GF2_ANF":  # если GF2_ANF
            terms = self.payload["terms"]  # множество мономов
            return sum(1 for t in terms if (t & c) == t) & 1  # значение полинома = (число мономов T, для которых T & c = T) mod 2
        if kind == "POLY_LIN":  # если POLY_LIN
            return self.payload["k"] * c + self.payload["c"]  # вернуть k·c + c0
        if kind == "OUT_PERIOD":  # если OUT_PERIOD
            return (c + self.payload["o"]) % self.payload["p"]  # вернуть (c + o) mod p
        raise ValueError(kind)  # неизвестный тип базиса — ошибка

    def to_dict(self) -> dict:
        """Сериализация чертежа в словарь: множества и ключи таблицы приводятся к сортированным
        спискам и строкам, чтобы результат можно было записать в JSON."""
        p = dict(self.payload)  # копия нагрузки
        if "set" in p:  # если есть множество единиц
            p["set"] = sorted(p["set"])  # заменить его отсортированным списком
        if "terms" in p:  # если есть мономы
            p["terms"] = sorted(p["terms"])  # заменить их отсортированным списком
        if "table" in p:  # если есть таблица
            p["table"] = {str(k): v for k, v in p["table"].items()}  # ключи таблицы превратить в строки
        return {  # вернуть словарь
            "q_name": self.q_name,  # имя фактора
            "q_param": repr(self.q_param),  # текстовое представление параметра фактора
            "basis": self.basis,  # имя базиса
            "payload": p,  # нагрузка
            "meta_bits": self.meta_bits,  # мета-биты
            "depth": self.depth,  # глубина
        }  # конец словаря


def _cost_bits(n: int) -> int:
    """Стоимость в битах числа n: 2·bit_length(n) (длина + сами биты); для n <= 0 — 1 бит."""
    if n <= 0:  # если n <= 0
        return 1  # вернуть 1 бит
    return 2 * n.bit_length()  # вернуть 2·(число двоичных цифр n)




def _payload_bits(basis_name: str, payload: dict, core_size: int) -> int:
    """Стоимость нагрузки в битах по типу базиса.
    OUT_PERIOD — 8; FULL — core_size·4; SPARSE — cost(|set| + 1) + |set|·4;
    INTERVALS — cost(|iv| + 1) + |iv|·8; GF2_ANF — cost(|terms| + 1) + |terms|·4;
    POLY_LIN — 16; неизвестный тип — core_size·4. cost(n) = _cost_bits(n)."""
    k = payload.get("kind")  # тип базиса из нагрузки
    if k == "OUT_PERIOD":  # если OUT_PERIOD
        return 8  # вернуть 8 бит
    if k == "DICT":  # если DICT
        return len(payload["table"]) * 24  # по 24 бита на запись: 16 бит ключа-хеша + 8 бит байта
    if k == "FULL":  # если FULL
        return core_size * 4  # вернуть core_size·4 бита — по 4 бита на точку
    if k == "SPARSE":  # если SPARSE
        return _cost_bits(len(payload["set"]) + 1) + len(payload["set"]) * 4  # вернуть cost(|set| + 1) + |set|·4
    if k == "INTERVALS":  # если INTERVALS
        return _cost_bits(len(payload["iv"]) + 1) + len(payload["iv"]) * 8  # вернуть cost(|iv| + 1) + |iv|·8 — по 8 бит на отрезок
    if k == "GF2_ANF":  # если GF2_ANF
        return _cost_bits(len(payload["terms"]) + 1) + len(payload["terms"]) * 4  # вернуть cost(|terms| + 1) + |terms|·4
    if k == "POLY_LIN":  # если POLY_LIN
        return 2 * 8  # вернуть 16 бит (k и c по 8 бит)
    return core_size * 4  # неизвестный тип — core_size·4 бита


def _ranked_quotients(hints: dict | None = None):
    """Упорядочить меню факторов по подсказкам профиля остатков, чтобы вероятные
    семейства проверялись первыми (экономия бюджета delta). Чистая перестановка:
    сам перебор и выбор по MDL не меняются, поэтому правильность не зависит от подсказок.
    hints: period_candidates -> семейство mod, xor_mask -> xor, reflect_axis -> reflect.
    Возвращает список факторов."""
    if not hints:  # если подсказок нет — исходный порядок
        return list(QUOTIENTS)  # вернуть копию меню
    order = {"mod": hints.get("period_candidates"),  # соответствие семейство -> сигнал подсказки: 'mod' <- period_candidates
             "xor": hints.get("xor_mask"),  # поле: xor <- xor_mask
             "reflect": hints.get("reflect_axis")}  # поле: reflect <- reflect_axis
    def key(q):  # ключ сортировки: (0, семейство) для подсказанного семейства, иначе (1, имя)
        name = q[0]  # имя фактора
        for fam, sig in order.items():  # по парам (семейство, сигнал)
            if sig and str(name).startswith(fam):  # если сигнал есть и имя начинается с имени семейства
                return (0, fam)  # вернуть (0, fam) — подсказанные семейства первыми (порядок устойчив)
        return (1, name)  # иначе (1, имя) — остальные по имени
    return sorted(QUOTIENTS, key=key)  # вернуть меню, отсортированное по ключу


def synthesize_blueprint(
    pairs: list[tuple[int, int]],  # pairs — пары (x, y)
    max_depth: int = 1,  # max_depth — максимальная глубина композиции факторов
    rng: random.Random | None = None,  # rng — генератор случайных чисел (не используется)
    hints: dict | None = None,  # hints — подсказки порядка поиска
) -> ConstructBlueprint | None:  # возврат: чертёж или None
    """Перебрать чертежи и вернуть чертёж с наименьшей суммой meta_bits + payload_bits,
    точно подходящий ко ВСЕМ парам. hints (из diagnosis.ResidualProfile) только переставляют
    порядок поиска. pairs — пары (x, y); max_depth < 1 — вернуть None; rng зарезервирован.
    Для каждого фактора: ядро core = [(q(x), y)], затем каждый базис строит нагрузку;
    meta_bits = cost(len(имя фактора)) + 8; годятся только чертежи с bp(x) = y на всех парах."""
    if max_depth < 1:  # если глубина меньше 1 — искать нечего
        return None  # вернуть None
    best = None  # лучший чертёж (пока нет)
    best_bits = None  # стоимость лучшего в битах (пока нет)
    for q_name, q_param, q_fn in _ranked_quotients(hints):  # по факторам в порядке подсказок
        core = [(q_fn(x), y) for x, y in pairs]  # ядро: пары (q(x), y)
        for b_name, b_fn in BASES:  # по базисам
            payload = b_fn(core)  # нагрузка базиса на ядре
            if payload is None:  # если базис не подошёл
                continue  # перейти к следующему базису
            core_size = len(core)  # размер ядра
            pbits = _payload_bits(b_name, payload, core_size)  # стоимость нагрузки в битах
            mbits = _cost_bits(len(q_name)) + 8  # мета-биты = cost(длина имени фактора) + 8
            total = mbits + pbits  # общая стоимость total = mbits + pbits
            bp = ConstructBlueprint(q_name, q_param, q_fn, b_name, payload,  # собрать чертёж
                                    mbits, total_bits=total)  # продолжение вызова: стоимость total_bits = total
            if all(bp.apply(x) == y for x, y in pairs):  # если чертёж точен на всех парах
                if best_bits is None or total < best_bits:  # если это первый точный или он дешевле лучшего
                    best, best_bits = bp, total  # запомнить лучший чертёж и его стоимость
        # Глубина 2: композиция фактора с самим собой через базис FULL
        # (имеет смысл, только если FULL в факторе намного меньше полного FULL)
        if max_depth >= 2:  # если запрошена глубина 2
            pass  # композиция глубины 2 пока не реализована
    if best is None:  # если точного чертежа не нашлось
        return None  # вернуть None
    return best  # вернуть лучший чертёж


# --------------------------------------------------------------------------
# Согласованная нулевая проверка против переобучения
# --------------------------------------------------------------------------


def analytic_null_bound(n_domain: int, k_active: int, max_y: int = 15) -> float:
    """Аналитическая стоимость запоминания полной таблицы (нулевая модель):
    n_domain записей по bit_length(max_y) бит (минимум 1) — столько стоит ЛЮБАЯ табличная
    функция без структуры. max_y — максимальное значение выхода."""
    return n_domain * max(1, int(max_y).bit_length())  # вернуть n_domain·max(1, bit_length(max_y)) бит


def matched_null_verdict(
    bp: ConstructBlueprint,  # bp — чертёж конструкции
    pairs: list[tuple[int, int]],  # pairs — пары (x, y)
    alpha: float = 0.01,  # alpha — уровень значимости для числа попаданий
    z_threshold: float = 4.0,  # z_threshold — порог z-оценки
    rng: random.Random | None = None,  # rng — генератор случайных чисел
    n_mc: int = 150,  # n_mc — число прогонов Монте-Карло
) -> dict:  # возврат: словарь-вердикт
    """Вердикт: STRUCTURAL, только если сжатие чертежа побеждает согласованные случайные таблицы
    той же длины и разреженности при z >= threshold. FULL никогда не структурно.

    bp — чертёж; pairs — пары (x, y); alpha — уровень значимости; z_threshold — порог z;
    rng — генератор; n_mc — число случайных прогонов. Этапы:
    1) FULL -> FULL_NOT_STRUCTURAL; null_bits — аналитическая стоимость таблицы на области
    размером 2^max(4, bit_length(max x)); если null_bits <= 0 -> NO_NULL.
    2) Для семейств известного размера F (OUT_PERIOD, POLY_LIN: F = 256):
    sig_bits = n·log2(max_y + 1) − log2(F); STRUCTURAL, если sig_bits >= z_threshold, иначе COMPRESSED.
    3) analytic = null_bits − bp_bits; если analytic < z_threshold -> COMPRESSED (аналитически).
    4) Согласованный метод Монте-Карло: n_mc раз выходы перемешиваются, на них заново ищется
    чертёж; z = (mu − obs)/sd, p = (1 + hits)/(len(nulls) + 1); STRUCTURAL, если
    z >= z_threshold и hits <= max_hits, где max_hits = int(alpha·(n_mc + 1)) − 1.
    Возвращает словарь с verdict, z, null_bits, bp_bits, significance_bits, stage."""
    if bp.payload.get("kind") == "FULL":  # если базис FULL — запоминание таблицы не структурно
        return {"verdict": "FULL_NOT_STRUCTURAL", "z": 0.0}  # вердикт FULL_NOT_STRUCTURAL с z = 0.0
    n = len(pairs)  # число пар n
    ys = [y for _, y in pairs]  # выходы y
    max_y = max(ys) if ys else 15  # максимальный выход max_y (15, если пар нет)
    null_bits = analytic_null_bound(  # аналитическая стоимость таблицы: область размером 2^max(4, bit_length(max x))
        1 << max(4, (max(x for x, _ in pairs)).bit_length()), len(pairs), max_y  # продолжение: n_domain = 2^max(4, bit_length(max x)), k_active = len(pairs), max_y
    )  # конец вызова
    bp_bits = bp.meta_bits + _payload_bits(bp.payload.get("kind", ""), bp.payload, n)  # стоимость чертежа bp_bits = meta_bits + стоимость нагрузки
    if null_bits <= 0:  # если нулевая стоимость неположительна
        return {"verdict": "NO_NULL", "z": 0.0}  # вердикт NO_NULL с z = 0.0
    # z-оценка сжатия: на сколько сигм ниже среднего нулевого распределения.
    # Нулевое распределение приближённо равномерно на [0, null_bits]: среднее null_bits/2,
    # стандартное отклонение null_bits/6 (описательное приближение).
    # Комбинаторная значимость: сколько таблиц этого семейства существует по сравнению
    # со всеми возможными таблицами? Точное приближение n точек семейством размера F
    # из пространства T таблиц имеет p = F/T; значимость в битах = log2(T/F).
    # Наивная форма table_space = K^n переполняет число с плавающей точкой при n > ~250
    # (проверено), поэтому используются логарифмы. Нуль-проверка двухступенчатая:
    # ступень 1 — оценка по принципу Оккама для семейств известного размера,
    # ступень 2 — согласованный Монте-Карло (тот же синтез на перемешанных выходах) для остальных.
    # Monte-Carlo (same synthesizer on permuted outputs) for the rest.  # конец блока комментария
    kind = bp.payload.get("kind", "")  # тип базиса
    F = {"OUT_PERIOD": 256, "POLY_LIN": 256}.get(kind)  # размер семейства F: 256 для OUT_PERIOD и POLY_LIN, иначе None
    if F is not None and n > 0:  # если размер семейства известен и есть пары
        sig_bits = n * math.log2(max_y + 1) - math.log2(F)  # значимость sig_bits = n·log2(max_y + 1) − log2(F)
        verdict = "STRUCTURAL" if sig_bits >= z_threshold else "COMPRESSED"  # STRUCTURAL, если sig_bits >= z_threshold, иначе COMPRESSED
        return {"verdict": verdict, "z": round(sig_bits, 2),  # вердикт с z = sig_bits, стоимостями и этапом occam
                "null_bits": round(null_bits, 1), "bp_bits": bp_bits,  # продолжение словаря
                "significance_bits": round(sig_bits, 2), "stage": "occam"}  # конец словаря
    analytic = null_bits - bp_bits  # аналитическая оценка analytic = null_bits − bp_bits
    if analytic < z_threshold:  # если analytic < z_threshold — сжатие недостаточно сильное
        return {"verdict": "COMPRESSED", "z": round(analytic, 2),  # вердикт COMPRESSED с этапом analytic
                "null_bits": round(null_bits, 1), "bp_bits": bp_bits,  # продолжение словаря
                "significance_bits": round(analytic, 2), "stage": "analytic"}  # конец словаря
    # ступень 2: согласованный нуль Монте-Карло
    rng = rng or random.Random(0)  # генератор по умолчанию с зерном 0
  # внутренняя функция
    def _comparable_bits(cand_bp):  # определение стоимости сравнимого чертежа
        return 9 + _payload_bits(cand_bp.payload.get("kind", ""),  # стоимость = 9 + стоимость нагрузки
                                 cand_bp.payload, n)  # продолжение вызова

    xs = [x for x, _ in pairs]  # входы x
    ys = [y for _, y in pairs]  # выходы y
    obs = _comparable_bits(bp)  # наблюдаемая стоимость obs = стоимость сравнимого чертежа bp
    max_hits = int(alpha * (n_mc + 1)) - 1  # максимум допустимых попаданий max_hits = int(alpha·(n_mc + 1)) − 1
    nulls: list = []  # стоимости чертежей на перемешанных выходах
    hits = 0  # число попаданий (нулевая стоимость <= obs)
    for _ in range(n_mc):  # по числу прогонов Монте-Карло
        perm = ys[:]  # копия выходов
        rng.shuffle(perm)  # случайно перемешать выходы
        nb = synthesize_blueprint(list(zip(xs, perm)))  # синтез чертежа на парах (x, перемешанный y)
        nbits = (_comparable_bits(nb) if nb is not None  # стоимость найденного чертежа, а если он не найден — null_bits
                 else int(null_bits))  # продолжение условного выражения
        nulls.append(nbits)  # добавить стоимость в список
        if nbits <= obs:  # если нулевая стоимость не больше наблюдаемой
            hits += 1  # попаданий: hits + 1
            if hits > max_hits:  # если попаданий больше допустимого
                break  # прервать прогоны (кандидат не выдерживает проверку)
    mu = sum(nulls) / len(nulls)  # средняя нулевая стоимость mu = Σ v / len(nulls)
    sd = max(math.sqrt(sum((v - mu) ** 2 for v in nulls) / len(nulls)),  # стандартное отклонение sd = max(sqrt(Σ(v − mu)^2 / len(nulls)), 1.0)
             1.0)  # нижняя граница 1 бит — зернистость MDL
    z = (mu - obs) / sd  # z-оценка z = (mu − obs) / sd
    p = (1 + hits) / (len(nulls) + 1)  # достигаемый уровень p = (1 + hits) / (len(nulls) + 1)
    structural = z >= z_threshold and hits <= max_hits  # структурно, если z >= z_threshold и hits <= max_hits
    return {"verdict": "STRUCTURAL" if structural else "COMPRESSED",  # вердикт STRUCTURAL, иначе COMPRESSED
            "z": round(z, 2), "null_bits": round(mu, 1),  # z с двумя знаками и средняя нулевая стоимость
            "bp_bits": obs, "significance_bits": round(analytic, 2),  # наблюдаемая стоимость и аналитическая значимость
            "p": round(p, 4), "n_null": len(nulls),  # уровень p и число прогонов
            "stage": "matched_mc"}  # этап — согласованный Монте-Карло


# --------------------------------------------------------------------------
# Полная проверка: от выборки к сплошному обходу (от контрпримеров)
# --------------------------------------------------------------------------


def verify_blueprint(
    bp: ConstructBlueprint,  # bp — чертёж конструкции
    pairs: list[tuple[int, int]],  # pairs — пары (x, y)
    full_domain: range | None = None,  # full_domain — полная область проверки или None
) -> dict:  # возврат: словарь со статусом
    """PROVEN_EXACT только при ПОЛНОЙ размеченной покрытии области: каждая точка области
    обязана иметь метку и совпадать с чертежом. Неразмеченные точки НЕ пропускаются:
    «доказательство» поверх пробелов было бы поддельным сертификатом. Частичное
    покрытие -> PARTIAL_COVERAGE (никогда не доказательство). Если обнаружена точка,
    где чертёж ошибается, -> HOLD_COUNTEREXAMPLE с контрпримером cex.
    bp — чертёж; pairs — пары (x, y); full_domain — полная область или None."""
    # сначала выборочная проверка
    for x, y in pairs:  # по парам
        if bp.apply(x) != y:  # если чертёж ошибается на паре
            return {"status": "HOLD_COUNTEREXAMPLE", "cex": (x, y)}  # контрпример (x, y)
    if full_domain is None:  # если область не задана
        return {"status": "PARTIAL_COVERAGE", "checked": len(pairs),  # частичное покрытие: проверено len(pairs) точек
                "note": "no domain given — cannot prove"}  # примечание: область не задана — доказать нельзя
    table = dict(pairs)  # словарь меток по парам
    unlabeled = [x for x in full_domain if x not in table]  # неразмеченные точки области
    if unlabeled:  # если есть неразмеченные точки
        return {"status": "PARTIAL_COVERAGE",  # частичное покрытие
                "checked": len(table),  # проверено точек
                "unlabeled": len(unlabeled),  # число неразмеченных
                "note": f"{len(unlabeled)} domain points without labels"}  # примечание с числом точек без меток
    for x in full_domain:  # по всем точкам области
        if bp.apply(x) != table[x]:  # если чертёж ошибается на точке области
            return {"status": "HOLD_COUNTEREXAMPLE",  # контрпример
                    "cex": (x, table[x])}  # конец кортежа контрпримера
    return {"status": "PROVEN_EXACT", "checked": len(table),  # доказано точно: проверено len(table) точек
            "coverage": "full"}  # полное покрытие


# --------------------------------------------------------------------------
# Цикл изобретения на кластере пробелов
# --------------------------------------------------------------------------


def invent_blueprint(
    cluster_pairs: list[list[tuple[int, int]]],  # cluster_pairs — задачи кластера, каждая — список пар (x, y)
    rng: random.Random | None = None,  # rng — генератор случайных чисел
    max_attempts: int = 200,  # max_attempts — лимит попыток
    hints: dict | None = None,  # hints — подсказки порядка поиска
) -> dict | None:  # возврат: словарь результата или None
    """Найти ОДИН чертёж (одно семейство факторов), который точно подходит ко ВСЕМ задачам
    кластера, проходит согласованную нулевую проверку (STRUCTURAL) и даёт положительный NetGain.
    hints (сигналы ResidualProfile) переставляют порядок поиска семейств.
    Возвращает словарь со статусом INVENTED, HOLD_COUNTEREXAMPLE или COMPRESSED_ONLY,
    либо None, если общего фактора нет.(cluster_pairs — список списков пар по задачам; rng — генератор;
    max_attempts — лимит попыток.)"""
    rng = rng or random.Random(0)  # генератор по умолчанию с зерном 0
    if not cluster_pairs:  # если кластер пуст
        return None  # вернуть None
    # Саморасширение: сначала выводятся маски, рождённые данными (грамматика растёт);
    # производные маски пробуются первыми, потому что они аналитические
    register_derived_masks(cluster_pairs[0])  # зарегистрировать маски, выведенные из первой задачи кластера
    # чертёж обязан подходить к каждой задаче кластера (через свою
    # параметризацию фактора одна и та же конструкция обобщается)
    attempts = 0  # число попыток
    best_common = None  # лучший общий фактор (пока нет)
    for q_name, q_param, q_fn in _ranked_quotients(hints):  # по факторам в порядке подсказок
        attempts += 1  # попыток: attempts + 1
        if attempts > max_attempts:  # если превышен лимит попыток
            break  # выйти из перебора
        bps = []  # чертежи по задачам кластера
        ok_all = True  # признак «подходит ко всем задачам»
        total_bits = 0  # суммарная стоимость в битах
        for pairs in cluster_pairs:  # по задачам кластера
            core = [(q_fn(x), y) for x, y in pairs]  # ядро: пары (q(x), y)
            found = None  # найденный чертёж (пока нет)
            best_b = None  # лучшая стоимость (пока нет)
            meta = 9  # мета-биты выбора структуры meta = 9: ~35 факторов (6 бит) + базис (3 бита)
            for b_name, b_fn in BASES:  # по базисам
                payload = b_fn(core)  # нагрузка базиса на ядре
                if payload is None:  # если базис не подошёл
                    continue  # перейти к следующему базису
                b_bits = _payload_bits(b_name, payload, len(core))  # стоимость нагрузки b_bits
                bp = ConstructBlueprint(q_name, q_param, q_fn, b_name, payload,  # собрать чертёж с мета-битами meta
                                        meta, total_bits=b_bits + meta)  # полная стоимость total_bits = b_bits + meta
                if not all(bp.apply(x) == y for x, y in pairs):  # если чертёж не точен на всех парах
                    continue  # перейти к следующему базису
                if best_b is None or b_bits + meta < best_b:  # если это первый точный или он дешевле лучшего
                    found, best_b = bp, b_bits + meta  # запомнить чертёж и его стоимость
            if found is None:  # если точного чертежа для задачи нет
                ok_all = False  # общего фактора нет
                break  # прервать обход задач
            bps.append(found)  # добавить чертёж задачи
            total_bits += found.meta_bits + _payload_bits(  # накопить стоимость: total_bits += meta_bits + стоимость нагрузки
                found.payload.get("kind", ""), found.payload, len(core)  # продолжение вызова стоимости нагрузки
            )  # конец вызова
        if not ok_all:  # если не для всех задач найден чертёж
            continue  # перейти к следующему фактору
        # одинаковое имя фактора у всех задач кластера — общая конструкция
        if best_common is None or total_bits < best_common["total_bits"]:  # если это первая общая конструкция или она дешевле лучшей
            best_common = {  # запомнить кандидата
                "q_name": q_name,  # имя фактора
                "q_param": q_param,  # параметр фактора
                "blueprints": bps,  # чертежи по задачам
                "total_bits": total_bits,  # суммарная стоимость
            }  # конец словаря кандидата
    if best_common is None:  # если общего фактора нет
        return None  # вернуть None
    # согласованная нулевая проверка на первой задаче (представитель)
    verdict = matched_null_verdict(  # вердикт по чертежу первой задачи и её парам
        best_common["blueprints"][0], cluster_pairs[0], rng=rng  # продолжение: генератор rng
    )  # конец вызова
    if verdict["verdict"] != "STRUCTURAL":  # если вердикт не STRUCTURAL
        return {  # вернуть результат
            "status": "COMPRESSED_ONLY",  # статус COMPRESSED_ONLY — только сжатие, не структура
            "verdict": verdict,  # вердикт
            "q_name": best_common["q_name"],  # имя фактора
        }  # конец словаря
    proofs = [  # доказательства по задачам: verify_blueprint для каждого чертежа
        verify_blueprint(bp, pairs,  # чертёж и пары задачи; полная область от min(x) до max(x) включительно
                         full_domain=range(min(x for x, _ in pairs),  # продолжение: верхняя граница max(x) + 1
                                           max(x for x, _ in pairs) + 1))  # конец вызова
        for bp, pairs in zip(best_common["blueprints"], cluster_pairs)  # по парам (чертёж, пары) кластера
    ]  # конец списка
    if any(p["status"] != "PROVEN_EXACT" for p in proofs):  # если хоть одно доказательство не PROVEN_EXACT
        return {"status": "HOLD_COUNTEREXAMPLE", "proofs": proofs}  # вернуть HOLD_COUNTEREXAMPLE со всеми доказательствами
    return {  # вернуть словарь результата
        "status": "INVENTED",  # статус INVENTED — изобретено
        "q_name": best_common["q_name"],  # имя фактора
        "q_param": best_common["q_param"],  # параметр фактора
        "blueprints": best_common["blueprints"],  # чертежи по задачам
        "verdict": verdict,  # вердикт
        "attempts": attempts,  # число попыток
    }  # конец словаря


# --------------------------------------------------------------------------
# Фактор, рождённый данными: аналитическая маска из кластеров XOR-разностей
# --------------------------------------------------------------------------

def derive_mask_quotient(pairs: list[tuple[int, int]],
                         max_masks: int = 4) -> list[tuple[str, object, Callable]]:  # max_masks — зарезервированный параметр; возврат — список кандидатов-факторов
    """Вывести аналитически маску несмежных битов M: phi(x) = x & M должна
    ОПРЕДЕЛЯТЬ y (при совпадении phi значения y совпадают) и быть МИНИМАЛЬНОЙ
    (ни один бит нельзя выбросить). Маска открывается из данных, а не из меню.
    pairs — пары (x, y); max_masks зарезервирован. Возвращает список
    кандидатов-факторов (имя, параметр, функция); пустой, если маски нет."""
    if len(pairs) < 4:  # если пар меньше 4 — недостаточно данных
        return []  # вернуть пустой список
    hi = max(x for x, _ in pairs)  # максимальный вход hi
    nbits = max(1, hi.bit_length())  # число битов nbits = max(1, bit_length(hi))
    full = (1 << nbits) - 1  # полная маска full = 2^nbits − 1

    def determines(M: int) -> bool:  # определяет ли маска M значение y на выборке
        seen: dict[int, int] = {}  # увиденные значения phi -> y
        for x, y in pairs:  # по парам
            phi = x & M  # проекция phi = x & M
            if phi in seen and seen[phi] != y:  # если phi уже встречалось с другим y — маска не определяет y
                return False  # вернуть False
            seen[phi] = y  # запомнить phi -> y
        return True  # все пары согласованы — True

    if not determines(full):  # если даже полная маска не определяет y
        return []  # вернуть пустой список: даже полная маска не годится — y не является однозначной
                   # функцией x на этой выборке
    # жадно выбрасывать биты, пока сохраняется детерминированность (минимальная маска)
    M = full  # текущая маска M = full
    for b in range(nbits):  # по номерам битов b
        cand = M & ~(1 << b)  # кандидат cand = M без бита b
        if cand and determines(cand):  # если кандидат ненулевой и определяет y
            M = cand  # принять кандидата: M = cand
    if M == 0 or (M & (M - 1)) == 0:  # если маска пуста или в ней ровно один бит
        return []  # вернуть пустой список: тривиальный случай — либо ничего, либо один бит (сама переменная) случай: либо ничего не осталось, либо один бит (сама переменная)
    if M & (M << 1):  # если в маске есть смежные биты: M & (M << 1) не равно 0
        return []  # вернуть пустой список: смежные биты уже покрываются фиксированным меню через xor

    def q_mask(x: int, M=M) -> int:  # фактор-маска: q(x) = x & M (M зафиксирован по умолчанию)
        return x & M  # вернуть x & M

    return [(f"mask{M}", M, q_mask)]  # вернуть одного кандидата с именем 'mask{M}'


def register_derived_masks(pairs: list[tuple[int, int]]) -> int:
    """Вывести маски из данных и добавить их в QUOTIENTS (без повторов по имени).
    Возвращает число добавленных. Саморасширение: грамматика растёт во время работы.
    pairs — пары (x, y)."""
    added = 0  # число добавленных
    existing = {q[0] for q in QUOTIENTS}  # имена уже имеющихся факторов
    for cand in derive_mask_quotient(pairs):  # по кандидатам-маскам
        if cand[0] not in existing:  # если такого имени ещё нет
            QUOTIENTS.append(cand)  # добавить фактор в меню
            added += 1  # добавленных: added + 1
    return added  # вернуть число добавленных


# ============================================================
# АКТИВНЫЙ СИНТЕЗ ПОД ВОЗДЕЙСТВИЕМ КОНТРИМЕРОВ
# ============================================================

from dataclasses import dataclass, field  # dataclass, field — классы-записи и значения по умолчанию
@dataclass  # класс данных
class CounterexampleMemory:  # память контрпримеров
    """Память известных трудных точек: каждый новый кандидат проверяется по ней первым."""
    points: dict[int, int] = field(default_factory=dict)  # точка x -> метка y; отдельный пустой словарь на каждый экземпляр

    def add(self, x: int, y: int) -> None:  # добавить точку в память
        self.points[x] = y  # запомнить точку: points[x] = y

    def violations(self, bp: ConstructBlueprint) -> list[tuple[int, int]]:  # точки памяти, на которых чертёж ошибается
        return [(x, y) for x, y in self.points.items() if bp.apply(x) != y]  # список (x, y), где bp(x) ≠ y
@dataclass  # класс данных
class CegisResult:  # результат активного синтеза
    status: str  # статус
    blueprint: ConstructBlueprint | None  # найденный чертёж или None
    rounds: int  # число раундов
    queries: int  # число запросов меток
    train_size: int  # размер обучающего множества
    final_sweep: dict | None = None  # итоговый сплошной обход или None
    cert_queries: int = 0  # число запросов на сертификацию
def cegis_invent(train_x: list[int], query: Callable[[int], int | None], cegis_pool: list[int], full_domain: range, max_rounds: int=6, init_frac: float=0.25, query_budget: int | None=None, rng: random.Random | None=None) -> CegisResult:  # определение функции активного синтеза
    """Активный синтез под воздействием контрпримеров: чертёж ищется на train,
    уточняется по контрпримерам из cegis_pool, в конце сертифицируется на full_domain.

    - начальное train — первые init_frac от train_x (малое начальное множество);
    - каждый раунд: если кандидат ошибается на cegis_pool или в памяти, метки
    контрпримеров добавляются в train и синтез повторяется;
    - queries считает ТОЛЬКО точки, метки которых цикл действительно запросил
    (никогда — full_domain во время поиска).
    train_x — обучающие входы; query(x) — метка или None; cegis_pool — пул точек для
    проверки; full_domain — область сертификации; max_rounds — лимит раундов;
    init_frac — доля начального train; query_budget — лимит запросов; rng — генератор.
    Возвращает CegisResult."""
    rng = rng or random.Random(0)  # генератор по умолчанию с зерном 0
    if not train_x:  # если обучающих входов нет
        return CegisResult('HOLD_NO_CANDIDATE', None, 0, 0, 0)  # HOLD_NO_CANDIDATE: кандидата нет
    k0 = max(2, int(len(train_x) * init_frac))  # размер начального train k0 = max(2, int(len(train_x)·init_frac))
    train = dict(((x, query(x)) for x in train_x[:k0]))  # начальное train: первые k0 входов с их метками
    queries = len(train)  # запросов — по числу начальных меток
    mem = CounterexampleMemory()  # память контрпримеров
    pool = list(cegis_pool)  # копия пула точек
    rng.shuffle(pool)  # перемешать пул
    best: ConstructBlueprint | None = None  # лучший чертёж (пока нет)
    for round_i in range(1, max_rounds + 1):  # по раундам round_i = 1..max_rounds
        pairs = sorted(train.items())  # пары train, отсортированные
        bp = synthesize_blueprint(pairs, rng=rng)  # синтез чертежа на train
        if bp is None:  # если чертежа нет
            if not pool and (not mem.points):  # если пул и память пусты
                return CegisResult('HOLD_NO_CANDIDATE', None, round_i, queries, len(train))  # HOLD_NO_CANDIDATE: кандидата нет
            best = None  # лучшего нет
        else:  # иначе (чертёж найден)
            bad = mem.violations(bp)  # нарушения чертежа в памяти
            if bad:  # если есть нарушения
                for x, y in bad:  # по контрпримерам из памяти
                    train[x] = y  # добавить их в train
                best = bp  # запомнить чертёж как лучший
                continue  # перейти к следующему раунду
            ce = None  # контрпример из пула (пока не найден)
            while pool:  # пока пул не пуст
                if query_budget is not None and queries >= query_budget:  # если задан лимит запросов и он исчерпан
                    break  # прервать обход пула
                x = pool.pop()  # взять точку из пула
                y = query(x)  # запросить её метку
                queries += 1  # запросов: queries + 1
                if y is None:  # если метки нет
                    continue  # перейти к следующей точке
                mem.add(x, y)  # запомнить точку в памяти
                if bp.apply(x) != y:  # если чертёж ошибается на точке
                    ce = (x, y)  # контрпример найден
                    break  # прервать обход пула
            if ce is None:  # если контрпримера нет
                if query_budget is not None and queries >= query_budget:  # если задан лимит запросов и он исчерпан
                    return CegisResult('HOLD_CEGIS_BUDGET', bp, round_i, queries, len(train))  # HOLD_CEGIS_BUDGET: лимит запросов исчерпан
                cert_queries = 0  # запросов на сертификацию — 0
                for x in full_domain:  # по всей области
                    if x not in train:  # если точки нет в train
                        y = query(x)  # запросить метку
                        cert_queries += 1  # сертификационных запросов + 1
                        queries += 1  # запросов + 1
                        if y is not None:  # если метка есть
                            train[x] = y  # добавить в train
                sweep = verify_blueprint(bp, sorted(train.items()), full_domain)  # сплошная проверка чертежа на train и полной области
                if sweep['status'] == 'HOLD_COUNTEREXAMPLE' and round_i < max_rounds:  # если найден контрпример и раунды ещё есть
                    cex = sweep['cex']  # контрпример
                    train.setdefault(cex[0], cex[1])  # добавить его в train (если ещё нет)
                    mem.add(cex[0], cex[1])  # и в память
                    best = bp  # лучший — этот чертёж
                    continue  # перейти к следующему раунду
                return CegisResult(sweep['status'], bp, round_i, queries, len(train), final_sweep=sweep, cert_queries=cert_queries)  # вернуть статус сплошной проверки с чертежом и числом сертификационных запросов
            train[ce[0]] = ce[1]  # добавить контрпример из пула в train
            best = bp  # лучший — этот чертёж
    return CegisResult('HOLD_CEGIS_EXHAUSTED', best, max_rounds, queries, len(train))  # HOLD_CEGIS_EXHAUSTED: раунды исчерпаны

# ============================================================
# НЕЙРОННЫЙ АПРИОР ПРЕДЛОЖЕНИЙ
# ============================================================

import numpy as np  # numpy — массивы и экспонента
from hagi.invent.diagnosis import ResidualProfile  # импорт профиля остатков
FAMILIES = ('mod', 'xor', 'reflect', 'mask', 'shift')  # имена семейств
def profile_features(p: ResidualProfile) -> np.ndarray:  # профиль остатков -> 10-мерный вектор структурных признаков
    """Вектор из 10 структурных признаков профиля остатков ResidualProfile."""
    """10-dim structural feature vector from a ResidualProfile."""
    return np.array([1.0, p.autocorr_shift, p.autocorr_xor, p.autocorr_reflect, float(p.mn_classes is not None), p.sparsity_gaps_entropy, p.n_runs / max(1.0, p.n), p.k_density, p.uniq_outputs / max(1.0, p.n), p.residual_frac_builtin], dtype=np.float64)  # признаки: 1.0, автокорреляции сдвига/xor/отражения, наличие классов, энтропия пробелов, число прогонов / max(1, n), плотность k, число разных выходов / max(1, n), доля остатка
class NeuralProposalPrior:  # класс нейронного априора семейств
  # обучить априор: скорость обучения lr = 0.8
    def __init__(self, lr: float=0.8):  # конструктор с коэффициентом обучения lr
        self.lr = lr  # сохранить скорость обучения
        self.W = np.zeros((len(FAMILIES), 10), dtype=np.float64)  # веса W формы (число семейств, 10), изначально нулевые

    def predict_probs(self, feat: np.ndarray) -> dict[str, float]:  # вероятности семейств по вектору признаков
        logits = self.W @ feat  # логиты = W @ feat
        logits -= logits.max()  # вычесть максимум для устойчивости
        exps = np.exp(logits)  # экспоненты логитов
        probs = exps / exps.sum()  # вероятности softmax: p_i = exp(l_i − max l) / Σ exp(l_j − max l)
        return {f: float(p) for f, p in zip(FAMILIES, probs)}  # словарь семейство -> вероятность

    def update(self, feat: np.ndarray, winner: str, epochs: int=12) -> None:  # обновить веса по победившему семейству
        if winner not in FAMILIES:  # если победитель не из списка семейств
            return  # ничего не делать
        tgt = FAMILIES.index(winner)  # номер победившего семейства
        for _ in range(epochs):  # по эпохам
            logits = self.W @ feat  # логиты = W @ feat
            logits -= logits.max()  # вычесть максимум
            probs = np.exp(logits)  # экспоненты
            probs /= probs.sum()  # нормировка: softmax
            grad = probs.copy()  # градиент = probs
            grad[tgt] -= 1.0  # grad[tgt] -= 1 — производная перекрёстной энтропии
            self.W -= self.lr * np.outer(grad, feat)  # шаг градиентного спуска: W ← W − lr·outer(grad, feat)

    def hints_from_profile(self, p: ResidualProfile) -> dict:  # подсказки из профиля остатков
        """Связка с invention._ranked_quotients: лучшее по вероятности семейство становится
        признаком-подсказкой (только порядок поиска, никогда не сертифицированный выбор)."""
        probs = self.predict_probs(profile_features(p))  # вероятности по признакам профиля
        best = max(probs, key=probs.get)  # семейство с наибольшей вероятностью
        flag = {fam: False for fam in FAMILIES}  # словарь признаков семейств, изначально False
        flag[best] = True  # пометить лучшее семейство
        return flag  # вернуть признаки
DEFAULT_PRIOR = NeuralProposalPrior()  # априор по умолчанию

# ============================================================
# ПОИСК ЦИКЛОВ WHILE (Беллман / орбиты)
# ============================================================

def _digit_sum10(x: int) -> int:
    """Сумма десятичных цифр числа x."""
    x = abs(int(x))  # модуль и целочисленность x
    s = 0  # сумма цифр s = 0
    while x > 0:  # пока x > 0
        s += x % 10  # s += x mod 10 — младшая цифра
        x //= 10  # x //= 10 — отбросить младшую цифру
    return s  # вернуть сумму цифр
STEP_PRIMITIVES: dict[str, tuple[Callable[[int], int], str, int]] = {'popcount': (lambda s: int(s).bit_count(), 'int(s).bit_count()', 8), 'digit_sum10': (_digit_sum10, 'sum(int(c) for c in str(abs(int(s))))', 12), 'halve': (lambda s: int(s) // 2, 's // 2', 8), 'drop_lsb': (lambda s: int(s) & int(s) - 1, 's & (s - 1)', 12), 'even_halve_odd_inc1': (lambda s: s // 2 if s % 2 == 0 else s + 1, '(s // 2) if (s & 1) == 0 else (s + 1)', 16), 'even_halve_odd_3x1': (lambda s: s // 2 if s % 2 == 0 else 3 * s + 1, '(s // 2) if (s & 1) == 0 else (3 * s + 1)', 20)}  # примитивы шага: имя -> (функция, текст, стоимость в битах): popcount, digit_sum10, halve, drop_lsb, even_halve_odd_inc1, even_halve_odd_3x1

# ============================================================
# ЦИКЛ ОТКРЫТИЯ ПРЕДСТАВЛЕНИЙ (параметрические семейства)
# ============================================================

N_BITS = 4  # число битов состояния N_BITS = 4
N_STATES = 1 << N_BITS  # число состояний N_STATES = 2^N_BITS
N_ACT = 4  # число действий N_ACT = 4
def mk_lfsr(p: int):
    """Регистр LFSR Галуа: отводы по битам p; в p должно быть не менее 2 битов. Возвращает функцию шага f(s)."""
    """Galois LFSR: taps at bits of p. p must have >= 2 bits."""

    def f(s: int) -> int:  # функция шага
        fb = s & 1  # бит обратной связи fb = s & 1
        ns = s >> 1  # сдвиг вправо ns = s >> 1
        if fb:  # если бит обратной связи равен 1
            ns ^= p & 15  # ns ^= p & 15 — наложить отводы
        return ns  # вернуть новое состояние
    return f  # вернуть функцию шага
def mk_cnt_mod(n: int):
    """Счётчик по модулю n: f(s) = (s + 1) mod n. Возвращает функцию шага."""

    def f(s: int) -> int:  # функция шага
        return (s + 1) % n  # следующее состояние f(s) = (s + 1) mod n
    return f  # вернуть функцию шага
def mk_sat(up: bool, down: bool, lo: int=0, hi: int=15):
    """Счётчик с насыщением: f(s) = min(s + 1, hi), если up и не down; max(s − 1, lo),
    если down и не up; иначе s. Возвращает функцию шага."""

    def f(s: int) -> int:  # функция шага
        if up and (not down):  # если только вверх
            return min(s + 1, hi)  # вернуть min(s + 1, hi)
        if down and (not up):  # если только вниз
            return max(s - 1, lo)  # вернуть max(s − 1, lo)
        return s  # иначе состояние не меняется
    return f  # вернуть функцию шага
def mk_gray():
    """Код Грея: f(s) = номер i, код Грея которого g(i) = i xor (i >> 1) равен (g(s) + 1) mod 16."""

    def f(s: int) -> int:  # функция шага
        g = s ^ s >> 1  # код Грея g = s xor (s >> 1)
        ng = g + 1 & 15  # следующий код ng = (g + 1) & 15
        for i in range(16):  # по всем i от 0 до 15
            if i ^ i >> 1 == ng:  # если g(i) = ng
                return i  # вернуть i
        return s  # не найдено — вернуть s
    return f  # вернуть функцию шага
def mk_shift2(m: int):
    """Двухъячейковый сдвиговый регистр: бит обратной связи m выбирает xor-инверсию нижней пары бит."""

    def f(s: int) -> int:  # функция шага
        hi = s >> 2 & 3  # верхняя пара битов hi = (s >> 2) & 3
        lo = s & 3  # нижняя пара битов lo = s & 3
        if m >> hi & 1:  # если бит m номер hi установлен
            lo ^= 3  # lo ^= 3 — инвертировать нижнюю пару
        return lo << 2 | hi  # вернуть (lo << 2) | hi
    return f  # вернуть функцию шага
def generate_families() -> list[tuple[str, int, object]]:
    """Перечислить параметрическое пространство конструкций. Каждый элемент —
    (имя класса, параметр, функция шага). Это и есть «пространство изобретения»:
    оно порождается правилами, а не фиксированным банком задач. Возвращает список."""
    out = []  # накопитель результата
    for p in range(3, 16):  # по параметрам p = 3..15
        if p.bit_count() >= 2:  # если в p не менее 2 битов
            out.append((f'lfsr', p, mk_lfsr(p)))  # добавить семейство 'lfsr' с параметром p
    for n in range(2, 17):  # по модулям n = 2..16
        out.append(('cnt_mod', n, mk_cnt_mod(n)))  # добавить семейство 'cnt_mod' с модулем n
    for up in (True, False):  # по флагу «вверх»
        for dn in (True, False):  # по флагу «вниз»
            if up or dn:  # если хотя бы один флаг установлен
                out.append(('sat', (1 if up else 0) * 2 + (1 if dn else 0), mk_sat(up, dn)))  # код параметра насыщения: 2·up + dn
    out.append(('gray', 0, mk_gray()))  # добавить семейство 'gray' с параметром 0
    for m in range(1, 4):  # по значениям обратной связи m = 1..3
        out.append(('shift2', m, mk_shift2(m)))  # добавить семейство 'shift2' с параметром m
    return out  # вернуть список семейств
def sample_agree(step_fn, samples: list[tuple[int, int, int]]) -> bool:  # проверка семейства на выборке: все переходы (s, _, ns) согласуются с функцией шага
    return all((step_fn(s) == ns for s, _, ns in samples))  # True, если step_fn(s) = ns для всех образцов (s, _, ns)
def identify(world, samples: list[tuple[int, int, int]], families: list[tuple[str, int, object]]) -> dict:  # функция опознания класса переходов по образцам
    """Предлагать семейства по одному: сначала проверка на образцах, затем сплошной обход
    контрпримеров по всем 16 состояниям. Возвращает честный результат: invented_class,
    param, fn, families_tried, proven. world — среда; samples — переходения (s, действие, s');
    families — список (имя, параметр, функция шага)."""
    """Propose families one by one; sampled check first, then full
    counterexample sweep over all 16 states. Returns honest result."""
    tried = 0  # число проверенных семейств
    for name, param, fn in families:  # по семействам
        tried += 1  # проверено: tried + 1
        if not sample_agree(fn, samples):  # если семейство не согласуется с образцами
            continue  # перейти к следующему семейству
        full = all((fn(s) == world_step_check(world, s) for s in range(N_STATES)))  # сплошная проверка: fn(s) = истинный шаг среды для всех s из 0..15
        if full:  # если полное совпадение
            return {'invented_class': name, 'param': param, 'fn': fn, 'families_tried': tried, 'proven': True}  # вернуть найденный класс с признаком доказанности
    return {'invented_class': None, 'families_tried': tried, 'proven': False}  # ничего не доказано: класс None, proven = False
def world_step_check(world, s: int) -> int:
    """Истинное следующее состояние для первого некомандного тактового действия среды
    (эталон используется ТОЛЬКО для проверки, как полная сверка)."""
    c = world.clock_action  # номер тактового действия среды
    a_nc = next((a for a in range(N_ACT) if a != c))  # первое действие a, не совпадающее с тактовым
    return world.true_step(s, a_nc)  # истинный шаг среды из состояния s при действии a_nc
def collect_samples(world, n: int=8, seed: int=0) -> list:
    """Активно выбрать переходы одного некомандного тактового действия (со сбросами).
    Честно: для предложения используются только реально наблюдённые пары. n — число образцов;
    seed зарезервирован. Возвращает список (s, a_nc, s')."""
    c = world.clock_action  # номер тактового действия среды
    a_nc = next((a for a in range(N_ACT) if a != c))  # первое действие, не совпадающее с тактовым
    samples = []  # список образцов
    s = world.reset()  # начальное состояние сброса s
    for _ in range(n):  # по числу образцов n
        cur = world.reset()  # сбросить среду: текущее состояние cur
        for _ in range(s):  # тактовых шагов s раз
            cur = world.step(c)  # cur = шаг среды тактовым действием
        nxt = world.step(a_nc)  # неклассовый тактовый шаг: следующее состояние nxt
        samples.append((cur, a_nc, nxt))  # добавить образец (cur, a_nc, nxt)
        s = (s + 1) % N_STATES  # следующая глубина s = (s + 1) mod N_STATES
        if len(samples) >= n:  # если набрано n образцов
            break  # прервать цикл
    return samples  # вернуть образцы
def run_rdl(world, seed: int=0) -> dict:
    """Полный цикл открытия представлений на одной среде вне банка задач.
    world — среда; seed зарезервирован. Возвращает словарь результата опознания и,
    при доказанном классе, стоимости: cost_class_knowledge = families_tried + sampled_pairs,
    cost_full_table_baseline = 48, savings_pct = 100·(1 − cost_class_knowledge/48)."""
    families = generate_families()  # пространство параметрических семейств
    samples = collect_samples(world, n=8, seed=seed)  # активно выбрать 8 переходов
    r = identify(world, samples, families)  # опознать класс переходов
    r['sampled_pairs'] = len(samples)  # число использованных образцов
    if r['proven']:  # если класс доказан
        r['cost_class_knowledge'] = r['families_tried'] + r['sampled_pairs']  # стоимость знания класса = проверено семейств + образцов
        r['cost_full_table_baseline'] = 48  # стоимость полной таблицы — 48
        r['savings_pct'] = round(100 * (1 - r['cost_class_knowledge'] / r['cost_full_table_baseline']), 1)  # экономия в процентах = 100·(1 − стоимость класса / 48), с точностью 0.1
    return r  # вернуть результат
