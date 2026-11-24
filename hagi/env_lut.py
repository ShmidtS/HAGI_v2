"""Среда ENV LUT: банк гипотез, компилятор таблиц переходов, связь с
нетлистом и независимый верификатор повторного воспроизведения.

Скрытый мир: 4-битное состояние s, 4 действия с ролями clock, shift, toggle,
coupled; наблюдение obs(s) = (x0, x1, x2 XOR x3) состояние не показывает.

Банк гипотез — все перестановки ролей и таблицы f, g из 16 значений, без
дубликатов по хешу sha таблицы. Активное обучение выбирает зонд (s, a, k)
с максимальной оценкой (H − 0.02·E/n) / (s + 1 + k), где H — энтропия разбиения
кандидатов по сигнатуре, E = Σ v²/n. Таблица переходов переводится в
алгебраическую нормальную форму (АНФ) преобразованием Мёбиуса над GF(2)."""
from __future__ import annotations  # отложенное вычисление аннотаций типов


# ============================================================
# ENV LUT: банк гипотез + OOD (чужеродная динамика)
# ============================================================

from hagi.util import sha  # хеш-функция sha для таблиц переходов
import argparse, itertools, json, math, random, statistics, time  # стандартные модули: аргументы командной строки, перестановки, json, math, случайность, статистика, время
from collections import deque  # двусторонняя очередь для поиска в ширину
from dataclasses import dataclass, asdict  # декоратор dataclass и преобразование в словарь
from pathlib import Path  # класс Path для путей
N_BITS = 4  # число скрытых бит состояния N_BITS = 4
N_STATES = 1 << N_BITS  # число состояний N_STATES = 2^N_BITS = 16
N_ACTIONS = 4  # число действий N_ACTIONS = 4
_MODEL_CACHE = {}  # кэш сгенерированных моделей: ключ — номер действия часов
_PROBE_INDEX_CACHE = {}  # кэш индекса зондов: ключ — номер действия часов
def bit(s, i):
    """Бит i (с младшего) числа s: (s >> i) & 1."""
    return s >> i & 1  # бит i числа s: (s >> i) & 1
def lut2(lut, a, b):
    """Значение двухвходовой таблицы lut: lut >> (a | 2·b) & 1, адрес = a + 2·b."""
    return lut >> (a | b << 1) & 1  # значение двухвходовой таблицы: бит номер a + 2·b числа lut
def obs(s):
    """Наблюдение состояния: биты 0 и 1 без изменений, бит 2 = x2 XOR x3."""
    return bit(s, 0) | bit(s, 1) << 1 | (bit(s, 2) ^ bit(s, 3)) << 2  # наблюдение: y0 = x0, y1 = x1, y2 = x2 XOR x3, где xi = бит i состояния s
def clock(s):
    """Роль часов: следующее состояние (s + 1) mod 16."""
    return s + 1 & 15  # часы: (s + 1) mod 16
def rotate4(s):
    """Циклический сдвиг влево на 1 бит в 4 битах: ((s << 1) & 15) | (s >> 3)."""
    return s << 1 & 15 | s >> 3  # циклический сдвиг влево на 1 бит: ((s << 1) & 15) | (s >> 3)
def toggle(s):
    """Переключение бита 2: s XOR 4, если x0 = 1 и x1 = 0, иначе s."""
    return s ^ (4 if bit(s, 0) and (not bit(s, 1)) else 0)  # переключение бита 2: s XOR 4, если x0 = 1 и x1 = 0, иначе s
def coupled(s, f, g):
    """Связная роль: nx0 = x0 XOR f(x1, x2); nx3 = x3 XOR g(nx0, x1); биты 1, 2 без изменений."""
    x0, x1, x2, x3 = (bit(s, i) for i in range(4))  # разложить s на биты x0, x1, x2, x3
    nx0 = x0 ^ lut2(f, x1, x2)  # новый бит 0: nx0 = x0 XOR f(x1, x2)
    nx3 = x3 ^ lut2(g, nx0, x1)  # новый бит 3: nx3 = x3 XOR g(nx0, x1)
    return nx0 | x1 << 1 | x2 << 2 | nx3 << 3  # собрать состояние: nx0 | x1<<1 | x2<<2 | nx3<<3
def deep_ood(s, f3, g):
    """Чужеродная связная роль: nx0 = x0 XOR f3[x1 | x2<<1 | x3<<2]; nx3 = x3 XOR g(nx0, x1)."""
    x0, x1, x2, x3 = (bit(s, i) for i in range(4))  # разложить s на биты x0, x1, x2, x3
    idx = x1 | x2 << 1 | x3 << 2  # адрес 3-входовой таблицы: idx = x1 | x2<<1 | x3<<2
    nx0 = x0 ^ f3 >> idx & 1  # новый бит 0: nx0 = x0 XOR бит idx таблицы f3
    nx3 = x3 ^ lut2(g, nx0, x1)  # новый бит 3: nx3 = x3 XOR g(nx0, x1)
    return nx0 | x1 << 1 | x2 << 2 | nx3 << 3  # собрать состояние: nx0 | x1<<1 | x2<<2 | nx3<<3
def apply_role(s, role, f=0, g=0, ood=None):
    """Применить к состоянию s роль с параметрами f, g; ood = (f3, g) для роли deep_ood."""
    if role == 'clock':  # роль clock:
        return clock(s)  # вернуть clock(s)
    if role == 'shift':  # роль shift:
        return rotate4(s)  # вернуть rotate4(s)
    if role == 'toggle':  # роль toggle:
        return toggle(s)  # вернуть toggle(s)
    if role == 'coupled':  # роль coupled:
        return coupled(s, f, g)  # вернуть coupled(s, f, g)
    if role == 'deep_ood':  # роль deep_ood:
        return deep_ood(s, ood[0], ood[1])  # вернуть deep_ood(s, f3, g), где ood = (f3, g)
    raise ValueError(role)  # неизвестная роль — ошибка ValueError
@dataclass  # декоратор dataclass: генерирует методы по полям
class World:
    """Скрытый 4-битный мир с действиями-ролями и неполным наблюдением.

        Поля: seed — зерно; roles — роли по действиям; f, g — таблицы связной роли;
        goal — целевое состояние; ood, f3 — чужеродная динамика; state — скрытое
        состояние; steps, resets — счётчики; noisy_repeat — шум при повторе
        перехода; seen — счётчик переходов (s, a).
        Наблюдение obs(s) показывает биты 0, 1 и XOR битов 2 и 3."""
    seed: int  # seed — зерно генератора мира
    roles: tuple[str, ...]  # roles — кортеж ролей по номерам действий
    f: int  # f — первая таблица (0..15)
    g: int  # g — вторая таблица (0..15)
    goal: int  # goal — целевое скрытое состояние
    ood: bool = False  # ood — включена ли чужеродная динамика (по умолчанию нет)
    f3: int = 0  # f3 — трёхвходовая таблица (0..255), по умолчанию 0
    state: int = 0  # state — текущее скрытое состояние, по умолчанию 0
    steps: int = 0  # steps — число сделанных шагов, по умолчанию 0
    resets: int = 0  # resets — число сбросов, по умолчанию 0
    noisy_repeat: bool = False  # noisy_repeat — инвертировать ли бит 2 при повторном переходе, по умолчанию нет
    seen: dict = None  # seen — счётчик переходов: ключ (s, a) -> число повторов, по умолчанию None

    def __post_init__(self):
        """После создания: генератор rng из seed XOR 4570863; seen — пустой словарь, если не задан."""
        self.rng = random.Random(self.seed ^ 4570863)  # генератор rng из seed XOR 4570863
        self.seen = {} if self.seen is None else self.seen  # если seen не задан — пустой словарь

    @property  # декоратор property: читать как атрибут
    def clock_action(self):
        """Номер действия, у которого роль clock."""
        return self.roles.index('clock')  # номер действия часов в roles: roles.index('clock')

    def reset(self):
        """Сброс мира в состояние 0; возвращает наблюдение obs(0)."""
        self.state = 0  # сброс: состояние = 0
        self.resets += 1  # счётчик сбросов +1
        return obs(self.state)  # вернуть наблюдение obs(0)

    def true_step(self, s, a):
        """Истинный переход из состояния s действием a без шума и учёта повторов."""
        role = self.roles[a]  # роль действия a
        if role == 'coupled' and self.ood:  # если это coupled и ood включён:
            return apply_role(s, 'deep_ood', self.f, self.g, (self.f3, self.g))  # применить deep_ood с таблицами (f3, g)
        return apply_role(s, role, self.f, self.g)  # иначе применить роль напрямую

    def step(self, a):
        """Шаг мира: применить действие a и вернуть наблюдение y = obs(s').

            Счётчик seen[(s, a)] растёт на 1; при noisy_repeat и повторном переходе
            бит 2 наблюдения инвертируется: y XOR 4."""
        old = self.state  # запомнить старое состояние
        nxt = self.true_step(old, a)  # вычислить новое состояние
        self.state = nxt  # записать новое состояние
        self.steps += 1  # счётчик шагов +1
        key = (old, a)  # ключ перехода (s, a)
        cnt = self.seen.get(key, 0)  # сколько раз этот переход уже встречался
        self.seen[key] = cnt + 1  # записать cnt + 1
        y = obs(nxt)  # наблюдение нового состояния y = obs(nxt)
        if self.noisy_repeat and cnt >= 1:  # если шум включён и переход повторный (cnt >= 1):
            y ^= 4  # инвертировать бит 2 наблюдения: y XOR 4
        return y  # вернуть y

    def goal_test(self):
        """Достигнута ли цель: state == goal."""
        return self.state == self.goal  # True, если state == goal
def transition_table(roles, f, g):
    """Таблица переходов table[s][a] = apply_role(s, roles[a], f, g), размер 16×4."""
    return [[apply_role(s, roles[a], f, g) for a in range(N_ACTIONS)] for s in range(N_STATES)]  # таблица переходов: table[s][a] = apply_role(s, roles[a], f, g)
def model_digest(table):
    """Digest модели — хеш sha её таблицы переходов (таблицы-дубликаты склеиваются)."""
    return sha(table)  # вернуть sha(table)
def generate_models(clock_action):
    """Все модели с часами на действии clock_action (с кэшем по clock_action).

        Перебираются 6 перестановок ролей shift, toggle, coupled по остальным
        действиям и все f, g из 0..15 (6·256 = 1536 вариантов); модели с уже
        виденным digest отбрасываются. Возвращает список словарей с ключами
        roles, f, g, table, digest."""
    if clock_action in _MODEL_CACHE:  # если уже в кэше:
        return _MODEL_CACHE[clock_action]  # вернуть из кэша
    rest = [a for a in range(N_ACTIONS) if a != clock_action]  # остальные действия (кроме часов)
    models = []  # список моделей
    seen = set()  # множество уже виденных digest
    for perm in itertools.permutations(('shift', 'toggle', 'coupled')):  # по всем 6 перестановкам ролей (shift, toggle, coupled)
        roles = [None] * N_ACTIONS  # roles — список из 4 пустых мест
        roles[clock_action] = 'clock'  # на месте clock_action — роль clock
        for a, r in zip(rest, perm):  # остальным действиям — роли из перестановки
            roles[a] = r  # присвоить роль r действию a
        for f in range(16):  # по всем f (0..15)
            for g in range(16):  # по всем g (0..15)
                tab = transition_table(tuple(roles), f, g)  # таблица переходов для этих ролей
                key = model_digest(tab)  # digest таблицы
                if key in seen:  # если такая таблица уже была — пропустить
                    continue  # пропустить
                seen.add(key)  # добавить digest в seen
                models.append({'roles': tuple(roles), 'f': f, 'g': g, 'table': tab, 'digest': key})  # добавить модель: роли, f, g, таблица, digest
    _MODEL_CACHE[clock_action] = models  # сохранить в кэш
    return models  # вернуть модели
def observation_signature_from_state(s, suffix_len):
    """Сигнатура состояния s: (obs(s), obs(clock(s)), …) длины suffix_len + 1."""
    out = [obs(s)]  # первое наблюдение obs(s)
    for _ in range(suffix_len):  # suffix_len шагов часами
        s = clock(s)  # сдвиг часами: s = clock(s)
        out.append(obs(s))  # добавить obs(s)
    return tuple(out)  # вернуть кортеж наблюдений
def identify_clock(world):
    """Найти действие часов: то, чьи 16 шагов дают obs((i+1) mod 16), i = 0..15.

        Возвращает (номер действия или None, если кандидатов не ровно один;
        стоимость в шагах; число кандидатов)."""
    expected = tuple((obs(i + 1 & 15) for i in range(16)))  # ожидаемая последовательность: obs((i+1) mod 16) при i = 0..15
    matches = []  # совпадающие действия
    cost = 0  # счётчик шагов (стоимость)
    for a in range(N_ACTIONS):  # по каждому действию a
        world.reset()  # сбросить мир
        seq = []  # наблюдённая последовательность
        for _ in range(16):  # 16 шагов
            seq.append(world.step(a))  # записать наблюдение шага a
            cost += 1  # стоимость +1
        if tuple(seq) == expected:  # если последовательность совпала с ожидаемой
            matches.append(a)  # a — кандидат в часы
    return (matches[0] if len(matches) == 1 else None, cost, len(matches))  # вернуть единственного кандидата (иначе None), стоимость и число кандидатов
def probe_world(world, c_action, state, action, suffix_len):
    """Выполнить зонд в мире: state шагов часами, действие action, suffix_len шагов часами.

        Возвращает ((наблюдения), стоимость в шагах = state + 1 + suffix_len)."""
    world.reset()  # сбросить мир в состояние 0
    cost = 0  # стоимость = 0
    for _ in range(state):  # state раз применить часы: скрытое состояние станет state
        world.step(c_action)  # шаг часами
        cost += 1  # стоимость +1
    seq = [world.step(action)]  # наблюдение после действия action
    cost += 1  # стоимость +1
    for _ in range(suffix_len):  # suffix_len раз
        seq.append(world.step(c_action))  # добавить наблюдение после шага часами
        cost += 1  # стоимость +1
    return (tuple(seq), cost)  # вернуть (последовательность, стоимость)
def query_signature(model, state, action, suffix_len, c_action):
    """Предсказать сигнатуру зонда по таблице модели (без обращений к миру)."""
    s = model['table'][state][action]  # s = table[state][action] — новое состояние
    out = [obs(s)]  # первое наблюдение obs(s)
    for _ in range(suffix_len):  # suffix_len раз
        s = model['table'][s][c_action]  # s = table[s][c_action] — шаг часами внутри модели
        out.append(obs(s))  # добавить obs(s)
    return tuple(out)  # вернуть кортеж
def build_probe_index(c_action):
    """Предвычислить точные разбиения по сигнатурам для конечной грамматики гипотез.

        Множество кандидатов хранится как целое число-битовая маска: не нужно
        заново моделировать каждую гипотезу для каждого эксперимента.
        Возвращает (модели, partitions, маска всех моделей), где
        partitions[(s, a, k)][сигнатура] = маска моделей, дающих эту сигнатуру."""
    if c_action in _PROBE_INDEX_CACHE:  # если в кэше:
        return _PROBE_INDEX_CACHE[c_action]  # вернуть из кэша
    models = generate_models(c_action)  # все модели для этих часов
    partitions = {}  # partitions: (s, a, k) -> {сигнатура: битовая маска моделей}
    for s in range(N_STATES):  # по состояниям s
        for a in range(N_ACTIONS):  # по действиям a
            if a == c_action:  # действие часов не зондируется
                continue  # пропустить
            for k in range(5):  # по длинам суффикса k = 0..4
                groups = {}  # группы по сигнатуре
                key = (s, a, k)  # ключ зонда (s, a, k)
                for idx, model in enumerate(models):  # по всем моделям с номером idx
                    sig = query_signature(model, s, a, k, c_action)  # сигнатура модели для этого зонда
                    groups[sig] = groups.get(sig, 0) | 1 << idx  # добавить бит idx в группу сигнатуры: mask | 1<<idx
                partitions[key] = groups  # сохранить группы
    out = (models, partitions, (1 << len(models)) - 1)  # итог: модели, разбиения и маска всех моделей 2^M − 1
    _PROBE_INDEX_CACHE[c_action] = out  # сохранить в кэш
    return out  # вернуть кортеж
def choose_active_probe_index(c_action, candidate_mask, used):
    """Выбрать зонд (s, a, k) с максимальной оценкой (H − 0.02·E/n) / (s + 1 + k).

        n — число кандидатов в маске; H — энтропия разбиения кандидатов по
        сигнатурам, E = Σ v²/n. При равном score выигрывает меньшая стоимость,
        потом меньшие s, a, k. Использованные зонды пропускаются; None, если
        различающих зондов нет."""
    models, partitions, _ = build_probe_index(c_action)  # взять индекс
    n = candidate_mask.bit_count()  # n — число оставшихся кандидатов: popcount(candidate_mask)
    best = None  # лучший пока не найден
    for key, all_groups in partitions.items():  # по всем зондам индекса
        if key in used:  # уже использованный зонд
            continue  # пропустить
        counts = [(candidate_mask & mask).bit_count() for mask in all_groups.values()]  # число кандидатов в каждой группе: popcount(candidate_mask & mask)
        counts = [v for v in counts if v]  # отбросить пустые группы
        if len(counts) <= 1:  # зонд ничего не различает
            continue  # пропустить
        h = -sum((v / n * math.log2(v / n) for v in counts))  # энтропия H = −Σ (v/n)·log2(v/n) по группам
        s, a, k = key  # разобрать ключ
        cost = s + 1 + k  # стоимость зонда = s + 1 + k шагов
        expected = sum((v * v for v in counts)) / n  # ожидаемый остаток E = Σ v² / n
        score = (h - 0.02 * (expected / n)) / cost  # оценка score = (H − 0.02·E/n) / cost
        item = (score, -cost, -s, -a, -k, key)  # кортеж для сравнения: score, −cost, −s, −a, −k, ключ
        if best is None or item[:5] > best[:5]:  # сравнить по первым пяти полям (лучшему соответствует большее)
            best = item  # запомнить лучший
    return None if best is None else best[5]  # вернуть ключ лучшего или None
def entropy(groups, n):
    """Энтропия Шеннона разбиения: H = −Σ (c/n)·log2(c/n) по размерам групп c."""
    h = 0.0  # начальное значение h = 0
    for c in groups.values():  # по размерам групп c
        p = c / n  # доля группы p = c/n
        h -= p * math.log2(p)  # вычесть p·log2(p)
    return h  # вернуть h
def choose_active_probe(models, c_action, used):
    """То же, что choose_active_probe_index, но группы считаются перебором списка моделей."""
    rest = [a for a in range(N_ACTIONS) if a != c_action]  # действия кроме часов
    n = len(models)  # n — число моделей
    best = None  # лучший не найден
    for s in range(N_STATES):  # по состояниям
        for a in rest:  # по действиям
            for k in range(5):  # по длинам суффикса
                key = (s, a, k)  # ключ зонда
                if key in used:  # уже использован
                    continue  # пропустить
                groups = {}  # группы по сигнатуре
                for m in models:  # по моделям
                    sig = query_signature(m, s, a, k, c_action)  # сигнатура модели
                    groups[sig] = groups.get(sig, 0) + 1  # группа +1 модель
                if len(groups) <= 1:  # не различает
                    continue  # пропустить
                h = entropy(groups, n)  # энтропия H
                cost = s + 1 + k  # стоимость = s + 1 + k
                expected = sum((v * v for v in groups.values())) / n  # ожидаемый остаток E = Σ v² / n
                score = (h - 0.02 * (expected / n)) / cost  # оценка score = (H − 0.02·E/n) / cost
                item = (score, -cost, -s, -a, -k, key, groups)  # кортеж с группами
                if best is None or item[:5] > best[:5]:  # сравнить
                    best = item  # запомнить лучший
    return None if best is None else best[5]  # вернуть ключ лучшего или None
def learn_active(world, initial_models=None, max_queries=80, strategy='active'):
    """Активное обучение: сузить банк гипотез зондами мира и проверить кандидата.

        world — мир; initial_models — стартовый список моделей (None — весь банк
        через битовый индекс); max_queries — лимит зондов; strategy — 'active'
        (информативный зонд) или случайный.
        Сначала ищутся часы, затем каждый зонд (s, a, k) оставляет только модели,
        чья сигнатура совпала с наблюдением. Возвращает словарь со статусом:
        HOLD_CLOCK_AMBIGUOUS, GRAMMAR_MISMATCH (мир не из грамматики),
        PROVISIONAL_SAMPLED (кандидат прошёл 6 выборочных проверок) или
        HOLD_AMBIGUOUS, плюс расходы, журнал evidence и проверки."""
    c_action, clock_cost, nclock = identify_clock(world)  # найти действие часов
    if c_action is None:  # часы не найдены однозначно
        return {'status': 'HOLD_CLOCK_AMBIGUOUS', 'clock_cost': clock_cost, 'queries': 0, 'interactions': clock_cost}  # статус HOLD_CLOCK_AMBIGUOUS, запросов 0
    indexed = initial_models is None  # indexed — использовать битовый индекс (модели не переданы)
    if indexed:  # если индекс
        models, partitions, full_mask = build_probe_index(c_action)  # взять модели, разбиения и полную маску
        candidate_mask = full_mask  # вначале кандидаты — все модели
        start_n = len(models)  # начальное число кандидатов
    else:  # иначе — переданный список моделей
        models = list(initial_models)  # копия списка
        candidate_mask = None  # candidate_mask не нужен
        partitions = None  # partitions не нужен
        start_n = len(models)  # начальное число кандидатов
    used = set()  # использованные зонды
    q = 0  # число запросов q
    interactions = clock_cost  # взаимодействия: столько ушло на поиск часов
    evidence = []  # журнал свидетельств
    rng = random.Random(world.seed ^ 2641)  # генератор для случайной стратегии: seed XOR 2641
    while q < max_queries:  # пока не исчерпан лимит запросов
        current_n = candidate_mask.bit_count() if indexed else len(models)  # текущее число кандидатов: popcount маски или длина списка
        if current_n <= 1:  # остался один кандидат
            break  # стоп
        if strategy == 'active':  # стратегия active — информативный зонд
            probe = choose_active_probe_index(c_action, candidate_mask, used) if indexed else choose_active_probe(models, c_action, used)  # выбрать зонд индексом (если он есть) или перебором
        else:  # иначе — случайный зонд
            options = [(s, a, k) for s in range(N_STATES) for a in range(N_ACTIONS) if a != c_action for k in range(5) if (s, a, k) not in used]  # варианты (s, a, k): не часы и ещё не использованные
            probe = rng.choice(options) if options else None  # случайный выбор или None
        if probe is None:  # зондов не осталось
            break  # стоп
        s, a, k = probe  # разобрать зонд
        used.add(probe)  # пометить зонд использованным
        observed, cost = probe_world(world, c_action, s, a, k)  # выполнить зонд в мире
        interactions += cost  # учесть стоимость
        q += 1  # запросов +1
        before = current_n  # число кандидатов до запроса
        if indexed:  # если используется индекс
            group_mask = partitions[probe].get(observed, 0)  # маска группы с наблюдённой сигнатурой (0, если такой нет)
            candidate_mask &= group_mask  # кандидаты &= маска группы
            after = candidate_mask.bit_count()  # число кандидатов после
        else:  # иначе
            models = [m for m in models if query_signature(m, s, a, k, c_action) == observed]  # отфильтровать список моделей по сигнатуре
            after = len(models)  # после — длина списка
        evidence.append({'state': s, 'action': a, 'suffix': k, 'obs': list(observed), 'before': before, 'after': after, 'cost': cost})  # записать свидетельство: зонд, наблюдение, до/после, стоимость
        if after == 0:  # кандидатов не осталось — мир не из грамматики
            return {'status': 'GRAMMAR_MISMATCH', 'clock_action': c_action, 'queries': q, 'interactions': interactions, 'evidence': evidence, 'candidate_start': start_n}  # статус GRAMMAR_MISMATCH с журналом
    if indexed:  # если индекс: оставить модели, чьи биты входят в маску
        models = [m for i, m in enumerate(models) if candidate_mask >> i & 1]  # кандидаты по битам маски
    if len(models) == 1 or len({m['digest'] for m in models}) == 1:  # остался один кандидат или все с одним digest
        candidate = models[0]  # выбранный кандидат
        nonclock = [a for a in range(N_ACTIONS) if a != c_action]  # нечасовые действия
        checks = [(14, a) for a in nonclock]  # проверки: состояние 14 по каждому нечасовому действию
        vrng = random.Random(world.seed ^ 658711)  # генератор проверок: seed XOR 658711
        checks += [(vrng.randrange(N_STATES), vrng.choice(nonclock)) for _ in range(3)]  # плюс 3 случайные проверки (состояние, действие)
        validation = []  # журнал проверок
        mismatch = []  # несовпадения
        for vs, va in checks:  # по проверкам
            observed, cost = probe_world(world, c_action, vs, va, 4)  # выполнить зонд с суффиксом 4
            interactions += cost  # учесть стоимость
            predicted = query_signature(candidate, vs, va, 4, c_action)  # предсказание кандидата
            item = {'state': vs, 'action': va, 'observed': list(observed), 'predicted': list(predicted), 'cost': cost}  # запись проверки
            validation.append(item)  # добавить в журнал
            if observed != predicted:  # наблюдение не совпало с предсказанием
                mismatch.append(item)  # записать несовпадение
        if mismatch:  # есть несовпадения
            return {'status': 'GRAMMAR_MISMATCH', 'clock_action': c_action, 'queries': q, 'interactions': interactions, 'evidence': evidence, 'candidate_start': start_n, 'candidate_left': len(models), 'validation': validation, 'mismatch': mismatch, 'provisional_model': candidate}  # статус GRAMMAR_MISMATCH с проверками
        return {'status': 'PROVISIONAL_SAMPLED', 'clock_action': c_action, 'model': candidate, 'clock_cost': clock_cost, 'queries': q, 'interactions': interactions, 'evidence': evidence, 'candidate_start': start_n, 'candidate_left': len(models), 'validation': validation, 'validation_status': 'PASS_SAMPLED_ONLY'}  # статус PROVISIONAL_SAMPLED: модель найдена, проверена выборочно
    return {'status': 'HOLD_AMBIGUOUS', 'clock_action': c_action, 'clock_cost': clock_cost, 'queries': q, 'interactions': interactions, 'evidence': evidence, 'candidate_start': start_n, 'candidate_left': len(models)}  # иначе статус HOLD_AMBIGUOUS: кандидат неоднозначен
def decode_hidden_from_clock_signature(seq):
    """По последовательности из 5 наблюдений часов найти скрытое состояние; None, если нет такого."""
    sigmap = {observation_signature_from_state(s, 4): s for s in range(16)}  # карта: сигнатура длины 5 -> состояние
    return sigmap.get(tuple(seq))  # искать сигнатуру в карте; None, если её нет
def full_table_fallback(world, c_action):
    """Измерить все нечасовые переходы напрямую; вернуть (таблица, стоимость, журнал строк).

        Часовые переходы известны из роли clock, остальные измеряются зондами
        с суффиксом 4; следующее состояние декодируется по сигнатуре часов."""
    table = [[None] * N_ACTIONS for _ in range(N_STATES)]  # таблица из None
    for s in range(N_STATES):  # по состояниям
        table[s][c_action] = clock(s)  # действие часов известно заранее: table[s][c] = clock(s)
    interactions = 0  # счётчик взаимодействий
    rows = []  # журнал строк
    for s in range(N_STATES):  # по состояниям
        for a in range(N_ACTIONS):  # по действиям
            if a == c_action:  # часы пропустить
                continue  # пропустить
            seq, cost = probe_world(world, c_action, s, a, 4)  # зонд с суффиксом 4
            nxt = decode_hidden_from_clock_signature(seq)  # декодировать скрытое состояние по сигнатуре
            interactions += cost  # учесть стоимость
            if nxt is None:  # не удалось
                raise RuntimeError('clock suffix failed to decode latent state')  # ошибка: суффикс часов не декодирует скрытое состояние
            table[s][a] = nxt  # записать переход
            rows.append({'state': s, 'action': a, 'observations': list(seq), 'next_state': nxt, 'cost': cost})  # записать строку журнала
    if any((v is None for row in table for v in row)):  # если в таблице остался None
        raise AssertionError('incomplete table')  # ошибка: таблица неполна
    return (table, interactions, rows)  # вернуть таблицу, стоимость и журнал
def apply_mobius(values, n):
    """Преобразование Мёбиуса над GF(2): таблицу значений булевой функции от n аргументов
    превратить в список мономов АНФ с коэффициентом 1 (номер монома — маска его аргументов)."""
    a = [int(v) & 1 for v in values]  # a[x] = значение x mod 2
    for bit_i in range(n):  # по битам i
        step = 1 << bit_i  # шаг = 2^i
        for x in range(1 << n):  # по всем x
            if x & step:  # если бит i в x установлен
                a[x] ^= a[x ^ step]  # a[x] ^= a[x XOR 2^i] — шаг Мёбиуса
    return [i for i, v in enumerate(a) if v]  # номера x, где a[x] = 1
def table_to_anf(table):
    """Перевести таблицу переходов в АНФ; возвращает anf[a][b] — мономы бита b выхода действия a."""
    actions = []  # по действиям: anf[a][b] — мономы бита b выхода действия a
    for a in range(N_ACTIONS):  # действие a
        outs = []  # по битам выхода
        for ob in range(N_BITS):  # бит выхода ob
            vals = [bit(table[s][a], ob) for s in range(N_STATES)]  # значения этого бита на всех состояниях
            outs.append(apply_mobius(vals, N_BITS))  # мономы этого бита
        actions.append(outs)  # добавить
    return actions  # вернуть
def eval_anf(terms, s):
    """Значение АНФ на состоянии s: XOR членов, чей моном t состоит из битов s."""
    out = 0  # начальное значение out = 0
    for t in terms:  # по мономам
        if s & t == t:  # моном t входит в s: s & t == t
            out ^= 1  # инвертировать бит: out ^= 1
    return out  # вернуть out
def verify_table_anf(table, anfs):
    """Сверить АНФ с таблицей по всем 64 парам (s, a); вернуть (признак, число проверок, подробности)."""
    checks = 0  # счётчик проверок
    for s in range(N_STATES):  # по состояниям
        for a in range(N_ACTIONS):  # по действиям
            got = sum((eval_anf(anfs[a][b], s) << b for b in range(N_BITS)))  # got = Σ_b eval_anf(anfs[a][b], s) << b — состояние, собранное из АНФ
            if got != table[s][a]:  # если не совпало с таблицей
                return (False, checks, (s, a, got, table[s][a]))  # вернуть False и подробности
            checks += 1  # проверок +1
    return (True, checks, None)  # вернуть успех
def bfs_plan(table, start, goal):
    """Поиск в ширину по таблице; вернуть кратчайший список действий от start до goal или None."""
    q = deque([start])  # очередь из start
    prev = {start: None}  # предок состояния (у start — None)
    prev_a = {}  # действие, которым пришли в состояние
    while q:  # пока очередь не пуста
        s = q.popleft()  # достать s
        if s == goal:  # цель достигнута
            break  # выход
        for a in range(N_ACTIONS):  # по действиям
            t = table[s][a]  # новое состояние: t = table[s][a]
            if t not in prev:  # t ещё не посещено
                prev[t] = s  # запомнить предка
                prev_a[t] = a  # и действие
                q.append(t)  # в очередь
    if goal not in prev:  # цель недостижима
        return None  # вернуть None
    path = []  # путь (в обратном порядке)
    cur = goal  # текущее состояние — цель
    while prev[cur] is not None:  # пока не дошли до start
        path.append(prev_a[cur])  # добавить действие
        cur = prev[cur]  # перейти к предку
    return list(reversed(path))  # вернуть в прямом порядке
def execute_plan(world, table, path, goal):
    """Исполнить план в мире, сверяя каждое наблюдение с таблицей; вернуть (успех, трасса, сообщение)."""
    world.reset()  # сбросить мир
    s = 0  # s = 0 — предсказанное состояние
    trace = []  # трасса
    for a in path:  # по действиям пути
        pred = table[s][a]  # предсказанное состояние
        observed = world.step(a)  # наблюдение мира
        if observed != obs(pred):  # наблюдение не совпало с предсказанием
            return (False, trace, 'observation mismatch')  # неудача
        s = pred  # s = предсказанное состояние
        trace.append({'action': a, 'predicted_state': s, 'observed': observed})  # записать шаг трассы
    return (world.goal_test() and s == goal, trace, 'ok' if world.goal_test() and s == goal else 'goal mismatch')  # успех, если цель мира достигнута и s == goal; сообщение об исходе
def make_world(seed, ood=False, same_dynamics=None, noisy=False, f3=None):
    """Создать World: случайные перестановка ролей, таблицы и цель по зерну seed.

        ood — чужеродная динамика (случайная f3 с нечётным числом единиц);
        same_dynamics — задать (f, g); noisy — шум при повторе перехода;
        f3 — задать чужеродную таблицу явно."""
    rng = random.Random(seed ^ 12648430)  # генератор: seed XOR 12648430
    roles_list = list(('clock', 'shift', 'toggle', 'coupled'))  # список ролей
    rng.shuffle(roles_list)  # перемешать роли
    roles = tuple(roles_list)  # кортеж ролей
    if same_dynamics is None:  # если динамика не задана
        f = rng.randint(1, 14)  # f — случайное из 1..14
        g = rng.randint(1, 14)  # g — случайное из 1..14
    else:  # иначе
        f, g = same_dynamics  # взять заданные f, g
    if f3 is None:  # если f3 не задана
        f3 = 128  # f3 = 128 (по умолчанию)
        if ood:  # если включена чужеродная динамика
            choices = [x for x in range(1, 256) if x.bit_count() & 1 and x not in (129, 165)]  # варианты: f3 с нечётным числом единиц, кроме 129 и 165
            f3 = rng.choice(choices)  # случайный выбор f3
    goal = rng.randrange(1, 16)  # цель — случайное из 1..15
    return World(seed, roles, f, g, goal, ood, f3, 0, 0, 0, noisy, {})  # создать World со всеми параметрами
def audit_candidate_full(world, c_action, model):
    """Exhaustively check all 16 x 3 non-clock transitions via reset + clock addressing."""
    rows = []  # журнал строк
    mismatches = []  # несовпадения
    interactions = 0  # счётчик взаимодействий
    for state in range(N_STATES):  # по состояниям
        for action in range(N_ACTIONS):  # по действиям
            if action == c_action:  # действие часов:
                if model['table'][state][action] != clock(state):  # если часы в модели расходятся с истинными clock(state):
                    mismatches.append({'state': state, 'action': action, 'reason': 'clock transition'})  # записать несовпадение
                continue  # дальше
            observed, cost = probe_world(world, c_action, state, action, 4)  # зонд с суффиксом 4
            interactions += cost  # учесть стоимость
            got = decode_hidden_from_clock_signature(observed)  # наблюдённое следующее состояние (по сигнатуре часов)
            want = model['table'][state][action]  # предсказанное моделью состояние
            row = {'state': state, 'action': action, 'observed_next': got, 'predicted_next': want, 'cost': cost}  # строка результата
            rows.append(row)  # в журнал
            if got != want:  # если не совпало
                mismatches.append(row)  # записать несовпадение
    return {'complete': len(rows) == N_STATES * (N_ACTIONS - 1), 'pairs_checked': len(rows), 'total_pairs': N_STATES * N_ACTIONS, 'interactions': interactions, 'pass': len(rows) == N_STATES * (N_ACTIONS - 1) and (not mismatches), 'mismatches': mismatches, 'rows': rows}  # итог: полнота = 48 строк, pass = полнота и нет несовпадений
def table_from_full_audit(audit, c_action):
    """Свести полный аудит в таблицу переходов; вернуть (таблица, стоимость, журнал строк)."""
    if not audit.get('complete') or len(audit.get('rows', [])) != N_STATES * (N_ACTIONS - 1):  # если аудит неполный — ошибка
        raise ValueError('cannot build table from incomplete audit')  # ошибка: нельзя строить таблицу из неполного аудита
    table = [[None] * N_ACTIONS for _ in range(N_STATES)]  # таблица из None
    rows = []  # новый журнал строк
    for state in range(N_STATES):  # по состояниям
        table[state][c_action] = clock(state)  # часовые переходы: table[state][c] = clock(state)
    for r in audit['rows']:  # по строкам аудита
        state, action, nxt = (r['state'], r['action'], r['observed_next'])  # взять состояние, действие и наблюдённое следующее состояние
        if table[state][action] is not None:  # если переход уже записан — ошибка
            raise ValueError('duplicate transition in audit')  # ошибка: повторяющийся переход в аудите
        table[state][action] = nxt  # записать переход
        rows.append({'state': state, 'action': action, 'observations': [], 'next_state': nxt, 'cost': r['cost'], 'source': 'full_candidate_audit'})  # строка журнала с пометкой источника
    if any((v is None for row in table for v in row)):  # если остались незаполненные клетки
        raise ValueError('full audit left missing transitions')  # ошибка: в аудите не хватало переходов
    return (table, audit['interactions'], rows)  # вернуть таблицу, стоимость и журнал
def run_adversarial_unseen_transition(seed):
    """Построить чужеродную среду, отличающуюся одним переходом, который обучающийся не пробовал."""
    baseline = make_world(seed)  # базовый мир по зерну
    base_result = learn_active(baseline)  # обучиться на базовом мире
    if base_result.get('status') != 'PROVISIONAL_SAMPLED':  # если модель не найдена — настройка не удалась
        return {'status': 'SETUP_FAILED', 'base_status': base_result.get('status')}  # статус SETUP_FAILED
    c = base_result['clock_action']  # действие часов
    visited = {(e['state'], e['action']) for e in base_result['evidence']}  # пары (s, a), которые обучающийся реально пробовал
    visited.update(((v['state'], v['action']) for v in base_result.get('validation', [])))  # добавить пары выборочных проверок
    visited.update(((s, c) for s in range(N_STATES)))  # добавить все часовые пары (s, c)
    target = next(((s, a) for s in range(N_STATES) for a in range(N_ACTIONS) if a != c and (s, a) not in visited), None)  # первая непосещённая нечасовая пара (s, a) или None
    if target is None:  # если все пары посещены
        return {'status': 'SETUP_FAILED', 'reason': 'no unseen pair'}  # статус SETUP_FAILED
    world = make_world(seed)  # новый мир с тем же зерном
    original = world.true_step  # запомнить исходный истинный переход
    ts, ta = target  # целевые состояние и действие

    def altered_true_step(self, s, a):
        """Подменённый истинный переход: для целевой пары (ts, ta) инвертируется бит 0."""
        nxt = original(s, a)  # исходный переход
        return nxt ^ 1 if (s, a) == (ts, ta) else nxt  # для целевой пары инвертировать бит 0: nxt XOR 1, иначе nxt
    world.true_step = altered_true_step.__get__(world, World)  # подменить метод true_step мира (привязка к world)
    learned = learn_active(world)  # обучиться на изменённом мире
    if learned.get('status') != 'PROVISIONAL_SAMPLED':  # если изменение заметили уже при выборочном обучении
        return {'status': 'UNEXPECTED_DETECTION_DURING_SAMPLE', 'target': list(target), 'learn_status': learned.get('status')}  # статус UNEXPECTED_DETECTION_DURING_SAMPLE
    candidate = learned['model']  # найденная модель
    full = audit_candidate_full(world, learned['clock_action'], candidate)  # полный аудит по всем 48 парам
    return {'status': 'PASS' if not full['pass'] and target in [(m['state'], m['action']) for m in full['mismatches']] else 'FAIL', 'target': list(target), 'learner_status': learned['status'], 'sampled_validation': learned.get('validation_status'), 'full_audit_detected': not full['pass'], 'pairs_checked': full['pairs_checked'], 'mismatches': full['mismatches']}  # PASS, если полный аудит не прошёл и несовпадение — целевая пара, иначе FAIL
def run_seed(seed, strategy='active'):
    """Один прогон: создать мир, обучиться стратегией strategy, спланировать и исполнить план."""
    world = make_world(seed)  # мир по зерну
    learned = learn_active(world, strategy=strategy)  # обучиться стратегией strategy
    if learned['status'] not in ('PROVISIONAL_SAMPLED', 'IDENTIFIED', 'IDENTIFIED_EQUIVALENCE_CLASS'):  # если модель не отождествлена — вернуть статус обучения
        return {'seed': seed, 'status': learned['status'], 'learn': learned}  # результат обучения без плана
    model = learned['model']  # найденная модель
    table = model['table']  # её таблица переходов
    path = bfs_plan(table, 0, world.goal)  # кратчайший план из 0 в цель
    ok, trace, reason = execute_plan(world, table, path, world.goal)  # исполнить план
    return {'seed': seed, 'status': 'PASS_SAMPLED' if ok else 'FAIL', 'model_evidence': 'SAMPLED_ONLY', 'clock_action': learned['clock_action'], 'f': world.f, 'g': world.g, 'model_digest': model['digest'], 'candidate_start': learned['candidate_start'], 'candidate_left': learned['candidate_left'], 'queries': learned['queries'], 'learn_interactions': learned['interactions'], 'evidence': learned['evidence'], 'validation': learned.get('validation', []), 'validation_status': learned.get('validation_status', 'NONE'), 'goal': world.goal, 'plan': path, 'plan_length': len(path), 'execution': trace, 'execution_ok': ok, 'reason': reason}  # PASS_SAMPLED, если план исполнен; иначе FAIL; все данные запуска
def infer_deep_ood_rule(table, clock_action):
    """Recover a novel 3-input update followed by a 2-input update from the exact table."""
    for action in range(N_ACTIONS):  # по действиям
        if action == clock_action:  # часы пропустить
            continue  # пропустить
        fvals = {}  # восстановленные значения f3 по адресам idx
        gvals = {}  # восстановленные значения g по адресам nx0 + 2·x1
        ok = True  # действие пока подходит
        for s in range(N_STATES):  # по состояниям
            x0, x1, x2, x3 = (bit(s, i) for i in range(4))  # биты входа
            nxt = table[s][action]  # следующее состояние
            nx0, nx1, nx2, nx3 = (bit(nxt, i) for i in range(4))  # биты выхода
            if nx1 != x1 or nx2 != x2:  # биты 1 и 2 должны сохраняться
                ok = False  # не подходит
                break  # выход
            idx = x1 | x2 << 1 | x3 << 2  # адрес f3: idx = x1 | x2<<1 | x3<<2
            fv = x0 ^ nx0  # значение f3 в этой точке: fv = x0 XOR nx0
            if idx in fvals and fvals[idx] != fv:  # если тот же адрес уже дал другое значение — противоречие
                ok = False  # не подходит
                break  # выход
            fvals[idx] = fv  # запомнить значение
            idx2 = nx0 | x1 << 1  # адрес g: idx2 = nx0 | x1<<1
            gv = x3 ^ nx3  # значение g: gv = x3 XOR nx3
            if idx2 in gvals and gvals[idx2] != gv:  # противоречие с прежним значением
                ok = False  # не подходит
                break  # выход
            gvals[idx2] = gv  # запомнить значение
        if not ok or len(fvals) != 8 or len(gvals) != 4:  # нужны все 8 адресов f3 и все 4 адреса g
            continue  # следующее действие
        f3 = sum((v << i for i, v in fvals.items()))  # собрать f3 = Σ_i v_i·2^i
        g = sum((v << i for i, v in gvals.items()))  # собрать g = Σ_i v_i·2^i
        if f3.bit_count() & 1 == 0:  # число единиц f3 должно быть нечётным
            continue  # следующее действие
        reconstructed = [None] * N_ACTIONS  # восстановленные строки таблицы по действиям
        reconstructed[clock_action] = [clock(s) for s in range(N_STATES)]  # часовая строка: clock(s)
        reconstructed[action] = [deep_ood(s, f3, g) for s in range(N_STATES)]  # строка целевого действия: deep_ood(s, f3, g)
        for a in range(N_ACTIONS):  # по действиям
            if a in (clock_action, action):  # часы и целевое действие уже заполнены
                continue  # пропустить
            reconstructed[a] = [table[s][a] for s in range(N_STATES)]  # остальные строки — из измеренной таблицы
        if any((reconstructed[action][s] != table[s][action] for s in range(N_STATES))):  # если deep_ood всё же расходится с таблицей
            continue  # следующее действие
        return {'action': action, 'f3': f3, 'g': g, 'reconstructed_rows': N_STATES}  # найденное правило: действие, f3, g
    return None  # правило не восстановлено
def run_ood(seed):
    """Прогон чужеродного мира: обучение, полная таблица, АНФ, план и восстановление правила.

        Если выборочный кандидат прошёл полный аудит — BASE_EQUIVALENT_EXACT; если отвергнут
        или мир не из грамматики — полная таблица из аудита или прямым перебором. Статус PASS,
        если АНФ сверена, план исполнен и правило f3, g совпало с истинным."""
    world = make_world(seed, ood=True)  # мир с чужеродной динамикой
    learned = learn_active(world)  # обучиться
    c = learned.get('clock_action')  # действие часов
    sampled_audit = None  # аудит выборочного кандидата пока нет
    if learned['status'] == 'PROVISIONAL_SAMPLED':  # если найден выборочный кандидат
        sampled_audit = audit_candidate_full(world, c, learned['model'])  # полный аудит кандидата
        if sampled_audit['pass']:  # если кандидат совпал по всем 48 парам
            return {'seed': seed, 'status': 'BASE_EQUIVALENT_EXACT', 'clock_action': c, 'sampled_candidate_full_audit': sampled_audit, 'true_f3': world.f3, 'true_g': world.g}  # чужеродный мир оказался неотличим: статус BASE_EQUIVALENT_EXACT
        grammar_status = 'PROVISIONAL_REJECTED_BY_FULL_AUDIT'  # иначе кандидат отвергнут полным аудитом
        table, cost, rows = table_from_full_audit(sampled_audit, c)  # таблица строится из строк полного аудита
        full_table_source = 'REUSED_EXHAUSTIVE_CANDIDATE_AUDIT'  # источник полной таблицы — аудит кандидата
    elif learned['status'] == 'GRAMMAR_MISMATCH':  # если мир не из грамматики
        grammar_status = 'GRAMMAR_MISMATCH'  # статус грамматики
        table, cost, rows = full_table_fallback(world, c)  # таблица измеряется прямым перебором
        full_table_source = 'DIRECT_FULL_TABLE_FALLBACK'  # источник таблицы — прямой перебор
    else:  # иначе
        return {'seed': seed, 'status': 'LEARN_FAILURE', 'learn': learned}  # статус LEARN_FAILURE
    anfs = table_to_anf(table)  # АНФ таблицы
    vok, checks, cex = verify_table_anf(table, anfs)  # сверить АНФ с таблицей
    path = bfs_plan(table, 0, world.goal)  # кратчайший план
    ok, trace, reason = execute_plan(world, table, path, world.goal)  # исполнить план
    inferred = infer_deep_ood_rule(table, c)  # восстановить правило чужеродной динамики
    rule_exact = inferred is not None and inferred['f3'] == world.f3 and (inferred['g'] == world.g)  # правило точно, если f3 и g совпали с истинными
    total_interactions = learned.get('interactions', 0) + cost  # всего взаимодействий: обучение + построение таблицы
    return {'seed': seed, 'status': 'PASS' if vok and ok and rule_exact else 'FAIL', 'clock_action': c, 'grammar_status': grammar_status, 'active_queries_before_hold': learned.get('queries', 0), 'active_interactions_before_hold': learned.get('interactions', 0), 'sampled_candidate_full_audit': sampled_audit, 'fallback_transition_pairs': len(rows), 'fallback_interactions': cost, 'full_table_source': full_table_source, 'ood_total_interactions': total_interactions, 'anf_checks': checks, 'anf_verified': vok, 'counterexample': cex, 'model_digest': model_digest(table), 'goal': world.goal, 'plan': path, 'plan_length': len(path), 'execution_ok': ok, 'execution_trace': trace, 'reason': reason, 'rows': rows, 'anfs': anfs, 'true_f3': world.f3, 'true_g': world.g, 'inferred_novel_rule': inferred, 'novel_rule_exact': rule_exact}  # PASS, если АНФ верна, план исполнен и правило точно; иначе FAIL
def self_tests():
    """Встроенные самопроверки: наблюдение, таблицы, обучение, чужеродная динамика, восстановление правила."""
    tests = []  # список проверок
    aliases = {}  # карта: наблюдение -> список скрытых состояний
    for s in range(16):  # по состояниям
        aliases.setdefault(obs(s), []).append(s)  # добавить s в класс своего наблюдения
    assert len(aliases) == 8 and all((len(v) == 2 for v in aliases.values()))  # 8 классов по 2 состояния: наблюдение половину состояний не различает
    sigs = [observation_signature_from_state(s, 4) for s in range(16)]  # сигнатуры длины 5 для всех состояний
    assert len(set(sigs)) == 16  # все 16 сигнатур различны
    tests.append({'name': 'partial_observation_and_clock_disambiguation', 'pass': True, 'observation_classes': len(aliases), 'unique_clock_signatures': len(set(sigs))})  # записать проверку
    lut_checks = 0  # счётчик проверок таблиц
    for lut in range(16):  # по всем 16 двухвходовым таблицам
        for a, b in itertools.product((0, 1), repeat=2):  # по всем парам входов
            assert lut2(lut, a, b) == lut >> (a | b << 1) & 1  # прямая адресация бита совпадает с lut2
            lut_checks += 1  # проверок +1
    tests.append({'name': 'all_lut2_functions', 'pass': True, 'checks': lut_checks})  # записать проверку
    w = make_world(45001)  # мир с фиксированным зерном
    r = learn_active(w)  # обучиться
    assert r['status'] == 'PROVISIONAL_SAMPLED'  # модель должна найтись
    assert r['model']['digest'] == model_digest(transition_table(w.roles, w.f, w.g))  # её digest равен digest истинной таблицы
    tests.append({'name': 'active_compositional_discovery', 'pass': True, 'queries': r['queries'], 'candidate_start': r['candidate_start']})  # записать проверку
    o = run_ood(45002)  # прогон чужеродного мира
    assert o['status'] == 'PASS' and o['fallback_transition_pairs'] == 48 and (o['anf_checks'] == 64)  # 48 пар, 64 проверки АНФ, успех
    tests.append({'name': 'ood_full_table_and_anf_expansion', 'pass': True, 'pairs': 48, 'anf_checks': 64})  # записать проверку
    tab = transition_table(w.roles, w.f, w.g)  # истинная таблица
    original = model_digest(tab)  # её digest
    tab[3][w.clock_action] = tab[3][w.clock_action] ^ 1  # испортить одну клетку: XOR 1
    assert model_digest(tab) != original  # digest обязан измениться
    tests.append({'name': 'checkpoint_model_tamper_detection', 'pass': True})  # записать проверку
    adv = run_adversarial_unseen_transition(45003)  # непосещённый переход
    assert adv['status'] == 'PASS'  # должен быть обнаружен
    tests.append({'name': 'unseen_transition_defeats_sampled_audit_but_full_audit_catches_it', 'pass': True, 'target': adv['target'], 'full_pairs_checked': adv['pairs_checked']})  # записать проверку
    recovered = []  # результаты восстановления правила
    for j in range(8):  # 8 чужеродных миров
        rr = run_ood(45100 + j)  # прогон
        recovered.append(rr.get('novel_rule_exact', False))  # записать, восстановлено ли правило точно
    assert all(recovered)  # правило восстановлено во всех прогонах
    tests.append({'name': 'recover_novel_three_input_lut_from_ood_full_table', 'pass': True, 'trials': len(recovered)})  # записать проверку
    return tests  # вернуть список проверок
def exact_audit_record(run):
    """Точный аудит успешного запуска: найти модель по digest и проверить все 48 пар."""
    if run.get('status') not in ('PASS_SAMPLED', 'PASS'):  # если запуск не успешен
        return {'seed': run.get('seed'), 'status': 'HOLD_NO_CANDIDATE'}  # статус HOLD_NO_CANDIDATE
    world = make_world(run['seed'])  # восстановить мир по зерну
    c = run['clock_action']  # действие часов
    models = generate_models(c)  # банк моделей
    matches = [m for m in models if m['digest'] == run.get('model_digest')]  # модели с digest из запуска
    if len(matches) != 1:  # должна быть ровно одна
        return {'seed': run['seed'], 'status': 'HOLD_CANDIDATE_NOT_IN_GRAMMAR', 'match_count': len(matches)}  # статус HOLD_CANDIDATE_NOT_IN_GRAMMAR
    audit = audit_candidate_full(world, c, matches[0])  # полный аудит кандидата
    return {'seed': run['seed'], 'status': 'PROVEN_EXACT' if audit['pass'] else 'HOLD_COUNTEREXAMPLE', 'pairs_checked': audit['pairs_checked'], 'total_pairs': audit['total_pairs'], 'interactions': audit['interactions'], 'mismatches': audit['mismatches'], 'candidate_digest': run.get('model_digest')}  # PROVEN_EXACT, если аудит пройден, иначе HOLD_COUNTEREXAMPLE
def run_panel(start_seed, n, outdir):
    """Панель прогонов: n зёрен, активная и случайная стратегии, чужеродные миры; вернуть сводку."""
    outdir.mkdir(parents=True, exist_ok=True)  # создать каталог результатов
    active = []  # прогоны активной стратегии
    random_runs = []  # прогоны случайной стратегии
    ood = []  # прогоны чужеродных миров
    for i in range(n):  # по n зернам
        seed = start_seed + i  # очередное зерно
        active.append(run_seed(seed, 'active'))  # активный прогон
        random_runs.append(run_seed(seed, 'random'))  # случайный прогон
        ood.append(run_ood(seed + 100000))  # чужеродный прогон со сдвигом зерна на 100000
        if (i + 1) % 5 == 0:  # каждые 5 прогонов
            (outdir / f'progress_{i + 1}.json').write_text(json.dumps({'done': i + 1}, indent=2))  # записать файл прогресса
    active_exact = [exact_audit_record(x) for x in active]  # точный аудит каждого активного прогона
    random_exact = [exact_audit_record(x) for x in random_runs]  # точный аудит каждого случайного прогона

    def avg(xs):
        """Среднее арифметическое списка; 0.0 для пустого."""
        return statistics.mean(xs) if xs else 0.0  # среднее арифметическое, 0.0 для пустого списка
    a_ok = sum((x.get('status') in ('PASS', 'PASS_SAMPLED') for x in active))  # число успешных активных прогонов
    r_ok = sum((x.get('status') in ('PASS', 'PASS_SAMPLED') for x in random_runs))  # число успешных случайных прогонов
    o_ok = sum((x.get('status') == 'PASS' for x in ood))  # число успешных чужеродных прогонов
    good = lambda x: x.get('status') in ('PASS', 'PASS_SAMPLED')  # признак успешного прогона (обучен и план исполнен)
    a_q = [x.get('queries', 0) for x in active if good(x)]  # число запросов успешных активных прогонов
    r_q = [x.get('queries', 0) for x in random_runs if good(x)]  # число запросов успешных случайных прогонов
    a_cost = [x.get('learn_interactions', 0) for x in active if good(x)]  # взаимодействия успешных активных прогонов
    r_cost = [x.get('learn_interactions', 0) for x in random_runs if good(x)]  # взаимодействия успешных случайных прогонов
    o_fallback = [x.get('fallback_interactions', 0) for x in ood if x.get('status') == 'PASS']  # стоимость построения полной таблицы в успешных чужеродных прогонах
    summary = {'version': '4.6', 'seeds': n, 'active_success': a_ok, 'random_success': r_ok, 'ood_success': o_ok, 'active_mean_queries': avg(a_q), 'random_mean_queries': avg(r_q), 'active_mean_interactions': avg(a_cost), 'random_mean_interactions': avg(r_cost), 'active_interaction_reduction_pct': (1 - avg(a_cost) / avg(r_cost)) * 100 if avg(r_cost) else 0, 'ood_full_table_pairs_mean': avg([x.get('fallback_transition_pairs', 0) for x in ood]), 'ood_fallback_interactions_mean': avg(o_fallback), 'ood_total_interactions_mean': avg([x.get('ood_total_interactions', 0) for x in ood if x.get('status') == 'PASS']), 'ood_inferred_novel_rules': sum((bool(x.get('novel_rule_exact')) for x in ood)), 'active_exact_commit_passes': sum((x.get('status') == 'PROVEN_EXACT' for x in active_exact)), 'random_exact_commit_passes': sum((x.get('status') == 'PROVEN_EXACT' for x in random_exact)), 'active_total_cost_mean_with_exact_audit': avg([a.get('learn_interactions', 0) + e.get('interactions', 0) for a, e in zip(active, active_exact)]), 'random_total_cost_mean_with_exact_audit': avg([a.get('learn_interactions', 0) + e.get('interactions', 0) for a, e in zip(random_runs, random_exact)]), 'exact_commit_policy': 'PROVEN_EXACT_ONLY_AFTER_48_PAIR_AUDIT', 'mean_plan_length': avg([x.get('plan_length', 0) for x in active if good(x)]), 'candidate_commit_status': 'PROVISIONAL_SAMPLED_ONLY', 'active_candidate_start': active[0].get('candidate_start'), 'all_ood_anf_checks': all((x.get('anf_verified', False) for x in ood if x.get('status') == 'PASS'))}  # сводка: успехи, средние значения, снижение стоимости в процентах = (1 − среднее_a/среднее_r)·100
    (outdir / 'active_runs.json').write_text(json.dumps(active, indent=2))  # сохранить активные прогоны
    (outdir / 'random_runs.json').write_text(json.dumps(random_runs, indent=2))  # сохранить случайные прогоны
    (outdir / 'ood_runs.json').write_text(json.dumps(ood, indent=2))  # сохранить чужеродные прогоны
    (outdir / 'active_exact_audits.json').write_text(json.dumps(active_exact, indent=2))  # сохранить точные аудиты активных прогонов
    (outdir / 'random_exact_audits.json').write_text(json.dumps(random_exact, indent=2))  # сохранить точные аудиты случайных прогонов
    (outdir / 'panel_summary.json').write_text(json.dumps(summary, indent=2))  # сохранить сводку
    return summary  # вернуть сводку
def main():
    """Точка входа командной строки: --self-test (самопроверка) или панель прогонов."""
    ap = argparse.ArgumentParser()  # парсер аргументов
    ap.add_argument('--self-test', action='store_true')  # флаг самопроверки
    ap.add_argument('--panel', type=int, default=20)  # число прогонов панели
    ap.add_argument('--seed', type=int, default=20261120)  # стартовое зерно
    ap.add_argument('--out', type=Path, default=Path('results/panel20'))  # каталог результатов
    args = ap.parse_args()  # разобрать аргументы
    if args.self_test:  # если запрошена самопроверка
        tests = self_tests()  # выполнить проверки
        print(json.dumps({'all_pass': all((t['pass'] for t in tests)), 'tests': tests}, indent=2))  # напечатать результат: все проверки пройдены и их список
        return  # выход
    t0 = time.time()  # время начала
    summary = run_panel(args.seed, args.panel, args.out)  # прогнать панель
    summary['elapsed_seconds'] = round(time.time() - t0, 3)  # затраченное время в секундах
    print(json.dumps(summary, indent=2))  # напечатать сводку
if __name__ == '__main__':
    main()  # запуск основной функции

# ============================================================
# КОМПИЛЯТОР: таблица среды -> GF(2) (общий)
# ============================================================

from hagi import skill_engine as v40  # импорт skill_engine как v40 (преобразование Мёбиуса и компилятор АНФ)
def compile_transition_table(table, n_bits: int, n_act_bits: int) -> dict:
    """Скомпилировать таблицу переходов в сеть вентилей.

        table[s][a] — следующее состояние (целое). Возвращает словарь с ключами net (сеть),
        anf (АНФ), gates (число вентилей), depth (глубина) и behavioral_exact — признак
        поведенческой точности, проверенной на ВСЕХ парах (s, a)."""
    n = n_bits + n_act_bits  # число входных переменных n = n_bits + n_act_bits
    n_states, n_actions = (1 << n_bits, 1 << n_act_bits)  # число состояний 2^n_bits и число действий 2^n_act_bits
    rows: list = [None] * (n_states * n_actions)  # строки таблицы истинности, индекс = s | a<<n_bits
    for s in range(n_states):  # по состояниям
        for a in range(n_actions):  # по действиям
            ns = table[s][a]  # следующее состояние
            rows[s | a << n_bits] = tuple((ns >> i & 1 for i in range(n_bits)))  # кортеж битов следующего состояния: ((ns >> i) & 1, i = 0..n_bits−1)
    anf = v40.mobius(rows, n)  # АНФ по всем выходным битам (преобразование Мёбиуса)
    net = v40.ProofCarryingANFLearner.compile_optimal(n, anf)  # скомпилировать оптимальную сеть вентилей

    def ev(s, a):
        """Выход сети на паре (состояние s, действие a) — предсказанное следующее состояние."""
        wv = [s >> i & 1 for i in range(n_bits)] + [a >> i & 1 for i in range(n_act_bits)]  # входы: биты состояния, затем биты действия
        for g in net['gates']:  # по вентилям в топологическом порядке
            x, y, m = (wv[g['a']], wv[g['b']], g['mask'])  # входы x, y и тип вентиля m
            wv.append(0 if m == 0 else 1 if m == 15 else x ^ y if m == 6 else x & y)  # выход вентиля: 0 при m = 0; 1 при m = 15; x XOR y при m = 6; иначе x AND y
        return sum((wv[net['outs'][o]] << o for o in range(n_bits)))  # выход = Σ_o wv[outs[o]] << o
    ok = all((ev(s, a) == table[s][a] for s in range(n_states) for a in range(n_actions)))  # поведенческая точность: ev(s, a) == table[s][a] на всех парах
    if not ok:  # если хоть одна пара не совпала
        raise RuntimeError('bridge netlist behavioral mismatch')  # ошибка: поведение сети расходится с таблицей
    return {'net': net, 'table': table, 'anf': anf, 'gates': len(net['gates']), 'depth': net['metrics']['depth'], 'behavioral_exact': True}  # вернуть сеть, таблицу, АНФ, число вентилей, глубину и признак точности

# ============================================================
# СВЯЗКА ENV LUT: мир -> нетлист
# ============================================================

from hagi import env_lut as env  # импорт этого же модуля как env
def env_model_to_netlist(seed: int=777, ood: bool=False) -> dict:
    """Активно обучиться миру, провести аудит кандидата или полный перебор и скомпилировать таблицу в нетлист."""
    w = env.make_world(seed, ood=ood)  # мир по зерну
    res = env.learn_active(w)  # активное обучение
    out = {'learn_status': res['status'], 'probes': res['queries'], 'clock_action': res['clock_action'], 'ood': ood}  # заготовка результата: статус, число зондов, часы, признак чужеродной динамики
    if res['status'] == 'PROVISIONAL_SAMPLED':  # если найден выборочный кандидат
        audit = env.audit_candidate_full(w, res['clock_action'], res['model'])  # полный аудит кандидата
        table, audit_inter, _ = env.table_from_full_audit(audit, res['clock_action'])  # таблица из аудита
        out['audit_interactions'] = audit_inter  # стоимость аудита
        out['proven'] = bool(audit.get('complete'))  # доказано, если аудит полон
    else:  # иначе
        table, inter2, _ = env.full_table_fallback(w, res['clock_action'])  # таблица прямым перебором
        out['audit_interactions'] = inter2  # стоимость перебора
        out['proven'] = False  # не доказано
        rule = env.infer_deep_ood_rule(table, res['clock_action'])  # восстановить правило чужеродной динамики
        out['inferred_ood_rule'] = rule  # записать правило
    out.update(compile_transition_table(table, n_bits=4, n_act_bits=2))  # добавить результат компиляции для 4 бит состояния и 2 бит действия
    return out  # вернуть результат

# ============================================================
# ENV LUT VERIFY: независимое воспроизведение
# ============================================================

from hagi.util import sha  # хеш-функция sha
import argparse, json, random  # стандартные модули: аргументы, json, случайность
from collections import deque, Counter  # двусторонняя очередь и счётчик
from pathlib import Path  # класс Path
N = 16  # число состояний N = 16
A = 4  # число действий A = 4
def b(s, i):
    """Бит i (с младшего) числа s: (s >> i) & 1."""
    return s >> i & 1  # бит i числа s: (s >> i) & 1
def ob(s):
    """Наблюдение состояния: биты 0 и 1 без изменений, бит 2 = x2 XOR x3."""
    return b(s, 0) | b(s, 1) << 1 | (b(s, 2) ^ b(s, 3)) << 2  # наблюдение: y0 = x0, y1 = x1, y2 = x2 XOR x3
def clock(s):
    """Роль часов: следующее состояние (s + 1) mod 16."""
    return s + 1 & 15  # часы: (s + 1) mod 16
def rotate(s):
    """Циклический сдвиг влево на 1 бит: ((s << 1) & 15) | (s >> 3)."""
    return s << 1 & 15 | s >> 3  # циклический сдвиг влево на 1 бит
def lut2(lut, x, y):
    """Значение двухвходовой таблицы lut: lut >> (x | 2·y) & 1."""
    return lut >> (x | y << 1) & 1  # значение двухвходовой таблицы: бит номер x + 2·y числа lut
def transition(s, role, f, g, f3=None, ood=False):
    """Независимая реализация перехода: роль clock, shift, toggle или coupled (обычный или чужеродный)."""
    x0, x1, x2, x3 = (b(s, i) for i in range(4))  # биты входа
    if role == 'clock':  # роль clock:
        return clock(s)  # вернуть clock(s)
    if role == 'shift':  # роль shift:
        return rotate(s)  # вернуть rotate(s)
    if role == 'toggle':  # роль toggle:
        return s ^ (4 if x0 and (not x1) else 0)  # s XOR 4, если x0 = 1 и x1 = 0, иначе s
    if role == 'coupled' and (not ood):  # обычная связная роль:
        nx0 = x0 ^ lut2(f, x1, x2)  # новый бит 0: nx0 = x0 XOR f(x1, x2)
        nx3 = x3 ^ lut2(g, nx0, x1)  # новый бит 3: nx3 = x3 XOR g(nx0, x1)
        return nx0 | x1 << 1 | x2 << 2 | nx3 << 3  # собрать состояние
    if role == 'coupled' and ood:  # чужеродная связная роль:
        idx = x1 | x2 << 1 | x3 << 2  # адрес f3: idx = x1 | x2<<1 | x3<<2
        nx0 = x0 ^ f3 >> idx & 1  # новый бит 0: nx0 = x0 XOR f3[idx]
        nx3 = x3 ^ lut2(g, nx0, x1)  # новый бит 3: nx3 = x3 XOR g(nx0, x1)
        return nx0 | x1 << 1 | x2 << 2 | nx3 << 3  # собрать состояние
    raise ValueError(role)  # неизвестная роль — ошибка
def reconstruct_world(seed, ood=False):
    """Заново построить мир по зерну (тот же генератор, что и при создании).

        Возвращает (roles, f, g, f3, goal, table) — параметры и истинную таблицу переходов."""
    rng = random.Random(seed ^ 12648430)  # тот же генератор, что в make_world: seed XOR 12648430
    roles = ['clock', 'shift', 'toggle', 'coupled']  # список ролей
    rng.shuffle(roles)  # перемешать
    f = rng.randint(1, 14)  # f — случайное из 1..14
    g = rng.randint(1, 14)  # g — случайное из 1..14
    f3 = 128  # f3 = 128 (по умолчанию)
    if ood:  # если чужеродная динамика
        choices = [x for x in range(1, 256) if x.bit_count() & 1 and x not in (129, 165)]  # варианты: нечётное число единиц, кроме 129 и 165
        f3 = rng.choice(choices)  # случайный выбор f3
    goal = rng.randrange(1, 16)  # цель — случайное из 1..15
    table = [[transition(s, roles[a], f, g, f3, ood) for a in range(A)] for s in range(N)]  # истинная таблица: table[s][a] = transition(s, roles[a], f, g, f3, ood)
    return (roles, f, g, f3, goal, table)  # вернуть роли, таблицы, цель и таблицу переходов
def sig_for(table, state, action, suffix):
    """Сигнатура зонда по истинной таблице: наблюдение после (state, action) и suffix шагов часами."""
    s = table[state][action]  # s = table[state][action] — состояние после действия
    out = [ob(s)]  # первое наблюдение ob(s)
    ca = next((i for i, r in enumerate(ROLES) if r == 'clock'))  # номер действия часов в ROLES
    for _ in range(suffix):  # suffix раз
        s = table[s][ca]  # s = table[s][ca] — шаг часами
        out.append(ob(s))  # добавить наблюдение
    return out  # вернуть список наблюдений
def bfs(table, goal):
    """Поиск в ширину по таблице от состояния 0 до goal; вернуть список действий или None."""
    q = deque([0])  # очередь из состояния 0
    prev = {0: None}  # предок состояния (у 0 — None)
    pa = {}  # действие, которым пришли в состояние
    while q:  # пока очередь не пуста
        s = q.popleft()  # достать s
        if s == goal:  # цель достигнута
            break  # выход
        for a in range(A):  # по действиям
            t = table[s][a]  # новое состояние: t = table[s][a]
            if t not in prev:  # t ещё не посещено
                prev[t] = s  # запомнить предка
                pa[t] = a  # и действие
                q.append(t)  # в очередь
    if goal not in prev:  # цель недостижима
        return None  # вернуть None
    out = []  # путь (в обратном порядке)
    s = goal  # текущее состояние — цель
    while prev[s] is not None:  # пока не дошли до 0
        out.append(pa[s])  # добавить действие
        s = prev[s]  # перейти к предку
    return list(reversed(out))  # вернуть в прямом порядке
def verify_plan(table, goal, path, trace):
    """Сверить трассу исполнения с таблицей; вернуть True, если всё совпало и цель достигнута."""
    s = 0  # s = 0 — состояние по таблице
    if path is None or len(path) != len(trace):  # план должен существовать, и его длина равна длине трассы
        return False  # неудача
    for idx, a in enumerate(path):  # по шагам плана
        s = table[s][a]  # новое состояние: s = table[s][a]
        row = trace[idx]  # строка трассы
        if row.get('action') != a or row.get('predicted_state') != s or row.get('observed') != ob(s):  # действие, предсказанное состояние и наблюдение должны совпадать с ob(s)
            return False  # неудача
    return s == goal and len(path) == len(bfs(table, goal) or [])  # успех: достигнута цель и длина плана равна длине кратчайшего пути
def verify_run(run, ood=False):
    """Независимая проверка запуска: восстановить мир по зерну и сверить все заявления запуска."""
    seed = run['seed']  # зерно запуска
    roles, f, g, f3, goal, table = reconstruct_world(seed, ood)  # заново построить истинный мир по зерну
    clock_a = roles.index('clock')  # истинное действие часов
    checks = {'seed': seed, 'pass': True, 'probes': 0, 'plan': False, 'table_digest': False}  # заготовка результата проверок

    def fail(msg):
        """Пометить проверку как проваленную и записать сообщение msg."""
        checks['pass'] = False  # пометить провал
        checks.setdefault('errors', []).append(msg)  # добавить сообщение об ошибке
    if run.get('clock_action') != clock_a:  # действие часов должно совпадать
        fail('wrong clock action')  # ошибка: неверные часы
    if not ood:  # для обычного мира:
        if run.get('status') != 'PASS_SAMPLED':  # статус должен быть PASS_SAMPLED
            fail('wrong status label')  # ошибка: неверный статус
        if run.get('model_evidence') != 'SAMPLED_ONLY' or run.get('validation_status') != 'PASS_SAMPLED_ONLY':  # модель должна быть помечена как выборочная
            fail('model not marked provisional')  # ошибка: модель не помечена как предварительная
        if run.get('candidate_left') != 1:  # должен остаться один кандидат
            fail('candidate not unique')  # ошибка: кандидат не единственный
        if sha(table) != run.get('model_digest'):  # digest кандидата равен digest истинной таблицы
            fail('candidate digest differs from true model')  # ошибка: digest отличается от истинной модели
        if run.get('f') != f or run.get('g') != g:  # скрытые параметры f и g должны совпасть
            fail('latent parameters mismatch')  # ошибка: параметры не совпали
        for item in run.get('evidence', []):  # по журналам зондов
            if sig_for(table, item['state'], item['action'], item['suffix']) != item['obs']:  # сигнатура из истинной таблицы должна равняться наблюдению
                fail('training evidence mismatch')  # ошибка: обучающее свидетельство не совпало
            checks['probes'] += 1  # зондов +1
        for item in run.get('validation', []):  # по выборочным проверкам
            if sig_for(table, item['state'], item['action'], 4) != item['observed']:  # сигнатура длины 5 должна совпасть с наблюдением
                fail('holdout validation mismatch')  # ошибка: контрольное наблюдение не совпало
            if item['observed'] != item['predicted']:  # наблюдение должно совпадать с предсказанием
                fail('holdout prediction mismatch')  # ошибка: контрольное предсказание не совпало
            checks['probes'] += 1  # зондов +1
        if len(run.get('validation', [])) != 6:  # контрольных проверок должно быть 6
            fail('unexpected holdout count')  # ошибка: неверное число проверок
        expected_cost = 64 + sum((x['cost'] for x in run.get('evidence', []))) + sum((x['cost'] for x in run.get('validation', [])))  # ожидаемая стоимость: 64 шага на часы + стоимости всех зондов и проверок
        if expected_cost != run.get('learn_interactions'):  # она должна равняться заявленным взаимодействиям
            fail('interaction accounting mismatch')  # ошибка: учёт взаимодействий не сходится
        if run.get('goal') != goal:  # цель должна совпасть с истинной
            fail('goal mismatch')  # ошибка: цель не совпала
        path = bfs(table, goal)  # кратчайший путь (для проверки длины плана)
        checks['plan'] = verify_plan(table, goal, run.get('plan'), run.get('execution', []))  # сверить план и трассу запуска с таблицей
        if not checks['plan']:  # если не сошлось
            fail('plan is not shortest or trace mismatch')  # ошибка: план не кратчайший или трасса не совпала
        checks['table_digest'] = True  # digest таблицы проверен
    else:  # для чужеродного мира:
        if run.get('status') != 'PASS':  # статус должен быть PASS
            fail('OOD full-table task not pass')  # ошибка: задача с полной таблицей не пройдена
        if run.get('true_f3') != f3 or run.get('true_g') != g:  # истинные f3 и g должны совпасть с восстановленным миром
            fail('OOD generator params mismatch')  # ошибка: параметры генератора не совпали
        if run.get('novel_rule_exact') is not True:  # правило должно быть восстановлено точно
            fail('novel rule not exactly recovered')  # ошибка: правило восстановлено неточно
        inf = run.get('inferred_novel_rule') or {}  # восстановленное правило
        if inf.get('f3') != f3 or inf.get('g') != g or roles[inf.get('action', -1)] != 'coupled':  # f3, g и действие (роль coupled) должны совпасть с истинными
            fail('inferred mechanism mismatch')  # ошибка: восстановленный механизм не совпал
        rows = run.get('rows', [])  # строки полной таблицы
        expected = {(s, a) for s in range(N) for a in range(A) if a != clock_a}  # ожидаемое множество 48 нечасовых пар
        got = {(r.get('state'), r.get('action')) for r in rows}  # множество пар из строк
        if got != expected or len(rows) != 48:  # пары должны совпасть, строк ровно 48
            fail('incomplete or duplicate full transition table')  # ошибка: таблица неполна или содержит дубликаты
        for r in rows:  # по строкам
            if r.get('next_state') != table[r['state']][r['action']]:  # следующее состояние должно совпадать с истинной таблицей
                fail('full transition row mismatch')  # ошибка: строка таблицы не совпала
        if run.get('fallback_transition_pairs') != 48:  # число пар должно быть 48
            fail('wrong transition pair count')  # ошибка: неверное число пар
        expected_cost = sum((r['cost'] for r in rows))  # ожидаемая стоимость = сумма стоимостей строк
        if expected_cost != run.get('fallback_interactions'):  # она должна равняться заявленной
            fail('full fallback cost mismatch')  # ошибка: стоимость перебора не совпала
        if run.get('ood_total_interactions') != run.get('active_interactions_before_hold', 0) + expected_cost:  # всего = взаимодействий до остановки + стоимость перебора
            fail('OOD total interaction accounting mismatch')  # ошибка: общий учёт взаимодействий не сходится
        if run.get('grammar_status') not in ('GRAMMAR_MISMATCH', 'PROVISIONAL_REJECTED_BY_FULL_AUDIT'):  # статус грамматики должен быть одним из двух ожидаемых
            fail('unexpected grammar status')  # ошибка: неожиданный статус грамматики
        audit = run.get('sampled_candidate_full_audit')  # аудит выборочного кандидата
        if run.get('grammar_status') == 'PROVISIONAL_REJECTED_BY_FULL_AUDIT':  # если кандидат был отвергнут полным аудитом:
            if not audit or audit.get('pass') or audit.get('pairs_checked') != 48:  # аудит должен быть, не пройден и охватывать 48 пар
                fail('failed to reject sampled candidate by full audit')  # ошибка: полный аудит не отверг выборочного кандидата
            if len(audit.get('mismatches', [])) == 0:  # должно быть хоть одно несовпадение
                fail('full audit had no mismatch')  # ошибка: в полном аудите нет несовпадений
        anfs = run.get('anfs', [])  # АНФ из запуска
        if len(anfs) != A or any((len(x) != 4 for x in anfs)):  # по 4 действиям, в каждом 4 бита выхода
            fail('ANF arity')  # ошибка: неверная форма АНФ
        for s in range(N):  # по состояниям
            for a in range(A):  # по действиям
                predicted = sum((1 << bit_i for bit_i, terms in enumerate(anfs[a]) if sum((1 for t in terms if s & int(t) == int(t))) & 1))  # предсказание АНФ: predicted = Σ_b 2^b·(число мономов бита b, входящих в s, mod 2)
                if predicted != table[s][a]:  # должно совпасть с истинной таблицей
                    fail('ANF value mismatch')  # ошибка: значение АНФ не совпало
                checks['probes'] += 1  # проверок +1
        if run.get('goal') != goal:  # цель должна совпасть
            fail('goal mismatch')  # ошибка: цель не совпала
        checks['plan'] = verify_plan(table, goal, run.get('plan'), run.get('execution_trace', []))  # сверить план и трассу
        if not checks['plan']:  # если не сошлось
            fail('OOD plan mismatch')  # ошибка: план чужеродного мира не совпал
        if sha(table) != run.get('model_digest'):  # digest истинной таблицы равен digest из запуска
            fail('OOD table digest mismatch')  # ошибка: digest таблицы не совпал
        checks['table_digest'] = True  # digest проверен
    return checks  # вернуть результаты проверок
ROLES = None  # глобальная переменная ROLES (роли текущего проверяемого мира), пока None
def verify_bundle(root: Path):
    """Независимо проверить пакет результатов панели: все прогоны и записи точных аудитов."""
    global ROLES  # изменять глобальную ROLES
    p = root / 'results' / 'panel20'  # каталог результатов панели
    active = json.loads((p / 'active_runs.json').read_text())  # загрузить активные прогоны
    random_runs = json.loads((p / 'random_runs.json').read_text())  # загрузить случайные прогоны
    ood = json.loads((p / 'ood_runs.json').read_text())  # загрузить чужеродные прогоны
    active_audits = json.loads((p / 'active_exact_audits.json').read_text())  # загрузить точные аудиты активных прогонов
    random_audits = json.loads((p / 'random_exact_audits.json').read_text())  # загрузить точные аудиты случайных прогонов
    out = {'active': [], 'random': [], 'ood': [], 'active_exact_audits': [], 'random_exact_audits': []}  # результаты по группам
    for target, items, is_ood in (('active', active, False), ('random', random_runs, False), ('ood', ood, True)):  # по трём группам прогонов
        for run in items:  # по прогонам группы
            roles, *_ = reconstruct_world(run['seed'], is_ood)  # восстановить роли мира по зерну
            ROLES = roles  # запомнить роли для sig_for
            out[target].append(verify_run(run, is_ood))  # добавить результат независимой проверки
    for target, audits, runs in (('active_exact_audits', active_audits, active), ('random_exact_audits', random_audits, random_runs)):  # по двум группам точных аудитов
        for audit, run in zip(audits, runs):  # пары (запись аудита, прогон)
            roles, f, g, f3, goal, table = reconstruct_world(run['seed'], False)  # восстановить обычный мир по зерну
            ok = audit.get('status') == 'PROVEN_EXACT' and audit.get('pairs_checked') == 48 and (audit.get('total_pairs') == 64) and (audit.get('interactions') == 600) and (audit.get('candidate_digest') == sha(table)) and (audit.get('seed') == run.get('seed'))  # запись верна: статус PROVEN_EXACT, 48 пар, всего 64, 600 взаимодействий, digest и зерно совпадают
            out[target].append({'seed': run['seed'], 'pass': ok, 'pairs_checked': audit.get('pairs_checked'), 'status': audit.get('status'), 'errors': [] if ok else ['exact audit record invalid']})  # добавить результат проверки записи
    summary = {'active_pass': sum((x['pass'] for x in out['active'])), 'random_pass': sum((x['pass'] for x in out['random'])), 'ood_pass': sum((x['pass'] for x in out['ood'])), 'active_exact_audits_pass': sum((x['pass'] for x in out['active_exact_audits'])), 'random_exact_audits_pass': sum((x['pass'] for x in out['random_exact_audits'])), 'total': sum((len(v) for v in out.values())), 'all_pass': all((x['pass'] for k in out for x in out[k])), 'ood_rule_recoveries': sum((1 for x in ood if x.get('novel_rule_exact'))), 'panel_summary': json.loads((p / 'panel_summary.json').read_text())}  # сводка: число пройденных по группам, общее число и признак «все пройдены»
    result = {'summary': summary, 'details': out}  # полный результат
    (root / 'results' / 'independent_verification.json').write_text(json.dumps(result, indent=2))  # записать результат в файл независимой проверки
    return result  # вернуть результат
def main():
    """Точка входа командной строки: проверить пакет в каталоге --root и напечатать сводку."""
    ap = argparse.ArgumentParser()  # парсер аргументов
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)  # корневой каталог (по умолчанию — каталог этого файла)
    args = ap.parse_args()  # разобрать аргументы
    r = verify_bundle(args.root)  # независимая проверка пакета результатов
    print(json.dumps(r['summary'], indent=2))  # напечатать сводку
    raise SystemExit(0 if r['summary']['all_pass'] else 1)  # код выхода 0, если всё пройдено, иначе 1
if __name__ == '__main__':
    main()  # запуск основной функции