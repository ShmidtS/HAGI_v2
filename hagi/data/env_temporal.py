"""Среда ENV TEMPORAL: модель мира, связь с таблицами и независимый верификатор.

Мир: 5-битное состояние (32 состояния), 4 действия; одно — часы (s + 1) mod 32,
остальные три — роли xor (s XOR xor_mask), guard (s XOR effect_mask, если
s & guard_mask == guard_value) и delay (бит 4 — флаг ожидания: первый вызов
запоминает младшие биты и ставит флаг, второй применяет delay_mask).
Состояние наблюдается полностью.

Банк моделей: 12 наборов параметров с фиксированным зерном. Кандидат — (модель,
перестановка ролей). Часы находятся по орбите +1; модель — активными запросами
пары (s, a) с максимальной энтропией разбиения кандидатов. Хранится также
минимальный различающий набор пар (поиск в ширину по маскам альтернатив)."""
from __future__ import annotations  # отложенное вычисление аннотаций типов


# ============================================================
# ENV TEMPORAL: 5-битный мир, активное отождествление
# ============================================================

from hagi.base.util import sha  # хеш-функция sha
import argparse, itertools, json, math, random, statistics  # стандартные модули: аргументы командной строки, перестановки, json, math, случайность, статистика, время
from collections import deque  # двусторонняя очередь для поиска в ширину
from dataclasses import dataclass, field  # декоратор dataclass и фабрика полей по умолчанию
from pathlib import Path  # класс Path для путей
from typing import Iterable  # тип Iterable для аннотаций
N_BITS = 5  # число бит состояния N_BITS = 5
N_STATES = 1 << N_BITS  # число состояний N_STATES = 2^5 = 32
N_ACTIONS = 4  # число действий N_ACTIONS = 4
TRUSTED_FAMILY = 'TEMPORAL_AFFINE_DELAY_FAMILY_V1'  # имя доверенного семейства моделей
TRUSTED_ISSUER = 'HAGI_TEMPORAL_SIM_V1'  # имя доверенного издателя мира
ROLES = ('xor', 'guard', 'delay')  # нечасовые роли: xor, guard, delay
def clock_fn(s: int) -> int:
    """Часы: следующее состояние (s + 1) mod 32."""
    return (s + 1) % N_STATES  # часы: (s + 1) mod 32
def role_fn(role: str, s: int, p: dict) -> int:
    """Переход роли role из состояния s с параметрами p.

        clock: (s + 1) mod 32; xor: s XOR xor_mask; guard: s XOR effect_mask, если
        s & guard_mask == guard_value, иначе s; delay: бит 4 — флаг ожидания (0 — запомнить
        младшие биты и поставить флаг: low | 16; 1 — применить маску: low XOR delay_mask)."""
    if role == 'clock':  # роль clock:
        return clock_fn(s)  # вернуть clock_fn(s)
    if role == 'xor':  # роль xor:
        return s ^ int(p['xor_mask'])  # побитовое исключающее или: s XOR xor_mask
    if role == 'guard':  # роль guard:
        if s & int(p['guard_mask']) == int(p['guard_value']):  # если s & guard_mask == guard_value:
            return s ^ int(p['effect_mask'])  # побитовое исключающее или: s XOR effect_mask
        return s  # иначе s без изменений
    if role == 'delay':  # роль delay:
        pending = s >> 4 & 1  # ожидающий бит: бит 4 состояния
        low = s & 15  # младшие 4 бита
        if pending == 0:  # если ожидания нет:
            return low | 16  # запомнить младшие биты и выставить бит ожидания: low | 16
        return low ^ int(p['delay_mask'])  # если ожидание было: применить маску к запомненным битам: low XOR delay_mask
    raise ValueError(role)  # неизвестная роль — ошибка
def generate_bank(n_models: int=12) -> list[dict]:
    """Сгенерировать банк из n_models моделей с фиксированным зерном.

        Каждая модель: xor_mask = 6, случайные guard_mask (>= 2 единиц), guard_value
        (подмножество guard_mask), effect_mask, delay_mask. Отбрасываются дубликаты по
        сигнатуре и модели, где роли xor, guard, delay совпадают. В конце проверяется, что
        все 12 моделей x 6 перестановок ролей дают различные полные таблицы."""
    rng = random.Random(440044)  # генератор с фиксированным зерном 440044
    bank = []  # банк моделей
    seen = set()  # множество уже виденных сигнатур
    guard_masks = [m for m in range(1, 32) if m.bit_count() >= 2]  # маски guard с числом единиц не меньше 2
    while len(bank) < n_models:  # пока в банке меньше n_models моделей
        p = {'xor_mask': 6, 'guard_mask': rng.choice(guard_masks), 'guard_value': 0, 'effect_mask': rng.choice([1, 2, 4, 8, 16, 3, 5, 6, 9, 10, 12]), 'delay_mask': rng.randint(1, 15)}  # параметры модели: xor_mask = 6, случайные guard_mask, effect_mask, delay_mask
        sub = [x for x in range(1, 32) if x & p['guard_mask'] == x]  # подмножества guard_mask (включая само): x & guard_mask == x
        p['guard_value'] = rng.choice(sub)  # случайное guard_value из подмножеств
        sig = tuple((role_fn(r, s, p) for r in ('xor', 'guard', 'delay') for s in range(N_STATES)))  # сигнатура модели: переходы трёх ролей по всем 32 состояниям
        if sig in seen:  # если такая сигнатура уже есть
            continue  # пропустить
        role_sigs = [tuple((role_fn(r, s, p) for s in range(N_STATES))) for r in ('xor', 'guard', 'delay')]  # сигнатуры каждой роли отдельно
        if len(set(role_sigs)) != 3:  # все три роли должны различаться
            continue  # пропустить
        seen.add(sig)  # запомнить сигнатуру
        p['model_id'] = len(bank)  # номер модели — её индекс в банке
        bank.append(p)  # добавить модель
    signatures = set()  # сигнатуры шаблонов (роль + перестановка)
    for p in bank:  # по моделям банка
        for perm in itertools.permutations(ROLES):  # по всем 6 перестановкам ролей
            sig = tuple((role_fn('clock', s, p) for s in range(N_STATES)))  # сигнатура начинается с часовой строки
            for role in perm:  # затем строки трёх ролей по порядку перестановки
                sig += tuple((role_fn(role, s, p) for s in range(N_STATES)))  # добавить строку роли
            if sig in signatures:  # два шаблона не могут дать одну таблицу
                raise AssertionError('duplicate mechanism template')  # ошибка: повторяющийся шаблон механизма
            signatures.add(sig)  # запомнить сигнатуру
    return bank  # вернуть банк
BANK = generate_bank()  # банк из 12 моделей, общий для всех
@dataclass  # декоратор dataclass для класса мира
class TemporalWorld:
    """5-битный мир с 4 действиями-ролями и полным наблюдением состояния.

        Состояние наблюдается напрямую (step возвращает само состояние). Мир либо из
        семейства (модель банка по model_id или свои params), либо вне семейства
        (out_of_family — случайная таблица переходов, часы сохранены)."""
    seed: int  # seed — зерно генератора мира
    model_id: int | None  # model_id — номер модели в банке (None — своя таблица)
    action_roles: tuple[str, ...]  # action_roles — роли по номерам действий
    params: dict | None = None  # params — параметры модели (None — взять из банка)
    context_id: str = 'shared'  # context_id — имя контекста, по умолчанию 'shared'
    family: str | None = TRUSTED_FAMILY  # family — семейство моделей, по умолчанию доверенное
    issuer: str = TRUSTED_ISSUER  # issuer — издатель мира, по умолчанию доверенный
    goal: int = 0  # goal — целевое состояние, по умолчанию 0
    out_of_family: bool = False  # out_of_family — мир вне семейства (случайная таблица), по умолчанию нет
    noisy_repeat: bool = False  # noisy_repeat — шум при повторном переходе, по умолчанию нет
    state: int = 0  # state — текущее состояние, по умолчанию 0
    steps: int = 0  # steps — число шагов, по умолчанию 0
    resets: int = 0  # resets — число сбросов, по умолчанию 0
    _pair_seen: dict[tuple[int, int], int] = field(default_factory=dict)  # счётчик переходов (s, a) -> число повторов, по умолчанию пустой
    _rng: random.Random = field(init=False)  # генератор мира, не задаётся в конструкторе

    def __post_init__(self):
        """После создания: генератор, параметры из банка, проверка ролей и таблица вне семейства."""
        self._rng = random.Random(self.seed ^ 17569)  # генератор из seed XOR 17569
        if self.params is None and self.model_id is not None:  # если параметры не заданы, взять их из банка по model_id
            self.params = BANK[self.model_id]  # взять параметры: params = BANK[model_id]
        if len(self.action_roles) != N_ACTIONS or self.action_roles.count('clock') != 1:  # ровно 4 действия и ровно одни часы
            raise ValueError('action_roles must map exactly one clock and 3 nonclock actions')  # ошибка: action_roles должны давать одни часы и 3 прочих действия
        if self.out_of_family and self.params is None:  # если мир вне семейства и параметры не заданы:
            self._ood_table = [[None for _ in range(N_ACTIONS)] for _ in range(N_STATES)]  # таблица вне семейства из None, 32×4
            for s in range(N_STATES):  # по состояниям
                for a, role in enumerate(self.action_roles):  # по действиям и их ролям
                    if role == 'clock':  # часы:
                        self._ood_table[s][a] = clock_fn(s)  # часовой переход известен: clock_fn(s)
                    else:  # иначе
                        self._ood_table[s][a] = self._rng.randrange(N_STATES)  # случайное следующее состояние из 0..31
        else:  # иначе (мир из семейства)
            self._ood_table = None  # таблица вне семейства не нужна

    @property  # декоратор property: читать как атрибут
    def clock_action(self) -> int:
        """Номер действия с ролью clock."""
        return self.action_roles.index('clock')  # номер действия с ролью clock

    def reset(self) -> int:
        """Сброс мира в состояние 0; возвращает 0."""
        self.state = 0  # сброс: состояние = 0
        self.resets += 1  # счётчик сбросов +1
        return self.state  # вернуть 0 — здесь состояние наблюдается полностью

    def true_transition(self, s: int, a: int) -> int:
        """Истинный переход из состояния s действием a (без шума)."""
        if self.out_of_family:  # если мир вне семейства:
            return int(self._ood_table[s][a])  # следующее состояние из случайной таблицы
        return role_fn(self.action_roles[a], s, self.params)  # иначе применить роль действия a с параметрами модели

    def step(self, action: int) -> int:
        """Шаг мира: применить действие и вернуть наблюдение нового состояния.

            Наблюдение — само состояние; при noisy_repeat и повторном переходе
            бит 0 наблюдения инвертируется."""
        if not 0 <= action < N_ACTIONS:  # действие должно быть в 0..3
            raise ValueError('action out of range')  # ошибка: действие вне диапазона
        old = self.state  # запомнить старое состояние
        nxt = self.true_transition(old, action)  # вычислить новое состояние
        self.state = nxt  # записать новое состояние
        self.steps += 1  # счётчик шагов +1
        observed = nxt  # наблюдение — само состояние (без шума)
        key = (old, action)  # ключ перехода (s, a)
        count = self._pair_seen.get(key, 0)  # сколько раз переход уже встречался
        self._pair_seen[key] = count + 1  # записать count + 1
        if self.noisy_repeat and count >= 1:  # если шум включён и переход повторный:
            observed ^= 1  # инвертировать бит 0 наблюдения
        return observed  # вернуть наблюдение
@dataclass  # декоратор dataclass для класса памяти
class AgentMemory:
    """Память агента: контексты, история и тревоги с контрольным хешем."""
    version: str = '4.4'  # version — версия формата памяти
    contexts: dict = field(default_factory=dict)  # contexts — память по контекстам
    history: list = field(default_factory=list)  # history — история
    alerts: list = field(default_factory=list)  # alerts — список тревог

    def payload(self) -> dict:
        """Сериализовать память: тело записи плюс его хеш sha (ключ digest)."""
        body = {'version': self.version, 'contexts': self.contexts, 'history': self.history, 'alerts': self.alerts}  # тело записи: версия, контексты, история, тревоги
        return {**body, 'digest': sha(body)}  # вернуть тело вместе с его хешем sha

    @classmethod  # декоратор classmethod — конструктор от класса
    def restore(cls, payload: dict) -> 'AgentMemory':
        """Восстановить память из записи; ошибка, если хеш или версия не совпадают."""
        body = dict(payload)  # копия записи
        got = body.pop('digest', None)  # изъять и запомнить хеш
        if got != sha(body) or body.get('version') != '4.4':  # хеш и версия должны совпадать
            raise ValueError('checkpoint integrity/version failure')  # ошибка: целостность или версия контрольной точки нарушены
        return cls(version=body['version'], contexts=body['contexts'], history=body['history'], alerts=body['alerts'])  # создать AgentMemory из полей записи
def candidate_records(clock_action: int, model_ids: Iterable[int] | None=None) -> list[dict]:
    """Все кандидаты: каждая модель банка в каждой перестановке ролей по нечасовым действиям."""
    nonclock = [a for a in range(N_ACTIONS) if a != clock_action]  # нечасовые действия
    mids = list(range(len(BANK))) if model_ids is None else list(model_ids)  # номера моделей: весь банк или заданный список
    out = []  # список кандидатов
    for mid in mids:  # по номерам моделей
        for perm in itertools.permutations(ROLES):  # по всем 6 перестановкам ролей
            mapping = {a: role for a, role in zip(nonclock, perm)}  # распределение ролей по нечасовым действиям
            out.append({'model_id': mid, 'mapping': mapping})  # добавить кандидата (модель, распределение ролей)
    return out  # вернуть кандидатов
def predict_candidate(cand: dict, params_bank: list[dict], state: int, action: int, clock_action: int) -> int:
    """Предсказание кандидата: часы дают (state + 1) mod 32, иначе роль с параметрами модели."""
    if action == clock_action:  # часовое действие:
        return clock_fn(state)  # вернуть clock_fn(state)
    return role_fn(cand['mapping'][action], state, params_bank[cand['model_id']])  # иначе применить роль действия с параметрами модели кандидата
def identify_clock(world: TemporalWorld) -> tuple[int | None, dict, int]:
    """Найти единственное действие часов (+1 mod 32); при коллизиях проверить полную орбиту.

        Возвращает (номер действия, наблюдения, стоимость); −2 — недетерминированные часы,
        None — часы не найдены."""
    observations = {}  # словарь наблюдений: (состояние, действие) -> наблюдение
    cost = 0  # счётчик взаимодействий (сбросы и шаги)
    found = []  # действия, давшие 1 из состояния 0
    first_step = {}  # первый шаг каждого действия из 0
    for a in range(N_ACTIONS):  # по всем действиям
        world.reset()  # сбросить мир
        cost += 1  # сброс считается взаимодействием
        nxt = world.step(a)  # шаг действием a
        cost += 1  # шаг считается взаимодействием
        observations[0, a] = nxt  # запомнить наблюдение пары (0, a)
        first_step[a] = nxt  # запомнить первый шаг
        if nxt == 1:  # если из 0 получилась 1 — кандидат в часы
            found.append(a)  # добавить кандидата
    if len(found) == 1:  # если кандидат ровно один:
        candidate = found[0]  # проверяемый кандидат
        world.reset()  # сбросить мир
        cost += 1  # стоимость +1
        state = 0  # текущее состояние 0
        valid = True  # орбита пока верна
        for expected in range(1, N_STATES):  # ожидаемые значения 1..31
            observed = world.step(candidate)  # шаг кандидатом
            cost += 1  # стоимость +1
            prior_value = observations.get((state, candidate))  # ранее виденное наблюдение этой пары, если было
            if prior_value is not None and prior_value != observed:  # если оно было и отличается — мир недетерминирован
                world.reset()  # сбросить мир
                cost += 1  # стоимость +1
                return (-2, observations, cost)  # вернуть −2: недетерминированные часы
            observations[state, candidate] = observed  # запомнить наблюдение
            if observed != expected:  # орбита нарушилась
                valid = False  # отметить
                break  # выход
            state = observed  # перейти к новому состоянию
        if valid:  # если орбита 1..31 пройдена:
            observed = world.step(candidate)  # 32-й шаг должен вернуть 0
            cost += 1  # стоимость +1
            prior_value = observations.get((state, candidate))  # ранее виденное наблюдение
            if prior_value is not None and prior_value != observed:  # если было и отличается — недетерминизм
                world.reset()  # сбросить мир
                cost += 1  # стоимость +1
                return (-2, observations, cost)  # вернуть −2
            observations[state, candidate] = observed  # запомнить
            valid = observed == 0  # часы верны, если орбита замкнулась в 0
        world.reset()  # сбросить мир
        cost += 1  # стоимость +1
        if valid:  # если орбита замкнулась:
            return (candidate, observations, cost)  # вернуть часы, наблюдения и стоимость
    orbit_found = []  # действия с полной орбитой +1
    for a in range(N_ACTIONS):  # по всем действиям
        world.reset()  # сбросить мир
        cost += 1  # стоимость +1
        state = 0  # состояние 0
        ok = True  # орбита пока верна
        for expected in range(1, N_STATES):  # ожидаемые значения 1..31
            observed = world.step(a)  # шаг действием a
            cost += 1  # стоимость +1
            prior_value = observations.get((state, a))  # ранее виденное наблюдение
            if prior_value is not None and prior_value != observed:  # если было и отличается — недетерминизм
                world.reset()  # сбросить мир
                cost += 1  # стоимость +1
                return (-2, observations, cost)  # вернуть −2
            observations[state, a] = observed  # запомнить
            if observed != expected:  # орбита нарушилась
                ok = False  # отметить
                break  # выход
            state = observed  # новое состояние
        if ok:  # если орбита 1..31 пройдена:
            observed = world.step(a)  # 32-й шаг
            cost += 1  # стоимость +1
            prior_value = observations.get((state, a))  # ранее виденное наблюдение
            if prior_value is not None and prior_value != observed:  # недетерминизм
                world.reset()  # сбросить мир
                cost += 1  # стоимость +1
                return (-2, observations, cost)  # вернуть −2
            observations[state, a] = observed  # запомнить
            if observed == 0:  # орбита замкнулась
                orbit_found.append(a)  # запомнить действие
    world.reset()  # сбросить мир
    cost += 1  # стоимость +1
    if len(orbit_found) == 1:  # часы найдены однозначно
        return (orbit_found[0], observations, cost)  # вернуть часы, наблюдения и стоимость
    return (None, observations, cost)  # часы не найдены: None
def navigate_query(world: TemporalWorld, clock_action: int, current_state: int, target_state: int, action: int) -> tuple[int, int, int]:
    """Зонд: довести мир часами до current → target и применить action; вернуть (наблюдение, nav, 1)."""
    nav = (target_state - current_state) % N_STATES  # число шагов часами: nav = (target − current) mod 32
    cur = current_state  # текущее состояние
    for _ in range(nav):  # nav раз
        cur = world.step(clock_action)  # шаг часами
    observed = world.step(action)  # зонд: шаг действием action
    return (observed, nav, 1)  # вернуть (наблюдение, число шагов часами, 1 зонд)
def query_pair(world: TemporalWorld, clock_action: int, current_state: int, pair: tuple[int, int]) -> tuple[int, int, int]:
    """Выполнить зонд для пары (s, a); вернуть (наблюдение, шаги часами, число зондов)."""
    s, a = pair  # разобрать пару
    observed, nav, probes = navigate_query(world, clock_action, current_state, s, a)  # выполнить зонд
    return (observed, nav, probes)  # вернуть результат
def choose_pair(candidates: list[dict], untested: list[tuple[int, int]], params_bank: list[dict], clock_action: int, current_state: int, rng: random.Random, strategy: str) -> tuple[int, int] | None:
    """Выбрать следующую пару (s, a) для запроса или None, если непроверенных нет.

        'random' — случайная пара; иначе максимум по оценке (H, −наибольшая группа, −nav, −s, −a),
        где H — энтропия разбиения кандидатов по предсказаниям, nav — шаги часами до s."""
    if not untested:  # непроверенных пар нет
        return None  # вернуть None
    if strategy == 'random':  # случайная стратегия
        return rng.choice(untested)  # случайная непроверенная пара
    best_pair = None  # лучшая пара
    best_score = None  # её оценка
    for s, a in untested:  # по непроверенным парам
        groups = {}  # группы кандидатов по предсказанию
        for c in candidates:  # по кандидатам
            y = predict_candidate(c, params_bank, s, a, clock_action)  # предсказание кандидата
            groups[y] = groups.get(y, 0) + 1  # группа +1
        if len(groups) <= 1:  # пара не различает
            continue  # пропустить
        total = len(candidates)  # число кандидатов
        entropy = -sum((v / total * math.log2(v / total) for v in groups.values()))  # энтропия H = −Σ (v/total)·log2(v/total)
        worst = max(groups.values())  # размер наибольшей группы
        nav = (s - current_state) % N_STATES  # шагов часами до состояния s
        score = (entropy, -worst, -nav, -s, -a)  # оценка: (H, −наибольшая группа, −nav, −s, −a) — больше лучше
        if best_score is None or score > best_score:  # сравнить оценки
            best_score, best_pair = (score, (s, a))  # запомнить лучшее
    return best_pair  # вернуть лучшую пару или None
def filter_candidates(candidates: list[dict], observations: dict[tuple[int, int], int], params_bank: list[dict], clock_action: int) -> list[dict]:
    """Оставить кандидатов, чьи предсказания совпадают со всеми наблюдениями."""
    return [c for c in candidates if all((predict_candidate(c, params_bank, s, a, clock_action) == y for (s, a), y in observations.items()))]  # оставить кандидатов, согласующихся со всеми наблюдениями
def active_identify(world: TemporalWorld, clock_action: int, observations: dict, current_state: int, strategy: str='active', rng=None, initial_candidates: list[dict] | None=None) -> tuple[dict | None, dict]:
    """Активное отождествление: запросами сузить кандидатов до одного.

        Возвращает (кандидат или None, статистика: состояние, шаги часами, число зондов,
        порядок запросов, число оставшихся кандидатов)."""
    rng = rng or random.Random(world.seed)  # генератор по умолчанию — из seed мира
    candidates = filter_candidates(initial_candidates or candidate_records(clock_action), observations, BANK, clock_action)  # оставить кандидатов, совместимых с исходными наблюдениями
    untested = [(s, a) for s in range(N_STATES) for a in range(N_ACTIONS) if a != clock_action and (s, a) not in observations]  # непроверенные нечасовые пары
    nav_total = probe_total = 0  # суммарные шаги часами и число зондов
    query_order = []  # порядок запросов
    while len(candidates) > 1:  # пока кандидатов больше одного
        pair = choose_pair(candidates, untested, BANK, clock_action, current_state, rng, strategy)  # выбрать пару для запроса
        if pair is None:  # различающих пар нет
            break  # стоп
        y, nav, probes = query_pair(world, clock_action, current_state, pair)  # выполнить запрос
        s, a = pair  # разобрать пару
        current_state = y  # текущее состояние — результат запроса
        nav_total += nav  # учесть шаги часами
        probe_total += probes  # учесть зонды
        query_order.append([s, a, y])  # записать запрос
        observations[pair] = y  # запомнить наблюдение
        untested.remove(pair)  # пара проверена
        candidates = filter_candidates(candidates, {pair: y}, BANK, clock_action)  # отфильтровать кандидатов по новому наблюдению
        if not candidates:  # совместимых не осталось
            break  # стоп
    chosen = candidates[0] if len(candidates) == 1 else None  # выбранный кандидат, если остался ровно один, иначе None
    return (chosen, {'current_state': current_state, 'nav_steps': nav_total, 'probe_queries': probe_total, 'query_order': query_order, 'remaining_candidates': len(candidates)})  # вернуть кандидата и статистику
def minimum_distinguishing_pairs(target_model: int, mapping: dict, params_bank: list[dict], clock_action: int, prior_observations: dict) -> list[tuple[int, int]]:
    """Точный минимальный по числу пар набор, отделяющий хранимую модель от прочих моделей банка.

        Альтернативы — биты маски; каждая пара (s, a) отбрасывает свой набор альтернатив.
        Поиск в ширину по маскам даёт кратчайший набор проверок. Пустой список — проверки
        не нужны (или наблюдения противоречат модели)."""
    alts = [mid for mid in range(len(params_bank)) if mid != target_model]  # номера всех прочих моделей банка
    if not alts:  # если прочих моделей нет
        return []  # проверки не нужны
    full = (1 << len(alts)) - 1  # маска «все альтернативы отброшены»: 2^n − 1
    covered = 0  # маска уже отброшенных альтернатив
    for (s, a), y in prior_observations.items():  # по уже сделанным наблюдениям
        if a == clock_action:  # часовые пары не различают
            continue  # пропустить
        expected = role_fn(mapping[a], s, params_bank[target_model])  # предсказание целевой модели
        if expected != y:  # наблюдение противоречит целевой модели — модели не отделить
            return []  # вернуть пустой список
        for i, mid in enumerate(alts):  # по альтернативам
            if role_fn(mapping[a], s, params_bank[mid]) != expected:  # альтернатива, дающая другое предсказание, отброшена
                covered |= 1 << i  # добавить её бит в маску
    if covered == full:  # если всё уже отброшено
        return []  # проверки не нужны
    tests = []  # возможные проверки: (маска отброшенных альтернатив, пара)
    for s in range(N_STATES):  # по состояниям
        for a, role in mapping.items():  # по нечасовым действиям
            exp = role_fn(role, s, params_bank[target_model])  # предсказание целевой модели
            mask = 0  # маска альтернатив, расходящихся в этой паре
            for i, mid in enumerate(alts):  # по альтернативам
                if role_fn(role, s, params_bank[mid]) != exp:  # расхождение
                    mask |= 1 << i  # добавить бит
            if (s, a) not in prior_observations and mask:  # берутся только непроверенные пары, отбрасывающие хоть что-то
                tests.append((mask, (s, a)))  # добавить проверку
    queue = deque([covered])  # поиск в ширину по маскам от уже покрытого
    parent = {covered: None}  # предыдущая маска
    used_test = {}  # какая проверка привела в маску
    target = None  # целевая маска пока не найдена
    while queue:  # пока очередь не пуста
        cur = queue.popleft()  # достать маску
        if cur == full:  # все альтернативы отброшены
            target = cur  # запомнить
            break  # выход
        for mask, pair in tests:  # по проверкам
            nxt = cur | mask  # новая маска: cur OR mask
            if nxt not in parent:  # если такая маска ещё не встречалась
                parent[nxt] = cur  # запомнить предка
                used_test[nxt] = pair  # запомнить проверку
                queue.append(nxt)  # в очередь
    if target is None:  # если полный набор не найден
        raise RuntimeError('no distinguishing set for model bank')  # ошибка: для банка моделей нет различающего набора
    result = []  # собранный набор проверок (в обратном порядке)
    while target != covered:  # идти от полной маски к исходной
        result.append(used_test[target])  # добавить проверку
        target = parent[target]  # перейти к предку
    result.reverse()  # перевернуть в порядок выполнения
    return result  # вернуть набор пар минимальной длины
def build_candidate_table(cand: dict, clock_action: int, params_bank: list[dict]) -> list[list[int]]:
    """Таблица переходов кандидата: часы — (s + 1) mod 32, остальные клетки по ролям."""
    table = [[0] * N_ACTIONS for _ in range(N_STATES)]  # нулевая таблица 32×4
    for s in range(N_STATES):  # по состояниям
        for a in range(N_ACTIONS):  # по действиям
            if a == clock_action:  # часы:
                table[s][a] = clock_fn(s)  # часовая клетка: table[s][a] = (s + 1) mod 32
            else:  # иначе
                table[s][a] = role_fn(cand['mapping'][a], s, params_bank[cand['model_id']])  # применить роль действия с параметрами модели кандидата
    return table  # вернуть таблицу
def full_table_learn(world: TemporalWorld, clock_action: int, observations: dict, current_state: int) -> tuple[list[list[int]], dict]:
    """Измерить полную таблицу переходов; вернуть (таблица, статистика).

        Часы известны, сделанные наблюдения перенесены, остальные пары измеряются
        по очереди — сначала ближайшие по числу шагов часами."""
    nav_total = probe_total = 0  # суммарные шаги часами и число зондов
    query_order = []  # порядок запросов
    table = [[clock_fn(s) if a == clock_action else None for a in range(N_ACTIONS)] for s in range(N_STATES)]  # таблица: часы известны, прочее None
    for (s, a), y in observations.items():  # по уже сделанным наблюдениям
        if a != clock_action:  # нечасовые
            table[s][a] = y  # заполнить клетку
    remaining = {(s, a) for s in range(N_STATES) for a in range(N_ACTIONS) if a != clock_action and table[s][a] is None}  # ещё не измеренные нечасовые пары
    while remaining:  # пока есть неизмеренные пары
        pair = min(remaining, key=lambda p: ((p[0] - current_state) % N_STATES, p[1], p[0]))  # ближайшая по числу шагов часами пара: минимум (s − current_state) mod 32, затем a, затем s
        s, a = pair  # разобрать пару
        y, nav, probes = query_pair(world, clock_action, current_state, pair)  # измерить пару
        current_state = y  # текущее состояние — результат
        nav_total += nav  # учесть шаги часами
        probe_total += probes  # учесть зонды
        query_order.append([s, a, y])  # записать запрос
        table[s][a] = y  # заполнить клетку
        observations[s, a] = y  # запомнить наблюдение
        remaining.remove(pair)  # пара измерена
    return (table, {'current_state': current_state, 'nav_steps': nav_total, 'probe_queries': probe_total, 'query_order': query_order})  # вернуть таблицу и статистику
def bfs_plan(table: list[list[int]], start: int, goal: int) -> list[int] | None:
    """Поиск в ширину: кратчайший путь от start до goal; список действий или None."""
    q = deque([start])  # очередь из start
    prev = {start: None}  # предок состояния (у start — None)
    prev_a = {}  # действие, которым пришли в состояние
    while q:  # пока очередь не пуста
        s = q.popleft()  # достать s
        if s == goal:  # цель достигнута
            break  # выход
        for a in range(N_ACTIONS):  # по действиям
            ns = table[s][a]  # следующее состояние: ns = table[s][a]
            if ns not in prev:  # ns ещё не посещено
                prev[ns] = s  # запомнить предка
                prev_a[ns] = a  # и действие
                q.append(ns)  # в очередь
    if goal not in prev:  # цель недостижима
        return None  # вернуть None
    path = []  # путь (в обратном порядке)
    s = goal  # текущее состояние — цель
    while prev[s] is not None:  # пока не дошли до start
        path.append(prev_a[s])  # добавить действие
        s = prev[s]  # перейти к предку
    return list(reversed(path))  # вернуть в прямом порядке
class TemporalGrowthAgent:
    """Агент: отождествляет модель мира, планирует и исполняет план, ведёт память."""

    def __init__(self, memory: AgentMemory | None=None, strategy: str='active'):  # создать агента: память и стратегия
        self.memory = memory or AgentMemory()  # память агента (по умолчанию новая)
        self.strategy = strategy  # стратегия выбора запросов

    def run_task(self, world: TemporalWorld, rng=None) -> dict:
        """Решить задачу в мире: найти часы, отождествить модель, спланировать и исполнить путь к цели."""
        rng = rng or random.Random(world.seed ^ 1905)  # генератор по умолчанию: seed XOR 1905
        start_steps, start_resets = (world.steps, world.resets)  # запомнить начальные счётчики мира
        clock_action, observations, clock_cost = identify_clock(world)  # найти действие часов
        if clock_action == -2:  # часы недетерминированы:
            self.memory.alerts.append({'context': world.context_id, 'kind': 'NONDETERMINISTIC_CLOCK_PROBE'})  # записать тревогу
            return {'mode': 'HOLD_NONDETERMINISTIC', 'success': False, 'exact_model': False, 'nondeterministic': True, 'probe_queries': 0, 'navigation_steps': 0, 'total_interactions': world.steps - start_steps + (world.resets - start_resets), 'total_steps': world.steps - start_steps}  # режим HOLD_NONDETERMINISTIC, успех False
        if clock_action is None:  # часы не найдены:
            return {'mode': 'HOLD_CLOCK_UNIDENTIFIED', 'success': False, 'exact_model': False, 'probe_queries': 0, 'navigation_steps': 0, 'total_steps': world.steps - start_steps}  # режим HOLD_CLOCK_UNIDENTIFIED, успех False
        current_state = 0  # после поиска часов мир сброшен: состояние 0
        trusted = world.family == TRUSTED_FAMILY and world.issuer == TRUSTED_ISSUER  # мир доверенный, если семейство и издатель совпадают с доверенными
        prior = self.memory.contexts.get(world.context_id) if trusted else None  # сохранённая модель контекста (только для доверенного мира)
        mode = 'FRESH_ACTIVE'  # режим по умолчанию — свежее активное обучение
        selected = None  # выбранный кандидат
        table = None  # таблица переходов
        nav_total = 0  # суммарные шаги часами
        probe_total = 0  # число зондов
        query_order = [[s, a, y] for (s, a), y in observations.items() if a != clock_action and s != 0]  # порядок запросов начинается с нечасовых наблюдений не из состояния 0
        transfer_checks = []  # проверки переноса модели
        if trusted and prior is not None:  # если мир доверенный и модель контекста сохранена:
            mode = 'TRANSFER_VALIDATE'  # режим — проверка переноса
            prior_map_cands = candidate_records(clock_action, [int(prior['model_id'])])  # кандидаты: та же модель, все перестановки ролей
            mapping_cand, info = active_identify(world, clock_action, observations, current_state, 'active', rng, prior_map_cands)  # активно отождествить распределение ролей
            current_state = info['current_state']  # текущее состояние
            nav_total += info['nav_steps']  # учесть шаги часами
            probe_total += info['probe_queries']  # учесть зонды
            query_order.extend(info['query_order'])  # добавить запросы
            if mapping_cand is not None:  # если распределение ролей найдено:
                mapping = mapping_cand['mapping']  # найденное распределение
                try:  # попытаться
                    checks = minimum_distinguishing_pairs(int(prior['model_id']), mapping, BANK, clock_action, observations)  # минимальный набор пар, отличающий сохранённую модель от прочих
                except RuntimeError:  # если набора нет
                    checks = []  # проверок нет
                transfer_ok = True  # перенос пока верен
                for pair in checks:  # по проверочным парам
                    y, nav, probes = query_pair(world, clock_action, current_state, pair)  # измерить пару
                    current_state = y  # текущее состояние — результат
                    nav_total += nav  # учесть шаги часами
                    probe_total += probes  # учесть зонды
                    query_order.append([pair[0], pair[1], y])  # записать запрос
                    observations[pair] = y  # запомнить наблюдение
                    expected = role_fn(mapping[pair[1]], pair[0], BANK[int(prior['model_id'])])  # ожидание: предсказание сохранённой модели
                    if y != expected:  # если не совпало — мир изменился
                        transfer_ok = False  # перенос не удался
                        break  # выход
                if transfer_ok:  # если все проверки пройдены:
                    selected = mapping_cand  # взять кандидата
                    table = build_candidate_table(selected, clock_action, BANK)  # построить таблицу
                    mode = 'TRANSFER_CONFIRMED'  # режим — перенос подтверждён
                    transfer_checks = checks  # запомнить проверки
                else:  # иначе
                    mode = 'DRIFT_DETECTED'  # режим — обнаружено изменение мира
            else:  # распределение ролей не найдено
                mode = 'DRIFT_DETECTED'  # режим — обнаружено изменение мира
            if selected is None:  # если модель не выбрана — переобучить с нуля:
                full_cands = candidate_records(clock_action)  # все кандидаты банка
                selected, info2 = active_identify(world, clock_action, observations, current_state, self.strategy, rng, full_cands)  # активное отождествление стратегией агента
                current_state = info2['current_state']  # текущее состояние
                nav_total += info2['nav_steps']  # учесть шаги часами
                probe_total += info2['probe_queries']  # учесть зонды
                query_order.extend(info2['query_order'])  # добавить запросы
                if selected is not None:  # если модель найдена:
                    table = build_candidate_table(selected, clock_action, BANK)  # построить таблицу
                    mode = 'DRIFT_RELEARNED'  # режим — изменение мира, модель переобучена
        elif trusted:  # если мир доверенный, но модели контекста нет:
            selected, info = active_identify(world, clock_action, observations, current_state, self.strategy, rng)  # активное отождествление стратегией агента
            current_state = info['current_state']  # текущее состояние
            nav_total += info['nav_steps']  # учесть шаги часами
            probe_total += info['probe_queries']  # учесть зонды
            query_order.extend(info['query_order'])  # добавить запросы
            if selected is not None:  # если модель найдена:
                table = build_candidate_table(selected, clock_action, BANK)  # построить таблицу
                mode = 'LEARNED_ACTIVE'  # режим — модель выучена активно
        else:  # мир недоверенный
            mode = 'FULL_TABLE_UNTRUSTED'  # режим — полная таблица без доверия
        if table is None:  # если таблицы нет — измерить её полностью
            table, info = full_table_learn(world, clock_action, observations, current_state)  # полное измерение таблицы
            current_state = info['current_state']  # текущее состояние
            nav_total += info['nav_steps']  # учесть шаги часами
            probe_total += info['probe_queries']  # учесть зонды
            query_order.extend(info['query_order'])  # добавить запросы
            mode = 'FULL_TABLE_FALLBACK' if trusted else 'FULL_TABLE_UNTRUSTED'  # режим — полная таблица как запасной путь (для доверенного) или без доверия
        audit_pair = next(((s, a) for s, a in observations if a != clock_action), None)  # первая нечасовая наблюдённая пара — для проверки детерминизма
        nondeterministic = False  # недетерминизм пока не обнаружен
        if audit_pair is not None:  # если есть что перепроверить:
            world.reset()  # сбросить мир
            current_state = 0  # состояние 0
            audit_s, audit_a = audit_pair  # разобрать пару
            for _ in range(audit_s):  # audit_s шагов часами до состояния audit_s
                current_state = world.step(clock_action)  # шаг часами
            y2 = world.step(audit_a)  # повторный зонд той же пары
            expected_seen = observations[audit_pair]  # прежнее наблюдение
            if y2 != expected_seen:  # если не совпало — мир недетерминирован
                nondeterministic = True  # отметить
                mode = 'HOLD_NONDETERMINISTIC'  # режим — остановка из-за недетерминизма
                self.memory.alerts.append({'context': world.context_id, 'pair': [audit_s, audit_a], 'expected': expected_seen, 'observed': y2})  # записать тревогу
        exact_model = table is not None and (not nondeterministic)  # модель точна, если таблица есть и недетерминизма нет
        path = None  # план
        success = False  # успех
        plan_len = None  # длина плана
        if exact_model:  # если модель точна:
            path = bfs_plan(table, 0, world.goal)  # кратчайший план из 0 в цель
            if path is not None:  # если цель достижима:
                world.reset()  # сбросить мир
                state = 0  # состояние 0
                for a in path:  # по действиям плана
                    state = world.step(a)  # шаг действием a
                success = state == world.goal  # успех, если пришли в цель
                plan_len = len(path)  # длина плана
        if trusted and selected is not None and (not nondeterministic):  # если мир доверенный, модель выбрана и недетерминизма нет — запомнить её в контексте
            self.memory.contexts[world.context_id] = {'model_id': selected['model_id'], 'mapping': {str(k): v for k, v in selected['mapping'].items()}, 'source_seed': world.seed, 'version': int(prior.get('version', 0) + 1) if prior else 1, 'family': TRUSTED_FAMILY, 'issuer': TRUSTED_ISSUER, 'model_digest': sha([[table[s][a] for a in range(N_ACTIONS)] for s in range(N_STATES)])}  # запись контекста: модель, роли, зерно, версия +1, семейство, издатель, хеш таблицы
        action_count = world.steps - start_steps  # число шагов за задачу
        rec = {'context_id': world.context_id, 'mode': mode, 'model_id': selected['model_id'] if selected else None, 'clock_action': clock_action, 'goal': world.goal, 'success': success, 'exact_model': exact_model, 'plan_length': plan_len, 'probe_queries': sum((1 for _, a in observations if a != clock_action)), 'unique_observed_pairs': len(observations), 'navigation_steps': nav_total, 'clock_discovery_cost': clock_cost, 'total_interactions': action_count + (world.resets - start_resets), 'resets': world.resets - start_resets, 'query_order': query_order, 'transfer_checks': [list(x) for x in transfer_checks], 'role_mapping': {str(k): v for k, v in selected['mapping'].items()} if selected else None, 'plan': path, 'nondeterministic': nondeterministic, 'trusted': trusted, 'model_digest': sha(table) if table is not None else None}  # запись результата: режим, модель, часы, цель, успех, расходы, запросы, план, хеш
        self.memory.history.append(rec)  # добавить запись в историю
        return rec  # вернуть запись
def random_action_roles(seed: int) -> tuple[str, ...]:
    """Случайные роли по зерну: перемешанные clock, xor, guard, delay."""
    p = list(('clock',) + ROLES)  # список ролей: часы и три прочие
    random.Random(seed).shuffle(p)  # перемешать по зерну
    return tuple(p)  # вернуть кортеж ролей
def make_world(seed: int, model_id: int | None, context: str, goal: int, *, family=TRUSTED_FAMILY, issuer=TRUSTED_ISSUER, out_of_family=False, noisy_repeat=False, roles=None) -> TemporalWorld:
    """Создать мир: роли (заданные или случайные), параметры модели из банка и прочие поля."""
    roles = roles or random_action_roles(seed)  # роли: заданные или случайные по зерну
    params = BANK[model_id] if model_id is not None else None  # параметры модели из банка (None, если модель не задана)
    return TemporalWorld(seed, model_id, roles, params, context, family, issuer, goal, out_of_family, noisy_repeat)  # создать мир
def _audit_world(world: TemporalWorld) -> dict:
    """Снимок истинного мира для проверки: вне семейства — таблица, иначе параметры модели."""
    if world.out_of_family:  # если мир вне семейства:
        table = [[world.true_transition(s, a) for a in range(N_ACTIONS)] for s in range(N_STATES)]  # истинная таблица переходов
        return {'seed': world.seed, 'model_id': None, 'roles': list(world.action_roles), 'goal': world.goal, 'family': world.family, 'issuer': world.issuer, 'out_of_family': True, 'table': table}  # запись аудита с таблицей (модели нет)
    return {'seed': world.seed, 'model_id': world.model_id, 'roles': list(world.action_roles), 'goal': world.goal, 'family': world.family, 'issuer': world.issuer, 'out_of_family': False, 'params': dict(world.params)}  # запись аудита с параметрами модели
def run_seed(seed: int) -> dict:
    """Полный прогон зерна: обучение, стабильный контекст, изменение мира, новый контекст,
    чужой мир, чужой издатель, шум; вернуть все задачи, аудиты и контрольную точку."""
    rng = random.Random(seed)  # генератор по зерну
    base_model = rng.randrange(len(BANK))  # базовая модель банка
    mem = AgentMemory()  # память агента
    agent = TemporalGrowthAgent(mem, 'active')  # агент с активной стратегией и памятью
    train_world = make_world(seed + 100, base_model, 'core', rng.randrange(1, N_STATES))  # обучающий мир контекста core
    train = agent.run_task(train_world, random.Random(seed + 200))  # обучающая задача
    train_audit = _audit_world(train_world)  # аудит обучающего мира
    random_train_world = make_world(seed + 100, base_model, 'core', train_world.goal, roles=train_world.action_roles)  # тот же мир для случайной стратегии
    train_random = TemporalGrowthAgent(AgentMemory(), 'random').run_task(random_train_world, random.Random(seed + 200))  # прогон случайной стратегии
    train_random_audit = _audit_world(random_train_world)  # аудит того же мира
    stable = []  # прогоны стабильного контекста
    for j in range(2):  # два повтора
        w = make_world(seed + 300 + j, base_model, 'core', rng.randrange(1, N_STATES))  # мир того же контекста
        mem_agent = agent.run_task(w, random.Random(seed + 400 + j))  # прогон с памятью
        mem_audit = _audit_world(w)  # аудит мира
        fw = make_world(seed + 300 + j, base_model, 'core', w.goal, roles=w.action_roles)  # копия мира с теми же ролями
        fresh = TemporalGrowthAgent(AgentMemory(), 'active').run_task(fw, random.Random(seed + 400 + j))  # прогон без памяти
        stable.append({'memory': mem_agent, 'fresh_active': fresh, 'audit': mem_audit})  # записать оба прогона и аудит
    drift_model = (base_model + 1 + rng.randrange(len(BANK) - 1)) % len(BANK)  # модель с изменением: base + 1 + случайное смещение mod 12
    drift_world = make_world(seed + 500, drift_model, 'core', rng.randrange(1, N_STATES))  # мир с изменившейся моделью в том же контексте
    drift = agent.run_task(drift_world, random.Random(seed + 501))  # прогон с памятью (ожидается обнаружение изменения)
    drift_audit = _audit_world(drift_world)  # аудит мира изменения
    dfw = make_world(seed + 500, drift_model, 'core', drift_world.goal, roles=drift_world.action_roles)  # копия мира изменения
    drift_fresh = TemporalGrowthAgent(AgentMemory(), 'active').run_task(dfw, random.Random(seed + 501))  # прогон без памяти
    new_world = make_world(seed + 600, (base_model + 4) % len(BANK), 'new_context', rng.randrange(1, N_STATES))  # мир с новым контекстом
    new_context = agent.run_task(new_world, random.Random(seed + 601))  # прогон с памятью (новый контекст)
    new_audit = _audit_world(new_world)  # аудит
    ood_world = make_world(seed + 700, None, 'ood', rng.randrange(1, N_STATES), family=None, issuer='UNKNOWN', out_of_family=True)  # мир вне семейства, издатель неизвестен
    ood = agent.run_task(ood_world, random.Random(seed + 701))  # прогон
    ood_audit = _audit_world(ood_world)  # аудит
    spoof_world = make_world(seed + 800, base_model, 'spoof', rng.randrange(1, N_STATES), family=TRUSTED_FAMILY, issuer='ROGUE_ISSUER')  # доверенное семейство, но чужой издатель
    spoof = agent.run_task(spoof_world, random.Random(seed + 801))  # прогон
    spoof_audit = _audit_world(spoof_world)  # аудит
    noisy_world = make_world(seed + 900, base_model, 'core', rng.randrange(1, N_STATES), noisy_repeat=True)  # мир с шумом при повторе перехода
    noisy = agent.run_task(noisy_world, random.Random(seed + 901))  # прогон
    noisy_audit = _audit_world(noisy_world)  # аудит
    cp = mem.payload()  # контрольная точка памяти
    restored = AgentMemory.restore(cp)  # восстановить память из контрольной точки
    audits = {'train': train_audit, 'stable': [x['audit'] for x in stable], 'drift': drift_audit, 'new_context': new_audit, 'ood': ood_audit, 'spoof': spoof_audit, 'noisy': noisy_audit}  # словарь аудитов
    return {'seed': seed, 'base_model': base_model, 'train': train, 'train_audit': train_audit, 'train_random': train_random, 'train_random_audit': train_random_audit, 'stable': stable, 'drift': drift, 'drift_audit': drift_audit, 'drift_fresh': drift_fresh, 'new_context': new_context, 'new_context_audit': new_audit, 'ood': ood, 'ood_audit': ood_audit, 'spoof': spoof, 'spoof_audit': spoof_audit, 'noisy': noisy, 'noisy_audit': noisy_audit, 'audits': audits, 'checkpoint': cp, 'checkpoint_roundtrip': restored.payload() == cp}  # итог прогона: все задачи, аудиты, контрольная точка и признак её оборота
def self_tests() -> list[dict]:
    """Встроенные самопроверки: банк, часы, различающий набор, отождествление, перенос, недоверие, шум, целостность."""
    out = []  # список проверок
    assert len(BANK) == 12  # в банке ровно 12 моделей
    out.append({'test': 'bank_size_and_distinctness', 'pass': True, 'models': len(BANK)})  # записать проверку
    for i, p in enumerate(BANK):  # по моделям банка
        assert all((role_fn('clock', s, p) == (s + 1) % N_STATES for s in range(N_STATES)))  # часы во всех состояниях: (s + 1) mod 32
        assert role_fn('xor', 0, p) != 1 and role_fn('guard', 0, p) != 1 and (role_fn('delay', 0, p) != 1)  # ни одна прочая роль не даёт 1 из 0 — часы находятся однозначно
    out.append({'test': 'unique_clock_discovery_signature', 'pass': True})  # записать проверку
    mapping = {1: 'xor', 2: 'guard', 3: 'delay'}  # тестовое распределение ролей
    tests = minimum_distinguishing_pairs(0, mapping, BANK, 0, {})  # минимальный набор для модели 0
    for mid in range(1, len(BANK)):  # по прочим моделям
        assert any((role_fn(mapping[a], s, BANK[mid]) != role_fn(mapping[a], s, BANK[0]) for s, a in tests))  # хоть одна пара набора должна отличать модель
    out.append({'test': 'exact_minimum_distinguishing_set', 'pass': True, 'queries': len(tests)})  # записать проверку
    cases = 0  # счётчик случаев
    for mi in range(len(BANK)):  # по моделям банка
        for perm in itertools.permutations(ROLES):  # по перестановкам ролей
            roles_list = [None] * N_ACTIONS  # список ролей
            clock_pos = (mi + cases) % N_ACTIONS  # позиция часов меняется от случая к случаю
            roles_list[clock_pos] = 'clock'  # поставить часы
            other = [a for a in range(N_ACTIONS) if a != clock_pos]  # остальные действия
            for a, role in zip(other, perm):  # раздать роли
                roles_list[a] = role  # назначить роль
            w = make_world(2000 + cases, mi, 'self', (cases * 7 + 3) % N_STATES, roles=tuple(roles_list))  # мир
            a = TemporalGrowthAgent(AgentMemory())  # агент без памяти
            r = a.run_task(w)  # решить задачу
            assert r['exact_model'] and r['success'] and (r['model_id'] == mi)  # модель точна, план исполнен, модель и роли найдены верно
            assert r['role_mapping'] == {str(a): role for a, role in enumerate(roles_list) if role != 'clock'}  # распределение ролей совпадает с истинным
            cases += 1  # случаев +1
    out.append({'test': 'all_bank_models_all_action_permutations_identifiable_and_plannable', 'pass': True, 'cases': cases})  # записать проверку
    m = AgentMemory()  # память
    ag = TemporalGrowthAgent(m)  # агент с памятью
    first = ag.run_task(make_world(3101, 0, 'drift', 17))  # первая задача контекста drift
    changed = ag.run_task(make_world(3102, 3, 'drift', 21))  # вторая задача с другой моделью
    assert first['exact_model'] and changed['mode'] == 'DRIFT_RELEARNED' and (changed['model_id'] == 3)  # изменение обнаружено и модель переобучена
    out.append({'test': 'contextual_drift_detected_and_relearned', 'pass': True})  # записать проверку
    ood = TemporalGrowthAgent(AgentMemory()).run_task(make_world(3201, None, 'ood', 7, family=None, issuer='UNKNOWN', out_of_family=True))  # мир вне семейства
    assert ood['mode'] == 'FULL_TABLE_UNTRUSTED' and ood['success'] and (ood['probe_queries'] >= 90)  # полная таблица без доверия, успех, не меньше 90 зондов
    out.append({'test': 'out_of_family_exact_full_table_fallback', 'pass': True, 'probes': ood['probe_queries']})  # записать проверку
    spoof = TemporalGrowthAgent(AgentMemory()).run_task(make_world(3202, 0, 'spoof', 9, issuer='ROGUE_ISSUER'))  # чужой издатель
    assert spoof['mode'] == 'FULL_TABLE_UNTRUSTED' and spoof['success']  # полная таблица без доверия и успех
    out.append({'test': 'rogue_issuer_forces_full_table', 'pass': True})  # записать проверку
    noisy = TemporalGrowthAgent(AgentMemory()).run_task(make_world(3203, 0, 'noise', 9, noisy_repeat=True))  # шумный мир
    assert noisy['mode'] == 'HOLD_NONDETERMINISTIC' and (not noisy['success'])  # остановка из-за недетерминизма, успеха нет
    out.append({'test': 'repeated_observation_inconsistency_holds', 'pass': True})  # записать проверку
    mm = AgentMemory()  # память
    mm.contexts['x'] = {'model_id': 0}  # запись контекста
    cp = mm.payload()  # контрольная точка
    bad = json.loads(json.dumps(cp))  # испорченная копия
    bad['contexts']['x']['model_id'] = 1  # подменить номер модели
    rejected = False  # отклонение пока не произошло
    try:  # попытаться
        AgentMemory.restore(bad)  # восстановить испорченную запись
    except ValueError:  # ошибка целостности
        rejected = True  # отклонено
    assert rejected  # испорченная запись обязана быть отклонена
    out.append({'test': 'checkpoint_tamper_rejected', 'pass': True})  # записать проверку
    return out  # вернуть список проверок
def run_panel(start_seed: int, n_seeds: int, outdir: Path) -> dict:
    """Панель прогонов: n_seeds зёрен, контрольные точки в файлах, сводка со статистикой снижений."""
    outdir.mkdir(parents=True, exist_ok=True)  # создать каталог результатов
    runs = []  # прогоны
    for s in range(start_seed, start_seed + n_seeds):  # по зернам от start_seed до start_seed + n_seeds − 1
        r = run_seed(s)  # прогон зерна
        cpfile = outdir / f'checkpoint_{s}.json'  # файл контрольной точки
        cpfile.write_text(json.dumps(r['checkpoint'], indent=2, sort_keys=True))  # сохранить контрольную точку
        r.pop('checkpoint')  # убрать контрольную точку из итога
        runs.append(r)  # добавить прогон
    stable_mem = [x['memory'] for r in runs for x in r['stable']]  # прогоны стабильного контекста с памятью
    stable_fresh = [x['fresh_active'] for r in runs for x in r['stable']]  # те же прогоны без памяти

    def descriptive(values):
        """Описательная статистика: среднее, отклонение, 95%-й интервал (1.96·sd/√n), минимум, максимум."""
        vals = list(values)  # список значений
        mean = statistics.mean(vals) if vals else 0.0  # среднее (0.0 для пустого)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0  # стандартное отклонение (0.0, если меньше двух значений)
        half = 1.96 * sd / math.sqrt(len(vals)) if vals else 0.0  # половина 95%-го интервала: 1.96·sd/√n
        return {'mean_pct': round(mean, 3), 'sd_pct': round(sd, 3), 'ci95_low': round(mean - half, 3), 'ci95_high': round(mean + half, 3), 'min_pct': round(min(vals), 3) if vals else 0.0, 'max_pct': round(max(vals), 3) if vals else 0.0}  # описательная статистика: среднее, отклонение, интервал, минимум, максимум
    paired_train_probe = [100 * (1 - r['train']['probe_queries'] / max(1, r['train_random']['probe_queries'])) for r in runs]  # снижение зондов активной стратегии против случайной, %: 100·(1 − q_a/max(1, q_r))
    paired_train_inter = [100 * (1 - r['train']['total_interactions'] / max(1, r['train_random']['total_interactions'])) for r in runs]  # то же по полным взаимодействиям на обучении
    paired_stable_probe = [100 * (1 - sum((x['memory']['probe_queries'] for x in r['stable'])) / max(1, sum((x['fresh_active']['probe_queries'] for x in r['stable'])))) for r in runs]  # то же по зондам в стабильном контексте: память против свежего агента
    paired_stable_inter = [100 * (1 - sum((x['memory']['total_interactions'] for x in r['stable'])) / max(1, sum((x['fresh_active']['total_interactions'] for x in r['stable'])))) for r in runs]  # то же по взаимодействиям в стабильном контексте
    paired_drift_inter = [100 * (1 - r['drift']['total_interactions'] / max(1, r['drift_fresh']['total_interactions'])) for r in runs]  # то же по взаимодействиям при изменении мира
    summary = {'version': '4.4', 'seeds': n_seeds, 'paired_seed_descriptive_reductions': {'active_vs_random_probes': descriptive(paired_train_probe), 'active_vs_random_total_interactions': descriptive(paired_train_inter), 'memory_vs_fresh_stable_probes': descriptive(paired_stable_probe), 'memory_vs_fresh_stable_total_interactions': descriptive(paired_stable_inter), 'memory_vs_fresh_drift_total_interactions': descriptive(paired_drift_inter)}, 'train_exact': sum((bool(r['train']['exact_model']) for r in runs)), 'train_random_exact': sum((bool(r['train_random']['exact_model']) for r in runs)), 'train_active_probes': sum((r['train']['probe_queries'] for r in runs)), 'train_random_probes': sum((r['train_random']['probe_queries'] for r in runs)), 'train_active_vs_random_probe_reduction_pct': round((1 - sum((r['train']['probe_queries'] for r in runs)) / max(1, sum((r['train_random']['probe_queries'] for r in runs)))) * 100, 3), 'train_active_interactions': sum((r['train']['total_interactions'] for r in runs)), 'train_random_interactions': sum((r['train_random']['total_interactions'] for r in runs)), 'stable_memory_success': sum((bool(x['success']) for x in stable_mem)), 'stable_fresh_success': sum((bool(x['success']) for x in stable_fresh)), 'stable_memory_probes': sum((x['probe_queries'] for x in stable_mem)), 'stable_fresh_probes': sum((x['probe_queries'] for x in stable_fresh)), 'drift_relearned': sum((r['drift']['mode'] == 'DRIFT_RELEARNED' for r in runs)), 'drift_success': sum((bool(r['drift']['success']) for r in runs)), 'drift_fresh_success': sum((bool(r['drift_fresh']['success']) for r in runs)), 'new_context_success': sum((bool(r['new_context']['success']) for r in runs)), 'ood_full_table': sum((r['ood']['mode'] == 'FULL_TABLE_UNTRUSTED' for r in runs)), 'ood_success': sum((bool(r['ood']['success']) for r in runs)), 'spoof_full_table': sum((r['spoof']['mode'] == 'FULL_TABLE_UNTRUSTED' for r in runs)), 'spoof_success': sum((bool(r['spoof']['success']) for r in runs)), 'noisy_holds': sum((r['noisy']['mode'] == 'HOLD_NONDETERMINISTIC' for r in runs)), 'all_checkpoint_roundtrip': all((r['checkpoint_roundtrip'] for r in runs)), 'mean_train_probes': round(statistics.mean((r['train']['probe_queries'] for r in runs)), 3), 'mean_train_random_probes': round(statistics.mean((r['train_random']['probe_queries'] for r in runs)), 3), 'mean_transfer_probes': round(statistics.mean((x['memory']['probe_queries'] for r in runs for x in r['stable'])), 3), 'mean_fresh_probes': round(statistics.mean((x['fresh_active']['probe_queries'] for r in runs for x in r['stable'])), 3), 'memory_probe_reduction_pct': round((1 - sum((x['probe_queries'] for x in stable_mem)) / max(1, sum((x['probe_queries'] for x in stable_fresh)))) * 100, 3), 'mean_drift_probes': round(statistics.mean((r['drift']['probe_queries'] for r in runs)), 3), 'mean_ood_probes': round(statistics.mean((r['ood']['probe_queries'] for r in runs)), 3), 'mean_ood_interactions': round(statistics.mean((r['ood']['total_interactions'] for r in runs)), 3), 'stable_memory_interactions': sum((x['memory']['total_interactions'] for r in runs for x in r['stable'])), 'stable_fresh_interactions': sum((x['fresh_active']['total_interactions'] for r in runs for x in r['stable'])), 'stable_interaction_reduction_pct': round((1 - sum((x['memory']['total_interactions'] for r in runs for x in r['stable'])) / max(1, sum((x['fresh_active']['total_interactions'] for r in runs for x in r['stable'])))) * 100, 3), 'drift_memory_interactions': sum((r['drift']['total_interactions'] for r in runs)), 'drift_fresh_interactions': sum((r['drift_fresh']['total_interactions'] for r in runs)), 'mean_train_plan_length': round(statistics.mean((r['train']['plan_length'] or 0 for r in runs)), 3)}  # сводка: описательная статистика снижений, числа успехов, суммарные зонды и взаимодействия по всем задачам
    (outdir / 'panel_runs.json').write_text(json.dumps(runs, indent=2, sort_keys=True))  # сохранить прогоны
    (outdir / 'panel_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))  # сохранить сводку
    return summary  # вернуть сводку
def main():
    """Точка входа командной строки: --self-test или панель прогонов."""
    ap = argparse.ArgumentParser()  # парсер аргументов
    ap.add_argument('--self-test', action='store_true')  # флаг самопроверки
    ap.add_argument('--seed', type=int, default=20261119)  # стартовое зерно
    ap.add_argument('--panel', type=int, default=20)  # число зёрен панели
    ap.add_argument('--out', type=Path, default=Path('results/panel20'))  # каталог результатов
    args = ap.parse_args()  # разобрать аргументы
    if args.self_test:  # если запрошена самопроверка
        print(json.dumps(self_tests(), indent=2))  # напечатать проверки
        return  # выход
    print(json.dumps(run_panel(args.seed, args.panel, args.out), indent=2))  # напечатать сводку панели
if __name__ == '__main__':  # файл запущен напрямую — выполнить main()
    main()  # запуск основной функции

# ============================================================
# СВЯЗКА ENV TEMPORAL: мир -> нетлист
# ============================================================

from hagi.data import env_temporal as env  # импорт этого же модуля как env
def env_model_to_netlist(model_id: int, clock_action: int | None=None, seed: int=4242, context: str='core', goal: int=5) -> dict:
    """Активно обучиться миру и скомпилировать его таблицу в сеть вентилей."""
    bank = env.generate_bank()  # банк моделей
    w = env.make_world(seed, model_id, context, goal)  # мир
    agent = env.TemporalGrowthAgent()  # агент
    res = agent.run_task(w, random.Random(1))  # решить задачу
    cand = env.candidate_records(res['clock_action'])[res['model_id']]  # кандидат по найденным модели и часам
    table = env.build_candidate_table(cand, res['clock_action'], bank)  # таблица кандидата
    from hagi.data.env_lut import compile_transition_table  # импорт компилятора таблиц в сеть
    compiled = compile_transition_table(table, n_bits=5, n_act_bits=2)  # скомпилировать (5 бит состояния, 2 бита действия)
    net = compiled['net']  # сеть вентилей
    return {'net': net, 'table': table, 'model_id': res['model_id'], 'learn_mode': res['mode'], 'anf': compiled['anf'], 'gates': compiled['gates'], 'depth': compiled['depth'], 'behavioral_exact': compiled['behavioral_exact']}  # вернуть сеть, таблицу, модель, режим обучения, АНФ, число вентилей, глубину, признак точности

# ============================================================
# ПРОВЕРКА ENV TEMPORAL: независимое воспроизведение
# ============================================================

N = 32  # число состояний N = 32
A = 4  # число действий A = 4
MODEL_BANK = []  # банк моделей для верификатора (заполняется извне)
digest = sha  # синоним digest для sha
def next_state(role, s, p):
    """Независимая реализация перехода роли (clock, xor, guard, delay)."""
    if role == 'clock':  # роль clock:
        return (s + 1) % N  # часы: (s + 1) mod 32
    if role == 'xor':  # роль xor:
        return s ^ int(p['xor_mask'])  # побитовое исключающее или: s XOR xor_mask
    if role == 'guard':  # роль guard:
        return s ^ int(p['effect_mask']) if s & int(p['guard_mask']) == int(p['guard_value']) else s  # s XOR effect_mask, если s & guard_mask == guard_value, иначе s
    if role == 'delay':  # роль delay:
        return s & 15 ^ int(p['delay_mask']) if s >> 4 & 1 else s & 15 | 16  # если бит 4 установлен: (s & 15) XOR delay_mask; иначе (s & 15) | 16
    raise ValueError(role)  # неизвестная роль — ошибка
def make_table(audit):
    """Истинная таблица переходов по снимку мира: из таблицы (вне семейства) или по ролям и параметрам."""
    if audit['out_of_family']:  # если мир вне семейства:
        return [[int(v) for v in row] for row in audit['table']]  # таблица целиком из аудита
    roles = audit['roles']  # роли из аудита
    p = audit['params']  # параметры модели из аудита
    return [[next_state(roles[a], s, p) for a in range(A)] for s in range(N)]  # истинная таблица: next_state(roles[a], s, p)
def shortest_len(table, start, goal):
    """Длина кратчайшего пути от start до goal поиском в ширину; None, если цель недостижима."""
    q = deque([(start, 0)])  # очередь пар (состояние, расстояние) из start
    seen = {start}  # посещённые состояния
    while q:  # пока очередь не пуста
        s, d = q.popleft()  # достать состояние и расстояние
        if s == goal:  # цель достигнута
            return d  # вернуть расстояние
        for a in range(A):  # по действиям
            t = table[s][a]  # следующее состояние: t = table[s][a]
            if t not in seen:  # t ещё не посещено
                seen.add(t)  # пометить
                q.append((t, d + 1))  # в очередь с расстоянием d + 1
    return None  # цель недостижима
def verify_record(rec, audit, label):
    """Независимая проверка записи о задаче: хеш модели, покрытие, роли, проверки переноса, план."""
    table = make_table(audit)  # истинная таблица по аудиту
    if rec.get('mode') == 'HOLD_NONDETERMINISTIC' and rec.get('model_digest') is None:  # если запись — остановка из-за недетерминизма без модели:
        if not rec.get('nondeterministic', True):  # флаг недетерминизма должен стоять
            raise AssertionError(f'{label}: nondeterministic hold flag missing')  # ошибка: флаг остановки отсутствует
        return {'label': label, 'verified_model_table': False, 'plan_verified': False, 'status': 'HOLD'}  # результат: остановка, таблица и план не проверены
    if rec.get('model_digest') != digest(table):  # хеш модели должен равняться хешу истинной таблицы
        raise AssertionError(f'{label}: model digest mismatch')  # ошибка: хеш модели не совпал
    if rec.get('mode') in ('FULL_TABLE_UNTRUSTED', 'FULL_TABLE_FALLBACK'):  # если таблица строилась полностью:
        if int(rec.get('unique_observed_pairs', 0)) != N * A:  # должно быть зарегистрировано N·A = 128 пар
            raise AssertionError(f'{label}: full-table fallback did not log all {N * A} pairs')  # ошибка: записаны не все пары
        clock_for_coverage = rec.get('clock_action')  # действие часов из записи
        covered = {(0, a) for a in range(A)}  # покрытые пары: все действия из 0
        if clock_for_coverage is not None:  # если часы известны
            covered.update(((s, int(clock_for_coverage)) for s in range(N)))  # добавить все часовые пары (s, clock)
        covered.update(((int(s), int(a)) for s, a, y in rec.get('query_order', [])))  # добавить пары из порядка запросов
        if len(covered) != N * A:  # покрытие должно быть полным
            raise AssertionError(f'{label}: full-table evidence coverage is {len(covered)}/{N * A}')  # ошибка: неполное покрытие свидетельствами
    if rec.get('clock_action') is not None:  # если часы указаны
        clock = rec['clock_action']  # действие часов
        if audit['roles'][clock] != 'clock':  # в аудите на этом месте должны быть часы
            raise AssertionError(f'{label}: wrong clock action')  # ошибка: неверное действие часов
    mapping = rec.get('role_mapping')  # распределение ролей из записи
    if mapping is not None:  # если оно есть:
        norm = {int(k): v for k, v in mapping.items()}  # нормализовать ключи в целые
        expected = {a: r for a, r in enumerate(audit['roles']) if r != 'clock'}  # ожидание: нечасовые действия аудита
        if norm != expected:  # должно совпасть
            raise AssertionError(f'{label}: role mapping mismatch')  # ошибка: распределение ролей не совпало
        if rec.get('model_id') != audit.get('model_id'):  # номер модели должен совпасть
            raise AssertionError(f'{label}: model identity mismatch')  # ошибка: номер модели не совпал
        if rec.get('mode') == 'TRANSFER_CONFIRMED':  # если перенос подтверждён:
            checks = {(int(s), int(a)) for s, a in rec.get('transfer_checks', [])}  # множество проверочных пар
            params = audit['params']  # параметры модели
            target = int(audit['model_id'])  # номер целевой модели
            for alt in range(12):  # по 12 моделям банка
                if alt == target:  # целевую пропустить
                    continue  # пропустить
                separated = False  # альтернатива ещё не отделена
                for s, a in checks:  # по проверочным парам
                    role = norm[a]  # роль действия
                    if next_state(role, s, params) != next_state(role, s, MODEL_BANK[alt]):  # если альтернатива даёт другое предсказание — она отделена
                        separated = True  # отделена
                        break  # выход
                if not separated:  # каждая альтернатива должна быть отделена
                    raise AssertionError(f'{label}: transfer checks do not distinguish model {alt}')  # ошибка: проверки не отличают модель alt
    for s, a, y in rec.get('query_order', []):  # по зарегистрированным запросам
        if table[int(s)][int(a)] != int(y):  # наблюдение должно совпадать с истинной таблицей
            raise AssertionError(f'{label}: query evidence mismatch at {(s, a)}')  # ошибка: свидетельство запроса не совпало
    if rec.get('nondeterministic'):  # если запись помечена недетерминизмом:
        if rec.get('mode') != 'HOLD_NONDETERMINISTIC' or rec.get('plan') is not None:  # режим должен быть остановкой и плана быть не должно
            raise AssertionError(f'{label}: nondeterministic oracle was not held')  # ошибка: недетерминированный мир не был остановлен
        return {'label': label, 'verified_model_table': True, 'plan_verified': False, 'status': 'HOLD'}  # результат: таблица проверена, остановка
    plan = rec.get('plan')  # план из записи
    if plan is None:  # если плана нет
        if rec.get('success'):  # успешная запись обязана иметь план
            raise AssertionError(f'{label}: successful record has no plan')  # ошибка: у успешной записи нет плана
        return {'label': label, 'verified_model_table': True, 'plan_verified': False, 'status': rec.get('mode')}  # результат: таблица проверена, план не проверен
    state = 0  # начальное состояние state = 0
    for a in plan:  # по действиям плана
        state = table[state][int(a)]  # новое состояние: state = table[state][a]
    if state != int(audit['goal']) or not rec.get('success'):  # план должен привести в цель, и запись должна быть успешной
        raise AssertionError(f'{label}: plan does not reach goal')  # ошибка: план не ведёт к цели
    opt = shortest_len(table, 0, int(audit['goal']))  # длина кратчайшего пути
    if len(plan) != opt:  # план должен быть кратчайшим
        raise AssertionError(f'{label}: plan not shortest ({len(plan)} vs {opt})')  # ошибка: план не кратчайший
    if rec.get('plan_length') != len(plan):  # заявленная длина должна совпадать
        raise AssertionError(f'{label}: plan length metadata mismatch')  # ошибка: заявленная длина плана не совпала
    return {'label': label, 'verified_model_table': True, 'plan_verified': True, 'shortest': True, 'status': rec.get('mode')}  # результат: таблица и кратчайший план проверены
def verify_checkpoint(cp, label):
    """Независимая проверка контрольной точки: хеш, версия, диапазон моделей, роли, происхождение."""
    body = dict(cp)  # копия контрольной точки
    got = body.pop('digest', None)  # изъять хеш
    if got != digest(body):  # хеш должен совпасть
        raise AssertionError(f'{label}: checkpoint digest')  # ошибка: хеш контрольной точки не совпал
    if body.get('version') != '4.4':  # версия должна быть 4.4
        raise AssertionError(f'{label}: version')  # ошибка: неверная версия
    latest = {}  # последняя успешная запись по контексту
    for rec in body.get('history', []):  # по истории
        if rec.get('context_id') and rec.get('model_id') is not None and (not str(rec.get('mode', '')).startswith('HOLD')):  # запись с контекстом, моделью и без режима остановки:
            latest[rec['context_id']] = rec  # считается последней для контекста
    for ctx, model in body.get('contexts', {}).items():  # по сохранённым контекстам
        if int(model['model_id']) >= 12 or int(model['model_id']) < 0:  # номер модели должен быть в 0..11
            raise AssertionError(f'{label}: model id bounds')  # ошибка: номер модели вне диапазона
        mapping = {int(k): v for k, v in model['mapping'].items()}  # распределение ролей
        if len(mapping) != 3 or set(mapping.values()) != {'xor', 'guard', 'delay'}:  # ровно 3 роли: xor, guard, delay
            raise AssertionError(f'{label}: invalid role mapping')  # ошибка: неверное распределение ролей
        clock = next((a for a in range(4) if a not in mapping))  # действие часов — единственное не входящее в распределение
        if not MODEL_BANK or int(model['model_id']) >= len(MODEL_BANK):  # банк моделей должен быть доступен и содержать модель
            raise AssertionError(f'{label}: model bank unavailable')  # ошибка: банк моделей недоступен
        p = MODEL_BANK[int(model['model_id'])]  # параметры модели
        table = [[next_state('clock', s, p) if a == clock else next_state(mapping[a], s, p) for a in range(4)] for s in range(N)]  # таблица: часы на месте clock, прочие роли по распределению
        if digest(table) != model.get('model_digest'):  # хеш таблицы должен совпасть с сохранённым
            raise AssertionError(f'{label}: checkpoint model digest mismatch')  # ошибка: хеш модели контрольной точки не совпал
        if ctx not in latest:  # для контекста должна быть запись происхождения
            raise AssertionError(f'{label}: missing lineage record for {ctx}')  # ошибка: нет записи происхождения для контекста
        rec = latest[ctx]  # запись происхождения
        if int(rec['model_id']) != int(model['model_id']):  # номер модели должен совпадать
            raise AssertionError(f'{label}: checkpoint/history model lineage mismatch for {ctx}')  # ошибка: происхождение модели не совпало
        if rec.get('model_digest') != model.get('model_digest'):  # хеш модели должен совпадать
            raise AssertionError(f'{label}: checkpoint/history digest lineage mismatch for {ctx}')  # ошибка: происхождение хеша не совпало
    return {'label': label, 'digest': True, 'contexts': len(body.get('contexts', {})), 'history': len(body.get('history', []))}  # результат: хеш верен, число контекстов и записей истории
def main():
    """Точка входа командной строки верификатора."""
    ap = argparse.ArgumentParser()  # парсер аргументов верификатора
    ap.add_argument('--panel', type=Path, default=Path('results/panel20/panel_runs.json'))  # аргумент --panel: файл запусков панели
    ap.add_argument('--summary', type=Path, default=Path('results/panel20/panel_summary.json'))  # файл сводки панели
    ap.add_argument('--out', type=Path, default=Path('results/independent_verification.json'))  # файл результата проверки
    ap.add_argument('--checkpoint-dir', type=Path, default=Path('results/panel20'))  # каталог контрольных точек
    ap.add_argument('--bank', type=Path, default=Path('MODEL_BANK.json'))  # файл банка моделей
    a = ap.parse_args()  # разобрать аргументы
    runs = json.loads(a.panel.read_text())  # загрузить прогоны панели
    summary = json.loads(a.summary.read_text())  # загрузить сводку
    global MODEL_BANK  # изменять глобальный банк моделей
    MODEL_BANK = json.loads(a.bank.read_text())  # загрузить банк моделей
    checked = []  # результаты проверок записей
    for r in runs:  # по прогонам
        seed = r['seed']  # зерно прогона
        checked.append(verify_record(r['train'], r['train_audit'], f'{seed}/train'))  # проверить задачу обучения
        checked.append(verify_record(r['train_random'], r['train_random_audit'], f'{seed}/train_random'))  # проверить случайную стратегию
        for i, x in enumerate(r['stable']):  # по повторам стабильного контекста
            checked.append(verify_record(x['memory'], x['audit'], f'{seed}/stable{i}/memory'))  # проверить прогон с памятью
            checked.append(verify_record(x['fresh_active'], x['audit'], f'{seed}/stable{i}/fresh'))  # проверить прогон без памяти
        checked.append(verify_record(r['drift'], r['drift_audit'], f'{seed}/drift'))  # проверить задачу с изменением мира
        checked.append(verify_record(r['drift_fresh'], r['drift_audit'], f'{seed}/drift_fresh'))  # проверить её прогон без памяти
        checked.append(verify_record(r['new_context'], r['new_context_audit'], f'{seed}/new_context'))  # проверить новый контекст
        checked.append(verify_record(r['ood'], r['ood_audit'], f'{seed}/ood'))  # проверить мир вне семейства
        checked.append(verify_record(r['spoof'], r['spoof_audit'], f'{seed}/spoof'))  # проверить чужого издателя
        checked.append(verify_record(r['noisy'], r['noisy_audit'], f'{seed}/noisy'))  # проверить шумный мир
        cpfile = a.checkpoint_dir / f'checkpoint_{seed}.json'  # файл контрольной точки зерна
        cp = json.loads(cpfile.read_text())  # загрузить контрольную точку
        checked.append(verify_checkpoint(cp, f'{seed}/checkpoint'))  # проверить контрольную точку
    tamper_tests = []  # проверки на подделку
    if runs:  # если прогоны есть:
        cp = json.loads((a.checkpoint_dir / f"checkpoint_{runs[0]['seed']}.json").read_text())  # контрольная точка первого прогона
        bad = json.loads(json.dumps(cp))  # её копия
        ctx = next(iter(bad['contexts']))  # первый контекст
        bad['contexts'][ctx]['model_id'] = (int(bad['contexts'][ctx]['model_id']) + 1) % 12  # подменить номер модели (циклично +1 mod 12)
        body = dict(bad)  # тело без хеша
        body.pop('digest', None)  # убрать прежний хеш
        bad['digest'] = digest(body)  # пересчитать внешний хеш — подделка должна быть найдена по происхождению
        rejected = False  # отклонения пока нет
        try:  # попытаться
            verify_checkpoint(bad, 'tamper/model-id')  # проверить подделанную точку
        except AssertionError:  # отклонена
            rejected = True  # отклонена
        if not rejected:  # подделка обязана быть отклонена
            raise AssertionError('tampered checkpoint model-id not rejected')  # ошибка: подделанный номер модели не отклонён
        tamper_tests.append({'test': 'checkpoint_model_id_with_rehashed_outer_digest', 'rejected': True})  # записать проверку
    if summary['train_exact'] != len(runs) or summary['train_random_exact'] != len(runs):  # все прогоны должны быть точны при активной и случайной стратегии
        raise AssertionError('summary train count mismatch')  # ошибка: число точных обучений в сводке не совпало
    if summary['stable_memory_success'] != 2 * len(runs):  # в стабильном контексте по 2 успеха на прогон
        raise AssertionError('summary stable successes mismatch')  # ошибка: число успехов в сводке не совпало
    if summary['drift_relearned'] != len(runs):  # изменение мира переобучено в каждом прогоне
        raise AssertionError('summary drift count mismatch')  # ошибка: число переобучений в сводке не совпало
    if summary['ood_full_table'] != len(runs):  # чужой мир — полная таблица в каждом прогоне
        raise AssertionError('summary OOD fallback mismatch')  # ошибка: число полных таблиц в сводке не совпало
    if summary['spoof_full_table'] != len(runs):  # чужой издатель — полная таблица в каждом прогоне
        raise AssertionError('summary spoof fallback mismatch')  # ошибка: число полных таблиц для чужого издателя не совпало
    if summary['noisy_holds'] != len(runs):  # шумный мир — остановка в каждом прогоне
        raise AssertionError('summary noisy hold mismatch')  # ошибка: число остановок в сводке не совпало
    result = {'status': 'PASS', 'runs': len(runs), 'verified_records': len(checked), 'verified_plans': sum((x.get('plan_verified', False) for x in checked)), 'verified_holds': sum((x.get('status') == 'HOLD' for x in checked)), 'checkpoint_audits': sum(('checkpoint' in x['label'] for x in checked)), 'tamper_tests': tamper_tests, 'summary_checks': 'PASS', 'records': checked}  # итог: статус, число прогонов, проверенных записей, планов, остановок, контрольных точек
    a.out.parent.mkdir(parents=True, exist_ok=True)  # создать каталог результата
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True))  # записать результат
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))  # напечатать итог без списка записей
if __name__ == '__main__':  # файл запущен напрямую — выполнить main()
    main()  # запуск основной функции