#!/usr/bin/env python3
"""Ткань Венна: самособирающийся субстрат процессора из логических вентилей.

    Геном — словарь {W, n_inputs, gates, outs}:
      W — число битовых дорожек в шине (все ворота обрабатывают все дорожки одной маской —
      общие веса); n_inputs — число входов; gates — список вентилей {a, b, mask}, где a и b —
      номера проводов на входах, mask — 4-битная маска Венна; outs — номера проводов-выходов.
      Провод с номером g < n_inputs — вход, с номером n_inputs + i — выход i-го вентиля.

    Вентиль: для каждой дорожки bit из W по битам a_bit, b_bit входов выбирается бит маски:
      out_bit = (mask >> ((a_bit << 1) | b_bit)) & 1.
    Маска — это и есть программа: бит 3 задаёт значение при (1, 1), бит 2 — при (1, 0),
    бит 1 — при (0, 1), бит 0 — при (0, 0). 16 маск покрывают все булевы функции двух переменных.

    Цена (цель оптимизации): cost = program_bits + 4·depth, где program_bits — биты описания
    генома, depth — число вентилей на самом длинном пути от входа к выходу.

    Метод — «сначала сертификация». Фаза РОСТА: ищется любой геном, верный на всей таблице
    истинности (сигнал поиска — доля верных строк). Фаза ОПТИМИЗАЦИИ: верность — жёсткое
    условие, цена только убывает; давление цены до верности сваливает поиск в оптимум
    из одного вентиля.

    Экспортирует: MASKS, ALL_MASKS, NAME_OF, random_genome, program_bits, emulate, certify,
    correct_fraction, cost, mutate, crossover, evolve, render, full_adder_profile."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import json  # чтение и запись журнала
import random  # генератор случайных чисел
import time  # метка времени результата
from pathlib import Path  # работа с путями файлов
from typing import Any  # аннотации типов

ROOT = Path(__file__).resolve().parent.parent  # корень проекта
from hagi.util import CKPT_DIR as _CKPT  # каталог чекпойнтов
FABRIC_LOG = _CKPT / "self" / "fabric.json"  # журнал результатов эволюции

# --- 16 кодов Венна как 4-битные маски; порядок комбинаций (++),(+-),(-+),(--) в битах
# 3,2,1,0 (бит 3 = значение при (И, И), бит 0 = при (Л, Л)) ---
MASKS = {  # имя булевой функции -> маска
    "FALSE": 0b0000, "AND": 0b1000, "A_GT_B": 0b0100, "A": 0b1010,  # константа 0; a ∧ b; a ∧ ¬b; a
    "B_GT_A": 0b0010, "B": 0b1100, "XOR": 0b0110, "OR": 0b1110,  # ¬a ∧ b; b; a ⊕ b; a ∨ b
    "NOR": 0b0001, "XNOR": 0b1001, "NOT_B": 0b0011, "B_LE_A": 0b1011,  # ¬(a ∨ b); ¬(a ⊕ b); ¬b; a ∨ ¬b
    "NOT_A": 0b0101, "A_LE_B": 0b1101, "NAND": 0b0111, "TRUE": 0b1111,  # ¬a; ¬a ∨ b; ¬(a ∧ b); константа 1
}
ALL_MASKS = dict(MASKS)  # рабочая копия таблицы масок
NAME_OF = {v: k for k, v in MASKS.items()}  # обратная таблица: маска -> имя










# --- Геном ткани ---

def random_genome(rng: random.Random, W: int = 8, n_inputs: int = 2,
                  n_outputs: int = 1) -> dict:  # n_outputs — число выходов; возвращает геном
    """Случайный геном: от 4 до 10 вентилей и случайные выходы.

        rng — генератор случайных чисел; W — число битовых дорожек; n_inputs — число входов;
        n_outputs — число выходов. Входы i-го вентиля выбираются равномерно из проводов
        0..n_inputs + i − 1 (только уже существующие провода)."""
    gates = []  # накопитель вентилей
    n_gates = rng.randint(4, 10)  # число вентилей — целое равномерно из [4, 10]
    for i in range(n_gates):  # по вентилям
        srcs = n_inputs + i  # число доступных проводов: входы плюс уже созданные вентиля
        gates.append({"a": rng.randrange(srcs), "b": rng.randrange(srcs),  # вентиль с входами a, b — равномерно из 0..srcs−1
                      "mask": rng.choice(list(ALL_MASKS.values()))})  # маска — случайная из 16 возможных
    outs = [rng.randrange(n_inputs + n_gates) for _ in range(n_outputs)]  # выходы — случайные провода из всех проводов
    return {"W": W, "n_inputs": n_inputs, "gates": gates, "outs": outs}  # готовый геном


def program_bits(genome: dict) -> int:
    """Длина описания генома в битах.

        На вентиль: 4 бита маски + 2·ceil(log2(max(1, n_inputs + n_gates − 1))) бита на два входа;
        на выход: ceil(log2(max(1, n_inputs + n_gates))) бита."""
    n_in = genome["n_inputs"]  # число входов
    bits_per_gate = 4 + 2 * max(1, n_in + len(genome["gates"]) - 1).bit_length()  # битов на вентиль: 4 + 2·bit_length(max(1, n_in + n_gates − 1))
    # биты вентилей (n_gates · bits_per_gate) плюс биты выходов (n_outs · bit_length(max(1, n_in + n_gates)))
    return len(genome["gates"]) * bits_per_gate + \
        len(genome["outs"]) * max(1, n_in + len(genome["gates"])).bit_length()  # вторая часть суммы: биты выходов


def emulate(genome: dict, inputs: list[int]) -> list[int]:
    """Побитовое исполнение: входы — W-битные шины, каждый вентиль применяет свою маску Венна
    к каждой дорожке отдельно (маска общая для всех дорожек).

        inputs — список целых чисел (провода-входы). Возвращает список проводов-выходов."""
    W = genome["W"]  # число дорожек
    wires = list(inputs[:genome["n_inputs"]])  # провода: первые n_inputs входов
    for g in genome["gates"]:  # по вентилям по порядку
        a = wires[g["a"]] if g["a"] < len(wires) else 0  # вход a; если провод ещё не существует — 0
        b = wires[g["b"]] if g["b"] < len(wires) else 0  # вход b; если провод ещё не существует — 0
        m = g["mask"]  # маска вентиля
        res = 0  # результат вентиля
        for bit in range(W):  # по дорожкам bit = 0..W−1
            ab = (a >> bit) & 1  # бит a на дорожке bit
            bb = (b >> bit) & 1  # бит b на дорожке bit
            idx = (ab << 1) | bb          # номер бита маски: 2·a_bit + b_bit (бит3 = (И, И) ... бит0 = (Л, Л))
            res |= ((m >> idx) & 1) << bit  # out_bit = (m >> idx) & 1; поставить его на дорожку bit
        wires.append(res)  # выход вентиля — новый провод
    n = len(wires)  # всего проводов
    return [wires[o] if 0 <= o < n else wires[-1] for o in genome["outs"]]  # для каждого номера выхода: провод o, а если o вне диапазона — последний провод


def certify(genome: dict, truth: dict, inputs_from: callable):
    """Жёсткий порог: полная проверка таблицы истинности.

        truth — словарь «ключ -> ожидаемый выход»; inputs_from(key) — входы для ключа.
        Возвращает (True, "ok"), если все строки верны, иначе (False, текст с первым несовпадением)."""
    for key, want in truth.items():  # по всем строкам таблицы
        got = emulate(genome, inputs_from(key))  # выход генома на этой строке
        if got != want:  # выход не совпал с ожидаемым
            return False, f"mismatch {key}"  # не сертифицирован; назван ключ несовпадения
    return True, "ok"  # все строки верны


def correct_fraction(genome: dict, truth: dict, inputs_from: callable) -> float:
    """Доля верных строк таблицы — мягкий сигнал поиска до сертификации.
        Порог закрепления остаётся точным certify().

        Возвращает (число верных строк) / (число всех строк); 0.0 при пустой таблице."""
    n = ok = 0  # счётчики всех и верных строк
    for key, want in truth.items():  # по строкам таблицы
        n += 1  # всех строк: +1
        if emulate(genome, inputs_from(key)) == want:  # выход совпал
            ok += 1  # верных: +1
    return ok / n if n else 0.0  # ok / n (0.0, если строк нет)


def cost(genome: dict) -> float:
    """Цель оптимизации: биты описания плюс 4·глубина: cost = program_bits + 4·depth.

        depth(i) = 1 + max(depth(a), depth(b)) для i-го вентиля; глубина входа равна 0."""
    n_in = genome["n_inputs"]  # число входов
    depth = {}  # глубины проводов-выходов вентилей: номер провода -> depth
    for i, g in enumerate(genome["gates"]):  # по вентилям
        gi = n_in + i  # номер провода выхода этого вентиля
        da = depth.get(g["a"], 0) if g["a"] >= n_in else 0  # глубина входа a (0, если a — вход)
        db = depth.get(g["b"], 0) if g["b"] >= n_in else 0  # глубина входа b (0, если b — вход)
        depth[gi] = 1 + max(da, db)  # глубина вентиля: depth = 1 + max(depth(a), depth(b))
    return program_bits(genome) + 4 * max(depth.values(), default=0)  # program_bits + 4·max(depth); без вентилей — depth = 0


# --- Сертифицированные ходы ---

def mutate(genome: dict, rng: random.Random) -> dict:
    """Один случайный ход из шести, результат — новая копия генома.

        mask — заменить маску случайного вентиля; rewire — переподключить один вход вентиля;
        insert — вставить вентиль (не более 24); delete — удалить вентиль (остаётся больше 2),
        ссылки на него перенаправляются на его вход a; out — заменить номер одного выхода;
        xor_pair — добавить вентиль ⊕ (не более 22)."""
    g = {"W": genome["W"], "n_inputs": genome["n_inputs"],  # копия генома: W и n_inputs те же
         "outs": list(genome["outs"]),  # выходы — новый список
         "gates": [dict(x) for x in genome["gates"]]}  # каждый вентиль — новая копия
    n_in = g["n_inputs"]  # число входов
    ng = len(g["gates"])  # число вентилей
    move = rng.choice(["mask", "rewire", "insert", "delete", "out",  # выбрать ход равновероятно из шести
                       "xor_pair"])  # (продолжение списка ходов)
    if move == "mask" and ng:  # ход «маска», вентили есть
        # у случайного вентиля заменить маску
        g["gates"][rng.randrange(ng)]["mask"] = \
            rng.choice(list(ALL_MASKS.values()))  # на случайную из 16
    elif move == "rewire" and ng:  # ход «переподключение», вентили есть
        i = rng.randrange(ng)  # номер вентиля
        # у входа a или b (случайно) заменить номер провода
        g["gates"][i][("a", "b")[rng.randrange(2)]] = \
            rng.randrange(n_in + ng)  # на случайный провод 0..n_in + ng − 1
    elif move == "insert" and ng < 24:  # ход «вставка», вентилей меньше 24
        srcs = n_in + ng  # число доступных проводов
        g["gates"].insert(rng.randrange(ng + 1),  # вставить вентиль на случайную позицию 0..ng
                          {"a": rng.randrange(srcs),  # вход a — случайный провод
                           "b": rng.randrange(srcs),  # вход b — случайный провод
                           "mask": rng.choice(list(ALL_MASKS.values()))})  # маска — случайная
    elif move == "delete" and ng > 2:  # ход «удаление», вентилей больше 2
        i = rng.randrange(ng)  # номер удаляемого вентиля
        victim = g["gates"].pop(i)  # удалить вентиль i
        repl = victim["a"]  # его заменитель — его собственный вход a
        for other in g["gates"]:  # по оставшимся вентилям
            for s in ("a", "b"):  # по их входам
                if other[s] == n_in + i:  # вход ссылался на удалённый вентиль
                    other[s] = repl  # перенаправить на заменитель
                elif other[s] > n_in + i:  # вход ссылался на более поздний вентиль
                    other[s] -= 1  # сдвинуть номер на 1 назад
        g["outs"] = [o if o != n_in + i else repl for o in g["outs"]]  # выходы на удалённый вентиль -> на заменитель
        g["outs"] = [o - 1 if o > n_in + i else o for o in g["outs"]]  # выходы на более поздние вентили: номер − 1
    elif move == "out" and ng:  # ход «выход», вентили есть
        g["outs"][rng.randrange(len(g["outs"]))] = rng.randrange(n_in + ng)  # случайный выход заменяется случайным проводом
    elif move == "xor_pair" and ng < 22:  # ход «пара ⊕», вентилей меньше 22
        # лемма Жегалкина: добавить вентили a ⊕ b на случайных проводах
        srcs = n_in + ng  # число доступных проводов
        g["gates"].append({"a": rng.randrange(srcs),  # добавить вентиль с входом a — случайный провод
                           "b": rng.randrange(srcs),  # вход b — случайный провод
                           "mask": MASKS["XOR"]})  # маска — исключающее ИЛИ
    return g  # изменённый геном


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    """Скрещивание: первые cut вентилей из a, остальные из b; cut — случайное из 1..len(a.gates).

        Номера проводов обрезаются до n_in + i (вентиль i не может ссылаться вперёд); номера
        выходов — до n_in + len(gates) − 1."""
    cut = rng.randrange(1, len(a["gates"]) + 1)  # точка разреза: 1..число вентилей a
    # вентили: копии первых cut вентилей a плюс копии вентилей b с номера cut
    gates = [dict(x) for x in a["gates"][:cut]] + \
            [dict(x) for x in b["gates"][cut:]]  # вторая часть — хвост b
    n_in = a["n_inputs"]  # число входов
    for i, gg in enumerate(gates):  # по вентилям потомка
        lim = n_in + i  # старший допустимый номер провода для входов этого вентиля
        gg["a"] = min(gg["a"], lim)  # вход a: не больше lim
        gg["b"] = min(gg["b"], lim)  # вход b: не больше lim
    child = dict(a)  # потомок — копия генома a
    child["gates"] = gates  # его вентили
    child["outs"] = [min(o, n_in + len(gates) - 1) for o in a["outs"]]  # выходы a, номера обрезаны до старшего провода
    return child  # потомок


# --- Эволюция: РОСТ, затем ОПТИМИЗАЦИЯ ---

def evolve(truth: dict, inputs_from: callable, n_inputs: int = 2,
           pop_size: int = 40, generations: int = 300, seed: int = 7,  # pop_size — размер популяции; generations — предел поколений; seed — зерно
           log: bool = True, profile: str = "adder",
           seed_cells: list[dict] | None = None) -> dict:  # seed_cells — готовые ячейки для стартовой популяции; возвращает итог
    """Эволюция генома до сертифицированного и затем дешёвого решения.

        truth — таблица истинности (ключ "_W" задаёт число дорожек); inputs_from — входы по ключу;
        n_inputs — число входов; pop_size — размер популяции; generations — предел поколений;
        seed — зерно случайности; log — дописывать ли результат в журнал; profile — имя схемы;
        seed_cells — проверенные готовые ячейки для начальной популяции.

        Пока нет сертифицированного генома: родители — лучшая четверть по доле верных строк,
        потомок = mutate(crossover(x, y)); если 200 поколений без улучшения, половина популяции —
        свежие случайные геномы. Как только появились 2 и более сертифицированных: элита — лучшая
        по цене четверть, в популяцию попадают только сертифицированные потомки, цена лучшего
        только убывает. Останов: через 80 поколений после первой сертификации.

        Возвращает словарь {profile, seed, certified_gen, best, bits, history, ts}."""
    rng = random.Random(seed)  # генератор случайных чисел с зерном seed
    W = truth.get("_W", 8)  # число дорожек: ключ "_W" или 8 по умолчанию
    table = {k: v for k, v in truth.items() if k != "_W"}  # сама таблица истинности без служебного ключа
    n_out = len(next(iter(table.values())))  # число выходов — длина первого ожидаемого выхода
    pop = [random_genome(rng, W=W, n_inputs=n_inputs, n_outputs=n_out)  # начальная популяция
           for _ in range(pop_size)]  # из pop_size случайных геномов
    # учебная программа: в начальную популяцию добавляется проверенная ячейка
    # (как в библиотеках стандартных ячеек: ячейка несёт свою полную таблицу истинности,
    # и поиск композиции начинает с проверенных частей)
    if seed_cells:  # готовые ячейки переданы
        for sc in seed_cells[:max(1, pop_size // 4)]:  # по первым max(1, pop_size/4) ячейкам
            g = {"W": W, "n_inputs": n_inputs,  # геном из ячейки: W и n_inputs текущие
                 "gates": [dict(x) for x in sc["gates"]],  # копии вентилей ячейки
                 "outs": list(sc.get("outs", []))}  # копии выходов (нет поля — пусто)
            if g["outs"]:  # у ячейки есть выходы
                g["outs"] = [min(o, n_inputs + len(g["gates"]) - 1)  # обрезать номера выходов до старшего провода
                             for o in g["outs"]]  # по всем выходам ячейки
            pop[0] = g  # заменить первый геном популяции на ячейку
            break  # берётся только первая ячейка
    certified, best, grow_gen = None, None, None  # первый сертифицированный геном, лучший по цене, поколение сертификации
    best_soft = [0.0]  # лучшая доля верных строк до сертификации (список — для записи из цикла)
    best_soft_gen = [0]  # поколение этого улучшения
    history = []  # журнал истории поиска
    for gen in range(generations):  # по поколениям
        scored = []  # сертифицированные: (цена, случайное число, геном)
        soft = []  # несертифицированные: (доля верных, геном)
        for g in pop:  # по геномам популяции
            ok, _ = certify(g, table, inputs_from)  # верен ли геном на всей таблице
            if ok:  # сертифицирован
                scored.append((cost(g), rng.random(), g))  # цена и случайный разбор равных цен
            else:  # не сертифицирован
                soft.append((correct_fraction(g, table, inputs_from),  # доля верных строк
                             g))  # геном
        cert_pop = sorted(scored, key=lambda t: (t[0], t[1]))  # сертифицированные по возрастанию цены
        if cert_pop and certified is None:  # впервые появился сертифицированный геном
            certified = cert_pop[0][2]  # запомнить самый дешёвый из них
            grow_gen = gen  # поколение первой сертификации
            history.append({"phase": "GROW", "gen": gen,  # запись истории: фаза РОСТ
                            "gates": len(certified["gates"])})  # число вентилей
        if len(cert_pop) >= 2:  # фаза ОПТИМИЗАЦИИ: сертифицированных не меньше двух
            elite = [g for _, _, g in cert_pop[:max(2, pop_size // 4)]]  # элита — до четверти популяции, минимум 2
            children = list(elite)  # потомки начинаются с элиты
            while len(children) < pop_size:  # пока популяция не укомплектована
                c = mutate(crossover(rng.choice(elite), rng.choice(elite),  # скрестить двух случайных элитных геномов
                                     rng), rng)  # и применить одну мутацию
                ok, _ = certify(c, table, inputs_from)  # проверить потомка
                if ok:  # только сертифицированный потомок допускается
                    children.append(c)  # добавить в новую популяцию
            pop = children  # новая популяция
            best_now = min(children, key=cost)  # самый дешёвый в ней
            if best is None or cost(best_now) < cost(best):  # рекорд цены улучшен
                best = best_now  # новый лучший геном
                history.append({"phase": "OPTIMIZE", "gen": gen,  # запись истории: фаза ОПТИМИЗАЦИЯ
                                "gates": len(best["gates"]),  # число вентилей
                                "bits": program_bits(best),  # биты описания
                                "cost": round(cost(best), 1)})  # цена с 1 знаком
        else:  # сертифицированных нет или один: мягкий подъём к верности
            # до сертификации: мягкий подъём по доле верных строк
            # (порог закрепления остаётся точным certify())
            soft.sort(key=lambda t: -t[0])  # несертифицированные по убыванию доли верных
            if soft and soft[0][0] > best_soft[0]:  # новый рекорд доли верных
                best_soft[0] = soft[0][0]  # запомнить долю
                best_soft_gen[0] = gen  # и поколение
            # плато (200 поколений без улучшения) -> перезапуск разнообразия:
            # оставить элитную четверть, остальные — свежие «мигранты»
            plateau = gen - best_soft_gen[0] > 200  # плато: с последнего улучшения прошло больше 200 поколений
            elite = [g for _, g in soft[:max(2, pop_size // 4)]]  # элита — лучшая четверть, минимум 2
            children = list(elite)  # потомки начинаются с элиты
            while len(children) < pop_size:  # пока популяция не укомплектована
                if plateau and rng.random() < 0.5:  # на плато с вероятностью 0.5
                    children.append(random_genome(  # добавить свежий случайный геном
                        rng, W=W, n_inputs=n_inputs, n_outputs=n_out))  # с теми же W, n_inputs, n_out
                else:  # иначе
                    children.append(mutate(  # потомок от двух случайных элитных геномов
                        crossover(rng.choice(elite), rng.choice(elite),  # скрещивание
                                  rng), rng))  # плюс мутация
            pop = children  # новая популяция
        if certified is not None and gen >= grow_gen + 80:  # после первой сертификации прошло 80 поколений
            break  # остановка эволюции
    final = best or certified  # результат — лучший по цене, а если его нет — первый сертифицированный
    result = {"profile": profile, "seed": seed, "certified_gen": grow_gen,  # итог: имя схемы, зерно, поколение сертификации
              "best": final, "bits": program_bits(final) if final else None,  # лучший геном и его биты описания
              "history": history,  # история поиска
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}  # метка времени
    if log:  # запись в журнал включена
        FABRIC_LOG.parent.mkdir(parents=True, exist_ok=True)  # создать каталог журнала, если нужно
        # прочитать прежние записи журнала...
        prev = json.loads(FABRIC_LOG.read_text(encoding="utf-8")) \
            if FABRIC_LOG.exists() else []  # ...а если файла нет — пустой список
        prev.append({k: v for k, v in result.items()})  # добавить результат в список
        FABRIC_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=1),  # записать список в файл JSON с отступом 1
                              encoding="utf-8")  # в кодировке UTF-8
    return result  # итог эволюции


def render(genome: dict) -> str:
    """Текстовая схема генома: по строке на вентиль и на выход.

        Вентиль i: w{n_in+i} = ИМЯ_МАСКИ(вход a, вход b); входы названы inA, inB, ...,
        провода вентилей — w{i}. Выход j: OUT{j} = w{номер}."""
    lines = []  # накопитель строк
    n_in = genome["n_inputs"]  # число входов
    for i, g in enumerate(genome["gates"]):  # по вентилям
        gi = n_in + i  # номер провода выхода вентиля
        def nm(x):  # имя провода x
            return ("in" + "ABCDEF"[x]) if x < n_in else f"w{x}"  # вход — in + буква с номером x; иначе w{x}
        lines.append(f"w{gi} = {NAME_OF[g['mask']]}({nm(g['a'])}, {nm(g['b'])})")  # строка вентиля: w = функция(вход a, вход b)
    for j, o in enumerate(genome["outs"]):  # по выходам
        lines.append(f"OUT{j} = w{o}")  # строка выхода
    return "\n".join(lines)  # строки через перевод строки


def full_adder_profile(W: int = 8):
    """Полный сумматор: A, B, Cin -> SUM, COUT; исчерпывающая таблица из 8 строк.

        SUM = A ⊕ B ⊕ Cin; COUT = (A ∧ B) ∨ (Cin ∧ (A ⊕ B)). Бит умножается на
        full = 2^W − 1 (все W дорожек получают одно значение). Возвращает
        (таблица, функция входов, число входов 3)."""
    full = (1 << W) - 1  # full = 2^W − 1 — все W битов единичные
    truth = {}  # таблица истинности
    for a in (0, 1):  # перебор A = 0, 1
        for b in (0, 1):  # перебор B = 0, 1
            for cin in (0, 1):  # перебор Cin = 0, 1
                s = a ^ b ^ cin  # сумма по модулю 2: SUM = a ⊕ b ⊕ cin
                cout = (a & b) | (cin & (a ^ b))  # перенос: COUT = (a ∧ b) ∨ (cin ∧ (a ⊕ b))
                truth[(a, b, cin)] = [s * full, cout * full]  # строка таблицы: (A, B, Cin) -> [SUM·full, COUT·full]

    def ins(key):  # входы для ключа таблицы
        return [key[0] * full, key[1] * full, key[2] * full]  # каждый бит, размноженный на все W дорожек
    return truth, ins, 3  # таблица, функция входов и число входов
def main() -> None:  # запуск из командной строки
    import sys  # аргументы командной строки
    profile = sys.argv[1] if len(sys.argv) > 1 else "adder"  # имя схемы: первый аргумент или "adder"
    if profile != "adder":  # поддерживается только сумматор
        raise SystemExit(f"unknown profile {profile!r}; only 'adder' now")  # выход с сообщением
    truth, ins, n_in = full_adder_profile()  # таблица, входы и число входов сумматора
    res = evolve(truth, ins, n_inputs=n_in, profile=profile)  # эволюция сумматора
    print(json.dumps({k: v for k, v in res.items() if k != "best"},  # напечатать итог JSON без самого генома
                     indent=2, ensure_ascii=False))  # с отступами и без экранирования не-латиницы
    print(render(res["best"]))  # напечатать схему лучшего генома


if __name__ == "__main__":  # файл запущен как скрипт
    main()  # разбор аргументов и эволюция
