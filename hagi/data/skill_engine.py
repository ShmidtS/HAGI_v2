#!/usr/bin/env python3
"""Движок временных навыков: учит последовательностные функции (автоматы Мили) по оракулу,
собирает из них библиотеку композиций и хранит всё в виде схем из логических вентилей.

Представление. Функция от n битов — многочлен над GF(2) во множестве мономов:
моном — целое число, где бит i означает переменную x_i; пустой моном (число 0) — единица.
f(x) = XOR по мономам t из p, для которых t ⊆ x (то есть t & x == t).
Схема (нетлист) — словарь {n_inputs, gates, outs}: провода 0..n−1 — входы, провод n+i —
выход вентиля i; вентиль {a, b, mask}, где mask — 4-битная таблица истинности от (a, b),
индекс a | (b << 1): 0 — константа 0, 15 — константа 1, 6 — XOR, 8 — AND.

Основные части:
1. Многочлены GF(2), таблица Мёбиуса (переход таблица истинности -> АНФ), вычисление и
   символьная проверка схем.
2. Уплотнение схем с проверяемым доказательством, синтез схем с разложением на множители.
3. Минимизация автомата (разбиение на классы эквивалентности), кодирование состояний.
4. Навык Skill, алгебра композиций, глобальная схема TemporalDAG, библиотека.
5. Оракул Мили с фильтром шума, тесты различения состояний, активное обучение L*.
6. Агент: обучение примитивов, композиция, решение целей, сохранение и загрузка состояния.
7. Независимая проверка сохранённого состояния, самопроверка, командная строка."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import argparse, functools, hashlib, heapq, itertools, json, math, random, time  # импорт стандартной библиотеки: разбор аргументов, кэш, хеши, куча, перестановки, JSON, математика, случайность, время
from collections import deque  # двусторонняя очередь для обхода в ширину
from dataclasses import dataclass  # декоратор класса-записи с полями
from pathlib import Path  # работа с путями файлов
from typing import Callable, Iterable  # типы «функция» и «итератор» для аннотаций

DEFAULT_ISSUER = "HAGI_TEMPORAL_AGENT_V40"  # имя издателя по умолчанию (строка-константа)
GLOBAL_N = 24  # входные биты глобальной схемы: S0..S21 — состояние (биты 0..21), U0..U1 — вход (биты 22..23)
AUDIT_CAP = 12  # предел полного перебора: при числе переменных не больше 12 — полная проверка и минимизация, свыше — модульная композиция с доказательством уплотнения


# ---------------------------------------------------------------------------
# 1. Многочлены над GF(2), метрики схемы и символьная проверка
# ---------------------------------------------------------------------------

def xor(a: frozenset[int], b: frozenset[int]) -> frozenset[int]:
    """Сложение многочленов над GF(2): сумма по модулю 2.
    
            a, b: множества мономов. Возвращает симметрическую разность: моном входит в
            результат, если он ровно в одном из операндов (p + p = 0)."""
    return a.symmetric_difference(b)  # p = a Δ b (симметрическая разность: мономы ровно в одном из множеств)

def mul(a: frozenset[int], b: frozenset[int]) -> frozenset[int]:
    """Умножение многочленов над GF(2).
    
            a, b: множества мономов. Произведение мономов x^S · x^T = x^(S ∪ T);
            коэффициенты складываются по модулю 2, поэтому повторившийся моном сокращается.
            Возвращает frozenset мономов произведения."""
    z: set[int] = set()  # накопитель результата z (множество мономов)
    for x in a:  # по каждому моному x первого сомножителя
        for y in b:  # по каждому моному y второго сомножителя
            t = x | y  # произведение мономов: t = x ∪ y (поразрядное ИЛИ множеств переменных)
            if t in z: z.remove(t)  # если моном t уже встречался — он сокращается (коэффициент 1 + 1 = 0)
            else: z.add(t)  # иначе добавить моном t (коэффициент 1)
    return frozenset(z)  # результат как неизменяемое множество

def ev(p: frozenset[int], x: int) -> int:
    """Подставить точку x в многочлен p: p(x) = Σ_{t ⊆ x} 1 (mod 2).
    
            p: множество мономов, x: точка (бит i = значение x_i). Возвращает 0 или 1:
            число мономов t, у которых t & x == t (все переменные монома равны 1), по модулю 2."""
    return sum(1 for t in p if (t & x) == t) & 1  # p(x) = (число мономов t с t ⊆ x) mod 2

def deg(p: frozenset[int]) -> int:
    """Степень многочлена p: число переменных в самом длинном его мономе.
    
            Возвращает max popcount(t) по мономам t из p; для пустого p — 0."""
    return max((t.bit_count() for t in p), default=0)  # max popcount(t) по всем мономам; пустое множество даёт 0

from hagi.base.util import sha  # импорт хеш-функции из общего модуля проекта

digest = sha  # digest = sha: хеш содержимого, единый для всех объектов (ключ дедупликации)

def binom_p_ge(m: int, k: int) -> float:
    """Вероятность того, что число успехов не меньше k в m испытаниях Бернулли с p = 1/2.
    
            P(X ≥ k) = Σ_{j=k..m} C(m, j) / 2^m. Используется для порога «большинства голосов»."""
    return sum(math.comb(m, j) for j in range(k, m + 1)) / (1 << m)  # биномиальный хвост: P(X ≥ k) = Σ_{j=k..m} C(m, j) / 2^m

@functools.lru_cache(maxsize=8)  # кэш на 8 наборов аргументов (таблица зависит только от шага alpha и границ m)
def _binom_thresh_table(alpha_step: float, min_m: int, max_m: int) -> tuple[int, ...]:
    """Порог большинства для m повторений: наименьшее k, при котором P(X ≥ k) ≤ alpha_step.
    
            alpha_step: допустимая вероятность ошибки на один шаг, min_m..max_m: диапазон m.
            Идёт сверху: req опускается, пока binom_p_ge(m_tr, k) ≤ alpha_step; первое k, где
            условие нарушено, останавливает спуск (хвост только растёт при убывании k).
            Возвращает tbl[m] = req для min_m ≤ m ≤ max_m, для m < min_m — max_m + 1 (порог недостижим)."""
    tbl = [max_m + 1] * (max_m + 1)  # tbl = [max_m + 1] * (max_m + 1): по умолчанию порог недостижим
    for m_tr in range(min_m, max_m + 1):  # по каждому числу повторений m_tr
        req = m_tr + 1  # начальный порог req = m_tr + 1 (единогласие недостаточно)
        for k in range(m_tr, m_tr // 2, -1):  # k от m_tr вниз до m_tr // 2 (не ниже половины)
            if binom_p_ge(m_tr, k) <= alpha_step: req = k  # если P(X ≥ k) ≤ alpha_step — порог снижается до k
            else: break  # иначе останов: при меньшем k хвост P(X ≥ k) только растёт
        tbl[m_tr] = req  # запомнить порог для m_tr
    return tuple(tbl)  # неизменяемая таблица порогов

def mobius(rows: list[tuple[int, ...]], n: int) -> tuple[frozenset[int], ...]:
    """Таблица Мёбиуса: булеву таблицу значений превращает в коэффициенты АНФ (набор мономов).
    
            rows: 2^n строк таблицы (каждая — кортеж выходных битов), n: число переменных.
            Для каждого выхода j берётся столбец a[x]; преобразование Мёбиуса по каждой
            переменной bit: a[x] ^= a[x без бита bit] для всех x с единичным битом.
            Возвращает для каждого столбца множество точек x, где коэффициент a[x] = 1
            (точка x = множество переменных монома)."""
    out = []  # список результатов по столбцам
    for j in range(len(rows[0])):  # по каждому выходному столбцу j
        a = [int(r[j]) & 1 for r in rows]  # столбец значений как список битов a[x]
        for bit in range(n):  # по каждой переменной (биту) отдельно
            step = 1 << bit  # маска step = 2^bit этой переменной
            for x in range(1 << n):  # по всем точкам x
                if x & step: a[x] ^= a[x ^ step]  # если бит x равен 1: a[x] = a[x] ⊕ a[x ⊕ step] (добавить значение без этой переменной)
        out.append(frozenset(i for i, v in enumerate(a) if v))  # мономы столбца: все x, где коэффициент a[x] = 1
    return tuple(out)  # кортеж АНФ всех выходов

def substitute(p: frozenset[int], subs: tuple[frozenset[int], ...]) -> frozenset[int]:
    """Подстановка многочленов вместо переменных многочлена p.
    
            p: множество мономов, subs: кортеж многочленов,subs[i] подставляется вместо x_i.
            Каждый моном term заменяется произведением Π subs[i] по i, входящим в term;
            результаты складываются по модулю 2. Возвращает многочлен p(subs)."""
    out: frozenset[int] = frozenset()  # накопитель суммы out = 0
    for term in p:  # по каждому моном term многочлена p
        prod: frozenset[int] = frozenset({0})  # произведение начинается с единицы (моном 0)
        for i, q in enumerate(subs):  # по переменной i и её замене q
            if (term >> i) & 1: prod = mul(prod, q)  # если бит i входит в term: prod = prod · q
        out = xor(out, prod)  # out = out ⊕ prod (сумма мономов)
    return out  # результат подстановки

def remap(p: frozenset[int], mapping: tuple[int, ...]) -> frozenset[int]:
    """Переименовать переменные многочлена p: x_i -> x_{mapping[i]}.
    
            Возвращает многочлен с мономами, где бит i заменён на бит mapping[i]:
            t' = Σ 2^{mapping[i]} по битам i из t."""
    return frozenset(sum(1 << mapping[i] for i in range(len(mapping)) if (t >> i) & 1) for t in p)  # t' = { 2^{mapping[i]} : бит i входит в t } для каждого монома t

def _finalize_net(n: int, gates: list[dict], outs: list[int]) -> dict:
    """Завершить построение схемы: добавить вычисленные метрики.
    
            n: число входов, gates: список вентилей, outs: номера выходных проводов.
            depth провода вентиля = 1 + max(depth входов); fan — число использований провода.
            program_bits = len(gates)·(4 + 2·wb) + len(outs)·ob, где wb = бит на номер провода
            = ceil(log2 nw), ob = бит на выход = ceil(log2(nw + 1)). Возвращает словарь схемы с метриками."""
    nw = n + len(gates); fan = [0] * nw; depth = [0] * nw  # nw — всего проводов; fan и depth — нулевые списки длины nw
    for i, g in enumerate(gates):  # по каждому вентилю i (его провод n + i)
        depth[n + i] = 1 + max(depth[g["a"]], depth[g["b"]])  # depth вентиля = 1 + max(depth(a), depth(b))
        fan[g["a"]] += 1; fan[g["b"]] += 1  # входы a и b получают по одному использованию
    for o in outs: fan[o] += 1  # каждый выходной провод получает ещё одно использование
    wb = max(1, (nw - 1).bit_length()); ob = max(1, nw.bit_length())  # wb = бит на номер провода = ceil(log2 nw); ob = бит на выход = ceil(log2(nw + 1)); минимум 1
    return {"n_inputs": n, "gates": gates, "outs": outs,  # собрать словарь схемы
            "metrics": {"gate_count": len(gates), "depth": max((depth[o] for o in outs), default=0),  # метрики: число вентилей, глубина = max depth по выходам (пусто — 0)
                        "max_fanout": max(fan, default=0), "program_bits": len(gates) * (4 + 2 * wb) + len(outs) * ob}}  # максимальный разветвитель fan, размер программы в битах по формуле выше

def eval_net(net: dict, inputs: tuple[int, ...]) -> tuple[int, ...]:
    """Вычислить выходы схемы на заданных входах (по одному биту на вход).
    
            inputs: кортеж из n бит. Значения проводов идут по порядку: входы, затем вентили;
            значение вентиля по mask: 0 -> 0, 15 -> 1, 6 -> a ⊕ b, 8 -> a ∧ b, иначе ошибка.
            Возвращает кортеж значений выходных проводов."""
    n = net["n_inputs"]  # число входов n
    if len(inputs) != n: raise ValueError("input arity mismatch")  # число входов должно совпадать с арностью схемы
    w = [int(x) & 1 for x in inputs]  # значения входных проводов как биты (младший разряд целого)
    for i, g in enumerate(net["gates"]):  # по каждому вентилю i (его провод n + i)
        a, b, m = g["a"], g["b"], g["mask"]  # номера входных проводов и маска вентиля
        if not (0 <= a < n + i and 0 <= b < n + i): raise ValueError(f"forward edge {i}")  # оба входа должны быть уже вычислены (номер меньше n + i)
        w.append(0 if m == 0 else (1 if m == 15 else (w[a] ^ w[b] if m == 6 else (w[a] & w[b] if m == 8 else -1))))  # значение вентиля: 0 при m = 0, 1 при m = 15, a ⊕ b при m = 6, a ∧ b при m = 8, иначе −1 (неизвестно)
        if w[-1] == -1: raise ValueError(f"unsupported mask {m}")  # маска вне набора {0, 6, 8, 15} — ошибка
    if any(o < 0 or o >= len(w) for o in net["outs"]): raise ValueError("bad output")  # все выходы должны указывать на существующий провод
    return tuple(w[o] for o in net["outs"])  # значения выходных проводов

def symbolic_eval_net_general(net: dict, input_polys=None) -> tuple[frozenset[int], ...]:
    """Символьно вычислить схему: на выходах — многочлены АНФ от входов.
    
            net: схема, input_polys: многочлены входных проводов (по умолчанию провод i = x_i).
            Вентиль по mask: 0 -> 0, 15 -> 1, 6 -> XOR многочленов, 8 -> произведение.
            Возвращает кортеж многочленов выходов."""
    n = net["n_inputs"]  # n — число входов сети
    w = [frozenset({1 << i}) for i in range(n)] if input_polys is None else list(input_polys)  # вход i — многочлен из одного монома x_i (если многочлены входов не заданы)
    for i, g in enumerate(net["gates"]):  # по каждому вентилю i
        a, b, m = g["a"], g["b"], g["mask"]  # входы и маска вентиля
        if not (0 <= a < n + i and 0 <= b < n + i): raise ValueError(f"forward edge {i}")  # проверка ацикличности и порядка: входы раньше вентиля
        if m == 0: z: frozenset[int] = frozenset()  # константа 0 — пустой многочлен
        elif m == 15: z = frozenset({0})  # константа 1 — многочлен из монома 0
        elif m == 6: z = xor(w[a], w[b])  # XOR: z = a ⊕ b (симметрическая разность мономов)
        elif m == 8: z = mul(w[a], w[b])  # И: z = a · b (AND)
        else: raise ValueError(f"unsupported mask {m}")  # иная маска — ошибка
        w.append(z)  # добавить многочлен вентиля в список проводов
    if any(o < 0 or o >= len(w) for o in net["outs"]): raise ValueError("bad output")  # все выходы должны указывать на существующий провод
    return tuple(w[o] for o in net["outs"])  # многочлены выходов

def symbolic_verify_general(net: dict, expected: tuple[frozenset[int], ...]) -> bool:
    """Символьная проверка: совпадают ли многочлены выходов схемы с ожидаемыми.
    
            expected: кортеж многочленов. Возвращает True при полном совпадении, False при любом
            расхождении или ошибке вычисления (например, некорректная маска)."""
    try: return symbolic_eval_net_general(net) == expected  # True, если многочлены выходов равны expected
    except Exception: return False  # ошибка вычисления считается несовпадением


# ---------------------------------------------------------------------------
# 2. Уплотнение схемы с доказательством, синтез схем над GF(2) и входы АНФ
# ---------------------------------------------------------------------------

def compact_netlist_with_proof(net: dict) -> tuple[dict, dict]:
    """Уплотнить схему и выдать доказательство каждого шага.
    
            net: исходная схема. Правила для каждого вентиля: константы и повторяющиеся
            выражения используются повторно; u ⊕ u = 0, u ⊕ 0 = u, 1 ⊕ 1 = 0, u ∧ u = u,
            u ∧ 0 = 0, u ∧ 1 = u; входы XOR/AND упорядочиваются (a ≤ b).
            Затем удаляются вентили, не влияющие на выходы, и провода перенумеровываются.
            Возвращает (схему после уплотнения, доказательство).
            Доказательство — proof_steps: по одному шагу на провод исходной схемы, raw_wire -> opt_wire
            с именем правила; pre_dce_gates, pre_dce_outs — схема до удаления неиспользуемых вентилей;
            dce_remap — перенумерация проводов при удалении."""
    n = net["n_inputs"]; val: list[int | None] = [None] * n; alias = list(range(n))  # n — число входов; val — известная константа провода (0, 1 или None); alias: провод исходной схемы -> провод новой
    struct: dict[tuple[int, int, int], int] = {}; new_gates: list[dict] = []  # struct: сигнатура вентиля (mask, a, b) -> номер провода; new_gates — новые вентили
    proof_steps = [{"raw_wire": i, "rule": "INPUT", "opt_wire": i} for i in range(n)]  # доказательство: первые n проводов — входы, правило INPUT, провод сохраняется

    def get_const(c: int, raw_w: int, rule: str) -> int:
        """Получить провод константы c (0 или 1) и записать шаг доказательства.
        
                    c: значение константы, raw_w: провод исходной схемы, rule: имя правила.
                    Если такая константа уже есть — шаг «*_CSE» (повторное использование),
                    иначе создаётся вентиль — шаг «*_NEW». Возвращает номер провода константы."""
        sig = (15 if c else 0, 0, 0)  # сигнатура константы: маска 15 при c = 1, иначе 0, с нулевыми входами
        if sig in struct:  # константа уже создана
            w = struct[sig]; proof_steps.append({"raw_wire": raw_w, "rule": rule + "_CSE", "opt_wire": w}); return w  # вернуть её провод и записать шаг «повторное использование»
        w = n + len(new_gates); new_gates.append({"a": 0, "b": 0, "mask": 15 if c else 0})  # новый вентиль-константа: маска 15 при c = 1, иначе 0
        struct[sig] = w; val.append(c); proof_steps.append({"raw_wire": raw_w, "rule": rule + "_NEW", "opt_wire": w}); return w  # запомнить провод, значение и шаг «новый вентиль»; вернуть провод
  # вернуть номер провода константы
    for idx, g in enumerate(net["gates"]):  # по номеру idx и элементу g списка элементов сети
        raw_w = n + idx; a, b, m = alias[g["a"]], alias[g["b"]], g["mask"]  # по каждому вентилю исходной схемы (idx — номер, raw_w — его провод n + idx)
        if m == 0: alias.append(get_const(0, raw_w, "CONST0")); continue  # входы через alias (уже перенесённые провода), маска
        if m == 15: alias.append(get_const(1, raw_w, "CONST1")); continue  # константа 0 -> провод константы нуля, правило CONST0
        va, vb = val[a], val[b]  # константа 1 -> провод константы единицы, правило CONST1
        if m == 6:  # XOR: правила сокращения
            if a == b: alias.append(get_const(0, raw_w, "XOR_SELF")); continue  # самоисключение: u ⊕ u = 0
            if va == 0: alias.append(b); proof_steps.append({"raw_wire": raw_w, "rule": "XOR_ZERO_L", "opt_wire": b}); continue  # 0 ⊕ v = v (левый вход — константа 0), правило XOR_ZERO_L
            if vb == 0: alias.append(a); proof_steps.append({"raw_wire": raw_w, "rule": "XOR_ZERO_R", "opt_wire": a}); continue  # u ⊕ 0 = u, правило XOR_ZERO_R
            if va == 1 and vb == 1: alias.append(get_const(0, raw_w, "XOR_ONE_ONE")); continue  # сумма двух единиц: 1 ⊕ 1 = 0
            if b < a: a, b = b, a  # коммутативность: упорядочить входы a ≤ b
        elif m == 8:  # AND: правила сокращения
            if a == b: alias.append(a); proof_steps.append({"raw_wire": raw_w, "rule": "AND_IDEM", "opt_wire": a}); continue  # u ∧ u = u (идемпотентность)
            if va == 0 or vb == 0: alias.append(get_const(0, raw_w, "AND_ZERO")); continue  # 0 ∧ v = 0 или u ∧ 0 = 0
            if va == 1: alias.append(b); proof_steps.append({"raw_wire": raw_w, "rule": "AND_ONE_L", "opt_wire": b}); continue  # 1 ∧ v = v (левый вход — константа 1)
            if vb == 1: alias.append(a); proof_steps.append({"raw_wire": raw_w, "rule": "AND_ONE_R", "opt_wire": a}); continue  # правая единица: u ∧ 1 = u
            if b < a: a, b = b, a  # коммутативность: упорядочить входы a ≤ b
        sig = (m, a, b)  # сигнатура вентиля после упрощения
        if sig in struct:  # такой вентиль уже строился
            w = struct[sig]; alias.append(w); proof_steps.append({"raw_wire": raw_w, "rule": "CSE_REUSE", "opt_wire": w})  # вентиля расшарить: alias на существующий провод, шаг CSE_REUSE
        else:  # иначе
            w = n + len(new_gates); new_gates.append({"a": a, "b": b, "mask": m})  # новый вентиль с номером провода n + len(new_gates)
            struct[sig] = w; val.append(None); alias.append(w)  # запомнить сигнатуру, значение неизвестно, alias, шаг EMIT_GATE
            proof_steps.append({"raw_wire": raw_w, "rule": "EMIT_GATE", "opt_wire": w})  # шаг доказательства: исходный провод, правило EMIT_GATE, новый провод w
  # выходы до удаления неиспользуемых вентилей (через alias)
    outs_pre = [alias[o] for o in net["outs"]]  # выходы до чистки: подстановка alias для каждого выхода o
    used = {o for o in outs_pre if o >= n}; stack = list(used)  # обход в ширину от выходов: used — используемые провода вентилей (≥ n)
    while stack:  # пока есть непросмотренные провода
        w = stack.pop(); g = new_gates[w - n]  # взять провод w и его вентиль
        for ch in (g["a"], g["b"]):  # по обоим входам вентиля
            if ch >= n and ch not in used: used.add(ch); stack.append(ch)  # вход-вентиль, ещё не отмеченный, добавить в used и в стек
    remap_w = {i: i for i in range(n)}; final_gates = []  # перенумерация: входы сохраняют номера; final_gates — вентили в порядке возрастания
    for idx, old_w in enumerate(sorted(used)):  # по использованным проводам в возрастающем порядке (idx — новый номер)
        g = new_gates[old_w - n]; remap_w[old_w] = n + idx  # новый номер провода = n + idx
        final_gates.append({"a": remap_w[g["a"]], "b": remap_w[g["b"]], "mask": g["mask"]})  # вентиль с перенумерованными входами
    opt_net = _finalize_net(n, final_gates, [remap_w[o] for o in outs_pre])  # итоговая схема: выходы перенумерованы; вместе с доказательством (dce_remap — ключи строками для JSON)
    return opt_net, {"proof_steps": proof_steps, "pre_dce_gates": new_gates, "pre_dce_outs": outs_pre, "dce_remap": {str(k): v for k, v in remap_w.items()}}  # вернуть оптимизированную сеть и сведения о доказательстве (шаги, элементы и выходы до чистки, перенумерация проводов)

def compact_netlist(net: dict) -> dict:
    """Уплотнить схему (без доказательства).
    
            net: исходная схема. Возвращает только схему — первый элемент результата compact_netlist_with_proof."""
    return compact_netlist_with_proof(net)[0]  # вернуть первую часть результата (схему), доказательство отбрасывается

def verify_compaction_proof(raw_net: dict, opt_net: dict, proof: dict) -> tuple[bool, str]:
    """Проверить доказательство уплотнения: оптимизированная схема обязана быть равна исходной.
    
            raw_net: исходная схема, opt_net: уплотнённая, proof: доказательство.
            Проверка по шагам: число входов и выходов; число шагов = n + число вентилей;
            для каждого шага — законность правила (например, XOR_ZERO_L требует маску 6,
            константу 0 на левом входе и выход, равный правому входу); совпадение выходов
            до удаления неиспользуемых вентилей; взаимно однозначное соответствие вентилей
            после перенумерования; совпадение выходов.
            Возвращает (True, «verified») или (False, причина отказа)."""
    n = raw_net["n_inputs"]  # число входов
    if opt_net["n_inputs"] != n or len(opt_net["outs"]) != len(raw_net["outs"]): return False, "arity"  # число входов и число выходов обеих схем обязаны совпасть
    pre_gates, steps = proof["pre_dce_gates"], proof["proof_steps"]  # схема до удаления неиспользуемых вентилей и шаги доказательства
    if len(steps) != n + len(raw_net["gates"]): return False, "step count"  # шагов ровно n + число вентилей исходной схемы (по одному на провод)
    const_val: list[int | None] = [None] * n  # const_val: известная константа провода до удаления (0, 1, None — неизвестно, −1 — плохая маска)
    for i, g in enumerate(pre_gates):  # по каждому вентилю схемы до удаления
        if not (0 <= g["a"] < n + i and 0 <= g["b"] < n + i): return False, f"pre_dce forward {i}"  # оба входа вентиля должны быть раньше него
        const_val.append(0 if g["mask"] == 0 else (1 if g["mask"] == 15 else (None if g["mask"] in (6, 8) else -1)))  # константа вентиля: 0 при маске 0, 1 при маске 15, неизвестна при 6 и 8, иначе −1
        if const_val[-1] == -1: return False, "bad mask"  # неизвестная маска — отказ
    rep = [0] * (n + len(raw_net["gates"]))  # rep: провод исходной схемы -> провод оптимизированной (длина n + число вентилей)
    for i in range(n):  # по каждому входу i
        st = steps[i]  # шаг входа i
        if st["raw_wire"] != i or st["rule"] != "INPUT" or st["opt_wire"] != i: return False, f"input step {i}"  # шаг обязан быть INPUT с теми же номерами проводов
        rep[i] = i  # rep[i] = i (входы не перенумеровываются)
    for idx, g in enumerate(raw_net["gates"]):  # по каждому вентилю исходной схемы
        rw = n + idx; st = steps[rw]; ow, rule = st["opt_wire"], st["rule"]  # rw — его провод; шаг, выходной провод ow и правило
        if st["raw_wire"] != rw or not (0 <= ow < n + len(pre_gates)): return False, f"step bounds {rw}"  # номер провода в шаге и границы ow (0 ≤ ow < n + число вентилей до удаления) проверяются
        ra, rb, m = rep[g["a"]], rep[g["b"]], g["mask"]  # входы через rep и маска вентиля
        if rule.startswith("CONST0"): ok = (m == 0 and const_val[ow] == 0)  # CONST0*: маска 0 и выход — константа 0
        elif rule.startswith("CONST1"): ok = (m == 15 and const_val[ow] == 1)  # CONST1*: маска 15 и выход — константа 1
        elif rule.startswith("XOR_SELF"): ok = (m == 6 and ra == rb and const_val[ow] == 0)  # XOR_SELF*: маска 6, равные входы, выход 0
        elif rule == "XOR_ZERO_L": ok = (m == 6 and const_val[ra] == 0 and ow == rb)  # XOR_ZERO_L: маска 6, левый вход 0, выход равен правому входу
        elif rule == "XOR_ZERO_R": ok = (m == 6 and const_val[rb] == 0 and ow == ra)  # XOR_ZERO_R: маска 6, правый вход 0, выход равен левому входу
        elif rule.startswith("XOR_ONE_ONE"): ok = (m == 6 and const_val[ra] == 1 and const_val[rb] == 1 and const_val[ow] == 0)  # XOR_ONE_ONE*: маска 6, оба входа 1, выход 0
        elif rule == "AND_IDEM": ok = (m == 8 and ra == rb and ow == ra)  # AND_IDEM: маска 8, равные входы, выход равен им
        elif rule.startswith("AND_ZERO"): ok = (m == 8 and (const_val[ra] == 0 or const_val[rb] == 0) and const_val[ow] == 0)  # AND_ZERO*: маска 8, есть нулевой вход, выход 0
        elif rule == "AND_ONE_L": ok = (m == 8 and const_val[ra] == 1 and ow == rb)  # AND_ONE_L: маска 8, левый вход 1, выход равен правому входу
        elif rule == "AND_ONE_R": ok = (m == 8 and const_val[rb] == 1 and ow == ra)  # AND_ONE_R: маска 8, правый вход 1, выход равен левому входу
        elif rule in ("CSE_REUSE", "EMIT_GATE"):  # CSE_REUSE и EMIT_GATE: общий вид проверки
            ok = (ow >= n and pre_gates[ow - n]["mask"] == m and sorted((pre_gates[ow - n]["a"], pre_gates[ow - n]["b"])) == sorted((ra, rb)))  # выход — вентиль до удаления с той же маской и теми же входами (порядок не важен)
        else: ok = False  # иное правило недопустимо
        if not ok: return False, f"{rule} {rw}"  # правило не выполнено — отказ с именем правила и номером провода
        rep[rw] = ow  # запомнить соответствие провода
    if [rep[o] for o in raw_net["outs"]] != proof["pre_dce_outs"]: return False, "pre_dce outs"  # выходы через rep обязаны совпасть с выходами до удаления
    dce_map = {int(k): v for k, v in proof["dce_remap"].items()}  # карта перенумерования (ключи превращаются в целые числа)
    if any(dce_map.get(i) != i for i in range(n)): return False, "dce inputs"  # входы при перенумерации не двигаются
    for idx, og in enumerate(opt_net["gates"]):  # по каждому вентилю оптимизированной схемы
        if not (0 <= og["a"] < n + idx and 0 <= og["b"] < n + idx): return False, f"opt forward {idx}"  # оба входа вентиля должны быть раньше него
        matched = [k for k, v in dce_map.items() if v == n + idx and k >= n]  # ровно один старый провод k ≥ n отображается в этот вентиль
        if len(matched) != 1: return False, f"dce map {idx}"  # число прообразов не равно 1 — отказ
        pg = pre_gates[matched[0] - n]  # соответствующий вентиль до удаления
        if pg["mask"] != og["mask"] or og["a"] != dce_map.get(pg["a"]) or og["b"] != dce_map.get(pg["b"]): return False, f"dce mismatch {idx}"  # маска и перенумерованные входы обязаны совпасть
    if [dce_map.get(o) for o in proof["pre_dce_outs"]] != opt_net["outs"]: return False, "dce outs"  # выходы после перенумерования обязаны совпасть с выходами оптимизированной схемы
    return True, "verified"  # доказательство принято

@functools.lru_cache(maxsize=512)  # кэш на 512 наборов аргументов
def compile_factored_gf2(n: int, outputs: tuple[frozenset[int], ...],
                       xor_quad_resub: bool = False) -> dict:  # второй параметр xor_quad_resub (по умолчанию False)
    """Синтезировать схему для набора выходных многочленов над GF(2), используя уже построенные провода.
    
                n: число переменных, outputs: кортеж многочленов.
                xor_quad_resub (по умолчанию выключен): разрешить p = w ⊕ q, где q — единственный
                квадратичный моном, ещё не имеющий провода (q строится как AND, затем XOR).
                Это жадное первое совпадение: для умножителя 2×2 оно даёт меньше вентилей,
                но в среднем чуть больше вентилей и бит, поэтому вариант — лишь кандидат
                при выборе наименьшей схемы, а не замена.
                Возвращает уплотнённую схему."""
    """Multi-output Davio + Affine Divisor + 1-gate Functional Resubstitution over GF(2)."""
    gates: list[dict] = []  # список вентилей
    poly_wire: dict[frozenset[int], int] = {frozenset({1 << i}): i for i in range(n)}  # многочлен -> номер провода; вход i имеет многочлен {x_i}
    wire_poly: list[frozenset[int]] = [frozenset({1 << i}) for i in range(n)]  # провод -> многочлен (обратное соответствие); входы — переменные

    def emit(a: int, b: int, mask: int, poly: frozenset[int]) -> int:
        """Создать вентиль с данным многочленом или вернуть существующий провод.
        
                    a, b: входы, mask: маска, poly: многочлен результата. Если poly уже есть —
                    возвращается его провод. Иначе добавляется вентиль, для 6 и 8 входы упорядочиваются.
                    Возвращает номер провода."""
        if poly in poly_wire: return poly_wire[poly]  # многочлен уже на проводе — вернуть его
        if mask in (6, 8) and b < a: a, b = b, a  # для коммутативных вентилей входы упорядочиваются: a ≤ b
        w = n + len(gates); gates.append({"a": a, "b": b, "mask": mask})  # номер нового провода n + len(gates); добавить вентиль
        poly_wire[poly] = w; wire_poly.append(poly)  # запомнить многочлен провода в обеих сторонах
        return w  # вернуть номер провода

    def get_const(c: int) -> int:
        """Получить провод константы c (0 или 1).
        
                    Константа 1 — многочлен {0}, константа 0 — пустой. Возвращает существующий провод
                    или создаёт вентиль с маской 15 (для 1) или 0."""
        p = frozenset({0}) if c else frozenset()  # многочлен константы: {0} при c = 1, иначе пустой
        return poly_wire[p] if p in poly_wire else emit(0, 0, 15 if c else 0, p)  # вернуть провод, если он есть, иначе создать вентиль-константу

    def try_resub(p: frozenset[int]) -> int | None:
        """Построить многочлен p из имеющихся проводов одним вентилем, если возможно.
        
                    Порядок попыток: p уже есть; p = pa ⊕ rem, где rem = p ⊕ pa уже на проводе;
                    при xor_quad_resub: p = pa ⊕ (b1 ∧ b2) с единственным квадратичным моном;
                    иначе p = pa · pb для пары делителей (проверка mul(pa, pb) == p, отсев по степеням
                    deg(pa) + deg(pb) ≥ deg(p)). Возвращает провод или None."""
        if p in poly_wire: return poly_wire[p]  # многочлен уже на проводе
        for pa, wa in poly_wire.items():  # по каждому имеющемуся многочлену pa и его проводу wa
            rem = xor(p, pa)  # остаток rem = p ⊕ pa
            if rem in poly_wire: return emit(wa, poly_wire[rem], 6, p)  # rem уже на проводе: p = wa ⊕ rem (один вентиль XOR)
        if xor_quad_resub and len(wire_poly) <= 32:  # квадратичное повторное использование: только при числе проводов ≤ 32
            for pa, wa in poly_wire.items():  # по каждому имеющемуся многочлену pa
                rem = xor(p, pa)  # остаток rem = p ⊕ pa
                if len(rem) == 1 and next(iter(rem)).bit_count() == 2:  # rem — единственный моном степени 2
                    t = next(iter(rem)); b1 = t & -t; b2 = t ^ b1  # разложить моном t на две переменные: b1 — младший установленный бит, b2 = t ⊕ b1
                    wq = emit(poly_wire[frozenset({b1})], poly_wire[frozenset({b2})], 8, rem)  # построить q = b1 ∧ b2 (AND) и вернуть p = pa ⊕ q (XOR)
                    return emit(wa, wq, 6, p)  # вернуть p = pa ⊕ q
        if len(wire_poly) <= 32:  # делители: только при числе проводов ≤ 32
            items = [(pa, wa, deg(pa)) for pa, wa in poly_wire.items()]  # список (многочлен, провод, степень)
            dp = deg(p)  # степень p
            for i in range(len(items)):  # по парам делителей i < j
                pa, wa, da = items[i]  # очередной делитель и его степень
                if da == 0: continue  # константа как делитель бессмысленна: пропустить
                for j in range(i + 1, len(items)):  # по вторым делителям j > i
                    pb, wb, db = items[j]  # очередной второй делитель
                    if db == 0 or (da == 1 and db == 1 and len(pa) == 1 and len(pb) == 1): continue  # пропустить константу и пару двух разных переменных (их произведение — моном степени 2, для него есть отдельная ветка)
                    if da + db >= dp and mul(pa, pb) == p: return emit(wa, wb, 8, p)  # p = pa · pb точно (и сумма степеней не меньше deg(p)): вернуть AND двух проводов
        return None  # одним вентилем не получилось

    def build_xor_list(items: list[tuple[frozenset[int], int]]) -> int:
        """Сумма по модулю 2 списка слагаемых (многочлен, провод) с минимумом новых вентилей.
        
                    Сначала объединяются те пары, чья сумма уже на проводе (без нового вентиля),
                    иначе складываются два первых. Пустой список даёт константу 0.
                    Возвращает провод результата."""
        if not items: return get_const(0)  # пустой список — константа 0
        while len(items) > 1:  # пока слагаемых больше одного
            best = None  # лучшей пары нет
            for i in range(len(items)):  # по всем парам слагаемых i < j
                for j in range(i + 1, len(items)):  # по второму слагаемому
                    xp = xor(items[i][0], items[j][0])  # сумма xp = p_i ⊕ p_j
                    if xp in poly_wire: best = (i, j, xp, poly_wire[xp]); break  # сумма уже на проводе — бесплатная пара; запомнить и выйти
                if best: break  # выйти из внешнего цикла, если пара найдена
            if best:  # если бесплатная пара найдена
                i, j, xp, xw = best; items.pop(j); items.pop(i); items.append((xp, xw))  # заменить два слагаемых их суммой (без нового вентиля)
            else:  # иначе
                (p1, w1), (p2, w2) = items.pop(0), items.pop(0)  # взять два первых слагаемых
                xp = xor(p1, p2)  # сумма xp = p1 ⊕ p2
                items.append((xp, emit(w1, w2, 6, xp) if w1 != w2 else get_const(0)))  # если провода разные — новый XOR-вентиль, иначе p ⊕ p = 0 (константа)
        return items[0][1]  # вернуть провод единственного оставшегося слагаемого

    def build_poly(p: frozenset[int]) -> int:
        """Построить провод для многочлена p (рекурсивно).
        
                    Порядок: уже есть; константы; повторное использование одним вентилем (try_resub);
                    линейный многочлен — XOR-сумма; один моном — цепочка AND; общий множитель — один AND;
                    разложение по Давио с заменой x_a := x_b и x_a := 1 ⊕ x_b; вычитание короткого остатка;
                    иначе — выбор переменной разложения по счётчику стоимости и рекурсия.
                    Возвращает номер провода."""
        if p in poly_wire: return poly_wire[p]  # многочлен уже на проводе
        if not p: return get_const(0)  # пустой многочлен — константа 0
        if p == frozenset({0}): return get_const(1)  # моном 0 — константа 1
        rs = try_resub(p)  # попытка построить одним вентилем
        if rs is not None: return rs  # получилось — вернуть провод
        if max(t.bit_count() for t in p) <= 1:  # линейный многочлен (все мономы степени ≤ 1)
            return build_xor_list([(frozenset({0}), get_const(1)) if t == 0 else (frozenset({t}), poly_wire[frozenset({t})]) for t in sorted(p)])  # XOR-сумма: моном 0 — единица, остальные — переменные (мономы по возрастанию)
        if len(p) == 1:  # один моном степени ≥ 2: строится цепочкой AND
            t = next(iter(p)); bits = [1 << i for i in range(n) if (t >> i) & 1]  # моном t и список его переменных bits
            cur = [(frozenset({b}), poly_wire[frozenset({b})]) for b in bits]  # текущие сомножители cur — одиночные переменные с их проводами
            while len(cur) > 1:  # пока не осталось одно слагаемое
                found = None  # найденной пары нет
                for i in range(len(cur)):  # по парам сомножителей
                    for j in range(i + 1, len(cur)):  # по второму сомножителю
                        mp = frozenset({next(iter(cur[i][0])) | next(iter(cur[j][0]))})  # произведение mp = p_i · p_j (объединение мономов)
                        if mp in poly_wire: found = (i, j, mp, poly_wire[mp]); break  # произведение уже на проводе — запомнить пару
                    if found: break  # выйти, если пара найдена
                if found:  # пара найдена
                    i, j, mp, mw = found; cur.pop(j); cur.pop(i); cur.append((mp, mw))  # заменить пару её произведением (без вентиля)
                else:  # иначе
                    p1, w1 = cur.pop(0); p2, w2 = cur.pop(0)  # взять два первых сомножителя
                    mp = frozenset({next(iter(p1)) | next(iter(p2))})  # произведение mp = p1 · p2
                    cur.append((mp, emit(w1, w2, 8, mp)))  # добавить произведение через AND-вентиль
            return cur[0][1]  # вернуть провод произведения

        cm = (1 << n) - 1  # полная маска переменных: 2^n − 1
        for t in p: cm &= t  # cm = пересечение мономов (переменные, общие для всех мономов)
        if cm:  # если общий множитель есть
            b = cm & -cm  # его младшую переменную b
            return emit(poly_wire[frozenset({b})], build_poly(frozenset(t ^ b for t in p)), 8, p)  # p = b ∧ (p без b): один AND и рекурсия
        for v in range(n):  # по каждой переменной v
            vb = 1 << v  # маска vb = 2^v
            p0 = frozenset(t for t in p if not (t & vb))  # p0 — мономы без v
            p1 = frozenset(t ^ vb for t in p if (t & vb))  # p1 — мономы с v (сам бит v удалён)
            if p0 == p1 and p1: return emit(build_poly(frozenset({0, vb})), build_poly(p0), 8, p)  # если p0 == p1 и p1 непусто: p = (1 ⊕ v) ∧ p0
        for a in range(n):  # по каждой паре переменных a < b
            for b in range(a + 1, n):  # по второй переменной b
                ab, bb = 1 << a, 1 << b  # маски ab = 2^a, bb = 2^b
                sub_same: set[int] = set(); sub_inv: set[int] = set()  # подстановка x_a := x_b (sub_same) и x_a := 1 ⊕ x_b (sub_inv): множители симметрии по чётности
                for t in p:  # по каждому моном t
                    nt = (t ^ ab) | bb if (t & ab) else t  # nt = (t ⊕ ab) | bb, если в t есть x_a (замена a на b), иначе t
                    if nt in sub_same: sub_same.remove(nt)  # повторившийся моном сокращается
                    else: sub_same.add(nt)  # иначе добавить моном
                    if t & ab:  # если в t есть x_a
                        for nti in (t ^ ab, (t ^ ab) | bb):  # подстановка x_a := 1 ⊕ x_b даёт два монома: без b и с b
                            if nti in sub_inv: sub_inv.remove(nti)  # повторившийся моном сокращается
                            else: sub_inv.add(nti)  # иначе добавить
                    else:  # иначе (x_a нет в t)
                        if t in sub_inv: sub_inv.remove(t)  # повторившийся моном сокращается
                        else: sub_inv.add(t)  # иначе добавить
                if not sub_same:  # если после подстановки x_a := x_b всё сократилось
                    return emit(build_poly(frozenset({ab, bb})), build_poly(frozenset(t ^ ab for t in p if (t & ab))), 8, p)  # p = (x_a ⊕ x_b) ∧ q, где q = p при x_a := x_b
                if not sub_inv:  # если после подстановки x_a := 1 ⊕ x_b всё сократилось
                    return emit(build_poly(frozenset({0, ab, bb})), build_poly(frozenset(t ^ ab for t in p if (t & ab))), 8, p)  # p = (1 ⊕ x_a ⊕ x_b) ∧ q, где q = p при x_a := 1 ⊕ x_b

        if len(p) >= 3 and len(wire_poly) <= 64:  # вычитание имеющегося провода: p не короче 3 мономов и проводов не больше 64
            for pa, wa in list(poly_wire.items()):  # по каждому имеющемуся проводу (копия списка)
                if len(pa) <= 1 and pa != frozenset({0}): continue  # короткие pa (не больше 1 монома), кроме единицы, не дают выигрыша: пропустить
                rem = xor(p, pa)  # остаток rem = p ⊕ pa
                if len(rem) <= 2 and len(rem) < len(p) - 1: return emit(wa, build_poly(rem), 6, p)  # остаток из не более 2 мономов и короче p на 2 и более: p = pa ⊕ rem (XOR-вентиль)

        nl_vars = sorted({i for t in p if t.bit_count() >= 2 for i in range(n) if (t >> i) & 1}) or [0]  # переменные, входящие хотя бы в один нелинейный моном; если таких нет — [0]
        best_var = best_kind = best_score = None  # лучшие переменная, вид разложения и счётчик стоимости пока не выбраны
        def comp_cost(q: frozenset[int]) -> int:
            """Оценка стоимости построения многочлена q.
            
                            0 — константа или уже построен, 1 — линейный, иначе len(q) + сумма popcount(t) по мономам степени ≥ 2."""
            if not q or q == frozenset({0}) or q in poly_wire: return 0  # константа или уже построенный многочлен — бесплатно
            if max(t.bit_count() for t in q) <= 1: return 1  # линейный многочлен — один вентиль
            return len(q) + sum(t.bit_count() for t in q if t.bit_count() >= 2)  # нелинейный: len(q) + Σ popcount(t) по мономам степени ≥ 2
        for v in nl_vars:  # по каждой переменной-кандидату v
            vb = 1 << v  # маска vb = 2^v
            p0 = frozenset(t for t in p if not (t & vb))  # p0 — мономы без v
            p1 = frozenset(t ^ vb for t in p if (t & vb))  # p1 — мономы с v (бит v удалён)
            p_neg = xor(p0, p1)  # остаток по отрицательному разложению: p_neg = p0 ⊕ p1
            sc_pos = (comp_cost(p0) + comp_cost(p1), len(p0) + len(p1), v)  # счёт положительного разложения: (стоимость p0 + стоимость p1, число мономов, номер переменной)
            if best_score is None or sc_pos < best_score: best_score, best_var, best_kind = sc_pos, v, "POS"  # меньший счёт выигрывает: вид разложения POS
            c_notv = 0 if frozenset({0, vb}) in poly_wire else 1  # стоимость множителя (1 ⊕ v): 0, если он уже на проводе, иначе 1
            sc_neg = (comp_cost(p_neg) + comp_cost(p1) + c_notv, len(p_neg) + len(p1), v)  # счёт отрицательного разложения: (стоимость p_neg + стоимость p1 + стоимость (1 ⊕ v), число мономов, номер переменной)
            if sc_neg < best_score: best_score, best_var, best_kind = sc_neg, v, "NEG"  # если счёт меньше — вид разложения NEG
        vb = 1 << best_var  # маска выбранной переменной
        p0 = frozenset(t for t in p if not (t & vb))  # p0 — мономы без выбранной переменной
        p1 = frozenset(t ^ vb for t in p if (t & vb))  # p1 — мономы с ней (бит удалён)
        if best_kind == "POS":  # положительное разложение: p = p0 ⊕ (v ∧ p1)
            w_prod = emit(poly_wire[frozenset({vb})], build_poly(p1), 8, frozenset(t | vb for t in p1))  # произведение w_prod = v ∧ p1 (AND)
            return w_prod if not p0 else emit(build_poly(p0), w_prod, 6, p)  # если есть p0: p = p0 ⊕ w_prod, иначе p = w_prod
        p_neg = xor(p0, p1)  # отрицательное разложение: p_neg = p0 ⊕ p1, p = p_neg ⊕ ((1 ⊕ v) ∧ p1)
        w_prod = emit(build_poly(frozenset({0, vb})), build_poly(p1), 8, mul(frozenset({0, vb}), p1))  # произведение w_prod = (1 ⊕ v) ∧ p1
        return w_prod if not p_neg else emit(build_poly(p_neg), w_prod, 6, p)  # если есть p_neg: p = p_neg ⊕ w_prod, иначе p = w_prod

    order = sorted(range(len(outputs)), key=lambda idx: (deg(outputs[idx]), len(outputs[idx]), idx))  # порядок построения выходов: по возрастанию (степень, число мономов, номер)
    outs_map = {idx: build_poly(outputs[idx]) for idx in order}  # построить все выходы; outs_map: номер выхода -> провод
    return compact_netlist({"n_inputs": n, "gates": gates, "outs": [outs_map[i] for i in range(len(outputs))]})  # вернуть уплотнённую схему из построенных вентилей

@dataclass(frozen=True)  # декоратор: неизменяемый класс-запись (хеш и сравнение по полям)
class DegreeContract:
    """Контракт на степень: обещание, что степень АНФ не выше заданной.
    
            n_inputs: число входов, max_anf_degree: обещанная максимальная степень АНФ,
            issuer: кто выдал, version: версия контракта. Метод validate проверяет доверие к издателю и границы."""
    n_inputs: int; max_anf_degree: int; issuer: str; version: int = 1  # поля записи: число входов, предел степени, издатель, версия (по умолчанию 1)
    def validate(self, trusted_issuers: set[str]) -> tuple[bool, str]:
        """Проверить контракт: число входов ≥ 1, 0 ≤ степень ≤ n, издатель доверенный.
        
                    trusted_issuers: множество доверенных издателей. Возвращает (True, «trusted contract»)
                    или (False, причина)."""
        if self.n_inputs < 1: return False, "n_inputs must be positive"  # число входов положительно
        if not 0 <= self.max_anf_degree <= self.n_inputs: return False, "degree bound outside [0,n]"  # предел степени в отрезке [0, n]
        if self.issuer not in trusted_issuers: return False, "untrusted degree-contract issuer"  # издатель в списке доверенных
        return True, "trusted contract"  # контракт принят

@dataclass  # декоратор: класс-запись
class ANFCandidate:
    """Кандидат в навык: обученная функция и её сертификат.
    
            n_inputs, outputs: АНФ, netlist: схема, query_count: число запросов к оракулу, mode: способ получения,
            certificate: сертификат, metrics: метрики схемы."""
    n_inputs: int; outputs: tuple[frozenset[int], ...]; netlist: dict; query_count: int; mode: str; certificate: dict; metrics: dict  # поля записи: число входов, АНФ выходов, схема, число запросов, способ, сертификат, метрики
    def semantic_digest(self) -> str:
        """Хеш содержания кандидата: n и отсортированные мономы выходов -> sha256.
        
                Возвращает шестнадцатеричную строку хеша."""
        return hashlib.sha256(json.dumps({"n": self.n_inputs, "outputs": [sorted(x) for x in self.outputs]}, sort_keys=True).encode()).hexdigest()  # sha256 от JSON {"n": n_inputs, "outputs": отсортированные списки мономов} с устойчивым порядком ключей

class ProofCarryingANFLearner:
    """Обучение АНФ с доказательством: по оракулу получает многочлены и строит схему.
    
            trusted_issuers: доверенные издатели контрактов на степень,
            max_full_support_inputs: предел полного перебора всех входов (свыше — отказ)."""
    def __init__(self, trusted_issuers: Iterable[str] = (), max_full_support_inputs: int = 14):  # создать обучающего: запомнить доверенных издателей и предел полного перебора
        self.trusted_issuers = set(trusted_issuers); self.max_full_support_inputs = max_full_support_inputs  # множество доверенных издателей; предел полного перебора

    @staticmethod  # статический метод (вызывается без экземпляра)
    def subsets_upto(n: int, d: int) -> list[int]:
        """Список всех подмножеств {0..n−1} размера не больше d как битовых масок.
        
                    Возвращает [Σ 2^i по i из c] для k = 0..d и всех сочетаний c размера k."""
        return [sum(1 << i for i in c) for k in range(d + 1) for c in itertools.combinations(range(n), k)]  # маска подмножества = Σ 2^i по элементам сочетания; k от 0 до d

    @staticmethod  # статический метод
    def _interpolate(n: int, d: int, oracle: Callable[[int], tuple[int, ...]], n_outputs: int):
        """Интерполяция АНФ степени не выше d по точкам-подмножествам размера не выше d.
        
                    n: число входов, d: предел степени, oracle: функция x -> кортеж выходных битов,
                    n_outputs: число выходов. Матрица значений мономов унитреугольная, поэтому
                    коэффициент при s: a(s) = y(s) ⊕ Σ_{t ⊂ s, строго} a(t).
                    Возвращает (АНФ, labels), где labels — пары (точка s, ответ оракула)."""
        terms = [set() for _ in range(n_outputs)]; labels = []  # terms — множества мономов по выходам; labels — запросы
        for s in ProofCarryingANFLearner.subsets_upto(n, d):  # по каждой точке-подмножеству s размера ≤ d
            y = tuple(int(v) & 1 for v in oracle(s)); labels.append((s, *y))  # y = ответ оракула в точке s (биты); запомнить метку (s, *y)
            t = (s - 1) & s if s else -1; proper = []  # старт перебора собственных подмножеств: t = (s − 1) & s (для s = 0 — сразу −1)
            while t >= 0:  # пока есть подмножества
                proper.append(t)  # добавить t в список собственных подмножеств
                if t == 0: break  # после нуля подмножеств больше нет
                t = (t - 1) & s  # следующее подмножество: t = (t − 1) & s
            for o in range(n_outputs):  # по каждому выходу
                v = y[o]  # коэффициент v начинается со значения y_o(s)
                for sub in proper:  # по собственным подмножествам sub
                    if sub in terms[o]: v ^= 1  # если коэффициент при sub равен 1: v = v ⊕ 1 (вычесть член a(sub)·sub(s), где sub(s) = 1)
                if v: terms[o].add(s)  # если v = 1 — моном s входит в АНФ выхода
        return tuple(frozenset(t) for t in terms), labels  # (кортеж АНФ, метки)

    @staticmethod  # статический метод
    def _full_mobius(n: int, oracle: Callable[[int], tuple[int, ...]], n_outputs: int):
        """Полный перебор всех 2^n входов и преобразование Мёбиуса.
        
                    Возвращает (АНФ, число запросов = 2^n); при неверной арности ответа оракула — ошибка."""
        N = 1 << n; observed = [tuple(int(v) & 1 for v in oracle(x)) for x in range(N)]  # N = 2^n; observed — таблица ответов оракула по всем x
        if any(len(row) != n_outputs for row in observed): raise ValueError("oracle output arity mismatch")  # каждая строка обязана иметь n_outputs бит
        return mobius(observed, n), N  # (АНФ через mobius, число запросов N)

    @staticmethod  # статический метод
    def compile(n: int, outputs: tuple[frozenset[int], ...]) -> dict:
        """Компиляция АНФ в схему «сумма произведений».
        
                    n: число переменных, outputs: кортеж многочленов.
                    Каждый моном строится как дерево AND (деление списка переменных пополам),
                    сумма мономов — дерево XOR по куче глубин (объединяются два наименьших по глубине).
                    Возвращает схему с метриками."""
        gates: list[dict] = []; depth = [0] * n; cache: dict[int, int] = {}; const1 = const0 = None  # список вентилей; depth — глубина проводов; cache — моном -> провод; константы 1 и 0 ещё не созданы
        def add(a: int, b: int, mask: int) -> int:
            """Добавить вентиль и вернуть номер его провода; depth = 1 + max(depth(a), depth(b))."""
            w = n + len(gates); gates.append({"a": a, "b": b, "mask": mask}); depth.append(1 + max(depth[a], depth[b])); return w  # номер провода = n + число вентилей; добавить вентиль и глубину
        def monomial(s: int) -> int:
            """Провод монома s (битовая маска переменных).
            
                            Константа 1 — вентиль с маской 15; одиночная переменная — её входной провод;
                            иначе моном делится пополам и строится как AND двух половин."""
            nonlocal const1  # изменяемая переменная внешней функции (константа 1)
            if s in cache: return cache[s]  # моном уже построен — вернуть провод
            if s == 0:  # пустая маска — константа 1
                if const1 is None: const1 = add(0, 0, 15)  # если константа 1 ещё не создана — создать вентиль
                cache[s] = const1; return const1  # запомнить и вернуть провод константы 1
            if s & (s - 1) == 0: w = s.bit_length() - 1; cache[s] = w; return w  # одиночная переменная (s — степень двойки): провод = номер бита = log2 s
            bits = [1 << i for i in range(n) if (s >> i) & 1]; mid = len(bits) // 2  # список переменных монома; точка деления пополам
            w = add(monomial(sum(bits[:mid])), monomial(sum(bits[mid:])), 8); cache[s] = w; return w  # моном = AND(левая половина, правая половина); запомнить
        outs = []  # выходные провода
        for out_terms in outputs:  # по каждому выходному многочлену
            leaves = [monomial(s) for s in sorted(out_terms)]  # провода его мономов (по возрастанию)
            if not leaves:  # мономов нет — константа 0
                if const0 is None: const0 = add(0, 0, 0)  # если константа 0 ещё не создана — создать вентиль
                outs.append(const0)  # выход — константа 0
            elif len(leaves) == 1: outs.append(leaves[0])  # один моном — он и есть выход
            else:  # иначе — дерево XOR
                heap = [(depth[w], w) for w in leaves]; heapq.heapify(heap)  # куча (глубина, провод) листьев
                while len(heap) > 1:  # пока есть что объединять
                    da, a = heapq.heappop(heap); db, b = heapq.heappop(heap)  # извлечь два элемента наименьшей глубины
                    heapq.heappush(heap, (1 + max(da, db), add(a, b, 6)))  # объединить XOR-вентилем: глубина = 1 + max(da, db)
                outs.append(heap[0][1])  # выход — корень дерева
        return _finalize_net(n, gates, outs)  # схема с метриками

    @staticmethod  # статический метод
    def compile_optimal(n: int, outputs: tuple[frozenset[int], ...]) -> dict:
        """Компиляция с выбором наименьшей схемы среди проверенных кандидатов.
        
                    n: число переменных, outputs: кортеж многочленов. Кандидаты: схема «сумма произведений»
                    после уплотнения и (при n ≤ 12) оба варианта синтеза с разложением на множители,
                    прошедшие символьную проверку. Выбирается минимум по (число вентилей, глубина, биты программы).
                    При n > 12 возвращается только первый кандидат."""
        sop = compact_netlist(ProofCarryingANFLearner.compile(n, outputs))  # первый кандидат: сумма произведений, уплотнённая
        if n <= 12:  # перебор кандидатов только для небольших n
            cands = [sop]  # список кандидатов начинается с суммы произведений
            # оба варианта синтеза с разложением проходят проверку и остаются кандидатами
            # минимум берётся по ним всем, поэтому результат может только улучшиться
            for kwargs in ({}, {"xor_quad_resub": True}):  # по двум наборам параметров: без флага и с xor_quad_resub
                fac = compile_factored_gf2(n, outputs, **kwargs)  # схема-кандидат из синтеза с разложением
                if symbolic_verify_general(fac, outputs): cands.append(fac)  # в кандидат попадает только схема, равная ожидаемым многочленам
            return min(cands, key=lambda nt: (len(nt["gates"]), nt["metrics"]["depth"], nt["metrics"]["program_bits"]))  # вернуть минимум по (число вентилей, глубина, биты программы)
        return sop  # при большом n — только сумма произведений

    @staticmethod  # статический метод
    def symbolic_verify(netlist: dict, expected: tuple[frozenset[int], ...]) -> bool:
        """Символьная проверка схемы: равны ли её многочлены ожидаемым.
        
                    netlist: схема, expected: кортеж многочленов. Возвращает True или False."""
        return symbolic_verify_general(netlist, expected)  # True, если многочлены выходов совпали

    def fit(self, n_inputs: int, n_outputs: int, oracle: Callable[[int], tuple[int, ...]],  # обучение по оракулу; описание в docstring после сигнатуры
            contract: DegreeContract | None = None, allow_full_support: bool = True,  # продолжение сигнатуры: контракт и разрешения
            audit_full_support: bool = False) -> tuple[ANFCandidate | None, dict]:  # конец сигнатуры: тип возврата
        if n_inputs < 1 or n_outputs < 1: raise ValueError("input/output arity must be positive")  # ошибка, если входов или выходов меньше одного
        if contract is not None:  # контракт задан — обучение по контракту
            valid, reason = contract.validate(self.trusted_issuers)  # проверка контракта: valid — доверие и границы, reason — причина отказа
            if valid and contract.n_inputs == n_inputs:  # контракт действителен и число входов совпадает
                d = contract.max_anf_degree  # предел степени d из контракта
                outputs, labels = self._interpolate(n_inputs, d, oracle, n_outputs)  # интерполяция АНФ по d; labels — использованные точки
                net = self.compile_optimal(n_inputs, outputs); basis = [x[0] for x in labels]  # схема и список точек базиса
                cert = {"kind": "trusted-degree-unisolvent", "issuer": contract.issuer, "contract_version": contract.version,  # сертификат: вид, издатель, версия контракта, предел степени, размер базиса
                        "max_anf_degree": d, "basis_size": len(basis),  # продолжение словаря сертификата
                        "basis_points_sha256": hashlib.sha256(",".join(map(str, basis)).encode()).hexdigest(),  # sha256 от номеров точек базиса через запятую
                        "labels_sha256": hashlib.sha256(bytes(v for row in labels for v in row[1:])).hexdigest(), "unit_triangular": True}  # sha256 от байтов меток; матрица унитреугольная
                if not self.symbolic_verify(net, outputs): return None, {"action": "HOLD", "reason": "symbolic netlist verification failed"}  # символьная проверка схемы не пройдена — отказ
                audit_checks = 0  # счётчик проверенных точек
                if audit_full_support and n_inputs <= self.max_full_support_inputs:  # если включён аудит и n входит в предел полного перебора
                    mismatch = False  # признак расхождения с оракулом
                    for x in range(1 << n_inputs):  # по всем 2^n входам
                        if tuple(ev(terms, x) for terms in outputs) != tuple(int(v) & 1 for v in oracle(x)):  # значение АНФ в точке x не совпало с оракулом
                            mismatch = True; break  # отметить расхождение и выйти
                        audit_checks += 1  # учёт проверенной точки
                    if mismatch:  # если найдено расхождение
                        if not allow_full_support: return None, {"action": "HOLD", "reason": "degree contract failed full-support audit", "audit_checks": audit_checks}  # при запрете полного перебора — отказ: контракт не прошёл аудит
                        full_outputs, q = self._full_mobius(n_inputs, oracle, n_outputs)  # полный перебор: уточнённая АНФ и число запросов q
                        full_net = self.compile_optimal(n_inputs, full_outputs)  # схема для уточнённой АНФ
                        if not self.symbolic_verify(full_net, full_outputs): return None, {"action": "HOLD", "reason": "fallback symbolic verification failed"}  # символьная проверка не пройдена — отказ
                        # кандидат с уточнённой АНФ (режим full-support-escalated)
                        return ANFCandidate(n_inputs, full_outputs, full_net, q, "full-support-escalated",  # вернуть кандидата в режиме полного перебора
                                            # его сертификат: полный перебор, число запросов
                                            {"kind": "full-support-mobius", "full_support": True, "queries": q}, full_net["metrics"]), \
                               {"action": "COMMIT_CANDIDATE", "mode": "full-support-escalated", "query_count": q, "support_size": q, "degree_contract_rejected": True}  # решение COMMIT_CANDIDATE: режим, число запросов, размер опоры, контракт отклонён
                cert["audit_full_support"] = bool(audit_full_support and n_inputs <= self.max_full_support_inputs)  # сертификат: аудит проведён (в пределах полного перебора)
                cert["audit_rows_checked"] = audit_checks  # сертификат: число проверенных строк
                # кандидат по контракту (режим degree-contract)
                return ANFCandidate(n_inputs, outputs, net, len(labels), "degree-contract", cert, net["metrics"]), \
                       {"action": "COMMIT_CANDIDATE", "mode": "degree-contract", "query_count": len(labels), "support_size": 1 << n_inputs, "audit_rows_checked": audit_checks}  # решение COMMIT_CANDIDATE: режим, число запросов, размер опоры, проверенные строки
            reason = reason if not valid else "contract input arity mismatch"  # причина отказа: прежняя (контракт недействителен) или несовпадение арности входов
        else:  # иначе (контракта нет)
            reason = "no degree contract"  # причина — нет контракта на степень
        if not allow_full_support or n_inputs > self.max_full_support_inputs:  # полный перебор запрещён или n слишком велико
            return None, {"action": "HOLD", "reason": reason + "; exact full-support fallback unavailable"}  # отказ: причина + недоступность точного полного перебора
        outputs, q = self._full_mobius(n_inputs, oracle, n_outputs)  # полный перебор: АНФ и число запросов
        net = self.compile_optimal(n_inputs, outputs)  # схема для этой АНФ
        if not self.symbolic_verify(net, outputs): return None, {"action": "HOLD", "reason": "full-support symbolic netlist verification failed"}  # символьная проверка не пройдена — отказ
        # кандидат с полным перебором (режим full-support), сертификат
        return ANFCandidate(n_inputs, outputs, net, q, "full-support", {"kind": "full-support-mobius", "full_support": True, "queries": q}, net["metrics"]), \
               {"action": "COMMIT_CANDIDATE", "mode": "full-support", "query_count": q, "support_size": q}  # решение COMMIT_CANDIDATE: режим, число запросов, размер опоры

def surgical_repair_netlist(corrupted_net: dict, expected_anf: tuple[frozenset[int], ...],  # хирургическая починка схемы; описание в docstring после сигнатуры
                            oracle_rows: list[tuple[int, ...]], learner: ProofCarryingANFLearner,  # продолжение сигнатуры
                            issuer: str = DEFAULT_ISSUER) -> tuple[dict, list[int], int]:  # конец сигнатуры; описание в docstring ниже
    n = corrupted_net["n_inputs"]; got_anf = symbolic_eval_net_general(corrupted_net)  # число входов; АНФ выходов испорченной схемы
    bad = [idx for idx, (g, e) in enumerate(zip(got_anf, expected_anf)) if g != e]  # номера выходов, где АНФ испорченной схемы не совпала с ожидаемой
    if not bad: return corrupted_net, [], 0  # расхождений нет — вернуть исходную схему, ничего не делая
    cand, dec = learner.fit(n, len(bad), lambda x: tuple(oracle_rows[x][idx] for idx in bad),  # переобучить только испорченные выходы: оракул отдаёт их столбцы строки x
                            contract=DegreeContract(n, max(deg(expected_anf[idx]) for idx in bad), issuer, 2),  # контракт: степень не выше максимума степеней ожидаемой АНФ испорченных выходов
                            allow_full_support=True, audit_full_support=False)  # полный перебор разрешён, аудит выключен
    if cand is None: raise RuntimeError(f"surgical repair failed: {dec}")  # обучение не удалось — ошибка с причиной
    new_gates = [dict(g) for g in corrupted_net["gates"]]; new_outs = list(corrupted_net["outs"])  # копии вентилей и выходов испорченной схемы
    offset = n + len(new_gates); rep_net = cand.netlist  # смещение для врезки = n + число вентилей; rep_net — схема-заплатка
    for g in rep_net["gates"]:  # по каждому вентилю заплатки
        new_gates.append({"a": g["a"] if g["a"] < n else g["a"] - n + offset,  # вход a: входной провод остаётся, внутренний сдвигается на offset − n
                          "b": g["b"] if g["b"] < n else g["b"] - n + offset, "mask": g["mask"]})  # вход b (то же правило) и маска
    for pos, out_idx in enumerate(bad):  # по каждому испорченному выходу (pos — его номер в заплатке)
        ro = rep_net["outs"][pos]; new_outs[out_idx] = ro if ro < n else ro - n + offset  # заменить выход проводом заплатки (внутренние провода сдвигаются)
    repaired = compact_netlist({"n_inputs": n, "gates": new_gates, "outs": new_outs})  # исправленная схема после уплотнения
    if not symbolic_verify_general(repaired, expected_anf): raise RuntimeError("surgical repair verification failed")  # символьная проверка обязательна
    return repaired, bad, cand.query_count  # вернуть схему, список исправленных выходов и число запросов


# ---------------------------------------------------------------------------
# 3. Минимизация автомата по Хопкрофту, значения «безразлично» над GF(2) и кодирование состояний
# ---------------------------------------------------------------------------

def quotient_to_rows(k: int, ni: int, no: int, T) -> tuple[int, list[tuple[int, ...]]]:
    """Таблица переходов автомата Мили -> таблица строк для компиляции.
    
            k: число состояний, ni: число входных битов, no: число выходных битов, T: таблица переходов T[c][u] = (следующее состояние, выход).
            Строка x = s | (u << ns_min): биты следующего состояния (ns_min = ceil(log2 k)) и выход.
            Для кодов состояний ≥ k (не используются) переход в 0 с нулевым выходом.
            Возвращает (ns_min, rows)."""
    ns_min = 0 if k <= 1 else (k - 1).bit_length(); rows = []  # ns_min = 0 при k ≤ 1, иначе ceil(log2 k) = bit_length(k − 1)
    for x in range(1 << (ns_min + ni)):  # по всем x из 2^(ns_min + ni)
        c = x & ((1 << ns_min) - 1) if ns_min > 0 else 0; u = x >> ns_min  # c — код состояния (младшие ns_min бит, если ns_min > 0, иначе 0); u — вход (старшие биты)
        nxt_c, out_u = T[c][u] if c < k else (0, (0,) * no)  # переход и выход; для неиспользуемого кода — (0, нули)
        rows.append(tuple([(nxt_c >> i) & 1 for i in range(ns_min)] + list(out_u)))  # строка: биты следующего состояния nxt_c (младшие ns_min) + биты выхода
    return ns_min, rows  # вернуть (ns_min, строки)

def minimize_quotient_table(k_raw: int, ni: int, no: int, raw_T, reset: int = 0):
    """Минимизация таблицы переходов: классы эквивалентности состояний и фактор-таблица.
    
            k_raw: число состояний исходной таблицы, raw_T[s][u] = (next, out), reset: начальное состояние.
            1) обходом в ширину от reset находятся достижимые состояния;
            2) начальное разбиение — по вектору выходов; далее уточнение: состояния в одном классе, если
            их классы совпадают и для каждого u классы преемников совпадают, — до стабилизации;
            3) каноническая нумерация классов обходом в ширину от класса reset (класс reset = 0).
            Возвращает (k, хеш, T, phi, reach): число классов, хеш таблицы, фактор-таблица, отображение
            состояние -> класс, список достижимых состояний."""
    seen = {reset}; q = deque([reset])  # просмотренные состояния и очередь обхода с начального
    while q:  # пока очередь не пуста
        s = q.popleft()  # взять состояние s
        for u in range(1 << ni):  # по всем входам
            nxt, _ = raw_T[s][u]  # переход из s по u
            if nxt not in seen: seen.add(nxt); q.append(nxt)  # новое состояние добавить в просмотренные и в очередь
    reach = sorted(seen); cls: dict[int, int] = {}; sigs: dict = {}  # reach — достижимые состояния; cls — состояние -> класс; sigs — сигнатура -> номер класса
    for s in reach: cls[s] = sigs.setdefault(tuple(raw_T[s][u][1] for u in range(1 << ni)), len(sigs))  # начальное разбиение: сигнатура состояния — вектор его выходов по всем u
    while True:  # уточнение разбиения до стабилизации
        nsigs: dict = {}; ncls: dict[int, int] = {}  # новые сигнатуры и классы
        for s in reach: ncls[s] = nsigs.setdefault((cls[s], tuple(cls[raw_T[s][u][0]] for u in range(1 << ni))), len(nsigs))  # новая сигнатура: (класс s, классы преемников по всем u)
        if len(nsigs) == len(sigs): break  # число классов не изменилось — разбиение стабильно
        cls, sigs = ncls, nsigs  # принять новое разбиение
    canon_id = {cls[reset]: 0}; rep = {0: reset}; bq = deque([reset])  # каноническая нумерация: класс reset получает номер 0; rep — представитель класса
    while bq:  # обход в ширину от reset
        s = bq.popleft()  # взять состояние s
        for u in range(1 << ni):  # по всем входам
            nxt, _ = raw_T[s][u]; nc = cls[nxt]  # преемник nxt и его класс nc
            if nc not in canon_id: cid = len(canon_id); canon_id[nc] = cid; rep[cid] = nxt; bq.append(nxt)  # новый класс получает очередной номер, представитель — nxt
    k = len(canon_id); phi = {s: canon_id[cls[s]] for s in reach}  # k — число классов; phi: состояние -> канонический номер класса
    T = tuple(tuple((phi[raw_T[rep[c]][u][0]], raw_T[rep[c]][u][1]) for u in range(1 << ni)) for c in range(k))  # фактор-таблица: для представителя класса c — переход в класс преемника и выход
    return k, digest({"ni": ni, "no": no, "k": k, "T": [[list(x[1]) + [x[0]] for x in row] for row in T]}), T, phi, reach  # вернуть (k, хеш таблицы, T, phi, reach)

def minimize_mealy_from_rows(ns: int, ni: int, no: int, rows: list[tuple[int, ...]], reset: int = 0):
    """Минимизация автомата Мили по таблице строк с проверкой бисимуляции.
    
            ns, ni, no: разрядности, rows: таблица строк (биты следующего состояния, затем выход), reset: начальное состояние.
            Строится raw_T[s][u] = (следующее состояние, выход), вызывается minimize_quotient_table,
            затем для каждой достижимой пары (s, u) проверяется: выход совпадает, а класс преемника
            совпадает с преемником в фактор-таблице. При расхождении — AssertionError.
            Возвращает (ns_min, k, хеш, T, строки фактор-автомата, число достижимых состояний)."""
    raw_T = tuple(tuple((sum(v << i for i, v in enumerate(rows[s | (u << ns)][:ns])), tuple(rows[s | (u << ns)][ns:])) for u in range(1 << ni)) for s in range(1 << ns))  # исходная таблица переходов: (Σ v_i·2^i по битам состояния, кортеж выходов) для всех s, u
    k, bdig, T, phi, reach = minimize_quotient_table(1 << ns, ni, no, raw_T, reset)  # минимизация; bdig — хеш фактор-таблицы
    ns_min, min_rows = quotient_to_rows(k, ni, no, T)  # строки фактор-автомата
    for s in reach:  # по каждому достижимому состоянию s
        c = phi[s]  # его класс c
        for u in range(1 << ni):  # по всем входам
            ro = rows[s | (u << ns)]; rm = min_rows[c | (u << ns_min)]  # ro — строка исходной таблицы, rm — строка фактор-автомата для (c, u)
            if tuple(ro[ns:]) != tuple(rm[ns_min:]) or phi[sum(v << i for i, v in enumerate(ro[:ns]))] != sum(v << i for i, v in enumerate(rm[:ns_min])):  # выходы различаются или класс преемника не равен преемнику в фактор-таблице
                raise AssertionError("bisimulation verification failed")  # ошибка: бисимуляция не подтверждена
    return ns_min, k, bdig, T, min_rows, len(reach)  # вернуть результаты

def minimize_dont_care_rows_f2(ns: int, ni: int, no: int, rows: list[tuple[int, ...]], care_states: set[int]):
    """Упрощение таблицы на недостижимых кодах состояний (значения «безразлично») над GF(2).
    
            ns, ni, no: разрядности, rows: таблица строк, care_states: достижимые коды состояний.
            На недостижимых строках dc_xs значения можно менять свободно. Строится пространство
            допустимых изменений (в базисе мономов), оно приводится к ступенчатому виду (RREF) с
            запоминанием комбинаций недостижимых строк; затем для каждого выходного столбца АНФ
            очищаются старшие мономы, для которых это возможно, переворотом значений на выбранных
            недостижимых строках. Возвращает (новые строки, их АНФ). При n > 10 или отсутствии
            недостижимых строк таблица не меняется."""
    n = ns + ni; N = 1 << n; ncols = ns + no  # n — всего переменных; N = 2^n; ncols — число столбцов (биты состояния + выхода)
    dc_xs = [s | (u << ns) for s in range(1 << ns) if s not in care_states for u in range(1 << ni)]  # dc_xs — недостижимые строки x = s | (u << ns) с кодом состояния s вне care_states
    anf_base = mobius(rows, n)  # исходная АНФ
    if not dc_xs or n > 10: return rows, anf_base  # недостижимых строк нет или слишком много переменных — вернуть как есть
    monomials = sorted(range(N), key=lambda t: (t.bit_count(), t), reverse=True)  # мономы по убыванию (степень, номер)
    mono_pos = {t: idx for idx, t in enumerate(monomials)}; pivots = []  # позиция монома в списке; ступенчатый базис (старший бит, вектор, комбинация недостижимых строк)
    for idx_d, d in enumerate(dc_xs):  # по каждой недостижимой строке d
        vec = 0  # вектор-строка vec (битовая маска длины N)
        for pos, t in enumerate(monomials):  # по мономам в порядке списка
            if (d & t) == d: vec |= 1 << (N - 1 - pos)  # моном t ⊆ d: в позиции t ставится 1 (моном равен 1 в точке d)
        dc_combo = 1 << idx_d; cur = vec  # dc_combo — какая недостижимая строка использована (бит idx_d); cur — текущая строка
        for p_bit, p_vec, p_combo in pivots:  # по уже найденным опорным векторам
            if (cur >> p_bit) & 1: cur ^= p_vec; dc_combo ^= p_combo  # если бит старшего элемента установлен — вычесть опорную строку (cur ⊕= p_vec; комбинация ⊕= p_combo)
        if cur: pivots.append((cur.bit_length() - 1, cur, dc_combo)); pivots.sort(key=lambda x: -x[0])  # ненулевая строка — новая опорная; список сортируется по убыванию старшего бита
    new_rows = [list(r) for r in rows]  # копия строк для изменений
    for j in range(ncols):  # по каждому столбцу j
        avec = 0  # avec — вектор коэффициентов АНФ столбца j (бит = позиция монома)
        for t in anf_base[j]: avec |= 1 << (N - 1 - mono_pos[t])  # моном t из АНФ столбца j: бит в позиции t
        dc_used = 0  # какие недостижимые строки перевернуть, чтобы очистить старшие мономы
        for p_bit, p_vec, p_combo in pivots:  # по опорным векторам
            if (avec >> p_bit) & 1: avec ^= p_vec; dc_used ^= p_combo  # если старший бит установлен — вычесть опорную строку
        for idx_d, d in enumerate(dc_xs):  # по недостижимым строкам
            if (dc_used >> idx_d) & 1: new_rows[d][j] ^= 1  # если строка d входит в комбинацию — перевернуть бит столбца j в строке d
    fin_rows = [tuple(r) for r in new_rows]  # новые строки
    return fin_rows, mobius(fin_rows, n)  # вернуть (новые строки, их АНФ)

@functools.lru_cache(maxsize=256)  # кэш на 256 наборов аргументов
def optimal_encode_quotient(k: int, ni: int, no: int, T) -> tuple[int, list[tuple[int, ...]], tuple[frozenset[int], ...], dict]:  # поиск кодов состояний; описание в docstring ниже
    """Оптимальное кодирование состояний фактор-автомата: выбор кодов π(c) и компиляция.
    
            k: число состояний, ni, no: разрядности, T: фактор-таблица. Требуется π(0) = 0.
            При малом числе кодов перебираются все варианты, иначе — эвристика: последовательный
            и код Грея как начало, затем локальные обмены и замены кодов; лучшие кандидаты
            доуточняются упрощением на недостижимых кодах. Выбирается вариант с наименьшей схемой.
            Возвращает (ns, строки, АНФ, схема)."""
    ns = 0 if k <= 1 else (k - 1).bit_length()  # ns = 0 при k ≤ 1, иначе ceil(log2 k)
    if k <= 1 or ns + ni > 10:  # тривиальный случай или слишком много переменных — без поиска кодов
        ns_m, rows = quotient_to_rows(k, ni, no, T); anf = mobius(rows, ns_m + ni)  # строки и АНФ для естественной кодировки
        return ns_m, rows, anf, ProofCarryingANFLearner.compile_optimal(ns_m + ni, anf)  # вернуть (ns_m, строки, АНФ, схема)
    has_dc = k < (1 << ns)  # есть неиспользуемые коды состояний (значения «безразлично»)

    def eval_perm(perm_codes: tuple[int, ...], refine_dc: bool = False):
        """Оценить вариант кодирования perm_codes: построить строки и (если refine_dc) упростить их.
        
                        Возвращает ((число мономов, сумма степеней), строки, АНФ)."""
        rows = [(0,) * (ns + no)] * (1 << (ns + ni)); care = set(perm_codes)  # таблица по умолчанию — нули; достижимые коды — использованные
        for c in range(k):  # по каждому состоянию c
            sc = perm_codes[c]  # его код sc
            for u in range(1 << ni):  # по всем входам
                nc, out_u = T[c][u]; nsc = perm_codes[nc]  # переход и выход; код преемника
                rows[sc | (u << ns)] = tuple([(nsc >> i) & 1 for i in range(ns)] + list(out_u))  # строка для (sc, u): биты кода преемника + выход
        opt_rows, opt_anf = minimize_dont_care_rows_f2(ns, ni, no, rows, care) if (has_dc and refine_dc) else (rows, mobius(rows, ns + ni))  # упрощение на недостижимых кодах, если разрешено; иначе просто АНФ
        return (sum(len(p) for p in opt_anf), sum(deg(p) for p in opt_anf)), opt_rows, opt_anf  # (счёт (число мономов, сумма степеней), строки, АНФ)

    if k <= 4 and ns <= 2:  # маленький случай: полный перебор кодов
        top = sorted([(eval_perm((0,) + p, True), (0,) + p) for p in itertools.permutations(range(1, 1 << ns), k - 1)], key=lambda x: x[0][0])[:3]  # все перестановки кодов 1..2^ns−1 для состояний 1..k−1 (код 0 закреплён), берутся 3 лучших по счёту
        cands = [(sc, codes, r, a) for (sc, r, a), codes in top]  # кандидаты: (счёт, коды, строки, АНФ)
    else:  # иначе — эвристика
        gray_seq = [i ^ (i >> 1) for i in range(1 << ns)]  # код Грея: g_i = i ⊕ (i >> 1)
        seeds = [tuple(range(k)), tuple([0] + [g for g in gray_seq if g != 0][:k - 1])]  # два начальных варианта: натуральные коды и коды Грея
        best_cands = []  # накопитель лучших вариантов
        for init_codes in seeds:  # по каждому начальному варианту
            cur = list(init_codes); cur_sc, cur_r, cur_a = eval_perm(tuple(cur), True)  # текущие коды и их оценка с упрощением
            best_cands.append((cur_sc, tuple(cur), cur_r, cur_a))  # добавить в кандидаты
            if k <= 12:  # локальный поиск только при небольшом k
                used = set(cur); unused = [c for c in range(1, 1 << ns) if c not in used]  # used — занятые коды; unused — свободные коды (кроме 0)
                for i in range(1, k):  # по каждому состоянию i (кроме 0-го)
                    for j in range(i + 1, k):  # обмен кодами состояний i и j
                        cur[i], cur[j] = cur[j], cur[i]; sc, r, a = eval_perm(tuple(cur), False)  # оценка после обмена (без упрощения)
                        if sc < cur_sc: cur_sc, cur_r, cur_a = sc, r, a; best_cands.append((cur_sc, tuple(cur), cur_r, cur_a))  # улучшение — принять обмен и запомнить вариант
                        else: cur[i], cur[j] = cur[j], cur[i]  # иначе вернуть коды обратно
                    for idx_u, uc in enumerate(list(unused)):  # замена кода состояния i свободным кодом uc
                        old = cur[i]; cur[i] = uc; sc, r, a = eval_perm(tuple(cur), False)  # старый код; оценка после замены
                        if sc < cur_sc: cur_sc, cur_r, cur_a = sc, r, a; unused[idx_u] = old; best_cands.append((cur_sc, tuple(cur), cur_r, cur_a))  # улучшение — принять замену (освободившийся код идёт в unused)
                        else: cur[i] = old  # иначе вернуть старый код
        best_cands.sort(key=lambda x: x[0]); cands = []; seen_c = set()  # сортировка кандидатов по счёту
        for item in best_cands:  # по кандидатам
            if item[1] not in seen_c:  # только новые варианты кодов
                seen_c.add(item[1]); sc, r, a = eval_perm(item[1], True); cands.append((sc, item[1], r, a))  # оценка с упрощением; добавить в кандидаты
            if len(cands) >= 3: break  # не более 3 кандидатов

    best_net = best_rows = best_anf = None  # лучший результат пока не найден
    for _, _, r, a in cands:  # по кандидатам
        net = ProofCarryingANFLearner.compile_optimal(ns + ni, a)  # схема для АНФ кандидата
        if best_net is None or (len(net["gates"]), net["metrics"]["depth"], net["metrics"]["program_bits"]) < (len(best_net["gates"]), best_net["metrics"]["depth"], best_net["metrics"]["program_bits"]):  # кандидат лучше по (вентили, глубина, биты)
            best_net, best_rows, best_anf = net, r, a  # запомнить лучший
    return ns, best_rows, best_anf, best_net  # вернуть (ns, строки, АНФ, схема) лучшего варианта


# ---------------------------------------------------------------------------
# 4. Навык, алгебра типизированных композиций, глобальная схема TemporalDAG, модель себя, библиотека
# ---------------------------------------------------------------------------

@dataclass  # декоратор: класс-запись с полями
class Skill:
    """Навык: последовательностная функция (автомат Мили) в виде схемы и её описание.
    
            name: имя; ns, ni, no: разрядности состояния, входа и выхода; reset: начальное состояние;
            rows: таблица строк (биты следующего состояния, затем выход) или None; anf: АНФ таблицы или None;
            net: схема; route: откуда взялся навык; op: операция композиции или PRIMITIVE;
            parents: имена родителей; queries: число запросов к оракулу; compile_mode: способ компиляции;
            k_states: число состояний фактор-автомата; ns_min: минимальная разрядность состояния;
            behav_digest: хеш поведения; quotient_T: фактор-таблица переходов; bisim_verified: бисимуляция проверена;
            ag_cert: сертификат модульной композиции; generation: номер поколения."""
    name: str; ns: int; ni: int; no: int; reset: int  # поля: имя, разрядности состояния, входа, выхода, начальное состояние
    rows: list[tuple[int, ...]] | None; anf: tuple[frozenset[int], ...] | None; net: dict; route: str; op: str  # таблица строк и АНФ (None, если навык без полного аудита), схема, маршрут, операция
    parents: tuple[str, ...] = (); queries: int = 0; compile_mode: str = "FLAT"; k_states: int = 1; ns_min: int = 0  # родители, число запросов, способ компиляции (по умолчанию FLAT), число состояний, минимальная разрядность
    behav_digest: str = ""; quotient_T: tuple | None = None; bisim_verified: bool = True; ag_cert: dict | None = None; generation: int = 0  # хеш поведения, фактор-таблица, признак проверки бисимуляции, сертификат композиции, поколение

    def step(self, s: int, u: int, network: bool = False) -> tuple[int, tuple[int, ...]]:
        """Один шаг автомата: по состоянию s и входу u вернуть следующее состояние и выход.
        
                    x = (s mod 2^ns) | ((u mod 2^ni) << ns) — номер строки таблицы.
                    Ответ берётся из таблицы rows[x], а если её нет или network=True — вычисляется схемой.
                    Возвращает (следующее состояние = Σ v_i·2^i по битам состояния, кортеж выходов)."""
        x = (s & ((1 << self.ns) - 1)) | ((u & ((1 << self.ni) - 1)) << self.ns)  # номер строки x = (s mod 2^ns) + (u mod 2^ni)·2^ns
        y = eval_net(self.net, tuple((x >> i) & 1 for i in range(self.ns + self.ni))) if (network or self.rows is None) else self.rows[x]  # ответ: из схемы (если network=True или таблицы нет) или из строки x таблицы
        return sum(v << i for i, v in enumerate(y[:self.ns])), tuple(y[self.ns:])  # следующее состояние = Σ v_i·2^i по первым ns битам; выход — остальные биты

    @property  # свойство только для чтения
    def nvars(self) -> int: return self.ns + self.ni  # Число переменных функции перехода: nvars = ns + ni (биты состояния и входа).

    def semantic_digest(self) -> str:
        """Хеш содержания навыка.
        
                Если есть АНФ — хеш разрядностей и многочленов, иначе — разрядностей и схемы.
                Возвращает строку хеша."""
        if self.anf is not None: return digest({"ns": self.ns, "ni": self.ni, "no": self.no, "anf": [sorted(p) for p in self.anf]})  # с АНФ: хеш {ns, ni, no, anf — отсортированные мономы}
        return digest({"ns": self.ns, "ni": self.ni, "no": self.no, "ag_net": self.net})  # без АНФ: хеш {ns, ni, no, ag_net — схема}

    def export(self) -> dict:
        """Экспорт навыка в словарь для сохранения в JSON.
        
                Таблица, АНФ и фактор-таблица превращаются в списки; добавляется вычисленный хеш.
                Возвращает словарь."""
        return {"name": self.name, "ns": self.ns, "ni": self.ni, "no": self.no, "reset": self.reset,  # имя, разрядности, начальное состояние
                "rows": [list(r) for r in self.rows] if self.rows is not None else None,  # таблица строк списками (или None)
                "anf": [sorted(p) for p in self.anf] if self.anf is not None else None,  # АНФ — отсортированные списки мономов (или None)
                "net": self.net, "route": self.route, "op": self.op, "parents": list(self.parents),  # схема, маршрут, операция, список родителей
                "queries": self.queries, "compile_mode": self.compile_mode, "k_states": self.k_states, "ns_min": self.ns_min,  # число запросов, способ компиляции, число состояний, минимальная разрядность
                "behav_digest": self.behav_digest,  # хеш поведения
                "quotient_T": [[[x[0], list(x[1])] for x in row] for row in self.quotient_T] if self.quotient_T is not None else None,  # фактор-таблица: для каждой пары (состояние, вход) — [следующее состояние, список выхода] (или None)
                "bisim_verified": self.bisim_verified, "ag_cert": self.ag_cert, "generation": self.generation, "digest": self.semantic_digest()}  # признак бисимуляции, сертификат, поколение, хеш содержания

def lut4_anf(lut4: int) -> frozenset[int]: return mobius([((lut4 >> x) & 1,) for x in range(4)], 2)[0]  # АНФ двуместной логической функции по её 4-битной таблице истинности lut4.
def lut4_name(lut4: int) -> str:
    """Имя логической функции по 4-битной таблице истинности.
    
            Для известных таблиц — мнемоническое имя, иначе двузначный номер. Возвращает строку."""
    return f"LUT_{ {0:'CONST0',1:'NOR',2:'NOT_A_AND_B',4:'A_AND_NOT_B',6:'XOR',8:'AND',9:'XNOR',11:'B_IMPLIES_A',13:'A_IMPLIES_B',14:'OR',15:'CONST1'}.get(lut4, f'{lut4:02d}') }"  # имя из словаря известных таблиц (ключ — lut4), иначе номер в десятичном виде с ведущим нулём
def lut4_eval(lut4: int, ya: int, yb: int) -> int: return (lut4 >> (ya | (yb << 1))) & 1  # Вычислить значение двуместной функции: (lut4 >> (ya | (yb << 1))) & 1.

def _emit_subcircuit(net: dict, wire_map: list[int], n_total: int, gates: list[dict]) -> list[int]:
    """Встроить подсхему в общую схему: переписать номера её проводов.
    
            net: подсхема, wire_map: её входные провода -> провода общей схемы, n_total: число входов общей схемы,
            gates: общий список вентилей (пополняется). Возвращает список проводов выходов подсхемы."""
    w = list(wire_map)  # провода подсхемы: входы через карту соответствия
    for g in net["gates"]:  # по каждому вентилю подсхемы
        nw = n_total + len(gates); gates.append({"a": w[g["a"]], "b": w[g["b"]], "mask": g["mask"]}); w.append(nw)  # новый провод = n_total + число уже добавленных вентилей; входы через карту; запомнить провод
    return [w[o] for o in net["outs"]]  # выходы подсхемы

def compose_net_cascade(net_a: dict, ns_a: int, ni_a: int, net_b: dict, ns_b: int, ni_b: int) -> dict:
    """Каскад: выходы A идут на вход B — последовательное соединение.
    
            Входы общей схемы: состояние A (ns_a), состояние B (ns_b), вход A (ni_a).
            Выход A (кроме нового состояния A) подаётся на вход B. Возвращает схему с выходами:
            новое состояние A + выходы B."""
    n = ns_a + ns_b + ni_a; gates: list[dict] = []  # число входов n = ns_a + ns_b + ni_a; общий список вентилей
    oa = _emit_subcircuit(net_a, list(range(ns_a)) + list(range(ns_a + ns_b, n)), n, gates)  # A: входы — состояние A и вход A; выходы oa
    ob = _emit_subcircuit(net_b, list(range(ns_a, ns_a + ns_b)) + oa[ns_a:], n, gates)  # B: входы — состояние B и выходы A (все, кроме нового состояния A)
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns_a] + ob}  # выходы: новое состояние A + выходы B

def compose_net_parallel_pair(net_a: dict, ns1: int, net_b: dict, ns2: int, ni: int) -> dict:
    """Параллельная пара: A и B получают один и тот же вход.
    
            Входы общей схемы: состояние A, состояние B, общий вход (ni бит).
            Возвращает схему с выходами: новое состояние A + новое состояние B + выход A + выход B."""
    n = ns1 + ns2 + ni; gates: list[dict] = []  # число входов n = ns1 + ns2 + ni; общий список вентилей
    oa = _emit_subcircuit(net_a, list(range(ns1)) + list(range(ns1 + ns2, n)), n, gates)  # A: состояние A и общий вход
    ob = _emit_subcircuit(net_b, list(range(ns1, ns1 + ns2)) + list(range(ns1 + ns2, n)), n, gates)  # B: состояние B и общий вход
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns1] + ob[:ns2] + oa[ns1:] + ob[ns2:]}  # выходы: состояние A, состояние B, выход A, выход B

def compose_net_parallel_lut(net_a: dict, ns1: int, net_b: dict, ns2: int, ni: int, lut4: int) -> dict:
    """Параллельная пара с логическим элементом на выходах: y = lut4(ya, yb).
    
            Выходы A и B (по одному биту) подаются на входы a и b схемы LUT. Возвращает схему с выходами:
            новое состояние A + новое состояние B + выход LUT."""
    n = ns1 + ns2 + ni; gates: list[dict] = []  # число входов n = ns1 + ns2 + ni; общий список вентилей
    oa = _emit_subcircuit(net_a, list(range(ns1)) + list(range(ns1 + ns2, n)), n, gates)  # A: состояние A и общий вход
    ob = _emit_subcircuit(net_b, list(range(ns1, ns1 + ns2)) + list(range(ns1 + ns2, n)), n, gates)  # B: состояние B и общий вход
    oh = _emit_subcircuit(ProofCarryingANFLearner.compile(2, (lut4_anf(lut4),)), [oa[ns1], ob[ns2]], n, gates)  # LUT: схема функции lut4, её входы — выход A и выход B
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns1] + ob[:ns2] + oh}  # выходы: состояние A, состояние B, выход LUT

def compose_net_mux_2in(net_a: dict, ns1: int, net_b: dict, ns2: int) -> dict:
    """Мультиплексор 2 в 1: y = yb, если u1 = 1, иначе ya.
    
            Входы общей схемы: состояние A, состояние B, вход u0 (общий для A и B), выбор u1.
            Реализация: gd = ya ⊕ yb; gs = u1 ∧ gd; y = ya ⊕ gs = ya ⊕ (u1 ∧ (ya ⊕ yb)).
            Возвращает схему с выходами: новое состояние A + новое состояние B + y."""
    n = ns1 + ns2 + 2; u0, u1 = ns1 + ns2, ns1 + ns2 + 1; gates: list[dict] = []  # число входов n = ns1 + ns2 + 2; номера входов u0 и u1; общий список вентилей
    oa = _emit_subcircuit(net_a, list(range(ns1)) + [u0], n, gates)  # A: состояние A и вход u0
    ob = _emit_subcircuit(net_b, list(range(ns1, ns1 + ns2)) + [u0], n, gates)  # B: состояние B и вход u0
    ya, yb = oa[ns1], ob[ns2]  # выходы A и B (по одному биту)
    gd = n + len(gates); gates.append({"a": ya, "b": yb, "mask": 6})  # элемент «исключающее или»: gd = ya ⊕ yb
    gs = n + len(gates); gates.append({"a": u1, "b": gd, "mask": 8})  # элемент «и»: gs = u1 ∧ gd
    gy = n + len(gates); gates.append({"a": ya, "b": gs, "mask": 6})  # y = ya ⊕ gs; итог: y = ya ⊕ (u1 ∧ (ya ⊕ yb))
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns1] + ob[:ns2] + [gy]}  # выходы: состояние A, состояние B, y

def compose_net_moe_mux(net_s: dict, ns_s: int, net_a: dict, ns_a: int, net_b: dict, ns_b: int, ni: int) -> dict:
    """Мультиплексор «смесь экспертов»: выбор по биту схемы S.
    
            Входы: состояние S, состояние A, состояние B, общий вход (ni бит). y = ya ⊕ (ys ∧ (ya ⊕ yb)),
            где ys — выход S. Возвращает схему с выходами: состояния S, A, B (без последних выходов) + y."""
    n = ns_s + ns_a + ns_b + ni; off_u = ns_s + ns_a + ns_b; gates: list[dict] = []; u_wires = list(range(off_u, n))  # число входов n = ns_s + ns_a + ns_b + ni; off_u — начало входов; общий список вентилей; входные провода
    os_ = _emit_subcircuit(net_s, list(range(ns_s)) + u_wires, n, gates)  # S: состояние S и входы
    oa = _emit_subcircuit(net_a, list(range(ns_s, ns_s + ns_a)) + u_wires, n, gates)  # A: состояние A и входы
    ob = _emit_subcircuit(net_b, list(range(ns_s + ns_a, off_u)) + u_wires, n, gates)  # B: состояние B и входы
    ys, ya, yb = os_[-1], oa[-1], ob[-1]  # выходы S, A, B (последние биты)
    gd = n + len(gates); gates.append({"a": ya, "b": yb, "mask": 6})  # элемент «исключающее или»: gd = ya ⊕ yb
    gs = n + len(gates); gates.append({"a": ys, "b": gd, "mask": 8})  # элемент «и»: gs = ys ∧ gd
    gy = n + len(gates); gates.append({"a": ya, "b": gs, "mask": 6})  # элемент «исключающее или»: y = ya ⊕ gs = ya ⊕ (ys ∧ (ya ⊕ yb))
    return {"n_inputs": n, "gates": gates, "outs": os_[:-1] + oa[:-1] + ob[:-1] + [gy]}  # выходы: новые состояния S, A, B + y

def compose_net_delayed_feedback(net_m: dict, ns: int) -> dict:
    """Задержанная обратная связь: выход модуля через задержку возвращается на его вход.
    
            Модуль M (ns бит состояния, входы: задержанный выход d, внешний вход u0). Возвращает схему
            с ns + 2 входами (состояние, u1 = задержка, u0 = вход) и выходами: новое состояние M + y + y."""
    n = ns + 2; gates: list[dict] = []; mo = _emit_subcircuit(net_m, list(range(ns)) + [ns + 1, ns], n, gates)  # число входов n = ns + 2; M: входы — состояние, вход ns + 1 (задержка), вход ns
    return {"n_inputs": n, "gates": gates, "outs": mo[:ns] + [mo[ns], mo[ns]]}  # выходы: новое состояние, выход y, тот же y (он же — задержанный сигнал)

def compose_net_coupled_fb_2bit(net_a: dict, ns1: int, net_b: dict, ns2: int) -> dict:
    """Связанная обратная связь по 2 бита: выходы A и B взаимно задерживаются.
    
            Входы: состояние A, состояние B, da (задержка выхода A), db (задержка выхода B), общий вход u0.
            Возвращает схему с выходами: новое состояние A + новое состояние B + ya + yb + yb."""
    da, db, u0 = ns1 + ns2, ns1 + ns2 + 1, ns1 + ns2 + 2; n = u0 + 1; gates: list[dict] = []  # номера входов da, db, u0; число входов n = u0 + 1; общий список вентилей
    oa = _emit_subcircuit(net_a, list(range(ns1)) + [u0, db], n, gates)  # A: состояние A, входы u0 и db
    ob = _emit_subcircuit(net_b, list(range(ns1, ns1 + ns2)) + [u0, da], n, gates)  # B: состояние B, входы u0 и da
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns1] + ob[:ns2] + [oa[ns1], ob[ns2], ob[ns2]]}  # выходы: состояние A, состояние B, ya, yb, yb

def verify_ag_raw(op: str, raw_net: dict, pa: Skill, pb: Skill | None = None, pc: Skill | None = None, lut4: int = 0) -> tuple[bool, str]:
    """Точная проверка композиции «предполагай — гарантируй»: сырая схема обязана совпасть с эталоном.
    
            op: операция, raw_net: собранная вручную схема, pa, pb, pc: навыки, lut4: таблица функции.
            Для каждой операции строится эталонная схема и проверяется согласованность портов;
            затем сырая схема сравнивается по полям n_inputs, gates, outs.
            Возвращает (True, сообщение) или (False, причина)."""
    if op == "CASCADE": expected, ok_a = compose_net_cascade(pa.net, pa.ns, pa.ni, pb.net, pb.ns, pb.ni), pa.no == pb.ni  # каскад: эталон — compose_net_cascade; порты согласованы, если no(A) = ni(B)
    elif op == "PARALLEL_PAIR": expected, ok_a = compose_net_parallel_pair(pa.net, pa.ns, pb.net, pb.ns, pa.ni), pa.ni == pb.ni  # параллельная пара: порты согласованы, если ni(A) = ni(B)
    elif op == "PARALLEL_LUT": expected, ok_a = compose_net_parallel_lut(pa.net, pa.ns, pb.net, pb.ns, pa.ni, lut4), (pa.ni == pb.ni and pa.no == 1 and pb.no == 1)  # параллельная пара с LUT: ni(A) = ni(B) и по одному выходу у каждого
    elif op == "MUX_2IN": expected, ok_a = compose_net_mux_2in(pa.net, pa.ns, pb.net, pb.ns), (pa.ni == 1 and pb.ni == 1 and pa.no == 1 and pb.no == 1)  # мультиплексор 2 в 1: у обоих ni = 1 и no = 1
    elif op == "MOE_MUX": expected, ok_a = compose_net_moe_mux(pa.net, pa.ns, pb.net, pb.ns, pc.net, pc.ns, pa.ni), (pa.ni == pb.ni == pc.ni and pa.no == pb.no == pc.no == 1)  # «смесь экспертов»: ni равны у всех, по одному выходу у каждого
    elif op == "DELAYED_FEEDBACK": expected, ok_a = compose_net_delayed_feedback(pa.net, pa.ns), (pa.ni == 2 and pa.no == 1)  # задержанная обратная связь: ni = 2, no = 1
    elif op == "COUPLED_FEEDBACK_2BIT": expected, ok_a = compose_net_coupled_fb_2bit(pa.net, pa.ns, pb.net, pb.ns), (pa.ni == 2 and pb.ni == 2 and pa.no == 1 and pb.no == 1)  # связанная обратная связь: ni = 2 и no = 1 у обоих
    else: return False, f"unsupported AG op {op}"  # неизвестная операция
    if not ok_a: return False, "port arity mismatch"  # порты не согласованы
    if any(raw_net[k] != expected[k] for k in ("n_inputs", "gates", "outs")): return False, "subcircuit mismatch"  # сырая схема отличается от эталона хотя бы в одном поле
    return True, f"exact modular assume-guarantee ({op})"  # точная модульная проверка пройдена

def _compose_anf_and_rows(op: str, pa: Skill, pb: Skill | None = None, pc: Skill | None = None, lut4: int = 0):
    """АНФ и таблица строк для композиции навыков.
    
            op: операция, pa, pb, pc: навыки, lut4: таблица функции. Переменные перенумеровываются в общий
            набор (состояния, затем входы), многочлены родителей подставляются друг в друга.
            Возвращает (АНФ кортежем, строки таблицы)."""
    if op == "CASCADE":  # каскад
        ns1, ns2 = pa.ns, pb.ns  # разрядности состояний
        ao = tuple(remap(p, tuple(list(range(ns1)) + [ns1 + ns2 + j for j in range(pa.ni)])) for p in pa.anf)  # АНФ A: состояние A остаётся, вход A заменяется на входы после состояний A и B
        anf = tuple(ao[:ns1]) + tuple(substitute(p, tuple([frozenset({1 << (ns1 + j)}) for j in range(ns2)] + list(ao[ns1:]))) for p in pb.anf)  # АНФ: новое состояние A из A; выход B = B(выход A), подстановка многочленов A вместо входов B
        rows = []  # строки таблицы
        for x in range(1 << (ns1 + ns2 + pa.ni)):  # по всем x из 2^(ns1 + ns2 + ni)
            s1, s2, u = x & ((1 << ns1) - 1), (x >> ns1) & ((1 << ns2) - 1), x >> (ns1 + ns2)  # s1, s2 — состояния A и B (младшие биты), u — вход
            r1 = pa.rows[s1 | (u << ns1)]; r2 = pb.rows[s2 | (sum(v << i for i, v in enumerate(r1[ns1:])) << ns2)]  # r1 — шаг A; r2 — шаг B с входом, равным выходу A (в виде числа)
            rows.append(tuple(list(r1[:ns1]) + list(r2)))  # строка: новое состояние A + строка B
        return anf, rows  # вернуть (АНФ, строки)
    if op == "PARALLEL_PAIR":  # ветка параллельной пары
        ns1, ns2, ni = pa.ns, pb.ns, pa.ni  # параллельная пара: разрядности
        ao = tuple(remap(p, tuple(list(range(ns1)) + [ns1 + ns2 + j for j in range(ni)])) for p in pa.anf)  # АНФ A в общих переменных
        bo = tuple(remap(p, tuple(list(range(ns1, ns1 + ns2)) + [ns1 + ns2 + j for j in range(ni)])) for p in pb.anf)  # АНФ B в общих переменных
        anf = tuple(ao[:ns1]) + tuple(bo[:ns2]) + tuple(ao[ns1:]) + tuple(bo[ns2:])  # АНФ: состояние A + состояние B + выход A + выход B
        rows = [tuple(list(pa.rows[(x & ((1 << ns1) - 1)) | ((x >> (ns1 + ns2)) << ns1)][:ns1]) +  # строки: по всем x; берётся шаг A (состояние A, вход)
                      list(pb.rows[((x >> ns1) & ((1 << ns2) - 1)) | ((x >> (ns1 + ns2)) << ns2)][:ns2]) +  # шаг B (состояние B, вход)
                      list(pa.rows[(x & ((1 << ns1) - 1)) | ((x >> (ns1 + ns2)) << ns1)][ns1:]) +  # выход A
                      list(pb.rows[((x >> ns1) & ((1 << ns2) - 1)) | ((x >> (ns1 + ns2)) << ns2)][ns2:]))  # выход B; строка = новое состояние A + новое состояние B + выход A + выход B
                for x in range(1 << (ns1 + ns2 + ni))]  # перебор всех x
        return anf, rows  # вернуть (АНФ, строки)
    if op == "PARALLEL_LUT":  # ветка параллельной пары с общей таблицей функции
        ns1, ns2, ni = pa.ns, pb.ns, pa.ni  # параллельная пара с LUT: разрядности
        ao = tuple(remap(p, tuple(list(range(ns1)) + [ns1 + ns2 + j for j in range(ni)])) for p in pa.anf)  # АНФ A в общих переменных
        bo = tuple(remap(p, tuple(list(range(ns1, ns1 + ns2)) + [ns1 + ns2 + j for j in range(ni)])) for p in pb.anf)  # АНФ B в общих переменных
        anf = tuple(ao[:ns1]) + tuple(bo[:ns2]) + (substitute(lut4_anf(lut4), (ao[ns1], bo[ns2])),)  # АНФ: состояния + выход LUT(ya, yb) подстановкой многочленов выходов
        rows = []  # строки
        for x in range(1 << (ns1 + ns2 + ni)):  # по всем x
            r1 = pa.rows[(x & ((1 << ns1) - 1)) | ((x >> (ns1 + ns2)) << ns1)]  # шаг A
            r2 = pb.rows[((x >> ns1) & ((1 << ns2) - 1)) | ((x >> (ns1 + ns2)) << ns2)]  # шаг B
            rows.append(tuple(list(r1[:ns1]) + list(r2[:ns2]) + [lut4_eval(lut4, r1[ns1], r2[ns2])]))  # строка: состояние A + состояние B + lut4(ya, yb)
        return anf, rows  # вернуть (АНФ, строки)
    if op == "MUX_2IN":  # ветка мультиплексора двух входов
        ns1, ns2 = pa.ns, pb.ns; u0, u1 = ns1 + ns2, ns1 + ns2 + 1  # мультиплексор 2 в 1: разрядности; номера входов u0 (общий) и u1 (выбор)
        ao = tuple(remap(p, tuple(list(range(ns1)) + [u0])) for p in pa.anf)  # АНФ A в общих переменных (вход u0)
        bo = tuple(remap(p, tuple(list(range(ns1, ns1 + ns2)) + [u0])) for p in pb.anf)  # АНФ B в общих переменных (вход u0)
        anf = tuple(ao[:ns1]) + tuple(bo[:ns2]) + (xor(ao[ns1], mul(frozenset({1 << u1}), xor(ao[ns1], bo[ns2]))),)  # выход: y = ya ⊕ (u1 ∧ (ya ⊕ yb))
        rows = []  # строки
        for x in range(1 << (ns1 + ns2 + 2)):  # по всем x из 2^(ns1 + ns2 + 2)
            r1 = pa.rows[(x & ((1 << ns1) - 1)) | (((x >> (ns1 + ns2)) & 1) << ns1)]  # шаг A с входом u0
            r2 = pb.rows[((x >> ns1) & ((1 << ns2) - 1)) | (((x >> (ns1 + ns2)) & 1) << ns2)]  # шаг B с входом u0
            rows.append(tuple(list(r1[:ns1]) + list(r2[:ns2]) + [r2[ns2] if ((x >> (ns1 + ns2 + 1)) & 1) else r1[ns1]]))  # строка: состояние A + состояние B + (yb, если u1 = 1, иначе ya)
        return anf, rows  # вернуть (АНФ, строки)
    if op == "MOE_MUX":  # ветка мультиплексора смеси экспертов
        ns_s, ns_a, ns_b, ni = pa.ns, pb.ns, pc.ns, pa.ni; off_a, off_b, off_u = ns_s, ns_s + ns_a, ns_s + ns_a + ns_b  # «смесь экспертов»: разрядности; смещения состояния A, состояния B и входов
        u_map = [off_u + j for j in range(ni)]  # номера входных проводов u_map = off_u + j
        so = tuple(remap(p, tuple(list(range(ns_s)) + u_map)) for p in pa.anf)  # АНФ схемы S (переключателя) в общих переменных
        ao = tuple(remap(p, tuple(list(range(off_a, off_a + ns_a)) + u_map)) for p in pb.anf)  # АНФ эксперта A в общих переменных
        bo = tuple(remap(p, tuple(list(range(off_b, off_b + ns_b)) + u_map)) for p in pc.anf)  # АНФ эксперта B в общих переменных
        anf = tuple(so[:ns_s]) + tuple(ao[:ns_a]) + tuple(bo[:ns_b]) + (xor(ao[ns_a], mul(so[ns_s], xor(ao[ns_a], bo[ns_b]))),)  # АНФ: состояния S, A, B + y = ya ⊕ (ys ∧ (ya ⊕ yb))
        rows = []  # строки
        for x in range(1 << (ns_s + ns_a + ns_b + ni)):  # по всем x
            u = x >> off_u  # u — вход (старшие биты x)
            rs = pa.rows[(x & ((1 << ns_s) - 1)) | (u << ns_s)]  # шаг S
            ra = pb.rows[((x >> ns_s) & ((1 << ns_a) - 1)) | (u << ns_a)]  # шаг A
            rb = pc.rows[((x >> off_b) & ((1 << ns_b) - 1)) | (u << ns_b)]  # шаг B
            rows.append(tuple(list(rs[:ns_s]) + list(ra[:ns_a]) + list(rb[:ns_b]) + [rb[ns_b] if rs[ns_s] else ra[ns_a]]))  # строка: состояния S, A, B + (yb, если ys = 1, иначе ya)
        return anf, rows  # вернуть (АНФ, строки)
    if op == "DELAYED_FEEDBACK":  # ветка задержанной обратной связи
        ns = pa.ns; mo = tuple(remap(p, tuple(list(range(ns)) + [ns + 1, ns])) for p in pa.anf)  # задержанная обратная связь: АНФ A с входами (задержка d = бит ns + 1, вход u0 = бит ns)
        anf = tuple(mo[:ns]) + (mo[ns], mo[ns])  # АНФ: новое состояние + y + y
        rows = []  # строки
        for x in range(1 << (ns + 2)):  # по всем x из 2^(ns + 2)
            s, d, u0 = x & ((1 << ns) - 1), (x >> ns) & 1, (x >> (ns + 1)) & 1  # s — состояние, d — задержанный выход, u0 — вход
            rm = pa.rows[s | ((u0 | (d << 1)) << ns)]; y = rm[ns]  # шаг A с входом (u0, d): вход модуля = u0 | (d << 1); y — выход
            rows.append(tuple(list(rm[:ns]) + [y, y]))  # строка: новое состояние + y + y
        return anf, rows  # вернуть (АНФ, строки)
    ns1, ns2 = pa.ns, pb.ns; da, db, u0 = ns1 + ns2, ns1 + ns2 + 1, ns1 + ns2 + 2  # связанная обратная связь: разрядности; номера входов da, db (задержки), u0 (вход)
    ao = tuple(remap(p, tuple(list(range(ns1)) + [u0, db])) for p in pa.anf)  # АНФ A: входы u0 и db
    bo = tuple(remap(p, tuple(list(range(ns1, ns1 + ns2)) + [u0, da])) for p in pb.anf)  # АНФ B: входы u0 и da
    anf = tuple(ao[:ns1]) + tuple(bo[:ns2]) + (ao[ns1], bo[ns2], bo[ns2])  # АНФ: состояния A, B + ya + yb + yb
    rows = []  # строки
    for x in range(1 << (u0 + 1)):  # по всем x
        sa, sb = x & ((1 << ns1) - 1), (x >> ns1) & ((1 << ns2) - 1)  # sa, sb — состояния A и B
        va, vb, vu = (x >> da) & 1, (x >> db) & 1, x >> u0  # va, vb — задержанные выходы, vu — вход
        ra = pa.rows[sa | ((vu | (vb << 1)) << ns1)]; rb = pb.rows[sb | ((vu | (va << 1)) << ns2)]  # шаги A и B: вход A = vu | (vb << 1), вход B = vu | (va << 1)
        rows.append(tuple(list(ra[:ns1]) + list(rb[:ns2]) + [ra[ns1], rb[ns2], rb[ns2]]))  # строка: состояние A + состояние B + ya + yb + yb
    return anf, rows  # вернуть (АНФ, строки)

def compose_quotient(op: str, pa: Skill, pb: Skill | None = None, pc: Skill | None = None, lut4: int = 0):
    """Композиция фактор-автоматов: таблица переходов произведения и её минимизация.
    
            op: операция, pa, pb, pc: навыки, lut4: таблица функции. Произведение состояний кодируется числом
            (s1·k2 + s2) (или вложенно для трёх автоматов). Возвращает (k, хеш, T) — первые три результата
            minimize_quotient_table. Неизвестная операция — ValueError."""
    if op == "CASCADE":  # ветка каскада (выход первого автомата — вход второго)
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb, ni = len(Ta), len(Tb), pa.ni  # каскад: фактор-таблицы, числа состояний, вход A
        raw = [tuple((Ta[s1][u][0] * kb + Tb[s2][sum(v << i for i, v in enumerate(Ta[s1][u][1]))][0],  # сырая таблица: состояние (s1·kb + s2), по входу u: A переходит, его выход (как число) — вход B
                      Tb[s2][sum(v << i for i, v in enumerate(Ta[s1][u][1]))][1]) for u in range(1 << ni))  # продолжение: выход B
               for s1 in range(ka) for s2 in range(kb)]  # перебор всех пар состояний
        return minimize_quotient_table(ka * kb, ni, pb.no, raw, 0)[:3]  # минимизация произведения; вернуть (k, хеш, T)
    if op == "PARALLEL_PAIR":  # ветка параллельной пары
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb, ni = len(Ta), len(Tb), pa.ni  # параллельная пара
        raw = [tuple((Ta[s1][u][0] * kb + Tb[s2][u][0], Ta[s1][u][1] + Tb[s2][u][1]) for u in range(1 << ni)) for s1 in range(ka) for s2 in range(kb)]  # сырая таблица: один вход u, переходы обоих, выход — конкатенация выходов
        return minimize_quotient_table(ka * kb, ni, pa.no + pb.no, raw, 0)[:3]  # минимизация, вернуть (k, хеш, T)
    if op == "PARALLEL_LUT":  # ветка параллельной пары с общей таблицей функции
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb, ni = len(Ta), len(Tb), pa.ni  # параллельная пара с LUT
        raw = [tuple((Ta[s1][u][0] * kb + Tb[s2][u][0], (lut4_eval(lut4, Ta[s1][u][1][0], Tb[s2][u][1][0]),)) for u in range(1 << ni)) for s1 in range(ka) for s2 in range(kb)]  # сырая таблица: выход = lut4(ya, yb)
        return minimize_quotient_table(ka * kb, ni, 1, raw, 0)[:3]  # минимизация с одним выходным битом
    if op == "MUX_2IN":  # ветка мультиплексора двух входов
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb = len(Ta), len(Tb)  # мультиплексор 2 в 1: вход u = (u0, u1)
        raw = [tuple((Ta[s1][u & 1][0] * kb + Tb[s2][u & 1][0], Tb[s2][u & 1][1] if ((u >> 1) & 1) else Ta[s1][u & 1][1]) for u in range(4)) for s1 in range(ka) for s2 in range(kb)]  # сырая таблица: вход u0 = u & 1; выход B, если u1 = 1, иначе выход A
        return minimize_quotient_table(ka * kb, 2, 1, raw, 0)[:3]  # минимизация с двумя входными битами
    if op == "MOE_MUX":  # ветка мультиплексора смеси экспертов
        Ts, Ta, Tb = pa.quotient_T, pb.quotient_T, pc.quotient_T; ks, ka, kb, ni = len(Ts), len(Ta), len(Tb), pa.ni  # «смесь экспертов»: три фактор-таблицы
        raw = [tuple((((Ts[ss][u][0] * ka) + Ta[sa][u][0]) * kb + Tb[sb][u][0], (Tb[sb][u][1][0] if Ts[ss][u][1][0] else Ta[sa][u][1][0],)) for u in range(1 << ni))  # сырая таблица: состояние ((s_s·ka + s_a)·kb + s_b); выход B, если ys = 1, иначе A
               for ss in range(ks) for sa in range(ka) for sb in range(kb)]  # перебор всех троек состояний
        return minimize_quotient_table(ks * ka * kb, ni, 1, raw, 0)[:3]  # минимизация
    if op == "DELAYED_FEEDBACK":  # ветка задержанной обратной связи
        Tm = pa.quotient_T; km = len(Tm)  # задержанная обратная связь: модуль M, km состояний
        raw = [tuple((Tm[sm][u0 | (d << 1)][0] * 2 + Tm[sm][u0 | (d << 1)][1][0], (Tm[sm][u0 | (d << 1)][1][0],)) for u0 in (0, 1)) for sm in range(km) for d in (0, 1)]  # сырая таблица: состояние (s_m·2 + d); вход u0 | (d << 1); выход модуля; новое d = выход
        return minimize_quotient_table(km * 2, 1, 1, raw, 0)[:3]  # минимизация с одним входным битом
    if op == "COUPLED_FEEDBACK_2BIT":  # ветка связанной обратной связи на 2 бита
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb = len(Ta), len(Tb)  # связанная обратная связь
        raw = [tuple(((((Ta[sa][u0 | (db << 1)][0] * kb + Tb[sb][u0 | (da << 1)][0]) * 2 + Ta[sa][u0 | (db << 1)][1][0]) * 2 + Tb[sb][u0 | (da << 1)][1][0]),  # сырая таблица: состояние ((s_a·kb + s_b)·2 + da)·2 + db; выход B
                      (Tb[sb][u0 | (da << 1)][1][0],)) for u0 in (0, 1))  # продолжение: выход
               for sa in range(ka) for sb in range(kb) for da in (0, 1) for db in (0, 1)]  # перебор sa, sb, da, db
        return minimize_quotient_table(ka * kb * 4, 1, 1, raw, 0)[:3]  # минимизация с одним входным битом
    raise ValueError(op)  # неизвестная операция — ошибка

def canonical_poly(p: frozenset[int], ns: int, ni: int) -> frozenset[int]:
    """Перенумеровать переменные многочлена в глобальные номера: состояние 0..ns−1, вход 22..22+ni−1.
    
            Возвращает многочлен после remap."""
    return remap(p, tuple(list(range(ns)) + [22 + j for j in range(ni)]))  # множество переменных: list(range(ns)) + [22 + j для j < ni]

class TemporalDAG:  # глобальная схема всех навыков; описание в docstring ниже
    """Глобальная схема всех навыков (ациклический граф вентилей) с общими вентилями.
    
            Каждый навык добавляется со своими входами: состояние — провода 0..ns−1, вход — провода 22 и далее.
            Совпадающие по сигнатуре (mask, a, b) вентили разных навыков объединяются; для навыков с АНФ
            дополнительно объединяются вентили с одинаковым многочленом (семантическое слияние)."""
    def __init__(self):  # создать пустую схему
        self.n = GLOBAL_N; self.gates: list[dict] = []; self.depth = [0] * GLOBAL_N  # число входов GLOBAL_N; списки вентилей и глубин
        self.struct: dict[tuple[int, int, int], int] = {}  # struct: сигнатура вентиля -> номер провода
        self.wire_polys: list[frozenset[int] | None] = [frozenset({1 << i}) for i in range(GLOBAL_N)]  # многочлены проводов (вход i = {x_i})
        self.poly_struct: dict[frozenset[int], int] = {frozenset({1 << i}): i for i in range(GLOBAL_N)}  # poly_struct: многочлен -> номер провода
        self.semantic_merges = 0; self.outputs: dict[str, list[int]] = {}  # счётчик семантических слияний; outputs: имя навыка -> список выходных проводов
        self.expected: dict[str, list[list[int]]] = {}; self.ag_skills: dict[str, dict] = {}  # expected: имя -> ожидаемые АНФ выходов (списки мономов); ag_skills: имя -> схема навыка без АНФ

    def add(self, s: Skill):
        """Добавить навык в глобальную схему.
        
                Входы навыка: состояние — провода 0..ns−1, вход — провода 22..22+ni−1. Совпадающие
                вентили переиспользуются; для навыков с АНФ вентиль с уже известным многочленом тоже
                переиспользуется. Запоминаются выходы и (если есть АНФ) ожидаемые многочлены выходов."""
        wires = list(range(s.ns)) + [22 + j for j in range(s.ni)]; track_poly = s.anf is not None  # карта входов навыка: состояние 0..ns−1, вход 22+j; track_poly — вести многочлены (если есть АНФ)
        for g in s.net["gates"]:  # по каждому вентилю навыка
            a, b, m = wires[g["a"]], wires[g["b"]], g["mask"]  # входы через карту и маска
            if m in (0, 15): a = b = 0  # константы: оба входа — провод 0
            elif m in (6, 8) and b < a: a, b = b, a  # коммутативные вентили: входы упорядочиваются a ≤ b
            sig = (m, a, b)  # сигнатура вентиля
            if sig in self.struct: z = self.struct[sig]  # такой вентиль уже есть — использовать его провод
            else:  # иначе — создать новый
                gp: frozenset[int] | None = None  # многочлен результата gp (None, если не отслеживаем)
                if track_poly and self.wire_polys[a] is not None and self.wire_polys[b] is not None:  # если отслеживаем многочлены и оба входа известны
                    pa, pb = self.wire_polys[a], self.wire_polys[b]  # многочлены входов
                    gp = frozenset() if m == 0 else (frozenset({0}) if m == 15 else (xor(pa, pb) if m == 6 else mul(pa, pb)))  # gp = 0 / 1 / pa ⊕ pb / pa · pb по маске
                if gp is not None and gp in self.poly_struct:  # многочлен уже на проводе — семантическое слияние
                    z = self.poly_struct[gp]; self.struct[sig] = z; self.semantic_merges += 1  # использовать существующий провод и запомнить сигнатуру
                else:  # иначе — новый вентиль
                    z = self.n + len(self.gates); self.struct[sig] = z  # номер провода = n + число вентилей; запомнить сигнатуру
                    self.gates.append({"a": a, "b": b, "mask": m}); self.depth.append(1 + max(self.depth[a], self.depth[b]))  # добавить вентиль; глубина = 1 + max(depth(a), depth(b))
                    self.wire_polys.append(gp)  # добавить многочлен провода
                    if gp is not None: self.poly_struct[gp] = z  # если многочлен известен — запомнить его провод
            wires.append(z)  # провод выхода вентиля добавляется в карту
        self.outputs[s.name] = [wires[o] for o in s.net["outs"]]  # выходные провода навыка
        if s.anf is not None: self.expected[s.name] = [sorted(canonical_poly(p, s.ns, s.ni)) for p in s.anf]  # если есть АНФ — запомнить ожидаемые многочлены в глобальных переменных
        else: self.ag_skills[s.name] = s.net  # иначе — запомнить схему навыка без АНФ

    def verify(self, skills_dict: dict[str, Skill] | None = None) -> tuple[bool, list[str]]:
        """Проверить глобальную схему: порядок вентилей и совпадение выходов с ожидаемыми АНФ.
        
                skills_dict: имена навыков -> объекты (для навыков без АНФ, чтобы восстановить карту входов).
                Возвращает (True, []) или (False, список ошибок)."""
        w: list[frozenset[int] | None] = [frozenset({1 << i}) for i in range(self.n)]; err: list[str] = []  # многочлены проводов; список ошибок
        max_anf_gate = -1  # максимальный номер вентиля, нужный для АНФ-выходов (−1 — пока неизвестен)
        if self.expected:  # если есть навыки с АНФ
            need = {o for nm in self.expected for o in self.outputs[nm] if o >= self.n}  # need — выходные провода-вентили навыков с АНФ
            stack = list(need); max_anf_gate = max(need, default=-1); seen = set(need)  # обход назад от выходов: стек, максимум номера и просмотренные
            while stack:  # пока стек не пуст
                gw = stack.pop(); g = self.gates[gw - self.n]  # взять провод gw и его вентиль
                for ch in (g["a"], g["b"]):  # по обоим входам вентиля
                    if ch >= self.n and ch not in seen: seen.add(ch); stack.append(ch); max_anf_gate = max(max_anf_gate, ch)  # вход-вентиль добавить в просмотренные и в стек; обновить максимум
        for i, g in enumerate(self.gates):  # по каждому вентилю i
            a, b, m = g["a"], g["b"], g["mask"]  # входы и маска
            if not (0 <= a < self.n + i and 0 <= b < self.n + i): err.append(f"forward:{i}"); w.append(frozenset()); continue  # вход позже вентиля — ошибка; многочлен провода неизвестен; дальше
            w.append((frozenset() if m == 0 else (frozenset({0}) if m == 15 else (xor(w[a], w[b]) if m == 6 else (mul(w[a], w[b]) if m == 8 else frozenset())))) if self.n + i <= max_anf_gate else None)  # многочлен провода по маске (0, 1, XOR, AND); для вентилей за пределами нужных — None
        for name, exp in self.expected.items():  # по навыкам с АНФ
            if tuple(w[o] for o in self.outputs[name]) != tuple(frozenset(x) for x in exp): err.append(f"output_anf:{name}")  # многочлены выходов обязаны совпасть с ожидаемыми
        for name, net in self.ag_skills.items():  # по навыкам без АНФ
            sk = skills_dict.get(name) if skills_dict else None  # навык из словаря (если передан)
            wires = (list(range(sk.ns)) + [22 + j for j in range(sk.ni)]) if sk else (list(range(net["n_inputs"] - 1)) + [22])  # карта входов навыка: состояние 0..ns−1, вход 22+j (без объекта — по числу входов схемы)
            for g in net["gates"]:  # по вентилям схемы навыка
                a, b, m = wires[g["a"]], wires[g["b"]], g["mask"]  # входы через карту и маска
                if m in (0, 15): a = b = 0  # константы и порядок входов — как при добавлении
                elif m in (6, 8) and b < a: a, b = b, a  # коммутативные вентили: упорядочить
                if (m, a, b) not in self.struct: err.append(f"ag_dag_missing:{name}"); break  # вентиля нет в глобальной схеме — ошибка, дальше не проверяем
                wires.append(self.struct[(m, a, b)])  # провод вентиля из глобальной схемы
            if [wires[o] for o in net["outs"]] != self.outputs[name]: err.append(f"ag_dag_output:{name}")  # выходы навыка обязаны совпасть с выходами в глобальной схеме
        return not err, err  # вернуть (ошибок нет, список ошибок)

    def metrics(self) -> dict:
        """Метрики глобальной схемы.
        
                Возвращает словарь: число вентилей, глубина, максимальный разветвитель, биты программы
                (как в _finalize_net) и число семантических слияний."""
        fan = [0] * (self.n + len(self.gates))  # число использований каждого провода
        for g in self.gates: fan[g["a"]] += 1; fan[g["b"]] += 1  # каждый вход каждого вентиля получает использование
        for outs in self.outputs.values():  # по всем наборам выходов
            for x in outs: fan[x] += 1  # каждый выходной провод получает использование
        d = max((self.depth[o] for outs in self.outputs.values() for o in outs), default=0)  # глубина d = max depth по всем выходам
        wb = max(1, (self.n + len(self.gates) - 1).bit_length()); ob = max(1, (self.n + len(self.gates)).bit_length())  # бит на номер провода wb и на выход ob
        return {"gate_count": len(self.gates), "depth": d, "max_fanout": max(fan, default=0),  # словарь метрик
                "program_bits": len(self.gates) * (4 + 2 * wb) + sum(len(o) * ob for o in self.outputs.values()), "semantic_merges": self.semantic_merges}  # биты программы = вентили·(4 + 2·wb) + Σ по навыкам (число выходов)·ob; семантические слияния

    def export(self) -> dict:
        """Экспорт глобальной схемы в словарь для JSON.
        
                Возвращает число входов, вентили, выходы, ожидаемые АНФ, имена навыков без АНФ и метрики."""
        return {"n_inputs": self.n, "gates": self.gates, "outputs": self.outputs, "expected_anf": self.expected, "ag_skill_names": sorted(self.ag_skills.keys()), "metrics": self.metrics()}  # словарь с полями схемы, ожидаемыми АНФ, именами навыков без АНФ и метриками

class SelfModel:
    """Модель самого себя: статистика успешностей и предупреждения.
    
            Собирает число попыток и точных результатов, включает «уверенность» после min_trials
            точных подряд, журнал предупреждений, статистику перенаправления, экономию от адаптивных
            деревьев и синтеза, счётчики семейств решений."""
    def __init__(self, min_trials: int = 3):  # создать модель: минимальное число попыток, счётчики, журнал
        self.min_trials = min_trials; self.trials = 0; self.exact = 0; self.enabled = False  # счётчики попыток/точных, признак включения
        self.history: list[dict] = []; self.epistemic_alerts: list[dict] = []; self.mdl_predictions: list[dict] = []; self.family_hits: dict[str, int] = {}  # история, предупреждения, предсказания перенаправления, попадания по семействам
        self.adt_verifications = self.adt_baseline_wp_ticks = self.adt_actual_ticks = self.pac_lstar_solves = self.sop_baseline_gates = self.opt_chosen_gates = 0  # счётчики проверок деревьев и синтеза (всё по нулям)

    def observe(self, ok: bool, queries: int):
        """Записать результат попытки.
        
                ok: точен ли результат, queries: число запросов. Неудача выключает модель; min_trials точных
                подряд — включает. История пополняется записью (номер, точность, запросы)."""
        self.trials += 1; self.exact += int(ok); self.history.append({"trial": self.trials, "exact": bool(ok), "warmup_queries": queries})  # попыток +1, точных +ok; запись в историю
        if not ok: self.enabled = False  # неудача — выключить модель
        elif len(self.history) >= self.min_trials and all(x["exact"] for x in self.history[-self.min_trials:]): self.enabled = True  # min_trials точных подряд (последних) — включить

    def log_alert(self, kind: str, detail: str, blocked: bool):
        """Записать предупреждение.
        
                kind: вид, detail: описание, blocked: блокировка или эскалация. Добавляет запись в журнал."""
        self.epistemic_alerts.append({"kind": kind, "detail": detail, "blocked_or_escalated": bool(blocked)})  # добавить запись с признаком блокировки/эскалации

    def record_mdl(self, pred_hier: bool, actual_mode: str, flat_gates: int, hier_gates: int, min_gates: int):
        """Записать предсказание маршрутизации.
        
                pred_hier: предсказана ли иерархия, actual_mode: фактический способ, flat_gates, hier_gates, min_gates:
                размеры схем. Предсказание верно, если совпал вид или размеры равны. Добавляет запись в список."""
        self.mdl_predictions.append({"pred_hier": bool(pred_hier), "actual_mode": actual_mode,  # добавить запись: предсказание и фактический режим
                                     "correct_hier_call": bool(pred_hier == (actual_mode == "HIERARCHICAL") or hier_gates == flat_gates),  # верность предсказания: совпал режим или размеры равны
                                     "flat_gates": flat_gates, "hier_gates": hier_gates, "min_gates": min_gates})  # размеры схем

    def record_adt(self, wp_ticks: int, adt_ticks: int):
        """Записать результат проверки адаптивным деревом.
        
                wp_ticks: длина классического теста, adt_ticks: длина адаптивного. Увеличивает счётчик и суммы длин."""
        self.adt_verifications += 1; self.adt_baseline_wp_ticks += wp_ticks; self.adt_actual_ticks += adt_ticks  # проверок +1; суммировать длины

    def record_synthesis(self, sop_gates: int, chosen_gates: int):
        """Записать результат синтеза: sop_gates — базовый размер, chosen_gates — выбранный.
        
                Увеличивает суммы для расчёта экономии."""
        self.sop_baseline_gates += sop_gates; self.opt_chosen_gates += chosen_gates  # суммировать размеры

    def record_family_hit(self, family: str): self.family_hits[family] = self.family_hits.get(family, 0) + 1  # Записать попадание семейства решений: family_hits[family] + 1.
    def ranked_families(self, default_order: list[str]) -> list[str]:
        """Ранжировать семейства по числу попаданий (по убыванию), при равенстве — исходный порядок."""
        return sorted(default_order, key=lambda f: (-self.family_hits.get(f, 0), default_order.index(f)))  # сортировка по (−попадания, исходный индекс)

    def export(self) -> dict:
        """Экспорт модели в словарь.
        
                Возвращает все счётчики, историю, предупреждения, точность перенаправления (доля верных,
                округление 4 знака), экономию в процентах (округление 2 знака) и счётчики."""
        acc = (sum(x["correct_hier_call"] for x in self.mdl_predictions) / len(self.mdl_predictions)) if self.mdl_predictions else 1.0  # точность перенаправления = доля верных предсказаний (1.0, если предсказаний нет)
        adt_saved = (1.0 - self.adt_actual_ticks / self.adt_baseline_wp_ticks) if self.adt_baseline_wp_ticks > 0 else 0.0  # экономия деревьев = 1 − (длина адаптивного теста / длина классического), если классический > 0, иначе 0
        synth_saved = (1.0 - self.opt_chosen_gates / self.sop_baseline_gates) if self.sop_baseline_gates > 0 else 0.0  # экономия синтеза = 1 − (выбранные вентили / базовые), если базовые > 0, иначе 0
        return {"min_trials": self.min_trials, "trials": self.trials, "exact": self.exact, "enabled": self.enabled,  # словарь: параметры, счётчики
                "history": self.history, "epistemic_alerts": self.epistemic_alerts, "mdl_routing_accuracy": round(acc, 4),  # история, предупреждения, точность перенаправления
                "mdl_observations": len(self.mdl_predictions), "mdl_predictions": self.mdl_predictions, "family_hits": self.family_hits,  # число наблюдений, предсказания, попадания по семействам
                "adt_verifications": self.adt_verifications, "adt_baseline_wp_ticks": self.adt_baseline_wp_ticks,  # счётчики проверок деревьев
                "adt_actual_ticks": self.adt_actual_ticks, "adt_savings_pct": round(100.0 * adt_saved, 2),  # суммарная длина адаптивных тестов, экономия в процентах
                "pac_lstar_solves": self.pac_lstar_solves, "sop_baseline_gates": self.sop_baseline_gates,  # число решений через L* с фильтром шума
                "opt_chosen_gates": self.opt_chosen_gates, "synthesis_compression_pct": round(100.0 * synth_saved, 2)}  # суммы и экономия синтеза в процентах

class Library:
    """Библиотека навыков: хранение, дедупликация и глобальная схема.
    
            skills: навыки по имени, aliases: синонимы, cache: кэш композиций по ключу, by_behav: хеш поведения -> имя,
            dag: глобальная схема, local_gates и flat_equivalent_gates: суммы размеров схем."""
    def __init__(self):  # создать библиотеку: пустые словари, глобальная схема, нулевые суммы
        self.skills: dict[str, Skill] = {}; self.aliases: dict[str, str] = {}; self.cache: dict[str, str] = {}  # навыки, синонимы, кэш, хеши поведения, схема, счётчики вентилей
        self.by_behav: dict[str, str] = {}; self.dag = TemporalDAG(); self.local_gates = 0; self.flat_equivalent_gates = 0  # (продолжение) by_behav и dag; счётчики вентилей

    def add(self, s: Skill, flat_gate_count: int | None = None) -> tuple[str, bool]:
        """Добавить навык в библиотеку с дедупликацией.
        
                s: навык, flat_gate_count: размер «плоской» схемы для статистики. Если хеш поведения уже есть —
                имя становится синонимом существующего навыка. Если совпадает хеш содержания — тоже синоним.
                Возвращает (имя навыка, добавлен ли новый). Навык попадает в глобальную схему."""
        if s.behav_digest and s.behav_digest in self.by_behav:  # хеш поведения уже есть
            target = self.by_behav[s.behav_digest]; self.aliases[s.name] = target; return target, False  # имя становится синонимом существующего; вернуть (цель, False)
        dg = s.semantic_digest()  # хеш содержания
        for name, old in self.skills.items():  # по всем навыкам
            if old.semantic_digest() == dg: self.aliases[s.name] = name; return name, False  # совпадение хеша содержания — синоним; вернуть (имя, False)
        self.skills[s.name] = s  # добавить навык
        if s.behav_digest: self.by_behav[s.behav_digest] = s.name  # запомнить хеш поведения
        self.local_gates += len(s.net["gates"]); self.flat_equivalent_gates += flat_gate_count if flat_gate_count is not None else len(s.net["gates"])  # суммы вентилей: локальных и эквивалентных плоских; добавить в глобальную схему; вернуть (имя, True)
        self.dag.add(s); return s.name, True  # добавить навык в глобальную схему; вернуть (имя, True)

    def resolve(self, name: str) -> str:
        """Разрешить имя через цепочку синонимов.
        
                Возвращает каноническое имя; если его нет — KeyError."""
        while name in self.aliases: name = self.aliases[name]  # пока имя — синоним, подставлять цель
        if name not in self.skills: raise KeyError(name)  # нет такого навыка — ошибка
        return name  # каноническое имя


# ---------------------------------------------------------------------------
# 5. Оракул Мили, адаптивные деревья различения и активное обучение L* со случайным оракулом
# ---------------------------------------------------------------------------

class NonDeterministicOracleError(RuntimeError): pass  # ошибка: оракул выдал разные ответы на один и тот же запрос (недетерминирован)
class UnrecoverableNoiseError(RuntimeError): pass  # ошибка: шум не удалось отфильтровать за допустимое число повторений

class InteractiveMealyOracle:
    """Оракул автомата Мили: отвечает на последовательность входов выходами, с кэшем и фильтром шума.
    
            ni, no: разрядности входа и выхода, eval_fn: функция последовательность -> кортеж выходов,
            max_pac_reps: максимум повторений для фильтрации шума, alpha_step: допустимая вероятность
            ошибки на один шаг, min_pac_m: минимум повторений до фиксации бита.
            В обычном режиме ответы кэшируются и проверяются на детерминированность; в режиме pac_mode
            (после недетерминированности) каждый префикс опрашивается повторно, пока большинство не
            станет статистически значимым, и фиксируется («locked»)."""
    def __init__(self, ni: int, no: int, eval_fn: Callable[[tuple[int, ...]], tuple[tuple[int, ...], ...]],  # создать оракул; описание в docstring ниже
                 max_pac_reps: int = 120, alpha_step: float = 0.001, min_pac_m: int = 10):  # продолжение сигнатуры: параметры фильтра шума
        self.ni, self.no, self._eval_fn = ni, no, eval_fn  # запомнить разрядности и функцию ответа
        self.queries = self.resets = self.pac_samples = 0  # счётчики запросов, сбросов и выборок — нули
        self.cache: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}; self.pac_mode = False  # кэш последовательность -> ответ; режим фильтрации шума выключен
        self.trie: dict[tuple[int, ...], list[list[int]]] = {}; self.locked: dict[tuple[int, ...], tuple[int, ...]] = {}  # trie: префикс -> счётчики [число 0, число 1] по каждому выходному биту; locked — зафиксированные ответы
        self.alpha_step = alpha_step; self.max_pac_reps = max_pac_reps; self.min_pac_m = min_pac_m  # параметры фильтра шума
        self._thresh = _binom_thresh_table(alpha_step, min_pac_m, max_pac_reps * 2)  # таблица порогов большинства для m от min_pac_m до 2·max_pac_reps

    @classmethod  # метод класса (создаёт оракул по таблице)
    def from_rows(cls, ns: int, ni: int, no: int, rows: list[tuple[int, ...]], reset: int = 0) -> "InteractiveMealyOracle":
        """Создать оракул по таблице переходов автомата Мили.
        
                    ns, ni, no: разрядности, rows: таблица строк, reset: начальное состояние.
                    Возвращает InteractiveMealyOracle, отвечающий по таблице."""
        def _fn(seq: tuple[int, ...]):
            """Пройти последовательность от reset: ответ — кортеж выходов по шагам.
            
                            s — текущее состояние; для каждого u берётся строка s | (u << ns), выход — биты после состояния."""
            s = reset; outs = []  # начальное состояние; накопитель выходов
            for u in seq: r = rows[s | (u << ns)]; s = sum(v << i for i, v in enumerate(r[:ns])); outs.append(tuple(r[ns:]))  # по каждому входу: строка таблицы; новое состояние = Σ v_i·2^i по битам состояния; выход — остальные биты
            return tuple(outs)  # кортеж выходов
        return cls(ni, no, _fn)  # создать оракул с этой функцией ответа

    def _raw_eval(self, seq: tuple[int, ...]):
        """Сырой запрос: один сброс, ответ eval_fn, число запросов += длина последовательности.
        
                Возвращает кортеж выходов."""
        self.resets += 1; outs = self._eval_fn(seq); self.queries += len(seq); return outs  # сбросов +1; получить ответ; запросов + len(seq)

    def _update_trie(self, seq: tuple[int, ...], raw_outs: tuple[tuple[int, ...], ...]):
        """Обновить счётчики префиксов последовательности и зафиксировать биты, где большинство значимо.
        
                    seq: последовательность входов, raw_outs: полученные выходы. Для каждого префикса и каждого бита
                    считается число нулей c0 и единиц c1; большинство maj = max(c0, c1); требуется maj ≥ req,
                    где req — порог из таблицы (или m + 1, если хвост P(X ≥ maj) > alpha_step). Когда все биты
                    префикса прошли — ответ фиксируется как большинство."""
        thresh = self._thresh; tlen = len(thresh); min_m = self.min_pac_m  # пороги, их длина и минимум повторений
        for i in range(len(seq)):  # по каждому префиксу seq[:i + 1]
            pref = seq[:i + 1]; cnts = self.trie.setdefault(pref, [[0, 0] for _ in range(self.no)])  # счётчики префикса (по умолчанию нули для каждого бита)
            for b in range(self.no): cnts[b][raw_outs[i][b]] += 1  # для каждого бита увеличить счётчик полученного значения
            if pref not in self.locked:  # если префикс ещё не зафиксирован
                ok = True  # все биты прошли порог
                for c0, c1 in cnts:  # по каждому биту (c0 — нули, c1 — единицы)
                    m_tot = c0 + c1  # общее число наблюдений m_tot = c0 + c1
                    if m_tot < min_m: ok = False; break  # мало наблюдений — бит ещё не готов
                    maj = c0 if c0 >= c1 else c1  # большинство maj = max(c0, c1)
                    req = thresh[m_tot] if m_tot < tlen else (m_tot + 1 if binom_p_ge(m_tot, maj) > self.alpha_step else maj)  # порог: из таблицы, если m_tot < tlen; иначе m_tot + 1, если хвост ещё велик, иначе maj
                    if maj < req: ok = False; break  # большинство ниже порога — бит не готов
                if ok: self.locked[pref] = tuple(1 if c1 > c0 else 0 for c0, c1 in cnts)  # все биты готовы — зафиксировать ответ префикса как большинство

    def eval_seq(self, seq: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        """Выполнить последовательность входов и вернуть выходы.
        
                    В обычном режиме ответ берётся из кэша или получается одним сбросом; все префиксы
                    кэшируются и проверяются на детерминированность. В режиме pac_mode повторяет запрос
                    до max_pac_reps раз, пока все префиксы не будут зафиксированы; иначе UnrecoverableNoiseError."""
        if not self.pac_mode:  # обычный режим
            if seq in self.cache: return self.cache[seq]  # ответ в кэше — вернуть его
            outs = self._raw_eval(seq)  # сырой запрос
            for i in range(1, len(seq) + 1):  # по всем префиксам
                pref = seq[:i]  # очередной префикс
                if pref in self.cache and self.cache[pref] != outs[:i]:  # префикс уже в кэше с другим ответом — оракул недетерминирован
                    raise NonDeterministicOracleError(f"non-deterministic output on prefix {pref} at reset {self.resets}")  # ошибка недетерминированности
                self.cache[pref] = outs[:i]  # запомнить ответ префикса
            return outs  # вернуть ответ
        if all(seq[:i + 1] in self.locked for i in range(len(seq))):  # режим фильтрации шума: все префиксы зафиксированы
            return tuple(self.locked[seq[:i + 1]] for i in range(len(seq)))  # вернуть зафиксированные ответы префиксов
        for _ in range(self.max_pac_reps):  # до max_pac_reps повторений
            self._update_trie(seq, self._raw_eval(seq)); self.pac_samples += 1  # обновить счётчики; выборок +1
            if all(seq[:i + 1] in self.locked for i in range(len(seq))):  # все префиксы зафиксированы
                return tuple(self.locked[seq[:i + 1]] for i in range(len(seq)))  # вернуть зафиксированные ответы префиксов
        raise UnrecoverableNoiseError(f"Exact Binomial separation failed after {self.max_pac_reps} reps on seq len {len(seq)}")  # исчерпаны повторения — ошибка: шум не отфильтрован

    def eval_match(self, seq: tuple[int, ...], expected: tuple[tuple[int, ...], ...]) -> tuple[bool, tuple[int, ...] | None]:
        """Проверить соответствие гипотезы: совпадает ли ответ оракула с ожидаемым.
        
                    seq: последовательность входов, expected: ожидаемые выходы. Возвращает (True, None) при
                    совпадении, иначе (False, контрпример — префикс до первого расхождения). В режиме pac_mode
                    сначала проверяются уже зафиксированные префиксы, затем выполняются повторные запросы."""
        if not self.pac_mode:  # обычный режим
            got = self.eval_seq(seq)  # ответ оракула
            if got == expected: return True, None  # полное совпадение
            for i in range(len(seq)):  # по каждому шагу
                if got[i] != expected[i]: return False, seq[:i + 1]  # первое расхождение — контрпример (префикс до него)
            return True, None  # расхождений нет
        for i in range(len(seq)):  # режим фильтрации шума: по каждому префиксу
            pref = seq[:i + 1]  # очередной префикс
            if pref in self.locked and self.locked[pref] != expected[i]: return False, pref  # зафиксированный ответ не совпал с ожидаемым — контрпример
        if all(seq[:i + 1] in self.locked for i in range(len(seq))): return True, None  # все префиксы зафиксированы и совпали
        for _ in range(self.max_pac_reps):  # до max_pac_reps повторений
            self._update_trie(seq, self._raw_eval(seq)); self.pac_samples += 1  # обновить счётчики; выборок +1
            for i in range(len(seq)):  # по каждому префиксу
                pref = seq[:i + 1]  # очередной префикс
                if pref in self.locked and self.locked[pref] != expected[i]: return False, pref  # зафиксированный ответ не совпал — контрпример
            if all(seq[:i + 1] in self.locked for i in range(len(seq))): return True, None  # все зафиксировано и совпало
        raise UnrecoverableNoiseError(f"Exact Binomial separation failed after {self.max_pac_reps} reps on seq len {len(seq)}")  # исчерпаны повторения — ошибка

def eval_quotient_seq(T, seq: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Прогнать последовательность через таблицу переходов T от состояния 0.
    
            Возвращает кортеж выходов по шагам."""
    c = 0; out = []  # начальное состояние 0; накопитель выходов
    for u in seq: c, o = T[c][u]; out.append(o)  # по каждому входу: переход и выход
    return tuple(out)  # кортеж выходов

def run_seq_state_T(T, seq: tuple[int, ...]) -> int:
    """Конечное состояние после последовательности входов (от состояния 0).
    
            Возвращает номер состояния."""
    c = 0  # начальное состояние 0
    for u in seq: c, _ = T[c][u]  # по каждому входу — только переход
    return c  # конечное состояние

def build_PW_and_Wi(k: int, ni: int, T):
    """Классический метод Wp: префиксы состояний P, набор различающих слов W и наборы W_i для каждого состояния.
    
            k: число состояний, ni: число входных битов, T: таблица переходов.
            P[c] — кратчайший путь от 0 до c (обход в ширину). W строится итеративно: состояния делятся на блоки
            по векторам выходов на словах из W; для первого блока с более чем одним состоянием ищется слово,
            различающее пару его состояний (обход в ширину по парам), и добавляется в W, пока все блоки
            не станут одноэлементными. W_i — жадно выбранные слова из W, разделяющие состояние i с остальными.
            Возвращает (P, W, W_i)."""
    P = {0: ()}; q = deque([0])  # P: состояние -> путь от 0; очередь обхода
    while q:  # пока очередь не пуста
        c = q.popleft()  # взять состояние c
        for u in range(1 << ni):  # по всем входам
            nc, _ = T[c][u]  # переход из c
            if nc not in P: P[nc] = P[c] + (u,); q.append(nc)  # новое состояние: путь = путь до c + (u,)
    if k == 1: W = [(u,) for u in range(1 << ni)]; return [P[0]], W, {0: W}  # одно состояние: все однобуквенные слова; вернуть P = [()], W, W_0 = W
    def ev_w(c: int, w: tuple[int, ...]):
        """Выходы состояния c на слове w.
        
                        Возвращает кортеж выходов."""
        cur = c; out = []  # текущее состояние; накопитель выходов
        for u in w: cur, o = T[cur][u]; out.append(o)  # по каждому входу слова: переход и выход
        return tuple(out)  # кортеж выходов
    W = []  # W — пустой список различающих слов
    while True:  # пока есть блоки с более чем одним состоянием
        blocks: dict = {}  # blocks: вектор выходов на словах W -> список состояний
        for c in range(k): blocks.setdefault(tuple(ev_w(c, w) for w in W), []).append(c)  # для каждого состояния вычислить вектор выходов на W и добавить в блок
        big = [b for b in blocks.values() if len(b) > 1]  # big — блоки с более чем одним состоянием
        if not big: break  # одноэлементных блоков нет — все состояния различимы
        p, q_st = big[0][0], big[0][1]; pq = deque([(p, q_st, ())]); seen = {(p, q_st)}; found = None  # первая пара (p, q) из первого большого блока; поиск в ширину по парам состояний с префиксом
        while pq and found is None:  # пока очередь не пуста и слово не найдено
            a, b, pref = pq.popleft()  # взять пару (a, b) и префикс
            for u in range(1 << ni):  # по всем входам
                na, oa = T[a][u]; nb, ob = T[b][u]  # переходы пары
                if oa != ob: found = pref + (u,); break  # выходы различаются — различающее слово: pref + (u,)
                if (na, nb) not in seen: seen.add((na, nb)); pq.append((na, nb, pref + (u,)))  # новая пара состояний — добавить в очередь с расширенным префиксом
        if found not in W: W.append(found)  # добавить найденное слово в W
    Wi = {}  # W_i — наборы слов для каждого состояния
    for i in range(k):  # по каждому состоянию i
        rem = [j for j in range(k) if j != i]; chosen = []  # rem — состояния, которые ещё нужно отделить от i; chosen — выбранные слова
        while rem:  # пока есть неотделённые
            best_w = max(W, key=lambda w: (sum(1 for j in rem if ev_w(j, w) != ev_w(i, w)) / max(1, len(w)), -len(w)))  # лучшее слово: максимум доли отделяемых состояний (на единицу длины), при равенстве — короче
            sig_i = ev_w(i, best_w); rem = [j for j in rem if ev_w(j, best_w) == sig_i]; chosen.append(best_w)  # оставить только состояния с тем же ответом, что у i; запомнить слово
        Wi[i] = chosen  # W_i[i] = выбранные слова
    return [P[c] for c in range(k)], W, Wi  # вернуть (P по состояниям, W, W_i)

def build_hsi_adt(k: int, ni: int, T):  # построение адаптивного дерева различения; описание в docstring ниже
    """Точное дерево различения состояний минимальной стоимости и адаптивный лес HSI.
    
            k: число состояний, ni: число входных битов, T: таблица переходов.
            Для малых автоматов — точный поиск с мемоизацией (минимум размера, затем глубины),
            иначе — жадное адаптивное дерево с эвристиками. Возвращает (P, список деревьев, H_wp)."""
    P_map = {0: ()}; q = deque([0])  # P_map: состояние -> путь от 0; очередь обхода
    while q:  # пока очередь не пуста
        c = q.popleft()  # взять состояние c
        for u in range(1 << ni):  # по всем входам
            nc, _ = T[c][u]  # переход из c
            if nc not in P_map: P_map[nc] = P_map[c] + (u,); q.append(nc)  # новое состояние: путь = путь до c + (u,)
    P = [P_map[c] for c in range(k)]  # P — пути для всех состояний 0..k−1
    if k <= 1: W = [(u,) for u in range(1 << ni)]; return P, [{0: w} for w in W], {0: W}  # одно состояние: по дереву на каждый вход; HSI — все входы
    all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]  # all_pairs — все пары состояний i < j

    if k <= 24 and (1 << ni) <= 4:  # точный поиск только для небольших автоматов (k ≤ 24, входов ≤ 4)
        memo = {}; visiting = set()  # кэш и множество разбираемых множеств (защита от циклов)
        def solve_exact(cur_set: frozenset[int], d_left: int):
            """Точный поиск: разделить множество состояний префиксом длины не более d_left с минимальной стоимостью.
            
                            Стоимость = (общий размер, глубина). Возвращает (размер, глубина, карта состояние -> путь)
                            или None, если в пределах глубины не удалось."""
            if len(cur_set) <= 1: return (0, 0, {c: () for c in cur_set})  # одно состояние или пусто: стоимость 0, путь пуст
            if d_left == 0 or cur_set in visiting: return None  # глубина исчерпана или уже разбирается — отказ
            key = (cur_set, d_left)  # ключ кэша
            if key in memo: return memo[key]  # результат из кэша
            visiting.add(cur_set); best = None  # пометить множество как разбираемое; лучший результат не найден
            for u in range(1 << ni):  # по каждому входному символу u
                by_out = {}; merged = False  # разбиение по выходам; признак слияния
                for c in cur_set:  # по каждому состоянию множества
                    nc, o = T[c][u]; grp = by_out.setdefault(o, set())  # переход и выход; группа состояний с этим выходом
                    if nc in grp: merged = True; break  # два состояния перешли в одно (слияние) — этот символ не различает
                    grp.add(nc)  # добавить состояние в группу
                if merged or (len(by_out) == 1 and frozenset(next(iter(by_out.values()))) == cur_set): continue  # слияние или одна группа без движения — символ бесполезен
                ok = True; wd = 0; total_sz = 0; sub_map = {}  # все подзадачи решены; глубина wd, суммарный размер, карты детей
                for o, grp in by_out.items():  # по группам-выходам
                    res = solve_exact(frozenset(grp), d_left - 1)  # решить подзадачу для группы с глубиной d_left − 1
                    if res is None: ok = False; break  # не удалось — символ не годится
                    d_sub, sz_sub, child_map = res; wd = max(wd, 1 + d_sub); total_sz += len(grp) + sz_sub; sub_map[o] = child_map  # глубина = max(1 + глубина ребенка); размер += len(группа) + размер ребенка; запомнить карту
                if ok:  # все подзадачи решены
                    cand = (total_sz, wd, {c: (u,) + sub_map[T[c][u][1]][T[c][u][0]] for c in cur_set})  # кандидат: путь для c = (u,) + путь ребёнка для состояния, куда перешло c
                    if best is None or (cand[0], cand[1]) < (best[0], best[1]): best = cand  # лучше по (размер, глубина) — запомнить
            visiting.remove(cur_set); memo[key] = best; return best  # снять пометку; сохранить в кэш; вернуть лучший
        best_res = first_d = None  # лучший результат и первая глубина, на которой нашлось
        for d_lim in range(1, min(k + 2, 10)):  # по пределам глубины от 1 до min(k + 2, 10) − 1
            res = solve_exact(frozenset(range(k)), d_lim)  # решить для всех состояний
            if res is not None:  # решение есть
                if best_res is None or (res[0], res[1]) < (best_res[0], best_res[1]): best_res = res  # лучше по (размер, глубина) — запомнить
                if first_d is None: first_d = d_lim  # первая глубина, на которой нашлось решение
                if d_lim >= first_d + 1: break  # на один уровень глубже тоже посмотрели — стоп
        if best_res is not None: paths = best_res[2]; return P, [paths], {s: [paths[s]] for s in range(k)}  # решение найдено: одно дерево-карта путей; HSI — путь каждого состояния; вернуть

    def shortest_sep_word(ca: int, cb: int):
        """Кратчайшее слово, различающее состояния ca и cb (обход в ширину по парам).
        
                    Возвращает слово; если состояния неразличимы — RuntimeError."""
        pq = deque([(ca, cb, ())]); vis = {(ca, cb)}  # очередь пар с префиксом; просмотренные пары
        while pq:  # пока очередь не пуста
            x, y, pref = pq.popleft()  # взять пару (x, y) и префикс
            for u in range(1 << ni):  # по всем входам
                nx, ox = T[x][u]; ny, oy = T[y][u]  # переходы пары
                if ox != oy: return pref + (u,)  # выходы различаются — различающее слово
                if nx != ny and (nx, ny) not in vis: vis.add((nx, ny)); pq.append((nx, ny, pref + (u,)))  # новая пара состояний — в очередь с расширенным префиксом
        raise RuntimeError(f"States {ca}, {cb} indistinguishable")  # состояния неразличимы — ошибка

    def build_adaptive_tree(pref: tuple[int, ...], target_pairs: set[tuple[int, int]]):
        """Построить адаптивное дерево, различающее заданные пары состояний.
        
                pref: префикс (форсированное начало), target_pairs: множество пар, которые нужно различить.
                Возвращает карту состояние -> слово (пустое, если состояние не участвует)."""
        def split_node(state_map, rem_pairs, seen_configs, depth, forced_seq):
            """Рекурсивное разбиение узла: state_map — исходное состояние -> текущее, rem_pairs — оставшиеся пары,
                        seen_configs — уже разобранные конфигурации, depth — глубина, forced_seq — навязанный префикс.
                        Возвращает карту исходное состояние -> слово от этого узла."""
            if not rem_pairs or depth >= 2 * k: return {s: () for s in state_map}  # пар не осталось или глубина исчерпана — пустые пути
            if forced_seq:  # если есть навязанный префикс
                u0 = forced_seq[0]; by_out = {}; sub_pairs = {}  # первый символ префикса; группы по выходам; пары, оставшиеся внутри групп
                for orig_s, cur_s in state_map.items(): nc, o = T[cur_s][u0]; by_out.setdefault(o, {})[orig_s] = nc  # по каждому состоянию: переход по u0; группа по выходу
                for (a, b) in rem_pairs:  # по оставшимся парам
                    na, oa = T[state_map[a]][u0]; nb, ob = T[state_map[b]][u0]  # переходы обоих состояний пары
                    if oa == ob and na != nb: sub_pairs.setdefault(oa, set()).add((a, b))  # выход равен, состояния разные — пара остаётся внутри группы
                paths = {}  # пути
                for o, grp in by_out.items():  # по группам-выходам
                    sp = sub_pairs.get(o, set())  # пары этой группы
                    if not sp:  # пар нет — путь (u0,)
                        for s in grp: paths[s] = (u0,)  # каждому состоянию группы — путь (u0,)
                    else:  # иначе
                        sub = split_node(grp, sp, seen_configs, depth + 1, forced_seq[1:])  # рекурсия для группы с оставшимся навязанным префиксом
                        for s in grp: paths[s] = (u0,) + sub[s]  # путь = (u0,) + путь ребёнка
                return paths  # вернуть пути
            cfg = frozenset((min(state_map[a], state_map[b]), max(state_map[a], state_map[b])) for a, b in rem_pairs)  # конфигурация: множество неупорядоченных пар текущих состояний
            if cfg in seen_configs: return {s: () for s in state_map}  # конфигурация уже разбирались — пустые пути
            seen_configs = seen_configs | {cfg}; best_score = best_u = best_groups = None  # добавить конфигурацию; лучшие счёт, символ и группы пока не найдены
            for u in range(1 << ni):  # по каждому входному символу u
                by_o = {}; sp = {}; sep = mrg = 0  # группы по выходам, оставшиеся пары по группам; счётчики разделённых и слитых пар
                for (a, b) in rem_pairs:  # по каждой оставшейся паре
                    na, oa = T[state_map[a]][u]; nb, ob = T[state_map[b]][u]  # переходы пары
                    if oa != ob: sep += 1  # выходы различаются — пара разделена
                    elif na == nb: mrg += 1  # переходы слились — пара слита
                    else: sp.setdefault(oa, set()).add((a, b))  # иначе пара остаётся внутри группы своего выхода
                for orig_s, cur_s in state_map.items(): nc, o = T[cur_s][u]; by_o.setdefault(o, {})[orig_s] = nc  # по каждому состоянию: переход по u; группа по выходу
                next_cfg = frozenset((min(by_o[o][a], by_o[o][b]), max(by_o[o][a], by_o[o][b])) for o, s_p in sp.items() for a, b in s_p)  # следующая конфигурация (пары внутри групп после шага)
                if sep == 0 and (not next_cfg or next_cfg in seen_configs): continue  # символ ничего не разделяет и ведёт в разобранную конфигурацию — пропустить
                sc = (sep - mrg * 100, sep, -mrg, len(by_o))  # счёт символа: (разделено − 100·слито, разделено, −слито, число групп)
                if best_score is None or sc > best_score: best_score, best_u, best_groups = sc, u, (by_o, sp)  # лучший счёт — запомнить символ и группы
            if best_u is None or (best_score[1] == 0 and best_score[0] < 0):  # подходящего символа нет (или только слияния)
                best_w = min((shortest_sep_word(state_map[a], state_map[b]) for a, b in rem_pairs), key=len)  # взять кратчайшее различающее слово для одной из оставшихся пар
                u0 = best_w[0]; by_o = {}; sp = {}  # его первый символ; группы по выходам и оставшиеся пары
                for orig_s, cur_s in state_map.items(): nc, o = T[cur_s][u0]; by_o.setdefault(o, {})[orig_s] = nc  # по каждому состоянию: переход по u0; группа по выходу
                for (a, b) in rem_pairs:  # по оставшимся парам
                    na, oa = T[state_map[a]][u0]; nb, ob = T[state_map[b]][u0]  # переходы пары
                    if oa == ob and na != nb: sp.setdefault(oa, set()).add((a, b))  # выход равен, состояния разные — пара внутри группы
                next_cfg = frozenset((min(by_o[o][a], by_o[o][b]), max(by_o[o][a], by_o[o][b])) for o, s_p in sp.items() for a, b in s_p)  # следующая конфигурация
                if next_cfg not in seen_configs: best_u, best_groups = u0, (by_o, sp)  # конфигурация новая — использовать этот символ
                else: return {s: best_w for s in state_map}  # иначе — раздать всем найденное различающее слово
            by_out, sub_pairs = best_groups; paths = {}  # выбранные группы и пары; пути
            for o, grp in by_out.items():  # по группам-выходам
                sp = sub_pairs.get(o, set())  # пары группы
                if not sp:  # пар нет
                    for s in grp: paths[s] = (best_u,)  # путь каждого состояния группы — (выбранный символ,)
                else:  # иначе
                    sub = split_node(grp, sp, seen_configs, depth + 1, ())  # рекурсия для группы без навязанного префикса
                    for s in grp: paths[s] = (best_u,) + sub[s]  # путь = (символ,) + путь ребёнка
            return paths  # вернуть пути
        active_orig = tuple(sorted({s for p in target_pairs for s in p}))  # состояния, участвующие в заданных парах
        tp = split_node({s: s for s in active_orig}, set(target_pairs), set(), 0, pref)  # разбиение от начальной конфигурации (каждое состояние — само себе)
        return {s: tp.get(s, ()) for s in range(k)}  # карта для всех состояний (пустой путь, если не участвует)

    cands = []; seen_cands = set()  # кандидаты и множество их сигнатур
    prefixes = [()] + ([p for L in range(1, 5 if ni == 1 else 3) for p in itertools.product(range(1 << ni), repeat=L)] if (1 << ni) <= 4 else [(u,) for u in range(1 << ni)])  # префиксы: пустой и (для входов ≤ 4) все слова длины 1..4 (при ni = 1) или 1..2, иначе однобуквенные
    all_set = set(all_pairs)  # множество всех пар
    for pref in prefixes:  # по каждому префиксу
        tp = build_adaptive_tree(pref, all_set); sig = tuple(tp[s] for s in range(k))  # дерево с этим префиксом; сигнатура — кортеж путей состояний
        if sig not in seen_cands: seen_cands.add(sig); cands.append(tp)  # новая сигнатура — добавить в кандидаты
        if len(pref) >= 2:  # для префиксов длины ≥ 2
            tp_f = {s: pref for s in range(k)}; sig_f = tuple(tp_f[s] for s in range(k))  # кандидат «одинаковое слово для всех» = сам префикс
            if sig_f not in seen_cands: seen_cands.add(sig_f); cands.append(tp_f)  # новая сигнатура — добавить
    _, W_leg, _ = build_PW_and_Wi(k, ni, T)  # различающие слова классического метода
    for w in W_leg:  # по каждому слову W
        tp_w = {s: w for s in range(k)}; sig_w = tuple(tp_w[s] for s in range(k))  # кандидат «одно слово для всех»
        if sig_w not in seen_cands: seen_cands.add(sig_w); cands.append(tp_w)  # новая сигнатура — добавить

    def tree_pair_depths(tp):
        """Глубины разделения пар в дереве tp: на каком шаге слова пути состояний i и j расходятся по выходу.
        
                Возвращает словарь (i, j) -> номер шага (с 1)."""
        d = {}  # результат
        for i, j in all_pairs:  # по всем парам
            ci, cj = i, j  # текущие состояния
            for step, (u1, u2) in enumerate(zip(tp[i], tp[j]), 1):  # по шагам слов путей (нумерация с 1), пока символы совпадают
                if u1 != u2: break  # символы разошлись — стоп
                ci, oi = T[ci][u1]; cj, oj = T[cj][u2]  # переходы обоих состояний
                if oi != oj: d[(i, j)] = step; break  # выходы различаются — пара разделена на этом шаге
        return d  # вернуть глубины

    cand_covs = [tree_pair_depths(tp) for tp in cands]  # покрытие пар каждым кандидатом
    uncov = all_set - (set().union(*[set(c.keys()) for c in cand_covs]) if cand_covs else set())  # непокрытые пары = все − объединение покрытий
    while uncov:  # пока есть непокрытые пары
        tp = build_adaptive_tree((), uncov); d_map = tree_pair_depths(tp)  # дерево для непокрытых пар и его покрытие
        if not set(d_map.keys()) & uncov:  # дерево ничего не покрыло
            a, b = next(iter(uncov)); w = shortest_sep_word(a, b); tp = {s: w for s in range(k)}; d_map = tree_pair_depths(tp)  # взять кратчайшее различающее слово первой пары; кандидат — одно слово для всех
        cands.append(tp); cand_covs.append(d_map); uncov -= set(d_map.keys())  # добавить кандидата и его покрытие; убрать покрытые

    cand_singletons = [sum(1 for q in range(k) if all(((q, j) if q < j else (j, q)) in dm for j in range(k) if j != q)) for dm in cand_covs]  # для каждого кандидата: число состояний, которые он отделяет от всех остальных
    chosen_idxs = []; rem_pairs = set(all_pairs)  # выбранные кандидаты и оставшиеся пары
    while rem_pairs:  # пока есть пары
        best_idx = max(range(len(cands)), key=lambda idx: (  # лучший кандидат по составному ключу
            cand_singletons[idx] if any(all(((q, x) if q < x else (x, q)) in cand_covs[idx] for x in range(k) if x != q)  # приоритет кандидату, отделяющему какое-то состояние от всех остальных и покрывающему оставшиеся пары
                                        and any(((q, j) if q < j else (j, q)) in rem_pairs for j in range(k) if j != q) for q in range(k)) else 0,  # продолжение условия
            len(rem_pairs & set(cand_covs[idx].keys())), len(cand_covs[idx]), -sum(len(cands[idx][s]) for s in range(k))))  # далее: число покрытых оставшихся пар, общее покрытие, минус суммарная длина слов
        chosen_idxs.append(best_idx); rem_pairs -= set(cand_covs[best_idx].keys())  # добавить кандидата; убрать покрытые им пары

    chosen_trees = [cands[idx] for idx in chosen_idxs]; chosen_covs = [cand_covs[idx] for idx in chosen_idxs]; H_wp = {}  # выбранные деревья и их покрытия; H_wp — наборы слов для каждого состояния
    for q in range(k):  # по каждому состоянию q
        need = {j for j in range(k) if j != q}; chosen_q = {}  # need — состояния, которые нужно отделить от q; chosen_q — дерево -> глубина
        while need:  # пока есть неотделённые
            best_r = None; best_score = -1.0; best_L = 0; best_cov = set()  # лучшие дерево, счёт, глубина и покрытие
            for r in range(len(chosen_trees)):  # по выбранным деревьям
                cov_j = {j for j in need if ((q, j) if q < j else (j, q)) in chosen_covs[r]}  # состояния из need, разделённые с q этим деревом
                if not cov_j: continue  # ничего — дальше
                L = max(chosen_covs[r][(q, j) if q < j else (j, q)] for j in cov_j); sc = len(cov_j) / (L + 2)  # L — максимальная глубина разделения; счёт = число разделённых / (L + 2)
                if sc > best_score: best_score, best_r, best_L, best_cov = sc, r, L, cov_j  # лучший счёт — запомнить
            chosen_q[best_r] = best_L; need -= best_cov  # запомнить дерево и глубину; убрать покрытые
        H_wp[q] = [chosen_trees[r][q][:L] for r, L in sorted(chosen_q.items())]  # H_wp[q] — префиксы слов дерева q длины L
    return P, chosen_trees, H_wp  # вернуть (P, выбранные деревья, H_wp)

def dedup_seqs(seqs: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Удалить из списка последовательности, являющиеся префиксами других (длинные вперёд).
    
            Возвращает список без дубликатов и префиксов уже взятых."""
    seen = set(); res = []  # просмотренные префиксы; результат
    for seq in sorted(set(seqs), key=len, reverse=True):  # по уникальным последовательностям по убыванию длины
        if seq not in seen:  # последовательность не является префиксом уже взятой
            res.append(seq)  # добавить в результат
            for i in range(1, len(seq) + 1): seen.add(seq[:i])  # все её префиксы отметить как просмотренные
    return res  # вернуть результат

@functools.lru_cache(maxsize=512)  # кэш на 512 наборов аргументов
def build_wp_test_suite(k: int, ni: int, T, max_states: int = 8, use_adt: bool = True) -> tuple[tuple[int, ...], ...]:
    """Тестовый набор метода Wp с ограничением на число состояний гипотезы.
    
            k: число состояний гипотезы, ni: число входных битов, T: таблица, max_states: максимум состояний
            проверяемой машины (добавляются «прогулки» длины extra + 1), use_adt: использовать адаптивные деревья.
            Возвращает кортеж тестовых последовательностей (более короткий из классического и адаптивного наборов)."""
    extra = max(0, max_states - k)  # extra — на сколько состояний больше допускается
    if ((1 << ni) ** (extra + 1)) * max(1, k) <= 2048: extra_cap, extra_walks = extra, []  # если полный перебор слов длины extra + 1 укладывается в 2048 — берётся он
    else:  # иначе
        extra_cap = min(extra, (2 if k >= 10 else 5) if ni == 1 else 1)  # предел длины промежуточных слов снижается (1–5 в зависимости от k и ni)
        rng = random.Random(0xBEEF + k * 131 + ni)  # генератор случайных чисел с детерминированным зерном
        extra_walks = [tuple(rng.randrange(1 << ni) for _ in range(max(24, max_states + 6))) for _ in range(48)]  # 48 случайных «прогулок» длины max(24, max_states + 6)
    mids = [()] + [v for L in range(1, extra_cap + 2) for v in itertools.product(range(1 << ni), repeat=L)]  # промежуточные слова v: пустое и все слова длины 1..extra_cap + 1
    P_l, W_l, Wi_l = build_PW_and_Wi(k, ni, T)  # классические P, W, W_i
    leg_seqs = dedup_seqs([p + w for p in P_l for w in W_l] + [p + v + w for p in P_l for v in mids if v for w in Wi_l[run_seq_state_T(T, p + v)]] + extra_walks)  # классические последовательности: p + w и p + v + w (w из W_i состояния после p + v) плюс прогулки
    if not use_adt: return tuple(leg_seqs)  # без адаптивных деревьев — вернуть классический набор
    P, forest, H_q = build_hsi_adt(k, ni, T)  # адаптивные префиксы, лес и H_q
    adt_seqs = dedup_seqs([P[i] + forest[r][i] for i in range(k) for r in range(len(forest))] +  # адаптивные последовательности: P[i] + путь дерева для i
                          [p + v + w for p in P for v in mids if v for w in H_q[run_seq_state_T(T, p + v)]] + extra_walks)  # плюс p + v + w (w из H_q состояния после p + v) и прогулки
    return tuple(leg_seqs if sum(len(s) for s in leg_seqs) < sum(len(s) for s in adt_seqs) else adt_seqs)  # вернуть более короткий по суммарной длине набор

def wp_method_verify(k: int, ni: int, T, bb: InteractiveMealyOracle, max_states: int = 8, use_adt: bool = True):
    """Проверка гипотезы методом Wp: все тестовые последовательности сравниваются с оракулом.
    
            Возвращает (True, None) или (False, контрпример)."""
    for seq in build_wp_test_suite(k, ni, T, max_states=max_states, use_adt=use_adt):  # по всем тестовым последовательностям
        ok_s, cex = bb.eval_match(seq, eval_quotient_seq(T, seq))  # сравнить ответ оракула с ответом гипотезы
        if not ok_s: return False, cex  # расхождение — контрпример
    return True, None  # гипотеза подтверждена

def lstar_learn_mealy(bb: InteractiveMealyOracle, max_states: int = 8, use_adt: bool = True):
    """Активное обучение автомата Мили алгоритмом L* (таблица наблюдений с префиксами S и суффиксами E).
    
            bb: оракул, max_states: подсказка о числе состояний, use_adt: адаптивные деревья в проверке Wp.
            Цикл: замыкание таблицы (все префиксы S·u различимы, все одинаковые строки различимы по E),
            построение гипотезы, поиск контрпримера в кэше, быстрыми пробами и методом Wp; обработка
            контрпримера бинарным поиском (Ривест–Шапир) — добавление суффикса в E или префиксов в S.
            Возвращает (k, T)."""
    ni = bb.ni; alphabet = list(range(1 << ni)); S = [()]; E = [(u,) for u in alphabet]; row_cache: dict = {}  # алфавит 0..2^ni−1; S — префиксы (пустой), E — суффиксы (по одной букве); кэш строк
    def batch_eval(seqs):
        """Пакетно выполнить последовательности (после удаления префиксов-дубликатов)."""
        for seq in dedup_seqs(seqs): bb.eval_seq(seq)  # по последовательностям без префиксов-дубликатов
    def row(s: tuple[int, ...]):
        """Стока наблюдений префикса s: ответы на s + e для всех e из E (хвост после s).
        
                        Кэш по ключу (s, число суффиксов)."""
        key = (s, len(E))  # ключ кэша
        if key in row_cache: return row_cache[key]  # результат из кэша
        if bb.pac_mode: batch_eval([s + e for e in E])  # в режиме шума — сначала пакетно опросить s + e
        res = tuple(bb.eval_seq(s + e)[len(s):] for e in E); row_cache[key] = res; return res  # ответы на s + e без первых len(s) выходов; запомнить

    probe_len = max(12, max_states + 2)  # длина быстрых проб
    quick_probes = ([(1,) * probe_len, (0,) * probe_len, (0, 1) * (probe_len // 2), (1, 0) * (probe_len // 2),  # быстрые пробы для одного входного бита: периодические слова и две псевдослучайные маски
                     (1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1)[:probe_len], (0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1)[:probe_len]] if ni == 1  # продолжение списка проб
                    else [tuple([u] * probe_len) for u in alphabet] + [tuple((i + j) % (1 << ni) for i in range(probe_len)) for j in alphabet])  # для ni ≥ 2: постоянные слова и слова с нарастающим счётчиком

    while True:  # внешний цикл: пока гипотеза подтверждена
        while True:  # внутренний цикл: замыкание таблицы
            if bb.pac_mode: batch_eval([s + e for s in S for e in E] + [s + (u,) + e for s in S for u in alphabet for e in E])  # в режиме шума — пакетно опросить все s + e и s + u + e
            sr = {s: row(s) for s in S}; urows = {}  # строки префиксов; urows — строка -> первый префикс с ней
            for s in S:  # по префиксам S
                if sr[s] not in urows: urows[sr[s]] = s  # запомнить первый префикс с такой строкой
            added = False  # добавлений не было
            for s in list(S):  # по префиксам S (копия)
                for u in alphabet:  # по символам алфавита
                    if row(s + (u,)) not in urows: S.append(s + (u,)); added = True; break  # строка s + u новая — добавить префикс s + u; отметить добавление
                if added: break  # прервать перебор символов
            if added: continue  # было добавление — повторить замыкание
            by_row: dict = {}  # группы префиксов с одинаковыми строками
            for s in S: by_row.setdefault(sr[s], []).append(s)  # распределить префиксы по строкам
            inc = False  # увеличений E не было
            for grp in by_row.values():  # по группам
                if len(grp) < 2: continue  # одноэлементная группа — пропустить
                s1 = grp[0]  # первый префикс группы
                for s2 in grp[1:]:  # по остальным префиксам группы
                    for u in alphabet:  # по символам алфавита
                        r1, r2 = row(s1 + (u,)), row(s2 + (u,))  # строки s1 + u и s2 + u
                        if r1 != r2:  # строки различаются
                            for idx, e in enumerate(E):  # по номерам суффиксов
                                if r1[idx] != r2[idx] and ((u,) + e) not in E: E.append((u,) + e); inc = True; break  # первое расхождение: добавить (u,) + e в E; отметить увеличение
                        if inc: break  # прервать перебор суффиксов
                    if inc: break  # прервать перебор символов
                if inc: break  # прервать перебор префиксов группы
            if not inc: break  # без увеличений — таблица замкнута
        sr = {s: row(s) for s in S}; rep = []; r2id = {}  # строки префиксов; rep — представители состояний; r2id — строка -> номер состояния
        for s in S:  # по префиксам S
            if sr[s] not in r2id: r2id[sr[s]] = len(rep); rep.append(s)  # новая строка — новый представитель с очередным номером
        k = len(rep); T = tuple(tuple((r2id[row(s + (u,))], bb.eval_seq(s + (u,))[-1]) for u in alphabet) for s in rep)  # k — число состояний гипотезы; T[s][u] = (номер состояния строки s + u, последний выход s + u)

        cex = None  # контрпример не найден
        if bb.pac_mode:  # режим шума
            for seq in sorted(bb.locked.keys(), key=len):  # по зафиксированным последовательностям по возрастанию длины
                if seq and bb.locked[seq] != eval_quotient_seq(T, seq)[-1]: cex = seq; break  # зафиксированный ответ не совпал с гипотезой — контрпример
        else:  # обычный режим
            for seq in sorted(bb.cache.keys(), key=len):  # по кэшированным последовательностям по возрастанию длины
                if not seq: continue  # пустую пропустить
                exp_seq = eval_quotient_seq(T, seq)  # ответ гипотезы
                if bb.cache[seq] != exp_seq:  # кэш не совпал с гипотезой
                    for i in range(len(seq)):  # по шагам
                        if bb.cache[seq][i] != exp_seq[i]: cex = seq[:i + 1]; break  # первое расхождение — контрпример (префикс до него)
                    break  # прервать
        if cex is None and (k <= 2 or ni >= 2):  # быстрые пробы для малых k или ni ≥ 2
            for seq in quick_probes:  # по пробам
                ok_s, cex = bb.eval_match(seq, eval_quotient_seq(T, seq))  # сравнить пробу с гипотезой
                if not ok_s: break  # контрпример — стоп
        if cex is None:  # контрпримера нет
            ok, cex = wp_method_verify(k, ni, T, bb, max_states=max(max_states, k), use_adt=use_adt)  # проверка методом Wp
            if ok: return k, T  # подтверждено — вернуть (k, T)

        def hyp_match(i: int) -> bool:
            """Проверить гипотезу на суффиксе контрпримера: r_s + suff, где r_s — представитель состояния после cex[:i].
            
                        Возвращает True, если оракул согласен с гипотезой."""
            q_i = run_seq_state_T(T, cex[:i]); r_s = rep[q_i]; suff = cex[i:]  # состояние гипотезы после cex[:i]; его представитель; суффикс
            exp_full = eval_quotient_seq(T, r_s); exp_suff = []; c = q_i  # ожидаемые выходы: exp_full — на r_s, exp_suff — на суффиксе от состояния q_i
            for u in suff: c, o = T[c][u]; exp_suff.append(o)  # по символам суффикса: переход и выход
            return bb.eval_match(r_s + suff, exp_full + tuple(exp_suff))[0]  # сравнение оракула с ожиданием

        if not hyp_match(len(cex) - 1):  # гипотеза неверна уже на последнем символе: добавить все префиксы контрпримера в S
            for j in range(1, len(cex) + 1):  # по длинам префиксов
                if cex[:j] not in S: S.append(cex[:j])  # добавить префикс в S
        else:  # иначе — бинарный поиск границы Ривеста–Шапира
            lo, hi = 0, len(cex) - 1  # lo — неверная граница, hi — верная
            while lo + 1 < hi:  # пока границы не сошлись
                mid = (lo + hi) // 2  # середина
                if not hyp_match(mid): lo = mid  # в середине неверно — сдвинуть lo
                else: hi = mid  # иначе сдвинуть hi
            e_new = cex[lo + 1:]  # новый суффикс cex[lo + 1:]
            if e_new not in E: E.append(e_new)  # добавить его в E
            else:  # иначе
                s_new = rep[run_seq_state_T(T, cex[:lo])] + (cex[lo],)  # новый префикс: представитель состояния после cex[:lo] + символ cex[lo]
                if s_new not in S: S.append(s_new)  # добавить в S
                else:  # иначе
                    for j in range(1, len(cex) + 1):  # по длинам префиксов
                        if cex[:j] not in S: S.append(cex[:j])  # добавить префикс в S


# ---------------------------------------------------------------------------
# 6. Агент: обучение, композиция, решение целей и сохранение состояния
# ---------------------------------------------------------------------------

class AutonomousTemporalAgent:
    """Автономный агент: библиотека навыков, обучение примитивов, композиция и решение целей.
    
            issuer: издатель контрактов на степень, min_warmup_trials: число точных попыток для включения модели,
            use_adt: адаптивные деревья в проверке Wp, use_factored_synthesis: синтез с разложением на множители."""
    def __init__(self, issuer: str = DEFAULT_ISSUER, min_warmup_trials: int = 3,  # создать агента; описание в docstring ниже
                 use_adt: bool = True, use_factored_synthesis: bool = True):  # продолжение сигнатуры
        self.issuer = issuer; self.use_adt = use_adt; self.use_factored_synthesis = use_factored_synthesis  # запомнить издателя и режимы
        self.learner = ProofCarryingANFLearner({issuer}, max_full_support_inputs=14)  # обучающий с одним доверенным издателем и пределом полного перебора 14 переменных
        self.lib = Library(); self.model = SelfModel(min_warmup_trials); self.trace: list[dict] = []  # библиотека, модель себя, журнал трассировки

    def _compile_anf(self, n: int, anf: tuple[frozenset[int], ...]) -> dict:
        """Скомпилировать АНФ в схему и записать статистику синтеза.
        
                n: число переменных, anf: многочлены. Если включён синтез с разложением — берётся наименьшая
                из проверенных схем, иначе сумма произведений. Возвращает схему."""
        sop = compact_netlist(self.learner.compile(n, anf))  # базовая схема: сумма произведений, уплотнённая
        if self.use_factored_synthesis:  # если разрешён синтез с разложением
            opt = self.learner.compile_optimal(n, anf); self.model.record_synthesis(len(sop["gates"]), len(opt["gates"])); return opt  # оптимальная схема; записать (базовый размер, выбранный размер); вернуть её
        self.model.record_synthesis(len(sop["gates"]), len(sop["gates"])); return sop  # иначе записать (базовый, базовый) и вернуть сумму произведений

    def build_skill_from_rows(self, name: str, ns: int, ni: int, no: int, rows: list[tuple[int, ...]],  # Построить навык по таблице строк с выбором наименьшей схемы.
                              route: str, op: str = "PRIMITIVE", parents: tuple[str, ...] = (),  # продолжение сигнатуры
                              queries: int = 0, prefer_mdl: bool = True, generation: int = 0) -> Skill:  # конец сигнатуры
        n = ns + ni; anf = mobius(rows, n); ns_min, k, bdig, qT, min_rows, _ = minimize_mealy_from_rows(ns, ni, no, rows, 0)  # число переменных; АНФ таблицы; минимизация автомата (ns_min, k, хеш bdig, фактор-таблица qT, её строки)
        cands = [("FLAT", ns, rows, anf, self._compile_anf(n, anf))]  # первый кандидат — FLAT
        if prefer_mdl:  # если разрешён кандидат по фактор-автомату
            if self.use_factored_synthesis:  # с синтезом с разложением
                opt_ns, opt_rows, opt_anf, opt_net = optimal_encode_quotient(k, ni, no, qT)  # оптимальные коды состояний: разрядность, строки, АНФ, схема
                sop_m = compact_netlist(self.learner.compile(ns_min + ni, mobius(min_rows, ns_min + ni)))  # базовая схема для минимального автомата
                self.model.record_synthesis(len(sop_m["gates"]), len(opt_net["gates"]))  # записать статистику синтеза
                cands.append(("OPT_QUOTIENT", opt_ns, opt_rows, opt_anf, opt_net))  # добавить кандидата OPT_QUOTIENT
            else:  # иначе (без синтеза с разложением)
                min_anf = mobius(min_rows, ns_min + ni)  # АНФ минимального автомата
                cands.append(("MIN_QUOTIENT", ns_min, min_rows, min_anf, compact_netlist(self.learner.compile(ns_min + ni, min_anf))))  # добавить кандидата MIN_QUOTIENT
        mode, b_ns, b_rows, b_anf, b_net = min(cands, key=lambda c: (len(c[4]["gates"]), c[4]["metrics"]["depth"], c[1], c[4]["metrics"]["program_bits"]))  # выбрать кандидата по (вентили, глубина, разрядность состояния, биты программы)
        if not symbolic_verify_general(b_net, b_anf): raise RuntimeError(f"{name}: symbolic verify fail")  # символьная проверка выбранной схемы обязательна
        return Skill(name, b_ns, ni, no, 0, b_rows, b_anf, b_net, route, op, parents, queries, mode, k, ns_min, bdig, qT, True, None, generation)  # создать навык (0 — начальное состояние, bisim_verified = True, без сертификата)

    def learn_primitive(self, name: str, ns: int, ni: int, no: int, oracle_fn: Callable[[int], tuple[int, ...]],  # Обучить примитивную функцию по оракулу и добавить в библиотеку.
                        degree_bound: int | None = None, contract: DegreeContract | None = None,  # продолжение сигнатуры
                        allow_full_support: bool = True, generation: int = 0) -> Skill | None:  # конец сигнатуры
        n = ns + ni  # число переменных
        if contract is None and degree_bound is not None: contract = DegreeContract(n, degree_bound, self.issuer, 2)  # если контракта нет, но задан предел степени — создать контракт от имени агента (версия 2)
        c, dec = self.learner.fit(n, ns + no, oracle_fn, contract=contract, allow_full_support=allow_full_support, audit_full_support=True)  # обучение с аудитом по всем точкам
        if c is None:  # отказ — удержание
            self.model.log_alert("PRIMITIVE_LEARN_HOLD", f"{name}: {dec.get('reason')}", True)  # предупреждение в модели
            self.trace.append({"name": name, "op": "PRIMITIVE", "route": "HOLD", "queries": 0, "audit_rows": 0, "action": "HOLD", "generation": generation})  # запись в трассировку с маршрутом HOLD
            return None  # вернуть None
        if dec.get("degree_contract_rejected"):  # если контракт на степень был отвергнут и обучение перешло к полному перебору
            self.model.log_alert("FALSE_DEGREE_BOUND_ESCALATED", f"{name}: escalated to {c.query_count} full-support queries", True)  # предупреждение о ложной границе степени
        rows = [tuple(int(v) & 1 for v in oracle_fn(x)) for x in range(1 << n)]  # полная таблица выходов оракула по всем входам
        s = self.build_skill_from_rows(name, ns, ni, no, rows, c.mode.upper(), "PRIMITIVE", (), c.query_count, True, generation)  # построить навык (режим обучения — маршрут)
        target, added = self.lib.add(s)  # добавить в библиотеку
        self.trace.append({"name": name, "op": "PRIMITIVE", "route": s.route, "queries": c.query_count, "audit_rows": len(rows),  # запись в трассировку
                           "action": "COMMIT" if added else "REUSE", "parents": [], "nvars": s.nvars, "expert": target,  # продолжение: действие COMMIT или REUSE, без родителей, число переменных, цель
                           "compile_mode": s.compile_mode, "k_states": s.k_states, "global_gates_after": len(self.lib.dag.gates), "generation": generation})  # продолжение: способ компиляции, число состояний, число вентилей глобальной схемы, поколение
        return self.lib.skills[target]  # вернуть навык из библиотеки

    def compose(self, op: str, pa: Skill, pb: Skill | None, name: str, pc: Skill | None = None, lut4: int = 0,  # Композиция навыков: собрать новую схему из существующих с проверками и выбором наименьшей схемы.
                route_override: str | None = None, extra_queries: int = 0, generation: int = 0) -> Skill:  # продолжение сигнатуры
        pnames = (pa.name,) if pb is None else ((pa.name, pb.name) if pc is None else (pa.name, pb.name, pc.name))  # имена родителей (1, 2 или 3)
        key = digest({"op": op, "lut4": lut4, "parents": [pa.behav_digest or pa.semantic_digest(),  # ключ кэша: хеш операции, lut4 и хешей поведения родителей
                      None if pb is None else (pb.behav_digest or pb.semantic_digest()),  # продолжение: хеш второго родителя
                      None if pc is None else (pc.behav_digest or pc.semantic_digest())]})  # продолжение: хеш третьего родителя
        if key in self.lib.cache:  # композиция уже вычислялась
            old = self.lib.skills[self.lib.resolve(self.lib.cache[key])]  # существующий навык
            alias = Skill(name, old.ns, old.ni, old.no, old.reset, old.rows, old.anf, old.net, "MACRO_REUSE", op, pnames, 0,  # навык-синоним с маршрутом MACRO_REUSE
                          old.compile_mode, old.k_states, old.ns_min, old.behav_digest, old.quotient_T, True, old.ag_cert, generation)  # продолжение: параметры существующего навыка
            resolved, _ = self.lib.add(alias)  # добавить синоним в библиотеку
            self.trace.append({"name": name, "op": op, "route": "MACRO_REUSE", "queries": 0, "audit_rows": 0, "action": "REUSE", "parents": list(pnames), "expert": resolved, "nvars": old.nvars, "generation": generation})  # запись в трассировку
            return self.lib.skills[resolved]  # вернуть существующий навык

        if op == "CASCADE":  # каскад
            if pa.no != pb.ni: raise ValueError(f"cascade port mismatch {pa.no}!={pb.ni}")  # порты: no(A) = ni(B)
            ns, ni, no, raw_hier = pa.ns + pb.ns, pa.ni, pb.no, compose_net_cascade(pa.net, pa.ns, pa.ni, pb.net, pb.ns, pb.ni)  # разрядности и сырая иерархическая схема
        elif op == "PARALLEL_PAIR":  # параллельная пара
            if pa.ni != pb.ni: raise ValueError("parallel pair ni mismatch")  # порты: равные ni
            ns, ni, no, raw_hier = pa.ns + pb.ns, pa.ni, pa.no + pb.no, compose_net_parallel_pair(pa.net, pa.ns, pb.net, pb.ns, pa.ni)  # разрядности и сырая схема
        elif op == "PARALLEL_LUT":  # параллельная пара с LUT
            if pa.ni != pb.ni or pa.no != 1 or pb.no != 1: raise ValueError("parallel lut port mismatch")  # порты: равные ni, по одному выходу
            ns, ni, no, raw_hier = pa.ns + pb.ns, pa.ni, 1, compose_net_parallel_lut(pa.net, pa.ns, pb.net, pb.ns, pa.ni, lut4)  # разрядности и сырая схема
        elif op == "MUX_2IN":  # мультиплексор 2 в 1
            if not (pa.ni == pb.ni == 1 and pa.no == pb.no == 1): raise ValueError("mux_2in port mismatch")  # порты: ni = 1, no = 1 у обоих
            ns, ni, no, raw_hier = pa.ns + pb.ns, 2, 1, compose_net_mux_2in(pa.net, pa.ns, pb.net, pb.ns)  # разрядности и сырая схема
        elif op == "MOE_MUX":  # «смесь экспертов»
            if not (pa.ni == pb.ni == pc.ni and pa.no == pb.no == pc.no == 1): raise ValueError("moe port mismatch")  # порты: равные ni, по одному выходу
            ns, ni, no, raw_hier = pa.ns + pb.ns + pc.ns, pa.ni, 1, compose_net_moe_mux(pa.net, pa.ns, pb.net, pb.ns, pc.net, pc.ns, pa.ni)  # разрядности и сырая схема
        elif op == "DELAYED_FEEDBACK":  # задержанная обратная связь
            if pa.ni != 2 or pa.no != 1: raise ValueError("feedback requires ni=2, no=1")  # порты: ni = 2, no = 1
            ns, ni, no, raw_hier = pa.ns + 1, 1, 1, compose_net_delayed_feedback(pa.net, pa.ns)  # разрядности и сырая схема (состояние + 1 бит задержки)
        elif op == "COUPLED_FEEDBACK_2BIT":  # связанная обратная связь
            if not (pa.ni == pb.ni == 2 and pa.no == pb.no == 1): raise ValueError("coupled 2bit requires two ni=2,no=1")  # порты: ni = 2, no = 1 у обоих
            ns, ni, no, raw_hier = pa.ns + pb.ns + 2, 1, 1, compose_net_coupled_fb_2bit(pa.net, pa.ns, pb.net, pb.ns)  # разрядности и сырая схема (состояния + 2 бита задержки)
        else: raise ValueError(op)  # неизвестная операция

        n = ns + ni  # число переменных
        if n > AUDIT_CAP:  # за пределом полного аудита — модульная композиция с доказательством
            ok_ag, msg_ag = verify_ag_raw(op, raw_hier, pa, pb, pc, lut4)  # проверка сырой схемы по операции
            if not ok_ag: raise RuntimeError(f"{name}: AG fail: {msg_ag}")  # не прошла — ошибка
            opt_net, comp_proof = compact_netlist_with_proof(raw_hier)  # уплотнение с доказательством
            ok_cp, msg_cp = verify_compaction_proof(raw_hier, opt_net, comp_proof)  # проверка доказательства уплотнения
            if not ok_cp: raise RuntimeError(f"{name}: compaction proof fail: {msg_cp}")  # не прошла — ошибка
            ag_cert = {"kind": f"assume-guarantee-compacted-{op.lower()}", "parents": list(pnames),  # сертификат: вид, родители
                       "raw_gates": len(raw_hier["gates"]), "compacted_gates": len(opt_net["gates"]), "proof_steps": len(comp_proof["proof_steps"])}  # размеры до и после уплотнения, число шагов доказательства
            s = Skill(name, ns, ni, no, 0, None, None, opt_net, route_override or "BEYOND_AUDIT_MODULAR_AG", op, pnames, extra_queries,  # навык без таблицы и АНФ (None), маршрут BEYOND_AUDIT_MODULAR_AG или заданный
                      "MODULAR_AG_COMPACT", 0, ns, "AG:" + digest({"op": op, "parents": list(pnames), "lut4": lut4}), None, False, ag_cert, generation)  # продолжение: режим MODULAR_AG_COMPACT, 0 состояний, ns, хеш «AG:» + хеш композиции, без фактор-таблицы, бисимуляция не проверялась
            resolved, added = self.lib.add(s, flat_gate_count=len(raw_hier["gates"])); self.lib.cache[key] = resolved  # добавить в библиотеку с размером плоской схемы; запомнить в кэше
            self.trace.append({"name": name, "op": op, "route": s.route, "queries": extra_queries, "audit_rows": 0,  # запись в трассировку
                               "action": "COMMIT" if added else "REUSE", "parents": list(pnames), "nvars": n, "expert": resolved,  # продолжение: действие, родители, число переменных, цель
                               "compile_mode": "MODULAR_AG_COMPACT", "chosen_gates": len(opt_net["gates"]), "global_gates_after": len(self.lib.dag.gates), "generation": generation})  # продолжение: режим, выбранные вентили, глобальная схема, поколение
            return self.lib.skills[resolved]  # вернуть навык

        hier_net = compact_netlist(raw_hier); cand_anf, rows = _compose_anf_and_rows(op, pa, pb, pc, lut4)  # уплотнённая иерархическая схема; АНФ и строки композиции
        if not symbolic_verify_general(hier_net, cand_anf): raise RuntimeError(f"{name}: hierarchical symbolic fail")  # символьная проверка иерархической схемы обязательна
        expected = mobius(rows, n); bound = max((deg(p) for p in expected), default=0)  # ожидаемая АНФ по таблице и граница степени
        warm = (not self.model.enabled) and (op == "CASCADE") and (route_override is None); queries = extra_queries  # разогрев: модель не включена, каскад без переопределения маршрута; queries — дополнительные запросы
        if warm:  # если разогрев
            c, dec = self.learner.fit(n, ns + no, lambda x: rows[x], contract=DegreeContract(n, bound, self.issuer, 2), allow_full_support=True, audit_full_support=False)  # обучение по контракту с границей степени (полный перебор разрешён, аудит выключен)
            if c is None: raise RuntimeError(f"warmup synthesis failed {name}: {dec}")  # обучение не удалось — ошибка
            exact = tuple(c.outputs) == cand_anf; self.model.observe(exact, c.query_count); queries += c.query_count  # точно ли обучение совпало с АНФ композиции; наблюдение в модели; учёт запросов
            learned, route = (tuple(c.outputs), "CASCADE_WARMUP") if exact else (expected, "EXACT_FALLBACK")  # разогрев: обученная АНФ и маршрут CASCADE_WARMUP, иначе ожидаемая АНФ и EXACT_FALLBACK
        else:  # иначе
            learned, route = cand_anf, route_override or f"CERTIFIED_{op}"  # АНФ композиции; маршрут заданный или CERTIFIED_<op>
        if learned != expected: raise RuntimeError(f"{name}: candidate ANF mismatch")  # обученная АНФ обязана совпасть с ожидаемой

        ns_min, k, bdig, qT, min_rows, _ = minimize_mealy_from_rows(ns, ni, no, rows, 0)  # минимизация автомата композиции
        if bdig in self.lib.by_behav:  # хеш поведения уже есть в библиотеке
            old = self.lib.skills[self.lib.by_behav[bdig]]  # существующий навык
            alias = Skill(name, old.ns, old.ni, old.no, old.reset, old.rows, old.anf, old.net, "BEHAVIORAL_REUSE", op, pnames, queries,  # навык-синоним с маршрутом BEHAVIORAL_REUSE
                          old.compile_mode, old.k_states, old.ns_min, bdig, old.quotient_T, True, old.ag_cert, generation)  # продолжение: параметры существующего навыка
            resolved, _ = self.lib.add(alias); self.lib.cache[key] = resolved  # добавить синоним; запомнить в кэше
            self.trace.append({"name": name, "op": op, "route": "BEHAVIORAL_REUSE", "queries": queries, "audit_rows": len(rows), "action": "REUSE", "parents": list(pnames), "nvars": n, "expert": resolved, "generation": generation})  # запись в трассировку
            return self.lib.skills[resolved]  # вернуть существующий навык

        flat_sop = compact_netlist(self.learner.compile(n, learned)); flat_net = self._compile_anf(n, learned)  # плоская сумма произведений и схема после выбора
        if self.use_factored_synthesis:  # с синтезом с разложением
            opt_ns, opt_rows, opt_anf, opt_net = optimal_encode_quotient(k, ni, no, qT)  # оптимальные коды состояний
            min_sop = compact_netlist(self.learner.compile(ns_min + ni, mobius(min_rows, ns_min + ni)))  # базовая схема для минимального автомата
            self.model.record_synthesis(len(min_sop["gates"]), len(opt_net["gates"]))  # записать статистику синтеза
            q_cand = ("OPT_QUOTIENT", opt_ns, opt_rows, opt_anf, opt_net)  # кандидат OPT_QUOTIENT
        else:  # иначе
            min_anf = mobius(min_rows, ns_min + ni)  # АНФ минимального автомата
            q_cand = ("MIN_QUOTIENT", ns_min, min_rows, min_anf, compact_netlist(self.learner.compile(ns_min + ni, min_anf)))  # кандидат MIN_QUOTIENT
        mode, b_ns, b_rows, b_anf, b_net = min([("HIERARCHICAL", ns, rows, learned, hier_net), ("FLAT", ns, rows, learned, flat_net), q_cand],  # выбор среди HIERARCHICAL, FLAT и фактор-автомата по (вентили, глубина, разрядность, биты)
                                               key=lambda c: (len(c[4]["gates"]), c[4]["metrics"]["depth"], c[1], c[4]["metrics"]["program_bits"]))  # продолжение ключа выбора
        if not symbolic_verify_general(b_net, b_anf): raise RuntimeError(f"{name}: chosen netlist {mode} symbolic fail")  # символьная проверка выбранной схемы обязательна
        deg_a = max((deg(p) for p in pa.anf), default=0); deg_b = max((deg(p) for p in pb.anf), default=0) if pb is not None else 1  # максимальные степени АНФ родителей
        self.model.record_mdl((deg_a * deg_b >= 3) or (len(hier_net["gates"]) < len(flat_sop["gates"])), mode, len(flat_sop["gates"]), len(hier_net["gates"]), len(q_cand[4]["gates"]))  # записать предсказание маршрутизации: иерархия выгодна при произведении степеней ≥ 3 или меньшем размере иерархии

        s = Skill(name, b_ns, ni, no, 0, b_rows, b_anf, b_net, route, op, pnames, queries, mode, k, ns_min, bdig, qT, True, None, generation)  # создать навык
        resolved, added = self.lib.add(s, flat_gate_count=len(flat_sop["gates"])); self.lib.cache[key] = resolved  # добавить в библиотеку с размером плоской схемы; запомнить в кэше
        self.trace.append({"name": name, "op": op, "route": route, "queries": queries, "audit_rows": len(rows), "action": "COMMIT" if added else "REUSE",  # запись в трассировку
                           "parents": list(pnames), "nvars": n, "compile_mode": mode, "chosen_gates": len(b_net["gates"]), "k_states": k,  # продолжение: родители, число переменных, режим, выбранные вентили, число состояний
                           "global_gates_after": len(self.lib.dag.gates), "generation": generation})  # продолжение: глобальная схема, поколение
        return self.lib.skills[resolved]  # вернуть навык

    def solve_goal(self, name: str, oracle: InteractiveMealyOracle, max_states_hint: int = 12,  # Решить цель: найти или собрать навык с поведением заданного оракула.
                   allow_pac_denoise: bool = True, generation: int = 1) -> tuple[Skill | None, dict]:  # продолжение сигнатуры
        for _ in (0, 1):  # до двух попыток
            try:  # попытка
                return self._solve_goal_inner(name, oracle, max_states_hint, generation)  # решить цель (основная процедура)
            except NonDeterministicOracleError as e:  # недетерминированный оракул
                if not allow_pac_denoise or oracle.pac_mode:  # фильтр запрещён или уже включён
                    self.model.log_alert("NON_DETERMINISTIC_ORACLE_HOLD", str(e), True)  # предупреждение
                    self.trace.append({"name": name, "op": "GOAL_SOLVE", "route": "HOLD_NON_DETERMINISTIC", "queries": oracle.queries, "audit_rows": 0, "action": "HOLD", "reason": str(e), "generation": generation})  # запись в трассировку (удержание)
                    return None, {"solved_by": "HOLD_NON_DETERMINISTIC", "bb_ticks": oracle.queries}  # вернуть None с отчётом
                oracle.cache.clear(); oracle.pac_mode = True  # очистить кэш и включить режим фильтрации шума
            except UnrecoverableNoiseError as ue:  # неустранимый шум
                self.model.log_alert("UNRECOVERABLE_NOISE_HOLD", str(ue), True)  # предупреждение
                self.trace.append({"name": name, "op": "GOAL_SOLVE", "route": "HOLD_UNRECOVERABLE_NOISE", "queries": oracle.queries, "audit_rows": 0, "action": "HOLD", "reason": str(ue), "generation": generation})  # запись в трассировку
                return None, {"solved_by": "HOLD_UNRECOVERABLE_NOISE", "bb_ticks": oracle.queries}  # вернуть None с отчётом
        return None, {"solved_by": "HOLD", "bb_ticks": oracle.queries}  # исчерпаны попытки — удержание

    def _solve_goal_inner(self, name: str, bb: InteractiveMealyOracle, max_states_hint: int, generation: int) -> tuple[Skill | None, dict]:
        """Основная процедура решения цели (описание в docstring solve_goal).
        
                bb: оракул. Порядок: пробы; поиск в библиотеке навыка с тем же поведением; перебор семейств
                композиций (пары, обратные связи, каскады, LUT, мультиплексоры и т. д.) с проверкой гипотез;
                если ничего не подошло — активное обучение L*. Возвращает (навык, отчёт)."""
        ni, no = bb.ni, bb.no  # разрядности оракула
        probes = ([(0,)*8, (1,)*8, (0,1)*4, (1,0)*4, (1,0,1,1,0,1,0,0), (0,1,1,0,1,0,1,1), (1,1,1,0,0,0,1,0), (0,0,1,1,0,0,1,1,0,0,1,1)]  # пробные последовательности для одного входного бита (периодические и псевдослучайные)
                  if ni == 1 else [tuple([u] * 8) for u in range(1 << ni)] + [tuple((i + j) % (1 << ni) for i in range(8)) for j in range(1 << ni)])  # для ni ≥ 2: постоянные слова и слова с нарастающим счётчиком
        probe_obs = [(seq, bb.eval_seq(seq)) for seq in sorted(probes, key=len, reverse=True)]  # ответы оракула на пробы (длиннее — первыми)
        route_tag = "GOAL_PAC_DENOISED_SOLVED" if bb.pac_mode else "GOAL_COMPOSITION_SOLVED"  # метка маршрута решения

        def verify_T(k_cand: int, T_cand) -> tuple[bool, tuple[int, ...] | None]:
            """Проверить гипотезу T_cand с k_cand состояниями тестами Wp.
            
                        hint_m = max(подсказка, k_cand). С адаптивными деревьями — адаптивный набор, затем записывается
                        экономия относительно классического. Возвращает (True, None) или (False, контрпример)."""
            hint_m = max(max_states_hint, k_cand)  # предел числа состояний проверяемой машины
            if self.use_adt:  # с адаптивными деревьями
                adt_suite = build_wp_test_suite(k_cand, ni, T_cand, max_states=hint_m, use_adt=True)  # адаптивный тестовый набор
                for seq in adt_suite:  # по его последовательностям
                    ok_s, cex = bb.eval_match(seq, eval_quotient_seq(T_cand, seq))  # сравнить с оракулом
                    if not ok_s: return False, cex  # расхождение — контрпример
                leg_suite = build_wp_test_suite(k_cand, ni, T_cand, max_states=hint_m, use_adt=False)  # классический набор — для статистики
                self.model.record_adt(sum(len(s) for s in leg_suite), sum(len(s) for s in adt_suite))  # записать длины классического и адаптивного наборов
                return True, None  # гипотеза подтверждена
            return wp_method_verify(k_cand, ni, T_cand, bb, max_states=hint_m, use_adt=False)  # без адаптивных деревьев — классическая проверка Wp

        def build_probe_bitvec(obs, out_bit: int = 0):
            """Битовый вектор ответов оракула на пробы (по биту out_bit) и маска всех битов.
            
                        Возвращает (g_vec, (1 << total_bits) − 1)."""
            total_bits = sum(len(seq) for seq, _ in obs); g_vec = bit = 0  # общее число битов; вектор и номер бита
            for _, out in obs:  # по всем наблюдениям
                for o in out:  # по выходам последовательности
                    if o[out_bit]: g_vec |= 1 << bit  # если бит выхода равен 1 — установить бит вектора
                    bit += 1  # номер бита +1
            return g_vec, (1 << total_bits) - 1  # вернуть вектор и маску

        def skill_probe_vec(T, obs):
            """Битовый вектор ответов гипотезы T на те же пробы (первый выходной бит)."""
            v = bit = 0  # вектор и номер бита
            for seq, _ in obs:  # по наблюдениям
                for o in eval_quotient_seq(T, seq):  # по выходам гипотезы
                    if o[0]: v |= 1 << bit  # единица — установить бит
                    bit += 1  # номер бита +1
            return v  # вернуть вектор

        for s in list(self.lib.skills.values()):  # по всем навыкам библиотеки (копия)
            if s.ni == ni and s.no == no and s.quotient_T is not None and all(eval_quotient_seq(s.quotient_T, seq) == out for seq, out in probe_obs):  # та же арность, есть фактор-таблица и все пробы совпали
                ok, cex = verify_T(s.k_states, s.quotient_T)  # проверить гипотезу тестами Wp
                if ok:  # подтверждено
                    self.model.record_family_hit("LIBRARY_REUSE")  # записать попадание семейства «повторное использование библиотеки»
                    alias = Skill(name, s.ns, s.ni, s.no, s.reset, s.rows, s.anf, s.net, "GOAL_LIBRARY_REUSE", "REUSE", (s.name,), bb.queries, s.compile_mode, s.k_states, s.ns_min, s.behav_digest, s.quotient_T, True, s.ag_cert, generation)  # навык-синоним с маршрутом GOAL_LIBRARY_REUSE и родителем — найденным навыком
                    res, _ = self.lib.add(alias)  # добавить синоним в библиотеку
                    self.trace.append({"name": name, "op": "GOAL_SOLVE", "route": "GOAL_LIBRARY_REUSE", "queries": bb.queries, "audit_rows": 0, "action": "REUSE", "expert": res, "generation": generation})  # запись в трассировку
                    return self.lib.skills[res], {"solved_by": "LIBRARY_REUSE" + ("+PAC" if bb.pac_mode else ""), "bb_ticks": bb.queries, "pac_samples": bb.pac_samples}  # вернуть навык и отчёт (LIBRARY_REUSE, с +PAC в режиме шума)
                probe_obs.append((cex, bb.eval_seq(cex)))  # если гипотеза не подтвердилась — добавить контрпример и ответ оракула в пробы

        skills_list = [s for s in self.lib.skills.values() if s.quotient_T is not None and s.ns <= 6]  # навыки с фактор-таблицей и состоянием не шире 6 бит
        s11 = [s for s in skills_list if s.ni == ni and s.no == 1]; s21 = [s for s in skills_list if s.ni == 2 and s.no == 1]  # s11 — навыки с ni = 1, no = 1; s21 — с ni = 2, no = 1

        def try_commit(op_kind: str, parents_tuple: tuple[Skill, ...], lut4_val: int = 0, fam_tag: str | None = None):
            """Проверить композицию родителей тестами Wp и при успехе собрать навык.
            
                        op_kind: операция, parents_tuple: родители, lut4_val: таблица функции, fam_tag: имя семейства для статистики.
                        Возвращает (навык, отчёт) или None (контрпример добавляется в пробы)."""
            pa = parents_tuple[0]; pb = parents_tuple[1] if len(parents_tuple) > 1 else None; pc = parents_tuple[2] if len(parents_tuple) > 2 else None  # pa, pb, pc — родители по числу
            k_cand, _, T_cand = compose_quotient(op_kind, pa, pb, pc, lut4_val)  # фактор-автомат композиции: число состояний и таблица
            ok, cex = verify_T(k_cand, T_cand)  # проверка гипотезы
            if not ok: probe_obs.append((cex, bb.eval_seq(cex))); return None  # не подтвердилась — запомнить контрпример; вернуть None
            self.model.record_family_hit(fam_tag or op_kind)  # записать попадание семейства
            solved = self.compose(op_kind, pa, pb, name, pc=pc, lut4=lut4_val, route_override=route_tag, extra_queries=bb.queries, generation=generation)  # собрать навык через compose с маршрутом решения
            label = op_kind if op_kind != "PARALLEL_LUT" else f"PARALLEL_{lut4_name(lut4_val)}"  # метка операции (для LUT — имя функции)
            return solved, {"solved_by": f"COMPOSITION({label})" + ("+PAC" if bb.pac_mode else ""), "parents": [p.name for p in parents_tuple], "bb_ticks": bb.queries, "k_states": k_cand, "pac_samples": bb.pac_samples}  # вернуть навык и отчёт

        def run_family(fam: str):
            """Попробовать одно семейство композиций.
            
                    fam: имя семейства. Перебирает подходящие навыки библиотеки, для каждого проверяет пробы
                    (ответ композиции на пробных последовательностях), затем тесты Wp и собирает навык.
                    Возвращает (навык, отчёт) или None."""
            if fam == "PARALLEL_PAIR" and ni == 1 and no == 2:  # параллельная пара для цели с одним входом и двумя выходами
                g0, _ = build_probe_bitvec(probe_obs, 0); g1, _ = build_probe_bitvec(probe_obs, 1); sv = [skill_probe_vec(s.quotient_T, probe_obs) for s in s11]  # битовые векторы цели по выходам 0 и 1; векторы кандидатов
                for a in [s for i, s in enumerate(s11) if sv[i] == g0]:  # навыки a с вектором, равным вектору выхода 0
                    for b in [s for j, s in enumerate(s11) if sv[j] == g1]:  # навыки b с вектором, равным вектору выхода 1
                        if a.ns + b.ns <= 8 and (res := try_commit("PARALLEL_PAIR", (a, b), fam_tag=fam)) is not None: return res  # ограничение размера; проверить и собрать
            elif fam == "DELAYED_FEEDBACK" and ni == 1 and no == 1:  # задержанная обратная связь для цели с одним входом и выходом
                def ev_fb(Tm, seq):
                    """Ответ автомата с задержанной обратной связью на последовательность.
                    
                                    sm — состояние, d — задержанный выход; вход = u0 | (d << 1)."""
                    sm = d = 0; out = []  # начальные состояние и задержка; накопитель
                    for u0 in seq: sm, om = Tm[sm][u0 | (d << 1)]; d = om[0]; out.append(om)  # по каждому входу: шаг с входом u0 | (d << 1); d = первый бит выхода
                    return tuple(out)  # кортеж выходов
                for m in s21:  # по модулям с ni = 2, no = 1
                    if m.ns + 1 <= 10 and all(ev_fb(m.quotient_T, seq) == out for seq, out in probe_obs) and (res := try_commit("DELAYED_FEEDBACK", (m,), fam_tag=fam)) is not None: return res  # ограничение размера, совпадение проб; проверить и собрать
            elif fam == "COUPLED_FEEDBACK_2BIT" and ni == 1 and no == 1:  # связанная обратная связь
                def ev_cfb(Ta, Tb, seq):
                    """Ответ двух автоматов со связанной обратной связью на последовательность.
                    
                                    da, db — задержанные выходы; вход A = u0 | (db << 1), вход B = u0 | (da << 1); выход — B."""
                    sa = sb = da = db = 0; out = []  # начальные состояния и задержки; накопитель
                    for u0 in seq: sa, oa = Ta[sa][u0 | (db << 1)]; sb, ob = Tb[sb][u0 | (da << 1)]; da, db = oa[0], ob[0]; out.append(ob)  # по каждому входу: оба шага; задержки обновляются; выход — выход B
                    return tuple(out)  # кортеж выходов
                for a in s21:  # по a
                    for b in s21:  # по b
                        if a.ns + b.ns + 2 <= 8 and all(ev_cfb(a.quotient_T, b.quotient_T, seq) == out for seq, out in probe_obs) and (res := try_commit("COUPLED_FEEDBACK_2BIT", (a, b), fam_tag=fam)) is not None: return res  # ограничение размера, совпадение проб; проверить и собрать
            elif fam == "CASCADE" and no == 1:  # каскад для цели с одним выходом
                for a in [s for s in skills_list if s.ni == ni]:  # a — навыки с нужным ni
                    for b in skills_list:  # b — все ограниченные навыки
                        if a.no == b.ni and b.no == no and a.ns + b.ns <= 10:  # порты согласованы и размер в пределах
                            def ev_c(seq, Ta=a.quotient_T, Tb=b.quotient_T):
                                """Ответ каскада a → b на последовательность (вход b — выход a как число)."""
                                sa = sb = 0; out = []  # начальные состояния; накопитель
                                for u in seq: sa, o1 = Ta[sa][u]; sb, o2 = Tb[sb][sum(v << i for i, v in enumerate(o1))]; out.append(o2)  # по каждому входу: шаг a, затем шаг b с входом-числом; выход — выход b
                                return tuple(out)  # кортеж выходов
                            if all(ev_c(seq) == out for seq, out in probe_obs) and (res := try_commit("CASCADE", (a, b), fam_tag=fam)) is not None: return res  # совпадение проб; проверить и собрать
            elif fam == "PARALLEL_LUT" and ni == 1 and no == 1:  # параллельная пара с LUT
                g_vec, mask_all = build_probe_bitvec(probe_obs); sv = [skill_probe_vec(s.quotient_T, probe_obs) for s in s11]  # вектор цели и маска; векторы кандидатов
                for i, a in enumerate(s11):  # по a
                    va, na = sv[i], sv[i] ^ mask_all  # вектор a и его дополнение до маски
                    for j in range(i + 1, len(s11)):  # по b после a
                        b = s11[j]  # навык b
                        if a.ns + b.ns > 9: continue  # ограничение суммарного размера
                        vb, nb = sv[j], sv[j] ^ mask_all; ok_l, fixed_l, free_b = True, 0, []  # векторы b; ok_l — согласованность LUT; fixed_l — уже определённые биты таблицы; free_b — свободные
                        for bit, m in enumerate((na & nb, va & nb, na & vb, va & vb)):  # по четырём строкам таблицы (a, b): (0,0), (0,1), (1,0), (1,1): маски позиций проб с этими входами
                            if m == 0: free_b.append(bit)  # таких входов нет — бит таблицы свободен
                            elif (g_vec & m) == m: fixed_l |= 1 << bit  # везде цель даёт 1 — бит таблицы равен 1
                            elif (g_vec & m) != 0: ok_l = False; break  # цель даёт и 0, и 1 — противоречие
                        if ok_l:  # противоречий нет
                            luts = [fixed_l]  # варианты таблицы начинаются с определённых битов
                            for fb in free_b: luts = [l for x in luts for l in (x, x | (1 << fb))]  # каждый свободный бит — двумя значениями
                            for lc in sorted(luts, key=lambda l: (bin(l).count("1"), l)):  # по вариантам (меньше единиц, затем по номеру)
                                if lc not in (0, 15, 3, 5, 10, 12):  # таблица не должна быть вырожденной (константа, отрицание, проекция)
                                    if (res := try_commit("PARALLEL_LUT", (a, b), lut4_val=lc, fam_tag=fam)) is not None: return res  # проверить и собрать; при неудаче пробы могли пополниться — пересчитать вектор цели
                                    g_vec, mask_all = build_probe_bitvec(probe_obs)  # пересчитать вектор цели и маску
            elif fam == "PAIR2_TO_2IN" and ni == 1 and no == 1:  # семейство «модуль на два входа к одному» с одним входом и одним выходом
                small11 = [s for s in s11 if s.ns <= 2]  # малые навыки (состояние ≤ 2 бит)
                for m in s21:  # по модулям m с ni = 2
                    for a in small11:  # по a
                        for b in small11:  # по b
                            if a is b or a.ns + b.ns + m.ns > 8: continue  # разные навыки и ограничение размера
                            def ev_p2c(seq, Ta=a.quotient_T, Tb=b.quotient_T, Tm=m.quotient_T):
                                """Ответ композиции «пара → вход 2» на последовательность: m получает вход o1 | (o2 << 1)."""
                                sa = sb = sm = 0; out = []  # начальные состояния; накопитель
                                for u in seq: sa, o1 = Ta[sa][u]; sb, o2 = Tb[sb][u]; sm, om = Tm[sm][o1[0] | (o2[0] << 1)]; out.append(om)  # по каждому входу: шаги a и b, затем шаг m; выход — выход m
                                return tuple(out)  # кортеж выходов
                            if all(ev_p2c(seq) == out for seq, out in probe_obs):  # совпадение проб
                                _, _, Tab = compose_quotient("PARALLEL_PAIR", a, b)  # фактор-автомат пары
                                k_f, _, Tf = compose_quotient("CASCADE", Skill("__p", a.ns + b.ns, 1, 2, 0, None, None, {}, "", "", (), 0, "", 0, 0, "", Tab), m)  # фактор-автомат каскада пары и m (временный навык только с фактор-таблицей)
                                ok, cex = verify_T(k_f, Tf)  # проверка гипотезы
                                if ok:  # подтверждена
                                    self.model.record_family_hit(fam)  # записать попадание семейства
                                    pair_sk = self.compose("PARALLEL_PAIR", a, b, f"{name}_vec2", route_override="GOAL_PAIR2_INTERMEDIATE", generation=generation)  # собрать промежуточный навык — пару
                                    solved = self.compose("CASCADE", pair_sk, m, name, route_override="GOAL_PAIR2_CASCADE_SOLVED", extra_queries=bb.queries, generation=generation)  # собрать каскад пары и m
                                    return solved, {"solved_by": "COMPOSITION(PAIR2_TO_2IN)" + ("+PAC" if bb.pac_mode else ""), "parents": [a.name, b.name, m.name], "bb_ticks": bb.queries, "k_states": k_f, "pac_samples": bb.pac_samples}  # вернуть навык и отчёт
                                probe_obs.append((cex, bb.eval_seq(cex)))  # иначе — запомнить контрпример
            elif fam == "RECURRENT_MUX_FB" and ni == 1 and no == 1:  # семейство рекуррентного мультиплексора с обратной связью, один вход и один выход
                small11 = [s for s in s11 if s.ns <= 2]  # малые навыки (состояние ≤ 2 бит)
                for a in small11:  # по a
                    for b in small11:  # по b
                        if a is b or a.ns + b.ns > 4: continue  # разные навыки и ограничение размера
                        def ev_mf(seq, Ta=a.quotient_T, Tb=b.quotient_T):
                            """Ответ рекуррентного мультиплексора на последовательность: y = yb, если предыдущий y = 1, иначе ya."""
                            sa = sb = d = 0; out = []  # начальные состояния и задержанный выбор; накопитель
                            for u0 in seq: sa, oa = Ta[sa][u0]; sb, ob = Tb[sb][u0]; y = ob[0] if d else oa[0]; d = y; out.append((y,))  # по каждому входу: шаги a и b; выбор по задержанному y; y — новый выбор
                            return tuple(out)  # кортеж выходов
                        if all(ev_mf(seq) == out for seq, out in probe_obs):  # совпадение проб
                            _, _, Tmux = compose_quotient("MUX_2IN", a, b)  # фактор-автомат мультиплексора
                            k_fb, _, Tfb = compose_quotient("DELAYED_FEEDBACK", Skill("__m", a.ns + b.ns, 2, 1, 0, None, None, {}, "", "", (), 0, "", 0, 0, "", Tmux))  # фактор-автомат задержанной обратной связи мультиплексора (временный навык)
                            ok, cex = verify_T(k_fb, Tfb)  # проверка гипотезы
                            if ok:  # подтверждена
                                self.model.record_family_hit(fam)  # записать попадание семейства
                                mux_sk = self.compose("MUX_2IN", a, b, f"{name}_mux2", route_override="GOAL_MUX2IN_INTERMEDIATE", generation=generation)  # собрать промежуточный навык — мультиплексор
                                solved = self.compose("DELAYED_FEEDBACK", mux_sk, None, name, route_override="GOAL_RECURRENT_MUX_FB_SOLVED", extra_queries=bb.queries, generation=generation)  # собрать задержанную обратную связь мультиплексора
                                return solved, {"solved_by": "COMPOSITION(RECURRENT_MUX_FB)" + ("+PAC" if bb.pac_mode else ""), "parents": [a.name, b.name], "bb_ticks": bb.queries, "k_states": k_fb, "pac_samples": bb.pac_samples}  # вернуть навык и отчёт
                            probe_obs.append((cex, bb.eval_seq(cex)))  # иначе — запомнить контрпример
            elif fam == "MOE_MUX" and ni == 1 and no == 1:  # семейство мультиплексора смеси экспертов, один вход и один выход
                g_vec, mask_all = build_probe_bitvec(probe_obs); sv = [skill_probe_vec(s.quotient_T, probe_obs) for s in s11]  # «смесь экспертов»: вектор цели, маска, векторы кандидатов
                for i_s, sel in enumerate(s11):  # по переключателю sel
                    if sel.ns > 3: continue  # переключатель не шире 3 бит
                    vs, nvs = sv[i_s], sv[i_s] ^ mask_all  # вектор переключателя и его дополнение
                    va = [i for i in range(len(s11)) if i != i_s and s11[i].ns <= 3 and ((sv[i] ^ g_vec) & nvs) == 0]  # va — эксперты A: где переключатель = 0, ответ должен совпасть с целью
                    vb = [j for j in range(len(s11)) if j != i_s and s11[j].ns <= 3 and ((sv[j] ^ g_vec) & vs) == 0] if va else []  # vb — эксперты B: где переключатель = 1, ответ должен совпасть с целью
                    for i_a in va:  # по экспертам A
                        for i_b in vb:  # по экспертам B
                            if i_a != i_b and sel.ns + s11[i_a].ns + s11[i_b].ns <= 8 and (res := try_commit("MOE_MUX", (sel, s11[i_a], s11[i_b]), fam_tag=fam)) is not None: return res  # разные эксперты и ограничение размера; проверить и собрать
            elif fam == "DEPTH2_CASCADE" and ni == 1 and no == 1:  # каскад глубины 2
                small = [s for s in s11 if s.ns <= 2]  # малые навыки (состояние ≤ 2 бит)
                for a in small:  # по a
                    for b in small:  # по b
                        if a is b or a.ns + b.ns > 4: continue  # разные и ограничение размера
                        for c in small:  # по c
                            if a.ns + b.ns + c.ns > 6: continue  # ограничение суммарного размера
                            def ev_abc(seq, Ta=a.quotient_T, Tb=b.quotient_T, Tc=c.quotient_T):
                                """Ответ тройного каскада a → b → c на последовательность."""
                                sa = sb = sc = 0; out = []  # начальные состояния; накопитель
                                for u in seq: sa, o1 = Ta[sa][u]; sb, o2 = Tb[sb][o1[0]]; sc, o3 = Tc[sc][o2[0]]; out.append(o3)  # по каждому входу: шаги a, b (вход — выход a), c (вход — выход b); выход — выход c
                                return tuple(out)  # кортеж выходов
                            if all(ev_abc(seq) == out for seq, out in probe_obs):  # совпадение проб
                                _, _, Tab = compose_quotient("CASCADE", a, b)  # фактор-автомат каскада a и b
                                k_abc, _, Tabc = compose_quotient("CASCADE", Skill("__t", a.ns + b.ns, 1, 1, 0, None, None, {}, "", "", (), 0, "", 0, 0, "", Tab), c)  # фактор-автомат каскада с c (временный навык)
                                ok, cex = verify_T(k_abc, Tabc)  # проверка гипотезы
                                if ok:  # подтверждена
                                    self.model.record_family_hit(fam)  # записать попадание семейства
                                    mid = self.compose("CASCADE", a, b, f"{name}_mid", route_override="GOAL_DEPTH2_INTERMEDIATE", generation=generation)  # собрать промежуточный навык (a → b)
                                    solved = self.compose("CASCADE", mid, c, name, route_override="GOAL_DEPTH2_COMPOSITION_SOLVED", extra_queries=bb.queries, generation=generation)  # собрать каскад с c
                                    return solved, {"solved_by": "COMPOSITION(DEPTH2_CASCADE)" + ("+PAC" if bb.pac_mode else ""), "parents": [a.name, b.name, c.name], "bb_ticks": bb.queries, "k_states": k_abc, "pac_samples": bb.pac_samples}  # вернуть навык и отчёт
                                probe_obs.append((cex, bb.eval_seq(cex)))  # иначе — запомнить контрпример
            return None  # ничего не подошло

        default_order = ["PARALLEL_PAIR", "DELAYED_FEEDBACK", "COUPLED_FEEDBACK_2BIT", "CASCADE", "PARALLEL_LUT", "PAIR2_TO_2IN", "RECURRENT_MUX_FB", "MOE_MUX", "DEPTH2_CASCADE"]  # порядок семейств по умолчанию
        for fam in self.model.ranked_families(default_order):  # по семействам в ранжированном порядке
            if (res := run_family(fam)) is not None: return res  # первое подошедшее — вернуть результат

        k_l, T_l = lstar_learn_mealy(bb, max_states=max_states_hint, use_adt=self.use_adt)  # активное обучение L* (запасной путь)
        self.model.record_family_hit("LSTAR_FALLBACK")  # записать попадание семейства LSTAR_FALLBACK
        if bb.pac_mode: self.model.pac_lstar_solves += 1  # в режиме шума — учесть решение через L*
        ns_l, rows_l = quotient_to_rows(k_l, ni, no, T_l); lstar_route = "GOAL_PAC_LSTAR_LEARNED" if bb.pac_mode else "GOAL_LSTAR_LEARNED"  # таблица строк гипотезы; маршрут в зависимости от режима шума
        learned = self.build_skill_from_rows(name, ns_l, ni, no, rows_l, lstar_route, "LSTAR_BLACKBOX", (), bb.queries, True, generation)  # построить навык из таблицы (операция LSTAR_BLACKBOX)
        res, added = self.lib.add(learned)  # добавить в библиотеку
        self.trace.append({"name": name, "op": "LSTAR_BLACKBOX", "route": lstar_route, "queries": bb.queries, "audit_rows": len(rows_l),  # запись в трассировку
                           "action": "COMMIT" if added else "REUSE", "parents": [], "nvars": learned.nvars, "expert": res,  # продолжение: действие, без родителей, число переменных, цель
                           "k_states": k_l, "chosen_gates": len(learned.net["gates"]), "global_gates_after": len(self.lib.dag.gates), "generation": generation})  # продолжение: число состояний, выбранные вентили, глобальная схема, поколение
        return self.lib.skills[res], {"solved_by": "ACTIVE_STOCHASTIC_LSTAR_PAC" if bb.pac_mode else "ACTIVE_LSTAR_WP_FALLBACK",  # вернуть навык и отчёт
                                      "bb_ticks": bb.queries, "k_states": k_l, "pac_samples": bb.pac_samples}  # продолжение отчёта: запросы, число состояний, выборки шума

    def discover_novel(self, max_new: int = 8, seed: int = 0, generation: int = 2) -> list[Skill]:
        """Открыть новые навыки случайным перебором композиций малых примитивов.
        
                max_new: сколько добавить, seed: зерно генератора, generation: поколение. Перебираются каскады и
                пары с LUT (XOR, AND, XNOR) примитивов с ni = no = 1 и состоянием ≤ 5 бит; берутся композиции
                с числом состояний от 2 до 64 и новым поведением. Возвращает список добавленных навыков."""
        rng = random.Random(seed)  # генератор случайных чисел
        eligible = [s for s in self.lib.skills.values() if s.ni == 1 and s.no == 1 and s.quotient_T is not None and s.ns <= 5]  # подходящие примитивы
        pool = []  # пул кандидатов
        for a in eligible:  # по a
            for b in eligible:  # по b
                if a.ns + b.ns <= 8:  # ограничение суммарного размера
                    pool.append(("CASCADE", a, b, 0))  # кандидат — каскад
                    if a.name < b.name and a.ns + b.ns <= 7:  # для пар в алфавитном порядке и меньшего размера
                        for l4 in (6, 8, 9): pool.append(("PARALLEL_LUT", a, b, l4))  # кандидаты — пары с LUT 6 (XOR), 8 (AND), 9 (XNOR)
        rng.shuffle(pool); committed: list[Skill] = []  # перемешать пул; добавленные навыки
        for op, a, b, l4 in pool:  # по кандидатам
            if len(committed) >= max_new: break  # набрано max_new — стоп
            k_q, bdig_q, _ = compose_quotient(op, a, b, lut4=l4)  # фактор-автомат композиции
            if 2 <= k_q <= 64 and bdig_q not in self.lib.by_behav:  # от 2 до 64 состояний и новое поведение — собрать навык
                committed.append(self.compose(op, a, b, f"novel_{generation}_{len(committed):02d}", lut4=l4, generation=generation))  # собрать новый навык с именем novel_<поколение>_<номер> и добавить в список принятых
        return committed  # вернуть добавленные

    def repair_skill_netlist(self, skill_name: str, corrupted_net: dict) -> tuple[dict, list[int], int]:
        """Починить схему навыка хирургически (только испорченные выходы).
        
                skill_name: имя навыка, corrupted_net: испорченная схема. Возвращает результат surgical_repair_netlist."""
        sk = self.lib.skills[self.lib.resolve(skill_name)]  # навык из библиотеки
        if sk.anf is None or sk.rows is None: raise ValueError("surgical cone repair requires auditable skill")  # без АНФ и таблицы починка невозможна
        return surgical_repair_netlist(corrupted_net, sk.anf, sk.rows, self.learner, self.issuer)  # хирургическая починка

    def save_checkpoint(self, path: str | Path, meta: dict | None = None) -> dict:
        """Сохранить состояние агента в JSON-файл.
        
                path: путь, meta: дополнительные поля. Глобальная схема обязана пройти проверку.
                Возвращает словарь payload."""
        ok, errs = self.lib.dag.verify(self.lib.skills)  # проверка глобальной схемы
        if not ok: raise RuntimeError(f"global DAG invalid: {errs[:3]}")  # не прошла — ошибка (первые три ошибки)
        payload = {"version": "4.0", "issuer": self.issuer, "use_adt": self.use_adt, "use_factored_synthesis": self.use_factored_synthesis,  # payload: версия, издатель, режимы
                   "meta": meta or {}, "skills": {k: v.export() for k, v in self.lib.skills.items()}, "aliases": self.lib.aliases,  # продолжение: meta, навыки, синонимы
                   "cache": self.lib.cache, "by_behav": self.lib.by_behav, "self_model": self.model.export(), "trace": self.trace, "dag": self.lib.dag.export()}  # продолжение: кэш, хеши поведения, модель, трассировка, глобальная схема
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")  # записать файл (отступ 2)
        return payload  # вернуть payload

    @classmethod  # метод класса
    def load_checkpoint(cls, path: str | Path) -> "AutonomousTemporalAgent":
        """Загрузить агента из JSON-файла.
        
                path: путь. Восстанавливаются навыки, синонимы, кэш, модель и трассировка; глобальная схема
                пересобирается и сравнивается с сохранённой. Возвращает агента."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))  # прочитать JSON
        agent = cls(issuer=d.get("issuer", DEFAULT_ISSUER), min_warmup_trials=d["self_model"]["min_trials"],  # создать агента с сохранёнными параметрами
                    use_adt=d.get("use_adt", True), use_factored_synthesis=d.get("use_factored_synthesis", True))  # по всем навыкам
        for v in d["skills"].values():  # фактор-таблица из списков (или None)
            qT = tuple(tuple((x[0], tuple(x[1])) for x in row) for row in v["quotient_T"]) if v["quotient_T"] is not None else None  # таблица строк (или None)
            rows = [tuple(x) for x in v["rows"]] if v["rows"] is not None else None  # АНФ из списков (или None)
            anf = tuple(frozenset(x) for x in v["anf"]) if v["anf"] is not None else None  # создать навык по сохранённым полям
            agent.lib.add(Skill(v["name"], v["ns"], v["ni"], v["no"], v["reset"], rows, anf, v["net"], v["route"], v["op"],  # продолжение: имя, разрядности, начальное состояние, таблица, АНФ, схема, маршрут, операция
                                tuple(v["parents"]), v["queries"], v["compile_mode"], v["k_states"], v["ns_min"],  # продолжение: родители, запросы, способ компиляции, число состояний, минимальная разрядность
                                v["behav_digest"], qT, v["bisim_verified"], v.get("ag_cert"), v.get("generation", 0)))  # продолжение: хеш поведения, фактор-таблица, признак бисимуляции, сертификат, поколение; добавить в библиотеку
        agent.lib.aliases = dict(d["aliases"]); agent.lib.cache = dict(d["cache"]); agent.lib.by_behav = dict(d["by_behav"])  # синонимы, кэш, хеши поведения
        sm = d["self_model"]; agent.model.trials, agent.model.exact, agent.model.enabled = sm["trials"], sm["exact"], sm["enabled"]  # модель: счётчики попыток, точных результатов, признак включения
        agent.model.history, agent.model.epistemic_alerts = list(sm["history"]), list(sm["epistemic_alerts"])  # история и предупреждения
        agent.model.mdl_predictions = list(sm.get("mdl_predictions", [])); agent.model.family_hits = dict(sm.get("family_hits", {}))  # предсказания маршрутизации, попадания семейств
        agent.model.adt_verifications = sm.get("adt_verifications", 0); agent.model.adt_baseline_wp_ticks = sm.get("adt_baseline_wp_ticks", 0)  # счётчики проверок деревьев и сумма длин классических тестов
        agent.model.adt_actual_ticks = sm.get("adt_actual_ticks", 0); agent.model.pac_lstar_solves = sm.get("pac_lstar_solves", 0)  # суммарная длина адаптивных тестов, решения через L* с фильтром шума
        agent.model.sop_baseline_gates = sm.get("sop_baseline_gates", 0); agent.model.opt_chosen_gates = sm.get("opt_chosen_gates", 0)  # суммы размеров схем
        agent.trace = list(d.get("trace", []))  # трассировка
        if agent.lib.dag.export()["gates"] != d["dag"]["gates"] or agent.lib.dag.export()["outputs"] != d["dag"]["outputs"]:  # глобальная схема после пересборки обязана совпасть с сохранённой (вентили и выходы)
            raise RuntimeError("checkpoint reconstruction mismatch")  # ошибка: восстановление не совпало
        return agent  # вернуть агента

    def status(self) -> dict:
        """Статус агента.
        
                Возвращает словарь: версия, число навыков и синонимов, число навыков с несколькими выходами,
                число навыков с модульной компакцией, максимум переменных, метрики глобальной схемы, результат
                её проверки, ошибки и экспорт модели."""
        ok, errs = self.lib.dag.verify(self.lib.skills)  # проверка глобальной схемы
        return {"version": "4.0", "unique_behavioral_skills": len(self.lib.skills), "aliases": len(self.lib.aliases),  # словарь статуса: версия, число навыков и синонимов
                "multi_output_skills_no2": sum(1 for s in self.lib.skills.values() if s.no >= 2),  # число навыков с двумя и более выходами
                "ag_compacted_skills": sum(1 for s in self.lib.skills.values() if s.compile_mode == "MODULAR_AG_COMPACT"),  # число навыков с модульной компакцией
                "max_nvars": max((s.nvars for s in self.lib.skills.values()), default=0),  # максимальное число переменных среди навыков
                "global_dag": self.lib.dag.metrics(), "global_dag_exact": ok, "dag_errors": errs, "self_model": self.model.export()}  # метрики глобальной схемы, результат её проверки, ошибки, экспорт модели


# ---------------------------------------------------------------------------
# 7. Независимая проверка сохранённого состояния, встроенная самопроверка и командная строка
# ---------------------------------------------------------------------------

def zero_import_replay_verify(archive_path: str | Path) -> dict:  # независимая проверка сохранённого состояния; описание в docstring ниже
    """Независимая проверка сохранённого состояния (без обращения к остальному коду движка).
    
            archive_path: путь к JSON. Для каждого навыка с АНФ на всех входах проверяется равенство трёх
            вещей: локальной схемы, глобальной схемы и АНФ; для всех навыков — совпадение локальной и
            глобальной схемы (256 случайных шагов), для навыков с АНФ — ещё совпадение с фактор-таблицей.
            Дополнительно проверяется, что подмена выхода навыка без АНФ в глобальной схеме обнаруживается.
            Возвращает словарь отчёта."""
    d = json.loads(Path(archive_path).read_text(encoding="utf-8"))  # прочитать JSON
    dag, skills = d["dag"], d["skills"]  # dag — глобальная схема, skills — навыки
    def ev_net(gates, outs, inp):
        """Вычислить схему из списка вентилей (без словаря) на входах inp.
        
                        gates: вентили {a, b, mask}, outs: выходные провода. Возвращает кортеж выходов."""
        w = list(inp)  # провода начинаются с входов
        for g in gates:  # по каждому вентилю
            m = g["mask"]  # маска
            w.append(0 if m == 0 else (1 if m == 15 else (w[g["a"]] ^ w[g["b"]] if m == 6 else w[g["a"]] & w[g["b"]])))  # значение: 0 при m = 0, 1 при m = 15, a ⊕ b при m = 6, иначе a ∧ b
        return tuple(w[o] for o in outs)  # выходы
    rng = random.Random(0xCAFE); rows_checked = ticks_checked = ag_count = 0  # генератор случайных чисел; счётчики проверенных строк, шагов и навыков без АНФ
    for nm, sk in skills.items():  # по всем навыкам
        ns, ni, net, dag_outs = sk["ns"], sk["ni"], sk["net"], dag["outputs"][nm]  # разрядности, локальная схема и её выходы в глобальной схеме
        if sk["anf"] is not None:  # навык с АНФ
            anf = sk["anf"]  # АНФ из списка мономов
            for x in range(1 << (ns + ni)):  # по всем входам
                inp_l = tuple((x >> i) & 1 for i in range(ns + ni))  # вход локальной схемы: биты x
                inp_d = [0] * GLOBAL_N  # вход глобальной схемы: 24 нулевых бита
                for i in range(ns): inp_d[i] = (x >> i) & 1  # биты состояния — в провода 0..ns−1
                for j in range(ni): inp_d[22 + j] = (x >> (ns + j)) & 1  # биты входа — в провода 22..22+ni−1
                assert ev_net(net["gates"], net["outs"], inp_l) == ev_net(dag["gates"], dag_outs, inp_d) == tuple(sum(1 for t in ts if (t & x) == t) & 1 for ts in anf)  # локальная схема = глобальная схема = АНФ (обязаны совпасть)
                rows_checked += 1  # учёт строки
            qT = sk["quotient_T"]; s_d = s_q = 0  # фактор-таблица; состояния глобальной схемы (s_d) и фактор-автомата (s_q)
            for _ in range(256):  # 256 случайных шагов
                u = rng.randrange(1 << ni); inp_d = [0] * GLOBAL_N  # случайный вход; вход глобальной схемы — нули
                for i in range(ns): inp_d[i] = (s_d >> i) & 1  # биты состояния
                for j in range(ni): inp_d[22 + j] = (u >> j) & 1  # биты входа
                yd = ev_net(dag["gates"], dag_outs, inp_d); s_d = sum(v << i for i, v in enumerate(yd[:ns])); s_q, o_q = qT[s_q][u]  # выход глобальной схемы; новое состояние из битов состояния; шаг фактор-автомата
                assert yd[ns:] == tuple(o_q); ticks_checked += 1  # выходы глобальной схемы и фактор-автомата равны; учёт шага
        else:  # иначе — навык без АНФ
            ag_count += 1; s_l = s_d = 0  # учёт навыка; состояния локальной (s_l) и глобальной (s_d) схем
            for _ in range(256):  # 256 случайных шагов
                u = rng.randrange(1 << ni); inp_d = [0] * GLOBAL_N  # случайный вход; вход глобальной схемы — нули
                for i in range(ns): inp_d[i] = (s_d >> i) & 1  # биты состояния
                for j in range(ni): inp_d[22 + j] = (u >> j) & 1  # биты входа
                yl = ev_net(net["gates"], net["outs"], tuple([(s_l >> i) & 1 for i in range(ns)] + [(u >> j) & 1 for j in range(ni)]))  # выход локальной схемы (состояние + вход)
                yd = ev_net(dag["gates"], dag_outs, inp_d); assert yl == yd  # выход глобальной схемы; локальная = глобальная (обязаны совпасть)
                s_l = sum(v << i for i, v in enumerate(yl[:ns])); s_d = sum(v << i for i, v in enumerate(yd[:ns])); ticks_checked += 1  # новые состояния обеих схем; учёт шага
    tamper_rejected = False  # признак обнаружения подмены
    if dag["ag_skill_names"]:  # если есть навыки без АНФ
        t_nm = dag["ag_skill_names"][0]; bad_outs = list(dag["outputs"][t_nm]); bad_outs[-1] = 22 if bad_outs[-1] != 22 else 0  # первый такой навык; bad_outs — его выходы с заменённым последним проводом
        for _ in range(16):  # до 16 случайных входов
            inp_d = tuple(rng.randrange(2) for _ in range(GLOBAL_N))  # случайный вход глобальной схемы
            if ev_net(dag["gates"], dag["outputs"][t_nm], inp_d) != ev_net(dag["gates"], bad_outs, inp_d): tamper_rejected = True; break  # подменённый выход изменил ответ — подмена обнаружена
    return {"archive": str(archive_path), "skills_verified": len(skills), "ag_skills": ag_count, "dag_gates": len(dag["gates"]),  # отчёт: путь, число навыков, навыков без АНФ, вентилей глобальной схемы
            "rows_checked": rows_checked, "ticks_checked": ticks_checked, "tamper_rejected": tamper_rejected, "pass": True}  # продолжение: проверенные строки, шаги, признак обнаружения подмены, pass = True

def run_self_verify(seed: int = 20261127, wide_bits: int = 3, out_dir: str | Path | None = None) -> dict:
    """Полная самопроверка движка: эталонный агент, второй агент, цели, починка, сохранение и загрузка.
    
            seed: зерно, wide_bits: разрядность широкого счётчика, out_dir: каталог для файлов состояния.
            Все проверки — assert; при успехе возвращается словарь с зерном, временем, статусом, отчётом
            независимой проверки и результатами целей. Аргумент x: биты 0..1 — состояние, старшие биты — вход."""
    t0 = time.perf_counter(); rng = random.Random(seed)  # время начала; генератор
    o_par = lambda x: ((x & 1) ^ ((x >> 1) & 1),) * 2  # чётность: выход = x0 ⊕ x1, два одинаковых выхода
    o_edg = lambda x: ((x >> 1) & 1, ((x >> 1) & 1) & (1 ^ (x & 1)))  # фронт: выходы x1 и x1 ∧ ¬x0
    o_sh3 = lambda x: ((((x & 7) << 1) | ((x >> 3) & 1)) & 1, ((((x & 7) << 1) | ((x >> 3) & 1)) >> 1) & 1, ((((x & 7) << 1) | ((x >> 3) & 1)) >> 2) & 1, (x >> 2) & 1)  # сдвиговый регистр: s' = ((s & 7) << 1) | x3 (три бита выхода) и четвёртый выход x2
    o_wc = lambda x, m=(1 << wide_bits) - 1: tuple([(((x & m) + ((x >> wide_bits) & 1)) & m) >> i & 1 for i in range(wide_bits)] + [int((x & m) == m and ((x >> wide_bits) & 1) == 1)])  # широкий счётчик: s' = ((s & m) + x_w) mod m, m = 2^wide_bits − 1; выход — биты s' и признак (s = m и x_w = 1)
    o_101 = lambda x: (lambda s, u: (lambda ns, y: (ns & 1, (ns >> 1) & 1, y))(*((1 if u else 0, 0) if s == 0 else ((2 if u == 0 else 1, 0) if s == 1 else ((3 if u else 0, int(u == 1)) if s == 2 else (1 if u else 2, 0))))))(x & 3, (x >> 2) & 1)  # детектор «101»: s = x & 3, u = x2; переходы 0→(u, 0), 1→(¬u·2 + u, 0), 2→(u·3, u = 1), 3→(u·2 + ¬u, 0); выход — биты состояния и y
    o_rc = lambda x: (lambda ns: (ns & 1, (ns >> 1) & 1, int(ns == 3)))(0 if (x >> 3) & 1 else min(3, (x & 3) + ((x >> 2) & 1)))  # счётчик со сбросом: x3 — сброс в 0, иначе s' = min(3, s + x2); выход — два бита состояния и признак s' = 3
    o_gp = lambda x: ((x & 1) ^ (((x >> 1) & 1) & ((x >> 2) & 1)),) * 2  # управляемая чётность: выход = x0 ⊕ (x1 ∧ x2), два одинаковых выхода
    def mk_mod(km, p=0.0, sd=0):
        """Оракул счётчика по модулю km с шумом p (зерно sd).
        
                Возвращает InteractiveMealyOracle с ni = no = 1."""
        r = random.Random(sd)  # генератор шума
        return InteractiveMealyOracle(1, 1, lambda seq: _ev_mod(seq, km, p, r))  # оракул: функция ответа — _ev_mod
    def _ev_mod(seq, km, p, r):
        """Ответ модульного счётчика: s' = (s + u) mod km; y = 1, если s' = 0 и u = 1; шум инвертирует y с вероятностью p."""
        s = 0; out = []  # начальное состояние; накопитель
        for u in seq: s = (s + u) % km; y = 1 if (s == 0 and u == 1) else 0; out.append((y ^ (1 if p > 0 and r.random() < p else 0),))  # по каждому входу: обновить состояние, вычислить y, добавить (y с шумом,)
        return tuple(out)  # кортеж выходов
    def mk_mod_ni2(km, p=0.0, sd=0):
        """Оракул счётчика по модулю km с двумя входами (старший бит — сброс) и шумом p.
        
                Возвращает InteractiveMealyOracle с ni = 2, no = 1."""
        r = random.Random(sd)  # генератор шума
        def fn(seq):
            """Функция ответа на последовательность."""
            s = 0; out = []  # начальное состояние; накопитель
            for u in seq:  # по каждому входу
                if (u >> 1) & 1: s, y = 0, 0  # старший бит = 1 — сброс
                elif u & 1: y = 1 if s == km - 1 else 0; s = (s + 1) % km  # младший бит = 1: y = 1 при s = km − 1, иначе 0; s' = (s + 1) mod km
                else: y = 0  # иначе y = 0
                out.append((y ^ (1 if p > 0 and r.random() < p else 0),))  # добавить выход с шумом
            return tuple(out)  # кортеж выходов
        return InteractiveMealyOracle(2, 1, fn)  # создать оракул
    def mk_noisy(sk, p, sd):
        """Зашумлённый оракул по таблице навыка: выход инвертируется с вероятностью p.
        
                sk: навык, p: вероятность инверсии, sd: зерно. Возвращает InteractiveMealyOracle той же арности."""
        r = random.Random(sd); rows, ns = sk.rows, sk.ns  # генератор шума; таблица и разрядность состояния навыка
        def fn(seq):
            """Функция ответа на последовательность."""
            s = 0; out = []  # начальное состояние; накопитель
            for u in seq: row = rows[s | (u << ns)]; s = sum(v << i for i, v in enumerate(row[:ns])); out.append(tuple(v ^ (1 if r.random() < p else 0) for v in row[ns:]))  # по каждому входу: шаг по таблице, новое состояние, выход с шумом
            return tuple(out)  # кортеж выходов
        return InteractiveMealyOracle(sk.ni, sk.no, fn)  # создать оракул

    ref = AutonomousTemporalAgent()  # эталонный агент
    pa = ref.learn_primitive("parity", 1, 1, 1, o_par, 1); ed = ref.learn_primitive("rising_edge", 1, 1, 1, o_edg, 2)  # примитивы: чётность, фронт
    sh = ref.learn_primitive("shift3", 3, 1, 1, o_sh3, 1); s101 = ref.learn_primitive("seq101", 2, 1, 1, o_101, 3)  # сдвиговый регистр, детектор «101»
    rc = ref.learn_primitive("reset_counter", 2, 2, 1, o_rc, 4); gp = ref.learn_primitive("gated_parity", 1, 2, 1, o_gp, 2)  # счётчик со сбросом, управляемая чётность
    r_s101_pa = ref.compose("CASCADE", s101, pa, "r1"); r_re101 = ref.compose("CASCADE", ed, s101, "r2"); r_pa101 = ref.compose("CASCADE", pa, s101, "r3")  # композиции: каскады
    r_xor = ref.compose("PARALLEL_LUT", ed, pa, "r4", lut4=6); r_pp = ref.compose("PARALLEL_PAIR", ed, pa, "r5")  # пара с LUT XOR и параллельная пара
    r_p2gp = ref.compose("CASCADE", r_pp, gp, "r6"); r_dfb = ref.compose("DELAYED_FEEDBACK", rc, None, "r7")  # каскад пары и управляемой чётности, задержанная обратная связь
    r_cfb = ref.compose("COUPLED_FEEDBACK_2BIT", rc, gp, "r8"); r_rmux = ref.compose("DELAYED_FEEDBACK", ref.compose("MUX_2IN", ed, pa, "r9m"), None, "r9")  # связанная обратная связь, рекуррентный мультиплексор
    r_moe = ref.compose("MOE_MUX", pa, ed, "r10", pc=sh); r_d2 = ref.compose("CASCADE", r_re101, pa, "r11")  # «смесь экспертов», каскад глубины 2
    r_g3c = ref.compose("CASCADE", r_cfb, pa, "r12")  # каскад связанной обратной связи и чётности
    m5, _ = ref.solve_goal("m5", mk_mod(5), 8); m6, _ = ref.solve_goal("m6", mk_mod(6), 8)  # цели: счётчики по модулю 5 и 6
    m7, _ = ref.solve_goal("m7", mk_mod(7), 8); m5n2, _ = ref.solve_goal("m5n2", mk_mod_ni2(5), 8)  # счётчик по модулю 7 и счётчик с двумя входами
    r_g3m7 = ref.compose("CASCADE", m7, pa, "r13")  # каскад модуля 7 и чётности

    ag = AutonomousTemporalAgent()  # второй агент
    pa_g = ag.learn_primitive("parity", 1, 1, 1, o_par, 1); ed_g = ag.learn_primitive("rising_edge", 1, 1, 1, o_edg, 2)  # его примитивы: чётность, фронт
    sh_g = ag.learn_primitive("shift3", 3, 1, 1, o_sh3, 1); _ = ag.learn_primitive("wide_counter", wide_bits, 1, 1, o_wc, wide_bits + 1)  # сдвиговый регистр, широкий счётчик
    _ = ag.learn_primitive("seq101", 2, 1, 1, o_101, 3); rc_g = ag.learn_primitive("reset_counter", 2, 2, 1, o_rc, 4)  # детектор «101» с верной границей, счётчик со сбросом
    _ = ag.learn_primitive("gated_parity", 1, 2, 1, o_gp, 2); _ = ag.learn_primitive("seq101_false", 2, 1, 1, o_101, 1)  # управляемая чётность, детектор «101» с ложной границей степени
    assert ag.learn_primitive("untrusted", 1, 1, 1, o_par, contract=DegreeContract(2, 1, "BAD", 1), allow_full_support=False) is None  # контракт недоверенного издателя обязан дать отказ (None)
    w1 = ag.compose("CASCADE", ed_g, pa_g, "warm1"); _ = ag.compose("CASCADE", pa_g, ed_g, "warm2"); _ = ag.compose("CASCADE", sh_g, pa_g, "warm3")  # три разогревающих каскада
    assert ag.model.enabled  # после трёх точных разогревов модель включена

    bad = {"n_inputs": rc_g.net["n_inputs"], "gates": [dict(g) for g in rc_g.net["gates"]], "outs": list(rc_g.net["outs"])}  # испорченная схема счётчика (копия)
    bad["gates"].append({"a": bad["outs"][2], "b": 0, "mask": 6}); bad["outs"][2] = bad["n_inputs"] + len(bad["gates"]) - 1  # добавить XOR-вентиль и заменить третий выход им
    rep, bw, _ = ag.repair_skill_netlist("reset_counter", bad); assert bw == [2] and symbolic_verify_general(rep, rc_g.anf)  # починить: исправлен только выход 2, схема равна АНФ

    goals = [  # список целей: (имя, оракул, подсказка о числе состояний, ожидаемый хеш поведения)
        ("bb_goal_library_reuse", InteractiveMealyOracle.from_rows(w1.ns, w1.ni, w1.no, w1.rows), 8, w1.behav_digest),  # повторное использование библиотеки
        ("bb_goal_cascade", InteractiveMealyOracle.from_rows(r_s101_pa.ns, r_s101_pa.ni, r_s101_pa.no, r_s101_pa.rows), 12, r_s101_pa.behav_digest),  # каскад
        ("bb_goal_parallel_xor", InteractiveMealyOracle.from_rows(r_xor.ns, r_xor.ni, r_xor.no, r_xor.rows), 8, r_xor.behav_digest),  # пара с LUT XOR
        ("bb_goal_parallel_pair_no2", InteractiveMealyOracle.from_rows(r_pp.ns, r_pp.ni, r_pp.no, r_pp.rows), 8, r_pp.behav_digest),  # параллельная пара с двумя выходами
        ("bb_goal_pair2_to_2in", InteractiveMealyOracle.from_rows(r_p2gp.ns, r_p2gp.ni, r_p2gp.no, r_p2gp.rows), 12, r_p2gp.behav_digest),  # пара → вход 2
        ("bb_goal_delayed_feedback", InteractiveMealyOracle.from_rows(r_dfb.ns, r_dfb.ni, r_dfb.no, r_dfb.rows), 8, r_dfb.behav_digest),  # задержанная обратная связь
        ("bb_goal_coupled_feedback_2bit", InteractiveMealyOracle.from_rows(r_cfb.ns, r_cfb.ni, r_cfb.no, r_cfb.rows), 12, r_cfb.behav_digest),  # связанная обратная связь
        ("bb_goal_recurrent_mux_fb", InteractiveMealyOracle.from_rows(r_rmux.ns, r_rmux.ni, r_rmux.no, r_rmux.rows), 12, r_rmux.behav_digest),  # рекуррентный мультиплексор
        ("bb_goal_temporal_moe_mux", InteractiveMealyOracle.from_rows(r_moe.ns, r_moe.ni, r_moe.no, r_moe.rows), 20, r_moe.behav_digest),  # «смесь экспертов»
        ("bb_goal_depth2_cascade", InteractiveMealyOracle.from_rows(r_d2.ns, r_d2.ni, r_d2.no, r_d2.rows), 12, r_d2.behav_digest),  # каскад глубины 2
        ("bb_goal_noncomposable_mod5", mk_mod(5, 0.0, seed + 1), 8, m5.behav_digest),  # несоставимый счётчик по модулю 5
        ("bb_goal_noisy_5pct_pac", mk_noisy(r_re101, 0.05, seed + 2), 8, r_re101.behav_digest),  # шум 5 %
        ("bb_goal_noisy_10pct_pac", mk_noisy(r_pa101, 0.10, seed + 3), 8, r_pa101.behav_digest),  # шум 10 %
        ("bb_goal_noisy_noncomp_mod7_10pct", mk_mod(7, 0.10, seed + 4), 12, m7.behav_digest),  # несоставимый модуль 7 с шумом 10 %
        ("bb_goal_noisy_noncomp_mod6_15pct", mk_mod(6, 0.15, seed + 5), 10, m6.behav_digest),  # несоставимый модуль 6 с шумом 15 %
        ("bb_goal_noisy_noncomp_mod5_ni2_10pct", mk_mod_ni2(5, 0.10, seed + 6), 6, m5n2.behav_digest),  # модуль 5 с двумя входами и шумом 10 %
    ]  # конец списка
    g_res = {}  # результаты целей
    for nm, bb_g, hint, exp_bd in goals:  # по целям
        sk_g, inf_g = ag.solve_goal(nm, bb_g, max_states_hint=hint, generation=1)  # решить цель
        assert sk_g is not None and sk_g.behav_digest == exp_bd, (nm, inf_g)  # навык получен и хеш поведения совпал
        g_res[nm] = inf_g  # запомнить отчёт
    sk_ch, inf_ch = ag.solve_goal("bb_chaos45", mk_noisy(r_re101, 0.45, seed + 7), 8, generation=1)  # сильный шум (45 %)
    assert sk_ch is None and inf_ch["solved_by"] == "HOLD_UNRECOVERABLE_NOISE"  # ожидается удержание из-за неустранимого шума

    import tempfile  # импорт модуля временных каталогов
    tmp_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="hagi_v40_"))  # каталог для файлов состояния (заданный или временный)
    tmp_dir.mkdir(parents=True, exist_ok=True)  # создать каталог
    cp1 = tmp_dir / f"cp1_{seed}.json"; ag.save_checkpoint(cp1); ag = AutonomousTemporalAgent.load_checkpoint(cp1)  # первое сохранение и загрузка
    ag.discover_novel(max_new=18, seed=seed, generation=2)  # открыть новые навыки (поколение 2)
    pa2, ed2, sh2 = ag.lib.skills["parity"], ag.lib.skills["rising_edge"], ag.lib.skills["shift3"]  # примитивы после загрузки
    ag.compose("MUX_2IN", ed2, pa2, "gen2_mux2", generation=2)  # мультиплексор 2 в 1
    for l4 in (8, 9, 14): ag.compose("PARALLEL_LUT", ed2, sh2, f"gen2_plut_{l4}", lut4=l4, generation=2)  # пары с LUT (AND, XNOR, OR)
    c6 = ag.compose("CASCADE", sh2, sh2, "deep_c6", generation=2); c9 = ag.compose("CASCADE", c6, sh2, "deep_c9", generation=2)  # глубокие каскады сдвигового регистра: 6 и 9 переменных
    c12 = ag.compose("CASCADE", c9, sh2, "deep_c12", generation=2); c15 = ag.compose("CASCADE", c12, sh2, "deep_c15_ag", generation=2)  # 12 и 15 переменных (второй — за пределом полного аудита)
    c18 = ag.compose("CASCADE", c15, sh2, "deep_c18_ag", generation=2); c21 = ag.compose("CASCADE", c18, sh2, "deep_c21_ag", generation=2)  # 18 и 21 переменная (оба за пределом аудита)
    ag.compose("MOE_MUX", pa2, c6, "ag_moe_wide", pc=c6, generation=2); assert c21.nvars == 22  # «смесь экспертов» на широких схемах; итог — 22 переменные

    cp2 = tmp_dir / f"cp2_{seed}.json"; ag.save_checkpoint(cp2); ag = AutonomousTemporalAgent.load_checkpoint(cp2)  # второе сохранение и загрузка
    sk_g3c, inf_g3c = ag.solve_goal("bb_goal_gen3_coupled_transfer", InteractiveMealyOracle.from_rows(r_g3c.ns, r_g3c.ni, r_g3c.no, r_g3c.rows), 16, generation=3)  # цель третьего поколения: связанная обратная связь
    sk_g3m, inf_g3m = ag.solve_goal("bb_goal_gen3_mod7_transfer", InteractiveMealyOracle.from_rows(r_g3m7.ns, r_g3m7.ni, r_g3m7.no, r_g3m7.rows), 18, generation=3)  # цель третьего поколения: каскад модуля 7
    assert sk_g3c.behav_digest == r_g3c.behav_digest and sk_g3m.behav_digest == r_g3m7.behav_digest  # оба хеша поведения (третье поколение) обязаны совпасть с эталонными
    g_res["bb_goal_gen3_coupled_transfer"] = inf_g3c; g_res["bb_goal_gen3_mod7_transfer"] = inf_g3m  # оба хеша совпали
    ag.compose("PARALLEL_LUT", c15, c6, "ag_par_xor_22", lut4=6, generation=3)  # пара с LUT XOR за пределом аудита

    cp3 = tmp_dir / f"cp3_{seed}.json"; ag.save_checkpoint(cp3); ag = AutonomousTemporalAgent.load_checkpoint(cp3)  # третье сохранение и загрузка
    rep = zero_import_replay_verify(cp3); st = ag.status()  # независимая проверка файла; статус
    return {"seed": seed, "elapsed_sec": round(time.perf_counter() - t0, 3), "status": st, "replay": rep, "goals": g_res}  # итог: зерно, время, статус, отчёт проверки, цели

def main() -> None:
    """Командная строка: статус агента (из файла или нового), независимая проверка файла, полная самопроверка.
    
            Аргументы: --checkpoint — загрузить состояние, --replay — проверить файл, --verify — самопроверка,
            --seed — зерно самопроверки."""
    ap = argparse.ArgumentParser(description="HAGI Fusion v4.0 Single-File Universal Temporal Skill Engine")  # парсер аргументов
    ap.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint JSON and print status")  # --checkpoint: файл состояния для загрузки
    ap.add_argument("--replay", type=str, default=None, help="Run zero-import replay verification on checkpoint JSON")  # --replay: файл для независимой проверки
    ap.add_argument("--verify", action="store_true", help="Run full self-verification suite (18 goals + AG22 + PAC-L* + 3 checkpoints)")  # --verify: полная самопроверка
    ap.add_argument("--seed", type=int, default=20261127, help="Seed for --verify")  # --seed: зерно самопроверки
    args = ap.parse_args()  # разобрать аргументы
    if args.replay: print(json.dumps(zero_import_replay_verify(args.replay), indent=2)); return  # --replay: напечатать отчёт проверки
    if args.verify: print(json.dumps(run_self_verify(seed=args.seed), indent=2)); return  # --verify: напечатать отчёт самопроверки
    agent = AutonomousTemporalAgent.load_checkpoint(args.checkpoint) if args.checkpoint else AutonomousTemporalAgent()  # агент из файла (если задан) или новый; печать статуса
    print(json.dumps(agent.status(), indent=2))  # печать статуса

if __name__ == "__main__":  # запуск только как сценарий
    main()  # вызов main

class ExactMDLLibrary:  # библиотека с точной дедупликацией по поведению; описание в docstring ниже
    """Библиотека с точной семантической дедупликацией: на каждое поведение — самая дешёвая схема.
    
            Ключ — хеш содержания кандидата; при совпадении поведения остаётся кандидат с меньшим
            счётом (биты программы, глубина, число вентилей)."""
    def __init__(self):  # создать пустую библиотеку
        self._items = {}  # словарь хеш -> кандидат

    def commit(self, candidate) -> dict:
        """Принять или отклонить кандидата.
        
                candidate: объект с полями netlist, outputs, metrics и методом semantic_digest.
                При неверном доказательстве — HOLD; новое поведение — COMMIT; существующее — DEDUP с указанием,
                кто оставлен. Возвращает словарь решения."""
        if not symbolic_verify_general(candidate.netlist, candidate.outputs):  # доказательство неверно
            return {"action": "HOLD", "reason": "candidate proof invalid"}  # удержание
        key = candidate.semantic_digest(); old = self._items.get(key)  # ключ — хеш содержания; старый кандидат с таким ключом
        if old is None:  # нового поведения нет
            self._items[key] = candidate  # запомнить кандидата
            return {"action": "COMMIT", "digest": key, "library_size": len(self._items)}  # COMMIT: хеш и размер библиотеки
        old_score = (old.metrics["program_bits"], old.metrics["depth"], old.metrics["gate_count"])  # счёт старого кандидата (биты программы, глубина, вентили)
        new_score = (candidate.metrics["program_bits"], candidate.metrics["depth"], candidate.metrics["gate_count"])  # счёт нового кандидата
        if new_score < old_score: self._items[key] = candidate  # новый дешевле — заменить
        return {"action": "DEDUP", "digest": key, "library_size": len(self._items), "kept": "new" if new_score < old_score else "old"}  # DEDUP: кто оставлен

    def __len__(self):
        """Число уникальных поведений в библиотеке."""
        return len(self._items)  # размер словаря
