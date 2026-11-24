"""Нейро-символьный цикл саморазвития: одна модель, логические схемы внутри неё.

Порядок шагов (внешний сценарий запускает их за одну итерацию):
  1. РОСТ (нейросеть): обучить тело модели на очередном сертифицированном куске
     накопленного корпуса. В этом файле шага роста нет — здесь начинаются
     следующие шаги.
  2. ВЫВОД (символика), derive_tasks: выбрать булеву задачу с полной таблицей
     истинности — по номеру ступени из файла ladder_stage.json.
  3. СБОРКА СХЕМЫ, weave_task: сначала быстрый путь fusion_fastpath (схема
     восстанавливается многочленом Жегалкина с доказательством), иначе
     многократный запуск генетического поиска. Схема принимается только после
     полной проверки таблицы истинности.
  4. ЗАКРЕПЛЕНИЕ, commit: сертифицированные схемы дописываются в файл списка
     схем процессора processor_netlist.json.

Каждая запись несёт сертификат — результат полной проверки таблицы
истинности. Без сертификата схема не закрепляется: процессор строится только
из доказанных схем.

Экспортирует: derive_tasks, compose_adder2, fusion_fastpath, weave_task,
prev_gates, prev_netlist, commit, TAPEOUT, ROOT.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import json  # чтение и запись JSON
import sys  # системные параметры (импортирован без использования)
import time  # отметка времени закрепления
from pathlib import Path  # пути к файлам

from hagi import fabric, unified  # работа с логическими схемами; unified импортирован без использования
from hagi.util import CKPT_DIR  # каталог сохранённых состояний

ROOT = Path(__file__).resolve().parent.parent  # корень проекта: каталог над hagi
TAPEOUT = CKPT_DIR / "processor_netlist.json"  # файл списка схем процессора

# ---------- ВЫВОД: задачи для схем ----------

def derive_tasks(model, k: int, n_tasks: int = 3, seed: int = 13) -> list[dict]:
    """Выдать список из одной булевой задачи по текущей ступени сложности.

        Ступень — число из файла ladder_stage.json (ключ "stage"); если файла
        нет или он не читается — 0. model, k, n_tasks и seed не используются.
        Задача — словарь: name (имя), n_inputs (число входов), truth (кортеж
        входов -> список выходов), ins (ключ -> список входов), meta.stage.
        Ступень 0, одноразрядный сумматор (a, b, cin) -> (s, cout):
          s = a ⊕ b ⊕ cin;  cout = (a ∧ b) ∨ (cin ∧ (a ⊕ b)).
        Ступень 1, двухразрядный сумматор (a0,a1,b0,b1,cin) -> (s0,s1,c2):
          a = a0 + 2·a1; b = b0 + 2·b1; s = a + b + cin;
          s0 = s mod 2; s1 = ⌊s/2⌋ mod 2; c2 = ⌊s/4⌋ mod 2.
        Ступень 2 и далее, мультиплексор 2:1 (a, b, sel) -> out:
          out = a, если sel = 1, иначе b."""
    import numpy as np  # численные массивы (импортирован без использования)
    stage_file = CKPT_DIR / "ladder_stage.json"  # файл с номером ступени
    stage = 0  # ступень по умолчанию
    if stage_file.exists():  # файл ступени существует
        try:  # чтение может не удаться
            stage = json.loads(stage_file.read_text(encoding="utf-8")).get("stage", 0)  # stage = значение ключа "stage" или 0
        except Exception:  # файл повреждён или не словарь
            stage = 0  # ступень 0
    tasks = []  # список задач
    if stage == 0:  # ступень 0: полный одноразрядный сумматор
        # одноразрядный сумматор (a, b, cin) -> (сумма, перенос)
        truth = {}  # таблица истинности
        for a in (0, 1):  # по значению a
            for b in (0, 1):  # по значению b
                for cin in (0, 1):  # по переносу на входе
                    s = a ^ b ^ cin  # сумма: s = a ⊕ b ⊕ cin
                    cout = (a & b) | (cin & (a ^ b))  # перенос: cout = (a ∧ b) ∨ (cin ∧ (a ⊕ b))
                    truth[(a, b, cin)] = [s, cout]  # строка таблицы: (a, b, cin) -> [s, cout]
        tasks.append({"name": "adder_slice", "n_inputs": 3,  # задача «одноразрядный сумматор», 3 входа
                      "truth": truth,  # таблица истинности
                      "ins": lambda key: [key[0], key[1], key[2]],  # входы: [a, b, cin] из ключа
                      "meta": {"stage": 0}})  # метаданные: ступень 0
    elif stage == 1:  # ступень 1: двухразрядный сумматор
        # двухразрядный сумматор (a0,a1,b0,b1,cin) -> (s0,s1,c2): 32 строки,
        # соединение двух одноразрядных ячеек
        truth = {}  # таблица истинности
        for a0 in (0,1):  # по младшему биту a
            for a1 in (0,1):  # по старшему биту a
                for b0 in (0,1):  # по младшему биту b
                    for b1 in (0,1):  # по старшему биту b
                        for cin in (0,1):  # по переносу на входе
                            a = a0 | (a1 << 1)  # число a = a0 + 2·a1
                            b = b0 | (b1 << 1)  # число b = b0 + 2·b1
                            s = a + b + cin  # сумма как целое число
                            s0, s1, c2 = s & 1, (s >> 1) & 1, (s >> 2) & 1  # биты суммы: s0 = s mod 2, s1 = ⌊s/2⌋ mod 2, c2 = ⌊s/4⌋ mod 2
                            truth[(a0,a1,b0,b1,cin)] = [s0, s1, c2]  # строка таблицы
        tasks.append({"name": "adder2_bit", "n_inputs": 5,  # задача «двухразрядный сумматор», 5 входов
                      "truth": truth,  # таблица истинности
                      "ins": lambda key: list(key),  # входы: ключ как список
                      "meta": {"stage": 1}})  # метаданные: ступень 1
    else:  # ступень 2 и далее
        # мультиплексор 2:1: (a, b, sel) -> out; бит сравнения (a>b) здесь не строится
        truth = {}  # таблица истинности
        for a in (0,1):  # по значению a
            for b in (0,1):  # по значению b
                for sel in (0,1):  # по сигналу выбора
                    truth[(a,b,sel)] = [a if sel else b]  # out = a, если sel = 1, иначе b
        tasks.append({"name": "mux21", "n_inputs": 3,  # задача «мультиплексор 2:1», 3 входа
                      "truth": truth,  # таблица истинности
                      "ins": lambda key: list(key),  # входы: ключ как список
                      "meta": {"stage": 2}})  # метаданные: ступень 2
    return tasks  # список из одной задачи


def compose_adder2() -> dict:  # собрать двухразрядный сумматор из готовых ячеек
    """Ступень 1: двухразрядный сумматор из двух одноразрядных ячеек.

        Входы: 0 = a0, 1 = a1, 2 = b0, 3 = b1, 4 = cin. Ячейка
        adder_cell(A, B, C, base) занимает пять вентилей и возвращает
        (вентили, номер провода суммы, номер провода переноса). Первая ячейка:
        a0 + b0 + cin -> (s0, c1); вторая: a1 + b1 + c1 -> (s1, c2). Схема
        проверяется по полной таблице из 32 строк. Возврат: словарь результата
        или None, если хотя бы одна строка неверна."""
    from hagi import fabric  # эмуляция и проверка схем (повторный импорт безвреден)
    def adder_cell(srcA, srcB, srcC, base):  # входы A, B, перенос C; base — номер первого свободного провода
        g = [  # вентили ячейки; выход вентиля с индексом i имеет номер base + i
            {"a": srcA, "b": srcB, "mask": fabric.MASKS["XOR"]},  # вентиль: base+0: t = A ⊕ B
            {"a": base+0, "b": srcC, "mask": fabric.MASKS["XOR"]},  # base+1: сумма s = t ⊕ C
            {"a": base+0, "b": srcC, "mask": fabric.MASKS["AND"]},  # вентиль: base+2: u = t ∧ C
            {"a": srcA, "b": srcB, "mask": fabric.MASKS["AND"]},  # вентиль: base+3: v = A ∧ B
            {"a": base+3, "b": base+2, "mask": fabric.MASKS["OR"]},  # base+4: перенос = v ∨ u
        ]
        return g, base+1, base+4  # вентили, номер провода суммы, номер провода переноса
    n_in = 5  # число входов
    g1, s0, c1 = adder_cell(0, 2, 4, n_in)  # младший разряд: a0, b0, cin -> сумма s0, перенос c1
    gates = list(g1)  # начать список вентилей с первой ячейки
    base2 = n_in + len(gates)  # первый свободный провод: 5 + 5 = 10
    g2, s1, c2 = adder_cell(1, 3, c1, base2)  # старший разряд: a1, b1, c1 -> сумма s1, перенос c2
    gates += g2  # добавить вентили второй ячейки
    genome = {"W": 1, "n_inputs": n_in, "gates": gates,  # геном: 1 дорожка, 5 входов, 10 вентилей
              "outs": [s0, s1, c2]}  # выходы: s0, s1, c2
    truth = {}  # ожидаемая таблица истинности
    for a0 in (0,1):  # по младшему биту a
        for a1 in (0,1):  # по старшему биту a
            for b0 in (0,1):  # по младшему биту b
                for b1 in (0,1):  # по старшему биту b
                    for cin in (0,1):  # по переносу на входе
                        s = (a0|(a1<<1)) + (b0|(b1<<1)) + cin  # сумма s = (a0 + 2·a1) + (b0 + 2·b1) + cin
                        truth[(a0,a1,b0,b1,cin)] = [s&1, (s>>1)&1, (s>>2)&1]  # выходы: биты 0, 1, 2 числа s
    ok, _ = fabric.certify(genome, truth, lambda k: list(k))  # полная проверка: совпали ли все 32 строки
    if not ok:  # есть несовпадение
        return None  # схема не сертифицирована
    return {"profile": "adder2_composed", "seed": 0,  # результат: имя схемы и зерно 0
            "certified_gen": 0, "best": genome, "bits": None,  # поколение 0, геном, стоимость в битах не измерялась
            "history": [{"phase": "COMPOSE", "gates": len(gates)}]}  # история: фаза «соединение» и число вентилей


def fusion_fastpath(task: dict) -> dict | None:  # быстрый путь: многочлен Жегалкина вместо поиска
    """Быстрый путь до дорогого генетического поиска: восстановить схему
    многочленом Жегалкина (АНФ) с доказательством.

        1) Число входов n должно быть от 1 до 14, иначе None.
        2) Оракул: x -> truth[(x_0, …, x_{n−1})], где x_i = (x >> i) & 1.
        3) ProofCarryingANFLearner.fit(n, n_out, oracle, allow_full_support=True)
           подбирает многочлен. Если кандидата нет или решение не
           COMMIT_CANDIDATE — None.
        4) Схема многочлена превращается в геном и проверяется по полной таблице
           истинности задачи; при ошибке или несовпадении — None.
        Возврат: словарь с геномом, стоимостью в битах и сведениями о способе."""
    from hagi.skill_engine import (ProofCarryingANFLearner,  # подбор многочлена с доказательством
                                   ExactMDLLibrary)  # библиотека точных описаний (импортирована без использования)
    n = task["n_inputs"]  # число входов
    if n < 1 or n > 14:  # число входов вне поддерживаемого диапазона
        return None  # быстрый путь не применяется
    ins = task["ins"]  # функция «ключ -> список входов»
    key0 = next(iter(task["truth"]))  # первый ключ таблицы истинности
    n_out = len(task["truth"][key0])  # число выходов = длина вектора выходов
    def oracle(x):  # значение задачи на входном числе x
        return tuple(int(v) for v in  # целые значения выходов
                     task["truth"][tuple((x >> i) & 1  # ключ: x_i = (x >> i) & 1
                                          for i in range(n))])  # по i = 0..n−1
    L = ProofCarryingANFLearner(trusted_issuers={"hagi-truth-spec"})  # доверенный источник — сама таблица истинности
    try:  # подбор может вызвать ошибку
        cand, ev = L.fit(n, n_out, oracle, allow_full_support=True)  # кандидат и решение ev
    except Exception:  # ошибка при подборе
        return None  # быстрый путь не удался
    if cand is None or ev.get("action") != "COMMIT_CANDIDATE":  # кандидата нет или решение не «принять»
        return None  # быстрый путь не удался
    # Схема многочлена превращается в геном через список соединений и
    # проверяется по той же таблице истинности.
    genome = {"W": 1, "n_inputs": n, "gates": cand.netlist["gates"],  # геном: 1 дорожка, n входов, вентили из списка соединений
              "outs": cand.netlist["outs"]}  # выходы из списка соединений
    try:  # проверка может вызвать ошибку
        ok, _ = fabric.certify(genome, task["truth"], ins)  # полная проверка по таблице задачи
    except Exception:  # ошибка при проверке
        return None  # быстрый путь не удался
    if not ok:  # таблица не сошлась
        return None  # быстрый путь не удался
    return {"best": genome, "seed": "fusion-anf",  # результат: геном и метка способа
            "certified_gen": 0, "bits": cand.netlist["metrics"]["program_bits"],  # поколение 0; стоимость в битах из метрик
            "_fusion": {"mode": ev.get("mode"),  # сведения о способе: режим
                        "queries": ev.get("query_count"),  # число запросов к оракулу
                        "gate_count": cand.netlist["metrics"]["gate_count"],  # число вентилей
                        "depth": cand.netlist["metrics"]["depth"]}}  # глубина схемы


def weave_task(task: dict, pop_size: int = 100, generations: int = 3000,  # задача, размер популяции, предел поколений
               restarts: int = 10, seed: int = 5) -> dict | None:  # число перезапусков, зерно
    """Запустить сборку схемы для одной задачи; вернуть сертифицированный
    оптимизированный геном или None.

        1) fusion_fastpath(task): если удалось — его результат.
        2) Иначе restarts запусков fabric.evolve с зерном seed + r и одной
           дорожкой (_W = 1): задача булева, одна дорожка точна и в 4 раза
           дешевле в проверке. Лучший геном каждого запуска проверяется по
           полной таблице истинности; первый прошедший возвращается.
        3) Если схема не найдена — попытка открыть класс конструкции по
           наблюдениям (run_rdl); результат печатается и на возврат не влияет."""
    fast = fusion_fastpath(task)  # быстрый путь
    if fast is not None:  # быстрый путь удался
        return fast  # вернуть его результат
    truth = task["truth"]  # таблица истинности
    ins = task["ins"]  # функция «ключ -> список входов»
    for r in range(restarts):  # по номерам перезапусков r = 0..restarts−1
        # W=1: сумматор булев (входы — одиночные биты), одна дорожка точна
        # и в 4 раза дешевле в проверке
        res = fabric.evolve({**truth, "_W": 1}, ins,  # эволюция по таблице с ключом "_W" = 1
                            n_inputs=task["n_inputs"],  # число входов задачи
                            pop_size=pop_size, generations=generations,  # размер популяции и предел поколений
                            seed=seed + r, log=False,  # зерно seed + r; без записи в журнал
                            profile=task["name"])  # имя профиля поиска
        if res["best"] is not None:  # найден лучший геном
            ok, _ = fabric.certify(res["best"], truth, ins)  # полная проверка по таблице задачи
            if ok:  # все строки совпали
                return res  # вернуть результат запуска
    # Открытие представлений: задача не решена синтезом — попытка изобрести
    # класс конструкции по наблюдениям (честный отказ, если класс вне
    # параметрического пространства). Печатается, на закрепление не влияет.
    try:  # открытие представлений не должно ломать цикл
        from hagi import invention as _rdl  # модуль открытия классов переходов

        class _TaskWorld:  # среда для открытия классов (в коде ниже не используется)
            clock_action = 0  # действие «ждать» — номер 0

            def reset(self):  # начать заново
                return 0  # начальное состояние 0

            def step(self, a):  # шаг среды
                return 0  # всегда 0

            def true_step(self, s, a):  # истинный переход
                key = tuple((s >> i) & 1 for i in  # биты состояния s: s_i = (s >> i) & 1
                            range(task["n_inputs"] - 1))  # по i = 0..n_inputs−2
                out = task["truth"].get((s, 1) + key[1:] if False else None)  # условие ложно: ключ всегда None
                return 0  # всегда 0

        # Для открытия класса нужен поток переходов по одному действию; он
        # выводится из таблицы истинности, где возможно (бит действия —
        # последний вход). Иначе честный пропуск.
        if task["n_inputs"] >= 2:  # входов не меньше двух
            n_in = task["n_inputs"]  # число входов
            def tstep(s, a):  # шаг перехода: состояние s, действие a
                key = (s & ((1 << (n_in - 1)) - 1)) | (a << (n_in - 1))  # ключ key = (s mod 2^(n_in−1)) + a·2^(n_in−1)
                want = task["truth"].get(tuple((key >> i) & 1  # ожидаемый выход для битов ключа
                                               for i in range(n_in)))  # бит ключа key_i = (key >> i) & 1
                if want is None:  # такого входа в таблице нет
                    return s  # состояние не меняется
                # несколько выходов схлопываются в бит состояния 0, когда это возможно
                return want[0] if len(want) == 1 else s  # первый выход, если выход один, иначе состояние то же
            class _TW:  # среда с потоком переходов
                clock_action = 0  # действие «ждать» — номер 0

                def reset(self):  # начать заново
                    return 0  # начальное состояние 0

                def step(self, a):  # шаг среды
                    return 0  # всегда 0

                def true_step(self, s, a):  # истинный переход
                    return tstep(s, a)  # переход tstep(s, a)

            rr = _rdl.run_rdl(_TW())  # открыть класс переходов; savings_pct = 100·(1 − (семейств + образцов)/48)
            if rr.get("proven"):  # класс доказан
                print(f"  RDL discovered class={rr['invented_class']} "  # напечатать найденный класс
                      f"param={rr['param']} (savings {rr['savings_pct']}%)")  # и параметр с экономией в процентах
    except Exception:  # любая ошибка открытия игнорируется
        pass  # пропустить: на результат не влияет
    return None  # сертифицированной схемы нет


# ---------- ЗАКРЕПЛЕНИЕ: записать сертифицированный список схем ----------

def prev_gates(c: dict) -> int:  # число вентилей ранее решённой схемы
    p = c.get("_reuse_of") or {}  # запись о повторном использовании или пустой словарь
    return p.get("gates", 0)  # число вентилей из неё или 0

def prev_netlist(c: dict) -> str:  # текст ранее сертифицированной схемы
    return (c.get("_reuse_of") or {}).get("netlist", "")  # текст схемы из записи или пустая строка

def commit(circuits: list[dict], chunk_id: str, ce: float) -> None:  # схемы, имя куска, кросс-энтропия на отложенных данных
    """Дописать в TAPEOUT запись о закреплении: идентификатор куска, отложенная
    кросс-энтропия ce, время и список схем.

        Для схемы с "_reuse_of" (повтор ранее сертифицированной) записываются
        число вентилей и текст прежней схемы и сертификат «REUSE …». Для новой
        схемы: gates = len(best["gates"]), текст fabric.render(best) и
        сертификат «exhaustive truth table: PASS». Старый список читается из
        файла (или пуст), новая запись добавляется в конец."""
    # читается прежний список из файла, если он существует, иначе пустой список
    # прежний список читается из файла, если он существует; иначе пустой список
    prev = json.loads(TAPEOUT.read_text(encoding="utf-8")) \
        if TAPEOUT.exists() else []  # если файла нет — прежний список пуст
    tape = {  # запись закрепления
        "chunk": chunk_id, "holdout_ce": ce,  # идентификатор куска и кросс-энтропия на отложенных данных
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),  # время в формате ГОД-МЕСЯЦ-ДЕНЬ ЧЧ:ММ:СС
        "circuits": [  # по каждой схеме
            ({"name": c["profile"], "seed": c["seed"],  # повтор: имя и зерно
              "certified_gen": c["certified_gen"],  # поколение сертификации
              "gates": prev_gates(c), "bits": c["bits"],  # число вентилей прежней схемы и стоимость в битах
              "netlist": prev_netlist(c) if c.get("_reuse_of")  # текст прежней схемы, если это повтор
                         else fabric.render(c["best"]),  # иначе текст новой схемы (в этой ветке не случается)
              "certificate":  # сертификат
                  "REUSE of certified circuit (exhaustive PASS)"  # для повтора
                  if c.get("_reuse_of")  # если это повтор
                  else "exhaustive truth table: PASS"}  # иначе полная проверка пройдена
             if c.get("_reuse_of") else  # если это не повтор
             {"name": c["profile"], "seed": c["seed"],  # новая схема: имя и зерно
              "certified_gen": c["certified_gen"],  # поколение сертификации
              "gates": len(c["best"]["gates"]),  # число вентилей = длина списка вентилей
              "bits": c["bits"],  # стоимость в битах
              "netlist": fabric.render(c["best"]),  # текст схемы
              "certificate": "exhaustive truth table: PASS",  # сертификат: полная таблица истинности пройдена
              "truth_hash": c.get("truth_hash")})  # отпечаток таблицы истинности
            for c in circuits]}  # по всем схемам
    prev.append(tape)  # добавить запись в конец списка
    # записать список в файл с небуферизованными символами вне ASCII и отступом 2
    TAPEOUT.write_text(json.dumps(prev, ensure_ascii=False, indent=2),  # список в JSON: символы вне ASCII не экранируются, отступ 2
                       encoding="utf-8")  # кодировка UTF-8


# ---------- главный цикл (находится в сценарии запуска) ----------
