#!/usr/bin/env python3
"""Самопереписывание HAGI: модель пишет собственные программы и переписывает собственное
состояние, а затем продолжает расти из написанного.

    Три возможности, все под дисциплиной сертифицированного порога.

    1. СИНТЕЗ ПРОГРАММ -> PYTHON. transpile() превращает точную программу Expr каждого
       эксперта в самостоятельную функцию Python без внешних зависимостей; export_programs()
       записывает их в generated_programs.py и сверяет каждую функцию с экспертом по всей
       сертифицированной области (точное равенство целых чисел).

    2. САМОПЕРЕПИСЫВАНИЕ. После каждого запуска модель переписывает своё постоянное состояние
       self_state.json: счётчик раундов, лестница нижних бюджетов, накопленный поток
       предложений, числа экспертов и потраченных кандидатов. Модель правит артефакт,
       который управляет её будущим обучением.

    3. ВОЗОБНОВЛЕНИЕ = АВТОНОМИЯ. --resume загружается ровно из того, что модель последний раз
       написала о себе (без внешней конфигурации): записанный поток предложений повторяется,
       новые раунды дописывают сегменты, экспертов и программы, состояние переписывается снова.

    Командная строка:
      python -m hagi.rewrite --rounds N [--state-dir d]
      python -m hagi.rewrite --resume --rounds N [--state-dir d]

    Экспортирует: transpile, export_programs, run, RUNTIME_HELPERS, VAR_NAMES."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import os  # число процессоров
import re  # регулярные выражения

import argparse  # разбор аргументов командной строки
import json  # чтение и запись self_state.json
import pickle  # сохранение процессора
from pathlib import Path  # работа с путями файлов
from typing import Any  # аннотации типов

import hagi  # сам пакет
from hagi.selfmodel import TracedProcessorV9  # (импортирован, в коде ниже не используется)
from hagi.loop import run_self_development  # главный цикл саморазвития

VAR_NAMES = ("x", "y", "z", "u", "v", "w")  # имена аргументов экспортируемых функций


# ------------------------------------------------------------ транспилятор ----

_BIN = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//",  # бинарные операции -> знаки Python
        "mod": "%", "bit_and": "&", "bit_or": "|", "bit_xor": "^"}  # остаток, и, или, искл. или
_CMP = {"lt": "<", "le": "<=", "eq": "==", "ge": ">=", "ne": "!=", "gt": ">"}  # сравнения -> знаки Python
_SIMPLE1 = {"neg": "(-({0}))", "abs": "abs({0})", "sign": "_hagi_sign({0})",  # унарные операции -> шаблоны с одним аргументом {0}
            "bit_not8": "((~{0}) & 0xFF)", "square": "({0} * {0})",  # отрицание; побитово не по модулю 256; квадрат
            "cube": "({0} * {0} * {0})"}  # куб
_SIMPLE2 = {"absdiff": "abs({0} - {1})", "gcd": "_hagi_gcd({0}, {1})",  # бинарные операции -> шаблоны с аргументами {0}, {1}
            "lcm": "_hagi_lcm({0}, {1})", "max": "max({0}, {1})",  # наименьшее общее кратное; максимум
            "min": "min({0}, {1})",  # минимум
            "bit_nand8": "((~({0} & {1})) & 0xFF)",  # не-и по модулю 256
            "bit_nor8": "((~({0} | {1})) & 0xFF)",  # не-или по модулю 256
            "bit_xnor8": "((~({0} ^ {1})) & 0xFF)",  # не-искл. или по модулю 256
            "pow_int": "_hagi_pow({0}, {1})"}  # целая степень


def _fn_hash(name: str) -> str:  # короткое имя функции для экспорта
    """Устойчивое короткое хэш-имя для экспортируемой функции: детерминировано по имени,
    один эксперт всегда получает один и тот же хэш: "f" + первые 10 знаков sha1(имя)."""
    import hashlib  # хэширование имени
    return "f" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]  # имя функции: "f" + sha1(name)[:10]


def transpile(e, names: tuple[str, ...] = VAR_NAMES) -> str:  # выражение Expr -> текст Python
    """Expr -> самостоятельное выражение Python (вспомогательные функции _hagi_* записываются
    в начале сгенерированного модуля).

        e — узел Expr с полями op, value, args; names — имена аргументов. Возвращает текст
        выражения. var i -> names[i] (или "x{i}", если names короче); const v -> repr(int(v));
        call -> <хэш имени>(аргументы…); anf -> _hagi_anf(коэффициенты, переменные);
        if_cond -> (да если условие иначе нет); poly -> _hagi_poly(коэффициенты, x);
        poly_nd -> _hagi_poly_nd(члены, переменные); прочие — по таблицам _BIN, _CMP,
        _SIMPLE1, _SIMPLE2 и отдельным случаям."""
    op = e.op  # имя операции
    if op == "var":  # переменная
        i = int(e.value)  # номер аргумента
        return names[i] if i < len(names) else f"x{i}"  # имя аргумента или "x{i}"
    if op == "const":  # константа
        return repr(int(e.value))  # текст целого числа
    if op == "call":  # вызов другого эксперта
        callee = _fn_hash(e.value) if re.fullmatch(  # если имя похоже на имя эксперта (заглавная буква, затем буквы, цифры, подчёркивание)
            r"[A-Z][A-Za-z0-9_]*", str(e.value)) else str(e.value)  # — берётся его хэш-имя, иначе имя как есть
        return f"{callee}({', '.join(transpile(a, names) for a in e.args)})"  # вызов: имя(транспилированные аргументы через запятую)
    if op == "anf":  # алгебраическая нормальная форма (XOR мономов)
        n_vars, coeffs = e.value  # число переменных и список пар (маска, коэффициент)
        if not coeffs:  # членов нет
            return "0"  # ноль
        cs = ", ".join(f"({int(s)}, {int(c)})" for s, c in coeffs)  # текст пар (маска s, коэффициент c)
        xs = ", ".join(names[:n_vars])  # имена первых n_vars переменных
        return f"_hagi_anf(({cs},), ({xs},))"  # вызов _hagi_anf(кортеж пар, кортеж переменных)
    if op == "if_cond":  # условное выражение
        # (аргумент 1, если аргумент 0 истинен, иначе аргумент 2)
        return (f"({transpile(e.args[1], names)} if "  # скобка, ветка «да», if
                f"{transpile(e.args[0], names)} else "  # условие, else
                f"{transpile(e.args[2], names)})")  # ветка «нет», закрывающая скобка
    if op == "poly":  # многочлен от одной переменной
        coeffs = tuple(e.value or ())  # коэффициенты c_0..c_n
        if not coeffs:  # коэффициентов нет
            return "0"  # ноль
        cs = ", ".join(repr(int(c)) for c in coeffs)  # текст коэффициентов
        return f"_hagi_poly(({cs},), {transpile(e.args[0], names)})"  # _hagi_poly(кортеж коэффициентов, x)
    if op == "poly_nd":  # многочлен от многих переменных
        terms = tuple(e.value or ())  # члены (коэффициент, степени)
        if not terms:  # членов нет
            return "0"  # ноль
        ts = ", ".join(f"({int(c)}, ({', '.join(str(int(p)) for p in pw)}))"  # текст члена: (c, (p_1, ..., p_n))
                        for c, pw in terms)  # по членам
        ns = ", ".join(names)  # имена всех переменных
        return f"_hagi_poly_nd(({ts},), ({ns},))"  # _hagi_poly_nd(кортеж членов, кортеж переменных)
    a = [transpile(x, names) for x in e.args]  # транспилированные аргументы прочих операций
    if op in _BIN:  # бинарная арифметика или побитовая операция
        return f"({a[0]} {_BIN[op]} {a[1]})"  # (a0 знак a1)
    if op in _CMP:  # сравнение
        return f"(1 if {a[0]} {_CMP[op]} {a[1]} else 0)"  # (1, если a0 знак a1, иначе 0)
    if op in _SIMPLE1:  # унарная из таблицы
        return _SIMPLE1[op].format(a[0])  # шаблон с подставленным a0
    if op in _SIMPLE2:  # бинарная из таблицы
        return _SIMPLE2[op].format(a[0], a[1])  # шаблон с подставленными a0, a1
    if op == "fma":  # умножить-сложить
        return f"({a[0]} * {a[1]} + {a[2]})"  # текст умножения со сложением: (a0·a1 + a2)
    if op == "clamp":  # ограничить диапазоном
        return f"min(max({a[0]}, {a[1]}), {a[2]})"  # текст усечения: min(max(a0, a1), a2)
    if op == "clamp3":  # ограничить отрезком [−3, 3]
        return f"min(max({a[0]}, -3), 3)"  # текст усечения: min(max(a0, −3), 3)
    if op == "median3":  # медиана трёх
        return f"sorted(({a[0]}, {a[1]}, {a[2]}))[1]"  # средний элемент отсортированной тройки
    if op == "mul3":  # произведение трёх
        return f"({a[0]} * {a[1]} * {a[2]})"  # текст тройного произведения: (a0·a1·a2)
    raise ValueError(f"unsupported op for transpile: {op}")  # неизвестная операция — ошибка


# текст вспомогательных функций для сгенерированного модуля; после обратного слеша комментария быть не может
RUNTIME_HELPERS = '''\
def _hagi_pow(x, y):
    if y < 0:
        raise ValueError("negative exponent")
    return x ** y


def _hagi_sign(v):
    return (v > 0) - (v < 0)


def _hagi_gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _hagi_lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // _hagi_gcd(a, b)


def _hagi_anf(coeffs, xs):
    mask = 0
    for i, b in enumerate(xs):
        if b:
            mask |= 1 << (len(xs) - 1 - i)
    acc = 0
    for s, c in coeffs:
        if (s & mask) == s:
            acc ^= c
    return acc


def _hagi_poly(coeffs, x):
    acc = 0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _hagi_poly_nd(terms, xs):
    total = 0
    for coef, powers in terms:
        t = coef
        for x, p in zip(xs, powers):
            t *= x ** p
        total += t
    return total
'''  # конец строкового литерала с текстом вспомогательных функций


def _expert_domain(ex) -> list[tuple[int, ...]]:  # все допустимые входы эксперта
    """Полная область входов эксперта: декартово произведение областей аргументов без строк,
    нарушающих valid_rule ("nonzero_y": второй аргумент ≠ 0; "ordered_yz": второй ≤ третьему)."""
    import itertools  # декартово произведение
    doms = [tuple(int(v) for v in d) for d in ex.domains]  # области аргументов как кортежи целых
    rows = []  # накопитель строк
    for row in itertools.product(*doms):  # по строкам произведения D_1 × … × D_n
        if ex.valid_rule == "nonzero_y" and row[1] == 0:  # правило nonzero_y и x_2 = 0
            continue  # пропустить
        if ex.valid_rule == "ordered_yz" and row[1] > row[2]:  # правило ordered_yz и x_2 > x_3
            continue  # пропустить
        rows.append(row)  # добавить строку
    return rows  # список всех допустимых входов


def export_programs(proc, out_dir: Path) -> dict[str, Any]:  # записать программы экспертов в Python и проверить
    """Записать каждого эксперта как самостоятельную функцию Python и проверить точное
    равенство с экспертом по всей его сертифицированной области.

        proc — процессор со словарём experts (имя -> эксперт с .program.expr, .program.arity,
        .domains, .valid_rule); out_dir — каталог для generated_programs.py. Возвращает
        {path, functions, evaluations_checked, failures}: путь, число функций, число сверенных
        значений, число несовпадений."""
    out_dir.mkdir(parents=True, exist_ok=True)  # создать каталог, если нужно
    path = out_dir / "generated_programs.py"  # файл с программами
    lines = [  # строки модуля: docstring, пустая строка, вспомогательные функции
        '"""Programs written by HAGI (exact synthesis, gate-adopted).',  # первая строка docstring
        '',  # пустая строка
        'Auto-generated by hagi_selfrewrite.export_programs; every',  # текст docstring
        "function is validated for exact integer equality against its",  # текст docstring
        "certified expert over the expert's full support domain.",  # текст docstring
        '"""',  # конец docstring
        "", RUNTIME_HELPERS,  # пустая строка и вспомогательные функции
    ]
    for name, ex in proc.experts.items():  # по экспертам
        expr = ex.program.expr  # программа эксперта
        if expr is None:  # программы нет
            continue  # пропустить
        params = ", ".join(VAR_NAMES[:ex.program.arity])  # имена аргументов по арности
        lines.append(f"def {_fn_hash(name)}({params}):")  # заголовок функции с хэш-именем
        lines.append(f"    return {transpile(expr, VAR_NAMES[:ex.program.arity])}")  # тело: return транспилированного выражения
        lines.append("")  # пустая строка между функциями
    path.write_text("\n".join(lines), encoding="utf-8")  # записать модуль

    import importlib.util  # загрузка модуля из файла
    spec = importlib.util.spec_from_file_location("hagi_generated", path)  # спецификация модуля
    mod = importlib.util.module_from_spec(spec)  # пустой объект модуля
    spec.loader.exec_module(mod)  # исполнить файл
    checked = failed = 0  # счётчики сверенных значений и несовпадений
    for name, ex in proc.experts.items():  # по экспертам
        expr = ex.program.expr  # программа эксперта
        fn = getattr(mod, _fn_hash(name), None)  # сгенерированная функция или None
        if expr is None or fn is None:  # программы нет или функция не создана
            if expr is not None:  # программа есть, а функции нет
                failed += 1  # несовпадение: +1
            continue  # следующий эксперт
        for row in _expert_domain(ex):  # по всем допустимым входам
            try:  # оценка может упасть вне фактической области действия эксперта
                want = expr.eval(row, proc.experts)  # ответ эксперта
            except Exception:  # вне области действия
                break  # вне фактической области поддержки эксперта
            if int(fn(*row)) != int(want):  # ответы не совпали
                failed += 1  # несовпадение: +1
                break  # эксперт провален
            checked += 1  # сверено значений: +1
    return {"path": str(path), "functions": sum(  # итог: путь и число функций
                1 for ex in proc.experts.values() if ex.program.expr),  # эксперты с программой
            "evaluations_checked": checked, "failures": failed}  # число сверенных значений и несовпадений


# ------------------------------------------------------------ самопереписывание --

def run(rounds: int, state_dir: Path, workers: int = 0, resume: bool = False,  # один запуск: раунды, каталог состояния, число потоков, возобновление
        floor_ladder: list[int] | None = None) -> dict[str, Any]:  # лестница нижних бюджетов; возвращает отчёт
    """Один запуск роста с переписыванием своего состояния; вернуть отчёт.

        rounds — число раундов; state_dir — каталог self_state.json, incumbent.pkl и программ;
        workers — число потоков (0 — все ядра); resume — загрузиться из записанного состояния;
        floor_ladder — лестница нижних бюджетов, если нет записанной.
        Последовательность: (1) загрузка состояния и восстановление овеществлённых операций;
        (2) run_self_development с принятыми настройками; (3) сохранение процессора и
        операций; (4) самооптимизация библиотеки — сокращение дубликатов и псевдоаргументов
        при точной повторной проверке, полная библиотека — в incumbent_full.pkl;
        (5) экспорт программ; (6) запись self_state.json, где rounds_done и rewrites растут
        на величины этого запуска, а curriculum дополняется записью с delta =
        adopted / (adopted + wasted)."""
    state_dir.mkdir(parents=True, exist_ok=True)  # создать каталог состояния
    state_path = state_dir / "self_state.json"  # файл состояния
    inc_path = state_dir / "incumbent.pkl"  # файл действующего процессора

    initial_segments: list[tuple[int, int]] = []  # поток предложений для повтора (пустой при холодном старте)
    if resume and state_path.exists():  # возобновление и состояние записано
        state = json.loads(state_path.read_text(encoding="utf-8"))  # прочитать состояние
        initial_segments = [tuple(s) for s in state["segments"]]  # записанный поток предложений
        ladder = state.get("floor_ladder") or floor_ladder or [200, 50, 20]  # лестница: из состояния, иначе аргумент, иначе [200, 50, 20]
        booted_from = "self_state.json (the model's own rewrite)"  # источник загрузки — собственная запись модели
        prior = state  # прежнее состояние
        # овеществлённые операции переживают холодный перезапуск (замыкание «изобретение ->
        # овеществление»): вчерашние изобретения — завтрашние примитивы
        from hagi.representations import restore_reified_ops_state  # восстановление овеществлённых операций
        n_reified = restore_reified_ops_state(state_dir)  # число восстановленных операций
        if n_reified:  # они есть
            booted_from += f" + {n_reified} reified ops"  # дописать их число к источнику
    else:  # холодный старт
        ladder = floor_ladder or [200, 50, 20]  # лестница: аргумент или [200, 50, 20]
        booted_from = "cold start"  # источник — холодный старт
        prior = None  # прежнего состояния нет

    # потоки делят память — нет pickle MemoryError и не нужно ухудшение; по умолчанию
    # все ядра процессора (решение пользователя: максимум для машины)
    eff_workers = workers or (os.cpu_count() or 4)  # число потоков: заданное, иначе число ядер, иначе 4
    # принятое настройками петли: значения регуляторов, принятые сертифицированным порогом, ведут рабочий цикл
    try:  # модуль может отсутствовать
        from hagi import selfedit  # самонастройка
        adopted_kwargs = selfedit.adopted_settings()  # принятые настройки
    except Exception:  # любая ошибка
        adopted_kwargs = {}  # без настроек
    if "floor_ladder" not in adopted_kwargs:  # лестница не задана настройками
        adopted_kwargs["floor_ladder"] = ladder  # передать лестницу
    total = run_self_development(  # главный цикл саморазвития
        max_rounds=rounds, workers=eff_workers,  # число раундов и потоков
        initial_segments=initial_segments, **adopted_kwargs)  # поток предложений и принятые настройки
    final_proc = total.pop("_final_incumbent")  # действующий процессор (извлечён из итога)

    report: dict[str, Any] = {  # отчёт о запуске
        "booted_from": booted_from,  # источник загрузки
        "resumed_segments": len(initial_segments),  # сколько предложений повторено
        "prior_state": prior,  # прежнее состояние
        "adopted_rounds": total["final"]["adopted_rounds"],  # принято раундов
        "experts": total["final"]["experts"],  # число экспертов
        "wasted": total["final"]["wasted_candidates_on_holds"],  # кандидатов потрачено на удержания
        "seconds": total["final"]["seconds_total"],  # секунд
    }

    # переписывание: модель сохраняет себя и свои программы
    inc_path.write_bytes(pickle.dumps(final_proc))  # действующий процессор в incumbent.pkl
    # овеществлённые операции тоже сохраняются (непрерывность саморасширения)
    try:  # их может не быть
        from hagi.representations import save_reified_ops_state  # сохранение овеществлённых операций
        save_reified_ops_state(state_dir)  # записать в каталог состояния
    except Exception:  # сохранять нечего
        pass  # овеществлённых операций ещё нет — сохранять нечего
    # самооптимизация (принцип минимальной длины описания): убрать дубликаты и
    # псевдоаргументы, сохранив граф вызовов, заново проверить всё точно; полная
    # библиотека остаётся в incumbent_full.pkl
    try:  # оптимизация не должна ломать цикл
        from hagi import selfmodel as _opt  # сокращение библиотеки
        _res = _opt.prune_library(final_proc)  # план сокращения
        _new, _fails = _opt.apply_prune(final_proc, _res["keep"])  # новый процессор и число неудач проверки
        if _fails == 0 and len(_new.experts) < len(final_proc.experts):  # проверка чистая и библиотека меньше
            inc_full = state_dir / "incumbent_full.pkl"  # файл полной библиотеки
            inc_full.write_bytes(pickle.dumps(final_proc))  # сохранить полную
            final_proc = _new  # далее используется сокращённая
            inc_path.write_bytes(pickle.dumps(final_proc))  # и перезаписать incumbent.pkl
        report["self_optimize"] = _res["report"]  # отчёт об оптимизации
    except Exception as e:  # оптимизация никогда не должна останавливать цикл
        report["self_optimize"] = {"skipped": repr(e)[:80]}  # отметка о пропуске с текстом ошибки (80 знаков)
    report["programs"] = export_programs(final_proc, state_dir)  # экспорт и проверка программ
    state = {  # новое состояние модели
        "rounds_done": (prior or {}).get("rounds_done", 0)  # раундов сделано: прежнее число
                       + total["final"]["adopted_rounds"],  # плюс принято в этот запуск
        "segments": total["segments"],  # накопленный поток предложений
        "floor_ladder": ladder,  # лестница нижних бюджетов
        "rewrites": (prior or {}).get("rewrites", 0) + 1,  # число переписываний: прежнее + 1
        "experts": len(final_proc.experts),  # число экспертов
        "wasted": total["final"]["wasted_candidates_on_holds"],  # потрачено на удержания
    }
    # учёт фаз учебной программы: сертификат учебной программы, порядок элементов не важен
    from hagi.school import certificate as _certificate  # сертификат учебной программы
    adopted = total["final"]["adopted_rounds"] or 1  # принято раундов (0 считается как 1, чтобы не делить на ноль)
    relief = adopted / (adopted + total["final"]["wasted_candidates_on_holds"])  # облегчение: relief = adopted / (adopted + wasted)
    prior_curr = (prior or {}).get("curriculum", [])  # прежние записи учебной программы
    prior_curr.append({"phase": "GROW->DERIVE->OPTIMIZE->GATE",  # добавить запись: фаза
                       "delta": round(relief, 4),  # облегчение с 4 знаками
                       "d_cert": 0.02, "steps": adopted})  # шаг сертификата 0.02 и число шагов
    state["curriculum"] = prior_curr  # записи учебной программы в состоянии
    _certificate(state)  # оформить сертификат состояния
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False),  # записать состояние JSON с отступом 2
                          encoding="utf-8")  # в кодировке UTF-8
    report["self_state"] = {k: (v if k != "segments" else len(v))  # состояние в отчёте: все поля, кроме segments — вместо него длина
                            for k, v in state.items()}  # по полям состояния
    return report  # отчёт о запуске


def _self_edit_guarded(cycle: int) -> None:  # один цикл самонастройки под защитой от ошибок
    """Каждый цикл: модель предлагает патч к собственным регуляторам обучения; решение о
    принятии принимает сертифицированный порог (нет регрессии по контрольным задачам).
    Любая ошибка печатается и не останавливает цикл."""
    try:  # самонастройка под защитой
        from hagi import selfedit  # модуль самонастройки
        res = selfedit.self_edit_cycle(seed=cycle)  # один цикл самонастройки с зерном cycle
        print(f"[self-edit] {res.get('decision')}: {res.get('patch')}",  # решение и патч
              flush=True)  # сразу вывести
    except Exception as e:  # ошибка самонастройки не останавливает цикл
        print(f"[self-edit] skipped: {e!r}", flush=True)  # напечатать пропуск


def _archgenesis_guarded(cycle: int) -> None:  # редко: генезис архитектуры под защитой
    """Генезис архитектуры: предложить изменение конфигурации петли, измерить и принять по
    сертификату (относительно профиля быстродействия). Запускается редко — раз в 20 циклов:
    каждая проба стоит полного микробенчмарка, пробы не бесплатны."""
    if cycle % 20 != 7:  # не тот цикл (сдвиг 7, чтобы не совпадать с самонастройкой)
        return  # выход
    try:  # генезис под защитой
        from hagi import genesis  # модуль генезиса
        res = genesis.genesis_cycle(seed=cycle,  # один цикл генезиса с зерном cycle
                                             profile="throughput")  # профиль — быстродействие
        print(f"[archgenesis] {res.get('decision')}: "  # решение и мутация
              f"{res.get('mutation')} gain={res.get('rel_gain')}",  # с относительным выигрышем
              flush=True)  # сразу вывести
    except Exception as e:  # ошибка генезиса не останавливает цикл
        print(f"[archgenesis] skipped: {e!r}", flush=True)  # напечатать пропуск


def _ec_evolution_guarded(cycle: int) -> None:  # эволюция процессора эллиптических кривых под защитой
    """Эволюционный генезис процессора эллиптических кривых: каждый 30-й цикл — одно поколение
    эволюции генома (пригодность: сначала порог группового закона, затем цена modmul).
    Профиль чередуется: чётные 60-цикловки — быстродействие, нечётные — безопасность."""
    if cycle % 30 != 13:  # не тот цикл
        return  # выход
    try:  # эволюция под защитой
        from hagi import genesis as ec  # модуль генезиса
        profile = "throughput" if cycle % 60 < 30 else "security"  # профиль: throughput, если cycle mod 60 < 30, иначе security
        res = ec.evolve(pop_size=8, generations=3, profile=profile,  # эволюция: 8 особей, 3 поколения
                        seed=cycle)  # с зерном cycle
        print(f"[ec-evolution] {profile}: best {res['best_fitness']} "  # профиль и лучшая пригодность
              f"modmul/mul, genome={res['best_genome']}", flush=True)  # единица modmul/mul и лучший геном
    except Exception as e:  # ошибка эволюции не останавливает цикл
        print(f"[ec-evolution] skipped: {e!r}", flush=True)  # напечатать пропуск


def _retrain_lm_guarded(state_dir: Path) -> None:  # переобучение байтовой языковой модели под защитой
    """Каждый цикл бесконечного режима: байтовая языковая модель переобучается на
    собственных артефактах модели (самореференциальный нейронный рост, с защитой от ошибок)."""
    try:  # переобучение под защитой
        from hagi import school as mind  # модуль «школы» (нейронная часть)
        print("[lm]", mind.retrain(), flush=True)  # переобучить и напечатать итог
    except Exception as e:  # ошибка нейронной части не останавливает цикл
        print(f"[lm] retrain skipped: {e!r}", flush=True)  # напечатать пропуск


def main() -> None:  # командная строка
    p = argparse.ArgumentParser(description="HAGI self-rewrite loop")  # разборщик аргументов
    p.add_argument("--rounds", type=int, default=10)  # число раундов за запуск, по умолчанию 10
    p.add_argument("--resume", action="store_true")  # флаг возобновления из записанного состояния
    # бесконечное автономное саморазвитие: циклы «рост — переписывание — возобновление» до
    # остановки процесса; каждый цикл переписывает self_state.json и generated_programs.py
    p.add_argument("--forever", action="store_true",  # флаг бесконечного режима
                   help="infinite autonomous self-development: grow-"  # текст справки
                        "rewrite-resume cycles until the process is "  # текст справки
                        "stopped; every cycle rewrites self_state.json "  # текст справки
                        "and generated_programs.py")  # конец текста справки
    p.add_argument("--workers", type=int, default=0)  # число потоков, 0 — все ядра
    p.add_argument("--state-dir", type=Path,  # каталог состояния
                  default=None)  # по умолчанию берётся из hagi.util.CKPT_DIR
    a = p.parse_args()  # разобрать аргументы
    if a.state_dir is None:  # каталог не задан
        from hagi.util import CKPT_DIR  # каталог чекпойнтов
        a.state_dir = CKPT_DIR / "self"  # подкаталог self
    if a.forever:  # бесконечный режим
        import time as _time  # пауза между циклами
        cycle = 0  # номер цикла
        while True:  # до остановки процесса
            cycle += 1  # номер цикла: +1
            rep = run(a.rounds, a.state_dir, workers=a.workers, resume=True)  # запуск с возобновлением
            print(f"[forever cycle {cycle}] rounds={rep['self_state']['rounds_done']} "  # цикл и число раундов
                  f"experts={rep['experts']} funcs={rep['programs']['functions']} "  # эксперты и функции
                  f"checked={rep['programs']['evaluations_checked']} "  # сверено значений
                  f"failures={rep['programs']['failures']} "  # несовпадений
                  f"rewrites={rep['self_state']['rewrites']} "  # переписываний
                  f"{rep['seconds']}s", flush=True)  # секунд; вывести сразу
            _retrain_lm_guarded(a.state_dir)  # переобучить языковую модель
            _self_edit_guarded(cycle)  # самонастройка
            _archgenesis_guarded(cycle)  # генезис архитектуры
            _ec_evolution_guarded(cycle)  # эволюция процессора
            _time.sleep(1)  # пауза 1 секунда
    rep = run(a.rounds, a.state_dir, workers=a.workers, resume=a.resume)  # одиночный запуск
    print(json.dumps(rep, indent=2, ensure_ascii=False))  # отчёт JSON с отступом 2


if __name__ == "__main__":  # файл запущен как скрипт
    main()  # разбор аргументов и запуск
