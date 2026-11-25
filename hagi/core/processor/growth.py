"""Нециркулярный учебный план и цикл роста символьного ядра HAGI.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import json  # импорт стандартной библиотеки, нужный этой части
import time  # импорт стандартной библиотеки, нужный этой части
from pathlib import Path  # импорт стандартной библиотеки, нужный этой части
from typing import Any, Optional, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.core.tasks.tasks import TaskSpec  # имена из модуля ядра «tasks» (как до слияния)
from hagi.core.memory.audit import ram_available_bytes  # имена из модуля ядра «audit» (как до слияния)
from hagi.core.processor.processor import Expert  # имена из модуля ядра «processor» (как до слияния)
from hagi.core.processor.processor import GrowingHagiProcessor  # имена из «processor» (модуль, где они определены)
from hagi.core.processor.processor import TheoremMiner  # имена из модуля ядра «processor» (как до слияния)

# ==============================================================================
# 11. Нециркулярный разнообразный саморазвивающийся учебный план и цикл роста
# ==============================================================================


def propose_noncircular_task(  # Предложить новую составную задачу из уже выученных экспертов — разнообразную, невырожденную и ранее не встречавшуюся.
    experts: dict[str, Expert],  # библиотека экспертов
    rng: np.random.Generator,  # генератор случайных чисел
    round_idx: int,  # номер раунда
    seen_sigs: Optional[set[tuple[Any, ...]]] = None,  # сигнатуры уже предложенных задач
    seen_sigs_names: Optional[set[str]] = None,  # имена уже предложенных задач
) -> Optional[tuple[TaskSpec, tuple[str, ...]]]:  # возвращает (задача, имена родителей) или None
    """4567	Предложить новую составную задачу из уже выученных экспертов — разнообразную, невырожденную и ранее не встречавшуюся.
    
            Задача передаётся в learn() только как функция-учитель и области, без готового дерева выражения.
    
            experts — библиотека; rng — генератор случайных чисел; round_idx — номер раунда
            (входит в имя задачи); seen_sigs — сигнатуры уже предложенных задач (пополняется);
            seen_sigs_names — имена уже предложенных задач. Режимы: 0–2 — композиция
            «внешний ∘ внутренний», 3 — булева функция трёх аргументов f(g(x, y), h(y, z)).
            Возвращает (TaskSpec, имена экспертов-родителей) или None, если экспертов нет."""
    seen_sigs_names = seen_sigs_names if seen_sigs_names is not None else set()  # пустое множество, если имена не переданы
    arith2 = [  # бинарные небулевы эксперты
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"  # условие: арность 2 и семейство не «BoolVenn»
    ]
    arith1 = [  # унарные небулевы эксперты
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"  # условие: арность 1 и семейство не «BoolVenn»
    ]
    bool2 = [  # бинарные булевы эксперты
        e for e in experts.values() if e.program.arity == 2 and e.family == "BoolVenn"  # условие: арность 2 и семейство «BoolVenn»
    ]
    if not arith2:  # бинарных небулевых экспертов нет — задачу не построить
        return None  # вернуть None
    pool = arith2 + arith1  # пул внутренних экспертов: бинарные и унарные небулевы
    for attempt in range(32):  # до 32 попыток
        mode = int(rng.integers(0, 4))  # режим — случайное целое 0..3
        if mode == 3 and len(bool2) >= 2:  # режим 3 и есть хотя бы два булевых бинарных эксперта
            f_ex = bool2[int(rng.choice(len(bool2)))]  # случайный булевый эксперт f
            g_ex = bool2[int(rng.choice(len(bool2)))]  # случайный булевый эксперт g
            h_ex = bool2[int(rng.choice(len(bool2)))]  # случайный булевый эксперт h
            teacher = (  # учитель: f(g(x, y), h(y, z))
                lambda x, y, z, f=f_ex, g=g_ex, h=h_ex, env=experts: f.program.eval(  # лямбда с зафиксированными f, g, h и средой
                    (  # пара аргументов
                        g.program.eval((int(x), int(y)), env),  # вычислить g(x, y)
                        h.program.eval((int(y), int(z)), env),  # вычислить h(y, z)
                    ),
                    env,  # среда вычисления
                )
            )  # закрытие лямбды
            vals = {teacher(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)}  # множество значений учителя на всех 8 наборах x, y, z ∈ {0, 1}
            if len(vals) <= 1:  # вырожденная функция (одно значение)
                continue  # другая попытка
            sig = (  # сигнатура задачи
                "b3",  # вид «булева от трёх»
                tuple(teacher(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)),  # таблица значений на 8 наборах x, y, z ∈ {0, 1}
            )  # закрытие кортежа
            if seen_sigs is not None and sig in seen_sigs:  # такая сигнатура уже была
                continue  # другая попытка
            if seen_sigs is not None:  # множество сигнатур передано
                seen_sigs.add(sig)  # запомнить сигнатуру
            sp = TaskSpec(  # описание задачи
                f"V9_BOOL3_R{round_idx}_{attempt}",  # имя: булева от трёх, раунд, попытка
                3,  # арность 3
                ((0, 1), (0, 1), (0, 1)),  # три области {0, 1}
                teacher,  # учитель
                family="auto-bool3",  # семейство «auto-bool3»
            )
            return sp, (f_ex.name, g_ex.name, h_ex.name)  # вернуть задачу и имена f, g, h

        outer = arith2[int(rng.choice(len(arith2)))]  # внешний (бинарный) эксперт
        inner = pool[int(rng.choice(len(pool)))]  # внутренний эксперт из пула
        if inner.program.arity == 1:  # внутренний эксперт унарный
            xs = tuple(  # аргументы внешнего
                x  # первый аргумент примера
                for x in inner.domains[0]  # по значениям первой области внутреннего эксперта
                if abs(x) <= 6  # только небольшие: |x| ≤ 6
                and inner.supports((x,))  # внутренний эксперт поддерживает вход (x,)
                and inner.program.eval((x,), experts) in outer.domains[0]  # и его значение inner(x) лежит в первой области внешнего эксперта
            )
            ys = tuple(  # значения второго аргумента y
                y  # второй аргумент примера
                for y in outer.domains[1]  # по второй области внешнего эксперта
                if abs(y) <= 6  # только небольшие: |y| ≤ 6
                and all(  # и для всех x из xs
                    outer.supports((inner.program.eval((x,), experts), y)) for x in xs  # внешний эксперт поддерживает пару (inner(x), y)
                )
            )  # закрытие ys
            if len(xs) < 3 or len(ys) < 3:  # слишком узкая область (меньше 3 значений по любому аргументу)
                continue  # другая попытка
            tt = tuple(  # таблица значений tt: тройки (x, y, v)
                (  # открытие тройки
                    x,  # первый аргумент x
                    y,  # второй аргумент y
                    outer.program.eval((inner.program.eval((x,), experts), y), experts),  # значение v = outer(inner(x), y)
                )
                for x in xs  # по x из xs
                for y in ys  # по y из ys
            )
            vals = {v for _, _, v in tt}  # множество различных значений
            if (  # если функция вырождена
                len(vals) <= 2  # не более двух различных значений
                or all(v == x for x, _, v in tt)  # или всюду v = x
                or all(v == y for _, y, v in tt)  # или всюду v = y
            ):
                continue  # другая попытка
            sig = ("a2", xs, ys, tt)  # сигнатура: вид «a2», области и таблица значений
            if seen_sigs is not None and sig in seen_sigs:  # такая сигнатура уже была
                continue  # другая попытка
            if seen_sigs is not None:  # если передано множество уже виденных сигнатур
                seen_sigs.add(sig)  # запомнить сигнатуру
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(  # учитель: outer(inner(x), y)
                (i.program.eval((int(x),), env), int(y)), env  # пара (inner(x), y) как целые
            )
            sp = TaskSpec(  # описание задачи
                _short_auto_name(  # уникальное короткое имя
                    round_idx, attempt, inner.name, outer.name, seen_sigs_names  # из раунда, попытки и имён внутреннего и внешнего экспертов
                ),
                2,  # арность 2
                (xs, ys),  # области (xs, ys)
                teacher,  # учитель
                valid_rule=outer.valid_rule,  # правило корректности внешнего эксперта
                family="auto-composed",  # семейство «автоматическая композиция»
            )
            return sp, (outer.name, inner.name)  # вернуть задачу и имена внешнего и внутреннего экспертов
        else:  # иначе внутренний эксперт бинарный: задача outer(x, inner(x, y))
            xs = tuple(  # значения первого аргумента x
                x for x in inner.domains[0] if abs(x) <= 5 and x in outer.domains[0]  # из первой области внутреннего эксперта: |x| ≤ 5 и x в первой области внешнего
            )
            ys_cand = tuple(y for y in inner.domains[1] if abs(y) <= 5)  # кандидаты во второй аргумент: |y| ≤ 5
            ys = tuple(  # значения второго аргумента y
                y  # второй аргумент примера
                for y in ys_cand  # по кандидатам
                if all(  # если для всех x из xs
                    inner.supports((x, y))  # внутренний эксперт поддерживает (x, y)
                    and outer.supports((x, inner.program.eval((x, y), experts)))  # и внешний поддерживает (x, inner(x, y))
                    for x in xs  # по всем x из области xs
                )  # закрытие all
            )
            if len(xs) < 3 or len(ys) < 3:  # слишком узкая область
                continue  # другая попытка
            tt = tuple(  # таблица значений tt: тройки (x, y, v)
                (  # открытие тройки
                    x,  # первый аргумент x
                    y,  # второй аргумент y
                    outer.program.eval(  # вычисление v = outer(x, inner(x, y))
                        (x, inner.program.eval((x, y), experts)), experts  # пара (x, inner(x, y)) и эксперты — вычисление внешнего эксперта
                    ),  # закрытие вычисления inner(x, y)
                )
                for x in xs  # по x из xs
                for y in ys  # по y из ys
            )
            vals = {v for _, _, v in tt}  # множество различных значений
            if (  # если функция вырождена
                len(vals) <= 2  # не более двух различных значений
                or all(v == x for x, _, v in tt)  # или всюду v = x
                or all(v == y for _, y, v in tt)  # или всюду v = y
            ):
                continue  # другая попытка
            sig = ("a2", xs, ys, tt)  # сигнатура: вид «a2», области и таблица значений
            if seen_sigs is not None and sig in seen_sigs:  # такая сигнатура уже была
                continue  # пропустить эту попытку
            if seen_sigs is not None:  # запомнить сигнатуру
                seen_sigs.add(sig)  # запомнить сигнатуру
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(  # учитель: outer(x, inner(x, y))
                (int(x), i.program.eval((int(x), int(y)), env)), env  # пара (x, inner(x, y)) как целые
            )
            sp = TaskSpec(  # описание задачи
                _short_auto_name(  # уникальное короткое имя
                    round_idx, attempt, inner.name, outer.name, seen_sigs_names  # аргументы имени: раунд, попытка, имена внутреннего и внешнего экспертов, занятые имена
                ),  # арность 2
                2,  # арность 2
                (xs, ys),  # области (xs, ys)
                teacher,  # учитель
                valid_rule=inner.valid_rule,  # правило корректности внутреннего эксперта
                family="auto-composed",  # семейство «автоматическая композиция»
            )
            return sp, (outer.name, inner.name)  # вернуть задачу и имена внешнего и внутреннего экспертов
    return None  # все попытки исчерпаны — новой задачи нет


def _short_auto_name(  # Построить короткое уникальное имя автоматически составленной задачи.
    round_idx: int, attempt: int, inner: str, outer: str, seen: set | None = None  # имена внутреннего и внешнего экспертов
) -> str:  # множество занятых имён или None; возвращает имя
    """Вырезать из имени служебные приставки: убрать все приставки «V9_AUTO_» и «V9_BOOL3_»,
                сегменты раунда «R<число>_», ведущий «<число>_» и хвостовые «_<число>(_<число>)»
                (остаётся первая из двух хвостовых цифр). Возвращает содержательную часть имени."""

    def core(n: str) -> str:
        """Ядро имени: имя без служебных приставок и номеров.
        
                    n — исходное имя. Возвращает строку."""
        for part in ("V9_AUTO_", "V9_BOOL3_"):  # по служебным приставкам
            while n.startswith(part):  # пока имя начинается с приставки
                n = n[len(part) :]  # отрезать приставку: n = n[len(part):]
        # убрать сегмент раунда «R<число>_» и лишние номера раунда и попытки
        import re as _re  # импорт модуля регулярных выражений под локальным именем

        n = _re.sub(r"R\d+_", "", n)  # убрать «R<число>_» в любом месте
        n = _re.sub(r"^\d+_", "", n)  # убрать ведущий «<число>_»
        n = _re.sub(r"_\d+(_\d+)$", r"\1", n)  # хвост «_<число>_<число>» заменить на «_<первое число>»
        return n  # вернуть содержательную часть

    base = f"{core(inner)}_{core(outer)}"[:48]  # база имени «ядро_внутреннего_ядро_внешнего», не длиннее 48 символов
    name = base  # имя-кандидат
    k = 2  # счётчик суффиксов
    while seen is not None and name in seen:  # пока имя занято
        name = f"{base}_{k}"  # имя с суффиксом _k
        k += 1  # k = k + 1 — следующий суффикс
    if seen is not None:  # если множество передано
        seen.add(name)  # занять имя
    return name  # вернуть имя


def grow_deep(  # Цикл роста модели по раундам: выучить набор задач, предложить и выучить составные задачи, сжать библиотеку, найти теоремы.
    model: GrowingHagiProcessor,  # процессор
    suite: Sequence[TaskSpec],  # набор задач
    max_rounds: int = 3,  # число раундов
    train_neural: bool = True,  # обучать ли нейросетевые адаптеры
    out_dir: str = ".omc/v9_grow",  # каталог снимков
) -> dict[str, Any]:  # возвращает словарь с итогами роста
    out = Path(out_dir)  # каталог вывода
    out.mkdir(parents=True, exist_ok=True)  # создать каталог вместе с родительскими
    rnd = np.random.default_rng(20260101)  # генератор случайных чисел с фиксированным зерном
    seen_sigs: set[tuple[Any, ...]] = set()  # сигнатуры уже предложенных задач
    rounds = []  # итоги раундов
    for r in range(max_rounds):  # по раундам
        t0 = time.time()  # начало раунда
        commits = 0  # число принятых изменений в раунде
        for sp in suite:  # по задачам набора
            ev = model.learn(  # выучить задачу
                sp, seed=10_000 + r * 7919 + len(sp.name), train_neural=train_neural  # зерно зависит от раунда и длины имени
            )
            if ev["action"] == "COMMIT":  # принято
                commits += 1  # принятий + 1
        proposed = 0  # число предложенных задач
        for k in range(8):  # до 8 предложений
            prop = propose_noncircular_task(model.experts, rnd, r, seen_sigs)  # предложить составную задачу
            if prop is None:  # предложить не удалось
                continue  # следующее предложение
            sp, _ = prop  # задача (родители не нужны)
            if sp.name in model.experts:  # уже выучена
                continue  # эта задача уже выучена — пропустить
            proposed += 1  # предложено + 1
            ev = model.learn(  # выучить предложенную задачу
                sp, seed=500_000 + r * 104_729 + k, train_neural=train_neural  # зерно зависит от раунда и номера предложения
            )
            if ev["action"] == "COMMIT":  # принято
                commits += 1  # принятий + 1
        comp = model.compress_library_mdl()  # сжать библиотеку методом минимальной длины описания
        new_th = TheoremMiner().mine(model)  # найти новые теоремы
        nq = model.neural_quality_summary()  # сводка нейросетевого качества
        rounds.append(  # добавить итог раунда
            {  # открытие словаря итога раунда
                "round": r,  # номер раунда
                "experts": len(model.experts),  # число экспертов
                "commits": commits,  # принятий в раунде
                "proposed": proposed,  # предложено задач
                "theorems_total": len(model.theorems),  # всего теорем
                "theorems_new": len(new_th),  # новых теорем
                "sym_bits": comp["after_bits"],  # биты библиотеки после сжатия
                "dag_refactored": comp["refactored"],  # число преобразований графа вызовов
                "neural_bytes": nq["total_neural_bytes"],  # байты нейросетевой части
                "neural_val_acc": round(nq["mean_val_exact_acc"], 4),  # средняя точность адаптеров на val, округлённая до 4 знаков
                "neural_val_mse": round(nq["mean_val_mse"], 4),  # средний квадрат ошибки на val, округлённый до 4 знаков
                "certs_ok": model.verify_certificates()["all_ok"],  # сертификаты корректны
                "ram_avail_gb": round(ram_available_bytes() / 2**30, 2),  # доступная оперативная память в ГиБ: байты / 2^30, округлённо до 2 знаков
                "elapsed": round(time.time() - t0, 3),  # длительность раунда в секундах
            }
        )  # закрытие словаря итога
        (out / f"model_r{r}.json").write_text(  # записать снимок модели раунда
            json.dumps(  # превращение словаря в текст JSON
                {  # словарь снимка
                    "round": r,  # номер раунда
                    "experts": {n: e.to_dict() for n, e in model.experts.items()},  # эксперты как словари
                    "theorems": [vars(t) for t in model.theorems],  # теоремы как словари полей
                },
                indent=2,  # отступ 2
            ),  # закрытие json.dumps
            encoding="utf-8",  # кодировка UTF-8
        )
    return {  # итог словарём
        "rounds": rounds,  # итоги раундов
        "experts": len(model.experts),  # число экспертов
        "theorems": len(model.theorems),  # число теорем
    }

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'propose_noncircular_task', 'grow_deep',  # имя интерфейса
]  # конец перечня
