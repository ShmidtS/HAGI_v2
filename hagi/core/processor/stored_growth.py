"""Нециркулярный цикл роста: обучение экспертов, автопорождение задач, самопроверка.

propose_noncircular_task собирает новую задачу как композицию двух уже
принятых экспертов, не передавая готовое дерево в обучение: внешняя
операция outer от двух аргументов и внутренняя inner (один или два
аргумента). Для inner с одним аргументом: g(x) = inner(x), и ищутся
    xs = {x ∈ D1_inner : |x| ≤ 6, inner допустим на x, g(x) ∈ D1_outer}
    ys = {y ∈ D2_outer : |y| ≤ 6, outer допустен на (g(x), y) для всех x ∈ xs}
Учитель: teacher(x, y) = outer(inner(x), y). Для inner с двумя аргументами:
    xs = {x ∈ D1_inner : |x| ≤ 5, x ∈ D1_outer}
    ys = {y ∈ D2_inner : |y| ≤ 5, для всех x ∈ xs inner и outer допустимы}
Учитель: teacher(x, y) = outer(x, inner(x, y)). Задача принимается, если
|xs| ≥ 3, |ys| ≥ 3 и множество значений учителя на xs × ys содержит более
двух элементов (задача не вырождена). До 20 случайных попыток.

grow_from_suite выполняет max_rounds раундов: обучение по набору задач suite, до 8
порождённых задач, сжатие библиотеки, поиск теорем и аудит происхождения
(при недостоверном аудите — ошибка). В конце возвращаются итоги по раундам.

self_test_unified — проверка: обучение набора, отказ вне области, отказ на
испорченной метке, рост, индукция предиката простоты на отрезке [−20, 20]
и проверка его за пределами области, индукция числа делителей, суммы
делителей, функции Эйлера и факториала, предел переполнения 542!.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import json, math  # импорт стандартной библиотеки для разбора выражений, данных, графов и случайности
import time  # статистика, системные функции, замер времени
from pathlib import Path  # пути к файлам
from typing import Any, Optional, Sequence  # типы для аннотаций
import numpy as np  # численные массивы и генератор случайных чисел
from hagi.core.synthesis.venn import (  # из модуля диаграмм Венна импортировать:
    anf_to_call_dag,  # преобразование поляризованной полиномиальной формы в выражение
    anf_transform,  # преобразование таблицы истинности в полиномиальную форму
    pack_venn_16_flat  # упаковка 16 областей диаграммы Венна в плоский вид
)  # конец импорта
from hagi.core.neural.energy import (  # из энергетического модуля импортировать:
    CRAMEngine  # движок раскраски графа с проверкой
)  # конец импорта
from hagi.core.memory.ledger import (  # из журнала аудита импортировать:
    ram_available_bytes  # доступная оперативная память в байтах
)  # конец импорта
from hagi.core.processor.unified_processor import (  # из единого процессора импортировать:
    StoredExpert,  # запись эксперта версии единый
    GrowthProcessorBase,  # растущий процессор версии единый
    StoredTheoremMiner  # поиск теорем по библиотеке экспертов
)  # конец импорта
from hagi.core.tasks.stored_tasks import TaskSpec, make_task_suite  # имена из модуля «stored_tasks» (прямой импорт вместо обращения через псевдоним пакета)

# ==============================================================================  # разделитель
# 10. Нециркулярный саморазвивающийся цикл роста (grow_from_suite)  # заголовок раздела
# ==============================================================================  # разделитель


def propose_noncircular_task(  # предложить новую составную задачу
    experts: dict[str, StoredExpert],  # принятые эксперты по именам
    rng: np.random.Generator,  # генератор случайных чисел
    round_idx: int,  # номер раунда для имени задачи
) -> Optional[tuple[TaskSpec, tuple[str, ...]]]:  # задача и имена родителей или None
    """Предложить новую невырожденную составную задачу из двух экспертов.

        Возвращает (описание задачи, (outer, inner)) либо None, если
        двухаргументных арифметических экспертов нет или за 20 попыток
        не нашлось невырожденной композиции. Готовое дерево выражения в
        learn не передаётся — только описание задачи с учителем."""
    arith2 = [  # двухаргументные небулевы эксперты
        e for e in experts.values() if e.program.arity == 2 and e.family != "BoolVenn"  # арность 2 и семейство не BoolVenn
    ]  # конец списка
    arith1 = [  # одноаргументные небулевы эксперты
        e for e in experts.values() if e.program.arity == 1 and e.family != "BoolVenn"  # арность 1 и семейство не BoolVenn
    ]  # конец списка
    if not arith2:  # внешних операций нет
        return None  # задачу не построить
    pool = arith2 + arith1  # внутренние операции — все арифметические эксперты
    for _ in range(20):  # до 20 попыток
        outer = arith2[int(rng.choice(len(arith2)))]  # случайная внешняя операция
        inner = pool[int(rng.choice(len(pool)))]  # случайная внутренняя операция
        if inner.program.arity == 1:  # внутренняя операция с одним аргументом
            xs = tuple(  # допустимые первые аргументы
                x  # значение x
                for x in inner.domains[0]  # из первой области inner
                if abs(x) <= 6  # по модулю не больше 6: |x| ≤ 6
                and inner.supports((x,))  # inner допустим на x
                and inner.program.eval((x,), experts) in outer.domains[0]  # результат inner(x) лежит в первой области outer
            )  # конец tuple
            ys = tuple(  # допустимые вторые аргументы
                y  # значение y
                for y in outer.domains[1]  # из второй области outer
                if abs(y) <= 6  # по модулю не больше 6: |y| ≤ 6
                and all(  # для всех x из xs
                    outer.supports((inner.program.eval((x,), experts), y)) for x in xs  # outer допустим на (inner(x), y)
                )  # конец all
            )  # конец tuple
            if len(xs) < 3 or len(ys) < 3:  # область слишком мала
                continue  # следующая попытка
            vals = {  # значения композиции на xs × ys
                outer.program.eval((inner.program.eval((x,), experts), y), experts)  # композиция: outer(inner(x), y)
                for x in xs  # по x
                for y in ys  # по y
            }  # конец множества
            if len(vals) <= 2:  # не более двух различных значений — вырождено
                continue  # следующая попытка
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(  # учитель с закреплёнными внешней o, внутренней i операциями и средой env
                (i.program.eval((int(x),), env), int(y)), env  # композиция: outer(inner(x), y)
            )  # конец lambda
            sp = TaskSpec(  # описание задачи
                f"V8_AUTO_R{round_idx}_{inner.name}_IN_{outer.name}",  # имя: раунд, внутренняя и внешняя операции
                2,  # два аргумента
                (xs, ys),  # области значений
                teacher,  # учитель
                valid_rule=outer.valid_rule,  # правило допустимости внешней операции
                family="auto-composed",  # семейство — автокомпозиция
            )  # конец TaskSpec
            return sp, (outer.name, inner.name)  # задача и имена родителей
        else:  # внутренняя операция с двумя аргументами
            xs = tuple(  # допустимые первые аргументы
                x for x in inner.domains[0] if abs(x) <= 5 and x in outer.domains[0]  # x из первой области inner, |x| ≤ 5, x из первой области outer
            )  # конец tuple
            ys_cand = tuple(y for y in inner.domains[1] if abs(y) <= 5)  # кандидаты вторых аргументов: y ∈ D2_inner, |y| ≤ 5
            ys = tuple(  # отобранные вторые аргументы
                y  # значение y
                for y in ys_cand  # из кандидатов
                if all(  # для всех x из xs
                    inner.supports((x, y))  # inner допустим на (x, y)
                    and outer.supports((x, inner.program.eval((x, y), experts)))  # outer допустим на (x, inner(x, y))
                    for x in xs  # по x
                )  # конец all
            )  # конец tuple
            if len(xs) < 3 or len(ys) < 3:  # область слишком мала
                continue  # следующая попытка
            vals = {  # значения композиции на xs × ys
                outer.program.eval((x, inner.program.eval((x, y), experts)), experts)  # композиция: outer(x, inner(x, y))
                for x in xs  # по x
                for y in ys  # по y
            }  # конец множества
            if len(vals) <= 2:  # не более двух различных значений — вырождено
                continue  # следующая попытка
            teacher = lambda x, y, o=outer, i=inner, env=experts: o.program.eval(  # учитель с закреплёнными внешней o, внутренней i операциями и средой env
                (int(x), i.program.eval((int(x), int(y)), env)), env  # композиция: outer(x, inner(x, y))
            )  # конец lambda
            sp = TaskSpec(  # описание задачи
                f"V8_AUTO_R{round_idx}_{inner.name}_IN_{outer.name}",  # имя: раунд, внутренняя и внешняя операции
                2,  # два аргумента
                (xs, ys),  # области значений
                teacher,  # учитель
                valid_rule=inner.valid_rule,  # правило допустимости внутренней операции
                family="auto-composed",  # семейство — автокомпозиция
            )  # конец TaskSpec
            return sp, (outer.name, inner.name)  # задача и имена родителей
    return None  # подходящей композиции не нашлось


def grow_from_suite(  # запустить цикл роста
    model: GrowthProcessorBase,  # растущий процессор
    suite: Sequence[TaskSpec],  # набор базовых задач
    max_rounds: int = 4,  # число раундов
    train_neural: bool = True,  # обучать ли нейронные волокна
    out_dir: str = "work/v8_grow",  # каталог для сохранения моделей
) -> dict[str, Any]:  # итоги по раундам
    """Выполнить max_rounds раундов роста модели model на наборе задач suite.

        Раунд: обучение по suite; до 8 порождённых задач; сжатие библиотеки;
        поиск теорем; аудит происхождения (недостоверный аудит — ошибка);
        сохранение модели. Возвращает записи по раундам и итоговое
        состояние: число экспертов, число теорем, происхождение и граф
        вызовов."""
    if int(max_rounds) < 0:  # отрицательное число раундов
        raise ValueError("max_rounds must be nonnegative")  # ошибка: число раундов должно быть неотрицательным
    out = Path(out_dir)  # каталог вывода
    out.mkdir(parents=True, exist_ok=True)  # создать каталог (с родительскими), если его нет
    rnd = np.random.default_rng(20260101)  # генератор случайных чисел с фиксированным зерном
    rounds = []  # записи по раундам
    for r in range(max_rounds):  # по раундам
        t0 = time.time()  # начало раунда
        commits = 0  # число принятых экспертов в раунде
        for sp in suite:  # по базовым задачам
            ev = model.learn(  # обучиться задаче
                sp, seed=10_000 + r * 7919 + len(sp.name), train_neural=train_neural  # зерно: 10000 + r·7919 + длина имени задачи
            )  # конец learn
            if ev.action == "COMMIT":  # эксперт принят
                commits += 1  # счётчик принятых + 1
        proposed = 0  # число предложенных задач
        for k in range(8):  # до 8 попыток порождения
            prop = propose_noncircular_task(model.experts, rnd, r)  # предложить составную задачу
            if prop is None:  # предложить не удалось
                continue  # следующая попытка
            sp, parents = prop  # задача и имена родителей
            if sp.name in model.experts:  # такой эксперт уже есть
                continue  # следующая попытка
            proposed += 1  # счётчик предложенных + 1
            ev = model.learn(  # обучиться порождённой задаче
                sp,  # задача
                seed=500_000 + r * 104_729 + k,  # зерно: 500000 + r·104729 + k
                train_neural=train_neural,  # обучать ли нейронное волокно
                parents=parents,  # имена родителей
            )  # конец learn
            if ev.action == "COMMIT":  # эксперт принят
                commits += 1  # счётчик принятых + 1
        comp = model.compress_library_mdl()  # сжатие библиотеки по длине описания
        new_th = StoredTheoremMiner().mine(model)  # новые теоремы по библиотеке
        audit = model.verify_lineage()  # аудит происхождения и графа вызовов
        if not audit["valid"]:  # аудит не пройден
            raise RuntimeError(  # ошибка времени выполнения
                f"growth produced invalid provenance/call graph: {audit}"  # текст: рост дал недостоверное происхождение или граф вызовов
            )  # конец raise
        rounds.append(  # запись раунда
            {  # словарь показателей
                "round": r,  # номер раунда
                "experts": len(model.experts),  # число экспертов
                "commits": commits,  # принятых за раунд
                "proposed": proposed,  # предложенных задач
                "theorems_total": len(model.theorems),  # всего теорем
                "theorems_new": len(new_th),  # новых теорем
                "sym_bits": comp["after_bits"],  # битов в сжатой библиотеке
                "dag_refactored": comp["refactored"],  # был ли перестроен граф вызовов
                "estimated_packed_bytes": model.total_neural_bytes(),  # оценка размера нейронных весов в байтах
                "unique_programs": len(  # число различных программ
                    {  # множество текстов программ
                        json.dumps(e.program.to_dict(), sort_keys=True)  # JSON словаря программы с сортировкой ключей
                        for e in model.experts.values()  # по экспертам
                    }  # конец множества
                ),  # конец len
                "experts_with_parents": audit["growth_lineage"]["nodes_with_parents"],  # экспертов с родителями
                "lineage_edges": audit["growth_lineage"]["edges"],  # рёбер происхождения
                "ram_avail_gb": round(ram_available_bytes() / 2**30, 2),  # доступная память в ГБ: байты / 2^30, округление до 2 знаков
                "elapsed": round(time.time() - t0, 3),  # длительность раунда в секундах, округление до 3 знаков
            }  # конец словаря
        )  # конец append
        model.save(out / f"model_r{r}.json")  # сохранить модель раунда r
    final_audit = model.verify_lineage()  # итоговый аудит происхождения
    return {  # итоговый словарь
        "rounds": rounds,  # записи по раундам
        "experts": len(model.experts),  # число экспертов
        "theorems": len(model.theorems),  # число теорем
        "lineage": final_audit["growth_lineage"],  # происхождение
        "call_dag": final_audit["call_dag"],  # граф вызовов
    }  # конец словаря


def self_test_unified() -> None:  # самопроверка
    """Самопроверка: обучение набора, отказы, рост, индукция четырёх семейств задач.

        Прерывается ошибкой утверждения при любом расхождении; в конце
        печатает строку о прохождении."""
    suite = make_task_suite()  # базовый набор задач
    model = GrowthProcessorBase(node_budget=6000)  # процессор с бюджетом 6000 узлов
    for i, sp in enumerate(suite):  # по задачам с их номерами
        ev = model.learn(sp, seed=10_000 + i, train_neural=True)  # обучиться с зерном 10000 + i и нейронным волокном
        assert ev.action in ("COMMIT", "REUSE"), (sp.name, ev)  # каждая задача принята или уже известна
    check = model.verify_all(suite)  # проверка всех экспертов на наборе
    assert check["accuracy"] == 1.0, check  # точность равна 1.0
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"  # 100 вне области сложения — отказ
    poisoned = GrowthProcessorBase().learn(  # обучение с испорченной меткой на задаче ADD
        next(s for s in suite if s.name == "ADD"), seed=77, poison_one_label=True  # первая задача с именем ADD, зерно 77, порча метки
    )  # конец learn
    assert poisoned.action == "HOLD"  # с испорченной меткой эксперт не принимается
    rep = grow_from_suite(model, suite, max_rounds=2, out_dir=".omc/v8_grow_test")  # два раунда роста
    assert rep["experts"] >= len(suite)  # экспертов не меньше, чем базовых задач
    # индукция ограниченной свёртки, не зависящая от имени задачи: обучение только на [−20, 20],
    # затем проверка найденного предиката диапазона за пределами подтверждённой области
    prime_domain = tuple(range(-20, 21))  # область обучения: целые от −20 до 20
    prime_task = TaskSpec(  # задача предиката простоты
        "BOOLEAN_PATTERN_A",  # нейтральное имя
        1,  # один аргумент
        (prime_domain,),  # область
        lambda x: int(x >= 2 and all(x % d for d in range(2, math.isqrt(x) + 1))),  # 1, если x ≥ 2 и нет делителя d в [2, ⌊√x⌋]
        family="unseen-boolean-arithmetic",  # семейство
    )  # конец TaskSpec
    prime_model = GrowthProcessorBase(node_budget=1000)  # отдельный процессор с бюджетом 1000 узлов
    prime_event = prime_model.learn(prime_task, seed=88, train_neural=False)  # обучение без нейронного волокна
    assert (  # задача принята и в тексте программы есть квантор всеобщности
        prime_event.action == "COMMIT" and "forall" in prime_event.program  # условие
    ), prime_event  # сообщение при неудаче
    prime_expr = prime_model.experts[prime_task.name].program  # найденная программа
    for n in (101, 103, 104, 997, 1009, 1024, 104729, 104730, 1_000_003):  # проверочные числа вне области
        expected = int(n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1)))  # верный ответ: простой ли n
        assert prime_expr.eval((n,), prime_model.experts) == expected, n  # программа даёт верный ответ
    # арифметические агрегаты ограниченной свёртки: число делителей, сумма делителей, функция Эйлера, факториал.
    # Метки задач нейтральны; семейство гипотез не смотрит на имена.
    aggregate_domain = tuple(range(-10, 51))  # область обучения: целые от −10 до 50
    aggregate_tasks = [  # четыре задачи
        TaskSpec(  # число делителей
            "ARITHMETIC_AGGREGATE_A",  # имя
            1,  # один аргумент
            (aggregate_domain,),  # область
            lambda n: 0 if n <= 0 else sum(1 for d in range(1, n + 1) if n % d == 0),  # 0 при n ≤ 0, иначе Σ_{d=1..n} 1{d | n}
            family="unseen-integer-aggregate",  # семейство
        ),  # конец задачи
        TaskSpec(  # сумма делителей
            "ARITHMETIC_AGGREGATE_B",  # имя
            1,  # один аргумент
            (aggregate_domain,),  # область
            lambda n: 0 if n <= 0 else sum(d for d in range(1, n + 1) if n % d == 0),  # 0 при n ≤ 0, иначе Σ_{d=1..n, d | n} d
            family="unseen-integer-aggregate",  # семейство
        ),  # конец задачи
        TaskSpec(  # функция Эйлера
            "ARITHMETIC_AGGREGATE_C",  # имя
            1,  # один аргумент
            (aggregate_domain,),  # область
            lambda n: (  # значение — условное выражение
                0 if n <= 0 else sum(1 for d in range(1, n + 1) if math.gcd(n, d) == 1)  # 0 при n ≤ 0, иначе Σ_{d=1..n} 1{НОД(n, d) = 1}
            ),  # конец lambda
            family="unseen-integer-aggregate",  # семейство
        ),  # конец задачи
        TaskSpec(  # факториал
            "ARITHMETIC_AGGREGATE_D",  # имя
            1,  # один аргумент
            (aggregate_domain,),  # область
            lambda n: 0 if n < 0 else math.factorial(n),  # 0 при n < 0, иначе n!
            family="unseen-integer-aggregate",  # семейство
        ),  # конец задачи
    ]  # конец списка задач
    aggregate_model = GrowthProcessorBase(node_budget=1000)  # процессор для агрегатов с бюджетом 1000 узлов
    for aggregate_task in aggregate_tasks:  # по задачам
        aggregate_event = aggregate_model.learn(  # обучиться без нейронного волокна
            aggregate_task, seed=606, train_neural=False  # зерно 606
        )  # конец learn
        assert aggregate_event.action == "COMMIT", (  # задача должна быть принята
            aggregate_task.name,  # имя задачи
            aggregate_event,  # событие
        )  # конец assert
        assert aggregate_model.verify_all([aggregate_task])["accuracy"] == 1.0  # точность на всей области равна 1.0
    aggregate_expressions = [  # найденные программы
        aggregate_model.experts[t.name].program for t in aggregate_tasks  # программа эксперта каждой задачи
    ]  # конец списка
    for value in (51, 60, 97, 100, 997):  # значения вне области обучения
        expected = [  # верные ответы первых трёх задач
            sum(1 for d in range(1, value + 1) if value % d == 0),  # число делителей
            sum(d for d in range(1, value + 1) if value % d == 0),  # сумма делителей
            sum(1 for d in range(1, value + 1) if math.gcd(value, d) == 1),  # функция Эйлера
        ]  # конец списка
        actual = [  # ответы программ
            expr.eval((value,), aggregate_model.experts)  # программа на value
            for expr in aggregate_expressions[:3]  # первые три программы
        ]  # конец списка
        assert actual == expected, (value, actual, expected)  # ответы совпадают с верными
        if value <= 100:  # факториал проверяется только до 100
            assert aggregate_expressions[3].eval(  # программа факториала на value
                (value,), aggregate_model.experts  # аргументы
            ) == math.factorial(value)  # равна value!
    for value in (51, 60, 97, 100, 500):  # значения вне области обучения
        assert aggregate_expressions[3].eval(  # программа факториала на value
            (value,), aggregate_model.experts  # аргументы
        ) == math.factorial(value)  # равна value!
    # явный предел переполнения произведения: 542! больше 4096-битного накопителя
    with np.testing.assert_raises(OverflowError):  # ожидается ошибка переполнения
        aggregate_expressions[3].eval((542,), aggregate_model.experts)  # факториал 542!
    # проверка раскраски, упаковки диаграмм Венна и полиномиальной формы
    r = CRAMEngine.solve_coloring([(0, 1), (1, 2), (2, 3), (3, 0)], 4, 2)  # раскраска цикла из 4 вершин в 2 цвета
    assert r["verified"]  # раскраска проверена
    assert len(pack_venn_16_flat()) == 13  # плоская упаковка содержит 13 элементов
    tt = anf_transform([0, 1, 1, 0])  # полиномиальная форма таблицы [0, 1, 1, 0]
    assert anf_to_call_dag(tt, 2).eval((1, 0)) == 1  # выражение на входе (1, 0) равно 1
    print(  # строка о прохождении
        f"SELF_TEST PASS: {check['committed']} base experts acc 1.0, growth -> {rep['experts']} experts, "  # число базовых экспертов и экспертов после роста
        f"{len(model.theorems)} theorems, prime + tau/sigma/phi/factorial induction PASS, "  # число теорем и индукция
        f"estimated_packed_bytes={model.total_neural_bytes()}"  # оценка размера в байтах
    )  # конец print


# ==============================================================================  # разделитель
# ЧАСТЬ IV — слой версии 9.0  # заголовок раздела
# ==============================================================================  # разделитель

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'propose_noncircular_task', 'grow_from_suite', 'self_test_unified',  # имя интерфейса
]  # конец перечня
