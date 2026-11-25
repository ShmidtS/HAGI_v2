"""Автономный эксперимент: обучить растущий математический процессор на наборе задач.

Файл запускает один и тот же опыт для нескольких начальных чисел генератора
и собирает метрики в словарь:
  * обучение идёт стадиями (двухаргументные функции Венна, унарные, бинарные,
    составные тернарные задачи); внутри стадии порядок задач перемешивается
    генератором с числом seed + 101·длина_имени_стадии;
  * для каждого начального числа фиксируются: число закреплённых экспертов и доля
    commit_rate = закреплено / число задач, точность символьных программ при
    исчерпывающей проверке, средняя точность и средний корень из потери
    нейронных надстроек на отложенной выборке, число отказов на входах вне
    области определения, реакция на ошибочную метку и на неизвестную функцию;
  * по всем начальным числам считаются среднее и стандартное отклонение
    sd = sqrt(Σ(v − mean)² / (n − 1)).
Результат сохраняется в JSON, модель основного запуска — в отдельный JSON,
по задачам основного запуска — в CSV, и пишется итоговый текст в Markdown.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import json, math  # модули стандартной библиотеки для разбора, сортировки, перебора, JSON, математики, окружения и случайности
import time  # статистика, параметры интерпретатора и часы
from pathlib import Path  # путь к файлу
from typing import Any, Optional, Sequence  # аннотации типов
import numpy as np  # численные массивы
from hagi.core.tasks.stored_tasks import (  # открыть импорт из модуля набора задач
    TaskSpec,  # описание одной задачи
    make_task_suite  # функция, создающая набор задач
)  # конец импорта из tasks
from hagi.core.processor.experts import (  # открыть импорт из модуля экспертов
    GrowingMathProcessor,  # растущий математический процессор
    GrowthEvent  # запись о результате одного шага обучения
)  # конец импорта из experts
from hagi.base.config import STATE_FORMAT  # единая метка формата состояния (без номера версии)
from hagi.core.processor.benchmark import main  # имя main определено в части «benchmark» (явный импорт вместо инъекции)

# ---------------------------- Автономный эксперимент ----------------------------  # заголовок раздела


def run_experiment(
    seeds: Sequence[int] = (0, 1, 2), out_dir: str = "."  # начальные числа и каталог вывода
) -> dict[str, Any]:  # возврат: итоговый словарь
    """Провести эксперимент для каждого начального числа из seeds и вернуть итоговый словарь.

        seeds: начальные числа генератора, out_dir: каталог для файлов
        (создаётся при отсутствии). Для каждого числа создаётся новый процессор,
        обучается по стадиям, проверяется; строка результатов per_seed содержит
        метрики этого числа. Возврат: словарь с ключами version, scope, per_seed,
        aggregate (средние и отклонения по всем числам) и interpretation."""
    out = Path(out_dir)  # каталог вывода
    out.mkdir(parents=True, exist_ok=True)  # создать каталог вместе с родительскими; существующий — не ошибка
    suite = make_task_suite()  # набор задач
    per_seed = []  # результаты по каждому начальному числу
    main_model: Optional[GrowingMathProcessor] = None  # модель основного запуска
    main_seed = int(seeds[0]) if seeds else 0  # основное число — первое из seeds, иначе 0
    start = time.time()  # время начала всего эксперимента
    for seed in seeds:  # по начальному числу
        model = GrowingMathProcessor()  # новый процессор
        seed_t0 = time.time()  # время начала этого запуска
        events: list[GrowthEvent] = []  # события обучения
        # Упорядоченная программа: логика -> унарные -> арифметика -> предикаты и битовые -> композиции.
        stage_order = ["BoolVenn", "unary", "integer-binary", "composed-ternary"]  # порядок стадий
        growth_curve = []  # кривая роста по стадиям
        for stage in stage_order:  # по стадиям
            stage_tasks = [s for s in suite if s.family == stage]  # задачи текущей стадии
            # Порядок внутри стадии детерминирован, но перемешивается генератором, зависящим от числа, чтобы выявить эффект порядка.
            rng = np.random.default_rng(seed + len(stage) * 101)  # генератор с числом seed + 101·длина_имени_стадии
            if len(stage_tasks) > 1:  # задач больше одной
                order = rng.permutation(len(stage_tasks))  # случайная перестановка номеров
                stage_tasks = [stage_tasks[int(i)] for i in order]  # задачи в перемешанном порядке
            for j, spec in enumerate(stage_tasks):  # по задачам стадии
                events.append(  # добавить событие обучения
                    model.learn(spec, seed=seed * 1009 + j * 17 + len(spec.name))  # обучить; число генератора = seed·1009 + j·17 + длина имени
                )  # конец добавления
            growth_curve.append(  # добавить точку кривой роста
                {  # словарь точки
                    "stage": stage,  # имя стадии
                    "new_experts": len(stage_tasks),  # число задач стадии
                    "cumulative_experts": len(model.experts),  # всего экспертов после стадии
                    "cumulative_code_length": sum(  # суммарная длина кода
                        e.code_length for e in model.experts.values()  # Σ длина_кода по экспертам
                    ),  # конец суммы
                }  # конец словаря
            )  # конец добавления
        verification = model.verify_all(suite)  # исчерпывающая проверка всех задач
        committed = len(model.experts)  # число закреплённых экспертов
        neural_scores = [  # точности нейронных надстроек на отложенной выборке
            e.neural.validation_exact_accuracy  # значение точности
            for e in model.experts.values()  # по экспертам
            if e.neural is not None and np.isfinite(e.neural.validation_exact_accuracy)  # нейронная надстройка есть и точность — число
        ]  # конец списка
        neural_rmse = [  # корни из потерь нейронных надстроек
            math.sqrt(max(0.0, e.neural.validation_loss))  # sqrt(max(0, потеря))
            for e in model.experts.values()  # по экспертам
            if e.neural is not None and np.isfinite(e.neural.validation_loss)  # надстройка есть и потеря — число
        ]  # конец списка
        # Проверка защитных условий: каждая известная операция обязана отказать на входе вне своей области.
        ood_checks = ood_abstains = 0  # счётчики проверок и отказов
        for spec in suite:  # по задачам
            if spec.name not in model.experts:  # задача не выучена
                continue  # пропуск
            outside = [d[-1] + 1 for d in spec.domains]  # вход на 1 больше верхних границ всех областей
            result = model.predict(spec.name, *outside)  # прогноз на входе вне области
            ood_checks += 1  # проверок + 1
            ood_abstains += int(result["status"] == "ABSTAIN")  # +1, если ответ — отказ
        # Проверка на отравление: одна ошибочная метка в примерах сложения не должна создавать новый эксперт.
        poison_model = GrowingMathProcessor()  # отдельный процессор
        add_spec = next(s for s in suite if s.name == "ADD")  # задача сложения
        poison_event = poison_model.learn(  # обучить с одной ошибочной меткой
            add_spec, seed=seed + 44001, poison_one_label=True  # число генератора seed + 44001, признак отравления
        )  # конец вызова
        # Неизвестная нетривиальная функция имеет 17 точек в области: предел таблицы 16 -> удержание.
        prime_spec = TaskSpec(  # задача «простое ли число»
            "IS_PRIME",  # имя
            1,  # один аргумент
            (tuple(range(-8, 9)),),  # область: целые от −8 до 8
            lambda x: int(  # учитель: 1, если x ≥ 2 и нет делителя d из [2, ⌊√x⌋]
                x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))  # условие x ≥ 2 ∧ ∀d ∈ [2, ⌊√x⌋]: x mod d ≠ 0
            ),  # конец лямбды
            family="unseen",  # семейство «не встречалась»
        )  # конец описания
        unknown_event = GrowingMathProcessor().learn(prime_spec, seed=seed + 99001)  # обучить новый процессор неизвестной задаче
        elapsed = time.time() - seed_t0  # длительность запуска
        row = {  # строка результатов запуска
            "seed": int(seed),  # начальное число
            "committed": committed,  # закреплено экспертов
            "expected": len(suite),  # ожидаемое число = число задач
            "commit_rate": committed / len(suite),  # доля закреплённых = закреплено / число задач
            "symbolic_exact_accuracy": verification["accuracy"],  # точность символьных программ
            "symbolic_examples_checked": verification["checked"],  # число проверенных примеров
            "neural_holdout_exact_accuracy_mean": (  # средняя нейронная точность
                float(np.mean(neural_scores)) if neural_scores else float("nan")  # среднее, если список не пуст, иначе NaN
            ),  # конец условия
            "neural_holdout_rmse_mean": (  # средний корень из потери
                float(np.mean(neural_rmse)) if neural_rmse else float("nan")  # среднее, если список не пуст, иначе NaN
            ),  # конец условия
            "ood_checks": ood_checks,  # число проверок вне области
            "ood_abstains": ood_abstains,  # число отказов
            "poison_action": poison_event.action,  # действие при отравленной метке
            "unknown_is_prime_action": unknown_event.action,  # действие для неизвестной функции
            "growth_curve": growth_curve,  # кривая роста
            "elapsed_seconds": elapsed,  # длительность в секундах
        }  # конец словаря
        per_seed.append(row)  # добавить строку
        if seed == main_seed:  # основное число
            main_model = model  # сохранить модель
            # Полная детализация по задачам для основного числа.
            with (out / f"hagi_growing_math_seed{seed}_tasks.csv").open(  # открыть CSV-файл по задачам
                "w", encoding="utf-8"  # на запись, кодировка utf-8
            ) as f:  # конец open
                f.write(  # заголовок CSV
                    "task,family,committed,program,code_length,neural_val_exact_acc,neural_val_mse\n"  # имена столбцов
                )  # конец write
                for spec in suite:  # по задачам
                    ex = model.experts.get(spec.name)  # эксперт задачи или None
                    if ex is None:  # эксперт не создан
                        f.write(f"{spec.name},{spec.family},0,,0,,\n")  # строка: задача, семейство, 0, пустая программа, 0, пусто, пусто
                    else:  # эксперт есть
                        na = (  # нейронная точность на отложенной выборке
                            ex.neural.validation_exact_accuracy  # значение
                            if ex.neural  # если надстройка есть
                            else float("nan")  # иначе NaN
                        )  # конец условия
                        nm = ex.neural.validation_loss if ex.neural else float("nan")  # нейронная потеря либо NaN
                        prog = ex.program.pretty().replace('"', '""')  # текст программы; кавычки удвоены для CSV
                        f.write(  # строка CSV
                            f'"{spec.name}","{spec.family}",1,"{prog}",{ex.code_length},{na},{nm}\n'  # задача, семейство, 1, программа, длина кода, точность, потеря
                        )  # конец write
    assert main_model is not None  # модель основного запуска обязана существовать
    model_path = out / "hagi_growing_math_model.json"  # путь к файлу модели
    main_model.save(model_path)  # сохранить модель основного запуска

    # Быстрая проверка на сохранённой и заново загруженной модели: обычный вызов и вход вне области.
    reloaded = GrowingMathProcessor.load(model_path)  # загрузить модель
    reload_ok = reloaded.predict("ADD", 7, -3).get("value") == 4  # 7 + (−3) = 4 после загрузки
    ood_ok = reloaded.predict("ADD", 99, 2).get("status") == "ABSTAIN"  # вход 99 вне области — отказ

    aggregate: dict[str, Any] = {}  # агрегаты по всем начальным числам
    metric_keys = (  # усредняемые метрики
        "committed",  # число экспертов
        "symbolic_exact_accuracy",  # символьная точность
        "neural_holdout_exact_accuracy_mean",  # нейронная точность
        "neural_holdout_rmse_mean",  # нейронный корень из потери
        "ood_abstains",  # число отказов
        "elapsed_seconds",  # длительность
    )  # конец кортежа
    for key in metric_keys:  # по метрикам
        vals = [float(r[key]) for r in per_seed]  # значения метрики по всем запускам
        base = key[:-5] if key.endswith("_mean") else key  # имя без суффикса «_mean»
        aggregate[base + "_mean"] = float(np.mean(vals))  # среднее = Σ v / n
        aggregate[base + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0  # sd = sqrt(Σ(v − mean)² / (n − 1)); при n = 1 — ноль
    aggregate.update(  # добавить служебные агрегаты
        {  # словарь дополнений
            "seeds": list(map(int, seeds)),  # список начальных чисел
            "expected_operations": len(suite),  # ожидаемое число операций
            "finite_support_examples_per_operation": [len(s.inputs()) for s in suite],  # число примеров в области каждой задачи
            "total_finite_support_examples": int(sum(len(s.inputs()) for s in suite)),  # всего примеров
            "reload_roundtrip_pass": bool(reload_ok),  # сохранение и загрузка сохранили ответ
            "ood_abstain_pass": bool(ood_ok),  # отказ на входе вне области получен
            "total_runtime_seconds": time.time() - start,  # общее время эксперимента
        }  # конец словаря
    )  # конец update
    result = {  # итоговый словарь
        "version": STATE_FORMAT,  # версия
        "scope": "integer inputs -8..8 (with task-specific subdomains) + all 16 two-input Boolean functions",  # область: целые входы от −8 до 8 (с подмножествами по задачам) и всех 16 двухаргументных булевых функций
        "per_seed": per_seed,  # результаты по начальным числам
        "aggregate": aggregate,  # агрегаты
        "interpretation": [  # строки толкования
            "Exact results are certified only on each operation's finite declared support.",  # точные результаты сертифицированы только на конечной объявленной области каждой операции
            "The neural fibers are approximate and are not the exact inference path.",  # нейронные надстройки приближённые и не являются точным путём вывода
            "Unknown functions outside the learned library abstain; no claim of all mathematics is made.",  # неизвестные функции вызывают отказ; утверждение обо всей математике не делается
            "The teacher oracle is used by the offline experiment's independent validation/exhaustive audit.",  # учитель используется независимой исчерпывающей проверкой автономного эксперимента
        ],  # конец списка
    }  # конец словаря
    (out / "hagi_growing_math_experiment.json").write_text(  # записать итоги в JSON
        json.dumps(result, indent=2), encoding="utf-8"  # JSON с отступом 2, кодировка utf-8
    )  # конец вызова
    _write_report(result, out / "hagi_growing_math_report.md")  # записать итоговый текст в Markdown
    return result  # вернуть итоги


def _write_report(result: dict[str, Any], path: Path) -> None:
    """Записать итоговый текст эксперимента в Markdown-файл path по словарю result.

        Текст содержит: версию, список начальных чисел, каталог выученных операций,
        таблицу средних и отклонений по метрикам, таблицу по каждому запуску,
    кривую роста первого запуска, раздел о толковании и пределах, список
    родственных формальных спецификаций."""
    agg = result["aggregate"]  # агрегаты
    lines = [  # строки текста
        "# HAGI Growing Math Processor — autonomous experiment",  # заголовок
        "",  # пустая строка
        f"Version: {result['version']}",  # строка версии
        f"Seeds: {', '.join(map(str, agg['seeds']))}",  # начальные числа через запятую
        f"Declared operation set: {agg['expected_operations']} tasks (16 BoolVenn + integer unary/binary + ternary compositions).",  # объявленный набор операций
        "",  # пустая строка
        "## Learned operation catalogue",  # заголовок каталога операций
        "",  # пустая строка
        "- 16 BoolVenn functions: every two-input Boolean truth table (FALSE, NOR, implications, XOR/XNOR, AND/NAND, OR, TRUE, projections and negations).",  # пункт о 16 функциях Венна
        "- 7 unary integer functions: NEG, ABS, SIGN, SQUARE, CUBE, BIT_NOT8, CLAMP_TO_3.",  # пункт об унарных функциях
        "- 23 binary integer functions: ADD, SUB, MUL, FLOORDIV, MOD, GCD, LCM, MIN, MAX, ABSDIFF, EQ/NE/LT/LE/GT/GE, BIT_AND/OR/XOR/NAND8/NOR8/XNOR8, POW_INT.",  # пункт о бинарных функциях
        "- 3 ternary compositions: FMA, CLAMP, MEDIAN3.",  # пункт о тернарных композициях
        "",  # пустая строка
        "The expression interpreter additionally supports bounded integer arithmetic/bitwise expressions and selected real functions (sqrt, exp, log, trigonometric and hyperbolic functions). Those calculator functions are not part of the 49-operation exhaustive-learning score.",  # пояснение об интерпретаторе выражений
        "",  # пустая строка
        "## Aggregate results",  # заголовок агрегатов
        "",  # пустая строка
        "| Metric | Mean | SD |",  # заголовок таблицы
        "|---|---:|---:|",  # выравнивание столбцов
        f"| Committed experts | {agg['committed_mean']:.2f} | {agg['committed_sd']:.2f} |",  # строка: среднее и отклонение числа экспертов с 2 знаками
        f"| Exhaustive symbolic accuracy | {100*agg['symbolic_exact_accuracy_mean']:.2f}% | {100*agg['symbolic_exact_accuracy_sd']:.2f} pp |",  # символьная точность в процентах: 100·значение
        f"| Neural holdout exact accuracy | {100*agg['neural_holdout_exact_accuracy_mean']:.2f}% | {100*agg['neural_holdout_exact_accuracy_sd']:.2f} pp |",  # нейронная точность в процентах
        f"| Neural holdout RMSE | {agg['neural_holdout_rmse_mean']:.4f} | {agg['neural_holdout_rmse_sd']:.4f} |",  # нейронный корень из потери с 4 знаками
        f"| OOD abstentions | {agg['ood_abstains_mean']:.2f} | {agg['ood_abstains_sd']:.2f} |",  # число отказов
        f"| Runtime per seed | {agg['elapsed_seconds_mean']:.2f}s | {agg['elapsed_seconds_sd']:.2f}s |",  # время на запуск
        "",  # пустая строка
        "## Per-seed results",  # заголовок результатов по запускам
        "",  # пустая строка
        "| Seed | Experts | Symbolic exact | Neural holdout exact | Neural RMSE | OOD abstain | Poison | Unknown IS_PRIME |",  # заголовок таблицы по запускам
        "|---:|---:|---:|---:|---:|---:|---|---|",  # выравнивание столбцов
    ]  # конец списка
    for r in result["per_seed"]:  # по результатам запусков
        lines.append(  # добавить строку таблицы
            f"| {r['seed']} | {r['committed']}/{r['expected']} | {100*r['symbolic_exact_accuracy']:.2f}% | {100*r['neural_holdout_exact_accuracy_mean']:.2f}% | {r['neural_holdout_rmse_mean']:.4f} | {r['ood_abstains']}/{r['ood_checks']} | {r['poison_action']} | {r['unknown_is_prime_action']} |"  # число, эксперты, точности в процентах, потеря, отказы/проверки, действия
        )  # конец добавления
    lines += [  # добавить заголовок таблицы роста
        "",  # пустая строка
        "## Growth curve (seed 0)",  # заголовок кривой роста
        "",  # пустая строка
        "| Stage | New experts | Cumulative experts | Cumulative code units |",  # заголовок таблицы
        "|---|---:|---:|---:|",  # выравнивание столбцов
    ]  # конец списка
    for stage in result["per_seed"][0].get("growth_curve", []):  # по стадиям первого запуска
        lines.append(  # добавить строку
            f"| {stage['stage']} | {stage['new_experts']} | {stage['cumulative_experts']} | {stage['cumulative_code_length']} |"  # стадия, новые эксперты, всего экспертов, всего единиц кода
        )  # конец добавления
    lines += [  # добавить заключительные разделы
        "",  # пустая строка
        "## Interpretation and limits",  # заголовок о толковании и пределах
        "",  # пустая строка
        "The operation library grows by adding one expert after program synthesis, held-out validation, and exhaustive comparison over that operation's finite support. Each expert also receives a small trained neural fiber; approximate neural predictions are available separately, but exact inference uses the verified symbolic program.",  # абзац о росте библиотеки
        "",  # пустая строка
        "The 16 two-input Boolean functions are the complete BoolVenn truth-table family. Integer functions are certified only on the finite input ranges recorded in the task table. This is not a proof of correctness for unbounded integers, real analysis, transcendental functions, arbitrary user-defined functions, or every possible mathematical operation. The safe expression interpreter computes selected transcendental functions but does not discover or formally certify them as learned experts.",  # абзац о пределах сертификации
        "",  # пустая строка
        "Out-of-domain inputs abstain. The poisoned-label and unknown-prime probes are expected to HOLD rather than silently commit. The training oracle is available in the offline experiment; a deployed learner needs an independent verifier or new labelled examples to certify a new function.",  # абзац об отказах и проверках
        "",  # пустая строка
        "The design is MDL-inspired: candidates are ordered by AST description length and exact-fit programs are required to pass a hard finite-domain commit check. This is not a complete implementation of the repository's probabilistic/energy CommitGate or its Lean proof obligations. BoolVenn follows the 16-function finite lattice; the growth controller adds experts in stages. These formal components do not establish generalization beyond the specified finite domains.",  # абзац о принципе минимальной длины описания
        "",  # пустая строка
        "## Related Lean specifications",  # заголовок родственных формальных спецификаций
        "",  # пустая строка
        "- BoolVenn: https://github.com/ShmidtS/primes/blob/main/Hagi/Data/BoolVenn.lean",  # ссылка: функции Венна
        "- MDL objective: https://github.com/ShmidtS/primes/blob/main/Hagi/Step/MDLObjective.lean",  # ссылка: критерий минимальной длины описания
        "- Safe commit gate: https://github.com/ShmidtS/primes/blob/main/Hagi/Unified/CommitGate.lean",  # ссылка: безопасное закрепление
        "- Self-development search: https://github.com/ShmidtS/primes/blob/main/Hagi/Growth/SelfDevelopment.lean",  # ссылка: поиск саморазвития
        "- Recursive growth conditions: https://github.com/ShmidtS/primes/blob/main/Hagi/Growth/RecursiveSelfDevelopment.lean",  # ссылка: условия рекурсивного роста
    ]  # конец списка
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # записать строки через перевод строки плюс перевод в конце

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'run_experiment',  # имя интерфейса
]  # конец перечня
