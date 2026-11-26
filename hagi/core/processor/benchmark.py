"""Эксперимент по обучению набора задач на нескольких случайных зернах, самопроверка и запуск из командной строки.

Ключевые идеи:
  * один процесс на одно зерно: зерна независимы, поэтому работа делится между
    ядрами процессора; в каждом процессе создаётся своя модель, по очереди
    изучаются все задачи набора, затем модель сохраняется на диск;
  * порядок задач внутри стадии перемешивается: зерно генератора
    seed + len(стадия)·101, перестановка номеров задач;
  * зерно обучения задачи j: seed·1009 + j·17 + len(имя задачи);
  * агрегат по зернам: среднее mean = Σ x_i / n и несмещённое отклонение
    sd = sqrt(Σ (x_i − mean)² / (n − 1)); при n = 1 sd = 0;
  * проверка отказа вне области: для каждой задачи подаётся вход, у которого каждый
    аргумент на 1 больше максимума своей области: d[-1] + 1; ответ должен быть ABSTAIN;
  * проверка на испорченную метку и на незнакомую задачу: обе должны дать HOLD.

Экспортирует: run_optimal_experiment (эксперимент по зернам), self_test_optimal (проверка
ключевых свойств), main (точка входа), STATE_FORMAT (метка формата состояния).
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов
import argparse, json, math, os  # модули стандартной библиотеки: аргументы командной строки, абстрактное дерево, копирование, куча, переборы, JSON, математика, ОС, случайность
import time  # модули стандартной библиотеки: статистика, системные параметры, время
from pathlib import Path  # объектный путь к файлу
from typing import Any, Sequence  # типы аннотаций: произвольный, вызываемый, перечислимый, необязательный, последовательность
import numpy as np  # численные массивы
from hagi.core.tasks.stored_tasks import (  # импорт из модуля описания задач
    TaskSpec,  # описание задачи: имя, арность, области, учитель
    make_task_suite  # построение полного набора задач
)  # конец импорта
from concurrent.futures import ProcessPoolExecutor  # имена из «futures» (модуль, где они определены)
from hagi.core.synthesis.prefilter import _sys  # имена из «prefilter» (модуль, где они определены)
from hagi.core.language.code_length import (  # импорт из модуля оценки длины кода
    _program_description_bits  # длина описания программы в битах
)  # конец импорта
from hagi.core.synthesis.best_first import (  # импорт из модуля поиска лучший-первый
    SynthesisStats  # счётчики синтеза программ
)  # конец импорта
from hagi.core.processor.optimal_processor import (  # импорт из модуля процессора предыдущего поколения
    OptimalMathProcessor  # растущий символьный процессор
)  # конец импорта
from hagi.core.synthesis.induction import (  # импорт из модуля поиска теорем
    TheoremMiner  # поиск алгебраических свойств экспертов
)  # конец импорта
from hagi.core.processor.self_development import (  # импорт из модуля саморазвития
    grow_forever  # ограниченный цикл композиционного роста библиотеки
)  # конец импорта

# ---------------------------- Эксперимент и замеры ---------------------------


def _run_seed_worker(args: tuple[int, str, str]) -> dict[str, Any]:  # args — (зерно, устройство, каталог результатов)
    """Обучить модель на одном зерне и вернуть замеры по этому зерну.

        args = (seed, device, out_dir). Зерна независимы, поэтому функция
        выполняется в отдельном процессе и масштабируется по числу ядер;
        каждый процесс ограничивает только собственную память. Модель
        сохраняется в out_dir. Возвращает словарь: commit_rate = |experts| / |suite|,
        точность на всей области, суммарная длина кода Σ code_length, число
        порождённых кандидатов и вычислений, отказы вне области, действия при
        испорченной метке и незнакомой задаче, elapsed = t_конца − t_начала."""
    seed, device, out_dir = args  # распаковать зерно, устройство и каталог
    train_neural = os.environ.get("HAGI_NO_NEURAL") != "1"  # обучать нейронную часть, если переменная окружения не равна "1"
    out = Path(out_dir)  # каталог результатов
    suite = make_task_suite()  # полный набор задач
    model = OptimalMathProcessor(device=device)  # новая модель на заданном устройстве
    t0 = time.time()  # момент начала
    stats0 = SynthesisStats()  # начальные счётчики
    stats0.candidates_generated = model.stats.candidates_generated  # начальное число порождённых кандидатов
    stats0.evals = model.stats.evals  # начальное число вычислений
    stage_order = ["BoolVenn", "unary", "integer-binary", "composed-ternary"]  # порядок стадий: от булевых к составным
    for stage in stage_order:  # по стадиям
        stage_tasks = [s for s in suite if s.family == stage]  # задачи данной стадии
        rng = np.random.default_rng(seed + len(stage) * 101)  # генератор по зерну seed + len(стадия)·101
        if len(stage_tasks) > 1:  # задач больше одной — есть что перемешивать
            order = rng.permutation(len(stage_tasks))  # случайная перестановка номеров задач
            stage_tasks = [stage_tasks[int(i)] for i in order]  # задачи в перемешанном порядке
        for j, spec in enumerate(stage_tasks):  # по задачам с номером j
            model.learn(  # изучить задачу
                spec,  # описание задачи
                seed=seed * 1009 + j * 17 + len(spec.name),  # зерно обучения: seed·1009 + j·17 + len(имя)
                train_neural=train_neural,  # обучать ли нейронную часть
            )
    verification = model.verify_all(suite)  # проверка всех задач на полной области
    ood_checks = ood_abstains = 0  # число проверок вне области и число отказов
    for spec in suite:  # по всем задачам
        if spec.name not in model.experts:  # задача не изучена
            continue  # пропустить
        outside = [d[-1] + 1 for d in spec.domains]  # вход вне области: каждый аргумент = max области + 1
        ood_checks += 1  # проверок: +1
        ood_abstains += int(model.predict(spec.name, *outside)["status"] == "ABSTAIN")  # +1, если ответ — отказ
    poison_model = OptimalMathProcessor()  # отдельная модель для проверки испорченной метки
    poison_event = poison_model.learn(  # изучить сложение с испорченной меткой
        next(s for s in suite if s.name == "ADD"),  # задача сложения
        seed=seed + 44001,  # зерно
        poison_one_label=True,  # испортить одну метку
    )
    prime_spec = TaskSpec(  # незнакомая задача: простота числа
        "IS_PRIME",  # имя
        1,  # один аргумент
        (tuple(range(-8, 9)),),  # область: целые от −8 до 8
        lambda x: int(x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))),  # 1, если x >= 2 и x не делится ни на одно d из 2..⌊√x⌋
        family="unseen",  # семейство «не встречалось»
    )
    unknown_event = OptimalMathProcessor().learn(prime_spec, seed=seed + 99001)  # результат обучения незнакомой задачи
    reuse_count = sum(1 for e in model.growth_log if e.reason == "expert-reuse")  # число событий повторного использования эксперта
    gen = model.stats.candidates_generated - stats0.candidates_generated  # порождено кандидатов за эксперимент: разность счётчиков
    evs = model.stats.evals - stats0.evals  # число вычислений за эксперимент
    model.save(out / f"hagi_v7_model_seed{seed}.json")  # сохранить модель в файл с номером зерна
    return {  # замеры по зерну
        "seed": int(seed),  # зерно
        "committed": len(model.experts),  # число изученных экспертов
        "expected": len(suite),  # ожидаемое число — размер набора
        "commit_rate": len(model.experts) / len(suite),  # доля изученных = committed / expected
        "symbolic_exact_accuracy": verification["accuracy"],  # точность на полной области
        "total_code_length": sum(e.code_length for e in model.experts.values()),  # суммарная длина кода Σ code_length
        "candidates_generated": gen,  # порождено кандидатов
        "program_evals": evs,  # число вычислений программ
        "sieve_survivors": model.stats.sieve_survivors,  # выживших после просеивания
        "expert_reuse_commits": reuse_count,  # число повторных использований
        "ood_abstains": ood_abstains,  # отказов вне области
        "ood_checks": ood_checks,  # проверок вне области
        "poison_action": poison_event.action,  # действие при испорченной метке
        "unknown_is_prime_action": unknown_event.action,  # действие для незнакомой задачи
        "elapsed_seconds": time.time() - t0,  # затрачено секунд: now − t0
    }


def run_optimal_experiment(  # провести эксперимент на нескольких зернах
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),  # зерна случайности
    out_dir: str = ".omc/hagi_v7_results",  # каталог результатов
    workers: int = 0,  # число процессов (0 и меньше — выбрать автоматически)
    device: str = "cpu",  # устройство вычислений
) -> dict[str, Any]:  # аннотация возврата
    """Обучить модель на каждом зерне из seeds и вернуть результаты с агрегатами.

        Результаты per_seed сортируются по зерну. Модель первого зерна после
        загрузки проверяется и переименовывается в общий файл. Возвращает
        {"version", "per_seed", "aggregate"}; aggregate содержит для ряда метрик
        среднее и отклонение по зернам."""
    out = Path(out_dir)  # каталог результатов
    out.mkdir(parents=True, exist_ok=True)  # создать каталог (с родителями; существует — не ошибка)
    suite = make_task_suite()  # набор задач (создаётся, чтобы конструировать макет задачи)
    start = time.time()  # момент начала эксперимента
    args = [(int(s), device, str(out)) for s in seeds]  # аргументы рабочих процессов: (зерно, устройство, каталог)
    if workers <= 0:  # авто-выбор числа процессов
        workers = min(len(args), max(1, (os.cpu_count() or 2) - 1))  # min(число зёрен, max(1, число ядер − 1)); ядер нет — считаем 2
    workers = min(workers, len(args))  # процессов не больше, чем зёрен
    if workers > 1:  # несколько процессов
        with ProcessPoolExecutor(max_workers=workers) as ex:  # пул процессов
            per_seed = list(ex.map(_run_seed_worker, args))  # результаты по зернам (порядок как в args)
    else:  # один процесс
        per_seed = [_run_seed_worker(a) for a in args]  # последовательный запуск по зернам
    per_seed.sort(key=lambda r: r["seed"])  # отсортировать по зерну
    main_model_path = out / f"hagi_v7_model_seed{int(seeds[0])}.json"  # файл модели первого зерна
    if main_model_path.exists():  # файл сохранён
        OptimalMathProcessor.load(main_model_path)  # загрузить модель (проверка читаемости)
        main_model_path.replace(out / "hagi_v7_model.json")  # переместить в общий файл модели

    aggregate: dict[str, Any] = {}  # агрегаты по зернам
    for key in (  # метрики, усредняемые по зернам
        "committed",  # число экспертов
        "symbolic_exact_accuracy",  # точность
        "total_code_length",  # длина кода
        "candidates_generated",  # кандидаты
        "program_evals",  # вычисления
        "ood_abstains",  # отказы вне области
        "elapsed_seconds",  # время
    ):  # конец списка метрик
        vals = [float(r[key]) for r in per_seed]  # значения метрики по зернам
        aggregate[key + "_mean"] = float(np.mean(vals))  # среднее mean = Σ x_i / n
        aggregate[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0  # отклонение sd = sqrt(Σ (x_i − mean)² / (n − 1)); при n = 1 — 0
    aggregate.update(  # добавить прочие агрегаты
        {  # словарь дополнений
            "seeds": list(map(int, seeds)),  # использованные зерна
            "expert_reuse_commits_total": int(  # всего повторных использований
                sum(r["expert_reuse_commits"] for r in per_seed)  # Σ по зернам
            ),
            "poison_all_hold": all(r["poison_action"] == "HOLD" for r in per_seed),  # испорченная метка везде дала отказ
            "unknown_all_hold": all(  # незнакомая задача везде дала отказ
                r["unknown_is_prime_action"] == "HOLD" for r in per_seed  # по зернам
            ),
            "reload_roundtrip_pass": OptimalMathProcessor.load(  # сохранение и загрузка согласованы
                out / "hagi_v7_model.json"  # загружается общий файл модели
            )
            .predict("ADD", 7, -3)  # ответ загруженной модели для 7 + (−3)
            .get("value")  # взять значение
            == 4,  # истинно, если значение равно 4
            "total_runtime_seconds": time.time() - start,  # общее время: now − start
        }
    )
    result = {"version": STATE_FORMAT, "per_seed": per_seed, "aggregate": aggregate}  # итог: версия, по зернам, агрегаты
    (out / "hagi_v7_experiment.json").write_text(  # записать итог в файл
        json.dumps(result, indent=2), encoding="utf-8"  # JSON с отступом 2 в кодировке UTF-8
    )
    return result  # вернуть итог


def self_test_optimal() -> None:  # самопроверка ключевых свойств
    """Проверить обучение всего набора, отказы вне области, испорченную метку, полиномы и теоремы.

        Все задачи должны изучиться (COMMIT), точность на полной области равна 1,
        вход вне области даёт ABSTAIN, испорченная метка даёт HOLD, полиномы
        одной, двух и трёх переменных находятся в замкнутой форме, поиск теорем
        находит проверенные утверждения. При нарушении вызывается AssertionError."""
    suite = make_task_suite()  # набор задач
    model = OptimalMathProcessor()  # новая модель
    for i, spec in enumerate(suite):  # по задачам с номером i
        ev = model.learn(spec, seed=10_000 + i, train_neural=False)  # изучить без нейронной части, зерно 10000 + i
        assert ev.action == "COMMIT", (spec.name, ev)  # задача обязана быть изучена
    check = model.verify_all(suite)  # проверка на полной области
    assert check["committed"] == len(suite) and check["accuracy"] == 1.0  # изучены все и точность 1
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"  # 100 вне области сложения — отказ
    # Точный шаблон медианы трёх проверяется на всей объявленной области.
    med = model.experts["MEDIAN3"].program  # программа медианы
    assert med.eval((1, 5, 3)) == 3 and med.eval((9, 2, 7)) == 7  # медиана (1,5,3) = 3; медиана (9,2,7) = 7
    poisoned = OptimalMathProcessor().learn(  # обучение с испорченной меткой
        next(s for s in suite if s.name == "ADD"),  # задача сложения
        seed=77,  # зерно
        poison_one_label=True,  # испортить метку
        train_neural=False,  # без нейронной части
    )
    assert poisoned.action == "HOLD"  # испорченная задача отклонена
    # Индукция точной формулы: учитель 2x^3 − x^2 + 7x − 4 должен быть изучен
    # как полином в замкнутой форме с точной (машинной) арифметикой
    poly_spec = TaskSpec(  # задача кубического полинома
        "POLY_CUBIC",  # имя
        1,  # один аргумент
        (tuple(range(-12, 13)),),  # область: целые от −12 до 12
        lambda x: 2 * x**3 - x**2 + 7 * x - 4,  # учитель: 2x³ − x² + 7x − 4
        family="poly-unseen",  # семейство «полином не встречался»
    )
    pev = OptimalMathProcessor().learn(poly_spec, seed=31337, train_neural=False)  # обучение полинома
    assert pev.action == "COMMIT" and "poly(" in pev.program, (pev.action, pev.program)  # изучен как полином

    # Открытие точного полинома двух переменных; область проверяется полностью.
    poly2 = TaskSpec(  # задача квадратичного полинома
        "POLY2_QUADRATIC",  # имя
        2,  # два аргумента
        (tuple(range(-4, 5)), tuple(range(-3, 4))),  # области: x от −4 до 4, y от −3 до 3
        lambda x, y: 3 * x * x - 2 * x * y + y * y + 4 * x - 5 * y + 7,  # учитель: 3x² − 2xy + y² + 4x − 5y + 7
        family="poly2-unseen",  # семейство «не встречался»
    )
    p2ev = OptimalMathProcessor().learn(poly2, seed=31338, train_neural=False)  # обучение
    assert p2ev.action == "COMMIT" and "poly_nd(" in p2ev.program, (  # изучен как многомерный полином
        p2ev.action,  # сообщение ошибки: действие
        p2ev.program,  # сообщение ошибки: текст программы
    )
    p2model = OptimalMathProcessor()  # отдельная модель
    p2model.learn(poly2, seed=31338, train_neural=False)  # обучение той же задачи
    assert p2model.verify_all([poly2])["accuracy"] == 1.0  # точность на полной области равна 1

    # Открытие точного полинома трёх переменных с кубическим смешанным членом.
    poly3 = TaskSpec(  # задача
        "POLY3_CUBIC",  # имя
        3,  # три аргумента
        (tuple(range(-3, 4)), tuple(range(-3, 4)), tuple(range(-3, 4))),  # каждая область: целые от −3 до 3
        lambda x, y, z: 2 * x**3 - 3 * x * y * z + 4 * y * y - 5 * z + 11,  # учитель: 2x³ − 3xyz + 4y² − 5z + 11
        family="poly3-unseen",  # семейство «не встречался»
    )
    p3model = OptimalMathProcessor()  # отдельная модель
    p3ev = p3model.learn(poly3, seed=31339, train_neural=False)  # обучение
    assert p3ev.action == "COMMIT" and "poly_nd(" in p3ev.program, (  # изучен как многомерный полином
        p3ev.action,  # сообщение ошибки: действие
        p3ev.program,  # сообщение ошибки: текст программы
    )
    assert p3model.verify_all([poly3])["accuracy"] == 1.0  # точность на полной области равна 1

    assert model.predict("ADD", 7, -3)["value"] == 4  # сложение 7 + (−3) = 4
    assert model.experts["ADD"].code_length == _program_description_bits(  # длина кода сложения равна длине его описания
        model.experts["ADD"].program  # программа сложения
    )
    # Поиск теорем: только исчерпывающая проверка, неподтверждённых утверждений нет
    ths = TheoremMiner().mine(model)  # найти теоремы
    assert ths, "theorem miner found nothing"  # найдена хотя бы одна
    names = [t.name for t in model.theorems]  # имена всех теорем
    assert all(t.verified for t in model.theorems)  # все теоремы подтверждены
    assert any("commutativity(MIN)" in n for n in names), names  # есть коммутативность минимума
    assert any("idempotence(ABS)" in n for n in names), names  # есть идемпотентность модуля
    print(  # напечатать итог самопроверки
        f"SELF_TEST PASS: {len(suite)} experts, accuracy 1.0, OOD abstain, poison hold, "  # первая часть сообщения
        f"exact poly commit '{pev.program}', {len(model.theorems)} theorems"  # вторая часть: программа полинома и число теорем
    )


def main() -> None:  # точка входа командной строки
    """Разобрать аргументы командной строки и запустить самопроверку, рост или эксперимент."""
    p = argparse.ArgumentParser(  # разборщик аргументов
        description="HAGI: точная многомерная аппроксимация на конечных областях — исследовательский прототип"  # описание программы
    )
    p.add_argument("--self-test", action="store_true")  # флаг: запустить самопроверку
    p.add_argument("--experiment", action="store_true")  # флаг: запустить эксперимент по зернам
    p.add_argument(  # флаг роста библиотеки
        "--grow",  # имя флага
        action="store_true",  # истина, если флаг задан
        help="bounded compositional growth loop (10-minute default cap)",  # подсказка
    )
    p.add_argument(  # аргумент устройства
        "--device", default="cpu", help="cpu or cuda:0 (GPU batch prefilter)"  # устройство по умолчанию cpu
    )
    p.add_argument(  # аргумент предела времени
        "--max-minutes",  # имя
        type=float,  # вещественное число
        default=10.0,  # по умолчанию 10 минут
        help="wall-clock cap for --grow (default: 10 minutes)",  # подсказка
    )
    p.add_argument("--max-rounds", type=int, default=None)  # предел числа раундов роста (нет предела)
    p.add_argument("--snapshot-every", type=int, default=10)  # сохранять состояние каждые 10 раундов
    p.add_argument(  # флаг отключения нейронной части
        "--no-neural",  # имя флага
        action="store_true",  # истина, если задан
        help="disable neural fallback training (fastest pure-symbolic)",  # подсказка
    )
    p.add_argument(  # аргумент бюджета поиска
        "--search-budget",  # имя
        type=int,  # целое
        default=2_500,  # по умолчанию 2500
        help="per-task candidate budget during --grow; pop budget is twice this",  # подсказка
    )
    p.add_argument(  # аргумент числа процессов
        "--workers", type=int, default=0, help="parallel seed processes (0=auto)"  # 0 — автоматически
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])  # зерна: одно или несколько целых, по умолчанию 0..7
    p.add_argument("--out-dir", default=".omc/hagi_v7_results")  # каталог результатов
    args = p.parse_args()  # разобрать аргументы командной строки
    if args.self_test:  # задана самопроверка
        self_test_optimal()  # выполнить самопроверку
    elif args.grow:  # задан рост библиотеки
        suite = make_task_suite()  # набор задач
        model = OptimalMathProcessor(device=args.device)  # модель на заданном устройстве
        deadline = (  # предельный момент времени
            time.monotonic() + max(0.0, args.max_minutes * 60)  # now + max(0, минуты·60)
            if args.max_minutes is not None  # если предел задан
            else None  # иначе без предела
        )
        if args.no_neural:  # нейронная часть отключена
            os.environ["HAGI_NO_NEURAL"] = "1"  # переменная окружения отключает её
        report = grow_forever(  # итог цикла роста
            model,  # модель
            suite,  # набор задач
            device=args.device,  # устройство
            max_rounds=args.max_rounds,  # предел раундов
            snapshot_every=args.snapshot_every,  # период сохранения
            out_dir=args.out_dir,  # каталог результатов
            budget_cap=max(1, args.search_budget),  # бюджет поиска не меньше 1
            deadline=deadline,  # предельное время
        )
        print(  # напечатать краткий итог
            json.dumps(  # сериализовать в JSON
                {  # отобранные поля итога
                    k: report[k]  # значение поля k
                    for k in ("device", "stopped", "total_experts", "total_elapsed")  # устройство, признак остановки, число экспертов, время
                },
                indent=2,  # с отступом 2
            )
        )
    elif args.experiment:  # задан эксперимент
        result = run_optimal_experiment(  # провести эксперимент
            args.seeds, args.out_dir, workers=args.workers, device=args.device  # зерна, каталог, процессы, устройство
        )
        print(json.dumps(result["aggregate"], indent=2))  # напечатать агрегаты
    else:  # режим не задан
        p.print_help()  # напечатать справку


# ==============================================================================
# ЧАСТЬ III — единая система единая система
# ==============================================================================

"""Единая саморазвивающаяся нейросимвольная система с энергетическим решателем:
оптимальная структура, минимальный объём и минимум вычислений.
"""


import sys as _sys  # псевдоним модуля sys
from hagi.base.config import STATE_FORMAT  # единая метка формата состояния и перечень прежних меток


__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'run_optimal_experiment', 'self_test_optimal', 'main', # имя интерфейса
]  # конец перечня
