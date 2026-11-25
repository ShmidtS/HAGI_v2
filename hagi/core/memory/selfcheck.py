"""Самопроверка растущего математического процессора и командная оболочка.

self_test: на учебном наборе задач каждый эксперт обучается без нейросети
(seed = 10000 + номер задачи) и обязан дать действие COMMIT; затем
verify_all проверяет всех экспертов, точность должна равняться 1.0.
Дополнительно проверяется: предсказание ADD на всём квадрате −8..8,
воздержание вне области, загрузка сохранённой модели, вычисление
безопасных выражений, отказ на попытке импорта, HOLD на испорченной метке
и на неподдерживаемом алгоритме (проверка простоты).

Командная строка: --self-test, --experiment, --query OP ARG…, --expr с --var
(значение с точкой или буквой e читается как дробное, иначе целое),
--mode exact|neural, --model — путь к сохранённой модели.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import argparse, json, math, os  # модули стандартной библиотеки: аргументы, синтаксическое дерево, копирование, куча, комбинаторы, JSON, математика, ОС, случайность
from pathlib import Path  # пути файловой системы
from typing import Sequence  # аннотации типов
from hagi.core.tasks.stored_tasks import (  # импорт описания задач
    TaskSpec,  # класс спецификации задачи
    make_task_suite  # сборка учебного набора
)  # конец импорта
from hagi.core.processor.experts import (  # импорт процессора-родителя
    GrowingMathProcessor  # растущий математический процессор
)  # конец импорта
from hagi.core.memory.experiment import (  # импорт эксперимента
    run_experiment  # запуск эксперимента автономного роста
)  # конец импорта

# ----------------------------------- Тесты --------------------------------------


def self_test() -> None:
    """Прогнать самопроверку; при любом нарушении возбуждается AssertionError.

        1) каждый эксперт учебного набора обучается (train_neural = False)
           и даёт COMMIT; 2) verify_all: приняты все, точность 1.0;
           3) ADD верен на всём квадрате −8..8, вне области — ABSTAIN;
        4) выборочные ответы булевых функций; 5) сохранение и загрузка
           модели, выражения с gcd, сравнением, sqrt, sin; 6) попытка
           импорта модуля даёт ERROR; 7) испорченная метка и неизвестный
           алгоритм дают HOLD."""
    suite = make_task_suite()  # учебный набор задач
    model = GrowingMathProcessor()  # новый процессор
    for i, spec in enumerate(suite):  # по задачам с их номерами
        ev = model.learn(spec, seed=10_000 + i, train_neural=False)  # обучить без нейросети, seed = 10000 + i
        assert ev.action == "COMMIT", (spec.name, ev)  # эксперт обязан быть принят
    check = model.verify_all(suite)  # проверить всех экспертов на полных областях
    assert check["committed"] == len(suite)  # приняты все задачи набора
    assert check["accuracy"] == 1.0  # точность равна 1
    assert all(  # для всех пар x, y из −8..8
        model.predict("ADD", x, y)["value"] == x + y  # предсказание ADD равно x + y
        for x in range(-8, 9)  # x из −8..8
        for y in range(-8, 9)  # y из −8..8
    )  # конец all
    assert model.predict("ADD", 100, 3)["status"] == "ABSTAIN"  # аргумент 100 вне области — воздержание
    assert model.predict("BOOL_XOR", 1, 0)["value"] == 1  # исключающее ИЛИ: XOR(1, 0) = 1
    assert model.predict("BOOL_NAND", 1, 1)["value"] == 0  # И-НЕ: NAND(1, 1) = 0
    with_path = Path("/tmp/hagi_growing_math_selftest.json")  # путь файла сохранения
    model.save(with_path)  # сохранить модель
    loaded = GrowingMathProcessor.load(with_path)  # загрузить модель обратно
    assert loaded.predict("FMA", 2, 3, 4)["value"] == 10  # умножение со сложением: 2·3 + 4 = 10
    assert loaded.predict("CLAMP", 99, -2, 2)["status"] == "ABSTAIN"  # значение 99 вне области — воздержание
    assert loaded.evaluate_expression("x*y + gcd(x,y)", x=7, y=3)["value"] == 22  # произведение и наибольший общий делитель: 7·3 + gcd(7, 3) = 21 + 1 = 22
    assert (  # проверка логического выражения
        loaded.evaluate_expression("(x < y) and (y % 2 == 1)", x=2, y=3)["value"] == 1  # (2 < 3) и (3 нечётно) → 1
    )  # конец assert
    trig = loaded.evaluate_expression("sqrt(x*x) + sin(pi/2)", x=3)  # корень и синус: √(3²) + sin(π/2) = 4
    assert trig["status"] == "OK" and abs(trig["value"] - 4.0) < 1e-10  # статус OK и |значение − 4| < 10⁻¹⁰
    assert (  # попытка импорта через выражение
        loaded.evaluate_expression("__import__('os').system('false')", x=1)["status"]  # вычислить опасное выражение
        == "ERROR"  # должен быть отказ с ERROR
    )  # конец assert
    # Испорченные данные и неподдерживаемый алгоритм с большей областью должны быть отклонены.
    poisoned = GrowingMathProcessor().learn(  # обучение на испорченной метке
        next(s for s in suite if s.name == "ADD"),  # задача ADD
        seed=77,  # зерно генератора 77
        poison_one_label=True,  # испортить одну метку
        train_neural=False,  # без нейросети
    )  # конец learn
    assert poisoned.action == "HOLD"  # испорченные данные дают HOLD
    unknown = TaskSpec(  # задача, для которой нет готовой программы
        "IS_PRIME",  # имя операции
        1,  # один аргумент
        (tuple(range(-8, 9)),),  # аргумент из −8..8
        lambda x: int(x >= 2 and all(x % d for d in range(2, int(math.sqrt(x)) + 1))),  # простое: x ≥ 2 и ни один d из 2..⌊√x⌋ не делит x
    )  # конец TaskSpec
    assert (  # проверка результата обучения
        GrowingMathProcessor().learn(unknown, seed=78, train_neural=False).action  # обучение IS_PRIME с зерном 78
        == "HOLD"  # должно дать HOLD
    )  # конец assert
    print(  # напечатать строку успешного завершения
        f"SELF_TEST PASS: {len(suite)} operation experts, exhaustive finite-support checks, serialization, OOD abstain, poison hold"  # сообщение о прохождении с числом экспертов
    )  # конец print


def query_model(  # параметры: путь модели, имя операции, аргументы, режим
    model_path: str, name: str, args: Sequence[int], mode: str = "exact"  # путь к модели, имя эксперта, аргументы, режим (по умолчанию точный)
) -> None:  # возврата нет — результат печатается
    """Загрузить модель из model_path и напечатать ответ эксперта name
    на аргументах args (каждый приведён к int) в режиме mode."""
    model = GrowingMathProcessor.load(model_path)  # загрузить сохранённую модель
    print(json.dumps(model.predict(name, *map(int, args), mode=mode), indent=2))  # предсказание в JSON с отступом 2


def main() -> None:
    """Разобрать аргументы командной строки и выполнить выбранное действие.

        Приоритет: --self-test, затем --query, затем --expr; иначе —
        эксперимент на списке seed с выводом агрегатов и путей файлов."""
    p = argparse.ArgumentParser(description="HAGI Growing Math Processor")  # построитель аргументов
    p.add_argument("--self-test", action="store_true")  # флаг запуска самопроверки
    p.add_argument(  # флаг эксперимента
        "--experiment", action="store_true", help="run autonomous growth experiment"  # запуск эксперимента автономного роста
    )  # конец add_argument
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])  # список целых seed, по умолчанию 0, 1, 2
    p.add_argument("--out-dir", default="hagi_growing_math_results")  # каталог результатов
    p.add_argument(  # запрос к модели
        "--query", nargs="+", metavar=("OP", "ARG"), help="query saved model: OP ARG..."  # имя операции и аргументы
    )  # конец add_argument
    p.add_argument(  # вычисление выражения
        "--expr",  # текст выражения
        help="evaluate a safe arithmetic/logic expression, e.g. 'x*y + gcd(x,y)'",  # подсказка с примером выражения
    )  # конец add_argument
    p.add_argument(  # присваивание переменной выражения
        "--var",  # можно указать несколько раз
        action="append",  # каждый повтор добавляет значение в список
        default=[],  # по умолчанию список пуст
        help="expression variable assignment, e.g. x=7 or x=0.5",  # подсказка: имя=значение
    )  # конец add_argument
    p.add_argument("--mode", choices=["exact", "neural"], default="exact")  # режим: точный или нейросетевой
    p.add_argument(  # путь к модели
        "--model", default="hagi_growing_math_results/hagi_growing_math_model.json"  # файл модели по умолчанию
    )  # конец add_argument
    args = p.parse_args()  # разобрать аргументы командной строки
    if args.self_test:  # запрошена самопроверка
        self_test()  # выполнить её
    elif args.query:  # запрошен запрос к модели
        if len(args.query) < 2:  # нет имени операции или ни одного аргумента
            p.error("--query needs an operation name and at least one integer argument")  # сообщение об ошибке и выход
        query_model(args.model, args.query[0], args.query[1:], args.mode)  # запрос: имя — первый элемент, остальные — аргументы
    elif args.expr:  # запрошено вычисление выражения
        vals = {}  # значения переменных: имя -> число
        for item in args.var:  # по всем присваиваниям name=значение
            if "=" not in item:  # нет знака равенства
                p.error(f"invalid --var {item!r}; expected name=integer")  # сообщение об ошибке и выход
            k, v = item.split("=", 1)  # разделить по первому «=» на имя и значение
            raw = v.strip()  # значение без крайних пробелов
            vals[k.strip()] = (  # сохранить по имени без пробелов
                float(raw) if any(ch in raw.lower() for ch in (".", "e")) else int(raw)  # дробное, если есть «.» или «e», иначе целое
            )  # конец присваивания
        model = GrowingMathProcessor.load(args.model)  # загрузить модель
        print(json.dumps(model.evaluate_expression(args.expr, **vals), indent=2))  # результат выражения в JSON с отступом 2
    else:  # ни один режим не задан — эксперимент
        result = run_experiment(args.seeds, args.out_dir)  # эксперимент на списке seed
        print(json.dumps(result["aggregate"], indent=2))  # агрегаты в JSON с отступом 2
        print(f"Report: {Path(args.out_dir) / 'hagi_growing_math_report.md'}")  # путь текстового отчёта
        print(f"Model:  {Path(args.out_dir) / 'hagi_growing_math_model.json'}")  # путь файла модели

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'self_test', 'query_model', 'main',  # имя интерфейса
]  # конец перечня
