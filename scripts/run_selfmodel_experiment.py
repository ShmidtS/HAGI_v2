#!/usr/bin/env python3  # интерпретатор Python 3
"""Конечный эксперимент с моделью самого себя: извлечь символьную модель собственных успехов и сравнить политики бюджета на независимых зернах.

Этап 1 (след и вывод): процессор TracedProcessorV9 учит задачи набора и до 10
предложенных задач, каждая попытка попадает в след с признаками, исходом
успеха и стоимостью. derive_self_model ищет точную программу признаки -> успех,
impossibility_certificate ищет два вызова с одинаковыми признаками и разными
исходами. Этап 2 (проверка): run_experiment сравнивает 4 политики на 20 зернах
от 30. Результат записывается в experiments/selfmodel_result.json; код выхода 0,
если ошибок не было, иначе 1.
"""  # документация модуля
from __future__ import annotations  # отложенное вычисление аннотаций типов

import json  # запись результата в JSON
import os  # переменные окружения
import platform  # сведения о системе
import sys  # версия интерпретатора
import traceback  # текст исключения
from datetime import datetime, timezone  # время по Гринвичу
from pathlib import Path  # пути

ROOT = Path(__file__).resolve().parents[1]  # корень проекта
sys.path.insert(0, str(ROOT))  # добавить корень проекта в путь импорта

def main() -> int:
    """Выполнить эксперимент и записать словарь-результат; возвращает код выхода 0 или 1."""
    out = ROOT / "experiments" / "selfmodel_result.json"  # путь файла результата
    out.parent.mkdir(parents=True, exist_ok=True)  # создать каталог experiments, если его нет
    report = {  # словарь-результат, сначала общие сведения
        "experiment": "HAGI finite self-model / derive-and-optimize",  # название эксперимента
        "repository": "ShmidtS/HAGI_v2",  # имя хранилища
        "branch": os.getenv("GITHUB_REF_NAME", "local"),  # имя ветки из окружения, иначе "local"
        "commit": os.getenv("GITHUB_SHA", "unknown"),  # хеш ревизии из окружения, иначе "unknown"
        "started_utc": datetime.now(timezone.utc).isoformat(),  # время начала по Гринвичу в формате ISO
        "python": sys.version,  # версия интерпретатора
        "platform": platform.platform(),  # описание операционной системы
    }
    try:  # любой сбой записывается в результат вместо аварийного выхода
        import numpy as np  # численные массивы
        import hagi  # набор задач и синтез задач процессора
        from hagi.selfmodel import (  # процессор со следом и вывод модели самого себя
            FEATURE_KEYS, TracedProcessorV9, describe_pipeline,  # имена признаков, процессор, описание конвейера
            derive_self_model, impossibility_certificate,  # вывод символической модели и свидетель невозможности
        )

        # отдельный проход со следом: извлечь саму символическую модель.
        tracer = TracedProcessorV9(node_budget=2000)  # процессор с бюджетом 2000 узлов и записью следа
        rng = np.random.default_rng(20260101)  # генератор случайных чисел с фиксированным зерном
        seen = set()  # имена уже предложенных задач
        for spec in hagi.make_task_suite():  # по всем задачам набора
            tracer.learn(spec, seed=10_000, train_neural=False)  # выучить задачу без нейросети; запись добавится в след
        for k in range(10):  # до 10 дополнительных задач
            proposal = hagi.propose_noncircular_task(tracer.experts, rng, 0, seen)  # предложить новую задачу (задача, родители) или None
            if proposal is None:  # предложить нечего
                continue  # перейти к следующей попытке
            spec, _parents = proposal  # разделить пару на задачу и список родителей
            if spec.name not in tracer.experts:  # такой задачи ещё нет в библиотеке
                tracer.learn(spec, seed=500_000 + k, train_neural=False)  # выучить предложенную задачу; запись добавится в след
        symbolic = derive_self_model(tracer.trace)  # символическая модель: программы успеха и стоимости по следу
        report["self_model"] = {  # раздел результата о модели самого себя
            "trace_examples": len(tracer.trace),  # число записей следа
            "feature_schema": list(FEATURE_KEYS),  # список имён признаков
            "pipeline": describe_pipeline(tracer),  # описание конвейера процессора
            "symbolic_predictors": {  # тексты найденных программ
                k: v for k, v in symbolic.items() if not k.startswith("_")  # все ключи модели, кроме служебных (начинаются с "_")
            },
            "impossibility_certificate": impossibility_certificate(tracer.trace),  # пара вызовов с равными признаками и разными исходами или None
        }

        # затем проверить политики распределения бюджета на независимых зернах.
        from hagi.selfmodel import run_experiment  # сравнение политик бюджета на нескольких зернах
        selected_policies = ["baseline","online_exact_2holds_quarter","online_exact_1hold_quarter","online_exact_2holds_eighth"]  # сравниваемые политики: эталон и три более смелых
        result = run_experiment(  # сравнение политик
            seeds=20, policy_names=selected_policies, seed_start=30  # 20 зерен с 30 по 49, заданные политики
        )
        report["validation_protocol"] = {  # описание протокола проверки
            "seed_start": 30,  # первое зерно
            "seed_count": 20,  # число зерен
            "policies": selected_policies,  # список политик
            "purpose": "test more aggressive exact-profile thresholds against the validated policy on unseen seeds",  # цель проверки
        }
        report["status"] = "COMPLETED"  # статус — выполнено
        report["result"] = result  # итог сравнения политик
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()  # время окончания по Гринвичу
        report["exit_code"] = 0  # код выхода 0 — успех
    except BaseException as exc:  # перехватывается любое исключение, включая SystemExit и KeyboardInterrupt
        report["status"] = "ERROR"  # статус — ошибка
        report["error_type"] = type(exc).__name__  # имя класса исключения
        report["error"] = str(exc)  # текст исключения
        report["traceback"] = traceback.format_exc()  # отформатированная трассировка
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()  # время окончания по Гринвичу
        report["exit_code"] = 1  # код выхода 1 — сбой
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")  # записать результат в JSON с отступом 2 и переводом строки в конце
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))  # напечатать тот же JSON
    return int(report["exit_code"])  # вернуть код выхода из результата

if __name__ == "__main__":  # запуск только как сценарий
    raise SystemExit(main())  # завершить процесс с кодом выхода main()
