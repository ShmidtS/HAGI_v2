#!/usr/bin/env python3
"""Цикл саморазвития HAGI: многократный вывод модели самого себя и её принятие.

Надстройка над hagi.selfmodel (однократный цикл СЛЕД -> ВЫВОД -> ОПТИМИЗАЦИЯ -> ГЕЙТ):

  1. Точная модель самого себя в виде решающего списка: таблица с ключом из всех
     признаков следа; неизвестный ключ -> осторожный базовый бюджет. На следе она
     точна по построению; на отложенной части следа проверяется опасное направление
     (уверенно неверное предсказание HOLD блокирует принятие).
  2. Многораундовое движение: каждый принятый раунд дополняет библиотеку задач новыми
     некруговыми предложениями, заново выводит модель самого себя и снова проходит
     гейт. Остановка — на первом гейте HOLD или когда запас снижения стоимости
     меньше нормы MIN_RELIEF: нет принятого действия с гарантированной пользой,
     значит ресурс исчерпан.

Быстродействие:
  - Кэш повторного прогона: задачи-префикса, чей назначенный бюджет не изменился,
    используют записанный исход вместо повторного обучения. Полный повторный прогон
    раунда стоит O(новых задач), весь цикл линеен.
  - Пакеты заданий: задачи-предложения учатся на одном общем процессоре строго по
    порядку (без подпроцессов), параллельность даёт только BLAS/OMP внутри numpy и
    torch. Все слияния идут в порядке потока, поэтому оба прогона (рост и
    оптимизированный повтор) детерминированы и сравнимы.

Запуск: python -m hagi.loop [--max-rounds N] [--workers N] [--json]
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import os  # доступ к переменным окружения и числу процессоров
# Число потоков BLAS/OMP: отдельных процессов-исполнителей нет, поэтому все ядра
# машины отдаются численным библиотекам. Ставится только если пользователь не задал.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # по именам трёх переменных числа потоков
    os.environ.setdefault(_v, str(os.cpu_count() or 4))  # задать переменную, если её ещё нет: число процессоров (при 0 — 4)

import argparse  # разбор аргументов командной строки
import json  # запись отчёта в JSON
import time  # измерение времени
from pathlib import Path  # путь к файлу отчёта
from typing import Any, Optional  # типы для аннотаций

import numpy as np  # численные массивы и генератор случайных чисел

import hagi  # ядро системы: набор задач, предложения задач, процессоры
from hagi.selfmodel import (  # из модуля модели самого себя:
    FEATURE_KEYS,  # имена семи признаков задачи в следе
    SelfOptimizingProcessorV9,  # процессор с бюджетом от модели самого себя
    TracedProcessorV9,  # процессор, ведущий след своего обучения
    describe_pipeline,  # описание конвейера по журналу роста
    impossibility_certificate,  # свидетельство невозможности точного предсказателя успеха
)

FULL_KEY = FEATURE_KEYS + ("seed",)  # полный ключ задачи: семь признаков плюс зерно seed (внутри пакета библиотека неизменна, зерно различает задачи)
MIN_RELIEF = 0.02  # минимальная доля снижения стоимости на раунд: новое ≤ старое·(1 − 0.02)


# ------------------------------------------------------------------ model ----

class DecisionListSelfModel:
    """Точный по построению предсказатель успеха и стоимости по признакам следа.

    Решающий список: ключ — полный кортеж (семь признаков, зерно). Успех 1 или 0
    берётся только для ключей, у которых в следе один и тот же исход во всех записях;
    стоимость — максимум по ним. Ключи с разными исходами попадают в список
    неоднозначных и предсказания не дают; обобщение по грубому ключу не используется:
    однажды уверенно неверный HOLD стоил потерянного эксперта."""

    def __init__(self, trace: list[dict[str, Any]]):  # построить модель по следу trace
        succ: dict[tuple, set[int]] = {}  # ключ -> множество значений успеха в следе
        cost: dict[tuple, list[int]] = {}  # ключ -> список стоимостей в следе
        for t in trace:  # по записям следа
            k = tuple(t[f] for f in FULL_KEY)  # ключ k = (признаки задачи, зерно)
            succ.setdefault(k, set()).add(t["success"])  # добавить исход успеха в множество ключа k
            cost.setdefault(k, []).append(t["cost"])  # добавить стоимость в список ключа k
        # Согласованность считается ДО каких-либо изменений множеств: раньше множество
        # опустошалось обходом, проверка «ровно один исход» всегда была ложной, и
        # модель стоимости не работала.
        consistent = {k for k, v in succ.items() if len(v) == 1}  # согласованные ключи: |succ[k]| = 1 (исход всегда один)
        self._succ = {k: next(iter(succ[k])) for k in consistent}  # предсказание успеха: единственный исход согласованного ключа
        self._ambig = sorted(k for k, v in succ.items() if len(v) > 1)  # отсортированные неоднозначные ключи: |succ[k]| > 1
        self._cost = {k: max(vs) for k, vs in cost.items()  # стоимость согласованного ключа = максимум по его записям
                      if k in consistent}  # берутся только согласованные ключи

    def predict_success(self, feats: dict[str, Any]) -> Optional[int]:
        """Предсказать успех задачи по признакам feats: 0 или 1, либо None, если ключ неизвестен."""
        return self._succ.get(tuple(feats[f] for f in FULL_KEY))  # lookup по полному ключу (признаки, зерно)

    def predict_cost(self, feats: dict[str, Any]) -> Optional[int]:
        """Предсказать стоимость задачи в кандидатах по признакам feats; None, если ключ неизвестен."""
        return self._cost.get(tuple(feats[f] for f in FULL_KEY))  # lookup по полному ключу (признаки, зерно)

    @property  # доступ как к атрибуту
    def ambiguous_keys(self) -> list[tuple]:  # список ключей с противоречивыми исходами
        """Ключи, у которых в следе были и успех, и неуспех."""
        return self._ambig  # отсортированный список неоднозначных ключей

    def describe(self) -> dict[str, Any]:
        """Краткое описание модели: вид, число правил и число неоднозначных ключей."""
        return {  # словарь описания
            "kind": "exact-decision-list",  # вид: точный решающий список
            "full_key_rules": len(self._succ),  # число правил = числу согласованных ключей
            "ambiguous_keys": len(self._ambig),  # число неоднозначных ключей
        }  # конец словаря


def holdout_validate(trace: list[dict[str, Any]],
                     fraction: float = 0.30) -> dict[str, Any]:  # доля отложенной части следа; возвращает словарь проверки
    """Проверка модели самого себя на отложенной части следа.

    Случайная доля fraction записей откладывается; модель строится по остальным.
    На отложенных записях ищется опасное направление: уверенное предсказание неуспеха
    (уменьшенный бюджет) для задачи, которая на деле заканчивается COMMIT, — такое
    предсказание блокирует принятие. Неизвестный ключ не считается ошибкой: он даёт
    базовый бюджет, а любой регресс от этого ловит гейт безрегрессии.
    Возвращает словарь: размер отложенной части, число уверенно неверных предсказаний,
    число безопасных неизвестных, покрытие и признак exact (нет неверных и покрытие > 0)."""
    rng = np.random.default_rng(7)  # генератор с фиксированным зерном 7 — разбиение воспроизводимо
    idx = rng.permutation(len(trace))  # случайная перестановка номеров записей idx
    n_hold = max(1, int(len(trace) * fraction))  # размер отложенной части n_hold = max(1, int(len·fraction))
    hold_idx = set(idx[:n_hold].tolist())  # номера отложенных записей — первые n_hold элементов перестановки
    train = [t for i, t in enumerate(trace) if i not in hold_idx]  # обучающая часть: записи вне hold_idx
    hold = [trace[i] for i in sorted(hold_idx)]  # отложенная часть в порядке возрастания номеров
    m = DecisionListSelfModel(train)  # модель, построенная только по обучающей части
    confident_wrong = 0  # счётчик уверенно неверных предсказаний
    unknown_safe = 0  # счётчик неизвестных ключей у неуспешных задач
    for t in hold:  # по отложенным записям
        pred = m.predict_success(t)  # предсказание успеха: 0, 1 или None
        truth = t["success"]  # истинный исход записи: 1 или 0
        if pred is None:  # ключ модели неизвестен
            if truth == 0:  # задача в действительности не удалась
                unknown_safe += 1  # неизвестный ключ при неуспехе: безопасно, счётчик + 1
        elif pred != truth:  # предсказание есть и отлично от истины
            confident_wrong += 1  # уверенно неверное предсказание: счётчик + 1
    # Модель без покрытия (None для каждой отложенной записи) даёт 0 неверных
    # предсказаний и формально проходит гейт, ничего не предсказывая. Для вердикта
    # «точна» покрытие обязательно.
    covered = sum(1 for t in hold if m.predict_success(t) is not None)  # число отложенных записей с известным ключом
    coverage = covered / len(hold) if hold else 0.0  # покрытие = covered / |hold| (0 при пустой отложенной части)
    return {"holdout_size": len(hold), "confident_wrong": confident_wrong,  # вернуть словарь: размер отложенной части и число уверенно неверных
            "unknown_safe": unknown_safe, "coverage": round(coverage, 4),  # число безопасных неизвестных и покрытие с 4 знаками
            "exact": confident_wrong == 0 and coverage > 0.0}  # exact = (неверных 0) и (покрытие > 0)


# ---------------------------------------------------------- policy processor --

class SelfModelPolicyProcessor(SelfOptimizingProcessorV9):
    """Процессор, чья политика бюджета — решающий список DecisionListSelfModel.

    dl — модель; base_budget и floor_budget — верхний и нижний бюджет; *a, **kw —
    прочие аргументы базового класса."""

    def __init__(self, dl: DecisionListSelfModel, base_budget: int,  # решающий список, базовый бюджет
                 floor_budget: int, *a, **kw):  # нижний бюджет и прочие аргументы базового класса; создать процессор с политикой dl
        super().__init__({"_succ_expr": None}, base_budget, floor_budget, *a, **kw)  # конструктор базового класса с пустой моделью (её программа успеха не используется)
        self.dl = dl  # сохранить решающий список

    def learn(self, spec, seed=0, **kw):
        """Выучить задачу spec с бюджетом от решающего списка.

        Признаки задачи (семь, как в следе) вычисляются по текущему процессору и
        вместе с зерном seed образуют ключ. Предсказание 0 -> floor_budget, иначе
        (1 или неизвестный ключ) -> base_budget. Остальные аргументы kw передаются
        в learn базового класса с пропуском политики по программе; возвращает
        событие обучения."""
        from hagi.selfmodel import _features  # функция вектора признаков задачи (локальный импорт)
        feats: dict[str, Any] = dict(zip(  # словарь признаков: имя признака -> значение
            FEATURE_KEYS,  # имена семи признаков
            _features(spec, self, seed,  # вектор признаков задачи spec для этого процессора с зерном seed
                      float(kw.get("train_fraction", 0.70)))))  # доля обучения из kw (по умолчанию 0.70)
        feats["seed"] = int(seed)  # добавить в признаки зерно как целое
        pred = self.dl.predict_success(feats)  # предсказание успеха: 0, 1 или None
        if pred is None:  # ключ модели неизвестен
            pred = 1  # считать успехом — оставить базовый бюджет
        self.node_budget = self.floor_budget if pred == 0 else self.base_budget  # бюджет узлов = floor_budget при предсказанном неуспехе, иначе base_budget
        if pred == 0:  # бюджет уменьшен политикой
            self.policy_overrides += 1  # счётчик уменьшений: policy_overrides + 1
        return super(SelfOptimizingProcessorV9, self).learn(spec, seed=seed, **kw)  # learn класса-прародителя (минуя политику по программе): обычное обучение с поставленным бюджетом


# ------------------------------------------------------- parallel machinery ---

def _state_probe(proc) -> dict[str, Any]:
    """Снимок счётчиков и состояния процессора proc до или после одного обучения."""
    return {  # словарь снимка
        "cands": proc.candidates_generated, "evals": proc.evals,  # всего сгенерировано кандидатов и сделано вычислений
        "commits": proc.total_commits, "holds": proc.total_holds,  # всего принято задач и удержаний
        "version": proc.version,  # версия библиотеки
        "unsolved": proc.unsolved_objective,  # суммарная нерешённая цель
        "compute": proc.compute_budget, "risk": proc.risk_rem,  # израсходованный бюджет вычислений и остаток риска
        "n_experts": len(proc.experts), "n_log": len(proc.growth_log),  # число экспертов и длина журнала роста
        "n_cert": len(getattr(proc, "cert_log", [])), "n_trace": len(getattr(proc, "trace", [])),  # длина журнала сертификатов и длина следа
        "ram": proc.ram.snapshot(),  # снимок оперативной памяти (матрица ячейки -> вектор)
    }  # конец словаря


def _learn_worker_local(proc, spec, seed, budget, train_neural):
    """Выучить одну задачу на самом процессоре proc и вернуть отличие «после минус до».

    Без сериализации: та же логика, что у исполнителя в отдельном процессе, но
    всё происходит внутри текущего процесса. Задаются нейронное устройство cpu и
    бюджет узлов, снимаются два снимка состояния, вызывается proc.learn. Разница
    содержит: событие обучения, новые записи следа, журнала роста и сертификатов,
    новых экспертов, приросты счётчиков, изменённые ячейки памяти, снимки ядра
    и маршрутизатора и запись аудита (имя задачи, alpha)."""
    os.environ["HAGI_NEURAL_DEVICE"] = "cpu"  # нейронная часть выполняется на процессоре
    proc.node_budget = int(budget)  # бюджет узлов на задачу = int(budget)
    alpha = min(0.0005, proc.ledger.remaining() * 0.5)  # списываемая доля бюджета аудита alpha = min(0.0005, 0.5·остаток)
    pre = _state_probe(proc)  # снимок состояния до обучения
    ev = proc.learn(spec, seed=int(seed), train_neural=train_neural)  # выучить задачу с зерном seed; ev — событие обучения
    post = _state_probe(proc)  # снимок состояния после обучения
    names = list(proc.experts.keys())  # имена всех экспертов в порядке вставки
    new_names = names[pre["n_experts"]:]  # имена экспертов, добавленных этим обучением (хвост списка)
    diff = {  # словарь отличия
        "ev": ev,  # событие обучения
        "trace_row": getattr(proc, "trace", [])[pre["n_trace"]:] or None,  # новые записи следа; None, если их нет
        "growth_log": proc.growth_log[pre["n_log"]:],  # новые записи журнала роста
        "certs": proc.cert_log[pre["n_cert"]:],  # новые сертификаты
        "new_experts": {n: proc.experts[n] for n in new_names},  # новые эксперты: имя -> объект
        "deltas": {k: post[k] - pre[k] for k in  # приросты счётчиков: после − до
                   ("cands", "evals", "commits", "holds", "version",  # по кандидатам, вычислениям, принятиям, удержаниям, версии
                    "unsolved", "compute", "risk")},  # нерешённой цели, вычислениям, риску
        "ram_ops": _ram_diff(pre["ram"], post["ram"]),  # изменившиеся ячейки памяти
        "core": proc.core.snapshot(),  # снимок нейронного ядра
        "router": proc.router.snapshot(),  # снимок маршрутизатора
        "audit": {"task": spec.name, "alpha": alpha},  # запись аудита: имя задачи и списываемая доля
    }  # конец словаря
    return diff  # вернуть отличие




def _ram_diff(pre: np.ndarray, post: np.ndarray) -> list[tuple[int, Any]]:
    """Ячейки памяти, изменившиеся между снимками pre и post: список (номер ячейки, новый вектор)."""
    ops = []  # пустой список операций
    for slot in range(post.shape[0]):  # по номерам ячеек slot = 0..n−1
        if not np.array_equal(pre[slot], post[slot]):  # если вектор ячейки изменился
            ops.append((slot, post[slot].copy()))  # добавить (slot, копия нового вектора)
    return ops  # вернуть список изменённых ячеек


class ParallelLearner:
    """Пакетное обучение с сохранением порядка потока на одном процессоре.

    Все задачи пакета учатся на одном и том же процессоре по порядку; их отличия
    применяются прямо при обучении. Слияние через _merge нужно только для отличий,
    записанных кэшем на другом состоянии процессора; оно идёт в порядке потока.
    При train_neural=True каждое обучение обучает и нейронное ядро (адаптеры
    FactorizedSharedCore): состояния ядра и маршрутизатора входят в отличие и
    восстанавливаются в порядке потока. Точные исходы это не меняет (гейт приёма
    смотрит только точные программы), поэтому детерминизм сохраняется."""

    def __init__(self, proc, workers: int, train_neural: bool = False):
        """Создать обучатель для процессора proc; workers — заявленное число исполнителей,
        train_neural — обучать ли нейронное ядро."""
        self.proc = proc  # процессор, на котором идёт обучение
        self.workers = max(1, workers)  # число исполнителей, не меньше 1
        self.train_neural = bool(train_neural)  # признак обучения нейронного ядра как булево значение

    def batch(self, jobs: list[tuple[Any, int, int]]) -> list[dict[str, Any]]:
        """Выучить пакет задач и вернуть их отличия в исходном порядке.

        jobs — список (spec, seed, budget). Задачи выполняются последовательно на
        общем процессоре в самом этом процессе: ни потоков, ни процессов-исполнителей,
        ни сериализации данных. Параллельность возникает только внутри операций
        numpy и torch (BLAS/OMP). Задачи независимы по построению, поэтому порядок
        и результат детерминированы."""
        if not jobs:  # пакет пуст
            return []  # вернуть пустой список
        # Обучения идут в этом процессе строго по порядку на одном процессоре.
        # Отличия НЕ сливаются здесь: они уже применены общим процессором, вторичное
        # слияние удвоило бы счётчики, журналы и след (обновление словарей можно
        # повторить безвредно, остальное — нет).
        diffs = [_learn_worker_local(self.proc, spec, seed, budget,  # список отличий: по одной задаче
                                     self.train_neural)  # с общим признаком обучения нейронного ядра
                 for spec, seed, budget in jobs]  # по тройкам (задача, зерно, бюджет)
        # Слияние выполняет вызывающий код в порядке потока.
        return diffs  # вернуть отличия в порядке задач

    def _merge(self, d: dict[str, Any]) -> None:
        """Применить отличие d к процессору self.proc (эксперты, журналы, счётчики, память, ядро, аудит).

        Отличие, чей первый сертификат (задача и версия) уже есть в журнале
        сертификатов, считается уже применённым и пропускается — иначе версия,
        журнал сертификатов и нерешённая цель задваивались (одно принятие
        превращалось в два)."""
        p = self.proc  # процессор-приёмник
        # Защита от двойного применения: отличия, полученные на самом процессоре,
        # уже применены. Признак повтора — совпадение задачи и версии первого
        # сертификата отличия с записью в журнале.
        if d["certs"]:  # в отличии есть сертификаты
            first = d["certs"][0]  # первый сертификат отличия
            if any(c.version == first.version and c.task == first.task  # есть в журнале сертификат с той же версией и задачей
                   for c in p.cert_log):  # по всем сертификатам c журнала
                return  # уже применено — ничего не делать
        p.experts.update(d["new_experts"])  # добавить новых экспертов в библиотеку
        p.growth_log.extend(d["growth_log"])  # дописать журнал роста
        p.cert_log.extend(d["certs"])  # дописать журнал сертификатов
        dl = d["deltas"]  # приросты счётчиков
        p.candidates_generated += dl["cands"]  # сгенерировано кандидатов: + Δcands
        p.evals += dl["evals"]  # вычислений: + Δevals
        p.total_commits += dl["commits"]  # принятых задач: + Δcommits
        p.total_holds += dl["holds"]  # удержаний: + Δholds
        p.version += dl["version"]  # версия библиотеки: + Δversion
        p.unsolved_objective += dl["unsolved"]  # нерешённая цель: + Δunsolved
        p.compute_budget += dl["compute"]  # израсходованные вычисления: + Δcompute
        p.risk_rem += dl["risk"]  # остаток риска: + Δrisk
        for slot, vec in d["ram_ops"]:  # по парам (ячейка, вектор) изменённой памяти
            p.ram.store(slot, vec)  # записать вектор точно (gate_beta = 1: новая запись целиком заменяет старую)
        if d.get("core") is not None:  # в отличии есть снимок ядра
            p.core.restore(d["core"])  # восстановить нейронное ядро из снимка
        if d.get("router") is not None:  # в отличии есть снимок маршрутизатора
            p.router.restore(d["router"])  # восстановить маршрутизатор из снимка
        if d["trace_row"]:  # в отличии есть новые записи следа
            for row in d["trace_row"]:  # по записям row
                p.trace.append(row)  # дописать запись в след
        # Журнал аудита: повторить записанные списание, нижнюю границу и отказ по задаче
        ev = d["ev"]  # событие обучения
        task = ev["task"]  # имя задачи
        n_train = int(ev.get("train_examples", 1))  # число обучающих примеров (по умолчанию 1)
        n_val = int(ev.get("validation_examples", 1))  # число проверочных примеров (по умолчанию 1)
        p.ledger.allocate_stream(task, n_train + n_val)  # выделить задаче поток аудита на n_train + n_val примеров (существующий поток возвращается как есть)
        if d["audit"].get("alpha") is not None:  # в записи аудита есть доля списания
            p.ledger.spent[task] = (p.ledger.spent.get(task, 0.0)  # списано по задаче: прежнее значение (0, если нет)
                                    + float(d["audit"]["alpha"]))  # плюс alpha
            p.ledger.advance_stream(task, n_train + n_val)  # сдвинуть курсор потока: cursor = min(hi, cursor + n_train + n_val)
        # условие ниже: принято, ИЛИ принятие из-за проверенного покрытия эксперта, ИЛИ повторное использование
        if ev["action"] == "COMMIT" or ev.get("reason") == "verified-expert-coverage" \
                or ev["action"] == "REUSE":  # (продолжение условия: действие — REUSE)
            p.ledger.record_floor(task, 1.0)  # записать нижнюю границу качества 1.0 по задаче
        elif ev["action"] == "HOLD":  # иначе, если задача удержана
            p.ledger.record_reject(task)  # записать отказ по задаче


# ------------------------------------------------------------------ streams --

def _assign_budget(proc, spec, seed: int, train_fraction: float = 0.70) -> int:
    """Бюджет узлов, который политика процессора proc назначила бы задаче spec с зерном seed.

    Если у процессора есть решающий список dl — бюджет как в SelfModelPolicyProcessor
    (предсказанный неуспех -> floor_budget, иначе base_budget; счётчик уменьшений
    увеличивается). Иначе — текущий node_budget процессора."""
    if hasattr(proc, "dl"):  # у процессора есть атрибут dl (решающий список)
        from hagi.selfmodel import _features  # функция вектора признаков задачи (локальный импорт)
        feats = dict(zip(FEATURE_KEYS, _features(spec, proc, seed, train_fraction)))  # словарь признаков: имя -> значение вектора признаков
        feats["seed"] = int(seed)  # добавить зерно как целое
        pred = proc.dl.predict_success(feats)  # предсказание успеха: 0, 1 или None
        if pred == 0:  # предсказан неуспех
            proc.policy_overrides += 1  # счётчик уменьшений: policy_overrides + 1
            return proc.floor_budget  # вернуть нижний бюджет
        return proc.base_budget  # иначе вернуть базовый бюджет
    return proc.node_budget  # нет решающего списка — вернуть текущий бюджет узлов


class ReplayCache:
    """Кэш повторного прогона: позиция потока и бюджет -> записанное отличие.

    Позволяет не переучивать неизменившийся префикс повторного потока в каждом раунде."""

    def __init__(self) -> None:  # создать пустой кэш
        self.store: dict[tuple, dict[str, Any]] = {}  # хранилище: ключ (позиция, бюджет) -> отличие

    def key(self, pos: tuple, budget: int) -> Optional[dict[str, Any]]:
        """Вернуть записанное отличие для позиции pos и бюджета budget либо None, если его нет."""
        return self.store.get((pos, budget))  # lookup по ключу (pos, budget)

    def put(self, pos: tuple, budget: int, diff: dict[str, Any]) -> None:
        """Записать отличие diff для позиции pos и бюджета budget."""
        self.store[(pos, budget)] = diff  # сохранить по ключу (pos, budget)

def run_stream(proc, segments: list[tuple[int, int]],  # запустить детерминированный поток задач на процессоре proc
               suite: list | None = None, cache: ReplayCache | None = None,  # базовый набор задач и кэш повторного прогона (могут быть None)
               learner: ParallelLearner | None = None,  # обучатель пакетов (None — учить в этом процессе напрямую)
               batch_size: int = 8, train_neural: bool = False) -> None:  # размер пакета и признак обучения нейронного ядра
    """Детерминированный поток задач: базовый набор suite выполняется по одному, затем
    сегменты предложений — пакетами.

    segments — список (число предложений n_prop, базовое зерно seed_base). Кэш cache
    замыкает задачи, чьи позиция и бюджет уже встречались. Если у процессора есть
    списки stream_names и stream_actions, в них дописываются имена задач и действия
    (suite, имя) и (seg, имя, действие) в порядке потока."""
    names = getattr(proc, "stream_names", None)  # список имён задач потока, если процессор его ведёт
    actions = getattr(proc, "stream_actions", None)  # список действий потока, если процессор его ведёт
    for sp in (suite if suite is not None else hagi.make_task_suite()):  # по задачам базового набора (при None — полный учебный набор)
        if names is not None:  # имена записываются
            names.append(("suite", sp.name))  # дописать ("suite", имя задачи)
        ev = proc.learn(sp, seed=10_000, train_neural=train_neural)  # выучить задачу с зерном 10000; ev — событие обучения
        if actions is not None:  # действия записываются
            actions.append(("suite", sp.name, ev["action"]))  # дописать ("suite", имя задачи, действие)
    for seg_idx, (n_prop, seed_base) in enumerate(segments):  # по сегментам: номер seg_idx, число предложений, базовое зерно
        rng = np.random.default_rng(seed_base)  # генератор предложений с зерном seed_base: зерно задаёт тождество сегмента, а не позицию
        seen: set = set()  # множество уже предложенных подписей — от повторов
        k = 0  # счётчик предложенных задач в сегменте
        while k < n_prop:  # пока не набрано n_prop предложений
            # собрать один пакет предложений по текущему снимку библиотеки
            batch: list[tuple[Any, int]] = []  # пакет пар (задача, зерно)
            while k < n_prop and len(batch) < batch_size:  # пока есть лимит и пакет не полон
                prop = hagi.propose_noncircular_task(proc.experts, rng, 0, seen)  # предложить новую некруговую задачу по текущим экспертам; None — предложений нет
                if prop is None:  # подходящей задачи не нашлось
                    k = n_prop  # сегмент исчерпан: счётчик равен лимиту
                    break  # прервать набор пакета
                sp, _ = prop  # взять задачу (подпись не используется)
                if sp.name in proc.experts:  # эксперт с таким именем уже есть
                    continue  # пропустить — не учить заново
                batch.append((sp, seed_base + k))  # добавить в пакет пару (задача, зерно = seed_base + k)
                if names is not None:  # имена записываются
                    names.append(("seg", sp.name))  # дописать ("seg", имя задачи)
                k += 1  # предложенных: k + 1
            if not batch:  # пакет пуст
                continue  # перейти к следующему сегменту
            # бюджеты политики для пакета
            budgets = [_assign_budget(proc, sp, seed) for sp, seed in batch]  # бюджет каждой пары по политике процессора
            # Кэш: попадание разрешается мгновенно, новые задачи учатся на процессоре.
            # Все слияния идут строго в порядке позиций пакета, поэтому порядок
            # вставки экспертов в словарь (от него зависит выбор в генераторе
            # предложений) совпадает с исходным потоком.
            hits: list = []  # для каждой позиции пакета — отличие из кэша или None
            jobs: list[tuple[Any, int, int]] = []  # новые задачи к обучению: (задача, зерно, бюджет)
            fresh_pos: list[tuple[tuple, int]] = []  # их позиции и бюджеты: ((seg_idx, зерно), бюджет)
            for (sp, seed), budget in zip(batch, budgets):  # по парам (задача, зерно) и их бюджетам
                pos = (seg_idx, seed)  # позиция в потоке: (номер сегмента, зерно)
                hit = cache.key(pos, budget) if cache is not None else None  # отличие из кэша для (pos, budget), если кэш есть
                if hit is not None:  # попадание в кэш
                    hits.append(hit)  # записать отличие в позицию
                else:  # промах
                    hits.append(None)  # в позиции пока пусто
                    jobs.append((sp, seed, budget))  # задача идёт на обучение
                    fresh_pos.append((pos, budget))  # запомнить её позицию и бюджет
            if jobs:  # есть новые задачи
                if learner is None:  # обучатель не задан
                    # без обучателя: последовательное обучение в этом процессе
                    diffs = [_learn_worker_local(proc, sp, seed, budget,  # отличия по каждой новой задаче
                                                 train_neural)  # с общим признаком обучения нейронного ядра
                             for sp, seed, budget in jobs]  # по тройкам (задача, зерно, бюджет)
                else:  # обучатель задан
                    diffs = learner.batch(jobs)  # получить отличия пакета от обучателя
                for (pos, budget), d in zip(fresh_pos, diffs):  # по позициям и полученным отличиям
                    if cache is not None:  # кэш есть
                        cache.put(pos, budget, d)  # записать отличие по ключу (pos, budget)
            else:  # новых задач нет
                diffs = []  # список отличий пуст
            # Защита от двойного применения: отличия новых задач (слот None) уже
            # применены общим процессором при обучении. Сливать нужно только отличия
            # из кэша (полученные на другом состоянии процессора). Порядок слияния —
            # исходный порядок пакета.
            it = iter(diffs)  # итератор по отличиям новых задач
            for slot in hits:  # по позициям пакета в порядке потока
                if slot is None:  # новая задача — отличие уже применено
                    d = next(it)  # взять её отличие
                    if actions is not None:  # действия записываются
                        actions.append(("seg", d["ev"]["task"],  # дописать ("seg", имя задачи из события,
                                        d["ev"]["action"]))  # действие из события)
                    continue  # слияние не нужно — к следующей позиции
                d = slot  # отличие из кэша
                learner._merge(d)  # слить его с процессором
                if actions is not None:  # действия записываются
                    actions.append(("seg", d["ev"]["task"], d["ev"]["action"]))  # дописать ("seg", имя задачи, действие)


# ------------------------------------------------------------------- loop ----

def _first_divergence(incumbent, opt) -> Optional[dict]:
    """Первая позиция, где оптимизированный повтор отличается от исходного потока
    (диагностика отказа гейта): другая запись (имя задачи или действие) либо другая
    длина потока. Возвращает словарь с позицией и обеими записями либо None, если
    потоки совпадают."""
    a = getattr(incumbent, "stream_actions", None) or []  # записи действий исходного потока (пустой список, если атрибута нет)
    b = getattr(opt, "stream_actions", None) or []  # записи действий оптимизированного потока
    for i, (x, y) in enumerate(zip(a, b)):  # по общим позициям i с записями x (исходный) и y (оптимизированный)
        if x != y:  # записи различаются
            return {"pos": i, "incumbent": x, "optimized": y}  # вернуть позицию и обе записи
    if len(a) != len(b):  # длины потоков разные
        return {"pos": min(len(a), len(b)), "len_incumbent": len(a),  # вернуть позицию = min(длины) и обе длины
                "len_optimized": len(b)}  # длина оптимизированного потока
    return None  # расхождений нет


def run_self_development(max_rounds: int = 4,  # саморазвитие: не более max_rounds раундов
                         base_budget: int = 200000,  # базовый бюджет узлов на задачу; укладывается в 64 бита (вычисления — Python/numpy int64),
                         # промежуточные значения ограничены константой MAG_CAP = 2^28 в векторизованном вычислителе
                         floor_budget: int = 200,  # нижний бюджет узлов для предсказанно неудачных задач
                         proposals_per_round: int = 10,  # число новых предложений в раунде
                         workers: int = 0,  # число исполнителей (0 — все процессоры, не больше 32)
                         batch_size: int = 8,  # размер пакета задач
                         floor_ladder: list[int] | None = None,  # расписание нижнего бюджета; None — [floor_budget]
                         initial_segments: list[tuple[int, int]] | None = None,  # записанный ранее поток предложений для продолжения
                         train_neural: bool = True,  # обучать ли нейронное ядро при каждом обучении
                         ) -> dict[str, Any]:  # возвращает словарь-отчёт
    """Цикл саморазвития: раунды РОСТ -> ВЫВОД модели -> ОПТИМИЗАЦИЯ -> ГЕЙТ, до отказа.

    Раунд 0 — исходный поток: базовый набор задач, затем (при заданном
    initial_segments) повтор записанного потока предложений без политики. Каждый
    раунд: (1) к потоку добавляется сегмент из proposals_per_round новых предложений;
    (2) решающий список выводится по следу и проверяется на отложенной части —
      неверные уверенные предсказания дают HOLD и остановку;
    (3) тот же поток прогоняется процессором с бюджетом от модели (неизменившийся
      префикс берётся из кэша); (4) гейт безрегрессии: эксперты исходного сохранены,
      проверка набора точна и полна, стоимость принята ≤ стоимость исходного·(1 − MIN_RELIEF).
    При COMMIT оптимизированный процессор становится исходным. При нехватке
    снижения и незаконченном расписании floor_ladder меньший бюджет исчерпан —
    берётся следующий: задача, удержанная на базовом бюджете, не станет принятой
    при меньшем поиске, так что тот же HOLD лишь дешевеет. Цикл кончается, когда
    расписание исчерпано, либо на HOLD. Возвращает словарь-отчёт с полями rounds
    (по раунду), baseline, segments, final, converged и _final_incumbent (сам
    итоговый процессор)."""
    workers = workers or min(32, (os.cpu_count() or 4))  # исполнители: заданное число, иначе min(32, число процессоров или 4)
    ladder = list(floor_ladder) if floor_ladder else [floor_budget]  # расписание нижнего бюджета: копия floor_ladder либо [floor_budget]
    report: dict[str, Any] = {"rounds": [], "base_budget": base_budget,  # отчёт: пустой список раундов и базовый бюджет
                              "floor_ladder": ladder, "workers": workers}  # расписание нижнего бюджета и число исполнителей
    suite = hagi.make_task_suite()  # базовый учебный набор задач
    segments: list[tuple[int, int]] = [tuple(s) for s in (initial_segments or [])]  # поток предложений: копия initial_segments (пустой список, если None)
    round_offset = len(segments)  # смещение нумерации раундов = числу уже записанных сегментов
    cache = ReplayCache()  # пустой кэш повторного прогона

    # Раунд 0, исходная точка: только базовый набор, по одному.
    incumbent = TracedProcessorV9(node_budget=base_budget)  # исходный процессор со следом и бюджетом base_budget
    incumbent.stream_names = []  # список имён задач потока
    incumbent.stream_actions = []  # список действий потока
    for sp in suite:  # по задачам базового набора
        incumbent.stream_names.append(("suite", sp.name))  # дописать ("suite", имя задачи)
        incumbent.learn(sp, seed=10_000, train_neural=train_neural)  # выучить задачу с зерном 10000
    if segments:  # есть записанный поток предложений — продолжение
        # повторить записанный поток без политики (семантика роста)
        run_stream(incumbent, segments, suite=[], cache=None,  # прогон сегментов на исходном процессоре: без базового набора и без кэша
                   learner=ParallelLearner(incumbent, workers,  # обучатель на самом исходном процессоре
                                           train_neural=train_neural),  # с тем же признаком обучения нейронного ядра
                   batch_size=batch_size, train_neural=train_neural)  # размер пакета и признак обучения нейронного ядра
    prev_wasted = sum(t["cost"] for t in incumbent.trace if not t["success"])  # стоимость всех неудачных задач следа W = Σ cost по записям с success = 0
    report["baseline"] = {  # отчёт об исходной точке
        "learn_calls": len(incumbent.trace),  # число вызовов обучения = длина следа
        "commits": sum(t["success"] for t in incumbent.trace),  # число принятых задач = Σ success
        "wasted_candidates_on_holds": prev_wasted,  # кандидатов, потраченных на неудачи
        "pipeline": describe_pipeline(incumbent),  # описание конвейера исходного процессора
    }  # конец словаря

    adopted = True  # признак «предыдущий раунд принят» — цикл продолжается
    round_idx = 0  # номер раунда
    t_start = time.perf_counter()  # начало отсчёта общего времени
    while adopted and round_idx < max_rounds:  # пока раунды принимаются и лимит не исчерпан
        round_idx += 1  # номер раунда: round_idx + 1
        t0 = time.perf_counter()  # начало отсчёта времени раунда
        # 1. РОСТ: дополнить исходный поток новыми предложениями этого раунда.
        segments.append((proposals_per_round,  # добавить сегмент: число предложений proposals_per_round
                         500_000 + 1000 * (round_idx + round_offset)))  # базовое зерно = 500000 + 1000·(номер раунда + смещение)
        grow_learner = ParallelLearner(incumbent, workers,  # обучатель роста на исходном процессоре
                                       train_neural=train_neural)  # с признаком обучения нейронного ядра
        run_stream(incumbent, [segments[-1]], suite=[], cache=None,  # прогон только нового сегмента: без базового набора и кэша
                   learner=grow_learner, batch_size=batch_size,  # обучателем роста с заданным размером пакета
                   train_neural=train_neural)  # и признаком обучения нейронного ядра
        incumbent_wasted = sum(t["cost"] for t in incumbent.trace  # стоимость неудач исходного потока W_inc = Σ cost по записям
                               if not t["success"])  # с success = 0

        # 2. ВЫВОД модели самого себя по следу и её проверка на отложенной части.
        trace = incumbent.trace  # след исходного процессора
        dl = DecisionListSelfModel(trace)  # решающий список по следу
        val = holdout_validate(trace)  # проверка на отложенной части следа
        rnd_report: dict[str, Any] = {  # отчёт раунда
            "round": round_idx,  # номер раунда
            "learn_calls": len(trace),  # число вызовов обучения = длина следа
            "self_model": dl.describe(),  # описание модели самого себя
            "holdout": val,  # результат проверки на отложенной части
            "impossibility_certificate": impossibility_certificate(trace),  # свидетельство невозможности точного предсказателя (или None)
            "neural_adapters": sum(1 for ex in incumbent.experts.values()  # число экспертов с нейронным адаптером
                                    if ex.adapter is not None),  # = Σ 1 по экспертам, у которых adapter не None
        }  # конец словаря
        if not val["exact"]:  # модель не точна на отложенной части
            rnd_report["gate"] = {"action": "HOLD",  # гейт: отказ (HOLD)
                                  "reason": "confidently-wrong HOLD predictions on holdout"}  # причина: уверенно неверные предсказания неуспеха на отложенной части
            report["rounds"].append(rnd_report)  # дописать отчёт раунда
            adopted = False  # остановить цикл
            break  # выйти из цикла раундов

        # 3. ОПТИМИЗАЦИЯ: тот же детерминированный поток с бюджетом от модели
        #    самого себя (префикс из кэша — учатся только новые задачи).
        floor_budget = ladder[0]  # текущий нижний бюджет — первый элемент расписания
        opt = SelfModelPolicyProcessor(dl, base_budget, floor_budget,  # оптимизированный процессор с решающим списком dl и бюджетами base/floor
                                       node_budget=base_budget)  # текущий бюджет узлов = базовый
        opt.stream_names = []  # список имён задач потока
        opt.stream_actions = []  # список действий потока
        opt_learner = ParallelLearner(opt, workers, train_neural=train_neural)  # обучатель оптимизированного потока
        run_stream(opt, segments, suite=suite, cache=cache,  # прогон всех сегментов с базовым набором и кэшем
                   learner=opt_learner, batch_size=batch_size,  # обучателем и размером пакета
                   train_neural=train_neural)  # с признаком обучения нейронного ядра
        opt_wasted = sum(t["cost"] for t in opt.trace if not t["success"])  # стоимость неудач оптимизированного потока W_opt = Σ cost по записям с success = 0

        # 4. Гейт приёма без регрессии.
        base_names = set(incumbent.experts.keys())  # множество имён экспертов исходного потока E_inc
        opt_names = set(opt.experts.keys())  # множество имён экспертов оптимизированного потока E_opt
        experts_ok = opt_names >= base_names  # эксперты сохранены: E_inc ⊆ E_opt
        check = opt.verify_all(suite)  # точная проверка оптимизированного процессора на базовом наборе
        # Гейт смотрит на реальное покрытие базового набора, а не на число экспертов.
        floors_ok = (  # порог качества выполнен, если одновременно:
            check["accuracy"] == 1.0  # доля верных ответов равна 1
            and check.get("suite_complete", False)  # и нет задач набора без эксперта (при отсутствии ключа — False)
        )  # конец условия
        improved = opt_wasted <= int(incumbent_wasted * (1.0 - MIN_RELIEF))  # снижение подтверждено: W_opt ≤ int(W_inc·(1 − 0.02))
        # Запас снижения: улучшение раунда меньше MIN_RELIEF не является
        # подтверждённым ростом — лучше остановиться, чем коплить долг исследования.
        if experts_ok and floors_ok and improved:  # все три условия гейта выполнены
            rnd_report["gate"] = {"action": "COMMIT",  # гейт: принять (COMMIT)
                                  "experts_incumbent": len(base_names),  # число экспертов исходного
                                  "experts_optimized": len(opt_names),  # число экспертов оптимизированного
                                  "exact_accuracy": check["accuracy"],  # доля верных ответов проверки
                                  "wasted_candidates":  # стоимость неудач в виде текста
                                      f"{incumbent_wasted} -> {opt_wasted}",  # «исходная -> оптимизированная»
                                  "overrides": opt.policy_overrides,  # число задач с уменьшенным бюджетом
                                  "seconds": round(time.perf_counter() - t0, 2)}  # длительность раунда в секундах с 2 знаками
            prev_wasted = opt_wasted  # стоимость неудач теперь = W_opt (переменная далее не используется)
            incumbent = opt  # оптимизированный процессор становится исходным
            report["rounds"].append(rnd_report)  # дописать отчёт раунда
        else:  # хотя бы одно условие не выполнено
            reason = ("lost experts" if not experts_ok else  # причина: потеряны эксперты
                      "quality floor violated" if not floors_ok else  # иначе нарушен порог качества
                      f"no certified improvement (relief below "  # иначе снижение меньше запаса
                      f"{MIN_RELIEF:.0%} margin)")  # текст причины завершается: «ниже запаса 2%»
            if reason.startswith("no certified") and len(ladder) > 1:  # снижение не подтверждено и в расписании есть меньший бюджет
                # смена рычага, а не остановка: та же модель, меньший нижний бюджет
                used = ladder.pop(0)  # убрать из расписания исчерпанный бюджет used
                rnd_report["gate"] = {  # гейт: смена рычага
                    "action": "LEVER_CHANGE",  # действие — смена рычага
                    "reason": f"floor {used} exhausted -> floor {ladder[0]} "  # причина: бюджет used исчерпан, следующий — новый первый элемент расписания
                              "(R223: switch the lever, don't repeat it)",  # пояснение к причине
                    "wasted_candidates": f"{incumbent_wasted} -> {opt_wasted}",  # стоимость неудач «исходная -> оптимизированная»
                }  # конец словаря
                report["rounds"].append(rnd_report)  # дописать отчёт раунда
                continue  # продолжать рост с новым рычагом
            rnd_report["gate"] = {  # гейт: отказ
                "action": "HOLD",  # действие — отказ
                "reason": ("lost experts" if not experts_ok else  # причина: потеряны эксперты
                           "quality floor violated" if not floors_ok else  # иначе нарушен порог качества
                           f"no certified improvement (relief below "  # иначе снижение меньше запаса
                           f"{MIN_RELIEF:.0%} margin) — "  # с пояснением в конце
                           "self-development converged"),  # «саморазвитие сошлось»
                "experts_incumbent": len(base_names),  # число экспертов исходного
                "experts_optimized": len(opt_names),  # число экспертов оптимизированного
                "accuracy": check["accuracy"],  # доля верных ответов проверки
                "wasted_candidates": f"{incumbent_wasted} -> {opt_wasted}",  # стоимость неудач «исходная -> оптимизированная»
                "lost": sorted(base_names - opt_names)[:3],  # до 3 имён потерянных экспертов в алфавитном порядке
                "first_divergence": _first_divergence(incumbent, opt),  # первое расхождение потоков (диагностика)
            }  # конец словаря
            report["rounds"].append(rnd_report)  # дописать отчёт раунда
            adopted = False  # остановить цикл

    final = incumbent  # итоговый процессор
    report["segments"] = [[int(n), int(s)] for n, s in segments]  # поток предложений как списки [число предложений, зерно] — целые числа
    report["_final_incumbent"] = final  # сам итоговый процессор (убирается перед записью в файл)
    report["final"] = {  # итог цикла
        "adopted_rounds": sum(1 for r in report["rounds"]  # число принятых раундов
                              if r["gate"]["action"] == "COMMIT"),  # = Σ 1 по раундам с действием COMMIT
        "experts": len(final.experts),  # число экспертов
        "theorems": len(final.theorems),  # число теорем
        "neural_adapters": sum(1 for ex in final.experts.values()  # число экспертов с нейронным адаптером
                                if ex.adapter is not None),  # = Σ 1 по экспертам, у которых adapter не None
        "pipeline": describe_pipeline(final),  # описание конвейера
        "wasted_candidates_on_holds": sum(t["cost"] for t in final.trace  # стоимость всех неудач = Σ cost по записям следа
                                          if not t["success"]),  # с success = 0
        "seconds_total": round(time.perf_counter() - t_start, 2),  # общее время цикла в секундах с 2 знаками
        "workers": workers,  # число исполнителей
    }  # конец словаря
    report["converged"] = not adopted  # цикл сошёлся, если остановился отказом гейта, а не исчерпанием лимита раундов
    return report  # вернуть отчёт


def main() -> None:
    """Точка входа командной строки: запустить цикл саморазвития и записать отчёт.

    Отчёт без процессора пишется в experiments/selfloop_result.json; по флагу --json
    печатается целиком, иначе — одна итоговая строка и путь к файлу."""
    p = argparse.ArgumentParser(description="HAGI self-development loop")  # разборщик аргументов командной строки
    p.add_argument("--max-rounds", type=int, default=4)  # максимальное число раундов (по умолчанию 4)
    p.add_argument("--workers", type=int, default=0,  # число исполнителей (по умолчанию 0)
                   help="parallel workers (0 = all CPUs, capped at 32)")  # подсказка: 0 — все процессоры, не больше 32
    p.add_argument("--batch-size", type=int, default=8)  # размер пакета задач (по умолчанию 8)
    p.add_argument("--floor-ladder", type=int, nargs="*", default=[2000, 200, 50, 20],  # расписание нижнего бюджета: 2000, 200, 50, 20
                   help="certified lever-change schedule for the floor budget")  # подсказка к расписанию
    p.add_argument("--json", action="store_true")  # флаг печати отчёта в JSON
    a = p.parse_args()  # разобрать аргументы командной строки
    rep = run_self_development(max_rounds=a.max_rounds, workers=a.workers,  # запуск цикла с лимитом раундов и числом исполнителей
                               batch_size=a.batch_size,  # размером пакета
                               floor_ladder=a.floor_ladder)  # и расписанием нижнего бюджета
    rep.pop("_final_incumbent", None)  # убрать процессор из отчёта (не сериализуется); отсутствие ключа не ошибка
    out = Path("experiments") / "selfloop_result.json"  # путь файла отчёта experiments/selfloop_result.json
    out.parent.mkdir(exist_ok=True)  # создать каталог experiments, если его нет
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")  # записать отчёт с отступом 2, кириллица без экранирования
    if a.json:  # запрошен полный отчёт
        print(json.dumps(rep, indent=2, ensure_ascii=False))  # напечатать отчёт в JSON
    else:  # иначе
        print(f"rounds adopted: {rep['final']['adopted_rounds']}, "  # принято раундов
              f"converged: {rep['converged']}, experts: {rep['final']['experts']}, "  # сходимость и число экспертов
              f"wasted: {rep['final']['wasted_candidates_on_holds']}, "  # стоимость неудач
              f"{rep['final']['seconds_total']}s on {rep['final']['workers']} workers")  # время и число исполнителей
        print(f"report: {out}")  # путь к файлу отчёта


if __name__ == "__main__":  # запуск файла как скрипта, а не импорт
    main()  # точка входа командной строки
