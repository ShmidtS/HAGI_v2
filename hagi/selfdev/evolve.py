"""Эволюция алгоритма обучения: поиск гиперпараметров `train_optimal` популяцией с мутацией и
скрещиванием, принимаемый только парным гейтом по контрольной кросс-энтропии.

Идея: скорость обучения, пакет и момент — не константы и не мнение автора, а параметры,
которые модель подбирает о себе сама. Претендент обучает СНИМОК ствола на свежем чанке при
фиксированном бюджете шагов; эталон — те же шаги с текущими принятыми значениями. Претендент
принимается, если на КАЖДОМ из контрольных чанков не хуже эталона и в среднем строго лучше
(иначе — отклоняется; парный тест отсекает «повезло на одном чанке»). Журнал и принятое
состояние лежат в CKPT_DIR/evolve_state.json; `train_optimal` читает их через `current()`."""

from __future__ import annotations  # отложенное вычисление аннотаций типов

import json  # чтение и запись JSON
import math  # логарифм
import os as _os  # переменные окружения
import random  # генератор случайных чисел
import time  # метки времени
from typing import Any  # типы

# Пространство поиска: имя параметра -> (минимум, максимум, масштаб, исходное значение).
# Масштаб "log" — параметр положителен и варьируется в разы (шаги, вес затухания); "lin" — по линейной
# оси. Диапазоны широкие (шаги — в 100 раз, пакет — в 16 раз): границы задают только физический
# смысл, а не мнение автора; внутри мутация не ограничена никакой сеткой значений.
SPACE: dict[str, tuple[float, float, str, float]] = {  # пространство
    "lr_muon": (1e-4, 1.0, "log", 0.03),  # шаг Muon (матрицы)
    "lr_adam": (1e-5, 0.3, "log", 5e-3),  # шаг AdamW (таблицы и векторы)
    "batch": (4, 128, "lin", 16),  # размер пакета (число потоков при переносе состояния)
    "momentum": (0.0, 0.99, "lin", 0.9),  # момент Muon
    "wd": (0.0, 0.1, "log", 1e-4),  # затухание весов
    "warmup_frac": (0.02, 0.5, "lin", 1 / 15),  # доля шагов разогрева
    "min_scale": (0.0, 0.5, "lin", 0.05),  # нижний предел масштаба шага в конце отжига
    "beta2": (0.9, 0.9999, "lin", 0.999),  # второй момент AdamW
}
DEFAULTS: dict[str, float] = {k: v[3] for k, v in SPACE.items()}  # исходные значения (как у автора кода)
POP_SIZE = int(_os.environ.get("HAGI_EVOLVE_POP", "4"))  # размер популяции
PERIOD = int(_os.environ.get("HAGI_EVOLVE_EVERY", "8"))  # раз в столько чанков — одно поколение
BUDGET_STEPS = int(_os.environ.get("HAGI_EVOLVE_STEPS", "40"))  # шагов обучения на оценку фитнеса
SIGMA = float(_os.environ.get("HAGI_EVOLVE_SIGMA", "0.25"))  # исходная относительная ширина мутации
COST_LAMBDA = float(_os.environ.get("HAGI_EVOLVE_COST", "0.03"))  # цена вычислений: наты штрафа за удвоение числа примеров на шаг (batch); 0 — качество любой ценой
COST_REF = 16.0  # опорный пакет (его цена — ноль)


def _clip(k: str, v: float) -> float:  # вернуть значение в диапазон
    lo, hi, sc, _ = SPACE[k]  # границы и масштаб
    if sc == "log":  # логарифмическая ось
        return float(min(hi, max(lo, v)))  # положительное значение в границах
    return float(min(hi, max(lo, v)))  # линейная ось


def _mutate(g: dict[str, float], rng: random.Random, sigma: float = SIGMA) -> dict[str, float]:  # мутация генома
    """Гауссова мутация: лог-параметры умножаются на exp(N(0, σ)), линейные сдвигаются на
    σ·(макс−мин)·N(0,1); мутирует каждый ген с вероятностью 1/2 — никаких сеток значений."""
    g = dict(g)  # копия
    for k, (lo, hi, sc, _) in SPACE.items():  # по генам
        if rng.random() > 0.5:  # этот ген не мутирует
            continue  # дальше
        z = rng.gauss(0.0, 1.0)  # нормальное отклонение
        if sc == "log":  # логарифмическая ось
            g[k] = _clip(k, max(lo, g[k]) * math.exp(sigma * z))  # умножение
        else:  # линейная ось
            g[k] = _clip(k, g[k] + sigma * (hi - lo) * z)  # сдвиг
    return g  # потомок


def _crossover(a: dict[str, float], b: dict[str, float], rng: random.Random, alpha: float = 0.3) -> dict[str, float]:  # скрещивание
    """BLX-α: потомок по каждому гену — равномерно из расширенного на α интервала между родителями
    (выходит за родителей — поиск не запирается внутри выпуклой оболочки популяции)."""
    out = dict(a)  # заготовка
    for k in SPACE:  # по генам
        lo, hi = min(a[k], b[k]), max(a[k], b[k])  # интервал родителей
        w = hi - lo  # ширина
        out[k] = _clip(k, rng.uniform(lo - alpha * w, hi + alpha * w))  # точка из расширенного интервала
    return out  # потомок


def _round(g: dict[str, float]) -> dict[str, Any]:  # пакет — целое
    return {**g, "batch": int(max(4, round(g["batch"])))}  # целое число потоков


def _state_path():  # путь к журналу эволюции
    from hagi.base.util import CKPT_DIR  # каталог чекпойнтов
    return CKPT_DIR / "evolve_state.json"  # файл состояния


def current() -> dict[str, Any]:  # принятые значения (применяются в train_optimal)
    """Принятые значения параметров; отсутствующие берутся из DEFAULTS."""
    p = _state_path()  # путь
    st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}  # прочитать, если есть
    return _round({**DEFAULTS, **{k: v for k, v in st.get("current", {}).items() if k in SPACE}})  # только объявленные


def _clone(model):  # копия модели с теми же весами
    """Копия без модулей в атрибутах (модель кэширует torch как атрибут — его deepcopy не берёт)."""
    import types  # типы
    import copy  # копия
    m = object.__new__(type(model))  # пустой экземпляр того же класса
    m.__dict__.update({k: (v if isinstance(v, types.ModuleType) else copy.deepcopy(v)) for k, v in model.__dict__.items()})  # модули — по ссылке, остальное — глубоко
    return m  # копия


def fitness(model, genome: dict[str, Any], data: bytes) -> float:  # оценка генома
    """Обучить КОПИЮ ствола на data бюджетом BUDGET_STEPS шагов; вернуть контрольную CE (меньше — лучше)."""
    m = _clone(model)  # снимок ствола (ствол вызывающего не трогается)
    m.active_fiber = None; m.active_adapter = None  # без волокон и адаптеров — оцениваем ствол
    r = m.train_optimal(data, epochs=1, batch=int(genome["batch"]), seq_len=m.k, seed=7,  # обучение
                        lr_muon=float(genome["lr_muon"]), lr_adam=float(genome["lr_adam"]), momentum=float(genome["momentum"]),  # шаги, момент и пакет претендента
                        max_steps_per_epoch=BUDGET_STEPS, gpm_left={}, log_every=0)  # без GPM: сравниваем оптимизатор, а не защиту
    return float(r["val_ce_after"]) + COST_LAMBDA * math.log2(max(1.0, float(genome["batch"])) / COST_REF)  # контроль после обучения плюс цена вычислений на шаг


def evolve_step(model, chunks: list[bytes], seed: int | None = None) -> dict[str, Any]:  # одно поколение
    """Одно поколение: популяция из текущего генома + мутации/скрещивания; лучший кандидат
    проходит парный гейт против эталона на каждом чанке. Возвращает отчёт с решением."""
    rng = random.Random(seed if seed is not None else int(time.time() * 1000) % 2**31)  # генератор
    base = current()  # эталон — принятый геном
    st0 = json.loads(_state_path().read_text(encoding="utf-8")) if _state_path().exists() else {}  # прежнее состояние
    sigma = float(st0.get("sigma", SIGMA))  # текущая ширина мутации (1/5 правило: принятие — расширить, отказ — сузить)
    probe = chunks[-1]  # чанк для оценки фитнеса
    pop = [base]  # популяция: эталон
    while len(pop) < POP_SIZE:  # наполнить
        parents = rng.sample(pop, min(2, len(pop)))  # два родителя из уже имеющихся
        child = _crossover(parents[0], parents[-1], rng) if len(parents) == 2 and rng.random() < 0.4 else _mutate(rng.choice(pop), rng, sigma)  # скрещивание или мутация
        pop.append(child)  # потомок
    scored = sorted(((fitness(model, g, probe), i, g) for i, g in enumerate(pop)), key=lambda x: x[0])  # ранжирование по фитнесу
    best_f, _, best = scored[0]  # лучший
    decision = "rejected"  # по умолчанию — отклонён
    gate: list[dict[str, float]] = []  # парные значения по чанкам
    if best != base and best_f < scored[[g for _, _, g in scored].index(base)][0]:  # кандидат не эталон и лучше на зонде
        for ch in chunks[-3:]:  # парный тест на последних чанках
            fb, fc = fitness(model, base, ch), fitness(model, best, ch)  # эталон и кандидат
            gate.append({"base": round(fb, 4), "cand": round(fc, 4)})  # записать
        ok = all(g["cand"] <= g["base"] + 1e-3 for g in gate) and sum(g["cand"] - g["base"] for g in gate) < 0  # не хуже на каждом, лучше в среднем
        if ok:  # прошёл
            decision = "adopted"  # принят
    p = _state_path()  # путь состояния
    st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}  # прочитать
    st.setdefault("history", []).append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "best": best, "probe": round(best_f, 4),  # запись истории
                                         "decision": decision, "gate": gate, "pop": [(round(f, 4), g) for f, _, g in scored]})  # с популяцией
    st["sigma"] = min(1.0, sigma * (1.5 if decision == "adopted" else 0.7))  # адаптация ширины мутации (в границах 0.02..1.0)
    st["sigma"] = max(0.02, st["sigma"])  # не уже 0.02
    if decision == "adopted":  # принят
        st["current"] = _round(best)  # применить (пакет — целое)
    st["current"] = st.get("current", base)  # текущий геном
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")  # записать
    return {"decision": decision, "best": best, "probe": best_f, "gate": gate, "pop": scored[:3]}  # отчёт


ARCH_LAMBDA = float(_os.environ.get("HAGI_ARCH_LAMBDA", "0.02"))  # цена параметра: наты выигрыша на каждые 10% новых параметров
ARCH_PERIOD = int(_os.environ.get("HAGI_ARCH_EVERY", "24"))  # раз в столько чанков — проба архитектуры
ARCH_MAX_PARAMS = int(_os.environ.get("HAGI_ARCH_MAX", "2000000"))  # предел суммарного объёма ствола


def evolve_arch_step(unified, chunks: list[bytes], seed: int | None = None) -> dict[str, Any]:  # проба роста ширины
    """Эволюция архитектуры: претендент — точное расширение ширины ствола на delta (функция модели
    не меняется, после — дообучение тем же бюджетом шагов). Эталон — то же дообучение без расширения.
    Претендент принят, если его контрольная CE ниже эталона на величину цены новых параметров
    (ARCH_LAMBDA натов на каждые 10% параметров) и суммарный объём не выше ARCH_MAX_PARAMS."""
    from hagi.models.optimal_lm import grow_width as _gw  # точное расширение
    rng = random.Random(seed if seed is not None else int(time.time() * 1000) % 2**31)  # генератор
    g = current(); probe = chunks[-1]; hd = unified.model.dim // unified.model.n_heads  # геном, зонд, ширина головы
    deltas = sorted({max(hd * unified.model.n_heads, hd * k) for k in rng.sample([1, 2, 3, 4], 3)})  # три случайных шага ширины (кратно heads·hd)
    n0 = sum(q.numel() for q in unified.model.params())  # параметров сейчас
    f_base = fitness(unified.model, g, probe)  # эталон: дообучение без расширения
    best: tuple[float, int] | None = None  # лучший (выигрыш за вычетом цены, delta)
    tried: list[dict[str, Any]] = []  # пробы
    for dlt in deltas:  # по претендентам
        m = _clone(unified.model)  # копия ствола
        try:  # расширение может не пройти по кратности
            _gw(m, dlt)  # расширить
        except AssertionError:  # не делится на головы
            continue  # дальше
        n1 = sum(q.numel() for q in m.params())  # параметров после расширения
        if n1 > ARCH_MAX_PARAMS:  # предел объёма превышен
            continue  # дальше
        f_c = fitness(m, g, probe)  # кандидат: дообучение после расширения
        cost = ARCH_LAMBDA * ((n1 / n0) - 1.0) / 0.10  # цена в натах
        tried.append({"delta": dlt, "ce": round(f_c, 4), "cost": round(cost, 4)})  # записать
        net = f_base - f_c - cost  # чистый выигрыш
        if net > 0 and (best is None or net > best[0]):  # положителен и лучше прежних
            best = (net, dlt)  # запомнить
    if best is None:  # никто не прошёл
        _log_arch({"decision": "rejected", "base": round(f_base, 4), "tried": tried})  # журнал
        return {"decision": "rejected", "base": f_base, "tried": tried}  # отчёт
    info = unified.grow_width_optimal(best[1])  # применить к живой модели (волокна и базисы перенесены)
    _log_arch({"decision": "adopted", "delta": best[1], "net": round(best[0], 4), "base": round(f_base, 4), "tried": tried, "d": info["d"]})  # журнал
    return {"decision": "adopted", "delta": best[1], "net": best[0], "base": f_base, "tried": tried, **info}  # отчёт


def _log_arch(rec: dict[str, Any]) -> None:  # дописать пробу архитектуры в состояние
    p = _state_path()  # путь
    st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}  # прочитать
    st.setdefault("arch", []).append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec})  # добавить
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")  # записать


def maybe_evolve_arch(unified, chunks: list[bytes], counter: list[int]) -> dict[str, Any] | None:  # вызов из цикла
    """Раз в ARCH_PERIOD чанков — проба расширения ширины (модель не должна расти бесконечно:
    суммарный предел ARCH_MAX_PARAMS из окружения HAGI_ARCH_MAX, по умолчанию 2 000 000 параметров)."""
    if _os.environ.get("HAGI_ARCH_EVOLVE", "1") != "1" or len(chunks) < 4:  # выключено или мало данных
        return None  # ничего
    counter[0] += 1  # счётчик
    if counter[0] % ARCH_PERIOD:  # не время
        return None  # ничего
    return evolve_arch_step(unified, chunks)  # проба


def maybe_evolve(model, chunks: list[bytes], counter: list[int]) -> dict[str, Any] | None:  # вызов из цикла
    """Раз в PERIOD чанков — одно поколение;chunks — последние чанки (не менее 4)."""
    if _os.environ.get("HAGI_EVOLVE", "1") != "1" or len(chunks) < 4:  # выключено или мало данных
        return None  # ничего
    counter[0] += 1  # счётчик
    if counter[0] % PERIOD:  # не время
        return None  # ничего
    return evolve_step(model, chunks)  # поколение
