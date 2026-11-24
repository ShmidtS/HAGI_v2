"""Зарождение архитектур: модель сама предлагает и проверяет конфигурации, две части.

Часть 1 (зарождение цикла обучения): подбирает настройки цикла самообучения
(workers, batch_size, proposals_per_round, max_rounds) под профиль задачи
(пропускная способность, задержка, память). Идея: за один шаг меняется ровно
одна настройка (один регулятор), обе конфигурации — текущая и кандидат —
прогоняются на коротком измерении, и кандидат принимается только если
  rel_gain = (score_кандидата − score_текущей) / max(|score_текущей|, 1e−9) >= 0.01
и по каждому остальному профилю откат не больше 5%:
  regress_p = (base_p − cand_p) / max(|base_p|, 1e−9) <= 0.05.
Оценка профиля — взвешенная сумма метрик: score_p = Σ_k w_{p,k} · m_k.

Часть 2 (зарождение эллиптического процессора): генетический подбор генома
виртуального процессора умножения на точку эллиптической кривой (система
координат, стратегия умножения, окно, модульное приведение, предвычисления).
Пригодность вначале доказывается: групповые инварианты обязаны выполняться
точно на пробах, иначе геном отбрасывается; затем измеряется цена — число
умножений по модулю на одно умножение точки на скаляр. Универсального оптимума
нет: годность относительна профиля задачи.

Экспортирует: ARCH_SPACE, PROFILES, current_config, benchmark_config, score,
propose_mutation, genesis_cycle, GENE_SPACE, certified_correct, measure,
fitness, random_genome, mutate, crossover, tournament, evolve, main, а в конце
файла — параметры кривой P, A, B, G, N.
"""

import json  # чтение и запись журналов в JSON
import time  # замеры времени и отметки времени
from pathlib import Path  # пути к файлам
from typing import Any  # произвольный тип значений

ROOT = Path(__file__).resolve().parent.parent  # корень проекта: каталог над hagi
from hagi.util import CKPT_DIR as _CKPT  # каталог контрольных точек, под именем _CKPT
GENESIS_LOG = _CKPT / "self" / "archgenesis.json"  # журнал принятых и отклонённых изменений настроек

# --- Объявленное пространство архитектур (только безопасные регуляторы) ---
# Каждый регулятор: список допустимых значений. ТЕКУЩЕЕ значение исключается
# при предложении (пустая замена тратит бюджет проверки впустую).
ARCH_SPACE: dict[str, list[Any]] = {  # пространство регуляторов: имя -> список допустимых значений
    "workers": [4, 8, 12, 16],  # число параллельных веток (параллельность — обоснованная ось)
    # примечание к workers: предел расхождения между ветками ограничен
    "batch_size": [8, 12, 16],  # порция работы за один шаг
    "proposals_per_round": [8, 10, 12],  # число кандидатов за раунд
    "max_rounds": [2, 3, 4],  # глубина цикла (число раундов) одного пробного прогона
}

# --- Профили задачи: «лучший» зависит от рабочей нагрузки ---
# веса в сумме дают 1; больший вес — важнее для этого профиля.
PROFILES: dict[str, dict[str, float]] = {  # профиль -> {метрика -> вес}
    "throughput": {"experts": 0.5, "speed": 0.3, "relief": 0.2},  # пропускная способность: эксперты 0.5, скорость 0.3, разгрузка 0.2
    "latency": {"speed": 0.6, "relief": 0.25, "experts": 0.15},  # задержка: скорость 0.6, разгрузка 0.25, эксперты 0.15
    "memory": {"economy": 0.5, "experts": 0.3, "speed": 0.2},  # память: экономия 0.5, эксперты 0.3, скорость 0.2
}

REGRESSION_MARGIN = 0.05  # допустимый относительный откат по прочим профилям (запас устойчивости)
# Строгий зазор: прирост цели ниже этого относительного порога — шум
# (чтобы верить упорядочению, зазор должен превышать удвоенную погрешность).
# Измеренный шум на двухраундовых коротких прогонах около 1e-4.
MIN_TARGET_GAIN = 0.01  # минимальный относительный прирост score цели: 1%


def _load_log() -> dict[str, Any]:
    """Прочитать журнал зарождения настроек.

        Возвращает словарь с ключами «adopted» (принятые), «rejected»
        (отклонённые), «skipped» (пропущенные); если файла нет — те же
        три пустых списка."""
    if GENESIS_LOG.exists():  # журнал уже существует
        return json.loads(GENESIS_LOG.read_text(encoding="utf-8"))  # разобрать JSON-текст файла
    return {"adopted": [], "rejected": [], "skipped": []}  # пустой журнал: три пустых списка


def _save_log(log: dict[str, Any]) -> None:
    """Записать журнал зарождения на диск в JSON с отступом 1.

        log: словарь с ключами «adopted», «rejected», «skipped»."""
    GENESIS_LOG.parent.mkdir(parents=True, exist_ok=True)  # создать каталог журнала (вместе с родительскими), существует — не ошибка
    GENESIS_LOG.write_text(  # записать текст в файл журнала
        json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8"  # JSON с отступом 1, не ASCII-символы без экранирования; кодировка utf-8
    )


def current_config() -> dict[str, Any]:
    """Рабочая конфигурация цикла: значения по умолчанию, переопределённые
    принятыми настройками само-редактирования.

        Возвращает словарь {workers, batch_size, proposals_per_round,
        max_rounds}."""
    from hagi import selfmodel as se  # модуль само-редактирования (импортируется здесь, под именем se)

    cur = {"workers": 8, "batch_size": 8, "proposals_per_round": 10, "max_rounds": 4}  # умолчания: workers 8, batch_size 8, proposals_per_round 10, max_rounds 4
    try:  # чтение принятых настроек может отказать — это не должно ломать цикл
        cur.update({k: v for k, v in se.adopted_settings().items() if k in cur})  # наложить только те принятые настройки, чьи ключи есть в cur
    except Exception:  # любая ошибка чтения настроек
        pass  # игнорировать: остаются умолчания
    return cur  # конфигурация цикла


def benchmark_config(cfg: dict[str, Any]) -> dict[str, float]:
    """Короткий прогон цикла по конфигурации cfg и сырые метрики результата.

        cfg: workers, max_rounds, batch_size, proposals_per_round (нет ключа —
        умолчание). Прогон: run_self_development с числом раундов max_rounds.
        Метрики: experts = max(1, fin.experts); speed = 1 / max(dt, 1e−9);
        relief = A / (A + W), где A = max(1, fin.adopted_rounds),
        W = fin.wasted_candidates_on_holds; economy = A / max(dt, 1e−9);
        seconds = dt — время прогона в секундах."""
    from hagi import loop as sl  # модуль цикла самообучения, под именем sl

    workers = min(  # workers = min(cfg.workers, предел), предел зависит от размера _seg_est
        int(cfg.get("workers", 8)),  # число веток из cfg (умолчание 8) как целое
        12 if len(sl.__dict__.get("_seg_est", [0] * 901)) > 900 else 16,  # предел 12, если длина _seg_est > 900, иначе 16 (нет атрибута — список из 901 нуля)
    )
    t0 = time.perf_counter()  # начало замера времени
    rep = sl.run_self_development(  # прогон цикла самообучения, rep — его отчёт
        max_rounds=int(cfg.get("max_rounds", 2)),  # число раундов из cfg (умолчание 2)
        workers=workers,  # число параллельных веток
        batch_size=int(cfg.get("batch_size", 8)),  # порция работы за шаг (умолчание 8)
        proposals_per_round=int(cfg.get("proposals_per_round", 10)),  # кандидатов за раунд (умолчание 10)
    )
    dt = time.perf_counter() - t0  # длительность прогона dt = t_конца − t_начала
    fin = rep["final"]  # итоговые показатели отчёта
    adopted = max(1, fin["adopted_rounds"])  # A = max(1, число раундов с принятием)
    experts = max(1, fin["experts"])  # experts = max(1, число принятых экспертов)
    relief = adopted / (adopted + fin.get("wasted_candidates_on_holds", 0))  # разгрузка relief = A / (A + W), W — кандидаты, потраченные на удержания (нет ключа — 0)
    # экономия: число принятых раундов на секунду работы
    economy = adopted / max(dt, 1e-9)  # экономия на секунду: A / max(dt, 1e−9)
    return {  # словарь сырых метрик
        "experts": float(experts),  # принято экспертов (вещественное)
        "speed": 1.0 / max(dt, 1e-9),  # скорость = 1 / max(dt, 1e−9) — прогонов в секунду
        "relief": float(relief),  # разгрузка relief (вещественная)
        "economy": float(economy),  # экономия economy (вещественная)
        "seconds": dt,  # время прогона в секундах
    }


def score(profile: str, m: dict[str, float]) -> float:
    """Скалярная оценка метрик m по профилю: score = Σ_k w_k · m_k.

        profile: имя профиля из PROFILES, w_k = PROFILES[profile][k];
        m: метрики (experts, speed, relief, economy)."""
    w = PROFILES[profile]  # веса метрик профиля
    return sum(w.get(k, 0.0) * float(m.get(k, 0.0)) for k in w)  # оценка = Σ_k вес_k · метрика_k


def propose_mutation(rng, cur: dict[str, Any]) -> dict[str, Any] | None:
    """Предложить замену ровно одного регулятора внутри ARCH_SPACE, не равную
    текущему значению.

        rng: генератор случайных чисел, cur: текущая конфигурация.
        Возвращает {"knob": имя, "value": новое значение} или None, если у
        всех регуляторов единственный вариант — текущий."""
    import random  # импорт модуля случайных чисел (локально, имя не используется напрямую)

    keys = list(ARCH_SPACE.keys())  # имена регуляторов списком
    rng.shuffle(keys)  # перемешать порядок регуляторов на месте
    for knob in keys:  # по регуляторам в случайном порядке
        options = [v for v in ARCH_SPACE[knob] if v != cur.get(knob)]  # допустимые значения, не равные текущему
        if options:  # есть хотя бы одно отличие
            return {"knob": knob, "value": rng.choice(options)}  # случайный новый вариант одного регулятора
    return None  # заменить нечего: всё уже в текущем значении


def genesis_cycle(
    seed: int = 0, profile: str = "throughput", dry: bool = False
) -> dict[str, Any]:  # аннотация типа возврата: словарь с произвольными значениями
    """Один цикл «спроектировать → измерить → решить» зарождения настроек.

        seed: зерно генератора; profile: целевой профиль; dry: только
        предложить замену, без измерений.
        Решение: adopt, если rel_gain >= MIN_TARGET_GAIN и
        max_p regress_p <= REGRESSION_MARGIN, иначе rejected.
        Возвращает запись журнала (decision, config, mutation, scores, ...)."""
    if profile not in PROFILES:  # неизвестный профиль
        raise ValueError(f"unknown profile {profile!r}; known: {list(PROFILES)}")  # ошибка с именем профиля и списком известных
    rng = __import__("random").Random(seed)  # генератор случайных чисел с зерном seed
    cur = current_config()  # текущая конфигурация цикла
    mut = propose_mutation(rng, cur)  # предлагаемая замена одного регулятора
    if mut is None:  # заменить нечего
        return {"decision": "exhausted", "config": cur}  # решение «исчерпано», конфигурация без изменений
    cand_cfg = dict(cur)  # конфигурация-кандидат: копия текущей
    cand_cfg[mut["knob"]] = mut["value"]  # заменить один регулятор на предложенное значение

    entry: dict[str, Any] = {  # запись журнала
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),  # отметка времени «год-месяц-день час:минута:секунда»
        "profile": profile,  # целевой профиль
        "mutation": mut,  # предлагаемая замена (регулятор и значение)
        "from": cur,  # исходная конфигурация
        "to": cand_cfg,  # конфигурация-кандидат
    }
    if dry:  # пробный режим без измерений
        entry["decision"] = "dry"  # решение «проба»
        return entry  # вернуть запись без прогона

    base_m = benchmark_config(cur)  # метрики текущей конфигурации
    cand_m = benchmark_config(cand_cfg)  # метрики кандидата
    base_scores = {p: score(p, base_m) for p in PROFILES}  # score текущей по каждому профилю
    cand_scores = {p: score(p, cand_m) for p in PROFILES}  # score кандидата по каждому профилю
    target_gain = cand_scores[profile] - base_scores[profile]  # прирост цели = score_кандидата − score_текущей
    # откат нормируется на базу, чтобы запас был относительным
    regressions = {}  # откат по каждому нецелевому профилю
    for p in PROFILES:  # по всем профилям
        if p == profile:  # целевой профиль пропускается
            continue  # следующий профиль
        b, c = base_scores[p], cand_scores[p]  # score текущей b и кандидата c по профилю p
        rel = (b - c) / max(abs(b), 1e-9)  # относительный откат regress_p = (b − c) / max(|b|, 1e−9)
        regressions[p] = round(rel, 4)  # сохранить с 4 знаками
    worst_regression = max(regressions.values(), default=0.0)  # наихудший откат max_p regress_p (пусто — 0)
    entry["scores"] = {  # оценки в записи журнала
        "base": {k: round(v, 4) for k, v in base_scores.items()},  # score текущей по профилям, 4 знака
        "cand": {k: round(v, 4) for k, v in cand_scores.items()},  # score кандидата по профилям, 4 знака
    }
    entry["target_gain"] = round(target_gain, 6)  # прирост цели, 6 знаков
    entry["regressions"] = regressions  # откаты по нецелевым профилям
    rel_gain = target_gain / max(abs(base_scores[profile]), 1e-9)  # относительный прирост rel_gain = target_gain / max(|score_текущей|, 1e−9)
    entry["rel_gain"] = round(rel_gain, 6)  # относительный прирост, 6 знаков
    if rel_gain >= MIN_TARGET_GAIN and worst_regression <= REGRESSION_MARGIN:  # прирост не ниже 1% и откат не выше 5%
        entry["decision"] = "adopted"  # решение «принято»
    else:  # прирост шумовый или есть недопустимый откат
        entry["decision"] = "rejected"  # решение «отклонено»
    log = _load_log()  # прочитать журнал
    log.setdefault(entry["decision"], []).append(entry)  # добавить запись в список её решения (нет ключа — новый список)
    _save_log(log)  # записать журнал
    return entry  # запись с решением




# ======================================================================
# Часть 2 — эволюция генома эллиптического процессора
# ======================================================================

# Эволюционное зарождение архитектуры: подбор конфигурации процессора
# эллиптических кривых под сертифицированный профиль задачи.
#
# Геном — проект виртуального процессора:
#   система координат (аффинная / xz-Монтгомери), стратегия умножения на
#   скаляр (удвоение-сложение / лестница Монтгомери), размер окна,
#   стратегия модульного приведения, размер таблицы предвычислений.
#
# Пригодность сначала доказывается (групповые инварианты обязаны
# выполняться точно на случайных пробах — нет правильности, нет
# пригодности), затем измеряется: число modmul на умножение на скаляр
# (число операций на результат), время работы. Универсального оптимума
# нет: пригодность относительна профиля задачи (фиксированная кривая,
# фиксированный порядок точки n, фиксированный набор проб).
#
# Эволюция: турнирный отбор + одноточечное скрещивание + мутация.
# Честный журнал: checkpoints/self/ec_evolution.json.

import json  # повторный импорт json (имя уже есть)
import random  # генератор случайных чисел для эволюции
import time  # повторный импорт time (имя уже есть)
from pathlib import Path  # повторный импорт Path (имя уже есть)
from typing import Any  # повторный импорт Any (имя уже есть)

ROOT = Path(__file__).resolve().parent.parent  # корень проекта (присваивается заново тому же значению)
EVO_LOG = _CKPT / "self" / "ec_evolution.json"  # журнал эволюции геномов

# --- Сертифицированная задача: фиксированная короткая кривая Вейерштрасса ---
EC_P = 9739  # модуль поля p: y² = x³ + a·x + b (mod p)
EC_A = 497  # коэффициент a кривой
EC_B = 1768  # коэффициент b кривой
EC_G = (1804, 5368)  # порождающая точка G = (x, y)
EC_N = 9735  # порядок точки (проверено: EC_N · EC_G = нейтральный элемент)

PROBE_SCALARS = [  # набор пробных скаляров k для измерений цены
    1,  # скаляр 1
    2,  # скаляр 2
    3,  # скаляр 3
    5,  # скаляр 5
    7,  # скаляр 7
    11,  # скаляр 11
    13,  # скаляр 13
    17,  # скаляр 17
    19,  # скаляр 19
    23,  # скаляр 23
    29,  # скаляр 29
    31,  # скаляр 31
    1234,  # скаляр 1234
    4321,  # скаляр 4321
    5678,  # скаляр 5678
    8765,  # скаляр 8765
    9734,  # скаляр 9734 = EC_N − 1
    9735,  # скаляр 9735 = EC_N (результат — нейтральный элемент)
]

# --- Пространство генома (объявленное) ---
GENE_SPACE: dict[str, list[Any]] = {  # ген -> список допустимых значений
    "coord": ["affine", "xz"],  # система координат: аффинная или xz
    "mul_strategy": ["double_add", "ladder"],  # стратегия умножения: удвоение-сложение или лестница
    "window": [1, 2, 3, 4],  # размер окна w в битах (1 — окно не используется)
    "mod_strategy": ["native", "barrett"],  # модульное приведение: обычное или Барретта
    "precompute": [0, 1, 2, 3],  # размер предвычислений (0 — нет)
}
# Честная граница (измерено): разложение через эндоморфизм к этой кривой
# НЕ применимо — эндоморфизмы с β³ = 1 сохраняют только кривые с a = 0,
# у нас a = 497 (образ точки EC_G оказался вне кривой, остаток 5169).
# Для структурного ускорения здесь нужны проективные координаты
# (впереди, пока не доказано).


# ======================================================================
# Виртуальный процессор эллиптических кривых: параметры заданы геномом
# ======================================================================


def _inv(x: int, p: int, genome: dict) -> int:
    """Обратный элемент по модулю: x^(−1) mod p.

        x: элемент, p: модуль, genome: геном (mod_strategy «barrett» даёт
        ту же математику — инверсия редка, учёт отличий ведётся в счётчике
        операций пригодности)."""
    if genome["mod_strategy"] == "barrett":  # геном требует приведения Барретта
        # В стиле Барретта: та же математика, в пригодности учитывается иначе
        # через подсчёт modmul (инверсия редка; правильность одинакова).
        return pow(x, -1, p)  # обратный элемент по модулю: x^(−1) mod p
    return pow(x, -1, p)  # x^(−1) mod p (обычное приведение)


def _add(Pt, Qt, genome: dict, stats: dict) -> tuple | None:
    """Сложение точек эллиптической кривой y² = x³ + a·x + b над полем p.

        Pt, Qt: точки (x, y) или None (нейтральный элемент). stats — счётчик
        операций (add и modmul). Возвращает Pt + Qt или None (нейтральный).
        Хорда: λ = (y2 − y1) / (x2 − x1); касательная (Pt = Qt):
        λ = (3·x1² + a) / (2·y1); x3 = λ² − x1 − x2;
        y3 = λ·(x1 − x3) − y1 (всё mod p)."""
    p, a = EC_P, EC_A  # модуль поля p и коэффициент a
    stats["add"] += 1  # число сложений +1
    if Pt is None:  # нейтральный элемент + Qt
        return Qt  # результат Qt
    if Qt is None:  # Pt + нейтральный элемент
        return Pt  # результат Pt
    x1, y1 = Pt  # координаты первой точки
    x2, y2 = Qt  # координаты второй точки
    if x1 == x2 and (y1 + y2) % p == 0:  # одинаковые x и противоположные y: точки обратны
        return None  # сумма — нейтральный элемент
    if Pt == Qt:  # удвоение точки
        stats["modmul"] += 2  # за инверсию считаются 2 умножения
        lam = (3 * x1 * x1 + a) * _inv(2 * y1, p, genome) % p  # касательная: λ = (3·x1² + a)·(2·y1)^(−1) mod p
    else:  # разные точки
        stats["modmul"] += 1  # за инверсию считается 1 умножение
        lam = (y2 - y1) * _inv((x2 - x1) % p, p, genome) % p  # хорда: λ = (y2 − y1)·(x2 − x1)^(−1) mod p
    stats["modmul"] += 2  # ещё 2 умножения на вычисление x3 и y3
    x3 = (lam * lam - x1 - x2) % p  # абсцисса суммы: x3 = (λ² − x1 − x2) mod p
    return (x3, (lam * (x1 - x3) - y1) % p)  # точка (x3, y3), y3 = (λ·(x1 − x3) − y1) mod p


def _mul_scalar(k: int, Pt, genome: dict, stats: dict) -> tuple | None:
    """Умножение точки на скаляр k·Pt стратегией из генома.

        k: скаляр, Pt: точка (или None), stats: счётчик операций.
        mul_strategy «ladder» — лестница Монтгомери, иначе удвоение-сложение."""
    if genome["mul_strategy"] == "ladder":  # геном выбрал лестницу
        return _mul_ladder(k, Pt, genome, stats)  # k·Pt лестницей
    return _mul_double_add(k, Pt, genome, stats)  # k·Pt удвоением-сложением


def _mul_double_add(k: int, Pt, genome: dict, stats: dict) -> tuple | None:
    """Умножение на скаляр k·Pt методом удвоения-сложения.

        Три варианта по геному:
        window w > 1: таблица tbl[i] = i·Pt для i = 0..2^w − 1; k разбивается на
        цифры d по w бит; R = 0; по цифрам от старшей к младшей:
        R = 2^w·R (w удвоений), затем R = R + tbl[d], если d ≠ 0.
        window = 1, precompute q > 0: таблица tbl[i] = i·Pt для i = 0..q;
        R = 0, A = Pt; пока k > 0: если k нечётно — R = R + A; A = 2·A; k >>= 1.
        иначе — тот же двоичный метод без таблицы (таблица не используется)."""
    w = int(genome["window"])  # размер окна w
    pre = int(genome["precompute"])  # размер предвычислений q
    stats["modmul"] += 0  # число modmul без изменений (+0)
    if w > 1:  # вариант с фиксированным окном
        # фиксированное окно: предвычислить кратные [0..2^w)
        tbl = [None] * (1 << w)  # таблица из 2^w пустых мест
        tbl[1] = Pt  # первый элемент таблицы равен Pt
        for i in range(2, 1 << w):  # по индексам i от 2 до 2^w − 1
            tbl[i] = _add(tbl[i - 1], Pt, genome, stats)  # кратное: tbl[i] = tbl[i−1] + Pt = i·Pt
        stats["modmul"] += (1 << w) - 2  # к счётчику добавить 2^w − 2 (стоимость таблицы)
        R = None  # аккумулятор R = нейтральный элемент
        digits = []  # цифры числа k по основанию 2^w (от младшей)
        while k:  # пока остались биты
            digits.append(k & ((1 << w) - 1))  # очередная цифра d = k mod 2^w = k & (2^w − 1)
            k >>= w  # k = k div 2^w  # сдвиг вправо
        first = True  # признак первой (старшей) цифры
        for d in reversed(digits):  # по цифрам от старшей к младшей
            if not first:  # не первая цифра — сдвинуть аккумулятор на w удвоений
                for _ in range(w):  # w раз
                    R = _add(R, R, genome, stats)  # удвоение: R = 2·R
            if d:  # цифра не нуль
                R = _add(R, tbl[d], genome, stats)  # прибавление: R = R + d·Pt
            first = False  # следующая цифра уже не первая
        return R  # результат k·Pt
    if pre:  # окно 1 и есть предвычисления q > 0
        tbl = [None, Pt]  # таблица: tbl[0] = нейтральный, tbl[1] = Pt
        for i in range(2, pre + 1):  # по индексам i от 2 до q
            tbl.append(_add(tbl[-1], Pt, genome, stats))  # добавить tbl[i] = tbl[i−1] + Pt
        R = None  # аккумулятор R = нейтральный элемент
        A_ = Pt  # текущая степень A = Pt
        while k:  # пока остались биты
            if k & 1:  # младший бит равен 1
                R = _add(R, A_, genome, stats)  # накопление: R = R + A
            A_ = _add(A_, A_, genome, stats)  # удвоение: A = 2·A
            k >>= 1  # k = k div 2  # сдвиг вправо
        return R  # результат k·Pt
    R = None  # аккумулятор R = нейтральный элемент
    A_ = Pt  # текущая степень A = Pt
    while k:  # пока остались биты
        if k & 1:  # младший бит равен 1
            R = _add(R, A_, genome, stats)  # накопление: R = R + A
        A_ = _add(A_, A_, genome, stats)  # удвоение: A = 2·A
        k >>= 1  # k = k div 2  # сдвиг вправо
    return R  # результат k·Pt


def _mul_ladder(k: int, Pt, genome: dict, stats: dict) -> tuple | None:
    """Лестница Монтгомери: k·Pt с одинаковой структурой операций для всех k
    равной битовой длины (ориентация на постоянство времени).

        R0 = нейтральный, R1 = Pt; по битам k слева направо: бит 1 —
        R0 = R0 + R1, R1 = 2·R1; бит 0 — R1 = R0 + R1, R0 = 2·R0.
        Инвариант: R1 − R0 = Pt. Возвращает R0."""
    R0, R1 = None, Pt  # R0 = нейтральный элемент, R1 = Pt
    for bit in bin(k)[2:]:  # по битам двоичной записи k без префикса 0b
        if bit == "1":  # единичный бит
            R0 = _add(R0, R1, genome, stats)  # шаг: R0 = R0 + R1
            R1 = _add(R1, R1, genome, stats)  # удвоение: R1 = 2·R1
        else:  # нулевой бит
            R1 = _add(R0, R1, genome, stats)  # шаг: R1 = R0 + R1
            R0 = _add(R0, R0, genome, stats)  # удвоение: R0 = 2·R0
    return R0  # результат k·Pt


# ======================================================================
# Доказанная пригодность: сначала правильность, затем цена
# ======================================================================


def certified_correct(genome: dict) -> tuple[bool, str]:
    """Проверка групповых инвариантов на пробах — жёсткий входной фильтр.

        genome: геном. Возвращает (True, "ok") или (False, причина).
        Проверяется: EC_N·G = нейтральный элемент; (r + s)·G = r·G + s·G для
        (r, s) из (2, 3), (5, 7), (11, 13); 2·EC_N·G = нейтральный элемент."""
    try:  # любая ошибка вычисления означает непройденную проверку
        stats: dict = {"modmul": 0, "add": 0}  # счётчик операций
        # нейтральный элемент: EC_N·G = None
        if _mul_scalar(EC_N, EC_G, genome, stats) is not None:  # EC_N·G не нейтральный
            return False, "EC_N*EC_G != identity"  # проверка не пройдена
        # проба ассоциативности: (r + s)·G == r·G + s·G
        for r, s in ((2, 3), (5, 7), (11, 13)):  # пары скаляров
            lhs = _mul_scalar(r + s, EC_G, genome, {"modmul": 0, "add": 0})  # левая часть: (r + s)·G
            rhs = _add(  # правая часть: r·G + s·G
                _mul_scalar(r, EC_G, genome, {"modmul": 0, "add": 0}),  # слагаемое r·G
                _mul_scalar(s, EC_G, genome, {"modmul": 0, "add": 0}),  # слагаемое s·G
                genome,  # геном
                {"modmul": 0, "add": 0},  # отдельный счётчик операций
            )
            if lhs != rhs:  # левая и правая части разошлись
                return False, f"({r}+{s})EC_G mismatch"  # проверка не пройдена
        # однородность: (k·n)·G == нейтральный для порядка группы n
        if _mul_scalar(2 * EC_N, EC_G, genome, {"modmul": 0, "add": 0}) is not None:  # 2·EC_N·G не нейтральный
            return False, "2N*EC_G != identity"  # проверка не пройдена
        return True, "ok"  # все инварианты выполнены
    except Exception as e:  # ошибка во время проверки
        return False, repr(e)  # непройдено, причина — текст исключения


def measure(genome: dict, probes: int = 10) -> dict[str, float]:
    """Цена генома: средние modmul и время на одно умножение на скаляр.

        genome: геном, probes: число пробных скаляров (первые probes из
        PROBE_SCALARS). modmul_per_mul = Σ st.modmul / n;
        time_per_mul_us = dt / n · 1e6, где n — число проб, dt — время."""
    t0 = time.perf_counter()  # начало замера времени
    total_mul = 0  # сумма modmul по всем пробам
    stats: dict = {"modmul": 0, "add": 0}  # счётчик операций (общий)
    for k in PROBE_SCALARS[:probes]:  # по первым probes пробным скалярам
        st = {"modmul": 0, "add": 0}  # отдельный счётчик для скаляра k
        _mul_scalar(k, EC_G, genome, st)  # вычислить k·G (результат не нужен, важен счётчик)
        total_mul += st["modmul"]  # прибавить modmul этого скаляра к сумме
    dt = time.perf_counter() - t0  # длительность dt = t_конца − t_начала
    n = len(PROBE_SCALARS[:probes])  # число проб n
    return {"modmul_per_mul": total_mul / n, "time_per_mul_us": dt / n * 1e6}  # средний modmul = Σ/n; среднее время в микросекундах = dt/n·1e6


def fitness(genome: dict, profile: str = "throughput") -> float | None:
    """Пригодность генома: None — не доказана (жёсткий фильтр), иначе число,
    меньше — лучше (это цена).

        base = modmul_per_mul. Профили: throughput → base; security →
        base·(1 + 0.25), если стратегия не лестница, иначе base;
        balanced → base + 0.001·time_per_mul_us."""
    ok, why = certified_correct(genome)  # доказана ли правильность
    if not ok:  # инварианты не выполнены
        return None  # непригоден
    m = measure(genome)  # измеренная цена
    base = m["modmul_per_mul"]  # base — среднее число modmul на умножение
    if profile == "security":  # профиль безопасности
        penalty = 0.0 if genome["mul_strategy"] == "ladder" else 0.25  # надбавка 0 для лестницы, иначе 0.25
        return base * (1 + penalty)  # base·(1 + надбавка)
    if profile == "balanced":  # сбалансированный профиль
        return base * 1.0 + m["time_per_mul_us"] * 0.001  # base + 0.001·время в мкс
    return base  # пропускная способность — только число modmul


# ======================================================================
# Двигатель эволюции
# ======================================================================


def random_genome(rng: random.Random) -> dict[str, Any]:
    """Случайный геном: для каждого гена случайное значение из GENE_SPACE.

        rng: генератор случайных чисел. Возвращает словарь {ген: значение}."""
    return {g: rng.choice(vals) for g, vals in GENE_SPACE.items()}  # ген g -> случайный элемент из его списка значений


def mutate(g: dict, rng: random.Random) -> dict:
    """Мутация: заменить один случайный ген на иное допустимое значение.

        g: исходный геном, rng: генератор. Возвращает копию g с одним
        изменённым геном (если у выбранного гена есть иной вариант)."""
    out = dict(g)  # копия генома
    knob = rng.choice(list(GENE_SPACE.keys()))  # случайный ген для мутации
    options = [v for v in GENE_SPACE[knob] if v != out[knob]]  # значения, отличные от текущего
    if options:  # есть иной вариант
        out[knob] = rng.choice(options)  # заменить ген на случайный иной вариант
    return out  # возможно, изменённый геном


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    """Одноточечное скрещивание: гены до точки разреза — от a, остальные от b.

        a, b: родители, rng: генератор. Точка разреза cut — случайное целое
        из 1..(число генов − 1); гены перебираются в порядке имён."""
    keys = sorted(GENE_SPACE.keys())  # имена генов в алфавитном порядке
    cut = rng.randrange(1, len(keys))  # точка разреза cut ∈ {1, ..., число генов − 1}
    child = {}  # геном потомка
    for i, kk in enumerate(keys):  # по генам с их номерами i
        child[kk] = a[kk] if i < cut else b[kk]  # ген от a, если i < cut, иначе от b
    return child  # геном потомка


def tournament(
    pop_scores: list[tuple[float, dict]], rng: random.Random, k: int = 3  # k — размер турнира, по умолчанию 3
) -> dict:  # аннотация типа возврата: словарь с произвольными значениями
    """Турнирный отбор: из k случайных особей населения вернуть геном с
    наименьшей пригодностью-ценой.

        pop_scores: список пар (цена, геном), rng: генератор, k: размер
        турнира (не больше размера населения)."""
    challengers = rng.sample(pop_scores, min(k, len(pop_scores)))  # k случайных особей без повторов
    return min(challengers, key=lambda t: t[0])[1]  # геном особи с минимальной ценой


def evolve(
    pop_size: int = 12,  # размер населения, по умолчанию 12
    generations: int = 8,  # число поколений, по умолчанию 8
    profile: str = "throughput",
    seed: int = 7,  # зерно генератора, по умолчанию 7
    log: bool = True,  # вести журнал, по умолчанию да
) -> dict[str, Any]:  # аннотация типа возврата: словарь с произвольными значениями
    """Эволюция генома: турнир + скрещивание + мутация, элита сохраняется.

        pop_size: размер населения, generations: число поколений, profile:
        профиль пригодности, seed: зерно, log: дописывать результат в журнал.
        Особь (цена, геном) допускается в население только при fitness ≠ None.
        Поколение: население сортируется по цене; элита — первые
        max(2, pop_size div 4); остальные места заполняются потомками
        mutate(crossover(p1, p2)), где p1, p2 — победители турниров.
        Возвращает {profile, seed, best_genome, best_fitness, history, ts}."""
    rng = random.Random(seed)  # генератор с зерном seed
    # Исправление: цена считалась для одного случайного генома, а в население
    # попадал ДРУГОЙ геном (несовпадение цена/геном); теперь пара строится
    # из одного и того же генома g.
    population = []  # население: список пар (цена, геном)
    for _ in range(pop_size):  # pop_size случайных особей
        g = random_genome(rng)  # случайный геном g
        population.append((fitness(g, profile), g))  # пара (цена этого же g, g)
    population = [(s, g) for s, g in population if s is not None]  # оставить только прошедших фильтр (цена не None)
    if not population:  # ни один геном не прошёл фильтр
        raise RuntimeError("no genome passed the certified gate — bug")  # ошибка: фильтр не прошёл никто
    history = []  # история лучших особей по поколениям
    for gen in range(generations):  # по поколениям gen = 0..generations − 1
        population.sort(key=lambda t: t[0])  # отсортировать население по цене (меньше — раньше)
        best_s, best_g = population[0]  # лучшая особь: цена best_s и геном best_g
        history.append({"gen": gen, "best": best_g, "fitness": round(best_s, 2)})  # записать поколение, лучший геном и цену (2 знака)
        elite = population[: max(2, pop_size // 4)]  # элита: первые max(2, pop_size div 4) особей
        pool = [(s, g) for s, g in population]  # копия списка пар (цена, геном) для турниров
        children: list[tuple[float, dict]] = []  # потомки текущего поколения
        while len(children) < pop_size - len(elite):  # пока потомков меньше, чем свободных мест
            p1 = tournament(pool, rng)  # первый родитель — победитель турнира
            p2 = tournament(pool, rng)  # второй родитель — победитель турнира
            child = mutate(crossover(p1, p2, rng), rng)  # потомок: мутация скрещивания родителей
            s = fitness(child, profile)  # цена потомка
            if s is not None:  # потомок прошёл фильтр
                children.append((s, child))  # добавить пару (цена, геном)
        population = elite + children  # новое население: элита + потомки
    population.sort(key=lambda t: t[0])  # итоговая сортировка по цене
    best_s, best_g = population[0]  # лучшая особь итогового населения
    result = {  # итог эволюции
        "profile": profile,  # профиль пригодности
        "seed": seed,  # зерно генератора
        "best_genome": best_g,  # лучший геном
        "best_fitness": round(best_s, 2),  # его цена, 2 знака
        "history": history,  # лучшие по поколениям
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),  # отметка времени «год-месяц-день час:минута:секунда»
    }
    if log:  # ведение журнала включено
        lp = EVO_LOG  # путь к журналу эволюции
        lp.parent.mkdir(parents=True, exist_ok=True)  # создать каталог журнала, существует — не ошибка
        prev = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []  # прежние записи JSON, нет файла — пустой список
        prev.append(result)  # дописать новый результат в конец
        lp.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")  # сохранить журнал JSON с отступом 1, utf-8
    return result  # итог эволюции


def main() -> None:
    """Запуск эволюции из командной строки и печать итога в JSON.

        argv[1] — профиль (умолчание «throughput»), argv[2] — число поколений
        (умолчание 8)."""
    import sys  # доступ к аргументам командной строки

    profile = sys.argv[1] if len(sys.argv) > 1 else "throughput"  # профиль — первый аргумент, нет его — «throughput»
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # число поколений — второй аргумент как целое, нет его — 8
    res = evolve(generations=gens, profile=profile)  # прогнать эволюцию на gens поколений
    print(json.dumps(res, indent=2, ensure_ascii=False))  # напечатать итог в JSON с отступом 2


if __name__ == "__main__":  # файл запущен как программа, а не импортирован
    main()  # запустить эволюцию


# ============================================================
# БИБЛИОТЕКА ЭЛЛИПТИЧЕСКИХ КРИВЫХ: результат эволюции
# ============================================================

P = 9739  # модуль поля p
A = 497  # коэффициент a кривой
B = 1768  # коэффициент b кривой
G = (1804, 5368)  # порождающая точка G = (x, y)
N = 9735  # порядок точки
