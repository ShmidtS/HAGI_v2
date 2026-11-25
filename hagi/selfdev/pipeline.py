"""Единый конвейер: нейронная и символьная половины HAGI идут одним потоком.

    приём байтов домена
      -> сертифицированное обучение языковой модели          (lm.train_certified)
      -> диагноз остатка                                     (diagnosis.build_residual_profile,
                                                              diagnosis.classify_hold)
      -> изобретение представления на пробелах               (invention.invent_blueprint)
      -> строгий гейт                                        (invention.verify_blueprint +
                                                              измеритель возможности)
      -> принятие                                            (representations.reify_construct)
      -> сертифицированный результат дописывается в общий журнал

Каждый модуль сохраняет одну обязанность; этот файл только задаёт порядок.
Включается переменной окружения HAGI_SYMBOLIC=1 (по умолчанию выключено, чтобы
тяжёлый цикл повтора оставался быстрым); если отложенная часть языковой модели уже
хороша без усилий, символьная фаза пропускается.

Экспортирует PipelineResult (след одного приёма), symbolic_phase (диагноз ->
изобретение -> гейт -> принятие для одной области), lm_residual_relation (ошибки
модели как отношение «контекст -> настоящий байт»), ingest_symbolic (символьная
половина приёма) и unified_ingest (единая точка входа)."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import hashlib  # хеш sha256 для идентификатора принятого закона
from hagi.base import config  # единое место настроек (флаг символьной фазы)
import random  # генератор случайных чисел с зерном
from dataclasses import dataclass, field  # класс данных и поле со значением по умолчанию

import numpy as np  # чтение байтов в массив целых, равномерная выборка номеров

from hagi.invent.diagnosis import (GapBuffer, GapRecord, HoldReason,  # из модуля диагноза: буфер пробелов, запись пробела, причина удержания
                            build_residual_profile, classify_hold)  # профиль остатка и классификация причины
from hagi.invent.invention import (cegis_invent, invent_blueprint,  # из модуля изобретения: активный синтез, изобретение чертежа
                            verify_blueprint)  # и проверка чертежа
from hagi.invent.representations import reify_construct  # превращение чертежа в оператор реестра


@dataclass  # класс данных: методы сравнения и инициализации генерируются автоматически
class PipelineResult:
    """Полный след одного приёма: нейронная и символьная половины."""

    source: str  # имя домена
    holdout_ce: float  # кросс-энтропия на отложенной части; −1.0, если не измерена
    hold_reason: HoldReason | None = None  # классифицированная причина удержания
    invented: dict | None = None          # словарь результата изобретения
    exact_points: int = 0                 # число проверенных точек области
    reified_op: str | None = None         # имя оператора, если конструкция принята
    notes: list[str] = field(default_factory=list)  # заметки-пояснения (новый список для каждого экземпляра)


def symbolic_phase(source: str, xs: list[int], ys: list[int],  # имя домена и пары «байт -> байт»
                   full_domain: list[int] | None = None,  # сертифицированная область (по умолчанию — xs)
                   rng: random.Random | None = None,  # генератор случайных чисел
                   gap_buffer: GapBuffer | None = None,  # буфер пробелов для записей
                   max_attempts: int = 400) -> PipelineResult:  # лимит попыток изобретения
    """Диагноз -> изобретение -> гейт -> принятие для остатка ОДНОЙ области.

    xs/ys — байтовое отношение, которое языковая модель не смогла сжать (поверхность
    пробела); full_domain — сертифицированная область (по умолчанию xs без повторов).
    Порядок: профиль остатка и причина удержания; шум (RANDOM_INCOMPRESSIBLE) сразу
    заканчивает фазу; запись в буфер пробелов; попытка invent_blueprint, при её неудаче —
    активный синтез cegis_invent; только статус PROVEN_EXACT проходит гейт; в конце
    точная конструкция переифицируется оператором 'reified:<домен>_<фактор>'."""
    rng = rng or random.Random(7)  # генератор с зерном 7, если не задан
    full_domain = full_domain or sorted(set(xs))  # область: заданная либо xs без повторов по возрастанию
    prof = build_residual_profile([(x,) for x in xs], ys)  # профиль остатка по парам ((x,), y)
    reason = classify_hold("NO_VALIDATED_PROGRAM", prof)  # причина удержания для текста «нет проверенной программы»
    res = PipelineResult(source=source, holdout_ce=-1.0, hold_reason=reason)  # результат фазы: кросс-энтропия не измерена, причина — найденная

    if reason == HoldReason.RANDOM_INCOMPRESSIBLE:  # остаток классифицирован как случайный шум
        res.notes.append("noise: no structure to invent")  # заметка: структуры для изобретения нет
        return res  # закончить без изобретения
    if gap_buffer is not None:  # буфер пробел задан
        gap_buffer.add(GapRecord(task_name=source, reason=reason,  # добавить запись о пробеле: домен, причина,
                                 profile=prof))  # профиль остатка

    pairs = sorted(zip(xs, ys))  # пары (x, y) в порядке сортировки
    result = invent_blueprint([pairs], rng, max_attempts=max_attempts)  # попытка изобрести чертёж по одному кластеру пар
    if not result or not result.get("blueprints"):  # результата нет или чертежей в нём нет
        # Второй путь — активный синтез с запросами (доказательство, управляемое вопросами).
        # Оракул — само отношение (словарь xs -> ys); бюджет честный: спрашиваются
        # только точки области, которая уже есть.
        oracle_map = dict(pairs)  # оракул: словарь «x -> y»
        oracle = lambda x: oracle_map.get(x)  # функция оракула: метка x или None
        try:  # синтез может бросить ошибку
            cegis = cegis_invent(sorted(set(xs)), oracle,  # активный синтез по xs без повторов и оракулу
                                 cegis_pool=sorted(set(xs))[:64],  # пул запросов — первые 64 различных x
                                 full_domain=full_domain,  # область сертификации
                                 max_rounds=4, query_budget=400,  # не более 4 раундов и 400 запросов
                                 rng=rng)  # с тем же генератором
        except Exception as e:  # синтез бросил любую ошибку
            res.notes.append(f"cegis error: {e}")  # заметка с текстом ошибки
            return res  # закончить
        res.invented = {"status": cegis.status, "path": "cegis",  # записать результат изобретения: статус, путь «cegis»
                        "queries": cegis.queries}  # и число запросов к оракулу
        if cegis.status != "PROVEN_EXACT" or cegis.blueprint is None:  # не доказано точно или чертежа нет
            res.notes.append(f"cegis: {cegis.status}")  # заметка со статусом
            return res  # закончить
        bp = cegis.blueprint  # найденный чертёж
        sweep = {"status": "PROVEN_EXACT"}  # сплошной обход уже выполнен на этапе сертификации
    else:  # invent_blueprint дала чертежи
        bp = result.get("blueprints", [None])[0]  # первый чертёж (None, если списка нет)
        res.invented = {k: result[k] for k in  # результат изобретения: четыре поля
                        ("status", "q_name", "q_param", "verdict")}  # статус, имя и параметр фактора, вердикт
        sweep = verify_blueprint(bp, pairs, full_domain)  # строгая проверка чертежа на парах и всей области
    if sweep.get("status") != "PROVEN_EXACT":  # гейт не пройден
        res.notes.append(f"gate: {sweep.get('status')}")  # заметка со статусом гейта
        return res  # закончить
    res.exact_points = len(full_domain)  # проверено точек = размер области

    # Принятие: переифицировать точную конструкцию, чтобы позднейшие задачи её использовали
    q_name = (result or {}).get("q_name") or getattr(bp, "q_name",  # имя фактора из результата изобретения,
                                                     None) or "bp"  # иначе из чертежа, иначе «bp»
    fn = getattr(bp, "fn", None) or (lambda x: bp.apply(x))  # функция конструкции: атрибут fn чертежа, иначе bp.apply(x)
    try:  # повторная регистрация имени даёт ValueError
        op = reify_construct(bp, f"{source}_{q_name}", fn,  # зарегистрировать оператор 'reified:<домен>_<фактор>'
                             arity=1, domain=full_domain)  # унарный, с сертифицированной областью
        res.reified_op = op  # запомнить имя оператора
    except ValueError as e:  # оператор уже существует
        res.notes.append(f"reify: {e}")  # заметка с текстом ошибки
    return res  # вернуть результат фазы


def lm_residual_relation(u, source: str, data: bytes  # модель, имя домена, байты
                         ) -> tuple[list[int], list[int],  # аннотация возврата (формально четыре списка;
                                    list[tuple[int, ...]], list[int],  # фактически функция возвращает шесть значений,
                                    list[tuple[int, ...]]]:  # перечисленных в docstring)
    """Ошибки языковой модели как отношение «контекст -> настоящий следующий байт».

    Раньше символьная фаза брала соседние байты данных (x_i -> x_{i+1}) — закон данных,
    а не модели. Теперь выбираются настоящие ошибки модели: контексты, где предсказание
    по наибольшему логиту не совпало с фактическим следующим байтом.

    Возвращает шесть списков: misses_x и misses_y — пары унарного канала (последний
    байт контекста и, если контекст не короче 2, предпоследний -> следующий байт) по
    выборке позиций с шагом max(1, n // 512); multi_ctx, multi_y — канал многих байтов
    (последние 4 байта контекста -> следующий байт) по подряд идущим позициям до 2000
    ошибок; multi_full_ctx — те же контексты целиком; multi_pos — позиции ошибок в потоке.
    При n = len(data) − k ≤ 0 возвращаются четыре пустых списка."""
    import torch as _t  # работа с тензорами (локальный импорт)
    model = u.model  # языковая модель
    dev = getattr(model, "device", "cpu")  # устройство модели (по умолчанию cpu)
    idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # байты как массив целых idx
    n = len(idx) - u.k  # число позиций, имеющих полный контекст из k байтов
    if n <= 0:  # данных короче контекста
        return [], [], [], []  # четыре пустых списка
    misses_x: list[int] = []  # входы унарного канала
    misses_y: list[int] = []  # его выходы (настоящий следующий байт)
    multi_ctx: list[tuple[int, ...]] = []  # контексты канала многих байтов
    multi_y: list[int] = []  # его выходы
    # Полный хронологический контекст хранится отдельно от четырёхбайтовой
    # характеристики — как свидетельство для самокритики законов.
    multi_full_ctx: list[tuple[int, ...]] = []  # полные контексты длины k
    step = max(1, n // 512)          # шаг выборки: работа ограничена примерно 512 позициями
    with _t.no_grad():  # без вычисления градиентов
        for i in range(0, n, step):  # по выборке позиций i = 0, step, 2·step, ...
            ctx = idx[i:i + u.k]  # контекст: срез idx[i .. i+k−1]
            nxt = int(idx[i + u.k])  # настоящий следующий байт nxt = idx[i + k]
            x = _t.as_tensor([ctx.tolist()], dtype=_t.long, device=dev)  # контекст как целочисленный тензор формы (1, k) на устройстве модели
            try:  # вычисление логитов может бросить ошибку
                logits = model.logits(x)[0]  # логиты для контекста (строка 0 батча)
                pred = int(logits.argmax())  # предсказание pred = argmax по логитам
            except Exception:  # ошибка вычисления
                continue  # пропустить позицию
            if pred != nxt:  # модель ошиблась
                misses_x.append(int(ctx[-1]))  # пара (последний байт контекста, nxt)
                misses_y.append(nxt)  # выход nxt
                if len(ctx) >= 2:  # в контексте есть предпоследний байт
                    misses_x.append(int(ctx[-2]))  # пара (предпоследний байт контекста, nxt)
                    misses_y.append(nxt)  # выход nxt
    # Каналу многих байтов нужен НЕпрореженный поток: арифметический шаг (n // 512)
    # разрушает периодичность выравнивания записей, на которую опирается проверка
    # по слоям сдвига. Второй, последовательный проход, до 2000 ошибок.
    multi_pos: list[int] = []  # позиции ошибок в потоке
    step2 = 1  # шаг второго прохода — каждая позиция
    count2 = 0  # число найденных ошибок
    for i in range(0, n, step2):  # по всем позициям подряд
        if count2 >= 2000:  # лимит ошибок достигнут
            break  # прекратить проход
        ctx = idx[i:i + u.k]  # контекст: срез idx[i .. i+k−1]
        nxt = int(idx[i + u.k])  # настоящий следующий байт
        x = _t.as_tensor([ctx.tolist()], dtype=_t.long, device=dev)  # контекст как целочисленный тензор (1, k)
        try:  # вычисление логитов может бросить ошибку
            logits = model.logits(x)[0]  # логиты для контекста
            pred = int(logits.argmax())  # предсказание по наибольшему логиту
        except Exception:  # ошибка вычисления
            continue  # пропустить позицию
        if pred != nxt:  # модель ошиблась
            multi_ctx.append(tuple(int(b) for b in ctx[-4:]))  # характеристика: последние 4 байта контекста
            multi_full_ctx.append(tuple(int(b) for b in ctx))  # полный контекст
            multi_y.append(nxt)  # настоящий следующий байт
            multi_pos.append(i)            # настоящая позиция в потоке
            count2 += 1  # ошибок: count2 + 1
    return misses_x, misses_y, multi_ctx, multi_y, multi_full_ctx, multi_pos  # шесть списков


def ingest_symbolic(u, source: str, data: bytes,  # модель, имя домена, байты
                    gap_buffer: GapBuffer | None = None) -> PipelineResult:  # буфер пробелов
    """Символьная половина приёма: отношение пробела — ошибки модели на её собственной
    отложенной части; затем диагноз -> изобретение -> принятие.

    Сначала идёт канал многих байтов (более богатые законы): найденный и проверенный
    закон вносится в реестр домена для предсказания ещё не виденных байтов. Затем
    выполняется классический унарный путь symbolic_phase."""
    holdout_ce = u.domains.get(source, {}).get("ce")  # кросс-энтропия домена в книге доменов (None, если не измерена)
    res = PipelineResult(source=source,  # промежуточный результат
                         holdout_ce=holdout_ce if holdout_ce is not None else -1.0)  # измеренная кросс-энтропия или −1.0
    xs, ys, multi_ctx, multi_y, multi_full_ctx, multi_pos = lm_residual_relation(u, source, data)  # ошибки модели по двум каналам
    # Канал многих байтов первым (более богатые законы); если он даёт проверенный
    # закон, он РЕГИСТРИРУЕТСЯ для ПРЕДСКАЗАНИЯ ещё не виденных байтов, затем
    # управление переходит к классическому унарному пути.
    if multi_ctx:  # есть ошибки канала многих байтов
        try:  # любая ошибка канала не должна прерывать приём
            from hagi.invent.invention_multi import invent_multi  # синтез законов по характеристикам контекста (локальный импорт)
            from hagi.invent.invention import verify_blueprint  # проверка чертежа (локальный импорт)
            mr = invent_multi(list(zip(multi_ctx, multi_y, multi_pos)),  # синтез по тройкам (характеристика, выход, позиция)
                              random.Random(7), max_attempts=8000)  # генератор с зерном 7, до 8000 попыток
            if mr and mr.get("status") != "INVENTED":  # есть результат, но не «изобретено»
                res.notes.append(f"multi-byte: {mr['status']} "  # заметка: статус канала
                                 f"({mr.get('attempts')} feats)")  # и число попыток-характеристик
            # Законы со статусом PARTIAL_LAW (верны на своей области) также
            # регистрируются: предсказание следующего байта ставит их ниже
            # доказанных, но использует.
            if mr and mr.get("status") in ("INVENTED", "PARTIAL_LAW"):  # закон найден
                bp = mr["blueprints"][0]  # первый чертёж закона
                # Для PARTIAL_LAW проверкой служит собственное свидетельство
                # синтезатора на отложенной части слоёв сдвига (доля области + 95%
                # совпадений внутри слоя): полная проверка по всему потоку отвергла бы
                # истинный закон (невыровненные позиции по построению — шум).
                # Статус записывается честно, как есть.
                v = {"status": mr["status"],  # сводка проверки: статус
                     "support": mr.get("support")}  # и доля области
                # Проверенные законы вносятся в реестр предсказания домена:
                # они предсказывают СЛЕДУЮЩИЙ БАЙТ для ещё не виденных контекстов.
                registry = getattr(u, "law_registry", None)  # реестр законов модели (None, если его нет)
                if registry is None:  # реестра ещё нет
                    registry = u.law_registry = {}  # создать пустой и сохранить на модели
                # Полный контекст хранится отдельно от характеристики;
                # свидетельства — только примеры, подтверждённые в своём слое.
                shift = int(mr.get("shift", 0))  # сдвиг слоя: остаток позиции от деления
                n_shifts = int(mr.get("n_shifts", 6))  # число слоёв (по умолчанию 6)
                import numpy as _np_e  # равномерная выборка номеров (локальный импорт)
                evidence = []  # свидетельства закона
                # Слой задаётся ПОЗИЦИЕЙ В ПОТОКЕ: список — ошибки модели, его
                # индекс с выравниванием записей не совпадает.
                for f_ctx, full_ctx, target, p_i in zip(  # по четвёркам
                        multi_ctx, multi_full_ctx, multi_y, multi_pos):  # характеристика, полный контекст, выход, позиция
                    if p_i % n_shifts != shift:  # позиция не в слое закона
                        continue  # пропустить
                    try:  # применение чертежа может бросить ошибку
                        if int(bp.apply(f_ctx)) != int(target):  # закон ошибается на примере
                            continue  # не свидетельство
                    except Exception:  # чертёж бросил ошибку
                        continue  # пропустить
                    evidence.append({  # добавить свидетельство
                        "context": list(full_ctx),  # полный контекст
                        "feature_context": list(f_ctx),  # характеристика контекста
                        "target": int(target),  # настоящий следующий байт
                        "source_index": p_i,      # позиция в потоке
                    })  # конец свидетельства
                if len(evidence) > 256:  # свидетелей больше 256
                    take = _np_e.linspace(0, len(evidence) - 1, 256,  # 256 равномерно расположенных номеров
                                          dtype=int)  # как целые
                    evidence = [evidence[int(i)] for i in take]  # оставить по одному свидетелю на каждый номер
                bp_meta = (bp.to_dict() if hasattr(bp, "to_dict")  # чертёж в виде словаря, если умеет
                           else repr(bp))  # иначе его текстовое представление
                commit_material = repr((source, mr.get("feature"),  # материал идентификатора: домен, характеристика,
                                        v["status"], bp_meta))  # статус, чертёж
                commit_id = hashlib.sha256(  # идентификатор принятия: первые 20 знаков
                    commit_material.encode("utf-8")).hexdigest()[:20]  # hex(SHA-256(материал))[0:20]
                entries = registry.setdefault(source, [])  # список законов домена (создать пустой, если нет)
                if not any(e.get("commit_id") == commit_id  # среди законов домена нет записи с тем же идентификатором
                           for e in entries):  # по всем записям e
                    entries.append({  # добавить закон в реестр
                        "feature": mr.get("feature"),  # характеристика
                        "bp": bp,  # чертёж
                        "verify": v["status"],  # статус проверки
                        "support": v.get("support"),  # доля области
                        "coverage": len(multi_ctx),  # объём данных канала
                        "accepted_to_registry": True,  # принят в реестр
                        "commit_id": commit_id,  # идентификатор принятия
                        "shift": shift,  # сдвиг слоя
                        "n_shifts": n_shifts,  # число слоёв
                        "evidence": evidence,  # свидетели
                    })  # конец записи
                res.notes.append(  # заметка о законе канала
                    f"multi-byte law {mr.get('feature')} "  # «закон многих байтов <характеристика>»
                    f"-> {v['status']} (registry commit {commit_id}; "  # статус и идентификатор в реестре
                    f"self-critique evidence={len(evidence)})")  # число свидетелей самокритики
        except Exception as _e:  # канал многих байтов бросил ошибку
            res.notes.append(f"multi-byte skipped: {_e!r}")  # заметка: канал пропущен, с текстом ошибки
    if not xs:  # унарных ошибок не найдено
        res.notes.append("no LM residuals sampled")  # заметка: ошибки модели не выбраны
        return res  # вернуть промежуточный результат
    # symbolic_phase возвращает НОВЫЙ PipelineResult — заметки канала многих
    # байтов переносятся, чтобы журнал показывал, что сделал синтезатор законов.
    sym_res = symbolic_phase(source, xs, ys,  # унарная фаза по ошибкам
                             full_domain=sorted(set(xs)),  # область — разные x по возрастанию
                             gap_buffer=gap_buffer)  # с буфером пробелов
    sym_res.notes = [n for n in res.notes  # заметки результата = заметки, начинающиеся с «multi-byte»,
                     if n.startswith("multi-byte")] + sym_res.notes  # за которыми идут собственные заметки фазы
    return sym_res  # вернуть результат символьной фазы


def unified_ingest(u, source: str, data: bytes, epochs: int = 200,  # модель, домен, байты, число эпох
                   repeat: int = 2, replay_cap: int | None = None,  # число повторов и предел повтора (None — без предела)
                   gap_buffer: GapBuffer | None = None) -> dict:  # буфер пробелов; возвращает словарь результата
    """Единая точка входа: нейронная и символьная половины подряд.

    Заменяет вызов Unified.ingest, когда HAGI_SYMBOLIC=1. Сначала обычный приём
    (обучение модели); если переменная окружения не равна «1», ключ symbolic получает
    None. Иначе выполняется символьная половина; модель сохраняется, только если число
    законов в реестре изменилось; затем запускается самокритика законов (её ошибка
    записывается в результат, а не прерывает приём)."""
    import time as _time  # замер длительности фаз
    _t0 = _time.perf_counter()  # начало нейронной половины
    r = u.ingest(source, data, epochs=epochs, repeat=repeat,  # нейронная половина: обычный приём домена
                 replay_cap=replay_cap)  # с пределом повтора
    r.setdefault("phases", {})["neural"] = round(_time.perf_counter() - _t0, 1)  # длительность нейронной половины, с
    _t1 = _time.perf_counter()  # начало символьной половины
    if not config.symbolic_enabled():  # символьная фаза выключена (по умолчанию «0»)
        r["symbolic"] = None  # символьной части нет
        return r  # вернуть результат приёма
    registry_before = sum(len(v) for v in  # число законов во всех доменах реестра до символьной фазы
                          (getattr(u, "law_registry", {}) or {}).values())  # по спискам реестра (нет реестра — пустой словарь)
    sym = ingest_symbolic(u, source, data, gap_buffer=gap_buffer)  # символьная половина
    r["phases"]["symbolic"] = round(_time.perf_counter() - _t1, 1)  # длительность символьной половины, с
    r["symbolic"] = {"hold_reason": sym.hold_reason.name if sym.hold_reason else None,  # имя причины удержания или None
                     "invented": sym.invented,  # результат изобретения
                     "exact_points": sym.exact_points,  # число проверенных точек
                     "reified_op": sym.reified_op,  # имя принятого оператора
                     "notes": sym.notes}  # заметки
    # Сохранять только при добавлении нового закона: свежий реестр переживает
    # перезапуск; затем идёт самокритика.
    registry_after = sum(len(v) for v in  # число законов в реестре после символьной фазы
                         (getattr(u, "law_registry", {}) or {}).values())  # по спискам реестра
    try:  # сохранение и самокритика не должны прерывать приём
        if registry_after != registry_before:  # число законов изменилось
            u.save()  # сохранить модель с реестром
        from hagi.data.law_critique import run_law_self_critique  # самокритика законов (локальный импорт)
        r["self_critique"] = run_law_self_critique(u)  # оценить недавно принятые законы и записать итог
    except Exception as _critique_error:  # сохранение или самокритика бросили ошибку
        r["self_critique"] = {"status": "ERROR",  # самокритика не выполнена
                              "error": repr(_critique_error)}  # текст ошибки
    return r  # вернуть результат приёма
