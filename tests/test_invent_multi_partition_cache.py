"""Кеш отбракованных слоёв в invent_multi: слои с одним разбиением пар отбрасываются одинаково, а результат поиска от кеша не зависит."""
import random  # генератор случайных чисел

import numpy as np  # численные массивы

import hagi.invent.invention_multi as im  # модуль синтеза многобайтовых законов


def _noise_stream(n, k, seed):
    """Случайный поток из n пар (контекст из k байтов, ответ, позиция)."""
    r = random.Random(seed)  # генератор потока
    data = [r.randrange(256) for _ in range(n + k)]  # случайные байты
    return [(tuple(data[i:i + k]), data[i + k], i) for i in range(n)]  # пары (контекст, ответ, позиция)


def _law_stream(n, period, seed):
    """Поток с законом y = a + 2b + 7 на фазе 0 периода period; остальные позиции — шум, 5% ответов закона испорчены."""
    r = random.Random(seed)  # генератор потока
    data = [r.randrange(256) for _ in range(n + 3)]  # случайные байты
    for i in range(3, n + 3):  # по позициям целей
        if (i - 3) % period == 0:  # позиция выровнена по записи
            data[i] = (data[i - 3] + 2 * data[i - 2] + 7) % 256  # значение по закону
            if r.random() < 0.05:  # с вероятностью 5%
                data[i] = r.randrange(256)  # ответ испорчен
    return [(tuple(data[i:i + 3]), data[i + 3], i) for i in range(n)]  # пары (контекст, ответ, позиция)


def _grid_stream():
    """Поток, где позиции контекста имеют одинаковые гистограммы значений, но разные разбиения; ответ — x0 с редким шумом."""
    r = random.Random(3)  # генератор шума ответов
    pairs = []  # пары потока
    for i in range(3072):  # по 3072 позициям: каждое значение каждой позиции встречается ровно 192 раза
        x0, x1, x2 = i % 16, (i // 16) % 16, (i * 7 + i // 16) % 16  # три позиции контекста с равными гистограммами
        y = x0 if r.random() > 0.05 else r.randrange(256)  # ответ равен x0, в 5% случаев — шум
        pairs.append(((x0, x1, x2), y, i))  # пара (контекст, ответ, позиция)
    return pairs  # поток пар


def _first_seen(c):
    """Таблица первых вхождений значений характеристики, как её строит invent_multi."""
    rep = np.full(int(c.max()) + 1, -1, dtype=np.int64)  # rep[v] = −1 для v = 0..max(c)
    rep[c[::-1]] = np.arange(len(c))[::-1]  # запись в обратном порядке: остаётся первое вхождение
    return rep  # таблица первых вхождений


def _norm(res):
    """Результат поиска без функций: поля и чертежи (имя, параметр, базис, нагрузка, биты)."""
    if res is None:  # поиск вернул None
        return None  # сравнивать нечего
    out = {k: v for k, v in res.items() if k != "blueprints"}  # все поля, кроме чертежей
    out["bps"] = [(b.q_name, b.q_param, b.basis, b.payload, b.meta_bits, b.depth, b.total_bits)  # поля чертежей без функции
                  for b in res.get("blueprints", [])]  # по чертежам результата
    return out  # нормализованный результат


def _check_group_stats(pairs):
    """Для потока pairs: у характеристик с одной подписью разбиения итоги слоёв совпадают; вернуть наибольшее число значений слоя."""
    X = np.asarray([p[0] for p in pairs], dtype=np.int64)  # матрица контекстов
    y = np.asarray([p[1] for p in pairs], dtype=np.int64)  # ответы
    pos = np.arange(len(pairs))  # позиции пар
    groups = {}  # подпись разбиения -> список (имя, значения характеристики)
    for name, param, _fn in im.enumerate_features(3):  # по всем характеристикам
        if name.startswith("id"):  # словарные характеристики кеш не использует
            continue  # пропустить
        c = im._vfeat(im._kind(name), param, X)  # значения характеристики на всех парах
        groups.setdefault(im._partition_key(c, _first_seen(c)), []).append((name, c))  # сгруппировать по подписи
    multi = [g for g in groups.values() if len(g) > 1]  # группы с несколькими характеристиками
    assert sum(len(g) - 1 for g in multi) > 500  # повторов разбиения много: кеш имеет смысл
    most_rows = 0  # наибольшее число значимых значений на слое
    for g in multi:  # по группам
        seen = set()  # различные итоги слоёв внутри группы
        for _name, c in (g[0], g[-1]):  # по первой и последней характеристике группы
            stats = []  # итоги слоёв одной характеристики
            for P in (1, 2, 3):  # по периодам
                for s in range(P):  # по фазам
                    rows, support = im._majVotes(c, y, np.nonzero(pos % P == s)[0])  # голоса и опора слоя
                    stats.append((len(rows), support))  # число значений и опора
                    most_rows = max(most_rows, len(rows))  # обновить наибольшее число значений
            seen.add(tuple(stats))  # запомнить итоги характеристики
        assert len(seen) == 1, g[0][0]  # у обеих характеристик итоги одинаковы
    return most_rows  # наибольшее число значимых значений на слое


def test_equal_partition_means_equal_layer_stats():
    """У характеристик с одной подписью разбиения число значимых значений и опора на каждом слое совпадают бит в бит."""
    assert _check_group_stats(_law_stream(3000, 1, 5)) >= 24  # поток с законом: есть слои, проходящие порог числа значений
    assert _check_group_stats(_grid_stream()) >= 16  # сетка: слабая подпись (гистограмма) смешала бы разные разбиения и была бы поймана


def test_partition_key_separates_partitions():
    """Сдвиг, смена знака и умножение на нечётное число разбиения не меняют; умножение на 2 и другая позиция — меняют."""
    pairs = _noise_stream(1200, 3, 9)  # случайный поток
    X = np.asarray([p[0] for p in pairs], dtype=np.int64)  # матрица контекстов

    def key(coefs, bias):  # подпись разбиения аффинной характеристики с коэффициентами coefs и смещением bias
        c = im._vfeat("aff", (coefs, bias), X)  # значения аффинной характеристики
        return im._partition_key(c, _first_seen(c))  # подпись разбиения

    assert key((1, 0, 0), 0) == key((1, 0, 0), 97)  # сдвиг по модулю 256 — перестановка значений
    assert key((1, 0, 0), 0) == key((-1, 0, 0), 0)  # смена знака — перестановка значений
    assert key((1, 0, 0), 0) == key((3, 0, 0), 5)  # умножение на нечётное число и сдвиг — перестановка значений
    assert key((1, 0, 0), 0) != key((2, 0, 0), 0)  # умножение на 2 склеивает значения
    assert key((1, 0, 0), 0) != key((0, 1, 0), 0)  # другая позиция контекста — другое разбиение


def test_cache_does_not_change_search_result(monkeypatch):
    """Результат поиска и состояние генератора те же, что без кеша (подпись у каждой характеристики уникальна)."""
    cases = [  # входы: шум и поток с законом периода 3
        (_noise_stream(900, 3, 1), {"max_attempts": 500}),  # шум: закона нет
        (_law_stream(1800, 3, 2), {"max_attempts": 400}),  # закон периода 3: проходит порог и межслойную проверку
        (_grid_stream(), {"max_attempts": 300}),  # сетка: одинаковые гистограммы при разных разбиениях
    ]
    statuses = []  # статусы результатов
    for pairs, kw in cases:  # по входам
        r1, r2 = random.Random(11), random.Random(11)  # два одинаковых генератора
        cached = im.invent_multi(pairs, r1, **kw)  # поиск с кешем
        with monkeypatch.context() as m:  # подмена только на время поиска без кеша
            counter = iter(range(10 ** 9))  # счётчик уникальных подписей
            m.setattr(im, "_partition_key", lambda c, rep: next(counter).to_bytes(8, "big"))  # у каждой характеристики своя подпись
            plain = im.invent_multi(pairs, r2, **kw)  # поиск без кеша
        assert _norm(cached) == _norm(plain)  # результат тот же
        assert r1.getstate() == r2.getstate()  # поток случайных чисел израсходован так же
        statuses.append(cached["status"])  # запомнить статус
    assert "PARTIAL_LAW" in statuses  # закон найден хотя бы раз: ветка выживших слоёв задействована
