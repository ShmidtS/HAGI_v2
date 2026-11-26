"""Словарный закон «контекст -> байт»: синтез, отказ вне таблицы, сохранение и загрузка."""
import random  # генератор случайных чисел

import pytest  # фикстура tmp_path

from hagi.invent.invention import ConstructBlueprint, _payload_bits, basis_dict  # чертёж, стоимость нагрузки, словарный базис
from hagi.invent.invention_multi import _kind, _vfeat, DICT_MOD, enumerate_features, feature_fn, invent_multi  # части синтезатора


def _pairs(seed=0, n=3000):  # зерно, число пар
    """Пары «контекст 4 байта -> байт» из потока, похожего на исходный код: токены
    следуют в случайном порядке, а за суффиксом "def " всегда идёт "f" (закон-словарь)."""
    r = random.Random(seed)  # генератор с зерном seed
    toks = ["self.", "x = ", "return ", "def f", "(a, b):", "import os", "    "]  # токены потока; "def f" держит закон "def " -> "f"
    data = bytearray()  # поток байтов
    while len(data) < n + 4:  # до нужной длины
        data += r.choice(toks).encode("utf-8")  # дописать случайный токен
    data = data[:n + 4]  # обрезать ровно до n + 4 байтов
    return [(tuple(data[i:i + 4]), data[i + 4], i) for i in range(n)]  # тройки (контекст, следующий байт, позиция) подряд


def test_dict_law_is_found_and_abstains_outside_table():
    """Словарный закон находится (базис DICT), отвечает верно в таблице и не отвечает вне её."""
    r = invent_multi(_pairs(), random.Random(0))  # синтез по тройкам
    assert r["status"] == "PARTIAL_LAW"  # найден частичный закон
    bp = r["blueprints"][0]  # чертёж закона
    assert bp.basis == "DICT" and bp.payload["kind"] == "DICT"  # базис — словарная таблица
    assert bp.apply(tuple(b"def ")) == ord("f")  # пример из словаря: ответ верен
    with pytest.raises(KeyError):  # вне таблицы — ошибка, а не выдуманный байт
        bp.apply((1, 2, 3, 4))  # контекст, которого в словаре нет


def test_dict_feature_is_vectorised_exactly_and_roundtrips():
    """Векторизованный словарный хеш равен поэлементной функции; feature_fn восстанавливает её по имени."""
    import numpy as np  # массивы
    X = np.random.default_rng(1).integers(0, 256, (2000, 4), dtype=np.uint8)  # 2000 случайных контекстов
    feats = [f for f in enumerate_features(4) if f[0].startswith("id")]  # словарные характеристики
    assert [f[0] for f in feats] == ["id4", "id3", "id2"]  # длины суффиксов 4, 3, 2 — первыми в списке
    for name, param, fn in feats:  # по словарным характеристикам
        got = _vfeat(_kind(name), param, X)  # векторизованные значения
        ref = np.fromiter((fn(tuple(int(v) for v in row)) for row in X), dtype=np.int64, count=len(X))  # эталон — поэлементно
        assert not (got != ref).any(), name  # совпадают бит в бит
        row5 = tuple(int(v) for v in X[5])  # строка 5 как кортеж целых
        assert feature_fn(name, param)(row5) == fn(row5)  # восстановленная функция даёт то же значение
    assert _payload_bits("DICT", basis_dict([(1, 2), (3, 4)]), 0) == 48  # 2 записи по 24 бита


def test_registry_survives_save_load(tmp_path, monkeypatch):  # каталог состояния, подмена путей
    """Не аффинный закон реестра переживает сохранение и загрузку и продолжает отвечать."""
    from hagi.selfdev import unified as U  # модуль единой модели
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # состояние пишется в каталог теста, а не в рабочий
    r = invent_multi(_pairs(), random.Random(0))  # синтез словарного закона
    u = U.Unified(k=4)  # модель с окном 4 байта
    u.law_registry = {"dom": [{"feature": r["feature"], "bp": r["blueprints"][0],  # реестр с одним словарным законом: характеристика и чертёж
                               "verify": "PARTIAL_LAW", "support": r["support"],  # статус и опора
                               "shift": r["shift"], "n_shifts": r["n_shifts"]}]}  # слой закона; конец записи
    u.save()  # сохранить состояние вместе с реестром
    v = U.Unified.load().law_registry["dom"][0]  # загрузить состояние и взять закон домена
    assert v["feature"] == r["feature"] and v["bp"].payload == r["blueprints"][0].payload  # характеристика и таблица те же
    assert v["bp"].apply(tuple(b"def ")) == u.law_registry["dom"][0]["bp"].apply(tuple(b"def "))  # загруженный закон отвечает так же


def test_noise_has_no_dict_law():
    """На чистом шуме словарный закон не изобретается: NO_LAW."""
    r0 = random.Random(5)  # генератор пар
    pairs = [(tuple(r0.randbytes(4)), r0.randbytes(1)[0], i) for i in range(2000)]  # 2000 случайных троек
    assert invent_multi(pairs, random.Random(5))["status"] == "NO_LAW"  # закона нет
