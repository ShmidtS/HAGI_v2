"""Тесты байтовой модели HNetByteLM (архитектура «hnet») через общий интерфейс make_lm.

    Свойства:
      1. Фабрика make_lm("hnet") даёт модель с ARCH = "hnet", длиной контекста
         k = 64, размерами по умолчанию d = 64, h = 128 и ровно 105 088 параметров;
         у варианта d = 24, h = 48 параметров 28 048. logits для партии из n
         контекстов имеет форму (n, 256).
      2. Модель обучаема: после двух эпох CE на отложенных данных меньше
         log(256) − 0,5, где log(256) ≈ 5,545 — CE равномерного распределения.
      3. Unified с ARCH = "hnet" сохраняется и загружается бит в бит:
         max |θ_сохранённые − θ_загруженные| = 0.
      4. Unified() на пустой контрольной точке читает книгу доменов из
         unified_ledger.json, а не начинает «с чистого листа»."""
import numpy as np  # log(256) — CE равномерного распределения по байтам
import pytest  # инструменты тестирования


def test_hnet_factory_shapes_and_params():
    """Фабрика hnet: ARCH, k, форма logits (1, 256), размеры d/h и число параметров 105 088 и 28 048."""
    from hagi.models.lm import make_lm  # фабрика моделей по имени архитектуры
    m = make_lm("hnet", seed=1)  # модель hnet с размерами по умолчанию
    assert m.ARCH == "hnet" and m.k == 64  # имя архитектуры hnet, длина контекста 64
    out = m.logits(m.t.as_tensor([[0] * 64], dtype=m.t.long))  # logits для одного контекста из 64 нулевых байтов
    assert tuple(out.shape) == (1, 256)  # форма (число контекстов = 1, число классов = 256)
    # по умолчанию d = 64, h = 128 и 105 088 параметров; у варианта d = 24
    # параметров 28 048
    assert (m.dim, m.hidden, sum(p.numel() for p in m.params())) == (64, 128, 105088)  # d = 64, h = 128, Σ параметров = 105 088
    small = make_lm("hnet", dim=24, hidden=48, seed=1)  # уменьшенный вариант d = 24, h = 48
    assert sum(p.numel() for p in small.params()) == 28048  # у уменьшенного варианта Σ параметров = 28 048
    assert make_lm("rope", seed=1).ARCH == "rope"  # фабрика rope возвращает модель с ARCH = "rope"


def test_hnet_trains_and_beats_its_init_on_a_real_byte_stream():
    """После 2 эпох обучения best_holdout_ce < log(256) − 0,5: модель лучше своей случайной инициализации."""
    from hagi.models.lm import make_lm  # фабрика моделей
    data = bytes(range(0, 256, 3)) * 60  # поток байтов 0,3,6,…,255, повторённый 60 раз
    m = make_lm("hnet", seed=1)  # модель hnet по умолчанию
    r = m.train_guarded(data, epochs=2)  # обучить 2 эпохи с гейтом
    assert r["best_holdout_ce"] < np.log(256) - 0.5  # лучшая CE отложенных данных < log(256) − 0,5


def test_hnet_unified_roundtrip_save_load_and_ingest(tmp_path, monkeypatch):
    """Unified с ARCH = hnet: ingest даёт holdout_ce < log(256); после save/load веса совпадают точно (max |Δ| = 0)."""
    from hagi.selfdev import unified as U  # единая модель
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setattr(U.Unified, "ARCH", "hnet")  # архитектура по умолчанию := hnet
    u = U.Unified()  # новая единая модель
    assert type(u.model).__name__ == "HNetByteLM"  # модель имеет класс HNetByteLM
    (tmp_path / "domains").mkdir()  # каталог файлов доменов
    data = bytes(range(0, 256, 5)) * 900  # поток байтов 0,5,10,…,250, повторённый 900 раз
    (tmp_path / "domains" / "a.bin").write_bytes(data)  # сохранить поток как файл домена a
    r = u.ingest("a", data, epochs=1, repeat=1)  # проглотить домен a: 1 эпоха, 1 повтор
    assert r["holdout_ce"] < np.log(256)  # CE на отложенных данных < log(256)
    u.save()  # сохранить модель и книгу доменов
    v = U.Unified.load()  # загрузить копию
    assert type(v.model).__name__ == "HNetByteLM" and "a" in v.domains  # загружена модель HNetByteLM, домен a записан в книге
    for a, b in zip(u.model.params(), v.model.params()):  # по парам (исходный, загруженный) параметров
        assert float((a.detach().cpu() - b.detach().cpu()).abs().max()) == 0.0  # максимальный модуль разности |a − b| = 0


def test_fresh_model_keeps_the_domain_book(tmp_path, monkeypatch):
    """Unified() на пустой контрольной точке читает книгу доменов из unified_ledger.json: домен x на месте."""
    import json  # разбор книги доменов
    from hagi.selfdev import unified as U  # единая модель
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    (tmp_path / "unified_ledger.json").write_text(json.dumps({"x": {"sha": "s", "bytes": 9, "ce": None}}), encoding="utf-8")  # книга: домен x с sha s, 9 байтов, CE не измерена
    u = U.Unified()  # новая единая модель (файла весов нет)
    assert "x" in u.domains  # домен x прочитан из книги
