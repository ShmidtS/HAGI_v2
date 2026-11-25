"""Тесты живого гейта доменного адаптера при выводе и роста тела модели.

    Живой гейт (_live_adapter): адаптер домена ставится на модель только если
    он помогает СЕЙЧАСС. На 256 случайных окнах первых 85% байтов домена
    считается парный выигрыш d_i = CE_без_i − CE_с_i; нижняя граница
    mean(d) − 2,4·SE(d) должна превысить 0,005. Причина: адаптер принимался
    против старого тела, а тело с тех пор обучалось; на текущем теле адаптер
    может вредить. _DomainScope ставит адаптер на время блока и всегда
    возвращает прежнее состояние модели.

    Рост тела: grow() добавляет блок внимания с Wo = 0 — logits не меняются
    совсем; _maybe_grow оставляет блок, только если выигрыш CE превышает
    границу, иначе откатывает (ROLLBACK). Блок ремонта (старая память)
    замораживается в общем обучении; после роста старые домена переучиваются,
    пока забывание не исчезнет."""
import numpy as np  # массивы байтов
import pytest  # проверка исключений

from hagi.selfdev import unified as U  # единая модель


def _unified(tmp_path, monkeypatch):
    """Единая модель во временном каталоге с доменом d; байты домена — 0..255 ×12; тело обучено 3 эпохи."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setenv("HAGI_ARCH", "rope")  # эти тесты — о механизмах Rope-модели (адаптеры A/B, рост блоков W1/Wo); оптимальная модель по умолчанию проверяется отдельно
    monkeypatch.setattr(U.Unified, "ARCH", "rope")  # атрибут класса читается при создании Unified
    u = U.Unified()  # новая единая модель
    u.domains["d"] = {}  # пустая запись домена d
    u._domain_bytes = lambda s, e: bytes(range(256)) * 12  # байты домена: 0..255, повторённые 12 раз
    u.model.train_guarded(bytes(range(256)) * 12, epochs=3)  # обучить тело на законе домена, 3 эпохи
    return u  # вернуть модель


def test_no_domain_keeps_the_neutral_path(tmp_path, monkeypatch):
    """Без домена адаптер не ставится: при predict_next_byte(allow_neural=True) модель видит
    domain_adapter = None; исходный logits вызывается ровно один раз."""
    u = _unified(tmp_path, monkeypatch)  # модель с доменом d
    seen = []  # значения domain_adapter в момент вызова logits
    orig = u.model.logits  # исходный метод logits
    u.model.logits = lambda x: (seen.append(u.model.domain_adapter), orig(x))[1]  # подменить logits: записать адаптер в seen и вернуть orig(x)
    u.predict_next_byte(bytes(range(200, 216)), allow_neural=True)  # предсказать байт с разрешённой нейронной догадкой
    assert seen == [None]  # без домена модель ни разу не получила адаптер


def test_trained_adapter_is_applied_only_when_it_still_helps(tmp_path, monkeypatch):
    """Свежий адаптер проходит живой гейт и ставится только внутри _DomainScope; после обучения тела без
    адаптера решение гейта согласовано с независимой проверкой CE на новых позициях: адаптер
    применяется только если он реально снижает CE; без домена или с неизвестным доменом — None."""
    u = _unified(tmp_path, monkeypatch)  # модель с доменом d
    r = u._fit_domain_adapter("d")  # подогнать адаптер домена d
    assert r["status"] == "GROW"  # адаптер принят
    ad = u.adapter_bank.get("d")  # адаптер домена d
    assert u._live_adapter("d") is ad  # свежий адаптер помогает сейчас — живой гейт его возвращает
    with u._DomainScope(u, "d"):  # область с живым гейтом домена d
        assert u.model.domain_adapter is ad  # внутри области адаптер поставлен
    assert u.model.domain_adapter is None  # прежнее состояние восстановлено: адаптера нет
    # Устаревший адаптер = принятый против СТАРОГО тела. Старим тело так, как это
    # делает цикл: продолжаем обучать его на том же домене. Тело выучивает закон
    # само, и прежний адаптер перестаёт быть явным выигрышем — гейт не должен
    # ставить его вслепую. (Случайно перезапутать тело — НЕВЕРНЫЙ тест:
    # на равномерном теле «помогает» любой адаптер.)
    u.model.train_guarded(bytes(range(256)) * 12, epochs=25)  # дообучить тело на законе домена, 25 эпох
    verdict = u._live_adapter("d")  # решение живого гейта на новом теле
    live = verdict is not None  # гейт оставил адаптер (True) либо снял (False)
    # независимая проверка решения гейта на свежих позициях
    data = bytes(range(256)) * 12  # байты домена
    idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64); k = u.model.k  # idx — байты как целые; k — длина контекста
    pos = np.arange(2300, 2900)  # позиции 2300..2899 — дальше обучающей части
    X, y = u._windows("d", pos)  # окна: X_i = idx[p−k .. p−1], цели y_i = idx[p + k]
    import torch as t  # тензоры
    u.model.domain_adapter = None  # база меряется без адаптера
    with t.no_grad():  # без градиентов
        base = float(t.nn.functional.cross_entropy(u.model.logits(X), y))  # базовая CE = среднее по окнам −log softmax(logits(X))[y]
    with_ad = float(u._adapter_val_ce(ad, X, y).mean()); u.model.domain_adapter = None  # CE с адаптером (среднее по окнам); адаптер снят
    if live:  # если гейт оставил адаптер
        assert with_ad < base, "gate kept an adapter that hurts"  # тогда он обязан снижать CE
    else:  # если гейт снял адаптер
        assert with_ad >= base - 0.02, "gate dropped an adapter that clearly helps"  # тогда он не снижает CE заметно (допуск 0,02)
    assert u._live_adapter("nope") is None and u._live_adapter(None) is None  # неизвестный домен и отсутствие домена — без адаптера


def test_scope_restores_state_on_exception(tmp_path, monkeypatch):
    """_DomainScope восстанавливает domain_adapter даже при исключении внутри блока:
    после RuntimeError модель снова без адаптера."""
    u = _unified(tmp_path, monkeypatch)  # модель с доменом d
    u._fit_domain_adapter("d")  # подогнать адаптер домена d
    with pytest.raises(RuntimeError):  # ожидается RuntimeError
        with u._DomainScope(u, "d"):  # область с адаптером домена d
            raise RuntimeError("boom")  # исключение внутри области
    assert u.model.domain_adapter is None  # адаптер снят даже при исключении


# ---- Рост тела с сохранением функции ------------------------------------------
def test_grow_preserves_logits_exactly_and_trains_the_block(tmp_path, monkeypatch):
    """grow() добавляет блок с Wo = 0: logits не меняются точно (max |Δ| = 0). _maybe_grow при лимите
    2 блока даёт GROW / ROLLBACK / HOLD; при GROW CE не выросло и блоков 2, при ROLLBACK блок 1;
    после save/load число блоков и веса Wo сохраняются (расхождение < 1e-6)."""
    import torch as t  # тензоры
    from hagi.selfdev import unified as U  # единая модель
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setenv("HAGI_ARCH", "rope")  # тест о Rope-механизмах (блоки W1/Wo, адаптеры)
    monkeypatch.setattr(U.Unified, "ARCH", "rope")  # атрибут класса читается при создании
    u = U.Unified()  # новая единая модель
    u.domains["d"] = {}  # пустая запись домена d
    data = bytes(range(256)) * 30  # байты домена: 0..255 ×30
    u._domain_bytes = lambda s, e: data  # байты домена := data
    u.model.train_guarded(data, epochs=2)  # обучить тело, 2 эпохи
    g = t.Generator().manual_seed(0)  # генератор с зерном 0
    X = t.randint(0, 256, (16, u.model.k), generator=g)  # 16 случайных контекстов длины k
    with t.no_grad():  # без градиентов
        l0 = u.model.logits(X).clone()  # logits до роста
    assert u.model.grow() == 1  # рост добавил ровно один блок
    with t.no_grad():  # без градиентов
        assert float((u.model.logits(X) - l0).abs().max()) == 0.0  # точное тождество: логиты не изменились
    u.GROW_BLOCKS = 2  # лимит блоков := 2 (по умолчанию рост выключен: блоки вредят старым доменам)
    r = u._maybe_grow("d", data)  # попытка добавить блок роста на домене d
    assert r["status"] in ("GROW", "ROLLBACK", "HOLD", "HOLD_UNCERTAIN")  # статус — рост, откат, удержание или «не перекрыл границу и цену памяти»
    if r["status"] == "GROW":  # блок принят
        assert r["gain"] > r["margin"] + r["mem_cost"]  # правило приёма: выигрыш больше границы и цены памяти
    if r["status"] == "GROW":  # если блок принят
        assert r["ce_after"] <= r["ce_before"] and len(u.model.blocks) == 2  # CE не выросло и блоков 2
    if r["status"] == "ROLLBACK":  # если блок откатан
        assert len(u.model.blocks) == 1  # точный откат: добавленный блок снят
    # круглый ход сохранения оставляет блоки на месте
    u.save(); v = U.Unified.load()  # сохранить и загрузить
    assert len(v.model.blocks) == len(u.model.blocks)  # число блоков совпадает
    if u.model.blocks:  # если блоки есть
        for a, b in zip(u.model.blocks, v.model.blocks):  # по парам одинаковых блоков
            assert float((a["Wo"] - b["Wo"]).abs().max()) < 1e-6  # веса Wo совпадают: max |Δ| < 1e-6


def test_grow_rolls_back_a_block_that_gains_nothing(tmp_path, monkeypatch):
    """Блок, не дающий выигрыша CE (ce_before = ce_after), откатывается: при подмене cross_entropy
    константой 1,0 статус ROLLBACK либо HOLD и число блоков не меняется.

        Причина: блок с ce_before = ce_after (3,4673 → 3,4673) был принят как GROW;
        ничего не заработано — значит откат."""
    import torch as t  # тензоры
    from hagi.selfdev import unified as U  # единая модель
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setenv("HAGI_ARCH", "rope")  # тест о Rope-механизмах (блоки W1/Wo, адаптеры A/B)
    monkeypatch.setattr(U.Unified, "ARCH", "rope")  # атрибут класса читается при создании
    u = U.Unified()  # новая единая модель
    u.domains["d"] = {}  # пустая запись домена d
    data = bytes(range(256)) * 30  # байты домена
    u._domain_bytes = lambda s, e: data  # байты домена := data
    u.model.train_guarded(data, epochs=2)  # обучить тело, 2 эпохи
    n0 = len(u.model.blocks)  # число блоков до попытки роста
    calls = {"n": 0}  # счётчик вызовов (в проверках не используется)
    orig = t.nn.functional.cross_entropy  # исходная cross_entropy
    def frozen(*a, **k):  # любая оценка потери возвращает одно и то же значение
        calls["n"] += 1  # счётчик вызовов +1
        return orig(*a, **k).mul(0).add(1.0)  # 0·CE + 1 = 1,0 при любых аргументах
    u.GROW_BLOCKS = 2  # лимит блоков := 2
    monkeypatch.setattr(t.nn.functional, "cross_entropy", frozen)  # подменить cross_entropy константой
    r = u._maybe_grow("d", data)  # попытка роста: до и после CE одинакова
    monkeypatch.setattr(t.nn.functional, "cross_entropy", orig)  # вернуть исходную cross_entropy
    assert r["status"] in ("ROLLBACK", "HOLD", "HOLD_UNCERTAIN")  # блок не принят (сертифицированные шаги не дают значимого прироста — HOLD_UNCERTAIN)
    assert len(u.model.blocks) == n0  # число блоков не изменилось


def test_repair_block_is_frozen_in_general_training_and_persisted(tmp_path, monkeypatch):
    """Блок ремонта (блок 0 в frozen_blocks) не меняется общим обучением (max |Δ| = 0) и сохраняется
    в файле вместе с набором замороженных блоков.

        Причина: подвижный блок ремонта на реальной модели дал +2,62 к CE старых доменов против
        +0,55 без блоков — блок ремонта обязан быть заморожен."""
    import torch as t  # тензоры
    from hagi.selfdev import unified as U  # единая модель
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setenv("HAGI_ARCH", "rope")  # тест о Rope-механизмах (блоки W1/Wo, адаптеры A/B)
    monkeypatch.setattr(U.Unified, "ARCH", "rope")  # атрибут класса читается при создании
    u = U.Unified()  # новая единая модель
    u.domains["d"] = {}  # пустая запись домена d
    data = bytes(range(256)) * 30  # байты домена
    u._domain_bytes = lambda s, e: data  # байты домена := data
    u.model.train_guarded(data, epochs=1)  # обучить тело, 1 эпоха
    u.model.grow(seed=3)  # добавить блок (будущий блок ремонта)
    u.model.frozen_blocks = {0}  # заморозить блок 0
    w = [q.detach().clone() for q in u.model.blocks[0].values()]  # снимок весов блока 0
    u.model.train_guarded(data, epochs=1)  # ещё одна эпоха общего обучения
    for q, w0 in zip(u.model.blocks[0].values(), w):  # по весам блока 0 и снимку
        assert float((q - w0).abs().max()) == 0.0  # вес не изменился: max |Δ| = 0
    u.save(); v = U.Unified.load()  # сохранить и загрузить
    assert v.model.frozen_blocks == {0} and len(v.model.blocks) == 1  # заморожен блок 0, блоков 1


def test_repair_retrains_old_domains_until_forgetting_is_gone(tmp_path, monkeypatch):
    """После обучения на новом домене CE старых доменов растёт (забывание), а _repair_forgetting
    возвращает её к pre + FORGET_TOL итеративными раундами: раунд принят (RETRAIN) только если CE
    снизилась, иначе откат (ROLLBACK); CE по раундам не возрастает.

        Цикл: пока CE старых доменов выше значения до обучения — шаг переобучения на старых байтах;
        шаг без улучшения откатывается; стоп по достижении допуска либо по лимиту раундов."""
    import torch as t  # тензоры
    from hagi.selfdev import unified as U  # единая модель
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setenv("HAGI_ARCH", "rope")  # тест о Rope-механизмах (блоки W1/Wo, адаптеры A/B)
    monkeypatch.setattr(U.Unified, "ARCH", "rope")  # атрибут класса читается при создании
    u = U.Unified()  # новая единая модель
    u.domains.update({"a": {}, "b": {}})  # два старых домена a и b
    old = bytes(range(0, 128, 2)) * 200  # старые байты: 0,2,4,…,126 ×200
    u._domain_bytes = lambda s, e: old  # байты всех доменов := old
    u.model.train_guarded(old, epochs=2)  # обучить на старых байтах, 2 эпохи
    pre = u._old_domains_ce(exclude="c")  # CE старых доменов до забывания
    # забываем нарочно: жёстко обучаем на посторонних байтах
    u.model.train_guarded(bytes(range(255, 0, -1)) * 200, epochs=4)  # обучить на обратном законе 255..1, 4 эпохи
    assert u._old_domains_ce(exclude="c") > pre  # забывание: CE старых доменов выросла
    u._domain_bytes = lambda s, e: bytes(range(255, 0, -1)) * 200 if s == "c" else old  # домен c — новый закон, остальные — старые байты
    r = u._repair_forgetting("c", pre, [q.detach().clone() for q in u.model.params()])  # переобучить старые домена, исключая c
    assert r["mode"] == "replay" and r["rounds"]  # режим replay и есть раунды
    assert all(x["status"] in ("RETRAIN", "ROLLBACK") for x in r["rounds"])  # каждый раунд — RETRAIN либо ROLLBACK
    assert u._old_domains_ce(exclude="c") <= pre + u.FORGET_TOL or r["rounds"][-1]["status"] == "ROLLBACK"  # забывание устранено либо последний раунд откатан
    # цикл монотонен по значению гейта
    ces = [x["ce"] for x in r["rounds"]]  # CE по раундам
    assert ces == sorted(ces, reverse=True)  # CE не возрастает от раунда к раунду


def test_repair_continues_while_the_worst_domain_is_still_forgotten(tmp_path, monkeypatch):
    """Среднее забывание уже в допуске (post <= pre), но один домен всё ещё забыт на 0.5 ната:
    ремонт не останавливается — делает ещё раунды (каждый под гейтом), пока наихудший домен
    не войдёт в допуск WORST_TOL. Замеры CE подменены детерминированными векторами
    (домен a — в норме, домен b — забыт; каждый раунд снижает CE домена b на 0.3)."""
    from hagi.selfdev import unified as U  # единая модель
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setenv("HAGI_ARCH", "rope")  # тест о Rope-механизмах (блоки W1/Wo, адаптеры A/B)
    monkeypatch.setattr(U.Unified, "ARCH", "rope")  # атрибут класса читается при создании
    u = U.Unified()  # новая единая модель
    u.domains.update({"a": {}, "b": {}})  # два старых домена a и b
    old = bytes(range(0, 128, 2)) * 200  # старые байты: 0,2,4,…,126 ×200
    u._domain_bytes = lambda s, e: old  # байты всех доменов := old
    pre_vec = np.full(400, 1.0)  # вектор CE до: 400 окон по 1.0
    rounds_done = {"n": 0}  # число выполненных раундов обучения (0 — сразу после роста)
    def vec_after():  # вектор CE после последнего раунда по сценарию
        b_ce = 1.6 - 0.3 * rounds_done["n"]  # CE домена b: 1.6, затем 1.3, 1.0, …
        return np.concatenate([np.full(200, 1.0), np.full(200, b_ce)])  # домен a = 1.0, домен b = b_ce
    def fake_vec(exclude, max_domains=8):  # подменённый замер вектора CE
        u._vec_names = ["a", "b"]  # два домена по 200 окон
        u._vec_bounds = [0, 200, 400]  # срезы доменов в склеенном векторе
        return ["a", "b"], vec_after()  # имена и вектор
    monkeypatch.setattr(u, "_old_domains_ce_vec", fake_vec)  # подменить замер вектора
    monkeypatch.setattr(u, "_old_domains_ce", lambda exclude: float(vec_after().mean()))  # среднее CE по тому же сценарию
    def fake_train(*a, **k):  # раунд обучения: веса не меняются, сценарий сдвигается на один раунд
        rounds_done["n"] += 1  # выполнен ещё один раунд
    monkeypatch.setattr(u.model, "train_certified", fake_train)  # обучение ствола заменено сценарием
    r = u._repair_forgetting("c", pre=1.3, incumbent_state=[q.detach().clone() for q in u.model.params()], pre_vec=pre_vec)  # ремонт
    assert r["mode"] == "replay"  # режим переучивания
    # среднее после 1-го раунда: (200·1.0 + 200·1.3)/400 = 1.15 <= 1.3 — прежнее условие остановило бы;
    # наихудший домен b: 1.3 − 1.0 = +0.3 > WORST_TOL = 0.25 — ремонт продолжается
    assert len(r["rounds"]) >= 2  # минимум два раунда
    assert all(x["status"] == "RETRAIN" for x in r["rounds"])  # все раунды приняты (CE домена b снижается каждый раунд)
