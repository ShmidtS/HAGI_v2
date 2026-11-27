"""Точный обход: причинные словари 5 и 3 байт, отказ при противоречии, прибавка к логитам, сохранение."""
import torch as t  # тензоры

from hagi.models import optimal_lm as M  # модель


def test_trie_counts_conflicts_and_keeps_old_entries():
    """Повтор усиливает запись, противоречие помечает её сбойной, другие контексты не задевает."""
    d: dict = {}  # пустой словарь
    for nxt in (ord("a"),) * 3:  # три раза за контекстом идёт a
        M.HagiOptimalLM._trie_put(d, b"12345", nxt)  # голос
    assert d[b"12345"] == [ord("a"), 3, False]  # предсказание a, три голоса, живая
    M.HagiOptimalLM._trie_put(d, b"12345", ord("b"))  # противоречие
    assert d[b"12345"][2] is True  # сбойная — больше не отвечает
    M.HagiOptimalLM._trie_put(d, b"12345", ord("a"))  # после сбоя счёт не растёт
    assert d[b"12345"][1] == 3  # счёт прежний
    M.HagiOptimalLM._trie_put(d, b"99999", ord("z"))  # другой контекст
    assert d[b"99999"] == [ord("z"), 1, False]  # он жив


def test_bypass_lowers_ce_on_unseen_repetition_and_is_causal():
    """На отложенном куске потока с повторяющимися токенами обход снижает CE; ответ для позиции не зависит от будущих байтов."""
    t.manual_seed(0)  # зерно
    m = M.HagiOptimalLM()  # модель
    m.spike_k = None  # без спайкинга — сравниваем только обход
    m.bypass_boost = 8.0  # прибавка
    head = (b"def f(x):\n    return x\n" * 400) + bytes(t.randint(0, 256, (4000,)).tolist())  # обучающая часть: повтор + шум
    tail = b"def f(x):\n    return x\n" * 20  # отложенный кусок: только повтор
    m.bypass_fit(head)  # словари по обучающей части
    x = t.as_tensor(list(tail[:-1]))[None]  # вход
    y = t.as_tensor(list(tail[1:]), device=m.device)[None]  # цели (на устройстве модели)
    with t.no_grad():  # без градиентов
        on = float(t.nn.functional.cross_entropy(m.forward_seq(x).reshape(-1, 256), y.reshape(-1)))  # с обходом
        m.bypass_boost = 0.0  # выключить
        off = float(t.nn.functional.cross_entropy(m.forward_seq(x).reshape(-1, 256), y.reshape(-1)))  # без обхода
        m.bypass_boost = 8.0  # вернуть
        full = m.forward_seq(x)[0, :200]  # логиты первых 200 позиций полного окна
        part = m.forward_seq(x[:, :200])[0]  # те же позиции на укороченном окне
    assert on < off - 1.0 and t.allclose(full, part, atol=1e-5)  # обход помогает и причинен


def test_bypass_survives_save_load(tmp_path, monkeypatch):  # каталог состояния, подмена путей
    """Словари и прибавка переживают сохранение и загрузку; обучающая часть не заглядывает в контроль."""
    from hagi.selfdev import unified as U  # единая модель
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # состояние в каталог теста
    m = M.HagiOptimalLM()  # модель
    for _ in range(3):  # три голоса
        m._trie_put(m.bypass_t5, b"abcde", ord("q"))  # запись 5-байтового словаря
    u = U.Unified(k=4)  # обёртка
    u.model = m  # подставить модель с обходом
    u.save()  # сохранить
    v = U.Unified.load().model  # загрузить
    assert v.bypass_t5.get(b"abcde") == [ord("q"), 3, False] and v.bypass_boost == m.bypass_boost  # запись и прибавка те же
