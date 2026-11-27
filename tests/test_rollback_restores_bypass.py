"""Откат приёма возвращает модель побитово: веса, блоки и записи словарей точного обхода.

Записи обхода — изменяемые списки [прогноз, счёт, сбой]; обучение правит их на месте.
Поверхностная копия словаря делила записи, и после отката старые домены были испорчены
при побитово тех же весах (хвосты 0.485 -> 0.628, 0.180 -> 0.369)."""
import pytest  # фикстуры
import hagi.selfdev.unified as U  # единая модель


def _copy(d):  # побайтовая копия словаря обхода
    return {k: list(v) for k, v in d.items()}  # ключ -> копия записи


def test_rollback_restores_weights_blocks_and_bypass(tmp_path, monkeypatch):
    """После отката регрессии веса, число блоков и записи обхода равны состоянию до приёма."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # изолированный каталог
    monkeypatch.setenv("HAGI_REPLAY_FRAC", "1.0")  # полный повтор — как в рабочем цикле
    monkeypatch.setattr(U.Unified, "_repair_forgetting", lambda self, *a, **k: {"rounds": []})  # без ремонта: проверяем сам откат
    calls = []  # счётчик замеров CE старых доменов внутри одного приёма
    def fake_old_ce(self, exclude, max_domains=8):  # первый замер (до обучения) — 1.0, последующие (после) — 2.0
        calls.append(1); return 1.0 if len(calls) == 1 else 2.0  # регрессия в 2 раза
    monkeypatch.setattr(U.Unified, "_old_domains_ce", fake_old_ce)  # подмена замера
    import torch as t  # тензоры
    t.manual_seed(0)  # воспроизводимость
    u = U.Unified()  # новая модель
    old = (bytes(range(32, 127)) * 300)[:30000]  # старый домен
    calls.clear()  # приём старого домена — без подмены регрессии: сбросить счётчик и временно вернуть честный замер
    monkeypatch.setattr(U.Unified, "_old_domains_ce", lambda self, exclude, max_domains=8: None)  # старых доменов ещё нет
    u.ingest("old_dom", old, epochs=1)  # принять старый домен (обучает обход)
    monkeypatch.setattr(U.Unified, "_old_domains_ce", fake_old_ce)  # восстановить подмену для второго приёма
    if not getattr(u.model, "bypass_t5", None):  # обход пуст — обучить вручную
        u.model.bypass_fit(old)  # наполнить словари
    assert u.model.bypass_t5, "словарь обхода должен быть непустым"  # предусловие
    weights = [q.detach().clone() for q in u.model.params()]  # веса до
    names = u.model.param_names()  # имена
    bypass = (_copy(u.model.bypass_t5), _copy(u.model.bypass_t3))  # обход до
    nblocks = len(u.model.blocks)  # блоки до
    new = bytes(range(128, 200)) * 300  # новый домен — другой алфавит
    u.ingest("new_dom", new, epochs=1)  # приём с последующим откатом
    assert u.domains["new_dom"].get("regression") is True  # гейт сработал
    assert u.domains["new_dom"].get("rolled_back") is True  # домен помечен откатанным
    assert len(u.model.blocks) == nblocks  # блоки — как до приёма
    cur = dict(zip(u.model.param_names(), u.model.params()))  # текущие по именам
    assert all(t.equal(cur[n], w) for n, w in zip(names, weights))  # веса побитово
    assert _copy(u.model.bypass_t5) == bypass[0] and _copy(u.model.bypass_t3) == bypass[1]  # записи обхода побитово
