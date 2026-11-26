"""Блок роста причинной модели (optimal) переживает сохранение и загрузку.

Прежде блоки optimal не записывались в контрольную точку: принятый по гейту блок
жил только до перезапуска, и загруженная модель теряла его (хвост 0.66 -> 1.21)."""
import pytest  # фикстуры
import hagi.selfdev.unified as U  # единая модель


def test_optimal_growth_blocks_survive_save_load(tmp_path, monkeypatch):
    """Выращенный блок есть в файле и возвращается при загрузке с теми же весами."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # изолированный каталог контрольных точек
    monkeypatch.setenv("HAGI_REPLAY_FRAC", "1.0")  # полный повтор — как в рабочем цикле
    import torch as t  # тензоры
    t.manual_seed(0)  # воспроизводимая инициализация
    u = U.Unified()  # новая модель
    assert getattr(u.model, "ARCH", "") == "optimal", "тест для причинной модели"  # архитектура
    data = (bytes(range(32, 127)) * 400)[:50000]  # текст без служебных байтов
    u.ingest("blk_dom", data, epochs=1)  # приём: может вырастить блок
    u.model.grow(seed=3)  # гарантированно добавить блок
    with t.no_grad():  # задать блоку не тождественные веса
        for q in u.model.blocks[-1]:  # по тензорам блока
            q.add_(0.05)  # сдвиг
    n = len(u.model.blocks)  # число блоков
    before = [q.detach().clone() for q in u.model.blocks[-1]]  # веса блока
    ce_mem = u._tail_ce(data)  # CE в памяти
    u.save()  # сохранить
    u2 = U.Unified.load()  # загрузить
    assert len(u2.model.blocks) == n  # блок восстановлен
    assert all(t.equal(a, b.to(a.device)) for a, b in zip(before, u2.model.blocks[-1]))  # веса те же
    assert abs(u2._tail_ce(data) - ce_mem) < 1e-5  # CE та же
