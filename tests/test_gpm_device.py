"""Тест: базисы защиты GPM следуют за устройством модели.

Базисы gpm_left/gpm_right создаются при обновлении (numpy → as_tensor) и
восстанавливаются при загрузке — оба раза на процессоре. Если модель на GPU,
а базисы на процессоре, первый же шаг обучения падает:
RuntimeError: Expected all tensors to be on the same device (mat2 on cuda:0).

Сторож: если перенос базисов в sync_device убрать, test_gpm_bases_follow_device
падает на машине с GPU; на машине только с CPU проверяется форма пути
(базисы присутствуют после загрузки и устройство совпадает)."""
import pytest  # фикстуры
import hagi.selfdev.unified as U  # единая модель


def test_gpm_bases_follow_device(tmp_path, monkeypatch):
    """После загрузки книги с базисами каждый базис лежит на устройстве модели."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # изолированный каталог контрольных точек
    monkeypatch.setattr(U, "USE_GPM", True)  # защита включена
    monkeypatch.setenv("HAGI_REPLAY_FRAC", "1.0")  # полный повтор
    monkeypatch.setattr(U.Unified, "GROW_BLOCKS", 0)  # без роста: sync_device вызывается только в конструкторе и при загрузке — как в цикле между приёмами без роста
    import torch as t  # тензоры
    t.manual_seed(0)  # воспроизводимость
    u = U.Unified()  # новая модель
    data = (bytes(range(32, 127)) * 400)[:50000]  # текст
    u.ingest("dev_a", data, epochs=1)  # первый домен
    u.ingest("dev_b", data[700:] + data[:700], epochs=1)  # второй: появляются базисы
    u.save()  # сохранить
    canon = lambda d: str(t.empty(0, device=d).device)  # каноническое имя устройства (cuda → cuda:0)
    # В реальном цикле приёмы идут подряд без загрузки: базисы пересоздаются после каждого
    # принятого домена, и следующий приём сразу делает шаг с ними
    for i in range(3):  # три приёма подряд
        u.ingest(f"dev_{i}", data[i * 900:] + data[:i * 900], epochs=1)  # приём
        bases = getattr(u.model, "gpm_left", {}) or {}  # базисы после приёма
        assert bases, "нет базисов — нечего проверять"  # предусловие
        where = sorted({canon(v.device) for v in bases.values()})  # устройства базисов
        assert where == [canon(u.model.device)], f"после приёма {i} базисы не на устройстве модели: {where}"  # на устройстве
    u.save()  # сохранить книгу с базисами
    u2 = U.Unified.load()  # загрузить
    bases2 = getattr(u2.model, "gpm_left", {}) or {}  # базисы загруженной модели
    assert bases2 and sorted({canon(v.device) for v in bases2.values()}) == [canon(u2.model.device)], "после загрузки базисы не на устройстве модели"  # после load
    # Перенос модели на другое устройство (если есть) должен унести базисы с собой
    other = "cpu" if str(u2.model.device) != "cpu" else ("cuda:0" if t.cuda.is_available() else None)  # второе устройство
    if other is not None:  # есть куда переносить
        u2.model.device = other  # сменить устройство
        u2.model.sync_device()  # перенести всё
        assert all(canon(v.device) == canon(other) for v in (getattr(u2.model, "gpm_left", {}) or {}).values()), "после sync_device базисы остались на прежнем устройстве"  # перенесены
