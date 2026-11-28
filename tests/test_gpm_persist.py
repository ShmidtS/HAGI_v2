"""Защита GPM переживает сохранение и загрузку.

G включена по умолчанию (HAGI_GPM=1): после принятого домена модель хранит
базисы gpm_left — подпространства входов старых доменов, в которые не заходит
градиент и шаг. Если базисы не сохранить, загруженная модель теряет защиту, и
следующий приём затирает старые домена как без защиты.

Сторож: если базисы перестать писать в контрольную точку или не восстанавливать
при загрузке, test_gpm_bases_survive_save_load падает — у загруженной модели
gpm_left пуст или базисы отличаются."""
import pytest  # фикстуры
import hagi.selfdev.unified as U  # единая модель


def test_gpm_bases_survive_save_load(tmp_path, monkeypatch):
    """Базисы GPM записываются в контрольную точку и возвращаются равными исходным."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # изолированный каталог контрольных точек
    monkeypatch.setattr(U, "USE_GPM", True)  # защита включена (как по умолчанию)
    monkeypatch.setenv("HAGI_REPLAY_FRAC", "1.0")  # полный повтор — как в рабочем цикле
    import torch as t  # тензоры
    t.manual_seed(0)  # воспроизводимая инициализация
    u = U.Unified()  # новая модель
    data = (bytes(range(32, 127)) * 400)[:50000]  # текст без служебных байтов
    u.ingest("gpm_dom_a", data, epochs=1)  # первый домен: базисов ещё нет (нечего защищать)
    u.ingest("gpm_dom_b", data[1000:] + data[:1000], epochs=1)  # второй домен: накапливаются базисы первого
    bases = getattr(u.model, "gpm_left", {}) or {}  # базисы в памяти
    assert bases, "после принятого домена с GPM базисы должны появиться"  # защита накоплена
    before = {k: v.detach().clone() for k, v in bases.items()}  # копия базисов
    u.save()  # сохранить
    u2 = U.Unified.load()  # загрузить
    bases2 = getattr(u2.model, "gpm_left", {}) or {}  # базисы загруженной модели
    assert set(bases2) == set(before), f"загружены базисы иных матриц: {sorted(bases2)} против {sorted(before)}"  # тот же набор
    for k, v in before.items():  # по базисам
        assert t.equal(v, bases2[k].to(v.device)), f"базис {k} изменился при сохранении/загрузке"  # те же значения


def test_unprotected_book_loads_with_empty_gpm(tmp_path, monkeypatch):
    """Книга без базисов (записана до включения защиты) загружается с пустой защитой — не падает."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # изолированный каталог контрольных точек
    monkeypatch.setattr(U, "USE_GPM", True)  # защита включена (как по умолчанию)
    import torch as t  # тензоры
    t.manual_seed(0)  # воспроизводимая инициализация
    u = U.Unified()  # новая модель
    monkeypatch.setattr(U, "USE_GPM", False)  # сохранить книгу БЕЗ защиты (как старые книги)
    u.ingest("old_dom", (bytes(range(32, 127)) * 400)[:30000], epochs=1)  # приём без накопления базисов
    u.save()  # сохранить книгу без базисов
    monkeypatch.setattr(U, "USE_GPM", True)  # защита включена
    u2 = U.Unified.load()  # загрузить старую книгу с включённой защитой
    assert not (getattr(u2.model, "gpm_left", None) or {}), "у книги без базисов защита должна быть пустой"  # защита пуста
    assert "old_dom" in u2.domains, "домен книги потерян при загрузке"  # книга цела
