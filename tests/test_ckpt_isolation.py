"""Тесты изоляции тестов от рабочей контрольной точки.

    Свойство: ни один тест не может стереть или перезаписать рабочую
    контрольную точку. Пути к файлам строятся от ТЕКУЩЕГО значения
    U.CKPT_DIR на момент вызова (а не от константы, захваченной при импорте),
    поэтому подмена CKPT_DIR переносит и запись, и чтение. save() под pytest
    с CKPT_DIR, равным рабочему каталогу, обязан вызвать RuntimeError.

    Причина: раньше пути к файлам весов и книги доменов были константами,
    вычисленными при импорте; подмена CKPT_DIR их не перенаправляла, и вызов
    save() из теста перезаписывал рабочие веса."""
import pytest  # декораторы и проверка исключений

from hagi import unified as U  # единая модель: пути, save, load


def _prod():
    """Вернуть рабочий каталог контрольной точки, захваченный при импорте hagi.util.

        hagi.util.__dict__["CKPT_DIR"] — значение на момент импорта; его же
        копия сохранена в U._PROD_CKPT."""
    import hagi.util  # модуль путей
    return hagi.util.__dict__["CKPT_DIR"]  # CKPT_DIR из словаря модуля hagi.util


def test_paths_follow_the_current_ckpt_dir(tmp_path, monkeypatch):
    """Пути к весам и книге доменов следуют за текущим CKPT_DIR: родительский каталог = подменённый CKPT_DIR."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path / "sandbox")  # CKPT_DIR := временный каталог
    assert U._unified_pt().parent == tmp_path / "sandbox"  # каталог файла unified.pt = временный каталог
    assert U._ledger_js().parent == tmp_path / "sandbox"  # каталог файла unified_ledger.json = временный каталог


def test_sandboxed_save_never_touches_production(tmp_path, monkeypatch):
    """save() с подменённым CKPT_DIR пишет только во временную копию; рабочий файл не меняется
    по времени изменения (st_mtime_ns) и размеру (st_size); load() читает ту же временную копию."""
    before = (U._PROD_CKPT / "unified.pt")  # путь к рабочему файлу весов
    stamp = (before.stat().st_mtime_ns, before.stat().st_size) if before.exists() else None  # снимок рабочего файла (время, размер) либо None, если его нет
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path / "sb")  # CKPT_DIR := tmp_path/sb
    u = U.Unified()  # новая единая модель
    u.domains["t"] = {"sha": "x", "bytes": 1, "ce": 0.5}  # запись домена t в книге
    u.save()  # сохранить модель и книгу во временную копию
    assert (tmp_path / "sb" / "unified.pt").exists()  # файл весов создан во временной копии
    assert (tmp_path / "sb" / "unified_ledger.json").exists()  # файл книги создан во временной копии
    after = (before.stat().st_mtime_ns, before.stat().st_size) if before.exists() else None  # снимок рабочего файла после save
    assert after == stamp  # рабочий файл не изменился: время и размер те же
    assert U.Unified.load().domains == {"t": {"sha": "x", "bytes": 1, "ce": 0.5}}  # load читает из временной копии: книга содержит домен t


def test_save_refuses_production_path_under_pytest(monkeypatch):
    """save() под pytest с CKPT_DIR, равным рабочему каталогу, вызывает RuntimeError
    с текстом «production checkpoint»."""
    monkeypatch.setattr(U, "CKPT_DIR", U._PROD_CKPT)  # CKPT_DIR := рабочий каталог
    u = U.Unified()  # новая единая модель
    with pytest.raises(RuntimeError, match="production checkpoint"):  # ожидается RuntimeError с этим текстом
        u.save()  # попытка записи в рабочую контрольную точку — должна быть отклонена
