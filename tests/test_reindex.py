"""Тесты переиндексации книги доменов Unified.reindex_domains.

    Свойство: книга доменов восстанавливается из сырых байтов файлов
    CKPT_DIR/domains/*.bin. Запись с уже измеренной CE не пересчитывается;
    запись без CE получает sha = первые 10 знаков sha1(байтов), bytes = длина
    и ce = None. Файл короче 4096 байтов в книгу не попадает."""
import hashlib  # sha1 для ожидаемого отпечатка
from hagi import unified as U  # единая модель и её книга доменов


def test_reindex_rebuilds_the_book_from_bytes(tmp_path, monkeypatch):
    """Измеренная запись сохраняется, неизмеренная пересобирается из файла, короткий файл пропускается.

        tmp_path: пустой временный каталог; monkeypatch: подмена CKPT_DIR на tmp_path."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # каталог контрольной точки := tmp_path
    (tmp_path / "domains").mkdir()  # создать каталог domains
    a = bytes(range(256)) * 40  # данные домена a: байты 0..255, повторённые 40 раз (10 240 байтов)
    (tmp_path / "domains" / "a.bin").write_bytes(a)  # записать a в файл a.bin
    (tmp_path / "domains" / "tiny.bin").write_bytes(b"xx")  # файл из 2 байтов — короче порога 4096
    u = U.Unified()  # новая единая модель
    u.domains = {"a": {"sha": "old", "bytes": 1, "ce": 1.5},  # домен a: ce уже измерена (1.5) — запись сохраняется
                 "kept": {"sha": "k", "bytes": 9, "ce": 2.0}}  # книга: a с измеренной CE (сохраняется), kept — посторонняя запись
    assert u.reindex_domains() == 0  # у a CE уже измерена — пересборка ничего не добавляет
    u.domains["a"]["ce"] = None  # у a CE больше нет — запись устарела
    assert u.reindex_domains() == 1  # пересборка добавила ровно один домен (a)
    assert u.domains["a"] == {"sha": hashlib.sha1(a).hexdigest()[:10], "bytes": len(a), "ce": None}  # запись a: sha = sha1(a)[:10], bytes = len(a), CE не измерена
    assert "tiny" not in u.domains and u.domains["kept"]["ce"] == 2.0  # короткий файл не добавлен, запись kept не тронута
