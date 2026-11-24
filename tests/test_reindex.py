import hashlib
from hagi import unified as U


def test_reindex_rebuilds_the_book_from_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    (tmp_path / "domains").mkdir()
    a = bytes(range(256)) * 40
    (tmp_path / "domains" / "a.bin").write_bytes(a)
    (tmp_path / "domains" / "tiny.bin").write_bytes(b"xx")
    u = U.Unified()
    u.domains = {"a": {"sha": "old", "bytes": 1, "ce": 1.5},   # measured: kept
                 "kept": {"sha": "k", "bytes": 9, "ce": 2.0}}
    assert u.reindex_domains() == 0            # 'a' has a measured ce
    u.domains["a"]["ce"] = None
    assert u.reindex_domains() == 1
    assert u.domains["a"] == {"sha": hashlib.sha1(a).hexdigest()[:10], "bytes": len(a), "ce": None}
    assert "tiny" not in u.domains and u.domains["kept"]["ce"] == 2.0
