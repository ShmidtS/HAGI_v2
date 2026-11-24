"""WAVE 259: a test must never be able to erase the production
checkpoint (2026-11-24: test_live_adapter's u.save() did, because
UNET_CKPT/LEDGER were import-time constants that monkeypatching
CKPT_DIR did not redirect)."""
import pytest

from hagi import unified as U


def _prod():
    import hagi.util
    return hagi.util.__dict__["CKPT_DIR"]


def test_paths_follow_the_current_ckpt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path / "sandbox")
    assert U._unified_pt().parent == tmp_path / "sandbox"
    assert U._ledger_js().parent == tmp_path / "sandbox"


def test_sandboxed_save_never_touches_production(tmp_path, monkeypatch):
    before = (U._PROD_CKPT / "unified.pt")
    stamp = (before.stat().st_mtime_ns, before.stat().st_size) if before.exists() else None
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path / "sb")
    u = U.Unified()
    u.domains["t"] = {"sha": "x", "bytes": 1, "ce": 0.5}
    u.save()
    assert (tmp_path / "sb" / "unified.pt").exists()
    assert (tmp_path / "sb" / "unified_ledger.json").exists()
    after = (before.stat().st_mtime_ns, before.stat().st_size) if before.exists() else None
    assert after == stamp
    assert U.Unified.load().domains == {"t": {"sha": "x", "bytes": 1, "ce": 0.5}}


def test_save_refuses_production_path_under_pytest(monkeypatch):
    monkeypatch.setattr(U, "CKPT_DIR", U._PROD_CKPT)
    u = U.Unified()
    with pytest.raises(RuntimeError, match="production checkpoint"):
        u.save()
