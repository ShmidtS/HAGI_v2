"""WAVE 249: inference applies a domain adapter only if it passes the live paired gate."""
import numpy as np
import pytest

from hagi import unified as U


def _unified(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    u = U.Unified()
    u.domains["d"] = {}
    u._domain_bytes = lambda s, e: bytes(range(256)) * 12      # p8 law
    u.model.train_guarded(bytes(range(256)) * 12, epochs=3)
    return u


def test_no_domain_keeps_the_neutral_path(tmp_path, monkeypatch):
    u = _unified(tmp_path, monkeypatch)
    seen = []
    orig = u.model.logits
    u.model.logits = lambda x: (seen.append(u.model.domain_adapter), orig(x))[1]
    u.predict_next_byte(bytes(range(200, 216)), allow_neural=True)
    assert seen == [None]                                   # no adapter without a domain


def test_trained_adapter_is_applied_only_when_it_still_helps(tmp_path, monkeypatch):
    u = _unified(tmp_path, monkeypatch)
    r = u._fit_domain_adapter("d")
    assert r["status"] == "GROW"
    ad = u.adapter_bank.get("d")
    assert u._live_adapter("d") is ad                      # fresh = helps now
    with u._DomainScope(u, "d"):
        assert u.model.domain_adapter is ad
    assert u.model.domain_adapter is None                   # restored
    # A stale adapter = certified against an OLD trunk. Age the trunk the
    # way the cycle does: keep training it on the same domain. The trunk
    # then learns the law itself and the old adapter is no longer a clear
    # win -> the gate must not apply it blindly. (Randomising the trunk
    # instead is a WRONG test: a uniform trunk lets any adapter "help".)
    u.model.train_guarded(bytes(range(256)) * 12, epochs=25)
    verdict = u._live_adapter("d")
    live = verdict is not None
    # independent check of the gate's decision on fresh positions
    data = bytes(range(256)) * 12
    idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64); k = u.model.k
    pos = np.arange(2300, 2900)
    X, y = u._windows("d", pos)
    import torch as t
    u.model.domain_adapter = None
    with t.no_grad():
        base = float(t.nn.functional.cross_entropy(u.model.logits(X), y))
    with_ad = float(u._adapter_val_ce(ad, X, y).mean()); u.model.domain_adapter = None
    if live:
        assert with_ad < base, "gate kept an adapter that hurts"
    else:
        assert with_ad >= base - 0.02, "gate dropped an adapter that clearly helps"
    assert u._live_adapter("nope") is None and u._live_adapter(None) is None


def test_scope_restores_state_on_exception(tmp_path, monkeypatch):
    u = _unified(tmp_path, monkeypatch)
    u._fit_domain_adapter("d")
    with pytest.raises(RuntimeError):
        with u._DomainScope(u, "d"):
            raise RuntimeError("boom")
    assert u.model.domain_adapter is None
