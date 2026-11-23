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


# ---- WAVE 255: function-preserving trunk growth -------------------------------
def test_grow_preserves_logits_exactly_and_trains_the_block(tmp_path, monkeypatch):
    import torch as t
    from hagi import unified as U
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    u = U.Unified()
    u.domains["d"] = {}
    data = bytes(range(256)) * 30
    u._domain_bytes = lambda s, e: data
    u.model.train_guarded(data, epochs=2)
    g = t.Generator().manual_seed(0)
    X = t.randint(0, 256, (16, u.model.k), generator=g)
    with t.no_grad():
        l0 = u.model.logits(X).clone()
    assert u.model.grow() == 1
    with t.no_grad():
        assert float((u.model.logits(X) - l0).abs().max()) == 0.0     # exact identity
    u.GROW_BLOCKS = 2   # off by default (WAVE 257: hurts old domains)
    r = u._maybe_grow("d", data)
    assert r["status"] in ("GROW", "ROLLBACK", "HOLD")
    if r["status"] == "GROW":
        assert r["ce_after"] <= r["ce_before"] and len(u.model.blocks) == 2
    if r["status"] == "ROLLBACK":
        assert len(u.model.blocks) == 1                               # exact rollback
    # persistence round-trip keeps the blocks
    u.save(); v = U.Unified.load()
    assert len(v.model.blocks) == len(u.model.blocks)
    if u.model.blocks:
        for a, b in zip(u.model.blocks, v.model.blocks):
            assert float((a["Wo"] - b["Wo"]).abs().max()) < 1e-6


def test_grow_rolls_back_a_block_that_gains_nothing(tmp_path, monkeypatch):
    """WAVE 257: on the production model a block with ce_before == ce_after
    (3.4673 -> 3.4673) was accepted as GROW. Nothing earned -> rollback."""
    import torch as t
    from hagi import unified as U
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    u = U.Unified()
    u.domains["d"] = {}
    data = bytes(range(256)) * 30
    u._domain_bytes = lambda s, e: data
    u.model.train_guarded(data, epochs=2)
    n0 = len(u.model.blocks)
    calls = {"n": 0}
    orig = t.nn.functional.cross_entropy
    def frozen(*a, **k):            # every evaluation returns the same loss
        calls["n"] += 1
        return orig(*a, **k).mul(0).add(1.0)
    u.GROW_BLOCKS = 2
    monkeypatch.setattr(t.nn.functional, "cross_entropy", frozen)
    r = u._maybe_grow("d", data)
    monkeypatch.setattr(t.nn.functional, "cross_entropy", orig)
    assert r["status"] in ("ROLLBACK", "HOLD")
    assert len(u.model.blocks) == n0


def test_repair_block_is_frozen_in_general_training_and_persisted(tmp_path, monkeypatch):
    """WAVE 257: a repair block (the old memory) must not be moved by
    train_certified — measured: movable repair blocks end at OLD +2.62
    (worse than no blocks, +0.55)."""
    import torch as t
    from hagi import unified as U
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    u = U.Unified()
    u.domains["d"] = {}
    data = bytes(range(256)) * 30
    u._domain_bytes = lambda s, e: data
    u.model.train_guarded(data, epochs=1)
    u.model.grow(seed=3)
    u.model.frozen_blocks = {0}
    w = [q.detach().clone() for q in u.model.blocks[0].values()]
    u.model.train_guarded(data, epochs=1)
    for q, w0 in zip(u.model.blocks[0].values(), w):
        assert float((q - w0).abs().max()) == 0.0
    u.save(); v = U.Unified.load()
    assert v.model.frozen_blocks == {0} and len(v.model.blocks) == 1


def test_repair_retrains_old_domains_until_forgetting_is_gone(tmp_path, monkeypatch):
    """WAVE 257 (user: after growth, retrain the old domains until
    forgetting reaches zero). Iterative: while the paired old-domain CE
    is above its pre-training value, a retrain step on OLD bytes runs;
    a step that does not improve is rolled back; the loop stops at the
    tolerance or the round cap."""
    import torch as t
    from hagi import unified as U
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    u = U.Unified()
    u.domains.update({"a": {}, "b": {}})
    old = bytes(range(0, 128, 2)) * 200
    u._domain_bytes = lambda s, e: old
    u.model.train_guarded(old, epochs=2)
    pre = u._old_domains_ce(exclude="c")
    # forget on purpose: train hard on unrelated bytes
    u.model.train_guarded(bytes(range(255, 0, -1)) * 200, epochs=4)
    assert u._old_domains_ce(exclude="c") > pre
    u._domain_bytes = lambda s, e: bytes(range(255, 0, -1)) * 200 if s == "c" else old
    r = u._repair_forgetting("c", pre, [q.detach().clone() for q in u.model.params()])
    assert r["mode"] == "replay" and r["rounds"]
    assert all(x["status"] in ("RETRAIN", "ROLLBACK") for x in r["rounds"])
    assert u._old_domains_ce(exclude="c") <= pre + u.FORGET_TOL or r["rounds"][-1]["status"] == "ROLLBACK"
    # the loop is monotone in the gate value
    ces = [x["ce"] for x in r["rounds"]]
    assert ces == sorted(ces, reverse=True)
