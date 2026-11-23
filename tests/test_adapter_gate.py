"""WAVE 246: conditional domain adapter (paired validation gate)."""
import numpy as np
import pytest
import torch

from hagi import unified as U
from hagi.trunk_ops import LowRankAdapter, paired_gain


def test_paired_gain_positive_only_when_clearly_better():
    rng = np.random.default_rng(0)
    before = rng.normal(2.0, 0.1, 256)
    _, lower = paired_gain(before, before - 0.05)      # clear gain
    assert lower > 0
    _, lower = paired_gain(before, before + 0.05)      # clear loss
    assert lower < 0
    gain, lower = paired_gain(before, before + 0.001)  # noise
    assert abs(gain) < 0.005 and lower < 0
    with pytest.raises(ValueError):
        paired_gain([1.0], [float("nan"), 2.0])


def _unified(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    u = U.Unified()
    u.domains["d"] = {}
    u._domain_bytes = lambda s, e: bytes(range(256)) * 8   # p8 law
    return u


def test_adapter_rejects_noise_holds_identity(tmp_path, monkeypatch):
    u = _unified(tmp_path, monkeypatch)
    u._domain_bytes = lambda s, e: np.random.default_rng(1).integers(
        0, 256, 40_000, dtype=np.uint8).tobytes()
    r = u._fit_domain_adapter("d")
    assert r["status"] == "HOLD" and not u.adapter_bank.get("d").B.any()


def test_adapter_grows_on_structure(tmp_path, monkeypatch):
    u = _unified(tmp_path, monkeypatch)
    # WAVE 256 (lr 3e-3 -> 1e-3): the same displacement budget as the
    # old epochs=3 at 3e-3 (measured: at lr=1e-3 epochs=3 -> HOLD,
    # epochs=6 -> GROW). The test is about structure, not epoch count.
    u.model.train_guarded(bytes(range(256)) * 8, epochs=9)
    trunk = [q.detach().clone() for q in u.model.params()]
    r = u._fit_domain_adapter("d")
    assert r["status"] == "GROW" and r["rank"] in (1, 2, 4)
    ad = u.adapter_bank.get("d")
    assert ad.n_params == 2 * ad.r * ad.d
    flat = ad.params().copy()
    ad.set_params(flat)                                  # rank roundtrip
    assert np.allclose(ad.params(), flat)
    # trunk untouched by adapter fitting
    for q, q0 in zip(u.model.params(), trunk):      # frozen trunk
        assert torch.equal(q.detach(), q0)
