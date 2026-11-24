"""WAVE 261: HNetByteLM (torch port of the "HAGI v2 Optimal" byte LM)
behind the same interfaces as RopeAttentionByteLM."""
import numpy as np
import pytest


def test_hnet_factory_shapes_and_params():
    from hagi.lm import make_lm
    m = make_lm("hnet", seed=1)
    assert m.ARCH == "hnet" and m.k == 64
    out = m.logits(m.t.as_tensor([[0] * 64], dtype=m.t.long))
    assert tuple(out.shape) == (1, 256)
    # WAVE 262 measured default d=64 h=128 (105 088 params); the d=24
    # variant of the report has 28 048
    assert (m.dim, m.hidden, sum(p.numel() for p in m.params())) == (64, 128, 105088)
    small = make_lm("hnet", dim=24, hidden=48, seed=1)
    assert sum(p.numel() for p in small.params()) == 28048
    assert make_lm("rope", seed=1).ARCH == "rope"


def test_hnet_trains_and_beats_its_init_on_a_real_byte_stream():
    from hagi.lm import make_lm
    data = bytes(range(0, 256, 3)) * 60
    m = make_lm("hnet", seed=1)
    r = m.train_guarded(data, epochs=2)
    assert r["best_holdout_ce"] < np.log(256) - 0.5


def test_hnet_unified_roundtrip_save_load_and_ingest(tmp_path, monkeypatch):
    from hagi import unified as U
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    monkeypatch.setattr(U.Unified, "ARCH", "hnet")
    u = U.Unified()
    assert type(u.model).__name__ == "HNetByteLM"
    (tmp_path / "domains").mkdir()
    data = bytes(range(0, 256, 5)) * 900
    (tmp_path / "domains" / "a.bin").write_bytes(data)
    r = u.ingest("a", data, epochs=1, repeat=1)
    assert r["holdout_ce"] < np.log(256)
    u.save()
    v = U.Unified.load()
    assert type(v.model).__name__ == "HNetByteLM" and "a" in v.domains
    for a, b in zip(u.model.params(), v.model.params()):
        assert float((a.detach().cpu() - b.detach().cpu()).abs().max()) == 0.0


def test_fresh_model_keeps_the_domain_book(tmp_path, monkeypatch):
    """WAVE 262: Unified() on a fresh checkpoint must not start blind."""
    import json
    from hagi import unified as U
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    (tmp_path / "unified_ledger.json").write_text(json.dumps({"x": {"sha": "s", "bytes": 9, "ce": None}}), encoding="utf-8")
    u = U.Unified()
    assert "x" in u.domains
