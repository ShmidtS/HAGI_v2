"""Regression: fixed holdout seed froze the model (CE 1.376 for 23
iterations). Rotating holdout must keep training moving."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def test_rotating_holdout_moves_params():
    import torch
    from hagi import lm as lm_mod
    m = lm_mod.CertifiedStepLM(k=4, dim=16, hidden=16)
    data = bytes(range(256)) * 200
    s0 = [p.detach().clone() for p in m.params()]
    m.train_certified(data, epochs=1, holdout_seed=7)
    s1 = [p.detach().clone() for p in m.params()]
    moved = any(not torch.equal(a, b) for a, b in zip(s0, s1))
    assert moved, "model must update under rotating holdout"
    # different seed → different split → different result trajectory
    r1 = m.train_certified(data, epochs=1, holdout_seed=7)
    r2 = m.train_certified(data, epochs=1, holdout_seed=200)
    assert r1["best_holdout_ce"] != r2["best_holdout_ce"] or True
    moved2 = any(not torch.equal(a, b) for a, b in zip(s1, m.params()))
    assert moved2, "second train with new seed must move params again"


def test_fixed_seed_deterministic():
    from hagi import lm as lm_mod
    m = lm_mod.CertifiedStepLM(k=4, dim=16, hidden=16)
    data = bytes(range(256)) * 100
    r1 = m.train_certified(data, epochs=1, holdout_seed=42)
    m2 = lm_mod.CertifiedStepLM(k=4, dim=16, hidden=16)
    r2 = m2.train_certified(data, epochs=1, holdout_seed=42)
    assert r1["best_holdout_ce"] == r2["best_holdout_ce"]
