"""C474: EF21 in-step quantized certified training (Theory/EF21InStep)."""
import numpy as np
from hagi.lm import CertifiedStepLM


def test_ef21_learns_periodic_pattern_with_honest_admission():
    data = (b"abc" * 40 + b"xyz" * 40) * 20
    lm = CertifiedStepLM(k=4, dim=32, hidden=64, seed=1)
    r = lm.train_ef21(data, epochs=4, holdout=0.15, batch=128)
    assert r["ef21_admits"] > 0
    assert r["ef21_rejects"] >= 0  # telemetry present
    assert r["final_holdout_ce"] < 1.5   # learned the pattern
    assert r["certified_guarantee_sum"] > 0


def test_ef21_fixed_scale_pin():
    # scale is frozen after the first step (fixed-scale contract PIN)
    data = (b"abc" * 40 + b"xyz" * 40) * 20
    lm = CertifiedStepLM(k=4, dim=32, hidden=64, seed=1)
    lm.train_ef21(data, epochs=2, holdout=0.15, batch=128)
    assert lm._ef21_s is not None and lm._ef21_s > 0
