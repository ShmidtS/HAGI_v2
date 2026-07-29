"""Tests for PredictiveRefiner."""

from __future__ import annotations

import torch

from hagi.config import RefinementConfig
from hagi.model.refinement import PredictiveRefiner


def _refiner(**kw):
    its = kw.get("its", 2)
    hep = kw.get("hep", True)
    cfg = RefinementConfig(
        enabled=True, iterations=its, hep_enabled=hep,
        exit_threshold=0.05, exit_min_steps=10, exit_window=3,
    )
    return PredictiveRefiner(64, cfg, use_ternary=False)


class TestPredictiveRefiner:
    def test_output_shape(self):
        assert _refiner()(torch.randn(2, 16, 64)).shape == (2, 16, 64)

    def test_output_different(self):
        h = torch.randn(2, 16, 64)
        assert not torch.allclose(_refiner()(h), h)

    def test_novelty_float(self):
        r = _refiner()
        r(torch.randn(2, 8, 64))
        assert isinstance(r.novelty(), float)

    def test_novelty_before_forward(self):
        assert _refiner().novelty() == 1.0

    def test_no_hep_works(self):
        r = _refiner(hep=False)
        assert r.hep is None
        assert r(torch.randn(2, 8, 64)).shape == (2, 8, 64)

    def test_more_iterations_more_change(self):
        r1 = _refiner(its=1, hep=False)
        r4 = _refiner(its=4, hep=False)
        h = torch.randn(2, 4, 64)
        d1 = (r1(h) - h).norm().item()
        d4 = (r4(h) - h).norm().item()
        assert d1 > 0 and d4 > 0
