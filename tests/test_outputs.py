"""Tests for AuxLosses and ModelOutput dataclasses."""

from __future__ import annotations

import torch

from hagi.model.outputs import AuxLosses, ModelOutput


class TestAuxLosses:
    def test_all_none_default(self):
        aux = AuxLosses()
        assert all(getattr(aux, f) is None
                   for f in ("rate", "distortion", "vicreg", "infonce", "moe_lb",
                             "route_entropy", "water_filling", "refinement", "attn_entropy"))

    def test_set_fields(self):
        t = torch.tensor(0.5)
        aux = AuxLosses(rate=t, distortion=t)
        assert aux.rate is not None and aux.vicreg is None


class TestModelOutput:
    def test_creation(self):
        out = ModelOutput(logits=None, hidden=torch.randn(2, 8, 64), aux=AuxLosses())
        assert out.hidden.shape == (2, 8, 64) and out.logits is None and out.ce_loss is None
