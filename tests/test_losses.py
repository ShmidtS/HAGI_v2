"""Tests for LossAggregator."""

from __future__ import annotations

import pytest
import torch

from hagi.config import Config
from hagi.model.outputs import AuxLosses, ModelOutput
from hagi.train.losses import LossAggregator, selected_cross_entropy


class TestSelectedCE:
    def test_basic(self):
        assert selected_cross_entropy(torch.randn(4, 100), torch.randint(0, 100, (4,))).item() > 0.0

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shapes"):
            selected_cross_entropy(torch.randn(4, 100), torch.randint(0, 100, (4, 3)))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            selected_cross_entropy(torch.randn(0, 100), torch.randint(0, 100, (0,)))


@pytest.fixture
def agg():
    return LossAggregator(Config())


def _out(ce=2.0, **aux_vals):
    aux = AuxLosses(**aux_vals)
    return ModelOutput(logits=torch.randn(16, 1000), hidden=torch.randn(2, 8, 64), aux=aux, ce_loss=torch.tensor(ce))


class TestLossAggregator:
    def test_ce_only(self, agg):
        assert abs(agg(_out(3.0), step=0).item() - 3.0) < 1e-3

    def test_raises_without_ce(self, agg):
        with pytest.raises(ValueError, match="ce_loss"):
            agg(ModelOutput(logits=None, hidden=torch.randn(2, 8, 64), aux=AuxLosses()), step=0)

    def test_adds_rate(self, agg):
        assert abs(agg(_out(2.0, rate=torch.tensor(1.0)), step=0).item() - 2.01) < 1e-3

    def test_distortion_beta_anneal(self, agg):
        loss0 = agg(_out(2.0, distortion=torch.tensor(10.0)), step=0).item()
        assert abs(loss0 - 2.0) < 1e-3
        warmup = agg.warmup_steps
        loss_w = agg(_out(2.0, distortion=torch.tensor(10.0)), step=warmup).item()
        assert abs(loss_w - 2.1) < 1e-3

    def test_subtracts_routing_entropy(self, agg):
        assert abs(agg(_out(2.0, route_entropy=torch.tensor(1.0)), step=0).item() - 1.99) < 1e-3

    def test_exit_not_halted_initially(self, agg):
        assert not agg.exit_halted
