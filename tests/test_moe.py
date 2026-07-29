"""Tests for WaterFillingMoE: routing, dispatch, aux losses."""

from __future__ import annotations

import pytest
import torch

from hagi.config import MoEConfig
from hagi.model.moe import WaterFillingMoE


@pytest.fixture
def moe():
    cfg = MoEConfig(enabled=True, num_experts=4, top_k=1, n_shared=1, moe_every=2, intermediate_size=128)
    return WaterFillingMoE(64, 128, cfg, use_ternary=False)


class TestWaterFillingMoE:
    def test_output_shape(self, moe):
        assert moe(torch.randn(2, 16, 64)).shape == (2, 16, 64)

    def test_not_identity(self, moe):
        x = torch.randn(2, 16, 64)
        assert not torch.allclose(moe(x), x)

    def test_expert_counts(self, moe):
        assert len(moe.shared_experts) == 1 and len(moe.experts) == 4

    def test_router_shape(self, moe):
        assert moe.router.weight.shape == (4, 64)

    def test_lb_training(self, moe):
        moe.train()
        moe(torch.randn(2, 16, 64))
        lb = moe.last_load_balance
        assert lb is not None and 0.0 < lb.item() <= 4.0

    def test_lb_none_eval(self, moe):
        moe.eval()
        moe(torch.randn(2, 16, 64))
        assert moe.last_load_balance is None

    def test_route_entropy_training(self, moe):
        moe.train()
        moe(torch.randn(2, 16, 64))
        assert moe.last_routing_entropy is not None and moe.last_routing_entropy.item() > 0.0

    def test_wf_loss_training(self, moe):
        moe.train()
        moe(torch.randn(2, 16, 64))
        assert moe.last_water_filling_loss is not None

    def test_commit_ema_noop(self, moe):
        moe.commit_ema_update()

    def test_commit_ema_clears_deferred(self, moe):
        moe.train()
        moe(torch.randn(2, 8, 64))
        assert moe._deferred_residual is not None
        moe.commit_ema_update()
        assert moe._deferred_residual is None

    def test_top2_routing(self):
        cfg = MoEConfig(enabled=True, num_experts=4, top_k=2, n_shared=0, moe_every=1, intermediate_size=64)
        moe2 = WaterFillingMoE(32, 64, cfg, use_ternary=False)
        moe2.eval()
        assert moe2(torch.randn(2, 16, 32)).shape == (2, 16, 32)
