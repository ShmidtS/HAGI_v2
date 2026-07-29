"""Tests for WaterFillingAllocator."""

from __future__ import annotations

import pytest
import torch

from hagi.model.water_filling import WaterFillingAllocator


@pytest.fixture
def alloc():
    return WaterFillingAllocator(512, 4, min_width=16)


class TestWaterFillingAllocator:
    def test_init_snr_ones(self, alloc):
        assert torch.equal(alloc.snr_ema, torch.ones(4))

    def test_init_logits_zeros(self, alloc):
        assert torch.equal(alloc.allocation_logits, torch.zeros(4))

    def test_uniform_at_init(self, alloc):
        assert torch.allclose(alloc.allocation_probs(), torch.full((4,), 0.25), atol=1e-4)

    def test_probs_sum_to_one(self, alloc):
        assert abs(alloc.allocation_probs().sum().item() - 1.0) < 1e-5

    def test_get_widths_sum(self, alloc):
        widths = alloc.get_widths()
        assert len(widths) == 4 and sum(widths) == 512 and all(w >= 16 for w in widths)

    def test_reg_loss_zero_at_init(self, alloc):
        assert abs(alloc.regularization_loss().item()) < 1e-4

    def test_update_snr_ema(self, alloc):
        alloc.update_snr_ema(torch.tensor([1., 2., 4., 8.]), decay=0.)
        assert torch.allclose(alloc.snr_ema, 1. / torch.tensor([1., 2., 4., 8.]), atol=1e-4)

    def test_high_snr_gets_more(self, alloc):
        alloc.update_snr_ema(torch.tensor([0.5, 1., 2., 4.]), decay=0.)
        probs = alloc.allocation_probs()
        assert probs[0] > probs[3]

    def test_rejects_narrow_total(self):
        with pytest.raises(ValueError):
            WaterFillingAllocator(10, 4, min_width=32)

    def test_rejects_zero_experts(self):
        with pytest.raises(ValueError, match=">= 1"):
            WaterFillingAllocator(100, 0)

    def test_rejects_zero_temp(self):
        with pytest.raises(ValueError, match="positive"):
            WaterFillingAllocator(100, 4, temperature=0.)

    def test_min_width_auto(self):
        assert WaterFillingAllocator(1024, 4, min_width=0).min_width >= 16
