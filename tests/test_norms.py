"""Tests for RMSNorm."""

from __future__ import annotations

import torch

from hagi.model.norms import RMSNorm


class TestRMSNorm:
    def test_output_shape(self):
        y = RMSNorm(64)(torch.randn(4, 8, 64))
        assert y.shape == (4, 8, 64)

    def test_unit_variance(self):
        norm = RMSNorm(64)
        x = torch.randn(4, 8, 64) * 5.0
        y = norm(x)
        rms = torch.sqrt((y ** 2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_identity_ish(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 4, 32)
        x = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True))
        assert torch.allclose(norm(x), x, atol=0.1)

    def test_weight_scales(self):
        norm = RMSNorm(16)
        norm.weight.data = torch.full((16,), 2.0)
        assert torch.allclose(norm(torch.ones(2, 4, 16)), torch.full((2, 4, 16), 2.0))
