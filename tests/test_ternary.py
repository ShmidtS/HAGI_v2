"""Tests for BitNet b1.58: ternarize(), BitLinear, _TernarizeSTE."""

from __future__ import annotations

import pytest
import torch

from hagi.model.ternary import ternarize, BitLinear, _TernarizeSTE
from tests.conftest import assert_finite


class TestTernarize:
    def test_values_are_ternary(self):
        w = torch.randn(16, 32)
        eff, scale = ternarize(w)
        assert_finite(eff, "eff")
        assert_finite(scale, "scale")
        assert eff.shape == w.shape
        assert scale.shape == (16, 1)
        for i in range(16):
            s = scale[i, 0]
            row = eff[i]
            assert ((row == 0) | (row == s) | (row == -s)).all()

    def test_scale_positive(self):
        _, scale = ternarize(torch.randn(8, 64) * 0.5, eps=1e-5)
        assert_finite(scale, "scale")
        assert (scale > 0).all()

    def test_constant_input(self):
        w = torch.ones(4, 8) * 3.0
        eff, scale = ternarize(w)
        assert_finite(eff, "eff")
        assert_finite(scale, "scale")
        assert abs(scale[0, 0].item() - 3.0) < 1e-3
        assert (eff == 3.0).all()

    def test_zero_row(self):
        eff, scale = ternarize(torch.zeros(4, 8), eps=1e-5)
        assert_finite(eff, "eff")
        assert_finite(scale, "scale")
        assert (eff == 0).all()
        assert (scale == 1e-5).all()

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D"):
            ternarize(torch.randn(16))

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            ternarize(torch.randn(2, 3, 4))


class TestBitLinear:
    def test_forward_shape(self):
        y = BitLinear(32, 16)(torch.randn(4, 32))
        assert_finite(y, "y")
        assert y.shape == (4, 16)

    def test_forward_with_bias(self):
        y = BitLinear(32, 16, bias=True)(torch.randn(4, 32))
        assert_finite(y, "y")
        assert y.shape == (4, 16)

    def test_gradient_flows(self):
        layer = BitLinear(32, 16)
        x = torch.randn(4, 32, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0

    def test_inference_no_grad(self):
        layer = BitLinear(32, 16)
        with torch.inference_mode():
            y = layer(torch.randn(4, 32))
        assert_finite(y, "y")
        assert not y.requires_grad

    def test_extra_repr(self):
        rep = BitLinear(32, 16, bias=True, eps=1e-4).extra_repr()
        assert "in_features=32" in rep
        assert "bias=True" in rep
        assert "ternary=BitNet-b1.58" in rep


class TestTernarizeSTE:
    def test_forward_matches_ternarize(self):
        w = torch.randn(8, 12)
        assert torch.equal(_TernarizeSTE.apply(w, 1e-5), ternarize(w, 1e-5)[0])

    def test_backward_identity(self):
        w = torch.randn(4, 6, requires_grad=True)
        grad_out = torch.ones_like(w)
        _TernarizeSTE.apply(w, 1e-5).backward(grad_out)
        assert torch.allclose(w.grad, grad_out)

    def test_no_mutation(self):
        w = torch.randn(4, 8)
        w_clone = w.clone()
        _TernarizeSTE.apply(w, 1e-5)
        assert torch.equal(w, w_clone)
