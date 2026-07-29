"""Tests for HebbianBilinearFFN."""

from __future__ import annotations

import torch

from hagi.model.hebbian_ffn import HebbianBilinearFFN, HebbianFFNConfig


def _ffn(h=64):
    return HebbianBilinearFFN(h, HebbianFFNConfig(expansion=4), use_ternary=False)


class TestHebbianBilinearFFN:
    def test_output_shape(self):
        assert _ffn()(torch.randn(2, 16, 64)).shape == (2, 16, 64)

    def test_not_identity(self):
        x = torch.randn(2, 16, 64)
        assert not torch.allclose(_ffn()(x), x)

    def test_gate_init_zero(self):
        assert (_ffn().gate == 0).all()

    def test_no_ternary_uses_linear(self):
        from torch import nn
        ffn = HebbianBilinearFFN(32, HebbianFFNConfig(expansion=2), use_ternary=False)
        assert isinstance(ffn.A0, nn.Linear)

    def test_ternary_uses_bitlinear(self):
        from hagi.model.ternary import BitLinear
        ffn = HebbianBilinearFFN(32, HebbianFFNConfig(expansion=2), use_ternary=True)
        assert isinstance(ffn.A0, BitLinear)

    def test_gate_gets_gradient(self):
        ffn = _ffn(32)  # match tensor shape
        ffn(torch.randn(2, 8, 32, requires_grad=True)).sum().backward()
        assert ffn.gate.grad is not None and ffn.gate.grad.abs().sum() > 0
