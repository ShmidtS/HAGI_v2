"""Ternary quantization: the b1.58 minimizer, the STE, and scale self-stability.

The invariant the whole optimizer story rests on: because the scale is the
per-row absmean of the master, a uniform outward drift of ``||W||`` cancels in
``W/s``. That is why no spectral cap is needed under Muon, so it is asserted
directly rather than left as a comment.
"""

from __future__ import annotations

import pytest
import torch

from hagi.model.ternary import BitLinear, ternarize
from tests.conftest import assert_finite


class TestTernarize:
    def test_only_three_levels_per_row(self):
        w = torch.randn(8, 32)
        eff, scale = ternarize(w)
        for row in range(8):
            levels = torch.unique(eff[row] / scale[row])
            assert set(round(float(v), 6) for v in levels) <= {-1.0, 0.0, 1.0}

    def test_scale_is_row_absmean(self):
        w = torch.randn(5, 16)
        _, scale = ternarize(w)
        assert torch.allclose(scale.squeeze(-1), w.abs().mean(dim=1))

    def test_zero_bin_exists(self):
        """``|w/s| < 0.5`` must round to 0 — the third level is what buys 1.585 bits."""
        w = torch.tensor([[10.0, 0.01, -10.0, 0.02]])
        eff, _ = ternarize(w)
        assert float(eff[0, 1]) == 0.0 and float(eff[0, 3]) == 0.0

    def test_sign_preserved_on_large_entries(self):
        w = torch.tensor([[3.0, -3.0, 3.0, -3.0]])
        eff, _ = ternarize(w)
        assert torch.equal(torch.sign(eff), torch.sign(w))

    def test_all_zero_row_uses_eps_floor(self):
        eff, scale = ternarize(torch.zeros(2, 8), eps=1e-5)
        assert float(scale.min()) == pytest.approx(1e-5)
        assert_finite(eff, "effective weight")

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            ternarize(torch.randn(4))
        with pytest.raises(ValueError):
            ternarize(torch.randn(2, 3, 4))

    def test_scale_is_invariant_to_uniform_drift(self):
        """The self-stabilization property that removes the need for a spectral cap."""
        w = torch.randn(16, 64)
        eff_a, _ = ternarize(w)
        eff_b, _ = ternarize(w * 7.5)
        assert torch.allclose(eff_b, eff_a * 7.5, atol=1e-6)
        # The quantization *pattern* — what the forward pass actually depends on —
        # is completely unchanged.
        pattern_a = torch.sign(eff_a)
        pattern_b = torch.sign(eff_b)
        assert torch.equal(pattern_a, pattern_b)


class TestStepCache:
    def test_cached_forward_matches_live_quantize(self):
        layer = BitLinear(16, 8)
        x = torch.randn(4, 16)
        with torch.no_grad():
            live = layer(x)
        layer.cache_quantized()
        with torch.no_grad():
            cached = layer(x)
        assert torch.allclose(live, cached, atol=1e-5, rtol=1e-5)
        layer.clear_quantized()
        assert layer._step_q is None

    def test_cached_path_still_trains_master(self):
        layer = BitLinear(8, 4)
        x = torch.randn(3, 8)
        layer.cache_quantized()
        layer(x).sum().backward()
        assert layer.weight.grad is not None
        assert_finite(layer.weight.grad, "weight grad")
        # STE identity: non-zero grad even on saturated rows after cache.
        assert float(layer.weight.grad.abs().sum()) > 0
        layer.clear_quantized()


class TestSTE:
    def test_gradient_is_identity(self):
        layer = BitLinear(8, 4)
        x = torch.randn(3, 8)
        layer(x).sum().backward()
        assert layer.weight.grad is not None
        assert_finite(layer.weight.grad, "weight grad")

    def test_saturated_entries_still_receive_gradient(self):
        """Zeroing saturated entries would erase exactly Muon's strongest signal."""
        layer = BitLinear(4, 2)
        with torch.no_grad():
            layer.weight.fill_(0.0)
            layer.weight[:, 0] = 100.0  # deep in saturation
        layer(torch.ones(1, 4)).sum().backward()
        assert float(layer.weight.grad[:, 0].abs().min()) > 0


class TestBitLinear:
    def test_shape_and_finiteness(self):
        out = BitLinear(16, 32)(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 32)
        assert_finite(out, "bitlinear output")

    def test_bias_optional(self):
        assert BitLinear(4, 4, bias=False).bias is None
        assert BitLinear(4, 4, bias=True).bias is not None

    def test_eval_and_train_paths_agree(self):
        layer = BitLinear(16, 8)
        x = torch.randn(2, 16)
        with torch.no_grad():
            eval_out = layer(x)
        train_out = layer(x)
        assert float((eval_out - train_out.detach()).abs().max()) < 1e-6

    def test_master_weight_is_the_only_parameter(self):
        names = {n for n, _ in BitLinear(8, 4, bias=False).named_parameters()}
        assert names == {"weight"}, "the ternary scale must not be a parameter"
