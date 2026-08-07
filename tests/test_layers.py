"""Normalization, RoPE, and the SwiGLU mixer.

RMSNorm's fp32 variance accumulator is asserted directly: at bf16 the 7-bit
mantissa makes ``mean(x^2)`` lossy enough to bias the gain, and a biased gain is
a multiplicative distortion applied to every downstream matmul.

RoPE's defining property — the inner product depends only on the position
*difference* — is what makes it a rotation rather than an additive signal, so it
is asserted rather than assumed.
"""

from __future__ import annotations

import math

import pytest
import torch

from hagi.model.ffn import BranchScale, FeedForward, SwiGLU
from hagi.model.norms import HeadNorm, RMSNorm
from hagi.model.rope import (
    RotaryEmbedding,
    apply_rope,
    rope_cos_sin,
    rope_cos_sin_2d,
    rotate_half,
    rotate_pairs,
)
from tests.conftest import assert_finite


class TestRMSNorm:
    def test_unit_rms_output(self):
        out = RMSNorm(64)(torch.randn(2, 5, 64) * 7.0)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_scale_invariant(self):
        norm = RMSNorm(32)
        x = torch.randn(1, 4, 32)
        assert torch.allclose(norm(x), norm(x * 100.0), atol=1e-4)

    def test_gain_is_applied(self):
        norm = RMSNorm(16)
        with torch.no_grad():
            norm.weight.fill_(2.0)
        x = torch.randn(1, 3, 16)
        assert torch.allclose(norm(x), 2.0 * RMSNorm(16)(x), atol=1e-5)

    def test_variance_is_computed_in_fp32(self):
        """bf16 ``mean(x^2)`` biases the gain, and the bias multiplies every matmul."""
        norm = RMSNorm(256)
        x = (torch.randn(1, 4, 256) * 0.01).bfloat16()
        with torch.no_grad():
            out = norm(x)
            reference = norm(x.float())
        assert out.dtype == torch.bfloat16
        assert float((out.float() - reference).abs().max()) < 0.05

    def test_dtype_round_trip(self):
        assert RMSNorm(8)(torch.randn(1, 2, 8).bfloat16()).dtype == torch.bfloat16

    def test_zero_input_is_finite(self):
        assert_finite(RMSNorm(8)(torch.zeros(1, 2, 8)), "rmsnorm of zeros")

    def test_marked_for_fp32(self):
        assert RMSNorm(8).keep_fp32 is True


class TestHeadNorm:
    def test_normalizes_the_last_dimension(self):
        out = HeadNorm(32)(torch.randn(2, 4, 6, 32) * 5.0)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_one_gain_shared_across_heads(self):
        norm = HeadNorm(16)
        assert norm.weight.shape == (16,)

    def test_marked_for_fp32(self):
        assert HeadNorm(8).keep_fp32 is True


class TestRoPE:
    def test_rotate_half(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))

    def test_norm_is_preserved(self):
        """A rotation consumes no amplitude — that is why it beats an additive signal."""
        q = torch.randn(1, 2, 6, 32)
        k = torch.randn(1, 2, 6, 32)
        cos, sin = rope_cos_sin(torch.arange(6).float(), 32, 10000.0, q.device, q.dtype)
        rq, rk = apply_rope(q, k, cos, sin)
        assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
        assert torch.allclose(rk.norm(dim=-1), k.norm(dim=-1), atol=1e-4)

    def test_inner_product_depends_only_on_the_offset(self):
        head_dim = 32
        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)

        def score(p_q: int, p_k: int) -> float:
            cos_q, sin_q = rope_cos_sin(
                torch.tensor([float(p_q)]), head_dim, 10000.0, q.device, q.dtype
            )
            cos_k, sin_k = rope_cos_sin(
                torch.tensor([float(p_k)]), head_dim, 10000.0, q.device, q.dtype
            )
            rq, _ = apply_rope(q, q, cos_q, sin_q)
            rk, _ = apply_rope(k, k, cos_k, sin_k)
            return float((rq * rk).sum())

        assert score(5, 3) == pytest.approx(score(11, 9), abs=1e-4)
        assert score(5, 3) != pytest.approx(score(5, 4), abs=1e-3)

    def test_position_zero_is_the_identity(self):
        q = torch.randn(1, 1, 1, 16)
        cos, sin = rope_cos_sin(torch.zeros(1), 16, 10000.0, q.device, q.dtype)
        rotated, _ = apply_rope(q, q, cos, sin)
        assert torch.allclose(rotated, q, atol=1e-6)

    def test_odd_head_dim_raises(self):
        with pytest.raises(ValueError):
            rope_cos_sin(torch.zeros(2), 15, 10000.0, torch.device("cpu"), torch.float32)

    def test_2d_needs_head_dim_divisible_by_four(self):
        with pytest.raises(ValueError):
            rope_cos_sin_2d(
                torch.zeros(2), torch.zeros(2), 18, 10000.0, torch.device("cpu"), torch.float32
            )

    def test_2d_bands_are_independent(self):
        """Row position must not leak into the column band."""
        rows = torch.tensor([0.0, 1.0])
        cols = torch.tensor([0.0, 0.0])
        cos_a, _ = rope_cos_sin_2d(rows, cols, 32, 10000.0, torch.device("cpu"), torch.float32)
        cos_b, _ = rope_cos_sin_2d(
            torch.zeros(2), cols, 32, 10000.0, torch.device("cpu"), torch.float32
        )
        assert torch.allclose(cos_a[:, 16:], cos_b[:, 16:])
        assert not torch.allclose(cos_a[:, :16], cos_b[:, :16])

    def test_rotate_pairs_preserves_norm_with_a_2d_table(self):
        x = torch.randn(1, 2, 4, 32)
        cos, sin = rope_cos_sin_2d(
            torch.arange(4).float(), torch.arange(4).float(), 32, 10000.0,
            torch.device("cpu"), torch.float32,
        )
        out = rotate_pairs(x, cos, sin)
        assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


class TestRotaryEmbedding:
    def test_matches_the_functional_form(self):
        module = RotaryEmbedding(32, 10000.0)
        positions = torch.arange(6)
        cos, sin = module(positions, torch.device("cpu"), torch.float32)
        ref_cos, ref_sin = rope_cos_sin(
            positions.float(), 32, 10000.0, torch.device("cpu"), torch.float32
        )
        assert torch.allclose(cos, ref_cos, atol=1e-6)
        assert torch.allclose(sin, ref_sin, atol=1e-6)

    def test_cache_is_keyed_by_the_range_not_the_length(self):
        """Prefill and decode ask for different ranges of the same length."""
        module = RotaryEmbedding(16)
        early = module(torch.arange(0, 1), torch.device("cpu"), torch.float32)
        late = module(torch.arange(9, 10), torch.device("cpu"), torch.float32)
        assert not torch.allclose(early[0], late[0])

    def test_cache_hit_returns_the_same_object(self):
        module = RotaryEmbedding(16)
        first = module(torch.arange(4), torch.device("cpu"), torch.float32)
        second = module(torch.arange(4), torch.device("cpu"), torch.float32)
        assert first[0] is second[0]

    def test_cache_is_bounded(self):
        module = RotaryEmbedding(16)
        for start in range(40):
            module(torch.arange(start, start + 2), torch.device("cpu"), torch.float32)
        assert len(module._cache) <= 17

    def test_empty_positions(self):
        cos, sin = RotaryEmbedding(16)(
            torch.zeros(0, dtype=torch.long), torch.device("cpu"), torch.float32
        )
        assert cos.shape == (0, 16)

    def test_odd_head_dim_raises(self):
        with pytest.raises(ValueError):
            RotaryEmbedding(15)


class TestBranchScale:
    def test_init_and_clamp(self):
        bs = BranchScale(0.2, clamp_ratio=2.0)
        assert bs.scale.item() == pytest.approx(0.2)
        assert bs.scale.clamp(0.1, 0.4).item() == pytest.approx(0.2)

    def test_forward_scales_by_the_gain(self):
        bs = BranchScale(0.5, clamp_ratio=2.0)
        x = torch.randn(2, 3, 4)
        assert torch.allclose(bs(x), x * 0.5, atol=1e-6)

    def test_keeps_fp32_marker(self):
        assert BranchScale(0.1).keep_fp32 is True

    def test_not_a_channel_weight(self):
        """BranchScale is a scalar gain — AdamW, not Muon."""
        assert getattr(BranchScale(0.1), "is_channel_weight", False) is False


class TestSwiGLU:
    def test_shape_and_finiteness(self):
        out = SwiGLU(32, 64, use_ternary=False)(torch.randn(2, 5, 32))
        assert out.shape == (2, 5, 32)
        assert_finite(out, "swiglu output")

    def test_no_internal_norm_or_residual(self):
        """SwiGLU is only the nonlinear branch body.

        The branch cap is a scalar gain, not normalization or a residual.
        """
        mixer = SwiGLU(32, 64, use_ternary=False)
        names = {n for n, _ in mixer.named_modules() if n}
        assert names == {"gate", "up", "down", "branch_scale"}

    def test_residual_scale_shrinks_the_down_projection(self):
        wide = SwiGLU(32, 64, use_ternary=False, residual_scale=1.0)
        narrow = SwiGLU(32, 64, use_ternary=False, residual_scale=0.1)
        assert float(narrow.down.weight.detach().std()) < float(wide.down.weight.detach().std())

    def test_gate_attenuates(self):
        """The gate is a learned per-channel power control, so a closed gate zeroes."""
        mixer = SwiGLU(8, 16, use_ternary=False)
        with torch.no_grad():
            mixer.gate.weight.fill_(0.0)  # silu(0) = 0
            out = mixer(torch.randn(1, 3, 8))
        assert float(out.abs().max()) == 0.0

    def test_channel_weights_are_marked(self):
        mixer = SwiGLU(8, 16, use_ternary=False)
        for module in (mixer.gate, mixer.up, mixer.down):
            assert module.is_channel_weight is True


class TestFeedForward:
    def test_shape(self):
        out = FeedForward(32, 64, use_ternary=False)(torch.randn(2, 4, 32))
        assert out.shape == (2, 4, 32)

    def test_normalizes_its_input(self):
        ffn = FeedForward(32, 64, use_ternary=False).eval()
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            assert torch.allclose(ffn(x), ffn(x * 50.0), atol=1e-4)

    def test_expansion_matches_a_4x_mlp_budget(self):
        """8/3 SwiGLU: three matrices cost what a 4x two-matrix MLP's two do."""
        h = 512
        swiglu = 3 * h * round(8 / 3 * h)
        mlp = 2 * h * (4 * h)
        assert abs(swiglu - mlp) / mlp < 0.01


def test_ln_vocab_reference():
    """Guard on the baseline the whole project measures against."""
    assert math.log(262144) == pytest.approx(12.4766, abs=1e-4)
