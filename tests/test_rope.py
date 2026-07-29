"""Tests for RoPE (1D and 2D) and rotate_half."""

from __future__ import annotations

import pytest
import torch

from hagi.model.rope import (
    RotaryEmbedding, apply_rope, rope_cos_sin, rope_cos_sin_2d, rotate_half,
)


class TestRotaryEmbedding:
    def test_inv_freq_shape(self):
        assert RotaryEmbedding(64).inv_freq.shape == (32,)

    def test_cos_sin_shape(self):
        cos, sin = RotaryEmbedding(64)(torch.arange(16), torch.device("cpu"), torch.float32)
        assert cos.shape == (16, 64)
        assert sin.shape == (16, 64)

    def test_cos_sin_in_range(self):
        cos, sin = RotaryEmbedding(64)(torch.arange(8), torch.device("cpu"), torch.float32)
        assert (cos >= -1).all() and (cos <= 1).all()

    def test_caching(self):
        rope = RotaryEmbedding(32)
        pos = torch.tensor([0, 1, 2])
        cos1, _ = rope(pos, torch.device("cpu"), torch.float32)
        cos2, _ = rope(pos, torch.device("cpu"), torch.float32)
        assert torch.equal(cos1, cos2)

    def test_rejects_odd_head_dim(self):
        with pytest.raises(ValueError, match="even"):
            RotaryEmbedding(63)


class TestRotateHalf:
    def test_rotate_half(self):
        x = torch.tensor([1., 2., 3., 4.])
        assert torch.equal(rotate_half(x), torch.tensor([-3., -4., 1., 2.]))

    def test_batched(self):
        x = torch.randn(2, 8, 32)
        x1, x2 = x.chunk(2, dim=-1)
        assert torch.equal(rotate_half(x), torch.cat([-x2, x1], dim=-1))


class TestApplyRoPE:
    def test_shape_preserved(self):
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 2, 16, 32)
        cos, sin = rope_cos_sin(torch.arange(16).float(), 32, 10000., torch.device("cpu"), torch.float32)
        qr, kr = apply_rope(q, k, cos, sin)
        assert qr.shape == q.shape and kr.shape == k.shape

    def test_different_from_input(self):
        q = torch.ones(1, 2, 4, 16)
        k = torch.ones(1, 2, 4, 16)
        cos, sin = rope_cos_sin(torch.arange(4).float(), 16, 10000., torch.device("cpu"), torch.float32)
        qr, _ = apply_rope(q, k, cos, sin)
        assert not torch.allclose(qr, q)


class TestRope2D:
    def test_shape(self):
        rows = torch.tensor([0., 0., 1., 1.])
        cols = torch.tensor([0., 1., 0., 1.])
        cos, sin = rope_cos_sin_2d(rows, cols, 64, 10000., torch.device("cpu"), torch.float32)
        assert cos.shape == (4, 64) and sin.shape == (4, 64)

    def test_rejects_hd_not_div4(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            rope_cos_sin_2d(torch.arange(4).float(), torch.arange(4).float(), 62, 10000., torch.device("cpu"), torch.float32)
