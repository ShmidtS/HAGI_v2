"""Tests for Attention: GQA modes, masks, penalty, sliding window."""

from __future__ import annotations

import pytest
import torch

from hagi.model.attention import Attention, AttentionConfig, repeat_kv


class TestRepeatKV:
    def test_single_rep(self):
        x = torch.randn(2, 4, 16, 64)
        y = repeat_kv(x, 1)
        assert y.shape == x.shape
        # With n_rep=1, it's the same tensor (identity)
        assert y is x or torch.equal(y, x)

    def test_multi_rep(self):
        x = torch.randn(2, 2, 16, 64)
        y = repeat_kv(x, 4)
        assert y.shape == (2, 8, 16, 64)
        # Same shape check is sufficient; repeat semantics define content


@pytest.fixture
def attn():
    # 4 heads * 32 dim = 128 = hidden_size
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=32, attn_entropy_floor=0.0)
    return Attention(128, cfg, use_ternary=False)


@pytest.fixture
def attn_with_penalty():
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=32, attn_entropy_floor=0.1)
    return Attention(128, cfg, use_ternary=False)


class TestAttentionModes:
    def test_causal(self, attn):
        out, pen = attn(torch.randn(2, 16, 128), "causal")
        assert out.shape == (2, 16, 128) and pen is None

    def test_bidir(self, attn):
        out, _ = attn(torch.randn(2, 8, 128), "bidir")
        assert out.shape == (2, 8, 128)

    def test_prefix(self, attn):
        out, _ = attn(torch.randn(2, 12, 128), "prefix", prefix_len=4)
        assert out.shape == (2, 12, 128)

    def test_soft_causal(self, attn):
        out, _ = attn(torch.randn(2, 10, 128), "soft_causal", soft_beta=2.0)
        assert out.shape == (2, 10, 128)

    def test_unknown_mode_raises(self, attn):
        with pytest.raises(ValueError):
            attn(torch.randn(2, 8, 128), "invalid")


class TestAttentionPenalty:
    def test_penalty_when_training(self, attn_with_penalty):
        attn_with_penalty.train()
        _, pen = attn_with_penalty(torch.randn(2, 16, 128), "causal")
        assert pen is not None and pen.item() >= 0.0

    def test_no_penalty_at_eval(self, attn_with_penalty):
        attn_with_penalty.eval()
        _, pen = attn_with_penalty(torch.randn(2, 16, 128), "causal")
        assert pen is None


class TestSlidingWindow:
    def test_windowed_causal(self, attn):
        attn.sliding_window = 4
        out, _ = attn(torch.randn(2, 16, 128), "causal")
        assert out.shape == (2, 16, 128)

    def test_windowed_bidir(self, attn):
        attn.sliding_window = 4
        out, _ = attn(torch.randn(2, 8, 128), "bidir")
        assert out.shape == (2, 8, 128)
