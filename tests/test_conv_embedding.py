"""Tests for ConvEmbedding: factorized source encoder + causal conv."""

from __future__ import annotations

import pytest
import torch

from hagi.model.conv_embedding import ConvEmbedding


@pytest.fixture
def embed():
    return ConvEmbedding(1000, 64, 32, 3, 1e-6)


class TestConvEmbedding:
    def test_output_shape(self, embed):
        assert embed(torch.randint(0, 1000, (2, 16))).shape == (2, 16, 64)

    def test_weight_shape(self, embed):
        # ConvEmbedding.weight = token_expand.weight @ token_compress.weight.t()
        # shape is (H, V) — read the doc
        w = embed.weight
        assert w.shape == (64, 1000)

    def test_causal_no_future_leak(self, embed):
        ids1 = torch.randint(0, 1000, (2, 16))
        h1 = embed(ids1)
        ids2 = ids1.clone()
        ids2[:, 8] = (ids2[:, 8] + 1) % 1000
        h2 = embed(ids2)
        assert torch.allclose(h1[:, :8], h2[:, :8], atol=1e-4)
        assert not torch.allclose(h1[:, 8], h2[:, 8], atol=1e-4)

    def test_training_disables_cache(self, embed):
        embed.train()
        embed(torch.randint(0, 1000, (2, 16)))
        assert embed._conv_cache is None

    def test_eval_enables_cache(self, embed):
        embed.eval()
        embed(torch.randint(0, 1000, (2, 16)))
        assert embed._conv_cache is not None
        assert embed._conv_cache.shape[1] <= embed.left_pad

    def test_reset_conv_cache(self, embed):
        embed.eval()
        embed(torch.randint(0, 1000, (2, 16)))
        assert embed._conv_cache is not None
        embed.reset_conv_cache()
        assert embed._conv_cache is None
