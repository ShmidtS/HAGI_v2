"""Tests for KVCache -- incremental KV store."""

from __future__ import annotations

import pytest
import torch

from hagi.model.kv_cache import KVCache


class TestKVCache:
    def test_initial_length_zero(self):
        assert KVCache(128, 4, 64, torch.float32, torch.device("cpu")).length == 0

    def test_update_grows(self):
        cache = KVCache(128, 4, 64, torch.float32, torch.device("cpu"))
        cache.update(torch.randn(2, 4, 8, 64), torch.randn(2, 4, 8, 64))
        assert cache.length == 8

    def test_multiple_updates(self):
        cache = KVCache(128, 4, 64, torch.float32, torch.device("cpu"))
        cache.update(torch.randn(2, 4, 5, 64), torch.randn(2, 4, 5, 64))
        cache.update(torch.randn(2, 4, 3, 64), torch.randn(2, 4, 3, 64))
        assert cache.length == 8

    def test_get_returns_stored(self):
        cache = KVCache(128, 2, 16, torch.float32, torch.device("cpu"))
        k = torch.randn(1, 2, 4, 16)
        v = torch.randn(1, 2, 4, 16)
        cache.update(k, v)
        kc, vc = cache.get()
        assert kc.shape == (1, 2, 4, 16) and vc.shape == (1, 2, 4, 16)
        assert torch.equal(kc, k) and torch.equal(vc, v)

    def test_overflow(self):
        cache = KVCache(4, 2, 8, torch.float32, torch.device("cpu"))
        cache.update(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
        with pytest.raises(ValueError, match="overflow"):
            cache.update(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))

    def test_reset(self):
        cache = KVCache(128, 4, 64, torch.float32, torch.device("cpu"))
        cache.update(torch.randn(2, 4, 8, 64), torch.randn(2, 4, 8, 64))
        assert cache.length == 8
        cache.reset()
        assert cache.length == 0
        with pytest.raises(RuntimeError, match="empty"):
            cache.get()

    def test_get_before_update(self):
        with pytest.raises(RuntimeError, match="empty"):
            KVCache(128, 4, 64, torch.float32, torch.device("cpu")).get()
