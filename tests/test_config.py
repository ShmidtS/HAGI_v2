"""Tests for hagi.config: validation, auto_configure, sliding windows."""

from __future__ import annotations

import pytest

from hagi.config import (
    Config, auto_configure, validate_config, layer_sliding_windows,
)
from hagi.train.checkpoint import cfg_to_dict, cfg_from_dict


def _valid_cfg():
    """A self-consistent config for roundtrip tests."""
    cfg = Config()
    m = cfg.model
    m.vocab_size = 1000
    m.hidden_size = 256
    m.core_hidden_size = 128
    m.attention.num_query_heads = 4
    m.attention.num_kv_heads = 2
    m.attention.head_dim = 64
    m.body.num_layers = 4
    m.body.bottleneck.dim = 128
    m.embeddings.factor_rank = 64
    cfg.train.pad_token_id = 999  # must be < vocab_size
    return cfg


class TestDefaultConfig:
    def test_default_sizes(self):
        m = Config().model
        assert m.vocab_size == 49154
        assert m.hidden_size == 384
        assert m.core_hidden_size == 192
        assert m.hidden_size > m.core_hidden_size
        assert m.body.num_layers == 12
        assert m.attention.num_query_heads == 8
        assert m.attention.num_kv_heads == 4
        assert m.attention.head_dim == 64
        assert m.attention.num_kv_heads <= m.attention.num_query_heads
        assert m.attention.num_query_heads % m.attention.num_kv_heads == 0

    def test_sliding_windows_default_all_full(self):
        windows = layer_sliding_windows(Config().model)
        assert len(windows) == 12
        assert all(w == 0 for w in windows)

    def test_sliding_windows_full_every(self):
        cfg = Config()
        cfg.model.sliding.sliding_window = 128
        cfg.model.sliding.full_every = 4
        cfg.model.body.num_layers = 8  # temp
        windows = layer_sliding_windows(cfg.model)
        for i in range(8):
            assert windows[i] == (0 if i % 4 == 0 else 128), f"layer {i}"

    def test_sliding_windows_explicit(self):
        cfg = Config()
        cfg.model.body.num_layers = 6
        cfg.model.sliding.sliding_window = 64
        cfg.model.sliding.window_layers = (1, 3)
        assert layer_sliding_windows(cfg.model) == [0, 64, 0, 64, 0, 0]


class TestConfigValidation:
    def test_rejects_c_gte_h(self):
        cfg = Config()
        cfg.model.core_hidden_size = 400
        with pytest.raises(ValueError, match="0 < C < hidden_size"):
            validate_config(cfg)

    def test_rejects_n_kv_gt_n_q(self):
        cfg = Config()
        cfg.model.attention.num_kv_heads = 16
        with pytest.raises(ValueError, match="num_kv_heads"):
            validate_config(cfg)

    def test_rejects_gqa_not_divisible(self):
        cfg = Config()
        cfg.model.attention.num_kv_heads = 3
        with pytest.raises(ValueError, match="divisible"):
            validate_config(cfg)

    def test_rejects_h_not_divisible_by_n_q(self):
        cfg = Config()
        cfg.model.hidden_size = 385
        with pytest.raises(ValueError, match="hidden_size must be divisible"):
            validate_config(cfg)

    def test_rejects_zero_grad_accum(self):
        cfg = Config()
        cfg.train.grad_accum_steps = 0
        with pytest.raises(ValueError):
            validate_config(cfg)


class TestAutoConfigure:
    def test_15m_budget_validates(self):
        m = auto_configure(15_000_000)
        assert m.hidden_size > m.core_hidden_size > 0
        assert m.body.num_layers >= 2
        assert m.body.moe.enabled is False

    def test_150m_budget_moe_on(self):
        m = auto_configure(150_000_000)
        assert m.body.moe.enabled is True
        assert m.body.moe.num_experts >= 1
        assert 1 <= m.body.moe.top_k <= m.body.moe.num_experts

    def test_head_dim_in_range(self):
        m = auto_configure(40_000_000)
        assert 32 <= m.attention.head_dim <= 128

    def test_factor_rank_bounded(self):
        m = auto_configure(100_000_000)
        assert 8 <= m.embeddings.factor_rank <= 256

    def test_bottleneck_dim_matches_core(self):
        m = auto_configure(50_000_000)
        assert m.body.bottleneck.dim == m.core_hidden_size


class TestConfigRoundtrip:
    def test_default_roundtrip(self):
        cfg = _valid_cfg()
        d = cfg_to_dict(cfg)
        cfg2 = cfg_from_dict(d)
        assert cfg2.model.hidden_size == cfg.model.hidden_size
        assert cfg2.train.max_steps == cfg.train.max_steps
        assert cfg2.inference.temperature == cfg.inference.temperature

    def test_auto_configured_roundtrip(self):
        cfg = Config()
        cfg.model = auto_configure(25_000_000)
        d = cfg_to_dict(cfg)
        cfg2 = cfg_from_dict(d)
        validate_config(cfg2)
        assert cfg2.model.hidden_size == cfg.model.hidden_size
