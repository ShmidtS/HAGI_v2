"""Config invariants and analytic parameter counting.

The two things this file guards:

* ``count_params`` matches a real instantiation *exactly*. It is used to size
  every run, and an approximate answer silently produces a model that does not
  fit the memory budget it was planned against.
* ``validate_config`` rejects each geometry the model cannot express. Every check
  here corresponds to a failure that is otherwise silent or expensive.
"""

from __future__ import annotations

import pytest

from hagi.config import (
    Config,
    _apply_dict,
    count_params,
    describe,
    ffn_width,
    layer_windows,
    validate_config,
)
from hagi.model.model import HAGI
from tests.conftest import tiny_config


class TestParamCount:
    """Analytic count vs. a real module tree."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"model.embedding.tie_lm_head": False},
            {"model.embedding.conv_kernel": 1},
            {"model.attention.qk_norm": False},
            {"model.sliding.window": 16},
            {"model.ffn.intermediate_size": 96},
        ],
        ids=["dense", "untied", "no_conv", "no_qknorm", "windowed", "explicit_ffn"],
    )
    def test_analytic_equals_real(self, overrides):
        cfg = tiny_config(**overrides)
        model = HAGI(cfg)
        real = sum(p.numel() for p in model.parameters())
        assert count_params(cfg.model)["total"] == real

    def test_active_body_equals_body_when_dense(self):
        counts = count_params(tiny_config().model)
        assert counts["active_body"] == counts["body"]

    def test_tied_head_costs_nothing(self):
        tied = count_params(tiny_config().model)
        untied = count_params(tiny_config(**{"model.embedding.tie_lm_head": False}).model)
        assert tied["lm_head"] == 0
        assert untied["total"] - tied["total"] == tied["embedding"]


class TestLayerPatterns:
    def test_window_pattern_keeps_relays(self):
        cfg = tiny_config(**{"model.sliding.window": 16, "model.sliding.full_every": 4})
        windows = layer_windows(cfg.model)
        assert windows[0] == 0, "layer 0 must be a global relay"
        assert any(w > 0 for w in windows), "no windowed layer means the setting did nothing"

    def test_window_zero_is_all_full(self):
        assert layer_windows(tiny_config().model) == [0] * 4

    def test_ffn_width_rounds_up(self):
        cfg = tiny_config()
        cfg.model.ffn.intermediate_size = 0
        cfg.model.ffn.multiple_of = 64
        assert ffn_width(cfg.model) % 64 == 0


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"model.attention.head_dim": 30},  # 4*30 != 128
            {"model.attention.num_kv_heads": 3},  # does not divide 4
            {"model.attention.head_dim": 31, "model.attention.num_query_heads": 128},  # odd head_dim
            {"model.num_layers": 0},
            {"model.head.ce_chunk_rows": 0},
            {"model.head.logit_scale_init": -1.0},
            {"train.data.seq_len": 4096},  # exceeds max_seq_len
            {"train.precision": "fp8"},
            {"train.max_grad_norm": 0.0},
            {"train.schedule.decay_fraction": 1.0},
            {"train.schedule.min_lr_ratio": 1.0},
            {"train.data.pad_token_id": 1},  # equals eos
            {"train.data.eos_token_id": 99999},  # outside vocab
            {"inference.top_p": 0.0},
        ],
        ids=[
            "head_dim_mismatch",
            "kv_not_divisor",
            "odd_head_dim",
            "zero_layers",
            "zero_chunk",
            "negative_logit_scale",
            "seq_over_max",
            "bad_precision",
            "zero_clip",
            "full_decay",
            "min_lr_one",
            "pad_eq_eos",
            "eos_out_of_vocab",
            "top_p_zero",
        ],
    )
    def test_rejects(self, overrides):
        with pytest.raises(ValueError):
            tiny_config(**overrides)

    def test_window_without_relay_rejected(self):
        cfg = tiny_config()
        cfg.model.sliding.window = 16
        cfg.model.sliding.full_every = 0
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_unigram_prior_needs_path(self):
        cfg = tiny_config()
        cfg.model.head.unigram_prior = True
        cfg.model.head.unigram_path = ""
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_default_config_is_valid(self):
        validate_config(Config())

    def test_window_zero_allows_full_every(self):
        """W=0 means all layers are full-attention — full_every is irrelevant."""
        cfg = tiny_config()
        cfg.model.sliding.window = 0
        cfg.model.sliding.full_every = 4
        validate_config(cfg)  # should not raise
        assert all(w == 0 for w in layer_windows(cfg.model))

    def test_window_zero_with_full_every_one(self):
        cfg = tiny_config()
        cfg.model.sliding.window = 0
        cfg.model.sliding.full_every = 1
        validate_config(cfg)

    def test_v42_config_loads(self):
        """V42 config (W=0, T=512, ce_keep=0.5) loads and validates."""
        from hagi.config import load_config
        import pathlib
        path = pathlib.Path(__file__).parent.parent / "configs" / "v42_1b.yaml"
        if path.exists():
            cfg = load_config(str(path))
            assert cfg.model.sliding.window == 0
            assert cfg.train.data.seq_len == 512
            assert cfg.train.ce_keep_rate == 0.5
            assert cfg.train.batch_size == 48
            assert all(w == 0 for w in layer_windows(cfg.model))


class TestApplyDict:
    def test_unknown_key_raises(self):
        cfg = Config()
        with pytest.raises(ValueError, match="unknown config key"):
            _apply_dict(cfg, {"model": {"nonexistent": 1}})

    def test_nested_overlay(self):
        cfg = Config()
        _apply_dict(cfg, {"model": {"attention": {"rope_theta": 5000.0}}})
        assert cfg.model.attention.rope_theta == 5000.0

    def test_list_becomes_tuple(self):
        cfg = Config()
        _apply_dict(cfg, {"train": {"muon": {"ns_coeffs": [1.0, 2.0, 3.0]}}})
        assert isinstance(cfg.train.muon.ns_coeffs, tuple)


def test_describe_reports_shape_and_rate():
    text = describe(tiny_config())
    for token in ("H=128", "L=4", "params total=", "body share", "bits/weight"):
        assert token in text
