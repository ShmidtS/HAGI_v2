import subprocess
import sys

import pytest
import torch

from hagi.config import Config, validate_config
from hagi.model.merge import merge_experts, state_key_digest
from hagi.model.model import HAGI
from hagi.train.checkpoint import config_from_dict, config_to_dict
from tests.conftest import tiny_config


def _configs():
    merged = tiny_config(
        **{
            "model.num_layers": 2,
            "model.hidden_size": 64,
            "model.attention.num_query_heads": 2,
            "model.attention.num_kv_heads": 2,
            "model.attention.head_dim": 32,
            "model.ffn.intermediate_size": 128,
            "model.embedding.conv_kernel": 1,
        }
    )
    merged.merge.enabled = True
    merged.merge.n_experts = 2
    merged.merge.expert_hidden = 32
    merged.merge.mixer_type = "swiglu"
    validate_config(merged)
    expert = tiny_config(
        **{
            "model.num_layers": 2,
            "model.hidden_size": 32,
            "model.attention.num_query_heads": 1,
            "model.attention.num_kv_heads": 1,
            "model.attention.head_dim": 32,
            "model.ffn.intermediate_size": 64,
            "model.embedding.conv_kernel": 1,
        }
    )
    return merged, expert


def test_state_key_digest_is_schema_only_and_order_independent():
    state = {"b": torch.empty(2, 3, dtype=torch.float16), "a": torch.empty(1, dtype=torch.int64)}
    reordered = {"a": state["a"], "b": state["b"]}
    assert state_key_digest(state) == state_key_digest(reordered)
    assert state_key_digest(state) != state_key_digest({"a": state["a"]})
    with pytest.raises(TypeError):
        state_key_digest({"bad": 1})


def test_expert_weight_source_round_trip_and_validation():
    cfg = Config()
    cfg.merge.enabled = True
    cfg.merge.expert_weight_source = "effective_sparse"
    cfg.merge.ternary_depth = 0
    cfg.merge.ternary_tree_schema_version = 1
    validate_config(cfg)
    data = config_to_dict(cfg)
    assert data["merge"]["expert_weight_source"] == "effective_sparse"
    assert config_from_dict(data).merge.expert_weight_source == "effective_sparse"

    for value in ("unknown", 1, None):
        bad = Config()
        bad.merge.enabled = True
        bad.merge.expert_weight_source = value
        with pytest.raises(ValueError):
            validate_config(bad)

    for value in (-1, 1.0):
        bad = Config()
        bad.merge.enabled = True
        bad.merge.ternary_depth = value
        with pytest.raises(ValueError):
            validate_config(bad)

    for value in (0, -1, 1.0):
        bad = Config()
        bad.merge.enabled = True
        bad.merge.ternary_tree_schema_version = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_merge_experts_rejects_explicit_invalid_override():
    _, expert_cfg = _configs()
    merged_cfg, _ = _configs()
    original = merged_cfg.merge.expert_weight_source
    state = HAGI(expert_cfg).state_dict()
    with pytest.raises(ValueError):
        merge_experts(merged_cfg, [state, state], expert_weight_source="unknown")
    assert merged_cfg.merge.expert_weight_source == original
    with pytest.raises(ValueError):
        merge_experts(merged_cfg, [state, state], expert_weight_source=1)
    assert merged_cfg.merge.expert_weight_source == original


def test_merge_experts_invalid_count_does_not_mutate_caller_config():
    merged_cfg, expert_cfg = _configs()
    original = merged_cfg.merge.expert_weight_source
    state = HAGI(expert_cfg).state_dict()

    with pytest.raises(ValueError, match="expected 2 expert states, got 1"):
        merge_experts(
            merged_cfg,
            [state],
            expert_weight_source="effective_sparse",
        )

    assert merged_cfg.merge.expert_weight_source == original


def test_explicit_successful_override_is_retained_on_model_only():
    merged_cfg, expert_cfg = _configs()
    state = HAGI(expert_cfg).state_dict()

    merged = merge_experts(
        merged_cfg,
        [dict(state), dict(state)],
        expert_weight_source="effective_sparse",
    )

    assert merged_cfg.merge.expert_weight_source == "ternary_master"
    assert merged.cfg.merge.expert_weight_source == "effective_sparse"


def test_effective_sparse_preserves_nonbinary_hidden_matrix(monkeypatch):
    merged_cfg, expert_cfg = _configs()
    merged_cfg.merge.expert_weight_source = "effective_sparse"
    state = HAGI(expert_cfg).state_dict()
    key = next(k for k in state if k.endswith("out_proj.weight"))
    with torch.no_grad():
        state[key].zero_()
        state[key][0, 0] = 0.37
        state[key][0, 1] = -1.91
    monkeypatch.setattr(
        "hagi.model.merge._ternarize_block",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("effective_sparse must not ternarize")
        ),
    )
    merged = merge_experts(merged_cfg, [dict(state), dict(state)])
    merged_weight = merged.state_dict()[key]
    assert torch.equal(merged_weight[:32, :32], state[key])
    assert torch.equal(merged_weight[32:, 32:], state[key])
    assert torch.equal(merged_weight[:32, 32:], torch.zeros_like(merged_weight[:32, 32:]))
    assert torch.equal(merged_weight[32:, :32], torch.zeros_like(merged_weight[32:, :32]))
    assert merged_cfg.merge.expert_weight_source == "effective_sparse"


def test_default_is_legacy_ternary_master():
    assert Config().merge.expert_weight_source == "ternary_master"


def test_legacy_default_still_quantizes_hidden_matrix():
    merged_cfg, expert_cfg = _configs()
    state = HAGI(expert_cfg).state_dict()
    key = next(k for k in state if k.endswith("out_proj.weight"))
    with torch.no_grad():
        state[key].zero_()
        state[key][0, 0] = 0.37
        state[key][0, 1] = -1.91
    merged = merge_experts(merged_cfg, [dict(state), dict(state)])
    from hagi.model.merge import _ternarize_block

    expected = _ternarize_block(state[key], merged_cfg.model.ternary.eps)
    assert torch.equal(merged.state_dict()[key][:32, :32], expected)


def test_merge_cli_exposes_weight_format_override():
    result = subprocess.run(
        [sys.executable, "scripts/merge_experts.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--expert-weight-format" in result.stdout
    assert "effective_sparse" in result.stdout
