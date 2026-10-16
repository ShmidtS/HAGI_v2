"""Function-preserving depth growth (net2net-style layer insertion).

Tests that growing num_layers L -> L' with identity-appended blocks
reproduces the source model bit-for-bit (ARCHITECTURE.md principle 3).
"""

import torch

from hagi.config import Config, validate_config
from hagi.model.depth_grow import grow_config, grow_state_dict, zero_new_blocks_
from hagi.model.factory import build_model_for_config


def _base_cfg(num_layers: int, vocab: int = 128, hidden: int = 32) -> Config:
    cfg = Config()
    cfg.model.num_layers = num_layers
    cfg.model.vocab_size = vocab
    cfg.model.hidden_size = hidden
    cfg.model.attention.num_query_heads = 2
    cfg.model.attention.num_kv_heads = 2
    cfg.model.attention.head_dim = 16
    cfg.model.use_ternary = False
    validate_config(cfg)
    return cfg


def test_depth_grow_is_function_preserving():
    torch.manual_seed(0)
    cfg3 = _base_cfg(3)
    model3 = build_model_for_config(cfg3)
    model3.eval()

    cfg7 = grow_config(cfg3, 7)
    assert cfg7.model.num_layers == 7
    model7 = build_model_for_config(cfg7)
    model7.load_state_dict(grow_state_dict(model3.state_dict(), 3, 7), strict=False)
    zero_new_blocks_(model7, 3)
    model7.eval()

    idx = torch.randint(0, cfg3.model.vocab_size, (2, 16))
    with torch.no_grad():
        out3 = model3(idx, return_logits=True)
        out7 = model7(idx, return_logits=True)
    assert torch.equal(out3.logits, out7.logits)


def test_grow_state_dict_drops_old_layer_keys_beyond_old_layers():
    torch.manual_seed(0)
    cfg = _base_cfg(3)
    model = build_model_for_config(cfg)
    sd = model.state_dict()
    grown = grow_state_dict(sd, 3, 7)
    assert len(grown) == len(sd)
    assert all("blocks.6." not in k for k in grown)


def test_grow_config_rejects_shrink():
    import pytest

    cfg = _base_cfg(3)
    with pytest.raises(ValueError):
        grow_config(cfg, 3)
