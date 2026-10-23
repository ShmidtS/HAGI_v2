"""Интеграция backbone (attention | recurrent) в модельный контур.

Проверяет: сборку обеих конфигураций, forward+loss на recurrent, дефолт
неизменен (roundtrip сериализации конфига), function-preserving depth-grow
на recurrent-модели, инкрементальную генерацию через состояние.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from hagi.config import Config, validate_config
from hagi.inference.generate import generate
from hagi.model.depth_grow import grow_config, grow_state_dict, zero_new_blocks_
from hagi.model.factory import build_model_for_config


def _base_cfg(num_layers: int = 3, vocab: int = 128, hidden: int = 32) -> Config:
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


def _recurrent_cfg(num_layers: int = 3) -> Config:
    cfg = _base_cfg(num_layers)
    cfg.model.backbone = "recurrent"
    validate_config(cfg)
    return cfg


def test_default_backbone_is_attention():
    assert Config().model.backbone == "attention"


def test_config_roundtrip_serialization():
    cfg = _recurrent_cfg()
    data = dataclasses.asdict(cfg)
    restored = Config()
    for field_name, value in data.items():
        if field_name == "model":
            for k, v in value.items():
                setattr(restored.model, k, v)
        # остальные top-level поля проверяются отдельным roundtrip-механизмом
    assert restored.model.backbone == "recurrent"
    # дефолт: поле появляется в сериализации со значением по умолчанию
    assert dataclasses.asdict(Config())["model"]["backbone"] == "attention"


def test_recurrent_model_forward_and_loss():
    torch.manual_seed(0)
    cfg = _recurrent_cfg()
    model = build_model_for_config(cfg)
    model.eval()
    idx = torch.randint(0, cfg.model.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(idx, targets=targets, return_logits=True)
    assert out.logits.shape == (2, 16, cfg.model.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_backbone_changes_attn_module_not_structure():
    attn_model = build_model_for_config(_base_cfg())
    rec_model = build_model_for_config(_recurrent_cfg())
    from hagi.model.attention import Attention
    from hagi.model.recurrent import Recurrent

    assert isinstance(attn_model.blocks[0].attn, Attention)
    assert isinstance(rec_model.blocks[0].attn, Recurrent)
    # FFN-миксер общий
    assert type(attn_model.blocks[0].mixer) is type(rec_model.blocks[0].mixer)
    # state_dict-имена: специфичные тензоры веток не пересекаются
    # (out_proj/branch_scale/attn-norm-имена могут совпадать — но обе ветки
    # никогда не живут в одном state_dict, конфликта загрузки нет)
    attn_keys = {k.split(".attn.")[-1] for k in attn_model.state_dict() if ".attn." in k}
    rec_keys = {k.split(".attn.")[-1] for k in rec_model.state_dict() if ".attn." in k}
    assert not ("qkv_proj.weight" in rec_keys or "in_proj.weight" in attn_keys)
    assert "nu" in rec_keys and "nu" not in attn_keys
    # каждая ветка грузится строго в своё состояние
    attn_model.load_state_dict(attn_model.state_dict())
    rec_model.load_state_dict(rec_model.state_dict())


def test_recurrent_incremental_matches_full():
    torch.manual_seed(0)
    cfg = _recurrent_cfg()
    model = build_model_for_config(cfg)
    model.eval()
    idx = torch.randint(0, cfg.model.vocab_size, (2, 12))
    with torch.no_grad():
        full = model(idx, return_logits=True).logits
        model.reset_cache()
        model.allocate_cache(torch.float32, idx.device)
        pre = model(idx[:, :8], return_logits=True, use_cache=True).logits
        torch.testing.assert_close(full[:, :8], pre, rtol=1e-4, atol=1e-4)
        step = model(idx[:, 8:9], return_logits=True, use_cache=True).logits
        torch.testing.assert_close(full[:, 8:9], step, rtol=1e-4, atol=1e-4)
        model.reset_cache()


def test_generate_recurrent():
    torch.manual_seed(0)
    cfg = _recurrent_cfg()
    model = build_model_for_config(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.model.vocab_size, (2, 4))
    out = generate(
        model,
        prompt,
        max_new_tokens=5,
        eos_token_id=0,
        pad_token_id=0,
        temperature=0.0,
    )
    assert out.token_ids.shape == (2, 9)


def test_depth_grow_function_preserving_recurrent():
    torch.manual_seed(0)
    cfg3 = _recurrent_cfg(3)
    model3 = build_model_for_config(cfg3)
    model3.eval()
    # ненулевой out_proj, иначе тест тривиален
    with torch.no_grad():
        for b in model3.blocks:
            b.attn.out_proj.weight.normal_(std=0.05)

    cfg5 = grow_config(cfg3, 5)
    model5 = build_model_for_config(cfg5)
    model5.load_state_dict(grow_state_dict(model3.state_dict(), 3, 5), strict=False)
    zero_new_blocks_(model5, 3)
    model5.eval()

    idx = torch.randint(0, cfg3.model.vocab_size, (2, 16))
    with torch.no_grad():
        out3 = model3(idx, return_logits=True)
        out5 = model5(idx, return_logits=True)
    assert torch.equal(out3.logits, out5.logits)


def test_validate_rejects_recurrent_with_merge():
    from hagi.config import _apply_dict

    cfg = _recurrent_cfg()
    # включаем merge через lenient-apply поверх готового конфига
    _apply_dict(cfg, {"merge": {"enabled": True, "n_experts": 2}}, lenient=True)
    with pytest.raises(ValueError, match="recurrent"):
        validate_config(cfg)


def test_invalid_backbone_rejected():
    cfg = _base_cfg()
    cfg.model.backbone = "conv"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="backbone"):
        validate_config(cfg)
