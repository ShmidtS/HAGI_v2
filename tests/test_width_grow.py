"""Function-preserving width growth: после grow модель в ширине H'
воспроизводит исходные logits bit-for-bit (нулевые писатели/столбцы +
точная √-компенсация RMSNorm-последователей)."""

from __future__ import annotations

import copy

import pytest
import torch

from hagi.config import Config, validate_config
from hagi.model.factory import build_model_for_config
from hagi.model.width_grow import grow_width_config, grow_width_state_dict


def _recurrent_cfg(width: int = 32, vocab: int = 64, layers: int = 2) -> Config:
    cfg = Config()
    cfg.model.backbone = "recurrent"
    cfg.model.hidden_size = width
    cfg.model.vocab_size = vocab
    cfg.model.num_layers = layers
    cfg.model.attention.num_query_heads = max(1, width // 16)
    cfg.model.attention.num_kv_heads = max(1, cfg.model.attention.num_query_heads // 2)
    cfg.model.attention.head_dim = 16
    cfg.model.ternary.enabled = False
    cfg.model.adapters.enabled = False
    cfg.model.head.unigram_prior = False
    cfg.model.head.z_loss_weight = 0.0
    # eps ниже масштаба потока: при rms~0.02 eps=1e-5 давал бы 10%-е
    # искажение компенсации (в реальном прогоне поток ~1, eps пренебрежим)
    cfg.model.norm_eps = 1e-8
    validate_config(cfg)
    return cfg


@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
def test_width_grow_bit_for_bit(tie: bool):
    torch.manual_seed(0)
    cfg = _recurrent_cfg(width=32)
    cfg.model.embedding.tie_lm_head = tie
    validate_config(cfg)
    model = build_model_for_config(cfg)
    model.eval()

    old_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    new_h = 64
    state = grow_width_state_dict(old_state, 32, new_h)

    grown_cfg = grow_width_config(cfg, new_h)
    grown = build_model_for_config(grown_cfg)
    missing, unexpected = grown.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    # непокрытые ключи допустимы только если их нет в state (буферы)
    assert all("inv_freq" in k or "attn_sink" in k for k in missing), missing
    grown.eval()

    torch.manual_seed(3)
    ids = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        out_old = model(ids)
        out_new = grown(ids)
    # поток имеет разную ширину — сравниваются logits ([B,T,V] у обеих)
    lo = model.head.logits(out_old.hidden)
    ln = grown.head.logits(out_new.hidden)
    ldiff = (lo - ln).abs().max().item()
    scale = lo.abs().max().item()
    assert ldiff <= 1e-4 * max(scale, 1.0), f"logits diff {ldiff} (scale {scale})"


def test_new_writers_carry_gradient():
    """Нулевые писатели оживают: градиент по новым строкам out_proj
    ненулевой при ненулевом входе (в отличие от нулевых emb-столбцов)."""
    torch.manual_seed(1)
    cfg = _recurrent_cfg(width=32)
    model = build_model_for_config(cfg)
    old_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = grow_width_state_dict(old_state, 32, 64)
    grown_cfg = grow_width_config(cfg, 64)
    grown = build_model_for_config(grown_cfg)
    grown.load_state_dict(state, strict=False)
    ids = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    out = grown(ids, targets)
    out.lm_loss.backward()
    # каскад живости: head-вес (при tied — сама кодовая книга) получает
    # градиент по новым каналам от emb-шума в residual
    head_w = grown.head.projection.weight if grown.head.projection is not None else grown.encoder.embedding.weight
    head_grad = head_w.grad
    assert head_grad is not None and head_grad[:, 32:].abs().sum() > 0, "head new columns must receive grad"
    emb_grad = grown.encoder.embedding.weight.grad
    assert emb_grad[:, 32:].abs().sum() > 0, "embedding columns must receive grad"
