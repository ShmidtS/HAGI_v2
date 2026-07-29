"""End-to-end tests for HAGI model forward pass."""

from __future__ import annotations

import torch

from hagi.config import Config, auto_configure
from hagi.model.model import HAGI


def _valid_cfg():
    """Create a self-consistent config for testing."""
    cfg = Config()
    m = cfg.model
    # Use consistent sizes: H=256, C=128, n_q=4, n_kv=2, hd=64 (4*64=256)
    m.vocab_size = 1000
    m.hidden_size = 256
    m.core_hidden_size = 128
    m.attention.num_query_heads = 4
    m.attention.num_kv_heads = 2
    m.attention.head_dim = 64
    m.body.num_layers = 4
    m.body.bottleneck.dim = 128
    m.embeddings.factor_rank = 64
    m.sliding.sliding_window = 0
    m.sliding.full_every = 0
    cfg.train.pad_token_id = 999  # must be < vocab_size
    return cfg


def _model():
    m = HAGI(_valid_cfg())
    m.eval()
    return m


def _vtm(t):
    return torch.ones(2, t, dtype=torch.bool)


class TestHAGIForward:
    def test_causal_logits_shape(self):
        m = _model()
        ids = torch.randint(0, 1000, (2, 16))
        vtm = torch.ones(2, 16, dtype=torch.bool)
        out = m(ids, targets=None, prediction_mask=vtm, valid_target_mask=vtm, attention_mode="causal")
        # Both masks provided -> _gather_logits selects positions -> [B*T, V]
        assert out.logits.shape == (32, 1000)

    def test_bidir(self):
        m = _model()
        ids = torch.randint(0, 1000, (2, 8))
        vtm = torch.ones(2, 8, dtype=torch.bool)
        out = m(ids, targets=None, prediction_mask=vtm, valid_target_mask=vtm, attention_mode="bidir")
        assert out.logits.shape == (16, 1000)

    def test_prefix(self):
        m = _model()
        ids = torch.randint(0, 1000, (2, 12))
        vtm = torch.ones(2, 12, dtype=torch.bool)
        out = m(ids, targets=None, prediction_mask=vtm, valid_target_mask=vtm, attention_mode="prefix", prefix_len=4)
        assert out.logits.shape == (24, 1000)

    def test_aux_rate_not_none(self):
        m = _model()
        out = m(torch.randint(0, 1000, (2, 16)), targets=torch.randint(0, 1000, (2, 16)),
                prediction_mask=_vtm(16), valid_target_mask=_vtm(16), attention_mode="causal")
        assert out.aux.rate is not None and out.aux.distortion is not None
        assert out.aux.moe_lb is None

    def test_hidden_shape(self):
        m = _model()
        ids = torch.randint(0, 1000, (2, 16))
        out = m(ids, targets=None, prediction_mask=_vtm(16),
                valid_target_mask=_vtm(16), attention_mode="causal")
        assert out.hidden.shape == (2, 16, 256)

    def test_no_refinement_when_disabled(self):
        m = _model()
        ids = torch.randint(0, 1000, (2, 8))
        out = m(ids, targets=None, prediction_mask=_vtm(8),
                valid_target_mask=_vtm(8), attention_mode="causal")
        assert out.aux.refinement is None


class TestHAGICache:
    def test_allocate_for_cache(self):
        m = _model()
        caches = m.allocate_for_cache(2, torch.float32, torch.device("cpu"))
        assert len(caches) == 4  # num_layers=4
        for c in caches:
            assert c.max_seq_len == 4096 and c.n_kv_heads == 2

    def test_cached_forward(self):
        m = _model()
        m.allocate_for_cache(2, torch.float32, torch.device("cpu"))
        out = m(torch.randint(0, 1000, (2, 16)), targets=None, prediction_mask=_vtm(16),
                valid_target_mask=_vtm(16), attention_mode="causal")
        assert out.logits is not None
        m.reset_cache()
        assert all(blk.attn._kv_cache is None for blk in m.blocks)


class TestHAGIInit:
    def test_lm_head_weight_shape(self):
        w = HAGI(_valid_cfg()).lm_head_weight
        assert w.shape == (1000, 256)

    def test_params_nonzero(self):
        assert sum(p.numel() for p in HAGI(_valid_cfg()).parameters()) > 0

    def test_ternary_on_by_default(self):
        assert HAGI(_valid_cfg())._use_ternary is True

    def test_auto_configured_instantiates(self):
        """auto_configure should produce a valid config — verify with a stable budget."""
        m_cfg = auto_configure(25_000_000)  # medium budget, avoids edge cases
        cfg = Config()
        cfg.model = m_cfg
        cfg.train.pad_token_id = m_cfg.vocab_size - 1
        # Validate the auto-configured cfg first
        from hagi.config import validate_config
        validate_config(cfg)
        m = HAGI(cfg)
        m.eval()
        ids = torch.randint(0, m_cfg.vocab_size, (1, 8))
        vtm = torch.ones(1, 8, dtype=torch.bool)
        out = m(ids, targets=None, prediction_mask=vtm, valid_target_mask=vtm, attention_mode="causal")
        assert out.logits is not None
