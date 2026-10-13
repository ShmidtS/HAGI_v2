"""MultimodalBridge smoke coverage (research-only module, config-gated off
in the live line — ARCHITECTURE.md §"выключено"). These tests pin the
structural contract: fixed-rate prefix shape, per-modality encoding,
Grounding anti-collapse objective finiteness, and the no-modality path.
"""

import pytest
import torch

from hagi.config import Config
from hagi.model.multimodal import Grounding, MultimodalBridge


def _cfg() -> Config:
    cfg = Config()
    m = cfg.model
    m.hidden_size = 64
    m.attention.num_query_heads = 4
    m.attention.num_kv_heads = 4
    m.attention.head_dim = 16
    m.attention.max_seq_len = 256
    m.multimodal.enabled = True
    m.multimodal.bridge_heads = 4  # head_dim 16, divisible by 4
    m.multimodal.n_bridge_queries = 4
    m.multimodal.bridge_layers = 1
    return cfg


def test_bridge_fixed_rate_prefix_shape():
    torch.manual_seed(0)
    bridge = MultimodalBridge(_cfg())
    bridge.eval()
    images = torch.randn(2, 3, 32, 32)  # patch 16 -> 2x2 patches
    prefix, pooled = bridge(images=images)
    n_bridge = 4
    assert prefix is not None
    assert prefix.shape == (2, n_bridge, 64)  # fixed rate, H=64
    assert pooled is not None and pooled.shape == (2, 64)


def test_bridge_no_modality_returns_none():
    bridge = MultimodalBridge(_cfg())
    bridge.eval()
    prefix, pooled = bridge()
    assert prefix is None and pooled is None


def test_bridge_audio_fixed_rate():
    torch.manual_seed(0)
    bridge = MultimodalBridge(_cfg())
    bridge.eval()
    spec = torch.randn(2, 80, 8)  # mel_bins 80 (default), 8 frames
    prefix, pooled = bridge(spectrograms=spec)
    assert prefix is not None
    assert prefix.shape[0] == 2 and prefix.shape[2] == 64


def test_grounding_loss_finite_and_positive():
    torch.manual_seed(0)
    cfg = _cfg()
    g = Grounding(64)
    text = torch.randn(4, 64)
    modal = torch.randn(4, 64)
    loss = g(text, modal)
    assert torch.isfinite(loss) and float(loss) >= 0.0
