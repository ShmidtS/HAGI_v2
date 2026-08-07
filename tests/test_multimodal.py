"""Multimodal bridge: fixed-rate coding and modality dropout."""

from __future__ import annotations

import torch

from hagi.model.multimodal import MultimodalBridge
from tests.conftest import tiny_config


def _mm_cfg(**extra):
    overrides = {
        "model.multimodal.enabled": True,
        "model.multimodal.n_bridge_queries": 8,
        "model.multimodal.bridge_layers": 1,
        "model.multimodal.bridge_heads": 4,  # H=128 → head_dim=32, %4==0
        "model.multimodal.modality_dropout": 0.0,
        "model.multimodal.grounding_weight": 0.1,
    }
    overrides.update(extra)
    return tiny_config(**overrides)


class TestMultimodalBridge:
    def test_image_prefix_fixed_rate(self):
        cfg = _mm_cfg()
        bridge = MultimodalBridge(cfg)
        images = torch.randn(2, 3, 32, 32)
        prefix, pooled = bridge(images=images)
        assert prefix is not None and pooled is not None
        assert prefix.shape == (2, 8, cfg.model.hidden_size)
        assert pooled.shape == (2, cfg.model.hidden_size)

    def test_audio_prefix(self):
        cfg = _mm_cfg()
        bridge = MultimodalBridge(cfg)
        mel = torch.randn(2, 80, 16)
        prefix, pooled = bridge(spectrograms=mel)
        assert prefix.shape[1] == 8

    def test_two_modalities_each_keep_fixed_rate(self):
        cfg = _mm_cfg()
        bridge = MultimodalBridge(cfg)
        images = torch.randn(2, 3, 32, 32)
        mel = torch.randn(2, 80, 16)
        prefix, _ = bridge(images=images, spectrograms=mel)
        assert prefix.shape[1] == 2 * cfg.model.multimodal.n_bridge_queries

    def test_modality_dropout_can_drop_all(self):
        cfg = _mm_cfg(**{"model.multimodal.modality_dropout": 1.0})
        bridge = MultimodalBridge(cfg)
        bridge.train()
        images = torch.randn(2, 3, 32, 32)
        prefix, pooled = bridge(images=images)
        assert prefix is None and pooled is None

    def test_eval_ignores_dropout(self):
        cfg = _mm_cfg(**{"model.multimodal.modality_dropout": 1.0})
        bridge = MultimodalBridge(cfg)
        bridge.eval()
        images = torch.randn(2, 3, 32, 32)
        prefix, _ = bridge(images=images)
        assert prefix is not None
