"""Tests for checkpoint save/load roundtrip."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from hagi.config import Config
from hagi.model.model import HAGI
from hagi.train.checkpoint import (
    save_checkpoint, load_checkpoint_payload, load_model_checkpoint,
    latest_checkpoint, CHECKPOINT_FORMAT_VERSION, cfg_to_dict, cfg_from_dict,
)


def _valid_cfg():
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


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _model():
    return HAGI(_valid_cfg())


class TestCheckpointRoundtrip:
    def test_save_and_load_payload(self, tmpdir):
        path = save_checkpoint(_model(), _valid_cfg(), 500, tmpdir, keep_last=3)
        assert os.path.isfile(path)
        payload = load_checkpoint_payload(path)
        assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert payload["completed_updates"] == 500

    def test_load_model_strict(self, tmpdir):
        m1 = _model()
        cfg = _valid_cfg()
        path = save_checkpoint(m1, cfg, 100, tmpdir)
        m2 = HAGI(cfg)
        step, _ = load_model_checkpoint(path, m2, "cpu")
        assert step == 100
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1.data, p2.data)

    def test_rejects_wrong_format(self, tmpdir):
        path = save_checkpoint(_model(), _valid_cfg(), 0, tmpdir)
        state = load_checkpoint_payload(path)
        state["format_version"] = 99
        torch.save(state, path)
        with pytest.raises(Exception):
            load_checkpoint_payload(path)

    def test_latest(self, tmpdir):
        save_checkpoint(_model(), _valid_cfg(), 100, tmpdir)
        save_checkpoint(_model(), _valid_cfg(), 200, tmpdir)
        assert "step-000200" in str(latest_checkpoint(tmpdir))

    def test_rotation(self, tmpdir):
        m = _model()
        cfg = _valid_cfg()
        for step in [100, 200, 300, 400, 500]:
            save_checkpoint(m, cfg, step, tmpdir, keep_last=2)
        step_files = [f for f in os.listdir(tmpdir) if f.startswith("step-")]
        assert len(step_files) <= 2

    def test_config_roundtrip(self):
        cfg = _valid_cfg()
        d = cfg_to_dict(cfg)
        cfg2 = cfg_from_dict(d)
        assert cfg2.model.hidden_size == cfg.model.hidden_size
        assert cfg2.train.learning_rate == cfg.train.learning_rate
