from __future__ import annotations

import pytest

from hagi.config import Config
from hagi.model.model import HAGI
from hagi.orchestrator.evaluation import evaluate_packed_tokens
from tests.conftest import tiny_config


def test_packed_exact_evaluation_scores_every_token_once():
    cfg = tiny_config(**{"model.vocab_size": 8, "model.attention.max_seq_len": 32, "train.data.seq_len": 4})
    model = HAGI(cfg)
    result = evaluate_packed_tokens(model, cfg, [1, 2, 3, 4, 5, 6, 7])
    assert result["scored_token_count"] == 7
    assert isinstance(result["exact_ce"], float)
    assert result["exact_ce"] >= 0.0


def test_packed_exact_evaluation_rejects_invalid_inputs():
    cfg = Config()
    model = HAGI(cfg)
    with pytest.raises(ValueError):
        evaluate_packed_tokens(model, cfg, [])
    with pytest.raises(ValueError):
        evaluate_packed_tokens(model, cfg, [1, cfg.model.vocab_size])
    with pytest.raises(TypeError):
        evaluate_packed_tokens(object(), cfg, [1, 2])
