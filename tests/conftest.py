"""Shared fixtures.

Every test builds its model from :func:`tiny_config`, which is a real
:class:`~hagi.config.Config` through the real validator — a test that constructs
dataclasses by hand can pass against a configuration the loader would reject.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hagi.config import Config

TINY_VOCAB = 512


def assert_finite(t: torch.Tensor, name: str = "tensor") -> None:
    """Assert the tensor holds no NaN or Inf."""
    assert not torch.isnan(t).any(), f"{name} has NaN"
    assert not torch.isinf(t).any(), f"{name} has Inf"


def tiny_config(**overrides) -> Config:
    """A small but structurally complete config.

    Args:
        **overrides: dotted paths into the config, for example
            ``**{"model.head.sampled_softmax_k": 8}``.
    """
    cfg = Config()
    m = cfg.model
    m.vocab_size = TINY_VOCAB
    m.hidden_size = 128
    m.num_layers = 4
    m.attention.num_query_heads = 4
    m.attention.num_kv_heads = 2
    m.attention.head_dim = 32
    m.attention.max_seq_len = 128
    m.sliding.window = 0
    m.sliding.full_every = 4
    m.ffn.multiple_of = 32
    m.head.unigram_prior = False
    m.head.unigram_path = ""
    m.head.ce_chunk_rows = 16
    cfg.train.batch_size = 2
    cfg.train.grad_accum_steps = 2
    cfg.train.data.seq_len = 32
    cfg.train.max_steps = 100
    cfg.train.schedule.warmup_steps = 10
    cfg.train.grad_checkpointing = False

    for key, value in overrides.items():
        target: object = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        assert hasattr(target, parts[-1]), f"unknown override {key!r}"
        setattr(target, parts[-1], value)

    from hagi.config import validate_config

    validate_config(cfg)
    return cfg


@pytest.fixture
def cfg() -> Config:
    return tiny_config()


@pytest.fixture(autouse=True)
def deterministic():
    """Seed every test: a flaky numerical test is worse than no test."""
    torch.manual_seed(1234)
    np.random.seed(1234)
