"""Liveness two-axis stop (Unified/Liveness.lean liveness_two_axis).

Contrapositive semantics: STOP iff both axes frozen — the energy
axis (control CE not decreasing over the window) AND the data/growth
axis (grad-norm injection signal flat). While either axis is alive,
training must continue.
"""

import pytest

from hagi.config import Config, validate_config
from hagi.train.loop import liveness_two_axis_frozen

ETOL = 0.005
GTOL = 1e-4


def test_frozen_both_axes_stops():
    energy = [5.000, 5.001, 4.999, 5.000, 5.001]  # flat within tol
    growth = [1.0000, 1.00001, 0.99999, 1.0000, 1.00002]  # flat
    assert liveness_two_axis_frozen(energy, growth, ETOL, GTOL)


def test_alive_energy_continues():
    energy = [5.10, 5.05, 5.00, 4.95, 4.90]  # decreasing > tol
    growth = [1.0000, 1.00001, 0.99999, 1.0000, 1.00002]  # flat
    assert not liveness_two_axis_frozen(energy, growth, ETOL, GTOL)


def test_alive_growth_continues():
    energy = [5.000, 5.001, 4.999, 5.000, 5.001]  # flat
    growth = [1.00, 1.05, 1.02, 1.10, 1.03]  # injection signal varies
    assert not liveness_two_axis_frozen(energy, growth, ETOL, GTOL)


def test_short_window_never_stops():
    assert not liveness_two_axis_frozen([5.0], [1.0], ETOL, GTOL)
    assert not liveness_two_axis_frozen([], [], ETOL, GTOL)


def _base_cfg() -> Config:
    cfg = Config()
    cfg.model.vocab_size = 32768
    cfg.model.hidden_size = 128
    cfg.model.num_layers = 3
    cfg.model.attention.num_query_heads = 8
    cfg.model.attention.num_kv_heads = 4
    cfg.model.attention.head_dim = 16
    cfg.model.attention.max_seq_len = 512
    cfg.train.data.seq_len = 256
    return cfg


def test_config_liveness_defaults_off():
    cfg = _base_cfg()
    validate_config(cfg)  # default window=0 must validate
    assert cfg.train.liveness_window == 0


def test_config_liveness_validation():
    cfg = _base_cfg()
    cfg.train.liveness_window = 5  # < 10 -> reject
    with pytest.raises(ValueError, match="liveness_window"):
        validate_config(cfg)
    cfg.train.liveness_window = 50
    cfg.train.liveness_energy_tol = 0.0  # non-positive -> reject
    with pytest.raises(ValueError, match="liveness_energy_tol"):
        validate_config(cfg)
    cfg.train.liveness_energy_tol = 0.005
    validate_config(cfg)
