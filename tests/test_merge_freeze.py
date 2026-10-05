"""Tests for merge.freeze_experts.

The option was declared and documented in the config for the whole life
of merge.py and never implemented -- `grep -rn freeze_experts src/hagi`
returned the dataclass field and its docstring and nothing else. A
config setting it trained every parameter anyway, silently.

That matters beyond the missing feature: the gen-3 measurements found
the mixer's trainable gain DECAYING toward zero during joint training,
which says the optimiser preferred the channel off. With the experts
frozen the mixer is the only degree of freedom, so this mode is the
clean A/B for whether the channel pays.

These tests pin the two properties that make it a real mode rather than
a silent no-op in the other direction: the mixers stay trainable, and
almost nothing else does.
"""

from __future__ import annotations

import pytest
import torch

from hagi.config import load_config
from hagi.model.merge import MergedHAGI


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def build(config: str):
    cfg = load_config(str(ROOT / "configs" / config))
    return cfg, MergedHAGI(cfg, n_mixers=1,
                           mixer_init_scale=cfg.merge.mixer_init_scale)


# --- the option now does something -------------------------------------


def test_the_default_leaves_everything_trainable():
    """The common path must be untouched by the new code."""
    _, m = build("dbridge_gen6_joint.yaml")
    assert all(p.requires_grad for _, p in m.named_parameters())


def test_freezing_leaves_only_the_mixers_trainable():
    _, m = build("mixonly_gen3.yaml")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable
    for n in trainable:
        assert n.startswith("mixers."), f"{n} is not a mixer parameter"


def test_the_mixer_gain_stays_trainable():
    """The parameter the whole question is about."""
    _, m = build("mixonly_gen3.yaml")
    assert m.mixers[0].gain.requires_grad


def test_the_experts_are_actually_frozen():
    _, m = build("mixonly_gen3.yaml")
    # the embedding is the biggest tensor and must not move
    emb = m.encoder.embedding.weight
    assert emb.requires_grad is False


def test_freezing_removes_the_bulk_of_the_parameters():
    """The mode is only meaningful if it removes almost everything."""
    _, m = build("mixonly_gen3.yaml")
    frac = m.trainable_params / m.total_params
    assert frac < 0.01, f"only {frac:.2%} frozen -- not a mixer-only mode"


def test_the_reported_counts_match_the_parameters():
    _, m = build("mixonly_gen3.yaml")
    trainable = sum(p.numel() for _, p in m.named_parameters()
                    if p.requires_grad)
    total = sum(p.numel() for _, p in m.named_parameters())
    assert m.trainable_params == trainable
    assert m.total_params == total


# --- it fails loudly instead of silently doing nothing ------------------


def test_a_model_without_mixers_is_refused():
    """Freezing everything must be an error, not a run that cannot train."""
    _, m = build("mixonly_gen3.yaml")
    m.mixers = torch.nn.ModuleList()
    with pytest.raises(ValueError, match="no trainable parameters"):
        m._freeze_all_but_mixers()


def test_the_mode_reports_rather_than_assumes():
    """The caller can see what was frozen without walking named_parameters."""
    _, m = build("mixonly_gen3.yaml")
    assert hasattr(m, "trainable_params")
    assert m.trainable_params < m.total_params


# --- the step-0 invariant still holds -----------------------------------


def test_a_frozen_model_still_runs_forward():
    """Freezing is about gradients, not about the computation."""
    _, m = build("mixonly_gen3.yaml")
    ids = torch.randint(0, 32768, (2, 16))
    with torch.no_grad():
        out = m(ids, return_logits=True)
    assert out.logits is not None
    assert torch.isfinite(out.logits).all()