"""Arm S must differ from the merged arm ONLY in where the weights came from.

The M2 comparison was confounded partly by architecture: the merged checkpoint
carries ``BlockRMSNorm`` gains shaped (3, 384) and will not load into a plain
``HAGI``, while the baseline uses wide ``RMSNorm`` (1152,). To test merging
rather than architecture, the scratch arm needs the merged arm's structure.

These tests pin the structural equality that makes the comparison meaningful.
If they fail, an M-vs-S difference cannot be attributed to initialisation.
"""

from __future__ import annotations

import pytest
import torch

from hagi.config import load_config
from hagi.model.scratch_blocknorm import ScratchBlockNormHAGI
from hagi.train.checkpoint import config_from_dict, load_payload


@pytest.fixture(scope="module")
def merged_payload():
    return load_payload("checkpoints/m2_merged_joint/step-0009000.pt")


@pytest.fixture(scope="module")
def scratch(merged_payload):
    return ScratchBlockNormHAGI(
        config_from_dict(merged_payload["config"]), n_blocks=3
    )


def test_scratch_matches_merged_on_every_shared_key(scratch, merged_payload):
    """Same keys, same shapes: no architectural difference on the body."""
    scratch_state = scratch.state_dict()
    merged_state = merged_payload["model"]
    shared = set(scratch_state) & set(merged_state)
    assert shared, "expected the two models to share weights"
    mismatched = [
        key
        for key in shared
        if scratch_state[key].shape != merged_state[key].shape
    ]
    assert not mismatched, f"shape mismatch on {mismatched[:3]}"


def test_scratch_omits_only_the_mixers(scratch, merged_payload):
    """The only keys the merged arm has extra are the cross-block mixers.

    Those are 0.223M parameters (0.2% of the model) and their learned gain was
    -0.120, i.e. mixing was *shrunk* during joint training rather than
    composing the domains. Their contribution is therefore bounded and
    reported rather than equated away.
    """
    extra = set(merged_payload["model"]) - set(scratch.state_dict())
    assert all(key.startswith("mixers.") for key in extra), sorted(extra)
    extra_params = sum(merged_payload["model"][k].numel() for k in extra)
    assert extra_params / 1e6 < 0.3, f"mixer params grew to {extra_params/1e6:.2f}M"


def test_merged_weights_load_into_the_scratch_architecture(merged_payload, scratch):
    """The two arms are the same model, so the weights are interchangeable.

    This is the check that the merged checkpoint is a *model* rather than a
    different architecture: it loads with no missing and no unexpected keys
    beyond the mixers. If this fails, an M-vs-S comparison would be measuring
    architecture, not initialisation.
    """
    missing, unexpected = scratch.load_state_dict(merged_payload["model"], strict=False)
    assert not missing
    assert all(key.startswith("mixers.") for key in unexpected), unexpected


def test_norms_are_per_block_not_wide(scratch):
    """The defining structural property: per-expert normalisation."""
    assert scratch.blocks[0].attn.attn_norm.weight.shape == (3, 384)
    assert scratch.out_norm.weight.shape == (3, 384)


def test_rejects_an_indivisible_block_count(merged_payload):
    cfg = config_from_dict(merged_payload["config"])
    with pytest.raises(ValueError, match="divisible"):
        ScratchBlockNormHAGI(cfg, n_blocks=7)
