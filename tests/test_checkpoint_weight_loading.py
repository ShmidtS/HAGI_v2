"""Falsifier: a rebuilt model must carry the checkpoint's weights, not a fresh
random initialisation.

`build_model_from_payload` once constructed the model class from the config and
returned it without ever loading the state it was handed. Nothing raised: the
caller silently scored an unrelated model, and every CE delta measured that way
was noise. The pair SE (0.0251 nats) and the 9/9 lift deltas recorded in
AGENT_WORKLOG.md were taken through that path.

These tests assert behaviour, not implementation: they would fail again for any
build path that drops the weights, whether it forgets `load_state_dict` or
loads it into a discarded object.

Claim boundary: mechanism only. Nothing here asserts model quality, training
progress, or autonomy.
"""
from __future__ import annotations

import io

import torch

from hagi.model.merge import build_model_from_payload
from hagi.train.checkpoint import config_from_dict
from tests.conftest import tiny_config


def _payload(cfg, model):
    return {
        "format_version": 1,
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "config": config_to_payload(cfg),
        "completed_steps": 0,
    }


def config_to_payload(cfg):
    from hagi.train.checkpoint import config_to_dict

    return config_to_dict(cfg)


def _roundtrip(cfg, model):
    payload = _payload(cfg, model)
    rebuilt = build_model_from_payload(cfg, payload["model"], device="cpu")
    return payload["model"], rebuilt.state_dict()


def test_plain_model_state_is_actually_loaded():
    cfg = tiny_config()
    model = build_plain(cfg)
    saved, rebuilt = _roundtrip(cfg, model)

    key = "encoder.embedding.weight"
    assert key in saved, "fixture lost the key under test"
    assert torch.equal(rebuilt[key].cpu(), saved[key]), (
        "rebuilt model did not receive the checkpoint weights"
    )


def test_rebuilt_model_is_not_a_fresh_initialisation():
    """A same-shaped random init must NOT satisfy the equality above."""
    cfg = tiny_config()
    state = {k: v.detach().clone() for k, v in build_plain(cfg).state_dict().items()}
    state["encoder.embedding.weight"] = torch.zeros_like(state["encoder.embedding.weight"])

    rebuilt = build_model_from_payload(cfg, state, device="cpu")
    assert torch.equal(
        rebuilt.state_dict()["encoder.embedding.weight"].cpu(),
        torch.zeros_like(state["encoder.embedding.weight"]),
    ), "zeroed checkpoint weights were replaced by random ones"


def test_every_key_roundtrips_bitwise():
    cfg = tiny_config()
    model = build_plain(cfg)
    saved, rebuilt = _roundtrip(cfg, model)
    missing = set(saved) - set(rebuilt)
    assert not missing, f"keys absent after rebuild: {sorted(missing)[:3]}"
    for key, value in saved.items():
        assert torch.equal(rebuilt[key].cpu(), value), f"tensor diverged: {key}"


def build_plain(cfg):
    from hagi.model.model import HAGI

    torch.manual_seed(1234)
    return HAGI(cfg)


def test_checkpoint_serialisation_path_loads_weights():
    """Same property through the real save/load path, not a hand-made dict."""
    from hagi.train.checkpoint import save_checkpoint

    cfg = tiny_config()
    model = build_plain(cfg)
    buf = io.BytesIO()
    torch.save(
        {
            "format_version": 1,
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "config": config_to_payload(cfg),
            "completed_steps": 0,
        },
        buf,
    )
    del save_checkpoint
    payload = torch.load(io.BytesIO(buf.getvalue()), map_location="cpu", weights_only=True)
    rebuilt_cfg = config_from_dict(payload["config"])
    rebuilt = build_model_from_payload(rebuilt_cfg, payload["model"], device="cpu")
    for key, value in payload["model"].items():
        assert torch.equal(rebuilt.state_dict()[key].cpu(), value), key
