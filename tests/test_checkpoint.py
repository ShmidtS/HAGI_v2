"""Checkpoint contract: schema validation before mutation, atomic write, rotation.

A checkpoint is a contract between two processes weeks apart. The failure this
file exists to prevent is a *partially applied* load — a model that runs, trains,
and is silently wrong. So every rejection path is asserted, and each one is
checked to leave the model untouched.
"""

from __future__ import annotations

import pytest
import torch

from hagi.config import CHECKPOINT_FORMAT_VERSION
from hagi.model.model import HAGI
from hagi.train.checkpoint import (
    IncompatibleCheckpointError,
    config_from_dict,
    config_to_dict,
    latest_checkpoint,
    load_model,
    load_payload,
    save_checkpoint,
)
from hagi.train.optim import build_optimizer
from tests.conftest import tiny_config


@pytest.fixture
def saved(tmp_path):
    cfg = tiny_config()
    model = HAGI(cfg)
    path = save_checkpoint(model, cfg, 100, tmp_path)
    return path, cfg, model


class TestRoundTrip:
    def test_save_then_load(self, saved):
        path, cfg, model = saved
        fresh = HAGI(tiny_config())
        step, loaded_cfg = load_model(path, fresh)
        assert step == 100
        assert loaded_cfg.model.hidden_size == cfg.model.hidden_size
        for (name, a), (_, b) in zip(
            model.named_parameters(), fresh.named_parameters(), strict=True
        ):
            assert torch.equal(a.detach(), b.detach()), f"{name} differs after reload"

    def test_optimizer_state_survives(self, tmp_path):
        cfg = tiny_config()
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        ids = torch.randint(0, cfg.model.vocab_size, (2, 8))
        model(ids, ids).loss.backward()
        opt.step()
        path = save_checkpoint(model, cfg, 5, tmp_path, optimizer=opt)
        state = load_payload(path)["optimizer"]
        assert set(state) == {"muon", "adamw"}

    def test_config_dict_round_trip(self):
        cfg = tiny_config(**{"model.sliding.window": 16})
        back = config_from_dict(config_to_dict(cfg))
        assert back.model.sliding.window == 16

    def test_receiver_gain_is_persisted(self, saved):
        path, _, model = saved
        with torch.no_grad():
            model.head.logit_scale.fill_(0.5)
        path2 = save_checkpoint(model, tiny_config(), 7, path.parent)
        fresh = HAGI(tiny_config())
        load_model(path2, fresh)
        assert float(fresh.head.logit_scale.detach()) == pytest.approx(0.5)

    def test_moe_controller_state_is_persisted(self, tmp_path):
        cfg = tiny_config(
            **{"model.moe.enabled": True, "model.moe.num_experts": 4, "model.moe.moe_every": 2}
        )
        model = HAGI(cfg)
        moe = next(b.mixer for b in model.blocks if b.is_moe)
        with torch.no_grad():
            moe.expert_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.2]))
        path = save_checkpoint(model, cfg, 1, tmp_path)
        fresh = HAGI(cfg)
        load_model(path, fresh)
        fresh_moe = next(b.mixer for b in fresh.blocks if b.is_moe)
        assert torch.allclose(fresh_moe.expert_bias, moe.expert_bias)


class TestRejection:
    def payload(self, saved, **changes):
        path, _, _ = saved
        data = torch.load(path, map_location="cpu", weights_only=True)
        data.update(changes)
        return data

    def write(self, tmp_path, data):
        target = tmp_path / "bad.pt"
        torch.save(data, target)
        return target

    def test_wrong_format_version(self, saved, tmp_path):
        bad = self.write(tmp_path, self.payload(saved, format_version=CHECKPOINT_FORMAT_VERSION - 1))
        with pytest.raises(IncompatibleCheckpointError, match="format_version"):
            load_payload(bad)

    def test_unknown_field(self, saved, tmp_path):
        bad = self.write(tmp_path, self.payload(saved, surprise=1))
        with pytest.raises(IncompatibleCheckpointError, match="unknown fields"):
            load_payload(bad)

    def test_missing_field(self, saved, tmp_path):
        data = self.payload(saved)
        del data["completed_steps"]
        with pytest.raises(IncompatibleCheckpointError, match="missing fields"):
            load_payload(self.write(tmp_path, data))

    def test_negative_step_count(self, saved, tmp_path):
        bad = self.write(tmp_path, self.payload(saved, completed_steps=-1))
        with pytest.raises(IncompatibleCheckpointError):
            load_payload(bad)

    def test_model_must_map_names_to_tensors(self, saved, tmp_path):
        bad = self.write(tmp_path, self.payload(saved, model={"a": "not a tensor"}))
        with pytest.raises(IncompatibleCheckpointError):
            load_payload(bad)

    def test_unreadable_payload(self, tmp_path):
        broken = tmp_path / "broken.pt"
        broken.write_bytes(b"not a torch file")
        with pytest.raises(IncompatibleCheckpointError):
            load_payload(broken)

    def test_architecture_mismatch_is_strict(self, saved):
        """A mismatched state_dict must raise, never load the subset that matches."""
        path, _, _ = saved
        other = HAGI(tiny_config(**{"model.num_layers": 2}))
        with pytest.raises(IncompatibleCheckpointError):
            load_model(path, other)

    def test_failed_load_leaves_the_model_untouched(self, saved):
        path, _, _ = saved
        other = HAGI(tiny_config(**{"model.num_layers": 2}))
        before = {n: p.detach().clone() for n, p in other.named_parameters()}
        with pytest.raises(IncompatibleCheckpointError):
            load_model(path, other)
        for name, p in other.named_parameters():
            assert torch.equal(p.detach(), before[name]), f"{name} was mutated by a failed load"

    def test_invalid_stored_config(self, saved, tmp_path):
        data = self.payload(saved)
        data["config"]["model"]["num_layers"] = 0
        with pytest.raises(IncompatibleCheckpointError):
            load_model(self.write(tmp_path, data), HAGI(tiny_config()))


class TestRotation:
    def test_keeps_only_the_last_n(self, tmp_path):
        cfg = tiny_config()
        model = HAGI(cfg)
        for step in (1, 2, 3, 4, 5):
            save_checkpoint(model, cfg, step, tmp_path, keep_last=2)
        remaining = sorted(p.name for p in tmp_path.glob("step-*.pt"))
        assert remaining == ["step-0000004.pt", "step-0000005.pt"]

    def test_no_temporary_files_left(self, tmp_path):
        cfg = tiny_config()
        save_checkpoint(HAGI(cfg), cfg, 1, tmp_path)
        assert not list(tmp_path.glob("*.tmp"))

    def test_latest_uses_numeric_order(self, tmp_path):
        cfg = tiny_config()
        model = HAGI(cfg)
        for step in (9, 10, 100):
            save_checkpoint(model, cfg, step, tmp_path, keep_last=5)
        assert latest_checkpoint(tmp_path).name == "step-0000100.pt"

    def test_latest_on_empty_directory(self, tmp_path):
        assert latest_checkpoint(tmp_path) is None

    @pytest.mark.parametrize("bad", [-1, 1.5, "3"])
    def test_invalid_step_count_rejected(self, tmp_path, bad):
        cfg = tiny_config()
        with pytest.raises(ValueError):
            save_checkpoint(HAGI(cfg), cfg, bad, tmp_path)

    def test_keep_last_must_be_positive(self, tmp_path):
        cfg = tiny_config()
        with pytest.raises(ValueError):
            save_checkpoint(HAGI(cfg), cfg, 1, tmp_path, keep_last=0)
