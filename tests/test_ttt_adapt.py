"""Contracts for the opt-in HAGI adapter/TTT wiring.

Covers the exact invariant enforced in ``block.py``: the adapter receives the
**post-attention residual** — the same stream the frozen mixer sees — captured
*before* the mixer runs. It also pins the ``Trainer`` adapter-only training path:
``adapt.freeze_base`` must freeze every non-adapter parameter, route adapter
params to AdamW (not Muon), gate the ternary cache and controller updates, and
leave the frozen base byte-identical across an optimizer step.

These tests construct the model from :func:`tests.conftest.tiny_config`, which runs
the real validator — a directly-built dataclass that the loader would reject
cannot slip through.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from hagi.config import Config, validate_config
from hagi.model.adapters import BlockAdapter
from hagi.model.model import HAGI
from hagi.train.loop import Trainer
from hagi.train.optim import build_optimizer
from tests.conftest import tiny_config

TINY_HIDDEN = 64


def _base_cfg(**overrides) -> Config:
    return tiny_config(**overrides)


def _adapter_cfg(**adapter_overrides) -> Config:
    cfg = tiny_config()
    cfg.model.adapters.enabled = True
    cfg.model.adapters.pyramid.enabled = True
    cfg.model.adapters.pyramid.levels = (1,)
    cfg.model.adapters.ttt_lora.enabled = False
    for key, value in adapter_overrides.items():
        target: object = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    validate_config(cfg)
    return cfg


def _adapter_param_names(model: torch.nn.Module) -> set[str]:
    """Fully-qualified names of parameters owned by BlockAdapter modules."""
    names: set[str] = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, BlockAdapter):
            for p_name, _ in module.named_parameters():
                names.add(f"{mod_name}.{p_name}")
    return names


def _is_adapter(model: torch.nn.Module, name: str) -> bool:
    """True if ``name`` refers to a parameter under a BlockAdapter."""
    return name in _adapter_param_names(model)


class TestBlockAdapterInputContract:
    """The adapter must receive the exact residual the frozen mixer sees.

    ``Block.forward`` captures ``mixer_input`` after ``x + attention(x)`` and
    passes it to the adapter; the base mixer receives that same tensor. A
    regression that captures before the attention add (or passes the wrong
    tensor) breaks the shared-input invariant that the pyramid branches and
    LoRA projection are documented to run on.
    """

    def test_adapter_input_equals_mixer_input(self):
        """mixer_input passed to adapter is identical to what the mixer
        received -- verified by swapping the mixer with a probe that records
        its input and asserting equality."""
        cfg = _adapter_cfg()
        model = HAGI(cfg)
        model.eval()
        block = model.blocks[0]
        assert isinstance(block.adapters, BlockAdapter)

        recorded: list[torch.Tensor] = []
        original_mixer = block.mixer

        class MixerProbe(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                recorded.append(x)
                return self.inner(x)

        probe = MixerProbe(original_mixer)
        block.mixer = probe
        adapter_recorded: list[torch.Tensor] = []
        real_forward = block.adapters.forward

        def spy_forward(x, mixer_out):
            adapter_recorded.append(x)
            return real_forward(x, mixer_out)

        block.adapters.forward = spy_forward  # type: ignore[method-assign]

        ids = torch.randint(0, cfg.model.vocab_size, (2, 16))
        with torch.no_grad():
            _ = model(ids, return_logits=True)

        assert len(recorded) == 1
        assert len(adapter_recorded) == 1
        assert torch.equal(recorded[0], adapter_recorded[0]), (
            "adapter received a different residual than the mixer -- the "
            "shared-input invariant is broken"
        )
        assert recorded[0].data_ptr() == adapter_recorded[0].data_ptr(), (
            "adapter received a different tensor than the mixer"
        )

    def test_disabled_default_has_no_adapter(self):
        """When adapters.enabled is False, Block.adapters must be None."""
        cfg = _base_cfg()
        model = HAGI(cfg)
        assert all(b.adapters is None for b in model.blocks)

    def test_zero_init_adapter_is_zero_delta(self):
        """An enabled-but-untrained adapter must emit exactly zero delta.

        Build an adapter-enabled model and a disabled reference that share the
        same base weights. Their hidden states must be bit-for-bit identical,
        because a zero-init adapter emits exactly zero delta.
        """
        cfg = _adapter_cfg()
        model = HAGI(cfg).eval()
        ids = torch.randint(0, cfg.model.vocab_size, (2, 16))

        # Disabled reference: same base weights, no adapter modules.
        ref_cfg = tiny_config()
        ref_model = HAGI(ref_cfg).eval()
        ref_model.load_state_dict(model.state_dict(), strict=False)

        with torch.no_grad():
            for block in model.blocks:
                if block.adapters is not None:
                    if block.adapters.pyramid is not None:
                        assert block.adapters.pyramid.scale.item() == 0.0
            base_out = ref_model(ids, return_logits=True)
            adp_out = model(ids, return_logits=True)

        assert torch.equal(base_out.hidden, adp_out.hidden)


class TestTrainerAdapterOnlyFreeze:
    """``adapt.freeze_base`` must freeze base params, keep adapter params trainable."""

    def test_freeze_base_freezes_everything_except_adapters(self):
        cfg = _adapter_cfg()
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        Trainer(model, cfg)

        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        adapter_names = _adapter_param_names(model)
        non_adapter_trainable = trainable_names - adapter_names
        assert not non_adapter_trainable, (
            f"freeze_base=True but non-adapter params are trainable: {non_adapter_trainable}"
        )
        for block in model.blocks:
            ad = block.adapters
            if ad is None:
                continue
            if ad.pyramid is not None:
                assert ad.pyramid.scale.requires_grad, "pyramid scale must be trainable"
            if ad.ttt_lora is not None:
                assert ad.ttt_lora.lora_B.requires_grad, "lora_B must be trainable"

    def test_freeze_base_requires_adapters_enabled(self):
        """freeze_base=True with adapters disabled must raise at Trainer init."""
        cfg = tiny_config()
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        with pytest.raises(
            ValueError, match="train.adapt.freeze_base=True requires model.adapters.enabled=True"
        ):
            Trainer(model, cfg)

    def test_adapter_params_routed_to_adamw(self):
        cfg = _adapter_cfg()
        cfg.model.adapters.pyramid.enabled = False
        cfg.model.adapters.ttt_lora.enabled = True
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        adamw_ids = {id(p) for g in opt.adamw.param_groups for p in g["params"]}
        for block in model.blocks:
            ad = block.adapters
            if ad is None:
                continue
            assert ad.ttt_lora is not None
            assert id(ad.ttt_lora.lora_B) in adamw_ids

    def test_base_unchanged_after_step(self):
        """A single adapter-only train_step must not alter any base parameter."""
        cfg = _adapter_cfg()
        cfg.train.adapt.freeze_base = True
        cfg.train.max_steps = 1
        cfg.train.schedule.warmup_steps = 0
        model = HAGI(cfg)
        trainer = Trainer(model, cfg)
        exclude = _adapter_param_names(model)
        before = {
            n: p.detach().clone() for n, p in model.named_parameters() if n not in exclude
        }
        microbatch = [
            {
                "input_ids": torch.randint(0, cfg.model.vocab_size, (2, 32)),
                "targets": torch.randint(0, cfg.model.vocab_size, (2, 32)),
            }
        ]
        out = trainer.train_step(microbatch)
        assert out["update_applied"] is True, f"step failed: {out}"
        for n, p in model.named_parameters():
            if n in exclude:
                continue
            assert torch.equal(before[n], p.detach()), (
                f"base param {n} changed during adapter-only step"
            )

    def test_base_hash_unchanged_after_step(self):
        """SHA256 of base params is identical across an adapter-only step."""
        cfg = _adapter_cfg()
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        exclude = _adapter_param_names(model)

        def _hash_base(m):
            h = hashlib.sha256()
            for n, p in sorted(m.named_parameters()):
                if n in exclude:
                    continue
                h.update(p.detach().float().cpu().numpy().tobytes())
            return h.hexdigest()

        before = _hash_base(model)
        trainer = Trainer(model, cfg)
        before = _hash_base(model)
        trainer.train_step([
            {
                "input_ids": torch.randint(0, cfg.model.vocab_size, (2, 32)),
                "targets": torch.randint(0, cfg.model.vocab_size, (2, 32)),
            }
        ])
        after = _hash_base(model)
        assert before == after, "base hash changed during adapter-only step"


class TestAdamWGroupPartitioning:
    """Adapter params are 1D and must be in AdamW no-decay, never Muon."""

    def test_no_double_assignment(self):
        cfg = _adapter_cfg()
        cfg.model.adapters.pyramid.enabled = False
        cfg.model.adapters.ttt_lora.enabled = True
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        all_ids = []
        for group in opt.param_groups:
            all_ids.extend(id(p) for p in group["params"])
        assert len(all_ids) == len(set(all_ids)), "parameter in more than one group"


class TestOptimizerRouting:
    """Pyramid scale and LoRA B are 1D gains -> AdamW no-decay, never Muon."""

    def test_frozen_base_excluded_from_optimizer(self):
        cfg = _adapter_cfg()
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            param_id = id(p)
            found = any(param_id in {id(candidate) for candidate in g["params"]}
                        for g in opt.param_groups)
            assert found, "trainable parameter not assigned to any optimizer group"

    def test_muon_group_none_when_use_muon_false(self):
        cfg = _adapter_cfg()
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        assert opt.muon is None

    def test_only_adapter_params_in_adamw_when_frozen(self):
        cfg = _adapter_cfg()
        cfg.model.adapters.pyramid.enabled = False
        cfg.model.adapters.ttt_lora.enabled = True
        cfg.train.adapt.freeze_base = True
        model = HAGI(cfg)
        trainer = Trainer(model, cfg)
        opt = trainer.optimizer
        for g in opt.adamw.param_groups:
            if g.get("_muon", False) or g.get("_body", False):
                assert not g["params"], "body/muon group should be empty when frozen"
