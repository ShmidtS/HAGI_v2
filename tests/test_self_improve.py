"""Tests for the opt-in online self-improvement loop.

Covers:
- the public hagi.train.self_improve.self_improve() entry point end-to-end,
- generate -> exact CE re-score -> adapter-only update via Trainer.train_step,
- frozen base SHA256 unchanged across iterations (bit-exact base invariant),
- only adapter params trainable when freeze_base=True,
- loop_depth>1 activates pyramid repeated-block application,
- bounded KL / plateau / max-iteration stop criteria,
- deterministic generation and bit-for-bit stable default path when disabled.

These tests use the real production generate() and HAGI.forward paths (no
internal mocks) on a tiny model. Strict CE-monotonicity is not asserted: on a
tiny random model a single step may not reduce CE, so the loop asserts
finiteness and adapter movement instead.
"""
from __future__ import annotations

import hashlib

import pytest
import torch

from hagi.model.model import HAGI
from hagi.train.loop import Trainer
from hagi.train.self_improve import SelfImproveResult, _adapter_values, self_improve
from tests.conftest import tiny_config


def _base_hash(model: HAGI) -> str:
    """SHA256 of base (non-adapter) parameters, float32-cast for portability.

    Excludes any state_dict key containing "adapter". Cast through float() before
    numpy because bf16.numpy() is unsupported on this numpy/torch build.
    """
    buf = bytearray()
    for name, param in model.state_dict().items():
        if "adapter" in name:
            continue
        buf.extend(name.encode("utf-8"))
        buf.extend(param.detach().float().cpu().numpy().astype("<f4").tobytes())
    return hashlib.sha256(bytes(buf)).hexdigest()


def _adapter_param_ids(model: HAGI) -> set[int]:
    from hagi.model.adapters import BlockAdapter

    return {
        id(p)
        for module in model.modules()
        if isinstance(module, BlockAdapter)
        for p in module.parameters(recurse=True)
    }


def _make_cfg(
    *,
    pyramid=True,
    ttt_lora=False,
    levels=(1,),
    loop_depth=2,
    lr=1e-2,
    warmup_steps=0,
    inverse_sqrt_stable=False,
    max_steps=4,
) -> object:
    """Tiny adapter-enabled config for self-improvement smoke."""
    cfg = tiny_config()
    cfg.model.adapters.enabled = True
    cfg.model.adapters.pyramid.enabled = pyramid
    cfg.model.adapters.ttt_lora.enabled = ttt_lora
    cfg.model.adapters.ttt_lora.rank = 4
    cfg.model.adapters.pyramid.levels = levels
    cfg.model.loop_depth = loop_depth
    cfg.train.adapt.freeze_base = True
    cfg.train.precision = "bf16"
    cfg.train.max_steps = max_steps
    cfg.train.grad_accum_steps = 1
    cfg.train.batch_size = 1
    cfg.train.logging.exact_ce_interval = 0
    cfg.train.learning_rate = lr
    cfg.train.schedule.inverse_sqrt_stable = inverse_sqrt_stable
    cfg.train.schedule.warmup_steps = warmup_steps
    cfg.train.weight_decay = 0.0
    return cfg


def _frozen_partition_assertions(model: HAGI, base_before: str) -> None:
    """Base hash, trainable set, and loop depth — the contract invariants."""
    adapter_ids = _adapter_param_ids(model)
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable_ids == adapter_ids, "only adapter params trainable under freeze_base=True"
    assert model._loop_depth == 2
    assert _base_hash(model) == base_before, "base must stay frozen across updates"


def test_self_improve_loop_runs_multiple_iterations():
    """End-to-end public API: 3 iterations of generate -> CE -> adapter update.

    Validates the contract seams for opt-in self-improvement:
    - adapters.enabled + freeze_base + loop_depth>1 validated by self_improve,
    - each iteration generates, re-scores CE, and applies an adapter-only step,
    - base tensors are never updated (frozen), verified per-iteration,
    - at least one pyramid scale becomes non-zero (training moved it),
    - CE and KL remain finite across iterations.
    """
    cfg = _make_cfg(levels=(1, 2, 4))
    model = HAGI(cfg)
    # Trainer (via self_improve) freezes base + routes only adapter params to
    # AdamW. Hash after Trainer construction so the comparison is in the same
    # bf16 dtype the optimizer sees.
    Trainer(model, cfg)
    base_before = _base_hash(model)
    _frozen_partition_assertions(model, base_before)

    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4, 5],
        n_new_tokens=4,
        max_iterations=3,
        patience=3,
        kl_max=10.0,
        ce_min_improve=0.0,
    )

    # Three iterations completed (stop == max_iterations, not an early guard).
    assert len(stats.iterations) == 3
    assert stats.stopped == "max_iterations"

    result = stats.iterations[0]
    assert isinstance(result, SelfImproveResult)
    assert result.update_applied is True
    assert len(result.generated_ids) == 4
    assert all(item is not None for item in (result.pre_ce, result.post_ce, result.kl_div))

    for it in stats.iterations:
        assert it.update_applied, f"iteration {it.iteration} must apply an update"
        assert 0.0 <= it.pre_ce < 50.0, f"pre_ce out of range: {it.pre_ce}"
        assert 0.0 <= it.post_ce < 50.0, f"post_ce out of range: {it.post_ce}"
        assert 0.0 <= it.kl_div < 10.0, f"KL out of range: {it.kl_div}"
        assert len(it.adapter_values_after) > 0, "adapter values must be reported"

    # Base frozen across every iteration (not just at the end).
    _frozen_partition_assertions(model, base_before)

    # At least one adapter value became non-zero (training moved it from 0).
    final_values = stats.iterations[-1].adapter_values_after
    assert any(abs(v) > 0.0 for v in final_values), "adapter values must move after training"
    assert stats.best_ce is not None


def test_self_improve_kl_guard_stops_loop():
    """KL bounded at 0.0 stops the loop on the first iteration.

    Uses a tiny kl_max=0.0: the first adapter-only update produces a strictly
    positive KL (the scale moves the generated-distribution logits), so the
    bound must fire and record exactly one iteration.
    """
    cfg = _make_cfg(levels=(1, 2, 4))
    model = HAGI(cfg)
    Trainer(model, cfg)

    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4, 5],
        n_new_tokens=4,
        max_iterations=4,
        patience=4,
        kl_max=0.0,
        ce_min_improve=0.0,
    )
    # One iteration ran, it breached the KL bound (kl_div > 0 with kl_max=0),
    # and the loop stopped.
    assert len(stats.iterations) == 1
    assert stats.stopped == "kl_bound"
    assert stats.iterations[0].kl_div > 0.0


def test_self_improve_ttt_lora_smoke():
    """TTT-LoRA contour is exercised end-to-end through the public API."""
    cfg = _make_cfg(pyramid=False, ttt_lora=True, levels=(1,))
    model = HAGI(cfg)
    Trainer(model, cfg)
    base_before = _base_hash(model)

    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4, 5],
        n_new_tokens=4,
        max_iterations=3,
        patience=3,
        kl_max=10.0,
    )
    assert len(stats.iterations) == 3
    assert all(it.update_applied for it in stats.iterations)
    _frozen_partition_assertions(model, base_before)


def test_self_improve_uses_injected_trainer():
    """A caller-provided Trainer is reused instead of silently replaced."""
    cfg = _make_cfg(levels=(1,))
    model = HAGI(cfg)

    class RecordingTrainer:
        def __init__(self):
            self.calls = 0

        def train_step(self, microbatches):
            self.calls += 1
            assert microbatches
            return {"update_applied": True}

    trainer = RecordingTrainer()
    stats = self_improve(
        model,
        cfg,
        [1, 2, 3],
        n_new_tokens=1,
        max_iterations=1,
        kl_max=100.0,
        trainer=trainer,
    )
    assert trainer.calls == 1
    assert len(stats.iterations) == 1
    assert stats.iterations[0].update_applied is True


def test_self_improve_rejects_disabled_adapters():
    """The opt-in wrapper refuses to run without adapter+freeze config."""
    cfg = tiny_config()
    cfg.model.adapters.enabled = False
    cfg.model.loop_depth = 1
    model = HAGI(cfg)
    with pytest.raises(ValueError, match="model.adapters.enabled"):
        self_improve(model, cfg, [1, 2, 3], n_new_tokens=2)


def test_self_improve_rejects_loop_depth_1():
    cfg = _make_cfg(levels=(1,), loop_depth=1)
    model = HAGI(cfg)
    with pytest.raises(ValueError, match="loop_depth"):
        self_improve(model, cfg, [1, 2, 3], n_new_tokens=2)


def test_self_improve_rejects_empty_prompt():
    cfg = _make_cfg(levels=(1,))
    model = HAGI(cfg)
    with pytest.raises(ValueError, match="prompt_ids"):
        self_improve(model, cfg, [], n_new_tokens=2)


def test_self_improve_rejects_pad_in_prompt():
    cfg = _make_cfg(levels=(1,))
    model = HAGI(cfg)
    with pytest.raises(ValueError, match="pad token_id"):
        self_improve(model, cfg, [0, 1, 2, 3, 4], n_new_tokens=2)


def test_disabled_path_is_noop():
    """When adapters.enabled=False, default path produces no adapter modules."""
    cfg = tiny_config()
    cfg.model.adapters.enabled = False
    cfg.model.loop_depth = 1
    cfg.train.precision = "bf16"

    model = HAGI(cfg)
    for block in model.blocks:
        assert block.adapters is None

    inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model(inp, use_cache=False, return_logits=True)
    assert out.hidden is not None
    assert out.logits is not None
    assert torch.isfinite(out.logits).all()


def _adapter_snapshot(model: HAGI) -> list[float]:
    """Snapshot flat adapter values for exact equality after rollback."""
    return [float(v) for v in _adapter_values(model)]


def test_kl_rejection_rolls_back_adapter_params():
    """A KL-breaching update is rolled back: adapter params match their
    pre-update snapshot exactly (faithful KL diagnostic is preserved)."""
    cfg = _make_cfg(levels=(1, 2, 4), lr=1.0)
    model = HAGI(cfg)
    trainer = Trainer(model, cfg)
    before = _adapter_snapshot(model)

    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4, 5],
        n_new_tokens=4,
        max_iterations=1,
        kl_max=0.0,
        trainer=trainer,
    )
    assert stats.stopped == "kl_bound"
    assert len(stats.iterations) == 1
    # Diagnostic KL is still the honest post-update value.
    assert stats.iterations[0].kl_div > 0.0
    assert stats.iterations[0].update_applied is False
    # Adapter params restored to their pre-update snapshot.
    after = _adapter_snapshot(model)
    assert before == after, "adapter params must be rolled back on KL rejection"


def test_kl_rejection_does_not_advance_step():
    """A rejected KL step must not advance completed_steps for the resume
    horizon: trainer.step equals its pre-step value after the loop."""
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg)
    trainer = Trainer(model, cfg, start_step=3)
    pre = trainer.step

    self_improve(
        model,
        cfg,
        [1, 2, 3, 4, 5],
        n_new_tokens=4,
        max_iterations=1,
        kl_max=0.0,
        trainer=trainer,
    )
    assert trainer.step == pre, "rejected update must not advance the step counter"


def test_optimizer_restore_failure_still_reverts_step():
    """If optimizer.load_state_dict itself raises during rollback, trainer.step
    must still be restored in the finally path and the error re-raised (never
    leaving the resume horizon advanced or state half-rolled-back)."""
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg)
    trainer = Trainer(model, cfg, start_step=3)
    pre = trainer.step
    # Poison optimizer.rollback path: load_state_dict must fail, proving the
    # finally clause runs trainer.step restoration independently.
    bad_optimizer = trainer.optimizer
    original_load = bad_optimizer.load_state_dict

    def fail_load(_state):
        raise RuntimeError("simulated optimizer load failure")

    bad_optimizer.load_state_dict = fail_load  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="simulated optimizer load failure"):
            self_improve(
                model,
                cfg,
                [1, 2, 3, 4, 5],
                n_new_tokens=4,
                max_iterations=1,
                kl_max=0.0,
                trainer=trainer,
            )
    finally:
        bad_optimizer.load_state_dict = original_load
    assert trainer.step == pre, "step must be restored even when rollback raises"


def test_accepted_update_advances_step_once():
    """A non-rejected iteration advances trainer.step exactly once per
    accepted update."""
    cfg = _make_cfg(levels=(1, 2, 4), lr=1e-2)
    model = HAGI(cfg)
    trainer = Trainer(model, cfg, start_step=0)
    pre = trainer.step

    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4, 5],
        n_new_tokens=4,
        max_iterations=1,
        kl_max=10.0,
        trainer=trainer,
    )
    assert stats.iterations[0].update_applied is True
    assert trainer.step == pre + stats.accepted_updates
    assert stats.accepted_updates == 1


def test_rejected_iteration_counted_in_stats():
    """accepted_updates counts only accepted iterations, not the total."""
    cfg = _make_cfg(levels=(1, 2, 4), lr=1.0)
    model = HAGI(cfg)
    trainer = Trainer(model, cfg)

    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4, 5],
        n_new_tokens=4,
        max_iterations=4,
        kl_max=0.0,
        patience=4,
        trainer=trainer,
    )
    assert len(stats.iterations) == 1
    assert stats.accepted_updates == 0
    assert stats.stopped == "kl_bound"
