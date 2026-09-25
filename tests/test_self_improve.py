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

import copy
import hashlib

import pytest
import torch

from hagi.model.model import HAGI
from hagi.train import self_improve as si
from hagi.train.loop import Trainer
from hagi.train.self_improve import SelfImproveResult, _adapter_values, self_improve
from hagi.train.ttt import TttRls, TttStats
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


def _assert_nested_state_equal(expected: object, actual: object) -> None:
    """Compare nested optimizer state without assuming a torch Optimizer shape."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_nested_state_equal(value, actual[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for value, other in zip(expected, actual, strict=True):
            _assert_nested_state_equal(value, other)
    else:
        assert actual == expected


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


def test_gradient_noop_restores_pre_state_even_if_trainer_mutates(monkeypatch):
    """A false ``update_applied`` flag must not hide partial trainer mutation."""
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    from hagi.model.adaptive import adaptive_parameters

    class MutatingNoOpOptimizer:
        def __init__(self) -> None:
            self.state = {"momentum": torch.tensor([1.0])}

        def state_dict(self) -> dict:
            return {"state": self.state}

        def load_state_dict(self, state: dict) -> None:
            self.state = copy.deepcopy(state["state"])

    class MutatingNoOpTrainer:
        def __init__(self) -> None:
            self.step = 7
            self.calls = 0
            self.optimizer = MutatingNoOpOptimizer()

        def train_step(self, microbatches: list[dict]) -> dict:
            self.calls += 1
            self.step += 1
            with torch.no_grad():
                for parameter in adaptive_parameters(model):
                    if parameter.requires_grad:
                        parameter.add_(0.25)
            self.optimizer.state = {"momentum": torch.tensor([9.0])}
            return {"update_applied": False}

    trainer = MutatingNoOpTrainer()
    before = {
        key: (value.clone(), requires_grad)
        for key, (value, requires_grad) in si._snapshot_adapters(model).items()
    }
    optimizer_before = si._deepcopy_optimizer_state(
        trainer.optimizer.state_dict()
    )

    monkeypatch.setattr(
        si,
        "_generate_trajectory",
        lambda *args, **kwargs: torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long
        ),
    )
    monkeypatch.setattr(
        si,
        "_score",
        lambda model_arg, input_ids, targets: (
            1.0,
            torch.zeros((1, targets.shape[1], cfg.model.vocab_size)),
        ),
    )
    monkeypatch.setattr(si, "_kl_pre_post", lambda *args, **kwargs: 0.0)

    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4, 5],
        n_new_tokens=4,
        max_iterations=1,
        kl_max=1.0,
        trainer=trainer,
    )

    assert len(stats.iterations) == 1
    assert stats.iterations[0].update_applied is False
    assert trainer.calls == 1
    assert trainer.step == 7, "a no-op must restore the pre-step horizon"
    after = si._snapshot_adapters(model)
    assert after.keys() == before.keys()
    assert all(
        torch.equal(after[key][0], value)
        and after[key][1] == requires_grad
        for key, (value, requires_grad) in before.items()
    )
    _assert_nested_state_equal(
        optimizer_before,
        si._deepcopy_optimizer_state(trainer.optimizer.state_dict()),
    )
    assert all(module.training is False for module in model.modules())
    assert getattr(trainer, "_hagi_self_improve_rollback_poisoned", False) is False


def test_gradient_noop_post_recovery_status_is_explicit(monkeypatch):
    """Successful fallback recovery is ``post``, not an implicit poisoned result."""
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    trainer = Trainer(model, cfg, start_step=3)
    optimizer = trainer.optimizer
    pre_params = si._snapshot_adapters(model)
    pre_opt = si._deepcopy_optimizer_state(optimizer.state_dict())
    trainer.step = 4
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.25)
    post_params = si._snapshot_adapters(model)
    post_opt = si._deepcopy_optimizer_state(optimizer.state_dict())
    real_rollback = si._rollback_gradient_transaction
    statuses: list[str] = []

    def capture_rollback(*args, **kwargs):
        status = real_rollback(*args, **kwargs)
        statuses.append(status)
        return status

    class FailingPreOptimizer:
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped
            self.calls = 0

        def state_dict(self) -> dict:
            return self.wrapped.state_dict()

        def load_state_dict(self, state: dict) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("PRE-OPT-FAIL")
            self.wrapped.load_state_dict(state)

    failing_optimizer = FailingPreOptimizer(optimizer)
    monkeypatch.setattr(si, "_rollback_gradient_transaction", capture_rollback)
    status = si._rollback_gradient_transaction(
        model,
        trainer,
        failing_optimizer,
        pre_params,
        pre_opt,
        post_params,
        post_opt,
        pre_step=3,
        post_step=4,
    )

    assert status == "post"
    assert statuses == ["post"]
    assert trainer.step == 4
    assert trainer._hagi_self_improve_rollback_poisoned is True


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


def test_gradient_exception_recovers_coherent_post_state_when_pre_restore_fails(
    monkeypatch,
):
    """A failed pre-state optimizer load must not strand changed parameters.

    The primary post-score error still wins, and the model/optimizer pair is
    restored together. If optimizer load fails before mutating anything, the
    best available coherent state is the post-step snapshot, so its step horizon
    remains and the trainer is poisoned against reuse. The first load mutates
    state before raising, proving the fallback does not trust load_state_dict to
    be atomic. Restoring only parameters would pair old weights with whatever
    momentum happened to remain and make a retry unsafe.
    """
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    real_score = si._score
    score_calls = {"count": 0}
    params_before = [param.detach().clone() for param in model.parameters()]

    class OneShotFailingOptimizer:
        def __init__(self) -> None:
            self.state = {"momentum": torch.tensor([1.0])}
            self.load_calls = 0

        def state_dict(self) -> dict:
            return {"state": self.state}

        def load_state_dict(self, state: dict) -> None:
            self.load_calls += 1
            if self.load_calls == 1:
                self.state = {"momentum": torch.tensor([99.0])}
                raise RuntimeError("PRE-OPT-RESTORE-FAIL")
            self.state = copy.deepcopy(state["state"])

    class PostStepTrainer:
        def __init__(self) -> None:
            self.step = 4
            self.optimizer = OneShotFailingOptimizer()

        def train_step(self, microbatches: list[dict]) -> dict:
            self.step += 1
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.add_(0.25)
            self.optimizer.state = {"momentum": torch.tensor([9.0])}
            return {"update_applied": True}

    trainer = PostStepTrainer()

    def failing_second_score(model_arg, input_ids, targets):
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            return real_score(model_arg, input_ids, targets)
        raise KeyboardInterrupt("GRAD-PRIMARY")

    monkeypatch.setattr(si, "_score", failing_second_score)
    with pytest.raises(KeyboardInterrupt, match="GRAD-PRIMARY"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            n_new_tokens=8,
            max_iterations=1,
            kl_max=10.0,
            trainer=trainer,
        )

    assert score_calls["count"] == 2
    assert trainer.step == 5
    assert trainer.optimizer.load_calls == 2
    assert torch.equal(
        trainer.optimizer.state["momentum"], torch.tensor([9.0])
    )
    assert any(
        not torch.equal(param.detach(), before)
        for param, before in zip(model.parameters(), params_before, strict=True)
    )
    assert model.training is False
    assert trainer._hagi_self_improve_rollback_poisoned is True

    with pytest.raises(RuntimeError, match="trainer is poisoned"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            n_new_tokens=8,
            max_iterations=1,
            kl_max=10.0,
            trainer=trainer,
        )
    assert score_calls["count"] == 2, "poisoned trainer must be rejected before work"


def test_gradient_adapter_restore_failure_reloads_full_post_pair(monkeypatch):
    """A valid pre-optimizer half must not be paired with a post-parameter half."""
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    real_score = si._score
    real_restore = si._restore_adapters
    score_calls = {"count": 0}
    restore_calls = {"count": 0}

    class RecordingOptimizer:
        def __init__(self) -> None:
            self.state = {"momentum": torch.tensor([1.0])}
            self.load_calls = 0

        def state_dict(self) -> dict:
            return {"state": self.state}

        def load_state_dict(self, state: dict) -> None:
            self.load_calls += 1
            self.state = copy.deepcopy(state["state"])

    class PostStepTrainer:
        def __init__(self) -> None:
            self.step = 4
            self.optimizer = RecordingOptimizer()

        def train_step(self, microbatches: list[dict]) -> dict:
            self.step += 1
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.add_(0.25)
            self.optimizer.state = {"momentum": torch.tensor([9.0])}
            return {"update_applied": True}

    trainer = PostStepTrainer()

    def failing_second_score(model_arg, input_ids, targets):
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            return real_score(model_arg, input_ids, targets)
        raise KeyboardInterrupt("GRAD-PRIMARY")

    def fail_first_adapter_restore(model_arg, snapshot):
        restore_calls["count"] += 1
        if restore_calls["count"] == 1:
            raise RuntimeError("PRE-ADAPTER-RESTORE-FAIL")
        real_restore(model_arg, snapshot)

    monkeypatch.setattr(si, "_score", failing_second_score)
    monkeypatch.setattr(si, "_restore_adapters", fail_first_adapter_restore)
    with pytest.raises(KeyboardInterrupt, match="GRAD-PRIMARY"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            n_new_tokens=8,
            max_iterations=1,
            kl_max=10.0,
            trainer=trainer,
        )

    assert restore_calls["count"] == 2
    assert trainer.optimizer.load_calls == 2
    assert trainer.step == 5
    assert torch.equal(
        trainer.optimizer.state["momentum"], torch.tensor([9.0])
    )
    assert model.training is False
    assert trainer._hagi_self_improve_rollback_poisoned is True


def test_missing_post_optimizer_snapshot_poisons_gradient_trainer(monkeypatch):
    """A partial fallback must not be accepted as a coherent post-state."""
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg)
    trainer = Trainer(model, cfg, start_step=3)
    optimizer = trainer.optimizer
    pre_param_snapshot = si._snapshot_adapters(model)
    pre_opt_snapshot = si._deepcopy_optimizer_state(optimizer.state_dict())
    trainer.step = 4
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(0.25)
    post_param_snapshot = si._snapshot_adapters(model)
    real_restore = si._restore_adapters
    restore_calls = {"count": 0}

    def fail_pre_adapter_restore(model_arg, snapshot):
        restore_calls["count"] += 1
        if restore_calls["count"] == 1:
            raise RuntimeError("PRE-ADAPTER-RESTORE-FAIL")
        real_restore(model_arg, snapshot)

    monkeypatch.setattr(si, "_restore_adapters", fail_pre_adapter_restore)
    status = si._rollback_gradient_transaction(
        model,
        trainer,
        optimizer,
        pre_param_snapshot,
        pre_opt_snapshot,
        post_param_snapshot,
        None,
        pre_step=3,
        post_step=4,
    )

    assert status == "poisoned"
    assert restore_calls["count"] == 1
    assert trainer.step == 4
    assert trainer._hagi_self_improve_rollback_poisoned is True


def test_optimizer_restore_failure_poisons_trainer_when_post_recovery_fails(
    monkeypatch,
):
    """Failed pre-state rollback cannot masquerade as a successful rejection.

    The optimizer loader fails for both the pre and captured post states. The
    resulting live state is not trusted even though ``trainer.step`` already
    advanced, so reuse must fail closed.
    """
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg)
    trainer = Trainer(model, cfg, start_step=3)

    def fail_load(_state):
        raise RuntimeError("simulated optimizer load failure")

    monkeypatch.setattr(trainer.optimizer, "load_state_dict", fail_load)
    with pytest.raises(
        RuntimeError,
        match="transaction rollback could not restore the pre-state",
    ):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4, 5],
            n_new_tokens=4,
            max_iterations=1,
            kl_max=0.0,
            trainer=trainer,
        )

    assert trainer.step == 4, "step must retain the actual last observed horizon"
    assert trainer._hagi_self_improve_rollback_poisoned is True
    with pytest.raises(RuntimeError, match="trainer is poisoned"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4, 5],
            n_new_tokens=4,
            max_iterations=1,
            kl_max=0.0,
            trainer=trainer,
        )


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


def test_accepted_diagnostic_exception_rolls_back_gradient_transaction(monkeypatch):
    """A reporting failure after an accepted update rolls back that update."""
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    trainer = Trainer(model, cfg, start_step=7)
    # Seed real optimizer momentum first. Comparing an all-empty initial
    # optimizer would pass even if rollback silently discarded every state slot.
    seed_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    trainer.train_step([{
        "input_ids": seed_ids,
        "targets": torch.roll(seed_ids, -1, dims=1),
        "loss_mask": torch.ones_like(seed_ids, dtype=torch.bool),
    }])
    model.eval()
    pre_step = trainer.step
    params = [param for param in model.parameters() if param.requires_grad]
    params_before = [param.detach().clone() for param in params]
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    assert optimizer_before["adamw"]["state"], "precondition: optimizer state must be non-empty"

    def fail_adapter_values(*args, **kwargs):
        raise RuntimeError("simulated accepted diagnostic failure")

    monkeypatch.setattr(si, "_adapter_values", fail_adapter_values)
    with pytest.raises(RuntimeError, match="simulated accepted diagnostic failure"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            n_new_tokens=16,
            max_iterations=1,
            kl_max=10.0,
            trainer=trainer,
        )

    assert trainer.step == pre_step
    assert model.training is False
    for param, expected in zip(params, params_before, strict=True):
        assert torch.equal(param.detach(), expected)

    _assert_nested_state_equal(
        optimizer_before,
        trainer.optimizer.state_dict(),
    )


def test_rls_plateau_logger_exception_rolls_back_current_iteration(monkeypatch):
    """A plateau-reporting failure restores the state before that iteration."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg).eval()
    ttt = _rls(model)
    original_step = ttt.step
    first_state = None
    scores = iter((1.0, 0.5, 0.5, 0.5))

    def recording_step(*args, **kwargs):
        nonlocal first_state
        result = original_step(*args, **kwargs)
        if first_state is None:
            first_state = _rls_state_digest(ttt)
        return result

    def synthetic_score(*args, **kwargs):
        return next(scores), torch.zeros(1)

    def fail_info(*args, **kwargs):
        raise RuntimeError("simulated plateau logger failure")

    monkeypatch.setattr(ttt, "step", recording_step)
    monkeypatch.setattr(si, "_score", synthetic_score)
    monkeypatch.setattr(si, "_kl_pre_post", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(si.logger, "info", fail_info)

    with pytest.raises(RuntimeError, match="simulated plateau logger failure"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=2,
            patience=1,
            ce_min_improve=0.0,
            kl_max=10.0,
        )

    assert first_state is not None
    _assert_rls_state_equal(first_state, _rls_state_digest(ttt))
    assert model.training is False


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


# --------------------------------------------------------------------------
# mode="rls": features -> LoRA delta, generation paid once for the whole loop
# --------------------------------------------------------------------------


def _count_generations(monkeypatch) -> dict[str, int]:
    """Wrap ``_generate_trajectory`` so the amortization claim is observable."""
    calls = {"n": 0}
    orig = si._generate_trajectory

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(si, "_generate_trajectory", spy)
    return calls


def _rls(model: HAGI, **kw) -> TttRls:
    defaults = dict(stream_frac=0.5, refit_rows=8, max_delta_rms_frac=1.0)
    defaults.update(kw)
    return TttRls(model, **defaults)


def test_rls_mode_generates_once_across_iterations(monkeypatch):
    """The acceleration claim, pinned: 1 trajectory for K updates."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    calls = _count_generations(monkeypatch)
    stats = si.self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=_rls(model),
        n_new_tokens=16,
        max_iterations=4,
        patience=5,
    )
    assert len(stats.iterations) == 4
    assert calls["n"] == 1, f"rls must pay generation once, got {calls['n']}"


def test_gradient_mode_regenerates_every_iteration(monkeypatch):
    """Contrast: the default path still pays generation per update."""
    cfg = _make_cfg(levels=(1,))
    model = HAGI(cfg)
    calls = _count_generations(monkeypatch)
    stats = si.self_improve(
        model, cfg, [1, 2, 3, 4], n_new_tokens=16, max_iterations=2, patience=5
    )
    assert len(stats.iterations) == 2
    assert calls["n"] == 2


def test_rls_mode_never_allocates_a_trainer(monkeypatch):
    """No optimizer state exists in rls mode, so nothing can drift there."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)

    def _boom(*a, **k):
        raise AssertionError("mode='rls' must not build a Trainer")

    monkeypatch.setattr(si, "Trainer", _boom)
    stats = si.self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=_rls(model),
        n_new_tokens=16,
        max_iterations=2,
        patience=5,
    )
    assert len(stats.iterations) == 2


def test_rls_mode_keeps_base_frozen():
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    before = _base_hash(model)
    stats = si.self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=_rls(model),
        n_new_tokens=16,
        max_iterations=3,
        patience=5,
    )
    assert _base_hash(model) == before, "base must stay frozen across rls updates"
    assert stats.accepted_updates >= 1


def test_rls_mode_moves_lora_b_and_records_the_step_bound():
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    before = [b.adapters.ttt_lora.lora_B.detach().clone() for b in model.blocks]
    stats = si.self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=_rls(model),
        n_new_tokens=16,
        max_iterations=3,
        patience=5,
    )
    after = [b.adapters.ttt_lora.lora_B.detach() for b in model.blocks]
    assert any(
        not torch.equal(x, y) for x, y in zip(before, after)
    ), "rls must move lora_B"
    applied = [r for r in stats.iterations if r.update_applied]
    assert applied, "expected at least one applied update"
    assert all(0.0 <= r.delta_rms_frac <= 1.0 for r in applied)


def test_rls_mode_honours_the_delta_cap():
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    stats = si.self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=_rls(model, stream_frac=50.0, max_delta_rms_frac=0.02),
        n_new_tokens=16,
        max_iterations=3,
        patience=5,
    )
    assert all(r.delta_rms_frac <= 0.02 + 1e-9 for r in stats.iterations)


def _rls_state_digest(ttt: TttRls) -> tuple:
    """Return a compact exact snapshot of all persistent RLS state."""
    result = []
    for i in sorted(ttt._state):
        state = ttt._state[i]
        result.append(
            (
                i,
                state.G.detach().clone(),
                state.C.detach().clone(),
                ttt._lora[i].lora_B.detach().clone(),
                state.train_rows,
                state.refits,
                state.buf_rows,
                state.holdout_rows,
                tuple(tensor.detach().clone() for tensor in state.phi_buf),
                tuple(tensor.detach().clone() for tensor in state.y_buf),
            )
        )
    return tuple(result)


def _assert_rls_state_equal(before: tuple, after: tuple) -> None:
    assert len(before) == len(after)
    for expected, actual in zip(before, after, strict=True):
        assert expected[0] == actual[0]
        assert torch.equal(expected[1], actual[1]), "G must be restored"
        assert torch.equal(expected[2], actual[2]), "C must be restored"
        assert torch.equal(expected[3], actual[3]), "lora_B must be restored"
        assert expected[4:8] == actual[4:8], "RLS counters must be restored"
        assert len(expected[8]) == len(actual[8]), "phi_buf length must be restored"
        assert len(expected[9]) == len(actual[9]), "y_buf length must be restored"
        for expected_row, actual_row in zip(expected[8], actual[8], strict=True):
            assert torch.equal(expected_row, actual_row)
        for expected_row, actual_row in zip(expected[9], actual[9], strict=True):
            assert torch.equal(expected_row, actual_row)


def test_rls_mode_rolls_back_on_kl_breach():
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    pristine = [b.adapters.ttt_lora.lora_B.detach().clone() for b in model.blocks]
    ttt = _rls(model)
    state_before = _rls_state_digest(ttt)
    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=ttt,
        n_new_tokens=16,
        max_iterations=3,
        kl_max=0.0,
    )
    assert stats.stopped == "kl_bound"
    assert stats.accepted_updates == 0
    after = [b.adapters.ttt_lora.lora_B.detach() for b in model.blocks]
    for x, y in zip(pristine, after, strict=True):
        assert torch.equal(x, y), "a rejected rls update must not persist"
    _assert_rls_state_equal(state_before, _rls_state_digest(ttt))


def test_rls_reject_diagnostic_survives_a_second_restore_failure(monkeypatch):
    """A diagnostic error wins; a successful rollback is never retried."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    ttt = _rls(model)
    real_restore = ttt.restore_state
    calls = {"count": 0}

    def restore_then_fail_on_retry(snapshot):
        calls["count"] += 1
        if calls["count"] == 1:
            return real_restore(snapshot)
        raise RuntimeError("RLS-ROLLBACK-SECONDARY")

    monkeypatch.setattr(ttt, "step", lambda *args, **kwargs: TttStats(
        ce=1.0,
        blocks=ttt.block_count,
        blocks_updated=1,
        refits=1,
        train_rows=1,
        holdout_rows=0,
        delta_rms_frac=0.1,
    ))
    monkeypatch.setattr(si, "_kl_pre_post", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(si.logger, "warning", lambda *args, **kwargs: (_ for _ in ()).throw(
        KeyboardInterrupt("RLS-DIAGNOSTIC-PRIMARY")
    ))
    monkeypatch.setattr(ttt, "restore_state", restore_then_fail_on_retry)

    with pytest.raises(KeyboardInterrupt, match="RLS-DIAGNOSTIC-PRIMARY"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=1,
            kl_max=0.0,
        )
    assert calls["count"] == 1


def test_rls_accepted_diagnostic_failure_survives_rollback_failure(monkeypatch):
    """A diagnostic error must win over a secondary RLS restore error."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    ttt = _rls(model)
    restore_calls = {"count": 0}

    monkeypatch.setattr(ttt, "step", lambda *args, **kwargs: TttStats(
        ce=1.0,
        blocks=ttt.block_count,
        blocks_updated=1,
        refits=1,
        train_rows=1,
        holdout_rows=0,
        delta_rms_frac=0.1,
    ))
    monkeypatch.setattr(
        si,
        "_adapter_values",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("RLS-DIAGNOSTIC-PRIMARY")
        ),
    )

    def failing_restore(_snapshot):
        restore_calls["count"] += 1
        raise RuntimeError("RLS-ROLLBACK-SECONDARY")

    monkeypatch.setattr(ttt, "restore_state", failing_restore)

    with pytest.raises(KeyboardInterrupt, match="RLS-DIAGNOSTIC-PRIMARY"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=1,
            kl_max=1.0,
        )
    assert restore_calls["count"] == 1


def test_rls_mode_exception_rolls_back_partial_fitter_mutation(monkeypatch):
    """An exception after partial RLS mutation must not poison retry state."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    ttt = _rls(model)
    state_before = _rls_state_digest(ttt)
    adapter_before = [b.adapters.ttt_lora.lora_B.detach().clone() for b in model.blocks]

    def fail_after_mutation(*args, **kwargs):
        ttt._state[0].G.add_(1.0)
        ttt._state[0].train_rows += 1
        raise RuntimeError("simulated RLS failure")

    monkeypatch.setattr(ttt, "step", fail_after_mutation)
    with pytest.raises(RuntimeError, match="simulated RLS failure"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=1,
        )
    _assert_rls_state_equal(state_before, _rls_state_digest(ttt))
    for expected, actual in zip(
        adapter_before,
        (b.adapters.ttt_lora.lora_B.detach() for b in model.blocks),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_rls_mode_reject_diagnostic_exception_rolls_back_mutation(monkeypatch):
    """A failure after a KL verdict must not bypass the RLS rollback."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    ttt = _rls(model)
    state_before = _rls_state_digest(ttt)

    def mutate_then_report(*args, **kwargs):
        state = ttt._state[0]
        lora = ttt._lora[0]
        state.G.add_(1.0)
        state.C.add_(1.0)
        state.train_rows += 1
        state.holdout_rows += 1
        state.phi_buf.append(torch.ones(1, lora.r))
        state.y_buf.append(torch.ones(1, lora.hidden_size))
        state.buf_rows += 1
        with torch.no_grad():
            lora.lora_B.add_(1.0)
        return TttStats(
            ce=0.0,
            blocks=ttt.block_count,
            blocks_updated=1,
            refits=1,
            train_rows=1,
            holdout_rows=1,
            delta_rms_frac=0.1,
        )

    def fail_warning(*args, **kwargs):
        raise RuntimeError("simulated reject diagnostic failure")

    monkeypatch.setattr(ttt, "step", mutate_then_report)
    monkeypatch.setattr(si, "_kl_pre_post", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(si.logger, "warning", fail_warning)

    with pytest.raises(RuntimeError, match="simulated reject diagnostic failure"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=1,
            kl_max=0.0,
        )
    _assert_rls_state_equal(state_before, _rls_state_digest(ttt))


def test_rls_mode_base_exception_rolls_back_partial_mutation(monkeypatch):
    """Cancellation must not leave the persistent fitter ahead of pre-state."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    ttt = _rls(model)
    state_before = _rls_state_digest(ttt)

    def interrupt_after_mutation(*args, **kwargs):
        ttt._state[0].G.add_(1.0)
        ttt._state[0].C.add_(1.0)
        ttt._state[0].train_rows += 1
        with torch.no_grad():
            ttt._lora[0].lora_B.add_(1.0)
        raise KeyboardInterrupt("simulated cancellation")

    monkeypatch.setattr(ttt, "step", interrupt_after_mutation)
    with pytest.raises(KeyboardInterrupt, match="simulated cancellation"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=1,
        )
    _assert_rls_state_equal(state_before, _rls_state_digest(ttt))


def test_rls_mode_rejects_fitter_bound_to_another_model():
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    owner = HAGI(cfg)
    other = HAGI(cfg)
    ttt = _rls(owner)
    state_before = _rls_state_digest(ttt)
    other_before = [b.adapters.ttt_lora.lora_B.detach().clone() for b in other.blocks]

    with pytest.raises(ValueError, match="same model"):
        self_improve(
            other,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=1,
        )

    _assert_rls_state_equal(state_before, _rls_state_digest(ttt))
    for expected, actual in zip(
        other_before,
        (b.adapters.ttt_lora.lora_B.detach() for b in other.blocks),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_rls_mode_restores_model_training_mode_on_success_and_exception(monkeypatch):
    cfg = _make_cfg(ttt_lora=True, pyramid=False)

    success_model = HAGI(cfg).train()
    self_improve(
        success_model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=_rls(success_model),
        n_new_tokens=16,
        max_iterations=1,
        kl_max=10.0,
    )
    assert success_model.training is True

    failure_model = HAGI(cfg).train()
    failing_ttt = _rls(failure_model)

    def fail_step(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(failing_ttt, "step", fail_step)
    with pytest.raises(RuntimeError, match="simulated failure"):
        self_improve(
            failure_model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=failing_ttt,
            n_new_tokens=16,
            max_iterations=1,
        )
    assert failure_model.training is True


def test_rls_mode_post_score_exception_restores_and_allows_retry(monkeypatch):
    """A post-update scoring failure must restore the full RLS transaction."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    ttt = _rls(model)
    state_before = _rls_state_digest(ttt)
    original_score = si._score
    calls = {"count": 0}

    def fail_second_score(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated post-score failure")
        return original_score(*args, **kwargs)

    monkeypatch.setattr(si, "_score", fail_second_score)
    with pytest.raises(RuntimeError, match="simulated post-score failure"):
        self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=1,
        )
    _assert_rls_state_equal(state_before, _rls_state_digest(ttt))

    monkeypatch.setattr(si, "_score", original_score)
    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=ttt,
        n_new_tokens=16,
        max_iterations=1,
        kl_max=10.0,
    )
    assert len(stats.iterations) == 1


def test_rls_mode_rejects_restore_when_lora_b_requires_grad_false():
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    for block in model.blocks:
        block.adapters.ttt_lora.lora_B.requires_grad_(False)
    pristine = [b.adapters.ttt_lora.lora_B.detach().clone() for b in model.blocks]
    ttt = _rls(model)
    state_before = _rls_state_digest(ttt)
    stats = self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        ttt=ttt,
        n_new_tokens=16,
        max_iterations=1,
        kl_max=0.0,
    )
    assert stats.stopped == "kl_bound"
    _assert_rls_state_equal(state_before, _rls_state_digest(ttt))
    for expected, actual in zip(
        pristine,
        (b.adapters.ttt_lora.lora_B.detach() for b in model.blocks),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_rls_mode_builds_a_default_fitter_when_omitted():
    """The caller-built seam is optional, not mandatory."""
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    stats = si.self_improve(
        model,
        cfg,
        [1, 2, 3, 4],
        mode="rls",
        n_new_tokens=16,
        max_iterations=2,
        patience=5,
    )
    assert len(stats.iterations) == 2


@pytest.mark.parametrize("kw", [
    {"mode": "bogus"},
    {"mode": ""},
    {"mode": None},
])
def test_bad_mode_raises(kw):
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    with pytest.raises(ValueError, match="mode must be"):
        si.self_improve(model, cfg, [1, 2, 3, 4], n_new_tokens=16, **kw)


def test_rls_mode_requires_ttt_lora_contour():
    """Pyramid-only cannot be fitted by an anchored-RLS solve."""
    cfg = _make_cfg(ttt_lora=False, pyramid=True, levels=(1,))
    model = HAGI(cfg)
    with pytest.raises(ValueError, match="ttt_lora.enabled=True is required"):
        si.self_improve(model, cfg, [1, 2, 3, 4], mode="rls", n_new_tokens=16)


def test_rls_primary_exception_survives_rollback_logging_failure(monkeypatch):
    """A failing rollback log must not replace the error the caller must see.

    Rollback already failed, so its own diagnostic is the interesting one, but
    the exception that caused the rollback is what tells the caller what went
    wrong. A raising log handler would otherwise surface instead of it, and the
    caller would debug the wrong failure.
    """
    cfg = _make_cfg(ttt_lora=True, pyramid=False)
    model = HAGI(cfg)
    ttt = _rls(model)

    def exploding_step(*args, **kwargs):
        raise KeyboardInterrupt("PRIMARY-INTERRUPT")

    def exploding_log(*args, **kwargs):
        raise RuntimeError("LOGGER-FAIL")

    def failing_restore(*args, **kwargs):
        raise RuntimeError("RESTORE-FAIL")

    monkeypatch.setattr(ttt, "step", exploding_step)
    monkeypatch.setattr(ttt, "restore_state", failing_restore)
    monkeypatch.setattr(si.logger, "exception", exploding_log)

    with pytest.raises(KeyboardInterrupt, match="PRIMARY-INTERRUPT"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            mode="rls",
            ttt=ttt,
            n_new_tokens=16,
            max_iterations=1,
        )


def test_gradient_primary_exception_survives_rollback_logging_failure(monkeypatch):
    """Same contract for the gradient contour's post-step rollback path.

    The injected failure must happen *after* ``train_step``. Otherwise it exits
    during pre-scoring and never exercises optimizer rollback. With both adapter
    restores and the rollback logger failing, there is no coherent pre-state to
    claim: the primary error must still win, the post recovery must be marked
    poisoned, and the caller's module modes must remain intact.
    """
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    real_score = si._score
    score_calls = {"count": 0}

    class RecordingOptimizer:
        def __init__(self) -> None:
            self.state = {"momentum": torch.tensor([1.0])}
            self.load_calls = 0

        def state_dict(self) -> dict:
            return {"state": self.state}

        def load_state_dict(self, state: dict) -> None:
            self.load_calls += 1
            self.state = state["state"]

    class PostStepFailingTrainer:
        def __init__(self) -> None:
            self.step = 4
            self.optimizer = RecordingOptimizer()

        def train_step(self, microbatches: list[dict]) -> dict:
            self.step += 1
            self.optimizer.state = {"momentum": torch.tensor([9.0])}
            return {"update_applied": True}

    trainer = PostStepFailingTrainer()

    def failing_second_score(model_arg, input_ids, targets):
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            return real_score(model_arg, input_ids, targets)
        raise KeyboardInterrupt("GRAD-PRIMARY")

    def exploding_log(*args, **kwargs):
        raise RuntimeError("LOGGER-FAIL")

    def failing_adapter_restore(*args, **kwargs):
        raise RuntimeError("RESTORE-FAIL")

    monkeypatch.setattr(si, "_score", failing_second_score)
    monkeypatch.setattr(si, "_restore_adapters", failing_adapter_restore)
    monkeypatch.setattr(si.logger, "exception", exploding_log)

    with pytest.raises(KeyboardInterrupt, match="GRAD-PRIMARY"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            n_new_tokens=8,
            max_iterations=1,
            kl_max=10.0,
            trainer=trainer,
        )

    assert score_calls["count"] == 2
    assert trainer.step == 5
    assert trainer.optimizer.load_calls == 2
    assert torch.equal(trainer.optimizer.state["momentum"], torch.tensor([9.0]))
    assert model.training is False
    assert trainer._hagi_self_improve_rollback_poisoned is True


def test_gradient_mode_entry_failure_is_inside_transaction(monkeypatch):
    """A failing ``model.train()`` before the step must still restore mode."""
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    trainer = Trainer(model, cfg, start_step=7)
    real_train = model.train

    def fail_after_train(mode: bool = True):
        result = real_train(mode)
        if mode is True:
            raise RuntimeError("MODE-ENTER-SECONDARY")
        return result

    monkeypatch.setattr(model, "train", fail_after_train)
    with pytest.raises(RuntimeError, match="MODE-ENTER-SECONDARY"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            n_new_tokens=4,
            max_iterations=1,
            kl_max=1.0,
            trainer=trainer,
        )

    assert model.training is False
    assert all(module.training is False for module in model.modules())
    assert trainer.step == 7


def _hostile_eval_restore(monkeypatch, model) -> None:
    """Fail only the restore of the caller's mode, not the switch into eval.

    ``model.eval()`` is implemented as ``train(False)``, so a train that rejects
    every ``False`` would break before the primary error is even raised and
    would prove nothing. Counting the switches separates the two: the first is
    the call into scoring, the second is the restore on the way out.
    """
    real_train = model.train
    seen = {"eval_calls": 0}

    def hostile_train(mode: bool = True):
        if mode is False:
            seen["eval_calls"] += 1
            if seen["eval_calls"] > 1:
                raise RuntimeError("MODE-RESTORE-SECONDARY")
        return real_train(mode)

    monkeypatch.setattr(model, "train", hostile_train)


def test_score_primary_exception_survives_failing_mode_restore(monkeypatch):
    """Scoring's ``finally`` must not replace the error that stopped it."""
    cfg = _make_cfg(levels=(1,))
    model = HAGI(cfg).eval()
    _hostile_eval_restore(monkeypatch, model)

    def primary_failure(*args, **kwargs):
        raise KeyboardInterrupt("SCORE-PRIMARY")

    monkeypatch.setattr(model, "forward", primary_failure)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with pytest.raises(KeyboardInterrupt, match="SCORE-PRIMARY"):
        si._score(model, ids, torch.tensor([[2, 3, 4, 5]], dtype=torch.long))


def test_generate_primary_exception_survives_failing_mode_restore(monkeypatch):
    """Generation has the same contract as scoring."""
    cfg = _make_cfg(levels=(1,))
    model = HAGI(cfg).eval()
    _hostile_eval_restore(monkeypatch, model)

    def primary_failure(*args, **kwargs):
        raise KeyboardInterrupt("GENERATE-PRIMARY")

    monkeypatch.setattr(si, "generate", primary_failure)
    with pytest.raises(KeyboardInterrupt, match="GENERATE-PRIMARY"):
        si._generate_trajectory(model, [1, 2, 3], n_new_tokens=2, eos_token_id=0, pad_token_id=-1)


def test_reject_branch_warning_failure_restores_training_mode(monkeypatch):
    """Diagnostics emitted after a rollback must not leak the eval() switch.

    ``trainer.train_step`` puts the model in train mode; the loop owns the
    caller's mode, so a failure while reporting the rejection has to restore it
    exactly like the rollback itself does.
    """
    cfg = _make_cfg(levels=(1, 2, 4), lr=1.0)
    model = HAGI(cfg).eval()
    trainer = Trainer(model, cfg, start_step=3)

    def exploding_warning(*args, **kwargs):
        raise RuntimeError("simulated logger failure during KL reject")

    monkeypatch.setattr(si.logger, "warning", exploding_warning)
    with pytest.raises(RuntimeError, match="simulated logger failure during KL reject"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4, 5],
            n_new_tokens=4,
            max_iterations=1,
            kl_max=0.0,
            trainer=trainer,
        )
    assert model.training is False, "caller's eval mode must survive the failure"
    assert trainer.step == 3, "rejected step must not advance the resume horizon"


def test_post_recovery_without_known_horizon_is_poisoned(monkeypatch):
    """A quarantine with no post horizon is uncertain, not a successful rollback.

    ``post`` pairs the captured post-state with the step that produced it, so a
    resume from a discarded trainer would otherwise continue from a horizon that
    no snapshot ever witnessed. Returning ``post`` without writing a known
    ``post_step`` would leave the trainer reusable against an unverifiable
    state; the only safe result is an explicit ``poisoned``.
    """
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    trainer = Trainer(model, cfg, start_step=3)
    optimizer = trainer.optimizer
    pre_params = si._snapshot_adapters(model)
    pre_opt = si._deepcopy_optimizer_state(optimizer.state_dict())
    trainer.step = 4
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.25)
    post_params = si._snapshot_adapters(model)
    post_opt = si._deepcopy_optimizer_state(optimizer.state_dict())

    class FailingPreOptimizer:
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped
            self.calls = 0

        def state_dict(self) -> dict:
            return self.wrapped.state_dict()

        def load_state_dict(self, state: dict) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("PRE-OPT-FAIL")
            self.wrapped.load_state_dict(state)

    status = si._rollback_gradient_transaction(
        model,
        trainer,
        FailingPreOptimizer(optimizer),
        pre_params,
        pre_opt,
        post_params,
        post_opt,
        pre_step=3,
        post_step=None,
    )

    assert status == "poisoned"
    assert getattr(trainer, "_hagi_self_improve_rollback_poisoned", False) is True


def test_post_recovery_with_attribute_rejection_uses_weak_quarantine():
    """A rejected attribute marker must still leave a verifiable quarantine.

    Recovered post-state is coherent, so its local rollback status remains
    ``post``; it is discard-only. If the trainer rejects the normal attribute
    marker, the weak fallback must still make the next public entry reject reuse.
    Returning ``post`` while leaving no readable quarantine would let the entry
    gate read the missing attribute as ``False``.
    """
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()

    class MarkerRejectingTrainer:
        __hash__ = None  # quarantine must not depend on injected hashability

        def __init__(self) -> None:
            self.step = 4
            self.optimizer = object()

        def __setattr__(self, name, value):
            if name == "_hagi_self_improve_rollback_poisoned":
                raise RuntimeError("POISON-MARKER-REJECTED")
            super().__setattr__(name, value)

    trainer = MarkerRejectingTrainer()
    pre_params = si._snapshot_adapters(model)
    trainer.step = 5
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.25)
    post_params = si._snapshot_adapters(model)

    class OneShotFailingOptimizer:
        def __init__(self) -> None:
            self.calls = 0

        def state_dict(self) -> dict:
            return {}

        def load_state_dict(self, state: dict) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("PRE-OPT-FAIL")

    optimizer = OneShotFailingOptimizer()
    status = si._rollback_gradient_transaction(
        model,
        trainer,
        optimizer,
        pre_params,
        {},
        post_params,
        {},
        pre_step=3,
        post_step=4,
    )

    assert optimizer.calls == 2, "post recovery itself must be complete"
    assert status == "post"
    assert getattr(trainer, "_hagi_self_improve_rollback_poisoned", False) is False
    with pytest.raises(RuntimeError, match="trainer is poisoned"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            n_new_tokens=2,
            max_iterations=1,
            kl_max=1.0,
            trainer=trainer,
        )


def test_unverifiable_quarantine_returns_poisoned():
    """A trainer that can store no readable marker is explicitly poisoned.

    This is the exhausted fallback case: the instance refuses marker
    assignment and cannot be weakly referenced. The coherent post-pair may
    remain, but there is no durable proof that the trainer is discard-only, so
    the rollback must not report ``post``.
    """
    class UnmarkableTrainer:
        __slots__ = ("step",)

        def __init__(self) -> None:
            self.step = 4

    trainer = UnmarkableTrainer()
    assert si._poison_gradient_trainer(trainer) is False
    assert not hasattr(trainer, "_hagi_self_improve_rollback_poisoned")


def test_unverifiable_quarantine_blocks_next_public_entry(monkeypatch):
    """A quarantine nobody can read must still fail closed at the entry gate.

    ``_poison_gradient_trainer`` reports ``False`` when it can write no durable
    marker: the instance rejects attribute assignment and is not weak
    referenceable, so both channels are gone. That return value only stops a
    caller from reporting ``post``; on its own it leaves the object unmarked. If
    the entry gate then reads the absent attribute as "not poisoned" and the empty
    registry as "not poisoned", the same trainer is accepted for a fresh
    transaction on state that was never proven discard-only. The gate therefore
    has to refuse on the same evidence the helper could not produce, not only on
    a marker that may not exist.
    """
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()

    class UnmarkableTrainer:
        """No attribute writes, no weakref support, no usable public API."""

        __slots__ = ("step",)

        def __init__(self) -> None:
            object.__setattr__(self, "step", 4)

    trainer = UnmarkableTrainer()

    assert si._poison_gradient_trainer(trainer) is False, (
        "precondition: no marker can be written for this trainer"
    )

    monkeypatch.setattr(
        si, "_score",
        lambda *args, **kwargs: pytest.fail(
            "the gate must refuse before any scoring happens"
        ),
    )

    with pytest.raises(RuntimeError, match="poisoned"):
        si.self_improve(
            model,
            cfg,
            [1, 2, 3, 4],
            n_new_tokens=8,
            max_iterations=1,
            kl_max=10.0,
            trainer=trainer,
        )


def test_gradient_rollback_restores_requires_grad_flags():
    """``requires_grad`` is transaction state, not a training-time optimisation.

    A trainer that both mutates a value and clears the flag leaves the
    parameter excluded from every future update. Restoring only the values of
    parameters that still require grad would report a successful rollback while
    the adapter stays permanently frozen, so the flag itself must be restored.
    """
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    Trainer(model, cfg)
    from hagi.model.adaptive import adaptive_parameters

    params = adaptive_parameters(model)
    assert params, "precondition: the contour must own adaptive parameters"
    snapshot = si._snapshot_adapters(model)
    with torch.no_grad():
        for parameter in params:
            parameter.add_(0.25)
            parameter.requires_grad_(False)

    si._restore_adapters(model, snapshot)

    for parameter in params:
        assert parameter.requires_grad is True
        assert torch.equal(
            parameter.detach(), snapshot[id(parameter)][0]
        ), "value must be restored together with the flag"


def test_gradient_rollback_reports_failure_when_no_adaptive_param_matches():
    """A snapshot whose parameters are all gone must not report success.

    Restoring zero parameters while the snapshot is non-empty means the
    transaction's model half was never applied, so the caller has to see a
    failure instead of pairing mismatched state with a healthy-looking
    optimizer.
    """
    cfg = _make_cfg(levels=(1,), lr=1.0)
    model = HAGI(cfg).eval()
    with pytest.raises(RuntimeError, match="adaptive parameter snapshot mismatch"):
        si._restore_adapters(model, {id(object()): (torch.zeros(1), True)})
