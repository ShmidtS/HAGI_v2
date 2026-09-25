"""Tests for the feature→delta TTT contour (:mod:`hagi.train.ttt`).

The contour's contract is that it learns *without* generating: one
teacher-forced pass supplies the exact per-block local error, and an anchored
ridge solve fits ``lora_B``. These tests pin the invariants that make that
claim checkable — frozen base, honest holdout, bounded step, transactional
rollback, one forward per step — plus the behavioral assertion that the fit
actually lowers loss and survives a context reorder.
"""

from __future__ import annotations

import pytest
import torch

from hagi.config import Config, validate_config
from hagi.model.model import HAGI
from hagi.train.ttt import (
    _SNAPSHOT_HARD_MAX_BYTES,
    TttRls,
    _restore_training_mode,
    _snapshot_training_modes,
)
from tests.conftest import tiny_config


def _lora_cfg(
    *,
    ttt: bool = True,
    pyramid: bool = False,
    rank: int = 8,
    **overrides,
) -> Config:
    cfg = tiny_config()
    cfg.model.adapters.enabled = True
    cfg.model.adapters.pyramid.enabled = pyramid
    cfg.model.adapters.ttt_lora.enabled = ttt
    cfg.model.adapters.ttt_lora.rank = rank
    for key, value in overrides.items():
        target: object = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        assert hasattr(target, parts[-1]), f"unknown override {key!r}"
        setattr(target, parts[-1], value)
    validate_config(cfg)
    return cfg


def _model(cfg: Config) -> HAGI:
    torch.manual_seed(0)
    return HAGI(cfg)


def _window(cfg: Config, cols: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    ids = torch.randint(1, cfg.model.vocab_size, (1, cols))
    return ids, torch.roll(ids, -1, dims=1).clone()


def _ce(model: HAGI, ids: torch.Tensor, tgt: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(model(ids, tgt).ce)


def _base_hashes(model: HAGI) -> dict[str, float]:
    return {
        n: float(p.detach().abs().sum())
        for n, p in model.named_parameters()
        if "adapters" not in n
    }


def _lora_b(model: HAGI) -> list[torch.Tensor]:
    return [b.adapters.ttt_lora.lora_B.detach().clone() for b in model.blocks]


def _fitter(model: HAGI, **kw) -> TttRls:
    defaults = dict(
        stream_frac=0.1, prior=1.0, refit_rows=8, max_delta_rms_frac=1.0
    )
    defaults.update(kw)
    return TttRls(model, **defaults)


class TestAdapterRequirement:
    def test_no_adapters_raises(self):
        cfg = tiny_config()
        with pytest.raises(ValueError, match="no ttt_lora adapter"):
            TttRls(_model(cfg))

    def test_pyramid_only_raises(self):
        cfg = _lora_cfg(ttt=False, pyramid=True)
        cfg.model.adapters.pyramid.levels = (1,)
        validate_config(cfg)
        with pytest.raises(ValueError, match="no ttt_lora adapter"):
            TttRls(_model(cfg))


class TestHyperparameterValidation:
    @pytest.mark.parametrize("kw", [
        {"stream_frac": 0.0},
        {"stream_frac": -1.0},
        {"stream_frac": float("nan")},
        {"lam": 0.0},
        {"lam": 1.5},
        {"lam": float("inf")},
        {"refit_rows": 0},
        {"max_delta_rms_frac": 0.0},
        {"max_delta_rms_frac": float("nan")},
        {"rows_max": 0},
        {"prior": float("nan")},
        {"prior": 0.0},
        {"prior": -1.0},
        {"reg": 0.0},
        {"reg": -1.0},
        {"snapshot_max_bytes": 0},
        {"snapshot_max_bytes": -1},
        {"snapshot_max_bytes": 1.0},
    ])
    def test_bad_hyperparams_raise(self, kw):
        with pytest.raises(ValueError):
            _fitter(_model(_lora_cfg()), **kw)


class TestFrozenBaseInvariant:
    def test_rls_step_rejects_row_count_mismatch(self):
        """The reference checks this; without it a mismatch broadcasts or fits
        the wrong rows silently."""
        cfg = _lora_cfg()
        model = _model(cfg)
        fitter = _fitter(model)
        lora = fitter._lora[0]
        phi = torch.randn(12, lora.r)
        y = torch.randn(11, lora.hidden_size)
        with pytest.raises(ValueError, match="row count mismatch"):
            fitter.rls_step(0, phi, y, holdout=False)

    def test_base_params_unchanged(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        before = _base_hashes(model)
        fitter = _fitter(model)
        for _ in range(5):
            fitter.step(ids, tgt)
        assert _base_hashes(model) == before

    def test_only_lora_b_written(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        snap = {n: p.detach().clone() for n, p in model.named_parameters()}
        _fitter(model).step(ids, tgt)
        changed = [
            n for n, p in model.named_parameters()
            if not torch.equal(p.detach(), snap[n])
        ]
        assert changed, "expected the fit to move lora_B"
        assert all(n.endswith("lora_B") for n in changed), changed

    def test_lora_a_stays_frozen(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        before = [b.adapters.ttt_lora.lora_A.detach().clone() for b in model.blocks]
        fitter = _fitter(model)
        for _ in range(3):
            fitter.step(ids, tgt)
        for block, a in zip(model.blocks, before):
            assert torch.equal(block.adapters.ttt_lora.lora_A.detach(), a)


class TestNoGenerationInvariant:
    def test_exactly_one_forward_per_step(self):
        """A generation-free step must not re-enter the encoder.

        Counted on the encoder rather than on blocks: ``loop_depth`` repeats
        blocks, so a block counter would not be a stable invariant.
        """
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        calls = {"n": 0}

        def _count(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
            calls["n"] += 1

        handle = model.encoder.register_forward_pre_hook(_count)
        try:
            _fitter(model).step(ids, tgt)
        finally:
            handle.remove()
        assert calls["n"] == 1, f"expected 1 forward, got {calls['n']}"

    def test_hooks_are_removed(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        _fitter(model).step(ids, tgt)
        for block in model.blocks:
            assert not block._forward_hooks, "a leaked hook would fire forever"
            # The mixer carries a *pre*-hook (ttt.py registers
            # ``block.mixer.register_forward_pre_hook``), so asserting on
            # ``_forward_hooks`` here could never fail -- it was a vacuous
            # assertion, not a test of the cleanup.
            assert not block.mixer._forward_pre_hooks, \
                "a leaked pre-hook would fire forever"


class TestHoldoutExclusion:
    def test_rls_step_holdout_never_enters_normal_equations(self):
        """Direct arithmetic proof, independent of the model.

        Holding out the last 1/5 of ``n`` rows must leave exactly the same
        ``G``/``C``/``B`` as training on those first rows with holdout off.
        """
        cfg = _lora_cfg()
        m1 = _model(cfg)
        m2 = _model(cfg)
        f1 = _fitter(m1)
        f2 = _fitter(m2)
        lora1 = m1.blocks[0].adapters.ttt_lora
        torch.manual_seed(3)
        n = 30
        phi = torch.randn(n, lora1.r)
        y = torch.randn(n, lora1.hidden_size)

        split = n - n // 5
        f1.rls_step(0, phi, y, holdout=True)
        f2.rls_step(0, phi[:split], y[:split], holdout=False)

        s1, s2 = f1._state[0], f2._state[0]
        assert torch.equal(s1.G, s2.G), "holdout rows entered G"
        assert torch.equal(s1.C, s2.C), "holdout rows entered C"
        assert torch.equal(
            lora1.lora_B.detach(), m2.blocks[0].adapters.ttt_lora.lora_B.detach()
        )
        assert s1.holdout_rows == n // 5
        assert s2.holdout_rows == 0

    def test_holdout_row_counts(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg, cols=32)
        stats = _fitter(model, refit_rows=10**9).step(ids, tgt)
        n_blocks = len(model.blocks)
        assert stats.holdout_rows == n_blocks * (32 // 5)
        assert stats.train_rows == n_blocks * (32 - 32 // 5)

    def test_holdout_disabled_uses_every_row(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg, cols=32)
        stats = _fitter(model, refit_rows=10**9).step(ids, tgt, holdout=False)
        assert stats.holdout_rows == 0
        assert stats.train_rows == len(model.blocks) * 32


class TestRefitGate:
    def test_gate_refuses_a_fit_that_does_not_improve(self):
        """Zero features carry no information, so C is zero and the candidate
        equals the current ``B``; the gate must refuse rather than churn."""
        cfg = _lora_cfg()
        model = _model(cfg)
        fitter = _fitter(model)
        lora = model.blocks[0].adapters.ttt_lora
        phi = torch.zeros(16, lora.r)
        torch.manual_seed(5)
        y = torch.randn(16, lora.hidden_size)
        updated, frac = fitter.rls_step(0, phi, y)
        assert updated is False
        assert frac == 0.0
        assert float(lora.lora_B.detach().abs().sum()) == 0.0

    def test_gate_accepts_an_exactly_fittable_target(self):
        """If the target is generated from the subspace itself, the fit must be
        accepted and drive the residual toward zero."""
        cfg = _lora_cfg()
        model = _model(cfg)
        fitter = _fitter(model, prior=1e-6)
        lora = model.blocks[0].adapters.ttt_lora
        torch.manual_seed(5)
        phi = torch.randn(64, lora.r)
        B_true = torch.randn(lora.hidden_size, lora.r)
        y = phi @ B_true.T
        updated, _ = fitter.rls_step(0, phi, y, holdout=False)
        assert updated is True
        resid = float(
            (phi @ lora.lora_B.detach().float().T - y).pow(2).sum()
            / y.pow(2).sum()
        )
        assert resid < 1e-3, f"exact-linear target not recovered: {resid}"


class TestStepBound:
    def test_delta_frac_never_exceeds_cap(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        cap = 0.02
        fitter = _fitter(model, stream_frac=50.0, max_delta_rms_frac=cap)
        for _ in range(6):
            stats = fitter.step(ids, tgt)
            assert stats.delta_rms_frac <= cap + 1e-9

    def test_cap_is_binding_not_vacuous(self):
        cfg = _lora_cfg()
        ids, tgt = _window(cfg)
        tight = _fitter(_model(cfg), stream_frac=50.0, max_delta_rms_frac=0.01)
        loose = _fitter(_model(cfg), stream_frac=50.0, max_delta_rms_frac=1.0)
        assert loose.step(ids, tgt).delta_rms_frac > tight.step(ids, tgt).delta_rms_frac


class TestLearning:
    def test_ce_decreases(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        before = _ce(model, ids, tgt)
        fitter = _fitter(model)
        for _ in range(20):
            fitter.step(ids, tgt)
        after = _ce(model, ids, tgt)
        assert after < before, f"CE did not fall: {before} -> {after}"

    def test_transfers_to_reordered_context(self):
        """Not exact memorisation: the correction survives a context reorder.

        Honest scope note: the two windows share a token multiset, so this
        rules out position-specific rote learning. It is not a claim of
        generalisation to unseen tokens.
        """
        cfg = _lora_cfg()
        model = _model(cfg)
        torch.manual_seed(11)
        p = torch.randint(1, cfg.model.vocab_size, (8,))
        q = torch.randint(1, cfg.model.vocab_size, (8,))
        a_in = torch.cat([q, p, q, p]).unsqueeze(0)
        b_in = torch.cat([p, q, p, q]).unsqueeze(0)
        a_tg = torch.roll(a_in, -1, dims=1).clone()
        b_tg = torch.roll(b_in, -1, dims=1).clone()
        b_before = _ce(model, b_in, b_tg)
        fitter = _fitter(model, max_delta_rms_frac=0.05)
        for _ in range(20):
            fitter.step(a_in, a_tg)
        assert _ce(model, b_in, b_tg) < b_before


class TestLossMask:
    def test_masked_positions_excluded_from_rows(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg, cols=32)
        mask = torch.ones(1, 32, dtype=torch.long)
        mask[:, 20:] = 0
        stats = _fitter(model, refit_rows=10**9).step(ids, tgt, loss_mask=mask)
        assert stats.train_rows + stats.holdout_rows == len(model.blocks) * 20

    def test_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="loss_mask rows"):
            TttRls._flatten(torch.zeros(4, 2), torch.ones(8, dtype=torch.bool))


class TestTransactionalRollback:
    @pytest.mark.parametrize("was_training", [False, True])
    def test_mode_restore_fallback_recovers_full_module_tree(
        self, monkeypatch, was_training
    ):
        """A failed public mode switch must not leave submodules inconsistent."""
        cfg = _lora_cfg()
        model = _model(cfg)
        model.train(was_training)
        training_modes = _snapshot_training_modes(model)
        model.train(not was_training)

        def failing_train(mode: bool = True):
            raise RuntimeError("MODE-RESTORE-SECONDARY")

        monkeypatch.setattr(model, "train", failing_train)
        _restore_training_mode(model, training_modes)

        assert model.training is was_training
        assert all(module.training is was_training for module in model.modules())

    def test_harvest_entry_eval_failure_restores_mode_and_removes_hooks(
        self, monkeypatch
    ):
        """Cleanup must cover the mode switch that happens after hook setup."""
        cfg = _lora_cfg()
        model = _model(cfg).train()
        fitter = _fitter(model)
        ids, tgt = _window(cfg)

        def hook_state():
            return tuple(
                (
                    set(block._forward_pre_hooks),
                    set(block._forward_hooks),
                    set(block.mixer._forward_pre_hooks),
                )
                for block in model.blocks
            )

        before_hooks = hook_state()
        real_train = model.train
        calls = {"n": 0}

        def fail_after_switch(mode: bool = True):
            real_train(mode)
            calls["n"] += 1
            raise RuntimeError(f"MODE-{calls['n']}")

        monkeypatch.setattr(model, "train", fail_after_switch)
        with pytest.raises(RuntimeError, match="MODE-1"):
            fitter._harvest(ids, tgt, None)

        assert calls["n"] == 2  # failing entry eval + guarded restore attempt
        assert model.training is True
        assert all(module.training is True for module in model.modules())
        assert hook_state() == before_hooks

    def test_harvest_primary_survives_failing_restore_and_cleans_hooks(
        self, monkeypatch
    ):
        """The primary harvest error wins and the mode is still guaranteed."""
        cfg = _lora_cfg()
        model = _model(cfg).train()
        fitter = _fitter(model)
        ids, tgt = _window(cfg)
        before_hooks = tuple(
            (
                set(block._forward_pre_hooks),
                set(block._forward_hooks),
                set(block.mixer._forward_pre_hooks),
            )
            for block in model.blocks
        )
        real_train = model.train

        def fail_before_restore(mode: bool = True):
            if mode is True:
                raise RuntimeError("MODE-RESTORE-SECONDARY")
            return real_train(mode)

        def primary_failure(*args, **kwargs):
            raise KeyboardInterrupt("HARVEST-PRIMARY")

        monkeypatch.setattr(model, "train", fail_before_restore)
        monkeypatch.setattr(model, "forward", primary_failure)
        with pytest.raises(KeyboardInterrupt, match="HARVEST-PRIMARY"):
            fitter._harvest(ids, tgt, None)

        assert model.training is True
        assert all(module.training is True for module in model.modules())
        after_hooks = tuple(
            (
                set(block._forward_pre_hooks),
                set(block._forward_hooks),
                set(block.mixer._forward_pre_hooks),
            )
            for block in model.blocks
        )
        assert after_hooks == before_hooks

    def test_harvest_preserves_mixed_module_modes(self):
        """A supported per-submodule mode setup must survive harvesting exactly.

        ``nn.Module.train`` is allowed to leave a child in eval mode while the
        root is in train mode. Restoring only the root flag would silently
        normalize that deliberate configuration, so cleanup must preserve each
        module's own ``training`` flag.
        """
        cfg = _lora_cfg()
        model = _model(cfg).train()
        model.blocks[0].eval()
        before = tuple((id(module), module.training) for module in model.modules())
        fitter = _fitter(model)
        ids, tgt = _window(cfg)

        fitter.step(ids, tgt)

        after = tuple((id(module), module.training) for module in model.modules())
        assert after == before

    def test_harvest_hook_cleanup_survives_failing_handle_remove(
        self, monkeypatch
    ):
        """Cleanup must not depend on ``RemovableHandle.remove()`` succeeding.

        The handle keeps no reference back to the module, so a raising removal
        would leave every later hook attached and let that secondary failure
        replace the error that stopped the step.
        """
        cfg = _lora_cfg()
        model = _model(cfg).train()
        fitter = _fitter(model)
        ids, tgt = _window(cfg)
        other_hook = object()
        block = model.blocks[0]
        block._forward_pre_hooks[9000] = other_hook
        before = tuple(
            (
                set(block._forward_pre_hooks),
                set(block._forward_hooks),
                set(block.mixer._forward_pre_hooks),
            )
            for block in model.blocks
        )

        def exploding_remove(self):
            raise RuntimeError("REMOVE-SECONDARY")

        def primary_failure(*args, **kwargs):
            raise KeyboardInterrupt("HARVEST-PRIMARY")

        monkeypatch.setattr(
            torch.utils.hooks.RemovableHandle,
            "remove",
            exploding_remove,
            raising=True,
        )
        monkeypatch.setattr(model, "forward", primary_failure)
        with pytest.raises(KeyboardInterrupt, match="HARVEST-PRIMARY"):
            fitter._harvest(ids, tgt, None)

        after = tuple(
            (
                set(block._forward_pre_hooks),
                set(block._forward_hooks),
                set(block.mixer._forward_pre_hooks),
            )
            for block in model.blocks
        )
        assert after == before
        assert block._forward_pre_hooks[9000] is other_hook
        assert model.training is True
        assert all(module.training is True for module in model.modules())

    def test_harvest_registration_failure_cleans_already_attached_hooks(
        self, monkeypatch
    ):
        """A failure halfway through registration must not leave earlier hooks."""
        cfg = _lora_cfg()
        model = _model(cfg).train()
        fitter = _fitter(model)
        ids, tgt = _window(cfg)
        first_block = model.blocks[0]
        before = tuple(
            (set(b._forward_pre_hooks), set(b._forward_hooks), set(b.mixer._forward_pre_hooks))
            for b in model.blocks
        )

        def fail_registration(*args, **kwargs):
            raise RuntimeError("REGISTER-SECONDARY")

        monkeypatch.setattr(first_block, "register_forward_hook", fail_registration)
        with pytest.raises(RuntimeError, match="REGISTER-SECONDARY"):
            fitter._harvest(ids, tgt, None)

        after = tuple(
            (set(b._forward_pre_hooks), set(b._forward_hooks), set(b.mixer._forward_pre_hooks))
            for b in model.blocks
        )
        assert after == before
        assert model.training is True
        assert all(module.training is True for module in model.modules())

    def test_harvest_cleanup_uses_public_remove_on_success(self, monkeypatch):
        """Normal cleanup follows PyTorch's public handle contract."""
        cfg = _lora_cfg()
        model = _model(cfg)
        fitter = _fitter(model)
        ids, tgt = _window(cfg)
        real_remove = torch.utils.hooks.RemovableHandle.remove
        calls = {"n": 0}

        def counting_remove(self):
            calls["n"] += 1
            return real_remove(self)

        monkeypatch.setattr(
            torch.utils.hooks.RemovableHandle,
            "remove",
            counting_remove,
            raising=True,
        )
        fitter.step(ids, tgt)
        assert calls["n"] == 2 * len(model.blocks)

    def test_harvest_success_path_leaves_no_hooks(self):
        """A clean step must not accumulate hook registrations on the blocks."""
        cfg = _lora_cfg()
        model = _model(cfg)
        fitter = _fitter(model)
        ids, tgt = _window(cfg)
        for _ in range(3):
            fitter.step(ids, tgt)
        for block in model.blocks:
            assert not block._forward_pre_hooks
            assert not block._forward_hooks
            assert not block.mixer._forward_pre_hooks

    def test_snapshot_cap_cannot_be_raised_after_construction(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        fitter = _fitter(model, snapshot_max_bytes=1)
        with pytest.raises(AttributeError, match="snapshot_max_bytes"):
            fitter.snapshot_max_bytes = _SNAPSHOT_HARD_MAX_BYTES * 100
        with pytest.raises(MemoryError, match="rollback snapshot"):
            fitter.snapshot_state()

    def test_snapshot_memory_cap_fails_before_mutation(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        fitter = _fitter(model, snapshot_max_bytes=1)
        before_g = fitter._state[0].G.detach().clone()
        before_b = fitter._lora[0].lora_B.detach().clone()
        with pytest.raises(MemoryError, match="rollback snapshot"):
            fitter.snapshot_state()
        assert torch.equal(fitter._state[0].G, before_g)
        assert torch.equal(fitter._lora[0].lora_B, before_b)

    @pytest.mark.parametrize(
        "malformed", ["phi_width", "target_width", "phi_dtype", "row_count"]
    )
    def test_malformed_row_geometry_poisons_until_valid_restore(self, malformed):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        fitter = _fitter(model)
        fitter.step(ids, tgt)
        valid = fitter.snapshot_state()
        entry = valid[0]
        phi_rows = list(entry[8])
        y_rows = list(entry[9])
        assert phi_rows and y_rows

        if malformed == "phi_width":
            phi_rows[0] = torch.empty(
                (phi_rows[0].shape[0], phi_rows[0].shape[1] + 1),
                dtype=phi_rows[0].dtype,
                device=phi_rows[0].device,
            )
        elif malformed == "target_width":
            y_rows[0] = torch.empty(
                (y_rows[0].shape[0], y_rows[0].shape[1] + 1),
                dtype=y_rows[0].dtype,
                device=y_rows[0].device,
            )
        elif malformed == "phi_dtype":
            phi_rows[0] = phi_rows[0].to(dtype=torch.float64)
        else:
            y_rows[0] = y_rows[0][:0]

        invalid = list(valid)
        invalid[0] = (*entry[:8], tuple(phi_rows), tuple(y_rows), entry[10])
        with pytest.raises(ValueError, match="row geometry"):
            fitter.restore_state(tuple(invalid))
        with pytest.raises(RuntimeError, match="poisoned"):
            fitter.step(ids, tgt)
        fitter.restore_state(valid)
        assert fitter.step(ids, tgt).blocks == len(model.blocks)

    def test_snapshot_row_tamper_fails_closed_instead_of_restoring(self):
        """A snapshot that was mutated after the fact must not be restored.

        Row tensors are retained by reference, so a snapshot the caller edited
        in place is the same object the live buffer still holds. Restoring it
        would silently "roll back" to the tampered values and clear the poison
        flag, making the fitter look usable with values nobody verified.
        """
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        fitter = _fitter(model)
        fitter.step(ids, tgt)
        snapshot = fitter.snapshot_state()
        assert fitter._state[0].phi_buf
        snapshot[0][8][0].add_(1.0)
        with pytest.raises(ValueError, match="digest mismatch"):
            fitter.restore_state(snapshot)
        with pytest.raises(RuntimeError, match="poisoned"):
            fitter.step(ids, tgt)

    @pytest.mark.parametrize("position", [1, 2, 3])
    def test_snapshot_accumulator_tamper_fails_closed(self, position):
        """``G``, ``C`` and ``lora_B`` come back as clones.

        Cloning means an external edit cannot reach live state, but it also
        means the snapshot itself can be rewritten after the fact, and geometry
        checks cannot see it: a snapshot whose ``G`` was scaled looks perfectly
        well-formed. Restoring it would present unverified values as a rolled
        back fitter, so every mutable tensor has to be covered by the digest.
        """
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        fitter = _fitter(model)
        fitter.step(ids, tgt)
        snapshot = fitter.snapshot_state()
        snapshot[0][position].add_(1.0)
        with pytest.raises(ValueError, match="digest mismatch"):
            fitter.restore_state(snapshot)
        with pytest.raises(RuntimeError, match="poisoned"):
            fitter.step(ids, tgt)

    @pytest.mark.parametrize(
        "position,message",
        [
            (4, "digest mismatch"),
            (5, "digest mismatch"),
            (6, "digest mismatch"),
            (7, "digest mismatch"),
        ],
    )
    def test_snapshot_counter_tamper_fails_closed(self, position, message):
        """The four bookkeeping counters are part of the snapshot unit too.

        ``train_rows``/``refits``/``holdout_rows`` are plain ints, so a type and
        sign check cannot tell an honest counter from an edited one: a snapshot
        claiming rows that were never consumed restores a fitter whose online
        schedule no longer matches the data it holds. ``buf_rows`` already
        fails on its own cross-check, and is pinned here so both routes stay
        fail-closed.
        """
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        fitter = _fitter(model)
        fitter.step(ids, tgt)
        snapshot = fitter.snapshot_state()
        entry = snapshot[0]
        counter = entry[position]
        tampered = (*entry[:position], counter + 1, *entry[position + 1 :])
        invalid = (tampered, *snapshot[1:])
        with pytest.raises(ValueError, match=message):
            fitter.restore_state(invalid)
        with pytest.raises(RuntimeError, match="poisoned"):
            fitter.step(ids, tgt)

    def test_valid_snapshot_digest_does_not_create_false_negatives(self):
        """The digest must reject edited snapshots without rejecting real ones.

        An ordinary loop repeatedly restores the same snapshot while rows keep
        being recorded, so a digest that drifted on a legitimate state change
        would break the actual update path, not just an adversarial one.
        """
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        fitter = _fitter(model, rows_max=4, refit_rows=10**9)
        for _ in range(6):
            fitter.step(ids, tgt)
        snapshot = fitter.snapshot_state()
        for _ in range(4):
            fitter.step(ids, tgt)
            fitter.restore_state(snapshot)
            assert fitter._rollback_poisoned is False
        assert fitter.step(ids, tgt).blocks == len(model.blocks)

    def test_malformed_snapshot_poisons_until_valid_restore(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        fitter = _fitter(model)
        valid = fitter.snapshot_state()
        invalid = list(valid)
        invalid[0] = (
            valid[0][0],
            valid[0][1],
            valid[0][2],
            valid[0][3][:, :1],
            *valid[0][4:],
        )
        with pytest.raises(ValueError, match="lora_B mismatch"):
            fitter.restore_state(tuple(invalid))
        with pytest.raises(RuntimeError, match="poisoned"):
            fitter.step(ids, tgt)
        fitter.restore_state(valid)
        stats = fitter.step(ids, tgt)
        assert stats.blocks == len(model.blocks)

    def test_snapshot_restore_roundtrip(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        fitter = _fitter(model)
        fitter.step(ids, tgt)
        before = _lora_b(model)
        before_g = fitter._state[0].G.detach().clone()
        before_c = fitter._state[0].C.detach().clone()
        snapshot = fitter.snapshot_state()
        for _ in range(5):
            fitter.step(ids, tgt)
        assert any(
            not torch.equal(x, y) for x, y in zip(before, _lora_b(model))
        ), "the later steps should have moved lora_B"
        fitter.restore_state(snapshot)
        for x, y in zip(before, _lora_b(model)):
            assert torch.equal(x, y)
        assert torch.equal(fitter._state[0].G, before_g)
        assert torch.equal(fitter._state[0].C, before_c)


class TestLoopDepth:
    def test_repeated_blocks_pair_features_with_errors(self):
        """``loop_depth`` fires each block twice per forward.

        ``feats[i]`` and ``outs[i]`` must stay parallel, or feature rows would
        be paired with the wrong local error and the fit would be silently
        wrong rather than loudly broken.
        """
        cfg = _lora_cfg(**{"model.loop_depth": 2})
        model = _model(cfg)
        ids, tgt = _window(cfg, cols=32)
        stats = _fitter(model, refit_rows=10**9).step(ids, tgt)
        assert stats.train_rows + stats.holdout_rows == len(model.blocks) * 32 * 2
