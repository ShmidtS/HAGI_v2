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
from hagi.train.self_improve import _restore_adapters, _snapshot_adapters
from hagi.train.ttt import TttRls, TttStats
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
    ])
    def test_bad_hyperparams_raise(self, kw):
        with pytest.raises(ValueError):
            _fitter(_model(_lora_cfg()), **kw)


class TestFrozenBaseInvariant:
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
    def test_stats_report_zero_generations(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        stats = _fitter(model).step(ids, tgt)
        assert isinstance(stats, TttStats)
        assert stats.generations == 0

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
            assert not block.mixer._forward_hooks


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
    def test_snapshot_restore_roundtrip(self):
        cfg = _lora_cfg()
        model = _model(cfg)
        ids, tgt = _window(cfg)
        fitter = _fitter(model)
        fitter.step(ids, tgt)
        snap = _snapshot_adapters(model)
        before = _lora_b(model)
        for _ in range(5):
            fitter.step(ids, tgt)
        assert any(
            not torch.equal(x, y) for x, y in zip(before, _lora_b(model))
        ), "the later steps should have moved lora_B"
        _restore_adapters(model, snap)
        for x, y in zip(before, _lora_b(model)):
            assert torch.equal(x, y)


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
