"""Optimizers: Newton-Schulz behaviour, the partition, and the LR schedule.

The partition assertion is the load-bearing one. A parameter in no group trains
at rate zero for the whole run and nothing reports it; a parameter in both groups
gets a double update. Both are silent, so ``build_optimizer`` raises and the test
pins that contract.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from hagi.model.model import HAGI
from hagi.model.ternary import BitLinear
from hagi.train.optim import (
    HybridOptimizer,
    Muon,
    build_optimizer,
    learning_rate_at,
    newton_schulz,
    set_learning_rate,
)
from tests.conftest import assert_finite, tiny_config


class TestNewtonSchulz:
    def test_singular_values_approach_one(self):
        """The point of the iteration: every direction advances equally."""
        g = torch.randn(32, 48)
        out = newton_schulz(g, steps=5).float()
        s = torch.linalg.svdvals(out)
        assert float(s.mean()) == pytest.approx(1.0, abs=0.25)
        assert float(s.std()) < 0.3, f"singular values not equalized: std {float(s.std()):.3f}"

    def test_shape_preserved_both_orientations(self):
        assert newton_schulz(torch.randn(8, 32)).shape == (8, 32)
        assert newton_schulz(torch.randn(32, 8)).shape == (32, 8)

    def test_finite_on_a_tiny_gradient(self):
        assert_finite(newton_schulz(torch.randn(8, 8) * 1e-8), "orthogonalized update")

    def test_finite_on_a_rank_deficient_gradient(self):
        g = torch.zeros(8, 8)
        g[0, 0] = 1.0
        assert_finite(newton_schulz(g), "orthogonalized update")

    def test_dtype_preserved(self):
        assert newton_schulz(torch.randn(8, 8, dtype=torch.float64)).dtype == torch.float64


class TestMuon:
    def test_rejects_non_2d(self):
        p = nn.Parameter(torch.randn(8))
        p.grad = torch.randn(8)
        with pytest.raises(ValueError):
            Muon([p], lr=0.01).step()

    def test_updates_the_parameter(self):
        p = nn.Parameter(torch.randn(8, 8))
        before = p.detach().clone()
        p.grad = torch.randn(8, 8)
        Muon([p], lr=0.01).step()
        assert float((p.detach() - before).abs().max()) > 0

    def test_skips_parameters_without_gradient(self):
        p = nn.Parameter(torch.randn(4, 4))
        before = p.detach().clone()
        Muon([p], lr=0.1).step()
        assert torch.equal(p.detach(), before)

    def test_weight_decay_shrinks_the_master(self):
        p = nn.Parameter(torch.ones(8, 8))
        p.grad = torch.zeros(8, 8)
        Muon([p], lr=0.1, weight_decay=0.5).step()
        assert float(p.detach().abs().max()) < 1.0

    def test_wide_output_decays_harder(self):
        """Wide-output matrices see more gradient per row, so the decay is scaled."""
        square = nn.Parameter(torch.ones(16, 16))
        wide = nn.Parameter(torch.ones(64, 16))
        for p in (square, wide):
            p.grad = torch.zeros_like(p)
        Muon([square, wide], lr=0.1, weight_decay=0.5).step()
        assert float(wide.detach().abs().mean()) < float(square.detach().abs().mean())

    def test_momentum_offload_keeps_the_buffer_on_host(self):
        p = nn.Parameter(torch.randn(8, 8))
        p.grad = torch.randn(8, 8)
        opt = Muon([p], lr=0.01, momentum_offload=True)
        opt.step()
        assert opt.state[p]["momentum_buffer"].device.type == "cpu"

    def test_nesterov_and_heavy_ball_differ(self):
        def run(nesterov: bool) -> torch.Tensor:
            torch.manual_seed(0)
            p = nn.Parameter(torch.zeros(8, 8))
            opt = Muon([p], lr=0.05, nesterov=nesterov)
            g = torch.randn(8, 8)
            for _ in range(3):
                p.grad = g.clone()
                opt.step()
            return p.detach().clone()

        assert float((run(True) - run(False)).abs().max()) > 1e-6


class TestPartition:
    def test_every_trainable_parameter_is_assigned_exactly_once(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        seen: list[int] = []
        for group in opt.param_groups:
            seen.extend(id(p) for p in group["params"])
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        assert len(seen) == len(set(seen)), "a parameter landed in two groups"
        assert set(seen) == trainable, "a trainable parameter landed in no group"

    def test_muon_takes_the_channel_weights(self):
        cfg = tiny_config(**{"train.use_muon": True})
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        expected = {
            id(m.weight)
            for m in model.modules()
            if getattr(m, "is_channel_weight", False) and hasattr(m, "weight")
        }
        got = {id(p) for g in opt.muon.param_groups for p in g["params"]}
        assert got == expected

    def test_every_channel_weight_is_a_bitlinear_when_ternary(self):
        model = HAGI(tiny_config())
        marked = [m for m in model.modules() if getattr(m, "is_channel_weight", False)]
        assert marked and all(isinstance(m, BitLinear) for m in marked)

    def test_codebook_goes_to_adamw(self):
        cfg = tiny_config(**{"train.use_muon": True})
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
        assert id(model.encoder.weight) not in muon_ids

    def test_receiver_gain_goes_to_adamw(self):
        cfg = tiny_config(**{"train.use_muon": True})
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
        adam_ids = {id(p) for g in opt.adamw.param_groups for p in g["params"]}
        assert id(model.head.logit_scale) in adam_ids
        assert id(model.head.logit_scale) not in muon_ids

    def test_norm_gains_are_not_decayed(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        no_decay = next(g for g in opt.adamw.param_groups if g["weight_decay"] == 0.0)
        ids = {id(p) for p in no_decay["params"]}
        assert id(model.out_norm.weight) in ids

    def test_ternary_ablation_keeps_the_partition(self):
        """Disabling quantization must not move the body from Muon to AdamW.

        Muon's argument is about the geometry of a hidden-mixing matrix, not about
        its storage rate, so a rate ablation that also swapped the optimizer would
        be measuring two changes at once.
        """
        cfg = tiny_config(**{"model.ternary.enabled": False, "train.use_muon": True})
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
        assert id(model.blocks[0].attn.qkv_proj.weight) in muon_ids
        assert id(model.blocks[0].mixer.mixer.gate.weight) in muon_ids
        assert id(model.encoder.weight) not in muon_ids

    def test_partition_size_is_identical_across_the_ablation(self):
        ternary = build_optimizer(HAGI(tiny_config(**{"train.use_muon": True})), tiny_config(**{"train.use_muon": True}))
        plain_cfg = tiny_config(**{"model.ternary.enabled": False, "train.use_muon": True})
        plain = build_optimizer(HAGI(plain_cfg), plain_cfg)
        count = lambda opt: sum(len(g["params"]) for g in opt.muon.param_groups)  # noqa: E731
        assert count(ternary) == count(plain)


class TestHybridOptimizer:
    def test_step_moves_both_families(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        set_learning_rate(opt, 50, cfg)

        ids = torch.randint(0, cfg.model.vocab_size, (2, 8))
        model(ids, ids).loss.backward()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        moved = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])]
        assert len(moved) > 10

    def test_state_dict_round_trip(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        ids = torch.randint(0, cfg.model.vocab_size, (2, 8))
        model(ids, ids).loss.backward()
        set_learning_rate(opt, 50, cfg)
        opt.step()

        state = opt.state_dict()
        assert set(state) == {"muon", "adamw"}
        fresh = build_optimizer(HAGI(tiny_config()), cfg)
        fresh.load_state_dict(state)

    def test_zero_grad_clears_both(self):
        cfg = tiny_config()
        model = HAGI(cfg)
        opt = build_optimizer(model, cfg)
        ids = torch.randint(0, cfg.model.vocab_size, (2, 8))
        model(ids, ids).loss.backward()
        opt.zero_grad()
        assert all(p.grad is None for p in model.parameters())

    def test_param_groups_carry_the_muon_flag(self):
        cfg = tiny_config(**{"train.use_muon": True})
        opt = build_optimizer(HAGI(cfg), cfg)
        assert any(g.get("_muon") for g in opt.param_groups)
        assert any(not g.get("_muon") for g in opt.param_groups)

    def test_no_muon_flag_when_disabled(self):
        cfg = tiny_config()  # use_muon defaults False
        opt = build_optimizer(HAGI(cfg), cfg)
        assert opt.muon is None
        assert not any(g.get("_muon") for g in opt.param_groups)

    def test_not_an_optimizer_subclass(self):
        """Deliberate: the two carry incompatible state and would make LR ambiguous."""
        assert not issubclass(HybridOptimizer, torch.optim.Optimizer)


class TestSchedule:
    def test_warmup_is_linear_from_zero(self):
        cfg = tiny_config()
        cfg.train.max_steps = 1000
        cfg.train.schedule.warmup_steps = 100
        assert learning_rate_at(0, 1.0, cfg) == 0.0
        assert learning_rate_at(50, 1.0, cfg) == pytest.approx(0.5)

    def test_peak_is_at_the_end_of_warmup(self):
        cfg = tiny_config()
        cfg.train.max_steps = 1000
        cfg.train.schedule.warmup_steps = 100
        cfg.train.schedule.inverse_sqrt_stable = False
        assert learning_rate_at(100, 1.0, cfg) == pytest.approx(1.0)

    def test_inverse_sqrt_stable_decays(self):
        cfg = tiny_config()
        cfg.train.max_steps = 10000
        cfg.train.schedule.warmup_steps = 100
        cfg.train.schedule.inverse_sqrt_stable = True
        early = learning_rate_at(200, 1.0, cfg)
        late = learning_rate_at(4000, 1.0, cfg)
        assert late < early

    def test_never_below_min_ratio(self):
        cfg = tiny_config()
        cfg.train.max_steps = 1000
        cfg.train.schedule.warmup_steps = 10
        cfg.train.schedule.min_lr_ratio = 0.05
        for step in range(0, 1200, 25):
            assert learning_rate_at(step, 1.0, cfg) >= 0.0
        assert learning_rate_at(1000, 1.0, cfg) == pytest.approx(0.05)

    def test_monotone_in_the_decay_phase(self):
        cfg = tiny_config()
        cfg.train.max_steps = 1000
        cfg.train.schedule.warmup_steps = 10
        cfg.train.schedule.decay_fraction = 0.5
        values = [learning_rate_at(s, 1.0, cfg) for s in range(500, 1000, 25)]
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:], strict=False))

    def test_both_base_rates_are_applied(self):
        cfg = tiny_config(**{"train.use_muon": True})
        cfg.train.learning_rate = 3e-4
        cfg.train.muon.lr = 0.02
        cfg.train.adam.body_lr_scale = 8.0
        opt = build_optimizer(HAGI(cfg), cfg)
        adam_lr, muon_lr = set_learning_rate(opt, cfg.train.schedule.warmup_steps, cfg)
        assert adam_lr != muon_lr
        for group in opt.param_groups:
            if group.get("_muon"):
                assert group["lr"] == muon_lr
            elif group.get("_body"):
                assert group["lr"] == adam_lr * cfg.train.adam.body_lr_scale
            else:
                assert group["lr"] == adam_lr
