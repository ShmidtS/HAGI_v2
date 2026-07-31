"""MoE: bias controller convergence, gradient hygiene, dispatch correctness.

The V28 failure this replaces was not a bug in a formula — it was an auxiliary
loss losing an argument with the LM gradient for 50k steps while the log showed
``moe_lb`` pinned at its ceiling. So the tests here assert the two properties
that make a controller different from a loss: the balancing bias must never
touch the combining weights (no gradient competition), and it must actually
correct a deliberately skewed load within a bounded number of steps.
"""

from __future__ import annotations

import math

import pytest
import torch

from hagi.config import MoEConfig
from hagi.model.moe import MoE
from tests.conftest import assert_finite


def make_moe(experts=4, top_k=2, shared=1, gamma=0.01, hidden=32, inter=64) -> MoE:
    cfg = MoEConfig(
        enabled=True,
        num_experts=experts,
        top_k=top_k,
        n_shared=shared,
        bias_update_rate=gamma,
        z_loss_weight=1e-3,
    )
    return MoE(hidden, inter, cfg, use_ternary=False)


class TestShapes:
    def test_output_shape_and_finiteness(self):
        moe = make_moe()
        out = moe(torch.randn(2, 5, 32))
        assert out.shape == (2, 5, 32)
        assert_finite(out, "moe output")

    def test_rejects_degenerate_geometry(self):
        with pytest.raises(ValueError):
            make_moe(experts=1)
        with pytest.raises(ValueError):
            make_moe(experts=4, top_k=5)

    def test_no_shared_experts_is_allowed(self):
        out = make_moe(shared=0)(torch.randn(1, 4, 32))
        assert out.shape == (1, 4, 32)

    def test_top_k_equal_to_experts_is_dense_routing(self):
        out = make_moe(experts=4, top_k=4)(torch.randn(1, 4, 32))
        assert_finite(out, "dense-routed moe output")


class TestBiasIsolation:
    def test_bias_does_not_enter_the_combining_weights(self):
        """A large bias may change *which* experts run, never *how much* they count.

        If the bias leaked into the weights it would be a gradient on the expert
        outputs, which is precisely the competition with CE that V28 lost.
        """
        moe = make_moe(experts=4, top_k=4, shared=0).eval()
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            base = moe(x)
            # With top_k == num_experts the selected set cannot change, so any
            # output difference must have come through the weights.
            moe.expert_bias.copy_(torch.tensor([5.0, -3.0, 1.0, -3.0]))
            biased = moe(x)
        assert float((base - biased).abs().max()) < 1e-6

    def test_bias_does_change_selection(self):
        moe = make_moe(experts=4, top_k=1, shared=0).eval()
        x = torch.randn(1, 8, 32)
        with torch.no_grad():
            base = moe(x)
            moe.expert_bias.copy_(torch.tensor([50.0, 0.0, 0.0, 0.0]))
            forced = moe(x)
        assert float((base - forced).abs().max()) > 1e-6

    def test_controller_is_outside_autograd(self):
        moe = make_moe()
        out = moe(torch.randn(1, 4, 32))
        out.sum().backward()
        assert moe.expert_bias.grad is None
        assert not moe.expert_bias.requires_grad


class TestController:
    def test_corrects_a_skewed_router(self):
        """The controller must rescue a router that started collapsed.

        ``load_ema`` is initialized to perfect balance, so a fair-router test
        measures nothing. Instead one expert's router row is amplified until
        top-1 routing sends every token to it, which is the V28 failure state,
        and the controller has to pull the load back out.
        """
        moe = make_moe(experts=4, top_k=1, shared=0, gamma=0.05)
        with torch.no_grad():
            moe.router.weight[0].mul_(8.0).add_(1.0)
            moe.load_ema.zero_()
            moe.load_ema[0] = 1.0
        moe.train()
        x = torch.randn(1, 64, 32)

        start = moe.load_stats()
        for _ in range(400):
            with torch.no_grad():
                moe(x)
            moe.commit_bias_update()
        end = moe.load_stats()

        assert start["entropy_ratio"] < 0.05, "the test did not start from collapse"
        assert end["entropy_ratio"] > 0.5, (
            f"controller failed to rebalance: entropy_ratio {start['entropy_ratio']:.3f} "
            f"-> {end['entropy_ratio']:.3f}"
        )
        assert end["max_load"] < start["max_load"]

    def test_bias_moves_against_the_overloaded_expert(self):
        moe = make_moe(experts=4, top_k=1, shared=0, gamma=0.1)
        with torch.no_grad():
            moe.router.weight[0].mul_(8.0).add_(1.0)
        moe.train()
        with torch.no_grad():
            moe(torch.randn(1, 32, 32))
        moe.commit_bias_update()
        assert float(moe.expert_bias[0]) == min(float(b) for b in moe.expert_bias)

    def test_gauge_is_anchored(self):
        """Only bias differences matter, so the mean must not drift."""
        moe = make_moe(gamma=0.1)
        moe.train()
        for _ in range(50):
            with torch.no_grad():
                moe(torch.randn(1, 32, 32))
            moe.commit_bias_update()
        assert abs(float(moe.expert_bias.mean())) < 1e-5

    def test_zero_rate_freezes_the_bias(self):
        moe = make_moe(gamma=0.0)
        moe.train()
        with torch.no_grad():
            moe(torch.randn(1, 16, 32))
        moe.commit_bias_update()
        assert float(moe.expert_bias.abs().max()) == 0.0

    def test_commit_without_forward_is_a_noop(self):
        moe = make_moe()
        moe.commit_bias_update()
        assert float(moe.expert_bias.abs().max()) == 0.0

    def test_eval_forward_records_nothing(self):
        moe = make_moe().eval()
        with torch.no_grad():
            moe(torch.randn(1, 8, 32))
        assert moe._pending_load is None
        assert moe.last_router_z_loss is None


class TestLoadStats:
    def test_perfect_balance_is_one(self):
        moe = make_moe(experts=8)
        with torch.no_grad():
            moe.load_ema.fill_(1.0 / 8)
        assert moe.load_stats()["entropy_ratio"] == pytest.approx(1.0, abs=1e-5)

    def test_total_collapse_is_zero(self):
        moe = make_moe(experts=8)
        with torch.no_grad():
            moe.load_ema.zero_()
            moe.load_ema[0] = 1.0
        stats = moe.load_stats()
        assert stats["entropy_ratio"] < 1e-4
        assert stats["max_load"] == pytest.approx(1.0)

    def test_keys_present(self):
        assert set(make_moe().load_stats()) == {
            "entropy_ratio",
            "max_load",
            "min_load",
            "bias_span",
        }


class TestRouterZLoss:
    def test_recorded_while_training(self):
        moe = make_moe()
        moe.train()
        moe(torch.randn(1, 8, 32))
        assert moe.last_router_z_loss is not None
        assert float(moe.last_router_z_loss.detach()) >= 0.0

    def test_penalizes_large_logits(self):
        moe = make_moe()
        moe.train()
        x = torch.randn(1, 8, 32)
        moe(x)
        small = float(moe.last_router_z_loss.detach())
        with torch.no_grad():
            moe.router.weight.mul_(20.0)
        moe(x)
        assert float(moe.last_router_z_loss.detach()) > small


class TestDispatch:
    def test_matches_a_dense_reference(self):
        """Sorted segmented dispatch must equal the naive per-token loop."""
        moe = make_moe(experts=4, top_k=2, shared=0).eval()
        flat = torch.randn(11, 32)
        idx = torch.tensor([[0, 1], [2, 3], [1, 1], [0, 0], [3, 2], [1, 0],
                            [2, 2], [3, 0], [0, 3], [1, 2], [2, 0]])
        weights = torch.rand(11, 2)

        with torch.no_grad():
            got = moe._dispatch(flat, idx, weights)
            expected = torch.zeros_like(got)
            for token in range(11):
                for slot in range(2):
                    e = int(idx[token, slot])
                    expected[token] += moe.experts[e](flat[token : token + 1])[0] * weights[token, slot]
        assert float((got - expected).abs().max()) < 1e-5

    def test_empty_input(self):
        moe = make_moe()
        out = moe._dispatch(
            torch.zeros(0, 32), torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, 2)
        )
        assert out.shape == (0, 32)

    def test_unselected_expert_contributes_nothing(self):
        moe = make_moe(experts=4, top_k=1, shared=0).eval()
        flat = torch.randn(5, 32)
        idx = torch.zeros(5, 1, dtype=torch.long)
        with torch.no_grad():
            got = moe._dispatch(flat, idx, torch.ones(5, 1))
            direct = moe.experts[0](flat)
        assert float((got - direct).abs().max()) < 1e-5


def test_entropy_ceiling_matches_log_e():
    """Sanity on the normalizer itself: the ceiling is ln(E), not something else."""
    moe = make_moe(experts=5)
    with torch.no_grad():
        moe.load_ema.fill_(0.2)
    load = moe.load_ema
    entropy = float(-(load * load.log()).sum())
    assert entropy == pytest.approx(math.log(5), abs=1e-6)
