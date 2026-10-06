"""R176 BitAlloc + R174c/d/e thermodynamic layer — runtime ports.

* bit allocation (``transfer_exact`` / ``imbalance_yields_gain`` /
  ``stable_factor_two`` / ``two_layer_equalize`` /
  ``distill_quant_composite``): bits want to live where the error is
  large; greedy transfer stabilizes factor-2 balanced
* entropy production (``epRate`` / detailed balance ⟺ zero EP):
  equilibrium is dissipation-free, a NESS dissipates strictly
* annealing by batch growth (``anneal_by_batch``): the geometric
  noise budget σ²·c/(B₀·(c−1)), horizon-independent
* stationary flatness (tail-sum identity / ``flatness_moment_bound``):
  ``P[X ≥ k] ≤ C/k²`` ⟹ ``E[X] ≤ 2C``
"""
from __future__ import annotations

import torch

from hagi.train.bit_alloc import (
    apply_transfer,
    distill_quant_composite,
    greedy_transfer,
    layer_error,
    total_error,
    transfer_delta,
    two_layer_equalize,
)
from hagi.train.thermo_layer import (
    anneal_by_batch,
    detailed_balance,
    edge_flow,
    ep_rate,
    flatness_moment_bound,
    tail_constant,
    tail_sum_identity,
)


class TestBitAlloc:
    def test_layer_error_halves(self):
        # layerError_antitone: one more bit halves the error
        assert abs(layer_error(3.0, 4) - 3.0 / 16) < 1e-12
        assert abs(layer_error(3.0, 5) - layer_error(3.0, 4) / 2) < 1e-12

    def test_transfer_identity_exact(self):
        # transfer_exact: moving a bit k -> j changes the total by
        # EXACTLY e_k - e_j/2
        c = [1.0, 4.0, 2.0]
        f = [4, 2, 3]
        j, k = 1, 0
        before = total_error(c, f)
        after = total_error(c, apply_transfer(f, j, k))
        assert abs((after - before) - transfer_delta(c, f, j, k)) < 1e-12

    def test_transfer_nonincreasing_iff_receiver_2x(self):
        # the exchange improves exactly when e_k <= e_j/2
        c = [1.0, 5.0]
        f = [4, 2]
        # e_0 = 1/16 = 0.0625, e_1 = 5/4 = 1.25: 2*e_0 = 0.125 < 1.25 -> improving
        assert transfer_delta(c, f, j=1, k=0) < 0
        # reverse: e_1/2 = 0.625 > e_0 -> worsens
        assert transfer_delta(c, f, j=0, k=1) > 0

    def test_greedy_reaches_factor2_balance(self):
        # imbalance_yields_gain + stable_factor_two: greedy transfer
        # stabilizes with every error within a factor 2
        c = [1.0, 4.0, 16.0]
        f = [8, 2, 1]
        g, rounds = greedy_transfer(c, f)
        assert sum(g) == sum(f)  # budget preserved
        errs = [layer_error(ci, bi) for ci, bi in zip(c, g)]
        for ej in errs:
            for ek in errs:
                if ek > 0:
                    assert ej <= 2.0 * ek + 1e-12

    def test_greedy_improves_total(self):
        c = [1.0, 8.0, 2.0]
        f = [7, 1, 2]
        g, _ = greedy_transfer(c, f)
        assert total_error(c, g) <= total_error(c, f) + 1e-12
        assert total_error(c, g) < total_error(c, f)

    def test_two_layer_balancing_dominates(self):
        # two_layer_equalize: e(x) + e(x+2d) >= 2*e(x+d)
        for x, d in ((2, 0), (2, 3), (0, 5), (1, 1)):
            gap = two_layer_equalize(4.0, x, d)
            assert gap >= -1e-12

    def test_composite_certificate(self):
        # distill_quant_composite: S_n + n*g <= E_0 + delta_n + totalError
        s_n, e0, delta_n, n, g = 1.0, 2.0, 0.1, 3, 0.2
        c, f = [1.0, 2.0], [4, 4]
        bound, slack = distill_quant_composite(s_n, e0, delta_n, n, g, c, f)
        assert slack >= 0  # certificate holds on this instance
        assert bound == e0 + delta_n + total_error(c, f)

    def test_domain_errors(self):
        for bad in (
            lambda: layer_error(1.0, -1),
            lambda: transfer_delta([1.0], [1], 0, 0),
            lambda: transfer_delta([1.0, 1.0], [1, 0], 0, 1),  # giver empty
            lambda: total_error([1.0], [1, 2]),
        ):
            try:
                bad()
                raised = False
            except ValueError:
                raised = True
            assert raised


class TestNessDissipation:
    def test_detailed_balance_zero_ep(self):
        # a symmetric chain: detailed balance holds, EP is exactly zero
        P = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
        pi = torch.tensor([1.0 / 3.0, 2.0 / 3.0])
        assert detailed_balance(P, pi)
        assert abs(ep_rate(P, pi)) < 1e-9

    def test_ness_dissipates_strictly(self):
        # a 3-state directed cycle (all flows positive): no detailed
        # balance on a 2-chain by symmetry -- irreversibility needs
        # at least 3 states
        P = torch.tensor([[0.05, 0.90, 0.05], [0.05, 0.05, 0.90],
                          [0.90, 0.05, 0.05]])
        pi = torch.full((3,), 1.0 / 3.0)
        assert not detailed_balance(P, pi)
        assert ep_rate(P, pi) > 1e-6

    def test_flows_sum_to_one(self):
        P = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
        pi = torch.tensor([6.0 / 13.0, 7.0 / 13.0])
        forward, backward = edge_flow(P, pi)
        # the input pi is float32 (6/13 not exactly representable);
        # the kernel computes in double ON that input
        assert abs(float(forward.sum()) - 1.0) < 1e-6
        assert abs(float(backward.sum()) - 1.0) < 1e-6

    def test_ep_requires_positive_flows(self):
        P = torch.zeros(2, 2)
        pi = torch.tensor([0.5, 0.5])
        try:
            ep_rate(P, pi)
            raised = False
        except ValueError:
            raised = True
        assert raised


class TestAnnealByBatch:
    def test_budget_horizon_independent(self):
        # the total over any horizon stays under the geometric budget
        for n in (1, 5, 20, 100):
            r = anneal_by_batch(sigma2=2.0, B0=32.0, c=1.5, n=n)
            assert r["total_noise"] <= r["budget"] + 1e-12
        r_inf = anneal_by_batch(sigma2=2.0, B0=32.0, c=1.5, n=10_000)
        assert r_inf["total_noise"] < r_inf["budget"]

    def test_budget_value_exact(self):
        r = anneal_by_batch(sigma2=2.0, B0=32.0, c=1.5, n=3)
        # B_t = 32 * 1.5^t: 2/32 + 2/48 + 2/72
        expected = 2.0 * (1 / 32 + 1 / 48 + 1 / 72)
        assert abs(r["total_noise"] - expected) < 1e-12
        assert abs(r["budget"] - 2.0 * 1.5 / (32 * 0.5)) < 1e-12
        assert abs(r["final_B"] - 32 * 1.5 ** 3) < 1e-9

    def test_growth_cools_monotone(self):
        # temperature 1/B_t decays geometrically
        seq = [anneal_by_batch(1.0, 16.0, 2.0, t + 1)["total_noise"]
               for t in range(5)]
        diffs = [seq[t + 1] - seq[t] for t in range(4)]
        assert all(d2 < d1 for d1, d2 in zip(diffs, diffs[1:]))

    def test_domain_errors(self):
        for bad in (
            lambda: anneal_by_batch(-1.0, 32.0, 1.5, 3),
            lambda: anneal_by_batch(1.0, 0.0, 1.5, 3),
            lambda: anneal_by_batch(1.0, 32.0, 1.0, 3),
            lambda: anneal_by_batch(1.0, 32.0, 1.5, -1),
        ):
            try:
                bad()
                raised = False
            except ValueError:
                raised = True
            assert raised


class TestStationaryFlatness:
    def test_tail_sum_identity_exact(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0])
        probs = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15])
        # float32 inputs: 0.1/0.25 etc. not exactly representable;
        # the identity holds to input precision
        direct = float((probs.double() * x.double()).sum())
        assert abs(tail_sum_identity(probs, x) - direct) < 1e-6

    def test_flatness_moment_bound_value(self):
        # P[X >= k] <= C/k^2  =>  E[X] <= 2C
        C = 0.75
        assert abs(flatness_moment_bound(None, C) - 1.5) < 1e-12

    def test_tail_constant_and_bound_consistent(self):
        # samples with a genuine k^-2 tail: the estimated C certifies E[X] <= 2C
        torch.manual_seed(0)
        # geometric-ish tail samples bounded by C/k^2 with C ~ 0.6
        samples = torch.tensor([1.0] * 50 + [2.0] * 14 + [3.0] * 5 + [5.0, 8.0])
        c_hat = tail_constant(samples)
        mean = float(samples.double().mean())
        assert mean <= 2.0 * c_hat + 1e-9

    def test_tail_constant_domain(self):
        for bad in (lambda: tail_constant(torch.tensor([])),
                    lambda: tail_constant(torch.tensor([1.0, -0.5]))):
            try:
                bad()
                raised = False
            except ValueError:
                raised = True
            assert raised
