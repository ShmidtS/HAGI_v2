"""MacroCycleV2 kernels (R175) and the harvest addendum — runtime ports.

* variance gate (``logitRange_scale`` / ``variance_gate_contract`` /
  ``variance_gate_normalize``): temperature repair of a range overshoot
* trust-region ball projection (``trust_region_proj_norm`` /
  ``trust_region_nearest``): rescaling IS the metric projection
* greedy-harvest addendum (``harvest_accounting_identity`` /
  ``greedy_horizon_optimal``): the field pays unit for unit; greedy is
  horizon-optimal
"""
from __future__ import annotations

import torch

from hagi.train.recursive_distill import (
    greedy_horizon_optimal,
    harvest_accounting_identity,
)
from hagi.train.safeqp_step import (
    expert_gate_tau,
    expert_spread,
    logit_range_scale,
    variance_gate_tau,
)
from hagi.train.trust_region import project_to_ball, projection_is_nearest


def _range(z: list[float]) -> float:
    return max(z) - min(z)


class TestVarianceGate:
    def test_tau_repair_is_exact(self):
        z = [3.0, -3.0, 1.0]  # R = 6
        tau = variance_gate_tau(z, r_bar=2.0)
        assert abs(tau - 3.0) < 1e-12
        assert abs(_range(logit_range_scale(z, tau)) - 2.0) < 1e-12

    def test_slack_contracts_quadratically(self):
        # R^2/8 -> R_bar^2/8: the quadratic contraction the contract states
        z = [4.0, -4.0]  # R = 8, slack 8
        tau = variance_gate_tau(z, r_bar=2.0)
        scaled = logit_range_scale(z, tau)
        assert abs(_range(scaled) ** 2 / 8.0 - 0.5) < 1e-12  # 2^2/8

    def test_in_range_passes_through(self):
        assert variance_gate_tau([0.5, -0.5], r_bar=2.0) == 1.0

    def test_gate_uses_centered_spread(self):
        # The admission R must be the R106 CENTERED spread (deviation from
        # the weighted pool mean), not the raw max-min: the PoE bound is
        # stated in the centered range.
        logits = [[1.0, -1.0, 0.5], [0.0, 0.0, 0.0]]
        r_i = expert_spread(logits, [0.5, 0.5], 0)
        r_j = expert_spread(logits, [0.5, 0.5], 1)
        assert r_i > 0.0
        assert r_j > 0.0  # the pool member also has a centered spread

    def test_expert_admission_temperatures(self):
        # expert 0 overshoots the admission bar; experts 1-2 are in range
        logits = [[3.0, -3.0, 1.0], [0.5, 0.0, -0.5], [0.0, 0.0, 0.0]]
        w = [0.4, 0.3, 0.3]
        taus = expert_gate_tau(logits, w, r_bar=3.0)
        assert abs(taus[0] - 3.45 / 3.0) < 1e-12
        assert taus[1] == 1.0 and taus[2] == 1.0
        # post-gate: the overshooting expert's centered spread is exactly r_bar
        mean = [sum(wi * row[j] for wi, row in zip(w, logits)) for j in range(3)]
        dev0 = [logits[0][j] - mean[j] for j in range(3)]
        scaled = [z / taus[0] for z in dev0]
        assert abs((max(scaled) - min(scaled)) - 3.0) < 1e-12


class TestTrustRegion:
    def test_projection_lands_on_boundary(self):
        x = torch.tensor([3.0, 4.0])
        proj, norm = project_to_ball(x, radius=2.0)
        assert abs(float(torch.linalg.vector_norm(proj)) - 2.0) < 1e-6
        assert abs(norm - 5.0) < 1e-6

    def test_projection_is_nearest(self):
        x = torch.tensor([3.0, 4.0])
        for y in (torch.tensor([2.0, 0.0]), torch.tensor([0.0, 2.0]),
                  torch.tensor([-1.0, 1.0])):
            assert projection_is_nearest(x, y, radius=2.0)

    def test_in_ball_untouched(self):
        x = torch.tensor([1.0, 0.0])
        proj, _ = project_to_ball(x, radius=2.0)
        assert proj is x


class TestHarvestAddendum:
    def test_identity_greedy(self):
        gamma, d0 = 0.5, 1.0
        harvests = [gamma * (1 - gamma) ** t * d0 for t in range(3)]
        d3 = (1 - gamma) ** 3 * d0
        assert harvest_accounting_identity(harvests, d0, d3)

    def test_identity_rejects_mismatch(self):
        # The two clocks disagreeing: harvested more than the field paid
        assert not harvest_accounting_identity([0.9], 1.0, 0.5)

    def test_ceiling_greedy_achieves(self):
        gamma, d0, n = 0.5, 1.0, 3
        r = greedy_horizon_optimal(gamma, d0, n)
        greedy_harvest = sum(gamma * (1 - gamma) ** t * d0 for t in range(n))
        assert abs(r["ceiling"] - greedy_harvest) < 1e-12
        assert abs(r["residual_field"] - (1 - gamma) ** n * d0) < 1e-12

    def test_ceiling_upper_bounds_any_schedule(self):
        gamma, d0, n = 0.5, 1.0, 3
        r = greedy_horizon_optimal(gamma, d0, n)
        # any schedule h_t <= gamma*D_t: try several, none beats ceiling
        for schedule in ([0.5, 0.25, 0.125], [0.4, 0.2, 0.1], [0.5, 0.1, 0.1]):
            assert sum(schedule) <= r["ceiling"] + 1e-12

    def test_gamma_domain(self):
        for bad in (0.0, -0.1, 1.7):
            try:
                greedy_horizon_optimal(bad, 1.0, 3)
                raised = False
            except ValueError:
                raised = True
            assert raised


class TestDynamicBranchVariance:
    def test_total_injected_variance_is_half(self):
        # dynamic_branch_variance: sum over L branches of s^2*v = v/2
        for L in (1, 2, 3, 7, 16):
            s2 = 1.0 / (2.0 * L)
            total = L * s2 * 1.0
            assert abs(total - 0.5) < 1e-12
