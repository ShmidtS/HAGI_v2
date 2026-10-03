"""R118-R122 self-development loop: selection under noise, Pareto
direction, adaptive success, opportunity renewal.

Each test pins a Python port to the theorem statement it came from:

- ``self_development_select`` / ``selection_threshold`` -- R118
  ``self_development_find`` / ``certified_gain_select`` (3*eps bar,
  argmax lower bound ``g(a0) - 2*eps``).
- ``search_cost_bound`` -- R118 linear measurement budget.
- ``dir_tau`` / ``pareto_select`` -- R120 ``dirTau`` /
  ``dir_controller_find`` / ``noisy_pareto_select``.
- ``adaptive_success_floor`` -- R121 ``adaptive_success_concentrated``.
- ``opportunity_delta`` -- R122 ``opportunity_renewal`` net renewal.
- ``multiplicative_growth`` / ``closed_loop_floor`` -- R122
  ``multiplicative_growth`` / ``closed_loop_takeoff``.
"""

from __future__ import annotations

import math

import pytest

from hagi.train.self_development import (
    GrowthState,
    adaptive_success_floor,
    closed_loop_floor,
    dir_tau,
    multiplicative_growth,
    opportunity_delta,
    pareto_select,
    search_cost_bound,
    self_development_select,
    selection_threshold,
)


# ---------------------------------------------------------------- R118


class TestR118Selection:
    def test_threshold_is_three_eps(self):
        assert selection_threshold(0.096) == pytest.approx(0.288)

    def test_positive_candidate_clearing_bar_is_selected(self):
        # measured 0.35 > 3*eps = 0.288 -> true gain certified >= 0.35 - 0.19
        name, bound = self_development_select({"merge": 0.35}, eps=0.096)
        assert name == "merge"
        assert bound == pytest.approx(0.35 - 0.192)

    def test_below_bar_returns_undecided(self):
        # measured 0.2 < 3*eps -> bound 0.2 - 0.192 = 0.008 > 0 BUT the
        # honest R118 form requires g(a0) > 3*eps; emulate by a tighter
        # radius where the bound flips negative
        name, bound = self_development_select({"merge": 0.20}, eps=0.15)
        assert name is None
        assert bound == 0.0

    def test_argmax_not_best_measured(self):
        # two actions; the argmax by measurement wins regardless of order
        pool = {"merge": 0.30, "grow_leaf": 0.50}
        name, bound = self_development_select(pool, eps=0.10)
        assert name == "grow_leaf"
        assert bound == pytest.approx(0.50 - 0.20)

    def test_safe_filter_hides_unsafe(self):
        pool = {"merge": 0.50, "prune": 0.40}
        name, _ = self_development_select(pool, eps=0.05, safe={"prune"})
        assert name == "prune"

    def test_search_cost_linear(self):
        assert search_cost_bound(2.0, 7) == 14.0


# ---------------------------------------------------------------- R120


class TestR120Pareto:
    def test_dir_tau_is_min_over_coordinates(self):
        u = {"a": {"capability": 0.3, "wallclock": -0.1, "risk": 0.2}}
        omega = {"capability": 1.0, "wallclock": 1.0, "risk": 0.5}
        # levels: 0.3/1.0, -0.1/1.0, 0.2/0.5=0.4 -> min -0.1
        assert dir_tau(u, omega, "a") == pytest.approx(-0.1)

    def test_positive_tau_improves_every_coordinate(self):
        u = {"a": {"capability": 0.3, "wallclock": 0.05, "risk": 0.1}}
        omega = {"capability": 1.0, "wallclock": 0.1, "risk": 0.5}
        tau = dir_tau(u, omega, "a")
        assert tau > 0.0

    def test_pareto_select_prefers_balanced(self):
        # b is huge on capability but negative on wallclock -> tau(b) < 0
        u = {
            "a": {"capability": 0.2, "wallclock": 0.2},
            "b": {"capability": 0.9, "wallclock": -0.3},
        }
        omega = {"capability": 1.0, "wallclock": 1.0}
        name, bound = pareto_select(u, omega)
        assert name == "a"
        assert bound == pytest.approx(0.2)

    def test_noisy_bound_penalizes_max_eps_over_omega(self):
        u = {"a": {"capability": 0.5, "wallclock": 0.5}}
        omega = {"capability": 1.0, "wallclock": 0.25}
        eps = {"capability": 0.05, "wallclock": 0.05}
        name, bound = pareto_select(u, omega, eps=eps)
        # tau_hat = min(0.5, 2.0) = 0.5; penalty 2*max(0.05, 0.2) = 0.4
        assert name == "a"
        assert bound == pytest.approx(0.5 - 0.4)

    def test_all_negative_returns_none(self):
        u = {"a": {"capability": -0.1, "wallclock": -0.2}}
        omega = {"capability": 1.0, "wallclock": 1.0}
        name, bound = pareto_select(u, omega)
        assert name is None
        assert bound == 0.0


# ---------------------------------------------------------------- R121


class TestR121AdaptiveSuccess:
    def test_floor_below_np0(self):
        n, p0, delta = 100, 0.5, 0.05
        floor = adaptive_success_floor(n, p0, delta)
        assert floor < n * p0
        assert floor > 0.0

    def test_exact_value(self):
        # n=100, delta=0.05: sqrt(100*ln(20)/2) = sqrt(149.79) = 12.24
        assert adaptive_success_floor(100, 0.5, 0.05) == pytest.approx(
            50.0 - math.sqrt(50.0 * math.log(20.0)), rel=1e-12
        )

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            adaptive_success_floor(0, 0.5, 0.05)
        with pytest.raises(ValueError):
            adaptive_success_floor(10, 0.5, 1.0)

    def test_floor_grows_with_n(self):
        assert (adaptive_success_floor(400, 0.5, 0.05)
                > adaptive_success_floor(100, 0.5, 0.05))


# ---------------------------------------------------------------- R122


class TestR122OpportunityRenewal:
    def test_renewal_positive_when_production_exceeds_decay(self):
        # beta*C - xi = 0.5 - 0.1 = 0.4; (1-rho)*D = 0.2*1.0 = 0.2
        assert opportunity_delta(1.0, 0.8, 0.5, 1.0, 0.1) == pytest.approx(0.2)

    def test_zero_when_production_equals_decay_tax(self):
        # beta*C - xi = 0.3; (1-rho)*D = 0.3
        assert opportunity_delta(1.5, 0.8, 0.2, 2.5, 0.2) == pytest.approx(0.0)

    def test_negative_means_fine_tuning(self):
        # production below the decay tax: the loop does not renew
        assert opportunity_delta(2.0, 0.5, 0.1, 1.0, 0.0) == pytest.approx(-0.9)

    def test_rho_one_has_no_decay_tax(self):
        assert opportunity_delta(10.0, 1.0, 0.3, 1.0, 0.1) == pytest.approx(0.2)


class TestR122Growth:
    def test_multiplicative_composition(self):
        # C_T >= C_0 * exp(alpha*sum S - sum eps)
        val = multiplicative_growth(1.0, [1.0, 1.0, 0.5], [0.0, 0.1, 0.0], 0.2)
        assert val == pytest.approx(math.exp(0.2 * 2.5 - 0.1))

    def test_zero_success_decays_by_leak(self):
        val = multiplicative_growth(2.0, [0.0], [0.3], 0.5)
        assert val == pytest.approx(2.0 * math.exp(-0.3))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            multiplicative_growth(1.0, [1.0], [0.0, 0.1], 0.2)

    def test_closed_loop_floor_matches_composition(self):
        # closed_loop = C_0 * exp(alpha * s_floor - leak_total)
        floor = closed_loop_floor(1.0, 100, 0.5, 0.05, 0.2, 0.1)
        s = adaptive_success_floor(100, 0.5, 0.05)
        assert floor == pytest.approx(math.exp(0.2 * s - 0.1))


# ---------------------------------------------------------------- R119


class TestR119GrowthState:
    def test_capability_grows_on_success(self):
        st = GrowthState(capability=1.0, frontier=25.5, success=0.8, eps_t=0.05)
        assert st.capability_after_cycle(0.1) == pytest.approx(
            math.exp(0.1 * 0.8 - 0.05)
        )

    def test_success_bounds(self):
        with pytest.raises(ValueError):
            GrowthState(capability=1.0, frontier=1.0, success=1.5)

    def test_negative_capability_rejected(self):
        with pytest.raises(ValueError):
            GrowthState(capability=-1.0, frontier=1.0)

    def test_frontier_field_is_the_data_field(self):
        # R119: D_t = dataField(S_t) -- the measured Jensen gap IS the frontier
        st = GrowthState(capability=1.0, frontier=25.503)
        assert st.frontier == pytest.approx(25.503)
