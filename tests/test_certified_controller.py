"""Tests for the certified controller core (ALGORITHMS.md §0/§1/§9/§17).

The controller implements the §0 loop decision core as a PURE module:
no subprocess, no GPU, no torch at import time (the merge-price gate is
imported lazily inside ``merge_gate_admission``). These tests pin the
formulas as transcribed from the SSOT (C:/primes/ALGORITHMS.md):

- §1  action gains: argmax Gamma/K selection
- §9  certified A/B: accept iff CE_A - CE_B > 2*eps with the Hoeffding n
- §0/§4 stop_condition (liveness_two_axis R80)
- §17 leak_gate (distill_leak_gate R134) and exhausted_when
- R72 prune certificate: savings - delta^2/4
- §3  merge admission delegates to merge_price.merge_gate (twoGap)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hagi.train import certified_controller as cc  # noqa: E402


# --- §1: estimate_actions / select_action --------------------------------


def test_estimate_actions_computes_each_theory_formula():
    m = {
        "two_gap": 0.30, "merge_cost": 2.0,
        "eta": 0.5, "d_star_norm_sq": 0.8, "joint_cost": 4.0,
        "g_insight": 0.10, "internalize_cost": 1.0,
        "savings": 0.20, "delta_prune": 0.4, "prune_cost": 0.5,
        "g_new": 0.50, "grow_cost": 0.15, "grow_exec_cost": 1.0,
        "c_k": 0.12, "delta_k": 0.04, "distill_cost": 1.0,
    }
    acts = {a.name: a for a in cc.estimate_actions(m)}
    assert acts["merge"].gamma == pytest.approx(0.30)                  # twoGap
    assert acts["joint"].gamma == pytest.approx(0.5 * 0.8 / 2.0)       # eta*||d*||^2/2
    assert acts["internalize"].gamma == pytest.approx(0.10)            # G_insight
    assert acts["prune"].gamma == pytest.approx(0.20 - 0.4 ** 2 / 4.0)  # savings - d^2/4
    assert acts["grow_leaf"].gamma == pytest.approx(0.50 - 0.15)       # G_new - cost
    assert acts["distill"].gamma == pytest.approx(0.12 - 0.04)         # c_k - delta_k


def test_prune_certificate_formula_is_exactly_savings_minus_delta_sq_over_4():
    # R72: the certificate is the closed form, not a tuneable.
    for savings, delta in [(1.0, 0.0), (2.0, 1.0), (0.5, 0.2)]:
        a = cc.estimate_actions({
            "savings": savings, "delta_prune": delta, "prune_cost": 1.0,
        })[0]
        assert a.gamma == pytest.approx(savings - delta ** 2 / 4.0)


def test_select_action_is_argmax_ratio():
    acts = [
        cc.Action("a", gamma=1.0, cost=10.0),
        cc.Action("b", gamma=1.0, cost=2.0),   # ratio 0.5 -- the winner
        cc.Action("c", gamma=3.0, cost=10.0),  # ratio 0.3
    ]
    assert cc.select_action(acts).name == "b"


def test_select_action_excludes_non_positive_gain_and_unaffordable():
    acts = [
        cc.Action("neg", gamma=-1.0, cost=1.0),
        cc.Action("zero", gamma=0.0, cost=1.0),
        cc.Action("big", gamma=100.0, cost=99.0),
    ]
    assert cc.select_action(acts, budget=50.0) is None
    assert cc.select_action(acts, budget=100.0).name == "big"
    assert cc.select_action([]) is None


def test_select_action_breaks_ties_on_list_order():
    a = cc.Action("first", gamma=2.0, cost=2.0)
    b = cc.Action("second", gamma=1.0, cost=1.0)   # identical ratio
    assert cc.select_action([a, b]).name == "first"
    assert cc.select_action([b, a]).name == "second"


# --- §9: certified_ab -----------------------------------------------------


def test_hoeffding_n_matches_the_formula():
    assert cc.hoeffding_n(0.1, 0.05) == \
        math.ceil(math.log(2 / 0.05) / (2 * 0.1 ** 2))
    assert cc.hoeffding_n(0.5, 0.05) == \
        math.ceil(math.log(2 / 0.05) / (2 * 0.5 ** 2))


def test_certified_ab_accepts_a_clear_improvement_with_enough_n():
    v = cc.certified_ab(3.00, 10_000, 2.80, 10_000, eps=0.05, delta=0.05)
    assert v.accepted and not v.undecided
    assert v.delta == pytest.approx(0.20)


def test_certified_ab_rejects_a_clear_regression_with_enough_n():
    v = cc.certified_ab(2.80, 10_000, 3.00, 10_000, eps=0.05, delta=0.05)
    assert v.rejected and not v.accepted


def test_certified_ab_is_undecided_below_the_hoeffding_threshold():
    # Same 0.20-nat margin as above, but n far below log(2/delta)/(2eps^2):
    # the margin CANNOT be certified, in either direction.
    v = cc.certified_ab(3.00, 200, 2.80, 200, eps=0.05, delta=0.05)
    assert v.undecided and not v.accepted and not v.rejected
    assert v.n < v.n_required


def test_certified_ab_boundary_is_strictly_greater_than_two_eps():
    n = 100_000
    # d == 2*eps exactly: not accepted (strict inequality, §9).
    v = cc.certified_ab(0.1, n, 0.0, n, eps=0.05, delta=0.05)
    assert not v.accepted
    # Just above: accepted.
    v = cc.certified_ab(0.1 + 1e-9, n, 0.0, n, eps=0.05, delta=0.05)
    assert v.accepted


def test_certified_ab_uses_the_weaker_sample_size():
    # min(n_a, n_b) bounds the premise: a huge A sample cannot certify
    # against a small B sample.
    v = cc.certified_ab(3.00, 10 ** 9, 2.80, 10, eps=0.05, delta=0.05)
    assert v.undecided


def test_certified_ab_reject_margin_can_be_narrower():
    # deep200_certify's historical asymmetry: regression certified at eps.
    v = cc.certified_ab(2.0, 100_000, 2.0 + 0.06, 100_000,
                        eps=0.05, delta=0.05, reject_margin=0.05)
    assert v.rejected
    v = cc.certified_ab(2.0, 100_000, 2.0 + 0.06, 100_000,
                        eps=0.05, delta=0.05)
    assert v.undecided  # default reject margin is the same 2*eps


def test_certified_ab_and_deep200_certify_verdicts_agree():
    # The script delegates its rule to certified_ab; pin the equivalence.
    sys.path.insert(0, str(ROOT / "scripts" / "growth"))
    import deep200_certify  # noqa: E402
    for d in (0.25, -0.15, 0.05, 0.0, -0.05):
        s = deep200_certify.verdict(d, 0.096)
        v = cc.certified_ab(0.0, math.inf, -d, math.inf,
                            eps=0.096, delta=0.05, reject_margin=0.096)
        assert (s == "IMPROVED (certified)") == v.accepted
        assert (s == "REGRESSION (certified)") == v.rejected


# --- §0/§4/§17: gates -----------------------------------------------------


def test_stop_condition_is_consenus_and_small_injection():
    # liveness_two_axis (R80): stop iff consensus AND inj <= xi.
    assert cc.stop_condition(True, 0.0, 0.1) is True
    assert cc.stop_condition(True, 0.05, 0.1) is True
    assert cc.stop_condition(False, 0.0, 0.1) is False   # disagreement alive
    assert cc.stop_condition(True, 0.5, 0.1) is False    # data axis alive


def test_leak_gate_stops_when_slack_reaches_the_gain():
    # distill_leak_gate (R134): delta >= c => STOP the recursion.
    assert cc.leak_gate(cycle_gain=0.12, distill_slack=0.04) is True
    assert cc.leak_gate(cycle_gain=0.12, distill_slack=0.12) is False
    assert cc.leak_gate(cycle_gain=0.12, distill_slack=0.20) is False


def test_exhausted_check_is_the_decayed_frontier_below_eps_over_gamma():
    # exhausted_when: rho^n * D0 < eps/gamma => EXHAUSTED (switch corpus).
    assert cc.exhausted_check(0.5, 0.08, 0.1, 2.0) is True   # 0.04 < 0.05
    assert cc.exhausted_check(0.5, 0.2, 0.1, 2.0) is False   # 0.10 >= 0.05
    assert cc.exhausted_check(0.5, 1.0, 0.1, 2.0) is False   # 0.5 >= 0.05
    assert cc.exhausted_check(0.5, 1.0, 0.1, 2.0, n=5) is True  # 0.03125 < 0.05
    with pytest.raises(ValueError):
        cc.exhausted_check(1.5, 1.0, 0.1, 2.0)
    with pytest.raises(ValueError):
        cc.exhausted_check(0.5, 1.0, 0.0, 2.0)


# --- §3: merge gate delegation (twoGap) -----------------------------------


def test_merge_gate_admission_delegates_to_merge_price():
    torch = pytest.importorskip("torch")
    from hagi.train.merge_price import merge_gate

    # The delegation must be the SAME predicate, not a re-implementation.
    for two_gap in (0.5, 1.0, 2.0):
        assert cc.merge_gate_admission(two_gap, [0.1], kappa=0.05, s=1.0, n=200) == \
            merge_gate(two_gap, [0.1], kappa=0.05, s=1.0, n=200)


def test_merge_gate_admission_signs():
    pytest.importorskip("torch")
    # twoGap below the price+compression cost: refuse.
    assert cc.merge_gate_admission(0.05, [0.1], kappa=0.05, s=1.0, n=200) is False
    # twoGap well above: admit.
    assert cc.merge_gate_admission(5.0, [0.1], kappa=0.05, s=1.0, n=200) is True


# --- §23: phase annotations ------------------------------------------------


def test_phase_log_line_names_the_phase():
    line = cc.phase_log_line(cc.GrowthPhase.GATE, "gen8", "detail")
    assert "[phase GATE]" in line and "lane=gen8" in line
    with pytest.raises(ValueError):
        cc.phase_log_line("NOT_A_PHASE", "x")
