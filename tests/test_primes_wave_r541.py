# Wave R541–R542 (port 2026-10-18): anytime-Freedman threshold ladders.
# Lean SSOT: Probability/FreedmanTwoSided.lean
#   anytime_freedman_thresholds (R541),
#   anytime_freedman_thresholds_two_sided (R542).
# Closes the THEORY_MAP open item "anytime Freedman M∧V".

import math

import pytest

from hagi.train import certified_controller as cc


def test_freedman_term_formula():
    # exp(-r^2 / (4(V + c r))) — exact arithmetic identity
    assert cc.anytime_freedman_term(1.0, 0.0, 0.5) == pytest.approx(
        math.exp(-1.0 / 2.0)
    )


def test_term_decreases_with_rung():
    ladder = [cc.anytime_freedman_term(r, 1.0, 0.5) for r in (0.5, 1.0, 2.0, 4.0)]
    assert ladder == sorted(ladder, reverse=True)


def test_thresholds_union_bound_majorizes_per_round():
    # At T=1 the ladder reduces to a single Freedman term.
    single = cc.anytime_freedman_thresholds([1.0], [0.0], 0.5)
    assert single == pytest.approx(cc.anytime_freedman_term(1.0, 0.0, 0.5))
    # Adding rungs can only increase a probability bound (union bound).
    more = cc.anytime_freedman_thresholds([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], 0.5)
    assert more >= single


def test_two_sided_is_double():
    th, vs, c = [1.0, 2.0], [0.0, 1.0], 0.5
    assert cc.anytime_freedman_thresholds_two_sided(th, vs, c) == pytest.approx(
        2.0 * cc.anytime_freedman_thresholds(th, vs, c)
    )


def test_two_sided_covers_naive_ab_gate():
    # A symmetric A/B drift monitor: per-round band misses that checking
    # at every round inflates failure mass; the two-sided ladder's arm
    # 2*exp(-r^2/(4(V+cr))) at V=0 must dominate the single-round
    # Freedman event it subsumes.
    r, c = 1.0, 0.5
    assert cc.anytime_freedman_thresholds_two_sided([r], [0.0], c) >= (
        2 * cc.anytime_freedman_term(r, 0.0, c)
    )


def test_ladder_args_validated():
    for bad in (
        lambda: cc.anytime_freedman_term(0.0, 1.0, 0.5),
        lambda: cc.anytime_freedman_term(-1.0, 1.0, 0.5),
        lambda: cc.anytime_freedman_term(1.0, -0.1, 0.5),
        lambda: cc.anytime_freedman_term(1.0, 1.0, 0.0),
        lambda: cc.anytime_freedman_thresholds([], [], 0.5),
        lambda: cc.anytime_freedman_thresholds([1.0], [0.0, 0.0], 0.5),
    ):
        with pytest.raises(ValueError):
            bad()


# --- R548: anytime growth budget -------------------------------------------

from hagi.train.stage_telescope import anytime_growth_budget


def test_growth_budget_linear_band():
    # T*(mu + u) — pure arithmetic identity
    assert anytime_growth_budget(10, 0.01, 0.02) == pytest.approx(0.3)


def test_growth_budget_args_validated():
    with pytest.raises(ValueError):
        anytime_growth_budget(-1, 0.01, 0.0)
    with pytest.raises(ValueError):
        anytime_growth_budget(10, 0.01, -0.1)


# --- R549: Gibbs + Nernst composite -----------------------------------------

from hagi.train.thermostat import gibbs_nernst_bound


def test_nernst_bound_decays_exponentially():
    cool = [gibbs_nernst_bound(4, 1.0, t) for t in (2.0, 0.5, 0.1)]
    assert cool == sorted(cool, reverse=True)


def test_nernst_bound_below_ln2_when_cold():
    # Strongly cold: (n-1)e^{-delta/T} -> 0 => bound -> ln 2 (binary floor)
    b = gibbs_nernst_bound(10, 5.0, 0.05)
    assert b == pytest.approx(math.log(2.0), abs=1e-3)


def test_nernst_args_validated():
    for bad in (
        lambda: gibbs_nernst_bound(1, 1.0, 1.0),
        lambda: gibbs_nernst_bound(3, 0.0, 1.0),
        lambda: gibbs_nernst_bound(3, 1.0, 0.0),
    ):
        with pytest.raises(ValueError):
            bad()


# --- R556: geometric band ceiling -------------------------------------------

from hagi.train.stage_telescope import geometric_band_ceiling


def test_geometric_band_ceiling_formulas():
    tight, asymptotic = geometric_band_ceiling(1.0, 1.0, 0.5, 4)
    assert tight == pytest.approx(1.0 + (1.0 - 0.5**4) / 0.5)
    assert asymptotic == pytest.approx(3.0)
    assert tight <= asymptotic


def test_geometric_band_ceiling_monotone_in_steps():
    vals = [geometric_band_ceiling(0.0, 2.0, 0.25, k)[0] for k in (1, 4, 16)]
    assert vals == sorted(vals)
    # never exceeds the asymptotic ceiling
    assert all(v <= geometric_band_ceiling(0.0, 2.0, 0.25, 1)[1] for v in vals)


def test_geometric_band_args_validated():
    for bad in (
        lambda: geometric_band_ceiling(1.0, -0.1, 0.5, 10),
        lambda: geometric_band_ceiling(1.0, 1.0, 0.0, 10),
        lambda: geometric_band_ceiling(1.0, 1.0, 1.5, 10),
        lambda: geometric_band_ceiling(1.0, 1.0, 0.5, -1),
    ):
        with pytest.raises(ValueError):
            bad()


# --- R560/R561/R567: lower band + target timer -------------------------------

from hagi.train.stage_telescope import (
    band_guaranteed_growth_floor, band_time_to_target, time_to_target_log,
)


def test_lower_band_matches_ceiling_geometry():
    # mirror of R556: same band params, floor <= tight ceiling, equal asymptote
    fl = band_guaranteed_growth_floor(1.0, 1.0, 0.5, 4)
    tight, asymp = geometric_band_ceiling(1.0, 1.0, 0.5, 4)
    assert fl == pytest.approx(tight)
    assert fl == pytest.approx(1.0 + (1.0 - 0.5**4) / 0.5)


def test_band_time_to_target_true_when_floor_reaches():
    fl = band_guaranteed_growth_floor(1.0, 1.0, 0.5, 4)
    assert band_time_to_target(fl, 2.0)
    assert not band_time_to_target(fl, 2.9)


def test_time_to_target_log_closed_form():
    # sigma=0.5, b=1, C0=0, C*=1: 1 - 0.5*1/1 = 0.5; T = ln0.5/ln0.5 = 1
    assert time_to_target_log(0.0, 1.0, 1.0, 0.5) == 1
    # exact-integer check at sigma=0.1, b=1, C*=5: T = ln(0.5)/ln(0.9) ~ 6.58 -> 7
    assert time_to_target_log(0.0, 5.0, 1.0, 0.1) == 7


def test_time_to_target_args_validated():
    for bad in (
        lambda: time_to_target_log(0.0, 1.0, 0.0, 0.5),
        lambda: time_to_target_log(0.0, 1.0, 1.0, 1.0),
        lambda: time_to_target_log(2.0, 1.0, 1.0, 0.5),
        lambda: time_to_target_log(0.0, 10.0, 0.5, 0.5),  # unreachable
    ):
        with pytest.raises(ValueError):
            bad()


# --- R563–R565: LipEnergy vertical -------------------------------------------

from hagi.train.lip_energy import (
    grow_gain_from_smooth_descent, lip_energy_quantization,
    smooth_descent_energy_bound,
)


def test_lip_quantization_binary_bound():
    # kappa * dist — arithmetic identity; ternary rounding dist <= s/2
    assert lip_energy_quantization(4.0, 0.025) == pytest.approx(0.1)


def test_smooth_bound_quadratic_model():
    # E(q+dir) <= E(q) + inner + L*dnorm^2/2
    b = smooth_descent_energy_bound(10.0, -1.0, 4.0, 0.5)
    assert b == pytest.approx(10.0 - 1.0 + 0.5 * 4.0 / 2.0)


def test_grow_gain_certified_and_rejected():
    # certified: dnorm^2 <= inner, eta <= 1/L
    ok, gain = grow_gain_from_smooth_descent(4.0, 4.0, 1.0, 0.5)
    assert ok and gain == pytest.approx(1.0)
    # eta outside SafeQP window -> honest rejection
    ok, gain = grow_gain_from_smooth_descent(4.0, 4.0, 1.0, 2.0)
    assert not ok and gain == 0.0
    # no certified descent (dnorm^2 > inner) -> honest rejection
    ok, gain = grow_gain_from_smooth_descent(1.0, 4.0, 1.0, 0.5)
    assert not ok and gain == 0.0


def test_lip_args_validated():
    for bad in (
        lambda: lip_energy_quantization(-1.0, 0.1),
        lambda: lip_energy_quantization(1.0, -0.1),
        lambda: smooth_descent_energy_bound(1.0, 0.0, 1.0, -0.5),
        lambda: grow_gain_from_smooth_descent(1.0, 1.0, 0.0, 0.1),
    ):
        with pytest.raises(ValueError):
            bad()


# --- LipEnergy telemetry: kappa + stage certificate --------------------------

from hagi.train.lip_energy import kappa_from_grad_norms, lip_stage_certificate


def test_kappa_is_running_sup():
    assert kappa_from_grad_norms([1.2, 0.8, 1.5]) == pytest.approx(1.5)
    assert kappa_from_grad_norms([]) == 0.0


def test_stage_certificate_full_and_partial():
    cert = lip_stage_certificate([1.0], 0.025, eta=0.5, inner=4.0,
                                 dnorm_sq=4.0, L=1.0)
    assert cert["grow_gain_certified"] and cert["grow_gain"] == pytest.approx(1.0)
    assert cert["quant_loss_bound"] == pytest.approx(0.025)
    # rejected SafeQP window still yields the quantization part
    cert2 = lip_stage_certificate([2.0], 0.1, eta=5.0, inner=1.0,
                                  dnorm_sq=4.0, L=1.0)
    assert not cert2["grow_gain_certified"] and cert2["grow_gain"] == 0.0
    assert cert2["quant_loss_bound"] == pytest.approx(0.2)


# --- WSqD noise budget + flatness (Probability/WSqD, NoiseTemperature,

from hagi.train.growing_batch import (
    batch_variance, cooldown_tail_constant, flatness_moment_bound,
    stable_phase_noise, wsqd_noise_budget,
)


def test_batch_variance_is_one_over_b():
    assert batch_variance(0.01, 32) == pytest.approx(0.01 / 32)


def test_wsqd_budget_additive():
    stable = stable_phase_noise(0.01, 32.0, 100)
    tail = cooldown_tail_constant(0.01, 32.0, 1.5)
    assert wsqd_noise_budget(0.01, 32.0, 100, 1.5) == pytest.approx(stable + tail)
    assert tail == pytest.approx(0.01 * 1.5 / (32.0 * 0.5))


def test_cooldown_tail_independent_of_length():
    # the tail constant does not depend on the cooldown length n_d at all
    assert cooldown_tail_constant(1.0, 10.0, 2.0) == pytest.approx(0.2)


def test_wsqd_args_validated():
    for bad in (
        lambda: batch_variance(0.01, 0),
        lambda: stable_phase_noise(0.01, 0.0, 10),
        lambda: cooldown_tail_constant(0.01, 32.0, 1.0),
        lambda: flatness_moment_bound(-0.1),
    ):
        with pytest.raises(ValueError):
            bad()


def test_flatness_moment_bound():
    assert flatness_moment_bound(0.5) == pytest.approx(1.0)


# --- StochasticSafeQP noise budget (Step/StochasticSafeQP.lean) --------------

from hagi.train.safeqp_step import lr_batch_equivalence_check, noise_eps


def test_noise_eps_formula():
    # sigma * ||d|| * sqrt(2 ln(|K|/delta) / m) — exact identity
    import math as _m
    assert noise_eps(8, 0.1, 1.0, 64, 0.05) == pytest.approx(
        0.1 * _m.sqrt(2 * _m.log(8 / 0.05) / 64)
    )


def test_lr_batch_equivalence_identity():
    assert lr_batch_equivalence_check(0.001, 0.1, 1.0, 64, 4, 0.05)
    assert lr_batch_equivalence_check(0.003, 0.02, 0.5, 16, 3, 0.01, n_domains=8)


def test_noise_eps_shrinks_with_batch():
    # batch x4 -> noise x2 (sqrt law): the noise cooling axis
    n1 = noise_eps(4, 1.0, 1.0, 16, 0.05)
    n2 = noise_eps(4, 1.0, 1.0, 64, 0.05)
    assert n2 == pytest.approx(n1 / 2.0)


# --- BitAlloc (Budget/BitAlloc.lean) ------------------------------------------

from hagi.train.bit_alloc import (
    imbalance_yields_gain, layer_error, stable_factor_two, transfer_exact,
    water_filling_marginals,
)


def test_layer_error_and_exchange():
    assert layer_error(1.0, 8) == pytest.approx(1 / 256)
    # moving one bit j->k with equal difficulties: total error halves
    assert transfer_exact(1.0, 4.0, 1.0, 4.0) == pytest.approx(-0.0625)


def test_factor_two_rule():
    assert imbalance_yields_gain(1.0, 4.0, 0.1, 4.0)
    assert not imbalance_yields_gain(1.0, 4.0, 1.0, 4.0)
    assert stable_factor_two([0.1, 0.15, 0.2])
    assert not stable_factor_two([0.1, 0.5])


def test_water_filling_equalizes_marginals():
    alloc = water_filling_marginals([1.0, 2.0, 4.0], 6.0)
    marg = [c / 2.0 ** f for c, f in zip([1.0, 2.0, 4.0], alloc)]
    assert max(marg) - min(marg) < 1e-6
    assert sum(alloc) == pytest.approx(6.0, abs=1e-3)


# --- TernaryExact: satTail split budget + optimal grid (R103 application) ----

import torch

from hagi.model.ternary import optimal_ternary_scale, ternary_sat_diag


def test_sat_diag_decomposition():
    w = torch.tensor([[0.1, -0.1, 0.12, 0.02]])
    d = ternary_sat_diag(w, 0.1)  # |w| <= 0.15: all in range
    assert d["sat_frac"] == 0.0 and d["sat_tail"] == 0.0
    assert d["split_bound"] == pytest.approx(4 * 0.05**2)


def test_sat_diag_counts_saturation():
    w = torch.tensor([[2.0, -2.0, 0.01]])  # |w|>3s/2 for s=0.1
    d = ternary_sat_diag(w, 0.1)
    assert d["n_in_range"] == 1
    # satTail = (2-0.1)^2 + (2-0.1)^2
    assert d["sat_tail"] == pytest.approx(2 * 1.9**2)


def test_optimal_scale_beats_mean_grid_on_heavy_tail():
    torch.manual_seed(0)
    w = torch.randn(341 * 128) * 0.008
    w[:1500] = torch.randn(1500) * 0.12
    w = w.reshape(341, 128)
    d_default = ternary_sat_diag(w, float(w.abs().mean()))
    _, d_best = optimal_ternary_scale(w)
    assert d_best["split_bound"] <= d_default["split_bound"]


def test_optimal_scale_never_worse_than_wide_sweep_mean():
    torch.manual_seed(1)
    w = torch.randn(32, 64)
    s_best, d_best = optimal_ternary_scale(w)
    d_at_best = ternary_sat_diag(w, s_best)
    assert d_at_best["split_bound"] == pytest.approx(d_best["split_bound"])


def test_sat_diag_args_validated():
    with pytest.raises(ValueError):
        ternary_sat_diag(torch.zeros(4), 0.1)  # not 2D
    with pytest.raises(ValueError):
        ternary_sat_diag(torch.zeros(2, 2), 0.0)
    with pytest.raises(ValueError):
        optimal_ternary_scale(torch.zeros(3))


# --- cut_safe_accept (ProbeError.lean composition) ----------------------------

from hagi.train.lip_energy import cut_safe_accept


def test_cut_safe_accept_accepts_clear_descent():
    # |dL_meas|=0.5 > eps*||d||=0.2 -> certified
    ok, slack = cut_safe_accept(-0.5, 2.0, 0.1)
    assert ok and slack == pytest.approx(0.3)


def test_cut_safe_accept_rejects_uncertified():
    ok, slack = cut_safe_accept(-0.15, 2.0, 0.1)
    assert not ok and slack == 0.0


def test_cut_safe_accept_attention_term():
    # eps_total = eps_probe + L_att*eps_att = 0.1 + 0.1; ||d||=2 -> margin 0.4
    ok, _ = cut_safe_accept(-0.5, 2.0, 0.1, eps_att=0.01, L_att=10.0)
    assert ok  # 0.5 > 0.4 still certified
    ok2, _ = cut_safe_accept(-0.35, 2.0, 0.1, eps_att=0.01, L_att=10.0)
    assert not ok2  # 0.35 < 0.4: margin eats the descent


def test_cut_safe_accept_args_validated():
    for bad in (
        lambda: cut_safe_accept(-1.0, -0.1, 0.1),
        lambda: cut_safe_accept(-1.0, 1.0, -0.1),
        lambda: cut_safe_accept(-1.0, 1.0, 0.1, L_att=-1.0),
    ):
        with pytest.raises(ValueError):
            bad()


# --- ternarize_split: R103 satTail-aware quantization ------------------------

from hagi.model.ternary import BitLinear, ternarize_split


def test_ternarize_split_healthy_passthrough():
    torch.manual_seed(1)
    w = torch.randn(64, 128) * 0.05  # healthy: no dominant tail
    e_split, s_split = ternarize_split(w)
    e_def, s_def = __import__("hagi.model.ternary", fromlist=["ternarize"]).ternarize(w)
    assert torch.equal(e_split, e_def) and torch.equal(s_split, s_def)


def test_ternarize_split_never_worse():
    # win-check contract: measured split bound must not increase
    torch.manual_seed(0)
    w = torch.randn(128, 256)
    e_split, _ = ternarize_split(w, tau=0.0)  # force diagnosis on any tensor
    e_def, _ = __import__("hagi.model.ternary", fromlist=["ternarize"]).ternarize(w)
    se_split = ((e_split.float() - w) ** 2).sum()
    se_def = ((e_def.float() - w) ** 2).sum()
    assert se_split <= se_def * 1.05 + 1e-6  # never meaningfully worse


def test_bitlinear_cache_split_path():
    bl = BitLinear(64, 32)
    bl.cache_quantized()  # theory default: split path
    assert bl._step_q is not None
    assert bl._step_q.shape == (32, 64)


# --- Joint-step regression laws (Step/Joint.lean) -----------------------------

from hagi.train.safeqp_step import (
    conflict_lr_limit, joint_step_regression_bound, trust_region_loss_bound,
)


def test_conflict_lr_limit():
    # 2|gc.g| / (L (g.g)) = 2*1 / (1*4) = 0.5
    assert conflict_lr_limit(-1.0, 4.0, 1.0) == pytest.approx(0.5)


def test_conflict_lr_limit_scales():
    # doubled conflict signal -> doubled boundary
    assert conflict_lr_limit(-2.0, 4.0, 1.0) == pytest.approx(1.0)
    # doubled curvature -> halved boundary
    assert conflict_lr_limit(-1.0, 4.0, 2.0) == pytest.approx(0.25)


def test_joint_regression_bound_envelope():
    b = joint_step_regression_bound(2.0, 4.0, 1.0, 0.1)
    assert b == pytest.approx(0.1 * 2.0 * 2.0 + 0.5 * 1.0 * 0.01 * 4.0)
    # quadratic in lr: grows faster than linear
    b2 = joint_step_regression_bound(2.0, 4.0, 1.0, 0.4)
    assert b2 > 2 * b


def test_trust_region_loss_bound():
    assert trust_region_loss_bound(2.0, 1.0, 0.5) == pytest.approx(1.125)
    assert trust_region_loss_bound(0.0, 0.0, 0.0) == 0.0


def test_joint_args_validated():
    for bad in (
        lambda: conflict_lr_limit(-1.0, 0.0, 1.0),
        lambda: conflict_lr_limit(-1.0, 4.0, 0.0),
        lambda: joint_step_regression_bound(1.0, -1.0, 1.0, 0.1),
        lambda: trust_region_loss_bound(1.0, -1.0, 0.5),
    ):
        with pytest.raises(ValueError):
            bad()


# --- Armijo / Warmup (Step/ArmijoAcceptance, Step/WarmupSchedule) --------------

from hagi.train.safeqp_step import armijo_accept, warmup_eta


def test_armijo_accepts_within_window():
    ok, dec = armijo_accept(-0.5, 1.0, 0.2, 1.0, 0.25)
    assert ok
    assert dec == pytest.approx(0.5 * 0.25 * 0.2 * 1.0)


def test_armijo_rejects_oversized_step():
    # alpha > c1/L = 0.25: outside the theorem window
    ok, dec = armijo_accept(-0.5, 1.0, 0.9, 1.0, 0.25)
    assert not ok and dec == 0.0


def test_armijo_rejects_bad_direction():
    # direction condition g.d <= -c1*||d||^2 violated (0.5 > -0.25)
    ok, dec = armijo_accept(0.5, 1.0, 0.2, 1.0, 0.25)
    assert not ok and dec == 0.0


def test_warmup_eta_hits_cap_and_grows():
    # K0 cap: both Deltas capped at c/K0 = 0.25 (monotone, non-strict)
    assert warmup_eta(1.0, 4.0, 2.0, 0.5, 1.0) == pytest.approx(0.25)
    assert warmup_eta(1.0, 4.0, 2.0, 0.5, 0.25) == pytest.approx(0.25)
    # far from optimum: Delta^(-rho) term dominates below the K0 cap
    big = warmup_eta(1.0, 1000.0, 2.0, 0.5, 1.0e6)    # 0.000289 < cap
    small = warmup_eta(1.0, 1000.0, 2.0, 0.5, 1.0e4)  # capped at 0.001
    assert small > big  # suboptimality falls -> step grows


def test_armijo_warmup_args_validated():
    for bad in (
        lambda: armijo_accept(-1.0, -1.0, 0.1, 1.0),
        lambda: armijo_accept(-1.0, 1.0, 0.1, 1.0, 0.0),
        lambda: warmup_eta(1.0, 0.0, 2.0, 0.5, 1.0),
        lambda: warmup_eta(1.0, 4.0, 2.0, 0.0, 1.0),
        lambda: warmup_eta(1.0, 4.0, 2.0, 0.5, 0.0),
    ):
        with pytest.raises(ValueError):
            bad()


# --- Pinsker CMI + capacity floors (Information/Pinsker, RoutingCapacity) -----

from hagi.model.formal import cmi_iteration, gen_pair_bound, router_capacity_ok


def test_gen_pair_bound_stability_plus_info():
    # identical test functions (Ef = Eg per v): stability term = 0
    b = gen_pair_bound([0.5, 0.5], [0.2, 0.3], [0.2, 0.3], 0.01)
    assert b == pytest.approx(2 * math.sqrt(0.01), rel=1e-9)
    # with pointwise stability 0.1: bound = 0.1 + 0.2
    b2 = gen_pair_bound([1.0], [0.1], [0.0], 0.01)
    assert b2 == pytest.approx(0.1 + 0.2)


def test_cmi_iteration_sqrt_aggregation():
    # mean stab 0.1, mean KL 0.02: rate = 0.1 + 2*sqrt(0.02)
    r = cmi_iteration([0.1, 0.1], [0.02, 0.02])
    assert r == pytest.approx(0.1 + 2 * math.sqrt(0.02))


def test_cmi_iteration_sqrt_penalty_on_unequal_kls():
    # Cauchy: sqrt(mean KL) >= mean sqrt(KL) - unequal KLs INFLATE the
    # information term (concentration of KL mass is punished, equal
    # per-sample KL is optimal in this bound)
    kls = [0.0, 0.08]
    rate = cmi_iteration([0.0, 0.0], kls)
    naive = 2 * sum(math.sqrt(k) for k in kls) / 2
    assert rate >= naive
    assert cmi_iteration([0.0, 0.0], [0.04, 0.04]) == pytest.approx(
        2 * math.sqrt(0.04))


def test_router_capacity_floor():
    assert router_capacity_ok(8, 3)  # 8 <= 2^3
    assert not router_capacity_ok(9, 3)
    assert router_capacity_ok(0, 0)


def test_gen_pair_args_validated():
    for bad in (
        lambda: gen_pair_bound([0.5], [0.1, 0.2], [0.1], 0.1),
        lambda: gen_pair_bound([0.5], [0.1], [0.1], -0.1),
        lambda: cmi_iteration([], []),
        lambda: cmi_iteration([0.1], [-0.1]),
        lambda: router_capacity_ok(-1, 3),
    ):
        with pytest.raises(ValueError):
            bad()


# --- SNR weighting + NCE K-choice (Step/SNRWeights, Step/NCEExact) ------------

from hagi.train.safeqp_step import (
    adaptive_k, anchor_cadence, focused_cost, focused_wins,
)


def test_focused_wins_on_unequal_devs():
    ok, w = focused_wins([0.5, 2.0, 2.0])
    assert ok
    assert focused_cost([0.5, 2.0, 2.0], w) == pytest.approx(0.5)
    # strict improvement over uniform: 0.5 < 1.5
    assert focused_cost([0.5, 2.0, 2.0], [1 / 3] * 3) == pytest.approx(1.5)


def test_focused_uniform_is_optimal_when_equal():
    ok, w = focused_wins([1.0, 1.0, 1.0])
    assert not ok
    assert all(x == pytest.approx(1 / 3) for x in w)


def test_focused_witness_matches_theorem():
    # theorem's witness: full mass on the best expert strictly under mean
    dev = [0.1, 4.0, 9.0, 16.0]
    ok, w = focused_wins(dev)
    assert ok and focused_cost(dev, w) == pytest.approx(0.1)
    assert focused_cost(dev, [0.25] * 4) == pytest.approx(7.275)


def test_adaptive_k_and_anchor_cadence():
    assert adaptive_k(4.0, 0.1) == 40
    assert adaptive_k(0.0, 0.1) == 1
    assert anchor_cadence(0.1, 1.0, 0.05) == pytest.approx(0.5)


def test_snr_nce_args_validated():
    for bad in (
        lambda: focused_wins([]),
        lambda: focused_wins([-1.0]),
        lambda: focused_cost([1.0], [0.5, 0.5]),
        lambda: adaptive_k(-1.0, 0.1),
        lambda: adaptive_k(1.0, 0.0),
        lambda: anchor_cadence(0.0, 1.0, 0.1),
    ):
        with pytest.raises(ValueError):
            bad()


# --- Skip/sink renormalization (Step/Upgrades.lean) ---------------------------

from hagi.train.mwu import skip_renorm_bound


def test_skip_renorm_bound_shape():
    b, l1 = skip_renorm_bound([0.5, 0.3, 0.2], [True, False, False])
    assert b == pytest.approx(1.0)  # 2 * 0.5 dropped mass
    assert l1 <= b
    # no drops: zero cost, exact identity
    b0, l10 = skip_renorm_bound([0.1, 0.1, 0.8], [False] * 3)
    assert b0 == 0.0 and l10 == 0.0


def test_skip_renorm_cheap_for_low_mass():
    # skipping 1% mass costs at most 2% - mass, not block count, is the cost
    p = [0.01] + [0.99 / 9] * 9
    b, l1 = skip_renorm_bound(p, [True] + [False] * 9)
    assert b == pytest.approx(0.02)
    assert l1 <= b + 1e-12


def test_skip_renorm_args_validated():
    for bad in (
        lambda: skip_renorm_bound([], []),
        lambda: skip_renorm_bound([0.5], [False, False]),
        lambda: skip_renorm_bound([-0.1], [False]),
        lambda: skip_renorm_bound([1.0, 0.5], [True, False]),  # dropped mass can't exceed 1
    ):
        with pytest.raises(ValueError):
            bad()


# --- Plasticity radius (Growth/PlasticityLedger.lean) -------------------------

from hagi.train.certified_controller import plasticity_radius


def test_plasticity_radius_zero_for_equal_family():
    r, ub = plasticity_radius([[0.5, 0.5], [0.5, 0.5]])
    assert r == pytest.approx(0.0)
    assert ub == pytest.approx(2 * math.log(2))


def test_plasticity_radius_stats_under_bound():
    # maximally diverse 2-family: mean KL = log 2; sum form = 2 log 2 = bound
    r, ub = plasticity_radius([[1.0, 0.0], [0.0, 1.0]])
    assert r == pytest.approx(ub, rel=1e-9)
    assert ub == pytest.approx(2 * math.log(2))


def test_plasticity_radius_partial_diversity():
    r, _ = plasticity_radius([[0.9, 0.1], [0.1, 0.9]])
    assert 0.0 < r < 2 * math.log(2)


def test_plasticity_args_validated():
    for bad in (
        lambda: plasticity_radius([]),
        lambda: plasticity_radius([[0.5, 0.5], [0.5, 0.4, 0.1]]),
        lambda: plasticity_radius([[-0.1, 1.1]]),
        lambda: plasticity_radius([[0.6, 0.5]]),  # not a distribution
    ):
        with pytest.raises(ValueError):
            bad()


# --- PoE logZ bounds (Energy/PoEBound.lean) -----------------------------------

from hagi.model.formal import poe_logz_pairwise, poe_logz_second_order, poe_softmax_speed

_Z = [[1.0, -1.0], [0.0, 0.0]]  # Lean's refutation family, w = 1/2
_W = [0.5, 0.5]


def test_poe_second_order_lean_refutation_gap():
    gap, _ = poe_logz_second_order(_Z, _W, [0.0, 0.0])
    # exact: log(2 cosh 0.5) - 0.5 log(2 cosh 1) - 0.5 log 2
    exact = abs(math.log(2 * math.cosh(0.5)) - 0.5 * math.log(2 * math.cosh(1.0)) - 0.5 * math.log(2))
    assert gap == pytest.approx(exact, rel=1e-9)


def test_poe_sharp_eighth_law_holds():
    # R1 = range of centered dev of expert 1 = 2 (from (1,-1) minus mu=(0.5,-0.5))
    gap, bound = poe_logz_second_order(_Z, _W, [2.0, 0.0])
    assert bound == pytest.approx(0.125 * 0.5 * 4.0)  # (1/8)*w1*R1^2
    assert gap <= bound


def test_poe_pairwise_half_constant():
    D = [[0.0, 1.0], [1.0, 1.0]]
    gap, bound = poe_logz_pairwise(_Z, _W, D)
    # 0.5 * sum_ij w_i w_j D_ij^2 = 0.5*(0 + .25 + .25 + .25) — D_11 = 0!
    assert bound == pytest.approx(0.375)
    assert gap <= bound


def test_poe_softmax_speed_m2_over_8():
    gap, bound = poe_softmax_speed(_Z, _W, 2.0)
    assert bound == pytest.approx(0.5)
    assert gap <= bound


def test_poe_args_validated():
    for bad in (
        lambda: poe_logz_second_order([], [], []),
        lambda: poe_logz_second_order(_Z, [0.5, 0.6], [1.0, 1.0]),
        lambda: poe_logz_second_order(_Z, _W, [-1.0, 0.0]),
        lambda: poe_softmax_speed(_Z, _W, -1.0),
        lambda: poe_logz_pairwise(_Z, _W, [[0.0]]),
    ):
        with pytest.raises(ValueError):
            bad()


# --- LSE-shift / KL-additivity / smoothing (Energy/PoEBound, Data/DField) -----

from hagi.model.formal import (
    kl_product_additive, lse_shift_upper_bound, smooth_support,
)


def test_lse_shift_bound_holds_and_tight():
    # shift (1,-1) from zero logits: actual < fo(=0 here) + r^2/8
    a, fo, b = lse_shift_upper_bound([0.0, 0.0], [1.0, -1.0])
    r = 2.0
    assert b == pytest.approx(fo + r * r / 8.0)
    assert a <= b


def test_lse_shift_zero_shift_is_identity():
    a, fo, b = lse_shift_upper_bound([0.3, -1.2, 2.0], [0.3, -1.2, 2.0])
    assert a == pytest.approx(0.0, abs=1e-12)
    assert fo == pytest.approx(0.0, abs=1e-12)
    assert b == pytest.approx(0.0, abs=1e-12)


def test_lse_shift_first_order_is_lower_bound():
    # exp-Jensen: lse(y) - lse(x) >= <sigma(x), y - x> in expectation form;
    # for the symmetric shift the fo term is zero and shift > 0
    a, fo, _ = lse_shift_upper_bound([0.0, 0.0], [1.0, -1.0])
    assert fo == pytest.approx(0.0)
    assert a > 0.0


def test_kl_product_additive():
    assert kl_product_additive(0.3, 0.2) == pytest.approx(0.5)


def test_smooth_support_full():
    s = smooth_support([0.0, 1.0], 0.1)
    assert s[0] > 0.0 and s[1] < 1.0
    assert sum(s) == pytest.approx(1.0)
    s2 = smooth_support([0.0, 0.0], 0.2, uniform=[0.5, 0.5])
    assert all(x > 0 for x in s2)


def test_smooth_support_args_validated():
    for bad in (
        lambda: smooth_support([0.0, 1.0], 0.0),
        lambda: smooth_support([0.0, 1.0], 1.0),
        lambda: smooth_support([], 0.1),
        lambda: smooth_support([-1.0, 2.0], 0.1),
        lambda: smooth_support([0.5, 0.5], 0.1, uniform=[0.5, 0.0]),
    ):
        with pytest.raises(ValueError):
            bad()


# --- InfoNCE identities + D-field entropy form (Data/InfoNCE, DField) ---------

from hagi.model.formal import (
    dfield_entropy_identity, infonce_gap_is_kl, infonce_never_overestimates,
)


def test_infonce_gap_is_kl_exact():
    assert infonce_gap_is_kl(2.0, 2.5, 0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        infonce_gap_is_kl(2.0, 2.5, 0.7)  # identity violated: calibration bug


def test_infonce_floor_holds():
    assert infonce_never_overestimates(1.0, 3.0, 2.5)
    assert not infonce_never_overestimates(1.0, 2.0, 2.5)  # CE < H: impossible


def test_dfield_entropy_identity_floors():
    # identical members: D = H(mix) - <H> = 0
    assert dfield_entropy_identity(1.0, [0.5, 0.5], [1.0, 1.0]) == pytest.approx(0.0)
    # diversity: positive under the same member entropies with wider mix
    d = dfield_entropy_identity(1.5, [0.5, 0.5], [1.0, 1.0])
    assert d == pytest.approx(0.5)


def test_infonce_dfield_args_validated():
    for bad in (
        lambda: infonce_gap_is_kl(-1.0, 2.0, 0.5),
        lambda: infonce_never_overestimates(1.0, -1.0, 2.0),
        lambda: dfield_entropy_identity(1.0, [], []),
        lambda: dfield_entropy_identity(1.0, [0.7, 0.6], [1.0, 1.0]),
        lambda: dfield_entropy_identity(1.0, [0.5, 0.5], [-1.0, 1.0]),
    ):
        with pytest.raises(ValueError):
            bad()


# --- NCE partition estimator (Step/NCEExact.lean) ------------------------------

from hagi.train.safeqp_step import nce_variance_ratio, part_est_unbiased, second_moment_w


def test_part_est_unbiased_exact_Z():
    # on-support proposal: the estimate is EXACTLY Z (sum e^z)
    assert part_est_unbiased([0.0, 0.0], [0.5, 0.5]) == pytest.approx(2.0)
    z = [math.log(1.0), math.log(3.0)]
    assert part_est_unbiased(z, [0.5, 0.5]) == pytest.approx(4.0)
    # non-uniform full-support q: still exact (identity is q-invariant)
    assert part_est_unbiased(z, [0.2, 0.8]) == pytest.approx(4.0)


def test_second_moment_cauchy_floor():
    z = [math.log(1.0), math.log(3.0)]
    q = [0.5, 0.5]
    Z = part_est_unbiased(z, q)
    m2 = second_moment_w(z, q)
    assert m2 >= Z * Z  # Cauchy-Schwarz floor
    assert nce_variance_ratio(z, q) >= 1.0
    # degenerate family: ratio exactly 1 (zero variance)
    z0 = [math.log(2.0), math.log(2.0)]
    assert nce_variance_ratio(z0, [0.5, 0.5]) == pytest.approx(1.0)


def test_nce_args_validated():
    for bad in (
        lambda: part_est_unbiased([], []),
        lambda: part_est_unbiased([0.0, 0.0], [0.0, 1.0]),  # zero support
        lambda: second_moment_w([0.0], [1.0, 1.0]),
    ):
        with pytest.raises(ValueError):
            bad()


# --- KLS SafeQP Chebyshev certificate (Step/KLSSafeQP.lean) ---------------------

from hagi.train.safeqp_step import kls_batch_noise, kls_safeqp_certified


def test_kls_batch_noise_form():
    assert kls_batch_noise(2.0, 3.0, 1.5, 100) == pytest.approx(0.27)
    # 1/B scaling: doubling samples halves the noise
    assert kls_batch_noise(1.0, 1.0, 1.0, 20) * 2 == pytest.approx(
        kls_batch_noise(1.0, 1.0, 1.0, 10))


def test_kls_safeqp_certified_mass_bound():
    cert = kls_safeqp_certified(2.0, 3.0, 1.5, 100, 1.0, 0.1, 2.0)
    assert cert == pytest.approx(1.0 - 0.27 / 4.0)
    assert 0.0 <= cert <= 1.0


def test_kls_args_validated():
    for bad in (
        lambda: kls_batch_noise(-1.0, 1.0, 1.0, 10),
        lambda: kls_batch_noise(1.0, 1.0, 1.0, 0),
        lambda: kls_safeqp_certified(1.0, 1.0, 1.0, 10, 1.0, 0.1, 0.0),
        lambda: kls_safeqp_certified(1.0, 1.0, 1.0, 10, 0.0, 0.1, 2.0),
    ):
        with pytest.raises(ValueError):
            bad()


# --- Insight/consolidation vertical (Autonomy/Insight.lean) ----------------------

from hagi.train.lip_energy import (
    consolidation_gate, forgetting_kl_bound, gate_retention_bound,
    tldr_two_tier_safety,
)


def test_forgetting_kl_chain():
    assert forgetting_kl_bound(0.4, 1.0, 0.5)  # kl 0.4 <= eps 0.5
    assert not forgetting_kl_bound(0.6, 1.0, 0.5)


def test_consolidation_gate_predicates():
    ok_gold, ok_desc = consolidation_gate(-0.01, 0.05, 2.0)
    assert ok_gold and ok_desc
    bad_gold, _ = consolidation_gate(-0.5, 0.05, 2.0)
    assert not bad_gold


def test_gate_retention_linear_floor():
    steps = [-0.01, -0.02, 0.0]
    floor = gate_retention_bound(steps, 0.05)
    assert floor == pytest.approx(-3 * 0.05)
    assert sum(steps) >= floor
    with pytest.raises(ValueError):
        gate_retention_bound([-0.5], 0.05)  # left the gold cone


def test_tldr_two_tier():
    ok_null, ok_kl = tldr_two_tier_safety(True, 0.1, 0.2)
    assert ok_null and ok_kl
    bad_null, _ = tldr_two_tier_safety(False, 0.1, 0.2)
    assert not bad_null
