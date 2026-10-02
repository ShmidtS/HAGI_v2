"""Tests for the PPT discovery layer (R98).

The module's own docstring records the boundary it must not cross: the power
target is a power of the JOINT SEQUENCE probability, NOT a tokenwise
temperature, and the Concat law ``softmax(c z) != softmax(z)`` does not
transfer. So these tests pin the SEQUENCE-level behaviour and refuse to
assert any tokenwise equivalence that was never proved.
"""

from __future__ import annotations

import math

import pytest
import torch

from hagi.train.discovery_ppt import (
    mh_acceptance_ratio,
    ppt_success_lower,
    ppt_target,
    stationary_good_mass,
    steps_to_mix,
    truncation_bias,
    tv_contraction_bound,
)


# --- pptTarget_isProbability / _pos ---------------------------------------


def test_target_is_a_distribution():
    """The theorem's claim: a proper distribution, so ``sum == 1``."""
    p = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    t = ppt_target(p, 1.0)
    assert float(t.sum()) == pytest.approx(1.0, abs=1e-12)
    assert bool((t >= 0).all())


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_every_alpha_gives_a_distribution(alpha):
    """``alpha > 0`` is the whole premise; any such alpha is valid."""
    p = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    t = ppt_target(p, alpha)
    assert float(t.sum()) == pytest.approx(1.0, abs=1e-12)


def test_alpha_one_is_the_input_normalized():
    """``alpha = 1`` is the identity: ``p^1 / Z_1 == p / sum(p)``."""
    p = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    assert torch.allclose(ppt_target(p, 1.0), p / p.sum(), atol=1e-12)


def test_scaling_the_input_does_not_change_the_target():
    """``pi_alpha`` is scale-free in its input: only ratios matter."""
    p = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    assert torch.allclose(ppt_target(p, 2.0), ppt_target(p * 137.0, 2.0), atol=1e-12)


def test_larger_alpha_concentrates_on_the_heavy_weight():
    """Sharpening moves mass TOWARD the largest weight."""
    p = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    masses = [float(ppt_target(p, a)[0]) for a in (0.5, 1.0, 2.0, 4.0)]
    assert masses == sorted(masses)
    assert masses[-1] > masses[0]


def test_good_mass_is_monotone_in_alpha():
    """The R95 premise: raising alpha raises the chance of a good draw."""
    p = torch.tensor([0.9, 0.05, 0.03, 0.02], dtype=torch.float64)
    good = torch.tensor([True, False, False, False])
    m = [stationary_good_mass(p, good, a) for a in (0.5, 1.0, 2.0, 8.0)]
    assert m == sorted(m)
    assert m[0] < 1.0


def test_a_dominant_weight_is_already_mass_under_alpha_one():
    """The boundary the sharpening operates on: mass is near one at
    ``alpha = 1`` and saturates at one, so the useful regime is small
    alpha, where the sharpening actually reorders mass."""
    p = torch.tensor([0.001, 0.999], dtype=torch.float64)
    good = torch.tensor([False, True])
    assert stationary_good_mass(p, good, 1.0) == pytest.approx(0.999)
    assert stationary_good_mass(p, good, 200.0) == pytest.approx(1.0, abs=1e-6)


def test_an_empty_good_set_has_mass_zero_at_every_alpha():
    """No good candidate is a certainty of failure, not a small number."""
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    good = torch.zeros(2, dtype=torch.bool)
    assert stationary_good_mass(p, good, 3.0) == 0.0


def test_all_good_is_mass_one_and_is_independent_of_alpha():
    p = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    good = torch.ones(3, dtype=torch.bool)
    for a in (0.3, 1.0, 7.0):
        assert stationary_good_mass(p, good, a) == pytest.approx(1.0, abs=1e-12)


# --- pptTarget_isProbability's premises are enforced, not assumed ---------


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.001])
def test_non_positive_alpha_is_refused(bad):
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        ppt_target(p, bad)


def test_an_all_zero_input_is_refused():
    """Every premise of the proof is checked before the arithmetic."""
    with pytest.raises(ValueError):
        ppt_target(torch.zeros(4, dtype=torch.float64), 1.0)


def test_negative_weights_are_clamped_rather_than_propagated():
    """A negative weight is not a probability; it must not enter the sum."""
    p = torch.tensor([0.6, -0.2, 0.6], dtype=torch.float64)
    t = ppt_target(p, 1.0)
    assert float(t.sum()) == pytest.approx(1.0, abs=1e-12)
    assert bool((t >= 0).all())


def test_a_single_positive_entry_is_still_a_distribution():
    """The boundary ``Z_alpha -> 0``: handled, not raised on."""
    p = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    t = ppt_target(p, 3.0)
    assert float(t.sum()) == pytest.approx(1.0, abs=1e-12)
    assert int(t.argmax()) == 2


def test_float32_input_is_computed_in_double():
    """No precision loss from the input dtype: the port computes in f64."""
    p = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float32)
    assert float(ppt_target(p, 1.0).sum()) == pytest.approx(1.0, abs=1e-12)
    assert ppt_target(p, 1.0).dtype == torch.float64


# --- mh_stationary -------------------------------------------------------


def test_a_favourable_move_is_always_accepted():
    """``min(1, r) == 1`` when ``r >= 1`` -- no needless rejections."""
    assert mh_acceptance_ratio(-1.0, -2.0, -1.0, -1.0) == 1.0
    assert mh_acceptance_ratio(-1.0, -5.0, 0.0, 0.0) == 1.0


def test_a_zero_ratio_move_is_accepted():
    assert mh_acceptance_ratio(-2.0, -2.0, -1.0, -1.0) == 1.0


def test_the_acceptance_is_the_proposal_corrected_ratio():
    """``pi(y) q(x|y) / (pi(x) q(y|x))``, computed in log space."""
    # log pi(y) - log pi(x) = -1, and q is symmetric so the proposal cancels.
    assert mh_acceptance_ratio(-1.0, 0.0, 0.0, 0.0) == pytest.approx(math.exp(-1.0))


def test_an_asymmetric_proposal_is_respected():
    """The q-terms are not decorative: a backward-favourable move is taken."""
    # log pi(y)/pi(x) = 0, but q(x|y)/q(y|x) = e^1 > 1.
    assert mh_acceptance_ratio(0.0, 0.0, 1.0, 0.0) == 1.0
    # ... and the reverse costs acceptance.
    assert mh_acceptance_ratio(0.0, 0.0, 0.0, 1.0) == pytest.approx(math.exp(-1.0))


def test_detailed_balance_needs_the_proposal_terms_corrected():
    """The theorem's content: ``pi(x) q(y|x) a(x,y)`` is symmetric.

    The balance identity is on the JOINT move ``pi(x) q(y|x) a(x,y)``,
    not on ``pi(x) a(x,y)`` alone -- omitting ``q`` is what would make
    the check fail. With the proposal included the two directions agree,
    which is exactly what ``mh_stationary`` asserts.
    """
    log_px, log_py = -3.0, -1.0
    log_qxy, log_qyx = -0.5, -2.0
    a_xy = mh_acceptance_ratio(log_py, log_px, log_qyx, log_qxy)
    a_yx = mh_acceptance_ratio(log_px, log_py, log_qxy, log_qyx)
    joint_xy = math.exp(log_px + log_qxy) * a_xy
    joint_yx = math.exp(log_py + log_qyx) * a_yx
    assert joint_xy == pytest.approx(joint_yx)


def test_balancing_on_pi_alone_would_fail():
    """Why the proposal terms are in the signature: this is the check
    that a bare ``pi(x) a(x,y)`` does NOT satisfy."""
    log_px, log_py = -3.0, -1.0
    log_qxy, log_qyx = -0.5, -2.0
    a_xy = mh_acceptance_ratio(log_py, log_px, log_qyx, log_qxy)
    a_yx = mh_acceptance_ratio(log_px, log_py, log_qxy, log_qyx)
    assert math.exp(log_px) * a_xy != pytest.approx(math.exp(log_py) * a_yx)


def test_extreme_log_ratios_do_not_overflow():
    """The reason this is computed in log space at all."""
    assert mh_acceptance_ratio(1e300, -1e300, 0.0, 0.0) == 1.0
    assert mh_acceptance_ratio(-1e300, 1e300, 0.0, 0.0) == 0.0


def test_acceptance_always_lies_in_the_unit_interval():
    for lpn, lpo, lqb, lqf in (
        (0.0, 0.0, 0.0, 0.0), (-5.0, 5.0, -5.0, 5.0), (5.0, -5.0, 5.0, -5.0),
        (0.0, -0.001, 0.0, 0.0), (-1e-9, 0.0, 0.0, 0.0),
    ):
        a = mh_acceptance_ratio(lpn, lpo, lqb, lqf)
        assert 0.0 <= a <= 1.0


# --- pptMixing / ppt_tvContraction ----------------------------------------


def test_contraction_is_geometric_and_starts_at_one():
    assert tv_contraction_bound(0.5, 0.3, 0) == pytest.approx(1.0)
    assert tv_contraction_bound(0.5, 0.3, 1) == pytest.approx(0.7)
    assert tv_contraction_bound(0.5, 0.3, 2) == pytest.approx(0.49)
    assert tv_contraction_bound(0.5, 0.3, 3) == pytest.approx(0.343)


def test_contraction_is_monotone_in_steps_and_in_gap():
    vals = [tv_contraction_bound(0.5, 0.2, t) for t in range(5)]
    assert vals == sorted(vals, reverse=True)
    assert tv_contraction_bound(0.5, 0.5, 3) < tv_contraction_bound(0.5, 0.2, 3)


def test_a_very_slow_gap_still_contracts():
    """``gap -> 0`` is slow, not non-decreasing: no plateau at 1."""
    assert tv_contraction_bound(0.99, 0.001, 10_000) < 1e-3


@pytest.mark.parametrize("eps,gap", [(0.0, 0.3), (1.5, 0.3), (-0.5, 0.3),
                                      (0.5, 0.0), (0.5, 1.5), (0.5, -0.3)])
def test_out_of_range_contraction_inputs_are_refused(eps, gap):
    with pytest.raises(ValueError):
        tv_contraction_bound(eps, gap, 1)


def test_a_negative_step_count_is_refused():
    with pytest.raises(ValueError):
        tv_contraction_bound(0.5, 0.3, -1)


def test_total_variation_cannot_exceed_one():
    """A multiplier above 1 would be a vacuous bound."""
    for t in range(0, 50):
        assert tv_contraction_bound(0.5, 0.02, t) <= 1.0


# --- steps_to_mix ---------------------------------------------------------


def test_the_mix_time_is_the_exact_inverse_of_the_contraction():
    """``steps_to_mix`` and ``tv_contraction_bound`` are the same formula."""
    eps, gap, tol = 0.5, 0.25, 0.01
    n = steps_to_mix(eps, gap, tol)
    assert tv_contraction_bound(eps, gap, n) < tol
    assert tv_contraction_bound(eps, gap, n - 1) >= tol


def test_mixing_is_at_least_one_step_even_for_a_tiny_tolerance():
    assert steps_to_mix(0.5, 0.9, 1e-12) >= 1


def test_a_gap_of_one_reaches_the_tolerance_immediately():
    """``gap = 1`` means the bound is already 0 after a single step."""
    assert steps_to_mix(0.5, 1.0, 0.5) == 1


def test_a_slower_gap_needs_more_steps():
    assert steps_to_mix(0.5, 0.1, 0.01) > steps_to_mix(0.5, 0.4, 0.01)


@pytest.mark.parametrize("args", [(0.0, 0.3, 0.01), (0.5, 1.5, 0.01),
                                  (0.5, 0.3, 0.0), (0.5, 0.3, 1.0),
                                  (0.5, 0.3, 1.5)])
def test_out_of_range_mix_inputs_are_refused(args):
    with pytest.raises(ValueError):
        steps_to_mix(*args)


# --- truncation_bias -----------------------------------------------------


def test_keeping_everything_is_lossless():
    """Zero bias means the claim in the docstring does not apply."""
    p = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    keep = torch.ones(3, dtype=torch.bool)
    assert truncation_bias(p, keep) == 0.0


def test_dropping_everything_is_total_bias():
    p = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    keep = torch.zeros(3, dtype=torch.bool)
    assert truncation_bias(p, keep) == pytest.approx(1.0)


def test_the_bias_is_the_dropped_weight_share():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    keep = torch.tensor([True, False, False])
    assert truncation_bias(p, keep) == pytest.approx(0.5)


def test_bias_is_monotone_in_how_much_is_kept():
    p = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    vals = []
    for n in (0, 1, 2, 3, 4):
        keep = torch.zeros(4, dtype=torch.bool)
        keep[:n] = True
        vals.append(truncation_bias(p, keep))
    assert vals == sorted(vals, reverse=True)


def test_the_retained_target_is_a_different_distribution():
    """The docstring's operational point, stated as an identity.

    Renormalizing what survives does NOT give back the original: the
    retained distribution concentrates by exactly the dropped share.
    """
    p = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    keep = torch.tensor([True, False, False])
    before = float(ppt_target(p, 1.0)[0])
    after = float(ppt_target(p[keep], 1.0)[0])
    assert after == pytest.approx(1.0)
    assert after > before
    assert after - before == pytest.approx(truncation_bias(p, keep))


def test_bias_is_invariant_to_rescaling_the_input():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    keep = torch.tensor([True, False, True])
    assert truncation_bias(p, keep) == pytest.approx(truncation_bias(p * 91.0, keep))


def test_an_all_zero_input_is_refused_rather_than_reporting_full_bias():
    with pytest.raises(ValueError):
        truncation_bias(torch.zeros(3, dtype=torch.float64),
                        torch.ones(3, dtype=torch.bool))


# --- pptSuccess / pptSuccess_lower ---------------------------------------


def test_three_certain_events_give_a_certainty():
    assert ppt_success_lower(1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_one_impossible_event_drives_the_bound_to_zero():
    assert ppt_success_lower(0.0, 1.0, 1.0) == 0.0
    assert ppt_success_lower(1.0, 0.0, 1.0) == 0.0
    assert ppt_success_lower(1.0, 1.0, 0.0) == 0.0


def test_the_bound_is_exactly_the_union_bound():
    """``find + verify + budget - 2`` -- no slack, no tuning."""
    assert ppt_success_lower(0.9, 0.8, 0.7) == pytest.approx(0.4)
    assert ppt_success_lower(0.5, 0.5, 0.5) == 0.0


def test_the_bound_is_a_floor_under_the_true_joint():
    """Never claims more than the independent joint probability."""
    import random

    rng = random.Random(0)
    for _ in range(200):
        f, v, b = (rng.random() for _ in range(3))
        joint = f * v * b
        assert ppt_success_lower(f, v, b) <= joint + 1e-12


def test_a_floor_composes_with_R95_as_a_p0():
    """The consumer: a takeoff computed from this floor is a worst case."""
    floor = ppt_success_lower(0.95, 0.9, 0.95)
    assert 0.0 <= floor <= min(0.95, 0.9, 0.95)


def test_out_of_range_probabilities_are_clipped_not_exploded():
    """A caller passing nonsense gets a clipped answer, not a bad one."""
    assert ppt_success_lower(1.5, 1.5, 1.5) == pytest.approx(1.0)
    assert ppt_success_lower(-1.0, 0.5, 0.5) == 0.0


# --- the boundary the module insists on -----------------------------------


def test_the_power_target_is_not_a_tokenwise_temperature():
    """The docstring's negative claim, pinned so it cannot be quietly lost.

    Powering a JOINT sequence probability is a different operator from
    scaling logits by a temperature: the two disagree here, and the module
    claims only the former. If a future change made this test fail by
    turning ``ppt_target`` into ``softmax(logits / T)``, the claimed
    correspondence with ``pptTarget_isProbability`` would be false.
    """
    # A 2-token sequence with a genuinely different per-step structure.
    seq = torch.tensor([0.4, 0.6], dtype=torch.float64)
    joint = ppt_target(seq, 2.0)

    # The tokenwise-temperature reading of the same numbers: normalizing
    # ``log p`` scaled by ``1/alpha`` as if it were a logit vector.
    logits = torch.log(seq)
    tokenwise = torch.softmax(logits / 2.0, dim=0)

    assert not torch.allclose(joint, tokenwise, atol=1e-6)
    assert float(joint.sum()) == pytest.approx(1.0)
    assert float(tokenwise.sum()) == pytest.approx(1.0)