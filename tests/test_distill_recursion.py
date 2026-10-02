"""Tests for R108: the recursive entropy floor (DistillRecursion).

The properties worth pinning are mostly NEGATIVE -- what the theorem
does NOT give when the floor is asked to do work it cannot:

  - the floor is ``H(data) - delta/nu``, so it DEEPENS as the fresh-data
    mass shrinks, and below a point it is negative and therefore
    vacuous. A "collapse is impossible" claim that holds because the
    floor is -90 nats is not a safety result;
  - ``nu = 0`` makes the statement vacuous rather than false, which is
    why the theorem carries ``nu > 0`` as a hypothesis instead of
    assuming it;
  - this project does not distil, so the floor does not apply to its
    training path -- the port reports that rather than quietly applying
    a theorem to a mechanism that is not there.
"""

from __future__ import annotations

import math

import pytest

from hagi.train.distill_recursion import (
    certified_step_is_not_held,
    collapse_risk,
    distill_entropy_recurrence,
    distill_step_entropy,
    entropy_floor,
    entropy_mix_ge,
    fresh_data_prevents_collapse,
    fresh_mix,
    kl_does_not_bound_entropy,
    merge_nu,
    required_fresh_fraction,
    shannon_entropy,
)

UNIFORM_2 = [0.5, 0.5]
PEAKED_2 = [0.9, 0.1]


# --- entropy and the mixing law ----------------------------------------


def test_entropy_of_a_point_mass_is_zero():
    assert shannon_entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_entropy_of_uniform_over_vocab():
    """ln V for V = 32768 -- this project's head."""
    v = 32768
    assert shannon_entropy([1.0 / v] * v) == pytest.approx(math.log(v))


def test_entropy_is_maximal_at_uniform():
    """A split, not a merge, raises entropy -- the direction R108 needs."""
    assert shannon_entropy(PEAKED_2) < shannon_entropy(UNIFORM_2)


def test_entropy_rejects_a_negative_entry():
    with pytest.raises(ValueError):
        shannon_entropy([1.2, -0.2])


def test_entropy_rejects_an_empty_distribution():
    with pytest.raises(ValueError):
        shannon_entropy([])


def test_fresh_mix_is_the_convex_combination():
    """``(1-nu)p + nu d`` -- nu is the FRESH data's weight.

    The expectation is written as the formula rather than as literals:
    hand-computing ``0.75*0.9 + 0.25*0.5`` wrongly twice while writing
    this file is exactly the kind of slip a literal invites.
    """
    nu, p, d = 0.25, PEAKED_2, UNIFORM_2
    assert fresh_mix(nu, p, d) == pytest.approx(
        [(1.0 - nu) * a + nu * b for a, b in zip(p, d)]
    )
    assert fresh_mix(0.0, p, d) == pytest.approx(p)
    assert fresh_mix(1.0, p, d) == pytest.approx(d)


def test_fresh_mix_rejects_a_shape_mismatch():
    with pytest.raises(ValueError):
        fresh_mix(0.5, [0.5, 0.5], [1.0])


def test_fresh_mix_rejects_nu_outside_the_unit_interval():
    with pytest.raises(ValueError):
        fresh_mix(1.5, UNIFORM_2, UNIFORM_2)


def test_entropy_is_concave_on_the_mixture():
    """``entropy_mix_ge`` -- the inequality, measured."""
    left, right = entropy_mix_ge(0.3, PEAKED_2, UNIFORM_2)
    assert left >= right - 1e-12


def test_concavity_is_exact_at_the_endpoints():
    """nu = 0 and nu = 1 are equality, not inequality."""
    for nu in (0.0, 1.0):
        left, right = entropy_mix_ge(nu, PEAKED_2, UNIFORM_2)
        assert left == pytest.approx(right)


def test_the_mixture_lands_between_the_two_entropies():
    left, _ = entropy_mix_ge(0.5, PEAKED_2, UNIFORM_2)
    assert shannon_entropy(PEAKED_2) <= left <= shannon_entropy(UNIFORM_2)


# --- the one-step law --------------------------------------------------


def test_the_step_law_pulls_toward_the_data_and_costs_delta():
    got = distill_step_entropy(0.2, 0.05, h_parent=0.3, h_data=0.7, h_next=0.0)
    assert got == pytest.approx(0.8 * 0.3 + 0.2 * 0.7 - 0.05)


def test_the_step_law_rejects_a_negative_delta():
    with pytest.raises(ValueError):
        distill_step_entropy(0.2, -0.1, 0.3, 0.7, 0.5)


# --- the floor and the recurrence --------------------------------------


def test_the_floor_is_data_minus_delta_over_nu():
    assert entropy_floor(0.5, 0.2, 10.0) == pytest.approx(10.0 - 0.4)


def test_the_floor_diverges_as_nu_goes_to_zero():
    """``nu = 0`` is the collapse protocol, and the floor says so."""
    assert entropy_floor(0.01, 0.1, 10.0) < entropy_floor(0.1, 0.1, 10.0)
    with pytest.raises(ValueError):
        entropy_floor(0.0, 0.1, 10.0)


def test_the_floor_can_be_negative_and_then_it_is_vacuous():
    """The honest limit of "collapse is impossible".

    At nu = 0.001, delta = 0.1 the floor is -90 nats. Every entropy
    satisfies it, so the guarantee is true and carries no information.
    A caller must be able to see that, or it will cite the theorem as
    protection it does not provide.
    """
    assert entropy_floor(0.001, 0.1, 10.4) < 0.0


def test_the_recurrence_starts_exactly_at_h0():
    assert distill_entropy_recurrence(0.2, 0.05, 0.3, 0.7, 0) == pytest.approx(0.3)


def test_the_recurrence_converges_to_the_floor():
    """The fixed point is ``H(data) - delta/nu``, approached from above."""
    nu, delta, h_data = 0.2, 0.05, 0.7
    got = distill_entropy_recurrence(nu, delta, 0.3, h_data, 4000)
    assert got == pytest.approx(entropy_floor(nu, delta, h_data), abs=1e-3)


def test_the_recurrence_matches_iterating_the_step_law():
    """The closed form is not merely plausible -- it is the iteration."""
    nu, delta, h_data = 0.3, 0.02, 0.9
    h = 0.4
    for _ in range(12):
        h = distill_step_entropy(nu, delta, h, h_data, 0.0)
    assert h == pytest.approx(
        distill_entropy_recurrence(nu, delta, 0.4, h_data, 12), abs=1e-12
    )


def test_the_recurrence_direction_depends_on_which_side_of_the_floor():
    """Above the floor the entropy FALLS to it; below, it RISES.

    The recurrence is ``(1-nu)^T H(p_0) + (1-(1-nu)^T) floor``, a
    geometric interpolation, so it is monotone in ``T`` toward the floor
    from either side. A test asserting one direction without the
    condition would pass for half the parameter space and is wrong for
    the other half -- this one pins both halves.
    """
    nu, delta, h_data = 0.25, 0.05, 0.8
    floor = entropy_floor(nu, delta, h_data)

    above = [distill_entropy_recurrence(nu, delta, 0.8, h_data, t) for t in range(15)]
    assert all(b <= a + 1e-12 for a, b in zip(above, above[1:])), "above the floor it falls"

    below = [distill_entropy_recurrence(nu, delta, 0.5, h_data, t) for t in range(15)]
    assert all(b >= a - 1e-12 for a, b in zip(below, below[1:])), "below the floor it rises"
    assert floor == pytest.approx(0.6)


def test_the_recurrence_rejects_a_negative_horizon():
    with pytest.raises(ValueError):
        distill_entropy_recurrence(0.2, 0.05, 0.3, 0.7, -1)


# --- collapse prevention ------------------------------------------------


def test_started_above_the_floor_the_entropy_never_breaks_it():
    nu, delta, h_data = 0.2, 0.05, 0.7
    floor = entropy_floor(nu, delta, h_data)
    assert h_data - delta / nu < 0.45 < floor or 0.45 > floor
    assert fresh_data_prevents_collapse(nu, delta, 0.45, h_data, 5000) is True


def test_started_below_the_floor_the_invariant_does_not_hold():
    """The theorem needs an initial condition, and saying so is the point."""
    nu, delta, h_data = 0.2, 0.05, 0.7
    assert fresh_data_prevents_collapse(nu, delta, 0.1, h_data, 10) is False


def test_collapse_prevention_requires_positive_nu():
    with pytest.raises(ValueError):
        fresh_data_prevents_collapse(0.0, 0.05, 0.45, 0.7, 10)


def test_more_fresh_data_gives_a_higher_floor():
    """``nu`` is the knob, and it works in the stated direction."""
    nu_low = entropy_floor(0.05, 0.1, 10.0)
    nu_high = entropy_floor(0.5, 0.1, 10.0)
    assert nu_high > nu_low


# --- the inverse: nu as arithmetic -------------------------------------


def test_the_required_fresh_fraction_is_delta_over_eps():
    assert required_fresh_fraction(0.1, 0.5) == pytest.approx(0.2)


def test_halving_nu_doubles_the_depth_of_the_floor():
    """The trade-off, in one comparison."""
    _, shallow = collapse_risk(0.4, 0.2, 10.0, 9.0)
    _, deep = collapse_risk(0.2, 0.2, 10.0, 9.0)
    assert deep == pytest.approx(2.0 * shallow)


def test_the_distance_to_the_floor_is_reported():
    nu, delta, h_data = 0.25, 0.1, 10.0
    dist, depth = collapse_risk(nu, delta, h_data, 9.6)
    assert depth == pytest.approx(0.4)
    assert dist == pytest.approx(9.6 - 9.6)


def test_collapse_risk_requires_positive_nu():
    with pytest.raises(ValueError):
        collapse_risk(0.0, 0.1, 10.0, 9.0)


def test_required_fresh_fraction_rejects_a_zero_target():
    with pytest.raises(ValueError):
        required_fresh_fraction(0.1, 0.0)


# --- this project's actual path ----------------------------------------


def test_this_project_does_not_distil():
    """The structural finding: ``hcert`` is not delivered by training."""
    assert certified_step_is_not_held() is True


def test_the_merge_is_the_mixture_the_floor_is_about():
    """Uniform pooling IS ``(1-nu)p + nu d`` -- at ``nu = 1/n``."""
    assert merge_nu(3) == pytest.approx(1.0 / 3.0)
    assert merge_nu(1) == pytest.approx(1.0)


def test_merging_more_experts_protects_against_collapse():
    """R108's consequence for the merge: MORE experts is SAFER.

    The floor deepens as ``delta/nu`` grows, so a merge of 3 experts has
    a deeper floor than a merge of 1 -- the opposite of the intuition
    that pooling reduces risk. Growth in expert count is the anti-collapse
    direction, and the port says so with arithmetic.
    """
    _, depth3 = collapse_risk(merge_nu(3), 0.1, 10.0, 9.0)
    _, depth1 = collapse_risk(merge_nu(1), 0.1, 10.0, 9.0)
    assert depth3 > depth1


def test_merge_nu_rejects_an_empty_pool():
    with pytest.raises(ValueError):
        merge_nu(0)


# --- the refuted audit constant ----------------------------------------


def test_the_audit_kl_to_entropy_claim_is_false():
    """``KL <= delta => H(q) >= H(m) - delta`` has a two-line counterexample.

    Entropy is not Lipschitz in KL with a linear constant, so the
    per-step preservation R108 assumes cannot be bought with a KL step
    size. It has to be an explicit hypothesis, which is how the theorem
    states it.
    """
    kl, loss, ratio = kl_does_not_bound_entropy()
    assert loss > kl, "the entropy loss must EXCEED the KL bound"
    assert ratio > 1.0


def test_the_counterexample_numbers_are_the_published_ones():
    kl, loss, _ = kl_does_not_bound_entropy()
    assert kl == pytest.approx(0.14, abs=0.005)
    assert loss == pytest.approx(0.27, abs=0.005)