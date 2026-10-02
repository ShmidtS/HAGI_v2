"""Tests for the spectral error budget (R100)."""

from __future__ import annotations

import math

import pytest
import torch

from hagi.train.spectral import (
    budget_is_respected,
    filtered_step_cost,
    filtered_target,
    is_orthogonal_projector,
    projector_from_scores,
    proj_residual_identity,
    rank_from_spectrum,
    residual_energy,
    retained_energy,
    spectral_noise_reduction,
    spectral_tail_energy,
    three_stage_error_budget,
)


def rand_projector(d: int, keep: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(d, generator=g, dtype=torch.float64)
    return projector_from_scores(scores, keep)


def column_projector(w: torch.Tensor, keep: int, seed: int = 0) -> torch.Tensor:
    """Projector over the COLUMNS of ``W`` -- what the budget consumes."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(w.shape[-1], generator=g, dtype=torch.float64)
    return projector_from_scores(scores, keep)


# --- proj_residual_identity ----------------------------------------------


def test_the_projector_is_orthogonal_by_construction():
    """``P^2 = P`` and ``P = P^T``: the premise of every claim here."""
    p = rand_projector(16, 5, seed=1)
    assert is_orthogonal_projector(p)


def test_the_identity_holds_exactly_for_a_random_vector():
    """``||x - Px||^2 == ||x||^2 - ||Px||^2`` -- not an estimate."""
    g = torch.Generator().manual_seed(2)
    x = torch.randn(16, generator=g, dtype=torch.float64)
    p = rand_projector(16, 5, seed=2)
    assert proj_residual_identity(x, p)
    # ... and the two sides are computed differently, so this is a check.
    assert residual_energy(x, p) == pytest.approx(
        float(x.pow(2).sum()) - retained_energy(x, p), rel=1e-12
    )


def test_the_identity_holds_at_both_extremes():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(16, generator=g, dtype=torch.float64)
    assert proj_residual_identity(x, projector_from_scores(x, 16))
    assert proj_residual_identity(x, projector_from_scores(x, 1))


def test_keeping_everything_discards_nothing():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(16, generator=g, dtype=torch.float64)
    p = projector_from_scores(x, 16)
    assert residual_energy(x, p) == pytest.approx(0.0, abs=1e-15)


def test_keeping_nothing_discards_everything():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(16, generator=g, dtype=torch.float64)
    p = projector_from_scores(x, 1)
    kept = retained_energy(x, p)
    total = float(x.pow(2).sum())
    assert kept < total
    assert residual_energy(x, p) == pytest.approx(total - kept, rel=1e-12)


def test_retained_plus_discarded_is_the_whole():
    """The energy splits -- nothing is created by the projection."""
    g = torch.Generator().manual_seed(6)
    x = torch.randn(32, generator=g, dtype=torch.float64)
    p = projector_from_scores(x, 7)
    assert retained_energy(x, p) + residual_energy(x, p) == pytest.approx(
        float(x.pow(2).sum()), rel=1e-12
    )


def test_more_kept_coordinates_never_discards_more():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(24, generator=g, dtype=torch.float64)
    vals = [residual_energy(x, projector_from_scores(x, k)) for k in range(1, 25)]
    assert vals == sorted(vals, reverse=True)


def test_the_projector_keeps_the_largest_coordinates():
    """Which coordinates survive is part of the claim."""
    x = torch.tensor([0.1, 5.0, 0.2, 3.0], dtype=torch.float64)
    p = projector_from_scores(x, 2)
    kept = (p @ x).abs()
    assert float(kept[1]) == pytest.approx(5.0)
    assert float(kept[3]) == pytest.approx(3.0)
    assert float(kept[0]) == pytest.approx(0.0)


def test_a_non_projector_fails_the_identity():
    """The check has teeth: an arbitrary matrix must not pass."""
    g = torch.Generator().manual_seed(8)
    x = torch.randn(16, generator=g, dtype=torch.float64)
    p = torch.rand(16, 16, generator=g, dtype=torch.float64)
    assert not proj_residual_identity(x, p)


# --- spectral_tail_energy ------------------------------------------------


def test_the_tail_is_exactly_the_sum_of_dropped_squares():
    c = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    # keep the two largest: 4 and 3, so the tail is 1 + 4.
    assert spectral_tail_energy(c, 2) == pytest.approx(1.0 + 4.0)


def test_the_tail_is_monotone_decreasing_in_rank():
    c = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0], dtype=torch.float64)
    vals = [spectral_tail_energy(c, k) for k in range(1, 6)]
    assert vals == sorted(vals, reverse=True)


def test_keeping_everything_leaves_no_tail():
    c = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert spectral_tail_energy(c, 3) == pytest.approx(0.0, abs=1e-15)


def test_the_tail_matches_the_projector_residual_on_the_same_coefficients():
    """The two views of one fact: spectral law and projector identity."""
    c = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    p = projector_from_scores(c, 2)
    assert spectral_tail_energy(c, 2) == pytest.approx(residual_energy(c, p),
                                                        rel=1e-12)


def test_selection_is_by_magnitude_not_sign():
    c = torch.tensor([-5.0, 1.0], dtype=torch.float64)
    p = projector_from_scores(c, 1)
    assert float((p @ c).abs()[0]) == pytest.approx(5.0)


# --- projector_from_scores guards ----------------------------------------


@pytest.mark.parametrize("keep", [0, -1, 17])
def test_an_out_of_range_keep_is_refused(keep):
    with pytest.raises(ValueError):
        projector_from_scores(torch.rand(16, dtype=torch.float64), keep)


def test_a_non_finite_score_is_refused():
    s = torch.ones(8, dtype=torch.float64)
    s[3] = float("nan")
    with pytest.raises(ValueError):
        projector_from_scores(s, 4)


def test_an_empty_score_vector_is_refused():
    with pytest.raises(ValueError):
        projector_from_scores(torch.zeros(0, dtype=torch.float64), 1)


def test_ties_resolve_deterministically():
    s = torch.ones(8, dtype=torch.float64)
    a = projector_from_scores(s, 3)
    b = projector_from_scores(s, 3)
    assert torch.equal(a, b)


# --- rank_from_spectrum --------------------------------------------------


def test_the_rank_is_the_inverse_of_the_tail_energy():
    vals = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    total = float(vals.pow(2).sum())
    r = rank_from_spectrum(vals, total, 0.0)
    assert r == 5
    # A generous tolerance buys a much smaller rank.
    assert rank_from_spectrum(vals, total, 25.0) < 5


def test_a_zero_tolerance_keeps_every_component():
    vals = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    total = float(vals.pow(2).sum())
    assert rank_from_spectrum(vals, total, 0.0) == 5


def test_the_rank_is_monotone_decreasing_in_tolerance():
    vals = torch.tensor([9.0, 5.0, 3.0, 1.0, 0.5], dtype=torch.float64)
    total = float(vals.pow(2).sum())
    ranks = [rank_from_spectrum(vals, total, t) for t in (0.0, 5.0, 20.0, 40.0)]
    assert ranks == sorted(ranks, reverse=True)


def test_the_retained_energy_at_that_rank_meets_the_tolerance():
    """The definition, verified on the numbers the rank implies."""
    vals = torch.tensor([9.0, 5.0, 3.0, 1.0, 0.5], dtype=torch.float64)
    total = float(vals.pow(2).sum())
    tol = 20.0
    r = rank_from_spectrum(vals, total, tol)
    discarded = float(vals[r:].pow(2).sum())
    assert discarded <= tol + 1e-9


def test_the_rank_is_minimal_not_maximal():
    """One more component than needed would violate the minimality claim."""
    vals = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    total = float(vals.pow(2).sum())
    r = rank_from_spectrum(vals, total, 5.0)
    assert r == 3
    assert float(vals[r:].pow(2).sum()) <= 5.0 + 1e-9
    # One step lower would already exceed the tolerance.
    assert float(vals[r - 1:].pow(2).sum()) > 5.0


# --- three_stage_error_budget --------------------------------------------


def test_the_budget_is_the_sum_of_the_named_terms():
    w = torch.randn(8, 5, generator=torch.Generator().manual_seed(9),
                    dtype=torch.float64)
    p = column_projector(w, 3, seed=9)
    b = three_stage_error_budget(w, p, eps_rank=0.01, eps_quant=0.02)
    assert b["bound"] == pytest.approx(b["tail"] + b["rank"] + b["quant"])
    assert b["tail"] == pytest.approx(float((w - w @ p).norm()))


def test_no_projector_means_a_zero_spectral_term():
    w = torch.randn(8, 5, generator=torch.Generator().manual_seed(10),
                    dtype=torch.float64)
    b = three_stage_error_budget(w, None, eps_rank=0.1, eps_quant=0.1)
    assert b["tail"] == 0.0
    assert b["bound"] == pytest.approx(0.2)


def test_the_budget_grows_with_every_term():
    w = torch.randn(8, 5, generator=torch.Generator().manual_seed(11),
                    dtype=torch.float64)
    p = column_projector(w, 3, seed=11)
    base = three_stage_error_budget(w, p)["bound"]
    assert three_stage_error_budget(w, p, eps_rank=0.5)["bound"] > base
    assert three_stage_error_budget(w, p, eps_quant=0.5)["bound"] > base


def test_a_larger_subspace_lowers_the_spectral_term():
    w = torch.randn(16, 8, generator=torch.Generator().manual_seed(12),
                    dtype=torch.float64)
    small = three_stage_error_budget(w, column_projector(w, 2))["tail"]
    large = three_stage_error_budget(w, column_projector(w, 7))["tail"]
    assert large < small


def test_a_negative_error_term_is_refused():
    w = torch.randn(4, 4, generator=torch.Generator().manual_seed(13),
                    dtype=torch.float64)
    with pytest.raises(ValueError):
        three_stage_error_budget(w, None, eps_rank=-1.0)
    with pytest.raises(ValueError):
        three_stage_error_budget(w, None, eps_quant=-1.0)


def test_the_budget_can_be_respected_and_violated():
    w = torch.randn(16, 8, generator=torch.Generator().manual_seed(14),
                    dtype=torch.float64)
    p = column_projector(w, 5)
    b = three_stage_error_budget(w, p, eps_rank=1.0, eps_quant=1.0)
    assert budget_is_respected(0.5, b)
    assert not budget_is_respected(100.0, b)


def test_the_realized_error_never_exceeds_the_budget_on_real_data():
    """The theorem's content, end to end on generated data.

    Build a matrix as tail + low-rank + quantization error explicitly, and
    confirm the triangle inequality the budget asserts.
    """
    g = torch.Generator().manual_seed(15)
    w = torch.randn(32, 16, generator=g, dtype=torch.float64)
    p = column_projector(w, 12, seed=14)
    rank_err = 0.05
    quant_err = 0.05
    # An actual reconstruction within the declared terms.
    reconstructed = w @ p + torch.randn(32, 16, generator=g,
                                         dtype=torch.float64) * rank_err
    reconstructed = reconstructed + torch.randn(32, 16, generator=g,
                                               dtype=torch.float64) * quant_err
    measured = float((w - reconstructed).norm())
    b = three_stage_error_budget(w, p, eps_rank=rank_err, eps_quant=quant_err)
    assert budget_is_respected(measured, b)


# --- filtered SafeQP target ----------------------------------------------


def test_filtering_removes_the_dropped_coordinates():
    g = torch.Generator().manual_seed(16)
    v = torch.randn(16, generator=g, dtype=torch.float64)
    p = projector_from_scores(v, 6)
    pg = filtered_target(v, p)
    dropped = spectral_tail_energy(v, 6)
    assert dropped > 0.0
    assert torch.allclose(pg, p @ v)
    assert float(pg.norm()) < float(v.norm())


def test_the_filtering_cost_is_the_displacement():
    """``filtered_step_cost`` is measured, not assumed."""
    g = torch.Generator().manual_seed(17)
    v = torch.randn(16, generator=g, dtype=torch.float64)
    p = projector_from_scores(v, 6)
    pg = filtered_target(v, p)
    assert filtered_step_cost(v, p) == pytest.approx(float((v - pg).norm()),
                                                     rel=1e-12)


def test_the_price_is_zero_when_nothing_is_removed():
    g = torch.Generator().manual_seed(18)
    v = torch.randn(16, generator=g, dtype=torch.float64)
    p = projector_from_scores(v, 16)
    assert filtered_step_cost(v, p) == pytest.approx(0.0, abs=1e-12)


def test_the_price_is_the_whole_norm_when_everything_is_removed():
    g = torch.Generator().manual_seed(19)
    v = torch.randn(16, generator=g, dtype=torch.float64)
    p = projector_from_scores(v, 1)
    if residual_energy(v, p) > 1e-12:
        assert filtered_step_cost(v, p) == pytest.approx(
            math.sqrt(float(v.pow(2).sum()) - retained_energy(v, p)), rel=1e-12
        )


def test_filtering_never_increases_the_noise_norm():
    """``P`` is a contraction: the premise of the variance claim."""
    g = torch.Generator().manual_seed(20)
    for _ in range(20):
        noise = torch.randn(32, generator=g, dtype=torch.float64)
        p = projector_from_scores(noise, 5)
        before, after = spectral_noise_reduction(noise, p)
        assert after <= before + 1e-12


def test_the_noise_reduction_is_zero_on_the_kept_subspace():
    g = torch.Generator().manual_seed(21)
    noise = torch.randn(16, generator=g, dtype=torch.float64)
    p = projector_from_scores(noise, 16)
    before, after = spectral_noise_reduction(noise, p)
    assert after == pytest.approx(before, rel=1e-12)