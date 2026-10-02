"""Tests for the factorized merge and the unified budget (R101)."""

from __future__ import annotations

import pytest
import torch

from hagi.train.factorized_merge import (
    Decomposition,
    budget_within,
    core_is_worth_it,
    decompose,
    decomposition_gap,
    decomposition_gaps,
    factorized_error_bound,
    factorized_parameter_cost,
    factorization_saves_parameters,
    routed_eval,
    routed_eval_is_exact,
    unified_error_budget,
)


def rand_matrix(v: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(v, d, generator=g, dtype=torch.float64)


# --- routed_eval_exact ---------------------------------------------------


def test_an_exact_decomposition_reproduces_the_expert():
    """``routed_eval_exact``: on route ``i``, ``C + R_i == W_i``."""
    expert = rand_matrix(8, 6, 1)
    core = rand_matrix(8, 6, 2)
    d = decompose(expert, core)
    assert routed_eval_is_exact(expert, d)
    assert torch.allclose(routed_eval(d), expert)


def test_the_residual_is_exactly_what_is_left_over():
    expert = rand_matrix(8, 6, 3)
    core = rand_matrix(8, 6, 4)
    d = decompose(expert, core)
    assert torch.allclose(d.core + d.residual, expert)
    assert torch.allclose(d.residual, expert - core)


def test_an_approximate_residual_costs_exactly_its_gap():
    """``delta_i`` is measured, not asserted."""
    expert = rand_matrix(8, 6, 5)
    core = rand_matrix(8, 6, 6)
    err = torch.randn(8, 6, generator=torch.Generator().manual_seed(7),
                      dtype=torch.float64) * 0.1
    d = Decomposition(core=core, residual=expert - core + err)
    assert routed_eval_is_exact(expert, d) is False
    assert decomposition_gap(expert, d) == pytest.approx(float(err.norm()))


def test_a_zero_core_is_the_plain_expert():
    expert = rand_matrix(8, 6, 8)
    d = decompose(expert, torch.zeros_like(expert))
    assert torch.allclose(routed_eval(d), expert)


def test_a_shape_mismatch_is_refused():
    expert = rand_matrix(8, 6, 9)
    with pytest.raises(ValueError):
        decompose(expert, rand_matrix(8, 5, 10))


def test_a_decomposed_shape_mismatch_is_refused():
    with pytest.raises(ValueError):
        Decomposition(core=rand_matrix(4, 4, 11),
                      residual=rand_matrix(4, 5, 12))


# --- factorized_error_bound ---------------------------------------------


def test_exact_decompositions_cost_nothing():
    """Every ``delta_i == 0`` means the factorization is free of error."""
    assert factorized_error_bound([0.0, 0.0, 0.0]) == 0.0


def test_the_bound_is_the_mean_of_the_gaps():
    assert factorized_error_bound([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_one_bad_expert_costs_a_third():
    """A single failing expert is diluted, not ignored."""
    assert factorized_error_bound([0.0, 0.0, 3.0]) == pytest.approx(1.0)


def test_the_bound_is_the_mean_of_measured_gaps():
    """The gaps come from the experts, not from the residuals."""
    experts = [rand_matrix(8, 6, 20 + i) for i in range(3)]
    core = rand_matrix(8, 6, 30)
    gaps = decomposition_gaps(experts, core)
    assert factorized_error_bound(gaps) == pytest.approx(0.0, abs=1e-12)


def test_the_gap_is_not_the_residual_norm():
    """The distinction the module exists to keep: a large residual is not
    an error -- it is the expert's deviation from the core."""
    expert = rand_matrix(8, 6, 40)
    core = rand_matrix(8, 6, 41)
    d = decompose(expert, core)
    assert decomposition_gap(expert, d) == pytest.approx(0.0, abs=1e-12)
    assert float(d.residual.norm()) > 1.0


def test_a_single_expert_is_its_own_bound():
    assert factorized_error_bound([2.5]) == pytest.approx(2.5)


def test_an_empty_gap_list_is_refused():
    with pytest.raises(ValueError):
        factorized_error_bound([])


def test_a_negative_gap_is_refused():
    with pytest.raises(ValueError):
        factorized_error_bound([1.0, -0.5])


# --- factorized_budget ---------------------------------------------------


def test_a_low_rank_residual_saves_parameters():
    """``r << d``: the regime the theorem names."""
    cost = factorized_parameter_cost(v=1000, d=384, n_experts=3, rank=8)
    assert cost["factored"] < cost["full"]
    assert cost["saving"] == cost["full"] - cost["factored"]


def test_the_formula_is_the_stated_one():
    v, d, n, r = 100, 64, 3, 4
    cost = factorized_parameter_cost(v, d, n, r)
    assert cost["factored"] == v * d + n * r * (v + d)
    assert cost["full"] == n * v * d


def test_a_large_residual_stops_paying():
    """The regime check, not the assumption: at ``r = d`` it cannot save."""
    cost = factorized_parameter_cost(v=64, d=64, n_experts=3, rank=64)
    assert cost["saving"] <= 0
    assert not factorization_saves_parameters(64, 64, 3, 64)


def test_the_saving_grows_with_the_gap_between_the_experts():
    """More experts at the same rank means more to save."""
    cheap = factorized_parameter_cost(1000, 384, 3, 8)["saving"]
    roomy = factorized_parameter_cost(1000, 384, 16, 8)["saving"]
    assert roomy > cheap


def test_a_zero_rank_costs_exactly_one_core():
    v, d, n = 100, 64, 3
    cost = factorized_parameter_cost(v, d, n, 0)
    assert cost["factored"] == v * d
    assert cost["full"] == n * v * d


@pytest.mark.parametrize("args", [(0, 10, 3, 4), (10, 0, 3, 4), (10, 10, 0, 4)])
def test_non_positive_dimensions_are_refused(args):
    with pytest.raises(ValueError):
        factorized_parameter_cost(*args)


def test_a_negative_rank_is_refused():
    with pytest.raises(ValueError):
        factorized_parameter_cost(10, 10, 3, -1)


def test_the_go_no_go_is_a_parameter_comparison():
    """The open empirical item reduced to the one thing it decides."""
    assert core_is_worth_it(1000, 384, 3, 8, mean_gap=0.0)
    assert not core_is_worth_it(64, 64, 3, 64, mean_gap=0.0)


# --- unified_error_budget -----------------------------------------------


def test_the_total_is_the_sum_of_the_five_terms():
    b = unified_error_budget(shared=0.1, spectral=0.2, rank=0.3,
                              quant=0.4, routing=0.5)
    assert b["total"] == pytest.approx(1.5)


def test_an_empty_pipeline_costs_nothing():
    assert unified_error_budget()["total"] == 0.0


def test_every_term_appears_individually():
    b = unified_error_budget(shared=1.0, spectral=2.0, rank=3.0,
                              quant=4.0, routing=5.0)
    assert (b["shared"], b["spectral"], b["rank"], b["quant"],
            b["routing"]) == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_a_larger_budget_is_harder_to_exceed():
    small = unified_error_budget(quant=0.1)["total"]
    big = unified_error_budget(quant=1.0)["total"]
    assert budget_within(small, small)
    assert not budget_within(big, small)


@pytest.mark.parametrize("term", ["shared", "spectral", "rank", "quant",
                                   "routing"])
def test_a_negative_term_is_refused(term):
    with pytest.raises(ValueError):
        unified_error_budget(**{term: -1.0})


def test_the_five_terms_compose_with_the_spectral_module():
    """The budget is one object the two modules share, not two."""
    from hagi.train.spectral import three_stage_error_budget

    w = rand_matrix(16, 8, 40)
    from hagi.train.spectral import projector_from_scores

    p = projector_from_scores(torch.rand(8, dtype=torch.float64), 4)
    spec = three_stage_error_budget(w, p, eps_rank=0.05, eps_quant=0.05)
    merged = factorized_error_bound([0.01, 0.01, 0.01])
    budget = unified_error_budget(shared=merged, spectral=spec["tail"],
                                  rank=spec["rank"], quant=spec["quant"])
    assert budget["total"] == pytest.approx(
        merged + spec["bound"]
    )
    assert budget_within(budget["total"], budget["total"])