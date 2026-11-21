"""C428/C472: closed-form bit-budget allocation."""
import math
from hagi.budget import weighted_exp_allocation, fisher_tolerances


def test_allocation_satisfies_budget():
    c = [1.0, 2.0, 0.5]
    kappa = [1.0, 2.0, 4.0]
    w = [1.0, 1.0, 1.0]
    B = 3.0
    alloc = weighted_exp_allocation(c, kappa, w, B)
    got = sum(wi * ri for wi, ri in zip(w, alloc))
    assert abs(got - B) < 1e-9   # budget constraint exact


def test_fisher_alloc_mono():
    # larger curvature -> smaller tolerance (more bits), MONOTONE
    curv = [1.0, 5.0, 20.0]
    tol = fisher_tolerances(curv, B=3.0)
    assert tol[0] > tol[1] > tol[2]
    assert all(t > 0 for t in tol)


def test_isotropic_uniform():
    tol = fisher_tolerances([2.0, 2.0, 2.0], B=3.0)
    assert abs(max(tol) - min(tol)) < 1e-12  # no gain in isotropic case
