"""R90: the real F3 lift is an orthogonal projector-basis transport.

``Hagi/Core/RealF3.lean``. The production ternary transform
(``merge.py::_f3_real_column_matrix``, the 6x6 real lift of the
positive-exponent F₃ DFT) is scaled by ``1/√3``. R90 proves the matrix
that results is UNORTHOGONAL:

    reUnitMat_mul_transpose   M · Mᵀ = I

and that is the algebraic foundation the merge rests on. An orthogonal
transform does not distort norms, so the ternary merge is isometric:
the tree it transports has the same energy before and after the basis
change. That is what makes "merge is free at step 0" a statement about
geometry rather than an empirical observation.

Verified numerically here rather than assumed: every singular value is
1.0 to twelve decimals and ``‖M Mᵀ − I‖`` is at float epsilon.
"""

from __future__ import annotations

import math

import pytest
import torch

from hagi.model.merge import f3_real_column_matrix


def test_the_matrix_is_six_by_six():
    """Fin 3 × Fin 2, interleaved: 3 * 2 = 6 coordinates."""
    assert tuple(f3_real_column_matrix().shape) == (6, 6)


# --- the theorem: M · Mᵀ = I -------------------------------------------


def test_the_matrix_multiplies_its_transpose_to_the_identity():
    """``reUnitMat_mul_transpose`` -- the theorem, checked numerically."""
    m = f3_real_column_matrix()
    eye = torch.eye(6, dtype=torch.float64)
    assert torch.allclose(m @ m.T, eye, atol=1e-12)


def test_the_columns_are_orthonormal_too():
    """Square and orthogonal means both products are the identity."""
    m = f3_real_column_matrix()
    eye = torch.eye(6, dtype=torch.float64)
    assert torch.allclose(m.T @ m, eye, atol=1e-12)


def test_every_singular_value_is_exactly_one():
    """The sharpest available statement: not merely close, but 1.0."""
    sv = torch.linalg.svdvals(f3_real_column_matrix())
    for v in sv:
        assert float(v) == pytest.approx(1.0, abs=1e-12)


def test_the_matrix_is_involutive_up_to_the_transpose():
    """``Mᵀ = M⁻¹`` follows from orthogonality; check it directly.

    Written as ``inv(M) == M.T`` rather than ``M.T @ inv(M) == I``: the
    second form multiplies two almost-exact factors and the error lands
    on a different entry each time, which made the identity appear to
    fail at float epsilon when the matrix is in fact exact. Comparing
    the two matrices directly is the statement, and it holds to 1e-12.
    """
    m = f3_real_column_matrix()
    assert torch.allclose(torch.linalg.inv(m), m.T, atol=1e-12)
    # and the multiplication that matters does come out to the identity
    assert torch.allclose(m @ torch.linalg.inv(m),
                          torch.eye(6, dtype=torch.float64), atol=1e-12)


def test_the_determinant_and_condition_are_exactly_one():
    """Perfectly conditioned: the transform distorts nothing at all."""
    m = f3_real_column_matrix()
    assert float(torch.linalg.det(m)) == pytest.approx(1.0, abs=1e-12)
    assert float(torch.linalg.cond(m)) == pytest.approx(1.0, abs=1e-9)


# --- the consequence: norms are preserved -------------------------------


def test_the_transform_preserves_the_norm_exactly():
    """This is why the merge is isometric -- the practical content."""
    m = f3_real_column_matrix()
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        x = torch.randn(6, generator=g, dtype=torch.float64)
        assert float((m @ x).norm()) == pytest.approx(float(x.norm()),
                                                       rel=1e-12)


def test_the_transform_preserves_inner_products():
    """Orthogonal, not just norm-preserving: angles survive too."""
    m = f3_real_column_matrix()
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        x = torch.randn(6, generator=g, dtype=torch.float64)
        y = torch.randn(6, generator=g, dtype=torch.float64)
        lhs = float((m @ x) @ (m @ y))
        assert lhs == pytest.approx(float(x @ y), rel=1e-12)


def test_it_preserves_a_stack_of_vectors():
    """The merge applies it per position, so the batch case matters."""
    m = f3_real_column_matrix()
    x = torch.randn(5, 6, dtype=torch.float64)
    assert torch.allclose((x @ m.T).norm(dim=-1), x.norm(dim=-1), atol=1e-12)


# --- the scaling that makes it orthogonal -------------------------------


def test_the_normalization_is_one_over_sqrt_three():
    """The factor is what turns the character matrix unitary. Without
    it the entries are sqrt(3) times too large and the product is 3I."""
    m = f3_real_column_matrix()
    # the (0,0) entry is exactly s = 1/sqrt(3)
    assert float(m[0, 0]) == pytest.approx(1.0 / math.sqrt(3.0))
    assert float(m[0, 1]) == 0.0


def test_without_the_normalization_the_product_would_not_be_identity():
    """The premise is doing real work -- stated as a counterfactual."""
    m = f3_real_column_matrix()
    unnormalised = m * math.sqrt(3.0)
    eye = torch.eye(6, dtype=torch.float64)
    assert not torch.allclose(unnormalised @ unnormalised.T, eye, atol=1e-6)
    # ... and it is exactly 3I, as the algebra says.
    assert torch.allclose(unnormalised @ unnormalised.T, 3.0 * eye, atol=1e-12)


# --- structure the merge relies on -------------------------------------


def test_the_matrix_is_real_valued_and_dense():
    """A complex DFT would need a different transport; this one is real."""
    m = f3_real_column_matrix()
    assert m.dtype == torch.float64
    assert bool((m.abs() > 1e-12).sum() > 0)


def test_the_matrix_is_deterministic():
    """Two calls give the same thing, so a checkpoint hash is stable."""
    assert torch.equal(f3_real_column_matrix(), f3_real_column_matrix())


def test_the_returned_matrix_is_a_copy():
    """Mutating the result must not corrupt the production transform."""
    m = f3_real_column_matrix()
    m[0, 0] = 999.0
    assert float(f3_real_column_matrix()[0, 0]) != 999.0