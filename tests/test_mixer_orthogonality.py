"""Runtime orthonormality certificates for the merge mixers (RealF3 bridge).

The Lean core (Hagi.Core.DFT3 / Hadamard / RealF3) proves the ABSTRACT
operators unitary/orthogonal. This test pins the PRODUCTION matrices to
that claim numerically — the minimal runtime side of the implementation
bridge: if a refactor silently breaks orthonormality, the step-0 exact
merge identity dies with it.

The audit note on orientation: Lean's ``reCharMat`` (character-negative
convention) equals the production COLUMN matrix transposed. Both
conventions are correct transforms; what must hold in either is
R^T R = I to machine precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from hagi.model.merge import (  # noqa: E402
    _hadamard_matrix,
    f3_real_column_matrix,
    f3_real_row_matrix,
)


def test_f3_column_matrix_orthonormal():
    c = f3_real_column_matrix().double()
    gram = c.T @ c
    assert torch.allclose(gram, torch.eye(6, dtype=torch.float64), atol=1e-12), gram
    # equal-modulus property (|F_ij| = 1/sqrt(3)) lifted to the real basis:
    # every row has the same norm 1
    norms = c.norm(dim=1)
    assert torch.allclose(norms, torch.ones(6, dtype=torch.float64), atol=1e-12)


def test_f3_row_matrix_is_column_transpose():
    c = f3_real_column_matrix()
    r = f3_real_row_matrix()
    assert torch.equal(r, c.transpose(0, 1).contiguous())
    # the row-action matrix is orthonormal too (R^T R = I follows from C)
    rd = r.double()
    assert torch.allclose(rd.T @ rd, torch.eye(6, dtype=torch.float64), atol=1e-12)


def test_hadamard_orthogonal_up_to_scale():
    for n in (2, 4, 8, 16, 64):
        h = _hadamard_matrix(n).double()
        gram = h @ h.T
        assert torch.allclose(gram, n * torch.eye(n, dtype=torch.float64), atol=1e-12)
        q = h / (n**0.5)
        assert torch.allclose(q @ q.T, torch.eye(n, dtype=torch.float64), atol=1e-12)
    try:
        _hadamard_matrix(3)
        raised = False
    except ValueError:
        raised = True
    assert raised  # non-power-of-two refuses loudly, not silently
