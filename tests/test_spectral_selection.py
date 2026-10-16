"""Property tests for the SpectralProjector/RankGrowth/SparseStep0 port
in src/hagi/train/spectral.py — mirror the Lean theorems."""

import math

import pytest
import torch

from hagi.train.spectral import (
    is_orthogonal_projector, orthogonal_growth_ok, rank_orthogonal_growth,
    sparse_step0_ok, spectral_projector, spectral_tail_energy_split,
)


def _orthonormal(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, k, generator=g).double()
    q, _ = torch.linalg.qr(a)
    return q


def test_projector_idempotent_selfadjoint():
    u = _orthonormal(6, 4, seed=1)
    p = spectral_projector(u, [0, 2])
    assert is_orthogonal_projector(p)
    # projecting twice == once (spectralProj_idempotent)
    assert torch.allclose(p @ p, p, atol=1e-12)
    # acts as identity on kept directions, zero on dropped
    kept = u[:, 0]
    assert torch.allclose(p @ kept, kept, atol=1e-12)
    dropped = u[:, 1]
    assert torch.allclose(p @ dropped, torch.zeros_like(dropped), atol=1e-12)


def test_tail_energy_identity_exact():
    u = _orthonormal(5, 5, seed=2)
    x = torch.randn(5, generator=torch.Generator().manual_seed(3)).double()
    coeffs = u.T @ x
    keep = torch.tensor([True, True, False, False, True])
    kept, tail = spectral_tail_energy_split(coeffs, keep)
    # exact identity: residual norm² == tail (spectral_tail_energy)
    p = spectral_projector(u, [0, 1, 4])
    resid = float((x - p @ x).pow(2).sum())
    assert math.isclose(tail, resid, rel_tol=1e-10, abs_tol=1e-12)
    # kept + tail == total energy
    assert math.isclose(kept + tail, float(x.pow(2).sum()), rel_tol=1e-10)


def test_rank_orthogonal_growth():
    g = torch.Generator().manual_seed(4)
    m = torch.randn(4, 6, generator=g).double()
    # complete the row space, then take the orthogonal complement rows
    _, _, vt = torch.linalg.svd(m, full_matrices=False)
    null_basis = torch.linalg.qr(vt.T, mode="complete")[0][:, vt.shape[0]:]
    new_rows = null_basis.T[:2]  # orthogonal to row(M), independent
    assert orthogonal_growth_ok(m, new_rows)
    rank_m = int(torch.linalg.matrix_rank(m))
    assert rank_orthogonal_growth(m, new_rows) == rank_m + 2
    # non-orthogonal new rows fail the certificate
    bad = torch.randn(2, 6, generator=g).double()
    assert not orthogonal_growth_ok(m, bad)


def test_sparse_step0_gate_boundary():
    coeffs = torch.tensor([1.0, 1.0, 1.0, 0.2])
    keep = torch.tensor([True, True, True, False])
    # tail = 0.04; tol=0.2 → tol²=0.04 — strict < : boundary is False
    assert sparse_step0_ok(coeffs, keep, tol=0.3)
    assert not sparse_step0_ok(coeffs, keep, tol=0.2)
    assert sparse_step0_ok(coeffs, keep, tol=0.19) is False  # 0.04 > 0.0361


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
