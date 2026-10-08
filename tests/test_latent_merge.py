"""Numerical tests for the fixed-width latent merge (theory SSOT: the
LatentAlign / ResidualSplit / FactorizedBPW Lean modules).

Small tensors, CPU, float64 tolerance 1e-5 — every claim below is one of
the Lean theorems stated as running code.
"""

from __future__ import annotations

import math

import pytest
import torch

from hagi.model.latent_merge import (
    bpw_lt_one_rank_bound,
    factorized_bpw,
    factorize_delta,
    latent_align,
    merge_fixed_width,
    reconstruct_expert,
    root_contrast_merge,
    spectral_compress,
)

ATOL = 1e-5


def _orthonormal(d: int, r: int, gen: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(d, r, generator=gen, dtype=torch.float64))
    return q


def _random_orthogonal(r: int, gen: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(r, r, generator=gen, dtype=torch.float64))
    return q


def test_rotated_factors_same_matrix() -> None:
    # LatentAlign.rotated_factors_same_matrix: (U Q)(V Q)^T == U V^T.
    gen = torch.Generator().manual_seed(0)
    U = torch.randn(32, 8, generator=gen, dtype=torch.float64)
    V = torch.randn(32, 8, generator=gen, dtype=torch.float64)
    Q = _random_orthogonal(8, gen)
    assert torch.allclose((U @ Q) @ (V @ Q).T, U @ V.T, atol=ATOL)


def test_sign_flip_cancellation_recovered() -> None:
    # THE R242 theorem in code: naive_latent_merge_kills_flipped vs align.
    gen = torch.Generator().manual_seed(1)
    d, r = 32, 8
    U = _orthonormal(d, r, gen)
    V = torch.randn(d, r, generator=gen, dtype=torch.float64)
    W = U @ V.T
    signs = torch.tensor(
        [1.0 if i % 2 else -1.0 for i in range(r)], dtype=torch.float64
    )
    factors = [(U, V), (U * signs, V * signs)]  # same matrix, flipped latents
    aligned = latent_align(factors)
    merged = root_contrast_merge(aligned, torch.zeros_like(W))
    err_aligned = float((merged["root_matrix"] - W).norm())
    naive_U = 0.5 * (factors[0][0] + factors[1][0])
    naive_V = 0.5 * (factors[0][1] + factors[1][1])
    err_naive = float((naive_U @ naive_V.T - W).norm())
    scale = float(W.norm())
    assert err_aligned < 1e-5 * scale
    assert err_naive > 0.25 * scale  # naive merge loses real energy


def test_procrustes_reduces_distance() -> None:
    # After alignment ||U_i' - U_ref|| <= ||U_i - U_ref|| (Procrustes optimum).
    gen = torch.Generator().manual_seed(2)
    d, r = 24, 6
    U_ref = _orthonormal(d, r, gen)
    U_i = U_ref @ _random_orthogonal(r, gen)
    V_i = torch.randn(d, r, generator=gen, dtype=torch.float64)
    before = float((U_i - U_ref).norm())
    aligned = latent_align([(U_ref, torch.randn(d, r, generator=gen)), (U_i, V_i)])
    after = float((aligned[1][0] - U_ref).norm())
    assert after <= before + ATOL


def test_latent_align_preserves_products() -> None:
    # Alignment is free: every expert's U' V'^T == U V^T exactly.
    gen = torch.Generator().manual_seed(3)
    factors = []
    for _ in range(4):
        U = torch.randn(16, 5, generator=gen, dtype=torch.float64)
        V = torch.randn(16, 5, generator=gen, dtype=torch.float64)
        factors.append((U, V))
    aligned = latent_align(factors, ref=2)
    for (U, V), (Ua, Va) in zip(factors, aligned):
        assert torch.allclose(Ua @ Va.T, U @ V.T, atol=ATOL)


def test_latent_align_handles_rank_mismatch() -> None:
    # Ranks differ: align in the shared min-rank subspace, pad, still exact.
    gen = torch.Generator().manual_seed(4)
    U0 = torch.randn(16, 6, generator=gen, dtype=torch.float64)
    V0 = torch.randn(16, 6, generator=gen, dtype=torch.float64)
    U1 = torch.randn(16, 3, generator=gen, dtype=torch.float64)
    V1 = torch.randn(16, 3, generator=gen, dtype=torch.float64)
    aligned = latent_align([(U0, V0), (U1, V1)])
    assert aligned[0][0].shape[1] == aligned[1][0].shape[1] == 6
    assert torch.allclose(aligned[0][0] @ aligned[0][1].T, U0 @ V0.T, atol=ATOL)
    assert torch.allclose(aligned[1][0] @ aligned[1][1].T, U1 @ V1.T, atol=ATOL)


def test_factorize_delta_energy_rank() -> None:
    gen = torch.Generator().manual_seed(5)
    d, r = 32, 6
    U = torch.randn(d, r, generator=gen, dtype=torch.float64)
    V = torch.randn(d, r, generator=gen, dtype=torch.float64)
    W_shared = torch.randn(d, d, generator=gen, dtype=torch.float64)
    W = W_shared + 0.1 * (U @ V.T)
    Uf, Vf, svals = factorize_delta(W, W_shared, energy=0.99)
    assert Uf.shape[1] == Vf.shape[1] == svals.shape[0] <= 6
    approx = Uf @ Vf.T
    total = float((0.1 * U @ V.T).pow(2).sum())
    kept = float(approx.pow(2).sum())
    assert kept >= 0.99 * total - 1e-6
    assert torch.allclose(
        Uf.double() @ Vf.double().T, approx.double(), atol=ATOL
    )


def test_root_contrast_reconstructs_experts() -> None:
    # ResidualSplit: root + contrast_i == W_shared + U'_i V'_i^T (exact),
    # which equals expert i within factorization error.
    gen = torch.Generator().manual_seed(6)
    d, r = 24, 8
    W_shared = torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.1
    experts = []
    factors = []
    for _ in range(3):
        U = _orthonormal(d, r, gen)
        V = torch.randn(d, r, generator=gen, dtype=torch.float64)
        W = W_shared + U @ V.T
        Uf, Vf, _ = factorize_delta(W, W_shared, energy=1.0)
        experts.append(W)
        factors.append((Uf, Vf))
    aligned = latent_align(factors)
    merged = root_contrast_merge(aligned, W_shared)
    for i, W in enumerate(experts):
        recon = reconstruct_expert(merged, i)
        assert float((recon - W).norm()) < 1e-5 * float(W.norm())


def test_spectral_compress_energy_and_rank() -> None:
    gen = torch.Generator().manual_seed(7)
    d, r = 32, 10
    # Decaying spectrum: each component carries 1/2 of the previous one.
    U = _orthonormal(d, r, gen)
    V = torch.randn(d, r, generator=gen, dtype=torch.float64)
    scales = 2.0 ** (-torch.arange(r, dtype=torch.float64))
    root_factors = (U * scales, V)
    full = root_factors[0] @ root_factors[1].T
    total = float(full.pow(2).sum())
    for energy in (0.5, 0.9, 1.0):
        Uc, Vc = spectral_compress(root_factors, energy=energy)
        rank = int(Uc.shape[1])
        assert 1 <= rank <= r
        kept = float((Uc @ Vc.T).pow(2).sum())
        assert kept >= energy * total - 1e-6


def test_bpw_anchor_and_bound() -> None:
    # FactorizedBPW anchor: d=4096, r=384 -> ~0.305 BPW.
    b = factorized_bpw(4096, 4096, 384)
    assert math.isclose(b, 0.305, rel_tol=5e-3)
    # bpw_lt_one_bound: b1 < 1 iff r < (d^2 - 32d) / (2 log2(3) d + 16).
    d = 1024
    bound = bpw_lt_one_rank_bound(d)
    r = int(bound) - 1
    assert factorized_bpw(d, d, r) < 1.0
    r = int(bound) + 1
    assert factorized_bpw(d, d, r) > 1.0


def test_merge_fixed_width_preserves_width() -> None:
    # The headline property: output width == input width. NO 3x.
    gen = torch.Generator().manual_seed(8)
    d = 16
    shared = {
        "w": torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.1,
        "norm": torch.ones(d, dtype=torch.float64),
    }
    experts = []
    for _ in range(3):
        sd = {
            "w": shared["w"]
            + 0.05 * torch.randn(d, d, generator=gen, dtype=torch.float64),
            "norm": shared["norm"],
        }
        experts.append(sd)
    merged = merge_fixed_width(experts, shared)
    assert merged.n_experts == 3
    assert set(merged.root_state_dict) == set(shared)
    for key, value in merged.root_state_dict.items():
        assert value.shape == shared[key].shape
        assert value.dtype == shared[key].dtype
    report = merged.bpw_report()
    assert report["total_weights"] == d * d  # only the 2D float weight merges
    assert report["total_bpw"] > 0.0
    # Reconstruction identity survives the full pipeline per layer.
    for i, sd in enumerate(experts):
        recon = merged.root_state_dict["w"] + merged.contrast_matrices["w"][i]
        # within factorization + root compression error, not exact
        assert float((recon - sd["w"]).norm()) < 0.5 * float(
            (sd["w"] - shared["w"]).norm()
        )


def test_merge_fixed_width_two_identical_experts() -> None:
    # End-to-end R242: equal experts, latent bases rotated — the merged
    # root still equals the experts' matrix.
    gen = torch.Generator().manual_seed(9)
    d, r = 24, 6
    U = _orthonormal(d, r, gen)
    V = torch.randn(d, r, generator=gen, dtype=torch.float64)
    W_shared = torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.05
    delta = U @ V.T
    Q = _random_orthogonal(r, gen)
    W = W_shared + delta
    e0 = {"w": W}
    e1 = {"w": W_shared + (U @ Q) @ (V @ Q).T}  # same matrix, rotated latents
    shared = {"w": W_shared}
    merged = merge_fixed_width([e0, e1], shared, energy=1.0, root_energy=1.0)
    err = float((merged.root_state_dict["w"] - W).norm())
    assert err < 1e-4 * float(W.norm())
