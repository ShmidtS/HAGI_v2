"""Шаг 2 плана формализации: streaming-PCA U_cov — тесты.

Ядро: FD-скетч восстанавливает top-r базис активаций без хранения
потока (память O(ell·d)); цепочка гарантий ε_cov → Weyl → Davis-Kahan
→ ε_i-бюджет R131. Вычитание σ_min²: PSD-невязка, сигнал выживает на
длинных потоках (halving убил бы его геометрически).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.train.safeqp_gpm import (  # noqa: E402
    activation_subspace,
    gpm_step,
    space_lora_factor,
)
from hagi.train.streaming_gpm import (  # noqa: E402
    FrequentDirections,
    streaming_activation_basis,
)

D, N, ELL = 96, 512, 32
CHUNK = 37  # взаимно просто с ELL: переполнения попадают в середину чанков


def _lowrank(rank: int = 6, noise: float = 0.0, seed: int = 0, n: int = N):
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(D, rank, generator=g))[0].to(torch.float64)
    coef = torch.randn(n, rank, generator=g).to(torch.float64)
    x = coef @ basis.T
    if noise:
        x = x + noise * torch.randn(n, D, generator=g).to(torch.float64)
    return x, basis


def _fd_over(x: torch.Tensor, ell: int = ELL, chunk: int = CHUNK):
    fd = FrequentDirections(x.shape[1], ell)
    for s in range(0, x.shape[0], chunk):
        fd.update(x[s:s + chunk])
    return fd


def test_basis_orthonormal_exact_rank():
    x, _ = _lowrank()
    b = _fd_over(x).basis()
    assert b.shape == (D, 6)
    assert torch.allclose(b.T @ b, torch.eye(6, dtype=torch.float64), atol=1e-10)


def test_streaming_matches_true_subspace():
    x, u_true = _lowrank()
    b = _fd_over(x).basis()
    # cos главных углов = 1: полный recovery точного ранга (допуск —
    # fp64-накопление округлений за ~15 переполнений скетча)
    cos = torch.linalg.svdvals(u_true.T @ b)
    assert (1.0 - cos).abs().max() < 1e-6


def test_batch_equal_when_stream_fits():
    # поток помещается в скетч (без переполнений) — базис = batch SVD
    x, _ = _lowrank(n=64, seed=1)
    fd = _fd_over(x, ell=128)
    p_stream = _projector(fd.basis())
    p_batch = _projector(activation_subspace(x))
    assert torch.allclose(p_stream, p_batch, atol=1e-12)


def _projector(b: torch.Tensor) -> torch.Tensor:
    return b @ b.T


def test_cov_error_bound_and_psd_residual():
    # затухающий спектр (реалистичные активации) + много переполнений
    x, _ = _lowrank(rank=8, noise=0.05, seed=3, n=1500)
    fd = _fd_over(x, ell=16, chunk=97)
    diff = x.T @ x - fd.sketch().T @ fd.sketch()
    eigs = torch.linalg.eigvalsh(diff)
    scale = float(torch.linalg.matrix_norm(x, ord="fro") ** 2)
    assert eigs.min() >= -1e-6 * scale  # PSD-невязка (Alg 2)
    assert eigs.max() <= fd.cov_error_bound()  # универсальный envelope


def test_weyl_eigenvalue_perturbation():
    x, _ = _lowrank(rank=8, noise=1e-3, seed=5)
    fd = _fd_over(x)
    ev_true = torch.linalg.eigvalsh(x.T @ x).flip(0)
    ev_sketch = torch.linalg.eigvalsh(fd.sketch().T @ fd.sketch()).flip(0)
    eps = fd.cov_error_bound()
    assert (ev_true - ev_sketch).abs().max() <= eps
    # PSD ⟹ спектр скетча снизу: λ_i(BᵀB) ≤ λ_i(XᵀX)
    assert (ev_sketch - ev_true).max() <= 1e-6 * eps


def test_gpm_step_streaming_vs_batch():
    g = torch.Generator().manual_seed(7)
    x, _ = _lowrank(rank=6, noise=0.0, seed=7)
    step = torch.randn(D, generator=g).to(torch.float64)
    d_stream = gpm_step(step, _fd_over(x).basis())
    d_batch = gpm_step(step, activation_subspace(x))
    assert torch.allclose(d_stream, d_batch, atol=1e-6)


def test_space_lora_zero_residual_response():
    g = torch.Generator().manual_seed(8)
    x, _ = _lowrank(rank=6, noise=0.0, seed=8)
    a_in = torch.randn(4, D, generator=g).to(torch.float64)
    a_proj = space_lora_factor(a_in, _fd_over(x).basis())
    assert torch.linalg.matrix_norm(x @ a_proj.T) < 1e-6 * torch.linalg.matrix_norm(
        x @ a_in.T
    )


def test_helper_streaming_activation_basis():
    x, _ = _lowrank()
    chunks = (x[s:s + CHUNK] for s in range(0, N, CHUNK))
    b = streaming_activation_basis(chunks, D, ELL)
    assert torch.allclose(_projector(b), _projector(_fd_over(x).basis()), atol=1e-12)


def test_memory_bounded_over_long_stream():
    x, _ = _lowrank(rank=4, noise=0.1, seed=9, n=5000)
    fd = _fd_over(x, ell=16, chunk=53)  # ~90 переполнений
    assert fd.sketch().shape == (16, D)  # память не растёт с потоком
    b = fd.basis()
    assert b.shape[1] >= 4  # сигнал 4 ранга выжил после вычитаний
    assert torch.allclose(b.T @ b, torch.eye(b.shape[1], dtype=torch.float64), atol=1e-10)


def test_validation():
    with pytest.raises(ValueError):
        FrequentDirections(dim=8, sketch_rows=1)
    fd = FrequentDirections(dim=8, sketch_rows=4)
    with pytest.raises(ValueError):
        fd.update(torch.randn(2, 9))
    fd.update(torch.randn(2, 8))
    fd.reset()
    assert fd.basis().shape == (8, 0)
