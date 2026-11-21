"""GPU (torch/CUDA-ROCm) backend for the HAGI neural core.

Ports the compute-heavy parts of ContinualGPMMemory — the EF21-Muon
training loop (Newton-Schulz orthogonalization + backtracking line
search) and the SVD subspace consolidation — to torch tensors on the
first available CUDA device.

Device policy (honest, measured):
  HAGI_NEURAL_DEVICE = auto | cpu | cuda   (default auto)
  - `auto` benchmarks one EF21-Muon step batch on GPU vs CPU at first
    use and keeps the faster backend (GPU kernel-launch overhead can
    dominate for the core's small matrices; the GPU wins once tasks
    are batched or the core is widened).
  - parallel spawn workers should pin `cpu` (one GPU context per
    worker would multiply GPU memory); the main process uses the GPU.

The GPU path is bit-compatible in structure but not bit-identical to
numpy (float64 GPU matmuls may differ in the last ulps); the certified
commit gates never depend on the neural core, so this only affects
adapter quality metrics, never accepted programs.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

_DEVICE: str | None = None          # resolved backend: "cpu" | "cuda"
_BENCH: dict[str, float] | None = None


def _torch():
    import torch
    return torch


def resolve_device() -> str:
    """Resolve and memoize the neural backend device."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    mode = os.environ.get("HAGI_NEURAL_DEVICE", "auto").lower()
    if mode == "cpu":
        _DEVICE = "cpu"
        return _DEVICE
    try:
        t = _torch()
        if not t.cuda.is_available():
            _DEVICE = "cpu"
            return _DEVICE
        if mode == "cuda":
            _DEVICE = "cuda"
            return _DEVICE
        # auto: quick benchmark, same shapes as a real fit step
        rng = np.random.default_rng(0)
        X = rng.normal(size=(256, 48))
        W = rng.normal(size=(48, 16)) * 0.1
        Y = rng.normal(size=(256, 16))
        cpu_t = _bench_step(np, X, W, Y)
        cuda_t = _bench_step_cuda(X, W, Y)
        _DEVICE = "cuda" if cuda_t < cpu_t else "cpu"
        return _DEVICE
    except Exception:
        _DEVICE = "cpu"
        return _DEVICE


def _bench_step(np_mod, X, W, Y) -> float:
    import time
    W = W.copy()
    t0 = time.perf_counter()
    for _ in range(30):
        err = X @ W - Y
        grad = (2.0 / len(X)) * (X.T @ err)
        W = W - 0.05 * grad
    return time.perf_counter() - t0


def _bench_step_cuda(X, W, Y) -> float:
    import time
    t = _torch()
    Xt = t.as_tensor(X, dtype=t.float64, device="cuda")
    Wt = t.as_tensor(W, dtype=t.float64, device="cuda").clone()
    Yt = t.as_tensor(Y, dtype=t.float64, device="cuda")
    t.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        err = Xt @ Wt - Yt
        grad = (2.0 / Xt.shape[0]) * (Xt.T @ err)
        Wt = Wt - 0.05 * grad
    t.cuda.synchronize()
    return time.perf_counter() - t0


def _newton_schulz_torch(G: Any, iters: int = 4) -> Any:
    """C296/C297 on-device: X <- X(3I - X^T X)/2 (right-multiply form,
    exact port of newton_schulz_orthogonalize from hagi/_v9.py)."""
    t = _torch()
    X = G.to(t.float64).clone()
    norm = float(X.norm())
    if norm < 1e-12:
        return X
    X = X / norm
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    I_m = t.eye(X.shape[0], dtype=t.float64, device=X.device)
    for _ in range(iters):
        X = 0.5 * (3.0 * I_m - X @ X.T) @ X
    return X.T if transposed else X


def train_task_ef21_muon_gpu(W: np.ndarray, ef_v: np.ndarray,
                             U_basis: np.ndarray, X: np.ndarray, Y: np.ndarray,
                             steps: int = 35, lr: float = 0.2
                             ) -> tuple[np.ndarray, np.ndarray, float]:
    """GPU EF21-Muon; mirrors ContinualGPMMemory.train_task_ef21_muon.

    Returns (W, ef_v, final_mse); caller writes them back into the GPM.
    """
    t = _torch()
    dev = "cuda"
    Xt = t.as_tensor(X, dtype=t.float64, device=dev)
    Yt = t.as_tensor(Y, dtype=t.float64, device=dev)
    Wt = t.as_tensor(W, dtype=t.float64, device=dev)
    Vt = t.as_tensor(ef_v, dtype=t.float64, device=dev)
    Ut = (t.as_tensor(U_basis, dtype=t.float64, device=dev)
          if U_basis.shape[1] else None)
    n = Xt.shape[0]

    def proj(G):
        return G if Ut is None else G - Ut @ (Ut.T @ G)

    for _ in range(steps):
        err = Xt @ Wt - Yt
        grad = (2.0 / n) * (Xt.T @ err)
        g_proj = proj(grad)
        Vt = Vt + 0.5 * (g_proj - Vt)
        step = proj(_newton_schulz_torch(Vt, iters=4))
        cur = float(t.mean((Xt @ Wt - Yt) ** 2))
        for eta in (lr, lr * 0.5, lr * 0.1, 0.01):
            cand = Wt - eta * step
            if float(t.mean((Xt @ cand - Yt) ** 2)) < cur:
                Wt = cand
                break
    mse = float(t.mean((Xt @ Wt - Yt) ** 2))
    W_out = Wt.cpu().numpy()
    V_out = Vt.cpu().numpy()
    t.cuda.synchronize()
    return W_out, V_out, mse


def consolidate_subspace_gpu(X: np.ndarray, U_basis: np.ndarray,
                             energy_thresh: float = 0.995,
                             max_basis: int | None = None) -> np.ndarray:
    """GPU SVD subspace consolidation; returns the new U_basis."""
    t = _torch()
    dev = "cuda"
    Xt = t.as_tensor(X, dtype=t.float64, device=dev)
    if U_basis.shape[1]:
        Ut = t.as_tensor(U_basis, dtype=t.float64, device=dev)
        res = Xt - (Xt @ Ut) @ Ut.T
    else:
        res = Xt
    if float(res.norm()) < 1e-10:
        return U_basis
    U, S, Vh = t.linalg.svd(res, full_matrices=False)
    S = S[S > 1e-10]
    Vh = Vh[: len(S)]
    tot = float(t.sum(S ** 2))
    if tot <= 0.0:
        return U_basis
    cum = t.cumsum(S ** 2, dim=0) / tot
    # WAVE 172 (C485): energy_thresh >= 1.0 means FULL span — float
    # cumsum caps at 0.99999... and silently drops trailing directions
    # (rank loss broke gpm_zero_forgetting: loss(A) +32% after train(B)).
    if energy_thresh >= 1.0:
        k = len(S)
    else:
        k = min(len(S), int(t.searchsorted(cum, t.as_tensor(energy_thresh, device=cum.device))) + 1)
    new_dirs = Vh[:k].T
    if U_basis.shape[1] == 0:
        Q, _ = t.linalg.qr(new_dirs)
        out = Q
    else:
        combined = t.cat([Ut, new_dirs], dim=1)
        Us, ss, _ = t.linalg.svd(combined, full_matrices=False)
        keep = ss > 1e-9
        out = Us[:, keep]
    if max_basis is None:
        max_basis = X.shape[1]
    # WAVE 172: keep the FULL span up to max_basis — the old
    # "max_basis - 2" margin always left 2 directions unconstrained,
    # letting updates leak into protected task-A directions.
    if out.shape[1] > max_basis:
        out = out[:, :max_basis]
    return out.cpu().numpy()


class GpuGpmMixin:
    """Drop-in acceleration for ContinualGPMMemory on the resolved device."""

    def _use_gpu(self) -> bool:
        return resolve_device() == "cuda"

    def train_task_ef21_muon(self, X: np.ndarray, Y: np.ndarray,
                             steps: int = 35, lr: float = 0.2) -> float:
        if len(X) == 0:
            return 0.0
        if self._use_gpu():
            W, ef_v, mse = train_task_ef21_muon_gpu(
                self.W, self.ef_v, self.U_basis, X, Y, steps=steps, lr=lr)
            self.W, self.ef_v = W, ef_v
            return mse
        return self._train_task_ef21_muon_cpu(X, Y, steps=steps, lr=lr)

    def consolidate_subspace(self, X: np.ndarray,
                             energy_thresh: float = 0.995) -> None:
        if self._use_gpu():
            self.U_basis = consolidate_subspace_gpu(
                X, self.U_basis, energy_thresh=energy_thresh,
                max_basis=self.W.shape[0])
            return
        self._consolidate_subspace_cpu(X, energy_thresh=energy_thresh)
