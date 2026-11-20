"""C477 Hadamard incoherence gate — rotate-if-measured-better before
ternary quantization (Theory/HadamardIncoherence.lean).

cohSq(w) = ‖w‖∞² / ‖w‖₂² (measured coherence², exact over explicit
max/sum). Gate: rotate only if cohSq(H w) < cohSq w (strict, ℚ-computable).
Scale invariance cohSq(c·w) = cohSq w — the ±1 Sylvester H needs no
normalization. Parseval: ‖H w‖₂² = 2^k·‖w‖₂² (energy preserved).
Honest negative: constant vectors CONCENTRATE under H (cohSq rises)
— the gate must refuse (rotate_hurts_constant).
"""
from __future__ import annotations

import numpy as np


def sylvester(k: int) -> np.ndarray:
    """Integer ±1 Sylvester Hadamard of size 2^k (recursive Kronecker)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    H = np.ones((1, 1), dtype=np.int64)
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    return H


def coh_sq(w: np.ndarray) -> float:
    """cohSq = ‖w‖∞² / ‖w‖₂² (0 if w is all-zero: nothing to measure)."""
    w = np.asarray(w, dtype=float).ravel()
    l2 = float((w * w).sum())
    if l2 <= 0:
        return 0.0
    return float(w.max() ** 2 if w.max() >= 0 else (-w).max() ** 2) / l2 \
        if False else float(np.abs(w).max()) ** 2 / l2


def rotate_gate(w: np.ndarray) -> dict:
    """Measure cohSq(w) vs cohSq(H w); return the decision. H and w
    must be length 2^k."""
    w = np.asarray(w, dtype=float).ravel()
    n = w.size
    if n & (n - 1):
        raise ValueError("w length must be a power of two")
    H = sylvester(int(np.log2(n))).astype(float)
    hw = H @ w
    before, after = coh_sq(w), coh_sq(hw)
    return {"cohSq_before": before, "cohSq_after": after,
            "rotate": after < before,
            "energy_ratio": float((hw * hw).sum() / (w * w).sum())
            if (w * w).sum() > 0 else 1.0}


def quantize_rows_with_gate(W: np.ndarray) -> np.ndarray:
    """Per-row: rotate-if-measured-better, then b1.58 ternary with the
    row scale computed IN THE ROTATED BASIS (s = mean|v| of the rotated
    row). Note: dequantization needs the row's rotation decision —
    consumers should store the per-row gate flags.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[1]
    if n & (n - 1):
        raise ValueError("row length must be a power of two")
    H = sylvester(int(np.log2(n))).astype(float)
    out = np.empty_like(W)
    flags: list[bool] = []
    for i in range(W.shape[0]):
        v = W[i]
        g = rotate_gate(v)
        if g["rotate"]:
            v, flag = H @ v, True
        else:
            flag = False
        s = np.abs(v).mean()
        s = s if s > 1e-12 else 1.0
        q = np.round(v / s).clip(-1.0, 1.0) * s
        out[i] = q
        flags.append(flag)
    return out, flags
