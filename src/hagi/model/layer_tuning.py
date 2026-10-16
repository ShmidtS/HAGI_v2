"""Exact layer/embedding tunings from the formalization
(External/Layers.lean wave-5, External/Transfer.lean wave-4).

Every function transcribes a proved theorem into a runtime
knob-derivation; measured quantities (branch variances v_l,
layer values V_l, vocab-sweep losses) are arguments — h_emp_
discipline, nothing tuned.

* ``branchscale_variance_init`` — BranchScale min-max floor
  (Layers.lean ``branchscale_minmax``): with Σ s_l² = B the
  worst per-layer contribution max_l s_l²·v_l ≥ B/Σ(1/v_l),
  attained when s_l²·v_l is UNIFORM — per-layer residual scales
  compensate the measured branch variance (a uniform scale is
  provably suboptimal whenever v_l varies).
* ``dead_layer_stop`` — DepthBench adaptive depth
  (``dead_layer_stop``): with monotone decreasing layer values
  V_l = ΔQ_remove(l)/T_l, layer L dropping under the marginal
  bar λ stops the depth — L* = max{L : V_L > λ}; a non-monotone
  table is the dead-layer signature (reported, not hidden).
* ``vocab_critical_point`` — compute-optimal vocab
  (Transfer.lean ``vocab_critical_point``): the cost model
  L(V) = log V + c/V has its critical point at V* = c; fit c
  from a measured 3-point |V| sweep (least squares on the
  linear form V·L = V·log V + c), predict the optimum.
* ``admission_variance_ratio`` — next-expert admission
  (``admission_var_criterion``): under the quadratic gap law the
  argmax of ΔVar/ΔC IS the argmax of ΔGap/ΔC (the ½ factor
  commutes with the ratio ordering) — admit for new INFORMATION
  per compute, never for standalone loss (selection_hurts).
"""
from __future__ import annotations

import math

import numpy as np


def branchscale_variance_init(
    branch_vars: np.ndarray, budget: float | None = None
) -> np.ndarray:
    """s_l² ∝ 1/v_l normalized to Σ s_l² = budget (default: the
    uniform allocation's budget L·(1/√(2L))² = 1/2, so the total
    residual energy matches the existing 1/√(2L) convention)."""
    v = np.asarray(branch_vars, dtype=float)
    if np.any(v <= 0):
        raise ValueError("branch variances must be positive")
    n = v.shape[0]
    if budget is None:
        budget = 0.5  # L·(1/√(2L))² = 1/2 for any L
    inv = 1.0 / v
    s_sq = inv / inv.sum() * budget
    return np.sqrt(s_sq)


def branchscale_minmax_floor(branch_vars: np.ndarray, budget: float) -> float:
    """The proved LOWER bound: max_l s_l²·v_l ≥ B/Σ(1/v_l)."""
    v = np.asarray(branch_vars, dtype=float)
    if np.any(v <= 0):
        raise ValueError("branch variances must be positive")
    return budget / float(np.sum(1.0 / v))


def layer_value_table(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """Report per-layer values V_l = ΔQ_remove(l)/T_l and whether the
    table is monotone decreasing (the healthy DepthBench regime)."""
    v = np.asarray(values, dtype=float)
    mono = bool(np.all(np.diff(v) <= 1e-12))
    return v, mono


def dead_layer_stop(values: np.ndarray, lam: float) -> int:
    """L* = max{L : V_L > λ} (0-based index + 1 = the healthy depth).

    Raises ValueError when the table is non-monotone: a non-monotone
    value table is the dead-layer SIGNATURE (Pre-LN collapse), not a
    stop condition to act on."""
    v, mono = layer_value_table(values)
    if not mono:
        raise ValueError(
            "non-monotone layer-value table: dead-layer signature, "
            "diagnose before pruning"
        )
    alive = np.nonzero(v > lam)[0]
    return int(alive[-1]) + 1 if alive.size else 0


def vocab_critical_point(
    vocab_sizes: np.ndarray, losses: np.ndarray
) -> tuple[float, float]:
    """Fit c in L(V) = log V + c/V from a measured |V| sweep
    (linear least squares on V·(L − log V) = c), return (c, V*).

    V* = c is the predicted compute-optimal vocab (the critical
    point of the uniform-softmax baseline + sequence-length cost)."""
    vs = np.asarray(vocab_sizes, dtype=float)
    ls = np.asarray(losses, dtype=float)
    if vs.shape != ls.shape or vs.size < 2 or np.any(vs <= 1):
        raise ValueError("need matching arrays with V > 1")
    resid = vs * (ls - np.log(vs))  # each point estimates c
    c = float(resid.mean())
    return c, c  # V* = c


def admission_variance_ratio(
    delta_var: np.ndarray, delta_cost: np.ndarray
) -> int:
    """argmax ΔVar/ΔC — the certified admission order under the
    quadratic gap law (½ commutes with the ratio ordering)."""
    dv = np.asarray(delta_var, dtype=float)
    dc = np.asarray(delta_cost, dtype=float)
    if dv.shape != dc.shape or np.any(dc <= 0):
        raise ValueError("costs must be positive and arrays matching")
    return int(np.argmax(dv / dc))
