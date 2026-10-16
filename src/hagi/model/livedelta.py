"""LiveDelta — low-rank plasticity of living weights (RCDL mode-A port).

Ports ``Hagi/Architecture/LiveDelta.lean``, ``Hagi/Step/DeltaHistory.lean``,
``Hagi/Step/LiveDeltaSafe.lean``, ``Hagi/Step/DeltaConsolidate.lean``,
``Hagi/Step/MDLThreeCurrency.lean`` and ``Hagi/Unified/QuantSafeUpdate.lean``
(C:\\primes, read-only SSOT). Pure torch, CPU.

State: base (compressed) weight matrix W0 plus a list of accepted
low-rank blocks ΔW = Aᵀ·B. Exact properties, no approximations:

* ``effectiveWeights`` — base + Σ AᵀB (`LiveDelta.effectiveWeights`);
* ``add_block`` / ``rollback`` — accept/reject, rollback is exact
  (`LiveDelta.rollback_addBlock`);
* ``update_newest`` — replaces only the newest B-factor; length and
  state cost unchanged (`LiveDeltaSafe.updateNewest_*`);
* ``from_history`` — fold rank-1 steps into blocks; the EXACT bridge
  `descentFold_eq_effective`: effective(from_history(W0,h)) == W0 + Σ v⊗x;
* ``history_storage_cost`` — k rank-1 steps cost k·(m+n) elements
  (`DeltaHistory.history_storage_cost`);
* ``blocks_cost`` — k rank-r blocks cost k·r·(m+n) (`LiveDelta.blocks_cost`);
* ``frob_sum_le`` / ``replace_blocks_delta`` — additive Frobenius
  consolidation budget (`DeltaConsolidate.frob_sum_le`,
  `effective_replaceBlocks`);
* ``J3`` — three-currency MDL objective with monotonicity
  (`MDLThreeCurrency.J3_nonincreasing`), double-counting detector
  (`double_counting_identified`) and the entropy floor
  (`no_compression_below_entropy`);
* ``quantized_safe_window`` — ideal SafeQP window + explicit
  quantization price (η‖g‖q + Lη²(‖d‖q + q²/2))
  (`QuantSafeUpdate.quantized_safe_window`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "LiveDelta",
    "effective_weights",
    "add_block",
    "rollback",
    "update_newest",
    "from_history",
    "history_storage_cost",
    "blocks_cost",
    "frob_sum_le",
    "replace_blocks_delta",
    "J3",
    "j3_nonincreasing",
    "double_counting_identified",
    "no_compression_below_entropy",
    "quantized_safe_window",
]


@dataclass
class LiveDelta:
    """Living weights: base + accepted low-rank blocks (newest first).

    Mirrors ``Hagi.Architecture.LiveDelta``: ``base`` is m×n, each block
    is a pair ``(A: r×m, B: r×n)`` contributing ``Aᵀ·B``.
    """

    base: torch.Tensor
    blocks: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    @property
    def m(self) -> int:
        return self.base.shape[0]

    @property
    def n(self) -> int:
        return self.base.shape[1]


def effective_weights(s: LiveDelta) -> torch.Tensor:
    """Base + Σ Aᵀ·B — `LiveDelta.effectiveWeights` (exact)."""
    out = s.base.clone()
    for a, b in s.blocks:
        out = out + a.t() @ b
    return out


def add_block(a: torch.Tensor, b: torch.Tensor, s: LiveDelta) -> LiveDelta:
    """Accept a block at the head of the list — `LiveDelta.addBlock`."""
    return LiveDelta(base=s.base, blocks=[(a, b)] + list(s.blocks))


def rollback(s: LiveDelta) -> LiveDelta:
    """Reject the newest block; empty list is identity —
    `LiveDelta.rollback` (exact undo, `rollback_addBlock`)."""
    if not s.blocks:
        return LiveDelta(base=s.base, blocks=[])
    return LiveDelta(base=s.base, blocks=list(s.blocks[1:]))


def update_newest(b_new: torch.Tensor, s: LiveDelta) -> LiveDelta:
    """Replace only the newest B-factor; base, length and state cost
    unchanged — `LiveDeltaSafe.updateNewest` (`updateNewest_base`,
    `updateNewest_length`, `updateNewest_cost`)."""
    if not s.blocks:
        return LiveDelta(base=s.base, blocks=[])
    (a, _old), rest = s.blocks[0], s.blocks[1:]
    return LiveDelta(base=s.base, blocks=[(a, b_new)] + list(rest))


def from_history(w0: torch.Tensor,
                 hist: list[tuple[torch.Tensor, torch.Tensor]]) -> LiveDelta:
    """Fold rank-1 steps (v, x) into blocks over base W0 —
    `DeltaHistory.fromHistory` (mode A: W_{t+1} = W_t + v_t ⊗ x_t)."""
    s = LiveDelta(base=w0, blocks=[])
    for v, x in hist:
        # rank-1 block: outer factors shaped (1,m) and (1,n)
        s = add_block(v.reshape(1, -1), x.reshape(1, -1), s)
    return s


def history_storage_cost(w0: torch.Tensor,
                         hist: list[tuple[torch.Tensor, torch.Tensor]]
                         ) -> tuple[int, int]:
    """(livedelta, dense) element counts for k rank-1 steps:
    k·(m+n) vs k·(m·n) — `DeltaHistory.history_storage_cost`."""
    m, n = w0.shape
    k = len(hist)
    return k * (m + n), k * (m * n)


def blocks_cost(s: LiveDelta) -> int:
    """Exact state cost: k blocks × r·(m+n) factor elements —
    `LiveDelta.blocks_cost` / `blockCost`."""
    m, n = s.m, s.n
    return sum(a.shape[0] * (m + n) for a, _b in s.blocks)


def frob_sum_le(deltas: list[torch.Tensor]) -> tuple[float, float]:
    """Additive Frobenius consolidation budget:
    ‖ΣΔᵢ‖²_F ≤ k·Σ‖Δᵢ‖²_F — `DeltaConsolidate.frob_sum_le`.
    Returns (lhs, rhs); lhs ≤ rhs holds exactly (Cauchy–Schwarz)."""
    k = len(deltas)
    total = deltas[0]
    for d in deltas[1:]:
        total = total + d
    lhs = float(total.pow(2).sum())
    rhs = k * sum(float(d.pow(2).sum()) for d in deltas)
    return lhs, rhs


def replace_blocks_delta(old: list[tuple[torch.Tensor, torch.Tensor]],
                         new: list[tuple[torch.Tensor, torch.Tensor]]
                         ) -> torch.Tensor:
    """Replacing factors blockwise changes effective weights by EXACTLY
    Σ(A'ᵀB' − AᵀB) — `DeltaConsolidate.effective_replaceBlocks` /
    `sum_sub_eq`."""
    out = None
    for (a, b), (a2, b2) in zip(old, new):
        d = a2.t() @ b2 - a.t() @ b
        out = d if out is None else out + d
    assert out is not None, "empty block lists"
    return out


def J3(l_pred: float, c_state: float, r_prot: float,
       lam_m: float, lam_r: float) -> float:
    """Three-currency objective L + λ_M·C + λ_R·R —
    `MDLThreeCurrency.J3`."""
    return l_pred + lam_m * c_state + lam_r * r_prot


def j3_nonincreasing(old: tuple[float, float, float],
                     new: tuple[float, float, float],
                     lam_m: float, lam_r: float) -> bool:
    """All three currencies non-increasing (weights ≥ 0) ⇒ J3
    non-increasing — `MDLThreeCurrency.J3_nonincreasing`."""
    lp, lp2 = old[0], new[0]
    cs, cs2 = old[1], new[1]
    rp, rp2 = old[2], new[2]
    if lp2 <= lp and cs2 <= cs and rp2 <= rp and lam_m >= 0 and lam_r >= 0:
        assert J3(lp2, cs2, rp2, lam_m, lam_r) <= J3(lp, cs, rp, lam_m, lam_r)
        return True
    return False


def double_counting_identified(l_pred: float, r_prot: float,
                               c_state: float, lam_m: float,
                               lam_r: float) -> bool:
    """Detects the naive currency substitution C_state = L_pred: J3 then
    degenerates to (1+λ_M)·L_pred + λ_R·R — the same compression counted
    twice — `MDLThreeCurrency.double_counting_identified`."""
    return math.isclose(c_state, l_pred, rel_tol=1e-12, abs_tol=1e-12) and (
        math.isclose(J3(l_pred, c_state, r_prot, lam_m, lam_r),
                     (1 + lam_m) * l_pred + lam_r * r_prot,
                     rel_tol=1e-9, abs_tol=1e-12))


def no_compression_below_entropy(p: torch.Tensor,
                                 q: torch.Tensor) -> tuple[float, float]:
    """(H(p), CE(p,q)): expected code length by the model is never below
    the source entropy — dataCodeLen − sourceEntropy = KL ≥ 0 —
    `MDLThreeCurrency.no_compression_below_entropy` (bridge from
    `kl_nonneg`)."""
    h = float(-(p * p.log()).sum())
    ce = float(-(p * q.log()).sum())
    assert h <= ce + 1e-6
    return h, ce


def quantized_safe_window(g_norms: list[float], lips: list[float],
                          d_norm: float, eta: float, q: float,
                          eps: list[float]) -> list[float]:
    """Per-domain bounds for a perturbed step d+δ, ‖δ‖ ≤ q:
    dL_i ≤ ε_i + η‖g_i‖q + L_i·η²·(‖d‖q + q²/2) —
    `QuantSafeUpdate.quantized_safe_window`
    (ideal window plus an EXPLICIT quantization price)."""
    price_lin = eta * q
    return [eps[i] + price_lin * g_norms[i]
            + lips[i] * eta * eta * (d_norm * q + q * q / 2)
            for i in range(len(eps))]
