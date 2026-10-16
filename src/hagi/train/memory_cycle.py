"""LiveDelta memory cycle (Unified/MemoryCycle.lean, Growth/
MemoryController.lean R273, Step/DeltaConsolidate.lean R272).

Memory lives as rank-1 LiveDelta blocks: effective weights =
base + Σ a_i ⊗ b_i. The theory closes the chain R271–R273:

* admission controller (R273): an entry is admitted only when
  BOTH checks pass — novelty of the key (far from stored keys)
  AND certified gain covering state cost, protected risk, and
  the certification slack. Novelty alone never admits (noise is
  far from keys too); the decision is a conjunction.
* consolidation (R272): replacing block factors changes the
  effective weights by exactly Σ(new − old); the quadratic
  Frobenius budget frob(eff − eff') ≤ k · Σᵢ frob(bᵢ − bᵢ')
  (Cauchy for sums).
* full cycle (MemoryCycle.lean ``memory_cycle_quality``): under a
  measured L-Lipschitz loss premise (h_emp_), consolidated memory
  changes the loss by at most L·√(k·Σ εᵢ).

Pure module — measured quantities (Ĝ, ΔR_protected, L) are
arguments, not computed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Decision(str, Enum):
    ADMIT = "admit"
    REJECT = "reject"


@dataclass(frozen=True)
class AdmissionThresholds:
    """Explicit controller knobs (no magic defaults hidden inside)."""

    tau: float            # novelty threshold: dist(key, stored) > tau
    lam_m: float          # state-cost currency weight (λ_M)
    lam_r: float          # protected-risk currency weight (λ_R)
    eps_cert: float       # certification slack ε_cert


@dataclass(frozen=True)
class AdmissionMeasurements:
    """h_emp_ measurements supplied by the caller (never computed here)."""

    gain_hat: float       # Ĝ(x): certified expected gain
    delta_c_state: float  # ΔC_state: storage/state cost of the entry
    delta_r_protected: float  # ΔR_protected: risk to protected domains


def novelty(key: np.ndarray, stored_keys: list[np.ndarray]) -> float:
    """Cosine distance to the nearest stored key (1.0 for empty memory)."""
    if not stored_keys:
        return 1.0
    key_n = key / (np.linalg.norm(key) + 1e-12)
    best = 0.0
    for sk in stored_keys:
        sk_n = sk / (np.linalg.norm(sk) + 1e-12)
        cos = float(np.clip(np.dot(key_n, sk_n), -1.0, 1.0))
        best = max(best, cos)
    return 1.0 - best


def admit(
    key: np.ndarray,
    stored_keys: list[np.ndarray],
    m: AdmissionMeasurements,
    t: AdmissionThresholds,
) -> tuple[Decision, str]:
    """R273 conjunction: novelty ∧ certified gain. Reject reasons are
    explicit so callers can log the honest stopping cause."""
    nov = novelty(key, stored_keys)
    if not nov > t.tau:
        return Decision.REJECT, f"novelty {nov:.4f} ≤ τ {t.tau}"
    need = t.lam_m * m.delta_c_state + t.lam_r * m.delta_r_protected + t.eps_cert
    if not m.gain_hat >= need:
        return Decision.REJECT, f"gain {m.gain_hat:.4f} < cost {need:.4f}"
    return Decision.ADMIT, "novelty ∧ certified gain"


def consolidation_frob_budget(
    olds: list[tuple[np.ndarray, np.ndarray]],
    news: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Exact per-block sum Σᵢ frob(bᵢ − bᵢ') (Frobenius squares)."""
    if len(olds) != len(news):
        raise ValueError("olds/news length mismatch")
    total = 0.0
    for (ao, bo), (an, bn) in zip(olds, news):
        diff = np.outer(ao, bo) - np.outer(an, bn)
        total += float(np.sum(diff * diff))
    return total


def memory_cycle_quality_bound(k: int, eps_sum: float, lipschitz_l: float) -> float:
    """``memory_cycle_quality``: |Δloss| ≤ L·√(k·Σ εᵢ)."""
    if k < 0 or eps_sum < 0 or lipschitz_l < 0:
        raise ValueError("k, ε-sum and L must be nonnegative")
    return lipschitz_l * float(np.sqrt(k * eps_sum))


class LiveDeltaMemory:
    """Rank-1 LiveDelta store: eff = base + Σ blocks.

    ``blocks`` only ever grows through :meth:`try_admit` (the R273
    controller); rejected entries leave the memory unchanged — the
    "memory grows only on admit" theorem holds by construction.
    """

    def __init__(self, base: np.ndarray) -> None:
        self.base = base
        self.blocks: list[tuple[np.ndarray, np.ndarray]] = []

    def eff_weights(self) -> np.ndarray:
        eff = self.base.copy()
        for a, b in self.blocks:
            eff = eff + np.outer(a, b)
        return eff

    def keys(self) -> list[np.ndarray]:
        return [a for a, _ in self.blocks]

    def try_admit(
        self,
        a: np.ndarray,
        b: np.ndarray,
        m: AdmissionMeasurements,
        t: AdmissionThresholds,
    ) -> tuple[Decision, str]:
        decision, reason = admit(a, self.keys(), m, t)
        if decision is Decision.ADMIT:
            self.blocks.append((a, b))
        return decision, reason

    def consolidate(
        self,
        news: list[tuple[np.ndarray, np.ndarray]],
        *,
        max_eps_sum: float | None = None,
    ) -> float:
        """Replace block factors with ``news``.

        Returns the actual Frobenius-square deviation of the effective
        weights; raises ValueError when it exceeds the Cauchy budget
        k·Σᵢ frob(bᵢ − bᵢ') or the explicit ``max_eps_sum`` cap (the
        theory's bound is checkable, so we check it).
        """
        budget = len(self.blocks) * consolidation_frob_budget(
            self.blocks, news
        )
        before = self.eff_weights()
        self.blocks = list(news)
        after = self.eff_weights()
        diff = float(np.sum((before - after) ** 2))
        if diff > budget + 1e-9:
            raise ValueError(
                f"consolidation deviates {diff:.6f} > Cauchy budget {budget:.6f}"
            )
        if max_eps_sum is not None and diff > max_eps_sum:
            raise ValueError(
                f"consolidation deviates {diff:.6f} > cap {max_eps_sum:.6f}"
            )
        return diff


def admission_margin_sound(
    t: AdmissionThresholds,
    m: AdmissionMeasurements,
    gain_true: float,
) -> tuple[bool, float]:
    """R419 soundness certificate (`MemoryController.
    admission_margin_sound`): if the TRUE gain is within ε_cert of
    the estimate (θ-calibration R410/R413: ε_cert =
    (b−a)·√(ln(1/θ)/2)) and the gate would ADMIT (gain_hat ≥
    cost+risk+ε_cert), then the TRUE gain covers cost + risk:
        λ_M·ΔC + λ_R·ΔR ≤ gain_true.
    Returns (sound, covered_true_need); sound requires the premise
    |gtrue−gain_hat| ≤ ε_cert (checked) and the admit premise."""
    need_hat = (t.lam_m * m.delta_c_state
                + t.lam_r * m.delta_r_protected)
    admit_premise = m.gain_hat >= need_hat + t.eps_cert
    close = abs(gain_true - m.gain_hat) <= t.eps_cert
    covered = gain_true - need_hat
    return (admit_premise and close and covered >= 0.0), covered
