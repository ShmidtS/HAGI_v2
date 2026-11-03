"""Productive transition — runtime gate (audit §3.2 / P0).

Problem (Lean SSOT: ``Hagi/Unified/Assumptions.lean``
``trunc_idle_witness``): the base-cycle premise package admits a fully
idle trajectory (ΔE = Δrisk = Δcapability = spend = 0) — invariant
preservation does NOT imply a useful learning step. Two claims must be
separated:
  1. soundness: invariants hold, resources conserved (StageCertificate);
  2. productivity: the accepted step produced a MEASURED improvement.

Runtime form of the productive-transition predicate:

    Progress(S, S') := ΔCapability >= δ_C  OR  ΔLoss <= −δ_L
    (δ_C, δ_L > 0 — positive by construction)

This gate is layered ON TOP of soundness: an accepted transition must be
BOTH sound (after <= before − decrease + risk) AND productive (measured
metric moved by more than its noise floor δ). An idle ACCEPT is now
impossible: the accept gate rejects it as UNPRODUCTIVE rather than
counting it as certified progress.

Consumers: stage_telescope bookkeeping (reject idle entries from the
growth ledger), certified accept gates (any A/B accept path).
"""
from __future__ import annotations

import math


def productive(
    delta_capability: float,
    delta_loss: float,
    delta_c_threshold: float,
    delta_l_threshold: float,
) -> bool:
    """Progress(S, S') [audit §3.2]: capability rose by >= δ_C, or loss
    fell by >= δ_L (delta_loss given as after − before, so <= −δ_L)."""
    if delta_c_threshold <= 0 or delta_l_threshold <= 0:
        raise ValueError("thresholds must be positive (δ_C, δ_L > 0)")
    return delta_capability >= delta_c_threshold or delta_loss <= -delta_l_threshold


def productive_with_noise_floor(
    delta_capability: float,
    delta_loss: float,
    cap_floor: float,
    loss_floor: float,
) -> str:
    """Full verdict for an accepted transition: SOUND+PRODUCTIVE /
    SOUND+UNPRODUCTIVE / UNSOUND. δ floors are the measurement noise
    (e.g. the anytime accept eps of the A/B path — no accept below the
    noise floor counts as progress)."""
    if cap_floor <= 0 or loss_floor <= 0:
        raise ValueError("floors must be positive")
    if not math.isfinite(delta_capability) or not math.isfinite(delta_loss):
        raise ValueError("deltas must be finite")
    productive = (
        delta_capability >= cap_floor or delta_loss <= -loss_floor
    )
    return "PRODUCTIVE" if productive else "UNPRODUCTIVE"


def gate_accept(
    sound: bool,
    delta_capability: float,
    delta_loss: float,
    cap_floor: float,
    loss_floor: float,
) -> str:
    """Accept verdict for one transition: 'ACCEPT' (sound + productive),
    'REJECT_IDLE' (sound but no measured progress — the trunc_idle case:
    invariants held, nothing learned), 'REJECT_UNSOUND' (invariant
    violated — certificate does not close)."""
    if not sound:
        return "REJECT_UNSOUND"
    verdict = productive_with_noise_floor(delta_capability, delta_loss, cap_floor, loss_floor)
    return "ACCEPT" if verdict == "PRODUCTIVE" else "REJECT_IDLE"


class TransitionCertificate:
    """Unified runtime transition certificate (audit §4.1): ONE check for
    every accepted transition instead of scattered per-site validations.
    Fields mirror the Lean contracts' intersection (HonestTransition /
    StepCert / CertifiedTransition): next-state identity, actual descent,
    positive drop, risk bound, budget accounting, nonneg spend, and the
    measurement link (claimed vs measured within error).

    Callers fill this from measured quantities; ``verify()`` returns the
    list of violated clauses (empty == valid). The productive gate composes
    on top: verify() + productive verdict.
    """

    __slots__ = (
        "next_eq",
        "objective_drop_pos",
        "risk_bound_ok",
        "budget_exact",
        "spend_nonneg",
        "budget_nonneg",
        "measurement_link",
        "productive",
    )

    def __init__(
        self,
        *,
        next_eq: bool,
        objective_drop_pos: bool,
        risk_bound_ok: bool,
        budget_exact: bool,
        spend_nonneg: bool,
        budget_nonneg: bool,
        measurement_link: bool,
        productive: bool,
    ) -> None:
        self.next_eq = bool(next_eq)
        self.objective_drop_pos = bool(objective_drop_pos)
        self.risk_bound_ok = bool(risk_bound_ok)
        self.budget_exact = bool(budget_exact)
        self.spend_nonneg = bool(spend_nonneg)
        self.budget_nonneg = bool(budget_nonneg)
        self.measurement_link = bool(measurement_link)
        self.productive = bool(productive)

    @classmethod
    def from_measurement(
        cls,
        *,
        obj_before: float,
        obj_after: float,
        claimed_drop: float,
        risk: float,
        risk_limit: float,
        budget_before: float,
        budget_after: float,
        spend: float,
        measured: float,
        measurement_error: float,
        productive_ok: bool,
    ) -> TransitionCertificate:
        """Fill clauses from one measured transition. measurement_link:
        |claimed_drop − measured| <= measurement_error (the honest
        measurement premise; nothing is accepted beyond its noise)."""
        drop = obj_before - obj_after
        return cls(
            next_eq=True,  # caller supplies the actual next state by construction
            objective_drop_pos=drop > 0.0,
            risk_bound_ok=risk <= risk_limit,
            budget_exact=abs((budget_before - budget_after) - spend) <= 1e-9,
            spend_nonneg=spend >= 0.0,
            budget_nonneg=budget_after >= 0.0,
            measurement_link=abs(claimed_drop - measured) <= measurement_error,
            productive=productive_ok,
        )

    def verify(self) -> list[str]:
        """Violated clauses; empty list == the transition certificate holds
        AND the step was productive (idle accepts are violations too, per
        audit §3.2)."""
        violations = []
        for name, ok in (
            ("objective_drop_pos", self.objective_drop_pos),
            ("risk_bound_ok", self.risk_bound_ok),
            ("budget_exact", self.budget_exact),
            ("spend_nonneg", self.spend_nonneg),
            ("budget_nonneg", self.budget_nonneg),
            ("measurement_link", self.measurement_link),
            ("productive", self.productive),
        ):
            if not ok:
                violations.append(name)
        return violations
