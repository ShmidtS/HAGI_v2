#!/usr/bin/env python3
"""Curriculum phase accounting — port of primes CurriculumCert (C330+).

The theory (Hagi/Contracts/CurriculumCert.lean) certifies mixed-phase
curricula: phase i contributes delta_i / d_i (its realized gap drop
over its certified uniform drop). The total is ORDER-FREE
(curriculumTotal_comm): phase ordering does not change the
certificate value — an honest limit of phase-rate accounting.

Here the phases are the selfloop stages GROW -> DERIVE -> OPTIMIZE ->
GATE. Each phase records:
  delta  — realized improvement of its gate metric this phase
  d      — certified uniform drop guaranteed per phase step
and the curriculum certificate is the order-free sum delta_i/d_i,
persisted into self_state so growth trajectories carry an explicit
curriculum certificate (theory-aligned, honestly measured — no claim
beyond what the numbers say).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

PHASES = ("GROW", "DERIVE", "OPTIMIZE", "GATE")


@dataclass
class PhaseRecord:
    phase: str
    delta: float          # realized metric drop this phase (>=0 good)
    d_cert: float         # certified uniform drop per phase (floor)
    steps: int = 1

    @property
    def contribution(self) -> float:
        return self.delta / self.d_cert if self.d_cert > 0 else 0.0


def curriculum_total(records: list[PhaseRecord]) -> float:
    """Order-free certificate: sum of per-phase delta/d (theory
    curriculumTotal; commutativity is exact by construction here)."""
    return round(sum(r.contribution for r in records), 6)


def certificate(state: dict[str, Any]) -> dict[str, Any]:
    """Attach/refresh the curriculum certificate to a self_state dict."""
    recs = [PhaseRecord(**r) for r in state.get("curriculum", [])]
    state["curriculum_certificate"] = {
        "phases": [asdict(r) for r in recs],
        "total": curriculum_total(recs),
        "order_free": True,   # theory: curriculumTotal_comm
        "theorem": "curriculum_phase_accounting (CurriculumCert.lean)",
    }
    return state
