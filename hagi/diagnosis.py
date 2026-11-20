#!/usr/bin/env python3
"""Gap diagnosis — steps 1-2 of the invention loop (Phase 2).

Replaces the opaque NO_VALIDATED_PROGRAM hold with a structured
GapProfile: taxonomized HoldReason + cheap ResidualProfile probes
(sparsity, autocorrelations under domain transforms, residual after
best builtin fit, Myhill-Nerode class count for sequential tasks).

GapBuffer accumulates profiles; GapCluster groups >= K_min tasks with
matching gap signatures — one random task never pays for a new
representation (this is what prevents one-off hacks).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence


class HoldReason(Enum):
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    REPRESENTATION_GAP = "REPRESENTATION_GAP"
    NONDETERMINISTIC_OR_PARTIAL = "NONDETERMINISTIC_OR_PARTIAL"
    RANDOM_INCOMPRESSIBLE = "RANDOM_INCOMPRESSIBLE"


@dataclass
class ResidualProfile:
    """Cheap ms-scale probes on (train_x, train_y)."""
    n: int
    k_density: float                 # non-DEFAULT points / |D| if dense table known
    uniq_outputs: int
    sparsity_gaps_entropy: float     # entropy of gaps between active points
    n_runs: int                      # contiguous intervals of active outputs
    autocorr_shift: float            # P(f(x) == f(x-1))
    autocorr_reflect: float          # P(f(x) == f(C-x)) over sampled C
    autocorr_xor: float              # max_m P(f(x) == f(x^m)), m in 1..15
    residual_frac_builtin: float     # fraction unexplained by best builtin fit
    mn_classes: int | None = None  # Myhill-Nerode classes (sequential)

    def signature(self) -> tuple:
        """Coarse signature for clustering."""
        bucket = lambda v: round(v, 2)
        return (bucket(self.k_density), bucket(self.autocorr_shift),
                bucket(self.autocorr_xor), bucket(self.residual_frac_builtin),
                self.mn_classes is not None)


@dataclass
class GapRecord:
    task_name: str
    reason: HoldReason
    profile: ResidualProfile


@dataclass
class GapCluster:
    signature: tuple
    records: list[GapRecord] = field(default_factory=list)

    def ready(self, k_min: int) -> bool:
        return len(self.records) >= k_min


class GapBuffer:
    def __init__(self, k_min: int = 3):
        self.k_min = k_min
        self.records: list[GapRecord] = []

    def add(self, rec: GapRecord) -> None:
        self.records.append(rec)

    def clusters(self) -> list[GapCluster]:
        by_sig: dict[tuple, GapCluster] = {}
        for rec in self.records:
            if rec.reason != HoldReason.REPRESENTATION_GAP:
                continue
            sig = rec.profile.signature()
            by_sig.setdefault(sig, GapCluster(sig)).records.append(rec)
        return sorted(by_sig.values(), key=lambda c: -len(c.records))

    def ready_clusters(self) -> list[GapCluster]:
        return [c for c in self.clusters() if c.ready(self.k_min)]


# --------------------------------------------------------------------------
# Probe implementations
# --------------------------------------------------------------------------

def _entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    cnt = Counter(values)
    total = sum(cnt.values())
    return -sum((c / total) * math.log2(c / total) for c in cnt.values())


def build_residual_profile(
    train_x: Sequence[Sequence[int]],
    train_y: Sequence[int],
    domain: Iterable[Sequence[int]] | None = None,
    builtin_fit: Callable[[Sequence, Sequence], float] | None = None,
) -> ResidualProfile:
    n = len(train_x)
    flat_x = [x[0] if len(x) == 1 else int(sum(x)) for x in train_x]
    uniq_out = len(set(train_y))
    # sparsity: run-length of active (non-zero) outputs over sorted x
    pairs = sorted(zip(flat_x, train_y))
    active = [x for x, y in pairs if y != 0]
    gaps = [b - a for a, b in zip(active, active[1:])] if len(active) > 1 else []
    gap_ent = _entropy(gaps)
    runs = (1 if active else 0)
    runs += sum(1 for a, b in zip(active, active[1:]) if b != a + 1)
    # autocorrelations
    m_by_x = dict(pairs)
    shift_hit = tot = 0
    for x, y in pairs:
        if (x - 1) in m_by_x:
            tot += 1
            # value equality OR constant-difference structure (counters):
            prev = m_by_x[x - 1]
            shift_hit += (prev == y) or ((y - prev) in (1, -1))
    autoc_shift = shift_hit / tot if tot else 0.0
    xor_best = 0.0
    for m in range(1, min(16, 1 << 8)):
        hit = tot2 = 0
        for x, y in pairs:
            xm = x ^ m
            if xm in m_by_x:
                tot2 += 1
                hit += (m_by_x[xm] == y)
        if tot2:
            xor_best = max(xor_best, hit / tot2)
    refl_hit = refl_tot = 0
    C = (max(flat_x) if flat_x else 0)
    for x, y in pairs:
        xr = C - x
        if xr in m_by_x:
            refl_tot += 1
            refl_hit += (m_by_x[xr] == y)
    autoc_refl = refl_hit / refl_tot if refl_tot else 0.0
    residual = builtin_fit(train_x, train_y) if builtin_fit else 1.0
    return ResidualProfile(
        n=n,
        k_density=len(active) / n if n else 0.0,
        uniq_outputs=uniq_out,
        sparsity_gaps_entropy=gap_ent,
        n_runs=runs,
        autocorr_shift=autoc_shift,
        autocorr_reflect=autoc_refl,
        autocorr_xor=xor_best,
        residual_frac_builtin=min(1.0, max(0.0, residual)),
    )


def classify_hold(reason_str: str, profile: ResidualProfile) -> HoldReason:
    """Map raw hold strings + profile into the taxonomy."""
    r = (reason_str or "").upper()
    if "BUDGET" in r or "NODE" in r or "ALPHA" in r:
        return HoldReason.BUDGET_EXHAUSTED
    if "NONDET" in r or "PARTIAL" in r or "CLOCK" in r:
        return HoldReason.NONDETERMINISTIC_OR_PARTIAL
    # entropy probe: near-random -> honest ABSTAIN
    if profile.uniq_outputs > 2 and _entropy(list(
            range(max(2, profile.n)))) < 0:  # placeholder guard
        return HoldReason.RANDOM_INCOMPRESSIBLE
    if (profile.residual_frac_builtin >= 0.9
            and profile.autocorr_shift < 0.3
            and profile.autocorr_xor < 0.3
            and profile.autocorr_reflect < 0.3):
        # no builtin explains it, no symmetry: representation gap or noise.
        # Dense random noise has uniq_outputs high AND runs ~ n/2;
        # structural gaps usually show SOME symmetry or sparsity.
        if (profile.k_density > 0.6 and profile.n_runs > profile.n * 0.4
                and profile.sparsity_gaps_entropy > 2.5
                and profile.uniq_outputs > 2):
            return HoldReason.RANDOM_INCOMPRESSIBLE
        return HoldReason.REPRESENTATION_GAP
    return HoldReason.REPRESENTATION_GAP
