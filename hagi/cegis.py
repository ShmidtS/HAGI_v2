"""V10-1: Active CEGIS over invention blueprints.

Cycle: synthesize on a small train sample -> query an oracle only on
disjoint counterexample-stream points -> refine (add counterexamples
to the train set) -> repeat up to K rounds. Full-domain sweep is used
only for the final certificate, never to guide synthesis (no all_y
leak into hypothesis search).

Oracle protocol: the caller passes `query(x) -> y` (or None if x is
unlabeled). The CEGIS loop decides WHICH points to query — that is
the point: label budget is spent actively, not by reading the table.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from hagi.invention import (BASES, QUOTIENTS, ConstructBlueprint,
                            synthesize_blueprint)


@dataclass
class CounterexampleMemory:
    """Known hard points; every new candidate is checked here first."""
    points: dict[int, int] = field(default_factory=dict)

    def add(self, x: int, y: int) -> None:
        self.points[x] = y

    def violations(self, bp: ConstructBlueprint) -> list[tuple[int, int]]:
        return [(x, y) for x, y in self.points.items()
                if bp.apply(x) != y]


@dataclass
class CegisResult:
    status: str                 # PROVEN_EXACT only after complete labeled-domain audit
    blueprint: Optional[ConstructBlueprint]
    rounds: int
    queries: int                # oracle label budget spent
    train_size: int             # final train size (init + counterexamples)
    final_sweep: Optional[dict] = None   # full-domain certificate result


def cegis_invent(train_x: list[int],
                 query: Callable[[int], Optional[int]],
                 cegis_pool: list[int],
                 full_domain: range,
                 max_rounds: int = 6,
                 init_frac: float = 0.25,
                 query_budget: Optional[int] = None,
                 rng: Optional[random.Random] = None) -> CegisResult:
    """Active CEGIS with a budgeted, label-complete final certificate.

    Search queries are drawn from the disjoint pool. Before returning
    PROVEN_EXACT, the final audit obtains every missing full-domain label
    from the oracle, charging those calls to the same query budget.
    """
    rng = rng or random.Random(0)
    if not train_x:
        return CegisResult("HOLD_NO_CANDIDATE", None, 0, 0, 0)

    k0 = max(2, int(len(train_x) * init_frac))
    train: dict[int, int] = {}
    queries = 0
    for raw_x in train_x[:k0]:
        x = int(raw_x)
        if x in train:
            continue
        if query_budget is not None and queries >= query_budget:
            return CegisResult("HOLD_CEGIS_BUDGET", None, 0, queries, len(train))
        y = query(x)
        queries += 1
        if y is None:
            return CegisResult("HOLD_CEGIS_UNLABELED", None, 0, queries, len(train))
        train[x] = int(y)

    mem = CounterexampleMemory()
    pool = list(cegis_pool)
    rng.shuffle(pool)
    best: Optional[ConstructBlueprint] = None

    for round_i in range(1, max_rounds + 1):
        pairs = sorted(train.items())
        bp = synthesize_blueprint(pairs, rng=rng)
        if bp is None:
            if not pool:
                return CegisResult("HOLD_NO_CANDIDATE", None, round_i, queries, len(train))
            best = None
            # Without a candidate, no labels are useful until a later search
            # round changes the training set; retain the pool and continue.
            continue

        bad = mem.violations(bp)
        if bad:
            for x, y in bad:
                train[x] = y
            best = bp
            continue

        ce = None
        while pool:
            if query_budget is not None and queries >= query_budget:
                break
            x = int(pool.pop())
            if x in train:
                y = train[x]
            elif x in mem.points:
                y = mem.points[x]
            else:
                y = query(x)
                queries += 1
            if y is None:
                continue
            y = int(y)
            old = mem.points.get(x, train.get(x))
            if old is not None and old != y:
                return CegisResult("HOLD_ORACLE_INCONSISTENT", bp, round_i,
                                   queries, len(train))
            mem.add(x, y)
            if bp.apply(x) != y:
                ce = (x, y)
                break

        if ce is not None:
            train[ce[0]] = ce[1]
            best = bp
            continue

        # A budget stop while points remain is not a completed probe pass.
        if pool:
            return CegisResult("HOLD_CEGIS_BUDGET", bp, round_i, queries, len(train))

        # Final certificate: merge all known labels, and actively label any
        # full-domain point not already queried during induction/probing.
        known = dict(train)
        known.update(mem.points)
        audit_budget_hit = False

        def audit_query(x: int) -> Optional[int]:
            nonlocal queries, audit_budget_hit
            if x in known:
                return known[x]
            if query_budget is not None and queries >= query_budget:
                audit_budget_hit = True
                return None
            y = query(int(x))
            queries += 1
            if y is not None:
                known[int(x)] = int(y)
                mem.add(int(x), int(y))
                return int(y)
            return None

        sweep = verify_blueprint(bp, sorted(known.items()), full_domain,
                                 oracle=audit_query)
        sweep["oracle_queries"] = queries
        if sweep["status"] == "HOLD_UNLABELED" and audit_budget_hit:
            sweep["status"] = "HOLD_CEGIS_BUDGET"
        if sweep["status"] == "HOLD_COUNTEREXAMPLE":
            x, y = sweep["cex"]
            train[int(x)] = int(y)
            best = bp
            continue
        return CegisResult(sweep["status"], bp, round_i, queries,
                           len(train), final_sweep=sweep)

    return CegisResult("HOLD_CEGIS_EXHAUSTED", best, max_rounds, queries,
                       len(train))

# re-export so callers keep one import site
from hagi.invention import verify_blueprint  # noqa: E402,F401
