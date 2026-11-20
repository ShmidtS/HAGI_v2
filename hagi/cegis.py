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
from typing import Callable

from hagi.invention import ConstructBlueprint, synthesize_blueprint


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
    status: str                 # PROVEN_EXACT | HOLD_CEGIS_EXHAUSTED | HOLD_NO_CANDIDATE
    blueprint: ConstructBlueprint | None
    rounds: int
    queries: int                # oracle label budget spent
    train_size: int             # final train size (init + counterexamples)
    final_sweep: dict | None = None   # full-domain certificate result
    cert_queries: int = 0   # WAVE 175: full-domain labeling, tracked
                            # separately from the search budget


def cegis_invent(train_x: list[int],
                 query: Callable[[int], int | None],
                 cegis_pool: list[int],
                 full_domain: range,
                 max_rounds: int = 6,
                 init_frac: float = 0.25,
                 query_budget: int | None = None,
                 rng: random.Random | None = None) -> CegisResult:
    """Active CEGIS: synthesize on train, refine on counterexamples from
    cegis_pool, certify on full_domain at the end.

    - init train = first init_frac of train_x (small induction seed)
    - each round: candidate fails on cegis_pool or memory -> add the
      counterexample labels to train and re-synthesize
    - queries counts ONLY points whose labels the loop actually asked
      for (never full_domain during search).
    """
    rng = rng or random.Random(0)
    if not train_x:
        return CegisResult("HOLD_NO_CANDIDATE", None, 0, 0, 0)

    k0 = max(2, int(len(train_x) * init_frac))
    train = dict((x, query(x)) for x in train_x[:k0])
    queries = len(train)
    mem = CounterexampleMemory()
    pool = list(cegis_pool)
    rng.shuffle(pool)

    best: ConstructBlueprint | None = None
    for round_i in range(1, max_rounds + 1):
        pairs = sorted(train.items())
        bp = synthesize_blueprint(pairs, rng=rng)
        if bp is None:
            # richer train may help later rounds; keep going only if we
            # can still add counterexamples
            if not pool and not mem.points:
                return CegisResult("HOLD_NO_CANDIDATE", None, round_i,
                                   queries, len(train))
            best = None
        else:
            # 1) memory violations (cheap, no oracle calls)
            bad = mem.violations(bp)
            if bad:
                for x, y in bad:
                    train[x] = y
                best = bp
                continue
            # 2) cegis pool: probe actively until a counterexample or
            #    the pool is consistent with the candidate
            ce = None
            while pool:
                if query_budget is not None and queries >= query_budget:
                    break
                x = pool.pop()
                y = query(x)
                queries += 1
                if y is None:
                    continue
                mem.add(x, y)
                if bp.apply(x) != y:
                    ce = (x, y)
                    break
            if ce is None:
                if query_budget is not None and queries >= query_budget:
                    return CegisResult("HOLD_CEGIS_BUDGET", bp, round_i,
                                       queries, len(train))
                # CERTIFICATION phase (wave 169 + WAVE 175): label the
                # FULL domain; cert_queries tracked separately from the
                # search budget. A certification counterexample with
                # rounds remaining feeds back into train (refine), not
                # a dead-end HOLD.
                cert_queries = 0
                for x in full_domain:
                    if x not in train:
                        y = query(x)
                        cert_queries += 1
                        queries += 1
                        if y is not None:
                            train[x] = y
                sweep = verify_blueprint(bp, sorted(train.items()),
                                         full_domain)
                if (sweep["status"] == "HOLD_COUNTEREXAMPLE"
                        and round_i < max_rounds):
                    cex = sweep["cex"]
                    train.setdefault(cex[0], cex[1])
                    mem.add(cex[0], cex[1])
                    best = bp
                    continue
                return CegisResult(sweep["status"], bp, round_i, queries,
                                   len(train), final_sweep=sweep,
                                   cert_queries=cert_queries)
            train[ce[0]] = ce[1]
            best = bp

    return CegisResult("HOLD_CEGIS_EXHAUSTED", best, max_rounds, queries,
                       len(train))


# re-export so callers keep one import site
from hagi.invention import verify_blueprint  # noqa: E402,F401
