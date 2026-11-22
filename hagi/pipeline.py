"""Unified pipeline (WAVE 185): one flow, not scattered pieces.

The neural half and the symbolic half of HAGI run as ONE pipeline:

    ingest domain bytes
      -> certified LM training            (lm.train_certified)
      -> residual diagnosis               (diagnosis.build_residual_profile,
                                           classify_hold)
      -> representation invention on gaps (invention.invent_blueprint)
      -> strict gate                      (invention.verify_blueprint +
                                           capability meter)
      -> adoption                         (representations.reify_construct)
      -> certified result appended to the unified ledger

Every module keeps its own single responsibility; this file only
sequencing. Enabled by env HAGI_SYMBOLIC=1 (off by default so the
heavy replay loop stays fast); the symbolic phase is skipped when
the LM holdout is already trivially good.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from hagi.diagnosis import (GapBuffer, GapRecord, HoldReason,
                            build_residual_profile, classify_hold)
from hagi.invention import (cegis_invent, invent_blueprint,
                            verify_blueprint)
from hagi.representations import reify_construct


@dataclass
class PipelineResult:
    """One ingest's full trace: neural + symbolic halves."""
    source: str
    holdout_ce: float
    hold_reason: HoldReason | None = None
    invented: dict | None = None          # invention result dict
    exact_points: int = 0                 # verified domain points
    reified_op: str | None = None         # op name if adopted
    notes: list[str] = field(default_factory=list)


def symbolic_phase(source: str, xs: list[int], ys: list[int],
                   full_domain: list[int] | None = None,
                   rng: random.Random | None = None,
                   gap_buffer: GapBuffer | None = None,
                   max_attempts: int = 400) -> PipelineResult:
    """Diagnose -> invent -> gate -> adopt for ONE domain's residuals.

    xs/ys: the byte-level relation the LM could NOT compress (the
    gap surface). full_domain: the certified support (default: xs).
    """
    rng = rng or random.Random(7)
    full_domain = full_domain or sorted(set(xs))
    prof = build_residual_profile([(x,) for x in xs], ys)
    reason = classify_hold("NO_VALIDATED_PROGRAM", prof)
    res = PipelineResult(source=source, holdout_ce=-1.0, hold_reason=reason)

    if reason == HoldReason.RANDOM_INCOMPRESSIBLE:
        res.notes.append("noise: no structure to invent")
        return res
    if gap_buffer is not None:
        gap_buffer.add(GapRecord(task_name=source, reason=reason,
                                 profile=prof))

    pairs = sorted(zip(xs, ys))
    result = invent_blueprint([pairs], rng, max_attempts=max_attempts)
    if not result or not result.get("blueprints"):
        # WAVE 187: second path — Active CEGIS (query-driven proof).
        # The oracle is the relation itself (xs->ys map); budget
        # honest: it probes only the domain we already possess.
        oracle_map = dict(pairs)
        oracle = lambda x: oracle_map.get(x)
        try:
            cegis = cegis_invent(sorted(set(xs)), oracle,
                                 cegis_pool=sorted(set(xs))[:64],
                                 full_domain=full_domain,
                                 max_rounds=4, query_budget=400,
                                 rng=rng)
        except Exception as e:
            res.notes.append(f"cegis error: {e}")
            return res
        res.invented = {"status": cegis.status, "path": "cegis",
                        "queries": cegis.queries}
        if cegis.status != "PROVEN_EXACT" or cegis.blueprint is None:
            res.notes.append(f"cegis: {cegis.status}")
            return res
        bp = cegis.blueprint
        sweep = {"status": "PROVEN_EXACT"}  # cert-phase already swept
    else:
        bp = result.get("blueprints", [None])[0]
        res.invented = {k: result[k] for k in
                        ("status", "q_name", "q_param", "verdict")}
        sweep = verify_blueprint(bp, pairs, full_domain)
    if sweep.get("status") != "PROVEN_EXACT":
        res.notes.append(f"gate: {sweep.get('status')}")
        return res
    res.exact_points = len(full_domain)

    # adoption: reify the exact construct so later tasks reuse it
    q_name = (result or {}).get("q_name") or bp.name
    fn = getattr(bp, "fn", None) or (lambda x: bp.apply(x))
    try:
        op = reify_construct(bp, f"{source}_{q_name}", fn,
                             arity=1, domain=full_domain)
        res.reified_op = op
    except ValueError as e:
        res.notes.append(f"reify: {e}")
    return res


def lm_residual_relation(u, source: str, data: bytes
                         ) -> tuple[list[int], list[int]]:
    """WAVE 196 FIX (P1): the symbolic phase previously used ADJACENT
    byte pairs (x_i -> x_{i+1}) — a law of the data, NOT of the LM.
    Now: sample real LM errors — contexts where the model's argmax
    prediction missed the actual next byte. The relation handed to
    invention is (context -> actual byte) on the model's OWN gaps.
    """
    import torch as _t
    model = u.model
    dev = getattr(model, "device", "cpu")
    idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    n = len(idx) - u.k
    if n <= 0:
        return [], []
    misses_x: list[int] = []
    misses_y: list[int] = []
    step = max(1, n // 512)          # sample, bounded work
    with _t.no_grad():
        for i in range(0, n, step):
            ctx = idx[i:i + u.k]
            nxt = int(idx[i + u.k])
            x = _t.as_tensor([ctx.tolist()], dtype=_t.long, device=dev)
            try:
                logits = model.logits(x)[0]
                pred = int(logits.argmax())
            except Exception:
                continue
            if pred != nxt:
                # WAVE 235b (external audit P5 + user follow-up): the
                # symbolic layer synthesizes arity-1 laws, so each
                # miss is offered from TWO context positions: the last
                # byte (ctx[-1]) and the previous byte (ctx[-2]).
                # A PROVEN_EXACT verdict certifies the corresponding
                # single-position law — NOT the full k-context law.
                # Laws reachable only from earlier context positions
                # remain an honest HOLD until the arity-k
                # synthesizer lands.
                misses_x.append(int(ctx[-1]))
                misses_y.append(nxt)
                if len(ctx) >= 2:
                    misses_x.append(int(ctx[-2]))
                    misses_y.append(nxt)
    return misses_x, misses_y


def ingest_symbolic(u, source: str, data: bytes,
                    gap_buffer: GapBuffer | None = None) -> PipelineResult:
    """Symbolic half of an ingest: the gap relation is the LM's OWN
    holdout misses (P1 fix), then diagnose->invent->adopt."""
    holdout_ce = u.domains.get(source, {}).get("ce")
    res = PipelineResult(source=source,
                         holdout_ce=holdout_ce if holdout_ce is not None else -1.0)
    xs, ys = lm_residual_relation(u, source, data)
    if not xs:
        res.notes.append("no LM residuals sampled")
        return res
    return symbolic_phase(source, xs, ys,
                          full_domain=sorted(set(xs)),
                          gap_buffer=gap_buffer)


def unified_ingest(u, source: str, data: bytes, epochs: int = 200,
                   repeat: int = 2, replay_cap: int | None = None,
                   gap_buffer: GapBuffer | None = None) -> dict:
    """The ONE entry point: neural + symbolic halves in sequence.
    Drop-in for Unified.ingest when HAGI_SYMBOLIC=1."""
    r = u.ingest(source, data, epochs=epochs, repeat=repeat,
                 replay_cap=replay_cap)
    if os.environ.get("HAGI_SYMBOLIC", "0") != "1":
        r["symbolic"] = None
        return r
    sym = ingest_symbolic(u, source, data, gap_buffer=gap_buffer)
    r["symbolic"] = {"hold_reason": sym.hold_reason.name if sym.hold_reason else None,
                     "invented": sym.invented,
                     "exact_points": sym.exact_points,
                     "reified_op": sym.reified_op,
                     "notes": sym.notes}
    return r
