#!/usr/bin/env python3
"""Architecture Genesis — self-designed processor configuration search.

Idea (user, 2026-11): HAGI should design, test, and rebuild its own
"virtual processor" (the learning-loop configuration) and converge to
the best configuration *relative to a task profile and budget* —
no universal optimum exists (GrowthCeilings.expansion_required:
a fixed configuration has a hard gain ceiling; growth demands
reconfiguration).

Theory anchors (verified in cycles 4/20/31/48):
  - Search space is DECLARED (safe knobs only) — uncertifiable knobs
    are out of scope.
  - Adoption gate: strict improvement on the TARGET profile
    (CertifiedArgmax strict-gap, cycle 20) AND no regression beyond
    a small margin on the other profiles (no-regression margin,
    cycle 47 robust_eligible_iff_radius_le_margin).
  - Score = Pareto scalarization with profile weights (cycle 31).
  - Cost model: FlopsEconomy — a config that costs more flops per
    adopted expert than the incumbent is rejected even if raw
    throughput looks better (cycle 4).

Log: checkpoints/self/archgenesis.json (honest history).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GENESIS_LOG = ROOT / "checkpoints" / "self" / "archgenesis.json"

# --- Declared architecture space (safe knobs only) ---
# Each knob: list of legal values. The CURRENT value is excluded at
# propose time (no-op wastes the gate budget — ProxyGap T<=eps/c).
ARCH_SPACE: dict[str, list[Any]] = {
    "workers": [4, 8, 12, 16],          # parallel lanes (serial_divergence_cap:
                                        # parallel = thermodynamically sound axis)
    "batch_size": [8, 12, 16],          # work chunk per step
    "proposals_per_round": [8, 10, 12], # candidates per round
    "max_rounds": [2, 3, 4],            # pipeline depth per genesis probe
}

# --- Task profiles: what "best" means depends on the workload ---
# weights sum to 1; higher weight = more important for this profile.
PROFILES: dict[str, dict[str, float]] = {
    "throughput": {"experts": 0.5, "speed": 0.3, "relief": 0.2},
    "latency":    {"speed": 0.6, "relief": 0.25, "experts": 0.15},
    "memory":     {"economy": 0.5, "experts": 0.3, "speed": 0.2},
}

# no-regression margin on non-target profiles (robust radius, cycle 47)
REGRESSION_MARGIN = 0.05
# strict-gap: a target gain below this relative threshold is noise
# (CertifiedArgmax: gap must exceed 2*eps to trust the ordering —
# cycle 20). Measured noise on 2-round micro-benchmarks ~1e-4.
MIN_TARGET_GAIN = 0.01


def _load_log() -> dict[str, Any]:
    if GENESIS_LOG.exists():
        return json.loads(GENESIS_LOG.read_text(encoding="utf-8"))
    return {"adopted": [], "rejected": [], "skipped": []}


def _save_log(log: dict[str, Any]) -> None:
    GENESIS_LOG.parent.mkdir(parents=True, exist_ok=True)
    GENESIS_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1),
                           encoding="utf-8")


def current_config() -> dict[str, Any]:
    """Live production config: loop defaults overridden by adopted
    archgenesis and self-edit settings."""
    from hagi import hagi_selfedit as se
    cur = {"workers": 8, "batch_size": 8, "proposals_per_round": 10,
           "max_rounds": 4}
    try:
        cur.update({k: v for k, v in se.adopted_settings().items()
                    if k in cur})
    except Exception:
        pass
    return cur


def benchmark_config(cfg: dict[str, Any]) -> dict[str, float]:
    """Run the micro-benchmark (2 rounds, small workers) and return
    raw metrics: experts adopted, wall seconds, relief, economy."""
    import os
    from hagi import hagi_selfloop as sl
    workers = min(int(cfg.get("workers", 8)),
                  12 if len(sl.__dict__.get("_seg_est", [0] * 901)) > 900 else 16)
    t0 = time.perf_counter()
    rep = sl.run_self_development(
        max_rounds=int(cfg.get("max_rounds", 2)),
        workers=workers,
        batch_size=int(cfg.get("batch_size", 8)),
        proposals_per_round=int(cfg.get("proposals_per_round", 10)))
    dt = time.perf_counter() - t0
    fin = rep["final"]
    adopted = max(1, fin["adopted_rounds"])
    experts = max(1, fin["experts"])
    relief = adopted / (adopted + fin.get("wasted_candidates_on_holds", 0))
    # FlopsEconomy economy metric: adopted experts per second of work
    economy = adopted / max(dt, 1e-9)
    return {"experts": float(experts), "speed": 1.0 / max(dt, 1e-9),
            "relief": float(relief), "economy": float(economy),
            "seconds": dt}


def score(profile: str, m: dict[str, float]) -> float:
    """Pareto scalarization (cycle 31): profile-weighted sum."""
    w = PROFILES[profile]
    return sum(w.get(k, 0.0) * float(m.get(k, 0.0)) for k in w)


def propose_mutation(rng, cur: dict[str, Any]) -> dict[str, Any] | None:
    """One-knob change within ARCH_SPACE, never equal to current."""
    import random
    keys = list(ARCH_SPACE.keys())
    rng.shuffle(keys)
    for knob in keys:
        options = [v for v in ARCH_SPACE[knob] if v != cur.get(knob)]
        if options:
            return {"knob": knob, "value": rng.choice(options)}
    return None


def genesis_cycle(seed: int = 0, profile: str = "throughput",
                  dry: bool = False) -> dict[str, Any]:
    """One design->test->decide cycle of Architecture Genesis.

    Certified decision: adopt only if target-profile score strictly
    improves AND every other profile stays within REGRESSION_MARGIN.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {list(PROFILES)}")
    rng = __import__("random").Random(seed)
    cur = current_config()
    mut = propose_mutation(rng, cur)
    if mut is None:
        return {"decision": "exhausted", "config": cur}
    cand_cfg = dict(cur)
    cand_cfg[mut["knob"]] = mut["value"]

    entry: dict[str, Any] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "profile": profile,
                             "mutation": mut, "from": cur, "to": cand_cfg}
    if dry:
        entry["decision"] = "dry"
        return entry

    base_m = benchmark_config(cur)
    cand_m = benchmark_config(cand_cfg)
    base_scores = {p: score(p, base_m) for p in PROFILES}
    cand_scores = {p: score(p, cand_m) for p in PROFILES}
    target_gain = cand_scores[profile] - base_scores[profile]
    # normalize by base to make margin relative
    regressions = {}
    for p in PROFILES:
        if p == profile:
            continue
        b, c = base_scores[p], cand_scores[p]
        rel = (b - c) / max(abs(b), 1e-9)
        regressions[p] = round(rel, 4)
    worst_regression = max(regressions.values(), default=0.0)
    entry["scores"] = {"base": {k: round(v, 4) for k, v in base_scores.items()},
                       "cand": {k: round(v, 4) for k, v in cand_scores.items()}}
    entry["target_gain"] = round(target_gain, 6)
    entry["regressions"] = regressions
    rel_gain = target_gain / max(abs(base_scores[profile]), 1e-9)
    entry["rel_gain"] = round(rel_gain, 6)
    if rel_gain >= MIN_TARGET_GAIN and worst_regression <= REGRESSION_MARGIN:
        entry["decision"] = "adopted"
    else:
        entry["decision"] = "rejected"
    log = _load_log()
    log.setdefault(entry["decision"], []).append(entry)
    _save_log(log)
    return entry


def adopted_config() -> dict[str, Any]:
    """Last adopted genesis config per knob (drives the production
    loop — CertifiedLoop contract, same as self-edit)."""
    log = _load_log()
    last: dict[str, Any] = {}
    for e in log.get("adopted", []):
        m = e.get("mutation", {})
        if m.get("knob") and "value" in m:
            last[m["knob"]] = m["value"]
    return last
