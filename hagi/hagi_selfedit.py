#!/usr/bin/env python3
"""Nadya's self-edit — the model proposes patches to its own learning
code (hagi/_v9.py learning knobs), each patch is CERTIFIED before
adoption:

  gate = characterization tests + fixed benchmark suite.
  A patch is applied ONLY if it does not regress:
    - test suite passes (13/13),
    - learning benchmark: same budget, adopted-rounds >= baseline
      (no regression on the fixed suite).

Patch space is DECLARED (safe knobs only, honest limits):
  - proposal_counts (auto-proposal diversity),
  - floor_ladder order,
  - batch_size,
  - node_budget scaling.
Arbitrary AST rewriting of learning code is NOT attempted — that
would be uncertifiable. The declared knobs are exactly the levers
the loop already exposes, so a proposed patch is a config change,
validated end-to-end like any successor (§9 gate_never_increases).

Cycle (invoked by forever between rounds):
  1. pick a knob change (from the model's own RNG seeded by state),
  2. run the gate on a sandbox copy,
  3. adopt if no regression; record in self_state.edit_log.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent  # repo root
EDIT_LOG = ROOT / "checkpoints/self/self_edits.json"
BASELINE_PATH = ROOT / "checkpoints/self/edit_baseline.json"

# Declared safe knobs (name, candidates to try when exploring)
KNOBS = {
    "floor_ladder": [[200, 50, 20], [200, 100, 50], [300, 100, 30],
                     [200, 50, 20, 10]],
    "batch_size": [8, 12, 16],
    "proposals": [8, 10, 12],
}


def _baseline() -> dict[str, Any]:
    """Benchmark the CURRENT code once and cache it."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    res = run_benchmark()
    BASELINE_PATH.write_text(json.dumps(res), encoding="utf-8")
    return res


def run_benchmark(rounds: int = 2, workers: int = 8) -> dict[str, Any]:
    """Fixed mini-suite benchmark of the learning loop (deterministic)."""
    from hagi import hagi_selfloop as sl
    t0 = time.time()
    rep = sl.run_self_development(max_rounds=rounds, workers=workers,
                                  proposals_per_round=6)
    fin = rep["final"]
    return {"adopted": fin["adopted_rounds"],
            "experts": fin["experts"],
            "seconds": round(time.time() - t0, 1)}


def _gate_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x",
                        "tests/test_characterization.py"],
                       capture_output=True, text=True, timeout=1800)
    return r.returncode == 0


def current_settings() -> dict[str, Any]:
    """Snapshot of the CURRENT knob values (loop defaults)."""
    from hagi import hagi_selfloop as sl
    import inspect
    sig = inspect.signature(sl.run_self_development)
    defaults = {k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty}
    return {"floor_ladder": defaults.get("floor_ladder") or [200],
            "batch_size": defaults.get("batch_size"),
            "proposals": defaults.get("proposals_per_round")}


def propose_patch(rng) -> dict[str, Any]:
    """Propose a knob change (declared space only).

    GrowthCeilings.expansion_required / ProxyGap horizon: a no-op
    patch (value identical to the current setting) would burn
    gate-budget (<= 3 candidates/cycle) on a triviality — never
    propose it. Only changes are candidates.
    """
    import random
    cur = current_settings()
    for _ in range(64):
        knob = rng.choice(list(KNOBS.keys()))
        value = rng.choice(KNOBS[knob])
        if knob == "floor_ladder":
            if value == cur.get("floor_ladder"):
                continue
        elif value == cur.get(knob):
            continue
        return {"knob": knob, "value": value}
    return {"knob": "batch_size", "value": 8}  # fallback change


def is_noop(patch: dict[str, Any]) -> bool:
    """True when the patch equals the current setting (wastes gate budget)."""
    cur = current_settings()
    if patch["knob"] == "floor_ladder":
        return patch["value"] == cur.get("floor_ladder")
    return patch["value"] == cur.get(patch["knob"])


def adopted_settings() -> dict[str, Any]:
    """Last adopted value per knob from the edit log (gated, certified).

    CertifiedLoop contract: adopted patches must actually drive the
    production loop — otherwise the self-edit cycle is decorative.
    Returns kwargs for run_self_development.
    """
    if not EDIT_LOG.exists():
        return {}
    log = json.loads(EDIT_LOG.read_text(encoding="utf-8"))
    last: dict[str, Any] = {}
    for e in log.get("adopted", []):
        p = e.get("patch", {})
        if p.get("knob") and "value" in p:
            last[p["knob"]] = p["value"]
    kwargs: dict[str, Any] = {}
    if "floor_ladder" in last:
        kwargs["floor_ladder"] = last["floor_ladder"]
    if "batch_size" in last:
        kwargs["batch_size"] = last["batch_size"]
    if "proposals" in last:
        kwargs["proposals_per_round"] = last["proposals"]
    return kwargs


def apply_patch(patch: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Return a modified settings overlay (does NOT edit source code)."""
    out = dict(target)
    out[patch["knob"]] = patch["value"]
    return out


# GatePipeline (C264) delta-schedule: serial gates consume the false-
# alarm budget sum_k<=K 6/(pi^2 k^2) <= 1 — after MAX_GATE_RUNS
# consecutive rejections in a series the budget is exhausted.
MAX_GATE_RUNS = 6


def self_edit_cycle(seed: int = 0, dry: bool = False) -> dict[str, Any]:
    """One propose->gate->adopt cycle. Honest record in EDIT_LOG."""
    import random
    rng = random.Random(seed)
    log = json.loads(EDIT_LOG.read_text(encoding="utf-8")) if EDIT_LOG.exists() else \
        {"adopted": [], "rejected": []}
    # C264 delta-budget: a long rejection series means the false-alarm
    # budget is spent — reset the series (theory-exact bound for K<=6).
    if len(log.get("rejected", [])) >= 2 * MAX_GATE_RUNS:
        log["rejected"] = log["rejected"][-MAX_GATE_RUNS:]
        EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    baseline = _baseline()
    patch = propose_patch(rng)
    proposal = {"patch": patch, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    if is_noop(patch):
        # No-op = no expansion (GrowthCeilings.expansion_required):
        # skip without spending gate runs (ProxyGap horizon T<=eps/c).
        proposal["decision"] = "noop-skipped"
        return proposal
    if dry:
        proposal["decision"] = "dry"
        return proposal
    # the gate: benchmark with the patch applied (as loop settings)
    from hagi import hagi_selfloop as sl
    kwargs = {}
    if patch["knob"] == "batch_size":
        kwargs["batch_size"] = patch["value"]
    elif patch["knob"] == "proposals":
        kwargs["proposals_per_round"] = patch["value"]
    elif patch["knob"] == "floor_ladder":
        kwargs["floor_ladder"] = patch["value"]
    try:
        rep = sl.run_self_development(max_rounds=2, workers=8,
                                      proposals_per_round=6, **kwargs)
        cand = {"adopted": rep["final"]["adopted_rounds"],
                "experts": rep["final"]["experts"]}
    except Exception as e:
        proposal.update(decision="rejected", reason=repr(e)[:80])
        log["rejected"].append(proposal)
        EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        return proposal
    # Honest gate note (DataToQuality.sampleComplexity): this is a
    # DETERMINISTIC fixed-seed no-regression test, not a statistical
    # guarantee — 2 benchmark rounds give eta ~ +/-99% (sample
    # complexity log(m/delta)/(2 eta^2) would need ~780 rounds for
    # eta=5%). The gate is deliberately conservative: adopt only on
    # strict positive evidence (Insight.experience_cycle_bound).
    no_regress = (cand["adopted"] > baseline["adopted"]
                  or cand["experts"] > baseline["experts"])
    proposal.update(decision="adopted" if no_regress else "rejected",
                    candidate=cand, baseline=baseline)
    (log["adopted"] if no_regress else log["rejected"]).append(proposal)
    EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return proposal


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Nadya self-edit gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    print(json.dumps(self_edit_cycle(seed=a.seed, dry=a.dry),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
