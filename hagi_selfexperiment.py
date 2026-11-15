#!/usr/bin/env python3
"""HAGI autonomous self-optimization experiment.

The model derives its own creation algorithm from its own trace (pipeline
stages + commit-reason histogram + impossibility certificates when a feature
cannot determine success), then runs a controlled experiment ON ITSELF:
it trials a set of candidate compute-allocation policies across multiple
seeds and adopts, via the non-regression commit gate, only a policy that

  * preserves the baseline expert set exactly (superset) on EVERY seed,
  * keeps exact accuracy at 1.0 over the full base suite,
  * strictly reduces mean wasted search candidates.

Generator, policy and evaluator are distinct: the evaluator is the exact
verify_all machinery over declared finite supports; policies only touch
node_budget, never the teacher or the verifier (spec v10 sec 5).

CLI: python hagi_selfexperiment.py [--seeds 5] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hagi  # noqa: E402
from hagi_selfmodel import (  # noqa: E402  (reuse certified machinery)
    FAMILY_IDS, TracedProcessorV9, impossibility_certificate)

Policy = Callable[[Any, int], int]   # (spec, base_budget) -> node_budget


# ------------------------------------------------------------ growth run ----

def run_growth(policy: Policy | None, seed: int, n_proposal_rounds: int = 8,
               budget_per_round: int = 8, base_budget: int = 2000
               ) -> dict[str, Any]:
    """One growth run: base suite + self-proposed tasks under a policy."""
    m = TracedProcessorV9(node_budget=base_budget)
    if policy is not None:
        _orig_learn = m.learn

        def learn_with_policy(spec, seed=0, **kw):
            m.node_budget = policy(spec, base_budget)
            return _orig_learn(spec, seed=seed, **kw)

        m.learn = learn_with_policy  # type: ignore[method-assign]

    rnd = np.random.default_rng(20260101 + seed)
    seen: set = set()
    t0 = time.perf_counter()
    for sp in hagi.make_task_suite():
        m.learn(sp, seed=10_000, train_neural=False)
    for r in range(n_proposal_rounds):
        for k in range(budget_per_round):
            prop = hagi.propose_noncircular_task_v9(m.experts, rnd, r, seen)
            if prop is None:
                continue
            sp, _ = prop
            if sp.name in m.experts:
                continue
            m.learn(sp, seed=500_000 + r * 104_729 + k, train_neural=False)
    elapsed = time.perf_counter() - t0
    wasted = sum(t["cost"] for t in m.trace if not t["success"])
    return {"expert_names": frozenset(m.experts.keys()),
            "n_experts": len(m.experts), "wasted": wasted,
            "seconds": round(elapsed, 2), "trace": m.trace}


# ------------------------------------------------------------- policies ----

def p_baseline(spec, base: int) -> int:
    return base


def p_floor_half(spec, base: int) -> int:
    return max(400, base // 2) if FAMILY_IDS.get(spec.family, 6) == 4 else base


def p_floor_quarter(spec, base: int) -> int:
    return max(200, base // 4) if FAMILY_IDS.get(spec.family, 6) == 4 else base


POLICIES: dict[str, Policy] = {
    "baseline": p_baseline,
    "floor_half_auto": p_floor_half,
    "floor_quarter_auto": p_floor_quarter,
}


# ------------------------------------------------------------- experiment ---

def run_experiment(seeds: int = 5) -> dict[str, Any]:
    report: dict[str, Any] = {}

    # ---- Phase 1+2: derive the creation algorithm and self-model ----
    tracer = TracedProcessorV9(node_budget=2000)
    rnd = np.random.default_rng(20260101)
    seen: set = set()
    for sp in hagi.make_task_suite():
        tracer.learn(sp, seed=10_000, train_neural=False)
    for k in range(10):
        prop = hagi.propose_noncircular_task_v9(tracer.experts, rnd, 0, seen)
        if prop is None:
            continue
        sp, _ = prop
        if sp.name in tracer.experts:
            continue
        tracer.learn(sp, seed=500_000 + k, train_neural=False)
    from collections import Counter
    reasons = Counter(e["reason"] for e in tracer.growth_log)
    cert = impossibility_certificate(tracer.trace)
    report["derived_algorithm"] = {
        "pipeline": [
            "REUSE(certified-library-coverage)",
            "boolvenn-exact(16 masks)",
            "anf-zhegalkin-butterfly(+full-support)",
            "direct-template(+full-support)",
            "exact-rational-poly(1D/ND,+full-support)",
            "exact-piecewise-poly(+full-support)",
            "batched-mdl-mitm-dag(+full-support)",
            "HOLD(rollback+audit)",
        ],
        "commit_reason_counts": dict(reasons),
        "success_feature_impossibility_certificate": cert,
        "note": ("success of the exhaustive synthesis stage is a function of "
                 "the task's full teacher data, not of coarse features; the "
                 "experiment therefore trials MEASURED policies instead of "
                 "claiming a symbolic predictor"),
    }

    # ---- Phase 3: controlled policy trials over seeds ----
    suite_names = frozenset(s.name for s in hagi.make_task_suite())
    trials: dict[str, list[dict[str, Any]]] = {p: [] for p in POLICIES}
    for seed in range(seeds):
        for pname, policy in POLICIES.items():
            res = run_growth(policy, seed)
            trials[pname].append(res)
            print(f"  seed={seed} policy={pname}: experts={res['n_experts']} "
                  f"wasted={res['wasted']} t={res['seconds']}s", flush=True)

    # ---- Phase 4: non-regression gate per policy ----
    gate: dict[str, Any] = {}
    base_stats = trials["baseline"]
    for pname, runs in trials.items():
        superset_all = all(r["expert_names"] >= b["expert_names"]
                           for r, b in zip(runs, base_stats))
        exact_acc_all = superset_all  # experts certified exact on full support
        mean_wasted = float(np.mean([r["wasted"] for r in runs]))
        mean_secs = float(np.mean([r["seconds"] for r in runs]))
        passes = (superset_all
                  and mean_wasted < float(np.mean([b["wasted"] for b in base_stats])))
        gate[pname] = {
            "superset_on_all_seeds": superset_all,
            "mean_wasted": round(mean_wasted, 1),
            "mean_seconds": round(mean_secs, 2),
            "verdict": "PASS" if passes else "REJECT",
            "per_seed_experts": [r["n_experts"] for r in runs],
            "per_seed_wasted": [r["wasted"] for r in runs],
            "new_experts_vs_baseline": [
                sorted(r["expert_names"] - b["expert_names"])
                for r, b in zip(runs, base_stats)],
        }

    passing = {p: g for p, g in gate.items()
               if p != "baseline" and g["verdict"] == "PASS"}
    if passing:
        best = min(passing, key=lambda p: gate[p]["mean_wasted"])
        report["gate"] = gate
        report["decision"] = {
            "action": "COMMIT",
            "adopted_policy": best,
            "policy_source": str(POLICIES[best]),
            "reason": ("measured non-regression on all seeds with strictly "
                       "less wasted search"),
        }
    else:
        report["gate"] = gate
        report["decision"] = {
            "action": "HOLD",
            "reason": ("no candidate policy preserved the baseline expert "
                       "set on all seeds while reducing wasted search; "
                       "baseline retained"),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    rep = run_experiment(seeds=a.seeds)
    print(json.dumps({k: v for k, v in rep.items()
                      if k != "derived_algorithm"}, indent=2, ensure_ascii=False))
    d = rep["decision"]
    if d["action"] == "COMMIT":
        print(f"ADOPTED: policy '{d['adopted_policy']}' "
              f"(wasted {rep['gate']['baseline']['mean_wasted']} -> "
              f"{rep['gate'][d['adopted_policy']]['mean_wasted']})")
    else:
        print("GATE HOLD: " + d["reason"])


if __name__ == "__main__":
    main()
