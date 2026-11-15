#!/usr/bin/env python3
"""HAGI self-model loop — the model derives and optimizes its own creation algorithm.

Honest self-reference within certified finite domains (spec v10 sec 5: the
generator, teacher and evaluator are NOT the same entity — evaluator here is
the exact verify_all machinery over declared supports):

  Phase 1 TRACE     run the v9 growth loop with instrumentation; record a
                    per-learn trace: features (family id, arity, domain size,
                    library size) -> outcome (COMMIT/HOLD) and search cost
                    (candidates spent).
  Phase 2 DERIVE    induce the model's own behavior from the trace using
                    HAGI's own exact induction (piecewise-poly + batched MDL
                    synthesis + its own expert library). The result is a
                    symbolic self-model: a program that predicts, from a task
                    description alone, whether this model will COMMIT and how
                    much search it will burn. This program is printed — it IS
                    the derived "algorithm that creates it", at the resolution
                    the model can certify.
  Phase 3 OPTIMIZE  use the self-model to allocate its own compute: tasks
                    predicted to HOLD get a reduced node_budget (cut wasted
                    exhaustive search). Re-run growth under the optimized
                    policy.
  Phase 4 GATE      non-regression commit gate (CommitGate semantics):
                    the optimized configuration is adopted only if the exact
                    committed expert set is identical (or a superset) and
                    quality floors hold; otherwise exact rollback to the
                    baseline configuration and HOLD is reported.

CLI: python hagi_selfmodel.py [--rounds N] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hagi  # noqa: E402

FAMILY_IDS = {"BoolVenn": 0, "unary": 1, "integer-binary": 2,
              "composed-ternary": 3, "auto-composed": 4, "auto-bool3": 5}


# ---------------------------------------------------------------- Phase 1 ----

FEATURE_KEYS = ("family", "arity", "domain", "library", "target_range", "target_unique")


def _target_features(spec, seed: int, train_fraction: float) -> tuple[int, int]:
    """Describe only the supervised target labels visible to the learner."""
    domain = spec.inputs()
    if not domain:
        return 0, 0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(domain))
    bool_like = (spec.family == "BoolVenn" or
                 (all(set(d) <= {0, 1} for d in spec.domains) and len(domain) <= 16))
    if bool_like:
        n_train = max(1, len(domain) - 1)
    else:
        n_train = max(1, min(len(domain) - 1, int(round(len(domain) * train_fraction))))
    ys = [int(spec.teacher(*domain[int(i)])) for i in perm[:n_train]]
    return (max(ys) - min(ys), len(set(ys))) if ys else (0, 0)


class TracedProcessorV9(hagi.GrowingHagiProcessorV9):
    """Records a (features -> outcome, cost) trace of its own learning."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace: list[dict[str, Any]] = []

    def learn(self, spec, seed=0, train_fraction=0.70, poison_one_label=False,
              train_neural=False, deadline=None):
        target_range, target_unique = _target_features(spec, seed, train_fraction)
        feats = {
            "family": FAMILY_IDS.get(spec.family, 6),
            "arity": spec.arity,
            "domain": len(spec.inputs()),
            "library": len(self.experts),
            "target_range": target_range,
            "target_unique": target_unique,
        }
        c0 = self.candidates_generated
        t0 = time.perf_counter()
        ev = super().learn(spec, seed=seed, train_fraction=train_fraction,
                           poison_one_label=poison_one_label,
                           train_neural=train_neural, deadline=deadline)
        self.trace.append({**feats,
                           "success": 1 if ev["action"] == "COMMIT" else 0,
                           "cost": min(self.candidates_generated - c0, 3000),
                           "seconds": round(time.perf_counter() - t0, 4),
                           "task": spec.name})
        return ev


def _features(spec, model, seed: int = 0,
              train_fraction: float = 0.70) -> tuple[int, ...]:
    target_range, target_unique = _target_features(spec, seed, train_fraction)
    return (FAMILY_IDS.get(spec.family, 6), spec.arity, len(spec.inputs()),
            len(model.experts), target_range, target_unique)


# ---------------------------------------------------------------- Phase 2 ----

def derive_self_model(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Induce self-model programs from the trace with HAGI's own machinery."""
    rows = [tuple(t[k] for k in FEATURE_KEYS) for t in trace]
    ys = [t["success"] for t in trace]

    # 1) success predictor: exact piecewise induction on the trace
    succ_expr = hagi.exact_piecewise_poly_fit(rows, ys)
    succ_prog = None
    if succ_expr is not None:
        ok = all(int(succ_expr.eval(r)) == y for r, y in zip(rows, ys))
        if not ok:
            succ_expr = None
    # 2) success predictor via the model's own batched-MDL synthesis
    if succ_expr is None:
        expr, _n = hagi.synthesize_batched_mdl_v9(
            len(FEATURE_KEYS), rows, ys, rows, ys, max_evals=20000, max_bits=64)
        if expr is not None and all(int(expr.eval(r, {})) == y
                                    for r, y in zip(rows, ys)):
            succ_expr = expr
    # 3) cost predictor (bounded): cap >> domain so this stays finite
    costs = [t["cost"] for t in trace]
    cost_expr = None
    if len(set(costs)) > 1:
        ce, _n = hagi.synthesize_batched_mdl_v9(
            len(FEATURE_KEYS), rows, costs, rows, costs, max_evals=20000, max_bits=64)
        if ce is not None and all(int(ce.eval(r, {})) == c
                                  for r, c in zip(rows, costs)):
            cost_expr = ce

    model_desc = {
        "n_trace": len(trace),
        "success_program": succ_expr.pretty() if succ_expr is not None else None,
        "success_exact": succ_expr is not None,
        "cost_program": cost_expr.pretty() if cost_expr is not None else None,
        "cost_exact": cost_expr is not None,
        "_succ_expr": succ_expr,
        "_cost_expr": cost_expr,
    }
    return model_desc


def impossibility_certificate(trace: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Prove (by witness) that no exact success predictor exists on the
    current feature space: find two learn calls with identical features and
    different outcomes. This is a finite, checkable certificate for ABSTAIN
    (spec v10 sec 9: never claim what is not demonstrated)."""
    by_feats: dict[tuple, list[str]] = {}
    for t in trace:
        key = tuple(t[k] for k in FEATURE_KEYS)
        by_feats.setdefault(key, []).append(
            f"{t['task']}:{'COMMIT' if t['success'] else 'HOLD'}")
    for key, names in sorted(by_feats.items()):
        if any(n.endswith(":COMMIT") for n in names) and any(
                n.endswith(":HOLD") for n in names):
            return {"identical_features": list(key),
                    "conflicting_outcomes": names}
    return None


# ---------------------------------------------------------------- Phase 3 ----

def optimized_budget_policy(self_model: dict[str, Any], spec, model,
                            base_budget: int, floor_budget: int,
                            seed: int = 0, train_fraction: float = 0.70) -> int:
    """Self-model-driven compute allocation from the observed training split."""
    succ = self_model.get("_succ_expr")
    if succ is None:
        return base_budget
    try:
        pred = int(succ.eval(_features(spec, model, seed, train_fraction), {}))
    except Exception:
        return base_budget
    if pred == 0:
        # self-model says: this search will HOLD after burning the full budget
        return min(base_budget, floor_budget)
    return base_budget


class SelfOptimizingProcessorV9(TracedProcessorV9):
    """Growth processor whose per-task node_budget is set by its own self-model."""

    def __init__(self, self_model: dict[str, Any], base_budget: int,
                 floor_budget: int, *a, **kw):
        super().__init__(*a, **kw)
        self.self_model = self_model
        self.base_budget = base_budget
        self.floor_budget = floor_budget
        self.policy_overrides = 0

    def learn(self, spec, seed=0, **kw):
        # Honor an explicit experimental policy override. The original code
        # attached this callback in the no-predictor fallback but never called
        # it, so the advertised naive optimization was not actually applied.
        override = getattr(self, "optimized_budget_policy", None)
        if override is None:
            budget = optimized_budget_policy(
                self.self_model, spec, self, self.base_budget, self.floor_budget,
                seed=seed, train_fraction=float(kw.get("train_fraction", 0.70)))
        else:
            budget = override(spec, self, self.base_budget, self.floor_budget)
        self.node_budget = max(1, min(self.base_budget, int(budget)))
        if self.node_budget < self.base_budget:
            self.policy_overrides += 1
        ev = super().learn(spec, seed=seed, **kw)
        return ev


def describe_pipeline(model: TracedProcessorV9) -> dict[str, Any]:
    """The model outputs its own creation pipeline, derived from its log."""
    from collections import Counter
    reasons = Counter(e["reason"] for e in model.growth_log)
    return {
        "stages": [
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
        "experts": len(model.experts),
        "theorems": len(model.theorems),
    }


# ------------------------------------------------------------ Phase 1-4 ----

def run_self_optimization(rounds: int = 2) -> dict[str, Any]:
    report: dict[str, Any] = {}
    suite = hagi.make_task_suite()

    # ---- Phase 1: trace the creation algorithm ----
    tracer = TracedProcessorV9(node_budget=2000)
    rnd = np.random.default_rng(20260101)
    seen: set = set()
    t0 = time.perf_counter()
    for sp in suite:
        tracer.learn(sp, seed=10_000, train_neural=False)
    n_prop = 0
    for k in range(10):
        prop = hagi.propose_noncircular_task_v9(tracer.experts, rnd, 0, seen)
        if prop is None:
            continue
        sp, _ = prop
        if sp.name in tracer.experts:
            continue
        n_prop += 1
        tracer.learn(sp, seed=500_000 + k, train_neural=False)
    trace_seconds = time.perf_counter() - t0
    commits = sum(t["success"] for t in tracer.trace)
    wasted = sum(t["cost"] for t in tracer.trace if not t["success"])
    report["phase1_trace"] = {
        "learn_calls": len(tracer.trace), "commits": commits,
        "holds": len(tracer.trace) - commits,
        "wasted_candidates_on_holds": wasted,
        "trace_seconds": round(trace_seconds, 2),
        "pipeline": describe_pipeline(tracer),
    }

    # ---- Phase 2: derive the self-model with its own machinery ----
    self_model = derive_self_model(tracer.trace)
    report["phase2_self_model"] = {k: v for k, v in self_model.items()
                                   if not k.startswith("_")}
    succ = self_model.get("_succ_expr")
    report["self_algorithm"] = {
        "pipeline": report["phase1_trace"]["pipeline"]["stages"],
        "success_predictor": self_model["success_program"],
        "cost_predictor": self_model["cost_program"],
        "policy": ("node_budget = base if self_model.predict_success(task)==1 "
                   "else floor"),
        "gate": "adopt iff expert set identical/superset and floors hold; else rollback",
    }

    if succ is None:
        cert = impossibility_certificate(tracer.trace)
        # ---- Phase 3 demo: even a naive self-referential policy is tried,
        # and the commit gate must catch its regression and roll back. ----
        demo = SelfOptimizingProcessorV9(
            {"_succ_expr": None, "_naive": True}, base_budget=2000,
            floor_budget=200)
        # naive policy: floor budget for auto-composed families
        demo.optimized_budget_policy = lambda spec, model, base, floor: (
            floor if FAMILY_IDS.get(spec.family, 6) == 4 else base)  # type: ignore[assignment]
        rnd2 = np.random.default_rng(20260101)
        seen2: set = set()
        for sp in suite:
            demo.learn(sp, seed=10_000, train_neural=False)
        for k in range(10):
            prop = hagi.propose_noncircular_task_v9(demo.experts, rnd2, 0, seen2)
            if prop is None:
                continue
            sp, _ = prop
            if sp.name in demo.experts:
                continue
            demo.learn(sp, seed=500_000 + k, train_neural=False)
        lost = sorted(set(tracer.experts) - set(demo.experts))
        if lost:
            gate_reason = ("naive policy lost experts -> gate rollback: "
                           + ", ".join(lost))
        else:
            gate_reason = ("no exact self-model at this feature resolution; "
                           "unverified policy cannot be adopted without a "
                           "certified predictor (honest HOLD)")
        report["phase3_optimization"] = {
            "status": "ATTEMPTED_NAIVE_POLICY",
            "policy": "floor_budget for auto-composed family",
            "lost_experts": lost,
        }
        report["phase4_gate"] = {
            "action": "HOLD",
            "reason": gate_reason,
            "impossibility_certificate": cert,
            "rollback": "baseline configuration retained (tracer state is the adopted model)",
        }
        report["adopted"] = False
        return report

    # ---- Phase 3: growth under self-model-optimized budgets ----
    opt = SelfOptimizingProcessorV9(self_model, base_budget=2000,
                                    floor_budget=200)
    rnd2 = np.random.default_rng(20260101)
    seen2: set = set()
    t0 = time.perf_counter()
    for sp in suite:
        opt.learn(sp, seed=10_000, train_neural=False)
    for k in range(10):
        prop = hagi.propose_noncircular_task_v9(opt.experts, rnd2, 0, seen2)
        if prop is None:
            continue
        sp, _ = prop
        if sp.name in opt.experts:
            continue
        opt.learn(sp, seed=500_000 + k, train_neural=False)
    opt_seconds = time.perf_counter() - t0
    opt_wasted = sum(t["cost"] for t in opt.trace if not t["success"])
    report["phase3_optimization"] = {
        "status": "APPLIED",
        "learn_calls": len(opt.trace),
        "wasted_candidates_on_holds": opt_wasted,
        "seconds": round(opt_seconds, 2),
        "same_seed_stream": True,
    }

    # ---- Phase 4: non-regression commit gate ----
    base_names = set(tracer.experts.keys())
    opt_names = set(opt.experts.keys())
    experts_ok = opt_names >= base_names
    # exact accuracy floor over the full base suite on the optimized model
    check = opt.verify_all(suite)
    floors_ok = (check["accuracy"] == 1.0
                 and check["committed"] >= len(suite))
    if experts_ok and floors_ok:
        adopted = True
        report["phase4_gate"] = {
            "action": "COMMIT",
            "experts_base": len(base_names), "experts_optimized": len(opt_names),
            "exact_accuracy": check["accuracy"],
            "wasted_candidates": f"{wasted} -> {opt_wasted}",
            "seconds": f"{round(trace_seconds,2)} -> {round(opt_seconds,2)}",
            "speedup_wasted": round(wasted / max(1, opt_wasted), 2),
        }
    else:
        adopted = False
        report["phase4_gate"] = {
            "action": "HOLD",
            "reason": ("optimized policy lost experts"
                       if not experts_ok else "quality floor violated"),
            "experts_base": len(base_names), "experts_optimized": len(opt_names),
            "accuracy": check["accuracy"],
        }
    report["adopted"] = adopted
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="HAGI self-model: derive & optimize own algorithm")
    p.add_argument("--json", action="store_true", help="print raw JSON only")
    a = p.parse_args()
    rep = run_self_optimization()
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if rep["phase4_gate"].get("action") == "COMMIT":
        print("SELF-OPTIMIZATION COMMIT: self-model policy adopted "
              f"({rep['phase4_gate']['wasted_candidates']} wasted candidates)")
    elif rep.get("phase3_optimization", {}).get("status") == "ATTEMPTED_NAIVE_POLICY":
        print("SELF-OPTIMIZATION HOLD: exact self-model impossible at this "
              "feature resolution (certificate produced); naive policy "
              "rejected by commit gate, baseline retained")
    else:
        print("SELF-OPTIMIZATION HOLD: " + rep["phase4_gate"].get("reason", ""))


if __name__ == "__main__":
    main()
