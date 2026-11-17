#!/usr/bin/env python3
"""HAGI self-development loop — iterated self-model derivation and adoption.

Extends hagi.selfmodel (single-shot TRACE->DERIVE->OPTIMIZE->GATE) with:

  1. An exact decision-list self-model (lookup keyed on full trace
     features; unknown -> conservative default). Exact on the trace by
     construction; holdout-validated (confidently-wrong HOLD predictions
     block adoption).
  2. Multi-round growth: each adopted round enlarges the task library
     with fresh non-circular proposals, re-derives the self-model and
     re-gates. Stops on the first HOLD gate or when the certified relief
     margin (MIN_RELIEF) is not met — the certified-convergence stop
     (theory R223: no certified-positive action = resource exhausted).

Performance (2026-11-15):
  - Replay cache: prefix tasks whose assigned budget is unchanged reuse
    the recorded outcome instead of re-learning -> the per-round full
    replay becomes O(new tasks), the whole loop linear.
  - Parallel batches: proposal tasks are learned in worker processes
    (in-process, deterministic order; parallelism via BLAS/OMP) against
    a snapshot of the library; results merge in stream order.
    Batch semantics: tasks within one batch do not see each other's
    commits. Both incumbent growth and optimized replay use the same
    batching, so streams stay deterministic and comparable.

CLI: python -m hagi.loop [--max-rounds N] [--workers N] [--json]
"""
from __future__ import annotations

import os
# BLAS/OMP threads: with no spawn workers (2026-11-17), all machine
# cores go to the numeric libraries. Only set if the user hasn't.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(os.cpu_count() or 4))

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import hagi
from hagi.selfmodel import (
    FEATURE_KEYS,
    SelfOptimizingProcessorV9,
    TracedProcessorV9,
    describe_pipeline,
    impossibility_certificate,
)

FULL_KEY = FEATURE_KEYS + ("seed",)         # +seed: task-exact key (batch
                                            # semantics make `library` constant
                                            # within a batch — seed disambiguates)
MIN_RELIEF = 0.02                            # certified relief margin per round


# ------------------------------------------------------------------ model ----

class DecisionListSelfModel:
    """Exact-by-construction success/cost predictor over trace features.

    Only exact full-key predictions floor a task: the replay stream is
    deterministic, so keys re-occur verbatim. Anything less (coarse-key
    generalization) once produced a confidently-wrong HOLD and lost an
    expert (round-2 gate reject) — uncertified generalization stays out.
    """

    def __init__(self, trace: list[dict[str, Any]]):
        succ: dict[tuple, set[int]] = {}
        cost: dict[tuple, list[int]] = {}
        for t in trace:
            k = tuple(t[f] for f in FULL_KEY)
            succ.setdefault(k, set()).add(t["success"])
            cost.setdefault(k, []).append(t["cost"])
        self._succ = {k: v.pop() for k, v in succ.items() if len(v) == 1}
        self._ambig = sorted(k for k, v in succ.items() if len(v) > 1)
        self._cost = {k: max(vs) for k, vs in cost.items() if len(succ[k]) == 1}

    def predict_success(self, feats: dict[str, Any]) -> Optional[int]:
        return self._succ.get(tuple(feats[f] for f in FULL_KEY))

    def predict_cost(self, feats: dict[str, Any]) -> Optional[int]:
        return self._cost.get(tuple(feats[f] for f in FULL_KEY))

    @property
    def ambiguous_keys(self) -> list[tuple]:
        return self._ambig

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "exact-decision-list",
            "full_key_rules": len(self._succ),
            "ambiguous_keys": len(self._ambig),
        }


def holdout_validate(trace: list[dict[str, Any]],
                     fraction: float = 0.30) -> dict[str, Any]:
    """Derive on train split, check the dangerous direction on holdout:
    a confident HOLD prediction for a task that would COMMIT blocks
    adoption. Unknown predictions default to base budget (safe; any
    regression they cause is caught by the non-regression gate)."""
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(trace))
    n_hold = max(1, int(len(trace) * fraction))
    hold_idx = set(idx[:n_hold].tolist())
    train = [t for i, t in enumerate(trace) if i not in hold_idx]
    hold = [trace[i] for i in sorted(hold_idx)]
    m = DecisionListSelfModel(train)
    confident_wrong = 0
    unknown_safe = 0
    for t in hold:
        pred = m.predict_success(t)
        truth = t["success"]
        if pred is None:
            if truth == 0:
                unknown_safe += 1
        elif pred != truth:
            confident_wrong += 1
    return {"holdout_size": len(hold), "confident_wrong": confident_wrong,
            "unknown_safe": unknown_safe,
            "exact": confident_wrong == 0}


# ---------------------------------------------------------- policy processor --

class SelfModelPolicyProcessor(SelfOptimizingProcessorV9):
    """Processor whose budget policy is a DecisionListSelfModel."""

    def __init__(self, dl: DecisionListSelfModel, base_budget: int,
                 floor_budget: int, *a, **kw):
        super().__init__({"_succ_expr": None}, base_budget, floor_budget, *a, **kw)
        self.dl = dl

    def learn(self, spec, seed=0, **kw):
        from hagi.selfmodel import _features
        feats: dict[str, Any] = dict(zip(
            FEATURE_KEYS,
            _features(spec, self, seed,
                      float(kw.get("train_fraction", 0.70)))))
        feats["seed"] = int(seed)
        pred = self.dl.predict_success(feats)
        if pred is None:
            pred = 1
        self.node_budget = self.floor_budget if pred == 0 else self.base_budget
        if pred == 0:
            self.policy_overrides += 1
        return super(SelfOptimizingProcessorV9, self).learn(spec, seed=seed, **kw)


# ------------------------------------------------------- parallel machinery ---

def _state_probe(proc) -> dict[str, Any]:
    return {
        "cands": proc.candidates_generated, "evals": proc.evals,
        "commits": proc.total_commits, "holds": proc.total_holds,
        "version": proc.version,
        "unsolved": proc.unsolved_objective,
        "compute": proc.compute_budget, "risk": proc.risk_rem,
        "n_experts": len(proc.experts), "n_log": len(proc.growth_log),
        "n_cert": len(getattr(proc, "cert_log", [])), "n_trace": len(getattr(proc, "trace", [])),
        "ram": proc.ram.snapshot(),
    }


def _learn_worker_local(proc, spec, seed, budget, train_neural):
    """In-process variant of _learn_worker (no pickling — for jobs too
    large to serialize when memory is tight)."""
    os.environ["HAGI_NEURAL_DEVICE"] = "cpu"
    proc.node_budget = int(budget)
    alpha = min(0.0005, proc.ledger.remaining() * 0.5)
    pre = _state_probe(proc)
    ev = proc.learn(spec, seed=int(seed), train_neural=train_neural)
    post = _state_probe(proc)
    names = list(proc.experts.keys())
    new_names = names[pre["n_experts"]:]
    diff = {
        "ev": ev,
        "trace_row": getattr(proc, "trace", [])[pre["n_trace"]:] or None,
        "growth_log": proc.growth_log[pre["n_log"]:],
        "certs": proc.cert_log[pre["n_cert"]:],
        "new_experts": {n: proc.experts[n] for n in new_names},
        "deltas": {k: post[k] - pre[k] for k in
                   ("cands", "evals", "commits", "holds", "version",
                    "unsolved", "compute", "risk")},
        "ram_ops": _ram_diff(pre["ram"], post["ram"]),
        "core": proc.core.snapshot(),
        "router": proc.router.snapshot(),
        "audit": {"task": spec.name, "alpha": alpha},
    }
    return diff




def _ram_diff(pre: np.ndarray, post: np.ndarray) -> list[tuple[int, Any]]:
    ops = []
    for slot in range(post.shape[0]):
        if not np.array_equal(pre[slot], post[slot]):
            ops.append((slot, post[slot].copy()))
    return ops


class ParallelLearner:
    """Order-preserving batch learning in worker processes.

    All tasks in a batch are learned against the same library snapshot;
    their commits are merged into the parent in stream order (batch
    semantics, identical for growth and replay -> deterministic).
    With train_neural=True every learn also fits the neural core
    (FactorizedSharedCoreV9 adapters); core/router states ride the diff
    and are restored in stream order. Exact outcomes are unaffected
    (commit gate uses exact programs only), so determinism holds."""

    def __init__(self, proc, workers: int, train_neural: bool = False):
        self.proc = proc
        self.workers = max(1, workers)
        self.train_neural = bool(train_neural)

    def batch(self, jobs: list[tuple[Any, int, int]]) -> list[dict[str, Any]]:
        """jobs: [(spec, seed, budget)] -> merged diffs in order.

        Threads, not processes (2026-11-17 user decision: no spawn
        workers at all): shared memory, no payload pickling, no
        BrokenProcessPool degradation path. numpy/torch release the
        GIL in the hot paths, so threads parallelize the numeric work.
        Determinism unchanged: jobs are independent, merged in order.
        """
        if not jobs:
            return []
        # NO worker processes/threads pool (user decision 2026-11-17):
        # in-process sequential learning, one shared proc. Parallelism
        # comes from BLAS/OMP (OPENBLAS_NUM_THREADS = all cores) inside
        # numpy/torch ops. Deterministic order, zero race, zero spawn.
        diffs = [_learn_worker_local(self.proc, spec, seed, budget,
                                     self.train_neural)
                 for spec, seed, budget in jobs]
        # NOTE: caller merges diffs in stream order — merging here would
        # double-apply counters/logs/trace (dict updates are idempotent,
        # the rest is not).
        return diffs

    def _merge(self, d: dict[str, Any]) -> None:
        p = self.proc
        p.experts.update(d["new_experts"])
        p.growth_log.extend(d["growth_log"])
        p.cert_log.extend(d["certs"])
        dl = d["deltas"]
        p.candidates_generated += dl["cands"]
        p.evals += dl["evals"]
        p.total_commits += dl["commits"]
        p.total_holds += dl["holds"]
        p.version += dl["version"]
        p.unsolved_objective += dl["unsolved"]
        p.compute_budget += dl["compute"]
        p.risk_rem += dl["risk"]
        for slot, vec in d["ram_ops"]:
            p.ram.store(slot, vec)  # gate_beta=1: exact overwrite
        if d.get("core") is not None:
            p.core.restore(d["core"])
        if d.get("router") is not None:
            p.router.restore(d["router"])
        if d["trace_row"]:
            for row in d["trace_row"]:
                p.trace.append(row)
        # audit ledger: replay the recorded spend/floor/reject for this task
        ev = d["ev"]
        task = ev["task"]
        n_train = int(ev.get("train_examples", 1))
        n_val = int(ev.get("validation_examples", 1))
        p.ledger.allocate_stream(task, n_train + n_val)
        if d["audit"].get("alpha") is not None:
            p.ledger.spent[task] = (p.ledger.spent.get(task, 0.0)
                                    + float(d["audit"]["alpha"]))
            p.ledger.advance_stream(task, n_train + n_val)
        if ev["action"] == "COMMIT" or ev.get("reason") == "verified-expert-coverage" \
                or ev["action"] == "REUSE":
            p.ledger.record_floor(task, 1.0)
        elif ev["action"] == "HOLD":
            p.ledger.record_reject(task)


# ------------------------------------------------------------------ streams --

def _assign_budget(proc, spec, seed: int, train_fraction: float = 0.70) -> int:
    """Budget the policy would assign (mirrors SelfModelPolicyProcessor)."""
    if hasattr(proc, "dl"):
        from hagi.selfmodel import _features
        feats = dict(zip(FEATURE_KEYS, _features(spec, proc, seed, train_fraction)))
        feats["seed"] = int(seed)
        pred = proc.dl.predict_success(feats)
        if pred == 0:
            proc.policy_overrides += 1
            return proc.floor_budget
        return proc.base_budget
    return proc.node_budget


class ReplayCache:
    """position -> outcome under a given budget; skips re-learning the
    unchanged prefix of the replay stream each round."""

    def __init__(self) -> None:
        self.store: dict[tuple, dict[str, Any]] = {}

    def key(self, pos: tuple, budget: int) -> Optional[dict[str, Any]]:
        return self.store.get((pos, budget))

    def put(self, pos: tuple, budget: int, diff: dict[str, Any]) -> None:
        self.store[(pos, budget)] = diff

def run_stream(proc, segments: list[tuple[int, int]],
               suite: list | None = None, cache: ReplayCache | None = None,
               learner: ParallelLearner | None = None,
               batch_size: int = 8, train_neural: bool = False) -> None:
    """Deterministic stream: suite (sequential) + proposal segments
    (parallel batches). `cache` short-circuits unchanged prefix tasks."""
    names = getattr(proc, "stream_names", None)
    actions = getattr(proc, "stream_actions", None)
    for sp in (suite if suite is not None else hagi.make_task_suite()):
        if names is not None:
            names.append(("suite", sp.name))
        ev = proc.learn(sp, seed=10_000, train_neural=train_neural)
        if actions is not None:
            actions.append(("suite", sp.name, ev["action"]))
    for seg_idx, (n_prop, seed_base) in enumerate(segments):
        rng = np.random.default_rng(seed_base)  # segment identity, not position
        seen: set = set()
        k = 0
        while k < n_prop:
            # generate one batch of proposals from the current snapshot
            batch: list[tuple[Any, int]] = []
            while k < n_prop and len(batch) < batch_size:
                prop = hagi.propose_noncircular_task_v9(proc.experts, rng, 0, seen)
                if prop is None:
                    k = n_prop
                    break
                sp, _ = prop
                if sp.name in proc.experts:
                    continue
                batch.append((sp, seed_base + k))
                if names is not None:
                    names.append(("seg", sp.name))
                k += 1
            if not batch:
                continue
            # policy budgets for the batch
            budgets = [_assign_budget(proc, sp, seed) for sp, seed in batch]
            # replay cache: hits resolve instantly, fresh tasks run in the
            # pool — but ALL merges happen strictly in batch position order,
            # so the experts-dict insertion order (which proposal generation
            # indexes via rng.choice) matches the incumbent stream exactly.
            hits: list = []
            jobs: list[tuple[Any, int, int]] = []
            fresh_pos: list[tuple[tuple, int]] = []
            for (sp, seed), budget in zip(batch, budgets):
                pos = (seg_idx, seed)
                hit = cache.key(pos, budget) if cache is not None else None
                if hit is not None:
                    hits.append(hit)
                else:
                    hits.append(None)
                    jobs.append((sp, seed, budget))
                    fresh_pos.append((pos, budget))
            if jobs:
                if learner is None:
                    # no pool provided: honest in-process sequential path
                    diffs = [_learn_worker_local(proc, sp, seed, budget,
                                                 train_neural)
                             for sp, seed, budget in jobs]
                else:
                    diffs = learner.batch(jobs)
                for (pos, budget), d in zip(fresh_pos, diffs):
                    if cache is not None:
                        cache.put(pos, budget, d)
            else:
                diffs = []
            # merge in original batch order: substitute diffs for their slots
            it = iter(diffs)
            for slot in hits:
                d = next(it) if slot is None else slot
                learner._merge(d)
                if actions is not None:
                    actions.append(("seg", d["ev"]["task"], d["ev"]["action"]))


# ------------------------------------------------------------------- loop ----

def _first_divergence(incumbent, opt) -> Optional[dict]:
    """First stream position where the optimized replay saw a different
    task or a different action than the incumbent (gate-reject diagnosis)."""
    a = getattr(incumbent, "stream_actions", None) or []
    b = getattr(opt, "stream_actions", None) or []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"pos": i, "incumbent": x, "optimized": y}
    if len(a) != len(b):
        return {"pos": min(len(a), len(b)), "len_incumbent": len(a),
                "len_optimized": len(b)}
    return None


def run_self_development(max_rounds: int = 4,
                         base_budget: int = 200000,  # 64-bit safe: evals are Python/numpy int64,
                         # intermediate values guarded by MAG_CAP=2^28 in the vectorized evaluator
                         floor_budget: int = 200,
                         proposals_per_round: int = 10,
                         workers: int = 0,
                         batch_size: int = 8,
                         floor_ladder: list[int] | None = None,
                         initial_segments: list[tuple[int, int]] | None = None,
                         train_neural: bool = True,
                         ) -> dict[str, Any]:
    """floor_ladder: certified lever-change schedule (R223: on exhaustion
    of a lever, SWITCH the lever, don't repeat it). When relief drops below
    MIN_RELIEF with floor F, retry with the next smaller floor — a task that
    HOLDs at base budget cannot commit with less search, so a smaller floor
    only makes the same HOLD cheaper. Loop ends when the ladder is spent.

    initial_segments: resume hook — a previously recorded proposal stream
    (e.g. read back from the model's own self_state.json). Growth replays
    it policy-free first, so the stream continues instead of restarting;
    the self-model is re-derived from the rebuilt trace (cheap, exact)."""
    workers = workers or min(32, (os.cpu_count() or 4))
    ladder = list(floor_ladder) if floor_ladder else [floor_budget]
    report: dict[str, Any] = {"rounds": [], "base_budget": base_budget,
                              "floor_ladder": ladder, "workers": workers}
    suite = hagi.make_task_suite()
    segments: list[tuple[int, int]] = [tuple(s) for s in (initial_segments or [])]
    round_offset = len(segments)
    cache = ReplayCache()

    # Round-0 baseline: suite only (sequential, golden-pinned semantics).
    incumbent = TracedProcessorV9(node_budget=base_budget)
    incumbent.stream_names = []
    incumbent.stream_actions = []
    for sp in suite:
        incumbent.stream_names.append(("suite", sp.name))
        incumbent.learn(sp, seed=10_000, train_neural=train_neural)
    if segments:
        # resume: replay the recorded stream policy-free (growth semantics)
        run_stream(incumbent, segments, suite=[], cache=None,
                   learner=ParallelLearner(incumbent, workers,
                                           train_neural=train_neural),
                   batch_size=batch_size, train_neural=train_neural)
    prev_wasted = sum(t["cost"] for t in incumbent.trace if not t["success"])
    report["baseline"] = {
        "learn_calls": len(incumbent.trace),
        "commits": sum(t["success"] for t in incumbent.trace),
        "wasted_candidates_on_holds": prev_wasted,
        "pipeline": describe_pipeline(incumbent),
    }

    adopted = True
    round_idx = 0
    t_start = time.perf_counter()
    while adopted and round_idx < max_rounds:
        round_idx += 1
        t0 = time.perf_counter()
        # 1. GROW: extend the incumbent with this round's fresh proposals.
        segments.append((proposals_per_round,
                         500_000 + 1000 * (round_idx + round_offset)))
        grow_learner = ParallelLearner(incumbent, workers,
                                       train_neural=train_neural)
        run_stream(incumbent, [segments[-1]], suite=[], cache=None,
                   learner=grow_learner, batch_size=batch_size,
                   train_neural=train_neural)
        incumbent_wasted = sum(t["cost"] for t in incumbent.trace
                               if not t["success"])

        # 2. DERIVE + validate the self-model on the incumbent trace.
        trace = incumbent.trace
        dl = DecisionListSelfModel(trace)
        val = holdout_validate(trace)
        rnd_report: dict[str, Any] = {
            "round": round_idx,
            "learn_calls": len(trace),
            "self_model": dl.describe(),
            "holdout": val,
            "impossibility_certificate": impossibility_certificate(trace),
            "neural_adapters": sum(1 for ex in incumbent.experts.values()
                                    if ex.adapter is not None),
        }
        if not val["exact"]:
            rnd_report["gate"] = {"action": "HOLD",
                                  "reason": "confidently-wrong HOLD predictions on holdout"}
            report["rounds"].append(rnd_report)
            adopted = False
            break

        # 3. OPTIMIZE: replay the SAME deterministic stream with
        #    self-model-driven budgets (cached prefix -> only new tasks run).
        floor_budget = ladder[0]
        opt = SelfModelPolicyProcessor(dl, base_budget, floor_budget,
                                       node_budget=base_budget)
        opt.stream_names = []
        opt.stream_actions = []
        opt_learner = ParallelLearner(opt, workers, train_neural=train_neural)
        run_stream(opt, segments, suite=suite, cache=cache,
                   learner=opt_learner, batch_size=batch_size,
                   train_neural=train_neural)
        opt_wasted = sum(t["cost"] for t in opt.trace if not t["success"])

        # 4. Non-regression commit gate.
        base_names = set(incumbent.experts.keys())
        opt_names = set(opt.experts.keys())
        experts_ok = opt_names >= base_names
        check = opt.verify_all(suite)
        floors_ok = check["accuracy"] == 1.0 and check["committed"] >= len(suite)
        improved = opt_wasted <= int(incumbent_wasted * (1.0 - MIN_RELIEF))
        # certified-convergence margin: micro-improvements below MIN_RELIEF
        # per round are not certified growth — stop instead of accumulating
        # unbounded exploration debt.
        if experts_ok and floors_ok and improved:
            rnd_report["gate"] = {"action": "COMMIT",
                                  "experts_incumbent": len(base_names),
                                  "experts_optimized": len(opt_names),
                                  "exact_accuracy": check["accuracy"],
                                  "wasted_candidates":
                                      f"{incumbent_wasted} -> {opt_wasted}",
                                  "overrides": opt.policy_overrides,
                                  "seconds": round(time.perf_counter() - t0, 2)}
            prev_wasted = opt_wasted
            incumbent = opt
            report["rounds"].append(rnd_report)
        else:
            reason = ("lost experts" if not experts_ok else
                      "quality floor violated" if not floors_ok else
                      f"no certified improvement (relief below "
                      f"{MIN_RELIEF:.0%} margin)")
            if reason.startswith("no certified") and len(ladder) > 1:
                # lever change, not a stop: same self-model, cheaper floor
                used = ladder.pop(0)
                rnd_report["gate"] = {
                    "action": "LEVER_CHANGE",
                    "reason": f"floor {used} exhausted -> floor {ladder[0]} "
                              "(R223: switch the lever, don't repeat it)",
                    "wasted_candidates": f"{incumbent_wasted} -> {opt_wasted}",
                }
                report["rounds"].append(rnd_report)
                continue  # keep growing with the new lever
            rnd_report["gate"] = {
                "action": "HOLD",
                "reason": ("lost experts" if not experts_ok else
                           "quality floor violated" if not floors_ok else
                           f"no certified improvement (relief below "
                           f"{MIN_RELIEF:.0%} margin) — "
                           "self-development converged"),
                "experts_incumbent": len(base_names),
                "experts_optimized": len(opt_names),
                "accuracy": check["accuracy"],
                "wasted_candidates": f"{incumbent_wasted} -> {opt_wasted}",
                "lost": sorted(base_names - opt_names)[:3],
                "first_divergence": _first_divergence(incumbent, opt),
            }
            report["rounds"].append(rnd_report)
            adopted = False

    final = incumbent
    report["segments"] = [[int(n), int(s)] for n, s in segments]
    report["_final_incumbent"] = final
    report["final"] = {
        "adopted_rounds": sum(1 for r in report["rounds"]
                              if r["gate"]["action"] == "COMMIT"),
        "experts": len(final.experts),
        "theorems": len(final.theorems),
        "neural_adapters": sum(1 for ex in final.experts.values()
                                if ex.adapter is not None),
        "pipeline": describe_pipeline(final),
        "wasted_candidates_on_holds": sum(t["cost"] for t in final.trace
                                          if not t["success"]),
        "seconds_total": round(time.perf_counter() - t_start, 2),
        "workers": workers,
    }
    report["converged"] = not adopted
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="HAGI self-development loop")
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--workers", type=int, default=0,
                   help="parallel workers (0 = all CPUs, capped at 32)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--floor-ladder", type=int, nargs="*", default=[2000, 200, 50, 20],
                   help="certified lever-change schedule for the floor budget")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    rep = run_self_development(max_rounds=a.max_rounds, workers=a.workers,
                               batch_size=a.batch_size,
                               floor_ladder=a.floor_ladder)
    rep.pop("_final_incumbent", None)
    out = Path("experiments") / "selfloop_result.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(f"rounds adopted: {rep['final']['adopted_rounds']}, "
              f"converged: {rep['converged']}, experts: {rep['final']['experts']}, "
              f"wasted: {rep['final']['wasted_candidates_on_holds']}, "
              f"{rep['final']['seconds_total']}s on {rep['final']['workers']} workers")
        print(f"report: {out}")


if __name__ == "__main__":
    main()
