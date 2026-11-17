#!/usr/bin/env python3
"""Run the HAGI finite self-model experiment and persist a machine-readable report."""
from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> int:
    out = ROOT / "experiments" / "selfmodel_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "HAGI finite self-model / derive-and-optimize",
        "repository": "ShmidtS/HAGI_v2",
        "branch": os.getenv("GITHUB_REF_NAME", "local"),
        "commit": os.getenv("GITHUB_SHA", "unknown"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import numpy as np
        import hagi
        from hagi.selfmodel import (
            FEATURE_KEYS, TracedProcessorV9, describe_pipeline,
            derive_self_model, impossibility_certificate,
        )

        # Separate trace/derive pass: extract the symbolic self-model itself.
        tracer = TracedProcessorV9(node_budget=2000)
        rng = np.random.default_rng(20260101)
        seen = set()
        for spec in hagi.make_task_suite():
            tracer.learn(spec, seed=10_000, train_neural=False)
        for k in range(10):
            proposal = hagi.propose_noncircular_task_v9(tracer.experts, rng, 0, seen)
            if proposal is None:
                continue
            spec, _parents = proposal
            if spec.name not in tracer.experts:
                tracer.learn(spec, seed=500_000 + k, train_neural=False)
        symbolic = derive_self_model(tracer.trace)
        report["self_model"] = {
            "trace_examples": len(tracer.trace),
            "feature_schema": list(FEATURE_KEYS),
            "pipeline": describe_pipeline(tracer),
            "symbolic_predictors": {
                k: v for k, v in symbolic.items() if not k.startswith("_")
            },
            "impossibility_certificate": impossibility_certificate(tracer.trace),
        }

        # Then trial measured allocation policies over independent seeds.
        from hagi.experiment import run_experiment
        selected_policies = ["baseline","online_exact_2holds_quarter","online_exact_1hold_quarter","online_exact_2holds_eighth"]
        result = run_experiment(
            seeds=20, policy_names=selected_policies, seed_start=30
        )
        report["validation_protocol"] = {
            "seed_start": 30,
            "seed_count": 20,
            "policies": selected_policies,
            "purpose": "test more aggressive exact-profile thresholds against the validated policy on unseen seeds",
        }
        report["status"] = "COMPLETED"
        report["result"] = result
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["exit_code"] = 0
    except BaseException as exc:
        report["status"] = "ERROR"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["exit_code"] = 1
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return int(report["exit_code"])

if __name__ == "__main__":
    raise SystemExit(main())
