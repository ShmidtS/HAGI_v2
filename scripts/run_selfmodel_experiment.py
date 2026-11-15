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
        from hagi_selfexperiment import run_experiment
        result = run_experiment(seeds=5)
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
