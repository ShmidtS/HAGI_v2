#!/usr/bin/env python3
"""Run the automatic critique over laws already accepted into law_registry.

This CLI intentionally contains no hand-written law definitions. The live
training pipeline calls the same service after a new law is registered.
"""
import json
from hagi.unified import Unified
from hagi.law_critique import run_law_self_critique


def main() -> int:
    u = Unified.load() if Unified.exists() else Unified()
    report = run_law_self_critique(u)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
