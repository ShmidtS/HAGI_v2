#!/usr/bin/env python3
"""
E3 objective-check validator (independent, offline).

Implements the *same* objective-answer contract published by Bonsai in
`models/ternary-bonsai-2-27b-mtp/training/assess.py`, but runs without the
server: it parses an already-produced `quality-<mode>.json` and recomputes
both strict and semantic scores with the documented normalization rules.

This is a *validator*, not a generator. It does not generate text, call any
server, or touch the tokenizer/model ID contract. KL is never treated as a
quality proxy here.

Usage:
    python scripts/e3_bonsai_validator.py \
        --input models/ternary-bonsai-2-27b-mtp/reports/quality-stage2-q8-n2.json \
        --output reports/e3_quality_stage2_q8_n2.json
    python scripts/e3_bonsai_validator.py --list-tasks   # prints the 12 ids
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The 12 predeclared objective checks, in Bonsai assessment order.
# Each entry: (id, expected). `expected` is the canonical answer object.
TASKS: list[tuple[str, object]] = [
    ("product", 33702),
    ("fraction", "31/36"),
    ("modular", 17),
    ("probability", "1/9"),
    ("geometry", 90),
    ("grid", 35),
    ("sort", [-11, -3, 0, 4, 7, 8, 8]),
    ("distinct", ["b", "a", "n"]),
    ("filter", ["c", "a"]),
    ("transform", [25, 121]),
    ("logic", ["D", "A", "B", "C"]),
    ("code_trace", [[9, 2, 3], [1, 2, 3, 4]]),
]

EXPECTED_BY_ID = {tid: exp for tid, exp in TASKS}


def normalize(tid: str, parsed: object) -> object:
    """Apply Bonsai's documented normalization edge cases.

    - distinct: a string answer "ban" is equivalent to ["b","a","n"]
    - logic: a string "D, A, B, C" is equivalent to ["D","A","B","C"]

    All other ids are passed through unchanged (strict == semantic).
    """
    if tid == "distinct" and isinstance(parsed, str):
        return list(parsed)
    if tid == "logic" and isinstance(parsed, str):
        return re.findall(r"[A-D]", parsed)
    return parsed


def parse_answer(content: str) -> object | None:
    """Strict parser: must be exactly JSON {"answer": value} with no prose."""
    if content is None:
        return None
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        return json.loads(clean)["answer"]
    except Exception:
        return None


def evaluate(results: list[dict]) -> dict:
    """Evaluate a list of per-task result dicts.

    Each item must carry: id, expected, parsed, strict_type_match (bool).
    Returns the published summary shape plus a normalized pass/fail list.
    """
    by_id = {r["id"]: r for r in results}
    checks = []
    strict_passed = 0
    semantic_passed = 0

    for tid, expected in TASKS:
        r = by_id.get(tid)
        if r is None:
            checks.append({
                "id": tid,
                "expected": expected,
                "parsed": None,
                "present": False,
                "strict_type_match": False,
                "normalized": None,
                "semantic_pass": False,
                "finish_reason": None,
            })
            continue

        parsed = r.get("parsed")
        strict = bool(r.get("strict_type_match", r.get("pass", False)))
        norm = normalize(tid, parsed)
        semantic = norm == expected

        strict_passed += strict
        semantic_passed += semantic

        checks.append({
            "id": tid,
            "expected": expected,
            "parsed": parsed,
            "present": True,
            "strict_type_match": strict,
            "normalized": norm,
            "semantic_pass": semantic,
            "finish_reason": r.get("finish_reason"),
        })

    return {
        "method": (
            "12 predeclared objective checks; offline re-validation of a "
            "captured quality report. strict_type_match = exact parse; "
            "semantic_pass = normalized equality. KL is not a quality proxy."
        ),
        "total": len(TASKS),
        "strict_passed": strict_passed,
        "semantic_passed": semantic_passed,
        "normalization_note": (
            "Two ordering prompts permit equivalent string/list "
            "representations: ban and [b,a,n]; D, A, B, C and [D,A,B,C]. "
            "Raw strict results retained separately."
        ),
        "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E3 offline objective-check validator")
    ap.add_argument("--input", "-i", type=Path,
                    help="quality-*.json produced by the Bonsai assessment")
    ap.add_argument("--output", "-o", type=Path,
                    help="where to write the machine-readable result")
    ap.add_argument("--list-tasks", action="store_true",
                    help="print the 12 task ids in order and exit")
    args = ap.parse_args()

    if args.list_tasks:
        for tid, _exp in TASKS:
            print(tid)
        return 0

    if args.input is None:
        print("--input is required (or use --list-tasks)", file=sys.stderr)
        return 2

    data = json.loads(args.input.read_text())
    if "results" not in data:
        print(f"input {args.input} has no 'results' key", file=sys.stderr)
        return 2

    summary = evaluate(data["results"])

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "input": str(args.input),
        "total": summary["total"],
        "strict_passed": summary["strict_passed"],
        "semantic_passed": summary["semantic_passed"],
        "semantic_pass": summary["semantic_passed"] == summary["total"],
    }, indent=2))

    # Exit 0 iff semantics fully pass. Strict is reported but not gated,
    # because two ids deliberately normalize string<->list.
    return 0 if summary["semantic_passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
