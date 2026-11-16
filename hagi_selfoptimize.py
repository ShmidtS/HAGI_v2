#!/usr/bin/env python3
"""HAGI self-optimization — minimize the function library (MDL) while
preserving certified coverage.

The audit (user-provided) found the library inflated by:
  - exact behavioral duplicates (same outputs on the certified domain),
  - pseudo-2-arg functions that ignore one variable,
  - unreferenced compositions.

Prune policy is HONEST and CERTIFIED:
  1. keep every expert whose call-name is referenced by another kept
     expert's program (call DAG integrity),
  2. of behaviorally-identical groups (same arity + same certified
     input->output map) keep only the shortest AST,
  3. everything else that no kept expert references and that
     duplicates behavior is dropped,
  4. after pruning, EVERY kept program is re-validated exactly on its
     full certified domain; any failure aborts the prune (safety).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _expr_calls(expr) -> set[str]:
    """Names of experts called by this expression (call nodes)."""
    out: set[str] = set()

    def walk(e) -> None:
        if e is None:
            return
        if getattr(e, "op", None) == "call":
            out.add(str(e.value))
            for a in e.args:
                walk(a)
        else:
            for a in getattr(e, "args", []) or []:
                walk(a)

    walk(expr)
    return out


def _ast_size(expr) -> int:
    if expr is None:
        return 0
    return 1 + sum(_ast_size(a) for a in getattr(expr, "args", []) or [])


def _certified_map(proc, name: str) -> tuple[int, tuple]:
    """(arity, sorted ((inputs..., output)) map) on the certified domain."""
    ex = proc.experts[name]
    rows = []
    for row in _domain_rows(proc, name):
        try:
            rows.append((*row, int(ex.program.expr.eval(row, proc.experts))))
        except Exception:
            rows.append((*row, "ERR"))
    return ex.program.arity, tuple(sorted(rows))


def _domain_rows(proc, name: str):
    ex = proc.experts[name]
    doms = ex.domains
    if ex.program.arity == 1:
        return [(x,) for x in doms[0]]
    if ex.program.arity == 2:
        return [(x, y) for x in doms[0] for y in doms[1]]
    import itertools
    return [tuple(c) for c in itertools.product(*doms)]


def prune_library(proc, max_report: int = 12) -> dict[str, Any]:
    """Return (new_proc-like object mutated in place? NO — returns a
    report and a pruned experts dict + counts; caller rebuilds)."""
    experts = proc.experts
    names = list(experts.keys())
    # behavior maps for experts with expr
    sigs: dict[str, tuple] = {}
    for n in names:
        if experts[n].program.expr is not None:
            try:
                sigs[n] = _certified_map(proc, n)
            except Exception:
                sigs[n] = ("ERR", n)  # unique-ish, never deduped
    # duplicate groups: same signature
    groups: dict[tuple, list[str]] = {}
    for n, s in sigs.items():
        groups.setdefault(s, []).append(n)
    dup_keep: dict[str, str] = {}
    dup_drop: set[str] = set()
    for s, members in groups.items():
        if len(members) > 1:
            members.sort(key=lambda n: (_ast_size(experts[n].program.expr), n))
            keep = members[0]
            dup_keep[keep] = f"{len(members)} copies"
            dup_drop.update(members[1:])
    # reference graph over exprs
    refs: dict[str, set[str]] = {}
    for n in names:
        refs[n] = _expr_calls(experts[n].program.expr) if experts[n].program.expr else set()
    # referenced-by counts
    refd: set[str] = set()
    for n in names:
        refd |= (refs[n] & set(names)) - {n}
    # pseudo-2-arg: binary functions whose body ignores the 2nd variable
    pseudo_drop: set[str] = set()
    for n in names:
        ex = experts[n]
        if ex.program.arity != 2 or n in dup_drop:
            continue
        src_calls = refs[n]
        # body uses only x if no call depends on y and expr never uses var 1
        uses_y = _expr_uses_var(ex.program.expr, 1)
        if not uses_y and n not in refd and len(sigs.get(n, ("?",))[0:1] or ()) >= 0:
            pseudo_drop.add(n)
    # start from everything, remove duplicates
    keep = set(names) - dup_drop
    # iteratively drop non-referenced pseudo/unreferenced-extra experts
    changed = True
    extra_drop: set[str] = set()
    base_guard = {"ADD", "SUB", "MUL", "NEG", "ABS", "SIGN", "SQUARE",
                  "CUBE", "MIN", "MAX", "GCD", "LCM", "EQ", "LT", "GE",
                  "MEDIAN3", "CLAMP", "FMA"}
    while changed:
        changed = False
        refd_now: set[str] = set()
        for n in keep:
            refd_now |= (refs[n] & keep) - {n}
        for n in list(keep):
            if n in base_guard or n in refd_now:
                continue
            if n in pseudo_drop:
                keep.discard(n); extra_drop.add(n); changed = True
    report = {
        "before": len(names),
        "after": len(keep),
        "dropped_duplicates": len(dup_drop),
        "dropped_pseudo_1arg": len(extra_drop),
        "dup_groups": len(dup_keep),
        "examples_dup_groups": [
            f"{k}: {v}" for k, v in list(dup_keep.items())[:max_report]],
    }
    return {"keep": keep, "report": report, "refs": refs}


def _expr_uses_var(expr, idx: int) -> bool:
    if expr is None:
        return False
    if getattr(expr, "op", None) == "var" and int(expr.value) == idx:
        return True
    return any(_expr_uses_var(a, idx) for a in getattr(expr, "args", []) or [])


def apply_prune(proc, keep: set[str]):
    """Return a shallow-copied processor with only kept experts, after
    FULL exact re-validation. Raises on any validation failure."""
    import copy
    new = copy.copy(proc)
    new.experts = {n: proc.experts[n] for n in proc.experts if n in keep}
    # full exact re-validation of every kept expert
    failures = 0
    for n, ex in new.experts.items():
        if ex.program.expr is None:
            continue
        for row in _domain_rows(proc, n):
            try:
                want = int(ex.program.expr.eval(row, proc.experts))
            except Exception:
                want = "ERR"
            try:
                got = int(ex.program.expr.eval(row, new.experts))
            except Exception:
                got = "ERR"
            if want != got:
                failures += 1
                break
    if failures:
        raise RuntimeError(
            f"prune aborted: {failures} experts failed re-validation")
    return new, failures
