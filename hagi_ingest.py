#!/usr/bin/env python3
"""HAGI data ingestion — learn exact programs from ANY available data.

The certified core can only commit programs that are exact on a finite
domain. This module turns arbitrary files into such domains honestly:

  numeric CSV/TSV        for every column pair (x, y): if y is a
                        function of x on the observed rows (no
                        conflicting values), build a task over the
                        observed domain and try to induce a compact
                        exact program (poly/MDL/piecewise). Commit
                        covers 100% of the data.
  JSON                   same, over flattened numeric leaves /
                        arrays of numbers (pairwise adjacent).
  any text file          deterministic next-byte model: keep only
                        bytes whose successor is unambiguous in the
                        text; induce a byte->byte program on that
                        subset (coverage reported honestly).

A task that cannot be compressed to an exact program HOLDs — nothing
is claimed beyond what is demonstrated.

CLI:
  python hagi_ingest.py <file> [--budget 200000] [--json]
  python hagi_ingest.py <dir>          # ingest every file inside
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hagi  # noqa: E402


# --------------------------------------------------------------- parsing ----

def _int_or_none(v: str) -> int | None:
    try:
        return int(v.strip())
    except (ValueError, AttributeError):
        return None


def _numeric_columns(path: Path) -> dict[str, list[int]]:
    delim = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    cols: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=delim)
        for row in reader:
            if not row:
                continue
            # header heuristic: first row all non-numeric -> names
            vals = [_int_or_none(c) for c in row]
            if all(v is None for v in vals):
                names = [c.strip() or f"c{i}" for i, c in enumerate(row)]
                for n in names:
                    cols.setdefault(n, [])
                continue
            if not cols:
                cols = {f"c{i}": [] for i in range(len(row))}
            keys = list(cols.keys())
            for i, v in enumerate(vals):
                if v is not None and i < len(keys):
                    cols[keys[i]].append(v)
    return cols


def _json_numbers(obj: Any) -> list[int]:
    out: list[int] = []

    def walk(o: Any) -> None:
        if isinstance(o, bool):
            return
        if isinstance(o, int):
            out.append(o)
        elif isinstance(o, float) and o.is_integer():
            out.append(int(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def _pair_tasks(xs: list[int], ys: list[int], name: str
                ) -> list[hagi.TaskSpec]:
    """Exact-function test: y must be unambiguous for each observed x."""
    mapping: dict[int, int] = {}
    for x, y in zip(xs, ys):
        if x in mapping and mapping[x] != y:
            return []  # not a function of x -> honest skip
        mapping[x] = y
    if len(mapping) < 4:
        return []  # too little evidence for a certified domain
    dom = tuple(sorted(mapping))

    def teacher(*args, _m=mapping) -> int:
        v = _m.get(int(args[0]))
        if v is None:
            raise ValueError("outside ingested domain")
        return v

    return [hagi.TaskSpec(name, 1, (dom,), teacher, family="ingest")]


def tasks_from_numeric_file(path: Path, max_tasks: int = 16
                            ) -> list[hagi.TaskSpec]:
    tasks: list[hagi.TaskSpec] = []
    if path.suffix.lower() == ".json":
        nums = _json_numbers(json.loads(path.read_text(encoding="utf-8",
                                                       errors="replace")))
        if len(nums) >= 8:
            tasks += _pair_tasks(nums[::2][:512], nums[1::2][:512],
                                 f"INGEST_{path.stem}_json_pairs")
    else:
        cols = _numeric_columns(path)
        names = [n for n in cols if len(cols[n]) >= 8]
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j or len(tasks) >= max_tasks:
                    continue
                tasks += _pair_tasks(cols[names[i]], cols[names[j]],
                                     f"INGEST_{path.stem}_{names[i]}_to_{names[j]}")
    return tasks


def tasks_from_text_file(path: Path) -> list[hagi.TaskSpec]:
    data = path.read_bytes()
    nxt: dict[int, int] = {}
    conflict: set[int] = set()
    for a, b in zip(data, data[1:]):
        if a in nxt and nxt[a] != b:
            conflict.add(a)
        else:
            nxt[a] = b
    for a in conflict:
        nxt.pop(a, None)
    if len(nxt) < 8:
        return []
    dom = tuple(sorted(nxt))

    def teacher(*args, _m=nxt) -> int:
        v = _m.get(int(args[0]))
        if v is None:
            raise ValueError("outside deterministic subset")
        return v

    return [hagi.TaskSpec(f"INGEST_{path.stem}_nextbyte", 1, (dom,), teacher,
                          family="ingest")]


def ingest_file(path: Path) -> list[hagi.TaskSpec]:
    if path.suffix.lower() == ".json":
        return tasks_from_numeric_file(path)
    if path.suffix.lower() in (".csv", ".tsv", ".tab"):
        return tasks_from_numeric_file(path)
    # any other file: bytes are bytes — text, images, binaries
    tasks = tasks_from_text_file(path)
    numeric = tasks_from_numeric_file(path)  # cheap: parses if it can
    return tasks + numeric


# --------------------------------------------------------------- learning ---

def learn_from_data(paths: list[Path], budget: int = 200000
                    ) -> dict[str, Any]:
    proc = hagi.GrowingHagiProcessorV9(node_budget=budget)
    report: dict[str, Any] = {"files": [], "tasks": 0, "committed": 0}
    for path in paths:
        try:
            tasks = ingest_file(path)
        except Exception as e:  # unparseable file is a honest skip
            report["files"].append({"file": str(path), "error": repr(e)[:120]})
            continue
        f_info: dict[str, Any] = {"file": str(path), "tasks": len(tasks),
                                  "results": []}
        for sp in tasks:
            ev = proc.learn(sp, seed=10_000, train_neural=True)
            f_info["results"].append({
                "task": sp.name,
                "action": ev["action"],
                "reason": ev.get("reason", "")[:40],
                "program": ev.get("program", "")[:80],
            })
            report["tasks"] += 1
            report["committed"] += int(ev["action"] == "COMMIT")
        report["files"].append(f_info)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Learn exact programs from any data")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--budget", type=int, default=200000)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    paths: list[Path] = []
    for i in a.inputs:
        paths.extend(sorted(i.iterdir()) if i.is_dir() else [i])
    rep = learn_from_data(paths, budget=a.budget)
    if a.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        for f in rep["files"]:
            print(f"{f['file']}: {f.get('tasks', '?')} tasks")
            for r in f.get("results", []):
                print(f"  {r['task']}: {r['action']} ({r['reason']})")
        print(f"total: {rep['committed']}/{rep['tasks']} committed")


if __name__ == "__main__":
    main()
