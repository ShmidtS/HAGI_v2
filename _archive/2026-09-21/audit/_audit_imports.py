#!/usr/bin/env python3
"""Read-only intra-scripts import graph.

Classifies each scripts/*.py as:
  LIB   - imported by >=1 other script/module
  LEAF  - never imported by anything (entry point or one-off)
and records whether it is cited in README/docs/plans.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

ROOT = Path(r"C:/HAGI_v2")
SCRIPTS = ROOT / "scripts"

DOC_FILES = [ROOT / "README.md", ROOT / "README_RU.md", ROOT / "README_HF.md",
             ROOT / "ARCHITECTURE.md", ROOT / "BENCHMARKS.md",
             ROOT / "GROWING_HYPOTHESIS.md", ROOT / "AGENT_WORKLOG.md",
             ROOT / "QWEN_PYRAMID_MTP_TTT_PLAN.md"]
DOC_FILES += list((ROOT / "docs").glob("*.md"))
DOC_FILES += list((ROOT / ".omo" / "plans").glob("*.md"))


def imported_names(path: Path) -> set[str]:
    """Module names imported by this file that could resolve to scripts/."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[-1])
            for a in node.names:
                out.add(a.name)
    return out


def main() -> int:
    files = sorted(SCRIPTS.glob("*.py"))
    stems = {f.stem for f in files}

    # who imports whom (only among scripts/)
    users: dict[str, set[str]] = {f.stem: set() for f in files}
    for f in files:
        for nm in imported_names(f):
            if nm in stems and nm != f.stem:
                users[nm].add(f.stem)

    # external users: src/, tests/, root .py
    ext: dict[str, set[str]] = {f.stem: set() for f in files}
    scan: list[Path] = []
    for d in ("src", "tests"):
        for dirpath, dirnames, filenames in os.walk(ROOT / d):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            scan += [Path(dirpath) / fn for fn in filenames if fn.endswith(".py")]
    scan += list(ROOT.glob("*.py"))
    texts = {}
    for p in scan:
        try:
            texts[p] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    for stem in stems:
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(stem)}(?![A-Za-z0-9_])")
        for p, t in texts.items():
            if pat.search(t):
                ext[stem].add(p.relative_to(ROOT).as_posix())

    # doc citations
    doc_hits: dict[str, set[str]] = {s: set() for s in stems}
    for d in DOC_FILES:
        if not d.exists():
            continue
        t = d.read_text(encoding="utf-8", errors="replace")
        for stem in stems:
            if stem in t:
                doc_hits[stem].add(d.name)

    rows = []
    for f in files:
        s = f.stem
        lib = bool(users[s] or ext[s])
        cited = bool(doc_hits[s])
        rows.append((s, lib, cited, len(users[s]), len(ext[s]),
                     sorted(users[s])[:4], sorted(ext[s])[:3],
                     sorted(doc_hits[s]), f.stat().st_size))

    leaves = [r for r in rows if not r[1]]
    uncited_leaves = [r for r in leaves if not r[2]]

    print(f"scripts={len(rows)} libs={len(rows) - len(leaves)} "
          f"leaves={len(leaves)} uncited_leaves={len(uncited_leaves)}")
    print("\n=== LIB (imported by something) ===")
    for r in sorted(rows, key=lambda x: -x[3]):
        if not r[1]:
            continue
        print(f"{r[0]:<32} users={r[3]} ext={r[4]} doc={len(r[7])} {r[5] or r[6]}")
    print("\n=== LEAF, cited in docs (documented entry point) ===")
    for r in sorted(leaves, key=lambda x: x[0]):
        if r[2]:
            print(f"{r[0]:<32} doc={r[7]}")
    print("\n=== LEAF, UNCITED anywhere (candidate: spent one-off) ===")
    for r in sorted(uncited_leaves, key=lambda x: -x[8]):
        print(f"{r[0]:<32} {r[8]:>6}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
