#!/usr/bin/env python3
"""Read-only dead-symbol audit for src/hagi.

For every top-level def/class in src/hagi/**.py, count references anywhere in
the repo (src, tests, scripts, root .py, docs) excluding the defining line
itself. Reports symbols with zero external users.
"""
from __future__ import annotations

import ast
import re
import os
from pathlib import Path

ROOT = Path(r"C:/HAGI_v2")
SRC = ROOT / "src" / "hagi"

SCAN_DIRS = ["src", "tests", "scripts", "docs", "configs"]
SCAN_ROOT_PY = list(ROOT.glob("*.py"))


def defs_in(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((node.name, node.lineno))
    return out


def corpus() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in filenames:
                if fn.endswith((".py", ".md", ".sh", ".yaml", ".toml")):
                    files.append(Path(dirpath) / fn)
    files.extend(SCAN_ROOT_PY)
    return files


def main() -> int:
    files = corpus()
    texts = {}
    for f in files:
        try:
            texts[f] = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

    rows = []
    for py in sorted(SRC.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        rel = py.relative_to(ROOT).as_posix()
        try:
            syms = defs_in(py)
        except SyntaxError:
            continue
        for name, lineno in syms:
            if name.startswith("__"):
                continue
            pat = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_])")
            hits = 0
            for f, t in texts.items():
                if f == py:
                    # count uses inside the defining file, minus the def line
                    for i, line in enumerate(t.splitlines(), 1):
                        if i == lineno:
                            continue
                        hits += len(pat.findall(line))
                    continue
                hits += len(pat.findall(t))
            private = name.startswith("_")
            rows.append((rel, name, hits, private, lineno))

    dead = [r for r in rows if r[2] == 0]
    print(f"symbols={len(rows)} zero_reference={len(dead)}")
    print("\n=== zero-reference top-level symbols ===")
    for rel, name, hits, private, lineno in sorted(dead, key=lambda r: (r[0], r[1])):
        tag = "private" if private else "PUBLIC"
        print(f"{tag:>7} {rel}:{lineno} {name}")
    print("\n=== low-reference (<3) public symbols ===")
    for rel, name, hits, private, lineno in sorted(rows, key=lambda r: r[2]):
        if private or hits == 0 or hits >= 3:
            continue
        print(f"{hits:>3} uses  {rel}:{lineno} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
