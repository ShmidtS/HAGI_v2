#!/usr/bin/env python3
"""Read-only reference graph for the HAGI_v2 tree.

Collects every mention of a *.py / *.sh / *.ps1 basename in source-like
files, then reports which real files are never mentioned anywhere else.

Writes a JSON report; changes nothing on disk.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

ROOT = Path(r"C:/HAGI_v2")

# Directories that are data/build artifacts, never scanned for references
# and never treated as candidates for removal by this audit.
SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "data", "models", "reports", "eval",
    "research_refs", "glm5_pod", "glm5_gguf", "glm5_kvp", "dsv4_release",
    "dsv4_bank", "checkpoints", "checkpoints_dsv4", "logs", ".omo", ".pi",
    ".omc", "node_modules", "build", "dist", "evolve_state", "self_talk_logs",
    "llama-glm5", "llama-ds4", ".pytest_cache", ".mypy_cache",
}

SRC_EXT = {".py", ".sh", ".ps1", ".md", ".toml", ".yaml", ".yml", ".json",
           ".cpp", ".h", ".txt", ".rs", ".psm1", ".bat", ".cmd"}

MENTION = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*\.(?:py|sh|ps1)\b")

MAX_SCAN_BYTES = 4 * 1024 * 1024


def tracked_paths() -> set[str]:
    import subprocess
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    return {p.strip().replace("\\", "/") for p in out.splitlines() if p.strip()}


def all_files() -> list[Path]:
    res: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT).replace("\\", "/")
        base = os.path.basename(dirpath)
        if rel == "." :
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            continue
        if base in SKIP_DIRS or rel.startswith(".git/"):
            dirnames[:] = []
            continue
        for fn in filenames:
            res.append(Path(dirpath) / fn)
    return res


def main() -> int:
    tr = tracked_paths()
    files = all_files()

    # 1. mentions: scan source-like files for "*.py|sh|ps1" tokens
    mentions: dict[str, set[str]] = {}
    for p in files:
        if p.suffix.lower() not in SRC_EXT:
            continue
        try:
            if p.stat().st_size > MAX_SCAN_BYTES:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for m in MENTION.findall(text):
            mentions.setdefault(m.lower(), set()).add(rel)

    # 2. candidates: executable/script files in the audited area
    rows = []
    for p in files:
        if p.suffix.lower() not in {".py", ".sh", ".ps1"}:
            continue
        rel = p.relative_to(ROOT).as_posix()
        stem = p.name.lower()
        # a file "references itself"; strip that so self-mentions do not hide
        # the absence of external users
        users = {u for u in mentions.get(stem, set()) if u != rel}
        try:
            st = p.stat()
        except OSError:
            continue
        rows.append({
            "path": rel,
            "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%d", time.localtime(st.st_mtime)),
            "tracked": rel in tr,
            "n_users": len(users),
            "users": sorted(users)[:6],
        })

    orphans = [r for r in rows if r["n_users"] == 0]
    orphans.sort(key=lambda r: (-r["bytes"], r["path"]))

    report = {
        "n_files_scanned": len(files),
        "n_scripts": len(rows),
        "n_tracked_scripts": sum(1 for r in rows if r["tracked"]),
        "n_orphans": len(orphans),
        "orphan_bytes": sum(r["bytes"] for r in orphans),
        "orphans": orphans,
        "all": sorted(rows, key=lambda r: r["path"]),
    }
    out = ROOT / "_audit_refs.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"scanned={report['n_files_scanned']} scripts={report['n_scripts']} "
          f"tracked={report['n_tracked_scripts']} orphans={report['n_orphans']} "
          f"orphan_bytes={report['orphan_bytes']}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
