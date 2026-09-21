#!/usr/bin/env python3
"""Reversible cleanup of spent one-off scripts and root scratch.

Moves (never deletes) files into _archive/<STAMP>/ with a MANIFEST.md that
records, per file: reason, size, tracked-or-not. Byte-identical duplicate logs
and __pycache__ are the only true deletions (zero information loss).

Dry-run by default; pass --apply to move.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:/HAGI_v2")
STAMP = "2026-09-21"
DEST = ROOT / "_archive" / STAMP

# --- scripts/: spent one-offs from concluded compression tracks -------------
# Criterion (all must hold): never imported by another script/module, never
# invoked by any .sh/.ps1 driver, never cited in README/docs/plans, and not an
# evidence harness for a report that still stands.
ARCHIVE_SCRIPTS = {
    # GLM-5 TQ1 / POD / STE track — concluded: docs/COMPRESSION_PLAN.md records
    # the FFN experts as already reduced by the channel-per-expert method; the
    # glm5_pod conclusion notes are already gone from disk.
    "glm5_cascade_ste", "glm5_tac_train", "glm5_distill_expert", "glm5_ste",
    "glm_diag_w2", "glm_actaware_all", "glm_actaware_probe", "glm_actaware_build",
    "glm_cascade_fit", "glm_cascade_all", "glm_split_build", "glm_split_stream",
    "glm_verify_decode", "glm_absmean_all", "glm_absmean_build", "glm_scale_schemes",
    "build_tq1ste_test", "build_ced_mixed", "build_dither",
    "skel8_train", "extract_skel8_tensors",
    "probe_w2_overfit", "probe_realtext_resid", "sweep_approaches",
    "_test_ste_one", "test_moe_train", "test_moe_smoke",
    # DeepSeek-V4 diagnostics sitting on top of the documented live pipeline
    # (dsv4_experts / dsv4_collect_seq / dsv4_refit_experts / dsv4_generate_ttt
    # / dsv4_train_gate / gate_moe_fast all stay).
    "dsv4_collect_all_tokens", "dsv4_compare_expert", "dsv4_build_skeleton",
    "dsv4_split_layers", "dsv4_test_kvcache_int8",
    "probe_router_drift", "probe_fisher_experts", "eval_token_agreement",
}

# --- root scratch: untracked, regenerable, or superseded --------------------
ARCHIVE_ROOT = {
    "_glm_index.json": "8.4 MB GGUF index dump from the GLM-5 build (Sep 12); "
                       "regenerable from models/, nothing reads it",
    "_glm_config.json": "GGUF metadata dump paired with _glm_index.json; "
                        "regenerable",
    "_glm_cmake_build.ps1": "one-shot cmake helper for the GLM-5 fork build",
    "_glm_cmake_config.ps1": "one-shot cmake configure helper",
    "_m1.log": "scratch log from the M1 micro-opt pass (Sep 12)",
    "_probe_ste.py": "underscore-prefixed scratch STE probe; superseded by "
                     "scripts/glm5_cascade_ste (archived with the track)",
    "download_orig.py": "one-shot HF weights downloader (Aug 12), zero users",
    "self_evolve_llamacpp.py": "superseded by bonsai_evolution_daemon.py "
                               "(bounded daemon with resume + confidence gates)",
    "self_talk_llamacpp.py": "superseded by bonsai_evolution_daemon.py",
}

# --- true deletions: byte-identical duplicates + interpreter cache ----------
DELETE_DUPS = [
    "bonsai_server_8090_stdout.log",
    "bonsai_server_8090_gpu_stdout.log",
]


def tracked(rel: str) -> bool:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
        capture_output=True,
    )
    return r.returncode == 0


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    apply = "--apply" in sys.argv
    rows: list[tuple[str, str, str, int, bool]] = []

    for stem in sorted(ARCHIVE_SCRIPTS):
        src = ROOT / "scripts" / f"{stem}.py"
        if not src.exists():
            print(f"MISSING scripts/{stem}.py", file=sys.stderr)
            continue
        rows.append((f"scripts/{stem}.py", "scripts",
                     "spent one-off: no importer, no driver, no doc citation",
                     src.stat().st_size, tracked(f"scripts/{stem}.py")))

    for rel, why in sorted(ARCHIVE_ROOT.items()):
        src = ROOT / rel
        if not src.exists():
            print(f"MISSING {rel}", file=sys.stderr)
            continue
        rows.append((rel, "root", why, src.stat().st_size, tracked(rel)))

    total = sum(r[3] for r in rows)
    print(f"archive: {len(rows)} files, {total} bytes -> _archive/{STAMP}/")
    for rel, group, why, size, trk in rows:
        print(f"  {'T' if trk else 'U'} {size:>8} {group:<7} {rel}")

    # duplicate logs: verify byte-identity before removing anything
    dup_rows = []
    for rel in DELETE_DUPS:
        p = ROOT / rel
        twin = rel.replace("_stdout", "")
        tp = ROOT / twin
        if not p.exists() or not tp.exists():
            print(f"SKIP dup {rel} (file or twin missing)", file=sys.stderr)
            continue
        if md5(p) != md5(tp):
            print(f"REFUSE dup {rel}: NOT byte-identical to {twin}", file=sys.stderr)
            continue
        dup_rows.append((rel, twin, p.stat().st_size))
    dup_bytes = sum(d[2] for d in dup_rows)
    print(f"\ndelete byte-identical dups: {len(dup_rows)} files, {dup_bytes} bytes")
    for rel, twin, size in dup_rows:
        print(f"  {size:>8} {rel} == {twin}")

    pyc = ROOT / "__pycache__"
    pyc_n = len(list(pyc.glob("*"))) if pyc.exists() else 0
    print(f"delete __pycache__/: {pyc_n} entries")

    if not apply:
        print("\nDRY RUN — pass --apply to execute")
        return 0

    for rel, group, why, size, trk in rows:
        src = ROOT / rel
        dst = DEST / group / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    for rel, twin, size in dup_rows:
        (ROOT / rel).unlink()
    if pyc.exists():
        shutil.rmtree(pyc)

    lines = [
        f"# Archived {STAMP}",
        "",
        "Reversible cleanup of spent one-off scripts and root scratch.",
        "Nothing here was deleted; `git checkout HEAD -- <path>` restores tracked",
        "files, and `_archive/<STAMP>/<group>/<name>` restores the rest by move-back.",
        "",
        "| path | group | tracked | bytes | reason |",
        "|---|---|---|---|---|",
    ]
    for rel, group, why, size, trk in rows:
        lines.append(f"| `{rel}` | {group} | {'yes' if trk else 'no'} | {size} | {why} |")
    lines += [
        "",
        "Byte-identical duplicate logs removed outright (zero information loss):",
        "",
    ]
    for rel, twin, size in dup_rows:
        lines.append(f"- `{rel}` == `{twin}` ({size} bytes)")
    lines.append("- `__pycache__/` (interpreter cache)")
    (DEST / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\napplied; manifest -> {DEST / 'MANIFEST.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
