"""Self-development loop for the release model.

Cycles: evolve sessions (the model talks to itself; TTT adapts experts on
its own thoughts) -> quality gate -> keep or roll back adaptations.
Usage: python scripts/self_evolve.py [--batches 4] [--session-sec 120]
Quality gate: control prompts through the fast one-shot path; a prompt
fails if the 40-token continuation has trigram repetition ratio >= 0.5.
Rollback: expert backups taken before each batch are restored on failure.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

PY = sys.executable
PROMPTS = [
    "The capital of France is Paris. The capital of Italy is",
    "Столица Франции — Париж. Столица Италии —",
    "Water freezes at 0 degrees. Water boils at",
]


def trigram_rep(text):
    toks = text.split()
    if len(toks) < 6:
        return 1.0
    tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return 1.0 - len(set(tri)) / len(tri)


def backup_experts(dst):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in glob.glob("dsv4_reduced/layer_*/expert_*.pt"):
        rel = os.path.relpath(f, "dsv4_reduced")
        dst_f = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(dst_f), exist_ok=True)
        shutil.copy2(f, dst_f)
        n += 1
    return n


def restore_experts(src):
    n = 0
    for f in glob.glob(os.path.join(src, "layer_*", "expert_*.pt")):
        rel = os.path.relpath(f, src)
        dst = os.path.join("dsv4_reduced", rel)
        if os.path.exists(dst):
            shutil.copy2(f, dst)
        n += 1
    return n


def run_batch(session_sec, sessions):
    cmd = [PY, "scripts/release_gen.py", "", "0", "--evolve",
           "--session-sec", str(session_sec), "--sessions", str(sessions),
           "--turns", "6", "--gen-tokens", "48"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", AMD_SERIALIZE_KERNEL="1",
               HAGI_FAST_MOE="0")
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def gate():
    ok = True
    for p in PROMPTS:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", AMD_SERIALIZE_KERNEL="1",
                   HAGI_TEMP="0.1", HAGI_FAST_MOE="1")
        try:
            r = subprocess.run([PY, "scripts/release_gen.py", p, "40"],
                               env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=900)
        except subprocess.TimeoutExpired:
            print("  gate [TIMEOUT]")
            ok = False
            continue
        out = r.stdout
        marker = "=== OUTPUT ==="
        text = out.split(marker, 1)[1].strip() if marker in out else ""
        rep = trigram_rep(text)
        status = "OK" if rep < 0.5 else "DEGENERATE"
        print(f"  gate [{status}] rep={rep:.2f}: {text[:80]!r}")
        if rep >= 0.5:
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--session-sec", type=int, default=120)
    ap.add_argument("--sessions", type=int, default=1)
    args = ap.parse_args()

    bak = "dsv4_experts_backup"
    n = backup_experts(bak)
    print(f"backed up {n} expert checkpoints -> {bak}")

    for b in range(args.batches):
        print(f"=== batch {b + 1}/{args.batches}: evolve {args.sessions} session(s) ===")
        r = run_batch(args.session_sec, args.sessions)
        tail = (r.stdout or "").splitlines()[-6:]
        print(chr(10).join(tail))
        if r.returncode != 0:
            print(f"evolve exited {r.returncode}; stderr tail:")
            print((r.stderr or "")[-1500:])
            print("restoring backup")
            restore_experts(bak)
            continue
        print("quality gate:")
        if gate():
            print("gate passed: keeping adaptations")
        else:
            print("gate FAILED: rolling back adaptations")
            restore_experts(bak)
    print("self-evolve loop finished")


if __name__ == "__main__":
    main()
