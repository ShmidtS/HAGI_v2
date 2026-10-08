"""Autonomous growth-cycle driver (the self-sustaining loop, Gall-minimal).

One invocation = ONE full generation of the recursive cycle:

  1. N same-origin leaves: train.py per leaf config, all
     ``init_from`` the current ROOT (shared prior), distinct data
     seeds (same-origin requirement, §23[1]).
  2. latent merge: pilot_latent_merge.py --leaves ... --shared-init
     ROOT (theory-correct pipeline: factorize around the PRIOR,
     merge_gate per tensor, fibers saved — R242/R243).
  3. joint fine-tune of the merged root (init_from root.pt).
  4. certified accept: the new ft root replaces the incumbent ONLY
     if it wins the §9 certified A/B on a common-protocol eval;
     otherwise the cycle HALTS (stop_condition — no uncertified
     growth, audit P1-1 lesson).

The ladder is logged to checkpoints/growth_ladder.jsonl (one line per
generation: incumbent CE, merged CE, ft CE, verdict). Designed to be
re-invoked (cron/supervisor) — each call advances at most one
generation and is idempotent-safe via the marker files.

Usage:
    python scripts/growth/growth_cycle.py --config-dir configs \
        --leaf-prefix gen --root checkpoints/gen3_root_ft/best.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "checkpoints" / "growth_ladder.jsonl"


def sh(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(ROOT)).returncode


def avg_ce(config: str, ckpt: str, batches: int = 10) -> float | None:
    """Common-protocol AVG exact CE of a checkpoint (None on failure)."""
    out = ROOT / "checkpoints" / "_cycle_eval.json"
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "-u", str(ROOT / "scripts/eval_domains.py"),
         "--config", config, "--resume", ckpt, "--batches", str(batches),
         "--device", "cuda"],
        capture_output=True, text=True, cwd=str(ROOT))
    out.write_text(r.stdout, encoding="utf-8")
    for line in r.stdout.splitlines():
        if line.strip().startswith("AVG"):
            return float(line.split("exact_ce=")[1].split()[0])
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gen", type=int, required=True, help="generation to RUN (e.g. 5)")
    p.add_argument("--root", required=True, help="incumbent root checkpoint (shared prior)")
    p.add_argument("--domains", nargs="+", default=["math", "lang", "code"])
    p.add_argument("--leaf-config", default="configs/gen4_leaf_{d}.yaml",
                   help="config template; {{d}} is the domain")
    p.add_argument("--ft-config", default="configs/gen4_root_ft.yaml")
    p.add_argument("--common-config", default="configs/latent_leaf_math.yaml")
    args = p.parse_args()
    g = args.gen

    # 1. leaves ------------------------------------------------------------
    for d in args.domains:
        leaf_dir = ROOT / f"checkpoints/gen{g}_leaf_{d}"
        if (leaf_dir / "best.pt").exists():
            print(f"[gen{g}] leaf {d}: best.pt exists — skip")
            continue
        cfg = args.leaf_config.format(d=d)
        rc = sh([sys.executable, "-X", "utf8", "-u", "scripts/train.py",
                 "--config", cfg, "--checkpoint-dir", str(leaf_dir),
                 "--log-dir", str(leaf_dir / "logs")], leaf_dir / "cycle.log")
        if rc != 0:
            print(f"[gen{g}] leaf {d} FAILED rc={rc} — halting")
            return rc

    # 2. merge --------------------------------------------------------------
    gen_root = ROOT / f"checkpoints/gen{g}_root"
    if not (gen_root / "root.pt").exists():
        leaves = [str(ROOT / f"checkpoints/gen{g}_leaf_{d}/best.pt")
                  for d in args.domains]
        rc = sh([sys.executable, "-X", "utf8", "-u",
                 "scripts/growth/pilot_latent_merge.py",
                 "--leaves", *leaves, "--shared-init", args.root,
                 "--out", str(gen_root)], gen_root / "cycle.log")
        if rc != 0:
            print(f"[gen{g}] merge gate/abort rc={rc} — halting (theory §3)")
            return rc

    # 3. fine-tune ----------------------------------------------------------
    ft_dir = ROOT / f"checkpoints/gen{g}_root_ft"
    ft_ckpt = ft_dir / "best.pt"
    if not ft_ckpt.exists():
        rc = sh([sys.executable, "-X", "utf8", "-u", "scripts/train.py",
                 "--config", args.ft_config,
                 "--checkpoint-dir", str(ft_dir),
                 "--log-dir", str(ft_dir / "logs")], ft_dir / "cycle.log")
        if rc != 0:
            print(f"[gen{g}] root ft FAILED rc={rc}")
            return rc
    if not ft_ckpt.exists():  # best-at-step-1 fix still can miss on abort
        cands = sorted(ft_dir.glob("step-*.pt"))
        if not cands:
            print(f"[gen{g}] no ft checkpoint — halting")
            return 1
        ft_ckpt = cands[-1]

    # 4. certified accept ----------------------------------------------------
    inc_ce = avg_ce(args.common_config, args.root)
    new_ce = avg_ce(args.common_config, str(ft_ckpt))
    if inc_ce is None or new_ce is None:
        print("eval failed — halting (no uncertified growth)")
        return 1
    entry = {"gen": g, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "incumbent": args.root, "incumbent_ce": inc_ce,
             "merged_ce": avg_ce(args.common_config, str(gen_root / "root.pt")),
             "ft_ce": new_ce, "delta": inc_ce - new_ce}
    # §9 conservative: the generation ships only on a strict improvement
    # beyond eval noise (2-eps band approximated by 0.02 nats at n=10
    # batches/domain); ties and regressions halt the cycle.
    margin = 0.02
    entry["accepted"] = bool(entry["delta"] > margin)
    with LADDER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(json.dumps(entry, indent=2))
    if not entry["accepted"]:
        print(f"[gen{g}] NOT certified (delta {entry['delta']:+.4f} <= {margin}) "
              "— cycle halts; incumbent stands (§9)")
        return 0
    print(f"[gen{g}] accepted: next gen uses {ft_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
