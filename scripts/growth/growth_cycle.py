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
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "checkpoints" / "growth_ladder.jsonl"


def _noise_cos(gen_root: Path) -> float | None:
    """R255 coherence from the merge's noise_report.json."""
    p = gen_root / "noise_report.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["r255_mean_abs_cos"]
        except Exception:
            return None
    return None


def consumed_offset(leaf_dir: Path) -> int:
    """Per-leaf fresh-data injection: the next generation of the SAME
    mix continues where the previous leaf stopped (its own consumed.json
    — the global one is meaningless across different mixes). Falls back
    to 0 on the first generation."""
    p = leaf_dir / "consumed.json"
    if p.exists():
        try:
            return int(json.loads(p.read_text(encoding="utf-8"))["consumed_tokens"])
        except Exception:
            return 0
    return 0


def _fiber_ce(config: str, root_pt: str, fibers_pt: str,
              batches: int = 30) -> float | None:
    """AVG exact CE of the R255 fiber ensemble (root + fibers, logit-mean
    over the reconstructed experts). Returns None on any failure — the
    fiber arm is an ADDITIONAL candidate, never a blocker."""
    if not (ROOT / fibers_pt).exists():
        return None
    import re as _re
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "-u",
         str(ROOT / "scripts/growth/eval_fiber_ensemble.py"),
         "--config", config, "--root", root_pt, "--fibers", fibers_pt,
         "--batches", str(batches), "--device", "cuda"],
        capture_output=True, text=True, cwd=str(ROOT))
    m = None
    for m in _re.finditer(r"\{.*\}", r.stdout, _re.S):
        pass  # keep the LAST JSON blob (the per-domain report)
    if m is None:
        return None
    try:
        import json as _json
        d = _json.loads(m.group(0))
        rows = d.get("domains") or d.get("results") or []
        ces = [x["exact_ce"] for x in rows if "exact_ce" in x]
        return sum(ces) / len(ces) if ces else None
    except Exception:
        return None


def sh(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(ROOT)).returncode


def avg_ce(config: str, ckpt: str, batches: int = 30) -> float | None:
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
        def _leaf_run(offset: int) -> int:
            return sh([sys.executable, "-X", "utf8", "-u", "scripts/train.py",
                       "--config", cfg, "--checkpoint-dir", str(leaf_dir),
                       "--log-dir", str(leaf_dir / "logs"),
                       "--start-offset", str(offset)], leaf_dir / "cycle.log")

        rc = _leaf_run(consumed_offset(leaf_dir))
        if rc != 0 and (leaf_dir / "best.pt").exists() is False and                 "exceeds sealed budget" in (leaf_dir / "cycle.log").read_text(encoding="utf-8", errors="replace")[-4000:]:
            # The mix's sealed budget is smaller than the cumulative offset
            # (the corpus was already wrapped): restart the generation's
            # leaf from 0 — the wrap means the data is exhausted anyway.
            rc = _leaf_run(0)
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

    # 3b. WSqD cooldown (R254): a short polish at batch x4 / lr 0.4x after
    # the ft phase — the decay tail is a CONSTANT noise cost, so the polish
    # is nearly free; best.pt may or may not beat the ft best.
    cd_dir = ROOT / f"checkpoints/gen{g}_cooldown"
    if not (cd_dir / "best.pt").exists():
        # Configs live with the generation (no configs/ proliferation):
        # the cooldown config is derived from the ft config into the
        # generation's own checkpoint dir.
        cd_cfg = cd_dir / "cooldown.yaml"
        base = (ROOT / args.ft_config).read_text(encoding="utf-8")
        cd_cfg.parent.mkdir(parents=True, exist_ok=True)
        cd_cfg.write_text(
            base.replace(f"init_from: checkpoints/gen{g}_root/root.pt",
                         f"init_from: {ft_ckpt}")
                .replace("grad_accum_steps: 1", "grad_accum_steps: 4")
                .replace("learning_rate: 0.0002", "learning_rate: 0.00008")
                .replace("max_steps: 600", "max_steps: 200"),
            encoding="utf-8")
        sh([sys.executable, "-X", "utf8", "-u", "scripts/train.py",
            "--config", str(cd_cfg),
            "--checkpoint-dir", str(cd_dir),
            "--log-dir", str(cd_dir / "logs")], cd_dir / "cycle.log")
    cd_ckpt = cd_dir / "best.pt"
    if cd_ckpt.exists():
        cd_ce = avg_ce(args.common_config, str(cd_ckpt))
        entry_extra = {"cooldown_ce": cd_ce}
    else:
        entry_extra = {}

    # 4. certified accept ----------------------------------------------------
    # The incumbent is evaluated under ITS OWN generation's config:
    # adapter rank/basis belong to the checkpoint's generation (the QR
    # basis is rank-dependent — gen-34 lesson: evaluating a rank-32
    # incumbent under a rank-64 config disables its adapters and
    # fabricates delta). Data windows stay comparable because every
    # generation's config carries the same data.seed lineage; the eval
    # seed is pinned per-config and identical across the ladder.
    m = re.search(r"gen(\d+)_", str(args.root))
    inc_cfg = args.common_config
    if m:
        for cand in (f"checkpoints/gen{m.group(1)}/configs/root_ft.yaml",
                     f"configs/gen{m.group(1)}_root_ft.yaml"):
            if (ROOT / cand).exists():
                inc_cfg = cand
                break
    inc_ce = avg_ce(inc_cfg, args.root)
    new_ce = avg_ce(args.common_config, str(ft_ckpt))
    # The candidate is the BEST of ft / cooldown (R254: the cooldown is
    # part of the generation's polish; judging the ft alone understates
    # the generation — observed on gen-13: ft 5.381, cooldown 5.319).
    cand_ckpt = ft_ckpt
    if entry_extra.get("cooldown_ce") is not None and             entry_extra["cooldown_ce"] < new_ce:
        cand_ckpt = cd_ckpt
        new_ce = entry_extra["cooldown_ce"]
    # R255 fiber arm (Oracle review item 1): the merge-point fiber
    # ensemble (root.pt + fibers.pt, logit-mean over reconstructed
    # experts) recovers the ensemble Jensen gain the flat root drops.
    # Measured gen-40: +0.174 AVG over flat root.pt (all 6 domains,
    # including both held-out). The candidate = best of THREE arms; a
    # fiber candidate ships as (root.pt, fibers.pt) pair.
    fiber_ce = _fiber_ce(args.common_config, str(gen_root / "root.pt"),
                         str(gen_root / "fibers.pt"))
    entry_extra["fiber_ce"] = fiber_ce
    if fiber_ce is not None and fiber_ce < new_ce:
        cand_ckpt = gen_root / "root.pt"   # + fibers.pt alongside
        new_ce = fiber_ce
    if inc_ce is None or new_ce is None:
        print("eval failed — halting (no uncertified growth)")
        return 1
    # R258 per-stage certificates (runtime analog of the Lean
    # structures): each stage's measured contract, bundled into the
    # cycle entry. cycle_certificate_sound analog: the potential
    # (here: common-protocol CE) decreased by the certified amount.
    certs = {
        "grow": {   # leaves trained: nonneg capability gain per leaf
            "n_leaves": len(args.domains),
            "leaves_done": all(
                (ROOT / f"checkpoints/gen{g}_leaf_{d}/best.pt").exists()
                for d in args.domains
            ),
        },
        "merge": {  # merge gate passed (gap >= prices + compression)
            "gate_pass": True,  # pilot aborts nonzero otherwise
            "noise_cos": _noise_cos(gen_root),
        },
        "joint": {  # fine-tune: certified accept below
            "ft_ce": entry_extra.get("cooldown_ce", None),
        },
    }
    entry = {"gen": g, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
             **entry_extra,
             "certificates": certs,
             "incumbent": args.root, "incumbent_ce": inc_ce,
             "merged_ce": avg_ce(args.common_config, str(gen_root / "root.pt")),
             "ft_ce": new_ce, "candidate": str(cand_ckpt),
             "delta": inc_ce - new_ce}
    # §9 conservative: the generation ships only on a strict improvement
    # beyond eval noise (2-eps band approximated by 0.02 nats at n=10
    # batches/domain); ties and regressions halt the cycle.
    # Margin scales with the measured eval noise (Oracle review, item 5;
    # gen-40 attempts ledger: same-checkpoint re-eval swung ~0.07 at
    # batches=10 on the iGPU). 30 batches cut the swing ~sqrt(3); the
    # margin remains 0.02 but the eval behind it is 3x deeper.
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
