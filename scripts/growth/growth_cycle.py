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
import math
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


def _logit_twoGap(gen: int, domains: list[str]) -> tuple[float, float] | None:
    """Formula-exact logit twoGap of this generation's leaves
    (math domain, 1 batch, CPU) with the §AW split-half bracket:
    measured twice on disjoint eval samples (seeds 15111/15112),
    eps_gamma = |a-b|/2 is the measurement uncertainty of Gamma.
    Returns (mean, eps_gamma); None on any failure (advisory —
    never blocks the cycle)."""
    import subprocess as _sp
    vals: list[float] = []
    for seed in (15111, 15112):
        r = _sp.run(
            [sys.executable, "-X", "utf8", "-u",
             str(ROOT / "scripts/growth/measure_logit_twogap.py"),
             "--gen", str(gen), "--domains", *domains,
             "--batches", "1", "--domain", "math", "--device", "cpu",
             "--seed", str(seed)],
            capture_output=True, text=True, cwd=str(ROOT))
        try:
            import json as _json
            d = _json.loads(r.stdout[r.stdout.index("{"):])
            v = d.get("ensemble_mean_twoGap")
            if v is not None:
                vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return None
    if len(vals) == 1:  # one half failed: value without a bracket
        return vals[0], float("nan")
    return (vals[0] + vals[1]) / 2.0, abs(vals[0] - vals[1]) / 2.0


def _last_ladder_entry() -> dict | None:
    """Last line of growth_ladder.jsonl (None if empty)."""
    if not LADDER.exists():
        return None
    lines = LADDER.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def _select_next_action(gen: int, domains: list[str]) -> dict:
    """§0/§1 action selector: argmax Gamma_i/K_i from MEASURED quantities
    of the previous generation (ratio_dominance R86 — concentrating the
    budget on the argmax Gamma/K dominates any split).

    Measurements (all from the last ladder entry, i.e. already-paid
    observations — no extra compute):
      - merge:  Gamma = logit_twoGap (GapLaw, exact Lean identity);
                K = last generation wall-clock seconds.
      - grow_leaf: Gamma = last accepted delta (capability gain);
                K = last generation wall-clock seconds.
      - distill: c_k/delta_k are not measured yet -> action omitted
        (estimate_actions skips missing keys — honest, not guessed).

    Returns the decision record; the caller logs it. When the certified
    argmax is not the default merge cycle, the cycle HALTS with the
    diagnosis — principle 6: a rejected/exhausted lever is SWITCHED,
    not retried (the actual switch is a human/new-data decision).
    """
    from hagi.train.certified_controller import (
        certified_argmax, estimate_actions,
    )
    last = _last_ladder_entry()
    meas: dict = {}
    meta: dict = {"gen": gen}
    if last is not None:
        certs = last.get("certificates", {})
        two_gap = certs.get("merge", {}).get("logit_twoGap")
        if two_gap is not None:
            meas["two_gap"] = float(two_gap)
            meas["merge_cost"] = 1.0  # cost unmeasured -> Gamma ranking
            meta["two_gap"] = two_gap
        eps_gamma = certs.get("merge", {}).get("eps_gamma")
        delta = last.get("delta")
        if delta is not None and delta > 0:
            meas["g_new"] = float(delta)
            meas["grow_cost"] = 1.0
            meta["g_new"] = delta
    actions = estimate_actions(meas)
    # §AW (CertifiedArgmax + tie_ambiguity): the argmax over MEASURED
    # Gammas is legitimate only when top1-top2 ratio gap > 2*eps_gamma
    # (split-half bracket, gen<N> certificates.merge.eps_gamma). Inside
    # the bracket the choice is AMBIGUOUS -> halt with the diagnosis
    # instead of acting on noise.
    chosen, aw_meta = certified_argmax(
        actions, float(eps_gamma) if eps_gamma is not None else 0.0)
    meta["actions"] = [{"name": a.name, "gamma": a.gamma,
                        "cost": a.cost, "ratio": a.ratio} for a in actions]
    meta["chosen"] = chosen.name if chosen else None
    meta["aw"] = aw_meta
    # back-compat audit field: raw top1-top2 ratio gap
    ratios = sorted((a.ratio for a in actions), reverse=True)
    meta["argmax_gap"] = (ratios[0] - ratios[1]) if len(ratios) > 1 else None
    return meta


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

    # §0/§1 action selector (R86 ratio_dominance): argmax Gamma_i/K_i
    # from the previous generation's MEASURED certificates. No
    # certified-positive action at all == E_dev exhausted (R223
    # round_exhausted) -> the lever must be switched (data/rank), not
    # retried; the cycle refuses to burn compute on an unviable round.
    selector = _select_next_action(g, args.domains)
    print(f"[gen{g}] §1 selector: {json.dumps(selector)}")
    if selector["chosen"] is None:
        aw = selector.get("aw", {})
        if aw.get("verdict") == "AMBIGUOUS":
            print(f"[gen{g}] §AMBIGUOUS argmax (tie_ambiguity): top1-top2 "
                  f"ratio gap {aw.get('gap')} <= 2*eps_gamma "
                  f"{aw.get('threshold')} — selection would act on "
                  "noise, halting")
        else:
            print(f"[gen{g}] no certified-positive action (E_dev exhausted, "
                  "R223) — switch the lever (fresh data / rank), halting")
        return 0

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
        "selector": selector,  # §0/§1 argmax Gamma/K decision (pre-cycle)
        "grow": {   # leaves trained: nonneg capability gain per leaf
            "n_leaves": len(args.domains),
            "leaves_done": all(
                (ROOT / f"checkpoints/gen{g}_leaf_{d}/best.pt").exists()
                for d in args.domains
            ),
        },
        "merge": {  # merge gate: weight-level prices (pilot aborts
            # nonzero) + the EXACT Lean GapLaw certificate (logit twoGap).
            # BLOCKING (theory §3 + GapLaw.twoGap_zero_iff): twoGap = 0
            # iff consensus — an ensemble of agreeing leaves has ZERO
            # merge gain, the round is exhausted (R223 round_exhausted):
            # fine-tune would burn compute on an unviable round.
            "gate_pass": True,  # pilot aborted nonzero otherwise
            "noise_cos": _noise_cos(gen_root),
            "logit_twoGap": None,  # filled below (value + eps bracket)
        },
        "joint": {  # fine-tune: certified accept below
            "ft_ce": entry_extra.get("cooldown_ce", None),
        },
    }
    # §AW split-half bracket: measure twoGap on two disjoint eval
    # halves; eps_gamma feeds certified_argmax (CertifiedArgmax /
    # tie_ambiguity: argmax is legit only when top1-top2 > 2*eps_gamma).
    _tg = _logit_twoGap(g, args.domains)
    if _tg is not None:
        certs["merge"]["logit_twoGap"] = _tg[0]
        if _tg[1] == _tg[1]:  # not NaN
            certs["merge"]["eps_gamma"] = _tg[1]
    two_gap_exact = certs["merge"]["logit_twoGap"]
    entry = {"gen": g, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
             **entry_extra,
             "certificates": certs,
             "incumbent": args.root, "incumbent_ce": inc_ce,
             "merged_ce": avg_ce(args.common_config, str(gen_root / "root.pt")),
             "ft_ce": new_ce, "candidate": str(cand_ckpt),
             "delta": inc_ce - new_ce}
    # Blocking exact-merge certificate (GapLaw.twoGap_zero_iff): a
    # zero/negative exact twoGap = consensus = ZERO ensemble gain
    # (R223 round_exhausted) — the accept below cannot certify growth.
    two_gap_exact = certs["merge"]["logit_twoGap"]
    if two_gap_exact is not None and two_gap_exact <= 0.0:
        certs["merge"]["gate_pass"] = False
        entry["accepted"] = False
        with LADDER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(f"[gen{g}] exact twoGap = {two_gap_exact:.4f} <= 0 (consensus, "
              "GapLaw.twoGap_zero_iff / R223) — round exhausted, halting")
        return 0
    # §9 certified accept (THEORY-EXACT, R114/R118): accept iff
    # delta > 2*eps at n >= log(2/delta)/(2 eps^2). The eval samples
    # n = batches * n_domains batches; eps is DERIVED from the actual
    # sample size at delta=0.05 (the deep-200 protocol anchor: n=200
    # -> eps~=0.096), so the margin is never a hand-tuned constant.
    # UNDECIDED with a positive observed delta = not enough evidence:
    # DEEPEN the eval (deep-200 heritage). Adaptive ladder
    # 30 -> 150 -> 600 -> 2400 batches/domain: eps shrinks as 1/sqrt(n)
    # (n = batches * n_domains), so each rung certifies deltas of
    # ~2*eps; a persistent positive-but-small delta keeps deepening
    # until certified or the cap — REJECT with delta>0 never fires,
    # so the loop is finite by construction. Inference-only cost.
    from hagi.train.certified_controller import certified_ab, hoeffding_n
    n_eval = 30 * len(args.domains)  # batches per domain * domains
    eps_cert = math.sqrt(math.log(2.0 / 0.05) / (2.0 * n_eval))
    verdict = certified_ab(inc_ce, n_eval, new_ce, n_eval,
                           eps=eps_cert, delta=0.05)
    prev_delta = None
    for deep_batches in (150, 600, 2400):
        if not (verdict.undecided and entry["delta"] > 0):
            break
        eps_next = math.sqrt(math.log(2.0 / 0.05)
                             / (2.0 * deep_batches * len(args.domains)))
        # noise-chase guard: a delta that did not GROW between rungs
        # while staying below the next rung's certification margin is
        # a noise floor, not a pending certificate — deepening further
        # burns GPU for an estimate that will not move (measured
        # gen51: +0.0112@240 -> +0.0022@1200 -> +0.0022@4800).
        if (prev_delta is not None and entry["delta"] <= prev_delta
                and entry["delta"] < 2.0 * eps_next):
            print(f"[gen{g}] delta {entry['delta']:+.4f} flat below the "
                  f"next margin {2*eps_next:.4f} — noise floor, "
                  "stopping the deepening (honest reject)")
            break
        prev_delta = entry["delta"]
        print(f"[gen{g}] UNDECIDED at n={n_eval} (2*eps={2*eps_cert:.4f}, "
              f"delta={entry['delta']:+.4f}) — deepening the eval to "
              f"{deep_batches} batches/domain (adaptive deep-200)")
        inc_ce_d = avg_ce(inc_cfg, args.root, batches=deep_batches)
        new_ce_d = avg_ce(args.common_config, str(cand_ckpt), batches=deep_batches)
        if inc_ce_d is None or new_ce_d is None:
            break
        inc_ce, new_ce = inc_ce_d, new_ce_d
        entry["incumbent_ce"], entry["ft_ce"] = inc_ce, new_ce
        entry["delta"] = inc_ce - new_ce
        n_eval = deep_batches * len(args.domains)
        eps_cert = math.sqrt(math.log(2.0 / 0.05) / (2.0 * n_eval))
        verdict = certified_ab(inc_ce, n_eval, new_ce, n_eval,
                               eps=eps_cert, delta=0.05)
    entry["accept_eps"] = round(eps_cert, 4)
    entry["accept_n"] = n_eval
    entry["accept_required_n"] = hoeffding_n(eps_cert, 0.05)
    entry["accepted"] = verdict.accepted
    with LADDER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(json.dumps(entry, indent=2))
    if not entry["accepted"]:
        print(f"[gen{g}] NOT certified ({verdict.verdict}, delta {entry['delta']:+.4f} "
              f"vs 2*eps={2*eps_cert:.4f} at n={n_eval}) "
              "— cycle halts; incumbent stands (§9)")
        return 0
    print(f"[gen{g}] accepted: next gen uses {ft_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
