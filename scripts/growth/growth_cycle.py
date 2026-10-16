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


def _cuda_ok() -> bool:
    """iGPU available (advisory measurements run on the idle GPU —
    the CPU path cost ~5.5h wall on gen68)."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _logit_twoGap(gen: int, domains: list[str]) -> tuple[float, float] | None:
    """Formula-exact logit twoGap of this generation's leaves
    (math domain, 1 batch, idle-GPU-else-CPU) with the §AW split-half
    bracket: measured twice on disjoint eval samples (seeds
    15111/15112), eps_gamma = |a-b|/2 is the measurement uncertainty
    of Gamma.
    Returns (mean, eps_gamma); None on any failure (advisory —
    never blocks the cycle)."""
    import subprocess as _sp
    vals: list[float] = []
    for seed in (15111, 15112):
        r = _sp.run(
            [sys.executable, "-X", "utf8", "-u",
             str(ROOT / "scripts/growth/measure_logit_twogap.py"),
             "--gen", str(gen), "--domains", *domains,
             # GPU idle at both call sites (end-of-gen certificates /
             # pre-leaf selector): the CPU measurement cost ~5.5h wall on
             # gen68 (10 leaves × full-vocab logsumexp) vs seconds on the
             # iGPU — measured 2026-10-13. Same fp32 math, split-half
             # seeds unchanged; falls back to CPU if no CUDA.
             "--batches", "1", "--domain", "math", "--device",
             ("cuda" if _cuda_ok() else "cpu"),
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


def _thermostat_gate(
    gen: int,
    last_delta: float | None,
    t0: float, alpha: float, theta: float, start_gen: int,
) -> dict:
    """R388 thermostat gate: exploration stops BY LAW when the Gibbs
    competitor mass e^{-delta/T_t} <= theta (T_t = T0*alpha^t,
    t = gen - start_gen). ``last_delta`` is the measured (h_emp_)
    certified A/B delta — the energy gap between the incumbent and
    the candidate ensemble. Returns the phase record for the selector
    meta; ``phase == EXPLOIT`` means: no further leaf exploration is
    lawful — switch to consolidation/cooldown levers (honest stop,
    same family as E_dev exhausted / AMBIGUOUS)."""
    from hagi.train import thermostat as th

    if last_delta is None or last_delta <= 0:
        return {"active": False,
                "reason": "delta unmeasured (no accepted generation yet)"}
    cfg = th.ThermostatConfig(t0=t0, alpha=alpha, theta=theta)
    t = max(0, gen - start_gen)
    ph, mass = th.phase(cfg, last_delta, t)
    n = th.stop_horizon(cfg, last_delta)
    return {
        "active": True,
        "t": t,
        "delta": last_delta,
        "temperature": th.temperature(cfg, t),
        "competitor_mass": mass,
        "theta": theta,
        "horizon": n,
        "phase": ph.value,
    }


def _select_next_action(
    gen: int,
    domains: list[str],
    thermostat_cfg: tuple[float, float, float, int] | None = None,
) -> dict:
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
    eps_gamma = None  # cold start: no bracket yet — see bootstrap below
    if last is None:
        # Cold start (2026-10-16 from-scratch ladder): the very first
        # generation has no measured Gammas by construction — the cycle
        # IS the bootstrap (grow leaves from the seed root prior, merge
        # them). Honest default: merge action with an explicit
        # bootstrap marker; no fake Gamma is recorded.
        meta["bootstrap"] = True
        meta["chosen"] = "merge"
        meta["aw"] = {"verdict": "BOOTSTRAP",
                      "note": "first generation: no prior measurements"}
        return meta
    if last is not None:
        certs = last.get("certificates", {})
        two_gap = certs.get("merge", {}).get("logit_twoGap")
        eps_gamma = certs.get("merge", {}).get("eps_gamma")
        last_gen = last.get("gen")
        if two_gap is None and last_gen is not None:
            # Backfill: the last generation's twoGap is missing (older
            # entry or a measurement failure) but its leaves may still
            # be on disk — measure NOW instead of silently dropping the
            # merge action (a missing Gamma is not a zero Gamma).
            fresh = _logit_twoGap(int(last_gen), domains)
            if fresh is not None:
                two_gap, eps_gamma = fresh
                meta["two_gap_remeasured"] = True
        if two_gap is not None:
            meas["two_gap"] = float(two_gap)
            meas["merge_cost"] = 1.0  # cost unmeasured -> Gamma ranking
            meta["two_gap"] = two_gap
        delta = last.get("delta")
        if delta is not None and delta > 0:
            meas["g_new"] = float(delta)
            # grow_cost stays UNMEASURED (defaults to 0.0): estimate_actions
            # SUBTRACTS it from Gamma in gain units (test_certified_controller:
            # 0.50-0.15), so a placeholder in nats would flip the sign and
            # fake "E_dev exhausted" (the gen52-56 halt artifact). K for
            # ranking = grow_exec_cost (wall-clock, unmeasured -> 1.0).
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
    # R433-R444 hedge re-weighting: the argmax over PREDICTED Gamma must
    # answer to the REALIZED losses (18 merge rejections in a row at
    # gen63-80 while two_gap stayed ~3.7 = the Gamma premise is
    # empirically broken for the end-to-end cycle). MWU weights
    # w_a ∝ exp(-eta* L_a) with L_a = per-action realized generation
    # losses (rejected generation = 1, accepted = 0); eta* = ln n /
    # (2<ell> - ell_min) is DERIVED (R443/R444), not tuned. When the
    # hedge distribution no longer favours the argmax action, the
    # selector records the switch signal (the cycle halts with the
    # diagnosis — a rejected lever is SWITCHED, not retried).
    try:
        from hagi.train.mwu import eta_star
        entries = [json.loads(l) for l in
                   LADDER.read_text(encoding="utf-8").strip().splitlines()]
        # realized loss per chosen action over the recorded history
        # realized losses count only EXECUTED rounds (an entry with
        # cooldown_ce measured = the generation actually ran) and only
        # since the last lane expansion (a new data regime is a new
        # game: R255 fresh-seed lanes must get an unprejudiced trial).
        state_p = ROOT / "checkpoints" / "ladder_state.json"
        exp_gen = -1
        if state_p.exists():
            try:
                _state = json.loads(state_p.read_text(encoding="utf-8"))
                # mwu_from_gen: first generation whose verdicts come from a
                # trustworthy accept gate. Losses recorded before it were
                # produced by a broken measurement instrument (the gen52-86
                # hoeffding ceil off-by-one froze every verdict at UNDECIDED)
                # and must not falsify the merge-Gamma premise.
                exp_gen = int(_state.get("mwu_from_gen", _state.get("expansion_gen", -1)))
            except (ValueError, OSError):
                exp_gen = -1
        losses: dict[str, list[float]] = {}
        for e in entries:
            if e.get("gen", 0) <= exp_gen:
                continue
            c = e.get("certificates", {}).get("selector", {})
            act = c.get("chosen")
            if act is None or "cooldown_ce" not in e:
                continue  # selector-halt no-op: not a played round
            losses.setdefault(act, []).append(0.0 if e.get("accepted") else 1.0)
        names = sorted({a.name for a in actions} | set(losses))
        if len(names) > 1:
            seqs = [losses.get(nm, [0.0]) for nm in names]
            flat = [sum(s) / max(len(s), 1) for s in seqs]
            try:
                eta = float(eta_star(flat))
            except ValueError:
                eta = 1.0
            import math as _math
            wts = [_math.exp(-eta * sum(losses.get(nm, [0.0])))
                   for nm in names]
            tot = sum(wts)
            hedge = {nm: round(wt / tot, 4)
                     for nm, wt in zip(names, wts)}
            meta["hedge"] = {"eta": round(eta, 4), "weights": hedge}
            top_hedge = max(hedge, key=hedge.get)
            if chosen is not None and top_hedge != chosen.name \
                    and hedge[top_hedge] > 0.5:
                meta["hedge"]["switch_signal"] = top_hedge
    except Exception as exc:  # telemetry only — never block selection
        meta["hedge"] = {"error": repr(exc)}
    meta["aw"] = aw_meta
    # R388 thermostat (opt-in): lawful exploration-stop circuit
    if thermostat_cfg is not None:
        meta["thermostat"] = _thermostat_gate(
            gen, last.get("delta") if last is not None else None,
            *thermostat_cfg)
    # back-compat audit field: raw top1-top2 ratio gap
    ratios = sorted((a.ratio for a in actions), reverse=True)
    meta["argmax_gap"] = (ratios[0] - ratios[1]) if len(ratios) > 1 else None
    # R530 gamma-deficit verdict (Growth/GammaDeficit.lean, port
    # 2026-10-16): honest takeoff requires a mixer-harvest budget
    # gamma >= alpha * C* / Dbar. Measured premises from the selector's
    # own quantities: C* = grow_leaf Gamma (capability target rate),
    # Dbar = twoGap (diversity proxy — the disagreement the mixer can
    # harvest), gamma = grow_leaf Gamma / twoGap as the realized harvest
    # coefficient, alpha = the grow gate fraction 0.5. Advisory
    # annotation (non-blocking): a deficit verdict means the only lever
    # is disagreement_distill / fresh diversity, not tuning.
    try:
        from hagi.train.certified_controller import (
            gamma_deficit_no_takeoff,
        )
        grow_g = next((a.gamma for a in actions
                       if a.name == "grow_leaf"), None)
        merge_g = next((a.gamma for a in actions
                        if a.name == "merge"), None)
        if grow_g is not None and merge_g and merge_g > 0:
            alpha_gate = 0.5
            gamma_real = grow_g / merge_g
            deficit, gamma_req = gamma_deficit_no_takeoff(
                gamma_real, alpha_gate, c_star=grow_g, d_bar=merge_g)
            meta["gamma_deficit"] = {
                "gamma": round(gamma_real, 4),
                "gamma_required": round(gamma_req, 4),
                "deficit": deficit,
                "ceiling": round(gamma_real * merge_g / alpha_gate, 4),
            }
    except Exception as exc:  # telemetry only
        meta["gamma_deficit"] = {"error": repr(exc)}
    return meta


def _ckpt_embedded_config_yaml(ckpt: str) -> Path | None:
    """Materialize the checkpoint's EMBEDDED config as a temp YAML.

    Fix for the depth-7 root-override freeze (2026-10-16): deriving
    the incumbent's eval config from the gen id in --root guesses the
    ARCHITECTURE (gen51 -> 3-layer config) and crashes on a grown
    checkpoint (gen51_depth7 -> 7 layers) with an empty-subprocess
    "eval failed" — the ladder halted gens 87-89 on this. The
    checkpoint carries its true config (format_version >= 12 embeds
    it); the embedded one is also the honest protocol (it is the mix
    the incumbent actually trained under; verified identical domain
    weights to the current generation's config). Returns None when
    the checkpoint has no embedded config (legacy) — callers fall
    back to the old gen-id guess.
    """
    import hashlib
    import torch
    try:
        ck = torch.load(str(ROOT / ckpt if not Path(ckpt).is_absolute()
                            else ckpt),
                        map_location="cpu", weights_only=False)
        cfg_dict = ck.get("config") if isinstance(ck, dict) else None
        if not cfg_dict:
            return None
    except Exception:
        return None
    import yaml
    digest = hashlib.sha256(
        json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:16]
    out = ROOT / "checkpoints" / f"_eval_cfg_{digest}.yaml"
    if not out.exists():
        out.write_text(yaml.safe_dump(cfg_dict, sort_keys=True),
                       encoding="utf-8")
    return out


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
    """Common-protocol AVG exact CE of a checkpoint (None on failure).

    Кэш (checkpoint/_ce_cache.jsonl): eval seed закреплён per-config
    (детерминизм задокументирован), поэтому CE = f(config, ckpt,
    batches). Инкамбент пере-оценивается на каждой ступени глубокого
    §9-eval КАЖДОГО поколения (2400 батчей × 10 доменов — часы GPU);
    кэш режет это до одного прогона на (ckpt, n) за всю лестницу.
    Ключ включает mtime+size чекпойнта — замена файла инвалидирует.
    """
    import hashlib
    ck = Path(ckpt)
    st = ck.stat()
    key_src = f"{config}|{ck.resolve()}|{batches}|{st.st_mtime_ns}|{st.st_size}"
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
    cache = ROOT / "checkpoints" / "_ce_cache.jsonl"
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("key") == key:
                    return float(rec["ce"])
    out = ROOT / "checkpoints" / "_cycle_eval.json"
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "-u", str(ROOT / "scripts/eval_domains.py"),
         "--config", config, "--resume", ckpt, "--batches", str(batches),
         "--device", "cuda"],
        capture_output=True, text=True, cwd=str(ROOT))
    out.write_text(r.stdout, encoding="utf-8")
    for line in r.stdout.splitlines():
        if line.strip().startswith("AVG"):
            ce = float(line.split("exact_ce=")[1].split()[0])
            with cache.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "ce": ce,
                                     "config": str(config), "ckpt": str(ckpt),
                                     "batches": batches}) + "\n")
            return ce
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
    p.add_argument("--anytime-horizon", type=int, default=100,
                   help="R525 anytime validity: per-generation level "
                        "delta = 0.05/horizon (union bound over the "
                        "ladder's generations; 0 disables)")
    p.add_argument("--thermostat-t0", type=float, default=None,
                   help="R388 thermostat: enable with T0 (explore/exploit "
                        "circuit; delta = last accepted A/B delta)")
    p.add_argument("--thermostat-alpha", type=float, default=0.9)
    p.add_argument("--thermostat-theta", type=float, default=0.05)
    p.add_argument("--thermostat-start-gen", type=int, default=None,
                   help="generation index where cooling starts (default: this gen)")
    args = p.parse_args()
    g = args.gen

    # §0/§1 action selector (R86 ratio_dominance): argmax Gamma_i/K_i
    # from the previous generation's MEASURED certificates. No
    # certified-positive action at all == E_dev exhausted (R223
    # round_exhausted) -> the lever must be switched (data/rank), not
    # retried; the cycle refuses to burn compute on an unviable round.
    thermostat_cfg = None
    if args.thermostat_t0 is not None:
        thermostat_cfg = (args.thermostat_t0, args.thermostat_alpha,
                          args.thermostat_theta,
                          args.thermostat_start_gen
                          if args.thermostat_start_gen is not None else g)
    selector = _select_next_action(g, args.domains, thermostat_cfg)
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

    # R388 thermostat (opt-in): exploration exhausted BY LAW — the
    # Gibbs competitor mass dropped below theta, further leaf growth
    # would burn compute on an annealed-out lever. Honest stop, same
    # family as E_dev exhausted / AMBIGUOUS.
    th_state = (selector.get("thermostat") or {})
    if th_state.get("active") and th_state.get("phase") == "exploit":
        print(f"[gen{g}] §Thermostat R388: exploration exhausted by law "
              f"(competitor_mass {th_state['competitor_mass']:.4g} <= "
              f"theta {th_state['theta']}, T={th_state['temperature']:.4g}, "
              f"horizon N={th_state['horizon']}) — switch to "
              "consolidation/cooldown levers, halting")
        return 0

    # R433-R444 hedge switch (MWU/hedge, realized-loss reweighting):
    # when the exponential-weights distribution over actions no longer
    # favours the argmax pick (merge lost 25+ generations in a row at
    # gen56-80 while its PREDICTED Gamma stayed ~3.7 — the Gamma
    # premise is empirically broken), the lever is SWITCHED, not
    # retried. Honest halt with the diagnosis; the switch itself is
    # a human/new-data decision (fresh lanes, new rank, distill).
    hedge = selector.get("hedge") or {}
    if "switch_signal" in hedge:
        print(f"[gen{g}] §MWU R433-R444 switch: argmax '{selector['chosen']}' "
              f"but hedge weights {hedge['weights']} (eta={hedge['eta']}) "
              f"favour '{hedge['switch_signal']}' — the merge Gamma "
              "premise is falsified by the realized rejection ledger, "
              "halting (switch the lever: expand lanes / new data / "
              "distill, do not re-merge)")
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
        import re as _re
        cd_cfg.write_text(
            _re.sub(r"max_steps: \d+", "max_steps: 200",
                _re.sub(r"batch_growth_delta: [0-9.]+", "batch_growth_delta: null",
                    base
                    .replace(f"init_from: checkpoints/gen{g}_root/root.pt",
                             f"init_from: {ft_ckpt}")
                    .replace("grad_accum_steps: 1", "grad_accum_steps: 4")
                    .replace("learning_rate: 0.0002", "learning_rate: 0.00008"))),
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
    # Architecture-true incumbent config: prefer the checkpoint's
    # EMBEDDED config (correct for depth-grown / overridden roots —
    # the gen-id guess loads a mismatched architecture and kills the
    # eval subprocess with empty stdout, freezing the ladder);
    # fall back to the gen-id guess for legacy checkpoints without
    # an embedded config.
    inc_cfg = (_ckpt_embedded_config_yaml(args.root)
               or args.common_config)
    if inc_cfg is args.common_config and m:
        for cand in (f"checkpoints/gen{m.group(1)}/configs/root_ft.yaml",
                     f"configs/gen{m.group(1)}_root_ft.yaml"):
            if (ROOT / cand).exists():
                inc_cfg = cand
                break
    inc_ce = avg_ce(str(inc_cfg), args.root)
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
    # R525 anytime validity (CertifiedEstimator.lean, port 2026-10-16):
    # a gate evaluated at EVERY generation with a fixed level alpha
    # inflates the family-wise false-accept mass T-fold. The uniform
    # level over the ladder horizon is delta_gen = alpha / T (the
    # R523 union bound behind anytime_safe_threshold). Enabled by
    # default (--anytime-horizon 0 turns it off).
    level = (0.05 / args.anytime_horizon
             if args.anytime_horizon and args.anytime_horizon > 0
             else 0.05)
    entry["accept_level"] = level
    n_eval = 30 * len(args.domains)  # batches per domain * domains
    eps_cert = math.sqrt(math.log(2.0 / level) / (2.0 * n_eval))
    verdict = certified_ab(inc_ce, n_eval, new_ce, n_eval,
                           eps=eps_cert, delta=level)
    prev_delta = None
    for deep_batches in (150, 600, 2400):
        if not (verdict.undecided and entry["delta"] > 0):
            break
        eps_next = math.sqrt(math.log(2.0 / level)
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
        eps_cert = math.sqrt(math.log(2.0 / level) / (2.0 * n_eval))
        verdict = certified_ab(inc_ce, n_eval, new_ce, n_eval,
                               eps=eps_cert, delta=level)
    entry["accept_eps"] = round(eps_cert, 4)
    entry["accept_n"] = n_eval
    entry["accept_required_n"] = hoeffding_n(eps_cert, level)
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
