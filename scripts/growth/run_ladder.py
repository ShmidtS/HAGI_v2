"""Autonomous ladder: run generations until the certified gate halts.

Wraps growth_cycle.py: derives each generation's leaf/ft configs from
the previous one (init_from -> latest accepted root, checkpoint_dir ->
gen N dirs), runs one generation, stops when the §9 gate declines.

Usage:
    python scripts/growth/run_ladder.py --from-gen 6 [--max-gen 12]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "checkpoints" / "growth_ladder.jsonl"
DOMAINS = ["math", "code", "ru", "enedu", "enweb", "enwiki"]


def incumbent_root(before_gen: int) -> str | None:
    entries = [json.loads(l) for l in
               LADDER.read_text(encoding="utf-8").strip().splitlines()]
    acc = [e for e in entries if e.get("accepted") and e["gen"] < before_gen]
    if not acc:
        return None
    # The TRUE incumbent is the accepted candidate (may be a cooldown
    # checkpoint, not the ft best) — reading the ladder's candidate field,
    # not a filename pattern. gen-14 was judged against the wrong root
    # before this fix.
    e = acc[-1]
    cand = e.get("candidate")
    if cand:
        p = Path(cand)
        return str(p.relative_to(ROOT)) if p.is_absolute() else cand
    return f"checkpoints/gen{e['gen']}_root_ft/best.pt"


def derive(gen: int, root_ckpt: str, domains: list[str],
           seed_base: int | None = None) -> dict:
    """Materialize gen-N configs via the SSOT builder (gen_configs.py).

    Configs live in ``checkpoints/gen<N>/configs/`` — NOT in configs/:
    no per-generation YAML proliferation. ``seed_base`` defaults to
    ``gen + 5`` (e.g. gen 41 -> 46 -> seeds 461xx), a fresh family per
    generation; pass explicitly to retry a generation with a new family.
    """
    from gen_configs import write_generation
    if seed_base is None:
        seed_base = gen + 5
    return write_generation(gen, root_ckpt, seed_base,
                            domains=domains)


def accepted_last() -> bool:
    last = json.loads(LADDER.read_text(encoding="utf-8").strip().splitlines()[-1])
    return bool(last.get("accepted"))


def record_stage_certificate() -> int | None:
    """§0 certificate-telescope: append the accepted generation's
    StageCertificate to checkpoints/stage_telescope.jsonl (one chain
    entry per ACCEPTED generation — rejected generations keep the
    incumbent, no stage). Returns the generation number, or None when
    the entry could not be certified from the ladder record.
    """
    from hagi.train.stage_telescope import (
        certificate_from_ladder_entry, append_certificate,
        load_ledger, telescope_bound,
    )
    last = json.loads(LADDER.read_text(encoding="utf-8").strip().splitlines()[-1])
    cert = certificate_from_ladder_entry(last)
    if cert is None:
        return None
    # contiguity vs the previous accepted stage: eval-noise allowance
    # 2*accept_eps (Hoeffding) — explicit, not hidden.
    tol = 2.0 * float(last.get("accept_eps") or 0.0)
    chain = load_ledger(ROOT / "checkpoints")
    if chain:
        telescope_bound(chain + [cert], contiguity_tol=max(tol, 1e-9))
    append_certificate(ROOT / "checkpoints", cert)
    return cert.generation


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from-gen", type=int, required=True)
    p.add_argument("--domains", nargs="+", default=DOMAINS)
    p.add_argument("--max-gen", type=int, default=99)
    p.add_argument("--expand-after", type=int, default=2,
                   help="consecutive rejections before EXPANDING the "
                        "ladder with extra fresh-seed lanes (R255: the "
                        "noise floor deepens linearly in N)")
    p.add_argument("--expand-lanes", nargs="+",
                   default=["math2", "code2"],
                   help="expansion lane suffixes (base domains reused "
                        "with a disjoint seed family)")
    p.add_argument("--hard-stop", type=int, default=4,
                   help="consecutive rejections after expansion that "
                        "halt for good (honest floor)")
    args = p.parse_args()
    gen = args.from_gen
    domains = list(args.domains)
    rejects = 0
    expanded = False
    while gen <= args.max_gen:
        root_ckpt = incumbent_root(gen)
        if root_ckpt is None:
            print("no incumbent root — cannot start")
            return 1
        derive(gen, root_ckpt, domains)
        cfg_dir = f"checkpoints/gen{gen}/configs"
        rc = subprocess.run(
            [sys.executable, "-X", "utf8", "-u",
             "scripts/growth/growth_cycle.py", "--gen", str(gen),
             "--root", root_ckpt,
             "--leaf-config", f"{cfg_dir}/leaf_{{d}}.yaml",
             "--ft-config", f"{cfg_dir}/root_ft.yaml",
             "--common-config", f"{cfg_dir}/root_ft.yaml",
             "--domains", *domains],
            cwd=str(ROOT)).returncode
        if rc == 0 and accepted_last():
            gen_done = record_stage_certificate()
            if gen_done is not None:
                print(f"[ladder] gen {gen_done}: stage certificate appended "
                      f"(§0 telescope)")
            rejects = 0
            gen += 1
            continue
        rejects += 1
        if not expanded and rejects >= args.expand_after:
            domains = domains + [d for d in args.expand_lanes
                                 if d not in domains]
            expanded = True
            rejects = 0
            print(f"[ladder] {args.expand_after} rejections — EXPANDING "
                  f"to {len(domains)} leaves: {domains} (R255 N-floor)")
            gen += 1
            continue
        if expanded and rejects >= args.hard_stop:
            print(f"ladder halted at gen {gen} (rc={rc}): floor even "
                  f"after expansion — honest stop")
            return rc or 0
        print(f"[ladder] gen {gen} rejected ({rejects} consecutive) — "
              f"fresh-seed retry")
        gen += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
