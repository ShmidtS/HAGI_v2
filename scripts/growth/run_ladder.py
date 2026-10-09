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
DOMAINS = ["math", "lang", "code"]


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


def derive(gen: int, root_ckpt: str) -> None:
    prev = gen - 1
    for d in DOMAINS:
        s = (ROOT / f"configs/gen{prev}_leaf_{d}.yaml").read_text(encoding="utf-8")
        s = re.sub(r"init_from:.*", f"init_from: {root_ckpt}", s, count=1)
        s = re.sub(r"checkpoint_dir:.*", f"checkpoint_dir: checkpoints/gen{gen}_leaf_{d}", s, count=1)
        (ROOT / f"configs/gen{gen}_leaf_{d}.yaml").write_text(s, encoding="utf-8")
    s = (ROOT / f"configs/gen{prev}_root_ft.yaml").read_text(encoding="utf-8")
    s = re.sub(r"init_from: checkpoints/\S+", f"init_from: checkpoints/gen{gen}_root/root.pt", s, count=1)
    (ROOT / f"configs/gen{gen}_root_ft.yaml").write_text(s, encoding="utf-8")


def accepted_last() -> bool:
    last = json.loads(LADDER.read_text(encoding="utf-8").strip().splitlines()[-1])
    return bool(last.get("accepted"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from-gen", type=int, required=True)
    p.add_argument("--max-gen", type=int, default=99)
    args = p.parse_args()
    gen = args.from_gen
    while gen <= args.max_gen:
        root_ckpt = incumbent_root(gen)
        if root_ckpt is None:
            print("no incumbent root — cannot start")
            return 1
        derive(gen, root_ckpt)
        rc = subprocess.run(
            [sys.executable, "-X", "utf8", "-u",
             "scripts/growth/growth_cycle.py", "--gen", str(gen),
             "--root", root_ckpt,
             "--leaf-config", f"configs/gen{gen}_leaf_{{d}}.yaml",
             "--ft-config", f"configs/gen{gen}_root_ft.yaml"],
            cwd=str(ROOT)).returncode
        if rc != 0 or not accepted_last():
            print(f"ladder halted at gen {gen} (rc={rc})")
            return rc or 0
        gen += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
