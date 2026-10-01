"""Per-domain benchmark: does growth preserve and beat its experts?

The stopping condition for this project claims three things about HAGI --
that it GROWS (each generation better than the last), that it is
GENERAL (not one domain), and that it is SELF-IMPROVING. This script
measures all three on checkpoints that already exist, so the claims are
backed by numbers rather than by a training log.

What it reports
---------------
Per model: exact full-alphabet CE on RU / EN / MATH / CODE windows and
the average. Then three derived verdicts:

``growth``
    AVG CE of this generation vs the previous one. Lower is growth.

``generalization``
    The WORST domain CE relative to the model's own best domain, and the
    spread. A model that is uniformly mediocre is not general; a model
    whose worst domain is close to its best one is.

``merge_is_not_a_regression``
    For each expert, the merged model's CE on that expert's OWN domain
    vs the expert's CE there. The ternary merge is only worth doing if
    the merged model is not worse than the specialist it absorbed -- if
    it were, the specialist was strictly better than the merge and the
    growth step destroyed information.

Usage::

    python scripts/growth_benchmark.py --batches 12 \
        --gen3 checkpoints/dbridge_gen3_joint/step-0001600.pt \
        --prev checkpoints/dbridge_gen2_joint/step-0001600.pt \
        --expert math=checkpoints/gen2_dsib_math/step-0001600.pt \
        --expert lang=checkpoints/gen2_dsib_lang/step-0001600.pt \
        --expert code=checkpoints/gen2_dsib_code/step-0001600.pt

Every measurement reuses ``eval_domains.py``'s loader and windows, so the
numbers are comparable across models by construction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "eval_domains.py"


def evaluate(config: str, checkpoint: str, batches: int, device: str) -> dict:
    """Run eval_domains on one checkpoint and parse its per-domain CEs."""
    out = subprocess.run(
        [
            sys.executable, "-u", str(EVAL),
            "--config", config,
            "--resume", checkpoint,
            "--batches", str(batches),
            "--device", device,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"eval failed for {checkpoint}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
        )
    domains: dict[str, float] = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].startswith("exact_ce="):
            name = parts[0]
            domains[name] = float(parts[1].split("=", 1)[1])
    if not domains:
        raise RuntimeError(f"no domain lines parsed from:\n{out.stdout[-2000:]}")
    return domains


def report(name: str, d: dict) -> dict:
    values = [v for k, v in d.items() if k != "AVG"]
    best, worst = min(values), max(values)
    avg = d.get("AVG", sum(values) / len(values))
    return {
        "model": name,
        "domains": d,
        "avg": avg,
        "best_domain_ce": best,
        "worst_domain_ce": worst,
        # 0 = uniform across domains (fully general), 1 = all the skill
        # concentrated in one domain.
        "domain_spread": (worst - best) / avg if avg else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dbridge_gen3_joint.yaml")
    ap.add_argument(
        "--gen", required=True, metavar="NAME=CONFIG=PATH",
        help="current generation: name, the config that BUILDS it, and its checkpoint",
    )
    ap.add_argument(
        "--prev", metavar="NAME=CONFIG=PATH",
        help="previous generation, same form",
    )
    ap.add_argument(
        "--expert", action="append", default=[], metavar="DOMAIN=CONFIG=PATH",
        help="a domain expert absorbed by the merge (repeatable)",
    )
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    def split_spec(spec: str) -> tuple[str, str, str]:
        """``NAME=CONFIG=PATH`` -> the three parts.

        Every model needs the config that BUILDS it: the mixer rank and
        type differ between the lines, so loading gen-2's checkpoint
        through gen-3's config raises a shape mismatch.
        """
        parts = spec.split("=")
        if len(parts) != 3:
            raise SystemExit(
                f"expected NAME=CONFIG=PATH, got {spec!r}"
            )
        return parts[0], parts[1], parts[2]

    results = []
    gen_name, gen_config, gen_path = split_spec(args.gen)
    results.append(
        report(gen_name, evaluate(gen_config, gen_path, args.batches, args.device))
    )

    if args.prev:
        name, cfg, path = split_spec(args.prev)
        results.append(report(name, evaluate(cfg, path, args.batches, args.device)))

    experts: dict[str, dict] = {}
    for spec in args.expert:
        domain, cfg, path = split_spec(spec)
        experts[domain] = evaluate(cfg, path, args.batches, args.device)

    merged = results[0]
    print()
    print("=== per-domain exact CE ===")
    keys = sorted(set(merged["domains"]) - {"AVG"})
    header = "model".ljust(16) + "".join(k.rjust(10) for k in keys) + "AVG".rjust(10)
    print(header)
    for r in results:
        row = r["model"].ljust(16)
        row += "".join(f"{r['domains'].get(k, float('nan')):10.4f}" for k in keys)
        row += f"{r['avg']:10.4f}"
        print(row)

    print()
    print("=== verdicts ===")
    # Growth.
    if len(results) > 1:
        prev = results[1]
        delta = merged["avg"] - prev["avg"]
        verdict = "GROWTH" if delta < 0 else "NO GROWTH"
        print(f"{verdict}: {merged['model']} AVG {merged['avg']:.4f} vs "
              f"{prev['model']} {prev['avg']:.4f} (delta {delta:+.4f})")
    else:
        print("GROWTH: not measured (no --prev)")

    # Generality.
    spread = merged["domain_spread"]
    print(f"GENERALITY: worst/best spread {spread:.3f} "
          f"(worst {merged['worst_domain_ce']:.4f}, best {merged['best_domain_ce']:.4f}); "
          f"0 means every domain is equally good")

    # Merge fidelity. Domain names are matched case-insensitively: the eval
    # prints them upper-case (MATH, CODE) while callers naturally write them
    # lower-case, and a silent "no such domain" would hide the whole check.
    merged_by_lower = {k.lower(): v for k, v in merged["domains"].items()}
    for domain, d in experts.items():
        key = domain.lower()
        if key not in merged_by_lower:
            print(f"MERGE {domain}: expert CE unavailable "
                  f"(model has {sorted(merged['domains'])})")
            continue
        merged_ce = merged_by_lower[key]
        expert_by_lower = {k.lower(): v for k, v in d.items()}
        if key not in expert_by_lower:
            print(f"MERGE {domain}: expert has no {domain} measurement")
            continue
        expert_ce = expert_by_lower[key]
        delta = merged_ce - expert_ce
        verdict = "OK" if delta <= 0 else "REGRESSION"
        print(f"MERGE {domain}: merged {merged_ce:.4f} vs expert {expert_ce:.4f} "
              f"(delta {delta:+.4f}) -> {verdict}")

    if args.out:
        payload = {
            "models": results,
            "experts": experts,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
