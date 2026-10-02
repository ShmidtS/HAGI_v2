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
import math
import subprocess
import sys
from pathlib import Path

# Measured unigram entropy of this corpus, in nats/token. This is the
# honest "knows only token frequencies" floor: src/hagi/model/head.py
# records it as 8.06 against ln V = 10.40 for the 32768 vocabulary.
UNIGRAM_CE = 8.06

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


def universality(d: dict, floor_ce: float = UNIGRAM_CE) -> dict:
    """Turn "GENERAL" from a self-assessment into a number.

    Spread alone is gameable: a model can have a small spread while being
    uniformly bad, which is not generality, it is uniform incompetence.
    So the number is the gap between the model's WORST domain and a floor:
    the CE of a model that knows only unigram frequencies.

    The floor is the MEASURED unigram entropy of this corpus (8.06 nats),
    not ``ln V`` (10.40). That distinction matters and was got wrong the
    first time: with ``ln V`` as the floor every model this project has
    trained still collapses, because the floor has to sit ABOVE the
    model's scores for the ratio to mean anything. The unigram floor is
    the honest "knows nothing but token frequencies" level -- a model
    scoring worse than that on some domain has learned less than a lookup
    table.

        universality = (floor - worst) / (floor - best)

    Note the direction: a trained model scores BELOW the floor, so both
    numerators are ``floor - ce``, not ``ce - floor``. Written the other
    way round the denominator goes negative and the metric silently reads
    0.0 for every good model, which is a metric that cannot fail and so
    measures nothing.

    ``1.0`` means every domain is as good as the best one; ``0.0`` means
    the worst domain is no better than unigram frequencies.

    Read it as a RATIO and not as a quality score. Being a ratio, it has
    a documented counter-intuitive case: DEGRADING THE BEST DOMAIN can
    RAISE it, because ``F - best`` shrinks in the denominator faster than
    ``F - worst`` shrinks in the numerator. A model that got uniformly
    worse can therefore look more general here. That is why ``avg`` is
    reported next to it and why the two are never read separately.
    ``tests/test_universality.py`` pins this behaviour so it cannot drift.

    Args:
        d: per-domain CEs including ``AVG``.
        floor_ce: the no-knowledge floor; defaults to the measured
            unigram entropy of this corpus.

    Returns:
        ``{"universality", "worst_ce", "best_ce", "floor_ce", "n_domains"}``.
    """
    values = [v for k, v in d.items() if k != "AVG"]
    best, worst = min(values), max(values)
    floor = floor_ce
    span = floor - best
    if span <= 0.0:
        # Even the BEST domain is at or worse than unigram frequencies:
        # there is nothing to be general about, and dividing would invent
        # a number.
        u = 0.0
    else:
        u = (floor - worst) / span
    return {
        "universality": max(0.0, min(1.0, u)),
        "worst_ce": worst,
        "best_ce": best,
        "floor_ce": floor,
        "n_domains": len(values),
    }


def report(name: str, d: dict) -> dict:
    values = [v for k, v in d.items() if k != "AVG"]
    best, worst = min(values), max(values)
    avg = d.get("AVG", sum(values) / len(values))
    out = {
        "model": name,
        "domains": d,
        "avg": avg,
        "best_domain_ce": best,
        "worst_domain_ce": worst,
        # 0 = uniform across domains (fully general), 1 = all the skill
        # concentrated in one domain.
        "domain_spread": (worst - best) / avg if avg else float("nan"),
    }
    out.update(universality(d))
    return out


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

    # Transfer on the HELD-OUT domains. These corpora appear in no
    # training mix, so their CE measures transfer rather than
    # memorisation. Reported separately and never folded into AVG,
    # because a "general" claim computed over training domains only is
    # not evidence of generality.
    held = sorted(k for k in merged["domains"]
                  if k.upper().startswith("HELD"))
    if held:
        print("TRANSFER (held-out corpora, never trained on):")
        for r in results:
            cells = "  ".join(f"{k}={r['domains'][k]:.4f}" for k in held
                              if k in r["domains"])
            print(f"  {r['model'].ljust(16)}{cells}")
        print()

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
    print(f"GENERALITY: universality {merged['universality']:.4f} "
          f"(1 = every domain as good as the best one, 0 = worst domain no "
          f"better than unigram frequencies at {merged['floor_ce']:.2f}); "
          f"worst {merged['worst_domain_ce']:.4f}, best {merged['best_domain_ce']:.4f}, "
          f"spread {spread:.3f} over {merged['n_domains']} domains")

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
