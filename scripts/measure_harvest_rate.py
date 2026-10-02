"""How much of the frontier does the merge actually harvest?

The measurements so far localise the bottleneck precisely:

  - the frontier is LARGE: D_t = 18.25 nats between three domain
    experts (scripts/measure_frontier.py);
  - the harvest is EXACT: the realised gen-2 -> gen-3 gain of +0.0365
    is far inside gamma*D (scripts/check_harvest_exact.py);
  - so the realised gain is three orders of magnitude BELOW what the
    disagreement offers.

That points at the merge operator: it is not converting the available
disagreement into capability. This script measures the conversion rate
directly, on checkpoints that exist, and reports the implied gamma so
the harvest rate becomes a number rather than an assumption.

  realised gamma = G / D_t

    python scripts/measure_harvest_rate.py \
        --runs logs/gen2_joint.log logs/gen3_joint.log --frontier 18.2495
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

STEP = re.compile(r"step (\d+) \| ce=([\d.]+)")


def read(path: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = STEP.search(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def tail_ce(ce: dict[int, float], n: int = 100) -> float | None:
    ks = sorted(ce)
    if len(ks) < n:
        n = len(ks)
    return st.mean([ce[k] for k in ks[-n:]]) if n else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="training logs, oldest generation first")
    ap.add_argument("--frontier", type=float, required=True)
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    vals = []
    for r in args.runs:
        ce = read(r)
        t = tail_ce(ce, args.window)
        if t is not None:
            vals.append((Path(r).stem, t))

    print(f"{'generation':<24} {'tail CE':>10}")
    for name, t in vals:
        print(f"{name:<24} {t:>10.4f}")

    gains = []
    for (n0, v0), (n1, v1) in zip(vals, vals[1:]):
        gains.append((f"{n0} -> {n1}", v0 - v1))

    if not gains:
        print("\nneed at least two generations with data")
        return 1

    print(f"\nfrontier D_t = {args.frontier:.4f} nats")
    print(f"{'transition':<44} {'gain G':>10} {'gamma=G/D':>12}")
    rates = []
    for name, g in gains:
        gamma = g / args.frontier
        rates.append(gamma)
        print(f"{name:<44} {g:>+10.4f} {gamma:>12.6f}")

    print()
    if len(rates) > 1:
        spread = max(rates) / min(r for r in rates if r > 0)
        print(f"gamma across transitions spans {spread:.1f}x")
        if spread > 3.0:
            print("A constant harvest rate does NOT fit: the merge converts a")
            print("varying share of the disagreement, so renewal's constant-gamma")
            print("recurrence is not directly applicable and gamma must be")
            print("measured per transition rather than assumed.")
        else:
            print("A constant harvest rate is a reasonable working assumption here.")
    else:
        print("One transition only -- not enough to say whether gamma is stable.")
        print("The rate above is the honest single data point, not a constant.")

    best = max(rates)
    print()
    print(f"At the best observed rate ({best:.6f}), the frontier would sustain")
    print(f"the takeoff gate up to capability "
          f"{best * args.frontier / 0.1:.1f} at alpha=0.1 -- so the ceiling")
    print("is set by how little the merge harvests, not by the frontier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())