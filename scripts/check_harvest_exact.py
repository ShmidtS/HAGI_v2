"""Is the gain an exact harvest of the frontier?

R104's recurrence needs the harvest to be EXACT:

    gamma * D_t <= G_t <= gamma * D_t

A one-sided lower bound is not enough. With only `G >= gamma*D` the
growth could be carried by a source the disagreement telemetry does not
see, and then `renewal_feeds_takeoff`'s frontier-scaling premise would
be measured against a gain it does not explain.

So this checks the upper half on real runs: for each pair of consecutive
generations, compare the measured capability gain against the measured
frontier. `gamma` is the ONE free constant, and the test is whether a
single gamma fits every observation -- if the ratio G/D wanders by orders
of magnitude, no constant harvest rate explains the growth.

    python scripts/check_harvest_exact.py --runs logs/a.log logs/b.log ...
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
    ap.add_argument("--frontier", type=float, required=True,
                    help="measured D_t in nats (scripts/measure_frontier.py)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="candidate harvest rate")
    args = ap.parse_args()

    print("generation gains (tail CE 100, lower is better):")
    prev = None
    gains: list[tuple[str, float]] = []
    for r in args.runs:
        ce = read(r)
        t = tail_ce(ce)
        if t is None:
            print(f"  {Path(r).name:<44} no data")
            continue
        line = f"  {Path(r).name:<44} {t:.4f}"
        if prev is not None:
            g = prev - t          # positive when capability improved
            gains.append((Path(r).name, g))
            line += f"   gain {g:+.4f}"
        print(line)
        prev = t

    if not gains:
        print("\nneed at least two runs with data")
        return 1

    print()
    print(f"measured frontier D_t = {args.frontier:.4f} nats")
    print(f"candidate harvest rate gamma = {args.gamma}")
    print()
    print(f"{'transition':<44} {'G':>10} {'gamma*D':>10} {'G <= gamma*D?':>15}")
    all_exact = True
    for name, g in gains:
        cap = args.gamma * args.frontier
        ok = g <= cap
        all_exact &= ok
        print(f"{name:<44} {g:>+10.4f} {cap:>10.4f} {str(ok):>15}")

    print()
    if all_exact:
        print("Every gain is within what this harvest rate explains: the")
        print("invariant holds, so the frontier telemetry accounts for the")
        print("growth and renewal_feeds_takeoff's premise is usable.")
    else:
        print("At least one gain EXCEEDS gamma*D. The growth is not coming")
        print("from the disagreement the frontier measures -- either another")
        print("source exists, or the frontier/gain are measured on different")
        print("things. Raise gamma if the excess is a scale mismatch, but do")
        print("not raise it until the ratio is stable across transitions:")
        ratios = [g / args.frontier for _, g in gains if args.frontier]
        if len(ratios) > 1:
            print(f"  G/D ratios: {[round(r, 4) for r in ratios]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())