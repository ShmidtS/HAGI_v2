"""Is the growth loop's growth real, or is it smaller than the noise?

Every version of this project has reported a harvest rate:

    gamma = G / D,  G = incumbent CE - candidate CE

and three commits have reasoned from ``gamma = 0.0041``. That number
came from a tail-window mean over training logs. This script asks the
question that has to be answered before any rate is worth trusting:
**how big is the measurement uncertainty on G, compared with G
itself?**

The arithmetic that matters:

  - logs record every 10th step, so a "last 100 points" window is
    990 real steps and consecutive windows overlap almost entirely;
  - the per-step CE series has a standard deviation around 0.20 nats;
  - pairing the two runs step by step gives a per-point difference
    with SE = sd/sqrt(n_points).

Measured over the gen-2 -> gen-3 transition: **G = 0.0356 +- 0.0470
nats.** The standard error is LARGER than the quantity. A harvest rate
computed from that difference is not a rate; it is one sample.

This matters beyond bookkeeping. R104's gate ``alpha*C <= gamma*D``
and R107's production threshold ``beta >= (alpha/gamma)((1+alpha)-rho)``
both consume ``gamma``. Feeding them a number whose uncertainty exceeds
its value produces verdicts that are decided by which windows were
chosen -- which is exactly the failure already recorded elsewhere in
this project, where an incomplete window was read as confirmation.

The script also reports the per-transition breakdown, because the two
transitions differ by 5x: gen-1 -> gen-2 improved 0.179 nats and
gen-2 -> gen-3 improved 0.036. If growth is decaying per generation,
that is a more actionable finding than any single rate, and it is
visible only in the sequence rather than in one transition.

    python scripts/measure_growth_uncertainty.py \
        --runs logs/gen1_joint.log logs/gen2_joint.log logs/gen3_joint.log
"""

from __future__ import annotations

import argparse
import math
import re
import statistics as st
from pathlib import Path

STEP = re.compile(r"step (\d+) \| ce=([\d.]+)")


def read(path: str) -> dict[int, float]:
    """CE by step. Duplicate log lines collapse -- they are two handlers
    writing the same record, and keeping both would halve every window's
    effective step count without changing its mean."""
    out: dict[int, float] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = STEP.search(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def tail_mean(ce: dict[int, float], n: int) -> tuple[float, float]:
    keys = sorted(ce)[-n:]
    vals = [ce[k] for k in keys]
    return st.mean(vals), st.pstdev(vals)


def transition(prev: dict[int, float], nxt: dict[int, float],
                n: int) -> dict:
    """One generation's gain, with the uncertainty that makes it honest.

    The per-point difference ``prev[k] - nxt[k]`` is PAIRED: both runs
    are evaluated at the same logged steps, so the step-to-step
    variation of the data mix -- the dominant noise source -- largely
    cancels instead of being counted twice.
    """
    keys = [k for k in sorted(prev) if k in nxt][-n:]
    diffs = [prev[k] - nxt[k] for k in keys]
    mean = st.mean(diffs)
    sd = st.pstdev(diffs)
    return {
        "points": len(diffs),
        "gain": mean,
        "sd": sd,
        "stderr": sd / math.sqrt(len(diffs)) if diffs else float("nan"),
        "first_step": keys[0] if keys else None,
        "last_step": keys[-1] if keys else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="joint-phase logs, oldest generation first")
    ap.add_argument("--window", type=int, default=40,
                    help="logged points per window (logs are every 10th step)")
    args = ap.parse_args()

    series = [read(r) for r in args.runs]
    print("=== per-generation tail CE ===")
    for r, s in zip(args.runs, series):
        m, sd = tail_mean(s, args.window)
        print(f"  {r:28s} CE {m:.4f}   (sd {sd:.4f}, {len(s)} points)")

    print()
    print("=== per-transition gain, paired ===")
    gains = []
    for a, b, ra, rb in zip(series, series[1:], args.runs, args.runs[1:]):
        t = transition(a, b, args.window)
        gains.append(t["gain"])
        ratio = t["gain"] / t["stderr"] if t["stderr"] else float("inf")
        print(f"  {Path(ra).stem} -> {Path(rb).stem}")
        print(f"    steps {t['first_step']}..{t['last_step']}  "
              f"({t['points']} points)")
        print(f"    gain {t['gain']:+.4f} +- {t['stderr']:.4f} nats "
              f"({ratio:.2f} sigma)")

    if len(gains) >= 2 and gains[1] != 0:
        print()
        print(f"=== is growth DECAYING per generation? ===")
        print(f"  first transition {gains[0]:+.4f}, second {gains[1]:+.4f}")
        print(f"  ratio {gains[0] / gains[1]:.1f}x")

    last = transition(series[-2], series[-1], args.window) if len(series) >= 2 else None
    print()
    if last and last["stderr"] > abs(last["gain"]):
        print("VERDICT: the last transition's gain is INSIDE its own")
        print("         standard error. gamma is unresolved, and any")
        print("         R104/R107 verdict computed from it is decided")
        print("         by the window, not by the system.")
        return 0
    print("VERDICT: the last transition's gain exceeds its standard error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())