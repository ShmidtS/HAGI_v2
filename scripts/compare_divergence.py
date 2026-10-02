"""Divergence of the falling runs against their stable twin, step by step.

Everything structural is ruled out: same H, same LR schedule (identical to
step 1040), same mix, same data proportions, same stream offsets, same
worker coverage. Only `data.seed` differs and only the 129xx family
falls.

So compare the runs against each other directly, not against a
hypothesis. This finds the FIRST step at which the falling run separates
from the stable one, which is where the cause has to act -- and every
step before it is where it does not.

    python scripts/compare_divergence.py \
        --falling logs/growth_gen4_domain_dbridge_gen4_sib1.log \
        --stable logs/swap_sib2.log
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--falling", required=True)
    ap.add_argument("--stable", required=True)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="nats of trailing-mean separation that counts as "
                         "the onset")
    args = ap.parse_args()

    f = read(args.falling)
    s = read(args.stable)
    common = sorted(set(f) & set(s))
    if not common:
        print("no common steps")
        return 1

    print(f"falling: {args.falling}  ({len(f)} steps)")
    print(f"stable : {args.stable}  ({len(s)} steps)")
    print(f"common : {len(common)} steps, 0..{common[-1]}")
    print()

    w = args.window
    onset = None
    print(f"{'step':>6} {'falling':>9} {'stable':>9} {'delta':>9}")
    prev = None
    for i in range(0, len(common) - w, max(1, w // 2)):
        hi = common[i + w]
        fv = [f[k] for k in common if hi - w <= k <= hi]
        sv = [s[k] for k in common if hi - w <= k <= hi]
        if not fv or not sv:
            continue
        fm, sm = st.mean(fv), st.mean(sv)
        d = fm - sm
        marker = ""
        if onset is None and d > args.threshold:
            onset = hi
            marker = "  <-- onset"
        if hi % (w * 4) == 0 or marker:
            print(f"{hi:>6} {fm:>9.4f} {sm:>9.4f} {d:>+9.4f}{marker}")
        prev = d

    print()
    if onset is None:
        print(f"no separation above {args.threshold} nats within the "
              f"{w}-step window")
        print("the two runs stayed together for the whole common horizon")
    else:
        print(f"onset at step {onset}: the falling run first trails the "
              f"stable one by >{args.threshold} nats")
        print(f"(last window separation {prev:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())