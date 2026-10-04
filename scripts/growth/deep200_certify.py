"""Deep-200 certification verdict (AT protocol, R114 2-eps margin).

Reads an eval_domains output log, extracts per-domain exact_ce, and
compares against a baseline with the certified-margin rule from
FORMALIZATION_TODO SSAT:

    accept an upgrade only if CE_base - CE_new > 2*eps on the domains
    that matter; a regression is CERTIFIED if CE_new - CE_base > eps.

eps = 0.096 (split-half bracket measured on the gen6_joint 200-batch
evals; noise of a single 200-batch domain mean). Usage:

    python scripts/growth/deep200_certify.py --log logs/gen7_joint_eval.log \
        [--baseline-json logs/gen6_baseline.json] [--eps 0.096]

Without --baseline-json the gen6_joint deep-200 baseline (SSAT) is used.
Exit code 0 always -- the verdict is printed, not asserted; the growth
controller reads the table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# gen6_joint deep-200 baseline (SSAT, logs/gen6_deep200_baseline)
BASELINE = {
    "EN": 3.3651,
    "HELD_CHAT": 4.4869,
    "MATH": 3.7196,
    "CODE": 1.1956,
    "HELD_MATH": 2.3739,
    "AVG": 3.1770,
    "RU": 3.9212,
}
EPS_DEFAULT = 0.096

LINE_RE = re.compile(
    r"^\s*(?P<dom>[A-Z_]+)\s+exact_ce=(?P<ce>\d+\.\d+)", re.MULTILINE
)


def parse_log(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in LINE_RE.finditer(text):
        out[m.group("dom")] = float(m.group("ce"))
    if not out:
        raise SystemExit("no 'DOMAIN exact_ce=...' lines found in the log")
    return out


def verdict(delta: float, eps: float) -> str:
    # delta = base - new: positive = CE improved.
    if delta > 2 * eps:
        return "IMPROVED (certified)"
    if delta < -eps:
        return "REGRESSION (certified)"
    return "noise"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="eval_domains output log")
    ap.add_argument("--baseline-json", default=None,
                    help="JSON {domain: ce}; defaults to the gen6_joint SSAT table")
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT)
    args = ap.parse_args()

    baseline = BASELINE
    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))

    new = parse_log(Path(args.log).read_text(encoding="utf-8", errors="replace"))

    regressions, improvements = [], []
    print(f"{'domain':<10} {'base':>8} {'new':>8} {'delta':>8}  verdict (2eps={2*args.eps:.3f}, eps={args.eps:.3f})")
    for dom, base_ce in baseline.items():
        if dom not in new:
            print(f"{dom:<10} {base_ce:>8.4f} {'--':>8} {'--':>8}  MISSING")
            continue
        d = base_ce - new[dom]
        v = verdict(d, args.eps)
        print(f"{dom:<10} {base_ce:>8.4f} {new[dom]:>8.4f} {d:>+8.3f}  {v}")
        if v.startswith("REGRESSION"):
            regressions.append(dom)
        elif v.startswith("IMPROVED"):
            improvements.append(dom)

    n_held = sum(1 for d in ("HELD_CHAT", "HELD_MATH") if d in regressions)
    upgrade = not regressions and improvements
    print()
    if upgrade:
        print(f"VERDICT: UPGRADE certified ({', '.join(improvements)} improved, no regression > eps)")
    elif regressions:
        print(f"VERDICT: REJECTED -- certified regressions: {', '.join(regressions)}"
              + (" (incl. HELD domains)" if n_held else ""))
    else:
        print("VERDICT: INCONCLUSIVE -- no domain cleared the 2eps margin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
