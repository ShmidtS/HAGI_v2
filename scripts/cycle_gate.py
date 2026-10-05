"""Per-cycle recursion gate (RecursiveDistill.lean runtime decision).

The cycle's honest accounting, fed by two measurements the pipeline
already produces:

``delta_k``
    The bridge slack: gate CE of the distillate MINUS gate CE of the
    teacher (the generation's joint), both from ``gate_score.py`` on
    the SAME canonical windows -- the only comparable number.

``gain c_k``
    The certified growth: deep-200 of the distillate vs the PREVIOUS
    generation's distillate (``deep200_certify.py`` verdict, eps from
    the baseline it carries).

This script owns the LEDGER: a JSON file mapping cycle -> {delta,
gain}. Each call appends the measured pair and prints the
``cycle_report`` verdict the pipeline obeys mechanically:

    CONTINUE    this cycle's net is positive and the field is rich
    STOP_LEAK   delta >= gain: distillation leaks more than growth
                certified -- the chain may degrade, stop the loop
    EXHAUSTED   remaining field < eps/gamma: no future cycle can
                certify eps -- switch CORPUS, not more cycles

gamma/eps/d0/rho default to the plan's values (gamma=1: no field-to-
gain amplification assumed; eps=0.05: the deep-200 certify floor;
rho=0.5: the geometric decay the disagreement measurements showed).

Usage:
    python scripts/cycle_gate.py --ledger logs/cycle_ledger.json \
        --cycle 7 --delta 0.087 --gain 0.121
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.train.recursive_distill import cycle_report  # noqa: E402


def load_ledger(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"cycles": {}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path("logs/cycle_ledger.json"))
    ap.add_argument("--cycle", type=int, required=True,
                    help="generation number whose distillate was measured")
    ap.add_argument("--delta", type=float, required=True,
                    help="gate CE student - gate CE teacher (nats)")
    ap.add_argument("--gain", type=float, required=True,
                    help="deep-200 gain vs previous distillate (nats)")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--d0", type=float, default=1.0,
                    help="cycle-1 disagreement field (nats)")
    ap.add_argument("--rho", type=float, default=0.5)
    args = ap.parse_args()

    if args.delta < 0 or args.gain < 0:
        raise SystemExit("delta and gain must be non-negative")
    if args.cycle < 1:
        raise SystemExit("cycle numbering starts at 1")

    ledger = load_ledger(args.ledger)
    cycles = ledger["cycles"]
    key = str(args.cycle)
    if key in cycles:
        print(f"cycle {args.cycle} already in ledger: {cycles[key]} "
              f"(overwrite with --force is deliberate policy; refusing)")
        return 1
    cycles[key] = {"delta": args.delta, "gain": args.gain}
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    ordered = sorted(cycles.items(), key=lambda kv: int(kv[0]))
    deltas = [v["delta"] for _, v in ordered]
    gains = [v["gain"] for _, v in ordered]
    r = cycle_report(deltas, gains, gamma=args.gamma, eps=args.eps,
                     d0=args.d0, rho=args.rho)
    print(f"cycle {args.cycle}: delta={args.delta:.4f} gain={args.gain:.4f}")
    print(f"ledger: {len(deltas)} cycles, slack_total={r['slack_total']:.4f}, "
          f"net_total={r['net_total']:.4f}, g_min={r['g_min']:.4f}")
    print(f"field={r['field']:.4f} threshold={r['threshold']:.4f} "
          f"harvest_budget={r['harvest_budget']:.4f}")
    print(f"VERDICT: {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
