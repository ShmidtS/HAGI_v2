"""Score the three F3 arms on the same held-out and apply the pre-registered rule.

Why this exists: `.omc/attempts/f3_tree_vs_flat_2026-09-26.md` pre-registers
the claim BEFORE any arm finished, so the decision rule has to be applied by
something that cannot move the goalposts afterwards. This reads the three
reports, prints the per-domain deltas, and applies the margin rule that
`growth_supervisor.py` uses (DEFAULT_TOTAL_MARGIN / DEFAULT_DOMAIN_MARGIN) so
the flat and tree arms are judged by the same standard as the N=6/N=7 lanes.

It deliberately does NOT decide "the tree won". It prints the numbers and the
verdict, and writes the verdict to the ledger. A missing report is a missing
arm, never a zero.

Usage:
    python scripts/f3_verdict.py
    python scripts/f3_verdict.py --flat reports/f3_flat_d1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports/growth_ledger.jsonl"

ARMS = {
    "tree_cortex": ROOT / "reports/f3_tree_cortex.json",
    "tree_nocortex": ROOT / "reports/f3_tree_nocortex.json",
    "flat_control": ROOT / "reports/f3_flat_d1.json",
}

# Same margins the supervisor used for the flat lanes (growth_supervisor.py).
DEFAULT_TOTAL_MARGIN = 0.05
DEFAULT_DOMAIN_MARGIN = 0.02


def mean_ce(report: dict) -> float | None:
    """Mean exact CE over the domains that actually scored.

    Mirrors growth_supervisor.mean_ce: a missing or errored domain is excluded,
    and a lane with no scored domain has no number rather than a zero.
    """
    values = [
        float(d["exact_ce"])
        for d in report.get("domains", {}).values()
        if isinstance(d, dict) and d.get("exact_ce") is not None
    ]
    return sum(values) / len(values) if values else None


def per_domain(report: dict) -> dict[str, float]:
    return {
        name: float(d["exact_ce"])
        for name, d in report.get("domains", {}).items()
        if isinstance(d, dict) and d.get("exact_ce") is not None
    }


def compare(candidate: dict, incumbent: dict) -> tuple[float | None, dict[str, float]]:
    a, b = mean_ce(candidate), mean_ce(incumbent)
    if a is None or b is None:
        return None, {}
    return a - b, {k: per_domain(candidate)[k] - per_domain(incumbent)[k] for k in per_domain(candidate) if k in per_domain(incumbent)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-margin", type=float, default=DEFAULT_TOTAL_MARGIN)
    ap.add_argument("--domain-margin", type=float, default=DEFAULT_DOMAIN_MARGIN)
    for name, default in ARMS.items():
        ap.add_argument(f"--{name.replace('_', '-')}", default=str(default))
    args = ap.parse_args()

    reports: dict[str, dict] = {}
    for name in ARMS:
        path = Path(getattr(args, name))
        if not path.is_file():
            print(f"[verdict] MISSING arm {name}: {path}")
            continue
        reports[name] = json.loads(path.read_text(encoding="utf-8"))

    print("[verdict] arms scored: " + (", ".join(reports) or "none"))
    for name, report in reports.items():
        print(f"[verdict] {name:15s} mean_ce={mean_ce(report)} domains={per_domain(report)}")

    if "flat_control" not in reports:
        print("[verdict] no flat control: the claim is not testable yet")
        return 1

    verdicts: dict[str, dict] = {}
    for name, report in reports.items():
        if name == "flat_control":
            continue
        delta, deltas = compare(report, reports["flat_control"])
        if delta is None:
            verdicts[name] = {"accepted": False, "reason": "incomparable report"}
            continue
        won_every_domain = all(
            d < -args.domain_margin for d in deltas.values()
        ) if deltas else False
        accepted = delta < -args.total_margin
        verdicts[name] = {
            "accepted": bool(accepted and won_every_domain),
            "mean_ce_delta_vs_flat": delta,
            "per_domain_delta": deltas,
            "beats_flat_on_total": bool(accepted),
            "beats_flat_on_every_domain": bool(won_every_domain),
            "reason": (
                "tree beats flat on total and on every scored domain"
                if accepted and won_every_domain
                else f"delta={delta:+.4f} (margin -{args.total_margin}); "
                     f"every-domain={won_every_domain}"
            ),
        }
        print(f"[verdict] {name} vs flat: {verdicts[name]['reason']}")

    if not verdicts:
        print("[verdict] no tree arm scored")
        return 1

    # The claim is "tree beats flat", so any tree arm winning is a positive.
    claimed = any(v["accepted"] for v in verdicts.values())
    verdict = {
        "lane": "f3_tree_vs_flat",
        "phase": "evaluated",
        "incumbent": "flat_control",
        "report": "reports/f3_verdict.json",
        "accepted": bool(claimed),
        "reason": (
            "pre-registered claim holds: F3 tree with root cortex beats the "
            "flat merge at equal parameters and equal joint budget"
            if claimed
            else "pre-registered claim NOT supported at the registered margin"
        ),
        "arms": verdicts,
        "scored_domains": sorted(
            {d for r in reports.values() for d in per_domain(r)}
        ),
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(verdict, ensure_ascii=False) + "\n")
    out = ROOT / "reports/f3_verdict.json"
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] accepted={claimed} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
