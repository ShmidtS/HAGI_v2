"""G005-вердикт: gen6_joint против gen5_joint по R114/R118-дисциплине.

Считает вердикт из двух логов eval_domains (контроль gen5_joint:
AVG 2.9227; кандидат gen6_joint) по правилам FORMALIZATION_TODO §W:

1. Сертифицированное улучшение: AVG-улучшение > eps (eps=0.096
   при n=200 deep, 0.304 при n=20 стандарт, delta=0.05) И
   ни один домен не деградирует больше 0.05 — GROW.
2. Номинальное улучшение: AVG и HELD (оба) улучшаются без
   домен-деградации > 0.05 — NOMINAL (след, не сертификат).
3. Иначе UNDECIDED/REJECT.

Использование:
    python scripts/g005_verdict.py logs/gen5_joint_eval.log logs/gen6_joint_eval.log [--eps-from-n 200]
"""

from __future__ import annotations

import argparse
import math
import re
import sys

sys.path.insert(0, "src")

from hagi.train.self_development import selection_threshold  # noqa: E402

DOMAINS = (
    "RU",
    "EN",
    "MATH",
    "CODE",
    "HELD_MATH",
    "HELD_CHAT",
    "AVG",
)

_LINE_RE = re.compile(
    r"^\s*(RU|EN|MATH|CODE|HELD_MATH|HELD_CHAT|AVG)\s+exact_ce=([0-9.]+)",
    re.MULTILINE,
)


def parse_eval_log(path: str) -> dict[str, float]:
    """Парсит вывод eval_domains.py: строки 'DOMAIN: ce=X.XXXX'."""
    text = open(path, encoding="utf-8", errors="replace").read()
    found = _LINE_RE.findall(text)
    if not found:
        raise SystemExit(f"no domain metrics parsed from {path}")
    return {dom: float(val) for dom, val in found}


def hoeffding_eps(n: int, delta: float = 0.05) -> float:
    """R114: eps = sqrt(ln(2/delta)/(2n))."""
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def verdict(
    control: dict[str, float],
    candidate: dict[str, float],
    eps: float,
    dom_tol: float = 0.05,
) -> tuple[str, dict[str, float]]:
    """Вердикт G005: improvements положительны = CE ниже."""
    impr = {
        dom: control[dom] - candidate[dom]
        for dom in DOMAINS
        if dom in control and dom in candidate
    }
    worst_dom = min(
        (d for d in impr if d != "AVG"),
        key=lambda d: impr[d],
    )
    no_dom_regression = impr[worst_dom] > -dom_tol
    certified = impr.get("AVG", 0.0) > eps and no_dom_regression
    nominal = (
        impr.get("AVG", 0.0) > 0.0
        and impr.get("HELD_CHAT", 0.0) > 0.0
        and impr.get("HELD_MATH", 0.0) > 0.0
        and no_dom_regression
    )
    if certified:
        v = "GROW_CERTIFIED"
    elif nominal:
        v = "NOMINAL"
    else:
        v = "UNDECIDED"
    return v, impr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("control_log")
    ap.add_argument("candidate_log")
    ap.add_argument(
        "--n",
        type=int,
        default=200,
        help="батчей на домен в eval (200=deep, 20=standard)",
    )
    args = ap.parse_args()

    control = parse_eval_log(args.control_log)
    candidate = parse_eval_log(args.candidate_log)
    eps = hoeffding_eps(args.n)
    v, impr = verdict(control, candidate, eps)

    print(f"eps (n={args.n}, delta=0.05): {eps:.4f}")
    print(f"3*eps R118 bar: {selection_threshold(eps):.4f}")
    for dom in DOMAINS:
        if dom in impr:
            print(f"{dom:10s} {control[dom]:.4f} -> {candidate[dom]:.4f} "
                  f"(impr {impr[dom]:+.4f})")
    print(f"VERDICT: {v}")


if __name__ == "__main__":
    main()
