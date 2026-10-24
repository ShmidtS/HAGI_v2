"""Единый certified A/B (Hoeffding, протокол common-150b) — параметризованный.

Заменяет все одноразовые certify-скрипты поколений. Сравнивает кандидатов
(обычно best.pt + end_merged.pt) с инкумбентом; ACCEPT iff AVG-дельта за
порогом 2ε И ни один домен не регрессирует сверх 2ε (no-regression clause).

    python scripts/growth/certify.py --tag g8 \
        --incumbent configs/rnn_gen6_root_ft.yaml checkpoints/rnn_gen6_root_ft/end_merged.pt \
        --candidate configs/g8_root_ft.yaml checkpoints/g8_root_ft/best.pt \
                     configs/g8_root_ft.yaml checkpoints/g8_root_ft/end_merged.pt
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
BATCHES = 150
DELTA = 0.05
TOKENS = BATCHES * 32 * 1024
EPS = math.sqrt(math.log(2.0 / DELTA) / (2.0 * TOKENS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="префикс артефактов (логи/отчёт)")
    ap.add_argument("--incumbent", nargs=2, required=True, metavar=("CONFIG", "CKPT"))
    ap.add_argument("--candidate", nargs=2, action="append", required=True,
                    metavar=("CONFIG", "CKPT"), help="повторяемый: несколько кандидатов")
    ap.add_argument("--batches", type=int, default=BATCHES)
    args = ap.parse_args()
    eps = math.sqrt(math.log(2.0 / DELTA) / (2.0 * args.batches * 32 * 1024))

    runs: dict[str, tuple[str, str]] = {"incumbent": tuple(args.incumbent)}
    for i, (cfg, ckpt) in enumerate(args.candidate):
        runs[f"cand{i}"] = (cfg, ckpt)

    results: dict[str, dict[str, float]] = {}
    names: dict[str, str] = {}
    for name, (cfg, ckpt) in runs.items():
        out = ROOT / f"logs/{args.tag}_eval_{name}.json"
        r = subprocess.run(
            [PY, "scripts/eval_domains.py", "--config", cfg, "--resume", ckpt,
             "--batches", str(args.batches), "--json-out", str(out)],
            capture_output=True, text=True, cwd=ROOT,
        )
        print(f"=== {name} ({ckpt}) exit={r.returncode}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return 1
        results[name] = {d["domain"]: d["exact_ce"] for d in json.loads(out.read_text(encoding="utf-8"))}
        names[name] = ckpt

    def avg6(r: str) -> float:
        return sum(results[r].values()) / len(results[r])

    cands = [k for k in results if k != "incumbent"]
    best = min(cands, key=avg6)
    delta = avg6(best) - avg6("incumbent")
    no_regression = all(
        results[best][d] <= results["incumbent"][d] + 2 * eps for d in results["incumbent"]
    )
    verdict = "ACCEPT" if (delta < -2 * eps and no_regression) else "REJECT"
    report = {
        "protocol": f"common-{args.batches}b, avg по всем доменам",
        "eps": eps,
        "avg": {k: round(avg6(k), 4) for k in results},
        "candidate": best,
        "candidate_ckpt": names[best],
        "delta": round(delta, 4),
        "no_regression": no_regression,
        "per_domain": results,
        "verdict": verdict,
    }
    print(json.dumps(report, indent=2))
    (ROOT / f"logs/{args.tag}_ab_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
