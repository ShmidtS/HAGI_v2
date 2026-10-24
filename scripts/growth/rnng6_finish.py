"""Автономное завершение цикла gen6: ждёт листья → merge → ft → end-merge → certified A/B → ladder.

Запускается после rnng6_chain.py; каждые 60с проверяет прогресс цепочки
(по logs/rnng6_chain.log), затем выполняет конвейер и дописывает
сертифицированный вердикт в checkpoints/rnn_ladder.jsonl.
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
DOMS = ["math", "code", "ru", "enedu", "enwiki", "books"]
BATCHES = 150
EPS = math.sqrt(math.log(2.0 / 0.05) / (2.0 * BATCHES * 32 * 1024))


def sh(cmd: list[str], log: str | None = None) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    if log:
        with open(log, "a") as fh:
            return subprocess.run(cmd, stdout=fh, stderr=fh, cwd=ROOT).returncode
    return subprocess.run(cmd, cwd=ROOT).returncode


def wait_for_leaves() -> None:
    while True:
        done = 0
        for d in DOMS:
            err = ROOT / f"logs/rnng6_leaf_{d}.err"
            if err.exists() and "training complete" in err.read_text(encoding="utf-8", errors="ignore"):
                done += 1
        print(f"leaves {done}/6 done", flush=True)
        if done == 6:
            return
        time.sleep(60)


def main() -> int:
    wait_for_leaves()

    leaves = [f"checkpoints/rnn6g_leaf_{d}/best.pt" for d in DOMS]
    rc = sh([PY, "scripts/growth/pilot_latent_merge.py", "--leaves", *leaves,
             "--shared-init", "checkpoints/rnn_gen4_root_ft/end_merged.pt",
             "--out", "checkpoints/rnn_gen6_merged"],
            log="logs/rnng6_merge.log")
    if rc != 0:
        print("MERGE FAILED", flush=True)
        return 1

    rc = sh([PY, "scripts/train.py", "--config", "configs/rnn_gen6_root_ft.yaml"],
            log="logs/rnng6_ft.log")
    if rc != 0:
        print("FT FAILED", flush=True)
        return 1

    rc = sh([PY, "scripts/growth/end_merge.py", "--config", "configs/rnn_gen6_root_ft.yaml",
             "--checkpoint-dir", "checkpoints/rnn_gen6_root_ft"],
            log="logs/rnng6_endmerge.log")
    if rc != 0:
        print("END-MERGE FAILED (non-fatal: A/B по best.pt)", flush=True)

    rc = sh([PY, "scripts/growth/rnng6_certify.py"], log="logs/rnng6_certify.log")
    if rc != 0:
        print("CERTIFY FAILED", flush=True)
        return 1

    ab = json.loads((ROOT / "logs/rnng6_ab_report.json").read_text(encoding="utf-8"))
    def avg6(r): return sum(ab["per_domain"][r].values()) / 6
    cand = ab["candidate"]
    inc = "gen4"
    entry = {
        "gen": "rnn-6",
        "model": f"checkpoints/rnn_gen6_root_ft/{'end_merged.pt' if cand.endswith('em') else 'best.pt'}",
        "avg_ce": round(avg6(cand), 4),
        "delta": round(avg6(cand) - avg6(inc), 4),
        "protocol": "common-150b, avg по всем 6 доменам",
        "date": "2026-10-24",
        "backbone": "recurrent",
        "gate": ab["verdict"],
        "levers": "СВЕЖИЕ ДАННЫЕ --start-frac 0.3 (seed_only_grow_stop) + LR 0.0005 + prior rnn-4 end_merged",
    }
    with (ROOT / "checkpoints/rnn_ladder.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("LADDER:", json.dumps(entry, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
