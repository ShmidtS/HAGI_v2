"""Certified A/B gen4 vs gen3 incumbent (protocol common-150b).

Hoeffding eps per domain = sqrt(ln(2/delta)/(2n)), delta=0.05, n = batches
* batch_size * seq_len. ACCEPT iff AVG delta < -2*eps and no domain regresses
beyond its own eps (certified_controller discipline, §9 Hoeffding).
"""
import json
import math
import subprocess
import sys

BATCHES = 150
DELTA = 0.05
TOKENS = BATCHES * 32 * 1024
EPS = math.sqrt(math.log(2.0 / DELTA) / (2.0 * TOKENS))

RUNS = {
    "gen3": ("configs/rnn_gen3_root_ft.yaml", "checkpoints/rnn_gen3_root_ft/best.pt"),
    "gen4": ("configs/rnn_gen4_root_ft.yaml", "checkpoints/rnn_gen4_root_ft/best.pt"),
    "gen4em": ("configs/rnn_gen4_root_ft.yaml", "checkpoints/rnn_gen4_root_ft/end_merged.pt"),
}

results = {}
for name, (cfg, resume) in RUNS.items():
    out = f"logs/eval_{name}_150b.json"
    r = subprocess.run(
        [sys.executable, "scripts/eval_domains.py", "--config", cfg,
         "--resume", resume, "--batches", str(BATCHES), "--json-out", out],
        capture_output=True, text=True)
    print(f"=== {name} exit={r.returncode}")
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(1)
    with open(out, encoding="utf-8") as fh:
        results[name] = {d["domain"]: d["exact_ce"] for d in json.load(fh)}

common = ["RU", "EN", "MATH", "CODE"]
avg = {n: sum(results[n][d] for d in common) / len(common) for n in results}
# кандидат gen4 = лучший из (best, end_merged) по AVG на общих доменах
cand = "gen4em" if avg["gen4em"] < avg["gen4"] else "gen4"
delta = avg[cand] - avg["gen3"]
verdict = "ACCEPT" if (delta < -2 * EPS and all(
    results[cand][d] <= results["gen3"][d] + 2 * EPS
    for d in results["gen3"])) else "REJECT"

report = {
    "protocol": "common-150b",
    "tokens_per_domain": TOKENS,
    "eps_hoeffding": EPS,
    "avg": avg,
    "candidate": cand,
    "delta": delta,
    "per_domain": results,
    "verdict": verdict,
}
print(json.dumps(report, indent=2))
with open("logs/rnng4_ab_report.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
