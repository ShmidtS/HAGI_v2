import subprocess, sys
# gen6: СВЕЖИЕ ДАННЫЕ (--start-frac 0.3, seed_only_grow_stop) — каждый корпус
# продвигается на 30% своего размера; LR-полка возвращена к 0.0005 (гипотеза
# gen5 «ниже LR» отвергается её же A/B; инъекция свежести — рычаг по теореме).
# Инкумбент-приор подставляется при запуске (см. rnng6_run.sh).
DOMS = ["math","code","ru","enedu","enwiki","books"]
import os
prior = os.environ.get("RNN6_PRIOR", "checkpoints/rnn_gen5_root_ft/end_merged.pt")
for d in DOMS:
    log = open(f"logs/rnng6_leaf_{d}.err","a")
    print(f"=== {d} start", flush=True)
    r = subprocess.run([sys.executable,"scripts/train.py","--config",f"configs/rnng6_leaf_{d}.yaml",
                        "--init-from", prior, "--start-frac", "0.3"],
                       stdout=log, stderr=log)
    print(f"=== {d} exit={r.returncode}", flush=True)
