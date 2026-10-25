import subprocess, sys
# gen5: 6 листьев от gen4 prior, LR-полка -30% (0.0005 -> 0.00035):
# правило IDEAL_CYCLE §1 — дивергенция gen4 на 4/6 листьев = задним числом
# измеренная L выросла (fixed CE растит маржу до m*), кап eta=1/L опускается.
DOMS = ["math","code","ru","enedu","enwiki","books"]
for d in DOMS:
    log = open(f"logs/rnng5_leaf_{d}.err","a")
    print(f"=== {d} start", flush=True)
    r = subprocess.run([sys.executable,"scripts/train.py","--config",f"configs/rnng5_leaf_{d}.yaml",
                        "--init-from","checkpoints/rnn_gen4_root_ft/end_merged.pt"],
                       stdout=log, stderr=log)
    print(f"=== {d} exit={r.returncode}", flush=True)
