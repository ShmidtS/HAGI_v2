import subprocess, sys
DOMS = ["math","code","ru","enedu","enwiki","books"]
for d in DOMS:
    log = open(f"logs/rnng3_leaf_{d}.err","a")
    print(f"=== {d} start", flush=True)
    r = subprocess.run([sys.executable,"scripts/train.py","--config",f"configs/rnng3_leaf_{d}.yaml",
                        "--init-from","checkpoints/rnn_gen2_root_ft/best.pt"],
                       stdout=log, stderr=log)
    print(f"=== {d} exit={r.returncode}", flush=True)
