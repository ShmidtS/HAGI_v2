import subprocess, time, sys
for d in ["code","ru","enedu","enweb","enwiki"]:
    log = open(f"logs/rnn_leaf_{d}.err","a")
    cfg = f"configs/rnn_leaf_{d}.yaml"
    print(f"=== {d} start", flush=True)
    r = subprocess.run([sys.executable,"scripts/train.py","--config",cfg], stdout=log, stderr=log)
    print(f"=== {d} exit={r.returncode}", flush=True)
