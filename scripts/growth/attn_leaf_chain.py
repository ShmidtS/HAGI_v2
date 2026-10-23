import subprocess, sys
DOMS = ["math","code","ru","enedu","enwiki"]
for d in DOMS:
    log = open(f"logs/attn1_leaf_{d}.err","a")
    print(f"=== {d} start", flush=True)
    r = subprocess.run([sys.executable,"scripts/train.py","--config",f"configs/attn1_leaf_{d}.yaml",
                        "--init-from","checkpoints/attn_shared_init/init.pt",
                        "--checkpoint-dir",f"checkpoints/attn1_leaf_{d}"],
                       stdout=log, stderr=log)
    print(f"=== {d} exit={r.returncode}", flush=True)
