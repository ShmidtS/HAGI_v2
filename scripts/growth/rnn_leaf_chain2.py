import subprocess, sys, shutil
# листья от ОБЩЕГО init (R242: same-origin prior), свои data-seed из SSOT-семейства
DOMS = ["math","code","ru","enedu","enwiki"]
for d in DOMS:
    log = open(f"logs/rnn2_leaf_{d}.err","a")
    print(f"=== {d} start", flush=True)
    r = subprocess.run([sys.executable,"scripts/train.py","--config",f"configs/rnn_leaf_{d}.yaml",
                        "--init-from","checkpoints/rnn_shared_init/init.pt",
                        "--checkpoint-dir",f"checkpoints/rnn2_leaf_{d}"], stdout=log, stderr=log)
    print(f"=== {d} exit={r.returncode}", flush=True)
