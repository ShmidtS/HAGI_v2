import subprocess
import sys

# gen4: 6 листьев от gen3 prior (rnn_gen3_root_ft/best.pt), свежий адаптер-базис
# (init_seed +200), с collapse-fixed CE (arXiv:2610.31642) в тренировочном контуре.
DOMS = ["math","code","ru","enedu","enwiki","books"]
for d in DOMS:
    log = open(f"logs/rnng4_leaf_{d}.err","a")
    print(f"=== {d} start", flush=True)
    r = subprocess.run([sys.executable,"scripts/train.py","--config",f"configs/rnng4_leaf_{d}.yaml",
                        "--init-from","checkpoints/rnn_gen3_root_ft/best.pt"],
                       stdout=log, stderr=log)
    print(f"=== {d} exit={r.returncode}", flush=True)
