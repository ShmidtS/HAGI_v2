#!/bin/bash
# Full sequential ternary pass, block-3: 43 layers.
# Block b: collect layers L..L+2 through compressed prefix <L, refit them, delete acts.
# Layer 0 already refit (terni4, no prefix needed).
# Disk hygiene: block acts (~2.4GB) deleted after each refit.
cd /c/HAGI_v2
PY=.venv/Scripts/python.exe
POD=checkpoints_dsv4/pod_all_tokens
LOG=seq_full_pass.log
DONE=seq_full_done.txt

# guard: single instance
if [ -f seq_full.lock ]; then echo "already running"; exit 1; fi
touch seq_full.lock
trap 'rm -f seq_full.lock' EXIT

echo "=== full sequential tern pass started $(date) ===" >> $LOG

b=1
while [ $b -le 42 ]; do
  hi=$((b + 2)); [ $hi -gt 42 ] && hi=42
  layers=$(seq -s, $b $hi)
  prefix=$(seq -s, 0 $((b - 1)))
  echo "--- block layers=$layers prefix=0..$((b-1)) $(date) ---" >> $LOG

  # 1. collect block acts through compressed prefix (single model pass)
  # block 1: prefix = {0} (layer 0 is already terni4-compressed!)
  pfx=$prefix
  SEQ_CH=4096 SEQ_LAYERS=$layers I4X_LAYERS=$pfx $PY scripts/dsv4_collect_seq.py \
      --no-vocab --max-tokens 8192 --out-dir checkpoints_dsv4/seq_full \
      >> collect_full.log 2>&1 || { echo "COLLECT FAILED block $b" >> $LOG; exit 2; }

  # 2. refit each layer of the block (overwrite old checkpoints; fresh done-log)
  for L in $(seq $b $hi); do
    rm -f dsv4_reduced/layer_$L/expert_*.pt
    cp checkpoints_dsv4/seq_full/acts_layer$L.pt $POD/acts_layer$L.pt
    W13_MODE=tern W13_BITS=2 W13_GS=128 W2_GPTQ=1 PTQ_ONLY=1 \
      $PY scripts/dsv4_refit_experts.py --start-layer $L --end-layer $((L + 1)) \
      --n-procs 1 --done-log $DONE >> refit_full.log 2>&1 \
      || { echo "REFIT FAILED layer $L: checkpoints deleted, restart manually: re-run with a wrapper that skips completed blocks (grep 'layer $L done' $LOG)" >> $LOG; exit 3; }
    echo "layer $L done $(date)" >> $LOG
  done

  # 3. disk hygiene: drop this block's acts (checkpoints are the artifact)
  rm -f checkpoints_dsv4/seq_full/acts_layer*.pt
  for L in $(seq $b $hi); do rm -f $POD/acts_layer$L.pt; done
  echo "--- block $b..$hi complete $(date), acts freed ---" >> $LOG
  b=$((hi + 1))
done

# also free the original full acts for layers 1..42 (not needed anymore; keep layer 0 as reference)
echo "=== PASS COMPLETE $(date) ===" >> $LOG
