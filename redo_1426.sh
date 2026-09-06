#!/bin/bash
# Redo layers 14..26 with the SVD dead-expert fix (16 already done).
# Telescopic order matters: lower layers first, each fitted through the
# already-fixed prefix. Reuses the seq_v2_resume.sh retry policy.
cd /c/HAGI_v2
PY=.venv/Scripts/python.exe
POD=checkpoints_dsv4/pod_all_tokens
LOG=redo_1426.log
export AMD_SERIALIZE_KERNEL=3

if [ -f redo_1426.lock ]; then echo "already running"; exit 1; fi
touch redo_1426.lock
trap 'rm -f redo_1426.lock' EXIT

echo "=== redo 14..26 (dead-fix) started $(date) ===" >> $LOG

for L in $(seq 14 26); do
  [ "$L" -eq 16 ] && continue   # already validated & refit with the fix

  # collect through the (progressively fixed) compressed prefix
  for TRY in 1 2 3; do
    SEQ_LAYERS=$L I4X_LAYERS=$(seq -s, 0 $((L - 1))) TOKENS=262144 \
      $PY scripts/dsv4_collect_seq.py >> collect_redo_L$L.log 2>&1
    [ -f checkpoints_dsv4/seq/acts_layer$L.pt ] && break
    echo "collect layer $L try $TRY failed, cooling 120s" >> $LOG
    sleep 120
  done
  [ -f checkpoints_dsv4/seq/acts_layer$L.pt ] || { echo "COLLECT FAILED layer $L" >> $LOG; exit 2; }
  cp checkpoints_dsv4/seq/acts_layer$L.pt $POD/acts_layer$L.pt

  # purge ALL checkpoints of layer L (they are v2-era but built with the old
  # pool distill for dead experts; refit threshold cannot tell them apart)
  $PY - <<PYEOF
import glob, os
n = 0
for fp in glob.glob("dsv4_reduced/layer_$L/expert_*.pt"):
    os.remove(fp); n += 1
print("purged", n)
PYEOF

  for TRY in 1 2 3 4 5 6 7 8; do
    THRESH=1e-3
    [ $TRY -gt 1 ] && THRESH=1.0
    rm -f dsv4_reduced/layer_$L/expert_*.pt.tmp
    echo "--- refit layer $L try $TRY thresh $THRESH $(date) ---" >> $LOG
    PYTORCH_HIP_ALLOC_CONFIG=expandable_segments:True \
    W13_MODE=tern W13_BITS=2 W13_GS=128 W2_GPTQ=1 W13_GPTQ=1 PTQ_ONLY=1 VAL_FRAC=0.2 \
      $PY scripts/dsv4_refit_experts.py --start-layer $L --end-layer $((L + 1)) \
      --n-procs 1 --refit-threshold $THRESH >> refit_redo_L$L.log 2>&1
    RC=$?
    N=$(ls dsv4_reduced/layer_$L/expert_*.pt 2>/dev/null | wc -l)
    echo "refit layer $L try $TRY rc=$RC ckpts=$N/256 $(date)" >> $LOG
    [ "$N" -ge 256 ] && break
    sleep 120
  done
  N=$(ls dsv4_reduced/layer_$L/expert_*.pt 2>/dev/null | wc -l)
  [ "$N" -ge 256 ] || { echo "REFIT FAILED layer $L ($N/256)" >> $LOG; exit 3; }

  rm -f checkpoints_dsv4/seq/acts_layer$L.pt $POD/acts_layer$L.pt
  echo "--- layer $L complete, acts freed $(date) ---" >> $LOG
done

# quick e2e sanity after the full redo: prefix 0..20 must be COHERENT now
I4X_LAYERS=$(seq -s, 0 20) $PY scripts/dsv4_generate_ttt.py \
  "In the beginning of the 21st century" 100 --no-ttt --no-save >> gen_redo_0_20.log 2>&1
echo "=== redo 14..26 DONE $(date) ===" >> $LOG
