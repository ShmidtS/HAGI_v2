#!/bin/bash
# Backfill pass: real drifted rows for experts the main pass marked dead
# (n_real == 0 in their terni4 checkpoints).
#
# Phase 1 (collect): ONE model pass, 96K tokens through the fully compressed
#   prefix 0..42, capture all layers, cap 64 rows/expert (~4.5 GB RAM).
# Phase 2 (refit): per layer, keep only dead experts' rows, drop their stale
#   checkpoints, run the regular terni4 refit on those experts.
#
# Run AFTER seq_full_pass.sh completes. Single instance (lock), HIP-safe chunk.
cd /c/HAGI_v2
PY=.venv/Scripts/python.exe
LOG=backfill_pass.log
DONE=seq_full_done.txt

if [ -f backfill.lock ]; then echo "already running"; exit 1; fi
touch backfill.lock
trap 'rm -f backfill.lock' EXIT

echo "=== backfill started $(date) ===" >> $LOG

LAYERS=$(seq -s, 1 42)
SEQ_CH=4096 BACKFILL_ANY_PREFIX=1 SEQ_LAYERS=$LAYERS I4X_LAYERS=$LAYERS \
  $PY scripts/dsv4_collect_seq.py --no-vocab --max-tokens 96000 --cap 64 \
  --out-dir checkpoints_dsv4/seq_backfill >> collect_backfill.log 2>&1 \
  || { echo "COLLECT FAILED" >> $LOG; exit 2; }

for L in $(seq 1 42); do
  SEQ_BACKFILL=checkpoints_dsv4/seq_backfill $PY scripts/filter_backfill_acts.py $L >> $LOG 2>&1
  if [ -f checkpoints_dsv4/pod_all_tokens/acts_layer$L.pt ]; then
    W13_MODE=tern W13_BITS=2 W13_GS=128 W2_GPTQ=1 PTQ_ONLY=1 PARTIAL_ACTS=1 \
      $PY scripts/dsv4_refit_experts.py --start-layer $L --end-layer $((L + 1)) \
      --n-procs 1 --done-log $DONE >> refit_backfill.log 2>&1 \
      || { echo "REFIT FAILED layer $L" >> $LOG; exit 3; }
    echo "layer $L backfill done $(date)" >> $LOG
  fi
done

rm -f checkpoints_dsv4/seq_backfill/acts_layer*.pt
echo "=== backfill complete $(date) ===" >> $LOG
