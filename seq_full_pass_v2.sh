#!/bin/bash
# Full sequential ternary pass v2 - honest extrapolation edition (2026-09-02).
#
# Per-layer sequential (1 layer per pass, NOT blocks of 3):
#   * telescope-honest: layer L is fit on inputs from a FULLY refit prefix
#   * tail truncation in the collector makes early layers cheap (layers > L
#     are identity-skipped) - avg pass ~ half the model
#   * cap 8192 rows/expert, MAX_LAYER_ROWS=1.1M/layer (32GB CPU RAM budget)
#   * wLS scales + VAL_FRAC=0.2: checkpoint residual = held-out number
#   * saturation early-stop cuts the random stream when 95% of fired experts
#     are capped
# Env: TOKENS (default 262144), LAYERS="3,4" to redo specific layers only,
#      TEXT_FILE=... for a real-text stream, W13_GPTQ=0|1 (canary A/B).
cd /c/HAGI_v2
PY=.venv/Scripts/python.exe
POD=checkpoints_dsv4/pod_all_tokens
LOG=seq_full_pass_v2.log
DONE=seq_full_done_v2.txt
TOKENS=${TOKENS:-262144}
LAYERS=${LAYERS:-}
TEXT=${TEXT_FILE:-}
W13G=${W13_GPTQ:-1}

if [ -f seq_full_v2.lock ]; then echo "already running"; exit 1; fi
touch seq_full_v2.lock
trap 'rm -f seq_full_v2.lock' EXIT

echo "=== v2 pass started $(date) tokens=$TOKENS W13_GPTQ=$W13G ===" >> $LOG

TEXT_ENV=""
if [ -n "$TEXT" ]; then TEXT_ENV="COLLECT_TEXT_FILE=$TEXT COLLECT_TEXT_TOKENS=131072"; fi

if [ -z "$LAYERS" ]; then
  LAYERS=$(seq -s, 1 42)
fi

for L in $(echo $LAYERS | tr ',' ' '); do
  if [ "$L" -eq 0 ]; then continue; fi   # layer 0 already refit
  prefix=""
  if [ "$L" -gt 1 ]; then prefix=$(seq -s, 0 $((L - 1))); else prefix="0"; fi
  echo "--- layer $L (prefix $prefix) $(date) ---" >> $LOG

  env $TEXT_ENV SEQ_CH=4096 SEQ_LAYERS=$L I4X_LAYERS=$prefix $PY scripts/dsv4_collect_seq.py \
      --no-vocab --max-tokens $TOKENS --out-dir checkpoints_dsv4/seq_v2 \
      >> collect_v2.log 2>&1 || { echo "COLLECT FAILED layer $L" >> $LOG; exit 2; }

  rm -f dsv4_reduced/layer_$L/expert_*.pt
  cp checkpoints_dsv4/seq_v2/acts_layer$L.pt $POD/acts_layer$L.pt
  W13_MODE=tern W13_BITS=2 W13_GS=128 W2_GPTQ=1 W13_GPTQ=$W13G PTQ_ONLY=1 VAL_FRAC=0.2 \
    $PY scripts/dsv4_refit_experts.py --start-layer $L --end-layer $((L + 1)) \
    --n-procs 1 --done-log $DONE >> refit_v2.log 2>&1 \
    || { echo "REFIT FAILED layer $L (checkpoints deleted; rerun with LAYERS=$L)" >> $LOG; exit 3; }
  echo "layer $L done $(date)" >> $LOG

  rm -f checkpoints_dsv4/seq_v2/acts_layer$L.pt $POD/acts_layer$L.pt
  echo "--- layer $L complete, acts freed $(date) ---" >> $LOG
done

echo "=== v2 PASS COMPLETE $(date) ===" >> $LOG
