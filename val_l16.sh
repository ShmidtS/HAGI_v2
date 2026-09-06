#!/bin/bash
# L16 validation refit with reboot-grade retry (same policy as seq_v2_resume.sh)
cd /c/HAGI_v2
PY=.venv/Scripts/python.exe
export AMD_SERIALIZE_KERNEL=3
LOG=val_l16.log
for TRY in 1 2 3 4 5 6 7 8; do
  THRESH=1e-3
  [ $TRY -gt 1 ] && THRESH=1.0
  rm -f dsv4_reduced/layer_16/expert_*.pt.tmp
  echo "--- refit L16 try $TRY thresh $THRESH $(date) ---" >> $LOG
  PYTORCH_HIP_ALLOC_CONFIG=expandable_segments:True \
  W13_MODE=tern W13_BITS=2 W13_GS=128 W2_GPTQ=1 W13_GPTQ=1 PTQ_ONLY=1 VAL_FRAC=0.2 \
    $PY scripts/dsv4_refit_experts.py --start-layer 16 --end-layer 17 \
    --n-procs 1 --refit-threshold $THRESH >> refit_l16_fix.log 2>&1
  RC=$?
  N=$(ls dsv4_reduced/layer_16/expert_*.pt 2>/dev/null | wc -l)
  echo "refit L16 try $TRY rc=$RC ckpts=$N/256 $(date)" >> $LOG
  [ "$N" -ge 256 ] && break
  sleep 120
done
N=$(ls dsv4_reduced/layer_16/expert_*.pt 2>/dev/null | wc -l)
if [ "$N" -ge 256 ]; then
  rm -f checkpoints_dsv4/seq/acts_layer16.pt checkpoints_dsv4/pod_all_tokens/acts_layer16.pt
  echo "--- e2e gen prefix 0..16 $(date) ---" >> $LOG
  I4X_LAYERS=$(seq -s, 0 16) $PY scripts/dsv4_generate_ttt.py \
    "In the beginning of the 21st century" 100 --no-ttt --no-save >> gen_l16_after.log 2>&1
  echo "gen rc=$? $(date)" >> $LOG
  echo "=== L16 VALIDATION DONE $(date) ===" >> $LOG
else
  echo "REFIT FAILED ($N/256)" >> $LOG
fi
