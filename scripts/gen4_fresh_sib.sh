#!/usr/bin/env bash
# gen4_fresh_sib.sh — G004 fresh-data injection trial.
# Waits for the gen-5 pipeline to release the GPU, then trains the
# slimpajama-fresh sibling and evals it on all domains.
# A/B control: dbridge_gen4_sib_math (same parent, same seed 13801,
# only the data mix differs: slimpajama 0.10 in, edu 0.25->0.20,
# python_instruct 0.1->0.05).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gen4_fresh_sib.log
PY=.venv/Scripts/python.exe
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

JOINT_CKPT=checkpoints/dbridge_gen5_joint/step-0001300.pt

log "waiting for gen5 pipeline to release the GPU"
# Poll until the joint checkpoint exists AND no train.py holds the GPU.
while true; do
  if [ -f "$JOINT_CKPT" ]; then
    BUSY=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -like '*scripts/train.py*'}).Count" 2>/dev/null | tr -d '\r')
    if [ "${BUSY:-1}" = "0" ]; then
      break
    fi
  fi
  sleep 60
done
log "GPU free, launching fresh sib"

"$PY" scripts/train.py --config configs/dbridge_gen4_fresh_sib1.yaml >> "$LOG" 2>&1
rc=$?
log "fresh_sib1 exit=$rc"
if [ $rc -ne 0 ] || [ ! -f checkpoints/dbridge_gen4_fresh_sib1/step-0001300.pt ]; then
  log "FATAL fresh_sib1 (rc=$rc)"
  exit 1
fi

log "eval_domains on fresh sib"
"$PY" scripts/eval_domains.py --config configs/dbridge_gen4_fresh_sib1.yaml \
  --resume checkpoints/dbridge_gen4_fresh_sib1/step-0001300.pt >> "$LOG" 2>&1
log "fresh sib done"
