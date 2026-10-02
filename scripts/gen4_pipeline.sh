#!/usr/bin/env bash
# gen4_pipeline.sh — финальные фазы gen-4: дождаться merged, запустить joint.
# Идемпотентен: если joint уже завершён — выход. Логирует в logs/gen4_pipeline.log.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gen4_pipeline.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "pipeline start"

# 1. Ждать merged step-1300 (до 3 часов)
for i in $(seq 1 360); do
  if [ -f checkpoints/dbridge_gen4_merged/step-0001300.pt ]; then
    log "merged step-1300 present"
    break
  fi
  sleep 30
done
if [ ! -f checkpoints/dbridge_gen4_merged/step-0001300.pt ]; then
  log "FATAL: merged never reached 1300"
  exit 1
fi

# 2. Joint (если ещё не завершён)
if [ ! -f checkpoints/dbridge_gen4_joint/step-0001300.pt ]; then
  log "launching joint"
  .venv/Scripts/python.exe scripts/train.py --config configs/dbridge_gen4_joint.yaml >> logs/gen4_joint_autorun.log 2>&1
  rc=$?
  log "joint exit=$rc"
  if [ $rc -ne 0 ]; then exit $rc; fi
else
  log "joint already complete"
fi

# 3. Финальная eval-сводка по доменам
log "running eval_domains on joint"
.venv/Scripts/python.exe scripts/eval_domains.py --config configs/dbridge_gen4_joint.yaml \
  --resume checkpoints/dbridge_gen4_joint/step-0001300.pt >> logs/gen4_joint_eval.log 2>&1 || \
  log "eval failed (non-fatal)"
log "pipeline done"
