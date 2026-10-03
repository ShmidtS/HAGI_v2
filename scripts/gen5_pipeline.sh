#!/usr/bin/env bash
# gen5_pipeline.sh — последовательный конвейер gen-5: sib math -> sib lang ->
# sib code -> merged (GAM phase 5) -> joint. Логирует в logs/gen5_pipeline.log.
# Идемпотентен: каждый шаг пропускается, если его чекпойнт уже существует.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gen5_pipeline.log
PY=.venv/Scripts/python.exe
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "pipeline start (pid $$)"

run_stage() { # name config ckpt log
  local name="$1" config="$2" ckpt="$3" out="$4"
  if [ -f "$ckpt" ]; then
    log "$name: already complete ($ckpt)"
    return 0
  fi
  log "$name: launching ($config)"
  "$PY" scripts/train.py --config "$config" >> "$out" 2>&1
  local rc=$?
  log "$name: exit=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$ckpt" ]; then
    log "$name: FATAL (rc=$rc, ckpt=$([ -f "$ckpt" ] && echo present || echo missing))"
    exit 1
  fi
  return 0
}

# 1-3. Доменные сибы (GPU serial)
run_stage "sib_math" configs/dbridge_gen5_sib_math.yaml \
  checkpoints/dbridge_gen5_sib_math/step-0001300.pt logs/gen5_sib_math.log
run_stage "sib_lang" configs/dbridge_gen5_sib_lang.yaml \
  checkpoints/dbridge_gen5_sib_lang/step-0001300.pt logs/gen5_sib_lang.log
run_stage "sib_code" configs/dbridge_gen5_sib_code.yaml \
  checkpoints/dbridge_gen5_sib_code/step-0001300.pt logs/gen5_sib_code.log

# 4. Merged с GAM phase 5
run_stage "merged" configs/dbridge_gen5_merged.yaml \
  checkpoints/dbridge_gen5_merged/step-0001300.pt logs/gen5_merged.log

# 5. Joint
run_stage "joint" configs/dbridge_gen5_joint.yaml \
  checkpoints/dbridge_gen5_joint/step-0001300.pt logs/gen5_joint.log

# 6. Финальная eval-сводка по доменам на gen-5 joint
log "running eval_domains on gen5 joint"
"$PY" scripts/eval_domains.py --config configs/dbridge_gen5_joint.yaml \
  --resume checkpoints/dbridge_gen5_joint/step-0001300.pt >> logs/gen5_joint_eval.log 2>&1 || \
  log "eval failed (non-fatal)"
log "pipeline done"
