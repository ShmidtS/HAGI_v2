#!/usr/bin/env bash
# gen5_pipeline.sh — GAM A/B: re-merge the SAME gen-4 sibs with GAM phase 5
# and train it under the gen-4 merged recipe, then joint, then eval.
# The gen-4 line (fixed-Hadamard mixer) is the control arm.
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

# 1. GAM merged (re-merge of gen-4 sibs, phase 5, init_scale 0.1)
run_stage "merged_gam" configs/dbridge_gen5_merged.yaml \
  checkpoints/dbridge_gen5_merged/step-0001300.pt logs/gen5_merged.log

# 2. Joint
run_stage "joint" configs/dbridge_gen5_joint.yaml \
  checkpoints/dbridge_gen5_joint/step-0001300.pt logs/gen5_joint.log

# 3. channel split on the GAM merged step-0 (does the learned channel carry?)
log "channel_split on gen5 merged step-0"
"$PY" scripts/measure_channel_split.py --ckpt checkpoints/dbridge_gen5_merged/step-0000000.pt \
  --config configs/dbridge_gen5_merged.yaml >> logs/gen5_channel_split.log 2>&1 || \
  log "channel split failed (non-fatal)"

# 4. Финальная eval-сводка по доменам на gen-5 joint
log "running eval_domains on gen5 joint"
"$PY" scripts/eval_domains.py --config configs/dbridge_gen5_joint.yaml \
  --resume checkpoints/dbridge_gen5_joint/step-0001300.pt >> logs/gen5_joint_eval.log 2>&1 || \
  log "eval failed (non-fatal)"
log "pipeline done"
