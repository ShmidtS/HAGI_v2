#!/usr/bin/env bash
# gen6_pipeline.sh — GAM phase-6 + fresh-data axis combined (G005).
# Merge the gen-4 pool with fresh_sib1 replacing sib_math (the
# fresh-data arm, R80/R116), GAM phase 6 renewal channel, and a
# slimpajama-reopened training mix (nu=0.145) on BOTH stages so the
# data axis stays alive through joint. Control arm: gen5_joint
# (GAM phase 5, no fresh data, eval AVG 2.9227).
# Идемпотентен: каждый шаг пропускается, если его чекпойнт уже существует.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gen6_pipeline.log
PY=.venv/Scripts/python.exe
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "pipeline start (pid $$)"

# 1. merged: GAM phase 6, fresh expert pool
if [ ! -f checkpoints/dbridge_gen6_merged/step-0001300.pt ]; then
  log "merged_gam6: launching (configs/dbridge_gen6_merged.yaml)"
  "$PY" scripts/train.py --config configs/dbridge_gen6_merged.yaml >> logs/gen6_merged.log 2>&1
  rc=$?
  log "merged_gam6: exit=$rc"
  if [ $rc -ne 0 ] || [ ! -f checkpoints/dbridge_gen6_merged/step-0001300.pt ]; then
    log "merged_gam6 FATAL (rc=$rc)"
    exit 1
  fi
else
  log "merged_gam6: checkpoint exists, skipping"
fi

# 2. channel split on merged step-0 (GAM health at init)
log "channel_split on gen6 merged step-0"
"$PY" scripts/measure_channel_split.py --ckpt checkpoints/dbridge_gen6_merged/step-0000000.pt \
  --config configs/dbridge_gen6_merged.yaml >> logs/gen6_channel_split.log 2>&1

# 3. joint: fresh-data mix carried through
if [ ! -f checkpoints/dbridge_gen6_joint/step-0001300.pt ]; then
  log "joint: launching (configs/dbridge_gen6_joint.yaml)"
  "$PY" scripts/train.py --config configs/dbridge_gen6_joint.yaml >> logs/gen6_joint.log 2>&1
  rc=$?
  log "joint: exit=$rc"
  if [ $rc -ne 0 ] || [ ! -f checkpoints/dbridge_gen6_joint/step-0001300.pt ]; then
    log "joint FATAL (rc=$rc)"
    exit 1
  fi
else
  log "joint: checkpoint exists, skipping"
fi

# 4. eval by domains on gen6 joint
log "running eval_domains on gen6 joint"
"$PY" scripts/eval_domains.py --config configs/dbridge_gen6_joint.yaml \
  --resume checkpoints/dbridge_gen6_joint/step-0001300.pt >> logs/gen6_joint_eval.log 2>&1

log "pipeline complete"
