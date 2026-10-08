#!/usr/bin/env bash
# density_ladder.sh — §17/R134 REVERSE density ladder (2026-10-08):
# gen7 (10368) -> 3456 -> 1152 -> 384. Width strictly DECREASES, so VRAM
# can only shrink (the gen8 3x-merge OOM is structurally impossible).
# Each level: student at the historical generation width, initialized
# from the same-width prior joint (L3: fresh — no gen1 ckpt on disk),
# distilled from the level above (teacher best.pt).
# Verdict per level: certified 2eps (deep200_certify) against the
# same-width historical baseline (L1 vs gen6_joint, L2 vs gen3_joint,
# L3 record-only). Idempotent: a stage is skipped when its best.pt
# already exists; crashed stages auto-resume from latest step-*.pt.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/density_ladder.log
PY=.venv/Scripts/python.exe
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "density ladder start (pid $$)"

run_stage() { # name config out
  local name="$1" config="$2" out="$3"
  local dir
  dir=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1],encoding='utf-8'))['train']['checkpoint_dir'])" "$config")
  if [ -f "$dir/best.pt" ] || [ -f "$dir/step-0001300.pt" ]; then
    log "$name: already complete ($dir)"
    return 0
  fi
  local resume_args=()
  if ls "$dir"/step-*.pt >/dev/null 2>&1; then
    resume_args=(--resume)
    log "$name: partial run found, resuming from latest"
  fi
  log "$name: launching ($config)"
  "$PY" -X utf8 -u scripts/train.py --config "$config" "${resume_args[@]}" >> "$out" 2>&1
  local rc=$?
  log "$name: exit=$rc"
  if [ $rc -ne 0 ] || { [ ! -f "$dir/best.pt" ] && [ ! -f "$dir/step-0001300.pt" ]; }; then
    log "$name FATAL (rc=$rc)"
    exit 1
  fi
  return 0
}

# Gate 0: the ladder starts from the FINISHED gen7 joint (teacher of L1).
if [ ! -f checkpoints/dbridge_gen7_joint/best.pt ]; then
  log "gen7_joint best.pt missing — finish gen7 joint training first"
  exit 1
fi

# L1: 10368 -> 3456 (init gen6_joint, teacher gen7_joint)
run_stage "L1_3456" configs/dbridge_gen7_distill.yaml logs/density_L1_3456.log
# L2: 3456 -> 1152 (init gen3_joint, teacher L1 distillate)
run_stage "L2_1152" configs/dbridge_density_1152.yaml logs/density_L2_1152.log
# L3: 1152 -> 384 (fresh init, teacher L2 distillate)
run_stage "L3_384" configs/dbridge_density_384.yaml logs/density_L3_384.log

log "density ladder complete"
