#!/usr/bin/env bash
# Growth cycle: decorrelated corpora -> 3 small experts to saturation ->
# lossless compaction -> DFT-3 merge into one -> joint train to saturation.
#
#   [1] analyze corpus correlations (sort into weakly-correlated domains)
#   [2] train 3 small level-0 experts one at a time to saturation
#   [3] losslessly compact each expert checkpoint
#   [4] merge the 3 experts into one MergedHAGI with the ternary DFT-3 mixer
#   [5] joint-train the merged model to saturation
#
# Usage: bash scripts/run_growth_cycle.sh
set -u
cd /c/HAGI_v2

PY=".venv/Scripts/python.exe"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_pid() {
  local pid="$1"
  log "waiting for PID $pid to exit..."
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
  done
  log "PID $pid exited"
}

latest_ckpt() { ls -1 "$1"/*.pt 2>/dev/null | sort | tail -1; }

EXPERTS=(ru_general en_general math_code)
CKPT_DIR="checkpoints_l0"

# --- [1] sort corpora into weakly-correlated domains ------------------------
log "== [1] corpus correlation analysis =="
"$PY" scripts/analyze_corpora.py --data-dir data --samples 10000000 --groups 3 \
  2>&1 | tee "$LOG_DIR/analyze_corpora.log" | tail -8

# --- [2] train 3 small experts to saturation (one at a time) ---------------
for name in "${EXPERTS[@]}"; do
  if [ -d "$CKPT_DIR/$name" ] && [ -n "$(ls "$CKPT_DIR/$name"/*.pt 2>/dev/null)" ]; then
    log "$name already trained, skipping"
    continue
  fi
  log "== [2] training $name to saturation =="
  nohup "$PY" scripts/train.py --config "configs/level0_experts/expert_$name.yaml" \
    > "$LOG_DIR/train_$name.log" 2>&1 &
  T_PID=$!
  wait_pid "$T_PID"
  log "$name done: $(latest_ckpt "$CKPT_DIR/$name")"
done

# --- [3] lossless compaction of each expert checkpoint ----------------------
log "== [3] lossless compaction =="
for name in "${EXPERTS[@]}"; do
  ckpt=$(latest_ckpt "$CKPT_DIR/$name")
  [ -z "$ckpt" ] && log "ERROR: no checkpoint for $name" && exit 1
  "$PY" scripts/compact_checkpoint.py --ckpt "$ckpt"
done

# --- [4] merge 3 experts -> H=384 with ternary DFT-3 mixer ------------------
log "== [4] DFT-3 merge (3 experts -> H=384) =="
RU=$(latest_ckpt "$CKPT_DIR/ru_general")
EN=$(latest_ckpt "$CKPT_DIR/en_general")
MATH=$(latest_ckpt "$CKPT_DIR/math_code")
"$PY" scripts/merge_experts.py \
  --config configs/level0_merged_3.yaml \
  --experts "$RU" "$EN" "$MATH" \
  --out checkpoints_l0_merged/step-0000000.pt \
  --device cpu 2>&1 | tee "$LOG_DIR/merge_growth.log" | tail -6

# --- [5] joint-train merged model to saturation -----------------------------
log "== [5] joint-train merged model to saturation =="
if [ -f checkpoints_l0_merged/step-0000000.pt ]; then
  nohup "$PY" scripts/train.py --config configs/level0_merged_3.yaml \
    --resume checkpoints_l0_merged/step-0000000.pt \
    > "$LOG_DIR/train_level0_merged.log" 2>&1 &
  J_PID=$!
  wait_pid "$J_PID"
  log "joint-train done: $(latest_ckpt checkpoints_l0_merged)"
else
  log "ERROR: merged checkpoint missing"
  exit 1
fi

log "=== GROWTH CYCLE COMPLETE ==="
