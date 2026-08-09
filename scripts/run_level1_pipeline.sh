#!/usr/bin/env bash
# Sequential level-1 growth pipeline driver.
# Waits for the currently-running expert (instruct) to finish, then trains
# math_code, merges all 4 experts into H=9216 (~1.018B), and joint-trains.
#
# Usage: bash scripts/run_level1_pipeline.sh
set -u
cd /c/HAGI_v2

PY=".venv/Scripts/python.exe"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# --- helpers ---------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Wait until no python train.py process is running.
wait_for_no_train() {
  log "waiting for training processes to finish..."
  while ps aux 2>/dev/null | grep -iE "train\.py|python" | grep -v grep | grep -q .; do
    sleep 60
  done
  log "no training process running"
}

# Wait until a specific PID is gone.
wait_pid() {
  local pid="$1"
  log "waiting for PID $pid to exit..."
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  log "PID $pid exited"
}

# --- Step 1: wait for instruct (PID 999) to finish -------------------------
if kill -0 999 2>/dev/null; then
  wait_pid 999
fi
log "instruct finished. checkpoints:"
ls -1 checkpoints_l1/instruct/ 2>/dev/null | tail -3

# --- Step 2: train math_code (sequential) ----------------------------------
# First wait for any currently-running training process (e.g. a math_code that
# was started manually) so we never launch a second one in parallel.
wait_for_no_train
if [ ! -d checkpoints_l1/math_code ] || [ -z "$(ls checkpoints_l1/math_code/ 2>/dev/null)" ]; then
  log "training math_code..."
  nohup "$PY" scripts/train.py --config configs/level1/expert_math_code.yaml \
    > "$LOG_DIR/train_math_code.log" 2>&1 &
  MATH_PID=$!
  log "math_code started (PID $MATH_PID)"
  wait_pid "$MATH_PID"
  log "math_code finished. checkpoints:"
  ls -1 checkpoints_l1/math_code/ 2>/dev/null | tail -3
else
  log "math_code already trained, skipping"
fi

# --- Step 3: merge 4 experts into H=9216 -----------------------------------
log "merging 4 experts -> H=9216 (~1.018B)..."
"$PY" scripts/merge_level1.py 2>&1 | tee "$LOG_DIR/merge_level1.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  log "ERROR: merge_level1 failed"
  exit 1
fi
log "merge done. checkpoints:"
ls -1 checkpoints_l1_merged/ 2>/dev/null | tail -3

# --- Step 4: joint-train merged model --------------------------------------
if [ -f checkpoints_l1_merged/step-0000000.pt ]; then
  log "joint-training merged model (H=9216)..."
  nohup "$PY" scripts/train.py --config configs/level1_merged.yaml \
    --resume checkpoints_l1_merged/step-0000000.pt \
    > "$LOG_DIR/train_level1_merged.log" 2>&1 &
  MERGED_PID=$!
  log "joint-train started (PID $MERGED_PID)"
  wait_pid "$MERGED_PID"
  log "joint-train finished. checkpoints:"
  ls -1 checkpoints_l1_merged/ 2>/dev/null | tail -3
else
  log "ERROR: merged checkpoint not found, cannot joint-train"
  exit 1
fi

log "=== LEVEL-1 PIPELINE COMPLETE ==="
