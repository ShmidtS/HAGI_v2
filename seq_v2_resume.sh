#!/bin/bash
# v2 full pass, layers 6..42, reboot-resilient:
#  - outer per-attempt loop: process-level resume (--refit-threshold 1.0 on
#    retries skips existing checkpoints; HIP crashes ~700s are systemic on
#    this box, monotonic skip makes each attempt advance)
#  - collect is idempotent-ish (overwrites acts); refit resumes
# Usage: bash seq_v2_resume.sh [START_LAYER]
cd /c/HAGI_v2
PY=.venv/Scripts/python.exe
POD=checkpoints_dsv4/pod_all_tokens
LOG=seq_full_pass_v2.log
DONE=seq_full_done_v2.txt
TOKENS=${TOKENS:-262144}
START=${1:-6}

if [ -f seq_v2r.lock ]; then echo "already running"; exit 1; fi
touch seq_v2r.lock
trap 'rm -f seq_v2r.lock' EXIT

echo "=== v2 resume pass started $(date) from layer $START ===" >> $LOG

for L in $(seq $START 42); do
  prefix=$(seq -s, 0 $((L - 1)))
  # skip layer only when ALL 256 checkpoints are v2-era (n_val marker);
  # v1 files (no n_val) get purged and refit below
  NV=$("$PY" - "$L" <<'PYEOF'
import sys, glob, torch
L = sys.argv[1]
files = glob.glob(f"dsv4_reduced/layer_{L}/expert_*.pt")
n = 0
for fp in files:
    try:
        e = torch.load(fp, map_location="cpu", weights_only=False)
        if "n_val" in e:
            n += 1
    except Exception:
        pass
print(n)
PYEOF
)
  if [ "$NV" -ge 256 ] && [ -z "$FORCE" ]; then
    echo "layer $L already complete (v2, $NV ckpts), skip" >> $LOG
    continue
  fi

  # collect (fresh attempt per layer; overwrites acts file)
  for TRY in 1 2 3 4 5; do
    if [ -f checkpoints_dsv4/seq_v2/acts_layer$L.pt ] && [ "$TRY" -gt 1 ]; then
      break  # acts already collected on an earlier attempt
    fi
    echo "--- collect layer $L try $TRY (prefix $prefix) $(date) ---" >> $LOG
    SEQ_CH=4096 SEQ_LAYERS=$L I4X_LAYERS=$prefix $PY scripts/dsv4_collect_seq.py \
        --no-vocab --max-tokens $TOKENS --out-dir checkpoints_dsv4/seq_v2 \
        >> collect_v2.log 2>&1
    [ -f checkpoints_dsv4/seq_v2/acts_layer$L.pt ] && break
    echo "collect layer $L try $TRY failed, cooling 120s" >> $LOG
    sleep 120
  done
  [ -f checkpoints_dsv4/seq_v2/acts_layer$L.pt ] || { echo "COLLECT FAILED layer $L after retries" >> $LOG; exit 2; }

  cp checkpoints_dsv4/seq_v2/acts_layer$L.pt $POD/acts_layer$L.pt

  # purge v1-era checkpoints (no n_val key; their in-sample residuals would
  # wrongly pass the resume check). v2 partials (n_val present) survive -
  # this is what makes reboot-resume work.
  "$PY" - "$L" <<'PYEOF'
import sys, os, glob, torch
L = sys.argv[1]
for fp in glob.glob(f"dsv4_reduced/layer_{L}/expert_*.pt"):
    try:
        e = torch.load(fp, map_location="cpu", weights_only=False)
        if "n_val" not in e:
            os.remove(fp)
    except Exception:
        os.remove(fp)  # corrupt (crash mid-write) - recompute
PYEOF

  # refit with process-level resume
  for TRY in 1 2 3 4 5 6 7 8; do
    THRESH=1e-3
    [ $TRY -gt 1 ] && THRESH=1.0
    rm -f dsv4_reduced/layer_$L/expert_*.pt.tmp
    echo "--- refit layer $L try $TRY thresh $THRESH $(date) ---" >> $LOG
    PYTORCH_HIP_ALLOC_CONFIG=expandable_segments:True \
    W13_MODE=tern W13_BITS=2 W13_GS=128 W2_GPTQ=1 W13_GPTQ=1 PTQ_ONLY=1 VAL_FRAC=0.2 \
      $PY scripts/dsv4_refit_experts.py --start-layer $L --end-layer $((L + 1)) \
      --n-procs 1 --done-log $DONE --refit-threshold $THRESH \
      >> refit_v2_L$L.log 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
      echo "layer $L done $(date)" >> $LOG
      break
    fi
    N=$(ls dsv4_reduced/layer_$L/expert_*.pt 2>/dev/null | wc -l)
    echo "refit layer $L try $TRY rc=$RC ($N ckpts saved), cooling 120s" >> $LOG
    if [ "$N" -ge 256 ]; then break; fi
    sleep 120
  done

  N=$(ls dsv4_reduced/layer_$L/expert_*.pt 2>/dev/null | wc -l)
  [ "$N" -ge 256 ] || { echo "REFIT FAILED layer $L ($N/256) - rerun: bash seq_v2_resume.sh $L" >> $LOG; exit 3; }

  rm -f checkpoints_dsv4/seq_v2/acts_layer$L.pt $POD/acts_layer$L.pt
  echo "--- layer $L complete, acts freed $(date) ---" >> $LOG
done

echo "=== v2 PASS COMPLETE $(date) ===" >> $LOG
