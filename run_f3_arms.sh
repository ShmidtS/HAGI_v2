#!/bin/bash
# The three arms of the pre-registered tree-vs-flat claim, run back to back.
# One at a time: STATUS.md records that a second trainer on top of the first
# cost 45 GPU-minutes reproducing an existing checkpoint.
set -u
cd /c/HAGI_v2
for arm in f3_parent_d1 f3_flat_d1 f3_parent_d1_nocortex; do
  echo "=== START $arm $(date +%H:%M:%S) ==="
  python scripts/train.py --config configs/$arm.yaml --device cuda \
      > logs/arm_$arm.log 2>&1
  echo "=== END $arm rc=$? $(date +%H:%M:%S) ==="
done
echo "ALL ARMS DONE $(date +%H:%M:%S)"
