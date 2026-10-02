#!/bin/bash
# Autonomous growth cycle with self-restart. The child training runs are
# killed periodically in this environment (no traceback, exit non-zero,
# log simply stops), so the cycle restarts itself and resumes from the
# last checkpoint. Written as a script rather than a nohup line so the
# supervisor's parent survives a shell teardown.
cd "$(dirname "$0")/.." || exit 1
for i in $(seq 1 200); do
  echo "=== cycle start $i $(date +%H:%M:%S) ===" >> logs/growth_cycle.log
  .venv/Scripts/python.exe -u scripts/growth/growth_supervisor.py \
    --plan configs/growth_gen4_seedfix.yaml --device cuda --max-lanes 1 \
    >> logs/growth_cycle.log 2>&1
  echo "=== cycle exited $i rc=$? $(date +%H:%M:%S) ===" >> logs/growth_cycle.log
  sleep 3
done
