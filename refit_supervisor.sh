#!/bin/bash
# Supervisor: iterative refinement driver.
# Waits for the running pass1 (--steps 100 coverage) to finish, then:
#   1. round-trip measurement on the covered model
#   2. pass2: --steps 5000 (bulk convergence, stall-stop active)
#   3. round-trip again
# Then STOPS for a decision (relax threshold / --warm 30000 / done).
cd /c/HAGI_v2

echo "$(date +%H:%M) waiting for pass1 (--steps 100) to finish..."
while powershell -ExecutionPolicy Bypass -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -like '*dsv4_refit_experts.py*--steps 100*'} | Measure-Object).Count" 2>/dev/null | grep -qv '^0'; do
  sleep 120
done
echo "$(date +%H:%M) pass1 done. round-trip #1..."
.venv/Scripts/python.exe scripts/dsv4_roundtrip.py --max-tokens 512 > roundtrip_p1.log 2>&1
echo "$(date +%H:%M) round-trip #1 done. pass2 (--steps 5000, stall-stop)..."
.venv/Scripts/python.exe -u scripts/dsv4_refit_experts.py --steps 5000 --n-procs 4 \
  --done-log refit_done_q3.txt --refit-threshold 1e-4 > refit_pass2.log 2>&1
echo "$(date +%H:%M) pass2 done. round-trip #2..."
.venv/Scripts/python.exe scripts/dsv4_roundtrip.py --max-tokens 512 > roundtrip_p2.log 2>&1
echo "$(date +%H:%M) supervisor finished — check roundtrip_p2.log and decide next pass"
