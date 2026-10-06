#!/usr/bin/env bash
# gen7_pipeline.sh — merge-cycle gen7 (§AV follow-up): the GROW-licensed
# mechanism after the rank channel failed deep-200 certification.
# Ladder step: 3 domain siblings at the PARENT width (H=3456, init from
# gen6_joint — сиб-на-домен R66: chat/lang/math target the weakest gen6
# domains HELD_CHAT/RU) -> GAM merge phase 7 (fresh rank subspace,
# orthogonal to phases 1-6) -> joint at H=10368 -> eval_domains.
# Идемпотентен: каждый шаг пропускается, если его чекпойнт уже существует.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gen7_pipeline.log
PY=.venv/Scripts/python.exe
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "pipeline start (pid $$)"

run_stage() { # name config ckpt out
  local name="$1" config="$2" ckpt="$3" out="$4"
  # A stage is complete when EITHER marker exists: the full-horizon
  # step-0001300.pt, or best.pt written by the saturation stop (a run
  # that stopped early on plateau legitimately has no final ckpt).
  if [ -f "$ckpt" ] || [ -f "$(dirname "$ckpt")/best.pt" ]; then
    log "$name: already complete ($ckpt)"
    return 0
  fi
  # Auto-resume: a crashed stage restarts from its latest step-*.pt
  # (with optimizer state) instead of from scratch. train.py gives
  # --resume precedence over any init_from, and a missing-latest is
  # simply a fresh start -- no branching needed here.
  local resume_args=()
  if ls "$(dirname "$ckpt")"/step-*.pt >/dev/null 2>&1; then
    resume_args=(--resume)
    log "$name: partial run found, resuming from latest"
  fi
  log "$name: launching ($config)"
  "$PY" scripts/train.py --config "$config" "${resume_args[@]}" >> "$out" 2>&1
  local rc=$?
  log "$name: exit=$rc"
  if [ $rc -ne 0 ] || { [ ! -f "$ckpt" ] && [ ! -f "$(dirname "$ckpt")/best.pt" ]; }; then
    log "$name FATAL (rc=$rc, ckpt=$([ -f "$ckpt" ] && echo present || echo missing))"
    exit 1
  fi
  return 0
}

# 1-3. domain siblings at parent width (gen6_joint prior)
run_stage "sib_chat" configs/dbridge_gen7_sib_chat.yaml \
  checkpoints/dbridge_gen7_sib_chat/step-0001300.pt logs/gen7_sib_chat.log
run_stage "sib_lang" configs/dbridge_gen7_sib_lang.yaml \
  checkpoints/dbridge_gen7_sib_lang/step-0001300.pt logs/gen7_sib_lang.log
run_stage "sib_math" configs/dbridge_gen7_sib_math.yaml \
  checkpoints/dbridge_gen7_sib_math/step-0001300.pt logs/gen7_sib_math.log

# 4. merged: GAM phase 7, hierarchical (drop_expert_mixers)
run_stage "merged_gam7" configs/dbridge_gen7_merged.yaml \
  checkpoints/dbridge_gen7_merged/step-0001300.pt logs/gen7_merged.log

# 5. channel split on merged step-0 (GAM health at init)
if [ -s logs/gen7_channel_split.log ]; then
  log "channel_split: already measured (logs/gen7_channel_split.log)"
else
  log "channel_split on gen7 merged step-0"
  "$PY" scripts/measure_channel_split.py --ckpt checkpoints/dbridge_gen7_merged/step-0000000.pt \
    --config configs/dbridge_gen7_merged.yaml >> logs/gen7_channel_split.log 2>&1 || \
    log "channel split failed (non-fatal)"
fi

# 6. joint
run_stage "joint" configs/dbridge_gen7_joint.yaml \
  checkpoints/dbridge_gen7_joint/step-0001300.pt logs/gen7_joint.log

# 7. eval by domains on gen7 joint
log "running eval_domains on gen7 joint"
"$PY" scripts/eval_domains.py --config configs/dbridge_gen7_joint.yaml \
  --resume checkpoints/dbridge_gen7_joint/best.pt >> logs/gen7_joint_eval.log 2>&1 || \
  log "eval failed (non-fatal)"

# 8. distill: the COMPRESS half (RecursiveDistill.lean) — compact student
# H=3456 distilled from the generation's own joint (3H). The cycle repeats
# with the distillate as the next generation's parent: the model grows
# DENSITY at fixed width instead of parameters. delta = gate CE student vs
# teacher feeds cycle_report (leak gate / exhaustion); c = deep-200 vs the
# previous distillate (gen6_joint §AT baseline) certifies growth.
run_stage "distill" configs/dbridge_gen7_distill.yaml \
  checkpoints/dbridge_gen7_distill/step-0001300.pt logs/gen7_distill.log

# 9. eval by domains on the distillate
log "running eval_domains on gen7 distill"
"$PY" scripts/eval_domains.py --config configs/dbridge_gen7_distill.yaml \
  --resume checkpoints/dbridge_gen7_distill/best.pt >> logs/gen7_distill_eval.log 2>&1 || \
  log "eval failed (non-fatal)"

# 10. delta: the bridge slack = gate CE of the student MINUS gate CE of
# the teacher, on the SAME canonical windows (gate_score.py reproduces
# the in-run gate exactly, so the two numbers are comparable).
log "measuring cycle delta (gate CE student vs teacher)"
T_GATE=$("$PY" scripts/gate_score.py --config configs/dbridge_gen7_joint.yaml \
  --ckpt checkpoints/dbridge_gen7_joint/best.pt 2>/dev/null | sed 's/.*gate_ce=//')
S_GATE=$("$PY" scripts/gate_score.py --config configs/dbridge_gen7_distill.yaml \
  --ckpt checkpoints/dbridge_gen7_distill/best.pt 2>/dev/null | sed 's/.*gate_ce=//')
log "teacher_gate=$T_GATE student_gate=$S_GATE"

# 11. gain: deep-200 certification of the distillate against the gen6
# baseline the certify script carries (EN/HELD_CHAT/MATH/CODE/...).
log "deep-200 certify on gen7 distill"
"$PY" scripts/growth/deep200_certify.py --log logs/gen7_distill_eval.log >> logs/gen7_distill_certify.log 2>&1 || \
  log "deep200 certify failed (non-fatal)"

# 12. cycle verdict: append (delta, gain) to the ledger and obey the
# cycle_report verdict (CONTINUE / STOP_LEAK / EXHAUSTED).
# NOTE: gain must be read from the certify log -- plug the certified
# AVG improvement over the gen6 baseline in manually once the log is
# written; the gate refuses to guess.
log "cycle_gate pending manual gain (see logs/gen7_distill_certify.log)"

log "pipeline complete"
