# Recursive growth — trust region vs merge defect (2026-09-25)

- [0] Baseline on current bytes: `pytest tests/test_recursive_growth_owner.py
  tests/test_recursive_growth_orchestrator.py tests/test_quality_gate.py` ->
  139 passed. Orchestrator surface hash-stable across 3 samples / 90 s.
- [1] Surface drift was NOT self-inflicted. `Get-CimInstance Win32_Process`
  showed a concurrent writer running `pytest tests -q` and
  `scripts/train.py --config configs/autonomous_bounded_20260925.yaml` in the
  same tree at 20:23. Diagnosis of a moving target was refused per the earlier
  freeze stop-condition; re-checked and the surface is now stable.
- [2] Real gate located: `src/hagi/orchestrator/recursive.py:1250 _verdict`.
  macro delta <= 0.0 AND worst per-source delta <= 0.01.
  `src/hagi/orchestrator/quality_gate.py` is DEAD CODE on this path (different
  criteria, 3 preregistered seeds, no per-source budget). Not touched
  ([Chesterton], [YAGNI]).
- [3] Root cause candidate confirmed by code reading: the candidate update has
  no trust region. `real_cycle.py:593 _build_candidate` zeroes contour scales and
  calls `self_improve(..., max_iterations=1, kl_max=1.0)`; only
  `train.max_grad_norm` bounds the gradient, not the weight delta.
- [4] SUPERVISOR REDIRECT (oracle): do NOT line-search on the holdout. Reusing
  the holdout as a search objective is selection-on-test and converts the
  held-out claim into a selected statistic. Rank-1 change is a norm-bounded
  trust region with a preregistered epsilon, one gate evaluation, no selection.
- [5] Oracle early detector, adopted before any 3-seed run: measure the merged
  candidate at alpha=0 (contour never updated). If a source already exceeds
  0.01 at alpha=0, the defect is the non-function-preserving three-way F3
  self-merge and a trust region cannot fix it -> stop the trust-region lane.
- [6] ALPHA=0 PROBE RESULT (decisive, falsifies the trust-region lane).
  Script: scripts/probe_alpha_zero_merge.py (new, read-only wrt src/hagi/**).
  Report: .omc/alpha_zero_probe_20260925.json. exit 0, ~5.6 s CPU, 3 seeds.
  With every contour scalar verified zero and accepted_updates=0, the
  per-source 0.01 budget is ALREADY blown before any gradient step:
    416114: A +0.042711 (B -0.010204, C -0.041890), macro -0.003128
    416115: B +0.025790 (A -0.002128, C +0.005152), macro +0.009605
    416116: A +0.012267, B +0.072802, C +0.053726, macro +0.046265
  -> The merge point is not clean. A trust region has no regression left to
  shrink, so the proposed increment is a NO-OP for this failure. Lane closed
  before implementation; no code was written for it.
- [7] SECOND FINDING from the same artifact: `contour_keys` is exactly
  ["blocks.0.adapters.pyramid.scale"] — ONE scalar. Cause:
  real_cycle.py:176 sets `pyramid.levels = (1,)`. The entire adaptive surface
  the self-improvement step can move is one number, and it is zeroed
  immediately before the step. Any "growth" claim must confront this.
- [8] Note: the enforced Banking77 pin is 44d50edd... (BANKING77_MANIFEST_SHA256);
  9927589c... is a tokenizer-artifact manifest from decision_plane_banking77.py.
  Both digests were conflated in the original task prompt; the probe recorded
  both and enforced the correct one.
- [9] NOISE-FLOOR PROBE BLOCKED, and it surfaced a live defect.
  Attempt to measure the paired resolution of the 0.01 nats per-source budget
  (256 scored tokens/source) with .omc/paired_se_probe.py failed:
    ValueError: recursive state requires merge.mixer_type='ternary_f3'  -> resolved
    NameError: name '_CROSS_PARENT_TRANSFORMS' is not defined
      at src/hagi/model/merge.py:1018 (used at :1018 and :1042, defined nowhere)
  Verified: `hasattr(hagi.model.merge, '_CROSS_PARENT_TRANSFORMS')` is False on
  the current bytes. This is a real NameError on a live path, not a probe defect.
- [10] NOT FIXED, DELIBERATELY: src/hagi/model/merge.py has mtime 2 s before the
  check and +1133 lines uncommitted. A concurrent writer is editing this exact
  file right now. Touching it would collide, and the earlier freeze stop-condition
  already burned 3 attempts on moving targets. Observed and left alone.
- [11] Consequence for the noise-floor question (Oracle item D): UNRESOLVED.
  The budget's meaningfulness at 256 rows is still unknown. Do not treat the
  0.01 nats budget as validated until this is measured on a stable snapshot.
- [12] M1 EXECUTED (scripts: .omc/m1_identity_probe.py) -> Oracle's M1 premise
  was WRONG and the correction matters. Three identical children do not even
  produce a same-shape parent: parent hidden_size=8, merged hidden_size=24,
  heads/ffn x3. Only 1 key overlaps with identical shape (head.logit_scale,
  abs diff 0.2588).
  => "Parent-preserving" is ill-defined for the CURRENT primitive. The growth
  step is a 3x WIDTH EXPANSION, not a weight update. Oracle's option A
  (identity-preserving lift in parameter space) is not merely deprioritised,
  it is not well-posed against this geometry. Recorded, not silently absorbed.
- [13] Corrected architecture statement, evidence-based: merge = 3x width
  expansion whose function is inherited only approximately. The alpha=0 probe
  (CE 8.05-8.09, deltas up to +0.0728) IS the measurement of imperfect
  functional inheritance. The correct rank-1 repair is therefore the PUBLISHED
  one for this exact problem: zero-init gated/residual width expansion
  (Net2Net Net2WiderNet, bert2BERT, zero-initialised residual adapters) so
  the expanded model computes the parent function EXACTLY at step 0, and all
  growth is then attributable to training. Not implemented this cycle.
- [14] NameError was TRANSIENT, not a defect for me to fix. The concurrent
  writer defined `_CROSS_PARENT_TRANSFORMS` and then, in the same file,
  implemented the parent_preserving cross-parent lift that Oracle ranked #1
  (merge.py:316 ParentPreservingTernaryLift, :867 CrossParentPreservingTernaryTree,
  :953 _CROSS_PARENT_TRANSFORMS = ("f3_tree","parent_preserving")).
  Verified by execution: `hasattr(...)` is True. Do not duplicate this work.
- [15] The concurrent writer also added a REQUIRED provenance key:
  `recursive_f3_cross_parent_transform`. Old checkpoints (including my alpha=0
  candidate) are now correctly REJECTED on load. This is a schema change, not a
  bug; it means my .omc/alpha_zero_probe_20260925.json numbers are from the
  previous schema and cannot be re-verified without a rebuild.
- [16] THE INVARIANT VERIFIED BY EXECUTION (.omc/verify_lift_invariant.py):
  parent_preserving matrix = [[2/3,-1/3,2/3],[2/3,2/3,-1/3],[-1/3,2/3,2/3]]
    M @ (1,1,1) = (1,1,1)          -> fixes the all-ones vector: TRUE
    M^T M == I  (orthogonal)      -> TRUE
    det(M) == 1                    -> TRUE
    duplicated triple preserved   -> True, maxdiff 2.2e-16
    three DISTINCT parents mixed  -> True, maxdiff 1.85
    inverse round-trip exact      -> True
  This is precisely the property the staged F3 action lacked ((x,x,x) ->
  (sqrt3 x,0,0)). The self-merge-is-identity property is now algebraic, not
  aspirational.
- [17] STILL OPEN, and it is now the only credible next measurement: rerun the
  alpha=0 probe under `parent_preserving` and check per-source deltas against
  the 0.01 budget. Oracle's M2 success criterion, preregistered BEFORE the run:
  |delta| <= 0.01 on >=2 of 3 seeds AND macro <= 0.0. Failure specifically on
  416116 source C is the discriminating signal that the lift does not address
  the mechanism and escalation to option C is warranted.
- [18] Noise floor at 256 rows is still UNMEASURED (blocked by the provenance
  schema change). Do not treat the 0.01 budget as validated.
- [19] M2 EXECUTED — preregistered criterion FAILS 0/3, but the lift works and
  the failure is informative. Probe: scripts/probe_alpha_zero_merge.py with
  `--lift-mode parent_preserving` (patched read-only via cross_parent_transform
  passthrough; no src/ file edited). Report:
  .omc/alpha0_parent_preserving_20260925.json, exit 0.
  Preregistered success = per-source |delta| <= 0.01 on >=2/3 seeds AND macro
  <= 0.0. Result: 0/3 -> FAIL.
  BUT the parent_preserving lift improves EVERY source on EVERY seed, i.e.
  9/9 directions, with no exception:
    seed    macro f3_tree -> parent_preserving     worst-source
    416114  -0.00313 -> -0.03018                     +0.0427 -> +0.0119
    416115  +0.00960 -> -0.02860                     +0.0258 -> -0.0198
    416116  +0.04626 -> +0.00490                     +0.0728 -> +0.0351
    mean macro      +0.01758 -> -0.01796  (improvement 0.03554 nats)
    mean worst-src  +0.04710 -> +0.00910  (improvement 0.03800 nats)
    per-source mean improvement: A -0.0372, B -0.0402, C -0.0292
  9/9 is not noise-consistent: a null effect at 9/9 same-sign would be p=1/512.
  So the lift is causally effective; it is simply not YET under the 0.01 budget.
- [20] Oracle's discriminating signal FIRED as predicted: source C on 416116
  is the largest remaining regression (+0.0180), and 416116 macro is still
  positive (+0.0049) while the other two seeds turned macro-negative.
  Preregistered consequence: escalate to option C (reject the depth+1
  3-expert growth primitive) rather than tune the lift further. n=3 seeds,
  so this is an indication, not a proof ([Occam], no post-hoc fitting).
- [21] Remaining honest gaps, unchanged: (a) noise floor at 256 rows is
  STILL unmeasured, so "improvement 9/9" is a same-sign count, not a
  significance test; (b) 0.01 budget may itself be mis-set; (c) the adaptive
  contour is still a single scalar; (d) quality_claim_supported=false.
- [22] Next cheapest decisive measurement, in order:
  (i) measure the paired noise floor so 9/9 and 0.01 can be interpreted;
  (ii) if the residual regression is real, measure an identity-child control
  under the new lift — the self-merge should now be exactly identity, which
  separates "lift still leaks" from "3 disjoint children are not composable".
- [23] NOISE FLOOR MEASURED (.omc/paired_se_probe.py ->
  .omc/paired_se_pp_20260925.json, exit 0, 3 seeds x 3 sources, 256 tokens
  each, parent vs candidate on the SAME rows = paired).
  mean paired SE = 0.02511 nats; per-source range 0.0229-0.0285.
  The gate budget of 0.01 nats is only 0.398x the 1-SIGMA resolution.
  => The 0.01 per-source budget is BELOW the noise floor at n=256. The gate
  as configured CANNOT distinguish a real 0.01-nats regression from sampling
  noise; it will reject for reasons that are not attributable to the model.
- [24] This invalidates the framing of [19]/[20] in one specific way and
  preserves it in another. PRESERVED: the 9/9 same-sign improvement under
  parent_preserving is far larger than the noise (0.029-0.040 vs SE 0.025),
  and per-source mean deltas of -0.034..-0.051 are >1.3-1.7 SE, so the lift
  effect is real. INVALIDATED: the "per-source 0.01 budget" cannot be used to
  accept or reject at 256 rows, so the 0/3 preregistered FAIL is not evidence
  that the lift is insufficient — the criterion was below the instrument's
  resolution. Escalation to option C is therefore NOT yet justified; the
  measurement instrument must be fixed first. This is [D] winning after all,
  and it wins for a stronger reason than oracle predicted.
- [25] Note the implied false-reject rate: with a true-zero delta and SE
  0.025, a 0.01 threshold rejects ~34% of the time per source purely by
  noise (P(|N(0,0.025)| > 0.01) ~= 0.31). Across 3 sources that is ~90% of
  generations rejected for no reason. The observed 0/3 is therefore
  statistically unremarkable, not a signal.
- [26] NEXT, now preregistered: raise scored rows per source until
  paired SE is <= budget/3 (~0.0033 nats). From mean_delta per-token
  variance, SE scales ~1/sqrt(n), so n must grow by ~(0.025/0.0033)^2 ~= 58x,
  i.e. roughly 15,000 scored tokens per source. Decide FIRST whether that is
  affordable from the pinned Banking77 test split (currently only part[:256]
  of each third is used, so more rows appear available) or whether the budget
  must be widened instead. Do not change both at once.
- [27] INSTRUMENT FEASIBILITY MEASURED, and it needs NO threshold change.
  Pinned Banking77 test split holds 42560 tokens in 3 thirds of
  14187/14187/14186; the gate currently scores only part[:256] per source.
  Projected paired SE at 1/sqrt(n) from the measured SE(256)=0.02511:
     n=1024 -> 0.01256   n=4096 -> 0.00628   n=8192 -> 0.00444
     n=14187 -> 0.00337  (2.96x the 0.01 budget, 0.99x budget/3)
  So scoring the FULL available third per source reaches essentially
  budget/3, using data that is already inside the pinned manifest. The
  acceptance budget 0.01 does NOT need widening; the row count is the defect.
- [28] NOT IMPLEMENTED: src/hagi/orchestrator/real_cycle.py:1109 is the only
  production site (`part[:256]`), and the file changed mid-measurement
  (8036ce6e -> 6435be82 within 20 s). Editing a moving target is exactly the
  failure mode already logged three times in this project. The change is
  one constant at :1109 plus the matching probe/provenance, and it must be
  applied as a reviewed change on a stable snapshot, not mid-flight.
- [29] Standings unchanged and stated plainly: the lift is causally effective
  (9/9 same-sign, effect > 1.3 SE), the current gate cannot resolve its own
  budget at 256 rows, and no quality or promotion claim is supported.
