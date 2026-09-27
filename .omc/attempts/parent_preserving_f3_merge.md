# Parent-preserving recursive F3 merge — attempt lineage

## Phase A: diagnosis (root cause, execution-grounded)

- [1] Read the worklog claim that `TernaryF3Tree` breaks self-merge; verified
  in float64 with the in-repo class: `TernaryF3Tree(1,2)` and `TernaryF3Tree(2,2)`
  both map an all-ones stream to per-leaf norms `[sqrt(3)x, 0, 0]` — CONFIRMED.
  Orthonormality is not the issue; staged aggregation is.
- [2] Searched for the minimal fix and found the existing, previously
  implemented but UNUSED `ParentPreservingTernaryLift` (Q(pi/2)) in
  `src/hagi/model/merge.py`. Verified: orthogonal, fixes the all-ones vector.
  => It was written but never wired into the merge path. This is the real gap.

## Phase B: design search (three candidate designs)

- [3] Candidate "Q outer + F3 inner". My first probe suggested it failed, but the
  probe had mixed axes (misuse of `apply_row`, which takes a FLAT width).
  Re-probed correctly: the flat API on duplicated parents IS duplication
  preserving. Hypothesis survived, probe did not.
- [4] Independent design critique raised a stronger objection: F3 annihilates the
  consensus direction `1`, Q only fixes it, so ANY composition containing an F3
  level on a path that can see three identical streams is not
  duplication-preserving. => "Q outer + F3 inner" is rejected on principle,
  not just on measurement. Adopted the critique.
- [5] Candidate "staged Q at every level" (drop-in for `TernaryF3Tree`). Measured
  F1 FAIL for parent_depth 1 and 2: applying Q to the thinnest level re-mixes
  inside a parent stream, which the parent has already done with F3. Rejected.
- [6] Candidate "outer Q only, inner identity" — SELECTED. All gates pass:
  F1 duplication preservation at parent_depth 0/1/2, F2 orthogonal, F3 fixes
  ones, F4 invertible, F5 mixes distinct parents with full norm retention.
  Stand: `.omc/tmp/f3_final.py`.

## Stand failures (my tooling, not the design)

- [7] Three consecutive failures of the comparison harness before switching to a
  minimal in-repo-API stand. Root cause: I was writing a bespoke staged
  reshape/matmul instead of using `ParentPreservingTernaryLift.apply_row` and
  `TernaryF3Tree.apply_row` as-is. Lesson: for an in-repo property, the fastest
  reliable stand calls the in-repo API directly. Changed approach rather than
  iterating on the reshape ([Sunk Cost]).

## Phase C: implementation

- [8] SUPERVISOR REDIRECT: design frozen to a plan, implementation delegated to a
  single executor with falsifying tests required (F5, F6) and an explicit
  MUST NOT touch list for the F3 contract.

## Phase D: implementation landed, three defects found after

- [9] Implementation landed and was independently verified by me: distinct-children
  merges stay finite and the two transforms genuinely differ on real logits
  (max|delta| 6.5e-01). F3 contract confirmed byte-identical: `_f3_real_column_matrix`
  still digests to the pinned `07e2571f2bfd0ff9`.
- [10] Independent review returned 0 blockers, 4 should-fix. All four reproduced by
  direct file reads before delegating: config/provenance disagreement on the explicit
  argument path, one dead function, tests that never exercised the inner step at
  parent_depth>0, and a provenance test that could not fail by construction.
- [11] Fixed and verified. Falsification checked by the executor, not asserted: a
  temporary swap of `parent_tree` produced 8 failures. One honest caveat recorded —
  the swap trips the depth guard first, so a second test was added to prove the
  type assertion is load-bearing rather than vacuous.
- [12] BLOCKER found by the quality-gate lane, not by any code reading: the new
  transform cannot run through the orchestrator AT ALL. `recursive.py:436` pins one
  single `_COORDINATE_LAYOUT` string, and the new tree declared
  `branch_major_outer_ternary:cross_parent:1:2` — geometry stuffed into a layout
  NAME. Root cause understood: geometry already lives in `transform_digest`.
  => A unit-level green suite said nothing about reachability. This is the cost of
  not gating on the real entry point earlier.

## Measurement discipline note

- [13] The first quality-gate attempt built a merge-level harness instead of using the
  real orchestrator. It showed the legacy arm degrading by ~0 instead of the known
  +0.236. The lane correctly identified its own harness as unfaithful and REFUSED to
  report the favourable numbers it had produced. That refusal is the right call and
  is the reason the blocker above surfaced at all. Recorded so the next lane treats
  "green harness I wrote myself" as untrusted until it reproduces a known baseline.

## Phase E: the "delta = 0.0" claim was wrong — root cause of the error

- [14] A later lane reported "identity-child CE delta exactly 0.0" for the new
  transform. I did not accept it. Re-ran the SAME harness with the SAME seed:
  got `+0.0267` (gen-1) and `+6.95e-08` (gen-2), not `0.0`. My own independent
  stand agreed the transform helps but does not reach zero.
  => The claim was withdrawn and the plan corrected. [Map != Territory]
  [Confirmation Bias: I nearly accepted a favourable number from an agent
  report without a reproduction run.]
- [15] Why the claim was believable at all: the harness had TWO arms
  (identity_child and trained_child) and several deltas per run, and the earlier
  JSON I read was a summary of one arm, not the counterfactual. A number
  extracted from a summary without checking which arm it came from is not
  evidence. Lesson: for a two-arm measurement, re-derive each number from raw
  output before repeating it.

## Still open (next experiment, not a claim)

- [16] Isolate the residual self-merge perturbation. Three candidates, none
  confirmed: the `head_scale_divisor = 3.0` head-receiver compensation, the
  `BlockTreeNorm` per-leaf statistics under duplicated leaves, and
  `RecursiveBranchScale`. The cross-parent step itself is exact (logit gain
  1.0), so the cause is downstream of it.
- [17] The Banking77 frozen pin `44d50edd...` no longer matches the artifact on
  disk (`9927589c...`). Cause is a `retrieval_timestamp` inside the manifest.
  Resolve by making the pin cover content, not metadata — do NOT simply
  re-pin, which would silently weaken the gate.
