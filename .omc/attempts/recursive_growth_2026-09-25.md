# Recursive autonomous growth — attempt lineage (opened 2026-09-25)

[1] Attempted to declare the HAGI loop "autonomous" from green mechanism
    evidence -> NO-GO. Momus review: transaction correctness, model
    quality, and autonomy are three different claims. Only the first is
    supported. `quality_supported=false` remains in force.

[2] Attempted to find a real-data execution path for one generation
    -> blocked, root cause measured: `src/hagi/orchestrator/recursive.py`
    and `scripts/recursive_growth.py` do not exist. The transactional
    primitives in `state.py` are implemented (811 non-blocked tests pass),
    but there is no owner that binds three children, a candidate, an
    independent holdout evaluator, and the CAS commit into one cycle.
    This is a missing binding, not a missing experiment.

[3] Attempted to find an autonomous quality-improvement source without
    backprop -> already refuted in earlier lanes and re-confirmed by
    the research map:
    - `bridge` as credit: deep dLoss=-2.2523e-02 (0.60 of autograd),
      broadcast 0.57, bridge 0.55, autograd -3.7647e-02
      (AGENT_WORKLOG.md:1479-1549).
    - head credit landing on held-out: 6/7 steps positive, no
      significant improvement, max |t| = 2.06 (AGENT_WORKLOG.md:1656-1874).
    - observability floor: 0.0021 nats at N=1023, resolvable
      rel_rms=0.100 (AGENT_WORKLOG.md:749-810, 1343-1404).
    => Do not relaunch the bridge/credit lane. Sunk-cost zone.

SELECTED NEXT: build the owner binding (one bounded slice), because it is
the only missing link that makes any future quality claim falsifiable.
No CLI, no training, and no new architecture component in that slice.

[4] Owner executor returned a new owner and tests while the slice was
    still in progress -> focused execution initially exposed 13 failures
    (path-vs-digest fixture misuse, callback signature mismatch, missing
    assertions) and Ruff F841. This is a test-contract failure, not a
    reason to weaken gates. Strategy change: freeze the test fixture/API,
    repair the owner contract, then add the two-generation fixture before
    reviewing any production binding.

[5] Independent state review found CAS post-write idempotency, parent
    lineage binding, lease-takeover race, and directory-durability gaps.
    These are state-machine blockers, not optional polish. Do not run real
    training or claim owner completion until red-green regressions cover
    each case.

[6] Fixed run_growth_cycle.sh to set -euo pipefail and propagate wait exit status; bash -n and injected exit-7 regression pass. No external reference found: web/MCP search unavailable, so used POSIX bash wait/set -e semantics and local regression.

[7] Delegated implementation failed twice (502 provider, then no-diff executor) -> changed strategy per Sunk Cost and implemented the deterministic split helper locally. Added split_packed_tokens(src/hagi/data/artifacts.py) with contiguous disjoint train/holdout ranges + fail-closed republish; 24 focused tests pass, ruff/py_compile/diff --check pass, non-gguf suite 838 passed. Still no owner wiring and no training.

[8] Added production-compatible state_key_digest helper in src/hagi/model/merge.py, deterministic over sorted state keys and tensor schema; tests cover order independence, shape/schema changes, and non-tensor rejection. Focused model/owner suite 41 passed; Ruff, py_compile, diff --check pass. No training or quality claim.

[10] The first production-wiring delegation produced no runner files; a
    concurrent full `pytest -q` process also stalled with zero CPU and no
    captured output. The RLS/CLI changes present in the dirty tree are a
    separate transaction-safety slice and were not used as evidence for
    growth wiring. Strategy change: do not wait on the stalled harness or
    relaunch it; run focused tests only, verify the artifact layout, then
    delegate a fresh narrowly scoped executor that may touch only the new
    runner/module/tests. If the executor again returns no files, change
    mechanism rather than retrying the same delegation.


[10] Reconciled schema-2 parent lineage: owner-bound test uses bound decision, legacy unbound path remains rejected, restored _accepted_decision_binds, corrected recovery-owner guard while preserving observed-owner allowlist. Orchestrator 45 passed; non-gguf full suite passed twice: 880 passed, 15 warnings each. No training or quality claim.

[12] Preregistered seed-4242 v1 (v2 protocol, seed 416114) was stopped by a
    mandatory preflight BEFORE any cycle ran. Removing only `resolved_seed`
    from `.omc/plans/two_generation_cpu_gate_v2.json`
    (sha256 95b45b9a182cf6e495cfc393e79f8439b28c0db5976dfd3c9b93e8b36ade230e)
    and hashing with `canonical_json_bytes` reproduces the recorded protocol
    digest `ce4f4369400b73f277e44fec45eaa70ffe20f20beed5ca39fb69cb3d86b3fc17`
    exactly, but that digest yields seed 301097, not the recorded 416114.
    v2 is therefore mathematically inconsistent with its own frozen rule and is
    INVALIDATED — NOT EXECUTED. v2 was left byte-for-byte unchanged; the
    finding is recorded in
    `.omc/plans/two_generation_cpu_gate_v2_invalidation.md`.

[13] Independent read-only adjudication (variant B chosen) rejected silently
    editing v2 and rejected weakening the derivation rule as post-hoc. A frozen
    v3 was created as a byte-for-byte copy of v2 with exactly one substitution
    (`resolved_seed` 416114 -> 301097), so the protocol digest stays
    `ce4f4369...` and `stored == derived == 301097`. Executable, CLI, and test
    pins now target v3. Red-first proof: the new preregistration contract test
    failed with `416114 != 301097` and passed after the correction; the runner
    suite is green (26, then 28 passed) and the Banking77 path keeps its own
    explicit seed instead of inheriting the synthetic pin.

[15] Second protocol defect found while adjudicating the seed question, and it
    is more important than the seed itself. Frozen v3 sets
    `claim_boundary.mechanism_supported_on_success=true`, but its acceptance
    block requires `gen1_and_gen2_accepted` from the unchanged owner
    `_verdict`, and `_verdict` decides purely on quality:
    `accepted = regression <= 0.0 and worst <= 0.01` (recursive.py:1322),
    where `path_only_evaluator` supplies real exact-CE metrics. So a
    transactionally perfect but non-CE-improving candidate is REJECTED, and
    the two-generation traversal can never be observed. The seed-4242
    execution failed exactly this way: macro CE improved by 0.0141862 nats
    but source B regressed 0.0307527, above the 0.01 budget, so gen2 was
    never reached. v3 therefore cannot support a mechanism claim; it only
    re-runs a quality screen. This was missed when v3 was frozen and is
    recorded rather than silently patched, because v3 is a frozen
    preregistration artifact. Read-only adjudication in flight over the
    narrowest defensible fix. Claims unchanged; gate still NOT run.

[14] The exact v3 gate has NOT run. Every verification attempt so far was
    invalidated by a concurrent external writer, not by our code: repeated
    120-second quiescence windows failed, and hash-guarded runs reported
    `real_cycle.py` / `recursive.py` changing mid-verification. Two independent
    reviews returned SNAPSHOT STALE for the same reason. The runner suite
    passed 28/28 on a snapshot that the hash gate simultaneously rejected, so
    that pass is explicitly NOT treated as evidence. Next required step: obtain
    120s quiescence, re-run the focused suites hash-guarded, obtain fresh dual
    SHIP verdicts, then execute exactly one v3 gate. Claims unchanged:
    `quality_supported=false`, `security_supported=false`,
    `autonomy_supported=false`, `production_promotion=false`,
    `mechanism_supported` unproven.

[16] Quiescence obtained for the first time this session (aggregate
    `2033701372e31677` stable across 5 x 30 s probes, then `8a4e17c607fea57f`
    across 4 x 30 s). On that stable snapshot two INDEPENDENT reviews ran
    (correctness + security/provenance) and both independently reported the
    SAME root defect, which is the strongest signal obtained this session:
    `_digest_of(report_path)` made the terminal report self-verifying
    (`x == x`), so the gate's root of trust had none. Three real defects were
    then fixed in the new gate only, never in the owner:
      (a) report digest now anchored in the owner-side ledger
          `state.json:terminal_report_sha256` (verified byte-equal on a real
          run) instead of against itself;
      (b) `MechanismGateError` no longer carries the expected/actual digests --
          a holdout digest in a builder-visible error is the same membership
          oracle that `.omc/research/holdout_isolation_design.md` removed;
      (c) every evidence-supplied path is now containment-checked against the
          run store root.
    Each fix is proven by MUTATION, not by assertion: reverting the anchor to
    `_digest_of`, re-leaking the digest, and removing the containment check each
    make a specific test fail, and all three are green again after restore.
    Note the first version of the anchor test was itself weak (it called the
    helper directly, so it passed against the mutant); it was rewritten to
    drive the real `_observe` path with a probe that rewrites a report field no
    check inspects. Tamper matrix is 13/13 detected.

[17] Full CPU suite on the stable pin: 1184 passed, 2 skipped, 15 warnings,
    hash gate clean. The recursive 6-file suite: 276 passed. This is +156 over
    the earlier 1028 without ANY owner change (`recursive.py`, `state.py`
    untouched) -- the growth is entirely new gate + new tests. The
    `mechanism_supported` honesty defect from [15] is confirmed on bytes:
    `recursive.py` writes it as a literal at :1436, :1462 and :1577, and
    `GenerationResult` requires it True at :687 even for a REJECTED generation.
    No production module calls the mechanism gate, so the flag remains an
    unearned constant. An external writer added two tests asserting exactly
    this; one of them is RED and stays RED on purpose -- fixing it would mean
    editing the frozen owner's claim semantics, which is out of scope. Claims
    therefore remain: `mechanism_supported` UNPROVEN as a persisted field,
    `quality_supported=false`, `security_supported=false`,
    `autonomy_supported=false`, `production_promotion=false`.
    The exact v3 gate is STILL NOT RUN: the last full-suite hash gate caught
    `tests/test_mechanism_gate.py` changing mid-run (the external writer), so
    that run is void as evidence and must be repeated on a fresh quiescent pin.

[18] v4 PREREGISTERED AND EXECUTED -- ONCE. Preregistration
    `.omc/plans/two_generation_cpu_gate_v4.json`, file SHA-256
    `19e3f68cc65ed7e30204a9259e6cb0fc28a16d607f047d3acf50676942e3db0c`.
    v4 copies v3 exactly and changes ONLY `acceptance`, `claim_boundary`,
    `schema`, `supersedes`, `seed_derivation` (+ one new provenance field);
    the experimental protocol is byte-identical to v3: synthetic domain,
    device=cpu, max_steps_per_child=1, and `no_go` all compare equal, and
    `resolved_seed` is unchanged at 301097.
    A defect was caught BEFORE execution and is worth recording: recomputing
    the seed derivation over the v4 envelope yields 785355, not 301097, because
    the acceptance change enters the hashed envelope. That is exactly the
    stored-vs-derived inconsistency that INVALIDATED v2. It is resolved by
    declaring `seed_derivation_protocol_sha256 = ce4f4369...` (the frozen v3
    protocol) inside v4, and verifying 301097 = int(ce4f4369[:8],16) % 1e6.
    Seed was NOT re-picked: it is the v3 output for a protocol v4 inherits.

    Execution, exactly one run, on pin `e3736f6bc570070608aeff1c33a65e3b`
    (all 11 pinned files unchanged before AND after the run):
      seed 301097, max_steps=1, device=cpu, transform=f3_tree, 2 calls.
    Owner outcome -- VERBATIM:
      generation-1: decision=rejected, parent=depth-zero, state=terminal
      candidate_macro_ce      = 4.969789028167725
      ce_regression           = 0.09561379750569632   (limit 0.0)
      worst_source_regression = 0.17634805043538382  (limit 0.01)
      per-source deltas: A +0.17634805043538382,
                         B +0.020524024963378906,
                         C +0.08996931711832623
      so ALL THREE sources regressed. gen2 was never created; the second call
      replayed the same terminal gen1.
    Receipts: report.json
      267b7f616b02f84a28e16b4dee6f4dfc92ef98bd5a9c3f20c83526b5e1f081ea
      holdout-evidence.json
      bd56ad5054f1787e14f789fbec58b567e15ec188e7d064474a738b6685cb30cd
      candidate checkpoint
      efa148ac2cad7e49bd66d451c206e9fbc29b00d7cf3e10a15a91e60ec6e5fe69
    The gate verdict is PASS, and it is NOT a rescue of 4242 and NOT a
    quality result: all 31 Tier 1 mechanism checks passed, meaning the
    owner-committed artifacts are internally consistent and correctly bound to
    the preregistered request. Tier 2 `accepted_gen1_gen2_traversal` is FALSE
    (`fewer than two generations were executed`), so the two-generation
    traversal remains unobserved. Claims after this run:
      mechanism claim is LIMITED to artifact integrity + binding under the
      unchanged quality screen -- that is what Tier 1 earned;
      the accepted gen1->gen2 traversal is NOT supported;
      quality_supported=false, security_supported=false,
      autonomy_supported=false, production_promotion=false.
    The `mechanism_supported: true` literal in the evidence is owner
    boilerplate ([17]) and is not read as evidence by this gate. Banking77
    training was NOT run. The failure mode is now unambiguous and is a
    modelling result, not a bug: the frozen F3 tree lift at max_steps=1 makes
    the merged model WORSE on every source, so `_verdict` correctly refuses
    to commit it. Growing under this screen requires a candidate that
    actually reduces exact CE, which one step of depth-lift does not do.

[19] DIAGNOSIS OF THE v4 REJECTION. A read-only agent proposed that the root
    cause was "insufficiently related train/holdout streams" and recommended
    making the holdout a variation of the training streams. THAT ROOT CAUSE IS
    REFUTED BY MEASUREMENT, and its supporting formulas do not exist in the
    code. The agent claimed train used `2 + 11*i + pos%23` while the holdout
    used `1 + 7*i + pos%19` -- i.e. different period (23 vs 19) and different
    inter-source stride (11 vs 7) -- and treated that asymmetry as the primary
    design defect. Grep for `% 19`, `pos % 19` and `1 + 7 *` returns nothing in
    `src/`. The actual code, `real_cycle.py:343 _synthetic_streams`, is a single
    definition with ONE period and ONE stride:
        [2 + source_index * 11 + position % 23 for position in range(48)]
    and `real_cycle.py:365` slices that same stream as
    `training = stream[:24]`, `holdout = stream[24:48]`. Measured: for every
    source the holdout is the training stream continued by exactly 24
    positions, with identical unit deltas in both halves (A 2->3, B 13->14,
    C 24->25, all deltas +1). The holdout is therefore as tightly coupled to
    the training slice as the protocol permits; it is a contiguous
    continuation, not a shifted or resampled progression. No offsets were
    chosen after seeing a result, and none are proposed: changing them would
    be the holdout-fitting this project forbids, and there is no need, because
    the claimed defect is not present.
    The agent's remaining hypotheses were also not evidenced: H3 (merge shifts
    CE) is contradicted by the existing depth-one and unequal-scale invariants;
    H6 (broader mixture) would require a joint optimizer that the v1 plan
    explicitly excludes, plus a new manifest/provenance contract.
    WHAT THE NUMBERS ACTUALLY SHOW, measured: each child is trained for
    `max_steps=1` step on a 16-token sequence drawn from a 24-token slice --
    roughly one gradient update over ~24 tokens. The three merged children have
    therefore barely moved from the shared depth-zero parent, and the F3 lift
    reweights three near-identical, barely-trained representations. The v4
    outcome is consistent with that: ALL THREE sources regressed
    (A +0.1763, B +0.0205, C +0.0900), which is the signature of an
    under-trained candidate rather than a broken merge or a leaky holdout.
    CONCLUSION: the honest blocker is the training budget, not the data
    definition, not the transform, and not the verdict. `max_steps_per_child`
    is pinned at 1 by the frozen preregistration, so raising it is NOT a
    repair of this gate -- it is a new preregistered experiment. The failing v4
    result stands as recorded in [18]; it is not re-run under a larger budget,
    because that would be exactly the post-hoc parameter search the seed-4242
    rescue was refused for. Claims unchanged.

[20] THE BUDGET HYPOTHESIS (H1) IS ALSO REFUTED, BY MEASUREMENT. [19]
    concluded that the children were under-trained. That was tested directly
    rather than argued, and it is wrong. A DIAGNOSTIC sweep (not a gate, not
    evidence, no preregistration -- explicitly not a registered execution) ran
    the same pinned seed 301097 at max_steps 1, 2, 4 and 8 on CPU, with the
    owner hash-gated before and after:
        max_steps=1  macro_reg=+0.09561  worst=+0.17635  rejected
        max_steps=2  macro_reg=+0.09498  worst=+0.17612  rejected
        max_steps=4  macro_reg=+0.09574  worst=+0.17686  rejected
        max_steps=8  macro_reg=+0.09536  worst=+0.17940  rejected
    Eight times the training budget moves macro regression by 0.0003 and does
    not change a single verdict. More steps cannot fix this gate, so the
    blocker is NOT an undertrained candidate and `max_steps` is not the lever.
    [19]'s own recommendation (raise the budget) is therefore withdrawn.
    WHAT IS ACTUALLY HAPPENING, from the same evidence: all three children
    carry an IDENTICAL config digest (child_config_sha256 collapses to one
    value), they start from one shared depth-zero parent, and each is trained
    for one step on its own source only. They are therefore near-copies that
    have barely separated. The F3 lift then re-expresses those three
    near-identical representations at ternary_depth 0 -> 1 while rescaling the
    logit scale 1.0597601791140758 -> 0.6118528246879578. Because the
    incumbent IS the depth-zero parent and the candidate IS the depth-1 lift,
    the owner's own comparison measures the lift's net effect in isolation:
    +0.0956 macro CE, worse on ALL THREE sources (A +0.1763, B +0.0205,
    C +0.0900), zero sources improved. A change that helps nothing and hurts
    everywhere is not a near-miss; it is a net-negative transformation at this
    scale. The self_improve stage is also inert here: the evidence records
    `accepted_updates: 0` with `n_new_tokens: 8`, so it contributed nothing.
    So the causal picture is: at this budget and this data scale, the depth-1
    ternary lift plus logit rescale is itself the regression, and the children
    are too similar for the merge to carry new information. Neither more
    steps nor more data plausibly fixes a transformation whose isolated effect
    is negative on every source. The merge is not "broken" in the sense of the
    existing depth-one/unequal-scale invariants, which still hold; the
    invariant is about logit equivalence of the representation, NOT about CE
    being preserved, so passing those tests never implied quality.
    The next honest step is a fresh preregistration that changes ONE
    preregistered variable and predicts the sign in advance, or an
    investigation of the logit rescale on its own. Claims unchanged:
    quality_supported=false, security_supported=false, autonomy_supported=false,
    production_promotion=false; the v4 result in [18] stands unrepaired.

[21] HISTORICAL REVIEW c58035ac CLOSED AS SUPERSEDED, NOT RE-ADOPTED. The
    long-running security review ("Review v2 handoff security") returned
    `SNAPSHOT PASS — SHIP` and its notification arrived late. Its verdict is
    NOT carried forward and is NOT treated as a current SHIP: its pin
    predates the final oracle remediation, the v2 invalidation, the v3
    correction, the new mechanism gate, and the v4 execution, so it does not
    describe the code that now exists. Stating that explicitly prevents a
    stale green verdict from being laundered into a current one.
    Two of its factual claims were re-verified against the current tree
    because they were load-bearing, and both hold:
      (a) the v2 preregistration is untouched -- `.omc/plans/
          two_generation_cpu_gate_v2.json` still hashes to
          `95b45b9a182cf6e495cfc393e79f8439b28c0db5976dfd3c9b93e8b36ade230e`,
          byte-identical to the frozen artifact, still carrying
          `status = PRE-REGISTERED — not yet executed`, and no
          `.tmp-two-generation-gate-v2` output directory exists. v2 remains
          INVALIDATED and NEVER EXECUTED; the separate invalidation record
          stands and v2 was not silently edited.
      (b) the holdout-isolation property of [15.1-A] still holds. Verified by
          field inspection, not by assertion: `ChildContext` carries
          `['generation_id','parent_checkpoint_path','parent_checkpoint_sha256',
          'parent_manifest_sha256','name','source_id','seed','span',
          'protocol_sha256']`, `CandidateContext` carries no data digest, and
          `ChildArtifact` carries no data/holdout/row-id field. The transitive
          path `CandidateContext.children -> ChildArtifact` is therefore clean.
          `HoldoutContract` DOES still carry `data_manifest_sha256` and
          `row_ids_sha256`, and that is not a regression: `run_generation`
          (`recursive.py:1643`) binds `HoldoutContract` only into
          `EvaluationContext`, which reaches the `evaluate` callback. The
          builder callbacks `build_child` and `build_candidate` are typed
          `Callable[[ChildContext], ...]` and
          `Callable[[CandidateContext], ...]` and can never receive it, so
          the membership oracle remains unreachable from a builder.
    The review's own `Must-fix: none` and its `security_supported=false`
    boundary are consistent with the current claim set. Its SHIP does not
    license any claim beyond that boundary, and does not substitute for the
    two fresh reviews obtained on the current pin in [16].

[22] BOTH EARLIER REVIEWS (fcb61d45 correctness, 353afcca security) CLOSED AS
    STALE-NO-VERDICT, AND THE REASON IS NOW EXPLAINED. Both stopped exactly as
    instructed: each detected that
    `tests/test_recursive_growth_orchestrator.py` hashed to
    `dc40e5c0455db7d7bb4ceca0622f6cce7c7f2ba71f7c0dee0da33135b12a794d` while its
    expected pin was `b54a5d48aa0f3ed6a8c5021ce70bc408f257acc4b6767f830cebefe88ac4e568`,
    and both refused to issue a verdict rather than reviewing a moving target.
    That refusal is the correct behaviour and is recorded as a pass of the
    review protocol, not as a failure of review.
    The drift is now resolved in a specific direction: `dc40e5c0...` is the
    value recorded in pin `8ee48f48393ca453b5201d274f706e74`, i.e. the external
    writer's change to that test file was absorbed into the snapshot that the
    two FRESH reviews in [16] then covered. So the stale verdict is superseded
    by real, current dual review, and no review gap remains on that file.
    A NEW drift appeared while closing these out, and is recorded rather than
    hidden: `src/hagi/orchestrator/real_cycle.py` moved again, from the pinned
    `59f8e4fbed9b7de365084ed4ea39d4d0767cfa7e76c759b11d3288a1dbc8239a` to
    `f54722f11a28e91162f3e349dcf0eac9e1df2b903511982b2b76a4b999f1123e`.
    Assessed for impact rather than assumed harmless:
      - owner decision core is untouched: `state.py` still hashes to
        `09b96f46...` and `recursive.py` still hashes to `30d1568e...`,
        matching pin 8ee48f48 exactly, so `_verdict`, the CAS, the state
        machine and the candidate validation are unchanged;
      - the preregistration guards survive: `SYNTHETIC_V3_SEED = 301097`
        (:63), `SYNTHETIC_V2_SEED = 416114` retained only to bind the
        invalidated artifact (:65), the API guard still raises
        "synthetic v3 requires preregistered seed 301097" at :889, and the
        CPU and `max_steps` range checks still precede `root.mkdir`, so
        fail-closed-before-side-effects still holds;
      - ruff and py_compile are clean on the drifted file.
    The v4 receipt from [18] is UNAFFECTED and was re-verified against its own
    bytes rather than assumed: `report.json` still hashes to
    `267b7f616b02f84a28e16b4dee6f4dfc92ef98bd5a9c3f20c83526b5e1f081ea` and
    still matches the owner ledger's `terminal_report_sha256`, the evidence
    still hashes to
    `bd56ad5054f1787e14f789fbec58b567e15ec188e7d064474a738b6685cb30cd`,
    the candidate checkpoint to
    `efa148ac2cad7e49bd66d451c206e9fbc29b00d7cf3e10a15a91e60ec6e5fe69`,
    and the recorded numbers are unchanged: decision `rejected`, macro
    regression 0.09561379750569632, worst 0.17634805043538382, deltas
    A +0.17634805043538382, B +0.020524024963378906, C +0.08996931711832623.
    The receipt is self-consistent and independently re-verifiable.
    CONSEQUENCE FOR NEXT WORK: any further execution must re-derive its own pin.
    The `8ee48f48` pin is spent. Claims unchanged: quality_supported=false,
    security_supported=false, autonomy_supported=false, production_promotion=false.
