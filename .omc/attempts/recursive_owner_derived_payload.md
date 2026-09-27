# Recursive owner derived-payload verification attempts

- [1] hypothesis: manifest/state-key digests prove that a candidate is the F3
  assembly of its three children -> refuted. Tampering `head.projection.weight`
  and recomputing `checkpoint_sha256` plus the manifest was accepted by
  `run_generation` (RED: no exception). Digests prove binding, not derivation.
- [2] fix: `_validate_derived_f3_payload` reconstructs the target with the
  production constructor `merge_recursive_f3` from the three durable child
  snapshots and compares every tensor. Reference: in-toto/SLSA verify the
  produced artifact against the rule, not against a self-consistent digest.
  Result: the same tamper now fails closed, parent stays byte-identical, and a
  durable `failure.json` is written. Owner suite 43 passed.
- [3] a scratch test file was briefly copied into `tests/` for the RED run and
  removed again; `git status` shows no residual file. Lesson: keep RED probes in
  the scratchpad, not in the shared tree, even when a full-path import requires
  a real module.
- [4] the shared checkout has at least four other live sessions editing
  `state.py` / `test_recursive_growth_orchestrator.py`. Hash stability is not a
  licence to reason about those files. Scope discipline: this slice touched only
  `recursive.py` and `test_recursive_growth_owner.py`; `state.py` stayed untouched
  because it is another lane's write authority.
- [5] closed: the engine and owner now represent a compliant fresh
  `PyramidAdapter` candidate (`scale=[0]`) through the derived-F3 contour
  exception and typed self-improvement ledger; no arbitrary adaptive key is
  allowed.
- [6] hypothesis: removing holdout fields from `ChildContext` and
  `CandidateContext` establishes callback isolation -> refuted. A filesystem
  probe found `holdout-*.bin` and owner-stamped strict child manifests already
  present in `g1/inputs` during later build callbacks.
- [7] fix: publish owner-bound strict manifests only after `build_candidate`
  returns and materialize the holdout into `evaluator-inputs` immediately before
  the evaluator. The regression enumerates the entire run during each of the
  four build callbacks and requires the owner-bound file set to be empty.
  RED: `1 failed`; GREEN: `1 passed`; owner suite: `70 passed`.
- [8] persisted `1`/`0` values could satisfy boolean claims through Python
  equality -> fixed with exact `type(value) is bool` checks before PREPARED and
  during terminal replay. Integer-coercion regressions preserve the incumbent
  parent bytes and publish a durable failure.
- [9] hash-stable final mechanism gate: production runner `24 passed`, owner +
  orchestrator + prepared + legacy `190 passed`, full suite `1075 passed,
  15 warnings`; scoped Ruff, `py_compile`, and `git diff --check` passed.
  This is not model-quality evidence.
- [10] hypothesis: late materialization is enough, because the holdout snapshot
  is verified when the owner writes it -> refuted. An independent review
  (agent `ad64db04-231b-4c5`, NOT PASS, Medium) showed that an evaluator may
  rewrite `evaluator-inputs/holdout-*.bin` after reading it, and the owner
  still committed an accepted parent: `holdout ACCEPTED accepted`. The
  equivalent mutation of a child or candidate manifest was already rejected
  with `bound evidence digest mismatch`, which is the intended behavior.
- [11] fix: re-verify the exact original holdout digest twice, immediately
  after `evaluate(...)` and again after owner evidence validation but before
  the evidence file and PREPARED. Reference: fixity checks compare a stored
  digest against the artifact as it currently exists, before and after the
  processing step (CASRAI fixity guidance; CWE-345 insufficient verification
  of data authenticity). Two regressions cover the two windows; both RED
  before the change, both GREEN after. Owner suite: `71 passed`.
- [12] concurrency lesson: a single logical fix cannot be verified on a moving
  checkout. `recursive.py`, the owner test, and `real_cycle.py` were changed
  by other sessions mid-run; one transient `contour_state_sha256` mismatch in
  the owner test helper was caused by a foreign half-applied contour-binding
  change, and the full-suite run had to be discarded with hash guard exit 97.
  Independent reviews must therefore run against an immutable copy whose
  imported module path is asserted.
- [13] hypothesis: verifying the parent digest once before the build callbacks
  is enough, because builders only read it -> refuted. An independent review
  (agent `4234bdde-20ee-4ce`, NOT PASS, High) and a separate probe agreed: a
  `build_child` callback mutated the live `parent.pt` in place, the run still
  finished `accepted`, the parent pointer advanced, no `failure.json` was
  written, and `report.json` attested the declared parent digest that no child
  or candidate had consumed. A follow-up probe showed the false lineage is
  promoted again in the next generation, so it becomes durable.
  This is not the pre-declared same-user limitation: the owner itself wrote a
  false provenance binding into its own content-addressed evidence.
- [14] fix: publish `inputs/parent-checkpoint-<digest>.bin` before the first
  callback, hand builders that snapshot instead of the live pointer target,
  and re-verify the live file after `build_candidate` before evidence
  validation. Two regressions cover the two properties: a mutation of the live
  file must fail closed before PREPARED, and all four builder contexts must
  point at the single owner snapshot. Both RED before the change, GREEN after.
- [15] fix (same review, Low): the late holdout snapshot name is
  content-addressed and therefore predictable by a callback that knows the
  holdout digest, so a pre-seeded byte-identical file was previously adopted
  idempotently. Contract section 15.2-A calls such an attempt an adversarial
  failure, so `_holdout_snapshot` now fails closed on any pre-existing target.
- [16] verified on the following snapshot: persisted-boolean review
  (agent `32e2d940-3f88-40d`) returned **PASS / 0 High / 0 Medium / 0 Low**,
  including a coherent re-sign attempt of evidence, prepared, terminal,
  `state.json`, and the parent pointer for each of the five boolean claims.
- [17] an external `ruff format` reformat of `recursive.py` landed during the
  full suite. A normalized diff against the pre-reformat snapshot shows only
  the import block plus the four intended fixes, and the owner suite still
  reports `75 passed`; any verdict pinned to the pre-reformat hash is stale.
- [18] hypothesis: handing builders the owner snapshot is sufficient because
  the snapshot is owner-created -> refuted. The re-review (agent
  `4a6182e9-644a-441`, NOT PASS, High) showed the same failure mode moved from
  the live path to the snapshot: a builder overwrote
  `inputs/parent-checkpoint-<digest>.bin` in place and the run still committed
  `accepted` while evidence attested a digest no builder had consumed. My own
  probe reproduced it exactly: `decision accepted`, attested `3e107c5c…`,
  snapshot on disk `1bd7931f…`.
- [19] fix: `_regular_bytes` re-verification of both the live parent and the
  owner snapshot after the last build callback, replacing the ad-hoc
  `sha256_file` comparison. `_regular_bytes` also rejects symlink and junction
  components, so a builder can no longer substitute the live parent with an
  identical-content symlink. A dedicated regression covers snapshot mutation.
- [20] refuted claim: the same review reported a High for a child hardlinked or
  byte-copied from the parent snapshot, on the grounds that the new snapshot
  path was missing from the alias set. Direct probe on both the snapshot and
  the live parent produced `ValueError: bound evidence digest mismatch` in both
  cases, so the alias was already rejected and nothing regressed. Lesson: a
  reviewer finding is a hypothesis until the reviewer and the owner both
  reproduce it; reporting it verbatim would have added a pointless guard.
- [21] current verified state on `9ee09427…` / `a7608e95…`: owner suite
  `76 passed`, owner + runner + orchestrator + prepared `177 passed`, full
  suite `1126 passed, 15 warnings` with both slice files byte-identical around
  the run. All three attack probes fail closed: snapshot mutation, live-parent
  symlink, and pre-seeded holdout.
- [22] a late notification resurrected an older verdict (agent
  `e81833ab-914d-491`) that was never collected because its task record had
  been cleaned. It is stale by hash, not by opinion: it pinned `recursive.py`
  `46d89300…`, the owner test `2133b1fa…`, and `state.py` `13c53a11…`, while
  the current bytes are `9ee09427…`, `a7608e95…`, and `09b96f46…`. Its two
  claims were re-checked rather than dismissed on age alone. The `Medium` about
  `data_manifest_sha256` still being present in `ChildContext` and
  `CandidateContext` at lines 585-606 is false on current bytes: an executed
  `dataclasses.fields()` and `dir()` scan shows no data or holdout field on
  either context type, and the owner callbacks see no owner-bound artifact.
  Its `Medium` about `_validate_evidence` accepting persisted JSON integers is
  equally false: the strict `type(value) is not bool` guard covers all five
  claims, and the nine targeted regression tests pass. Lesson reinforced:
  collect background verdicts before closing a cycle, and re-verify any
  resurrected verdict by hash and by execution instead of by age.
- [23] a second late notification (agent `4bbc255d-f0d6-4f9`, correctness audit)
  reported two High findings and one Low. It pinned `recursive.py`
  `cc6878fd…` and the owner test `f846de7b…`, which are the pre-late-
  materialization snapshots, so it described code that no longer exists. Both
  High findings were nonetheless re-tested by execution on a fresh immutable
  copy of the current bytes (`cce170c0…` / `a7608e95…`) rather than dismissed by
  age. A filesystem-capability probe that walks the whole run tree during each
  of the four build callbacks reported `child_A []`, `child_B []`,
  `child_C []`, `candidate []`, and the holdout present only afterwards in
  `evaluator-inputs/`. Its Low asked for negative assertions on the order of
  materialization; that assertion now exists permanently at
  `test_recursive_growth_owner.py:712` as
  `assert callback_leaks == [(), (), (), ()]`, and the orchestrator suite keeps
  the independent `test_build_callbacks_cannot_materialize_holdout_snapshot`.
  So all three items of this verdict are closed on the current contract, and the
  two High claims are refuted by probe rather than by argument.
- [24] while the above was being checked, a foreign lane began editing
  `recursive.py` again at 21:07:17 and left the module temporarily
  inconsistent: a new call to `_declared_transform_tree` at `:1073` referenced a
  function that was not yet defined, so two owner regressions failed with
  `NameError`. This is a half-applied foreign write, not a regression of this
  slice: the symbol is absent from the 21:01 snapshot as well, and the owner
  file was still changing every 30 seconds. The foreign edit was not reverted,
  because Chesterton's fence applies to a lane that is actively mid-edit. The
  correct action is to stop writing the slice, wait for quiescence, and
  re-baseline rather than to guess the intended helper. Quiescence arrived at
  21:11 with `30d1568e…`, and the four suites returned green.
- [25] the fifth late verdict (agent `9bc43e9a-92cc-485`, the original security
  audit) reported NOT PASS with one High: the holdout is snapshotted into the
  callback-visible run tree before any build callback, so a builder can read it
  undetected while the run still finishes accepted. It pinned `cc6878fd…` and
  `f846de7b…`, the same pre-late-materialization baseline as item 23, so it
  reviewed code that no longer exists. Its exact exploit was nevertheless
  re-implemented against the re-baselined bytes: a `build_child` that walks the
  entire run tree with `rglob` and additionally globs `inputs/holdout-*` read
  `NONE` bytes, while `evaluator-inputs/` held the single snapshot and only
  after all builds. Its NO-GO conclusion for training and promotion is
  unchanged and still correct; only the High itself is refuted. The permanent
  guard for this exact property is the owner regression asserting an empty
  owner-bound file set in all four build callbacks, plus the orchestrator
  regression `test_build_callbacks_cannot_materialize_holdout_snapshot`.
- [26] the sixth late verdict (agent `dea235a1-97c4-4ac`) returned PASS with
  `0 High / 0 Medium / 0 Low` for the narrowly defined late-materialization fix,
  pinned to `8540643…` and `563a2a4b…`. That is the pre-parent-snapshot state,
  so the verdict is superseded rather than contradictory: its scope is a strict
  subset of the final review in item 21, which covered the same property plus
  the parent-snapshot and re-check changes on later bytes. The property it
  endorsed was re-verified on the current baseline anyway, both by the
  filesystem-capability probe and by the four targeted regressions
  (`materializes_holdout`, both holdout-mutation tests, and the pre-seeded
  snapshot test) at `4 passed`. A PASS is only as good as the hash it names,
  but a superseded PASS is not a licence to relax: the current bytes were
  re-checked rather than assumed.
- [27] the seventh late verdict (agent `ad64db04-231b-4c5`, correctness) reported
  NOT PASS with one Medium: the evaluator can rewrite the late holdout snapshot
  after reading it and the owner still publishes an accepted result, pinned to
  `8540643…` and `563a2a4b…`. This is the finding closed in item 11, so the
  verdict describes the pre-fix state. Its exploit was re-run verbatim against
  the current bytes: the evaluator appends bytes to the snapshot
  (`12 -> 20 bytes`), and the owner now answers `ValueError: bound evidence
  digest mismatch` with the parent pointer byte-identical, no
  `prepared-report.json`, no `holdout-evidence.json`, no `report.json`, and a
  durable `failure.json`. Both regression tests pass. The report also credited
  a large amount of correctly verified surface (F3 reconstruction, ledger,
  strict booleans, two-generation lineage, recovery, symlink rejection, alias
  rejection), which is consistent with the final PASS in item 21.
- [28] the eighth notification repeated agent `32e2d940` rather than adding a
  new verdict. A duplicate is still checked, because a PASS pinned to older
  bytes does not transfer on faith. Here it did transfer on stronger grounds
  than the usual: the three functions that review depended on, `_validate_
  evidence`, `_read_terminal`, and `_result_from_evidence`, are byte-identical
  between its snapshot and the current file, and the `boolean_claims` tuple
  matches name for name. The seven bool-related regressions pass on current
  bytes. This is the one case where a superseded verdict could be adopted
  without re-running the whole adversarial battery, because the reviewed
  surface provably did not move.
- [29] the ninth notification repeated agent `4234bdde`, the TOCTOU audit whose
  High and Low were closed in items 14 and 15. Here rule 7 from the lessons
  note forced the opposite decision from item 28: the reviewed surface had
  moved, so its verdict could not be adopted by diff. `run_generation` grew
  from 9770 to 10611 characters and `_holdout_snapshot` did not exist in its
  snapshot at all, while `_snapshot` and `_write_exclusive` were byte-identical.
  The full battery was therefore re-run against the current bytes as one
  consolidated, reusable probe suite
  (`$TEMP/probe_owner_isolation.py`, six scenarios). Every one fails closed:
  live-parent mutation gives `bound evidence digest mismatch` with the parent
  still at `g0`; owner-snapshot mutation gives the same with a diverged
  digest; a pre-seeded holdout gives `late holdout snapshot already exists`; a
  symlinked live parent gives `symlink or junction is forbidden`; the
  capability walk reports an empty owner-bound set in all four build
  callbacks on an otherwise accepted run; and evaluator mutation leaves the
  parent pointer untouched with a durable failure marker.
- [30] the tenth notification repeated agent `4a6182e9`, whose NOT PASS with
  two High and one Low had already been triaged. The surface test again came
  out mixed: `run_generation` and `_holdout_snapshot` differ from its snapshot,
  so a full re-verification was required, while `_path_key` and `_validate_child`
  are byte-identical. The six-scenario battery still fails closed on
  `30d1568e…`. The refuted High H2 was re-tested more thoroughly than before,
  as a dedicated probe covering all four combinations of a child aliased by
  hardlink or by byte copy, against either the live parent or the owner
  snapshot. All four give `ValueError: bound evidence digest mismatch` with the
  parent pointer still at `g0`. The refutation therefore now rests on a
  2x2 matrix rather than on a single probe, which is the right standard for
  declining to implement a reviewer's High.
