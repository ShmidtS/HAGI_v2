# Bootstrap pointer idempotency — 2026-09-25 18:48–18:52

## Trigger

A late-finishing review agent (`1c1a8ada-96f2-47e`, started 10:44) returned
`BLOCKED` with two High findings. It reviewed an **old snapshot**: it read `state.py`
at 1398 lines and did not run tests. Current `state.py` is materially different
(`commit_accepted_terminal`, schema-2 bindings, `_fsync_directory`). Per
[Map ≠ Territory](omc-software-laws#map-is-not-the-territory) I adjudicated every
finding against the pinned bytes instead of accepting the verdict.

## Adjudication of the old findings

| Old finding | Status on current bytes | Evidence |
|---|---|---|
| High: CAS not idempotent after publish | **already fixed** | `state.py:1806-1809` — `current == replacement.as_dict()` → `return`; test `test_parent_cas_is_idempotent_after_successful_commit` |
| High: CAS does not bind accepted generation to `expected` lineage | **already fixed** | `state.py:1773-1782` compares the full parent triple and raises `parent lineage binding mismatch`; `test_parent_cas_rejects_terminal_bound_to_different_preflight_parent` |
| Medium: generation lifecycle lock not held to pointer write | **already fixed** | `state.py:1786-1794` enters `_recovery_locks` inside the parent lock; `_recovery_locks` (`:774-780`) takes `.lease-recovery.lock` then `.lifecycle.guard` |
| Medium: no parent-directory fsync | **already fixed** | `state.py:530-539` `_fsync_directory`; called at `:552` after `os.replace` and in the no-replace publish path; Windows is an explicit no-op with the claim scoped to process crash |
| High: bootstrap rejects a pointer it already committed | **REAL, fixed in this slice** | see below |

## The real gap

The two branches were asymmetric. The replacement-CAS path recognised a repeated
identical commit, but the bootstrap path raised unconditionally:

```python
if path.exists():
    if expected is None:
        raise ValueError("parent pointer already exists")   # state.py:1797-1798 (before fix)
```

`recursive_growth_v1.md:436` requires that a crash after pointer replacement "is
committed and is recovered idempotently from the pointer". A retry of the same
bootstrap therefore failed even though the commit had succeeded.

This is not a fail-open: the trusted token binds `owner_id` plus the pointer
digest (`state.py:1762-1766`), so an exact `current == replacement.as_dict()` match
proves it is the same commit. Any other stored pointer still fails closed.

## TDD evidence

RED, before the fix (`state.py=13c53a11…`, test `7007dce9…`, HASH_STABLE):

```
FAILED tests/test_recursive_growth_orchestrator.py::test_bootstrap_is_idempotent_after_successful_pointer_commit
state.py:1799: ValueError: parent pointer already exists
1 failed in 0.28s
```

GREEN, after the fix (`state.py=7c3cea26…`, test `7007dce9…`, HASH_STABLE):

```
5 passed in 0.48s
  test_bootstrap_is_idempotent_after_successful_pointer_commit
  test_parent_cas_is_idempotent_after_successful_commit
  test_parent_cas_recovers_after_process_crash
  test_owner_bound_parent_cas_and_pointer_owner
  test_parent_cas_rejects_terminal_bound_to_different_preflight_parent
```

Ruff: `All checks passed!`; `py_compile`: rc=0.

## Scope

Edited: `src/hagi/orchestrator/state.py` (one branch inside
`compare_and_swap_parent`) and `tests/test_recursive_growth_orchestrator.py` (one new
regression). No other file was touched. No commit was made.

## Reference note

Web search was unavailable (`BRAVE_SEARCH_API_KEY is not set`), so the fix adapts the
idempotent-commit pattern that already exists in the same function
(`state.py:1806-1809`) rather than an external source.

## Claims

`quality_supported=false`, `security_supported=false`, `autonomy_supported=false`.
This closes one High on the transaction layer. §15.1-A (holdout isolation) remains open
and blocks the security gate.

---

## Addendum — 18:51–18:59 (independent dual review + test hardening)

Two independent read-only reviews on pin `state.py=7c3cea26…`, test `7007dce9…`:

- Correctness (`2643be0f-6b3b-470`): `SNAPSHOT: PASS`, `VERDICT: PASS`. No code defect.
- Security (`2c04bb6c-7d58-448`): `SNAPSHOT: PASS`, `VERDICT: PASS`. No code defect.

Both verdicts were `PASS` **on the code**, and both found that my regression was
weaker than the invariant. That is a test-quality finding, not a code finding
([Pesticide](omc-software-laws#pesticide-paradox): a guard must be provably
breakable). Findings and remediation:

| Finding | Severity | Action |
|---|---|---|
| Negative case used a different `generation_id`, so a generation-only comparison would pass | Medium (test) | Probe 1 added: same `g0`, different checkpoint digest |
| Replay with a mismatched token was never exercised, so a pre-token-validation return would pass | Medium (test) | Probe 3 added: wrong `pointer_sha256` against the committed pointer |
| Same immutable identity with a different `owner_id` was never exercised, so a `_parent_identity` comparison would pass | Low (test) | Probe 2 added: identical digests, `owner-b` |
| A non-dict `current-parent.json` reported "parent pointer already exists", which is misleading | Low (code) | `state.py:1798-1800` now raises `malformed current-parent.json` first |

One Low from the correctness review — the idempotent early `return` skips
`_fsync_directory(path.parent)`, unlike `_publish_regular_no_replace` — was judged
not applicable: the no-op path performs no write, so there is no directory entry to
sync, and a lost pointer is recreated by the next real commit. No change made.

### Verification after hardening (pin `state.py=09b96f46…`, test `d2d5b286…`)

```
pytest tests/test_recursive_growth_orchestrator.py -k bootstrap
  2 passed, 55 deselected in 0.60s     HASH_STABLE
pytest tests/test_recursive_growth_orchestrator.py -k "not builder_contexts_expose_no_holdout_derived_digest"
  56 passed, 1 deselected in 4.30s     HASH_STABLE
ruff: All checks passed!   py_compile: rc=0
```

`bootstrap_parent` bootstrap-path regressions now exist:
`test_bootstrap_is_idempotent_after_successful_pointer_commit` and
`test_bootstrap_rejects_malformed_existing_pointer`.

### Still red, by design

`test_builder_contexts_expose_no_holdout_derived_digest` remains the only failing
test in the whole slice. It is the §15.1-A contract whose GREEN implementation
lives in `recursive.py` / `real_cycle.py`, outside this session's write lane. It is
deliberately red and is not counted as a regression.

---

## Addendum — 19:00 (late audit `7b487bbe` adjudicated: STALE)

A late-finishing audit agent (`7b487bbe-0ab0-4a0`, started 16:19) returned two High
findings about parent-receipt schema binding. It is **STALE**: it cited
`recursive.py:906-912` and `recursive.py:740-749`, but the current file is 1604
lines (`recursive.py=46d89300…`). Adjudicated against the pinned bytes:

| Stale finding | Status on current bytes | Evidence |
|---|---|---|
| High: `run_generation()` does not pass `request.parent` lineage into `PreparedDecision` | **FALSE on current bytes** | `recursive.py:1583-1585` passes `parent_generation_id`, `parent_checkpoint_sha256`, `parent_manifest_sha256` from `request.parent` |
| High: `_source_decision` drops parent fields | **FALSE on current bytes** | `recursive.py:1365-1367` restores all three from the persisted payload via `payload.get(...)` |
| Medium: `_decision_matches_report` omits parent fields | **FALSE on current bytes** | `state.py:1237-1239` compares all three explicitly |
| Medium: `_resume_terminal` early return could bypass lineage | **FALSE on current bytes** | `recursive.py:1416-1432` performs no early `current == replacement` return; it always goes through `store.recover` (`:1418`), which calls `_require_committed_accepted_terminal` |
| High: legacy schema-1 accepted can be promoted | **FALSE on current bytes** | `state.py:333-334` rejects `schema_version == 1` with `decision == "accepted"` (`accepted schema v1 unsupported`); tests `test_schema_v1_accepted_terminal_is_rejected_at_exact_committed_parent` (`:1145`) and `test_unbound_legacy_accepted_is_rejected_before_parent_cas` (`:1338`) |
| Medium: schema-1/2 strict separation and all-or-none parent fields | **PASS on current bytes** | `state.py:202-212` all-or-none; `:322-328` disjoint exact key sets for schema 1 and 2; `:289` `schema_version = 2 if parent_bound else 1` |

The only genuinely new statement in the late audit was a design outline for a migration
that is already implemented. Its own proposed outline (items 2-9) matches the shipped
`state.py` exactly. No code change is warranted from this audit.

The pinned snapshot used for this adjudication:
```
46d89300c4ee6878798e588d4f5282154953e7d6acaa940708a2a5f6cc85f205 *src/hagi/orchestrator/recursive.py
09b96f4626ca8f10d6ba3f64a6b1575703d4ec55a885877031cfe6cc12e0a537 *src/hagi/orchestrator/state.py
```

---

## Addendum — 19:01 (late dual audit `b61758e1` + `d04759ab` adjudicated: STALE, no findings survive)

Both late audits (`b61758e1-c832-454` correctness, 40 tool uses; `d04759ab-20c5-44a`
security, 25 tool uses) self-reported `STALE`. They read `state.py=8e7ff17c…`,
`recursive.py=afa98eb0…` (correctness) / `a609e6d3…` (security), and
`test_recursive_growth_orchestrator.py=46be2cc9…`. Current bytes are:

```
09b96f4626ca8f10d6ba3f64a6b1575703d4ec55a885877031cfe6cc12e0a537 *src/hagi/orchestrator/state.py
46d89300c4ee6878798e588d4f5282154953e7d6acaa940708a2a5f6cc85f205 *src/hagi/orchestrator/recursive.py
d2d5b286099545f6d84061bb2e034c88434ccb803f9f6a48b03e0943bb85e130 *tests/test_recursive_growth_orchestrator.py
e8dfb91a8b68fa2ab23f85b665cbfe7e4d1a766af43722c084260471784561ab *tests/test_recursive_growth_owner.py
```

The security agent produced no findings text (only a plan + a bare `STALE` with hashes),
so it carries no independent weight. The correctness agent produced three High. Adjudicated
against current bytes:

| Stale High | Status on current bytes | Evidence |
|---|---|---|
| Owner path still calls `publish_terminal()` before CAS; `commit_accepted_terminal()` unused | **FALSE** | `recursive.py:1469` and `:1600` call `store.commit_accepted_terminal(...)`; `publish_terminal` is reached only on the rejected branch (`:1473`, `:1597`) |
| `publish_terminal()` / `recover()` allow accepted terminal without a committed parent | **FALSE** | `state.py:1375-1376` rejects `accepted` unconditionally (`accepted terminal requires unified commit`); `recover()` at `:1354-1357` and `:1486-1489` calls `_require_committed_accepted_terminal` (`:1318-1338`) before accepting any accepted report |
| Crash test encoded the forbidden terminal-before-pointer window | **FALSE** | `tests/test_recursive_growth_owner.py:1440-1470` now asserts `state == "prepared"`, `prepared-report.json` exists, `report.json` absent, pointer bytes unchanged; plus the four crash-window tests after it |

No new defect survives adjudication. The only genuinely open blocker is still §15.1-A
(holdout isolation, Medium), whose GREEN implementation lives in `recursive.py` /
`real_cycle.py`, outside this session's write lane.

Notably, `tests/test_recursive_growth_owner.py` changed again during this adjudication
(`2133b1fa…` -> `e8dfb91a…`), so it must be re-hashed before any future run is quoted.
