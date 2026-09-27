# Snapshot stabilization gate — 2026-09-25 17:27–17:46

Status: **BLOCKED — snapshot never quiesced.** No verdict, no GO, no CPU gate credit.

## What happened

Verification of the transactional owner slice could not be completed because the
workspace is being mutated continuously by a writer outside this session. Three
consecutive verification attempts failed for the same reason (Sunk Cost rule of
three reached -> approach changed from "verify now" to "measure quiescence").

Attempt 1 (17:27:42) — focused 5-file CPU suite, HASH_STABLE across the run:
`160 passed, 1 failed`. The single failure is a test-contract mismatch in
`tests/test_recursive_growth_owner.py::test_builder_context_graph_exposes_no_holdout_or_store`
(expects the old exact `CandidateContext` field set; the runtime now also carries
`base_seed` and `self_improve_seed`). At that snapshot `state.py=bd194b31`,
`recursive.py=2ef8e1bc`, orchestrator test=`e9a89766`, owner test=`432b40e9`.

Attempt 2 (17:29:11) — targeted 3 tests, HASH_STABLE, `2 failed, 1 passed`:
`TypeError: CandidateArtifact.__init__() missing 1 required positional argument:
'self_improve'` at `tests/test_recursive_growth_owner.py:302`. Immediately after
the run, `recursive.py` changed again (64109 B, 17:29:05), so this result is
**STALE** and is not recorded as a defect.

Attempt 3 (17:34:20–17:45:01) — 10-minute quiescence poll over the 7-file
fingerprint. `STILL_ACTIVE_AFTER_10M`: within the window, 5 of 7 files changed at
least once (`state.py`, `recursive.py`, `test_recursive_growth_prepared.py`,
`test_recursive_growth_orchestrator.py`, `test_recursive_growth_owner.py`).
Max observed quiescent window was 62 s, below the 120 s threshold.

## Fingerprints

HASH_STABLE confirmed runs (7-file set, sha256, first 8 hex):
- 17:27:42 run: `bd194b31 2ef8e1bc 0f30547e cc3e7813 e9a89766 432b40e9 d8f0d9f4`
- 17:29:11 run: `bd194b31 b8a64b98 0f30547e cc3e7813 e9a89766 432b40e9 d8f0d9f4`

Final observed state (17:45:36), no run performed on it:
- `state.py` 83785 B 17:37:33 `01d1dca5`
- `recursive.py` 67684 B 17:40:28 `b665a5f9`
- `test_recursive_growth.py` 32221 B 11:56:35 `0f30547e`
- `test_recursive_growth_prepared.py` 12059 B 17:35:22 `acb017fb`
- `test_recursive_growth_orchestrator.py` 58103 B 17:43:18 `66347a00`
- `test_recursive_growth_owner.py` 64832 B 17:42:35 `a252b3d7`
- `test_orchestrator_evaluation.py` 994 B 16:46:01 `d8f0d9f4`

All seven files are untracked (`git status --short` -> `??`). No run result is
transferable to the current state; no commit is possible (commits forbidden).

## Consequence for the evidence ladder

- Focused 5-file CPU gate: **not established** on any current hash set.
- Full non-blocked CPU suite: **not established** (previous `910 passed, 2 skipped`
  is STALE — its hash set was superseded).
- CPU two-generation gate (plan §8): **not established** on a current hash set.
- Independent read-only correctness + security review: **not obtained** (no
  stable snapshot to pin). No self-approval is possible or permitted.
- `quality_supported=false`, `security_supported=false`, `autonomy_supported=false`
  remain in force. `mechanism_supported: pending-final-gate` unchanged.
- Real-data smoke, training, GPU runs: remain blocked.

## Also observed (not a defect, not fixed here)

- `tests/test_recursive_growth_owner.py:391` — Ruff `UP012`
  (`"tokens".encode("utf-8")`), mechanical.
- `src/hagi/orchestrator/recursive.py:12` — Ruff `I001` import sorting
  (aliased `state_key_digest` import placed before `canonical_json_bytes`).
- The owner test's exact-field-set assertion is now out of sync with the runtime
  dataclass, and its `_candidate_builder` does not yet pass `self_improve` to
  `CandidateArtifact`. Both are in `tests/test_recursive_growth_owner.py`, which
  is **outside** this session's assigned write lane
  (`src/hagi/orchestrator/state.py` + `tests/test_recursive_growth_orchestrator.py`),
  so no edit was made. Fence policy also forbids editing a workspace under active
  concurrent mutation.

---

## Addendum — 18:12–18:16 (gate established, then superseded)

Strategy change that worked: a short hash-guarded retry loop
(`.omc/focused_retry_gate.sh`) instead of waiting for quiescence. The observed quiet
window was up to 62 s; a 5-file run takes ~23 s, so a run can fit inside it.

- Focused 5-file CPU gate on pin `b9cfa6a2`: `175 passed in 22.68s`, HASH_STABLE, attempt 1.
- Full non-blocked CPU suite on the same pin: `945 passed, 2 skipped, 15 warnings in
  80.99s`, HASH_STABLE, attempt 1 (`.omc/full_retry_gate.sh`).
- Ruff: `All checks passed!`; `py_compile`: rc=0. `git diff --check` still rc=2 on
  `AGENT_WORKLOG.md:1911` (outside every write lane; untouched).
- Two independent read-only reviews on the same pin both returned `SNAPSHOT: PASS`,
  `VERDICT: BLOCKED`. No High. Details in `.omc/research/evidence_ladder_b9cfa6a2.md`.

At 18:14:40 the external writer changed `tests/test_recursive_growth_owner.py` to
`8a07b5f6`, adding two tests that pin the correctness reviewer's Low finding about the
inert `_validate_persisted_f3_derivation`. The new focused run is
`2 failed, 175 passed` (HASH_STABLE) — a genuine RED state:

- `test_terminal_replay_invokes_durable_f3_reconstruction` — asserts the validator is
  called during terminal replay; it is currently never called.
- `test_persisted_f3_derivation_rejects_coherently_resigned_child_snapshot` — the
  tautological digest check inside the validator.

`src/hagi/orchestrator/recursive.py` is unchanged since 18:07:49, so the implementation
has not started. No edit made from this session: the file is outside this lane.

Consequence: the established gate is now bound to a superseded pin. Re-establishment on
a new pin is required before any of it may be cited.

---

## Addendum — 18:17–18:35 (§15.1-A adjudicated on real bytes)

### Gate state on pin `39969c7c` (recursive.py)

- Focused 5-file: `177 passed in 25.30s`, HASH_STABLE, attempt 1 (`.omc/pinned_snapshot_177.txt`).
- Full non-blocked CPU: `959 passed, 2 skipped, 15 warnings in 77.49s`, HASH_STABLE, attempt 1.
- Ruff `All checks passed!`; `py_compile` rc=0.
- §15.1-B is now genuinely closed: `_validate_persisted_f3_derivation` at
  `recursive.py:975` pins each child snapshot to `run/inputs/<name>-checkpoint-<digest>.bin`,
  recomputes the config digest, and reconstructs the candidate; it is invoked from
  `_validate_evidence` on terminal replay (`recursive.py:1307`).

### Independent verdicts on `39969c7c`

- Correctness (`9eb0f73b-5633-417`): `SNAPSHOT: PASS`, `VERDICT: PASS`, `NO FINDINGS`.
  Confirmed the F3 fix is real, uses exact `torch.equal` comparison, and that both new
  tests can fail.
- Security (`0c72c83e-561c-4a5`): `SNAPSHOT: PASS`, `VERDICT: BLOCKED`.
  - Medium `recursive.py:1521`/`1552` — STILL PRESENT (holdout-derived digest in builder
    contexts). This is §15.1-A.
  - Low `recursive.py:1580` — STILL PRESENT (`f"{type(exc).__name__}: {exc}"` persisted to
    durable `failure.json`).
  - Low `state.py:530` — FIXED (Windows durability claim now scoped to process crash).

### Medium §15.1-A adjudicated as TRUE, on real bytes

`HoldoutContract.data_manifest_sha256` is set by `real_cycle.py:839` to
`actual_manifest_sha` = `sha256_file(root / "manifest.json")` of the pinned Banking77
artifact (`real_cycle.py:750`, `:59`). That manifest is
`artifacts/datasets/banking77/57ec275d8078af65b7731c2a98be812d844a6d6b/manifest.json`,
sha256 `44d50edd…`, and its `files[]` contains a token shard for the holdout split:

```
TOK train/shard-000000.bin  99982 tokens  30f5d6acf7d2…
TOK train/shard-000001.bin  50140 tokens  c5ead7e62592…
TOK test/shard-000000.bin   42560 tokens  566855e61c31…   <-- holdout membership
split_policy "official train/test CSV rows; exact text retained"
test_rows 3080, train_rows 10003
```

`real_cycle.py:823-831` builds `holdout_parts` from `test_ids` and then passes
`actual_manifest_sha` as the data binding. So the digest reaching `ChildContext`
(`recursive.py:1521`) and `CandidateContext` (`recursive.py:1552`) is a hash over a
document that enumerates the holdout shard. A trusted-but-curious builder can attempt
membership inference against that enumeration. The finding is therefore real, not a
false positive from opaque fixture strings.

### Note on a review-authoring defect I introduced

The earlier review prompt contained a mistyped hash
(`d8f0d9f4…909499…` instead of `d8f0d9f4…409499…`). Both reviewers re-ran the hash
check independently and reported all seven matching, so no verdict was distorted; the
correction is logged in `.omc/research/evidence_ladder_b9cfa6a2.md`.

### Delegation reliability

Two consecutive waves of subagents failed with
`502 {"error":{"message":"no credentials available for provider: deepseek"}}`. A model
override to `haiku` was rejected as unknown; `local-proxy/zai/glm-5.3-flash` is available.
Per Sunk Cost the §15.1-A design analysis was therefore performed locally from read bytes
rather than by a third retry of the same mechanism.

### Current block

GO remains withheld. Zero-blocking CPU gate is established on `39969c7c`, but the
security gate is BLOCKED by one Medium (§15.1-A) and one Low (`failure.json` reason text).
`quality_supported=false`, `security_supported=false`, `autonomy_supported=false` remain
in force. No GPU, no training, no real-data quality run.
