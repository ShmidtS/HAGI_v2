# Legacy terminal-receipt in the unified accepted commit

## Snapshot (reproduced 2026-09-25)

- `src/hagi/orchestrator/state.py`: `01d1dca563ab93714b4d1166839bf2f94de8162ff8515466fa0c6e7a5057069c`
- `src/hagi/orchestrator/recursive.py`: `3e458d43c2d00ac0ea0a49fdc60999aac0a95c74679766d015e392901934dac0`
- `tests/test_recursive_growth_prepared.py`: `9258f399b5792a1817afcd904a82cd73f951345acc48748223710a91c4d5b7c1`
- `tests/test_recursive_growth_orchestrator.py`: `66347a00671a70940f6f0509b52b88fe3ede4b5ed806fba303459a190e27dbfe`
- `tests/test_recursive_growth_owner.py`: `489efdc96d8103f015a66ca7b93e5d050105b6c1081e16452986dab6b42fe3bc`

## Reproduction (independent of the review agent)

Stage a bound accepted decision with `_prepared_accepted`, drop a legacy
`terminal-receipt.json` into the run directory, then call
`store.commit_accepted_terminal(old, replacement, decision, owner_id="owner-a")`:

```
RECEIPT_COMMIT=ACCEPTED
REPORT_EXISTS=true
PARENT_GENERATION=g1
HASH_GUARD=UNCHANGED
```

## Root cause

`_accepted_payload_preflight` (state.py) checks `failure.json` but never checks
the legacy `terminal-receipt.json`, so the commit boundary is the only terminal
path that accepts a second, unsupported terminal marker. Every other path
rejects it as a conflict:

- `publish_terminal`: `legacy terminal receipt is unsupported`
- `recover`: `legacy terminal receipt is unsupported`
- `publish_failure`: `cannot publish failure after terminal marker`
- `mark_prepared`: `cannot prepare after terminal marker`

## Why it is a fail-closed gap, not a cosmetic mismatch

The commit point replaces `current-parent.json` and materializes `report.json`
while a foreign, unsupported terminal marker coexists in the run. The
transaction therefore reaches a terminal state that its own recovery contract
refuses to interpret: `recover` raises on the receipt before it can validate the
committed parent. A committed parent plus a permanently unrecoverable run is
worse than refusing the commit.

## Resolution (RED → GREEN)

After a 60-second quiescent window with no active `pytest`/`py_compile` and no
hash changes, the following pinned snapshot was tested:

- `src/hagi/orchestrator/state.py`:
  `01d1dca563ab93714b4d1166839bf2f94de8162ff8515466fa0c6e7a5057069c`
- `tests/test_recursive_growth_orchestrator.py`:
  `66347a00671a70940f6f0509b52b88fe3ede4b5ed806fba303459a190e27dbfe`

Regression added: `test_unified_commit_rejects_legacy_terminal_receipt`.
RED: `DID NOT RAISE ValueError` with `HASH_GUARD=UNCHANGED`.

Minimal production fix: `_accepted_payload_preflight` now rejects an existing
legacy `terminal-receipt.json` beside its existing `failure.json` guard, before
the first durable write. No commit ordering or persistence format changed.

GREEN: `1 passed in 0.16s`, then the full focused suite
`tests/test_recursive_growth.py`, `test_recursive_growth_prepared.py`,
`test_recursive_growth_orchestrator.py`, and `test_recursive_growth_owner.py`
returned **173 passed in 17.50s** with `HASH_GUARD=UNCHANGED`. Ruff,
`py_compile`, and targeted `git diff --check` pass. The final pinned hashes for
independent review are `state.py=13c53a11…` and
`test_recursive_growth_orchestrator.py=3663795b…`.

## Contract for the fix

Reject the commit before the first durable write, next to the existing
`failure.json` guard, reusing the same wording as the other terminal paths
(`legacy terminal receipt is unsupported`). Regression: an accepted PREPARED run
carrying a legacy receipt must keep the incumbent parent, stay `prepared`, and
create no `report.json`. `security_supported` remains `false`: the threat model
is trusted-local single-user, and same-UID filesystem adversaries are out of
scope.
