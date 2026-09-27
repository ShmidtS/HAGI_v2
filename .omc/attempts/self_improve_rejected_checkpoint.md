# Rejected self-improvement checkpoint — attempt record

Date: 2026-09-25
Status: implementation locally verified; independent review pending

## Problem

`scripts/self_improve.py` previously called `save_checkpoint()` and returned
exit code 0 even when `SelfImproveStats.accepted_updates == 0`. A KL,
non-finite, plateau, or exception rollback could therefore leave a
`step-*.pt` that looked like persisted progress to a later `--resume`.

The loop evaluates the same generated window it uses for adaptation and has
no immutable external holdout. Even an accepted update is operational
progress only; it is not standalone model-quality evidence.

## Options considered

1. Persist nothing when no update was accepted; return exit 2. Selected:
   smallest fail-closed policy and no rejected candidate can enter a parent
   directory by accident.
2. Persist rejected weights under an attempts directory. Rejected: useful
   diagnostics, but expands the artifact surface and can later be selected
   without a proven evaluator contract.
3. Persist as before and add a rejected label. Rejected: leaves the unsafe
   default in place.

The repository already uses exit 2 for a guard failure in
`scripts/qwen_pyramid_smoke.py`; the implementation follows that local
pattern.

## TDD evidence

Before the fix, `tests/test_self_improve_cli.py` produced four failures:

- rejected run returned 0 instead of 2;
- rejected run created `step-0000000.pt`;
- accepted report had no `quality_supported`;
- report did not expose `accepted_updates`.

After the fix:

- `tests/test_self_improve_cli.py`: 5 passed;
- `tests/test_self_improve.py tests/test_self_improve_cli.py`: 34 passed;
- Ruff check: passed;
- Ruff format check: passed;
- `py_compile`: passed.

## Implemented contract

- derive `accepted_updates` directly from `SelfImproveStats`;
- increment checkpoint step only by accepted updates;
- if none were accepted: do not call `save_checkpoint`, report
  `checkpoint: null`, explain rejection on stderr, return 2;
- if at least one was accepted: persist exactly one checkpoint and return 0;
- always report `quality_supported: false` until an independent immutable
  holdout evaluator exists.

## Claim boundary

This fixes persistence policy only. It does not prove quality, universality,
or autonomous growth and does not modify the recursive F3 owner.

## External reference check

MLflow documents validation loss as the persisted-run selection signal
(https://mlflow.org/docs/latest/ml/tracking); Ray Train supports retaining
only checkpoints meeting a reported metric
(https://docs.ray.io/en/latest/train/user-guides/checkpoints.html). No
external framework was imported: the project already has a suitable
checkpoint primitive and local exit-code convention, so a thin glue fix is
smaller and clearer than adding a tracking dependency.
