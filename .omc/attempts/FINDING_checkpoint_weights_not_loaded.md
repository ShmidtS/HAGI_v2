# FINDING — past SE / alpha=0 / M2 numbers were measured on RANDOM weights

Route: skill `code-review` (no self-approval) + `debugging` (root cause of
an unusable measurement). Discovered while reviewing the delegated
`external_eval` module, which called `load_state_dict(strict=True)` explicitly
and thereby exposed that the surrounding repo does not.

## Claim under audit

`.omc/paired_se_probe.py` measured the paired standard error of the
parent-vs-candidate exact CE difference (SE = 0.0251 nats, 256 tokens/source)
and the `parent_preserving` lift deltas (mean macro +0.0176 -> -0.0180,
worst-source +0.0471 -> +0.0091, 9/9 same-sign).

## Root cause, verified by execution

`src/hagi/model/merge.py:1660 build_model_from_payload` builds the model
class from the config but never loads `state` into it:

    if cfg.merge.enabled:
        return MergedHAGI(cfg, ...).to(device)      # weights discarded
    return HAGI(cfg).to(device)                      # weights discarded

Only the recursive branch (`RecursiveF3HAGI.from_state_dict`) actually loads
weights. Verified on real artifacts:

    parent checkpoint .omc/runs/alpha-zero-probe-20260925c/seed-416115/parent/step-0000000.pt
      is_recursive_state: False
      encoder.embedding.weight loaded correctly: False   <-- random init

Both `.omc/paired_se_probe.py:22` and
`src/hagi/orchestrator/real_cycle.py:459` (`_score_checkpoint_bytes`) call
`build_model_from_payload` without a subsequent `load_state_dict`. So the
scored model is a fresh random initialisation, not the checkpoint.

Second, independent defect found in the same artifacts: the parent-guard path
now raises on its own data.

    build_model_from_payload(candidate state) ->
      ValueError: recursive state_dict missing provenance:
      ['recursive_f3_cross_parent_transform']

The alpha=0 candidate cannot be rebuilt at all under the current
`from_state_dict` contract, so the alpha=0 lane is not re-runnable as written.

## Consequence for the research record

- The SE number 0.0251 and the 9/9 lift deltas were measured between two
  random-initialised models plus whatever the config changes. They are NOT a
  valid measurement of the lift's effect on a trained parent. Treat them as
  withdrawn pending re-measurement, not as refuted.
- The conclusion "the 0.01 budget is below instrument resolution" was derived
  from that same probe, so it is also provisional. It must be re-derived
  after the loader is fixed.
- Nothing about the *direction* of the earlier work is invalidated: the
  parent_preserving invariant (orthogonal, det=1, fixes the all-ones vector,
  duplicated triple preserved) is algebraic and was verified by
  `.omc/verify_lift_invariant.py`, which does not go through this loader.

## The fix (small, do it before any further measurement)

In `build_model_from_payload`, load the state after construction:

    model = HAGI(cfg).to(device)
    model.load_state_dict(state, strict=True)
    return model

and likewise for the `MergedHAGI` branch. `strict=True` is what makes a
silently-unloaded checkpoint impossible. Guard: a caller that passes weights
it expects to be ignored should have no such path at all.

Risk: a checkpoint whose keys do not match the class will now fail loudly
instead of scoring a random model. That is the intended direction, but it may
surface latent mismatches in existing stored artifacts. Those are real
defects, not regressions to be silenced.

## Not claimed

No training, no gate re-run, no quality or autonomy claim.
`quality_claim_supported=false`, `production_promotion=false`.
