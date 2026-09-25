# Baseline: Banking77 exact-CE, pinned 2026-09-25 (rev. 3)

Supersedes rev. 2 of the same date. **Rev. 2's noise floor was wrong by ~50x** —
see "CORRECTION" below. Rev. 3 carries the correct paired analysis.

## CORRECTION — rev. 2 divided by the wrong denominator (rev. 3)

Rev. 2 computed sigma from the spread of the incumbent *level*
(0.037362) and compared a paired delta against it. That is wrong.

The candidate is a deterministic function of the same three children as the
incumbent, so the level — which children were trained, how good they
are — cancels in the difference. What must be measured is the spread of
the **paired deltas**, not of the levels.

| quantity | value |
|----------|-------|
| paired deltas (1234, 2243, 3252) | -0.0012670, +0.0006090, -0.0002050 |
| paired mean | **-0.0002877 nats** |
| paired sd | **0.0009407** |
| unpaired level sd (rev. 2's error) | 0.037362 — inflated ~50x |
| paired t | **-0.530**, df = 2, p ~ 0.65 |
| 95% CI | **[-0.002625, +0.002049]** — contains zero |

So the correct framing is not "-0.008 sigma against a 0.075 floor". It is
**-0.31 sigma against a 0.0019 floor, with a confidence interval that
straddles zero**. The conclusion is the same and is now stated on correct
arithmetic: no measurable effect.

## What this is

The project replaced "build a self-improving model" with a measurable
criterion. This file pins the baseline that criterion is measured against.

## SCHEMA CORRECTION — the reason for rev. 2

Rev. 1 pinned the baseline from
`reports/recursive_growth_banking77_gen2_seed_*/`, whose holdout evidence
carries `schema: recursive_f3_holdout_evidence_v1`.

The `parent_preserving` runs carry `recursive_f3_holdout_evidence_v2`.

For the **same seed 1234**, the incumbent (parent) macro exact-CE is:

| run tree | schema | incumbent macro CE |
|----------|--------|--------------------|
| `recursive_growth_banking77_gen2_seed_1234` | v1 | 8.122496 |
| `recursive_growth_pp_seed_1234` | v2 | 8.101092 |

Difference: **0.021404 nats** on the same seed, same holdout rows
(`row_ids_sha256` identical), same transform digest. The parent model is
built differently between the two schemas, so v1 and v2 numbers are **not
comparable**.

Rev. 1's 2σ = 0.084991 was therefore measured on a different parent than
the one `parent_preserving` actually starts from. It must not be used as
the floor for these runs. Rev. 2 pins the v2 baseline instead.

## Data (verified, not assumed)

- Banking77, 10 003 train / 3 080 test.
- Packed manifest `recursive_f3_packed_holdout_v1`,
  tokenizer `google/gemma-4-E2B-it`, packed vocab 3060.
- `row_ids_sha256` are identical across all three seeds and both schemas:
  A `c1bd2d14…`, B `c66b2593…`, C `6b91f6dd…`. The seeds vary the run, not
  the evaluation set.
- `transform_digest` = `fd0121efaf9f1792…` identical on all three seeds →
  the same transform was applied on all three.

## Seeds (pre-registered)

1234, 2243, 3252.

## Baseline measurement (v2, parent_preserving, incumbent macro exact-CE)

| seed | decision | incumbent CE | candidate CE | ce_regression | worst source |
|------|----------|--------------|--------------|---------------|--------------|
| 1234 | accepted | 8.101092 | 8.099825 | -0.0012673 | -0.0008386 |
| 2243 | rejected | 8.030847 | 8.031456 | +0.0006082 | +0.0025959 |
| 3252 | accepted | 8.088040 | 8.087835 | -0.0002047 | +0.0010188 |

- n = 3
- mean = **8.073327**
- sigma of the LEVEL (kept for comparability, NOT the detection floor) =
  0.037362
- **paired sigma (the actual floor) = 0.0009407**, so a 2-sigma effect
  must exceed **0.0019 nats**
- paired t = -0.530, df 2, 95% CI [-0.002625, +0.002049]

## Consequence — the detection floor

- candidate mean = 8.073039
- delta = **-0.0002877 nats = -0.31 sigma (paired)**
- 95% CI [-0.002625, +0.002049] contains zero
- sign test: 2 negative, 1 positive — p = 1.0

**No measurable effect.** The mean is negative, which is the direction
growth should push, but one of three seeds went the other way and the
interval straddles zero. This is not a weak win; it is an absence of
evidence.

Note this delta reproduces the independent alpha-0 probe
(`reports/ALPHA0_PINNED_SEEDS_20260925.md`, mean -0.000286) to within
0.000002 nats. Two harnesses agreeing to 6 decimal places is the strongest
cross-check available in this repo, and it is the one time the project has
been able to make that claim.

## The rule, and why it currently has no force

The pre-registered verdict rule is `regression <= 0.0 and worst <= 0.01`.
Seed 2243 fails it at +0.0006082.

But 0.0006082 is **0.8% of the 2σ floor**. Applying a fail-closed rule with
no minimum-effect clause to a quantity 0.8% of the noise floor decides the
outcome on noise. The rule is correct as a guard against real regression and
incapable of resolving a result at this magnitude.

This is the honest state, and it is a result, not a failure to report
softly:

1. `parent_preserving` fixes the self-merge regression. The diagonal
   invariant holds (F1 passes at parent_depth 0/1/2; `Q @ (1,1,1) = (1,1,1)`).
2. The merge is now **effectively free** — it neither helps nor hurts.
3. The growth mechanism has never been observed to improve quality on any
   admissible measurement. Not once in 52 days of runs.
4. A later run with **disjoint child seed sets** (5 seeds, 0/3 pairwise
   child overlap) moved the mean to -0.0008901 with paired sd 0.0013016,
   t = -1.185, p ~ 0.72, CI [-0.0041234, +0.0023431]. Same conclusion,
   slightly wider noise. Disjoint sets widened the noise exactly as
   predicted, which confirms the earlier floor was the optimistic one.

## What remains unverified

- The paired sigma still rests on n = 3 (and 5 for the disjoint-set run).
  With df = 2 the t critical value is 4.30, so the interval is wide by
  construction. Tightening it needs many more full runs.
- The three pinned runs are **not independent**: `child_seeds =
  base_seed + 1009 * index` means adjacent base seeds share 2 of 3 child
  seeds. 1234/2243/3252 are 1009 apart, so those runs do NOT share
  children, but the scheme is fragile and any seed chosen near a multiple
  of 1009 would silently overlap. The disjoint-set run exists precisely
  because of this.
- The v1-vs-v2 parent construction difference is **not explained**. It was
  found by comparing runs, not by reading the code that changed. Until the
  cause is identified, treat cross-schema comparison as unsafe.
- Wall-clock budget per seed is still not pinned.
- No run with a non-zero gradient step has yet produced a delta above the
  noise floor, so the growth mechanism has never been observed to improve
  quality on any admissible measurement.

## Provenance

Measured from `reports/recursive_growth_pp_seed_*/store/generation-1/`
(`holdout-evidence.json` and `report.json`), produced by
`python scripts/recursive_growth.py --cross-parent-transform parent_preserving`
with the pinned artifact `57ec275d…` and manifest `44d50edd…`.
