# Holdout resolution — the 0.01 budget is reachable, data already pinned

Route: skill `debugging` (root cause of a stuck lane) + `code-review` (no
self-approval of a foreign measurement).

## The stuck fact being explained

`.omc/attempts/trust_region_2026-09-25.md` recorded:

- paired SE at 256 scored tokens/source = **0.0251 nats**
- gate budget = 0.01 nats per source = **0.40 sigma**
- required for a usable instrument: SE <= budget/3 = 0.0033 nats
- naive scaling said "needs ~14 515 tokens/source, i.e. 58x more rows,
  and the pinned split may not have them; first check whether the data is
  available, otherwise raise the budget. Do not change both measures at once."

## Measurement (read-only, no training, no gate change)

- `.omc/runs/alpha-zero-probe-20260925c/seed-416115/banking77-holdout.json`
  pins **exactly 256 tokens per source** (A/B/C), 768 total, schema
  `recursive_f3_packed_holdout_v1`, tokenizer `google/gemma-4-E2B-it`,
  vocab_size 3060. The 256 is a *choice in the contract*, not a data limit.
- `data/artifacts/banking77-20260925/manifest.json` pins the test shard
  `test/shard-000000.bin` with **42 560 tokens**, sha256
  `566855e6...7e01f5f4928daed`, already hash-bound in the artifact.

## Computation (1/sqrt(n) for a mean)

    target SE      = 0.01/3            = 0.0033 nats
    required tokens = 256*(0.0251/0.0033)^2 = 14 515 per source
    available        = 42 560            -> 2.9x margin
    SE at full shard = 0.0251*sqrt(256/42560) = 0.0019 nats
    budget 0.01     = 5.1 sigma  (was 0.40 sigma)

## Conclusion

- The preregistered question is answered: **the data is available**. The
  "58x more rows" figure was the consequence of pinning 256, not of the
  dataset. No new data, no external download, no budget relaxation.
- The correct next move is to raise the pinned holdout span from 256 to the
  full test shard, and keep the 0.01 budget unchanged. That is one measure
  changed, which is what the preregistration required.
- Expected consequence: a materially higher true acceptance rate for the
  parent_preserving lift, whose effect (0.029-0.040 nats) is already
  1.3-1.7 SE at n=256 and would become ~7-10 SE at n=42 560. This does
  NOT by itself prove quality or autonomy; it makes the measurement
  capable of supporting such a claim.
- NOT measured here: no re-run of the gate, no acceptance-rate measurement,
  no model training. Status unchanged: `quality_claim_supported=false`,
  `production_promotion=false`.
