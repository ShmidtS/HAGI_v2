# HAGI V41 Architecture - Interleaved Conditional Receiver

V41 keeps the V40 source coder and ternary channel. It changes only the finite
sample seen by the receiver.

## Communication contract

```
source codebook + prior
  -> L4 ternary channel (W256 local / full relay / W256 / full relay)
  -> shared K=64 interference bank
       32 deterministic in-batch target symbols
       32 source-prior draws
  -> conditional NCE gradient
  -> periodic exact full-alphabet CE calibration
```

Targets in a packed language batch are samples from the same source marginal as
the unigram proposal. Reusing evenly spaced target positions as negative
interferers therefore preserves the source-matched interpretation while adding
batch-specific hard symbols. It does not materialize `[N,K,H]`: all 64 negative
correlations still use one GEMM.

This reuses the old HAGI interleaver idea at the statistically correct boundary.
Earlier versions transformed hidden channels and added inverse transforms. V41
interleaves receiver symbols only, as variance reduction and alphabet coverage;
the main channel remains unchanged.

## Evidence

Controlled packed-corpus runs used identical model/data seed and an independent
exact-CE RNG stream, so proposal sampling cannot change calibration rows.

| Receiver | Steps | Median ms | body tok/s | exact CE at last checkpoint |
|---|---:|---:|---:|---:|
| prior-only K64 | 100 | 1843 | 50.0k | 7.5412 (step 90) |
| **50% in-batch + 50% prior K64** | **100** | **1848** | **49.9k** | **7.1775 (step 90)** |

A separate seed-888 50-step run showed the same direction: exact CE at step 45
was 7.4096 for 50% in-batch versus 7.5158 for prior-only. The cost is about
0.3-0.4% wall-time.

Rejected experiments:

- K256: +1.8% wall-time and no exact-CE improvement over K64 after 100 steps.
- Prior/uniform proposal mixture: +0.8% wall-time with no stable exact-CE gain.
- L2 looped twice: same wall-time and nearly identical early exact CE as L4, but
  half the independent channel parameters. It remains a compression profile,
  not the maximum-learning ship model.

## Fixed invariants

- Full supervision (`ce_keep_rate=1`).
- Exact CE is the coding-cost SSOT; local NCE is never reported as perplexity.
- Tied full-rank receiver and unigram source prior.
- Fixed-rate multimodal bridge, water filling off, text-only self-sufficiency.
- Checkpoint format 12; train from scratch.
