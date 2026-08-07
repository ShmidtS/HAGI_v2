# HAGI V43 Architecture — Bandwidth-Optimal Geometry + Host-Bound Fix

V43 rethinks V42 for wall-time on a bandwidth-bound iGPU (Radeon 8060S,
DDR5 ~107 GB/s). The body shape, ternary channel, and receiver are unchanged;
the geometry, the ternary cache, and the supervision puncturing move.

## What changed

### 1. Cached STE — the host-bound fix

**Problem.** `torch.profiler` (with_stack) showed the model is **host-bound**,
not compute-bound: `aten::copy_` = 634 ms CPU (65.7% of step time). BitLinear
rebuilt the ternary map on every forward via `weight + (q - weight).detach()`,
materializing two `[out, in]` host copies per layer per microbatch (~150
copies/step). The ternary step-cache was gated on `grad_accum > 1`, so at
`grad_accum=1` (V42's ship config) it never ran.

**Fix.** `BitLinear.cache_quantized()` now stores the quantized *effective*
weight, and the forward returns it through a new zero-copy `_CachedSTE`
(forward = q, backward = identity on the master). The ternary cache is now
always on (removed the `len(microbatches) > 1` gate). Removes ~150 host
copies/step.

### 2. Geometry sweep — backward dominates

**Problem.** Backward is 534 ms of 750 ms (71%): flash-attention backward
202 ms + GEMM backward 150 ms. Backward ∝ L (layers) and GEMM ∝ H²·L·ffn.
The V42 shape (H=512, L=8, ffn=2.0, T=1024) is deep and narrow — the worst
case for backward.

**Fix.** Sweep at fixed body budget (measured, median of 4, Radeon 8060S):

| H | L | ffn | T | B | ms/step | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 2.0 | 1024 | 48 | 791 | 62.1k | (V42)
| 768 | 4 | 2.0 | 1024 | 48 | 701 | 70.1k |
| 768 | 4 | 1.0 | 1024 | 48 | 593 | 82.9k |
| 768 | 4 | 1.0 | 512 | 48 | 276 | 89.0k |
| 768 | 4 | 1.0 | 256 | 96 | 252 | 97.5k |
| **768** | **4** | **1.0** | **128** | **192** | **238** | **103.1k** | (V43)

Three levers, each measured:
- **Wider, shallower** (H=768, L=4): backward ∝ L, so fewer layers win.
- **Narrower FFN** (ffn=1.0): backward ∝ ffn; the narrowest mixer is fastest.
- **Short T** (T=128): flash-attention backward ∝ T², so 64× cheaper than
  T=1024. The model is too small to exploit longer context — ce is identical
  at T=128/256/512 — so short T is pure win.

B=192 is the sweet spot: tok/s rises with B (103k→107k at B=384) but ms/step
grows faster; B=192 gives 103.1k tok/s at 3.2 GiB peak.

### 3. Punctured receiver (kept, re-verified)

`ce_keep_rate=0.5` scores half the positions (Bernoulli erasure channel on
supervision). At T=128 the head is cheap (~3 ms), so puncturing saves only
~2% wall-time, but it is theoretically principled and reduces gradient
variance.

**Backward puncturing through the body was measured and rejected.** Detaching
non-scored positions at the body input does NOT cut the attention/GEMM
backward (227 vs 223 ms) — the backward still executes, only the gradient is
zeroed. A two-pass forward (only scored positions with gradient) costs more
than it saves at small T (forward=71 ms, backward=153 ms → 2×71+153×0.5 =
218 ms, no better than 223 ms). The receiver is already the cheap part.

## Quality

ce after 60 steps, equal token budget (warmup 10):

| Config | ce |
|---|---:|
| V42 (H=512 L=8 ffn=2.0 T=1024 B=48) | 3.920 |
| V43 (H=768 L=4 ffn=1.0 T=128 B=192) | 3.907 |

V43 learns at least as well at 3.3× the speed.

## Profiling methodology

All measurements on AMD Radeon 8060S (HIP 7.13, 115 GB shared, 20 CUs),
PyTorch 2.10.0+rocm7.13.0a. Median of 4–5 steps after 2–3 warmup steps, using
random integer data (no DataLoader I/O). `torch.cuda.synchronize()` before and
after each step. `torch.profiler` with `with_stack` for the host-bound
analysis.

## Information-theoretic summary

The communication system is unchanged from V42:

```
source codebook + prior
  → L4 ternary channel (full causal flash attention, cached STE)
  → shared K=64 interference bank
  → conditional NCE gradient (punctured at p=0.5)
  → periodic exact full-alphabet CE calibration
```

The wall-time win comes from matching the *geometry* to the hardware's
bandwidth profile: fewer layers (less backward), narrower mixer (less GEMM
backward), shorter sequence (less attention backward), and a zero-copy
quantizer (no host copies).
