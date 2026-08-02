# HAGI V35 Architecture

V35 is the **bandwidth-response release**: it accepts that the Radeon 8060S
iGPU (107 GB/s DDR5) is strictly memory-bandwidth-bound and restructures the
model to minimize *memory traffic per step* rather than FLOPs. It also fixes a
silent bug in the V34 Explorative Modeling gate that made XM never explore.

## The measured bandwidth ceiling (V35 re-measurement)

| Metric | V34 (L=13) | V35 (L=8) | Δ |
|---|---|---|---|
| Step time | 2.63 s | 1.81 s | **-31%** |
| tok/s | 11,647 | **17,018** | **+46%** |
| VRAM | 0.76 GB | 0.95 GB | free |
| Body params | 109M | 67M | -38% |

Re-confirmed on V35 shape:
- **B=30→60→90 linear** (15910→16438→16465 tok/s): tok/s plateaus at B≥60,
  memory bandwidth is the wall.
- **grad_accum 3×30 vs single 90**: identical (~16.8k vs ~16.65k tok/s).
- **torch.compile**: +4% (2486 vs 2584 ms).
- **grad_ckpt**: −19% (recompute burns bandwidth).
- **fp8/int8 matmul**: unsupported on this ROCm build (`addmm_cuda not
  implemented`), so activation quantization is off the table.

## Why L=8, not more depth or width

The body's activation traffic is ~per-layer: fewer layers = less data moved per
step. L=13→8 gives +46% throughput. The tempting alternative, a *wider* H at
the same depth, measured 2× slower (H=1536 at L=8 → 8354 tok/s vs 15920 at
H=1152): under-filled iGPU GEMMs, and more activation bytes per token. On this
part, depth is the cheap knob; width is expensive.

Capacity is traded for tokens: V35 is 67M body vs V34's 109M. The run's ce
trajectory vs the V34 log shows the trade. The correct response to a higher ce
plateau is **not** more depth (bandwidth-bound) but more training tokens at the
faster rate — which V35 provides 46% more of per hour.

## Changes vs V34

1. **L=13→8, H=1152 fixed** (`v35_1b.yaml`) — +46% tok/s.
2. **batch 30→90 via grad_accum 3×30** — +3.5% (launch amortization at the
   bandwidth wall).
3. **`head.ce_save_logits: true`** — keeps the tied `[V,H]` codebook full-rank
   (V31 rejected rank-r factoring at a +1.42 nats KL floor) while caching chunked
   logits in VRAM instead of recomputing `[N,V]` in backward. −12% head time.
   VRAM is free here (0.95 GB of ~115 GB).
4. **`use_muon: false`** — Newton-Schulz measured +7% step time for no quality
   benefit on the ternary body (the quantizer reads only the sign pattern;
   AdamW `body_lr_scale` rebalances gradient scales at a fraction of NS cost).
5. **XM gate fix** — V34's entropy gate compared posterior entropy against
   `0.9 * ln V = 9.36`, but the unigram prior caps posterior entropy at the
   unigram entropy (~8.06 nats), so *no position could ever pass*: XM silently
   never explored and paid its forward cost for zero benefit. Now:
   - `entropy_gate` is measured against the *achievable* maximum (unigram
     entropy when a prior is loaded, else `ln V`).
   - `explore_fraction` (new) caps the explored set to the top-X% of ambiguous
     positions within the gated set, bounding the K-candidate head passes that
     cost +32% on this bandwidth-bound part.
   - `mode_std` raised 0.05→0.1 so candidates actually diverge.
   - Fixed + tested (`tests/test_head.py::TestXM`, 3 tests).

## Explorative Modeling (XM) — still opt-in, now correct

The V34 doc claimed XM "doesn't help" because mode-codes were too similar. The
real reason was the gate bug: it never explored at all. With the fix, XM
explores (mode_codes receive gradient, verified), but the K candidate head
passes cost **+32% step time** on this bandwidth-bound part. On a compute-bound
GPU where forwards are cheap this is the right scaling lever; on the 8060S it is
a quality-vs-throughput trade, so `xm.enabled: false` remains the shipped
default and the quality study is a config flip away.

## Parameters (analytic)

H=1152 L=8, 18q/3kv × 64, ffn=1536 (exp 1.33): body=67.3M embed=37.7M
total=105M. Ternary rate 1.585 bits/weight → body 0.013 GB packed.

Checkpoint format unchanged (10). V35 trains from scratch.
