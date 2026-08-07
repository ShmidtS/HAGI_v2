# HAGI V42 Architecture — Full Causal Channel + Punctured Receiver

V42 rethinks V41 for wall-time on a bandwidth-bound iGPU (Radeon 8060S,
DDR5 ~107 GB/s). The body shape, ternary channel, and receiver are unchanged;
only the attention pattern, sequence geometry, and supervision puncturing move.

## What changed

### 1. Full causal attention on every layer (W=0)

**Problem.** V41 interleaved full-attention relay layers (0, 2) with windowed
layers (1, 3) using `compressed_history_attention`. The windowed path constructs
a custom `[1, 1, W, W+T/stride]` mask per chunk inside a Python loop (4 SDPA
calls per layer). On this iGPU, the mask construction and kernel-launch overhead
of 4 small SDPA calls exceeded the cost of one full causal flash-attention call.

**Fix.** Set `window=0`, making every layer use `F.scaled_dot_product_attention`
with `is_causal=True` — a single optimized flash-attention kernel per layer.

**Measurement** (B=30, T=1024, 3 microbatches, median of 5):

| Attention | ms/step | tok/s |
|---|---:|---:|
| V41 mixed (W=256, 2 full + 2 windowed+history) | 1845 | 50.0k |
| All full (W=0) | 1748 | 52.7k |

The flash kernel on ROCm (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`) is fast
enough that O(T²) full attention beats chunked O(T·W) on this hardware.

### 2. Sequence length T=512 (halves attention FLOPs)

**Problem.** T=1024 doubles attention FLOPs vs T=512 (O(T²)). At 4 layers and
71M params, the channel cannot meaningfully exploit 1K-token context — the body
is too shallow for long-range dependencies to survive through the residual
stream.

**Fix.** T=512 with B=48 to keep total tokens/step high (73 728 vs V41's 92 160).

**Measurement** (W=0, median of 5):

| Config | ms/step | tok/s | tokens/step |
|---|---:|---:|---:|
| T=1024, B=30, accum=3 | 1748 | 52.7k | 92 160 |
| T=512, B=48, accum=3 | 1288 | 57.2k | 73 728 |

Attention FLOPs drop 4× (T²→T²/4); GEMM FLOPs unchanged (N=B·T constant per
microbatch up to rounding). The 5% tok/s increase comes from better batch
geometry (fewer sequence positions per token → less attention overhead in the
fixed-cost per-layer dispatch).

### 3. Punctured CE (ce_keep_rate=0.5)

**Rationale.** The receiver (sampled softmax K=64) costs only ~5 ms/microbatch
forward, so puncturing saves <3% wall-time. But it is theoretically principled
(an erasure channel on supervision — the body still processes every symbol) and
reduces gradient variance slightly.

**Measurement**: 1288 ms (keep=0.5) vs 1323 ms (keep=1.0) → 2.7% faster.

### 4. torch.compile — net negative

**Finding.** `torch.compile(mode="default")` adds ~250 ms/step on this ROCm
build due to CUDAGraph memory management overhead. The fused pointwise kernels
save ~20 ms but the graph bookkeeping costs ~270 ms. `compile_model` stays
false in the ship config.

Required fixes were still applied (RoPE cache key no longer uses `.item()`
during compilation, `cudagraph_mark_step_begin()` per microbatch) so that
compile can be enabled on future ROCm builds where CUDAGraph overhead is lower.

## Profiling methodology

All measurements on AMD Radeon 8060S (HIP 7.13, 115 GB shared, 20 CUs),
PyTorch 2.10.0+rocm7.13.0a. Median of 5–7 steps after 2–3 warmup steps, using
random integer data (no DataLoader I/O). `torch.cuda.synchronize()` before and
after each step.

**Step time breakdown** (V41 baseline, per microbatch):

| Component | Time | % |
|---|---:|---:|
| Body forward (4 layers) | 467 ms | 76% |
| Body backward + head f+b | 148 ms | 24% |
| Optimizer step | 8 ms | — |
| **Total per microbatch** | **615 ms** | |
| **× 3 microbatches + optimizer** | **1845 ms** | |

The body forward dominates. It is **compute-bound** (not bandwidth-bound) at
H=1152: each GEMM has arithmetic intensity ~1130 FLOP/byte, well above the
roofline threshold of ~100 FLOP/byte for this GPU (7.2 TFLOPS / 70 GB/s).

This means FLOPs reduction (shorter T, fewer attention score computations) is
the right optimization axis, not activation compression.

## Batch-size / accumulation sweep

tok/s is constant (~57k) across all T=512 configurations — confirming the GPU is
compute-bound. The B/accum choice trades tokens-per-step vs ms-per-step:

| Config | ms/step | tok/s | tokens/step | eff. batch |
|---|---:|---:|---:|---:|
| B=72, accum=1 | 643 | 57.3k | 36 864 | 72 |
| B=96, accum=1 | 861 | 57.1k | 49 152 | 96 |
| B=48, accum=2 | 860 | 57.2k | 49 152 | 96 |
| B=36, accum=3 | 974 | 56.8k | 55 296 | 108 |
| B=48, accum=3 (**ship**) | **1288** | **57.2k** | **73 728** | **144** |

B=48×3 is shipped: it maximizes tokens-per-step at acceptable latency, giving
the largest effective batch (144) for gradient quality.

## Information-theoretic summary

The communication system is unchanged from V41:

```
source codebook + prior
  → L4 ternary channel (full causal flash attention on every layer)
  → shared K=64 interference bank
  → conditional NCE gradient (punctured at p=0.5)
  → periodic exact full-alphabet CE calibration
```

The two changes (full attention + shorter T) optimize the **channel** for the
hardware's compute profile. The **receiver** changes (puncturing) are minor:
the sampled softmax head is not the bottleneck at K=64.

The KL gap (exact_ce − nce ≈ 3 nats) remains unchanged — it is a property of
the K=64 local partition, not of the attention pattern. Closing it requires a
larger K (rejected: +1.8% wall-time at K=256 with no CE improvement) or a
different receiver architecture.

## Fixed invariants (unchanged from V41)

- Tied full-rank receiver and unigram source prior.
- Ternary (b1.58) body weights with OFDM-coherence step cache.
- Fixed-rate multimodal bridge (disabled), text-only self-sufficiency.
- Exact CE is the coding-cost SSOT; local NCE is never reported as perplexity.
- Checkpoint format 12; train from scratch.
