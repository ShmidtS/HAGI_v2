# HAGI V32 Architecture

HAGI V32 scales V31 to a healthy regime. The channel model is unchanged — every
decision below is a *sizing* or *enabling* fix with a measured reason. Nothing
structural about the codec was touched, because V31's structure was measured to
be right; what was wrong was its configuration.

## What changed from V31

### 1. Vocabulary compaction 262144 -> 131072 (the main fix)

Only 176,512 of the 262,144 Gemma-4 ids ever occur in this corpus, and the top
131,072 carry 99.93% of all token mass. `scripts/compact_vocab.py` rebuilds the
`.bin` streams against a dense id map (dropped ids -> UNK, never deleted, so
document lengths do not shift), and halves the codebook from `262144*H` to
`131072*H`.

The effect that matters is the **body share of total parameters**:

```
V31:  body=110.8M  embed=268.4M  total=379.2M   body share 29.2%
V32:  body=688.0M  embed=151.0M  total=839.0M   body share 82.0%
```

The V31 config repeated V30's pathology: a body too small relative to the
codebook to learn language. The V30 documentation records a 43%-embedding config
plateauing at ce~5 then diverging; V31 at 29% body was still under-weighting the
body. Halving the codebook is what lets the same budget buy a 6x larger body.

### 2. MoE enabled (E=8, top_k=2)

V31 shipped dense with `moe.enabled: false`. MoE is the only lever that grows
total capacity without growing per-token compute, which makes it the scaling
mechanism. E=8/top_k=2 gives 688M total body against 306M active body.

Load balance is the V31 bias controller: `b_e <- b_e + gamma*sign(target - load_e)`,
updated once per optimizer step after backward, affecting selection only never
combining weights. There is no auxiliary load-balance loss competing with CE.

### 3. Sliding window 256 -> 1024, seq_len 512 -> 1024

V31's window was *shorter* than the sequence: at seq_len=512, W=256 meant every
position could attend to at most half the training window. At W=seq_len the
window stops clipping long-range dependence inside the window; full-attention
relays every 4 layers carry global structure. Long-context generation beyond W
is still bounded, which is the intended finite-state-channel behaviour.

### 4. Training budget 0.41B -> 1.77B tokens

V31 ran 200k steps at 2048 tokens/step = 0.41B tokens against a ~3.2B-token
corpus — less than one pass over the larger sources. V32 runs 108k steps at
16384 tokens/step.

The budget is set by measured throughput, not by desire: this ROCm build has no
flash-attention (only the MATH SDPA backend, ~13.8 ms/layer at T=1024), which
caps the model near 3000 tokens/s regardless of batch or sequence length. 108k
steps at 16384 tokens/step ≈ 7 days of continuous training. The 4x budget
increase is real, and it is what a from-scratch model needs against this corpus.

## What stayed, with the reason

- **Full-rank tied codebook.** A rank-r factored head leaves ~0.5 nats/token of
  irreducible KL at r=128 (measured in V31). The codebook is paid once and tied
  to the receiver.
- **Ternary quantizer as the rate constraint.** `{-s,0,+s}` per output channel,
  `s = absmean(W, dim=1)`, identity STE. log2(3) = 1.585 bits/weight. The
  per-row absmean cancels uniform ||W|| drift, which is why no spectral cap is
  needed under Muon.
- **QK-norm.** Bounded logit range under a matrix-sign optimizer; the V30
  divergence fix.
- **Muon / AdamW split.** Muon (orthogonalized update, isotropic on a ternary
  master) for 2D channel weights; AdamW for codebooks, gains, routers.
- **Unigram prior.** Zero-order source code counted before training. V32 reads
  the compacted counts `data/unigram.compact.npy`; the dry-run starts at ce~8.04
  against the compact-vocab unigram entropy 8.04.
- **Chunked cross-entropy.** Never materializes an `[N,V]` logit tensor; with
  V=131072 it is cheaper than V31 and the head is faster.
- **Packed corpus, block-diagonal doc mask, proportional interleaving.**
  `data/*.compact.bin` streams; `dataset_path` prefers the compact stream when
  present.

## New: compact-vocab id mapping

`src/hagi/data/vocab_map.py` bridges the tokenizer's old id space (262144) and
the model's compact id space (131072). Training reads the compact `.bin` streams
directly and never sees a dropped id; inference (`scripts/infer.py`) maps
tokenizer ids through `data/vocab_map.npz` (`old_to_new` / `new_to_old`) before
the forward pass and back after decoding. Dropped ids fall back to UNK (id 3,
always reserved).

## Config

`configs/v32_1b.yaml` is the only shipped configuration: H=1152, L=13,
18q/3kv x 64, ffn=3072, window=1024 with full relays every 4 layers, MoE E=8
top_k=2 on layers [1,3,5,7,9,11], tied codebook at V=131072, conv_kernel=4,
ternary on, unigram prior on `data/unigram.compact.npy`, ce_chunk_rows=4096,
batch 4 x accum 4 x seq 1024 = 16384 tokens/step, 108k steps = 1.77B tokens.
Muon lr 0.02, AdamW lr 3e-4. Checkpoint format 9.

## Observables (unchanged from V31)

- `ce` in nats/token against the compact-vocab unigram entropy 8.04.
- `qk_gain` — mean QK-norm gain; rising means the correlator heads for
  saturation.
- `moe/entropy_ratio` — usable fraction of expert channels; 1.0 is perfect
  balance (measured 1.000 at step 0).
- `logit_scale` — receiver gain; starts at `1/sqrt(H)` and should rise as the
  conditional part of the code becomes informative.

## Performance optimizations (V32.1, measured on this ROCm build)

This ROCm build has **no flash-attention** (only the MATH SDPA backend), no
Triton, and no torch.compile. Initial V32 throughput was ~2900 tokens/s. Four
optimizations, all measured, raised it to ~4900 tokens/s (+71%):

1. **bf16 chunked cross-entropy.** The head's matmul/softmax now run in the
   parameter dtype (bf16 under bf16 training) with an online-stable logsumexp
   (shift by max). fp32 measured 2.9x slower for a 0.23% accuracy cost, which is
   below the noise of one optimizer step. fp64 inputs (gradient checks) keep
   fp64. `tests/test_head.py` asserts exactness.
2. **grad_checkpointing off.** With 96 GB VRAM the recompute costs time, not
   memory (peak is ~19 GB). Turning it off was +23%.
3. **MoE n_shared=0.** The always-on shared expert was 10.6M params of constant
   compute; with top_k=2 every token is already routed to two specialized
   experts. Removing it was +9%.
4. **Physical window truncation (training only).** A windowed layer slices K/V
   to the last `window` keys before SDPA, making it O(T*W) instead of O(T^2).
   With window=512 at T=1024 that is +6%. It is confined to `self.training`:
   eval must stay bit-exact against incremental decoding (a truncated prefill
   and a small cache disagree on the earliest positions, asserted by
   `test_incremental_matches_full`).
5. **MoE top_k=1 (Switch-style routing).** Half the routed FFN FLOPs of top_k=2;
   load balance is still measured at entropy_ratio=1.000 (the bias controller is
   selection-only, so top-1 routing stays balanced). +11%.
6. **Batch 16, accum 1.** No grad-accum kernel relaunch overhead. +2%.

CUDA graphs measured 1.00x (matmul-bound, not launch-bound) and fp16 == bf16 on
this GPU, so neither helps further.

The config runs `batch 16 x seq 1024 = 16384 tokens/step` at ~3.3s/step and the
budget is 196k steps = 3.21B tokens over ~7 days.

### Why not a bottleneck head

Head matmul was 54% of step time at V=131072. A bottleneck `H -> 288 -> V`
head measured 2.7x faster, but V31's factored-head measurement showed a
~0.5-0.9 nats loss at these ranks, which is ~1.6x perplexity — too expensive to
trade for speed. The vocab compaction to 131072 already halved the head; a
further compaction to 65536 (98.3% mass) is the clean next lever if needed.

## V32.3 — second compaction 65536 -> 32768

Measured exact step-cost split on the V32.2 config (batch 12 x 1024, V=65536,
ROCm, no flash-attn):

```
fwd+bwd  2.03s   = head backward 1.05s (51%)
                   + blocks backward 0.69s
                   + fwd blocks 0.41s
Muon NS   0.70s  (204 matrices, all > 1e5 params)
AdamW     0.13s
```

The head dominates because the chunked CE's backward materializes **four**
`[N,V,H]` projections (forward 1 + backward recompute + `g@W` + `g^T h`) plus an
fp32 softmax pass over all V. Every one scales linearly in V.

**Logit sparsity was measured and rejected.** At init the logits std is only
0.02 (receiver gain 0.029), so all 65536 logits sit within `max-1.0`; a top-K
truncation would save nothing. The cheap lever is vocabulary size itself.

Compaction to **V=32768** (96.2% mass, unk 1-8%) halves all four projections and
the softmax:

```
V32.2:  2.90s/step   4280 tok/s   head 75.5M params
V32.3:  2.41s/step   5106 tok/s   head 37.7M params   (+19%)
```

Head params drop to 37.7M; body share rises to ~94%. Budget re-scaled to
212k steps = 2.6B tokens.

### Map consistency pitfall (measured)

`compact_vocab.py`'s recompaction path builds the 65536->32768 map from the
65536-space `.compact.bin`, which **loses the original 262144-space identity**:
the transitive map disagrees with a direct full map by ±1-4 ids (frequency-sort
tie-break). For inference the tokenizer emits old-space ids, so `vocab_map.npz`
must be the **full 262144->32768 map** and the binaries must be rebuilt from raw
`.bin` through it (`scripts/rebuild_compact2.py`). Verified roundtrip: survivors
32768/32768 exact, dropped tokens -> UNK(3).

### Stability note

The V=65536 run died silently at step 280 (rest_grad_norm grew 0.24 -> 1.1, no
traceback, process vanished). A fresh V=32768 run stays stable over 165+ steps
(gr 0.21-0.44 tracking the warmup lr, gb 0.009). The smaller codebook also
shrinks the AdamW codebook-grad contribution that was the growing component.

## How to train

```bash
# data/ already carries the compact streams and vocab_map.npz; if re-running:
python scripts/rebuild_compact2.py          # full 262144->32768 map + binaries
python scripts/train.py --config configs/v32_1b.yaml --dry-run   # ~8.0 ce at init
python scripts/train.py --config configs/v32_1b.yaml             # 212k steps
```
