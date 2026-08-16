# Compression plan: attention + top layer (lm_head / embed / MTP)

The FFN experts are already reduced with the **channel-per-expert** method
(POD input + ternary SwiGLU core + learnable int4 Q; covered experts ≤0.01%
on real routed activations, uncovered 13% ~0.5–1.8% on a router-weight proxy).
This plan covers the parts we have **not** touched yet: the attention
projections, the output head, the embedding, and the MTP layers.

## Core idea (in plain words)

Each routed expert is a **separate communication channel** `x → y`. We do not
compress it as part of a layer — we **measure its transfer function** by
driving it with a universal test signal (unifold: bootstrap of the real POD
manifold + 0.1σ jitter) and refit a compact block against the recorded `(x, y)`
pairs, per expert. We probe only the **real input manifold** (where the router
actually sends tokens) plus a small neighbourhood — not the full ±5σ volume,
because the ternary kernel cannot express the expert there (25–36% residual).
Experts the router never activated (13%) are refit on a proxy manifold from
their nearest router-weight neighbours.

## Current uncompressed budget (bf16 unless noted)

| Component | Size | Notes |
|---|---|---|
| Attention (43 layers) | **10.2 GB** | o_a 2.9 + o_b 2.9 + q_b 2.9 + compressor 1.0 + q_a 0.36 + kv 0.18 |
| `lm_head` [129280×4096] | **1.06 GB** | output projection to vocab |
| `embed_tokens` [129280×4096] | **1.06 GB** | input embedding |
| MTP (mtp_0..2) | **≈9.7 GB** | full MoE: 256 experts × 3 layers, FP4 |

Total ≈ **22 GB** of untouched weights.

## Core insight (already proven on the FFN)

The hidden states live in a ~302-dim subspace (top-512 explains 99.93% of
variance). **Every** linear map that touches the hidden state therefore only
needs to act on that subspace:

- W [out, 4096]  →  W_core [out, r] @ P [r, 4096]        (r ≈ 512)
- W [4096, out]  →  Pᵀ [4096, r] @ W_core [r, out]
- lm_head [V, 4096] → U [V, r_vocab] @ S [r_vocab, r] @ P [r, 4096]
  (double low-rank: the vocab side is low-rank too)

The same POD basis P is shared by all projections in a layer — compute it once
from the layer's real hidden states (as we do for the FFN input).

## Per-component plan

### 1. Attention (10.2 GB → target ~3–4 GB)

Apply POD-on-activations to each projection, no ternary (attention is more
sensitive than FFN; keep the core in int8):

- `o_a_proj` [8192×4096] → [8192×r] @ P, r=512 → 2.9 GB → 0.56 GB.
- `o_b_proj` [4096×8192] → Pᵀ @ [r×8192], r=512 → 2.9 GB → 0.56 GB.
- `q_b_proj` [32768×1024] → [32768×r'] @ Pq (POD on the q_a output, r'≈384)
  → 2.9 GB → ~0.9 GB.
- `q_a_proj` [1024×4096] → [1024×r] @ P → 0.36 GB → 0.13 GB.
- `kv_proj` [512×4096] → [512×r] @ P → 0.18 GB → 0.06 GB.
- `compressor` (sparse-attention indexer) — leave as-is in v1; measure first.

Verification: per-layer Q/K/V/O activation MSE + end-to-end logit error must
stay <1% before shipping. If the ternary core on attention hurts, keep int8
low-rank only.

### 2. lm_head (1.06 GB → ~0.14 GB)

The final hidden state is the same low-rank subspace. Factor:

```
lm_head ≈ U [129280, r_vocab] @ S [r_vocab, r] @ P [r, 4096]
```

- r = 512 (hidden side), r_vocab = 512 (vocab side, from SVD of the
  trained lm_head).
- Size: 129280×512 + 512×512 + 512×4096 ≈ 68.6M params ≈ **137 MB** (bf16)
  = **7.7×** smaller. int8 core → ~69 MB.
- The output logits are the most quality-sensitive surface: measure top-k
  agreement / logit MSE against the original, not just weight MSE.

### 3. embed_tokens (1.06 GB → ~0.14 GB)

Same double low-rank: `E ≈ E_vocab [129280, r_vocab] @ S [r_vocab, r] @ P`.
If `tie_word_embeddings` is enabled, share the factorisation with lm_head.

### 4. MTP layers (≈9.7 GB → ~2.1 GB)

mtp_0..2 are full MoE layers (256 experts each). Reuse the exact
`dsv4_reduce_layer.py` pipeline (POD input 4096→512, ternary inter=4096,
int8 Q output 4096→384) on `mtp.{i}.ffn.experts.{k}` instead of
`layers.{li}.ffn.experts.{k}`. Expected: 3 × 256 × 2.77 MB ≈ **2.1 GB**.

## Order of work (each step independently verifiable)

1. **MTP first** — zero new machinery, reuse `dsv4_reduce_layer.py`; biggest
   cheapest win (~7.6 GB saved).
2. **lm_head + embed** — low-rank SVD, quick, ~1.9 GB saved.
3. **Attention** — POD per projection, ~6 GB saved, most sensitive; land last
   with strict logit-error gates.

## Gates (do not pass without evidence)

- Per-surface residual: `MSE(pred, target)/MSE(target, 0)` < 1%.
- Full round-trip: `reduced_model(x) vs original_model(x)` over a held-out
  prompt set — the only honest end-to-end metric.
- Generation smoke test: `dsv4_generate_reduced.py` still decodes coherent
  text after each component is swapped in.

## Goal update — increase context, not shrink size

Decision: on the reduced model, spend the KV-POD win on **context**, not on
parameter count.

**VERIFIED (full 43-layer reduced model, 3000 tokens/layer):**

- KV subspace (512-dim, K==V MQA): top-256 = **98.9–100%** across all 43
  layers (L0=100.0%, min L40=98.918%, mean ~99.4%). top-384 ≥ 99.77%.
- KV cache today: 512 dim × 43 layers × 2 B = 43 KB/token → 43 GB per 1M.
- After POD: 256 dim/token → **2M tokens fit in the same 43 GB** (sliding
  branch; CSA/HCA compressor entries still 512-dim in this first pass →
  ~1.66× total, the sliding KV is the 80% win).
- Bases: `checkpoints_dsv4/pod_reduced/P_kv_L*.pt` (43 bases, rank 256).

**Integration (DONE + verified end-to-end):**

1. `scripts/dsv4_collect_kv_reduced.py` — KV/x_L hooks on the reduced model
   (43 layers, 3000 tokens).
2. `scripts/dsv4_kvcache_pod.py` — `KVCompressor` (compress/decompress, P
   cast to input dtype) + `install_kv_compression(cache, kvc)` (patches
   `DynamicCache.update` to store z=kv@P_kv, reconstruct z@P_kv.T on read) +
   `patch_yarn_factor(config, 32)` (MUST run BEFORE `from_pretrained` — the
   YaRN `inv_freq` buffer is baked at init).
3. `scripts/dsv4_test_kvpod.py` — end-to-end: cache stores
   `(1,1,17,256)` (head_dim 256, was 512), generation works, YaRN factor
   verified (compress_inv_freq DIFFERS from factor 16, main rope unchanged).

**Open item (non-blocking):** collect post-RoPE KV (what the cache actually
stores) and confirm the pre-RoPE basis matches; the rope slice (64 dims) is
position-rotated, so a tiny extra loss is possible on that slice.
