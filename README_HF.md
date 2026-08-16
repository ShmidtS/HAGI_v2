---
license: other
license_name: deepseek
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
language:
  - en
  - zh
tags:
  - deepseek-v4
  - moe
  - compression
  - ternary
  - pod
  - int4
  - qat
  - long-context
  - 2m-context
pipeline_tag: text-generation
---

# HAGI-DeepSeek-V4-Flash-0731-2M

**Source code & pipeline:** <https://github.com/ShmidtS/HAGI_v2>

A **lossy-compressed** derivative of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(256 MoE experts/layer × 43 layers) with an **extended 2M-token context**.

Every routed FFN expert is replaced by a compact
**POD + ternary SwiGLU + int4-QAT** block, and the KV cache is stored in a
256-dim POD subspace with a **distance-dependent (pyramid) rank**, so the
context grows to 2M tokens **without raising the YaRN factor**.

## Core idea: each expert is a communication channel

The 11008 routed experts are treated as **independent unitary units**, not as
parts of a layer. Each expert is a channel `x → y`: we measure its transfer
function by driving it with a **universal test signal (unifold)** and refit a
compact block against the recorded `(x, y)` pairs — per expert, individually.

The signal covers the **real input manifold** (where the router actually sends
tokens) plus a small 0.1σ jitter around it. We deliberately do NOT probe the
full ±5σ volume: on empty regions the ternary kernel cannot express the expert
response (25–36% residual), so full-volume coverage is a waste — the manifold
+ neighbourhood is the honest working band.

Experts the router never activated on the 774K-sample collection (13%) have no
real manifold. They are refit on a **proxy manifold from their nearest
router-weight neighbours** (cosine similarity), bootstrap + 0.1σ jitter, with a
stall-based stop once the residual plateaus.

> ⚠️ This is a compressed approximation, not the original model. It trades a
> small per-expert residual (~0.012% median) for a ~3× size reduction of the
> MoE. Generation is coherent but degraded relative to the base model.

## What's inside

| Component | Format | Size |
|-----------|--------|------|
| Skeleton (non-expert weights) | `model.safetensors` + `config.json` | 16.69 GB |
| Routed experts (43 × 256) | `reduced/layer_{L}/expert_{k}.pt` | ~14.5 GB |
| KV POD bases + means | `pod_reduced/P_kv_L{L}.pt`, `mean_kv_L{L}.pt` | ~22 MB |
| **Total** | | **~31 GB** |

### Expert format (per expert, ~1.32 MB)

```
P        [4096, 512]  fp32   input POD projection (per layer)
mu       [4096]       fp32   input mean (subtract before projecting)
w1, w3   [1024, 103]  uint8  ternary gate/up, packed 5 trits/byte (inter=1024)
w1_scale, w3_scale [1024] fp32
w2       [512, 205]   uint8  ternary down, packed (inter=1024 → ceil(1024/5)=205)
w2_scale [512]        fp32
Q        [4096, 256]  uint8  int4 output basis, packed 2 nibbles/byte (kp=512)
Q_scale  [512]        fp32   per-column scale (max/28)
Q_bits   scalar       int64  4 (format marker)
```

Forward per routed expert:

```
z   = (x - mu) @ P                  # 4096 → 512
g   = silu(z @ W1) * (z @ W3)       # ternary SwiGLU, inter=1024
y   = (g @ W2) @ Q.T                # 512 → 4096 (int4 Q, dequant on read)
```

## Compression method

The core idea: **compress the FFN on real activations, not on weights**.
Routed-expert weights are white noise (weight-rank reduction is useless), but
the activations are low-rank. Each expert is replaced by a spectral (POD)
factorization + a ternary kernel + a quantized output basis:

1. **Input POD** — `P [4096, 512]` from SVD of the layer's FFN-input
   activations; `z = (x − μ) P` projects onto the 512-dim signal subspace.
2. **Ternary SwiGLU kernel** — `z → silu(z·W1)·(z·W3)·W2` with
   `W1,W3,W2 ∈ {−1,0,1}` (packed 5 trits/byte), `inter = 1024`. Trained with
   straight-through estimator + Muon (zeropower) optimizer, bf16 autocast.
3. **int4 QAT output basis** — `Q [4096, 512]` is quantization-aware trained
   to int4 (2 nibbles/byte, `scale = max/28`, levels −7..7) against the full
   4096-dim target. Per-column scale stored as fp32.

**Per-expert quality (router-based):**

- **Covered experts (87%)** — early-stop on the honest residual over their
  **real routed activations** at **≤ 0.01%** (full 4096-dim loss).
- **Uncovered experts (13%)** — no real samples; residual on the proxy
  manifold ~0.5–1.8% (they are rare, so their weighted contribution is small).
- **Adaptive size** — `inter/kp` chosen by activation count `n_k`:
  `<200→(1024,512)`, `200–400→(2048,512)`, `≥400→(4096,768)`. Different
  experts may end up different sizes; packing and inference are per-expert, so
  this is fine.

## Attention: KV-POD + pyramid for 2M context

The context is extended **without YaRN extrapolation** (factor stays 8, inside
the trained factor-16 range):

- **KV-cache POD** — K/V are stored in a 256-dim POD subspace
  (`P_kv [512, 256]` + `mean_kv [512]` per layer), halving KV memory.
- **Pyramid rank** — tokens within `window=4096` of the query read back at
  full rank 256; older tokens read back at `r(d) = clamp(256 >> ⌊log2(d/4096+1)⌋, 16, 256)`.
  Far tokens cost less, so the sliding window grows to 2M at the same memory
  budget (≈5.7 GB KV cache at 2M tokens, bf16).

## Usage

```python
# The reduced experts + POD bases are loaded by the pipeline in
# github.com/ShmidtS/HAGI_v2 (scripts/dsv4_generate_reduced.py).
# The skeleton is a standard transformers checkpoint.
```

## Limitations

- Lossy: per-expert residual ~0.012% (median); hard experts can be higher.
- Generation quality is degraded relative to the base model.
- Custom ternary/int4 dequant on read — not a drop-in GGUF; load via the
  HAGI_v2 pipeline or the bundled safetensors (`dsv4_reduced.safetensors`).
