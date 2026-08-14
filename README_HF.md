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
  - long-context
  - 2m-context
pipeline_tag: text-generation
---

# HAGI-DeepSeek-V4-Flash-0731-2M

**Source code & pipeline:** <https://github.com/ShmidtS/HAGI_v2>

A **lossy-compressed** derivative of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(256 MoE experts/layer × 43 layers) with an **extended 2M-token context**.

The 256 routed FFN experts of every layer are replaced by compact
**POD + ternary SwiGLU + int8-output** blocks, and the KV cache is stored in a
256-dim POD subspace so **2× more tokens fit in the same memory**.

> ⚠️ This is a compressed approximation, not the original model. It trades a
> small per-expert residual (~0.2–1.1%) for ~45 GB total size. Generation is
> coherent but degraded relative to the base model.

## What's inside

| Component | Format | Size |
|-----------|--------|------|
| Skeleton (non-expert weights) | `model.safetensors` + `config.json` | 16.69 GB |
| Routed experts (43 × 256) | `reduced/layer_{L}/expert_{k}.pt` | 30.69 GB |
| **Total** | | **~47 GB** |

### Expert format (per expert, ~2.77 MB)

```
P        [4096, 512]  fp32   input POD projection (per layer)
mu       [1, 4096]    fp32   input mean (subtract before projecting)
w1, w3   [4096, 103]  uint8  ternary gate/up, packed 5 trits/byte (inter=4096)
w1_scale, w3_scale [4096] fp32
w2       [384, 820]   uint8  ternary down, packed (inter=4096 → ceil(4096/5)=820)
w2_scale [384]        fp32
Q        [4096, 384]  int8   output basis (per-column scale)
Q_scale  [384]        fp32
```

Forward per routed expert:

```
z   = (x - mu) @ P                  # 4096 → 512
g   = silu(z @ W1) * (z @ W3)       # ternary SwiGLU, inter=4096
y   = (g @ W2) @ Q.T                # 384 → 4096 (int8 Q)
```

## Compression method

The core idea: **compressing the FFN on real activations, not on weights**.
Routed-expert weights are white noise (top-64 ≈ 8.8% of energy, s1/s64 ≈ 1.37),
so weight-rank reduction (SVD/Markov) is useless. The activations, however, are
low-rank (~276–302 effective dims), so each expert is replaced by a
**spectral (POD) input/output factorization + a ternary kernel**:

1. **Input POD** — `P [4096, 512]` from SVD of the layer's FFN-input
   activations; `z = (x − μ) P` projects onto the 512-dim signal subspace.
2. **Ternary SwiGLU kernel** — `z → silu(z·W1)·(z·W3)·W2` with
   `W1,W3,W2 ∈ {−1,0,1}` (packed 5 trits/byte), `inter = 4096`. Trained
   with straight-through estimator (Adam, cosine LR 2e-3, bf16 autocast).
3. **Output POD** — `Q [4096, 384]` from SVD of the expert output; the kernel
   targets `y·Q` (384 dims), reconstructed as `(·)Qᵀ` with int8 Q
   (per-column scale, 0.012% quant error).
4. **KV-cache POD** — the sliding-window KV (512-dim, K==V MQA) is stored in
   its 256-dim POD subspace (top-256 = 98.9–100% over all 43 layers),
   reconstructed on read → 2× tokens in the same KV memory.
5. **YaRN 16→32** — positional extrapolation for 2M tokens
   (65536 × 32 = 2,097,152).

Per-expert forward: `z = (x−μ)P;  g = silu(zW1)·(zW3);  y = (gW2)Qᵀ`.

## How it was built

1. **POD on real activations** (not weights): the FFN input is projected onto
   a 512-dim orthonormal subspace (`P`), the output onto 384 dims (`Q`).
   Weights are white noise (top-64 ≈ 8.8% energy) — POD on activations is the
   spectral decomposition of the layer-to-layer Markov transition.
2. **Ternary SwiGLU kernel** (`inter=4096`, ~5.8M ternary params/expert),
   trained with straight-through estimator (Adam, cosine LR, bf16 autocast).
3. **int8 Q** with per-column scale (0.012% quant error, better than fp8).
4. **KV-cache POD** 512→256 (top-256 = **98.9–100%** across all 43 layers) so
   the sliding-window KV is stored at half the dims.
5. **YaRN factor 16→32** for positional validity out to 2M tokens
   (65536 × 32 = 2,097,152), inference-time extrapolation beyond the trained
   factor 16.

## Results (per-expert residual, MSE(y_hat,y)/MSE(y,0))

| | |
|---|---|
| Layer 0 | 0.142–0.192% |
| Layers 1–41 | mostly 0.2–0.7%, max ~1.1% |
| KV-POD top-256 | 98.9–100% (min L40 = 98.9%) |

A centering bug in the first pipeline (teacher target computed on
`FFN(x−μ)` instead of `FFN(x)`) caused a 25–27% error and was fixed — it was
the dominant cause of incoherent output.

## Usage

The experts are a **custom packed format**, not standard `transformers`
weights. Load them with the provided scripts (see the source repo).

```bash
# assemble skeleton + reduced experts, then generate
python scripts/dsv4_generate_reduced.py "The capital of France is" 24

# 2M-context run (KV-POD 512->256 + YaRN 32)
python scripts/dsv4_test_kvpod.py "The capital of France is" 12
```

Requirements: `torch` (ROCm/CUDA, bf16), `transformers` (deepseek_v4),
`safetensors`, `gigatoken`. ~115 GB GPU VRAM recommended.

## Limitations

- Lossy: per-expert residual accumulates over 43 layers; output degrades and
  can repeat on long generations.
- 2M context is an **extrapolation** (YaRN factor 32 beyond the trained 16);
  long-tail positional quality is unverified on >1M-token prompts.
- CSA/HCA compressor entries are not yet POD-compressed (only the sliding KV).
