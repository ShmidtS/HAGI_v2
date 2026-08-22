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
  - int8
  - kv-cache
  - long-context
  - 2m-context
pipeline_tag: text-generation
---

# HAGI-DeepSeek-V4-Flash-0731-2M

**Source code & pipeline:** <https://github.com/ShmidtS/HAGI_v2>

A **lossy-compressed** derivative of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(256 MoE experts/layer × 43 layers) with an **extended 2M-token context**.

> ⚠️ **Status: in progress / experimental.** The compression is not yet
> published and the reduced model is not yet fully assembled. The description
> below reflects the current design, not a released artifact.

Every routed FFN expert is replaced by a compact
**full-rank two-stage ternary SwiGLU** block, and the KV cache is stored in
**int8** with a **distance-dependent (pyramid) sliding window**, so the
context grows to 2M tokens **without raising the YaRN factor**.

## Core idea: each expert is a communication channel

The 11008 routed experts are treated as **independent unitary units**, not as
parts of a layer. Each expert is a channel `x → y`: we measure its transfer
function by driving it with a **universal test signal (unifold)** and refit a
compact block against the recorded `(x, y)` pairs — per expert, individually.

**Key finding — the weights are noise, and so (almost) are the activations.**
Expert weights are pairwise orthogonal white noise (cannot be merged or
factored). The activations are *almost* full-rank: an earlier "top-512 =
99.93%" claim was an overfit artifact of a 3000-token sample; the honest
spectrum (259072 tokens) gives ~59% energy at K=512 (≈41% out-of-sample) and
~95% only at K=3072. So the compression keeps the **full 4096-dim rank** and
instead uses a **two-stage ternary** kernel for the size win.

## What's inside

| Component | Format | Notes |
| ----------- | -------- | ------- |
| Skeleton (non-expert weights) | `model.safetensors` + `config.json` | 16.69 GB |
| Routed experts (43 × 256) | `dsv4_reduced/layer_{L}/expert_{k}.pt` | two-stage ternary |
| Per-layer rotation + mean | `P.pt` [4096,4096], `mu.pt` [1,4096] | fp32, per layer |
| int8 KV scales | `kv_int8_scales.pt` (43 × 512) | per-channel, RoPE-safe |

### Expert format (per expert)

```
P              [4096, 4096]  fp32    per-layer orthogonal rotation (whitening)
mu             [1, 4096]     fp32    per-layer input mean
w1, w3         [2048, 820]   uint8   ternary gate/up, packed 5 trits/byte (inter=2048)
w1_q2, w3_q2   [2048, 820]   uint8   second ternary stage (residual refinement)
w2             [4096, 410]   uint8   ternary down, packed (inter=2048)
w2_q2          [4096, 410]   uint8   second ternary stage
scales         [...]         fp32    per-row scale for each stage
```

Forward per routed expert:

```
z = (x - mu) @ P                 # 4096 → 4096 (rotation only)
g = silu(z @ W1) * (z @ W3)      # two-stage ternary SwiGLU, inter = 2048
y = g @ W2                       # 4096 → 4096 (identity output)
```

Each ternary weight `W` is a **sum of two ternary matrices**
`W = W_q · s + W_q2 · s2`, which roughly halves the residual versus a single
ternary matrix at the same bit cost.

## Compression method

1. **Full-rank rotation** — per-layer orthogonal `P [4096,4096]` + mean
   `mu` whiten the FFN inputs; no dimensionality reduction (POD was rejected).
2. **Two-stage ternary SwiGLU kernel** — `z → silu(z·W1)·(z·W3)·W2` with
   `W1,W3,W2 ∈ {−1,0,1}` (packed 5 trits/byte), `inter = 2048`, each weight a
   sum of two ternary stages. Trained with straight-through estimator + Muon
   (zeropower) optimizer, bf16 autocast.
3. **Identity output** — `kp = 4096`, no output basis. The int4-QAT output
   basis Q was abandoned once the model ran at full rank.

**Per-expert quality (full 4096-dim, weighted MSE(y_pred,y)/MSE(y,0)):**
~0.6–0.9% residual (measured on the current refit). Earlier "≤0.01%" figures
were a reduced 384-dim output-space loss and are not comparable.

## Attention: int8 KV-cache + pyramid for 2M context

The context is extended **without YaRN extrapolation**:

- **int8 KV-cache** — K/V stored per-channel int8 (static scale, RoPE-safe:
  K is stored post-RoPE), ~2× KV memory with ~0.005% reconstruction error
  (worst 0.0125% across 43 layers). Replaces an earlier low-rank KV-POD.
- **Pyramid sliding window** — nearby tokens keep full 512 channels; older
  tokens are channel-truncated per
  `r(d) = clamp(512 >> ⌊log2(d/1024 + 1)⌋, 32, 512)` (base 512, window 1024,
  min 32). Far tokens cost less, so the window grows to 2M at the same budget.

## Usage

```python
# Utilities: scripts/dsv4_experts.py (decode / load / ternary pack-unpack),
# and the refit in scripts/dsv4_refit_experts.py. Generation currently runs
# the EXACT model from lossless_layers via scripts/dsv4_generate_fast.py /
# dsv4_generate_real.py. Reduced-model generation is not yet wired up.
```

## Limitations

- Lossy: per-expert residual ~0.6–0.9% (full 4096-dim).
- Refit is in progress (239 / 11008 experts across 3 layers); the reduced
  model is not yet assembled or generating.
- Custom ternary dequant on read — not a drop-in GGUF.
