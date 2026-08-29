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
  - quantization
  - gptq
  - int4
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

Every routed FFN expert is replaced by a compact **mixed-precision
full-rank SwiGLU** block (binary gate/up + GPTQ int4 down, ~1.9× smaller than
the FP4 original), and the KV cache is stored in **int8** with a
**distance-dependent (pyramid) sliding window**, so the context grows to 2M
tokens **without raising the YaRN factor**.

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
wins the size via **mixed-precision quantization**: binary gate/up + int4
down with GPTQ error feedback.

## What's inside

| Component | Format | Notes |
| ----------- | -------- | ------- |
| Skeleton (non-expert weights) | `model.safetensors` + `config.json` | 16.69 GB |
| Routed experts (43 × 256) | `dsv4_reduced/layer_{L}/expert_{k}.pt` | bin W13 + int4 W2, mode `i1i4` |
| Per-layer rotation + mean | `P.pt` [4096,4096], `mu.pt` [1,4096] | fp32, per layer |
| int8 KV scales | `kv_int8_scales.pt` (43 × 512) | per-channel, RoPE-safe |

### Expert format (per expert, 6.6 MB vs 12.6 MB FP4 = 1.91×)

```text
w1a, w3a   [2048, 512]  uint8   binary ±1 gate/up, 1 bit/weight (4096 in)
w1a_scale, w3a_scale [2048] fp32  per-row LS scale (mean|w|)
w2a        [4096, 1024] uint8   int4 grid ±7 down, 4 bits/weight (2048 in)
w2a_scale  [4096, 16]   fp32   per-(row, group-of-128) LS scales
bias1a, bias3a [2048]   fp32   mu folded in
bounds, inter, mode     —       full-rank split, mode marker "i1i4"
```

Forward per routed expert:

```text
z = (x - mu) @ P                 # 4096 → 4096 (rotation only)
g = soft_lim(z @ W1ᵀ + b1)       # binary gate, per-row scale
u = soft_lim(z @ W3ᵀ + b3)       # binary up, per-row scale
y = (silu(g) * u) @ W2ᵀ          # int4 down (decoded from packed on the fly)
```

## Compression method

1. **Full-rank rotation** — per-layer orthogonal `P [4096,4096]` + mean
   `mu` whiten the FFN inputs; no dimensionality reduction (POD was rejected).
   `mu` is folded into the gate/up biases.
2. **Binary gate/up (W1, W3)** — ±1 with per-row fp32 scale (the LS-optimal
   `mean|w|`); signs from exact FP32 rotated weights with a tie-break
   (FP4 zeros must not become a silent `-1`).
3. **GPTQ int4 down (W2)** — int4 grid ±7 with per-(row, group-128) LS
   scales, quantized with **LDLQ sequential error feedback** over the
   activation Hessian `hᵀh` from real text (h computed with the quantized
   W13 already in place). GPTQ minimizes the *functional* expert output
   error — with naive rounding W2 alone contributed 13.6% norm error;
   GPTQ removes essentially all of it.
4. **Identity output** — `kp = 4096`, no output basis (the int4-QAT basis Q
   was abandoned once the model ran at full rank).

**Per-expert quality** (norm error ‖y_pred−y‖/‖y‖ on real activations):
median ~13–16%, dominated by the binary-W13 floor (~12.4%). A noise-injection
benchmark on the FP4 model (ε added to expert outputs) showed 5–10% → clean
text, 15% → still coherent, 20%+ → degradation; the measured levels sit
inside the coherent band. E2E text quality gate on the compressed model is
the current milestone.

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
# Utilities: scripts/dsv4_experts.py (decode / load / bit-pack: binary,
# int4, n-bit, int6), refit in scripts/dsv4_refit_experts.py
# (env: W13_BITS=1 W2_BITS=4 W2_GPTQ=1). Generation from compressed files:
# scripts/dsv4_generate_ttt.py  (INT4X_OFF=1 -> FP4 baseline A/B,
#                                EXPERT_NOISE=eps -> noise benchmark)
# TTT (anchored RLS "eternal thinking") and --evolve self-talk run on top
# of the same persisted expert files.
```

## Limitations

- Lossy: per-expert norm error ~13–16% (median, real activations); the
  e2e text-quality gate on the compressed model is still pending.
- Refit is in progress (v19b, ~20/43 layers done in the `i1i4` format).
- Custom packed format (mode marker per file) — not a drop-in GGUF; decode
  from packed tensors on the fly.
