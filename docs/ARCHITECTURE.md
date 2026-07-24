# HAGI — Ternary RD-Channel Causal Language Model

HAGI is a causal autoregressive language model reframed as a communication
channel. The transformer body is **ternary** (BitNet b1.58: weights in
`{-1, 0, +1}`), and that quantization is the *genuine* discrete channel —
its noise is the only impairment. There is no self-inflicted AWGN/LDPC
physical channel (the abandoned V8–V23 design that caused chronic divergence).

This document describes the **working** architecture after the four root-cause
fixes that ended a long "garbage generation" period. See
[Root causes of garbage generation](#root-causes-of-garbage-generation) below.

---

## 1. Signal path (forward)

```
input_ids
  │
  ▼  source encoder (factorized, CAUSAL conv — no future leak)
  h  ───────────────────────────────────────────────────┐
  │                                                     │
  ▼  context stack  (ternary transformer, any attn mode)│
  h_ctx ──────────────────────────────────────────────► │  auxiliary:
  │                                                     │  InformationBottleneck
  │                                                     │  → KL / distortion / perception
  ▼  expression stack (ternary transformer, causal)     │  (regularizer only — does NOT
  h_dec                                                 │   touch the LM signal)
  │                                                     │
  ▼  RMSNorm → factored LM head (rank-r)                │
  logits
```

**The main path is `source-encode → context → expression → LM head`.**
Everything else (IB, predictive decoder, multimodal fusion) is auxiliary or
opt-in and never intercepts the LM signal.

### Why the IB is off the main path
Inserting the variational `InformationBottleneck + PredictiveDecoder` directly
in the main path (`context → IB(z) → PD → rate_up → LM head`) **deadlocks
from-scratch training**: the next-token CE stalls at ≈ ln(V) (uniform random).
Keeping the IB as an auxiliary KL-rate regularizer on `h_ctx` recovers
learning (CE 11.4 → 0.39 in 200 steps in the ablation). `body.bottleneck_in_path`
exists only to reproduce the failed design for comparison.

---

## 2. Modules

| File | Role |
|------|------|
| `model/model.py` | `HAGI` — the full model (forward path above). |
| `model/conv_embedding.py` | Factorized source encoder `V×r + r×H` + **causal** depthwise Conv1d pulse-shaping filter. |
| `model/block.py` | Ternary `TransformerBlock` = `Attention` (RoPE, bidir/causal/prefix/soft_causal) + `HebbianBilinearFFN`. Reusable pieces: `RotaryEmbedding`, `apply_rope`, mask builders, `AttentionConfig`. |
| `model/hebbian_ffn.py` | `HebbianBilinearFFN` config + the dense warm-start helper. |
| `model/ternary.py` | `BitLinear` (BitNet b1.58: per-output-channel absmean scale, identity STE) + `ternarize`. |
| `model/bottleneck.py` | `InformationBottleneck` (H→C variational encoder, KL rate, RD distortion, RDP perception). |
| `model/predictive.py` | `PredictiveDecoder` (extrinsic error highway + HEP + Kalman blend) — opt-in, off the main path. |
| `model/multimodal.py` | `MultimodalFusion` (per-modality source encoders + shared/specific subspace + inv-var gating). |
| `model/uncertainty.py` | `LearnedUncertainty` + `inverse_variance_update` (K=P/(P+R) Kalman blend). |
| `model/norms.py` | `RMSNorm` (fp32 variance under AMP). |
| `model/outputs.py` | `AuxLosses`, `ModelOutput`. |
| `config.py` | All knobs: `Config` / `ModelConfig` / `TrainConfig` / `InferenceConfig` + `auto_configure`. |

### `src/hagi_v4/` layout
```
config.py            # all hyperparameters (no hardcoded constants elsewhere)
version.py
data/                # dataset.py, sequential.py, tokenizer.py
model/               # model.py + the modules above
train/               # loop.py, losses.py, optim.py, checkpoint.py, distillation.py
inference/           # generate.py
```

---

## 3. Configuration

Two canonical configs cover the two intended deployments:

- **`configs/smollm2.yaml`** — SmolLM2 teacher / tokenizer (vocab 49154), RTX
  3070 8GB. ~50M params, text-only.
- **`configs/google.yaml`** — Gemma teacher / tokenizer (vocab 262146), cloud
  T4/V100 16GB. ~365M params, multimodal (image + audio) enabled, online
  distillation from Gemma.

`target_params` drives `auto_configure`, which solves hidden/layer/head sizes
from the non-embedding body budget. Any size field set explicitly in the YAML
(`hidden_size`, `core_hidden_size`, `context_layers`, `attention.num_query_heads`,
`attention.head_dim`) overrides the auto values. **Every other knob lives in
the YAML** — the training loop reads it from config and contains no hardcoded
constants.

```bash
python scripts/train_v4.py --config configs/smollm2.yaml --no-distill
python scripts/infer_v4.py --checkpoint checkpoints/step-010000.pt --interactive
```

---

## 4. Training

`train/loop.py` trains **causal next-token prediction** (the inference regime).
A causal-dominant attention curriculum mixes in `soft_causal`/`bidir` for a
denser representation gradient early, ramped out by mid-training. Loss =
`CE + w_rate·KL + w_distortion·(annealed)·distortion + w_perception·(annealed)·perception
+ w_attn_entropy·entropy_floor_penalty`. Distortion/perception β-anneal over
warmup so the LM signal shapes the representation first.

Optimizer (`train/optim.py`): **Muon** (Newton-Schulz orthogonalization +
scale-aware weight decay) for 2D weights; **AdamW** for embeddings, norms,
gates, the rate-critical FP32 bottleneck linears, and multimodal source
codebooks. Ternary 2D masters ride Muon; their FP latents are trained, the
`{-1,0,1}` values recomputed every forward.

---

## 5. Inference

`inference/generate.py` is pure GPT-style causal AR: feed the prompt, take the
logits at the last position (`[B*T,V]` → last per row), sample, append, repeat.
The model sees the **real** context — nothing is erased.

---

## Root causes of garbage generation

"Garbage output" had four independent causes, each found only because the
previous fix did not resolve the symptom. All are fixed:

1. **IB + PredictiveDecoder in the main LM path** → from-scratch deadlock
   (CE ≈ ln(V)). Fix: keep the IB as an auxiliary regularizer off the path.
2. **Bidir-first warmup curriculum** → the causal/AR path never trained, so
   every checkpoint taken during warmup was out-of-distribution for
   generation. Fix: causal-dominant from step 0.
3. **Inference mask/shape bug** → causal generation marked the last prompt
   token as "erased" (feeding the model its learned `unknown_embed` instead
   of the real token) and treated `[B*T,V]` logits as `[B,V]`. Fix: real
   context (`semantic_unknown_mask` all-False) + correct reshape.
4. **Non-causal embedding conv** (the last and most subtle) → the
   pulse-shaping Conv1d used symmetric padding, so `hidden[t]` read future
   tokens `t+1, t+2`. The attention therefore never learned next-token
   conditioning (the conv handed it the answer from the future); at inference
   the last position has no future, so generation collapsed to the marginal
   distribution of frequent tokens — the *same* words on *every* prompt,
   prompt having zero effect. Fix: causal left-pad conv
   (`output[t]` uses only `input[0..t]`). Verified: 0 future-token leaks after
   the fix; 300 causal steps from scratch gave CE 10.0 → 4.3 and prompt-
   conditioned generation on tinystories.

**Lesson:** when training metrics are healthy but generation is garbage,
verify the embedding is causal and the inference path matches the training
path before suspecting undertraining.
