# HAGI — Ternary RD-Channel Causal Language Model

HAGI is a causal autoregressive language model reframed as a communication
channel. The transformer body is **ternary** (BitNet b1.58: weights in
`{-1, 0, +1}`), and that quantization is the *genuine* discrete channel — its
noise is the only impairment. There is no self-inflicted AWGN/LDPC physical
channel.

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
learning. `body.bottleneck_in_path` exists only to reproduce the failed
design for comparison.

---

## 2. Project layout

```
src/hagi/
  config.py            # all hyperparameters (no hardcoded constants elsewhere)
  version.py           # __version__, __architecture__
  data/
    dataset.py         # MemmapDataset / MixedDataset (.bin token stores)
    sequential.py      # two-stage curriculum cycling iterator
    tokenizer.py       # gigatoken wrapper (fast) with HF fallback
  model/
    model.py           # HAGI — the full model (forward path above)
    conv_embedding.py  # factorized source encoder + CAUSAL depthwise conv
    block.py           # TransformerBlock = Attention (RoPE) + HebbianBilinearFFN
    hebbian_ffn.py     # HebbianBilinearFFN config + warm-start helper
    ternary.py         # BitLinear (BitNet b1.58) + ternarize
    bottleneck.py      # InformationBottleneck (KL rate, RD distortion, perception)
    predictive.py      # PredictiveDecoder (extrinsic error highway) — opt-in
    multimodal.py      # MultimodalFusion (per-modality source encoders)
    uncertainty.py     # LearnedUncertainty + inverse_variance_update (Kalman)
    norms.py           # RMSNorm (fp32 variance under AMP)
    outputs.py         # AuxLosses, ModelOutput
  train/
    loop.py            # causal next-token training loop
    losses.py          # LossAggregator (CE + annealed aux regularizers)
    optim.py           # Muon + AdamW hybrid
    checkpoint.py      # strict save/load (format version 5)
    distillation.py    # online hidden-state distillation (opt-in)
  inference/
    generate.py        # pure causal AR generation
configs/
  smollm2.yaml         # SmolLM2 teacher, RTX 3070 8GB, ~50M, text-only
  google.yaml          # Gemma teacher, cloud 16GB, ~365M, multimodal
scripts/
  train.py             # training entry point
  infer.py             # inference entry point
  download_model.py    # teacher snapshot download
docs/
  ARCHITECTURE.md      # this file
```

---

## 3. Configuration

Two canonical configs cover the two intended deployments:

- **`configs/smollm2.yaml`** — SmolLM2 teacher / tokenizer (vocab 49154), RTX
  3070 8GB. ~50M params, text-only.
- **`configs/google.yaml`** — Gemma teacher / tokenizer (vocab 262146), cloud
  T4/V100 16GB. ~365M params, multimodal (image + audio) enabled, online
  distillation from Gemma.

`model.target_params` drives `auto_configure`, which solves hidden/layer/head
sizes from the non-embedding body budget. Any size field set explicitly in the
YAML (`hidden_size`, `core_hidden_size`, `context_layers`,
`attention.num_query_heads`, `attention.head_dim`) overrides the auto values.

**Every other knob lives in the YAML** — the training loop, optimizer, LR
schedule, attention curriculum, and logging cadence all read their parameters
from config and contain no hardcoded constants. The config dataclasses are:

| Section | Role |
|---------|------|
| `ModelConfig` | architecture: `vocab_size`, `hidden_size`, `core_hidden_size`, attention, embeddings, multimodal, body |
| `TrainConfig` | training: schedule, optimizers (`muon`/`adam`), loss weights, data, checkpoints, `attention_curriculum`, `logging` |
| `InferenceConfig` | generation: temperature, top_k, repetition penalty, max_new_tokens |
| `MuonConfig` | Newton-Schulz steps/coeffs, scale-aware WD cap |
| `AdamConfig` | AdamW betas/eps |
| `ScheduleConfig` | cosine LR shape (`stable_fraction`, `min_lr_ratio`) |
| `AttentionCurriculumConfig` | causal/soft/bidir probability mix + soft_beta range |

```bash
python scripts/train.py --config configs/smollm2.yaml --no-distill
python scripts/infer.py --checkpoint checkpoints/step-010000.pt --interactive
```

---

## 4. Training

`train/loop.py` trains **causal next-token prediction** (the inference regime).
A causal-dominant attention curriculum (driven by `attention_curriculum`)
mixes in `soft_causal`/`bidir` for a denser representation gradient early,
ramped out by mid-training. Loss =
`CE + w_rate·KL + w_distortion·(annealed)·distortion + w_perception·(annealed)·perception
+ w_attn_entropy·entropy_floor_penalty`. Distortion/perception β-anneal over
warmup so the LM signal shapes the representation first.

Optimizer (`train/optim.py`): **Muon** (Newton-Schulz orthogonalization +
scale-aware weight decay bounded by `muon.wd_cap`) for 2D weights; **AdamW**
(`adam.beta1/beta2/eps`) for embeddings, norms, gates, the rate-critical FP32
bottleneck linears, and multimodal source codebooks. Ternary 2D masters ride
Muon; their FP latents are trained, the `{-1,0,1}` values recomputed every
forward.

---

## 5. Inference

`inference/generate.py` is pure GPT-style causal AR: feed the prompt, take the
logits at the last position (`[B*T,V]` → last per row), sample, append, repeat.
The model sees the **real** context — nothing is erased.

---

## 6. Checkpoints

`train/checkpoint.py` writes a strict schema (`format_version` 5):
`{format_version, model, config, completed_updates, optimizer?}`. Loading
validates the format version, the config, and applies the model state strictly
(any mismatch raises `IncompatibleCheckpointError`). The config is persisted
via `dataclasses.asdict`, so every knob is reconstructable from a checkpoint
alone — inference needs nothing but the `.pt` file and the tokenizer.

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
   distribution of frequent tokens. Fix: causal left-pad conv
   (`output[t]` uses only `input[0..t]`).

**Lesson:** when training metrics are healthy but generation is garbage,
verify the embedding is causal and the inference path matches the training
path before suspecting undertraining.
