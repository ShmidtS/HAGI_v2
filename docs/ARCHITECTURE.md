# HAGI-2 -- Ternary RD-Channel Causal Language Model (V28)

HAGI-2 is a causal autoregressive language model designed as a communication
system. The transformer body is **ternary** (BitNet b1.58: weights in
{-1, 0, +1} stored at 1.585 bits/weight). That quantization is the *genuine*
discrete channel -- its noise is the only impairment. There is no
self-inflicted AWGN/LDPC physical channel.

---

## 1. Signal Path (Forward)

```
input_ids [B, T_text]
    |
    v
[STAGE 1: Source Encode]
  ConvEmbedding:
    token_compress(V, r) -> token_expand(r, H) -> CAUSAL Conv1d (left-pad only) -> RMSNorm
    |
    v  h [B, T_text, H]

  (Multimodal, if enabled):
    Image: patches -> Linear -> 2D-RoPE -> inv-var gate -> Q-Former -> [B, n_bridge, H]
    Audio: mel frames -> Linear -> 1D-RoPE -> inv-var gate -> Q-Former -> [B, n_bridge, H]
    h = concat([prefix, h_text])   [B, prefix_len + T_text, H]
    |
    v
[STAGE 2: UNIFIED Ternary Transformer Stack]
  L x TransformerBlock (pre-norm, causal KV-cache compatible):
    h = h + Attention(RMSNorm(h), RoPE, GQA, optional sliding window)
    h = h + (WaterFillingMoE(h) if MoE-layer else HebbianBilinearFFN(h))
  Collects: attn_entropy_penalty, moe_lb, routing_entropy, water_filling_loss
    |
    v  h_ctx [B, T, H]
    |
[STAGE 3: Auxiliary IB -- OFF-PATH, h_ctx.detach()]
  InformationBottleneck(H -> C -> H):
    q(z|h) = N(mu, exp(logvar))
    rate = KL[q||N(0,I)], distortion = ||h - h_hat||^2 / ||h||^2
    |
    v
[STAGE 4: Main LM Path]
  h_dec = final_norm(h_ctx)
  h_text = h_dec[:, prefix_len:]          (text-only positions)
  logits = lm_expand(lm_compress(h_text)) (factored rank-r head)
  ce_loss = cross_entropy(logits, targets)
    |
    v
[STAGE 4b: Off-path HEP Refinement -- h_ctx.detach(), opt-in]
  PredictiveRefiner: iterative extrinsic correction on a clone of h_ctx
    |
    v
[STAGE 5: Grounded Infomax -- on h_ctx (NOT detached), multimodal only]
  VICReg + InfoNCE on per-modality pooled embeddings
    |
    v
[LOSS = CE + sum(w_i * aux_loss_i)]
```

---

## 2. Modules

| Module | File | Params | Role |
|---|---|---|---|
| ConvEmbedding | conv_embedding.py | V*r + r*H + H*K + 2H | Factorized source encoder + causal pulse-shaping |
| Attention | attention.py | ~(2n_q+2n_kv)*hd*H + H | GQA + RoPE, 4 attention modes, sliding window, KV-cache |
| HebbianBilinearFFN | hebbian_ffn.py | 12H^2 + 2H | Bilinear SwiGLU: (A0*h) * silu(A1*h), gate-modulated |
| WaterFillingMoE | moe.py | (E+n_shared)*3*inter*H + router | SNR-gated top-k routing + batched dispatch |
| BitLinear | ternary.py | [out, in] FP master | BitNet b1.58: {-scale, 0, +scale} effective weights |
| InformationBottleneck | bottleneck.py | 3*C*H + H | Off-path variational H->C->H, rate+distortion |
| PredictiveRefiner | refinement.py | ~3*H^2 | Off-path HEP iterative hidden refinement |
| WaterFillingAllocator | water_filling.py | 2E | Per-expert capacity allocation via SNR EMA |
| MultimodalFusion | multimodal.py | varies | Q-Former bridge for fixed-size multimodal prefix |
| GroundedInfomax | grounded.py | M*H^2 | VICReg + InfoNCE joint embedding alignment |

---

## 3. Project Layout

```
src/hagi/
  config.py              # all hyperparameters + auto_configure (no hardcoded constants)
  version.py             # __version__ = "2.1.0", __architecture__ = "hagi2-codec-channel-v28"
  data/
    dataset.py           # MemmapDataset / MixedDataset (.bin token stores)
    sequential.py        # two-stage curriculum cycling iterator
    tokenizer.py         # gigatoken wrapper (fast) with HF/tokenizers fallback
  model/
    model.py             # HAGI -- the full model (forward path above)
    conv_embedding.py    # factorized source encoder + CAUSAL depthwise conv
    block.py             # TransformerBlock = pre-norm Attention + FFN/MoE
    attention.py         # GQA + RoPE + 4 attention modes + sliding window + KV-cache
    hebbian_ffn.py       # HebbianBilinearFFN: bilinear SwiGLU with gate modulation
    ternary.py           # BitLinear (BitNet b1.58) + ternarize() + _TernarizeSTE
    bottleneck.py        # InformationBottleneck (KL rate, RD distortion, FP32 params)
    moe.py               # WaterFillingMoE: SNR-gated routing + batched dispatch
    water_filling.py     # WaterFillingAllocator: per-expert capacity allocation
    refinement.py        # PredictiveRefiner + RefinementHead (off-path HEP)
    multimodal.py        # MultimodalFusion: Q-Former bridge + 2D/1D-RoPE
    grounded.py          # GroundedInfomax: VICReg + InfoNCE joint embedding
    norms.py             # RMSNorm (fp32 variance on CUDA)
    rope.py              # RoPE: 1D, 2D, RotaryEmbedding
    kv_cache.py          # KVCache: ring-free preallocated KV store
    exit_chart.py        # EXITChartHalt: convergence criterion
    outputs.py           # AuxLosses, ModelOutput dataclasses
  train/
    loop.py              # causal next-token training loop + LR schedule
    losses.py            # LossAggregator (CE + annealed aux regularizers)
    optim.py             # Muon (type-based: BitLinear.weight) + AdamW (everything else)
    checkpoint.py        # strict save/load (format version 7)
    distillation.py      # online hidden-state distillation (opt-in)
    _rocm_fsdp_stub.py   # FSDP no-op stubs for ROCm Windows without torch._C._distributed_c10d
  inference/
    generate.py          # pure causal AR generation with incremental KV-cache
configs/
  smollm2.yaml           # SmolLM2 tokenizer, RTX 3070 8GB, ~39.5M, text-only
  google.yaml            # Gemma tokenizer, Strix Halo 96GB, ~8.55B, everything on
  google_small.yaml      # Gemma tokenizer, Strix Halo 96GB, H=512, MoE, V30 auto-calc
scripts/
  train.py               # training entry point
  infer.py               # inference entry point (interactive REPL mode)
  run_ablation.py        # automated ablation experiment runner with CSV logging
  download_model.py      # teacher snapshot download
  preprocess_gemma.py    # data preprocessing
tests/                   # 164 unit tests (all CPU-compatible)
```

---

## 4. Configuration

Three canonical configs:

- **`configs/smollm2.yaml`** -- SmolLM2 tokenizer (vocab 49154), RTX 3070 8GB. ~39.5M params, text-only.
- **`configs/google.yaml`** -- Gemma tokenizer (vocab 262144), Strix Halo APU 96GB. ~8.55B params, MoE + refinement + sliding window + distillation.
- **`configs/google_small.yaml`** -- Gemma tokenizer, Strix Halo 96GB. H=512, 4 MoE experts, V30 auto-calculate.

`model.target_params` drives `auto_configure`, which solves H, L, C, n_q, n_kv,
head_dim, factor_rank, and MoE params from the non-embedding body budget.

**Every knob lives in the YAML** -- the training loop, optimizer, LR schedule,
and logging cadence all read from config with no hardcoded constants.

---

## 5. Optimizer

Type-based routing:
- **BitLinear.weight** -> Muon (Newton-Schulz orthogonalization, scale-aware WD)
- **All other params** -> AdamW (fused where available, weight_decay/no_decay split)

---

## 6. Design Rules

1. **Off-path auxiliaries.** IB, refinement are on `h_ctx.detach()` (no gradient to body).
   GroundedInfomax sends gradient to body deliberately (forms multimodal representation).
2. **C < H.** Core hidden size must be strictly less than hidden size (real compression).
3. **CAUSAL conv only.** Left-pad, no future leak. V25 symmetric-pad was root cause #4.
4. **No magic numbers.** `auto_configure(target_params)` derives all sizes.
5. **Checkpoint format V7.** Incompatible with V27 and earlier.
6. **ensure_fp32_params().** IB parameters converted to FP32 after bf16 model cast.
7. **Deferred MoE EMA.** SNR EMA committed after backward, outside checkpoint scope.

---

## 7. Inference

`inference/generate.py` is GPT-style causal AR with an incremental KV-cache:
1. Prefetch: `model(prompt, attention_mode="causal")` -- all prompt KV cached
2. Per-step: single-position forward, `process_generation_logits()`, sample, append
3. Constraints: forbidden tokens, min_new_tokens, repetition_penalty (1.2),
   no_repeat_ngram (2), temperature (0.8), top_k (50)

Generation is O(T) with KV-cache, O(T^2) without.

---

## 8. Checkpoints

`train/checkpoint.py` writes strict format V7:
`{format_version: 7, model, config, completed_updates, optimizer?}`.

Saves atomically via tempfile + `os.replace`. Rotates `keep_last=3` checkpoints.

Loading validates format version, config schema, and applies model state strictly
(any mismatch raises `IncompatibleCheckpointError`). Inference needs only the
`.pt` file and the tokenizer.

---

## Root Causes of Garbage Generation (V25)

All four independent causes are fixed in V28:

1. **IB + PredictiveDecoder in the main LM path** -> from-scratch deadlock
   (CE ~ ln(V)). Fix: keep auxiliaries strictly off-path on `h_ctx.detach()`.
2. **Bidir-first warmup curriculum** -> causal/AR path never trained.
   Fix: causal-dominant from step 0.
3. **Inference mask/shape bug** -> wrong logits shape + masked last position.
   Fix: real context visibility + correct reshape.
4. **Non-causal embedding conv** -> symmetric padding leaked future tokens into
   `hidden[t]`. Fix: causal left-pad conv (output[t] uses only input[0..t]).

**Lesson:** when training metrics are healthy but generation is garbage, verify
the embedding is causal and the inference path matches training before suspecting
undertraining.
