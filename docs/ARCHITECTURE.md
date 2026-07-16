# HAGI V8 — Probabilistic Codec Language Model

## Channel-Correct Training Contract

Semantic erasure and physical channel corruption are independent events. `semantic_unknown_mask` replaces unknown token embeddings with the learned `semantic_unknown_embed` before cache writes, spectral mixing, bottlenecking, or source-state derivation; changing placeholder IDs under the same mask therefore cannot change the model input at the causal seam. The source encoder then produces the systematic latent, learned continuous analog redundancy is derived from that clean latent, and only after that does the independently sampled physical channel corrupt received systematic symbols. Erasure is never represented by a token ID.

Four boolean `[B,T]` masks define the runtime contract: `semantic_unknown_mask` controls semantic visibility, `prediction_mask` selects logits/objective rows, `valid_target_mask` excludes padding and invalid targets, and `physical_corruption_mask` controls the independent latent channel. `prediction_mask` must be a subset of both semantic unknown and valid target positions; physical corruption does not modify semantic visibility or token IDs.

Training samples a random-unknown or full-suffix task independently per sequence with a 50/50 policy. A valid sequence has one terminal EOS before padding; suffix prediction includes that EOS. Generation starts with a fully unknown suffix, obtains compact gathered logits, emits left-to-right, accepts EOS only at generated lengths `2..max_new_tokens`, pads shorter rows, and returns `GenerationOutput(token_ids, generated_lengths, finished)`.

`step` always means completed optimizer updates. Each update consumes exactly `grad_accum_steps` distinct dataloader yields, and curriculum selection is set explicitly from that optimizer step before microbatch collection.

During active distillation, reconstruction and hidden-state alignment form one joint objective on every update. `distill_align` is student-owned and registered before optimizer construction; decoder correction is trained against the known clean-minus-received channel error rather than by minimizing extrinsic magnitude.

Checkpoint format v3 is a strict fresh-only inference artifact with exactly `format_version`, `model`, `config`, and `completed_updates`. Training resume, migration, partial loading, and legacy compatibility are not supported. Checkpoints use flat `step-XXXXXX.pt` paths with no run-specific subdirectories, and training refuses to start when the checkpoint root contains any entry. Old artifacts, including the collapsed step-501 probe and v1/v2 checkpoints, remain diagnostic-only; generation claims require fresh retraining.

Diagnostics report raw gradient norm and gradient RMS. Optional clipping is configured with `train.max_grad_norm`; a finite raw norm such as 40 is a measurement, not an explosion classification. `objective_loss` (with compatibility alias `loss`) names the optimized blended objective, while `masked_ce` is pure CE over gathered prediction rows and `bpt` is only `masked_ce / ln(2)`. `suffix_ce`, suffix/random task ratios, semantic/physical mask ratios, top-2 posterior mass, and posterior entropy separate objective viability from generation correctness without another model forward.

The locally retained, untracked log `logs/train_20260715_180755.log` covered steps `0..503`: loss fell from `8.8538` to `1.0447` (`-88.2%`), first/last-100 means were `6.7324`/`0.9648`, the minimum was `0.8416` at step 438, no NaN/Inf appeared, and throughput was about `1.076 steps/s`. This establishes early objective optimization only. The top-2 `of`/`and` mass `0.99977`, entropy `0.0927`, and `5.91..7.06` CE advantage are historical diagnostic measurements from the 2026-07-15 session, not a tracked repository artifact or repository-verifiable proof. The causal diagnosis of information-support leakage/mismatch is supported by reproduction, but no repository-verifiable log records those probe values; pre-mixing semantic erasure addresses the reproduced mismatch.

## Architecture inspired by Shannon information theory and 5G NR

### Concept

A language model is treated as a codec and designed with analogies to the
5G NR physical layer. These mappings describe design inspiration only: the
learned real-valued codec does not implement exact LDPC belief propagation,
a probabilistically derived Bayesian filter, or a formal EXIT-chart algorithm.

Source-Channel Separation Theorem: optimal communication is achieved
by separate optimization of source coding (compression) and channel
coding (error correction).

The model is a masked LM (bidirectional, same-position prediction),
not a causal next-token LM. Generation obtains a posterior over a fully
unknown suffix and applies constrained left-to-right token decisions with
adaptive EOS stopping.

---

## Pipeline

`HAGIv4.forward` executes five explicit stage boundaries:
`_source_encode` → `_channel_encode` → `_apply_erasure` → `_channel_decode` → `_source_decode`.
`_source_encode` replaces semantically unknown token embeddings before cache
writes and frequency-domain mixing. `_channel_encode` then derives learned
real-valued redundancy from the clean systematic latent; `_apply_erasure`
corrupts only the received systematic part while preserving that redundancy.
`DecodeState` carries Kalman covariance, HARQ feedback, iteration
marker, and cache intent through the channel decoder. `SpectralCache`
owns boundary context and stores a persistent snapshot of this state.
Local immutable contracts (`codec_contracts.py`) isolate runtime
model/train/inference stages from root config.

The embedding table can remain on CPU at inference for VRAM savings;
cross-device tensor transfers occur only at embedding lookup and lm_head
projection. Fresh embeddings are trainable from scratch. Distillation is
optional but enabled by default in `TrainConfig` and in the canonical profile;
an enabled run requires the configured distillation teacher to be available
locally, which may require network access during prior model setup, and fails
fast when that teacher cannot be loaded. Set `train.distill_enabled: false`
explicitly for a teacher-free baseline.

```
Token IDs
    |
    v
SOURCE ENCODE
  embedding -> semantic unknown replacement -> cache write -> FreqBlock mixing
  -> CQI-controlled frequency bottleneck -> clean systematic latent
    |
    v
CHANNEL ENCODE
  SparseParityEncoder(clean systematic) -> learned real-valued redundancy
  -> concatenate systematic/redundancy -> interleave
    |
    v
APPLY ERASURE
  deinterleave -> AWGN and/or physical replacement on received systematic only
  -> preserve redundancy -> reinterleave
    |
    v
CHANNEL DECODE
  iterative FreqBlock prediction -> sparse parity residual
  -> Kalman-form gated correction -> extrinsic accumulation/HARQBuffer
  -> optional directional-novelty convergence halt
    |
    v
SOURCE DECODE
  frequency-domain expansion C -> H -> FreqBlock mixing -> RMSNorm -> LM head
```

The sparse parity graph, iterative correction, Kalman equations, HARQ buffer,
and convergence proxy are analogies to communications techniques. They are not
exact LDPC, Bayesian inference, HARQ retransmission, or EXIT-chart algorithms.

---

## Components

### 2D FFT replaces Attention (OFDM)

2D rFFT over (T, head_dim) per head — O(T * H * log(T*H)).
FFT = OFDM demodulation, IFFT = OFDM modulation.
Complex weight = MIMO channel equalizer (low-rank, rank=16).
Soft frequency gating = adaptive modulation (5G AMC).

No QKV. No softmax. No RoPE. No causal mask.
Position is encoded by phase in the frequency domain.

### Per-Mode Frequency Response (fading)

Per-mode learnable complex fading in FreqCoding2D:

```python
ch_t = exp(1j * channel_response_t[:F_t])  # learnable, init ~0 (identity)
ch_h = exp(1j * channel_response_h[:F_h])
X_f = X_f * ch_2d  # 2D fading per (T_mode, H_mode)
```

Initialization `std=0.02` → near-identity at start, the model learns
which frequencies matter more.

5G analog: Frequency-selective fading channel + per-subcarrier equalization.

### Kalman-Form Residual Correction

The decoder tracks diagonal state `P` and uses learned, sigmoid-bounded `Q`
and `R` in the Kalman-form equations `P_pred = P + Q`,
`K = P_pred / (P_pred + R)`, and `P = (1-K) * P_pred`. It applies `K` to
the sparse parity residual averaged over checks and passes the correction
through a learned gate. This is a Kalman-filter analogy and implementation
pattern, not a claim that the model computes an exact Bayesian posterior.

### HARQ Buffer

`HARQBuffer` stores extrinsic updates, reads top-ranked stored updates, and
combines them with the current update using tracked uncertainty. Feedback is
serialized through `DecodeState.harq_feedback` for cache continuity. HARQ soft
combining is the communications analogy; this is not retransmission-level
Chase combining or a decision-feedback equalizer.

### HARQ Soft Combining

On decoder iterations after the first, `HARQBuffer.read` supplies stored
extrinsic information and `HARQBuffer.combine` weights it using mean diagonal
Kalman uncertainty:

```python
stored_ext = self.harq.read(z_sys + ext, top_k=self.harq.cfg.top_k)
uncertainty = p.unsqueeze(0).expand(B, T, -1).mean(dim=-1)
ext = self.harq.combine(ext, stored_ext, uncertainty)
```

High uncertainty → more trust in stored state (prior).
Low uncertainty → more trust in current state.

5G analogy: uncertainty-weighted soft combining. The implementation combines
learned extrinsic tensors, not retransmitted codewords.

### Dynamic Bottleneck — AMC Analogy

CQI controls both magnitude gate and bandwidth cutoff:

```python
bw_base = sigmoid(cqi_bw_logit)
bw_scale = bw_base + (1.0 - bw_base) * cqi
cutoff = n_bins * bw_scale
dyn_mask = sigmoid((cutoff - bin_idx) * (1.0 + abs(bw_base) * n_bins))
mag_base = sigmoid(cqi_mag_logit)
gate = base_gate * dyn_mask * (mag_base + (1.0 - mag_base) * cqi)
```

Higher CQI moves the soft cutoff toward the full retained band and increases
the magnitude scale; lower CQI moves both toward their learned bases.

5G analogy: AMC-like CQI control. The implementation changes latent frequency
bandwidth and magnitude; it does not select a standardized modulation and
coding scheme.

### Deterministic Bottleneck (Rate Matching)

Deterministic rFFT truncation H→C + CQI-adaptive gate.
5G rate matching = deterministic puncturing, not stochastic.

### AWGN Noise Injection (training-only)

After channel encoding, Gaussian noise is added to the received systematic
symbols in `_apply_erasure`; learned redundancy remains clean:

```python
if self.training and awgn_sigma > 0.0:
    systematic.add_(awgn_sigma * torch.randn_like(systematic))
```

Sigma annealing: `0.005 → 0.0` linearly until 50% of training.

5G analog: Training with AWGN channel — model robustness to additive noise.

### Sparse Real-Valued Redundancy

`SparseParityEncoder` maps each systematic latent `[B,T,C]` to learned
real-valued redundancy `[B,T,M]` with a fixed sparse connectivity mask,
learned weights, and `RMSNorm`. The systematic and redundancy tensors are
concatenated and interleaved before physical corruption. The decoder-side
`SparseParityChecker` shares the encoder mask, weights, and normalization and
computes `parity_received - parity_computed`. This sparse graph is LDPC-style
inspiration only: values are continuous learned features, and decoding is not
the standardized binary LDPC sum-product or min-sum algorithm.

### Embeddings

Embeddings remain trainable from scratch (`freeze_embeddings=False`), but the
current defaults enable teacher distillation rather than a teacher-free baseline:
`TrainConfig` and `configs/8gb_canonical.yaml` enable optional distillation.
That profile requires the configured distillation teacher to be available
locally, which may require network access during prior model setup, and startup
fails fast when the enabled distillation teacher cannot be loaded. Set
`train.distill_enabled: false` explicitly for a teacher-free baseline. Weight
tying remains supported.

### Masked LM + LLaDA-style Generation

Same-position masked LM (bidirectional). Training mixes random unknowns
and full unknown suffixes 50/50. Generation performs one final refinement
over the fully unknown suffix, returns gathered logits only for prediction
positions, and applies the unified logits processor left-to-right. EOS is
forbidden before two generated tokens and ends each row adaptively through
`max_new_tokens`; shorter rows remain padded in `GenerationOutput`.

Coding analogy: iterative refinement. Generation itself performs constrained
left-to-right decisions from gathered masked-LM posteriors; it is not LDPC
belief propagation.

### Teacher as pilot generator (DM-RS analog)

Teacher (Gemma/SmolLM2) can optionally be used as a pilot/reference signal
generator (5G DM-RS analog). `TrainConfig` and the canonical profile enable
this optional distillation path by default; set `train.distill_enabled: false`
explicitly for a teacher-free baseline:
- **Embedding transfer**: copy (or project) teacher embeddings into
  student. Random projection when hidden_size differs.
- **Hidden state alignment**: MSE(student_hidden, teacher_hidden) on
  original (unmasked) input_ids. Teacher forward on unmasked input
  generates reference hidden states; student aligns its representations
  with teacher. Direction-agnostic — works for causal teacher and
  masked student.

KL distillation on logits is disabled: causal teacher logits (next-token)
conflict with masked LM student (same-position).

### Semantic Erasure Embedding

`semantic_unknown_embed` is learned and randomly initialized as an explicit
erasure indicator. It replaces unknown token embeddings before any mixing.

### Mutation Rank

rank=32 — capacity for noise injection in the turbo loop.

---

## Training Objective And Evidence Metrics

```
Level 1: objective = CE
Level 2: objective = CE + lambda_rate * Rate_distortion
                        + lambda_parity * log1p(Parity)
                        + lambda_contrastive * Contrastive
Level 3: objective = Level 2 + lambda_whiteness * Whiteness
                           + lambda_correction * Correction_alignment
```

The displayed `Loss` is the blended optimization objective and is logged as
`objective_loss` with the compatibility alias `loss`. It must not be used to
derive bits per token. Pure evidence metrics are computed under `no_grad`
from the existing gathered `[N_prediction,V]` logits:

- `masked_ce`: CE over every gathered prediction row.
- `bpt`: exactly `masked_ce / ln(2)`.
- `suffix_ce`: CE only over rows belonging to suffix tasks; NaN when no suffix row exists.
- `top2_mass` and `posterior_entropy`: compact posterior concentration evidence, not generation correctness claims.
- `suffix_task_ratio`, `random_task_ratio`, `semantic_mask_ratio`, and `physical_corruption_ratio`: observed task/channel support.

The staged auxiliary schedule is discrete: level 1 before `warmup_steps`,
level 2 until `2 * warmup_steps`, then level 3.

5G analog: Initial transmission without channel coding (uncoded),
then gradual coding.

### Three-Phase Mask Schedule (AMC analog)

| Phase | Steps | Mask ratio | 5G analog |
|-------|-------|------------|-----------|
| 1 (fitting) | 0–50% | 0.15 | Low SNR training, easy reconstruction |
| 2 (compression) | 50–80% | 0.35 | Medium SNR, less context |
| 3 (LLaDA compat) | 80–100% | 0.50 | High SNR, generation-ready |

Phase 3 prepares the model for 100% mask ratio during LLaDA-style generation.

### Distillation

```
Distill_loss = alpha * CE + (1 - alpha) * MSE(student_hidden, teacher_hidden)
```

Where:
- **CE** = masked cross-entropy on masked positions, same-position targets
- **Parity** = sparse parity-check residual energy accumulated by the decoder
- **Whiteness** = lag-1 autocorrelation penalty (decorrelated residual)
- **Extrinsic_info** = innovation norm, **penalty** (reward convergence)
- **Efficiency** = iterations used (minimize computation)
- **Rate_distortion** = MSE between pre-bottleneck and post-dematch hidden
- **MSE alignment** = hidden state distillation (teacher as DM-RS pilot)

Alpha schedule: linear ramp alpha_start → alpha_end over distill phase,
then 1.0 (CE only). Teacher freed after distill_end_frac.

---

## Inference

### LLaDA-style Iterative Decoding

```
1. Init: prompt + `max_new_tokens` placeholders plus a fully true suffix `semantic_unknown_mask`.
2. Run the configured final refinement and gather logits for suffix prediction positions only.
3. Process each suffix position left-to-right: forbidden IDs, minimum-length EOS gate, repetition penalty, no-repeat n-gram, temperature, top-k, then sampling/argmax.
4. Stop each row at its first valid EOS in `2..max_new_tokens`; pad unused suffix positions.
5. Return `GenerationOutput(token_ids, generated_lengths, finished)`.
```

`max_iterations` controls internal turbo refinement; it does not perform
confidence-ranked repeated full-model generation forwards.

### VRAM Efficiency

Embedding table (83% params, tied with lm_head) can remain on CPU.
Token IDs → CPU embedding lookup → hidden states on GPU for compute →
logits via CPU lm_head projection. VRAM: ~0.5 GB for 230M non-embed
params (bf16) instead of ~10 GB for full model.

---

## 5G NR Physical Layer Mapping

| 5G NR Stage | HAGI Component | File |
|-------------|----------------|------|
| Transport block + CRC | Visible embedding or pre-mixing semantic erasure | hagi_v4.py |
| Pilot/reference signal (DM-RS) | Teacher hidden states (distillation) | distillation.py |
| Scrambling | semantic_unknown_embed (learned erasure indicator) | hagi_v4.py |
| OFDM modulation (IFFT) | torch.fft.irfft2 | freq_layer.py |
| Modulation mapping (QAM) | FreqCoding2D phase modulation | freq_layer.py |
| Layer mapping + precoding | Multi-head + complex weight (rank=16) | freq_layer.py |
| Frequency-selective fading | Per-mode channel_response (learnable) | freq_layer.py |
| AWGN channel | Gaussian noise injection (training, anneal) | hagi_v4.py |
| Rate matching (puncturing) | rFFT bottleneck (H→C) + dynamic CQI bandwidth | hagi_v4.py |
| Channel (fading) | Per-mode complex fading in FreqBlock | freq_layer.py |
| OFDM demodulation (FFT) | torch.fft.rfft2 | freq_layer.py |
| Channel estimate | Kalman filter (P, Q, R) | kalman.py |
| Equalization | Complex weight @ freq modes | freq_layer.py |
| Sparse channel redundancy (LDPC analogy only) | `SparseParityEncoder` learned real-valued projection | sparse_parity.py |
| Sparse parity residual (LDPC analogy only) | `SparseParityChecker` with shared encoder parameters | sparse_parity.py |
| Iterative correction (belief-propagation analogy only) | `LDPCDecoder`: FreqBlock reasoning, residual correction, extrinsic accumulation | hagi_v4.py |
| Kalman-style estimation (not exact Bayesian inference) | Learned diagonal `P`, `Q`, `R` residual update | kalman.py, hagi_v4.py |
| HARQ soft-combining analogy | `HARQBuffer` extrinsic write/read/combine | msa.py |
| EXIT analogy only | Directional-novelty convergence proxy | exit_chart.py, hagi_v4.py |
| Adaptive modulation (AMC) | CQI gate + dynamic bandwidth + soft freq gating | hagi_v4.py, freq_layer.py |
| Rate dematching | rFFT zero-pad (C→H) + CQI gate | hagi_v4.py |
| Demodulation → bits | LM head over gathered prediction states | hagi_v4.py |
| Iterative decoding analogy | Masked-LM posterior generation | generate.py |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.awgn_enabled` | `true` | AWGN noise injection |
| `train.awgn_sigma_start` | `0.005` | Initial noise sigma |
| `train.awgn_sigma_end` | `0.0` | Final sigma (anneal to zero) |
| `train.awgn_end_frac` | `0.5` | Fraction of training for anneal |
| `train.freeze_embeddings` | `false` | Fresh baseline trains embeddings from scratch |
| `train.distill_enabled` | `true` | Enables optional teacher distillation in `TrainConfig` and the canonical profile; set `false` for a teacher-free baseline |
| `model.codec.code_rate` | `0.5` | Configured systematic code-rate target |
| `model.codec.edges_per_check` | `4` | Default sparse inputs per redundancy check |

Per-mode frequency response — learnable parameters in FreqCoding2D:
| Parameter | Init | Description |
|-----------|------|-------------|
| `freq.channel_response_t` | `N(0, 0.02)` | Per-T-mode phase fading |
| `freq.channel_response_h` | `N(0, 0.02)` | Per-H-mode phase fading |
