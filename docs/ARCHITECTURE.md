# HAGI V7.2 — Codec Language Model (5G NR pipeline)

## Architecture based on Shannon information theory and 5G NR

### Concept

A language model is a codec in Shannon's sense, designed by analogy
with the 5G NR physical layer.

Source-Channel Separation Theorem: optimal communication is achieved
by separate optimization of source coding (compression) and channel
coding (error correction).

The model is a masked LM (bidirectional, same-position prediction),
not a causal next-token LM. Generation uses LLaDA-style iterative
decoding: prompt + N mask tokens → iterative filling by confidence
(LDPC belief propagation analog).

---

## Pipeline

`HAGIv4.forward` executes four explicit stage boundaries:
`_source_encode` → `_rate_match` → `_turbo_decode` → `_source_decode`.
`DecodeState` carries Kalman covariance, MSA/DFE feedback, iteration
marker, and cache intent through the turbo decoder. `SpectralCache`
owns boundary context and stores a persistent snapshot of this state.
Local immutable contracts (`codec_contracts.py`) isolate runtime
model/train/inference stages from root config.

Embedding table (83% of parameters) can remain on CPU at inference
for VRAM savings; cross-device tensor transfers occur only at embedding
lookup and lm_head projection. Embeddings are frozen
(`freeze_embeddings=True`): copied from teacher, not trained.

```
Token IDs
    |
    v
+-----------------------------------------------+
|           SOURCE ENCODER                       |
|                                                |
|  Embedding (vocab -> H, FROZEN, CPU offload)   |
|  Masking (BEC: erasure, mask_embed=zero init)  |
|  Pilot equalization (DC removal, spacing=8)    |
|  FreqBlock x2 (2D FFT = OFDM pre-equalization) |
|    + per-mode channel response (fading)        |
|  AWGN noise injection (training-only, anneal)  |
|                                                |
|  5G: transport block + CRC + OFDM mod          |
+----------------------+-------------------------+
                       |
                       v
+-----------------------------------------------+
|         RATE MATCHING (deterministic)          |
|                                                |
|  rFFT(H) -> truncate to C bins                |
|  + CQI-adaptive dynamic bandwidth (AMC)       |
|  + CQI magnitude gate (0.5+0.5*CQI)           |
|  + raised-cosine rolloff                      |
|  -> irFFT(C) + RMSNorm                        |
|  core_mask_embed (erasure in compressed domain)|
|                                                |
|  5G: puncturing/shortening (deterministic)     |
|  CQI-adaptive gate + bandwidth = true AMC      |
+----------------------+-------------------------+
                       |
                       v
+-----------------------------------------------+
|      LDPC ITERATIVE DECODER (turbo loop)       |
|                                                |
|  for iteration in range(n_iters):             |
|                                                |
|    DFE: MSA.read(z)  [iter > 0]                |
|      + HARQ soft combining (uncertainty-weighted)|
|      Channel memory = past decisions cancel ISI|
|      5G: DFE + HARQ Chase Combining            |
|                                                |
|    Component A: FreqBlock (prediction)          |
|      Shared weights across reasoning layers    |
|      2D FFT -> per-mode fading -> soft gate    |
|      -> complex weight -> phase modulation     |
|      -> 2D IFFT                                |
|      5G: OFDM equalization                    |
|                                                |
|    Kalman predict: P_pred = P_prev + Q         |
|      5G: channel estimation, uncertainty grow  |
|                                                |
|    Component B: GP2D (measurement)             |
|      MultiScale geometric product = parity     |
|      Gates init: -4, -5, -6 (conservative)     |
|      5G: LDPC parity check                    |
|                                                |
|    Kalman update:                              |
|      K = P_pred / (P_pred + R)                 |
|      z = z_pred + K * (z_meas - z_pred)        |
|      P = (1 - K) * P_pred                      |
|      5G: optimal Bayesian channel estimation   |
|                                                |
|    SIC: freeze confident positions (training)  |
|      confidence = 1/(1+|GP2D residual|)        |
|      sic_gate = sigmoid(conf*sic_w + sic_b)    |
|      z = z*(1-gate) + z.detach()*gate          |
|      5G: Successive Interference Cancellation  |
|                                                |
|    Mutation: rank-32 bottleneck perturbation   |
|      5G: noise injection for exploration       |
|                                                |
|    tanh clipping: z = scale * tanh(z / scale)  |
|      5G: LLR clipping in LDPC decoding         |
|                                                |
|    HARQ: MSA.write(z)                          |
|      Store parity-checked state for combining  |
|      5G: HARQ buffer (Chase combining)         |
|                                                |
|    Convergence: ||innovation|| < eps -> stop   |
|      5G: EXIT chart stopping criterion         |
|                                                |
+----------------------+-------------------------+
                       |
                       v
+-----------------------------------------------+
|         RATE DEMATCHING                        |
|                                                |
|  rFFT(C) -> zero-pad to H bins + CQI gate     |
|  -> irFFT(H)                                   |
|  FreqBlock x2 (expression = shared perception) |
|                                                |
|  5G: rate dematching (de-puncturing)           |
+----------------------+-------------------------+
                       |
                       v
+-----------------------------------------------+
|         SOURCE DECODER (demodulation)          |
|                                                |
|  RMSNorm -> LM Head (H -> vocab)              |
|  Weight tying: lm_head.weight = embed.weight  |
|  (frozen — no gradients to embeddings)         |
|  CE on masked positions only (same-position)   |
|                                                |
|  5G: demodulation -> bits                     |
+-----------------------------------------------+
```

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

### Kalman Filter

Optimal Bayesian blend via Kalman gain.
Kalman gain is adaptive: high uncertainty → trust measurement (GP2D).
Low uncertainty → trust prediction (FreqBlock).

Diagonal covariance: O(C) per iteration, negligible parameter count.

### Channel Memory (MSA) = DFE + HARQ

MSA has two explicit roles:
- **DFE (Decision Feedback Equalizer)**: read from previous iterations
  cancels ISI. Past decisions = feedback signal.
- **HARQ buffer**: write after parity check. Stored states combined
  across iterations = Chase combining.

MSA ring buffer: `TensorSlotRegistry` with monotonic `num_written` counter,
correctly handling wrap-around. Serialize/restore feedback via
`DecodeState.msa_feedback` for cache continuity.

### HARQ Soft Combining

MSA.read is combined with previous residual, weighted by uncertainty:

```python
uncertainty = prev_residual.abs().mean(dim=-1)
harq_alpha = sigmoid(harq_gate) * sigmoid(uncertainty * 10.0)
z = z + harq_alpha * msa_out  # harq_gate=-5: nearly OFF at start
```

High uncertainty → more trust in stored state (prior).
Low uncertainty → more trust in current state.

5G analog: HARQ Chase Combining — soft combining of retransmitted codewords.

### SIC — Successive Interference Cancellation

After each turbo loop iteration, per-position confidence is computed
from GP2D residual. Confident positions are frozen (detach gradient),
gradients flow only to uncertain positions.

```python
confidence = 1.0 / (1.0 + gp2d_residual.abs().mean(dim=-1))
sic_gate = sigmoid(confidence * sic_w + sic_b)  # sic_b=-10: nearly OFF at start
z = z * (1 - gate) + z.detach() * gate
```

Initialization `sic_b=-10` → `sigmoid(-10)≈0.00005` — SIC is nearly off
at start, the model activates it through learning.

5G analog: SIC — decode strong signal → subtract → decode weak.

### Dynamic Bottleneck — true AMC

CQI controls both magnitude gate and bandwidth cutoff:

```python
bw_scale = 1.0 - 0.15 * cqi           # high CQI → fewer bins (more compression)
cutoff = n_bins * bw_scale
dyn_mask = sigmoid((cutoff - bin_idx) * 6.0)
gate = base_gate * dyn_mask * (0.5 + 0.5 * cqi)
```

High CQI → fewer bins = more compression (good channel).
Low CQI → more bins = more redundancy (bad channel).

5G analog: True AMC — CQI determines both modulation order and code rate.

### Deterministic Bottleneck (Rate Matching)

Deterministic rFFT truncation H→C + CQI-adaptive gate.
5G rate matching = deterministic puncturing, not stochastic.

### AWGN Noise Injection (training-only)

Gaussian noise is added to hidden states after FreqBlock
in the source encoder:

```python
if self.training and awgn_sigma > 0.0:
    h = h + awgn_sigma * torch.randn_like(h)
```

Sigma annealing: `0.005 → 0.0` linearly until 50% of training.

5G analog: Training with AWGN channel — model robustness to additive noise.

### MultiScale GP2D

Multi-scale geometric product with interleaving for burst error protection:
- Scale 1 (window=1): adjacent parity — high-freq errors
- Scale 2 (window=4): mid-range parity — burst errors
- Scale 3 (window=16): long-range parity — structural errors

Gates init: `-4.0, -5.0, -6.0` (sigmoid → 0.02, 0.007, 0.002).
GP2D is nearly off at start — does not dominate before the model
learns basic representations. Activated gradually through learning.

### Frozen Embeddings

`freeze_embeddings=True` — embeddings are copied from teacher
(SmolLM2-135M / Gemma) and frozen. Gradients do not flow into the
embedding table, `lm_head.weight = embed.weight` (tied, frozen).

VRAM savings: no optimizer state for the largest parameter group.
Quality: teacher embeddings already contain semantic structure,
no need to relearn.

### Masked LM + LLaDA-style Generation

Same-position masked LM (bidirectional). Generation via
LLaDA-style iterative decoding:
1. Create prompt + N mask tokens
2. Forward pass → logits for all mask positions
3. Fill confident positions (top-50% by confidence)
4. Repeat forward, gradually filling remaining positions
5. Stop when all masks replaced or max_iterations reached

5G analog: LDPC belief propagation — fill confident positions first,
using them as additional pilot signals for the remaining ones.

### Teacher as pilot generator (DM-RS analog)

Teacher (Gemma/SmolLM2) is used as a pilot/reference signal
generator (5G DM-RS analog):
- **Embedding transfer**: copy (or project) teacher embeddings into
  student. Random projection when hidden_size differs.
- **Hidden state alignment**: MSE(student_hidden, teacher_hidden) on
  original (unmasked) input_ids. Teacher forward on unmasked input
  generates reference hidden states; student aligns its representations
  with teacher. Direction-agnostic — works for causal teacher and
  masked student.

KL distillation on logits is disabled: causal teacher logits (next-token)
conflict with masked LM student (same-position).

### mask_embed Init

`zero_()` — mask embedding starts with zero norm,
the model learns the optimal mask signal independently.

### Mutation Rank

rank=32 — capacity for noise injection in the turbo loop.

---

## Training Objective

```
Loss = CE                                    # Fidelity: same-position masked prediction
      + aux_w(step) * lambda_1 * sigmoid(Parity * 5)    # Redundancy: soft-clamped (subtracted)
      + aux_w(step) * lambda_2 * Whiteness               # GP2D residual autocorrelation
      + aux_w(step) * lambda_3 * Extrinsic_info          # Convergence penalty (innovation norm)
      + aux_w(step) * lambda_4 * Efficiency              # Iterations used (minimize computation)
      + aux_w(step) * lambda_5 * Rate_distortion         # Reconstruction MSE (pre/post bottleneck)
      + aux_w(step) * lambda_6 * MSA_load_balance        # Slot usage uniformity
      + aux_w(step) * lambda_7 * Contrastive             # Modality alignment (multimodal only)
```

Where `aux_w(step)`:
- `step < warmup_steps` → `0.0` (Phase 0: CE only)
- `warmup_steps ≤ step < 2*warmup_steps` → linear ramp `0→1`
- `step ≥ 2*warmup_steps` → `1.0` (full loss)

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
- **Parity** = GP2D residual energy, soft-clamped via sigmoid (subtract = reward)
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
1. Init: prompt + max_new_tokens mask tokens
2. For iteration in range(max_iterations):
   a. Run full pipeline: Embed → FreqBlock → Bottleneck → Turbo → Up → LM
   b. Get logits at all mask positions
   c. Compute confidence per mask position
   d. Fill top-50% confident positions (last iteration: fill all remaining)
   e. Ban mask_token_id and padding from logits
   f. Apply echo cancellation (repetition penalty) on visible tokens
3. Stop when all mask filled or EOS detected
```

5G analog: LDPC belief propagation — iterative filling of confident
positions, using filled ones as additional pilot signals for the
remaining. Turbo loop iterations controllable via `max_iterations`.

### VRAM Efficiency

Embedding table (83% params, tied with lm_head) can remain on CPU.
Token IDs → CPU embedding lookup → hidden states on GPU for compute →
logits via CPU lm_head projection. VRAM: ~0.5 GB for 230M non-embed
params (bf16) instead of ~10 GB for full model.

---

## 5G NR Physical Layer Mapping

| 5G NR Stage | HAGI Component | File |
|-------------|----------------|------|
| Transport block + CRC | Embedding + Masking (zero init, frozen) | hagi_v4.py |
| Pilot/reference signal (DM-RS) | Teacher hidden states (distillation) | distillation.py |
| Scrambling | mask_embed (zero init, learnable) | hagi_v4.py |
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
| LDPC iterative decode | TurboLoop (4 iterations) | hagi_v4.py |
| Parity check | MultiScaleGP2D (gates init -4,-5,-6) | gp2d.py |
| Successive Interference Cancellation | SIC: freeze confident positions | hagi_v4.py |
| Noise injection | Mutation (rank=32 bottleneck) | hagi_v4.py |
| LLR clipping | tanh scaling (scale=10) | hagi_v4.py |
| HARQ buffer (Chase combining) | MSA.write + soft combining (uncertainty-weighted) | msa.py |
| Decision Feedback Equalizer | MSA.read | msa.py |
| EXIT chart stopping | Convergence halt (innovation norm) | hagi_v4.py |
| Adaptive modulation (AMC) | CQI gate + dynamic bandwidth + soft freq gating | hagi_v4.py, freq_layer.py |
| Rate dematching | rFFT zero-pad (C→H) + CQI gate | hagi_v4.py |
| Demodulation → bits | LM head (tied embeddings, frozen) | hagi_v4.py |
| Belief propagation | LLaDA iterative decoding | generate.py |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.awgn_enabled` | `true` | AWGN noise injection |
| `train.awgn_sigma_start` | `0.005` | Initial noise sigma |
| `train.awgn_sigma_end` | `0.0` | Final sigma (anneal to zero) |
| `train.awgn_end_frac` | `0.5` | Fraction of training for anneal |
| `train.freeze_embeddings` | `true` | Freeze embedding table |
| `model.gp2d.multiscale_gate_inits` | `[-4, -5, -6]` | Conservative GP2D gates |

SIC and HARQ soft combining — learnable parameters in TurboLoop:
| Parameter | Init | Description |
|-----------|------|-------------|
| `turbo.sic_w` | `10.0` | SIC confidence weight |
| `turbo.sic_b` | `-10.0` | SIC bias (OFF at start) |
| `turbo.harq_gate` | `-5.0` | HARQ combining gate (OFF at start) |

Per-mode frequency response — learnable parameters in FreqCoding2D:
| Parameter | Init | Description |
|-----------|------|-------------|
| `freq.channel_response_t` | `N(0, 0.02)` | Per-T-mode phase fading |
| `freq.channel_response_h` | `N(0, 0.02)` | Per-H-mode phase fading |
