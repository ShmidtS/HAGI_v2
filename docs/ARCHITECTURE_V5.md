# HAGI V7.1 — Codec Language Model (5G NR pipeline)

## Архитектура на основе теории связи (Shannon) и 5G NR

### Концепция

Языковая модель — это кодек в смысле Шеннона, спроектированный по
аналогии с physical layer 5G NR. Прошли путь от V4 (transformer+HRM)
через V5 (extrinsic separation) к V7.1 (2D FFT OFDM + Kalman + turbo +
LLaDA-style generation).

Теорема разделения (Source-Channel Separation Theorem): оптимальная
коммуникация достигается раздельной оптимизацией source coding
(компрессия) и channel coding (помехоустойчивость).

Модель — masked LM (bidirectional, same-position prediction), не causal
next-token LM. Генерация — LLaDA-style iterative decoding: prompt + N
mask tokens → итеративное заполнение по confidence (LDPC belief
propagation analog).

---

## Pipeline (V7.1)

`HAGIv4.forward` исполняет четыре явные stage-границы:
`_source_encode` → `_rate_match` → `_turbo_decode` → `_source_decode`.
`DecodeState` передаёт Kalman covariance, MSA/DFE feedback, маркер
итерации и cache intent через turbo decoder. `SpectralCache` владеет
boundary context и хранит persistent snapshot этого состояния.
Локальные immutable contracts (`codec_contracts.py`) изолируют runtime
model/train/inference stages от root config.

Embedding table (83% параметров) может оставаться на CPU при inference
для экономии VRAM; cross-device tensor transfers происходят только на
embedding lookup и lm_head projection.

```
Token IDs
    |
    v
+-----------------------------------------------+
|           SOURCE ENCODER                       |
|                                                |
|  Embedding (vocab -> H)                        |
|  Masking (BEC: erasure, mask_embed=zero init)  |
|  Pilot equalization (DC removal, spacing=8)    |
|  FreqBlock x2 (2D FFT = OFDM pre-equalization) |
|                                                |
|  5G: transport block + CRC + OFDM mod          |
+----------------------+-------------------------+
                       |
                       v
+-----------------------------------------------+
|         RATE MATCHING (deterministic)          |
|                                                |
|  rFFT(H) -> truncate to C bins + CQI gate     |
|  -> irFFT(C) + RMSNorm                        |
|  core_mask_embed (erasure in compressed domain)|
|                                                |
|  5G: puncturing/shortening (deterministic)     |
|  CQI-adaptive gate = AMC (Adaptive Modulation) |
+----------------------+-------------------------+
                       |
                       v
+-----------------------------------------------+
|      LDPC ITERATIVE DECODER (turbo loop)       |
|                                                |
|  for iteration in range(n_iters):             |
|                                                |
|    DFE: MSA.read(z)  [iter > 0]                |
|      Channel memory = past decisions cancel ISI|
|      5G: Decision Feedback Equalizer           |
|                                                |
|    Component A: FreqBlock (prediction)          |
|      Shared weights across reasoning layers    |
|      2D FFT -> soft gate -> complex weight     |
|      -> phase modulation -> 2D IFFT            |
|      5G: OFDM equalization                    |
|                                                |
|    Kalman predict: P_pred = P_prev + Q         |
|      5G: channel estimation, uncertainty grow  |
|                                                |
|    Component B: GP2D (measurement)             |
|      MultiScale geometric product = parity     |
|      5G: LDPC parity check                    |
|                                                |
|    Kalman update:                              |
|      K = P_pred / (P_pred + R)                 |
|      z = z_pred + K * (z_meas - z_pred)        |
|      P = (1 - K) * P_pred                      |
|      5G: optimal Bayesian channel estimation   |
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
|  CE on masked positions only (same-position)   |
|                                                |
|  5G: demodulation -> bits                     |
+-----------------------------------------------+
```

---

## Ключевые инновации V7.1 vs V5

### 1. 2D FFT заменяет Attention (OFDM)

**V5**: QKV attention O(T^2 * H) — квадратичная сложность.

**V7.1**: 2D rFFT over (T, head_dim) per head — O(T * H * log(T*H)).
FFT = OFDM demodulation, IFFT = OFDM modulation.
Complex weight = MIMO channel equalizer (low-rank, rank=16).
Soft frequency gating = adaptive modulation (5G AMC).

Нет QKV. Нет softmax. Нет RoPE. Нет causal mask.
Позиция кодируется фазой в frequency domain.

### 2. Kalman Filter заменяет alpha extrinsic

**V5**: `h = h_prior + alpha * extrinsic` — фиксированный scalar alpha.

**V7.1**: Оптимальный Bayesian blend через Kalman gain.
Kalman gain адаптивен: высокая неопределённость → доверие measurement
(GP2D). Низкая неопределённость → доверие prediction (FreqBlock).

Diagonal covariance: O(C) per iteration, пренебрежимо мало параметров.

### 3. Channel Memory (MSA) = DFE + HARQ

**V7.1**: MSA имеет две явные роли:
- **DFE (Decision Feedback Equalizer)**: read из предыдущих итераций
  отменяет ISI. Past decisions = feedback signal.
- **HARQ buffer**: write после parity check. Stored states combined
  across iterations = Chase combining.

MSA ring buffer: `TensorSlotRegistry` с monotonic `num_written` counter,
корректно работающим после wrap-around. Serialize/restore feedback
через `DecodeState.msa_feedback` для cache continuity.

### 4. Deterministic Bottleneck (Rate Matching)

**V5**: Variational Information Bottleneck (VIB) — stochastic encoding.

**V7.1**: Deterministic rFFT truncation H→C + CQI-adaptive gate.
5G rate matching = deterministic puncturing, не stochastic.

### 5. Masked LM + LLaDA-style Generation

**V5/V7**: Next-token prediction (causal LM).

**V7.1**: Same-position masked LM (bidirectional). Generation через
LLaDA-style iterative decoding:
1. Создать prompt + N mask tokens
2. Forward pass → logits для всех mask позиций
3. Заполнить confident positions (top-50% по confidence)
4. Повторять forward, постепенно заполняя оставшиеся
5. Останавливаться когда все mask заменены или max_iterations

5G analog: LDPC belief propagation — заполняем уверенные позиции первыми,
используя их как additional pilot signals для оставшихся.

### 6. Teacher как задающий генератор (DM-RS analog)

**V7.1**: Teacher (Gemma/SmolLM2) используется как pilot/reference signal
generator (5G DM-RS analog):
- **Embedding transfer**: копирование (или projection) teacher embeddings
  в student. Random projection при несовпадающих hidden_size.
- **Hidden state alignment**: MSE(student_hidden, teacher_hidden) на
  оригинальных (не masked) input_ids. Teacher forward на unmasked input
  генерирует reference hidden states; student выравнивает свои
  представления с teacher. Direction-agnostic — работает для causal
  teacher и masked student.

KL distillation на logits отключена: causal teacher logits (next-token)
конфликтуют с masked LM student (same-position).

### 7. Three-Phase Training Schedule (AMC analog)

**V7.1**: Three-phase mask ratio schedule для LLaDA generation compatibility:

| Phase | Steps | Mask ratio | 5G analog |
|-------|-------|------------|-----------|
| 1 (fitting) | 0–50% | 0.15 | Low SNR training, easy reconstruction |
| 2 (compression) | 50–80% | 0.35 | Medium SNR, less context |
| 3 (LLaDA compat) | 80–100% | 0.50 | High SNR, generation-ready |

Phase 3 готовит модель к 100% mask ratio при LLaDA-style generation.

### 8. Loss Changes

| Loss term | V7 | V7.1 | Причина |
|---|---|---|---|
| CE | same-position masked | same (исправлено с shifted) | Masked LM |
| Parity | `clamp(0,1)` subtract | `sigmoid(parity*5)` subtract | Soft clamp |
| Extrinsic_info | subtract (reward) | **add** (penalty) | Reward convergence, not divergence |
| Whiteness | add | add | Без изменений |
| Efficiency | add | add | Без изменений |
| Rate_distortion | add | add | Без изменений |
| Contrastive | add | add | Multimodal only |

### 9. mask_embed Init

**V7**: `mean(embed.weight)` — mask embedding имеет высокую norm,
доминирует в training.

**V7.1**: `zero_()` — mask embedding стартует с нулевой norm,
модель выучивает оптимальный mask signal самостоятельно.

### 10. Mutation Rank

**V7**: rank=8 — слишком слабый signal для exploration.

**V7.1**: rank=32 — больше capacity для noise injection в turbo loop.

---

## Component Mapping (V5 -> V7.1)

| V5 компонент | V7.1 роль | Изменения |
|---|---|---|
| Embedding | Source Encoder | Cross-device (CPU offload) |
| Masking (BEC) | Erasure Channel | mask_embed = zero init |
| Perception (Attention) | OFDM Pre-equalization | FreqBlock (2D FFT) заменяет attention |
| VIB Bottleneck | Rate Matching | rFFT truncate + CQI gate (deterministic) |
| HRM refinement | LDPC Turbo Decoder | TurboLoop: FreqBlock + GP2D + Kalman |
| Extrinsic alpha | Kalman Gain | Optimal Bayesian, не scalar |
| MSA | DFE + HARQ | Ring buffer + serialize/restore feedback |
| GP2D | Parity Check | MultiScaleGP2D, sigmoid soft clamp |
| GDR | УДАЛЕН | Нет 5G аналога |
| MoE | УДАЛЕН | FreqBlock equalization заменяет |
| Expression layers | Shared with perception | Weight sharing (encode/decode) |
| Coherence head | УДАЛЕН | CRC (hard check) в 5G |
| Deep supervision | УДАЛЕН | Нет intermediate loss в 5G |
| Water filling | УДАЛЕН | Dead code |
| z_H/z_L state | УДАЛЕН | 5G iterations stateless |
| KL distillation | Hidden state MSE | Teacher = DM-RS pilot generator |

---

## Training Objective (V7.1)

```
Loss = CE                                    # Fidelity: same-position masked prediction
      + lambda_1 * sigmoid(Parity * 5)       # Redundancy: soft-clamped GP2D energy (subtracted)
      + lambda_2 * Whiteness                 # GP2D residual autocorrelation
      + lambda_3 * Extrinsic_info            # Convergence penalty (innovation norm)
      + lambda_4 * Efficiency                # Iterations used (minimize computation)
      + lambda_5 * Rate_distortion           # Reconstruction MSE (pre/post bottleneck)
      + lambda_6 * MSA_load_balance          # Slot usage uniformity
      + lambda_7 * Contrastive               # Modality alignment (multimodal only)
```

Distillation (если включена):
```
Distill_loss = alpha * CE + (1 - alpha) * MSE(student_hidden, teacher_hidden)
```

Где:
- **CE** = masked cross-entropy на masked позициях, same-position targets
- **Parity** = GP2D residual energy, soft-clamped через sigmoid (subtract = reward)
- **Whiteness** = lag-1 autocorrelation penalty (decorrelated residual)
- **Extrinsic_info** = innovation norm, **penalty** (reward convergence)
- **Efficiency** = iterations used (minimize computation)
- **Rate_distortion** = MSE между pre-bottleneck и post-dematch hidden
- **MSE alignment** = hidden state distillation (teacher как DM-RS pilot)

Alpha schedule: linear ramp alpha_start → alpha_end over distill phase,
then 1.0 (CE only). Teacher freed after distill_end_frac.

---

## Инференс (V7.1)

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

5G analog: LDPC belief propagation — итеративное заполнение confident
positions, используя заполненные как additional pilot signals для
оставшихся. Turbo loop iterations controllable via `max_iterations`.

### VRAM Efficiency

Embedding table (83% params, tied с lm_head) может оставаться на CPU.
Token IDs → CPU embedding lookup → hidden states на GPU для compute →
logits через CPU lm_head projection. VRAM: ~0.5 GB для 230M non-embed
params (bf16) вместо ~10 GB для full model.

---

## 5G NR Physical Layer Mapping (полная таблица)

| 5G NR Stage | HAGI V7.1 Component | Файл |
|-------------|---------------------|------|
| Transport block + CRC | Embedding + Masking (zero init) | hagi_v4.py |
| Pilot/reference signal (DM-RS) | Teacher hidden states (distillation) | distillation.py |
| Scrambling | mask_embed (zero init, learnable) | hagi_v4.py |
| OFDM modulation (IFFT) | torch.fft.irfft2 | freq_layer.py |
| Modulation mapping (QAM) | FreqCoding2D phase modulation | freq_layer.py |
| Layer mapping + precoding | Multi-head + complex weight (rank=16) | freq_layer.py |
| Rate matching (puncturing) | rFFT bottleneck (H→C) + CQI gate | hagi_v4.py |
| Channel (AWGN, fading) | (implicit, training noise + masking) | — |
| OFDM demodulation (FFT) | torch.fft.rfft2 | freq_layer.py |
| Channel estimate | Kalman filter (P, Q, R) | kalman.py |
| Equalization | Complex weight @ freq modes | freq_layer.py |
| LDPC iterative decode | TurboLoop (4 iterations) | hagi_v4.py |
| Parity check | MultiScaleGP2D (sigmoid soft clamp) | gp2d.py |
| Noise injection | Mutation (rank=32 bottleneck) | hagi_v4.py |
| LLR clipping | tanh scaling (scale=10) | hagi_v4.py |
| HARQ buffer (Chase combining) | MSA.write + serialize/restore | msa.py |
| Decision Feedback Equalizer | MSA.read | msa.py |
| EXIT chart stopping | Convergence halt (innovation norm) | hagi_v4.py |
| Adaptive modulation (AMC) | CQI gate + soft frequency gating | hagi_v4.py, freq_layer.py |
| Rate dematching | rFFT zero-pad (C→H) + CQI gate | hagi_v4.py |
| Demodulation → bits | LM head (tied embeddings) | hagi_v4.py |
| Belief propagation | LLaDA iterative decoding | generate.py |
