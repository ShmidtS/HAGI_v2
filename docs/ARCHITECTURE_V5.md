# HAGI V7 — Codec Language Model (5G NR pipeline)

## Архитектура на основе теории связи (Shannon) и 5G NR

### Концепция

Языковая модель — это кодек в смысле Шеннона, спроектированный по
аналогии с physical layer 5G NR. Прошли путь от V4 (transformer+HRM)
через V5 (extrinsic separation) к V7 (2D FFT OFDM + Kalman + turbo).

Теорема разделения (Source-Channel Separation Theorem): оптимальная
коммуникация достигается раздельной оптимизацией source coding
(компрессия) и channel coding (помехоустойчивость).

---

## Pipeline (V7)

```
Token IDs
    |
    v
+-----------------------------------------------+
|           SOURCE ENCODER                       |
|                                                |
|  Embedding (vocab -> H=576)                   |
|  Masking (BEC: erasure, mask_embed)            |
|  FreqBlock x2 (2D FFT = OFDM pre-equalization)|
|                                                |
|  5G: transport block + CRC + OFDM mod         |
+----------------------+-------------------------+
                       |
                       v
+-----------------------------------------------+
|         RATE MATCHING (deterministic)          |
|                                                |
|  Linear(H -> C=288) + RMSNorm                 |
|  core_mask_embed (erasure in compressed domain)|
|                                                |
|  5G: puncturing/shortening (deterministic)    |
|  V5 VIB удалён: stochastic noise не нужен     |
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
|    Component A: FreqBlock x7 (prediction)      |
|      2D FFT -> soft gate -> complex weight     |
|      -> phase modulation -> 2D IFFT            |
|      5G: OFDM equalization                    |
|                                                |
|    Kalman predict: P_pred = P_prev + Q         |
|      5G: channel estimation, uncertainty grow  |
|                                                |
|    Component B: GP2D (measurement)             |
|      Geometric product = parity check          |
|      5G: LDPC parity check                    |
|                                                |
|    Kalman update:                              |
|      K = P_pred / (P_pred + R)                 |
|      z = z_pred + K * (z_meas - z_pred)        |
|      P = (1 - K) * P_pred                      |
|      5G: optimal Bayesian channel estimation   |
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
|  Linear(C=288 -> H=576)                       |
|  5G: rate dematching (de-puncturing)          |
+----------------------+-------------------------+
                       |
                       v
+-----------------------------------------------+
|         SOURCE DECODER (demodulation)          |
|                                                |
|  RMSNorm -> LM Head (H -> vocab)              |
|  Weight tying: lm_head.weight = embed.weight  |
|  5G: demodulation -> bits                     |
+-----------------------------------------------+
```

---

## Ключевые инновации V7 vs V5

### 1. 2D FFT заменяет Attention (OFDM)

**V5**: QKV attention O(T^2 * H) — квадратичная сложность.

**V7**: 2D FFT over (T, head_dim) per head — O(T * H * log(T*H)).
FFT = OFDM demodulation, IFFT = OFDM modulation.
Complex weight = MIMO channel equalizer.
Soft frequency gating = adaptive modulation (5G AMC).

Нет QKV. Нет softmax. Нет RoPE. Нет causal mask.
Позиция кодируется фазой в frequency domain.

```python
# V7 FreqCoding2D
X_f = torch.fft.fft2(h)              # OFDM demod
gate = sigmoid(learnable_freq_gates)  # adaptive modulation
X_f = X_f * gate                      # frequency-selective filtering
X_f[:Kt] = X_f[:Kt] * phase           # PSK
X_f[:Kt] = X_f[:Kt] @ complex_weight  # MIMO equalizer
x = torch.fft.ifft2(X_f).real         # OFDM mod
```

### 2. Kalman Filter заменяет alpha extrinsic

**V5**: `h = h_prior + alpha * extrinsic` — фиксированный scalar alpha.

**V7**: Оптимальный Bayesian blend через Kalman gain.
Kalman gain адаптивен: высокая неопределённость → доверие measurement
(GP2D). Низкая неопределённость → доверие prediction (FreqBlock).

```python
# V7 Kalman
K = P_pred / (P_pred + R)              # optimal gain
z = z_pred + K * (z_meas - z_pred)     # Bayesian update
P = (1 - K) * P_pred                   # covariance update
```

Diagonal covariance: O(C) per iteration, +574 параметра.

### 3. Channel Memory (MSA) = DFE + HARQ

**V5**: MSA = "long-range parity checks" (vague role).

**V7**: MSA имеет две явные роли:
- **DFE (Decision Feedback Equalizer)**: read из предыдущих итераций
  отменяет ISI. Past decisions = feedback signal.
- **HARQ buffer**: write после parity check. Stored states combined
  across iterations = Chase combining.

```python
# V7 turbo loop
if iteration > 0:
    z = z + msa.read(z)    # DFE: cancel ISI from past decisions
z = reasoning(z)            # predict
z = gp2d(z)                 # measure (parity)
msa.write(z)                # HARQ: store for next iteration
```

### 4. Deterministic Bottleneck (Rate Matching)

**V5**: Variational Information Bottleneck (VIB) — stochastic encoding
z = mu + eps * sigma, KL loss.

**V7**: Deterministic Linear(H -> C). 5G rate matching = deterministic
puncturing, не stochastic. VIB добавлял noise без benefit.

### 5. Удалённые компоненты (no 5G analog)

| Компонент | Причина удаления |
|-----------|-----------------|
| GDR (grade decomposition) | 5G использует uniform modulation per layer |
| z_H/z_L state machine | 5G iterations stateless (no recurrent state) |
| Water filling allocator | Dead code, не вызывается в forward |
| Coherence head | 5G использует CRC (hard check), gate=-5 (off) |
| Deep supervision | 5G LDPC не имеет intermediate loss |
| Perception/Expression split | 5G один pipeline, не два |
| VIB (stochastic) | 5G rate matching deterministic |
| MoE (dense) | Убран в пользу FreqBlock equalization |

### 6. Soft Frequency Gating (Adaptive Modulation)

**V5**: Hard cutoff — только K_t temporal modes сохранялись, 98% терялось.

**V7**: Learnable sigmoid gates per frequency mode.
Модель сама выбирает какие моды сохранять.
Низкие частоты (gates ~1) — сохраняются.
Высокие частоты (gates ~0) — подавляются, но могут быть открыты обучением.

```python
# V7 soft gating
gate_t = sigmoid(learnable_logits_t[:F_t])  # [F_t]
gate_h = sigmoid(learnable_logits_h[:F_h])  # [F_h]
gate_2d = gate_t[:, None] * gate_h[None, :]  # [F_t, F_h]
out_f = X_f * gate_2d                        # soft filtering
```

---

## Component Mapping (V5 -> V7)

| V5 компонент | V7 роль | Изменения |
|---|---|---|
| Embedding | Source Encoder | Без изменений |
| Masking (BEC) | Erasure Channel | mask_embed = mean embedding |
| Perception (Attention) | OFDM Pre-equalization | FreqBlock (2D FFT) заменяет attention |
| VIB Bottleneck | Rate Matching | Linear (deterministic), не stochastic |
| HRM refinement | LDPC Turbo Decoder | TurboLoop: FreqBlock + GP2D + Kalman |
| Extrinsic alpha | Kalman Gain | Optimal Bayesian, не scalar |
| MSA | DFE + HARQ | Channel memory: read=DFE, write=HARQ |
| GP2D | Parity Check | MultiScaleGP2D, без изменений |
| GDR | УДАЛЕН | Нет 5G аналога |
| MoE | УДАЛЕН | FreqBlock equalization заменяет |
| Expression layers | УДАЛЕН | 5G один pipeline |
| Coherence head | УДАЛЕН | CRC (hard check) в 5G |
| Deep supervision | УДАЛЕН | Нет intermediate loss в 5G |
| Water filling | УДАЛЕН | Dead code |
| z_H/z_L state | УДАЛЕН | 5G iterations stateless |

---

## Training Objective (V7)

```
Loss = CE                              # Fidelity: correctly predict masked tokens
      + lambda_1 * Parity_strength     # Redundancy: ||GP(h)||^2 (clamped max=1.0)
      + lambda_2 * Whiteness           # GP2D residual autocorrelation
      - lambda_3 * Extrinsic_info      # Decoding: innovation norm (reward convergence)
      + lambda_4 * Efficiency          # Convergence: iterations used
```

Где:
- **CE** = masked cross-entropy (fidelity)
- **Parity** = GP2D residual energy (redundancy for error correction)
- **Whiteness** = lag-1 autocorrelation penalty (decorrelated residual)
- **Extrinsic_info** = innovation norm (reward new information per iteration)
- **Efficiency** = iterations used (minimize computation)

Удалённые loss terms:
- IB/KL loss (VIB удалён)
- Coherence loss (coherence head удалён)
- Deep supervision loss (deep supervision удалён)
- GDR router loss (GDR удалён)
- MoE load balance (MoE удалён)
- Grade specialization (GDR удалён)

---

## Параметры (V7 vs V5)

| Метрика | V5 (оригинал) | V7 (5G pipeline) | Разница |
|---------|---------------|-------------------|---------|
| Параметры | 65,088,833 | 50,373,340 | -23% |
| ms/step (GPU) | 804 | 249 | -69% |
| VRAM | ~1261 MB | 1192 MB | -5% |
| bpt@step40 | 12.18 | 11.22 | -8% |

---

## Конфигурация (V7)

```yaml
model:
  vocab_size: 49154
  hidden_size: 576          # H = full bandwidth
  core_hidden_size: 288     # C = channel capacity (2:1 compression)
  perception_layers: 2      # OFDM pre-equalization
  reasoning_layers: 7       # LDPC iterative decode
  expression_layers: 2      # (не используется в V7, legacy)

  freq_coding:
    enabled: true
    n_modes_t: 16           # temporal frequency modes (OFDM subcarriers)
    n_modes_h: 12           # feature frequency modes

  refinement:
    num_iterations: 4       # turbo decode iterations
    min_iterations: 1
    extrinsic_alpha: 0.5    # (legacy, Kalman заменяет)
    convergence_threshold: 0.01

  gp2d:
    use_multiscale: true
    multiscale_windows: [1, 4, 16]  # LDPC parity check degrees

  msa:
    max_slots: 4096         # HARQ buffer size
    top_k: 6                # DFE feedback taps

train:
  learning_rate: 0.0003
  w_parity: 0.1
  w_whiteness: 0.01
  w_extrinsic_info: 0.01
  w_efficiency: 0.001
```

---

## Инференс (V7)

### Masked Autoregressive Generation

```
1. Init: prompt + mask_token (generation position)
2. For each block of n_block tokens:
   a. Run full pipeline: Embed -> FreqBlock -> Bottleneck -> Turbo -> Up -> LM
   b. Get logits at masked positions
   c. Apply echo cancellation (repetition penalty)
   d. Sample/argmax from logits
   e. Refine: re-run with filled tokens, blend logits (0.5/0.5)
   f. Append to sequence
3. Stop at EOS or max_tokens
```

Turbo loop iterations controllable via `max_iterations` parameter.
At inference: convergence halt uses innovation norm (Kalman residual).

---

## 5G NR Physical Layer Mapping (полная таблица)

| 5G NR Stage | HAGI V7 Component | Файл |
|-------------|-------------------|------|
| Transport block + CRC | Embedding + Masking | hagi_v4.py |
| LDPC encode | (implicit, weights = code) | — |
| Rate matching (puncturing) | Linear bottleneck (H -> C) | hagi_v4.py |
| Scrambling | mask_embed | hagi_v4.py |
| Modulation mapping (QAM) | FreqCoding2D phase modulation | freq_layer.py |
| Layer mapping + precoding | Multi-head + complex weight | freq_layer.py |
| OFDM modulation (IFFT) | torch.fft.ifft2 | freq_layer.py |
| Channel (AWGN, fading) | (implicit, training noise) | — |
| OFDM demodulation (FFT) | torch.fft.fft2 | freq_layer.py |
| Channel estimate | Kalman filter predict | kalman.py |
| Equalization | Complex weight @ freq modes | freq_layer.py |
| LDPC iterative decode | TurboLoop | hagi_v4.py |
| Parity check | GP2D (geometric product) | gp2d.py |
| HARQ buffer (Chase combining) | MSA.write | msa.py |
| Decision Feedback Equalizer | MSA.read | msa.py |
| EXIT chart stopping | Convergence halt | hagi_v4.py |
| Adaptive modulation (AMC) | Soft frequency gating | freq_layer.py |
| Rate dematching | Linear (C -> H) | hagi_v4.py |
| Demodulation -> bits | LM head | hagi_v4.py |
