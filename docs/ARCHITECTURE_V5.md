# HAGI V5 — Codec Language Model

## Архитектура на основе теории связи (Shannon)

### Концепция

Языковая модель — это кодек в смысле Шеннона:
- **Source** (текст) → **Encoder** (компрессия) → **Channel** (masключение = erasure) → **Decoder** (восстановление) → **Destination** (генерация)

Теорема разделения (Source-Channel Separation Theorem): оптимальная
коммуникация достигается раздельной оптимизацией source coding
(компрессия) и channel coding (помехоустойчивость).

---

## Pipeline

```
Token IDs
    │
    ▼
┌──────────────────────────────────────────────┐
│           SOURCE ENCODER (компрессия)         │
│                                                │
│  Embedding → Bidirectional Attention           │
│  → Information Bottleneck (H → H/r)           │
│  → Compressed Latent Z [B, T, H/r]            │
│                                                │
│  Цель: max I(Z;Y) - β·I(X;Z)                  │
│  Z = sufficient statistic для предсказания     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│        CHANNEL ENCODER (избыточность)          │
│                                                │
│  Z → Systematic Code: Z + Parity(Z)           │
│  → GP2D: geometric product соседних позиций    │
│  → Expand H/r → H (parity bits заполняют)     │
│  → Coded Latent Z' [B, T, H]                  │
│                                                │
│  Цель: ||Z' - Z||² = redundancy strength       │
│  Geometric product = parity check              │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│          CHANNEL (erasure noise)               │
│                                                │
│  Mask(p): случайно стереть p позиций           │
│  p = adaptive mask ratio                       │
│  Capacity C = 1 - p                            │
│                                                │
│  mask_embed = max-entropy vector               │
│  Сигнал модели: "здесь erasure, восстанови"    │
│                                                │
│  p adapts: p = 1 - avg_confidence              │
│  (capacity matching — как adaptive modulation) │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│      CHANNEL DECODER (iterative BP)            │
│                                                │
│  Iteration k:                                  │
│    prior_k = posterior_{k-1}                   │
│    → Attention (message passing)               │
│    → MoE (capacity allocation by entropy)      │
│    → MSA (memory = long-range parity checks)   │
│    → h_out                                     │
│    extrinsic_k = h_out - prior_k  ← КЛЮЧ       │
│    posterior_k = prior_k + extrinsic_k         │
│                                                │
│  Convergence: ||extrinsic_k|| < ε → stop       │
│                                                │
│  КЛЮЧЕВОЕ: передаётся только extrinsic,        │
│  не полный hidden state. Это предотвращает     │
│  information recycling (как в LDPC/Turbo).     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│           SOURCE DECODER (генерация)           │
│                                                │
│  Recovered Latent → Expression Layers          │
│  → Coherence Regularization (smooth output)    │
│  → Soft Token Distribution P(token | context)  │
│                                                │
│  Soft output: полные распределения,            │
│  discretize только в финальном output          │
└──────────────────────────────────────────────┘
```

---

## Ключевые инновации vs V4

### 1. Extrinsic Information Separation (главное)

**V4**: HRM feedback = full hidden state (intrinsic + extrinsic смешаны).
Информация рециклируется — та же информация усиливается каждую итерацию
вместо добавления новой.

**V5**: Каждая итерация вычисляет extrinsic = h_out - h_in.
Только extrinsic передаётся дальше. Prior обновляется отдельно:
posterior = prior + extrinsic. Это как LDPC decoder: передаётся
extrinsic LLR, не full posterior.

```python
# V4 (текущий)
for iteration in range(n_iters):
    h = reasoning_blocks(h)  # полный hidden state
    h = gdr(h)
    h = gp2d(h)

# V5 (новый)
for iteration in range(n_iters):
    h_prior = h
    h_new = reasoning_blocks(h)  # full computation
    h_new = gdr(h_new)
    h_new = gp2d(h_new)
    extrinsic = h_new - h_prior       # только НОВАЯ информация
    h = h_prior + extrinsic * alpha   # belief update
    if extrinsic.norm() < epsilon:
        break  # convergence — beliefs стабилизировались
```

### 2. Mask Embedding = Uncertainty Indicator

**V4**: `mask_embed` инициализируется нулём → обучается → остаётся ~0
(norm=0.07). Модель "не знает, что она не знает".

**V5**: `mask_embed` = max-entropy вектор. Инициализация: равномерное
распределение по всем grade dimensions. Сигнал: "здесь erasure,
восстанови из контекста".

```python
# V5 init
self.mask_embed = nn.Parameter(torch.ones(hidden_size) / (hidden_size ** 0.5))
# Не ноль — модель сразу видит "я не знаю эту позицию"
```

### 3. Adaptive Mask Ratio = Capacity Matching

**V4**: Fixed schedule 15% → 30% (progressive).

**V5**: Измерять average prediction confidence → p = 1 - confidence.
Если модель уверена → больше mask (harder). Если не уверена → меньше
mask (easier). Это как adaptive modulation в 5G.

```python
# V5
with torch.no_grad():
    avg_conf = probs.max(dim=-1).values.mean().item()
target_mask_ratio = 1.0 - avg_conf  # capacity matching
mask_ratio = 0.9 * mask_ratio + 0.1 * target_mask_ratio  # EMA smoothing
```

### 4. GP2D as Channel Code (Systematic Parity)

**V4**: GP2D = residual prediction + whiteness loss. Просто фильтр.

**V5**: GP2D = systematic parity. h = data + GP(h) = systematic code.
Geometric product между соседями = parity bits. Decoder может проверить
consistency через обратный GP.

Структура: z (compressed, H/r) → expand → h (H) = [z | parity(z)]
где parity(z) = GP(z[t-1], z[t]) — избыточная информация о связи.

### 5. Convergence-Based Halting (теоретически обоснованный)

**V4**: Adaptive halt based on relative delta norm (эвристика).

**V5**: Halt based on extrinsic information rate (EXIT chart analysis).
||extrinsic_k|| < ε → beliefs converged → stop. Это теоретически
обоснованный критерий из iterative decoding theory.

### 6. MoE as Capacity Allocation

**V4**: Top-1 routing with load balancing (content-based).

**V5**: Expert routing based on per-position uncertainty (entropy).
Low uncertainty → simple expert (few bits needed).
High uncertainty → complex expert (more bits needed).
Это как variable-rate coding в современных системах связи.

```python
# V5 routing
entropy = -probs * probs.log().sum(dim=-1)  # per-position entropy
router_input = torch.cat([h, entropy.unsqueeze(-1)], dim=-1)
gate = router(router_input)  # route based on uncertainty + content
```

### 7. Soft Generation (Belief State Decoding)

**V4**: Hard token predictions (argmax/multinomial) during generation.

**V5**: Maintain soft distributions (belief states) throughout decoding.
Только в финальном output: sample/argmax from final distribution.

```python
# V5 generation
beliefs = torch.zeros(B, T, V)  # soft belief state
for step in range(max_new_tokens):
    # Run decoder → update beliefs
    new_beliefs = decode(model, full_ids, beliefs)
    beliefs = beliefs * 0.7 + new_beliefs * 0.3  # belief update
    # Commit only when confidence is high enough
    if beliefs[step].max() > 0.8:
        token = beliefs[step].argmax()
        full_ids = cat(full_ids, token)
```

---

## Training Objective

Вместо одного masked CE, multi-objective aligned с communication theory:

```
Loss = CE                              # Fidelity: I(Z;Y)
      + λ₁ · IB_loss                   # Compression: -I(X;Z)
      + λ₂ · Parity_strength           # Redundancy: ||GP(h)||
      + λ₃ · Extrinsic_info_rate       # Decoding: I(extrinsic;Y)
      + λ₄ · Efficiency                # Convergence: iterations
```

Где:
- **CE** = masked cross-entropy (fidelity — правильно предсказывать)
- **IB_loss** = information bottleneck (compression — ограничить rate)
- **Parity_strength** = ||GP(h)||² (redundancy — enough for error correction)
- **Extrinsic_info_rate** = I(extrinsic_k; Y) per iteration (decoding quality)
- **Efficiency** = iterations_used (minimize computation)

---

## Component Mapping (V4 → V5)

| V4 компонент | V5 роль | Изменения |
|---|---|---|
| Perception layers | Source Encoder | + IB bottleneck после |
| GP2D | Channel Encoder | Systematic parity, не просто фильтр |
| HRM refinement | Channel Decoder | Extrinsic separation, convergence halt |
| MSA | Channel Decoder | Long-range parity checks (memory) |
| MoE | Capacity Allocation | Route by entropy, не только content |
| Expression layers | Source Decoder | Без изменений |
| Coherence head | Source Decoder | Soft output, не hard logits |
| mask_embed | Erasure Indicator | Max-entropy init, не zero |
| Mask schedule | Capacity Matching | Adaptive, confidence-based |

---

## Configuration (V5)

```yaml
model:
  hidden_size: 576
  bottleneck_ratio: 0.5        # compress to H/2, then expand
  perception_layers: 2
  reasoning_layers: 7
  expression_layers: 2

  channel_encoder:
    type: "gp2d_systematic"    # systematic code, не просто фильтр
    parity_weight: 0.1         # λ₂

  channel_decoder:
    type: "extrinsic_bp"       # belief propagation с extrinsic
    extrinsic_alpha: 1.0       # belief update weight
    convergence_threshold: 0.01 # ||extrinsic|| < ε → stop
    min_iterations: 1
    max_iterations: 6

  masking:
    type: "adaptive_erasure"   # capacity matching
    initial_ratio: 0.15
    max_ratio: 0.50
    adaptation_rate: 0.01      # EMA smoothing
    mask_embed_init: "max_entropy"  # не zero

  moe:
    routing: "entropy_aware"   # route by uncertainty + content
    num_experts: 4

train:
  loss_weights:
    ce: 1.0
    ib: 0.01                   # λ₁
    parity: 0.1                # λ₂
    extrinsic_info: 0.01       # λ₃
    efficiency: 0.001          # λ₄

  generation:
    type: "soft_belief"        # maintain distributions
    belief_momentum: 0.7       # prior weight in belief update
    commit_threshold: 0.8      # confidence to commit token
```

---

## Инференс (V5)

### Masked Autoregressive с Soft Beliefs

```
1. Init: prompt + mask_token (generation position)
2. For each new token:
   a. Run channel decoder (iterative BP с extrinsic)
   b. Get soft belief: P(token | context) [B, V]
   c. Update belief state: belief = momentum * prior + (1-momentum) * new
   d. If max(belief) > commit_threshold:
      - Commit: token = belief.argmax()
      - Append mask_token for next position
   e. Else: run another refinement iteration
3. Stop at EOS or max_tokens
```

Это sequential decoding с soft decisions — как stack decoder в
convolutional codes, но с continuous beliefs вместо hard paths.

---

## Почему это лучше V4

1. **Extrinsic separation** предотвращает information recycling —
   каждая итерация добавляет НОВУЮ информацию, а не повторяет старую.
   Это = быстрее convergence + лучше quality при том же числе итераций.

2. **Capacity matching** держит mask ratio около channel capacity —
   модель всегда работает near-optimal rate, не слишком easy/hard.

3. **Max-entropy mask embed** даёт модели ясный сигнал "здесь erasure" —
   модель может различить "я знаю этот токен" vs "это erasure".

4. **Soft generation** сохраняет uncertainty до конца —
   модель не делает преждевременных hard decisions.

5. **Convergence-based halting** останавливается когда beliefs стабилизировались —
   теоретически обоснованный критерий, не эвристика.

6. **Parity structure** в GP2D даёт явную избыточность для error correction —
   модель может проверять consistency своих предсказаний.
