# HAGI V4 — Анализ через призму Rate-Distortion теории

> Адаптация теоремы Шеннона о rate-distortion, Information Bottleneck (Тишби)
> и теоремы Guo & Li (2026) к архитектуре HAGI V4. Каждый компонент
> рассматривается как примитив сжатия, с формулами, аналогиями и конкретными
> рекомендациями по оптимизации.

---

## 0. Теоретический фундамент

### 0.1. Тождество «предсказание = сжатие» (Shannon, 1948)

Фундаментальная связь между предсказанием и сжатием установлена Шенноном
в его теореме о source coding. Если модель может предсказывать следующее
событие из распределения `P(X)` с_cross-entropy `H(P) = -Σ P(x) log P(x)`,
то она может сжимать данные из этого распределения до `H(P)` бит на символ.

Для языковой модели, обученной на next-token prediction:

```
Cross-entropy loss: L = -Σ_t log P(x_t | x_{<t})
Perplexity:         PP = exp(L)
Compression ratio:  C = L / log2(V)  (бит на токен / бит без сжатия)
```

Модель с perplexity 10 на словаре 50000 токенов сжимает текст до
`log2(10) ≈ 3.32` бит на токен (vs `log2(50000) ≈ 15.6` без сжатия).
**Веса модели = сжатое представление тренировочного датасета.**

### 0.2. Галлюцинации = артефакты сжатия с потерями (Guo & Li, 2026)

Guo & Li (arXiv:2602.00906) формализуют меморизацию фактов LLM как
**membership testing problem** — обобщение Bloom filter с непрерывными
confidence scores.

**Постановка задачи:**
- `U` — универсум всех возможных утверждений
- `K ⊆ U` — множество известных фактов (true)
- Запрос `i ∈ U` → модель выдаёт confidence score `x̂_i ∈ [0,1]`
- Галлюцинация = non-key `i ∉ K` получает высокую confidence

**Rate-Distortion теорема (sparse limit, |K|/|U| → 0):**

```
R_min = n · KL(μ_K ‖ μ_N) + o(n)    [бит на факт]

где:
  n       — число фактов для запоминания
  μ_K     — распределение confidence scores для фактов (keys)
  μ_N     — распределение confidence scores для не-фактов (non-keys)
  KL(·‖·) — Kullback-Leibler divergence, base-2
```

**Ключевой результат:** при ограниченной ёмкости модели (capacity < R_min)
и log-loss целевой функции, information-theoretically оптимальная стратегия —
**не abstain (отказаться) и не forget (забыть), а назначать высокую
confidence части не-фактов**. Галлюцинация — оптимальный режим ошибки.

**Аналогия с JPEG:** при сильном сжатии JPEG не «отказывается»
отображать блок — он заполняет его правдоподобными, но неточными
артефактами (ringing, blockiness). Модель делает то же самое с фактами:
вместо «не знаю» она генерирует правдоподобное, но неверное утверждение.

### 0.3. Information Bottleneck (Тишби, 2015)

Тишби (arXiv:1503.02406) формулирует deep learning как
information-theoretic tradeoff между сжатием и предсказанием:

```
Information Bottleneck:
  minimize:  F_β[p(Z|X)] = I(X;Z) − β · I(Y;Z)

где:
  I(X;Z) — mutual information между входом и представлением (complexity)
  I(Y;Z) — mutual information между представлением и таргетом (expressivity)
  β      — trade-off параметр (Lagrange multiplier)
```

**Две фазы тренировки (подтверждено Conklin et al. 2026, arXiv:2604.07569):**

```
Фаза 1: Fitting   — I(Y;Z) растёт (модель учится предсказывать)
                    представления расширяются, растёт complexity

Фаза 2: Compression — I(X;Z) падает (модель сжимает, отбрасывает шум)
                    representations сжимаются, generalization улучшается
```

**Information plane:** 2D пространство `(I(X;Z), I(Y;Z))`. Оптимальное
сжатие — на IB bound, где каждый дополнительный бит complexity даёт
бит expressivity, пока не достигнуто `I(Y;Z) = I(Y;X)`.

```
IB bound:  I(X;Z) = I(Z;Y)   (до насыщения I(Y;Z) = I(Y;X))

Выше bound — недостижимо (data processing inequality)
Ниже bound — субоптимально (избыточное хранение информации)
На bound  — оптимальное сжатие
```

### 0.4. HAGI V4 vs стандартный LLM — структурное сравнение

| Свойство | Стандартный LLM (GPT-style) | HAGI V4 |
|----------|---------------------------|---------|
| Режим предсказания | Авторегрессивный (последовательный) | Plane prediction (параллельный) |
| Аналог сжатия | Arithmetic coding (последовательное) | Transform coding (JPEG DCT) |
| Attention | Causal mask (видит только прошлое) | Bidirectional (видит всё) |
| Training objective | Next-token CE | Masked CE (предсказание замаскированных позиций) |
| Итеративная структура | Один forward pass | 4 итерации refinement |
| Структура hidden state | Плоский вектор | Cl(3,0,0) multivector (8 blades, 4 grades) |
| Память | KV cache (внутри последовательности) | MSA ring buffer (между итерациями) |
| Генерация | Left-to-right развёртывание | Progressive unmasking (по confidence) |

**Ключевое отличие:** стандартный LLM = последовательный кодек
(arithmetic coding) — декодирует по одному токену, каждый зависит
от предыдущего. HAGI V4 = параллельный кодек (transform coding) —
декодирует все позиции одновременно из «частотного» представления,
затем итеративно уточняет.

---

## 1. Аудит архитектуры — что HAGI V4 уже делает правильно

### 1.1. Cl(3,0,0) Grade Structure = Unequal Error Protection (UEP)

**Текущая реализация** (`config.py:18-23`, `clifford.py:29-31`):

```
Hidden state [576] разбит на 5 сегментов:
  Scalar    (64)  — grade 0: confidence/resolution — momentum 0.8
  Vector    (96)  — grade 1: entities/concepts     — momentum 0.5
  Bivector  (96)  — grade 2: relations             — momentum 0.0
  Trivector (64)  — grade 3: higher-order          — momentum 0.0
  Residual (256)  — unconstrained pass-through
```

Cl(3,0,0) имеет 8 basis blades:
```
  0b000 = 1        (grade 0, scalar)
  0b001 = e1       (grade 1, vector)
  0b010 = e2       (grade 1, vector)
  0b100 = e3       (grade 1, vector)
  0b011 = e1∧e2    (grade 2, bivector)
  0b101 = e1∧e3    (grade 2, bivector)
  0b110 = e2∧e3    (grade 2, bivector)
  0b111 = e1∧e2∧e3 (grade 3, trivector/pseudoscalar)
```

**Интерпретация через сжатие:**

Unequal Error Protection (UEP) — техника из теории кодирования, где
разные части данных защищаются с разной степенью избыточности.
В LTE/5G: signaling биты защищены сильнее, чем voice data.
В JPEG: DC коэффициенты (низкие частоты) квантуются меньше, чем AC.

В HAGI V4:
- **Scalar (grade 0)** = «DC компонента» — глобальная уверенность.
  Медленный momentum (0.8) = высокая защита, много бит эффективно.
  Аналог: DC коэффициент в JPEG, критический для всего блока.

- **Vector (grade 1)** = «низкочастотные AC» — сущности, концепты.
  Средний momentum (0.5) = средняя защита.
  Аналог: низкочастотные AC в JPEG, важны для общего вида.

- **Bivector (grade 2)** = «высокочастотные AC» — отношения, связи.
  Полный update (momentum 0.0) = низкая защита, быстро меняется.
  Аналог: высокочастотные AC в JPEG, детали, можно потерять.

- **Trivector (grade 3)** = «самые высокие частоты» — higher-order structure.
  Полный update (momentum 0.0) = минимальная защита.
  Аналог: крайние high-frequency коэффициенты, отбрасываются первыми.

- **Residual (256)** = «uncoded channel» — без структуры, pass-through.
  Аналог: raw bypass channel, не участвует в transform coding.

**GDR momentum как effective bit allocation:**

```
Effective bits per grade ∝ -log(1 - momentum)

  scalar:    -log(1 - 0.8) = -log(0.2)  = 2.32  →  максимальная защита
  vector:    -log(1 - 0.5) = -log(0.5)  = 1.00  →  средняя защита
  bivector:  -log(1 - 0.0) = -log(1.0)  = 0.00  →  нет защиты (full update)
  trivector: -log(1 - 0.0) = -log(1.0)  = 0.00  →  нет защиты
```

**Вердикт:** Близко к оптимальному. Grade structure реализует
semantic-aware rate allocation, которую плоские Transformers
не могут выразить. Momentum values создают иерархию важности.

---

### 1.2. Iterative Refinement = Successive Refinement Coding

**Текущая реализация** (`hrm.py:90-255`, `config.py:49-58`):

```
4 итерации refinement:
  - Deep supervision: CE loss на каждой итерации
    weight = deep_supervision_decay^iteration = 0.1^iter
    → [1.0, 0.1, 0.01, 0.001]
  - Adaptive halting: relative_delta < 0.01 → stop
  - Gradient checkpointing (no h.detach — полный gradient flow)
  - Deep supervision weight: 0.1 (10% от main CE)
```

Pipeline:
```
h → [perception blocks] → [GP2D] → iter 0: [reasoning + GDR + GP2D + MSA]
                                        → iter 1: [reasoning + GDR + GP2D + MSA]
                                        → iter 2: [reasoning + GDR + GP2D + MSA]
                                        → iter 3: [reasoning + GDR + GP2D + MSA]
                              → [expression blocks] → output
```

**Интерпретация через сжатие:**

Successive Refinement — фундаментальная концепция rate-distortion:
кодировать источник с прогрессивно возрастающей точностью, где каждый
слой refinement уменьшает distortion на определённую величину.
Декодер может остановиться на любом слое и получить качество,
пропорциональное потраченным битам.

```
R(D) curve для successive refinement:

  D
  ↑
  D_0 ─●
       │
  D_1 ─●
       │
  D_2 ─●
       │
  D_3 ─●
       │
       └────────→ R (bits)
       R_0 R_1 R_2 R_3

Каждая итерация = шаг вправо по R-оси (больше бит) и вниз по D (меньше ошибок)
```

**Аналогия с Progressive JPEG:**
- Progressive JPEG кодирует изображение в несколько сканов
- Scan 1: грубый контур (низкое качество, мало бит)
- Scan 2: детали (лучше качество, больше бит)
- Scan 3: fine details (ещё лучше, ещё больше бит)
- Scan 4: финальное качество (максимум бит)
- Декодер может остановиться после любого скана

HAGI V4 делает то же самое с текстом:
- Iteration 0: грубая структура текста (низкое качество предсказания)
- Iteration 1: уточнение сущностей и отношений
- Iteration 2: тонкая настройка
- Iteration 3: финальное качество

**Deep supervision decay = оптимальное bit allocation:**

```
w_i = decay^i = 0.1^i

  i=0: w = 1.0    (сильный gradient — большое уменьшение distortion)
  i=1: w = 0.1    (умеренный gradient)
  i=2: w = 0.01   (слабый gradient)
  i=3: w = 0.001  (минимальный gradient — маленькое уменьшение)
```

Это соответствует теоретическому предсказанию: marginal distortion
reduction убывает с каждой итерацией (diminishing returns).
Экспоненциальный decay близок к оптимальному для log-loss.

**Adaptive halting = early termination:**

```
halt condition: ||h_new - h_prev|| / ||h_prev|| < 0.01

Интерпретация: если representation не изменился (delta < 1%),
дальнейшие итерации не дадут значимого уменьшения distortion → stop.
```

**Вердикт:** Близко к оптимальному. Экспоненциальный decay весов
соответствует теоретическим предсказаниям для successive refinement
под log-loss. Adaptive halting корректно реализует early termination.

---

### 1.3. GP2D = Predictive Coding (DPCM)

**Текущая реализация** (`gp2d.py:24-54`):

```
Geometric Product 2D — temporal Clifford convolution

  Окно w=1: позиции t-1, t, t+1 (3 позиции)
  mv = h.reshape(B, T, n_heads, 8)  — multivector per head
  accumulated = Σ_i w_i · geometric_product(mv, shift(mv, delta_i))
  out = norm(proj(accumulated))
  h = h + sigmoid(gate) · out       — residual blend

  gate_init = -2.0 → sigmoid(-2) ≈ 0.12 (низкий начальный вклад)
```

Геометрическое произведение Cl(3,0,0):
```
  a · b = Σ_{i,j} a_i · b_j · (e_i · e_j)

  где e_i · e_j = e_i∧e_j  если i ≠ j  (внешнее произведение)
              = ±1       если i = j  (внутреннее, знак от metric)

  Результат: multivector, содержащий все grade components
  scalar часть:    внутреннее произведение (косинус-подобие)
  bivector часть:  внешнее произведение (ориентированная площадь)
```

**Интерпретация через сжатие:**

Predictive coding — кодировать не сам сигнал, а остаток (residual)
от предсказания. Это основа DPCM (Differential Pulse Code Modulation)
и всех современных видео кодеков (P-frames, B-frames в H.264/H.265).

```
DPCM:
  prediction:  x̂_t = f(x_{t-1}, x_{t-2}, ...)
  residual:    r_t = x_t - x̂_t
  encode:      r_t (обычно меньше по амплитуде → меньше бит)

HAGI V4 GP2D:
  prediction:  gp_out = geometric_product(h[t], h[t-1], h[t+1])
  residual:    h = h + sigmoid(gate) · proj(gp_out)
  → модель учится хранить только "невязку" между токеном и его контекстом
```

**Аналогия с видео кодеками:**

| Видео кодек | HAGI V4 GP2D |
|-------------|--------------|
| I-frame (intra) | Токен без GP2D (только self-attention) |
| P-frame (predict from previous) | GP2D с t-1 (predict from left context) |
| B-frame (bidirectional predict) | GP2D с t-1, t, t+1 (bidirectional) |
| Motion vectors | Geometric product (реляционная геометрия) |
| Residual coding | `h + gate · proj(gp_out)` (residual blend) |

**Геометрическое произведение vs dot-product:**

Dot-product (стандартный attention): `a · b = Σ a_i b_i` — скаляр,
мера сходства. Не различает «параллельные» и «перпендикулярные» отношения.

Geometric product: `a · b = a·b + a∧b` — multivector:
- `a·b` (scalar) — косинус-подобие (сходство)
- `a∧b` (bivector) — ориентированная площадь (отношение, rotation)

```
Пример:
  a = e1, b = e1   → a·b = 1 (сходство), a∧b = 0 (нет отношения)
  a = e1, b = e2   → a·b = 0 (ортогональны), a∧b = e1∧e2 (отношение)
  a = e1, b = e1+e2 → a·b = 1, a∧b = e1∧e2 (сходство + отношение)
```

GP2D захватывает **и сходство, и отношение** между токенами,
тогда как dot-product — только сходство.

**Низкий gate init (0.12):**

sigmoid(-2) ≈ 0.12 означает, что на старте тренировки GP2D вносит
~12% в hidden state. Это правильно:
- Рано: модель должна сначала научиться self-attention (I-frames)
- Потом: постепенно включает cross-token prediction (P-frames)
- Gate learnable → модель сама решает, когда доверять GP2D

**Вердикт:** Звуковая реализация. Геометрическое произведение —
более богатый предсказатель, чем dot-product. Низкий gate init
соответствует стратегии «сначала I-frames, потом P-frames».

---

### 1.4. GDR Cross-Grade Mixing = Nonlinear Transform Coding

**Текущая реализация** (`gdr.py:70-180`):

```
Grade-Decomposed Recurrence:

  1. Split hidden [576] на grades:
     [scalar(64), vector(96), bivector(96), trivector(64), residual(256)]

  2. Per-grade update с momentum:
     scalar_new    = σ(m_s) · scalar    + (1-σ(m_s)) · s_upd
     vector_new    = σ(m_v) · vector    + (1-σ(m_v)) · v_upd
     bivector_new  = b_upd                           (full update)
     trivector_new = t_upd                           (full update)

  3. Cross-grade mixing через geometric self-product:
     mv = vector_new.reshape(n_mv, 8)
     g0, g2 = geometric_product_self_g02(mv)
       — g0: grade-0 проекция (scalar) от v×v
       — g2: grade-2 проекция (bivector) от v×v

     scalar_new    += σ(gate_s) · linear_s(g0_flat)
     bivector_new  += σ(gate_b) · linear_b(g2_flat)

  4. Grade router (MoE-style):
     gate = softmax(linear(graded_ctx))  → [B, T, 4]
     s_upd *= gate[..., 0]
     v_upd *= gate[..., 1]
     b_upd *= gate[..., 2]
     t_upd *= gate[..., 3]
```

**Интерпретация через сжатие:**

Cross-grade mixing создаёт нелинейные зависимости между
информационными потоками. В source coding это аналог transform coding,
где коэффициенты взаимодействуют (DCT, wavelet).

```
Transform coding (DCT):
  X = [x1, x2, ..., xN]  — входные данные
  Y = DCT(X)              — transform coefficients
  Y[0] = DC component     — зависит от всех X
  Y[k] = AC component     — взаимодействие частот

HAGI V4 (geometric product):
  V = [v1, v2, ..., vN]  — vector grade (entities)
  S, B = gp_self(V)       — geometric transform
  S = scalar component    — confidence (зависит от всех entities)
  B = bivector component  — relations (зависит от пар entities)
```

**Геометрический смысл:**

```
vector × vector → scalar + bivector

  e1 × e1 = 1           (scalar — самосходство)
  e1 × e2 = e1∧e2       (bivector — отношение между e1 и e2)
  e1 × (e1+e2) = 1 + e1∧e2  (scalar + bivector)

Интерпретация:
  — scalar часть: «сколько entity похожа на себя» (confidence)
  — bivector часть: «как entity относится к другим» (relation)

Это структурная информация, которую плоский MLP не может извлечь.
```

**Аналогия с DCT:**

| DCT | HAGI V4 GDR |
|-----|-------------|
| DC coefficient (Y[0]) | Scalar output (g0) — global average |
| AC coefficients (Y[k>0]) | Bivector output (g2) — relational structure |
| Quantization matrix | Grade momentum (разная защита per grade) |
| Zigzag scan | Grade router (приоритизация grades) |
| Inverse DCT | Linear projections (geo_to_scalar, geo_to_bivector) |

**Gated contribution:**

```
scalar_new    += sigmoid(gate_s) · linear_s(g0)
bivector_new  += sigmoid(gate_b) · linear_b(g2)

gate init = 0.0 → sigmoid(0) = 0.5 (умеренный начальный вклад)
→ модель учится регулировать, сколько relational information
  вливать в confidence (scalar) и relations (bivector)
```

**Вердикт:** Новаторский и теоретически мотивированный.
Аналога в стандартных LLM нет. Геометрическое произведение
создаёт структурированный information transform, который
эксплуатирует алгебраическую структуру Cl(3,0,0).

---

### 1.5. MSA = Side Information / Slepian-Wolf Coding

**Текущая реализация** (`msa.py:1-117`):

```
Memory Sparse Attention:
  - Ring buffer: 4096 slots
  - slot_chunk_size: 4  (4 токена → 1 слот, 4:1 compression)
  - top_k: 6            (6 slots per query)
  - routing_key_dim: 64 (Clifford scalar routing key)
  - MLA compress_dim: 128 (latent compression of KV)
  - MLA up_dim: 288     (4 kv_heads × 72 head_dim)

Write:
  flat_h = h.reshape(B*T, H)
  chunked = flat_h.view(n_slots, 4, H).mean(dim=1)  — 4:1 compression
  keys = route_proj(chunked)           — 64d routing key
  kv = mla_compress(chunked)           — 128d compressed KV
  registry.write(keys, kv)             — ring buffer

Read:
  query = route_proj(flat_h)           — 64d query
  top_indices, top_scores = registry.read_topk(query, k=6)
  kv_compressed = registry.gather_kv(top_indices)
  k = mla_up_k(kv_compressed)          — decompress K
  v = mla_up_v(kv_compressed)          — decompress V
  attn_out = SDPA(q, k, v)             — attention
```

**Интерпретация через сжатие:**

Slepian-Wolf theorem (1973): если декодер имеет side information Y,
коррелированный с источником X, то требуемая rate для X:

```
R_X|Y < R_X = H(X)

  R_X   — rate без side information = H(X) (entropy)
  R_X|Y — rate с side information = H(X|Y) (conditional entropy)

  ΔR = H(X) - H(X|Y) = I(X;Y)  (экономия = mutual information)
```

В HAGI V4:
- X = текущий hidden state (нужно предсказать)
- Y = MSA slots (hidden states из предыдущих итераций)
- MSA предоставляет side information → снижает требуемую rate
  от скрытого состояния к предсказанию

**Двухстадийное сжатие:**

```
Стадия 1: Chunk compression (4:1)
  4 токена → average → 1 slot
  Lossy: теряется fine-grained token-level информация
  Аналог: downsampling изображения (4x4 → 1 pixel average)

Стадия 2: MLA compression (H → 128d)
  H=576 → compress_dim=128 (4.5:1 compression)
  Lossy: латентное сжатие KV через linear projection
  Аналог: JPEG DCT + quantization (пространство → частоты → отбрасывание)

Total compression: 4 × 4.5 = 18:1 (от raw hidden state до stored slot)
```

**Selective retrieval = conditional decoding:**

```
  top_k=6 из 4096 slots → только 0.15% памяти читается per query
  Аналог: random access в compressed stream — декодер читает только
  нужные блоки, не декодируя весь файл
```

**Аналогия с видео кодеками:**

| Видео кодек | HAGI V4 MSA |
|-------------|-------------|
| Reference frame buffer | Ring buffer (4096 slots) |
| Motion-compensated prediction | Top-k retrieval + attention |
| Frame compression (I/P frame) | Chunk + MLA compression |
| Reference frame selection | Top-k routing (cosine similarity) |
| Sliding window (DPB) | Ring buffer (FIFO overwrite) |

**Вердикт:** Реализует Slepian-Wolf-style exploitation side information.
Двухстадийное сжатие (chunk + MLA) агрессивно, но принципиально
обосновано. Ring buffer = sliding window в видео кодеках.

---

### 1.6. MoE = Conditional Computation = Variable-Rate Coding

**Текущая реализация** (`moe.py:41-131`):

```
Mixture of Experts:
  - 4 experts (SwiGLU)
  - top-1 routing (Switch Transformer style)
  - MoD skip slot для тривиальных токенов
  - Load-balance aux loss (Shazeer/Switch)

Forward:
  router_logits = router(x)              — [B*T, 5] (4 experts + skip)
  top_k_probs, top_k_indices = topk(...)  — top-1
  for each expert e:
    mask = (expert_idx == e)
    tokens = x[mask]
    output[mask] = expert_e(tokens) * prob[mask]
  if skip: output[skip_mask] = x * prob  — identity passthrough

Load balance:
  aux = α · N · Σ (fraction_e · mean_prob_e)
  → penalizes uneven expert utilization
```

**Интерпретация через сжатие:**

Variable-rate coding — разные части данных кодируются с разной rate
в зависимости от их сложности. В HAGI V4 каждый токен получает
ровно один эксперт = один «кодек».

```
Token complexity        Expert assigned       Effective rate
─────────────────────────────────────────────────────────────
Trivial (stopwords)  →  MoD skip (identity)  →  0 bits (passthrough)
Low complexity       →  Expert 0 (simple)    →  low rate
Medium complexity    →  Expert 1-2           →  medium rate
High complexity      →  Expert 3 (complex)   →  high rate
```

**Аналогия с адаптивным квантованием:**

| Адаптивное квантование | HAGI V4 MoE |
|------------------------|-------------|
| Flat regions → coarse quant | Trivial tokens → MoD skip |
| Detailed regions → fine quant | Complex tokens → full expert |
| Bit allocation per block | Expert routing per token |
| Rate-distortion optimization | Load-balance aux loss |

**MoD skip = zero-rate coding:**

```
  skip_out = x * prob  (identity, scaled by routing probability)

  Токен, которому назначен skip, не обрабатывается FFN вообще.
  Это экономит compute и capacity: тривиальные токены (the, a, is)
  не нуждаются в трансформации — они уже хорошо представлены.

  Аналог: в видео кодеках skip blocks — блоки, которые идентичны
  reference frame, не кодируются вообще (0 бит).
```

**Load-balance aux loss:**

```
  aux = α · N · Σ_e (f_e · P_e)

  где:
    f_e = fraction of tokens routed to expert e
    P_e = average routing probability for expert e
    N   = number of experts
    α   = load-balance weight (0.01)

  Цель: f_e ≈ 1/N для всех e (равномерная нагрузка)

  Без этого loss: router collapse — все токены идут к одному
  "лучшему" эксперту, остальные простаивают (wasted capacity).

  Аналог: в rate-distortion optimization, если все блоки используют
  один codec, остальные codec'и не используются → субоптимальное
  распределение бит.
```

**Вердикт:** Корректная реализация variable-rate conditional computation.
MoD skip = zero-rate для тривиальных токенов. Load-balance предотвращает
codec collapse.

---

### 1.7. Bidirectional Attention = Parallel Decompression

**Текущая реализация** (`attention.py:20-102`):

```
Grouped Query Attention:
  - 8 query heads, 4 KV heads (GQA 2:1)
  - head_dim = 72 (8 × 72 = 576)
  - RoPE positional encoding
  - bidirectional = True (no causal mask)
  - fp16 attention (bf16 → fp16 for SDPA softmax)

  is_causal = not bidirectional = False
  → F.scaled_dot_product_attention(q, k, v, is_causal=False)
  → все позиции видят все позиции
```

**Интерпретация через сжатие:**

Два фундаментальных подхода к декодированию:

```
1. Sequential decoding (autoregressive LLM):
   x_1 → predict x_2 → predict x_3 → ... → predict x_T
   
   Аналог: Arithmetic coding
   - Декодирует по одному символу за раз
   - Каждый символ зависит от всех предыдущих
   - O(T) шагов декодирования
   - Оптимально для sequential sources

2. Parallel decoding (HAGI V4 plane prediction):
   [x_1, x_2, ..., x_T] → model → [predictions for all positions]
   
   Аналог: Transform coding (JPEG DCT)
   - Декодирует все позиции одновременно
   - Все позиции независимы (после transform)
   - O(1) шагов декодирования (single forward pass)
   - Итеративное уточнение: O(log T) effective steps
```

**Сравнение эффективности:**

```
                    Sequential (AR)    Parallel (HAGI V4)
                    ───────────────    ───────────────────
Decode steps:       O(T)               O(1) + O(iter) ≈ O(log T)
Dependency:         x_t depends on     x_t depends on
                    x_{<t}             all positions (bidirectional)
Latency:            High (serial)      Low (parallel + 4 iterations)
Quality:            Token-by-token     Global optimization
Error propagation:  Left-to-right       Global (no cascade)
Training:           Teacher forcing     Masked CE (harder)
```

**Аналогия с JPEG vs arithmetic coding:**

| JPEG (transform coding) | HAGI V4 (plane prediction) |
|-------------------------|---------------------------|
| All pixels decoded at once | All positions predicted at once |
| DCT transform | Bidirectional attention |
| Quantization | Masked CE training |
| Progressive scan | Iterative refinement |
| No error cascade | No left-to-right error cascade |

**Вердикт:** Архитектурно превосходнее для compression analogy.
Trade-off: сложнее тренировки (нет teacher forcing от causal structure).
Bidirectional attention + iterative refinement = аналог progressive JPEG.

---

### 1.8. Distillation = Distributed Source Coding

**Текущая реализация** (`distillation.py:85-264`, `loop.py:114-145`):

```
Knowledge Distillation:
  Teacher: SmolLM2-360M (360M params, bf16 ~720MB)
  Student: HAGI V4 (~74M params)

Loss:
  L = α · CE_student + (1-α) · T² · KL(softmax(s/T) ‖ softmax(t/T))

  где:
    CE_student — cross-entropy student vs ground truth
    KL         — KL divergence between student and teacher logits
    T          — temperature (2.0)
    α          — alpha schedule (0.5 → 0.3)

Alpha schedule:
  α(step) = α_start + (α_end - α_start) · (step / distill_end_step)
  
  step 0:      α = 0.5  (50% CE + 50% KL)
  step 90k:    α = 0.3  (30% CE + 70% KL)
  step 90k+:   α = 1.0  (100% CE, distillation ended)

  distill_end_frac = 0.6 → ends at 60% of training
```

**Интерпретация через сжатие:**

Distributed source coding — кодирование источника с ограниченной rate
с использованием high-rate encoder как референса.

```
Teacher = high-rate encoder (360M params → много бит)
Student = low-rate encoder  (74M params → мало бит)

KL(student ‖ teacher) = distortion между high-rate и low-rate представлениями

Optimal distillation:
  minimize:  KL(p_student ‖ p_teacher)
  subject to: capacity(student) = 74M

  → это в точности rate-distortion problem для knowledge transfer
```

**Температура как rate controller:**

```
T → ∞: softmax становится равномерным → низкая rate (мало информации)
T = 1:  softmax оригинальный → стандартная rate
T → 0:  softmax становится argmax → максимальная rate (hard label)

При T = 2.0: softer distribution → student получает больше информации
о относительных вероятностях (dark knowledge), не только о top-1 prediction
```

**Alpha schedule = fitting → compression:**

```
Phase 1 (early training, α=0.5):
  50% CE + 50% KL
  → student учится базовой структуре (CE) + получает guidance от teacher (KL)
  → аналог: fitting phase в IB (рост I(Y;Z))

Phase 2 (late training, α=0.3):
  30% CE + 70% KL
  → больше веса на KL → student утончает compression до teacher
  → аналог: compression phase в IB (снижение I(X;Z))

Phase 3 (post-distillation, α=1.0):
  100% CE
  → student свободен от teacher, оптимизирует собственное сжатие
  → аналог: fine-tuning на собственной R(D) кривой
```

**Вердикт:** Хорошо спроектировано. Alpha schedule соответствует
compression theory (early = fitting phase, late = compression phase).
Завершение дистилляции на 60% тренировки — правильно: оставшиеся 40%
позволяют модели оптимизироваться самостоятельно.

---

### 1.9. Coherence Loss = Smoothness Prior = Low-Pass Filter

**Текущая реализация** (`cast.py:19-49`):

```
Coherence Head:
  coherence_loss = sigmoid(gate) · mean(||geometric_product(h[t], h[t+1])||²)

  gate_init = -5.0 → sigmoid(-5) ≈ 0.007 (почти выключен на старте)

Геометрическое произведение между соседними позициями:
  area = geometric_product(mv[:, :-1], mv[:, 1:])
  
  Bivector часть area = ориентированная площадь между h[t] и h[t+1]
  → мера "реляционной discontinuity" между соседними токенами
```

**Интерпретация через сжатие:**

Штраф за высокочастотную вариацию между соседними позициями.
Аналог: quantization matrix в JPEG, которая штрафует
высокочастотные DCT коэффициенты сильнее.

```
JPEG quantization matrix (example for luminance):
  [16 11 10 16 24  40  51  61]   ← низкие частоты: малый quant step
  [12 12 14 19 26  58  60  55]      (сохраняются)
  [14 13 16 24 40  57  69  56]
  [14 17 22 29 51  87  80  62]
  [18 22 37 56 68 109 103  77]
  [24 35 55 64 81 104 113  92]
  [49 64 78 87 103 121 120 101]
  [72 92 95 98 112 100 103  99]   ← высокие частоты: большой quant step
                                       (отбрасываются)

HAGI V4 coherence loss:
  penalizes ||gp(h[t], h[t+1])||² → штраф за high-frequency
  variation в geometric domain
  → гладкие сигналы имеют меньшую entropy → лучше сжимаются
```

**Математическая формулировка:**

```
  coherence_loss = σ(gate) · (1/(T-1)) · Σ_{t=0}^{T-2} ||h[t] × h[t+1]||²

  где × = geometric product, ||·||² = squared L2 norm

  Минимизация → h[t] и h[t+1] становятся «геометрически параллельными»
  → ||h[t] × h[t+1]||² → 0 когда h[t] ≈ h[t+1] (scalar part доминирует)
  → bivector part (discontinuity) → 0
```

**Near-off init (0.007):**

sigmoid(-5) ≈ 0.007 — loss почти выключен на старте.
Это правильно:
- Рано: не принуждать к smoothness, дать модели найти структуру
- Поздно: gate learnable, модель сама включает smoothness когда готова
- Аналог: в JPEG, quantization matrix применяется после DCT,
  не до — сначала transform, потом penalize high frequencies

**Вердикт:** Корректно. Действует как entropy prior — гладкие сигналы
имеют меньшую entropy, сжимаются лучше. Near-off init правильный.

---

## 2. Возможности оптимизации — по компонентам

### 2.1. Grade Dimensions = Capacity Allocation (HIGH IMPACT)

**Проблема:** `grade_dims = [64, 96, 96, 64, 256]` фиксированы.
Разные датасеты/задачи имеют разную entropy per grade.

**Rate-distortion принцип:**

Оптимальное распределение capacity выравнивает marginal distortion
по всем компонентам (equal slope condition):

```
  dD/dR_{grade0} = dD/dR_{grade1} = dD/dR_{grade2} = dD/dR_{grade3}

  где:
    D = total distortion (CE loss)
    R_{grade_k} = bits allocated to grade k = d_k × bits_per_dim

  Если dD/dR_{grade_k} > dD/dR_{grade_j}:
    → grade_k недофинансирован, нужно больше dims
    → grade_j перефинансирован, можно убрать dims
```

**Аналогия с bit allocation в JPEG:**

```
JPEG: для каждого 8×8 блока назначается QP (quantization parameter).
  Сложные блоки → низкий QP (больше бит)
  Простые блоки → высокий QP (меньше бит)
  → равномерное качество по всему изображению

HAGI V4: для каждого grade назначается dimension.
  High-entropy grade → больше dims (больше бит)
  Low-entropy grade → меньше dims (меньше бит)
  → равномерная marginal distortion
```

**Рекомендация:**

1. **Мониторинг** (quick win, low effort):
   Добавить логирование per-grade activation variance в `loop.py`:

```python
# В train_step, после model forward:
gdr = model.gdr
scalar_var = output.hidden[..., :gdr.d_scalar].var().item()
vector_var = output.hidden[..., gdr.d_scalar:gdr.d_scalar+gdr.d_vector].var().item()
bivector_var = output.hidden[..., ...].var().item()
# Log every 100 steps
```

   Если scalar variance низкая (confidence насыщена) → можно уменьшить
   d_scalar, увеличить d_bivector.

2. **Soft-learnable allocation** (high effort):
   Сделать grade dims мягко-обучаемыми через meta-router.
   Learnable temperature на grade projections → модель сама
   перераспределяет capacity.

---

### 2.2. Deep Supervision Decay = Optimal Bit Allocation (MEDIUM IMPACT)

**Проблема:** Фиксированный decay `0.1^iteration` = [1.0, 0.1, 0.01, 0.001].

**Rate-distortion принцип:**

Оптимальный вес для successive refinement пропорционален
ожидаемому уменьшению distortion на каждом слое:

```
  w_i = E[||h_i - h_{i-1}||²]  — actual delta, не фиксированный

  Если итерация i даёт большое изменение representation:
    → high information gain → большой вес (больше gradient)
  
  Если итерация i почти не меняет representation:
    → low information gain → маленький вес (меньше gradient)
```

**Текущий vs оптимальный:**

```
Текущий:    w = [1.0,    0.1,    0.01,   0.001]  (фиксированный)
Оптимальный: w = [Δ_0,   Δ_1,   Δ_2,   Δ_3]     (adaptive)

где Δ_i = EMA(||h_i - h_{i-1}||²) — exponential moving average
```

**Рекомендация:**

Заменить фиксированный decay на adaptive weighting в `hrm.py`:

```python
# В RefinementCore.forward, iteration loop:
# Track actual delta_norm between iterations
delta_norm = (h - h_prev).float().norm(dim=-1).mean()

# EMA of delta_norm per iteration
if not hasattr(self, 'delta_ema'):
    self.delta_ema = [0.0] * self.n_iterations
self.delta_ema[iteration] = 0.99 * self.delta_ema[iteration] + 0.01 * delta_norm.item()

# Use delta_ema as weight instead of fixed decay
weight = self.delta_ema[iteration] / max(sum(self.delta_ema), 1.0)
total_deep_supervision = total_deep_supervision + weight * ce
```

**Эффект:** Автоматически направляет больше gradient на итерации,
где модель реально меняет representation (high information gain),
и меньше — где representation сходится (low information gain).

---

### 2.3. MSA Chunk Size = Adaptive Compression Rate (HIGH IMPACT)

**Проблема:** `slot_chunk_size=4` фиксирован. Весь контент получает
4:1 сжатие независимо от entropy.

**Rate-distortion принцип:**

Оптимальная compression rate зависит от entropy источника.
High-entropy контент нуждается в меньших chunk'ах (меньше сжатия,
больше слотов). Low-entropy контент может использовать большие chunk'и.

```
  R_optimal(chunk_size) = H(source) / log2(chunk_size)

  High H → small chunk (больше бит per slot, меньше потеря)
  Low H  → large chunk  (меньше бит per slot, acceptable потеря)
```

**Аналогия с adaptive quantization:**

```
В H.264/H.265: Adaptive Quantization (AQ)
  - Complex scene → smaller QP (больше бит per macroblock)
  - Flat scene → larger QP (меньше бит per macroblock)
  → uniform perceptual quality

В HAGI V4 MSA:
  - High-entropy tokens (relations, reasoning) → chunk_size=2
  - Low-entropy tokens (facts, common patterns) → chunk_size=8
  → uniform distortion across content types
```

**Рекомендация:**

Сделать chunk size adaptive на основе grade router output:

```python
# В MSAModule.write:
# Use grade dominance to select chunk size
grade_gate = model.gdr.grade_router(graded_ctx)  # [B, T, 4]

# scalar-dominant (high confidence, low entropy) → chunk=8
# bivector-dominant (complex relations, high entropy) → chunk=2
if gate[..., 0].mean() > 0.5:    # scalar-dominant
    chunk_size = 8
elif gate[..., 2].mean() > 0.5:  # bivector-dominant
    chunk_size = 2
else:
    chunk_size = 4  # default
```

**Эффект:** High-entropy контент (relations, reasoning) получает
больше слотов (меньше потеря при сжатии). Low-entropy контент
(facts, patterns) сжимается сильнее (экономия памяти).

---

### 2.4. MoE Expert Specialization = Frequency Band Allocation (MEDIUM IMPACT)

**Проблема:** 4 generic эксперта без специализации, кроме load-balance.

**Rate-distortion принцип:**

В transform coding разные basis functions специализируются на разных
frequency bands. Эксперты должны специализироваться на разных
«information bands».

```
JPEG DCT basis functions:
  B[0] = DC (average)           → global brightness
  B[1-2] = low AC (vertical)    → coarse vertical structure
  B[3-4] = low AC (horizontal)  → coarse horizontal structure
  B[5-8] = high AC (diagonal)   → fine details

HAGI V4 MoE (proposed):
  Expert 0: scalar-dominant tokens (confidence resolution)
  Expert 1: vector-dominant tokens (entity processing)
  Expert 2: bivector-dominant tokens (relation processing)
  Expert 3: trivector-dominant tokens (higher-order structure)
  MoD skip: residual-dominant tokens (pass-through)
```

**Рекомендация:**

Добавить expert specialization signal. Во время тренировки
коррелировать expert routing с grade dominance:

```python
# В MoESwiGLU.forward, add aux loss:
# Correlate expert routing with grade dominance
grade_gate = model.gdr.grade_router(graded_ctx)  # [B, T, 4]
expert_probs = F.softmax(router_logits, dim=-1)   # [B*T, 5]

# Target: expert_e should correlate with grade_e
# (scalar→expert0, vector→expert1, bivector→expert2, trivector→expert3)
specialization_loss = -torch.log(
    (grade_gate * expert_probs[:4]).sum(dim=-1).mean()
)
# Add to total loss with small weight
```

**Эффект:** Структурированная специализация экспертов без принуждения.
Каждый эксперт становится specialist для своего grade — аналог
frequency band specialization в DCT.

---

### 2.5. GP2D Residual Whitening = Optimal Predictive Coding (MEDIUM IMPACT)

**Проблема:** GP2D output blending в residual stream без проверки,
является ли residual white (uncorrelated).

**Rate-distortion принцип:**

Оптимальный predictive coding производит white (uncorrelated) residuals.
Если residual имеет autocorrelation, модель не полностью эксплуатирует
предсказание — остаётся структура для сжатия.

```
  Optimal: E[r_t · r_{t+1}] = 0  (white residual, no remaining structure)
  
  If E[r_t · r_{t+1}] > 0: residual has autocorrelation
    → prediction is suboptimal
    → more structure could be extracted
    → more compression possible
```

**Аналогия:**

```
В видео кодеках: если residual (prediction error) имеет spatial
autocorrelation, значит motion compensation не оптимальный.
Решение: улучшить motion estimation или добавить deblocking filter.

В HAGI V4: если GP2D residual имеет temporal autocorrelation,
значит geometric product prediction не оптимальный.
Решение: добавить whiteness loss → модель учится лучше предсказывать.
```

**Рекомендация:**

Добавить residual whiteness loss в `gp2d.py` / `losses.py`:

```python
# После GP2D, compute autocorrelation of residual
residual = sigmoid(gate) * proj(gp_out)  # [B, T, H]
# Lag-1 autocorrelation along temporal dimension
r_t = residual[:, :-1].flatten(0, 1)   # [B*(T-1), H]
r_t1 = residual[:, 1:].flatten(0, 1)   # [B*(T-1), H]
# Normalized cross-correlation
whiteness_loss = (F.cosine_similarity(r_t, r_t1, dim=-1).abs().mean())
# Penalize nonzero lag-1 correlation
total_loss += w_whiteness * whiteness_loss
```

**Эффект:** Модель учится полностью эксплуатировать cross-token
prediction, максимизируя compression efficiency.

---

### 2.6. Mask Ratio = Rate-Distortion Operating Point (MEDIUM IMPACT)

**Проблема:** Progressive masking 15% → 30% эвристический.

**Rate-distortion принцип:**

Mask ratio = fraction of information removed = distortion.
Training при данном mask ratio помещает модель на конкретную точку
R(D) кривой. Оптимальная operating point зависит от желаемого
rate-distortion tradeoff на inference.

```
  D (distortion = mask ratio)
  ↑
  0.9 ┤                                    ●  (inference round 4: 90% masked)
  0.7 ┤                          ●  (inference round 3: 70% masked)
  0.5 ┤                ●  (inference round 2: 50% masked)
  0.3 ┤      ●  (inference round 1: 30% masked)
  0.1 ┤●  (training: 10% masked, low distortion)
      └──────────────────────────────────→ R (bits = model capacity)
```

**Текущая inference** (`generate.py:26`):

```python
confidence_schedule = (0.9, 0.7, 0.5, 0.1)
# Round 0: unmask tokens with confidence > 0.9 (~10% unmasked, 90% masked)
# Round 1: confidence > 0.7 (~30% unmasked, 70% masked)
# Round 2: confidence > 0.5 (~50% unmasked, 50% masked)
# Round 3: confidence > 0.1 (~90% unmasked, 10% masked)
```

**Рекомендация:**

Match training mask ratio к inference confidence schedule.
Вместо фиксированного progressive ramp, cycle через mask ratios
[0.1, 0.3, 0.5, 0.7]:

```python
# В loop.py, replace progressive_mask_ratio with:
def cyclic_mask_ratio(step):
    ratios = [0.1, 0.3, 0.5, 0.7]
    return ratios[step % len(ratios)]

# Или weighted sampling: больше шагов на 0.3 (основной operating point)
# но иногда 0.1, 0.5, 0.7 (cover inference distribution)
```

**Эффект:** Модель тренируется на нескольких operating points
R(D) кривой, соответствующих inference schedule. Лучше соответствует
инференс-распределению, чем фиксированный ramp.

---

### 2.7. Distillation Temperature = Rate Matching (LOW IMPACT)

**Проблема:** Фиксированная температура T = 2.0.

**Rate-distortion принцип:**

Температура контролирует «мягкость» target distribution.
High temperature = более равномерное = низкая rate (меньше информации).
Low temperature = более острое = высокая rate (больше информации).
Оптимальная температура соответствует rate student'а.

```
  T → ∞:  softmax(logits/T) → uniform  → H(soft_target) = log(V) → max rate
  T = 1:  softmax(logits)              → H(soft_target) = original
  T → 0:  softmax(logits/T) → one-hot  → H(soft_target) = 0 → min rate

  Optimal T: H(soft_target) ≈ H(student_representation)
  → rate matching: encoder rate = decoder rate
```

**Рекомендация:**

Anneal температуру от 4.0 (early) до 1.0 (late):

```python
# В distillation.py, alpha_at function:
def temperature_at(step, max_steps, T_start=4.0, T_end=1.0):
    """Anneal temperature from high (coarse) to low (fine)."""
    progress = min(1.0, step / max_steps)
    return T_start + (T_end - T_start) * progress

# Early training: T=4.0 → soft targets, student учится coarse structure
# Late training:  T=1.0 → sharp targets, student утончает fine details
```

**Эффект:** Rate matching — адаптация encoder rate к decoder capacity.
Ранний training: high T (low rate, coarse). Поздний: low T (high rate, fine).

---

### 2.8. Adaptive Halting Threshold = Rate-Distortion Stopping (MEDIUM IMPACT)

**Проблема:** Фиксированный threshold `relative_delta < 0.01`.

**Rate-distortion принцип:**

Optimal stopping в successive refinement: stop когда
marginal distortion reduction < marginal rate cost.

```
  Stop at iteration i when:
    ΔD_i = E[||h_i - h_{i-1}||²] < threshold(step)

  где threshold зависит от remaining capacity budget:
    Early training: threshold = 0.05  (coarse, stop early — save compute)
    Late training:  threshold = 0.001 (fine, iterate longer — maximize quality)
```

**Рекомендация:**

Make threshold training-aware (3 строки в `hrm.py`):

```python
# В AdaptiveHalting.forward:
# Ramp threshold from 0.05 (early) to 0.001 (late)
def adaptive_threshold(step, max_steps, start=0.05, end=0.001):
    progress = min(1.0, step / max_steps)
    return start + (end - start) * progress

# В RefinementCore.forward, pass step to adaptive_halt:
threshold = adaptive_threshold(step, cfg.train.max_steps)
new_halts = self.adaptive_halt(h, h_prev, iteration, halted, threshold=threshold)
```

**Эффект:** Больше compute (итераций) на поздних этапах тренировки,
когда модель утончает детали. Меньше compute на ранних —
когда грубая структура быстро сходится.

---

### 2.9. Per-Grade Coherence Weighting (LOW IMPACT)

**Проблема:** Coherence loss обрабатывает все grades одинаково.

**Rate-distortion принцип:**

Разные grades имеют разную ожидаемую smoothness.

```
  Scalar (confidence): должен быть гладким → high coherence expected
    → аналог: DC в JPEG, медленно меняется
  
  Vector (entities): умеренно гладкий → medium coherence
    → аналог: low AC, меняется плавно
  
  Bivector (relations): может резко меняться → low coherence expected
    → аналог: high AC, детали могут резко различаться
  
  Trivector (higher-order): может резко меняться → low coherence
    → аналог: highest AC, fine details
```

**Рекомендация:**

Apply coherence loss только к scalar и vector grades:

```python
# В cast.py, CoherenceHead.coherence_loss:
def coherence_loss(self, h):
    B, T, H = h.shape
    # Only apply to scalar + vector grades
    gdr = model.gdr
    scalar = h[..., :gdr.d_scalar]
    vector = h[..., gdr.d_scalar:gdr.d_scalar+gdr.d_vector]
    sv = torch.cat([scalar, vector], dim=-1)
    
    n_heads = sv.size(-1) // BLADE_COUNT
    mv = sv.reshape(B, T, n_heads, BLADE_COUNT)
    area = geometric_product(mv[:, :-1], mv[:, 1:])
    gate = torch.sigmoid(self.gate_logit)
    return gate * (area.float() ** 2).mean()
```

**Эффект:** Coherence применяется только к grades, где smoothness
ожидается. Bivector/trivector свободны для резких изменений.

---

### 2.10. Curriculum = Entropy-Ordered Source Coding (ALREADY PRESENT)

**Текущая реализация** (`config.py:180-182`):

```yaml
curriculum_enabled: true
curriculum_stage2_start: 100000  # step to switch to hard-reasoning subset
```

**Интерпретация через сжатие:**

Тренировка на low-entropy данных сначала (easy patterns), потом
high-entropy (hard reasoning) = entropy-ordered coding.

```
  Source entropy H(X):
    Stage 1 (easy):    low H → model converges fast → coarse structure
    Stage 2 (hard):    high H → model refines → fine details

  Это оптимально для successive refinement:
    1. Build coarse structure (low entropy, high compression ratio)
    2. Refine with high-entropy details (low compression ratio, but
       model already has good base)
```

**Вердикт:** Уже реализовано. Изменений не требуется.

---

## 3. Новые компоненты, вдохновлённые rate-distortion теорией

### 3.1. Information Bottleneck Regularizer (HIGH IMPACT)

**Идея:** Добавить явный IB-регуляризатор в training loss.

**Теория:**

```
  IB objective:
    minimize:  F_β = I(X;Z) − β · I(Y;Z)

  I(X;Z) — mutual information между входом и hidden state (complexity)
  I(Y;Z) — mutual information между hidden state и таргетом (expressivity)
```

Оценка mutual information через variational bounds:

```
  I(X;Z) ≈ E[log q(z|x)] - E[log p(z)]
    — MINE (Mutual Information Neural Estimator)
    — или проще: hidden state variance (proxy for complexity)

  I(Y;Z) ≈ H(Y) - H(Y|Z)
    — H(Y|Z) ≈ CrossEntropy(logits, targets) (negative predictive info)
    — I(Y;Z) = H(Y) - CE_loss
```

**Реализация:**

```python
# В losses.py, добавить:
def information_bottleneck_loss(h, targets, lm_head_weight, beta=1.0):
    """
    IB regularizer: I(X;Z) - beta * I(Y;Z)
    
    h: [B, T, H] — hidden state
    targets: [B, T] — target token IDs
    lm_head_weight: [V, H] — for computing logits
    beta: trade-off parameter
    """
    # I(X;Z) proxy: hidden state variance (complexity)
    # High variance = more information stored = higher rate
    complexity = h.float().var(dim=(0, 1)).sum()
    
    # I(Y;Z) proxy: negative cross-entropy (predictive information)
    # Low CE = high predictive info = high expressivity
    logits = F.linear(h, lm_head_weight)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    predictive_info = -ce  # negative CE = I(Y;Z) proxy
    
    # IB loss: minimize complexity, maximize predictive info
    return complexity - beta * predictive_info

# В loop.py, добавить к total loss:
ib_loss = information_bottleneck_loss(output.hidden, targets, model.lm_head.weight, beta=0.01)
loss = loss + cfg.train.w_ib * ib_loss  # w_ib = 0.01
```

**Эффект:** Явно направляет модель к Information Bottleneck bound
(теоретически оптимальное сжатие). Сейчас модель приближается к этому
bound неявно через gradient descent — регуляризатор делает это явно
и быстрее.

**Аналогия:** В JPEG, quantization matrix явно штрафует high frequencies.
Без него, DCT коэффициенты могли бы быть любыми. С ним — явно
направляется к оптимальному сжатию. IB regularizer делает то же
для hidden representations.

---

### 3.2. Rate-Distortion Aware Quantization (MEDIUM IMPACT)

**Идея:** Применить Radio-style rate-distortion optimization к bf16
квантованию HAGI V4.

**Текущая реализация:** `precision: "bf16"` — uniform 16-bit для всех весов.

**Rate-distortion принцип (equal slope condition):**

```
  Optimal bit allocation: dD/dR_i = dD/dR_j for all parameters i, j

  где:
    D = output distortion (perplexity)
    R_i = bits allocated to parameter i

  Параметры с высоким dD/dR (чувствительные) → больше бит
  Параметры с низким dD/dR (нечувствительные) → меньше бит
```

**Рекомендация:**

Mixed-precision на основе weight importance:

```
  Component                          Precision    Rationale
  ─────────────────────────────────────────────────────────────────
  GDR momentum (scalar_mom_logit)    fp32         critical for grade
    vector_mom_logit                              dynamics, tiny param count
  Grade trunk/head                   fp32         core recurrent computation
  Attention QKV/O proj               bf16         standard, high param count
  MoE expert weights                 int8+calib   large param count,
                                                  less sensitive to quantization
  Embedding                          bf16         already transferred from teacher
  Norms                              fp32         numerical stability
  GP2D temporal_weights              fp32         small, critical for prediction
  Coherence gate                     fp32         small, controls regularization
```

**Эффект:** Снижает эффективный размер модели на ~30% с минимальной
потерей качества, следуя Radio framework's equal-slope optimization.

---

### 3.3. Hallucination Floor Estimation (DIAGNOSTIC)

**Идея:** Оценить теоретический hallucination floor для HAGI V4
используя теорему Guo & Li.

**Формула:**

```
  R_min = n · KL(μ_K ‖ μ_N)    [бит, минимальная memory для n фактов]

  Model capacity:
    C = N_params × bits_per_param = 74M × 16 = 1.184M бит (bf16)

  If n · KL(μ_K ‖ μ_N) > C:
    → hallucination is information-theoretically inevitable
    → minimum hallucination rate = 1 - C / (n · KL(μ_K ‖ μ_N))

  If n · KL(μ_K ‖ μ_N) ≤ C:
    → hallucination could be eliminated (in principle)
    → but may persist due to optimization imperfections
```

**Реализация:**

```python
# Новый скрипт: scripts/estimate_hallucination_floor.py
def estimate_hallucination_floor(model, dataset):
    """
    Estimate minimum achievable hallucination rate.
    
    Based on Guo & Li rate-distortion theorem:
    R_min = n * KL(mu_K || mu_N)
    """
    # 1. Collect confidence scores on known vs unknown facts
    fact_confidences = []
    nonfact_confidences = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataset.known_facts:
            output = model(batch.input_ids)
            probs = F.softmax(output.logits, dim=-1)
            confidence = probs.max(dim=-1).values
            fact_confidences.append(confidence)
        
        for batch in dataset.unknown_claims:
            output = model(batch.input_ids)
            probs = F.softmax(output.logits, dim=-1)
            confidence = probs.max(dim=-1).values
            nonfact_confidences.append(confidence)
    
    # 2. Estimate distributions (histogram or KDE)
    fact_conf = torch.cat(fact_confidences).numpy()
    nonfact_conf = torch.cat(nonfact_confidences).numpy()
    
    # 3. Compute KL divergence
    mu_K = histogram_to_distribution(fact_conf, bins=50)
    mu_N = histogram_to_distribution(nonfact_conf, bins=50)
    kl_div = kl_divergence(mu_K, mu_N)
    
    # 4. Model capacity
    n_params = sum(p.numel() for p in model.parameters())
    bits_per_param = 16  # bf16
    capacity = n_params * bits_per_param
    
    # 5. Hallucination floor
    n_facts = len(dataset.known_facts)
    required_memory = n_facts * kl_div
    
    if required_memory > capacity:
        floor = 1.0 - capacity / required_memory
    else:
        floor = 0.0  # theoretically eliminable
    
    return {
        'kl_divergence': kl_div,
        'required_memory_bits': required_memory,
        'model_capacity_bits': capacity,
        'hallucination_floor': floor,
        'capacity_ratio': capacity / required_memory,
    }
```

**Эффект:** Теоретическая нижняя граница hallucination rate для текущей
ёмкости модели. Если фактический hallucination rate близок к floor —
дальнейшие улучшения тренировки имеют убывающую отдачу, нужен больше
capacity (большая модель) или external memory (RAG/MSA).

---

### 3.4. Entropy-Adaptive Refinement (HIGH IMPACT)

**Идея:** Количество refinement итераций должно зависеть от
entropy входа.

**Rate-distortion принцип:**

High-entropy входы нуждаются в большем refinement (больше бит
для распределения). Low-entropy входы сходятся быстро (мало бит).

```
  Input entropy proxy: H_proxy = Var(h, dim=1).mean()
    — high variance = high entropy = complex input
    — low variance = low entropy = simple input

  Iterations:
    High H_proxy → 6 iterations (больше compute, больше бит)
    Low H_proxy  → 2 iterations (меньше compute, экономия)
```

**Аналогия:**

```
В видео кодеках: сложные сцены (high entropy) получают больше бит,
простые сцены (low entropy) — меньше. Rate control адаптирует
bit allocation per frame.

В HAGI V4: сложные входы (reasoning, relations) получают больше
итераций refinement, простые (common patterns) — меньше.
```

**Реализация:**

```python
# В hagi_v4.py, перед refinement:
# Compute input entropy proxy after perception blocks
h_after_perception = ...  # after perception blocks
entropy_proxy = h_after_perception.var(dim=1).mean().item()

# Adaptive iteration count
if entropy_proxy > threshold_high:
    n_iterations = 6  # complex input
elif entropy_proxy < threshold_low:
    n_iterations = 2  # simple input
else:
    n_iterations = 4  # default

# Pass to RefinementCore
self.hrm.n_iterations = n_iterations
```

**Эффект:** Rate-adaptive coding — больше бит (итераций) на сложные
входы, меньше на простые. Снижает average inference latency без
потери качества на сложных входах.

---

## 4. Оптимизация training pipeline

### 4.1. Two-Phase Training Schedule (IB-Aligned)

**Идея:** Явно разделить тренировку на fitting и compression phases,
следуя Information Bottleneck траектории.

**Phase 1: Fitting (0-50% тренировки) — рост I(Y;Z)**

```yaml
# Phase 1 config
train:
  mask_ratio_start: 0.15         # low mask = high signal density
  distill_alpha: 0.5             # 50% CE + 50% KL (rely on teacher)
  learning_rate: 3.0e-4          # high LR, fast convergence
  gp2d_gate_init: -1.0           # sigmoid(-1) ≈ 0.27 (moderate prediction)
  w_coherence: 0.0001            # minimal smoothness pressure
  w_ib: 0.0                      # no IB regularizer yet
  # Objective: maximize I(Y;Z) — fit the target
```

**Phase 2: Compression (50-100% тренировки) — снижение I(X;Z)**

```yaml
# Phase 2 config
train:
  mask_ratio_end: 0.35           # high mask = force compression
  distill_alpha: 0.3             # 30% CE + 70% KL (own signal)
  learning_rate: 3.0e-4          # cosine decay to 0
  gp2d_gate_init: -2.0           # sigmoid(-2) ≈ 0.12 (minimal prediction)
  w_coherence: 0.001             # enforce smoothness
  w_ib: 0.01                     # explicit IB pressure
  # Objective: minimize I(X;Z) while maintaining I(Y;Z) — compress
```

**Аналогия с JPEG:**

```
Phase 1 (Fitting):
  Аналог: DCT transform — найти частотное представление
  → модель учится structure (what frequencies matter)

Phase 2 (Compression):
  Аналог: Quantization — отбросить неважные частоты
  → модель сжимает, отбрасывает noise, сохраняет signal
```

### 4.2. Data Curation = Source Entropy Reduction

**Rate-distortion принцип:**

```
  Cleaner data → lower source entropy H(X)
  For fixed model capacity R: lower H(X) → lower distortion D
  
  D(R, H) = R(D) curve depends on H(X)
  Lower H(X) → entire R(D) curve shifts down → less D for same R
```

**Для HAGI V4:**

1. **Deduplication:** удалить duplicate/near-duplicate sequences
   → снижает H(X), модель не тратит capacity на запоминание копий

2. **Quality filter:** удалить noisy, contradictory, low-quality text
   → снижает H(X), устраняет источник confusion

3. **Grade-aware data selection:** баланс scalar-heavy (factual) и
   bivector-heavy (relational) контента
   → обеспечивает равномерную тренировку всех grades

4. **Curriculum** (уже реализовано): stage1 (easy) → stage2 (hard)
   → entropy-ordered source coding

### 4.3. Progressive Capacity Growth (Optional)

**Идея:** Начать тренировку с меньшим числом активных grades,
прогрессивно включать больше.

**Rate-distortion принцип:**

Successive refinement на архитектурном уровне — начать с low-rate
encoder (мало grades), затем grow до full capacity.

```
  Steps 0-30k:    scalar + vector active (160 dims)   → low rate
  Steps 30k-60k:  + bivector (256 dims)               → medium rate
  Steps 60k-90k:  + trivector (320 dims)              → high rate
  Steps 90k+:     full model (576 dims)               → max rate
```

**Аналогия:** Scalable Video Coding (SVC) — base layer (low quality)
+ enhancement layers (progressively better).

**Реализация:**

```python
# В gdr.py, GradeDecomposedRecurrence.forward:
def forward(self, h, step=None):
    # Progressive capacity growth
    if step is not None:
        if step < 30000:
            # Only scalar + vector
            h[..., self.d_scalar+self.d_vector:] = 0  # zero out higher grades
        elif step < 60000:
            # + bivector
            h[..., self.d_scalar+self.d_vector+self.d_bivector:] = 0
        # else: full model
    
    # ... rest of forward
```

**Эффект:** Модель сначала учится low-grade structure (confidence,
entities), потом добавляет higher grades (relations, higher-order).
Аналог progressive JPEG: сначала DC, потом AC.

---

## 5. Сводная матрица приоритетов

| Оптимизация | Impact | Effort | Файл | Компонент |
|-------------|--------|--------|------|-----------|
| IB regularizer в loss | HIGH | MEDIUM | losses.py, loop.py | Training |
| Entropy-adaptive refinement | HIGH | MEDIUM | hagi_v4.py, hrm.py | Refinement |
| Adaptive MSA chunk size | HIGH | MEDIUM | msa.py | Memory |
| Per-grade capacity monitoring | HIGH | LOW | loop.py | Diagnostics |
| Two-phase training schedule | MEDIUM | LOW | config yaml | Training |
| Adaptive deep supervision weight | MEDIUM | LOW | hrm.py | Refinement |
| Expert specialization signal | MEDIUM | MEDIUM | moe.py | MoE |
| GP2D residual whitening | MEDIUM | LOW | gp2d.py, losses.py | GP2D |
| Adaptive halting threshold ramp | MEDIUM | LOW | hrm.py | Refinement |
| Mixed-precision (Radio-style) | MEDIUM | HIGH | optim.py | Quantization |
| Temperature annealing | LOW | LOW | distillation.py | Distillation |
| Per-grade coherence | LOW | LOW | cast.py | Coherence |
| Hallucination floor estimation | DIAGNOSTIC | MEDIUM | new script | Analysis |
| Progressive capacity growth | LOW | MEDIUM | gdr.py, loop.py | Training |

### Quick wins (low effort, high impact):

1. **Per-grade activation variance logging** — диагностика, без изменений
   кода. Добавить логирование в `loop.py`.

2. **Two-phase training schedule** — только config YAML. Разделить
   тренировку на fitting и compression phases с разными гиперпараметрами.

3. **Adaptive halting threshold ramp** — 3 строки в `hrm.py`.
   Ramp threshold 0.05 → 0.001 по шагам.

4. **Temperature annealing** — 3 строки в `distillation.py`.
   Anneal T от 4.0 до 1.0.

### Strategic investments (high impact, medium effort):

1. **Information Bottleneck regularizer** — новый loss term.
   Добавить `I(X;Z) − β·I(Y;Z)` через variational bounds.

2. **Entropy-adaptive refinement** — модификация `RefinementCore`.
   Количество итераций зависит от entropy входа.

3. **Adaptive MSA chunk size** — модификация `MSAModule`.
   Chunk size adaptive на основе grade router output.

---

## 6. Фундаментальные пределы

### 6.1. Невозможно полностью устранить галлюцинации

При finite capacity и high-entropy данных — теорема Guo & Li.
Минимальный hallucination rate:

```
  H_floor = max(0, 1 - C / (n · KL(μ_K ‖ μ_N)))

  где:
    C = model capacity (params × bits/param)
    n = number of facts to memorize
    KL = KL divergence between fact/non-fact confidence distributions
```

Для HAGI V4 (~74M params, bf16):
```
  C = 74M × 16 = 1.184M бит
  Если n=1M facts, KL=2 бит: required = 2M бит > C → floor ≈ 41%
  Если n=100K facts, KL=2 бит: required = 200K бит < C → floor = 0%
```

### 6.2. MSA как путь снижения hallucination floor

MSA (external memory) = side information → Slepian-Wolf:

```
  С MSA:     R_required = H(X|Y) = H(X) - I(X;Y)
  Без MSA:   R_required = H(X)

  ΔR = I(X;Y) — экономия за счёт side information

  Если MSA хранит факты → model capacity освобождается для patterns
  → effective hallucination floor снижается
```

### 6.3. Нельзя превзойти IB bound

IB bound — information-theoretic предел, как Shannon limit для сжатия.
Любая архитектура подчиняется этому пределу. HAGI V4 уже близка к
bound благодаря grade structure + iterative refinement.

### 6.4. Tradeoff между hallucination и forgetting

Guo & Li: устранение hallucination (false positives) на random facts
очень дорого без одновременного увеличения forgetting (false negatives)
или over-refusal. Post-processing для factual accuracy только двигает
вдоль memory-error frontier, не выходит за него.

```
  Memory-error frontier:
  
  Hallucination rate
  ↑
  │\
  │ \  ← frontier (rate-distortion curve)
  │  \
  │   \
  │    \_______
  │             \
  └──────────────→ Forgetting rate
```

Увеличение model capacity сдвигает frontier вниз-вправо (лучше оба).
RAG/MSA (external memory) выводит за пределы frontier для фактов
(не занимает model capacity).

---

## 7. Ключевые работы для углубления

| Работа | Год | Вклад |
|--------|-----|-------|
| Shannon, "A Mathematical Theory of Communication" | 1948 | Rate-distortion theory, source coding |
| Slepian & Wolf, "Noiseless Coding of Correlated Information Sources" | 1973 | Side information coding |
| Tishby, Pereira, Bialek, "Information Bottleneck Method" | 1999 | IB framework |
| Tishby & Zaslavsky, "Deep Learning and the Information Bottleneck" | 2015 | IB for deep learning |
| Conklin et al., "Learning is Forgetting: LLM Training As Lossy Compression" | 2026 | Empirical: LLMs approach IB bound |
| Guo & Li, "Hallucination is a Consequence of Space-Optimality" | 2026 | Rate-distortion theorem for hallucination |
| Young, "Radio: Rate-Distortion Optimization for LLM Compression" | 2025 | Rate-distortion for quantization |
| OpenAI, "Why Language Models Hallucinate" | 2025 | Hallucination from training/evaluation mismatch |
