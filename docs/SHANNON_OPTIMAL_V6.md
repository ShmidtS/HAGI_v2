# HAGI V6 — Shannon-Optimal Codec Language Model

## Полный mapping теории связи → архитектура + дизайн идеальной модели

---

## 1. Постулат: ИИ = система связи

Языковая модель — это кодек в смысле Шеннона. Каждый элемент архитектуры
имеет точный аналог в теории связи. Цель — приблизиться к пределу Шеннона
для пропускной способности канала при минимальных затратах ресурсов.

### Pipeline Shannon

```
Source (текст/токены)
    │
    ▼
Source Encoder (компрессия — IB bottleneck)     ── max I(Z;Y) - β·I(X;Z)
    │
    ▼
Channel Encoder (избыточность — parity)          ── systematic code: Z + Parity(Z)
    │
    ▼
Channel (erasure + noise)                        ── C = 1 - p (capacity)
    │
    ▼
Channel Decoder (iterative BP — extrinsic)       ── extrinsic_k = h_out - h_prior
    │                                              ||extrinsic_k|| < ε → stop
    ▼
Source Decoder (генерация — soft beliefs)        ── P(token | context)
    │
    ▼
Destination (выходные токены)
```

---

## 2. Полный mapping: каждый элемент и формула

### 2.1. Теорема Шеннона-Hartley: C = B·log₂(1 + SNR)

**Формула:** C = B·log₂(1 + S/N)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| C (capacity) | Пропускная способность модели | params × bits/param = 74M × 16 = 1.184M бит |
| B (bandwidth) | Число параметров (ширина канала) | 74M параметров |
| SNR (signal/noise) | Качество данных / шум градиентов | data quality / gradient noise |
| log₂(1+SNR) | Bits per parameter (эффективность) | ~16 бит (bf16), но эффективных ~4-8 |

**Проблема V5:** Фиксированная bf16 точность = фиксированная rate per parameter.
Эффективные биты меньше 16 из-за noise в training.

**Улучшение V6:** Mixed-precision с rate-distortion optimization.
Параметры с высоким dD/dR → больше бит, с низким → меньше.

```
Эффективная capacity V5:  74M × 4 ≈ 296M бит (эффективных)
Эффективная capacity V6:  74M × 6 ≈ 444M бит (mixed-precision + IB)
Прирост:                  1.5x capacity при том же числе параметров
```

### 2.2. Source Coding Theorem: H(X) ≤ R

**Формула:** H(X) = -Σ P(x) log P(x) — энтропия источника

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| H(X) | Энтропия языка | ~3.3 бит/токен (perplexity 10) |
| R (rate) | Бит на токен = log₂(V) × compression_ratio | log₂(49154) ≈ 15.6 → 3.3 (compression 4.7:1) |
| R(D) | Rate-distortion curve | IB objective: min I(X;Z) - β·I(Y;Z) |

**Реализация V5:** Embedding (H=576) → perception (2 layers) → bottleneck_down (H→H/2=288).
Bottleneck = linear projection, простая компрессия.

**Улучшение V6:** Variational Information Bottleneck с learned variance.
Вместо детерминированного linear projection — стохастическое сжатие:

```python
# V5: детерминированный bottleneck
z = bottleneck_down(h)  # linear, no variance

# V6: variational IB bottleneck
mu = bottleneck_mu(h)       # [B, T, C]
logvar = bottleneck_logvar(h)  # [B, T, C]
z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterization
kl_loss = -0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL(q(z|x) || p(z))
```

**Аналогия:** VAE-based compression = optimal source coding с known distortion.
Variational bottleneck явно моделирует I(X;Z) через KL-дивергенцию.

### 2.3. Channel Coding Theorem: R ≤ C

**Формула:** Для надёжной передачи: rate R ≤ channel capacity C

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| R (code rate) | Информационная нагрузка = 1 - mask_ratio | 0.7 (mask 30%) |
| C (capacity) | 1 - p (erasure channel capacity) | C = 1 - 0.3 = 0.7 |
| Parity bits | GP2D geometric product | systematic parity |

**Реализация V5:** GP2D как systematic parity. h = data + gate·GP(data).
Window=1 (3 позиции: t-1, t, t+1). Geometric product = parity check.

**Улучшение V6:** Multi-scale parity с interleaving (LDPC-like):

```python
# V5: single-scale parity (window=1)
accumulated = sum(w[i] * geometric_product(mv, shift(mv, delta_i)))

# V6: multi-scale parity (LDPC-like)
# Scale 1: window=1 (adjacent, t-1/t/t+1) — local parity
# Scale 2: window=3 (±3 positions) — mid-range parity
# Scale 3: window=8 (±8 positions) — long-range parity
# Interleaving: permute positions before each scale для burst error protection
for scale, window in enumerate([1, 3, 8]):
    shifted = roll_with_interleave(mv, window, scale)
    parity = geometric_product(mv, shifted)
    accumulated += w[scale] * parity
```

**Аналогия:** LDPC codes используют sparse parity-check matrix с variable
check node degrees. Multi-scale GP2D = multi-degree parity checks.

### 2.4. Rate-Distortion Theory: R(D) = min I(X;X̂) s.t. E[d(X,X̂)] ≤ D

**Формула:** R(D) = min I(X;X̂) при условии E[d(X,X̂)] ≤ D

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| D (distortion) | Prediction error = CE loss | cross-entropy |
| R (rate) | Model capacity used | params × bits |
| R(D) curve | IB bound | min I(X;Z) - β·I(Y;Z) |
| d(X,X̂) | Per-token loss | F.cross_entropy |

**Реализация V5:** IB loss через proxy (hidden state variance).
`complexity = h.var(dim=(0,1)).sum()`, `predictive_info = -ce`.

**Улучшение V6:** Явный variational IB с MINE estimator:

```python
def ib_loss_variational(z, x, y, beta=1.0):
    """Variational IB: I(X;Z) - beta * I(Y;Z).
    
    I(X;Z) ≈ KL(q(z|x) || p(z)) — analytical for Gaussian
    I(Y;Z) ≈ H(Y) - H(Y|Z) = H(Y) - CE(logits, y)
    """
    # I(X;Z) via KL divergence (analytical for Gaussian posterior)
    kl_xz = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    
    # I(Y;Z) via CE
    logits = lm_head(z)
    ce = F.cross_entropy(logits, y)
    i_yz = -ce  # proxy: -CE = I(Y;Z) up to constant
    
    return kl_xz - beta * i_yz
```

### 2.5. Slepian-Wolf Coding: R(X|Y) < R(X) = H(X)

**Формула:** R(X|Y) = H(X|Y) = H(X) - I(X;Y) — conditional entropy

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| X | Текущий hidden state | h [B, T, C] |
| Y | MSA slots (side information) | ring buffer 4096 slots |
| I(X;Y) | Mutual info = экономия rate | top_k=6 slots per query |
| R(X|Y) | Required rate with side info | reduced prediction uncertainty |

**Реализация V5:** MSA с 2-stage compression (chunk 4:1 + MLA 4.5:1 = 18:1 total).
Adaptive chunk size based on grade variance.

**Улучшение V6:** Asymmetric Slepian-Wolf — декодер использует side info
из PREVIOUS блоков при generation:

```python
# V6: cross-block side information
# При generation block N, MSA хранит representations из block N-1, N-2...
# Это как reference frame buffer в видео кодеках
for block_idx in range(n_blocks):
    # Read side info from previous blocks (frozen)
    side_info = msa.read(h_current, top_k=6)
    h_current = h_current + side_info  # Slepian-Wolf: use side info
    # Generate block
    tokens = generate_block(h_current)
    # Write current block to MSA for future blocks
    msa.write(h_current)
```

### 2.6. LDPC/Turbo Decoding: Extrinsic LLR

**Формула:** extrinsic_k = posterior_k - prior_k (только НОВАЯ информация)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| Extrinsic LLR | extrinsic = h_out - h_prior | `extrinsic = h - h_prior` |
| Prior | h_prior (предыдущая итерация) | `h_prior = h` (сохраняется) |
| Posterior | h (после reasoning blocks) | `h = h_prior + extrinsic * alpha` |
| Alpha (damping) | extrinsic_alpha = 1.0 | фиксированный 1.0 |

**Реализация V5:** Extrinsic separation реализован, но alpha=1.0 (без damping).
Convergence halt: `||extrinsic|| < ε`.

**Улучшение V6:** Turbo decoder — DUAL component decoders с extrinsic exchange:

```python
# V5: single decoder, 4 iterations
for iteration in range(n_iters):
    h_prior = h
    h = reasoning_blocks(h)
    extrinsic = h - h_prior
    h = h_prior + extrinsic * alpha

# V6: Turbo decoder — two component decoders
# Decoder A: attention-based (local parity checks)
# Decoder B: MSA-based (long-range parity checks)
# Exchange extrinsic between A and B
for iteration in range(n_iters // 2):
    # Component decoder A
    h_prior = h
    h_a = reasoning_blocks(h)  # attention-based
    ext_a = h_a - h_prior
    h = h_prior + ext_a * alpha_a
    
    # Component decoder B (uses A's extrinsic as prior)
    h_prior = h
    h_b = msa_refinement(h)  # memory-based
    ext_b = h_b - h_prior
    h = h_prior + ext_b * alpha_b
    
    # Convergence check on combined extrinsic
    if (ext_a.norm() + ext_b.norm()) < epsilon:
        break
```

**Аналогия:** Turbo codes = два parallel concatenated convolutional codes.
Декодер итеративно обменивается extrinsic information между компонентами.
V6: attention (local) + MSA (long-range) = два компонента.

### 2.7. EXIT Chart: Convergence Analysis

**Формула:** I_extrinsic(k) → 0 при k → ∞ (convergence)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| EXIT chart | extrinsic_norms per iteration | `extrinsic_norms` list |
| Convergence threshold | ε = 0.01 | convergence_threshold = 0.01 |
| Iteration halt | `||extrinsic_k|| < ε` | use_convergence_halt = True |

**Реализация V5:** Convergence halt основан на norm extrinsic.
`if ext_norm < convergence_threshold: break`.

**Улучшение V6:** Adaptive convergence threshold + EXIT chart tracking:

```python
# V6: adaptive threshold based on EXIT chart trajectory
# Track extrinsic norms to predict convergence
extrinsic_history = []
for iteration in range(n_iters):
    ...
    extrinsic_history.append(ext_norm)
    
    # Predict convergence: if rate of decrease is slowing
    if len(extrinsic_history) >= 3:
        rates = [extrinsic_history[i] - extrinsic_history[i+1] 
                 for i in range(len(extrinsic_history)-1)]
        avg_rate = sum(rates) / len(rates)
        # If remaining iterations won't reduce below threshold, halt early
        predicted_remaining = ext_norm / max(avg_rate, 1e-8)
        if predicted_remaining < 0.5:  # less than half iteration needed
            break
```

### 2.8. Adaptive Modulation (5G LTE)

**Формула:** Модуляция адаптируется к SNR: QPSK (low SNR) → 16QAM → 64QAM (high SNR)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| SNR | Model confidence | avg_confidence = max(probs).mean() |
| Modulation order | Mask ratio (bits per position) | mask_ratio = 1 - confidence |
| Adaptive modulation | Capacity matching | adaptive_mask_ratio() |

**Реализация V5:** `target_mask_ratio = 1.0 - avg_confidence` с EMA smoothing.
High confidence → more mask (harder), low confidence → less mask (easier).

**Улучшение V6:** Per-position adaptive modulation (вместо global):

```python
# V5: global mask ratio
avg_conf = probs.max(dim=-1).values.mean()
mask_ratio = 1.0 - avg_conf  # one value for all positions

# V6: per-position adaptive masking
# High-confidence positions: mask more (they can recover)
# Low-confidence positions: mask less (they need signal)
per_pos_conf = probs.max(dim=-1).values  # [B, T]
per_pos_mask_prob = 1.0 - per_pos_conf  # [B, T]
# Sample mask per position based on confidence
mask = torch.bernoulli(per_pos_mask_prob * mask_ratio_scale).bool()
```

### 2.9. Unequal Error Protection (UEP)

**Формула:** Разные части данных защищаются с разной избыточностью

| Grade | Размер | Momentum | Effective bits | Аналог | Защита |
|-------|--------|----------|---------------|--------|--------|
| Scalar (0) | 64 | 0.8 | 2.32 | DC в JPEG | Максимальная |
| Vector (1) | 96 | 0.5 | 1.00 | Low AC | Средняя |
| Bivector (2) | 96 | 0.0 | 0.00 | High AC | Минимальная |
| Trivector (3) | 64 | 0.0 | 0.00 | Highest AC | Минимальная |
| Residual | 256 | — | — | Bypass | Нет |

**Формула effective bits:** bits ∝ -log(1 - momentum)

```
scalar:    -log(1 - 0.8) = -log(0.2)  = 2.32  →  максимальная защита
vector:    -log(1 - 0.5) = -log(0.5)  = 1.00  →  средняя защита
bivector:  -log(1 - 0.0) = -log(1.0)  = 0.00  →  нет защиты
trivector: -log(1 - 0.0) = -log(1.0)  = 0.00  →  нет защиты
```

**Реализация V5:** GDR с per-grade momentum. Scalar/vecter learnable momentum.
Bivector/trivector = full update (momentum=0).

**Улучшение V6:** Water-filling capacity allocation:

```python
# V6: water-filling — оптимальное распределение capacity по grades
# Принцип: dD/dR_i = dD/dR_j для всех i,j (equal slope condition)
# Измеряем per-grade distortion (variance) и перераспределяем dims

def water_filling_allocation(grade_vars, total_dims):
    """Оптимальное распределение dimensions по grades.
    
    Water-filling: больше dims = больше capacity = меньше distortion.
    Равновесие: marginal distortion reduction одинаков для всех grades.
    """
    # Inverse variance = 1/D_i → allocate dims proportional to sqrt(variance)
    # High variance grade → more dims (more capacity needed)
    sqrt_vars = [v.sqrt() for v in grade_vars]
    total_sv = sum(sqrt_vars)
    allocation = [int(total_dims * sv / total_sv) for sv in sqrt_vars]
    # Ensure minimum 8 dims per grade, adjust residual
    ...
    return allocation
```

**Аналогия:** Water-filling в information theory — оптимальное распределение
power по частотам: больше power на частоты с высоким SNR.
В V6: больше dims на grades с высокой variance (entropy).

### 2.10. Successive Refinement / Progressive JPEG

**Формула:** R(D) = R(D₁) + R(D₂-D₁) + ... (progressive coding)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| Scan 1 (coarse) | Iteration 0 | deep_supervision weight = 1.0 |
| Scan 2 | Iteration 1 | weight = 0.1 |
| Scan 3 | Iteration 2 | weight = 0.01 |
| Scan 4 (fine) | Iteration 3 | weight = 0.001 |
| Weight decay | Exponential 0.1^i | adaptive EMA-based |

**Реализация V5:** 4 iterations с exponential decay или adaptive EMA weights.
Deep supervision CE at each iteration.

**Улучшение V6:** Progressive capacity growth (SVC analog):

```python
# V6: progressive capacity — активировать grades постепенно
# Step 0-30k:   scalar + vector only (160 dims)  → base layer
# Step 30k-60k: + bivector (256 dims)            → enhancement 1
# Step 60k-90k: + trivector (320 dims)           → enhancement 2
# Step 90k+:    full model (288 dims compressed) → full quality

def progressive_capacity(h, step, gdr):
    if step < 30000:
        # Zero out bivector + trivector + residual
        h[..., gdr.d_scalar+gdr.d_vector:] = 0
    elif step < 60000:
        h[..., gdr.d_scalar+gdr.d_vector+gdr.d_bivector:] = 0
    # else: full model
```

### 2.11. Predictive Coding (DPCM)

**Формула:** residual r_t = x_t - x̂_t, где x̂_t = f(x_{t-1}, x_{t-2}, ...)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| x̂_t (prediction) | GP2D geometric product | gp_out = GP(h[t], h[t±1]) |
| r_t (residual) | gate·proj(gp_out) | `residual = sigmoid(gate) * proj(gp_out)` |
| Encode | h = h + residual | `return h + residual, residual` |
| Whiteness | E[r_t · r_{t+1}] = 0 | whiteness_loss = cosine_sim(r_t, r_{t+1}) |

**Реализация V5:** GP2D с window=1, gate_init=-2.0 (sigmoid ≈ 0.12).
Whiteness loss penalizes lag-1 autocorrelation.

**Улучшение V6:** Multi-scale DPCM с adaptive prediction:

```python
# V6: multi-scale predictive coding
# Scale 1: adjacent (window=1) — fine prediction
# Scale 2: medium (window=4) — coarse prediction
# Scale 3: long-range (window=16) — structural prediction
# Each scale has own gate, learned independently

class MultiScaleGP2D(nn.Module):
    def __init__(self, cfg, hidden_size):
        super().__init__()
        self.scales = [(1, -2.0), (4, -3.0), (16, -4.0)]  # (window, gate_init)
        self.gp_layers = nn.ModuleList([
            GeometricProduct2D(cfg, hidden_size, window=w, gate_init=g)
            for w, g in self.scales
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))
    
    def forward(self, h):
        residual = sum(
            sw * gp(h)[1] for sw, gp in zip(softmax(scale_weights), gp_layers)
        )
        return h + residual, residual
```

### 2.12. Variable-Rate Coding / Conditional Computation

**Формула:** Разные части данных кодируются с разной rate

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| Rate allocation | Expert routing per token | top-1 из 4 experts |
| Zero-rate | MoD skip (identity) | `skip_out = x * prob` |
| Low rate | Simple expert | Expert 0-1 |
| High rate | Complex expert | Expert 2-3 |
| Rate control | Entropy-aware routing | `router_input = cat([x, entropy])` |

**Реализация V5:** MoE с 4 experts + MoD skip, entropy-aware routing.
Grade specialization loss коррелирует expert routing с grade dominance.

**Улучшение V6:** Hierarchical MoE — two-level routing:

```python
# V6: two-level MoE
# Level 1: route to "frequency band" (grade-based)
# Level 2: within band, route to specialized expert

class HierarchicalMoE(nn.Module):
    def __init__(self, cfg, hidden_size):
        super().__init__()
        # Level 1: grade-based router (4 bands)
        self.band_router = nn.Linear(hidden_size + 1, 4, bias=False)
        # Level 2: per-band experts (2 experts per band = 8 total)
        self.band_experts = nn.ModuleList([
            nn.ModuleList([SwiGLUExpert(...) for _ in range(2)] for _ in range(4))
        ])
    
    def forward(self, x):
        entropy = compute_entropy(x)
        band = softmax(self.band_router(cat([x, entropy])))  # [B*T, 4]
        # Route to band, then to expert within band
        ...
```

### 2.13. Polar Codes: Successive Cancellation

**Формула:** Полярное преобразование — каналы поляризуются (идеальные + бесполезные)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| Polar transform | Block-parallel generation | block_size=16 tokens per pass |
| Successive cancellation | Frozen context for next block | `full_ids = cat([full_ids, new_tokens])` |
| Frozen bits | Prompt tokens (unmasked) | mask[:, T-n_block:] = True |
| Information bits | Mask tokens (to generate) | mask_tokens appended |
| Rate | 1 - mask_ratio in generation | block_size / total_seq |

**Реализация V5:** Block-parallel generation (block_size=16).
Turbo refinement: 2 passes (rough + refined). Repetition penalty = echo cancellation.

**Улучшение V6:** Speculative block generation — predict next block while refining:

```python
# V6: speculative block generation
# While refining block N, start predicting block N+1
# If block N tokens are confident enough, commit and move on

def generate_speculative(model, prompt, max_tokens, block_size=16):
    blocks = []
    current_block = prompt
    
    while total_generated < max_tokens:
        # Pass 1: rough decode (with masks for current block)
        rough_tokens = model.generate_block(current_block, block_size)
        
        # SPECULATIVE: start next block while refining current
        speculative_input = cat([current_block, rough_tokens, mask_block])
        speculative_tokens = model.generate_block(speculative_input, block_size)
        
        # Pass 2: refined decode (current block unmasked, next block masked)
        refined_input = cat([current_block, rough_tokens, mask_block])
        refined_tokens = model.refine_block(refined_input, block_size)
        
        # Commit current block
        blocks.append(refined_tokens)
        current_block = cat([current_block, refined_tokens])
        
        # Use speculative prediction as starting point for next iteration
        # (saves one forward pass if speculation was correct)
    
    return cat([prompt] + blocks)
```

### 2.14. Distributed Source Coding (Knowledge Distillation)

**Формула:** KL(p_student || p_teacher) = rate-distortion для knowledge transfer

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| High-rate encoder | Teacher (SmolLM2-360M) | 360M params |
| Low-rate encoder | Student (HAGI V4/V5) | 74M params |
| Distortion | KL divergence | KL(softmax(s/T) || softmax(t/T)) |
| Rate controller | Temperature T | anneal 4.0 → 1.0 |
| Alpha schedule | Fitting → compression | 0.5 → 0.3 → 1.0 |

**Реализация V5:** KL distillation с temperature annealing.
Alpha schedule: 0.5 (50% CE + 50% KL) → 0.3 (30% CE + 70% KL) → 1.0 (pure CE).
Distillation ends at 60% of training.

**Улучшение V6:** Progressive distillation — multi-teacher cascade:

```python
# V6: progressive distillation from multiple teachers
# Phase 1 (0-20%): SmolLM2-135M (small, fast, coarse structure)
# Phase 2 (20-40%): SmolLM2-360M (medium, fine structure)
# Phase 3 (40-60%): SmolLM2-1.7B (large, dark knowledge) [optional]
# Phase 4 (60-100%): Pure CE (self-optimization)

teachers = [
    ("HuggingFaceTB/SmolLM2-135M", 0.0, 0.2),  # name, start_frac, end_frac
    ("HuggingFaceTB/SmolLM2-360M", 0.2, 0.4),
    # Optional: larger teacher
]
```

### 2.15. Transform Coding (DCT/Wavelet)

**Формула:** X → T(X) → quantize(T(X)) — transform domain compression

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| DCT transform | GDR cross-grade mixing | geometric_product_self_g02 |
| DC coefficient | Scalar output (g0) | global confidence |
| AC coefficients | Bivector output (g2) | relational structure |
| Quantization matrix | Grade momentum | per-grade protection |
| Inverse DCT | Linear projections | geo_to_scalar, geo_to_bivector |

**Реализация V5:** GDR cross-grade mixing: vector × vector → scalar + bivector.
Gated contribution via sigmoid gates.

**Улучшение V6:** Wavelet-like multi-resolution decomposition:

```python
# V6: multi-resolution grade decomposition
# Instead of flat Cl(3,0,0) grades, use hierarchical decomposition
# Level 0: scalar (64) — global average (lowest frequency)
# Level 1: vector (96) — entities (low frequency)  
# Level 2: bivector (96) — relations (mid frequency)
# Level 3: trivector (64) — higher-order (high frequency)
# Level 4: wavelet detail (256) — fine details (highest frequency)
#   Further decomposed into 2 sub-bands: [128 detail-coarse, 128 detail-fine]

class WaveletGDR(nn.Module):
    def __init__(self, cfg, hidden_size):
        super().__init__()
        # Standard GDR for grades 0-3
        self.gdr = GradeDecomposedRecurrence(cfg, hidden_size - 256)
        # Wavelet decomposition for residual (256 → 2×128)
        self.detail_coarse = nn.Linear(256, 128, bias=False)
        self.detail_fine = nn.Linear(256, 128, bias=False)
        self.detail_inverse = nn.Linear(128, 128, bias=False)
        # Wavelet-like mixing
        self.low_freq_gate = nn.Parameter(torch.tensor(0.0))
        self.high_freq_gate = nn.Parameter(torch.tensor(-2.0))
```

### 2.16. Channel Capacity with Erasure: C = 1 - p

**Формула:** Для erasure channel: C = 1 - p (где p = erasure probability)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| p (erasure prob) | Mask ratio | 0.15 → 0.35 (two-phase) |
| C (capacity) | 1 - mask_ratio | 0.85 → 0.65 |
| Erasure indicator | mask_embed | max-entropy init (V5) / mean embed (current) |

**Реализация V5:** mask_embed = mean of embeddings (current code).
`self.mask_embed.data.copy_(self.embed.weight.mean(dim=0))`.

**Внимание:** ARCHITECTURE_V5.md специфицирует max-entropy init, но текущий
код использует mean embed init. Это расхождение нужно исправить.

**Улучшение V6:** Multi-level erasure channel (BEC с multiple rates):

```python
# V6: multi-level erasure — разные mask ratios для разных grades
# Scalar: low mask (0.1) — DC protected (UEP)
# Vector: medium mask (0.2) — low AC medium protection
# Bivector: high mask (0.4) — high AC less protected
# Trivector: high mask (0.4) — highest AC least protected

def multi_level_mask(h, grade_bounds, mask_ratios):
    """Multi-level erasure: different mask rates per grade."""
    masks = []
    for (start, end), ratio in zip(grade_bounds, mask_ratios):
        grade_mask = torch.bernoulli(torch.full((B, T), ratio)).bool()
        masks.append(grade_mask)
    return combine_grade_masks(masks, grade_bounds)
```

### 2.17. Water-Filling Solution

**Формула:** Оптимальное распределение power: P_i = max(0, μ - 1/SNR_i)

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| P_i (power) | Dimensions per grade | [64, 96, 96, 64, 256] fixed |
| SNR_i | 1/variance per grade | measured via log_grade_variance |
| μ (water level) | Total capacity budget | total_dims = 288 (compressed) |

**Реализация V5:** Fixed grade dims [64, 96, 96, 64, 256].
Grade variance logging for monitoring, но не для adaptation.

**Улучшение V6:** Dynamic water-filling — перераспределение dims:

```python
class WaterFillingAllocator(nn.Module):
    """Dynamic capacity allocation across grades based on measured variance.
    
    Water-filling principle: allocate more dims to grades with higher variance
    (= higher entropy = more capacity needed).
    """
    def __init__(self, total_dims, min_dims=8, adaptation_rate=0.001):
        super().__init__()
        self.total_dims = total_dims
        self.min_dims = min_dims
        self.adaptation_rate = adaptation_rate
        # Learnable soft allocation via temperature-controlled softmax
        self.allocation_logits = nn.Parameter(torch.zeros(4))  # 4 grades
    
    def get_allocation(self):
        """Returns dims per grade via softmax with constraint."""
        probs = F.softmax(self.allocation_logits, dim=-1)
        dims = [max(self.min_dims, int(self.total_dims * p)) for p in probs]
        # Adjust to match total
        while sum(dims) > self.total_dims:
            idx = argmax(dims)
            dims[idx] -= 1
        while sum(dims) < self.total_dims:
            idx = argmin(dims)
            dims[idx] += 1
        return dims
```

### 2.18. Joint Source-Channel Coding

**Формула:** Для finite block length, joint coding > separate coding

| Параметр | Аналог в модели | Текущая реализация |
|----------|-----------------|-------------------|
| Separate coding | Bottleneck + GP2D separate | V5 separation |
| Joint coding | End-to-end optimization | gradient flows through all |
| Finite block | Token sequence (T=512) | seq_len = 512 |

**Реализация V5:** Shannon separation theorem implemented: source encoder
(bottleneck) separate from channel encoder (GP2D). Gradients flow through both.

**Улучшение V6:** Joint optimization via shared latent:

```python
# V6: joint source-channel coding
# Instead of separate bottleneck + parity, use a single joint encoder
# that simultaneously compresses and adds redundancy

class JointSourceChannelEncoder(nn.Module):
    def __init__(self, input_dim, compressed_dim, parity_dim):
        super().__init__()
        # Joint: compress to compressed_dim, add parity_dim redundancy
        self.joint_proj = nn.Linear(input_dim, compressed_dim + parity_dim)
        self.compressed_dim = compressed_dim
        self.parity_dim = parity_dim
    
    def forward(self, h):
        out = self.joint_proj(h)
        z = out[..., :self.compressed_dim]  # source-coded part
        parity = out[..., self.compressed_dim:]  # channel-coded part
        return z, parity
```

---

## 3. Gap Analysis: V5 vs Shannon Limit

### 3.1. Что V5 делает хорошо

| Компонент | Shannon аналог | Качество |
|-----------|---------------|----------|
| Extrinsic separation | LDPC/Turbo decoding | Отлично — prevents info recycling |
| Convergence halt | EXIT chart | Хорошо — теоретически обоснован |
| Grade structure (UEP) | Unequal Error Protection | Отлично — semantic-aware |
| GP2D parity | Systematic channel code | Хорошо — geometric product = parity |
| Iterative refinement | Successive refinement | Хорошо — progressive JPEG analog |
| MSA side information | Slepian-Wolf coding | Хорошо — 18:1 compression |
| MoE variable-rate | Conditional computation | Хорошо — entropy-aware routing |
| Bidirectional attention | Parallel decompression | Отлично — transform coding |
| Bottleneck (H→H/2) | Source coding / IB | Хорошо — 4x compute reduction |

### 3.2. Gap: что отделяет от Shannon limit

| Gap | Impact | Решение V6 |
|-----|--------|-----------|
| Фиксированный bottleneck | MEDIUM | Variational IB с learned variance |
| Single-scale GP2D parity | MEDIUM | Multi-scale LDPC-like parity |
| Alpha=1.0 (no damping) | LOW | Adaptive alpha per iteration |
| Single decoder (no turbo) | HIGH | Turbo decoder: attention + MSA |
| Fixed grade dims | MEDIUM | Water-filling dynamic allocation |
| No KV cache at inference | HIGH | KV cache for generation |
| Sequential block generation | HIGH | Speculative block generation |
| Global mask ratio | LOW | Per-position adaptive masking |
| Mean embed for mask | LOW | Max-entropy init (fix discrepancy) |
| No progressive capacity | LOW | Progressive capacity growth |

### 3.3. Метрики близости к Shannon limit

```
Shannon limit для языковой модели:
  C_shannon = H(language) / bits_per_token = 3.3 / 15.6 ≈ 0.21 (compression ratio)

V5 current:
  Compression: log2(perplexity) / log2(vocab) ≈ 3.3 / 15.6 ≈ 0.21
  → Теоретически на пределе (если perplexity = 10)
  
Но:
  1. Perplexity выше идеального (не 10, а ~15-30 в реальности)
  2. Effective bits per param < 16 (bf16 noise)
  3. Inference latency не оптимален (recompute everything)

V6 цели:
  1. Perplexity ↓ 20% (variational IB + turbo decoder)
  2. Effective bits per param ↑ 50% (mixed-precision + water-filling)
  3. Inference latency ↓ 2x (KV cache + speculative generation)
  4. Training speed ↑ 30% (progressive capacity + better curriculum)
```

---

## 4. V6 Architecture — Shannon-Optimal Design

### 4.1. Pipeline V6

```
Token IDs [B, T]
    │
    ▼
┌──────────────────────────────────────────────────┐
│  SOURCE ENCODER (variational IB)                  │
│                                                   │
│  Embed → Perception (2 layers, H=576)            │
│  → Variational Bottleneck:                        │
│    μ = Linear(H→C), σ = Linear(H→C)              │
│    z = μ + ε·σ (reparameterization)              │
│    KL(q(z|x)||N(0,1)) = IB regularizer           │
│  → Compressed Latent Z [B, T, C=288]             │
│                                                   │
│  Цель: min KL + β·max I(Z;Y)                     │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  CHANNEL ENCODER (multi-scale LDPC-like parity)   │
│                                                   │
│  Z → Multi-Scale GP2D:                            │
│    Scale 1: window=1 (adjacent parity)            │
│    Scale 2: window=4 (mid-range parity)           │
│    Scale 3: window=16 (long-range parity)         │
│  → Interleaving between scales (burst protection) │
│  → Coded Latent Z' [B, T, C=288]                 │
│                                                   │
│  Цель: ||multi-scale parity|| = redundancy        │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  CHANNEL (multi-level erasure)                    │
│                                                   │
│  Per-grade mask ratios (UEP):                     │
│    Scalar:    p=0.10 (max protection)             │
│    Vector:    p=0.20 (medium protection)          │
│    Bivector:  p=0.35 (standard)                   │
│    Trivector: p=0.35 (standard)                   │
│  mask_embed = max-entropy vector                  │
│                                                   │
│  C_grade = 1 - p_grade (per-grade capacity)       │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  CHANNEL DECODER (Turbo BP — dual component)      │
│                                                   │
│  Component A: Attention-based (local parity)      │
│    → 7 reasoning layers → extrinsic_A             │
│                                                   │
│  Component B: MSA-based (long-range parity)       │
│    → memory read + refinement → extrinsic_B       │
│                                                   │
│  Exchange: h = h + α_A·ext_A + α_B·ext_B          │
│  Adaptive α: based on per-component convergence   │
│  Convergence: ||ext_A|| + ||ext_B|| < ε → stop    │
│                                                   │
│  Water-filling: dynamic grade dim allocation      │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  SOURCE DECODER (soft belief generation)          │
│                                                   │
│  Bottleneck up (C→H) → Expression (2 layers)      │
│  → Soft belief state P(token | context) [B,T,V]   │
│  → Commit only when confidence > threshold        │
│  → KV cache for efficient autoregressive fallback │
│                                                   │
│  Speculative: predict next block while refining   │
└──────────────────────────────────────────────────┘
```

### 4.2. Новые модули V6

| Модуль | Файл | Shannon аналог | Impact |
|--------|------|---------------|--------|
| VariationalBottleneck | `model/variational_bottleneck.py` | VAE source coding | MEDIUM |
| MultiScaleGP2D | `model/multiscale_gp2d.py` | LDPC multi-degree parity | MEDIUM |
| TurboDecoder | `model/turbo_decoder.py` | Turbo code dual decoder | HIGH |
| WaterFillingAllocator | `model/water_filling.py` | Water-filling capacity | MEDIUM |
| SpeculativeGenerator | `inference/speculative.py` | Speculative decoding | HIGH |
| KVCache | `inference/kv_cache.py` | Incremental decoding | HIGH |
| MultiLevelMasking | `model/multi_level_mask.py` | UEP erasure channel | LOW |
| ProgressiveCapacity | `train/progressive.py` | SVC scalable coding | LOW |

### 4.3. Training Objective V6

```
Loss = CE                              # Fidelity: I(Z;Y)
     + λ₁ · KL_variational             # Compression: -I(X;Z) via VAE
     + λ₂ · MultiScale_Parity          # Redundancy: multi-scale GP2D
     + λ₃ · Turbo_Extrinsic            # Decoding: dual extrinsic info
     + λ₄ · Efficiency                 # Convergence: iterations
     + λ₅ · Water_Filling_reg          # Capacity: balanced allocation
     + aux losses (MoE/GDR/MSA/coherence/whiteness/grade_spec)
```

### 4.4. Configuration V6

```yaml
model:
  hidden_size: 576
  core_hidden_size: 288
  perception_layers: 2
  reasoning_layers: 7
  expression_layers: 2

  variational_bottleneck:
    enabled: true
    kl_weight: 0.01              # λ₁
    prior: "standard_normal"     # p(z) = N(0, I)

  multiscale_gp2d:
    enabled: true
    scales: [1, 4, 16]           # window sizes
    gate_inits: [-2.0, -3.0, -4.0]
    interleave: true             # burst error protection
    parity_weight: 0.1           # λ₂

  turbo_decoder:
    enabled: true
    component_a: "attention"     # local parity
    component_b: "msa"           # long-range parity
    alpha_a: 0.8                 # adaptive
    alpha_b: 0.8                 # adaptive
    convergence_threshold: 0.01
    min_iterations: 1
    max_iterations: 6

  water_filling:
    enabled: true
    adaptation_rate: 0.001
    min_dims: 8

  multi_level_mask:
    enabled: true
    grade_mask_ratios: [0.10, 0.20, 0.35, 0.35]  # scalar, vec, bi, tri
    mask_embed_init: "max_entropy"

  progressive_capacity:
    enabled: true
    stages: [30000, 60000, 90000]  # grade activation steps

  generation:
    type: "speculative"
    block_size: 16
    refine_passes: 2
    kv_cache: true
    commit_threshold: 0.8
    belief_momentum: 0.7

train:
  loss_weights:
    ce: 1.0
    kl_variational: 0.01         # λ₁
    multiscale_parity: 0.1       # λ₂
    turbo_extrinsic: 0.01        # λ₃
    efficiency: 0.001            # λ₄
    water_filling_reg: 0.001     # λ₅
```

---

## 5. Оптимизации: быстстрое обучение + быстрая генерация

### 5.1. Быстрое обучение (Source/Channel Encoding)

| Оптимизация | Ускорение | Механизм |
|-------------|-----------|----------|
| Bottleneck в core (H/2) | 4x compute | Уже в V5 |
| Progressive capacity | 1.3x early | Активация grades поэтапно |
| Mixed-precision (bf16) | 2x memory | Уже в V5 |
| Gradient checkpointing | 1.5x memory | Уже в V5 |
| Entropy-ordered curriculum | 1.2x convergence | Low entropy → high entropy |
| Adaptive iterations | 1.5x late | Fewer iters for easy inputs |
| Multi-teacher distillation | 1.3x convergence | Progressive teacher cascade |

**Совокупное ускорение training:** ~2.5x vs V5

### 5.2. Быстрая генерация (Source/Channel Decoding)

| Оптимизация | Ускорение | Механизм |
|-------------|-----------|----------|
| Block-parallel (16 tokens) | 16x vs autoregressive | Уже в V5 |
| KV cache | 2x inference | No recompute for frozen context |
| Speculative block generation | 1.5x | Predict next while refining |
| Early exit (confidence) | 1.3x | Stop when confident |
| 1-iteration inference | 4x vs 4-iter | Уже в V5 |
| Turbo convergence | 1.2x | Faster convergence with dual decoder |

**Совокупное ускорение inference:** ~3x vs V5, ~48x vs autoregressive

### 5.3. Максимизация пропускной способности

| Метод | Capacity gain | Механизм |
|-------|--------------|----------|
| Variational IB | +20% | Optimal compression → more effective bits |
| Water-filling | +15% | Optimal dim allocation per grade |
| Multi-scale parity | +10% | Better error correction → higher effective rate |
| Turbo decoding | +10% | Better extrinsic extraction → more info per iter |
| Mixed-precision | +50% | More effective bits per parameter |
| Multi-level masking | +5% | UEP → protect critical info, save capacity |

**Совокупный gain capacity:** ~2x vs V5

---

## 6. Формулы — полный референс

### 6.1. Shannon Capacity (erasure channel)
```
C = 1 - p

p = erasure probability = mask_ratio
C = 1 - mask_ratio = effective channel capacity
```

### 6.2. Information Bottleneck
```
F_β = I(X;Z) - β·I(Y;Z) → min

I(X;Z) ≈ KL(q(z|x) || p(z))    (variational upper bound)
I(Y;Z) ≈ H(Y) - H(Y|Z) = H(Y) - CE(logits, y)
```

### 6.3. Rate-Distortion
```
R(D) = min I(X;X̂)  s.t.  E[d(X,X̂)] ≤ D

d(X,X̂) = CE loss
R = model capacity (bits)
D = prediction error
```

### 6.4. Extrinsic Information (LDPC/Turbo)
```
extrinsic_k = h_out_k - h_prior_k
posterior_k = h_prior_k + α · extrinsic_k

Convergence: ||extrinsic_k|| < ε → halt
```

### 6.5. Slepian-Wolf (side information)
```
R(X|Y) = H(X|Y) = H(X) - I(X;Y)

ΔR = I(X;Y) = capacity savings from side info
MSA: ΔR = info from top-k memory slots
```

### 6.6. Water-Filling
```
P_i = max(0, μ - 1/SNR_i)  s.t.  Σ P_i = P_total

SNR_i = 1/var_i (per grade)
P_i = dims allocated to grade i
μ = water level (total capacity budget)
```

### 6.7. Unequal Error Protection
```
bits_i ∝ -log(1 - momentum_i)

scalar:    -log(1 - 0.8) = 2.32  (max protection)
vector:    -log(1 - 0.5) = 1.00  (medium)
bivector:  -log(1 - 0.0) = 0.00  (no protection)
```

### 6.8. EXIT Chart (convergence prediction)
```
I_ext(k) = ||extrinsic_k||

Convergence rate: ΔI = I_ext(k) - I_ext(k+1)
Halt when: I_ext(k) < ε  OR  ΔI < δ (rate slowing)
```

### 6.9. Adaptive Modulation (capacity matching)
```
p = 1 - confidence  (global V5)
p_i = 1 - confidence_i  (per-position V6)

High confidence → high p (harder, push capacity)
Low confidence → low p (easier, ensure recovery)
```

### 6.10. Progressive Refinement
```
D_i = D_0 · exp(-α · i)  (exponential distortion decay)

w_i = E[||h_i - h_{i-1}||²]  (adaptive weight)
Σ w_i = 1 (normalized)
```

### 6.11. Predictive Coding (DPCM)
```
r_t = x_t - x̂_t
x̂_t = GP(x_{t-1}, x_t, x_{t+1})  (geometric product prediction)

Whiteness: E[r_t · r_{t+1}] = 0  (optimal prediction)
Loss: |cosine_sim(r_t, r_{t+1})| → 0
```

### 6.12. Hallucination Floor (Guo & Li)
```
R_min = n · KL(μ_K || μ_N)  [bits]

H_floor = max(0, 1 - C / R_min)

C = model capacity = params × bits/param
n = number of facts
KL = divergence between fact/non-fact confidence
```

---

## 7. Implementation Roadmap

### Phase 1: High-impact modules (fast generation)

1. **KV Cache** (`inference/kv_cache.py`) — 2x inference speedup
2. **Speculative Block Generation** (`inference/speculative.py`) — 1.5x speedup
3. **Turbo Decoder** (`model/turbo_decoder.py`) — better quality + faster convergence

### Phase 2: Capacity optimization

4. **Variational Bottleneck** (`model/variational_bottleneck.py`) — optimal compression
5. **Water-Filling Allocator** (`model/water_filling.py`) — optimal dim allocation
6. **Multi-Scale GP2D** (`model/multiscale_gp2d.py`) — better error correction

### Phase 3: Training optimization

7. **Progressive Capacity** (`train/progressive.py`) — faster early training
8. **Multi-Level Masking** (`model/multi_level_mask.py`) — UEP erasure
9. **Config + Model updates** — integrate all V6 components

### Phase 4: Verification

10. **Build + tests** — ensure all modules work
11. **Benchmark** — measure capacity gain + speedup

---

## 8. Ожидаемые результаты

| Метрика | V5 | V6 (expected) | Improvement |
|---------|-----|---------------|-------------|
| Perplexity | ~15-30 | ~12-24 | -20% |
| Effective bits/param | ~4 | ~6 | +50% |
| Inference latency | 1x | 0.33x | 3x faster |
| Training convergence | 1x | 0.4x | 2.5x faster |
| Channel capacity | 1x | 2x | 2x higher |
| Params | 74M | 74M | same |
| Memory (training) | 8GB | 8GB | same |
| Memory (inference) | 1x | 0.8x | 20% less (KV cache) |
