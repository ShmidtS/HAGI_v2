# HAGI-2 V30: Design & Development Plan

---

## 1. Архитектурный план (Architecture Plan)

### 1.0 Конфигурационная система (`config.py`)

Единый источник истины: `Config` (dataclass, 3 секции — `model`, `train`, `inference`).
Никаких магических констант в коде тренировочного цикла или модели.

**Иерархия конфигов:**

```
Config
  model: ModelConfig
    vocab_size: int = 49154
    hidden_size: int = 384          # H
    core_hidden_size: int = 192     # C  (C < H — реальное сжатие)
    norm_eps: float = 1e-5
    target_params: int = 0          # авто-расчёт всех размеров из бюджета
    target_nonembed_params: int = 0
    attention: AttentionConfig
      num_query_heads: int = 8       # n_q
      num_kv_heads: int = 4         # n_kv <= n_q, n_q % n_kv == 0 (GQA)
      head_dim: int = 64            # n_q * hd == H
      rope_theta: float = 10000.0
      max_seq_len: int = 4096
      sliding_window: int = 0       # 0 = full; >0 = windowed causal
    sliding: SlidingWindowConfig
      sliding_window: int = 0
      window_layers: tuple[int] = () # explicit per-layer window mask
      full_every: int = 0           # every Nth layer full (global parity relay)
    embeddings: EmbeddingsConfig
      factor_rank: int = 128        # r — rank of V×H factorization
      kernel_size: int = 5          # causal depthwise Conv1d
      trainable: bool = True
    multimodal: MultimodalConfig
      enabled: bool = False
      image_patch_size: int = 16
      image_channels: int = 3
      audio_mel_bins: int = 80
      num_modalities: int = 3
      n_bridge_queries: int = 32    # O(1) multimodal prefix tokens
      bridge_layers: int = 2
      bridge_n_heads: int = 4
    grounded: GroundedInfomaxConfig
      vicreg_var_weight: float = 1.0
      vicreg_inv_weight: float = 1.0
      vicreg_cov_weight: float = 0.04
      vicreg_gamma: float = 1.0
      infonce_temperature: float = 0.07
    body: BodyConfig
      num_layers: int = 12          # L
      ternary: TernaryConfig
        use_ternary: bool = True    # BitNet b1.58
        hebbian_expansion: int = 4  # m = expansion * H
        eps: float = 1e-5           # per-channel scale floor
      bottleneck: BottleneckConfig
        dim: int = 192              # C — IB latent dim
        ib_beta: float = 0.001
        kl_free_bits: float = 0.01  # per-dim KL floor
        logvar_clamp: tuple = (-5.0, 5.0)
        distortion_eps: float = 1e-6
      moe: MoEConfig
        enabled: bool = False
        num_experts: int = 4
        top_k: int = 1
        n_shared: int = 1
        moe_every: int = 2          # MoE on every Nth block
        intermediate_size: int = 0  # 0 -> derived from expansion*H
        snr_gate_weight: float = 1.0
        route_entropy_weight: float = 0.01
    refinement: RefinementConfig
      enabled: bool = False
      iterations: int = 2
      update_hidden: int = 0        # 0 -> H
      hep_enabled: bool = True      # zero-init HEP feedback
      exit_threshold: float = 0.05
      exit_min_steps: int = 50
      exit_window: int = 5

  train: TrainConfig
    max_steps: int = 150000
    warmup_steps: int = 1600
    learning_rate: float = 0.0003    # AdamW base LR
    weight_decay: float = 0.1
    precision: str = "bf16"
    grad_accum_steps: int = 2
    max_grad_norm: float = 0.5
    batch_size: int = 8
    seq_len: int = 512
    muon: MuonConfig
      lr: float = 0.02
      momentum: float = 0.95
      nesterov: bool = True
      ns_steps: int = 5
      weight_decay: float = 0.1
      ns_coeffs: tuple = (3.4445, -4.7750, 2.0315)
      wd_cap: float = 2.0
      momentum_offload: bool = True
    adam: AdamConfig
      beta1: float = 0.9
      beta2: float = 0.95
      eps: float = 1e-8
    schedule: ScheduleConfig
      stable_fraction: float = 0.8
      min_lr_ratio: float = 0.1
    # Loss weights
    w_ce: float = 1.0
    w_rate: float = 0.01
    w_distortion: float = 0.01
    w_vicreg: float = 0.1
    w_infonce: float = 0.1
    w_moe_lb: float = 0.01
    w_route_entropy: float = 0.01
    w_water_filling: float = 0.01
    w_refine: float = 0.0
    w_attn_entropy: float = 0.01
    attn_entropy_floor: float = 0.0  # 0 = auto (1/n_q)
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
    eos_token_id: int = 0
    pad_token_id: int = 49152
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    checkpoint_keep_last: int = 3
    curriculum: CurriculumConfig
      enabled: bool = True
      stage2_start: int = 100000
      cycles_per_dataset: int = 3
      stage1_order: list[str] = [tinystories, python_instruct, smoltalk, ...]
      stage2_datasets: list[str] = [openwebmath, edu, slimpajama]
    distill: DistillationConfig
      enabled: bool = False
      teacher: str = "HuggingFaceTB/SmolLM2-360M"
      teacher_hidden_size: int = 960
      alpha_start/end, temperature, end_frac ...

  inference: InferenceConfig
    temperature: 0.8, top_k: 50, max_new_tokens: 64,
    repetition_penalty: 1.2, no_repeat_ngram_size: 2
```

**`auto_configure(target_params, vocab_size, num_experts)`** — решает ВСЕ размеры из бюджета:
- `L = round(log10(target/1e5) * 3)`
- `H = sqrt(target / (cost_per_h2 * L))`, cost_per_h2 = 2 (attn) + 12 (FFN) + MoE overhead
- `C = H // 2`
- `n_q, n_kv, head_dim`: поиск по сетке с предпочтением head_dim в [64, 128]
- `factor_rank = min(256, C)`
- 3 итерации до сходимости

**`layer_sliding_windows(model)`** — разрешает per-layer окна: explicit `window_layers` > `full_every` > all-windowed > all-full.

**Валидация** (`validate_config`): ~40 инвариантов — C < H, n_kv <= n_q, n_q*hd == H, L % moe_every == 0, и т.д.

---

### 1.1 Основной путь (Main LM Path) — `model/model.py`

#### HAGI (nn.Module)

**Стадия 0 — инициализация:**

```python
self.embed = ConvEmbedding(V, H, r, K, eps)      # source encoder
self.blocks = nn.ModuleList([TransformerBlock...]) # L блоков
self.final_norm = RMSNorm(H)                      # pre-head norm
self.bottleneck = InformationBottleneck(H, C, ...) # off-path IB
self.lm_compress = Linear(H, r, bias=False)        # factored LM head
self.lm_expand = Linear(r, V, bias=False)
```

**Стадия 1 — Source Encode:**
```python
h = self.embed(input_ids)           # [B, T_text] -> [B, T_text, H]
# Multimodal (если enabled):
#   h = concat([image_bridge, audio_bridge, h_text])
#   prefix_len = n_bridge * num_present_modalities
```

**Стадия 2 — Unified Ternary Stack:**
```python
h_ctx = self._stack_forward(h, attention_mode, prefix_len, soft_beta, positions)
# L проходов через TransformerBlock с gradient checkpointing
# Собирает: _last_attn_entropy_penalty, _last_moe_lb, _last_route_entropy, _last_water_filling
```

**Стадия 3 — Auxiliary IB (off-path):**
```python
bn_info = self.bottleneck(h_ctx.detach())  # rate, distortion
```

**Стадия 4 — Main LM Path:**
```python
h_dec = self.final_norm(h_ctx)
h_text = h_dec[:, prefix_len:]            # текстовые позиции
logits = lm_expand(lm_compress(h_text))   # factored head
ce_loss = cross_entropy(logits, targets)
```

**Стадия 4b — Off-path HEP Refinement (opt-in):**
```python
ref_loss, h_refined = self.refinement_head(h_ctx.detach(), main_logits, targets, ...)
```

**Стадия 5 — Grounded Infomax (multimodal, on h_ctx без detach):**
```python
vicreg, infonce = self.multimodal_input.grounded(h_ctx, modality_ids)
```

**Return:** `ModelOutput(logits, hidden, aux=AuxLosses(...), ce_loss, prediction_indices)`

#### Вспомогательные методы

| Метод | Назначение |
|---|---|
| `allocate_for_cache(B, dtype, device)` | Создать L x KVCache для инкрементальной генерации |
| `reset_cache()` | Сбросить все кеши (KV + conv history) |
| `commit_moe_ema_updates()` | Применить deferred MoE SNR EMA после backward |
| `ensure_fp32_params()` | Конвертировать IB параметры в FP32 после bf16 cast |
| `lm_head_weight` (property) | Материализовать `expand @ compress` формы `[V, H]` |

---

### 1.2 Source Encoder — `model/conv_embedding.py`

**ConvEmbedding(V, H, r, K, eps):**

```
token_compress:  Embedding(V, r)           [V, r]     — FP (AdamW)
token_expand:    Linear(r, H, bias=False)   [H, r]     — FP (AdamW)
local_conv:      Conv1d(H, H, K, groups=H, padding=0) [H, 1, K] — FP
norm:            RMSNorm(H)                 [H]        — FP
```

**Forward:**
1. `compressed = token_compress(input_ids)` — дискретные символы -> ранг-r латент
2. `h = token_expand(compressed)` — ранг-r -> размерность канала H
3. `h = left-pad(h, K-1); h = local_conv(h)` — CAUSAL pulse-shaping filter
4. `h = norm(h)` — нормализация перед телом

**KV-cache awareness:** при инкрементальном декодинге сохраняет последние `K-1` токенов в `_conv_cache` и prepend'ит их к новому токену. Только в eval mode.

**Инвариант:** left-pad только. V25 symmetric-pad conv утекал будущие токены в `hidden[t]` — root cause #4 мусора на инференсе.

---

### 1.3 TransformerBlock — `model/block.py`

```
TransformerBlock(H, attn_cfg, ffn_cfg, norm_eps, use_ternary, mixer=None):

  self.attn = Attention(H, attn_cfg, use_ternary=use_ternary)
  self.ffn  = mixer or HebbianBilinearFFN(H, ffn_cfg, use_ternary=use_ternary)

  is_moe: bool   # True когда mixer is MoESwiGLU
  moe: MoESwiGLU | None
```

**Forward (pre-norm residual):**
```python
attn_out, entropy_pen = self.attn(x, attention_mode, prefix_len, soft_beta, positions)
x = x + attn_out
x = self.ffn(x)
```

Атрибуты после forward:
- `_last_attn_entropy_penalty` — anti-collapse метрика
- `moe.last_load_balance`, `moe.last_routing_entropy`, `moe.last_water_filling_loss` — если MoE-блок

---

### 1.4 Attention (GQA + RoPE + sliding window) — `model/attention.py`

```
Attention(H, cfg, norm_eps, use_ternary=True):

  attn_norm:  RMSNorm(H)                              [H]        — FP
  q_proj:     BitLinear(H, n_q * hd, bias=False)       [n_q*hd, H] — тернарный
  kv_proj:    BitLinear(H, 2 * n_kv * hd, bias=False)  [2*n_kv*hd, H] — тернарный
  out_proj:   BitLinear(n_q * hd, H, bias=False)       [H, n_q*hd] — тернарный
  rope:       RotaryEmbedding(head_dim, theta)          — без параметров

  attn_entropy_floor: float
  sliding_window: int | None   # per-layer, задаётся снаружи
  _kv_cache: KVCache | None
```

**Параметры на слой:** `(2*n_q + 2*n_kv) * hd * H + H`

**Режимы внимания (4 штуки):**

| Режим | Маска | Применение |
|---|---|---|
| `causal` | Треугольная (верхний правый угол -inf) | Тренировка + инференс |
| `bidir` | Нет маски (всё видно) | Опционально, early training |
| `prefix` | Префикс полностью видим (bidir), текст causal | Multimodal |
| `soft_causal` | Экспоненциальный декей `exp(-beta * distance)` | Опционально |

**Sliding window:** дополнительно ограничивает каждый query последними W ключами. Комбинируется с любым режимом. `_build_mask()` / `_build_full_mask()` применяют окно поверх базовой маски.

**Anti-collapse penalty (training only):**
- Если `attn_entropy_floor > 0` и `self.training` — строит полную матрицу scores `[B, n_q, T, T]` и вычисляет `entropy_pen = mean(clamp(floor - H(attn), 0))`
- **Осторожно:** на 8B+ моделях при T=8192 это 4 GB на слой — OOM на backward recomputation

**KV-cache:**
- `attach_cache(cache)` / `detach_cache()` — привязать/отвязать KVCache
- При активном кеше: кеширует (k, v), читает полную историю, применяет маску над `[t_q, t_total]`
- `positions` авто-смещаются на `cache_len` для правильного RoPE

---

### 1.5 HebbianBilinearFFN — `model/hebbian_ffn.py`

```
HebbianBilinearFFN(H, cfg, norm_eps, use_ternary=True):
  m = expansion * H = 4H

  norm:  RMSNorm(H)               [H]        — FP
  A0:    BitLinear(H, m)           [m, H]     — тернарный
  A1:    BitLinear(H, m)           [m, H]     — тернарный
  W:     BitLinear(m, H)           [H, m]     — тернарный
  gate:  Parameter(H)              [H]        — FP (AdamW)
```

**Forward:**
```python
h = norm(x)
phi = A0(h) * silu(A1(h))           # билинейная карта признаков
return x + W(phi) * (1 + tanh(gate)) # gate модуляция
```

**Параметры на слой:** `3 * 4H^2 + 2H`

**Gate modulation:** `(1 + tanh(gate))`, инициализация `gate = 0`. На старте множитель = 1.0 — ветка жива, градиент течёт. Zero-init multiplier убил бы градиент (dead-gradient failure mode).

---

### 1.6 Ternary Quantization — `model/ternary.py`

**`ternarize(weight, eps)` — BitNet b1.58:**
```python
scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)  # [out, 1]
qweight = (weight / scale).clamp(-1, 1).round()                # {-1, 0, +1}
effective_w = qweight * scale                                   # {-scale, 0, +scale}
```

- Per-OUTPUT-channel absmean scale
- Нулевой бин IMPLICIT (`round(|w/scale| < 0.5) -> 0`) — не TWN explicit threshold
- 1.585 бит/вес vs FP32 32 бит/вес

**`_TernarizeSTE` (autograd.Function):**
- Forward: тернаризация
- Backward: identity STE (saturated-region градиенты НЕ зануляются — Muon-совместимый транспорт)
- Заменяет `w + (eff - w).detach()` — экономит ~17k detach/шаг + MulBackward0 узлы

**`BitLinear(in, out, bias=False, eps=1e-5)`:**
- Хранит FP master `self.weight` формы `[out, in]`
- `forward(x)`: тернаризует в `w_ste = _TernarizeSTE.apply(weight, eps)`, затем `F.linear(x, w_ste)`
- Inference fast path: без `torch.is_grad_enabled()` — тернаризация без STE overead
- **ТОЛЬКО для 2D hidden mixing матриц** — bias/gate/codebook остаются FP

---

### 1.7 Information Bottleneck — `model/bottleneck.py`

```
InformationBottleneck(H, cfg, norm_eps):

  norm:        RMSNorm(H)               [H]     — FP
  to_mu:       Linear(H, C, bias=False) [C, H]  — FP32 (численная стабильность)
  to_logvar:   Linear(H, C, bias=False) [C, H]  — FP32
  decompress:  Linear(C, H, bias=False) [H, C]  — FP32
```

**Параметры:** `3*C*H + H`

**Forward (off-path, на h_ctx.detach()):**
```python
h_n = norm(h).float()
mu = to_mu(h_n)
logvar = clamp(to_logvar(h_n), -5, 5)

# Репараметризация (training) / детерминированно (eval)
z = mu + std * randn  (training)
z = mu                (eval)

h_hat = decompress(z)

rate = mean(clamp(0.5*(mu^2 + exp(logvar) - logvar - 1), min=free_bits))
distortion = ||h - h_hat||^2 / (||h||^2 + eps)  # detached denominator
```

**`ensure_fp32()`:** конвертирует `to_mu`, `to_logvar`, `decompress` в FP32 после `cast_to_bf16(model)`. Без этого bf16 logvar коллапсирует → rate unbounded (0.93 → 4.23 за 48K шагов) → CE конфликтует.

**`kl_free_bits`:** per-dimension пол. При 0.01 даёт ~2.56 nats/token для C=256. При 0.5 (smollm2.yaml) = 96 nats/token — практически всегда clamped (фактически фиксированный rate).

---

### 1.8 MoE (Water-Filling) — `model/moe.py`

**WaterFillingMoE(H, inter, cfg, norm_eps, use_ternary):**

```
norm:            RMSNorm(H)                                     [H]        — FP
shared_experts:  n_shared x SwiGLUExpert(H, inter, ternary)     — тернарные
experts:         num_experts x SwiGLUExpert(H, inter, ternary)  — тернарные
router:          Linear(H, num_experts, bias=False)              [E, H]     — FP (AdamW)
allocator:       WaterFillingAllocator(total_width=E*inter, E)  [2E]       — FP
```

**SwiGLUExpert:** `down(silu(gate(x)) * up(x))` — 3 BitLinear. Параметры: `3 * inter * H` на эксперта.

**Forward (на x [B, T, H]):**
1. `flat = norm(x).reshape(B*T, H)`
2. Shared experts: `shared_out = sum(se(flat) for se in shared_experts)`
3. Residual + SNR gate: `residual = flat - shared_out`, `snr = w_snr / ||residual||_2`
4. Router: `logits = router(flat) * (1 + snr)` — SNR-gated
5. Allocator bias: `probs = softmax(logits + log(alloc_probs))`
6. Top-k: `topk_vals, topk_idx = probs.topk(k); topk_vals /= sum(topk_vals)`
7. **Batched dispatch** (`_batched_dispatch`):
   - Flatten (token, slot) → sort by expert → segment offsets
   - `index_select(residual, sel_tokens)` → `expert(sel_res) * sel_weights` → `index_add_(out, sel_tokens, expert_out)`
   - ОДИН `.tolist()` host sync (вместо `2*E` отдельных `.item()`)
8. `return x + (shared_out + routed_out)`

**Deferred EMA:**
- `_deferred_residual`, `_deferred_probs` — сохраняются в forward
- `commit_ema_update()` — вызывается из train loop ПОСЛЕ backward
- Причина: под `ckpt.checkpoint(use_reentrant=False)` save/recompute должны видеть ОДИНАКОВЫЙ `snr_ema`, иначе routing log-bias → probs → topk_idx расходятся → bincount shape mismatch → CheckpointError

**Auxiliaries (training only):**
- `moe_lb = E * sum(f * p_mean)` — Switch CV^2 load balance (1.0 = perfect uniform)
- `route_entropy = -mean(sum(probs * log(probs)))` — максимизируется (вычитается из loss)
- `water_filling_loss = log(E) - H(p_alloc)` — allocator entropy-gap (добавляется к loss)

---

### 1.9 Water-Filling Allocator — `model/water_filling.py`

**WaterFillingAllocator(total_width, num_experts, min_width, temperature=1.0, snr_weight=1.0):**

```
allocation_logits:  Parameter(E)  — FP (AdamW), init = zeros
snr_ema:            Buffer(E)      — FP, detached, init = ones
```

**Теорема Шеннона:** `P_i = max(0, mu - 1/SNR_i)` при `sum(P_i) = P_total`

**Реализация:**
- `update_snr_ema(per_expert_residual_var, decay=0.99)` — вызывается из MoE после backward
- `allocation_probs()` = `softmax((logits + snr_weight * log(snr_ema / mean(snr_ema))) / temperature)`
- `get_widths()` — дискретные ширины `max(min_width, round(total_width * p_i))`
- `regularization_loss()` = `log(E) - H(p_alloc)` — энтропийный регуляризатор (двойственная задача water-filling: максимизация энтропии на симплексе = распределение ёмкости по каналам)

---

### 1.10 HEP Predictive Refinement — `model/refinement.py`

**PredictiveRefiner(H, cfg, norm_eps, use_ternary):**

```
norm:       RMSNorm(H)                           [H]     — FP
up_proj:    BitLinear(H, update_hidden)           — тернарный
gate_proj:  BitLinear(H, update_hidden)           — тернарный
update_out: BitLinear(update_hidden, H)            — тернарный (small-init 0.02)
hep:        BitLinear(H, H) or None               — тернарный (zero-init)
```

**Forward (на клоне h_ctx):**
```python
z = h_ctx.clone()
for _ in range(iterations):
    h_n = norm(z)
    gate = sigmoid(gate_proj(h_n))
    update = update_out(silu(up_proj(h_n)) * gate)
    if hep is not None:
        update = update + hep(update)    # HEP feedback
    z = z + update                       # extrinsic-only: только инновация
ext_before = (z - h_ctx).detach()       # для novelty
ext_after = ...
return z
```

**Novelty:** `norm(ext_after) / (norm(ext_before) + eps)` — <0.05 значит "сошлось"

**RefinementHead(H, V, cfg, norm_eps, use_ternary):**
```
refiner:  PredictiveRefiner(H, ...)
norm:     RMSNorm(H)
to_logits: Linear(H, V, bias=False)      [V, H] — FP (small-init)
```

**Auxiliary loss (off-path):**
```python
ref_logits = to_logits(norm(h_refined[idx]))
ref_loss = CE(ref_logits, targets) + 0.1 * KL(ref_logits || main_logits.detach())
```
- KL чанкуется по rows (1024) — иначе два fp32 `[N, V]` log-softmax при V=262144 и длинном контексте ~17 GB

**EXIT-halt:** замораживает distortion beta-anneal при `novelty < threshold` (5 окон подряд).

---

### 1.11 Multimodal Bridge — `model/multimodal.py`

**MultimodalFusion(cfg, text_encoder=None):**

```
# Per-modality encoders
text_embed:      ConvEmbedding (shared)           — FP
image_embed:     Linear(3*P^2, H, bias=False)     [H, 3*P^2] — FP
audio_embed:     Linear(n_mels, H, bias=False)     [H, n_mels] — FP
image_unc:       _InvVarHead(H)                    — FP
audio_unc:       _InvVarHead(H)                    — FP

# Shared/specific split
shared_down:     Linear(H, H//4, bias=False)       [H//4, H] — FP
shared_up:       Linear(H//4, H, bias=False)       [H, H//4] — FP (zero-init)

# Modality type embeddings
modality_embeds: Parameter(3, H)                   — FP
bridge_queries:  Parameter(n_bridge, H)            — FP (init N(0, 0.02))

# Q-Former layers
image_qformer:   bridge_layers x QFormerLayer(H, bridge_n_heads)
audio_qformer:   bridge_layers x QFormerLayer(H, bridge_n_heads)

# Grounded infomax
grounded: GroundedInfomax(H, grounded_cfg, 3)
```

**Схема кодирования изображения:**
1. Patches: `images [B,C,h,w] -> unfold -> [B, n_h*n_w, C*P*P]`
2. Linear: `image_embed(patches)`
3. 2D-RoPE: (row, col) поворот — head_dim/2 на координату
4. Modality embedding: `h + modality_embeds[1]`
5. Fusion: `z_shared = shared_up(shared_down(h))`, `z_specific = h - z_shared`, `gate = sigmoid(-log_var_head(h))`, `h_fused = z_shared + gate * z_specific`

**Схема кодирования аудио:**
1. Mel frames: `spectrograms [B, n_mels, T_frames] -> transpose -> [B, T_frames, n_mels]`
2. Linear: `audio_embed(frames)`
3. 1D-RoPE: frame index rotation
4. Modality embedding + fusion (аналогично)

**Q-Former bridge:**
- `QFormerLayer`: self-attention (queries) + cross-attention (queries -> modality tokens) + FFN (GELU, 4x expansion)
- `bridge_queries [n_bridge, H]` — learnable queries, сжимают произвольное число токенов модальности до фиксированного префикса
- O(1) мультимодальных токенов независимо от размера изображения/аудио

**Forward:** `[image_prefix, audio_prefix, h_text]` → prefix-LM attention (prefix bidir, text causal)

---

### 1.12 Grounded Infomax — `model/grounded.py`

**GroundedInfomax(H, cfg, num_modalities=3):**

```
projectors: num_modalities x Linear(H, H, bias=False) [H, H] — FP
```

**VICReg (на каждую модальность отдельно):**
```python
z = projector(mean_pool(h[modality_mask]))
z = z - mean(z)                         # center
var_loss = relu(gamma - std(z)).mean()   # per-dim std >= gamma
cov = (z^T @ z) / (n-1)                 # covariance
cov_loss = sum(cov_off_diag^2) / d      # off-diagonal -> 0
```

**InfoNCE (между парами модальностей):**
```python
za, zb = normalize(project_a(pool_a)), normalize(project_b(pool_b))
logits = za @ zb^T / temperature
loss = 0.5 * (CE(logits, arange) + CE(logits^T, arange))
```

**Важно:** градиент течёт в тело (h_ctx без detach) — мультимодальное выравнивание формирует канал.

---

### 1.13 Вспомогательные модули

| Модуль | Файл | Параметры | Назначение |
|---|---|---|---|
| **RMSNorm** | `norms.py` | `[dim]` — weight | Pre-norm, fp32 variance на CUDA |
| **RotaryEmbedding** | `rope.py` | `[head_dim/2]` — inv_freq buffer | 1D-RoPE cos/sin по позициям |
| **rope_cos_sin_2d** | `rope.py` | 0 (функция) | 2D-RoPE (row + col bands) |
| **KVCache** | `kv_cache.py` | 0 (preallocated tensors) | `[B, n_kv, max_seq_len, hd]` |
| **EXITChartHalt** | `exit_chart.py` | 0 (state machine) | Threshold-based convergence halt |
| **AuxLosses** | `outputs.py` | 0 (dataclass) | Типизированные auxiliary loss terms |
| **ModelOutput** | `outputs.py` | 0 (dataclass) | logits, hidden, aux, ce_loss, indices |

---

## 2. Производственный план (Production/Factory Plan)

### 2.1 Данные (`data/`)

**Формат:** .bin файлы с uint16/uint32 токенами (плоский массив, без заголовков).

**MemmapDataset:**
- `MemmapDataset(path, seq_len, vocab_size, eos_token_id, pad_token_id)`
- `np.memmap` для zero-copy чтения
- EOS-delimited записи: каждая запись кончается ровно одним EOS
- Длинные записи разбиваются на `seq_len-1` чанков
- `__getitem__`: нормализация — content + EOS + pad до seq_len

**MixedDataset:**
- Пропорциональное смешивание из `mix.json`
- `RandomSubsetDataset` с фиксированным числом сэмплов
- Shuffle внутри батча

**SequentialCyclingIterator:**
- Циклический проход по датасетам в порядке curriculum
- N циклов на датасет (`cycles_per_dataset=3`)
- `RandomSubsetDataset(ds, samples_per_cycle)` — каждый цикл = свежая случайная выборка
- Checkpointable: `state_dict()` / `load_state_dict()`

**CurriculumBatchProvider:**
- Двухстадийный curriculum
- Stage 1 (0 .. stage2_start): все 9 датасетов в порядке от простого к сложному
- Stage 2 (stage2_start .. max_steps): только openwebmath + edu + slimpajama
- `set_optimizer_step(step)` — переключение по шагу

**Tokenizer (`data/tokenizer.py`):**
- `load_tokenizer(name, local_files_only=True)`:
  1. Попытка gigatoken (1000x быстрее Python HF)
  2. Попытка transformers.AutoTokenizer
  3. Fallback: `tokenizers.RustTokenizer` напрямую (без torch-зависимости — нужно на ROCm Windows где `torch._C._distributed_c10d` отсутствует)

---

### 2.2 Тренировочный цикл (`train/loop.py`)

**`configure_runtime()`:**
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — AMD flash attention
- `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6` — без expandable_segments (VA-space exhaustion на APU)
- TF32 + cudnn benchmark + float32 matmul precision high

**`train(model, dataloader, cfg, teacher, start_step, optimizer_state)`:**
```python
configure_runtime()
cast_to_bf16(model); model.ensure_fp32_params()
optimizer = build_optimizer(model, cfg)
loss_aggregator = LossAggregator(cfg, exit_halt=EXITChartHalt(...))

while step < max_steps:
    dataloader.set_optimizer_step(step)
    microbatches = [next(data_iter) for _ in range(grad_accum_steps)]
    set_lr(optimizer, step, cfg)
    metrics = train_step(model, microbatches, optimizer, cfg, step, ...)
    yield metrics
    if update_applied:
        step += 1
        if step % checkpoint_interval == 0:
            save_checkpoint(...)
```

**`train_step(model, microbatches, optimizer, cfg, step)`:**
```python
for micro in microbatches:
    output = model(input_ids, targets=None, ...)
    output.ce_loss = _causal_next_token_loss(output, input_ids, targets, valid_target_mask)
    loss = loss_aggregator(output, step=step)
    loss_aggregator.update_exit_novelty(output.aux.exit_novelty)
    (loss / accum).backward()

# После backward:
model.commit_moe_ema_updates()
gradient_stats(model, cfg.train)  # norm + clip
torch.cuda.empty_cache()  # каждые cache_release_interval шагов
optimizer.step()
```

**LR schedule:**
- Линейный warmup (0 -> base_lr за warmup_steps)
- Стабильное плато (base_lr до stable_fraction * max_steps)
- Косинусный спад до min_lr_ratio * base_lr
- Один schedule для AdamW и Muon (разные base_lr)

**Метрики на шаг:** `loss, masked_ce, bpt, top2_mass, posterior_entropy, rate, rate_bits, distortion, vicreg, infonce, moe_lb, route_entropy, water_filling, refinement, attn_entropy, avg_confidence, grad_norm, grad_rms, lr, exit_halted, update_applied, all_finite`

---

### 2.3 Оптимизатор (`train/optim.py`)

**TYPE-based routing:** `BitLinear.weight` -> Muon, всё остальное -> AdamW.

**Muon (Newton-Schulz orthogonalization):**
```python
# Momentum update
buf = momentum * buf + grad
update = grad + momentum * buf  (Nesterov) or buf

# Newton-Schulz 5th-order iteration
x = update / (norm(update) + eps)
for _ in range(5):
    A = x @ x^T
    # addmm fusion: 5 kernels вместо 8
    B = beta*A + alpha*(A@A)  # torch.addmm
    x = a*x + B@x             # torch.addmm

# Scale-aware weight decay
fan_ratio = min(max(1, sqrt(fan_out/fan_in)), wd_cap)
p *= (1 - lr * wd * fan_ratio)
p += -lr * fan_ratio * NS5(update)
```

**Ключевые свойства Muon:**
- `momentum_offload=True`: momentum buffer на CPU (экономит ~16 GB VRAM на 8B). Недостаток: 2 per-param host-sync на шаг (g.detach().to("cpu") + update.to(p.device)) — на launch-bound GPU съедает 22% CPU. Для малых моделей (`momentum_offload=False`) — momentum на GPU.
- `wd_cap=2.0`: ограничивает scale-aware WD — root fix для train_v1_1 divergence на step ~10000

**AdamW:**
- Fused где доступен (проверка probe-параметром)
- Две группы: decay (2D, не-norm) и no_decay (1D, norm, bias)
- lr=0.0003 (scaled down для 8B: 0.00015)

**CombinedOptimizer:**
- `step()`: Muon.step() затем AdamW.step()
- `state_dict()`: `{"muon": ..., "adamw": ...}` — сохраняется в чекпоинт для resume

---

### 2.4 Loss Aggregator (`train/losses.py`)

```python
total = CE
      + w_rate * rate
      + w_distortion * beta * distortion    # beta = min(1, step/warmup), frozen by EXIT-halt
      + w_vicreg * vicreg
      + w_infonce * infonce
      - w_route_entropy * route_entropy     # МАКСИМИЗАЦИЯ энтропии
      + w_water_filling * water_filling     # МИНИМИЗАЦИЯ entropy-gap
      + w_moe_lb * moe_lb
      + w_refine * refinement
      + w_attn_entropy * attn_entropy
```

**EXIT-halt:** замораживает `beta` при `halted=True`. Beta больше не растёт — distortion больше не annealed.

---

### 2.5 Чекпоинты (`train/checkpoint.py`)

**Формат V7** (incompatible с V27). Поля: `format_version, model, config, completed_updates, optimizer`.

**Сохранение:**
- tempfile + `torch.save()` + `os.replace()` — атомарно на Windows и POSIX
- Ротация: `keep_last=3` — удаляет старые `step-*.pt`

**Загрузка:**
- `load_checkpoint_payload(path)` — десериализует и валидирует (weights_only=True)
- `load_model_checkpoint(path, model)` — strict `load_state_dict()`
- `latest_checkpoint(dir)` — максимальный номер step-*.pt
- `cfg_from_dict(data)` — реконструирует Config из dict (рекурсивно)

---

### 2.6 Инференс (`inference/generate.py`)

**`generate(model, prompt_ids, **kwargs)`:**
1. Prefetch: `model(prompt_ids, attention_mode="causal")` — KV-кешируются все позиции промпта
2. `for step in range(max_new_tokens):`
   - `process_generation_logits()` — forbidden tokens, min_new_tokens, repetition_penalty, no_repeat_ngram, temperature, top_k
   - `next_tokens = multinomial/top_k/greedy`
   - `sequence[:, prompt_len + step] = next_tokens`
   - Если finished — pad_token_id в logits (чтобы constraints не упали)
   - `model(new_token, positions=[pos])` — single-position forward, KV-cache.append
3. `model.reset_cache()` в `finally`

**`process_generation_logits()`:**
- Forbidden tokens -> -inf
- min_new_tokens: EOS -> -inf пока не набрали минимум
- repetition_penalty: для повторяющихся токенов (в окне window): score<0 -> *penalty, score>0 -> /penalty
- no_repeat_ngram: запрет n-граммов
- Temperature scaling + top_k filtering
- multinomial (temperature>0) или argmax (temperature=0)

---

### 2.7 Distillation (`train/distillation.py`)

**Embedding transfer (`transfer_embeddings`):**
- Загрузка учительских эмбеддингов
- SVD проекция если H_teacher != H_student
- Для ConvEmbedding: truncated SVD факторизация в `token_compress`/`token_expand`
- Random projection fallback

**`DistillationTeacher`:**
- Загрузка учителя (AutoModelForCausalLM, bf16, eval mode, frozen)
- Gemma 4: `_text_only_base()` — спуск к `language_model`
- `get_hidden(input_ids, visibility_mask)` — скрытые состояния учителя
- `distillation_loss_chunked()`:
  - MSE между скрытыми состояниями (не KL на logits — teacher next-token vs student same-position конфликт)
  - alpha schedule: `alpha_start -> alpha_end` линейно, затем 1.0 (чистый CE)
  - `alpha <= 0` = CE only (без MSE)

**`_resolve_hf_cache_path()`:** разрешает HF repo-id в локальный путь (`refs/main` -> `snapshots/<rev>`)

---

### 2.8 ROCm/Windows совместимость (`train/_rocm_fsdp_stub.py`)

Проблема: ROCm Windows torch не имеет `torch._C._distributed_c10d`. transformers 5.x (Gemma-4) eager-imports FSDP.

Решение: pre-seed `sys.modules` с no-op стабами для `transformers.distributed.fsdp`, `transformers.distributed.sharding_utils`, и `DTensor` в `transformers.core_model_loading`. Все публичные имена возвращают inert no-ops.

---

### 2.9 Скрипты

**`scripts/train.py`:**
- CLI: `--config`, `--data-dir`, `--dry-run`, `--steps`, `--device`, `--no-distill`, `--no-embed-transfer`, `--checkpoint-dir`, `--log-dir`, `--resume [latest|path]`, `--profile N`
- Dry-run: один forward + loss aggregation, вывод loss + VRAM
- Profile: torch.profiler на первые N шагов (skip шаг 0 — AOTriton autotune warmup), chrome trace + operator table ranked by self CUDA time

**`scripts/infer.py`:**
- CLI: `--checkpoint`, `--prompt`, `--interactive`, `--device`, `--max-tokens`, `--temperature`, `--top-k`, `--tokenizer`
- Интерактивный режим: REPL с циклом input/generate/print
- Проверка eos_token_id/pad_token_id соответствия чекпоинту и токенизатору

---

## 3. Научный план (Scientific Plan)

### 3.1 Информационно-теоретическая модель: LM как RD-канал

**Фундаментальная метафора:** языковая модель = система кодирования для канала с шумом.

```
Дискретный источник (токены)
    |
    v
Source Encoder (ConvEmbedding) — сжатие V -> H, pulse-shaping фильтр
    |
    v
Канал (Transformer body) — тернарное квантование весов = шум канала
    |
    v
Channel Decoder (LM head) — H -> V, восстановление дискретных символов
```

**Терминология:**

| Концепт теории информации | HAGI-компонент |
|---|---|
| Источник дискретных символов | Токены (V=49154 / V=262144) |
| Source encoder | `ConvEmbedding`: Embedding(V,r) -> Linear(r,H) -> Causal Conv1d |
| Сигнал в канале | Скрытое состояние `h [B, T, H]` |
| Шум канала | Тернарное квантование весов (BitNet b1.58, 1.585 бит/вес, STE) |
| Пропускная способность канала | Определяется H и SNR (отношение сигнал/шум квантования) |
| Rate (скорость кодирования) | `KL[q(z\|h) \|\| N(0,I)]` — Information Bottleneck |
| Distortion (искажение) | `\|\|h - h_hat\|\|^2 / \|\|h\|\|^2` — нормализованная MSE |
| Rate-distortion trade-off | `w_rate * rate + w_distortion * beta * distortion` |
| Water-filling power allocation | MoE SNR-gated routing: capacity -> high-SNR positions |
| Распределённое кодирование источника (Slepian-Wolf) | GroundedInfomax: InfoNCE максимизирует I(text; image) |
| Свёрточный код | Sliding-window attention: конечная память O(T*W), parity relay через full-attention слои |
| EXIT chart (сходимость итеративного декодера) | EXITChartHalt: `norm(ext_after) / norm(ext_before) < 0.05` -> freeze beta |
| Pulse-shaping filter | Causal depthwise Conv1d (межсимвольная интерференция) |
| Состояние свёрточного декодера | KV-cache (замороженные ключи/значения префикса) |

---

### 3.2 Information Bottleneck: rate-distortion сжатия представления

**Теорема IB (Tishby et al.):** минимизация `I(X; Z) - beta * I(Z; Y)`.

**HAGI IB:**
- Стохастический энкодер: `q(z|h) = N(mu, exp(logvar))`
- Rate: `KL[q(z|h) || N(0,I)]` — сколько информации z содержит о h (в nats)
- Distortion: MSE между h и h_hat (восстановление из z)
- Per-dim `free_bits` = 0.01: пол для KL на размерность (предотвращает коллапс)
- **Off-path:** `h_ctx.detach()` — градиент только через rate+distortion, не через main LM CE

**Beta-anneal + EXIT-halt:**
- `beta = min(1, step / warmup_steps)` — плавное включение distortion
- При `novelty < threshold` (5 окон подряд) — beta замораживается
- Мотивация: на ранних шагах LM-сигнал должен доминировать; distortion включается когда представление начинает стабилизироваться; когда сходимость достигнута — дальнейший anneal бесполезен

---

### 3.3 Water-filling: оптимальное распределение ёмкости по каналам

**Теорема Шеннона (water-filling):**
```
P_i = max(0, mu - 1/SNR_i)    при  sum(P_i) = P_total
```
Ёмкость (power) распределяется пропорционально качеству канала (SNR).

**HAGI реализация на двух уровнях:**

**Уровень 1 — MoE routing (water-filling по позициям):**
- SNR proxy: `s = 1/||residual||_2` (high residual = низкий SNR = меньше capacity)
- Router: `logits = router(h) * (1 + w_snr * s)`
- Shared expert = baseline capacity; routed experts = variable capacity по позициям
- Двойственная задача: максимизация routing entropy = распределение capacity по всем expert-каналам

**Уровень 2 — Allocator (water-filling по экспертам):**
- Per-expert SNR EMA на основе residual variance токенов, диспатченных к этому эксперту
- Allocation probs: `softmax(logits + snr_weight * log(snr_ema / mean(snr_ema)))`
- Per-expert intermediate width (capacity): `max(min_width, round(total_width * p_i))`
- Двойственная задача: `log(E) - H(p_alloc)` -> 0 (равномерное распределение)

---

### 3.4 Distributed Source Coding: Slepian-Wolf через InfoNCE

**Теорема Slepian-Wolf:** два коррелированных источника могут быть закодированы раздельно и декодированы совместно без потерь, если суммарная rate >= joint entropy.

**HAGI InfoNCE:**
- `logits = za @ zb^T / tau` — similarity matrix
- `loss = 0.5 * (CE(logits, arange) + CE(logits^T, arange))` — symmetric
- Нижняя граница на `I(text; image)`: `log(N) - InfoNCE <= I(X; Y)` (Oord et al., CPC)
- Модальности кодируются раздельно (separate source encoders), выравниваются в joint space

---

### 3.5 Convergence (EXIT chart)

**Теория EXIT (EXtrinsic Information Transfer):** итеративный декодер сходится когда взаимная информация между итерациями перестаёт расти. EXIT chart визуализирует `I_A -> I_E` характеристики компонентных декодеров.

**HAGI EXITChartHalt:**
- `novelty = norm(ext_after) / norm(ext_before)` — proxy для `I_E / I_A`
- `novelty < 0.05` = декодер перестал извлекать новую extrinsic информацию
- Hysteresis: window=5 последовательных sub-threshold наблюдений
- `min_steps=50`: никогда не срабатывает до 50 наблюдений
- Применение: останов beta-anneal distortion (не refinement — refinement только diagnostic)

---

### 3.6 BIT-философия HAGI

**HAGI BIT (Behavioural-Intentional-Theoretical) Philosophy:**

1. **B — Behavioural (экспериментальная валидация):** Каждый компонент проверяется на from-scratch обучении. V25 IB in-path deadlock'нул CE — убран off-path. V25 refinement in-path deadlock'нул CE — реабилитирован off-path.

2. **I — Intentional (намерение дизайна):** Каждый компонент имеет явную информационно-теоретическую мотивацию. Нет компонентов "потому что так в Llama". Нет blind ablation stacking.

3. **T — Theoretical (математическое обоснование):** Rate-distortion теория, water-filling theorem, Slepian-Wolf, EXIT charts — не просто аналогии, а прямые источники формул и архитектурных решений.

**Применение BIT к архитектурным решениям:**

| Решение | BIT-обоснование |
|---|---|
| ConvEmbedding: factorized Vxr + rxH | Low-rank source coding: естественный язык имеет низкий эффективный ранг эмбеддингов |
| Ternary weights (BitNet b1.58) | Фиксированная rate хранения весов (1.585 бит/вес). Квантование = реалистичный канальный шум. Не нужно self-inflicted AWGN |
| Causal Conv1d (left-pad) | Pulse-shaping без future leak. Свёрточный код имеет конечную память |
| RoPE (1D + 2D) | Непрерывный поворот = непрерывное относительное позиционное кодирование (масштабируется) |
| GQA (Grouped Query Attention) | Разделение ключей/значений между query-головами = экономное использование пропускной способности |
| Sliding window + full_every | Конечная память O(T*W) + глобальное parity relay. Глаза в literature: Mistral, Gemma 2 |
| Water-filling MoE | SNR-gated capacity allocation = оптимальное распределение ёмкости |
| Q-Former bridge | O(1) multimodal tokens = сжатие prefix'а модальности (независимо от размера изображения) |
| GroundedInfomax (VICReg+InfoNCE) | VICReg = стабилизация joint embedding (аналог Barlow Twins). InfoNCE = lower bound на cross-modal MI |
| PredictiveRefiner (HEP, off-path) | Экстравертная итеративная коррекция ошибок. EXIT-chart критерий сходимости |
| Muon (Newton-Schulz) | Ортогонализация градиента = естественное градиентное обновление для 2D весов. Не нужен momentum в NS5 |
| Scale-aware WD (Muon) | `wd * min(sqrt(fan_out/fan_in), cap)` — bounding ||W|| — root fix для divergence |

---

### 3.7 Почему off-path, а не in-path

**Исторический урок V25:**
- IB in-path: CE застрял на ~ln(V) с шага 0 — стохастический bottleneck разрушал LM-сигнал
- PredictiveDecoder in-path (HEP refinement): тот же deadlock — refinement не мог улучшить ещё не сформированное представление

**V28 решение:**
- Все auxiliaries на `h_ctx.detach()`
- Каждый auxiliary имеет СВОЙ loss-терм и СВОЙ градиент
- Main LM CE течёт через h_ctx без помех
- Исключение: GroundedInfomax — градиент в тело осознанно (формирует multimodal представление канала)

---

## 4. Экспериментальный план (Experimental Plan)

### 4.1 Ablation Studies (Стандартная размерная сетка: H=384, C=192, L=12)

| Эксперимент | Что отключаем | Гипотеза | Метрики |
|---|---|---|---|
| **E0: Baseline** | Ничего (smollm2.yaml) | — | CE, bpt, rate, distortion |
| **E1: No IB** | `w_rate=0, w_distortion=0` | IB полезен как регуляризатор; без него CE должно быть чуть хуже | CE delta, posterior entropy |
| **E2: No ternarization** | `use_ternary=False` (все веса FP) | Тернарность даёт регуляризацию; FP должно дать чуть лучший CE ценой 21x памяти весов | CE delta, memory |
| **E3: No causal conv** | `kernel_size=1` (нет pulse-shaping) | Causal conv полезен для локального контекста | CE delta, особенно на именованных сущностях |
| **E4: Full MHA** | `n_kv = n_q` (без GQA) | GQA экономит параметры без потери качества | CE delta, param count |
| **E5: No Hebbian FFN** | Обычный SwiGLU (не билинейный) | Билинейная карта даёт лучшую ассоциативную память | CE delta |
| **E6: No attention entropy** | `w_attn_entropy=0` | Anti-collapse пенальти предотвращает attention collapse | Attention entropy distribution |
| **E7: Fixed distortion beta** | `beta=1.0` с шага 0 | Anneal важен: distortion на старте мешает CE | CE trajectory early steps |

### 4.2 Scaling Studies

| Эксперимент | Параметры | Бюджет (non-embed) | Ожидаемые изменения |
|---|---|---|---|
| **S1: 15M** | auto_configure(15e6) | ~15M | Базовый scaling baseline |
| **S2: 40M** | smollm2.yaml | ~27M | Текущий baseline |
| **S3: 150M** | auto_configure(150e6) + MoE | ~120M | MoE вступает в игру |
| **S4: 1B** | auto_configure(1e9) + MoE | ~800M | Проверка scaling laws |
| **S5: 8B** | google.yaml | ~8B | Полный масштаб, всё включено |

### 4.3 MoE Ablations (на 150M+ моделях)

| Эксперимент | Конфигурация MoE | Гипотеза |
|---|---|---|
| **M1: No MoE** | moe.enabled=false | Dense baseline |
| **M2: Uniform MoE** | Без water-filling (snr_gate_weight=0, allocator off) | Water-filling улучшает специализацию экспертов |
| **M3: Water-filling routing only** | snr_gate_weight=1.0, allocator off | SNR-gated routing без per-expert capacity allocation |
| **M4: Full water-filling** | snr_gate_weight=1.0, allocator on | Полный water-filling на двух уровнях |
| **M5: Vary top_k** | top_k in {1, 2, 4} | Оптимальное число экспертов на токен |
| **M6: Vary n_shared** | n_shared in {0, 1, 2} | Shared expert необходим? |
| **M7: Vary moe_every** | moe_every in {1, 2, 4} | Частота MoE-слоёв |

### 4.4 Refinement Ablations

| Эксперимент | Конфигурация | Гипотеза |
|---|---|---|
| **R1: No refinement** | refinement.enabled=false | Baseline |
| **R2: Refinement, no HEP** | hep_enabled=false | HEP необходим для depth-independent коррекции? |
| **R3: Refinement + HEP** | Полный refinement | Проверить улучшение log-perplexity |
| **R4: Vary iterations** | iterations in {1, 2, 4, 8} | Оптимальное число проходов |
| **R5: EXIT-halt** | exit_threshold in {0.01, 0.05, 0.1} | Оптимальный порог сходимости |

### 4.5 Multimodal Scaling (если данные доступны)

| Эксперимент | Конфигурация | Метрики |
|---|---|---|
| **MM1: Text only** | multimodal.enabled=false | CE text |
| **MM2: Image+text** | Только image bridge | CE text + VICReg/InfoNCE |
| **MM3: Audio+text** | Только audio bridge | CE text + VICReg/InfoNCE |
| **MM4: All modalities** | Всё включено | Cross-modal retrieval, zero-shot |

### 4.6 Optimization Ablations

| Эксперимент | Конфигурация | Гипотеза |
|---|---|---|
| **O1: AdamW only** | Все параметры -> AdamW (без Muon) | Muon лучше для тернарных 2D весов |
| **O2: Muon, no NS** | ns_steps=0 (просто momentum) | NS ортогонализация необходима |
| **O3: Muon, no Nesterov** | nesterov=false | Nesterov ускоряет сходимость |
| **O4: Muon, NS steps** | ns_steps in {3, 5, 10} | Оптимальное число NS итераций |
| **O5: WD cap** | wd_cap in {1.0, 2.0, 4.0, inf} | Оптимальное ограничение scale-aware WD |

### 4.7 Curriculum Studies

| Эксперимент | Конфигурация | Гипотеза |
|---|---|---|
| **C1: No curriculum** | Все датасеты в random mix | Curriculum помогает from-scratch |
| **C2: Vary stage2_start** | stage2_start in {50K, 100K, 120K} | Оптимальный момент перехода |
| **C3: Vary cycles_per_dataset** | cycles in {1, 3, 5} | Оптимальное число циклов |

### 4.8 Key Metrics to Track (единый приборный щит)

| Метрика | Что измеряет | Target |
|---|---|---|
| `masked_ce` | Next-token cross-entropy | Минимизация |
| `bpt` | Bits per token = CE / log(2) | ~4-6 для хорошей модели |
| `rate` | IB KL rate (nats/token) | Стабильный, не растёт |
| `rate_bits` | rate / log(2) | ~1-3 bits |
| `distortion` | IB reconstruction error | Убывает с anneal |
| `posterior_entropy` | Энтропия предсказаний | ~3-6 (не 0 = не коллапс) |
| `top2_mass` | Масса top-2 токенов | ~0.5-0.8 |
| `avg_confidence` | Средняя уверенность | ~0.1-0.3 |
| `moe_lb` | Load balance (1.0 = идеал) | ~0.8-1.2 |
| `route_entropy` | Routing entropy | ~log(E) для максимума |
| `attn_entropy` | Attention entropy penalty | ~0 (пол > H(attn) — не коллапсит) |
| `grad_norm` | Норма градиента | < max_grad_norm |
| `grad_rms` | RMS градиента | Стабильный, не NaN |
| `exit_halted` | EXIT-halt сработал | После 50+ шагов с novelty<0.05 |

---

## 5. Файловая карта проекта

```
C:\HAGI_v2\
  configs\
    smollm2.yaml          — RTX 3070 8GB, text-only, ~39.5M
    google.yaml            — Strix Halo 96GB, ~8.55B, всё включено
    google_small.yaml      — Strix Halo small, H=512, MoE, V30 auto-calculate
  data\                    — .bin файлы + mix.json
  scripts\
    train.py               — точка входа тренировки
    infer.py               — точка входа инференса
    download_model.py      — загрузка моделей
    preprocess_gemma.py    — препроцессинг данных
  src\hagi\
    __init__.py
    version.py             — __version__ = "2.1.0"
    config.py              — ВСЕ конфиги (646 строк, 18 dataclass'ов)
    data\
      __init__.py
      dataset.py           — MemmapDataset, MixedDataset, build_dataloader
      sequential.py        — SequentialCyclingIterator, CurriculumBatchProvider
      tokenizer.py         — gigatoken/HF/rust tokenizer loader
    model\
      __init__.py
      model.py             — HAGI(nn.Module): главный forward
      block.py             — TransformerBlock: pre-norm GQA + FFN/MoE
      attention.py         — Attention: GQA, 4 режима, sliding window, KV-cache
      hebbian_ffn.py       — HebbianBilinearFFN: билинейная SwiGLU
      ternary.py           — BitLinear, ternarize, _TernarizeSTE
      conv_embedding.py     — ConvEmbedding: factorized source encoder
      bottleneck.py         — InformationBottleneck: H->C->H off-path
      moe.py               — WaterFillingMoE: SNR-gated routing, batched dispatch
      water_filling.py     — WaterFillingAllocator: per-expert capacity
      refinement.py         — PredictiveRefiner + RefinementHead: off-path HEP
      multimodal.py         — MultimodalFusion: Q-Former bridge + 2D/1D-RoPE
      grounded.py           — GroundedInfomax: VICReg + InfoNCE
      norms.py              — RMSNorm (fp32 variance on CUDA)
      rope.py               — RoPE: 1D, 2D, RotaryEmbedding
      kv_cache.py           — KVCache: ring-free preallocated KV store
      exit_chart.py         — EXITChartHalt: convergence criterion
      outputs.py            — AuxLosses, ModelOutput (dataclass)
    train\
      __init__.py
      loop.py               — train(), train_step(), LR schedule, configure_runtime
      optim.py              — Muon, CombinedOptimizer, build_optimizer
      losses.py             — LossAggregator: CE + все auxiliaries
      checkpoint.py         — save/load, format V7, strict loading
      distillation.py       — transfer_embeddings, DistillationTeacher
      _rocm_fsdp_stub.py    — FSDP заглушка для ROCm Windows
    inference\
      __init__.py
      generate.py           — generate(), process_generation_logits()
```

---

## 6. Версионная история V27 -> V28 -> V30

| Версия | Дата | Ключевые изменения |
|---|---|---|
| V27 | 2026-07 | Единый KV-cacheable стек, GQA, MoE (python loop), IB off-path, ckpt fmt 6 |
| V28 | 2026-07-28 | Water-filling MoE (SNR gate, batched dispatch), sliding window, 2D/1D-RoPE, HEP refinement off-path, EXIT-halt, _TernarizeSTE, ensure_fp32, addmm fusion, ckpt fmt 7 |
| V30 | 2026-07-29 | auto_configure из target_params, устранение магических чисел, google_small.yaml |

Текущие коммиты в main:
- `1004c37` perf: ROCm flash attention, empty_cache каждые N шагов
- `4082ee7` feat(v30): auto-calculate params, eliminate magic numbers, fix dead IB rate
- `71ae01e` fix(moe): deferred SNR EMA for checkpoint compat; perf(ns5): addmm fusion
- `9d78a03` perf(ternary): autograd.Function STE, kill 17k detach/step
- `5f17a2b` perf(muon): momentum_offload flag, on-device path
