# HAGI — Архитектура

Единый документ: как устроены модель, код и пайплайн роста поколениями
в текущем состоянии. Версия **V42** (`hagi-channel-v42`, 4.2.0).
Исторические результаты и rationale — в git history и `.omc/attempts/`.

HAGI — **языковая модель, растущая поколениями**. Каждый цикл:
N узких листов-специалистов (общий prior — корень, разные data-seed)
обучаются, сливаются в один корень и коротко дообучаются; новый корень
становится prior'ом следующего поколения. Приёмка поколения — только
через сертифицированный A/B-гейт (§9 deep-200): без сертификата цикл
останавливается, несертифицированного роста нет.

**Рекордная линия (с gen17 — латентная линия R242–R246, fixed-width):**

```
листья H=128 (6 доменных: math, code, ru, enedu, enweb, enwiki)
  — dense тело 3 слоя: pre-norm GQA + QK-norm + RoPE, SwiGLU (exp=1),
    untied голова, fused CE, zero-init proj
  — ПОВЕРХ замороженного корня: TTT-LoRA адаптеры (rank 32; с gen34
    rank 64 — рычаг R242 «рост ранга, не ширины»)
  → latent merge фиксированной ширины (pilot_latent_merge.py:
     факторизация вокруг PRIOR, merge_gate пер-тензорный, волокна
     сохраняются — R242/R243)
  → joint fine-tune корня (root_ft, 600 шагов)
  → WSqD cooldown (R254): batch ×4 / lr 0.4× / 200 шагов — хвост
    затухания стоит константу, полировка почти бесплатна
  → §9 certified accept: кандидат (лучший из ft/cooldown) заменяет
    инкамбента только при выигрыше > margin 0.02
```

Лестница (checkpoints/growth_ladder.jsonl, common-protocol AVG CE):

```
gen-4  5.8126 → gen-11 5.4150 → gen-18 5.2895 (rank-32 + свежие
корпуса) → gen-28 5.1359 → gen-32 4.9147 (лучшая, инкамбент)
gen-33 4.9536 — отклонён гейтом (delta −0.039): пол R255
  (позднее разногласие — почти ортогональный шум, усреднение
  гибнет 1/N, floor r/(1−κ) достигнут; batch-64 листья NaN на iGPU)
gen-34 (запущен): рычаг — adapter rank 32→64 + новый data-seed 161xx
```

Выключено в линии (измерено/доказано избыточным на этом масштабе):
ternary body, cortex, decision, unigram_prior, NCE-приёмник,
conv-фильтр (k=1), head_up, рост батча (B_noise ≈ 100 токенов),
per-matrix NS (фикс 5 шагов достаточен), полный SwiGLU cross-mixer.

Словарь (механизмы, не метафоры): «latent merge» = слияние листей
фиксированной ширины через факторизацию вокруг prior (R242);
«leaf/sib» = эксперт H=128, потомок корня; «cooldown» = WSqD-полировка
R254; «certified accept» = §9 A/B с margin 0.02.

---

## 1. Структура репозитория

```
HAGI_v2/
├── src/hagi/
│   ├── config.py             # dataclass-конфиги + валидация + count_params
│   ├── version.py            # V42, идентичность архитектуры
│   ├── data/
│   │   ├── dataset.py        # packed-corpus поток + смешанные окна
│   │   ├── artifacts.py      # opt-in versioned manifests
│   │   └── vocab_map.py      # маппинг compact↔исходного словаря
│   ├── inference/
│   │   ├── generate.py       # авторегрессионная генерация + KV-cache
│   │   └── hedge_router.py   # §6 Hedge-роутер (SKIP: merged не экспонирует
│   │                         #   per-leaf logits; см. RouterConfig)
│   ├── model/
│   │   ├── model.py          # HAGI (главный класс)
│   │   ├── merge.py          # MergedHAGI, Hadamard/CrossMixer (истор. линия)
│   │   ├── latent_merge.py   # R242/R243 латентное слияние (живая линия)
│   │   ├── adapters.py       # TTT-LoRA адаптеры (ранговый канал роста)
│   │   ├── block.py, attention.py, ffn.py, embedding.py, head.py,
│   │   │   norms.py, rope.py, kv_cache.py, outputs.py
│   │   ├── ternary.py        # BitLinear (b1.58) — доступно, выключено
│   │   ├── table_lora.py     # TableLoRA на embedding/head
│   │   ├── cortex.py         # opt-in cross-level канал (выключен)
│   │   ├── decision.py       # opt-in decision head (research-only)
│   │   ├── multimodal.py     # Q-Former мост (выключен)
│   │   ├── adaptive.py, formal.py, scratch_blocknorm.py, factory.py
│   ├── orchestrator/         # real_cycle, recursive, merge_select,
│   │                         #   evaluation, state (потребители супервизора)
│   └── train/
│       ├── loop.py           # Trainer: train_step, гейты, gram-scan
│       ├── optim.py          # Muon + AdamW (HybridOptimizer), WSD
│       ├── checkpoint.py     # формат 12, атомарная запись, lenient reload
│       ├── certified_controller.py  # §0/§1/§9/§17 решающее ядро
│       ├── safeqp_controller.py / safeqp_step.py  # SafeQP сертификаты
│       ├── distill*.py       # дистилляция (transfer/recursion/recursive)
│       ├── growth_law → см. attic (ниже); takeoff_window, gain_renewal,
│       │                     ratio_takeoff, insight, hedge, bit_alloc,
│       │                     merge_price, saturation, spectral
│       ├── self_improve.py / self_development.py / ttt.py  # opt-in
│       └── _rocm_fsdp_stub.py
├── scripts/
│   ├── train.py, infer.py, eval_domains.py, eval_holdout.py  # ядро
│   ├── growth/               # ЖИВОЙ цикл: growth_cycle.py, run_ladder.py,
│   │                         #   growth_supervisor.py, pilot_latent_merge.py
│   ├── data/, lora/, audits/, kernels/, release/, dsv4/, research/
├── configs/                  # 8 живых YAML: cycle.yaml, dbridge×2,
│                             # latent-шаблоны, тестовая фикстура;
│                             # поколенческие конфиги НЕ живут здесь —
│                             # генерируются в checkpoints/gen<N>/configs/
│                             # из SSOT scripts/growth/gen_configs.py
├── tests/                    # pytest (562)
├── data/                     # .compact.bin корпуса, mix.json, boost_*
├── checkpoints/              # вне git; growth_ladder.jsonl — журнал
├── _raw/attic/               # снятые с рантайма модули + тесты (git-история)
└── .omc/attempts/            # ledgers раундов
```

**Чистка 2026-10-10** (attic, обратимо, история в git): anytime_budget
и stochastic_safeqp (anytime-гейт живёт в
`growth_supervisor.anytime_margin`), data_axis (совет — в супервизоре),
frontier_cone, growth_law, routing_optimal, safeqp_gpm, safeqp_pl,
ternary_exact, trust_region, orchestrator/external_eval,
orchestrator/mechanism_gate — теоретические порты без runtime-потребителя.

---

## 2. Модель

### 2.1 SourceEncoder (`model/embedding.py`)

Полная таблица `[V, H]` (не факторизуется: rank-r даёт информационный
потолок). Transmit filter — каузальный depthwise Conv1d (k=1, выключен).

### 2.2 Блок (`model/block.py`, `ffn.py`, `attention.py`)

`x + attention(x)`, затем `x + ffn(x)`, оба pre-norm.

- **Attention**: GQA, fused QKV, QK-norm (`HeadNorm`), RoPE, sink 4.
- **FFN**: SwiGLU `down(silu(gate(x)) * up(x))`, expansion 1.
- **BranchScale** — обучаемый per-branch масштаб, clamp-8 per-layer.

### 2.3 LMHead (`model/head.py`)

Untied, receiver gain, z-loss, chunked CE (логитсы блоками),
`exact_loss` — точный CE для гейтов.

### 2.4 TTT-LoRA адаптеры (`model/adapters.py`)

Ранговый канал роста живой линии: leaves обучают только адаптеры
(rank r, alpha 16) поверх замороженного тела корня; рост ёмкости
между поколениями = рост r (R242: sub-1-BPW ⟺ рост по латентной
ранге вместо H→3H). С gen34 r=64.

---

## 3. Слияние

### 3.1 Латентное слияние (живая линия) — `model/latent_merge.py`

R242/R243: факторизация вокруг PRIOR (shared-init корня), per-тензорный
merge_gate (экономический гейт ветви: R_expert ≠ R_quant), волокна
сохраняются по-экспертно (fiber-сохранение точное). Пайплайн:
factorize → latent-align (вращения бесплатны, sign-flip ловится —
R242 MergeCancellation) → root → spectral compress. Bregman-slice
селекция токенов (§17, disagreement_distill) — forward-declared.

### 3.2 Троичное слияние 3→1 (историческая линия, dbridge) — `model/merge.py`

Блочно-диагональная конкатенация 3 листьев (шаг-0 = ансамбль,
off-diagonal нули), Hadamard-миксер + rank-64 residual, head
pre-rotation (function-preserving инвариант). Довело линию до
gen-7 H=10368, упёрлось в VRAM → density ladder / латентная линия.
Код живой (тесты, гейты), в цикле роста не используется.

---

## 4. Пайплайн роста (живая латентная линия)

`scripts/growth/growth_cycle.py` — одно поколение за вызов
(idempotent-safe через marker-файлы):

1. **Листья**: 6 доменных листей H=128, `init_from` = инкамбент-корень
   (same-origin §23[1]), разные data-seed, fresh-data offsets
   (`--start-offset` от consumed.json листа — gen21-фикс).
2. **Latent merge** (pilot_latent_merge.py) — теоретически корректный
   пайплайн §3.1; noise_report.json даёт R255-когерентность
   (mean |cos| разногласий — в лестнице: 0.07–0.14).
3. **Root fine-tune** (600 шагов) + **WSqD cooldown** (R254: 200 шагов,
   batch ×4, lr 0.4×).
4. **Certified accept §9**: кандидат = лучший из ft/cooldown; замена
   инкамбента только при delta > 0.02 на common-protocol eval
   (eval_domains.py, AVG exact CE). Отказ → цикл стоит
   (stop_condition, R223: E_dev исчерпан → менять данные/ёмкость).

`scripts/growth/run_ladder.py` — мульти-поколенный драйвер: конфиги
поколения генерируются SSOT-билдером `gen_configs.py` (один шаблон +
пер-доменные overrides; воспроизводит линию бит-в-бит) в
`checkpoints/gen<N>/configs/` — proliferation YAML в configs/
устранён (2026-10-10, 185 файлов удалены); вызов growth_cycle,
останов на первом несертифицированном поколении.

Лестница: gen-4 … gen-32 monotone climb (5.81 → 4.9147), генерации
12–16/21/33 отклонены гейтом (честная остановка). После gen-33
(пол R255) рычаг gen-34: rank 64 + data-seed 161xx.

---

## 5. Данные — `data/dataset.py`

Packed-corpus: документы конкатенируются в плоский поток, окна
фиксированной длины, `doc_ids` → блочно-диагональная маска внимания.
Утилизация 0.41–0.80 против 0.10–0.43 у per-document.

- `PackedStream` — memory-mapped, конечный; следующий лист продолжает
  с `start_offset` (примитив рекурсивного роста, gen21).
- `PackedMixDataset` — бесконечные пропорционально-смешанные окна.
- `dataset_path`: `.compact2.bin` > `.compact.bin` > `.bin`;
  compact-словарь 32768.

Базовые 8 корпусов (mix.json): edu .3571, python_instruct .2232,
wikipedia_en .0893, wikipedia_ru .0714, oscar_ru .0625, openwebmath
.0893, tinystories .0536, smoltalk .0536. Доменные веса листьев —
свои (например math: openwebmath .60 / edu .25 / slimpajama .15).
Свежая инъекция gen18: boost_smoltalk + boost_tinystories 60%.

---

## 6. Обучение — `train/`

### 6.1 `loop.py`

Один forward на microbatch; accumulation нормализует по scored
tokens. Наблюдаемые: `ce`, `qk_gain`, `logit_scale`, `exact_ce`.
Saturation early-stop на exact_ce; divergence-гейт на стабильном
gate_ce (42cca2b); gram-scan (per-corpus градиентный скан,
SafeQP-сертификат, log-only). Sealed-budget fallback: превышение
бюджета микса → рестарт листа с offset 0 (данные исчерпаны).

### 6.2 `optim.py`

Muon (Newton–Schulz, ns=5) для 2D channel-весов + AdamW для
таблиц/гейнов; WSD-schedule. lr-полоса: H·lr 0.35–0.38 (0.0005/128);
lr НЕ масштабно-инвариантен (ген-4 sib3 на 1e-3 разошёлся до 74.77).

### 6.3 `checkpoint.py`

Формат 12, строгая валидация, lenient reload, атомарная запись.

### 6.4 `certified_controller.py`

§0/§1/§9/§17 решающее ядро: certified A/B (принимать ⟺
CE_A−CE_B > 2ε при n ≥ log(2/δ)/2ε²), stop_condition, leak_gate,
exhausted_check, фазы §23. Супервизор решает через certified_ab;
UNDECIDED → not-worse фоллбэк.

---

## 7. Генерация — `inference/generate.py`

Prefill + decode (KV-cache), repetition penalty → temperature →
top-k → top-p. Текстовый мост: compact id → vocab_map → gemma.

---

## 8. Конфигурация — `config.py`

Dataclass-конфиги (`ModelConfig` c `AdaptersConfig`/`TTTLoRAConfig`,
`AttentionConfig`, `MergeConfig`, `TrainConfig`, `DataConfig`, …),
`load_config` (YAML + dotted overrides), `validate_config`
(fail-fast инварианты), `count_params`. Конфиги поколений
генерируются `run_ladder.derive` (init_from → принятый кандидат,
checkpoint_dir → gen N) — вручную не редактировать (геометрия
наследуется, два падения gen-4 были именно на этом).

---

## 9. Тесты — `tests/`

562 теста: merge-инварианты (шаг-0 = ансамбль), hadamard-
ортогональность, latent-merge, certified-step, safeqp-controller,
config/checkpoint/dataset/attention/generate/head/vocab_map,
resume-precedence (4 CLI dry-run теста), супервизорные гварды,
R258 runtime-сертификаты.

---

## 10. Ключевые принципы

1. **При равном бюджете много узких специалистов лучше одного
   широкого**: слияние наследует обученные подпространства без
   потери качества (измерено: ~1 нат против scratch тем же бюджетом).
2. **Рост — рангом, не шириной** (R242): H фиксирован, ёмкость
   растёт через adapter rank; ширина H→3H упёрлась в VRAM (gen-7).
3. **Function-preserving инварианты обязательны**: шаг-0 тождество
   любого слияния проверяется тестом до обучения.
4. **Гиперпараметры вычисляются, а не подбираются**: LoRA-ранг — из
   спектра, BranchScale — из дисперсии активаций, батч — из шума
   градиента (исчезает на ~100 токенах), дорогие эксперименты —
   только после дешёвого сертификата на обученной модели.
5. **Отбор кандидатов — по вкладу в ансамбль** (Fisher/Jensen
   novelty), не по standalone CE.
6. **Некертифицированного роста нет**: §9 гейт — единственный путь
   замены инкамбента; отказ гейта = честная остановка и смена
   рычага (данные/ранг), а не «ещё раз так же».
7. **Порядок механизмов** (от дешёвых к дорогим): специализация →
   merge → связь → joint → cooldown → низкоранговые добавки.
