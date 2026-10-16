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
     факторизация вокруг PRIOR, merge_gate пер-тензорный (нарушение —
     fallback на same-origin mean, НЕ abort поколения: гейт — потензор-
     ная экономика Th.1+GapLaw, эквивалентность живёт в волокнах; Oracle
     2026-10-12, корень gen64–66 abort'ов на lora_B с twoGap≈0.35≤prices),
     волокна
     сохраняются — R242/R243)
  → joint fine-tune корня (root_ft, 600 шагов)
  → WSqD cooldown (R254): batch ×4 / lr 0.4× / 200 шагов — хвост
    затухания стоит константу, полировка почти бесплатна
  → §1 селектор ДЕЙСТВИЙ (2026-10-11): перед каждым поколением
    certified_controller.estimate_actions/select_action — argmax Γ/K
    из измерений прошлой генерации (Γ_merge = logit_twoGap, Γ_grow =
    принятая delta); нет certified-positive действий = E_dev исчерпан
    (R223) — останов со сменой рычага, не «ещё раз так же»
  → blocking twoGap-гейт: точный logit-twoGap ≤ 0 = консенсус
    (GapLaw.twoGap_zero_iff) — раунд исчерпан до ft
  → §9 certified accept: certified_ab, eps = √(log(2/δ)/(2n)) из
    фактического n (δ=0.05); UNDECIDED при delta>0 углубляет eval
    30→150 батчей (deep-200 наследие); кандидат = лучший из
    ft/cooldown/fiber, замена только по ACCEPT
```

Лестница (checkpoints/growth_ladder.jsonl, common-protocol AVG CE):

```
gen-4  5.8126 → gen-11 5.4150 → gen-18 5.2895 (rank-32 + свежие
корпуса) → gen-28 5.1359 → gen-32 4.9147 → gen-37 4.8346 (nested
r64) → gen-39 4.8215 → gen-41 4.8046 → gen-44 4.7727 →
**gen-51 4.6708 (инкамбент, cooldown-рука, +0.0367 при n=19200)**.
gen45–50 смешанные (44→51 — эпоха frontier-инъекции books/sci);
Отклонены гейтом (честные остановки): 12–16, 21, 33, 34–36, 38,
43, 45–50, 52–56 (последние — недообученные 800-шаговые leaves
против ~500M-токенного books-корпуса; с gen58 books/sci leaves
2400 шагов). ВАЖНО (fix 805b228): остановки gen52–56 селектором
("E_dev исчерпан") были АРТЕФАКТОМ — плейсхолдер grow_cost=1.0
нат вычитался из измеренного Γ; с фиксом селектор сертифицирует
grow_leaf при последнем положительном delta.
С gen42 в certified accept входит ТРЕТЬЯ рука — fiber-ансамбль
(R255, root.pt+fibers.pt merge-точки); gen40: fiber +0.174 AVG над
плоским root (все 6 доменов, вкл. оба held-out), с gen42 ft-рука
сильнее (волокна — кандидат, не замена).
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
├── _raw/attic/               # снятые с рантайма модули + тесты (git-история);
│                             # 2026-10-11: + scripts/ — gen7-эпоха (pipeline,
│                             #   gate, fold/lora-rank), заменённые SSOT
│                             #   конфигураторы, level-1 пайплайн, F3-эпоха,
│                             #   oneoff-вердикты (14 файлов)
└── .omc/attempts/            # ledgers раундов
```

**Чистка 2026-10-12** (attic, обратимо, AST-граф импортов от live-корней
scripts/growth+train+eval): bit_alloc, recursive_distill, hedge +
hedge_router (SKIP-цепочка), self_development оставлен (кросс-
потребитель в тестах saturation), scripts/cycle_gate.py — модули без
runtime-потребителя; тести перенесены рядом (test_hedge.py,
собирается из attic через sys.path).

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
3. **Root fine-tune** (600 шагов; R361: с gen61 — 100·N_листьев
   шагов при N=10 → 1000, warmup 1/6 бюджета; рычаг после 9 подряд
   отклонений gen52–60) + **WSqD cooldown** (R254: 200 шагов,
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

## R371 growing-batch расписание (порт 2026-10-13)

`train.batch_growth_delta` (+ обязательный `batch_growth_cap`):
эффективный batch растёт геометрически `accum(t) = base·δ^t`.
Теорема `growing_batch_stabilized` (primes R371): при δ > 1/ρ
(ρ = 1 − ηL) стохастический хвост гаснет на скорости контракции —
полировка хвоста выведена, а не подобрана (обобщение WSqD R254).
`consumed.json` при growth несёт фактический интеграл токенов.
None → фиксированный accum (наследие). Модуль:
`src/hagi/train/growing_batch.py` (паттерн R366 grokking_schedule).

## Порты теории 2026-10-13 (раунд 2): MemoryCycle / Muon / ModeState

- `train/memory_cycle.py` (Unified/MemoryCycle + Growth/MemoryController
  R271–R273): LiveDelta rank-1 память, admission-контроллер
  (novelty ∧ certified gain; шум вдали от ключей не допускается —
  новизна ≠ полезность), консолидация с Коши-бюджетом
  frob(eff−eff') ≤ k·Σᵢ εᵢ и границей качества L·√(k·Σε).
  Standalone; интеграция в ранговый рост адаптеров — отдельный
  раунд (смена живой линии — только через A/B).
- `train/optim.py`: `muon_eta_max_stable` (MuonStableLR: η_max =
  (2/λ)·(Σσ/m), среднее сингулярное ≤ σ_max — консервативен) и
  `min_ns_iterations` (NewtonPolar R380: сертифицированный минимум
  итераций |x−1| ≤ 2((x₀−1)/2)^{2^T}; σ_max ≥ 3 — вне бассейна,
  fail-fast). Обе опциональны (opt-in), дефолтная политика не изменилась.
- `train/gen_telemetry.py` (Generalization/ModeState): Qgen-вектор
  5 проб (h_emp_G), Φ_HAGI с Q-пламенем, mode-drop гейт,
  парето-выбор чекпоинтов (строго положительные веса).
- `train/thermostat.py` (Growth/Thermostat R387–R388 — замыкание
  термодинамического контура): расписание T_t = T₀·α^t,
  законная остановка эксплорации N = ⌈ln(δ/(T₀·ln(1/θ)))/ln α⌉
  (e^{−δ/T_t} ≤ θ — закон Гиббса вместо эвристики), фазы
  EXPLORE/EXPLOIT, admission-скейл T_t/T₀. δ — измеряемый
  зазор (  certified A/B delta). Интегрировано в селектор (opt-in):
  `growth_cycle.py --thermostat-t0/--thermostat-alpha/--thermostat-theta/
  --thermostat-start-gen` — фаза и горизонт пишутся в selector-мету
  каждого поколения; при EXPLOIT цикл останавливается по закону
  (§Thermostat halt, честный стоп в семействе E_dev exhausted /
  AMBIGUOUS). Тесты: test_thermostat.py, test_thermostat_gate.py.

## Liveness two-axis стоп + certificate-telescope (порт 2026-10-13)

- `train/loop.py::liveness_two_axis_frozen` (Unified/Liveness.lean,
  контрапозиция `liveness_two_axis`): честный стоп только когда ОБЕ
  оси мертвы — энергия (контрольный CE плоский) И инъекция
  (grad-norm сигнал плоский) — внутри окна `train.liveness_window`
  (default 0 = off, как saturation/divergence). Тест:
  test_liveness_stop.py.
- `train/stage_telescope.py` (System/GrowthState.lean
  `stage_certificate_telescope`/`termination_from_telescope`):
  каждое ПРИНЯТОЕ поколение пишет StageCertificate в
  `checkpoints/stage_telescope.jsonl`
  (run_ladder.record_stage_certificate); континуитет цепочки с
  шум-допуском 2ε (Hoeffding); backfill из истории лестницы: 22
  сертификата gen4–51 (реальный разрыв 0.03 на gen11→13 — смена
  протокола эвал, честная телеметрия). Φ-бухгалтерия поперёк
  поколений больше не подразумевается — проверяется. Тест:
  test_stage_telescope.py.

## Depth-7 линия (2026-10-15, gen87+)

- **hoeffding_n roundtrip-фикс**: eps, выведенный из n, теперь
  сертифицирует на n (rel-tolerance ceil). До фикса float-шум в ceil
  давал n_req = n+1 → вечный UNDECIDED: лестница была заморожена
  gen52–86 при фактически сертифицированных дельтах (gen80/86:
  delta 0.068–0.072 ≫ 2eps 0.023 при n=14400). Регрессия:
  tests/test_certified_controller.py (gen80 case).
- **Function-preserving depth growth** (`model/depth_grow.py`):
  num_layers 3 → 7 добавлением identity-блоков (зануление
  attn.out_proj и mixer.down новых слоёв; loRA-B/адаптеры новых
  слоёв zero-init по построению). Нюанс: residual_scale — не только
  init, но и runtime clamp-центр BranchScale → старые слои хранят
  исходный residual_scale через config-маркер
  `model.grown_from_layers` (восстанавливается в
  checkpoint.load_model). Паритет логитов бит-в-бит, тест
  test_depth_grow.py + --parity-batch в scripts/growth/depth_grow.py.
- Incumbent gen51_cooldown → gen51_depth7 (сертификат-нейтрально,
  паритет доказан); gen_configs SSOT: num_layers 3 → 7;
  run_ladder --root-override. gen87+ листья обучаются на 7-слойном
  теле (эффективная глубина 7·loop_depth).

## Волна порта R510–R530 (2026-10-16, primes обновлены 2026-10-16)

Порт свежей волны формализации (GibbsFluctuations / CertifiedEstimator /
ThermoPrune / ThermoCondensation / AnytimeCondensation / TopLevel /
GammaDeficit) в чистое ядро — по канону проекта: чистые функции,
посылки h_emp_ в docstring, тест на формулу.

- **Anytime accept-гейт (R525, закрытие открытого пункта THEORY_MAP)**:
  `certified_controller.anytime_safe_threshold(σ², T, α) = √(2σ²ln(T/α))`
  + `anytime_accept_gate` — гейт, оцениваемый в каждом раунде, больше
  не α-инфлирует: uniform-уровень α над T проверками требует anytime
  полосы u. Радиус конденсации = u, стоимость rebuild = u².
- **Risk-терминация телескопа (R529)**:
  `stage_telescope.risk_budget_termination` — T ≤ (Φ₀ − φmin + R)/ε;
  risk-спенд входит в горизонт линейно (сильнее и честнее прежней
  равномерной формы eps−r в знаменателе).
- **γ-бюджет миксера (R530, закрытие «γ-дефицит mixer.gain→0»)**:
  `gamma_min_for_target(α, C*, D̄) = αC*/D̄`, `gamma_ceiling`,
  `gamma_deficit_no_takeoff` — сертифицированный вердикт: при
  γ·D̄ < α·C* потолок capability γ·D̄/α и НИКАКОЙ schedule его не
  пробивает; выход только через рост γ (disagreement_distill) или
  diversity D̄.
- **Thermostat variance-stop (R498/R510–R512)**: Var_{p_T}(E) ≤
  Δ²(n−1)e^{−δ/T}; аналитическая температура останова
  T ≤ δ/ln(Δ²(n−1)/ε) и число шагов k+1 ≥ ln(Δ²(n−1)/ε) на графике
  T_k = δ/(k+1) — вторая ось термостата после R387–R388 (exploration
  stop): теперь обе оси останавливаются по закону, а не по эвристике.
- **Термо-конденсация (R522/R527/R528)**:
  `formal.condensation_budget_cost(M) = M²/4`,
  `condensation_allowed_deviation(ε) = 2√ε`,
  `anytime_condensation_cost(σ², T, α) = 2σ²ln(T/α)` — «знание в
  ≤1,5% каркаса, отрезаемого и перестраиваемого за ε» теперь
  числовой интерфейс для prune/rebuild-решений.

Тесты: tests/test_primes_wave_r510.py (17); полная батарея 684 passed.

## Ladder unfreeze (2026-10-16, ход 2–4 goal-сессии)

- **Root cause #1 (заморозка gen87–89 в ночь 15–16.10)**: при
  `--root-override gen51_depth7` конфиг инкумбента выводился из gen-id
  (gen51 → 3-слойный root_ft.yaml) против 7-слойного чекпоинта →
  eval-сабпроцесс падал IncompatibleCheckpointError с пустым stdout →
  avg_ce=None → «eval failed — halting». Фикс: `_ckpt_embedded_
  config_yaml` — конфиг инкумбента извлекается из чекпоинта
  (format_version ≥ 12 встраивает полный конфиг; веса доменов
  проверены идентичными общему протоколу). Коммит baff027.
- **Root cause #2**: тот же Path-объект падал в json.dumps кэша avg_ce
  на первой углубляющей ступени (150 батчей). Фикс: str() в записи
  кэша.
- **Честная переоценка**: инкумбент gen51_depth7, измеренный на своём
  конфиге, даёт дельту gen87 +0.0145@180 → +0.0348@900 → +0.0423@3600
  (монотонный рост сквозь ступени — реальный сигнал, noise-floor guard
  не сработал), а не +0.068 как против 3-слойного gen51_cooldown на
  несравнимом протоколе. Прежние «would-accept» дельты 0.068 — против
  другой базы; certified-гейт честно потребовал углубления.
- Гистерезис заморозки gen52–86 (accept_required_n = accept_n+1)
  задокументирован ранее (2026-10-15, roundtrip-фикс hoeffding_n);
  перезапуск после фикса показал, что реальная дельта на общем
  протоколе иная — гейт работал как задуман.
