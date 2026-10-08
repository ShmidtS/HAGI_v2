# HAGI — Архитектура

Единый документ: как устроены модель, код и пайплайн роста поколениями
в текущем состоянии. Версия **V42** (`hagi-channel-v42`, 4.2.0).
Исторические результаты и rationale — в git history и `.omc/attempts/`.

HAGI — **языковая модель, растущая поколениями**. Обучаем три
небольшие модели-специалиста (листья), объединяем их в одну большую
точным слиянием (на первом шаге объединённая модель работает ровно
как ансамбль трёх), добавляем между блоками дешёвую связь
(ортогональный миксер + низкоранговая добавка) и коротко дообучаем.
Результат становится отправной точкой трёх специалистов следующего
поколения: 3 листа → слияние → дообучение → новое поколение.

**Рекордная линия (mixed 8-corpora gate, exact CE):**

```
8 корпусов (здоровая смесь, веса в mix.json)
  → листья H=128 (dense тело, 3 слоя): pre-norm GQA + QK-norm + RoPE,
     SwiGLU (exp=1), untied голова, fused CE, zero-init proj
  → ТРОИЧНОЕ слияние 3 листа → родитель H=3·H_leaf
     (block-diagonal, шаг-0 = ансамбль, off-diagonal нули)
  → Hadamard-миксер (ортонормированный для любого N, rank-64 residual)
  → joint-обучение; BranchScale clamp-8 per-layer
  → поколение N+1: 3 сиба от joint-prior → merge → joint (рекурсия)
  → поверх: TableLoRA r16 на embedding/head (ранговый канал)
```

Линия роста — **троичное дерево 3→1 на каждом уровне** (не степени
двойки): gen-1 = 3 листа H=128 → merged H=384; gen-2 = 3 сиба H=384 →
merged H=1152; и т.д. Hadamard-трансформ строится для любого N:
степень двойки — прямой Sylvester, иначе pad → первые N строк →
QR-ортонормализация (см. `merge._hadamard_orthonormal`).

Выключено в победной линии (измерено/доказано избыточным на этом
масштабе): ternary body, cortex, adapters, decision, unigram_prior,
NCE-приёмник, conv-фильтр (k=1), head_up (rank-контроль), рост батча
(B_noise ≈ 100 токенов), per-matrix NS (фикс 5 шагов достаточен),
полный SwiGLU cross-mixer (hadamard равен при 12× меньших параметрах).

Словарь (механизмы, не метафоры): «ternary» = BitNet-квантование весов
(доступно, выключено); «merge» = блочно-диагональная конкатенация;
«puncturing» = подвыборка токенов в loss; «leaf/sib» = эксперт
H=128 / потомок родителя; «joint» = короткое обучение после merge.

---

## 1. Структура репозитория

```
HAGI_v2/
├── src/hagi/
│   ├── config.py             # dataclass-конфиги + валидация + count_params
│   ├── version.py            # V42, идентичность архитектуры
│   ├── data/
│   │   ├── dataset.py        # packed-corpus поток + смешанные окна
│   │   ├── artifacts.py      # opt-in versioned manifests (граница приёмки)
│   │   └── vocab_map.py      # маппинг compact↔исходного словаря
│   ├── inference/generate.py # авторегрессионная генерация + KV-cache
│   ├── model/
│   │   ├── model.py          # HAGI (главный класс)
│   │   ├── merge.py          # MergedHAGI, Hadamard/CrossMixer, слияние
│   │   ├── block.py          # трансформер-блок (pre-norm residual)
│   │   ├── attention.py      # GQA + QK-norm + RoPE + windowing
│   │   ├── embedding.py      # SourceEncoder (codebook + transmit filter)
│   │   ├── ffn.py            # SwiGLU + BranchScale
│   │   ├── head.py           # LMHead (receiver): fused CE, exact_loss
│   │   ├── ternary.py        # BitLinear (b1.58), STE, step-cache
│   │   ├── table_lora.py     # TableLoRA на embedding/head
│   │   ├── norms.py          # RMSNorm / BlockRMSNorm / HeadNorm
│   │   ├── rope.py           # 1D/2D RoPE
│   │   ├── kv_cache.py       # KV-cache
│   │   ├── cortex.py         # opt-in cross-level канал (выключен)
│   │   ├── decision.py       # opt-in decision head (research-only)
│   │   ├── multimodal.py     # Q-Former мост (выключен)
│   │   └── outputs.py        # ModelOutput
│   └── train/
│       ├── loop.py           # Trainer: train_step, гейты, gram-scan
│       ├── optim.py          # Muon + AdamW (HybridOptimizer), WSD
│       ├── checkpoint.py     # формат 12, атомарная запись, lenient reload
│       ├── safeqp_controller.py # per-corpus Gram-scan + SafeQP сертификат
│       ├── self_improve.py   # opt-in self-improvement (gradient | rls)
│       ├── ttt.py            # признаки → delta LoRA (anchored RLS)
│       └── _rocm_fsdp_stub.py
├── scripts/                  # CLI, разбит по модулям:
│   ├── train.py, infer.py, eval_domains.py, eval_holdout.py  # ядро
│   ├── data/                 # подготовка корпусов (compact, pack, unigram)
│   ├── growth/               # цикл роста (merge, gate, supervisor, f3)
│   ├── lora/                 # ранговый канал (TableLoRA)
│   ├── audits/               # измерительные сертификаты ручек
│   ├── kernels/              # triton-кернелы HAGI
│   ├── release/              # сборка/публикация релизов
│   ├── dsv4/                 # трек сжатия DeepSeek-V4 (отдельная линия)
│   └── research/             # исторические эксперименты (e3–e7, qwen, …)
├── configs/                  # YAML победной dbridge-линии (+фикстуры)
├── tests/                    # pytest (953)
├── data/                     # .compact.bin корпуса, mix.json, unigram
├── checkpoints/              # вне git
└── .omc/attempts/            # ledgers раундов (в git через add -f)
```

---

## 2. Модель

### 2.1 SourceEncoder (`model/embedding.py`)

Полная таблица `[V, H]` (не факторизуется: rank-r даёт информационный
потолок). Transmit filter — каузальный depthwise Conv1d (k=1 в линии,
т.е. выключен), несёт decode-state для инкрементальной генерации.

### 2.2 Блок (`model/block.py`, `ffn.py`, `attention.py`)

`x + attention(x)`, затем `x + ffn(x)`, оба pre-norm.

- **Attention**: GQA, fused QKV-проекция, QK-norm (`HeadNorm`,
  защита от насыщения softmax), RoPE. Полное внимание (W=0).
- **FFN**: SwiGLU `down(silu(gate(x)) * up(x))`, expansion 1.
- **BranchScale** — обучаемый per-branch масштаб с клампами
  `[r/branch_clamp_ratio, r·ratio]`; `branch_clamp_ratio=8` в линии —
  per-layer компенсация измеренной дисперсии ветвей.

Тело dense (ternary доступен через `BitLinear`, в линии выключен).

### 2.3 LMHead (`model/head.py`)

Untied голова. Receiver gain (обучаемый скаляр), z-loss на
нормализацию, chunked cross-entropy (logits блоками, `[N,V]` никогда
не материализуется), `exact_loss` — точный CE для гейтов.
`head_up` (up-projection) доступен, но в линии выключен.

### 2.4 KV-cache (`model/kv_cache.py`)

Декодирование O(T); `use_cache=False` — полный пересчёт (верификация).

---

## 3. Троичное слияние — `model/merge.py`

Схема «train-many-small, merge-into-big», **N=3 на каждом уровне**:

1. 3 эксперта (H=H_leaf) обучаются до насыщения (разные data-seed'ы,
   общий prior у поколений ≥ 1).
2. Скрытые пространства конкатенируются блочно-диагонально:
   `W = diag(W_A, W_B, W_C)`; parent H = 3·H_leaf.
3. Шаг 0 merged-модели — ровно ансамбль 3 экспертов (Jensen-выигрыш
   бесплатно, off-diagonal нули, `logit_scale / 3`).
4. Cross-block миксер — единственные новые связи; joint-обучение
   короткое.

### 3.1 Миксеры

- **`HadamardMixer`** (по умолчанию): фиксированный ортонормальный
  трансформ по оси экспертов + обучаемый low-rank residual (rank
  `mixer_rank=64`). Для N=3 — pad→QR-ортонормализация (любое N).
  На шаге 0 каждый блок видит нормированную сумму/разность остальных
  за O(NH log N) FLOPs и ноль параметров транспорта.
- **`CrossMixer`** (swiglu): полный SwiGLU H×2H — резервный путь,
  равен по качеству при 12× параметров.
- **Head pre-rotation**: при hadamard head-вес правомножается на Q,
  чтобы шаг-0 logits совпадали с блочно-диагональным слиянием
  (function-preserving инвариант `mixer_invisible_condition`).

### 3.2 Иерархический рост

`drop_expert_mixers=True` — при слиянии merged-экспертов их миксеры
сбрасываются, ставится свежий миксер следующего уровня. Родитель
поколения g становится prior'ом (`train.init_from`) сибов поколения
g+1 — рекурсивный цикл: 3 сиба → merge → joint → снова 3 сиба.

Гард: `train.zero_init_proj ∧ merge.enabled` → ValueError (конфиг
не пройдёт валидацию — zero-init уничтожает function-preserving шаг-0).

---

## 4. Пайплайн роста (победная dbridge-линия)

```
листья s1..s3 (H=128, seeds различны)         [configs/dbridge_leaf_s*.yaml]
  → gen1 merged (H=384)  + joint 1600 шагов   [dbridge_gen1_merged/joint.yaml]
  → сибы g2-s1..s3 (H=384, init_from gen1 joint, новые data-seed)
  → gen2 merged (H=1152, hadamard, clamp-8)
     + joint                                  [dbridge_gen2_merged_had_c8.yaml]
  → TableLoRA r16 поверх                      [scripts/lora/lora_gen2_joint_c8.py]
  → (далее) gen-3 сибы от gen2 joint — та же схема
```

Гейт (verdict-протокол): 8 корпусов, weighted exact CE, tail-окна
(−2M, 300k токенов), 2×1024 на корпус, сравниваются только
идентичные окна. Смешанная оценка через `head.exact_loss`.

Равный бюджет для всех рук: 1600 шагов joint ≈ 52M токенов;
контроль scratch (равные FLOPs) проигрывает росту ~1 нат.

---

## 5. Данные — `data/dataset.py`

Packed-corpus: документы конкатенируются в плоский поток, окна
фиксированной длины без паддинга, `doc_ids` → блочно-диагональная
маска внимания. Утилизация 0.41–0.80 против 0.10–0.43 у per-document.

- `PackedStream` — memory-mapped, конечный; следующий эксперт
  продолжает с `start_offset` (примитив рекурсивного роста).
- `PackedMixDataset` — бесконечные пропорционально-смешанные окна
  (каждый шаг видит всю дистрибуцию корпусов).
- `dataset_path`: `.compact2.bin` > `.compact.bin` > `.bin`;
  compact-словарь 32768 (`vocab_map.npz` + gemma-токенизатор для
  инференса-текста).

8 корпусов и веса: edu .3571, python_instruct .2232, wikipedia_en
.0893, wikipedia_ru .0714, oscar_ru .0625, openwebmath .0893,
tinystories .0536, smoltalk .0536.

---

## 6. Обучение — `train/`

### 6.1 `loop.py`

Один forward на microbatch; gradient accumulation нормализует scored
LM tokens. Ключевые наблюдаемые: `ce` (против unigram-энтропии 8.06
nats), `qk_gain`, `logit_scale`, `exact_ce` (периодический точный CE).

- `puncture_loss_mask` — erasure channel на supervision
  (`ce_keep_rate`).
- `clip_gradients_by_group` — раздельный клип Muon/AdamW.
- Saturation early-stop на exact_ce.
- **Gram-scan** (`train.gram_scan_interval`): периодический
  per-corpus градиентный скан (cos-матрица, доминация, конфликт-флаги,
  SafeQP-сертификат) — log-only канал измерения конфликтов смеси.

### 6.2 `optim.py`

Muon (Newton–Schulz, ns_steps=5) для 2D channel-весов + AdamW для
таблиц/гейнов; `HybridOptimizer` как один драйвер; WSD-schedule
(warmup-stable-decay, inverse-sqrt stable). В fresh-руках H≥384:
compile off, fused_ce off, z_loss 1e-4, Muon off, lr 1e-3;
`logit_scale_max` кламп в optim.

### 6.3 `checkpoint.py`

Формат 12. Строгая валидация схемы на live-конфигах; lenient reload
для чекпоинтов с удалёнными полями (только путь перезагрузки).
Атомарная запись (temp + os.replace), ротация keep_last.

### 6.4 `safeqp_controller.py`

`corpus_grad_gram()` — пер-корпусные градиенты на tail-калибровке,
Gram/cos/доминация, конфликт ⟺ ⟨g_i,ḡ⟩ < −ε‖g_i‖‖ḡ‖; при конфликте
двойственное решение SafeQP с сертификатом descent. Обновления не
меняет (log-only).

### 6.5 `self_improve.py`, `ttt.py`

Opt-in цикл «сыграл → оценил → обновился»; база заморожена, транзакционный
конверт с KL-границей и откатом. `ttt.py` — замкнутая RLS-подгонка
delta-LoRA из признаков (одна teacher-forced генерация, без второго
роллаута). В победной линии не задействованы.

---

## 7. Генерация — `inference/generate.py`

Prefill (один forward, KV-cache) + decode (single-position).
`filter_logits`: repetition penalty → temperature → top-k → top-p.
Текстовый мост: compact id → `vocab_map.npz` → gemma-токенизатор.

---

## 8. Конфигурация — `config.py`

Все параметры конфигурации описаны классами Python (dataclass):
`ModelConfig` (hidden, layers, rope, head_up, branch_clamp_ratio),
`AttentionConfig`, `EmbeddingConfig`, `FFNConfig`, `TernaryConfig`,
`HeadConfig`, `TrainConfig` (learning rate, puncture, gram-scan,
saturation, init_from), `MergeConfig` (n_experts, mixer_type,
mixer_rank, expert_checkpoints, drop_expert_mixers),
`InferenceConfig`, `MuonConfig`/`AdamConfig`/`ScheduleConfig`,
`DataConfig`, `LoggingConfig`.

- `load_config` — YAML + dotted overrides + `auto_configure`.
- `validate_config` — fail-fast инварианты (включая merge-гарды).
- `count_params` — аналитический подсчёт по группам.

Живые конфиги (`configs/`): dbridge-линия (leaf s1–s3, gen1
merged/joint, gen2 sib1–3/merged/joint/merged_had/merged_had_c8,
loop2 s1–s3, f3_d1, scratch_h384, leaf_rank384) + фикстура
`parent_h128_flat.yaml` (пиннинг merge-инварианта в тестах).

---

## 9. Тесты — `tests/`

953 теста: merge-инварианты (шаг-0 = ансамбль, temperature),
hadamard-ортогональность, formal-utils (safe_qp, ns, optimal_batch),
safeqp-controller (конфликт/сертификат/чистые грады), certified-step
(gate/step/''-пути), config, checkpoint, dataset, attention, generate,
head, vocab_map, distill-канал, resume-precedence, супервизорные гварды.

---

## 10. Ключевые принципы

1. **При равном бюджете вычислений много узких специалистов лучше
   одного широкого «с нуля»**: каждый учится на своём распределении
   данных, а слияние объединяет их без потери качества (измерено:
   выигрыш ~1 нат против обучения с нуля тем же бюджетом).
2. **Merge — центральный механизм роста**: слияние наследует уже
   обученные подпространства (не надо открывать их заново), а связь
   между блоками — дешёвая надстройка (ортогональный миксер без
   обучаемых параметров + небольшая низкоранговая добавка).
3. **Function-preserving инварианты обязательны**: любой новый миксер
   сначала проверяется на шаг-0 тождество (head pre-rotation, гард
   zero_init_proj).
4. **Гиперпараметры вычисляются, а не подбираются перебором**:
   LoRA-ранг — из спектра сингулярных чисел (какая доля сигнала не
   покрыта базовой моделью); масштабы ветвей (BranchScale) — из
   измеренной дисперсии активаций слоя; число итераций
   Newton–Schulz — из доли малых сингулярных чисел; размер батча —
   из измеренного шума градиента (на этой линии шум исчезает на
   ~100 токенах — большой батч бесполезен). Дорогие GPU-эксперименты
   запускаются только после дешёвого расчёта-сертификата на уже
   обученной модели (ρ_c-гейт, Gram-scan) — он предсказывает
   выигрыш до запуска.
5. **Отбор кандидатов — по вкладу в ансамбль, а не по одиночному
   качеству**: лучший отдельно обученный эксперт может оказаться
   худшим дополнением к уже имеющимся (доказано контрпримером);
   поэтому кандидат оценивается по новой информации, которую он
   привносит (расхождение с уже покрытыми направлениями — метрики
   Fisher/Jensen-novelty).
6. **Порядок добавления механизмов** (от дешёвых к дорогим):
   специализация листьев → слияние → дешёвая связь между блоками →
   короткое joint-дообучение → низкоранговые добавки (LoRA) поверх.
