# HAGI — Архитектура проекта

Единый документ о структуре HAGI_v2: что делает модель, как устроен код,
как устроен пайплайн рекурсивного роста и как всё связано между собой.

HAGI — это **тернарный RD-канальный каузальный языковой модуль** (BitNet
b1.58). Модель трактуется как канал связи:

```
tokens (+ опциональный фиксированный image/audio prefix)
  -> source coder   (codebook + каузальный pulse-shaping фильтр)
  -> ternary channel (L блоков, QK-нормализованный GQA + SwiGLU)
  -> conditional head (shared-bank NCE, q = unigram source prior)
  -> exact receiver  (полный алфавит для генерации и калибровки)
```

Текущая версия — **V42** (`hagi-channel-v42`): полное каузальное внимание на
всех слоях (W=0), T=512, punctured receiver. Исторические описания — в
`docs/V41_ARCHITECTURE.md` и `docs/V42_ARCHITECTURE.md`.

---

## 1. Обзор структуры репозитория

```
HAGI_v2/
├── src/hagi/                 # основной пакет
│   ├── config.py             # все архитектурные конфиги + валидация
│   ├── version.py            # версия и идентичность архитектуры
│   ├── data/                 # пайплайн данных
│   │   ├── dataset.py        # packed-corpus загрузчик
│   │   └── vocab_map.py      # маппинг компактного словаря
│   ├── inference/
│   │   └── generate.py       # авторегрессивная генерация с KV-cache
│   ├── model/                # архитектура модели
│   │   ├── model.py          # HAGI (главный класс)
│   │   ├── merge.py          # MergedHAGI + Hadamard/CrossMixer
│   │   ├── block.py          # трансформер-блок
│   │   ├── attention.py      # GQA + QK-norm + RoPE + windowing
│   │   ├── embedding.py      # SourceEncoder (codebook + conv-фильтр)
│   │   ├── ffn.py            # SwiGLU + BranchScale + linear
│   │   ├── head.py           # LMHead (receiver)
│   │   ├── norms.py          # RMSNorm / BlockRMSNorm / HeadNorm
│   │   ├── rope.py           # 1D/2D RoPE
│   │   ├── kv_cache.py       # KV-cache
│   │   ├── outputs.py        # ModelOutput
│   │   ├── ternary.py        # BitLinear (b1.58 квантование)
│   │   └── multimodal.py     # мультимодальный мост (опционально)
│   └── train/
│       ├── loop.py           # тренировочный цикл
│       ├── optim.py          # Muon + AdamW (HybridOptimizer)
│       ├── checkpoint.py     # сохранение/загрузка чекпоинтов
│       └── _rocm_fsdp_stub.py# заглушка для ROCm Windows
├── scripts/                  # CLI-скрипты
├── configs/                  # YAML-конфиги
├── docs/                     # историческая документация
├── tests/                    # pytest-тесты
├── data/                     # корпус (.bin, mix.json, unigram)
├── checkpoints_*/            # чекпоинты (gitignored)
├── logs/                     # логи обучения (gitignored)
├── GROWING_HYPOTHESIS.md     # гипотеза рекурсивного роста
├── AGENT_WORKLOG.md          # журнал работы агента
├── BENCHMARKS.md             # результаты бенчмарков
└── README.md                # краткое введение
```

---

## 2. Архитектура модели

### 2.1 Source coder — `model/embedding.py`

`SourceEncoder` — кодирование токенов в пространство канала:

- **Codebook** — полный `[V, H]`-таблица. Не факторизуется: rank-r
  факторизация даёт жёсткий информационный потолок (измерено: 1.42 nats
  остаточной KL при r=128 против 0.92 при r>=512). При `tie_lm_head=True`
  таблица разделяется с LM-головой.
- **Transmit filter** — каузальный depthwise Conv1d (pulse shaping). Смешивает
  каждый символ с `k-1` предыдущими. Только left-pad: симметричный pad
  протекает будущее в позицию t и ломает каузальность. Несёт decode-state для
  инкрементальной генерации.

### 2.2 Ternary channel — `model/ternary.py`, `model/block.py`

- **BitLinear** — 2D-вес квантуется в `{-s, 0, +s}` (BitNet b1.58, 1.585
  бит/вес). `s = mean(|W|, dim=1)` — per-output-channel масштаб. STE —
  identity. Step-cache (OFDM coherence interval): квантованная карта
  замораживается на один optimizer step.
- **Block** — `x + attention(x)` затем `x + mixer(x)`, оба pre-norm.
- **SwiGLU** — `down(silu(gate(x)) * up(x))`. `BranchScale` — обучаемый
  ограниченный масштаб ветви (защита от роста нормы под Muon).

### 2.3 Attention — `model/attention.py`

GQA-коррелятор с тремя информационно-теоретическими свойствами:

- **QK-norm** (`HeadNorm`) — ограничивает диапазон logits, не даёт softmax
  насытиться (главная защита от дивергенции V30).
- **Sliding window** — конечное состояние канала O(T·W); полные слои —
  глобальные relay. В V42 W=0 (все слои полные).
- **KV-cache** — замороженное состояние, декодирование O(T).

Fused QKV-проекция (одна матрица вместо трёх). Маски строятся один раз на
размер окна и разделяются между слоями.

### 2.4 Receiver — `model/head.py`

`LMHead` — четыре ключевые идеи:

- **Source prior** — фиксированный `log p_unigram` добавляется к logits.
  Нулевой порядок кода бесплатен (~4.4 nats/token на инициализации).
- **Receiver gain** — один обучаемый скаляр, инициализированный
  `1/sqrt(H)`. Без него tied codebook даёт +41 logit на собственном токене и
  уничтожает prior.
- **z-loss** — штраф `logsumexp(logits)^2`, пиннит нормализацию у нуля.
- **Chunked cross-entropy** — logits считаются блоками строк, никогда не
  материализуется `[N, V]`. При `sampled_softmax_k > 0` — shared-bank NCE
  (условный NCE против source prior).

### 2.5 Мультимодальность — `model/multimodal.py`

Опциональный Q-Former-подобный мост: сжимает любую модальность в фиксированное
число `n_bridge_queries` токенов. Выключен в текущих конфигах.

---

## 3. Блочно-диагональное слияние экспертов — `model/merge.py`

Метод «train-many-small, merge-into-big»:

1. N маленьких экспертов обучаются до насыщения на N корпусах.
2. Их скрытые пространства конкатенируются блочно-диагонально в одну широкую
   модель: `W_Q = diag(W_Q^A, ..., W_Q^N)`.
3. На шаге 0 модель — ровно N независимых экспертов (off-diagonal нули).
4. Небольшое число cross-block mixer'ов (identity-init, gain 0) — единственные
   новые связи; короткое joint-обучение учит блоки взаимодействовать.

### 3.1 `MergedHAGI`

- Encoder/out-norm/head — широкие (merged) версии.
- Body-блоки — блочно-диагональная конкатенация экспертов.
- Широкие RMSNorm заменяются на `BlockRMSNorm` (per-expert нормализация).
- Per-head QK-gain (`per_head_qk=True`) — каждый блок сохраняет свой gain.
- Тернарное квантование отключается при слиянии (не коммутирует с
  блочно-диагональностью); экспертные блоки тернаризуются индивидуально.

### 3.2 Mixer'ы

- **`CrossMixer`** (swiglu) — полный SwiGLU `H x 2H`, единственный
  cross-block канал. `y = x + gain * down(silu(gate(x)) * up(x))`.
- **`HadamardMixer`** (hadamard, по умолчанию) — фиксированный быстрый
  Hadamard-трансформ по оси экспертов `(H_n/sqrt(n)) ⊗ I_H` плюс малый
  обучаемый low-rank остаток (rank `mixer_rank`). На шаге 0 каждый блок видит
  нормированную сумму/разность всех остальных — за O(NH log N) FLOPs и ноль
  параметров. ~16x дешевле по параметрам и ~23% быстрее на шаг, с
  идентичными step-0 logits (head pre-rotated).

### 3.3 Рекурсивный/локальный Hadamard

`mixer_hadamard_groups: [2, 2]` задаёт рекурсивное дерево (16→4→1):
`H = H_{g_k} ⊗ ... ⊗ H_{g_1}` (Kronecker-произведение per-level Hadamard).
Каждый фактор ортонормален, произведение ортонормально. Для равномерных групп
это равно глобальному `H_n` с точностью до перестановки каналов — разница в
*порядке* sum/difference каналов, который решает, какие комбинации читаются
как «крупномасштабные общие» vs «мелкомасштабные различия». Механизм готов к
пайплайну роста 16→4→1, где каждый уровень несёт свой локальный mixer.

**Head pre-rotation**: при Hadamard-mixer'е скрытый поток вращается на
`Q = (H_n/sqrt(n)) ⊗ I_H`. Чтобы step-0 logits совпадали с блочно-диагональным
слиянием, head-вес правомножается на `Q` (`hadamard_apply_2d`). Входной
embedding НЕ вращается (он кормит первый блок, не head), кроме случая
`tie_lm_head=True`, когда embedding и есть head.

### 3.4 Иерархическое слияние

`drop_expert_mixers=True` — при иерархическом слиянии эксперты сами являются
`MergedHAGI` со своими mixer'ами, которые отбрасываются и заменяются свежим
mixer'ом следующего уровня. `logit_scale` делится на `sqrt(N)` (широкая head
суммирует N вкладов).

---

## 4. Пайплайн рекурсивного роста

Текущий пайплайн (level-1):

```
Level-0: 18 per-corpus экспертов (H=128) -> блочно-диагональное слияние -> H=2304
Level-1: 4 эксперта (ru/en/math/instruct, H=2304) из общего prior
         -> слияние с рекурсивным Hadamard [2,2] -> H=9216 (~1.018B)
Joint:   короткое обучение, учит блоки взаимодействовать
```

### 4.1 Скрипты

| Скрипт | Назначение |
|--------|-----------|
| `scripts/train.py` | train / resume / init-from модели |
| `scripts/merge_level1.py` | слияние 4 level-1 экспертов в H=9216 |
| `scripts/merge_experts.py` | низкоуровневый CLI блочно-диагонального слияния |
| `scripts/run_level1_pipeline.sh` | end-to-end драйвер level-1 |
| `scripts/eval_domains.py` | per-domain exact CE / perplexity |
| `scripts/infer.py` | интерактивный диалог |
| `scripts/compact_vocab.py` | компактизация словаря |
| `scripts/count_unigram.py` | подсчёт unigram-частот |
| `scripts/preprocess_gemma.py` | raw .jsonl -> .bin |
| `scripts/rebuild_compact2.py` | пересборка .compact2.bin |
| `scripts/bench_train_step.py` | wall-time бенчмарк шага |
| `scripts/bench_sampled_head.py` | A/B бенчмарк receiver |
| `scripts/breakdown.py` | разбивка времени шага по фазам |
| `scripts/prof_step.py` | профилирование шага |
| `scripts/smoke_real.py` | smoke-тест реального пайплайна |

### 4.2 Конфиги

- `configs/level0_merged.yaml` — level-0 merged (H=2304, 18 экспертов).
- `configs/level1/expert_*.yaml` — 4 level-1 эксперта (ru/en/math/instruct).
- `configs/level1_merged.yaml` — merged H=9216 (Hadamard `[2,2]`).

### 4.3 Чекпоинты

- `checkpoints_corpus/` — level-0 эксперты (по корпусам/срезам).
- `checkpoints_level0_merged/` — level-0 merged.
- `checkpoints_l1/` — level-1 эксперты (ru_general, en_general, math_code,
  instruct).
- `checkpoints_l1_merged/` — merged H=9216 + joint-обучение.

---

## 5. Данные — `data/dataset.py`

**Packed-corpus пайплайн**: документы конкатенируются в плоский поток токенов
и режутся на окна фиксированной длины без паддинга. Каждое окно несёт
`doc_ids`, маска внимания блочно-диагональна по документам. Утилизация 0.41–0.80
при T=512 против 0.10–0.43 при T=2048 у per-document схемы.

- `PackedStream` — memory-mapped поток, конечный (recursive-growth примитив:
  следующий эксперт продолжает с `start_offset`).
- `PackedMixDataset` — бесконечный итератор пропорционально-смешанных окон.
- `load_mix` — per-source веса из `mix.json`.
- `dataset_path` — предпочитает `.compact2.bin` > `.compact.bin` > `.bin`.

---

## 6. Обучение — `train/`

### 6.1 `loop.py`

Один objective, один раз на microbatch. Gradient accumulation взвешивает
каждый microbatch по числу scored-токенов. Ключевые наблюдаемые: `ce`
(против unigram-энтропии 8.06 nats), `qk_gain` (индикатор насыщения softmax),
`logit_scale` (рост gain'а receiver'а).

- `puncture_loss_mask` — erasure channel на supervision (ce_keep_rate).
- `cast_model` — bf16 + fp32-гейны (`keep_fp32`).
- `clip_gradients_by_group` — раздельный клип Muon/AdamW.
- Saturation early-stop на exact_ce.

### 6.2 `optim.py`

- **Muon** — momentum SGD с Newton-Schulz ортогонализацией для 2D channel
  весов (маркер `is_channel_weight`).
- **AdamW** — для codebook, 1D gain'ов, bias'ов, router'ов.
- `HybridOptimizer` — драйвер обоих как одного.
- WSD schedule (warmup-stable-decay), inverse-sqrt stable.

### 6.3 `checkpoint.py`

Формат 12. Строгая валидация схемы до применения (частично применённый load —
самый дорогой сбой). Атомарная запись (temp + `os.replace`), ротация
`keep_last`. `load_model` с `skip_prefixes` для свежих mixer'ов при
рекурсивном росте.

---

## 7. Генерация — `inference/generate.py`

Prefill (один forward, заполняет KV-cache) + decode (single-position forward).
`filter_logits` — repetition penalty → temperature → top-k → top-p (порядок
важен). `use_cache=False` — полный пересчёт (O(T²), для верификации кэша).

---

## 8. Конфигурация — `config.py`

Все архитектурные ручки в dataclass'ах: `ModelConfig`, `AttentionConfig`,
`SlidingWindowConfig`, `EmbeddingConfig`, `FFNConfig`, `TernaryConfig`,
`HeadConfig`, `MultimodalConfig`, `TrainConfig`, `MergeConfig`,
`InferenceConfig`, `MuonConfig`, `AdamConfig`, `ScheduleConfig`, `DataConfig`,
`LoggingConfig`.

- `load_config` — YAML + dotted overrides + `auto_configure` (решение H/L из
  body-бюджета).
- `validate_config` — все структурные инварианты fail-fast (включая
  `mixer_hadamard_groups`: степени двойки, произведение == n_experts).
- `count_params` — аналитический подсчёт параметров по группам.
- `describe` — одноблочное человекочитаемое резюме.

---

## 9. Тесты — `tests/`

pytest-набор по модулям: `test_attention`, `test_checkpoint`, `test_config`,
`test_dataset`, `test_embedding`, `test_generate`, `test_head`, `test_layers`,
`test_loop`, `test_model`, `test_multimodal`, `test_optim`, `test_ternary`,
`test_vocab_map`. Покрывают merge/hadamard/model/config пути.

---

## 10. Ключевые идеи (сводка)

1. **Канал связи**: source coder → ternary channel → receiver. Квантование —
   rate constraint, ошибка квантования — единственный шум.
2. **Source prior** — бесплатный нулевой порядок кода (~4.4 nats/token).
3. **QK-norm** — защита от насыщения softmax (главный фикс V30).
4. **Packed corpus** — 2-9x эффективный throughput против per-document.
5. **Блочно-диагональное слияние** — наследует N сформированных
   специализированных подпространств вместо их открытия с нуля.
6. **Hadamard mixer** — фиксированная sum/difference связь на шаге 0,
   рекурсивная версия готова к 16→4→1 росту.
7. **Recursive growth** — каждый уровень специализируется из общего prior
   предыдущего уровня.
