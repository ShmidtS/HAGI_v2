# HAGI

**HAGI** is a **language model that grows in generations**. Train
three small specialist models (leaves), merge them into one larger
model exactly (at step 0 the merged model behaves identically to the
ensemble of the three), add a cheap communication channel between
the blocks, and briefly fine-tune. The result seeds three new
specialists, and the cycle repeats: 3 leaves → merge → fine-tune →
next generation. At fixed compute, many narrow specialists beat one
wide from-scratch model (measured).

Full architecture and code description: **[ARCHITECTURE.md](ARCHITECTURE.md)**
(Russian; this summary is the English entry point).

---

**Языковая модель, растущая поколениями.** Обучаем три небольшие
модели-специалиста (листья), объединяем их в одну большую точным
слиянием (на первом шаге объединённая модель работает ровно как
ансамбль трёх), добавляем дешёвую связь между блоками и дообучаем.
Полученная модель становится отправной точкой для трёх новых
специалистов — и цикл повторяется: 3 листа → слияние → дообучение →
новое поколение.

Полное описание архитектуры и кода — **[ARCHITECTURE.md](ARCHITECTURE.md)**.
Измеренные результаты и rationale раундов — git history + `.omc/attempts/`.

Версия: **V42** (`hagi-channel-v42`, 4.2.0).

## Схема роста (победная dbridge-линия)

```
8 корпусов (смешанные окна, packed-corpus)
  → 3 листа H=128 (dense тело, GQA + QK-norm + RoPE, SwiGLU, fused CE)
  → троичное слияние → gen-1 H=384 + joint
  → 3 сиба H=384 (init_from gen-1 joint, новые data-seed)
  → троичное слияние → gen-2 H=1152 (Hadamard-миксер, BranchScale clamp-8)
    + joint
  → TableLoRA r16 на embedding/head поверх
  → gen-3: та же схема от gen-2 joint
```

Фактическое состояние линии (чекпойнты на диске):

```
gen-2 joint (H=1152, 1600 шагов, CE ~3.4-4.1)
  → 3 доменных сиба H=384  [math / lang / code]  ← checkpoints/gen2_dsib_*
  → gen-3 merged H=1152 (Hadamard-миксер, block-diag 3→1)   ← configs/dbridge_gen3_merged.yaml
  → gen-3 joint                                              ← configs/dbridge_gen3_joint.yaml
```

Замечание: `configs/dbridge_gen2_merged_had.yaml` ссылается на
`dbridge_gen2_sib*`, которых на диске нет — фактические сибы лежат в
`gen2_dsib_*`. Это и была причина `GEN2-MERGE-FAIL` в `logs/chain.log`.
Конфиг gen-3 указывает на существующие пути.

Слияние — **троичное 3→1 на каждом уровне** (не степени двойки);
Hadamard-трансформ строится для любого N (pad → QR-ортонормализация).
Шаг-0 merged-модели — ровно ансамбль экспертов (Jensen-выигрыш бесплатно),
затем единственные новые связи — миксер и joint-обучение.

## Быстрый старт

```bash
pip install -e .

# лист (эксперт H=128)
python -u scripts/train.py --config configs/dbridge_leaf_s1.yaml

# слияние 3 экспертов (gen-1)
python -u scripts/train.py --config configs/dbridge_gen1_merged.yaml
python -u scripts/train.py --config configs/dbridge_gen1_joint.yaml

# гейт: weighted exact CE на 8 корпусах (tail-окна, 2×1024/корпус)
# см. протокол в ARCHITECTURE.md §4
```

Тесты: `python -X utf8 -m pytest tests -q` (79 passed).

## Формализация: что портировано в код

Ядро — `ShmidtS/primes` (486 теорем, 0 `sorry`/`axiom`, Lean 4.32).
Каждый модуль ниже исполняет формулу доказанной теоремы и помечен её именем.

| Модуль | Теорема | Что заменяет |
|---|---|---|
| `train/analytic_step.py` | `optimal_step_unconstrained` | η\* = ⟨g,d⟩/(L‖d‖²) вместо `lr` |
| `train/stochastic_safeqp.py` | R92 `minibatch_inner_concentration` | полные градиенты → минибатчи с явным ε |
| `train/batch_law.py` | `amgm_equality` + `amgm_uniqueness` | подбор батча: B\* = √(Bₙt₀/c) |
| `train/hedge.py` | `router_regret_bound`, `gating_tail_bound` | η = √(2lnK/T), min k по хвосту |
| `train/controller_policy.py` | `ratio_dominance` | поиск разбивок бюджета |
| `train/growth_law.py` | `capability_takeoff_counted` | счётчики роста |
| `train/insight_currency.py` | `insight_kl_descent`, `tldr_drift_null` | раздельные метрики CE/KL |
| `train/data_axis.py` | `diversity_floor_strict_pos` | симуляция пола → замкнутая сумма |

**Честные отрицательные результаты** (`.omc/attempts/`, не задеплоены):

- `analytic_step` — порт корректен (3 теоремы, 19 тестов), но на
  реальном A/B проиграл baseline на **+1.02 CE**. Глобальная проба L
  растёт монотонно (4.5 → 19 → 500), отслеживая самое крутое
  направление, а не среднее. Флаг `analytic_step` остаётся `False`.
- `batch_law` — измерено `t₀ = −8.8` мс, т.е. фиксированного overhead
  практически нет: `grad_accum_steps=1` во всех конфигах, амортизировать
  нечего. Согласуется с п.4 принципов ниже (шум исчезает на ~100 токенах).
- `LeanMachineLearning/LML` — учебник по вероятности (MarkovKernels,
  Martingales); применимых оптимизационных теорем нет.
  `lean-dojo/TorchLean` богат (`CROWN`/`Lyapunov`/`DirectedBackward`),
  но это верификация, а не ускорение.

## Стек

- Python 3.13, PyTorch (ROCm/HIP), AMD iGPU
- `src/hagi/` — модель, merge, обучение, инференс (см. ARCHITECTURE.md §1)
- `scripts/` — CLI: train/merge/gate-аудиты/генерация; `dsv4_*` —
  отдельный экспериментальный трек сжатия DeepSeek-V4 (не часть линии HAGI)
- `configs/` — YAML победной линии (dbridge_*) + тестовая фикстура
- `data/` — компактные корпуса (32768 словарь), mix.json

## Принципы

1. При равном бюджете вычислений три узких специалиста лучше
   одного широкого «с нуля»: каждый учится на своём распределении
   данных, а слияние объединяет их без потери качества (измерено:
   выигрыш ~1 нат против обучения с нуля тем же бюджетом).
2. Merge — центральный механизм; коммуникация поверх, дёшево.
3. Function-preserving инварианты обязательны (шаг-0 тождество).
4. Не подбирать гиперпараметры перебором — вычислять их из
   измерений самой модели. Примеры: размер низкоранговых добавок
   (LoRA-ранг) — из спектра сингулярных чисел (какая доля сигнала
   реально не покрыта); масштабы ветвей — из измеренной дисперсии
   активаций каждого слоя (BranchScale); размер батча — из
   измеренного шума градиента (на этой линии шум исчезает уже
   на ~100 токенах, значит большой батч ничего не даёт);
   «запускать ли дорогой эксперимент» — сначала дешёвый расчёт-
   сертификат на уже обученной модели, и только если он говорит
   «да», тратить GPU-часы.
5. Отбор кандидатов по diversity (Fisher/Jensen), не по standalone CE.
6. Иерархия каналов: специализация → merge → коммуникация → joint →
   low-rank residual.
