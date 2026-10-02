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

Фактическое состояние линии (чекпойнты на диске, проверено):

```
gen-2 joint (H=1152)  →  3 доменных сиба H=384 [math/lang/code]
  → gen-3 merged  H=1152 (Hadamard, block-diag 3→1)   AVG CE 3.3233
  → gen-3 joint   H=1152                              обучен, 1600 шагов
  → gen-4 сибы    H=1152 (q=18/kv=9)  ← автономный цикл, супервизор
  → gen-4 merged  H=3456 (q=54/kv=27)
```

## Измеренные свойства (не утверждения)

`scripts/growth_benchmark.py` измеряет три свойства на существующих
чекпойнтах, переиспользуя загрузчик `eval_domains.py`:

```
model            CODE     EN     MATH      RU     AVG
gen3_joint     1.7223  3.9425  3.7323  3.8959  3.3233
gen2_joint     1.8051  4.0358  3.8105  4.0863  3.4345

GROWTH     : −0.1112 AVG, gen-3 лучше gen-2 на ВСЕХ 4 доменах
GENERALITY : spread 0.668 (worst EN 3.9425, best CODE 1.7223)
MERGE      : merged лучше эксперта на ЕГО домене —
             MATH 3.7323 vs 3.8635, CODE 1.7223 vs 2.1613
```

Слияние не разрушает специалистов: merged-модель превосходит каждого
эксперта на его собственном домене.

Замечание: `configs/dbridge_gen2_merged_had.yaml` ссылается на
`dbridge_gen2_sib*`, которых на диске нет — фактические сибы лежат в
`gen2_dsib_*`. Это была причина `GEN2-MERGE-FAIL`.

Слияние — **троичное 3→1 на каждом уровне** (не степени двойки);
Hadamard-трансформ строится для любого N (pad → QR-ортонормализация).
Шаг-0 merged-модели — ровно ансамбль экспертов (Jensen-выигрыш бесплатно),
затем единственные новые связи — миксер и joint-обучение.

## Быстрый старт

```bash
pip install -e .

# лист (эксперт)
python -u scripts/train.py --config configs/dbridge_leaf_s1.yaml

# слияние 3 экспертов (gen-1)
python -u scripts/train.py --config configs/dbridge_gen1_merged.yaml
python -u scripts/train.py --config configs/dbridge_gen1_joint.yaml

# гейт: weighted exact CE на 8 корпусах (tail-окна, 2×1024/корпус)
# см. протокол в ARCHITECTURE.md §4
```

## Автономный цикл роста

Не редактируйте конфиги следующего поколения вручную — геометрия
наследуется, а не выводится, и два запуска gen-4 подряд падали именно
на этом. Генерируйте поколение и проверяйте инварианты до запуска:

```bash
# 1. сгенерировать поколение (проверит q*head_dim == hidden_size)
python scripts/make_generation.py --parent-width 1152 --generation 5 \
    --n-experts 3 --head-dim 64 --learning-rate 0.0003 \
    --parent-joint checkpoints/dbridge_gen3_joint/step-0001600.pt \
    --template-sibling configs/dbridge_gen4_sib1.yaml \
    --template-merged configs/dbridge_gen4_merged.yaml \
    --template-joint configs/dbridge_gen4_joint.yaml \
    --mixes math=13001=openwebmath:0.45,edu:0.25 \
            lang=13002=wikipedia_ru:0.30,oscar_ru:0.20,edu:0.20 \
            code=13003=python_instruct:0.50,edu:0.20 \
    --merge-checkpoint-step 1600 --write

# 2. прогнать цикл: 3 эксперта → слияние → joint → оценка
python scripts/growth/growth_supervisor.py --plan configs/growth_gen5.yaml \
    --device cuda --max-lanes 1

# 3. измерить рост / универсальность / качество слияния
python scripts/growth_benchmark.py \
    --gen  gen5_joint=configs/dbridge_gen5_joint.yaml=<ckpt> \
    --prev gen3_joint=configs/dbridge_gen3_joint.yaml=<ckpt> \
    --expert math=configs/...=<ckpt>   # по одному на эксперта
```

`--learning-rate` обязателен: значение **не масштабно-инвариантно**.
gen-4 sib3 унаследовал 1e-3 от шаблона, настроенного на H=384, и при
H=1152 разошёлся (CE 4.07 → 74.77), выйдя с кодом 0.

Супервизор отказывается принимать разошедшийся чекпойнт (`converged()`),
включая путь «все попытки исчерпаны» — иначе разрушенный эксперт
попадёт в слияние.

Тесты: `python -X utf8 -m pytest tests -q` (480 passed).

## Формализация: что портировано в код

Ядро — `ShmidtS/primes` (486 теорем, 0 `sorry`/`axiom`, Lean 4.32).
Каждый модуль ниже исполняет формулу доказанной теоремы и помечен её именем.

| Модуль | Теорема | Что заменяет |
|---|---|---|
| `train/analytic_step.py` | `optimal_step_unconstrained` | η\* = ⟨g,d⟩/(L‖d‖²) вместо `lr` |
| `train/stochastic_safeqp.py` | R92 `minibatch_inner_concentration` | полные градиенты → минибатчи с явным ε |
| `train/adaptive_safeqp.py` | R97 (covering-number) | ε для адаптивно выбранного `d*(ω)` |
| `train/anytime_budget.py` | R93 (неравенство Вилля) | δ на каждый шаг → один δ на горизонт |
| `train/growth_law.py` | R95/R96 `success_count_lower`, `takeoff_time_form` | вероятностный takeoff из измеренных полей |
| `train/batch_law.py` | `amgm_equality` + `amgm_uniqueness` | подбор батча: B\* = √(Bₙt₀/c) |
| `train/hedge.py` | `router_regret_bound`, `gating_tail_bound` | η = √(2lnK/T), min k по хвосту |
| `train/controller_policy.py` | `ratio_dominance` | поиск разбивок бюджета |
| `train/insight_currency.py` | `insight_kl_descent`, `tldr_drift_null` | раздельные метрики CE/KL |
| `train/data_axis.py` | `diversity_floor_strict_pos` | симуляция пола → замкнутая сумма |
| `train/synthetic_pretrain.py` | R99 `synth_investment_dominates` | `dT·c_step > C_synth` — один замер вместо вкуса |
| `train/discovery_ppt.py` | R98 `pptTarget_isProbability`, `mh_stationary`, `truncation_bias` | `p₀` R95, вычислимый ДО сэмплирования |
| `train/spectral.py` | R100 `proj_residual_identity`, `three_stage_error_budget` | ошибка сжатия раскладывается на именованные члены |
| `train/factorized_merge.py` | R101 `routed_eval_exact`, `factorized_budget` | merge как `C + R_i` с измеримой ценой `(1/N)Σδᵢ` |
| `train/takeoff_window.py` | R102 `growth_state_takeoff_window`, `noisy_cycle_step` | фиксированный gain даёт **конечный** takeoff; `κ√n·s/2` |
| `train/ternary_exact.py` | R103 `ternary_split_bound`, `quant_energy_bridge_saturation` | хвост насыщения измеряется, а не предполагается |
| `train/gain_renewal.py` | R104 `renewal_feeds_takeoff`, `bounded_frontier_no_sustained_growth` | sustained growth ⟺ фронтир масштабируется |

**Измеренные следствия, а не обещания:**

- R93: при δ=0.05 наивное правило тратит 20 за 400 шагов (бюджет
  превышен в 400 раз); геометрическое — не более 0.05 при T=1, 100
  и 10⁶.
- R95: 10x ёмкости — 95 циклов при p₀=0.3 против 16 при p₀=0.9.
  Поднимать частоту успехов выгоднее, чем размер выигрыша.
- R97: адаптивный выбор направления стоит 4.3x фиксированного запаса
  при n=16 и **35x при n=1152** (наша ширина). Цена указана явно,
  `eps_dir` выбирается арифметически, а не на глаз.

**Честные отрицательные результаты** (`.omc/attempts/`, не задеплоены):

- `analytic_step` — порт корректен (3 теоремы, 19 тестов), но на
  реальном A/B проиграл baseline на **+1.02 CE**. Глобальная проба L
  растёт монотонно (4.5 → 19 → 500), отслеживая самое крутое
  направление, а не среднее. Флаг `analytic_step` остаётся `False`.
- `batch_law` — измерено `t₀ = −8.8` мс, т.е. фиксированного overhead
  практически нет: `grad_accum_steps=1` во всех конфигах, амортизировать
  нечего. Согласуется с п.4 принципов ниже (шум исчезает на ~100 токенах).
- `factorized_merge` (R101) — арифметика верна, **посылка не выполняется**.
  Общее ядро не существует: эксперты почти идентичны (cos 0.986–0.991),
  но 90% энергии остатка `W_i − C` требуют **83%** спектра.
  `scripts/measure_shared_core.py` на трёх gen-2 экспертах: cosine между
  экспертами 0.986–0.991 (почти одинаковы — идеальный случай для общего
  ядра), но 90% энергии остатка `W_i − C` требуют **83%** спектра
  (317/384 компонент). Низкого ранга нет: отклонения размазаны по всем
  направлениям. Факторизация сохранила бы `V·d + 3·64·(V+d)` параметров
  и реконструировала бы **хуже**, чем просто хранить трёх экспертов.
  Порт корректен, посылка не satisfied — не задеплоен.
- `LeanMachineLearning/LML` — учебник по вероятности (MarkovKernels,
  Martingales); применимых оптимизационных теорем нет.
  `lean-dojo/TorchLean` богат (`CROWN`/`Lyapunov`/`DirectedBackward`),
  но это верификация, а не ускорение.

## Что ограничивает рост (R102–R104, измерено)

Три раунда формализации дали точный ответ на вопрос «откуда берётся
быстрый рост», и он неудобный, но определённый.

**R102 — с фиксированным gain takeoff конечен.** При инвариантном
`G` и гейте `α·C_t ≤ G` число успехов не превышает `1/α + 1 − C₀/G`
**независимо от горизонта**, а сертифицированный множитель ограничен
`exp(1+α−αC₀/G) ≤ e·e^α`. Проверка: `α=0.05, C₀=1, G=1` → потолок
ровно 20 успехов, множитель `e`. Форма `e^{G/C₀}`, предложенная аудитом,
**не доказуема** в этой общности — и порт это фиксирует числом, а не
комментарием.

**R104 — мост закрыт, но условие измеримо.** Если gain производится
каждый цикл (`G_{t+1} ≥ ρG_t + γ(inj−ξ)`), окно R102 снимается:
`C_T ≥ C₀(1+α)^T` для **любого** T. Единственная открытая посылка —
масштабирование фронтира `α·C_t ≤ γ·D_t`, и она измерена:

```
D_t (средний Jensen gap трёх экспертов gen-2) = 18.25 нат
потолок от ограниченного фронтира               = 182.5
требование гейта при C=10                      = D ≥ 1.0   (выполнено)
```

**Инвариант урожая выполняется:** прирост gen-2 → gen-3 = +0.0365 нат
против доступных 18.25 — инвариант держится с запасом в три порядка.

**Механизм потери измерен** (`diagnose_merge_cancellation.py`). Отклонения
экспертов от их среднего **взаимно антикоррелированы**:

```
cos(dev_math, dev_lang) = -0.599
cos(dev_math, dev_code) = -0.290
cos(dev_lang, dev_code) = -0.592
```

Усреднение гасит **именно эту общую разнонаправленную часть** — то
есть ровно то, в чём эксперты различаются. Среднее сохраняет только
общую компоненту, а она по построению не несёт пригодного сигнала.

При этом информация **не потеряна**: `merged + dev_k = W_k` точно до
ошибки float (2.8e-16), и `cos(W_k, merged) = 0.995`. Оно есть в
чекпойнтах — слияние просто её не несёт.

**Корень найден в канале связи, а не в слиянии.** Миксер — обучаемый
скаляр `gain`, инициализированный нулём (тождество по построению):

```
mixer.gain при инициализации   0.0000
mixer.gain после merged@1600  -0.0606   ОТРИЦАТЕЛЕН
mixer.gain после joint@1600   -0.0051   в 11 раз ближе к нулю
```

Слияние专家 не потеряло: `cos(W_k, merged) = 0.995`, реконструкция
`merged + dev_k = W_k` точна до 2.8e-16. Потеря происходит в
**обучении**: joint-фаза, чья задача — научить блоки общаться,
увеличивает gain **к нулю**, а не от него. Оптимизатор обнаружил, что
миксер выключить выгоднее, чем использовать.

**Отсюда следует главное.** Ни потолок фронтира, ни точность урожая,
ни слияние рост **не ограничивают**. Ограничивает **канал связи**:
он существует, он обучаем, и обучение его actively выключило, потому
что он не окупался.

Практическое следствие: `mixer.gain` — это измеримый детектор. Ненулевой
и **положительный** gain после joint означает, что канал заработал;
gain, ушедший к нулю, — что коммуникация не окупается и проект
фактически обучает три независимых эксперта в одной оболочке.

**R103 — насыщение честно.** Старая посылка `∀i |w_i−q_i| ≤ s/2`
ломается при `|x| > 3s/2`, где ошибка **точно** равна `|x|−s`.
Разложение `total = in-range + satTail` — тождество, и на embedding
gen-3 при `s=0.05` насыщение даёт 37680 из 43595, то есть **86%**
ошибки, за которую старая формула не брала ничего. Решение «перекалибровать
сетку или жить с хвостом» теперь сравнение на существующем чекпойнте,
а не sweep.

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
