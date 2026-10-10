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

## Автономный цикл роста (текущий)

Конфиги поколений НЕ живут в `configs/` и не редактируются руками —
их генерирует SSOT-билдер `scripts/growth/gen_configs.py` (один шаблон
+ пер-доменные overrides: миксы, регуляризация, бюджеты шагов, схема
seed'ов) прямо в `checkpoints/gen<N>/configs/`. `configs/` сокращён
до 8 живых файлов (cycle, dbridge, latent-шаблоны, фикстура).

```bash
# 1. сгенерировать конфиги поколения (в checkpoints/gen41/configs/)
python scripts/growth/gen_configs.py --gen 41     --root checkpoints/gen39_root_ft/best.pt --seed-base 46

# 2. одно поколение: 6 листьев → latent merge → root ft → cooldown →
#    §9 certified A/B (останов, если не сертифицировано)
python -X utf8 -u scripts/growth/growth_cycle.py --gen 41     --root checkpoints/gen39_root_ft/best.pt     --leaf-config "checkpoints/gen41/configs/leaf_{d}.yaml"     --ft-config checkpoints/gen41/configs/root_ft.yaml     --common-config checkpoints/gen41/configs/root_ft.yaml     --domains math code ru enedu enweb enwiki

# 3. мульти-поколенно (сам делает п.1-2 и останавливается на отказе гейта)
python -X utf8 scripts/growth/run_ladder.py --from-gen 41
```

`--learning-rate` обязателен: значение **не масштабно-инвариантно**.
gen-4 sib3 унаследовал 1e-3 от шаблона, настроенного на H=384, и при
H=1152 разошёлся (CE 4.07 → 74.77), выйдя с кодом 0.

Супервизор отказывается принимать разошедшийся чекпойнт (`converged()`),
включая путь «все попытки исчерпаны» — иначе разрушенный эксперт
попадёт в слияние.

Тесты: `python -X utf8 -m pytest tests -q` (602 passed).

## Формализация: что портировано в код

Ядро — `ShmidtS/primes` (966 теорем, 0 `sorry`/`axiom`, Lean 4.32;
SSOT prescriptions — `C:/primes/ALGORITHMS.md` §0–§23).
Каждый модуль ниже исполняет формулу доказанной теоремы и помечен её именем.

### Certified-контроллер (§0/§1/§9/§17, рефакторинг 2026-10-08)

- `train/certified_controller.py` — решающее ядро цикла роста:
  argmax Γ_i/K_i (`ratio_dominance` R86), сертифицированный A/B
  `certified_ab` (принимать B ⟺ CE_A−CE_B > 2ε при
  n ≥ log(2/δ)/(2ε²)), `stop_condition` (`liveness_two_axis` R80),
  `leak_gate` (`distill_leak_gate` R134: δ ≥ c — стоп рекурсии),
  `exhausted_check` (ρⁿ·D₀ < ε/γ — менять корпус), enum фаз §23.
  `growth_supervisor` решает через `certified_ab` (UNDECIDED при
  n < порога → фоллбэк not-worse, задокументирован); standalone-CE
  скрининг — только дешёвый пре-фильтр (`selection_hurts` §1).
- `inference/hedge_router.py` — §6: Hedge-веса η=√(2lnK/T) + top-k
  tail-budget (SKIP: MergedHAGI не экспонирует per-leaf logits).
- `growth/measure_fiber_geometry.py` — §20 решающий эксперимент:
  per-layer спектр/d90 до и после merge.
- Гейты §10/§13 (ignition/bifurcation, saturation, takeoff_window)
  вызываются супервизором после каждой лейна — советуют,
  вердикт решает `certified_ab`.

### Density ladder (обратный рост плотности, 2026-10-08)

Рост ширины упёрся в VRAM на gen7 (joint 10368 = 2.6B, впритык).
Вместо gen8 3×-merge (OOM) — ОБРАТНАЯ лестница §17/R134:
ширина только убывает, плотность растёт на каждом уровне.

```
gen7_joint 10368 → дистилл → студент 3456 (init gen6_joint)   [L1]
L1-дистиллят      → дистилл → студент 1152 (init gen3_joint)   [L2]
L2-дистиллят      → дистилл → студент 384  (fresh, q6/kv3)     [L3]
```

- `scripts/density_ladder.sh` — идемпотентный конвейер (auto-resume,
  скип завершённых стейджей); стартует только при готовом
  gen7_joint/best.pt (учитель L1).
- Вердикт каждого уровня — сертифицированный 2ε против исторического
  joint той же ширины (L1 vs gen6_joint, L2 vs gen3_joint; L3 —
  рекорд). Память строго убывает по ходу лестницы — OOM исключён
  конструктивно.
- Кросс-ширинный дистилл поддержан рантаймом (`_build_distill_teacher`
  строит учителя из его собственного конфига, H=10368 → студент 3456).


### EmbeddingGemma-2 (вердикт 2026-10-08)

`google/embeddinggemma-2` (740M, мультимодальная, 768-dim, Apache 2.0)
  НЕ применима как замена таблицы токенных эмбеддингов HAGI:
  740M ≫ бюджета листа, другая роль (retrieval-модель), сломала бы
  step-0 merge-инварианты и тернарное сжатие. Применимое место —
  офлайн data-axis: семантическая кластеризация корпусов, подбор
  свежих данных для инъекции (§4), кросс-модальный gap (§21).

| Модуль | Теорема | Что заменяет |
|---|---|---|
| `train/analytic_step.py` | `optimal_step_unconstrained` | η\* = ⟨g,d⟩/(L‖d‖²) вместо `lr` |
| `train/hedge.py` | `router_regret_bound`, `gating_tail_bound` | η = √(2lnK/T), min k по хвосту |
| `train/spectral.py` | R100 `proj_residual_identity`, `three_stage_error_budget` | ошибка сжатия раскладывается на именованные члены |
| `train/takeoff_window.py` | R102 `growth_state_takeoff_window`, `noisy_cycle_step` | фиксированный gain даёт **конечный** takeoff; `κ√n·s/2` |
| `train/gain_renewal.py` | R104 `renewal_feeds_takeoff`, `bounded_frontier_no_sustained_growth` | sustained growth ⟺ фронтир масштабируется |
| `train/safeqp_step.py` | R105 `safeqp_eta_max`, R106 `poe_logZ_second_order` | **производный** безопасный LR; ошибка PoE-пулинга ≤ `R²/8` |

(Порты R92–R94, R95/R96, R103 — stochastic_safeqp, anytime_budget,
growth_law, data_axis, ternary_exact, routing_optimal — перенесены в
`_raw/attic/`, см. ниже.)

Модули с отрицательным результатом и теоретические порты без
продакшн-потребителя (adaptive_safeqp, batch_law, controller_policy,
insight_currency, synthetic_pretrain, discovery_ppt, factorized_merge,
growth_potential, universality, thermo_layer, streaming_gpm, а также
anytime_budget/stochastic_safeqp — сертифицированный anytime-гейт
живёт в growth_supervisor.anytime_margin, data_axis — совет
liveness_data_axis в супервизоре, frontier_cone, growth_law,
routing_optimal, safeqp_gpm, safeqp_pl, ternary_exact, trust_region,
orchestrator/external_eval, orchestrator/mechanism_gate)
перенесены в `_raw/attic/` вместе со своими тестами — история в git.

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
- `batch_law` (в attic) — измерено `t₀ = −8.8` мс, т.е. фиксированного overhead
  практически нет: `grad_accum_steps=1` во всех конфигах, амортизировать
  нечего. Согласуется с п.4 принципов ниже (шум исчезает на ~100 токенах).
- `factorized_merge` (R101, в attic) — арифметика верна, **посылка не выполняется**.
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

## Деградация gen-4: это был `data.seed`

Диагностика заняла четыре раунда, потому что первая гипотеза была
неверной. Первоначально винили смесь `lang` — этого не подтверждает
дизайн 2×2, который уже лежал в `logs/`, но не был собран.

Правильный ответ, измеренный на существующих прогонах:

| арм | `data.seed` | 0–800 | 800–1300 | tail |
|---|---|---|---|---|
| gen4_sib1 (падает) | **12901** | 3.617 | **5.632** | 5.824 |
| gen4_sib2 (падает) | **12902** | 3.813 | **3.985** | 4.190 |
| gen4_sib3 (убит) | **12903** | 3.326 | 3.718 | 3.336 |
| seedtest (тот же конфиг, seed 13701) | 13701 | 3.645 | 3.562 | **3.590** |
| swap_sib1 / swap_sib2 | 13701 / 13702 | — | — | 3.78 / 3.60 |
| bisect_ruonly / enonly | — | — | — | 3.48 / 3.29 |

Разделитель — семейство `129xx` против всего остального. Гипотезы о
механизме проверялись и **отвергались измерением**: перекос смеси
(≤1.0% отклонения, `measure_mix_skew.py`), стартовая позиция (все
читают с курсора 0), покрытие воркерами (расхождение ≤0.0027),
порядок чередования (run 16 против 10 при mean 1.48 — слишком мало для
CE 3.6 → 7.5).

**Горизонт опровергнут:** тот же здоровый конфиг на 1600 шагов дошёл до
1410 без деградации (3.5558 → 3.4173).

**Лечение применено:** gen-4 перегенерирован на seed 13801–13804.
Тот же math-эксперт, что деградировал до tail 5.82 на 12901, на 13801
даёт **tail 3.60** при max CE 4.32.

## Сколько там места (R105, применённый к нашим числам)

`safeqp_eta_max` даёт окно безопасного шага из измеренных величин.
На доменах с бюджетом регрессии 0.05 и кривизной L=10:

```
производное окно eta_max = 0.0120
настроенный lr            = 0.0003
                              → запас в 40 раз
```

То есть **размер шага не ограничивает рост** — безопасного места в
40 раз больше, чем берёт настроенный lr. Но измеренный
`mixer.gain = −0.005` показывает, что оптимизатор не использует ни
части этого запаса.

Вывод, который это меняет: узкое место не «шаг слишком мелкий», а
**отсутствие механизма, который хочет этим местом воспользоваться**.
Увеличение lr ничего не даст — уже проверено: gen-4 sib3 на lr=1e-3
разошёлся до 74.77.

## Насколько не хватает (арифметика, не оценочные слова)

Гейт R104: `α·C ≤ γ·D`. Подставляя измеренные величины
(`D = 18.25` нат, `C ≈ 3.3` — наша метрика CE, `α = 0.1`):

```
требуемая γ   = α·C/D = 0.1·3.3/18.25 = 0.0181
измеренная γ  = G/D   = 0.0365/18.25 = 0.0020
дефицит                              в 9 раз

что даёт joint-стадия:  0.0365 нат
что требует гейт:       0.33   нат
                        → 11% от потребного
```

То есть ни потолок фронтира (18.25 нат — много), ни точность урожая
не ограничивают рост. Ограничивает **ставка**, с которой оператор
превращает разногласие в прирост, и она в девять раз ниже нужной.

## Проверка: окупается ли канал связи?

`mixer.gain` уехал к нулю (см. выше). Две развилки это различают, и
`merge.freeze_experts` даёт чистый A/B — опция была объявлена и
задокументирована, но **никогда не реализована** (`grep` находил
только поле и docstring); теперь работает.

```bash
# обучаются только 7 тензоров миксера из 99.6M (0.22%)
python -u scripts/train.py --config configs/mixonly_gen3.yaml
```

- gain **растёт** → канал окупается, когда он единственная свобода;
  проблема в том, что в joint свобода есть и у экспертов;
- gain **падает** → канал не окупается сам по себе, нужен другой
  оператор (R101), а не увеличение lr.

Именно это требование делает R104 содержательным: `gain` должен
**производиться** каждый цикл, а не наследоваться. Параметр, который
оптимизатор может обнулить, — это не производство прироста.

## Три бага, найденные при разбирательстве цикла

Все три были **молчаливыми**: без ошибки, с кодом выхода 0.

**1. `--resume` был заглушён `init_from`.** В `train.py` ветка
инициализации проверялась первой, а `init_from` задан в каждом
sibling-конфиге (наследование общего ядра родителя). Ветка resume
была `elif` и **никогда не выполнялась**. Супервизор перезапускает с
чекпойнта на диске — то есть весь механизм восстановления молча
уничтожал все обученные шаги и начинал с родителя. Наблюдалось:
чекпойнт на шаге 1300, супервизор корректно передаёт `--resume`, в логе
`initialized weights from ... (fresh optimizer, step 0)`.

**2. `merge.freeze_experts` не была реализована.** `grep -rn
freeze_experts src/hagi` находил ровно два совпадения: поле
датакласа и его docstring. Конфиг, включавший её, обучал всё.

**3. `proc.wait()` зависал на Windows.** После смерти дочернего
процесса супервизор оставался в `State: S` с заблокированным
`wait()` и ничего не писал в лог — цикл просто останавливался, что
выглядит как «простаивает», а не «сломан».

Все три закрыты тестами. Первый — четырьмя тестами, запускающими
настоящий CLI в `--dry-run`, чтобы порядок веток не мог regress'нуть.

## Стек

- Python 3.13, PyTorch (ROCm/HIP), AMD iGPU
- `src/hagi/model/` — модель и слияние: `model`, `merge` (R242-R244
  latent merge поверх — `latent_merge`), `attention`, `rope`,
  `ternary` (b1.58), `cortex`, `formal`, `multimodal`, `adaptive`,
  `scratch_blocknorm` и вспомогательные блоки
- `src/hagi/train/` — живой рантайм: `loop`, `optim`, `checkpoint`,
  `saturation`, `merge_price`, `distill`/`distill_transfer`/
  `distill_recursion`/`recursive_distill`, `disagreement_distill` (§17),
  `safeqp_*` (step/controller), `analytic_step`,
  `certified_controller`, `hedge`, `spectral`,
  `self_improve`, `self_development` и др. (полный список — в дереве;
  снятые порты — в `_raw/attic/`)
- `src/hagi/orchestrator/` — real_cycle/recursive/gates (потребители:
  growth_supervisor, merge в `model/merge.py`)
- `src/hagi/inference/` — `generate`, `hedge_router`
- `scripts/` — CLI: train/merge/gate-аудиты/генерация; `growth/` —
  F3-конвейер и supervisor
- `configs/` — YAML победной линии + тестовая фикстура
- `data/` — компактные корпуса (32768 словарь), mix.json
- `_raw/attic/` — снятые с рантайма модули и их тесты (git-история
  сохранена; на диске, вне git)

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

## Growth ladder (autonomous, certified)

Same-origin leaves -> gated latent merge (R242/R243, merge_gate §3)
-> joint fine-tune -> WSqD cooldown (R254) -> §9 certified accept.
Drivers: `scripts/growth/growth_cycle.py` (one generation),
`run_ladder.py` (multi-generation, halts on the first
non-certified generation). Ladder log:
`checkpoints/growth_ladder.jsonl`.

| stage | AVG exact CE (common protocol) |
|---|---|
| gen-1 leaves (mean) | 7.74 |
| gen-1 ft root | 6.690 |
| gen-3 ft | 5.911 |
| gen-6 | 5.663 |
| gen-11 + halt | 5.415 |
| + WSqD cooldown | 5.385 |
| gen-13 (4 experts) | 5.319 |
| gen-16 (6 experts, +0.016 < margin) | 5.304* |
| gen-28..32 (fresh seeds + WSqD) | 4.9147 |
| gen-34 (r64, broken cross-rank load) | 5.093→fixed — NOT certified |
| gen-35 (nested r64, +0.0186 < margin) | 4.8961* |
| gen-37 (nested r64, seeds 19xxx) | **4.8346** (accepted, +0.080) |
| gen-38 (seeds 21xxx, −0.007) | rejected — incumbent gen-37 |

*rejected by the 0.02 §9 margin — the honest incumbent is gen-37
(4.8346).

Rank-growth seam fixed (2026-10-10): lora_A bases are now NESTED
(`_nested_lora_basis`: r32 bit-compatible with the whole line,
A(64)[:, :32] == A(32)), cross-rank load compensates alpha/r —
rank growth is function-preserving (regression-tested; the gen-34
incident: non-nested bases silently disabled the incumbent's adapters
and fabricated a +0.179 delta). r96 merge-gate REJECTED (theory §3:
twoGap does not pay the factorization price) — r64 is the working
rank. Single-generation deltas under ~2× margin are seed noise
(gen-35 +0.019 vs gen-37 +0.080 at identical levers).

Floor analysis (R255): late-generation disagreement is
near-orthogonal noise (mean|cos| ~ 0.1); averaging denoises 1/N,
the joint-ft contraction converts the residue, and the floor
r/(1-kappa) is reached. Batch-64 leaves NaN on the iGPU — the
noise floor is hardware-bound. Next architectural lever per
theory: latent-rank growth (R242 sub-1-BPW) — wider H is
forbidden.
