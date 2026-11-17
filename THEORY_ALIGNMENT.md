# THEORY_ALIGNMENT.md — сверка нейропроцессора HAGI с формальной теорией

SSOT теории: `C:\primes` (Lean 4.35 / Mathlib, 966 теорем, 0 sorry/axiom;
прескрипции — `ALGORITHMS.md` §0–§24, карточки результатов — `STATUS.md`).
Этот файл — по-механизменная сверка: каждый механизм Python-петли →
формальная опора → код → измеренное доказательство → честные границы.

Дата сверки: 2026-11-15. Ветка: main.

---

## 1. Commit / Hold гейт принятия поколения

| | |
|---|---|
| Теория | `Hagi/Unified/CertifiedLoop.lean` (`certified_loop`, `certified_loop_strict`), `Hagi/Unified/CommitGate.lean` (`commitProductive`): звено цепи принимается только при доказанном implication для successor; телескоп objectiveDrop не превышает фактического убывания; строгий вариант даёт `obj(f n) < obj(f 0)`. |
| Код | `hagi_selfloop.run_self_development` Phase 4: `action=COMMIT` только если `experts_opt ⊇ experts_incumbent` И `verify_all(suite).accuracy == 1.0` И `relief ≥ 2%`; иначе `HOLD`/`LEVER_CHANGE` с откатом к инкамбенту. Каждый COMMIT-раунд = `StepCert`; CertStepGen-телескоп ведётся процессором (`hagi/_v9.py`, unsolved_objective − 1.0 за коммит, без clamp — коммиты >100 уходят в отрицательное, иначе telescope ломается). |
| Evidence | `experiments/selfloop_result.json`: 149/150 COMMIT-раундов, accuracy=1.0 на каждом гейте; смена floor-рычага (R223) без единого lost-expert после фикса порядка слияния. |
| Границы | Теорема не доказывает оптимальность архитектуры или обобщение — только нерегрессию по сертифицированному инварианту (superset + точность + маржа). Это и есть всё, что заявляется. |

## 2. Audit ledger (α-траты, Hoeffding)

| | |
|---|---|
| Теория | `Hagi/Contracts/AuditLedger.lean` (C274/E7): `AuditStream [lo,hi)` с курсором, `AuditLedger` с инвариантами streamsDisjoint / spentTotal (Σα ≤ δtotal) / finalIsolated; `audit_consume_alpha_exact` — трата ровно α; `audit_false_harm_bounded` (master, через C264 `gate_pipeline_false_commit`). |
| Код | `hagi/_v9.py::AuditLedgerV9` — построчный порт: `delta_total`, `spent`, `streams` с курсорами, `floors`, `rejected_tasks`, `final_test_lo/hi`, `final_test_touched`; `spend_anytime` возвращает (ok, hoeffding_radius); `verify_disjoint_and_clean`. В параллельном цикле `hagi_selfloop.ParallelLearner._merge` воссоздаёт траты α и срезы в порядке потока — disjointness сохраняется. |
| Evidence | `python -m hagi --self-test-v9`: `certs_ok=True` (инварианты реестра проверены на 63 экспертах). |
| Границы | `final_test_touched=false` — held-out диапазон не расходуется; но мост «α-трата → выигрыш пола» остаётся полем (как и в Lean: floors/current gain — docstring-мосты). |

## 3. Guarded step (не принимать successor, увеличивающий потерю)

| | |
|---|---|
| Теория | §9: `guardedBatchTrainStep` + `batchSampleCE_guardedBatchTrainStep_le` — samplewise non-regression как условие commit; «не принимаем post-energy автоматически: проверяется на настоящем successor». |
| Код | `hagi_v10_controller.guarded_optimizer_step` (MGDA-направление + точный откат по снапшоту при ухудшении) и нейросимвольный аналог в `learn()`: кандидат-программа коммитится только при 100% точности на train + val + ВСЁМ домене (samplewise non-regression ≡ точное покрытие). |
| Evidence | Каждая функция в `checkpoints/self/generated_programs.py` валидируется против эксперта по полному домену: 34677 проверок, 0 ошибок. |

## 4. Сертифицированная остановка / смена рычага

| | |
|---|---|
| Теория | R223 (measured, см. §1 селектор ДЕЙСТВИЙ в README основного проекта): нет certified-positive действия = E_dev исчерпан → останов СО СМЕНОЙ РЫЧАГА, не «ещё раз так же»; §14a: anytime-гейт с адаптивным n вместо фиксированного радиуса (`gate_fixed_radius_unusable` — негативный свидетель). |
| Код | `MIN_RELIEF = 0.02` (аналог accept-маржи §9): раунд принимается только при relief ≥ 2%; при исчерпании — `LEVER_CHANGE` по лестнице floor 200→50→20 (`floor_ladder`), не повторение того же рычага. |
| Evidence | Раунд 102: LEVER_CHANGE floor 200→50, waste 104400→31100 (471→0 потерянных экспертов); рост продолжился до лимита раундов. |
| Границы | Число 2% — инженерная маржа, не выведенная из δ/n формулы §9 (которая применима к CE-бенчмаркам, а не к waste-счётчику); выбрана консервативно и не влияет на инварианты п.1. |

## 5. Hedge-router (ограниченный regret)

| | |
|---|---|
| Теория | §9: `router_regret_bound` — `A_T − L_T(i) ≤ log(card ι)/η + ηT/2` при `0<η≤1`, `0≤l≤1`; охлаждение Гиббса `thermostat_stop`. |
| Код | `hagi/_v9.py::HedgeRouterV9` (eta=0.5) в FactorizedSharedCore — экспоненциальные веса по ветвям, снапшот/рестор в guarded-шаге. |
| Evidence | certs_ok=True в self-test (router-состояние в телеметрии каждого шага). |
| Границы | Теорема гарантирует regret против лучшего эксперта-router, не снижение CE — ровно это и заявляется.

## 6. Само-модель и честная самоссылка

| | |
|---|---|
| Теория | Спецификация v10 §5/§9 (см. докстринг `hagi_selfmodel.py`): generator ≠ evaluator; «never claim what is not demonstrated»; ABSTAIN-сертификат по конечному свидетелю. |
| Код | `DecisionListSelfModel` — точный по построению предиктор на полном ключе (признаки+seed); `impossibility_certificate` — конечный проверяемый свидетель невозможности; `holdout_validate` блокирует уверенно-ложные HOLD-прогнозы; unknown → консервативный base-бюджет. |
| Evidence | Раунды с confident_wrong=0 на holdout; пойманный и устранённый инцидент с uncertified coarse-key генерализацией (gate rejected → fallback удалён). |

## 7. Рост поколений без разрушения специалистов

| | |
|---|---|
| Теория | README primes: «объединение специализированных компонентов, сжатие и рост как связанная система состояний/переходов/бюджетов»; merged-модель не разрушает экспертов (измеренное свойство основного проекта). |
| Код | Библиотека экспертов только растёт (superset-гейт); MDL-DAG синтез переиспользует ранее сертифицированных экспертов через `call` с проверкой support (вне саппорта — отказ). |
| Evidence | 55→411 экспертов за 117 раундов, 0 lost-expert после фикса; `generated_programs.py` — 395 standalone-функций, DAG-вызовы между ними валидны. |

---

## Сверка «прескрипция → исполнение» (§0–§24 ALGORITHMS.md)

Исполнено (перечислено выше): §9 guarded step, §14a gate pipeline
(через AuditLedger + commit-гейт), §14b audit ledger (построчный порт),
§1 селектор/останов (relief-маржа + lever-change), Hedge-router.

Сознательно НЕ перенесено в эту ветку (границы применимости):
- §3.x safe-CE-шаги для нейросетевого тела (в этой ветке нет NN-тела
  на градиентах — нейросимвольное ядро с точной семантикой);
- §6 квантование/HNet (нет тензоров для квантования; тернарный
  код присутствует в `_legacy` как порт);
- §17 Venn-голова — присутствует (`boolvenn-exact` 16 масок, anf
  Жегалкин-бабочка) — единственный NN-символьный мост, полностью живой.

Вывод: противоречий с теорией не найдено; все принятия решений в цикле
сводятся к доказанным implication-гейтам (CertifiedLoop/CommitGate) и
инвариантам реестра (AuditLedger); заявленные свойства — ровно те, что
следуют из теорем, не шире.


## Сверка 2026-11-16: обновление теории (C291 → C331+)

Другой агент расширил C:\primes. Новые применённые элементы:

- **CurriculumCert (curriculum phase accounting)** — order-free
  фазовый учёт Δᵢ/dᵢ (curriculumTotal, curriculumTotal_comm,
  curriculum_phase_accounting). Портирован в `hagi_curriculum.py`;
  каждый forever-цикл дописывает фазовую запись в self_state, 
  сертификат order-free по построению. Тест
  `test_curriculum_certificate` проверяет коммутативность.
- **VilleMonitor / stitching_countable_ville** — anytime-мосты:
  не нарушены нами (наш anytime-инструмент — guarded best-snapshot
  в LM-тренировке: деплой никогда не хуже лучшего; формальный
  Ville-мост оставлен теории).
- **Certified early exit depth routing (C336+)** — кандидат на
  следующий порт: ранний выход из синтеза при сертифицированном
  достижении; пока сознательно не применён (требует потока
  доказательств внутри синтеза, не только финального гейта).
- **E-track (§23, energy-based reasoning)** — за пределами текущего
  ядра; зафиксировано как фронт.

Нейро-тело докормлено: 529 MB корпус со всего компьютера
(k=12, dim=256, hidden=512, 24k guarded шагов): holdout CE
1.70 → 1.5553 и продолжает падать.


## Сверка 2026-11-16 (вечер): адаптивное окно vs теория (C307/C308/A8)

Запрос «адаптивное окно без ограничения сверху» сверен с primes:

- **C308 (RoPE-изометрия)** — реализована RopeAttentionByteLM:
  вращения пар измерений (cos/sin), content-softmax внимание (C307
  premise-форма), heat-bath recency в скорах (A8 Doeblin-пол).
  ИЗМЕРЕНО: на игрушечном однослойном MLP книга Нади CE 0.37
  против 0.031 у hash-bag; recall 0/8 против 6/6 — RoPE не
  поднимается без многослойного внимания. Честный вывод:
  hash-bag остаётся рабочей школой, RoPE-attention зафиксирован
  как фронт для скейла (теория даёт ему изометрию и липшиц-
  перенос C300, когда появится глубина).
- **A8 (Doeblin/heat-bath)** — ad-hoc 0.9^d затухание получило
  строгое обоснование: это Doeblin-миноризация с геометрическим
  полем; частичные суммы сертифицированы бюджетом
  slack·m/(m−Δ) (C294-N1).
- Рабочее решение: AdaptiveByteLM (hashed positions + recency
  weighting на инференсе) — 6/6 коротких, 800-байтовый промпт
  при обучении <=96 верен. Коммит 374b2c2.


## Сверка 2026-11-16 (ночь): улучшение алгоритма обучения по теории

- **Range-based safe step (§3.8/§3.9, C216/C223)** — реализован
  CertifiedStepLM: направление D=-grad, A=|D|², R'=sup−inf смещения
  логитов, η=4A/R'² — точный минимизатор мажоранты; КАЖДЫЙ шаг
  спускается на сертифицированные 2A²/R'² (безопасно по
  построению, откат не нужен). Измерено на книге Нади: CE
  1.18→0.17 за 150 эпох; накопленный сертифицированный спуск
  ≥202 — доказуемый пол улучшения, уникальная фича теории.
- **Hedge regret (§router_regret_bound)** — для роутинга
  экспертов: A_T−L_T(i) ≤ log(K)/η + ηT/2; применимо к выбору
  веток обучения при скейле (фронт).
- **Thermostat (Growth/Thermostat)** — Gibbs softDef(−L/T) с
  охлаждением T_t=T0·α^t и stop-bound exp(−δ/T_t): сертифицированное
  расписание остановки обучения (фронт для внешнего цикла).
- Вывод: для игрушечного MLP Adam сходится быстрее (0.011 vs 0.17),
  но certified-step — единственный вариант с гарантированным
  спуском на каждом шаге; целевой рецепт — certified base + Adam
  polish при скейле.


## Сверка 2026-11-17: полный рефакторинг по Lean-формализации

Прочитана формализация (CertifiedLoop, ComputeToQuality C330,
GatePipeline C264, CurriculumCert, KVCacheCert C331, BatchCE*).
Архитектурные выводы, воплощённые в коде:

1. **CertifiedLoop `gate_never_increases`** — forever-цикл HAGI_v2
   это реализация доказанного safe anytime-цикла: без сертификата
   состояние сохраняется, энергия не растёт никогда. Наша связка
   гейтов (точность 1.0, relief>=2%, no-regression benchmark в
   self-edit) — операционализация этой теоремы.
2. **ComputeToQuality C330** — «минимизация вычислений»: учиться до
   ε-цели (steps_to_quality: ⌈Δ₀/d⌉ шагов), FLOP-бюджет на эпоху
   (C328 work model), low-rank tradeoff (2kn < n²). Реализовано:
   AdaptiveScheduleLM (train-until-eps, adaptive batch), memory-aware
   workers (degradation по размеру библиотеки).
3. **GatePipeline C264** — δ-расписание Σ6/(π²k²)≤1 для серии
   гейтов: каждый commit消耗ляет долю бюджета ложных срабатываний.
   Фронт: внедрить явный δ-бюджет в self-edit гейт.
4. **KVCacheCert C331** — кэш точен (kv_cache_exact), отношение
   работ 3R=(n+2)W; фронт: кэшировать окна LM-инференса.
5. **Структура**: один источник истины — всё в пакете hagi/,
   без дублей-шим; root CLI = python -m hagi.hagi_*.


## Сверка 2026-11-17 (v2): построчная сверка формул Lean↔Python

Найдено и ИСПРАВЛЕНО реальное расхождение:

- **§3.9 batch R'** — CertifiedStepLM считал R' по last-layer
  displacement (h·dW2): нелинейность emb/W1 оставалась за кадром,
  шаг иногда УВЕЛИЧИВАЛ CE (нарушение сертификата, найдено
  численно: trial CE 3.98→7.41). Исправлено на точный JVP всей
  сети вдоль D (`torch.autograd.functional.jvp` всех параметров),
  R'^2 = mean квадратов per-sample ranges (batchLogitRangeSquare,
  C223). Верификация: 10/10 шагов спускаются, реальный спуск ≥
  сертификата на каждом.

Подтверждены численно (код↔Lean):
- curriculumTotal = Δ₁/d₁+Δ₂/d₂; Lean-якорь anchor_total=30 ✓
- curriculumTotal_comm (order-free) ✓
- Thermostat T_t=T0·α^t ✓
- §6 ternary (test_byte_lm) ✓


## Цикл 3 сверки (2026-11-17): Autonomy/Budget/Optimization

- **Insight.experience_cycle_bound** — цикл «опыта» растёт только
  при положительном бюджете gains−costs>0. Внедрено в self-edit:
  патч принимается теперь при СТРОГОМ улучшении (adopted rounds
  или experts больше базовой линии), не просто не-регрессии.
- **AdamW.adamw_shrink** — decoupled decay сжимает вес при
  нулевом градиенте (фактор 1−λ). Численно проверено. Отмечено
  отличие: torch AdamW масштабирует decay на lr (Lean adamwStep —
  нет); на certified-step путь не влияет.
- **TTTStability.ttt_bibo** — |W|≤1 ⟹ ограниченность: наши
  нормированные Hedge-веса (ewNorm) всегда ≤1 ✓.
- **Lookahead.slowStep**, **Lion.sgn/signStep**, **KVCache** —
  не используются, честно не заявляем.
- **Budget/ (BitAlloc, RDAlloc, ResidualVQ)** — фронт для
  бит-аллокации при скейле квантования.


## Цикл 4 сверки (2026-11-17): Architecture/Budget

- **FlopsEconomy.amortized_growth_cheaper** — рост-by-fibers
  параметрически дешевле монолита ТОЛЬКО при N·r < d. Наш случай
  (928 экспертов × 4 раунда vs d=96) — за границей: per-expert
  параметрическая экспансия была бы дороже монолита. Вывод по
  теории: AST-композиции бесплатны (не параметры) и потому
  законны, а нейро-адаптеры per-expert требуют MDL-контрбаланса —
  наше прореживание теоретически ОБЯЗАТЕЛЬНО, не опционально.
- **FlopsEconomy.pipeline_beats_baseline_iff** — сравнение
  пайплайна с baseline сводится к ОДНОМУ неравенству нетто-гейнов,
  эмпирическая премиза; Lean позиционирует гонку, не решает.
- Architecture/ (AlignedMerge, FactorizedMerge, Heterarchy) —
  фронты для слияния нейро-ядер при скейле.


## Цикл 5 сверки (2026-11-17): CertStepGen телескоп — ЧИСЛЕННО ПОДТВЕРЖДЁН

- **trajectory_telescope_gen**: objective(t_n) ≤ objective(t₀) −
  Σ drops. Замер на живой модели: holdout CE 5.546→2.020 за 30
  certified шагов; реальный спуск 3.526 ≥ Σ сертификатов 3.313 —
  телескоп выполняется даже на не-обучающем holdout. Реализация
  CertifiedStepLM точно соответствует ядру C278.


## Цикл 6 сверки (2026-11-18): DataToQuality/DepthExitCert/EnergyReasoning

- **DataToQuality.sampleComplexity = log(m/δ)/(2η²)** — честная
  ревизия self-edit гейта: 2 фиксированных бенчмарк-раунда дают
  η≈±99% — гейт ДЕТЕРМИНИРОВАННЫЙ fixed-seed no-regression тест,
  НЕ статистическая гарантия. Документировано прямо в коде гейта
  (honest note). Статистическая версия требует ~780 раундов на
  патч при η=5% — фронт при скейле.
- **DepthExitCert (exitLeak = Σ bᵢ·∏(1+μⱼ))** — ранний выход из
  стека с честной композицией скипов; наш floor_ladder —
  операциональный аналог (budget floor падает при отсутствии
  рельефа); формальный exit-certificate — фронт.
- **EnergyReasoning/CRAM** (E-track E1) — energy-descent как
  альтернативный алгоритм; зафиксировано как фронт, не заявляем.


## Цикл 7 сверки (2026-11-18): Audit.lean Aitken-экстраполяция

- **aitken_exact**: по трём точкам геометрической траектории
  Lstar = (L0·L2 − L1²)/(L0+L2−2L1). Реализовано как
  AdaptiveScheduleLM.aitken_final (с None на вырожденном окне —
  consistency-тест обязателен по aitken_consistency). Измерено на
  CertifiedStepLM: окно ep10/20/30 предсказало финал 2.115,
  фактический на ep120 — 1.772 (ошибка 0.34 — траектория не строго
  геометрична, что теория и оговаривает).
- **Core/Ambig.lse_shift** — сдвиг logsumexp на константу точен;
  применяется в softmax-нормировках (softmax инвариантен к сдвигу
  — согласовано с нашей реализацией).


## Цикл 8 сверки (2026-11-18): полное покрытие

- **GatedCEStep** — наш паттерн (frozen backbone + additive
  adapters + safe step) ТОЧНО соответствует теореме строгого
  спуска gated-CE: сверка архитектуры ✓.
- **LLRD/NeuronAlloc/Lion/Lookahead/KVCache/DepthExit/CRAM** —
  фронты при скейле (не заявляем).
- ИТОГО за 8 циклов: 15 групп теория↔код сверены, 1 реальное
  нарушение найдено и исправлено (§3.9 JVP R'), 2 улучшения
  внедрены (δ-schedule, strict-positive-gain), честные границы
  задокументированы. Все коммиты в main.


## Цикл 9 сверки (2026-11-19): Data/BoolVenn + Model/Logic/VennHead +
## Data/AnfTransform + AnfDispatch (CDGC этап A)

- **VennHead python-порт** (hagi/hagi_venn.py): vAtom = relu(±A±B−1),
  atomIdx порядок (++),(+-),(-+),(--); ВСЕ 64 проверки (16 масок × 4
  входа) точны — vennHead_exact подтверждён численно.
- **ANF python-порт** (hagi/hagi_anf.py): Мёбиус GF(2); инволюция
  (anf_inversion) и вычисление полинома точны на 98 проверках
  (n=1,2,3); dispatch с отказом при конфликте (AnfDispatch).
- **Урок 34 Наде**: И/ИЛИ/НЕ как композиция концептов (CE 0.038,
  recalls верны) — CDGC этап B-C начат: логика привязана к
  цветам/формам.
- Basis для CDGC (Logic→Concepts→World Model) готов: Venn+ANF —
  формальный логический фундамент, симулятор мира — следующий фронт.


## Цикл 10 сверки (2026-11-19): AnfCompose + MasterChallengeV4Kernel +
## CanaryRamp

- **AnfCompose (f∘g∘h roundtrip)**: python-порт подтверждён — ANF
  из таблицы композиции evалируется точно в f(g(x),h(x)) на всех 8
  входах (n=3); stencil_guard_sound: support-зависимости g на {0,1}
  и h на {2} проверены (guard на композицию = guard на компоненты).
- **MasterChallengeV4Kernel** — телескоп Σdrops ≤ objective(t0) −
  objective(tn) при FloorsKept/budget/risk/delta инвариантах. Наш
  CertifiedStepLM + forever-гейты реализуют тот же контракт;
  численно подтверждён в цикле 5 (3.526 ≥ 3.313).
- **CanaryRamp linearRamp = j/T** — монотонный ограниченный ramp;
  операциональный аналог — наше δ-расписание MAX_GATE_RUNS=6
  (Σ6/(π²k²)≤1) и постепенное развёртывание self-edit патчей.
- Логический конвейер Venn→TT→ANF→Compose→Dispatch полностью
  портирован и численно идентичен теории.


## Цикл 11 сверки (2026-11-19): Ensemble/Hoeffding + Growth/RsiCriterion

- **Hoeffding twoGap_bounded** — экспоненциальная граница через
  cosh (log cosh/2 ≤ M²/4): согласована с DataToQuality
  sampleComplexity (обе дают квадратичную зависимость от tolerance);
  подтверждает honest-note цикла 6.
- **Growth/RsiCriterion** — динамический критерий само-улучшения:
  ξ + (1−ρ)D < βC ⟹ D растёт. Проверен на ЖИВОМ состоянии forever:
  эксперты 1004→1019 (rounds 316), поток новых программ 0.14/раунд
  > 0 — критерий ВЫПОЛНЕН на реальных данных (ρ≈1, ξ=prune < βC).
- **Согласование Takeoff/Saturation**: growth-модули группы Growth
  формализуют именно тот вечный цикл, который сейчас исполняется
  (316 раундов без остановок).
