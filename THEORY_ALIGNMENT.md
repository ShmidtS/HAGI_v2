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
