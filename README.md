# HAGI — растущий нейро-символьный процессор

HAGI — саморазвивающийся нейро-символьный процессор (V9): библиотека
точных программ-экспертов, растущая через certified-цикл обучения.
Каждая задача либо закрывается переиспользованием существующего
эксперта, либо синтезируется новая точная программа (Venn-голова,
ANF/Жегалкин, шаблоны, точные полиномы, piecewise, batched MDL
meet-in-the-middle DAG), либо честно HOLD'ится с откатом и аудитом.

## Структура

- `hagi/_neural_gpu.py` — GPU-бэкенд нейроядра (torch/CUDA-ROCm,
  AMD Radeon 8060S): EF21-Muon + Newton-Schulz (C296/C297) и
  SVD-консолидация на устройстве; `HAGI_NEURAL_DEVICE=auto|cpu|cuda`,
  auto выбирает по бенчмарку; параллельные воркеры — всегда CPU.
- `hagi/` — ядро (бывший монолит `hagi.py`, разрезан с сохранением
  семантики перекрытия имён):
  - `hagi/_legacy.py` — поколения V5–V8 (исторические процессоры,
    guarded optimizer, тернарное квантование, CRAM, theorem miner);
  - `hagi/_v9.py` — каноническое ядро V9: `GrowingHagiProcessorV9`,
    точный синтез (`synthesize_batched_mdl_v9`,
    `exact_piecewise_poly_fit`), Venn-голова, Hadamard-микшер,
    `AuditLedgerV9` (Hoeffding-альфа), `CertStepGen`-сертификаты.
- `hagi_selfmodel.py` — одношаговый TRACE → DERIVE → OPTIMIZE → GATE:
  модель трассирует своё обучение и пытается вывести собственный
  предиктор успеха (ExprV9-синтез; при невозможности — честный HOLD).
- `hagi_selfloop.py` — много-раундовый цикл саморазвития (см. ниже).
- `hagi_selfrewrite.py` — self-rewrite: модель пишет собственные
  программы (standalone Python, точная валидация по каждому эксперту,
  0 ошибок на 19k проверок) и переписывает собственное состояние
  (`checkpoints/self/self_state.json`); `--resume` загружается из
  того, что модель сама о себе написала, и продолжает рост
  (измерено: 5 циклов рост→рестарт→рост, суммарно 117 раундов,
  411 экспертов, 395 standalone-функций, 34677 точных проверок
  функций против экспертов, 0 ошибок, 7 self-rewrites).
- `hagi_selfexperiment.py` — эксперименты с бюджетными политиками
  (seed-панели, online-conservative policies).
- `hagi_v10_controller.py` — guarded optimizer step (MGDA/мультигрань).
- `scripts/run_selfmodel_experiment.py` — CI-прогон self-model.
- `tests/test_characterization.py` — characterization-гварды.
- `experiments/selfloop_result.json` — последний отчёт цикла.

## Цикл саморазвития (`hagi_selfloop.py`)

```
GROW     инкамбент + N новых некруговых задач (параллельные батчи)
DERIVE   точный decision-list self-model по трассе
         (признаки: family, arity, domain, library, target_range,
          target_unique → COMMIT/HOLD)
         holdout-валидация: уверенно-ложный HOLD-прогноз = блок
OPTIMIZE реплей ТОГО ЖЕ детерминированного потока задач с бюджетной
         политикой из self-model (предсказанный HOLD → floor-бюджет)
GATE     нерегрессия: superset экспертов, точность 1.0 на всём suite,
         relief ≥ 2% (certified margin) → COMMIT, иначе HOLD-останов
```

Модель сама выводит свой алгоритм распределения вычислений и сама
гейтует своё принятие; останов — только сертифицированный (нет
certified-positive действия = ресурс исчерпан).

Домены: 49 математических задач + физика (4 закона — свободное
падение, равномерное движение, пружина, импульс; все выучены точными
полиномами или переиспользованы) + язык 3/4 (ATBASH, Цезарь-3 и
SHIFT (c+k)%26 выучены после расширения CONSTS алфавитным модулем 26
и подъёма бюджета синтеза до 20000 evals; IS_VOWEL — единственный
оставшийся HOLD-фронт). Нейроядро (FactorizedSharedCoreV9,
EF21-Muon GPM + Hedge-router) обучается вместе с каждым коммитом
(66 адаптеров на 82 экспертах в 5-раундовом прогоне), с GPU-путём
и автовыбором устройства.

## Производительность

- Параллель: батчи задач обучаются в воркерах
  (`ProcessPoolExecutor`, все ядра CPU, cloudpickle-транспорт),
  слияние в порядке потока (batch-семантика: задачи одного батча не
  видят коммитов соседей — детерминированно и для роста, и для реплея).
- Replay-кэш: неизменный префикс реплея не переучивается — цикл
  линеен по числу задач, а не квадратичен.
- Lever-change (R223): при исчерпании рычага (relief < 2%) цикл сам
  переключает floor-бюджет по лестнице 200→50→20 и продолжает рост —
  «останавливайся со сменой рычага, а не "ещё раз так же"».
- Измерено (32 воркера): 149/150 COMMIT-раундов, 55→495 экспертов за
  398 с, waste 104400→31100 после автономной смены рычага floor 200→50
  на раунде 102; рост продолжался до лимита раундов.

## Быстрый старт

```bash
python -m hagi --self-test-v9      # self-test ядра
python -m pytest                   # characterization-гварды
python -X utf8 -u hagi_selfloop.py --max-rounds 24 --workers 32
python -X utf8 -u hagi_selfmodel.py           # одношаговый self-model
python -X utf8 -u hagi_selfrewrite.py --rounds 10      # рост + self-rewrite
python -X utf8 -u hagi_selfrewrite.py --resume --rounds 10  # автономный рестарт
```

## Принципы

- Точность > приближение: эксперт коммитится только при 100% покрытии
  домена (train + val + full support).
- Ничего не принимается без гейта: нерегрессия, качество, certified
  margin; HOLD — честный результат.
- Аудит: каждая трата α фиксируется в `AuditLedgerV9` (Hoeffding),
  финальный тест-диапазон не трогается.
