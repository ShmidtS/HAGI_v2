# HAGI — растущий нейро-символьный процессор

HAGI — саморазвивающийся нейро-символьный процессор: библиотека
точных программ-экспертов, растущая через certified-цикл обучения.
Каждая задача либо закрывается переиспользованием существующего
эксперта, либо синтезируется новая точная программа (Venn-голова,
ANF/Жегалкин, шаблоны, точные полиномы, piecewise, batched MDL
meet-in-the-middle DAG), либо честно HOLD'ится с откатом и аудитом.

## Структура

- `hagi/hagi/gpu.py` — GPU-бэкенд нейроядра (torch/CUDA-ROCm,
  AMD Radeon 8060S): EF21-Muon + Newton-Schulz (C296/C297) и
  SVD-консолидация на устройстве; `HAGI_NEURAL_DEVICE=auto|cpu|cuda`,
  auto выбирает по бенчмарку; параллельные воркеры — всегда CPU.
- `hagi/` — ядро (бывший монолит `hagi.py`, разрезан с сохранением
  семантики перекрытия имён):
  - `hagi/core.py (merged)` — исторические процессоры (замороженное
    guarded optimizer, тернарное квантование, CRAM, theorem miner);
  - `hagi/core.py (merged)` — каноническое ядро: `GrowingHagiProcessor`,
    точный синтез (`synthesize_batched_mdl`,
    `exact_piecewise_poly_fit`), Venn-голова, Hadamard-микшер,
    `AuditLedger` (Hoeffding-альфа), `CertStepGen`-сертификаты.
- `hagi/hagi/selfmodel.py` — одношаговый TRACE → DERIVE → OPTIMIZE → GATE:
  модель трассирует своё обучение и пытается вывести собственный
  предиктор успеха (Expr-синтез; при невозможности — честный HOLD).
- `hagi/hagi/loop.py` — много-раундовый цикл саморазвития (см. ниже).
- `hagi/hagi/rewrite.py` — self-rewrite: модель пишет собственные
  программы (standalone Python, точная валидация по каждому эксперту,
  0 ошибок на 19k проверок) и переписывает собственное состояние
  (`checkpoints/self/self_state.json`); `--resume` загружается из
  того, что модель сама о себе написала, и продолжает рост
  (измерено: 5 циклов рост→рестарт→рост, суммарно 117 раундов,
  411 экспертов, 395 standalone-функций, 34677 точных проверок
  функций против экспертов, 0 ошибок, 7 self-rewrites).
- `hagi/hagi/experiment.py` — эксперименты с бюджетными политиками
  (seed-панели, online-conservative policies).
- `hagi/hagi/controller.py` — guarded optimizer step (MGDA/мультигрань).
- `hagi/hagi/execute.py` — ИСПОЛНИТЕЛЬ программ: модель запускает то, что
  написала — execute(name, *args) для любой из сотен программ,
  execute_text() прогоняет живой UTF-8 текст (кириллица, эмодзи)
  через байтовые шифры (XOR round-trip работает), CLI для прямой
  инференции.
- `hagi/hagi/ingest.py` — обучение на ЛЮБЫХ данных: числовые CSV/TSV/JSON
  (каждая пара столбцов проверяется на детерминированность y=f(x),
  затем индукция точной программы, покрывающей 100% данных) и любые
  файлы как байты (детерминированное подмножество prev→next).
  Демо: физический закон 20−5t² восстановлен точным полиномом из
  sensor.csv; неоднозначные пары честно HOLD.
- `hagi/hagi/lm.py` — нейро-тело по теории primes (§2 CE-loss по байтовому
  словарю, §3 guarded-шаги: деплой только лучшего holdout-снапшота,
  §6 тернарное квантование с guarded fine-tune): байтовый LM,
  обучаемый на любых данных, с генерацией. Первый вероятностный
  орган HAGI — измерено: CE 5.4→2.3 на 68 КБ кода/документации,
  тернарная версия CE 2.5 (10% цена за 3× сжатие).
- `scripts/run_selfmodel_experiment.py` — CI-прогон self-model.
- `tests/test_characterization.py` — characterization-гварды.
- `experiments/selfloop_result.json` — последний отчёт цикла.

## Цикл саморазвития (`hagi/hagi/loop.py`)

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
полиномами или переиспользованы) + язык: латиница 3/4 (ATBASH, Цезарь-3,
SHIFT mod 26) и ВЕСЬ пул UTF-8 на байтовом уровне 4/4 (ATBASH,
Цезарь, SHIFT, XOR-шифр на 0..255 — любой юникод-текст) после
расширения CONSTS (26, 256) и input-cap синтеза до 2048; честный
фронт — разреженные классификаторы IS_VOWEL и RU_VOWEL (кириллица). Нейроядро (FactorizedSharedCore,
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

### Единый конвейер (нейро + символ, волна 185)

```python
import os
os.environ["HAGI_CKPT_DIR"] = "checkpoints"   # путь чекпоинтов
os.environ["HAGI_SYMBOLIC"] = "1"             # вкл. символьную фазу
from hagi.pipeline import unified_ingest
from hagi.unified import Unified
from hagi.diagnosis import GapBuffer

u = Unified.load()          # ОДНА модель
r = unified_ingest(u, "my_domain", data_bytes,
                   gap_buffer=GapBuffer())
# r["holdout_ce"]  — нейронная фаза (certified LM)
# r["symbolic"]    — diagnose -> invent -> gate -> reify
```

Каждый ingest: certified обучение LM → диагноз остатка →
изобретение представления на гэпе → строгий гейт (PROVEN_EXACT) →
реификация в библиотеку операций. Шум честно отсекается
(RANDOM_INCOMPRESSIBLE).

### Классические точки входа

```bash
python -m hagi --self-test-core    # self-test ядра
python -m pytest                   # characterization-гварды
python -X utf8 -u hagi/loop.py --max-rounds 24 --workers 32
python -X utf8 -u hagi/selfmodel.py           # одношаговый self-model
python -X utf8 -u hagi/rewrite.py --rounds 10      # рост + self-rewrite
python -X utf8 -u hagi/rewrite.py --resume --rounds 10  # автономный рестарт
# Единый CLI (волна 190): одна точка входа для всех инструментов
python -m hagi.cli --help                          # список команд
python -m hagi.cli core --self-test-core           # self-test ядра
python -m hagi.cli execute --list                  # все программы
python -m hagi.cli execute --task LANG8_XOR_CIPHER --text "привет" --key 42
python -m hagi.cli ingest path/to/data.csv         # учиться на данных
python -m hagi.cli lm train README.md --ternary    # нейро-тело
python -m hagi.cli lm generate "HAGI — " --n 120   # генерация
```

## Принципы

- Точность > приближение: эксперт коммитится только при 100% покрытии
  домена (train + val + full support).
- Ничего не принимается без гейта: нерегрессия, качество, certified
  margin; HOLD — честный результат.
- Аудит: каждая трата α фиксируется в `AuditLedger` (Hoeffding),
  финальный тест-диапазон не трогается.
