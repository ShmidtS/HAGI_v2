# HAGI — растущая саморазвивающаяся модель

HAGI — единая модель, которая учится на любых данных (текст, код,
таблицы, любые байты) и растёт: нейронная часть предсказывает следующий
байт, символьная извлекает из данных точные программы-эксперты. Вне
своего домена модель честно отказывается (`ABSTAIN`/`HOLD`), а не угадывает.

Архитектура и направление зависимостей — в [ARCHITECTURE.md](ARCHITECTURE.md).
Соответствие теории формального репозитория `primes` и кода — в
[THEORY_ALIGNMENT.md](THEORY_ALIGNMENT.md).

## Из чего состоит

- `hagi/selfdev/unified.py` — **единая модель**: приём домена, гейты принятия,
  рост тела (блок внимания принимается, если выигрыш CE больше границы
  эмпирического Бернштейна и цены памяти), ремонт забывания старых доменов
  (байты ремонта распределяются по стоимости `n_i = √c_i·N/Σ√c_j`).
- `hagi/models/hnet_lm.py` — класс `HNetByteLM`: байтовая модель с вращением позиций (RoPE-внимание,
  доменные адаптеры низкого ранга, растущие блоки).
- `hagi/selfdev/pipeline.py`, `hagi/invent/diagnosis.py`, `hagi/invent/invention.py` — конвейер
  приёма: обучить → найти остаток (гэп) → изобрести представление →
  строгий гейт → добавить операцию в библиотеку.
- `hagi/core/` — символьное ядро, один пакет из шести групп по смыслу:
  `language` (выражения и код), `tasks`, `synthesis` (поиск программ),
  `neural`, `processor` (эксперты и процессоры), `memory` (аудит, журнал,
  самопроверка, командная строка).
- `hagi/models/archiver.py` — архиватор с префиксной моделью (без потерь).
- `hagi/memory_store/` — потоковая семантическая память.
- `scripts/neuro_symbolic_cycle.py` — рабочий цикл обучения
  (окружение запуска — `scripts/cycle_env.md`).

## Быстрый старт

```bash
python -m hagi.cli --help                       # все команды
python -m hagi.cli core --self-test-core        # самопроверка символьного ядра
python -m hagi.cli execute --list               # 1000+ программ-экспертов
python -m hagi.cli ingest path/to/data.csv      # извлечь точные программы из данных
python -m hagi.cli lm train README.md           # обучить байтовую модель
python -m hagi.cli lm generate "HAGI — " --n 120
```

Приём данных единой моделью из Python:

```python
from hagi.selfdev.unified import Unified
from hagi.selfdev.pipeline import unified_ingest
from hagi.invent.diagnosis import GapBuffer

u = Unified.load()                      # одна модель (checkpoints/unified.pt)
r = unified_ingest(u, "my_domain", data_bytes, gap_buffer=GapBuffer())
# r["new"], r["skipped"]  — что добавлено в библиотеку и что отвергнуто
# r["symbolic"]          — отчёт символьной фазы (диагноз, изобретение, гейт)
```

Путь к контрольной точке задаётся переменной `HAGI_CKPT_DIR`
(по умолчанию `checkpoints`). Одну контрольную точку пишет только один
процесс; эксперименты запускайте с отдельной папкой.

## Проверка

```bash
HAGI_DEVICE=cpu python -X utf8 -m pytest tests/ -q   # 253 тестов
python -X utf8 scripts/comment_audit.py              # 0/0/0 по всему проекту
python -X utf8 scripts/monitor_cycle.py              # состояние рабочего цикла
```

## Принципы

- **Точность > приближение.** Эксперт committится только при полном
  покрытии домена; иначе `HOLD` с откатом.
- **Ничего без гейта.** Рост, адаптер, раунд ремонта — каждый принимается
  только если парный выигрыш превышает статистическую границу; иначе откат.
- **Одна модель.** Все домены живут в одной сети; забывание старых доменов
  измеряется на фиксированных окнах и ремонтируется под тем же гейтом.
