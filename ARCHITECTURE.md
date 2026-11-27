# HAGI — архитектура

HAGI — саморазвивающаяся модель, работающая с любыми данными как с потоком
байтов. В ней два «органа»: символьное ядро (программы над целыми числами,
точное воспроизведение пар «вход → выход») и нейронная часть (языковая модель
по байтам p(b | контекст)). Единая модель `hagi.selfdev.unified` принимает данные и
связывает органы: адаптеры доменов, ремонт забывания, рост, законы, память.


## Слои и направление зависимостей (только вниз)

```
интерфейс      hagi/__init__.py (явный __all__), hagi/cli.py, hagi/__main__.py
hagi/base      фундамент: config, util, hashing, gpu, embeddings, trunk_ops,
               sandwich, measurement, certificates
hagi/core      символьное ядро (6 групп: language, tasks, synthesis, neural,
               processor, memory)
hagi/models    lm (RopeAttentionByteLM, CertifiedStepLM), hnet_lm (HNetByteLM),
               quantized, archiver
hagi/invent    diagnosis, invention, invention_multi, representations
hagi/data      io, env_lut, env_temporal, skill_engine, law_critique,
               fabric, genesis; рядом memory_store/*
hagi/selfdev   selfmodel, loop, school, rewrite, cycle, pipeline,
               unified (единая модель)
```

Зависимости идут только вниз: selfdev → data/invent → models → base; ядро
(core) ни от кого, кроме base, не зависит.

## Символьное ядро

`hagi/core/`

| группа | модули | что делает |
| --- | --- | --- |
| `language` | `expressions`, `expr_dsl`, `call_dag`, `code_length` | дерево выражения, программа, векторизованный вычислитель; безопасный язык выражений; префиксный код и граф вызовов; длина кода |
| `tasks` | `tasks`, `stored_tasks`, `candidates` | набор задач ядра; набор задач для хранимых экспертов; перечисление программ-кандидатов |
| `synthesis` | `synthesis`, `venv_anf`, `venn`, `prefilter`, `best_first`, `fast_mdl`, `induction` | точная полиномиальная индукция и МДЛ-синтез; голова Венна, Жегалкин, Адамар, 5-тритовый кодек; векторизованный префильтр; поиск «лучшие первыми»; быстрый МДЛ; решатель раскраски и майнер теорем |
| `neural` | `neural`, `segmentation`, `fiber`, `certified_ce`, `energy`, `chunking` | перекрёстная энтропия, EF21, GPM, CRAM; адаптивная сегментация; нейронное волокно; сертифицированная энтропия; энергетический решатель; растущая адресная память и маршрутизатор |
| `processor` | `processor`, `growth`, `stored_growth`, `experts`, `optimal_processor`, `unified_processor`, `benchmark`, `self_development` | эксперт, майнер теорем, `GrowingHagiProcessor`; цикл роста; хранилище экспертов (`GrowingMathProcessor`, `GrowthProcessorBase`); оптимум и саморазвитие; замеры |
| `memory` | `audit`, `ledger`, `aliases`, `selfcheck`, `experiment`, `selftest`, `cli` | сертифицированный шаг и журнал аудита; журнал статистических проверок; арифметические/битовые операции и префиксные коды; самопроверки; эксперимент; командная строка ядра |



## Нейронная часть и единая модель

- `lm.py`: RopeAttentionByteLM (RoPE-внимание, окна oldest-first,
  train_guarded с гейтом по holdout), CertifiedStepLM, `make_lm(arch)`.
- `hnet_lm.py`: HNetByteLM (torch-порт), точный рост ширины `grow_width`.
- `unified.py`: `Unified` — ingest доменов, низкоранговые адаптеры
  (`trunk_ops.LowRankAdapter`, живой гейт), `_repair_forgetting`
  (переучивание старых байт с доказательной границей `paired_margin_eb`),
  `_maybe_grow`, `grow_width`, save/load, `reindex_domains`.
- `archiver.py`: арифметический кодер поверх модели (пакетный и потоковый с
  контекстной таблицей `q = (1−λ)p + λr`).

## Конвейер приёма данных

```
hagi/selfdev/pipeline.py: unified_ingest(u, source, data)
  1. нейро:   u.ingest (сертифицированный шаг)
  2. диагноз: профиль остатка → classify_hold
  3. изобрет: invent_blueprint на промежутке
  4. гейт:    verify_blueprint (точность на всём домене)
  5. приём:   reify_construct → операция доступна следующим доменам
```
Флаг `HAGI_SYMBOLIC=1` включает шаги 2–5. Все настройки окружения читаются
только через `hagi/base/config.py`; каталог контрольной точки — `hagi.util.CKPT_DIR`
(переменная `HAGI_CKPT_DIR`).

## Командная строка

```
python -m hagi.cli <команда>     core | lm | ingest | execute | loop | rewrite | genesis | certificates
python -m hagi core --help       --self-test, --self-test-core, --self-test-optimal, --self-test-legacy, --grow, --grow-deep, --benchmark-core
python scripts/hagi_archive.py   compress | decompress | bench   (--prefix-head F — потоковый кодек)
python scripts/comment_audit.py  аудит комментариев (0/0/0)
```

## Файлы контрольной точки

`checkpoints/unified.pt` (веса, адаптеры, законы), `unified_ledger.json`
(книга доменов: sha, размер, CE), `domains/*.bin` (байты доменов),
`cycle.log` (журнал цикла, UTF-16 LE).


## Инструменты поддержки кода

Все — в `scripts/`, импортируемые (`main(argv)`):

- `comment_audit.py [путь] [--lines A B] [--ast-compare OLD NEW]` —
  покрытие комментариями, русский язык, запрет отсылок; сравнение AST
  до/после (код не изменён).
- `prune_imports.py <файлы|каталоги>` — убрать неиспользуемые и повторные
  импорты (учитывает имена, которые другие модули импортируют отсюда;
  `__init__.py` и `import *` не трогает; `__future__` сохраняет).
- `dead_audit.py <модуль> [...]` — функции/классы без единой ссылки в коде.


## Честные границы

Известная граница модульности: статический анализатор (basedpyright) насчитывает
22 цикла импорта — все проходят через фасады (`hagi/__init__`, `hagi/core/__init__`)
или через импорты под `TYPE_CHECKING` (аннотации), во время выполнения не возникают.
Устранение требует понизить зависимость модулей от фасада до прямых импортов.

- Забывание при приёме новых доменов замедлено ремонтом, но не остановлено.
- Архиватор с контекстной таблицей сжимает хуже zlib примерно на 25%.
- Дубли хешей с разными параметрами канонизации не объединены: хеши
  записаны в сохранённых законах и сертификатах.
