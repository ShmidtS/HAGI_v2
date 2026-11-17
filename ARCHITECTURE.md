# HAGI_v2 — архитектура (пост-рефакторинг)

## Пакет hagi/ — единый источник истины

| Модуль | Роль | Теория (primes) |
|---|---|---|
| `hagi/core.py (merged)` | ядро Expr/Program/TaskSpec/Expert (V5–V8 базис) | C278 CertStepGen-носитель |
| `hagi/core.py (merged)` | V9-канонический конвейер: синтез, гейты, рост | CertifiedLoop, GatePipeline |
| `hagi/hagi/gpu.py` | GPU-бэкенд нейроядра | C328 work model |
| `hagi/hagi/loop.py` | GROW→DERIVE→OPTIMIZE→GATE цикл, ParallelLearner (memory-aware) | C330 steps-to-ε |
| `hagi/hagi/rewrite.py` | персистентность + экспорт программ (хэш-имена) + forever | gate_never_increases |
| `hagi/hagi/mdl.py` | MDL-прореживание библиотеки (call-DAG сохранён) | минимизация объёма (C330) |
| `hagi/hagi/selfedit.py` | propose→gate→adopt правки рычагов обучения | §9 gate_never_increases |
| `hagi/hagi/curriculum.py` | order-free фазовый учёт Δᵢ/dᵢ | CurriculumCert |
| `hagi/hagi/lm.py` | нейро-тело: ByteLM/Adaptive/CertifiedStep/RoPE | §2/§3.8/3.9, §6, C216/C223 |
| `hagi/hagi/mind.py` | гибридный интерфейс EXACT/NEURAL | §17 Venn-голова |
| `hagi/hagi/execute.py` | исполнитель написанных программ | — |
| `hagi/hagi/ingest.py` | обучение на любых данных | DataToQuality |
| `hagi/hagi/lessons.py` | школьная книга (кумулятивная) | curriculum replay |
| `hagi/hagi/watchdog.pyw` | сторож: никогда не останавливаться | Ville anytime |

## Автономный цикл (вечный)

    forever: LEARN(suite+segments) → SELF-OPTIMIZE (MDL prune)
             → SELF-EDIT (gated knobs) → LM fine-tune (own artifacts)
             → curriculum certificate → sleep → repeat

Watchdog следит: процесс упал/лог замер → рестарт. Рост:
эксперты 987+, раунды 309+, rewrites 42+ (продолжает).

## Запуск

    python -m hagi.hagi.rewrite --forever --rounds 10 --workers 32
    python -m hagi.hagi.mind ask "gcd 12 18"
    python -m hagi.hagi.lm train data.txt --ternary
    python -m hagi.hagi.ingest data.csv
