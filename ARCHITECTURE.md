# HAGI_v2 — архитектура (пост-рефакторинг)

## Пакет hagi/ — единый источник истины

| Модуль | Роль | Теория (primes) |
|---|---|---|
| `hagi/_legacy.py` | ядро Expr/Program/TaskSpec/Expert (V5–V8 базис) | C278 CertStepGen-носитель |
| `hagi/_v9.py` | V9-канонический конвейер: синтез, гейты, рост | CertifiedLoop, GatePipeline |
| `hagi/_neural_gpu.py` | GPU-бэкенд нейроядра | C328 work model |
| `hagi/hagi_selfloop.py` | GROW→DERIVE→OPTIMIZE→GATE цикл, ParallelLearner (memory-aware) | C330 steps-to-ε |
| `hagi/hagi_selfrewrite.py` | персистентность + экспорт программ (хэш-имена) + forever | gate_never_increases |
| `hagi/hagi_selfoptimize.py` | MDL-прореживание библиотеки (call-DAG сохранён) | минимизация объёма (C330) |
| `hagi/hagi_selfedit.py` | propose→gate→adopt правки рычагов обучения | §9 gate_never_increases |
| `hagi/hagi_curriculum.py` | order-free фазовый учёт Δᵢ/dᵢ | CurriculumCert |
| `hagi/hagi_lm.py` | нейро-тело: ByteLM/Adaptive/CertifiedStep/RoPE | §2/§3.8/3.9, §6, C216/C223 |
| `hagi/hagi_mind.py` | гибридный интерфейс EXACT/NEURAL | §17 Venn-голова |
| `hagi/hagi_execute.py` | исполнитель написанных программ | — |
| `hagi/hagi_ingest.py` | обучение на любых данных | DataToQuality |
| `hagi/nadya_lessons.py` | школьная книга (кумулятивная) | curriculum replay |
| `hagi/hagi_watchdog.pyw` | сторож: никогда не останавливаться | Ville anytime |

## Автономный цикл (вечный)

    forever: LEARN(suite+segments) → SELF-OPTIMIZE (MDL prune)
             → SELF-EDIT (gated knobs) → LM fine-tune (own artifacts)
             → curriculum certificate → sleep → repeat

Watchdog следит: процесс упал/лог замер → рестарт. Рост:
эксперты 987+, раунды 309+, rewrites 42+ (продолжает).

## Запуск

    python -m hagi.hagi_selfrewrite --forever --rounds 10 --workers 32
    python -m hagi.hagi_mind ask "gcd 12 18"
    python -m hagi.hagi_lm train data.txt --ternary
    python -m hagi.hagi_ingest data.csv
