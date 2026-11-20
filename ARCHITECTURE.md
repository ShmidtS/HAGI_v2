# HAGI_v2 — архитектура (пост-рефакторинг 2026-11-19)

## Пакет hagi/ — единый источник истины

### Ядро и нейрон
| Модуль | Роль | Теория (primes) |
|---|---|---|
| `core.py` | merged kernel: Expr/Program/Expert V5-V8 + V9-канонический конвейер (8989 строк, later definitions win shared names) | C278, CertifiedLoop |
| `lm.py` | нейро-тело: CertifiedStepLM (§3.9 certified step, k=16/dim=128/hidden=256), rotating holdout | §3.9, PoEBound |
| `unified.py` | ЕДИНАЯ модель: ingest любых байтов, fresh-first replay (contiguous ≥2KB), ledger | DataToQuality |
| `fabric.py` | Venn-ткань: геномы, bitwise emulate, imply-оракул 256/256, GROW→OPTIMIZE | VennPacking |
| `gpu.py` | GPU-бэкенд нейроядра | C328 |

### Символьный синтез (стек модернизирован)
| Модуль | Роль |
|---|---|
| `fusion_v4_0.py` | Autonomous Temporal Skill Engine (v4.0, пользовательский код, 1-в-1): factored GF(2) synthesis (Davio/divisors/resub), Mealy quotient, PAC-L*, AG>12, zero-import replay. Компилятор по умолчанию: adder 5g, mux 3g |
| `fusion_anf.py` | v2.1 ANF front-end (fallback синтезатор) |
| `env_table_compiler.py` | общий компилятор transition-table → netlist (Rule of Three) |
| `env_growth_v4_4.py` + `_bridge.py` | среды 5-бит: active identification, перенос памяти, drift |
| `env_growth_v4_6.py` + `_bridge.py` | среды 4-бит: банк 1536 гипотез, OOD LUT3, PROVEN_EXACT аудит |
| `rdl.py` | **Representation Discovery Loop**: изобретение конструкций (LFSR/cnt-mod/sat/Gray/shift2), FAIL→propose→sampled→adversarial sweep→certificate, измерение экономии (52% vs full-table) |

### Цикл и контроль
| Модуль | Роль |
|---|---|
| `loop.py`, `rewrite.py`, `selfedit.py`, `mdl.py` | certified self-development: GROW→DERIVE→OPTIMIZE→GATE, MDL prune, gated knobs |
| `controller.py`, `selfmodel.py`, `experiment.py` | v10 controller, self-model, A/B эксперименты |
| `curriculum.py`, `lessons.py`, `ingest.py`, `execute.py`, `mind.py` | учебный контур и интерфейс |

## Нейро-символьный цикл (production)

```
scripts/neuro_symbolic_cycle.py <minutes>
свип C:/ (fresh-first) → certified train (eta=4A/R'²) →
derived tasks → fusion fast-path (v4.0→v2.1→GA) →
TAPE-OUT processor_netlist.json (exhaustive PASS)
```

Запуск: 1-поточный OpenBLAS (`OPENBLAS_NUM_THREADS=1` + `MAIN_FREE=1`)
против gemm malloc. Артефакты: checkpoints/{unified.pt,
unified_ledger.json, processor_netlist.json, domains/*.bin}.

## Верификация

- tests/ — 24 теста (characterization, fusion_anf, rotating_holdout, rdl)
- v4.0 self-verify: 57 skills, DAG exact, zero-import replay, tamper reject
- env v4.4/v4.6: independent verifiers (260/100 записей PASS)
- OOS-бенчмарк: scripts/oos_bench.py (фикс. тексты вне обучения)

## Честные границы

- Речь: морфемная (нужен порядок данных больше)
- RDL: классы из генератора параметрических семей, не произвольные
- OOD: точен внутри объявленной семьи + full-table fallback
