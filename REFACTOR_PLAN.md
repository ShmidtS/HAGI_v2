# REFACTOR_PLAN — приведение HAGI_v2 к формализации primes (2026-10-13)

Основание: аудит src ↔ `C:/primes/ALGORITHMS.md` (§0–§12) +
инвентаризация мёртвого кода. Принцип: код уже добросовестно
портирует теорию (formal.py, safeqp_step, grokking_schedule,
growing_batch — образцово). Полный снос архитектуры не оправдан
(Chesterton: рабочая измеренная линия V42). Цель — убрать лишнее и
перенести certified-контур §0 из sidecar-скриптов в ядро.

## Фаза A — очистка (DONE 2026-10-13)

- Удалены из configs/ (→ `_raw/attic/cleanup_2026_10/`):
  latent_leaf_code.yaml, latent_leaf_lang.yaml, latent_root_ft.yaml.
- Перенесены в attic (~30 one-shot скриптов): scripts/audits/*,
  scripts/lora/*, 8 сирот scripts/data/*, 14 сирот
  scripts/research/*, 6 корневых measure/merge-скриптов,
  growth/probe_alpha_zero_merge.
- Возвращены из attic (нашлись потребители): deep200_certify.py
  (test_certified_controller), distill_stage.py (γ-дефицит mixer.
  gain — единственный механизм против открытой зоны теории).
- Тесты: 557 passed, 0 новых падений.

## Фаза B — данный план (DONE)

## Фаза C — theory-conformance ядра (по приоритету аудита)

Статус 2026-10-13: ЗАКРЫТА.

1. **merge_select** — уже реализовано в `select_screened` (§1/§9:
   certified_ab решает, screen — только пре-фильтр; REFACTOR_PLAN
   отставал). Тесты test_merge_select.py зелёные.
2. **Liveness-стоп в loop.py** — СДЕЛАНО 2026-10-13:
   `liveness_two_axis_frozen` (Unified/Liveness.lean, контрапозиция
   `liveness_two_axis`: стоп ⟺ обе оси мертвы) + проводка в
   train(): окна по sat_signal/grad_norm, конфиг-поля
   liveness_{window,energy_tol,growth_tol,min_steps} (default off,
   паттерн saturation/divergence). Тесты test_liveness_stop.py.
3. **Certificate-telescope** — СДЕЛАНО 2026-10-13: чистый модуль
   `train/stage_telescope.py` (StageCertificate, telescope_bound c
   contiguity_tol = шум-допуск 2ε, termination_bound,
   rebuild_from_ladder); проводка в run_ladder.record_stage_certificate
   (каждое ACCEPTED поколение — сертификат в
   checkpoints/stage_telescope.jsonl). Backfill из истории: 22
   сертификата gen4–51; реальный разрыв 0.03 на gen11→13 (смена
   протокола эвал) — честная телеметрия. Тесты
   test_stage_telescope.py.
4. **SafeQP активный гвард** — уже реализовано: certified_step
   "gate"/"step" в loop.py (veto + eta_max кап «never above
   schedule», R105-окна). REFACTOR_PLAN отставал.
5. **η-политика** — без изменений (измеренная практика, смена только
   через полный A/B цикл).

## Фаза D — закрытие пробелов THEORY_MAP (статус 2026-10-13)

- multimodal.py: ПОКРЫТО — tests/test_multimodal.py (fixed-rate
  prefix, no-modality path, audio, Grounding InfoNCE конечность);
  модуль остаётся research-only (конфиг-выключен в живой линии).
- self_development.py: оставлен (кросс-потребитель в
  test_saturation; решение аудита 2026-10-12 подтверждено).
- Anytime/Freedman M∧V в accept-гейте (§7) — ОСТАЁТСЯ ОТКРЫТЫМ
  (нужна anytime-форма Freedman в primes — см. STATUS.md «открытые
  задачи»).
- analytic_step.py → поглощение safeqp_step.py — НЕ ПРОВОДИЛОСЬ
  (поведенческое, только с A/B).

## Фаза E — обучение/развитие (ортогонально рефакторингу)

- Продолжение density ladder / run_ladder с certified-гейтами;
  новые поколения только через certified A/B (§9).
- LR всегда передавать явно (инцидент gen-4 sib3); λ из
  grokking_schedule; batch-масштабирование вместо LR-снижения
  (lr_batch_equivalence).

## Честные границы

- h_emp_grow/smooth/lip не выводятся из Core — не «чинить» кодом.
- γ-дефицит mixer.gain — открытая зона теории; единственный
  инструмент — disagreement_distill (не удалять).
- Замена Muon-политики на чистый SafeQP-шаг НЕ проводится без
  измеренного A/B (риск сломать растущую линию).
