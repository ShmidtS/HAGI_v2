# Теория (C:\primes) ↔ код: карта соответствия

Сводка аудита 2026-10-13 по ALGORITHMS.md (290 модулей, 1712 theorem,
0 sorry). Статусы: ✔ реализовано, ~ частично, ✘ отсутствует.

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| 0 | certified growth loop / telescope | ✔ | `certified_controller.py` + `train/stage_telescope.py` (леджер принятых поколений, сшивка в run_ladder, 2026-10-13) |
| 1 | argmax Γ/K | ✔ | `certified_controller.py` estimate/select_action; merge_select.select_screened (certified_ab решает) |
| 2 | SafeQP, η=1/L | ✔ | `train/safeqp_step.py` (в главном loop) |
| 2b | LR↔batch (growing_batch) | ✔ | `train/growing_batch.py` → `loop.py` |
| 3 | jensenGap merge-гейт | ✔ | `model/formal.py` jensen_gap_*, `train/merge_price.py` |
| 4 | тернарная сетка | ✔ | `model/ternary.py` BitLinear (SSOT-тело, в линии выключено) |
| 4b | QuantBridge | ~ | `train/spectral.py` three_stage_error_budget |
| 5 | Eckart–Young + MDL ранг | ✔ | `model/formal.py` eckart_young_frobenius_error (R372); MDL — `model/latent_merge.py` mdl_rank |
| 6 | water-filling битов | ✔ | `model/formal.py` kv_waterfill_bits |
| 7 | Hoeffding/Azuma/Freedman | ✔ | Hoeffding `certified_controller.py`; Freedman `formal.py` freedman_tail_low (R372); anytime (Ville) — конфиг-поля, runtime-хуков нет |
| 7b | grokking η·λ | ✔ | `train/grokking_schedule.py` (λ выводится в config.validate) |
| 8 | конус/насыщение | ✔ | `train/{ratio_takeoff,saturation,takeoff_window,gain_renewal}.py`; liveness two-axis стоп в `loop.py` (2026-10-13, Unified/Liveness.lean) |
| 9 | капстоун-синтез | ~ | certificate-telescope собран (`stage_telescope.py`); интегрирующего капстоун-сертификата нет — ждёт closed h_emp_* в primes |
| 10 | insight-канал, liveness | ✔ | `train/insight.py` → loop; liveness в certified_controller.stop_condition |
| 11 | variance gate, trust region | ✔ | `safeqp_step.variance_gate_tau`, `formal.trust_region_bound` |

Честные границы: h_emp_* посылки не выводятся из Core (открыто в
формализации); «мёртвый» код достижим через growth_supervisor
(альтернативная точка входа), чистых orphan-модулей нет — массовое
удаление отменено (Chesterton: каждый «выключенный» механизм
измерено избыточен на этом масштабе, но достижим конфигом).

Открытые runtime-пробелы (кандидаты на порт после формализации):
- anytime-валидность (Вилль/Freedman M∧V) в accept-гейте;
- certificate-telescope в `orchestrator/real_cycle.py`;
- h_emp_grow/smooth/lip измерители в Core (см. FORMALIZATION_TODO).
