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
| MC | MemoryCycle / admission / консолидация (R271–R273) | ✔ | `train/memory_cycle.py` (admission conjunction, Cauchy-бюджет k·Σε, L·√(k·Σε) bound; интеграция в адаптеры — отдельный раунд) 2026-10-13 |
| Mu | MuonStableLR η=(2/λ)(Σσ/m) + NewtonPolar min-итерации (R379–R382) | ✔ | `optim.py: muon_eta_max_stable, min_ns_iterations` (opt-in, дефолт без изменений; σ≥3 вне бассейна — fail-fast) 2026-10-13 |
| Gen | ModeState Qgen/Φ_HAGI/mode-drop/Pareto | ✔ | `train/gen_telemetry.py` (5 проб h_emp_G, Q-пламя, гейт, парето-выбор чекпоинтов; интеграция — отдельный раунд) 2026-10-13 |
| Th | Thermostat T_t=T₀α^t + законная остановка (R387–R388) | ✔ | `train/thermostat.py` + ИНТЕГРАЦИЯ: `growth_cycle.py --thermostat-*` (opt-in CLI, дефолт off; фаза в селектор-мете, §Thermostat halt при EXPLOIT; δ = последний принятый A/B delta) 2026-10-13 |
| Ly | BranchScale min-max / dead-layer stop / vocab V*=c / admission ΔVar/ΔC (Layers wave-5, Transfer wave-4) | ✔ | `model/layer_tuning.py` (s_l²·v_l = const init, L*=max{V_l>λ}, c-fit из |V|-свипа, argmax ΔVar/ΔC) 2026-10-13 |

Честные границы: h_emp_* посылки не выводятся из Core (открыто в
формализации); «мёртвый» код достижим через growth_supervisor
(альтернативная точка входа), чистых orphan-модулей нет — массовое
удаление отменено (Chesterton: каждый «выключенный» механизм
измерено избыточен на этом масштабе, но достижим конфигом).

Открытые runtime-пробелы (кандидаты на порт после формализации):
- ~~anytime-валидность в accept-гейте~~ — ЗАКРЫТО 2026-10-16
  (R525 `anytime_safe_threshold` →
  `certified_controller.anytime_accept_gate`, волна R510–R530);
- certificate-telescope в `orchestrator/real_cycle.py`;
- h_emp_grow/smooth/lip измерители в Core (см. FORMALIZATION_TODO);
- anytime Freedman M∧V (Вилль-форма под variance-adaptive порог) —
  следующая после R528 ступень CertifiedEstimator.

## Волна R419–R443 (порт 2026-10-14, после обновления primes)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| Th+ | phase_point N (stop ∧ merge-gate, R407) | ✔ | `thermostat.phase_point` |
| Th+ | Gibbs-концентрация share ≥ 1−Cθ (R408) | ✔ | `thermostat.gibbs_share` |
| Th+ | consolidation_certificate 3-в-1 (R412) | ✔ | `thermostat.consolidation_certificate` |
| MWU | sandwich ηℓmin ≤ −ln Z ≤ 2η⟨ℓ⟩ (R433–437) | ✔ | `train/mwu.potential_sandwich` |
| MWU | двойная грань min(2η⟨ℓ⟩, ln n+ηℓmin) (R443) | ✔ | `train/mwu.mwu_double_upper` |
| MWU | выведенный η* = ln n/(2⟨ℓ⟩−ℓmin) — LR решён, не тюнится | ✔ | `train/mwu.eta_star` |
| MWU | Gibbs = hedge при η=1/T (R441): охлаждение = рост LR | ✔ | `train/mwu.gibbs_hedge` |
| MWU | статический regret, телескоп (R438–440) | ✔ | `train/mwu.static_regret` |
| Mem | admission_margin_sound: ε_cert — сертификат звучности (R419) | ✔ | `memory_cycle.admission_margin_sound` |
| RCDL | LiveDelta-семейство (exact bridge, J3, quant window) | ✔ | `model/livedelta.py` (порт 2026-10-14) |
| RV | round_viable/devEnergy-стоп/wallclock (GainOp-вертикаль) | ✔ | `train/round_viability.py` |
| Sp | spectralProjector/rankGrowth/sparseStep0 | ✔ | `train/spectral.py` (+функции) |

## Волна R510–R530 (порт 2026-10-16, после обновления primes)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| Th | variance-stop: Var ≤ Δ²(n−1)e^{−δ/T}, T ≤ δ/ln(Δ²(n−1)/ε) (R510) | ✔ | `thermostat.variance_stop_temperature` |
| Th | шаги остановки по дисперсии: k+1 ≥ ln(Δ²(n−1)/ε) (R512) | ✔ | `thermostat.variance_stop_steps` |
| Th | измеренный сертификат: Var, bound, стоп-T, шаги (R498/R510–512) | ✔ | `thermostat.variance_certificate` |
| AV | anytime-порог u = √(2σ²ln(T/α)) — uniform-уровень α (R525) | ✔ | `certified_controller.anytime_safe_threshold` |
| AV | anytime accept-гейт: accept ⟺ delta > u (R525×§9) | ✔ | `certified_controller.anytime_accept_gate` — закрыт открытый пункт «anytime в accept-гейте» |
| TC | конденсационный бюджет: девиация M → CE ≤ M²/4 (R522) | ✔ | `formal.condensation_budget_cost` |
| TC | обратный мост: бюджет ε → девиация 2√ε (R527) | ✔ | `formal.condensation_allowed_deviation` |
| TC | anytime-конденсация: монитор α ⟹ rebuild ≤ 2σ²ln(T/α) (R528) | ✔ | `formal.anytime_condensation_cost` |
| TT | risk-терминация: T ≤ (Φ₀−φmin+R)/ε, риск в числителе линейно (R529) | ✔ | `stage_telescope.risk_budget_termination` |
| GD | γ-бюджет: γ ≥ α·C*/D̄ обязателен для цели C* (R530) | ✔ | `certified_controller.gamma_min_for_target` |
| GD | γ-дефицит ⟹ потолок γ·D̄/α, взлёт невозможен (R530) | ✔ | `certified_controller.{gamma_ceiling,gamma_deficit_no_takeoff}` |

Тесты: `tests/test_primes_wave_r510.py` (17). γ-дефицит — то самое
открытие «γ-дефицит mixer.gain» из REFACTOR_PLAN: теперь у него есть
сертифицированный вердикт-гейт (γ измеряется, дефицит ⟹ единственный
инструмент — disagreement_distill / свежая diversity, не тюнинг).
