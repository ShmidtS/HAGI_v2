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
| Ly | BranchScale min-max / dead-layer stop / vocab V*=c / admission ΔVar/ΔC (Layers wave-5, Transfer wave-4) | ✘ удалено 2026-11-03 | `model/layer_tuning.py` удалён (0 src-импортов вне тестов; crescimento-скрипты не использовали; Chesterton-проверка git-истории выполнена). При необходимости восстановить из git ca6ecac^ |

Честные границы: h_emp_*-троица ЗАКРЫТА в primes (R563 lip, R564 grow,
R565 smooth — градиентная модель, Unified/DerivAlong.lean; порт 2026-10-18
— `train/lip_energy.py`); «мёртвый» код достижим через growth_supervisor
(альтернативная точка входа), чистых orphan-модулей нет — массовое
удаление отменено (Chesterton: каждый «выключенный» механизм
измерено избыточен на этом масштабе, но достижим конфигом).

Открытые runtime-пробелы (кандидаты на порт после формализации):
- ~~anytime-валидность в accept-гейте~~ — ЗАКРЫТО 2026-10-16
  (R525 `anytime_safe_threshold` →
  `certified_controller.anytime_accept_gate`, волна R510–R530);
- ~~certificate-telescope в `orchestrator/real_cycle.py`~~ — ЗАКРЫТО
  2026-10-16: каждое ACCEPTED поколение синтетической рекурсии пишет
  StageCertificate в stage_telescope.jsonl
  (`real_cycle._record_telescope_certificate`;
  tests/test_real_cycle_telescope.py);
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

## Волна R531–R534 (прочитана 2026-10-16, порт отложен)

| R | Модуль (primes) | Тип | Порт-решение |
| --- | --- | --- | --- |
| R531 | Unified/NonvacuityHorizon | theory-only | Свидетели-свидетельство nonvacuity; в рантайме нечего измерять — порт не требуется |
| R532 | Unified/FannesComplete | theory-only | Усиливает ценник KL≤Fannes в модельных мостах; меняет только константы — без runtime-поверхности |
| R533 | Unified/RecursiveClean | измеряемая интерпретируемость | Кандидат: clean-neuron сертификат (branchLocality/timeConsistency/indepMass) как телеметрия генераций; требует активационного анализа — отложено до стабильной лестницы (Chesterton: лестница только разморожена) |
| R534 | Unified/ModelComparison | измеряемый MDL-критерий | Кандидат: z-indicator (Δ_j = MDL-качество − λ·сложность, рекурсия vs суперпозиция) в selector-телеметрию; π̂_R πот k/N сертифицированных |

Честные границы прежние: h_emp_* не выводятся из Core;
anytime-Freedman (M∧V) — по-прежнему открыто в primes.

## Волна R535–R569 (порт 2026-10-18, после обновления primes)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| AV | anytime-Freedman лестница порогов (union bound по рунгам Freedman-термов, R541) | ✔ | `certified_controller.anytime_freedman_term / anytime_freedman_thresholds` |
| AV | двусторонняя |S|-лестница, фактор 2 (R542) | ✔ | `certified_controller.anytime_freedman_thresholds_two_sided` — ЗАКРЫТ последний открытый пункт «anytime Freedman M∧V» |
| GB | telescope_band / anytime_growth_budget: C_T ≤ C₀ + T(μ+u) (R548) | ✔ | `stage_telescope.anytime_growth_budget` |
| Th | Gibbs+Нернст композит: S(p_T) ≤ t̂·ln n + (1+t̂)·ln 2, t̂=(n−1)e^{−δ/T} (R549) | ✔ | `thermostat.gibbs_nernst_bound` |
| GB | затухающая полоса ⟹ capability-потолок C_T ≤ C₀ + b/σ (R556) | ✔ | `stage_telescope.geometric_band_ceiling` — мост к сигнатуре насыщения satC |

Theory-only (порт не требуется): R545 condensation_pipeline (композиция
уже портированных R522/R527/R528 в formal.py — новой измеряемой
поверхности нет), R547 (канонизация shannonEntropy — внутри Lean),
R550/R553 (аудит/export-чистка Lean), R554 certified_cone0_takeoff
(мост «margin → takeoff»: числовые компоненты — Hoeffding
`certified_ab` n-зависимость и γ-гейты R530 — уже в рантайме;
сама теорема связывает существующие сертификаты, отдельной
измеряемой поверхности не добавляет).

Тесты: `tests/test_primes_wave_r541.py` (14).

## Волна R560–R567 + LipEnergy (порт 2026-10-18, №2)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| GB | нижняя полоса роста: C_T ≥ C₀ + (b/σ)(1−(1−σ)^T) (R560) | ✔ | `stage_telescope.band_guaranteed_growth_floor` — полоса теперь ДВУСТОРОННЯЯ с R556 |
| GB | band_time_to_target: пол достиг C* ⟹ цель достигнута (R560) | ✔ | `stage_telescope.band_time_to_target` |
| GB | замкнутый лог-таймер цели: T = ln(1−σ(C*−C₀)/b)/ln(1−σ) (R567) | ✔ | `stage_telescope.time_to_target_log` |
| Lip | h_emp_lip: |E(w)−E(q)| ≤ κ·‖w−q‖, κ = sup‖∇E‖ (R563) | ✔ | `train/lip_energy.lip_energy_quantization` |
| Lip | h_emp_smooth: E(q+d) ≤ E(q)+⟨∇E,d⟩+L‖d‖²/2 (R565) | ✔ | `train/lip_energy.smooth_descent_energy_bound` |
| Lip | h_emp_grow: growGain = η·dnorm²/2 — измеримая константа (R564) | ✔ | `train/lip_energy.grow_gain_from_smooth_descent` (честный отказ при нарушении посылок) |

Theory-only: R555/R561 ч.①② (certified band-каскад —prob_ladder
поверх портированных Hoeffding-гейтов), R562/R566/R568/R569
(digamma-дуга пика Ω — d_eff-дисциплина, без рантайм-чисел),
R540 (grand_synthesis_weak — ослабленная передача посылок).

Тесты: `tests/test_primes_wave_r541.py` (22). Полная батарея: 709 passed.

## Практическое применение волны (2026-10-18)

- Двусторонняя полоса (R556↑ + R560↓) + `time_to_target_log` —
  каркас сертифицированного прогноза лестницы: измеренная
  затухающая поддержка b, σ даёт и потолок, и срок достижения C*.
- LipEnergy: κ = sup‖∇E‖ измеряется по градиентным нормамjoint-фазы;
  `grow_gain_from_smooth_descent` даёт сертификат для
  StageCertificate.decrease без эвал-пересчёта (honest rejection при
  нарушении окна SafeQP).

Телеметрия kappa (2026-10-18): `lip_energy.kappa_from_grad_norms`
(честный sup измеренных градиентных норм — нижний свидетель sup‖∇E‖)
+ `lip_energy.lip_stage_certificate` (R563-квантизация + R564-grow-gain
одним вызовом; honest rejection). Источник измерений — пер-доменная
градиентная телеметрия train/loop.py (h_emp_ поставщик).

Осознанно НЕ портировано (YAGNI, нет runtime-потребителя — по
прецеденту REFACTOR_PLAN attic-решений):
- ProbeError (мост Boost→Cut) и DHC HypothesisCompetition —
  research-суррогаты, контроллер цикла их не вызывает;
- SoftmaxStep (logit-CE модель, R350/R354) — частный случай уже
  портированного SafeQP/joint_stage пути (L=1/2);
- GaugeQuotient (R537/R538) — looped-reasoning early halting,
  применимо только при внедрении циклических ядер (не в линии);
- R533/R534/R535/R536 interpretability-стек — отложен по Chesterton
  до стабильной лестницы (нужен активационный анализ).

## cut_safe_accept интеграция (2026-10-18, вечер)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| PE | safe_certificate_margin: dL_exact ≤ dL_meas + ε‖d‖ (Коши-Шварц) | ✔ | `lip_energy.cut_safe_accept` (композиция) |
| PE | clamped_attention_grad_bound: член ≤ L_att·ε_att | ✔ | параметр (eps_att, L_att); =0 пока attention-quant калибровка не измерена — честный ноль |
| PE | cut_safe_accept / probe_safe_accept: margin-приём шага | ✔ | `lip_energy.cut_safe_accept` + опт-ин проводка в `loop._apply_certified_step` (`train.certified_cut_accept`, default off; CUT-VETO логируется) |

Также портированы: StochasticSafeQP noiseEps + lr_batch_equivalence
(`safeqp_step.noise_eps/lr_batch_equivalence_check`), BitAlloc
(`train/bit_alloc.py`: transfer_exact/factor-2/water-filling),
WSqD+NoiseTemperature (`growing_batch`: batch_variance, wsqd_noise_budget,
flatness_moment_bound), TernaryExact satTail-пакет (`model/ternary`:
ternary_sat_diag, optimal_ternary_scale — R103-насыщение, 1.9× снижение
split-bound на R103-синтетике). Тесты: `test_primes_wave_r541.py` — 44.
Полная батарея: 731 passed.

## Дефолты по теории + Ville-стек (2026-10-18, ночь)

| Изменение | Статус | Место |
| --- | ------ | ----- |
| cut_safe_accept DEFAULT ON (ProbeError: сертификация точного изменения потери с Коши–Шварц margin; честный CUT-VETO) | ✔ | `config.certified_cut_accept=True` + `loop._apply_certified_step` |
| use_muon DEFAULT ON (MuonStableLR R379-382: шаг выведен, не настроен) | ✔ | `config.use_muon=True` |
| liveness two-axis DEFAULT ON (Unified/Liveness: честный стоп — часть сертифицированного цикла; окно 32, min_steps 64) | ✔ | `config.liveness_window=32` |
| batch-термостат DEFAULT ON (NoiseTemperature + WSqD: ось batch — ось охлаждения шума; delta=1.02, cap=16 обязателен по теореме) | ✔ | `config.batch_growth_delta=1.02/cap 16` |
| Ville/anytime-стек (AnytimeMartingale: anytime_valid_budget, schedule_from_budget, anytime_safety_check, stochastic_noise_schedule) | ✔ | `certified_controller` |
| ternarize_split (TernaryExact + R103: satTail-осведомлённый масштаб с честным win-check; BitLinear.cache_quantized default split) | ✔ | `model/ternary.py` |

Оставлены off с измеренными причинами: certified_step-режим (полная
замена шага на d* — не имеет живого A/B; analytic_step-порт измеренно
проиграл +1.02 CE), thermostat-CLI (нужен измеренный t0),
ternary_fp32_master. Волна R560-567 + LipEnergy + WSqD + BitAlloc —
см. выше. Тесты: `test_primes_wave_r541.py` — 47; батарея 734 passed.

## Волна: Joint/Armijo/Warmup + Pinsker-CMI + capacity-этажи (2026-10-18, ночь-2)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| J | jointStep_conflict_regression: граница 2|gc·g|/(L(g·g)) — всё ниже строго вредит | ✔ | `safeqp_step.conflict_lr_limit` |
| J | jointStep_regression_bound: |lr|√gc·√g + (L/2)lr²‖g‖² — верхняя огибающая конфликта | ✔ | `safeqp_step.joint_step_regression_bound` |
| J | trustRegion_bound: √(gc·gc)·R + (L/2)R² — независимо от направления | ✔ | `safeqp_step.trust_region_loss_bound` |
| A | armijo_step_size: α ≤ c₁/L + g·d ≤ −c₁‖d‖² ⟹ J ≤ J₀ − (c₁/2)α‖d‖² (шаг из теоремы, не backtracking) | ✔ | `safeqp_step.armijo_accept` |
| W | warmupEta = c·min(1/K₀, Δ^(−ρ)/(3^ρ·K_ρ)); монотонность: Δ↓ ⟹ η↑ (следствие гладкости) | ✔ | `safeqp_step.warmup_eta` |
| P | gen_pair_bound: |E_qf − E_pg| ≤ E_q|f−g| + 2√KL — декомпозиция stability+information | ✔ | `formal.gen_pair_bound` |
| P | cmi_iteration: (1/n)ΣΔ ≤ (1/n)Σstab + 2√((1/n)ΣKL) — Cauchy-темп √(CMI/n); неравные KL — штраф (концентрация массы KL наказывается) | ✔ | `formal.cmi_iteration` |
| RC | router_capacity_bound / separated_needs_capacity: K ≤ 2^B — декодеры не создают информацию | ✔ | `formal.router_capacity_ok` |

Инвентаризация из Lean-кода завершена (263 модуля, 4 чтеца + сверки);
все runtime-значимые формулы с точными константами либо портированы
(≈80% контента), либо осознанно отложены с причиной (Dust, PPT,
GaugeQuotient, DHC, interpretability-стек). Тесты волны: 62; батарея
749 passed. Применение: conflict_lr_limit — защитник адаптеров от
регрессионной зоны; cmi_iteration — кандидат в gen_telemetry для
сертификата обобщения; armijo_accept/warmup_eta — замена ручных
расписаний при сертифицированных шагах.

## SNR/NCE-батч (2026-10-18, ночь-3)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| SNR | focused_beats_uniform: (dev_j)² < mean ⟹ фокус-вектор строго лучше uniform (свидетель-теорема: вся масса на лучшего) | ✔ | `safeqp_step.focused_wins / focused_cost` |
| NCE | adaptiveK_bound: K ≥ moment/eps — число noise-сэмплов выводится из измеренного второго момента | ✔ | `safeqp_step.adaptive_k` |
| NCE | anchor_cadence_criterion: s ≤ γ·ε/ρ ⟹ дрейф ρs/γ ≤ ε — каденция переякоривания выведена | ✔ | `safeqp_step.anchor_cadence` |

Тесты волны: 67; батарея 754 passed.

## Plasticity + skip/renorm (2026-10-18, ночь-4)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| PL | plasticityRadius = ΣKL(P_i‖famMix) ≤ |I|·log|I|; 0 ⟺ семья одинакова (E_dev-прокси на уровне распределений) | ✔ | `certified_controller.plasticity_radius` |
| U | skip_bound/sink_cost_bound: L1-цена скип/рёнормировки ≤ 2·сброшенная масса — первого порядка по МАССЕ, не по длине окна | ✔ | `mwu.skip_renorm_bound` |

Сверка шаблона gen_configs: batch_growth (1.0055/cap 16) и use_muon
пишутся явно; liveness/сertified_cut_accept не пишутся — наследуют
theory-дефолты (32/True) с gen6. compileall по src+scripts: чисто.
Тесты волны: 74; батарея: 761 passed.

## PoE-вертикаль (2026-10-19, ночь)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| PoE | poe_logZ_second_order: |logZ_pooled − Σw_i logZ_i| ≤ (1/8)Σw_iR_i² — острая форма по пер-экспертному размаху центрированной девиации | ✔ | `formal.poe_logz_second_order` |
| PoE | poe_softmax_speed: равномерный размах M ⟹ ≤ M²/8 на токен (K=2 восстанавливает pairwise с серединой) | ✔ | `formal.poe_softmax_speed` |
| PoE | poe_logZ_pairwise: честная ½-константа ≤ ½ΣΣw_iw_jD_ij² — ⅛-pairwise ОПРОВЕРГНУТА в Lean (z₁=(1,−1), z₂=(0,0), w=½) | ✔ | `formal.poe_logz_pairwise` |

Runtime-применение: certified merge-гейт для PoE-пулинга логитов
(сейчас merge использует усреднение с Jensen-gap; PoE-гейт — готовая
поверхность для взвешенного пулинга с сертификатом). Тесты волны: 79;
батарея: 766 passed.

## LSE-shift / KL / InfoNCE / D-field батч (2026-10-19, ночь-3)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| PoE | lse_shift_lower/upper: lse(y)−lse(x) = ⟨σ(x),y−x⟩ + r²/8 (первый порядок — нижняя грань, Hoeffding-MGF — верхняя) | ✔ | `formal.lse_shift_upper_bound` |
| D | kl_product: KL(p₁×p₂‖q₁×q₂) = KL₁+KL₂ — совместный бюджет = сумма пер-позиционных | ✔ | `formal.kl_product_additive` |
| D | dfield_smoothing: (1−α)p+αu > 0 почленно — мост к full-support KL-теоремам | ✔ | `formal.smooth_support` |
| I | infonce_gap_is_kl: зазор MI−InfoNCE = KL(post‖model) ТОЧНО (инвариант-верификатор калибровки) | ✔ | `formal.infonce_gap_is_kl` |
| I | infonce_never_overestimates: оценка не переоценивает пол (CE ≥ H) | ✔ | `formal.infonce_never_overestimates` |
| D | dfield_entropy_identity: D(w) = H(p_w) − Σw_iH(p_i) — D из измеренных энтропий без тачанья распределений | ✔ | `formal.dfield_entropy_identity` |

Тесты волны: 89; батарея: 776 passed.

## NCE-оценщик + KLSSafeQP сертификат (2026-10-19, ночь-4)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| N | partEst_unbiased: E_q[impW] = Z точно при full-support q (несмещённость NCE-оценки партионной функции; Clippage даёт bias -> smooth_support) | ✔ | `safeqp.part_est_unbiased` |
| N | secondMomentW + secondMomentW_ge_sq: M2 = Σq·impW² ≥ Z² (Koshi-Shvars); драйвер дисперсии M2/Z² ≥ 1 | ✔ | `safeqp.second_moment_w`, `nce_variance_ratio` |
| K | klsBatchNoise: V = C·L²·sqN(a)/B (Пуанкаре + B некоррелированных проб) | ✔ | `safeqp.kls_batch_noise` |
| K | klsSafeQP_certified: масса acceptance ≥ 1 − V/t² (Чебышёв по полю девиаций) — сертификат рядом с SafeQP-veto | ✔ | `safeqp.kls_safeqp_certified` |

Тесты волны: 95.

## Insight/LSL-мост вертикаль (2026-10-19, ночь-5)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| A | forgetting_kl_bound (R329): kl ≤ curv/2 ∧ curv ≤ 2ε ⟹ kl ≤ ε — сертификат забывания из эмпирической кривизны | ✔ | `lip_energy.forgetting_kl_bound` |
| A | insight_consolidation_safe: шаг консолидации в золотом конусе + descent-предикат — два вето для merge/root-ft шага | ✔ | `lip_energy.consolidation_gate` |
| A | gate_retention (§6.3): T-шаговый дрейф по золотому направлению ≥ −T·ε — кумулятивный бюджет поколения из пер-шаговых сертификатов | ✔ | `lip_energy.gate_retention_bound` |
| A | tldr_two_tier_safety: low-rank адаптер == 0 на ядре B + KL-цепочка — двухъярусная безопасность | ✔ | `lip_energy.tldr_two_tier_safety` |
| A | insight_kl_descent / experience_cycle_bound — арифметические тождества (ce_gap_kl_identity уже портирован как cmi/CE-стек; циклы — linarith) | — | покрыто существующими `kl_excess`, budget-функциями |

Тесты волны: 99.

## Merge-цена satTail-optimal (2026-10-19, утро)

| § | Алгоритм | Статус | Место |
| --- | -------- | ------ | ----- |
| §3 | pilot_latent_merge: цена сжатия s_val теперь из satTail-оптимального поиска (min n(s/2)²+satTail, R103) вместо max\|Δ\| — win-gate: только когда поиск сертифицированно снижает split-bound; иначе точный max\|Δ\| | ✔ | `scripts/growth/pilot_latent_merge.py` (встроено; применяется с gen7-merge, gen6 merge уже прошёл) |

Тесты волны: 99 (скрипт покрыт import+py_compile; численная win-проверка в ternary-тестах).

## Мониторинг gen5→gen6 + merge-оптимизация (2026-10-19, утро-2)

- gen5 ПРИНЯТ: delta 0.324 (2.7x лучше gen4 0.119), incumbent gen5_root_ft/best.pt, stage-certificate в telescope.
- gen6 обучается (листья math/code завершены, ru в работе); divergence-stop на gen6_leaf_code (шаг 355) — тот же паттерн, что gen5_leaf_code (механика идентична, не регрессия).
- token_weight_bias метрика: измеренная структурная вырожденность (cov=0 тождественно из-за общего скалярного gnorm) задокументирована в комментарии loop.py; пер-доменные нормы = отдельные backward — YAGNI.
- Сравнение листьев gen6 vs gen5 (parents разные, вердикт решит A/B): math +0.0008, ru +0.0379, code +0.4886 (оба code-листа имели divergence-stop).

## Скорость обучения: growth-batch протечка в листья (2026-10-19, утро-3)

**Измерено**: gen6 листья 8.03 с/шаг против 0.54-0.62 (gen1/gen3/gen5) — ×13 wall-clock.
**Причина**: дефолт config.py `batch_growth_delta=1.02` (<<включи всё по теории>>) протёк
в листовые конфиги gen6 (генератор пишет batch_growth ЯВНО только в root_ft — там 1.0055,
спроектирован против divergence в ft). Leaf accum рос 1→16 (cap), т.е. до 16 fwd+bwd на шаг.
**Качество не выросло**: gate_ce листьев gen6 vs gen5: ru 5.605 vs 5.567, math =, code хуже.
**Фикс**: config.py `batch_growth_delta=None` (быстрый дефолт; ft-конфиги задают 1.0055 явно).
Урок: dataclass-дефолт = неявная конфигурация всех, кто поле не пишет.

## Архитектурная рамка: RNN-бэкбон как сертифицируемая ветка (2026-10-20)

Аудит primes (265 модулей, 2148 theorem, 0 sorry; ALGORITHMS.md +
STATUS.md): прямого предписания «attention vs рекуррентность» в
теории НЕТ. Контракт теории архитектурно-агностичен:
- рост ранга ортогональными направлениями — R360
  `rank_orthogonal_growth` (Hagi/Spectral/RankGrowth.lean; «R242»
  в primes отсутствует — номер из внутренней нумерации репо HAGI_v2);
- latent merge (GapLaw/MergeBridge), certified-гейты, SafeQP,
  h_emp_*-измерения — операционный контур, работают с CE-дельтами
  и state_dict-тензорами.
Косвенные якоря рекуррентности: `recurrence_upper`
(Foundations/Recurrence.lean:41, контрактная оценка x(t+1)≤ρx+c),
Перрон-конус (Growth/ConeDynamics), ε_attention-члены
(Pretraining/ProbeError.lean) — ограничивают динамику, не выбор
seq-оператора.

Решение (Oracle 2026-10-20): RNN-переход = ПАРАЛЛЕЛЬНАЯ ветка через
те же гейты, не слом V42-линии. Бэкбон-кандидат: gated linear
recurrence (LRU/DeltaNet-класс) — параллелизуемый scan, O(1)-состояние,
линейность по параметрам сохраняет per-tensor latent_merge и
zero-init function-preserving depth-grow. Замена инкумбента — только
через §9 certified_ab (равный токен-бюджет, не шаги). «Growing
memory» (arXiv:2602.24281) реализуется системно — поколениями и
rank-growth, не в клетке.

## RNN-ветка gen1: измеренные факты (2026-10-21)

- Листья RNN (5 доменов, общий init, 7 слоёв H=128): train CE
  math 5.98 / code 7.22 / ru 6.79 / enedu 6.45 / enwiki 6.12.
  Eval (30 батчей): RNN math MATH 5.480 vs attention-fresh 5.427
  (паритет, delta 0.05); AVG 6.60 vs 6.32.
- enweb (чистый slimpajama): divergence на шаге 255/305 при любом
  lr (3 попытки, 2 стратегии — домен исключён из RNN-ветки; ось
  мертва, liveness-стоп).
- Latent merge RNN: точный (root CE 6.53 = mean листьев),
  mean|cos| дельт 0.099 — дельты ортогональны (MergeBridge
  работает на recurrent-тензорах без изменений).
- root_ft 500 шагов r32: CE не улучшил (Γ_ft ≈ 0.05–0.1 —
  интеграция, не рост). Вывод (Oracle 2026-10-21): рост —
  в листьях от prior на свежих data-offsets (Γ_grow ≈ 0.3–0.4),
  не в ft-шагах.
- Gen2 (по Oracle): 6 листьев от root_ft prior (вкл. books
  2400 шагов — свежая ось), ft r64 600 шагов, batch-growth 1.0055,
  λ=0.3 (grokking-выведено), Muon+AdamW. Цикл: leaves→merge→ft→
  certified A/B; REJECT = смена рычага, не повтор.

## RNN-ветка gen2: ACCEPT (2026-10-21)

- Лестница: rnn-1 AVG 6.3304 → **rnn-2 AVG 5.8289** (delta −0.5015,
  n=4.9M токенов/домен, ε≈0.0005 — ACCEPT за пределами сертификата
  на 3 порядка; регрессий нет ни в одном домене, включая оба
  held-out).
- Работавший рычаг (Oracle-предсказание подтверждено Γ_grow≈0.3–0.5):
  листья от root-prior на свежих data-offsets (+2xx), books-лист
  (свежая ось, 2400 шагов), ft r64 600 шагов batch-growth 1.0055.
- saturation-stop ft на 505/600 — термостатный стоп корректен
  (Var-гейт сработал раньше бюджета).
- Дивергенции листьев math/books (355/шаг) остановлены гвардом,
  best.pt валидны — механизм honest-stop работает.
- Следующий рычаг gen3 (по контуру): те же 6 листьев от gen2
  prior + новые offsets (+4xx), ft r64. При REJECT (delta<
  margin 0.02) — смена рычага: рост числа доменов (sci) или
  ранга r128.

## Волна Optimization (порт 2026-10-21, дословно из Hagi/Optimization)

Новый модуль теории «три геометрии» (ит. 43–52) + Energy.Variational
(термокоридор) — порт `train/adaptive_cycle.py` (12 тестов):
- welch_min_dimension / welch_capacity_full / welch_required_dimension
  (DirectionPacking:273,299) — Уэлч-триггер: M направлений с
  когерентностью μ вмещаются в d iff (M+1)/(1+Mμ²) ≤ d; исчерпана →
  расти рангом, не листами. Измерено на gen3-волокнах: μ=0.70
  (flattened), r64 НЕ исчерпана (req_dim=2 при M=6).
- proj_residual / proj_residual_ratio (SubspaceGrowth) — новое
  направление легитимно iff ‖r‖/‖v‖ ≥ δ (невязка к текущему базису).
- growth_finite (adaptive_cycle_expansions) — N ≤ C₀/δ расширений.
- inexact_contraction_halt_k / contraction_floor
  (InexactContraction) — явный горизонт останова ft:
  q^K < (τ−ε/(1−q))/B; пол ошибки ε/(1−q).
- angular_damping_window / chordal_residual (SphericalDamping) —
  окно шага 0<η<2/(1−ζ); критерий остановы 1−⟨u,v⟩=½‖u−v‖².

RNN-лестница: rnn-1 6.330 → rnn-2 5.829 (ACCEPT −0.50) →
rnn-3 5.545 (ACCEPT −0.284, все 6 доменов улучшились).
Замедление темпа (−0.50→−0.28) — конус насыщения (§8 теория);
по cert_horizon_log_bound следующее поколение при том же рычаге
даст ещё меньше — кандидат на смену рычага: рост M (новые домены)
или ранг, по Welch-триггеру.

## gen5 REJECT → gen6 ACCEPT (свежие данные) → v3 fast-cycle (2026-10-24)

- **rnn-5 REJECT**: LR-полка −30% — листья хуже на 4/6 доменов; урок:
  дивергенция листьев ≠ «L вырос». End-merge снова сильнейший кандидат.
- **rnn-6 ACCEPT**: avg6 5.376 → **5.281** (Δ−0.095, ВСЕ 6 доменов,
  RU −0.29, CODE −0.13, held-out −0.04..−0.05) — рычаг СВЕЖИХ ДАННЫХ
  (--start-frac 0.3, seed_only_grow_stop) подтверждён certified A/B.
- **v3 fast-cycle** (редизайн по полной перечитке + bf16-профиль):
  head-CE = 29% шага → ce_keep_rate 0.5 (−14%); КОРОТКИЕ циклы
  (WallClockTakeoff: best листьев на шагах 150-400; бюджеты срезаны
  math 700/прочие 500/books 1200/ft 400); ПАРАЛЛЕЛЬНЫЕ листья
  (3 concurrently, измерено 0.71 с/шаг на лист при 3× — агрегат
  4.3 шага/с против 1.5 последовательных = 2.9×); ft с KD от
  инкумбента (alpha 0.25, cancellation-fixed); ротация frac
  0.3+0.15·(gen−6); noise-probe телеметрия. Цикл: ~2ч → ~40мин.
  gen7 = первое v3-поколение (prior/учитель = rnn-6 end_merged,
  frac 0.45).



## Саморазвитие: evolve.py — замкнутый сертифицированный контур (2026-10-25)

Заменяет ВСЕ одноразовые карусели: контроллер в одном процессе сам
принимает каждое решение по теоремам, человек не участвует:

    SPECIALIZE (fresh-segment ft -> trajectory end-merge -> certified
    A/B; ACCEPT меняет инкумбент, SSOT checkpoints/incumbent.pt)
    WIDTH-GROW при 2 stall-циклах (growth_dichotomy; width_grow
    переносит лестницу функционально-сохраняюще)
    STOP по grow_epsilon_stop.

Измеренная история первого полного прогона (4 поколения):
    256-класс: 5.4580 -> 5.2951 ACCEPT -> 5.2117 ACCEPT
    (лучше старого 128-инкумбента 5.281 на 0.07 certified)
    gen3/gen4 REJECT (конец свежих данных frac 0.9)
    -> контроллер сам width-grow до 512 (42.5M), SSOT готов.

Технологии, на которых стоит контур (все — машинно-проверены):
  * width_grow: bit-for-bit перенос; RMSNorm форсирует √2-рост
    (rms' = rms/√2 при нулевых новых каналах) ⟹ инвариант потока
    S' = √2·S; ветки берут √2 от своих норм (out_norm состояния,
    down·(1/k)); читатели потока comp k; fused in_proj пересобирается
    по третям; emb-шум 1% оживляет градиентный каскад (строго-нулевые
    приращения мертвы: все пути через ноль);
  * fixed CE (2610.31642), cancellation-fixed KD, end-merge (noise-порт),
    certified A/B (Hoeffding + no-regression clause), growing batch,
    liveness/saturation/divergence-гварды, noise-probe телеметрия.

Инфраструктурные находки (все закрыты в коде): GPU-lock (один конвейер,
проверено в обе стороны), SSOT-инкумбент вне одноразовых каталогов,
п per-generation очистка артефактов, width-aware concurrency.

## gen7 REJECT (насыщение класса подтверждено) → gen8: расширение класса M=9 (2026-10-24)

- **rnn-7 REJECT** (5.288 vs 5.281): v3 fast-cycle на том же классе M=6/
  w128 — ТРЕТЬЯ точка потолка (−0.169, REJECT, −0.095, REJECT):
  growth_dichotomy подтверждён сертифицированно.
- **gen8 (M=9: +smoltalk/tinystories/wikien)**: УРОК — сломались ровно
  ОБЩИЕ en-миксы (enedu: edu .6/slim .4 → best 7.33; books: slimp .7 →
  8.96; CE растёт с шага 0 при здоровом старте 5.59): лист, чьи данные
  prior уже знает, ДЕГРАДИРУЕТ (wd 0.3 + высокий LR без свежего
  сигнала = чистый распад). Специалисты со свежими корпусами здоровы
  (tinystories 4.94, wikien 5.04, smoltalk 5.97). Правило дизайна
  листьев: КАЖДЫЙ лист — специалист на СВЕЖЕМ домене
  (domain_disagreement_floor: выигрыш слияния гарантируют именно
  ортогональные специалисты); общие миксы не переобучать каждое
  поколение. gen8b = merge из здоровых листьев (c8-специалисты +
  g7 enedu/books).
- Консолидация кода: cycle.py + certify.py (одны параметризованные;
  13 одноразовых скриптов удалены). Рычаг роста — флаги: --domains
  (M↑), --leaf-width (новый класс), --frac (свежесть).



## Полная перечитка Lean #2 (2026-10-24): ВСЕ каталоги, ~250 новых результатов

Четыре чтеца покрыли весь корпус (Core/Data/Ensemble/Probability 61
файл; Information/Foundations/Model/Architecture 30; Growth/Dynamics/
Unified 64; Autonomy/Discovery/Pretraining/Omni/Spectral/External/
Percolation/Examples/Prelude 35). Вывод: runtime-ядро (~90%) уже
портировано волнами R419–R568; ниже — новое с поверхностью действия.

### Портировано немедленно (train/grow_arithmetic.py, 15 тестов)

| § | Теорема (файл) | Порт |
| --- | -------------- | ---- |
| GA | `certifiedGain = (M−c)/(N+1)` + `grow_epsilon_stop` (Growth/Grow) | арифметика добавления листьев: выигрыш лучшего < ε ⟹ сертифицированный СТОП перебора кандидатов |
| MS | `twoGap_bounded`/`merge_skip_certificate` (Ensemble/Hoeffding): twoGap ≤ ½log cosh M ≤ M²/4 | скалярный пред-гейт merge: M²/4·η < ε ⟹ SKIP цикла (одна посылка вместо прогона) |
| DF | `domain_disagreement_floor`: одна пара δ ⟹ twoGap ≥ ½log(1+p_u p_v(cosh δ−1)) | гарантированный пол выигрыша слияния |
| NF | `disagreement_convergence_floor` (NoiseDisentangle): пол σ²/N | twoGap ниже σ²/N — шум, раунд исчерпан; N — рычаг денойзинга |
| SO | `seed_only_disp_bound`/`seed_only_grow_stop` (SeedOnly): идентичный init ⟹ дисперсия ≤ Λ²s·Σa^i, цель недостижима | ДИАГНОЗ замедления лестницы: gen-N листья от prior без свежих данных затухают геометрически; gen6+ — свежие data-offsets / новые домены обязательны |
| CB | `sfo_global_min` b*=9Y²/(4(ε−Z)²) (Budget/BatchCap) + `split_batch_noise_probe` (NoiseTemperature batch_variance_eq): σ̂²=‖g1−g2‖²/2 за одну пробу | кап batch-роста ВЫВОДИТСЯ из измеренного Y (log-only зонд в loop.py, `train.noise_probe_interval`) |

Также исправлен дефект: `saturation_min_steps` игнорировался в
стоп-условии (ранний стоп до стартового окна; best.pt-трекинг
намеренно с шага 1 — разные вопросы).

### Кандидаты (задокументировано, не срочно)

- **Спектральный слой** (Flywheel port, MatrixNormRelations/
  CoherenceCharacterization): точное тождество ‖A−A_k‖₂ = σ_k(A);
  coherence без SVD: μ²·rank = (‖E‖_F/‖E‖₂)²; erank = exp(H(σ²/Σσ²))
  — непрерывный измеритель ранга для spectral_compress; гэп-критерий
  decision_criterion (σ₁−σ₂ > ε‖E‖_F ⟹ дешёвый SVD достаточен).
- **Одно-логитные пробы** (Pretraining/Speedrun cePartition_factor):
  Z' = Z(1+w_j(e^δ−1)) — O(1) обновление партиции (та же математика,
  что noise-репо и наш fixed-CE); кандидат для insight-канала.
- **Гиттинс-правило** (External/Diversity): рычаг = argmax Δquality/
  Δcost, стоп когда ни один не бьёт инкумбента.
- **Chinchilla-член (b·N)^α** (SupervisorSafety capMember): тернарный
  b=log₂3 строго бьёт бинарный при равном N — формальная поддержка
  тернарной линии.
- **Pinsker-семейство усиления**: tv_le_pos_part_kl (линейный KL ≥ TV
  при малых KL), Бретаньоль–Юбер при больших — точные концы для
  KD-стопов.
- **WallClockTakeoff**: малые циклы доминируют (slack (k²/2)Στ²) —
  параллелить поколения мельче.
- **gibbs_posterior_ratio**: кандидат ниже лидера на δ несёт ≤ e^{−δ/T}
  массы — прунинг при сэмплировании ансамбля.
- **measured_gain_horizon** (ProbeError): k ≤ (E0−Emin)/gain — кап
  продуктивного горизонта прогона из измеренного спуска.



## Перечитка Lean #1 (2026-10-23: Optimization/Step + Energy/System/Runtime/Budget + выборочно Growth)

Аудит 178 результатов против портов. Вывод: ~90% runtime-значимого
уже портировано волнами R419–R568 (подтверждено поимённо). НОВОЕ с
runtime-поверхностью (кандидаты, не срочно):

| § | Результат | Теорема (файл) | Решение |
| --- | --------- | -------------- | ------- |
| BC | критический батч b* = 9Y²/(4(ε−Z)²), кап B_cap = C·b*²/A; A/N+C/K → N*=√(AB/C), K*=√(CB/A) | `sfo_global_min`, `optimal_split` (Budget/BatchCap) | кандидат: ВЫВЕСТИ batch_growth_cap из измеренного шума Y вместо хардкода 16; нужен измеритель Y (split-batch дисперсия градиента) — отдельный раунд |
| QT | LS-шкала квантователя s₀ = Σqx/Σq² (замкнутая форма; чередование snap↔LS монотонно) — «90% PTQ-выигрыша» | `lsScale_optimal`, `lsScale_gram` (Energy/Ternary) | кандидат для ternary-линии (RNN-ветка не квантуется); текущий sweep по split-bound оставить до A/B |
| TR | SafeQP trust-radius ‖d‖ ≤ √(2ε/L): ε-бюджет → 2ε-худший случай | `safeqp_trust_region` (Step/SafeQPStep) | для certified_step-режима (выключен в линии); поверх уже портированных гейтов |
| MO | импульс: β=0 оптимален в законе ошибки; скорректированный Нестеров (q+β)/(1+β), пол δ/(1−q) НЕ зависит от β | `momentum_optimal_beta`, `nesterov_floor` (Optimization/Momentum) | подтверждает Muon nesterov=True (скорректированная форма) — изменения не нужны |
| TC | термокоридор T ∈ [s/τ, ε/ln n] — двусторонняя граница температуры ансамбля | `temperature_corridor` (Energy/Cooling) | актуален при включении гиббсовских весов ансамбля в runtime (unified_step); сейчас не используется |
| EndM | end-merge вдоль траектории: EMA-банк + покомпонентный sweep, приёмка iff held-out ↓ | qlabs-eng/noise train/val/run.py (record #11) | ПОРТ: `scripts/growth/end_merge.py` (EMA по step-*.pt, per-tensor f×s sweep, 3 координатных прохода, gate-accept); ft gen4 пишет банк ckpt interval=100/keep=8 |

Также из noise/run.py задокументированы (та же математика, что наш
fixed-CE порт): замкнутые тождества возмущений log1p(p·expm1(Δ)) для
одно-логитных/одно-ключевых проб (measurement-инструмент для
credit-оценки без backprop — кандидат для insight-канала при
внедрении); ANVIL/warmdown-рецепты — чужой режим, не переносим.

## Волна fixed-CE (порт 2026-10-23, arXiv:2610.31642 Barron et al.)

Аудит числового ядра обучения против статьи «A Fix for Softmax and
Cross-Entropy» нашёл ОБА бага статьи живыми в этом репо:

| § | Баг | Где был | Фикс | Место |
| --- | --- | ------ | ----- | ----- |
| SC | Softmax Collapse: lse-сумма `1+e^{-m}` округляется (bf16 клифф ≈5.5, fp32 ≈16.6) → loss=0, g_t=0, Σg>0 → Slingshot-спайки под AdamW | `_ChunkedCrossEntropy` fwd+bwd (fp32-акк → клифф 16.6); fused-путь `F.cross_entropy` (bf16 → клифф ~5.5) | партиционная сумма ИСКЛЮЧАЕТ целевой логит (scatter 0 + fp32-акк): loss = softplus(-m), m = z_t − lse_{j≠t}, точность до underflow (m≈88); g_t = −sigmoid(−m) — Σg = 0 точно | `model/head.py` `_ChunkedCrossEntropy` |
| SC | fused-путь | `LMHead.loss` fused ветка | softplus-margin форма (Eq. 6 статьи) через autograd: g_t = −sigmoid(−m), g_j = p^nt_j·sigmoid(−m) — сумма строки 0 точно | `model/head.py` fused ветка |
| SC | sampled-путь | `_loss_sampled` (K+1 кандидатов) | m = z_target − lse(негативы) (слайс уже исключает цель) | `model/head.py _loss_sampled` |
| SC | sampled softmax: −log_softmax(logits)[:,0] | log_softmax округлялся за клиффом | softplus(−m) на кандидатном наборе | там же |
| Canc | Softmax Cancellation: KD-градиент q−p из раздельно округлённых вероятностей — ненулевой пол при z_S→z_T (студент дрейфует от учителя) | `logit_kl` (F.log_softmax ⊖ F.softmax) + учитель передавался как log-probs (двойной softmax при T≠1) | `soft_target_kl`: кастомный градиент из РАЗНОСТИ ЛОГИТОВ через expm1/log1p (Eq. 9 статьи), Σg=0 форсирован; учитель возвращает сырые логиты; T применяется к обеим сторонам + T² | `train/distill.py soft_target_kl`, `train/loop.py` KD-канал |
| ME | Margin equilibrium m*≈log(1/ε) (§5 статьи): под AdamW+wd маржа запоминаемых примеров растёт до равновесия | — не измерялось | `head.mean_logit_margin` в gate-eval каденции: телеметрия `gate_margin`; m→m* (18.4 при ε=1e-8) = сертификат завершения меморизации | `model/head.py`, `train/loop.py _gate_ce` |

Диагноз прошлых дивергенций: остановы листьев gen5/gen6 (code 355,
math/books 355, enweb 255/305) — паттерн Slingshot (спайк loss +
обвал) на клиффе точности; фиксированный CE по статье устраняет
механизм (m* = 18.4 > fp32-клиффа 16.6 — старый CE гарантированно
падал с обрыва до равновесия). grokking-расписание λ=0.3 (R366)
остаётся: статья подтверждает, fixed CE УСКОРЯЕТ grokking с weight
decay.

Численная проверка: тесты `tests/test_fixed_ce.py` (14) —
воспроизведение клиффа (старый loss ровно 0 при m=20, новый
log1p(7e−20)), нулевая сумма градиентов строки ТОЧНО, целевой
градиент жив за клиффом, gradcheck fp64, пример-числа статьи для KD
(совпадение с точным fp64-градиентом), grad ≡ 0 при z_S = z_T.
Батарея: 830 passed. Цена: bf16 chunked 54.3ms против ~45 (fwd+bwd,
[N,V]=[8192,32768]) — ~+20% на head, ~+6% на шаг.

Листья gen4 — первое поколение на фиксированном CE (тот же рычаг:
6 листьев от gen3 prior, адаптер-базис init_seed +200, ft r64 600).

## RNN-лестница: gen4 (2026-10-23, первое поколение на fixed CE)

Листья gen4 vs gen3 (gate_ce на фиксированных окнах, тот же prior):
math −0.20, code −0.35, ru −0.12, enedu −0.06, books +0.02,
enwiki +0.24 (дивергенс-стоп на 255; фикс CE ускоряет рост маржи
до равновесия m* — LR-полоса 0.35–0.38·H⁻¹ становится горячей,
дивергенс-гварды честно фиксируют best). Merge: mean|cos| дельт
0.127 (лучше gen3 0.167). ft 355 шагов (saturation). End-merge
(порт noise-репо): −0.051 gate, 33 хода.

Certified A/B (common-150b, оба плеча один протокол):
AVG 5.6730 (gen3, переизмерен) → 5.4836 (gen4 end_merged)
= −0.1894 при 2ε=0.0012 — ACCEPT; улучшены ВСЕ домены, включая
HELD_MATH −0.14 и HELD_CHAT −0.12 (перенос, не меморизация).
Инкумбент: rnn_gen4_root_ft/end_merged.pt. Замечание: исторические
записи лестницы (5.5451 для rnn-3) сняты drifted-протоколом —
сравнивать поколения по переизмеренным парам, не по абсолютам.

Следующий рычаг (по теории): замедление −0.50→−0.28→(−0.19 при
смене качества градиента) — конус насыщения; кандидаты gen5: рост
доменов M (boost_* корпусы доступны) или ранг r128 по Welch-триггеру
(измерено на gen3: r64 НЕ исчерпана, req_dim=2 при M=6); отдельный
кандидат — LR-полка ниже (фикс CE поднимает эффективную кривизну L
через рост маржи → по R347 η=1/L кап должен опуститься).



## HybridAgent (arXiv:2610.18861) — порт-решение

Архитектурная идея HybridLink (residual адаптер e = h + W₂σ(W₁h) из
пространства скрытых состояний в пространство эмбеддингов,
latent/text режимы в одной генерации, KV-working-memory между ходами)
проверена против теории: паттерн «zero-init residual adapter» уже в
наборе (ttt_lora, out_proj Recurrent); рекуррентная ветка — готовый
latent-носитель (O(1)-состояние). Решение: НЕ портировать в
текущий цикл (YAGNI: рост сейчас — поколениями и rank-growth по
rank_orthogonal_growth, не в клетке; зафиксировано в THEORY_MAP
2026-10-20). Кандидат на будущую фазу: «latent-thinking шаги» как
отдельная стадия ft после стабилизации лестницы.


## Волна target-architecture (2026-11-02): GDN-RAM + ByteHNet

Целевая архитектура (Qwenputer × Cartesia H-Net × Hagi Lean):

| Компонент | Статус | Место |
| --- | ------ | ----- |
| Gated DeltaNet (S' = gS + β(v−gSk)kᵀ), Lean SSOT gdnUpdate/gdnBridge | ✔ | `model/gdn.py` (backbone='gdn'; RAM-режим α=1,β=1,k=eᵢ = точная запись колонки `ram_write`/`ram_read`; zero-init out_proj — function-preserving; инкрементальный state = KVCache-контракт) |
| Byte-level H-Net (vocab 256, динамический чанкер STE + децункер) | ✔ | `model/byte_hnet.py` (input_mode='bytes_hnet'; min_chunk≥2 — precondition AdaptiveChunking `adaptive_le_static`; surrogate-градиент `train.chunker_weight`) |
| Проводка model.forward: bytes → chunk → backbone → dechunk → head | ✔ | `model/model.py` (LM head остаётся per-byte: exact-CE канал совместим) |
| Оптимизатор SafeQP/Muon/gate_never_increases | ✔ (без изменений) | `train/loop.py` + `safeqp_step.py` |
| Точная память: GDN-RAM доступ из контура обучения | каркас | `gdn.ram_write/ram_read` — интерфейс готов, external-loop consumers (BendLite-верификатор, RV32-микро-исполнитель) — следующий раунд |

Находки реализации:
- Cold-start trap децункера: zero-init последнего слоя MLP обнулял градиент
  ко всему бэкбону (dL/dchunk = W₂ᵀ·0 = 0). Фикс: small-random init 0.02
  (residual и так даёт near-identity старт). Тест
  test_full_model_byte_gdn_forward_backward фиксирует регрессию.
- GDN out_proj zero-init: на шаге 0 градиент получает только out_proj —
  стандартное поведение residual-zero-init (в изоляции in_proj градиентится,
  test_gdn_neural_mode_backward_and_incremental_consistency).


## Сверка с первоисточниками (2026-11-02, прямой перенос)

Склонированы и прочитаны напрямую: goombalab/hnet, divagr18/qwenputer
(вместе с docs/q35.md), voidash/qwenbend (engine_circuit.py — целиком),
vixhal-baraiya/qwenGPU (README + архитектура). Три переноса и один
корневой фикс:

| Что | Источник | Изменение |
| --- | -------- | --------- |
| AND-гейт | qwenputer `prims.t_and` (полуединичный запас) | `exact_engine.ExactAND`: gate = 512·(Σpos−Σneg) + 256 − 512·n, up = 1/256. Любое нарушение сдвигает гейт на 512 → silu(−256) = −0 ТОЧНО даже при fp32-шуме литералов (запас ×2 против прежнего 256/128) |
| Границы чанков | hnet `RoutingModule` | `byte_hnet.DynamicChunker`: p = (1−cos(h_t, h_{t+1}))/2, q = k = identity — БЕЗ обучаемых параметров роутера; STE поверх; p(0) = 1 (PAD_PROB-дисциплина) |
| Дечункер | hnet `DeChunkLayer.step` | `byte_hnet.dechunk(..., boundary_probs)`: EMA-бленд p·out + (1−p)·prev по вероятности границы |
| Дельта-скан | канонический DeltaNet (arXiv:2412.06464; тот же приём в qwenputer/q35 «Memory is the delta rule») | `gdn.gdn_scan`: ℓ2-нормировка ключей per-head ВНУТРИ скана. Без неё проекция (gS)k·kᵀ масштабирует состояние на ~β·‖k‖² (≈115× при Kh=128) — экспоненциальная расходимость, NaN к шагу ~50 (измерено). RAM-режим: one-hot ключи уже единичные — нормировка no-op, точная запись колонки сохранена (Lean gdnUpdate не нарушен) |

Проверено прогоном: байт-GDN линия (hidden 128, 2 слоя, data_bytes 30MB
сырых байт UTF-8) учится: CE 5.5435 → 5.5299 за 60 шагов, градиенты
конечны, обновления принимаются certified-контуром.

Open: GDN chunkwise-скан (WY-representation) — последовательный скан
ограничивает wall-clock на длинных T; перенос при переносе линии на
крупный масштаб.


## Обучение байт-GDN линии v1 (2026-11-02/03, измерено)

`scripts/train_byte_gdn.py` (3000 шагов заявлено; корпус 30MB исчерпан
на 1831): hidden 256, 4 слоя, GDN 1-головной, cosine-chunker 4-12,
batch 8x1024, AdamW 1e-3 bf16, iGPU ~3 c/шаг.

- CE: 5.546 → **best 2.756** (uniform-базлайн ln 256 = 5.545).
  Провал на шагах 450-500 (grokking-профиль байтового LM), плато 2.8-3.1.
- Чекпойнт: checkpoints/byte_gdn_v1/best (4.0M параметров).
- Генерация: train-режим CE 2.76, но greedy-continuation вне
  train-кондиционирования схлопывается в повтор ведущего байта
  (0xD0-цикл): модель ещё не учится останавливать UTF-8
  последовательности при teacher-forcing без reset. Это честная
  граница v1: next-byte CE низкий, свободная генерация требует
  больше данных/шагов (1831 шаг ~ 60M ток = 0.23 эпохи по CE-мере;
  тупик повторов — известный признак недообученного байтового LM).

Открытые шаги линии:
1. Corpus rotation (start_frac) + больше данных — убрать повтор-циклы.
2. Chunkwise GDN-скан (WY) — скорость.
3. Muon включить после стабилизации AdamW-базлайна (A/B).


## Волна §2b/§17/§18/§19 (порт 2026-11-03, по коду не по статусу)

Аудит покрытия ALGORITHMS.md §0–19 по фактическому коду. Пробелы,
найденные и закрытые в этом раунде:

| § | Пробел | Порт | SSOT |
| --- | ------ | ---- | ---- |
| 2b | sign-scale/Lion/SAM отсутствовали в optim | `train/sign_step.py`: signScale, scaled-sign aligned-резюдуал (сырой sign НЕ aligned — контрпример (−1,−1,0.5) в тесте), certified descent bound η≤1/L, Lion-когерентность, SAM-трансфер | Unified/SignDescent.lean |
| 17 | Active-Inference вертикаль отсутствовала | `train/active_inference.py`: proxy-дихотомия (нижний прокси сохраняет сертификат, верхний теряет), EFE-потолок G=P−I≤P, graceful-деградация гэпа, компаундинг L^H/(L²)^H + MPC-резет, амортизированный политик 2c | Probability/GibbsConcentration.lean (R462–467) |
| 18 | Pinsker-семейство частично | `train/pinsker_family.py`: tv, kl, tv≤Σpos-part, Бхаттачарья≤1, 1−BC≤KL/2, TV≤√KL, −lnBC≤KL/2, Бретаньоль–Юбер TV≤√(1−e^{−KL}), TV-дуальность | Probability/PinskerTV.lean |
| 19 | ДВ/Cramér отсутствовали | там же: Donsker–Varadhan E_p f − ln E_q e^f (достижение на f=ln(p/q)), (E−μ)²/2σ²≤KL, |E_p g−E_q g|≤(b−a)TV | Probability/ChernoffCGF.lean |

Тесты: +21 (902 total). Ловля контрпримеров: сырой sign-шаг нарушает
alignment; √KL vs KL при KL<1; DV достигается ровно на f*=ln(p/q).

Осталось вне кода: исполняемый exact-engine контур (firmware-уровня
qwenbend), GDN chunkwise-скан (WY), Landaauer-мосты как потребители.


## WY chunkwise GDN-скан (2026-11-03)

`gdn.wy_scan` — chunkwise WY-форма гейт-дельта-правила (Yang et al.
arXiv:2412.06464), ТОЧНО воспроизводит последовательный `gdn_scan`
(fp32-eps, проверено на C=8/16/64 и с ненулевым h0; тест
test_gdn_wy_scan_exact_vs_sequential). Ключевые ловушки, найденные
тестами: (1) кумулятивный decay входит ВНУТРЬ скалярного дота
Σ_k k̂_i k̃_l (D_i/D_l)_k, не снаружи; (2) write-key k̃=β⊙k̂ (Lean-форма
bt·(...)*k̂ᵀ), read-key k̂; (3) частичные суммы выводов j≤i, полное
только для конечного состояния; (4) накопление W списком + stack
(inplace-записи в градиентный буфер ломали autograd).
Forward GDN переключён: neural → wy_scan (chunk 64), ram →
последовательный gdn_scan (точная семантика колонок).

## wy_scan численная устойчивость (2026-11-03, отладка NaN v4)

Корень NaN при bf16 T=1024 h=384 (не OOM, не bf16-софтмакс):
1. **Ключи не нормированы в wy_scan** (gdn_scan нормирует внутри) — A[i,j]
   взрывались (norm(k)~16 → inf к t=68). Фикс: normalize внутри wy_scan
   (паритет с gdn_scan).
2. **Переполнение e^{−cg}**: при сильном распаде g≈0.04 накапливает cg за чанк
   до −210 → e^{+210}=inf в fp32. Фикс: сдвиг cg' (max-shift, точен в
   разности) + clamp_min(−70).
3. **cumprod underflow**: cumprod(g)→0.0 к шагу ~256 убивал chunk-relative
   разности. Фикс: log-space cumsum (точно).
4. **Двойной учёт pre-chunk decay**: глобальный cg в base/A считал распад
   прошлых чанков дважды. Фикс: chunk-relative cgL (cg − cg[s−1]).
5. **OOM**: наивный попарный цикл держал ~L²/2 rank-1 графов (~5 GB/блок).
   Фикс: батчевая pair-матрица einsum.
Итог: wy_scan бит-точен vs gdn_scan на всех масштабах (тест до 5e-4 при
сильном распаде, T=512); wy_chunk=32 (decay budget; 64 давал ошибку 0.02–0.18
через clamp-артефакты). Регрессионные тесты: strong_decay_finite,
raw_key_normalization.

## VHCL-аудит (2026-11-03): формальные гарантии ↔ 4 источника

Соответствие единой архитектуры (qwenputer/hnet/qwenGPU/qwenbend × primes):
- **DeltaNet RAM** (gdn_onehot_is_colUpdate, ram_roundtrip): gdn.py ram_write/
  ram_read + mode='ram' (последовательный скан, точная семантика колонок).
- **H-Net router** (cosine boundary, p=(1−cos)/2 clamp, STE, min_chunk≥2):
  byte_hnet.py DynamicChunker; STE-градиент через chunker_surrogate в loss
  (loop.py, chunker_weight=0.01).
- **DeChunkLayer** (p-weighted EMA blend): byte_hnet.dechunk (boundary_probs).
- **Muon** (Newton–Schulz quintic, stable-LR cap η=(2/λ)(Σσ/m)): optim.py.
- **SafeQP** (safeQP_descent/dual λ, certificate, η*=<g0,d*>/(L‖d‖²),
  R105 eta_max cap, no_domain_regresses veto): formal.safe_qp_solve +
  loop._apply_certified_step (certified_step='gate'|'step').
- **Quant bridge** (gridQ_energy_bound + smooth_from_lipschitz_gradient):
  lip_energy.py; BitLinear cached-STE + ternarize_split (TernaryExact).
- **hnet balance loss c N_target**: заменён на measured surrogate (v1
  решение); N_target-форма — кандидат после измерения дисперсии чанков.
- **VM freeze (θ_VM ∇=0)**: применим при включении exact_engine в обучение
  (сейчас opt-in, train-loop не активен — фиксировать нечего).
