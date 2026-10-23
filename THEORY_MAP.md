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

## Полная перечитка Lean (2026-10-23, #2: Optimization/Step 42 файла + Energy/System/Runtime/Budget 34 файла + выборочно Growth/Dynamics/Unified)

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
