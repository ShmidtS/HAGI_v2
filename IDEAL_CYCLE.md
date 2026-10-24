# IDEAL_CYCLE — идеальный цикл HAGI по доказанной теории

Источник: дословные формулировки из C:\primes\Hagi (Lean 4, 2148
theorem, 0 sorry). Каждый пункт — теорема, не эвристика. Формат:
правило → теорема → операционная реализация в этом репо.

## 0. Потенциал и учёт поколения

- Φ = energy + protectedRisk (`potential`, GrowthState:171);
  energy = token-weighted reverse-KL сертификат.
- Одна генерация grow→merge→joint→compress:
  Φ′ − Φ ≤ −(growGain + gap + η‖d‖²/2) + riskSpend,
  riskSpend = growRisk+mergeRisk+jointRisk+compressRisk
  (`growth_cycle_potential`, :429; `certificate_sound`, :470).
- Телескоп поколений: чистый темп = ε − r (decrease минус
  riskSpend), НЕ ε (`termination_from_telescope`: T ≤
  (Φ₀−φmin)/(ε−r)).
- Реализовано: train/stage_telescope.py, certified_controller.

## 1. Шаг (оптимизатор)

- η = 1/L КАП, всегда. η* = inner/(L·dnorm²) запрещён — при
  сертифицированном спуске η* > кап всегда
  (`eta_star_exceeds_cap`, StageCalculus:173; `joint_stage_cap_
  optimal`, :191 — максимум гарантии на допустимом интервале:
  E′ ≤ E − dnorm²/(2L)).
- L — ИЗМЕРЯЕМАЯ посылка h_emp_smooth (квадратичная модель не
  хуже фактического E2), не гиперпараметр. Реализация: lr-полоса
  H·lr 0.35–0.38 как прокси L⁻¹; при расхождении (divergence)
  полка понижается на шаг — это и есть L-измерение задним числом.
- Спуск гарантируется при dnorm² ≤ inner (`joint_stage`:79).
- Оптимизатор: Muon+AdamW (MuonStableLR η=(2/λ)(Σσ/m), R379–382).

## 1b. Числовое ядро CE (arXiv:2610.31642, порт 2026-10-23)

- Партиционная сумма CE ИСКЛЮЧАЕТ целевой логит: loss = softplus(−m),
  m = z_t − lse_{j≠t} — точность до underflow, а не до клиффа
  точности (bf16 ≈5.5, fp32 ≈16.6).
- Градиент цели g_t = −sigmoid(−m): Σg строки = 0 ТОЧНО. Старая
  форма (p_t − 1) округляла g_t в 0 за клиффом → Slingshot-спайки
  под AdamW (диагноз дивергенций листьев 355/шаг).
- Равновесие маржи m* ≈ log(1/ε) (18.4 при AdamW ε=1e-8): маржа
  растёт до равновесия весового распада и ε — `gate_margin`
  телеметрия; m → m* = меморизация завершена (кандидат в
  saturation-гейт).
- KD: градиент из разности сырых логитов через expm1/log1p
  (Cancellation-фикс); log-prob учитель запрещён (катастрофическое
  сокращение в центрировании d = δ − ⟨p,δ⟩).

## 2. Шум и батч

- ε_noise = σ‖d‖·√(2·log(|K|/δ)/m) (`noiseEps`,
  StochasticSafeQP:33): сертификаты держатся с запасом ε+ε_noise.
- LR↔batch: (η, m) → (kη, k²m) сохраняет шумовой бюджет
  (`lr_batch_equivalence`, :354; инвариант η²σ²/m). ПРАВИЛО:
  вместо снижения LR в k раз — батч в k² раз.
- Растущий батч b0·δ^t при ρδ>1 не накапливает шумовой пол
  (`growing_batch_stabilized`, :508). Реализовано:
  growing_batch.py (ft: delta 1.0055, cap 16).

## 3. Grokking (weight decay выведен)

- λ ≥ L·ln(r₀/ε)/T_target при ηλ ≤ 1 (`grokking_threshold_time`,
  Recurrence:335). λ — не тюнинг: при T_target = ft-бюджет и
  η-капе λ определён. Текущий adam.weight_decay 0.3 фикс.

## 4. Выбор действия

- Весь бюджет на argmax Γ/K (`budget_allocation_dominance`,
  ControllerPolicy:43).
- Argmax сертифицирован только при зазоре Γ̂_i − Γ̂_j > 2ε_Γ
  (ε_Γ = split-half шум; `argmax_pair_certified`,
  CertifiedArgmax:37). При зазоре ≤ 2ε_Γ — tie_ambiguity, выбор
  неинформативен: сертифицированное поле решает.
- Запрет отбора по standalone CE (`selection_hurts`).

## 5. Merge

- Блокирующий гейт: twoGap = ½logΣΣp_u p_v cosh(d_u−d_v) ≥ 0,
  равен 0 ⟺ листья — сдвиги друг друга (консенсус = раунд
  исчерпан) (`twoGap_zero_iff`, GapLaw).
- Термостатный merge-гейт: J ≥ θ·T_t, T_t = T₀·α^t
  (`merge_gate_eventually`, Thermostat:109) — любой J>0 рано или
  поздно проходит.
- Рост ранга: ранг корня растёт РОВНО на k при (а) k новых строк
  независимы, (б) каждая ортогональна каждой старой строке
  (dot=0) (`rank_orthogonal_growth`, RankGrowth:56). Тогда
  обновление в span новых направлений имеет нулевую интерференцию
  со старыми доменами при ε=0 (`span_growth_zero_interference`,
  :141). Реализация: verify_ortho.py — max|cos| новых факторов к
  старому row-space, гейт < порога.

## 6. Рантайм-гейты

- Variance gate: перегрев logitRange чинится ТОЧНО τ =
  logitRange/R̄ (`variance_gate_normalize`, ControllerGates:148).
- Trust region: выход за шар → проекция (R/‖x‖)x
  (`trust_region_proj_norm`).
- Шаг принят при dL_meas + (ε_probe + L_att·ε_att)·‖d‖ ≤ 0
  (`cut_safe_accept`, ProbeError): заморозка компонента законна
  только с измеренным ε_att·L_att-запасом.

## 7. Стопы (легальные условия остановки)

- Liveness: консенсус (twoGap=0) И inj ≤ ξ ОДНОВРЕМЕННО
  (`liveness_two_axis`, Liveness:55). Любая ось жива → есть
  действие.
- Термостат: t ≥ N = ⌈ln(δ/(T₀·ln(1/θ)))/ln α⌉ (`thermostat_stop`).
- Горизонт конуса: T ≤ 1 + [ln(σC*)−ln((γk+σ)C₀)]/ln(1+γk)
  (`cert_horizon_log_bound`, ArchitectureTheorem:1208) — сильнее
  конус → короче поколение; темп γk ≤ ρ (не выше retention).
- Телескоп: T ≤ (Φ₀−φmin)/(ε−r).

## 8. Архитектура (что предписано)

Теория НЕ предписывает seq-оператор (attention/recurrent
равноправны — зафиксировано в THEORY_MAP). Предписано:
- параметр-шеринг time-invariant + гиббсова смесь ансамбля;
- рост РАНГОМ на фикс. ширине (rank_orthogonal_growth), не
  шириной;
- per-tensor экономика merge (волокна сохраняются, сжатие только
  ROOT-факторов).

## 8b. Протокол классов (сброс ширины; волна 2026-10-24)

Сброс класса (новая leaf-width, fresh-init) меняет базу сравнения:

- **Лестница класса**: первые поколения нового класса сравниваются
  против ПРЕДЫДУЩЕГО ПОКОЛЕНИЯ СВОЕГО класса, не против инкумбента
  старого — иначе REJECT-записи будут ложным «отказом от роста».
- **Смена главной модели** — только когда новый класс сертифицированно
  превзошёл старый инкумбент (certify.py с парой инкумбент↔кандидат).
- **Fresh-листья**: (а) полный бюджет каждому (бюджет prior-домена
  оставляет leaf в разогреве приёмника), (б) стоп-гварды выключены
  (плато CE~10 первые 300-500 шагов — норма разогрева, не насыщение),
  (в) ожидаемая точка первого поколения ≈ уровень старого класса
  после всей его лестницы (enedu-256: 5.76 за 800 шагов ≈ enedu-128
  после 4 поколений) — прирост класса приходит со 2-3-го поколения.
- **Инцидент-протокол**: один обучающий конвейер на GPU (lock);
  width-aware concurrency (384//width); никогда не запускать листья
  поверх идущего цикла (трэшинг измерен: 3.4 с/шаг против 1.4).

## Текущая конфигурация RNN-ветки (выведена из этого)

- Листья: 6 доменов от root-prior, TTT-LoRA r32 поверх
  замороженного корня, шаги 1300 math / 2400 books / 800 прочие;
  lr-полоса 0.35–0.38·H⁻¹ (η-кап), saturation/divergence гварды.
- Merge: latent per-tensor (factorize_delta→latent_align→
  root_contrast_merge→spectral_compress), merge_gate per-tensor.
- ft: r64, 600 шагов (R361: 100×листья), lr 2e-4, batch-growth
  1.0055→16, λ=0.3, Muon+AdamW.
- Приёмка: certified A/B (Hoeffding eps = √(ln(2/δ)/2n)),
  deep-200 протокол; REJECT → смена рычага, не повтор.
