# Round 40: рецензия round-3 (a35e2f7 ↔ 5689bf7) — маршрутизация

## Главный тезис рецензента принят

Формализация = «компилятор архитектуры»: a* = argmin T(a) при
CE ≤ target, ΔCE_i ≤ ε_i, M ≤ M_max; внутренние параметры — по
KKT-отношению предельный-CE-на-wall-time против теневой цены λ.
Это центральный закон HAGI (переносится в ARCHITECTURE.md).

## P0 — честность Lean (→ задача формализатору, primes-репо)

- Compound.compound_c_decompose — hypothesis→hypothesis, demote или
  доказать настоящую декомпозицию c_t ≤ αG_t + J_t;
- DBridge.equilibrium_bracket — доказан только lo ≤ hi; нужна
  liminf/limsup-теорема из рекуррентности G_{t+1}=ρG_t+D+ξ;
- GenCycle «dies exactly» — нужен нижний бок G_{t+1} ≥ ρG_t + D_-;
- ns_poly_bound — a=3.4445>1 не contraction; нужна теорема об
  инвариантном диапазоне σ, где ρ<1 реально;
- amgm_batch_bound — уникальность B* требует c,B_n,t0 > 0;
- adaptive_ns_exists, supportSet_bound — усилить.

## P1 — DesignOpt v2 (дискретный, не continuous-KKT)

N, r, W, S, K, p дискретны → discrete resource allocation с
confidence bounds; привязать к recursive.py / merge_select.py.

## P2 — ANOVA-tree формализация

P_root ⊥ P_contrast, E = E_root + E_contrast, распределение rank
по измеренной энергии. Код-следствие: contrast highway в cortex
(§15) — следующий F3-эксперимент НЕ depth-2, а root+contrast
(объясняет 3.6635 vs 3.6357: root-cortex видит только E_root≈0.796).

## Код-очередь HAGI_v2 (§32, по убыванию ценности)

1. [ЭТОТ РАУНД] A/B mixer: dbridge-линия на swiglu, а hadamard —
   рекомендованный путь (12× меньше параметров: 73.7k vs 884.7k
   при H=384). A/B на gen2-merged→joint.
2. [ЭТОТ РАУНД, если успею] ridge-init mixer: U* = YΦᵀ(ΦΦᵀ+λI)⁻¹
   вместо gain=0-старта (использует Ridge.lean).
3. Contrast highway (root ⊕ contrast low-rank, ранги по энергии).
4. Gradient-Gram leaf selection: Score(E_i)=ΔVar_p(d)/T_i вместо
   standalone-CE screen (D_token → D_grad → D_F лестница).
5. Receiver-канал: K по Var[Z] ≥ (E[f²]−μ²)/ε_var, keep-rate по
   KKT, exact-CE anchor cadence — отдельно от body.
6. head_up — закрыто (round 39b, отрицательный контроль).
7. μP: update-to-stream ratio invariance — новая theorem family
   (норм-инвариантность ≠ optimizer-инвариантность).

## Согласовано/подтверждено рецензентом

- Ортогональный mixer ничего не создаёт (транспорт + learned
  residual — правильная декомпозиция) — соответствует нашему
  merge-замеру «налог +0.0067 плоский»;
- ternary ≠ в победной линии: рекорды — growth/merge-эксперименты;
  в доке разделить Growth / Growth+dense / Growth+ternary;
- R_semantic (1.585 bit) ≠ R_physical (packed 5-trit) — отдельные
  леммы;
- «23% медленнее mixer может быть лучше» — T_total = T_step × S,
  head_start_timeshift.

## Статус микса (сверка §19)

Подтверждено: gen1/gen2-конфиги — mixer_type: swiglu, rank 64;
MergeConfig default — hadamard. Развилка НЕ проверена A/B на
рекордной линии. Исправляем сейчас.
