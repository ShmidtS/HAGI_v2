# Раунд 22: low-rank элемент (Element.lean) — LoRA-листья бьют full-table

## A/B дизайн

Оба плеча: одинаковый prior (leafdat_2xcorp, v3-канон), одинаковые
init/data seed, 1600 шагов, canonical mix.
- **Arm A (full-table)**: e0leaf_s1..3 (раунд 19) — всё обучаемо.
- **Arm B (LoRA rank-16)**: lora_leaf_a1..3 — E₀/W₀ FROZEN (buffers),
  TableLoRA дельты на embedding+head (A: frozen orthonormal QR,
  B: zero-init → residualLeaf_zero), тело обучаемо.

## Результаты (mixed 8-corpora gate, точная алгебра)

| метрика | full-table | LoRA rank-16 | Δ |
|---|---|---|---|
| mean single leaf | 5.95 (s1) | **5.715** | −0.24 |
| flat3 ensemble | 5.875 | **5.679** | **−0.196** |
| Jensen gap | ~0.3 (независимые) | 0.036 | — |
| таблицы-параметры | N·2VH | VH + N·r·2(V+H) | **2.5× сжатие** |

**Frozen prior — регуляризатор, а не ограничение**: замороженная
E₀ предотвращает дрейф таблиц (механизм round-19: у полных таблиц
65% дельты — общий шум дрейфа, который LoRA-параметризация просто
запрещает, оставляя только специализацию).

Ссылки: LoRA-ансамбль 5.679 ≈ gen-1 sibling-ансамбль 5.621 (после
merge+joint!) — LoRA-элемент получает почти тот же результат БЕЗ
joint-шага. Комбинация LoRA-листья + joint обещает новое лучшее.

## Реализация

- `src/hagi/model/table_lora.py`: TableLoRA (AdaptiveComponent,
  base/A = buffers, B = zero-init trainable), lora_compression,
  recenter (§7: mean-delta → parent, children → zero-sum).
- `scripts/lora_leaf_ab.py`: build_lora_leaf (embedding-lookup без
  материализации: W0[ids] + A[ids]@B — O(T·rH); head через
  weight-property — стандартный F.linear путь), train_lora_leaf.
- Верификация: residualLeaf_zero бит-в-бит (CE 4.3666 = 4.3666 на
  edu при init); 18 тестов зелёные.

## Прочие переносы раунда (синтез-§)

- §27 merge_tax в formal.py; §2 уточнённый N* = (−1+√(1+4G/ε))/2
  (gen-1: 4.6 — согласуется с предсказанием); §30 adaptive_eps =
  z·√(se²+se²); §28 кэш pool-логитов в growth_gate (M(N+1) → N+M
  форвардов, live 10.5с на 2 кандидатов).
- next_leaf_gain(G_N, N) = G_N/(N²−1) — предиктор следующего листа
  без знания G∞.

## Очередь

1. LoRA + joint: merge LoRA-плоских3 → fine-tune @ lr=1e-3 → новый
   prior (комбинация двух победителей).
2. Ранг-sweep: r=8/16/32/64 (где насыщается).
3. §20 rank-budget waterfilling по SVD-спектрам.
4. §9-12 sampled prior-NCE receiver (главный speed-резерв).
5. §13-14 grad_checkpointing=false + ns_steps=3 (быстрые победы).
