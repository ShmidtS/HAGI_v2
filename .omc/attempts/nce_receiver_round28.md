# Раунд 28: prior-NCE receiver — 2.07× wall-time И −0.24 ната одновременно

## A/B дизайн (leaf, shared E₀ prior, те же seeds, 1600 шагов)

- **Arm A (fused full CE)**: e0leaf_s1 (раунд 19): 369 ms/step,
  ~9.5 мин/лист, exact CE 5.9512.
- **Arm B (prior-NCE K=2048)**: nce_leaf_s1: 91 ms/step (4.0×),
  4.6 мин/лист (2.07×), exact CE **5.7076** (−0.244!).

## Главный результат

Sampled prior-NCE доминирует по ОБЕИМ осям:
- wall-time-to-CE: 4.6 мин до 5.708 vs 9.5 мин до 5.951 — **2.07×
  дешевле при лучшем качестве**.
- Точность: K=2048 локальный partition = регуляризатор (аналог
  label smoothing на хвосте распределения: head не может
  переподгоняться на полный V). Аналог label-noise/regularization
  выигрышей NCE в литературе.

## Предостережения (честные границы)

1. Один seed на плечо — по протоколу HAGI требуется ≥3-5 seed
   для canon-обновления (результат НЕ обновляет канон до повтора).
2. K=2048 = 1/16 V: одна точка; sweep K ∈ {512, 1024, 4096} нужен
   для карты quality/K (предсказание P2 Lean-раунда: адаптивный K
   из Σp²/q).
3. Проверка через ensemble: NCE-листья могут иметь другой bias
   профилей (NCE-градиент ≠ CE-градиент) — gap/комплементарность
   пула NCE-листьев измерить отдельно перед включением в рекурсию.

## Реализованные фиксы по ходу

- `init_from` теперь skip'ает `head.log_prior` (unigram prior —
  детерминированная функция корпуса, не learned state; stale prior
  от источника не должен наследоваться).
- unigram.compact.npy (32k-mapped) — правильный путь для
  hagi-токенизатора (unigram.npy = старый 262k).

## Указания по синтезу (sparse-стек PR #360)

- Row-sparse LM-head: forward уже срезает (index_select); в
  optimizer-плане: sparse AdamW (β1=0, lazy v_r с timestamp) для
  embedding/head — СЛЕДУЮЩИЙ перенос (качественно другая scaling
  law: O(|S_t|H) на optimizer step вместо O(VH)).
- Бюджет в ramках формализации: P2 (NCE variance + adaptive-K)
  запрошен Lean-агенту; wall-time-закон теперь ИЗМЕРЕН.
- A-Cortex / параметрическая sparsity — условие запуска
  long-context/multi-GPU не изменилось.

## Очередь

1. NCE-A/B повтор на 3 seed → canon-обновление листа.
2. K-sweep {512, 1024, 4096} + Σp²/q карта слабых мест.
3. Sparse AdamW (β1=0 lazy) для codebook — оценка выигрыша на
   leaf-масштабе, перенос в merge/parent.
4. NCE-пул в рекурсии: 3 NCE-листа → ensemble → gap-замер.
