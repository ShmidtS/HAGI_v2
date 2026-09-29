# Round 33: внешние оптимизации (xLLM / CompoSimplex) — что применилось

Пока Lean-формализатор работает над round-33 запросом, применил
измеримые оптимизации из xLLM (fused/roofline-философия) и
CompoSimplex (simplex-композиция). Всё — с замером до/после и
фальсифицируемыми предсказаниями; отрицательные результаты
зафиксированы честно.

## 1. Geometric pool: ИДЕНТИЧЕН нашему logit-mean (теорема + замер)

Предсказание round-33-P2: geometric pool (product-of-experts,
reverse-KL-агрегат) отличается от arithmetic pool.

**Результат: тождество.** mean_i(z_i − lse_i) = mean z_i − mean lse_i
— константа, softmax сдвиг-инвариантен ⇒ **наш mean-logit ансамбль
УЖЕ является geometric/product-of-experts пулом** (reverse-KL
оптимумом). Это важное уточнение для Lean-запроса: тождество
geometric = logit-mean нужно формализовать (оно связывает P2
round-33 с доказанным CE_ens ≤ mean CE).

Истинная **вероятностная** смесь (mean of softmax) измерена отдельно
на dbridge-листьях (8 корпусов, 2048 позиций/корпус, 3 листа):

| pool | CE |
|---|---|
| single mean | 4.2002 |
| **logit-mean (= geometric, текущий)** | **4.1287** |
| prob-mean (истинная арифм. смесь) | 4.1321 (+0.0034) |
| median logits | 4.1441 |

Вердикт: текущий пул оптимален из трёх (per-corpus тоже везде
лучше или равен); CompoSimplex-стиль «композиция агрегатов» не
даёт бесплатного выигрыша на этом пуле. Median-пул хуже —
robust-агрегация теряет информацию при согласованных листах
(complementarity 1−ρ ≈ 0.9).

## 2. Профайлер (roofline-приоритет xLLM): диагноз узкого места

Профиль fused-шага (5 шагов, dbridge_leaf_s1, 32×1024):

- **_ChunkedCrossEntropy backward: 51% CUDA-времени** (1.29s/2.53s)
- aten::mm 17%, sub/exp внутри CE 11%+11%
- copy_ CPU 1.45s — Windows pinned-memory артефакт, H2D сам 0.1 мс
- optimizer: 0.6 мс/шаг (LazyAdam вердикт рецензента подтверждён)

Подтверждён memory-bound профиль: полный [N,V] CE — и есть цена
dense-обучения при V=32k (NCE как замена закрыт round-28/30).

## 3. save_logits A/B (xLLM fused-blocks рычаг): +3.6%, внедрено в конфиг-познание

| конфиг | ms/шаг | tok/s |
|---|---|---|
| recompute (текущий) | 392.3 | 84k |
| **save_logits=True** | **378.1** | **87k** |

chunk 4096/8192/16384 неразличимы. Memory-цена: 2.1 GB bf16-логитов
(64GB карта — дёшево). Рекомендация в конфиги gen-циклов:
`head.ce_save_logits: true`.

## 4. Triton fused CE-backward: ОТРИЦАТЕЛЬНЫЙ результат (честно)

Fused-кернел grad_hidden без материализации probs [N,V]
(`scripts/triton_ce_backward.py`): численно верифицирован
(rel err 1e-3 на эмуляции; масштаб 1/N-фактор — мой bench-баг,
кернел корректен). Замер: **triton 520 ms vs aten 314 ms —
проигрывает 1.7×**. Причина: два прохода по W[V,H] читают 2×
памяти против одного aten-прохода с материализованным z; на gfx1151
(64KB shared) BN=32/BV=128 — потолок. Вердикт: на этом
V/H-соотношении и карте fused-CE не окупается; xLLM-масштаб выигрыша
требует более широкого H (больше reuse на строку) — возвращаться
при H≥512 или на Triton ≥4.x с лучшим tl.dot-пайплайном.

## Итоги

- Применено: save_logits=True (+3.6% скорость) — в конфиги gen-циклов.
- Открыто: тождество geometric ≡ logit-mean (в Lean-запрос round-33-P2
  добавлено как обязательная лемма).
- Закрыто отрицательно: prob-mean pool (+0.0034 хуже), median pool,
  triton fused-CE (1.7× медленнее на этой карте).
- Тесты: 19 passed.
