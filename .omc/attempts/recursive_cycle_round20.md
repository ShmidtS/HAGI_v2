# Раунд 20: рекурсивный цикл замкнут — shared-prior + merge + joint = −0.42 ната

## Цепочка раунда (полный self-growth цикл)

| шаг | CE (mixed 8-corpora, физический замер) |
|---|---|
| v3-листья flat3 (независимые старты, ансамбль-алгебра) | 6.120 |
| **E₀-листья flat3** (shared prior, ансамбль-алгебра) | **5.875** |
| E₀-листья flat3 (физический merge, n_mixers=0) | 5.929 |
| **gen1 joint fine-tune @ lr=0.001** | **5.694** |

Итог: **−0.42 ната против независимой лестницы** — полная
рекурсивная петля «shared-prior листья → merge → joint training»
доминирует на каждом шаге. Это рабочий исполнительный алгоритм
саморазвития (без человека в петле).

## Открытия раунда

1. **merge-баг пойман**: qk_norm/sink_bias переносились при merge
   (per-head gains concat), но мой merge-скрипт выключал qk_norm
   вручную — CE 4.85 вместо 4.17 на edu. Исправлено: merge-конфиг
   обязан наследовать attention-структуру листа (qk_norm=True,
   sink_len=4). Урок: growth_gate merge-путь и «настоящий» merge
   должны строиться из ОДНОГО места (DRY-кандидат).

2. **LR-чувствительность joint-шага критична**: lr=0.01 (канон
   листа) УХУДШАЕТ merged prior (5.929→6.054); lr=0.001 улучшает
   (5.929→5.694). Fine-tune ≠ with-scratch training; у joint-шага
   свой масштаб LR (согласуется с warmup_fix.md — свежий
   оптимизатор на предобученных весах требует мягкого входа).

3. Рекурсия форм работает: init_from переносит body/embed/head
   merged flat3 в gen-1 модель той же формы (H=384 merge-config);
   merge-блок в конфиге обязателен (leaf-конфиг без merge строит
   монолит H=384 — shape mismatch).

## Что дальше (автономная очередь)

1. Лестница gen-1: e0gen1_joint_lr001 как новый shared prior для
  下一代 листьев H=128 (уже 5.694-старт) или растить вширь до
   flat9-equivalent (merge 3× gen1-joint).
2. Loop back: может ли flat3(5.694) быть E₀ для НОВОГО поколения
   листьев (distill-back в H=128 не тривиален — высота растёт).
3. Свип LR joint-шага (0.0005/0.001/0.002) — мы на склоне
   оптимума.
4. formal.py: перенести «shared-prior growth» как предписание
   (Element.lean residualLeaf_zero: rank-0 deltas = safe start).
