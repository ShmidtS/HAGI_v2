# Pyramidal Cross-Layer Cortex

Статус: **opt-in research MVP**. Модуль включён только при
`model.cortex.enabled=true`; путь по умолчанию и старый per-block
`PyramidAdapter` не изменяются.

## Что реализовано

`PyramidalCortex` (`src/hagi/model/cortex.py`) делит уникальные
Transformer-блоки на непрерывные уровни. На границе уровня публикуется
bottleneck summary:

\[
z_k = R_k(h_k)
\]

Для target-level \(l\) вычисляется направленный residual:

\[
h'_l = h_l + \rho\,U_l\left(\sum_{k<l} P_{k,l}z_k\right),
\]

где каждое направленное ребро \(k\rightarrow l\) имеет собственную
матрицу \(P_{k,l}\). По умолчанию разрешены adjacent edge и skip edge
(\(l-k\in\{1,2\}\)); циклов и dense all-to-all связей нет.

Реализация:

- один `down` на каждый используемый source-level;
- один `up` на каждый target-level;
- независимая zero-init `_DirectedLink` на каждое edge;
- additive residual с фиксированным `residual_scale`;
- state-контейнер создаётся заново на каждый pass;
- state не сохраняется между токенами или generation steps;
- `loop_depth` не переносит summary из предыдущего pass;
- `KV-cache` использует ту же per-token геометрию, поэтому decode
  parity проверяется отдельно.

## Это не старый `PyramidAdapter`

В проекте уже есть `PyramidAdapter` — per-block repeated-branch
residual contour. Новый Cortex — **model-global side channel** между
уровнями скрытых состояний. Они не взаимозаменяемы:

| Механизм | Scope | State lifetime | Назначение |
|---|---|---|---|
| `PyramidAdapter` | один Block | stateless внутри Block | repeated frozen mixer branch |
| `PyramidalCortex` | все уровни | один forward/pass | directed cross-level memory |
| `TTT-LoRA` | один Block | online RLS/gradient state | low-rank adaptation |

`PyramidalCortex` может сосуществовать с `TTT-LoRA`. RLS-режим
`self_improve(mode="rls")` по-прежнему обновляет только TTT-LoRA;
Cortex обновляется через gradient/trainer path.

## Конфигурация

```yaml
model:
  cortex:
    enabled: true
    num_levels: 2
    rank: 4
    link_strides: [1]
    residual_scale: 0.1
train:
  precision: bf16
  ternary_fp32_master: true
  adapt:
    freeze_base: true
```

`num_levels` находится в диапазоне `[2, model.num_layers]`.
`link_strides` должны быть положительными, уникальными и отсортированными;
каждый stride меньше `num_levels`.

## Training и precision

- При `freeze_base=true` ownership определяется маркером
  `AdaptiveComponent`, а не именем модуля. В optimizer попадают только
  параметры BlockAdapter и Cortex.
- При `train.precision=bf16` можно дополнительно включить
  `train.ternary_fp32_master=true`: `BitLinear` masters остаются FP32, а
  effective ternary matmul выполняется в dtype активаций (обычно BF16).
  Флаг тренировочный и по умолчанию `false`; физический packed storage
  не меняется.
- Все Cortex masters остаются FP32 при `train.precision=bf16`; BF16
  округление не должно убивать малые online updates.
- В runner указан **hypothetical** accounting model: ternary base,
  4-bit cortex, 8-bit LoRA, FP32 activation/RLS accumulation.
- Физические INT2/INT4/INT8/FP8 kernels и online accumulator
  serialization в текущем срезе **не реализованы**. Ternary STE уже
  существует, но это не физический упакованный storage; нельзя называть
  этот MVP измеренной реализацией предложенной precision stack.
- Для numerics/storage smoke запустите
  `scripts/ternary_precision_ab.py`: он сравнивает FP32, legacy BF16 и
  BF16 compute с FP32 ternary masters из одного pre-cast FP32 state source
  на идентичных batches. Post-cast BF16 tensors не считаются bitwise
  equal. Это mechanism smoke, не quality evidence и не измерение
  физического ternary storage.

## Merge semantics

При `merge_experts` expert-level `cortex.*` и `*.adapters.*` состояния
не сливаются как expert weights: новый merged level получает fresh
adaptive state. `mixers.*` сохраняют старый контракт и заменяются
только при `drop_expert_mixers=True`.

## Experiment gate

Runner: `scripts/pyramidal_cortex_ab.py`.

```bash
# герметичный CPU mechanism gate
python scripts/pyramidal_cortex_ab.py --steps 12 --num-seeds 3

# короткий bounded read из настоящего packed corpus
python scripts/pyramidal_cortex_ab.py \
  --steps 1 --num-seeds 1 --device cuda --real-corpus \
  --source wikipedia_ru --data-dir data \
  --start-offset 100000 --holdout-offset 10000000
```

Runner сравнивает `full_base`, `lora_only`, `cortex_only` и
`cortex_lora` на одинаковом starting base state и одинаковом train
stream. Holdout batch не входит в train stream. Synthetic holdout —
только mechanism signal; `quality_supported` всегда `false`. Положительный
quality verdict требует `--real-corpus`, минимум три seeds, конечные
измерения и строгое большинство побед в обоих сравнениях. Один bounded
run не является quality evidence.

## Что нужно проверить до следующего этапа

1. Несколько seeds на реальном held-out corpus и на корпусе с
   пересекающимися/разными доменами.
2. Сравнение `rank`, `num_levels`, `(1)` против `(1,2)` и границ
   `num_levels=2,3,4`.
3. Реальный 27B checkpoint и настоящий frozen-base adaptation path.
4. Physical mixed-precision kernels и отдельные измерения ошибки
   base/cortex/LoRA, а не только aggregate CE.
5. Sequential-task forgetting и online RLS accumulation для Cortex.
