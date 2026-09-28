# Раунд 23: LoRA + joint — новый рекорд 5.5318

## Траектория рекордов (mixed 8-corpora gate)

| ступень | CE | механизм |
|---|---|---|
| v3-independent flat3 | 6.120 | независимые старты |
| E₀ full-table flat3 | 5.875 | shared prior |
| gen1 joint | 5.694 | merge+joint |
| gen1 sibling ens | 5.6206 | + siblings |
| LoRA flat3 (rank-16) | 5.679 | frozen-prior элемент |
| **LoRA merge + joint** | **5.5318** | **комбинация победителей** |

−0.59 ната за сессию. Каждый механизм — из предписаний формализации
(shared prior ← Element.lean, joint ← синтез-20, LoRA ← синтез-22).

## Детали

- LoRA flat3 физический merge: 5.6801, **merge_tax = +0.0014**
  (почти идеален — плоский merge LoRA-листьев без mixer-потерь).
- Joint fine-tune @ lr=1e-3 от lora_flat3 prior: 1600 шагов → 5.5318.
- Комбинация двух победителей работает: LoRA-элемент даёт лучший
  prior (5.680 vs 5.929 full-table), joint-шаг улучшает его сильнее
  (−0.148 vs −0.235 у full — но старт лучше, итог ниже).

## Наблюдение

Joint-шаг на LoRA-prior сработал мягче (−0.148), чем на full-prior
(−0.235): LoRA-merge уже «зашумлён» меньше, дрейфовать некуда —
меньше запас, меньше выигрыш, но итоговое качество выше. Merge_tax
+0.0014 говорит, что физическая сборка почти без потерь — плоский
merge корректен.

## Очередь

1. Ранг-sweep r=8/32/64 (r=16 не обязан быть оптимумом).
2. LoRA-gen2: 3 sibling-листа от lora_gen_joint (как раунд 21, но
   LoRA-элемент) → ансамбль → рекурсия продолжается.
3. Sampled prior-NCE receiver (speed-резерв синтеза).
4. Grad-checkpointing/ns_steps (быстрые победы скорости).
