# HAGI — текущее состояние

Архитектура и код — [ARCHITECTURE.md](ARCHITECTURE.md). Этот файл —
краткое состояние: рекорды, живые чекпоинты, очередь. Полная история
раундов с измерениями — git history и `.omc/attempts/<round>.md`.

## Рекорды (mixed 8-corpora gate, exact CE, одинаковые окна)

| Рука | CE |
|---|---|
| LoRA r16 поверх clamp-8 (рекорд) | **3.3928** |
| clamp-8 merged+joint | 3.4354 |
| gen-2 joint (swiglu) | 3.4484 |
| gen-2 LoRA r16 | 3.4240 |
| gen-1 joint | 3.6357 |
| scratch H=384 (равный бюджет) | 4.6342 |

Траектория сессии: 6.12 → 3.3928 (−2.73 ната). Equal-compute
контроль: рост бьёт scratch на ~1 нат.

## Живые чекпоинты (checkpoints/)

- `dbridge_leaf_s1..s3`, `dbridge_loop2_s1..s3`, `leaf_rank384`,
  `leaf_s1_c8` — листья/руки
- `dbridge_gen1_merged/joint`, `dbridge_gen2_sib1-3`,
  `dbridge_gen2_merged_had`, `dbridge_gen2_merged_had_c8` (база
  рекорда), `dbridge_gen2_lora_c8` (рекорд)
- `dbridge_scratch_h384`, `f3_d1`, `merged_had` — контроли

## Конфиги (configs/, 20)

dbridge-линия (leaf, gen1/gen2 merged/joint, сибы, loop2, f3_d1,
scratch, rank384, leaf_c8) + `parent_h128_flat.yaml` (тестовая
фикстура merge-инварианта).

## Закрытые оси (измерено/доказано — не тратить время)

Рост батча (B_noise ≈ 100 токенов), адаптивный NS (5 шагов
достаточно), полный SwiGLU-миксер (hadamard равен, 12× дешевле),
head_up-проекция (ранговый потолок не активен), F3-дерево depth-1
(root-кора теряет 4.2% энергии; highway-прогноз 0.023 ната — не
окупает), LazyAdam (0.14% шага), cortex/adapters/decision/NCE.

## Очередь

1. Gen-3: 3 сиба H=1152 от `gen2_merged_had_c8` joint → merge →
   joint (movement-канал; затухание 0.35 → 0.19, ожидание ~0.1)
2. LoRA-углубление (r=32/64) поверх рекорда (rank-канал)
3. Gram-scan на первом joint-ранне (`train.gram_scan_interval: 200`,
   log-only) — измерить частоту градиентных конфликтов корпусов;
   при > 0 — применить SafeQP-проекцию
4. N*-адаптивность: число экспертов по Fisher/Jensen per wall-clock
5. Wave-7 формализация (CycleDecay + ChannelBudget) у формализатора

## Процесс

Одна тренировка на GPU единовременно (проверка процессов через
`powershell Get-CimInstance Win32_Process`); генераторы train()
молчат — трекать по чекпоинтам/логам, `python -u`; сравнение только
на идентичных окнах; 3 провала подряд → смена подхода; `.omc/` в git
через `add -f`.
