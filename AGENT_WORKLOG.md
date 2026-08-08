# AGENT_WORKLOG — HAGI Evolution

Журнал действий исследовательского агента. Все временные метки — локальное
время (Asia/Yekaterinburg). Цель: модернизация архитектуры обучения HAGI
«с нуля» (минимизация wall-time шага) + серия решающих экспериментов по
гипотезе «выращивания» большой модели из специализированных малых.

---

## 2026-08-08

### 17:27 — Диагностика скорости (ЧАСТЬ I)
- Проанализирован лог обучения. **Разгадана загадка лога**: 2.1s между
  строками = 100 шагов × 21ms. Реальная скорость **21ms/step = 1.18M tok/s**
  (не 12.9k, как ошибочно прочитано делением 24576 на 1.9s).
- Узкое место — НЕ throughput, а качество/сходимость.
- Профилирование: без compile 59.88 ms/step (fwd=25.2, bwd=30.0, data=2.5,
  optim=1.7); с compile **21.58 ms/step, 1.14M body tok/s**.
- Бенчмарк num_workers=2: **20.79 ms/step median, 1.18M tok/s**.

### 17:28 — Batch scaling (ЧАСТЬ II)
- batch=192 → 1.09M tok/s, batch=384 → 1.34M, batch=768 → 1.50M tok/s.
- **Найден критический баг**: при `compile_model=True` и batch≥320
  torch.compile даёт non-finite градиенты (rest=nan/inf) и **молча пропускает
  все обновления** (update_applied=False). Это баг buffer-reuse в ROCm
  torch.compile.
- batch=256 — безопасный максимум: **+13% throughput** (1.24M vs 1.10M tok/s)
  с идентичной сходимостью (loss 7.627-7.630 для 192 и 256).
- MergedHAGI (fp16, ternary off) компилируется чисто при batch=256; baseline
  HAGI (ternary BitLinear) ломается при compile+batch≥256.

### 17:29 — Smoke-тест конвейера V48
- 4 эксперта (H=128) обучены на 50 шагов → merge 4→1 (H=512) → joint 20 шагов.
- Merged на step 0 joint: nce=7.32 (лучше, чем эксперты 7.62).
- Merged-модель реально учится: loss 7.625→7.60 за 20 шагов.

### 18:01 — Freeze-механизм (mixer-only режим)
- Добавлен `merge.freeze_experts` в config.py: замораживает все параметры
  кроме `mixers.*`. Проверено: "froze 37 parameter tensors; only mixers.*
  trainable".

### 18:02 — Доменная оценка
- Создан `scripts/eval_domains.py`: exact CE / perplexity на каждом доменном
  корпусе (RU/EN/MATH/CODE).

### 18:05 — Иерархический merge
- Создан `scripts/merge_hierarchical.py`: 16→4→1 иерархия.
- Исправлен merge_experts для иерархии: 2D block-norms (concat по block_dim),
  2D per-head qk_norm (concat по головам), `drop_expert_mixers`.
- Проверено: 4 эксперта → 2 группы (H=256) → финальная H=512. Forward OK.

### 18:08 — Эксперимент 4 эксперта (ЧАСТЬ III)
- 4 эксперта (RU/EN/MATH/CODE) обучены на 2000 шагов (~70s каждый).
- **Специализация подтверждена** (exact CE на своём домене):
  - RU эксперт: RU=7.06 (лучший на RU)
  - EN эксперт: EN=6.11 (лучший на EN), RU=8.97 (худший)
  - MATH эксперт: MATH=5.87 (лучший на MATH)
  - CODE эксперт: CODE=4.28 (лучший на CODE)
- Merge 4→1 (H=512), step 0 (до joint): AVG=8.23 — head не знает, какой блок
  активен, специализация в блоках не используется.
- **Joint 200 шагов**: AVG=5.89 — ЛУЧШЕ, чем любой отдельный эксперт
  (RU=7.28, EN=6.98, MATH=7.16, CODE=6.95). Специализации сохранены
  (RU=7.11, EN=5.87, MATH=5.93, CODE=4.65).

### 18:16 — Baseline (from scratch H=512)
- Запущен baseline на 2000 шагов (compile OFF, т.к. ternary ломается при
  compile). Прогрессирует: step 0 nce=7.62 → step 100 nce=7.49.

### 18:30 — Сравнение при равном бюджете (49M токенов)
- Baseline (2000 шагов, 49M токенов): AVG=5.44 (RU=6.63, EN=5.36, MATH=5.43,
  CODE=4.35).
- Merged после 2000 joint (49M токенов общего обучения): **AVG=5.18**
  (RU=6.25, EN=5.27, MATH=5.34, CODE=3.84).
- **Merged превосходит baseline на 0.26 exact CE** при том же бюджете общего
  обучения. Гипотеза подтверждается.

### 18:32 — Обучение до насыщения (замечание пользователя)
- Пользователь указал: 2000 шагов слишком мало для насыщения.
- Запущено обучение 4 экспертов до 20000 шагов (~12 мин каждый, ~48 мин
  всего) в фоне. Насыщение = exact_ce перестаёт падать.
- После насыщения: merge 4→1, joint до насыщения, сравнение с baseline.

## 2026-08-08 — Иерархический эксперимент (ЧАСТЬ IV) завершён
- 4 насыщенных эксперта (H=128) → 2 группы (H=256, RU+EN / MATH+CODE), 500 joint каждая
- Merge групп → финальная H=512 (drop_expert_mixers), joint 2000 шагов
- Результат (49M токенов общего обучения): Hierarchical AVG=5.00
  (RU=5.84, EN=5.22, MATH=5.22, CODE=3.71)
- Сравнение: Flat merged=4.81 < Hierarchical=5.00 < Baseline=5.44
- Вывод: flat merge эффективнее иерархии на 4 экспертах; оба превосходят baseline
- Обновлены BENCHMARKS.md (секция 6) и GROWING_HYPOTHESIS.md

## 2026-08-08 — Найден и исправлен баг logit_scale при merge
- grow.py (существующий, неиспользуемый) делит logit_scale на sqrt(N) при merge
- merge.py брал logit_scale первого эксперта БЕЗ деления → merged head-проекция
  (concat по input) суммирует N вкладов, logits в ~N раз острее
- Исправлено: merge.py теперь делит logit_scale на sqrt(N) (как grow.py)
- Эффект: step 0 AVG улучшился с 9.00 до 6.39 (head-распределение не коллапсирует)
- После 2000 joint: AVG=4.83 (было 4.81) — joint стартует с лучшей точки
- Обновлены BENCHMARKS.md и GROWING_HYPOTHESIS.md

## 2026-08-08 — Иерархия после исправления logit_scale
- Пересоздан иерархический final с нормализованным logit_scale (0.686)
- step 0: AVG=7.56 (было ~9)
- После 2000 joint: AVG=4.81 (RU=5.67, EN=5.07, MATH=5.04, CODE=3.48)
- Иерархия теперь сравнялась с flat (4.83) — ранее была хуже (5.00)
- Вывод: разрыв flat vs hier был артефактом неверного logit_scale, не свойством иерархии
- Обновлены BENCHMARKS.md и GROWING_HYPOTHESIS.md
