# Growth Benchmark v2 — равная ёмкость, равный бюджет

Статус: **результат получен** (2026-09-25). Первый в проекте benchmark, где
baseline и growth-кандидат сравниваются без преимущества в compute.

## Почему понадобился новый benchmark

`GROWING_HYPOTHESIS.md` сообщает 4.83 против 5.44 average exact CE для
merged против from-scratch. Этот результат честно описан в самом документе,
но с оговоркой: эксперты до merge видели ~2.6B токенов каждый, поэтому
сравнение доказывало преимущество **полного end-to-end compute**, а не
структуры роста. Ответ на возражение и есть этот benchmark.

## Дизайн

| | merged (кандидат) | baseline |
|---|---|---|
| геометрия | H=1152, L=3, 18q/6kv × 64 | H=1152, L=3, 18q/6kv × 64 |
| параметры | 98.1M total / 22.6M body | 98.1M total / 22.6M body |
| происхождение | merge 3×H=384 доменных экспертов | from scratch |
| joint-шаги | 9000 | 9000 |
| joint-токены | 73.7M | 73.7M |
| mixture | 6 корпусов, равные веса | тот же |
| seed | 1234 | 1234 |

Эксперты: RU (wikipedia_ru + oscar_ru), EN (slimpajama + edu),
MATH+CODE (openwebmath + python_instruct), каждый H=384, 3000 шагов,
24.6M токенов. Их токены **не входят** в joint-бюджет сравнения: обе
финальные модели получили ровно 73.7M токенов.

## Метрика

`scripts/eval_holdout.py` — exact CE по полному алфавиту на хвостовом
участке корпуса, которого обучение не касалось. Все `exact_ce` в логах
трейнера — метрика на тренировочных батчах и для вывода о качестве не
годятся. Скрипт отказывается выдавать оценку, если корпус израсходован.

## Результат (seed 1234, 40 батчей × 128 токенов на домен)

| домен | merged | baseline | delta |
|---|---|---|---|
| ru | 5.2249 | 6.0473 | **-0.8224** |
| en | 6.2891 | 6.6452 | **-0.3561** |
| mathcode | 6.3461 | 6.8545 | **-0.5084** |
| **AVG** | **5.9534** | **6.5157** | **-0.5623** |

Выигрыш в каждом домене отдельно — среднее не скрывает провала.

## Отрицательные гейты, объявленные до запуска

1. exact CE хуже unigram entropy (≈8.06 nats) → stop. **Не сработал.**
2. merged хуже unigram-эксперта на его домене → merge деструктивен.
   **Не проверен отдельно** (см. ограничения).
3. `update_applied=False` > 5% шагов → invalid. **Не сработал** (0 skips).
4. joint-only линия не выигрывает → структурного преимущества нет.
   **Линия выиграла.**
5. Один seed не результат → нужна репликация. **Выполняется.**

## Ограничения — что этот benchmark НЕ доказывает

- **Не масштабирование.** Одна точка (3→1). Нет зависимости от N.
- **Не универсальность.** Измерены три домена, не задачи.
- **Не саморазвитие.** Online-адаптация не участвовала; кредит-landing
  контур остаётся мёртвым (см. `AGENT_WORKLOG.md`, коммиты `1cda6b5`,
  `500ea9e`).
- **Один seed на старте.** Репликация на 5678 обязательна до формулировки
  «установленный факт».
- **CODE-домен неполон.** `python_instruct` — всего 3.1M токенов и израсходован
  обучением; mathcode оценён только по openwebmath. Для честного CODE-holdout
  нужен отдельный корпус.
- **Гейт 2 не выполнен.** Сравнение «merged против его собственного
  unmerged-эксперта на том же домене» не проводилось: оно требует отдельной
  линии с per-domain evaluation экспертов.

## Воспроизведение

```bash
for d in ru en mathcode; do
  python scripts/train.py --config configs/m2_expert_$d.yaml --device cuda
done
python scripts/train.py --config configs/m2_baseline_merged.yaml --device cuda
python scripts/train.py --config configs/m2_merged_joint.yaml --device cuda

python scripts/eval_holdout.py --config configs/m2_merged_joint.yaml \
  --resume checkpoints/m2_merged_joint/step-0009000.pt --batches 40 --device cuda
python scripts/eval_holdout.py --config configs/m2_baseline_merged.yaml \
  --resume checkpoints/m2_baseline_h1152/step-0009000.pt --batches 40 --device cuda
```

Метрики: `reports/m2_merged_holdout.json`, `reports/m2_baseline_holdout.json`.
