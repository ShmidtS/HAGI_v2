# HAGI

**HAGI** is a **language model that grows in generations**. Train
three small specialist models (leaves), merge them into one larger
model exactly (at step 0 the merged model behaves identically to the
ensemble of the three), add a cheap communication channel between
the blocks, and briefly fine-tune. The result seeds three new
specialists, and the cycle repeats: 3 leaves → merge → fine-tune →
next generation. At fixed compute, many narrow specialists beat one
wide from-scratch model (measured).

Full architecture and code description: **[ARCHITECTURE.md](ARCHITECTURE.md)**
(Russian; this summary is the English entry point).

---

**Языковая модель, растущая поколениями.** Обучаем три небольшие
модели-специалиста (листья), объединяем их в одну большую точным
слиянием (на первом шаге объединённая модель работает ровно как
ансамбль трёх), добавляем дешёвую связь между блоками и дообучаем.
Полученная модель становится отправной точкой для трёх новых
специалистов — и цикл повторяется: 3 листа → слияние → дообучение →
новое поколение.

Полное описание архитектуры и кода — **[ARCHITECTURE.md](ARCHITECTURE.md)**.
Измеренные результаты и rationale раундов — git history + `.omc/attempts/`.

Версия: **V42** (`hagi-channel-v42`, 4.2.0).

## Схема роста (победная dbridge-линия)

```
8 корпусов (смешанные окна, packed-corpus)
  → 3 листа H=128 (dense тело, GQA + QK-norm + RoPE, SwiGLU, fused CE)
  → троичное слияние → gen-1 H=384 + joint
  → 3 сиба H=384 (init_from gen-1 joint, новые data-seed)
  → троичное слияние → gen-2 H=1152 (Hadamard-миксер, BranchScale clamp-8)
    + joint
  → TableLoRA r16 на embedding/head поверх
  → gen-3: та же схема от gen-2 joint
```

Слияние — **троичное 3→1 на каждом уровне** (не степени двойки);
Hadamard-трансформ строится для любого N (pad → QR-ортонормализация).
Шаг-0 merged-модели — ровно ансамбль экспертов (Jensen-выигрыш бесплатно),
затем единственные новые связи — миксер и joint-обучение.

## Быстрый старт

```bash
pip install -e .

# лист (эксперт H=128)
python -u scripts/train.py --config configs/dbridge_leaf_s1.yaml

# слияние 3 экспертов (gen-1)
python -u scripts/train.py --config configs/dbridge_gen1_merged.yaml
python -u scripts/train.py --config configs/dbridge_gen1_joint.yaml

# гейт: weighted exact CE на 8 корпусах (tail-окна, 2×1024/корпус)
# см. протокол в ARCHITECTURE.md §4
```

Тесты: `python -X utf8 -m pytest tests -q` (24 passed).

## Стек

- Python 3.13, PyTorch (ROCm/HIP), AMD iGPU
- `src/hagi/` — модель, merge, обучение, инференс (см. ARCHITECTURE.md §1)
- `scripts/` — CLI: train/merge/gate-аудиты/генерация; `dsv4_*` —
  отдельный экспериментальный трек сжатия DeepSeek-V4 (не часть линии HAGI)
- `configs/` — YAML победной линии (dbridge_*) + тестовая фикстура
- `data/` — компактные корпуса (32768 словарь), mix.json

## Принципы

1. При равном бюджете вычислений три узких специалиста лучше
   одного широкого «с нуля»: каждый учится на своём распределении
   данных, а слияние объединяет их без потери качества (измерено:
   выигрыш ~1 нат против обучения с нуля тем же бюджетом).
2. Merge — центральный механизм; коммуникация поверх, дёшево.
3. Function-preserving инварианты обязательны (шаг-0 тождество).
4. Не подбирать гиперпараметры перебором — вычислять их из
   измерений самой модели. Примеры: размер низкоранговых добавок
   (LoRA-ранг) — из спектра сингулярных чисел (какая доля сигнала
   реально не покрыта); масштабы ветвей — из измеренной дисперсии
   активаций каждого слоя (BranchScale); размер батча — из
   измеренного шума градиента (на этой линии шум исчезает уже
   на ~100 токенах, значит большой батч ничего не даёт);
   «запускать ли дорогой эксперимент» — сначала дешёвый расчёт-
   сертификат на уже обученной модели, и только если он говорит
   «да», тратить GPU-часы.
5. Отбор кандидатов по diversity (Fisher/Jensen), не по standalone CE.
6. Иерархия каналов: специализация → merge → коммуникация → joint →
   low-rank residual.
