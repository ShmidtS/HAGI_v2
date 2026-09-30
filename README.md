# HAGI

**Рекурсивно растущий ансамблевый язык-модель (LM).** Маленькие обученные
эксперты (листья) объединяются точным function-preserving слиянием,
коммуникация — дешёвая (ортогональный транспорт + low-rank residual),
и цикл повторяется поколениями: 3 сиба от prior'а-родителя → троичное
слияние → короткое joint-обучение → родитель становится prior'ом
следующего поколения.

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

1. Покупать разнообразие, а не параметры: много узких экспертов >
   один широкий scratch при равном бюджете (измерено).
2. Merge — центральный механизм; коммуникация поверх, дёшево.
3. Function-preserving инварианты обязательны (шаг-0 тождество).
4. Все ручки — из измерений: ранги из спектра, масштабы из дисперсий,
   batch из B_noise; сертификаты до GPU.
5. Отбор кандидатов по diversity (Fisher/Jensen), не по standalone CE.
6. Иерархия каналов: специализация → merge → коммуникация → joint →
   low-rank residual.
