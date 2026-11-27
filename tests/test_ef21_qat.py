"""Тест квантования с обучением (ef21_qat) на маленькой модели.

Свойства:
  - функция выполняется и возвращает упакованные параметры и объём
    (до исправления падала: optim.names — список, а не словарь);
  - тернарная доводка (bits=2) не хуже простого тернарного квантования:
    CE квантованной модели после QAT <= CE после PTQ.

Сторож: если QAT сломается (исключение) или станет вредить, тест падает."""

import torch as t  # тензоры

from hagi.models.optimal_lm import HagiOptimalLM, ef21_qat, dequantize_into, quantize_model  # модель, QAT, PTQ


def _ce(model: "HagiOptimalLM", data: bytes) -> float:  # CE на первых 256 окнах
    arr = t.tensor(list(data[:4096]), dtype=t.long)  # байты
    X = t.stack([arr[i:i + model.k] for i in range(0, 2048)]); y = t.stack([arr[i + model.k] for i in range(0, 2048)])  # окна
    with t.no_grad():  # без графа
        return float(t.nn.functional.cross_entropy(model.logits(X), y))  # кросс-энтропия


def _toy_data(seed: int = 5) -> bytes:  # синтетика: повторы шаблонов + шум
    rng = __import__("random").Random(seed)  # генератор
    out = bytearray()  # накопитель
    for _ in range(600):  # 600 фрагментов
        out += bytes([17, 34, 51, 68, 85] * rng.randint(1, 3)) + bytes([rng.randrange(256) for _ in range(rng.randint(0, 4))])  # шаблон + шум
    return bytes(out)  # байты


def test_ef21_qat_runs_and_beats_ptq():  # QAT работает и не хуже PTQ в тернах
    t.manual_seed(1)  # зерно
    m = HagiOptimalLM()  # модель
    m.spike_k = None  # без спайкинга
    data = _toy_data()  # данные
    m.train_optimal(data, epochs=2, batch=16, seed=3)  # краткое обучение FP32
    fp32 = _ce(m, data)  # CE до квантования
    assert fp32 < 5.0, f"модель не обучилась: CE {fp32:.3f}"  # предусловие
    q, _ = quantize_model(m, bits=2); dequantize_into(m, q); ptq = _ce(m, data)  # PTQ терны
    t.manual_seed(1); m = HagiOptimalLM(); m.spike_k = None; m.train_optimal(data, epochs=2, batch=16, seed=3)  # та же модель заново (PTQ-замер её переписал)
    r = ef21_qat(m, data, steps=30, bits=2)  # доводка (до исправления: AttributeError 'list' has no 'update')
    assert r["bytes"] > 0 and "q" in r, "нет отчёта о квантовании"  # отчёт
    dequantize_into(m, r["q"]); qat = _ce(m, data)  # CE после QAT
    assert qat <= ptq + 0.05, f"QAT хуже PTQ: {qat:.3f} против {ptq:.3f}"  # не хуже
