"""Тест второй стадии квантования строк словаря (HAGI_EMB_ROWS2).

Правило: при выгрузке строки emb с наибольшей ошибкой первой стадии
квантуются ещё раз, и их остаток прибавляется при восстановлении.
Основная раскладка уровней не меняется, объём учитывает обе стадии.

Сторож: если остаток прибавлять с неверным знаком или не учитывать объём,
тест падает (замер с неверным знаком: CE 5.08 -> 8.05 вместо 4.37)."""
import numpy as np  # массивы
import torch as t  # тензоры

import hagi.models.optimal_lm as OM  # модель и квантование
from hagi.models.optimal_lm import HagiOptimalLM  # модель


def _toy_model() -> HagiOptimalLM:  # маленькая обученная модель
    t.manual_seed(0)  # зерно
    m = HagiOptimalLM(); m.spike_k = None  # модель без спайкинга
    data = bytes((i * 7 + (i // 13) * 11) % 256 for i in range(6000))  # структурированные байты
    m.train_optimal(data, epochs=1, batch=16, seed=1)  # краткое обучение
    return m  # вернуть


def test_emb_rows2_improves_and_counts_bytes():  # вторая стадия улучшает восстановление и учтена в объёме
    m = _toy_model()  # модель
    emb = m.emb.detach().clone()  # исходный словарь
    OM.EMB_ROWS2 = 0; q0, nb0 = OM.quantize_model(m, bits=2, use_had=False)  # без второй стадии
    OM.EMB_ROWS2 = 128; q1, nb1 = OM.quantize_model(m, bits=2, use_had=False)  # с ней
    OM.EMB_ROWS2 = 0  # вернуть настройку
    assert nb1 > nb0, f"вторая стадия не учтена в объёме: {nb1} против {nb0}"  # объём растёт
    assert "rows2" in q1["emb"] and "rows2" not in q0["emb"], "вторая стадия не записана в параметры emb"  # записана
    OM.dequantize_into(m, q0); e0 = float((m.emb.detach() - emb).pow(2).sum())  # ошибка без второй стадии
    m.emb.data.copy_(emb); OM.dequantize_into(m, q1); e1 = float((m.emb.detach() - emb).pow(2).sum())  # ошибка с ней
    assert e1 < e0, f"вторая стадия не уменьшила ошибку словаря: {e1:.4f} против {e0:.4f}"  # точнее
    assert q1["emb"]["Q"].shape == q0["emb"]["Q"].shape, "основная раскладка уровней изменилась"  # форма сохранена
