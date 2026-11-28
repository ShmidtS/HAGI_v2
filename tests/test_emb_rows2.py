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


def test_emb_rows2_auto_picks_rows_by_marginal_contribution():  # auto: строки — top-k по вкладу, k — по предельному вкладу
    m = _toy_model()  # модель
    OM.EMB_ROWS2 = "0"; q0, nb0 = OM.quantize_model(m, bits=2, use_had=False)  # без второй стадии: остаток первой
    W = m.emb.detach().float().cpu().numpy(); par, Q = OM.quant_per_row(W, bits=2, use_had=False, rotate_rows=False)  # первая стадия словаря
    contrib = ((W - OM.dequant_per_row(par, Q)) ** 2).sum(axis=1); order = list(np.argsort(-contrib))  # вклад строк по убыванию
    k_expect = sum(1 for i in order if contrib[i] >= OM.EMB_ROWS2_MIN_FRAC * contrib[order[0]])  # ожидаемое число строк по правилу
    OM.EMB_ROWS2 = "auto"; qa, nba = OM.quantize_model(m, bits=2, use_had=False)  # авто
    OM.EMB_ROWS2 = "0"  # вернуть настройку
    rows = qa["emb"]["rows2"]  # выбранные строки
    assert 1 <= len(rows) <= 256 and len(rows) == k_expect, f"auto выбрал {len(rows)} строк, правило даёт {k_expect}"  # число строк по правилу
    assert sorted(rows) == sorted(order[:k_expect]), "auto выбрал не top-k строк по вкладу"  # ровно top-k
    assert len(set(rows)) == len(rows) and nba > nb0, "строки повторяются или объём не учтён"  # без повторов, объём учтён


def test_row_score_changes_row_choice():  # веса строк меняют выбор строк второй стадии
    m = _toy_model()  # модель
    OM.EMB_ROWS2 = "32"; q_n, _ = OM.quantize_model(m, bits=2, use_had=False); OM.EMB_ROWS2 = "0"  # выбор по норме
    score = np.zeros(256); score[:8] = 1.0  # вес только у первых 8 строк
    OM.EMB_ROWS2 = "32"; q_s, _ = OM.quantize_model(m, bits=2, use_had=False, row_score={"emb": score}); OM.EMB_ROWS2 = "0"  # выбор с весами
    rows_n = set(q_n["emb"]["rows2"]); rows_s = q_s["emb"]["rows2"]  # выбранные строки
    assert set(range(8)) <= set(rows_s), "строки с наибольшим весом не выбраны"  # взвешенные строки обязаны войти
    assert rows_s != sorted(rows_n), "веса не изменили выбор строк"  # выбор изменился
    # строка с нулевым весом идёт после всех строк с положительным весом
    victim = max(rows_n); score0 = np.ones(256); score0[victim] = 0.0  # обнуляем вес строки, выбранной по норме
    OM.EMB_ROWS2 = "255"; q_z, _ = OM.quantize_model(m, bits=2, use_had=False, row_score={"emb": score0}); OM.EMB_ROWS2 = "0"  # 255 из 256 строк
    assert victim not in q_z["emb"]["rows2"], "строка с нулевым весом выбрана раньше строк с положительным весом"  # исключена
