"""Канонический Gated DeltaNet: точный сброс при α = 0 и эквивалентность WY-формы пошаговой."""
import torch as t  # тензоры

from hagi.models.optimal_lm import gated_deltanet  # GDN: пошаговая и блочно-параллельная формы


def test_gdn_reset_forgets_old_state_exactly():
    """При α = 0, β = 1 память после шага равна ровно v kᵀ: старое состояние не просачивается."""
    t.manual_seed(0)  # зерно
    S0 = t.randn(1, 2, 8, 8)  # старое состояние
    Q, K, V = t.randn(1, 2, 1, 8), t.randn(1, 2, 1, 8), t.randn(1, 2, 1, 8)  # один шаг
    _, S = gated_deltanet(Q, K, V, t.zeros(1, 2, 1), t.ones(1, 2, 1), S_init=S0)  # шаг со сбросом
    target = V[0, :, 0].unsqueeze(-1) @ K[0, :, 0].unsqueeze(-2)  # точная запись v kᵀ
    assert t.allclose(S[0], target, atol=1e-6)  # память — ровно новая запись


def test_gdn_chunked_form_matches_stepwise():
    """WY-форма (чанк 16) совпадает с пошаговой рекурсией по выходам и финальному состоянию."""
    t.manual_seed(1)  # зерно
    B, H, T, D = 4, 2, 64, 12  # размеры
    Q, K, V = (t.randn(B, H, T, D) for _ in range(3))  # запросы, ключи, значения
    alpha, beta = t.rand(B, H, T) * 0.5, t.rand(B, H, T) * 1.5  # затухание и запись
    alpha[:, :, 5] = 0.0  # жёсткий сброс внутри окна
    S0 = t.randn(B, H, D, D)  # перенесённое состояние
    y1, S1 = gated_deltanet(Q, K, V, alpha, beta, S0, chunk=10 ** 6)  # пошагово
    y2, S2 = gated_deltanet(Q, K, V, alpha, beta, S0, chunk=16)  # чанками
    assert t.allclose(y1, y2, atol=1e-4) and t.allclose(S1, S2, atol=1e-5)  # одна и та же рекурсия
