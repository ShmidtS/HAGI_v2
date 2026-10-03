"""Тесты R123/R124 RatioTakeoff — гейт зажигания роста.

Каждый тест пришпиливает Python-порт к Lean-теореме, из которой
он пришёл:

- ``ignition_threshold`` / ``decay_ok`` / ``bifurcation_verdict`` --
  R124 ``cone_ratio_step`` / ``frontier_decay_no_growth``
  (порог зажигания beta >= gamma*k^2 + (1-rho)*k, декей
  rho >= gamma*k, бифуркация GROW/DECAY/INJECT).
- ``cone_step_from_dynamics`` / ``cone_ratio_invariant`` -- R124
  ``cone_ratio_step`` / ``cone_ratio_invariant`` (пошаговая
  перепроверка конуса по измерениям).
- ``ratio_takeoff_floor`` -- R124 ``ratio_takeoff``
  (C_T >= C0*(1+gamma*k)^T, односторонний, без верхней границы).
- ``frontier_decay_bound`` -- R124 ``frontier_decay_no_growth``
  (D_T <= rho^T*D0).
- ``step_ignition_slack`` -- xi/C-компенсация (докстринг
  cone_ratio_step; ALGORITHMS.md §10).
"""

import pytest

from hagi.train.ratio_takeoff import (
    IgnitionVerdict,
    bifurcation_verdict,
    cone_holds,
    cone_ratio_invariant,
    cone_step_from_dynamics,
    decay_ok,
    frontier_decay_bound,
    ignition_margin,
    ignition_threshold,
    ratio_takeoff_floor,
    step_ignition_slack,
)


class TestR124IgnitionThreshold:
    def test_exact_value(self):
        # gamma=0.1, rho=0.8, k=0.2: 0.1*0.04 + 0.2*0.2 = 0.044
        assert ignition_threshold(0.1, 0.8, 0.2) == pytest.approx(0.044)

    def test_rho_one_cancels_linear_term(self):
        # rho = 1: порог чисто квадратичный gamma*k^2
        assert ignition_threshold(0.5, 1.0, 0.4) == pytest.approx(0.5 * 0.16)

    def test_zero_gamma_or_k(self):
        assert ignition_threshold(0.0, 0.8, 0.2) == pytest.approx(0.2 * 0.2)
        assert ignition_threshold(0.1, 0.8, 0.0) == 0.0

    def test_margin_sign(self):
        assert ignition_margin(0.05, 0.1, 0.8, 0.2) == pytest.approx(0.006)
        assert ignition_margin(0.04, 0.1, 0.8, 0.2) == pytest.approx(-0.004)

    def test_decay_ok(self):
        assert decay_ok(0.8, 0.1, 0.2)  # 0.8 >= 0.02
        assert not decay_ok(0.01, 0.1, 0.2)  # 0.01 < 0.02


class TestR124ConeStep:
    def test_cone_holds_on_exact_edge(self):
        # Точная динамика D' = rho*D + beta*C, C' = C + gamma*D на
        # границе порога зажигания: конус сохраняется (Lean
        # cone_ratio_step при равенстве в hbeta).
        gamma, rho, beta, k = 0.1, 0.8, 0.044, 0.2
        C = 10.0
        D = k * C  # старт на границе конуса
        assert cone_step_from_dynamics(C, D, gamma, rho, beta, k)

    def test_cone_breaks_below_threshold(self):
        # beta ниже порога — конус сжимается: измеренный шаг честно
        # проваливает перепроверку (отличие от хрупкого h_C_cap).
        gamma, rho, k = 0.1, 0.8, 0.2
        beta = 0.02  # < 0.044
        C = 10.0
        D = k * C
        assert not cone_step_from_dynamics(C, D, gamma, rho, beta, k)

    def test_xi_compensation(self):
        # трение сдвигает эффективный порог вверх на xi/C:
        # beta = порог + slack*0.1, xi = 0.15 съедает slack
        gamma, rho, k = 0.1, 0.8, 0.2
        beta = 0.044 + 0.01
        C = 10.0
        D = k * C
        ok_no_xi = cone_step_from_dynamics(C, D, gamma, rho, beta, k)
        ok_with_xi = cone_step_from_dynamics(C, D, gamma, rho, beta, k, xi=0.15)
        assert ok_no_xi
        assert not ok_with_xi

    def test_cone_holds_direct(self):
        assert cone_holds(10.0, 2.0, 0.2)
        assert not cone_holds(10.0, 1.9, 0.2)

    def test_cone_invariant_on_trajectory(self):
        # инвариант по всей траектории с beta выше порога
        gamma, rho, beta, k = 0.1, 0.8, 0.05, 0.2
        traj_C, traj_D = [10.0], [2.0]
        for _ in range(10):
            C_prev, D_prev = traj_C[-1], traj_D[-1]
            traj_C.append(C_prev + gamma * D_prev)
            traj_D.append(rho * D_prev + beta * C_prev)
        assert cone_ratio_invariant(traj_C, traj_D, k)

    def test_cone_invariant_breaks_when_decay_too_fast(self):
        # rho < gamma*k: конус сжимается из состояния с большим
        # surplus D (r -> rho/gamma = 0.1 < k при большом D)
        gamma, rho, k = 0.5, 0.05, 0.2
        beta = 0.5
        traj_C, traj_D = [10.0], [100.0]
        for _ in range(10):
            C_prev, D_prev = traj_C[-1], traj_D[-1]
            traj_C.append(C_prev + gamma * D_prev)
            traj_D.append(rho * D_prev + beta * C_prev)
        assert not cone_ratio_invariant(traj_C, traj_D, k)


class TestR124RatioTakeoff:
    def test_floor_exact(self):
        assert ratio_takeoff_floor(10.0, 0.1, 0.2, 3) == pytest.approx(
            10.0 * (1.02) ** 3
        )

    def test_floor_T_zero(self):
        assert ratio_takeoff_floor(7.0, 0.1, 0.2, 0) == 7.0

    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            ratio_takeoff_floor(7.0, 0.1, 0.2, -1)

    def test_no_upper_bound_semantics(self):
        # односторонний сертификат: рост может быть и быстрее
        C0, gamma, k = 10.0, 0.1, 0.2
        floor = ratio_takeoff_floor(C0, gamma, k, 100)
        # система на границе конуса с запасом beta растёт выше floor
        C, D = C0, k * C0
        for _ in range(100):
            C, D = C + gamma * D, 0.9 * D + 0.05 * C
        assert C >= floor


class TestR124DecayBranch:
    def test_decay_bound_exact(self):
        assert frontier_decay_bound(5.0, 0.7, 4) == pytest.approx(0.7**4 * 5.0)

    def test_T_zero(self):
        assert frontier_decay_bound(5.0, 0.7, 0) == 5.0

    def test_negative_rho_raises(self):
        with pytest.raises(ValueError):
            frontier_decay_bound(5.0, -0.1, 3)

    def test_rho_one_no_decay(self):
        assert frontier_decay_bound(5.0, 1.0, 100) == pytest.approx(5.0)


class TestR124BifurcationVerdict:
    def test_grow(self):
        # beta=0.05 >= 0.044, rho=0.8 >= 0.02 -> GROW
        v = bifurcation_verdict(0.05, 0.8, 0.1, 0.2)
        assert v is IgnitionVerdict.GROW

    def test_inject_below_threshold(self):
        # beta > 0, но ниже порога -> INJECT (производство есть,
        # не покрывает расширение конуса)
        v = bifurcation_verdict(0.02, 0.8, 0.1, 0.2)
        assert v is IgnitionVerdict.INJECT

    def test_decay_no_production(self):
        v = bifurcation_verdict(0.0, 0.7, 0.1, 0.2)
        assert v is IgnitionVerdict.DECAY

    def test_decay_not_routable_to_grow(self):
        # rho < gamma*k убивает GROW даже при огромном beta
        # (декей быстрее расширения конуса) -> INJECT
        v = bifurcation_verdict(10.0, 0.01, 0.5, 0.2)
        assert v is IgnitionVerdict.INJECT

    def test_xi_lifts_threshold(self):
        # трение xi при capability C поднимает порог: beta, бывший
        # GROW, становится INJECT
        gamma, rho, beta, k = 0.1, 0.8, 0.044, 0.2
        assert bifurcation_verdict(beta, rho, gamma, k) is IgnitionVerdict.GROW
        assert (
            bifurcation_verdict(beta, rho, gamma, k, C=10.0, xi=1.0)
            is IgnitionVerdict.INJECT
        )

    def test_beta_zero_rho_one(self):
        # beta=0, rho=1: frontier не затухает, но и не растёт —
        # не DECAY (нет геометрического затухания), инъекция
        assert (
            bifurcation_verdict(0.0, 1.0, 0.1, 0.2)
            is IgnitionVerdict.INJECT
        )


class TestR124IgnitionSlack:
    def test_slack_positive_above_threshold(self):
        # beta*C - beta_ign*C - xi = 0.054*10 - 0.044*10 - 0.05 = 0.05
        assert step_ignition_slack(10.0, 2.0, 0.1, 0.8, 0.054, 0.2, 0.05) == (
            pytest.approx(0.05)
        )

    def test_slack_negative_signals_injection(self):
        assert step_ignition_slack(10.0, 2.0, 0.1, 0.8, 0.04, 0.2, 0.0) < 0.0

    def test_nonpositive_C_raises(self):
        with pytest.raises(ValueError):
            step_ignition_slack(0.0, 2.0, 0.1, 0.8, 0.05, 0.2)
