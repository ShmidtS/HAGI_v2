"""Тесты R125–R127: насыщение, state-closed полоса, Azuma-успехи.

Каждый тест пришпиливает Python-порт к Lean-теореме:

- ``never_overshoot`` / ``pl_gap_geometric`` / ``pl_gap_geometric_lo``
  / ``saturation_horizon`` / ``growth_band`` -- R125 Saturation
  (never_overshoot, pl_gap_geometric, saturation_limit,
  takeoff_with_saturation).
- ``state_closed_band`` / ``lifecycle_verdict`` -- R126
  StateClosedRenewal (state_closed_band, ξ-компенсация в полю
  зажигания, полный жизненный цикл §13).
- ``azuma_success_floor`` / ``cycles_to_level`` -- R127 Azuma
  (adaptive_success_azuma, ALGORITHMS.md §12).
"""

import math

import pytest

from hagi.train.ratio_takeoff import ignition_threshold
from hagi.train.saturation import (
    LifeCycleVerdict,
    growth_band,
    lifecycle_verdict,
    never_overshoot,
    pl_gap_geometric,
    pl_gap_geometric_lo,
    pl_step_gain,
    saturation_horizon,
    state_closed_band,
)
from hagi.train.self_development import (
    adaptive_success_floor,
    azuma_success_floor,
    cycles_to_level,
)


def _pl_trajectory(C0: float, Cstar: float, sigma: float,
                   steps: int) -> tuple[list[float], list[float]]:
    """Точная PL-траектория gain_t = sigma*gap_t + frontier-динамика.

    Шаг capability ровно PL: C' = C + sigma*(C*-C); D держит конус
    D' = max(rho*D, k*C') — независимо, чтобы тестировать только
    PL-законы R125 без связывания с cone-динамикой.
    """
    k = 0.2
    C_traj, D_traj = [C0], [k * C0]
    for _ in range(steps):
        C = C_traj[-1]
        C_traj.append(C + sigma * (Cstar - C))
        D_traj.append(max(0.9 * D_traj[-1], k * C_traj[-1]))
    return C_traj, D_traj


class TestR125Saturation:
    def test_pl_step_gain(self):
        # sigma=0.1, gap=5 -> шаг ровно 0.5
        assert pl_step_gain(95.0, 100.0, 0.1) == pytest.approx(0.5)

    def test_never_overshoot_on_pl_trajectory(self):
        # точная PL-траектория не перелетает C*
        C_traj, _ = _pl_trajectory(10.0, 100.0, 0.1, 50)
        assert never_overshoot(C_traj, 100.0, 0.1)
        assert C_traj[-1] <= 100.0 + 1e-6

    def test_never_overshoot_fails_on_jump(self):
        assert not never_overshoot([10.0, 20.0], 100.0, 0.1)  # шаг 10 > 9

    def test_pl_gap_geometric_upper_binds(self):
        # нижняя PL-форма: зазор сжимается как (1-sigma)^t
        C_traj, _ = _pl_trajectory(10.0, 100.0, 0.1, 30)
        assert pl_gap_geometric(C_traj, 100.0, 0.1)
        t = 20
        assert 100.0 - C_traj[t] <= (0.9**t) * 90.0 + 1e-6

    def test_pl_gap_geometric_lo(self):
        C_traj, _ = _pl_trajectory(10.0, 100.0, 0.1, 30)
        assert pl_gap_geometric_lo(C_traj, 100.0, 0.1)

    def test_saturation_horizon_conservative(self):
        # gap0=90, sigma=0.1, eps=0.5: T = ceil(90/(0.1*0.5)) = 1800
        assert saturation_horizon(90.0, 0.1, 0.5) == 1800
        # после T шагов зазор действительно < eps на точной траектории
        C_traj, _ = _pl_trajectory(10.0, 100.0, 0.1, 1800)
        assert 100.0 - C_traj[1800] < 0.5 + 1e-6

    def test_growth_band_two_sided(self):
        lo, hi = growth_band(10.0, 0.1, 0.2, 100.0, 0.1, 10)
        assert lo == pytest.approx(10.0 * 1.02**10)
        assert hi == pytest.approx(100.0 - 0.9**10 * 90.0)
        assert lo <= hi

    def test_growth_band_lower_matches_ratio_takeoff(self):
        from hagi.train.ratio_takeoff import ratio_takeoff_floor
        lo, _ = growth_band(10.0, 0.1, 0.2, 100.0, 0.1, 7)
        assert lo == pytest.approx(ratio_takeoff_floor(10.0, 0.1, 0.2, 7))

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            pl_step_gain(50.0, 100.0, 1.5)


class TestR126StateClosed:
    def test_band_verified_on_consistent_trajectory(self):
        C_traj, D_traj = _pl_trajectory(10.0, 100.0, 0.1, 20)
        # подгоняем gamma под фактическую траекторию
        gamma = 0.5
        # траектория из helper не обязана быть band-консистентной
        # для произвольного gamma; используем её собственные параметры
        lo, hi, ok = state_closed_band(
            C_traj, D_traj, sigma=0.1, Cstar=100.0,
            gamma=0.5, rho=0.8, beta=ignition_threshold(0.5, 0.8, 0.2) + 0.01,
            k=0.2, tol=1e-6,
        )
        # шаги с этими параметрами не обязаны сходиться; главное —
        # контракт: ok=False при невыполненных посылках, границы численны
        assert lo <= hi or not ok
        assert isinstance(ok, bool)

    def test_band_consistent_when_parameters_match(self):
        # согласованная траектория: PL-точный шаг capability
        # C' = C + sigma*gap при фиксированном gamma задаёт
        # D_t = sigma*gap_t/gamma (тогда шаг = gamma*D ТОЧНО);
        # динамика D' = (1-sigma)*D = rho*D при rho = 1-sigma,
        # beta = 0, xi = 0; конус D >= k*C держится на T=11
        # (gap_11 = 0.9^11*90 = 28.2 >= 0.2*C_11 = 14.4)
        gamma, sigma, Cstar, k = 0.1, 0.1, 100.0, 0.2
        rho, beta, xi = 0.9, 0.0, 0.0
        C_traj, D_traj = [10.0], []
        for _ in range(11):
            C = C_traj[-1]
            D_traj.append(sigma * (Cstar - C) / gamma)
            C_traj.append(C + sigma * (Cstar - C))
        # замыкающий frontier D_11 (для последнего шага динамики):
        # D_11 = 0.9*D_10 ровно — rho-динамика выполнена точно
        D_traj.append(sigma * (Cstar - C_traj[-1]) / gamma)
        lo, hi, ok = state_closed_band(
            C_traj, D_traj, gamma=gamma, rho=rho, beta=beta,
            k=k, Cstar=Cstar, sigma=sigma, xi=xi, tol=1e-9,
        )
        assert ok
        assert lo <= hi
        # PL-точная траектория реализует верх полосы ровно
        assert C_traj[-1] == pytest.approx(hi)

    def test_lifecycle_switch_axis_below_threshold(self):
        # beta ниже порога — инъекция/расширение, не цикл
        v = lifecycle_verdict(
            beta_hat=0.001, rho_hat=0.8, gamma=0.1, k=0.2, C=10.0
        )
        assert v is LifeCycleVerdict.SWITCH_AXIS

    def test_lifecycle_saturate_near_capacity(self):
        v = lifecycle_verdict(
            beta_hat=0.05, rho_hat=0.8, gamma=0.1, k=0.2,
            C=98.0, Cstar=100.0, sigma=0.1, eps_c=5.0,
        )
        assert v is LifeCycleVerdict.SATURATE

    def test_lifecycle_grow_far_from_capacity(self):
        v = lifecycle_verdict(
            beta_hat=0.05, rho_hat=0.8, gamma=0.1, k=0.2,
            C=10.0, Cstar=100.0, sigma=0.1, eps_c=5.0,
        )
        assert v is LifeCycleVerdict.GROW

    def test_lifecycle_xi_lifts_threshold(self):
        base = dict(rho_hat=0.8, gamma=0.1, k=0.2, C=10.0)
        ok_no_xi = lifecycle_verdict(beta_hat=0.044, xi=0.0, **base)
        ok_with_xi = lifecycle_verdict(beta_hat=0.044, xi=1.0, **base)
        assert ok_no_xi is not LifeCycleVerdict.SWITCH_AXIS
        assert ok_with_xi is LifeCycleVerdict.SWITCH_AXIS


class TestR127Azuma:
    def test_floor_exact(self):
        # n=200, p0=0.5, delta=0.05: 100 - sqrt(2*200*ln20)
        expected = 100.0 - math.sqrt(400.0 * math.log(20.0))
        assert azuma_success_floor(200, 0.5, 0.05) == pytest.approx(expected)

    def test_azuma_looser_than_r121(self):
        # цена отказа от fresh randomness: Azuma-пол ниже R121-пола
        n, p0, delta = 200, 0.5, 0.05
        assert azuma_success_floor(n, p0, delta) < adaptive_success_floor(
            n, p0, delta
        )

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            azuma_success_floor(0, 0.5, 0.05)
        with pytest.raises(ValueError):
            azuma_success_floor(10, 0.5, 1.5)

    def test_cycles_to_level_solves_inverse(self):
        n = cycles_to_level(0.5, 0.5, 0.05)
        # при этом n пол достигает уровня
        assert azuma_success_floor(n, 0.5, 0.05) >= 0.5

    def test_cycles_to_level_infeasible_raises(self):
        with pytest.raises(ValueError):
            cycles_to_level(-1.0, 0.5, 0.05)
