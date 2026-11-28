"""Тест сертифицированного гейта по наихудшему домену.

Правило: гейт сравнивает не точечный прирост CE домена, а прирост за вычетом
радиуса шума разности (эмпирический Бернстайн на одних окнах до/после, поправка
Бонферрони на WORST_GATE_BONF доменов): откат, только если
grow − radius > WORST_TOL_ROLLBACK.

Сторож: точечный порог при малом числе окон ломается на шумe (замер: 138
доменов, 16 окон, нулевой истинный рост — точечный порог срабатывал в 5.8 %
случаев на домен, то есть почти наверняка хотя бы раз за приём); а с полным
Бонферрони по всем 167 доменам радиус 1.73 ната пропускал разрушение на 1.75 ната
(поэтому WORST_GATE_BONF = 16). Сами отказы ловятся тестами test_noise_does_not_trip
и test_real_damage_trips; test_point_threshold_trips_on_noise показывает, что
точечный порог при 16 окнах ложится."""
import numpy as np  # массивы

from hagi.base.trunk_ops import paired_margin_eb  # радиус
import hagi.selfdev.unified as U  # цикл


def _gate(pre: np.ndarray, post: np.ndarray, n_bonf: int, tol: float = 0.25) -> tuple[bool, float, float]:
    """Верdict гейта для одного домена: (откат?, точечный прирост, радиус)."""
    grow = float(post.mean() - pre.mean())  # точечный прирост CE
    radius = paired_margin_eb(pre - post, n_states=n_bonf, delta=0.05)  # радиус шума разности
    return grow - radius > tol, grow, float(radius)  # тройка


def test_noise_does_not_trip():  # шум не вызывает откат
    rng = np.random.default_rng(0)  # генератор
    trips = 0  # счётчик срабатываний
    for _ in range(200):  # 200 доменов-псевдоприёмов
        base = rng.normal(2.0, 1.6, 237)  # CE окон до (типичный разброс окна ~1.6)
        post = base + rng.normal(0.0, 0.64, 237)  # после: тот же уровень, шум разности 0.64
        hit, grow, radius = _gate(base, post, n_bonf=U.Unified.WORST_GATE_BONF)  # вердикт
        trips += hit  # учесть
        assert radius < 0.6, f"радиус на 237 окнах слишком велик: {radius:.3f}"  # радиус применим
    assert trips == 0, f"сертифицированный гейт сработал на шуме {trips} раз из 200"  # ни одного ложного


def test_point_threshold_trips_on_noise():  # точечный порог при малом числе окон ложится на шуме (зачем нужен радиус)
    rng = np.random.default_rng(0)  # генератор
    trips = 0  # счётчик срабатываний точечного порога
    for _ in range(200):  # 200 доменов, 16 окон (бюджет 2048 на ~138 доменов)
        base = rng.normal(2.0, 1.6, 16); post = base + rng.normal(0.0, 0.64, 16)  # нулевой истинный рост, шум разности 0.64
        trips += float(post.mean() - base.mean()) > 0.25  # точечный порог
    assert trips >= 5, f"точечный порог должен ложиться на шуме при 16 окнах (срабатываний {trips}/200)"  # контроль


def test_real_damage_trips():  # настоящее разрушение ловится
    rng = np.random.default_rng(1)  # генератор
    base = rng.normal(2.0, 1.6, 237); post = base + 1.0 + rng.normal(0.0, 0.64, 237)  # истинный рост +1.0 ната
    hit, grow, radius = _gate(base, post, n_bonf=U.Unified.WORST_GATE_BONF)  # вердикт
    assert hit, f"разрушение +1.0 ната не поймано: прирост {grow:.3f}, радиус {radius:.3f}"  # поймано
