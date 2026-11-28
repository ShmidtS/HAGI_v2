"""Тест гейта по опоре (HAGI_ANCHOR_GATE).

Правило: при принятом домене в книгу записывается опора gate_anchor — его CE на
равномерной сетке окон. Гейт сравнивает текущую CE старых доменов с опорой, а не
с «до этого приёма»: три приёма по +0.2 проходят порог 0.25 по одному, а по опоре
дают +0.6 и видны сразу. Домены без опоры (откатанные, принятые до включения)
гейт по опоре не проверяет.

Сторож: если опору не записывать или не вычитать запас шума, тесты падают."""
import numpy as np  # массивы

import hagi.selfdev.unified as U  # цикл


class _Book:  # заглушка цикла: только книга доменов
    def __init__(self) -> None:  # пустая книга
        self.domains: dict = {}  # домены
        self.WORST_TOL_ROLLBACK = 0.25; self.ANCHOR_MIN_WINDOWS = 32; self.ANCHOR_SE_MULT = U.Unified.ANCHOR_SE_MULT  # допуск дрейфа, минимальное число окон, запас на шум (как в цикле)
        self._vec_names = ["d_new", "d_a", "d_b"]; self._vec_bounds = [0, 64, 128, 192]  # раскладка вектора: новый, два старых
    _anchor_exceeds = U.Unified._anchor_exceeds  # проверяемый метод (берётся у реального цикла)


def test_anchor_gate_sees_accumulated_drift():  # накопленный дрейф виден сразу
    b = _Book(); b.domains = {"d_a": {"gate_anchor": 1.0}, "d_b": {"gate_anchor": 2.0}}  # опоры
    v = np.concatenate([np.full(64, 0.5), np.full(64, 1.2), np.full(64, 2.05)])  # d_a: +0.2 (по одному приёму — в допуске), d_b: +0.05
    hit, drift, nm = b._anchor_exceeds(v)  # вердикт
    assert not hit and nm == "d_a", f"допустимый дрейф принят за превышение: {hit} {drift} {nm}"  # +0.2 в допуске
    v2 = np.concatenate([np.full(64, 0.5), np.full(64, 1.6), np.full(64, 2.05)])  # d_a: +0.6 от опоры (накопление)
    hit2, drift2, nm2 = b._anchor_exceeds(v2)  # вердикт
    assert hit2 and nm2 == "d_a", f"накопленный дрейф +0.6 не пойман: {hit2} {drift2} {nm2}"  # пойман
    assert drift2 > 0.25, f"величина дрейфа занижена: {drift2}"  # величина


def test_anchor_gate_ignores_domains_without_anchor():  # домены без опоры не проверяются
    b = _Book(); b.domains = {"d_a": {}, "d_b": {"gate_anchor": 2.0}}  # у d_a опоры нет
    v = np.concatenate([np.full(64, 0.5), np.full(64, 9.0), np.full(64, 2.05)])  # d_a сильно вырос, но опоры нет
    hit, drift, nm = b._anchor_exceeds(v)  # вердикт
    assert not hit, f"домен без опоры вызвал откат: {hit} {nm}"  # не блокирует


def test_anchor_gate_skips_domains_with_few_windows():  # мало окон — не проверяется (иначе проценты ложных откатов)
    b = _Book(); b._vec_bounds = [0, 10, 20, 30]; b.domains = {"d_a": {"gate_anchor": 1.0}}  # 10 окон у домена
    v = np.concatenate([np.full(10, 0.5), np.full(10, 3.0), np.full(10, 0.0)])  # d_a: +2.0, но окон мало
    hit, drift, nm = b._anchor_exceeds(v)  # вердикт
    assert not hit and drift is None, f"домен с 10 окнами проверен по опоре: {hit} {drift}"  # пропущен


def test_anchor_gate_noise_margin():  # шум не вызывает откат
    rng = np.random.default_rng(0)  # генератор
    trips = 0  # счётчик
    for _ in range(300):  # прогоны
        b = _Book(); b.domains = {"d_a": {"gate_anchor": 1.0}, "d_b": {"gate_anchor": 1.0}}  # опоры
        v = np.concatenate([np.full(64, 0.5), rng.normal(1.0, 1.6, 64), rng.normal(1.0, 1.6, 64)])  # истинный дрейф ноль, шум окна 1.6
        trips += b._anchor_exceeds(v)[0]  # учесть
    assert trips == 0, f"гейт по опоре сработал на шуме {trips} раз из 300"  # ни одного ложного
