"""Тесты условного доменного адаптера: парный гейт выигрыша и подгонка адаптера.

    Адаптер низкого ранга: f' = f + B·(A·f), A имеет форму (r, d), B — (d, r),
    2·r·d параметров; при B = 0 он в точности тождественный.

    Парный гейт paired_gain(before, after): по примерам i выигрыш
    d_i = before_i − after_i (> 0 — лучше); возвращает среднее mean(d) и нижнюю
    границу mean(d) − z·SE, где SE = std(d, ddof=1)/√n, z = 2,4. Адаптер
    принят (GROW), только если нижняя граница > ADAPTER_MIN_GAIN = 0,005.

    Свойства: на чистом шуме адаптер отклоняется (HOLD, B = 0); на структуре
    принят (GROW), тело модели не меняется, вектор параметров адаптера
    проходит круг set_params(params()) без потерь."""
import numpy as np  # генераторы случайных чисел, allclose
import pytest  # проверка исключений
import torch  # точное сравнение тензоров тела модели

from hagi.selfdev import unified as U  # единая модель с банком адаптеров
from hagi.base.trunk_ops import LowRankAdapter, paired_gain  # адаптер низкого ранга и парный гейт


def test_paired_gain_positive_only_when_clearly_better():
    """Нижняя граница парного выигрыша > 0 только при явном улучшении (после = до − 0,05);
    при явном ухудшении она < 0; при сдвиге 0,001 (шум) средний выигрыш < 0,005; NaN во входе даёт ValueError."""
    rng = np.random.default_rng(0)  # генератор с зерном 0
    before = rng.normal(2.0, 0.1, 256)  # потери до: 256 значений N(2,0; 0,1)
    _, lower = paired_gain(before, before - 0.05)  # потери после = до − 0,05
    assert lower > 0  # явное улучшение: нижняя граница mean(d) − 2,4·SE положительна
    _, lower = paired_gain(before, before + 0.05)  # потери после = до + 0,05
    assert lower < 0  # явное ухудшение: нижняя граница отрицательна
    gain, lower = paired_gain(before, before + 0.001)  # потери после = до + 0,001
    assert abs(gain) < 0.005 and lower < 0  # шум: |mean(d)| < 0,005 и нижняя граница < 0
    with pytest.raises(ValueError):  # ожидается ошибка значения
        paired_gain([1.0], [float("nan"), 2.0])  # потери содержат NaN и длины не совпадают — отказ


def _unified(tmp_path, monkeypatch):
    """Единая модель во временном каталоге с доменом d; байты домена — 0..255, повторённые 8 раз."""
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    monkeypatch.setenv("HAGI_ARCH", "rope")  # эти тесты — о механизмах Rope-модели (адаптеры A/B, рост блоков W1/Wo); оптимальная модель по умолчанию проверяется отдельно
    monkeypatch.setattr(U.Unified, "ARCH", "rope")  # атрибут класса читается при создании Unified
    u = U.Unified()  # новая единая модель
    u.domains["d"] = {}  # пустая запись домена d
    u._domain_bytes = lambda s, e: bytes(range(256)) * 8  # байты домена: 0..255, повторённые 8 раз
    return u  # вернуть модель


def test_adapter_rejects_noise_holds_identity(tmp_path, monkeypatch):
    """На случайных байтах адаптер не даёт явного выигрыша: статус HOLD и B = 0 (адаптер тождественный)."""
    u = _unified(tmp_path, monkeypatch)  # модель с доменом d
    u._domain_bytes = lambda s, e: np.random.default_rng(1).integers(  # байты домена := 40 000 случайных байтов 0..255
        0, 256, 40_000, dtype=np.uint8).tobytes()  # (продолжение аргументов integers)
    r = u._fit_domain_adapter("d")  # подогнать адаптер домена d
    assert r["status"] == "HOLD" and not u.adapter_bank.get("d").B.any()  # отклонено; в B нет ни одного ненулевого элемента


def test_adapter_grows_on_structure(tmp_path, monkeypatch):
    """На законе «байты 0..255 по кругу» адаптер принят (GROW, ранг 1/2/4), у него 2·r·d параметров,
    set_params(params()) возвращает те же веса, тело модели не изменилось."""
    u = _unified(tmp_path, monkeypatch)  # модель с доменом d
    # При уменьшенном шаге 1e-3 нужен тот же бюджет смещения, что давали 3 эпохи
    # при шаге 3e-3 (измерено: при lr=1e-3 3 эпохи дают HOLD, 6 эпох — GROW).
    # Проверяется структура, а не число эпох.
    u.model.train_guarded(bytes(range(256)) * 8, epochs=9)  # обучить тело на законе домена, 9 эпох
    trunk = [q.detach().clone() for q in u.model.params()]  # снимок параметров тела
    r = u._fit_domain_adapter("d")  # подогнать адаптер домена d
    assert r["status"] == "GROW" and r["rank"] in (1, 2, 4)  # принят с рангом 1, 2 или 4
    ad = u.adapter_bank.get("d")  # адаптер домена d из банка
    assert ad.n_params == 2 * ad.r * ad.d  # число параметров = 2·r·d
    flat = ad.params().copy()  # вектор параметров: A развёрнут, затем B развёрнут
    ad.set_params(flat)  # записать вектор обратно: A = flat[:r·d] формы (r, d), B = flat[r·d:] формы (d, r)
    assert np.allclose(ad.params(), flat)  # после записи обратно вектор параметров тот же
    # тело модели не тронуто подгонкой адаптера
    for q, q0 in zip(u.model.params(), trunk):  # по парам (параметр тела, снимок)
        assert torch.equal(q.detach(), q0)  # параметр тела совпадает со снимком точно
