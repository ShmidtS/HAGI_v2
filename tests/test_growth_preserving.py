"""Рост HNetByteLM и HagiOptimalLM сохраняет функцию; предсказание Unified — в порядке обучения.

Проверяются сами свойства (логиты не меняются при grow_width/grow), а не числа."""
import numpy as np  # массивы
import torch as t  # тензоры

from hagi.models.hnet_lm import HNetByteLM, _AB  # HNet и пара адаптера
from hagi.models.optimal_lm import HagiOptimalLM, grow_width as opt_grow_width, grow_depth as opt_grow_depth  # оптимальная модель


def _x(n: int = 64) -> t.Tensor:  # фиксированное окно байтов
    raw = open("README.md", "rb").read()[:n]  # первые n байтов файла документации
    return t.as_tensor(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))[None]  # форма: одна строка длины n


def test_hnet_grow_width_keeps_logits_with_adapter_and_blocks() -> None:  # HNet: ширина
    for nh in (1, 2, 4):  # по числу голов
        m = HNetByteLM(dim=24, hidden=48, k=64, n_heads=nh, seed=1)  # модель
        m.domain_adapter = _AB(t.randn(4, 24) * 0.1, t.randn(24, 4) * 0.1)  # адаптер
        m.grow()  # один блок
        x = _x()  # окно
        with t.no_grad(): lg0 = m.logits(x)  # логиты до
        m2 = m.grow_width(32, 48)  # рост 24 -> 32
        with t.no_grad(): diff = float((m2.logits(x) - lg0).abs().max())  # расхождение
        assert m2.nh == nh and len(m2.blocks) == 1 and m2.domain_adapter is not None  # головы, блок и адаптер перенесены
        assert diff < 1e-4, diff  # функция сохранена


def test_hnet_grow_depth_is_identity_then_trainable() -> None:  # HNet: глубина
    m = HNetByteLM(dim=24, hidden=48, k=64, seed=2)  # модель
    x = _x()  # окно
    with t.no_grad(): lg0 = m.logits(x)  # до
    assert m.grow() == 1  # один блок
    with t.no_grad(): diff = float((m.logits(x) - lg0).abs().max())  # после
    assert diff == 0.0, diff  # точно та же функция (W_d = 0)
    lg = m.logits(x)  # с графом
    lg.sum().backward()  # градиент
    assert m.blocks[0]["Wd"].grad is not None and float(m.blocks[0]["Wd"].grad.abs().sum()) > 0  # первый шаг обучаем (FirstLayerGrad)
    assert any(p is m.blocks[0]["Wd"] for p in m.params())  # блок — в списке параметров (сохраняется)


def test_optimal_grow_width_and_depth_keep_logits() -> None:  # оптимальная модель
    m = HagiOptimalLM(dim=24, hidden=48, n_heads=2, seed=3)  # модель
    x = _x(200)  # окно
    with t.no_grad(): lg0 = m.forward_seq(x)  # до
    m2 = HagiOptimalLM(dim=24, hidden=48, n_heads=2, seed=3)  # копия тем же зерном
    m2.load_state_dict_from(m)  # те же веса
    opt_grow_width(m2, 8)  # рост ширины
    with t.no_grad(): d1 = float((m2.forward_seq(x) - lg0).abs().max())  # расхождение
    m3 = HagiOptimalLM(dim=24, hidden=48, n_heads=2, seed=3); m3.load_state_dict_from(m)  # копия
    opt_grow_depth(m3)  # глубина
    with t.no_grad(): d2 = float((m3.forward_seq(x) - lg0).abs().max())  # расхождение
    assert d1 < 1e-3, d1  # ширина сохраняет функцию
    assert d2 == 0.0, d2  # глубина — точно
