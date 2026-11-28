"""Семантическая память предиктора (PROPOSAL 36): пары «скрытое состояние, байт» уже закодированных
позиций; доля голоса соседей калибруется по потоку и нулевая, пока соседи угадывают не чаще модели."""
import math  # логарифм
import os  # окружение
import numpy as np  # массивы
from hagi.models import archiver as A  # архиватор
from hagi.models.optimal_lm import HagiOptimalLM, MirrorHedgeStreamPredictor  # модель и предиктор


def _model():  # маленькая обученная модель
    m = HagiOptimalLM(); m.spike_k = None  # без спайкинга
    m.train_guarded(bytes(range(0, 256, 7)) * 20, epochs=2)  # обучить на законе 0,7,14,…,252 ×20
    return m  # вернуть модель


def _run(m, data):  # цена в битах на байт, доля позиций с голосом, точности соседей и модели
    p = MirrorHedgeStreamPredictor(m); tot = 0.0; act = 0  # предиктор, цена, число позиций с голосом
    for s in data:  # по байтам
        pr = p.predict_probs(); tot += -math.log2(max(float(pr[s]), 1e-12)); act += int(p.sem_last > 0.0); p.update(s)  # цена фактического байта
    return tot / len(data), act / len(data), float(np.mean([a for a, _ in p.sem_hist])), float(np.mean([b for _, b in p.sem_hist]))  # итог


def test_memory_is_on_by_default_and_stream_roundtrip_is_exact():  # умолчание и без потерь
    m = _model(); src = (bytes(range(0, 256, 7)) * 40)[:3000] + open(__file__, "rb").read()[:1500]  # шаблон и текст
    assert MirrorHedgeStreamPredictor(m).sem_k == 8  # память включена по умолчанию
    c = A.compress_stream_v2(m, src); assert A.decompress_stream_v2(m, c) == src  # точный обратный ход с памятью


def test_weight_is_zero_on_random_bytes_and_not_on_template():  # калибровка
    m = _model(); rnd = bytes(np.random.default_rng(0).integers(0, 256, 2500).tolist())  # случайные байты
    txt = open(__file__, "rb").read()[:2500]  # реальный текст (модель на нём ошибается)
    _, act_r, ak_r, am_r = _run(m, rnd); _, act_t, ak_t, am_t = _run(m, txt)  # прогоны
    assert ak_r <= am_r + 0.02 and act_r < 0.15  # на случайных байтах соседи не лучше модели — голос почти не включается
    assert ak_t > am_t and act_t > 0.5  # на тексте соседи угадывают чаще — голос включается


def test_k_zero_disables_memory():  # выключение
    m = _model(); src = (bytes(range(0, 256, 7)) * 40)[:1500]  # шаблон
    old = os.environ.get("HAGI_SEM_K"); os.environ["HAGI_SEM_K"] = "0"  # выключить память
    try:  # вернуть настройку
        b, act, _, _ = _run(m, src)  # прогон
    finally:  # восстановить окружение
        os.environ.pop("HAGI_SEM_K") if old is None else os.environ.__setitem__("HAGI_SEM_K", old)  # прежнее значение
    assert act == 0.0 and MirrorHedgeStreamPredictor(m).sem_k == 8  # голоса нет; после восстановления память снова включена
