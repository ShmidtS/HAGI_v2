"""Потоковый архиватор v2 (MirrorHedgeStreamPredictor + PPM 0..4 + LZP): круглый цикл на процессоре
и на видеокарте, причём архив не зависит от устройства модели (кодер и декодер обязаны считать одно и то же)."""
import numpy as np  # массивы
import pytest  # пропуски и проверки
import torch as t  # устройство модели

from hagi.models import archiver as A  # архиватор


def _tiny():  # маленькая модель ради скорости
    from hagi.models.optimal_lm import HagiOptimalLM  # рабочая модель
    m = HagiOptimalLM(); m.spike_k = None  # без спайкинга
    m.train_guarded(bytes(range(0, 256, 3)) * 6, epochs=1)  # одна эпоха на законе 0,3,6,…,255 ×6
    return m  # вернуть модель


def test_stream_v2_roundtrip_on_cpu_and_gpu_gives_same_archive():  # круглый цикл и совпадение устройств
    m = _tiny(); src = open(__file__, "rb").read()[:1500]  # модель и реальные байты
    c_cpu = A.compress_stream_v2(m, src); assert c_cpu[:5] == A.MAGIC6  # архив на процессоре
    assert A.decompress_stream_v2(m, c_cpu) == src  # точный обратный ход
    if not t.cuda.is_available():  # видеокарты нет
        pytest.skip("нет видеокарты")  # пропустить часть про устройство
    g = _tiny(); g.load_state_dict_from(m); g.device = "cuda"; g.sync_device()  # та же модель на видеокарте
    c_gpu = A.compress_stream_v2(g, src)  # архив на видеокарте (до исправления падал перевод тензора в массив)
    assert c_gpu == c_cpu  # архив побайтно тот же
    assert A.decompress_stream_v2(g, c_gpu) == src  # обратный ход на видеокарте
