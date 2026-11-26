"""Побайтовое состояние StreamState воспроизводит forward_seq: та же модель, что при обучении."""
import torch as t  # тензоры

from hagi.models import optimal_lm as M  # модель


def _diff(m, n=96):  # модель, длина потока; максимальная разница логитов в конце
    data = t.randint(0, 256, (n,))  # случайные байты
    st = M.StreamState(m)  # потоковое состояние
    outs = [st.step(int(b)).clone() for b in data]  # логиты после каждого байта
    k = m.k  # длина окна
    with t.no_grad():  # без графа
        ref = m.forward_seq(data[None])[0]  # логиты forward_seq по всему потоку
    return float((t.stack(outs) - ref).abs().max())  # на последних позициях поток и окно должны совпасть


def test_stream_state_matches_forward_seq_without_bypass():
    """Без точного обхода потоковый шаг и forward_seq совпадают (состояние с нуля в обоих случаях)."""
    t.manual_seed(0)  # зерно
    m = M.HagiOptimalLM()  # модель
    m.bypass_boost = 0.0  # обход выключен (у forward_seq окно = весь поток, истории совпадают)
    d0 = _diff(m)  # без спайкинга
    m.spike_k = 3.0  # со спайкингом
    d1 = _diff(m)  # поток и окно
    assert d0 < 1e-4 and d1 < 1e-4, (d0, d1)  # одна и та же функция


def test_stream_state_applies_bypass_like_forward_seq():
    """С включённым обходом потоковый шаг прибавляет те же ответы словарей, что и forward_seq."""
    t.manual_seed(1)  # зерно
    m = M.HagiOptimalLM()  # модель
    m.spike_k = None  # без спайкинга
    m.bypass_boost = 8.0  # обход включён
    seq = bytes([7, 8, 9, 10, 11, 42] * 6)  # поток, где за 7,8,9,10,11 всегда идёт 42
    for _ in range(3):  # накрутить счётчик
        m.bypass_fit(seq)  # словари
    data = t.as_tensor(list(seq))  # тензор байтов
    st = M.StreamState(m)  # состояние
    outs = [st.step(int(b)).clone() for b in data]  # потоковые логиты
    with t.no_grad():  # без графа
        ref = m.forward_seq(data[None])[0]  # логиты окна
    assert t.allclose(t.stack(outs), ref, atol=1e-4)  # совпадают
    pred = ref.argmax(-1)  # предсказание argmax на каждой позиции
    assert all(int(pred[k]) == int(data[k + 1]) for k in range(5, len(data) - 1))  # словарь сработал: позиции с полным контекстом предсказаны верно
