"""Тест переноса состояния GDN между чанками (state-carry BPTT).

Теория: primes Hagi/Theory/GDNBridge.lean — gdnScan_append (перенос состояния между
чанками даёт тот же результат, что один длинный прогон), carry_without_wrap_leaks (если
поток ушёл на новый старт, а состояние не обнулено, оно несёт контекст прежнего места —
утечка). Проверяется на настоящем train_optimal: состояние, переданное в следующий
чанк для обернувшейся строки, нулевое; для необернувшейся — нет.

Приём: train_optimal вызывается на данных, где период потока известен, а сдвиг
позиций отслеживается по x_seq первого столбца входа каждого чанка."""

import numpy as np  # численные массивы
import torch as t  # тензоры

from hagi.models import optimal_lm as M  # рабочая модель


def test_wrapped_stream_rows_resume_from_zero_state():
    t.manual_seed(0)  # зерно
    m = M.HagiOptimalLM()  # модель
    m.spike_k = None  # без спайкинга
    seq_len, batch = 16, 4  # параметры как в train_optimal ниже
    # Данные: байт в позиции i равен i mod 251 — по первому байту строки чанка восстанавливается её старт.
    data = bytes((i % 251 for i in range(2400)))  # 2400 байт (потоки обернутся)
    starts, states, prev_pos = [], [], None  # наблюдения: старты чанков, пришедшие состояния, прошлые позиции

    orig = m.forward_seq  # прежний прямой проход

    def spy(x_seq, state=None, capture=None):  # подслушивающий проход
        if state is not None and state.get("S1") is not None and state["S1"].shape[0] == batch:  # обучающий чанк с переносом
            starts.append(x_seq[:, 0].detach().cpu().numpy().copy())  # первые байты строк = старты
            states.append(state["S1"].detach().abs().sum(dim=(1, 2, 3)).cpu().numpy().copy())  # |S1| по строкам
        return orig(x_seq, state=state, capture=capture)  # настоящий проход

    m.forward_seq = spy  # подставить
    try:  # с возвратом
        m.train_optimal(data, epochs=4, batch=batch, seq_len=seq_len)  # несколько эпох: потоки обернутся
    finally:  # всегда
        m.forward_seq = orig  # вернуть
    assert len(starts) >= 3, "перенос состояния не наблюдался"  # перенос вообще был
    # Старт i-го чанка: pos_i = pos_{i-1} + seq_len; перенос на новый старт — если старт не равен ожидаемому
    wrapped_rows = {int(b) for b in range(batch)}  #строки (заполняется ниже)
    checked = 0  # число проверенных переносов
    for i in range(1, len(starts)):  # по чанкам (состояние в чанке i пришло из i−1)
        pos_prev = [int(v) for v in starts[i - 1]]  # старты строк в предыдущем чанке
        pos_cur = [int(v) for v in starts[i]]  # старты текущего
        for b in range(batch):  # по строкам
            expect = (pos_prev[b] + seq_len) % 251  # ожидаемый старт без переноса
            if pos_cur[b] != expect:  # строка ушла на новый старт (wrap)
                assert states[i][b] == 0.0, f"строка {b}: после переноса состояние не нулевое (|S1|={states[i][b]})"  # утечка
                checked += 1; wrapped_rows.add(b)  # учёт
            else:  # строка продолжилась
                assert states[i][b] > 0.0, f"строка {b}: состояние продолженной строки нулевое"  # перенос работает
    assert checked > 0, "ни одна строка не переносилась — тест ничего не проверил"  # проверка состоялась
