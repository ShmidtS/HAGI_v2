"""Спайкинг входа SwiGLU: выключен по умолчанию в первых 80% шагов, почти бесплатен, переживает загрузку."""
import torch as t  # тензоры

from hagi.models import optimal_lm as M  # модель


def test_spiking_is_causal_and_batch_independent():
    """Порог считается по позиции: ответ для префикса не зависит от будущих байтов и от других строк пакета."""
    t.manual_seed(0)  # зерно
    m = M.HagiOptimalLM()  # модель
    m.spike_k = 3.0  # включить спайкинг
    x = t.randint(0, 256, (2, 48))  # два контекста
    full = m.forward_hidden(x)[:, :24]  # первые 24 позиции при полном окне
    part = m.forward_hidden(x[:, :24])  # те же позиции на укороченном окне
    single = m.forward_hidden(x[:1])[:, :24]  # первая строка отдельно от второй
    assert t.allclose(full, part, atol=1e-5) and t.allclose(full[:1], single, atol=1e-5)  # причинность и независимость от пакета


def test_spiking_changes_logits_slightly_and_off_restores_float():
    """Со спайкингом логиты близки к float-логитам (не тождественны); spike_k = None возвращает float точно."""
    t.manual_seed(1)  # зерно
    m = M.HagiOptimalLM()  # модель
    m.spike_k = None  # вещественные значения без округления
    x = t.randint(0, 256, (2, 64))  # контексты
    with t.no_grad():  # без градиентов
        ref = m.logits(x)  # float-логиты
        m.spike_k, m.spike_off = 3.0, True  # включён, но временно снят
        assert t.equal(m.logits(x), ref)  # spike_off даёт ровно float
        m.spike_off = False  # действует
        sp = m.logits(x)  # спайкованные логиты
    assert not t.equal(sp, ref) and float((sp - ref).abs().max()) < 2.0  # отличается немного, а не разваливается


def test_train_reports_spike_fraction_and_off_disables_it():
    """train_optimal включает спайкинг только на последних 20% шагов и сообщает долю нулей; при spike_k = None доля 0."""
    data = bytes(range(256)) * 60  # 15 360 байтов
    for k in (3.0, None):  # включён и выключен
        t.manual_seed(2)  # зерно
        m = M.HagiOptimalLM()  # модель
        m.spike_k = k  # режим
        r = m.train_optimal(data, epochs=2, batch=8, seq_len=64, seed=0, max_steps_per_epoch=20, log_every=0)  # короткое обучение
        assert (r["spike_zero_frac"] > 0.0) == (k is not None) and r["spike_k"] == k  # нули есть только со спайкингом
        assert r["val_ce_after"] < r["val_ce_before"]  # обучение всё же сходится
        assert m.spike_off is False  # после обучения режим действует
