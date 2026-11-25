"""predict_next_byte: окно для нейронной догадки — в том же порядке, что при обучении (старые первыми).

Регрессия: раньше контекст разворачивался ([::-1]), модель получала перевёрнутое окно —
окно шло в обратном порядке — модель получала перевёрнутый контекст."""
import numpy as np  # массивы
import torch as t  # тензоры

from hagi.selfdev.unified import Unified  # единая модель


def test_neural_guess_uses_training_window_order(tmp_path, monkeypatch) -> None:  # сам тест
    monkeypatch.setenv("HAGI_CKPT_DIR", str(tmp_path / "ckpt"))  # изолированный каталог
    monkeypatch.setenv("HAGI_DEVICE", "cpu")  # процессор
    monkeypatch.setenv("HAGI_NO_NEURAL", "1")  # не обучать
    u = Unified.load()  # пустая модель в изолированном каталоге
    m = u.model  # модель
    k = m.k  # контекст
    with t.no_grad():  # фиксированные «weights»: случайный логит-шум по байтам окна
        seen = []  # захваченные окна

        def fake_logits(x):  # подмена: логиты = сумма байтов окна по позициям с разными весами (порядок важен)
            seen.append(x.detach().clone())  # запомнить окно
            w = t.arange(1, x.shape[1] + 1, dtype=t.float32)  # веса позиций: старые — легче, новые — тяжелее
            s = float((x[0].float() * w).sum())  # взвешенная сумма
            lg = t.zeros((1, 256)); lg[0, int(s) % 256] = 10.0  # один «уверенный» байт
            return lg  # логиты
        m.logits = fake_logits  # подменить
        ctx = bytes(range(10, 10 + 3 * k))  # контекст длиннее k
        r = u.predict_next_byte(ctx, allow_neural=True)  # нейронная догадка
    assert r["status"] == "NEURAL_GUESS"  # статус
    assert len(seen) == 1  # один вызов
    assert seen[0].shape == (1, k)  # форма
    assert seen[0][0].tolist() == list(ctx[-k:])  # последние k байтов СТАРЫЕ ПЕРВЫМИ — как при обучении
