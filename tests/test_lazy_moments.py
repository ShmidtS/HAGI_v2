"""Тест ленивых моментов Adam для строк таблиц.

Правило: строка таблицы, которой не было в батче, не обновляет моменты
v_m/v_v и не делает шага (маска шага по активным строкам). Иначе редкий байт после исчезновения из данных
продолжает шагать по инерции убывающего старого момента (замер до
исправления: сдвиг 0.095 при весе 0.51 за 50 шагов отсутствия).

Сторож: если маску шага убрать (шагать всеми строками), test_lazy_moments_no_drift
падает — вес редкой строки меняется без единого градиента."""

import torch as t  # тензоры

from hagi.models.optimal_lm import HybridMuonAdamW  # оптимизатор


def _rare_byte_run(rare_grads: int, idle_steps: int) -> tuple[float, float]:
    """Обучить таблицу 4x2: строки 0,1 частые (градиент каждый шаг), строка 2 —
    редкая (получает градиент rare_grads раз из 30), строка 3 без градиента.
    Затем idle_steps шагов без редкой строки. Вернуть (сдвиг строки 2 за простой,
    вес строки 2 до простоя)."""
    t.manual_seed(0)  # зерно
    table, vec = t.nn.Parameter(t.zeros(4, 2)), t.nn.Parameter(t.zeros(2))  # таблица и вектор
    opt = HybridMuonAdamW([], [table, vec], lr_muon=0.0, lr_adam=0.05, momentum=0.0, wd=1e-4)  # оптимизатор
    opt.names = ["tbl", "v"]  # имена

    def grad(rare: bool) -> t.Tensor:  # градиент шага
        g = t.zeros(4, 2); g[0] = 1.0; g[1] = 1.0  # частые строки
        if rare:  # шаг с редкой строкой
            g[2] = 1.0  # редкая строка
        return g  # вернуть

    for k in range(30):  # 30 шагов: редкая строка видна rare_grads раз
        table.grad, vec.grad = grad(k % (30 // rare_grads) == 0), t.zeros(2)  # задать градиенты
        opt.step()  # шаг
    w_before = table[2].detach().clone()  # вес редкой строки до простоя
    for _ in range(idle_steps):  # простой без редкой строки
        table.grad, vec.grad = grad(False), t.zeros(2)  # градиенты без строки 2
        opt.step()  # шаг
    drift = float((table[2].detach() - w_before).abs().max())  # сдвиг за простой
    return drift, float(w_before.abs().max())  # пара


def test_lazy_moments_no_drift():  # редкая строка не дрейфует без градиента
    drift, weight = _rare_byte_run(rare_grads=3, idle_steps=50)  # прогон
    assert weight > 0.0, "редкая строка должна получить вес до простоя"  # предусловие
    assert drift == 0.0, f"редкая строка сдвинулась на {drift:.5f} без градиента (шаг делается неактивными строками)"  # сторож


def test_active_rows_still_move():  # частые строки обучаются как раньше
    drift, _ = _rare_byte_run(rare_grads=3, idle_steps=0)  # только фаза с градиентом
    assert drift == 0.0  # без простоя сдвига нет
    t.manual_seed(0)  # зерно
    table, vec = t.nn.Parameter(t.zeros(4, 2)), t.nn.Parameter(t.zeros(2))  # таблица и вектор
    opt = HybridMuonAdamW([], [table, vec], lr_muon=0.0, lr_adam=0.05, momentum=0.0, wd=0.0)  # без затухания
    opt.names = ["tbl", "v"]  # имена
    for _ in range(5):  # 5 шагов
        table.grad, vec.grad = t.ones(4, 2), t.zeros(2)  # все строки активны
        opt.step()  # шаг
    assert float(table.abs().max()) > 0.0, "активные строки должны двигаться"  # проверка
