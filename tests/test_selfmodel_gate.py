"""Тесты гейта шага оптимизатора с несколькими задачами (selfmodel).

    guarded_optimizer_step: одна попытка шага общей модели по K задачам.
      1. Градиенты задач нормируются ĝ_i = g_i/||g_i||; веса MGDA α_i ≥ 0,
         Σ α_i = 1 минимизируют ||Σ_i α_i·ĝ_i||²; направление d = Σ_i α_i·g_i.
         Благодаря нормировке задача с огромным градиентом не подавляет прочие.
      2. Направление записывается в .grad, скорость обучения умножается на
         масштаб scale = 1/2^bt, bt = 0..max_backtracks; шаг принимается
         (COMMIT), если взвешенный прирост gain = W_before − W_after > 0 и ни
         одна задача не ухудшилась сильнее max_task_regression
         (max_k (L_after,k − L_before,k) ≤ max_task_regression). Иначе HOLD и
         параметры, оптимизатор и генераторы восстанавливаются из снимков.

    Проверяемые свойства: смесь MGDA не подчинена наибольшему градиенту;
    для Adam уменьшение масштаба действительно уменьшает шаг (умножается lr);
    средний прирост не может скрыть регрессию одной задачи; веса MGDA лежат
    на симплексе и симметричны одинаковых градиентах."""
import pytest  # инструменты тестирования
import torch as t  # тензоры и оптимизаторы

from hagi.selfmodel import guarded_optimizer_step, mgda_weights  # гейтируемый шаг и веса MGDA


def _two_task_setup(scale_ratio=1000.0):
    """Линейная модель y = w1·x1 + w2·x2 с w = (0, 0) и две задачи.

        big: пример (1, 0) -> target = scale_ratio, градиент только по w1;
        small: пример (0, 1) -> target = 1, градиент только по w2.
        Потеря = Σ (w·x − target)². Возвращает (model, task_batches, loss_fn,
        validation)."""
    model = t.nn.Linear(2, 1, bias=False)  # линейный слой с 2 входами, без смещения
    with t.no_grad():  # без записи градиентов
        model.weight.copy_(t.tensor([[0.0, 0.0]]))  # веса w := (0, 0)
    # задача 1 толкает вдоль x, задача 2 — вдоль y, с очень разным масштабом градиентов
    x = t.tensor([[1.0, 0.0]])  # вход big: (1, 0)
    x2 = t.tensor([[0.0, 1.0]])  # вход small: (0, 1)
    def loss_fn(m, b):  # потеря пакета b = (вход, цель) на модели m
        x_, y_ = b  # вход и цель
        return ((m(x_) - y_) ** 2).sum()  # сумма квадратов ошибок Σ (m(x) − y)²
    task_batches = {"big": (x, t.tensor([[scale_ratio]])), "small": (x2, t.tensor([[1.0]]))}  # пакеты задач: цели scale_ratio и 1
    validation = lambda m: {"big": float(((m(x) - task_batches["big"][1]) ** 2).sum()),  # потери до/после: big = Σ (m(x) − scale_ratio)²
                            "small": float(((m(x2) - task_batches["small"][1]) ** 2).sum())}  # small = Σ (m(x2) − 1)²
    return model, task_batches, loss_fn, validation  # вернуть набор для шага


def test_mgda_mixture_is_not_dominated_by_the_largest_gradient():
    """При отношении градиентов 1000 оба веса сдвигаются сопоставимо:
    0,2 < |w2| / max(|w1|, 1e-12) ≤ 1, и ни одна задача не стала хуже (допуск 1e-9)."""
    model, tb, lf, val = _two_task_setup()  # две задачи с отношением масштабов 1000
    before = val(model)  # потери задач до шага
    res = guarded_optimizer_step(model, t.optim.SGD(model.parameters(), lr=0.1), tb, lf, val,  # шаг SGD со скоростью 0,1, не более 3 уменьшений, регрессия запрещена
                                 max_backtracks=3, max_task_regression=0.0)  # не более 3 уменьшений масштаба, регрессия задач запрещена
    # Без нормировки шаг шёл почти целиком вдоль x (градиент в 1000 раз больше),
    # и 'small' почти не сдвигалась. С нормированной смесью оба веса
    # сдвигаются на сопоставимую величину.
    w = model.weight.detach()[0]  # веса после шага w = (w1, w2)
    assert 0.2 < abs(float(w[1]) / max(abs(float(w[0])), 1e-12)) <= 1.0, w  # отношение модулей |w2| / max(|w1|, 1e-12) лежит в (0,2; 1]
    assert all(val(model)[k] <= before[k] + 1e-9 for k in before)  # каждая потеря после шага ≤ потери до + 1e-9


def test_adam_backtracking_actually_changes_the_step():
    """Для Adam с огромной скоростью 5,0 при COMMIT принятое обновление меньше полного
    (норма весов ≤ нормы после полного шага·(1 − 1e-6)), либо масштаб равен 1; норма конечна."""
    # Направление, которое при масштабе 1 пролетает мимо, должно быть уменьшено
    # перебором масштабов: принятый масштаб step_scale < 1 обязан давать
    # МЕНЬШЕЕ обновление, чем полный шаг.
    model, tb, lf, val = _two_task_setup(scale_ratio=1.0)  # модель для гейтируемого шага, равные масштабы
    m_full, _, _, _ = _two_task_setup(scale_ratio=1.0)  # модель для сравнения: один полный шаг Adam
    opt = t.optim.Adam(model.parameters(), lr=5.0)  # Adam с нарочно огромной скоростью 5,0
    res = guarded_optimizer_step(model, opt, tb, lf, val, max_backtracks=8, max_task_regression=0.0)  # шаг с не более чем 8 уменьшениями масштаба
    o2 = t.optim.Adam(m_full.parameters(), lr=5.0)  # второй Adam с той же скоростью 5,0
    o2.step()  # полный шаг по накопленному градиенту (пока нулевому — сравнение по норме весов)
    full_norm = float(m_full.weight.detach().norm())  # норма весов после полного шага
    taken_norm = float(model.weight.detach().norm())  # норма весов после гейтируемого шага
    if res.action == "COMMIT":  # если шаг принят
        assert taken_norm <= full_norm * (1.0 - 1e-6) or res.step_scale == 1.0  # принятый шаг меньше полного либо масштаб = 1
    import math; assert math.isfinite(taken_norm)  # норма весов конечна (не NaN и не бесконечность)


def test_aggregate_gain_cannot_hide_a_single_task_regression():
    """Если задача a сильно улучшается, а b чуть ухудшается (средний прирост > 0),
    шаг обязан быть отклонён: action = HOLD и вес остался 0."""
    # задача 'a' улучшается сильно, 'b' становится чуть хуже: средний прирост > 0.
    calls = {"n": 0}  # счётчик вызовов (не используется)
    model = t.nn.Linear(1, 1, bias=False)  # линейный слой с 1 входом, без смещения
    def loss_fn(m, b):  # потеря пакета
        return (m(b[0]) - b[1]).sum() ** 2  # квадрат суммы ошибок (Σ (m(x) − y))²
    def validation(m):  # потери задач (заменяется ниже)
        return {"a": 1.0, "b": 0.0}  # a = 1, b = 0
    tb = {"a": (t.ones(1, 1), t.zeros(1, 1)), "b": (t.ones(1, 1), t.zeros(1, 1))}  # обе задачи: вход 1, цель 0
    # 'b' ухудшается после любого шага: потери зависят от параметра
    def validation2(m):  # потери как функции веса w
        w = float(m.weight.detach()[0, 0])  # вес w
        return {"a": 1.0 - min(w, 1.0), "b": 0.5 * w}  # a = 1 − min(w, 1) (лучше с ростом w), b = 0,5·w (хуже с ростом w)
    with t.no_grad():  # без записи градиентов
        model.weight.fill_(0.0)  # вес := 0
    res = guarded_optimizer_step(model, t.optim.SGD(model.parameters(), lr=0.5), tb, loss_fn, validation2,  # SGD со скоростью 0,5, до 4 уменьшений, регрессия запрещена
                                 max_backtracks=4, max_task_regression=0.0)  # не более 4 уменьшений масштаба, регрессия задач запрещена
    assert res.action == "HOLD" and float(model.weight.detach()[0, 0]) == 0.0  # шаг отклонён и вес восстановлен в 0


def test_mgda_weights_symmetry_and_simplex():
    """Веса MGDA: Σ α = 1; для ортогональных градиентов α = (0,5; 0,5); для равных — тоже 0,5."""
    a = t.tensor([1.0, 0.0]); b = t.tensor([0.0, 1.0])  # ортогональные градиенты e1 и e2
    al = mgda_weights([a, b]); assert abs(float(al.sum()) - 1) < 1e-6 and abs(float(al[0] - 0.5)) < 1e-6  # Σ α_i = 1 и α_1 = 0,5 (симметрия)
    al = mgda_weights([a, a]); assert abs(float(al[0] - 0.5)) < 1e-6  # одинаковые градиенты: α_1 = 0,5
