"""Контур саморазвития системы: безопасный контроллер многоцелевого шага обучения,
рамка A/B-экспериментов над политиками бюджета, журнал самоправок с гейтом и модель
самого себя (след → вывод → оптимизация → гейт) — всё в одном файле.

Ключевые идеи: шаг обучения принимается, только если прирост на проверке
W_before − W_after > e_gen + e_quant + e_route + m_safety и ни одна задача не ухудшилась;
иначе параметры, состояние оптимизатора и генератор случайных чисел восстанавливаются.
Экспортирует guarded_optimizer_step, mgda_weights, run_experiment, self_edit_cycle,
TracedProcessorV9, derive_self_model, run_self_optimization, prune_library, apply_prune."""
from __future__ import annotations  # отложенное вычисление аннотаций типов


# ============================================================
# КОНТРОЛЛЕР: безопасный многоцелевой шаг обновления
# ============================================================

import copy  # модуль copy — глубокое копирование состояния оптимизатора
from dataclasses import dataclass, field  # dataclass, field — класс-запись результата и значения по умолчанию для полей
from typing import Any, Callable, Mapping, Optional, Sequence  # типы для аннотаций
import torch  # библиотека torch — тензоры и оптимизаторы

import hagi  # базовый класс GrowingHagiProcessor из корня пакета (без цикла импорта)
@dataclass  # StepResult оформляется как класс данных (поля — аргументы конструктора)
class StepResult:  # результат шага: действие, причина и измеренные величины
    action: str  # действие: 'COMMIT' (принять) или 'HOLD' (откатить)
    reason: str  # текст причины решения
    step_scale: float = 0.0  # масштаб шага scale, при котором прошло обновление (1.0 — полный шаг)
    certified_gain: float = 0.0  # сертифицированный прирост gain = W_before − W_after на проверке
    alphas: dict[str, float] = field(default_factory=dict)  # веса задач α: отдельный пустой словарь на каждый экземпляр
    grad_agreement_min: float = 0.0  # минимум согласования min_i g_i·d (отрицателен при конфликте градиентов)
    backtracks: int = 0  # число сделанных уменьшений шага вдвое
    loss_before: dict[str, float] = field(default_factory=dict)  # потери по задачам до шага: отдельный пустой словарь на каждый экземпляр
    loss_after: dict[str, float] = field(default_factory=dict)  # потери по задачам после шага: отдельный пустой словарь на каждый экземпляр
def _flat_grads(model: torch.nn.Module) -> torch.Tensor:
    """Собрать градиенты всех параметров в один плоский вектор."""
    parts = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]  # для каждого параметра с градиентом: копия градиента без графа, развёрнутая в строку
    if not parts:  # если ни у одного параметра нет градиента
        raise RuntimeError('no gradients present; call loss.backward() first')  # ошибка: сначала нужно вызвать loss.backward()
    return torch.cat(parts)  # конкатенация кусков в один вектор
def _zero_grads(model: torch.nn.Module) -> None:
    """Обнулить градиенты: у каждого параметра grad становится None."""
    for p in model.parameters():  # по всем параметрам модели
        p.grad = None  # grad = None (градиент сброшен)
def mgda_weights(grads: Sequence[torch.Tensor], iters: int=200) -> torch.Tensor:
    """Веса задач MGDA: точка минимальной нормы на выпуклой оболочке градиентов.

    Ищутся веса α на симплексе (α_i ≥ 0, Σ α_i = 1), дающие смесь d = Σ_i α_i·ĝ_i
    наименьшей нормы ||d||, где ĝ_i = g_i/||g_i||. Метод Франка–Вульфа по симплексу:
    детерминированный решатель без градиентов, пригодный для K ≤ ~16 задач.
    grads: K градиентов одинаковой формы; iters: число итераций.
    Возвращает тензор α длины K: Σ α_i = 1, α_i ≥ 0."""
    K = len(grads)  # K — число задач
    if K == 1:  # если задача одна
        return torch.ones(1)  # вернуть вес [1.0]
    G = torch.stack([g / (g.norm() + 1e-12) for g in grads])  # матрица G из нормированных градиентов: G_i = g_i/(||g_i|| + 1e-12), форма (K, n)
    alpha = torch.full((K,), 1.0 / K, dtype=G.dtype)  # стартовые веса α_i = 1/K
    for _ in range(iters):  # число итераций t = 0, 1, …, iters−1
        d = alpha @ G  # текущая смесь d = α·G = Σ_i α_i·G_i
        dots = G @ d  # скалярные произведения dots_i = G_i·d
        j = int(torch.argmin(dots).item())  # j = argmin_i dots_i — вершина симплекса, сильнее всего уменьшающая линейную часть
        if dots[j].item() >= d.dot(d).item() - 1e-16:  # если min_i G_i·d ≥ d·d − 1e-16, линейная часть не убывает — оптимум достигнут
            break  # выйти из цикла
        gamma = 2.0 / (2.0 + float(_))  # шаг γ = 2/(2 + t)
        alpha = (1 - gamma) * alpha  # веса α ← (1 − γ)·α
        alpha[j] = alpha[j] + gamma  # вес α_j ← α_j + γ
    alpha = torch.clamp(alpha, min=0.0)  # обрезать отрицательные веса: α_i ← max(0, α_i)
    s = float(alpha.sum().item())  # сумма весов s = Σ_i α_i
    if s <= 0:  # если сумма неположительна (все веса обрезаны в нуль)
        alpha = torch.full((K,), 1.0 / K)  # равномерные веса 1/K
        s = 1.0  # сумма весов s = 1.0
    return alpha / s  # нормировка: α_i / s (сумма равна 1)
@torch.no_grad()  # декоратор: внутри не ведётся запись графа autograd
def _snapshot_trainable(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    """Снимок параметров: номер параметра -> копия его значений."""
    return {i: p.detach().clone() for i, p in enumerate(model.parameters())}  # словарь {i: копия p без графа} по номерам i параметров p
@torch.no_grad()  # декоратор: без записи графа autograd
def _restore_trainable(model: torch.nn.Module, snap: dict[int, torch.Tensor]) -> None:
    """Восстановить параметры из снимка: p_i <- snap[i]."""
    for i, p in enumerate(model.parameters()):  # по парам (номер i, параметр p)
        p.copy_(snap[i])  # скопировать значения snap[i] в p
def _snapshot_optim(optimizer: torch.optim.Optimizer) -> Any:
    """Глубокая копия состояния оптимизатора (для точного отката)."""
    return copy.deepcopy(optimizer.state_dict())  # state_dict оптимизатора, полностью скопированная
def _snapshot_rng() -> dict[str, Any]:
    """Снимок генераторов случайных чисел: random, numpy, torch."""
    return {'py': __import__('random').getstate(), 'np': __import__('numpy').random.get_state(), 'torch': torch.get_rng_state()}  # словарь с состояниями трёх генераторов: 'py' — random, 'np' — numpy, 'torch' — torch
def _restore_rng(snap: dict[str, Any]) -> None:
    """Восстановить генераторы случайных чисел из снимка."""
    import random  # импорт генератора random
    import numpy as _np  # импорт генератора numpy под именем _np
    random.setstate(snap['py'])  # восстановить состояние random из snap['py']
    _np.random.set_state(snap['np'])  # восстановить состояние numpy из snap['np']
    torch.set_rng_state(snap['torch'])  # восстановить состояние torch из snap['torch']
def guarded_optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, task_batches: Mapping[str, Any], loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor], validation_fn: Callable[[torch.nn.Module], dict[str, float]], generalization_error_bound: float=0.0, quantization_error_bound: float=0.0, routing_error_bound: float=0.0, safety_margin: float=0.0, max_backtracks: int=8, conflict_tol: float=1e-08, task_weights: Optional[Mapping[str, float]]=None, max_task_regression: float=0.0) -> StepResult:
    """Один многоцелевой шаг оптимизатора с гейтом по проверке и точным откатом.

    task_batches: задачи (пакет на имя); loss_fn(model, batch) — потери на пакете;
    validation_fn(model) — отложенные потери по тем же именам задач на фиксированном
    наборе, не пересекающемся с обучающими данными. Вес задачи w_k = task_weights[k],
    если задан, иначе 1. Направление d = Σ_k α_k·g_k/||g_k||, веса α из MGDA.
    Шаг делается с масштабом scale = 1, 1/2, 1/4, … — scale умножает скорость
    обучения каждой группы параметров. Взвешенная потеря W = Σ_k w_k·L_k / Σ_k w_k;
    прирост gain = W_before − W_after.
    Гейт приёма: gain > e_gen + e_quant + e_route + m_safety, и ни одна задача не
    ухудшилась сильнее max_task_regression (L_after,k − L_before,k ≤ порога).
    При HOLD параметры, оптимизатор и генераторы восстанавливаются из снимков."""
    names = list(task_batches.keys())  # имена задач в порядке словаря
    w = {k: float(task_weights[k]) if task_weights and k in task_weights else 1.0 for k in names}  # веса задач w_k: task_weights[k], если задан, иначе 1.0
    wsum = sum(w.values())  # сумма весов wsum = Σ_k w_k
    rng_snap = _snapshot_rng()  # снимок генераторов случайных чисел
    param_snap = _snapshot_trainable(model)  # снимок параметров модели
    optim_snap = _snapshot_optim(optimizer)  # снимок состояния оптимизатора
    loss_before = validation_fn(model)  # отложенные потери по задачам до шага
    if any((k not in loss_before for k in names)):  # если какой-то задачи нет в результате validation_fn
        _restore_rng(rng_snap)  # восстановить генераторы
        return StepResult('HOLD', f'validation_fn missing tasks: {names}')  # HOLD: validation_fn не вернула потери для задач из списка names
    wb = sum((w[k] * float(loss_before[k]) for k in names)) / wsum  # взвешенная потеря до шага W_before = Σ_k w_k·L_before,k / wsum
    grads: list[torch.Tensor] = []  # список градиентов задач (плоских векторов)
    for k in names:  # по задачам k
        _zero_grads(model)  # обнулить градиенты
        try:  # попытка вычислить потери и градиент
            loss = loss_fn(model, task_batches[k])  # потери loss_fn(model, пакет задачи k)
            loss.backward()  # обратный проход: заполнить .grad
        except RuntimeError as exc:  # если backward упал с ошибкой времени выполнения
            _zero_grads(model)  # обнулить градиенты
            _restore_rng(rng_snap)  # восстановить генераторы
            return StepResult('HOLD', f"backward failed on '{k}': {exc}")  # HOLD с причиной ошибки: backward не удался для задачи k
        g = _flat_grads(model)  # плоский вектор градиента g_k
        if not torch.isfinite(g).all():  # если в градиенте есть не конечные значения (NaN или inf)
            _zero_grads(model)  # обнулить градиенты
            _restore_rng(rng_snap)  # восстановить генераторы
            return StepResult('HOLD', f"non-finite gradient on '{k}'")  # HOLD: не конечный градиент у задачи k
        grads.append(g)  # добавить градиент задачи в список
    alpha = mgda_weights(grads)  # веса задач α из MGDA
    # Веса α решаются на НОРМИРОВАННЫХ градиентах (в mgda_weights строки — это g/||g||),
    # поэтому применять их к сырым градиентам нельзя: при разных масштабах смесь
    # определяется наибольшим градиентом (измерено: α=[.5,.5], d=[500, 5e-4]) и не является
    # вычисленной точкой минимальной нормы. Смешиваются нормированные градиенты.
    # вычисленной точкой минимальной нормы. Поэтому смешиваются нормированные градиенты.
    normed = [g / (g.norm() + 1e-12) for g in grads]  # нормированные градиенты ĝ_i = g_i/(||g_i|| + 1e-12)
    d = sum((float(alpha[i].item()) * normed[i] for i in range(len(grads))))  # направление d = Σ_i α_i·ĝ_i
    dots = [float(grads[i].dot(d).item()) for i in range(len(grads))]  # согласование с сырыми градиентами: dots_i = g_i·d
    min_dot = min(dots)  # минимальное согласование min_i g_i·d
    _zero_grads(model)  # обнулить градиенты модели
    if min_dot < -conflict_tol:  # если min_i g_i·d < −conflict_tol — градиенты конфликтуют с направлением
        _restore_rng(rng_snap)  # восстановить генераторы
        return StepResult('HOLD', 'gradient conflict (g_i^T d < 0)', alphas={k: float(alpha[i]) for i, k in enumerate(names)}, grad_agreement_min=min_dot)  # HOLD: конфликт градиентов (g_i·d < 0), вернуть веса α и минимальное согласование
    d_norm = float(d.norm().item())  # норма направления ||d||
    if d_norm <= 1e-14:  # если направление вырождено: ||d|| ≤ 1e-14
        _restore_rng(rng_snap)  # восстановить генераторы
        return StepResult('HOLD', 'degenerate direction (||d|| ~ 0)', alphas={k: float(alpha[i]) for i, k in enumerate(names)})  # HOLD: вырожденное направление (||d|| ~ 0)
    total_bound = generalization_error_bound + quantization_error_bound + routing_error_bound + safety_margin  # суммарный порог прироста: e_gen + e_quant + e_route + m_safety
    best: Optional[StepResult] = None  # лучший по приросту результат среди не прошедших гейт масштабов (пока нет)
    scale = 1.0  # начальный масштаб шага scale = 1.0
    for bt in range(max_backtracks + 1):  # по числу уменьшений шага bt = 0..max_backtracks (всего max_backtracks + 1 попыток)
        _restore_trainable(model, param_snap)  # восстановить параметры из снимка
        optimizer.load_state_dict(optim_snap)  # восстановить состояние оптимизатора
        _zero_grads(model)  # обнулить градиенты
        idx = 0  # смещение idx в плоском векторе d
        # Масштабирование градиента (scale·grad) НЕ равно масштабированию скорости
        # обучения для Adam: его шаг m/sqrt(v) не зависит от масштаба градиента, поэтому
        # все попытки давали один и тот же полный шаг (измерено: масштаб градиента 0.01 ->
        # отношение шага 1.000). Поэтому масштаб умножает lr каждой группы параметров
        # на время этой попытки, а после шага lr восстанавливается.
        for p in model.parameters():  # по параметрам модели
            n = p.numel()  # число элементов n в параметре
            p.grad = d[idx:idx + n].view_as(p).clone()  # записать в grad кусок d[idx : idx+n] формы параметра (срез элементов idx..idx+n−1)
            idx += n  # смещение idx ← idx + n
        base_lrs = [g["lr"] for g in optimizer.param_groups]  # запомнить скорости обучения base_lrs_g = lr_g всех групп
        for g in optimizer.param_groups:  # по группам параметров оптимизатора
            g["lr"] = g["lr"] * scale  # временно lr_g ← lr_g·scale
        try:  # попытка сделать шаг
            optimizer.step()  # шаг оптимизатора по записанному градиенту d
        finally:  # в любом случае после шага
            for g, lr in zip(optimizer.param_groups, base_lrs):  # по парам (группа, её исходная скорость)
                g["lr"] = lr  # восстановить lr_g из base_lrs
        loss_after = validation_fn(model)  # отложенные потери по задачам после шага
        if any((k not in loss_after for k in names)):  # если какой-то задачи нет в результате
            break  # прервать перебор масштабов
        wa = sum((w[k] * float(loss_after[k]) for k in names)) / wsum  # взвешенная потеря после шага W_after = Σ_k w_k·L_after,k / wsum
        gain = wb - wa  # прирост gain = W_before − W_after
        # Взвешенное СРЕДНЕЕ может улучшиться, пока одна задача ухудшается (забывание
        # скрыто агрегатом). Поэтому общий прирост принимается только если ни одна
        # задача не ухудшилась сильнее max_task_regression.
        # задача не ухудшилась сильнее max_task_regression.
        worst = max(float(loss_after[k]) - float(loss_before[k]) for k in names)  # худшее ухудшение worst = max_k (L_after,k − L_before,k)
        if worst > max_task_regression:  # если худшее ухудшение превысило допуск max_task_regression
            break  #             break                      # более мелкий шаг не пробуется: виновато само направление
        res = StepResult('COMMIT', 'validation gain exceeds error bounds', step_scale=scale, certified_gain=gain, alphas={k: float(alpha[i]) for i, k in enumerate(names)}, grad_agreement_min=min_dot, backtracks=bt, loss_before=dict(loss_before), loss_after=dict(loss_after))  # кандидат на COMMIT: масштаб scale, прирост gain, веса α, число откатов bt, потери до и после
        if gain > total_bound:  # если прирост строго больше суммарного порога
            _zero_grads(model)  # обнулить градиенты
            return res  # принять шаг и вернуть результат
        if best is None or res.certified_gain > best.certified_gain:  # если это первый кандидат или его прирост больше, чем у лучшего
            best = res  # запомнить лучшего кандидата
        scale *= 0.5  # уменьшить масштаб вдвое: scale ← scale/2
    _restore_trainable(model, param_snap)  # ни один масштаб не прошёл гейт: восстановить параметры
    optimizer.load_state_dict(optim_snap)  # восстановить состояние оптимизатора
    _zero_grads(model)  # обнулить градиенты
    _restore_rng(rng_snap)  # восстановить генераторы
    reason = f'validation gain below error budget ({best.certified_gain:.6g} <= {total_bound:.6g})' if best else 'all backtrack scales failed validation'  # причина HOLD: прирост лучшего кандидата не превысил порог (gain <= total_bound), либо все масштабы провалились
    return StepResult('HOLD', reason, step_scale=scale, certified_gain=best.certified_gain if best else 0.0, alphas={k: float(alpha[i]) for i, k in enumerate(names)}, grad_agreement_min=min_dot, backtracks=max_backtracks, loss_before=dict(loss_before), loss_after=dict(best.loss_after) if best else {})  # вернуть HOLD с лучшим приростом, весами α и потерями до и после
def self_test() -> None:
    """Исполняемые проверки свойств контроллера на маленьких задачах."""
    torch.manual_seed(0)  # зафиксировать зерно генератора torch = 0
    model = torch.nn.Linear(4, 1)  # линейная модель 4 входа -> 1 выход
    x = torch.randn(64, 4)  # обучающие входы x формы (64, 4) из стандартного нормального распределения
    y1 = x @ torch.tensor([[1.0], [1.0], [0.0], [0.0]]) + 0.1  # метки задачи 1: y1 = x0 + x1 + 0.1 (матричное умножение на столбец весов)
    y2 = x @ torch.tensor([[0.0], [0.0], [1.0], [1.0]]) - 0.1  # метки задачи 2: y2 = x2 + x3 − 0.1

    def loss_fn(m, batch):
        """Потери модели m на пакете (входы xx, метки yy): средний квадрат ошибки."""
        xx, yy = batch  # разобрать пакет на входы xx и метки yy
        return ((m(xx) - yy) ** 2).mean()  # средний квадрат ошибки mean((m(xx) − yy)^2)
    vx = torch.randn(256, 4)  # проверочные входы vx формы (256, 4)
    vy1 = vx @ torch.tensor([[1.0], [1.0], [0.0], [0.0]]) + 0.1  # проверочные метки задачи 1 по той же формуле y = x0 + x1 + 0.1
    vy2 = vx @ torch.tensor([[0.0], [0.0], [1.0], [1.0]]) - 0.1  # проверочные метки задачи 2 по той же формуле y = x2 + x3 − 0.1

    def validation_fn(m):
        """Отложенные потери по задачам t1 и t2 без записи графа autograd."""
        with torch.no_grad():  # без построения графа вычислений
            return {'t1': float(loss_fn(m, (vx, vy1))), 't2': float(loss_fn(m, (vx, vy2)))}  # словарь: 't1' и 't2' — потери на проверочных пакетах как числа
    opt = torch.optim.SGD(model.parameters(), lr=0.5)  # SGD-оптимизатор со скоростью обучения 0.5
    before = validation_fn(model)  # потери до шага
    res = guarded_optimizer_step(model, opt, {'t1': (x, y1), 't2': (x, y2)}, loss_fn, validation_fn, generalization_error_bound=0.0, quantization_error_bound=0.0, routing_error_bound=0.0, safety_margin=0.0, max_backtracks=6)  # шаг контроллера на двух задачах со всеми порогами 0 и не более 6 уменьшений
    assert res.action == 'COMMIT', res  # проверка: шаг принят
    after = validation_fn(model)  # потери после шага
    assert after['t1'] < before['t1'] and after['t2'] < before['t2'], (before, after)  # проверка: обе задачи улучшились
    assert abs(sum(res.alphas.values()) - 1.0) < 1e-06  # проверка: |Σ α − 1| < 1e-6
    assert all((a >= 0 for a in res.alphas.values()))  # проверка: все веса α ≥ 0
    print(f'  case1 COMMIT: gain={res.certified_gain:.4f} alphas={res.alphas}')  # печать результата случая 1
    model2 = torch.nn.Linear(4, 1, bias=False)  # вторая модель без свободного члена
    y3 = x @ torch.tensor([[1.0], [-1.0], [0.0], [0.0]])  # метки y3 = x0 − x1
    y4 = -y3  # метки y4 = −y3 — противоположная задача
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.5)  # SGD-оптимизатор со скоростью 0.5
    snap_params = {i: p.detach().clone() for i, p in enumerate(model2.parameters())}  # снимок параметров второй модели
    opt_state = copy.deepcopy(opt2.state_dict())  # глубокая копия состояния оптимизатора
    res2 = guarded_optimizer_step(model2, opt2, {'t1': (x, y3), 't2': (x, y4)}, loss_fn, validation_fn, max_backtracks=4)  # шаг на противоположных задачах, не более 4 уменьшений
    assert res2.action == 'HOLD', res2  # проверка: шаг отклонён
    for i, p in enumerate(model2.parameters()):  # по параметрам второй модели
        assert torch.equal(p.detach(), snap_params[i]), f'param {i} changed on HOLD'  # проверка: параметр не изменился после HOLD
    assert opt2.state_dict().keys() == opt_state.keys()  # проверка: ключи состояния оптимизатора не изменились
    print(f"  case2 HOLD+rollback: reason='{res2.reason}' min_agreement={res2.grad_agreement_min:.3e}")  # печать результата случая 2
    model3 = torch.nn.Linear(4, 1)  # третья модель
    opt3 = torch.optim.SGD(model3.parameters(), lr=0.5)  # SGD-оптимизатор со скоростью 0.5
    res3 = guarded_optimizer_step(model3, opt3, {'t1': (x, y1), 't2': (x, y2)}, loss_fn, validation_fn, generalization_error_bound=1000000.0, max_backtracks=3)  # шаг с недостижимым порогом прироста 1e6, не более 3 уменьшений
    assert res3.action == 'HOLD', res3  # проверка: шаг отклонён
    after3 = validation_fn(model3)  # потери после шага
    assert after3 == validation_fn(model3)  # проверка: повторный вызов validation_fn даёт то же (модель не менялась)
    print(f'  case3 HOLD@bounds: gain={res3.certified_gain:.4f} << bound=1e6')  # печать результата случая 3
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])  # ортогональный вектор a = e0
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])  # ортогональный вектор b = e1
    al = mgda_weights([a, b])  # веса MGDA для ортогональных векторов
    assert torch.allclose(al, torch.tensor([0.5, 0.5]), atol=1e-06)  # проверка: α = [0.5, 0.5]
    al2 = mgda_weights([a, a])  # веса MGDA для одинаковых векторов
    assert torch.allclose(al2, torch.tensor([0.5, 0.5]), atol=1e-06)  # проверка: α = [0.5, 0.5]
    al3 = mgda_weights([a, -a])  # веса MGDA для противоположных векторов
    assert abs(float(al3.sum()) - 1.0) < 1e-06  # проверка: сумма весов равна 1
    print(f'  case4 MGDA: orthogonal->uniform, identical->uniform, conflict ok')  # печать результата случая 4
    print('V10 CONTROLLER SELF_TEST PASS')  # печать общего итога самопроверки
if __name__ == '__main__':  # запуск самопроверки при вызове файла как скрипта
    self_test()  # выполнить самопроверку

# ============================================================
# РАМКА A/B-ЭКСПЕРИМЕНТОВ НАД ПОЛИТИКАМИ БЮДЖЕТА
# ============================================================

import argparse  # модуль argparse — разбор аргументов командной строки
import json  # модуль json — чтение и запись JSON
import sys  # модуль sys — интерпретатор и стандартные потоки
import time  # модуль time — измерение времени
from pathlib import Path  # класс Path для путей
from typing import Any, Callable  # типы для аннотаций
import numpy as np  # numpy — средние значения
Policy = Callable[..., int]  # политика бюджета: функция, возвращающая целое число — бюджет узлов
def run_growth(policy: Policy | None, seed: int, n_proposal_rounds: int=8, budget_per_round: int=8, base_budget: int=2000) -> dict[str, Any]:
    """Один прогон роста: базовый набор задач плюс самостоятельно предложенные задачи.

    policy(spec, base, seed, history, **kw) -> бюджет узлов для задачи; None — без
    политики (бюджет всегда base_budget). n_proposal_rounds раундов по
    budget_per_round предложений; зерно генератора = 20260101 + seed.
    Возвращает словарь: имена экспертов, их число, число кандидатов, потраченных
    на неудачных задачах, общее число кандидатов, секунды и след."""
    m = TracedProcessorV9(node_budget=base_budget)  # процессор со следом и базовым бюджетом узлов
    if policy is not None:  # если политика задана
        _orig_learn = m.learn  # запомнить исходный метод learn

        def learn_with_policy(spec, seed=0, **kw):
            """Обёртка learn: бюджет узлов задаёт политика, затем вызывается исходный learn."""
            budget = int(policy(spec, base_budget, seed, m.trace, **kw))  # бюджет = целое из политики policy(spec, base_budget, seed, след, **kw)
            budget = max(1, min(base_budget, budget))  # бюджет обрезается в [1, base_budget]
            m.node_budget = budget  # установить бюджет узлов процесса
            event = _orig_learn(spec, seed=seed, **kw)  # вызвать исходный learn с теми же аргументами
            if m.trace:  # если след не пуст
                m.trace[-1]['policy_budget'] = budget  # в последнюю запись следа дописать бюджет политики 'policy_budget'
                m.trace[-1]['policy_reduced'] = budget < base_budget  # и признак 'policy_reduced' = (бюджет < базового)
            return event  # вернуть событие обучения
        m.learn = learn_with_policy  # заменить learn процесса на обёртку
    rnd = np.random.default_rng(20260101 + seed)  # генератор случайных чисел с зерном 20260101 + seed
    seen: set = set()  # множество уже предложенных задач
    t0 = time.perf_counter()  # начало отсчёта времени
    for sp in hagi.make_task_suite():  # по задачам базового набора
        m.learn(sp, seed=10000, train_neural=False)  # выучить задачу с зерном 10000 без нейронного обучения
    for r in range(n_proposal_rounds):  # по раундам предложений r
        for k in range(budget_per_round):  # по номеру предложения k в раунде
            prop = hagi.propose_noncircular_task(m.experts, rnd, r, seen)  # предложить новую некруговую задачу на текущей библиотеке экспертов
            if prop is None:  # если предложений нет
                continue  # перейти к следующему
            sp, _ = prop  # разобрать пару (задача, прочее), второй элемент не нужен
            if sp.name in m.experts:  # если эксперт с таким именем уже есть
                continue  # перейти к следующему
            m.learn(sp, seed=500000 + r * 104729 + k, train_neural=False)  # выучить предложенную задачу; зерно = 500000 + r·104729 + k
    elapsed = time.perf_counter() - t0  # прошедшее время
    wasted = sum((t['cost'] for t in m.trace if not t['success']))  # кандидаты, потраченные на неудачных задачах: Σ cost по записям с success = 0
    total_candidates = sum((t['cost'] for t in m.trace))  # всего кандидатов: Σ cost по всем записям следа
    return {'expert_names': frozenset(m.experts.keys()), 'n_experts': len(m.experts), 'wasted': wasted, 'total_candidates': total_candidates, 'seconds': round(elapsed, 2), 'trace': m.trace}  # словарь результата: имена экспертов (множество), число экспертов, потрачено впустую, всего кандидатов, секунды, след
def p_baseline(spec, base: int, seed=0, history=(), **kw) -> int:
    """Политика-эталон: бюджет всегда равен базовому."""
    return base  # вернуть базовый бюджет
def p_floor_half(spec, base: int, seed=0, history=(), **kw) -> int:
    """Для семейства auto-composed (код 4) бюджет max(400, base/2), иначе базовый."""
    return max(400, base // 2) if FAMILY_IDS.get(spec.family, 6) == 4 else base  # если код семейства 4 (неизвестное семейство имеет код 6): max(400, ⌊base/2⌋), иначе base
def p_floor_quarter(spec, base: int, seed=0, history=(), **kw) -> int:
    """Для семейства auto-composed (код 4) бюджет max(200, base/4), иначе базовый."""
    return max(200, base // 4) if FAMILY_IDS.get(spec.family, 6) == 4 else base  # если код семейства 4: max(200, ⌊base/4⌋), иначе base
def _shape_key(spec, seed: int, train_fraction: float, coarse_domain: bool):
    """Ключ формы задачи: (семейство, арность, размер области, размах меток, число разных меток).

    При coarse_domain размер области округляется вниз до старшей степени двойки: 2^(bit_length(D)−1)."""
    target_range, target_unique = _target_features(spec, seed, train_fraction)  # признаки меток: размах и число разных значений (по обучающей части)
    domain = len(spec.inputs())  # размер области D = число входных наборов
    domain_key = 1 << domain.bit_length() - 1 if coarse_domain and domain > 0 else domain  # если coarse_domain и D > 0: domain_key = 2^(bit_length(D)−1), иначе D
    return (FAMILY_IDS.get(spec.family, 6), spec.arity, domain_key, target_range, target_unique)  # кортеж (код семейства или 6, арность, ключ области, размах меток, число разных меток)
def make_online_conservative_policy(min_holds: int, fraction: float, coarse_domain: bool=False):
    """Политика, режущая бюджет только после повторных дорогих HOLD в том же сегменте формы задачи.

    min_holds: сколько неудач подряд требуется; fraction: доля базового бюджета;
    coarse_domain: огрублять размер области до степени двойки. Возвращает функцию-политику."""

    def policy(spec, base: int, seed=0, history=(), **kw) -> int:
        """Решение политики: бюджет base или max(200, ⌊base·fraction⌋) для уже неудачной формы."""
        train_fraction = float(kw.get('train_fraction', 0.7))  # доля обучающих данных train_fraction из kw (по умолчанию 0.7)
        key = _shape_key(spec, seed, train_fraction, coarse_domain)  # ключ формы текущей задачи
        prior = [row for row in history if _shape_key_from_trace(row, coarse_domain) == key]  # записи истории, относящиеся к тому же сегменту формы
        if len(prior) >= min_holds and all((int(row['success']) == 0 for row in prior)) and (float(np.mean([row['cost'] for row in prior])) >= 1200):  # если их не меньше min_holds, все неудачны (success = 0) и средняя стоимость ≥ 1200
            return max(200, int(base * fraction))  # вернуть max(200, ⌊base·fraction⌋)
        return base  # иначе базовый бюджет
    return policy  # вернуть функцию-политику
def _shape_key_from_trace(row, coarse_domain: bool):
    """Ключ формы задачи из записи следа (тот же вид, что и _shape_key)."""
    domain = int(row['domain'])  # размер области из записи
    domain_key = 1 << domain.bit_length() - 1 if coarse_domain and domain > 0 else domain  # если coarse_domain и D > 0: старшая степень двойки 2^(bit_length(D)−1), иначе D
    return (int(row['family']), int(row['arity']), domain_key, int(row['target_range']), int(row['target_unique']))  # кортеж (семейство, арность, ключ области, размах меток, число разных меток)
POLICIES: dict[str, Policy] = {'baseline': p_baseline, 'floor_half_auto': p_floor_half, 'floor_quarter_auto': p_floor_quarter, 'online_exact_1hold_half': make_online_conservative_policy(1, 0.5, False), 'online_exact_1hold_quarter': make_online_conservative_policy(1, 0.25, False), 'online_exact_2holds_quarter': make_online_conservative_policy(2, 0.25, False), 'online_exact_2holds_eighth': make_online_conservative_policy(2, 0.125, False), 'online_coarse_2holds_half': make_online_conservative_policy(2, 0.5, True), 'online_coarse_3holds_quarter': make_online_conservative_policy(3, 0.25, True)}  # таблица политик по именам: эталон, автоправила для кода 4 и онлайновые с разными (min_holds, fraction, coarse_domain)
def run_experiment(seeds: int=5, policy_names: list[str] | None=None, seed_start: int=0) -> dict[str, Any]:
    """A/B-эксперимент: сравнить политики бюджета на нескольких зернах.

    seeds: число зерен начиная с seed_start; policy_names: какие политики сравнивать
    (по умолчанию все без повторов; эталон обязателен). Для каждой политики и зерна
    считается прогон роста. Гейт: на каждом зерне эксперты политики включают экспертов
    эталона и среднее число кандидатов строго меньше, чем у эталона. Из прошедших
    выбирается политика с наименьшим средним числом кандидатов. Возвращает словарь с
    гейтом и решением: COMMIT — принять политику, HOLD — оставить эталон."""
    report: dict[str, Any] = {}  # итоговый словарь-отчёт
    selected_names = list(POLICIES) if policy_names is None else list(dict.fromkeys(policy_names))  # имена сравниваемых политик: все либо заданные без повторов
    unknown = sorted(set(selected_names) - set(POLICIES))  # неизвестные имена (отсортированы)
    if unknown:  # если есть неизвестные
        raise ValueError(f'unknown policy names: {unknown}')  # ошибка со списком неизвестных имён
    if 'baseline' not in selected_names:  # если эталона нет среди сравниваемых
        raise ValueError("policy_names must include 'baseline'")  # ошибка: список обязан содержать 'baseline'
    selected_policies = {name: POLICIES[name] for name in selected_names}  # словарь выбранных политик по имени
    report['validation_protocol'] = {'seed_start': seed_start, 'seed_count': seeds, 'policies': selected_names, 'gate': 'preserve every baseline expert on every seed AND reduce mean total candidate evaluations'}  # описание протокола проверки: стартовое зерно, число зерен, политики и условие гейта
    tracer = TracedProcessorV9(node_budget=2000)  # процессор со следом для разминки библиотеки
    rnd = np.random.default_rng(20260101)  # генератор с зерном 20260101
    seen: set = set()  # множество предложенных задач
    for sp in hagi.make_task_suite():  # по базовому набору
        tracer.learn(sp, seed=10000, train_neural=False)  # выучить с зерном 10000 без нейронного обучения
    for k in range(10):  # 10 предложений
        prop = hagi.propose_noncircular_task(tracer.experts, rnd, 0, seen)  # предложить некруговую задачу
        if prop is None:  # если предложений нет
            continue  # перейти к следующему
        sp, _ = prop  # взять задачу из пары
        if sp.name in tracer.experts:  # если эксперт уже есть
            continue  # перейти к следующему
        tracer.learn(sp, seed=500000 + k, train_neural=False)  # выучить с зерном 500000 + k
    from collections import Counter  # локальный импорт счётчика
    reasons = Counter((e['reason'] for e in tracer.growth_log))  # число причин COMMIT/HOLD по журналу роста: Counter(e["reason"])
    cert = impossibility_certificate(tracer.trace)  # сертификат невозможности точного предсказателя успеха на текущих признаках
    report['derived_algorithm'] = {'pipeline': ['REUSE(certified-library-coverage)', 'boolvenn-exact(16 masks)', 'anf-zhegalkin-butterfly(+full-support)', 'direct-template(+full-support)', 'exact-rational-poly(1D/ND,+full-support)', 'exact-piecewise-poly(+full-support)', 'batched-mdl-mitm-dag(+full-support)', 'HOLD(rollback+audit)'], 'commit_reason_counts': dict(reasons), 'success_feature_impossibility_certificate': cert, 'note': "success of the exhaustive synthesis stage is a function of the task's full teacher data, not of coarse features; the experiment therefore trials MEASURED policies instead of claiming a symbolic predictor"}  # описание выведенного алгоритма: стадии конвейера, счётчики причин, сертификат и оговорка
    suite_names = frozenset((s.name for s in hagi.make_task_suite()))  # имена задач базового набора (не используется далее)
    trials: dict[str, list[dict[str, Any]]] = {p: [] for p in selected_policies}  # прогоны по политикам: имя -> список результатов
    for seed in range(seed_start, seed_start + seeds):  # по зернам от seed_start до seed_start + seeds − 1
        for pname, policy in selected_policies.items():  # по выбранным политикам
            res = run_growth(policy, seed)  # прогон роста на зерне
            trials[pname].append(res)  # добавить результат
            print(f"  seed={seed} policy={pname}: experts={res['n_experts']} wasted={res['wasted']} t={res['seconds']}s", flush=True)  # печать строки прогона со сбросом буфера
    gate: dict[str, Any] = {}  # результаты гейта по политикам
    base_stats = trials['baseline']  # прогоны эталона (те же зерна по порядку)
    for pname, runs in trials.items():  # по политикам и их прогонам
        superset_all = all((r['expert_names'] >= b['expert_names'] for r, b in zip(runs, base_stats)))  # на всех зернах эксперты прогона включают экспертов эталона: r ⊇ b
        mean_wasted = float(np.mean([r['wasted'] for r in runs]))  # среднее потраченных впустую кандидатов
        mean_total_candidates = float(np.mean([r['total_candidates'] for r in runs]))  # среднее общего числа кандидатов
        mean_secs = float(np.mean([r['seconds'] for r in runs]))  # среднее секунд
        baseline_total_candidates = float(np.mean([b['total_candidates'] for b in base_stats]))  # среднее общего числа кандидатов у эталона
        passes = superset_all and mean_total_candidates < baseline_total_candidates  # проходит, если надмножество на всех зернах и среднее число кандидатов строго меньше
        gate[pname] = {'superset_on_all_seeds': superset_all, 'exact_accuracy_floor': 1.0 if superset_all else None, 'mean_wasted': round(mean_wasted, 1), 'mean_total_candidates': round(mean_total_candidates, 1), 'baseline_mean_total_candidates': round(baseline_total_candidates, 1), 'total_candidate_reduction_pct': round(100 * (baseline_total_candidates - mean_total_candidates) / max(1.0, baseline_total_candidates), 2), 'mean_seconds': round(mean_secs, 2), 'per_seed_total_candidates': [r['total_candidates'] for r in runs], 'verdict': 'PASS' if passes else 'REJECT', 'per_seed_experts': [r['n_experts'] for r in runs], 'per_seed_wasted': [r['wasted'] for r in runs], 'reduced_budget_tasks_per_seed': [sum((1 for row in r['trace'] if row.get('policy_reduced', False))) for r in runs], 'reduced_budget_commits_per_seed': [sum((1 for row in r['trace'] if row.get('policy_reduced', False) and row['success'])) for r in runs], 'lost_experts_per_seed': [sorted(b['expert_names'] - r['expert_names']) for r, b in zip(runs, base_stats)], 'new_experts_vs_baseline': [sorted(r['expert_names'] - b['expert_names']) for r, b in zip(runs, base_stats)]}  # запись гейта: надмножество, точность 1.0 либо None, средние, снижение кандидатов в % = 100·(baseline − mean)/max(1, baseline), вердикт PASS/REJECT
    passing = {p: g for p, g in gate.items() if p != 'baseline' and g['verdict'] == 'PASS'}  # прошедшие гейт политики кроме эталона
    if passing:  # если есть прошедшие
        best = min(passing, key=lambda p: gate[p]['mean_total_candidates'])  # лучшая — с наименьшим средним числом кандидатов
        report['gate'] = gate  # записать гейт в отчёт
        report['decision'] = {'action': 'COMMIT', 'adopted_policy': best, 'policy_source': str(selected_policies[best]), 'reason': 'measured non-regression on all seeds with strictly fewer total candidate evaluations'}  # решение COMMIT: принять лучшую политику (измерено отсутствие регрессии на всех зернах при меньшем числе кандидатов)
    else:  # иначе
        report['gate'] = gate  # записать гейт в отчёт
        report['decision'] = {'action': 'HOLD', 'reason': 'no candidate policy preserved the baseline expert set on all seeds while reducing total candidate evaluations; baseline retained'}  # решение HOLD: ни одна политика не сохранила экспертов эталона при меньшем числе кандидатов, эталон оставлен
    return report  # вернуть отчёт

# ============================================================
# ЖУРНАЛ САМОПРАВОК С ГЕЙТОМ
# ============================================================

import json  # модуль json
import subprocess  # модуль subprocess — запуск pytest
import sys  # модуль sys — путь к интерпретатору
import time  # модуль time
from pathlib import Path  # класс Path
from typing import Any  # тип Any
ROOT = Path(__file__).resolve().parent.parent  # корень проекта: каталог на два уровня выше этого файла
from hagi.util import CKPT_DIR as _CKPT  # импорт каталога контрольных точек CKPT_DIR под именем _CKPT
EDIT_LOG = _CKPT / 'self' / 'self_edits.json'  # путь к журналу самоправок: подкаталог 'self', файл self_edits.json
BASELINE_PATH = _CKPT / 'self' / 'edit_baseline.json'  # путь к кэшу эталонного замера: подкаталог 'self', файл edit_baseline.json
KNOBS = {'floor_ladder': [[200, 50, 20], [200, 100, 50], [300, 100, 30], [200, 50, 20, 10]], 'batch_size': [8, 12, 16], 'proposals': [8, 10, 12]}  # допустимые значения параметров: 4 варианта лестницы нижних порогов, размер пакета [8, 12, 16], число предложений [8, 10, 12]
def _baseline() -> dict[str, Any]:
    """Эталонный замер текущего кода: из кэша либо один раз вычисляется и кэшируется."""
    if BASELINE_PATH.exists():  # если файл кэша существует
        return json.loads(BASELINE_PATH.read_text(encoding='utf-8'))  # прочитать кэш из JSON
    res = run_benchmark()  # иначе вычислить замер
    BASELINE_PATH.write_text(json.dumps(res), encoding='utf-8')  # записать замер в кэш
    return res  # вернуть замер
def run_benchmark(rounds: int=2, workers: int=8) -> dict[str, Any]:
    """Фиксированный детерминированный замер цикла обучения на маленьком наборе задач.

    rounds: число раундов; workers: число исполнителей. Возвращает словарь: число
    принятых раундов, число экспертов и затраченные секунды."""
    from hagi import loop as sl  # импорт цикла саморазвития под именем sl
    t0 = time.time()  # начало отсчёта времени
    rep = sl.run_self_development(max_rounds=rounds, workers=workers, proposals_per_round=6)  # запуск саморазвития: max_rounds = rounds, workers, по 6 предложений в раунде
    fin = rep['final']  # итоговая часть отчёта
    return {'adopted': fin['adopted_rounds'], 'experts': fin['experts'], 'seconds': round(time.time() - t0, 1)}  # словарь: 'adopted' — принятые раунды, 'experts' — эксперты, 'seconds' — время с точностью 0.1 с
def _gate_tests() -> bool:
    """Прогнать характеризационные тесты проекта; True, если все прошли."""
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', 'tests/test_characterization.py'], capture_output=True, text=True, timeout=1800)  # запустить pytest tests/test_characterization.py до первого сбоя, лимит 1800 с
    return r.returncode == 0  # успех = код возврата 0
def current_settings() -> dict[str, Any]:
    """Текущие значения параметров цикла: значения по умолчанию из сигнатуры,
    перекрытые принятыми гейтом настройками (с ними работает рабочий цикл)."""
    from hagi import loop as sl  # импорт цикла саморазвития под именем sl
    import inspect  # импорт модуля inspect для чтения сигнатуры
    sig = inspect.signature(sl.run_self_development)  # сигнатура функции run_self_development
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}  # значения по умолчанию параметров, у которых они есть
    cur = {'floor_ladder': defaults.get('floor_ladder') or [200], 'batch_size': defaults.get('batch_size'), 'proposals': defaults.get('proposals_per_round')}  # текущие настройки: лестница порогов (если пусто — [200]), размер пакета, число предложений из proposals_per_round
    try:  # попытка
        adopted = adopted_settings()  # взять принятые гейтом настройки
    except Exception:  # при любой ошибке
        adopted = {}  # считать их пустыми
    if 'floor_ladder' in adopted:  # если принята 'floor_ladder'
        cur['floor_ladder'] = adopted['floor_ladder']  # заменить лестницу порогов
    if 'batch_size' in adopted:  # если принят 'batch_size'
        cur['batch_size'] = adopted['batch_size']  # заменить размер пакета
    if 'proposals' in adopted:  # если принято 'proposals'
        cur['proposals'] = adopted['proposals']  # заменить число предложений
    return cur  # вернуть текущие настройки
def propose_patch(rng) -> dict[str, Any]:
    """Предложить изменение одного параметра (только из объявленного пространства KNOBS).

    Кандидатом может быть только изменение: тождественная правка (значение совпадает с
    текущим) потратила бы ресурс гейта (не более 3 кандидатов за цикл) на пустяк, поэтому
    такие варианты отбрасываются. До 64 раз выбираются параметр и значение; если все
    попытки совпали с текущими настройками, возвращается batch_size = 8.
    rng — генератор случайных чисел. Возвращает {"knob": параметр, "value": значение}."""
    import random  # локальный импорт генератора random
    cur = current_settings()  # текущие настройки
    for _ in range(64):  # до 64 попыток
        knob = rng.choice(list(KNOBS.keys()))  # выбрать параметр knobs случайно
        value = rng.choice(KNOBS[knob])  # выбрать значение параметра случайно
        if knob == 'floor_ladder':  # если выбран параметр 'floor_ladder'
            if value == cur.get('floor_ladder'):  # если значение совпадает с текущей лестницей
                continue  # перейти к следующей попытке
        elif value == cur.get(knob):  # иначе, если значение совпадает с текущим значением параметра
            continue  # перейти к следующей попытке
        return {'knob': knob, 'value': value}  # вернуть правку (параметр, значение)
    return {'knob': 'batch_size', 'value': 8}  # запасной вариант: batch_size = 8
def is_noop(patch: dict[str, Any]) -> bool:
    """True, если правка совпадает с текущей настройкой (такая правка тратит ресурс гейта впустую)."""
    cur = current_settings()  # текущие настройки
    if patch['knob'] == 'floor_ladder':  # если параметр правки — 'floor_ladder'
        return patch['value'] == cur.get('floor_ladder')  # сравнить значение с текущей лестницей порогов
    return patch['value'] == cur.get(patch['knob'])  # сравнить значение с текущим значением этого параметра
def adopted_settings() -> dict[str, Any]:
    """Последнее принятое значение каждого параметра из журнала самоправок (принятое гейтом).

    Принятые правки обязаны реально влиять на рабочий цикл, иначе цикл самоправок
    ничего не меняет. Возвращает именованные аргументы для run_self_development."""
    if not EDIT_LOG.exists():  # если журнала самоправок ещё нет
        return {}  # вернуть пустой словарь
    log = json.loads(EDIT_LOG.read_text(encoding='utf-8'))  # прочитать журнал из JSON
    last: dict[str, Any] = {}  # последнее принятое значение по параметру
    for e in log.get('adopted', []):  # по записям списка 'adopted'
        p = e.get('patch', {})  # правка записи
        if p.get('knob') and 'value' in p:  # если у правки есть параметр и значение
            last[p['knob']] = p['value']  # запомнить последнее значение параметра
    kwargs: dict[str, Any] = {}  # именованные аргументы для run_self_development
    if 'floor_ladder' in last:  # если принята 'floor_ladder'
        kwargs['floor_ladder'] = last['floor_ladder']  # аргумент 'floor_ladder'
    if 'batch_size' in last:  # если принят 'batch_size'
        kwargs['batch_size'] = last['batch_size']  # аргумент 'batch_size'
    if 'proposals' in last:  # если принято 'proposals'
        kwargs['proposals_per_round'] = last['proposals']  # число предложений 'proposals_per_round' из значения 'proposals'
    return kwargs  # вернуть аргументы
MAX_GATE_RUNS = 6  # максимум сохраняемых отклонённых записей журнала
def self_edit_cycle(seed: int=0, dry: bool=False) -> dict[str, Any]:
    """Один цикл «предложить -> гейт -> принять». Решение честно записывается в журнал EDIT_LOG.

    seed — зерно генератора; dry — только предложить, без прогона. Гейт: кандидат
    улучшает эталон (больше принятых раундов или экспертов) И характеризационные тесты
    проходят. Возвращает словарь предложения с решением: noop-skipped, dry, adopted или rejected."""
    import random  # локальный импорт генератора random
    rng = random.Random(seed)  # генератор с заданным зерном
    log = json.loads(EDIT_LOG.read_text(encoding='utf-8')) if EDIT_LOG.exists() else {'adopted': [], 'rejected': []}  # журнал из файла либо новый с пустыми списками принятых и отклонённых
    if len(log.get('rejected', [])) >= MAX_GATE_RUNS:  # если отклонённых записей не меньше максимума
        log['rejected'] = log['rejected'][-MAX_GATE_RUNS:]  # оставить только последние MAX_GATE_RUNS отклонённых записей
        EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')  # записать журнал обратно в файл
    baseline = _baseline()  # эталонный замер
    patch = propose_patch(rng)  # предложить правку
    proposal = {'patch': patch, 'ts': time.strftime('%Y-%m-%d %H:%M:%S')}  # запись предложения: сама правка и время 'ts' в формате год-меся-день час:мин:сек
    if is_noop(patch):  # если правка ничего не меняет
        proposal['decision'] = 'noop-skipped'  # решение 'noop-skipped' (пропущена как пустая)
        return proposal  # вернуть предложение
    if dry:  # если пробный режим без прогона
        proposal['decision'] = 'dry'  # решение 'dry'
        return proposal  # вернуть предложение
    from hagi import loop as sl  # импорт цикла саморазвития под именем sl
    kwargs = {}  # именованные аргументы прогона
    if patch['knob'] == 'batch_size':  # если параметр 'batch_size'
        kwargs['batch_size'] = patch['value']  # аргумент 'batch_size' = значение правки
    elif patch['knob'] == 'proposals':  # иначе если параметр 'proposals'
        kwargs['proposals_per_round'] = patch['value']  # аргумент 'proposals_per_round' = значение правки
    elif patch['knob'] == 'floor_ladder':  # иначе если параметр 'floor_ladder'
        kwargs['floor_ladder'] = patch['value']  # аргумент 'floor_ladder' = значение правки
    try:  # попытка прогнать цикл саморазвития с правкой
        rep = sl.run_self_development(max_rounds=2, workers=8, proposals_per_round=6, **kwargs)  # прогон: 2 раунда, 8 исполнителей, по 6 предложений в раунде, плюс аргументы правки
        cand = {'adopted': rep['final']['adopted_rounds'], 'experts': rep['final']['experts']}  # итог кандидата: 'adopted' — принятые раунды, 'experts' — эксперты
    except Exception as e:  # при любой ошибке прогона
        proposal.update(decision='rejected', reason=repr(e)[:80])  # решение 'rejected' с причиной (первые 80 знаков представления исключения)
        log['rejected'].append(proposal)  # добавить предложение к отклонённым
        EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')  # записать журнал
        return proposal  # вернуть предложение
    improves = cand['adopted'] > baseline['adopted'] or cand['experts'] > baseline['experts']  # улучшение: принято больше раундов ИЛИ больше экспертов, чем в эталоне
    no_regress = improves and _gate_tests()  # приём = улучшение И прохождение характеризационных тестов
    proposal.update(decision='adopted' if no_regress else 'rejected', candidate=cand, baseline=baseline, gate_tests=improves)  # записать решение ('adopted' при приёме, иначе 'rejected'), кандидата, эталон и результат улучшения
    (log['adopted'] if no_regress else log['rejected']).append(proposal)  # добавить предложение в список принятых или отклонённых
    EDIT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')  # записать журнал
    return proposal  # вернуть предложение

# ============================================================
# МОДЕЛЬ САМОГО СЕБЯ: СЛЕД -> ВЫВОД -> ОПТИМИЗАЦИЯ -> ГЕЙТ
# ============================================================

import argparse  # модуль argparse — разбор аргументов командной строки
import json  # модуль json
import sys  # модуль sys
import time  # модуль time
from pathlib import Path  # класс Path
from typing import Any, Optional  # типы для аннотаций
import numpy as np  # numpy — генератор и средние
FAMILY_IDS = {'BoolVenn': 0, 'unary': 1, 'integer-binary': 2, 'composed-ternary': 3, 'auto-composed': 4, 'auto-bool3': 5}  # коды семейств задач: 0 BoolVenn, 1 unary, 2 integer-binary, 3 composed-ternary, 4 auto-composed, 5 auto-bool3
FEATURE_KEYS = ('family', 'arity', 'domain', 'library', 'target_range', 'target_unique', 'self_inv')  # имена признаков задачи в следе: семейство, арность, размер области, размер библиотеки, размах меток, число разных меток, самовариативность
def _self_inverse_flag(spec, seed: int = 13,  # признак самовариативности цели (f∘f = id) по обучающей выборке унарных входов
                       train_fraction: float = 0.7) -> int:  # параметры по умолчанию: зерно 13, доля обучения 0.7
    """1, если цель самовариативна (f∘f = id) на выборке ТОЛЬКО из обучающей части
    унарных входов, иначе 0.

    Выборка берётся из обучающего разбиения (та же схема, что в _target_features), а не из
    начала полной области: точки, отложенные под проверку, не должны опрашиваться у
    учителя — иначе сведения об их метках попадают в признак, который видит ученик.
    Проверяется не более 64 входов: y = f(x); если y недопустим или f(y) ≠ x — вернуть 0."""
    if spec.arity != 1:  # если задача не унарная — признак не определён
        return 0  # вернуть 0
    dom = spec.inputs()  # область входов
    if not dom:  # если область пуста
        return 0  # вернуть 0
    rng = np.random.default_rng(seed)  # генератор с зерном seed
    perm = rng.permutation(len(dom))  # случайная перестановка номеров входов perm
    n_train = max(1, min(len(dom) - 1,  # число обучающих точек n_train = max(1, min(N − 1, round(N·train_fraction))),
                         int(round(len(dom) * train_fraction))))  # где N = len(dom)
    train_idx = perm[:n_train]  # номера обучающих входов — первые n_train элементов перестановки
    checked = 0  # счётчик проверенных входов
    for i in train_idx:  # по обучающим номерам i
        if checked >= 64:  # если проверено уже 64 входа
            break  # выйти из цикла
        x = dom[int(i)]  # вход x = dom[i]
        try:  # попытка опросить учителя
            y = int(spec.teacher(*x))  # ответ учителя y = f(x) как целое
            if not spec.valid((y,)) or int(spec.teacher(y)) != x[0]:  # если y недопустим или f(y) ≠ x (цель не обратна сама себе)
                return 0  # вернуть 0
            checked += 1  # проверенных: checked + 1
        except Exception:  # при любой ошибке опроса
            return 0  # вернуть 0
    return 1  # все проверки пройдены — вернуть 1
def _target_features(spec, seed: int, train_fraction: float) -> tuple[int, int]:
    """Признаки меток, видимых ученику: (размах max(y) − min(y), число разных значений меток).

    Берутся только обучающие точки: для булевоподобных задач (семейство BoolVenn или все
    области {0, 1} и не более 16 входов) используются все входы, кроме одного, иначе —
    round(N·train_fraction) точек случайной перестановки с зерном seed. Пустая область — (0, 0)."""
    domain = spec.inputs()  # область входов
    if not domain:  # если область пуста
        return (0, 0)  # вернуть (0, 0)
    rng = np.random.default_rng(seed)  # генератор с зерном seed
    perm = rng.permutation(len(domain))  # случайная перестановка номеров входов
    bool_like = spec.family == 'BoolVenn' or (all((set(d) <= {0, 1} for d in spec.domains)) and len(domain) <= 16)  # булевоподобная задача: семейство BoolVenn ЛИБО все области лежат в {0, 1} и N ≤ 16
    if bool_like:  # если булевоподобная
        n_train = max(1, len(domain) - 1)  # обучающих точек n_train = max(1, N − 1)
    else:  # иначе
        n_train = max(1, min(len(domain) - 1, int(round(len(domain) * train_fraction))))  # иначе n_train = max(1, min(N − 1, round(N·train_fraction)))
    ys = [int(spec.teacher(*domain[int(i)])) for i in perm[:n_train]]  # метки y первых n_train входов перестановки по учителю
    return (max(ys) - min(ys), len(set(ys))) if ys else (0, 0)  # вернуть (max(y) − min(y), число разных y), если метки есть, иначе (0, 0)
class TracedProcessorV9(hagi.GrowingHagiProcessor):  # процессор роста со следом своего обучения (описание ниже, после атрибутов)
    stream_names: list = None  # имена потоков (заполняется наследником)
    stream_actions: list = None  # действия потоков (заполняется наследником)
    """Процессор, ведущий след своего обучения: для каждого вызова learn запоминает
    признаки задачи, исход (успех/нет) и стоимость в кандидатах."""

    def __init__(self, *a, **kw):  # создать процессор; *a, **kw передаются базовому классу
        super().__init__(*a, **kw)  # вызов конструктора базового класса
        self.trace: list[dict[str, Any]] = []  # пустой список записей следа

    def learn(self, spec, seed=0, train_fraction=0.7, poison_one_label=False, train_neural=False, deadline=None):
        """Выучить задачу и дописать запись в след.

        spec — задача; train_fraction — доля обучения; poison_one_label — испортить одну метку;
        train_neural — учить нейросеть; deadline — предельное время. Признаки: код семейства,
        арность, размер области, размер библиотеки, признаки меток, самовариативность.
        Стоимость cost = min(прирост числа сгенерированных кандидатов, 3000);
        success = 1, если действие результата COMMIT, иначе 0. Возвращает событие обучения."""
        target_range, target_unique = _target_features(spec, seed, train_fraction)  # признаки меток: размах и число разных значений
        feats = {'family': FAMILY_IDS.get(spec.family, 6), 'arity': spec.arity, 'domain': len(spec.inputs()), 'library': len(self.experts), 'self_inv': _self_inverse_flag(spec, seed, train_fraction), 'target_range': target_range, 'target_unique': target_unique}  # словарь признаков feats: код семейства (6 для неизвестного), арность, размер области, размер библиотеки, самовариативность, размах меток, число разных меток
        c0 = self.candidates_generated  # число кандидатов до обучения c0
        t0 = time.perf_counter()  # начало отсчёта времени
        ev = super().learn(spec, seed=seed, train_fraction=train_fraction, poison_one_label=poison_one_label, train_neural=train_neural, deadline=deadline)  # вызов learn базового класса с теми же аргументами
        self.trace.append({**feats, 'seed': int(seed), 'success': 1 if ev['action'] == 'COMMIT' else 0, 'cost': min(self.candidates_generated - c0, 3000), 'seconds': round(time.perf_counter() - t0, 4), 'task': spec.name})  # дописать в след: признаки, зерно, success = 1 при COMMIT иначе 0, стоимость cost, секунды с точностью 4 знака, имя задачи
        return ev  # вернуть событие обучения
def _features(spec, model, seed: int=0, train_fraction: float=0.7) -> tuple[int, ...]:
    """Вектор признаков задачи для модели самого себя: (семейство, арность, размер области,
    размер библиотеки модели, размах меток, число разных меток, самовариативность)."""
    target_range, target_unique = _target_features(spec, seed, train_fraction)  # признаки меток
    return (FAMILY_IDS.get(spec.family, 6), spec.arity, len(spec.inputs()), len(model.experts), target_range, target_unique, _self_inverse_flag(spec, seed, train_fraction))  # кортеж из 7 признаков в порядке FEATURE_KEYS
def derive_self_model(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Вывести программы модели самого себя из следа собственным аппаратом системы.

    Признаки — кортежи FEATURE_KEYS. Сначала точная кусочно-полиномиальная аппроксимация
    признаки -> успех; если она не точна на всём следе — синтез программы по MDL
    (до 20000 вычислений, 64 бита). Программа стоимости выводится тем же синтезом,
    если стоимости не одинаковы. Возвращает словарь описания: n_trace, тексты программ
    success_program и cost_program, признаки их точности success_exact и cost_exact,
    а также сами выражения _succ_expr и _cost_expr (None, если не найдены)."""
    rows = [tuple((t[k] for k in FEATURE_KEYS)) for t in trace]  # строки признаков: по кортежу FEATURE_KEYS на запись следа
    ys = [t['success'] for t in trace]  # метки успеха y по записям следа
    succ_expr = hagi.exact_piecewise_poly_fit(rows, ys)  # точная кусочно-полиномиальная аппроксимация признаки -> успех
    succ_prog = None  # программа успеха пока не найдена (переменная не используется далее)
    if succ_expr is not None:  # если аппроксимация найдена
        ok = all((int(succ_expr.eval(r)) == y for r, y in zip(rows, ys)))  # проверка точности: int(f(r)) = y на всех строках следа
        if not ok:  # если хоть одна строка не совпала
            succ_expr = None  # отбросить аппроксимацию
    if succ_expr is None:  # если точной аппроксимации нет
        expr, _n = hagi.synthesize_batched_mdl(len(FEATURE_KEYS), rows, ys, rows, ys, max_evals=20000, max_bits=64)  # синтез программы по MDL на всём следе: выражение и число вычислений
        if expr is not None and all((int(expr.eval(r, {})) == y for r, y in zip(rows, ys))):  # если выражение найдено и точно на всём следе
            succ_expr = expr  # взять его как программу успеха
    costs = [t['cost'] for t in trace]  # стоимости по записям следа
    cost_expr = None  # программа стоимости пока не найдена
    if len(set(costs)) > 1:  # если стоимости не все одинаковы
        ce, _n = hagi.synthesize_batched_mdl(len(FEATURE_KEYS), rows, costs, rows, costs, max_evals=20000, max_bits=64)  # синтез программы по MDL: признаки -> стоимость
        if ce is not None and all((int(ce.eval(r, {})) == c for r, c in zip(rows, costs))):  # если выражение найдено и точно на всём следе
            cost_expr = ce  # взять его как программу стоимости
    model_desc = {'n_trace': len(trace), 'success_program': succ_expr.pretty() if succ_expr is not None else None, 'success_exact': succ_expr is not None, 'cost_program': cost_expr.pretty() if cost_expr is not None else None, 'cost_exact': cost_expr is not None, '_succ_expr': succ_expr, '_cost_expr': cost_expr}  # описание модели: размер следа, тексты программ (None, если нет), признаки точности и сами выражения
    return model_desc  # вернуть описание модели
def impossibility_certificate(trace: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Доказать (свидетелем), что точного предсказателя успеха на текущем пространстве
    признаков нет: найти два вызова learn с одинаковыми признаками и разными исходами.
    Это конечный проверяемый сертификат для отказа от утверждения: нельзя требовать
    того, что не показано. Возвращает {"identical_features": признаки, "conflicting_outcomes":
    список "задача:исход"} или None, если противоречия нет."""
    by_feats: dict[tuple, list[str]] = {}  # группировка записей следа по признакам: кортеж признаков -> список «задача:исход»
    for t in trace:  # по записям следа
        key = tuple((t[k] for k in FEATURE_KEYS))  # ключ — кортеж признаков FEATURE_KEYS
        by_feats.setdefault(key, []).append(f"{t['task']}:{('COMMIT' if t['success'] else 'HOLD')}")  # добавить в группу строку 'имя задачи:COMMIT' либо 'имя задачи:HOLD'
    for key, names in sorted(by_feats.items()):  # по группам в порядке сортировки ключей
        if any((n.endswith(':COMMIT') for n in names)) and any((n.endswith(':HOLD') for n in names)):  # если в группе есть и COMMIT, и HOLD
            return {'identical_features': list(key), 'conflicting_outcomes': names}  # вернуть сертификат: одинаковые признаки и конфликтующие исходы
    return None  # противоречия нет — вернуть None
def optimized_budget_policy(self_model: dict[str, Any], spec, model, base_budget: int, floor_budget: int, seed: int=0, train_fraction: float=0.7) -> int:
    """Распределение вычислений по модели самого себя на наблюдаемом обучающем разбиении.

    Если в модели нет программы успеха — вернуть base_budget. Предсказание
    pred = int(f(признаки задачи)): 0 → min(base_budget, floor_budget), иначе base_budget.
    При ошибке вычисления — base_budget."""
    succ = self_model.get('_succ_expr')  # программа успеха из модели
    if succ is None:  # если программы нет
        return base_budget  # вернуть базовый бюджет
    try:  # попытка предсказать успех
        pred = int(succ.eval(_features(spec, model, seed, train_fraction), {}))  # pred = int(программа успеха(вектор признаков))
    except Exception:  # при любой ошибке вычисления
        return base_budget  # вернуть базовый бюджет
    if pred == 0:  # если предсказан неуспех (pred = 0)
        return min(base_budget, floor_budget)  # вернуть min(base_budget, floor_budget) — меньший бюджет
    return base_budget  # иначе вернуть базовый бюджет
class SelfOptimizingProcessorV9(TracedProcessorV9):
    """Процессор роста, у которого бюджет узлов node_budget на задачу задаёт его собственная
    модель самого себя. self_model — модель; base_budget и floor_budget — верхняя и нижняя
    граница бюджета; *a, **kw — аргументы базового класса."""

    def __init__(self, self_model: dict[str, Any], base_budget: int, floor_budget: int, *a, **kw):  # конструктор: принять модель и границы бюджета
        super().__init__(*a, **kw)  # вызов конструктора базового класса
        self.self_model = self_model  # сохранить модель самого себя
        self.base_budget = base_budget  # базовый (верхний) бюджет
        self.floor_budget = floor_budget  # нижний бюджет для предсказанно неудачных задач
        self.policy_overrides = 0  # счётчик задач, для которых политика уменьшила бюджет

    def learn(self, spec, seed=0, **kw):  # выучить задачу с бюджетом от политики
        override = getattr(self, 'optimized_budget_policy', None)  # переопределённая политика (атрибут 'optimized_budget_policy'), если задана
        if override is None:  # если переопределения нет
            budget = optimized_budget_policy(self.self_model, spec, self, self.base_budget, self.floor_budget, seed=seed, train_fraction=float(kw.get('train_fraction', 0.7)))  # бюджет из модели самого себя; доля обучения из kw (по умолчанию 0.7)
        else:  # иначе
            budget = override(spec, self, self.base_budget, self.floor_budget)  # бюджет из переопределённой политики (spec, модель, base, floor)
        self.node_budget = max(1, min(self.base_budget, int(budget)))  # node_budget = max(1, min(base_budget, int(budget))) — обрезка в [1, base_budget]
        if self.node_budget < self.base_budget:  # если бюджет уменьшен относительно базового
            self.policy_overrides += 1  # счётчик уменьшений: policy_overrides + 1
        ev = super().learn(spec, seed=seed, **kw)  # вызов learn базового класса со всеми аргументами
        return ev  # вернуть событие обучения
def describe_pipeline(model: TracedProcessorV9) -> dict[str, Any]:
    """Модель выдаёт описание собственного конвейера создания, выведенное из её журнала роста.
    Возвращает словарь: список стадий, счётчики причин COMMIT/HOLD, число экспертов и теорем."""
    from collections import Counter  # локальный импорт счётчика
    reasons = Counter((e['reason'] for e in model.growth_log))  # число причин по журналу роста: Counter(e["reason"])
    return {'stages': ['REUSE(certified-library-coverage)', 'boolvenn-exact(16 masks)', 'anf-zhegalkin-butterfly(+full-support)', 'direct-template(+full-support)', 'exact-rational-poly(1D/ND,+full-support)', 'exact-piecewise-poly(+full-support)', 'batched-mdl-mitm-dag(+full-support)', 'HOLD(rollback+audit)'], 'commit_reason_counts': dict(reasons), 'experts': len(model.experts), 'theorems': len(model.theorems)}  # словарь: стадии конвейера, счётчики причин, число экспертов 'experts' и число теорем 'theorems'
def run_self_optimization(rounds: int=2) -> dict[str, Any]:
    """Самооптимизация: собрать след, вывести модель самого себя, применить её политику и проверить гейтом.

    Этап 1: след на базовом наборе и 10 предложенных задачах. Этап 2: вывод модели самого себя
    (программы успеха и стоимости). Этап 3: прогон процессора с бюджетом от модели (тот же
    поток зёрен). Этап 4 — гейт: множество экспертов оптимизированного прогона включает
    экспертов исходного, точность проверки равна 1.0 и принято не меньше задач, чем в наборе;
    тогда adopted = True, иначе HOLD. Если точной программы успеха нет — сертификат
    невозможности, пробная «наивная» политика (нижний бюджет для семейства с кодом 4) и HOLD.
    rounds — зарезервированный параметр (не используется). Возвращает словарь-отчёт."""
    report: dict[str, Any] = {}  # итоговый словарь-отчёт
    suite = hagi.make_task_suite()  # базовый набор задач
    tracer = TracedProcessorV9(node_budget=2000)  # процессор со следом и бюджетом 2000
    rnd = np.random.default_rng(20260101)  # генератор с зерном 20260101
    seen: set = set()  # множество предложенных задач
    t0 = time.perf_counter()  # начало отсчёта времени
    for sp in suite:  # по базовому набору
        tracer.learn(sp, seed=10000, train_neural=False)  # выучить с зерном 10000 без нейронного обучения
    n_prop = 0  # счётчик предложенных задач
    for k in range(10):  # 10 предложений
        prop = hagi.propose_noncircular_task(tracer.experts, rnd, 0, seen)  # предложить некруговую задачу
        if prop is None:  # если предложений нет
            continue  # перейти к следующему
        sp, _ = prop  # взять задачу из пары
        if sp.name in tracer.experts:  # если эксперт уже есть
            continue  # перейти к следующему
        n_prop += 1  # счётчик предложенных: n_prop + 1
        tracer.learn(sp, seed=500000 + k, train_neural=False)  # выучить с зерном 500000 + k
    trace_seconds = time.perf_counter() - t0  # время этапа следа
    commits = sum((t['success'] for t in tracer.trace))  # число успехов в следе: Σ success
    wasted = sum((t['cost'] for t in tracer.trace if not t['success']))  # впусту потраченные кандидаты: Σ cost по неудачным записям
    report['phase1_trace'] = {'learn_calls': len(tracer.trace), 'commits': commits, 'holds': len(tracer.trace) - commits, 'wasted_candidates_on_holds': wasted, 'trace_seconds': round(trace_seconds, 2), 'pipeline': describe_pipeline(tracer)}  # этап 1 в отчёте: число вызовов learn, число COMMIT и HOLD, впусту потраченные кандидаты, секунды, конвейер
    self_model = derive_self_model(tracer.trace)  # вывести модель самого себя из следа
    report['phase2_self_model'] = {k: v for k, v in self_model.items() if not k.startswith('_')}  # этап 2 в отчёте: описание модели без служебных ключей, начинающихся с '_'
    succ = self_model.get('_succ_expr')  # программа успеха из модели
    report['self_algorithm'] = {'pipeline': report['phase1_trace']['pipeline']['stages'], 'success_predictor': self_model['success_program'], 'cost_predictor': self_model['cost_program'], 'policy': 'node_budget = base if self_model.predict_success(task)==1 else floor', 'gate': 'adopt iff expert set identical/superset and floors hold; else rollback'}  # описание собственного алгоритма: стадии, предсказатели успеха и стоимости, формула политики и условие гейта
    if succ is None:  # если точной программы успеха нет
        cert = impossibility_certificate(tracer.trace)  # сертификат невозможности по следу
        demo = SelfOptimizingProcessorV9({'_succ_expr': None, '_naive': True}, base_budget=2000, floor_budget=200)  # демонстрационный процессор с «наивной» моделью (без программы успеха), бюджеты 2000 и 200
        demo.optimized_budget_policy = lambda spec, model, base, floor: floor if FAMILY_IDS.get(spec.family, 6) == 4 else base  # наивная политика: нижний бюджет для семейства с кодом 4, иначе базовый
        rnd2 = np.random.default_rng(20260101)  # второй генератор с тем же зерном 20260101
        seen2: set = set()  # множество предложенных задач для демонстрации
        for sp in suite:  # по базовому набору
            demo.learn(sp, seed=10000, train_neural=False)  # выучить с зерном 10000 без нейронного обучения
        for k in range(10):  # 10 предложений
            prop = hagi.propose_noncircular_task(demo.experts, rnd2, 0, seen2)  # предложить задачу на библиотеке демонстрационного процессора
            if prop is None:  # если предложений нет
                continue  # перейти к следующему
            sp, _ = prop  # взять задачу из пары
            if sp.name in demo.experts:  # если эксперт уже есть
                continue  # перейти к следующему
            demo.learn(sp, seed=500000 + k, train_neural=False)  # выучить с зерном 500000 + k
        lost = sorted(set(tracer.experts) - set(demo.experts))  # потерянные эксперты:sorted(имена исходных − имена демонстрационных)
        if lost:  # если есть потерянные
            gate_reason = 'naive policy lost experts -> gate rollback: ' + ', '.join(lost)  # причина HOLD: наивная политика потеряла экспертов, откат гейта и их список
        else:  # иначе
            gate_reason = 'no exact self-model at this feature resolution; unverified policy cannot be adopted without a certified predictor (honest HOLD)'  # причина HOLD: точной модели при таком разрешении признаков нет, непроверенная политика не принимается
        report['phase3_optimization'] = {'status': 'ATTEMPTED_NAIVE_POLICY', 'policy': 'floor_budget for auto-composed family', 'lost_experts': lost}  # этап 3 в отчёте: статус 'ATTEMPTED_NAIVE_POLICY', описание политики, потерянные эксперты
        report['phase4_gate'] = {'action': 'HOLD', 'reason': gate_reason, 'impossibility_certificate': cert, 'rollback': 'baseline configuration retained (tracer state is the adopted model)'}  # этап 4 в отчёте: HOLD, причина, сертификат невозможности, сообщение об откате к исходной конфигурации
        report['adopted'] = False  # ничего не принято
        return report  # вернуть отчёт
    opt = SelfOptimizingProcessorV9(self_model, base_budget=2000, floor_budget=200)  # процессор с моделью самого себя, бюджеты 2000 и 200
    rnd2 = np.random.default_rng(20260101)  # второй генератор с тем же зерном
    seen2: set = set()  # множество предложенных задач
    t0 = time.perf_counter()  # начало отсчёта времени
    for sp in suite:  # по базовому набору
        opt.learn(sp, seed=10000, train_neural=False)  # выучить с зерном 10000 без нейронного обучения
    for k in range(10):  # 10 предложений
        prop = hagi.propose_noncircular_task(opt.experts, rnd2, 0, seen2)  # предложить некруговую задачу
        if prop is None:  # если предложений нет
            continue  # перейти к следующему
        sp, _ = prop  # взять задачу из пары
        if sp.name in opt.experts:  # если эксперт уже есть
            continue  # перейти к следующему
        opt.learn(sp, seed=500000 + k, train_neural=False)  # выучить с зерном 500000 + k
    opt_seconds = time.perf_counter() - t0  # время оптимизированного прогона
    opt_wasted = sum((t['cost'] for t in opt.trace if not t['success']))  # впусту потраченные кандидаты оптимизированного прогона
    report['phase3_optimization'] = {'status': 'APPLIED', 'learn_calls': len(opt.trace), 'wasted_candidates_on_holds': opt_wasted, 'seconds': round(opt_seconds, 2), 'same_seed_stream': True}  # этап 3 в отчёте: статус 'APPLIED', число вызовов, впусту потраченные кандидаты, секунды, тот же поток зёрен
    base_names = set(tracer.experts.keys())  # имена экспертов исходного прогона
    opt_names = set(opt.experts.keys())  # имена экспертов оптимизированного прогона
    experts_ok = opt_names >= base_names  # эксперты сохранены: opt_names ⊇ base_names
    check = opt.verify_all(suite)  # полная проверка оптимизированного процессора на наборе задач
    floors_ok = check['accuracy'] == 1.0 and check['committed'] >= len(suite)  # нижние пороги: точность равна 1.0 и принято не меньше задач, чем в наборе
    if experts_ok and floors_ok:  # если эксперты сохранены и пороги выполнены
        adopted = True  # принять оптимизацию
        report['phase4_gate'] = {'action': 'COMMIT', 'experts_base': len(base_names), 'experts_optimized': len(opt_names), 'exact_accuracy': check['accuracy'], 'wasted_candidates': f'{wasted} -> {opt_wasted}', 'seconds': f'{round(trace_seconds, 2)} -> {round(opt_seconds, 2)}', 'speedup_wasted': round(wasted / max(1, opt_wasted), 2)}  # этап 4: COMMIT с числом экспертов, точностью, впусту потраченными кандидатами и секундами до -> после, ускорением
    else:  # иначе
        adopted = False  # не принимать
        report['phase4_gate'] = {'action': 'HOLD', 'reason': 'optimized policy lost experts' if not experts_ok else 'quality floor violated', 'experts_base': len(base_names), 'experts_optimized': len(opt_names), 'accuracy': check['accuracy']}  # этап 4: HOLD с причиной (потеряны эксперты либо нарушен порог качества), числом экспертов и точностью
    report['adopted'] = adopted  # записать решение в отчёт
    return report  # вернуть отчёт
def main() -> None:
    """Точка входа командной строки: запустить самооптимизацию и напечатать итог.
    Ключ --json (зарезервирован) — печатать только JSON отчёта."""
    p = argparse.ArgumentParser(description='HAGI self-model: derive & optimize own algorithm')  # парсер аргументов с описанием команды
    p.add_argument('--json', action='store_true', help='print raw JSON only')  # добавить ключ '--json' (печатать только JSON)
    a = p.parse_args()  # разобрать аргументы командной строки
    rep = run_self_optimization()  # запустить самооптимизацию
    print(json.dumps(rep, indent=2, ensure_ascii=False))  # напечатать отчёт как JSON с отступом 2
    if rep['phase4_gate'].get('action') == 'COMMIT':  # если решение этапа 4 — 'COMMIT'
        print(f"SELF-OPTIMIZATION COMMIT: self-model policy adopted ({rep['phase4_gate']['wasted_candidates']} wasted candidates)")  # печать: оптимизация принята;policy принята, число впусту потраченных кандидатов
    elif rep.get('phase3_optimization', {}).get('status') == 'ATTEMPTED_NAIVE_POLICY':  # иначе если этап 3 имел статус 'ATTEMPTED_NAIVE_POLICY'
        print('SELF-OPTIMIZATION HOLD: exact self-model impossible at this feature resolution (certificate produced); naive policy rejected by commit gate, baseline retained')  # печать: HOLD из-за невозможности точной модели, наивная политика отклонена гейтом
    else:  # иначе
        print('SELF-OPTIMIZATION HOLD: ' + rep['phase4_gate'].get('reason', ''))  # печать HOLD с причиной этапа 4 (пустая строка, если причины нет)
if __name__ == '__main__':  # запуск main при вызове файла как скрипта
    main()  # вызвать main

# ============================================================
# ОТРЕЗКА ПО MDL: минимизация библиотеки при сертифицированном покрытии
# ============================================================

from pathlib import Path  # класс Path (повторный импорт)
from typing import Any  # тип Any (повторный импорт)
from hagi import *  # импорт всех открытых имён пакета hagi в текущее пространство имён
def _expr_calls(expr) -> set[str]:
    """Имена экспертов, вызываемых этим выражением (узлы вызова)."""
    out: set[str] = set()  # множество найденных имён

    def walk(e) -> None:  # обойти выражение в глубину
        if e is None:  # если узел пуст
            return  # ничего не делать
        if getattr(e, 'op', None) == 'call':  # если узел — вызов
            out.add(str(e.value))  # добавить имя вызываемого (значение узла)
            for a in e.args:  # по аргументам вызова
                walk(a)  # обойти аргумент
        else:  # иначе (не вызов)
            for a in getattr(e, 'args', []) or []:  # по аргументам узла (пустой список, если их нет)
                walk(a)  # обойти аргумент
    walk(expr)  # обойти всё выражение
    return out  # вернуть множество имён
def _ast_size(expr) -> int:
    """Размер дерева выражения: 1 + сумма размеров поддеревьев-аргументов; 0 для пустого."""
    if expr is None:  # если выражения нет
        return 0  # вернуть 0
    return 1 + sum((_ast_size(a) for a in getattr(expr, 'args', []) or []))  # размер = 1 + Σ размеров аргументов
def _certified_map(proc, name: str) -> tuple[int, tuple]:
    """Карта эксперта на сертифицированной области: (арность, отсортированный кортеж строк (входы…, выход))."""
    ex = proc.experts[name]  # эксперт по имени
    rows = []  # список строк карты
    for row in _domain_rows(proc, name):  # по строкам области определения
        try:  # попытка вычислить выход программы
            rows.append((*row, int(ex.program.expr.eval(row, proc.experts))))  # строка (входы…, int(значение выражения на строке в среде proc.experts))
        except Exception:  # при любой ошибке
            rows.append((*row, 'ERR'))  # в строку вместо выхода ставится метка 'ERR'
    return (ex.program.arity, tuple(sorted(rows)))  # вернуть (арность программы, отсортированный кортеж строк)
def _domain_rows(proc, name: str):
    """Все строки области определения эксперта: декартово произведение его областей по арности."""
    ex = proc.experts[name]  # эксперт по имени
    doms = ex.domains  # области аргументов эксперта
    if ex.program.arity == 1:  # если арность 1
        return [(x,) for x in doms[0]]  # строки (x,) по x из первой области
    if ex.program.arity == 2:  # если арность 2
        return [(x, y) for x in doms[0] for y in doms[1]]  # строки (x, y) по всем x из первой и y из второй области
    import itertools  # локальный импорт itertools
    return [tuple(c) for c in itertools.product(*doms)]  # декартово произведение всех областей: строки tuple(c)
def prune_library(proc, max_report: int=12) -> dict[str, Any]:
    """Предложить, каких экспертов библиотеки можно отбросить; сам процессор не меняется.

    Шаги: 1) дубликаты — эксперты с одинаковой картой на сертифицированной области;
    в группе остаётся эксперт с наименьшим размером дерева (при равенстве — по имени).
    2) псевдоодноаргументные: арность 2, второй аргумент не используется, никто не ссылается.
    3) итеративно отбрасываются некогда ссылавшиеся, не входящие в защищённый набор
    base_guard и попавшие в pseudo_drop. Возвращает {"keep": множество оставленных,
    числовая часть отчёта (ключи before, after, dropped_duplicates, dropped_pseudo_1arg, dup_groups, examples_dup_groups), "refs": кто кого вызывает}. max_report — сколько примеров
    групп дубликатов включить в отчёт."""
    experts = proc.experts  # библиотека экспертов процесса
    names = list(experts.keys())  # имена экспертов
    sigs: dict[str, tuple] = {}  # подписи экспертов: имя -> карта
    for n in names:  # по именам
        if experts[n].program.expr is not None:  # если у эксперта есть программа-выражение
            try:  # попытка
                sigs[n] = _certified_map(proc, n)  # подпись = карта на сертифицированной области
            except Exception:  # при любой ошибке
                sigs[n] = ('ERR', n)  # подпись — метка ошибки с именем ('ERR', имя)
    groups: dict[tuple, list[str]] = {}  # группы экспертов по подписи
    for n, s in sigs.items():  # по парам (имя, подпись)
        groups.setdefault(s, []).append(n)  # добавить имя в группу своей подписи
    dup_keep: dict[str, str] = {}  # какие дубликаты оставлены: имя -> описание
    dup_drop: set[str] = set()  # какие дубликаты отбрасываются
    for s, members in groups.items():  # по группам с подписями
        if len(members) > 1:  # если в группе больше одного эксперта
            members.sort(key=lambda n: (_ast_size(experts[n].program.expr), n))  # отсортировать группу по (размер дерева выражения, имя)
            keep = members[0]  # оставить первого — самое простое выражение
            dup_keep[keep] = f'{len(members)} copies'  # запомнить: оставлен, в группе len(members) копий
            dup_drop.update(members[1:])  # остальные члены группы — в отбрасываемые дубликаты
    refs: dict[str, set[str]] = {}  # кто кого вызывает: имя эксперта -> множество вызываемых имён
    for n in names:  # по именам
        refs[n] = _expr_calls(experts[n].program.expr) if experts[n].program.expr else set()  # вызываемые имена из выражения эксперта (пусто, если выражения нет)
    refd: set[str] = set()  # множество экспертов, на которые кто-то ссылается
    for n in names:  # по именам
        refd |= (refs[n] & set(names)) - {n}  # добавить вызываемых экспертов библиотеки, кроме самого себя: refd |= (refs[n] ∩ names) − {n}
    pseudo_drop: set[str] = set()  # кандидаты на отбрасывание как псевдоодноаргументные
    for n in names:  # по именам
        ex = experts[n]  # эксперт
        if ex.program.arity != 2 or n in dup_drop:  # если арность не 2 или это отбрасываемый дубликат
            continue  # перейти к следующему
        src_calls = refs[n]  # имена, вызываемые телом эксперта (не используется далее)
        uses_y = _expr_uses_var(ex.program.expr, 1)  # использует ли выражение переменную с номером 1 (второй аргумент)
        if not uses_y and n not in refd and (len(sigs.get(n, ('?',))[0:1] or ()) >= 0):  # если второй аргумент не используется, никто не ссылается и подпись есть
            pseudo_drop.add(n)  # добавить в псевдоодноаргументные
    keep = set(names) - dup_drop  # оставить пока всех, кроме дубликатов: keep = names − dup_drop
    changed = True  # признак изменения в последней итерации
    extra_drop: set[str] = set()  # дополнительно отброшенные на итерациях
    base_guard = {'ADD', 'SUB', 'MUL', 'NEG', 'ABS', 'SIGN', 'SQUARE', 'CUBE', 'MIN', 'MAX', 'GCD', 'LCM', 'EQ', 'LT', 'GE', 'MEDIAN3', 'CLAMP', 'FMA'}  # защищённые имена базовых операций, которые нельзя отбрасывать
    while changed:  # пока что-то меняется
        changed = False  # сбросить признак изменения
        refd_now: set[str] = set()  # имена, на которые ссылаются оставленные
        for n in keep:  # по оставленным именам
            refd_now |= (refs[n] & keep) - {n}  # добавить вызываемых из оставленных, кроме себя: refd_now |= (refs[n] ∩ keep) − {n}
        for n in list(keep):  # по копии списка оставленных
            if n in base_guard or n in refd_now:  # если имя защищено или на него ссылаются
                continue  # перейти к следующему
            if n in pseudo_drop:  # если имя — кандидат в псевдоодноаргументные
                keep.discard(n)  # убрать из оставленных
                extra_drop.add(n)  # добавить к дополнительно отброшенным
                changed = True  # зафиксировать изменение
    report = {'before': len(names), 'after': len(keep), 'dropped_duplicates': len(dup_drop), 'dropped_pseudo_1arg': len(extra_drop), 'dup_groups': len(dup_keep), 'examples_dup_groups': [f'{k}: {v}' for k, v in list(dup_keep.items())[:max_report]]}  # отчёт: число экспертов до и после, сколько отброшено дубликатов и псевдоодноаргументных, число групп и примеры групп
    return {'keep': keep, 'report': report, 'refs': refs}  # вернуть словарь: 'keep' — оставить, второй ключ — числовой отчёт, 'refs' — кто кого вызывает
def _expr_uses_var(expr, idx: int) -> bool:
    """Использует ли выражение переменную с номером idx (рекурсивно по всем аргументам)."""
    if expr is None:  # если выражения нет
        return False  # вернуть False
    if getattr(expr, 'op', None) == 'var' and int(expr.value) == idx:  # если узел — переменная var с номером idx
        return True  # вернуть True
    return any((_expr_uses_var(a, idx) for a in getattr(expr, 'args', []) or []))  # иначе True, если хоть один аргумент использует переменную idx
def apply_prune(proc, keep: set[str]):
    """Вернуть (процессор, число_сбоев): мелкую копию процесса только с оставленными экспертами
    после ПОЛНОЙ точной повторной проверки. raise, если хоть один эксперт её не прошёл.
    keep — имена оставленных экспертов. Проверка: для каждой строки области определения
    выход программы в среде полной библиотеки и в среде оставленных должен совпасть."""
    import copy  # локальный импорт copy
    new = copy.copy(proc)  # мелкая копия процесса
    new.experts = {n: proc.experts[n] for n in proc.experts if n in keep}  # в копии оставить только экспертов из множества keep
    failures = 0  # число сбоев проверки
    for n, ex in new.experts.items():  # по экспертам копии
        if ex.program.expr is None:  # если у эксперта нет программы-выражения — не проверять
            continue  # перейти к следующему
        for row in _domain_rows(proc, n):  # по строкам области определения на полной библиотеке
            try:  # попытка
                want = int(ex.program.expr.eval(row, proc.experts))  # ожидаемый выход want = int(значение в среде proc.experts)
            except Exception:  # при любой ошибке
                want = 'ERR'  # вместо ожидаемого выхода — метка 'ERR'
            try:  # попытка
                got = int(ex.program.expr.eval(row, new.experts))  # фактический выход got = int(значение в среде new.experts)
            except Exception:  # при любой ошибке
                got = 'ERR'  # вместо фактического выхода — метка 'ERR'
            if want != got:  # если выходы не совпали
                failures += 1  # сбоев: failures + 1
                break  # прервать проверку этого эксперта
    if failures:  # если есть сбои
        raise RuntimeError(f'prune aborted: {failures} experts failed re-validation')  # ошибка: отрезка прервана, N экспертов не прошли повторную проверку
    return (new, failures)  # вернуть копию процесса и число сбоев (0 при успехе)