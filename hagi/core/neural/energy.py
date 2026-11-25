"""Энергетический решатель CRAM: раскраска графа и латентное поле расстояний до цели.

Идея: задача поиска задаётся энергетической функцией, и решение ищется спуском
по энергии с шумом и случайными перезапусками.

Раскраска графа (solve_coloring):
  * энергия E(s) = Σ_{(a,b) ∈ рёбра} [s_a == s_b] — число одноцветных рёбер;
    допустимая раскраска ⟺ E = 0;
  * поле вершины i: f_c = Σ_{j ∈ соседи(i)} [s_j == c]; смена цвета i уменьшает
    энергию ровно на f_{s_i} − f_c, поэтому выгоден цвет с минимальным f_c;
  * шум: f'_c = f_c + u_c, u_c ~ U[−eps, eps]; смена цвета только если
    f'_{s_i} − min_c f'_c > 2·eps (тогда улучшение гарантировано при любом шуме);
  * при остановке спуска — случайные шаги с температурой T:
    P(c) = exp(−f_c / T) / Σ_k exp(−f_k / T); после 8 неудачных проходов — перезапуск.

Латентный лабиринт (solve_latent_maze): сжимающее отображение
  z_i ← (Σ_{j ∈ соседи(i)} z_j + [i == goal]) / m, m = Δ + 1, Δ — максимум степени.
  Коэффициент сжатия q = Δ / m < 1; оценка ошибки после шага
  error_bound = ‖z_new − z‖_∞ · q / (1 − q); предсказанное число шагов
  n = ceil(log(eps·(1 − q)) / log q).

Экспортирует класс CRAMEngine с двумя статическими методами.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов
import math  # модули стандартной библиотеки: аргументы командной строки, абстрактное дерево, копирование, куча, переборы, JSON, математика, ОС, случайность
from dataclasses import field  # декоратор класса-записи и описание поля по умолчанию
from typing import Any, Optional, Sequence  # типы аннотаций: произвольный, вызываемый, перечислимый, необязательный, последовательность
import numpy as np  # численные массивы
from hagi.core.synthesis.prefilter import (  # импорт из модуля предварительной фильтрации
    Any,  # тип «любой»
    Optional,  # тип «необязательное значение»
    Sequence  # тип «последовательность»
)  # конец импорта

# ==============================================================================
# Энергетический решатель задач удовлетворения ограничений
# ==============================================================================


class CRAMEngine:
    """Решатель задач на графе через минимизацию энергии.

        Статические методы: solve_coloring — раскраска вершин в num_colors цветов
        так, чтобы концы каждого ребра различались; solve_latent_maze — латентное
        поле z, указывающее направление к цели, с оценкой ошибки."""

    @staticmethod  # метод без доступа к экземпляру
    def solve_coloring(  # найти раскраску графа
        edges: Sequence[tuple[int, int]],  # рёбра — пары номеров вершин
        n_nodes: int,  # число вершин
        num_colors: int,  # число цветов
        givens: Optional[dict[int, int]] = None,  # фиксированные цвета: вершина -> цвет
        eps_noise: float = 0.0,  # амплитуда шума eps
        max_sweeps: int = 800,  # предел числа проходов
        seed: int = 42,  # зерно генератора случайных чисел
    ) -> dict[str, Any]:  # аннотация возврата
        """Найти раскраску графа и вернуть её с метриками.

            edges — рёбра, n_nodes — число вершин, num_colors — число цветов,
            givens — заранее заданные цвета. Энергия E(s) = Σ_{(a,b)} [s_a == s_b].
            Проход: для каждой незакреплённой вершины цвет меняется, если
            f'_{s_i} − min_c f'_c > 2·eps, на случайный цвет с минимальным полем.
            Без улучшения — случайные шаги по P(c) ∝ exp(−f_c / T), T = 0.55;
            после 8 подряд неудачных проходов — перезапуск. Возвращает словарь:
            coloring, valid, energy, sweeps, hb_steps, verified (найдено решение),
            givens_respected, оценки объёма работы за проход."""
        if int(n_nodes) <= 0:  # число вершин не положительно
            raise ValueError("n_nodes must be positive")  # ошибка аргумента
        if int(num_colors) <= 0:  # число цветов не положительно
            raise ValueError("num_colors must be positive")  # ошибка аргумента
        if int(max_sweeps) < 0:  # отрицательный предел проходов
            raise ValueError("max_sweeps must be nonnegative")  # ошибка аргумента
        if not math.isfinite(float(eps_noise)) or float(eps_noise) < 0:  # шум не конечен или отрицателен
            raise ValueError("eps_noise must be finite and nonnegative")  # ошибка аргумента
        for a, b in edges:  # по рёбрам
            if not (0 <= int(a) < n_nodes and 0 <= int(b) < n_nodes):  # конец ребра вне диапазона 0..n_nodes−1
                raise ValueError(f"edge endpoint outside [0, {n_nodes})")  # ошибка ребра
        adj = [set() for _ in range(n_nodes)]  # списки смежности: множество соседей каждой вершины
        for a, b in edges:  # по рёбрам
            if a != b:  # петля игнорируется
                adj[a].add(b)  # b — сосед a
                adj[b].add(a)  # a — сосед b
        adj_list = [sorted(s) for s in adj]  # соседи каждой вершины отсортированы
        giv = {int(i): int(c) for i, c in (givens or {}).items()}  # фиксированные цвета: ключ и значение приведены к целым
        if any(i < 0 or i >= n_nodes for i in giv):  # номер закреплённой вершины вне графа
            raise ValueError("given node index outside graph")  # ошибка
        if any(c < 0 or c >= num_colors for c in giv.values()):  # закреплённый цвет вне палитры 0..num_colors−1
            raise ValueError("given color outside palette")  # ошибка
        rng = np.random.default_rng(seed)  # генератор случайных чисел по зерну seed
        y = [giv.get(i, int(rng.integers(0, num_colors))) for i in range(n_nodes)]  # начальная раскраска: заданный цвет или случайный из 0..num_colors−1

        def energy(st: Sequence[int]) -> int:  # энергия состояния st
            """Энергия E(st) = Σ_{(a,b)} [st_a == st_b] — число одноцветных рёбер."""
            return sum(1 for a, b in edges if st[a] == st[b])  # сосчитать рёбра с одинаковым цветом концов

        def sparse_fields(i: int, st: Sequence[int]) -> list[int]:  # поле вершины i в состоянии st
            """Поле вершины i: f_c = Σ_{j ∈ соседи(i)} [st_j == c]; возвращает список по цветам."""
            f = [0] * num_colors  # нулевые поля по всем цветам
            for j in adj_list[i]:  # по соседям вершины
                f[st[j]] += 1  # полю цвета соседа +1
            return f  # поля по цветам

        cur_e = energy(y)  # текущая энергия E(y)
        sweeps = hb_steps = stale = 0  # число проходов, случайных шагов и счётчик застоя — нули
        T = 0.55  # температура случайных шагов
        sparse_work = sum(len(adj_list[i]) * (num_colors**2) for i in range(n_nodes))  # работа разрежённой схемы за проход: Σ_i deg(i)·C²
        dense_work = n_nodes * max(1, n_nodes - 1) * (num_colors**2)  # работа плотной схемы за проход: n·max(1, n − 1)·C²

        while sweeps < max_sweeps and cur_e > 0:  # пока есть проходы и энергия ненулевая
            improved = False  # признак улучшения в этом проходе
            for i in range(n_nodes):  # по вершинам
                if i in giv:  # вершина закреплена
                    continue  # не менять
                fields = sparse_fields(i, y)  # поля f_c вершины i
                noisy = [  # зашумлённые поля f'_c
                    f  # значение f_c
                    + (  # плюс
                        float(rng.uniform(-eps_noise, eps_noise))  # шум u ~ U[−eps, eps]
                        if eps_noise > 0  # если шум включён
                        else 0.0  # иначе нуль
                    )
                    for f in fields  # по цветам
                ]  # конец списка
                cur_f = noisy[y[i]]  # поле текущего цвета вершины f'_{s_i}
                best_val = min(noisy)  # минимальное зашумлённое поле min_c f'_c
                if cur_f - best_val > 2.0 * eps_noise:  # гарантированное улучшение: f'_{s_i} − min f' > 2·eps
                    ties = [c for c, v in enumerate(noisy) if v == best_val]  # цвета с минимальным полем
                    y[i] = ties[int(rng.integers(0, len(ties)))]  # выбрать из них случайный
                    improved = True  # в проходе было улучшение
            cur_e = energy(y)  # энергия после прохода
            sweeps += 1  # проходов: +1
            if cur_e == 0:  # энергия нулевая — раскраска допустима
                break  # закончить цикл
            if not improved:  # проход без улучшения — застой
                stale += 1  # застоя: +1
                if stale >= 8:  # восемь застоев подряд
                    y = [  # перезапуск: новая случайная раскраска
                        giv.get(i, int(rng.integers(0, num_colors)))  # заданный цвет или случайный
                        for i in range(n_nodes)  # по вершинам
                    ]
                    stale = 0  # счётчик застоя обнулён
                    cur_e = energy(y)  # энергия новой раскраски
                    continue  # перейти к следующему проходу
                for _ in range(min(12, max(3, cur_e * 2))):  # случайные шаги: min(12, max(3, 2·E))
                    conf = [  # конфликтные вершины
                        i  # вершина
                        for i in range(n_nodes)  # по вершинам
                        if i not in giv and any(y[i] == y[j] for j in adj_list[i])  # не закреплена и есть сосед того же цвета
                    ]
                    if not conf:  # конфликтных вершин нет
                        break  # закончить шаги
                    pick = conf[int(rng.integers(0, len(conf)))]  # случайная конфликтная вершина
                    if rng.random() < 0.4 and adj_list[pick]:  # с вероятностью 0.4 при наличии соседей — сдвиг к соседу
                        free_nb = [j for j in adj_list[pick] if j not in giv]  # незакреплённые соседи выбранной вершины
                        if free_nb:  # такие соседи есть
                            pick = free_nb[int(rng.integers(0, len(free_nb)))]  # случайный из них
                    fields = sparse_fields(pick, y)  # поля выбранной вершины
                    logits = np.array([-f / T for f in fields], dtype=np.float64)  # логиты l_c = −f_c / T
                    logits -= np.max(logits)  # вычесть максимум: l_c − max l
                    probs = np.exp(logits)  # экспоненты e_c = exp(l_c − max l)
                    probs /= np.sum(probs)  # вероятности P(c) = e_c / сумма_k e_k
                    y[pick] = int(rng.choice(num_colors, p=probs))  # новый цвет по распределению P
                    hb_steps += 1  # случайных шагов: +1
                cur_e = energy(y)  # энергия после случайных шагов

        valid = all(y[a] != y[b] for a, b in edges) and all(  # допустимость: все рёбра разноцветны и
            y[i] == g for i, g in giv.items()  # все закреплённые цвета соблюдены
        )  # конец условия
        return {  # результат
            "coloring": y,  # найденная раскраска
            "valid": valid,  # допустима ли раскраска
            "energy": cur_e,  # энергия E(y)
            "sweeps": sweeps,  # число проходов
            "hb_steps": hb_steps,  # число случайных шагов
            # verified означает, что допустимая раскраска действительно найдена;
            # для локально согласованного, но нерешённого (или неразрешимого)
            # запуска оно обязано быть ложным.
            "verified": bool(valid and cur_e == 0),  # решение найдено: допустимо и энергия равна нулю
            "givens_respected": all(y[i] == g for i, g in giv.items()),  # закреплённые цвета соблюдены
            "sparse_work_per_sweep": sparse_work,  # работа разрежённой схемы за проход
            "dense_work_per_sweep": dense_work,  # работа плотной схемы за проход
        }

    @staticmethod  # метод без доступа к экземпляру
    def solve_latent_maze(  # вычислить латентное поле до цели
        n_nodes: int,  # число вершин
        adj: Sequence[Sequence[int]],  # списки смежности — по одной строке на вершину
        goal: int,  # номер целевой вершины
        eps_z: float = 1e-6,  # требуемая точность eps
        max_iters: int = 100_000,  # предел числа итераций
    ) -> dict[str, Any]:  # аннотация возврата
        """Сжимающая итерация латентного поля с явной оценкой ошибки остановки.

            Итерация: z_i ← (Σ_{j ∈ соседи(i)} z_j + [i == goal]) / m, где m = Δ + 1,
            Δ — максимум степени вершины. Коэффициент сжатия q = Δ / m < 1; после
            шага error_bound = ‖z_new − z‖_∞ · q / (1 − q). Предсказанное число
            шагов n = ceil(log(eps·(1 − q)) / log q). Останов при error_bound <= eps.
            Без рёбер (q = 0) решение точное: z_goal = 1 / m. Проверяется инвариант
            0 <= z_i <= 1 / (m − Δ). Возвращает z, iters, contraction_q,
            upper_bound, error_bound, tolerance, converged."""
        if int(n_nodes) <= 0:  # число вершин не положительно
            raise ValueError("n_nodes must be positive")  # ошибка аргумента
        if len(adj) != n_nodes:  # число строк смежности не равно числу вершин
            raise ValueError("adjacency must have one row per node")  # ошибка аргумента
        if not (0 <= int(goal) < n_nodes):  # цель вне диапазона 0..n_nodes−1
            raise ValueError("goal outside graph")  # ошибка аргумента
        if not math.isfinite(float(eps_z)) or float(eps_z) <= 0:  # точность не конечна или не положительна
            raise ValueError("eps_z must be finite and positive")  # ошибка аргумента
        if int(max_iters) <= 0:  # предел итераций не положителен
            raise ValueError("max_iters must be positive")  # ошибка аргумента
        adj_sets: list[tuple[int, ...]] = []  # очищенные списки соседей
        for i, row in enumerate(adj):  # по строкам смежности с номером i
            ns = tuple(sorted(set(int(j) for j in row)))  # соседи: целые, без повторов, по возрастанию
            if any(j < 0 or j >= n_nodes for j in ns):  # сосед вне диапазона 0..n_nodes−1
                raise ValueError(f"neighbor index outside graph in row {i}")  # ошибка
            adj_sets.append(ns)  # добавить строку
        delta = max((len(nb) for nb in adj_sets), default=0)  # максимум степени Δ; пустой список — 0
        m = float(delta + 1)  # знаменатель m = Δ + 1
        q = delta / m  # коэффициент сжатия q = Δ / m
        z = np.zeros(n_nodes, dtype=np.float64)  # начальное поле из нулей
        iters = 0  # число итераций
        error_bound = float("inf")  # оценка ошибки — бесконечность
        if q > 0:  # в графе есть рёбра
            predicted = max(1, math.ceil(math.log(eps_z * (1.0 - q)) / math.log(q)))  # предсказанное число шагов n = max(1, ceil(log(eps·(1 − q)) / log q))
            run_limit = min(int(max_iters), predicted)  # предел запуска: min(max_iters, n)
            for _ in range(run_limit):  # до предела
                z_new = np.array(  # новое поле
                    [  # по вершинам
                        (sum(z[j] for j in adj_sets[i]) + (1.0 if i == goal else 0.0))  # Σ_{j ∈ соседи(i)} z_j + [i == goal]
                        / m  # делённое на m
                        for i in range(n_nodes)  # по вершинам
                    ],
                    dtype=np.float64,  # вещественные числа двойной точности
                )
                diff = float(np.max(np.abs(z_new - z)))  # максимум модуля разности ‖z_new − z‖
                z = z_new  # заменить поле
                iters += 1  # итераций: +1
                error_bound = diff * q / (1.0 - q)  # оценка ошибки: diff·q / (1 − q)
                if error_bound <= eps_z:  # достигнута требуемая точность
                    break  # закончить итерации
        else:  # рёбер нет — решение точно за один шаг
            z[goal] = 1.0 / m  # только цель ненулевая: z_goal = 1 / m
            iters = 1  # одна итерация
            error_bound = 0.0  # ошибка равна нулю
        upper = 1.0 / (m - delta)  # теоретическая верхняя граница поля 1 / (m − Δ)
        if not (  # проверка инварианта
            np.all(np.isfinite(z))  # все значения конечны
            and np.all(z >= -1e-12)  # и неотрицательны (с допуск 1e-12)
            and np.all(z <= upper + 1e-12)  # и не превосходят upper (с допуском 1e-12)
        ):  # конец условия
            raise ArithmeticError("latent-field invariant failed")  # инвариант нарушен
        converged = bool(error_bound <= eps_z)  # сходимость: оценка ошибки не больше eps
        return {  # результат
            "z": z,  # латентное поле
            "iters": iters,  # число итераций
            "contraction_q": q,  # коэффициент сжатия
            "upper_bound": upper,  # верхняя граница поля
            "error_bound": error_bound,  # оценка ошибки
            "tolerance": float(eps_z),  # требуемая точность
            "converged": converged,  # достигнута ли сходимость
        }

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'CRAMEngine',  # имя интерфейса
]  # конец перечня
