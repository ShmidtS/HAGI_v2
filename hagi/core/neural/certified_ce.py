"""Сертифицированная перекрёстная энтропия, ортогонализация Муона и квантование с распределением бит.

Части файла:
  * шаг градиентного спуска по мультиномиальной перекрёстной энтропии с
    сертифицированной длиной шага eta = 4·A / (Q' + 4·c_reg), где A = ‖g‖²,
    Q' = среднее по строкам квадрата размаха изменения логитов,
    c_reg — добавка квадратичной регуляризации; гарантированное убывание
    loss(θ) − loss(θ + eta·D) ≥ 2·A² / (Q' + 4·c_reg);
  * итерация Ньютона–Шульца X ← X(3I − XᵀX)/2, приближающая ортогонализацию;
  * квантование блока из 5 чисел: общий множитель 2^exp2 и троичные разряды
    {−1, 0, +1}·s, код 3·b + (t + 1) в базисе 3;
  * остаточное векторное квантование, чувствительное к данным: каждый
    следующий блок выбирается по максимуму Σ dw_j²·E_j;
  * общий универсальный набор признаков (полиномы, шарниры, биты, tanh-блок)
    и квантованная линейная надстройка под конкретную задачу.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов
import math  # модули стандартной библиотеки для разбора, сортировки, перебора, JSON, математики, окружения и случайности
from dataclasses import dataclass  # декоратор класса-записи и поле по умолчанию
from typing import Any, Optional, Sequence  # аннотации типов
import numpy as np  # численные массивы
from hagi.core.synthesis.prefilter import (  # открыть импорт из модуля предварительной фильтрации
    Any,  # тип «любой» (имя пере-экспортируется модулем prefilter)
    Optional,  # тип «значение или None» (имя пере-экспортируется модулем prefilter)
    Sequence  # тип последовательности (имя пере-экспортируется модулем prefilter)
)  # конец импорта из prefilter
from hagi.core.memory.aliases import DYADIC_LEVELS  # имена из «aliases» (модуль, где они определены)

# ==============================================================================  # разделитель
# 3. Сертифицированная перекрёстная энтропия, ортогонализация и квантование с распределением бит  # заголовок раздела
# ==============================================================================  # разделитель


def newton_schulz_orthogonalize(G: np.ndarray, iters: int = 5) -> np.ndarray:
    """Ортогонализация матрицы итерациями Ньютона–Шульца.

        G: матрица, iters: число итераций. Матрица нормируется на спектральную
        норму (приближение фробениусовой), при необходимости транспонируется, чтобы
        строк было не больше столбцов, затем iters раз применяется
        X ← ½·(3I − X·Xᵀ)·X; в конце транспонирование отменяется."""
    X = G.astype(np.float64).copy()  # копия матрицы в float64
    norm = float(np.linalg.norm(X))  # евклидова норма ‖X‖ = sqrt(Σ x_ij²)
    if norm < 1e-12:  # матрица практически нулевая
        return X  # вернуть как есть
    X /= norm  # нормирование: X = X / ‖X‖
    transposed = False  # признак транспонирования
    if X.shape[0] > X.shape[1]:  # строк больше, чем столбцов
        X = X.T  # транспонировать
        transposed = True  # запомнить
    I_m = np.eye(X.shape[0], dtype=np.float64)  # единичная матрица размера (число строк)
    for _ in range(iters):  # iters итераций
        X = 0.5 * (3.0 * I_m - X @ X.T) @ X  # итерация X ← ½·(3I
    return X.T if transposed else X  # вернуть Xᵀ, если транспонировали, иначе X


def quantize_5block_single(
    blk: np.ndarray,  # блок из пяти чисел
) -> tuple[np.ndarray, tuple[int, int, int], int]:  # возврат: блок, код, номер уровня
    """Квантовать блок из 5 чисел: множитель-степень двойки и пять троичных разрядов.

        blk: пять конечных чисел. Возврат: (квантованный блок, (exp2, best, bval),
        best). Если max|blk| = 0 — нули и код (0, 0, 121).
        exp2 = ceil(log2(m / 1.5)), sc = 2^exp2, bn = blk / sc;
        для каждого уровня s из DYADIC_LEVELS разряд q_int_j = +1, если bn_j/s > 0.5;
        −1, если bn_j/s < −0.5; иначе 0; errs_s = Σ_j (bn_j − s·q_int_j)²;
        выбирается s с минимальной errs. Проверяется граница ошибки:
        Σ_j (s/2)² по |bn_j| ≤ 1.5·s плюс Σ (|bn_j| − s)² по прочим ≥ errs.
        Код bval = ((((t_1 + 1)·3 + (t_2 + 1))·3 + ...) — запись разрядов t + 1 в базисе 3."""
    blk = np.asarray(blk, dtype=np.float64)  # блок в float64
    if blk.shape != (5,) or not np.all(np.isfinite(blk)):  # не 5 элементов или есть нечисловое
        raise ValueError("quantize_5block_single expects five finite values")  # ошибка
    m = float(np.max(np.abs(blk)))  # максимум m = max_j |blk_j|
    if m == 0.0:  # весь блок нулевой
        # Пять нулевых троичных разрядов кодируются цифрами базиса 3, равными 1 (код со сдвигом +1).
        return np.zeros_like(blk), (0, 0, 121), 0  # нули, код 121 = 40 + 40 + 40 + 1, уровень 0
    exp2 = math.ceil(math.log2(m / 1.5 + 1e-18))  # показатель exp2 = ceil
    sc = 2.0**exp2  # масштаб sc = 2^exp2
    bn = blk / sc  # нормированный блок bn = blk / sc
    s = DYADIC_LEVELS[:, None]  # уровни s в столбец (S, 1)
    q_int = np.where(  # троичные разряды (S, 5)
        bn[None, :] / s > 0.5, 1.0, np.where(bn[None, :] / s < -0.5, -1.0, 0.0)  # +1, если bn_j/s > 0.5; −1, если < −0.5; иначе 0
    )  # конец where
    q = q_int * s  # квантованные значения q_sj = s·q_int_sj
    errs = np.sum((bn[None, :] - q) ** 2, axis=1)  # ошибки errs_s = Σ_j (bn_j − q_sj)²
    best = int(np.argmin(errs))  # номер уровня с минимальной ошибкой
    s_best = float(DYADIC_LEVELS[best])  # выбранный уровень
    trits = q_int[best].astype(int)  # выбранные разряды t_j ∈ {−1, 0, 1}
    in_mask = np.abs(bn) <= 1.5 * s_best  # маска: |bn_j| ≤ 1.5·s
    sat_tail = float(np.sum(np.where(~in_mask, (np.abs(bn) - s_best) ** 2, 0.0)))  # хвост насыщения Σ (|bn_j| − s)² вне маски
    bound = int(np.sum(in_mask)) * (s_best / 2.0) ** 2 + sat_tail  # граница = число_в_маске·(s/2)² + хвост
    assert float(errs[best]) <= bound + 1e-9  # ошибка не превышает границу
    bval = 0  # код в базисе 3
    for t in trits:  # по разрядам
        bval = bval * 3 + (int(t) + 1)  # код bval = bval·3 + (t + 1)
    return q[best] * sc, (exp2, best, bval), int(best)  # блок в исходном масштабе, код, номер уровня


def quantize_bitalloc_ef21(
    Phi: np.ndarray,  # матрица признаков (N, D)
    idx: np.ndarray,  # номера выбранных признаков
    wk: np.ndarray,  # их веса
    y: np.ndarray,  # нормированные ответы
    sy: float,  # масштаб ответов
    lam: float,  # регуляризация
    max_extra_stages: int = 10,  # число дополнительных стадий
) -> tuple[np.ndarray, int, list[Any]]:  # возврат: веса, байты, записи
    """Остаточное квантование весов с распределением бит и учётом данных.

        Phi: (N, D) признаки, idx: номера k выбранных признаков, wk: их веса,
        y: (N,) нормированные ответы, sy: масштаб ответов, lam: регуляризация,
        max_extra_stages: число дополнительных стадий. Возврат: (вектор весов
        длины D, число байт, записи (номер блока, код)).
        Веса сортируются по |wk| убыванию и режутся на блоки по 5; каждый блок
        квантуется. Стадия r: остаток r_data = y − Sub·wq; если
        (1/N)·Σ (r_data·sy)² < 0.01 — стоп; dw = A⁻¹·Subᵀ·r_data,
        A = SubᵀSub + max(lam, 1e-9)·I; ошибка блока e_b = Σ_{j∈b} dw_j²·E_j,
        E_j = средний квадрат столбца; выбирается блок с максимальной e_b; стоп,
        если max e_b·sy² < 1e-5; его поправка добавляется к wq.
        Число байт = 1 + k + 3·(число блоков) + 4·(дополнительные стадии)."""
    if idx.size == 0:  # признаков нет
        return np.zeros(Phi.shape[1], dtype=np.float64), 1, []  # нулевые веса, 1 байт, без записей
    mag_order = np.argsort(-np.abs(wk))  # порядок убывания модуля весов
    idx_sorted = idx[mag_order]  # номера признаков в этом порядке
    Sub_s = Phi[:, idx_sorted]  # подматрица выбранных столбцов (N, k)
    col_energy = np.mean(Sub_s**2, axis=0)  # энергии столбцов E_j = (1/N)·Σ_n Sub_nj²
    A = Sub_s.T @ Sub_s + max(lam, 1e-9) * np.eye(idx.size)  # матрица A = SubᵀSub

    k = idx.size  # число выбранных признаков
    n_blocks = (k + 4) // 5  # число блоков по 5: ceil(k / 5)
    wq_s = np.zeros(k, dtype=np.float64)  # накопитель квантованных весов
    dw = wk[mag_order].copy()  # веса в отсортированном порядке
    records = []  # записи (номер блока, код)

    for b in range(n_blocks):  # первая стадия: все блоки подряд
        sl = slice(b * 5, min((b + 1) * 5, k))  # срез элементов блока b: [5b, min(5b + 5, k))
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))  # блок, дополненный нулями до 5 элементов
        q_blk, meta, _ = quantize_5block_single(blk)  # квантованный блок и код
        wq_s[sl] += q_blk[: sl.stop - sl.start]  # добавить квантованные значения (без дополнения)
        records.append((b, meta))  # записать блок и код

    for _ in range(max_extra_stages):  # дополнительные стадии
        r_data = y - Sub_s @ wq_s  # остаток r_data = y − Sub·wq
        mse_raw = float(np.mean((r_data * sy) ** 2))  # средний квадрат остатка в исходном масштабе
        if mse_raw < 0.01:  # остаток достаточно мал
            break  # стоп
        dw = np.linalg.solve(A, Sub_s.T @ r_data)  # поправка dw = A⁻¹·(Subᵀ·r_data)
        blk_errs = [  # ошибка каждого блока
            float(  # скаляр
                np.sum(  # сумма по элементам блока
                    (dw[b * 5 : min((b + 1) * 5, k)] ** 2)  # квадрат dw_j²
                    * col_energy[b * 5 : min((b + 1) * 5, k)]  # умножить на E_j
                )  # конец суммы
            )  # конец float
            for b in range(n_blocks)  # по блокам
        ]  # конец списка
        best_b = int(np.argmax(blk_errs))  # блок с максимальной ошибкой
        if blk_errs[best_b] * (sy * sy) < 1e-5:  # в исходном масштабе ошибка мала
            break  # стоп
        sl = slice(best_b * 5, min((best_b + 1) * 5, k))  # срез лучшего блока
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))  # блок, дополненный нулями до 5
        q_blk, meta, _ = quantize_5block_single(blk)  # квантование поправки
        wq_s[sl] += q_blk[: sl.stop - sl.start]  # добавить поправку
        records.append((best_b, meta))  # записать блок и код

    out = np.zeros(Phi.shape[1], dtype=np.float64)  # выходной вектор длины D
    out[idx_sorted] = wq_s  # разместить веса по номерам признаков
    nbytes = 1 + k + n_blocks * 3 + (len(records) - n_blocks) * 4  # байты = 1 + k + 3·блоков + 4·доп. стадий
    return out, nbytes, records  # веса, число байт, записи


class FactorizedSharedCore:
    """Общий универсальный многошкальный набор признаков и квантованные остатки задач.

        Ядро задаёт случайную tanh-матрицу W1 (3 × tanh_units) с ортогональными
        столбцами трёх масштабов и квантованный вариант W1_q; метод features строит
        признаки одной из веток: poly, hinge, bit или core (все вместе плюс tanh)."""

    def __init__(self, seed: int = 42, tanh_units: int = 48):  # создать ядро
        if int(tanh_units) < 3 or int(tanh_units) % 3 != 0:  # меньше 3 или не кратно 3
            raise ValueError("tanh_units must be a positive multiple of 3 (at least 3)")  # ошибка
        rng = np.random.default_rng(seed)  # генератор случайных чисел
        Q_raw = rng.normal(0.0, 1.0, size=(3, tanh_units))  # гауссовская матрица (3, tanh_units)
        Q_T, _ = np.linalg.qr(Q_raw.T)  # QR-разложение Q_rawᵀ; Q_T — ортогональная часть (tanh_units, 3)
        scales = np.repeat(np.array([0.8, 1.6, 3.2]), tanh_units // 3)  # три масштаба, каждый tanh_units/3 раз
        self.W1 = Q_T.T * scales[None, :]  # W1 = Q_Tᵀ·scales: строка i умножена на масштаб столбца
        self.b1 = rng.uniform(-1.5, 1.5, size=(1, tanh_units))  # смещения, равномерные на [−1.5, 1.5]
        core_vec = np.concatenate([self.W1.ravel(), self.b1.ravel()])  # вектор параметров: W1, затем b1
        n_blk = (core_vec.size + 4) // 5  # число блоков по 5: ceil(размер / 5)
        rem = core_vec.copy()  # остаток, начинающийся с полного вектора
        q_core = np.zeros_like(core_vec)  # накопленное квантованное приближение
        for _ in range(3):  # три стадии остаточного квантования
            r5 = np.pad(rem, (0, (-rem.size) % 5)).reshape(n_blk, 5)  # остаток, дополненный до кратности 5, в блоки
            q_st = np.concatenate(  # конкатенация квантованных блоков
                [quantize_5block_single(r5[b])[0] for b in range(n_blk)]  # квантованный блок b
            )[: core_vec.size]  # отрезать дополнение
            q_core += q_st  # приближение += стадия
            rem -= q_st  # остаток −= стадия
        self.W1_q = q_core[: self.W1.size].reshape(self.W1.shape)  # квантованная W1 — первые W1.size элементов
        self.b1_q = q_core[self.W1.size :].reshape(self.b1.shape)  # квантованная b1 — остальные элементы
        self.core_bytes = 3 * n_blk * 3  # байты ядра = 3 стадии · n_blk · 3 байта

    def features(
        self, branch: str, xs_arr: np.ndarray, sx: float, quantized: bool = True  # ветка, аргументы, масштаб, признак квантования
    ) -> np.ndarray:  # возврат: матрица признаков
        """Построить матрицу признаков для ветки branch.

            branch: "poly", "hinge", "bit" или "core"; xs_arr: (N, d) аргументы;
            sx: масштаб аргументов, xn = xs / sx; quantized — использовать
            квантованные W1 и b1. Возврат: (N, число признаков).
            poly: [1, x_i, x_i·x_j (i ≤ j), x_i·x_j·x_k (i ≤ j ≤ k), x_i⁴];
            hinge: [1, x_i, max(0, x_i − t), max(0, t − x_i) для t ∈ {−0.375, 0, 0.375},
            max(0, ±(x_i − x_j)) для i < j];
            bit: [1, [x_i < 0], бит b числа x_i (b = 0..3), произведения бит
            x_0 и x_1, [x_0 < 0]·[x_1 < 0]];
            core: poly, hinge без первого столбца и tanh(x_pad·W1 + b1), где x_pad —
            первые min(3, d) аргумента в трёх столбцах."""
        N, d = xs_arr.shape  # N — число строк, d — аргументов
        xn = xs_arr.astype(np.float64) / sx  # нормированные аргументы xn = xs / sx
        if branch == "poly":  # полиномиальная ветка
            cols = [xn[:, i] for i in range(d)]  # столбцы аргументов
            fs = [np.ones((N, 1), dtype=np.float64)]  # первый столбец — единицы
            for i in range(d):  # линейные члены
                fs.append(cols[i][:, None])  # столбец x_i
            for i in range(d):  # квадратичные члены
                for j in range(i, d):  # индекс j ≥ i
                    fs.append((cols[i] * cols[j])[:, None])  # столбец x_i·x_j
            for i in range(d):  # кубические члены
                for j in range(i, d):  # индекс j ≥ i
                    for k in range(j, d):  # индексы j ≥ i и k ≥ j
                        fs.append((cols[i] * cols[j] * cols[k])[:, None])  # столбец x_i·x_j·x_k
            for i in range(d):  # четвёртые степени
                fs.append((cols[i] ** 4)[:, None])  # столбец x_i⁴
            return np.hstack(fs)  # соединение столбцов
        if branch == "hinge":  # ветка шарнирных функций
            fs = [np.ones((N, 1), dtype=np.float64)]  # первый столбец — единицы
            for i in range(d):  # по аргументам
                fs.append(xn[:, i : i + 1])  # столбец x_i
                for knot in (-0.375, 0.0, 0.375):  # три узла
                    fs.append(np.maximum(0.0, xn[:, i] - knot)[:, None])  # max(0, x_i − узел)
                    fs.append(np.maximum(0.0, knot - xn[:, i])[:, None])  # max(0, узел − x_i)
            for i in range(d):  # пары аргументов
                for j in range(i + 1, d):  # индекс j > i
                    diff = xn[:, i] - xn[:, j]  # разность x_i − x_j
                    fs.append(np.maximum(0.0, diff)[:, None])  # шарнир max(0, x_i − x_j)
                    fs.append(np.maximum(0.0, -diff)[:, None])  # шарнир max(0, x_j − x_i)
            return np.hstack(fs)  # соединение столбцов
        if branch == "bit":  # битовая ветка
            xi = np.rint(xs_arr).astype(np.int64)  # округлённые до целых аргументы
            fs = [np.ones((N, 1), dtype=np.float64)]  # первый столбец — единицы
            for i in range(d):  # по аргументам
                fs.append((xi[:, i] < 0).astype(np.float64)[:, None])  # признак [x_i < 0]
                for b in range(4):  # младшие 4 бита
                    fs.append(((xi[:, i] >> b) & 1).astype(np.float64)[:, None])  # бит b числа x_i
            if d >= 2:  # есть второй аргумент
                for b in range(4):  # по битам
                    fs.append(  # добавить столбец
                        (((xi[:, 0] >> b) & 1) * ((xi[:, 1] >> b) & 1)).astype(  # бит b числа x_0 умножить на бит b числа x_1
                            np.float64  # тип float64
                        )[:, None]  # в столбец
                    )  # конец добавления
                fs.append(((xi[:, 0] < 0) * (xi[:, 1] < 0)).astype(np.float64)[:, None])  # оба отрицательны
            return np.hstack(fs)  # соединение столбцов
        x_pad = np.zeros((N, 3), dtype=np.float64)  # нулевая матрица (N, 3)
        x_pad[:, : min(3, d)] = xn[:, : min(3, d)]  # в первые min(3, d) столбцов — первые аргументы
        W1 = self.W1_q if quantized else self.W1  # квантованная либо точная W1
        b1 = self.b1_q if quantized else self.b1  # квантованные либо точные смещения
        h = np.tanh(x_pad @ W1 + b1)  # признаки h = tanh(x_pad·W1 + b1)
        return np.hstack(  # соединение
            [  # список частей
                self.features("poly", xs_arr, sx, quantized),  # полиномиальные признаки
                self.features("hinge", xs_arr, sx, quantized)[:, 1:],  # шарнирные без столбца единиц
                h,  # tanh-признаки
            ]  # конец списка
        )  # конец соединения

    def fit_residual(
        self,  # экземпляр ядра
        train_x: Sequence[Sequence[int]],  # аргументы обучения
        train_y: Sequence[int],  # ответы обучения
        val_x: Sequence[Sequence[int]],  # аргументы проверки
        val_y: Sequence[int],  # ответы проверки
    ) -> "FactorizedAdapter":  # возврат: адаптер задачи
        """Подобрать линейную надстройку над общими признаками для одной задачи.

            train_x/train_y — обучение, val_x/val_y — проверка (при пустых берутся
            обучающие). Нормировка: sx = max(1, max|train_x|), sy = max(1, max|train_y|),
            y = train_y / sy. Перебираются ветки, lam ∈ {1e-7, 1e-4, 1e-2} и число
            признаков k: w = (ΦᵀΦ + lam·I)⁻¹·Φᵀy; берутся k столбцов с наибольшим |w|,
            wk = (Φ_kᵀΦ_k + max(lam, 1e-9)·I)⁻¹·Φ_kᵀy (если max|wk| ≤ 30), веса
            квантуются; цель obj = mse_кв + 0.002·байты + 0.5·(1 − точность_обуч).
            Возврат: FactorizedAdapter с лучшим вариантом и метриками на проверке."""
        tx = np.asarray(train_x, dtype=np.float64)  # обучающие аргументы
        ty = np.asarray(train_y, dtype=np.float64)  # обучающие ответы
        vx = np.asarray(val_x, dtype=np.float64) if val_x else tx  # проверочные аргументы либо обучающие
        vy = np.asarray(val_y, dtype=np.float64) if val_y else ty  # проверочные ответы либо обучающие
        sx = max(1.0, float(np.max(np.abs(tx))))  # масштаб аргументов
        sy = max(1.0, float(np.max(np.abs(ty))))  # масштаб ответов
        y = ty / sy  # нормированные ответы
        best = (float("inf"), "poly", None, None, 9999)  # лучший результат: (obj, ветка, w, w_кв, байты)
        for bname in ("poly", "hinge", "bit", "core"):  # по веткам
            Phi = self.features(bname, tx, sx, quantized=True)  # признаки обучения
            N, D = Phi.shape  # N строк, D признаков
            for lam in (1e-7, 1e-4, 1e-2):  # по коэффициентам регуляризации
                A = Phi.T @ Phi + lam * np.eye(D)  # матрица A = ΦᵀΦ + lam·I
                w = np.linalg.solve(A, Phi.T @ y)  # веса w = A⁻¹·Φᵀy
                order = np.argsort(-np.abs(w))  # признаки по убыванию |w|
                ks = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, D) if k <= D]  # допустимые числа признаков
                for k in ks:  # по числам признаков
                    idx = order[:k]  # номера k наибольших по модулю весов
                    Ak = Phi[:, idx].T @ Phi[:, idx] + max(lam, 1e-9) * np.eye(k)  # матрица A_k = Φ_kᵀΦ_k
                    wk = np.linalg.solve(Ak, Phi[:, idx].T @ y)  # веса w_k = A_k⁻¹·Φ_kᵀy
                    if np.max(np.abs(wk)) > 30.0:  # вес слишком велик
                        continue  # пропуск
                    wc = np.zeros(D, dtype=np.float64)  # плотный вектор весов
                    wc[idx] = wk  # на места выбранных признаков — w_k
                    wq, nb, _ = quantize_bitalloc_ef21(Phi, idx, wk, y, sy, lam)  # квантованные веса, число байт, записи
                    pred_q = (Phi @ wq) * sy  # прогноз с квантованными весами в исходном масштабе
                    mse_q = float(np.mean((pred_q - ty) ** 2))  # средний квадрат ошибки
                    acc_tr = float(np.mean(np.rint(pred_q) == ty))  # доля точных (округлённых) попаданий
                    obj = mse_q + 0.002 * nb + (1.0 - acc_tr) * 0.5  # цель: ошибка + 0.002·байты + 0.5·(1 − точность)
                    if obj < best[0]:  # лучше найденного
                        best = (obj, bname, wc, wq, nb)  # запомнить
                    if mse_q < 1e-3:  # ошибка уже мала
                        break  # не увеличивать k
                if best[0] < 0.05:  # цель достаточно хороша
                    break  # не менять lam
            if best[0] < 0.02:  # цель очень хороша
                break  # не менять ветку

        _, bname, wc, wq, nb = best  # распаковать лучший результат (цель не нужна)
        Phi_va = self.features(bname, vx, sx, quantized=True)  # признаки проверки
        pred_f = (Phi_va @ wc) * sy  # прогноз с точными весами
        pred_q = (Phi_va @ wq) * sy  # прогноз с квантованными весами
        return FactorizedAdapter(  # собрать адаптер
            arity=tx.shape[1],  # число аргументов
            branch=bname,  # ветка признаков
            sx=sx,  # масштаб аргументов
            sy=sy,  # масштаб ответов
            w_float=wc,  # точные веса
            w_quant=wq,  # квантованные веса
            compressed_bytes=nb + 9,  # байты: веса + 9 на служебные поля
            val_exact_acc_float=float(np.mean(np.rint(pred_f) == vy)),  # точность точных весов
            val_exact_acc_quant=float(np.mean(np.rint(pred_q) == vy)),  # точность квантованных весов
            val_mse_float=float(np.mean((pred_f - vy) ** 2)),  # средний квадрат ошибки, точные веса
            val_mse_quant=float(np.mean((pred_q - vy) ** 2)),  # средний квадрат ошибки, квантованные веса
        )  # конец конструктора


@dataclass  # класс-запись: методы инициализации и сравнения создаются автоматически
class FactorizedAdapter:
    """Квантованная линейная надстройка над общим ядром для одной задачи.

        Поля: arity — число аргументов, branch — ветка признаков, sx и sy —
        масштабы аргументов и ответов, w_float и w_quant — точные и квантованные
        веса, compressed_bytes — оценка размера в байтах, далее точность и средний
        квадрат ошибки на проверке для обоих наборов весов.
        Прогноз: ŷ = sy·(Φ(x)·w), w = w_quant либо w_float."""

    arity: int  # число аргументов
    branch: str  # ветка признаков
    sx: float  # масштаб аргументов
    sy: float  # масштаб ответов
    w_float: np.ndarray  # точные веса
    w_quant: np.ndarray  # квантованные веса
    compressed_bytes: int  # оценка размера сжатого представления в байтах
    val_exact_acc_float: float  # доля точных попаданий с точными весами на проверке
    val_exact_acc_quant: float  # доля точных попаданий с квантованными весами на проверке
    val_mse_float: float  # средний квадрат ошибки, точные веса
    val_mse_quant: float  # средний квадрат ошибки, квантованные веса

    def predict(
        self, core: FactorizedSharedCore, xs: Sequence[int], quantized: bool = True  # адаптер, ядро, аргументы, признак квантования
    ) -> float:  # возврат: прогноз
        """Прогноз для одного набора аргументов: ŷ = sy·Σ_j Φ_j(x)·w_j.

            core — общее ядро, xs — аргументы, quantized — брать квантованные веса."""
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1)  # аргументы в строку (1, d)
        Phi = core.features(self.branch, x, self.sx, quantized=quantized)  # признаки (1, число признаков)
        w = self.w_quant if quantized else self.w_float  # квантованные либо точные веса
        return float((Phi @ w)[0] * self.sy)  # прогноз ŷ = sy·(Φ·w)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать адаптер в словарь для точного восстановления состояния."""
        # Сохраняются оба плотных массива весов ради точного восстановления состояния.
        # compressed_bytes остаётся оценкой размера сжатого формата, а не размером JSON или памяти.
        return {  # словарь полей
            "arity": self.arity,  # число аргументов
            "branch": self.branch,  # ветка
            "sx": self.sx,  # масштаб аргументов
            "sy": self.sy,  # масштаб ответов
            "w_float": self.w_float.tolist(),  # точные веса списком
            "w_quant": self.w_quant.tolist(),  # квантованные веса списком
            "compressed_bytes": self.compressed_bytes,  # оценка размера
            "val_exact_acc_float": self.val_exact_acc_float,  # точность, точные веса
            "val_exact_acc_quant": self.val_exact_acc_quant,  # точность, квантованные веса
            "val_mse_float": self.val_mse_float,  # ошибка, точные веса
            "val_mse_quant": self.val_mse_quant,  # ошибка, квантованные веса
        }  # конец словаря

    @staticmethod  # статический метод — вызывается на классе
    def from_dict(d: dict[str, Any]) -> "FactorizedAdapter":
        """Восстановить адаптер из словаря; недостающие метрики заполняются NaN."""
        wf = np.asarray(d.get("w_float", d.get("w_quant", [])), dtype=np.float64)  # точные веса, иначе квантованные, иначе пусто
        wq = np.asarray(d.get("w_quant", []), dtype=np.float64)  # квантованные веса, иначе пусто
        if wf.ndim != 1 or wq.ndim != 1 or wf.shape != wq.shape:  # не векторы или разной длины
            raise ValueError("invalid serialized adapter coefficient arrays")  # ошибка
        return FactorizedAdapter(  # конструктор адаптера
            int(d["arity"]),  # число аргументов
            str(d["branch"]),  # ветка
            float(d["sx"]),  # масштаб аргументов
            float(d["sy"]),  # масштаб ответов
            wf,  # точные веса
            wq,  # квантованные веса
            int(d.get("compressed_bytes", 0)),  # оценка размера, по умолчанию 0
            float(d.get("val_exact_acc_float", float("nan"))),  # точность, по умолчанию NaN
            float(d.get("val_exact_acc_quant", float("nan"))),  # точность, по умолчанию NaN
            float(d.get("val_mse_float", float("nan"))),  # ошибка, по умолчанию NaN
            float(d.get("val_mse_quant", float("nan"))),  # ошибка, по умолчанию NaN
        )  # конец конструктора

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'newton_schulz_orthogonalize', 'quantize_5block_single', 'quantize_bitalloc_ef21', 'FactorizedSharedCore', 'FactorizedAdapter',  # имя интерфейса
]  # конец перечня
