"""Голова Венна, упаковка масок, полином Жегалкина, Адамар, 5-тритовый кодек — часть символьного ядра HAGI (секция 4 бывшего монолита core.py).

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import math  # импорт стандартной библиотеки, нужный этой части
from typing import Callable, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.core.memory.aliases import DYADIC_LEVELS  # имена из «aliases» (прежний общий импорт из пакета наследия)
from hagi.core.language.expressions import Expr  # класс выражения ядра (источник как до слияния)

# 4. Голова Венна, упаковка 16 масок в 13 байт, полином Жегалкина, Адамар, 5-тритовый кодек
#============================== разделитель разделов ==============================


def venn_atoms(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Базис атомов Венна из четырёх ReLU-нейронов на входах {−1, +1}: матрица (N, 4).

    s1, s2 — входы со значениями −1 или +1. Ровно один атом равен 1, остальные
    0: a_0 = max(0, (−s1 − s2)/2), a_1 = max(0, (−s1 + s2)/2),
    a_2 = max(0, (s1 − s2)/2), a_3 = max(0, (s1 + s2)/2)."""
    return np.column_stack(  # соединить четыре столбца-атома в матрицу
        [  # начало списка столбцов-атомов
            np.maximum(0.0, (-s1 - s2) * 0.5),  # атом 0: max(0, (−s1 − s2)/2) — активен при s1 = s2 = −1
            np.maximum(0.0, (-s1 + s2) * 0.5),  # атом 1: max(0, (−s1 + s2)/2) — активен при s1 = −1, s2 = +1
            np.maximum(0.0, (s1 - s2) * 0.5),  # атом 2: max(0, (s1 − s2)/2) — активен при s1 = +1, s2 = −1
            np.maximum(0.0, (s1 + s2) * 0.5),  # атом 3: max(0, (s1 + s2)/2) — активен при s1 = s2 = +1
        ]
    )


def venn_head_eval(mask: int, x_bin: Sequence[int]) -> int:
    """Вычислить булеву функцию маски Венна на двух булевых входах как нейрон «атомы · веса».

    Входы 0/1 переводятся в −1/+1: s = 2·x − 1; вес w_idx = бит маски idx;
    результат = round(Σ_idx a_idx·w_idx)."""
    s1 = np.array([2.0 * int(x_bin[0]) - 1.0])  # s1 = 2·x_1 − 1 как массив из одного элемента
    s2 = np.array([2.0 * int(x_bin[1]) - 1.0])  # итог: s2 = 2·x_2 − 1
    atoms = venn_atoms(s1, s2)[0]  # атомы Венна для этой пары входов
    w = np.array([(mask >> idx) & 1 for idx in range(4)], dtype=np.float64)  # веса: w_idx = (маска >> idx) & 1 для idx = 0..3
    return int(round(float(atoms @ w)))  # вернуть round(атомы · веса)


def pack_venn_16_flat() -> bytes:
    """Упаковать 16 четырёхбитных масок в 13 байт без потерь.

    Все 64 бита масок собираются в поток по 5 штук (последняя пятёрка
    дополняется нулями до 65 бит); байт = Σ_{i=0..4} t_i·3^(4−i) — число в
    интервале 0..242; 13 байт = 65 бит — точная нижняя граница."""
    """Упаковать 16 масок по 4 трита в 13 байт без потерь (точная нижняя граница)."""
    trits = [(mask >> idx) & 1 for mask in range(16) for idx in range(4)]  # поток битов: для каждой маски 0..15 её биты idx = 0..3
    while len(trits) % 5 != 0:  # пока длина не кратна 5
        trits.append(0)  # добавить нулевой бит
    out = bytearray()  # накопитель байтов
    for off in range(0, len(trits), 5):  # по пятёркам бит
        b = 0  # байт, изначально 0
        for t in trits[off : off + 5]:  # по пяти битам t пятёрки
            b = b * 3 + t  # b = b·3 + t — накопление числа по основанию 3
        out.append(b)  # добавить байт
    assert len(out) == 13  # проверка: ровно 13 байт
    return bytes(out)  # вернуть упакованные байты


def unpack_venn_16_flat(buf: bytes) -> list[int]:
    """Распаковать 13 байт обратно в список из 16 масок (действие, обратное pack_venn_16_flat)."""
    assert len(buf) == 13  # проверка длины
    trits = []  # накопитель битов
    for b in buf:  # по байтам
        chunk = [0] * 5  # пять тритов байта, изначально нули
        v = b  # остаток от деления
        for i in range(4, -1, -1):  # от младшего трита к старшему
            chunk[i] = v % 3  # трит i = v mod 3
            v //= 3  # итог: v = ⌊v / 3⌋
        trits.extend(chunk)  # добавить триты в поток
    masks = []  # накопитель масок
    for m in range(16):  # по номерам масок
        val = 0  # значение маски, изначально 0
        for idx in range(4):  # по битам idx = 0..3
            val |= (trits[m * 4 + idx] & 1) << idx  # маска |= (трит & 1) << idx
        masks.append(val)  # добавить маску
    return masks  # вернуть 16 масок


def anf_transform(truth_table: Sequence[int]) -> list[int]:
    """Преобразование Жегалкина (Мёбиуса) таблицы истинности в коэффициенты АНФ: «бабочка».

    n = длина таблицы — степень 2. Коэффициент при мономе s равен XOR значений
    таблицы на всех индексах t ⊆ s (поразрядно). Сложность O(n·log n);
    преобразование — инволюция: повторное применение возвращает исходную
    таблицу."""
    """Преобразование Жегалкина (Мёбиуса) таблицы истинности: «бабочка», O(n·log n), инволюция."""
    n_tot = len(truth_table)  # длина таблицы
    if n_tot == 0 or (n_tot & (n_tot - 1)) != 0:  # пустая таблица или длина не степень 2 (n AND (n − 1) ≠ 0)
        raise ValueError("truth_table length must be a power of 2")  # ошибка: длина должна быть степенью 2
    coeffs = [int(v) & 1 for v in truth_table]  # рабочая копия коэффициентов, приведённая к битам 0/1
    step = 1  # размер бабочки, изначально 1
    while step < n_tot:  # пока шаг меньше длины
        for i in range(0, n_tot, step * 2):  # по блокам длины 2·step
            for j in range(step):  # по позициям внутри половины блока
                coeffs[i + step + j] ^= coeffs[i + j]  # верхняя половина XOR нижняя: c[i + step + j] ^= c[i + j]
        step <<= 1  # итог: step = 2·step
    return coeffs  # вернуть коэффициенты АНФ


def anf_eval(coeffs: Sequence[int], n_vars: int, xs: Sequence[int]) -> int:
    """Вычислить значение АНФ на входе xs по списку коэффициентов (1 на месте коэффициента 1).

    mask_x = Σ x_i·2^(n−1−i); результат = XOR 1 по всем s, у которых
    коэффициент c_s = 1 и моном s ⊆ mask_x (s AND mask_x = s)."""
    mask_x = 0  # битовая маска входа, изначально 0
    for i in range(n_vars):  # по переменным
        if int(xs[i]) & 1:  # младший бит x_i равен 1
            mask_x |= 1 << (n_vars - 1 - i)  # маска |= 2^(n−1−i)
    acc = 0  # аккумулятор, изначально 0
    for s, c in enumerate(coeffs):  # по мономам s и коэффициентам c
        if c and (s & mask_x) == s:  # коэффициент равен 1 и моном входит в вход
            acc ^= 1  # итог: acc ^= 1
    return acc  # вернуть значение АНФ


def anf_compose3(
    f_bool: Callable[[int, int], int], g_tt: Sequence[int], h_tt: Sequence[int]  # типы аргументов: внешняя функция и две таблицы истинности
) -> list[int]:  # возвращает список коэффициентов АНФ
    """АНФ суперпозиции h(x) = f(g(x), h(x)) по таблицам истинности g и h.

    g_tt, h_tt — таблицы истинности внутренних функций на одних и тех же
    входах; f_bool — внешняя булева функция двух аргументов. Строится таблица
    t_i = f(g_i, h_i) и преобразуется Жегалкина."""
    """АНФ суперпозиции (f ∘ (g, h))(x) = f(g(x), h(x)) по таблицам истинности g и h."""
    assert len(g_tt) == len(h_tt)  # таблицы g и h обязаны быть одинаковой длины
    tt = [int(f_bool(int(g_tt[i]) & 1, int(h_tt[i]) & 1)) & 1 for i in range(len(g_tt))]  # таблица t_i = f(g_i & 1, h_i & 1) & 1
    return anf_transform(tt)  # вернуть её коэффициенты АНФ


def anf_to_expr(coeffs: Sequence[int], n_vars: int) -> Expr:
    """Выразить АНФ деревом выражения Expr.

    Не более 4 мономов — явная сумма XOR произведений (bit_and) переменных;
    пустая АНФ — константа 0; единственный пустой моном — константа 1;
    иначе узел anf."""
    sparse = tuple((s, 1) for s, c in enumerate(coeffs) if c)  # разреженный список пар (моном s, коэффициент 1) для c_s = 1
    if len(sparse) == 0:  # нет мономов
        return Expr.const(0)  # константа 0
    if len(sparse) == 1 and sparse[0][0] == 0:  # единственный моном — пустой (константа)
        return Expr.const(1)  # константа 1
    if len(sparse) <= 4:  # не более 4 мономов — развернуть в операции
        mons = []  # список мономов-выражений
        for s, _ in sparse:  # по мономам
            if s == 0:  # пустой моном
                mons.append(Expr.const(1))  # записывается как константа 1
            else:  # иначе
                vs = [  # переменные, чьи биты входят в моном
                    Expr.var(i) for i in range(n_vars) if (s >> (n_vars - 1 - i)) & 1  # узел x_i для i, у которых (s >> (n−1−i)) & 1 = 1
                ]
                t = vs[0]  # произведение начинается с первой переменной
                for v in vs[1:]:  # по остальным переменным монома
                    t = Expr("bit_and", (t, v))  # t = t AND v — конъюнкция слева направо
                mons.append(t)  # добавить моном
        res = mons[0]  # сумма начинается с первого монома
        for m in mons[1:]:  # по остальным мономам
            res = Expr("bit_xor", (res, m))  # итог: res = res XOR m
        return res  # вернуть выражение
    return Expr("anf", (), (n_vars, sparse))  # иначе узел anf с числом переменных и разреженным списком мономов


_HADAMARD_CACHE: dict[int, np.ndarray] = {}  # кэш матриц Адамара по показателю k


def sylvester_hadamard(k: int) -> np.ndarray:
    """Ортонормированная матрица Адамара Сильвестра порядка 2^k, построенная детерминированно.

    H_0 = [1]; H_{k+1} = [[H_k, H_k], [H_k, −H_k]]; итог делится на √(2^k),
    так что H·H^T = E."""
    if k in _HADAMARD_CACHE:  # матрица уже в кэше
        return _HADAMARD_CACHE[k]  # взять из кэша
    H = np.array([[1.0]], dtype=np.float64)  # итог: H_0 = [1]
    for _ in range(k):  # k раз удвоить порядок
        H = np.block([[H, H], [H, -H]])  # итог: H = [[H, H], [H, −H]]
    H /= math.sqrt(2**k)  # нормировка: H = H / √(2^k)
    _HADAMARD_CACHE[k] = H  # запомнить в кэше
    return H  # вернуть матрицу


def block_hadamard_rotate(v: np.ndarray, blk_size: int = 8) -> tuple[np.ndarray, int]:
    """Поворот вектора блоками умножением на матрицу Адамара: (повёрнутый вектор, исходная длина).

    Вектор дополняется нулями до кратности blk_size, делится на блоки и
    каждый блок умножается на H справа: v' = v·H. Разворачивается обратно
    в одну строку."""
    H = sylvester_hadamard(int(round(math.log2(blk_size))))  # матрица Адамара порядка blk_size (blk_size = 2^k)
    v1 = np.asarray(v, dtype=np.float64).ravel()  # вектор float64 в одну строку
    n = v1.size  # исходная длина n
    nb = (n + blk_size - 1) // blk_size  # число блоков nb = ⌈n / blk_size⌉
    vp = np.pad(v1, (0, nb * blk_size - n)).reshape(nb, blk_size)  # дополнить нулями до nb·blk_size и разрезать на блоки (nb, blk_size)
    return (vp @ H).ravel(), n  # вернуть (блоки · H) в одну строку и исходную длину n


def block_hadamard_unrotate(
    vr: np.ndarray, orig_n: int, blk_size: int = 8  # повёрнутый вектор, исходная длина и размер блока
) -> np.ndarray:  # возвращает вектор исходной длины
    """Обратный блочный поворот Адамара: (повёрнутый вектор, длина) -> вектор.

    Каждый блок умножается на H^T (обратная ортогональная матрица); лишние
    элементы отбрасываются до исходной длины orig_n."""
    H = sylvester_hadamard(int(round(math.log2(blk_size))))  # матрица Адамара порядка blk_size
    nb = vr.size // blk_size  # число блоков nb = длина / blk_size
    return (vr.reshape(nb, blk_size) @ H.T).ravel()[:orig_n]  # вернуть (блоки · H^T) в строку, обрезанную до orig_n


_TRIT_POW = np.array([81, 27, 9, 3, 1], dtype=np.int64)  # веса разрядов пяти тритов по основанию 3: [3^4, 3^3, 3^2, 3, 1]


def quantize_5block_exact(blk: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Точное квантование блока из 5 чисел: (квантованный блок, (exp2, best, bval)).

    Шаг = 2^exp2 · s, где exp2 = clamp(⌈log2(m/1.5)⌉, −126, 126), m = max|blk|,
    а уровень s перебирается по DYADIC_LEVELS. Для каждого s: трит
    q_i = 1, если b_i/s > 0.5; −1, если < −0.5; иначе 0. Выбирается s с
    минимальной ошибкой Σ(b_i − q_i·s)². bval = Σ (q_i + 1)·3^(4−i) — код
    тритов; (0, 0, 121) обозначает нулевой блок."""
    m = float(np.max(np.abs(blk)))  # m = max|blk| — наибольший модуль в блоке
    if m < 1e-15:  # блок практически нулевой
        return np.zeros_like(blk), (0, 0, 121)  # нулевой блок: нули и код (0, 0, 121)
    exp2 = max(-126, min(126, math.ceil(math.log2(m / 1.5 + 1e-18))))  # итог: exp2 = clamp(⌈log2(m/1.5)⌉, −126, 126)
    sc = 2.0**exp2  # масштаб sc = 2^exp2
    bn = blk / sc  # нормированный блок bn = blk / sc
    s = DYADIC_LEVELS[:, None]  # столбец уровней s (255, 1)
    ratio = bn[None, :] / s  # итог: ratio[i, j] = bn_j / s_i
    q_int = np.where(ratio > 0.5, 1.0, np.where(ratio < -0.5, -1.0, 0.0))  # q = 1, если ratio > 0.5; −1, если ratio < −0.5; иначе 0
    errs = np.sum((bn[None, :] - q_int * s) ** 2, axis=1)  # errs_i = Σ_j (bn_j − q_ij·s_i)² — ошибка уровня s_i
    best = int(np.argmin(errs))  # номер лучшего уровня (минимальная ошибка)
    s_best = float(DYADIC_LEVELS[best])  # лучший уровень s_best
    trits = q_int[best].astype(np.int64)  # триты лучшего уровня как int64
    in_mask = np.abs(bn) <= 1.5 * s_best  # маска элементов с |bn| ≤ 1.5·s_best — внутри диапазона
    sat_tail = float(np.sum(np.where(~in_mask, (np.abs(bn) - s_best) ** 2, 0.0)))  # хвост насыщения: Σ (|bn| − s_best)² вне диапазона
    bound = int(np.sum(in_mask)) * (s_best / 2.0) ** 2 + sat_tail  # верхняя граница ошибки: n_внутри·(s_best/2)² + хвост
    assert float(errs[best]) <= bound + 1e-9  # проверка: ошибка не больше границы
    bval = int(np.dot(trits + 1, _TRIT_POW))  # bval = Σ_i (трит_i + 1)·3^(4−i)
    return (trits * (s_best * sc)).astype(np.float64), (exp2, best, bval)  # вернуть триты·s_best·sc и код (exp2, best, bval)


def pack_vector_dyadic5(vec: np.ndarray, stages: int = 2) -> tuple[bytes, np.ndarray]:
    """Упаковать вектор float64 в байты 5-тритовым диадическим кодеком без потерь: (байты, раскодированное).

    Заголовок: длина n (2 байта, младший вперёд) и число стадий stages.
    Остаток кодируется стадиями: каждая стадия квантует блоки текущего
    остатка по 5 элементов и вычитает квантованное. Блок 3 байта:
    (exp2 + 128, best, bval) или один байт 255 для нулевого блока.
    Второй элемент — результат unpack_vector_dyadic5 для проверки."""
    v = np.asarray(vec, dtype=np.float64).ravel()  # вектор float64 в одну строку
    n = v.size  # длина n
    nblk = (n + 4) // 5  # число блоков nblk = ⌈n / 5⌉
    vp = np.pad(v, (0, nblk * 5 - n))  # дополнить нулями до nblk·5
    buf = bytearray()  # накопитель байтов
    buf.extend(int(n).to_bytes(2, "little"))  # записать n двумя байтами, младший вперёд
    buf.append(int(stages))  # записать число стадий
    rem = vp.copy()  # остаток — копия вектора
    for _ in range(stages):  # по стадиям
        for b in range(nblk):  # по блокам
            blk = rem[b * 5 : (b + 1) * 5]  # текущий блок остатка из 5 элементов
            q_blk, (exp2, best, bval) = quantize_5block_exact(blk)  # квантовать блок
            if exp2 == 0 and best == 0 and bval == 121:  # нулевой блок (exp2 = 0, best = 0, bval = 121)
                buf.append(255)  # записать один байт 255
            else:  # иначе
                buf.append(exp2 + 128)  # байт показателя exp2 + 128 (сдвиг в неотрицательные)
                buf.append(best)  # байт номера уровня
                buf.append(bval)  # байт кода тритов
            rem[b * 5 : (b + 1) * 5] -= q_blk  # остаток блока уменьшить на квантованное: rem −= q_blk
    raw = bytes(buf)  # готовые байты
    return raw, unpack_vector_dyadic5(raw)  # вернуть байты и их раскодировку


def unpack_vector_dyadic5(buf: bytes) -> np.ndarray:
    """Раскодировать вектор, упакованный pack_vector_dyadic5: массив float64 длины n."""
    n = int.from_bytes(buf[:2], "little")  # длина n из первых двух байтов
    stages = buf[2]  # число стадий из третьего байта
    nblk = (n + 4) // 5  # число блоков nblk = ⌈n / 5⌉
    out = np.zeros(nblk * 5, dtype=np.float64)  # результат, изначально нули длины nblk·5
    pos = 3  # позиция чтения — сразу после заголовка
    for _ in range(stages):  # по стадиям
        for b in range(nblk):  # по блокам
            tag = buf[pos]  # байт-метка блока
            pos += 1  # сдвинуть позицию
            if tag == 255:  # метка 255 — нулевой блок
                continue  # пропустить блок
            exp2 = tag - 128  # exp2 = метка − 128
            best = buf[pos]  # номер уровня
            pos += 1  # сдвинуть позицию
            bval = buf[pos]  # код тритов
            pos += 1  # сдвинуть позицию
            sc = 2.0**exp2  # масштаб sc = 2^exp2
            lvl = float(DYADIC_LEVELS[best])  # уровень lvl = DYADIC_LEVELS[best]
            trits = [0] * 5  # пять тритов, изначально нули
            v = bval  # остаток кода
            for i in range(4, -1, -1):  # от младшего трита к старшему
                trits[i] = (v % 3) - 1  # трит i = (v mod 3) − 1
                v //= 3  # итог: v = ⌊v / 3⌋
            out[b * 5 : (b + 1) * 5] += np.array(trits, dtype=np.float64) * (lvl * sc)  # прибавить к блоку триты·lvl·sc
    return out[:n]  # вернуть первые n элементов


def pack_bitalloc_records(
    k: int, idx_sorted: np.ndarray, records: list[tuple[int, tuple[int, int, int]]]  # типы аргументов: k, индексы и список записей
) -> bytes:  # возвращает упакованные байты
    """Упаковать записи о распределении бит: (k, отсортированные индексы, записи) -> байты.

    k — число ненулевых координат; idx_sorted — их отсортированные индексы;
    records — (номер блока, (exp2, best, bval)). Формат: k; признак wide
    (1, если максимальный индекс ≥ 255 — индексы по 2 байта); индексы;
    число записей; для записи с номером ≥ числа блоков — номер блока;
    затем 255 для нулевого блока или (exp2 + 128, best, bval)."""
    n_blocks = (k + 4) // 5  # число блоков по 5 координат: ⌈k / 5⌉
    buf = bytearray()  # накопитель байтов
    buf.append(int(k))  # записать k
    max_idx = int(np.max(idx_sorted)) if k > 0 else 0  # наибольший индекс (0, если k = 0)
    wide = 1 if max_idx >= 255 else 0  # wide = 1, если наибольший индекс ≥ 255, иначе 0
    buf.append(wide)  # записать признак wide
    for ix in idx_sorted:  # по индексам в порядке возрастания
        if wide:  # широкий формат
            buf.extend(int(ix).to_bytes(2, "little"))  # записать индекс двумя байтами, младший вперёд
        else:  # иначе
            buf.append(int(ix))  # записать индекс одним байтом
    buf.append(len(records))  # записать число записей
    for ri, (blk_id, (exp2, best, bval)) in enumerate(records):  # по записям с номерами ri
        if ri >= n_blocks:  # номер записи больше числа блоков — номер блока не подразумевается
            buf.append(int(blk_id))  # записать номер блока
        if exp2 == 0 and best == 0 and bval == 121:  # нулевой блок
            buf.append(255)  # записать 255
        else:  # иначе
            buf.append(exp2 + 128)  # записать exp2 + 128
            buf.append(int(best))  # записать номер уровня
            buf.append(int(bval))  # записать код тритов
    return bytes(buf)  # вернуть байты


def unpack_bitalloc_records(buf: bytes, D: int) -> np.ndarray:
    """Раскодировать записи о распределении бит в вектор длины D (обратно pack_bitalloc_records)."""
    pos = 0  # позиция чтения
    k = buf[pos]  # k — число ненулевых координат
    pos += 1  # сдвинуть позицию
    wide = buf[pos]  # признак широкого формата
    pos += 1  # сдвинуть позицию
    idx_sorted = []  # список индексов
    for _ in range(k):  # читать k индексов
        if wide:  # широкий формат
            idx_sorted.append(int.from_bytes(buf[pos : pos + 2], "little"))  # индекс из двух байтов, младший вперёд
            pos += 2  # позиция + 2
        else:  # иначе
            idx_sorted.append(buf[pos])  # индекс из одного байта
            pos += 1  # позиция + 1
    n_rec = buf[pos]  # число записей
    pos += 1  # сдвинуть позицию
    n_blocks = (k + 4) // 5  # число блоков ⌈k / 5⌉
    wq_s = np.zeros(k, dtype=np.float64)  # сжатые квантованные значения по k координатам, изначально нули
    for ri in range(n_rec):  # по записям
        blk_id = ri if ri < n_blocks else buf[pos]  # номер блока: ri, если ri < числа блоков, иначе читается из байта
        if ri >= n_blocks:  # номер блока был записан явно
            pos += 1  # позиция + 1
        tag = buf[pos]  # метка блока
        pos += 1  # позиция + 1
        if tag == 255:  # метка 255 — нулевой блок
            continue  # пропустить
        exp2 = tag - 128  # exp2 = метка − 128
        best = buf[pos]  # номер уровня
        pos += 1  # позиция + 1
        bval = buf[pos]  # код тритов
        pos += 1  # позиция + 1
        sc = 2.0**exp2  # масштаб sc = 2^exp2
        lvl = float(DYADIC_LEVELS[best])  # уровень lvl = DYADIC_LEVELS[best]
        trits = [0] * 5  # пять тритов, изначально нули
        v = bval  # остаток кода
        for i in range(4, -1, -1):  # от младшего трита к старшему
            trits[i] = (v % 3) - 1  # трит i = (v mod 3) − 1
            v //= 3  # итог: v = ⌊v / 3⌋
        sl = slice(blk_id * 5, min((blk_id + 1) * 5, k))  # срез координат блока: [5·blk_id, min(5·(blk_id + 1), k))
        wq_s[sl] += np.array(trits[: sl.stop - sl.start], dtype=np.float64) * (lvl * sc)  # прибавить к срезу триты (обрезанные по длине среза)·lvl·sc
    out = np.zeros(D, dtype=np.float64)  # выходной вектор длины D, изначально нули
    if k > 0:  # есть ненулевые координаты
        out[np.array(idx_sorted, dtype=np.int64)] = wq_s  # развернуть сжатые значения по их индексам: out[idx_i] = wq_s_i
    return out  # вернуть вектор

def anf_support_vars(coeffs: Sequence[int], n_vars: int) -> tuple[int, ...]:
    """Номера переменных, входящих хотя бы в один моном полинома.

        coeffs: коэффициенты полинома, n_vars: число переменных. Номер монома s —
        битовая маска переменных: переменная i соответствует биту n_vars − 1 − i.
        used = OR всех s с c_s = 1; возврат — кортеж i, у которых бит i установлен."""
    used = 0  # объединение масок мономов
    for s, c in enumerate(coeffs):  # по мономам
        if c:  # коэффициент равен 1
            used |= s  # объединение used = used OR s
    return tuple(i for i in range(n_vars) if (used >> (n_vars - 1 - i)) & 1)  # переменные, чей бит в used равен 1

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'venn_atoms', 'venn_head_eval', 'pack_venn_16_flat', 'unpack_venn_16_flat', 'anf_transform', 'anf_support_vars', 'anf_eval',  # имя интерфейса
    'anf_compose3', 'anf_to_expr', 'sylvester_hadamard', 'block_hadamard_rotate', 'block_hadamard_unrotate',  # имя интерфейса
    'quantize_5block_exact', 'pack_vector_dyadic5', 'unpack_vector_dyadic5', 'pack_bitalloc_records', 'unpack_bitalloc_records',  # имя интерфейса
]  # конец перечня
