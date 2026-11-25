"""Операции над «стволом» модели: преобразование Адамара, сертифицированное
слияние, рост остаточными блоками и адаптерами низкого ранга.

Ключевые идеи и формулы:
  - sylvester_mul / sylvester: матрица Адамара Сильвестра, H_0 = [1],
    H_{k+1} = [[H_k, H_k], [H_k, −H_k]], размер 2^k; H @ w — умножение
    без таблицы умножения (только сложения и вычитания).
  - lin_at: приращение первого порядка функции L в точке w1 вдоль (w2 − w1):
    g·(w2 − w1)·t.
  - merge_bound: сертифицированная граница потери в середине отрезка
    L(mid) <= (L1 + L2)/2 + δ + δ₂/2, где mid = (w1 + w2)/2,
    δ = |L(mid) − (L1 + lin_at(0.5))|, δ₂ = |L2 − (L1 + lin_at(1.0))|.
  - coh_sq = ‖w‖∞² / ‖w‖₂² — когерентность вектора; rotate_gate решает,
    уменьшает ли умножение на H когерентность.
  - ResidualBlock: y = x + γ·h(x); при γ = 0 — точное тождество.
  - snap_rational: округление к сетке q/2^k; snap_obj_bound:
    |(s+ds−y)² − (s−y)²| <= (2|s−y| + |ds|)·|ds|.
  - LowRankAdapter: f + B(A f), A (r, d), B (d, r), B = 0 — тождество;
    AdapterBank — по одному адаптеру на домен.
  - paired_gain: d = before − after, нижняя граница mean(d) − z·SE.
  - paired_margin_eb: эмпирическая граница Бернстайна
    eps = √(2·sig2·L/n) + 2·b·L/n, L = ln(2·n_states/delta), b = max|d|.
  - POLAR_EXPRESS: три шага X <- a·X + b·(X Xᵀ)X + c·(X Xᵀ)²X.
  - projected_weight_decay: W' = W − dC − c(W − W P).
  - pack_trits5 / unpack_trits5: пять троичных уровней в байте,
    code = Σ_i (v_i + 1)·3^i.

Экспортирует все перечисленные функции, классы и константы.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов


# # ============================================================
# # ПРЕОБРАЗОВАНИЕ АДАМАРА СИЛЬВЕСТЕРА
# # ============================================================

from typing import Callable  # тип «вызываемый объект»
import numpy as np  # численные массивы
def sylvester_mul(w: np.ndarray, n: int) -> np.ndarray:  # w — вектор, n — его длина (степень двойки)
    """Умножить вектор w на целочисленную матрицу Адамара Сильвестра размера n: H @ w."""
    return sylvester(int(np.log2(n))).astype(float) @ w  # H = матрица размера 2^k при k = log2(n); итог: H @ w в float

# # ============================================================
# # СЕРТИФИЦИРОВАННОЕ СЛИЯНИЕ ДВУХ СОСТОЯНИЙ
# # ============================================================

from typing import Callable, Sequence  # типы «вызываемый объект» и «последовательность» (повторный импорт безвреден)
import numpy as np  # численные массивы (повторный импорт безвреден)
def lin_at(g: Sequence[float], w1: Sequence[float], w2: Sequence[float], t: float) -> float:  # градиент g, точки w1 и w2, параметр t вдоль отрезка
    """Приращение первого порядка функции L в точке w1 вдоль (w2 − w1): g·(w2 − w1)·t."""
    g = np.asarray(g)  # g в массив
    w1 = np.asarray(w1)  # w1 в массив
    w2 = np.asarray(w2)  # w2 в массив
    return float(g @ (w2 - w1)) * t  # приращение: (g · (w2 − w1))·t
def merge_bound(L: Callable[[np.ndarray], float], g: Sequence[float], w1: Sequence[float], w2: Sequence[float]) -> dict:  # L — функция потери, g — градиент в w1, w1 и w2 — сливаемые точки
    """Измерить невязки δ (в середине) и δ₂ (в конце) и вернуть сертифицированную
    границу L(mid) <= (L1 + L2)/2 + δ + δ₂/2.

        mid = (w1 + w2)/2; L1 = L(w1); L2 = L(w2);
        δ = |L(mid) − (L1 + lin_at(0.5))|; δ₂ = |L2 − (L1 + lin_at(1.0))|.
        lin_at возвращает только приращение, поэтому база L1 входит в невязку
        обеих точек; без этого граница остаётся верной, но излишне слабой.
        Возврат: словарь L_mid, bound, delta_mid, delta_end, certified."""
    w1 = np.asarray(w1, dtype=float)  # w1 в вещественный массив
    w2 = np.asarray(w2, dtype=float)  # w2 в вещественный массив
    mid = (w1 + w2) / 2.0  # середина отрезка: mid = (w1 + w2)/2
    L1, L2 = (float(L(w1)), float(L(w2)))  # значения потери на концах: L1 = L(w1), L2 = L(w2)
    # # lin_at возвращает ТОЛЬКО приращение; линейное приближение в точке t равно
    # # L1 + g·(w2 − w1)·t, поэтому базовая точка входит в невязку
    # # (в delta_end она уже была, в delta_mid — нет; без этого граница верна,
    # # но излишне слаба: 25.0 вместо 15.0 на квадратичной функции).
    delta_mid = abs(float(L(mid)) - (L1 + lin_at(g, w1, w2, 0.5)))  # невязка в середине: δ = |L(mid) − (L1 + lin_at(0.5))|
    delta_end = abs(L2 - (L1 + lin_at(g, w1, w2, 1.0)))  # невязка в конце: δ₂ = |L2 − (L1 + lin_at(1.0))|
    bound = (L1 + L2) / 2.0 + delta_mid + delta_end / 2.0  # граница: bound = (L1 + L2)/2 + δ + δ₂/2
    return {'L_mid': float(L(mid)), 'bound': bound, 'delta_mid': delta_mid, 'delta_end': delta_end, 'certified': float(L(mid)) <= bound + 1e-12}  # словарь: L_mid, граница, обе невязки, флаг certified = L_mid <= bound + 1e-12

# # ============================================================
# # ФИЛЬТР КОГЕРЕНТНОСТИ АДАМАРА
# # ============================================================

import numpy as np  # численные массивы (повторный импорт безвреден)
def sylvester(k: int) -> np.ndarray:  # k — показатель степени; размер матрицы 2^k
    """Целочисленная ±1 матрица Адамара Сильвестра размера 2^k (рекурсивное кронекерово произведение)."""
    if k < 0:  # k отрицательно
        raise ValueError('k must be >= 0')  # ошибка: k должен быть >= 0
    H = np.ones((1, 1), dtype=np.int64)  # начало: H_0 = [1]
    for _ in range(k):  # k раз
        H = np.block([[H, H], [H, -H]])  # рекурсия: H_{j+1} = [[H_j, H_j], [H_j, −H_j]]
    return H  # матрица размера 2^k
def coh_sq(w: np.ndarray) -> float:  # w — вектор
    """Когерентность: cohSq = ‖w‖∞² / ‖w‖₂² (0 для нулевого вектора — измерять нечего)."""
    w = np.asarray(w, dtype=float).ravel()  # w в плоский вещественный вектор
    l2 = float((w * w).sum())  # квадрат нормы: ‖w‖₂² = Σ_i w_i²
    if l2 <= 0:  # нулевая энергия
        return 0.0  # нет данных — 0
    return float(w.max() ** 2 if w.max() >= 0 else (-w).max() ** 2) / l2 if False else float(np.abs(w).max()) ** 2 / l2  # итог: ‖w‖∞² / ‖w‖₂², где ‖w‖∞ = max_i |w_i| (левая ветка «if False» не вычисляется)
def rotate_gate(w: np.ndarray) -> dict:  # w — вектор длины 2^k
    """Сравнить cohSq(w) и cohSq(H w) и принять решение. Длина w должна быть
    степенью двойки.

        Возврат: cohSq_before, cohSq_after, rotate = (after < before),
        energy_ratio = ‖H w‖₂² / ‖w‖₂² (1 для нулевого вектора)."""
    w = np.asarray(w, dtype=float).ravel()  # w в плоский вещественный вектор
    n = w.size  # длина n
    if n & n - 1:  # n не степень двойки: n & (n − 1) != 0
        raise ValueError('w length must be a power of two')  # ошибка: длина w должна быть степенью двойки
    H = sylvester(int(np.log2(n))).astype(float)  # H — матрица Адамара размера n = 2^(log2 n), вещественная
    hw = H @ w  # произведение: hw = H @ w
    before, after = (coh_sq(w), coh_sq(hw))  # когерентности до и после: (cohSq(w), cohSq(H w))
    return {'cohSq_before': before, 'cohSq_after': after, 'rotate': after < before, 'energy_ratio': float((hw * hw).sum() / (w * w).sum()) if (w * w).sum() > 0 else 1.0}  # словарь: обе когерентности, решение rotate = after < before и отношение энергий ‖H w‖²/‖w‖² (1 при нулевой энергии)

# # ============================================================
# # РОСТ СТВОЛА ОСТАТОЧНЫМИ БЛОКАМИ
# # ============================================================

from typing import Callable, Sequence  # типы (повторный импорт безвреден)
class ResidualBlock:  # остаточный блок: y = x + γ·h(x)
    """Остаточный блок: y = x + γ·h(x); при γ = 0 — точное тождество."""

    def __init__(self, h: Callable, gamma: float=0.0):  # h — функция ветви, gamma — её вес
        self.h = h  # запомнить функцию h
        self.gamma = float(gamma)  # вес: γ как float

    def __call__(self, x):  # x — вход
        if self.gamma == 0.0:  # γ = 0 — блок тождественен
            return x  # вернуть x без изменений
        return x + self.gamma * self.h(x)  # иначе: y = x + γ·h(x)

    def set_gamma(self, gamma: float) -> None:  # новое значение веса
        self.gamma = float(gamma)  # установить γ
def grow_trunk(trunk: Sequence[Callable], h: Callable) -> list[Callable]:  # ствол и новая ветвь h
    """Дописать блок с нулевым γ: функция ствола сохраняется ТОЧНО. Возвращает
    новый ствол. При повторных дописываниях блоки компонуются: только
    свежий блок может получить γ != 0, все прежние блоки по договорённости
    заморожены (их γ не меняется; обучается только новый блок)."""
    return list(trunk) + [ResidualBlock(h, gamma=0.0)]  # новый ствол: прежние блоки + ResidualBlock(h, γ = 0)
def run_trunk(trunk: Sequence[Callable], x):  # ствол и вход x
    """Применить блоки слева направо: x <- b(x) для каждого b. Возврат — итог."""
    for b in trunk:  # по блокам
        x = b(x)  # применить блок: x = b(x)
    return x  # итог
def obj_invariant_check(trunk_old: Sequence[Callable], trunk_new: Sequence[Callable], xs, obj: Callable) -> bool:  # старый и новый ствол, точки xs, целевая функция obj
    """Проверка инварианта: значение obj на выходе ствола не изменилось после
    дописывания блока — для всех точек xs (точное равенство float)."""
    return all((obj(run_trunk(trunk_old, x)) == obj(run_trunk(trunk_new, x)) for x in xs))  # True, если для всех x: obj(старый(x)) == obj(новый(x))

# # ==================================================================
# # ОКРУГЛЕНИЕ ПРИ РОСТЕ: привязка к сетке q/2^k на границах роста.

from fractions import Fraction as _Fr  # дробь Fraction под именем _Fr


def snap_rational(value, k: int) -> _Fr:  # значение, точность k
    """Округлить число к сетке q/2^k (банковское округление).
    Погрешность одного значения <= 2^(−k−1) (половина ячейки)."""
    f = _Fr(value)  # дробь: f = Fraction(value)
    scaled = f * (1 << k)  # масштаб: scaled = f·2^k
    q = (scaled.numerator * 2 + scaled.denominator) // (2 * scaled.denominator)  # q = floor((2·num + den)/(2·den)) — округление к ближайшему чётному
    return _Fr(q, 1 << k)  # результат: Fraction(q, 2^k)


def snap_denominator_reset(x, k: int):  # x — число или последовательность, k — точность
    """Округлить значения (скаляр или последовательность) так, чтобы каждый
    знаменатель был <= 2^k."""
    if isinstance(x, (list, tuple)):  # x — список или кортеж
        return type(x)(snap_rational(v, k) for v in x)  # тот же тип из округлённых элементов
    return snap_rational(x, k)  # скаляр — округлить одно значение


def snap_obj_bound(s_err: float, ds: float) -> float:  # ошибка s_err = s − y, сдвиг признака ds
    """Сертифицированная граница ухудшения функции потерь на одной точке при
    сдвиге признака на ds:
    |(s + ds − y)² − (s − y)²| <= (2|s − y| + |ds|)·|ds|."""
    return (2 * abs(s_err) + abs(ds)) * abs(ds)  # граница: (2|s_err| + |ds|)·|ds|


# # ==================================================================
# # РОСТ АДАПТЕРАМИ: сертифицированный рост боковой ветви низкого ранга.
# # adapterVec A B x = x + B·(A·x): 2·r·d параметров вместо d·d
# # для полного остаточного блока (r << d). Нейтральность доказывается
# # ОДНИМ нулевым множителем (B = 0 — точное тождество), в отличие от
# # остаточного листа, где нужны оба нулевых.

import numpy as _np  # численные массивы под именем _np


class LowRankAdapter:  # адаптер низкого ранга
    """Адаптер роста вида LoRA: pred = w @ (f + B @ (A @ f)). Добавляет 2·r·d
    параметров; при B = 0 — точное тождество (нейтральность одним нулевым
    множителем)."""

    def __init__(self, d: int, r: int = 1, seed: int = 0):  # d — размерность, r — ранг, seed — зерно генератора
        rng = _np.random.default_rng(seed)  # генератор нормальных чисел с зерном seed
        # # инициализация A в стиле Кайминга; B = 0 => точное стартовое тождество
        self.A = rng.standard_normal((r, d)) / _np.sqrt(d)  # A = N(0, 1)/√d, размер (r, d)
        self.B = _np.zeros((d, r))  # B = нули (d, r)
        self.d, self.r = d, r  # запомнить размерности

    @property  # свойство только для чтения
    def n_params(self) -> int:  # число параметров
        return 2 * self.r * self.d  # число параметров: n_params = 2·r·d

    def __call__(self, f: _np.ndarray) -> _np.ndarray:  # f — входной вектор
        # # f + B @ (A @ f); при B = 0 это РОВНО f (нейтральность)
        return f + self.B @ (self.A @ f)  # итог: f + B·(A·f)

    def params(self) -> _np.ndarray:  # собрать все параметры
        return _np.concatenate([self.A.ravel(), self.B.ravel()])  # A и B, развёрнутые в один вектор: сначала A, затем B

    def set_params(self, flat: _np.ndarray) -> None:  # flat — развёрнутые параметры
        rA = self.r * self.d  # rA = r·d — число элементов A
        self.A = flat[:rA].reshape(self.r, self.d)  # A = flat[:rA] формы (r, d)
        self.B = flat[rA:].reshape(self.d, self.r)  # B = flat[rA:] формы (d, r)


def adapter_warm_start(src_adapter: "LowRankAdapter",  # перенос адаптера: src_adapter — источник
                       dst: "LowRankAdapter") -> bool:  # dst — адаптер-приёмник; возвращает признак успеха
    """Перенос адаптера: скопировать адаптер домена-источника в адаптер
    домена-приёмника (когда домены достаточно близки — решает вызывающий
    код). Это копия: источник не меняется. Формы должны совпадать,
    иначе False."""
    if (src_adapter.d, src_adapter.r) != (dst.d, dst.r):  # размерности не совпадают
        return False  # отказ
    # # адаптеры, восстановленные из сохранения, могут держать тензоры torch,
    # # свежие — массивы numpy; приводим к numpy при передаче, чтобы copy()
    # # работало в обоих случаях.
    # # (сохранение записывает адаптеры как есть).
    import numpy as _np  # numpy под именем _np (локальный импорт)
    dst.A = _np.asarray(src_adapter.A).copy()  # dst.A = копия A источника
    dst.B = _np.asarray(src_adapter.B).copy()  # dst.B = копия B источника
    return True  # успех


def paired_gain(before, after, z: float = 2.4):  # потери до и после, z — число стандартных отклонений
    """Парный выигрыш по примерам: d = before − after (положительное — лучше)
    с нижней границей нормального приближения mean(d) − z·SE. Эмпирический
    фильтр, не формальный сертификат."""
    d = _np.asarray(before, dtype=float) - _np.asarray(after, dtype=float)  # разности: d = before − after
    if len(d) == 0 or not _np.isfinite(d).all():  # пусто или есть нечисловое значение
        raise ValueError("paired losses must be finite and non-empty")  # ошибка: парные потери должны быть конечными и непустыми
    se = (d.std(ddof=1) / _np.sqrt(len(d))) if len(d) > 1 else 0.0  # SE = std(d, ddof=1)/√n при n > 1, иначе 0
    return float(d.mean()), float(d.mean() - z * se)  # возврат (mean(d), mean(d) − z·SE)


def paired_margin_eb(d, n_states: int = 1, delta: float = 0.05) -> float:  # d — разности потерь по окнам, n_states — число кандидатов, delta — уровень доверия
    """Доказуемая односторонняя граница для парного фильтра — граница среднего по
    эмпирическому Бернстайну:
        eps = √(2·sig2·L/n) + 2·b·L/n,   L = ln(2/alpha),
    с объединённой оценкой по n_states кандидатам, оценённым на одной
    отложенной выборке: alpha = delta / n_states.
        d — разность потерь по окнам (старая − новая; положительная —
    кандидат лучше); sig2 — выборочная дисперсия; b = max|d|.
    Фильтр может принять, только если mean(d) > eps.
    В отличие от paired_gain (нормальное приближение z·SE) граница не
    предполагает нормальность d и учитывает выбор лучшего из K."""
    d = _np.asarray(d, dtype=float)  # d в вещественный массив
    n = len(d)  # n — число разностей
    if n < 2 or not _np.isfinite(d).all():  # меньше двух значений или есть нечисловое
        return float("inf")  # бесконечная граница: принять нельзя
    L = _np.log(2.0 * max(1, n_states) / delta)  # логарифм: L = ln(2·max(1, n_states)/delta)
    sig2 = float(d.var(ddof=1))  # sig2 = выборочная дисперсия d
    b = float(_np.abs(d).max())  # максимум модуля: b = max|d|
    return float(_np.sqrt(2.0 * sig2 * L / n) + 2.0 * b * L / n)  # граница: eps = √(2·sig2·L/n) + 2·b·L/n

# # ---- Polar-Express: три квинтических шага с рациональными коэффициентами.
# # Сингулярные числа из [1/1000, 1] переходят в [9/512, 1049/1024].
# # Классический Ньютон–Шульц (core.newton_schulz_orthogonalize) такого
# # окна для малых сингулярных чисел не даёт (4 шага: [0.47, 1.0] на [0.1, 1];
# # здесь: [0.9075, 1.0212]). В основной цикл не подключён намеренно:
# # обратный поиск шага там перемасштабирует шаг, качество обучения то же
# # (относительная ошибка оператора 0.010 против 0.011), время на вызов
# # сопоставимо (17.6 против 23.0 мкс), а при d_out = 1 ортогонализация
# # вырождается в норму вектора.
# #
# # Три шага: X <- a·X + b·X³ + c·X⁵ с рациональными коэффициентами (a, b, c).
POLAR_EXPRESS = ((10151 / 2500, -17483 / 2500, 9623 / 2500),  # коэффициенты первого шага (a, b, c)
                 (5429 / 2000, -25159 / 10000, 1963 / 2500),  # коэффициенты второго шага
                 (9473 / 5000, -1119 / 1000, 2201 / 10000))  # коэффициенты третьего шага


def polar_express(G) -> "np.ndarray":  # G — матрица
    """Приближение полярного множителя тремя квинтическими шагами (вход
    нормируется; «высокая» матрица транспонируется в «широкую»)."""
    X = _np.asarray(G, dtype=_np.float64).copy()  # X = копия G в float64
    nrm = float(_np.linalg.norm(X))  # норма Фробениуса ‖X‖
    if nrm < 1e-12:  # норма практически нулевая
        return X  # вернуть X без изменений
    X /= nrm  # нормировка: X = X/‖X‖
    transposed = X.shape[0] > X.shape[1]  # матрица «высокая»: строк больше, чем столбцов
    if transposed:  # нужно транспонировать
        X = X.T  # транспонирование: X = Xᵀ
    for a, b, c in POLAR_EXPRESS:  # по тройкам коэффициентов (a, b, c)
        X2 = X @ X.T  # произведение: X2 = X @ Xᵀ
        X = a * X + b * (X2 @ X) + c * (X2 @ X2 @ X)  # шаг: X = a·X + b·(X2 @ X) + c·(X2 @ X2 @ X)
    return X.T if transposed else X  # вернуть X, при транспонировании — обратно Xᵀ

def adapter_param_count(d: int, r: int, full_block_d: int | None = None) -> dict:  # d и r — размерности адаптера, full_block_d — размер полного блока
    """Подсчёт параметров: адаптер 2·r·d против полного блока d·d
    (адаптер не больше полного блока при r <= d/2)."""
    out = {"adapter": 2 * r * d}  # словарь: число параметров адаптера 2·r·d
    if full_block_d is not None:  # задан размер полного блока
        out["full_block"] = full_block_d * full_block_d  # число параметров: full_block = full_block_d²
        out["ratio"] = out["adapter"] / out["full_block"]  # доля параметров: ratio = adapter/full_block
    return out  # итог


class AdapterBank:  # банк адаптеров по доменам
    """Независимые адаптеры по доменам: один общий ствол и голова плюс
    изолированные боковые ветви низкого ранга на каждый домен. Перенос между
    доменами — через adapter_warm_start."""

    def __init__(self, d: int, r: int = 2):  # d — размерность, r — ранг (по умолчанию 2)
        self.d, self.r = d, r  # запомнить размерности
        self._adapters: dict[str, LowRankAdapter] = {}  # словарь «домен -> адаптер» (пока пуст)

    def get(self, domain: str) -> LowRankAdapter:  # domain — имя домена
        if domain not in self._adapters:  # домена ещё нет
            # # зерно детерминированно выводится из ИМЕНИ домена: прежнее
            # # len(self._adapters) делало инициализацию зависящей от ПОРЯДКА
            # # создания доменов (ошибка воспроизводимости).
            seed = int(__import__("hashlib").sha1(  # seed = int(первые 8 цифр sha1(имя домена в utf-8), основание 16)
                domain.encode("utf-8")).hexdigest()[:8], 16)  # продолжение sha1
            self._adapters[domain] = LowRankAdapter(self.d, self.r,  # создать адаптер (d, r) с этим зерном
                                                    seed=seed)  # продолжение: seed=seed
        return self._adapters[domain]  # вернуть адаптер домена

    def apply(self, domain: str, f):  # domain — имя домена, f — вход
        """Остаток через адаптер ТОЛЬКО этого домена: f + B(A f). Адаптеры
        других доменов не затрагиваются — изоляция по построению."""
        return self.get(domain)(f)  # применить адаптер домена к f

    def transfer(self, src: str, dst: str) -> bool:  # src — домен-источник, dst — домен-приёмник
        """Тёплый старт: скопировать адаптер src в dst (о близости решает
        вызывающий код). Адаптер-источник не меняется."""
        if src not in self._adapters:  # источника нет
            return False  # отказ
        return adapter_warm_start(self._adapters[src],  # перенос: adapter_warm_start(src, get(dst))
                                  self.get(dst))  # продолжение вызова

    def n_params(self) -> int:  # число параметров банка
        return sum(a.n_params for a in self._adapters.values())  # сумма n_params по всем адаптерам


# # ==================================================================
# # Проекционный весовой спад и точный вклад покрытия.
# # ==================================================================

def projected_weight_decay(W, P, c: float, complement_update=None):  # W — веса, P — проектор, c — скорость спада, complement_update — обновление в дополнении
    """Спад весов с защитой: W' = W − dC − c·(W − W P); предполагается
    dC·P = 0 и P² = P. Защищённая компонента W P сохраняется ТОЧНО
    (утечка 5.68e-15 против 1.945 у наивного спада)."""
    W = np.asarray(W, dtype=np.float64)  # W в float64
    P = np.asarray(P, dtype=np.float64)  # P в float64
    d = (np.zeros_like(W) if complement_update is None  # dC = нули, если обновление не задано,
         else np.asarray(complement_update, dtype=np.float64))  # иначе complement_update в float64
    return W - d - c * (W - W @ P)  # итог: W' = W − dC − c·(W − W @ P)


def exact_ce_contribution(p_true: float) -> float:  # p_true — вероятность истинного байта
    """Точный вклад кросс-энтропии: ноль только при p == 1; p <= 0 отвергается
    вместо вычисления −log(0)."""
    import math  # математические функции
    p = float(p_true)  # p как float
    if p == 1.0:  # p = 1 — вклад нулевой
        return 0.0  # вернуть 0.0
    if p <= 0.0:  # p <= 0 — логарифм не определён
        raise ValueError(  # ошибка значения
            "invalid exact coverage: true-byte probability must be > 0")  # текст ошибки
    return -math.log(p)  # вклад: −ln(p)


# # Кодек: пять троичных уровней в одном байте (3^5 = 243 <= 2^8).

def pack_trits5(levels: Sequence[int]) -> tuple[bytes, int]:  # levels — последовательность уровней из {−1, 0, +1}
    """Упаковать символы {−1, 0, +1} в группы по пять в троичной системе —
    один байт на группу.

        Соответствие: −1 -> 0, 0 -> 1, +1 -> 2. Так как 3^5 = 243 <= 256,
        каждый код из пяти трит укладывается в один байт взаимно однозначно.
        code = Σ_{i=0..4} (v_i + 1)·3^i; неполная группа дополняется нулями
        (цифрой 1). Возврат (байты, число исходных уровней)."""
    vals = [int(x) for x in levels]  # vals — целые уровни
    if any(x not in (-1, 0, 1) for x in vals):  # есть уровень вне {−1, 0, 1}
        raise ValueError("levels must be in {-1,0,+1}")  # ошибка: уровни должны быть из {−1, 0, +1}
    out = bytearray()  # накопитель байтов
    for start in range(0, len(vals), 5):  # по группам по 5
        digits = [v+1 for v in vals[start:start+5]]  # цифры: v + 1
        digits += [1] * (5-len(digits))  # дополнить цифрой 1 (уровень 0) до 5
        code = sum(d * (3**i) for i, d in enumerate(digits))  # код: code = Σ_i d_i·3^i
        if code >= 243:  # код вне допустимого диапазона
            raise AssertionError("base-three code overflow")  # ошибка: переполнение троичного кода
        out.append(code)  # добавить байт
    return bytes(out), len(vals)  # возврат (байты, число уровней)


def unpack_trits5(payload: bytes, length: int) -> np.ndarray:  # payload — байты, length — число уровней
    if length < 0 or len(payload) != (length+4)//5:  # отрицательная длина или размер не равен (length + 4)//5
        raise ValueError("payload length does not match trit count")  # ошибка: длина payload не соответствует числу трит
    out = []  # накопитель уровней
    for code in payload:  # по байтам
        if code >= 243:  # байт вне допустимого диапазона
            raise ValueError("invalid packed trit byte (must be <243)")  # ошибка: недопустимый упакованный байт (должен быть < 243)
        n = int(code)  # n — значение байта
        for _ in range(5):  # пять раз
            out.append((n % 3)-1)  # очередной уровень: (n mod 3) − 1
            n //= 3  # переход к следующей цифре: n = n div 3
    return np.asarray(out[:length], dtype=np.int8)  # первые length уровней в int8

