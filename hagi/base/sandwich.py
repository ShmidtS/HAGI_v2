"""Сертифицированная «сэндвич»-оценка softmax: точные рациональные границы вероятностей.

Каждая формула здесь — прямой образ на Python доказанного утверждения о рациональном
softmax, монотонности двусторонних границ и их сужении:

  lo  = (1 + sv/nm) ** nm          — нижняя граница exp(sv) (следствие неравенства Бернулли)
  hi  = (1 - sv/nm) ** (-nm)       — верхняя граница exp(sv)
  sv  = z - max(z)                 — логиты со сдвигом на максимум, sv ∈ [−R, 0]

Обе границы монотонны по nm: с ростом знаменателя обе «стены» прижимаются к exp(sv),
поэтому цикл с удвоением nm рано или поздно разделяет любой действительно различимый
лидер и сертифицирует argmax.

Охрана диапазона: sv ≤ 0 и должно выполняться −sv < nm (иначе верхнее основание
1 − sv/nm выходит из (0, 1]); при нарушении функция честно возвращает None = HOLD.

Для точной рациональной арифметики подавайте Fraction; float годится только для
ориентировочных проверок.

Экспортирует sandwich_terms (границы lo/hi), sandwich_mid (середина интервала),
certified_argmax (сертифицированный argmax) и band_drift_check (проверка того, что
интервал всё ещё накрывает настоящий softmax)."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

from fractions import Fraction  # точные рациональные числа

DEFAULT_NM = 4  # параметр nm по умолчанию
EVAL_NM_MAX = 1024  # наибольшее nm в цикле удвоения


def sandwich_terms(z, nm: int = DEFAULT_NM):
    """Нижняя и верхняя границы exp(sv) для логитов z со сдвигом на максимум.

    M = max(z); sv_i = z_i − M; lo_i = (1 + sv_i/nm)^nm; hi_i = (1 − sv_i/nm)^(−nm).
    Возвращает списки (lo, hi) либо None = HOLD, когда охрана диапазона −min(sv) ≥ nm
    не выполнена (при таком nm сертификата не существует)."""
    from fractions import Fraction as F  # конструктор рационального числа (локальный импорт)
    M = max(z)  # наибольший логит M
    s = [v - M for v in z]  # сдвинутые логиты s_i = z_i − M
    if -min(s) >= nm:  # охрана диапазона: −min(s) ≥ nm — верхнее основание выйдет из (0, 1]
        return None  # сертификата нет — HOLD
    lo = [(F(1) + sv / nm) ** nm for sv in s]  # нижние границы lo_i = (1 + sv_i/nm)^nm
    hi = [(F(1) - sv / nm) ** (-nm) for sv in s]  # верхние границы hi_i = (1 − sv_i/nm)^(−nm)
    return lo, hi  # вернуть оба списка


def sandwich_mid(z, nm: int = DEFAULT_NM):
    """Середина интервала как пробная оценка: среднее нормированных границ.

    slo = Σ lo_i, shi = Σ hi_i; mid_i = (hi_i/shi + lo_i/slo)/2. Точность — половина
    ширины интервала. None = HOLD (охрана диапазона не выполнена)."""
    t = sandwich_terms(z, nm)  # границы lo/hi или None
    if t is None:  # охрана диапазона не выполнена
        return None  # отказ: HOLD
    lo, hi = t  # нижние и верхние границы
    slo, shi = sum(lo, type(lo[0])(0)), sum(hi, type(hi[0])(0))  # суммы границ slo = Σ lo_i, shi = Σ hi_i (нулевое значение того же типа)
    return [(hi_v / shi + lo_v / slo) / 2  # середина mid_i = (hi_i/shi + lo_i/slo) / 2
            for lo_v, hi_v in zip(lo, hi)]  # по парам (lo_i, hi_i)


def certified_argmax(z, nm: int = DEFAULT_NM, nm_max: int = EVAL_NM_MAX):
    """Сертифицированный argmax: лидер по рациональной середине, с удвоением nm до разделения.

    Границы вероятностей (знаменатель берёт ПРОТИВОПОЛОЖНЫЕ крайности остальных):
      L_i = l_i / (l_i + Σ_{j≠i} u_j)
      U_i = u_i / (u_i + Σ_{j≠i} l_j)
    Кандидат a — номер наибольшей середины; он сертифицирован, если L_a > U_j для всех
    j ≠ a. Иначе nm удваивается (интервал только сужается), пока nm ≤ nm_max.
    Ничья (например, [3, 3, 0]) сертифицирована быть не может.
    Возвращает (a, nm) либо (None, None) = HOLD."""
    nm = nm  # текущий параметр nm (изменяется в цикле)
    while nm <= nm_max:  # пока nm не превысил предел
        mid = sandwich_mid(z, nm)  # середины интервалов при текущем nm
        if mid is None:  # охрана диапазона −min(s) < nm не выполнена при текущем nm
            # Это НЕ значит, что сертификат невозможен: с ростом nm интервал сужается,
            # а сама охрана (условие −min(s) < nm) ослабевает. Нужно удвоить nm и
            # повторить, а не удерживать различимые логиты вроде [10, 0, 0].
            nm *= 2  # удвоить: nm = 2·nm
            continue  # повторить с большим nm
        # Корректные границы вероятностей: lo_i/Σ lo_j нижней границей НЕ является —
        # знаменатель обязан брать противоположные крайности остальных членов:
        #   нижняя L_i = l_i / (l_i + Σ_{j≠i} u_j)
        #   верхняя U_i = u_i / (u_i + Σ_{j≠i} l_j)
        # Теперь ничья не может быть ложно сертифицирована.
        t = sandwich_terms(z, nm)  # границы lo/hi (mid уже не None, поэтому и здесь не None)
        if t is None:  # охрана диапазона не выполнена
            nm *= 2  # удвоить: nm = 2·nm
            continue  # повторить
        lo, hi = t  # нижние и верхние границы
        n = len(z)  # число классов
        order = sorted(range(n), key=lambda i: mid[i], reverse=True)  # номера классов по убыванию середины
        L = [lo[i] / (lo[i] + sum(hi[j] for j in range(n) if j != i))  # нижние границы L_i = l_i / (l_i + Σ_{j≠i} u_j)
             for i in range(n)]  # по всем классам i
        U = [hi[i] / (hi[i] + sum(lo[j] for j in range(n) if j != i))  # верхние границы U_i = u_i / (u_i + Σ_{j≠i} l_j)
             for i in range(n)]  # по всем классам i
        a = order[0]  # кандидат — класс с наибольшей серединой
        if all(L[a] > U[j] for j in range(n) if j != a):  # нижняя граница кандидата выше верхних границ всех остальных
            return a, nm  # сертифицировано: класс a при данном nm
        nm *= 2  # интервалы ещё не разделились — nm = 2·nm
    return None, None  # предел nm исчерпан — HOLD


def band_drift_check(z, nm: int = DEFAULT_NM) -> dict:
    """Детектор устаревания интервала между сертифицированными границами и настоящим softmax.

    Интервал [L_i, U_i] по построению накрывает настоящие вероятности softmax. Если
    вычисленная точка softmax выходит из него, интервал устарел: логиты не из того
    диапазона или накопилась погрешность float. Возвращает словарь ok (нет выходов),
    escapes (номера вышедших классов), nm и ширины интервалов U_i − L_i."""
    from fractions import Fraction as F  # конструктор рационального числа (локальный импорт)
    import math  # экспонента для softmax (локальный импорт)
    t = sandwich_terms(z, nm)  # границы lo/hi или None
    if t is None:  # охрана диапазона не выполнена
        if nm * 2 <= EVAL_NM_MAX:  # удвоенное nm ещё в пределах
            return band_drift_check(z, nm * 2)  # повторить проверку с nm = 2·nm
        return {"ok": False, "reason": "range guard failed at nm_max"}  # отказ: охрана диапазона не выполнена на наибольшем nm
    lo, hi = t  # нижние и верхние границы
    n = len(z)  # число классов
    L = [lo[i] / (lo[i] + sum(hi[j] for j in range(n) if j != i))  # нижние границы L_i = l_i / (l_i + Σ_{j≠i} u_j)
         for i in range(n)]  # по всем классам
    U = [hi[i] / (hi[i] + sum(lo[j] for j in range(n) if j != i))  # верхние границы U_i = u_i / (u_i + Σ_{j≠i} l_j)
         for i in range(n)]  # по всем классам
    mx = max(z)  # наибольший логит для сдвига
    probs = [math.exp(float(v - mx)) for v in z]  # экспоненты e_i = exp(z_i − mx)
    s = sum(probs)  # знаменатель s = Σ e_i
    probs = [p / s for p in probs]  # настоящие вероятности p_i = e_i / Σ_j e_j
    escapes = [i for i in range(n)  # классы, для которых НЕ верно L_i ≤ p_i ≤ U_i
               if not (float(L[i]) <= probs[i] <= float(U[i]))]  # условие накрытия
    # При выходах на текущем nm — удвоить, как в certified_argmax (интервал сужается).
    if escapes and nm * 2 <= EVAL_NM_MAX:  # есть выходы и удвоенное nm в пределах
        return band_drift_check(z, nm * 2)  # повторить с nm = 2·nm
    return {"ok": not escapes, "escapes": escapes,  # итог: ok = (выходов нет), номера выходов
            "nm": nm, "widths": [float(U[i] - L[i]) for i in range(n)]}  # использованное nm и ширины интервалов U_i − L_i
