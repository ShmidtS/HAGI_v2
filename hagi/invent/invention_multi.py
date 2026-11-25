"""Синтез многобайтовых законов произвольной арности поверх окна контекста.
Ключевые идеи:
  - Закон = характеристическая функция phi: кортеж k байтов контекста -> число c
    от 0 до 255, затем базис y = f(c). Семейства характеристик:
      aff   — c = (a1·x1 + … + ak·xk + b) mod 256, коэффициенты из −3..3;
      diff  — c = (x[i] − x[j]) mod 256;
      xorm  — c = XOR_i (m_i & x_i) по битовой маске m;
      xfull — c = XOR_{i из I} x_i по подмножеству индексов I;
      pos   — c = x[i].
  - Поток разделён на слои по позиции: слой (P, s) — пары с pos mod P = s.
    В слое для каждого значения c берётся мажоритарный ответ y (доля >= 0.6,
    не менее 2 голосов). Базис POLY_LIN (y = k·c + c0) или OUT_PERIOD
    (y = (c + o) mod p) подгоняется по 60% значений и проверяется на остальных:
    доля совпадений >= 0.9 и выше доли самого частого y хотя бы на min_edge.
  - Межслойная проверка: в других фазах того же периода доля совпадений >= 0.7.
  - Кандидаты сравниваются по ключу (support, −total_bits); лучший
    возвращается со статусом PARTIAL_LAW, иначе {"status": "NO_LAW"}.

Экспортирует: _affine_k, _xor_mask, _diff_pair, _position, enumerate_features,
_vfeat, _kind, _majVotes, _mode_share, invent_multi.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import random  # генератор случайных чисел

import numpy as np  # численные массивы
from typing import Callable  # тип «вызываемый объект»

from hagi.invent.invention import BASES, ConstructBlueprint, _payload_bits  # меню базисов и чертёж «характеристика + базис»

# ---- характеристические функции от кортежей контекста ----------------------------


def _affine_k(coefs: tuple[int, ...], bias: int):  # коэффициенты и смещение
    """Вернуть функцию c = (a1·x1 + … + ak·xk + b) mod 256."""
    def f(x: tuple[int, ...]) -> int:  # x — кортеж байтов контекста
        s = bias  # накопитель: s = b
        for a, v in zip(coefs, x):  # по парам (коэффициент a_i, байт x_i)
            s += a * v  # накопитель: s = s + a_i·x_i
        return s % 256  # значение: c = s mod 256
    return f  # готовая функция


def _xor_mask(mask: tuple[int, ...]):  # битовая маска на каждый байт контекста
    """Вернуть функцию c = XOR_i (m_i & x_i) по битовой маске mask."""
    def f(x: tuple[int, ...]) -> int:  # x — кортеж байтов контекста
        s = 0  # накопитель
        for m, v in zip(mask, x):  # по парам (маска m_i, байт x_i)
            s ^= (m & v)  # накопитель: s = s XOR (m_i & x_i)
        return s  # значение: c = XOR_i (m_i & x_i)
    return f  # готовая функция


def _diff_pair(i: int, j: int):  # номера двух позиций в контексте
    """Вернуть функцию c = (x[i] − x[j]) mod 256 — закон относительной позиции."""
    def f(x: tuple[int, ...]) -> int:  # x — кортеж байтов контекста
        return (x[i] - x[j]) % 256  # значение: c = (x[i] − x[j]) mod 256
    return f  # готовая функция


def _position(i: int):  # номер позиции в контексте
    """Вернуть функцию c = x[i] mod 256 — значение одной позиции контекста."""
    def f(x: tuple[int, ...]) -> int:  # x — кортеж байтов контекста
        return x[i] % 256  # значение: c = x[i] mod 256
    return f  # готовая функция


def enumerate_features(k: int, max_features: int = 4000):  # арность контекста, предел числа характеристик
    """Перечислить комбинаторное семейство характеристик для контекста из k байтов.

        Порядок: сначала аффинные — все наборы коэффициентов из −3..3 с хотя бы
        одним ненулевым, отсортированные по (число ненулевых, сумма |a_i|), с
        каждым идут смещения b из {0, 1, 3, 7, 97}; если их больше max_features —
        стоп. Затем разности всех пар позиций, XOR с битовыми масками (2^k − 1
        непустых), XOR целых байтов по непустым подмножествам индексов и
        отдельные позиции. Возвращаются первые max_features.
        Элемент: (имя, параметр, функция)."""
    feats: list[tuple[str, object, Callable]] = []  # накопитель (имя, параметр, функция)
    from itertools import product as _prod  # декартово произведение
    # порядок по разреженности: законы с малым числом ненулевых коэффициентов
    # вероятнее и дешевле в проверке
    coef_sets = [c for c in _prod((-3, -2, -1, 0, 1, 2, 3), repeat=k)  # все наборы коэффициентов из −3..3
                 if any(v != 0 for v in c)]  # только с хотя бы одним ненулевым
    coef_sets.sort(key=lambda c: (sum(1 for v in c if v != 0),  # сортировка по числу ненулевых коэффициентов
                                  sum(abs(v) for v in c)))  # затем по сумме модулей Σ|a_i|
    for coefs in coef_sets:  # по наборам коэффициентов
        for bias in (0, 1, 3, 7, 97):  # по смещениям b
            feats.append((f"aff{coefs}+{bias}",  # имя «aff(коэффициенты)+смещение»
                          (coefs, bias),  # параметр: (коэффициенты, смещение)
                          _affine_k(coefs, bias)))  # функция (a·x + b) mod 256
        if len(feats) > max_features:  # превысили предел
            break  # не добавлять аффинные дальше
    for i in range(k):  # по первой позиции
        for j in range(i + 1, k):  # по второй позиции правее
            feats.append((f"diff{i}-{j}", (i, j), _diff_pair(i, j)))  # разность позиций i и j
    # XOR-комбинации: битовые маски с И и XOR целых байтов
    for bits in range(1, 1 << k):  # по всем непустым наборам битов
        mask = tuple((bits >> i) & 1 for i in range(k))  # маска: m_i = (bits >> i) & 1
        feats.append((f"xor{mask}", mask, _xor_mask(mask)))  # XOR величин (m_i & x_i)
    def _xor_full(idxs):  # XOR целых байтов по списку индексов
        def f(x):  # x — кортеж байтов контекста
            s = 0  # накопитель
            for i in idxs:  # по индексам
                s ^= x[i]  # накопитель: s = s XOR x_i
            return s  # c = XOR_{i из idxs} x_i
        return f  # готовая функция
    for bits in range(1, 1 << k):  # по всем непустым наборам битов
        idxs = [i for i in range(k) if (bits >> i) & 1]  # индексы, у которых бит bits равен 1
        feats.append((f"fxor{idxs}", tuple(idxs), _xor_full(idxs)))  # XOR байтов с этими индексами
    for i in range(k):  # по позициям
        feats.append((f"pos{i}", (i,), _position(i)))  # отдельная позиция i
    return feats[:max_features]  # первые max_features характеристик


# ---- синтез ----------------------------------------------------


def _vfeat(kind: str, param, X) -> "np.ndarray":  # тип характеристики, параметр, матрица контекстов
    """Векторные значения характеристики для контекстов X (N, k): массив int64
    из [0, 256). Совпадает бит в бит с функциями enumerate_features."""
    X = np.asarray(X, dtype=np.int64)  # матрица контекстов как int64
    if kind == "aff":  # аффинная: c = (X·a + b) mod 256
        return (X @ np.asarray(param[0], dtype=np.int64) + param[1]) % 256  # значения: c_n = (Σ_i a_i·X[n, i] + b) mod 256
    if kind == "diff":  # разность: c = (X[:, i] − X[:, j]) mod 256
        i, j = param  # позиции i и j
        return (X[:, i] - X[:, j]) % 256  # значения: c_n = (x_i − x_j) mod 256
    if kind == "xorm":  # маска: c = XOR_i (m_i & x_i)
        return np.bitwise_xor.reduce(X & np.asarray(param, dtype=np.int64), axis=1)  # XOR по оси контекста
    if kind == "xfull":  # XOR целых байтов: c = XOR_{i из I} x_i
        return np.bitwise_xor.reduce(X[:, list(param)], axis=1)  # XOR столбцов с индексами из param
    if kind == "pos":  # позиция: c = x[i]
        return X[:, param[0]]  # столбец с индексом param[0]
    raise KeyError(kind)  # неизвестный тип — ошибка


def _kind(name: str) -> str:  # имя характеристики
    """Вернуть тип характеристики по префиксу её имени."""
    for pre, kind in (("aff", "aff"), ("diff", "diff"), ("fxor", "xfull"),  # по парам (префикс имени, тип)
                      ("xor", "xorm"), ("pos", "pos")):  # продолжение списка пар
        if name.startswith(pre):  # имя начинается с префикса
            return kind  # соответствующий тип
    raise KeyError(name)  # неизвестное имя — ошибка


def _majVotes(c: "np.ndarray", y: "np.ndarray", idx: "np.ndarray"):  # значения характеристики, ответы, номера пар слоя
    """Мажоритарное голосование y по каждому значению характеристики на парах idx.

        H[v, t] = число пар слоя с c = v, y = t; tot[v] = Σ_t H[v, t];
        top[v] = argmax_t H[v, t]; top_n[v] = H[v, top[v]].
        Значение v проходит, если top_n >= 2 и top_n / max(tot, 1) >= 0.6.
        Возврат: (rows, support), rows = [(v, top[v])],
        support = Σ top_n по прошедшим / max(|idx|, 1)."""
    cc, yy = c[idx], y[idx]  # значения характеристики и ответы пар слоя
    m = int(cc.max()) + 1 if len(cc) else 0  # число возможных значений: max(c) + 1 или 0
    if m == 0:  # слой пуст
        return [], 0.0  # нет строк, опора 0
    H = np.bincount(cc * 256 + yy, minlength=m * 256).reshape(m, 256)  # таблица H[v, t]: индекс элемента = c·256 + y
    tot = H.sum(1)  # суммы голосов: tot[v] = Σ_t H[v, t]
    top = H.argmax(1)  # мажоритарный ответ: top[v] = argmax_t H[v, t]
    top_n = H[np.arange(m), top]  # число голосов: top_n[v] = H[v, top[v]]
    ok = (top_n >= 2) & (top_n / np.maximum(tot, 1) >= 0.6)  # условие: top_n >= 2 и top_n / max(tot, 1) >= 0.6
    vals = np.nonzero(ok)[0]  # значения, прошедшие условие
    rows = [(int(v), int(top[v])) for v in vals]  # строки (значение, мажоритарный ответ)
    return rows, float(top_n[ok].sum() / max(len(idx), 1))  # строки и опора = Σ top_n прошедших / max(|idx|, 1)


def _mode_share(ys) -> float:  # список ответов
    """Доля самого частого ответа: столько набрал бы совет «всегда предсказывай
    моду» — базовый уровень, который закон обязан превзойти."""
    if not len(ys):  # список пуст
        return 0.0  # доля 0
    c = np.bincount(np.asarray(ys, dtype=np.int64) % 256, minlength=256)  # c[t] = число ответов, равных t
    return float(c.max() / len(ys))  # доля = max_t c[t] / len(ys)


def invent_multi(pairs, rng: "random.Random | None" = None,  # пары; генератор случайных чисел
                 max_attempts: int = 8000, n_shifts: int = 8,  # предел числа характеристик; период по умолчанию
                 min_support: float = 0.30,  # минимальная опора слоя
                 max_period: int | None = None,  # наибольший период или None
                 min_edge: float = 0.10) -> dict | None:  # минимальный перевес над модой
    """Синтез частичных законов для потока байтов.

        Законы реального потока выполняются только на выровненных по записи
        позициях (например, каждый третий байт); остальные позиции — шум.
        Одна глобальная проверка такие законы отклоняет, поэтому поиск идёт по
        слоям.
        1) Кандидат — характеристика, отображающая контексты в значение c.
        2) Для слоя (P, s), то есть пар с pos mod P = s, по мажоритарному
           голосованию y для каждого значения c подгоняется обобщающий базис
           (POLY_LIN или OUT_PERIOD); затем проверка на отложенной части слоя:
           доля совпадений >= 0.9 и опора мажоритарных голосов >= min_support.
        3) Принятый закон возвращается со статусом PARTIAL_LAW и долей support:
           он предсказывает с известной надёжностью на выровненных позициях и
           не применяется на остальных. Статус PROVEN_EXACT не ставится: он
           требует полного детерминированного покрытия.
        Период P перебирается от 1 до max_period (по умолчанию n_shifts), для
        каждого — все фазы s. Оставляются периоды с n // P >= 80.
        Позиция пары берётся из третьего элемента (ctx, y, pos), иначе —
        индекс в списке (верно только для непрерывного потока).
        Кандидаты сравниваются по ключу (support, −total_bits); досрочный выход
        только при support >= 0.999. Нет прошедшего — {"status": "NO_LAW",
        "attempts": attempts}."""
    rng = rng or random.Random(0)  # генератор: данный или с зерном 0
    if not pairs:  # пар нет
        return None  # нечего искать
    k = len(pairs[0][0])  # арность контекста = длина первого контекста
    if any(len(x) != k for pr in pairs for x in (pr[0],)):  # есть контекст другой длины
        return None  # нечего искать
    if max_period is None:  # наибольший период не задан
        max_period = int(n_shifts)  # взять n_shifts
    GEN_BASES = [(n, f) for n, f in BASES if n in ("POLY_LIN", "OUT_PERIOD")]  # только обобщающие базисы
    X = np.asarray([pr[0] for pr in pairs], dtype=np.int64)  # матрица контекстов (N, k)
    y = np.asarray([pr[1] for pr in pairs], dtype=np.int64)  # вектор ответов (N,)
    n = len(pairs)  # число пар
    # Слои задаются позицией В ПОТОКЕ. Пара может нести её третьим элементом
    # (ctx, y, pos); иначе используется индекс в списке (верно только для
    # непрерывного, не прореженного потока). Конвейер передаёт сюда ошибки
    # модели: пропуск одной позиции сдвигает нумерацию, и закон теряется.
    # позиция пары — из третьего элемента, если он есть у всех пар,
    # иначе — её индекс в списке
    pos = np.asarray([int(pr[2]) for pr in pairs if len(pr) > 2],  # позиции из третьего элемента каждой пары
                     dtype=np.int64) if all(len(pr) > 2 for pr in pairs) else np.arange(n)  # если есть у всех пар, иначе 0..n−1
    periods = [P for P in range(1, max(1, max_period) + 1)  # периоды P = 1..max(1, max_period)
               if n // P >= 80]  # только если n // P >= 80
    attempts = 0  # число рассмотренных характеристик
    winner, winner_key = None, None  # лучший кандидат и его ключ
    for fname, fparam, _ffn in enumerate_features(k):  # по тройкам (имя, параметр, функция)
        attempts += 1  # попыток: +1
        if attempts > max_attempts:  # предел попыток исчерпан
            break  # конец поиска
        c = _vfeat(_kind(fname), fparam, X)  # значения характеристики на всех парах
        # ОДИН представитель контекста на каждое значение характеристики по ВСЕМ
        # парам (первое вхождение). Строится один раз на характеристику:
        # межслойная проверка читает значения, которых в текущем слое может не
        # быть; таблица только по слою давала −1 и проверяла против pairs[−1],
        # то есть против бессмысленного контекста.
        rep = np.full(int(c.max()) + 1, -1, dtype=np.int64)  # rep[v] = −1 для v = 0..max(c)
        rep[c[::-1]] = np.arange(n)[::-1]  # запись в обратном порядке: останется ПЕРВОЕ вхождение каждого значения
        for P in periods:  # по периодам
            for shift in range(P):  # по фазам s = 0..P−1
                idx = np.nonzero(pos % P == shift)[0]  # номера пар слоя: pos mod P = s
                if len(idx) < 80:  # в слое меньше 80 пар
                    continue  # пропустить слой
                rows, support = _majVotes(c, y, idx)  # мажоритарные голоса и опора слоя
                if len(rows) < 24 or support < min_support:  # меньше 24 значений или опора мала
                    continue  # пропустить слой
                order = list(range(len(rows)))  # порядок 0..len(rows)−1
                rng.shuffle(order)  # случайно перемешать
                rows = [rows[i] for i in order]  # строки в случайном порядке
                fit_n = max(8, int(0.6 * len(rows)))  # число строк подгонки: max(8, ⌊0.6·len(rows)⌋)
                fit, oos = rows[:fit_n], rows[fit_n:]  # первые fit_n — подгонка, остальные — проверка
                oos_ctx = [(pairs[int(rep[cc])][0], yy) for cc, yy in oos]  # отложенные пары (контекст-представитель, ответ)
                best = None  # лучший чертёж слоя
                for b_name, b_fn in GEN_BASES:  # по обобщающим базисам
                    payload = b_fn(fit)  # нагрузка базиса по строкам подгонки
                    if payload is None:  # базис не подошёл
                        continue  # следующий базис
                    b_bits = _payload_bits(b_name, payload, len(fit))  # стоимость нагрузки в битах
                    # чертёж хранит ИСХОДНУЮ скалярную функцию характеристики
                    bp = ConstructBlueprint(fname, fparam, _ffn, b_name,  # чертёж: имя, параметр, функция, базис
                                            payload, 9,  # нагрузка и мета-биты 9
                                            total_bits=b_bits + 9)  # полная стоимость = b_bits + 9
                    hits = sum(1 for x, yy in oos_ctx  # число совпадений на отложенных парах
                               if bp.apply(x) == yy)  # предсказание чертежа равно ответу
                    if oos and hits / len(oos) < 0.9:  # есть отложенные и доля совпадений < 0.9
                        continue  # базис не проходит
                    # закон обязан ОБГОНАТЬ моду своего отложенного набора. На реальном куске
                    # «закон» с долей 0.90 на подгонке набрал ровно базовый уровень
                    # «предсказывай моду» (0.233 = 0.233) на невидимой половине файла —
                    # это подгонка под самый частый байт потока, богатого нулями.
                    # 0.233) on the unseen half of the file — a fit to  # проверка: доля совпадений >= доля моды + min_edge
                    # the most frequent byte of a zero-rich stream.  # минимальный перевес над модой
                    if oos and hits / len(oos) < (  # доля совпадений не выше
                            _mode_share([yy for _, yy in oos_ctx])  # доли моды отложенных ответов
                            + min_edge):  # плюс минимальный перевес
                        continue  # не проходит
                    # межслойная проверка: мажоритарные голоса ДРУГИХ фаз того же
                    # периода; шумовые фазы не несут свидетельств и не должны
                    # блокировать закон
                    cross_ok = True  # межслойная проверка пока пройдена
                    for other in range(P):  # по всем фазам периода
                        if other == shift:  # та же фаза
                            continue  # пропустить
                        idx2 = np.nonzero(pos % P == other)[0]  # номера пар другой фазы
                        rows2, sup2 = _majVotes(c, y, idx2)  # её мажоритарные голоса и опора
                        if sup2 < min_support:  # опора мала (шумовая фаза)
                            continue  # не блокирует
                        cnt2 = np.bincount(c[idx2], minlength=len(rep))  # cnt2[v] = число пар фазы со значением v
                        maj = [(pairs[int(rep[cc])][0], yy)  # пары (контекст-представитель, ответ)
                               for cc, yy in rows2 if cnt2[cc] >= 3]  # только значения с числом голосов >= 3
                        if len(maj) < 8:  # недостаточно свидетельств
                            continue  # не блокирует
                        hit = sum(1 for x, yy in maj  # число совпадений в другой фазе
                                  if bp.apply(x) == yy)  # предсказание равно ответу
                        if hit / len(maj) < 0.7:  # доля совпадений < 0.7
                            cross_ok = False  # межслойная проверка провалена
                            break  # прекратить
                    if cross_ok and (best is None  # проверка пройдена и лучшего ещё нет
                                     or bp.total_bits < best.total_bits):  # или этот чертёж дешевле
                        best = bp  # новый лучший чертёж слоя
                if best is not None:  # в слое найден чертёж
                    cand = {"status": "PARTIAL_LAW",  # кандидат: статус частичного закона
                            "blueprints": [best],  # его чертёж
                            "feature": fname, "shift": shift,  # характеристика и фаза
                            "n_shifts": P,  # период
                            "support": round(support, 3),  # опора с точностью 0.001
                            "attempts": attempts}  # число попыток на этот момент
                    # «Первый прошедший» — не «закон». При переборе периода более
                    # слабая подгонка на кратном периоде может пройти все пороги
                    # на более ранней характеристике. Поэтому хранится лучший по
                    # (support, затем биты описания); досрочный выход — только
                    # при точном законе, который нечем улучшить.
                    # best by (support, then MDL bits); stop early only  # продолжение объяснения ранжирования
                    # on an EXACT law, which nothing can improve.  # ключ сравнения
                    key = (cand["support"], -best.total_bits)  # ключ: (опора, −стоимость в битах)
                    if winner is None or key > winner_key:  # лучшего ещё нет или ключ больше
                        winner, winner_key = cand, key  # новый лучший кандидат
                    if cand["support"] >= 0.999:  # опора почти полная — точный закон
                        return winner  # досрочный выход
    return winner or {"status": "NO_LAW", "attempts": attempts}  # лучший кандидат или «закон не найден»
