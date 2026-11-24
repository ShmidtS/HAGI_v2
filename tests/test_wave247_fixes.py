"""Тесты границ слияния, таксономии причин задержки, базиса-периода, синтеза законов
по слоям потока, компиляции битовых схем, полярной нормы и адаптивного шага.

    1. merge_bound: по потерям L1 = L(w1), L2 = L(w2) и градиенту g в w1 строится
       линейность ℓ(t) = L1 + g·(w2 − w1)·t; невязки δ = |L(mid) − ℓ(1/2)|,
       δ₂ = |L2 − ℓ(1)|, где mid = (w1 + w2)/2; сертифицированная граница
       bound = (L1 + L2)/2 + δ + δ₂/2, и L(mid) ≤ bound.
    2. Таксономия: шум узкого алфавита — RANDOM_INCOMPRESSIBLE (периодическая доля
       period_hit < 0,3, случайный уровень shift_chance = 3/K); периодический закон —
       REPRESENTATION_GAP с верным периодом; период ищется только на информативных
       парах (хотя бы один конец не равен самому частому ответу).
    3. basis_out_period: y = (x + o) mod p; при p > max(y) смещение o = (y0 − x0) mod p,
       а p обязан делить все d_i = (y_i − x_i) − (y0 − x0).
    4. invent_multi: закон действует на выровненных позициях потока (i mod P = s);
       период P и фаза s ищутся перебором; статус PARTIAL_LAW.
    5. Компиляция схем: compile_optimal даёт схему не хуже прежней и точную на всех входах.
    6. polar_express: приближение полярного множителя X(XᵀX)^(−1/2), сингулярные числа
       в заданном интервале.
    7. _adaptive_step: сертифицированное снижение потери с константой кривизны
       κ(R) = (e^R − 1 − R)/R²."""
import numpy as np  # массивы и случайные числа
import pytest  # параметризация и approx

from hagi.diagnosis import build_residual_profile, classify_hold  # профиль остатка и причина задержки
from hagi.invention import basis_out_period  # базис-период y = (x + o) mod p
from hagi.trunk_ops import merge_bound  # сертифицированная граница потери в средней точке


def _hold(y):
    """Построить профиль остатка таблицы y (аргументы 0..len(y)−1) и классифицировать задержку.

        Возвращает (метка причины, профиль)."""
    xs = list(range(len(y)))  # аргументы 0, 1, …, len(y)−1
    p = build_residual_profile([[v] for v in xs], [int(v) for v in y])  # профиль по парам (x, y)
    return classify_hold("NO_VALIDATED_PROGRAM", p).value, p  # метка причины задержки и профиль


# ---- 1. merge_bound: граница через линейность ℓ(t) = L1 + g·(w2−w1)·t ----
def test_merge_bound_matches_lean_linearisation():
    """На L(w) = 10 + Σw², w1 = (0, 0), w2 = (2, 0): g(w1) = 0, значит ℓ(t) = L1 = 10;
    L1 = 10, L2 = 14, mid = (1, 0), L(mid) = 11; δ = |11 − 10| = 1, δ₂ = |14 − 10| = 4,
    bound = (10 + 14)/2 + 1 + 4/2 = 15, и L(mid) ≤ bound."""
    L = lambda w: 10.0 + float(np.sum(np.asarray(w) ** 2))  # потеря L(w) = 10 + Σ_i w_i²
    g = lambda w: 2 * np.asarray(w, float)  # градиент g(w) = 2w
    r = merge_bound(L, g([0., 0.]), [0., 0.], [2., 0.])  # граница для w1 = (0, 0), w2 = (2, 0)
    assert r["delta_mid"] == pytest.approx(1.0)   # was 11.0 (L1 dropped)
    assert r["delta_end"] == pytest.approx(4.0)
    assert r["bound"] == pytest.approx(15.0)      # was 25.0
    assert r["L_mid"] == pytest.approx(11.0) and r["certified"]


def test_merge_bound_is_valid_on_a_nonconvex_loss():
    """На невыпуклой L(w) = Σ tanh(3·w_i) и 200 случайных пар: L(mid) ≤ bound + 1e-9 всегда."""
    rng = np.random.default_rng(0)  # генератор с зерном 0
    L = lambda w: float(np.sum(np.tanh(np.asarray(w) * 3.0)))  # потеря L(w) = Σ_i tanh(3·w_i) — невыпуклая
    for _ in range(200):  # 200 случайных проверок: ложная граница не допускается
        w1, w2 = rng.normal(size=6), rng.normal(size=6)  # две случайные точки w1, w2 из N(0,1) в 6 измерениях
        eps = 1e-6  # шаг численной производной eps
        g = np.array([(L(np.where(np.arange(6) == j, w1 + eps, w1))  # градиент в w1: g_j = (L(w1 + eps·e_j) − L(w1 − eps·e_j)) / (2·eps)
                       - L(np.where(np.arange(6) == j, w1 - eps, w1))) / (2 * eps)  # (продолжение: вычитание L со сдвигом −eps и деление на 2·eps)
                      for j in range(6)])  # по координатам j = 0..5
        r = merge_bound(L, g, w1, w2)  # сертифицированная граница для пары (w1, w2)
        assert r["L_mid"] <= r["bound"] + 1e-9


# ---- 2. таксономия причин задержки ----
def test_narrow_alphabet_noise_is_random_not_a_representation_gap():
    """Случайные ответы над алфавитом из K = 4, 8, 16 значений — это шум, а не пробел представления:
    метка RANDOM_INCOMPRESSIBLE, period_hit < 0,3, shift_chance = 3/K."""
    r = np.random.default_rng(0)  # генератор с зерном 0
    for k in (4, 8, 16):  # по размерам алфавита K
        label, p = _hold(r.integers(0, k, 2000))  # профиль случайной таблицы из 2000 ответов 0..K−1 и метка задержки
        assert label == "RANDOM_INCOMPRESSIBLE", k  # метка — несжимаемый шум (K — только для сообщения об ошибке)
        assert p.period_hit < 0.3 and p.shift_chance == pytest.approx(3 / k)  # периодическая доля < 0,3; случайный уровень сдвига = 3/K


def test_periodic_recurrence_laws_are_not_called_noise():
    """Фибоначчи по модулю m с периодами 28, 16, 24 не называется шумом:
    метка REPRESENTATION_GAP и найден верный период."""
    def fibmod(m, n=4000):
        """Ряд a_i: a_0 = 0, a_1 = 1, a_{i+1} = (a_i + a_{i−1}) mod m, длины n."""
        f, a, b = [], 0, 1  # ряд f пуст; a = 0, b = 1
        for _ in range(n):  # n членов ряда
            f.append(a); a, b = b, (a + b) % m  # дописать a; (a, b) := (b, (a + b) mod m)
        return f  # вернуть ряд
    for m, period in ((13, 28), (7, 16), (16, 24)):  # по парам (модуль, ожидаемый период)
        label, p = _hold(fibmod(m))  # профиль ряда и метка задержки
        assert label == "REPRESENTATION_GAP" and p.period == period  # метка — пробел представления, период найден верно


def test_period_probe_reports_unknown_on_sparse_tables():
    """На разреженной таблице (2% ненулевых случайных байтов) период не выражен:
    period_hit = 0,0, метка REPRESENTATION_GAP."""
    r = np.random.default_rng(0)  # генератор с зерном 0
    label, p = _hold(np.where(r.random(2000) < 0.02, r.integers(1, 256, 2000), 0))  # таблица: 2% позиций — случайный байт 1..255, остальные 0
    assert p.period_hit == 0.0 and label == "REPRESENTATION_GAP"  # периода нет, метка — пробел представления


def test_period_probe_ignores_the_dominant_value():
    """Если 95% ответов равны 0, период не считается за счёт совпадения нулей:
    period_hit < 0,95; покрытие period_cover < 0,2 оценивается по 5% меньшинства."""
    r = np.random.default_rng(0)  # генератор с зерном 0
    y = np.where(r.random(2000) < 0.05, 7, 0)  # таблица: 5% ответов равны 7, остальные 0 (одного значения закона нет)
    _, p = _hold(y)  # только профиль
    assert p.period_hit < 0.95  # периодическая доля < 0,95 (раньше было 0,95 из-за совпадения нулей)
    assert p.period_cover < 0.2  # покрытие < 0,2: оценка только по 5% меньшинства


# ---- 3. basis_out_period: полный перебор как эталон на малых примерах ----
def _reference(pairs):
    """Эталон: наименьший период p из 2..256 и смещение o из 0..p−1, для которых (x + o) mod p = y
    на всех парах; возвращает словарь базиса или None."""
    for p in range(2, 257):  # перебор периода p
        for o in range(p):  # перебор смещения o
            if all((x + o) % p == y for x, y in pairs):  # все пары удовлетворяют (x + o) mod p = y — базис найден
                return {"kind": "OUT_PERIOD", "p": p, "o": o}
    return None  # ни одна пара (p, o) не подошла


@pytest.mark.parametrize("seed", range(12))
def test_basis_out_period_equals_exhaustive_scan(seed):
    """basis_out_period совпадает с полным перебором на 150 случайных наборах пар для каждого из 12 зёрен
    (шесть видов зависимостей y от x)."""
    r = np.random.default_rng(seed)  # генератор с зерном seed
    for _ in range(150):  # 150 случайных задач
        k = int(r.integers(3, 9))  # число пар k из 3..8
        mode = r.integers(0, 6)  # вид зависимости y от x: 0..5
        xs = r.integers(0, 256, k)  # аргументы xs
        ys = {0: xs % 2, 1: r.integers(250, 256, k), 2: xs,  # словарь зависимостей: 0 — xs mod 2; 1 — случайные 250..255; 2 — тождество
              3: r.integers(0, 256, k),  # 3 — случайные 0..255
              4: (xs + int(r.integers(0, 26))) % int(r.choice([3, 13, 26])),  # 4 — (xs + o) mod m для m из {3, 13, 26}
              5: (xs * 7 + 3) % 256}[mode]  # 5 — (7·xs + 3) mod 256; выбран вариант mode
        pairs = list(zip(map(int, xs), map(int, ys)))  # пары (x, y) целыми числами
        assert basis_out_period(pairs) == _reference(pairs), pairs  # базис совпадает с полным перебором (pairs — для сообщения)


def test_basis_out_period_is_fast():
    """Скорость: y = (x + 5) mod 13 на 256 парах распознаётся как {p = 13, o = 5} менее чем за 0,01 с."""
    import time  # секундомер
    pairs = [(x, (x + 5) % 13) for x in range(256)]  # 256 пар y = (x + 5) mod 13
    t = time.perf_counter(); out = basis_out_period(pairs)  # засечь время и найти базис
    assert out == {"kind": "OUT_PERIOD", "p": 13, "o": 5}  # базис — период 13, смещение 5
    assert time.perf_counter() - t < 0.01  # время поиска < 0,01 с


# ---- 3b. invent_multi: поиск периода слоя ----
K = 3  # длина контекста K = 3 байта


def _stream(period, fn, n=3000, seed=0):
    """Случайный поток длины n + K, где на позициях (i − K) mod period = 0 байт задаётся законом
    fn от K предыдущих байтов ( mod 256); возвращает пары (контекст из 3 байтов, байт)."""
    import random  # генератор случайных байтов
    r = random.Random(seed)  # генератор с зерном seed
    data = bytearray(r.randbytes(n + K))  # случайный поток n + K байтов
    for i in range(K, len(data)):  # по позициям i от K до конца
        if (i - K) % period == 0:  # позиция выровнена: (i − K) mod period = 0
            data[i] = fn(data[i - 3], data[i - 2], data[i - 1]) & 255  # байт задаётся законом fn от трёх предыдущих байтов, приведённым к 0..255 операцией & 255
    return [(tuple(data[i - K:i]), data[i]) for i in range(K, len(data))]  # пары (контекст из 3 байтов, следующий байт)


@pytest.mark.parametrize("period", [3, 4, 5, 6, 7, 8])
def test_invent_multi_finds_laws_on_every_period_up_to_8(period):
    """Закон y = a + 2b − c + 7 на каждом периоде 3..8 находится: статус PARTIAL_LAW,
    n_shifts = period, точность на слое фазы > 0,99."""
    # 6000 байтов: в каждом слое тогда не меньше 750 пар. При 3000 слой периода 8
    # (375 пар) слишком мал для почти инъективного закона — он всё равно НАХОДИТСЯ,
    # с поддержкой 0,744 и сырой точностью 0,971 (контракт PARTIAL_LAW сообщает именно это).
    import random  # генератор случайных чисел
    from hagi.invention_multi import invent_multi  # синтез законов по слоям потока
    pairs = _stream(period, lambda a, b, c: a + 2 * b - c + 7, n=6000)  # поток периода period с законом y = a + 2b − c + 7, 6000 пар
    r = invent_multi(pairs, random.Random(0))  # результат синтеза с зерном 0
    assert r["status"] == "PARTIAL_LAW"  # статус — частичный закон
    bp, P, s = r["blueprints"][0], r["n_shifts"], r["shift"]  # чертёж закона, найденный период P и фаза s
    assert P == period  # найденный период равен истинному
    idx = [i for i in range(len(pairs)) if i % P == s]  # номера пар слоя: i mod P = s
    assert sum(bp.apply(pairs[i][0]) == pairs[i][1] for i in idx) / len(idx) > 0.99  # точность чертежа на слое: доля верных > 0,99


def test_invent_multi_best_law_wins_over_the_first_that_passes():
    """Если на кратном периода проходит более слабый закон, поиск возвращает истинный период:
    n_shifts = 4 и feature = aff(1, 2, −1)+0."""
    # Более слабая подгонка на КРАТНОМ периода проходит все гейты на более раннем признаке;
    # поиск обязан вернуть истинный период.
    import random  # генератор случайных чисел
    from hagi.invention_multi import invent_multi  # синтез законов по слоям потока
    pairs = _stream(4, lambda a, b, c: a + 2 * b - c + 7)  # поток периода 4 с законом y = a + 2b − c + 7
    r = invent_multi(pairs, random.Random(0))  # результат синтеза с зерном 0
    assert r["n_shifts"] == 4 and r["feature"] == "aff(1, 2, -1)+0"  # найден период 4 и чертёж — аффинный с коэффициентами (1, 2, −1) и сдвигом 0


def test_invent_multi_never_invents_a_law_for_noise():
    """На чистом шуме (8 зёрен, по 3000 случайных пар) закон не изобретается: NO_LAW."""
    import random  # генератор случайных чисел
    from hagi.invention_multi import invent_multi  # синтез законов по слоям потока
    for seed in range(8):  # по 8 зернам
        r0 = random.Random(100 + seed)  # генератор пар с зерном 100 + seed
        pairs = [(tuple(r0.randbytes(K)), r0.randbytes(1)[0])  # 3000 случайных пар: контекст из K случайных байтов и случайный байт
                 for _ in range(3000)]  # (продолжение: число пар)
        assert invent_multi(pairs, random.Random(seed))["status"] == "NO_LAW"  # для шума закон не найден


def test_vectorised_features_are_bit_exact_with_the_callables():
    """Векторизованные значения признаков _vfeat совпадают с обычными функциями признаков
    для всех признаков enumerate_features при длине контекста k = 2, 3, 4 (2000 строк)."""
    from hagi.invention_multi import _kind, _vfeat, enumerate_features  # вид признака, векторизованное значение и список признаков
    r = np.random.default_rng(0)  # генератор с зерном 0
    for k in (2, 3, 4):  # по длинам контекста k
        X = r.integers(0, 256, (2000, k), dtype=np.uint8)  # 2000 случайных строк из k байтов
        for name, param, fn in enumerate_features(k):  # по признакам: имя, параметр, функция
            got = _vfeat(_kind(name), param, X)  # векторизованное значение признака на всех строках
            ref = np.fromiter((fn(tuple(int(v) for v in row)) for row in X),  # эталон: значения функции fn по каждой строке
                              dtype=np.int64, count=len(X))  # тип int64, ровно len(X) значений
            assert not (got != ref).any(), name  # векторизованные значения совпадают побитно (name — для сообщения)


# ---- 4. compile_factored_gf2 / compile_optimal: компиляция битовых схем ----
def _anf(n, fn):
    """Алгебраическая нормальная форма булевой функции fn от n входов: набор мономов каждой координаты выхода.

        Коэффициент при мономе S: c_S = ⊕_{T ⊆ S} fn(T), формула Мёбиуса над GF(2).
        Возвращает кортеж замороженных множеств номеров мономов по выходам."""
    outs = []  # мономы по выходам
    for o in range(len(fn((0,) * n))):  # по выходным координатам o
        T = set()  # множество мономов выхода o
        for s in range(1 << n):  # перебор монома S (битовая маска)
            c = 0  # коэффициент c
            for t in range(1 << n):  # перебор входа T
                if t & s == t:  # T — подмножество S
                    c ^= fn(tuple((t >> i) & 1 for i in range(n)))[o]  # c := c ⊕ fn(T)[o] — исключающее ИЛИ со значением выхода o
            if c:  # коэффициент ненулевой
                T.add(s)  # добавить моном S
        outs.append(frozenset(T))  # замороженное множество мономов выхода
    return tuple(outs)  # кортеж по выходам


def _adder2(x):
    """Двухбитовый сумматор: s = x0 + 2·x1 + x2 + 2·x3; возвращает биты s3, s2, s1."""
    s = x[0] + 2 * x[1] + x[2] + 2 * x[3]  # сумма двух двухбитовых чисел: s = (x0 + 2·x1) + (x2 + 2·x3)
    return ((s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1)  # биты суммы s3, s2, s1


def _mul2(x):
    """Двухбитовый умножитель: p = (x0 + 2·x1)·(x2 + 2·x3); возвращает биты p3, p2, p1, p0."""
    p = (x[0] + 2 * x[1]) * (x[2] + 2 * x[3])  # произведение двух двухбитовых чисел: p = (x0 + 2·x1)·(x2 + 2·x3)
    return ((p >> 3) & 1, (p >> 2) & 1, (p >> 1) & 1, p & 1)  # биты произведения p3, p2, p1, p0


def _exact(net, n, fn):
    """Схема net точна для fn от n входов: символическая проверка по АНФ и совпадение выхода
    на всех 2^n входах."""
    import hagi.skill_engine as S  # движок схем
    return S.symbolic_verify_general(net, _anf(n, fn)) and all(  # символическая проверка схемы по АНФ и
        S.eval_net(net, tuple((q >> i) & 1 for i in range(n)))  # совпадение выхода схемы с fn
        == fn(tuple((q >> i) & 1 for i in range(n))) for q in range(1 << n))  # на всех 2^n входах q


def test_compile_optimal_shrinks_the_2x2_multiplier():
    """compile_optimal уменьшает умножитель 2×2: не более 7 вентилей, глубина ≤ 3, ≤ 100 бит программы;
    схема точна."""
    import hagi.skill_engine as S  # движок схем
    net = S.ProofCarryingANFLearner.compile_optimal(4, _anf(4, _mul2))  # оптимальная компиляция АНФ умножителя по 4 входам
    m = net["metrics"]  # метрики схемы
    assert len(net["gates"]) <= 7 and m["depth"] <= 3 and m["program_bits"] <= 100  # не более 7 вентилей, глубина ≤ 3, не более 100 бит программы
    assert _exact(net, 4, _mul2)  # схема точна


def test_compile_optimal_keeps_the_adder_and_is_never_worse_than_before():
    """Для сумматора compile_optimal даёт 8 вентилей и точную схему; на случайных функциях 3–5 входов
    схема не хуже прежней по кортежу (число вентилей, глубина, биты программы)."""
    import hagi.skill_engine as S  # движок схем
    net = S.ProofCarryingANFLearner.compile_optimal(4, _anf(4, _adder2))  # оптимальная компиляция АНФ сумматора по 4 входам
    assert len(net["gates"]) == 8 and _exact(net, 4, _adder2)  # ровно 8 вентилей и схема точна
    r = np.random.default_rng(0)  # генератор с зерном 0
    for n in (3, 4, 5):  # по числу входов n
        for _ in range(8):  # 8 случайных функций
            outs = []  # АНФ случайной функции по выходам
            for _ in range(int(r.integers(1, 4))):  # число выходов 1..3
                tt = r.integers(0, 2, 1 << n)  # таблица истинности выхода: 2^n битов
                outs.append(frozenset(s for s in range(1 << n)  # моном S входит, если сумма битов таблицы по T ⊆ S нечётна
                                      if sum(tt[t] for t in range(1 << n) if t & s == t) % 2))  # (продолжение условия)
            outs = tuple(outs)  # кортеж выходов
            opt = S.ProofCarryingANFLearner.compile_optimal(n, outs)  # оптимальная компиляция
            base = S.compile_factored_gf2(n, outs)  # прежняя компиляция (флаг выключен)
            assert S.symbolic_verify_general(opt, outs)  # схема символически верна
            key = lambda nt: (len(nt["gates"]), nt["metrics"]["depth"], nt["metrics"]["program_bits"])  # ключ сравнения: (число вентилей, глубина, биты программы)
            assert key(opt) <= min(key(base), key(S.compact_netlist(S.ProofCarryingANFLearner.compile(n, outs))))  # оптимальная не хуже обеих прежних схем по ключу


# ---- 5. Сертификат polar_express: сингулярные числа лежат в заданном интервале ----
def test_polar_express_singular_spectrum_lies_in_the_lean_box():
    """Для диагональной матрицы с числами s ∈ [lo, 1] сингулярные числа результата лежат в интервале:
    [0,9075; 1,0212] при lo = 0,1 и [9/512; 1049/1024] при lo = 0,001 (допуск 1e-9)."""
    from hagi.trunk_ops import polar_express  # приближение полярного множителя
    # Функция действует только на сингулярные числа; подаем её маленькой
    # диагональной матрицей (U = V = I, спектр = диагональ) партиями.
    # (Первая версия строила одну диагональную 20001×20001 — около 8e12 операций.)
    for lo, box in ((0.1, (0.9075, 1.0212)), (1e-3, (9 / 512, 1049 / 1024))):  # по нижним границам спектра и допустимым интервалам
        s = np.linspace(lo, 1.0, 20001)  # 20001 значений s от lo до 1
        for i in range(0, len(s), 64):  # партиями по 64
            out = np.linalg.svd(polar_express(np.diag(s[i:i + 64])), compute_uv=False)  # сингулярные числа polar_express от диагональной матрицы s
            assert out.min() >= box[0] - 1e-9 and out.max() <= box[1] + 1e-9, (lo, out.min(), out.max())  # все сингулярные числа в интервале [box0, box1] с допуском 1e-9


def test_polar_express_handles_wide_rank_deficient_and_zero():
    """Широкая матрица ранга 2 сохраняет форму; наибольшее сингулярное число ≤ 1049/1024, остальные около 0;
    нулевая матрица переходит в нулевую."""
    from hagi.trunk_ops import polar_express  # приближение полярного множителя
    r = np.random.default_rng(0)  # генератор с зерном 0
    G = r.normal(size=(8, 2)) @ r.normal(size=(2, 64))  # широкая матрица 8×64 ранга 2
    out = polar_express(G)  # результат
    s = np.linalg.svd(out, compute_uv=False)  # сингулярные числа результата
    assert out.shape == G.shape and s[0] <= 1049 / 1024 + 1e-9 and s[2:].max() < 1e-9  # форма сохранена; наибольшее число ≤ 1049/1024, остальные < 1e-9
    assert np.array_equal(polar_express(np.zeros((3, 3))), np.zeros((3, 3)))  # нулевая матрица даёт нулевую


# ---- 6. Слои по позиции В ПОТОКЕ, а не по номеру в списке ----
def test_invent_multi_survives_a_sampled_miss_list():
    """При выбрасывании одной позиции нумерация списка сдвигается, но закон периода 3 всё равно находится,
    если пары несут позицию в потоке (PARTIAL_LAW, n_shifts = 3); без позиций — NO_LAW."""
    # Конвейер передаёт invent_multi промахи модели. Выбрасывание ОДНОЙ позиции
    # сдвигает нумерацию списка; закон всё равно должен находиться, когда пары
    # несут свою позицию в потоке.
    import random  # генератор случайных чисел
    from hagi.invention_multi import invent_multi  # синтез законов по слоям потока
    r = random.Random(0)  # генератор с зерном 0
    raw = r.randbytes(9000 + K)  # 9000 + K случайных байтов
    data = bytearray(raw)  # изменяемая копия
    for i in range(K, len(data)):  # по позициям i от K до конца
        if (i - K) % 3 == 0:  # позиция выровнена под период 3
            data[i] = (data[i - 3] + 2 * data[i - 2] - data[i - 1] + 7) & 255  # байт задаётся законом a + 2b − c + 7, приведённым к 0..255
    pairs = [((data[i], data[i + 1], data[i + 2]), data[i + 3], i)  # пары (контекст, цель, позиция i)
             for i in range(0, 6000) if i % 97 != 0]  # для позиций 0..5999, кроме кратных 97 — нерегулярная подвыборка
    with_pos = invent_multi(pairs, random.Random(7))  # синтез с позициями, зерно 7
    assert with_pos["status"] == "PARTIAL_LAW" and with_pos["n_shifts"] == 3  # закон найден: период 3
  # те же пары без позиций дают NO_LAW
    # (это объясняет, почему конвейер обязан передавать позиции)
    assert invent_multi([(c, y) for c, y, _ in pairs], random.Random(7))["status"] == "NO_LAW"


def test_invent_multi_without_positions_keeps_the_old_contract():
    """Пары без позиций сохраняют прежний договор: закон периода 4 находится (PARTIAL_LAW, n_shifts = 4)."""
    import random  # генератор случайных чисел
    from hagi.invention_multi import invent_multi  # синтез законов по слоям потока
    pairs = _stream(4, lambda a, b, c: a + 2 * b - c + 7, n=6000)  # поток периода 4, 6000 пар
    r = invent_multi([(tuple(x), y) for x, y in pairs], random.Random(0))  # синтез пар без позиций, зерно 0
    assert r["status"] == "PARTIAL_LAW" and r["n_shifts"] == 4  # статус — частичный закон, период 4


def test_pipeline_multibyte_channel_runs_end_to_end(tmp_path, monkeypatch):
    """Канал многобайтовых законов в конвейере работает целиком: в примечаниях есть «multi-byte:»
    и строка «multi-byte: PARTIAL_LAW», нет «skipped», закон записан в реестр домена.

        Регрессия: канал отключался ошибкой имени внутри except, и кусок не сообщал
        ничего вместо NO_LAW / PARTIAL_LAW."""
    import random  # генератор случайных чисел
    from hagi import unified as U  # единая модель
    from hagi import pipeline as P  # конвейер символьного обучения
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)  # контрольная точка := tmp_path
    u = U.Unified()  # новая единая модель
    u.domains["d"] = {}  # пустая запись домена d
    r = random.Random(0)  # генератор с зерном 0
    data = bytearray(r.randbytes(9000 + K))  # 9000 + K случайных байтов
    for i in range(K, len(data)):  # по позициям i от K до конца
        if (i - K) % 3 == 0:  # позиция выровнена под период 3
            data[i] = (data[i - 3] + 2 * data[i - 2] - data[i + -1] + 7) & 255  # байт задаётся законом a + 2b − c + 7, приведённым к 0..255 (i + −1 = i − 1)
    data = bytes(data)  # неизменяемые байты
    u.model.train_guarded(data, epochs=2)  # обучить модель, 2 эпохи
    res = P.ingest_symbolic(u, "d", data)  # символьное обучение домена d
    notes = " | ".join(res.notes)  # примечания одной строкой через « | »
    assert "skipped" not in notes and "multi-byte:" in notes, notes  # нет «skipped» и есть «multi-byte:»
    assert any(n.startswith("multi-byte: PARTIAL_LAW") for n in res.notes), notes  # есть примечание «multi-byte: PARTIAL_LAW»
    assert u.law_registry.get("d"), "law was not registered"  # закон записан в реестр домена d


# ---- 7. Закон обязан побеждать базу «предсказать моду" ----
def test_invent_multi_rejects_a_mode_fit():
    """Поток без закона, на 55% из нулей: общий аффинный признак может подогнать самый частый байт,
    но закон не принимается — NO_LAW."""
    # Поток, богатый нулями, БЕЗ закона: общий аффинный признак может подогнать
    # самый частый байт. На реальном куске такой «закон» набирал ровно базу
    # «предсказывать моду» на невиданных данных.
    import random  # генератор случайных чисел
    from hagi.invention_multi import invent_multi  # синтез законов по слоям потока
    r = random.Random(3)  # генератор с зерном 3
    data = bytearray(r.randbytes(9000 + K))  # 9000 + K случайных байтов
    for i in range(len(data)):  # по всем позициям: 55% нулей, остальные — редкий шум
        if r.random() < 0.55:  # с вероятностью 0,55
            data[i] = 0  # байт := 0
    # контекст = 4 байта ДО цели (первая версия теста пропускала цель в её же
    # контекст, и синтезатор справедливо находил «y = −x3» с поддержкой 1,0).
    pairs = [((data[i], data[i + 1], data[i + 2], data[i + 3]), data[i + 4], i)  # пары (контекст из 4 байтов до цели, цель, позиция)
             for i in range(0, 8000)]  # для позиций 0..7999
    res = invent_multi(pairs, random.Random(7))  # синтез с зерном 7
    assert res["status"] == "NO_LAW"  # статус — закон не найден


# ---- 8. Кривизна адаптивного шага CertifiedStepLM._adaptive_step ----
def test_adaptive_step_curvature_is_kappa_times_var_and_never_overclaims():
    """Сертифицированное снижение шага никогда не завышено: из 40 шагов хотя бы один принят
    и ни разу (до − после) < заявлено − 1e-6; константа кривизны κ(R) = (e^R − 1 − R)/R²:
    κ(0,001) ≈ 0,5, κ(1) > 0,7, а при R = 1e-8 формула даёт отрицательное значение
    (поэтому в коде берётся предел 1/2)."""
    import math  # экспонента и математика
    import torch as t  # тензоры
    from hagi.lm import CertifiedStepLM  # модель с сертифицированным шагом
    seen = {}  # словарь наблюдений (не используется)
    orig_jvp = t.autograd.functional.jvp  # исходная функция JVP (не используется)
    m = CertifiedStepLM(dim=32, hidden=64, seed=0)  # модель d = 32, h = 64, зерно 0
    g = t.Generator().manual_seed(0)  # генератор с зерном 0
    viol = taken = 0  # счётчики нарушений и принятых шагов
    for _ in range(40):  # 40 попыток шага
        ctx = t.randint(0, 256, (16, m.k), generator=g)  # 16 случайных контекстов длины k
        nxt = t.randint(0, 256, (16,), generator=g)  # 16 случайных следующих байтов
        snap = [q.detach().clone() for q in m.params()]  # снимок параметров
        before = float(t.nn.functional.cross_entropy(m.logits(ctx), nxt))  # потеря до шага: CE(logits(ctx), nxt)
        claimed = m._adaptive_step(ctx, nxt, lr_scale=0.05)  # адаптивный шаг с масштабом 0,05 — малый масштаб делает адаптивную ветку доступной; возврат — заявленное снижение
        after = float(t.nn.functional.cross_entropy(m.logits(ctx), nxt))  # потеря после шага
        with t.no_grad():  # без записи градиентов
            for q, s in zip(m.params(), snap):  # по парам (параметр, снимок)
                q.copy_(s)  # вернуть параметр из снимка
        if claimed > 0:  # шаг принят (заявлено снижение > 0)
            taken += 1  # принятых шагов + 1
            viol += (before - after) < claimed - 1e-6  # нарушение, если фактическое снижение меньше заявленного − 1e-6
    assert taken > 0 and viol == 0  # хотя бы один шаг принят, нарушений нет
  # константа кривизны κ(R) = (e^R − 1 − R)/R²
    kappa = lambda R: (math.exp(R) - 1 - R) / R ** 2  # κ(0,001) ≈ 0,5 и κ(1) > 0,7
    assert kappa(1e-3) == pytest.approx(0.5, abs=1e-3) and kappa(1.0) > 0.7  # κ(1e-8) < 0 — поэтому в коде берётся предел 1/2
    assert kappa(1e-8) < 0            # отрицательное значение объясняет, почему в коде берётся предел 1/2


def test_adaptive_step_curvature_stays_positive_for_tiny_ranges():
    """При очень малых диапазонах логитов (веса умножены на 1e-6, диапазоны около 1e-6)
    шаг не увеличивает потерю: после ≤ до + 1e-9."""
    import math  # экспонента и математика
    import torch as t  # тензоры
    from hagi.lm import CertifiedStepLM  # модель с сертифицированным шагом
    m = CertifiedStepLM(dim=16, hidden=32, seed=0)  # модель d = 16, h = 32, зерно 0
    with t.no_grad():  # без записи градиентов
        for q in m.params():  # по всем параметрам
            q.mul_(1e-6)  # параметр умножить на 1e-6: логиты около 0, диапазоны около 1e-6
    g = t.Generator().manual_seed(0)  # генератор с зерном 0
    ctx = t.randint(0, 256, (8, m.k), generator=g)  # 8 случайных контекстов длины k
    nxt = t.randint(0, 256, (8,), generator=g)  # 8 случайных следующих байтов
    before = float(t.nn.functional.cross_entropy(m.logits(ctx), nxt))  # потеря до шага
    snap = [q.detach().clone() for q in m.params()]  # снимок параметров
    m._adaptive_step(ctx, nxt)  # адаптивный шаг
    after = float(t.nn.functional.cross_entropy(m.logits(ctx), nxt))  # потеря после шага
    assert after <= before + 1e-9  # шаг никогда не поднимает потерю: после ≤ до + 1e-9
