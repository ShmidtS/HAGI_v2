"""Замер оптимальной доли повторного обучения (replay) для модели RopeAttentionByteLM: два конфликтных домена при равном бюджете обновлений.

Проверяемая гипотеза: лучший результат даёт доля 1/e ≈ 0.368. Схема: 200 шагов
на домене A, затем STEPS шагов на домене B, где в каждом пакете n_old = ⌊BATCH·frac⌋
примеров взято из A, остальные BATCH − n_old — из B. Оценивается кросс-энтропия
CE = −ln P(следующий байт) на отложенных позициях 3500..4499 обоих доменов.
"""
import sys, os, math, time  # путь импорта, окружение, число e, замер времени
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # добавить корень проекта в путь импорта
os.environ.setdefault("HAGI_DEVICE", "cpu")  # устройство по умолчанию — процессор (если переменная не задана)
import numpy as np, torch  # численные массивы и тензоры
from hagi.lm import RopeAttentionByteLM  # байтовая модель с вниманием

K = 96  # длина контекста модели
FRACS = [0.26, 0.28, 1 / math.pi, 0.32, 0.34, 0.368]  # проверяемые доли повторного обучения
SEEDS = [7, 11, 23, 41, 73, 97, 131]  # зерна случайности
STEPS, BATCH = 300, 128  # число шагов на домене B и размер пакета


def domain(seed, kind):
    """Сгенерировать синтетический домен из 6000 байтов по закону, заданному kind.

        kind "A": b_i = (7i + 5·(i mod 3)) mod 256.
        kind "B": b_i = (13i mod 256) XOR (i mod 7).
        seed не используется. Возвращает массив uint8."""
    rng = np.random.default_rng(seed)  # генератор случайных чисел (в генерации не используется)
    if kind == "A":   # периодический закон
        a = np.arange(6000) * 7 % 256  # массив a_i = 7i mod 256
        return ((a + (np.arange(6000) % 3) * 5) % 256).astype(np.uint8)  # b_i = (a_i + 5·(i mod 3)) mod 256 в uint8
    # B: другой закон
    a = np.arange(6000) * 13 % 256  # массив a_i = 13i mod 256
    return ((a ^ (np.arange(6000) % 7)) % 256).astype(np.uint8)  # b_i = a_i XOR (i mod 7) в uint8


def windows(d, pos):
    """Вырезать примеры: X_j = d[p_j .. p_j+K−1] в обратном порядке, y_j = d[p_j + K].

        d: массив байтов, pos: позиции p. Возвращает (тензор X формы (n, K), тензор y формы (n,))."""
    pos = np.asarray(pos)  # позиции в массив numpy
    X = np.stack([d[p:p + K][::-1] for p in pos]).astype(np.int64)  # X_j = d[p_j .. p_j+K−1], развёрнутое: d[p_j+K−1], ..., d[p_j]
    return torch.as_tensor(X), torch.as_tensor(d[pos + K].astype(np.int64))  # (X, y_j = d[p_j + K]) тензорами


def ce(m, X, y):
    """Средняя кросс-энтропия CE = Σ_s CE(пакет s)·len(s) / len(X) по пакетам по 512 примеров."""
    with torch.no_grad():  # без вычисления градиентов
        tot = 0  # накопитель суммы взвешенных потерь
        for s in range(0, len(X), 512):  # пакеты примеров s, s+512, ...
            tot += float(torch.nn.functional.cross_entropy(m.logits(X[s:s+512]), y[s:s+512])) * len(X[s:s+512])  # tot += средняя CE пакета · размер пакета
    return tot / len(X)  # итог: tot / число примеров


def run(frac, seed):
    """Обучить модель с долей повтора frac и зерном seed; возвращает (CE на A, CE на B).

        Шаг: обнуление градиентов, CE(m.logits(X), y), backward, шаг Adam с lr = 3e-3
        над emb, W1, b1, W2, b2."""
    g = torch.Generator().manual_seed(seed)  # генератор torch с зерном seed (далее не используется)
    A, B = domain(1, "A"), domain(2, "B")  # домены A и B
    m = RopeAttentionByteLM(dim=64, hidden=128, seed=seed)  # модель шириной 64 и скрытым слоем 128
    rng = np.random.default_rng(seed)  # генератор случайных перестановок позиций
    a_pos = np.arange(0, 3000)  # обучающие позиции домена A: 0..2999
    opt = torch.optim.Adam([m.emb, m.W1, m.b1, m.W2, m.b2], lr=3e-3)  # оптимизатор Adam по пяти тензорам модели
    def step(X, y):
        """Один шаг оптимизации по пакету (X, y)."""
        opt.zero_grad(); torch.nn.functional.cross_entropy(m.logits(X), y).backward(); opt.step()  # обнулить градиенты; CE назад; шаг оптимизатора
    for _ in range(200):  # 200 шагов на домене A
        step(*windows(A, rng.permutation(a_pos)[:BATCH]))  # шаг по 128 случайным позициям A
    # обучение на B с долей повторного обучения frac
    b_pos = np.arange(0, 3000)  # обучающие позиции домена B: 0..2999
    n_old = int(BATCH * frac)  # число примеров из A в пакете: ⌊128·frac⌋
    for _ in range(STEPS):  # STEPS шагов на домене B
        pb = rng.permutation(b_pos)[:BATCH - n_old]  # позиции B: 128 − n_old случайных
        if n_old:  # повтор включён
            po = rng.permutation(a_pos)[:n_old]  # позиции A: n_old случайных
            Xb, yb = windows(B, pb); Xo, yo = windows(A, po)  # примеры из B и из A
            X, y = torch.cat([Xb, Xo]), torch.cat([yb, yo])  # склеить пакеты: сначала B, потом A
        else:  # повтора нет
            X, y = windows(B, pb)  # пакет только из B
        step(X, y)  # шаг оптимизации
    testA, testB = np.arange(3500, 4500), np.arange(3500, 4500)  # отложенные позиции 3500..4499
    return ce(m, *windows(A, testA)), ce(m, *windows(B, testB))  # (CE на A, CE на B)


if __name__ == "__main__":  # запуск только как сценарий
    t0 = time.time()  # начало отсчёта
    print("frac      A_CE    B_CE   balanced")  # строка заголовков
    res = {}  # доля -> средние (CE_A, CE_B)
    for f in FRACS:  # по всем долям
        allr = np.array([run(f, s) for s in SEEDS])  # матрица (число зерен, 2): CE на A и B
        r = allr.mean(0); b = allr.mean(1)  # r — среднее по зернам для A и B; b_i = (CE_A_i + CE_B_i)/2 по зерну i
        res[f] = r  # запомнить среднее для доли f
        print(f"{f:5.3f}  A={r[0]:6.3f} B={r[1]:6.3f} bal={b.mean():6.3f} "  # печать доли, CE на A, CE на B и bal = среднее b
              f"±{b.std(ddof=1)/math.sqrt(len(b)):.3f}", flush=True)  # печать ошибки среднего std(b)/√7
    best = min(res, key=lambda f: res[f].mean())  # доля с наименьшим средним (CE_A + CE_B)/2
    print(f"best fraction: {best:.3f} (1/e={1/math.e:.3f}) | {time.time()-t0:.0f}s")  # печать лучшей доли, 1/e и времени
