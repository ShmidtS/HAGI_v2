"""Перекрёстная энтропия с защитой диапазона, EF21, GPM, нейронное ядро, CRAM — модуль нейросетевой группы пакета hagi.core.neural.

Каждая строка снабжена русским комментарием: что делает и по какой формуле.
"""

import math  # импорт стандартной библиотеки, нужный этой части
from dataclasses import dataclass  # импорт стандартной библиотеки, нужный этой части
from typing import TYPE_CHECKING, Any, Optional, Sequence  # импорт стандартной библиотеки, нужный этой части
import numpy as np  # импорт стандартной библиотеки, нужный этой части
from hagi.base.gpu import GpuGpmMixin  # примесь диспетчеризации на графический ускоритель: база ContinualGPMMemory
from hagi.core.synthesis.venv_anf import (  # открыть явный импорт из предыдущей части: имена venv_anf
    block_hadamard_rotate,  # block_hadamard_rotate — имя из предыдущей части
    block_hadamard_unrotate,  # block_hadamard_unrotate — имя из предыдущей части
    pack_bitalloc_records,  # pack_bitalloc_records — имя из предыдущей части
    pack_vector_dyadic5,  # pack_vector_dyadic5 — имя из предыдущей части
    quantize_5block_exact,  # quantize_5block_exact — имя из предыдущей части
    sylvester_hadamard,  # sylvester_hadamard — имя из предыдущей части
    unpack_bitalloc_records,  # unpack_bitalloc_records — имя из предыдущей части
    unpack_vector_dyadic5  # unpack_vector_dyadic5 — имя из предыдущей части
)  # конец импорта из предыдущей части
if TYPE_CHECKING:  # только для проверки типов аннотаций (анализатор видит имя, импорта во время выполнения нет — нет цикла)
    from hagi.core.processor.processor import HedgeRouter  # маршрутизатор из части «processor»: имя нужно лишь в строковых аннотациях

# 5. Логарифмически-линейная перекрёстная энтропия с защитой диапазона, BF16, EF21-Muon, GPM и нейронное ядро
#============================== разделитель разделов ==============================


def bf16_nearest_float(x: float) -> float:
    """Округлить число до ближайшего в формате BF16 (16 бит: 8 показательных, 7 значащих).

    Нормальные числа: шаг = 2^(⌊log2|x|⌋ − 7); результат =
    sign(x)·round(|x| / шаг)·шаг (округление к ближайшему чётному).
    Модуль меньше 2^−126 даёт 0; ноль, NaN и бесконечности возвращаются как есть."""
    if x == 0.0 or not math.isfinite(x):  # ноль, NaN или бесконечность — округлять нечего
        return 0.0 if x == 0.0 else x  # ноль остаётся нулём, остальное возвращается как есть
    ax = abs(x)  # модуль числа
    if ax < 2.0 ** (-126):  # модуль меньше наименьшей нормальной степени 2^−126
        return 0.0  # округляется до нуля
    exp2 = math.floor(math.log2(ax))  # exp2 = ⌊log2|x|⌋ — двоичная экспонента
    step = 2.0 ** (exp2 - 7)  # шаг сетки: 2^(exp2 − 7)
    return math.copysign(round(ax / step) * step, x)  # sign(x)·round(|x| / шаг)·шаг


def batch_logit_safe_step_ce(
    X_feat: np.ndarray,  # признаки (B, d)
    y_int: np.ndarray,  # номера правильных классов (B,)
    num_classes: int,  # число классов
    steps: int = 300,  # максимум шагов
    l2_reg: Optional[np.ndarray] = None,  # коэффициенты L2-регуляризации или None
    use_bf16_guard: bool = False,  # проверять округление BF16
) -> dict[str, Any]:  # возвращает словарь с весами и статистикой шагов
    """Градиентный шаг логистической регрессии с сертифицированной длиной шага.

    X_feat — (B, d) признаки; y_int — (B,) номера классов; num_classes —
    число классов; steps — максимум шагов; l2_reg — (d,) коэффициенты
    L2-регуляризации или None; use_bf16_guard — проверка того, что
    округление весов до BF16 не съедает гарантированное убывание.

    Потери: CE = mean_i (lse_i − logit правильного класса) +
    ½·Σ reg_j·θ_j². Шаг — вдоль антиградиента D = −ḡ, где
    ḡ = (p − y)^T X / B + reg·θ. Длина шага η = 4A / (Q' + 4·curv),
    A = ‖ḡ‖², Q' = mean_i (range_i(X·D^T))², curv — вклад регуляризации;
    гарантированное убывание 2A² / (Q' + 4·curv)."""
    B, d = X_feat.shape  # B — число примеров, d — число признаков
    theta = np.zeros((num_classes, d), dtype=np.float64)  # веса θ (num_classes, d), изначально нули
    reg = (  # регуляризация:
        np.zeros(d, dtype=np.float64)  # нули, если l2_reg не задан,
        if l2_reg is None  # если регуляризация не задана
        else np.asarray(l2_reg, dtype=np.float64)  # иначе заданный массив как float64
    )

    """Потери перекрёстной энтропии с регуляризацией на весах th: число."""
    def ce_loss(th: np.ndarray) -> float:  # локальная функция потерь от весов th
        lg = X_feat @ th.T  # логиты lg = X·θ^T, форма (B, num_classes)
        lg_s = lg - np.max(lg, axis=1, keepdims=True)  # логиты без максимума по строке — для устойчивости exp
        lse = np.max(lg, axis=1) + np.log(np.sum(np.exp(lg_s), axis=1))  # lse = max_j lg_ij + log Σ_j exp(lg_s_ij) — логсуммаэксп по строке
        target_l = lg[np.arange(B), y_int]  # логит правильного класса target_i = lg_i,y_i
        reg_term = 0.5 * float(np.sum(reg[None, :] * (th**2)))  # штраф ½·Σ_j reg_j·Σ_c θ_cj²
        return float(np.mean(lse - target_l)) + reg_term  # CE = mean_i (lse_i − target_i) + штраф

    loss0 = ce_loss(theta)  # начальные потери
    cur_loss = loss0  # текущие потери
    accepted = held = 0  # счётчики принятых и удержанных шагов
    bf16_vec = np.vectorize(bf16_nearest_float)  # поточечное округление до BF16
    for _ in range(steps):  # до steps шагов
        lg = X_feat @ theta.T  # логиты текущих весов
        lg_s = lg - np.max(lg, axis=1, keepdims=True)  # логиты без максимума по строке
        exp_l = np.exp(lg_s)  # exp логитов
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)  # итог: softmax: p_ij = e_ij / Σ_c e_ic
        res = probs.copy()  # копия вероятностей — заготовка невязки
        res[np.arange(B), y_int] -= 1.0  # res = p − onehot(y): вычесть 1 в правильном классе
        g_bar = (res.T @ X_feat) / B + (reg[None, :] * theta)  # ḡ = res^T·X / B + reg·θ — средний градиент с регуляризацией
        D = -g_bar  # направление спуска D = −ḡ
        A = float(np.sum(g_bar * g_bar))  # A = Σ ḡ² — квадрат нормы градиента
        if A < 1e-18:  # градиент практически нулевой
            held += 1  # шаг удержан
            break  # стоп
        d_logits = X_feat @ D.T  # изменения логитов X·D^T
        r_prime = np.max(d_logits, axis=1) - np.min(d_logits, axis=1)  # размах по строке r'_i = max_j − min_j
        Q_prime = float(np.mean(r_prime * r_prime))  # итог: Q' = mean r'^2
        curv_reg = (  # кривизна регуляризации:
            2.0 * float(np.max(reg)) * float(np.sum(D * D)) if np.any(reg > 0) else 0.0  # 2·max(reg)·Σ D², если регуляризация есть, иначе 0
        )
        denom = Q_prime + 4.0 * curv_reg  # знаменатель Q' + 4·curv
        if denom <= 1e-18:  # знаменатель практически нулевой
            held += 1  # шаг удержан
            break  # стоп
        eta = 4.0 * A / denom  # длина шага η = 4A / знаменатель
        cert_drop = 2.0 * (A**2) / denom  # гарантированное убывание 2A² / знаменатель
        cand = theta + eta * D  # кандидат θ + η·D
        if use_bf16_guard:  # проверка округления BF16
            cand_bf = bf16_vec(cand)  # кандидат, округлённый до BF16
            lg_exact = X_feat @ cand.T  # точные логиты кандидата
            lg_bf = X_feat @ cand_bf.T  # логиты округлённого кандидата
            eps_pert = float(np.max(np.abs(lg_bf - lg_exact)))  # наибольшее возмущение логитов ε = max |lg_bf − lg_exact|
            if 4.0 * eps_pert >= cert_drop:  # 4ε не меньше гарантированного убывания — шаг не оправдан
                held += 1  # шаг удержан
                break  # стоп
            cand = cand_bf  # использовать округлённый кандидат
        cand_loss = ce_loss(cand)  # потери кандидата
        if cand_loss <= cur_loss + 1e-12:  # кандидат не хуже текущего (допуск 1e-12)
            theta = cand  # принять веса
            cur_loss = cand_loss  # обновить текущие потери
            accepted += 1  # принятых шагов + 1
        else:  # иначе
            held += 1  # шаг удержан
            break  # стоп
    return {  # вернуть словарь результатов
        "theta": theta,  # итоговые веса
        "loss0": loss0,  # начальные потери
        "loss_final": cur_loss,  # итоговые потери
        "total_drop": loss0 - cur_loss,  # суммарное убывание loss0 − текущие
        "accepted_steps": accepted,  # число принятых шагов
        "hold_steps": held,  # число удержанных шагов
    }


def newton_schulz_orthogonalize(G: np.ndarray, iters: int = 4) -> np.ndarray:
    """Ортогонализация Ньютона–Шульца: за iters итераций X ← X·(3E − X^T·X)/2 приближает матрицу к ортогональной.

    Матрица предварительно делится на норму Фробениуса и при необходимости
    транспонируется (строк не больше, чем столбцов); в конце вид восстанавливается."""
    X = G.astype(np.float64).copy()  # рабочая копия в float64
    norm = float(np.linalg.norm(X))  # норма Фробениуса
    if norm < 1e-12:  # норма практически нулевая
        return X  # вернуть как есть
    X /= norm  # нормировать: X = X / ‖X‖
    transposed = False  # признак транспонирования
    if X.shape[0] > X.shape[1]:  # строк больше, чем столбцов
        X = X.T  # транспонировать
        transposed = True  # запомнить
    I_m = np.eye(X.shape[0], dtype=np.float64)  # единичная матрица размера число строк
    for _ in range(iters):  # iters раз
        X = 0.5 * (3.0 * I_m - X @ X.T) @ X  # итог: X = ½·(3E − X·X^T)·X
    return X.T if transposed else X  # вернуть X^T, если транспонировали, иначе X


class ContinualGPMMemory(GpuGpmMixin):
    """Память последовательного обучения с защитой от забывания.

        Хранит веса W, базис U защищённого подпространства и вектор v фильтра
        EF21. Обучение задачи и консолидация подпространства на GPU выполняет
        hagi.base.gpu, если он доступен; процессорные реализации сохранены как
        методы *_cpu."""

    def __init__(self, d_in: int, d_out: int, seed: int = 0):  # d_in — входная размерность, d_out — выходная, seed — зерно генератора
        rng = np.random.default_rng(seed)  # генератор случайных чисел
        self.W = rng.normal(0.0, 0.1, size=(d_in, d_out))  # W ~ N(0, 0.1), форма (d_in, d_out)
        self.U_basis = np.zeros((d_in, 0), dtype=np.float64)  # базис защищённого подпространства U, изначально пустой (d_in, 0)
        self.ef_v = np.zeros_like(self.W)  # вектор EF21 v, изначально нулевой, той же формы, что W

    def snapshot_optimizer(self) -> dict[str, np.ndarray]:
        """Снимок состояния обучения: копии W, U_basis и ef_v."""
        return {  # словарь снимка
            "W": self.W.copy(),  # копия весов
            "U_basis": self.U_basis.copy(),  # копия базиса
            "ef_v": self.ef_v.copy(),  # копия вектора EF21
        }

    def restore_optimizer(self, snap: dict[str, np.ndarray]) -> None:
        """Восстановить W, U_basis и ef_v из снимка snap."""
        self.W = snap["W"].copy()  # веса из снимка
        self.U_basis = snap["U_basis"].copy()  # базис из снимка
        self.ef_v = snap["ef_v"].copy()  # вектор EF21 из снимка

    def project_orthogonal(self, G: np.ndarray) -> np.ndarray:
        """Спроецировать градиент G на подпространство, ортогональное базису U: (E − U·U^T)·G."""
        if self.U_basis.shape[1] == 0:  # базис пуст
            return G  # проекция не меняет G
        return G - self.U_basis @ (self.U_basis.T @ G)  # итог: G − U·(U^T·G)

    def _train_task_ef21_muon_cpu(
        self, X: np.ndarray, Y: np.ndarray, steps: int = 35, lr: float = 0.2  # self, примеры X, цели Y, число шагов и начальная длина шага
    ) -> float:  # возвращает средний квадрат ошибки
        """Обучить веса W на парах (X, Y) на процессоре; вернуть средний квадрат ошибки.

        err = X·W − Y; градиент grad = (2/N)·X^T·err проецируется ортогонально
        базису U; вектор EF21 v ← v + 0.5·(g_proj − v); направление —
        ортогонализация Ньютона–Шульца вектора v, снова спроецированная.
        Шаг W − η·step_dir принимается с первым η из (lr, lr/2, lr/10, 0.01),
        уменьшающим ошибку mean((X·W − Y)²)."""
        if len(X) == 0:  # нет примеров
            return 0.0  # ошибка 0
        for _ in range(steps):  # steps шагов
            err = X @ self.W - Y  # ошибка err = X·W − Y
            grad = (2.0 / len(X)) * (X.T @ err)  # градиент grad = (2/N)·X^T·err
            g_proj = self.project_orthogonal(grad)  # проекция градиента ортогонально базису
            diff = g_proj - self.ef_v  # разность diff = g_proj − v
            self.ef_v = self.ef_v + 0.5 * diff  # итог: v = v + 0.5·diff
            step_dir = newton_schulz_orthogonalize(self.ef_v, iters=4)  # направление шага — ортогонализация v
            step_dir = self.project_orthogonal(step_dir)  # снова спроецировать ортогонально базису
            if self.U_basis.shape[1] > 0:  # базис непуст
                drift = float(np.max(np.abs(self.U_basis.T @ step_dir)))  # максимальное смещение шага в защищённом подпространстве
                assert drift <= 1e-9, f"SpaceDecoupling violated: {drift}"  # проверка: смещение не больше 1e-9
            cur = float(np.mean((X @ self.W - Y) ** 2))  # текущая средняя квадратичная ошибка cur
            for eta in (lr, lr * 0.5, lr * 0.1, 0.01):  # по убывающим длинам шага η
                cand = self.W - eta * step_dir  # кандидат W − η·step_dir
                if float(np.mean((X @ cand - Y) ** 2)) < cur:  # ошибка кандидата меньше текущей
                    self.W = cand  # принять кандидата
                    break  # прекратить перебор η
        return float(np.mean((X @ self.W - Y) ** 2))  # вернуть средний квадрат ошибки

    def _consolidate_subspace_cpu(
        self, X: np.ndarray, energy_thresh: float = 0.995  # self, входы X и порог доли энергии
    ) -> None:  # ничего не возвращает
        """Пополнить базис U направлениями входов X, ортогональными уже защищённым.

        res_X — часть X вне span(U). Сингулярные направления берутся в порядке
        убывания сингулярных чисел, пока доля накопленной энергии
        Σ_{i≤k} S_i² / Σ S_i² не достигнет energy_thresh (при пороге ≥ 1.0 —
        все направления). Новые направления объединяются с U через SVD."""
        res_X = (  # остаток входов:
            X if self.U_basis.shape[1] == 0 else X - (X @ self.U_basis) @ self.U_basis.T  # X, если базис пуст, иначе X − (X·U)·U^T
        )
        if float(np.linalg.norm(res_X)) < 1e-10:  # остаток практически нулевой
            return  # нечего добавлять
        _, S, Vt = np.linalg.svd(res_X, full_matrices=False)  # сингулярное разложение остатка: A = U·diag(S)·Vt
        pos = S > 1e-10  # маска сингулярных чисел больше 1e-10
        if not np.any(pos):  # нет значимых сингулярных чисел
            return  # выйти
        S = S[pos]  # оставить только значимые сингулярные числа
        Vt = Vt[pos]  # и соответствующие правые сингулярные векторы
        tot = float(np.sum(S**2))  # полная энергия Σ S²
        cum = np.cumsum(S**2) / tot  # накопленная доля энергии cum_k = Σ_{i≤k} S_i² / полная
        # Накопленная сумма в float может остановиться на 0.99999…, и тогда
        # последние направления молча теряются — защищённое подпространство
        # становится неполным, и ошибка ранее обученной задачи растёт.
        # Порог energy_thresh ≥ 1.0 означает всё подпространство целиком.
        if energy_thresh >= 1.0:  # порог ≥ 1.0
            k = len(S)  # взять все направления: k = len(S)
        else:  # иначе
            k = min(len(S), int(np.searchsorted(cum, energy_thresh)) + 1)  # k = min(len(S), первый индекс cum ≥ порог + 1)
        new_dirs = Vt[:k].T  # новые направления — первые k правых сингулярных векторов как столбцы
        if self.U_basis.shape[1] == 0:  # базис пуст
            Q, _ = np.linalg.qr(new_dirs)  # ортогонализация новых направлений QR-разложением
            self.U_basis = Q  # базис = Q
        else:  # иначе
            combined = np.hstack([self.U_basis, new_dirs])  # объединить старый базис с новыми направлениями
            U_svd, s_svd, _ = np.linalg.svd(combined, full_matrices=False)  # SVD объединённой матрицы
            keep = s_svd > 1e-9  # оставить только значимые (s > 1e-9) направления
            self.U_basis = U_svd[:, keep]  # базис = соответствующие левые сингулярные векторы
        # Базис ограничен входной размерностью d_in: запас «−2» оставлял бы
        # защищённые направления без ограничения и допускал утечку забывания.
        if self.U_basis.shape[1] > self.W.shape[0]:  # столбцов больше, чем строк у W
            self.U_basis = self.U_basis[:, : self.W.shape[0]]  # обрезать базис до первых d_in столбцов


def quantize_bitalloc_exact(
    Phi: np.ndarray,  # матрица признаков (N, D)
    idx: np.ndarray,  # номера выбранных признаков
    wk: np.ndarray,  # их веса
    y: np.ndarray,  # цель
    sy: float,  # масштаб цели
    lam: float,  # L2-параметр
    max_extra_stages: int = 12,  # максимум дополнительных стадий
) -> tuple[np.ndarray, bytes]:  # возвращает квантованный вектор в размерности D и упакованные байты
    """Квантование вектора весов по выбранным признакам с точным байтовым кодеком: (вектор, байты).

    Phi — (N, D) признаки; idx — выбранные номера признаков; wk — их веса;
    y — цель; sy — масштаб цели; lam — L2-параметр. Веса сортируются по
    убыванию модуля, режутся на блоки по 5 и кодируются квантованием
    quantize_5block_exact. Затем до max_extra_stages раз: остаток
    r = y − Sub·w_q решает задачу dw = A^−1·Sub^T·r, A = Sub^T·Sub +
    max(lam, 1e-9)·E; выбирается блок с наибольшей ошибкой
    Σ dw_i²·energy_i и его поправка добавляется. Останавливается при
    mean((r·sy)²) < 0.005 или малой ошибке блока."""
    if idx.size == 0:  # признаки не выбраны
        return np.zeros(Phi.shape[1], dtype=np.float64), b"\x00\x00\x00"  # нулевой вектор и заголовок из трёх нулевых байтов
    mag_order = np.argsort(-np.abs(wk))  # порядок убывания модуля весов
    idx_sorted = idx[mag_order]  # номера признаков в этом порядке
    Sub_s = Phi[:, idx_sorted]  # подматрица признаков Sub_s = Phi[:, idx_sorted]
    col_energy = np.mean(Sub_s**2, axis=0)  # средняя энергия столбца energy_j = mean_i Sub_s_ij²
    A = Sub_s.T @ Sub_s + max(lam, 1e-9) * np.eye(idx.size)  # итог: A = Sub_s^T·Sub_s + max(lam, 1e-9)·E
    k = idx.size  # k — число выбранных признаков
    n_blocks = (k + 4) // 5  # число блоков ⌈k / 5⌉
    wq_s = np.zeros(k, dtype=np.float64)  # квантованные веса по k признакам, изначально нули
    dw = wk[mag_order].copy()  # рабочая копия весов dw в порядке убывания модуля
    records: list[tuple[int, tuple[int, int, int]]] = []  # список записей (номер блока, (exp2, best, bval))
    for b in range(n_blocks):  # по блокам
        sl = slice(b * 5, min((b + 1) * 5, k))  # срез блока: [5b, min(5(b + 1), k))
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))  # блок, дополненный нулями до 5 элементов
        q_blk, meta = quantize_5block_exact(blk)  # квантовать блок
        wq_s[sl] += q_blk[: sl.stop - sl.start]  # прибавить квантованные веса (без дополнения)
        records.append((b, meta))  # запомнить запись блока
    for _ in range(max_extra_stages):  # до max_extra_stages дополнительных стадий
        r_data = y - Sub_s @ wq_s  # остаток данных r = y − Sub_s·w_q
        if float(np.mean((r_data * sy) ** 2)) < 0.005:  # масштабная средняя мощность остатка < 0.005 — стоп
            break  # выйти из цикла стадий
        dw = np.linalg.solve(A, Sub_s.T @ r_data)  # поправка dw = A^−1·(Sub_s^T·r)
        blk_errs = [  # ошибки блоков:
            float(  # ошибка блока b как число
                np.sum(  # сумма по блоку
                    (dw[b * 5 : min((b + 1) * 5, k)] ** 2)  # квадратов поправок dw_i²
                    * col_energy[b * 5 : min((b + 1) * 5, k)]  # умноженных на энергию столбца energy_i
                )
            )
            for b in range(n_blocks)  # по блокам
        ]
        best_b = int(np.argmax(blk_errs))  # номер блока с наибольшей ошибкой
        if blk_errs[best_b] * (sy * sy) < 1e-6:  # ошибка блока в масштабе цели < 1e-6 — стоп
            break  # выйти из цикла стадий
        sl = slice(best_b * 5, min((best_b + 1) * 5, k))  # срез лучшего блока
        blk = np.pad(dw[sl], (0, 5 - (sl.stop - sl.start)))  # блок, дополненный нулями до 5 элементов
        q_blk, meta = quantize_5block_exact(blk)  # квантовать блок
        wq_s[sl] += q_blk[: sl.stop - sl.start]  # прибавить поправку к квантованным весам
        records.append((best_b, meta))  # запомнить запись блока
    packed = pack_bitalloc_records(k, idx_sorted, records)  # упаковать k, номера признаков и записи
    rec = unpack_bitalloc_records(packed, Phi.shape[1])  # раскодировать обратно в размерность Phi для проверки
    return rec, packed  # вернуть раскодированный вектор и упакованные байты


def _omp_candidates(
    Phi: np.ndarray, y: np.ndarray, max_k: int = 16, lam: float = 1e-7  # Phi — (N, D), максимум признаков, L2-параметр
) -> list[tuple[np.ndarray, np.ndarray]]:  # возвращает список пар (номера признаков, веса)
    """Кандидаты жадного выбора признаков (пошаговое ортогональное согласование): список (idx, wk).

    На каждом шаге берётся нормированный столбец Phi_n с наибольшей
    |корреляцией| с остатком r; веса wk = A^−1·Sub^T·y, A = Sub^T·Sub +
    lam·E; остаток r = y − Sub·wk. Не более max_k шагов; стоп при
    корреляции ≤ 1e-12 или mean(r²) < 1e-9."""
    N, D = Phi.shape  # N, D — размеры матрицы признаков
    norms = np.sqrt(np.sum(Phi**2, axis=0)) + 1e-12  # нормы столбцов + 1e-12 для защиты от нуля
    Phi_n = Phi / norms[None, :]  # нормированные признаки Phi_n = Phi / нормы
    r = y.copy()  # остаток r = y
    selected: list[int] = []  # выбранные номера признаков по порядку
    sel_set: set[int] = set()  # множество выбранных
    cands: list[tuple[np.ndarray, np.ndarray]] = []  # список кандидатов
    for _ in range(min(max_k, D, max(1, N - 1))):  # не более min(max_k, D, max(1, N − 1)) шагов
        corrs = np.abs(Phi_n.T @ r)  # модули корреляций |Phi_n^T·r|
        for s in sel_set:  # по уже выбранным
            corrs[s] = -1.0  # исключить: корреляция −1
        j = int(np.argmax(corrs))  # номер столбца с наибольшей корреляцией
        if corrs[j] <= 1e-12:  # корреляция практически нулевая
            break  # стоп
        selected.append(j)  # добавить признак в порядок
        sel_set.add(j)  # в множество
        idx = np.array(selected, dtype=np.int64)  # массив выбранных номеров
        Sub = Phi[:, idx]  # подматрица выбранных признаков
        A = Sub.T @ Sub + lam * np.eye(len(idx))  # итог: A = Sub^T·Sub + lam·E
        wk = np.linalg.solve(A, Sub.T @ y)  # итог: wk = A^−1·(Sub^T·y)
        r = y - Sub @ wk  # новый остаток r = y − Sub·wk
        cands.append((idx.copy(), wk.copy()))  # запомнить кандидата (копии idx и wk)
        if float(np.mean(r * r)) < 1e-9:  # средний квадрат остатка < 1e-9
            break  # стоп
    return cands  # вернуть список кандидатов


def _press_score_ridge(
    Phi_sub: np.ndarray,  # подматрица выбранных признаков
    wk: np.ndarray,  # веса ридж-регрессии
    y: np.ndarray,  # целевая переменная в исходном масштабе
    ty: np.ndarray,  # целые цели
    sy: float,  # масштаб цели
    lam: float,  # L2-параметр
) -> tuple[float, float]:  # возвращает (ошибка оставь один, точность)
    """Оценка модели по «оставь один» (PRESS) через диагональ матрицы проекции: (ошибка, точность).

    wk — ридж-веса (A = Phi_sub^T·Phi_sub + max(lam, 1e-8)·E);
    ty — целые цели; sy — масштаб цели. h_i = diag(Phi_sub·A^−1·Phi_sub^T);
    прогноз без i-го примера loo_i = ty_i − resid_i / max(0.08, 1 − h_i),
    resid = ty − Phi_sub·wk·sy. Ошибка = mean((loo − ty)²) + 1e-3·Σwk²·sy² / max(1, N);
    точность = доля round(loo) = ty."""
    N, D = Phi_sub.shape  # N, D — размеры подматрицы признаков
    pred = (Phi_sub @ wk) * sy  # прогноз (Phi_sub·wk)·sy
    resid = ty - pred  # невязка resid = ty − прогноз
    if D >= N:  # признаков не меньше, чем примеров — оценка оставь один невозможна
        return float(np.mean(resid**2)) * 100.0 + float(np.sum(wk**2)), 0.0  # штраф 100·mean(resid²) + Σwk², точность 0
    A = Phi_sub.T @ Phi_sub + max(lam, 1e-8) * np.eye(D)  # итог: A = Phi_sub^T·Phi_sub + max(lam, 1e-8)·E
    try:  # решение может не существовать
        A_inv_Pt = np.linalg.solve(A, Phi_sub.T)  # итог: A^−1·Phi_sub^T
    except np.linalg.LinAlgError:  # вырожденная матрица
        return float(np.mean(resid**2)) * 100.0, 0.0  # штраф 100·mean(resid²), точность 0
    h = np.sum(Phi_sub * A_inv_Pt.T, axis=1)  # итог: h_i = diag(Phi_sub·A^−1·Phi_sub^T) = Σ_j Phi_ij·(A^−1·Phi^T)_ji
    denom = np.maximum(0.08, 1.0 - h)  # знаменатель max(0.08, 1 − h_i)
    loo_pred = ty - resid / denom  # loo = ty − resid / знаменатель
    w_norm_pen = 1e-3 * float(np.sum(wk**2)) * (sy**2) / max(1, N)  # штраф за веса 1e-3·Σwk²·sy² / max(1, N)
    loo_mse = float(np.mean((loo_pred - ty) ** 2)) + w_norm_pen  # средний квадрат ошибки loo + штраф
    loo_acc = float(np.mean(np.rint(loo_pred) == ty))  # доля точных округлённых прогнозов
    return loo_mse, loo_acc  # вернуть ошибку и точность


class FactorizedSharedCore:
    """Разделяемое ядро малого объёма: случайные признаки (Адамар + SwiGLU) и память GPM.

    Ветви: poly — мономы; hinge — ломаные признаки; bit — биты и их
    произведения; core — нелинейные признаки нейронного слоя. Веса слоёв
    строятся из ортогональных матриц, поэтому ядро занимает 8 байт — зерно
    генератора и ширина."""

    BRANCHES = ("poly", "hinge", "bit", "core")  # имена ветвей признаков: полиномиальная, ломаная, битовая, ядро

    def __init__(self, seed: int = 42, width: int = 48):  # seed — зерно, width — ширина слоя
        self.seed = seed  # запомнить зерно
        self.width = width  # запомнить ширину
        rng = np.random.default_rng(seed)  # генератор случайных чисел
        Q1, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, 8)))  # Q1 — ортогональная матрица (width, 8) из QR-разложения случайной
        scales = np.repeat([0.6, 1.2, 2.4, 4.8], width // 4)  # масштабы 0.6, 1.2, 2.4, 4.8, повторённые width/4 раз
        self.W1 = Q1.T * scales[None, :]  # W1 = Q1^T·diag(масштабы), форма (8, width)
        self.b1 = rng.uniform(-1.5, 1.5, (1, width))  # b1 — смещение, равномерное на [−1.5, 1.5], форма (1, width)
        Q2, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, width)))  # Q2 — ортогональная матрица (width, width)
        self.W2a = Q2 * 1.3  # W2a = Q2·1.3 — вес ветви значения SwiGLU
        self.b2a = rng.uniform(-0.8, 0.8, (1, width))  # b2a — её смещение, равномерное на [−0.8, 0.8]
        Q2g, _ = np.linalg.qr(rng.normal(0.0, 1.0, (width, width)))  # Q2g — вторая ортогональная матрица (width, width)
        self.W2g = Q2g * 1.3  # W2g = Q2g·1.3 — вес ветви gating SwiGLU
        self.b2g = rng.uniform(-0.8, 0.8, (1, width))  # b2g — её смещение, равномерное на [−0.8, 0.8]
        self.H8 = sylvester_hadamard(3)  # H8 — нормированная матрица Адамара порядка 8
        self.gpm = ContinualGPMMemory(d_in=16, d_out=1, seed=seed + 7)  # память GPM на 16 входов и 1 выход
        self.core_bytes = 8  # объём ядра — 8 байт

    def snapshot(self) -> dict[str, Any]:
        """Снимок состояния ядра: снимок памяти GPM и объём ядра."""
        return {"gpm": self.gpm.snapshot_optimizer(), "core_bytes": self.core_bytes}  # словарь снимка

    def restore(self, snap: dict[str, Any]) -> None:
        """Восстановить ядро из снимка snap."""
        self.gpm.restore_optimizer(snap["gpm"])  # восстановить память GPM
        self.core_bytes = snap["core_bytes"]  # восстановить объём ядра

    def features(self, branch: str, xs_arr: np.ndarray, sx: float) -> np.ndarray:
        """Матрица признаков ветви branch для входов xs_arr при масштабе sx.

        branch — одна из BRANCHES. Возвращает (N, число признаков) float64."""
        N, d = xs_arr.shape  # N — число входов, d — число аргументов
        xn = xs_arr.astype(np.float64) / sx  # нормированные входы xn = xs / sx
        step1 = 1.0 / sx  # шаг одного целого в нормированной шкале 1 / sx
        if branch == "poly":  # ветвь poly — мономы
            cols = [xn[:, i] for i in range(d)]  # столбцы-переменные
            fs = [np.ones((N, 1), dtype=np.float64)]  # начать с столбца единиц
            for i in range(d):  # по переменным
                fs.append(cols[i][:, None])  # добавить линейный член x_i
            for i in range(d):  # по парам i ≤ j
                for j in range(i, d):  # по j от i до d−1
                    fs.append((cols[i] * cols[j])[:, None])  # добавить квадратичный член x_i·x_j
            for i in range(d):  # по тройкам i ≤ j ≤ k
                for j in range(i, d):  # по j
                    for k in range(j, d):  # по k
                        fs.append((cols[i] * cols[j] * cols[k])[:, None])  # добавить кубический член x_i·x_j·x_k
            if d <= 2:  # арность не больше 2 — добавить и четвёртые степени
                for i in range(d):  # по i
                    for j in range(i, d):  # по j
                        for k in range(j, d):  # по k
                            for l in range(k, d):  # по l
                                fs.append(  # добавить член
                                    (cols[i] * cols[j] * cols[k] * cols[l])[:, None]  # итог: x_i·x_j·x_k·x_l
                                )
            else:  # иначе
                for i in range(d):  # по переменным
                    fs.append((cols[i] ** 4)[:, None])  # добавить только x_i^4
            return np.hstack(fs)  # соединить столбцы в матрицу
        if branch == "hinge":  # ветвь hinge — ломаные признаки
            fs = [np.ones((N, 1), dtype=np.float64)]  # начать с столбца единиц
            for i in range(d):  # по переменным
                fs.append(xn[:, i : i + 1])  # добавить x_i
                fs.append(np.abs(xn[:, i : i + 1]))  # добавить |x_i|
                for knot in (-0.375, -step1, 0.0, step1, 0.375):  # по узлам излома −0.375, −шаг, 0, шаг, 0.375
                    fs.append(np.maximum(0.0, xn[:, i] - knot)[:, None])  # добавить max(0, x_i − узел)
                    fs.append(np.maximum(0.0, knot - xn[:, i])[:, None])  # добавить max(0, узел − x_i)
                fs.append(np.maximum(0.0, 1.0 - sx * np.abs(xn[:, i : i + 1])))  # добавить max(0, 1 − sx·|x_i|) — индикатор близости к целому
            for i in range(d):  # по парам разных переменных i < j
                for j in range(i + 1, d):  # по j > i
                    diff = xn[:, i] - xn[:, j]  # итог: diff = x_i − x_j
                    fs.append(np.abs(diff)[:, None])  # добавить |diff|
                    for knot in (-step1, 0.0, step1):  # по узлам −шаг, 0, шаг
                        fs.append(np.maximum(0.0, diff - knot)[:, None])  # добавить max(0, diff − узел)
                        fs.append(np.maximum(0.0, knot - diff)[:, None])  # добавить max(0, узел − diff)
                    fs.append(np.maximum(0.0, 1.0 - sx * np.abs(diff))[:, None])  # добавить max(0, 1 − sx·|diff|)
            return np.hstack(fs)  # соединить столбцы в матрицу
        if branch == "bit":  # ветвь bit — битовые признаки
            xi = np.rint(xs_arr).astype(np.int64)  # входы, округлённые до целых
            fs = [np.ones((N, 1), dtype=np.float64)]  # начать с столбца единиц
            for i in range(d):  # по переменным
                fs.append((xi[:, i] < 0).astype(np.float64)[:, None])  # добавить признак x_i < 0
                for b in range(8):  # по битам b = 0..7
                    fs.append(((xi[:, i] >> b) & 1).astype(np.float64)[:, None])  # добавить бит b числа x_i: (x_i >> b) & 1
            if d >= 2:  # арность не меньше 2 — парные битовые признаки
                for b in range(6):  # по младшим 6 битам
                    b0 = ((xi[:, 0] >> b) & 1).astype(np.float64)[:, None]  # бит b первого аргумента
                    b1 = ((xi[:, 1] >> b) & 1).astype(np.float64)[:, None]  # бит b второго аргумента
                    fs.append(b0 * b1)  # добавить И: b0·b1
                    fs.append(np.abs(b0 - b1))  # добавить XOR: |b0 − b1|
                fs.append(((xi[:, 0] < 0) * (xi[:, 1] < 0)).astype(np.float64)[:, None])  # добавить «оба отрицательные»
                fs.append(((xi[:, 0] < 0) ^ (xi[:, 1] < 0)).astype(np.float64)[:, None])  # добавить «в точности одно отрицательное»
            if d == 3:  # арность 3 — произведения младших бит
                b0 = (xi[:, 0] & 1).astype(np.float64)[:, None]  # младший бит x_0
                b1 = (xi[:, 1] & 1).astype(np.float64)[:, None]  # младший бит x_1
                b2 = (xi[:, 2] & 1).astype(np.float64)[:, None]  # младший бит x_2
                fs.extend([b0 * b2, b1 * b2, b0 * b1 * b2])  # добавить b0·b2, b1·b2, b0·b1·b2
            return np.hstack(fs)  # соединить столбцы в матрицу
        xp = np.zeros((N, 8), dtype=np.float64)  # ветвь core — восемь смесей входов xp, изначально нули
        xp[:, : min(3, d)] = xn[:, : min(3, d)]  # первые min(3, d) компонент — сами переменные
        xp[:, 3] = xn[:, 0] - xn[:, 1] if d >= 2 else np.abs(xn[:, 0])  # xp_3 = x_0 − x_1 при d ≥ 2, иначе |x_0|
        xp[:, 4] = xn[:, 1] - xn[:, 2] if d >= 3 else np.maximum(0.0, xn[:, 0])  # xp_4 = x_1 − x_2 при d ≥ 3, иначе max(0, x_0)
        xp[:, 5] = xn[:, 0] - xn[:, 2] if d >= 3 else np.maximum(0.0, -xn[:, 0])  # xp_5 = x_0 − x_2 при d ≥ 3, иначе max(0, −x_0)
        xp[:, 6] = np.abs(xp[:, 3])  # итог: xp_6 = |xp_3|
        xp[:, 7] = np.abs(xp[:, 4])  # итог: xp_7 = |xp_4|
        xh = xp @ self.H8  # поворот Адамара: xh = xp·H8
        z1 = xh @ self.W1 + self.b1  # z1 = xh·W1 + b1, форма (N, width)
        h1_t = np.tanh(z1)  # ветвь tanh: tanh(z1)
        h1_r = np.maximum(0.0, z1)  # ветвь ReLU: max(0, z1)
        z2a = h1_r @ self.W2a + self.b2a  # линейный слой значения: z2a = h1_r·W2a + b2a, форма (N, width)
        z2g = h1_r @ self.W2g + self.b2g  # линейный слой вентиля: z2g = h1_r·W2g + b2g
        sig2g = 1.0 / (1.0 + np.exp(-np.clip(z2g, -15.0, 15.0)))  # сигмоида с обрезкой аргумента: sig2g = 1 / (1 + exp(−clip(z2g, −15, 15)))
        h2_sg = z2a * sig2g  # вентилируемый слой: h2_sg = z2a·sig2g
        four = []  # пустой список гармонических признаков
        for f in (1.0, 2.0, 4.0):  # по частотам f = 1, 2, 4
            for i in range(min(3, d)):  # по первым min(3, d) переменным
                four.append(np.sin(math.pi * f * xn[:, i : i + 1]))  # добавить sin(π·f·x_i)
                four.append(np.cos(math.pi * f * xn[:, i : i + 1]))  # добавить cos(π·f·x_i)
        return np.hstack(  # вернуть конкатенацию всех блоков признаков по столбцам
            [  # начало списка блоков
                self.features("poly", xs_arr, sx),  # полиномиальные признаки
                self.features("hinge", xs_arr, sx)[:, 1:],  # ломаные признаки без столбца единиц
                h1_t,  # tanh-ветвь скрытого слоя
                h1_r,  # ReLU-ветвь скрытого слоя
                h2_sg,  # вентилируемый второй слой
            ]  # конец списка блоков
            + four  # плюс гармонические признаки
        )

    def _mlp_encode(self, A: np.ndarray, sx: float) -> np.ndarray:
        """Матрица входов корректора-перцептрона по строкам A.

            A: (N, d) целочисленные аргументы, sx: масштаб входа.
            Столбцы: a_i = A_i / sx для каждой переменной, |a_i|, столбец единиц,
            а при d ≥ 2 для всех пар i < j — разности a_i − a_j и |a_i − a_j|.
            Возвращает (N, 2d + 1 + d(d−1)) float64."""
        an = A / sx  # нормировка: an = A / sx
        d = A.shape[1]  # d — число переменных (арность)
        cols = [an, np.abs(an), np.ones((A.shape[0], 1))]  # начать с a_i, |a_i| и столбца единиц
        if d >= 2:  # парные признаки нужны только при d ≥ 2
            for i in range(d):  # по переменным i
                for j in range(i + 1, d):  # по j > i
                    cols.append(an[:, i : i + 1] - an[:, j : j + 1])  # добавить разность a_i − a_j
                    cols.append(np.abs(an[:, i : i + 1] - an[:, j : j + 1]))  # добавить |a_i − a_j|
        return np.hstack(cols)  # соединить столбцы в матрицу (N, число признаков)

    def _fit_stage2_mlp(
        self,  # экземпляр
        tx: np.ndarray,  # (N, d) обучающие аргументы
        resid_n: np.ndarray,  # (N,) остаток в нормированных единицах y / sy
        vx: np.ndarray,  # (M, d) проверочные аргументы
        sx: float,  # масштаб входа
        hidden: int = 12,  # ширина скрытого слоя по умолчанию 12
        steps: int = 280,  # число шагов обучения по умолчанию 280
        seed: int = 7,  # зерно генератора по умолчанию 7
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:  # возвращает (прогноз на train, прогноз на val, словарь весов)
        """Обучить двухслойный перцептрон, предсказывающий остаток основной модели.

            Сеть: h = tanh(X·W1 + b1), pred = h·W2 + b2, потеря MSE = mean((pred − r)^2).
            tx: (N, d) обучающие аргументы, resid_n: (N,) остаток в нормированных
            единицах y / sy, vx: (M, d) проверочные аргументы, sx: масштаб входа,
            hidden: ширина скрытого слоя, steps: число шагов, seed: зерно генератора.
            W1 обновляется методом MUON (момент, ортогонализация Ньютона–Шульца,
            шаг 0.7·lr), W2, b1, b2 — Adam. Шаг lr затухает по косинусу:
            lr_t = lr0·0.5·(1 + cos(π·t/steps)), lr0 = 0.04. Хранится лучший по MSE
            набор весов. В конце все веса слитно поворачиваются Адамаром, квантуются
            dyadic5 и разворачиваются обратно. Возвращает (pred_train, pred_val,
            словарь квантованных весов и упакованных байтов)."""
        rng = np.random.default_rng(seed)  # генератор случайных чисел с зерном seed
        Xin = self._mlp_encode(tx, sx)  # матрица входов обучающей выборки
        Xvin = self._mlp_encode(vx, sx)  # матрица входов проверочной выборки
        N, din = Xin.shape  # N — число примеров, din — число входных признаков
        W1 = rng.normal(0.0, 1.0 / math.sqrt(din), (din, hidden))  # W1 ~ N(0, 1/√din), форма (din, hidden)
        b1 = np.zeros((1, hidden))  # смещение скрытого слоя — нули
        W2 = np.zeros((hidden, 1))  # веса выхода — нули (первый прогноз равен 0)
        b2 = np.zeros((1, 1))  # смещение выхода — нули
        yn = resid_n.reshape(-1, 1)  # целевой остаток в столбец (N, 1)
        m1 = np.zeros_like(W1)  # момент первого порядка для W1 (нулевая матрица той же формы)
        m2 = np.zeros_like(W2)  # момент первого порядка для W2
        v2 = np.zeros_like(W2)  # момент второго порядка для W2
        mb1 = np.zeros_like(b1)  # момент первого порядка для b1
        vb1 = np.zeros_like(b1)  # момент второго порядка для b1
        mb2 = np.zeros_like(b2)  # момент первого порядка для b2
        vb2 = np.zeros_like(b2)  # момент второго порядка для b2
        best = (float(np.mean(yn**2)), W1.copy(), b1.copy(), W2.copy(), b2.copy())  # старт: MSE при нулевом прогнозе = mean(yn^2) и нулевые веса
        lr0 = 0.04  # начальная скорость обучения lr0
        for t in range(1, steps + 1):  # по шагам t = 1..steps
            cos_lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * t / steps))  # косинусное затухание: cos_lr = lr0·0.5·(1 + cos(π·t/steps))
            h = np.tanh(Xin @ W1 + b1)  # скрытый слой: h = tanh(Xin·W1 + b1)
            pred = h @ W2 + b2  # прогноз: pred = h·W2 + b2
            err = pred - yn  # ошибка: err = pred − yn
            loss = float(np.mean(err**2))  # потеря: loss = mean(err^2)
            if loss < best[0]:  # потеря меньше рекордной
                best = (loss, W1.copy(), b1.copy(), W2.copy(), b2.copy())  # запомнить рекорд и копии весов
            dp = (2.0 / N) * err  # градиент по pred: dp = (2/N)·err
            gW2 = h.T @ dp + 2e-4 * W2  # градиент W2: hᵀ·dp + 2e-4·W2 (L2-регуляризация)
            gb2 = np.sum(dp, axis=0, keepdims=True)  # градиент b2: Σ_n dp_n
            dh = (dp @ W2.T) * (1.0 - h * h)  # градиент по h: dp·W2ᵀ, умножённый на производную tanh 1 − h^2
            gW1 = Xin.T @ dh + 2e-4 * W1  # градиент W1: Xinᵀ·dh + 2e-4·W1
            gb1 = np.sum(dh, axis=0, keepdims=True)  # градиент b1: Σ_n dh_n
            m1 = 0.85 * m1 + 0.15 * gW1  # момент MUON: m1 = 0.85·m1 + 0.15·gW1
            W1 -= (cos_lr * 0.7) * newton_schulz_orthogonalize(m1, iters=3)  # W1 -= 0.7·cos_lr·orth(m1), orth — ортогонализация Ньютона–Шульца (3 итерации)
            m2 = 0.9 * m2 + 0.1 * gW2  # метод Adam: m2 = 0.9·m2 + 0.1·gW2
            v2 = 0.999 * v2 + 0.001 * (gW2 * gW2)  # метод Adam: v2 = 0.999·v2 + 0.001·gW2^2
            W2 -= cos_lr * (m2 / (1 - 0.9**t)) / (np.sqrt(v2 / (1 - 0.999**t)) + 1e-8)  # метод Adam: W2 -= cos_lr·(m2/(1−0.9^t)) / (√(v2/(1−0.999^t)) + 1e-8)
            mb1 = 0.9 * mb1 + 0.1 * gb1  # метод Adam: mb1 = 0.9·mb1 + 0.1·gb1
            vb1 = 0.999 * vb1 + 0.001 * (gb1 * gb1)  # метод Adam: vb1 = 0.999·vb1 + 0.001·gb1^2
            b1 -= cos_lr * (mb1 / (1 - 0.9**t)) / (np.sqrt(vb1 / (1 - 0.999**t)) + 1e-8)  # метод Adam: b1 -= cos_lr·(mb1/(1−0.9^t)) / (√(vb1/(1−0.999^t)) + 1e-8)
            mb2 = 0.9 * mb2 + 0.1 * gb2  # метод Adam: mb2 = 0.9·mb2 + 0.1·gb2
            vb2 = 0.999 * vb2 + 0.001 * (gb2 * gb2)  # метод Adam: vb2 = 0.999·vb2 + 0.001·gb2^2
            b2 -= cos_lr * (mb2 / (1 - 0.9**t)) / (np.sqrt(vb2 / (1 - 0.999**t)) + 1e-8)  # метод Adam: b2 -= cos_lr·(mb2/(1−0.9^t)) / (√(vb2/(1−0.999^t)) + 1e-8)
        _, W1, b1, W2, b2 = best  # взять лучший набор весов
        all_w = np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])  # все веса одним вектором
        wr, orig_n = block_hadamard_rotate(all_w, 8)  # повёрнутый вектор wr (блоки по 8) и исходная длина orig_n
        packed_mlp, wr_q = pack_vector_dyadic5(wr, stages=2)  # квантование dyadic5 (2 стадии): байты packed_mlp и квантованный вектор wr_q
        all_wq = block_hadamard_unrotate(wr_q, orig_n, 8)  # обратный поворот и обрезка до orig_n: квантованные веса
        p1 = W1.size  # граница W1: p1 = число элементов W1
        p2 = p1 + b1.size  # граница b1: p2 = p1 + число элементов b1
        p3 = p2 + W2.size  # граница W2: p3 = p2 + число элементов W2
        W1q = all_wq[:p1].reshape(W1.shape)  # W1q — первые p1 элементов в форме W1
        b1q = all_wq[p1:p2].reshape(b1.shape)  # b1q — срез p1..p2−1 в форме b1
        W2q = all_wq[p2:p3].reshape(W2.shape)  # W2q — срез p2..p3−1 в форме W2
        b2q = all_wq[p3:].reshape(b2.shape)  # b2q — остаток с p3 в форме b2
        return (  # вернуть прогнозы и веса
            (np.tanh(Xin @ W1q + b1q) @ W2q + b2q).ravel(),  # прогноз на train: tanh(Xin·W1q + b1q)·W2q + b2q, развёрнутый в вектор
            (np.tanh(Xvin @ W1q + b1q) @ W2q + b2q).ravel(),  # прогноз на val: tanh(Xvin·W1q + b1q)·W2q + b2q, развёрнутый в вектор
            {"W1": W1q, "b1": b1q, "W2": W2q, "b2": b2q, "packed": packed_mlp},  # словарь квантованных весов и упакованных байтов
        )

    def fit_residual(
        self,  # экземпляр
        train_x: Sequence[Sequence[int]],  # обучающие аргументы (строки целых)
        train_y: Sequence[int],  # обучающие значения
        val_x: Sequence[Sequence[int]],  # проверочные аргументы
        val_y: Sequence[int],  # проверочные значения
        router: Optional["HedgeRouter"] = None,  # маршрутизатор ветвей (задаёт порядок перебора и получает потери)
    ) -> "FactorizedAdapter":  # возвращает обученный адаптер
        """Обучить компактный адаптер, предсказывающий y по целочисленным аргументам.
        
                train_x/train_y: обучающие пары, val_x/val_y: проверочные пары (пустые — берутся
                обучающие), router: маршрутизатор ветвей. Масштабы: sx = max(1, max|train_x|),
                sy = max(1, max|train_y|), y = train_y / sy.
        
                Перебираются ветви признаков Phi. Для трёх λ ∈ {1e-6, 1e-3, 1e-1} отбор OMP
                даёт не более max_k весов wk; кандидат с max|wk| > 8 отбрасывается. Цель:
                f = 0.4·mse_train + 0.6·mse_loo + 0.002·nb + 0.5·(1 − 0.5·(acc_train + acc_loo)),
                где mse_loo/acc_loo — оценка leave-one-out по PRESS, nb = 9 + k + 3·⌈k/5⌉,
                k — число признаков. У трёх лучших кандидатов веса квантуются (nb = 6 + длина packed).
        
                При 2..5 классах и N ≥ 12 дополнительно обучается классификатор с перекрёстной
                энтропией на не более 4 ломаных признаках. Если регрессия неточна, на её остатке
                обучается корректор-перцептрон (режим rvq2, если он снижает проверочную MSE
                более чем на 28%). Возвращает FactorizedAdapter с метриками на val."""
        tx = np.asarray(train_x, dtype=np.float64)  # обучающие аргументы в массив float64
        ty = np.asarray(train_y, dtype=np.float64)  # обучающие значения в массив float64
        vx = np.asarray(val_x, dtype=np.float64) if val_x else tx  # проверочные аргументы; если пусто — обучающие
        vy = np.asarray(val_y, dtype=np.float64) if val_y else ty  # проверочные значения; если пусто — обучающие
        sx = max(1.0, float(np.max(np.abs(tx))))  # масштаб входа: sx = max(1, max|tx|)
        sy = max(1.0, float(np.max(np.abs(ty))))  # масштаб выхода: sy = max(1, max|ty|)
        y = ty / sy  # нормированная цель: y = ty / sy
        N, d = tx.shape  # N — число примеров, d — арность

        gpm_in = self.features("poly", tx, sx)  # полиномиальные признаки обучающей выборки
        if gpm_in.shape[1] < 16:  # столбцов меньше 16
            gpm_in = np.pad(gpm_in, ((0, 0), (0, 16 - gpm_in.shape[1])))  # дополнить нулями справа до 16 столбцов
        else:  # столбцов не меньше 16
            gpm_in = gpm_in[:, :16]  # взять первые 16 столбцов
        self.gpm.train_task_ef21_muon(gpm_in, y.reshape(-1, 1), steps=15, lr=0.15)  # обучить гейт-модель gpm: 15 шагов, скорость 0.15
        self.gpm.consolidate_subspace(gpm_in, energy_thresh=0.995)  # закрепить подпространство, объясняющее 99.5% энергии

        branch_order = list(self.BRANCHES)  # порядок перебора ветвей — исходный список
        if router is not None:  # маршрутизатор задан
            branch_order = [self.BRANCHES[i] for i in np.argsort(-router.weights)]  # ветви по убыванию веса: argsort(−weights)

        best = (float("inf"), "poly", "reg", None, None, b"", None, None, None)  # рекорд: (цель, ветвь, режим, веса, байты, байты, метаданные CE, прогноз train/sy, метаданные MLP)
        branch_losses = np.ones(len(self.BRANCHES), dtype=np.float64)  # потери ветвей, изначально 1

        for bname in branch_order:  # по ветвям в выбранном порядке
            Phi = self.features(bname, tx, sx)  # матрица признаков Phi (N, число признаков)
            max_k = min(14, max(2, N // 2) if N >= 20 else max(1, min(8, N - 2)))  # предел числа признаков: min(14, max(2, N/2)) при N ≥ 20, иначе min(8, max(1, N − 2))
            b_best_acc = 0.0  # лучшая LOO-точность ветви, изначально 0
            float_cands = []  # список кандидатов с вещественными весами
            for lam in (1e-6, 1e-3, 1e-1):  # по коэффициентам регуляризации λ
                for idx, wk in _omp_candidates(Phi, y, max_k=max_k, lam=lam):  # по парам (набор признаков idx, веса wk) от OMP
                    if np.max(np.abs(wk)) > 8.0:  # наибольший |вес| больше 8
                        continue  # отклонить кандидата
                    pred_f = (Phi[:, idx] @ wk) * sy  # прогноз: pred_f = (Phi_idx·wk)·sy
                    mse_tr_f = float(np.mean((pred_f - ty) ** 2))  # MSE на train: mean((pred_f − ty)^2)
                    acc_tr_f = float(np.mean(np.rint(pred_f) == ty))  # точность на train: доля round(pred_f) = ty
                    mse_loo, acc_loo = _press_score_ridge(  # оценка leave-one-out по PRESS: MSE и точность
                        Phi[:, idx], wk, y, ty, sy, lam=lam  # по выбранным признакам, весам и λ
                    )
                    b_best_acc = max(b_best_acc, acc_loo)  # обновить лучшую LOO-точность ветви
                    est_nb = 9 + len(idx) + ((len(idx) + 4) // 5) * 3  # оценка байтов: 9 + k + 3·⌈k/5⌉, k = len(idx)
                    f_obj = (  # цель кандидата
                        0.4 * mse_tr_f  # 0.4·MSE на train
                        + 0.6 * mse_loo  # + 0.6·MSE leave-one-out (оценка по одному)
                        + 0.002 * est_nb  # + 0.002·оценка байтов
                        + (1.0 - 0.5 * (acc_tr_f + acc_loo)) * 0.5  # + 0.5·(1 − 0.5·(точность train + точность leave-one-out))
                    )
                    float_cands.append((f_obj, lam, idx, wk, mse_loo, acc_loo))  # сохранить кандидата
            float_cands.sort(key=lambda c: c[0])  # отсортировать кандидатов по возрастанию цели
            for _, lam, idx, wk, mse_loo, acc_loo in float_cands[:3]:  # по трём лучшим кандидатам
                wq, packed = quantize_bitalloc_exact(Phi, idx, wk, y, sy, lam)  # квантование с распределением бит: веса wq и упакованные байты
                pred_q = (Phi @ wq) * sy  # прогноз квантованной модели: pred_q = (Phi·wq)·sy
                mse_tr = float(np.mean((pred_q - ty) ** 2))  # MSE на train: mean((pred_q − ty)^2)
                acc_tr = float(np.mean(np.rint(pred_q) == ty))  # точность на train: доля round(pred_q) = ty
                nb = 6 + len(packed)  # число байтов: nb = 6 + длина packed
                obj = (  # цель квантованного кандидата
                    0.4 * mse_tr  # 0.4·MSE на train
                    + 0.6 * mse_loo  # + 0.6·MSE leave-one-out (оценка по одному) (вещественных весов)
                    + 0.002 * nb  # + 0.002·число байтов
                    + (1.0 - 0.5 * (acc_tr + acc_loo)) * 0.5  # + 0.5·(1 − 0.5·(точность train + точность leave-one-out))
                )
                if obj < best[0]:  # цель меньше рекордной
                    best = (  # новый рекорд
                        obj,  # цель
                        bname,  # ветвь признаков
                        "reg",  # режим — регрессия
                        wq,  # квантованные веса
                        packed,  # упакованные байты
                        packed,  # упакованные байты (второй слот)
                        None,  # метаданных CE нет
                        pred_q / sy,  # прогноз train в нормированных единицах
                        None,  # метаданных MLP нет
                    )
            b_idx = self.BRANCHES.index(bname)  # номер ветви в списке BRANCHES
            branch_losses[b_idx] = 1.0 - b_best_acc  # потеря ветви = 1 − лучшая LOO-точность
            if best[0] < 0.025:  # цель уже меньше 0.025
                break  # достаточно — прекратить перебор

        classes = sorted(set(ty.astype(np.int64).tolist()))  # отсортированные классы — различные целые значения ty
        if 2 <= len(classes) <= 5 and N >= 12:  # от 2 до 5 классов и не меньше 12 примеров
            cls_map = {c: i for i, c in enumerate(classes)}  # соответствие «значение класса → индекс»
            y_idx = np.array([cls_map[int(c)] for c in ty], dtype=np.int64)  # индексы классов для каждого примера
            Phi_h = self.features("hinge", tx, sx)  # ломаные признаки
            scores = np.zeros(Phi_h.shape[1])  # оценки столбцов, изначально нули
            for c_i in range(len(classes)):  # по классам
                yc = (y_idx == c_i).astype(np.float64) - np.mean(y_idx == c_i)  # центрированный индикатор класса: 1[c = c_i] − доля класса c_i
                scores += np.abs(Phi_h.T @ yc)  # scores += |Phi_hᵀ·yc| — связь признака с индикатором
            top_cols = np.argsort(-scores)[: min(4, Phi_h.shape[1])]  # до 4 столбцов с наибольшей оценкой
            Phi_sub = Phi_h[:, top_cols]  # матрица отобранных признаков
            res = batch_logit_safe_step_ce(  # один безопасный шаг перекрёстной энтропии
                Phi_sub,  # по отобранным признакам
                y_idx,  # с индексами классов
                len(classes),  # число классов
                steps=300,  # 300 шагов
                l2_reg=np.full(top_cols.size, 1e-6),  # L2-регуляризация 1e-6 на каждый признак
            )
            theta = res["theta"]  # матрица весов классов (число классов, число признаков)
            raw_ce, th_dec = pack_vector_dyadic5(theta.ravel(), stages=2)  # квантование dyadic5: байты raw_ce и квантованный вектор th_dec
            th_q = th_dec.reshape(theta.shape)  # квантованные веса в форме theta
            nb_ce = 6 + top_cols.size + len(raw_ce)  # число байтов: 6 + число признаков + длина raw_ce
            pred_tr = np.array(  # прогноз на train
                [classes[i] for i in np.argmax(Phi_sub @ th_q.T, axis=1)],  # класс с максимальной логитой: classes[argmax(Phi_sub·th_qᵀ)]
                dtype=np.float64,  # в float64
            )
            acc_tr = float(np.mean(pred_tr == ty))  # точность на train: доля совпадений с ty
            mse_tr = float(np.mean((pred_tr - ty) ** 2))  # MSE на train: mean((pred_tr − ty)^2)
            obj_ce = mse_tr + 0.002 * nb_ce + (1.0 - acc_tr) * 0.5  # цель CE: mse_tr + 0.002·nb_ce + 0.5·(1 − acc_tr)
            if obj_ce < best[0]:  # цель меньше рекордной
                ce_meta = {  # метаданные классификатора
                    "classes": classes,  # список классов
                    "cols": top_cols.tolist(),  # номера столбцов признаков
                    "theta": th_q,  # квантованные веса
                    "packed": raw_ce,  # упакованные байты
                }
                best = (  # новый рекорд
                    obj_ce,  # цель
                    "hinge",  # ветвь признаков
                    "ce",  # режим — классификация
                    None,  # весов регрессии нет
                    raw_ce,  # упакованные байты
                    raw_ce,  # упакованные байты (второй слот)
                    ce_meta,  # метаданные CE
                    pred_tr / sy,  # прогноз train в нормированных единицах
                    None,  # метаданных MLP нет
                )

        if router is not None:  # маршрутизатор задан
            router.update(branch_losses)  # передать ему потери ветвей

        _, bname, mode, wq, packed, _, ce_meta, ptr_n, mlp_meta = best  # распаковать рекорд

        if ptr_n is None:  # ни один кандидат не принят — вырожденная подгонка
            # Все кандидаты OMP отклонены (|w| > 8), ветвь классификации неприменима.
            # Вместо сбоя сертифицируется консервативный нулевой прогноз: дальнейший
            # допуск коммита увидит истинные (плохие) метрики val и поставит HOLD на этом эксперте.
            wq = np.zeros(Phi.shape[1], dtype=np.float64)  # нулевые веса (прогноз 0)
            packed = pack_vector_dyadic5(wq, stages=1)[0]  # упакованные байты нулевых весов (1 стадия)
            mode, bname, ce_meta, mlp_meta = "reg", "poly", None, None  # режим reg, ветвь poly, без метаданных CE и MLP
            ptr_n = np.zeros(N, dtype=np.float64)  # нулевой прогноз train в нормированных единицах

        acc_tr_cur = float(np.mean(np.rint(ptr_n * sy) == ty))  # точность на train: доля round(ptr_n·sy) = ty
        mse_tr_cur = float(np.mean((ptr_n * sy - ty) ** 2))  # MSE на train: mean((ptr_n·sy − ty)^2)
        if acc_tr_cur < 0.99 and N >= 120 and mse_tr_cur > 0.08 and mode == "reg":  # точность < 0.99, N ≥ 120, MSE > 0.08, режим reg — пробовать корректор
            resid_tr = y - ptr_n  # остаток в нормированных единицах: resid_tr = y − ptr_n
            cut = int(0.8 * N)  # граница нарезки: cut = ⌊0.8·N⌋
            _, r_icv, _ = self._fit_stage2_mlp(  # внутренняя проверка: MLP на первых 80%, прогноз остатка на последних 20%
                tx[:cut], resid_tr[:cut], tx[cut:], sx, hidden=12, steps=260, seed=7  # train — срез 0..cut−1, проверка — срез cut..N−1
            )
            mse_icv_0 = float(np.mean((resid_tr[cut:] * sy) ** 2))  # MSE остатка без корректора на хвосте: mean((resid·sy)^2)
            mse_icv_1 = float(np.mean(((resid_tr[cut:] - r_icv) * sy) ** 2))  # MSE остатка после корректора: mean(((resid − r_icv)·sy)^2)
            if mse_icv_1 < 0.72 * mse_icv_0:  # корректор снижает MSE более чем на 28%
                _, _, mlp_meta = self._fit_stage2_mlp(  # обучить корректор на всей выборке
                    tx, resid_tr, vx, sx, hidden=12, steps=280, seed=7  # train — все данные, val — vx
                )
                mode = "rvq2"  # режим — регрессия плюс корректор

        Phi_va = self.features(bname, vx, sx)  # матрица признаков проверочной выборки

        def _align_cols(P: np.ndarray, n: int) -> np.ndarray:  # привести число столбцов P к n
            """Число столбцов признаков на train и val может различаться для ветвей, зависящих от данных;
            поэтому матрица приводится к длине вектора весов: лишние столбцы отбрасываются,
            недостающие дополняются нулями. Возвращает P из n столбцов."""
            if P.shape[1] == n:  # число столбцов совпадает
                return P  # вернуть без изменений
            if P.shape[1] > n:  # столбцов больше, чем нужно
                return P[:, :n]  # взять первые n столбцов
            return np.pad(P, ((0, 0), (0, n - P.shape[1])))  # дополнить нулями справа до n столбцов

        if mode == "ce":  # режим классификации
            assert ce_meta is not None  # метаданные CE обязательны
            Pv = Phi_va[:, ce_meta["cols"]]  # отобранные столбцы проверочной матрицы
            pred_q = np.array(  # прогноз на val
                [  # по каждому примеру
                    ce_meta["classes"][i]  # класс с индексом максимальной логиты
                    for i in np.argmax(Pv @ ce_meta["theta"].T, axis=1)  # выбранные классы: i = argmax(Pv·thetaᵀ)
                ],
                dtype=np.float64,  # в float64
            )
            nb_total = 6 + len(ce_meta["cols"]) + len(ce_meta["packed"])  # всего байтов: 6 + число признаков + длина packed
        elif mode == "rvq2":  # режим регрессии с корректором
            assert mlp_meta is not None and wq is not None  # метаданные MLP и веса обязательны
            base_va = _align_cols(Phi_va, len(wq)) @ wq  # базовый прогноз: Phi_va (выровненная)·wq
            Xv_mlp = self._mlp_encode(vx, sx)  # входы корректора на val
            r_va = (  # поправка корректора на val
                np.tanh(Xv_mlp @ mlp_meta["W1"] + mlp_meta["b1"]) @ mlp_meta["W2"]  # скрытый слой: tanh(X·W1 + b1)·W2
                + mlp_meta["b2"]  # смещение: + b2
            ).ravel()  # развёрнуто в вектор
            pred_q = (base_va + r_va) * sy  # прогноз на val: (base + поправка)·sy
            nb_total = 6 + len(packed) + len(mlp_meta["packed"])  # всего байтов: 6 + байты регрессии + байты корректора
        else:  # режим чистой регрессии
            assert wq is not None  # веса обязательны
            pred_q = (_align_cols(Phi_va, len(wq)) @ wq) * sy  # прогноз на val: (Phi_va·wq)·sy
            nb_total = 6 + len(packed)  # всего байтов: 6 + длина packed

        return FactorizedAdapter(  # собрать адаптер
            arity=d,  # арность
            branch=bname,  # ветвь признаков
            mode=mode,  # режим (reg, ce или rvq2)
            sx=sx,  # масштаб входа
            sy=sy,  # масштаб выхода
            w_quant=wq,  # квантованные веса регрессии
            packed_bytes=packed,  # упакованные байты
            ce_meta=ce_meta,  # метаданные классификатора
            mlp_meta=mlp_meta,  # метаданные корректора
            compressed_bytes=nb_total,  # оценка размера в байтах
            val_exact_acc_quant=float(np.mean(np.rint(pred_q) == vy)),  # точность на val: доля round(pred_q) = vy
            val_mse_quant=float(np.mean((pred_q - vy) ** 2)),  # MSE на val: mean((pred_q − vy)^2)
        )


@dataclass  # поля класса автоматически становятся аргументами конструктора
class FactorizedAdapter:  # обученный компактный адаптер
    """Обученный адаптер: компактная модель y = f(x) по целочисленным аргументам.
    
        Поля: arity — арность; branch — ветвь признаков (poly, hinge, bit, core); mode — режим
        (reg — линейная регрессия по признакам, ce — классификатор, rvq2 — регрессия плюс
        корректор-перцептрон); sx, sy — масштабы входа и выхода; w_quant — квантованные веса;
        packed_bytes — упакованные байты; ce_meta, mlp_meta — метаданные классификатора и
        корректора; compressed_bytes — оценка размера; val_exact_acc_quant, val_mse_quant —
        точность и MSE на val. Прогноз: reg → (Phi(x)·w)·sy; ce → класс с максимальной логитой;
        rvq2 → ((Phi(x)·w) + mlp(x))·sy."""
    arity: int  # арность — число аргументов
    branch: str  # ветвь признаков
    mode: str  # режим: reg, ce или rvq2
    sx: float  # масштаб входа
    sy: float  # масштаб выхода
    w_quant: Optional[np.ndarray]  # квантованные веса регрессии
    packed_bytes: bytes  # упакованные байты
    ce_meta: Optional[dict[str, Any]]  # метаданные классификатора
    mlp_meta: Optional[dict[str, Any]]  # метаданные корректора
    compressed_bytes: int  # оценка размера в байтах
    val_exact_acc_quant: float  # точность на val
    val_mse_quant: float  # MSE на val

    def predict(self, core: FactorizedSharedCore, xs: Sequence[int]) -> float:  # предсказать значение по одному набору аргументов
        """Предсказать значение для одного набора аргументов xs.
        
                core: ядро с построителем признаков. Возвращает число по формулам, приведённым в описании класса."""
        x = np.asarray(xs, dtype=np.float64).reshape(1, -1)  # аргументы в строку-матрицу (1, d)
        Phi = core.features(self.branch, x, self.sx)  # признаки Phi (1, число признаков)
        if self.mode == "ce" and self.ce_meta is not None:  # режим классификации
            P = Phi[:, self.ce_meta["cols"]]  # отобранные столбцы P
            idx = int(np.argmax((P @ self.ce_meta["theta"].T)[0]))  # индекс класса с максимальной логитой: argmax(P·thetaᵀ)
            return float(self.ce_meta["classes"][idx])  # вернуть значение класса
        assert self.w_quant is not None  # веса обязательны
        val = float((Phi @ self.w_quant)[0])  # базовый прогноз в нормированных единицах: val = Phi·w
        if self.mode == "rvq2" and self.mlp_meta is not None:  # режим регрессии с корректором
            Xm = core._mlp_encode(x, self.sx)  # входы корректора
            r = float(  # поправка корректора
                (  # открытие выражения
                    np.tanh(Xm @ self.mlp_meta["W1"] + self.mlp_meta["b1"])  # скрытый слой: tanh(X·W1 + b1)
                    @ self.mlp_meta["W2"]  # умножение: · W2
                    + self.mlp_meta["b2"]  # смещение: + b2
                )[0, 0]  # взять единственный элемент
            )
            val += r  # val = val + поправка
        return val * self.sy  # вернуть в исходном масштабе: val·sy

    def to_dict(self) -> dict[str, Any]:  # сериализация в словарь
        """Словарь для сериализации: все поля, packed — в hex; для ce — классы и столбцы признаков,
                для rvq2 — упакованные байты и формы весов корректора."""
        out: dict[str, Any] = {  # словарь полей
            "arity": self.arity,  # арность
            "branch": self.branch,  # ветвь признаков
            "mode": self.mode,  # режим
            "sx": self.sx,  # масштаб входа
            "sy": self.sy,  # масштаб выхода
            "compressed_bytes": self.compressed_bytes,  # оценка размера
            "packed_hex": self.packed_bytes.hex(),  # упакованные байты в hex
            "val_exact_acc_quant": self.val_exact_acc_quant,  # точность на val
            "val_mse_quant": self.val_mse_quant,  # MSE на val
        }
        if self.mode == "ce" and self.ce_meta is not None:  # режим классификации — добавить классы и столбцы
            out["ce_classes"] = self.ce_meta["classes"]  # список классов
            out["ce_cols"] = self.ce_meta["cols"]  # номера столбцов
        elif self.mode == "rvq2" and self.mlp_meta is not None:  # режим с корректором — добавить его байты и формы
            out["mlp_packed_hex"] = self.mlp_meta["packed"].hex()  # упакованные байты корректора в hex
            out["mlp_shapes"] = [  # формы весов корректора
                list(self.mlp_meta["W1"].shape),  # форма W1
                list(self.mlp_meta["b1"].shape),  # форма b1
                list(self.mlp_meta["W2"].shape),  # форма W2
                list(self.mlp_meta["b2"].shape),  # форма b2
            ]
        return out  # вернуть словарь

    @staticmethod  # статический метод — не получает self
    def from_dict(  # восстановить адаптер из словаря
        d: dict[str, Any], core: "FactorizedSharedCore"  # d — словарь, core — ядро с построителем признаков
    ) -> "FactorizedAdapter":  # возвращает адаптер
        """Восстановить адаптер из словаря d (обратная операция к to_dict).
        
                d: словарь из to_dict, core: ядро (нужен для ширины матрицы признаков D).
                Веса регрессии распаковываются из packed записей распределения бит по D признаков;
                веса CE — из dyadic5-упаковки в форму (число классов, число столбцов);
                веса корректора — dyadic5-упаковка, обратный поворот Адамара и разрезка по формам."""
        arity = int(d["arity"])  # арность
        branch = str(d["branch"])  # ветвь признаков
        mode = str(d["mode"])  # режим
        sx = float(d["sx"])  # масштаб входа
        sy = float(d["sy"])  # масштаб выхода
        packed = bytes.fromhex(d["packed_hex"])  # упакованные байты из hex
        D = core.features(branch, np.zeros((1, arity), dtype=np.float64), sx).shape[1]  # ширина матрицы признаков D — по нулевому входу
        w_quant = None  # веса регрессии — пока нет
        ce_meta = None  # метаданные CE — пока нет
        mlp_meta = None  # метаданные корректора — пока нет
        if mode == "ce":  # режим классификации
            classes = [int(c) for c in d["ce_classes"]]  # классы
            cols = [int(c) for c in d["ce_cols"]]  # номера столбцов
            th = unpack_vector_dyadic5(packed).reshape(len(classes), len(cols))  # распаковать веса theta в форму (число классов, число столбцов)
            ce_meta = {"classes": classes, "cols": cols, "theta": th, "packed": packed}  # метаданные классификатора
        else:  # режим регрессии
            w_quant = unpack_bitalloc_records(packed, D)  # распаковать D весов из записей распределения бит
            if mode == "rvq2" and "mlp_packed_hex" in d:  # режим с корректором и его байты есть
                mp = bytes.fromhex(d["mlp_packed_hex"])  # упакованные байты корректора
                s1, sb1, s2, sb2 = [tuple(int(x) for x in sh) for sh in d["mlp_shapes"]]  # формы W1, b1, W2, b2
                n_all = (  # общее число весов
                    s1[0] * s1[1] + sb1[0] * sb1[1] + s2[0] * s2[1] + sb2[0] * sb2[1]  # n_all = |W1| + |b1| + |W2| + |b2| — число весов
                )
                wr_q = unpack_vector_dyadic5(mp)  # распакованный повёрнутый вектор весов
                all_wq = block_hadamard_unrotate(wr_q, n_all, 8)  # обратный поворот Адамара (блоки 8) и обрезка до n_all
                p1 = s1[0] * s1[1]  # граница W1: p1 = |W1|
                p2 = p1 + sb1[0] * sb1[1]  # граница b1: p2 = p1 + |b1|
                p3 = p2 + s2[0] * s2[1]  # граница W2: p3 = p2 + |W2|
                mlp_meta = {  # метаданные корректора
                    "W1": all_wq[:p1].reshape(s1),  # W1 — первые p1 элементов
                    "b1": all_wq[p1:p2].reshape(sb1),  # b1 — срез p1..p2−1
                    "W2": all_wq[p2:p3].reshape(s2),  # W2 — срез p2..p3−1
                    "b2": all_wq[p3:].reshape(sb2),  # b2 — остаток
                    "packed": mp,  # упакованные байты
                }
        return FactorizedAdapter(  # собрать адаптер
            arity=arity,  # арность
            branch=branch,  # ветвь
            mode=mode,  # режим
            sx=sx,  # масштаб входа
            sy=sy,  # масштаб выхода
            w_quant=w_quant,  # веса
            packed_bytes=packed,  # байты
            ce_meta=ce_meta,  # метаданные CE
            mlp_meta=mlp_meta,  # метаданные корректора
            compressed_bytes=int(d["compressed_bytes"]),  # размер
            val_exact_acc_quant=float(d["val_exact_acc_quant"]),  # точность на val
            val_mse_quant=float(d["val_mse_quant"]),  # MSE на val
        )


class CRAMEngine:  # движок CRAM: энергия, среднее поле, скрытые состояния, псевдо-правдоподобие

    @staticmethod  # статический метод
    def solve_coloring(  # раскраска графа локальным спуском
        edges: Sequence[tuple[int, int]],  # рёбра
        n_nodes: int,  # число вершин
        num_colors: int,  # число цветов
        givens: Optional[dict[int, int]] = None,  # фиксированные цвета вершин
        eps_noise: float = 0.0,  # амплитуда шума поля
        max_sweeps: int = 800,  # предел проходов
        seed: int = 42,  # зерно генератора
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Решить задачу раскраски графа локальным спуском с шумом и теплованной баней.
        
                edges: рёбра (a, b); n_nodes: число вершин; num_colors: число цветов; givens:
                фиксированные цвета вершин; eps_noise: амплитуда шума поля; max_sweeps: предел
                проходов; seed: зерно. Энергия E = число рёбер с одинаковыми цветами концов.
                Поле вершины i: f[c] = число соседей цвета c. Вершина перекрашивается, если
                f[y_i] − min_c f[c] > 2·eps_noise (шум не мог создать улучшение), цвет выбирается
                случайно среди лучших. Без улучшений 8 проходов подряд — случайный перезапуск;
                иначе — heat-bath шаги: P(c) ∝ exp(−f[c]/max(0.15, T)), затем T *= 0.96.
                Возвращает словарь с раскраской, энергией, счётчиками работы и сертификатом
                корректности: energy < 1, все рёбра правильные и условия givens выполнены."""
        adj = [set() for _ in range(n_nodes)]  # список множеств соседей
        for a, b in edges:  # по рёбрам
            if a != b:  # без петель
                adj[a].add(b)  # b — сосед a
                adj[b].add(a)  # a — сосед b
        adj_list = [sorted(s) for s in adj]  # отсортированные списки соседей
        giv = dict(givens or {})  # копия фиксированных цветов (пусто, если None)
        rng = np.random.default_rng(seed)  # генератор с зерном
        y = [giv.get(i, int(rng.integers(0, num_colors))) for i in range(n_nodes)]  # начальная раскраска: фиксированные или случайный цвет

        def energy(st: Sequence[int]) -> int:  # энергия раскраски st
            """Энергия E = число рёбер, у которых концы одного цвета."""
            return sum(1 for a, b in edges if st[a] == st[b])  # энергия E = Σ_{(a,b)} 1[st[a] = st[b]]

        def sparse_fields(i: int, st: Sequence[int]) -> list[int]:  # локальное поле вершины i
            """Поле f[c] = число соседей i цвета c в раскраске st."""
            f = [0] * num_colors  # нули по цветам
            for j in adj_list[i]:  # по соседям
                f[st[j]] += 1  # f[цвет соседа] + 1
            return f  # вернуть поле

        cur_e = energy(y)  # текущая энергия
        sweeps = hb_steps = stale = 0  # счётчики: проходы, шаги тепловой бани, застоявшиеся проходы
        T = 0.85  # начальная температура
        alpha_cool = 0.96  # коэффициент охлаждения
        sparse_work = sum(len(adj_list[i]) * (num_colors**2) for i in range(n_nodes))  # работа разреженного поля за проход: Σ_i deg(i)·C^2
        dense_work = n_nodes * max(1, n_nodes - 1) * (num_colors**2)  # работа плотного поля: N·max(1, N−1)·C^2

        while sweeps < max_sweeps and cur_e > 0:  # пока есть проходы и энергия > 0
            improved = False  # улучшений в проходе пока нет
            for i in range(n_nodes):  # по вершинам
                if i in giv:  # фиксированная вершина
                    continue  # пропустить
                fields = sparse_fields(i, y)  # поле вершины
                noisy = [  # зашумлённое поле
                    f  # значение поля
                    + (  # плюс
                        float(rng.uniform(-eps_noise, eps_noise))  # равномерный шум в [−eps, eps]
                        if eps_noise > 0  # если шум включён
                        else 0.0  # иначе 0
                    )
                    for f in fields  # по цветам
                ]
                cur_f = noisy[y[i]]  # поле текущего цвета
                best_val = min(noisy)  # минимальное поле
                if cur_f - best_val > 2.0 * eps_noise:  # улучшение больше удвоенного шума — оно настоящее
                    ties = [c for c, v in enumerate(noisy) if v == best_val]  # цвета с минимальным полем
                    y[i] = ties[int(rng.integers(0, len(ties)))]  # случайный из лучших
                    improved = True  # было улучшение
            cur_e = energy(y)  # пересчитать энергию
            sweeps += 1  # проходов + 1
            if cur_e == 0:  # энергия 0 — раскраска правильная
                break  # выход
            if not improved:  # улучшений не было
                stale += 1  # застой + 1
                if stale >= 8:  # застой ≥ 8 — перезапуск
                    y = [  # новая случайная раскраска
                        giv.get(i, int(rng.integers(0, num_colors)))  # фиксированный или случайный цвет
                        for i in range(n_nodes)  # по вершинам
                    ]
                    stale = 0  # сброс застоя
                    T = 0.85  # сброс температуры
                    cur_e = energy(y)  # пересчитать энергию
                    continue  # к следующему проходу
                for _ in range(min(12, max(3, cur_e * 2))):  # тепловая баня: min(12, max(3, 2E)) шагов
                    conf = [  # конфликтные вершины
                        i  # вершина
                        for i in range(n_nodes)  # по вершинам
                        if i not in giv and any(y[i] == y[j] for j in adj_list[i])  # не фиксирована и есть сосед того же цвета
                    ]
                    if not conf:  # конфликтов нет
                        break  # выход
                    pick = conf[int(rng.integers(0, len(conf)))]  # случайная конфликтная вершина
                    if rng.random() < 0.4 and adj_list[pick]:  # с вероятностью 0,4 перейти к соседу
                        free_nb = [j for j in adj_list[pick] if j not in giv]  # соседи без фиксированного цвета
                        if free_nb:  # такие есть
                            pick = free_nb[int(rng.integers(0, len(free_nb)))]  # случайный сосед
                    fields = sparse_fields(pick, y)  # поле выбранной вершины
                    logits = np.array(  # логиты
                        [-f / max(0.15, T) for f in fields], dtype=np.float64  # логит цвета: logit_c = −f[c] / max(0.15, T)
                    )
                    logits -= np.max(logits)  # вычесть максимум для устойчивости
                    probs = np.exp(logits)  # экспоненты
                    probs /= np.sum(probs)  # нормировка: P(c) = exp(l_c) / Σ exp(l)
                    y[pick] = int(rng.choice(num_colors, p=probs))  # выбор цвета по распределению
                    hb_steps += 1  # шагов бани + 1
                T *= alpha_cool  # охлаждение: T = T·0.96
                cur_e = energy(y)  # пересчитать энергию

        valid = all(y[a] != y[b] for a, b in edges) and all(  # правильность: все рёбра разноцветные и
            y[i] == g for i, g in giv.items()  # все фиксированные цвета соблюдены
        )
        return {  # вернуть результат
            "coloring": y,  # раскраска
            "valid": valid,  # признак правильности
            "energy": cur_e,  # энергия
            "sweeps": sweeps,  # число проходов
            "hb_steps": hb_steps,  # число шагов бани
            "validity_certificate": bool(cur_e < 1 and valid),  # сертификат: энергия < 1 и раскраска правильная
            "givens_respected": all(y[i] == g for i, g in giv.items()),  # фиксированные цвета соблюдены
            "sparse_work_per_sweep": sparse_work,  # работа разреженного поля
            "dense_work_per_sweep": dense_work,  # работа плотного поля
        }

    @staticmethod  # статический метод
    def mean_field_kl_step(  # шаг координатного среднего поля
        B: np.ndarray, K: np.ndarray, p: np.ndarray, beta: float = 2.0, iters: int = 12  # B, K — поля; p — распределение; beta — обратная температура; iters — проходы
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Один шаг координатного среднего поля (mean-field) для KL-вариационной задачи.
        
                B: (N, A) унарные поля; K: (N, N, A, A) парные; p: (N, A) текущее распределение;
                beta: обратная температура; iters: число проходов. Свободная энергия
                F(p) = Σ B·p + 0.5·Σ_{i<j} p_i K_sym p_j + (1/beta)·Σ p·log p. Для каждой вершины
                новое q_i ∝ exp(−beta·h_i), h_i = B_i + Σ_j K_sym[i] p_j − K_sym[i,i] p_i.
                Гарантия спуска: F(p) − F(q) = (1/beta)·KL(p_i‖q_i) ≥ 0 (проверяется утверждением).
                Возвращает p, свободные энергии и признак монотонности."""
        N, A = p.shape  # N — вершины, A — состояния
        p_cur = p.copy()  # текущее распределение — копия
        K_sym = 0.5 * (K + K.transpose(1, 0, 3, 2))  # симметризация: K_sym[i,j,a,b] = (K[i,j,a,b] + K[j,i,b,a]) / 2

        def free_energy(pm: np.ndarray) -> float:  # свободная энергия распределения pm
            """F(pm) = Σ B·pm + 0.5·Σ pm_i K_sym pm_j + (1/beta)·Σ pm·log pm."""
            unary = float(np.sum(B * pm))  # унарный член: Σ_{i,a} B[i,a]·pm[i,a]
            pair = 0.5 * float(np.einsum("ia,ijab,jb->", pm, K_sym, pm))  # парный член: 0.5·Σ pm_ia·K_sym[i,j,a,b]·pm_jb
            ent = (  # энтропийный член
                float(  # сумма
                    np.sum(  # сумма
                        np.where(pm > 1e-18, pm * np.log(np.maximum(pm, 1e-18)), 0.0)  # pm·log(pm) там, где pm > 1e-18, иначе 0
                    )
                )
                / beta  # делённая на beta
            )
            return unary + pair + ent  # F = унарный + парный + энтропийный

        f0 = free_energy(p_cur)  # начальная свободная энергия
        f_hist = [f0]  # история значений
        for _ in range(iters):  # по проходам
            for i in range(N):  # по вершинам
                hz = (  # локальное поле вершины
                    B[i]  # унарное поле B_i
                    + np.einsum("jab,jb->a", K_sym[i], p_cur)  # симметричный вклад: + Σ_{j,b} K_sym[i,j,a,b]·p_jb
                    - np.einsum("ab,b->a", K_sym[i, i], p_cur[i])  # − Σ_b K_sym[i,i,a,b]·p_ib (без самосвязи)
                )
                logits = -beta * hz  # логиты: −beta·h
                logits -= np.max(logits)  # вычесть максимум
                q_i = np.exp(logits)  # экспоненты
                q_i /= np.sum(q_i)  # нормировка: q_i = exp(−beta·h) / Σ exp(−beta·h)
                p_old_i = p_cur[i].copy()  # старое распределение вершины
                kl = float(  # KL-расхождение старого и нового
                    np.sum(  # сумма
                        np.where(  # где
                            p_old_i > 1e-18,  # где старое значение p_old > 1e-18
                            p_old_i  # старое значение p_old
                            * np.log(  # умножить на логарифм
                                np.maximum(p_old_i, 1e-18) / np.maximum(q_i, 1e-18)  # отношение p_old / q
                            ),
                            0.0,  # иначе 0
                        )
                    )
                )
                f_before = free_energy(p_cur)  # F до замены
                p_cur[i] = q_i  # заменить распределение вершины
                f_after = free_energy(p_cur)  # F после
                assert abs((f_before - f_after) - kl / beta) <= 1e-7  # проверка: F(p) − F(q) = KL/beta с точностью 1e-7
                f_hist.append(f_after)  # добавить в историю
        return {  # вернуть
            "p": p_cur,  # распределение
            "free_energy_0": f0,  # начальная F
            "free_energy_final": f_hist[-1],  # конечная F
            "monotonic": all(  # монотонность
                f_hist[k + 1] <= f_hist[k] + 1e-9 for k in range(len(f_hist) - 1)  # F_{k+1} ≤ F_k + 1e-9 для всех k
            ),
        }

    @staticmethod  # статический метод
    def solve_latent_maze(  # решение скрытого лабиринта итерацией сжатия
        n_nodes: int, adj: Sequence[Sequence[int]], goal: int, eps_z: float = 1e-6  # вершины, соседи, цель, точность
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Решить «скрытый лабиринт»: стационарную точку m·z = A·z + 1_G, где m = Δ + 1.
        
                adj: списки соседей; goal: вершина-цель; eps_z: точность. A — матрица смежности,
                Δ — максимальная степень. Итерации z_new[i] = (Σ_{j∈adj[i]} z[j] + 1[i = goal]) / m;
                это сжатие с коэффициентом q = Δ/m, поэтому остаток ≤ eps_z за
                ⌈log(eps_z·(1 − q)) / log(q)⌉ шагов (оценка: diff·q/(1 − q) ≤ eps_z).
                Границы: 0 ≤ z ≤ 1/(m − Δ). Возвращает z, число итераций, q и верхнюю границу."""
        delta = max((len(nb) for nb in adj), default=0)  # максимальная степень Δ
        m = float(delta + 1)  # множитель m = Δ + 1
        q = delta / m  # коэффициент сжатия q = Δ / m
        z = np.zeros(n_nodes, dtype=np.float64)  # z — нулевой вектор
        iters = 0  # число итераций
        if q > 0:  # есть рёбра
            max_it = max(1, math.ceil(math.log(eps_z * (1.0 - q)) / math.log(q)))  # предел итераций: max(1, ⌈log(eps·(1−q)) / log(q)⌉)
            for _ in range(max_it):  # по итерациям
                z_new = np.array(  # новое z
                    [  # начало списка значений
                        (sum(z[j] for j in adj[i]) + (1.0 if i == goal else 0.0)) / m  # новое значение z_i = (Σ_{j∈adj(i)} z_j + 1[i = goal]) / m
                        for i in range(n_nodes)  # по вершинам
                    ]
                )
                diff = float(np.max(np.abs(z_new - z)))  # изменение: max|z_new − z|
                z = z_new  # заменить z
                iters += 1  # итераций + 1
                if diff * q / (1.0 - q) <= eps_z:  # оценка остатка diff·q/(1−q) ≤ eps_z
                    break  # достаточно
        else:  # рёбер нет
            z[goal] = 1.0 / m  # только цель: z_goal = 1/m
            iters = 1  # одна итерация
        assert np.all(z >= -1e-12) and np.all(z <= 1.0 / (m - delta) + 1e-12)  # проверка границ: 0 ≤ z ≤ 1/(m − Δ)
        return {  # вернуть
            "z": z,  # вектор z
            "iters": iters,  # число итераций
            "contraction_q": q,  # коэффициент сжатия
            "upper_bound": 1.0 / (m - delta),  # верхняя граница 1/(m − Δ)
        }

    @staticmethod  # статический метод
    def attention_coupling_matrix(  # симметричная матрица парной связи
        kappa: np.ndarray, scores_z: np.ndarray, f_compat: np.ndarray  # смежность, оценки скрытых состояний, совместимость
    ) -> np.ndarray:  # возвращает K (N, N, A, A)
        """Построить симметричную матрицу парной связи из внимания и совместимости цветов.
        
                kappa: (N, N) смежность; scores_z: (N, N) оценки скрытых состояний;
                f_compat: (A, A) совместимость цветов. Внимание w = softmax по строкам scores_z;
                kw = kappa·w. K[i,j,a,b] = kw[i,j]·f[a,b] + kw[j,i]·f[b,a] — симметрично по построению;
                диагональ K[i,i] = 0."""
        s = scores_z - np.max(scores_z, axis=1, keepdims=True)  # вычесть максимум по строке
        w = np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True)  # внимание: softmax по строке
        kw = kappa * w  # произведение kw = kappa·w
        K = (  # матрица связи
            kw[:, :, None, None] * f_compat[None, None, :, :]  # вклад ребра: kw[i,j]·f[a,b]
            + kw.T[:, :, None, None] * f_compat.T[None, None, :, :]  # + kw[j,i]·f[b,a] — симметризация
        )
        for i in range(K.shape[0]):  # по вершинам
            K[i, i] = 0.0  # диагональ в 0
        return K  # вернуть K

    @staticmethod  # статический метод
    def alternating_latent_visible_solve(  # чередующееся решение (z, y)
        n_nodes: int,  # вершины
        adj: Sequence[Sequence[int]],  # соседи
        goal: int,  # цель
        num_colors: int = 2,  # число цветов
        rounds: int = 3,  # раунды
        seed: int = 42,  # зерно
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Чередующееся решение (z, y): скрытые z → связь внимания → спуск по цветам.
        
                Сначала z из solve_latent_maze; затем K из attention_coupling_matrix с
                scores_z = 4·z⊗z и единичной совместимостью. Энергия E = 0.5·Σ_{i≠j} K[i,j,y_i,y_j].
                В каждом раунде вершина берёт цвет c с минимальным полем Σ_j K[i,j,c,y_j],
                если он меньше текущего. Возвращает z, раскраску, энергии до/после и признак
                неубывания энергии."""
        res_z = CRAMEngine.solve_latent_maze(n_nodes, adj, goal, eps_z=1e-6)  # скрытые состояния z
        z = res_z["z"]  # вектор z
        rng = np.random.default_rng(seed)  # генератор
        kappa = np.zeros((n_nodes, n_nodes), dtype=np.float64)  # нулевая смежность
        for i, nbs in enumerate(adj):  # по вершинам и соседям
            for j in nbs:  # по соседям
                kappa[i, j] = 1.0  # отметить ребро
        f_compat = np.eye(num_colors, dtype=np.float64)  # единичная совместимость: штраф только за одинаковые цвета
        scores_z = np.outer(z, z) * 4.0  # оценки: 4·z_i·z_j
        K = CRAMEngine.attention_coupling_matrix(kappa, scores_z, f_compat)  # матрица связи
        assert np.allclose(K, K.transpose(1, 0, 3, 2))  # проверка симметрии
        y = [int(rng.integers(0, num_colors)) for _ in range(n_nodes)]  # случайная начальная раскраска

        def pair_e(st):  # энергия раскраски st
            """E(st) = 0.5·Σ_{i≠j} K[i,j,st_i,st_j]."""
            return 0.5 * sum(  # половина суммы
                K[i, j, st[i], st[j]]  # элемент связи K[i,j,st_i,st_j]
                for i in range(n_nodes)  # по i
                for j in range(n_nodes)  # по j
                if i != j  # без диагонали
            )

        e0 = pair_e(y)  # начальная энергия
        for _ in range(rounds):  # по раундам
            for i in range(n_nodes):  # по вершинам
                fields = [  # поля по цветам
                    sum(K[i, j, c, y[j]] for j in range(n_nodes) if j != i)  # поле цвета: f_c = Σ_{j≠i} K[i,j,c,y_j]
                    for c in range(num_colors)  # по цветам
                ]
                best_c = int(np.argmin(fields))  # цвет с минимальным полем
                if fields[best_c] < fields[y[i]] - 1e-12:  # строго лучше текущего
                    y[i] = best_c  # перекрасить
        ef = pair_e(y)  # конечная энергия
        return {  # вернуть
            "z": z,  # запомнить: z
            "y": y,  # раскраска
            "energy_0": e0,  # энергия до
            "energy_final": ef,  # энергия после
            "monotone_drop": ef <= e0 + 1e-12,  # энергия не возросла
        }

    @staticmethod  # статический метод
    def train_cram_pseudolikelihood(  # обучение унарных полей по псевдо-правдоподобию
        samples: np.ndarray, num_colors: int, beta: float = 1.0, steps: int = 80  # выборки, цвета, обратная температура, шаги
    ) -> dict[str, Any]:  # возвращает словарь результата
        """Обучить унарные поля B по псевдо-правдоподобию: перекрёстная энтропия локального поля.
        
                samples: (M, N) выборки цветов; num_colors: число цветов; beta: обратная
                температура; steps: число шагов. Для каждой вершины i логиты единичного
                признака обучаются предсказывать samples[:, i]; B[i] = −theta[:, 0] / beta.
                Возвращает B и среднюю потерю."""
        M, N = samples.shape  # M — выборки, N — вершины
        B_learned = np.zeros((N, num_colors), dtype=np.float64)  # обучаемые поля B — нули
        losses = []  # потери по вершинам
        for i in range(N):  # по вершинам
            y_i = samples[:, i]  # цвета вершины i по выборкам
            X_i = np.ones((M, 1), dtype=np.float64)  # единственный признак — столбец единиц
            res = batch_logit_safe_step_ce(  # обучить логиты
                X_i, y_i, num_colors, steps=steps, l2_reg=np.array([1e-4])  # по одному признаку, L2 1e-4
            )
            B_learned[i] = -res["theta"][:, 0] / beta  # сдвиг: B_i = −theta_i / beta
            losses.append(res["loss_final"])  # запомнить потерю
        return {"B": B_learned, "mean_pl_loss": float(np.mean(losses))}  # вернуть B и среднюю потерю

__all__ = [  # публичные имена модуля: только определённые здесь (импортированные не выносит `from <модуль> import *`)
    'bf16_nearest_float', 'batch_logit_safe_step_ce', 'newton_schulz_orthogonalize', 'ContinualGPMMemory', 'quantize_bitalloc_exact', 'FactorizedSharedCore',  # имя интерфейса
    'FactorizedAdapter', 'CRAMEngine',  # имя интерфейса
]  # конец перечня
