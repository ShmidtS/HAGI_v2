"""GPU-реализация (torch, CUDA/ROCm) тяжёлых частей нейронного ядра HAGI.

Переносит на тензор torch вычислительно дорогие части ContinualGPMMemory —
цикл обучения EF21-Muon (ортогонализация Ньютона — Шульца + обратный поиск шага)
и укрепление подпространства через SVD — на первое доступное устройство CUDA.

Политика устройства (по измерениям, честно):
  HAGI_NEURAL_DEVICE = auto | cpu | cuda   (по умолчанию auto)
  - `auto` при первом обращении сравнивает время одной серии шагов EF21-Muon на GPU
    и на CPU и оставляет более быстрый вариант (для небольших матриц ядра накладные
    расходы на запуск ядер GPU могут преобладать; GPU выигрывает, когда задачи
    объединяются в пакеты или ядро расширяется).
  - порождаемым параллельным исполнителям следует задавать `cpu` (свой контекст GPU
    у каждого умножил бы память GPU); основной процесс использует GPU.

GPU-путь совпадает с numpy по структуре, но не побитово (умножения float64 на GPU
могут отличаться в последних разрядах); гейты приёма никогда не зависят от
нейронного ядра, поэтому это влияет только на метрики качества адаптеров, но не
на принятые программы.

Экспортирует resolve_device (выбранное устройство с запоминанием), _newton_schulz_torch
(ортогонализация), train_task_ef21_muon_gpu (обучение на GPU), consolidate_subspace_gpu
(SVD-укрепление базиса) и GpuGpmMixin (примесь-ускоритель для ContinualGPMMemory)."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import os  # чтение переменной окружения HAGI_NEURAL_DEVICE
from typing import Any  # тип для аннотаций

import numpy as np  # генератор случайных массивов для сравнения устройств

_DEVICE: str | None = None          # выбранное устройство: "cpu" | "cuda"; None — ещё не выбрано
_BENCH: dict[str, float] | None = None  # зарезервированное место для измерений времени (не используется)


def _torch():
    """Вернуть модуль torch (ленивый импорт: без него файл импортируется и без torch)."""
    import torch  # нейросетевая библиотека (локальный импорт)
    return torch  # модуль torch


def resolve_device() -> str:
    """Выбрать устройство нейронного ядра и запомнить его: "cpu" или "cuda".

    mode = HAGI_NEURAL_DEVICE в нижнем регистре (по умолчанию "auto"). mode = "cpu" —
    сразу cpu. Иначе при отсутствии CUDA — cpu; при mode = "cuda" — cuda. При "auto"
    генерируются X (256, 48), W (48, 16)·0.1, Y (256, 16) из нормального распределения
    с зерном 0, замеряется 30 шагов градиентного спуска на numpy и на GPU; выбирается
    cuda, если время GPU меньше. Любая ошибка даёт cpu."""
    global _DEVICE  # изменяется общая переменная модуля
    if _DEVICE is not None:  # устройство уже выбрано
        return _DEVICE  # вернуть запомненное значение
    mode = os.environ.get("HAGI_NEURAL_DEVICE", "auto").lower()  # режим: значение переменной в нижнем регистре (по умолчанию "auto")
    if mode == "cpu":  # явно задан процессор
        _DEVICE = "cpu"  # выбрать cpu
        return _DEVICE  # вернуть "cpu"
    try:  # torch или CUDA могут быть недоступны — тогда cpu
        t = _torch()  # модуль torch
        if not t.cuda.is_available():  # CUDA недоступна
            _DEVICE = "cpu"  # выбрать cpu
            return _DEVICE  # вернуть "cpu"
        if mode == "cuda":  # явно задан GPU
            _DEVICE = "cuda"  # выбрать cuda
            return _DEVICE  # вернуть "cuda"
        # auto: короткое сравнение на тех же формах, что у реального шага обучения
        rng = np.random.default_rng(0)  # генератор с зерном 0
        X = rng.normal(size=(256, 48))  # X — нормальный шум формы (256, 48)
        W = rng.normal(size=(48, 16)) * 0.1  # W — нормальный шум формы (48, 16), умноженный на 0.1
        Y = rng.normal(size=(256, 16))  # Y — нормальный шум формы (256, 16)
        cpu_t = _bench_step(np, X, W, Y)  # время 30 шагов на numpy
        cuda_t = _bench_step_cuda(X, W, Y)  # время 30 шагов на GPU
        _DEVICE = "cuda" if cuda_t < cpu_t else "cpu"  # выбрать более быстрое устройство
        return _DEVICE  # вернуть выбор
    except Exception:  # torch недоступен или измерение не удалось
        _DEVICE = "cpu"  # выбрать cpu
        return _DEVICE  # вернуть "cpu"


def _bench_step(np_mod, X, W, Y) -> float:
    """Время 30 шагов градиентного спуска на numpy, секунды.

    На каждом шаге: err = X·W − Y; grad = (2/n)·Xᵀ·err, n = len(X); W = W − 0.05·grad.
    W копируется, аргумент не меняется."""
    import time  # измеритель времени (локальный импорт)
    W = W.copy()  # рабочая копия весов
    t0 = time.perf_counter()  # начало отсчёта
    for _ in range(30):  # 30 шагов
        err = X @ W - Y  # ошибка err = X·W − Y
        grad = (2.0 / len(X)) * (X.T @ err)  # градиент grad = (2/len(X))·Xᵀ·err
        W = W - 0.05 * grad  # шаг W = W − 0.05·grad
    return time.perf_counter() - t0  # прошедшее время в секундах


def _bench_step_cuda(X, W, Y) -> float:
    """Время 30 шагов градиентного спуска на GPU (float64), секунды; формулы те же, что в _bench_step."""
    import time  # измеритель времени (локальный импорт)
    t = _torch()  # модуль torch
    Xt = t.as_tensor(X, dtype=t.float64, device="cuda")  # X на GPU в float64
    Wt = t.as_tensor(W, dtype=t.float64, device="cuda").clone()  # рабочая копия W на GPU в float64
    Yt = t.as_tensor(Y, dtype=t.float64, device="cuda")  # Y на GPU в float64
    t.cuda.synchronize()  # дождаться окончания отложенных операций до старта таймера
    t0 = time.perf_counter()  # начало отсчёта
    for _ in range(30):  # 30 шагов
        err = Xt @ Wt - Yt  # ошибка err = X·W − Y
        grad = (2.0 / Xt.shape[0]) * (Xt.T @ err)  # градиент grad = (2/n)·Xᵀ·err, n = Xt.shape[0]
        Wt = Wt - 0.05 * grad  # шаг W = W − 0.05·grad
    t.cuda.synchronize()  # дождаться окончания всех шагов
    return time.perf_counter() - t0  # прошедшее время в секундах


def _newton_schulz_torch(G: Any, iters: int = 4) -> Any:
    """Ортогонализация Ньютона — Шульца на устройстве: за iters итераций X <- X·(3I − Xᵀ·X)/2.

    Точный перенос newton_schulz_orthogonalize на torch. G приводится к float64 и
    копируется; X = G/‖G‖ (при ‖G‖ < 1e-12 возвращается исходная копия); «высокая»
    матрица транспонируется, чтобы итерировать по строкам; X <- 0.5·(3I − X·Xᵀ)·X;
    в конце форма возвращается обратным транспонированием."""
    t = _torch()  # модуль torch
    X = G.to(t.float64).clone()  # рабочая копия в float64
    norm = float(X.norm())  # норма Фробениуса norm = ‖G‖
    if norm < 1e-12:  # матрица почти нулевая — делить нельзя
        return X  # вернуть копию как есть
    X = X / norm  # нормировка X = G/‖G‖
    transposed = False  # признак транспонирования
    if X.shape[0] > X.shape[1]:  # строк больше, чем столбцов
        X = X.T  # работать с транспонированной матрицей
        transposed = True  # запомнить транспонирование
    I_m = t.eye(X.shape[0], dtype=t.float64, device=X.device)  # единичная матрица размера X.shape[0] на том же устройстве
    for _ in range(iters):  # iters итераций
        X = 0.5 * (3.0 * I_m - X @ X.T) @ X  # итерация: X <- 0.5·(3I − X·Xᵀ)·X
    return X.T if transposed else X  # вернуть исходную форму, если транспонировали


def train_task_ef21_muon_gpu(W: np.ndarray, ef_v: np.ndarray,  # веса W и вектор ошибки EF21 ef_v
                             U_basis: np.ndarray, X: np.ndarray, Y: np.ndarray,  # базис защищённого подпространства U, входы X, цели Y
                             steps: int = 35, lr: float = 0.2  # число шагов и начальный шаг обучения
                             ) -> tuple[np.ndarray, np.ndarray, float]:  # возвращает (W, ef_v, итоговая MSE)
    """Обучение EF21-Muon на GPU; повторяет ContinualGPMMemory.train_task_ef21_muon.

    На каждом шаге: err = X·W − Y; grad = (2/n)·Xᵀ·err; g = P(grad), где
    P(G) = G − U·(Uᵀ·G) (при пустом U — сам G); ef_v = ef_v + 0.5·(g − ef_v);
    step = P(NS(ef_v)). Затем обратный поиск: для eta из (lr, lr/2, lr/10, 0.01)
    кандидат W_c = W − eta·step принимается, если MSE(W_c) < MSE(W), MSE = mean((X·W − Y)²).
    Возвращает (W, ef_v, MSE); вызывающий код записывает их обратно в память."""
    t = _torch()  # модуль torch
    dev = "cuda"  # устройство — GPU
    Xt = t.as_tensor(X, dtype=t.float64, device=dev)  # входы X на GPU в float64
    Yt = t.as_tensor(Y, dtype=t.float64, device=dev)  # цели Y на GPU в float64
    Wt = t.as_tensor(W, dtype=t.float64, device=dev)  # веса W на GPU в float64
    Vt = t.as_tensor(ef_v, dtype=t.float64, device=dev)  # ef_v на GPU в float64
    Ut = (t.as_tensor(U_basis, dtype=t.float64, device=dev)  # базис U на GPU в float64,
          if U_basis.shape[1] else None)  # если у него есть столбцы, иначе None
    n = Xt.shape[0]  # число примеров n

    def proj(G):  # проекция на дополнение подпространства, натянутого на столбцы U
        """Спроецировать G: G − U·(Uᵀ·G); при пустом базисе — само G."""
        return G if Ut is None else G - Ut @ (Ut.T @ G)  # P(G) = G − U·(Uᵀ·G), если U задан, иначе G

    for _ in range(steps):  # steps шагов
        err = Xt @ Wt - Yt  # ошибка err = X·W − Y
        grad = (2.0 / n) * (Xt.T @ err)  # градиент grad = (2/n)·Xᵀ·err
        g_proj = proj(grad)  # проекция градиента g = P(grad)
        Vt = Vt + 0.5 * (g_proj - Vt)  # обновление вектора ошибки: ef_v = ef_v + 0.5·(g − ef_v)
        step = proj(_newton_schulz_torch(Vt, iters=4))  # step = P(NS(ef_v)) — 4 итерации Ньютона — Шульца
        cur = float(t.mean((Xt @ Wt - Yt) ** 2))  # текущая ошибка cur = mean((X·W − Y)²)
        for eta in (lr, lr * 0.5, lr * 0.1, 0.01):  # обратный поиск: eta из (lr, 0.5·lr, 0.1·lr, 0.01)
            cand = Wt - eta * step  # кандидат W_c = W − eta·step
            if float(t.mean((Xt @ cand - Yt) ** 2)) < cur:  # принять, если ошибка кандидата меньше: MSE(W_c) < cur
                Wt = cand  # принять кандидата
                break  # прекратить перебор eta
    mse = float(t.mean((Xt @ Wt - Yt) ** 2))  # итоговая среднеквадратичная ошибка MSE = mean((X·W − Y)²)
    W_out = Wt.cpu().numpy()  # веса обратно в numpy
    V_out = Vt.cpu().numpy()  # ef_v обратно в numpy
    t.cuda.synchronize()  # дождаться окончания операций GPU
    return W_out, V_out, mse  # вернуть (W, ef_v, MSE)


def consolidate_subspace_gpu(X: np.ndarray, U_basis: np.ndarray,  # новые данные X и прежний базис U
                             energy_thresh: float = 0.995,  # доля энергии, которую должен удерживать базис
                             max_basis: int | None = None) -> np.ndarray:  # предел числа столбцов; возвращает новый U_basis
    """Укрепление подпространства через SVD на GPU; возвращает новый базис U_basis.

    Остаток res = X − (X·U)·Uᵀ (при пустом U — X); если ‖res‖ < 1e-10, базис не меняется.
    res = U_s·diag(S)·Vh; отбрасываются сингулярные числа ≤ 1e-10. Пусть
    cum_j = Σ_{i≤j} S_i² / Σ_i S_i². Число направлений k = len(S) при energy_thresh ≥ 1.0,
    иначе min(len(S), i + 1), где i — позиция energy_thresh в cum (searchsorted).
    Новые направления — первые k правых сингулярных векторов. Пустой базис — QR-ортогонализация
    направлений; иначе U и направления объединяются в одну матрицу, SVD оставляет
    левые сингулярные векторы с числами > 1e-9. Базис обрезается до max_basis столбцов
    (по умолчанию — число столбцов X)."""
    t = _torch()  # модуль torch
    dev = "cuda"  # устройство — GPU
    Xt = t.as_tensor(X, dtype=t.float64, device=dev)  # X на GPU в float64
    if U_basis.shape[1]:  # у базиса есть столбцы
        Ut = t.as_tensor(U_basis, dtype=t.float64, device=dev)  # базис U на GPU в float64
        res = Xt - (Xt @ Ut) @ Ut.T  # остаток res = X − (X·U)·Uᵀ
    else:  # базис пуст
        res = Xt  # остаток — сам X
    if float(res.norm()) < 1e-10:  # остаток почти нулевой — новых направлений нет
        return U_basis  # вернуть прежний базис
    U, S, Vh = t.linalg.svd(res, full_matrices=False)  # сингулярное разложение res: U·diag(S)·Vh, без полных матриц
    S = S[S > 1e-10]  # оставить сингулярные числа больше 1e-10
    Vh = Vh[: len(S)]  # соответствующие им правые сингулярные векторы
    tot = float(t.sum(S ** 2))  # суммарная энергия tot = Σ S_i²
    if tot <= 0.0:  # энергии нет
        return U_basis  # вернуть прежний базис
    cum = t.cumsum(S ** 2, dim=0) / tot  # накопленная доля энергии cum_j = Σ_{i≤j} S_i² / tot
    # energy_thresh ≥ 1.0 означает ПОЛНЫЙ охват: вещественная накопленная сумма
    # останавливается на 0.99999… и молча отбрасывает последние направления (потеря
    # ранга ломала защиту от забывания: после обучения задаче B потери по задаче A росли).
    if energy_thresh >= 1.0:  # нужен полный охват
        k = len(S)  # взять все направления
    else:  # нужен заданный порог энергии
        k = min(len(S), int(t.searchsorted(cum, t.as_tensor(energy_thresh, device=cum.device))) + 1)  # k = min(len(S), i + 1), i — позиция вставки порога в cum
    new_dirs = Vh[:k].T  # новые направления: первые k строк Vh как столбцы
    if U_basis.shape[1] == 0:  # прежний базис пуст
        Q, _ = t.linalg.qr(new_dirs)  # ортогонализация направлений: Q из QR-разложения
        out = Q  # новый базис — Q
    else:  # прежний базис есть
        combined = t.cat([Ut, new_dirs], dim=1)  # объединить U и новые направления по столбцам
        Us, ss, _ = t.linalg.svd(combined, full_matrices=False)  # SVD объединённой матрицы
        keep = ss > 1e-9  # оставить компоненты с сингулярным числом больше 1e-9
        out = Us[:, keep]  # новый базис — соответствующие левые сингулярные векторы
    if max_basis is None:  # предел не задан
        max_basis = X.shape[1]  # предел = число столбцов X
    # Держится ПОЛНЫЙ охват до max_basis: прежний запас «max_basis − 2» всегда
    # оставлял 2 направления незащищёнными, и обновления попадали в защищённые
    # направления задачи A.
    if out.shape[1] > max_basis:  # столбцов больше предела
        out = out[:, :max_basis]  # обрезать до первых max_basis столбцов
    return out.cpu().numpy()  # вернуть новый базис в numpy


class GpuGpmMixin:
    """Примесь-ускоритель для ContinualGPMMemory: работает на устройстве из resolve_device."""

    def _use_gpu(self) -> bool:
        """True, если выбранное устройство — cuda."""
        return resolve_device() == "cuda"  # сравнение результата resolve_device с "cuda"

    def train_task_ef21_muon(self, X: np.ndarray, Y: np.ndarray,  # входы и цели
                             steps: int = 35, lr: float = 0.2) -> float:  # число шагов и шаг обучения; возвращает MSE
        """Обучить задачу EF21-Muon на GPU при доступном GPU, иначе на CPU; возвращает итоговую MSE.

        Пустой X даёт 0.0. На GPU результат записывается в self.W и self.ef_v."""
        if len(X) == 0:  # примеров нет
            return 0.0  # ошибка 0.0
        if self._use_gpu():  # выбран GPU
            W, ef_v, mse = train_task_ef21_muon_gpu(  # обучение на GPU с весами, ef_v и базисом памяти
                self.W, self.ef_v, self.U_basis, X, Y, steps=steps, lr=lr)  # вернуть (W, ef_v, MSE)
            self.W, self.ef_v = W, ef_v  # записать веса и ef_v обратно
            return mse  # вернуть MSE
        return self._train_task_ef21_muon_cpu(X, Y, steps=steps, lr=lr)  # процессорная реализация с теми же аргументами

    def consolidate_subspace(self, X: np.ndarray,  # новые данные
                             energy_thresh: float = 0.995) -> None:  # доля удерживаемой энергии
        """Укрепить базис подпространства по X: на GPU — с пределом max_basis = числу строк W, иначе на CPU."""
        if self._use_gpu():  # выбран GPU
            self.U_basis = consolidate_subspace_gpu(  # новый базис из GPU-реализации
                X, self.U_basis, energy_thresh=energy_thresh,  # по X, прежнему базису и порогу энергии
                max_basis=self.W.shape[0])  # предел = число строк W
            return  # готово
        self._consolidate_subspace_cpu(X, energy_thresh=energy_thresh)  # процессорная реализация
