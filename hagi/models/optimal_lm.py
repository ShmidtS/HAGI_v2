"""HagiOptimalLM — причинная байтовая модель: мульти-лаговая свёртка, хэш-биграммы,
двухуровневый Gated DeltaNet с гейтом границ, SwiGLU, связанные эмбеддинги.

Перенесена с numpy-шима (аудит checkpoints/audit.json, запрос 055) на torch.
Формат последовательный: forward_seq даёт логиты (B, T, 256) по всем позициям
сразу, поэтому потеря считается по T целей за один прогон (у HNetByteLM — одна).
Оптимизатор HybridMuonAdamW: Muon (Ньютон–Шульц, ортогонализация) для матриц,
AdamW для векторов; опционально проекция на дополнение подпространства GPM."""
from __future__ import annotations  # аннотации не вычисляются при импорте

import math  # квадратный корень, косинус, число пи
import os  # переменные окружения
from typing import Any  # произвольный тип

import numpy as np  # численные массивы
import torch as t  # тензоры и автоградиент
from hagi.base import config  # чтение HAGI_DEVICE (cpu / cuda / auto)


def rms_norm(x: t.Tensor, g: t.Tensor, eps: float = 1e-6) -> t.Tensor:
    """Норма RMS по последней оси с масштабом g: x / sqrt(mean(x²) + eps) · g."""
    return x * t.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * g  # нормировать и умножить на масштаб


def causal_lags(E: t.Tensor, w_lag: t.Tensor, lags: tuple[int, ...]) -> t.Tensor:
    """Причинная сумма E[:, t − lag, :] · w_lag[i] по лагам; на позициях t < lag вклад нулевой.

        E: (B, T, d), w_lag: (L, d). Возвращает (B, T, d)."""
    out = t.zeros_like(E)  # накопитель
    T = E.shape[1]  # длина последовательности
    for i, lag in enumerate(lags):  # по лагам
        if lag < T:  # лаг помещается в последовательность
            out[:, lag:, :] = out[:, lag:, :] + E[:, : T - lag, :] * w_lag[i]  # сдвинуть и взвесить
    return out  # результат


def swiglu(x: t.Tensor, W_gate: t.Tensor, W_up: t.Tensor, W_down: t.Tensor) -> t.Tensor:
    """SwiGLU-блок: (silu(x·W_gate) ⊙ (x·W_up)) · W_down."""
    g_pre = x @ W_gate  # вентиль
    g_act = g_pre * t.sigmoid(g_pre)  # SiLU-активация
    return (g_act * (x @ W_up)) @ W_down  # произведение и проекция назад


def _bypass_boost_default() -> float:
    """Прибавка к логиту точного словаря по умолчанию: HAGI_BYPASS_BOOST (по умолчанию 8); «0» — выключено."""
    return float(os.environ.get("HAGI_BYPASS_BOOST", "8").strip() or 0.0)  # число; пусто — 0


BYPASS_T5_MIN = 3  # словарь 5 байтов отвечает после стольких одинаковых наблюдений
BYPASS_T3_MIN = 6  # словарь 3 байтов — после стольких
BYPASS_T5_W, BYPASS_T3_W = 0.85, 0.65  # доли прибавки для 5- и 3-байтового словаря


def _spike_k_default() -> float | None:  # настройка спайкинга из окружения
    """Порог спайкинга по умолчанию: HAGI_SPIKE_K (по умолчанию 3); «0» или пусто — выключено."""
    v = os.environ.get("HAGI_SPIKE_K", "3").strip()  # значение окружения
    return float(v) if v not in ("", "0") else None  # число или None (выключено)


# Предел числа контрольных окон при обучении (равномерно по хвосту контроля). Выбор лучшего снимка
# по контрольной CE — «подглядывание» среди N оцененных состояний; двусторонняя гарантия
# (двусторонняя дельта гейта контрольной выборки) при N = 10, δ = 0.05, ε = 0.05·B
# требует k ≥ ln(2N/δ)/(2(ε/B)²) = 1199 окон. Замер (3 новых домена, 4 эпохи): 256 — выигрыш новых
# 4.46, потеря старых 0.114, 557 с; 2048 — 4.47 / 0.108, 651 с: качество то же, +17% времени.
VAL_WINDOWS = int(os.environ.get("HAGI_VAL_WINDOWS", "2048"))  # предел числа контрольных окон
SPIKE_START = 0.8  # спайкинг включается с этой доли шагов обучения (последние 20%)


def gated_deltanet(Q: t.Tensor, K: t.Tensor, V: t.Tensor, alpha: t.Tensor, beta: t.Tensor, S_init: t.Tensor | None = None,  # запросы, ключи, значения, затухание, запись, состояние
                   chunk: int = 16) -> tuple[t.Tensor, t.Tensor]:  # размер чанка; выходы и финальное состояние
    """Канонический Gated DeltaNet: S_t = α_t S_{t−1} + β_t (v_t − α_t S_{t−1} k_t) k_tᵀ; y_t = S_t q_t.

        Q, K, V: (B, H, T, D); alpha, beta: (B, H, T); S_init — начальное состояние (B, H, D, D) или None.
        Формула — gdnUpdate из GDNBridge: остаток считается от ЗАТУХШЕГО состояния α_t S_{t−1}.
        Прежняя запись брала остаток от незатухшего S_{t−1}, и при жёстком сбросе (α = 0)
        старое состояние просачивалось в память (замер: невязка 4.76 вместо 0).
        Два вида вычисления — математически одна и та же рекурсия:
          - T кратно chunk и T >= 2·chunk: блочно-параллельная WY-форма. Внутри блока
            неизвестные u_t = β_t (v_t − α_t S_{t−1} k_t) связаны треугольной системой
            (I + strict_lower(β_t · decay · K Kᵀ)) u = β (v − g S₀ k), где
            g_t = Π_{i<=t} α_i, decay[t, j] = Π_{j<i<=t} α_i; она решается одним
            batched solve_triangular, внешний цикл — T / chunk шагов вместо T;
          - иначе: пошаговая рекурсия (любая длина)."""
    B, H, T, D = Q.shape  # размеры
    S = t.zeros((B, H, D, D), dtype=Q.dtype, device=Q.device) if S_init is None else S_init.to(Q.dtype)  # состояние: нулевое или перенесённое
    if T % chunk != 0 or T < 2 * chunk:  # длина не делится на чанк или короткая
        outs = []  # выходы по шагам
        for i in range(T):  # по времени
            k, v, q = K[:, :, i], V[:, :, i], Q[:, :, i]  # ключ, значение, запрос шага
            a, b = alpha[:, :, i][..., None, None], beta[:, :, i][..., None, None]  # затухание и скорость записи
            aS = a * S  # затухшее состояние α_t S_{t−1}
            S = aS + b * ((v - (aS @ k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1) @ k.unsqueeze(-2))  # запись: S_t = aS + β (v − aS k) kᵀ
            outs.append((S @ q.unsqueeze(-1)).squeeze(-1))  # выход: y_t = S_t q_t
        return t.stack(outs, dim=2), S  # выходы и финальное состояние
    C, N = chunk, T // chunk  # длина чанка и число чанков
    Qc, Kc, Vc = (x.reshape(B, H, N, C, D) for x in (Q, K, V))  # те же тензоры по чанкам
    ac, bc = alpha.reshape(B, H, N, C), beta.reshape(B, H, N, C)  # гейты по чанкам
    g = t.cumprod(ac, dim=-1)  # g_t = Π_{i<=t} α_i внутри чанка
    # decay[t, j] = Π_{j<i<=t} α_i (единица при t = j): строка t = e_t + α_t · строка t−1
    eye = t.eye(C, dtype=Q.dtype, device=Q.device)  # единичная матрица C×C
    decay = t.zeros((B, H, N, C, C), dtype=Q.dtype, device=Q.device)  # накопитель матрицы затухания
    prev = t.zeros((B, H, N, C), dtype=Q.dtype, device=Q.device)  # предыдущая строка (строка t−1)
    for i in range(C):  # по строкам
        prev = ac[..., i:i + 1] * prev + eye[i]  # строка i: α_i · (строка i−1) + e_i
        decay[..., i, :] = prev  # записать строку
    lower = t.tril(t.ones((C, C), dtype=Q.dtype, device=Q.device), diagonal=-1)  # строго нижняя треугольная маска
    L = eye + lower * (bc[..., None] * decay * (Kc @ Kc.transpose(-1, -2)))  # треугольная система: I + strict_lower(β_t · decay · K Kᵀ)
    W_attn = t.tril(Qc @ Kc.transpose(-1, -2)) * decay  # вклад внутричанковых записей в выход: tril(Q Kᵀ) · decay
    outs = []  # выходы по чанкам
    for n in range(N):  # по чанкам — T / chunk итераций
        Kn, Qn, Vn = Kc[:, :, n], Qc[:, :, n], Vc[:, :, n]  # чанк ключей, запросов, значений
        SK = (S @ Kn.transpose(-1, -2)).transpose(-1, -2)  # S₀ k_j по чанку: (B, H, C, D)
        SQ = (S @ Qn.transpose(-1, -2)).transpose(-1, -2)  # S₀ q_t по чанку
        rhs = bc[:, :, n][..., None] * (Vn - g[:, :, n][..., None] * SK)  # правая часть: β (v − g S₀ k)
        u = t.linalg.solve_triangular(L[:, :, n], rhs, upper=False, unitriangular=True)  # решения u_t треугольной системы
        outs.append(g[:, :, n][..., None] * SQ + W_attn[:, :, n] @ u)  # выходы чанка: g S₀ q + Σ_j decay (qᵀk_j) u_j
        S = g[:, :, n, -1][..., None, None] * S + (u * decay[:, :, n, -1][..., None]).transpose(-1, -2) @ Kn  # состояние после чанка
    return t.cat(outs, dim=2), S  # выходы и финальное состояние


def muon_orth(G: t.Tensor, iters: int = 5) -> t.Tensor:
    """Ортогонализация полярным множителем: polar_express (3 квинтических шага,
        полярная факторизация без SVD; «высокая» матрица
        транспонируется в «широкую»). Считается на устройстве тензора в float64 — без
        копирования GPU → CPU → GPU на каждом шаге."""
    from hagi.base.trunk_ops import POLAR_EXPRESS  # тройки коэффициентов квинтических шагов
    X = G.detach().to(t.float64)  # копия в float64 на том же устройстве
    nrm = float(X.norm())  # норма Фробениуса ‖X‖ (один синхронный вызов)
    if nrm < 1e-12:  # норма практически нулевая
        return G.detach().clone()  # без изменений
    X = X / nrm  # нормировка: X = X/‖X‖
    transposed = X.shape[0] > X.shape[1]  # «высокая» матрица
    if transposed:  # нужно транспонировать
        X = X.T  # транспонирование
    for a, b, c in POLAR_EXPRESS:  # по тройкам коэффициентов
        X2 = X @ X.T  # произведение на транспонированную
        X = a * X + b * (X2 @ X) + c * (X2 @ X2 @ X)  # квинтический шаг
    if transposed:  # вернуть форму
        X = X.T  # обратно
    return X.to(G.dtype)  # в тип градиента


class HybridMuonAdamW:
    """Muon для 2D-весов, AdamW для остальных; проекция градиента и шага на дополнение GPM."""

    def __init__(self, matrix_params: list[t.Tensor], vec_params: list[t.Tensor],  # параметры по группам
                 lr_muon: float = 0.03, lr_adam: float = 5e-3, momentum: float = 0.9,  # шаги и момент
                 wd: float = 1e-4, ns_iters: int = 5, beta2: float = 0.999) -> None:  # затухание, число итераций и второй момент AdamW
        self.beta2 = float(beta2)  # второй момент AdamW
        self.mats = [p for p in matrix_params if p.requires_grad]  # матрицы (Muon)
        self.names: list[str] = []  # имена весов в порядке mats + vecs (заполняет модель: opt.names = model.param_names(); ключи GPM — имена, стабильны между перезапусками)  # прежльны при save/load)
        self.vecs = [p for p in vec_params if p.requires_grad]  # векторы и таблицы (AdamW)
        self.lr_muon, self.lr_adam = float(lr_muon), float(lr_adam)  # шаги
        self.momentum, self.wd, self.ns_iters = float(momentum), float(wd), int(ns_iters)  # прочее
        self.m_buf: list[t.Tensor] = [t.zeros_like(p) for p in self.mats]  # момент Muon
        self.v_m: list[t.Tensor] = [t.zeros_like(p) for p in self.vecs]  # первый момент AdamW
        self.v_v: list[t.Tensor] = [t.zeros_like(p) for p in self.vecs]  # второй момент AdamW
        self.step_count = 0  # счётчик шагов
        self.gpm_left: dict[str, t.Tensor] = {}  # ИМЯ веса -> базис левого подпространства (столбцы)
        self.gpm_right: dict[str, t.Tensor] = {}  # ИМЯ веса -> базис правого подпространства (столбцы)
        self._scale = 1.0  # множитель шага (косинусное расписание)
        # Экспоненциальное среднее весов (EMA): сглаживает шум шагов; в конце обучения сравнивается
        # с лучшим снимком по контролю и берётся лучшее. Замер (3 зерна, код): лучший снимок
        # 0.545/0.531/0.499 против EMA 0.493/0.499/0.497.
        self.use_ema = os.environ.get("HAGI_EMA", "1") != "0"  # включено по умолчанию; HAGI_EMA=0 выключает
        self.ema: list[t.Tensor] | None = ([p.detach().clone().float() for p in self.mats + self.vecs] if self.use_ema else None)  # копии весов для среднего

    def zero_grad(self) -> None:  # сбросить градиенты
        for p in self.mats + self.vecs:  # по всем параметрам
            p.grad = None  # сбросить

    def set_lr_scale(self, scale: float) -> None:  # множитель шага
        self._scale = float(scale)  # запомнить

    @staticmethod  # статический метод (без self)
    def _proj_out(x: t.Tensor, Ul: t.Tensor | None, Ur: t.Tensor | None) -> t.Tensor:  # проекция на дополнение
        if Ul is not None and Ul.shape[1] > 0:  # есть левое подпространство
            x = x - Ul @ (Ul.T @ x)  # убрать компоненту в Ul
        if Ur is not None and Ur.shape[1] > 0:  # есть правое
            x = x - (x @ Ur) @ Ur.T  # убрать компоненту в Ur
        return x  # результат

    def step(self) -> None:  # один шаг
        self.step_count += 1  # счётчик
        lr_m = self.lr_muon * self._scale  # шаг Muon
        lr_a = self.lr_adam * self._scale  # шаг AdamW
        with t.no_grad():  # без графа
            for idx, p in enumerate(self.mats):  # матрицы
                if p.grad is None:  # нет градиента
                    continue  # пропустить
                Ul, Ur = self.gpm_left.get(self.names[idx]), self.gpm_right.get(self.names[idx])  # подпространства веса (по имени)
                g = self._proj_out(p.grad.float(), Ul, Ur)  # градиент вне защищённого подпространства
                buf = self.m_buf[idx]  # буфер момента
                buf.mul_(self.momentum).add_(g)  # момент: затухание плюс новый градиент
                upd = muon_orth(g + self.momentum * buf, self.ns_iters)  # Нестеров + ортогонализация
                r, c = p.shape  # размеры
                upd = upd * math.sqrt(max(1.0, r / max(1, c)))  # масштаб по форме
                upd = self._proj_out(upd, Ul, Ur)  # шаг не заходит в подпространство
                W = p.float()  # веса
                if self.wd > 0:  # затухание — проецированное (WeightDecay: не трогает защищённое)
                    W = W - (lr_m * self.wd) * self._proj_out(W, Ul, Ur)  # проецированное затухание: вне защищённого подпространства
                p.copy_((W - lr_m * upd).to(p.dtype))  # записать
            b1, b2, eps = 0.9, self.beta2, 1e-8  # коэффициенты AdamW (второй момент — задаваемый)
            ts = self.step_count  # номер шага
            for idx, p in enumerate(self.vecs):  # векторы и таблицы
                if p.grad is None:  # нет градиента
                    continue  # пропустить
                Ul, Ur = self.gpm_left.get(self.names[len(self.mats) + idx]), self.gpm_right.get(self.names[len(self.mats) + idx])  # подпространства (по имени — те же ключи, что у базисов)
                g = self._proj_out(p.grad.float(), Ul, Ur)  # проекция градиента
                if self.wd > 0 and p.ndim >= 2 and Ul is None and Ur is None:  # затухание только для незащищённых таблиц
                    # ленивое затухание (LazyAdam: lazyDecay_exact): таблица — это строки по байтам/токенам;
                    # строка, которой не было в батче (нулевой градиент), не затухает — иначе редкие байты
                    # сжимаются к нулю, сколько бы их ни видели
                    act = (p.grad != 0).any(dim=-1, keepdim=True)  # (rows, 1): строка участвовала в батче
                    p.mul_(t.where(act, t.as_tensor(1.0 - lr_a * self.wd, device=p.device), t.as_tensor(1.0, device=p.device)))  # затухание только активных строк
                self.v_m[idx].mul_(b1).add_((1.0 - b1) * g)  # первый момент
                self.v_v[idx].mul_(b2).add_((1.0 - b2) * g * g)  # второй момент
                mh = self.v_m[idx] / (1.0 - b1 ** ts)  # коррекция смещения
                vh = self.v_v[idx] / (1.0 - b2 ** ts)  # коррекция смещения
                upd_v = self._proj_out(lr_a * mh / (vh.sqrt() + eps), Ul, Ur)  # шаг AdamW с проекцией
                p.copy_((p.float() - upd_v).to(p.dtype))  # записать
            if self.ema is not None:  # EMA включён
                d = min(0.99, (1.0 + self.step_count) / (10.0 + self.step_count))  # скорость среднего: растёт от ~0.09 к 0.99
                with t.no_grad():  # без графа
                    for e, p_ in zip(self.ema, self.mats + self.vecs):  # по всем параметрам (и без градиента тоже)
                        e.mul_(d).add_((1.0 - d) * p_.detach().float())  # среднее: e умножить на d и прибавить (1−d)·p


class HagiOptimalLM:
    """Причинная модель байтовых последовательностей: мульти-лаги, биграммы, двухуровневый Gated DeltaNet, SwiGLU, связанные эмбеддинги."""

    VOCAB = 256  # число байтов
    LAGS = (1, 2, 3, 4, 6, 8)  # лаги локальной свёртки

    def __init__(self, dim: int = 24, hidden: int = 48, n_heads: int = 2, d_bi: int = 8,  # ширины
                 use_hier_gdn: bool = True, use_bigram_logit_skip: bool = True, use_gdn_norm: bool = True,  # переключатели
                 k: int = 64, seed: int = 7) -> None:  # контекст и зерно
        assert dim % n_heads == 0  # ширина делится на головы
        self.dim, self.hidden, self.n_heads, self.k, self.d_bi = dim, hidden, n_heads, k, d_bi  # сохранить
        self.use_hier_gdn, self.use_bigram_logit_skip, self.use_gdn_norm = use_hier_gdn, use_bigram_logit_skip, use_gdn_norm  # сохранить
        self.seed = seed  # зерно инициализации
        # Устройство модели: HAGI_DEVICE=cpu -> процессор; иначе GPU, если доступен (у рабочей модели цикл
        # на GPU в ~2.7 раза быстрее: замер 3 доменов 193 с против 519 с). Порог dim·hidden >= 8192 общего
        # правила auto здесь не применяется: он защищает крошечные тестовые модели, а не рабочую.
        if config.device() == "cpu" or not t.cuda.is_available():  # процессор запрошен или GPU нет
            self.device = "cpu"  # процессор
        else:  # GPU доступен
            self.device = "cuda"  # видеокарта
        self.t = t  # модуль torch (Unified._windows берёт torch из модели)
        self.adapter_dim = dim  # ширина, к которой применяется доменный адаптер
        self.rms_eps = 1e-6  # eps RMSNorm (масштабируется при росте ширины)
        self.domain_adapter = None  # доменный адаптер (пара A, B) — не задан
        self.active_fiber = None  # активное доменное волокно (CortexFiber) — не задано
        g = t.Generator().manual_seed(seed)  # генератор инициализации

        def p(shape: tuple[int, ...], sc: float = 0.05) -> t.nn.Parameter:  # параметр N(0, sc²)
            return t.nn.Parameter(t.randn(*shape, generator=g) * sc)  # создать

        H, d = n_heads, dim  # короткие имена
        self.emb = p((self.VOCAB, d), 0.12)  # таблица байтов (она же — выходная проекция, weight tying)
        self.w_lag = t.nn.Parameter(t.ones((len(self.LAGS), d)) / len(self.LAGS))  # веса лагов
        self.W_loc = p((d, d), 0.08); self.g_loc = t.nn.Parameter(t.ones(d))  # локальная проекция и масштаб
        self.E_bi = p((256, d_bi), 0.08); self.W_bi = p((d_bi, d), 0.08)  # хэш-биграммные вложения
        if use_bigram_logit_skip:  # факторизованный скип в логиты
            self.B_u = p((256, 8), 0.04); self.B_v = p((8, 256), 0.04)  # ранг 8
        self.w_sc = t.nn.Parameter(t.tensor([[0.7] * d, [0.2] * d, [0.1] * d]))  # 3-tap короткая свёртка
        self.W_gq, self.W_gk, self.W_gv = p((d, d), 0.08), p((d, d), 0.08), p((d, d), 0.08)  # GDN уровень 1: запросы, ключи, значения
        self.W_go = p((d, d), 0.05)  # выход GDN 1
        self.W_alpha = p((d, H), 0.05); self.b_alpha = t.nn.Parameter(t.linspace(0.5, 2.5, H))  # удержание
        self.W_beta = p((d, H), 0.05); self.b_beta = t.nn.Parameter(t.zeros(H))  # запись
        if use_gdn_norm:  # нормировка выхода GDN 1
            self.g_gdn1 = t.nn.Parameter(t.ones(d))  # масштаб
        if use_hier_gdn:  # уровень 2 — медленный семантический
            self.w_bnd = p((2 * d, 1), 0.08); self.b_bnd = t.nn.Parameter(t.tensor([-0.5]))  # гейт границ
            self.W_hq, self.W_hk, self.W_hv = p((d, d), 0.08), p((d, d), 0.08), p((d, d), 0.08)  # Q, K, V уровня 2
            self.W_ho = p((d, d), 0.04)  # выход GDN 2
            self.a_slow = t.nn.Parameter(t.full((H,), 2.0)); self.b_slow = t.nn.Parameter(t.zeros(H))  # медленные удержание/запись
            if use_gdn_norm:  # нормировка выхода GDN 2
                self.g_gdn2 = t.nn.Parameter(t.ones(d))  # масштаб
        self.g_mlp = t.nn.Parameter(t.ones(d))  # масштаб перед MLP
        self.W_gate, self.W_up, self.W_down = p((d, hidden), 0.08), p((d, hidden), 0.08), p((hidden, d), 0.04)  # SwiGLU-блок
        self.g_out = t.nn.Parameter(t.ones(d)); self.out_scale = t.nn.Parameter(t.tensor([1.0]))  # масштаб выхода
        self.b_out = t.nn.Parameter(t.zeros(self.VOCAB))  # смещение логитов
        self.blocks: list[tuple[t.nn.Parameter, t.nn.Parameter, t.nn.Parameter]] = []  # дополнительные SwiGLU-блоки (рост глубины)
        self.active_adapter: tuple[t.Tensor, t.Tensor, t.Tensor] | None = None  # активный доменный адаптер (A, B, масштаб)
        self.spike_k: float | None = _spike_k_default()  # порог спайкинга входа SwiGLU (средний модуль / k); None — выключен
        self.spike_off = False  # True — спайкинг временно выключен (первые 80% шагов обучения)
        self.spike_zeros = 0  # счётчик нулевых компонент прокинутое входа (для доли нулей)
        self.spike_total = 0  # счётчик всех компонент
        self.spike_width = self.dim  # ширина, на которую делится сумма модулей порога: при росте ширины не меняется — новые нулевые каналы не сдвигают порог, функция сохраняется
        self.bypass_boost = _bypass_boost_default()  # прибавка к логиту точного словаря (0 — выключен)
        self.bypass_t5: dict[bytes, list] = {}  # 5-байтовый словарь: контекст -> [предсказание, счёт, сбой]
        self.bypass_t3: dict[bytes, list] = {}  # 3-байтовый словарь: контекст -> [предсказание, счёт, сбой]
        self.sync_device()  # параметры — на устройство модели


    def matrix_and_vec_params(self) -> tuple[list[t.nn.Parameter], list[t.nn.Parameter]]:  # группы параметров
        mats = [self.W_loc, self.W_bi, self.W_gq, self.W_gk, self.W_gv, self.W_go,  # матрицы
                self.W_alpha, self.W_beta, self.W_gate, self.W_up, self.W_down]  # продолжение списка
        vecs = [self.emb, self.w_lag, self.g_loc, self.E_bi, self.w_sc, self.b_alpha, self.b_beta,  # векторы и таблицы
                self.g_mlp, self.g_out, self.out_scale, self.b_out]  # продолжение списка
        if self.use_gdn_norm:  # нормировки
            vecs.append(self.g_gdn1)  # уровень 1
        if self.use_bigram_logit_skip:  # скип
            mats.extend([self.B_u, self.B_v])  # низкого ранга
        if self.use_hier_gdn:  # уровень 2
            mats.extend([self.w_bnd, self.W_hq, self.W_hk, self.W_hv, self.W_ho])  # матрицы
            vecs.extend([self.b_bnd, self.a_slow, self.b_slow])  # векторы
            if self.use_gdn_norm:  # нормировка
                vecs.append(self.g_gdn2)  # уровень 2
        for Wg, Wu, Wd in self.blocks:  # дополнительные блоки
            mats.extend([Wg, Wu, Wd])  # все — матрицы
        return mats, vecs  # вернуть группы

    def train_guarded(self, data: bytes, epochs: int = 60, holdout: float = 0.15, batch: int | None = None, log_every: int = 0, holdout_seed: int | None = None, lr: float | None = None, **_ignored) -> dict:  # синоним train_certified (интерфейс Rope/HNet)
        return self.train_certified(data, epochs=epochs, holdout=holdout, batch=batch, log_every=log_every, holdout_seed=holdout_seed, lr=lr)  # делегировать

    def grow(self, seed: int = 0) -> int:  # рост глубины (интерфейс Unified): блок-тождество, W_down = 0
        return grow_depth(self, seed)  # делегировать функции модуля

    def full_logits(self, ctx: t.Tensor) -> t.Tensor:
        """Логиты следующего байта для каждого префикса ctx[:, :i] (B, L, 256) — ОДНИМ причинным проходом.
        Позиция i даёт предсказание байта i+1 по ctx[:, :i+1]."""
        return self.forward_seq(ctx)  # forward_seq причинный: позиция i видит только ctx[:, :i+1]

    def generate(self, prompt, n_bytes: int = 120, temp: float = 0.5, seed: int = 1):  # продолжение байтов
        """Продолжить prompt (байты или str) на n_bytes байтов; temp → 0 — argmax. Возвращает bytes."""
        if isinstance(prompt, str):  # строка
            prompt = prompt.encode("utf-8")  # в байты
        if len(prompt) < self.k:  # контекст короче окна
            prompt = b" " * (self.k - len(prompt)) + prompt  # дополнить пробелами слева
        g = t.Generator().manual_seed(int(seed))  # генератор
        buf = list(prompt[-self.k:])  # окно
        out = bytearray()  # результат
        with t.no_grad():  # без графа
            for _ in range(int(n_bytes)):  # по байтам
                lg = self.forward_seq(t.as_tensor(buf, dtype=t.long)[None])[:, -1, :][0]  # логиты последней позиции
                if temp <= 1e-6:  # жадно
                    b = int(lg.argmax())  # байт с максимальным логитом
                else:  # выбор по температуре
                    pr = t.softmax(lg / float(temp), dim=-1)  # вероятности
                    b = int(t.multinomial(pr, 1, generator=g))  # выбрать
                out.append(b); buf = buf[1:] + [b]  # добавить и сдвинуть
        return bytes(out)  # байты

    def train_certified(self, data: bytes, epochs: int = 60, holdout: float = 0.15, batch: int | None = None,  # обучение, совместимое с Unified
                        log_every: int = 0, holdout_seed: int | None = None, lr: float | None = None) -> dict:  # (имена аргументов как у HNetByteLM)
        """Сертифицированное обучение: train_optimal + лучшее по контролю; GPM-базисы берутся из self.gpm_left, если заданы."""
        if batch is None:  # пакет не задан
            from hagi.selfdev.evolve import current as _evb  # принятые эволюцией параметры
            batch = int(_evb()["batch"])  # пакет из эволюции (batch=0 тоже значит «не задан»)
        r = self.train_optimal(data, epochs=max(1, epochs), batch=batch, seq_len=self.k, seed=holdout_seed or 7, log_every=log_every, holdout=holdout)  # обучить (GPM — из self.gpm_left)
        return {"best_holdout_ce": r["val_ce_after"], "holdout_ce": r["val_ce_after"], "val_ce_before": r["val_ce_before"], "steps": r["steps"],  # CE до и число шагов
                "spike_k": r.get("spike_k"), "spike_zero_frac": r.get("spike_zero_frac")}  # порог спайкинга и доля нулей  # как у HNet

    # ---- доменный адаптер в стиле HNet: h_out ← h_out + (h_out @ Aᵀ) @ Bᵀ, A (r, d), B (d, r) ----
    def _apply_adapter(self, h_out: t.Tensor) -> t.Tensor:  # применить доменный адаптер к нормированному скрытому
        ad = getattr(self, "domain_adapter", None)  # адаптер или None
        if ad is None:  # нет
            return h_out  # без изменений
        At = t.as_tensor(ad.A, device=h_out.device, dtype=h_out.dtype); Bt = t.as_tensor(ad.B, device=h_out.device, dtype=h_out.dtype)  # A, B как тензоры
        return h_out + (h_out @ At.T) @ Bt.T  # добавка того же вида, что у HNet

    def logits(self, ctx: t.Tensor) -> t.Tensor:  # логиты следующего байта (B, 256) с адаптером, если задан
        ctx = ctx.to(self.device)  # контекст — на устройство модели (вход может прийти с процессора)
        h_out = rms_norm(self.forward_hidden(ctx), self.g_out)  # скрытое последней позиции
        h_out = self._apply_adapter(h_out)  # адаптер домена
        h_last = h_out[:, -1, :]  # последняя позиция
        lg = (h_last @ self.emb.T) * self.out_scale + self.b_out  # связанные эмбеддинги
        if self.bypass_boost > 0:  # точный обход — та же функция, что при обучении и в forward_seq (гейты видят модель целиком)
            lg = lg + self.bypass_apply(ctx[:, -5:], t.zeros((ctx.shape[0], min(5, ctx.shape[1]), self.VOCAB), dtype=lg.dtype, device=lg.device))[:, -1]  # прибавка по последним 5 байтам (ответ последней позиции окна)
        if self.use_bigram_logit_skip:  # скип
            lg = lg + self.B_u[ctx[:, -1]] @ self.B_v  # по последнему байту
        if getattr(self, "active_fiber", None) is not None:  # волокно домена
            lg = lg + self.active_fiber.logits_delta(h_out, ctx)[:, -1, :]  # добавка волокна на последней позиции
        return lg  # логиты

    def logits_with(self, ctx: t.Tensor, At: t.Tensor, Bt: t.Tensor) -> t.Tensor:  # логиты с явной парой адаптера
        h_out = rms_norm(self.forward_hidden(ctx), self.g_out)[:, -1, :]  # скрытое последней позиции
        lg = ((h_out + (h_out @ At.T) @ Bt.T) @ self.emb.T) * self.out_scale + self.b_out  # с адаптером
        if self.use_bigram_logit_skip:  # скип
            lg = lg + self.B_u[ctx[:, -1]] @ self.B_v  # добавить
        return lg  # логиты

    def adapter_head_step(self, ctx: t.Tensor, nxt: t.Tensor, lr: float = 1e-2) -> None:  # один шаг адаптера
        """Один шаг обучения ТОЛЬКО доменного адаптера (A, B) на пакете (ctx, nxt); ствол заморожен."""
        ad = getattr(self, "domain_adapter", None)  # адаптер
        if ad is None:  # нет
            return  # ничего
        At = t.as_tensor(ad.A, device=self.device, dtype=t.float32).requires_grad_(True)  # A с градиентом
        Bt = t.as_tensor(ad.B, device=self.device, dtype=t.float32).requires_grad_(True)  # B с градиентом
        opt = t.optim.Adam([At, Bt], lr=lr)  # Адам по A и B
        opt.zero_grad()  # сброс
        t.nn.functional.cross_entropy(self.logits_with(ctx, At, Bt), nxt.to(self.device)).backward()  # контрольная кросс-энтропия и градиент
        opt.step()  # шаг
        ad.A, ad.B = At.detach(), Bt.detach()  # записать обратно

    def ce(self, ctx: t.Tensor, nxt: t.Tensor) -> t.Tensor:  # контрольная кросс-энтропия на пакете окон (одна цель на окно)
        return t.nn.functional.cross_entropy(self.logits(ctx), nxt.to(self.device))  # кросс-энтропия

    def features(self, ctx: t.Tensor) -> t.Tensor:  # скрытое последней позиции (для голов дистилляции)
        with t.no_grad():  # без графа
            return rms_norm(self.forward_hidden(ctx), self.g_out)[:, -1, :]  # форма: пакет, ширина


    ARCH = "optimal"  # имя архитектуры (для Unified.save/load)

    def param_names(self) -> list[str]:  # имена параметров строго в порядке params()
        base = {id(p): n for n, p in self.__dict__.items() if isinstance(p, t.nn.Parameter)}  # id -> имя атрибута
        mats, vecs = self.matrix_and_vec_params()  # группы (в том же порядке, что params())
        blk_names = {id(W): f"blk{i}.{nm}" for i, (Wg, Wu, Wd) in enumerate(self.blocks) for W, nm in ((Wg, "Wg"), (Wu, "Wu"), (Wd, "Wd"))}  # блоки роста
        return [base.get(id(p), blk_names.get(id(p))) for p in mats + vecs]  # по порядку params()

    def sync_device(self) -> None:
        """Перенести данные всех параметров (включая блоки роста) на self.device.
        Без переноса параметр остаётся на процессоре, а входы строятся на GPU — устройства расходятся."""
        with t.no_grad():  # без графа
            for q in self.params():  # по всем параметрам
                q.data = q.data.to(self.device)  # перенести данные параметра
            if self.active_adapter is not None:  # адаптер активен
                A, B_adp, sc = self.active_adapter  # распаковать
                self.active_adapter = (A.to(self.device), B_adp.to(self.device), sc)  # адаптер на устройстве

    def state_dict(self) -> dict[str, t.Tensor]:  # снимок весов по именам атрибутов
        return {n: p.detach().clone() for n, p in self.__dict__.items() if isinstance(p, t.nn.Parameter)}  # копии

    def load_state_dict_from(self, other: "HagiOptimalLM") -> None:  # загрузить веса из другой модели (той же формы)
        with t.no_grad():  # без графа
            for n, v in other.state_dict().items():  # по именам
                getattr(self, n).copy_(v)  # скопировать

    def params(self) -> list[t.nn.Parameter]:  # все параметры
        m, v = self.matrix_and_vec_params()  # группы
        return m + v  # одним списком

    def _rms(self, x: t.Tensor, g: t.Tensor) -> t.Tensor:  # RMSNorm с eps модели
        return rms_norm(x, g, getattr(self, "rms_eps", 1e-6))  # нормировка

    @staticmethod  # статический метод (без self)
    def _trie_put(d: dict, key: bytes, nxt: int) -> None:
        """Записать в словарь наблюдение «за key идёт nxt»: первый раз — новая запись,
        повтор того же — счёт +1, противоречие — запись навсегда помечается сбойной."""
        e = d.get(key)  # запись контекста
        if e is None:  # контекст новый
            d[key] = [nxt, 1, False]  # предсказание, один голос, без сбоя
        elif not e[2]:  # запись живая
            if e[0] == nxt:  # тот же следующий байт
                e[1] = min(e[1] + 1, 60000)  # счёт +1
            else:  # другой следующий байт
                e[2] = True  # контекст неоднозначен — больше не отвечает

    def bypass_fit(self, data: bytes) -> None:
        """Обучить точные словари причинно по потоку: за 5 и 3 предыдущими байтами — следующий.
        Записи накапливаются между доменами (старые контексты остаются в силе)."""
        if self.bypass_boost <= 0:  # обход выключен
            return  # ничего не делать
        t5, t3, put = self.bypass_t5, self.bypass_t3, self._trie_put  # словари и запись
        for i in range(5, len(data)):  # по позициям следующего байта
            put(t5, data[i - 5:i], data[i])  # голос 5-байтового контекста
            put(t3, data[i - 3:i], data[i])  # голос 3-байтового контекста

    def bypass_apply(self, x_seq: t.Tensor, lg: t.Tensor) -> t.Tensor:
        """Прибавить к логитам уверенные ответы словарей: позиция t отвечает по байтам t−4..t
        (5-байтовый словарь), иначе по t−2..t (3-байтовый). Строго причинно; вне словаря — 0."""
        if self.bypass_boost <= 0 or not (self.bypass_t5 or self.bypass_t3):  # выключен или пуст
            return lg  # без изменений
        xs = x_seq.detach().to("cpu").numpy().astype(np.uint8)  # байты (B, T) как массив по одному байту на символ
        boost, t5, t3 = self.bypass_boost, self.bypass_t5, self.bypass_t3  # прибавка и словари
        rows, cols, preds, vals = [], [], [], []  # индексы и значения прибавки
        for b in range(xs.shape[0]):  # по строкам пакета
            row = xs[b].tobytes()  # байты строки подряд
            for i in range(4, xs.shape[1]):  # по позициям, где есть 5-байтовый контекст
                e = t5.get(row[i - 4:i + 1])  # запись 5-байтового словаря
                if e is not None and not e[2] and e[1] >= BYPASS_T5_MIN:  # словарь уверен
                    rows.append(b); cols.append(i); preds.append(e[0]); vals.append(boost * BYPASS_T5_W)  # прибавка
                    continue  # 3-байтовый не спрашивается
                e = t3.get(row[i - 2:i + 1])  # запись 3-байтового словаря
                if e is not None and not e[2] and e[1] >= BYPASS_T3_MIN:  # словарь уверен
                    rows.append(b); cols.append(i); preds.append(e[0]); vals.append(boost * BYPASS_T3_W)  # прибавка 3-байтового
        if not rows:  # словари нигде не ответили
            return lg  # без изменений
        delta = t.zeros_like(lg)  # прибавка — тензор той же формы
        delta[rows, cols, preds] = t.as_tensor(vals, dtype=lg.dtype, device=lg.device)  # записать прибавки
        return lg + delta  # логиты с обходом

    def _spike(self, hn: t.Tensor) -> t.Tensor:  # нормированный вход SwiGLU (B, T, d)
        """Спайкинг входа SwiGLU: hn -> round(hn / V_th)·V_th, градиент — напрямую (STE).

        Порог V_th = сумма модулей по ширине В ДАННОЙ ПОЗИЦИИ / (ширина создания · k) — причинный
        (не заглядывает вперёд) и не зависящий от состава пакета. Значения
        кратны порогу, умножения на них сводятся к целочисленным; доля нулей —
        около 10% при k = 3. Замер: контрольная кросс-энтропия 2.1267 против
        2.1281 (+0.0014 ната)."""
        k = getattr(self, "spike_k", None)  # порог
        if k is None or getattr(self, "spike_off", False):  # выключен
            return hn  # как есть
        vth = hn.detach().abs().sum(-1, keepdim=True) / (self.spike_width * k)  # порог по позиции (B, T, 1): сумма модулей / (ширина создания · k)
        q = hn / vth  # в единицах порога
        q = q + (t.round(q) - q).detach()  # округление вперёд, тождество назад
        with t.no_grad():  # статистика без графа
            self.spike_zeros += int((q.detach() == 0).sum()); self.spike_total += q.numel()  # нули и всего
        return q * vth  # обратно в исходном масштабе

    def forward_hidden(self, x_seq: t.Tensor, capture: dict[int, t.Tensor] | None = None, state: dict | None = None) -> t.Tensor:  # скрытое состояние (B, T, d); capture — входы матриц (для GPM)
        x_seq = x_seq.to(self.device)  # порт входа: байты — на устройство модели
        B, T = x_seq.shape  # размеры
        d, H = self.dim, self.n_heads  # ширина и головы
        hd = d // H  # ширина головы
        E = self.emb[x_seq]  # вложения байтов (B, T, d)
        acc = causal_lags(E, self.w_lag, self.LAGS)  # взвешенная сумма лагов
        if capture is not None: capture[id(self.W_loc)] = (E + acc).detach().reshape(-1, d)  # вход W_loc
        x0 = rms_norm(E + acc @ self.W_loc, self.g_loc)  # локальный признак
        x_m1 = t.cat([t.zeros((B, 1), dtype=t.long, device=x_seq.device), x_seq[:, : T - 1]], dim=1)  # предыдущий байт (0 на позиции 0)
        bi_idx = ((x_m1 * 131) ^ x_seq) % 256  # хэш пары байтов
        if capture is not None: capture[id(self.W_bi)] = self.E_bi[bi_idx].detach().reshape(-1, self.E_bi.shape[1])  # вход W_bi — вложения пары байтов
        h = x0 + self.E_bi[bi_idx] @ self.W_bi  # добавить биграммный признак
        h_in = causal_lags(h, self.w_sc, (0, 1, 2))  # короткая 3-tap свёртка
        if capture is not None:  # входы Q/K/V уровня 1 — один h_in
            for W in (self.W_gq, self.W_gk, self.W_gv): capture[id(W)] = h_in.detach().reshape(-1, d)  # записать
        if capture is not None:  # вентили уровня 1 — тот же h_in
            for W in (self.W_alpha, self.W_beta): capture[id(W)] = h_in.detach().reshape(-1, d)  # записать

        def qkv(W: t.Tensor, x: t.Tensor) -> t.Tensor:  # проекция в головы (B, H, T, hd)
            return (x @ W).reshape(B, T, H, hd).permute(0, 2, 1, 3)  # переставить оси
        Q1 = t.nn.functional.normalize(qkv(self.W_gq, h_in), dim=-1)  # запросы (нормированы)
        K1 = t.nn.functional.normalize(qkv(self.W_gk, h_in), dim=-1)  # ключи (нормированы)
        V1 = qkv(self.W_gv, h_in)  # значения
        a1 = t.sigmoid(h_in @ self.W_alpha + self.b_alpha).permute(0, 2, 1) * 0.96  # удержание < 1
        b1 = t.sigmoid(h_in @ self.W_beta + self.b_beta).permute(0, 2, 1)  # запись
        y1, S1 = gated_deltanet(Q1, K1, V1, a1, b1, S_init=(state.get("S1") if state else None))  # уровень 1 (с перенесённым состоянием)
        if state is not None: state["S1"] = S1.detach().clone()  # финальное состояние (копия: in-place сброс строк не должен трогать граф)
        y1 = y1.permute(0, 2, 1, 3).reshape(B, T, d)  # в раскладку (B, T, d)
        if self.use_gdn_norm:  # нормировка
            y1 = rms_norm(y1, self.g_gdn1)  # стабилизирует длинные последовательности
        if capture is not None: capture[id(self.W_go)] = y1.detach().reshape(-1, d)  # вход W_go
        h = h + y1 @ self.W_go  # остаточное соединение
        if self.use_hier_gdn:  # уровень 2
            h_prev = t.cat([t.zeros((B, 1, d), device=h.device), h[:, : T - 1]], dim=1)  # предыдущее скрытое
            if capture is not None: capture[id(self.w_bnd)] = t.cat([h, h_prev], dim=-1).detach().reshape(-1, 2 * d)  # вход гейта границ
            p_bnd = t.sigmoid(t.cat([h, h_prev], dim=-1) @ self.w_bnd + self.b_bnd)  # вероятность границы (B, T, 1)
            Q2 = t.nn.functional.normalize(qkv(self.W_hq, h), dim=-1)  # запросы уровня 2
            K2 = t.nn.functional.normalize(qkv(self.W_hk, h), dim=-1)  # ключи
            V2 = qkv(self.W_hv, h)  # значения
            p_h = p_bnd.permute(0, 2, 1)  # форма: пакет, 1, время
            a2 = (1.0 - p_h * (1.0 - t.sigmoid(self.a_slow)[None, :, None])) * 0.985  # удержание: граница → забвение
            b2 = p_h * t.sigmoid(self.b_slow)[None, :, None]  # запись только на границах
            if capture is not None:  # входы уровня 2
                for W in (self.W_hq, self.W_hk, self.W_hv): capture[id(W)] = h.detach().reshape(-1, d)  # h перед уровнем 2
            y2, S2 = gated_deltanet(Q2, K2, V2, a2, b2, S_init=(state.get("S2") if state else None))  # уровень 2
            if state is not None: state["S2"] = S2.detach().clone()  # финальное состояние (копия)
            y2 = y2.permute(0, 2, 1, 3).reshape(B, T, d)  # в раскладку
            if self.use_gdn_norm:  # нормировка
                y2 = rms_norm(y2, self.g_gdn2)  # стабилизация
            if capture is not None: capture[id(self.W_ho)] = y2.detach().reshape(-1, d)  # вход W_ho
            h = h + y2 @ self.W_ho  # остаточное соединение
        hn = self._spike(rms_norm(h, self.g_mlp))  # перед MLP (со спайкингом, если включён)
        if capture is not None:  # входы W_gate и W_up
            for W in (self.W_gate, self.W_up): capture[id(W)] = hn.detach().reshape(-1, d)  # записать
        g_pre = hn @ self.W_gate  # вентиль
        mid = (g_pre * t.sigmoid(g_pre)) * (hn @ self.W_up)  # произведение внутри SwiGLU — вход W_down
        h = h + mid @ self.W_down  # основной блок
        if capture is not None: capture[id(self.W_down)] = mid.detach().reshape(-1, mid.shape[-1])  # вход W_down
        for Wg, Wu, Wd in self.blocks:  # дополнительные блоки (zero-init W_down ⇒ тождество)
            hnb = self._spike(rms_norm(h, self.g_mlp))  # нормированный остаток (со спайкингом, если включён)
            if capture is not None:  # входы блока
                for W in (Wg, Wu): capture[id(W)] = hnb.detach().reshape(-1, d)  # записать
            g_pre = hnb @ Wg; mid_b = (g_pre * t.sigmoid(g_pre)) * (hnb @ Wu)  # SwiGLU внутри блока
            h = h + mid_b @ Wd  # добавить
            if capture is not None: capture[id(Wd)] = mid_b.detach().reshape(-1, mid_b.shape[-1])  # вход Wd
        return h  # форма: пакет, время, ширина

    def forward_seq(self, x_seq: t.Tensor, state: dict | None = None, capture: dict | None = None) -> t.Tensor:  # логиты по всем позициям (B, T, 256); state — перенос состояния GDN между чанками
        h_out = rms_norm(self.forward_hidden(x_seq, capture=capture, state=state), self.g_out)  # нормированный выход (словарь захвата передаётся внутрь)
        if capture is not None: capture[-1] = h_out.detach().reshape(-1, self.dim)  # вход softmax-головы emb (правое подпространство)
        lg = (h_out @ self.emb.T) * self.out_scale + self.b_out  # связанные эмбеддинги: выходная проекция = embᵀ
        if self.use_bigram_logit_skip:  # скип биграмм в логиты
            lg = lg + self.B_u[x_seq] @ self.B_v  # аддитивно
        if self.bypass_boost > 0:  # точный обход включён
            lg = self.bypass_apply(x_seq, lg)  # прибавить ответы словарей (и при обучении, и при выводе)
        if self.active_adapter is not None:  # доменный адаптер
            A, B_adp, sc = self.active_adapter  # (d, r), (r, 256), масштаб
            lg = lg + ((h_out @ A) @ B_adp) * sc  # низкоранговая добавка
        return lg  # логиты

    def train_optimal(self, data: bytes, epochs: int = 5, batch: int = 16, seq_len: int = 64,  # обучение
                      lr_muon: float | None = None, lr_adam: float | None = None, momentum: float | None = None, wd: float | None = None,  # шаги, момент, затухание: None — из эволюции (selfdev.evolve.current), иначе заданное значение
                      warmup_frac: float | None = None, min_scale: float | None = None, beta2: float | None = None, max_steps_per_epoch: int | None = None,  # предел масштаба и beta2 — из эволюции при None; потолок шагов на эпоху: None — выводится из данных (один проход потока), иначе внешний бюджет
                      holdout: float = 0.15, seed: int = 7, opt_override: Any = None, gpm_left: dict | None = None, state_carry: bool = True,  # контроль, оптимизатор, базисы GPM, перенос состояния между чанками
                      log_every: int = 0) -> dict:  # печать
        import numpy as np  # численные массивы
        n = len(data); cut = int(n * (1.0 - holdout))  # граница обучение/контроль
        tr_arr = np.frombuffer(data[:cut], dtype=np.uint8).astype(np.int64)  # обучение
        va_arr = np.frombuffer(data[cut:], dtype=np.uint8).astype(np.int64)  # контроль
        steps_data = max(1, (len(tr_arr) - seq_len - 1) // (batch * seq_len))  # шагов на один проход данных: batch потоков идут подряд, каждый шаг сдвигает на seq_len
        steps_ep = steps_data if max_steps_per_epoch is None else min(max_steps_per_epoch, max(8, steps_data))  # без потолка — из данных; с потолком — не больше бюджета
        total_steps = max(1, epochs * steps_ep)  # всего шагов
        from hagi.selfdev.evolve import current as _evolved  # принятые эволюцией параметры
        _ev = _evolved() if any(v is None for v in (lr_muon, lr_adam, momentum, wd, warmup_frac, min_scale, beta2)) else {}  # читать только если что-то не задано явно
        _lm = float(lr_muon if lr_muon is not None else _ev["lr_muon"]); _la = float(lr_adam if lr_adam is not None else _ev["lr_adam"]); _mo = float(momentum if momentum is not None else _ev["momentum"])  # разрешить значения
        _wd = float(wd if wd is not None else _ev["wd"]); _wf = float(warmup_frac if warmup_frac is not None else _ev["warmup_frac"]); _ms = float(min_scale if min_scale is not None else _ev["min_scale"]); _b2 = float(beta2 if beta2 is not None else _ev["beta2"])  # разрешить значения
        opt = opt_override or HybridMuonAdamW(*self.matrix_and_vec_params(), lr_muon=_lm, lr_adam=_la, momentum=_mo, wd=_wd, beta2=_b2)  # оптимизатор
        gpm = gpm_left if gpm_left is not None else getattr(self, "gpm_left", None)  # базисы: аргумент или атрибут модели (ключи — имена)
        gpm_r = getattr(self, "gpm_right", None) or {}  # правые базисы модели (softmax-голова)
        opt.gpm_right.update(gpm_r)  # передать
        opt.names = self.param_names()  # имена весов по порядку mats + vecs: без них ключи GPM (имена) не находятся и защита молча не действует
        if gpm:  # заданы
            opt.gpm_left.update(gpm)  # градиент, шаг и затухание не заходят в защищённое подпространство
        rng = np.random.default_rng(seed)  # выборка окон
        # Контрольные окна расставлены РАВНОМЕРНО по всему хвосту (до 256 окон, а не первые 32):
        # прежние 32 первых окна покрывали лишь 2 КБ из десятков КБ хвоста нового домена, и оценка
        # контроля была шумной (замер в цикле: контроль 0.82 при CE всего хвоста 0.41).
        _n_val = min(VAL_WINDOWS, max(1, len(va_arr) // seq_len))  # число контрольных окон (VAL_WINDOWS из среды HAGI_VAL_WINDOWS)
        v_starts = (np.arange(0, max(1, len(va_arr) - seq_len - 1), seq_len)[:_n_val] if len(va_arr) // seq_len <= VAL_WINDOWS else (np.linspace(0, len(va_arr) - seq_len - 1, _n_val)).astype(np.int64))  # старты: подряд, если окон не больше лимита, иначе равномерно по хвосту
        vx = t.as_tensor(np.stack([va_arr[s: s + seq_len] for s in v_starts]), dtype=t.long, device=self.device)  # входы
        vy = t.as_tensor(np.stack([va_arr[s + 1: s + 1 + seq_len] for s in v_starts]), dtype=t.long, device=self.device)  # цели

        def val_ce() -> float:  # контрольная кросс-энтропия на контроле
            with t.no_grad():  # без графа
                return float(t.nn.functional.cross_entropy(self.forward_seq(vx).reshape(-1, self.VOCAB), vy.reshape(-1)))  # по всем позициям
        h0 = val_ce(); best_val = h0; best_snap = [p.detach().clone() for p in self.params()]  # до обучения
        max_start = max(1, len(tr_arr) - seq_len - 1)  # число стартов
        warmup = max(1, int(round(_wf * total_steps)))  # число шагов разогрева масштаба шага (доля — из эволюции)
        stride = max(1, max_start // batch)  # шаг потока: batch потоков равномерно по домену
        stream_pos = np.array([(i * stride) % max_start for i in range(batch)], dtype=np.int64)  # позиции потоков
        carry: dict = {}  # переносимое состояние GDN (state-carry BPTT: длинный контекст без удлинения T)
        self.bypass_fit(data[:cut])  # словари точного обхода — только по ОБУЧАЮЩЕЙ части (контроль остаётся невидимым)
        self.spike_off, self.spike_zeros, self.spike_total = True, 0, 0  # спайкинг выключен до последних 20% шагов; счётчики нулей
        for step in range(total_steps):  # по шагам
            self.spike_off = step < SPIKE_START * total_steps  # последние 20% шагов — со спайкингом (доводка STE)
            opt.set_lr_scale((0.2 + 0.8 * (step + 1) / warmup) if step < warmup else (_ms + (1.0 - _ms) * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))))  # разогрев, затем косинусный отжиг
            if state_carry:  # поток: подряд идущие чанки, состояние переносится
                starts = stream_pos  # старты потоков
                state = {k: v for k, v in carry.items()}  # состояние с предыдущего шага
                bx = t.as_tensor(np.stack([tr_arr[s: s + seq_len] for s in starts]), dtype=t.long, device=self.device)  # входы
                by = t.as_tensor(np.stack([tr_arr[s + 1: s + 1 + seq_len] for s in starts]), dtype=t.long, device=self.device)  # цели
                opt.zero_grad()  # сброс
                loss = t.nn.functional.cross_entropy(self.forward_seq(bx, state=state).reshape(-1, self.VOCAB), by.reshape(-1))  # потеря по всем T позициям
                carry = {"S1": state["S1"], "S2": state.get("S2")}  # новое состояние (detached)
                stream_pos = stream_pos + seq_len  # сдвинуть потоки
                wrap = stream_pos >= max_start  # поток дошёл до конца
                if np.any(wrap):  # есть обернувшиеся
                    stream_pos[wrap] = rng.integers(0, max(1, len(tr_arr) // 3), size=int(np.sum(wrap)))  # новые случайные старты
                    for bi in np.where(wrap)[0]:  # сброс состояния обернувшихся строк
                        carry["S1"][bi] = 0  # нулевое состояние
                        if carry["S2"] is not None: carry["S2"][bi] = 0  # нулевое
                loss.backward(); opt.step()  # градиент и шаг
                if (step + 1) % max(1, steps_ep // 2) == 0 or step == total_steps - 1:  # контроль
                    hv = val_ce()  # контрольная кросс-энтропия
                    if hv < best_val: best_val = hv; best_snap = [p.detach().clone() for p in self.params()]  # снимок
                    if log_every and (step + 1) % log_every == 0: print(f"  шаг {step + 1}/{total_steps}: потеря {float(loss):.4f}, контроль {hv:.4f}", flush=True)  # прогресс
                continue  # следующий шаг
            starts = rng.integers(0, max_start, size=batch)  # случайные старты
            bx = t.as_tensor(np.stack([tr_arr[s: s + seq_len] for s in starts]), dtype=t.long, device=self.device)  # входы
            by = t.as_tensor(np.stack([tr_arr[s + 1: s + 1 + seq_len] for s in starts]), dtype=t.long, device=self.device)  # цели
            opt.zero_grad()  # сброс
            loss = t.nn.functional.cross_entropy(self.forward_seq(bx).reshape(-1, self.VOCAB), by.reshape(-1))  # потеря по всем T позициям
            loss.backward()  # градиент
            opt.step()  # шаг
            if (step + 1) % max(1, steps_ep // 2) == 0 or step == total_steps - 1:  # контроль
                hv = val_ce()  # контрольная кросс-энтропия
                if hv < best_val:  # лучше
                    best_val = hv; best_snap = [p.detach().clone() for p in self.params()]  # снимок
                if log_every and (step + 1) % log_every == 0:  # печать
                    print(f"  шаг {step + 1}/{total_steps}: потеря {float(loss):.4f}, контроль {hv:.4f}", flush=True)  # прогресс
        with t.no_grad():  # восстановить лучший
            for p, s in zip(self.params(), best_snap):  # по параметрам
                p.copy_(s)  # вернуть
        picked = "snapshot"  # что выбрано по контролю: пока — лучший снимок
        ema = getattr(opt, "ema", None)  # средние веса оптимизатора
        if ema is not None:  # EMA включён — сравнить с лучшим снимком на контроле
            live = [p.detach().clone() for p in self.params()]  # текущие (лучший снимок)
            with t.no_grad():  # без графа
                for p, e in zip(self.params(), ema): p.copy_(e.to(p.dtype))  # поставить средние
            ev = val_ce()  # контроль средних весов
            if ev < best_val:  # средние лучше снимка
                best_val, picked = ev, "ema"  # взять их
            else:  # снимок лучше
                with t.no_grad():  # без графа
                    for p, q in zip(self.params(), live): p.copy_(q)  # вернуть снимок
        self.spike_off = False  # после обучения спайкинг действует всегда (если задан)
        return {"val_ce_before": h0, "val_ce_after": best_val, "steps": total_steps,  # итог: CE до/после и число шагов
                "weights": picked,  # какие веса выбраны контролем: snapshot (лучший снимок) или ema (среднее)
                "spike_k": getattr(self, "spike_k", None),  # порог спайкинга (None — выключен)
                "spike_zero_frac": (self.spike_zeros / self.spike_total) if self.spike_total else 0.0}  # доля нулей на прокинутом входе за доводку


def eval_oos_exact(model: HagiOptimalLM, raw: bytes, stride: int = 4) -> float:  # контрольная кросс-энтропия вне выборки
    """CE по позициям вне выборки: окно k, шаг stride, одна цель на окно (как у HNetByteLM — для честного сравнения)."""
    import numpy as np  # массивы
    k = model.k  # контекст
    pos = np.arange(k, len(raw), stride)  # позиции целей
    idx = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)  # байты
    tot = cnt = 0.0  # сумма и счётчик
    with t.no_grad():  # без графа
        for i in range(0, len(pos), 128):  # пакетами
            p = pos[i:i + 128]  # пакет позиций
            X = t.as_tensor(idx[p[:, None] - k + np.arange(k)[None, :]], dtype=t.long, device=model.device)  # окна
            y = t.as_tensor(idx[p], dtype=t.long, device=model.device)  # цели
            tot += float(t.nn.functional.cross_entropy(model.logits(X), y, reduction="sum")); cnt += len(p)  # накопить
    return tot / max(1.0, cnt)  # средняя CE


def grow_width(model: "HagiOptimalLM", delta: int = 8) -> int:
    """Точный рост ширины на delta: логиты не меняются (масштабы нормировки умножаются на sqrt(d0/d1),
    старые координаты голов на своих местах, новые — нулевые). Возвращает новую ширину."""
    assert delta > 0 and (model.dim + delta) % model.n_heads == 0  # ширина головы остаётся целой
    d0, d1, H = model.dim, model.dim + delta, model.n_heads  # старая, новая ширина, головы
    hd0, hd1 = d0 // H, d1 // H  # ширины голов
    sc = math.sqrt(d0 / d1)  # множитель масштабов нормировки
    with t.no_grad():  # без графа
        def e2(p: t.Tensor, r: int, c: int) -> t.Tensor:  # расширить матрицу нулями
            out = t.zeros((r, c), dtype=p.dtype); out[: p.shape[0], : p.shape[1]] = p.detach(); return out  # блок в углу
        def e1(p: t.Tensor, n: int, s: float = 1.0) -> t.Tensor:  # расширить вектор
            out = t.zeros((n,), dtype=p.dtype); out[: p.shape[0]] = p.detach() * s; return out  # старое с масштабом
        def eqkv(W: t.Tensor) -> t.Tensor:  # входная проекция в головы: по голове и по ширине головы
            out = t.zeros((d1, d1), dtype=W.dtype); out.reshape(d1, H, hd1)[:d0, :, :hd0] = W.detach().reshape(d0, H, hd0); return out  # блоки на местах
        def ewo(W: t.Tensor) -> t.Tensor:  # выходная проекция из голов
            out = t.zeros((d1, d1), dtype=W.dtype); out.reshape(H, hd1, d1)[:, :hd0, :d0] = W.detach().reshape(H, hd0, d0); return out  # блоки на местах
        for name in ("W_gq", "W_gk", "W_gv", "W_hq", "W_hk", "W_hv"):  # входные проекции голов
            if hasattr(model, name): setattr(model, name, t.nn.Parameter(eqkv(getattr(model, name))))  # заменить
        for name in ("W_go", "W_ho"):  # выходные проекции голов
            if hasattr(model, name): setattr(model, name, t.nn.Parameter(ewo(getattr(model, name))))  # заменить
        model.emb = t.nn.Parameter(e2(model.emb, model.VOCAB, d1)); model.w_lag = t.nn.Parameter(e2(model.w_lag, len(model.LAGS), d1))  # таблицы
        model.W_loc = t.nn.Parameter(e2(model.W_loc, d1, d1)); model.W_bi = t.nn.Parameter(e2(model.W_bi, model.d_bi, d1))  # матрицы
        model.w_sc = t.nn.Parameter(e2(model.w_sc, 3, d1))  # короткая свёртка
        for name in ("W_alpha", "W_beta"):  # вентили (d, H)
            setattr(model, name, t.nn.Parameter(e2(getattr(model, name), d1, H)))  # заменить
        def eg(p: t.Tensor) -> t.Tensor:  # масштаб по головам (раскладка H·hd): старые координаты каждой головы на своих местах
            out = t.zeros((d1,), dtype=p.dtype); out.reshape(H, hd1)[:, :hd0] = p.detach().reshape(H, hd0); return out * sc  # блоки + sqrt(d0/d1)
        for name in ("g_loc", "g_mlp", "g_out"):  # масштабы обычных нормировок
            if hasattr(model, name): setattr(model, name, t.nn.Parameter(e1(getattr(model, name), d1, sc)))  # с sqrt(d0/d1)
        for name in ("g_gdn1", "g_gdn2"):  # масштабы выходов GDN — по головам
            if hasattr(model, name): setattr(model, name, t.nn.Parameter(eg(getattr(model, name))))  # расширить по головам
        if model.use_hier_gdn:  # гейт границ принимает [h, h_prev]
            wb = model.w_bnd.detach()  # прежние веса
            nb = t.zeros((2 * d1, 1), dtype=wb.dtype); nb[:d0, 0] = wb[:d0, 0]; nb[d1: d1 + d0, 0] = wb[d0:, 0]  # на свои места
            model.w_bnd = t.nn.Parameter(nb)  # заменить
        model.W_gate = t.nn.Parameter(e2(model.W_gate, d1, model.hidden)); model.W_up = t.nn.Parameter(e2(model.W_up, d1, model.hidden)); model.W_down = t.nn.Parameter(e2(model.W_down, model.hidden, d1))  # SwiGLU-блок
        model.blocks = [(t.nn.Parameter(e2(Wg, d1, Wg.shape[1])), t.nn.Parameter(e2(Wu, d1, Wu.shape[1])), t.nn.Parameter(e2(Wd, Wd.shape[0], d1))) for Wg, Wu, Wd in model.blocks]  # блоки — нулями
        if model.active_adapter is not None:  # адаптер — нулями вниз
            A, B_adp, s_ = model.active_adapter; model.active_adapter = (e2(A, d1, A.shape[1]), B_adp, s_)  # расширить
        model.dim = d1  # новая ширина
        model.adapter_dim = d1  # адаптеры применяются к новой ширине
        model.rms_eps = getattr(model, "rms_eps", 1e-6) * (d0 / d1)  # eps нормировки масштабируется как d0/d1 (точное равенство при малых активациях)
    model.sync_device()  # расширенные параметры — на устройство модели
    return d1  # вернуть


def grow_depth(model: "HagiOptimalLM", seed: int = 0) -> int:
    """Точный рост глубины: SwiGLU-блок с W_down = 0 — функция не меняется (FirstLayerGrad: первый шаг ненулевой)."""
    g = t.Generator().manual_seed(model.seed + 500 + len(model.blocks) + seed)  # зерно
    model.blocks.append((t.nn.Parameter(t.randn((model.dim, model.hidden), generator=g) * 0.06),  # вентиль
                         t.nn.Parameter(t.randn((model.dim, model.hidden), generator=g) * 0.06),  # подъём
                         t.nn.Parameter(t.zeros((model.hidden, model.dim)))))  # нулевой выход — тождество
    model.sync_device()  # новые параметры блока — на устройство модели
    return len(model.blocks)  # число блоков


def update_gpm_bases(model: "HagiOptimalLM", data: bytes, gpm_left: dict[str, t.Tensor],  # базисы защищённого подпространства (ИМЯ веса -> базис; по id после перезапуска не совпадали)
                     right_names: tuple[str, ...] | None = None,  # имена весов с правым базисом (softmax-голова: h_out @ emb.T); по умолчанию — из HAGI_GPM_RIGHT (замер: ранг 15 из 24 на emb душит новый домен — средний CE ствола 2.76 -> 3.34; включать осознанно)
                     energy_thresh: float = 0.90, max_rank_frac: float = 0.65) -> None:  # порог энергии и доля ранга
    """Обновить базисы GPM по активациям домена: для каждой матрицы — входы, SVD, 90% энергии,
    не более max_rank_frac·d_in столбцов; прежний базис сохраняется (объединение через QR)."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # байты
    if len(arr) < 66:  # слишком коротко
        return  # ничего
    starts = np.linspace(0, len(arr) - 65, num=min(16, max(1, len(arr) // 64)), dtype=np.int64)  # старты окон
    bx = t.as_tensor(np.stack([arr[s: s + 64] for s in starts]), dtype=t.long, device=model.device)  # форма: окна, 64
    acts: dict[int, t.Tensor] = {}  # входы матриц
    acts_out: dict[int, t.Tensor] = {}  # выходные активации (для правых базисов)
    with t.no_grad():  # без графа
        model.forward_seq(bx, capture=acts)  # собрать (forward_seq передаёт словарь в forward_hidden и добавляет h_out под ключом −1)
        acts_out[id(model.emb)] = acts.pop(-1)  # h_out — вход читающей стороны emb
    names = {id(p): n for n, p in model.__dict__.items() if isinstance(p, t.nn.Parameter)}  # id -> имя
    import os as _os  # переменные окружения
    if right_names is None:  # не задано явно
        right_names = ("emb",) if _os.environ.get("HAGI_GPM_RIGHT", "0") == "1" else ()  # по умолчанию выключено
    gpm_right = getattr(model, "gpm_right", None)  # правые базисы модели (имя -> базис)
    if gpm_right is None:  # ещё нет
        gpm_right = model.gpm_right = {}  # создать
    names.update({id(p): f"blk{i}.{nm}" for i, (Wg, Wu, Wd) in enumerate(getattr(model, "blocks", [])) for p, nm in ((Wg, "Wg"), (Wu, "Wu"), (Wd, "Wd"))})  # блоки роста
    for k, A in acts.items():  # по матрицам
        k = names.get(k, k)  # ключ — имя параметра
        X = A.double().cpu().numpy(); d_in = X.shape[1]  # активации (N·T, d_in)
        U_old = gpm_left.get(k)  # прежний базис
        Uo = U_old.double().cpu().numpy() if U_old is not None and U_old.shape[1] > 0 else None  # в numpy
        if Uo is not None:  # есть
            if Uo.shape[0] < d_in:  # после роста ширины — дополнить нулями
                Up = np.zeros((d_in, Uo.shape[1])); Up[: Uo.shape[0]] = Uo; Uo = Up  # дополнить
            X = X - (X @ Uo) @ Uo.T  # убрать уже защищённое
        _, S, Vt = np.linalg.svd(X, full_matrices=False)  # сингулярное разложение
        e = S * S; cum = np.cumsum(e) / (float(np.sum(e)) + 1e-12)  # накопленная энергия
        k_new = int(np.searchsorted(cum, energy_thresh) + 1)  # сколько новых направлений
        cur_r = Uo.shape[1] if Uo is not None else 0  # текущий ранг
        max_k = max(1, int(d_in * max_rank_frac))  # предел
        k_new = max(0, min(k_new, max_k - cur_r))  # не больше предела
        if k_new <= 0:  # нечего добавлять
            continue  # дальше
        U_add = Vt[:k_new].T  # новые направления
        U_new = U_add if Uo is None else np.linalg.qr(np.concatenate([Uo, U_add], axis=1))[0][:, :max_k]  # объединить
        gpm_left[k] = t.as_tensor(U_new, dtype=t.float32)  # сохранить
    for k, A in acts_out.items():  # правые подпространства (выходные активации)
        nm = names.get(k)  # имя веса
        if nm not in right_names:  # не защищается справа
            continue  # дальше
        X = A.double().cpu().numpy(); d_r = X.shape[1]  # активации (N·T, d_r)
        Uo = gpm_right[nm].double().cpu().numpy() if nm in gpm_right and gpm_right[nm].shape[1] > 0 else None  # прежний базис
        if Uo is not None:  # есть прежний
            X = X - (X @ Uo) @ Uo.T  # убрать уже защищённое
        _, S, Vt = np.linalg.svd(X, full_matrices=False)  # сингулярное разложение
        e = S * S; cum = np.cumsum(e) / (float(np.sum(e)) + 1e-12)  # накопленная энергия
        k_new = int(np.searchsorted(cum, energy_thresh) + 1)  # сколько новых направлений
        cur_r = Uo.shape[1] if Uo is not None else 0  # текущий ранг
        max_k = max(1, int(d_r * max_rank_frac))  # предел
        k_new = max(0, min(k_new, max_k - cur_r))  # не больше предела
        if k_new <= 0:  # нечего добавлять
            continue  # дальше
        U_add = Vt[:k_new].T  # новые направления
        U_new = U_add if Uo is None else np.linalg.qr(np.concatenate([Uo, U_add], axis=1))[0][:, :max_k]  # объединить
        gpm_right[nm] = t.as_tensor(U_new, dtype=t.float32)  # сохранить (по имени: emb — и вход, и голова)


class CortexFiber:
    """Доменный волоконный адаптер: SiLU-горлышко ранга r в скрытом слое + выходная головка ранга r
    + биграммное смещение ранга 4. Ствол не трогается; ~4,9 тыс. параметров на домен."""

    def __init__(self, model: "HagiOptimalLM", rank: int = 8, seed: int = 0) -> None:  # создать
        d = model.dim  # ширина ствола
        g = t.Generator().manual_seed(seed)  # зерно
        self.A_h = t.nn.Parameter(t.randn((d, rank), generator=g) * 0.05)  # вход горлышка
        self.B_h = t.nn.Parameter(t.zeros((rank, d)))  # выход горлышка (0 ⇒ тождество в начале)
        self.U_o = t.nn.Parameter(t.randn((d, rank), generator=g) * 0.05)  # вход головки
        self.V_o = t.nn.Parameter(t.zeros((rank, 256)))  # выход головки (0 ⇒ прибавки нет)
        self.b_f = t.nn.Parameter(t.zeros(256))  # смещение
        self.Bu_f = t.nn.Parameter(t.randn((256, 4), generator=g) * 0.04)  # биграммный вход
        self.Bv_f = t.nn.Parameter(t.zeros((4, 256)))  # биграммный выход
        with t.no_grad():  # без графа
            for q in self.params():  # по всем параметрам волокна
                q.data = q.data.to(model.device)  # параметр — на устройство модели

    def params(self) -> list[t.nn.Parameter]:  # все параметры
        return [self.A_h, self.B_h, self.U_o, self.V_o, self.b_f, self.Bu_f, self.Bv_f]  # списком

    def num_params(self) -> int:  # число параметров
        return sum(p.numel() for p in self.params())  # сумма

    def logits_delta(self, h_norm: t.Tensor, x_seq: t.Tensor) -> t.Tensor:  # добавка к логитам; h_norm (B, T, d)
        h_adj = h_norm + t.nn.functional.silu(h_norm @ self.A_h) @ self.B_h  # нелинейное горлышко
        return (h_adj @ self.U_o) @ self.V_o + self.Bu_f[x_seq] @ self.Bv_f + self.b_f  # головка + биграммы + смещение


def fit_domain_fiber(model: "HagiOptimalLM", tr: bytes, rank: int = 8, steps: int = 120, batch: int = 16,  # обучение волокна
                     seed: int = 0, lr: float = 1.5e-2) -> CortexFiber:  # на замороженных активациях ствола
    """Обучить волокно домена на замороженных скрытых состояниях ствола (один прогон ствола)."""
    arr = np.frombuffer(tr, dtype=np.uint8).astype(np.int64)  # байты
    max_s = max(1, len(arr) - 65)  # число стартов
    starts = np.arange(0, max_s, max(1, max_s // 64))[:64]  # до 64 перекрывающихся окон
    bx = t.as_tensor(np.stack([arr[s: s + 64] for s in starts]), dtype=t.long, device=model.device)  # входы
    by = t.as_tensor(np.stack([arr[s + 1: s + 65] for s in starts]), dtype=t.long, device=model.device)  # цели
    with t.no_grad():  # ствол заморожен
        h_all = rms_norm(model.forward_hidden(bx), model.g_out)  # форма: окна, 64, ширина
        lg_base = (h_all @ model.emb.T) * model.out_scale + model.b_out  # логиты ствола
        if model.use_bigram_logit_skip:  # скип ствола
            lg_base = lg_base + model.B_u[bx] @ model.B_v  # добавить
    fib = CortexFiber(model, rank=rank, seed=seed)  # новое волокно
    opt = t.optim.AdamW(fib.params(), lr=lr, weight_decay=1e-4)  # Адам
    sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-3)  # отжиг
    rng = np.random.default_rng(seed + 11)  # выборка
    N_c = len(starts)  # число окон
    for _ in range(steps):  # шаги
        idx = rng.integers(0, N_c, size=min(batch, N_c))  # индексы
        lg_f = lg_base[idx] + fib.logits_delta(h_all[idx].detach(), bx[idx])  # логиты с волокном
        loss = t.nn.functional.cross_entropy(lg_f.reshape(-1, 256), by[idx].reshape(-1))  # по всем позициям
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()  # шаг
    return fib  # обученное волокно


def eval_with_fiber(model: "HagiOptimalLM", raw: bytes, fib: "CortexFiber | None" = None) -> float:
    """CE по всей последовательности (T целей за прогон); с волокном — добавка к логитам ствола."""
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)  # байты
    if len(arr) < model.k + 2:  # коротко
        return float("nan")  # не определено
    x = t.as_tensor(arr[:-1], dtype=t.long, device=model.device)[None]; y = t.as_tensor(arr[1:], dtype=t.long, device=model.device)  # вход и цели
    with t.no_grad():  # без графа
        h = rms_norm(model.forward_hidden(x), model.g_out)  # скрытое
        lg = (h @ model.emb.T) * model.out_scale + model.b_out  # логиты ствола
        if model.use_bigram_logit_skip: lg = lg + model.B_u[x] @ model.B_v  # скип
        if fib is not None: lg = lg + fib.logits_delta(h, x)  # волокно
        return float(t.nn.functional.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)))  # контрольная кросс-энтропия


class MirrorHedgeStreamPredictor:
    """Потоковый предиктор: модель на окне k + онлайн PPM порядков 0..4 + эксперт LZP (длинное
    повторное совпадение контекста); смешивание Mirror-Hedge MWU с полом 0.01 (эксперт не умирает).
    Обновления зависят только от уже данных байтов ⇒ без потерь (декодер повторяет кодер).

    LZP-эксперт: ищет самое длинное повторение предыдущих 6/4/3 байтов в якорной таблице; при
    совпадении ≥ 3 байт ставит массу min(0.94, 0.30 + 0.08·len) на следующий за ним байт. Ловит
    дальние копии (имена, повторы строк), которые PPM ≤ 4 не различает."""

    def __init__(self, model: "HagiOptimalLM | None" = None, eta: float = 0.25) -> None:  # создать
        self.model, self.eta = model, float(eta)  # модель и шаг
        # веса: [модель, PPM, LZP] (без модели — [PPM, LZP])
        self.weights = np.array([0.60, 0.30, 0.10], dtype=np.float64) if model is not None else np.array([0.8, 0.2])  # веса
        self.uni = np.ones(256, dtype=np.float64)  # счётчик порядка 0
        self.ctx: list[dict[int, dict[int, int]]] = [{}, {}, {}, {}]  # таблицы порядков 1..4
        self.hist: list[int] = []  # история ≤ 16 байтов
        self.long_hist: bytearray = bytearray()  # полная история (для LZP: байты и позиции совпадений)
        self.lz_anchors: dict[bytes, list[int]] = {}  # контекст 3..6 байтов -> позиции его концов (≤ 12 последних)
        self.st = StreamState(model) if model is not None else None  # побайтовое состояние модели (O(1) на байт)
        self._lg = np.zeros(256)  # последние логиты состояния
    def _lzp_probs(self) -> tuple[np.ndarray, int]:  # распределение LZP и длина совпадения
        n = len(self.long_hist)  # длина истории
        best_len, best_byte = 0, -1  # лучшее совпадение
        for kl in (6, 4, 3):  # длины ключа от длинного к короткому
            if n >= kl:  # ключ помещается
                key = bytes(self.long_hist[-kl:])  # ключ
                for pos in reversed(self.lz_anchors.get(key, ())[-8:]):  # последние ≤ 8 позиций конца ключа
                    if pos < n:  # позиция короче истории
                        ml = kl  # длина совпадения (считая от конца)
                        while ml < 32 and ml < pos and self.long_hist[pos - ml - 1] == self.long_hist[n - ml - 1]:  # расширяем влево
                            ml += 1  # ещё один байт совпал
                        if ml > best_len:  # лучше
                            best_len, best_byte = ml, self.long_hist[pos]  # запомнить
                if best_len >= 10:  # достаточно длинное совпадение
                    break  # короткие ключи не спрашиваются
        if best_byte < 0 or best_len < 3:  # совпадения нет
            return None, 0  # эксперт молчит
        return best_byte, best_len  # что предсказано и насколько уверенно

    def _ctx_probs(self) -> np.ndarray:  # PPM 0..4 с интерполяцией
        p = self.uni / self.uni.sum()  # порядок 0
        n = len(self.hist)  # длина истории
        for order in range(1, 5):  # порядки
            if n < order:  # не хватает истории
                break  # конец
            h = 0  # хэш контекста
            for b in self.hist[-order:]: h = (h * 257 + b) & 0xFFFFFFFFFFFFFFFF  # свернуть
            tbl = self.ctx[order - 1].get(h)  # таблица продолжений
            if tbl:  # есть
                tot = sum(tbl.values()); alpha = tot / (tot + len(tbl) + 0.8)  # вес порядка (Witten–Bell)
                p_ord = p * (1.0 - alpha); inv = alpha / tot  # сглаживание
                for s_, c in tbl.items(): p_ord[s_] += c * inv  # добавить счётчики
                p = p_ord  # следующий порядок опирается на этот
        return p  # распределение

    def predict_probs(self) -> np.ndarray:  # распределение следующего байта
        p_ctx = self._ctx_probs()  # контекстная модель
        b_lz, ml = self._lzp_probs()  # кандидат LZP и длина совпадения
        if b_lz is not None:  # LZP нашёл повтор
            cf = min(0.94, 0.30 + 0.08 * ml); p_lz = p_ctx * (1.0 - cf); p_lz[b_lz] += cf  # распределение LZP
        else:  # нет повторения
            p_lz = p_ctx  # эксперт нейтрален
        if self.model is None:  # без модели
            self._last = (p_ctx, p_lz); return self.weights[0] * p_ctx + self.weights[1] * p_lz  # смесь контекстной и LZP-оценок
        if self.st.n == 0:  # истории нет
            p_neu = np.full(256, 1.0 / 256.0)  # равномерно
        else:  # логиты уже посчитаны состоянием на предыдущем байте
            lg = self._lg  # логиты следующего байта
            e = np.exp(lg.double().numpy() - lg.max().item()); p_neu = e / e.sum()  # вероятности
        self._last = (p_neu, p_ctx, p_lz)  # запомнить
        return self.weights[0] * p_neu + self.weights[1] * p_ctx + self.weights[2] * p_lz  # смесь

    def update(self, s: int) -> None:  # после фактического байта s
        if self._last is not None:  # обновление весов экспертов (Mirror-Hedge по потерям)
            exps = self._last  # предсказания экспертов
            w = np.array([w_ * math.exp(-self.eta * min(10.0, -math.log(max(1e-12, float(e[s]))))) for w_, e in zip(self.weights, exps)])  # взвешенные потери
            z = float(w.sum()); self.weights = 0.98 * (w / z) + 0.01  # нормировка с полом
        self.uni[s] += 1.0  # порядок 0
        n = len(self.hist)  # история
        for order in range(1, 5):  # порядки
            if n < order: break  # конец
            h = 0  # накапливаемый хэш контекста
            for b in self.hist[-order:]: h = (h * 257 + b) & 0xFFFFFFFFFFFFFFFF  # хэш
            row = self.ctx[order - 1].setdefault(h, {}); row[s] = row.get(s, 0) + 1  # счётчик
        self.hist.append(s); self.hist = self.hist[-16:]  # обрезать
        n_l = len(self.long_hist)  # позиция нового байта в длинной истории
        for kl in (6, 4, 3):  # якоря LZP: ключ = последние kl байтов ДО нового байта
            if n_l >= kl:  # ключ помещается
                lst = self.lz_anchors.setdefault(bytes(self.long_hist[-kl:]), []); lst.append(n_l); del lst[:-12]  # ≤ 12 позиций
        self.long_hist.append(s)  # добавить байт
        if self.st is not None:  # модель есть
            self._lg = self.st.step(int(s))  # один шаг состояния O(1): логиты следующего байта


def hadamard_mat(n: int) -> np.ndarray:
    """Нормированная матрица Адамара порядка n (степень двойки), построение Sylvester."""
    H = np.array([[1.0]], dtype=np.float64)  # порядок 1
    while H.shape[0] < n:  # удваивать
        H = np.block([[H, H], [H, -H]])  # построение Адамара удвоением
    return H / math.sqrt(n)  # нормировать


def quant_per_row(W: np.ndarray, bits: int = 2, use_had: bool = True, rotate_rows: bool = True, stages: int = 1) -> tuple[dict, np.ndarray]:
    """Построчное квантование W: (опционально) поворот Адамара по блокам, порог/шкала на строку.
        bits=2 — триты {−1, 0, +1}; bits=4 — 16 уровней. Таблицы поиска (emb) — без rotate_rows.
        stages > 1 — остаточное квантование: каждая следующая стадия кодирует остаток
        (W − сумма предыдущих стадий) тем же способом; уровни стадий конкатенируются по строкам
        (Q формы (stages·r, c), шкалы (stages·r,)). Ошибка убывает, объём растёт пропорционально.
        Возвращает (параметры, уровни int8)."""
    R = W.astype(np.float64, copy=True)  # копия
    r, c = R.shape  # размеры
    if use_had:  # поворот Адамара
        br = max((b for b in (64, 32, 16, 8, 4, 2) if r % b == 0), default=1) if rotate_rows else 1  # блок строк
        if br > 1: R = np.einsum("ij,ajb->aib", hadamard_mat(br), R.reshape(r // br, br, c)).reshape(r, c)  # по строкам
        bc = max((b for b in (64, 32, 16, 8, 4, 2) if c % b == 0), default=1)  # блок столбцов
        if bc > 1: R = (R.reshape(r, c // bc, bc) @ hadamard_mat(bc)).reshape(r, c)  # по столбцам
    n_st = max(1, int(stages))  # число стадий
    scs = np.empty(n_st * r, dtype=np.float32); Q = np.zeros((n_st * r, c), dtype=np.int8)  # шкалы и уровни всех стадий
    recon = np.zeros((r, c), dtype=np.float64)  # накопленная reconstruction Σ стадий
    for st_ in range(n_st):  # по стадиям
        for i in range(r):  # по строкам
            row = R[i] - recon[i] if st_ > 0 else R[i]  # строка (на стадиях > 1 — остаток)
            if bits == 2:  # триты
                from hagi.models.quantized import _best_mse_scale  # точный масштаб по минимальной MSE
                sc, levels, _err = _best_mse_scale(row)  # масштаб, уровни {−1,0,1}, ошибка
                Q[st_ * r + i] = levels.astype(np.int8)  # уровни стадии
                recon[i] += levels.astype(np.float64) * sc  # добавить вклад стадии
            else:  # 4 бита
                m = float(np.max(np.abs(row))) or 1e-6; sc = m / 7.5  # шкала
                Q[st_ * r + i] = np.clip(np.round(row / sc), -7, 8).astype(np.int8)  # уровни стадии
                recon[i] += Q[st_ * r + i].astype(np.float64) * sc  # вклад стадии
            scs[st_ * r + i] = sc  # сохранить шкалу стадии
    return {"bits": int(bits), "use_had": bool(use_had), "rotate_rows": bool(rotate_rows), "shape": (r, c), "scales": scs, "stages": n_st}, Q  # вернуть


def dequant_per_row(par: dict, Q: np.ndarray) -> np.ndarray:
    """Обратно к quant_per_row: уровни · шкала и обратный поворот Адамара (H симметрична, H·H = I)."""
    r, c = par["shape"]; n_st = int(par.get("stages", 1))  # размеры и число стадий
    R = (Q.astype(np.float64) * par["scales"][:, None].astype(np.float64)).reshape(n_st, r, c).sum(axis=0)  # сумма стадий: Σ_k Q_k·s_k
    if par["use_had"]:  # повернуть обратно
        bc = max((b for b in (64, 32, 16, 8, 4, 2) if c % b == 0), default=1)  # блок столбцов
        if bc > 1: R = (R.reshape(r, c // bc, bc) @ hadamard_mat(bc)).reshape(r, c)  # обратно
        if par["rotate_rows"]:  # по строкам
            br = max((b for b in (64, 32, 16, 8, 4, 2) if r % b == 0), default=1)  # блок
            if br > 1: R = np.einsum("ij,ajb->aib", hadamard_mat(br), R.reshape(r // br, br, c)).reshape(r, c)  # обратно
    return R  # восстановленная матрица


def gauge_balance(model: "HagiOptimalLM") -> dict[str, float]:
    """Калибровка некомпактных симметрий масштабирования: функция модели НЕ меняется (сдвиг логитов
    ~1e-6), а нормы сомножителей выравниваются.

    Симметрии:
      - SwiGLU (основная и блоки роста): столбец j матрицы up × c_j, строка j матрицы down / c_j;
        баланс c_j = sqrt(||down_j|| / ||up_j||) — норма строки и столбца становится средней
        геометрической (arXiv:2611.27889, Sec. 5);
      - скип биграмм B_u @ B_v: перенос масштаба между сомножителями, чтобы нормы столбцов B_u
        и строк B_v совпали (диагональная балансировка ранга 8).
    Зачем: при затухании весов, стремящемся к 0, параметры уходят по этим слоям на бесконечность,
    раздувая динамический диапазон строк — а он напрямую портит построчное троичное/4-битное
    квантование. Калибровка возвращает параметры в сбалансированные координаты перед экспортом.

    model — модель. Возвращает словарь: максимальный |log(c_j)| по SwiGLU-парам и по биграммам."""
    with t.no_grad():  # без графа
        max_sw = 0.0  # наибольший сдвиг масштаба в SwiGLU
        for W_up, W_down in [(model.W_up, model.W_down)] + [(b[1], b[2]) for b in model.blocks]:  # все SwiGLU
            cu = t.linalg.norm(W_up.detach(), dim=0).clamp_min(1e-12); cd = t.linalg.norm(W_down.detach(), dim=1).clamp_min(1e-12)  # нормы столбцов up и строк down
            c = t.sqrt(cd / cu)  # баланс: c_j = sqrt(||down_j|| / ||up_j||)
            max_sw = max(max_sw, float(c.log().abs().max()))  # учесть сдвиг
            W_up.mul_(c[None, :]); W_down.mul_(1.0 / c[:, None])  # применить: up · c, down / c (функция та же)
        Bu, Bv = getattr(model, "B_u", None), getattr(model, "B_v", None)  # скип биграмм
        max_bg = 0.0  # наибольший сдвиг в биграммах
        if Bu is not None and Bv is not None:  # скип есть
            nu = t.linalg.norm(Bu.detach(), dim=0).clamp_min(1e-12); nv = t.linalg.norm(Bv.detach(), dim=1).clamp_min(1e-12)  # нормы
            c = t.sqrt(nv / nu)  # перенос масштаба
            Bu.mul_(c[None, :]); Bv.mul_(1.0 / c[:, None])  # применить
            max_bg = float(c.log().abs().max())  # сдвиг
    return {"max_log_shift_swiglu": max_sw, "max_log_shift_bigram": max_bg}  # итог


def quantize_model(model: "HagiOptimalLM", bits: int = 4, use_had: bool = True) -> tuple[dict, int]:
    """Квантовать все 2D-веса per-row; таблицу emb (256 строк — словарь) — без поворота строк.
        Возвращает (параметры по именам, число байт уровней)."""
    out: dict[str, dict] = {}; nbytes = 0  # результат и объём
    for name, p in model.__dict__.items():  # по атрибутам
        if not isinstance(p, t.nn.Parameter) or p.ndim != 2:  # только 2D
            continue  # дальше
        W = p.detach().cpu().numpy()  # в numpy
        par, Q = quant_per_row(W, bits=bits, use_had=use_had, rotate_rows=(name != "emb"))  # emb — по словарю не вращать
        par["Q"] = Q; par["dtype"] = str(p.dtype); out[name] = par; nbytes += Q.size * (2 if bits == 2 else 4) // 8 + par["scales"].nbytes  # объём
    return out, nbytes  # вернуть


def dequantize_into(model: "HagiOptimalLM", q: dict) -> None:
    """Записать квантованные веса обратно в модель (для оценки потерь)."""
    with t.no_grad():  # без графа
        for name, par in q.items():  # по именам
            getattr(model, name).copy_(t.as_tensor(dequant_per_row(par, par["Q"]), dtype=getattr(model, name).dtype))  # записать


class StreamState:
    """Побайтовое O(1)-состояние HagiOptimalLM: каждый шаг — один байт, логит следующего байта.

        Воспроизводит forward_seq точно (проверено тестом): буферы лагов, короткой свёртки,
        предыдущего скрытого, матрицы состояния GDN по головам. Состояние — тензоры, обновление
        на месте; без графа (инференс)."""

    def __init__(self, model: "HagiOptimalLM") -> None:  # создать по модели
        m = model; d, H, hd = m.dim, m.n_heads, m.dim // m.n_heads  # размеры
        self.m = m; self.d, self.H, self.hd = d, H, hd  # запомнить
        dev = m.W_loc.device  # устройство состояния = устройство параметров модели (иначе устройства расходятся)
        f = lambda *shape: t.zeros(*shape, dtype=t.float32, device=dev)  # нулевой тензор
        self.max_lag = max(m.LAGS)  # наибольший лаг
        self.e_buf = f(self.max_lag, d)  # вложения последних байтов (старые — первые)
        self.h_buf = f(2, d)  # скрытые для короткой свёртки: [-2], [-1]
        self.h_prev = f(d)  # скрытое предыдущего шага (гейт границ)
        self.x_prev = 0  # предыдущий байт (для хэша биграмм)
        self.hist = bytearray()  # последние 5 байтов истории (для словарей точного обхода)
        self.S1 = f(H, hd, hd)  # состояние GDN уровня 1
        self.S2 = f(H, hd, hd)  # состояние GDN уровня 2
        self.n = 0  # число шагов

    def step(self, b: int) -> t.Tensor:
        """Обработать байт b и вернуть логиты следующего байта (256). Равен forward_seq(ctx + [b])[:, -1]."""
        m = self.m; d, H, hd = self.d, self.H, self.hd  # размеры
        with t.no_grad():  # без графа
            self.hist.append(b); del self.hist[:-5]  # история: последние 5 байтов, включая текущий (как в forward_seq)
            e = m.emb[b]  # вложение байта, (d)
            acc = t.zeros(d, device=self.e_buf.device)  # взвешенная сумма лагов
            for li, lag in enumerate(m.LAGS):  # по лагам
                if self.n >= lag:  # лаг уже есть в истории (до текущего байта)
                    acc = acc + self.e_buf[-lag] * m.w_lag[li]  # вклад
            self.e_buf = t.cat([self.e_buf[1:], e[None]], dim=0)  # сдвинуть буфер, добавить e
            x0 = rms_norm(e + acc @ m.W_loc, m.g_loc)  # локальный признак
            bi_idx = ((self.x_prev * 131) ^ b) % 256  # хэш пары (предыдущий, текущий)
            self.x_prev = b  # запомнить
            h = x0 + m.E_bi[bi_idx] @ m.W_bi  # с биграммным признаком
            h_in = h * m.w_sc[0] + self.h_buf[-1] * m.w_sc[1] + self.h_buf[-2] * m.w_sc[2]  # короткая свёртка
            self.h_buf = t.cat([self.h_buf[1:], h[None]], dim=0)  # сдвинуть
            q = (h_in @ m.W_gq).reshape(H, hd); q = t.nn.functional.normalize(q, dim=-1)  # запросы по головам
            k = (h_in @ m.W_gk).reshape(H, hd); k = t.nn.functional.normalize(k, dim=-1)  # ключи
            v = (h_in @ m.W_gv).reshape(H, hd)  # значения
            a = t.sigmoid(h_in @ m.W_alpha + m.b_alpha) * 0.96  # удержание, (H)
            bb = t.sigmoid(h_in @ m.W_beta + m.b_beta)  # запись, (H)
            aS = a[:, None, None] * self.S1  # затухшее состояние α·S (каноническая формула GDNBridge)
            Sk = (aS @ k.unsqueeze(-1)).squeeze(-1)  # (α·S) k по головам, (H, hd)
            self.S1 = aS + bb[:, None, None] * ((v - Sk).unsqueeze(-1) @ k.unsqueeze(-2))  # запись: aS + β (v − aS k) kᵀ
            y1 = (self.S1 @ q.unsqueeze(-1)).squeeze(-1).reshape(d)  # выход уровня 1
            if m.use_gdn_norm:  # нормировка
                y1 = rms_norm(y1, m.g_gdn1)  # стабилизация
            h = h + y1 @ m.W_go  # остаток
            if m.use_hier_gdn:  # уровень 2
                p_bnd = t.sigmoid(t.cat([h, self.h_prev]) @ m.w_bnd + m.b_bnd)  # вероятность границы, (1)
                h_before = h  # скрытое до уровня 2 (для h_prev)
                q2 = t.nn.functional.normalize((h @ m.W_hq).reshape(H, hd), dim=-1)  # запросы
                k2 = t.nn.functional.normalize((h @ m.W_hk).reshape(H, hd), dim=-1)  # ключи
                v2 = (h @ m.W_hv).reshape(H, hd)  # значения
                a2 = (1.0 - p_bnd * (1.0 - t.sigmoid(m.a_slow))) * 0.985  # удержание, (H)
                b2 = p_bnd * t.sigmoid(m.b_slow)  # запись, (H)
                aS2 = a2[:, None, None] * self.S2  # затухшее состояние уровня 2
                Sk2 = (aS2 @ k2.unsqueeze(-1)).squeeze(-1)  # произведение затухшего состояния на ключ
                self.S2 = aS2 + b2[:, None, None] * ((v2 - Sk2).unsqueeze(-1) @ k2.unsqueeze(-2))  # каноническая запись
                y2 = (self.S2 @ q2.unsqueeze(-1)).squeeze(-1).reshape(d)  # выход
                if m.use_gdn_norm:  # нормировка
                    y2 = rms_norm(y2, m.g_gdn2)  # стабилизация
                h = h + y2 @ m.W_ho  # остаток
                self.h_prev = h_before  # h_prev следующего шага — скрытое ДО уровня 2 (как в forward_hidden: cat([h, h_prev]) с h_prev = h_t−1 после уровня 1)
            hn = m._spike(rms_norm(h, m.g_mlp))  # перед MLP (со спайкингом, если включён — как в forward_hidden)
            h = h + swiglu(hn, m.W_gate, m.W_up, m.W_down)  # основной блок
            for Wg, Wu, Wd in m.blocks:  # дополнительные блоки
                h = h + swiglu(m._spike(rms_norm(h, m.g_mlp)), Wg, Wu, Wd)  # добавить
            h_out = rms_norm(h, m.g_out)  # выходное скрытое
            h_out = m._apply_adapter(h_out[None])[0]  # доменный адаптер, если задан
            lg = (h_out @ m.emb.T) * m.out_scale + m.b_out  # логиты
            if m.use_bigram_logit_skip:  # скип
                lg = lg + m.B_u[b] @ m.B_v  # по текущему байту
            fib = getattr(m, "active_fiber", None)  # волокно домена
            if fib is not None:  # задано
                lg = lg + fib.logits_delta(h_out[None], t.as_tensor([b], device=h_out.device))[0, 0]  # добавка волокна (h_out — (d,))
            if m.bypass_boost > 0:  # точный обход включён — как в forward_seq (по последним 5 байтам истории)
                lg = lg + m.bypass_apply(t.as_tensor(list(self.hist), dtype=t.long, device=h_out.device)[None], t.zeros((1, len(self.hist), 256), device=h_out.device))[0, -1]  # прибавка словарей: история как окно (1, n), логит как позиция (1, 1, 256)
        self.n += 1  # счётчик шагов
        return lg  # логиты следующего байта


def _slot(model, params: list[t.Tensor]):
    """Найти слот параметров: (kind, container, keys). kind: "block" — блок модели (dict HNet или список optimal), "model" — атрибуты модели."""
    for bi, blk in enumerate(getattr(model, "blocks", [])):  # по блокам
        vals = list(blk.values()) if isinstance(blk, dict) else list(blk)  # значения блока
        if params and all(any(v is p for v in vals) for p in params):  # все параметры — из этого блока
            if isinstance(blk, dict):  # словарь HNet
                return ("block", blk, [k for p in params for k, v in blk.items() if v is p])  # словарь и ключи
            lst = list(blk); model.blocks[bi] = lst  # optimal: кортеж → список (изменяемый) на время шага
            return ("block", lst, [next(i for i, v in enumerate(lst) if v is p) for p in params])  # список и индексы
    return ("model", vars(model), [k for p in params for k, v in vars(model).items() if v is p])  # атрибуты модели


def certified_step(model, logits_fn, params: list[t.Tensor], X: t.Tensor, y: t.Tensor) -> dict:
    """Один шаг по антиградиенту с АНАЛИТИЧЕСКОЙ длиной из квадратичного мажоризатора.

        Теория (квадратичный мажоризатор перекрёстной энтропии):
            CE(θ + ηD) ≤ CE(θ) + η·dirAction + (η·dirRange)²/8,  dirAction = mean_i⟨r_i, JD_i⟩,
        dirRange = 2·mean_i Σ_v |JD_iv|, JD = J·D (смещение логитов на единичный шаг). Минимум параболы:
            η* = −4·dirAction / dirRange²,  гарантированное убывание 2·dirAction²/dirRange².
        Learning rate отсутствует: длина шага выводится из данных. Шаг применяется только если
        фактическое убывание неотрицательно (иначе откат)."""
    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual  # прямой режим автодифференцирования
    kind, cont, keys = _slot(model, params)  # слот параметров
    loss = t.nn.functional.cross_entropy(logits_fn(X), y)  # кросс-энтропия до шага
    ce0 = float(loss.detach())  # число
    g = t.autograd.grad(loss, params)  # градиенты по параметрам
    D = [-gi for gi in g]  # направление — антиградиент
    A = float(sum((gi ** 2).sum() for gi in g))  # сумма квадратов градиентов
    none = {"eta": 0.0, "gain_pred": 0.0, "ce_before": ce0, "ce_after": ce0, "gain": 0.0, "grad_norm_sq": A}  # итог без шага
    if A == 0.0:  # градиент нулевой
        return none  # шага нет
    saved = [cont[k] for k in keys]  # прежние значения
    try:  # с восстановлением
        with dual_level():  # прямой режим
            for k, p_, d_ in zip(keys, params, D): cont[k] = make_dual(p_, d_)  # пары (θ, D)
            JD = unpack_dual(logits_fn(X)).tangent  # касательная = J·D, (B, 256)
    finally:  # всегда
        for k, v in zip(keys, saved): cont[k] = v  # вернуть
        if kind == "block" and isinstance(cont, list):  # optimal: список → обратно в кортеж
            model.blocks[model.blocks.index(cont)] = tuple(cont)  # вернуть кортеж
    B = X.shape[0]  # число образцов
    pr = t.softmax(logits_fn(X), dim=-1)  # вероятности (B, V)
    r = pr - t.nn.functional.one_hot(y, pr.shape[-1]).to(pr.dtype)  # остаток (B, V)
    a_b = (r * JD).sum(dim=-1)  # dirAction по образцам: ⟨r_b, JD_b⟩, (B,)
    R_b = JD.max(dim=-1).values - JD.min(dim=-1).values  # dirLogitRange по образцам: sup_v JD − inf_v JD (острый диапазон; при V = 256 в ~72 раза меньше 2·Σ|JD|), (B,)
    a_sum = float(a_b.sum()); R2_sum = float((R_b ** 2).sum())  # Σ_b a_b и Σ_b R_b² (мажоризатор — сумма по пакету)
    if R2_sum <= 0.0 or a_sum >= 0.0:  # направление не уменьшает CE (сумма dirAction ≥ 0) или смещения нет
        return none  # шага нет
    eta = -4.0 * a_sum / R2_sum  # η* = argmin параболы η·Σa + η²·ΣR²/8
    gain_pred = 2.0 * a_sum ** 2 / R2_sum / B  # гарантированное убывание 2·(Σa)²/ΣR², на образец
    with t.no_grad():  # применить шаг
        for p_, d_ in zip(params, D): p_.add_(d_, alpha=eta)  # шаг по направлению
        ce1 = float(t.nn.functional.cross_entropy(logits_fn(X), y))  # CE после
        if ce1 > ce0:  # убывания нет (округление) — откат
            for p_, d_ in zip(params, D): p_.add_(d_, alpha=-eta)  # вернуть
            ce1 = ce0  # CE прежняя
    return {"eta": eta, "gain_pred": gain_pred, "ce_before": ce0, "ce_after": ce1, "gain": ce0 - ce1, "grad_norm_sq": A, "dir": [d.detach() for d in D]}  # итог (dir — направление, для расширения шага)



def derived_lipschitz(model, logits_fn, params: list[t.Tensor], X: t.Tensor, y: t.Tensor) -> dict:
    """Вывести локальную липшицеву константу градиента L и оптимальный шаг γ* = 1/L из мажоризатора.

        Теория (выводимая конфигурация обучения): γ* = 1/L (nlDrop максимален,
        NegativeTransfer.nlDrop_le_optimal), η* = −4a/R² (FunctionPreservingGrowth.useful_first_step).
        Обе величины — из одного квадратичного мажоризатора BatchCEDescent.dirMajorizer:
            drop(η*) = 2a²/R²  и  drop(γ*) = G²/(2L)  ⇒  L = G²·R²/(4a²),  G = ‖grad‖.
        Измеряется на одном батче; параметров нет. Возвращает L, γ*, η*, предсказанный спуск."""
    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual  # прямой режим
    kind, cont, keys = _slot(model, params)  # слот параметров
    loss = t.nn.functional.cross_entropy(logits_fn(X), y)  # кросс-энтропия до шага
    g = t.autograd.grad(loss, params)  # градиенты
    D = [-gi for gi in g]  # антиградиент
    G2 = float(sum((gi ** 2).sum() for gi in g))  # квадрат нормы градиента
    if G2 == 0.0:  # нулевой градиент
        return {"L": float("inf"), "gamma": 0.0, "eta": 0.0, "drop": 0.0}  # шага нет
    saved = [cont[k] for k in keys]  # прежние значения
    try:  # с восстановлением
        with dual_level():  # прямой режим
            for k, p_, d_ in zip(keys, params, D): cont[k] = make_dual(p_, d_)  # пары (θ, D)
            JD = unpack_dual(logits_fn(X)).tangent  # смещение логитов на единичный шаг
    finally:  # всегда
        for k, v in zip(keys, saved): cont[k] = v  # вернуть
        if kind == "block" and isinstance(cont, list):  # optimal: список → кортеж
            model.blocks[model.blocks.index(cont)] = tuple(cont)  # вернуть
    with t.no_grad():  # оценки
        pr = t.softmax(logits_fn(X), dim=-1)  # вероятности
        r = pr - t.nn.functional.one_hot(y, pr.shape[-1]).to(pr.dtype)  # остаток
        a_b = (r * JD).sum(dim=-1)  # dirAction по образцам
    R_b = JD.max(dim=-1).values - JD.min(dim=-1).values  # dirLogitRange по образцам: sup_v JD − inf_v JD (острый диапазон), (B,)
    a_sum = float(a_b.sum()); R2_sum = float((R_b ** 2).sum())  # суммы по пакету
    if R2_sum <= 0.0 or a_sum >= 0.0:  # направления нет
        return {"L": float("inf"), "gamma": 0.0, "eta": 0.0, "drop": 0.0}  # шага нет
    L = G2 * R2_sum / (4.0 * a_sum ** 2)  # липшицева константа из мажоризатора
    return {"L": L, "gamma": 1.0 / L, "eta": -4.0 * a_sum / R2_sum, "drop": 2.0 * a_sum ** 2 / R2_sum / X.shape[0]}  # γ* = 1/L, η*, спуск на образец


def derived_probe_size(N: int, delta: float, eps: float, B: float) -> int:
    """Минимальный размер пробы по теории: k ≥ ln(2N/δ) / (2·(ε/B)²).

        N — число проверяемых гипотез (состояний блока), δ — вероятность ложного приёма,
        ε — различие, которое надо заметить (нат CE), B — граница |выигрыш| на пробе (Hoeffding)."""
    import math as _m  # логарифм
    if eps <= 0 or B <= 0 or N <= 0:  # некорректные входы
        return 1  # нейтрально
    return int(_m.ceil(_m.log(2.0 * N / delta) / (2.0 * (eps / B) ** 2)))  # минимальный размер пробы


def ef21_qat(model: "HagiOptimalLM", data: bytes, steps: int = 35, lr_muon: float = 0.01, lr_adam: float = 1.5e-3) -> dict:
    """Квантование с обучением (EF21-QAT): градиент берётся на КВАНТОВАННЫХ весах, шаг применяется
        к полноточным master-весам; масштаб шага линейно гасится. Перенесено с numpy-шима аудита
        (run_ef21_qat_q4). Возвращает потери CE до/после квантования (float и QAT)."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # индексы
    if len(arr) < 70:  # мало
        arr = np.tile(arr, max(2, 128 // max(1, len(arr))))  # повторить
    mats, vecs = model.matrix_and_vec_params()  # группы
    opt = HybridMuonAdamW(mats, vecs, lr_muon=lr_muon, lr_adam=lr_adam, wd=0.0)  # оптимизатор без затухания
    opt.names.update({id(p): n for n, p in model.__dict__.items() if isinstance(p, t.nn.Parameter)})  # имена
    rng = np.random.default_rng(77)  # выборка
    master = {n: p.detach().clone() for n, p in model.__dict__.items() if isinstance(p, t.nn.Parameter)}  # полноточные веса
    for stp in range(int(steps)):  # шаги доводки
        q, _ = quantize_model(model, bits=4)  # квантовать текущие
        dequantize_into(model, q)  # веса → квантованные (прямник видит их)
        starts = rng.integers(0, len(arr) - 65, size=16)  # старты
        X = t.as_tensor(np.stack([arr[s: s + 64] for s in starts]), dtype=t.long); y = t.as_tensor(np.stack([arr[s + 1: s + 65] for s in starts]))  # батч
        loss = t.nn.functional.cross_entropy(model.forward_seq(X).reshape(-1, 256), y.reshape(-1))  # CE на квантованных
        opt.zero_grad(); loss.backward()  # градиент
        with t.no_grad():  # восстановить master
            for n, w in master.items(): getattr(model, n).copy_(w)  # вернуть
        opt.set_lr_scale(1.0 - stp / max(1, steps))  # линейное затухание масштаба
        opt.step()  # шаг по master
        master = {n: p.detach().clone() for n, p in model.__dict__.items() if isinstance(p, t.nn.Parameter)}  # новый master
    q, nbytes = quantize_model(model, bits=4)  # финальное квантование
    return {"q": q, "bytes": nbytes}  # параметры и объём
