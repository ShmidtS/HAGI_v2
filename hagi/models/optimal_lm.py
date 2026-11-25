"""HagiOptimalLM — причинная байтовая модель: мульти-лаговая свёртка, хэш-биграммы,
двухуровневый Gated DeltaNet с гейтом границ, SwiGLU, связанные эмбеддинги.

Перенесена с numpy-шима (аудит checkpoints/audit.json, запрос 055) на torch.
Формат последовательный: forward_seq даёт логиты (B, T, 256) по всем позициям
сразу, поэтому потеря считается по T целей за один прогон (у HNetByteLM — одна).
Оптимизатор HybridMuonAdamW: Muon (Ньютон–Шульц, ортогонализация) для матриц,
AdamW для векторов; опционально проекция на дополнение подпространства GPM."""
from __future__ import annotations  # аннотации не вычисляются при импорте

import math  # квадратный корень, косинус, число пи
from typing import Any  # произвольный тип

import numpy as np  # численные массивы
import torch as t  # тензоры и автоградиент


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


def gated_deltanet(Q: t.Tensor, K: t.Tensor, V: t.Tensor, alpha: t.Tensor, beta: t.Tensor) -> t.Tensor:
    """Gated DeltaNet по времени: S_t = α_t S_{t−1} + β_t (v_t − S_{t−1} k_t) k_tᵀ; y_t = S_t q_t.

        Q, K, V: (B, H, T, D); alpha, beta: (B, H, T). Возвращает (B, H, T, D).
        Цикл по T с автоградиентом (state-carry BPTT по всей последовательности)."""
    B, H, T, D = Q.shape  # размеры
    S = t.zeros((B, H, D, D), dtype=Q.dtype, device=Q.device)  # состояние — нулевая матрица на голову
    outs = []  # выходы по шагам
    for i in range(T):  # по времени
        k = K[:, :, i]  # форма: пакет, головы, ширина головы
        v = V[:, :, i]  # форма: пакет, головы, ширина головы
        q = Q[:, :, i]  # форма: пакет, головы, ширина головы
        a = alpha[:, :, i][..., None, None]  # форма: пакет, головы, 1, 1
        b = beta[:, :, i][..., None, None]  # форма: пакет, головы, 1, 1
        Sk = (S @ k.unsqueeze(-1)).squeeze(-1)  # произведение состояния на ключ
        S = a * S + b * ((v - Sk).unsqueeze(-1) @ k.unsqueeze(-2))  # обновление состояния (правило Δ)
        outs.append((S @ q.unsqueeze(-1)).squeeze(-1))  # выход: состояние на запрос
    return t.stack(outs, dim=2)  # форма: пакет, головы, время, ширина головы


def muon_orth(G: t.Tensor, iters: int = 5) -> t.Tensor:
    """Классический Muon NS: X ← aX + b(XXᵀ)X + c(XXᵀ)²X. Возвращает G·(GᵀG)^{−1/2}."""
    a, b, c = 3.4445, -4.7750, 2.0315  # коэффициенты
    X = G.float()  # рабочая копия
    X = X / (X.norm() + 1e-7)  # нормировка
    tr = X.size(0) > X.size(1)  # нужна ли транспонация
    if tr:  # высокая
        X = X.T  # сделать широкой
    for _ in range(iters):  # итерации
        A = X @ X.T  # произведение X на транспонированное X
        X = a * X + b * (A @ X) + c * (A @ A @ X)  # полином пятой степени
    if tr:  # вернуть форму
        X = X.T  # назад
    return X  # результат


class HybridMuonAdamW:
    """Muon для 2D-весов, AdamW для остальных; проекция градиента и шага на дополнение GPM."""

    def __init__(self, matrix_params: list[t.Tensor], vec_params: list[t.Tensor],  # параметры по группам
                 lr_muon: float = 0.03, lr_adam: float = 5e-3, momentum: float = 0.9,  # шаги и момент
                 wd: float = 1e-4, ns_iters: int = 5) -> None:  # затухание и число итераций
        self.mats = [p for p in matrix_params if p.requires_grad]  # матрицы (Muon)
        self.names: dict[int, str] = {}  # id веса -> имя (заполняется моделью; ключи GPM — имена, стабильны при save/load)
        self.vecs = [p for p in vec_params if p.requires_grad]  # векторы и таблицы (AdamW)
        self.lr_muon, self.lr_adam = float(lr_muon), float(lr_adam)  # шаги
        self.momentum, self.wd, self.ns_iters = float(momentum), float(wd), int(ns_iters)  # прочее
        self.m_buf: list[t.Tensor] = [t.zeros_like(p) for p in self.mats]  # момент Muon
        self.v_m: list[t.Tensor] = [t.zeros_like(p) for p in self.vecs]  # первый момент AdamW
        self.v_v: list[t.Tensor] = [t.zeros_like(p) for p in self.vecs]  # второй момент AdamW
        self.step_count = 0  # счётчик шагов
        self.gpm_left: dict[int, t.Tensor] = {}  # id веса -> базис левого подпространства (столбцы)
        self.gpm_right: dict[int, t.Tensor] = {}  # id веса -> базис правого подпространства (столбцы)
        self._scale = 1.0  # множитель шага (косинусное расписание)

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
                Ul, Ur = self.gpm_left.get(self.names.get(id(p), id(p))), self.gpm_right.get(self.names.get(id(p), id(p)))  # подпространства веса (по имени)
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
            b1, b2, eps = 0.9, 0.999, 1e-8  # коэффициенты AdamW
            ts = self.step_count  # номер шага
            for idx, p in enumerate(self.vecs):  # векторы и таблицы
                if p.grad is None:  # нет градиента
                    continue  # пропустить
                Ul, Ur = self.gpm_left.get(id(p)), self.gpm_right.get(id(p))  # подпространства
                g = self._proj_out(p.grad.float(), Ul, Ur)  # проекция градиента
                if self.wd > 0 and p.ndim >= 2 and Ul is None and Ur is None:  # затухание только для незащищённых таблиц
                    p.mul_(1.0 - lr_a * self.wd)  # обычное затухание
                self.v_m[idx].mul_(b1).add_((1.0 - b1) * g)  # первый момент
                self.v_v[idx].mul_(b2).add_((1.0 - b2) * g * g)  # второй момент
                mh = self.v_m[idx] / (1.0 - b1 ** ts)  # коррекция смещения
                vh = self.v_v[idx] / (1.0 - b2 ** ts)  # коррекция смещения
                upd_v = self._proj_out(lr_a * mh / (vh.sqrt() + eps), Ul, Ur)  # шаг AdamW с проекцией
                p.copy_((p.float() - upd_v).to(p.dtype))  # записать


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
        self.seed, self.device = seed, "cpu"  # зерно и устройство
        self.t = t  # модуль torch (Unified._windows берёт torch из модели)
        self.adapter_dim = dim  # ширина, к которой применяется доменный адаптер
        self.domain_adapter = None  # доменный адаптер (пара A, B) — не задан
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
        r = self.train_optimal(data, epochs=max(1, epochs // 10), batch=batch or 16, seq_len=self.k, seed=holdout_seed or 7, log_every=log_every)  # обучить (GPM — из self.gpm_left)
        return {"best_holdout_ce": r["val_ce_after"], "holdout_ce": r["val_ce_after"], "val_ce_before": r["val_ce_before"], "steps": r["steps"]}  # как у HNet

    # ---- доменный адаптер в стиле HNet: h_out ← h_out + (h_out @ Aᵀ) @ Bᵀ, A (r, d), B (d, r) ----
    def _apply_adapter(self, h_out: t.Tensor) -> t.Tensor:  # применить доменный адаптер к нормированному скрытому
        ad = getattr(self, "domain_adapter", None)  # адаптер или None
        if ad is None:  # нет
            return h_out  # без изменений
        At = t.as_tensor(ad.A, device=h_out.device, dtype=h_out.dtype); Bt = t.as_tensor(ad.B, device=h_out.device, dtype=h_out.dtype)  # A, B как тензоры
        return h_out + (h_out @ At.T) @ Bt.T  # добавка того же вида, что у HNet

    def logits(self, ctx: t.Tensor) -> t.Tensor:  # логиты следующего байта (B, 256) с адаптером, если задан
        h_out = rms_norm(self.forward_hidden(ctx), self.g_out)  # скрытое последней позиции
        h_out = self._apply_adapter(h_out)  # адаптер домена
        h_last = h_out[:, -1, :]  # последняя позиция
        lg = (h_last @ self.emb.T) * self.out_scale + self.b_out  # связанные эмбеддинги
        if self.use_bigram_logit_skip:  # скип
            lg = lg + self.B_u[ctx[:, -1]] @ self.B_v  # по последнему байту
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

    def param_names(self) -> list[str]:  # имена параметров в порядке params()
        names = [n for n, p in self.__dict__.items() if isinstance(p, t.nn.Parameter)]  # собственные имена
        mats, vecs = self.matrix_and_vec_params()  # группы
        base = {id(p): n for n, p in self.__dict__.items() if isinstance(p, t.nn.Parameter)}  # id -> имя
        out = [base.get(id(p)) for p in mats + vecs if id(p) in base]  # сначала матрицы, потом векторы — как params()
        for i, (Wg, Wu, Wd) in enumerate(self.blocks):  # блоки роста
            out.extend([f"blk{i}.Wg", f"blk{i}.Wu", f"blk{i}.Wd"])  # как у HNet
        return out  # список

    def state_dict(self) -> dict[str, t.Tensor]:  # снимок весов по именам атрибутов
        return {n: p.detach().clone() for n, p in self.__dict__.items() if isinstance(p, t.nn.Parameter)}  # копии

    def load_state_dict_from(self, other: "HagiOptimalLM") -> None:  # загрузить веса из другой модели (той же формы)
        with t.no_grad():  # без графа
            for n, v in other.state_dict().items():  # по именам
                getattr(self, n).copy_(v)  # скопировать

    def params(self) -> list[t.nn.Parameter]:  # все параметры
        m, v = self.matrix_and_vec_params()  # группы
        return m + v  # одним списком

    def forward_hidden(self, x_seq: t.Tensor, capture: dict[int, t.Tensor] | None = None) -> t.Tensor:  # скрытое состояние (B, T, d); capture — входы матриц (для GPM)
        B, T = x_seq.shape  # размеры
        d, H = self.dim, self.n_heads  # ширина и головы
        hd = d // H  # ширина головы
        E = self.emb[x_seq]  # вложения байтов (B, T, d)
        acc = causal_lags(E, self.w_lag, self.LAGS)  # взвешенная сумма лагов
        if capture is not None: capture[id(self.W_loc)] = (E + acc).detach().reshape(-1, d)  # вход W_loc
        x0 = rms_norm(E + acc @ self.W_loc, self.g_loc)  # локальный признак
        x_m1 = t.cat([t.zeros((B, 1), dtype=t.long), x_seq[:, : T - 1]], dim=1)  # предыдущий байт (0 на позиции 0)
        bi_idx = ((x_m1 * 131) ^ x_seq) % 256  # хэш пары байтов
        h = x0 + self.E_bi[bi_idx] @ self.W_bi  # добавить биграммный признак
        h_in = causal_lags(h, self.w_sc, (0, 1, 2))  # короткая 3-tap свёртка
        if capture is not None:  # входы Q/K/V уровня 1 — один h_in
            for W in (self.W_gq, self.W_gk, self.W_gv): capture[id(W)] = h_in.detach().reshape(-1, d)  # записать

        def qkv(W: t.Tensor, x: t.Tensor) -> t.Tensor:  # проекция в головы (B, H, T, hd)
            return (x @ W).reshape(B, T, H, hd).permute(0, 2, 1, 3)  # переставить оси
        Q1 = t.nn.functional.normalize(qkv(self.W_gq, h_in), dim=-1)  # запросы (нормированы)
        K1 = t.nn.functional.normalize(qkv(self.W_gk, h_in), dim=-1)  # ключи (нормированы)
        V1 = qkv(self.W_gv, h_in)  # значения
        a1 = t.sigmoid(h_in @ self.W_alpha + self.b_alpha).permute(0, 2, 1) * 0.96  # удержание < 1
        b1 = t.sigmoid(h_in @ self.W_beta + self.b_beta).permute(0, 2, 1)  # запись
        y1 = gated_deltanet(Q1, K1, V1, a1, b1).permute(0, 2, 1, 3).reshape(B, T, d)  # уровень 1
        if self.use_gdn_norm:  # нормировка
            y1 = rms_norm(y1, self.g_gdn1)  # стабилизирует длинные последовательности
        if capture is not None: capture[id(self.W_go)] = y1.detach().reshape(-1, d)  # вход W_go
        h = h + y1 @ self.W_go  # остаточное соединение
        if self.use_hier_gdn:  # уровень 2
            h_prev = t.cat([t.zeros((B, 1, d)), h[:, : T - 1]], dim=1)  # предыдущее скрытое
            p_bnd = t.sigmoid(t.cat([h, h_prev], dim=-1) @ self.w_bnd + self.b_bnd)  # вероятность границы (B, T, 1)
            Q2 = t.nn.functional.normalize(qkv(self.W_hq, h), dim=-1)  # запросы уровня 2
            K2 = t.nn.functional.normalize(qkv(self.W_hk, h), dim=-1)  # ключи
            V2 = qkv(self.W_hv, h)  # значения
            p_h = p_bnd.permute(0, 2, 1)  # форма: пакет, 1, время
            a2 = (1.0 - p_h * (1.0 - t.sigmoid(self.a_slow)[None, :, None])) * 0.985  # удержание: граница → забвение
            b2 = p_h * t.sigmoid(self.b_slow)[None, :, None]  # запись только на границах
            if capture is not None:  # входы уровня 2
                for W in (self.W_hq, self.W_hk, self.W_hv): capture[id(W)] = h.detach().reshape(-1, d)  # h перед уровнем 2
            y2 = gated_deltanet(Q2, K2, V2, a2, b2).permute(0, 2, 1, 3).reshape(B, T, d)  # уровень 2
            if self.use_gdn_norm:  # нормировка
                y2 = rms_norm(y2, self.g_gdn2)  # стабилизация
            if capture is not None: capture[id(self.W_ho)] = y2.detach().reshape(-1, d)  # вход W_ho
            h = h + y2 @ self.W_ho  # остаточное соединение
        hn = rms_norm(h, self.g_mlp)  # перед MLP
        if capture is not None:  # входы W_gate и W_up
            for W in (self.W_gate, self.W_up): capture[id(W)] = hn.detach().reshape(-1, d)  # записать
        h = h + swiglu(hn, self.W_gate, self.W_up, self.W_down)  # основной блок
        for Wg, Wu, Wd in self.blocks:  # дополнительные блоки (zero-init W_down ⇒ тождество)
            h = h + swiglu(rms_norm(h, self.g_mlp), Wg, Wu, Wd)  # добавить
        return h  # форма: пакет, время, ширина

    def forward_seq(self, x_seq: t.Tensor) -> t.Tensor:  # логиты по всем позициям (B, T, 256)
        h_out = rms_norm(self.forward_hidden(x_seq), self.g_out)  # нормированный выход
        lg = (h_out @ self.emb.T) * self.out_scale + self.b_out  # связанные эмбеддинги: выходная проекция = embᵀ
        if self.use_bigram_logit_skip:  # скип биграмм в логиты
            lg = lg + self.B_u[x_seq] @ self.B_v  # аддитивно
        if self.active_adapter is not None:  # доменный адаптер
            A, B_adp, sc = self.active_adapter  # (d, r), (r, 256), масштаб
            lg = lg + ((h_out @ A) @ B_adp) * sc  # низкоранговая добавка
        return lg  # логиты

    def train_optimal(self, data: bytes, epochs: int = 5, batch: int = 16, seq_len: int = 64,  # обучение
                      lr_muon: float = 0.03, lr_adam: float = 5e-3, max_steps_per_epoch: int = 25,  # шаги
                      holdout: float = 0.15, seed: int = 7, opt_override: Any = None, gpm_left: dict | None = None,  # контроль, оптимизатор, базисы GPM
                      log_every: int = 0) -> dict:  # печать
        import numpy as np  # численные массивы
        n = len(data); cut = int(n * (1.0 - holdout))  # граница обучение/контроль
        tr_arr = np.frombuffer(data[:cut], dtype=np.uint8).astype(np.int64)  # обучение
        va_arr = np.frombuffer(data[cut:], dtype=np.uint8).astype(np.int64)  # контроль
        steps_ep = min(max_steps_per_epoch, max(8, (len(tr_arr) - seq_len - 1) // (batch * 8)))  # шагов на эпоху
        total_steps = max(1, epochs * steps_ep)  # всего шагов
        opt = opt_override or HybridMuonAdamW(*self.matrix_and_vec_params(), lr_muon=lr_muon, lr_adam=lr_adam, wd=1e-4)  # оптимизатор
        gpm = gpm_left if gpm_left is not None else getattr(self, "gpm_left", None)  # базисы: аргумент или атрибут модели
        if gpm:  # заданы
            opt.gpm_left.update(gpm)  # градиент, шаг и затухание не заходят в защищённое подпространство
        opt.names.update({id(p): n for n, p in self.__dict__.items() if isinstance(p, t.nn.Parameter)})  # имена для ключей GPM
        rng = np.random.default_rng(seed)  # выборка окон
        v_starts = np.arange(0, max(1, len(va_arr) - seq_len - 1), seq_len)[: min(32, max(1, len(va_arr) // seq_len))]  # старты контрольных окон
        vx = t.as_tensor(np.stack([va_arr[s: s + seq_len] for s in v_starts]), dtype=t.long)  # входы
        vy = t.as_tensor(np.stack([va_arr[s + 1: s + 1 + seq_len] for s in v_starts]), dtype=t.long)  # цели

        def val_ce() -> float:  # контрольная кросс-энтропия на контроле
            with t.no_grad():  # без графа
                return float(t.nn.functional.cross_entropy(self.forward_seq(vx).reshape(-1, self.VOCAB), vy.reshape(-1)))  # по всем позициям
        h0 = val_ce(); best_val = h0; best_snap = [p.detach().clone() for p in self.params()]  # до обучения
        max_start = max(1, len(tr_arr) - seq_len - 1)  # число стартов
        for step in range(total_steps):  # по шагам
            opt.set_lr_scale(0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, total_steps))))  # косинусный отжиг
            starts = rng.integers(0, max_start, size=batch)  # случайные старты
            bx = t.as_tensor(np.stack([tr_arr[s: s + seq_len] for s in starts]), dtype=t.long)  # входы
            by = t.as_tensor(np.stack([tr_arr[s + 1: s + 1 + seq_len] for s in starts]), dtype=t.long)  # цели
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
        return {"val_ce_before": h0, "val_ce_after": best_val, "steps": total_steps}  # итог


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
            X = t.as_tensor(idx[p[:, None] - k + np.arange(k)[None, :]], dtype=t.long)  # окна
            y = t.as_tensor(idx[p], dtype=t.long)  # цели
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
    return d1  # вернуть


def grow_depth(model: "HagiOptimalLM", seed: int = 0) -> int:
    """Точный рост глубины: SwiGLU-блок с W_down = 0 — функция не меняется (FirstLayerGrad: первый шаг ненулевой)."""
    g = t.Generator().manual_seed(model.seed + 500 + len(model.blocks) + seed)  # зерно
    model.blocks.append((t.nn.Parameter(t.randn((model.dim, model.hidden), generator=g) * 0.06),  # вентиль
                         t.nn.Parameter(t.randn((model.dim, model.hidden), generator=g) * 0.06),  # подъём
                         t.nn.Parameter(t.zeros((model.hidden, model.dim)))))  # нулевой выход — тождество
    return len(model.blocks)  # число блоков


def update_gpm_bases(model: "HagiOptimalLM", data: bytes, gpm_left: dict[int, t.Tensor],  # базисы защищённого подпространства
                     energy_thresh: float = 0.90, max_rank_frac: float = 0.65) -> None:  # порог энергии и доля ранга
    """Обновить базисы GPM по активациям домена: для каждой матрицы — входы, SVD, 90% энергии,
    не более max_rank_frac·d_in столбцов; прежний базис сохраняется (объединение через QR)."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # байты
    if len(arr) < 66:  # слишком коротко
        return  # ничего
    starts = np.linspace(0, len(arr) - 65, num=min(16, max(1, len(arr) // 64)), dtype=np.int64)  # старты окон
    bx = t.as_tensor(np.stack([arr[s: s + 64] for s in starts]), dtype=t.long)  # форма: окна, 64
    acts: dict[int, t.Tensor] = {}  # входы матриц
    with t.no_grad():  # без графа
        model.forward_hidden(bx, capture=acts)  # собрать
    names = {id(p): n for n, p in model.__dict__.items() if isinstance(p, t.nn.Parameter)}  # id -> имя
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
    bx = t.as_tensor(np.stack([arr[s: s + 64] for s in starts]), dtype=t.long)  # входы
    by = t.as_tensor(np.stack([arr[s + 1: s + 65] for s in starts]), dtype=t.long)  # цели
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
    x = t.as_tensor(arr[:-1], dtype=t.long)[None]; y = t.as_tensor(arr[1:], dtype=t.long)  # вход и цели
    with t.no_grad():  # без графа
        h = rms_norm(model.forward_hidden(x), model.g_out)  # скрытое
        lg = (h @ model.emb.T) * model.out_scale + model.b_out  # логиты ствола
        if model.use_bigram_logit_skip: lg = lg + model.B_u[x] @ model.B_v  # скип
        if fib is not None: lg = lg + fib.logits_delta(h, x)  # волокно
        return float(t.nn.functional.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)))  # контрольная кросс-энтропия


class MirrorHedgeStreamPredictor:
    """Потоковый предиктор: модель на окне k + онлайн PPM порядков 0..4; смешивание Mirror-Hedge MWU
    с полом 0.01 (эксперт не умирает). Обновления зависят только от уже данных байтов ⇒ без потерь."""

    def __init__(self, model: "HagiOptimalLM | None" = None, eta: float = 0.25) -> None:  # создать
        self.model, self.eta = model, float(eta)  # модель и шаг
        self.weights = np.array([0.65, 0.35], dtype=np.float64) if model is not None else np.array([1.0])  # веса
        self.uni = np.ones(256, dtype=np.float64)  # счётчик порядка 0
        self.ctx: list[dict[int, dict[int, int]]] = [{}, {}, {}, {}]  # таблицы порядков 1..4
        self.hist: list[int] = []  # история ≤ 16 байтов
        self.st = StreamState(model) if model is not None else None  # побайтовое состояние модели (O(1) на байт)
        self._lg = np.zeros(256)  # последние логиты состояния
        self._last: tuple[np.ndarray, np.ndarray] | None = None  # последние предсказания

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
        if self.model is None:  # без модели
            self._last = (p_ctx, p_ctx); return p_ctx  # только PPM
        if self.st.n == 0:  # истории нет
            p_neu = np.full(256, 1.0 / 256.0)  # равномерно
        else:  # логиты уже посчитаны состоянием на предыдущем байте
            lg = self._lg  # логиты следующего байта
            e = np.exp(lg.double().numpy() - lg.max().item()); p_neu = e / e.sum()  # вероятности
        self._last = (p_neu, p_ctx)  # запомнить
        return self.weights[0] * p_neu + self.weights[1] * p_ctx  # смесь

    def update(self, s: int) -> None:  # после фактического байта s
        if self.model is not None and self._last is not None:  # обновление весов экспертов
            l0 = -math.log(max(1e-12, float(self._last[0][s]))); l1 = -math.log(max(1e-12, float(self._last[1][s])))  # потери
            w0 = self.weights[0] * math.exp(-self.eta * min(10.0, l0)); w1 = self.weights[1] * math.exp(-self.eta * min(10.0, l1))  # экспоненты
            z = w0 + w1; self.weights[0] = 0.98 * (w0 / z) + 0.01; self.weights[1] = 0.98 * (w1 / z) + 0.01  # нормировка с полом
        self.uni[s] += 1.0  # порядок 0
        n = len(self.hist)  # история
        for order in range(1, 5):  # порядки
            if n < order: break  # конец
            h = 0  # накапливаемый хэш контекста
            for b in self.hist[-order:]: h = (h * 257 + b) & 0xFFFFFFFFFFFFFFFF  # хэш
            row = self.ctx[order - 1].setdefault(h, {}); row[s] = row.get(s, 0) + 1  # счётчик
        self.hist.append(s); self.hist = self.hist[-16:]  # обрезать
        if self.st is not None:  # модель есть
            self._lg = self.st.step(int(s))  # один шаг состояния O(1): логиты следующего байта


def hadamard_mat(n: int) -> np.ndarray:
    """Нормированная матрица Адамара порядка n (степень двойки), построение Sylvester."""
    H = np.array([[1.0]], dtype=np.float64)  # порядок 1
    while H.shape[0] < n:  # удваивать
        H = np.block([[H, H], [H, -H]])  # построение Адамара удвоением
    return H / math.sqrt(n)  # нормировать


def quant_per_row(W: np.ndarray, bits: int = 2, use_had: bool = True, rotate_rows: bool = True) -> tuple[dict, np.ndarray]:
    """Построчное квантование W: (опционально) поворот Адамара по блокам, порог/шкала на строку.
        bits=2 — триты {−1, 0, +1}; bits=4 — 16 уровней. Таблицы поиска (emb) — без rotate_rows.
        Возвращает (параметры, уровни int8)."""
    R = W.astype(np.float64, copy=True)  # копия
    r, c = R.shape  # размеры
    if use_had:  # поворот Адамара
        br = max((b for b in (64, 32, 16, 8, 4, 2) if r % b == 0), default=1) if rotate_rows else 1  # блок строк
        if br > 1: R = np.einsum("ij,ajb->aib", hadamard_mat(br), R.reshape(r // br, br, c)).reshape(r, c)  # по строкам
        bc = max((b for b in (64, 32, 16, 8, 4, 2) if c % b == 0), default=1)  # блок столбцов
        if bc > 1: R = (R.reshape(r, c // bc, bc) @ hadamard_mat(bc)).reshape(r, c)  # по столбцам
    scs = np.empty(r, dtype=np.float32); Q = np.zeros((r, c), dtype=np.int8)  # шкалы и уровни
    for i in range(r):  # по строкам
        row = R[i]  # строка
        if bits == 2:  # триты
            ma = float(np.mean(np.abs(row))); thr = 0.7 * ma; pos = np.abs(row) > thr  # порог
            sc = float(np.mean(np.abs(row[pos]))) if np.any(pos) else max(ma, 1e-6)  # шкала
            Q[i] = np.where(row > thr, 1, np.where(row < -thr, -1, 0)).astype(np.int8)  # уровни
        else:  # 4 бита
            m = float(np.max(np.abs(row))) or 1e-6; sc = m / 7.5  # шкала
            Q[i] = np.clip(np.round(row / sc), -7, 8).astype(np.int8)  # уровни
        scs[i] = sc  # сохранить
    return {"bits": int(bits), "use_had": bool(use_had), "rotate_rows": bool(rotate_rows), "shape": (r, c), "scales": scs}, Q  # вернуть


def dequant_per_row(par: dict, Q: np.ndarray) -> np.ndarray:
    """Обратно к quant_per_row: уровни · шкала и обратный поворот Адамара (H симметрична, H·H = I)."""
    r, c = par["shape"]  # размеры
    R = Q.astype(np.float64) * par["scales"][:, None].astype(np.float64)  # расшкалить
    if par["use_had"]:  # повернуть обратно
        bc = max((b for b in (64, 32, 16, 8, 4, 2) if c % b == 0), default=1)  # блок столбцов
        if bc > 1: R = (R.reshape(r, c // bc, bc) @ hadamard_mat(bc)).reshape(r, c)  # обратно
        if par["rotate_rows"]:  # по строкам
            br = max((b for b in (64, 32, 16, 8, 4, 2) if r % b == 0), default=1)  # блок
            if br > 1: R = np.einsum("ij,ajb->aib", hadamard_mat(br), R.reshape(r // br, br, c)).reshape(r, c)  # обратно
    return R  # восстановленная матрица


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
        dev = "cpu"  # состояние на процессоре (побайтный шаг мал; GPU-переключение дороже)
        f = lambda *shape: t.zeros(*shape, dtype=t.float32, device=dev)  # нулевой тензор
        self.max_lag = max(m.LAGS)  # наибольший лаг
        self.e_buf = f(self.max_lag, d)  # вложения последних байтов (старые — первые)
        self.h_buf = f(2, d)  # скрытые для короткой свёртки: [-2], [-1]
        self.h_prev = f(d)  # скрытое предыдущего шага (гейт границ)
        self.x_prev = 0  # предыдущий байт (для хэша биграмм)
        self.S1 = f(H, hd, hd)  # состояние GDN уровня 1
        self.S2 = f(H, hd, hd)  # состояние GDN уровня 2
        self.n = 0  # число шагов

    def step(self, b: int) -> t.Tensor:
        """Обработать байт b и вернуть логиты следующего байта (256). Равен forward_seq(ctx + [b])[:, -1]."""
        m = self.m; d, H, hd = self.d, self.H, self.hd  # размеры
        with t.no_grad():  # без графа
            e = m.emb[b]  # вложение байта, (d)
            acc = t.zeros(d)  # взвешенная сумма лагов
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
            Sk = (self.S1 @ k.unsqueeze(-1)).squeeze(-1)  # S k по головам, (H, hd)
            self.S1 = a[:, None, None] * self.S1 + bb[:, None, None] * ((v - Sk).unsqueeze(-1) @ k.unsqueeze(-2))  # правило Δ
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
                Sk2 = (self.S2 @ k2.unsqueeze(-1)).squeeze(-1)  # произведение состояния на ключ
                self.S2 = a2[:, None, None] * self.S2 + b2[:, None, None] * ((v2 - Sk2).unsqueeze(-1) @ k2.unsqueeze(-2))  # правило Δ
                y2 = (self.S2 @ q2.unsqueeze(-1)).squeeze(-1).reshape(d)  # выход
                if m.use_gdn_norm:  # нормировка
                    y2 = rms_norm(y2, m.g_gdn2)  # стабилизация
                h = h + y2 @ m.W_ho  # остаток
                self.h_prev = h_before  # h_prev следующего шага — скрытое ДО уровня 2 (как в forward_hidden: cat([h, h_prev]) с h_prev = h_t−1 после уровня 1)
            hn = rms_norm(h, m.g_mlp)  # перед MLP
            h = h + swiglu(hn, m.W_gate, m.W_up, m.W_down)  # основной блок
            for Wg, Wu, Wd in m.blocks:  # дополнительные блоки
                h = h + swiglu(rms_norm(h, m.g_mlp), Wg, Wu, Wd)  # добавить
            h_out = rms_norm(h, m.g_out)  # выходное скрытое
            h_out = m._apply_adapter(h_out[None])[0]  # доменный адаптер, если задан
            lg = (h_out @ m.emb.T) * m.out_scale + m.b_out  # логиты
            if m.use_bigram_logit_skip:  # скип
                lg = lg + m.B_u[b] @ m.B_v  # по текущему байту
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
    R_b = 2.0 * JD.abs().sum(dim=-1)  # dirRange по образцам: 2·Σ_v |JD_bv|, (B,)
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
        R_b = 2.0 * JD.abs().sum(dim=-1)  # dirRange по образцам
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
