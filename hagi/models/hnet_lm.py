"""HNetByteLM — байтовая языковая модель HNet на torch: по окну из k байтов
предсказывает распределение вероятностей следующего байта (256 логитов).

Ключевые идеи.
  Контекст ctx имеет форму (B, k), байты идут от старых к новым; предсказывается
  байт сразу после окна. Из окна берутся два источника информации:
  1. «близкие» байты на позициях loc_idx относительно конца окна — их вложения
     сцепляются в вектор x0 (B, 6·d) и линейно сжимаются матрицей W_loc в (B, d);
  2. «дальний» контекст — последние M·r байтов (M = k // r) сжимаются в M pooling-
     векторов: C[m] = Σ_{j=0..r−1} p_pool[j]·E[m·r + j], затем нормируются.
  Запрос Q строится из x0, ключи K и значения V — из C; внимание однонаправленное
  (позиция запроса — последняя), с поворотом RoPE и сдвигом ALiBi. Результат
  внимания добавляется к x0, затем gated-блок (гейш-подобный): h2 = h1 + (gp·σ(gp) ⊙
  (h1_n·W_u))·W_d, gp = h1_n·W_g. Логиты: lg = h2·W_out + b2, и если включены
  bigram-вложения, lg += bi_emb[idx]·W_bi, idx = (ctx[−2]·131 + ctx[−1]) mod bi_buckets.

Нормирование — RMSNorm: rms(x) = x / sqrt(mean(x²) + eps) ⊙ g.
RoPE (на пару координат 2i, 2i+1): (x0, x1) -> (x0·cos − x1·sin, x0·sin + x1·cos),
  углы ang[m, i] = m·freq_i, freq_i = theta0^(−2i/hd).
ALiBi: сдвиг оценки для позиции m равен −lam·(M − 1 − m): чем позиция старше,
  тем штраф больше. Внимание: w = softmax(sc), sc = (K·Qr)/sqrt(hd) + alibi.

Точный рост ширины grow_width: новая модель шире, логиты совпадают со старыми
  точно (при n_heads = 1): матрицы расширяются нулевыми блоками, шкалы RMSNorm
  умножаются на sqrt(d0/d1), eps — на d0/d1, блок W_q — на sqrt(d1/d0).

Тренировка train_guarded/eval_holdout_ce/generate заимствуются у класса
  RopeAttentionByteLM из hagi.lm ( train_certified вызывает train_guarded и
  помечает уточнение как отсутствующее). Сохранение/загрузка — по именам из
  param_names.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import math  # квадратный корень и число пи


class HNetByteLM:
    """Байтовая языковая модель HNet: logits(ctx) -> (B, 256) логитов следующего байта.

        Атрибуты-размеры: dim — ширина потока d; hidden — ширина скрытого слоя
        gated-блока; pool_r — число байтов r в одном pooling-векторе; n_heads —
        число голов nh; hd = dim // n_heads — размер головы; k — длина окна.
        Веса: emb (256, d), W_loc (6·d, d), p_pool (r, d), W_q, W_k, W_v, W_o
        (d, d), rms1, rms2 (d,), W_g, W_u (d, hidden), W_d (hidden, d), b2 (256,),
        W_out (d, 256); дополнительно, если bi_buckets > 0: bi_emb (bi_buckets,
        d_bi), W_bi (d_bi, 256). Таблицы cos_m, sin_m (1, M, 1, hd/2), alibi
        (1, 1, M)."""

    ARCH = "hnet"  # имя архитектуры (по нему выбирают модель)
    k = 64  # длина окна контекста в байтах
    adapter_dim = None   # в __init__ становится dim — ширина, к которой подключается доменный адаптер
    _TRAIN_MIXIN = None   # класс-донор тренировочных методов, задаётся ниже в _install

    def __init__(self, dim: int = 24, hidden: int = 48, pool_r: int = 4,  # создать модель: d, hidden, r, ...
                 n_heads: int = 1, bi_buckets: int = 256, d_bi: int = 12,  # ...nh, число bigram-карманов и их размерность
                 theta0: float = 10000.0, lam: float = 0.15, seed: int = 0,  # ...база RoPE, наклон ALiBi, зерно генератора
                 **_ignored):  # прочие именованные аргументы игнорируются
        import torch as t  # импорт torch (отложенный, внутри конструктора)
        self.t = t  # сохранить ссылку на torch в экземпляре
        self.dim, self.hidden, self.r, self.nh = dim, hidden, pool_r, n_heads  # запомнить размеры: d, hidden, r, nh
        self.hd = dim // n_heads  # размер одной головы hd = d // nh
        self.adapter_dim = dim  # адаптер оперирует векторами ширины d
        self.theta0, self.lam = theta0, lam  # база частот RoPE и наклон ALiBi
        self.bi_buckets, self.d_bi = bi_buckets, d_bi  # число bigram-карманов и размерность их вложений
        g = t.Generator().manual_seed(seed)  # генератор случайных чисел с зерном seed
        def rn(*sh, s=1.0):  # создать обучаемый параметр заданной формы
            """Параметр из нормальных чисел N(0, 1), умноженных на масштаб s: w = s·n."""
            return t.nn.Parameter(t.randn(*sh, generator=g) * s)  # форма sh, значения n·s, n ~ N(0,1)
        d = dim  # короткое имя ширины d
        self.emb = rn(256, d, s=0.08)  # вложения байтов (256, d), масштаб 0.08
        self.loc_idx = (-8, -6, -4, -3, -2, -1)  # позиции «близких» байтов от конца окна (−1 — последний)
        self.W_loc = rn(len(self.loc_idx) * d, d, s=1 / math.sqrt(len(self.loc_idx) * d))  # W_loc (6d, d), масштаб 1/sqrt(6d)
        self.p_pool = t.nn.Parameter(t.full((pool_r, d), 1.0 / pool_r))  # веса pooling (r, d), начально 1/r — среднее
        self.W_q = rn(d, d, s=1 / math.sqrt(d)); self.W_k = rn(d, d, s=1 / math.sqrt(d))  # матрицы запроса и ключа (d, d), масштаб 1/sqrt(d)
        self.W_v = rn(d, d, s=1 / math.sqrt(d)); self.W_o = rn(d, d, s=1 / math.sqrt(d))  # матрицы значения и выхода внимания (d, d), масштаб 1/sqrt(d)
        self.rms1 = t.nn.Parameter(t.ones(d)); self.rms2 = t.nn.Parameter(t.ones(d))  # шкалы RMSNorm (d,), начально единицы
        self.W_g = rn(d, hidden, s=1 / math.sqrt(d)); self.W_u = rn(d, hidden, s=1 / math.sqrt(d))  # W_g, W_u (d, hidden) для веток гейта и значения
        self.W_d = rn(hidden, d, s=1 / math.sqrt(hidden))  # W_d (hidden, d) — выход gated-блока
        self.b2 = t.nn.Parameter(t.zeros(256))  # смещение логитов (256,), нули
        self.W_out = rn(d, 256, s=1 / math.sqrt(d))  # выходная матрица W_out (d, 256)
        self._extra = []  # дополнительные параметры (bigram), пока пусто
        self.rms_eps = 1e-5  # eps в RMSNorm
        if bi_buckets > 0:  # bigram-вложения включены
            self.bi_emb = rn(bi_buckets, d_bi, s=0.05)  # вложения пар байтов (bi_buckets, d_bi), масштаб 0.05
            self.W_bi = rn(d_bi, 256, s=1 / math.sqrt(d_bi))  # матрица bigram-поправки (d_bi, 256)
            self._extra = [self.bi_emb, self.W_bi]  # они входят в список параметров
        M = self.k // pool_r  # число pooling-позиций M = k // r
        half = self.hd // 2  # число пар координат в голове half = hd // 2
        freq = theta0 ** (-2.0 * t.arange(half, dtype=t.float32) / max(1, self.hd))  # частоты freq_i = theta0^(−2i/hd), i = 0..half−1
        ang = t.arange(M, dtype=t.float32)[:, None] * freq[None, :]  # углы ang[m, i] = m·freq_i, форма (M, half)
        self.register_buffer = None  # заглушка: у класса нет буферов torch, атрибут заменён на None
        self.cos_m = t.cos(ang)[None, :, None, :]  # косинусы углов, форма (1, M, 1, half)
        self.sin_m = t.sin(ang)[None, :, None, :]  # синусы углов, форма (1, M, 1, half)
        self.alibi = (-lam * (M - 1 - t.arange(M, dtype=t.float32)))[None, None, :]  # alibi[m] = −lam·(M − 1 − m), форма (1, 1, M)
        self.blocks: list[dict] = []  # список блоков — всегда пуст (у модели нет добавляемых блоков)
        self.frozen_blocks: set = set()  # множество замороженных блоков — пусто
        self.domain_adapter = None  # доменный адаптер (пара A, B) — не задан
        self.device = "cpu"  # устройство по умолчанию
        self._resolve_device()  # выбрать устройство и перенести параметры

# -- служебные методы --------------------------------------------------------
    def _resolve_device(self):
        """Выбрать устройство (cpu или cuda) и перенести на него параметры и таблицы."""
        import os  # импорт os (не используется далее)
        from hagi.lm import _resolve_device  # общий выбор устройства по torch, dim и hidden
        self.device = _resolve_device(self.t, self.dim, self.hidden)  # имя устройства: "cpu" или "cuda"
        if self.device == "cuda":  # выбран cuda
            for p in self.params():  # по всем параметрам
                p.data = p.data.to(self.device)  # перенести данные параметра на устройство
            self.cos_m, self.sin_m, self.alibi = (x.to(self.device) for x in (self.cos_m, self.sin_m, self.alibi))  # перенести таблицы cos, sin, alibi

    _NAMES = ("emb", "W_loc", "p_pool", "W_q", "W_k", "W_v", "W_o", "rms1", "rms2", "W_g", "W_u", "W_d", "b2", "W_out")  # имена основных 14 параметров

    def param_names(self):
        """Список имён параметров: 14 основных и, если включены bigram-вложения, bi_emb и W_bi."""
        return list(self._NAMES) + (["bi_emb", "W_bi"] if self._extra else [])  # копия имён + ["bi_emb", "W_bi"], если self._extra непусто

    def params(self):
        """Список обучаемых параметров в том же порядке, что имена из param_names."""
        return [self.emb, self.W_loc, self.p_pool, self.W_q, self.W_k, self.W_v, self.W_o,  # первые 7 параметров: emb, W_loc, p_pool, W_q, W_k, W_v, W_o
                self.rms1, self.rms2, self.W_g, self.W_u, self.W_d, self.b2, self.W_out] + self._extra  # + rms1, rms2, W_g, W_u, W_d, b2, W_out и дополнительные bi_emb, W_bi

    def _rope_tables(self, M: int, half: int):
        """Таблицы RoPE (cos, sin) для M позиций и half пар; каждая формы (1, M, 1, half).

            freq_i = theta0^(−2i/hd), ang[m, i] = m·freq_i, результат cos(ang), sin(ang)."""
        t = self.t  # короткое имя torch
        freq = self.theta0 ** (-2.0 * t.arange(half, dtype=t.float32) / max(1, self.hd))  # частоты freq_i = theta0^(−2i/hd), i = 0..half−1
        ang = t.arange(M, dtype=t.float32)[:, None] * freq[None, :]  # углы ang[m, i] = m·freq_i
        return t.cos(ang)[None, :, None, :], t.sin(ang)[None, :, None, :]  # cos(ang) и sin(ang) формы (1, M, 1, half)

    def grow_width(self, d_new: int, h_new: int, seed: int = 0) -> "HNetByteLM":
        """Точный рост ширины: новая модель (d_new, h_new) даёт РОВНО те же логиты, что старая.

            Требуется n_heads = 1, d_new ≥ d0, h_new ≥ h0, d_new чётное. Правила переноса:
              rms_eps_new = rms_eps·d0/d1; rms1_new, rms2_new = rms·sqrt(d0/d1) (новые
              координаты 0); emb, p_pool, W_loc, W_k, W_v, W_o, W_g, W_u, W_d, W_out —
              блоки старой матрицы в левом верхнем углу, остальное 0; W_q — блок,
              умноженный на sqrt(d1/d0); таблицы RoPE — старые частоты, новые пары
              cos = sin = 0; b2 и alibi копируются. Тогда logits_new(x) = logits_old(x).
            Возвращает новую модель."""
        assert self.nh == 1, "width growth is proven for n_heads = 1 only"  # рост доказан только для одной головы
        t = self.t  # короткое имя torch
        d0, h0 = self.dim, self.hidden  # старые ширины d0, h0
        assert d_new >= d0 and h_new >= h0 and d_new % 2 == 0  # новая ширина не меньше старой и чётная
        m = HNetByteLM(dim=d_new, hidden=h_new, pool_r=self.r, n_heads=1,  # новая модель шириной d_new, h_new, с теми же r, nh = 1, ...
                       bi_buckets=self.bi_buckets, d_bi=self.d_bi,  # ...теми же bi_buckets, d_bi, ...
                       theta0=self.theta0, lam=self.lam, seed=seed)  # ...theta0, lam и зерном seed
        rs = math.sqrt(d0 / d_new)  # множитель шкал нормирования rs = sqrt(d0/d1)
        with t.no_grad():  # далее без построения графа градиентов
            m.emb.zero_(); m.emb[:, :d0] = self.emb  # формула:  emb_new = 0; emb_new[:, :d0] = emb_old
            m.p_pool.fill_(1.0 / self.r); m.p_pool[:, :d0] = self.p_pool  # формула:  p_pool_new = 1/r; p_pool_new[:, :d0] = p_pool_old
            m.rms1.zero_(); m.rms1[:d0] = self.rms1 * rs  # формула:  rms1_new = 0; rms1_new[:d0] = rms1_old·rs
            m.rms2.zero_(); m.rms2[:d0] = self.rms2 * rs  # формула:  rms2_new = 0; rms2_new[:d0] = rms2_old·rs
            m.b2.copy_(self.b2)  # b2 не меняется
            n_loc = len(self.loc_idx)  # число близких позиций n_loc = 6
            m.W_loc.zero_()  # формула:  W_loc_new = 0
            for si in range(n_loc):  # по блокам si = 0..n_loc−1 (по одному на близкую позицию)
                m.W_loc[si * d_new:si * d_new + d0, :d0] = self.W_loc[si * d0:(si + 1) * d0, :]  # блок si: строки si·d1..si·d1+d0−1, столбцы 0..d0−1 = старый блок строк si·d0..(si+1)·d0−1
            qsc = math.sqrt(d_new / d0)  # множитель блока W_q qsc = sqrt(d1/d0)
            for name, sc in (("W_q", qsc), ("W_k", 1.0), ("W_v", 1.0)):  # W_q умножается на qsc, W_k и W_v — на 1
                getattr(m, name).zero_(); getattr(m, name)[:d0, :d0] = getattr(self, name) * sc  # новая матрица = 0; блок [:d0, :d0] = старая·sc
            m.W_o.zero_(); m.W_o[:d0, :d0] = self.W_o  # формула:  W_o_new = 0; W_o_new[:d0, :d0] = W_o_old
            m.W_g.zero_(); m.W_g[:d0, :h0] = self.W_g  # формула:  W_g_new = 0; W_g_new[:d0, :h0] = W_g_old
            m.W_u.zero_(); m.W_u[:d0, :h0] = self.W_u  # формула:  W_u_new = 0; W_u_new[:d0, :h0] = W_u_old
            m.W_d.zero_(); m.W_d[:h0, :d0] = self.W_d  # формула:  W_d_new = 0; W_d_new[:h0, :d0] = W_d_old
            m.W_out.zero_(); m.W_out[:d0, :] = self.W_out  # W_out_new = 0; W_out_new[:d0, :] = W_out_old (все 256 столбцов)
            if self._extra:  # bigram-вложения включены
                m.bi_emb.copy_(self.bi_emb); m.W_bi.copy_(self.W_bi)  # их форма не зависит от d — копируются как есть
        # RoPE: старые частоты в точности прежние, новые пары нулевые
        half0, half1 = d0 // 2, d_new // 2  # число пар координат: старое half0 = d0/2, новое half1 = d1/2
        c0, s0 = self._rope_tables(m.k // m.r, half0)  # таблицы cos, sin для старых half0 пар и M позиций
        M = m.k // m.r  # формула:  M = k // r
        zeros = t.zeros((1, M, 1, half1 - half0), device=c0.device)  # нули для новых пар, форма (1, M, 1, half1 − half0)
        m.cos_m = t.cat([c0, zeros], dim=-1); m.sin_m = t.cat([s0, zeros], dim=-1)  # cos_m = [c0 | 0], sin_m = [s0 | 0] по последней оси
        m.alibi = self.alibi.clone()  # alibi копируется
        m.rms_eps = self.rms_eps * (d0 / d_new)  # формула:  eps_new = eps_old·d0/d1
        m.device = self.device  # то же устройство
        if self.device == "cuda":  # устройство cuda
            for p_ in m.params():  # по параметрам новой модели
                p_.data = p_.data.to(self.device)  # перенести на устройство
            m.cos_m, m.sin_m, m.alibi = (x.to(self.device) for x in (m.cos_m, m.sin_m, m.alibi))  # перенести таблицы
        return m  # вернуть новую модель

    def grow(self, seed: int = 0) -> int:      # формула:  interface compat; no-op identity
        """Ничего не делает (у модели нет блоков); возвращает число блоков — 0."""
        return len(self.blocks)  # формула:  len(self.blocks) = 0

    def _block_add(self, a, h):
        """Заглушка добавления блока: остаток не меняется, возвращается входной остаток a."""
        return h  # вернуть h

    # -- модель -----------------------------------------------------------
    def _rms(self, x, g, eps=None):
        """RMSNorm: x / sqrt(mean(x², последняя ось) + eps) ⊙ g, где eps по умолчанию self.rms_eps."""
        eps = self.rms_eps if eps is None else eps  # eps = self.rms_eps, если аргумент не задан
        return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt() * g  # формула:  x / sqrt(mean_j(x_j²) + eps) ⊙ g

    def _rot(self, x, cos, sin):
        """Поворот RoPE на парах координат (2i, 2i+1): четные -> x0·cos − x1·sin, нечетные -> x0·sin + x1·cos."""
        half = self.hd // 2  # число пар half = hd // 2
        if half == 0:  # пар нет (hd < 2)
            return x  # вернуть x без изменений
        x0, x1 = x[..., 0:2 * half:2], x[..., 1:2 * half:2]  # x0 = чётные координаты 0, 2, .., x1 = нечётные 1, 3, ..
        out = x.clone()  # копия x
        out[..., 0:2 * half:2] = x0 * cos - x1 * sin  # чётные: x0·cos − x1·sin
        out[..., 1:2 * half:2] = x0 * sin + x1 * cos  # нечётные: x0·sin + x1·cos
        return out  # повернутый вектор

    def logits(self, ctx):
        """Логиты следующего байта (B, 256) для окна ctx (B, k).

            Шаги: x0 = concat(E[ctx[p]] для p из loc_idx)·W_loc; pooling C[b, m] =
            Σ_j p_pool[j]⊙E[b, m·r + j]; Q = rms(x0)·W_q, K = rms(C)·W_k, V = rms(C)·W_v;
            поворот RoPE; sc = K·Qr/sqrt(hd) + alibi; attn = softmax(sc)·V; остаток
            h1 = x0 + attn·W_o; h2 = h1 + (gp·σ(gp) ⊙ (rms(h1)·W_u))·W_d, gp = rms(h1)·W_g;
            lg = h2·W_out + b2 (+ bi_emb[idx]·W_bi). Промежуточный h2 сохраняется в _last_h2."""
        t = self.t  # короткое имя torch
        if ctx.device != self.device:  # контекст не на устройстве модели
            ctx = ctx.to(self.device)  # перенести контекст на устройство модели
        B, k = ctx.shape  # размер пакета B и длина окна k
        d, r, nh, hd = self.dim, self.r, self.nh, self.hd  # формула:  d, r, nh, hd
        M = k // r  # число pooling-позиций M = k // r
        E = self.emb[ctx]  # вложения байтов окна, форма (B, k, d)
        x0 = E[:, list(self.loc_idx), :].reshape(B, len(self.loc_idx) * d) @ self.W_loc  # x0 = concat(E[:, p] для p из loc_idx)·W_loc, форма (B, 6d)·(6d, d) -> (B, d)
        C = E[:, -M * r:, :].reshape(B, M, r, d).mul(self.p_pool[None, None]).sum(2)  # формула:  C[b, m] = Σ_{j=0..r−1} p_pool[j] ⊙ E[b, m·r + j]
        C_n = self._rms(C, self.rms1)  # C_n = rms(C)·rms1, форма (B, M, d)
        x0_n = self._rms(x0, self.rms1)  # формула:  x0_n = rms(x0)·rms1
        Q = (x0_n @ self.W_q).reshape(B, 1, nh, hd)  # запрос Q = x0_n·W_q, форма (B, 1, nh, hd)
        K = (C_n @ self.W_k).reshape(B, M, nh, hd)  # ключ K = C_n·W_k, форма (B, M, nh, hd)
        V = (C_n @ self.W_v).reshape(B, M, nh, hd)  # значение V = C_n·W_v, форма (B, M, nh, hd)
        Qr = self._rot(Q, self.cos_m[:, -1:], self.sin_m[:, -1:])  # запрос поворачивается на угол ПОСЛЕДНЕЙ позиции: cos_m[:, −1:], sin_m[:, −1:]
        Kr = self._rot(K, self.cos_m, self.sin_m)  # ключи поворачиваются на углы своих позиций m
        sc = t.einsum("bmhd,bnhd->bhm", K, Qr) / math.sqrt(hd) + self.alibi      # формула:  (B,nh,M)
        # оценка sc[b, h, m] = Σ_{j} K[b, m, h, j]·Qr[b, h, j] / sqrt(hd) + alibi[m]
        w = t.softmax(sc, dim=-1)  # w = softmax по m: w_m = exp(sc_m − max sc) / Σ_n exp(sc_n − max sc)
        attn = t.einsum("bhm,bmhd->bhd", w, V).reshape(B, d)  # attn[b, h, :] = Σ_m w[b, h, m]·V[b, m, h, :], затем (B, nh·hd) = (B, d)
        h1 = x0 + attn @ self.W_o  # остаток h1 = x0 + attn·W_o
        h1_n = self._rms(h1, self.rms2)  # формула:  h1_n = rms(h1)·rms2
        gp = h1_n @ self.W_g  # ветка гейта gp = h1_n·W_g, форма (B, hidden)
        h2 = h1 + ((gp * t.sigmoid(gp)) * (h1_n @ self.W_u)) @ self.W_d  # формула:  h2 = h1 + (gp·σ(gp) ⊙ (h1_n·W_u))·W_d, σ(x) = 1/(1 + e^(−x))
        ad = getattr(self, "domain_adapter", None)  # доменный адаптер или None
        if ad is not None:                       # формула:  same residual form as Rope: h + (h@A^T)@B^T
            # адаптер задан: добавка того же вида, что у Rope-модели, h + (h·Aᵀ)·Bᵀ
            At = t.as_tensor(ad.A, device=h2.device, dtype=h2.dtype)  # матрица A как тензор на устройстве и в типе h2
            Bt = t.as_tensor(ad.B, device=h2.device, dtype=h2.dtype)  # матрица B как тензор на устройстве и в типе h2
            h2 = h2 + (h2 @ At.T) @ Bt.T  # формула:  h2 = h2 + (h2·Aᵀ)·Bᵀ
        self._last_h2 = h2  # запомнить остаток h2 (его возвращает features)
        lg = h2 @ self.W_out + self.b2  # логиты lg = h2·W_out + b2, форма (B, 256)
        if self._extra:  # bigram-вложения включены
            idx = (ctx[:, -2].long() * 131 + ctx[:, -1].long()) % self.bi_buckets  # номер кармана idx = (ctx[−2]·131 + ctx[−1]) mod bi_buckets
            lg = lg + self.bi_emb[idx] @ self.W_bi  # формула:  lg = lg + bi_emb[idx]·W_bi
        return lg  # логиты (B, 256)

    def features(self, ctx):
        """Остаток h2 (B, dim) последнего прямого прохода — признаки, которые читает
        выходная голова (по ним обучается голова дистилляции смысла)."""
        if ctx.device != self.device:  # контекст не на устройстве модели
            ctx = ctx.to(self.device)  # перенести контекст на устройство модели
        with self.t.no_grad():  # без графа градиентов
            self.logits(ctx)  # прямой проход (заполняет self._last_h2)
        return self._last_h2  # остаток h2 формы (B, dim)

    def full_logits(self, ctx):
        """Логиты последней позиции для каждого префикса ctx[:, :i], i = 1..L — форма
        (B, L, 256). Считается L прямыми проходами по префиксам (только для генерации;
        при обучении используется logits)."""
        return t_stack_logits(self, ctx)  # собрать логиты всех префиксов

    def train_certified(self, data: bytes, epochs: int = 60,  # обучение; epochs — число эпох, ...
                        holdout: float = 0.1, batch: int = 64,  # ...holdout — доля контрольных данных, batch — размер пакета, ...
                        log_every: int = 0, holdout_seed: int | None = None,  # ...период печати (не используется), зерно разбиения, ...
                        lr: float | None = None) -> dict:  # ...скорость обучения (None — стандартная); возвращает словарь итогов
        """Обучение, совместимое с Unified: вызывает train_guarded.

            Фазы сертифицированного уточнения нет: уточнение модели Rope — производная
            именно её прямого прохода; здесь шаг принимается только по измеренному
            контрольному гейту (это отбор, а не сертификат). В итог добавляется
            r["refine"] = "none (hnet: selection gate, no certificate)"."""
        r = self.train_guarded(data, epochs=epochs, holdout=holdout, batch=batch,  # r = итог train_guarded(data, epochs, holdout, batch, ...
                               holdout_seed=holdout_seed, **({} if lr is None else {"lr": lr}))  # ...holdout_seed и lr, если lr задана)
        r["refine"] = "none (hnet: selection gate, no certificate)"  # пометка: уточнения нет, только гейт отбора
        return r  # итог обучения

    def adapter_head_step(self, ctx, nxt, lr: float = 1e-2):
        """Один шаг обучения ТОЛЬКО доменного адаптера (A, B) на пакете (ctx, nxt);
        остальные параметры заморожены (аналог метода модели Rope).

            Шаг: A' = A − lr·Adam(∇ CE), где CE = cross_entropy(logits_with(ctx, A, B), nxt)."""
        ad = getattr(self, "domain_adapter", None)  # доменный адаптер или None
        if ad is None:  # адаптера нет
            return  # ничего не делать
        t = self.t  # короткое имя torch
        At = t.as_tensor(ad.A, device=self.device, dtype=t.float32).requires_grad_(True)  # A как тензор float32 на устройстве, с градиентом
        Bt = t.as_tensor(ad.B, device=self.device, dtype=t.float32).requires_grad_(True)  # B как тензор float32 на устройстве, с градиентом
        opt = t.optim.Adam([At, Bt], lr=lr)  # оптимизатор Adam по A и B со скоростью lr
        opt.zero_grad()  # обнулить градиенты
        t.nn.functional.cross_entropy(self.logits_with(ctx, At, Bt), nxt.to(self.device)).backward()  # CE(logits_with(ctx, A, B), nxt) и вычислить градиенты
        opt.step()  # шаг Adam: A, B обновляются
        ad.A, ad.B = At.detach().cpu().numpy(), Bt.detach().cpu().numpy()  # записать новые A, B в адаптер как массивы numpy

    def logits_with(self, ctx, At, Bt):
        """logits() с явно заданной парой адаптера (At, Bt); self.domain_adapter после вызова восстанавливается."""
        prev = self.domain_adapter  # запомнить текущий адаптер
        self.domain_adapter = _AB(At, Bt)  # временно подставить пару (At, Bt)
        try:  # выполнить с восстановлением в любом случае
            return self.logits(ctx)  # логиты с временным адаптером
        finally:  # всегда после try
            self.domain_adapter = prev  # восстановить прежний адаптер

    def ce(self, ctx, nxt):
        """Кросс-энтропия logits(ctx) относительно истинных байтов nxt (среднее по пакету)."""
        nxt = nxt.to(ctx.device if ctx.device == self.device else self.device)  # метки на то же устройство, что и ctx (или модель)
        return self.t.nn.functional.cross_entropy(self.logits(ctx), nxt)  # формула:  CE = −mean_i log softmax(logits(ctx))[nxt_i]


def t_stack_logits(m, ctx):
    """(B, L, 256): логиты следующего байта после каждого префикса. Модели нужно ровно
    k столбцов (pooling по r), поэтому короткие префиксы дополняются нулевыми байтами
    СЛЕВА — та же конструкция, что train_guarded использует для первых k позиций файла."""
    t = m.t  # короткое имя torch
    B, L = ctx.shape  # размер пакета B и длина последовательности L
    outs = []  # список логитов по префиксам
    for i in range(1, L + 1):  # по длинам префиксов i = 1..L
        if i >= m.k:  # префикс не короче окна
            outs.append(m.logits(ctx[:, i - m.k:i]))  # логиты по окну ctx[:, i−k : i]
        else:  # префикс короче окна
            pad = t.zeros((B, m.k - i), dtype=ctx.dtype, device=ctx.device)  # нулевые байты формы (B, k − i)
            outs.append(m.logits(t.cat([pad, ctx[:, :i]], dim=1)))  # логиты по [нули | ctx[:, :i]] формы (B, k)
    return t.stack(outs, dim=1)  # объединить в тензор (B, L, 256)


# -- тренировочный миксин: используется цикл обучения RopeAttentionByteLM ---------------
def _install():
    """Приписать HNetByteLM методы обучения класса RopeAttentionByteLM, если их ещё нет."""
    from hagi import lm as _lm  # модуль языковых моделей
    R = _lm.RopeAttentionByteLM  # класс-донор
    for name in ("train_guarded", "eval_holdout_ce", "_read_all_windows",  # по именям: train_guarded, eval_holdout_ce, _read_all_windows, ...
                 "generate", "generate_auto"):  # формула:  ...generate, generate_auto
        if hasattr(R, name) and not hasattr(HNetByteLM, name):  # метод есть у донора и нет у HNetByteLM
            setattr(HNetByteLM, name, getattr(R, name))  # скопировать метод донора в HNetByteLM
    HNetByteLM._TRAIN_MIXIN = R  # запомнить класс-донор


_install()  # выполнить установку методов при импорте модуля


class _AB:
    """Пара матриц доменного адаптера (A, B) в том виде, в котором её читает logits."""

    def __init__(self, A, B):  # создать пару
        self.A, self.B = A, B  # сохранить A и B
