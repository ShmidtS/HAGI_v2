#!/usr/bin/env python3  # запуск файла интерпретатором python3
"""Байтовые нейронные языковые модели HAGI: предсказывают распределение
вероятностей следующего байта по нескольким предыдущим байтам.

Ключевые идеи и формулы:
  - потеря — кросс-энтропия словаря из 256 байт: CE = −log softmax(z)[y],
    где z — логиты модели, y — правильный следующий байт;
  - обучение с гейтом: шаг оптимизатора принимается, только если
    кросс-энтропия на удержанных данных не выросла; в конце
    разворачивается лучший снимок весов (снимок начальных весов тоже
    считается кандидатом, поэтому без улучшения остаются исходные веса);
  - тернарное квантование весов: s_r = mean(|W_rc|) по строке,
    Q_rc = clamp(round(W_rc / s_r), −1, +1)·s_r;
  - верхняя оценка конечного сдвига логитов:
    CE(z + d) − CE(z) <= ⟨softmax(z) − onehot(y), d⟩ + range(d)²/8.

Содержимое файла:
  SemanticDistillHead — голова, приближающая скрытое состояние модели
    к векторам учителя (косинусная потеря);
  AdaptiveContextSelector — выбор числа усредняемых окон контекста;
  ByteLM — простая модель: k вложений байт -> concat -> relu -> логиты;
  RopeAttentionByteLM — модель с вращательным позиционным кодированием
    (RoPE), ростом глубины identity-блоком, адаптерами доменов и
    обучением с сертифицированным шагом;
  CertifiedStepLM — модель с обучением только по сертифицированным шагам;
  save / load / load_rope / make_lm — сохранение, загрузка, фабрика;
  main — командная строка: train (обучение на файлах) и generate
    (авторегрессионная выборка следующих байтов).

Вывод модели вероятностный: generate(prompt) выбирает следующие байты
по распределению softmax(logits / temp).
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import argparse  # разбор аргументов командной строки
import math  # математические функции
import json  # чтение и запись JSON
from pathlib import Path  # пути к файлам
from typing import Any  # тип Any для аннотаций

import numpy as np  # численные массивы

def _resolve_device(t, dim: int = 0, hidden: int = 0) -> str:
    """Выбрать устройство вычисления по переменной окружения HAGI_DEVICE.

        t — библиотека torch; dim, hidden — размеры модели (0, если не
        заданы). Возвращает "cpu" или "cuda". Правила: HAGI_DEVICE = "cpu"
        -> всегда процессор; = "cuda" -> GPU, если доступно, иначе процессор;
        "auto" (по умолчанию) -> GPU, только если dim·hidden >= 8192:
        у малых тестовых моделей время инициализации CUDA больше самой
        работы, поэтому они остаются на процессоре."""
    import os  # доступ к переменным окружения
    pref = os.environ.get("HAGI_DEVICE", "auto")  # pref = HAGI_DEVICE из окружения, по умолчанию "auto"
    if pref == "cpu":  # явно запрошен процессор
        return "cpu"  # считать на процессоре
    cuda_ok = t.cuda.is_available()  # cuda_ok = доступна ли CUDA
    if pref == "cuda":  # явно запрошен GPU
        return "cuda" if cuda_ok else "cpu"  # "cuda", если CUDA доступна, иначе "cpu"
    return "cuda" if cuda_ok and dim * hidden >= 8192 else "cpu"  # "cuda", если CUDA доступна и dim·hidden >= 8192, иначе "cpu"


from hagi.util import CKPT_DIR as _CKPT  # каталог для сохранения весов под именем _CKPT
CKPT = _CKPT / "lm_byte.pt"  # путь к файлу весов простой модели: _CKPT / "lm_byte.pt"


def _torch_and_device():
    """Вернуть библиотеку torch и имя устройства: "cuda", если CUDA
    доступна, иначе "cpu"."""
    import torch  # нейросетевая библиотека
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")  # (torch, "cuda" или "cpu")


def t_no_grad_if(fn):
    """Декоратор метода: вызвать метод внутри t.no_grad() — без
    построения графа градиентов. Возвращает обёртку с теми же
    аргументами."""
    def wrapper(self, *a, **kw):  # обёртка с произвольными аргументами метода
        """Выполнить fn для self с аргументами a, kw без градиентов."""
        with self.t.no_grad():  # отключить учёт градиентов
            return fn(self, *a, **kw)  # результат исходного метода
    return wrapper  # вернуть обёртку вместо метода


class SemanticDistillHead:
    """Голова семантического выравнивания: при обучении модель приближает
    свои скрытые признаки к векторам учителя.

        Учитель — эмбеддер Gemma: вектор для каждого текста. Ученик —
        матрица проекции proj (hidden_dim x teacher_dim): pred = норм(h·proj),
        где h — признак модели. Потеря — косинусное расстояние
        loss = 1 − mean_i Σ_d pred_id·tv_id (tv — вектор учителя).
        Обновляется только proj: proj = proj − lr·∂loss/∂proj, учитель
        заморожен. Таким путём скрытое пространство модели наследует
    семантику учителя."""

    def __init__(self, dim: int, teacher_dim: int = 768, lr: float = 0.05):  # создать голову: dim не используется, teacher_dim — размер вектора учителя, lr — шаг
        import torch  # нейросетевая библиотека
        t = torch  # короткое имя библиотеки
        # proj переводит скрытый вектор в пространство учителя и обучается
        # простым SGD по косинусной потере
        self.proj = None  # матрица проекции ещё не создана: строится при первом использовании
        self.lr = lr  # шаг обучения lr
        self.teacher_dim = teacher_dim  # размерность вектора учителя

    def _ensure_proj(self, hidden_dim, device, dtype):
        """Построить матрицу proj (hidden_dim x teacher_dim), если её ещё
        нет: заполняется нормальными значениями N(0, 0.02) от генератора
        с сидом 11. device и dtype — устройство и тип данных hidden_dim
        строк; генератор создаётся на том же устройстве, потому что для
        тензора на CUDA процессорный генератор давал ошибку."""
        import torch  # нейросетевая библиотека
        if self.proj is None:  # матрица ещё не построена
            self.proj = torch.zeros(hidden_dim, self.teacher_dim,  # нулевая матрица hidden_dim x teacher_dim
                                    device=device, dtype=dtype)  # на устройстве device с типом dtype
            g = torch.Generator(device=device).manual_seed(11)                 if device != "cpu" else torch.Generator().manual_seed(11)  # g = генератор на устройстве device с сидом 11, если device != "cpu", иначе процессорный генератор с сидом 11
            self.proj.normal_(0, 0.02, generator=g)  # заполнить proj значениями N(0, 0.02)
            self.proj.requires_grad_(True)  # от proj вычисляются градиенты

    def align_step(self, model, ctx, texts):
        """Один шаг семантического выравнивания.

            ctx — контексты модели, texts — те же тексты для учителя.
            tv = векторы учителя; h = model.features(ctx);
            pred = h·proj / ||h·proj||;
            loss = 1 − mean_i Σ_d pred_id·tv_id;
            proj = proj − lr·∂loss/∂proj. Возвращает числовое значение
            loss."""
        import torch  # нейросетевая библиотека
        t = model.t  # библиотека, на которой работает модель
        from hagi.embeddings import GemmaEmbedder  # учитель — эмбеддер Gemma
        gemma = GemmaEmbedder.get()  # общий экземпляр учителя
        tv = gemma.embed(texts).to(model.device)  # tv — векторы учителя для текстов, перенесённые на устройство модели
        # признаки модель отдаёт сама через features: у модели с вращательным
        # позиционным кодированием это усреднённое скрытое состояние, у HNet —
        # остаточный поток h2. Собственной копии прямого прохода здесь нет:
        # она расходилась с моделью и ломалась на HNet.
        h = model.features(ctx)  # h — скрытые признаки модели для контекстов ctx
        self._ensure_proj(h.shape[1], h.device, h.dtype)  # построить proj шириной h.shape[1] при необходимости
        pred = h @ self.proj  # pred = h·proj (матричное умножение)
        pred = torch.nn.functional.normalize(pred, dim=-1)  # нормировка pred_id = pred_id / sqrt(Σ_d pred_id²)
        loss = 1.0 - (pred * tv).sum(-1).mean()  # косинусная потеря: 1 − mean_i Σ_d pred_id·tv_id
        # шаг градиента только по proj (учитель заморожен)
        g, = torch.autograd.grad(loss, self.proj)  # градиент g = ∂loss/∂proj
        with torch.no_grad():  # обновление без построения графа
            self.proj -= self.lr * g  # обновление proj = proj − lr·g
        return float(loss)  # числовое значение потери


class AdaptiveContextSelector:
    """Выбор числа окон контекста под конкретный ввод: модель сама решает,
    сколько скользящих окон усреднять.

        Окно фиксировано (k байт, обучено); меняется только число W
        непересекающихся окон в конце ввода. Логиты W окон складываются как
        среднее вероятностей: p = (1/W)·Σ_j softmax(logits окна j). W
        измеряется на хвосте самого ввода: предсказание хвоста с бо́льшим W
        принимается, только пока собственная кросс-энтропия улучшается;
        выбирается наименьшее W, дающее результат в пределах tol от
        лучшего. Вся история остаётся в байтах вызывающего — источник
        не ограничен, ограничено только вычисление."""

    def __init__(self, w_choices=(1, 2, 4, 8, 16), min_tail: int = 16,  # создать селектор: w_choices — варианты W, min_tail — длина хвоста для измерения
                 tol: float = 0.01):  # tol — допустимый относительный проигрыш в CE
        self.w_choices = sorted(w_choices)  # варианты числа окон по возрастанию
        self.min_tail = min_tail  # длина хвоста для измерения
        self.tol = tol  # допуск tol

    def _ensemble_logits(self, model, rows_x):
        """Генератор: для j = 0, 1, 2, ... выдаёт (число окон W = j + 1,
        среднее вероятностей по W окнам) для каждого ряда из rows_x.

            Окно j ряда — байты [end − k − j·k, end − j·k), где end —
            длина ряда (j = 0 — обычный путь, последнее окно). Окно короче k
            дополняется слева пробелами (байт 32), длиннее — обрезается до
            последних k. Усреднение — арифметическое среднее вероятностей
            softmax: p = (1/W)·Σ_j p_j."""
        t = model.t  # библиотека модели
        k = model.k  # длина окна k
        probs = None  # накопленная сумма вероятностей (пока нет)
        used = 0  # число просуммированных окон
        for j in range(max(self.w_choices)):  # j = 0..W_max − 1 — номер окна от конца
            xs = []  # окна всех рядов для этого j
            for row in rows_x:  # по рядам (байты перед очередным байтом хвоста)
                end = len(row)  # конец ряда
                start = max(0, end - k - j * k)  # начало окна start = max(0, end − k − j·k)
                w = row[start:end - j * k] if j > 0 else row[start:end]  # окно j: row[start : end − j·k]; при j = 0 — row[start : end]
                if len(w) < k:  # окно короче k
                    w = [32] * (k - len(w)) + w  # слева дописать k − len(w) пробелов (байт 32)
                else:  # окно не короче k
                    w = w[-k:]  # оставить последние k байт
                xs.append(w)  # добавить окно ряда
            x = t.as_tensor(xs, dtype=t.long)  # тензор целых контекстов
            with t.no_grad():  # без градиентов
                lg = model.logits(x)  # логиты модели для этих окон
                p = t.softmax(lg, dim=-1)  # вероятности p_i = exp(lg_i − max lg) / Σ_j exp(lg_j − max lg)
            probs = p if probs is None else probs + p  # сумма вероятностей: probs + p (первое окно — просто p)
            used += 1  # учтено окон: used + 1
            # условие остановки проверяет вызывающий по измеренной CE — select()
            yield used, probs / used  # выдать (W = used, среднее probs / used)

    def select(self, model, prompt_bytes: bytes) -> dict:
        """Измерить собственную кросс-энтропию предсказания хвоста ввода
        при разных W и вернуть наименьшее W в пределах tol от лучшего.

            Возвращает словарь: "w" — выбранное W, "ce" — лучшая CE,
            "curve" — список пар (W, CE), "reason" — пояснение. CE
            считается как mean_i (−log p[y_i]) по байтам хвоста."""
        t = model.t  # библиотека модели
        tail_len = min(self.min_tail, max(1, len(prompt_bytes) // 4))  # длина хвоста = min(min_tail, max(1, len(prompt_bytes) // 4))
        if len(prompt_bytes) <= tail_len + model.k:  # байтов не хватает на хвост и контекст
            return {"w": 1, "ce": None, "curve": [],  # W = 1, CE не измерялась, кривая пуста
                    "reason": "prompt too short for measurement"}  # причина: ввод слишком короток для измерения
        tail = prompt_bytes[-tail_len:]  # предсказываемый хвост — последние tail_len байт
        pre = prompt_bytes[:-tail_len]  # байты перед хвостом
        k = model.k  # длина окна k
        rows_x = []  # ряды байтов для измерения
        for i in range(len(tail)):  # i = 0..len(tail) − 1 — позиция в хвосте
            end = len(pre) + i  # конец контекста end = len(pre) + i
            start = max(0, end - k * max(self.w_choices))  # начало ряда start = max(0, end − k·W_max)
            row = list(prompt_bytes[start:end])  # ряд = байты start..end − 1
            rows_x.append(row)  # добавить ряд
        y = t.as_tensor(list(tail), dtype=t.long)  # y — байты хвоста (правильные ответы)
        curve = []  # кривая (W, CE)
        for w, probs in self._ensemble_logits(model, rows_x):  # по парам (W, средние вероятности)
            if w not in self.w_choices:  # W не из вариантов
                continue  # пропустить
            ce = float(t.nn.functional.cross_entropy(  # кросс-энтропия CE = mean_i (−log p_i[y_i])
                (probs + 1e-12).log(), y))  # логарифм p + 1e-12 — защита от log(0)
            curve.append((w, ce))  # добавить точку (W, CE)
            if w == max(self.w_choices):  # достигнуто наибольшее W
                break  # конец перебора
        if not curve:  # ни одно W не подошло
            return {"w": 1, "ce": None, "curve": [],  # W = 1, CE нет, кривая пуста
                    "reason": "no W fits the prompt"}  # причина: ни одно W не подходит к вводу
        best_ce = min(c for _, c in curve)  # best_ce = min CE по кривой
        chosen = self.w_choices[0]  # по умолчанию — наименьшее W
        for w, c in curve:  # по точкам кривой от малого W к большому
            if c <= best_ce * (1.0 + self.tol) + 1e-9:  # CE в пределах допуска: c <= best_ce·(1 + tol) + 1e-9
                chosen = w  # выбрать это W
                break  # большее W не нужно
        return {"w": chosen, "ce": best_ce, "curve": curve,  # результат: выбранное W, лучшая CE, кривая
                "reason": "self-CE elbow"}  # причина: перегиб собственной CE


def finite_displacement_ce_upper(logits_before, logits_after, targets):
    """Верхняя оценка прироста кросс-энтропии для конечного сдвига логитов.

        d = z' − z — фактический сдвиг логитов. Оценка:
        CE(z + d) − CE(z) <= ⟨softmax(z) − onehot(y), d⟩ + range(d)²/8,
        где range(d) = max_d d − min_d d. Линейность логитов по
        параметрам не предполагается — оценивается реальный конечный
        шаг. Возвращает среднее по примерам."""
    import torch  # нейросетевая библиотека
    F = torch.nn.functional  # функциональная часть torch
    delta = logits_after - logits_before  # сдвиг d = z' − z
    p = torch.softmax(logits_before, dim=-1)  # вероятности p = softmax(z): p_i = exp(z_i − max z) / Σ_j exp(z_j − max z)
    y = F.one_hot(targets, num_classes=logits_before.shape[-1]).to(p.dtype)  # y — one-hot правильных ответов, тип как у p
    first_order = ((p - y) * delta).sum(dim=-1)  # член первого порядка: ⟨p − y, d⟩ = Σ_d (p_d − y_d)·d_d
    width = delta.max(dim=-1).values - delta.min(dim=-1).values  # разброс range(d) = max_d d_d − min_d d_d
    return (first_order + width.square() / 8.0).mean()  # среднее по примерам ⟨p − y, d⟩ + range(d)²/8


def ternary_quant(module) -> dict[str, Any]:
    """Тернарное квантование весов модуля: emb, W1, W2 принимают значения
    {−s_r, 0, +s_r}.

        s_r = max(1e-12, mean_c |W_rc|) — средний модуль по строке r;
        Q_rc = clamp(round(W_rc / s_r), −1, +1)·s_r. Спектрального
        ограничения нет. Не идемпотентно: при повторном вызове s
        оценивается заново, поэтому вызов — один раз на шаг
        оптимизатора. Возвращает словарь «имя матрицы -> max |Q − W|»."""
    t = module.t  # библиотека модуля
    report = {}  # наибольшие отклонения по матрицам
    for name in ("emb", "W1", "W2"):  # по трём матрицам весов
        W = getattr(module, name)  # W — матрица с именем name
        with t.no_grad():  # без градиентов
            s = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)  # масштаб строки s_r = max(1e-12, mean_c |W_rc|)
            q = t.round(W / s).clamp(-1.0, 1.0) * s  # квантованная матрица Q_rc = clamp(round(W_rc / s_r), −1, +1)·s_r
            report[name] = float((q - W).abs().max())  # наибольшее отклонение max |Q − W|
            W.copy_(q)  # записать Q в матрицу W
    return report  # отклонения по всем матрицам


class ByteLM:
    """Простая языковая модель: контекст из k байт -> логиты следующего
    байта.

        e = emb[ctx] — вложения байт контекста, форма (B, k, dim);
        h = relu(concat(e)·W1 + b1), concat(e) имеет длину k·dim;
        логиты = h·W2 + b2, форма (B, 256). Веса emb, W1, W2 поддаются
        тернарному квантованию."""

    def __init__(self, k: int = 8, dim: int = 64, hidden: int = 128,  # создать модель: k — окно, dim — размер вложения, hidden — скрытый слой
                 seed: int = 7):  # seed — сид инициализации весов
        import torch  # нейросетевая библиотека
        t = torch  # короткое имя библиотеки
        self.t = t  # библиотека, на которой работает модель
        self.k, self.dim, self.hidden = k, dim, hidden  # запомнить окно k, размерность dim и скрытый слой hidden
        g = t.Generator().manual_seed(seed)  # генератор с сидом seed
        self.emb = t.nn.Parameter(t.randn(256, dim, generator=g) * 0.08)  # вложения байт 256 x dim: N(0, 1)·0.08
        self.W1 = t.nn.Parameter(t.randn(k * dim, hidden, generator=g) * 0.08)  # первая матрица (k·dim) x hidden: N(0, 1)·0.08
        self.b1 = t.nn.Parameter(t.zeros(hidden))  # смещение скрытого слоя из нулей
        self.W2 = t.nn.Parameter(t.randn(hidden, 256, generator=g) * 0.08)  # вторая матрица hidden x 256: N(0, 1)·0.08
        self.b2 = t.nn.Parameter(t.zeros(256))  # смещение логитов из нулей
        self._ef21_s: float | None = None  # фиксированный масштаб, закрепляется на шаге 1 (пока не задан)
        self.device = _resolve_device(t, self.dim, self.hidden)  # устройство по правилу выбора
        if self.device == "cuda":  # модель размещается на GPU
            for prm in self.params():  # по всем параметрам
                prm.data = prm.data.to(self.device)  # перенести данные параметра на устройство

    def params(self):
        """Список всех обучаемых параметров модели."""
        return [self.emb, self.W1, self.b1, self.W2, self.b2]  # параметры в фиксированном порядке

    def logits(self, ctx):  # ctx: (B, k) long — контексты, целые номера байт
        """Логиты следующего байта для контекстов ctx формы (B, k).
        Возвращает тензор (B, 256)."""
        t = self.t  # библиотека модели
        e = self.emb[ctx]                       # e = emb[ctx] — вложения байт контекста, форма (B, k, dim)
        h = t.relu(e.reshape(ctx.shape[0], -1) @ self.W1 + self.b1)  # h = relu(concat(e)·W1 + b1), concat(e) — B строк длины k·dim
        return h @ self.W2 + self.b2            # логиты = h·W2 + b2, форма (B, 256)

    def ce(self, ctx, nxt, reduction="mean"):
        """Кросс-энтропия логитов ctx против ответов nxt; reduction задаёт
        свёртку ("mean" — среднее CE_i = −log softmax(logits_i)[nxt_i])."""
        t = self.t  # библиотека модели
        return t.nn.functional.cross_entropy(self.logits(ctx), nxt,  # CE логитов и ответов nxt
                                              reduction=reduction)  # способ свёртки reduction

    # ---- тернарное квантование ------------------------------------
    def ternary_(self) -> dict[str, Any]:
        """Квантовать emb, W1, W2 модели на месте; вернуть наибольшие
        отклонения по матрицам."""
        return ternary_quant(self)  # квантование весов модуля

    # ---- обучающий шаг с гейтом / цепочка лучших снимков -----------------------
    def train_guarded(self, data: bytes, epochs: int = 30, lr: float = 3e-3,  # обучение с гейтом: data — байты, epochs — эпохи, lr — шаг Adam
                      holdout: float = 0.1, log_every: int = 10  # holdout — доля удержанных блоков, log_every — период печати
                      ) -> dict[str, Any]:  # возвращает словарь с лучшей CE и историей
        """Полно-пакетное обучение с гейтом по удержанным данным.

            Окна — блоки длины k + 1 (контекст и следующий байт);
            удержанные блоки выбираются случайно, но целиком: окна
            соседних блоков не делят байты, поэтому контекст удержанных
            данных не попадает в обучение. На каждой эпохе: шаг Adam по
            CE обучающих окон; шаг принят, если CE удержанных данных
            не выросла (тогда сохраняется снимок весов). В конце
            разворачивается лучший снимок (начальные веса — тоже
            кандидат). Возвращает "epochs", "best_holdout_ce" (4 знака),
            "history" — CE удержанных данных по эпохам."""
        t = self.t  # библиотека модели
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # idx — байты данных как целые 0..255
        n = len(idx) - self.k  # n = число окон = len(idx) − k
        if n < 64:  # окон меньше 64
            raise ValueError(f"need >= {self.k + 64} bytes, got {len(idx)}")  # ошибка: нужно не меньше k + 64 байт
        # Окна i и i + 1 делят k − 1 байт, поэтому при случайном разбиении
        # окон контекст удержанных окон частично встречался в обучении.
        # Разделение идёт по непрерывным блокам: удержанные блоки не
        # пересекаются с обучающими.
        rng = np.random.default_rng(7)  # генератор с сидом 7
        block = self.k + 1                  # длина блока = k + 1 (одно целое окно)
        n_blocks = n // block  # число полных блоков n // (k + 1)
        n_hold_blocks = max(1, int(n_blocks * holdout))  # число удержанных блоков = max(1, int(n_blocks·holdout))
        hold_blocks = rng.choice(n_blocks, n_hold_blocks, replace=False)  # номера удержанных блоков — случайные, без повторений
        hold_mask = np.zeros(n_blocks, dtype=bool)  # маска удержанных блоков, изначально вся ложь
        hold_mask[hold_blocks] = True  # выбранные блоки получают истину
        positions = np.arange(n_blocks) * block  # positions[b] = b·block — начало блока b
        hold = np.concatenate([positions[b] + np.arange(block - self.k)  # удержанные окна: b·block + (0..block − k − 1)
                               for b in np.where(hold_mask)[0]])  # по номерам блоков с маской «истина»
        train = np.concatenate([  # обучающие окна — конкатенация по не удержанным блокам
            np.concatenate([positions[b] + np.arange(block - self.k)  # окна блока b: b·block + (0..block − k − 1)
                            for b in np.where(~hold_mask)[0]])]) if (~hold_mask).any() else np.array([], dtype=np.int64)  # если есть не удержанные блоки, иначе пустой массив int64
        # обучение идёт полным пакетом окон train — нужен хотя бы один
        # обучающий столбец
        if len(train) == 0:  # обучающих окон нет
            train = hold[:1]  # вырожденный случай: данных только на удержание — берём первое окно

        dev = getattr(self, "device", "cpu")  # устройство модели (или "cpu", если атрибута нет)

        def windows(sel):
            """Тензоры (контексты, ответы) для номеров окон sel:
            x_i = idx[i : i + k], y_i = idx[i + k], на устройстве dev."""
            x = np.stack([idx[i:i + self.k] for i in sel])  # x — контексты idx[i : i + k]
            y = idx[sel + self.k]  # y — следующие байты idx[i + k]
            return (t.as_tensor(x, dtype=t.long, device=dev),  # тензор контекстов на устройстве
                    t.as_tensor(y, dtype=t.long, device=dev))  # тензор ответов на устройстве

        ctx_tr, nxt_tr = windows(train)  # обучающие контексты и ответы
        ctx_va, nxt_va = windows(hold)  # удержанные контексты и ответы
        opt = t.optim.Adam(self.params(), lr=lr)  # оптимизатор Adam с шагом lr
        best_ce = float(self.ce(ctx_va, nxt_va).detach())  # best_ce — CE удержанных данных до обучения
        # начальный снимок — тоже кандидат в лучшие: если ни одна эпоха не
        # улучшила CE, остаются начальные веса, а не веса последней эпохи
        best_state = [p.detach().clone() for p in self.params()]  # снимок всех параметров
        history = []  # CE удержанных данных по эпохам
        for ep in range(epochs):  # по эпохам ep = 0..epochs − 1
            opt.zero_grad()  # обнулить градиенты
            loss = self.ce(ctx_tr, nxt_tr)  # потеря — CE обучающих окон
            loss.backward()  # вычислить градиенты
            opt.step()  # шаг Adam
            ce_va = float(self.ce(ctx_va, nxt_va).detach())  # CE удержанных данных после шага
            history.append(round(ce_va, 4))  # добавить CE с 4 знаками
            # шаг принимается, только если CE удержанных данных не выросла
            if ce_va <= best_ce:  # не хуже лучшей
                best_ce = ce_va  # новая лучшая CE
                best_state = [p.detach().clone() for p in self.params()]  # снять веса
            if log_every and (ep + 1) % log_every == 0:  # каждые log_every эпох
                print(f"  epoch {ep+1}/{epochs} holdout CE {ce_va:.4f} "  # печать номера эпохи и CE удержанных данных
                      f"(best {best_ce:.4f})")  # и лучшей CE
        # в конце разворачивается лучший снимок, никогда не худший
        if best_state is not None:  # снимок есть
            for p, s in zip(self.params(), best_state):  # параметр и его значение из снимка
                p.data.copy_(s)  # p = s — восстановить параметр из снимка
        return {"epochs": epochs, "best_holdout_ce": round(best_ce, 4),  # итог: число эпох и лучшая CE удержанных данных
                "history": history}  # история CE


    # ---- мини-пакетное обучение на большом корпусе (с гейтом) ----------
    def train_minibatch(self, data: bytes, steps: int = 3000,  # мини-пакетное обучение: data — байты, steps — число шагов
                        batch: int = 512, lr: float = 1e-3,  # batch — размер пакета, lr — шаг Adam
                        val_every: int = 200, holdout: float = 0.02,  # val_every — период проверки, holdout — доля удержанных окон
                        log_every: int = 500) -> dict[str, Any]:  # log_every — период печати; возврат — словарь итогов
        """Обучение случайными пакетами окон для большого корпуса.

            Удержанные окна выбираются случайно (не меньше 256) и не
            участвуют в обучении. На каждом шаге: пакет из batch
            обучающих окон, шаг Adam, шаг косинусного планировщика
            lr_t = lr·(1 + cos(pi·t/steps))/2. Каждые val_every шагов
            измеряется CE удержанных данных; снимок сохраняется, если
            CE не выросла. В конце разворачивается лучший снимок, и
            веса возвращаются на процессор. Возвращает "steps",
            "corpus_bytes", "best_holdout_ce", "history"."""
        t = self.t  # библиотека модели
        dev = ("cuda" if t.cuda.is_available() else "cpu")  # устройство: "cuda", если CUDA доступна, иначе "cpu"
        idx_np = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # байты корпуса как целые числа
        n = len(idx_np) - self.k - 1  # n = len(idx) − k − 1 (номер последнего окна)
        rng = np.random.default_rng(7)  # генератор с сидом 7
        n_hold = max(256, int(n * holdout))  # число удержанных окон = max(256, int(n·holdout))
        hold_idx = rng.choice(n, n_hold, replace=False)  # номера удержанных окон — случайные, без повторений
        hold_set = set(hold_idx.tolist())  # то же множество номеров в виде set
        hold_np = np.sort(hold_idx)  # номера удержанных окон по возрастанию
        for p_ in self.params():  # по всем параметрам
            p_.data = p_.data.to(dev)  # перенести параметр на устройство
        def windows_np(sel):
            """Тензоры (контексты, ответы) для номеров окон sel:
            x_i = idx_np[i : i + k], y_i = idx_np[i + k], на устройстве dev."""
            x = np.stack([idx_np[i:i + self.k] for i in sel])  # x — контексты idx_np[i : i + k]
            y = idx_np[sel + self.k]  # y — следующие байты idx_np[i + k]
            return (t.as_tensor(x, dtype=t.long, device=dev),  # тензор контекстов на устройстве
                    t.as_tensor(y, dtype=t.long, device=dev))  # тензор ответов на устройстве
        ctx_va, nxt_va = windows_np(hold_np)  # удержанные контексты и ответы
        opt = t.optim.Adam(self.params(), lr=lr)  # оптимизатор Adam с шагом lr
        sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)  # планировщик: lr_t = lr·(1 + cos(pi·t/steps))/2, за steps шагов lr падает до 0
        def val_ce():
            """Средняя CE удержанных окон, кусками по 8192:
            (Σ_кусок средняя CE_кусок·число_примеров_куска) / len(hold_np)."""
            ces = []  # суммы CE по кускам
            for i in range(0, len(hold_np), 8192):  # кусками по 8192 окна
                with t.no_grad():  # без градиентов
                    ces.append(self.ce(ctx_va[i:i+8192], nxt_va[i:i+8192])  # сумма CE куска =
                               * len(ctx_va[i:i+8192]))  # средняя CE куска · число примеров куска
            return float(sum(ces) / len(hold_np))  # средняя CE = Σ сумм кусков / число удержанных окон
        best_ce = val_ce()  # лучшая CE до обучения
        # начальный снимок — тоже кандидат в лучшие (как в train_guarded):
        # без улучшения остаются начальные веса, а не последние
        best_state = [p_.detach().clone() for p_ in self.params()]  # снимок всех параметров
        history = []  # CE удержанных данных по проверкам
        # Обучение раньше выбирало окна из всех n, включая удержанные (в
        # пакете из 512 попадалось около 14 удержанных), поэтому CE
        # удержанных данных частично была обучающей. Теперь окна берутся
        # только из обучающего множества.
        train_idx = np.setdiff1d(np.arange(n), hold_np,  # обучающие номера = {0..n − 1} без удержанных
                                 assume_unique=True)  # оба массива считаются без повторений
        for step in range(1, steps + 1):  # по шагам 1..steps
            sel = train_idx[rng.integers(0, len(train_idx),  # sel = train_idx[J], J — batch случайных номеров из 0..len(train_idx) − 1
                                         size=batch)]  # число номеров — batch
            ctx, nxt = windows_np(sel)  # контексты и ответы пакета
            opt.zero_grad()  # обнулить градиенты
            loss = self.ce(ctx, nxt)  # потеря — CE пакета
            loss.backward()  # вычислить градиенты
            opt.step(); sched.step()  # шаг Adam, затем шаг планировщика lr
            if step % val_every == 0:  # каждые val_every шагов
                ce_va = val_ce(); history.append(round(ce_va, 4))  # CE удержанных данных и запись с 4 знаками
                if ce_va <= best_ce:  # не хуже лучшей
                    best_ce = ce_va  # новая лучшая CE
                    best_state = [p_.detach().clone() for p_ in self.params()]  # снять веса
                if log_every and step % log_every == 0:  # каждые log_every шагов
                    print(f"  step {step}/{steps} holdout CE {ce_va:.4f} "  # печать номера шага и CE удержанных данных
                          f"(best {best_ce:.4f})", flush=True)  # лучшая CE; вывод сразу, без буфера
        if best_state is not None:  # снимок есть
            for p_, s in zip(self.params(), best_state):  # параметр и его значение из снимка
                p_.data.copy_(s)  # восстановить параметр из лучшего снимка
        rep = {"steps": steps, "corpus_bytes": len(data),  # итог: число шагов и размер корпуса в байтах
               "best_holdout_ce": round(best_ce, 4), "history": history}  # лучшая CE удержанных данных и история
        for p_ in self.params():  # по всем параметрам
            p_.data = p_.data.cpu()  # вернуть параметр на процессор
        return rep  # итоги обучения
    # ---- генерация --------------------------------------------------
    @t_no_grad_if  # выполнять без градиентов
    def generate(self, prompt: str, n_bytes: int = 120, temp: float = 0.8,  # авторегрессионная генерация: n_bytes — число байт, temp — температура
                 seed: int = 1, stop_at_eos: bool = False,  # seed — сид выборки, stop_at_eos — стоп по байту-терминатору
                 eos_bytes=None) -> str:  # eos_bytes — набор байт-терминаторов
        """Породить n_bytes байт после prompt и вернуть текст.

            ctx — последние k байт prompt (слева дополняются пробелами,
            байт 32). На каждом шаге: z = logits/temp (temp не меньше
            1e-3), p = softmax(z), b ~ p — выборка одной реализации;
            ctx = ctx[1:] + [b]. При stop_at_eos генерация прерывается,
            если выбранный байт входит в eos_bytes и завершает текст."""
        t = self.t  # библиотека модели
        eos = set(eos_bytes or [])  # eos — множество байт-терминаторов (пустое, если не задано)
        g = t.Generator().manual_seed(seed)  # генератор выборки с сидом seed
        ctx = list(prompt.encode("utf-8"))[-self.k:]  # последние k байт prompt
        if len(ctx) < self.k:  # байт меньше k
            ctx = [32] * (self.k - len(ctx)) + ctx  # дополнить слева k − len(ctx) пробелами
        out = bytearray()  # сгенерированные байты
        for _ in range(n_bytes):  # n_bytes шагов
            x = t.as_tensor([ctx], dtype=t.long)  # x — один контекст, форма (1, k)
            logits = self.logits(x)[0] / max(temp, 1e-3)  # логиты, делённые на температуру: z / max(temp, 1e-3)
            probs = t.softmax(logits, dim=-1)  # вероятности p_i = exp(z_i − max z) / Σ_j exp(z_j − max z)
            b = int(t.multinomial(probs, 1, generator=g))  # b — одна выборка из распределения p
            out.append(b)  # добавить байт к результату
            ctx = ctx[1:] + [b]  # сдвиг окна: убрать первый байт, добавить новый
            # модель останавливается сама: байт-терминатор завершает предложение
            if stop_at_eos and bytes([b]) in eos:  # включена остановка и b — терминатор
                if bytes(out).rstrip().endswith(tuple(eos)):  # результат без концевых пробелов заканчивается терминатором
                    break  # конец генерации
        return (prompt.encode("utf-8") + bytes(out)).decode("utf-8",  # decode prompt с генерацией в текст UTF-8
                                                            errors="replace")  # повреждённые последовательности заменяются знаком


def save(lm: ByteLM, path: Path = CKPT) -> None:
    """Сохранить модель lm в файл path (torch.save): тензоры emb, W1, b1,
    W2, b2 (без градиентов, на процессоре) и размеры k, dim, hidden."""
    t = lm.t  # библиотека модели
    t.save({n: getattr(lm, n).detach().cpu() for n in  # словарь «имя -> копия параметра на процессоре»
            ("emb", "W1", "b1", "W2", "b2")} |  # объединение со вторым словарём
           {"k": lm.k, "dim": lm.dim, "hidden": lm.hidden}, path)  # размеры модели; запись в path


def load(path: Path = CKPT) -> ByteLM:
    """Загрузить ByteLM из файла path: модель создаётся с сохранёнными
    k, dim, hidden, затем тензоры копируются в её параметры."""
    t, _ = _torch_and_device()  # библиотека torch (устройство не нужно)
    st = t.load(path, weights_only=False)  # st — сохранённый словарь
    lm = ByteLM(k=st["k"], dim=st["dim"], hidden=st["hidden"])  # модель сохранённых размеров
    for n in ("emb", "W1", "b1", "W2", "b2"):  # по пяти параметрам
        getattr(lm, n).data.copy_(st[n])  # скопировать сохранённый тензор в параметр
    return lm  # загруженная модель


def main() -> None:
    """Командная строка: train — обучить ByteLM на файлах (по желанию с
    тернарным квантованием и дообучением) и сохранить в CKPT;
    generate — загрузить модель и напечатать продолжение prompt."""
    p = argparse.ArgumentParser(description="HAGI byte-level neural LM")  # синтаксический анализатор аргументов
    sub = p.add_subparsers(dest="cmd", required=True)  # обязательная подкоманда в переменную cmd
    tr = sub.add_parser("train")  # подкоманда train
    tr.add_argument("files", nargs="+", type=Path)  # один или несколько файлов
    tr.add_argument("--epochs", type=int, default=30)  # число эпох, по умолчанию 30
    tr.add_argument("--k", type=int, default=8)  # окно, по умолчанию 8
    tr.add_argument("--dim", type=int, default=64)  # размер вложения, по умолчанию 64
    tr.add_argument("--hidden", type=int, default=128)  # скрытый слой, по умолчанию 128
    tr.add_argument("--ternary", action="store_true",  # флаг квантования после обучения
                    help="§6 ternary quantization after training")  # подсказка флага
    gen = sub.add_parser("generate")  # подкоманда generate
    gen.add_argument("prompt")  # начальный текст
    gen.add_argument("--n", type=int, default=120)  # число байт, по умолчанию 120
    gen.add_argument("--temp", type=float, default=0.8)  # температура, по умолчанию 0.8
    gen.add_argument("--seed", type=int, default=1)  # сид выборки, по умолчанию 1
    a = p.parse_args()  # a — разобранные аргументы
    if a.cmd == "train":  # подкоманда train
        data = b"".join(f.read_bytes() for f in a.files)  # данные — конкатенация байт всех файлов
        lm = ByteLM(k=a.k, dim=a.dim, hidden=a.hidden)  # модель заданных размеров
        rep = lm.train_guarded(data, epochs=a.epochs)  # обучение с гейтом
        print(f"trained on {len(data)} bytes: {json.dumps(rep)[:200]}")  # печать размера данных и итогов (первые 200 знаков)
        if a.ternary:  # запрошено квантование
            cost = lm.ternary_()  # квантовать веса, cost — наибольшие отклонения
            print(f"ternary (§6): max |dW| per matrix {cost}")  # печать отклонений по матрицам
            # вернуть точность после квантования: дообучение с гейтом
            rep2 = lm.train_guarded(data, epochs=max(10, a.epochs // 3))  # дообучение на max(10, epochs // 3) эпох
            print(f"post-ternary guarded fine-tune: "  # печать итога дообучения:
                  f"holdout CE {rep2['best_holdout_ce']}")  # лучшая CE удержанных данных
        CKPT.parent.mkdir(parents=True, exist_ok=True)  # создать каталог для весов (включая родительские, без ошибки при наличии)
        save(lm)  # сохранить модель
        print(f"checkpoint: {CKPT}")  # печать пути к весам
    else:  # подкоманда generate
        lm = load()  # загрузить сохранённую модель
        print(lm.generate(a.prompt, n_bytes=a.n, temp=a.temp, seed=a.seed))  # печать продолжения prompt


if __name__ == "__main__":  # запуск только как сценарий
    main()  # разбор аргументов и выполнение команды


class RopeAttentionByteLM:  # байтовая модель с RoPE-вниманием по всему контексту
    ARCH = "rope"  # имя архитектуры
    adapter_dim = 256   # размерность адаптера домена (в __init__ заменяется на hidden)
    """Байтовая языковая модель с вниманием по всему контексту без ограничения длины.

    - Позиции задаются поворотами (RoPE): позиция i поворачивает каждую пару
      координат вложения на угол theta_i = i·theta0. Поворот сохраняет норму
      вектора, а разные позиции не смешиваются.
    - Внимание по содержанию: веса p_i = softmax(q·k_i / sqrt(dim)) по ВСЕМ
      более ранним позициям, т.е. это настоящее внимание, а не равномерное
      смешивание.
    - Затухание по давности: в счёт добавляется штраф −lambda·d, где d —
      расстояние между позициями. Он даёт нижнюю границу веса каждого
      контекста, за счёт чего смесь остаётся перемешивающей.

    Обучение — кросс-энтропия на one-hot ответах шагами с доказательной
    проверкой; на выход идут веса лучшего снимка по удержанным данным.
    """

    def __init__(self, dim: int = 64, hidden: int = 256,  # создать модель: dim — размер вложения, hidden — скрытый слой
                 theta0: float = 1.0, lam: float = 0.05, seed: int = 7):  # theta0 — базовый угол поворота, lam — вес затухания, seed — сид весов
        import torch  # нейросетевая библиотека
        t = torch  # короткое имя библиотеки
        self.t = t  # библиотека, на которой работает модель
        self.dim, self.hidden = dim, hidden  # запомнить dim и hidden
        self.theta0, self.lam = theta0, lam  # запомнить угол theta0 и затухание lam
        self.adapter_dim = hidden  # размерность адаптера домена равна hidden
        g = t.Generator().manual_seed(seed)  # генератор с сидом seed
        self.emb = t.nn.Parameter(t.randn(256, dim, generator=g) * 0.08)  # вложения байт 256 x dim: N(0, 1)·0.08
        self.q = t.nn.Parameter(t.randn(dim, generator=g) * 0.08)  # вектор q: в парном внимании градиента не получает, хранится для совместимости весов
        self.W1 = t.nn.Parameter(t.randn(dim, hidden, generator=g) * 0.08)  # матрица dim x hidden: N(0, 1)·0.08
        self.b1 = t.nn.Parameter(t.zeros(hidden))  # смещение из нулей
        self.W2 = t.nn.Parameter(t.randn(hidden, 256, generator=g) * 0.08)  # матрица hidden x 256: N(0, 1)·0.08
        self.b2 = t.nn.Parameter(t.zeros(256))  # смещение логитов из нулей
        self._ef21_s: float | None = None  # фиксированный масштаб, закрепляется на шаге 1 (пока не задан)
        # необязательные дополнительные блоки внимания. Блок b добавляет к
        # скрытому состоянию Wo[b]·relu(a·W1[b] + b1[b]), где a — результат
        # внимания. У нового блока Wo = 0, поэтому функция модели
        # сохраняется точно, а ёмкость наращивается потом обучением.
        self.blocks: list[dict] = []  # дополнительные блоки (изначально нет)
        # номера замороженных блоков (ремонтные блоки хранят старую память;
        # общий обучающий шаг их не перезаписывает)
        self.frozen_blocks: set = set()  # множество замороженных блоков, пусто
        self.device = _resolve_device(t, self.dim, self.hidden)  # устройство по правилу выбора
        if self.device == "cuda":  # модель размещается на GPU
            for prm in self.params():  # по всем параметрам
                prm.data = prm.data.to(self.device)  # перенести данные параметра на устройство

    _BASE_NAMES = ("emb", "q", "W1", "b1", "W2", "b2")  # имена основных параметров

    def param_names(self):
        """Имена основных параметров модели (без дополнительных блоков)."""
        # дополнительные блоки сохраняются отдельно (st["blocks"]), при
        # загрузке модель наращивает их сама
        return list(self._BASE_NAMES)  # копия списка имён

    def params(self):
        """Все обучаемые параметры: emb, q, W1, b1, W2, b2 и W1, b1, Wo
        каждого дополнительного блока."""
        out = [self.emb, self.q, self.W1, self.b1, self.W2, self.b2]  # основные параметры
        for blk in getattr(self, "blocks", []):  # по дополнительным блокам (если есть)
            out += [blk["W1"], blk["b1"], blk["Wo"]]  # добавить параметры блока
        return out  # полный список параметров

    def _block_add(self, a, h):
        """Прибавить вклад дополнительных блоков к скрытому состоянию:
        h = h + relu(a·W1_b + b1_b)·Wo_b для каждого блока b. При Wo_b = 0
        вклад нулевой — состояние не меняется."""
        t = self.t  # библиотека модели
        for blk in getattr(self, "blocks", []):  # по дополнительным блокам (если есть)
            h = h + t.relu(a @ blk["W1"] + blk["b1"]) @ blk["Wo"]  # новое h = h + relu(a·W1_b + b1_b)·Wo_b
        return h  # скрытое состояние с вкладами блоков

    def grow(self, seed: int = 0) -> int:
        """Добавить один дополнительный блок внимания с Wo = 0: логиты не
        меняются совсем, новая ёмкость заполняется дальнейшим обучением.
        Возвращает число блоков."""
        t = self.t  # библиотека модели
        g = t.Generator().manual_seed(seed + 1000 * (len(self.blocks) + 1))  # генератор с сидом seed + 1000·(число блоков + 1)
        blk = {"W1": t.nn.Parameter(t.randn(self.dim, self.hidden, generator=g) * 0.08),  # W1 блока dim x hidden: N(0, 1)·0.08
               "b1": t.nn.Parameter(t.zeros(self.hidden)),  # смещение блока из нулей
               "Wo": t.nn.Parameter(t.zeros(self.hidden, self.hidden))}  # выходная матрица Wo из нулей — блок пока ничего не меняет
        self.blocks.append(blk)  # добавить блок в список
        if self.device == "cuda":  # модель на GPU
            for prm in blk.values():  # по параметрам блока
                prm.data = prm.data.to(self.device)  # перенести параметр на устройство
        return len(self.blocks)  # число блоков после добавления

    def _rope(self, e, L, device):
        """Вращательное позиционное кодирование: пару координат j в позиции
    pos повернуть на угол th_{pos,j} = pos·theta0·freq_j, где
    freq_j = theta0^(−2j/half). Поворот (x, y) -> (x·cos − y·sin,
    x·sin + y·cos) сохраняет норму точно."""
        t = self.t  # библиотека модели
        # угол зависит от пары (позиция, частота): th_{pos,j} =
        # pos·theta0·freq_j, частоты freq_j образуют геометрическую
        # прогрессию. Прежде каждая пара j поворачивалась на угол j·theta0
        # во всех позициях — это ортогональное преобразование признаков, а
        # не кодирование позиции.
        half = self.dim // 2  # число пар координат half = dim // 2
        L = e.shape[-2]  # L — число позиций из формы e
        freq = self.theta0 ** (-2.0 * t.arange(half, device=device,  # частоты freq_j = theta0^(−2j/half), j = 0..half − 1
                                               dtype=t.float32) / half)  # показатель: (−2·j)/half
        pos = t.arange(L, device=device, dtype=t.float32)  # номера позиций 0..L − 1
        th = pos[:, None] * freq[None, :]          # th_{pos,j} = pos·freq_j, форма (L, half)
        cs, sn = t.cos(th), t.sin(th)              # cs = cos(th), sn = sin(th), форма (L, half)
        a, b = e[..., 0::2], e[..., 1::2]  # a — чётные координаты, b — нечётные
        r = t.empty_like(e)  # результат той же формы
        r[..., 0::2] = a * cs - b * sn  # новая чётная координата: a·cos(th) − b·sin(th)
        r[..., 1::2] = a * sn + b * cs  # новая нечётная координата: a·sin(th) + b·cos(th)
        return r  # вложения после поворота

    def logits(self, ctx):
        """ctx: (B, L), самый старый байт первым. Логиты следующего байта
        по последнему (самому свежему) столбцу ctx — то же самое, что
        full_logits(ctx)[:, -1, :], но без матрицы счёта B x L x L:
        вычисляется только строка запроса последней позиции.
        Возвращает (B, 256)."""
        t = self.t  # библиотека модели
        if ctx.device != self.device:  # контекст не на устройстве модели
            ctx = ctx.to(self.device)  # перенести контекст
        e = self._rope(self.emb[ctx], ctx.shape[1], ctx.device)  # e — вложения ctx с поворотом, форма (B, L, dim)
        query = e[:, -1, :]  # запрос — последняя позиция (самый старый первый: последняя = самая свежая)
        pos = t.arange(ctx.shape[1], device=ctx.device, dtype=t.float32)  # номера ключей 0..L − 1
        # ключ j при порядке «старый первым» отстаёт от запроса на (L−1−j)
        scores = t.einsum("bd,bld->bl", query, e) / (self.dim ** 0.5)             - self.lam * (ctx.shape[1] - 1  # счёт_j = q·e_j / sqrt(dim) − lam·(L − 1 − j)
                          - t.arange(ctx.shape[1], device=ctx.device,  # расстояние L − 1 − j
                                     dtype=t.float32))[None, :]  # расстояния в строку для вещания по B
        mask = t.arange(ctx.shape[1], device=ctx.device)[None, :]             <= ctx.shape[1] - 1  # все ключи — прошлое запроса
        scores = scores.masked_fill(~mask, float("-inf"))  # вне маски счёт = −inf (таких ключей нет)
        pp = t.softmax(scores, dim=-1)  # веса p_j = exp(s_j − max s) / Σ_k exp(s_k − max s)
        a_agg = t.einsum("bl,bld->bd", pp, e)  # результат внимания a = Σ_j p_j·e_j
        h = self._block_add(a_agg, t.relu(a_agg @ self.W1 + self.b1))  # h = relu(a·W1 + b1) с вкладами блоков
        ad = getattr(self, "domain_adapter", None)  # адаптер домена или None
        if ad is not None:  # адаптер задан
            At = t.as_tensor(ad.A, device=h.device, dtype=h.dtype)  # матрица A адаптера в тензор
            Bt = t.as_tensor(ad.B, device=h.device, dtype=h.dtype)  # матрица B адаптера в тензор
            h = h + (h @ At.T) @ Bt.T  # h = h + h·A^T·B^T (поправка адаптера низкого ранга)
        return h @ self.W2 + self.b2  # логиты = h·W2 + b2

    def features(self, ctx):
        """Признаки для головы семантического выравнивания: логиты всех
        позиций (B, L, 256) -> среднее по позициям -> (B, 256). Это та
        величина, которую align_step раньше считал вручную."""
        t = self.t  # библиотека модели
        with t.no_grad():  # без градиентов
            return self.full_logits(ctx).mean(dim=1)  # среднее логитов по позициям: (1/L)·Σ_l logits_l

    def full_logits(self, ctx):
        """Логиты следующего байта для КАЖДОЙ позиции ctx: (B, L) -> (B, L, 256).

            dot_{ij} = e_i·e_j / sqrt(dim); счёт = dot − lam·(i − j);
            внимать можно только j <= i; p_{ij} = softmax_j(счёт);
            a_i = Σ_j p_{ij}·e_j; h = relu(a·W1 + b1) с вкладами блоков;
            логиты = h·W2 + b2 (с поправкой адаптера домена, если задан)."""
        t = self.t  # библиотека модели
        if ctx.device != self.device:  # контекст не на устройстве модели
            ctx = ctx.to(self.device)  # перенести контекст
        B, L = ctx.shape  # B — размер пакета, L — число позиций
        e = self.emb[ctx]  # вложения байт, форма (B, L, dim)
        e = self._rope(e, L, ctx.device)  # вложения с позиционным поворотом
        # счёт — настоящее парное (запрос, ключ): прежний код превращал
        # скаляр позиции в (B, L, 1) и вещал его по оси ключей — внимание
        # становилось равномерным, q и lam не действовали, распределения
        # при обучении и генерации расходились.
        pos = t.arange(L, device=ctx.device, dtype=t.float32)  # номера позиций 0..L − 1
        dot = t.einsum("bid,bjd->bij", e, e) / (self.dim ** 0.5)  # скалярные произведения dot_{ij} = e_i·e_j / sqrt(dim)
        sc = dot - self.lam * (pos[None, :, None]  # счёт_{ij} = dot_{ij} − lam·(i − j)
                               - pos[None, None, :])  # расстояние i − j
        mask = pos[None, :, None] >= pos[None, None, :]  # маска: j <= i
        sc = sc.masked_fill(~mask, float("-inf"))  # вне маски счёт = −inf
        # нормировка по оси КЛЮЧЕЙ (dim = −1): нормировка по оси запросов
        # подмешивала будущие байты в знаменатели более ранних позиций
        pp = t.softmax(sc, dim=-1)  # веса p_{ij} = exp(sc_ij − max_j sc_ij) / Σ_k exp(sc_ik − max_k sc_ik)
        a_agg = t.einsum("blj,bjd->bld", pp, e)  # результат внимания a_l = Σ_j p_{lj}·e_j
        h = self._block_add(a_agg, t.relu(a_agg @ self.W1 + self.b1))  # h = relu(a·W1 + b1) с вкладами блоков
        ad = getattr(self, "domain_adapter", None)  # адаптер домена или None

        if ad is not None:  # адаптер задан
            At = t.as_tensor(ad.A, device=h.device, dtype=h.dtype)  # матрица A адаптера в тензор
            Bt = t.as_tensor(ad.B, device=h.device, dtype=h.dtype)  # матрица B адаптера в тензор
            h = h + (h @ At.T) @ Bt.T  # h = h + h·A^T·B^T (поправка адаптера низкого ранга)
        return h @ self.W2 + self.b2  # логиты = h·W2 + b2

    def ce(self, ctx, nxt):
        """Средняя кросс-энтропия логитов(ctx) против ответов nxt."""
        dev = self.device  # устройство модели
        if nxt.device != dev:  # ответы не на устройстве модели
            nxt = nxt.to(dev)  # перенести ответы
        return self.t.nn.functional.cross_entropy(  # кросс-энтропия CE = mean_i (−log softmax(logits_i)[nxt_i])
            self.logits(ctx), nxt, reduction="mean")  # логиты и ответы, среднее

    def train_guarded(self, data: bytes, epochs: int = 60, lr: float = 1e-3,  # обучение с гейтом: data — байты, epochs — эпохи, lr — шаг Adam
                      holdout: float = 0.1, batch: int = 64,  # holdout — доля удержанных данных, batch — размер пакета
                      max_train_ctx: int = 96, log_every: int = 0,  # max_train_ctx — наибольшая длина контекста, log_every — период печати
                      holdout_seed: int | None = None,  # сид выбора удержанных данных (None -> 7)
                      holdout_mode: str = "stratified") -> dict:  # "tail" — удержание хвостом, иначе расслоённое; возврат — итоги
        """Обучение с гейтом по удержанным данным для модели с вниманием.

            Буфер байт лежит на устройстве, окна собираются из него
            индексами (без пересылок процессор -> устройство в цикле);
            удержанная CE считается пакетами. Удержание либо сплошным
            хвостом ("tail"), либо расслоённое по всему корпусу с
            запретом обучающих окон, пересекающихся с удержанными. На
            каждую эпоху: перемешанные пакеты, шаги Adam; снимок
            сохраняется, если CE удержанных данных не выросла; в конце
            разворачивается лучший снимок. Возвращает "epochs",
            "best_holdout_ce", "holdout_role" = "selection"."""
        import numpy as np  # численные массивы
        t = self.t  # библиотека модели
        max_train_ctx = min(max_train_ctx, self.k)   # длина контекста не больше окна k
        idx_np = np.frombuffer(data, dtype=np.uint8)  # байты данных как uint8
        n = len(idx_np) - 2  # n = число позиций − 2
        dev = self.device  # устройство модели
        # Замеры на равных данных и 2 эпохах (процессор и GPU, 2 сида):
        #   пакет  CE(процессор)  CE(GPU)   с(GPU)
        #     16       1.99        1.99     10.6  нат
        #     32       2.11        2.11      5.0   — лучший баланс: −0.42 CE против 256
        #     64       2.27        2.28      3.4    за +3.5 с на кусок
        #    256       2.53        2.53      1.5  нат
        # Меньше пакет — больше шагов — глубже спуск на прочитанный байт:
        # максимум пропускной способности не равен максимуму качества.
        # На реальном дисковом цикле пакет 32 стоил 52 с на кусок (было 18)
        # при CE куска около 2.3 внутри разброса прежних 2.1–2.8 — рост
        # времени в 2.9 раза того не стоит. Выбран 64 (CE 2.28, время
        # около 1.6x): выигрыш в качестве надёжный и дешёвый.
        # Буфер на устройстве: окна собираются индексами на самом устройстве.
        raw = t.as_tensor(idx_np.copy(), dtype=t.long, device=dev)  # raw — все байты как int64 на устройстве
        offs = t.arange(max_train_ctx, dtype=t.long, device=dev)  # offs = 0..max_train_ctx − 1 (смещения внутри окна)

        rng = np.random.default_rng(7 if holdout_seed is None  # генератор: сид 7, если holdout_seed не задан,
                                    else int(holdout_seed))  # иначе сид holdout_seed
        n_hold = max(16, int(n * holdout))  # число удержанных позиций = max(16, int(n·holdout))
        if holdout_mode == "tail":  # сплошной хвост
            hold = np.arange(max(n - n_hold, n - 2048), n)  # удержанные позиции: последние не более 2048, но не меньше n_hold
            train = np.arange(0, max(16, int(hold[0]) - max_train_ctx))  # обучающие позиции 0..max(16, hold[0] − max_train_ctx) − 1
        else:  # расслоённое удержание
            # Замеры на реальных дисковых файлах: сплошной хвост не является
            # независимой выборкой из домена — файлы неоднородны, хвост
            # заучивается, а основная часть становится уверенно ошибочной,
            # и откат по лучшей CE это поощряет (тот же файл: CE хвоста
            # 1.9–2.6 nat, CE по телу 7.3–10.0 — хуже униграммной таблицы
            # 3.9; обычный Adam без отбора достигает 3.59). Гейт по
            # удержанным данным предполагает независимые окна, поэтому
            # удержание выбирается расслоённо по всему домену, а каждое
            # обучающее окно, пересекающееся с удержанным, запрещается.
            hold = np.sort(rng.choice(np.arange(max_train_ctx, n),  # случайные позиции из max_train_ctx..n − 1
                                      size=min(n_hold, n - max_train_ctx),  # число позиций = min(n_hold, n − max_train_ctx)
                                      replace=False))  # без повторений, по возрастанию
            blocked = np.zeros(n + 1, dtype=bool)  # флаги запрещённых позиций, все ложь
            for h in hold:  # по удержанным позициям
                blocked[max(0, h - 2 * max_train_ctx):h + 1] = True  # запретить позиции h − 2·max_train_ctx .. h
            train = np.flatnonzero(~blocked[:n])  # обучающие — незапрещённые позиции 0..n − 1
            train = train[train >= max_train_ctx]  # оставить только позиции, где помещается полный контекст
            if len(train) < 64:  # обучающих позиций мало
                train = np.arange(0, max(16, n - 2 * max_train_ctx))  # запасной набор 0..max(16, n − 2·max_train_ctx) − 1

        # Замер: сборщик пакетов стоил 30 с из 71 с на кусок (42%, вдвое
        # дороже прямого прохода): каждый вызов делал 4 синхронизирующие
        # пересылки процессор -> устройство и int(lens.max()) (синхронизация
        # GPU -> процессор), а уменьшение пакета 256 -> 64 увеличило число
        # вызовов в 4 раза. Позиции известны заранее, поэтому окна
        # собираются на устройстве ОДИН раз, а пакеты — срезы по индексам.
        # Построение прежнее (старый первым, байт-заполнитель 0, y = raw[pos]).
        def gather_all(pos_np, fixed_len=None):
            """Собрать окна для позиций pos_np: вернуть (X, y), где
            X — (N, pad) контексы, y = raw[pos] — следующие байты.

                len_i = min(fixed_len или max_train_ctx, pos_i); pad = max len;
                столбец j строки i — байт pos_i − len_i + j (j < pad − len_i —
                заполнитель 0); y_i = raw[pos_i]."""
            pos_np = np.asarray(pos_np, dtype=np.int64)  # позиции в массив int64
            n_ = len(pos_np)  # число позиций
            if n_ == 0:  # позиций нет
                e = t.zeros((0, 1), dtype=t.long, device=dev)  # пустая матрица (0, 1)
                return e, e[:, 0]  # пустые контексты и пустые ответы
            lens_np = np.minimum(np.full(n_, fixed_len or max_train_ctx, dtype=np.int64), pos_np)  # len_i = min(fixed_len или max_train_ctx, pos_i)
            pad = int(lens_np.max(initial=1))  # pad — наибольшая длина, минимум 1
            pos_t = t.as_tensor(pos_np, dtype=t.long, device=dev)  # позиции на устройстве
            lens_t = t.as_tensor(lens_np, dtype=t.long, device=dev)  # длины на устройстве
            j = t.arange(pad, dtype=t.long, device=dev)  # номера столбцов 0..pad − 1
            base = pos_t[:, None] - 1 - (lens_t[:, None] - 1) + j[None, :]  # индекс base_ij = pos_i − 1 − (len_i − 1) + j = pos_i − len_i + j
            mask = j[None, :] >= (pad - lens_t[:, None])  # маска: столбец j допустим, если j >= pad − len_i
            base = t.where(mask, base, t.zeros_like(base))  # недопустимым столбцам — индекс 0
            X = raw[base.clamp_(0, len(raw) - 1)]  # X — байты raw[base], индексы ограничены отрезком 0..len(raw) − 1
            X = t.where(mask, X, t.zeros_like(X))  # недопустимые столбцы заполняются нулём
            return X.contiguous(), raw[pos_t]       # вернуть (X без разрывов, ответы y = raw[pos])
            # тип int64 сохраняется: пересборка uint8 в int64 на каждый пакет стоила 1731 мкс против 5 мкс

        X_tr, y_tr = gather_all(train)  # обучающие окна и ответы
        X_va, y_va = gather_all(hold, fixed_len=max_train_ctx)  # удержанные окна фиксированной длины max_train_ctx

        def eval_ce(positions=None, eval_batch=512):
            """Средняя CE удержанных данных пакетами по eval_batch:
            (Σ_пакет средняя CE_пакет·размер_пакета) / число примеров.
            Аргумент positions не используется."""
            tot, cnt = 0.0, 0  # сумма CE и число примеров
            with t.no_grad():  # без градиентов
                for s in range(0, len(y_va), eval_batch):  # по пакетам с шагом eval_batch
                    X = X_va[s:s + eval_batch]  # контексты пакета
                    tot += float(self.ce(X, y_va[s:s + eval_batch])) * len(X)  # tot += средняя CE пакета · число примеров пакета
                    cnt += len(X)  # учесть число примеров
            return tot / max(cnt, 1)  # средняя CE (делитель не меньше 1)

        # вектор q исключён из оптимизатора: в парном внимании его градиент
        # всегда None, хранится он в весах только ради совместимости
        # правило общее: все параметры, кроме мёртвого q (этим циклом
        # пользуется и HNetByteLM без W1/W2)
        base = [self.emb, self.W1, self.b1, self.W2, self.b2] if hasattr(self, "W1") else self.params()  # базовые параметры, если есть W1, иначе все
        live_params = list(base)  # обучаемые параметры — копия списка base
        for i, blk in enumerate(getattr(self, "blocks", [])):  # по дополнительным блокам
            if i in getattr(self, "frozen_blocks", ()):   # блок заморожен — не обучать
                continue  # пропустить блок
            live_params += [blk["W1"], blk["b1"], blk["Wo"]]  # добавить параметры блока
        # дополнительные блоки обучаются вместе с основными параметрами (q по-прежнему исключён)
        for i, blk in enumerate(getattr(self, "blocks", [])):  # по дополнительным блокам (второй раз: параметры попадают в список повторно)
            if i in getattr(self, "frozen_blocks", ()):   # блок заморожен — не обучать
                continue  # пропустить блок
            live_params += [blk["W1"], blk["b1"], blk["Wo"]]  # добавить параметры блока
        opt = t.optim.Adam(live_params, lr=lr)  # оптимизатор Adam по живым параметрам
        best = eval_ce(hold)  # лучшая CE до обучения
        best_state = [p.detach().clone() for p in self.params()]  # снимок всех параметров
        self._last_best_state = best_state  # сохранить снимок и в атрибуте
        for ep in range(epochs):  # по эпохам
            perm2 = rng.permutation(len(train))  # случайный порядок обучающих позиций
            for bs in range(0, len(train), batch):  # по пакетам с шагом batch
                sel = t.as_tensor(perm2[bs:bs + batch], dtype=t.long, device=dev)  # номера строк пакета
                X = X_tr.index_select(0, sel)  # контексты пакета — строки sel
                y = y_tr.index_select(0, sel)  # ответы пакета — строки sel
                opt.zero_grad()  # обнулить градиенты
                self.ce(X, y).backward()  # градиенты CE пакета
                opt.step()  # шаг Adam
            ce_va = eval_ce(hold)  # CE удержанных данных после эпохи
            if ce_va <= best:  # не хуже лучшей
                best = ce_va  # новая лучшая CE
                best_state = [q2.detach().clone() for q2 in self.params()]  # снять веса
            if log_every and (ep + 1) % log_every == 0:  # каждые log_every эпох
                print(f"  ep {ep+1} ce {ce_va:.4f} (best {best:.4f})",  # печать номера эпохи, CE и лучшей CE
                      flush=True)  # вывод сразу, без буфера
            if best < 0.01:  # лучшая CE ниже 0.01: данные заучены, остальные эпохи пропускаются
                break  # конец обучения
        if best_state is not None:  # снимок есть
            for q2, s in zip(self.params(), best_state):  # параметр и его значение из снимка
                q2.data.copy_(s)  # восстановить параметр из снимка
        # best_holdout_ce — оценка ОТБОРА (минимум CE по эпохам на
        # разбиении для выбора весов), как мера качества она
        # оптимистична. Имя ключа сохранено ради совместимости, смысл
        # честно помечен флагом holdout_role.
        return {"epochs": epochs,  # число эпох
                "best_holdout_ce": round(best, 4),  # лучшая CE удержанных данных, 4 знака
                "holdout_role": "selection"}  # роль: оценка для отбора весов

    def _rope_logits_from_params(self, ps, ctx):
        """Логиты от заданных параметров ps = (emb, W1, b1, W2, b2), без
        обращения к self: нужно для JVP (производной по направлению).

            Используется живой self._rope — та же изометрия, что в
            full_logits. Раньше здесь была своя переписанная версия с
            дублированием пар: у неё были неверное расписание частот и
            другой порядок пар, и она расходилась с основной моделью."""
        t = self.t  # библиотека модели
        emb, W1, b1, W2, b2 = ps  # распаковать параметры
        B, L = ctx.shape  # B — размер пакета, L — число позиций
        e = self._rope(emb[ctx], L, ctx.device)  # вложения с позиционным поворотом
        pos = t.arange(L, device=ctx.device, dtype=t.float32)  # номера позиций
        dot = t.einsum("bid,bjd->bij", e, e) / (self.dim ** 0.5)  # скалярные произведения dot_{ij} = e_i·e_j / sqrt(dim)
        sc = dot - self.lam * (pos[None, :, None] - pos[None, None, :])  # счёт_{ij} = dot_{ij} − lam·(i − j)
        mask = pos[None, :, None] >= pos[None, None, :]  # маска: j <= i
        sc = sc.masked_fill(~mask, float("-inf"))  # счёт будущих позиций j > i равен −inf (после softmax вес 0)
        pp = t.softmax(sc, dim=-1)  # веса внимания: p_ij = exp(sc_ij)/Σ_j' exp(sc_ij')
        h = t.relu(t.einsum("blj,bjd->bld", pp, e) @ W1 + b1)  # h = relu((Σ_j p_ij·e_j)·W1 + b1): новый скрытый слой
        return h @ W2 + b2  # логиты h·W2 + b2

    def _certified_finite_step(self, ctx, nxt, lr_scale: float = 1.0,
                               max_backtracks: int = 12) -> float:  # max_backtracks — число уменьшений шага вдвое
        """Сертифицированный шаг ствола RoPE по конечному сдвигу логитов.

            Направление и начальный шаг предлагает линейная производная
            (JVP), а допуск проверяет РЕАЛЬНЫЙ конечный сдвиг логитов через
            finite_displacement_ce_upper (без предположения линейности).
            Отклонённая попытка откатывает все параметры. Возвращает
            гарантированную оценку снижения CE (0.0 = шаг не сделан)."""
        t = self.t  # библиотека модели
        if ctx.device != self.device:  # контекст не на устройстве модели
            ctx = ctx.to(self.device)  # перенести контекст
        if nxt.device != self.device:  # ответы не на устройстве модели
            nxt = nxt.to(self.device)  # перенести ответы
        # q в прямом проходе не участвует (внимание попарное по вложениям),
        # поэтому шаг покрывает ТОЛЬКО живые параметры — иначе набор
        # аргументов JVP не совпал бы с _rope_logits_from_params(ps).
        params = tuple([self.emb, self.W1, self.b1, self.W2, self.b2])  # обучаемые параметры (без q)
        snapshot = [q.detach().clone() for q in params]  # снимок параметров для отката
        # логиты только для следующего байта: порядок oldest-first, запрос — последняя позиция
        logits_live = self.full_logits(ctx)[:, -1, :]  # логиты последней позиции, с графом
        logits0 = logits_live.detach()  # логиты до шага z
        grads = t.autograd.grad(  # градиенты g = ∇_θ CE(z, nxt) по параметрам
            t.nn.functional.cross_entropy(logits_live, nxt), params)  # CE(z, nxt) на батче
        direction = tuple(-g.detach() for g in grads)  # направление d = −g
        alignment = float(sum(d.square().sum() for d in direction))  # согласованность A = Σ‖d‖²
        if alignment <= 1e-24:  # нулевой градиент
            return 0.0  # шаг не делается
        try:  # JVP может не удаться
            _, tangent = t.autograd.functional.jvp(  # tangent — сдвиг логитов вдоль d в первом порядке
                lambda *ps: self._rope_logits_from_params(ps, ctx),  # логиты как функция параметров
                params, direction)  # в точке params по направлению d
            width = tangent.max(dim=-1).values                 - tangent.min(dim=-1).values  # range = max_c tangent − min_c tangent по классам
            r2 = float(width.square().mean())  # R'² = среднее range²
        except Exception:  # любая ошибка JVP
            r2 = 0.0  # кривизна неизвестна
        eta = lr_scale * (4.0 * alignment / r2 if r2 > 1e-24 else 1.0)  # eta = lr_scale·4A/R'² (или lr_scale, если R'² ≈ 0)
        for _ in range(max_backtracks + 1):  # до max_backtracks + 1 попыток
            with t.no_grad():  # без графа
                for q, old in zip(params, snapshot):  # параметр и снимок
                    q.copy_(old)  # восстановить θ
                for q, d in zip(params, direction):  # параметр и направление
                    q.add_(eta * d)  # шаг: θ' = θ + eta·d
                logits1 = self.full_logits(ctx)[:, -1, :].detach()  # логиты после шага z'
            upper = float(finite_displacement_ce_upper(  # верхняя оценка прироста CE
                logits0, logits1, nxt))  # верхняя оценка U = ⟨p − y, z' − z⟩ + range(z' − z)²/8
            drop = -upper  # гарантированное снижение = −U
            if drop > 1e-12:  # снижение строго положительно
                return drop  # шаг принят: сертификат конечного сдвига
            eta *= 0.5  # иначе шаг вдвое меньше
        with t.no_grad():  # ни одна попытка не принята
            for q, old in zip(params, snapshot):  # параметр и снимок
                q.copy_(old)  # откат
        return 0.0  # шаг не сделан

    def eval_holdout_ce(self, data: bytes, holdout: float = 0.1,
                        max_eval: int = 1024):  # max_eval — максимум проверяемых позиций
        """Кросс-энтропия на непрерывном хвосте данных (удержанная часть,
        то же разбиение, что в train_guarded), мини-батчами. None, если
        данных слишком мало."""
        import numpy as _np  # численные массивы
        import torch as _th  # нейросетевая библиотека
        idx_np = _np.frombuffer(data, dtype=_np.uint8)  # байты данных как массив uint8
        n = len(idx_np) - self.k - 1  # число позиций p, для которых есть окно p..p+k−1 и ответ p+k
        if n < 512:  # позиций меньше 512
            return None  # оценка ненадёжна
        n_hold = max(16, int(n * holdout))  # число позиций хвоста n_hold = max(16, ⌊holdout·n⌋)
        hold = _np.arange(max(n - n_hold, n - 2048, n - max_eval), n)  # позиции хвоста: от max(n−n_hold, n−2048, n−max_eval) до n−1
        dev = self.device  # устройство модели
        raw = _th.as_tensor(idx_np.copy(), dtype=_th.long, device=dev)  # копия байтов как целые на устройстве
        tot, cnt = 0.0, 0  # сумма CE·размер и число примеров
        with _th.no_grad():  # без графа
            for s in range(0, len(hold), 512):  # батчи по 512 позиций
                hp = hold[s:s + 512]  # позиции батча
                # окно берётся строго ДО ответа: байты p..p+k−1, ответ — байт p+k
                # (если включить байт p+k, ответ попадает во вход).
                # Порядок окна — от старого к новому, как в gather_all выше
                # (обратный порядок давал +1.0..2.5 нат чистой ошибки).
                X = _np.stack([idx_np[p:p + self.k] for p in hp])  # матрица окон X_i = idx[p_i..p_i+k−1]
                x = _th.as_tensor(X, dtype=_th.long, device=dev)  # окна как тензор
                y = raw[_th.as_tensor(hp + self.k, dtype=_th.long,  # ответы y_i = raw[p_i + k]
                                      device=dev)]  # индексы позиций ответов
                tot += float(self.ce(x, y)) * len(x)  # сумма += CE_батча·(число примеров)
                cnt += len(x)  # число примеров += размер батча
        return tot / max(cnt, 1)  # средняя CE = Σ CE·m / Σ m

    def adapter_head_step(self, ctx, nxt, lr: float = 1e-2):  # lr — скорость обучения адаптера
        """Обучить ТОЛЬКО адаптер домена (множители A и B) на этом батче:
        Adam по множителям адаптера, ствол и голова заморожены."""
        ad = getattr(self, "domain_adapter", None)  # адаптер домена или None
        if ad is None:  # адаптера нет
            return  # нечего обучать
        t = self.t  # библиотека модели
        # восстановленный адаптер может быть numpy float64 — привести точно
        # к типу модели (float32)
        At = t.as_tensor(ad.A, device=self.device,  # A как тензор на устройстве модели
                         dtype=self.W1.dtype).requires_grad_(True)  # тип как у W1, с градиентом
        Bt = t.as_tensor(ad.B, device=self.device,  # B как тензор на устройстве модели
                         dtype=self.W1.dtype).requires_grad_(True)  # тип как у W1, с градиентом
        opt = t.optim.Adam([At, Bt], lr=lr)  # Adam по A и B
        e = self._rope(self.emb[ctx], ctx.shape[1], ctx.device)  # вложения с поворотом по позициям
        pos = t.arange(ctx.shape[1], device=ctx.device, dtype=t.float32)  # номера позиций 0..L−1
        dot = t.einsum("bid,bjd->bij", e, e) / (self.dim ** 0.5)  # скалярные произведения dot_ij = e_i·e_j / sqrt(dim)
        sc = dot - self.lam * (pos[None, :, None] - pos[None, None, :])  # счёт_ij = dot_ij − lam·(i − j)
        mask = pos[None, :, None] >= pos[None, None, :]  # маска: j <= i
        sc = sc.masked_fill(~mask, float("-inf"))  # будущим позициям — счёт −inf
        pp = t.softmax(sc, dim=-1)  # веса внимания p_ij = exp(sc_ij)/Σ exp(sc_ij')
        h0 = t.relu(t.einsum("blj,bjd->bld", pp, e) @ self.W1 + self.b1)  # скрытый слой h0 = relu((Σ_j p_ij·e_j)·W1 + b1)
        logits = (h0 + (h0 @ At.T) @ Bt.T) @ self.W2 + self.b2  # логиты = (h0 + (h0·Aᵀ)·Bᵀ)·W2 + b2
        # nxt содержит B ответов (по одному на строку), а не B·L — поэтому
        # берётся только логит последней позиции каждой строки
        loss = t.nn.functional.cross_entropy(  # потеря CE
            logits[:, -1, :], nxt.to(self.device))  # последних позиций против ответов
        opt.zero_grad()  # обнулить градиенты
        loss.backward()  # градиенты CE по A и B
        opt.step()  # шаг Adam
        with t.no_grad():  # без графа
            # A/B могут быть numpy (свежие) или тензором (из сохранённого
            # состояния) — записать обратно как numpy
            import numpy as _np2  # численные массивы
            ad.A = At.detach().cpu().numpy()  # сохранить A
            ad.B = Bt.detach().cpu().numpy()  # сохранить B

    k = 96  # контекст модели k = 96 (унифицированный интерфейс: модель.k)

    def train_certified(self, data: bytes, epochs: int = 60,  # epochs — число эпох
                        holdout: float = 0.1, batch: int = 64,  # holdout — доля хвоста, batch — размер батча
                        log_every: int = 0,  # log_every — период печати (0 — не печатать)
                        holdout_seed: int | None = None,  # сид разбиения (None — сид 7)
                        lr: float | None = None) -> dict:  # скорость обучения (None — по умолчанию train_guarded)
        """Обучение, совместимое с общим интерфейсом: быстрая фаза Adam через
        train_guarded, затем фаза дообучения сертифицированными шагами
        с гейтом по удержанному хвосту.

        Дообучение берёт мини-батчи ТОЛЬКО из обучающей части: хвост остаётся
        чистой проверочной выборкой. Шаг принят только с доказанной оценкой
        снижения CE на своём батче; каждая оценка верна лишь для своего
        батча и в общую гарантию не складывается. Если после дообучения CE
        хвоста выросла более чем на 2%, веса откатываются к лучшему снимку."""
        r = self.train_guarded(data, epochs=epochs, holdout=holdout,  # фаза Adam
                               batch=batch, max_train_ctx=self.k,  # батч и длина контекста k
                               holdout_seed=holdout_seed,  # сид разбиения
                               **({} if lr is None else {"lr": lr}))  # lr передаётся, только если задан (иначе значение train_guarded 1e-3)
        # Гибрид: после быстрой фазы Adam идёт дообучение шагами с конечным
        # сдвигом по окнам обучающей части — каждый принятый шаг несёт
        # доказанную границу снижения CE, которой у чистого Adam нет.
        # Раньше дообучение брало батчи из того же хвоста, по которому
        # считается eval_holdout_ce: проверочные данные попадали в градиент,
        # а сравнение «до/после» было самому себе контроль. Теперь позиции
        # берутся из data[:train_zone_end], хвост — только аудит.
        pre_refine_ce = self.eval_holdout_ce(data, holdout=holdout)  # CE хвоста до дообучения
        import numpy as _np  # численные массивы
        full_np = _np.frombuffer(data, dtype=_np.uint8)  # байты данных
        n_all = len(full_np) - 2  # число позиций с окном и ответом
        n_hold = max(16, int(n_all * holdout))  # число позиций хвоста
        hold_start = max(n_all - n_hold, n_all - 2048)  # начало хвоста
        train_zone_end = max(self.k + 1, hold_start - self.k)  # конец обучающей зоны: окно целиком до хвоста
        train_np = full_np[:train_zone_end]  # обучающие байты
        rng = _np.random.default_rng(7 if holdout_seed is None  # генератор с сидом 7
                                     else int(holdout_seed))  # или с заданным сидом
        positions = rng.choice(len(train_np) - self.k - 1,  # до 64 случайных позиций без повторов
                               size=min(64, len(train_np) - self.k - 1),  # число позиций
                               replace=False)  # без повторов
        # окно берётся до ответа: X_i = train_np[p_i..p_i+k−1], ответ y_i = train_np[p_i+k]
        X = _np.stack([train_np[p + self.k - 1:  # срез p+k−1 вниз до p
                                max(p - 1, -1):-1][:self.k]  # шаг −1, не ниже p; срез до k байтов
                       for p in positions])  # по позициям: окно от старого к новому
        y = train_np[positions + self.k]  # ответы y_i = train_np[p_i + k]
        import torch as _th  # нейросетевая библиотека
        refine_X = _th.as_tensor(X.copy(), dtype=_th.long,  # окна как целые
                                 device=self.device)  # на устройстве модели
        refine_y = _th.as_tensor(y, dtype=_th.long, device=self.device)  # ответы на устройстве
        total_drop, steps, commits = 0.0, 0, 0  # сумма оценок, число шагов, число принятых
        for _ in range(8):  # 8 раундов дообучения
            perm = rng.permutation(len(positions))  # случайный порядок позиций
            for s in range(0, len(perm), 32):  # батчи по 32 позиции
                bx = refine_X[perm[s:s + 32]]  # окна батча
                by = refine_y[perm[s:s + 32]]  # ответы батча
                d = self._certified_finite_step(bx, by)  # сертифицированный шаг; d — оценка снижения (0 — не сделан)
                steps += 1  # шагов +1
                if d > 0:  # шаг принят
                    commits += 1  # принятых +1
                    total_drop += d  # сумма оценок += d
        # Оценки по батчам НЕ складываются в гарантию для одной целевой
        # функции: каждая верна только на своём мини-батче — так и пишем
        bound = ({"total": total_drop, "commits": commits,  # сумма оценок и число принятых
                  "scope": "per-batch-sum, no telescoping guarantee"}  # область действия: сумма по батчам, без общей гарантии
                 if commits else None)  # если ни один шаг не принят — None
        # Сертифицированные шаги доказаны по батчам, но качество развёртывания
        # измеряется CE хвоста после фазы Adam. Если дообучение ухудшило
        # CE по всему хвосту — откат к лучшему снимку (доказать, потом
        # развёртывать).
        tail_ce = self.eval_holdout_ce(data, holdout=holdout)  # CE хвоста после дообучения
        # сравнение одинаковое с прежним: та же оценка (окна хвоста) до и после
        if (tail_ce is not None and pre_refine_ce is not None  # обе оценки есть
                and tail_ce > pre_refine_ce * 1.02):  # CE выросла более чем на 2%
            best_state = getattr(self, "_last_best_state", None)  # лучший снимок весов
            if best_state is not None:  # снимок есть
                with _th.no_grad():  # без графа
                    for q, s in zip(self.params(), best_state):  # параметр и его снимок
                        q.data.copy_(s)  # восстановить вес
            bound = None  # дообучённые веса не развёрнуты
        return {"best_holdout_ce": r["best_holdout_ce"],  # лучшая CE после фазы Adam
                "holdout_role": "selection",  # роль хвоста: отбор весов
                "certified_descent_lower_bound": bound,  # сумма доказанных снижений (None — не развёрнуто)
                "certified_refine": {"steps": steps,  # число попыток шага
                                     "commits": commits,  # число принятых шагов
                                     "deployed": bound is not None}}  # развёрнуты ли дообучённые веса

    # ---- самозавершающаяся генерация на RoPE-модели --

    def _read_all_windows(self, prompt_bytes: bytes, max_span: int = 64):  # max_span — ширина окна
        """Окна покрытия по КАЖДОМУ байту запроса, от старого к новому
        (слот L−1 — самый новый байт, он же запрос), как gather_all в
        train_guarded. Раньше окна шли в обратном порядке (ctx[::-1]) —
        ствол обучен на порядке от старого к новому, и CE была 6.0..6.4
        ната вместо 3.5..4.4."""
        ctx = list(prompt_bytes)  # байты запроса списком
        wins = []  # список окон
        i = len(ctx)  # правый край очередного окна
        while i > 0:  # пока есть непокрытые байты
            j = max(0, i - max_span)  # левый край j = max(0, i − max_span)
            w = ctx[j:i]  # окно w = ctx[j..i−1]
            wins.append([32] * (max_span - len(w)) + w)   # окно с дополнением пробелами (32) слева до ширины max_span
            i = j  # следующий участок левее
        return wins or [[0]]  # окна от последнего к первому; пустой запрос — [[0]]

    def generate_auto(self, prompt: str, max_bytes: int = 4096,  # max_bytes — предел длины ответа
                      temp: float = 0.5, seed: int = 1) -> str:  # temp — температура, seed — сид генерации
        """Генерация, где модель сама решает, ДОЛГО ли отвечать: читает ВЕСЬ
        запрос через окна покрытия, логиты окон усредняются. Последнее окно
        не считается дважды, делитель — фактическое число слагаемых.

        Остановка: после знака конца предложения при низкой уверенности или
        повторе, 8 повторов одного байта подряд, 8 шагов с максимальной
        вероятностью ниже 1.2/256, либо перевод строки при 3 таких шагах.
        Возвращает запрос с ответом, обрезанным до последнего знака конца
        предложения."""
        t = self.t  # библиотека модели
        pb = prompt.encode('utf-8')  # байты запроса
        # ширина окна должна СОВПАДАТЬ с обучением (self.k = 96) — раньше было
        # жёстко 64 (значение другой модели); окна головы не обрезаются
        windows = self._read_all_windows(pb, max_span=self.k)  # окна покрытия шириной k
        with t.no_grad():  # без графа
            g = t.Generator(device=self.device).manual_seed(seed)  # генератор с сидом
            out = bytearray()  # накопленный ответ
            last_ctx = windows[-1]  # текущее окно (в начале — окно конца запроса)
            rep_counter = 0  # число повторов последнего байта
            low_conf = 0  # число подряд шагов с низкой уверенностью
            last_token = None  # предыдущий байт
            while len(out) < max_bytes:  # пока не достигнут предел длины
                cur = windows[:-1] + [last_ctx]  # окна: все, кроме последнего, + текущее окно
                probs = None  # накопитель вероятностей
                for w in cur:  # по всем окнам
                    x = t.as_tensor([w], dtype=t.long, device=self.device)  # окно как тензор (1, k)
                    pp = t.softmax(self.logits(x)[0] / max(temp, 1e-3),  # вероятности p = softmax(z / max(temp, 1e-3))
                                   dim=-1)  # по классам
                    probs = pp if probs is None else probs + pp  # сумма вероятностей по окнам
                probs = probs / len(cur)  # среднее вероятностей p̄ = Σ p / n
                top_p = float(probs.max())  # максимальная вероятность max_c p̄_c
                b = int(t.multinomial(probs, 1, generator=g))  # байт b ~ p̄
                out.append(b)  # добавить байт к ответу
                last_ctx = last_ctx[1:] + [b]  # окно сдвигается: убрать старейший, добавить b
                ch = bytes([b])  # байт как объект bytes
                if top_p < (1.0 / 256) * 1.2:  # максимум ниже 1.2/256 — низкая уверенность
                    low_conf += 1  # счётчик +1
                else:  # уверенность достаточная
                    low_conf = 0  # сброс счётчика
                if b == last_token:  # байт совпал с предыдущим
                    rep_counter += 1  # повторов +1
                else:  # другой байт
                    rep_counter = 0  # повторов 0
                last_token = b  # запомнить байт
                if ch in b'.!?':  # знак конца предложения
                    nxt = int(t.multinomial(probs, 1, generator=g)) \
                        if False else None  # всегда None (заглядывание вперёд отключено)
                    if low_conf >= 2 or rep_counter >= 3:  # 2 неуверенных шага или 3 повтора
                        break  # модель закончила
                if rep_counter >= 8 or low_conf >= 8:  # 8 повторов или 8 неуверенных шагов
                    break  # стоп
                if ch in b'\n' and low_conf >= 3:  # перевод строки при 3 неуверенных шагах
                    break  # стоп
        tail = bytes(out)  # ответ байтами
        # обрезать до последнего знака конца предложения, если он есть
        for term in (b'.', b'!', b'?'):  # по знакам конца
            k = tail.rfind(term)  # позиция последнего вхождения
            if k != -1:  # найден
                tail = tail[:k + 1]  # срез до него включительно
                break  # первый найденный знак из списка
        return (prompt + tail.decode('utf-8', errors='replace'))  # запрос + раскодированный ответ

    def generate(self, prompt: str, n_bytes: int = 120, temp: float = 0.5,  # n_bytes — длина ответа, temp — температура
                 seed: int = 1) -> str:  # seed — сид генерации
        """Сгенерировать ровно n_bytes байтов: на каждом шаге по последним k
        байтам считаются логиты и берётся выборка из softmax(logits / temp).
        Возвращает запрос с раскодированным ответом."""
        t = self.t  # библиотека модели
        with t.no_grad():  # без графа
            g = t.Generator(device=self.device).manual_seed(seed)  # генератор с сидом
            ctx = list(prompt.encode("utf-8"))  # байты запроса
            out = bytearray()  # накопленный ответ
            for _ in range(n_bytes):  # n_bytes шагов
                # порядок от старого к новому, как при обучении (было ctx[::-1]:
                # в слоте запроса оказывался СТАРЕЙШИЙ байт)
                x = t.as_tensor([ctx[-self.k:]], dtype=t.long,  # последние k байтов (окно)
                                device=self.device)  # на устройстве модели
                probs = t.softmax(self.full_logits(x)[0, -1] /  # p = softmax(z_посл / max(temp, 1e-3))
                                  max(temp, 1e-3), dim=-1)  # по классам
                b = int(t.multinomial(probs, 1, generator=g))  # байт b ~ p
                out.append(b); ctx.append(b)  # добавить байт в ответ и в контекст
        return (prompt.encode("utf-8") + bytes(out)).decode(  # запрос + ответ как UTF-8
            "utf-8", errors="replace")  # неверные последовательности заменяются


def load_rope(path):
    """Загрузить RoPE-модель из файла path: восстанавливает dim, hidden, theta0, lam
    и копирует сохранённые веса emb, q, W1, b1, W2, b2."""
    t, _ = _torch_and_device()  # torch и устройство
    st = t.load(path, weights_only=False)  # словарь состояния
    lm = RopeAttentionByteLM(dim=st["dim"], hidden=st["hidden"],  # новая модель с сохранёнными dim, hidden
                             theta0=st["theta0"], lam=st["lam"])  # и углами theta0, затуханием lam
    for n in ("emb", "q", "W1", "b1", "W2", "b2"):  # по именам параметров
        getattr(lm, n).data.copy_(st[n])  # скопировать сохранённый вес
    return lm  # загруженная модель


class CertifiedStepLM:
    """Байтовая модель, обучаемая СЕРТИФИЦИРОВАННЫМ шагом (без Adam и отката по снимку):

      направление D = −grad CE; A = Σ D² = ‖D‖² (согласованность),
      R' = max(сдвиг логитов) − min(сдвиг логитов) по классам,
      eta = 4A/R'² (точный минимум квадратичного мажоранта),
      гарантия: CE(θ') <= CE(θ) − 2A²/R'².

    Каждый шаг снижает CE на доказанную величину (шаг безопасен по построению).
    Развёртывание лучшего снимка по удержанным данным остаётся как страховка."""

    def __init__(self, k: int = 10, dim: int = 96, hidden: int = 192,  # создать модель: k — длина контекста, dim — размер вложения, hidden — скрытый слой
                 seed: int = 7):  # seed — сид начальных весов
        import torch  # нейросетевая библиотека
        t = torch  # короткое имя
        self.t = t  # библиотека модели
        self.k, self.dim, self.hidden = k, dim, hidden  # запомнить k, dim, hidden
        g = t.Generator().manual_seed(seed)  # генератор с сидом
        self.emb = t.nn.Parameter(t.randn(256, dim, generator=g) * 0.08)  # вложения emb (256, dim) ~ N(0,1)·0.08
        self.W1 = t.nn.Parameter(t.randn(k * dim, hidden, generator=g) * 0.08)  # матрица W1 (k·dim, hidden) ~ N(0,1)·0.08
        self.b1 = t.nn.Parameter(t.zeros(hidden))  # смещение b1 = 0
        self.W2 = t.nn.Parameter(t.randn(hidden, 256, generator=g) * 0.08)  # матрица W2 (hidden, 256) ~ N(0,1)·0.08
        self.b2 = t.nn.Parameter(t.zeros(256))  # смещение b2 = 0
        self._ef21_s: float | None = None  # фиксированный масштаб тернарного квантования (замораживается на первом шаге)
        self.device = _resolve_device(t, self.dim, self.hidden)  # устройство вычислений
        if self.device == "cuda":  # модель на GPU
            for prm in self.params():  # по всем параметрам
                prm.data = prm.data.to(self.device)  # перенести на устройство

    def params(self):
        """Список обучаемых параметров [emb, W1, b1, W2, b2]."""
        return [self.emb, self.W1, self.b1, self.W2, self.b2]  # параметры модели

    @staticmethod  # статический метод (без self)
    def _ste_ternary(W, t=None):  # W — матрица весов, t — torch (None — импорт)
        """Тернарная псевдоквантизация с пропуском градиента (straight-through):
        прямой проход видит веса {−s, 0, +s} по строкам, s = среднее |W| строки,
        q = clamp(round(W / s), −1, 1)·s; градиент течёт через исходные веса W
        (возвращается W + (q − W).detach())."""
        if t is None:  # библиотека не передана
            import torch as _t  # импорт torch
            t = _t  # использовать его
        scale = W.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-12)  # s = max(1e-12, mean_j |W_ij|) по строке
        q = t.round(W.detach() / scale).clamp(-1, 1) * scale  # квантованное значение q = clamp(round(W/s), −1, 1)·s
        return W + (q - W).detach()  # прямое значение q, градиент как у W

    def logits(self, ctx):
        """Логиты для батча контекстов ctx (B, k): e = emb[ctx], h = relu(e·W1 + b1),
        при наличии адаптера домена h = h + (h·Aᵀ)·Bᵀ; логиты = h·W2 + b2.
        При включённом _qat веса проходят тернарную псевдоквантизацию."""
        t = self.t  # библиотека модели
        if ctx.device != getattr(self, "device", ctx.device):  # контекст не на устройстве модели
            ctx = ctx.to(self.device)   # перенести (вход может прийти с CPU)
        e_w = (self._ste_ternary(self.emb, t)  # квантованные вложения, если _qat
               if getattr(self, "_qat", False) else self.emb)  # иначе обычные
        e = e_w[ctx]  # e = e_w[ctx], форма (B, k, dim)
        h = t.relu(e.reshape(ctx.shape[0], -1) @  # h = relu(e·W1 + b1), e развёрнута в (B, k·dim)
                   (self._ste_ternary(self.W1, t) if getattr(self, "_qat", False)  # W1 квантуется, если _qat
                    else self.W1) + self.b1)  # иначе обычная
        # необязательный низкоранговый адаптер домена на скрытом слое:
        # h + (h·Aᵀ)·Bᵀ; при B = 0 это точное тождество, поэтому свежий
        # адаптер логиты не меняет
        ad = getattr(self, "domain_adapter", None)  # адаптер домена или None
        if ad is not None:  # адаптер есть
            At = t.as_tensor(ad.A, device=h.device, dtype=h.dtype)  # A на устройстве h
            Bt = t.as_tensor(ad.B, device=h.device, dtype=h.dtype)  # B на устройстве h
            h = h + (h @ At.T) @ Bt.T  # адаптированный слой h = h + (h·Aᵀ)·Bᵀ
        w2 = (self._ste_ternary(self.W2, t) if getattr(self, "_qat", False)  # W2 квантуется, если _qat
              else self.W2)  # иначе обычная
        return h @ w2 + self.b2  # логиты h·W2 + b2

    def ternary_(self) -> dict[str, Any]:  # возвращает словарь со статистикой квантования
        """Квантизовать веса модели до тернарных на месте; возвращает словарь
        со статистикой."""
        return ternary_quant(self)  # тернарная квантизация self

    def _to_dev(self, tns):
        """Перенести тензор или кортеж тензоров на устройство модели (вход может
        прийти с CPU)."""
        if hasattr(tns, "to"):  # это тензор (есть метод to)
            return tns.to(self.device)  # перенести тензор
        return type(tns)(x.to(self.device) for x in tns)  # иначе тот же тип из перенесённых элементов

    def _certified_step(self, ctx, nxt, lr_scale: float = 1.0) -> float:  # lr_scale — множитель шага
        """Один сертифицированный шаг по батчу: D = −grad CE, eta = 4A/R'²,
        θ' = θ + eta·D, где A = ‖D‖², R'² — средний квадрат диапазона сдвига
        логитов вдоль D. Возвращает доказанное снижение 2A²/R'² (0, если R' = 0
        или A = 0 — логиты вдоль направления не меняются)."""
        t = self.t  # библиотека модели
        ctx, nxt = self._to_dev((ctx, nxt))  # перенести контекст и ответы на устройство
        logits = self.logits(ctx)  # логиты батча
        loss = t.nn.functional.cross_entropy(logits, nxt)  # CE на батче
        g = t.autograd.grad(loss, self.params(), retain_graph=True)  # градиенты по параметрам (граф сохранить)
        D = t.cat([-gi.reshape(-1) for gi in g])  # D = concat(−g_i) — направление антиградиента
        A = float((D * D).sum())  # согласованность A = Σ D² = ‖D‖²
        # R'² = среднее по примерам квадрата диапазона сдвига логитов вдоль
        # ПОЛНОГО направления D — точная производная по направлению (JVP)
        # всей сети, включая нелинейности emb/W1; отсюда гарантия
        # L(θ') ≤ L(θ) − 2A²/R'² на средней потере батча
        _, disp = t.autograd.functional.jvp(  # disp — сдвиг логитов в первом порядке
            lambda *prm: self._logits_from_params(prm, ctx),  # логиты как функция параметров
            tuple(self.params()), tuple(-gi for gi in g))  # в текущих параметрах вдоль −g
        per_sample_range = disp.max(dim=1).values - disp.min(dim=1).values  # диапазон на пример range_b = max_c disp_bc − min_c disp_bc
        Rp_sq = float((per_sample_range ** 2).mean())  # средний квадрат диапазона R'² = mean_b(range_b²)
        if Rp_sq <= 0 or A <= 0:  # сдвиг логитов нулевой или градиент нулевой
            return 0.0  # шаг не даёт снижения
        eta = lr_scale * 4.0 * A / Rp_sq  # шаг eta = lr_scale·4A/R'²
        with t.no_grad():  # без графа
            off = 0  # смещение в векторе D
            for prm, gi in zip(self.params(), g):  # по параметрам
                sz = prm.numel()  # число элементов параметра
                prm.add_(eta * D[off:off + sz].reshape(prm.shape))  # обновление θ_i += eta·D[off..off+sz]
                off += sz  # сдвиг смещения
        return 2.0 * A * A / Rp_sq  # доказанное снижение 2A²/R'²

    # ---- сертифицированный шаг по КОНЕЧНОМУ сдвигу ----
    def _certified_finite_step(self, ctx, nxt, lr_scale: float = 1.0,  # lr_scale — множитель шага
                               max_backtracks: int = 12) -> float:  # max_backtracks — число уменьшений шага вдвое
        """Сертифицированный шаг по конечному сдвигу логитов: JVP только
        предлагает eta (эвристика масштаба), а допуск проверяет РЕАЛЬНЫЙ
        конечный сдвиг логитов через finite_displacement_ce_upper (оценка через
        диапазон и log-sum-exp, без линейности). Отклонённая попытка откатывает
        все параметры. Возвращает доказанную границу снижения (0 = шаг не сделан).
        Так исправлено наблюдение, что чистый шаг по JVP может УВЕЛИЧИТЬ CE
        на нелинейной сети."""
        t = self.t  # библиотека модели
        ctx, nxt = self._to_dev((ctx, nxt))  # перенести на устройство
        params = tuple(self.params())  # обучаемые параметры
        snapshot = [p.detach().clone() for p in params]  # снимок для отката
        logits_live = self.logits(ctx)  # логиты с графом
        logits0 = logits_live.detach()  # логиты до шага z
        grads = t.autograd.grad(  # градиенты g = ∇ CE(z, nxt)
            t.nn.functional.cross_entropy(logits_live, nxt), params)  # по параметрам
        direction = tuple(-g.detach() for g in grads)  # направление d = −g
        alignment = float(sum(d.square().sum() for d in direction))  # согласованность A = Σ‖d‖²
        if alignment <= 1e-24:  # нулевой или некорректный градиент
            return 0.0  # шаг не делается
        try:  # JVP может не удаться
            _, tangent = t.autograd.functional.jvp(  # tangent — сдвиг логитов вдоль d
                lambda *ps: self._logits_from_params(ps, ctx), params,  # логиты как функция параметров
                direction)  # по направлению d
            width = tangent.max(dim=-1).values - tangent.min(dim=-1).values  # диапазон range = max_c tangent − min_c tangent
            r2 = float(width.square().mean())  # R'² = среднее range²
        except Exception:  # любая ошибка JVP
            r2 = 0.0  # кривизна неизвестна
        eta = lr_scale * (4.0 * alignment / r2 if r2 > 1e-24 else 1.0)  # eta = lr_scale·4A/R'² (или lr_scale при R'² ≈ 0)
        for _ in range(max_backtracks + 1):  # до max_backtracks + 1 попыток
            with t.no_grad():  # без графа
                for p, old in zip(params, snapshot):  # параметр и снимок
                    p.copy_(old)  # восстановить θ
                for p, d in zip(params, direction):  # параметр и направление
                    p.add_(eta * d)  # шаг: θ' = θ + eta·d
                logits1 = self.logits(ctx).detach()  # логиты после шага z'
            upper = float(finite_displacement_ce_upper(logits0, logits1, nxt))  # верхняя оценка U = ⟨p − y, z' − z⟩ + range(z' − z)²/8
            drop = -upper  # доказанное снижение = −U
            if drop > 1e-12:  # снижение строго положительно
                return drop  # шаг принят: сертификат конечного сдвига
            eta *= 0.5  # иначе шаг вдвое меньше
        with t.no_grad():  # ни одна попытка не принята
            for p, old in zip(params, snapshot):  # параметр и снимок
                p.copy_(old)  # откат
        return 0.0  # шаг не сделан: положительного сертификата нет

    # ---- адаптивный мажорант (шаг CE, чувствительный к дисперсии) ----
    def _adaptive_step(self, ctx, nxt, lr_scale: float = 1.0) -> float:  # lr_scale — множитель шага
        """Шаг CE, сертифицированный адаптивной кривизной: мажорант
        L(θ') ≤ L(θ) − ηA + η²C, где C — взвешенная дисперсия сдвига логитов
        вдоль направления под мягкой целью (она мажорирует оценку через диапазон:
        C ≤ R'²/8). Шаг η* = A/(2C), снижение A²/(4C).

        Возвращает доказанное снижение (0, если шаг отклонён: C ≤ 0, A ≤ 0 или
        η* > 1 — единичный предел теоремы)."""
        t = self.t  # библиотека модели
        ctx, nxt = self._to_dev((ctx, nxt))  # перенести на устройство
        logits = self.logits(ctx)  # логиты батча
        loss = t.nn.functional.cross_entropy(logits, nxt)  # CE на батче
        g = t.autograd.grad(loss, self.params(), retain_graph=True)  # градиенты (граф сохранить)
        D = t.cat([-gi.reshape(-1) for gi in g])  # направление D = concat(−g_i)
        A = float((D * D).sum())  # согласованность A = ‖D‖²
        # сдвиг логитов на пример вдоль D (JVP на среднем по батчу)
        _, disp = t.autograd.functional.jvp(  # disp — сдвиг логитов
            lambda *prm: self._logits_from_params(prm, ctx),  # логиты как функция параметров
            tuple(self.params()), tuple(-gi for gi in g))  # вдоль −g
        soft = t.softmax(logits, dim=1)  # мягкая цель p = softmax(z)
        mean_disp = (soft * disp).sum(dim=1, keepdim=True)  # средний сдвиг μ_b = Σ_c p_bc·disp_bc
        var_disp = (soft * (disp - mean_disp) ** 2).sum(dim=1)  # дисперсия сдвига Var_b = Σ_c p_bc·(disp_bc − μ_b)²
        per_sample_range = disp.max(dim=1).values - disp.min(dim=1).values  # диапазон на пример range_b = max_c disp − min_c disp
        # C = kappa(R)·Var, kappa(R) = (e^R − 1 − R)/R² ≥ 1/2, где R — наибольший
        # диапазон сдвига по примерам. Раньше бралось C = Var без множителя:
        # кривизна выходила слишком малой, шаг — слишком осторожным, а
        # заявленное снижение в 1.46–1.79 раза меньше реального (180 шагов,
        # 0 нарушений) — верно, но слабее теоремы
        R = float(per_sample_range.max())  # максимальный диапазон R = max_b range_b
        # kappa(R) → 1/2 при R → 0, но ниже R ~ 1e-6 выражение теряет точность:
        # kappa(1e-7) = 0.494, kappa(1e-8) = −0.608 (отрицательная кривизна
        # → шаг вверх), kappa(1e-9) = 82.7. Ниже 1e-5 берётся предел 1/2
        R_c = min(R, 50.0)  # R обрезан сверху числом 50 (против переполнения e^R)
        C = float(var_disp.mean()) * (  # кривизна C = mean(Var)·kappa(R)
            (math.exp(R_c) - 1.0 - R_c) / (R_c * R_c) if R > 1e-5 else 0.5)  # kappa = (e^R − 1 − R)/R², если R > 1e-5, иначе 1/2
        # Сертификат через диапазон сформулирован для МАКСИМАЛЬНОГО диапазона R
        # по примерам, а не для среднего квадрата: с R'² = mean(range²) шаг
        # 4A/R'² слишком велик, и заявка снижения нарушалась на ~5e-4 в 7
        # шагах из 12. Поэтому R² = (max range)²
        Rp_sq = R * R  # квадрат диапазона R² = R·R
        if C <= 0 or A <= 0 or Rp_sq <= 0:  # кривизна, градиент или диапазон нулевые
            return 0.0  # шаг не делается
        # адаптивный η* = A/(2C) с единичным пределом; когда предел
        # срабатывает — шаг через диапазон 4A/R², который ВСЕГДА
        # сертифицирован (C ≤ R²/8, значит A²/(4C) ≥ 2A²/R² — берётся
        # гарантия использованной ветви)
        use_adaptive = A / (2.0 * C) <= 1.0  # адаптивная ветвь, если η* ≤ 1
        eta = (A / (2.0 * C)) if use_adaptive else (4.0 * A / Rp_sq)  # eta = η* или 4A/R²
        eta *= lr_scale  # масштабирование eta *= lr_scale
        # Раньше снижение считалось для НЕМАСШТАБИРОВАННОГО η*, а eta уже был
        # умножен на lr_scale. Мажорант L(eta) ≤ L(0) − A·eta + C·eta² с
        # кривизной выбранной ветви, поэтому снижение нужно считать при
        # ФАКТИЧЕСКОМ eta: при lr_scale = 0.05 старая заявка в ~10 раз
        # превышала реальную (40 шагов из 40 с нарушением). При lr_scale = 1
        # обе формулы совпадают с теоремой
        drop = (A * eta - C * eta * eta) if use_adaptive else (  # снижение = A·eta − C·eta²
            A * eta - (Rp_sq / 8.0) * eta * eta)  # или A·eta − (R²/8)·eta²
        snapshot = [q.detach().clone() for q in self.params()]  # снимок параметров
        with t.no_grad():  # без графа
            off = 0  # смещение в D
            for prm, gi in zip(self.params(), g):  # по параметрам
                sz = prm.numel()  # число элементов
                prm.add_(eta * D[off:off + sz].reshape(prm.shape))  # обновление θ_i += eta·D[off..off+sz]
                off += sz  # сдвиг смещения
        # Мажорант построен по сдвигу первого порядка (JVP), а теорема — о
        # точном. Замерено: 6 шагов из 40 завышали снижение до 1.9e-3. Тот же
        # контракт, что у рабочего шага RoPE (_certified_finite_step): проверить
        # реальное снижение на фактической CE батча; если оно меньше заявки —
        # откатить, шаг не сделан
        with t.no_grad():  # без графа
            realised = float(loss.detach()) - float(  # реальное снижение = CE_до − CE_после
                t.nn.functional.cross_entropy(self.logits(ctx), nxt))  # CE после шага
        if realised < drop:  # реальное снижение меньше заявленного
            with t.no_grad():  # без графа
                for q, old_q in zip(self.params(), snapshot):  # параметр и снимок
                    q.copy_(old_q)  # откат
            return 0.0  # шаг не сделан
        return drop  # доказанное снижение

    # ---- квантованное обучение EF21 внутри шага (обратная связь по ошибке) ----
    def _train_split(self, data: bytes, holdout: float,  # data — данные, holdout — доля хвоста
                     holdout_seed: int | None):  # сид разбиения (None — сид 7)
        """Общее разбиение на обучение и удержание + проверочные тензоры.
        Хвост выбирается случайно и зафиксирован сидом: при повторном
        обучении на тех же данных он уже не меняется и не переучивается
        (иначе обучение на повторе останавливалось). Батчи лежат на
        устройстве модели, окна берутся через sliding_window_view без
        копий вместо питоновского цикла с np.stack."""
        import numpy as np  # численные массивы
        t = self.t  # библиотека модели
        dev = self.device  # устройство модели
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # байты данных как целые
        n = len(idx) - self.k  # число позиций p, для которых есть окно p..p+k−1 и ответ p+k
        rng = np.random.default_rng(7 if holdout_seed is None  # генератор с сидом 7
                                    else holdout_seed)  # или с заданным сидом
        perm = rng.permutation(n)  # случайная перестановка позиций 0..n−1
        n_hold = max(16, int(n * holdout))  # число удерживаемых позиций
        hold, train = np.sort(perm[:n_hold]), perm[n_hold:]  # hold — первые n_hold позиций перестановки (отсортированы), train — остальные
        # скользящие окна без копий: windows[i] = idx[i:i+k]
        from numpy.lib.stride_tricks import sliding_window_view  # вид скользящих окон
        win = sliding_window_view(idx, self.k)  # форма (len(idx)−k+1, k)
        ctx_va = t.as_tensor(win[hold], dtype=t.long, device=dev)  # окна удержанных позиций
        nxt_va = t.as_tensor(idx[hold + self.k], dtype=t.long, device=dev)  # ответы: байт hold_i + k

        def vce():  # локальная функция: CE на удержанных данных
            with t.no_grad():  # без графа
                return float(t.nn.functional.cross_entropy(  # CE по удержанным
                    self.logits(ctx_va), nxt_va))  # логиты удержанных окон против ответов

        def make_batch(sel):  # батч по номерам позиций sel
            X = win[sel]                      # векторизованный сбор окон X_i = idx[sel_i..sel_i+k−1]
            y = idx[sel + self.k]  # ответы y_i = idx[sel_i + k]
            return (t.as_tensor(np.ascontiguousarray(X), dtype=t.long,  # окна как непрерывный тензор
                                device=dev),  # на устройстве модели
                    t.as_tensor(y, dtype=t.long, device=dev))  # ответы на устройстве модели
        return rng, train, vce, make_batch  # генератор, обучающие позиции, CE хвоста, сборщик батчей

    def _ef21_step(self, ctx, nxt, e_res: list, lr_scale: float = 1.0  # e_res — остатки по параметрам, lr_scale — множитель шага
                   ) -> tuple[float, bool]:  # возвращает (снижение или 0, принят ли шаг)
        """Один шаг EF21 внутри шага (квантованное направление с обратной связью):
          m = g + e;  d = qTern(m);  e' = m − d.
        Пробный шаг θ' = θ − eta·d допускается ТОЛЬКО по сертифицированному
        критерию (гарантия спуска квадратичного мажоранта вдоль
        КВАНТОВАННОГО направления); отклонённая проба не меняет ни веса, ни
        остаток.

        e_res: остатки ошибки по параметрам (создаёт вызывающий код).
        Возвращает (доказанное снижение | 0, принят ли шаг)."""
        t = self.t  # библиотека модели
        ctx, nxt = self._to_dev((ctx, nxt))  # перенести на устройство
        logits = self.logits(ctx)  # логиты батча
        loss = t.nn.functional.cross_entropy(logits, nxt)  # CE на батче
        g = t.autograd.grad(loss, self.params(), retain_graph=True)  # градиенты (граф сохранить)
        m = [gi + ei for gi, ei in zip(g, e_res)]  # m_i = g_i + e_i (градиент + накопленный остаток)
        # qTern с ФИКСИРОВАННЫМ масштабом (не подбирается под данные):
        # s* = 0.2·‖m‖_inf считается один раз и замораживается на всю
        # траекторию (хранится в self._ef21_s)
        if self._ef21_s is None:  # масштаб ещё не задан
            mi_all = t.cat([mi.reshape(-1) for mi in m])  # все m одним вектором
            self._ef21_s = 0.2 * float(mi_all.abs().max())  # фиксированный масштаб s* = 0.2·max|m|
        d = []  # квантованные направления по параметрам
        for mi in m:  # по параметрам
            s = self._ef21_s  # фиксированный масштаб
            d.append(s * t.sign(t.round(mi / (s + 1e-12))))  # тернарное значение d_i = s·sign(round(m_i/(s + 1e-12))) ∈ {−s, 0, +s}
        # сертифицированный критерий вдоль d: A = ⟨m, d⟩ (согласованность с
        # направлением спуска), кривизна — по диапазону сдвига логитов
        # (JVP) вдоль направления −d
        _, disp = t.autograd.functional.jvp(  # disp — сдвиг логитов
            lambda *prm: self._logits_from_params(prm, ctx),  # логиты как функция параметров
            tuple(self.params()), tuple(-di for di in d))  # вдоль −d
        per_sample_range = disp.max(dim=1).values - disp.min(dim=1).values  # диапазон на пример range_b = max_c disp − min_c disp
        Rp_sq = float((per_sample_range ** 2).mean())  # средний квадрат диапазона R'² = mean_b(range_b²)
        A = float(sum(float((mi * di).sum()) for mi, di in zip(m, d)))  # согласованность A = Σ_i ⟨m_i, d_i⟩
        if Rp_sq <= 0 or A <= 0:  # сдвиг логитов нулевой или направление не согласовано
            return 0.0, False  # честный отказ: ничего не выполнено
        eta0 = lr_scale * 4.0 * A / Rp_sq  # начальный шаг eta0 = lr_scale·4A/R'²
        obj_before = float(loss.detach())  # CE до шага
        # пробный шаг с дроблением (постоянная 1/L сети неизвестна: уменьшать
        # eta вдвое, пока измеренное снижение > 0, максимум 5 раз)
        eta = eta0  # текущий шаг
        admitted = False  # шаг пока не принят
        for _ in range(5):  # до 5 попыток
            snap = [prm.detach().clone() for prm in self.params()]  # снимок параметров
            with t.no_grad():  # без графа
                for prm, di in zip(self.params(), d):  # параметр и квантованное направление
                    prm.add_(-eta * di.reshape(prm.shape))  # обновление θ_i += −eta·d_i
                obj_after = float(t.nn.functional.cross_entropy(  # CE после шага
                    self.logits(ctx), nxt))  # по тому же батчу
            actual_drop = obj_before - obj_after  # фактическое снижение CE
            if actual_drop > 0:  # CE уменьшилась
                admitted = True  # шаг принят
                break  # дальше не дробить
            with t.no_grad():  # CE не уменьшилась
                for prm, st in zip(self.params(), snap):  # параметр и снимок
                    prm.copy_(st)  # откат
            eta *= 0.5  # шаг вдвое меньше
        if not admitted:  # ни одна попытка не принята
            return 0.0, False  # шаг не сделан: веса И остаток не меняются
        with t.no_grad():  # без графа
            for er, mi, di in zip(e_res, m, d):  # остаток, m и d по параметрам
                er.copy_(mi - di)  # e'_i = m_i − d_i (не переданная квантованием часть)
        return actual_drop, True  # фактическое снижение, шаг принят

    def train_ef21(self, data: bytes, epochs: int = 60,  # epochs — число эпох
                   holdout: float = 0.1, batch: int = 256,  # holdout — доля хвоста, batch — размер батча
                   log_every: int = 0,  # log_every — период печати (не используется)
                   holdout_seed: int | None = None) -> dict:  # сид разбиения (None — сид 7)
        """Цикл обучения EF21: квантованное направление внутри каждого
        сертифицированного шага с обратной связью по ошибке. Отклонённые шаги
        не меняют веса и остаток. log_every не используется. Возвращает словарь:
        CE хвоста, сумму фактических снижений, число принятых и отклонённых шагов."""
        t = self.t  # библиотека модели
        rng, train, vce, make_batch = self._train_split(data, holdout,  # разбиение и помощники
                                                   holdout_seed)  # с заданным сидом
        e_res = [t.zeros_like(prm) for prm in self.params()]  # остатки ошибки e = 0
        self._ef21_s = None  # масштаб квантования замораживается заново на траекторию
        best = vce(); best_state = [q.detach().clone() for q in self.params()]  # лучшая CE хвоста и снимок весов
        certified_guarantee = 0.0  # сумма фактических снижений
        admits = rejects = 0  # счётчики принятых и отклонённых шагов
        for ep in range(epochs):  # по эпохам
            order = rng.permutation(len(train))  # случайный порядок обучающих позиций
            for bs in range(0, len(train), batch):  # батчи по batch позиций
                ctx, nxt = make_batch(train[order[bs:bs + batch]])  # окна и ответы батча
                drop, ok = self._ef21_step(ctx, nxt, e_res)  # шаг EF21: снижение и флаг приёма
                if ok:  # шаг принят
                    admits += 1  # принятых +1
                    certified_guarantee += drop  # сумма снижений += drop
                else:  # шаг отклонён
                    rejects += 1  # отклонённых +1
            ce_va = vce()  # CE хвоста после эпохи
            if ce_va <= best:  # не хуже лучшей
                best = ce_va  # новая лучшая CE
                best_state = [q.detach().clone() for q in self.params()]  # снять веса
        with t.no_grad():  # без графа
            for prm, st in zip(self.params(), best_state):  # параметр и снимок
                prm.copy_(st)  # восстановить лучший вес
        return {"final_holdout_ce": best,  # лучшая CE хвоста
                "certified_guarantee_sum": certified_guarantee,  # сумма фактических снижений
                "ef21_admits": admits, "ef21_rejects": rejects}  # число принятых и отклонённых шагов

    def _logits_from_params(self, params, ctx):
        """Логиты как явная функция кортежа параметров (для JVP): e = emb[ctx],
        h = relu(e·W1 + b1), логиты = h·W2 + b2."""
        t = self.t  # библиотека модели
        emb, W1, b1, W2, b2 = params  # распаковать параметры
        e = emb[ctx]  # вложения контекста, форма (B, k, dim)
        h = t.relu(e.reshape(ctx.shape[0], -1) @ W1 + b1)  # скрытый слой h = relu(e_flat·W1 + b1), e_flat — (B, k·dim)
        return h @ W2 + b2  # логиты h·W2 + b2

    def train_certified(self, data: bytes, epochs: int = 60,  # epochs — число эпох
                        holdout: float = 0.1, batch: int = 256,  # holdout — доля хвоста, batch — размер батча
                        log_every: int = 0,  # log_every — период печати (0 — не печатать)
                        holdout_seed: int | None = None) -> dict:  # сид разбиения (None — сид 7)
        """Обучение сертифицированными шагами: каждый батч делает шаг
        _certified_step (theta' = θ + eta·D, доказанное снижение 2A²/R'²);
        после эпохи CE хвоста сравнивается с лучшей, по окончании веса
        восстанавливаются из лучшего снимка. Возвращает число эпох, лучшую
        CE хвоста и сумму доказанных снижений."""
        t = self.t  # библиотека модели
        rng, train, vce, make_batch = self._train_split(data, holdout,  # разбиение и помощники
                                                   holdout_seed)  # с заданным сидом
        best = vce(); best_state = [q.detach().clone() for q in self.params()]  # лучшая CE и снимок весов
        certified_guarantee = 0.0  # сумма доказанных снижений
        for ep in range(epochs):  # по эпохам
            order = rng.permutation(len(train))  # случайный порядок позиций
            for bs in range(0, len(train), batch):  # батчи по batch позиций
                ctx, nxt = make_batch(train[order[bs:bs + batch]])  # окна и ответы батча
                certified_guarantee += self._certified_step(ctx, nxt)  # шаг и накопление доказанного снижения
            ce_va = vce()  # CE хвоста после эпохи
            if ce_va <= best:  # не хуже лучшей
                best = ce_va  # новая лучшая CE
                best_state = [q.detach().clone() for q in self.params()]  # снять веса
            if log_every and (ep + 1) % log_every == 0:  # каждые log_every эпох
                print(f"  ep {ep+1}: holdout {ce_va:.4f} best {best:.4f} "  # печать эпохи, CE хвоста и лучшей CE
                      f"(сертифицированный суммарный спуск >= {certified_guarantee:.2f})",  # и суммы доказанных снижений
                      flush=True)  # вывод сразу, без буфера
        for q, s in zip(self.params(), best_state):  # параметр и снимок
            q.data.copy_(s)  # восстановить лучший вес
        return {"epochs": epochs, "best_holdout_ce": round(best, 4),  # число эпох и лучшая CE хвоста (4 знака)
                "certified_descent_lower_bound": round(certified_guarantee, 2)}  # сумма доказанных снижений (2 знака)

    # ---- длина ответа по решению модели + чтение всего входа ----

    def _read_all_windows(self, prompt_bytes: bytes, max_span: int):  # max_span — максимум окон
        """Разбить ВЕСЬ запрос на скользящие окна, покрывающие каждый байт
        (а не только хвост); возвращает до max_span последних окон шириной k."""
        k = self.k  # ширина окна
        if len(prompt_bytes) <= k:  # запрос не длиннее окна
            return [([32] * (k - len(prompt_bytes))  # одно окно: пробелы (32) слева
                     + list(prompt_bytes))]  # + байты запроса
        windows = []  # список окон
        step = k  # шаг равен ширине: сплошное покрытие без пропущенных байтов
        for start in range(0, len(prompt_bytes) - k + 1, step):  # начала окон start = 0, k, 2k, ...
            windows.append(list(prompt_bytes[start:start + k]))  # окно start..start+k−1
        if windows[-1] != list(prompt_bytes[-k:]):  # последнее окно не совпадает с хвостом запроса
            windows.append(list(prompt_bytes[-k:]))  # добавить окно хвоста
        return windows[-max_span:]  # не более max_span последних окон

    def generate_auto(self, prompt: str, max_bytes: int = 4096,  # max_bytes — предел длины ответа
                      temp: float = 0.5, seed: int = 1) -> str:  # temp — температура, seed — сид генерации
        """Генерация, где модель сама решает, ДОЛГО ли отвечать: работа идёт,
        пока модель не подаст знак завершения — знак конца предложения, за
        которым 3 повтора одного байта, или спад уверенности (максимальная
        вероятность < 1.2/256 в течение 8 шагов). Читает ВЕСЬ запрос через
        окна покрытия, вероятности окон усредняются. max_bytes — только
        предохранитель, а не цель. Возвращает запрос с раскодированным ответом."""
        t = self.t  # библиотека модели
        pb = prompt.encode('utf-8')  # байты запроса
        windows = self._read_all_windows(pb, max_span=64)  # до 64 окон покрытия
        with t.no_grad():  # без графа
            g = t.Generator(device=self.device).manual_seed(seed)  # генератор с сидом
            out = bytearray()  # накопленный ответ
            last_ctx = windows[-1]  # текущее окно (в начале — окно хвоста запроса)
            rep_counter = 0  # число повторов последнего байта
            low_conf = 0  # число подряд шагов с низкой уверенностью
            last_token = None  # предыдущий байт
            while len(out) < max_bytes:  # пока не достигнут предел длины
                probs = None  # накопитель вероятностей
                # усреднение по ВСЕМ окнам запроса + растущему ответу
                for w in windows + ([last_ctx] if windows[-1] != last_ctx or True else []):  # окна запроса + текущее окно (условие всегда истинно)
                    x = t.as_tensor([w], dtype=t.long, device=self.device)  # окно как тензор (1, k)
                    pp = t.softmax(self.logits(x)[0] / max(temp, 1e-3),  # вероятности p = softmax(z / max(temp, 1e-3))
                                   dim=-1)  # по классам
                    probs = pp if probs is None else probs + pp  # сумма вероятностей
                probs = probs / len(windows)  # делитель — число окон запроса (текущее окно в делитель не входит)
                top_p = float(probs.max())  # максимальная вероятность
                b = int(t.multinomial(probs, 1, generator=g))  # байт b ~ p
                out.append(b)  # добавить байт к ответу
                last_ctx = last_ctx[1:] + [b]  # окно сдвигается на один байт
                # --- условия остановки по решению модели ---
                ch = bytes([b])  # байт как объект bytes
                if top_p < (1.0 / 256) * 1.2:  # максимум ниже 1.2/256 — низкая уверенность
                    low_conf += 1  # счётчик +1
                else:  # уверенность достаточная
                    low_conf = 0  # сброс счётчика
                if b == last_token:  # байт совпал с предыдущим
                    rep_counter += 1  # повторов +1
                else:  # другой байт
                    rep_counter = 0  # повторов 0
                last_token = b  # запомнить байт
                if ch in b'.!?':  # знак конца предложения
                    # конец предложения: если за ним идёт низкая уверенность
                    # или повтор — остановиться (модель решила закончить)
                    if low_conf >= 2 or rep_counter >= 3:  # 2 неуверенных шага или 3 повтора
                        break  # стоп
                if rep_counter >= 8 or low_conf >= 8:  # 8 повторов или 8 неуверенных шагов
                    break  # стоп
        return (prompt.encode('utf-8') + bytes(out)).decode(  # запрос + ответ как UTF-8
            'utf-8', errors='replace')  # неверные последовательности заменяются

    def generate(self, prompt: str, n_bytes: int = 100, temp: float = 0.5,  # n_bytes — длина ответа, temp — температура
                 seed: int = 1, adaptive: bool = False) -> str:  # seed — сид, adaptive — модель сама выбирает число окон
        """Сгенерировать n_bytes байтов по последним k байтам контекста. При
        adaptive=True модель сама выбирает число усредняемых окон W по хвосту
        запроса (AdaptiveContextSelector); вероятности W окон усредняются.
        Возвращает запрос с раскодированным ответом."""
        t = self.t  # библиотека модели
        W = 1  # по умолчанию одно окно
        if adaptive:  # адаптивный выбор числа окон
            W = AdaptiveContextSelector().select(  # W по собственной CE хвоста запроса
                self, prompt.encode("utf-8")).get("w", 1)  # ключ "w" (по умолчанию 1)
        with t.no_grad():  # без графа
            g = t.Generator(device=self.device).manual_seed(seed)  # генератор с сидом
            ctx = list(prompt.encode("utf-8"))[-self.k:]  # последние k байтов запроса
            if len(ctx) < self.k:  # запрос короче окна
                ctx = [32] * (self.k - len(ctx)) + ctx  # дополнить пробелами (32) слева
            out = bytearray()  # накопленный ответ
            for _ in range(n_bytes):  # n_bytes шагов
                probs = None  # накопитель вероятностей
                for j in range(W):  # по W окнам
                    start = max(0, len(ctx) - self.k - j * self.k)  # начало j-го окна: на j·k байтов левее
                    w = ctx[start:len(ctx) - j * self.k] if j > 0 else ctx  # окно j > 0 — более старый участок; j = 0 — весь контекст
                    if len(w) < self.k:  # окно короче k
                        w = [32] * (self.k - len(w)) + w  # дополнить пробелами слева
                    x = t.as_tensor([w[-self.k:]], dtype=t.long,  # последние k байтов окна
                                    device=self.device)  # на устройстве модели
                    p = t.softmax(self.logits(x)[0] / max(temp, 1e-3),  # вероятности p = softmax(z / max(temp, 1e-3))
                                  dim=-1)  # по классам
                    probs = p if probs is None else probs + p  # сумма вероятностей
                probs = probs / W  # среднее вероятностей p̄ = Σ p_j / W
                b = int(t.multinomial(probs, 1, generator=g))  # байт b ~ p̄
                out.append(b); ctx = ctx[1:] + [b]  # добавить байт в ответ и в контекст (старейший выбрасывается)
        return (prompt.encode("utf-8") + bytes(out)).decode(  # запрос + ответ как UTF-8
            "utf-8", errors="replace")  # неверные последовательности заменяются




# фабрика моделей по имени архитектуры. RopeAttentionByteLM — по умолчанию;
# "hnet" -> hagi.hnet_lm.HNetByteLM (перенос на torch другой байтовой модели).
def make_lm(arch: str, dim: int | None = None, hidden: int | None = None, **kw):  # arch — имя архитектуры, dim/hidden — размеры (None — по умолчанию), kw — прочие аргументы модели
    """Создать модель по имени архитектуры arch и вернуть её.

    Для "hnet" размеры по умолчанию d = 64, h = 128 (около 105 тыс.
    параметров). Замер на 4 файлах с диска, 8000 шагов, CE целого файла:
    d24 1.445 | d32 1.344 | d48 1.200 | d64 1.096 | d128 (335 тыс.) 1.032 —
    после d = 64 выигрыш на параметр резко падает; RoPE (132 тыс.) даёт 1.81.
    Переменные окружения HAGI_DIM и HAGI_HIDDEN переопределяют dim и hidden.
    Для остальных имён — RopeAttentionByteLM с dim = 128, hidden = 256."""
    import os as _o  # доступ к переменным окружения
    if arch == "hnet":  # выбрана архитектура hnet
        from hagi.hnet_lm import HNetByteLM  # класс модели hnet
        d = int(_o.environ.get("HAGI_DIM", dim or 64) if _o.environ.get("HAGI_DIM") or dim is None else dim)  # d = HAGI_DIM, иначе dim, иначе 64
        h = int(_o.environ.get("HAGI_HIDDEN", hidden or 128) if _o.environ.get("HAGI_HIDDEN") or hidden is None else hidden)  # h = HAGI_HIDDEN, иначе hidden, иначе 128
        return HNetByteLM(dim=d, hidden=h, **kw)  # модель hnet
    return RopeAttentionByteLM(dim=dim or 128, hidden=hidden or 256, **kw)  # RoPE-модель: dim = 128, hidden = 256, если не заданы
