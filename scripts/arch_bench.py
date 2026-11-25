"""Бенчмарк архитектур на чистых текстовых доменах: качество/параметр, время, забывание.

Домены — уникальные (по sha1) текстовые куски из checkpoints/domains, печатность > 0.9.
Для каждой архитектуры: обучение на 80% каждого домена до заданного бюджета времени,
CE на 20%-хвосте (вне выборки), затем последовательное обучение 4 доменов (без повтора)
и CE на всех четырёх после каждого шага — забывание. Использование:
    HAGI_DEVICE=cuda python -X utf8 scripts/arch_bench.py [--budget 60] [--archs rope,hnet,optimal]
"""
from __future__ import annotations  # аннотации не вычисляются при импорте

import argparse  # разбор аргументов
import glob  # поиск файлов
import hashlib  # дедупликация
import os  # пути
import time  # замеры

import numpy as np  # массивы
import torch as t  # тензоры


def pick_domains(n: int = 4, min_bytes: int = 60000) -> list[tuple[str, bytes]]:
    """Взять n уникальных текстовых доменов (печатность > 0.9, без дублей по sha1), как можно более разные."""
    cands = []  # кандидаты
    seen = set()  # хеши
    for p in sorted(glob.glob("checkpoints/domains/*.bin")):  # по файлам
        b = open(p, "rb").read()[:300000]  # первые 300 КБ
        if len(b) < min_bytes:  # мал
            continue  # дальше
        pr = sum(32 <= c < 127 or c in (9, 10, 13) or c >= 192 for c in b[:100000]) / min(len(b), 100000)  # печатность
        if pr < 0.9:  # не текст
            continue  # дальше
        h = hashlib.sha1(b).hexdigest()  # хеш содержимого
        if h in seen:  # дубль
            continue  # дальше
        seen.add(h); cands.append((os.path.basename(p), b))  # добавить
    # разнообразие: берём по очереди из разных концов списка
    step = max(1, len(cands) // n)  # шаг
    return [cands[i * step] for i in range(n)]  # n доменов


def oos_ce(m, data: bytes, tail: float = 0.2, stride: int = 3) -> float:
    """CE на хвосте data (вне выборки): окна k, шаг stride; для причинной модели — по всем позициям хвоста."""
    k = m.k  # контекст
    cut = int(len(data) * (1.0 - tail))  # граница
    seg = data[cut - k:] if cut > k else data  # хвост с k байтами контекста
    idx = np.frombuffer(seg, dtype=np.uint8).astype(np.int64)  # индексы
    with t.no_grad():  # без графа
        if hasattr(m, "forward_seq"):  # причинная модель: все позиции за проход
            x = t.as_tensor(idx[:-1], dtype=t.long, device=m.device)[None]; y = t.as_tensor(idx[1:], dtype=t.long, device=m.device)  # вход и цели
            return float(t.nn.functional.cross_entropy(m.forward_seq(x).reshape(-1, 256), y.reshape(-1)))  # кросс-энтропия
        pos = np.arange(k, len(idx), stride)  # позиции целей
        tot = cnt = 0.0  # сумма и счётчик
        for s in range(0, len(pos), 512):  # пакетами
            p = pos[s: s + 512]  # пакет
            X = t.as_tensor(idx[p[:, None] - k + np.arange(k)[None, :]], dtype=t.long, device=m.device); yy = t.as_tensor(idx[p], dtype=t.long, device=m.device)  # окна и цели
            tot += float(t.nn.functional.cross_entropy(m.logits(X), yy, reduction="sum")); cnt += len(p)  # накопить
        return tot / max(1.0, cnt)  # средняя


def train_budget(m, data: bytes, budget_s: float, gpm_flag: bool = False) -> dict:
    """Обучать m на data[:80%] до бюджета времени; вернуть лучший holdout (последние 10% обучающей части)."""
    k = m.k  # контекст
    cut = int(len(data) * 0.8)  # обучающая часть
    tr = data[:cut]  # обучение
    va = data[cut - k: cut + int(len(data) * 0.1)] if cut + int(len(data) * 0.1) <= len(data) else data[cut - k:]  # контроль (10%)
    best = float("inf"); best_state = None  # лучший holdout и снимок
    t0 = time.time(); steps = 0  # старт и счётчик
    arr = np.frombuffer(tr, dtype=np.uint8).astype(np.int64)  # индексы
    rng = np.random.default_rng(0)  # выборка
    gpm = None  # базисы GPM (None — обычный Adam)
    if gpm_flag and hasattr(m, "forward_seq"):  # optimal с GPM: оптимизатор модели с проекцией
        from hagi.models.optimal_lm import HybridMuonAdamW  # оптимизатор с проекцией
        gpm = getattr(m, "gpm_left", {}); m.gpm_left = gpm  # базисы хранятся на модели
        opt = HybridMuonAdamW(*m.matrix_and_vec_params(), lr_muon=0.03, lr_adam=5e-3, wd=1e-4); opt.gpm_left.update(gpm); opt.names.update({id(p): n for n, p in m.__dict__.items() if isinstance(p, t.nn.Parameter)})  # создать
    else:  # обычный Адам
        opt = t.optim.Adam(m.params(), lr=1e-3)  # Адам с явным шагом (одинаков для всех архитектур)
    while time.time() - t0 < budget_s:  # до бюджета
        if hasattr(m, "forward_seq"):  # причинная: окна T=64, потеря по всем позициям
            s = rng.integers(0, len(arr) - k - 1, 16); X = t.as_tensor(np.stack([arr[i: i + k] for i in s]), dtype=t.long, device=m.device); y = t.as_tensor(np.stack([arr[i + 1: i + 1 + k] for i in s]), device=m.device)  # батч
            loss = t.nn.functional.cross_entropy(m.forward_seq(X).reshape(-1, 256), y.reshape(-1))  # кросс-энтропия
        else:  # одна цель на окно
            s = rng.integers(k, len(arr) - 1, 64); X = t.as_tensor(np.stack([arr[i - k: i] for i in s]), dtype=t.long, device=m.device); y = t.as_tensor(arr[s + 1], dtype=t.long, device=m.device)  # батч
            loss = t.nn.functional.cross_entropy(m.logits(X), y)  # кросс-энтропия
        opt.zero_grad(); loss.backward(); opt.step(); steps += 1  # шаг
        if steps % 50 == 0:  # контроль
            c = oos_ce(m, va, tail=0.0 if False else 0.5, stride=4)  # CE на контрольном отрезке (его вторая половина — вне обучения)
            if c < best:  # лучше
                best = c; best_state = [p.detach().clone() for p in m.params()]  # снимок
    if best_state is not None:  # восстановить лучший
        with t.no_grad():
            for p, s_ in zip(m.params(), best_state): p.copy_(s_)  # вернуть
    return {"holdout": best, "steps": steps, "sec": time.time() - t0}  # итог


def main() -> None:  # точка входа
    ap = argparse.ArgumentParser()  # разбор
    ap.add_argument("--budget", type=float, default=60.0)  # секунды на домен
    ap.add_argument("--archs", default="rope,hnet,optimal")  # архитектуры
    ap.add_argument("--gpm", action="store_true")  # GPM-проекция + Muon/AdamW для optimal
    a = ap.parse_args()  # аргументы
    doms = pick_domains(4)  # домены
    print("домены:", [(n, len(b)) for n, b in doms], flush=True)  # отчёт
    from hagi.models.lm import make_lm  # фабрика
    from hagi.models.optimal_lm import quantize_model  # квантование
    for arch in a.archs.split(","):  # по архитектурам
        t0 = time.time(); m = make_lm(arch)  # модель по умолчанию (hnet 64/128; optimal 24/48; rope 128/256)
        if arch == "rope":  # rope крупнее — для честности по объёму дать близкий размер
            m = make_lm(arch, 64, 128)  # размеры как у hnet
        npar = sum(p.numel() for p in m.params())  # параметры
        qb = quantize_model(m, bits=4)[1] if hasattr(m, "forward_seq") else None  # байты Q4 (для optimal)
        ce = []  # CE по доменам после обучения на каждом
        mat = []  # матрица CE: строка — после шага i, столбец — домен j
        for name, data in doms:  # последовательное обучение
            r = train_budget(m, data, a.budget, a.gpm)  # обучение
            if a.gpm and hasattr(m, "forward_seq"):  # после домена — защитить его входы
                from hagi.models.optimal_lm import update_gpm_bases
                update_gpm_bases(m, data[: int(len(data) * 0.8)], m.gpm_left)  # SVD входов
            row = [oos_ce(m, d) for _, d in doms]  # CE на всех доменах
            mat.append(row); ce.append(r["holdout"])  # записать
            print(f"[{arch:8s}] {name:14s} holdout {r['holdout']:.3f} | шагов {r['steps']:5d} | CE по доменам {np.round(row, 2).tolist()}", flush=True)  # отчёт
        mat = np.array(mat)  # матрица
        final = mat[-1]  # итоговая строка
        diag = np.array([mat[i, i] for i in range(len(doms))])  # CE домена сразу после его обучения
        forget = float((final - diag).mean())  # забывание: итог − сразу после обучения
        print(f"== {arch:8s}: параметров {npar:,} | Q4 {qb and f'{qb:,} Б'} | финальная CE по доменам {np.round(final, 3).tolist()} | средняя {final.mean():.3f} нат ({final.mean() / 5.545:.3f} б/б) | забывание {forget:+.3f} | {time.time() - t0:.0f} с", flush=True)  # итог


if __name__ == "__main__":  # запуск
    main()  # вызвать
