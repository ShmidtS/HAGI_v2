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


def train_budget(m, data: bytes, budget_s: float, gpm_flag: bool = False, replay: bytes = b"") -> dict:
    """Обучать m на data[:80%] до бюджета времени; вернуть лучший holdout (последние 10% обучающей части).

        Для причинной модели (optimal) используется ЕЁ рабочий алгоритм — train_optimal (Muon/AdamW,
        перенос состояния, расписание шага), а не посторонний Adam: бенчмарк измеряет то, чем
        учится цикл. replay — байты прежних доменов; они чередуются с новыми (_interleave_replay)."""
    if hasattr(m, "forward_seq"):  # рабочая причинная модель
        from hagi.selfdev.unified import _interleave_replay  # чередование повтора с новым доменом
        cut = int(len(data) * 0.8)  # обучающая часть нового домена
        combined = _interleave_replay(data[:cut], replay)  # новый домен вперемешку с повтором; хвост — чисто новый (контроль)
        steps = max(30, int(budget_s * 35))  # число шагов из бюджета времени (замер: ~35 шагов/с на GPU)
        with t.no_grad():  # снимок на случай отката
            snap = [p.detach().clone() for p in m.params()]  # копии параметров
        per_epoch = max(1, (len(combined) - m.k - 1) // (16 * m.k))  # шагов на один проход данных (как в train_optimal)
        r = m.train_optimal(combined, epochs=max(1, -(-steps // per_epoch)), batch=16, seq_len=m.k, seed=0, gpm_left=(getattr(m, "gpm_left", None) if gpm_flag else {}))  # рабочий шаг обучения
        return {"holdout": float(r["val_ce_after"]), "steps": int(r["steps"]), "sec": budget_s}  # лучший контроль и число шагов
    raise RuntimeError('arch_bench: не-causal модель без train_optimal больше не поддерживается')  # прочие архитектуры: вне этого бенчмарка


def main() -> None:  # точка входа
    ap = argparse.ArgumentParser()  # разбор
    ap.add_argument("--budget", type=float, default=60.0)  # секунды на домен
    ap.add_argument("--archs", default="rope,hnet,optimal")  # архитектуры
    ap.add_argument("--gpm", action="store_true")  # GPM-проекция + Muon/AdamW для optimal
    ap.add_argument("--replay", action="store_true")  # повтор прежних доменов вперемешку с новым (30%)
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
        seen: list[bytes] = []  # уже пройденные домены (для повтора)
        mat = []  # матрица CE: строка — после шага i, столбец — домен j
        for name, data in doms:  # последовательное обучение
            rep_b = b"".join(d[: int(float(os.environ.get("HAGI_REPLAY_FRAC", "0.3")) * len(data) / max(1, len(seen)))] for d in seen) if (a.replay and seen) else b""  # повтор: 30% объёма нового, поровну из прежних доменов
            r = train_budget(m, data, a.budget, a.gpm, rep_b)  # обучение (с повтором, если задано)
            if a.gpm and hasattr(m, "forward_seq"):  # после домена — защитить его входы
                from hagi.models.optimal_lm import update_gpm_bases  # обновление базисов
                update_gpm_bases(m, data[: int(len(data) * 0.8)], m.gpm_left)  # SVD входов
            seen.append(data)  # домен стал прежним
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
