#!/usr/bin/env python3
"""Обучение «грамотности» по выровненным строкам: каждый обучающий контекст начинается
с начала строки, так что модель всегда видит узор целиком.

    Спуск — шаги Adam с тем же смыслом, что у CertifiedStepLM. Одна строка корпуса —
    один пример: префикс из 64 предыдущих байтов + строка; вход x[:-1], цели x[1:];
    потеря = Σ CE по последним (len(line) − 1) позициям / число таких позиций.
    Примеры делятся на обучение и контроль (около 1/7 — контроль); лучшая модель
    по контрольной CE сохраняется в checkpoints/literacy_lines.pt."""
import sys, os, time  # аргументы, окружение, время
from pathlib import Path  # пути
ROOT = Path(__file__).resolve().parents[2]  # корень проекта — на два уровня выше файла
sys.path.insert(0, str(ROOT))  # добавить корень проекта в путь импорта
os.environ.setdefault("HAGI_DEVICE", "cuda")  # устройство по умолчанию — cuda (если не задано иначе)
import numpy as np  # массивы
import torch  # тензоры
from hagi.lm import RopeAttentionByteLM  # байтовая модель с RoPE-вниманием

CORPUS = ROOT / "training" / "literacy" / "bits_corpus.txt"  # файл корпуса
CKPT = ROOT / "checkpoints" / "literacy_lines.pt"  # файл сохраняемой модели

def main():
    """Обучить модель на строках корпуса, сохранить лучшую по контрольной CE и проверить 6 примеров."""
    data = CORPUS.read_bytes()  # прочитать корпус как байты
    idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)  # байты как массив целых чисел
    # позиции начала строк
    starts = [0] + [i + 1 for i, b in enumerate(data) if b == 0x0A]  # начало 0 и позиция после каждого перевода строки 0x0A
    starts = [s for s in starts if s + 1 < len(idx)]  # оставить только начала, за которыми есть хотя бы один байт
    print(f"corpus {len(data)} bytes, {len(starts)} line starts")  # печать числа байтов корпуса и числа начал строк
    lm = RopeAttentionByteLM(dim=256, hidden=512, seed=1)  # модель d = 256, h = 512, зерно 1
    print("device:", lm.device)  # печать устройства модели
    t = lm.t  # библиотека torch из модели
    opt = t.optim.Adam(lm.params(), lr=8e-4)  # Adam со скоростью 8e-4 по параметрам модели
    rng = np.random.default_rng(7)  # генератор с зерном 7
    n = len(starts)  # число начал строк
    hold = set(rng.choice(n, max(16, n // 7), replace=False))  # контрольные индексы начал: max(16, ⌊n/7⌋) штук без повторов
    train = [i for i in range(n) if i not in hold]  # обучающие индексы — остальные
    dev = lm.device  # устройство модели
    def batch(positions):
        """Заготовка пакетирования по позициям начал (не используется: пакет строится ниже)."""
        X, Y = [], []  # списки входов и целей (не заполняются)
        for p in positions:  # по номерам начал строк
            s = starts[p]  # позиция начала строки
            ctx = idx[max(0, s - 64):s + 1]  # срез байтов max(0, s − 64) .. s (не используется)
            # используются окна: предсказать байты s..s+L−1 по предшествующему контексту
        return None  # заготовка не используется — пакет строится ниже
    # проще: каждая строка целиком — один пример: ctx = line[:-1],
    # targets = line[1:], а 64 предыдущих байта — префикс памяти
    samples = []  # список примеров (префикс, строка)
    for s in starts:  # по началам строк
        e = s  # конец строки e — идёт вперёд от начала
        while e < len(idx) and idx[e] != 0x0A:  # пока не встретится перевод строки
            e += 1  # сдвиг конца строки на 1 байт вперёд: e := e + 1
        line = idx[s:e + 1]  # строка вместе с переводом строки: idx[s : e]
        if len(line) < 4:  # строки короче 4 байтов пропускаются
            continue  # перейти к следующей строке
        prefix = idx[max(0, s - 64):s]  # префикс — до 64 байтов до начала строки
        samples.append((prefix, line))  # добавить пример (префикс, строка)
    print(f"line samples: {len(samples)}")  # печать числа примеров
    hold_s = set(rng.choice(len(samples), max(8, len(samples)//7), replace=False))  # контрольные номера примеров: max(8, ⌊N/7⌋)
    tr = [s for i, s in enumerate(samples) if i not in hold_s]  # обучающие примеры
    va = [s for i, s in enumerate(samples) if i in hold_s]  # контрольные примеры

    def vce():
        """Контрольная CE: сумма CE по контрольным примерам / суммарное число предсказываемых байтов."""
        tot = cnt = 0  # сумма потерь и число позиций
        with t.no_grad():  # без градиентов
            for prefix, line in va:  # по контрольным примерам
                ids = torch.as_tensor(np.concatenate([prefix, line]),  # последовательность префикс + строка в тензор целых
                                      dtype=torch.long, device=dev)  # тип long на устройстве модели
                L = len(line) - 1  # число предсказываемых байтов L = len(line) − 1
                logits = lm.full_logits(ids[:-1].unsqueeze(0))[0, -L:, :]  # логиты последних L позиций: для каждой предсказывается следующий байт
                tgt = torch.as_tensor(line[1:], dtype=torch.long, device=dev)  # цели — строка без первого байта
                tot += float(t.nn.functional.cross_entropy(  # сумма CE по позициям (без усреднения)
                    logits, tgt, reduction='sum'))  # продолжение: логиты, цели и сумма по позициям
                cnt += L  # число позиций cnt += L
        return tot / max(1, cnt)  # средняя CE = tot / max(1, cnt)

    best = vce()  # начальная лучшая контрольная CE
    print(f"init holdout CE {best:.4f}")  # печать начальной контрольной CE с 4 знаками
    t0 = time.time()  # время начала обучения
    for ep in range(150):  # 150 эпох
        order = rng.permutation(len(tr))  # случайный порядок обучающих примеров
        for bs in range(0, len(tr), 32):  # пакеты по 32 примера
            sel = order[bs:bs + 32]  # номера примеров пакета
            ids, tgts, lens = [], [], []  # последовательности, длины (tgts не используется)
            for i in sel:  # по примерам пакета
                prefix, line = tr[i]  # пример (префикс, строка)
                ids.append(np.concatenate([prefix, line]))  # последовательность префикс + строка
                lens.append(len(line) - 1)  # число предсказываемых байтов len(line) − 1
            maxlen = max(len(x) for x in ids)  # длина самой длинной последовательности
            X = np.full((len(ids), maxlen - 1), 32)  # входы: матрица (число примеров, maxlen − 1), заполненная пробелом (байт 32)
            Y = np.full((len(ids), maxlen - 1), 32)  # цели: та же форма, заполненная пробелом
            for j, (x, L) in enumerate(zip(ids, lens)):  # по парам (последовательность, длина)
                x_in, y_out = x[:-1], x[1:]  # вход — всё кроме последнего байта, цель — всё кроме первого
                X[j, -len(x_in):] = x_in  # вход прижат к правому краю строки матрицы
                Y[j, -len(y_out):] = y_out  # цель прижата к правому краю
                # позиции префикса маскируются (в loss не входят)
            Xt = torch.as_tensor(X, dtype=torch.long, device=dev)  # входы в тензор
            Yt = torch.as_tensor(Y, dtype=torch.long, device=dev)  # цели в тензор
            loss_mask = torch.zeros_like(Yt, dtype=torch.bool)  # маска потерь — сначала все False
            for j, L in enumerate(lens):  # по примерам
                loss_mask[j, -L:] = True  # учитываются только последние L позиций — сама строка
            logits = lm.full_logits(Xt)  # логиты для всего пакета
            loss = t.nn.functional.cross_entropy(  # CE каждой позиции без усреднения: форма (число позиций, 256) против целей
                logits.view(-1, 256), Yt.view(-1), reduction='none')  # продолжение: логиты (число позиций, 256), цели и CE каждой позиции
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()  # средняя CE по позициям строки: Σ loss·mask / Σ mask
            opt.zero_grad()  # обнулить градиенты
            loss.backward()  # обратный проход
            opt.step()  # шаг оптимизатора
        if ep % 20 == 0 or ep == 199:  # каждые 20 эпох (эпохи кратны 20); ep == 199 недостижимо при 150 эпохах
            v = vce()  # контрольная CE
            print(f"ep {ep}: holdout CE {v:.4f} ({time.time()-t0:.0f}s)", flush=True)  # печать эпохи, контрольной CE и секунд от начала обучения
            if v < best:  # если CE улучшилась
                best = v  # запомнить лучшую CE
                torch.save({nm: getattr(lm, nm).detach().cpu()  # сохранить тензоры emb, q, W1, b1, W2, b2 на CPU
                            for nm in ("emb", "q", "W1", "b1", "W2", "b2")}  # генератор словаря: nm -> тензор параметра без градиента на CPU
                           | {"dim": lm.dim, "hidden": lm.hidden,  # объединение со словарём гиперпараметров: dim, hidden, …
                              "theta0": lm.theta0, "lam": lm.lam}, CKPT)  # продолжение: theta0, lam; сохранить в CKPT
    print(f"best CE {best:.4f}; saved {CKPT}")  # печать лучшей CE и пути сохранённой модели
    # проверка грамотности (argmax при низкой температуре)
    from hagi.lm import load_rope  # загрузка модели rope
    lm2 = load_rope(CKPT)  # загрузить сохранённую модель
    lm2.device = lm.device  # устройство загруженной модели := устройство обученной
    for nm in lm2.params():  # по параметрам загруженной модели
        nm.data = nm.data.to(lm2.device)  # перенести параметр на устройство модели
    ok = 0  # число верных ответов
  # формат задач совпадает со сгенерированным корпусом (4-битные таблицы)
    tests = [("0101 XOR 0011 = ", "0110"), ("0101 AND 0011 = ", "0001"),  # первые две проверки: вопрос -> ожидаемый ответ
             ("1100 XOR 1010 = ", "0110"), ("NOT 0101 = ", "1010"),  # ещё две проверки: вопрос -> ожидаемый ответ
             ("6 XOR 3 = ", "5"), ("CHR 97 ", "HEX 61")]  # 6 проверок
    for tt, want in tests:  # по парам (вопрос, ответ)
        out = lm2.generate(tt, n_bytes=len(want)+2, temp=0.1)  # сгенерировать len(want) + 2 байтов при температуре 0,1
        gen = out[len(tt):].strip()  # ответ — сгенерированная часть без вопроса, без крайних пробелов
        hit = gen.startswith(want)  # ответ начинается с ожидаемого
        ok += hit  # верные ответы + hit (1 или 0)
        print(f"  {tt!r} -> {gen!r} want {want!r} {'OK' if hit else 'MISS'}")  # печать вопроса, ответа, ожидания и вердикта
    print(f"LITERACY: {ok}/{len(tests)}")  # печать числа верных ответов и числа проверок

if __name__ == "__main__":  # истинно, когда файл запущен как сценарий
    main()  # вызов основной функции
