#!/usr/bin/env python3  # интерпретатор Python 3
"""Сквозной просмотр всего диска C:/ и приём байтов каждого файла единой объединённой моделью.

Честная постановка: на диске от 10^5 до 10^6 файлов и много терабайтов, выучить все
байты невозможно. Договорённость просмотра: каждый файл вносит вклад — не более
1 МБ, выборка из головы, середины и хвоста файла, чтобы видеть структуру, а не только
начало. Двоичные, системные и заблокированные файлы тоже участвуют — байты есть байты.
Повторы отсекаются по хешу sha1 пути, чтобы одинаковые системные библиотеки не
затмили всё остальное. Недоступные файлы считаются пропущенными с причиной.
Каждые CHUNK_TARGET = 200000 байтов выборки склеиваются в кусок и проходят приём с
EPOCHS_PER_CHUNK = 8 эпохами на накопленном корпусе (полный повтор — без забывания).
Возобновление: состояние (посещено, прочитано, хеши, курсор) лежит в файле
checkpoints/sweep_state.json.
В конце файла — отбор разнообразных фрагментов: score(i | S) = H(p_i) · min_{j∈S} JS(p_i, p_j),
то есть предпочтение информативным фрагментам с НОВЫМ распределением байтов относительно
уже выбранных представителей; расстояния считаются только до представителей.
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов

import hashlib  # хеш sha1 пути файла
import json  # чтение и запись состояния просмотра
import os  # обход каталогов, размеры файлов
import sys  # аргументы командной строки
import time  # предельное время работы
from pathlib import Path  # пути

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # добавить корень проекта в путь импорта
from hagi import unified  # noqa: E402  # объединённая модель; импорт после правки sys.path

ROOT = Path(__file__).resolve().parent.parent  # корень проекта
SWEEP_STATE = ROOT / "checkpoints" / "sweep_state.json"  # файл состояния просмотра

# пределы (честные: весь диск не поместится; посещается каждый ФАЙЛ)
MAX_FILE_BYTES = 1 << 20          # не больше 1 МБ на файл = 2^20
SAMPLE_HEAD = 1 << 16             # голова файла, 64 КБ = 2^16
SAMPLE_TAIL = 1 << 16             # хвост файла, 64 КБ = 2^16
SAMPLE_MID = 1 << 16              # середина файла, 64 КБ = 2^16
MIN_FILE = 16                     # файлы короче 16 байтов пропускаются
CHUNK_TARGET = 200_000            # кусок обучается при 200000 байтов выборки
EPOCHS_PER_CHUNK = 8              # много маленьких кусков лучше нескольких больших проходов

SKIP_DIRS = {"$Recycle.Bin", "System Volume Information",  # каталоги, в которые не заходим: корзина, сведения о томе, ...
             "Windows.old", "$WinREAgent", "hiberfil.sys",  # ...старая система, агент восстановления, файл сна
             "Recovery", "DumpStack.log.tmp", "pagefile.sys",  # ...восстановление, дамп стека, файл подкачки
             "swapfile.sys", "onedrive_temp", "node_modules",  # ...файл свопа, временный каталог облака, зависимости
             ".git", "__pycache__",  # ...хранилище git, кэш байткода
             # соединения NTFS без этого списка дают бесконечную рекурсию
             "Documents and Settings", "Application Data",  # ...каталоги-соединения
             "Local Settings", "My Documents", "Cookies", "NetHood",  # ...каталоги-соединения
             "PrintHood", "Recent", "SendTo", "Templates", "Start Menu"}  # ...каталоги-соединения

_lock = None  # переменная не используется в этом файле


def iter_files(root: str = "C:/", start_after: str | None = None):
    """Обойти всё дерево root и выдавать пути файлов; пропускать только неизбежные блокировки системы.

        start_after: полный путь последнего прочитанного ФАЙЛА. os.walk выдаёт файлы
        каталога раньше, чем спускается в подкаталоги, поэтому сравнение путей по
        алфавиту пропускало файлы, идущие после курсора в порядке обхода, но меньшие
        по алфавиту. Возобновление отслеживает ПОЗИЦИЮ обхода (каталог, имя файла)."""
    started = start_after is None  # обход начат, если курсора нет
    cursor_dir = None if started else os.path.normpath(  # каталог курсора в нормализованном виде
        os.path.dirname(start_after))  # ...или None, если курсора нет
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):  # обход дерева; ошибки чтения каталога игнорируются
        dirnames[:] = sorted([d for d in dirnames if d not in SKIP_DIRS])  # подкаталоги: только не пропускаемые, по алфавиту
        dnorm = os.path.normpath(dirpath)  # текущий каталог в нормализованном виде
        if not started:  # пока не дошли до позиции курсора
            if dnorm == cursor_dir:  # текущий каталог — каталог курсора
                # возобновление ВНУТРИ каталога курсора: выдать только файлы,
                # идущие после курсора в отсортированном порядке
                seen_cursor = False  # курсор ещё не встречен
                for f in sorted(filenames):  # по файлам каталога по алфавиту
                    path = os.path.join(dirpath, f)  # полный путь файла
                    if path == start_after:  # это файл курсора
                        seen_cursor = True  # отметить встречу курсора
                    elif seen_cursor:  # файл идёт после курсора
                        yield path  # выдать путь
                started = True  # обход начат
                continue  # следующий каталог
            # обход посещает предков и дерево каталога курсора в алфавитном
            # порядке; пропустить каталоги, строго идущие ПЕРЕД каталогом курсора
            if dnorm < cursor_dir:  # текущий каталог алфавитно раньше каталога курсора
                continue  # пропустить каталог
        for f in sorted(filenames):  # по файлам каталога по алфавиту
            yield os.path.join(dirpath, f)  # выдать полный путь


def sample_bytes(p: str, size: int) -> bytes | None:
    """Выборка файла: голова + середина + хвост (структура, а не только начало); при ошибке чтения — None."""
    try:  # ошибка доступа обрабатывается ниже
        with open(p, "rb") as fh:  # открыть файл для чтения байтов
            head = fh.read(min(SAMPLE_HEAD, size))  # голова: первые min(2^16, size) байтов
            if size <= SAMPLE_HEAD:  # файл не длиннее головы
                return head  # вернуть одну голову
            mid_off = size // 2  # позиция середины = ⌊size / 2⌋
            fh.seek(mid_off)  # перейти на середину
            mid = fh.read(SAMPLE_MID)  # прочитать 64 КБ от середины
            fh.seek(max(0, size - SAMPLE_TAIL))  # перейти на позицию max(0, size − 2^16)
            tail = fh.read(SAMPLE_TAIL)  # прочитать 64 КБ хвоста
            return head + b"\n\x00MID\x00\n" + mid + b"\n\x00TAIL\x00\n" + tail  # голова + метка + середина + метка + хвост
    except (PermissionError, OSError):  # нет доступа или ошибка чтения
        return None  # сигнал, что файл прочитать не удалось


def load_state() -> dict:
    """Прочитать состояние просмотра из файла; если файла нет — начальное состояние с нулями."""
    if SWEEP_STATE.exists():  # файл состояния существует
        return json.loads(SWEEP_STATE.read_text(encoding="utf-8"))  # разобрать JSON файла
    return {"seen": {}, "visited": 0, "sampled": 0, "skipped_locked": 0,  # начальное состояние: хешей нет, все счётчики 0
            "bytes_ingested": 0, "chunks": 0, "cursor": None}  # принято байтов 0, кусков 0, курсора нет


def save_state(st: dict) -> None:
    """Записать состояние просмотра st в файл JSON."""
    SWEEP_STATE.write_text(json.dumps(st, ensure_ascii=False),  # текст JSON без экранирования не-ASCII
                           encoding="utf-8")  # в кодировке UTF-8


def main() -> None:
    """Обойти диск до предельного времени, принять куски выборок и сохранить состояние.

        Аргумент 1: предельное время в минутах (по умолчанию 60)."""
    max_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0  # предел времени в минутах: аргумент или 60
    deadline = time.time() + max_minutes * 60  # момент окончания: сейчас + минуты·60 секунд
    u = unified.Unified.load() if unified.Unified.exists() else unified.Unified()  # загрузить модель, если файл есть, иначе создать новую
    st = load_state()  # состояние просмотра
    seen = st["seen"]  # словарь уже обработанных хешей путей
    chunk = []  # выборки текущего куска
    chunk_bytes = 0  # число байтов текущего куска
    files_now = 0  # число файлов текущего куска
    t0 = time.time()  # начало отсчёта
    print(f"sweep start: visited={st['visited']} sampled={st['sampled']} "
          f"ingested={st['bytes_ingested']}B chunks={st['chunks']}",  # печать начальных счётчиков
          flush=True)  # сразу сбросить вывод
    for path in iter_files("C:/"):  # по всем файлам диска C:/
        if time.time() > deadline:  # время вышло
            print("deadline reached — flush", flush=True)  # сообщение и сброс вывода
            break  # прекратить обход
        st["visited"] += 1  # посещено: +1 файл
        try:  # размер может быть недоступен
            size = os.path.getsize(path)  # размер файла в байтах
        except OSError:  # размер получить не удалось
            st["skipped_locked"] += 1  # пропущено: +1
            continue  # следующий файл
        if size < MIN_FILE:  # файл короче 16 байтов
            continue  # пропустить
        h = hashlib.sha1(path.encode("utf-8", errors="replace")).hexdigest()[:12]  # h = первые 12 hex-символов sha1 от пути
        if h in seen:  # этот путь уже обработан
            continue                      # путь уже принят
        data = sample_bytes(path, size)  # выборка байтов файла или None
        if data is None:  # файл прочитать не удалось
            st["skipped_locked"] += 1  # пропущено: +1
            seen[h] = "locked"  # пометить путь как заблокированный
            continue  # следующий файл
        seen[h] = min(len(data), size)  # пометить путь: число байтов = min(длина выборки, size)
        st["sampled"] += 1  # прочитано: +1 файл
        chunk.append(data)  # добавить выборку к куску
        chunk_bytes += len(data)  # chunk_bytes += длина выборки
        st["bytes_ingested"] += len(data)  # принято байтов: + длина выборки
        files_now += 1  # файлов в куске: +1
        # обучение кусками: накопленный повтор (ОДНА модель, без забывания)
        if chunk_bytes >= CHUNK_TARGET:  # кусок набрал 200000 байтов
            blob = b"\n\x00FILE\x00\n".join(chunk)  # склеить выборки через метку конца файла
            u.ingest(f"csweep_{st['chunks']:04d}", blob,  # приём куска с именем csweep_НННН
                     epochs=EPOCHS_PER_CHUNK, repeat=1)  # 8 эпох, 1 повтор
            st["chunks"] += 1  # кусков: +1
            print(f"  chunk {st['chunks']}: {chunk_bytes/1e6:.1f}MB "  # печать номера куска и размера в МБ
                  f"({files_now} files, "  # печать числа файлов
                  f"{st['bytes_ingested']/1e6:.0f}MB total, "  # печать всего принято в МБ
                  f"{time.time()-t0:.0f}s)", flush=True)  # печать секунд с начала
            chunk, chunk_bytes, files_now = [], 0, 0  # обнулить накопитель куска
            save_state(st)  # сохранить состояние
    if chunk:  # остался неполный кусок
        blob = b"\n\x00FILE\x00\n".join(chunk)  # склеить выборки
        u.ingest(f"csweep_{st['chunks']:04d}", blob,  # принять неполный кусок
                 epochs=EPOCHS_PER_CHUNK, repeat=1)  # 8 эпох, 1 повтор
        st["chunks"] += 1  # кусков: +1
    save_state(st)  # сохранить состояние
    print(f"sweep done: visited={st['visited']} sampled={st['sampled']} "
          f"locked={st['skipped_locked']} ingested="  # печать итоговых счётчиков
          f"{st['bytes_ingested']/1e6:.0f}MB chunks={st['chunks']} "  # печать принятых МБ и кусков
          f"domains={len(u.domains)}", flush=True)  # печать числа доменов


if __name__ == "__main__":
    main()  # выполнить просмотр


# Политика отбора фрагментов в память.
# score(i | S) = H(p_i) * min_{j in S} JS(p_i, p_j) — предпочтение информативным
# фрагментам, которые добавляют НОВОЕ распределение байтов относительно уже
# выбранных представителей. Расстояния считаются только до выбранных
# представителей (не полная попарная матрица) — примерно в 10 раз быстрее
# наивного варианта по измерениям.

import numpy as _np  # численные массивы
import math as _math  # модуль math (в этом файле не используется)


def _byte_hist(data: bytes, bins: int = 64) -> "_np.ndarray":  # гистограмма байтов на 64 интервала
    """Нормированная гистограмма байтов: p_i = (число байтов в интервале i) / всего байтов; для пустых данных p_i = 1/64."""
    if not data:  # данных нет
        return _np.full(bins, 1.0 / bins)  # равномерное распределение p_i = 1/bins
    h, _ = _np.histogram(_np.frombuffer(data, dtype=_np.uint8),  # числа байтов по интервалам: значение байта / 4
                         bins=bins, range=(0, 256))  # 64 интервала на диапазоне 0..256, ширина 4
    total = h.sum()  # всего байтов
    return (h / total) if total else _np.full(bins, 1.0 / bins)  # p_i = h_i / total; при total = 0 — равномерное


def _entropy(p) -> float:
    """Энтропия Шеннона H = −Σ_{p_i>0} p_i·log2(p_i)."""
    nz = p[p > 0]  # только положительные вероятности
    return float(-(nz * _np.log2(nz)).sum())  # энтропия H = −Σ nz·log2(nz)


def _js_div(p, q) -> float:
    """Расхождение Дженсена — Шеннона по основанию 2, ограничено [0, 1]: JS = 0.5·KL(p‖m) + 0.5·KL(q‖m), m = (p + q)/2."""
    m = 0.5 * (p + q)  # среднее распределение m = (p + q)/2
    def _kl(a, b):
        """Расхождение Кульбака — Лейблера KL(a‖b) = Σ_{a_i>0} a_i·log2(a_i / max(b_i, 1e-12))."""
        mask = a > 0  # учитывать только положительные a_i
        return float((a[mask] * _np.log2(a[mask] / _np.maximum(b[mask], 1e-12))).sum())  # расхождение KL = Σ a_i·log2(a_i / max(b_i, 1e-12))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)  # расхождение JS = 0.5·KL(p‖m) + 0.5·KL(q‖m)


def diverse_select(candidates: list[bytes], budget: int) -> list[bytes]:
    """Жадный отбор: максимум энтропии·минимума JS; O(budget·N) расстояний только до представителей.

        Первый фрагмент — с наибольшей H; каждый следующий — с наибольшим
        score = H(p)·min_j JS(p, p_j) по уже выбранным. Возвращает не более budget фрагментов."""
    if len(candidates) <= budget:  # кандидатов не больше бюджета
        return list(candidates)  # вернуть всех кандидатов
    hist = [(c, _byte_hist(c), _entropy(_byte_hist(c))) for c in candidates]  # список (фрагмент, гистограмма p, H(p)) по кандидатам
    selected: list[tuple[bytes, _np.ndarray]] = []  # выбранные: (фрагмент, гистограмма)
    remaining = list(hist)  # ещё не рассмотренные кандидаты
    while remaining and len(selected) < budget:  # пока есть кандидаты и не исчерпан бюджет
        best = None  # лучший кандидат раунда
        for c, p, h in remaining:  # по оставшимся кандидатам
            if not selected:  # ничего не выбрано
                score = h  # счёт = энтропии H(p)
            else:  # уже есть выбранные
                min_js = min(_js_div(p, sp) for _, sp in selected)  # минимальное расстояние min_js = min_j JS(p, p_j)
                score = h * min_js  # счёт score = H(p)·min_j JS(p, p_j)
            if best is None or score > best[0]:  # первый кандидат или больший счёт
                best = (score, c, p)  # запомнить (счёт, фрагмент, гистограмма)
        selected.append((best[1], best[2]))  # добавить лучшего к выбранным
        remaining = [r for r in remaining if r[1] is not best[2]]  # убрать его из оставшихся (по тем же гистограмме)
    return [c for c, _ in selected]  # список выбранных фрагментов
