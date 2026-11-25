#!/usr/bin/env python3  # интерпретатор Python 3
"""HAGI как архиватор без потерь: языковая модель — источник распределения для арифметического кодера.

Команды (compress — сжать, decompress — распаковать, третья команда — замер на кусках):
  python scripts/hagi_archive.py compress   IN OUT [--domain NAME] [--prefix-head 0.8]
  python scripts/hagi_archive.py decompress IN OUT [--domain NAME] [--prefix FILE]
  python scripts/hagi_archive.py <имя команды замера> [N]     # отношение размера архива к zlib на реальных кусках

--prefix-head F : первые F·len(файла) байт остаются вне архива и подаются декодеру
  как известный префикс -> потоковый кодек HAGI4 с таблицей контекстов разных
  порядков; если обычный архив HAGI3 меньше — берётся он. --prefix FILE отдаёт
  этот префикс обратно при распаковке. Коэффициент измеряется в битах на байт:
  b/B = 8·размер_архива / размер_исходника.

Архив раскодирует ТОЛЬКО та же модель: содержимое привязано к её весам.
"""  # документация модуля
import argparse  # разбор аргументов командной строки
import hashlib  # хеш sha256 для контроля распаковки
import os  # путь к корню проекта
import pathlib  # работа с путями и файлами
import sys  # путь импорта
import time  # замер длительности сжатия и распаковки
import zlib  # стороннее сжатие для сравнения

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # добавить корень проекта в путь импорта


def main():
    """Разобрать команду и выполнить сжатие, распаковку или замер на кусках доменов."""
    ap = argparse.ArgumentParser()  # разборщик аргументов
    ap.add_argument("cmd", choices=["compress", "decompress", "bench"])  # команда: сжать, распаковать или замерить
    ap.add_argument("paths", nargs="*")  # остальные позиционные аргументы: пути либо число файлов N
    ap.add_argument("--domain", default=None)  # имя адаптера домена (по умолчанию нет)
    ap.add_argument("--prefix-head", type=float, default=None,  # доля файла, оставляемая известным префиксом
                    help="fraction of the file kept as a known prefix (stream codec)")  # подсказка для пользователя
    ap.add_argument("--prefix", default=None, help="prefix file for decompress")  # файл с префиксом для распаковки
    a = ap.parse_args()  # разобрать аргументы командной строки
    import numpy as np  # численные массивы
    from hagi.models import archiver as A  # кодек арифметического сжатия
    from hagi.selfdev.unified import Unified  # объединённая модель
    u = Unified.load()  # загрузить модель с диска
    if a.cmd == "compress":  # команда — сжатие
        import pathlib  # работа с путями
        data = pathlib.Path(a.paths[0]).read_bytes()  # байты исходного файла
        if a.prefix_head:  # задана доля префикса (истинно, если не None и не 0)
            cut = int(len(data) * a.prefix_head)  # граница префикса cut = ⌊len(data)·F⌋
            pathlib.Path(a.paths[1] + ".prefix").write_bytes(data[:cut])  # записать префикс data[0:cut] рядом с архивом
            payload = A.compress_stream(u.model, data[cut:], prefix=data[:cut], lam=0.70)  # потоковый архив хвоста data[cut:] с известным префиксом
            pathlib.Path(a.paths[1]).write_bytes(payload)  # записать архив
            raw, out = len(data) - cut, len(payload)  # raw = длина хвоста, out = длина архива
            print(f"compressed {a.paths[0]} -> {a.paths[1]} (tail {raw:,} B, prefix {cut:,} B kept in {a.paths[1]}.prefix): "  # печать исходника, архива, хвоста и префикса
                  f"{out:,} B ({out * 8 / max(raw, 1):.3f} b/B, format {payload[:5]!r})")  # печать размера, b/B = 8·out/max(raw, 1) и сигнатуры payload[:5]
        else:  # обычный архив без префикса
            raw, out = A.compress_file(u, a.paths[0], a.paths[1], a.domain)  # сжать файл; raw и out — длины исходника и архива
            print(f"compressed {a.paths[0]} -> {a.paths[1]}: {raw:,} -> {out:,} B ({out * 8 / max(raw, 1):.3f} b/B)")  # печать размеров и b/B = 8·out/max(raw, 1)
    elif a.cmd == "decompress":  # команда — распаковка
        import pathlib  # работа с путями
        if a.prefix:  # задан файл с префиксом
            payload = pathlib.Path(a.paths[0]).read_bytes()  # байты архива
            data = A.decompress_stream(u.model, payload, prefix=pathlib.Path(a.prefix).read_bytes())  # раскодировать поток с тем же префиксом
            pathlib.Path(a.paths[1]).write_bytes(data)  # записать восстановленные байты
        else:  # обычный архив
            data = A.decompress_file(u, a.paths[0], a.paths[1], a.domain)  # распаковать архив в файл, вернуть исходные байты
        print(f"decompressed -> {a.paths[1]}: {len(data):,} B sha256={hashlib.sha256(data).hexdigest()}")  # печать длины и хеша sha256 восстановленных данных
    else:  # команда — замер на реальных кусках
        n = int(a.paths[0]) if a.paths else 10  # число файлов N (по умолчанию 10)
        files = sorted(pathlib.Path(u.__class__.__module__ and "checkpoints/domains").glob("ns_*.bin"))[-n:]  # последние N (по имени) кусков ns_*.bin; выражение в скобках всегда равно "checkpoints/domains"
        print(f"{'file':14} {'raw':>8} {'ours':>8} {'b/B':>6} {'zlib b/B':>8} {'lossless':>8} {'c/s':>6} {'d/s':>6}")  # строка заголовков столбцов
        for f in files:  # по выбранным кускам
            d = f.read_bytes()[:65536]  # первые 65536 байтов куска
            t0 = time.perf_counter(); c = A.compress_bytes(u, d); t1 = time.perf_counter()  # сжать d: c — архив; t0, t1 — время до и после
            back = A.decompress_bytes(u, c); t2 = time.perf_counter()  # распаковать архив: back; t2 — время после
            z = len(zlib.compress(d, 9))  # длина сжатия zlib уровня 9
            print(f"{f.name:14} {len(d):8} {len(c):8} {len(c) * 8 / len(d):6.3f} {z * 8 / len(d):8.3f} "  # имя, длины и b/B = 8·длина_архива/len(d) для модели и zlib
                  f"{str(back == d):>8} {len(d) / (t1 - t0):6.0f} {len(d) / (t2 - t1):6.0f}")  # без потерь (back == d), скорость сжатия len(d)/(t1 − t0) и распаковки len(d)/(t2 − t1) байт/с


if __name__ == "__main__":  # запуск только как сценарий
    main()  # выполнить команду
