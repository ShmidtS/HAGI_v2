"""Однострочный снимок здоровья цикла обучения: читает журнал цикла и печатает
число нейронных чанков, время последнего и медианного чанка, число диалогов и
критик, первый и последний семантический лосс, число сбоев.

Журнал пишется процессом цикла в кодировке UTF-16 LE (2 байта на символ,
младший байт первым), поэтому файл читается как байты и декодируется вручную.

Строки в журнале, на которые опирается разбор:
  [ns_N] neural: +X.XMB (F files) → CE C.CC (T.Ts)
  semantic loss S.SS
  [chat] ... : critique stream ...
Итоговая строка: fresh = now − mtime(журнал); median = отсортированные
времена чанков, элемент с индексом n // 2 (верхняя медиана).
"""
import os, re, time  # os — путь и время изменения файла, re — поиск по журналу, time — текущее время
LOG = os.path.join(os.path.dirname(os.path.dirname(  # путь к журналу: <корень проекта>/checkpoints/cycle.log, корень = каталог над scripts/
    os.path.abspath(__file__))), "checkpoints", "cycle.log")  # __file__ → абсолютный путь, dirname дважды: файл → scripts → корень
raw = open(LOG, "rb").read()  # байты журнала целиком (режим "rb" — двоичный, без перекодировки)
txt = raw.decode("utf-16-le", errors="ignore")  # текст = декодирование UTF-16 LE; сбойные байты отбрасываются
ns = re.findall(r"\[(ns_\d+)\] neural: .*?CE (\d+\.\d+) \((\d+\.\d)s\)", txt)  # все строки нейронного шага: (id чанка ns_N, CE, секунды); CE = cross-entropy на чанке
secs = sorted(float(s) for _, _, s in ns)  # времена чанков по возрастанию: t_(1) ≤ … ≤ t_(n)
sem = re.findall(r"semantic loss (\d+\.\d+)", txt)  # значения семантического лосса в порядке появления (строками)
fresh = int(time.time() - os.path.getmtime(LOG))  # давность журнала в целых секундах: int(текущее время − время изменения файла)
print(f"fresh {fresh}s | chunks {len(ns)} last {ns[-1] if ns else None} | "  # печать: давность | число чанков n | последний чанк (кортеж id, CE, секунды) или None, если чанков нет
      f"median {secs[len(secs) // 2] if secs else 0:.0f}s | "  # медиана времени чанка = secs[n // 2] (верхняя медиана), 0 при пустом списке, формат без дробной части
      f"chat {txt.count('[chat]')} crit {txt.count('critique stream')} | "  # число вхождений подстроки "[chat]" и "critique stream" в тексте
      f"sem {sem[:1]}->{sem[-1:]} | err {txt.count('Traceback')}")  # первый [sem[0]] и последний [sem[-1]] лосс (одноэлементные срезы, пустой список если нет) | число сбоев по слову "Traceback"
