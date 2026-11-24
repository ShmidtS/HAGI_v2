#!/usr/bin/env python3  # интерпретатор Python 3
"""Нейро-символьный цикл самообучения: ОДНА модель и схемы на её основе.

Итерация цикла:
  1. РОСТ (нейронное): просмотр C:/ -> выборка файлов -> сертифицированное
     обучение кусками на накопленном корпусе (одно тело).
  2. ВЫВОД (символьное): из обученного тела выводятся булевы задачи по тому,
     что только что выучено, — например структура знака строк W1 или предикаты
     следующего бита на энергичных байтах. Каждая задача — таблица истинности.
  3. СБОРКА схемы: сначала верность (правильность роста и нейтральный дрейф),
     затем оптимизация стоимости под жёстким запретом — схема собирается
     на задачу эволюцией с многократными перезапусками.
  4. ФИКСАЦИЯ: сертифицированные схемы дописываются в список схем процессора
     (processor_netlist.json); модель сохраняется.

Каждая фиксация несёт сертификат — исчерпывающую проверку таблицы истинности.
Без сертификата фиксации нет: процессор строится только из доказанных схем.
Параллельно обрабатывается ящик входящих сообщений checkpoints/inbox: файл .chat
изучается как диалог (или как поток байтов для критики), ответ пишется в .reply.
"""
from __future__ import annotations  # отложенное вычисление аннотаций типов

import json  # чтение и запись JSON
import sys  # аргументы командной строки
import time  # предельное время работы
from pathlib import Path  # пути

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # добавить корень проекта в путь импорта
from hagi import fabric, unified  # noqa: E402  # сборка схем и объединённая модель
from hagi.cycle import (compose_adder2, commit, derive_tasks,  # noqa: E402  # сборка двухбитного сумматора, фиксация, вывод задач
                        fusion_fastpath, weave_task)  # noqa: E402  # быстрый путь синтеза, сборка схемы
from hagi.util import CKPT_DIR  # каталог контрольных точек
from scripts.feed_all_c import (CHUNK_TARGET, SKIP_DIRS, iter_files,  # noqa: E402  # размер куска, пропускаемые каталоги, обход диска
                                load_state, save_state, sample_bytes)  # noqa: E402  # состояние просмотра и выборка файла

from hagi.util import CKPT_DIR  # повторный импорт каталога контрольных точек (импорт уже сделан выше)
TAPEOUT = CKPT_DIR / "processor_netlist.json"  # путь списка схем процессора (в этом файле далее не используется)

# ---------- ВЫВОД: задачи для схем ИЗ обученного тела ----------

def main() -> None:
    """Выполнять итерации «просмотр -> приём -> ящик сообщений -> семантика -> схемы -> фиксация» до предельного времени.

        Аргумент 1: предельное время в минутах (по умолчанию 480)."""
    max_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 480.0  # предел времени в минутах: аргумент или 480
    deadline = time.time() + max_minutes * 60  # момент окончания: сейчас + минуты·60 секунд
    # загрузить модель, если файл есть, иначе создать новую
    u = unified.Unified.load() if unified.Unified.exists() \
        else unified.Unified()  # ...иначе новая
    st = load_state()  # состояние просмотра диска
    seen = st["seen"]  # словарь уже обработанных хешей путей
    chunk, chunk_bytes = [], 0  # выборки текущего куска и число его байтов
    files_now = 0  # число файлов текущего куска
    it = 0  # число завершённых итераций
    (CKPT_DIR / 'inbox').mkdir(parents=True, exist_ok=True)  # создать каталог ящика входящих сообщений
    print(f"neuro-symbolic cycle start: chunks={st['chunks']}", flush=True)  # печать числа уже принятых кусков
    while time.time() < deadline:  # пока не истекло предельное время
        # ---- 1. РОСТ: набрать кусок с C:/ (постоянный курсор) ----
        last_path = None  # последний пройденный путь (курсор возобновления)
        for path in iter_files("C:/", start_after=st.get("cursor")):  # по файлам диска, начиная после курсора
            last_path = path  # запомнить путь как новый курсор
            if chunk_bytes >= CHUNK_TARGET or time.time() > deadline:  # кусок набран или время вышло
                break  # прекратить набор
            st["visited"] += 1  # посещено: +1 файл
            try:  # размер может быть недоступен
                size = __import__("os").path.getsize(path)  # размер файла в байтах (модуль os импортируется на месте)
            except OSError:  # размер получить не удалось
                continue  # следующий файл
            if size < 16:  # файл короче 16 байтов
                continue  # пропустить
            h = __import__("hashlib").sha1(  # h = первые 12 hex-символов sha1 от пути (модуль hashlib импортируется на месте)
                path.encode("utf-8", errors="replace")).hexdigest()[:12]  # путь в байтах UTF-8 с заменой ошибок
            if h in seen:  # путь уже обработан
                continue  # следующий файл
            data = sample_bytes(path, size)  # выборка байтов файла или None
            if data is None:  # файл прочитать не удалось
                seen[h] = "locked"  # пометить путь как заблокированный
                continue  # следующий файл
            seen[h] = len(data)  # пометить путь числом байтов выборки
            st["sampled"] += 1  # прочитано: +1 файл
            chunk.append(data)  # добавить выборку к куску
            chunk_bytes += len(data)  # chunk_bytes += длина выборки
            st["bytes_ingested"] += len(data)  # принято байтов: + длина выборки
            files_now += 1  # файлов в куске: +1
        if last_path:  # за итерацию пройден хотя бы один файл
            st["cursor"] = last_path     # возобновить здесь на следующей итерации
        if not chunk:  # новый кусок пуст
            print("no more files — disk fully swept", flush=True)  # диск просмотрен полностью
            break  # закончить цикл
        blob = b"\n\x00FILE\x00\n".join(chunk)  # склеить выборки через метку конца файла
        cid = f"ns_{st['chunks']:04d}"  # имя куска ns_НННН по его номеру
        (CKPT_DIR / "domains").mkdir(parents=True, exist_ok=True)  # создать каталог доменов
        (CKPT_DIR / "domains" / f"{cid}.bin").write_bytes(blob)  # сохранить кусок в файл
        # ---- нейронное обучение на накопленном корпусе ----
        # предел повтора снят: при каждом приёме повторяются ВСЕ прежние домены
        # (полное отсутствие забывания; затраты времени приняты).
        # Переменная окружения HAGI_SYMBOLIC=1 добавляет после сертифицированного
        # шага языковой модели символьную половину: диагностика -> изобретение
        # закона -> проверка -> закрепление; по умолчанию цикл повтора остаётся быстрым.
        import os as _os  # переменные окружения
        if _os.environ.get("HAGI_SYMBOLIC") == "1":  # символьная половина включена
            from hagi.pipeline import unified_ingest  # единый приём: нейронная и символьная половины
            r = unified_ingest(u, cid, blob, epochs=2, repeat=1,  # приём куска: 2 эпохи, 1 повтор,
                               replay_cap=None)  # без предела повтора
        else:  # обычный приём
            r = u.ingest(cid, blob, epochs=2, repeat=1, replay_cap=None)  # приём куска: 2 эпохи, 1 повтор, без предела повтора
            # перепроверить принятые законы и без изобретения законов на этом куске
            try:  # сбой критики не должен прерывать цикл
                from hagi.law_critique import run_law_self_critique  # самокритика принятых законов
                r["self_critique"] = run_law_self_critique(u)  # добавить её итог к результату приёма
            except Exception as _ce:  # ошибка самокритики
                print("    [self-critique] skipped: " + repr(_ce),  # печать причины пропуска
                      flush=True)  # сразу сбросить вывод
        st["chunks"] += 1  # кусков: +1
        ce = r.get("holdout_ce", -1)  # CE на отложенных данных, −1, если ключа нет
        # живой диалог во время обучения: файл .chat кладётся в каталог inbox;
        # его изучает САМ процесс обучения (веса пишет только он, без гонок
        # с сохранением), ответ попадает в файл .reply
        try:  # ящик сообщений необязателен
            inbox = CKPT_DIR / "inbox"  # каталог ящика
            if inbox.exists():  # каталог существует
                for chat_file in sorted(inbox.glob("*.chat")):  # по файлам сообщений по алфавиту
                    reply = []  # строки ответа
                    try:  # ошибка одного сообщения не должна останавливать остальные
                        msg = chat_file.read_text(  # текст сообщения
                            encoding="utf-8", errors="ignore")  # недопустимые байты отбрасываются
                        dom = "chat_" + chat_file.stem  # имя домена: chat_ + имя файла без расширения
                        if "АВТОСАМОКРИТИКА" in msg:  # сообщение — автоматическая критика
                            # критика несёт hex-пары (контекст -> цель). Изучается
                            # ПОТОК БАЙТ, а не hex-текст; консолидация пропускается
                            # (минуты на сообщение, без смысловой ценности для hex-дампов).
                            import re as _re  # регулярные выражения
                            import binascii  # раскодирование hex
                            raw = bytearray()  # накапливаемый поток байтов
                            for m in _re.finditer(  # по соответствиям шаблона
                                    r"^([0-9a-f]{8,})\s*->\s*"  # от начала строки: 8 и более hex-символов, стрелка
                                    r"([0-9a-f]{2})", msg, _re.M):  # ...и ровно 2 hex-символа; режим многострочности
                                raw += binascii.unhexlify(m.group(1))  # добавить байты контекста
                                raw += bytes([int(m.group(2), 16)])  # добавить один байт цели
                            if len(raw) >= 4096:  # потока достаточно для обучения
                                u.ingest(dom, bytes(raw), epochs=1,  # принять поток как домен: 1 эпоха
                                         repeat=1)  # 1 повтор
                            else:  # поток слишком короток
                                u.remember(msg)  # сохранить текст сообщения в памяти
                            reply.append("learned: raw critique "  # строка ответа: изучен поток критики
                                         "stream " + str(len(raw))  # ...длиной len(raw)
                                         + "B")  # ...байтов
                            print("    [chat] " + chat_file.stem  # печать имени сообщения
                                  + ": critique stream "  # ...изучен поток критики
                                  + str(len(raw)) + "B", flush=True)  # ...длиной len(raw) байтов
                            (inbox / (chat_file.stem  # файл ответа <имя>.reply в каталоге ящика
                                      + ".reply")).write_text(  # записать
                                chr(10).join(reply), encoding="utf-8")  # строки ответа через перевод строки
                            chat_file.unlink()  # удалить обработанное сообщение
                            continue  # следующее сообщение
                        u.learn_from_dialog(msg, name=dom)  # выучить диалог как домен dom
                        dom_ce = u.domains.get(dom, {}).get("ce")  # CE домена диалога или None
                        reply.append("learned: domain " + dom  # строка ответа: изучен домен
                                     + " CE=" + str(dom_ce))  # ...с его CE
                        ans = u.consolidate(msg[:400])  # ответ по памяти на первые 400 символов сообщения
                        reply.append("answer: " + ans[:600])  # строка ответа: первые 600 символов ответа
                        print("    [chat] " + chat_file.stem  # печать имени сообщения
                              + ": learned+replied", flush=True)  # ...изучено и отвечено
                    except Exception as chat_err:  # ошибка обработки сообщения
                        reply.append("error: " + repr(chat_err))  # строка ответа: текст ошибки
                        print("    [chat] " + chat_file.stem  # печать имени сообщения
                              + " error: " + repr(chat_err),  # ...и ошибки
                              flush=True)  # сразу сбросить вывод
                    reply_path = inbox / (chat_file.stem + ".reply")  # путь файла ответа
                    reply_path.write_text(chr(10).join(reply),  # записать строки ответа через перевод строки
                                          encoding="utf-8")  # в кодировке UTF-8
                    chat_file.unlink()  # удалить обработанное сообщение
        except Exception:  # ящик необязателен
            pass  # ящик сообщений не обязателен — ошибка игнорируется
        # итог синтезатора законов для этого куска
        sym = r.get("symbolic")  # символьный итог приёма или None
        if sym:  # символьная половина что-то вернула
            for note in sym.get("notes", []):  # по замечаниям синтезатора
                if "multi-byte" in note or "law" in note.lower():  # замечание о многобайтовых законах или о законах
                    print("    " + note, flush=True)  # напечатать замечание
        # семантическая дистилляция всегда включена в цикле полного диска —
        # тот же протокол, что в restore_domains: учитель -> косинусная
        # дистилляция скрытых признаков языковой модели на фрагментах куска
        try:  # сбой дистилляции не должен прерывать цикл
            from hagi.lm import SemanticDistillHead  # голова семантического выравнивания
            bank = getattr(u, "sem_head", None)  # существующая голова или None
            if bank is None:  # головы ещё нет
                bank = u.sem_head = SemanticDistillHead(dim=0)  # создать голову и сохранить её в модели
            blob_u = blob.decode("utf-8", errors="ignore")  # текст куска: недопустимые байты отбрасываются
            frags = [blob_u[i:i + 200]  # фрагменты text[i .. i+199]
                     for i in range(0, min(len(blob_u), 2000), 200)][:8]  # i = 0, 200, ... пока i < min(длина, 2000); первые 8
            frags = [f.ljust(200) for f in frags]  # каждый фрагмент дополнен пробелами справа до длины 200
            if frags:  # есть хотя бы один фрагмент
                import torch as _th  # тензоры
                dev = getattr(u.model, "device", "cpu")  # устройство модели, по умолчанию "cpu"
                idx = _th.tensor(  # тензор контекстов
                    [[(ord(c) % 256) for c in f[:u.model.k]]  # для фрагмента f: коды ord(c) mod 256 первых k символов
                     for f in frags], dtype=_th.long, device=dev)  # по всем фрагментам; тип long; на устройстве модели
                if idx.shape[1] < u.model.k:  # фрагменты короче контекста
                    pad = _th.full(  # прокладка из кода 32 (пробел)
                        (idx.shape[0], u.model.k - idx.shape[1]), 32,  # форма (число фрагментов, k − длина); значение 32
                        dtype=_th.long)  # тип long
                    idx = _th.cat([pad, idx], dim=1)  # приклеить прокладку слева по оси 1: длина строки станет k
                u.model.domain_adapter = None  # отключить адаптер домена
                sl = bank.align_step(u.model, idx, frags)  # один шаг выравнивания; sl — потеря
                print(f"    semantic loss {sl:.3f}", flush=True)  # печать потери с 3 знаками
        except Exception as _e:  # любая ошибка дистилляции
            print(f"    semantic skipped: {_e!r}", flush=True)  # печать причины пропуска
        print(f"[{cid}] neural: +{chunk_bytes/1e6:.1f}MB "  # печать имени куска и прибавки в МБ
              f"({files_now} files) → CE {ce} "  # числа файлов и CE
              f"({r.get('seconds', '?')}s)", flush=True)  # длительности приёма ('?', если ключа нет)
        rp = r.get("repair") or {}  # итог ремонта забывания или пустой словарь
        if rp.get("rounds"):  # были раунды ремонта
            print(f"    repair: forgot {rp.get('forgot')} blocks={len(u.model.blocks)} "  # печать числа забытых блоков и числа блоков модели
                  f"rounds={[(x['status'], round(x['ce'], 3), round(x['after'], 3) if x['after'] else None) for x in rp['rounds']]}", flush=True)  # по раундам: статус, CE с 3 знаками, CE после ремонта или None
        gr = r.get("growth") or {}  # итог роста ёмкости или пустой словарь
        if gr.get("status") in ("GROW", "ROLLBACK"):  # статус — рост или откат
            print(f"    growth: {gr.get('status')} blocks={gr.get('blocks')} "  # печать статуса и числа блоков
                  f"signal={gr.get('signal')} CE {gr.get('ce_before')}->{gr.get('ce_after')}", flush=True)  # сигнала и CE до и после
        chunk, chunk_bytes, files_now = [], 0, 0  # обнулить накопитель куска
        # ---- 2+3. ВЫВОД + СБОРКА: самосборка схем ----
        model = u.model  # языковая модель
        tasks = derive_tasks(model, k=u.k,  # вывести задачи из модели с контекстом k
                             seed=13 + st['chunks'] * 7)  # зерно = 13 + 7·номер куска
        # дедупликация: хеш таблицы истинности — та же булева задача даёт ту же схему;
        # честный «повтор» вместо траты бюджета поиска
        solved = {}  # имя -> ранняя схема из прежних фиксаций (в этом файле только заполняется)
        netlist_path = CKPT_DIR / "processor_netlist.json"  # путь списка схем процессора
        if netlist_path.exists():  # файл списка существует
            for tape in json.loads(netlist_path.read_text(encoding="utf-8")):  # по всем прошлым фиксациям
                for c in tape["circuits"]:  # по схемам фиксации
                    solved[c["name"]] = c  # запомнить схему по её имени
        circuits, reused = [], 0  # схемы этой итерации и число повторов
        seen_tables = {}  # хеш таблицы -> имя задачи, уже увиденной в этой итерации
        # индекс: хеш таблицы -> сертифицированная схема из прошлых фиксаций
        solved_tables = {}  # хеш таблицы -> схема
        netlist_path = CKPT_DIR / "processor_netlist.json"  # путь списка схем (присваивается повторно)
        if netlist_path.exists():  # файл списка существует
            for tape in json.loads(  # по всем прошлым фиксациям
                    netlist_path.read_text(encoding="utf-8")):  # прочитанный JSON
                for c in tape["circuits"]:  # по схемам фиксации
                    th = c.get("truth_hash")  # хеш таблицы истинности или None
                    if th:  # хеш есть
                        solved_tables[th] = c  # индексировать схему по хешу
        for i, task in enumerate(tasks):  # по задачам, i — номер
            thash = __import__("hashlib").sha1(  # thash = первые 10 hex-символов sha1 от JSON отсортированных пар таблицы
                json.dumps(sorted(task["truth"].items()),  # пары (вход, выход) таблицы в порядке сортировки
                           default=str).encode()).hexdigest()[:10]  # непривычные типы — через str; байты UTF-8
            if task.get("meta", {}).get("stage", 0) == 1:  # задача ступени 1 (двухбитный сумматор)
                res = compose_adder2()  # собрать сумматор из сертифицированных ячеек или None
                if res:  # сборка удалась
                    res["truth_hash"] = thash  # записать хеш таблицы
                    res["profile"] = f"{task['name']}_{thash}"  # имя профиля: имя задачи + хеш
                    circuits.append(res)  # добавить схему к фиксации
                    # лестница: ступень 1 собрана ОДИН раз -> открыта ступень 2
                    sf = CKPT_DIR / "ladder_stage.json"  # файл текущей ступени
                    if json.loads(sf.read_text(encoding="utf-8")).get("stage", 0) == 1:  # текущая ступень — 1
                        sf.write_text(json.dumps({"stage": 2}),  # записать новую ступень 2
                                      encoding="utf-8")  # в кодировке UTF-8
                        print(f"  LADDER: stage 1 composed → stage 2 "  # печать: ступень 1 собрана
                              f"(MUX) unlocked", flush=True)  # ...открыта ступень 2 (мультиплексор)
                    print(f"  fabric [{task['name']}]: COMPOSED from "  # печать: схема составлена
                          f"certified cells — CERTIFIED 32/32",  # ...из сертифицированных ячеек, проверено 32 из 32
                          flush=True)  # сразу сбросить вывод
                else:  # сборка не удалась
                    print(f"  fabric [{task['name']}]: composition "  # печать: сборка
                          f"failed (honest: no commit)", flush=True)  # ...не удалась, без фиксации
                continue  # следующая задача
            if thash in seen_tables:  # таблица уже встречалась в этой итерации
                print(f"  fabric [{task['name']}]: DUPLICATE of "  # печать: дубликат
                      f"[{seen_tables[thash]}] — reused, no re-search",  # ...имени первой задачи; повтор, без поиска
                      flush=True)  # сразу сбросить вывод
                reused += 1  # повторов: +1
                continue  # следующая задача
            seen_tables[thash] = task["name"]  # запомнить таблицу
            if thash in solved_tables:  # задача уже решена в прошлой фиксации: честный повтор
                prev = solved_tables[thash]  # прежняя схема
                print(f"  fabric [{task['name']}]: SOLVED earlier "  # печать: решена ранее
                      f"[{prev['name']}] — reusing certified circuit",  # ...именем; используется сертифицированная схема
                      flush=True)  # сразу сбросить вывод
                circuits.append({"profile": f"{task['name']}_{thash}",  # новая запись: имя профиля
                                 "seed": prev.get("seed", 0),  # зерно прежней схемы или 0
                                 "certified_gen": prev.get("certified_gen"),  # сертифицированное поколение прежней схемы
                                 "best": None,  # генома нет — схема берётся из прежней записи
                                 "bits": prev.get("bits"),  # стоимость в битах
                                 "_reuse_of": prev})  # ссылка на прежнюю схему
                continue  # следующая задача
            res = weave_task(task, seed=5 + i)  # собрать схему эволюцией с зерном 5 + i или None
            if res:  # схема найдена
                res["truth_hash"] = thash  # записать хеш таблицы
                # лестница: задача ступени решена -> повысить сложность
                stage = task.get("meta", {}).get("stage", 0)  # ступень задачи или 0
                sf = CKPT_DIR / "ladder_stage.json"  # файл текущей ступени
                sf.write_text(json.dumps({"stage": stage + 1}),  # записать новую ступень stage + 1
                              encoding="utf-8")  # в кодировке UTF-8
                print(f"  LADDER: stage {stage} solved → stage "  # печать: ступень решена
                      f"{stage+1} unlocked", flush=True)  # ...открыта следующая
            if res:  # схема найдена
                res["profile"] = f"{task['name']}_{thash}"  # имя профиля: имя задачи + хеш
                circuits.append(res)  # добавить схему к фиксации
                nb = res["bits"]  # стоимость в битах
                ng = len(res["best"]["gates"])  # число вентилей
                print(f"  fabric [{task['name']}]: CERTIFIED "  # печать: сертифицирована
                      f"{ng} gates, {nb} bits, gen "  # ...число вентилей, битов
                      f"{res['certified_gen']}", flush=True)  # ...поколение сертификации
            else:  # схема не найдена
                print(f"  fabric [{task['name']}]: no certified circuit "  # печать: сертифицированной схемы нет
                      f"(honest: no commit)", flush=True)  # ...без фиксации
        # ---- 4. ФИКСАЦИЯ ----
        if circuits:  # есть что фиксировать
            commit(circuits, cid, ce)  # дописать схемы к списку схем процессора с именем куска и CE
            print(f"  TAPE-OUT: {len(circuits)} certified circuits "  # печать числа зафиксированных схем
                  f"→ processor_netlist.json", flush=True)  # ...и файла списка
        # ---- сохранить состояние одной модели ----
        u.save()  # сохранить модель и книгу доменов
        save_state(st)  # сохранить состояние просмотра
        it += 1  # итераций: +1
    print(f"cycle done: iterations={it} chunks={st['chunks']} "  # печать числа итераций и кусков
          f"ingested={st['bytes_ingested']/1e6:.0f}MB", flush=True)  # ...и принятых мегабайтов


if __name__ == "__main__":
    main()  # запустить цикл  # запустить цикл
