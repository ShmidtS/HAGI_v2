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

sys.path.insert( 0, str(Path(__file__).resolve().parent.parent) )  # добавить корень проекта в путь импорта
from hagi.data import fabric  # эмуляция и проверка логических схем
from hagi.selfdev import unified  # объединённая модель
from hagi.selfdev.cycle import ( compose_adder2, commit, derive_tasks, fusion_fastpath, weave_task, )  # сборка нейросимвольных задач цикла
from hagi.base.util import CKPT_DIR  # каталог контрольных точек
from scripts.feed_all_c import ( CHUNK_TARGET, SKIP_DIRS, iter_files, load_state, save_state, sample_bytes, )  # загрузчик потока данных и его настройки


TAPEOUT = ( CKPT_DIR / "processor_netlist.json" )  # путь списка схем процессора (в этом файле далее не используется)

  # ---------- ВЫВОД: задачи для схем ИЗ обученного тела ----------


def main() -> None:
    """Выполнять итерации «просмотр -> приём -> ящик сообщений -> семантика -> схемы -> фиксация» до предельного времени.

    Аргумент 1: предельное время в минутах (по умолчанию 480)."""
    max_minutes = ( float(sys.argv[1]) if len(sys.argv) > 1 else 480.0 )  # предел времени в минутах: аргумент или 480
    deadline = ( time.time() + max_minutes * 60 )  # момент окончания: сейчас + минуты·60 секунд
  # загрузить модель, если файл есть, иначе создать новую
    u = ( unified.Unified.load() if unified.Unified.exists() else unified.Unified() )  # ...иначе новая
    st = load_state()  # состояние просмотра диска
    TEXT_ONLY = ( __import__("os").environ.get("HAGI_TEXT_ONLY", "0") == "1" )  # текстовая диета: HAGI_TEXT_ONLY=1 — только печатные файлы без дублей
    seen_contents = ( set((CKPT_DIR / "seen_contents.txt").read_text(encoding="utf-8").split(chr(10))) if (CKPT_DIR / "seen_contents.txt").exists() else set() )  # хеши уже принятых текстов
    seen = st["seen"]  # словарь уже обработанных хешей путей
    chunk, chunk_bytes = [], 0  # выборки текущего куска и число его байтов
    files_now = 0  # число файлов текущего куска
    it = 0  # число завершённых итераций
    (CKPT_DIR / "inbox").mkdir( parents=True, exist_ok=True )  # создать каталог ящика входящих сообщений
    print( f"neuro-symbolic cycle start: chunks={st['chunks']}", flush=True )  # печать числа уже принятых кусков
    recent: list[bytes] = ( [] )  # последние чанки (материал для эволюции алгоритма обучения)
    ev_counter = [0]  # счётчик чанков для эволюции (одно поколение раз в PERIOD)
    arch_counter = [ 0 ]  # счётчик чанков для эволюции архитектуры (проба раз в ARCH_PERIOD)
    while time.time() < deadline:  # пока не истекло предельное время
  # ---- 1. РОСТ: набрать кусок с C:/ (постоянный курсор) ----
        last_path = None  # последний пройденный путь (курсор возобновления)
        chunk_domain = [ None ]  # домен текущего куска (проект X/Y); кусок не смешивает проекты
        added_path = None  # последний добавленный в кусок файл (курсор возобновления)
        for path in iter_files( "C:/", start_after=st.get("cursor") ):  # по файлам диска, начиная после курсора
            last_path = path  # запомнить путь как новый курсор
            if ( chunk_bytes >= CHUNK_TARGET or time.time() > deadline ):  # кусок набран или время вышло
                break  # прекратить набор
  # Домен = один проект: первые два уровня каталога после корня (C:/X/Y/). Кусок не смешивает
  # проекты — иначе 3..39 случайных файлов с диска дают шум без повторяющихся закономерностей,
  # и ни языковая модель, ни поиск законов не находят в нём структуры.
            _parts = ( __import__("os").path.normpath(path).replace("\\", "/").split("/") )  # компоненты пути
            _domain = ( "/".join(_parts[:3]) if len(_parts) > 3 else "/".join(_parts[:2]) )  # домен пути: X/Y (или X для файлов в корне)
            if ( chunk and _domain != chunk_domain[0] ):  # проект сменился — закрыть кусок на границе
                break  # прекратить набор (следующий кусок начнётся с этого пути: курсор не тронут)
            chunk_domain[0] = _domain  # домен текущего куска
            st["visited"] += 1  # посещено: +1 файл
            try:  # размер может быть недоступен
                size = __import__("os").path.getsize( path )  # размер файла в байтах (модуль os импортируется на месте)
            except OSError:  # размер получить не удалось
                continue  # следующий файл
            if size < 16:  # файл короче 16 байтов
                continue  # пропустить
            h = ( __import__("hashlib") .sha1( path.encode("utf-8", errors="replace") ) .hexdigest()[:12] )  # h = первые 12 hex-символов sha1 от пути (модуль hashlib импортируется на месте) # путь в байтах UTF-8 с заменой ошибок
            if h in seen:  # путь уже обработан
                continue  # следующий файл
            data = sample_bytes(path, size)  # выборка байтов файла или None
            if ( TEXT_ONLY and data is not None ):  # текстовая диета: только печатное, без дублей
                pr = sum( 32 <= c < 127 or c in (9, 10, 13) or c >= 192 for c in data[:65536] ) / max( 1, min(len(data), 65536) )  # доля печатных байтов в голове
                if pr < 0.9:  # не текст
                    seen[h] = "nontext"  # пометить: пропущен как нетекстовый
                    continue  # следующий файл
                dh = ( __import__("hashlib").sha1(data).hexdigest()[:16] )  # хеш содержимого выборки
                if dh in seen_contents:  # такой текст уже был
                    seen[h] = "dup"  # пометить дубль
                    continue  # следующий файл
                seen_contents.add(dh)  # запомнить содержимое
            if data is None:  # файл прочитать не удалось
                seen[h] = "locked"  # пометить путь как заблокированный
                continue  # следующий файл
            seen[h] = len(data)  # пометить путь числом байтов выборки
            st["sampled"] += 1  # прочитано: +1 файл
            chunk.append(data)  # добавить выборку к куску
            added_path = path  # курсор — последний ДОБАВЛЕННЫЙ файл (первый файл следующего проекта не теряется)
            chunk_bytes += len(data)  # chunk_bytes += длина выборки
            st["bytes_ingested"] += len(data)  # принято байтов: + длина выборки
            files_now += 1  # файлов в куске: +1
        if added_path:  # в кусок добавлен хотя бы один файл
            st["cursor"] = ( added_path )  # возобновить после последнего добавленного (файл нового проекта не пропускается)
        elif ( last_path and not chunk ):  # ничего не добавлено (все просмотренные — нетекст, дубли, занятые): не залипать на них
            st["cursor"] = last_path  # продолжить после последнего просмотренного
        if not chunk:  # новый кусок пуст
            print( "no more files — disk fully swept", flush=True )  # диск просмотрен полностью
            break  # закончить цикл
        blob = b"\n\x00FILE\x00\n".join( chunk )  # склеить выборки через метку конца файла
        (CKPT_DIR / "domains").mkdir( parents=True, exist_ok=True )  # создать каталог доменов
        # номер куска не должен занимать имя уже принятого домена: после утери sweep_state.json
        # счётчик начинался с 0 и новый кусок перезаписывал байты ns_0000.. книги
        while (CKPT_DIR / "domains" / f"ns_{st['chunks']:04d}.bin").exists() or (f"ns_{st['chunks']:04d}" in u.domains):  # имя занято файлом или записью книги
            st["chunks"] += 1  # взять следующий номер
        cid = f"ns_{st['chunks']:04d}"  # имя куска ns_НННН по свободному номеру
        (CKPT_DIR / "domains" / f"{cid}.bin").write_bytes( blob )  # сохранить кусок в файл
  # ---- нейронное обучение на накопленном корпусе ----
  # предел повтора снят: при каждом приёме повторяются ВСЕ прежние домены
  # (полное отсутствие забывания; затраты времени приняты).
  # Переменная окружения HAGI_SYMBOLIC=1 добавляет после сертифицированного
  # шага языковой модели символьную половину: диагностика -> изобретение
  # закона -> проверка -> закрепление; по умолчанию цикл повтора остаётся быстрым.
        import os as _os  # переменные окружения

        t_ingest = time.time()  # начало приёма куска (для отчёта о времени фаз)
        if _os.environ.get("HAGI_SYMBOLIC") == "1":  # символьная половина включена
            from hagi.selfdev.pipeline import ( unified_ingest, )  # единый приём: нейронная и символьная половины

            r = unified_ingest( u, cid, blob, epochs=2, repeat=1, replay_cap=None, )  # приём куска: 2 эпохи, 1 повтор, # без предела повтора
        else:  # обычный приём
            r = u.ingest( cid, blob, epochs=2, repeat=1, replay_cap=None )  # приём куска: 2 эпохи, 1 повтор, без предела повтора
  # перепроверить принятые законы и без изобретения законов на этом куске
            try:  # сбой критики не должен прерывать цикл
                from hagi.data.law_critique import ( run_law_self_critique, )  # самокритика принятых законов

                r["self_critique"] = run_law_self_critique( u )  # добавить её итог к результату приёма
            except Exception as _ce:  # ошибка самокритики
                print( "    [self-critique] skipped: " + repr(_ce), flush=True, )  # печать причины пропуска # сразу сбросить вывод
        st["chunks"] += 1  # кусков: +1
        if r.get("new_ce_before") is not None and not r.get("skipped"):  # измерен перенос и приём выполнен (пропущенный кусок короче 4096 байт возвращает только skipped)
            print( f"  перенос: новый домен до обучения {r['new_ce_before']:.3f} -> после {r.get('holdout_ce', float('nan')):.3f}" + ( f" | хвост домена {r['tail_ce']:.3f}" if r.get("tail_ce") is not None else "" ) + ( f" | старые домены до приёма {r['old_ce_pre']:.3f}" if r.get("old_ce_pre") is not None else "" ), flush=True, )  # CE нового до/после # и CE старых (накопление)
        if isinstance(r.get("programs"), dict):  # модель писала программы в этом приёме
            pg = r["programs"]  # итог выгрузки и исполнения
            print( f"  programs: функций {pg.get('functions', 0)} | исполнено {pg.get('executed', 0)} | " f"провалов {pg.get('failures', 0)} | отказано {pg.get('refused', 0)}" + ( f" | ошибка {pg.get('error')}" if pg.get("status") == "ERROR" else "" ), flush=True, )  # сколько написано и исполнено # сколько провалило сертификат и сколько вне белого списка # с текстом ошибки, если была
        ce = r.get("holdout_ce", -1)  # CE на отложенных данных, −1, если ключа нет
        if r.get("skipped"):  # приём пропущен (кусок короче 4096 байт)
            print(f"  пропуск: кусок {chunk_bytes} байтов короче 4096 — приём не выполнялся", flush=True)  # явная строка вместо молчаливого «CE -1»
  # живой диалог во время обучения: файл .chat кладётся в каталог inbox;
  # его изучает САМ процесс обучения (веса пишет только он, без гонок
  # с сохранением), ответ попадает в файл .reply
        try:  # ящик сообщений необязателен
            inbox = CKPT_DIR / "inbox"  # каталог ящика
            if inbox.exists():  # каталог существует
                for chat_file in sorted( inbox.glob("*.chat") ):  # по файлам сообщений по алфавиту
                    reply = []  # строки ответа
                    try:  # ошибка одного сообщения не должна останавливать остальные
                        msg = chat_file.read_text( encoding="utf-8", errors="ignore" )  # текст сообщения # недопустимые байты отбрасываются
                        dom = ( "chat_" + chat_file.stem )  # имя домена: chat_ + имя файла без расширения
                        if ( "АВТОСАМОКРИТИКА" in msg ):  # сообщение — автоматическая критика
  # критика несёт hex-пары (контекст -> цель). Изучается
  # ПОТОК БАЙТ, а не hex-текст; консолидация пропускается
  # (минуты на сообщение, без смысловой ценности для hex-дампов).
                            import re as _re  # регулярные выражения
                            import binascii  # раскодирование hex

                            raw = bytearray()  # накапливаемый поток байтов
                            for m in _re.finditer( r"^([0-9a-f]{8,})\s*->\s*" r"([0-9a-f]{2})", msg, _re.M, ):  # по соответствиям шаблона # от начала строки: 8 и более hex-символов, стрелка # ...и ровно 2 hex-символа; режим многострочности
                                raw += binascii.unhexlify( m.group(1) )  # добавить байты контекста
                                raw += bytes( [int(m.group(2), 16)] )  # добавить один байт цели
                            if len(raw) >= 4096:  # потока достаточно для обучения
                                u.ingest( dom, bytes(raw), epochs=1, repeat=1, )  # принять поток как домен: 1 эпоха # 1 повтор
                            else:  # поток слишком короток
                                u.remember(msg)  # сохранить текст сообщения в памяти
                            reply.append( "learned: raw critique " "stream " + str(len(raw)) + "B" )  # строка ответа: изучен поток критики # ...длиной len(raw) # ...байтов
                            print( "    [chat] " + chat_file.stem + ": critique stream " + str(len(raw)) + "B", flush=True, )  # печать имени сообщения # ...изучен поток критики # ...длиной len(raw) байтов
                            ( inbox / ( chat_file.stem + ".reply" ) ).write_text( chr(10).join(reply), encoding="utf-8" )  # файл ответа <имя>.reply в каталоге ящика # записать # строки ответа через перевод строки
                            chat_file.unlink()  # удалить обработанное сообщение
                            continue  # следующее сообщение
                        u.learn_from_dialog( msg, name=dom )  # выучить диалог как домен dom
                        dom_ce = u.domains.get(dom, {}).get( "ce" )  # CE домена диалога или None
                        reply.append( "learned: domain " + dom + " CE=" + str(dom_ce) )  # строка ответа: изучен домен # ...с его CE
                        ans = u.consolidate( msg[:400] )  # ответ по памяти на первые 400 символов сообщения
                        reply.append( "answer: " + ans[:600] )  # строка ответа: первые 600 символов ответа
                        print( "    [chat] " + chat_file.stem + ": learned+replied", flush=True, )  # печать имени сообщения # ...изучено и отвечено
                    except Exception as chat_err:  # ошибка обработки сообщения
                        reply.append( "error: " + repr(chat_err) )  # строка ответа: текст ошибки
                        print( "    [chat] " + chat_file.stem + " error: " + repr(chat_err), flush=True, )  # печать имени сообщения # ...и ошибки # сразу сбросить вывод
                    reply_path = inbox / ( chat_file.stem + ".reply" )  # путь файла ответа
                    reply_path.write_text( chr(10).join( reply ), encoding="utf-8", )  # записать строки ответа через перевод строки # в кодировке UTF-8
                    chat_file.unlink()  # удалить обработанное сообщение
        except Exception:  # ящик необязателен
            pass  # ящик сообщений не обязателен — ошибка игнорируется
  # итог синтезатора законов для этого куска
        sym = r.get("symbolic")  # символьный итог приёма или None
        if sym:  # символьная половина что-то вернула
            for note in sym.get("notes", []):  # по замечаниям синтезатора
                if ( "multi-byte" in note or "law" in note.lower() ):  # замечание о многобайтовых законах или о законах
                    print("    " + note, flush=True)  # напечатать замечание
  # семантическая дистилляция всегда включена в цикле полного диска —
  # тот же протокол, что в restore_domains: учитель -> косинусная
  # дистилляция скрытых признаков языковой модели на фрагментах куска
        t_sem = time.time()  # начало семантической дистилляции (для отчёта о времени)
        try:  # сбой дистилляции не должен прерывать цикл
            from hagi.models.lm import ( SemanticDistillHead, )  # голова семантического выравнивания

            bank = getattr(u, "sem_head", None)  # существующая голова или None
            if bank is None:  # головы ещё нет
                bank = u.sem_head = SemanticDistillHead( dim=0 )  # создать голову и сохранить её в модели
            blob_u = blob.decode( "utf-8", errors="ignore" )  # текст куска: недопустимые байты отбрасываются
            frags = [ blob_u[i : i + 200] for i in range(0, min(len(blob_u), 2000), 200) ][ :8 ]  # фрагменты text[i .. i+199] # i = 0, 200, ... пока i < min(длина, 2000); первые 8
            frags = [ f.ljust(200) for f in frags ]  # каждый фрагмент дополнен пробелами справа до длины 200
            if frags:  # есть хотя бы один фрагмент
                import torch as _th  # тензоры

                dev = getattr( u.model, "device", "cpu" )  # устройство модели, по умолчанию "cpu"
                idx = _th.tensor( [ [ (ord(c) % 256) for c in f[: u.model.k] ] for f in frags ], dtype=_th.long, device=dev, )  # тензор контекстов # для фрагмента f: коды ord(c) mod 256 первых k символов # по всем фрагментам; тип long; на устройстве модели
                if idx.shape[1] < u.model.k:  # фрагменты короче контекста
                    pad = _th.full( (idx.shape[0], u.model.k - idx.shape[1]), 32, dtype=_th.long, )  # прокладка из кода 32 (пробел) # форма (число фрагментов, k − длина); значение 32 # тип long
                    idx = _th.cat( [pad, idx], dim=1 )  # приклеить прокладку слева по оси 1: длина строки станет k
                u.model.domain_adapter = None  # отключить адаптер домена
                sl = bank.align_step( u.model, idx, frags )  # один шаг выравнивания; sl — потеря
                print( f"    semantic loss {sl:.3f} ({time.time() - t_sem:.1f}s)", flush=True, )  # печать потери с 3 знаками и длительности семантической дистилляции
        except Exception as _e:  # любая ошибка дистилляции
            print( f"    semantic skipped: {_e!r}", flush=True )  # печать причины пропуска
        t_ingest = round(time.time() - t_ingest, 1)  # длительность приёма куска, с
        print( f"[{cid}] neural: +{chunk_bytes/1e6:.1f}MB " f"({files_now} files) → CE {ce}" + ( f" | хвост {r['tail_ce']:.3f}" if r.get("tail_ce") is not None else "" ) + f" " f"(нейро {r.get('phases', {}).get('neural', '?')}s + символ {r.get('phases', {}).get('symbolic', '?')}s = {t_ingest}s) " f"фазы {r.get('phases')}", flush=True, )  # печать имени куска и прибавки в МБ # числа файлов и CE # длительности половин приёма # , flush=True)  # длительности приёма ('?', если ключа нет)
        rp = r.get("repair") or {}  # итог ремонта забывания или пустой словарь
        if rp.get("rounds"):  # были раунды ремонта
            print( f"    repair: forgot {rp.get('forgot')} worst {rp.get('worst')} blocks={len(u.model.blocks)} " f"rounds={[(x['status'], round(x['ce'], 3), round(x['after'], 3) if x['after'] else None) for x in rp['rounds']]}", flush=True, )  # печать числа забытых блоков и числа блоков модели # по раундам: статус, CE с 3 знаками, CE после ремонта или None
        gr = r.get("growth") or {}  # итог роста ёмкости или пустой словарь
        if gr.get("status") in ("GROW", "ROLLBACK"):  # статус — рост или откат
            print( f"    growth: {gr.get('status')} blocks={gr.get('blocks')} " f"signal={gr.get('signal')} CE {gr.get('ce_before')}->{gr.get('ce_after')}", flush=True, )  # печать статуса и числа блоков # сигнала и CE до и после

        chunk, chunk_bytes, files_now = [], 0, 0  # обнулить накопитель куска
  # ---- 2+3. ВЫВОД + СБОРКА: самосборка схем ----
        model = u.model  # языковая модель
        tasks = derive_tasks( model, k=u.k, seed=13 + st["chunks"] * 7, )  # вывести задачи из модели с контекстом k # зерно = 13 + 7·номер куска
  # дедупликация: хеш таблицы истинности — та же булева задача даёт ту же схему;
  # честный «повтор» вместо траты бюджета поиска
        solved = ( {} )  # имя -> ранняя схема из прежних фиксаций (в этом файле только заполняется)
        netlist_path = ( CKPT_DIR / "processor_netlist.json" )  # путь списка схем процессора
        if netlist_path.exists():  # файл списка существует
            for tape in json.loads( netlist_path.read_text(encoding="utf-8") ):  # по всем прошлым фиксациям
                for c in tape["circuits"]:  # по схемам фиксации
                    solved[c["name"]] = c  # запомнить схему по её имени
        circuits, reused = [], 0  # схемы этой итерации и число повторов
        seen_tables = {}  # хеш таблицы -> имя задачи, уже увиденной в этой итерации
  # индекс: хеш таблицы -> сертифицированная схема из прошлых фиксаций
        solved_tables = {}  # хеш таблицы -> схема
        netlist_path = ( CKPT_DIR / "processor_netlist.json" )  # путь списка схем (присваивается повторно)
        if netlist_path.exists():  # файл списка существует
            for tape in json.loads( netlist_path.read_text(encoding="utf-8") ):  # по всем прошлым фиксациям # прочитанный JSON
                for c in tape["circuits"]:  # по схемам фиксации
                    th = c.get("truth_hash")  # хеш таблицы истинности или None
                    if th:  # хеш есть
                        solved_tables[th] = c  # индексировать схему по хешу
        for i, task in enumerate(tasks):  # по задачам, i — номер
            thash = ( __import__("hashlib") .sha1( json.dumps( sorted( task["truth"].items() ), default=str, ).encode() ) .hexdigest()[:10] )  # thash = первые 10 hex-символов sha1 от JSON отсортированных пар таблицы # пары (вход, выход) таблицы в порядке сортировки # непривычные типы — через str; байты UTF-8
            if ( task.get("meta", {}).get("stage", 0) == 1 ):  # задача ступени 1 (двухбитный сумматор)
                res = ( compose_adder2() )  # собрать сумматор из сертифицированных ячеек или None
                if res:  # сборка удалась
                    res["truth_hash"] = thash  # записать хеш таблицы
                    res["profile"] = ( f"{task['name']}_{thash}" )  # имя профиля: имя задачи + хеш
                    circuits.append(res)  # добавить схему к фиксации
  # лестница: ступень 1 собрана ОДИН раз -> открыта ступень 2
                    sf = CKPT_DIR / "ladder_stage.json"  # файл текущей ступени
                    if ( json.loads(sf.read_text(encoding="utf-8")).get("stage", 0) == 1 ):  # текущая ступень — 1
                        sf.write_text( json.dumps({"stage": 2}), encoding="utf-8", )  # записать новую ступень 2 # в кодировке UTF-8
                        print( f"  LADDER: stage 1 composed → stage 2 " f"(MUX) unlocked", flush=True, )  # печать: ступень 1 собрана # ...открыта ступень 2 (мультиплексор)
                    print( f"  fabric [{task['name']}]: COMPOSED from " f"certified cells — CERTIFIED 32/32", flush=True, )  # печать: схема составлена # ...из сертифицированных ячеек, проверено 32 из 32 # сразу сбросить вывод
                else:  # сборка не удалась
                    print( f"  fabric [{task['name']}]: composition " f"failed (honest: no commit)", flush=True, )  # печать: сборка # ...не удалась, без фиксации
                continue  # следующая задача
            if thash in seen_tables:  # таблица уже встречалась в этой итерации
                print( f"  fabric [{task['name']}]: DUPLICATE of " f"[{seen_tables[thash]}] — reused, no re-search", flush=True, )  # печать: дубликат # ...имени первой задачи; повтор, без поиска # сразу сбросить вывод
                reused += 1  # повторов: +1
                continue  # следующая задача
            seen_tables[thash] = task["name"]  # запомнить таблицу
            if ( thash in solved_tables ):  # задача уже решена в прошлой фиксации: честный повтор
                prev = solved_tables[thash]  # прежняя схема
                print( f"  fabric [{task['name']}]: SOLVED earlier " f"[{prev['name']}] — reusing certified circuit", flush=True, )  # печать: решена ранее # ...именем; используется сертифицированная схема # сразу сбросить вывод
                circuits.append( { "profile": f"{task['name']}_{thash}", "seed": prev.get("seed", 0), "certified_gen": prev.get( "certified_gen" ), "best": None, "bits": prev.get("bits"), "_reuse_of": prev, } )  # новая запись: имя профиля # зерно прежней схемы или 0 # сертифицированное поколение прежней схемы # генома нет — схема берётся из прежней записи # стоимость в битах # ссылка на прежнюю схему
                continue  # следующая задача
            res = weave_task( task, seed=5 + i )  # собрать схему эволюцией с зерном 5 + i или None
            if res:  # схема найдена
                res["truth_hash"] = thash  # записать хеш таблицы
  # лестница: задача ступени решена -> повысить сложность
                stage = task.get("meta", {}).get("stage", 0)  # ступень задачи или 0
                sf = CKPT_DIR / "ladder_stage.json"  # файл текущей ступени
                sf.write_text( json.dumps( {"stage": stage + 1} ), encoding="utf-8", )  # записать новую ступень stage + 1 # в кодировке UTF-8
                print( f"  LADDER: stage {stage} solved → stage " f"{stage+1} unlocked", flush=True, )  # печать: ступень решена # ...открыта следующая
            if res:  # схема найдена
                res["profile"] = ( f"{task['name']}_{thash}" )  # имя профиля: имя задачи + хеш
                circuits.append(res)  # добавить схему к фиксации
                nb = res["bits"]  # стоимость в битах
                ng = len(res["best"]["gates"])  # число вентилей
                print( f"  fabric [{task['name']}]: CERTIFIED " f"{ng} gates, {nb} bits, gen " f"{res['certified_gen']}", flush=True, )  # печать: сертифицирована # ...число вентилей, битов # ...поколение сертификации
            else:  # схема не найдена
                print( f"  fabric [{task['name']}]: no certified circuit " f"(honest: no commit)", flush=True, )  # печать: сертифицированной схемы нет # ...без фиксации
  # ---- 4. ФИКСАЦИЯ ----
        if circuits:  # есть что фиксировать
            commit( circuits, cid, ce )  # дописать схемы к списку схем процессора с именем куска и CE
            print( f"  TAPE-OUT: {len(circuits)} certified circuits " f"→ processor_netlist.json", flush=True, )  # печать числа зафиксированных схем # ...и файла списка
  # ---- 5. ЭВОЛЮЦИЯ АЛГОРИТМА ОБУЧЕНИЯ ----
  # Модель сама подбирает параметры своего обучения (шаги, пакет, момент, затухание,
  # разогрев, отжиг): популяция с мутацией и скрещиванием, претендент принимается только
  # парным гейтом по контрольной кросс-энтропии на последних чанках (hagi/selfdev/evolve.py).
        recent.append(blob)  # помнить последние чанки для оценки фитнеса
        del recent[:-6]  # держать не более шести
        try:  # сбой эволюции не должен прерывать цикл
            from hagi.selfdev.evolve import ( maybe_evolve as _maybe_evolve, )  # вызов с периодом

            ev = _maybe_evolve( u.model, recent, ev_counter )  # одно поколение, если пришло время
            if ev is not None:  # поколение было
                print( f"  evolve: {ev['decision']} | зонд {ev['probe']:.4f} | гейт {ev['gate']} | " f"genome {{{', '.join(f'{k}={v:.4g}' for k, v in ev['best'].items())}}}", flush=True, )  # решение и замеры # лучший геном
        except Exception as _ee:  # ошибка эволюции
            print( "    [evolve] skipped: " + repr(_ee)[:160], flush=True )  # причина пропуска
        try:  # проба архитектуры под защитой
            from hagi.selfdev.evolve import ( maybe_evolve_arch as _maybe_arch, )  # вызов с периодом

            ar = _maybe_arch( u, recent, arch_counter )  # проба расширения ширины, если пришло время
            if ar is not None:  # проба была
                print( f"  arch-evolve: {ar['decision']} | эталон {ar['base']:.4f} | пробы {ar['tried']}" + (f" | ширина {ar['d']}" if ar.get("d") else ""), flush=True, )  # решение и замеры # с новой шириной
        except Exception as _ae:  # ошибка пробы архитектуры
            print( "    [arch-evolve] skipped: " + repr(_ae)[:160], flush=True )  # причина пропуска
  # ---- сохранить состояние одной модели ----
        u.save()  # сохранить модель и книгу доменов
        save_state(st)  # сохранить состояние просмотра
        if TEXT_ONLY:  # текстовая диета — сохранить хеши принятых текстов
            (CKPT_DIR / "seen_contents.txt").write_text( chr(10).join(sorted(seen_contents)), encoding="utf-8" )  # по одному хешу на строку
        it += 1  # итераций: +1
    print( f"cycle done: iterations={it} chunks={st['chunks']} " f"ingested={st['bytes_ingested']/1e6:.0f}MB", flush=True, )  # печать числа итераций и кусков # ...и принятых мегабайтов


if __name__ == "__main__":  # запуск файла как скрипта
    main()  # запустить цикл
