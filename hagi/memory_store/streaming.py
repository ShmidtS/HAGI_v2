"""Контекст на диске с ограниченной оперативной памятью для сколь угодно длинных входных потоков.

Это подложка контекста, а не утверждение, что конечное нейронное состояние способно
без потерь запомнить бесконечно много независимых фактов. Текст режется на страницы
и индексируется инкрементально в SQLite FTS5; запрос выбирает лишь ограниченное число
подошедших страниц. Рабочий набор ограничен размером страницы и бюджетом запроса,
а постоянное хранилище растёт вместе с объёмом принятого текста.

Содержимое: записи MemoryHit (попадание поиска), FactRecord (факт с версией),
MemoryStats (статистика хранилища); DiskBackedContext — индекс страниц и фактов на
SQLite; StreamingContext — сборщик контекстной подсказки ограниченного размера."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

from dataclasses import dataclass  # декоратор класса данных
import os  # размер файлов базы
import re  # выделение слов запроса
import sqlite3  # база данных SQLite с полнотекстовым поиском FTS5
from pathlib import Path  # создание каталога базы
from typing import Iterable, Iterator  # типы для аннотаций


@dataclass(frozen=True)  # неизменяемый класс данных
class MemoryHit:
    """Одно попадание полнотекстового поиска: страница, её границы, оценка, текст и фрагмент."""

    page_id: int  # номер страницы (rowid в таблице pages)
    char_start: int  # позиция первого символа страницы в потоке
    char_end: int  # позиция после последнего символа страницы в потоке
    score: float  # оценка релевантности bm25 (меньше — лучше)
    text: str  # полный текст страницы
    snippet: str  # фрагмент текста вокруг найденных слов


@dataclass(frozen=True)  # неизменяемый класс данных
class FactRecord:
    """Одна версия факта."""

    key: str  # нормализованный ключ факта
    revision: int  # номер версии (начиная с 1)
    value: str  # значение факта в этой версии
    source: str  # источник значения


@dataclass(frozen=True)  # неизменяемый класс данных
class MemoryStats:
    """Статистика хранилища: страницы, проиндексированные и ожидающие символы, размер базы."""

    pages: int  # число страниц в индексе
    indexed_chars: int  # число символов, перенесённых в страницы
    pending_chars: int  # число символов в буфере, ещё не ставших страницей
    max_pending_chars: int  # наибольший размер буфера за всё время
    page_chars: int  # предел размера страницы
    db_bytes: int  # суммарный размер файлов базы в байтах


class DiskBackedContext:
    """Индексировать длинный текстовый поток инкрементально, не удерживая его в оперативной памяти.

    Аргументы:
        path: путь к базе SQLite; ``:memory:`` допустимо для модульных тестов.
        page_chars: верхняя граница размера сохраняемой страницы, кроме случая
            одного неразрывного слова длиннее этого значения.
        max_query_pages: наибольшее число страниц, возвращаемых одним запросом.

    Полнотекстовый поиск использует токенизатор unicode61 из SQLite. Поиск
    лексический, а не смысловой; семантический индекс можно добавить за этим
    интерфейсом позже."""

    def __init__(
        self,  # создаваемый объект хранилища
        path: str | os.PathLike[str],  # путь к базе SQLite
        page_chars: int = 4096,  # предел размера страницы в символах
        max_query_pages: int = 4,  # предел числа страниц в ответе на запрос
    ) -> None:  # результат конструктора — сам объект
        """Создать или открыть базу по пути path с пределом страницы page_chars и пределом ответа max_query_pages."""
        if page_chars < 128:  # страница слишком мала
            raise ValueError("page_chars must be >= 128")  # ошибка: предел страницы меньше 128
        if max_query_pages < 1:  # нельзя вернуть меньше одной страницы
            raise ValueError("max_query_pages must be >= 1")  # ошибка: предел ответа меньше 1
        self.path = str(path)  # путь к базе как строка
        self.page_chars = int(page_chars)  # предел страницы как целое
        self.max_query_pages = int(max_query_pages)  # предел ответа как целое
        if self.path != ":memory:":  # база на диске, а не в памяти
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)  # создать её каталог вместе с родительскими (существует — не ошибка)
        self.db = sqlite3.connect(self.path)  # соединение с базой
        self.db.execute("PRAGMA journal_mode=WAL")  # журнал упреждающей записи: чтение не блокируется записью
        self.db.execute("PRAGMA synchronous=NORMAL")  # обычная синхронизация с диском (в режиме WAL надёжна и быстрее полной)
        # полнотекстовая таблица страниц: один столбец content, токенизатор unicode61
        self.db.execute("""\
CREATE VIRTUAL TABLE IF NOT EXISTS pages
                          USING fts5(content, tokenize='unicode61')"""  # полнотекстовая таблица pages (столбец content, токенизатор unicode61), если её нет
                          )
        # таблица положения страниц: номер — ключ, начало и конец в потоке, длина
        self.db.execute("""\
CREATE TABLE IF NOT EXISTS page_meta (
                          page_id INTEGER PRIMARY KEY,
                          char_start INTEGER NOT NULL,
                          char_end INTEGER NOT NULL,
                          char_count INTEGER NOT NULL)"""  # char_count — длина страницы в символах; конец текста создания page_meta
                          )
        # история фактов: ключ, номер версии, значение, источник; ключ пары (ключ, версия)
        self.db.execute("""\
CREATE TABLE IF NOT EXISTS fact_history (
                          fact_key TEXT NOT NULL,
                          revision INTEGER NOT NULL,
                          value TEXT NOT NULL,
                          source TEXT NOT NULL,
                          PRIMARY KEY(fact_key, revision))"""  # первичный ключ — пара (fact_key, revision); конец текста создания fact_history
                          )
        # текущие факты: по одной строке на ключ — номер последней версии, значение, источник
        self.db.execute("""\
CREATE TABLE IF NOT EXISTS fact_current (
                          fact_key TEXT PRIMARY KEY,
                          revision INTEGER NOT NULL,
                          value TEXT NOT NULL,
                          source TEXT NOT NULL)"""  # source — источник факта; конец текста создания fact_current
                          )
        self.db.commit()  # зафиксировать создание таблиц
        self._pending = ""  # буфер символов, ещё не ставших страницей
        self._indexed_chars = int(self.db.execute(  # проиндексировано символов = наибольший char_end в базе
            "SELECT COALESCE(MAX(char_end), 0) FROM page_meta"  # MAX(char_end); пустая таблица даёт 0
        ).fetchone()[0])  # значение первой строки и первого столбца как целое
        self._max_pending_chars = 0  # наибольший размер буфера (пока 0)
        self._pages_since_commit = 0  # страниц с последней фиксации
        self._commit_every_pages = 64  # фиксация каждые 64 страницы
        self._closed = False  # хранилище открыто

    def _insert_page(self, text: str) -> None:
        """Записать страницу text в индекс: полнотекстовая строка, положение в потоке и фиксация группами.

        Положение страницы: char_start = проиндексировано ранее, char_end = char_start + len(text).
        Каждые _commit_every_pages страниц изменения фиксируются."""
        if not text:  # пустой текст не записывается
            return  # ничего не делать
        start = self._indexed_chars  # начало страницы: сколько символов проиндексировано ранее
        end = start + len(text)  # конец страницы: end = start + len(text)
        cur = self.db.execute("INSERT INTO pages(content) VALUES (?)", (text,))  # вставить текст в полнотекстовую таблицу (параметризованный запрос)
        page_id = int(cur.lastrowid)  # номер новой страницы = последний вставленный rowid
        self.db.execute(  # вставить положение страницы
            "INSERT INTO page_meta(page_id,char_start,char_end,char_count) VALUES (?,?,?,?)",  # вставить положение страницы: четыре параметра
            (page_id, start, end, len(text)),  # номер, начало, конец и длина страницы
        )
        self._indexed_chars = end  # проиндексировано теперь до end
        self._pages_since_commit += 1  # страниц с последней фиксации: + 1
        if self._pages_since_commit >= self._commit_every_pages:  # накопилось 64 и более страниц
            self.db.commit()  # зафиксировать изменения
            self._pages_since_commit = 0  # счётчик страниц обнулён

    def append(self, text: str) -> None:
        """Добавить один фрагмент потока; размер фрагмента не задаёт размер страницы.

        Фрагмент расходуется кусками: в буфер добавляется min(свободное место, остаток
        фрагмента). Когда буфер достигает page_chars, отрезается страница по границе
        перевода строки или пробела (не ближе половины страницы), иначе страница равна
        целому буферу."""
        self._ensure_open()  # хранилище обязано быть открыто
        if not isinstance(text, str):  # аргумент не строка
            raise TypeError("append expects str")  # ошибка типа: ожидается str
        if not text:  # пустая строка
            return  # ничего не делать
        # Большой фрагмент вызывающего кода расходуется постепенно. Собственный буфер
        # хранилища никогда не превышает одной страницы, даже если text много больше.
        offset = 0  # позиция в text, с которой ещё не скопировано
        while offset < len(text):  # пока в тексте есть нескопированное
            room = self.page_chars - len(self._pending)  # свободное место в буфере room = page_chars − len(буфера)
            take = min(room, len(text) - offset)  # сколько взять: take = min(room, остаток текста)
            self._pending += text[offset:offset + take]  # дописать к буферу срез text[offset .. offset + take − 1]
            offset += take  # сдвинуть позицию на take
            self._max_pending_chars = max(self._max_pending_chars, len(self._pending))  # наибольший буфер = max(прошлый, текущий размер)
            if len(self._pending) == self.page_chars:  # буфер достиг размера страницы
                # Границу предпочитаем на переводе строки или пробеле, чтобы не разрывать слова.
                cut = self._pending.rfind("\n")  # позиция последнего перевода строки (−1, если нет)
                if cut < self.page_chars // 2:  # перевод строки отсутствует или в первой половине
                    cut = self._pending.rfind(" ")  # искать последний пробел
                if cut < self.page_chars // 2:  # и пробела во второй половине нет
                    cut = self.page_chars  # резать по концу буфера
                else:  # граница найдена во второй половине
                    cut += 1  # включить сам разделитель в страницу
                page = self._pending[:cut]  # страница = первые cut символов буфера
                self._pending = self._pending[cut:]  # в буфере остаётся хвост с позиции cut
                self._insert_page(page)  # записать страницу в индекс
        # Страницы фиксируются небольшими группами внутри _insert_page: без фиксации
        # на каждую страницу, но с ограниченным объёмом работы после сбоя.

    def ingest(self, fragments: Iterable[str]) -> None:
        """Принять набор фрагментов, не собирая его целиком в памяти."""
        for fragment in fragments:  # по фрагментам по одному
            self.append(fragment)  # добавить фрагмент в поток

    def flush(self) -> None:
        """Дописать буфер последней страницей и зафиксировать все изменения."""
        self._ensure_open()  # хранилище обязано быть открыто
        if self._pending:  # в буфере есть символы
            self._insert_page(self._pending)  # записать их страницей
            self._pending = ""  # очистить буфер
        self.db.commit()  # зафиксировать изменения
        self._pages_since_commit = 0  # счётчик страниц обнулён

    @staticmethod  # статический метод: не получает self
    def _fts_query(query: str) -> str:  # текст запроса; возвращает выражение для MATCH
        """Превратить текст запроса в безопасное выражение полнотекстового поиска.

        Выделяются только буквенно-цифровые слова, каждое берётся в кавычки (внутренняя
        кавычка удваивается) и соединяется через AND. Сырой ввод вызывающего кода
        никогда не попадает в разбор как синтаксис поиска; SQL при этом параметризован."""
        # Белый список лексических слов; каждое в кавычках. Сырой ввод не передаётся
        # как синтаксис полнотекстового поиска; сам SQL параметризован.
        terms = re.findall(r"[^\W_]+", query, flags=re.UNICODE)  # слова запроса: непрерывные цепочки букв и цифр (без знака подчёркивания)
        if not terms:  # слов нет
            return ""  # пустой запрос
        return " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)  # «"слово1" AND "слово2" ...», кавычки внутри слова удвоены

    def search(self, query: str, limit: int | None = None,  # текст запроса; предел числа страниц (None — предел хранилища)
               page_ids: Iterable[int] | None = None) -> list[MemoryHit]:  # искать только на этих страницах (None — по всему индексу)
        """Вернуть ограниченное число полнотекстовых совпадений; можно ограничить набор страниц рабочего набора.

        Перед поиском буфер дописывается. Число результатов k = min(limit или max_query_pages,
        max_query_pages). Страницы упорядочиваются по bm25 (меньше — релевантнее)."""
        self._ensure_open()  # хранилище обязано быть открыто
        self.flush()  # непроиндексированный буфер должен попасть в поиск
        fts = self._fts_query(query)  # выражение поиска
        if not fts:  # в запросе нет слов
            return []  # пустой список результатов
        k = self.max_query_pages if limit is None else int(limit)  # предел результатов k
        if k < 1:  # предел меньше единицы
            raise ValueError("limit must be >= 1")  # ошибка: предел меньше 1
        k = min(k, self.max_query_pages)  # предел не может превышать max_query_pages
        if page_ids is None:  # искать по всему индексу
            # столбцы: номер страницы, начало, конец, оценка bm25, полный текст и
            # фрагмент (столбец 0, найденные слова в [match]…[/match], пропуски « … », до 40 токенов);
            # страницы соединяются со своим положением по номеру; условие — совпадение с
            # выражением поиска, порядок по bm25, не более k строк
            rows = self.db.execute(  # строки результата
                """\
SELECT p.rowid, m.char_start, m.char_end, bm25(pages), p.content,
                          snippet(pages, 0, '[match]', '[/match]', ' … ', 40)
                   FROM pages AS p JOIN page_meta AS m ON m.page_id=p.rowid
                   WHERE pages MATCH ? ORDER BY bm25(pages) LIMIT ?"""  # условие совпадения с запросом, порядок по bm25, не более k строк; конец текста запроса
                   ,
                (fts, k),  # параметры: выражение поиска и предел
            ).fetchall()  # выбрать все строки
        else:  # искать только на заданных страницах
            allowed = sorted({int(pid) for pid in page_ids})  # разрешённые номера: целые, без повторов, по возрастанию
            if not allowed:  # набор пуст
                return []  # пустой список результатов
            self.db.execute("CREATE TEMP TABLE IF NOT EXISTS active_page_filter (page_id INTEGER PRIMARY KEY)")  # временная таблица-фильтр страниц (создаётся при первом использовании)
            self.db.execute("DELETE FROM active_page_filter")  # очистить её от прошлого запроса
            self.db.executemany("INSERT OR IGNORE INTO active_page_filter(page_id) VALUES (?)",  # вставить все разрешённые номера
                                ((pid,) for pid in allowed))  # по одному параметру-кортежу на номер
            # тот же запрос, что и выше, плюс соединение с фильтром: остаются только страницы из него
            rows = self.db.execute(  # строки результата
                """\
SELECT p.rowid, m.char_start, m.char_end, bm25(pages), p.content,
                          snippet(pages, 0, '[match]', '[/match]', ' … ', 40)
                   FROM pages AS p JOIN page_meta AS m ON m.page_id=p.rowid
                   JOIN active_page_filter AS f ON f.page_id=p.rowid
                   WHERE pages MATCH ? ORDER BY bm25(pages) LIMIT ?"""  # условие совпадения с запросом, порядок по bm25, не более k строк; конец текста запроса
                   ,
                (fts, k),  # параметры: выражение поиска и предел
            ).fetchall()  # выбрать все строки
        return [MemoryHit(int(r[0]), int(r[1]), int(r[2]), float(r[3]), str(r[4]), str(r[5]))  # каждая строка -> MemoryHit: номер, начало, конец, оценка, текст, фрагмент
                for r in rows]  # по всем строкам

    @staticmethod  # статический метод: не получает self
    def _normalize_fact_key(key: str) -> str:  # ключ факта; возвращает нормализованный ключ
        """Нормализовать ключ факта: нижний регистр, пробелы сжаты в один, крайние убраны.

        Пустой ключ — ValueError."""
        normalized = " ".join(str(key).casefold().split())  # str(key) -> нижний регистр -> разбиение по пробелам -> соединение одним пробелом
        if not normalized:  # ключ не содержит ни одного символа, кроме пробелов
            raise ValueError("fact key must be non-empty")  # ошибка: ключ факта пуст
        return normalized  # нормализованный ключ

    def store_fact(self, key: str, value: str, source: str = "") -> FactRecord:
        """Сохранить структурированный факт: и последнее значение, и всю историю версий.

        Номер версии = наибольший номер этого ключа в истории + 1. Строка дописывается
        в fact_history; в fact_current ключ вставляется, а при его наличии обновляется.
        Это явный интерфейс: языковая модель или разборщик сами решают, какой текст
        считать фактом, и задают ключ и значение. Исходный текст остаётся в полнотекстовом
        архиве для последующей проверки и выборки."""
        self._ensure_open()  # хранилище обязано быть открыто
        fact_key = self._normalize_fact_key(key)  # нормализованный ключ факта
        value = str(value)  # значение как строка
        source = str(source)  # источник как строка
        row = self.db.execute(  # наибольший номер версии ключа в истории
            "SELECT COALESCE(MAX(revision), 0) FROM fact_history WHERE fact_key=?",  # MAX(revision); версий нет — 0
            (fact_key,),  # параметр — ключ факта
        ).fetchone()  # одна строка результата
        revision = int(row[0]) + 1  # новый номер версии = прошлый наибольший + 1
        self.db.execute(  # дописать версию в историю
            "INSERT INTO fact_history(fact_key,revision,value,source) VALUES (?,?,?,?)",  # четыре параметра
            (fact_key, revision, value, source),  # ключ, номер версии, значение, источник
        )
        # вставить строку текущего факта; если ключ уже есть — заменить её номер версии,
        # значение и источник новыми (excluded — строка, которую пытались вставить)
        self.db.execute(  # вставить или обновить строку текущего факта
            """\
INSERT INTO fact_current(fact_key,revision,value,source) VALUES (?,?,?,?)
               ON CONFLICT(fact_key) DO UPDATE SET
               revision=excluded.revision, value=excluded.value, source=excluded.source"""  # при конфликте по ключу заменить версию, значение и источник; конец текста запроса
               ,
            (fact_key, revision, value, source),  # ключ, номер версии, значение, источник
        )
        self.db.commit()  # зафиксировать изменения
        self._pages_since_commit = 0  # фиксация уже сделана — счётчик страниц обнулён
        return FactRecord(fact_key, revision, value, source)  # запись о сохранённой версии

    def get_fact(self, key: str) -> FactRecord | None:
        """Вернуть текущую версию факта с ключом key либо None, если факта нет."""
        self._ensure_open()  # хранилище обязано быть открыто
        fact_key = self._normalize_fact_key(key)  # нормализованный ключ факта
        row = self.db.execute(  # строка текущего факта
            "SELECT fact_key,revision,value,source FROM fact_current WHERE fact_key=?",  # ключ, номер версии, значение, источник по ключу
            (fact_key,),  # параметр — ключ факта
        ).fetchone()  # одна строка или None
        return None if row is None else FactRecord(str(row[0]), int(row[1]), str(row[2]), str(row[3]))  # None, если строки нет, иначе FactRecord из четырёх столбцов

    def fact_history(self, key: str) -> list[FactRecord]:
        """Вернуть все версии факта с ключом key в порядке возрастания номера версии."""
        self._ensure_open()  # хранилище обязано быть открыто
        fact_key = self._normalize_fact_key(key)  # нормализованный ключ факта
        rows = self.db.execute(  # строки истории факта
            "SELECT fact_key,revision,value,source FROM fact_history WHERE fact_key=? ORDER BY revision",  # по ключу, в порядке номера версии
            (fact_key,),  # параметр — ключ факта
        ).fetchall()  # все строки
        return [FactRecord(str(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in rows]  # каждая строка -> FactRecord

    def get_stats(self) -> MemoryStats:
        """Вернуть статистику хранилища: страницы, символы, буфер и размер файлов базы."""
        self._ensure_open()  # хранилище обязано быть открыто
        pages = int(self.db.execute("SELECT COUNT(*) FROM page_meta").fetchone()[0])  # число страниц = число строк page_meta
        self.db.commit()  # зафиксировать изменения
        self._pages_since_commit = 0  # счётчик страниц обнулён
        if self.path == ":memory:":  # база в оперативной памяти
            db_bytes = 0  # файла нет — размер 0
        else:  # база на диске
            db_bytes = sum(  # суммарный размер существующих файлов базы
                os.path.getsize(p) for p in (self.path, self.path + "-wal", self.path + "-shm")  # самой базы, журнала WAL и файла общей памяти
                if os.path.exists(p)  # только если файл существует
            )
        return MemoryStats(  # статистика хранилища
            pages=pages,  # число страниц
            indexed_chars=self._indexed_chars,  # проиндексировано символов
            pending_chars=len(self._pending),  # в буфере символов
            max_pending_chars=self._max_pending_chars,  # наибольший буфер
            page_chars=self.page_chars,  # предел страницы
            db_bytes=db_bytes,  # размер файлов базы
        )

    def _ensure_open(self) -> None:
        """Проверить, что хранилище не закрыто; иначе RuntimeError."""
        if self._closed:  # хранилище закрыто
            raise RuntimeError("context store is closed")  # ошибка: хранилище закрыто

    def close(self) -> None:
        """Закрыть хранилище: дописать буфер, зафиксировать и закрыть соединение (повторно — ничего)."""
        if not self._closed:  # ещё не закрыто
            self.flush()  # дописать буфер и зафиксировать
            self.db.close()  # закрыть соединение с базой
            self._closed = True  # пометить закрытым

    def __enter__(self) -> "DiskBackedContext":  # вход в блок with возвращает сам объект
        """Вход в блок with: проверить, что хранилище открыто, и вернуть self."""
        self._ensure_open()  # хранилище обязано быть открыто
        return self  # сам объект

    def __exit__(self, exc_type, exc, tb) -> None:  # тип, значение и трассировка исключения (не используются)
        """Выход из блока with (в том числе по исключению): закрыть хранилище."""
        self.close()  # закрыть хранилище


class StreamingContext:
    """Небольшой сборщик подсказки поверх дискового индекса контекста.

    Полная история никогда не склеивается: build_context возвращает не более
    max_context_chars символов найденного текста вместе с запросом пользователя."""

    def __init__(self, memory: DiskBackedContext, max_context_chars: int = 8192,  # индекс контекста и предел размера подсказки в символах
                 utility_index=None):  # индекс полезности страниц (может быть None)
        """Создать сборщик над хранилищем memory с пределом max_context_chars и индексом полезности utility_index."""
        if max_context_chars < 1:  # предел неположителен
            raise ValueError("max_context_chars must be >= 1")  # ошибка: предел меньше 1
        self.memory = memory  # дисковый индекс контекста
        self.max_context_chars = int(max_context_chars)  # предел размера подсказки как целое
        self.utility_index = utility_index  # индекс полезности (или None)

    def search(self, query: str, top_k: int = 4) -> list[MemoryHit]:
        """Искать сначала в выученном рабочем наборе, при нехватке — во всём архиве.

        Без индекса полезности — обычный поиск по архиву. Иначе: поиск по страницам
        рабочего набора; если нашлось не меньше top_k — они и возвращаются. Иначе
        оба списка объединяются по порядку (рабочий набор первым) без повторов по
        (страница, начало, конец), до top_k результатов."""
        if self.utility_index is None:  # индекс полезности не подключён
            return self.memory.search(query, limit=top_k)  # поиск по всему архиву с пределом top_k
        cached_ids = self.utility_index.working_set.cached_page_ids  # номера страниц рабочего набора
        cached_hits = self.memory.search(query, limit=top_k, page_ids=cached_ids) if cached_ids else []  # совпадения в рабочем наборе (пустой набор — пустой список)
        if len(cached_hits) >= top_k:  # рабочий набор дал достаточно
            return cached_hits[:top_k]  # первые top_k совпадений
        archive_hits = self.memory.search(query, limit=top_k)  # совпадения по всему архиву
        merged: list[MemoryHit] = []  # объединённый результат
        seen: set[tuple[int, int, int]] = set()  # уже взятые (страница, начало, конец)
        for hit in cached_hits + archive_hits:  # сначала рабочий набор, затем архив
            key = (hit.page_id, hit.char_start, hit.char_end)  # ключ от повторов
            if key not in seen:  # такое совпадение ещё не добавлено
                seen.add(key)  # запомнить ключ
                merged.append(hit)  # добавить совпадение
            if len(merged) >= top_k:  # набралось top_k результатов
                break  # прекратить объединение
        return merged  # объединённый список

    def feedback(self, page_id: int, reward: float = 1.0):
        """Передать в индекс полезности оценку reward полезности страницы page_id (принятой или отвергнутой)."""
        if self.utility_index is None:  # индекс полезности не подключён
            raise RuntimeError("no utility index is attached to this StreamingContext")  # ошибка: индекс полезности не подключён
        return self.utility_index.record_page_feedback(page_id, reward=reward)  # записать оценку страницы

    def refresh_utility_memory(self):
        """Обновить ограниченный рабочий набор после серии оценок страниц."""
        if self.utility_index is None:  # индекс полезности не подключён
            raise RuntimeError("no utility index is attached to this StreamingContext")  # ошибка: индекс полезности не подключён
        return self.utility_index.refresh()  # перестроить рабочий набор

    def ingest(self, fragments: Iterable[str]) -> None:
        """Передать фрагменты потока в дисковый индекс."""
        self.memory.ingest(fragments)  # принять фрагменты по одному

    def build_context(self, query: str, top_k: int = 4) -> str:
        """Собрать текст подсказки: найденные фрагменты с пометкой источника и хвост запроса.

        Бюджет B = max(0, max_context_chars − len(хвост запроса) − 1). Каждый кусок —
        пометка «[memory page=… chars=начало:конец]», перевод строки и фрагмент
        находки (или весь текст страницы, если фрагмента нет), обрезанные до B;
        после добавления B уменьшается на длину куска + 1. Куски соединяются переводом
        строки, за ними через перевод строки идёт хвост запроса."""
        hits = self.search(query, top_k=top_k)  # не более top_k совпадений
        query_tail = query[-self.max_context_chars:]  # последние max_context_chars символов запроса
        budget = max(0, self.max_context_chars - len(query_tail) - 1)  # бюджет на найденный текст B
        pieces: list[str] = []  # куски найденного текста
        for hit in hits:  # по совпадениям в порядке релевантности
            if budget <= 0:  # бюджет исчерпан
                break  # прекратить сборку
            provenance = f"[memory page={hit.page_id} chars={hit.char_start}:{hit.char_end}]\n"  # пометка источника: номер страницы и границы символов, затем перевод строки
            # Берётся фрагмент находки вокруг совпавших слов, а не начало страницы:
            # нужное свидетельство может лежать в её конце.
            evidence = hit.snippet if hit.snippet else hit.text  # фрагмент находки, а если его нет — весь текст страницы
            piece = (provenance + evidence)[:budget]  # кусок = пометка + свидетельство, обрезанные до бюджета
            if piece:  # кусок не пуст
                pieces.append(piece)  # добавить кусок
                budget -= len(piece) + 1  # бюджет: B − len(куска) − 1 (запас на разделитель)
        retrieved = "\n".join(pieces)  # найденный текст: куски через перевод строки
        return (retrieved + "\n" if retrieved else "") + query_tail  # найденный текст с переводом строки (если он есть) + хвост запроса
