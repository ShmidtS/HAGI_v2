"""Внешнее хранилище долговременной памяти: архив эпизодов и семантический поиск.

Ключевые идеи:
  - Архив лежит на диске в SQLite с полнотекстовым индексом FTS5: весь текст
    сохраняется независимо от размера рабочей памяти.
  - Текст нарезается на страницы по page_chars символов (page_id = rowid страницы).
  - Поиск гибридный, двумя фазами:
      1) лексическая: FTS5 отбирает до 64 страниц, содержащих слова запроса;
      2) семантическая: страницы повторно упорядочиваются по косинусной
         близости векторов. Векторы страниц и запроса даёт кодировщик
        GemmaEmbedder (первая фраза — до 512 символов).
    Скалярное произведение _cos(q, p) = q · v_p (без деления на нормы).
  - Векторы считаются лениво и только для новых страниц.

Экспортирует: DiskBackedContext, StreamingContext, MemoryHit (из streaming),
SemanticMemory.
"""
from hagi.memory_store.streaming import (  # импорт из модуля потокового хранения; noqa: F401 — имена перепродаются
    DiskBackedContext, StreamingContext, MemoryHit)  # перепродажа классов контекста и результата поиска наружу


class SemanticMemory:
    """Семантическая память: дисковый архив страниц + поиск FTS5 + повторное
    упорядочивание по близости эмбеддингов.

        Атрибуты:
          store — дисковый архив DiskBackedContext (SQLite + FTS5);
          _vectors — словарь page_id -> вектор страницы (заполняется лениво);
          _pages — словарь page_id -> текст страницы (заполняется при поиске)."""

    def __init__(self, db_path: str, page_chars: int = 2048):
        """Открыть память по пути db_path; страницы длиной page_chars символов."""
        self.store = DiskBackedContext(db_path, page_chars=page_chars)  # создать/открыть дисковый архив
        self._vectors = {}  # page_id -> вектор страницы (пока пусто, векторы считает лениво)

    def add(self, text: str) -> None:
        """Добавить текст в архив и сразу сбросить буфер на диск."""
        self.store.append(text)  # дописать текст в буфер архива
        self.store.flush()  # сбросить буфер: страницы записаны в базу

    def _ensure_vectors(self):
        """Дополнить кэш векторов: вектор для каждой страницы, которой его ещё нет.

            Причина инкрементности: страницы, добавленные после первого поиска,
            иначе остались бы без вектора, и поиск их не находил бы.
            Читаются все страницы (rowid, content); внедряются только те rowid,
            которых нет в _vectors; текст страницы для эмбеддинга — первые
            512 символов."""
        from hagi.base.embeddings import GemmaEmbedder  # кодировщик текста в вектор (импортируется лениво)
        gemma = GemmaEmbedder.get()  # единственный экземпляр кодировщика
        self.store._ensure_open()  # убедиться, что база открыта
        rows = self.store.db.execute(  # прочитать из таблицы pages все пары
            "SELECT rowid, content FROM pages").fetchall()  # (rowid страницы, текст)
        texts = {rid: content for rid, content in rows}  # словарь rowid -> текст
        missing = [pid for pid in texts if pid not in self._vectors]  # страницы без вектора
        if missing:  # есть новые страницы
            import numpy as np  # численные массивы
            vecs = gemma.embed([texts[pid][:512] for pid in missing])  # векторы новых страниц, каждая по первым 512 символам
            for pid, v in zip(missing, vecs):  # пара (номер страницы, её вектор)
                self._vectors[pid] = (v.detach().cpu().numpy()  # тензор: копия на CPU и в массив numpy
                                      if hasattr(v, "cpu")  # если у объекта есть метод cpu (это тензор)
                                      else np.asarray(v))  # иначе: массив numpy как есть
        self._pages = texts  # обновить словарь текстов страниц

    def search(self, q: str, limit: int = 4):
        """Псевдоним query для совместимости с memory_agent: вернуть до limit
        объектов с атрибутами page_id и text."""
        res = self.query(q, top_k=limit)  # список кортежей (близость, page_id, текст)

        class _Hit:  # облегчённое попадание поиска
            def __init__(self, pid, text):  # номер страницы pid и её текст
                self.page_id = pid  # атрибут page_id
                self.text = text  # атрибут text
        return [_Hit(pid, text) for _cos, pid, text in res]  # для каждого кортежа — объект _Hit(page_id, text)

    def query(self, q: str, top_k: int = 4):
        """Гибридный поиск по запросу q; вернуть до top_k кортежей
        (близость, page_id, текст) в порядке убывания близости.

            1) Лексическая фаза: из запроса берутся слова из латиницы/цифр длиной
               не меньше 3 (первые 8), каждое берётся в кавычки и экранируется;
               FTS5 отбирает до 64 страниц, где они встречаются.
               Если совпадений нет или запрос к FTS5 ошибочен — кандидаты все
               страницы с векторами.
            2) Фаза близости: score_p = q · v_p, где q — вектор первых 512
               символов запроса; список сортируется по score по убыванию."""
        import numpy as np  # численные массивы
        from hagi.base.embeddings import GemmaEmbedder  # кодировщик текста в вектор
        gemma = GemmaEmbedder.get()  # единственный экземпляр кодировщика
        self._ensure_vectors()  # у всех страниц должны быть векторы
        qv = gemma.embed([q[:512]])[0]  # вектор запроса по первым 512 символам
        if hasattr(qv, "cpu"):  # если это тензор
            qv = qv.detach().cpu().numpy()  # копия на CPU и в массив numpy

        def _cos(pid):  # близость страницы pid к запросу
            v = self._vectors[pid]  # вектор страницы
            return float(np.dot(qv, v))  # скалярное произведение q · v_p

        # лексическая фаза: совпадение слов запроса по FTS5
        cand = None  # множество кандидатов пока не найдено
        try:  # запрос к FTS5 может быть ошибочным — тогда полный перебор
            terms = [t for t in  # слова запроса
                     __import__("re").findall(r"[A-Za-z0-9]{3,}", q)][:8]  # последовательности латиницы/цифр длиной ≥ 3, первые 8
            if terms:  # если есть хоть одно слово
                esc = " ".join('"' + t.replace('"', '""') + '"'  # каждое слово в кавычках, внутренняя кавычка удваивается
                               for t in terms)  # слова соединены пробелом
                rows = self.store.db.execute(  # найти страницы, соответствующие выражению
                    "SELECT rowid FROM pages WHERE pages MATCH ? "  # FTS5-поиск по таблице pages
                    "LIMIT 64", (esc,)).fetchall()  # не больше 64 строк
                if rows:  # есть хотя бы одна страница
                    cand = {r[0] for r in rows}  # множество rowid страниц-кандидатов
        except Exception:  # любая ошибка FTS5 (например, ошибочный запрос)
            cand = None  # кандидатов нет — ниже будет перебор всех страниц
        ids = sorted(cand) if cand else sorted(self._vectors)  # номера кандидатов по возрастанию; если пусто — все страницы с векторами
        scored = [(_cos(pid), pid, self._pages.get(pid, ""))  # кортежи (близость, page_id, текст или "")
                  for pid in ids if pid in self._vectors]  # только кандидаты с вектором
        scored.sort(reverse=True)  # сортировка по близости по убыванию
        return scored[:top_k]  # первые top_k результатов
