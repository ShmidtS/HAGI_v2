"""Обучаемая по исходам модель полезности: небольшой рабочий набор поверх долговременной памяти.

    Исходный архив остаётся источником истины и никогда не меняется. Модуль учится на ответах
    поиска (нашлось полезное или нет), какие признаки текста повторяются в полезных записях,
    и выбирает из перебираемого архива небольшой быстрый рабочий набор. Обучение намеренно
    простое, локальное, детерминированное и доступное осмотру; это не языковая модель и не
    определитель истинности.

    Признаки — слова текста (строчные буквы или цифры, без подчёркивания) длиной не меньше
    min_token_length, без цифр и без стоп-слов. Каждое слово хэшируется в номер ячейки:
      idx(слово) = blake2b(слово, 8 байт) как little-endian целое mod feature_bins.
    Хэширование держит размер модели постоянным при любом размере архива.

    Оценка текста по весам w (число ячеек в тексте m):
      score(текст) = (Σ_{i ∈ признаки} w_i) / sqrt(m).

    Обучение на одном ответе с наградой r из [−1, 1]:
      w ← decay·w;  w_i ← w_i + learning_rate·r / sqrt(m) для всех i из признаков текста.
    Положительная награда поднимает веса признаков полезной записи, отрицательная — опускает;
    затухание w ← decay·w при каждом событии позволяет набору следовать за сменой интересов.

    Рабочий набор UtilityWorkingSet ограничен по размеру: при обновлении перебирается архив,
      каждая запись получает score, и в куче из capacity элементов остаются лучшие;
      при равенстве выигрывает запись с большим идентификатором (сравнение кортежей
      (score, id)). Память отбора O(capacity), а не O(архив).
    Инвертированный индекс SQLiteUtilityIndex хранит на диске пары (страница, признак);
      при обновлении оцениваются только страницы с ненулевыми весами признаков:
      score = (Σ весов активных признаков страницы) / sqrt(число признаков страницы).

    Экспортирует: MemoryUseLearner, UtilityWorkingSet, SQLiteUtilityIndex, UtilityRecord,
    UtilityFeedback, UtilityRefreshStats."""
from __future__ import annotations  # отложенное вычисление аннотаций типов

from dataclasses import dataclass  # классы-записи
from hagi.base.hashing import hash_feature_bin  # номер ячейки признака из хеша слова
from pathlib import Path  # работа с путями файлов
import re  # регулярные выражения
import time  # замер времени
from typing import Iterable, Iterator  # аннотации типов

import numpy as np  # численные массивы

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)  # слово: непрерывная цепочка букв или цифр (без знака подчёркивания)
_STOP = {  # стоп-слова: не становятся признаками
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are", "is",  # английские служебные слова
    "has", "have", "had", "into", "onto", "about", "after", "before", "under", "over",  # английские служебные слова
    "record", "note", "entry", "fact", "value", "item", "asset", "uses", "using",  # общие слова о записях
    "used", "shows", "states", "says", "saying", "according", "reference", "details",  # общие слова о записях
    "what", "which", "when", "where", "who", "how", "why", "did", "does", "can",  # вопросительные и модальные слова
    "could", "would", "should", "please", "tell", "me", "find", "look", "up",  # модальные слова и слова запросов
}


def _hash_feature(token: str, bins: int) -> int:  # номер ячейки признака: 0..bins−1 (общая реализация — hagi.base.hashing)
    return hash_feature_bin(token, bins)  # номер ячейки = hash_token_u64(token) mod bins


@dataclass(frozen=True)  # неизменяемая запись
class UtilityRecord:
    """Запись архива для оценки полезности: идентификатор и текст."""
    record_id: str  # идентификатор записи (у страницы архива — её номер)
    text: str  # текст записи


@dataclass(frozen=True)  # неизменяемая запись
class UtilityRefreshStats:
    """Итоги обновления рабочего набора."""
    scanned: int  # сколько записей просмотрено
    retained: int  # сколько записей попало в набор
    seconds: float  # затраченное время в секундах
    threshold_score: float  # наименьшая оценка среди вошедших в набор (0.0, если набор пуст)


@dataclass(frozen=True)  # неизменяемая запись
class UtilityFeedback:
    """Сведения об одном принятом ответе поиска."""
    record_id: str  # идентификатор записи
    was_cached: bool  # была ли запись в рабочем наборе до этого ответа
    reward: float  # награда из [−1, 1]
    update_count: int  # общее число обновлений модели после этого ответа


class MemoryUseLearner:
    """Онлайновый ученик полезности текста, обновляемый по исходам поиска.

        Положительная награда увеличивает веса признаков полезной записи, отрицательная —
        уменьшает. Все веса слегка затухают при каждом событии, поэтому рабочий набор может
        следить за сменой интересов. Хэширование признаков держит размер модели постоянным
        при любом размере архива."""

    VERSION = 1  # версия сохраняемой модели

    def __init__(self, feature_bins: int = 4096, learning_rate: float = 1.0,  # создать ученика
                 decay: float = 0.97, min_token_length: int = 3):  # decay — множитель затухания; min_token_length — минимальная длина слова
        if feature_bins < 128:  # ячеек меньше 128
            raise ValueError("feature_bins must be >=128")  # ошибка
        if learning_rate <= 0:  # скорость обучения не положительна
            raise ValueError("learning_rate must be >0")  # ошибка
        if not 0.0 < decay <= 1.0:  # затухание вне интервала (0, 1]
            raise ValueError("decay must be in (0,1]")  # ошибка
        if min_token_length < 2:  # минимальная длина слова меньше 2
            raise ValueError("min_token_length must be >=2")  # ошибка
        self.feature_bins = int(feature_bins)  # число ячеек признаков
        self.learning_rate = float(learning_rate)  # скорость обучения
        self.decay = float(decay)  # множитель затухания
        self.min_token_length = int(min_token_length)  # минимальная длина слова
        self.weights = np.zeros(self.feature_bins, dtype=np.float32)  # веса ячеек: сначала все 0
        self.update_count = 0  # число обновлений

    def feature_indices(self, text: str) -> np.ndarray:  # номера ячеек признаков текста (отсортированные, без повторов)
        tokens = {  # множество слов в нижнем регистре
            t.casefold() for t in _TOKEN_RE.findall(str(text))  # по всем словам текста, приведённым к нижнему регистру
            if len(t) >= self.min_token_length and not any(ch.isdigit() for ch in t)  # длина не меньше min_token_length и нет цифр
            and t.casefold() not in _STOP  # и слово не стоп-слово
        }
        if not tokens:  # признаков нет
            return np.empty(0, dtype=np.int64)  # пустой массив
        return np.asarray(sorted({_hash_feature(t, self.feature_bins) for t in tokens}), dtype=np.int64)  # сортированные номера ячеек {idx(слово)}

    def score(self, text: str) -> float:  # оценка текста
        """Оценка текста: score = (Σ_{i ∈ признаки} w_i) / sqrt(m), m — число признаков; 0.0 без признаков."""
        idx = self.feature_indices(text)  # признаки текста
        if idx.size == 0:  # признаков нет
            return 0.0  # оценка 0
        return float(np.sum(self.weights[idx], dtype=np.float64) / np.sqrt(float(idx.size)))  # оценка текста: score = Σ w_i / sqrt(m)

    def observe(self, text: str, reward: float = 1.0) -> None:  # одно обновление по исходу поиска
        """Обновить веса по исходу поиска: w ← decay·w, затем w_i += learning_rate·r / sqrt(m)."""
        reward = float(reward)  # награда как число
        if not -1.0 <= reward <= 1.0:  # награда вне [−1, 1]
            raise ValueError("reward must be in [-1,1]")  # ошибка
        self.weights *= self.decay  # затухание всех весов: w ← decay·w
        idx = self.feature_indices(text)  # признаки текста
        if idx.size:  # признаки есть
            self.weights[idx] += np.float32(self.learning_rate * reward / np.sqrt(float(idx.size)))  # обновление веса: w_i ← w_i + learning_rate·r / sqrt(m)
        self.update_count += 1  # обновлений: +1

    def save(self, path: str | Path) -> None:  # сохранить модель в сжатый файл npz
        target = Path(path)  # путь как объект Path
        target.parent.mkdir(parents=True, exist_ok=True)  # создать каталог файла, если нужно
        np.savez_compressed(  # сохранить массивы в сжатый архив
            target,  # путь файла
            version=np.asarray([self.VERSION], dtype=np.int32),  # версия модели
            feature_bins=np.asarray([self.feature_bins], dtype=np.int32),  # число ячеек
            learning_rate=np.asarray([self.learning_rate], dtype=np.float32),  # скорость обучения
            decay=np.asarray([self.decay], dtype=np.float32),  # затухание
            min_token_length=np.asarray([self.min_token_length], dtype=np.int32),  # минимальная длина слова
            update_count=np.asarray([self.update_count], dtype=np.int64),  # число обновлений
            weights=self.weights,  # веса ячеек
        )

    @classmethod  # метод класса: создаёт объект, не имея экземпляра
    def load(cls, path: str | Path) -> "MemoryUseLearner":  # загрузить модель из файла npz
        """Загрузить модель из файла, созданного save; версия обязана совпадать с VERSION."""
        with np.load(path, allow_pickle=False) as payload:  # открыть архив массивов (без pickle — безопасно)
            if int(payload["version"][0]) != cls.VERSION:  # версия не совпадает
                raise ValueError("unsupported utility-model checkpoint")  # ошибка: неподдерживаемая версия
            obj = cls(  # создать ученика с сохранёнными параметрами
                feature_bins=int(payload["feature_bins"][0]),  # число ячеек
                learning_rate=float(payload["learning_rate"][0]),  # скорость обучения
                decay=float(payload["decay"][0]),  # затухание
                min_token_length=int(payload["min_token_length"][0]),  # минимальная длина слова
            )
            weights = np.asarray(payload["weights"], dtype=np.float32)  # сохранённые веса
            if weights.shape != (obj.feature_bins,):  # форма весов не равна (feature_bins,)
                raise ValueError("invalid utility-model weight shape")  # ошибка: неверная форма весов
            obj.weights[:] = weights  # скопировать веса в модель
            obj.update_count = int(payload["update_count"][0])  # восстановить число обновлений
        return obj  # готовый ученик


class UtilityWorkingSet:
    """Ограниченный перестраиваемый кэш, выбираемый из внешнего перебираемого архива.

        record_feedback вызывается только когда исход внешнего поиска уже известен. Сам архив
        этим кэшем никогда не удаляется и не меняется. Перестройка идёт через кучу из k
        наибольших, поэтому память отбора O(capacity), а не O(архив)."""

    def __init__(self, capacity: int, learner: MemoryUseLearner | None = None):  # создать набор; learner — ученик или None (создаётся новый)
        if capacity < 1:  # вместимость меньше 1
            raise ValueError("capacity must be >=1")  # ошибка
        self.capacity = int(capacity)  # вместимость набора
        self.learner = learner or MemoryUseLearner()  # ученик полезности
        self.cached_ids: set[str] = set()  # идентификаторы записей в наборе
        self._feedback_since_refresh = 0  # число ответов поиска с последней перестройки
        self.last_refresh: UtilityRefreshStats | None = None  # итоги последней перестройки

    def contains(self, record_id: str) -> bool:  # есть ли запись в наборе
        return str(record_id) in self.cached_ids  # проверка принадлежности: record_id ∈  cached_ids

    def record_feedback(self, record: UtilityRecord, reward: float = 1.0) -> UtilityFeedback:  # принять исход поиска по записи
        record_id = str(record.record_id)  # идентификатор как строка
        was_cached = record_id in self.cached_ids  # была ли запись в наборе до обновления
        self.learner.observe(record.text, reward=reward)  # обновить веса ученика по тексту записи
        self._feedback_since_refresh += 1  # ответов с последней перестройки: +1
        return UtilityFeedback(record_id, was_cached, float(reward), self.learner.update_count)  # сведения об ответе

    def refresh(self, records: Iterable[UtilityRecord]) -> UtilityRefreshStats:  # перестроить набор по перебираемому архиву
        """Выбрать записи с наибольшей оценкой при ограниченной дополнительной памяти."""
        import heapq  # куча на списке

        started = time.perf_counter()  # начало замера времени
        heap: list[tuple[float, str]] = []  # куча из не более capacity кортежей (score, id)
        scanned = 0  # число просмотренных записей
        for record in records:  # по всем записям архива
            scanned += 1  # просмотрено: +1
            rid = str(record.record_id)  # идентификатор как строка
            score = self.learner.score(record.text)  # оценка текста записи
            # устойчивый разбор равенства по идентификатору; хранить только capacity лучших идентификаторов
            item = (score, rid)  # кандидат — кортеж (оценка, идентификатор)
            if len(heap) < self.capacity:  # куча не заполнена
                heapq.heappush(heap, item)  # добавить кандидата
            elif item > heap[0]:  # кандидат больше худшего в куче
                heapq.heapreplace(heap, item)  # заменить худшего кандидатом
        self.cached_ids = {rid for _, rid in heap}  # новый набор — идентификаторы из кучи
        self._feedback_since_refresh = 0  # счётчик ответов обнулён
        threshold = min((score for score, _ in heap), default=0.0)  # наименьшая оценка в куче (0.0, если куча пуста)
        stats = UtilityRefreshStats(scanned, len(self.cached_ids), time.perf_counter() - started, float(threshold))  # итоги: просмотрено, оставлено, секунд, порог
        self.last_refresh = stats  # запомнить итоги
        return stats  # вернуть итоги

    def feedback_and_refresh(self, record: UtilityRecord,  # принять ответ и сразу перестроить набор
                             records: Iterable[UtilityRecord], reward: float = 1.0) -> tuple[UtilityFeedback, UtilityRefreshStats]:  # records — архив; reward — награда
        feedback = self.record_feedback(record, reward=reward)  # обновить ученика по записи
        return feedback, self.refresh(records)  # сведения об ответе и итоги перестройки

    def record_page_feedback(self, memory, page_id: int, reward: float = 1.0) -> UtilityFeedback:  # принять исход поиска по странице SQLite-архива
        """Учёба на найденной странице SQLite-архива (например, оценённом результате поиска)."""
        row = memory.db.execute("SELECT content FROM pages WHERE rowid=?", (int(page_id),)).fetchone()  # текст страницы с номером page_id
        if row is None:  # страницы нет
            raise KeyError(f"archive page {page_id} does not exist")  # ошибка: страница не существует
        return self.record_feedback(UtilityRecord(str(int(page_id)), str(row[0])), reward=reward)  # обновить ученика по тексту страницы

    def refresh_from_memory(self, memory) -> UtilityRefreshStats:  # перестроить набор, перебирая страницы SQLite-архива
        """Перестроить ограниченный набор страниц, читая строки архива SQLite одну за другой."""
        cursor = memory.db.execute("SELECT rowid,content FROM pages ORDER BY rowid")  # курсор по страницам по возрастанию номера
        def records() -> Iterator[UtilityRecord]:  # генератор записей из курсора
            for page_id, content in cursor:  # по строкам курсора
                yield UtilityRecord(str(int(page_id)), str(content))  # запись (номер страницы, текст)
        return self.refresh(records())  # перестроить набор по генератору

    @property  # свойство: вызывается без скобок
    def cached_page_ids(self) -> set[int]:  # идентификаторы набора как целые номера страниц
        """Идентификаторы набора как целые числа — для прямого использования с номерами страниц архива."""
        result: set[int] = set()  # накопитель номеров
        for value in self.cached_ids:  # по идентификаторам набора
            try:  # не каждый идентификатор обязан быть числом
                result.add(int(value))  # добавить как целое число
            except ValueError:  # идентификатор не число
                continue  # пропустить
        return result  # множество целых номеров

    def state(self) -> dict:  # снимок состояния набора
        return {  # словарь состояния
            "capacity": self.capacity,  # вместимость
            "cached_ids": sorted(self.cached_ids),  # отсортированные идентификаторы набора
            "feedback_since_refresh": self._feedback_since_refresh,  # ответов с последней перестройки
            "learner_updates": self.learner.update_count,  # всего обновлений ученика
        }


class SQLiteUtilityIndex:
    """Постоянный инвертированный индекс признаков для быстрой перестройки кэша полезности.

        Архив в DiskBackedContext только дополняется. Этот класс один раз сохраняет на диск
        пары «страница -> признак», после чего перестройка оценивает только страницы, у которых
        есть признаки с текущими ненулевыми весами, вместо повторного разбиения всего архива на
        слова. Дополнительная память O(feature_bins + K)."""

    def __init__(self, memory, working_set: UtilityWorkingSet):  # создать индекс по архиву memory и рабочему набору
        self.memory = memory  # архив с атрибутом db (соединение SQLite)
        self.working_set = working_set  # рабочий набор с учеником
        db = memory.db  # соединение SQLite
        db.execute("""\
CREATE TABLE IF NOT EXISTS utility_page_stats (
            page_id INTEGER PRIMARY KEY, feature_count INTEGER NOT NULL)""")  # таблица числа признаков: page_id — номер страницы, feature_count — число её признаков
        db.execute("""\
CREATE TABLE IF NOT EXISTS utility_page_features (
            page_id INTEGER NOT NULL, feature_idx INTEGER NOT NULL,
            PRIMARY KEY(page_id, feature_idx)) WITHOUT ROWID""")  # таблица пар (номер страницы, ячейка признака); первичный ключ — пара; хранение без rowid
        db.execute("CREATE INDEX IF NOT EXISTS idx_utility_feature_page ON utility_page_features(feature_idx,page_id)")  # индекс (признак, страница) для поиска страниц по признаку
        db.execute("""\
CREATE TEMP TABLE IF NOT EXISTS utility_active_weights (
            feature_idx INTEGER PRIMARY KEY, weight REAL NOT NULL) WITHOUT ROWID""")  # временная таблица активных весов: ячейка -> вес
        db.commit()  # зафиксировать создание таблиц и индекса
        self.indexed_page_id = int(db.execute(  # наибольший уже индексированный номер страницы
            "SELECT COALESCE(MAX(page_id),0) FROM utility_page_stats"  # MAX(page_id), а если строк нет — 0
        ).fetchone()[0])  # первое значение первой строки

    def index_new_pages(self, batch_pages: int = 512) -> int:  # индексировать только новые страницы архива; вернуть их число
        """Разбить на слова и индексировать только добавленные страницы архива; вернуть число страниц."""
        if batch_pages < 1:  # размер пачки меньше 1
            raise ValueError("batch_pages must be >=1")  # ошибка
        db = self.memory.db  # соединение SQLite
        cur = db.execute(  # курсор по страницам новее последней индексированной
            "SELECT rowid,content FROM pages WHERE rowid>? ORDER BY rowid",  # rowid > indexed_page_id по возрастанию
            (self.indexed_page_id,),  # параметр — последний индексированный номер
        )
        indexed = 0  # число индексированных страниц
        while True:  # читать пачками, пока есть строки
            rows = cur.fetchmany(batch_pages)  # очередная пачка строк
            if not rows:  # строк больше нет
                break  # конец чтения
            stat_rows: list[tuple[int, int]] = []  # строки для таблицы числа признаков
            feature_rows: list[tuple[int, int]] = []  # строки для таблицы пар (страница, признак)
            for page_id, content in rows:  # по строкам пачки
                pid = int(page_id)  # номер страницы как целое
                indices = self.working_set.learner.feature_indices(str(content))  # признаки текста страницы
                stat_rows.append((pid, int(indices.size)))  # (номер страницы, число признаков)
                feature_rows.extend((pid, int(idx)) for idx in indices.tolist())  # по одной паре (номер страницы, ячейка) на каждый признак
                self.indexed_page_id = pid  # последняя индексированная страница обновляется
            db.executemany(  # записать числа признаков
                "INSERT OR REPLACE INTO utility_page_stats(page_id,feature_count) VALUES (?,?)",  # вставить или заменить строку
                stat_rows,  # данные
            )
            if feature_rows:  # признаки были
                db.executemany(  # записать пары (страница, признак)
                    "INSERT OR REPLACE INTO utility_page_features(page_id,feature_idx) VALUES (?,?)",  # вставить или заменить
                    feature_rows,  # данные
                )
            db.commit()  # зафиксировать пачку
            indexed += len(rows)  # индексировано: + размер пачки
        return indexed  # число индексированных страниц

    def record_page_feedback(self, page_id: int, reward: float = 1.0) -> UtilityFeedback:  # ответ поиска по странице
        """Принять ответ полезности по одной странице, при необходимости сначала её индексируя."""
        self.index_new_pages()  # добавить в индекс новые страницы
        pid = int(page_id)  # номер страницы как целое
        row = self.memory.db.execute(  # текст страницы
            "SELECT content FROM pages WHERE rowid=?", (pid,)  # по номеру
        ).fetchone()  # первая строка или None
        if row is None:  # страницы нет
            raise KeyError(f"archive page {pid} does not exist")  # ошибка: страница не существует
        self.memory.db.execute(  # гарантировать строку в таблице числа признаков
            "INSERT OR IGNORE INTO utility_page_stats(page_id,feature_count) VALUES (?,0)", (pid,)  # вставить с нулём, если строки нет
        )
        indices = self.working_set.learner.feature_indices(str(row[0]))  # признаки текста страницы
        self.memory.db.execute("DELETE FROM utility_page_features WHERE page_id=?", (pid,))  # удалить прежние пары этой страницы
        if indices.size:  # признаки есть
            self.memory.db.executemany(  # записать новые пары (страница, признак)
                "INSERT INTO utility_page_features(page_id,feature_idx) VALUES (?,?)",  # вставка
                [(pid, int(idx)) for idx in indices.tolist()],  # по паре на каждый признак
            )
        self.memory.db.execute(  # обновить число признаков страницы
            "UPDATE utility_page_stats SET feature_count=? WHERE page_id=?", (int(indices.size), pid)  # новое число и номер страницы
        )
        self.memory.db.commit()  # зафиксировать изменения
        return self.working_set.record_feedback(  # обновить ученика по тексту страницы
            UtilityRecord(str(pid), str(row[0])), reward=reward  # запись (номер страницы, текст) и награда
        )

    def refresh(self) -> UtilityRefreshStats:  # перестроить набор через инвертированный индекс
        """Оценить подходящие страницы через инвертированный индекс и оставить только k наибольших."""
        import heapq  # куча на списке

        started = time.perf_counter()  # начало замера времени
        self.index_new_pages()  # индексировать новые страницы
        db = self.memory.db  # соединение SQLite
        active = np.flatnonzero(np.abs(self.working_set.learner.weights) > 1e-9)  # ячейки с весом по модулю больше 1e-9
        db.execute("DELETE FROM utility_active_weights")  # очистить таблицу активных весов
        if active.size:  # активные ячейки есть
            db.executemany(  # записать активные веса
                "INSERT INTO utility_active_weights(feature_idx,weight) VALUES (?,?)",  # вставка (ячейка, вес)
                [(int(i), float(self.working_set.learner.weights[i])) for i in active.tolist()],  # по строке на каждую активную ячейку
            )
            # для каждой страницы с активным признаком: номер, сумма весов её активных признаков,
            # число всех её признаков (соединение пар, активных весов и счётчиков по ячейке и странице,
            # группировка по странице)
            cur = db.execute(  # выполнить запрос и получить курсор
                """\
SELECT f.page_id,SUM(w.weight),s.feature_count
                   FROM utility_page_features AS f
                   JOIN utility_active_weights AS w ON w.feature_idx=f.feature_idx
                   JOIN utility_page_stats AS s ON s.page_id=f.page_id
                   GROUP BY f.page_id""")  # запрос выполнен; cur — курсор по строкам (page_id, weight_sum, feature_count)
        else:  # активных ячеек нет
            cur = iter(())  # пустой перебор

        heap: list[tuple[float, int]] = []  # куча из не более capacity кортежей (score, номер страницы)
        candidate_pages = 0  # число страниц-кандидатов
        for page_id, weight_sum, feature_count in cur:  # по страницам-кандидатам
            candidate_pages += 1  # кандидатов: +1
            if int(feature_count) <= 0:  # у страницы нет признаков
                continue  # не оценивается
            score = float(weight_sum) / np.sqrt(float(feature_count))  # score = (Σ весов активных признаков) / sqrt(число признаков)
            if score <= 0.0:  # оценка не положительна
                continue  # страница не нужна
            item = (score, int(page_id))  # кандидат — кортеж (оценка, номер страницы)
            if len(heap) < self.working_set.capacity:  # куча не заполнена
                heapq.heappush(heap, item)  # добавить кандидата
            elif item > heap[0]:  # кандидат больше худшего в куче
                heapq.heapreplace(heap, item)  # заменить худшего кандидатом

        selected = {str(pid) for _, pid in heap}  # выбранные страницы — идентификаторы из кучи
        # свободные места заполняются самыми свежими страницами архива (курсор ограничен)
        if len(selected) < self.working_set.capacity:  # выбрано меньше вместимости
            cur_recent = db.execute("SELECT rowid FROM pages ORDER BY rowid DESC")  # номера страниц от свежих к старым
            for (page_id,) in cur_recent:  # по номерам
                selected.add(str(int(page_id)))  # добавить страницу
                if len(selected) >= self.working_set.capacity:  # набор заполнен
                    break  # конец заполнения
        self.working_set.cached_ids = selected  # новый набор
        self.working_set._feedback_since_refresh = 0  # счётчик ответов обнулён
        threshold = min((score for score, _ in heap), default=0.0)  # наименьшая оценка в куче (0.0, если куча пуста)
        stats = UtilityRefreshStats(  # итоги перестройки
            scanned=candidate_pages,  # просмотрено страниц-кандидатов
            retained=len(selected),  # оставлено в наборе
            seconds=time.perf_counter() - started,  # затрачено секунд
            threshold_score=float(threshold),  # пороговая оценка
        )
        self.working_set.last_refresh = stats  # запомнить итоги
        return stats  # вернуть итоги
